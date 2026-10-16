// Adapted in part from AnyLanguageModel (https://github.com/huggingface/AnyLanguageModel).
// Copyright AnyLanguageModel contributors. Licensed under the Apache License, Version 2.0.

import Foundation

#if Llama
    import LlamaSwift

    /// Global storage for the current log level threshold.
    /// This is needed because the C callback can't capture Swift context.
    /// Access is synchronized by llama.cpp's internal logging mechanism.
    nonisolated(unsafe) private var currentLogLevel: LlamaDecisionModel.LogLevel = .warn

    /// Custom log callback that filters messages based on the current log level.
    private func llamaLogCallback(
        level: ggml_log_level,
        text: UnsafePointer<CChar>?,
        userData: UnsafeMutableRawPointer?
    ) {
        guard level.rawValue >= currentLogLevel.ggmlLevel.rawValue else { return }
        if let text = text {
            fputs(String(cString: text), stderr)
        }
    }

    /// A decision model that runs a GGUF model locally with llama.cpp.
    ///
    /// The model reads decisions from next-token probabilities in one forward pass,
    /// with no text generation, like ``MLXDecisionModel``.
    /// For each allowed answer label, it sums the probability of the label's
    /// single-token variants, then normalizes across the allowed answers.
    /// Logits are read in float32.
    ///
    /// Choice options are labeled A to Z, so a choice can have at most 26 options.
    /// Score levels are labeled 0 to 9, so a score can have at most 10 levels.
    ///
    /// The prompt uses the chat template embedded in the GGUF file,
    /// applied by `llama_chat_apply_template`,
    /// which supports a predefined list of templates rather than full Jinja.
    ///
    /// Sessions share loaded weights.
    /// Each session keeps its own llama.cpp context
    /// and clears its memory between questions,
    /// so the shared state prefix is evaluated again for every question.
    ///
    /// Probabilities are raw by default. Set ``calibration`` to apply a fitted calibration.
    ///
    /// ```swift
    /// let model = LlamaDecisionModel(modelPath: "/path/to/model.gguf")
    /// ```
    public struct LlamaDecisionModel: DecisionModel {
        /// The verbosity level for llama.cpp logging.
        public enum LogLevel: Int, Hashable, Comparable, Sendable, CaseIterable {
            /// No logging output.
            case none = 0
            /// Debug messages and above (most verbose).
            case debug = 1
            /// Info messages and above.
            case info = 2
            /// Warning messages and above (default).
            case warn = 3
            /// Only error messages.
            case error = 4

            /// Maps to the corresponding ggml log level.
            var ggmlLevel: ggml_log_level {
                switch self {
                case .none: return GGML_LOG_LEVEL_NONE
                case .debug: return GGML_LOG_LEVEL_DEBUG
                case .info: return GGML_LOG_LEVEL_INFO
                case .warn: return GGML_LOG_LEVEL_WARN
                case .error: return GGML_LOG_LEVEL_ERROR
                }
            }

            public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
                lhs.rawValue < rhs.rawValue
            }
        }

        /// The minimum log level for llama.cpp output.
        ///
        /// This is a global setting that affects all `LlamaDecisionModel` instances
        /// since llama.cpp uses a single global log callback.
        public nonisolated(unsafe) static var logLevel: LogLevel = .warn {
            didSet {
                currentLogLevel = logLevel
                llama_log_set(llamaLogCallback, nil)
            }
        }

        /// The default GPU layer count for the current platform.
        ///
        /// All layers are offloaded by default.
        /// The simulator defaults to CPU-only execution,
        /// which remains the reliable configuration there.
        public static var defaultGPULayerCount: Int32 {
            #if targetEnvironment(simulator)
                return 0
            #else
                return -1
            #endif
        }

        /// The default system prompt for local decision models.
        public static let defaultSystemPrompt = TokenScoring.defaultSystemPrompt

        /// The path to the GGUF model file.
        public let modelPath: String

        /// The number of model layers offloaded to the GPU.
        ///
        /// The default is ``defaultGPULayerCount``.
        /// Pass 0 for CPU-only execution.
        public let gpuLayers: Int32

        /// The calibration applied to answer probabilities, or `nil` for raw probabilities.
        public var calibration: Calibration?

        /// Whether choice questions are asked once per rotation of the option order
        /// and the results averaged.
        ///
        /// This reduces position bias and requires one prompt evaluation per option.
        /// It is off by default.
        public var rotationDebiasing: Bool

        /// Whether to append a closed thinking block when the prompt has none.
        ///
        /// Enable this for reasoning models whose chat templates
        /// open a thinking block in the generation prompt,
        /// so that the answer token follows the prompt.
        /// The vocabulary must have both `<think>` and `</think>` markers.
        /// This option is off by default because models without a thinking mode
        /// can also have these markers.
        public var closedThinkFallback: Bool

        /// The system prompt.
        public var systemPrompt: String

        /// The context size to allocate for each session, in tokens.
        public var contextSize: UInt32

        /// The largest number of tokens evaluated in one batch.
        public var batchSize: UInt32

        /// The number of threads to use for computation.
        public var threads: Int32

        /// Creates a llama.cpp decision model.
        ///
        /// - Parameters:
        ///   - modelPath: The path to the GGUF model file.
        ///   - gpuLayers: The number of model layers offloaded to the GPU.
        ///   - calibration: A calibration for answer probabilities. Pass `nil` for raw probabilities.
        ///   - rotationDebiasing: Whether to average choice results over rotations of the option order.
        ///   - closedThinkFallback: Whether to append a closed thinking block
        ///     if the template omits it and the vocabulary has thinking markers.
        ///     Defaults to `false`.
        ///   - systemPrompt: The system prompt.
        ///   - contextSize: The context size to allocate for each session, in tokens.
        ///   - batchSize: The largest number of tokens evaluated in one batch.
        ///   - threads: The number of threads to use for computation.
        public init(
            modelPath: String,
            gpuLayers: Int32 = LlamaDecisionModel.defaultGPULayerCount,
            calibration: Calibration? = nil,
            rotationDebiasing: Bool = false,
            closedThinkFallback: Bool = false,
            systemPrompt: String = LlamaDecisionModel.defaultSystemPrompt,
            contextSize: UInt32 = 4096,
            batchSize: UInt32 = 512,
            threads: Int32 = Int32(ProcessInfo.processInfo.processorCount)
        ) {
            self.modelPath = modelPath
            self.gpuLayers = gpuLayers
            self.calibration = calibration
            self.rotationDebiasing = rotationDebiasing
            self.closedThinkFallback = closedThinkFallback
            self.systemPrompt = systemPrompt
            self.contextSize = contextSize
            self.batchSize = batchSize
            self.threads = threads
        }

        /// The model identifier reported in responses.
        ///
        /// This is the model file's name without its extension.
        public var modelID: String {
            URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent
        }

        public var capabilities: DecisionCapabilities {
            DecisionCapabilities(
                maximumChoiceOptions: TokenScoring.choiceLabels.count,
                maximumScoreLevels: TokenScoring.scoreLabels.count
            )
        }

        public func makeExecutor() -> any DecisionModelExecutor {
            LlamaDecisionExecutor(model: self)
        }

        /// Whether the weights for this model are loaded.
        public var isLoaded: Bool {
            modelCache.isLoaded(cacheKey)
        }

        /// Removes this model's weights from the shared cache.
        ///
        /// Existing sessions keep their loaded weights until they are released.
        public func removeFromCache() {
            modelCache.remove(cacheKey)
        }

        /// Removes all llama.cpp decision models from the shared cache.
        public static func removeAllFromCache() {
            modelCache.removeAll()
        }

        var cacheKey: String {
            "\(URL(fileURLWithPath: modelPath).standardizedFileURL.path)|gpuLayers=\(gpuLayers)"
        }

        func loadModel() async throws -> LlamaModel {
            let modelPath = modelPath
            let gpuLayers = gpuLayers
            do {
                return try await modelCache.value(for: cacheKey) {
                    guard FileManager.default.fileExists(atPath: modelPath) else {
                        throw DecisionError.modelUnavailable("No model file at \"\(modelPath)\".")
                    }
                    return try LlamaModel(modelPath: modelPath, gpuLayers: gpuLayers)
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as DecisionError {
                throw error
            } catch {
                throw DecisionError.modelUnavailable(String(describing: error))
            }
        }
    }

    // MARK: - Shared weights

    private let modelCache = ModelCache<LlamaModel>()

    /// Serializes one-time llama.cpp backend initialization.
    private let backendLock = NSLock()
    nonisolated(unsafe) private var backendInitialized = false

    private func initializeBackendIfNeeded() {
        backendLock.lock()
        defer { backendLock.unlock() }
        guard !backendInitialized else { return }
        llama_backend_init()
        llama_log_set(llamaLogCallback, nil)
        // The shared cache would otherwise keep weights loaded until the process ends.
        // ggml's Metal device asserts at exit that no residency sets remain,
        // so free the cached models and the backend from an exit handler,
        // which runs before ggml's static destructors.
        atexit {
            modelCache.removeAll()
            llama_backend_free()
        }
        backendInitialized = true
    }

    /// A loaded llama.cpp model and its vocabulary.
    ///
    /// Creating contexts from one model at the same time is not safe,
    /// so context creation takes `contextLock`.
    /// Reading the vocabulary needs no lock.
    final class LlamaModel: @unchecked Sendable {
        let pointer: OpaquePointer
        let vocab: OpaquePointer
        let contextLock = NSLock()

        init(modelPath: String, gpuLayers: Int32) throws {
            initializeBackendIfNeeded()

            var params = llama_model_default_params()
            params.n_gpu_layers = gpuLayers
            params.load_mode = LLAMA_LOAD_MODE_MMAP

            guard let model = llama_model_load_from_file(modelPath, params) else {
                throw DecisionError.modelUnavailable("Failed to load the model at \"\(modelPath)\".")
            }
            pointer = model
            vocab = llama_model_get_vocab(model)
        }

        deinit {
            llama_model_free(pointer)
        }
    }

    // MARK: - Token log-probabilities

    extension LlamaDecisionModel {
        /// Returns the log-probability of each mapped token under the full distribution.
        ///
        /// - Parameters:
        ///   - logits: The logits for every token in the vocabulary.
        ///   - tokenMap: For each answer, the token IDs of its label variants.
        static func tokenLogProbabilities(logits: [Float], tokenMap: [[Int]]) -> [[Double]] {
            tokenLogProbabilities(logits: logits.withUnsafeBufferPointer { $0 }, tokenMap: tokenMap)
        }

        static func tokenLogProbabilities(
            logits: UnsafeBufferPointer<Float>,
            tokenMap: [[Int]]
        ) -> [[Double]] {
            let maximum = logits.max() ?? 0
            var sum = 0.0
            for logit in logits {
                sum += exp(Double(logit - maximum))
            }
            let logDenominator = Double(maximum) + log(sum)
            return tokenMap.map { ids in
                ids.map { id in
                    guard id >= 0, id < logits.count else { return -Double.infinity }
                    return Double(logits[id]) - logDenominator
                }
            }
        }
    }

    // MARK: - Executor

    private final class LlamaDecisionExecutor: DecisionModelExecutor, @unchecked Sendable {
        let model: LlamaDecisionModel
        private let mutex = AsyncMutex()

        /// The loaded weights, retained for the executor's lifetime.
        ///
        /// The context is created from these weights,
        /// so they must stay loaded even if the shared cache drops its entry.
        /// `deinit` frees the context before stored properties release the weights.
        private let retainedModel = Locked<LlamaModel?>(nil)

        // The context and batch are used only while `mutex` is held.
        private var context: OpaquePointer?
        private var batch: llama_batch?

        init(model: LlamaDecisionModel) {
            self.model = model
        }

        deinit {
            if let batch {
                llama_batch_free(batch)
            }
            if let context {
                llama_free(context)
            }
        }

        func prewarm(for state: DecisionState) async throws {
            let loaded = try await retainedLoadedModel()
            try await mutex.withLock {
                _ = try self.ensureContext(for: loaded)
            }
        }

        func decide(_ questions: [Question], about state: DecisionState) async throws
            -> DecisionSession.Response
        {
            let loaded = try await retainedLoadedModel()
            return try await mutex.withLock {
                try self.decide(questions, about: state, loaded: loaded)
            }
        }

        /// Returns the weights this executor retains, loading them on first use.
        private func retainedLoadedModel() async throws -> LlamaModel {
            if let loaded = retainedModel.withLock({ $0 }) {
                return loaded
            }
            let loaded = try await model.loadModel()
            return retainedModel.withLock { state in
                if let existing = state {
                    return existing
                }
                state = loaded
                return loaded
            }
        }

        private func decide(
            _ questions: [Question],
            about state: DecisionState,
            loaded: LlamaModel
        ) throws -> DecisionSession.Response {
            let context = try ensureContext(for: loaded)
            let vocab = loaded.vocab
            let vocabSize = Int(llama_vocab_n_tokens(vocab))

            let encode = { (text: String) in
                Self.tokenize(vocab: vocab, text: text, addSpecial: false, parseSpecial: false)
                    .map { Int($0) }
            }
            let plans = try questions.enumerated().map { index, question -> Plan in
                let labels = TokenScoring.labels(for: question)
                let tokenMap = try TokenScoring.tokenMap(for: labels, index: index, encode: encode)
                var rotations = [Array(labels.indices)]
                if model.rotationDebiasing, case .choice = question, labels.count > 1 {
                    rotations = labels.indices.map { r in
                        Array(labels.indices[r...]) + Array(labels.indices[..<r])
                    }
                }
                return Plan(question: question, tokenMap: tokenMap, rotations: rotations)
            }

            var answers: [Answer] = []
            var diagnostics: [DecisionSession.Diagnostics] = []
            var inputTokenCount = 0

            for (index, plan) in plans.enumerated() {
                let questionStart = ContinuousClock.now
                let isBinary = if case .binary = plan.question { true } else { false }
                var totals = [Double](repeating: 0, count: plan.tokenMap.count)
                var mass = 0.0
                var evaluatedTokenCount = 0

                for order in plan.rotations {
                    try Task.checkCancellation()
                    let user = TokenScoring.userMessage(
                        state: state,
                        question: plan.question,
                        optionOrder: order
                    )
                    let tokens = try promptTokens(user: user, loaded: loaded)
                    guard tokens.count < Int(model.contextSize) else {
                        throw DecisionError.unsupportedQuestion(
                            index: index,
                            reason:
                                "The prompt has \(tokens.count) tokens, "
                                + "which does not fit the context size of \(model.contextSize)."
                        )
                    }
                    let logProbabilities = try evaluate(
                        tokens: tokens,
                        context: context,
                        vocabSize: vocabSize,
                        tokenMap: plan.tokenMap
                    )
                    let (logMasses, allowedMass) = TokenScoring.logMasses(logProbabilities)
                    let probabilities = (model.calibration ?? .identity).probabilities(
                        logMasses: logMasses,
                        binary: isBinary
                    )
                    // Position i in this rotation holds answer order[i].
                    for (position, answerIndex) in order.enumerated() {
                        totals[answerIndex] += probabilities[position] / Double(plan.rotations.count)
                    }
                    mass += allowedMass / Double(plan.rotations.count)
                    evaluatedTokenCount += tokens.count
                    inputTokenCount += tokens.count
                }

                answers.append(TokenScoring.answer(for: plan.question, probabilities: totals))
                diagnostics.append(
                    DecisionSession.Diagnostics(
                        allowedAnswerMass: mass,
                        cachedTokenCount: 0,
                        evaluatedTokenCount: evaluatedTokenCount,
                        duration: questionStart.duration(to: .now)
                    )
                )
            }

            return DecisionSession.Response(
                modelID: model.modelID,
                answers: answers,
                usage: DecisionSession.Usage(inputTokenCount: inputTokenCount, outputTokenCount: 0),
                diagnostics: diagnostics
            )
        }

        private struct Plan: Sendable {
            let question: Question
            let tokenMap: [[Int]]
            let rotations: [[Int]]
        }

        /// Returns the session's context, creating it on first use.
        private func ensureContext(for loaded: LlamaModel) throws -> OpaquePointer {
            if let context { return context }

            var params = llama_context_default_params()
            params.n_ctx = model.contextSize
            params.n_batch = model.batchSize
            params.n_threads = model.threads
            params.n_threads_batch = model.threads

            let created: OpaquePointer? = loaded.contextLock.withLock {
                llama_init_from_model(loaded.pointer, params)
            }
            guard let created else {
                throw DecisionError.modelUnavailable("Failed to create a llama.cpp context.")
            }
            context = created
            return created
        }

        /// Returns the session's batch, allocating it on first use.
        private func ensureBatch() -> llama_batch {
            if let batch { return batch }
            let allocated = llama_batch_init(Int32(model.batchSize), 0, 1)
            batch = allocated
            return allocated
        }

        /// Renders the system and user messages with the model's chat template
        /// and tokenizes the result.
        private func promptTokens(user: String, loaded: LlamaModel) throws -> [llama_token] {
            let rendered = try Self.renderPrompt(
                system: model.systemPrompt,
                user: user,
                model: loaded.pointer
            )
            let tokens = Self.tokenize(
                vocab: loaded.vocab,
                text: rendered,
                addSpecial: true,
                parseSpecial: true
            )
            guard !tokens.isEmpty else {
                throw DecisionError.modelUnavailable("Failed to tokenize the prompt.")
            }
            guard model.closedThinkFallback else { return tokens }
            let vocab = loaded.vocab
            return TokenScoring.appendingClosedThinkBlock(
                to: tokens.map { Int($0) },
                closedThinkTokens: Self.closedThinkTokens(vocab: vocab),
                decode: { ids in
                    ids.compactMap { Self.tokenText(vocab: vocab, token: llama_token($0)) }.joined()
                }
            ).map { llama_token($0) }
        }

        private static func closedThinkTokens(vocab: OpaquePointer) -> [Int] {
            let opening = tokenize(vocab: vocab, text: "<think>", addSpecial: false, parseSpecial: true)
            let closing = tokenize(vocab: vocab, text: "</think>", addSpecial: false, parseSpecial: true)
            guard opening.count == 1, closing.count == 1 else { return [] }

            let blankLines = tokenize(vocab: vocab, text: "\n\n", addSpecial: false, parseSpecial: false)
            return ([opening[0]] + blankLines + [closing[0]] + blankLines).map { Int($0) }
        }

        /// Returns the text of a token, keeping special tokens.
        private static func tokenText(vocab: OpaquePointer, token: llama_token) -> String? {
            var capacity: Int32 = 64
            var buffer = UnsafeMutablePointer<CChar>.allocate(capacity: Int(capacity))
            defer { buffer.deallocate() }

            var written = llama_token_to_piece(vocab, token, buffer, capacity, 0, true)
            if written < 0 {
                capacity = -written
                buffer.deallocate()
                buffer = UnsafeMutablePointer<CChar>.allocate(capacity: Int(capacity))
                written = llama_token_to_piece(vocab, token, buffer, capacity, 0, true)
            }

            let count = Int(max(0, written))
            guard count > 0 else { return nil }
            return String(decoding: UnsafeRawBufferPointer(start: buffer, count: count), as: UTF8.self)
        }

        /// Applies the GGUF chat template to a system and a user message,
        /// ending with the start of an assistant message.
        private static func renderPrompt(
            system: String,
            user: String,
            model: OpaquePointer
        ) throws -> String {
            let messages = [(role: "system", content: system), (role: "user", content: user)]

            // Keep C strings alive while using them
            let cRoles = messages.map { strdup($0.role) }
            let cContents = messages.map { strdup($0.content) }
            defer {
                cRoles.forEach { free($0) }
                cContents.forEach { free($0) }
            }

            var cMessages = [llama_chat_message]()
            for i in 0 ..< messages.count {
                cMessages.append(llama_chat_message(role: cRoles[i], content: cContents[i]))
            }

            let template = llama_model_chat_template(model, nil)

            let requiredSize = llama_chat_apply_template(
                template,
                cMessages,
                cMessages.count,
                true,
                nil,
                0
            )
            guard requiredSize > 0 else {
                throw DecisionError.modelUnavailable(
                    "The model's chat template is missing or unsupported by llama.cpp."
                )
            }

            var buffer = [CChar](repeating: 0, count: Int(requiredSize) + 1)
            let result = llama_chat_apply_template(
                template,
                cMessages,
                cMessages.count,
                true,
                &buffer,
                Int32(buffer.count)
            )
            guard result > 0 else {
                throw DecisionError.modelUnavailable(
                    "The model's chat template is missing or unsupported by llama.cpp."
                )
            }

            return buffer.withUnsafeBytes { rawBuffer in
                String(decoding: rawBuffer.prefix(Int(result)), as: UTF8.self)
            }
        }

        /// Evaluates one prompt and returns the log-probability of each mapped token
        /// at the last position.
        private func evaluate(
            tokens: [llama_token],
            context: OpaquePointer,
            vocabSize: Int,
            tokenMap: [[Int]]
        ) throws -> [[Double]] {
            llama_memory_clear(llama_get_memory(context), true)
            var batch = ensureBatch()
            defer { self.batch = batch }

            let capacity = Int(model.batchSize)
            var start = 0
            while start < tokens.count {
                let count = min(capacity, tokens.count - start)
                batch.n_tokens = Int32(count)
                for i in 0 ..< count {
                    batch.token[i] = tokens[start + i]
                    batch.pos[i] = Int32(start + i)
                    batch.n_seq_id[i] = 1
                    if let seqIDs = batch.seq_id, let seqID = seqIDs[i] {
                        seqID[0] = 0
                    }
                    batch.logits[i] = 0
                }
                if start + count == tokens.count {
                    batch.logits[count - 1] = 1
                }

                guard llama_decode(context, batch) == 0 else {
                    throw DecisionError.modelUnavailable("llama.cpp failed to evaluate the prompt.")
                }
                start += count
            }

            guard let logits = llama_get_logits_ith(context, -1) else {
                throw DecisionError.modelUnavailable("llama.cpp returned no logits for the prompt.")
            }
            return LlamaDecisionModel.tokenLogProbabilities(
                logits: UnsafeBufferPointer(start: logits, count: vocabSize),
                tokenMap: tokenMap
            )
        }

        /// Tokenizes text with the model's vocabulary.
        private static func tokenize(
            vocab: OpaquePointer,
            text: String,
            addSpecial: Bool,
            parseSpecial: Bool
        ) -> [llama_token] {
            let utf8Count = text.utf8.count
            var capacity = Int32(max(utf8Count * 2, 8))
            var tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(capacity))
            defer { tokens.deallocate() }

            var tokenCount = llama_tokenize(
                vocab,
                text,
                Int32(utf8Count),
                tokens,
                capacity,
                addSpecial,
                parseSpecial
            )
            if tokenCount < 0 {
                // The buffer was too small. The result is the needed capacity, negated.
                tokens.deallocate()
                capacity = -tokenCount
                tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(capacity))
                tokenCount = llama_tokenize(
                    vocab,
                    text,
                    Int32(utf8Count),
                    tokens,
                    capacity,
                    addSpecial,
                    parseSpecial
                )
            }
            guard tokenCount > 0 else { return [] }
            return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount)))
        }
    }
#endif
