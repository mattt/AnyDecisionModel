import Foundation

#if MLX
    import HuggingFace
    import MLX
    import MLXHuggingFace
    import MLXLLM
    import MLXLMCommon
    import Tokenizers

    /// A decision model that runs a causal language model locally with MLX.
    ///
    /// The model reads decisions from next-token probabilities in one forward pass,
    /// with no text generation.
    /// For each allowed answer label, it sums the probability of the label's
    /// single-token variants, then normalizes across the allowed answers.
    /// Logits are read in float32.
    ///
    /// Choice options are labeled A to Z, so a choice can have at most 26 options.
    /// Score levels are labeled 0 to 9, so a score can have at most 10 levels.
    ///
    /// Sessions share loaded weights.
    /// Each session keeps its own cache for the prompt prefix that contains the state,
    /// and every question runs on a copy of that cache.
    ///
    /// Probabilities are raw by default. Set ``calibration`` to apply a fitted calibration.
    ///
    /// ```swift
    /// let model = MLXDecisionModel(modelID: "mlx-community/Qwen3-4B-Instruct-2507-4bit")
    /// ```
    public struct MLXDecisionModel: DecisionModel {
        /// The default model.
        public static let defaultModelID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

        /// The default system prompt for local decision models.
        public static let defaultSystemPrompt = TokenScoring.defaultSystemPrompt

        /// The Hugging Face model identifier.
        public let modelID: String

        /// The Hugging Face Hub client used for downloads, or `nil` for the default client.
        public let hub: HubClient?

        /// A local directory with the model files, or `nil` to download the model.
        public let directory: URL?

        /// The calibration applied to answer probabilities, or `nil` for raw probabilities.
        public var calibration: Calibration?

        /// Whether choice questions are asked once per rotation of the option order
        /// and the results averaged.
        ///
        /// This reduces position bias and costs one forward pass per option.
        /// It is off by default.
        public var rotationDebiasing: Bool

        /// Whether sessions reuse a cache for the shared prompt prefix.
        ///
        /// This is off by default because MLX 0.31.1 has a causal-mask error
        /// for some split-prefill shapes.
        /// Enable it only after checking cached results for the selected model and workload.
        public var prefixCaching: Bool

        /// Whether to append a closed thinking block when the prompt has none.
        ///
        /// Enable this for reasoning models whose chat templates ignore
        /// `enable_thinking: false`.
        /// The tokenizer must have both `<think>` and `</think>` markers.
        /// This option is off by default because models without a thinking mode
        /// can also have these markers.
        public var closedThinkFallback: Bool

        /// The system prompt.
        public var systemPrompt: String

        /// Creates an MLX decision model.
        ///
        /// - Parameters:
        ///   - modelID: The Hugging Face model identifier.
        ///   - hub: A Hugging Face Hub client for downloads. Pass `nil` for the default client.
        ///   - directory: A local directory with the model files.
        ///     If you pass a directory, the model loads from it instead of downloading.
        ///   - calibration: A calibration for answer probabilities. Pass `nil` for raw probabilities.
        ///   - rotationDebiasing: Whether to average choice results over rotations of the option order.
        ///   - prefixCaching: Whether sessions reuse a cache for the shared prompt prefix.
        ///   - closedThinkFallback: Whether to append a closed thinking block
        ///     if the template omits it and the tokenizer has thinking markers.
        ///     Defaults to `false`.
        ///   - systemPrompt: The system prompt.
        public init(
            modelID: String = MLXDecisionModel.defaultModelID,
            hub: HubClient? = nil,
            directory: URL? = nil,
            calibration: Calibration? = nil,
            rotationDebiasing: Bool = false,
            prefixCaching: Bool = false,
            closedThinkFallback: Bool = false,
            systemPrompt: String = MLXDecisionModel.defaultSystemPrompt
        ) {
            self.modelID = modelID
            self.hub = hub
            self.directory = directory
            self.calibration = calibration
            self.rotationDebiasing = rotationDebiasing
            self.prefixCaching = prefixCaching
            self.closedThinkFallback = closedThinkFallback
            self.systemPrompt = systemPrompt
        }

        public var capabilities: DecisionCapabilities {
            DecisionCapabilities(
                maximumChoiceOptions: TokenScoring.choiceLabels.count,
                maximumScoreLevels: TokenScoring.scoreLabels.count
            )
        }

        public func makeExecutor() -> any DecisionModelExecutor {
            MLXDecisionExecutor(model: self)
        }

        /// Whether the weights for this model are loaded.
        public var isLoaded: Bool {
            containerCache.isLoaded(cacheKey)
        }

        /// Removes this model's weights from the shared cache.
        ///
        /// Existing sessions keep their loaded weights until they are released.
        public func removeFromCache() {
            containerCache.remove(cacheKey)
        }

        /// Removes all MLX decision models from the shared cache.
        public static func removeAllFromCache() {
            containerCache.removeAll()
        }

        var cacheKey: String {
            directory?.standardizedFileURL.path ?? modelID
        }

        func loadContainer() async throws -> ModelContainer {
            let modelID = modelID
            let hub = hub
            let directory = directory
            do {
                return try await containerCache.container(for: cacheKey) {
                    // mlx-swift-lm 3 takes explicit downloader and tokenizer loaders.
                    // The macros back them with HubClient and AutoTokenizer.
                    if let directory {
                        return try await loadModelContainer(
                            from: directory,
                            using: #huggingFaceTokenizerLoader()
                        )
                    }
                    if let hub {
                        return try await loadModelContainer(
                            from: #hubDownloader(hub),
                            using: #huggingFaceTokenizerLoader(),
                            id: modelID
                        )
                    }
                    return try await loadModelContainer(
                        from: #hubDownloader(),
                        using: #huggingFaceTokenizerLoader(),
                        id: modelID
                    )
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

    /// Loads each model once and shares the container across sessions.
    private final class ModelContainerCache: Sendable {
        private let entries = Locked<[String: Task<ModelContainer, Error>]>([:])
        private let loaded = Locked<Set<String>>([])

        func container(
            for key: String,
            loader: @escaping @Sendable () async throws -> ModelContainer
        ) async throws -> ModelContainer {
            let task = entries.withLock { entries in
                if let task = entries[key] {
                    return task
                }
                let task = Task { try await loader() }
                entries[key] = task
                return task
            }

            do {
                // Waiting callers do not cancel the shared load.
                let container = try await task.value
                _ = loaded.withLock { $0.insert(key) }
                try Task.checkCancellation()
                return container
            } catch {
                if !(error is CancellationError) || task.isCancelled {
                    entries.withLock { entries in
                        if entries[key] == task { entries[key] = nil }
                    }
                }
                throw error
            }
        }

        func isLoaded(_ key: String) -> Bool {
            loaded.withLock { $0.contains(key) }
        }

        func remove(_ key: String) {
            entries.withLock { $0[key] = nil }
            _ = loaded.withLock { $0.remove(key) }
        }

        func removeAll() {
            entries.withLock { $0.removeAll() }
            loaded.withLock { $0.removeAll() }
        }
    }

    private let containerCache = ModelContainerCache()

    // MARK: - Executor

    /// The prompt-prefix cache for one session.
    ///
    /// The cache is read and written only inside `ModelContainer.perform`,
    /// which serializes access across all sessions that share the container.
    private final class PrefixCache: @unchecked Sendable {
        var tokens: [Int] = []
        var layers: [any KVCache]?
        var state: DecisionState?
    }

    private final class MLXDecisionExecutor: DecisionModelExecutor {
        let model: MLXDecisionModel
        private let prefix = PrefixCache()

        /// The number of tokens evaluated per forward pass when filling a cache.
        private let prefillStepSize = 512

        init(model: MLXDecisionModel) {
            self.model = model
        }

        func prewarm(for state: DecisionState) async throws {
            let container = try await model.loadContainer()
            guard model.prefixCaching else { return }
            try await container.perform { context in
                try self.preparePrefix(for: state, context: context)
            }
        }

        func decide(_ questions: [Question], about state: DecisionState) async throws
            -> DecisionSession.Response
        {
            let container = try await model.loadContainer()
            return try await container.perform { context in
                try self.decide(questions, about: state, context: context)
            }
        }

        private struct Plan {
            let question: Question
            let tokenMap: [[Int]]
            let rotations: [[Int]]
        }

        private func decide(
            _ questions: [Question],
            about state: DecisionState,
            context: ModelContext
        ) throws -> DecisionSession.Response {
            let tokenizer = context.tokenizer
            let encode = { (text: String) in tokenizer.encode(text: text, addSpecialTokens: false) }

            // Check every token mapping before running inference.
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

            if model.prefixCaching {
                try preparePrefix(for: state, context: context)
            }

            var answers: [Answer] = []
            var diagnostics: [DecisionSession.Diagnostics] = []
            var inputTokenCount = 0

            for plan in plans {
                let isBinary = if case .binary = plan.question { true } else { false }
                var totals = [Double](repeating: 0, count: plan.tokenMap.count)
                var mass = 0.0
                var cachedTokenCount = 0
                var evaluatedTokenCount = 0

                let start = ContinuousClock.now
                for order in plan.rotations {
                    try Task.checkCancellation()
                    let user = TokenScoring.userMessage(
                        state: state,
                        question: plan.question,
                        optionOrder: order
                    )
                    let tokens = try promptTokens(user: user, tokenizer: tokenizer)
                    let result = try logProbabilities(of: tokens, tokenMap: plan.tokenMap, context: context)
                    let (logMasses, allowedMass) = TokenScoring.logMasses(result.tokenLogProbabilities)
                    let probabilities = (model.calibration ?? .identity).probabilities(
                        logMasses: logMasses,
                        binary: isBinary
                    )
                    // Position i in this rotation holds answer order[i].
                    for (position, answerIndex) in order.enumerated() {
                        totals[answerIndex] += probabilities[position] / Double(plan.rotations.count)
                    }
                    mass += allowedMass / Double(plan.rotations.count)
                    cachedTokenCount += result.cachedTokenCount
                    evaluatedTokenCount += tokens.count - result.cachedTokenCount
                    inputTokenCount += tokens.count
                }
                let duration = start.duration(to: .now)

                answers.append(TokenScoring.answer(for: plan.question, probabilities: totals))
                diagnostics.append(
                    DecisionSession.Diagnostics(
                        allowedAnswerMass: mass,
                        cachedTokenCount: cachedTokenCount,
                        evaluatedTokenCount: evaluatedTokenCount,
                        duration: duration
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

        private func promptTokens(user: String, tokenizer: any MLXLMCommon.Tokenizer) throws -> [Int] {
            let messages: [[String: any Sendable]] = [
                ["role": "system", "content": model.systemPrompt],
                ["role": "user", "content": user],
            ]
            // Disable thinking so that the answer token follows the generation prompt.
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: ["enable_thinking": false]
            )
            guard model.closedThinkFallback else { return tokens }
            return TokenScoring.appendingClosedThinkBlock(
                to: tokens,
                closedThinkTokens: closedThinkTokens(tokenizer: tokenizer),
                decode: { tokenizer.decode(tokenIds: $0, skipSpecialTokens: false) }
            )
        }

        private func closedThinkTokens(tokenizer: any MLXLMCommon.Tokenizer) -> [Int] {
            guard let opening = tokenizer.convertTokenToId("<think>"),
                let closing = tokenizer.convertTokenToId("</think>")
            else { return [] }

            let blankLines = tokenizer.encode(text: "\n\n", addSpecialTokens: false)
            return [opening] + blankLines + [closing] + blankLines
        }

        /// Fills the session's prefix cache for a state, if it is not already filled.
        ///
        /// The prefix is the longest run of tokens that two different questions share,
        /// which covers the system prompt and the state.
        private func preparePrefix(for state: DecisionState, context: ModelContext) throws {
            if prefix.state == state, prefix.layers != nil { return }

            let tokenizer = context.tokenizer
            let probeA = TokenScoring.userMessage(state: state, question: .binary(instructions: "A"))
            let probeB = TokenScoring.userMessage(state: state, question: .binary(instructions: "Z"))
            let tokensA = try promptTokens(user: probeA, tokenizer: tokenizer)
            let tokensB = try promptTokens(user: probeB, tokenizer: tokenizer)
            let shared = zip(tokensA, tokensB).prefix { $0 == $1 }.count
            let prefixTokens = Array(tokensA.prefix(shared))

            guard !prefixTokens.isEmpty else {
                prefix.tokens = []
                prefix.layers = nil
                prefix.state = state
                return
            }

            let cache = context.model.newCache(parameters: nil)
            try prefill(prefixTokens, cache: cache, model: context.model)
            prefix.tokens = prefixTokens
            prefix.layers = cache
            prefix.state = state
        }

        private func prefill(_ tokens: [Int], cache: [any KVCache], model: any LanguageModel) throws {
            var start = 0
            while start < tokens.count {
                try Task.checkCancellation()
                let end = min(start + prefillStepSize, tokens.count)
                let input = MLXArray(tokens[start ..< end].map { Int32($0) })[.newAxis]
                _ = model(input, cache: cache)
                eval(cache)
                start = end
            }
        }

        private struct EvaluationResult {
            let tokenLogProbabilities: [[Double]]
            let cachedTokenCount: Int
        }

        /// Runs the prompt and returns the next-token log-probabilities of each label's tokens,
        /// in label position order.
        private func logProbabilities(
            of tokens: [Int],
            tokenMap: [[Int]],
            context: ModelContext
        ) throws -> EvaluationResult {
            var cache: [any KVCache]
            var start = 0

            if model.prefixCaching, let layers = prefix.layers,
                tokens.count > prefix.tokens.count,
                tokens.starts(with: prefix.tokens)
            {
                // Copy the prefix cache so that this question cannot change it.
                cache = layers.map { $0.copy() }
                start = prefix.tokens.count
            } else {
                cache = context.model.newCache(parameters: nil)
            }

            let cachedTokenCount = start
            let body = Array(tokens[start...])
            if body.count > prefillStepSize {
                try prefill(Array(body.dropLast()), cache: cache, model: context.model)
                start = tokens.count - 1
            }

            let input = MLXArray(tokens[start...].map { Int32($0) })[.newAxis]
            let logits = context.model(input, cache: cache)[0, -1].asType(.float32)
            let logProbabilities = logits - logSumExp(logits)

            let ids = tokenMap.flatMap { $0 }
            let gathered = logProbabilities[MLXArray(ids.map { Int32($0) })].asArray(Float.self)

            var result: [[Double]] = []
            var offset = 0
            for tokens in tokenMap {
                result.append(gathered[offset ..< offset + tokens.count].map(Double.init))
                offset += tokens.count
            }
            return EvaluationResult(tokenLogProbabilities: result, cachedTokenCount: cachedTokenCount)
        }
    }
#endif
