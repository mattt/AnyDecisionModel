import Foundation

#if CoreAI
    import CoreAIKit
    import Tokenizers

    /// A decision model that runs a Core AI language bundle locally
    /// through [coreai-kit](https://github.com/john-rocky/coreai-kit).
    ///
    /// The model reads decisions from next-token probabilities in one forward pass,
    /// with no text generation.
    /// It uses the same prompt, answer labels, and readout as `MLXDecisionModel`.
    /// For each allowed answer label, it sums the probability of the label's
    /// single-token variants, then normalizes across the allowed answers.
    /// The Core AI engine computes logits in half precision on Apple silicon;
    /// the model reads them in float32 and normalizes them in double precision.
    ///
    /// Choice options are labeled A to Z, so a choice can have at most 26 options.
    /// Score levels are labeled 0 to 9, so a score can have at most 10 levels.
    ///
    /// Sessions share one loaded engine for each bundle, and calls on that engine run one at a time.
    /// The engine keeps the tokens of the last prompt in its cache.
    /// With prefix caching, each prompt reuses the cached tokens that it shares with the previous prompt,
    /// so questions about the same state evaluate only their question.
    ///
    /// Probabilities are raw by default. Set ``calibration`` to apply a fitted calibration.
    ///
    /// ```swift
    /// if #available(macOS 27, iOS 27, *) {
    ///     let model = CoreAIDecisionModel(modelID: "qwen3-0.6b")
    /// }
    /// ```
    @available(macOS 27, iOS 27, *)
    public struct CoreAIDecisionModel: DecisionModel {
        /// The default catalog model.
        public static let defaultModelID = "qwen3-0.6b"

        /// The default system prompt for local decision models.
        public static let defaultSystemPrompt = TokenScoring.defaultSystemPrompt

        /// The coreai-kit catalog identifier of the model.
        ///
        /// When you load a local bundle, this is the identifier reported in responses.
        public let modelID: String

        /// A local bundle directory, or `nil` to download the catalog model.
        public let bundle: URL?

        /// The model store used for downloads, or `nil` for the default store.
        public let store: ModelStore?

        /// The Core AI engine that loads the bundle.
        ///
        /// The default, `.auto`, selects an engine that returns logits:
        /// the static-shape engine for a catalog model that requires it,
        /// and the sequential engine otherwise.
        /// Select `.staticShape` for a local bundle that needs the static-shape engine.
        /// The pipelined engine does not return logits and cannot answer questions.
        public let engineVariant: EngineVariant

        /// The calibration applied to answer probabilities, or `nil` for raw probabilities.
        public var calibration: Calibration?

        /// Whether choice questions are asked once per rotation of the option order
        /// and the results averaged.
        ///
        /// This reduces position bias and requires one prompt evaluation per option.
        /// It is off by default.
        public var rotationDebiasing: Bool

        /// Whether prompts reuse the engine's cache for the prompt prefix that they share.
        ///
        /// This is on by default.
        /// The engine rewinds its cache to the prefix that a prompt shares with the previous prompt
        /// and evaluates only the remaining tokens.
        /// Because the engine computes in half precision,
        /// cached and uncached probabilities can differ slightly.
        /// When this is off, every prompt is evaluated from the start.
        /// Models that differ only in this setting load separate engines.
        ///
        /// Engines for models with recurrent layers cannot rewind into a prompt,
        /// so they reuse cached tokens only when a prompt extends the previous prompt.
        /// Each answer's ``DecisionSession/Diagnostics/cachedTokenCount`` reports the reused tokens.
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

        /// Creates a Core AI decision model.
        ///
        /// - Parameters:
        ///   - modelID: The coreai-kit catalog identifier of a chat model.
        ///     If you pass a bundle, this is only the identifier reported in responses.
        ///   - bundle: A local bundle directory with `metadata.json`, the model, and the tokenizer.
        ///     If you pass a bundle, the model loads from it instead of downloading.
        ///   - store: A model store for downloads. Pass `nil` for the default store.
        ///   - engineVariant: The Core AI engine that loads the bundle. Defaults to `.auto`.
        ///   - calibration: A calibration for answer probabilities. Pass `nil` for raw probabilities.
        ///   - rotationDebiasing: Whether to average choice results over rotations of the option order.
        ///   - prefixCaching: Whether prompts reuse the engine's cache for their shared prefix.
        ///     Defaults to `true`.
        ///   - closedThinkFallback: Whether to append a closed thinking block
        ///     if the template omits it and the tokenizer has thinking markers.
        ///     Defaults to `false`.
        ///   - systemPrompt: The system prompt.
        public init(
            modelID: String = CoreAIDecisionModel.defaultModelID,
            bundle: URL? = nil,
            store: ModelStore? = nil,
            engineVariant: EngineVariant = .auto,
            calibration: Calibration? = nil,
            rotationDebiasing: Bool = false,
            prefixCaching: Bool = true,
            closedThinkFallback: Bool = false,
            systemPrompt: String = CoreAIDecisionModel.defaultSystemPrompt
        ) {
            self.modelID = modelID
            self.bundle = bundle
            self.store = store
            self.engineVariant = engineVariant
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
            CoreAIDecisionExecutor(model: self)
        }

        /// Whether the engine for this model is loaded.
        public var isLoaded: Bool {
            DeciderCache.shared.isLoaded(cacheKey)
        }

        /// Removes this model's engine from the shared cache.
        ///
        /// Existing sessions keep their loaded engine until they are released.
        public func removeFromCache() {
            DeciderCache.shared.remove(cacheKey)
        }

        /// Removes all Core AI decision models from the shared cache.
        public static func removeAllFromCache() {
            DeciderCache.shared.removeAll()
        }

        var cacheKey: String {
            let source = bundle?.standardizedFileURL.path ?? "\((store ?? .default).directory.path)#\(modelID)"
            return "\(source)#\(engineVariant.rawValue)#\(prefixCaching ? "shared" : "unshared")"
        }

        var decisionsConfiguration: TypedDecisions.Configuration {
            var configuration = TypedDecisions.Configuration()
            configuration.engineVariant = engineVariant
            // Without prefix sharing, the engine rewinds to the start before every prompt.
            configuration.sharePrefix = prefixCaching
            return configuration
        }

        func loadDecider() async throws -> TypedDecisions {
            let modelID = modelID
            let bundle = bundle
            let store = store
            let configuration = decisionsConfiguration
            do {
                return try await DeciderCache.shared.decider(for: cacheKey) {
                    if let bundle {
                        return try await TypedDecisions(bundleAt: bundle, configuration: configuration)
                    }
                    return try await TypedDecisions(
                        catalog: modelID,
                        store: store ?? .default,
                        configuration: configuration
                    )
                }
            } catch {
                throw decisionError(for: error)
            }
        }

        /// Returns the prompt tokens that every question about a state shares.
        func prefixTokens(for state: DecisionState) async throws -> [Int] {
            try prefixTokens(for: state, tokenizer: try await loadDecider().tokenizer)
        }

        /// Returns the longest run of tokens that two different questions about a state share.
        ///
        /// The prefix covers the system prompt and the state.
        func prefixTokens(for state: DecisionState, tokenizer: any Tokenizer) throws -> [Int] {
            let probeA = TokenScoring.userMessage(state: state, question: .binary(instructions: "A"))
            let probeB = TokenScoring.userMessage(state: state, question: .binary(instructions: "Z"))
            let tokensA = try promptTokens(user: probeA, tokenizer: tokenizer)
            let tokensB = try promptTokens(user: probeB, tokenizer: tokenizer)
            let shared = zip(tokensA, tokensB).prefix { $0 == $1 }.count
            return Array(tokensA.prefix(shared))
        }

        /// Renders the chat prompt for a user message.
        func promptTokens(user: String, tokenizer: any Tokenizer) throws -> [Int] {
            let messages: [[String: any Sendable]] = [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": user],
            ]
            // Disable thinking so that the answer token follows the generation prompt.
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: ["enable_thinking": false]
            )
            guard closedThinkFallback else { return tokens }
            return TokenScoring.appendingClosedThinkBlock(
                to: tokens,
                closedThinkTokens: closedThinkTokens(tokenizer: tokenizer),
                decode: { tokenizer.decode(tokens: $0, skipSpecialTokens: false) }
            )
        }

        /// Returns the log-probability of each answer token under the full next-token distribution.
        ///
        /// - Parameters:
        ///   - logits: The logits for every token in the vocabulary.
        ///   - tokenMap: The token IDs for each answer label.
        /// - Returns: The log-probabilities grouped by label, in label order.
        /// - Throws: ``DecisionError/invalidResponse(_:)`` if a token ID is outside the logits.
        static func tokenLogProbabilities(_ logits: [Float], tokenMap: [[Int]]) throws -> [[Double]] {
            guard tokenMap.joined().allSatisfy({ logits.indices.contains($0) }) else {
                throw DecisionError.invalidResponse(
                    "The engine returned \(logits.count) logits, fewer than the answer token IDs need."
                )
            }
            let normalizer = logits.lazy.map(Double.init).logSumExp()
            return tokenMap.map { ids in ids.map { Double(logits[$0]) - normalizer } }
        }

        private func closedThinkTokens(tokenizer: any Tokenizer) -> [Int] {
            guard let opening = tokenizer.convertTokenToId("<think>"),
                let closing = tokenizer.convertTokenToId("</think>")
            else { return [] }

            let blankLines = tokenizer.encode(text: "\n\n", addSpecialTokens: false)
            return [opening] + blankLines + [closing] + blankLines
        }
    }

    // MARK: - Errors

    /// Maps an error from coreai-kit or the tokenizer to a ``DecisionError``.
    ///
    /// - Parameters:
    ///   - error: The error.
    ///   - index: The position of the question in its batch, if the error belongs to one question.
    @available(macOS 27, iOS 27, *)
    func decisionError(for error: any Error, index: Int? = nil) -> any Error {
        switch error {
        case is CancellationError:
            return CancellationError()
        case let error as DecisionError:
            return error
        case CoreAIKit.DecisionError.promptTooLong(let count, let maximum):
            let reason = "The prompt is \(count) tokens, and the model takes at most \(maximum)."
            if let index {
                return DecisionError.unsupportedQuestion(index: index, reason: reason)
            }
            return DecisionError.modelUnavailable(reason)
        case CoreAIKit.DecisionError.noLogits:
            return DecisionError.invalidResponse("The engine returned no logits.")
        case let error as LocalizedError:
            return DecisionError.modelUnavailable(error.errorDescription ?? String(describing: error))
        default:
            return DecisionError.modelUnavailable(String(describing: error))
        }
    }

    // MARK: - Shared engines

    /// Loads each bundle once and shares the engine across sessions.
    ///
    /// `TypedDecisions` runs one engine call at a time, in the order of the calls,
    /// so sessions can share it without another lock.
    /// Sessions with different states rewind the engine to their common prefix when they alternate.
    @available(macOS 27, iOS 27, *)
    private final class DeciderCache: Sendable {
        static let shared = DeciderCache()

        private let entries = Locked<[String: Task<TypedDecisions, Error>]>([:])
        private let loaded = Locked<Set<String>>([])

        func decider(
            for key: String,
            loader: @escaping @Sendable () async throws -> TypedDecisions
        ) async throws -> TypedDecisions {
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
                let decider = try await task.value
                _ = loaded.withLock { $0.insert(key) }
                try Task.checkCancellation()
                return decider
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

    // MARK: - Executor

    @available(macOS 27, iOS 27, *)
    private actor CoreAIDecisionExecutor: DecisionModelExecutor {
        let model: CoreAIDecisionModel

        /// The tokens that every question about ``prefixState`` shares.
        private var prefixTokens: [Int] = []
        private var prefixState: DecisionState?

        init(model: CoreAIDecisionModel) {
            self.model = model
        }

        func prewarm(for state: DecisionState) async throws {
            let decider = try await model.loadDecider()
            guard model.prefixCaching else { return }
            try await preparePrefix(for: state, decider: decider, fill: true)
        }

        private struct Plan {
            let question: Question
            let tokenMap: [[Int]]
            let rotations: [[Int]]
        }

        func decide(_ questions: [Question], about state: DecisionState) async throws
            -> DecisionSession.Response
        {
            let decider = try await model.loadDecider()
            let tokenizer = decider.tokenizer

            // Check every token mapping before running inference.
            let encode = { (text: String) in tokenizer.encode(text: text, addSpecialTokens: false) }
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
                // Fill the prefix only for a state that is new to this session.
                // Otherwise the engine holds the prefix or rewinds to it for the first question.
                try await preparePrefix(for: state, decider: decider, fill: prefixState != state)
            }

            var answers: [Answer] = []
            var diagnostics: [DecisionSession.Diagnostics] = []
            var inputTokenCount = 0

            for (index, plan) in plans.enumerated() {
                let isBinary = if case .binary = plan.question { true } else { false }
                var totals = [Double](repeating: 0, count: plan.tokenMap.count)
                var mass = 0.0
                var cachedTokenCount = 0
                var evaluatedTokenCount = 0
                var duration = Duration.zero

                for order in plan.rotations {
                    try Task.checkCancellation()
                    let start = ContinuousClock.now
                    let user = TokenScoring.userMessage(state: state, question: plan.question, optionOrder: order)
                    let tokens: [Int]
                    do {
                        tokens = try model.promptTokens(user: user, tokenizer: tokenizer)
                    } catch {
                        throw decisionError(for: error, index: index)
                    }
                    var elapsed = start.duration(to: .now)

                    let logits = try await self.logits(for: tokens, decider: decider, index: index)
                    let readStart = ContinuousClock.now
                    let tokenLogProbabilities = try CoreAIDecisionModel.tokenLogProbabilities(
                        logits.values,
                        tokenMap: plan.tokenMap
                    )
                    let (logMasses, allowedMass) = TokenScoring.logMasses(tokenLogProbabilities)
                    let probabilities = (model.calibration ?? .identity).probabilities(
                        logMasses: logMasses,
                        binary: isBinary
                    )
                    // Position i in this rotation holds answer order[i].
                    for (position, answerIndex) in order.enumerated() {
                        totals[answerIndex] += probabilities[position] / Double(plan.rotations.count)
                    }
                    mass += allowedMass / Double(plan.rotations.count)
                    cachedTokenCount += logits.timing.reusedTokens
                    evaluatedTokenCount += logits.timing.processedTokens
                    inputTokenCount += tokens.count
                    elapsed += readStart.duration(to: .now)
                    duration += elapsed + .seconds(logits.timing.seconds)
                }

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

        /// Runs one prompt through the engine and returns the logits after its last token.
        private func logits(for tokens: [Int], decider: TypedDecisions, index: Int) async throws -> Decision.Logits {
            do {
                return try await decider.logits(for: tokens.map(Int32.init))
            } catch {
                throw decisionError(for: error, index: index)
            }
        }

        /// Computes the shared prefix for a state and, if requested, fills the engine's cache with it.
        private func preparePrefix(for state: DecisionState, decider: TypedDecisions, fill: Bool) async throws {
            if prefixState != state {
                do {
                    prefixTokens = try model.prefixTokens(for: state, tokenizer: decider.tokenizer)
                } catch {
                    throw decisionError(for: error)
                }
                prefixState = state
            }
            // A prefix that fills the context leaves no room for a question.
            // Each question then reports that its prompt is too long.
            guard fill, !prefixTokens.isEmpty, prefixTokens.count < (await decider.maxContextLength) else { return }
            do {
                _ = try await decider.prefill(tokens: prefixTokens.map(Int32.init))
            } catch {
                throw decisionError(for: error)
            }
        }
    }
#endif
