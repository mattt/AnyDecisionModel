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

        /// The default for the largest number of prompts evaluated together in one forward pass.
        public static let defaultMaximumBatchSize = 16

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
        /// This is off by default.
        /// mlx-swift 0.31.6 and earlier include MLX 0.31.1,
        /// whose Metal attention kernel applies the causal mask incorrectly
        /// for some split-prefill shapes.
        /// MLX 0.31.2 fixes this.
        /// For models whose layers use plain key-value caches, such as Qwen3,
        /// the model passes an explicit causal mask array instead of the affected symbolic mask.
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

        /// The largest number of prompts evaluated together in one forward pass.
        ///
        /// When a request has several questions, or a choice with rotation debiasing,
        /// the model evaluates the prompt tokens that they share once,
        /// then evaluates the rest of each prompt in batches of up to this size.
        /// A group of prompts that shares more tokens, such as the rotations of one choice,
        /// also shares the evaluation of those tokens.
        /// Batches also stay within fixed limits on padded tokens and cache memory.
        /// Batching applies to models whose layers use plain key-value caches, such as Qwen3;
        /// other models evaluate each prompt separately.
        /// Set it to 1 to evaluate each prompt separately.
        /// Values less than 1 are clamped to 1.
        /// It defaults to ``defaultMaximumBatchSize``.
        public var maximumBatchSize: Int {
            didSet { maximumBatchSize = max(1, maximumBatchSize) }
        }

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
        ///   - maximumBatchSize: The largest number of prompts evaluated together in one forward pass.
        ///     Defaults to ``defaultMaximumBatchSize``.
        public init(
            modelID: String = MLXDecisionModel.defaultModelID,
            hub: HubClient? = nil,
            directory: URL? = nil,
            calibration: Calibration? = nil,
            rotationDebiasing: Bool = false,
            prefixCaching: Bool = false,
            closedThinkFallback: Bool = false,
            systemPrompt: String = MLXDecisionModel.defaultSystemPrompt,
            maximumBatchSize: Int = MLXDecisionModel.defaultMaximumBatchSize
        ) {
            self.modelID = modelID
            self.hub = hub
            self.directory = directory
            self.calibration = calibration
            self.rotationDebiasing = rotationDebiasing
            self.prefixCaching = prefixCaching
            self.closedThinkFallback = closedThinkFallback
            self.systemPrompt = systemPrompt
            self.maximumBatchSize = max(1, maximumBatchSize)
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

    /// A lock that an async caller can hold across suspension points.
    ///
    /// Waiting callers get the lock in the order in which they asked for it.
    private actor AsyncMutex {
        private var isLocked = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func withLock<T: Sendable>(_ body: @Sendable () async throws -> T) async rethrows -> T {
            if isLocked {
                await withCheckedContinuation { waiters.append($0) }
            }
            isLocked = true
            defer {
                if waiters.isEmpty {
                    isLocked = false
                } else {
                    // The lock stays locked for the next caller.
                    waiters.removeFirst().resume()
                }
            }
            return try await body()
        }
    }

    /// Lets one request at a time tokenize its prompts.
    private let tokenizationGate = AsyncMutex()

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
        private let tokenizer = Locked<(any MLXLMCommon.Tokenizer)?>(nil)

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
            let tokenizer = await tokenizer(from: container)

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

            // Prompt preparation needs only the CPU, so it runs outside the container's lock.
            // It can then run during the prefix prefill and during the GPU work of other sessions.
            async let prepared = tokenizationGate.withLock {
                let start = ContinuousClock.now
                let jobs = try await self.jobs(for: plans, about: state, tokenizer: tokenizer)
                return (jobs, start.duration(to: .now))
            }
            if model.prefixCaching {
                try await container.perform { context in
                    try self.preparePrefix(for: state, context: context)
                }
            }
            let (jobs, preparationDuration) = try await prepared

            return try await container.perform { context in
                try self.respond(to: plans, jobs: jobs, preparationDuration: preparationDuration, context: context)
            }
        }

        private struct Plan: Sendable {
            let question: Question
            let tokenMap: [[Int]]
            let rotations: [[Int]]
        }

        /// Returns the container's tokenizer.
        ///
        /// The container gives out its tokenizer under its lock,
        /// so the executor keeps it to not wait for the GPU work of other sessions again.
        private func tokenizer(from container: ModelContainer) async -> any MLXLMCommon.Tokenizer {
            if let tokenizer = tokenizer.withLock({ $0 }) { return tokenizer }
            let tokenizer = await container.tokenizer
            self.tokenizer.withLock { $0 = tokenizer }
            return tokenizer
        }

        /// The number of tasks that tokenize prompts at the same time.
        ///
        /// In measurements, tokenization got faster up to 4 tasks and slower above 6,
        /// so ``tokenizationGate`` also lets only one request tokenize at a time.
        private let tokenizationTaskCount = 4

        /// Renders and tokenizes one prompt for every question
        /// and for every option-order rotation of a choice.
        private func jobs(
            for plans: [Plan],
            about state: DecisionState,
            tokenizer: any MLXLMCommon.Tokenizer
        ) async throws -> [Job] {
            let prompts = plans.enumerated().flatMap { planIndex, plan in
                plan.rotations.map { (planIndex: planIndex, order: $0) }
            }
            let taskCount = min(tokenizationTaskCount, prompts.count)
            return try await withThrowingTaskGroup(of: [(Int, Job)].self) { group in
                for task in 0 ..< taskCount {
                    group.addTask {
                        try stride(from: task, to: prompts.count, by: taskCount).map { index in
                            try Task.checkCancellation()
                            let prompt = prompts[index]
                            let user = TokenScoring.userMessage(
                                state: state,
                                question: plans[prompt.planIndex].question,
                                optionOrder: prompt.order
                            )
                            let tokens = try self.promptTokens(user: user, tokenizer: tokenizer)
                            return (index, Job(planIndex: prompt.planIndex, order: prompt.order, tokens: tokens))
                        }
                    }
                }
                var jobs = [Job?](repeating: nil, count: prompts.count)
                for try await lane in group {
                    for (index, job) in lane { jobs[index] = job }
                }
                return jobs.map { $0! }
            }
        }

        /// Evaluates the prompts and combines the results into one answer for each question.
        private func respond(
            to plans: [Plan],
            jobs: [Job],
            preparationDuration: Duration,
            context: ModelContext
        ) throws -> DecisionSession.Response {
            let start = ContinuousClock.now
            let results = try evaluate(jobs, tokenMaps: plans.map(\.tokenMap), context: context)
            let elapsed = preparationDuration + start.duration(to: .now)

            var answers: [Answer] = []
            var diagnostics: [DecisionSession.Diagnostics] = []
            var inputTokenCount = 0

            for (planIndex, plan) in plans.enumerated() {
                let isBinary = if case .binary = plan.question { true } else { false }
                var totals = [Double](repeating: 0, count: plan.tokenMap.count)
                var mass = 0.0
                var cachedTokenCount = 0
                var evaluatedTokenCount = 0
                var jobCount = 0

                for (job, result) in zip(jobs, results) where job.planIndex == planIndex {
                    let (logMasses, allowedMass) = TokenScoring.logMasses(result.tokenLogProbabilities)
                    let probabilities = (model.calibration ?? .identity).probabilities(
                        logMasses: logMasses,
                        binary: isBinary
                    )
                    // Position i in this rotation holds answer order[i].
                    for (position, answerIndex) in job.order.enumerated() {
                        totals[answerIndex] += probabilities[position] / Double(plan.rotations.count)
                    }
                    mass += allowedMass / Double(plan.rotations.count)
                    cachedTokenCount += result.cachedTokenCount
                    evaluatedTokenCount += job.tokens.count - result.cachedTokenCount
                    inputTokenCount += job.tokens.count
                    jobCount += 1
                }

                answers.append(TokenScoring.answer(for: plan.question, probabilities: totals))
                diagnostics.append(
                    DecisionSession.Diagnostics(
                        allowedAnswerMass: mass,
                        cachedTokenCount: cachedTokenCount,
                        evaluatedTokenCount: evaluatedTokenCount,
                        // Batched prompts share the elapsed time in proportion to their number.
                        duration: elapsed * Double(jobCount) / Double(max(1, jobs.count))
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

            let cache = makeCache(for: context.model)
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
                eval(cache.flatMap(\.state))
                start = end
            }
        }

        private struct Job: Sendable {
            let planIndex: Int
            let order: [Int]
            let tokens: [Int]
        }

        private struct EvaluationResult {
            let tokenLogProbabilities: [[Double]]
            let cachedTokenCount: Int
        }

        /// The largest number of padded prompt tokens in one batched forward pass.
        ///
        /// The model computes logits for every position, so this bounds the size of that array.
        private let batchTokenBudget = 1_024

        /// The largest number of bytes of cached keys and values copied for one batch.
        ///
        /// Attention needs a copy of the shared keys and values for each prompt in a batch.
        private let batchCacheByteBudget = 2 << 30

        /// The smallest number of tokens in a shared prefill.
        ///
        /// MLX selects other kernels for a short pass, which round differently,
        /// so a shorter prefill could make an answer change with the other questions in the request.
        /// A short prefill also saves little work.
        private let minimumSharedPrefillLength = 32

        /// The prompts, the caches, and the results of one call to `evaluate`.
        private struct Evaluation {
            let jobs: [Job]
            let tokenMaps: [[[Int]]]
            let context: ModelContext
            let maximumBatchSize: Int
            var results: [EvaluationResult?]
            /// The number of tokens that the current prompts read from the session's prefix cache.
            var cachedTokenCount = 0
        }

        /// Scores every prompt and returns the results in prompt order.
        ///
        /// When the model's cache supports it,
        /// prompts that begin with the same tokens share one evaluation of those tokens,
        /// and the remaining tokens of several prompts run in one forward pass.
        private func evaluate(
            _ jobs: [Job],
            tokenMaps: [[[Int]]],
            context: ModelContext
        ) throws -> [EvaluationResult] {
            let empty = makeCache(for: context.model)
            let maximumBatchSize = empty is [BatchKVCache] ? model.maximumBatchSize : 1

            // Prompts that are evaluated together must all begin with the prefix cache to use it.
            let groups = maximumBatchSize > 1 ? [Array(jobs.indices)] : jobs.indices.map { [$0] }
            var evaluation = Evaluation(
                jobs: jobs,
                tokenMaps: tokenMaps,
                context: context,
                maximumBatchSize: maximumBatchSize,
                results: [EvaluationResult?](repeating: nil, count: jobs.count)
            )
            for group in groups {
                var cache = empty
                var depth = 0
                if model.prefixCaching, let layers = prefix.layers,
                    group.allSatisfy({
                        jobs[$0].tokens.count > prefix.tokens.count && jobs[$0].tokens.starts(with: prefix.tokens)
                    })
                {
                    cache = layers
                    depth = prefix.tokens.count
                }
                evaluation.cachedTokenCount = depth
                let rest = try score(group, after: depth, cache: cache, in: &evaluation)
                try scoreBatches(of: rest, after: depth, cache: cache, in: &evaluation)
            }
            return evaluation.results.map { $0! }
        }

        /// Scores the groups of prompts for which a shared prefill saves work,
        /// and returns the prompts that remain.
        ///
        /// `cache` holds the first `depth` tokens of every prompt in the group and is not changed.
        /// The prompts form a prefix tree.
        /// This method evaluates the tokens that the whole group shares,
        /// then does the same for each subgroup that continues with the same token.
        private func score(
            _ group: [Int],
            after depth: Int,
            cache: [any KVCache],
            in evaluation: inout Evaluation
        ) throws -> [Int] {
            guard group.count > 1 else { return group }
            let jobs = evaluation.jobs

            // Find the number of tokens that every prompt shares.
            // Prompts can be identical, so each prompt keeps at least one token of its own.
            let first = jobs[group[0]].tokens
            var common = group.map { jobs[$0].tokens.count }.min()! - 1
            for job in group.dropFirst() {
                common = zip(first, jobs[job].tokens).prefix(common).prefix { $0 == $1 }.count
            }

            var depth = depth
            var cache = cache
            let extended = common - depth >= minimumSharedPrefillLength
            if extended {
                cache = cache.map { $0.copy() }
                try prefill(Array(first[depth ..< common]), cache: cache, model: evaluation.context.model)
                depth = common
            }

            // Prompts that continue with the same token can share more tokens.
            var rest: [Int] = []
            var subgroups: [Int: [Int]] = [:]
            for job in group {
                subgroups[jobs[job].tokens[common], default: []].append(job)
            }
            for subgroup in subgroups.values.sorted(by: { $0[0] < $1[0] }) {
                // Identical prompts, or a prompt that begins another, cannot be divided further.
                rest +=
                    subgroup.count < group.count
                    ? try score(subgroup, after: depth, cache: cache, in: &evaluation) : subgroup
            }

            guard extended else { return rest }
            try scoreBatches(of: rest, after: depth, cache: cache, in: &evaluation)
            return []
        }

        /// Scores prompts in batches.
        ///
        /// `cache` holds the first `depth` tokens of every prompt and is not changed.
        private func scoreBatches(
            of group: [Int],
            after depth: Int,
            cache: [any KVCache],
            in evaluation: inout Evaluation
        ) throws {
            let jobs = evaluation.jobs
            let model = evaluation.context.model

            // Each prompt in a batch gets a copy of the cached keys and values.
            let cacheByteCount = cache.flatMap(\.state).reduce(0) { $0 + $1.nbytes }
            let maximumBatchSize = min(
                evaluation.maximumBatchSize,
                max(1, batchCacheByteBudget / max(1, cacheByteCount))
            )

            // Sort the prompts by length so that each batch needs little padding.
            var batches: [[Int]] = []
            for job in group.sorted(by: { jobs[$0].tokens.count < jobs[$1].tokens.count }) {
                // This prompt is the longest so far, so it sets the width of the batch.
                let width = jobs[job].tokens.count - depth
                if let count = batches.last?.count,
                    count < maximumBatchSize, (count + 1) * width <= batchTokenBudget
                {
                    batches[batches.count - 1].append(job)
                } else {
                    batches.append([job])
                }
            }

            for batch in batches {
                try Task.checkCancellation()
                var suffixes = batch.map { Array(jobs[$0].tokens[depth...]) }
                // Only a ``BatchKVCache`` allows a batch size above 1.
                let cache: [any KVCache] =
                    batch.count == 1
                    ? cache.map { $0.copy() }
                    : cache.map { ($0 as! BatchKVCache).broadcast(to: batch.count) }

                if batch.count == 1, suffixes[0].count > prefillStepSize {
                    // A prompt too long for one forward pass has all but its last token prefilled.
                    try prefill(Array(suffixes[0].dropLast()), cache: cache, model: model)
                    suffixes[0] = [suffixes[0].last!]
                }

                // Right-pad each suffix with its last token.
                // Causal attention keeps the padding from affecting the real positions.
                let width = suffixes.map(\.count).max()!
                var input: [Int32] = []
                input.reserveCapacity(batch.count * width)
                for suffix in suffixes {
                    input += suffix.map { Int32($0) }
                    input += [Int32](repeating: Int32(suffix.last!), count: width - suffix.count)
                }
                let logits = model(MLXArray(input, [batch.count, width]), cache: cache)
                let vocabularySize = logits.dim(-1)

                // Read the logits after each prompt's last real token.
                let positions = suffixes.enumerated().map { Int32($0 * width + $1.count - 1) }
                let last = take(logits.reshaped(-1, vocabularySize), MLXArray(positions), axis: 0)
                    .asType(.float32)
                let logProbabilities = last - logSumExp(last, axis: -1, keepDims: true)

                var indices: [Int32] = []
                for (row, job) in batch.enumerated() {
                    for id in evaluation.tokenMaps[jobs[job].planIndex].joined() {
                        indices.append(Int32(row * vocabularySize + id))
                    }
                }
                let gathered = take(logProbabilities.reshaped(-1), MLXArray(indices)).asArray(Float.self)

                var offset = 0
                for job in batch {
                    // Group the log-probabilities by label, in label position order.
                    let tokenLogProbabilities = evaluation.tokenMaps[jobs[job].planIndex].map { ids in
                        defer { offset += ids.count }
                        return gathered[offset ..< offset + ids.count].map(Double.init)
                    }
                    evaluation.results[job] = EvaluationResult(
                        tokenLogProbabilities: tokenLogProbabilities,
                        cachedTokenCount: evaluation.cachedTokenCount
                    )
                }
            }
        }

        /// Returns an empty cache for the model.
        ///
        /// Models whose layers all use plain key-value caches get a ``BatchKVCache`` for each layer.
        /// It can hold a batch of prompts,
        /// and it uses an explicit causal mask when queries follow cached keys.
        private func makeCache(for model: any LanguageModel) -> [any KVCache] {
            let cache = model.newCache(parameters: nil)
            guard cache.allSatisfy({ type(of: $0) == KVCacheSimple.self }) else { return cache }
            return cache.map { _ in BatchKVCache() }
        }
    }

    /// A key-value cache that holds a batch of sequences with a shared length.
    ///
    /// It masks attention with an explicit causal mask array.
    /// The symbolic causal mask in MLX 0.31.1 is wrong for some shapes
    /// in which the queries follow cached keys.
    private final class BatchKVCache: KVCache {
        private var keys: MLXArray?
        private var values: MLXArray?
        private(set) var offset = 0

        var maxSize: Int? { nil }

        func innerState() -> [MLXArray] {
            [keys, values].compactMap { $0 }
        }

        func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, keys], axis: 2)
                self.values = concatenated([currentValues, values], axis: 2)
            } else {
                self.keys = keys
                self.values = values
            }
            offset += keys.dim(2)
            return (self.keys!, self.values!)
        }

        var state: [MLXArray] {
            get { innerState() }
            set {
                keys = newValue.first
                values = newValue.last
                offset = keys?.dim(2) ?? 0
            }
        }

        var metaState: [String] {
            get { [""] }
            set {}
        }

        var isTrimmable: Bool { false }

        func trim(_ n: Int) -> Int { 0 }

        func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
            n == 1 ? .none : .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }

        func copy() -> any KVCache {
            let copy = BatchKVCache()
            copy.keys = keys
            copy.values = values
            copy.offset = offset
            return copy
        }

        /// Returns a cache that holds `count` copies of this cache's one sequence.
        ///
        /// The copies are broadcast views,
        /// so the keys and values are copied once, when the next update concatenates them.
        func broadcast(to count: Int) -> BatchKVCache {
            let copy = BatchKVCache()
            copy.keys = keys.map { MLX.broadcast($0, to: [count] + $0.shape.dropFirst()) }
            copy.values = values.map { MLX.broadcast($0, to: [count] + $0.shape.dropFirst()) }
            copy.offset = offset
            return copy
        }
    }
#endif
