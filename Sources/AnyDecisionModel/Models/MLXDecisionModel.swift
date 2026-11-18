import Foundation

#if MLX
    import HuggingFace
    import MLX
    import MLXHuggingFace
    import MLXLLM
    import MLXLMCommon
    import MLXNN
    import MLXVLM
    import Tokenizers

    /// A decision model that runs a causal language model locally with MLX.
    ///
    /// The model reads decisions from next-token probabilities in one forward pass,
    /// with no text generation.
    /// For each allowed answer label, it sums the probability of the label's
    /// single-token variants, then normalizes across the allowed answers.
    /// The model converts half-precision parameters to float32 when it loads,
    /// so the forward pass and the logits are float32.
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
    ///
    /// The model also runs Liquid AI's d1 decision models on LFM2-VL,
    /// such as `LiquidAI/d1-3B-MLX-8bit`, with their own prompt and readout.
    /// See ``PromptFormat``.
    public struct MLXDecisionModel: DecisionModel {
        /// How the model writes prompts and reads answers.
        #if hasAttribute(nonexhaustive)
            @nonexhaustive
        #endif
        public enum PromptFormat: Hashable, Sendable {
            /// Selects the format from the model's `config.json` when the model loads.
            ///
            /// A model whose `model_type` is `lfm2_vl`
            /// and whose `auto_map.AutoModel` ends in `.D1Model` uses ``d1LFM2``,
            /// and every other model uses ``standard``.
            case automatic

            /// The package's prompt for causal language models.
            ///
            /// The prompt has the system prompt and a user message with the state and the question,
            /// written with the model's chat template.
            /// Binary answers are read from "yes" and "no",
            /// choices from A to Z, and scores from 0 to 9.
            /// Each answer sums the probability of its label's single-token variants.
            case standard

            /// The prompt and readout of Liquid AI's d1 decision models on LFM2-VL, such as d1-3B.
            ///
            /// This format doesn't cover d1 models on LFM3, such as d1-2B and d1-Omni-2B,
            /// which use a different prompt and readout.
            ///
            /// This follows `prompt.py` in the
            /// [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) repository:
            /// a user turn with the state, then `QUESTION:`, the question, and its reply instructions,
            /// with no system prompt.
            /// A JSON state is written with an indent of two spaces and sorted keys.
            /// Choice options get d1's option codes,
            /// and each answer scores the largest log-probability among its token variants.
            /// ``MLXDecisionModel/systemPrompt`` and ``MLXDecisionModel/closedThinkFallback``
            /// don't apply.
            case d1LFM2
        }

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
        /// This reduces position bias and requires one prompt evaluation per option.
        /// Several of these evaluations can share a batched forward pass.
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
        /// The ``PromptFormat/d1LFM2`` format ignores it,
        /// including when ``PromptFormat/automatic`` selects that format.
        public var closedThinkFallback: Bool

        /// The system prompt.
        ///
        /// The ``PromptFormat/d1LFM2`` format has no system prompt and ignores this value,
        /// including when ``PromptFormat/automatic`` selects that format.
        public var systemPrompt: String

        /// The largest number of prompts evaluated together in one forward pass.
        ///
        /// When a request has several questions, or a choice with rotation debiasing,
        /// the model evaluates their prompts in batches of up to this size.
        /// Only the session's prefix cache can share evaluation across prompts.
        /// Batches also stay within fixed limits on padded tokens and cache memory.
        /// Batching applies to models whose layers use plain key-value caches, such as Qwen3;
        /// other models evaluate each prompt separately.
        /// Set it to 1 to evaluate each prompt separately.
        /// Values less than 1 are clamped to 1.
        /// It defaults to ``defaultMaximumBatchSize``.
        public var maximumBatchSize: Int {
            didSet { maximumBatchSize = max(1, maximumBatchSize) }
        }

        /// How the model writes prompts and reads answers.
        ///
        /// It defaults to ``PromptFormat/automatic``,
        /// which uses ``PromptFormat/d1LFM2`` for d1 models on LFM2-VL
        /// and ``PromptFormat/standard`` for other models.
        public var promptFormat: PromptFormat

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
        ///     The ``PromptFormat/d1LFM2`` format ignores it.
        ///   - systemPrompt: The system prompt.
        ///     The ``PromptFormat/d1LFM2`` format ignores it.
        ///   - maximumBatchSize: The largest number of prompts evaluated together in one forward pass.
        ///     Defaults to ``defaultMaximumBatchSize``.
        public init(
            modelID: String,
            hub: HubClient? = nil,
            directory: URL? = nil,
            calibration: Calibration? = nil,
            rotationDebiasing: Bool = false,
            prefixCaching: Bool = false,
            closedThinkFallback: Bool = false,
            systemPrompt: String = MLXDecisionModel.defaultSystemPrompt,
            maximumBatchSize: Int = MLXDecisionModel.defaultMaximumBatchSize
        ) {
            self.init(
                modelID: modelID,
                hub: hub,
                directory: directory,
                calibration: calibration,
                rotationDebiasing: rotationDebiasing,
                prefixCaching: prefixCaching,
                closedThinkFallback: closedThinkFallback,
                systemPrompt: systemPrompt,
                maximumBatchSize: maximumBatchSize,
                promptFormat: .automatic
            )
        }

        /// Creates an MLX decision model with a prompt format.
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
        ///     The ``PromptFormat/d1LFM2`` format ignores it.
        ///   - systemPrompt: The system prompt.
        ///     The ``PromptFormat/d1LFM2`` format ignores it.
        ///   - maximumBatchSize: The largest number of prompts evaluated together in one forward pass.
        ///     Defaults to ``defaultMaximumBatchSize``.
        ///   - promptFormat: How the model writes prompts and reads answers.
        ///     Defaults to ``PromptFormat/automatic``.
        @_disfavoredOverload
        public init(
            modelID: String,
            hub: HubClient? = nil,
            directory: URL? = nil,
            calibration: Calibration? = nil,
            rotationDebiasing: Bool = false,
            prefixCaching: Bool = false,
            closedThinkFallback: Bool = false,
            systemPrompt: String = MLXDecisionModel.defaultSystemPrompt,
            maximumBatchSize: Int = MLXDecisionModel.defaultMaximumBatchSize,
            promptFormat: PromptFormat = .automatic
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
            self.promptFormat = promptFormat
        }

        public var capabilities: DecisionCapabilities {
            DecisionCapabilities(
                maximumChoiceOptions: TokenScoring.choiceLabels.count,
                maximumScoreLevels: TokenScoring.scoreLabels.count,
                probabilitySource: .native
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

        /// Returns the format that applies to a loaded model.
        ///
        /// - Parameter declaresD1LFM2: Whether the model's `config.json` declares a d1 model.
        func resolvedFormat(declaresD1LFM2: Bool) -> PromptFormat {
            guard promptFormat == .automatic else { return promptFormat }
            return declaresD1LFM2 ? .d1LFM2 : .standard
        }

        /// Loads the model, or returns it from the shared cache.
        ///
        /// An LFM2-VL model (`model_type` `lfm2_vl`), such as d1, loads through `MLXVLM`,
        /// because `MLXLLM` doesn't have that architecture.
        /// Only the language model runs; the vision tower stays loaded but unused.
        /// Every other model loads through `MLXLLM`.
        /// The loader depends only on `config.json`, not on ``promptFormat``,
        /// because models with the same path or identifier share one loaded container.
        /// The model chooses the factory itself instead of calling `loadModelContainer`,
        /// which tries `MLXVLM` before `MLXLLM`
        /// and would load model types that both register, such as `gemma3`, as vision models.
        func loadModel() async throws -> LoadedModel {
            let modelID = modelID
            let hub = hub
            let directory = directory
            do {
                return try await containerCache.value(for: cacheKey) {
                    // mlx-swift-lm 3 takes explicit downloader and tokenizer loaders.
                    // The macros back them with HubClient and AutoTokenizer.
                    let tokenizerLoader = #huggingFaceTokenizerLoader()
                    let downloader = directory == nil ? (hub.map { #hubDownloader($0) } ?? #hubDownloader()) : nil

                    let configuration: Data
                    if let directory {
                        configuration = try Data(contentsOf: directory.appendingPathComponent("config.json"))
                    } else {
                        let snapshot = try await downloader!.download(
                            id: modelID,
                            revision: "main",
                            matching: ["config.json"],
                            useLatest: false,
                            progressHandler: { _ in }
                        )
                        configuration = try Data(contentsOf: snapshot.appendingPathComponent("config.json"))
                    }
                    let declaresD1LFM2 = D1LFM2Prompt.isD1LFM2Configuration(configuration)
                    let architecture = ModelArchitecture(configuration: configuration)

                    let container: ModelContainer
                    if architecture == .lfm2VL {
                        var modelDirectory = directory
                        if modelDirectory == nil {
                            // These are the files that mlx-swift-lm's loaders download.
                            modelDirectory = try await downloader!.download(
                                id: modelID,
                                revision: "main",
                                matching: ["*.safetensors", "*.json", "*.jinja"],
                                useLatest: false,
                                progressHandler: { _ in }
                            )
                        }
                        container = try await loadLFM2VL(
                            from: modelDirectory!,
                            configuration: configuration,
                            using: tokenizerLoader
                        )
                    } else if let directory {
                        container = try await LLMModelFactory.shared.loadContainer(
                            from: directory,
                            using: tokenizerLoader
                        )
                    } else {
                        container = try await LLMModelFactory.shared.loadContainer(
                            from: downloader!,
                            using: tokenizerLoader,
                            configuration: ModelConfiguration(id: modelID, revision: "main")
                        )
                    }
                    await container.perform { context in
                        // A vision model's vision tower never runs, so it keeps its parameters.
                        convertFloatingPointParametersToFloat32(
                            context.model,
                            under: architecture == .lfm2VL ? "language_model" : nil
                        )
                    }
                    return LoadedModel(container: container, declaresD1LFM2: declaresD1LFM2)
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

    /// Loads an LFM2-VL model, such as d1, through MLXVLM.
    ///
    /// mlx-swift-lm 3.32.3 reads the MLP width of LFM2-VL's language model from `block_ff_dim` only,
    /// and LFM2.5-VL checkpoints such as d1 give it as `intermediate_size`.
    /// When ``amendedLFM2VLConfiguration(_:)`` changes the configuration,
    /// the model loads from a directory that links to every other file
    /// and holds the amended `config.json`.
    ///
    /// This is a workaround.
    /// [ml-explore/mlx-swift-lm#695](https://github.com/ml-explore/mlx-swift-lm/pull/695),
    /// merged on 2026-10-16, reads `intermediate_size`,
    /// but mlx-swift-lm 3.32.3 doesn't include it.
    /// Remove this workaround and ``amendedLFM2VLConfiguration(_:)``
    /// once the package requires a release with that fix.
    private func loadLFM2VL(
        from directory: URL,
        configuration: Data,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContainer {
        guard let amended = amendedLFM2VLConfiguration(configuration) else {
            return try await VLMModelFactory.shared.loadContainer(from: directory, using: tokenizerLoader)
        }
        let linked = try makeLinkedDirectory(to: directory, excluding: ["config.json"])
        defer { try? FileManager.default.removeItem(at: linked) }
        try amended.write(to: linked.appendingPathComponent("config.json"))
        return try await VLMModelFactory.shared.loadContainer(from: linked, using: tokenizerLoader)
    }

    /// Returns an LFM2-VL configuration with the text model keys that mlx-swift-lm 3.32.3 reads,
    /// or `nil` if it already has them.
    ///
    /// Transformers 5 writes the MLP width of the text model as `intermediate_size`
    /// and the rotary base as `rope_parameters.rope_theta`.
    /// This copies them to `block_ff_dim` and `rope_theta` when those keys are missing.
    func amendedLFM2VLConfiguration(_ data: Data) -> Data? {
        guard var object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            var text = object["text_config"] as? [String: Any]
        else { return nil }
        var changed = false
        if text["block_ff_dim"] == nil, let width = text["intermediate_size"] {
            text["block_ff_dim"] = width
            changed = true
        }
        if text["rope_theta"] == nil,
            let rope = text["rope_parameters"] as? [String: Any],
            let theta = rope["rope_theta"]
        {
            text["rope_theta"] = theta
            changed = true
        }
        guard changed else { return nil }
        object["text_config"] = text
        return try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    /// The architecture that decides which mlx-swift-lm library loads a model.
    enum ModelArchitecture: Equatable {
        /// A causal language model, which `MLXLLM` loads.
        case languageModel

        /// An LFM2-VL model (`model_type` `lfm2_vl`), which `MLXVLM` loads.
        case lfm2VL

        /// Reads the architecture from a model's `config.json`.
        init(configuration data: Data) {
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            self = object?["model_type"] as? String == "lfm2_vl" ? .lfm2VL : .languageModel
        }
    }

    /// A loaded model and what its `config.json` declares.
    struct LoadedModel: Sendable {
        let container: ModelContainer

        /// Whether the model's `config.json` declares a d1 model.
        let declaresD1LFM2: Bool
    }

    /// Converts a model's half-precision parameters to float32,
    /// so that the forward pass computes in float32.
    ///
    /// Quantized weights keep their packed integer type;
    /// their scales and biases, norms, and unquantized weights become float32.
    ///
    /// In MLX 0.32, a half-precision matrix multiplication can round a row differently
    /// depending on how many rows the call has.
    /// So a prompt's logits depend on how much of it the prefix cache holds
    /// and on which other prompts share its batch.
    /// With Qwen3 4B 4-bit in bfloat16, one answer probability was 0.05, 0.10, 0.18, or 0.38,
    /// depending on prefix caching and batching;
    /// in float32, it was 0.075 on every path.
    /// The conversion makes Qwen3 4B 4-bit 11% larger, from 2.26 GB to 2.51 GB,
    /// and doubles the size of an unquantized half-precision model.
    ///
    /// - Parameters:
    ///   - model: The model.
    ///   - module: The key of the module whose parameters to convert,
    ///     or `nil` to convert every parameter.
    private func convertFloatingPointParametersToFloat32(_ model: any LanguageModel, under module: String? = nil) {
        let parameters = model.parameters().flattened().compactMap { key, value -> (String, MLXArray)? in
            if let module, !key.hasPrefix(module + ".") { return nil }
            return value.dtype == .bfloat16 || value.dtype == .float16 ? (key, value.asType(.float32)) : nil
        }
        guard !parameters.isEmpty else { return }
        model.update(parameters: ModuleParameters.unflattened(parameters))
        eval(model)
    }

    // MARK: - Shared weights

    private let containerCache = ModelCache<LoadedModel>()

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
            let loaded = try await model.loadModel()
            guard model.prefixCaching else { return }
            let format = model.resolvedFormat(declaresD1LFM2: loaded.declaresD1LFM2)
            try await loaded.container.perform { context in
                try self.preparePrefix(for: state, format: format, context: context)
            }
        }

        func decide(_ request: DecisionRequest) async throws -> DecisionSession.Response {
            let questions = request.questions
            let state = request.state
            let loaded = try await model.loadModel()
            let container = loaded.container
            let format = model.resolvedFormat(declaresD1LFM2: loaded.declaresD1LFM2)
            let tokenizer = await tokenizer(from: container)

            // Check every token mapping before running inference.
            let encode = { (text: String) in tokenizer.encode(text: text, addSpecialTokens: false) }
            let plans = try questions.enumerated().map { index, question -> Plan in
                let labels = TokenScoring.labels(for: question)
                var rotations = [Array(labels.indices)]
                if model.rotationDebiasing, case .choice = question {
                    rotations = labels.indices.map { r in
                        Array(labels.indices[r...]) + Array(labels.indices[..<r])
                    }
                }
                switch format {
                case .d1LFM2:
                    // d1's option codes can depend on the option order, so each rotation has its own tokens.
                    let prompts = try rotations.map { order in
                        let rendered = try D1LFM2Prompt.render(
                            question,
                            optionOrder: order,
                            index: index,
                            encode: encode
                        )
                        return PlannedPrompt(order: order, tokenMap: rendered.tokenGroups, d1Question: rendered)
                    }
                    return Plan(question: question, prompts: prompts)
                case .automatic, .standard:
                    let tokenMap = try TokenScoring.tokenMap(for: labels, index: index, encode: encode)
                    let prompts = rotations.map { PlannedPrompt(order: $0, tokenMap: tokenMap, d1Question: nil) }
                    return Plan(question: question, prompts: prompts)
                }
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
                    try self.preparePrefix(for: state, format: format, context: context)
                }
            }
            let (jobs, preparationDuration) = try await prepared

            return try await container.perform { context in
                try self.respond(
                    to: plans,
                    jobs: jobs,
                    format: format,
                    preparationDuration: preparationDuration,
                    context: context
                )
            }
        }

        private struct Plan: Sendable {
            let question: Question
            /// One prompt for each option-order rotation.
            let prompts: [PlannedPrompt]
        }

        /// One prompt for a question, before tokenization.
        private struct PlannedPrompt: Sendable {
            /// The order of the answers in the prompt, as indices into the question's answers.
            let order: [Int]
            /// The token IDs for each answer, in prompt order.
            let tokenMap: [[Int]]
            /// The rendered question for the d1 format, or `nil` for the standard format.
            let d1Question: D1LFM2Prompt.RenderedQuestion?
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
                plan.prompts.map { (planIndex: planIndex, prompt: $0) }
            }
            let taskCount = min(tokenizationTaskCount, prompts.count)
            return try await withThrowingTaskGroup(of: [(Int, Job)].self) { group in
                for task in 0 ..< taskCount {
                    group.addTask {
                        try stride(from: task, to: prompts.count, by: taskCount).map { index in
                            try Task.checkCancellation()
                            let (planIndex, prompt) = prompts[index]
                            let tokens = try self.promptTokens(
                                state: state,
                                question: plans[planIndex].question,
                                prompt: prompt,
                                tokenizer: tokenizer
                            )
                            let job = Job(
                                planIndex: planIndex,
                                order: prompt.order,
                                tokens: tokens,
                                tokenMap: prompt.tokenMap
                            )
                            return (index, job)
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
            format: MLXDecisionModel.PromptFormat,
            preparationDuration: Duration,
            context: ModelContext
        ) throws -> DecisionSession.Response {
            let start = ContinuousClock.now
            let results = try evaluate(jobs, context: context)
            let elapsed = preparationDuration + start.duration(to: .now)

            var answers: [Answer] = []
            var diagnostics: [DecisionSession.Diagnostics] = []
            var inputTokenCount = 0

            for (planIndex, plan) in plans.enumerated() {
                let isBinary = if case .binary = plan.question { true } else { false }
                var totals = [Double](repeating: 0, count: plan.prompts[0].order.count)
                var mass = 0.0
                var cachedTokenCount = 0
                var evaluatedTokenCount = 0
                var jobCount = 0

                for (job, result) in zip(jobs, results) where job.planIndex == planIndex {
                    // d1 scores each answer by its best token; the standard format sums its tokens.
                    let (logMasses, allowedMass) =
                        format == .d1LFM2
                        ? D1LFM2Prompt.scores(result.tokenLogProbabilities)
                        : TokenScoring.logMasses(result.tokenLogProbabilities)
                    let probabilities = (model.calibration ?? .identity).probabilities(
                        logMasses: logMasses,
                        binary: isBinary
                    )
                    // Position i in this rotation holds answer order[i].
                    for (position, answerIndex) in job.order.enumerated() {
                        totals[answerIndex] += probabilities[position] / Double(plan.prompts.count)
                    }
                    mass += allowedMass / Double(plan.prompts.count)
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

        /// Renders and tokenizes the prompt for a question about a state.
        private func promptTokens(
            state: DecisionState,
            question: Question,
            prompt: PlannedPrompt,
            tokenizer: any MLXLMCommon.Tokenizer
        ) throws -> [Int] {
            if let d1Question = prompt.d1Question {
                return try d1PromptTokens(state: state, question: d1Question, tokenizer: tokenizer)
            }
            let user = try TokenScoring.userMessage(state: state, question: question, optionOrder: prompt.order)
            return try promptTokens(user: user, tokenizer: tokenizer)
        }

        /// Tokenizes a d1 prompt.
        ///
        /// The prompt has its own chat markup and beginning-of-sequence token,
        /// so the tokenizer adds no special tokens, as in `prompt.py`.
        private func d1PromptTokens(
            state: DecisionState,
            question: D1LFM2Prompt.RenderedQuestion,
            tokenizer: any MLXLMCommon.Tokenizer
        ) throws -> [Int] {
            let text = try D1LFM2Prompt.prompt(state: state, question: question, bos: tokenizer.bosToken ?? "")
            return tokenizer.encode(text: text, addSpecialTokens: false)
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
        /// which covers the system prompt, if any, and the state.
        private func preparePrefix(
            for state: DecisionState,
            format: MLXDecisionModel.PromptFormat,
            context: ModelContext
        ) throws {
            if prefix.state == state, prefix.layers != nil { return }

            let tokenizer = context.tokenizer
            let tokensA: [Int]
            let tokensB: [Int]
            if format == .d1LFM2 {
                let probeA = D1LFM2Prompt.RenderedQuestion(text: "A", tokenGroups: [])
                let probeB = D1LFM2Prompt.RenderedQuestion(text: "Z", tokenGroups: [])
                tokensA = try d1PromptTokens(state: state, question: probeA, tokenizer: tokenizer)
                tokensB = try d1PromptTokens(state: state, question: probeB, tokenizer: tokenizer)
            } else {
                let probeA = try TokenScoring.userMessage(state: state, question: .binary(instructions: "A"))
                let probeB = try TokenScoring.userMessage(state: state, question: .binary(instructions: "Z"))
                tokensA = try promptTokens(user: probeA, tokenizer: tokenizer)
                tokensB = try promptTokens(user: probeB, tokenizer: tokenizer)
            }
            let shared = zip(tokensA, tokensB).prefix { $0 == $1 }.count
            let prefixTokens = Array(tokensA.prefix(shared))

            guard !prefixTokens.isEmpty else {
                prefix.tokens = []
                prefix.layers = nil
                prefix.state = state
                return
            }

            let cache = try makeCache(for: context.model)
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
            /// The token IDs for each answer, in prompt order.
            let tokenMap: [[Int]]
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

        /// The prompts, the caches, and the results of one call to `evaluate`.
        private struct Evaluation {
            let jobs: [Job]
            let context: ModelContext
            let maximumBatchSize: Int
            var results: [EvaluationResult?]
            /// The number of tokens that the current prompts read from the session's prefix cache.
            var cachedTokenCount = 0
        }

        /// Scores every prompt and returns the results in prompt order.
        ///
        /// When the model's cache supports it, several prompts run in one forward pass.
        /// Prefill boundaries must not depend on the other prompts in the request:
        /// changing those boundaries can change probabilities through rounding.
        private func evaluate(_ jobs: [Job], context: ModelContext) throws -> [EvaluationResult] {
            let empty = try makeCache(for: context.model)
            let maximumBatchSize = empty is [BatchKVCache] ? model.maximumBatchSize : 1

            // Prompts that are evaluated together must all begin with the prefix cache to use it.
            let groups = maximumBatchSize > 1 ? [Array(jobs.indices)] : jobs.indices.map { [$0] }
            var evaluation = Evaluation(
                jobs: jobs,
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
                try scoreBatches(of: group, after: depth, cache: cache, in: &evaluation)
            }
            return evaluation.results.map { $0! }
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
                    for id in jobs[job].tokenMap.joined() {
                        indices.append(Int32(row * vocabularySize + id))
                    }
                }
                let gathered = take(logProbabilities.reshaped(-1), MLXArray(indices)).asArray(Float.self)

                var offset = 0
                for job in batch {
                    // Group the log-probabilities by label, in label position order.
                    let tokenLogProbabilities = jobs[job].tokenMap.map { ids in
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
        private func makeCache(for model: any LanguageModel) throws -> [any KVCache] {
            let cache = try model.newCache(parameters: nil)
            guard cache.allSatisfy({ type(of: $0) == KVCacheSimple.self }) else { return cache }
            return cache.map { _ in BatchKVCache() }
        }
    }
#endif
