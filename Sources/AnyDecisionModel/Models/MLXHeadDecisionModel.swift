import Foundation

#if MLX
    import HuggingFace
    import MLX
    import MLXHuggingFace
    import MLXLLM
    import MLXLMCommon
    import MLXNN
    import Tokenizers

    /// A decision model that runs a backbone language model and a trained decision head locally with MLX.
    ///
    /// Some decision models put a small head on a language model.
    /// The head reads the backbone's final hidden states at positions that its prompt format marks,
    /// such as the tokens of each option, and scores every option in one forward pass.
    /// The model reads no next-token probabilities, so it has no limit on the number of options or levels.
    ///
    /// Choose the head format when you create the model:
    ///
    /// ```swift
    /// let model = MLXHeadDecisionModel(
    ///     modelID: "mlx-community/clef-flash-4bit",
    ///     head: .jointSchema
    /// )
    /// ```
    ///
    /// Each question is evaluated as its own prompt, so answers don't depend on the other questions in a batch.
    /// Sessions share loaded weights.
    /// With prefix caching, each session evaluates the prompt prefix that contains the state once,
    /// and each question evaluates only its own tokens.
    ///
    /// Probabilities are raw by default. Set ``calibration`` to apply a fitted calibration.
    public struct MLXHeadDecisionModel: DecisionModel {
        /// The format of a model's decision head: the files it reads, its prompt format, and its network.
        public struct Head: Hashable, Sendable, CustomStringConvertible {
            enum Format: String, Hashable, Sendable {
                case jointSchema
            }

            let format: Format

            /// The joint schema head of Cloudflare's Clef models.
            ///
            /// The model directory must contain `joint_head.safetensors` and `joint_head_config.json`.
            /// A prompt can have at most 16,384 tokens.
            public static let jointSchema = Head(format: .jointSchema)

            public var description: String { format.rawValue }
        }

        /// The Hugging Face model identifier.
        public let modelID: String

        /// The Hugging Face Hub client used for downloads, or `nil` for the default client.
        public let hub: HubClient?

        /// A local directory with the model files, or `nil` to download the model.
        public let directory: URL?

        /// The format of the model's decision head.
        public let head: Head

        /// The calibration applied to answer probabilities, or `nil` for raw probabilities.
        public var calibration: Calibration?

        /// Whether sessions reuse a cache for the prompt prefix that contains the state.
        ///
        /// This is on by default.
        /// Cached and uncached probabilities can differ slightly through rounding.
        public var prefixCaching: Bool

        /// Creates an MLX decision model with a decision head.
        ///
        /// - Parameters:
        ///   - modelID: The Hugging Face model identifier.
        ///   - hub: A Hugging Face Hub client for downloads. Pass `nil` for the default client.
        ///   - directory: A local directory with the model files.
        ///     If you pass a directory, the model loads from it instead of downloading.
        ///   - head: The format of the model's decision head.
        ///   - calibration: A calibration for answer probabilities. Pass `nil` for raw probabilities.
        ///   - prefixCaching: Whether sessions reuse a cache for the prompt prefix that contains the state.
        public init(
            modelID: String,
            hub: HubClient? = nil,
            directory: URL? = nil,
            head: Head,
            calibration: Calibration? = nil,
            prefixCaching: Bool = true
        ) {
            self.modelID = modelID
            self.hub = hub
            self.directory = directory
            self.head = head
            self.calibration = calibration
            self.prefixCaching = prefixCaching
        }

        public var capabilities: DecisionCapabilities {
            DecisionCapabilities()
        }

        public func makeExecutor() -> any DecisionModelExecutor {
            switch head.format {
            case .jointSchema:
                HeadDecisionExecutor<JointSchemaHead>(model: self)
            }
        }

        /// Whether the weights for this model are loaded.
        public var isLoaded: Bool {
            loadCache.isLoaded(cacheKey)
        }

        /// Removes this model's weights from the shared cache.
        ///
        /// Existing sessions keep their loaded weights until they are released.
        public func removeFromCache() {
            loadCache.remove(cacheKey)
        }

        /// Removes all MLX head decision models from the shared cache.
        public static func removeAllFromCache() {
            loadCache.removeAll()
        }

        var cacheKey: String {
            "\(head.format.rawValue):" + (directory?.standardizedFileURL.path ?? modelID)
        }

        // MARK: - Loading

        /// The file patterns that mlx-swift-lm downloads for a model. They include the head files.
        private static let downloadPatterns = ["*.safetensors", "*.json", "*.jinja"]

        /// Returns the local directory with the model files, downloading them if needed.
        ///
        /// - Parameter fileNames: The files that the head reads. The directory must contain them.
        func modelDirectory(requiring fileNames: [String]) async throws -> URL {
            if let directory {
                try checkFiles(fileNames, in: directory)
                return directory
            }
            let downloader: any Downloader = if let hub { #hubDownloader(hub) } else { #hubDownloader() }
            return try await modelDirectory(using: downloader, requiring: fileNames)
        }

        /// Downloads the model, checking for the head's files before it downloads the weights.
        ///
        /// A model ID for a model without the head fails before a download of several gigabytes.
        func modelDirectory(using downloader: any Downloader, requiring fileNames: [String]) async throws -> URL {
            let headDirectory = try await downloader.download(
                id: modelID,
                revision: nil,
                matching: fileNames,
                useLatest: false,
                progressHandler: { _ in }
            )
            try checkFiles(fileNames, in: headDirectory)
            return try await downloader.download(
                id: modelID,
                revision: nil,
                matching: Self.downloadPatterns,
                useLatest: false,
                progressHandler: { _ in }
            )
        }

        /// Throws if a directory does not contain the head's files.
        private func checkFiles(_ fileNames: [String], in directory: URL) throws {
            let missing = fileNames.filter {
                !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
            }
            guard missing.isEmpty else {
                throw DecisionError.modelUnavailable(
                    "The model \(modelID) has no \(missing.joined(separator: " or ")). "
                        + "The \(head) head needs \(fileNames.joined(separator: " and "))."
                )
            }
        }

        /// Returns the backbone and head, loading them once for all sessions.
        func load<Head: DecisionHead>(_: Head.Type) async throws -> LoadedHeadModel<Head> {
            let model = self
            do {
                let value = try await loadCache.value(for: cacheKey) {
                    try await model.loadFromDisk(Head.self) as LoadedHeadModel<Head>
                }
                // The cache key includes the head format, so the value has this type.
                return value as! LoadedHeadModel<Head>
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as DecisionError {
                throw error
            } catch {
                throw DecisionError.modelUnavailable(String(describing: error))
            }
        }

        private func loadFromDisk<Head: DecisionHead>(_: Head.Type) async throws -> LoadedHeadModel<Head> {
            let directory = try await modelDirectory(requiring: Head.fileNames)
            let head = try Head.load(from: directory)

            // mlx-swift-lm 3.31 loads every .safetensors file in a directory,
            // including the head's weights, and then rejects them as unused.
            // Load the backbone from a directory that links to every other file.
            // (mlx-swift-lm 3.32 loads only the files in model.safetensors.index.json.)
            let container = try await Self.loadBackbone(from: directory, excluding: Set(Head.fileNames))
            let outputEmbedding = try await container.perform { context in
                try Self.replaceOutputProjection(in: context.model)
            }
            return LoadedHeadModel(container: container, head: head, outputEmbedding: outputEmbedding)
        }

        /// Loads the backbone and tokenizer from a model directory, skipping the head's files.
        private static func loadBackbone(from directory: URL, excluding excluded: Set<String>) async throws
            -> ModelContainer
        {
            let backboneDirectory = try linkedDirectory(to: directory, excluding: excluded)
            defer { try? FileManager.default.removeItem(at: backboneDirectory) }
            return try await loadModelContainer(from: backboneDirectory, using: LenientTokenizerLoader())
        }

        /// Creates a temporary directory with a symbolic link to each file in a directory.
        private static func linkedDirectory(to directory: URL, excluding excluded: Set<String>) throws -> URL {
            let fileManager = FileManager.default
            let linked = fileManager.temporaryDirectory
                .appendingPathComponent("AnyDecisionModel-\(UUID().uuidString)", isDirectory: true)
            try fileManager.createDirectory(at: linked, withIntermediateDirectories: true)
            do {
                for name in try fileManager.contentsOfDirectory(atPath: directory.path) where !excluded.contains(name) {
                    try fileManager.createSymbolicLink(
                        at: linked.appendingPathComponent(name),
                        withDestinationURL: directory.appendingPathComponent(name)
                    )
                }
            } catch {
                try? fileManager.removeItem(at: linked)
                throw error
            }
            return linked
        }

        /// Replaces the backbone's output projection with an identity,
        /// so that the backbone returns its final hidden states instead of token logits.
        ///
        /// - Returns: The original output projection.
        private static func replaceOutputProjection(in model: any LanguageModel) throws -> OutputEmbedding {
            let projections = model.namedModules().filter { key, _ in
                key == "lm_head" || key.hasSuffix(".lm_head")
            }
            guard projections.count == 1, let (key, module) = projections.first.map({ ($0.0, $0.1) }),
                let projection = module as? Linear
            else {
                throw DecisionError.modelUnavailable(
                    "The backbone has no separate output projection (lm_head), which the head reads."
                )
            }
            let embedding = OutputEmbedding(projection)
            model.update(modules: ModuleChildren.unflattened([(key, HiddenStateProjection())]))
            return embedding
        }
    }

    // MARK: - Shared weights

    /// A loaded backbone and head.
    ///
    /// The head and the output embedding are used only inside `container.perform`,
    /// which serializes access across all sessions that share the container.
    final class LoadedHeadModel<Head: DecisionHead>: @unchecked Sendable {
        let container: ModelContainer
        let head: Head
        let outputEmbedding: OutputEmbedding

        init(container: ModelContainer, head: Head, outputEmbedding: OutputEmbedding) {
            self.container = container
            self.head = head
            self.outputEmbedding = outputEmbedding
        }
    }

    private let loadCache = ModelCache<any Sendable>()

    /// Loads Hugging Face tokenizers and accepts tokenizer classes that swift-transformers doesn't register.
    ///
    /// Decision models are often saved with newer tokenizer class names, such as `Qwen3_5Tokenizer`.
    /// swift-transformers 1.3 rejects unregistered names in strict mode;
    /// outside strict mode, it uses its standard BPE tokenizer with the model's `tokenizer.json`,
    /// as it does for `Qwen2Tokenizer`.
    private struct LenientTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let upstream = try await AutoTokenizer.from(modelFolder: directory, strict: false)
            return #adaptHuggingFaceTokenizer(upstream)
        }
    }

    /// A linear layer that returns its input, so that a backbone returns hidden states.
    private final class HiddenStateProjection: Linear {
        init() {
            super.init(weight: MLXArray.zeros([1, 1]))
        }

        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            x
        }
    }

    // MARK: - Executor

    /// The prompt prefix cache for one session.
    ///
    /// The cache is read and written only inside `ModelContainer.perform`,
    /// which serializes access across all sessions that share the container.
    private final class HeadPrefixCache: @unchecked Sendable {
        var state: DecisionState?
        var tokens: [Int] = []
        var layers: [any KVCache]?
        var hidden: MLXArray?
    }

    private final class HeadDecisionExecutor<Head: DecisionHead>: DecisionModelExecutor {
        let model: MLXHeadDecisionModel
        private let prefix = HeadPrefixCache()
        private let tokenizer = Locked<(any MLXLMCommon.Tokenizer)?>(nil)
        private let prefixTokens = Locked<(state: DecisionState, tokens: [Int])?>(nil)

        init(model: MLXHeadDecisionModel) {
            self.model = model
        }

        private struct Row: Sendable {
            let question: Question
            let tokens: [Int]
            let prompt: Head.Prompt
        }

        func prewarm(for state: DecisionState) async throws {
            let loaded = try await model.load(Head.self)
            guard model.prefixCaching else { return }
            let tokens = try await prefixTokens(for: state, container: loaded.container)
            // No question fits after a prefix this long, and `decide` reports the error.
            guard tokens.count < Head.maximumPromptLength else { return }
            try await loaded.container.perform { context in
                try self.preparePrefix(tokens, for: state, model: context.model)
            }
        }

        func decide(_ questions: [Question], about state: DecisionState) async throws
            -> DecisionSession.Response
        {
            let loaded = try await model.load(Head.self)
            let start = ContinuousClock.now
            let prefixTokens = try await prefixTokens(for: state, container: loaded.container)
            let tokenizer = await tokenizer(from: loaded.container)

            // Check every prompt before running inference.
            var rows: [Row] = []
            for (index, question) in questions.enumerated() {
                try Task.checkCancellation()
                let (tokens, prompt) = Head.prompt(for: question) {
                    tokenizer.encode(text: $0, addSpecialTokens: false)
                }
                let length = prefixTokens.count + tokens.count
                guard length <= Head.maximumPromptLength else {
                    throw DecisionError.unsupportedQuestion(
                        index: index,
                        reason: "The prompt has \(length) tokens, and the model accepts at most "
                            + "\(Head.maximumPromptLength). Shorten the state or the question."
                    )
                }
                rows.append(Row(question: question, tokens: tokens, prompt: prompt))
            }
            let preparationDuration = start.duration(to: .now)

            return try await loaded.container.perform { [rows] context in
                try self.respond(
                    to: rows,
                    prefixTokens: prefixTokens,
                    state: state,
                    loaded: loaded,
                    preparationDuration: preparationDuration,
                    backbone: context.model
                )
            }
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

        /// Returns the prefix tokens for a state, tokenizing the state once for each session.
        private func prefixTokens(for state: DecisionState, container: ModelContainer) async throws -> [Int] {
            if let cached = prefixTokens.withLock({ $0 }), cached.state == state {
                return cached.tokens
            }
            let tokenizer = await tokenizer(from: container)
            try Task.checkCancellation()
            let tokens = Head.prefixTokens(for: state) { tokenizer.encode(text: $0, addSpecialTokens: false) }
            prefixTokens.withLock { $0 = (state, tokens) }
            return tokens
        }

        /// Evaluates each question's prompt and reads its answer from the head.
        private func respond(
            to rows: [Row],
            prefixTokens: [Int],
            state: DecisionState,
            loaded: LoadedHeadModel<Head>,
            preparationDuration: Duration,
            backbone: any LanguageModel
        ) throws -> DecisionSession.Response {
            if model.prefixCaching {
                try preparePrefix(prefixTokens, for: state, model: backbone)
            }

            var answers: [Answer] = []
            var diagnostics: [DecisionSession.Diagnostics] = []
            var inputTokenCount = 0
            for row in rows {
                try Task.checkCancellation()
                let start = ContinuousClock.now
                let tokens = prefixTokens + row.tokens

                let hidden: MLXArray
                let cachedTokenCount: Int
                if model.prefixCaching, let layers = prefix.layers, let prefixHidden = prefix.hidden {
                    let cache = layers.map { $0.copy() }
                    hidden = concatenated(
                        [prefixHidden, try hiddenStates(row.tokens, cache: cache, model: backbone)],
                        axis: 0
                    )
                    cachedTokenCount = prefixTokens.count
                } else {
                    hidden = try hiddenStates(tokens, cache: nil, model: backbone)
                    cachedTokenCount = 0
                }

                let logits = loaded.head.logits(
                    hidden: hidden,
                    tokens: MLXArray(tokens.map { Int32($0) }),
                    prefixLength: prefixTokens.count,
                    prompt: row.prompt,
                    outputEmbedding: loaded.outputEmbedding
                )
                let isBinary = if case .binary = row.question { true } else { false }
                let probabilities = (model.calibration ?? .identity).probabilities(
                    logMasses: logits.map(Double.init),
                    binary: isBinary
                )
                answers.append(TokenScoring.answer(for: row.question, probabilities: probabilities))
                diagnostics.append(
                    DecisionSession.Diagnostics(
                        // The head distributes all probability over the allowed answers.
                        allowedAnswerMass: 1,
                        cachedTokenCount: cachedTokenCount,
                        evaluatedTokenCount: tokens.count - cachedTokenCount,
                        duration: preparationDuration / max(1, rows.count) + start.duration(to: .now)
                    )
                )
                inputTokenCount += tokens.count
            }

            return DecisionSession.Response(
                modelID: model.modelID,
                answers: answers,
                usage: DecisionSession.Usage(inputTokenCount: inputTokenCount, outputTokenCount: 0),
                diagnostics: diagnostics
            )
        }

        /// Fills the session's prefix cache for a state, if it is not already filled.
        private func preparePrefix(_ tokens: [Int], for state: DecisionState, model: any LanguageModel) throws {
            if prefix.state == state, prefix.layers != nil { return }
            try Task.checkCancellation()

            let cache = makeCache(for: model)
            let hidden = try hiddenStates(tokens, cache: cache, model: model)
            prefix.layers = cache
            prefix.hidden = hidden
            prefix.tokens = tokens
            prefix.state = state
        }

        /// The largest number of tokens evaluated in one forward pass.
        ///
        /// MLX 0.31.1 has no fused attention kernel for Clef's head dimension of 256,
        /// so attention holds a score for every pair of query and key positions.
        /// Evaluating long inputs in chunks bounds those scores to one chunk of queries.
        private let chunkLength = 1_024

        /// Returns the backbone's final hidden states for tokens, with shape `[tokens.count, hiddenSize]`.
        ///
        /// Without a cache, an input that fits in one chunk runs in one forward pass, as in the reference.
        /// Longer inputs, and inputs after a cached prefix, run in chunks with a cache.
        private func hiddenStates(_ tokens: [Int], cache: [any KVCache]?, model: any LanguageModel) throws
            -> MLXArray
        {
            if cache == nil, tokens.count <= chunkLength {
                return forward(tokens, cache: nil, model: model)
            }
            let cache = cache ?? makeCache(for: model)
            var chunks: [MLXArray] = []
            for start in stride(from: 0, to: tokens.count, by: chunkLength) {
                try Task.checkCancellation()
                let end = min(start + chunkLength, tokens.count)
                let chunk = forward(Array(tokens[start ..< end]), cache: cache, model: model)
                eval([chunk] + cache.flatMap(\.state))
                chunks.append(chunk)
            }
            return chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 0)
        }

        /// Returns an empty cache whose attention layers pass an explicit causal mask.
        ///
        /// The symbolic causal mask in MLX 0.31.1 is wrong for some shapes
        /// in which queries follow cached keys.
        /// Other layers, such as Gated DeltaNet layers, keep the model's own caches.
        private func makeCache(for model: any LanguageModel) -> [any KVCache] {
            model.newCache(parameters: nil).map { layer in
                type(of: layer) == KVCacheSimple.self ? BatchKVCache() : layer
            }
        }

        /// Returns the backbone's final hidden states for one forward pass.
        private func forward(_ tokens: [Int], cache: [any KVCache]?, model: any LanguageModel) -> MLXArray {
            let input = MLXArray(tokens.map { Int32($0) })[.newAxis]
            return model(input, cache: cache)[0]
        }
    }
#endif
