import Foundation

#if MLX
    import MLX
    import MLXNN

    /// The joint schema head of Cloudflare's Clef models.
    ///
    /// The head reads the backbone's final hidden states.
    /// It routes evidence from the whole prompt to each option,
    /// summarizes the options of each field,
    /// and scores every option against its field.
    /// This is a port of `JointSchemaHead` in the reference implementation, `clef_mlx.py`.
    /// Its weights load from `joint_head.safetensors` without conversion.
    final class JointSchemaHead: Module, DecisionHead {
        /// The head configuration in `joint_head_config.json`.
        struct Configuration: Codable, Hashable {
            var hiddenSize: Int
            var width: Int
            var routingLayers: Int
            var layers: Int
            var heads: Int
            var feedforward: Int

            enum CodingKeys: String, CodingKey {
                case hiddenSize = "hidden_size"
                case width
                case routingLayers = "routing_layers"
                case layers, heads, feedforward
            }
        }

        /// The positions that the head reads for one field, in prompt coordinates.
        struct FieldSpans {
            /// The question type: 0 for binary, 1 for choice, 2 for score.
            var typeID: Int

            /// The positions of the instruction tokens.
            var questionSpan: Range<Int>

            /// The positions of each option's tokens, in head order.
            var optionSpans: [Range<Int>]
        }

        @ModuleInfo(key: "hidden_norm") var hiddenNorm: LayerNorm
        @ModuleInfo(key: "memory_projection") var memoryProjection: Linear
        @ModuleInfo(key: "question_projection") var questionProjection: Linear
        @ModuleInfo(key: "option_question_projection") var optionQuestionProjection: Linear
        @ModuleInfo(key: "global_projection") var globalProjection: Linear
        @ModuleInfo(key: "option_context_projection") var optionContextProjection: Linear
        @ModuleInfo(key: "option_lexical_projection") var optionLexicalProjection: Linear
        @ModuleInfo(key: "type_embedding") var typeEmbedding: Embedding
        @ModuleInfo(key: "evidence_layers") var evidenceLayers: [EvidenceRoutingLayer]
        @ModuleInfo(key: "option_summary_norm") var optionSummaryNorm: LayerNorm
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "field_norm") var fieldNorm: LayerNorm
        @ModuleInfo(key: "option_norm") var optionNorm: LayerNorm
        @ModuleInfo(key: "scorer1") var scorer1: Linear
        @ModuleInfo(key: "scorer2") var scorer2: Linear
        @ParameterInfo(key: "prior_logit_scale") var priorLogitScale: MLXArray
        @ParameterInfo(key: "joint_logit_scale") var jointLogitScale: MLXArray
        @ParameterInfo(key: "residual_gate") var residualGate: MLXArray

        init(_ configuration: Configuration) {
            let (hidden, width) = (configuration.hiddenSize, configuration.width)
            let (heads, feedforward) = (configuration.heads, configuration.feedforward)
            _hiddenNorm.wrappedValue = LayerNorm(dimensions: hidden)
            _memoryProjection.wrappedValue = Linear(hidden, width, bias: false)
            _questionProjection.wrappedValue = Linear(hidden, width, bias: false)
            _optionQuestionProjection.wrappedValue = Linear(hidden, width, bias: false)
            _globalProjection.wrappedValue = Linear(hidden, width, bias: false)
            _optionContextProjection.wrappedValue = Linear(hidden, width, bias: false)
            _optionLexicalProjection.wrappedValue = Linear(hidden, width, bias: false)
            _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: width)
            _evidenceLayers.wrappedValue = (0 ..< configuration.routingLayers).map { _ in
                EvidenceRoutingLayer(width: width, heads: heads, feedforward: feedforward)
            }
            _optionSummaryNorm.wrappedValue = LayerNorm(dimensions: width)
            _layers.wrappedValue = (0 ..< configuration.layers).map { _ in
                DecoderLayer(width: width, heads: heads, feedforward: feedforward)
            }
            _fieldNorm.wrappedValue = LayerNorm(dimensions: width)
            _optionNorm.wrappedValue = LayerNorm(dimensions: width)
            _scorer1.wrappedValue = Linear(width * 4, width)
            _scorer2.wrappedValue = Linear(width, 1)
            _priorLogitScale.wrappedValue = MLXArray.zeros([])
            _jointLogitScale.wrappedValue = MLXArray.zeros([])
            _residualGate.wrappedValue = MLXArray.zeros([])
        }

        /// Renames PyTorch `nn.Sequential` keys to the names of this port's layers.
        static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
            Dictionary(
                uniqueKeysWithValues: weights.map { key, value in
                    let key =
                        key
                        .replacingOccurrences(of: ".feedforward.0.", with: ".feedforward.fc1.")
                        .replacingOccurrences(of: ".feedforward.3.", with: ".feedforward.fc2.")
                        .replacingOccurrences(of: "residual_scorer.0.", with: "scorer1.")
                        .replacingOccurrences(of: "residual_scorer.3.", with: "scorer2.")
                    return (key, value)
                }
            )
        }

        // MARK: - Forward pass

        /// Returns the option logits of each field.
        ///
        /// - Parameters:
        ///   - hidden: The final hidden states of one prompt, with shape `[length, hiddenSize]`.
        ///   - inputIDs: The token IDs of the prompt.
        ///   - fields: The positions of each field.
        ///   - lexical: A function that returns the output-embedding rows for token IDs.
        /// - Returns: For each field, one logit for each option, in head order.
        func callAsFunction(
            hidden: MLXArray,
            inputIDs: MLXArray,
            fields: [FieldSpans],
            lexical: (MLXArray) -> MLXArray
        ) -> [MLXArray] {
            let h = hiddenNorm(hidden)
            let memory = memoryProjection(h)[.newAxis]
            let globalVector = h[h.dim(0) - 1]
            let questionVectors = stacked(fields.map { h[$0.questionSpan].mean(axis: 0) })
            let typeIDs = MLXArray(fields.map { Int32($0.typeID) })

            let optionContexts = fields.map { field in
                stacked(field.optionSpans.map { h[$0].mean(axis: 0) })
            }
            let lexicalOptions = fields.map { field in
                stacked(field.optionSpans.map { lexical(inputIDs[$0]).mean(axis: 0) })
            }

            let optionQueries = fields.indices.map { index in
                optionContextProjection(optionContexts[index])
                    + optionLexicalProjection(lexicalOptions[index])
                    + optionQuestionProjection(questionVectors[index])[.newAxis]
            }
            var routed = concatenated(optionQueries, axis: 0)[.newAxis]
            for layer in evidenceLayers {
                routed = layer(routed, memory: memory)
            }
            routed = routed[0]
            var splitOptions: [MLXArray] = []
            var start = 0
            for field in fields {
                splitOptions.append(routed[start ..< start + field.optionSpans.count])
                start += field.optionSpans.count
            }

            let baseFields = questionProjection(questionVectors)
            let summaries = splitOptions.enumerated().map { index, options in
                let weights = softmax(
                    matmul(options, baseFields[index]) / Float(options.dim(-1)).squareRoot(),
                    axis: 0
                )
                return (weights[0..., .newAxis] * options).sum(axis: 0)
            }
            var fieldStates =
                (baseFields
                + optionSummaryNorm(stacked(summaries))
                + globalProjection(globalVector)[.newAxis]
                + typeEmbedding(typeIDs))[.newAxis]
            for layer in layers {
                fieldStates = layer(fieldStates, memory: memory)
            }
            fieldStates = fieldNorm(fieldStates[0])

            let maximumScale = Float(log(100.0))
            let priorScale = exp(minimum(priorLogitScale, maximumScale))
            let jointScale = exp(minimum(jointLogitScale, maximumScale))
            let gate = sigmoid(residualGate)
            return fields.indices.map { index in
                let anchor = l2Normalized(questionVectors[index] + globalVector)
                let prior = priorScale * matmul(l2Normalized(lexicalOptions[index]), anchor)
                let options = optionNorm(splitOptions[index])
                let field = broadcast(fieldStates[index][.newAxis], to: options.shape)
                let cosine =
                    (field * options).sum(axis: -1)
                    / maximum(
                        MLXLinalg.norm(field, axis: -1) * MLXLinalg.norm(options, axis: -1),
                        Float(1e-8)
                    )
                let features = concatenated([field, options, field * options, abs(field - options)], axis: -1)
                let residual = scorer2(gelu(scorer1(features)))[0..., 0]
                return prior + gate * (jointScale * cosine + residual)
            }
        }

        // MARK: - DecisionHead

        static let fileNames = ["joint_head.safetensors", "joint_head_config.json"]

        static var maximumPromptLength: Int { JointSchemaEncoding.maximumLength }

        static func load(from directory: URL) throws -> JointSchemaHead {
            let configuration = try JSONDecoder().decode(
                Configuration.self,
                from: Data(contentsOf: directory.appendingPathComponent("joint_head_config.json"))
            )
            let head = JointSchemaHead(configuration)
            let weights = try loadArrays(url: directory.appendingPathComponent("joint_head.safetensors"))
            try head.update(parameters: ModuleParameters.unflattened(sanitize(weights)), verify: .all)
            eval(head)
            return head
        }

        static func prefixTokens(for state: DecisionState, encode: (String) -> [Int]) -> [Int] {
            JointSchemaEncoding.prefixTokens(for: state, encode: encode)
        }

        static func prompt(
            for question: Question,
            encode: (String) -> [Int]
        ) -> (tokens: [Int], prompt: JointSchemaEncoding.Field) {
            let field = JointSchemaEncoding.field(for: question, encode: encode)
            return (field.tokens, field)
        }

        func logits(
            hidden: MLXArray,
            tokens: MLXArray,
            prefixLength: Int,
            prompt: JointSchemaEncoding.Field,
            outputEmbedding: OutputEmbedding
        ) -> [Float] {
            let spans = FieldSpans(
                typeID: prompt.typeID,
                questionSpan: prompt.questionSpan.shifted(by: prefixLength),
                optionSpans: prompt.optionSpans.map { $0.shifted(by: prefixLength) }
            )
            let headLogits = self(hidden: hidden, inputIDs: tokens, fields: [spans], lexical: outputEmbedding.rows)[0]
                .asType(.float32)
                .asArray(Float.self)
            var logits = [Float](repeating: 0, count: headLogits.count)
            for (position, answerIndex) in prompt.answerIndices.enumerated() {
                logits[answerIndex] = headLogits[position]
            }
            return logits
        }
    }

    /// Returns `x` divided by its L2 norm along the last axis, as `_l2norm` in the reference does.
    private func l2Normalized(_ x: MLXArray) -> MLXArray {
        x / maximum(MLXLinalg.norm(x, axis: -1, keepDims: true), Float(1e-12))
    }

    private extension Range where Bound == Int {
        func shifted(by offset: Int) -> Range<Int> {
            (lowerBound + offset) ..< (upperBound + offset)
        }
    }

    // MARK: - Layers

    extension JointSchemaHead {
        /// `torch.nn.MultiheadAttention` with packed input projections and `batch_first=True`.
        final class PackedMultiheadAttention: Module {
            let heads: Int

            @ParameterInfo(key: "in_proj_weight") var inProjectionWeight: MLXArray
            @ParameterInfo(key: "in_proj_bias") var inProjectionBias: MLXArray
            @ModuleInfo(key: "out_proj") var outProjection: Linear

            init(width: Int, heads: Int) {
                self.heads = heads
                _inProjectionWeight.wrappedValue = MLXArray.zeros([3 * width, width])
                _inProjectionBias.wrappedValue = MLXArray.zeros([3 * width])
                _outProjection.wrappedValue = Linear(width, width)
            }

            func callAsFunction(_ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray) -> MLXArray {
                let w = inProjectionWeight
                let b = inProjectionBias
                let d = w.dim(1)
                let q = matmul(queries, w[0 ..< d].T) + b[0 ..< d]
                let k = matmul(keys, w[d ..< 2 * d].T) + b[d ..< 2 * d]
                let v = matmul(values, w[(2 * d)...].T) + b[(2 * d)...]
                let (batch, queryLength, keyLength) = (q.dim(0), q.dim(1), k.dim(1))
                let headDimension = d / heads
                let output = MLXFast.scaledDotProductAttention(
                    queries: q.reshaped(batch, queryLength, heads, headDimension).transposed(0, 2, 1, 3),
                    keys: k.reshaped(batch, keyLength, heads, headDimension).transposed(0, 2, 1, 3),
                    values: v.reshaped(batch, keyLength, heads, headDimension).transposed(0, 2, 1, 3),
                    scale: Float(1 / Double(headDimension).squareRoot()),
                    mask: .none
                )
                return outProjection(output.transposed(0, 2, 1, 3).reshaped(batch, queryLength, d))
            }
        }

        /// Two linear layers with a GELU between them.
        final class FeedForward: Module, UnaryLayer {
            @ModuleInfo var fc1: Linear
            @ModuleInfo var fc2: Linear

            init(width: Int, feedforward: Int) {
                _fc1.wrappedValue = Linear(width, feedforward)
                _fc2.wrappedValue = Linear(feedforward, width)
            }

            func callAsFunction(_ x: MLXArray) -> MLXArray {
                fc2(gelu(fc1(x)))
            }
        }

        /// Lets option queries attend to the whole prompt.
        final class EvidenceRoutingLayer: Module {
            @ModuleInfo(key: "query_norm") var queryNorm: LayerNorm
            @ModuleInfo(key: "memory_norm") var memoryNorm: LayerNorm
            @ModuleInfo var attention: PackedMultiheadAttention
            @ModuleInfo(key: "feedforward_norm") var feedforwardNorm: LayerNorm
            @ModuleInfo var feedforward: FeedForward

            init(width: Int, heads: Int, feedforward: Int) {
                _queryNorm.wrappedValue = LayerNorm(dimensions: width)
                _memoryNorm.wrappedValue = LayerNorm(dimensions: width)
                _attention.wrappedValue = PackedMultiheadAttention(width: width, heads: heads)
                _feedforwardNorm.wrappedValue = LayerNorm(dimensions: width)
                _feedforward.wrappedValue = FeedForward(width: width, feedforward: feedforward)
            }

            func callAsFunction(_ queries: MLXArray, memory: MLXArray) -> MLXArray {
                let normalizedMemory = memoryNorm(memory)
                let routed = queries + attention(queryNorm(queries), normalizedMemory, normalizedMemory)
                return routed + feedforward(feedforwardNorm(routed))
            }
        }

        /// `torch.nn.TransformerDecoderLayer` with `norm_first=True` and a GELU activation.
        final class DecoderLayer: Module {
            @ModuleInfo(key: "self_attn") var selfAttention: PackedMultiheadAttention
            @ModuleInfo(key: "multihead_attn") var crossAttention: PackedMultiheadAttention
            @ModuleInfo var linear1: Linear
            @ModuleInfo var linear2: Linear
            @ModuleInfo var norm1: LayerNorm
            @ModuleInfo var norm2: LayerNorm
            @ModuleInfo var norm3: LayerNorm

            init(width: Int, heads: Int, feedforward: Int) {
                _selfAttention.wrappedValue = PackedMultiheadAttention(width: width, heads: heads)
                _crossAttention.wrappedValue = PackedMultiheadAttention(width: width, heads: heads)
                _linear1.wrappedValue = Linear(width, feedforward)
                _linear2.wrappedValue = Linear(feedforward, width)
                _norm1.wrappedValue = LayerNorm(dimensions: width)
                _norm2.wrappedValue = LayerNorm(dimensions: width)
                _norm3.wrappedValue = LayerNorm(dimensions: width)
            }

            func callAsFunction(_ x: MLXArray, memory: MLXArray) -> MLXArray {
                let h = norm1(x)
                var x = x + selfAttention(h, h, h)
                x = x + crossAttention(norm2(x), memory, memory)
                return x + linear2(gelu(linear1(norm3(x))))
            }
        }
    }
#endif
