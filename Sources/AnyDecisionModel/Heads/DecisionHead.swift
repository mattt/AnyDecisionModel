import Foundation

#if MLX
    import MLX
    import MLXNN

    /// A trained decision head and the prompt format that it reads.
    ///
    /// A backbone language model reads a prompt and returns its final hidden states.
    /// The head reads the hidden states at the positions that its format marks
    /// and returns one logit for each answer.
    /// ``MLXHeadDecisionModel`` runs the backbone; a conforming type does the rest.
    protocol DecisionHead: AnyObject, SendableMetatype {
        /// What the head reads from one question's prompt, such as the positions of its options.
        associatedtype Prompt: Sendable

        /// The files that the head reads from the model directory.
        ///
        /// The backbone loader skips these files.
        static var fileNames: [String] { get }

        /// The largest number of tokens in one prompt.
        static var maximumPromptLength: Int { get }

        /// Loads the head from a model directory.
        static func load(from directory: URL) throws -> Self

        /// Returns the tokens that every prompt about a state starts with.
        static func prefixTokens(for state: DecisionState, encode: (String) -> [Int]) -> [Int]

        /// Returns the tokens that follow the prefix in the prompt for a question,
        /// and what the head reads from them.
        static func prompt(for question: Question, encode: (String) -> [Int]) -> (tokens: [Int], prompt: Prompt)

        /// Returns one logit for each answer, in the question's answer order.
        ///
        /// - Parameters:
        ///   - hidden: The final hidden states of the whole prompt, with shape `[length, hiddenSize]`.
        ///   - tokens: The token IDs of the whole prompt.
        ///   - prefixLength: The number of prefix tokens before the question's tokens.
        ///   - prompt: What the head reads from the question's tokens.
        ///   - outputEmbedding: The backbone's output projection.
        func logits(
            hidden: MLXArray,
            tokens: MLXArray,
            prefixLength: Int,
            prompt: Prompt,
            outputEmbedding: OutputEmbedding
        ) -> [Float]
    }

    /// The rows of a backbone's output projection, the matrix that maps hidden states to token logits.
    ///
    /// Some heads read the rows for a prompt's tokens as a lexical representation of those tokens.
    struct OutputEmbedding: @unchecked Sendable {
        private let weight: MLXArray
        private let quantization:
            (scales: MLXArray, biases: MLXArray?, groupSize: Int, bits: Int, mode: QuantizationMode)?

        /// Creates an output embedding from a projection layer.
        ///
        /// - Parameter projection: A `Linear` or `QuantizedLinear` layer.
        init(_ projection: Linear) {
            weight = projection.weight
            if let quantized = projection as? QuantizedLinear {
                quantization = (quantized.scales, quantized.biases, quantized.groupSize, quantized.bits, quantized.mode)
            } else {
                quantization = nil
            }
        }

        /// Returns the rows for token IDs, dequantized if the projection is quantized.
        func rows(_ ids: MLXArray) -> MLXArray {
            guard let quantization else { return weight[ids] }
            return dequantized(
                weight[ids],
                scales: quantization.scales[ids],
                biases: quantization.biases?[ids],
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                mode: quantization.mode
            )
        }
    }
#endif
