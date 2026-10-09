import Foundation

#if MLX
    import MLX
    import MLXLMCommon

    /// A key-value cache that holds a batch of sequences with a shared length.
    ///
    /// It masks attention with an explicit causal mask array.
    /// The symbolic causal mask in MLX 0.31.1 is wrong for some shapes
    /// in which the queries follow cached keys.
    final class BatchKVCache: KVCache {
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
