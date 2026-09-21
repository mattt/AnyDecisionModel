import Foundation

extension Collection where Element == Double {
    /// Returns `\log \sum_i e^{x_i}` without overflow.
    func logSumExp() -> Double {
        guard let maximum = self.max(), maximum.isFinite else {
            return self.max() ?? -.infinity
        }
        let total = reduce(0) { $0 + exp($1 - maximum) }
        return maximum + log(total)
    }
}
