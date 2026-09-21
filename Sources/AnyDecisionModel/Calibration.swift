import Foundation

/// A post-hoc calibration for local decision probabilities.
///
/// Raw probabilities from a language model are often overconfident.
/// A temperature above 1 softens every primitive.
/// The bias shifts binary log-odds only, because choice and score options
/// have no shared direction to shift.
///
/// For a binary question, the calibrated probability is
/// `\sigma((\log p_{yes} - \log p_{no}) / T + b)`,
/// where `T` is the temperature and `b` is the binary bias.
/// For choice and score questions, it is `\operatorname{softmax}(\log p / T)`.
///
/// Fit calibration on data that is separate from the data you evaluate.
public struct Calibration: Hashable, Sendable, Codable {
    /// The temperature applied to log-probabilities. Must be greater than 0.
    public var temperature: Double

    /// The bias added to binary log-odds after the temperature.
    public var binaryBias: Double

    /// Creates a calibration.
    ///
    /// - Parameters:
    ///   - temperature: The temperature applied to log-probabilities. Must be greater than 0.
    ///   - binaryBias: The bias added to binary log-odds.
    public init(temperature: Double, binaryBias: Double = 0) {
        precondition(temperature > 0, "The temperature must be greater than 0.")
        self.temperature = temperature
        self.binaryBias = binaryBias
    }

    /// The calibration that leaves probabilities unchanged.
    public static let identity = Calibration(temperature: 1, binaryBias: 0)

    /// Returns calibrated probabilities for log-masses of allowed answers.
    ///
    /// - Parameters:
    ///   - logMasses: The unnormalized log-probability of each allowed answer.
    ///   - binary: Whether the answers are "yes" and "no", in that order.
    public func probabilities(logMasses: [Double], binary: Bool) -> [Double] {
        if binary {
            precondition(logMasses.count == 2, "Binary calibration needs two log-masses.")
            let logOdds = (logMasses[0] - logMasses[1]) / temperature + binaryBias
            let yes = sigmoid(logOdds)
            return [yes, 1 - yes]
        }
        return logMasses.map { $0 / temperature }.softmax()
    }
}

// MARK: - Numerical helpers

/// Returns `1 / (1 + e^{-x})`, stable for large magnitudes.
private func sigmoid(_ x: Double) -> Double {
    if x >= 0 {
        return 1 / (1 + exp(-x))
    }
    let e = exp(x)
    return e / (1 + e)
}

private extension Collection where Element == Double {
    /// Returns probabilities proportional to `e^{x_i}`.
    func softmax() -> [Double] {
        let normalizer = logSumExp()
        return map { exp($0 - normalizer) }
    }
}
