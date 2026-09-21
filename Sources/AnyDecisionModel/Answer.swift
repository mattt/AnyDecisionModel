import Foundation

/// The answer to one ``Question``.
///
/// The confidence of a choice or score answer
/// tells you how concentrated the distribution is, from 0 to 1.
/// Jev computes this value remotely and does not publish the formula.
/// Local models report `1 - H(p) / \log n`, the normalized entropy complement.
/// This measures concentration, not the probability that the answer is correct.
/// Binary answers have no confidence value.
public enum Answer: Hashable, Sendable {
    /// The answer to a binary question.
    ///
    /// - Parameter probability: The probability that the answer is "yes", from 0 to 1.
    case binary(probability: Double)

    /// The answer to a choice question.
    ///
    /// - Parameters:
    ///   - name: The name of the most probable option.
    ///   - probabilities: The probability of each option, keyed by option name.
    ///   - confidence: How concentrated the distribution is, from 0 to 1.
    case choice(name: String, probabilities: [String: Double], confidence: Double)

    /// The answer to a score question.
    ///
    /// - Parameters:
    ///   - value: The expected level index under `probabilities`.
    ///   - probabilities: The probability of each level, indexed from the lowest level.
    ///   - legend: The level descriptions, if the backend returns them.
    ///   - confidence: How concentrated the distribution is, from 0 to 1.
    case score(value: Double, probabilities: [Double], legend: [String]? = nil, confidence: Double)
}

// MARK: - Validation

extension Answer {
    func validate(for question: Question, index: Int, tolerance: Double) throws {
        func fail(_ reason: String) -> DecisionError {
            .invalidResponse("Question at index \(index): \(reason)")
        }
        func checkProbability(_ value: Double, _ name: String) throws {
            guard value.isFinite, value >= -tolerance, value <= 1 + tolerance else {
                throw fail("\(name) \(value) is not a probability.")
            }
        }
        func checkTotal(_ values: some Collection<Double>) throws {
            let total = values.reduce(0, +)
            guard abs(total - 1) <= tolerance else {
                throw fail("The probabilities sum to \(total), not 1.")
            }
        }

        switch (question, self) {
        case (.binary, .binary(let probability)):
            try checkProbability(probability, "binary")

        case (.choice(_, let options), .choice(let name, let probabilities, let confidence)):
            let expected = Set(options.map(\.name))
            guard Set(probabilities.keys) == expected else {
                throw fail("The probabilities do not match the options.")
            }
            guard expected.contains(name) else {
                throw fail("The choice \"\(name)\" is not an option.")
            }
            for (option, value) in probabilities {
                try checkProbability(value, "The probability of \"\(option)\"")
            }
            try checkTotal(probabilities.values)
            try checkProbability(confidence, "The confidence")

        case (.score(_, let levels), .score(let value, let probabilities, _, let confidence)):
            guard probabilities.count == levels.count else {
                throw fail("Expected \(levels.count) level probabilities, got \(probabilities.count).")
            }
            for (level, probability) in probabilities.enumerated() {
                try checkProbability(probability, "The probability of level \(level)")
            }
            try checkTotal(probabilities)
            guard value.isFinite,
                value >= -tolerance,
                value <= Double(levels.count - 1) + tolerance
            else {
                throw fail("The score \(value) is outside the scale.")
            }
            try checkProbability(confidence, "The confidence")

        case (.binary, _):
            throw fail("Expected a binary answer.")
        case (.choice, _):
            throw fail("Expected a choice answer.")
        case (.score, _):
            throw fail("Expected a score answer.")
        }
    }
}

// MARK: - Choice

/// The result of a typed choice.
public struct Choice<Option: Hashable & Sendable>: Hashable, Sendable {
    /// The most probable option.
    public var value: Option

    /// The options with their probabilities, in the order of the question.
    public var distribution: [Outcome]

    /// How concentrated the distribution is, from 0 to 1.
    ///
    /// This measures concentration, not the probability that ``value`` is correct.
    public var confidence: Double

    /// An option with its probability.
    public struct Outcome: Hashable, Sendable {
        /// The option.
        public var option: Option

        /// The probability of the option.
        public var probability: Double
    }

    /// Returns the probability of an option, or 0 if it is not in the distribution.
    public func probability(of option: Option) -> Double {
        distribution.first { $0.option == option }?.probability ?? 0
    }

    /// Creates a typed choice from the values of a choice answer.
    ///
    /// - Parameters:
    ///   - name: The name of the most probable option.
    ///   - probabilities: The probability of each option, keyed by option name.
    ///   - confidence: How concentrated the distribution is.
    ///   - options: The options with their names, in the order of the question.
    init(
        name: String,
        probabilities: [String: Double],
        confidence: Double,
        options: [(Option, String)]
    ) throws {
        var distribution: [Outcome] = []
        var value: Option?
        for (option, optionName) in options {
            guard let probability = probabilities[optionName] else {
                throw DecisionError.invalidResponse("No probability for option \"\(optionName)\".")
            }
            distribution.append(Outcome(option: option, probability: probability))
            if optionName == name {
                value = option
            }
        }
        guard let value else {
            throw DecisionError.invalidResponse("The choice \"\(name)\" is not an option.")
        }
        self.value = value
        self.distribution = distribution
        self.confidence = confidence
    }
}

// MARK: - Score

/// The result of a score question.
public struct Score: Hashable, Sendable {
    /// The expected level index, from 0 to `levels.count - 1`.
    public var value: Double

    /// The probability of each level, indexed from the lowest level.
    public var probabilities: [Double]

    /// The level descriptions.
    public var levels: [String]

    /// How concentrated the distribution is, from 0 to 1.
    ///
    /// This measures concentration, not the probability that the score is correct.
    public var confidence: Double

    /// The index of the most probable level.
    public var mostLikelyLevelIndex: Int {
        probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
    }
}
