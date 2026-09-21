import Foundation

/// Builds prompts and answers for models that read decisions from next-token probabilities.
///
/// The model sees the state, the question, and a list of allowed answer labels.
/// It runs one forward pass, and the probability of each label's tokens
/// at the last position becomes the probability of that answer.
/// The helpers here do not depend on MLX, so they can be tested with synthetic logits.
enum TokenScoring {
    /// The labels for choice options, in option order.
    static let choiceLabels = (UnicodeScalar("A").value ... UnicodeScalar("Z").value)
        .map { String(UnicodeScalar($0)!) }

    /// The labels for score levels, in level order.
    static let scoreLabels = (0 ... 9).map(String.init)

    /// The labels for binary answers: "yes", then "no".
    static let binaryLabels = ["yes", "no"]

    /// The default system prompt for local decision models.
    static let defaultSystemPrompt =
        "You are a decision function. Read the state and the question, "
        + "then reply with exactly one token from the allowed answers. "
        + "If the state does not determine the answer, stay uncertain."

    /// Returns the labels that a question's answers are read from.
    ///
    /// - Parameter question: The question.
    /// - Returns: One label for each answer, in answer order.
    static func labels(for question: Question) -> [String] {
        switch question {
        case .binary:
            return binaryLabels
        case .choice(_, let options):
            return Array(choiceLabels.prefix(options.count))
        case .score(_, let levels):
            return Array(scoreLabels.prefix(levels.count))
        }
    }

    /// Returns the user message for a question about a state.
    ///
    /// The state comes first, so that every question about the same state
    /// shares a prompt prefix.
    ///
    /// - Parameters:
    ///   - state: The state.
    ///   - question: The question.
    ///   - optionOrder: For choice questions, the order in which to list the options,
    ///     as indices into the question's options. Pass `nil` for the question's order.
    static func userMessage(
        state: DecisionState,
        question: Question,
        optionOrder: [Int]? = nil
    ) -> String {
        "State:\n\(state.jsonValue.promptText)\n\n\(questionText(question, optionOrder: optionOrder))"
    }

    static func questionText(_ question: Question, optionOrder: [Int]?) -> String {
        var lines = ["Question: \(question.instructions)"]
        switch question {
        case .binary(_, let criteria):
            if let criteria {
                lines.append("Answer yes if: \(criteria.whenTrue)")
                lines.append("Answer no if: \(criteria.whenFalse)")
            }
        case .choice(_, let options):
            let order = optionOrder ?? Array(options.indices)
            for (position, index) in order.enumerated() {
                let option = options[index]
                var line = "\(choiceLabels[position]). \(option.name)"
                if let description = option.description {
                    line += ": \(description)"
                }
                lines.append(line)
            }
        case .score(_, let levels):
            for (index, level) in levels.enumerated() {
                lines.append("\(index). \(level)")
            }
        }
        let labels = labels(for: question)
        return lines.joined(separator: "\n") + "\n\nAllowed answers: " + labels.joined(separator: ", ")
    }

    /// Appends a closed thinking block when a chat template ignores the thinking flag.
    ///
    /// - Parameters:
    ///   - tokens: The rendered prompt tokens.
    ///   - closedThinkTokens: The tokens for `<think>\n\n</think>\n\n`,
    ///     or an empty array if the tokenizer has no thinking markers.
    ///   - decode: A function that decodes tokens with special tokens kept.
    static func appendingClosedThinkBlock(
        to tokens: [Int],
        closedThinkTokens: [Int],
        decode: ([Int]) -> String
    ) -> [Int] {
        guard !closedThinkTokens.isEmpty else { return tokens }

        // A template can encode the markers as several tokens.
        // Allow four extra tokens and check the decoded text.
        let tail = Array(tokens.suffix(closedThinkTokens.count + 4))
        guard !decode(tail).contains("</think>") else { return tokens }
        return tokens + closedThinkTokens
    }

    /// Returns the single-token variants of an answer label.
    ///
    /// Variants cover case changes and an optional leading space.
    ///
    /// - Parameters:
    ///   - label: The answer label.
    ///   - encode: A function that tokenizes text without special tokens.
    /// - Returns: The sorted token IDs of every variant that encodes to exactly one token.
    static func singleTokenVariants(of label: String, encode: (String) -> [Int]) -> [Int] {
        var ids = Set<Int>()
        let variants: Set<String> = [label, label.lowercased(), label.uppercased(), label.capitalized]
        for variant in variants {
            for prefix in ["", " "] {
                let encoded = encode(prefix + variant)
                if encoded.count == 1 {
                    ids.insert(encoded[0])
                }
            }
        }
        return ids.sorted()
    }

    /// Maps answer labels to token IDs and checks that the mapping is usable.
    ///
    /// - Parameters:
    ///   - labels: The answer labels.
    ///   - index: The position of the question in its batch, used in the error.
    ///   - encode: A function that tokenizes text without special tokens.
    /// - Returns: The token IDs for each label.
    /// - Throws: ``DecisionError/unsupportedQuestion(index:reason:)``
    ///   if a label has no single-token variant or two labels share a token.
    static func tokenMap(for labels: [String], index: Int, encode: (String) -> [Int]) throws -> [[Int]] {
        var owner: [Int: String] = [:]
        var map: [[Int]] = []
        for label in labels {
            let ids = singleTokenVariants(of: label, encode: encode)
            guard !ids.isEmpty else {
                throw DecisionError.unsupportedQuestion(
                    index: index,
                    reason: "The answer label \"\(label)\" has no single-token encoding."
                )
            }
            for token in ids {
                if let other = owner[token] {
                    throw DecisionError.unsupportedQuestion(
                        index: index,
                        reason: "The answer labels \"\(other)\" and \"\(label)\" share token \(token)."
                    )
                }
                owner[token] = label
            }
            map.append(ids)
        }
        return map
    }

    /// Sums the probability of each answer's tokens.
    ///
    /// - Parameter tokenLogProbabilities: For each answer,
    ///   the log-probabilities of its tokens under the full next-token distribution.
    /// - Returns: The log-mass of each answer,
    ///   and the total probability of all allowed answer tokens.
    static func logMasses(_ tokenLogProbabilities: [[Double]]) -> (
        logMasses: [Double], allowedAnswerMass: Double
    ) {
        let masses = tokenLogProbabilities.map { $0.logSumExp() }
        let coverage = exp(masses.logSumExp())
        return (masses, min(max(coverage, 0), 1))
    }

    /// Builds an answer from calibrated answer probabilities.
    ///
    /// - Parameters:
    ///   - question: The question.
    ///   - probabilities: The probability of each answer, in the question's answer order.
    static func answer(for question: Question, probabilities: [Double]) -> Answer {
        switch question {
        case .binary:
            return .binary(probability: probabilities[0])
        case .choice(_, let options):
            let best = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
            var byName: [String: Double] = [:]
            for (option, probability) in zip(options, probabilities) {
                byName[option.name] = probability
            }
            return .choice(
                name: options[best].name,
                probabilities: byName,
                confidence: concentration(probabilities)
            )
        case .score(_, let levels):
            return .score(
                value: expectedLevel(probabilities),
                probabilities: probabilities,
                legend: levels,
                confidence: concentration(probabilities)
            )
        }
    }

    /// Returns the normalized entropy complement, `1 - H(p) / \log n`.
    ///
    /// The result is 1 for a one-hot distribution and 0 for a uniform one.
    /// A single-outcome distribution has confidence 1.
    /// This measures concentration, not the probability of a correct answer.
    static func concentration(_ probabilities: [Double]) -> Double {
        guard probabilities.count > 1 else { return 1 }
        let entropy = probabilities.reduce(0) { total, p in
            p > 0 ? total - p * log(p) : total
        }
        let value = 1 - entropy / log(Double(probabilities.count))
        return min(max(value, 0), 1)
    }

    /// Returns the expected level index, `\sum_i i \cdot p_i`.
    static func expectedLevel(_ probabilities: [Double]) -> Double {
        probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }
    }
}
