import Foundation
import Testing

@testable import AnyDecisionModel

/// A tokenizer for tests: each known word is one token, and anything else is split per character.
struct FakeTokenizer {
    var vocabulary: [String: Int]

    func encode(_ text: String) -> [Int] {
        if let id = vocabulary[text] { return [id] }
        return text.unicodeScalars.map { 1000 + Int($0.value) }
    }
}

/// A reference sigmoid for expected values.
private func sigmoid(_ x: Double) -> Double {
    1 / (1 + exp(-x))
}

@Suite("Numerical helpers")
struct NumericalHelperTests {
    @Test func logSumExpIsStable() {
        #expect(abs([1000.0, 1000].logSumExp() - (1000 + log(2))) < 1e-9)
        #expect([-Double.infinity, -.infinity].logSumExp() == -.infinity)
        #expect([Double]().logSumExp() == -.infinity)
    }

    @Test func concentrationIsOneForCertaintyAndZeroForUniform() {
        #expect(TokenScoring.concentration([1, 0, 0]) == 1)
        #expect(abs(TokenScoring.concentration([0.25, 0.25, 0.25, 0.25])) < 1e-12)
        #expect(TokenScoring.concentration([1]) == 1)
        let skewed = TokenScoring.concentration([0.9, 0.1])
        let expected = 1 - (-(0.9 * log(0.9) + 0.1 * log(0.1)) / log(2))
        #expect(abs(skewed - expected) < 1e-12)
    }

    @Test func expectedLevel() {
        #expect(abs(TokenScoring.expectedLevel([0, 0.32, 0.68]) - 1.68) < 1e-12)
    }
}

@Suite("Calibration")
struct CalibrationTests {
    @Test func identityLeavesProbabilitiesUnchanged() {
        let masses = [log(0.2), log(0.3)]
        let p = Calibration.identity.probabilities(logMasses: masses, binary: true)
        #expect(abs(p[0] - 0.4) < 1e-12)
        let q = Calibration.identity.probabilities(logMasses: masses, binary: false)
        #expect(abs(q[0] - 0.4) < 1e-12)
    }

    @Test func identityNormalizesWithSoftmax() {
        let p = Calibration.identity.probabilities(logMasses: [log(1), log(3)], binary: false)
        #expect(abs(p[0] - 0.25) < 1e-12)
        #expect(abs(p[1] - 0.75) < 1e-12)

        // Large log-masses do not overflow.
        let large = Calibration.identity.probabilities(logMasses: [1000, 1000], binary: false)
        #expect(abs(large[0] - 0.5) < 1e-12)
        #expect(abs(large[1] - 0.5) < 1e-12)
    }

    @Test func binaryCalibrationIsStableForLargeLogOdds() {
        let p = Calibration.identity.probabilities(logMasses: [0, -2000], binary: true)
        #expect(p == [1, 0])
        let q = Calibration.identity.probabilities(logMasses: [-2000, 0], binary: true)
        #expect(q == [0, 1])
    }

    @Test func temperatureSoftensAllPrimitives() {
        let calibration = Calibration(temperature: 4)
        let binary = calibration.probabilities(logMasses: [0, -8], binary: true)
        #expect(abs(binary[0] - sigmoid(2)) < 1e-12)
        let choice = calibration.probabilities(logMasses: [0, -8, -8], binary: false)
        let total = 1 + 2 * exp(-2.0)
        let expected = [1 / total, exp(-2.0) / total, exp(-2.0) / total]
        for (a, b) in zip(choice, expected) {
            #expect(abs(a - b) < 1e-12)
        }
    }

    @Test func biasAppliesToBinaryLogOddsOnly() {
        let calibration = Calibration(temperature: 1, binaryBias: 1)
        let binary = calibration.probabilities(logMasses: [0, 0], binary: true)
        #expect(abs(binary[0] - sigmoid(1)) < 1e-12)
        let choice = calibration.probabilities(logMasses: [0, 0], binary: false)
        #expect(choice == [0.5, 0.5])
    }

    @Test func matchesLogitCalibrationFormula() {
        // The expected value is `\sigma(\operatorname{logit}(p) / T + b)` with `T = 9.25` and `b = 0.5`.
        let p = 0.999
        let calibration = Calibration(temperature: 9.25, binaryBias: 0.5)
        let result = calibration.probabilities(logMasses: [log(p), log(1 - p)], binary: true)[0]
        let expected = sigmoid(log(p / (1 - p)) / 9.25 + 0.5)
        #expect(abs(result - expected) < 1e-12)
    }
}

@Suite("Token scoring")
struct TokenScoringTests {
    let tokenizer = FakeTokenizer(vocabulary: [
        "yes": 1, " yes": 2, "Yes": 3, " Yes": 4, "YES": 5,
        "no": 6, " no": 7, "No": 8,
        "A": 10, " A": 11, "B": 12, " B": 13, "C": 14,
        "0": 20, "1": 21, "2": 22,
    ])

    @Test func singleTokenVariantsCoverCaseAndLeadingSpace() {
        let ids = TokenScoring.singleTokenVariants(of: "yes", encode: tokenizer.encode)
        #expect(ids == [1, 2, 3, 4, 5])
    }

    @Test func tokenMapRejectsLabelWithoutSingleToken() {
        #expect(throws: DecisionError.self) {
            try TokenScoring.tokenMap(for: ["A", "maybe"], index: 0, encode: tokenizer.encode)
        }
    }

    @Test func tokenMapRejectsSharedTokens() {
        let shared = FakeTokenizer(vocabulary: ["A": 1, "B": 1])
        #expect(throws: DecisionError.self) {
            try TokenScoring.tokenMap(for: ["A", "B"], index: 0, encode: shared.encode)
        }
    }

    @Test func labelsMatchQuestionShape() {
        #expect(TokenScoring.labels(for: .binary(instructions: "x")) == ["yes", "no"])
        let choice = Question.choice(
            instructions: "x",
            options: [ChoiceOption("a"), ChoiceOption("b"), ChoiceOption("c")]
        )
        #expect(TokenScoring.labels(for: choice) == ["A", "B", "C"])
        #expect(TokenScoring.labels(for: .score(instructions: "x", levels: ["l", "m", "h"])) == ["0", "1", "2"])
        #expect(TokenScoring.choiceLabels.count == 26)
        #expect(TokenScoring.scoreLabels.count == 10)
    }

    @Test func logMassesSumVariantsAndReportCoverage() {
        // A synthetic vocabulary of 4 tokens: yes (2 variants), no (1 variant), other.
        let logits: [Double] = [2, 1, 0, 1]
        let normalizer = logits.logSumExp()
        let lp = logits.map { $0 - normalizer }
        let (masses, coverage) = TokenScoring.logMasses([[lp[0], lp[1]], [lp[2]]])

        let total = logits.map(exp).reduce(0, +)
        #expect(abs(exp(masses[0]) - (exp(2) + exp(1)) / total) < 1e-12)
        #expect(abs(exp(masses[1]) - 1 / total) < 1e-12)
        #expect(abs(coverage - (exp(2) + exp(1) + 1) / total) < 1e-12)

        // Normalizing across allowed answers removes the other token's mass.
        let p = Calibration.identity.probabilities(logMasses: masses, binary: true)
        #expect(abs(p[0] - (exp(2) + exp(1)) / (exp(2) + exp(1) + 1)) < 1e-12)
    }

    @Test func coverageIsNotConfidence() {
        // All allowed-answer mass on one answer, but most of the total mass elsewhere.
        let (masses, coverage) = TokenScoring.logMasses([[log(0.05)], [log(1e-12)]])
        #expect(abs(coverage - 0.05) < 1e-9)
        let answer = TokenScoring.answer(
            for: .choice(instructions: "x", options: [ChoiceOption("a"), ChoiceOption("b")]),
            probabilities: Calibration.identity.probabilities(logMasses: masses, binary: false)
        )
        guard case .choice(_, _, let confidence) = answer else {
            Issue.record("Expected a choice answer.")
            return
        }
        #expect(confidence > 0.99)
    }

    @Test func answersUseLocalConfidence() {
        let choice = TokenScoring.answer(
            for: .choice(instructions: "x", options: [ChoiceOption("heads"), ChoiceOption("tails")]),
            probabilities: [0.5, 0.5]
        )
        #expect(choice == .choice(name: "heads", probabilities: ["heads": 0.5, "tails": 0.5], confidence: 0))

        let single = TokenScoring.answer(
            for: .choice(instructions: "x", options: [ChoiceOption("only")]),
            probabilities: [1]
        )
        #expect(single == .choice(name: "only", probabilities: ["only": 1], confidence: 1))

        let score = TokenScoring.answer(
            for: .score(instructions: "x", levels: ["Calm", "Frustrated", "Very angry"]),
            probabilities: [0, 0.32, 0.68]
        )
        guard case .score(let value, _, let legend, _) = score else {
            Issue.record("Expected a score answer.")
            return
        }
        #expect(abs(value - 1.68) < 1e-12)
        #expect(legend == ["Calm", "Frustrated", "Very angry"])
    }

    @Test func promptPutsStateBeforeQuestion() {
        let state = DecisionState.text("I flipped a coin.")
        let a = TokenScoring.userMessage(state: state, question: .binary(instructions: "Heads?"))
        let b = TokenScoring.userMessage(
            state: state,
            question: .choice(instructions: "Which side?", options: [ChoiceOption("heads"), ChoiceOption("tails")])
        )
        let sharedPrefix = zip(a, b).prefix { $0 == $1 }.count
        #expect(a.prefix(sharedPrefix).contains("I flipped a coin."))
        #expect(a.hasSuffix("Allowed answers: yes, no"))
        #expect(b.contains("A. heads\nB. tails"))
    }

    @Test func promptRendersOptionRotationAndCriteria() {
        let question = Question.choice(
            instructions: "Route.",
            options: [
                ChoiceOption("billing", description: "payments"), ChoiceOption("sales"), ChoiceOption("technical"),
            ]
        )
        let rotated = TokenScoring.userMessage(state: "s", question: question, optionOrder: [1, 2, 0])
        #expect(rotated.contains("A. sales\nB. technical\nC. billing: payments"))

        let binary = TokenScoring.userMessage(
            state: "s",
            question: .binary(instructions: "Urgent?", criteria: BinaryCriteria(whenTrue: "today", whenFalse: "later"))
        )
        #expect(binary.contains("Answer yes if: today\nAnswer no if: later"))
    }

    @Test func structuredValuesRenderDeterministically() {
        let value: JSONValue = ["b": 1, "a": ["x", true]]
        #expect(value.promptText == #"{"a":["x",true],"b":1}"#)
    }
}
