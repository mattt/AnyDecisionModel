import Foundation
import Testing

@testable import AnyDecisionModel

/// The kind of distribution under test.
enum DistributionKind: String, CaseIterable, Sendable, CustomTestStringConvertible {
    case choice, score

    var testDescription: String { rawValue }

    /// Returns a question with `count` options or levels.
    func question(count: Int) -> Question {
        switch self {
        case .choice:
            .choice(instructions: "Which?", options: (0 ..< count).map { ChoiceOption("o\($0)") })
        case .score:
            .score(instructions: "How much?", levels: (0 ..< count).map { "l\($0)" })
        }
    }

    /// Returns an answer with the given probabilities, in option or level order.
    func answer(_ probabilities: [Double], confidence: Double = 0.5) -> Answer {
        let best = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
        switch self {
        case .choice:
            let keyed = Dictionary(uniqueKeysWithValues: probabilities.enumerated().map { ("o\($0)", $1) })
            return .choice(name: "o\(best)", probabilities: keyed, confidence: confidence)
        case .score:
            return .score(value: Double(best), probabilities: probabilities, confidence: confidence)
        }
    }

    /// Validates the probabilities as the only answer to a question of this kind.
    func validate(_ probabilities: [Double], confidence: Double = 0.5) throws {
        let response = DecisionSession.Response(
            modelID: "m",
            answers: [answer(probabilities, confidence: confidence)]
        )
        try response.validate(against: [question(count: probabilities.count)])
    }
}

/// A SplitMix64 generator, so that random distributions are the same on every run.
struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// A distribution with a rounding of it to a fixed number of decimal places.
struct RoundedDistribution {
    /// The exact distribution, which sums to 1.
    var exact: [Double]

    /// The rounded values.
    var rounded: [Double]

    /// The direction in which ties round in a worst case.
    enum Direction: String, Sendable {
        case up, down
    }

    /// Rounds a random distribution with `count` values to the nearest multiple of 10^-`decimals`.
    static func random(count: Int, decimals: Int, seed: UInt64) -> Self {
        var generator = SeededGenerator(state: seed)
        // An exponential draw for each value gives a uniform Dirichlet distribution.
        let draws = (0 ..< count).map { _ in -log(Double.random(in: .ulpOfOne ..< 1, using: &generator)) }
        let total = draws.reduce(0, +)
        let exact = draws.map { $0 / total }
        let scale = pow(10, Double(decimals))
        return Self(exact: exact, rounded: exact.map { ($0 * scale).rounded() / scale })
    }

    /// Builds a distribution whose rounding moves the total as far from 1 as possible.
    ///
    /// Each value that is not a multiple of the unit u = 10^-`decimals`
    /// lies halfway between two multiples, (k + ½)u,
    /// so rounding moves it by u/2 in `direction`.
    /// The values are nonnegative and sum to 1,
    /// so at most 2 / u of them can be halfway values.
    /// The rounded values are computed in integer units,
    /// so the tie direction does not depend on binary floating point.
    static func worstCase(count: Int, decimals: Int, direction: Direction) -> Self {
        var units = 1
        for _ in 0 ..< decimals { units *= 10 }
        let halfwayCount = min(count - count % 2, 2 * units)
        // The whole units left after the halfway values take ½ unit each.
        var remaining = units - halfwayCount / 2
        var wholeUnits = [Int](repeating: 0, count: count)
        var index = 0
        while remaining > 0 {
            wholeUnits[index % count] += 1
            remaining -= 1
            index += 1
        }
        var exact: [Double] = []
        var rounded: [Double] = []
        for (index, whole) in wholeUnits.enumerated() {
            let unit = Double(units)
            if index < halfwayCount {
                exact.append((Double(whole) + 0.5) / unit)
                rounded.append(Double(direction == .up ? whole + 1 : whole) / unit)
            } else {
                exact.append(Double(whole) / unit)
                rounded.append(Double(whole) / unit)
            }
        }
        return Self(exact: exact, rounded: rounded)
    }

    /// Checks that `rounded` is a rounding of `exact` and that `exact` is a distribution.
    func checkIsRounding(decimals: Int, sourceLocation: SourceLocation = #_sourceLocation) {
        let halfUnit = 0.5 * pow(10, -Double(decimals))
        #expect(abs(exact.reduce(0, +) - 1) < 1e-12, sourceLocation: sourceLocation)
        for (x, r) in zip(exact, rounded) {
            #expect(x >= 0 && x <= 1, sourceLocation: sourceLocation)
            #expect(abs(x - r) <= halfUnit + 1e-12, sourceLocation: sourceLocation)
        }
    }
}

@Suite("Probability totals")
struct ProbabilityTotalTests {
    // MARK: Google's example

    /// 0.03 + 0.83 + 0.15 is 0.010000000000000009 above 1 in `Double`.
    @Test func acceptsRoundedScoreExample() throws {
        let answer = Answer.score(value: 1.12, probabilities: [0.03, 0.83, 0.15], confidence: 0.6)
        let question = Question.score(instructions: "How urgent?", levels: ["Low", "Medium", "High"])
        try DecisionSession.Response(modelID: "m", answers: [answer]).validate(against: [question])
    }

    @Test func acceptsRoundedChoiceExample() throws {
        let answer = Answer.choice(
            name: "bug",
            probabilities: ["billing": 0.03, "bug": 0.83, "account": 0.15],
            confidence: 0.6
        )
        let question = Question.choice(
            instructions: "Which label?",
            options: [ChoiceOption("billing"), ChoiceOption("bug"), ChoiceOption("account")]
        )
        try DecisionSession.Response(modelID: "m", answers: [answer]).validate(against: [question])
    }

    // MARK: Rounded distributions

    /// Rounding to two or three decimal places moves the total of up to 26 values
    /// by at most 26 × 0.005 = 0.13, so every such rounding passes.
    @Test(arguments: DistributionKind.allCases, [(2, 2), (2, 3), (3, 2), (3, 3), (10, 2), (10, 3), (26, 2), (26, 3)])
    func acceptsRandomRounding(kind: DistributionKind, shape: (count: Int, decimals: Int)) throws {
        for seed in UInt64(1) ... 50 {
            let distribution = RoundedDistribution.random(count: shape.count, decimals: shape.decimals, seed: seed)
            distribution.checkIsRounding(decimals: shape.decimals)
            #expect(throws: Never.self, "seed \(seed): \(distribution.rounded)") {
                try kind.validate(distribution.rounded)
            }
        }
    }

    /// Rounding to three decimal places moves the total of 100 values by at most 0.05.
    @Test(arguments: DistributionKind.allCases)
    func acceptsRandomThreeDecimalRoundingOfManyValues(kind: DistributionKind) throws {
        for seed in UInt64(1) ... 50 {
            let distribution = RoundedDistribution.random(count: 100, decimals: 3, seed: seed)
            distribution.checkIsRounding(decimals: 3)
            #expect(throws: Never.self, "seed \(seed)") { try kind.validate(distribution.rounded) }
        }
    }

    struct WorstCase: Sendable, CustomTestStringConvertible {
        var count: Int
        var decimals: Int
        var direction: RoundedDistribution.Direction
        var accepted: Bool

        var testDescription: String {
            "\(count) values, \(decimals) decimals, \(direction.rawValue): \(accepted ? "accepted" : "rejected")"
        }
    }

    /// The largest number of choice options that a local backend allows.
    static let maximumLocalCount = TokenScoring.choiceLabels.count

    /// Worst-case roundings move the total by min(n, 2 / u) × u/2.
    ///
    /// Two-decimal and three-decimal roundings pass up to 26 values,
    /// and three-decimal roundings pass for 100 values.
    /// One-decimal roundings move the total by at least 0.1, so they fail.
    /// Two-decimal roundings of 100 values move it by 0.5, so they fail.
    static let worstCases: [WorstCase] = {
        var cases: [WorstCase] = []
        for direction in [RoundedDistribution.Direction.up, .down] {
            for count in [2, 3, 10, maximumLocalCount, 100] {
                cases.append(WorstCase(count: count, decimals: 1, direction: direction, accepted: false))
                cases.append(WorstCase(count: count, decimals: 2, direction: direction, accepted: count <= 26))
                cases.append(WorstCase(count: count, decimals: 3, direction: direction, accepted: true))
            }
        }
        return cases
    }()

    @Test func maximumLocalCountIsTwentySix() {
        #expect(Self.maximumLocalCount == 26)
        #expect(TokenScoring.scoreLabels.count <= Self.maximumLocalCount)
    }

    @Test(arguments: DistributionKind.allCases, worstCases)
    func worstCaseRounding(kind: DistributionKind, worstCase: WorstCase) {
        let distribution = RoundedDistribution.worstCase(
            count: worstCase.count,
            decimals: worstCase.decimals,
            direction: worstCase.direction
        )
        distribution.checkIsRounding(decimals: worstCase.decimals)
        if worstCase.accepted {
            #expect(throws: Never.self) { try kind.validate(distribution.rounded) }
        } else {
            #expect(throws: DecisionError.self) { try kind.validate(distribution.rounded) }
        }
    }

    /// One-decimal roundings pass when their errors cancel and fail when they add up.
    @Test(
        arguments: DistributionKind.allCases,
        [
            ([0.5, 0.5], true),
            ([0.62, 0.27, 0.11], true),  // 0.6 + 0.3 + 0.1 = 1.0
            ([0.34, 0.33, 0.33], false),  // 0.3 + 0.3 + 0.3 = 0.9
            ([0.26, 0.26, 0.48], false),  // 0.3 + 0.3 + 0.5 = 1.1
        ]
    )
    func oneDecimalRounding(kind: DistributionKind, example: (exact: [Double], accepted: Bool)) {
        let rounded = example.exact.map { ($0 * 10).rounded() / 10 }
        if example.accepted {
            #expect(throws: Never.self) { try kind.validate(rounded) }
        } else {
            #expect(throws: DecisionError.self) { try kind.validate(rounded) }
        }
    }

    // MARK: Boundaries

    /// The allowed difference from 1 is min(max(n, 2) × 0.005, 0.15).
    @Test(
        arguments: DistributionKind.allCases,
        [(2, 0.01), (3, 0.015), (10, 0.05), (26, 0.13), (30, 0.15), (100, 0.15), (1000, 0.15)]
    )
    func boundary(kind: DistributionKind, limit: (count: Int, allowed: Double)) {
        func values(total: Double) -> [Double] {
            [Double](repeating: total / Double(limit.count), count: limit.count)
        }
        for sign in [1.0, -1.0] {
            let inside = 1 + sign * (limit.allowed - 1e-6)
            let outside = 1 + sign * (limit.allowed + 1e-6)
            #expect(throws: Never.self, "total \(inside)") { try kind.validate(values(total: inside)) }
            #expect(throws: DecisionError.self, "total \(outside)") { try kind.validate(values(total: outside)) }
        }
    }

    /// A choice with one option keeps the tolerance of 0.01.
    @Test func singleOptionBoundary() {
        #expect(throws: Never.self) { try DistributionKind.choice.validate([0.99 + 1e-6]) }
        #expect(throws: Never.self) { try DistributionKind.choice.validate([1.01 - 1e-6]) }
        #expect(throws: DecisionError.self) { try DistributionKind.choice.validate([0.99 - 1e-6]) }
        #expect(throws: DecisionError.self) { try DistributionKind.choice.validate([1.01 + 1e-6]) }
    }

    // MARK: Rejections

    @Test(
        arguments: DistributionKind.allCases,
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.1, 0.1, 0.1],
            [0.6, 0.6],
            [0.3, 0.2],
            [70.0, 30.0],
            [0.5, 0.55, -0.05],
            [1.05, -0.05],
            [0.5, .nan],
            [0.5, .infinity],
            [-Double.infinity, 1],
            [Double](repeating: 0.008, count: 100),
            [Double](repeating: 0.0, count: 26),
        ]
    )
    func rejectsWrongDistribution(kind: DistributionKind, probabilities: [Double]) {
        #expect(throws: DecisionError.self) { try kind.validate(probabilities) }
    }

    @Test func rejectsValueAboveOne() {
        #expect(
            throws: DecisionError.invalidResponse(
                "Question at index 0: The probability of level 0 1.05 is not a probability."
            )
        ) {
            try DistributionKind.score.validate([1.05, -0.05])
        }
    }

    @Test func rejectsNegativeValue() {
        #expect(
            throws: DecisionError.invalidResponse(
                "Question at index 0: The probability of level 2 -0.05 is not a probability."
            )
        ) {
            try DistributionKind.score.validate([0.5, 0.55, -0.05])
        }
    }

    @Test func rejectsTotalFarFromOne() {
        #expect(throws: DecisionError.invalidResponse("Question at index 0: The probabilities sum to 0.0, not 1.")) {
            try DistributionKind.score.validate([0, 0, 0])
        }
    }

    @Test func rejectsMissingOrExtraKey() {
        let question = DistributionKind.choice.question(count: 3)
        let missing = Answer.choice(name: "o1", probabilities: ["o0": 0.03, "o1": 0.97], confidence: 0.5)
        let extra = Answer.choice(
            name: "o1",
            probabilities: ["o0": 0.03, "o1": 0.83, "o2": 0.14, "o3": 0.0],
            confidence: 0.5
        )
        let renamed = Answer.choice(
            name: "o1",
            probabilities: ["o0": 0.03, "o1": 0.83, "x": 0.14],
            confidence: 0.5
        )
        for answer in [missing, extra, renamed] {
            #expect(
                throws: DecisionError.invalidResponse(
                    "Question at index 0: The probabilities do not match the options."
                )
            ) {
                try DecisionSession.Response(modelID: "m", answers: [answer]).validate(against: [question])
            }
        }
    }

    @Test func rejectsWrongLevelCount() {
        let question = DistributionKind.score.question(count: 3)
        for probabilities in [[0.17, 0.83], [0.03, 0.83, 0.1, 0.05]] {
            let answer = Answer.score(value: 1, probabilities: probabilities, confidence: 0.5)
            #expect(throws: DecisionError.self) {
                try DecisionSession.Response(modelID: "m", answers: [answer]).validate(against: [question])
            }
        }
    }

    // MARK: Values that keep their tolerance

    @Test(arguments: [0, 1, 0.5, -0.005, 1.005, -0.01, 1.01])
    func acceptsBinaryProbability(probability: Double) throws {
        let question = Question.binary(instructions: "Yes?")
        try DecisionSession.Response(modelID: "m", answers: [.binary(probability: probability)])
            .validate(against: [question])
    }

    @Test(arguments: [-0.02, 1.02, 1.5, .nan, .infinity, -.infinity])
    func rejectsBinaryProbability(probability: Double) {
        let question = Question.binary(instructions: "Yes?")
        #expect(throws: DecisionError.self) {
            try DecisionSession.Response(modelID: "m", answers: [.binary(probability: probability)])
                .validate(against: [question])
        }
    }

    @Test(arguments: DistributionKind.allCases, [0, 1, -0.005, 1.005])
    func acceptsConfidence(kind: DistributionKind, confidence: Double) throws {
        try kind.validate([0.03, 0.83, 0.14], confidence: confidence)
    }

    @Test(arguments: DistributionKind.allCases, [-0.02, 1.02, .nan, .infinity])
    func rejectsConfidence(kind: DistributionKind, confidence: Double) {
        #expect(throws: DecisionError.self) { try kind.validate([0.03, 0.83, 0.14], confidence: confidence) }
    }

    @Test func checksScoreValueRange() throws {
        let question = DistributionKind.score.question(count: 3)
        func validate(_ value: Double) throws {
            let answer = Answer.score(value: value, probabilities: [0.25, 0.5, 0.25], confidence: 0.5)
            try DecisionSession.Response(modelID: "m", answers: [answer]).validate(against: [question])
        }
        try validate(-0.01)
        try validate(2.01)
        #expect(throws: DecisionError.self) { try validate(-0.02) }
        #expect(throws: DecisionError.self) { try validate(2.02) }
        #expect(throws: DecisionError.self) { try validate(.nan) }
    }
}
