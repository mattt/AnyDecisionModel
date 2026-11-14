import Testing

import AnyDecisionModel

/// Checks that code written against the version 0.9 `DecisionSession` API still compiles,
/// including references to its methods and initializer as function values,
/// and that calls with and without `options:` pick the right overloads.
///
/// This file uses a plain import, so it sees only the public API.
@Suite("DecisionSession source compatibility")
struct DecisionSessionSourceCompatibilityTests {
    @Test func versionZeroNineSignaturesWorkAsFunctionValues() async throws {
        let model = ThresholdDecisionModel()
        let makeSession: (any DecisionModel, DecisionState) -> DecisionSession = DecisionSession.init(model:state:)
        var sessionOptions = DecisionOptions(cacheKey: "session")
        sessionOptions[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)
        let makeSessionWithOptions: (any DecisionModel, DecisionState, DecisionOptions) -> DecisionSession =
            DecisionSession.init(model:state:options:)

        let session = makeSession(model, "ticket")
        #expect(session.options == DecisionOptions())
        #expect(makeSessionWithOptions(model, "ticket", sessionOptions).options == sessionOptions)

        let decideBatch: ([Question]) async throws -> DecisionSession.Response = session.decide
        let decideOne: (Question) async throws -> Answer = session.decide
        let probability: (String, BinaryCriteria?) async throws -> Double = session.probability(of:criteria:)
        let choiceOfType: (String, Department.Type) async throws -> Choice<Department> = session.choice
        let choiceFromOptions: (String, [ChoiceOption]) async throws -> Choice<String> = session.choice
        let score: (String, [String]) async throws -> Score = session.score

        #expect(try await decideBatch([.binary(instructions: "A?")]).answers.count == 1)
        #expect(try await decideOne(.binary(instructions: "B?")) == .binary(probability: 0.5))
        #expect(try await probability("C?", nil) == 0.5)
        #expect(try await choiceOfType("D?", Department.self).value == .billing)
        #expect(try await choiceFromOptions("E?", [ChoiceOption("x"), ChoiceOption("y")]).value == "x")
        #expect(try await score("F?", ["low", "high"]).levels == ["low", "high"])
        #expect(model.received.withLock { $0.count } == 6)
    }

    @Test func newSignaturesWorkAsFunctionValues() async throws {
        let model = ThresholdDecisionModel()
        let session = DecisionSession(model: model, state: "ticket")
        var options = DecisionOptions()
        options[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)

        let decideBatch: ([Question], DecisionOptions) async throws -> DecisionSession.Response = session.decide
        let decideOne: (Question, DecisionOptions) async throws -> Answer = session.decide
        let probability: (String, BinaryCriteria?, DecisionOptions) async throws -> Double =
            session.probability(of:criteria:options:)
        let choiceOfType: (String, Department.Type, DecisionOptions) async throws -> Choice<Department> =
            session.choice
        let choiceFromOptions: (String, [ChoiceOption], DecisionOptions) async throws -> Choice<String> =
            session.choice
        let score: (String, [String], DecisionOptions) async throws -> Score = session.score

        #expect(try await decideBatch([.binary(instructions: "A?")], options).answers == [.binary(probability: 0.9)])
        #expect(try await decideOne(.binary(instructions: "B?"), options) == .binary(probability: 0.9))
        #expect(try await probability("C?", nil, options) == 0.9)
        _ = try await choiceOfType("D?", Department.self, options)
        _ = try await choiceFromOptions("E?", [ChoiceOption("x"), ChoiceOption("y")], options)
        _ = try await score("F?", ["low", "high"], options)
        #expect(model.received.withLock { $0.map(\.options) } == Array(repeating: options, count: 6))
    }

    @Test func callsWithoutOptionsUseSessionOptions() async throws {
        let model = ThresholdDecisionModel()
        var options = DecisionOptions(cacheKey: "session")
        options[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)
        let session = DecisionSession(model: model, state: "ticket", options: options)

        _ = try await session.decide([.binary(instructions: "A?")])
        _ = try await session.decide(.binary(instructions: "B?"))
        #expect(try await session.probability(of: "C?") == 0.9)
        #expect(
            try await session.probability(of: "C?", criteria: BinaryCriteria(whenTrue: "yes", whenFalse: "no")) == 0.9
        )
        let team: Choice<Department> = try await session.choice("D?")
        _ = try await session.choice("D?", from: Department.self)
        _ = try await session.choice("E?", from: [ChoiceOption("x"), ChoiceOption("y")])
        _ = try await session.score("F?", levels: ["low", "high"])

        #expect(team.value == .billing)
        #expect(model.received.withLock { $0.map(\.options) } == Array(repeating: options, count: 8))
    }

    @Test func callsWithOptionsReplaceSessionOptions() async throws {
        let model = ThresholdDecisionModel()
        var sessionOptions = DecisionOptions(cacheKey: "session")
        sessionOptions[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)
        let session = DecisionSession(model: model, state: "ticket", options: sessionOptions)
        let options = DecisionOptions(userID: "call")

        _ = try await session.decide([.binary(instructions: "A?")], options: options)
        _ = try await session.decide(.binary(instructions: "B?"), options: options)
        #expect(try await session.probability(of: "C?", options: options) == 0.5)
        #expect(
            try await session.probability(
                of: "C?",
                criteria: BinaryCriteria(whenTrue: "yes", whenFalse: "no"),
                options: options
            ) == 0.5
        )
        let team: Choice<Department> = try await session.choice("D?", options: options)
        _ = try await session.choice("D?", from: Department.self, options: options)
        _ = try await session.choice("E?", from: [ChoiceOption("x"), ChoiceOption("y")], options: options)
        _ = try await session.score("F?", levels: ["low", "high"], options: options)

        #expect(team.value == .billing)
        #expect(model.received.withLock { $0.map(\.options) } == Array(repeating: options, count: 8))
    }
}
