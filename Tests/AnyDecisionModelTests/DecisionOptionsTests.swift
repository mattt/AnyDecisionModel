import Foundation
import Testing

import AnyDecisionModel

/// A backend written as an external package would write it,
/// against the public `decide(_ request:)` requirement.
///
/// It answers every question with the same fixed answer
/// and reads its own custom options to choose the binary probability.
struct ThresholdDecisionModel: DecisionModel {
    struct CustomOptions: CustomDecisionOptions {
        var probability: Double
    }

    let received = LockedState<[DecisionRequest]>([])

    var modelID: String { "threshold" }

    var capabilities: DecisionCapabilities { DecisionCapabilities() }

    func makeExecutor() -> any DecisionModelExecutor {
        Executor(model: self)
    }

    struct Executor: DecisionModelExecutor {
        let model: ThresholdDecisionModel

        func decide(_ request: DecisionRequest) async throws -> DecisionSession.Response {
            model.received.withLock { $0.append(request) }
            let probability = request.options[custom: ThresholdDecisionModel.self]?.probability ?? 0.5
            let answers: [Answer] = request.questions.map { question in
                switch question {
                case .binary:
                    return .binary(probability: probability)
                case .choice(_, let options):
                    let share = 1 / Double(options.count)
                    return .choice(
                        name: options[0].name,
                        probabilities: Dictionary(uniqueKeysWithValues: options.map { ($0.name, share) }),
                        confidence: 0
                    )
                case .score(_, let levels):
                    let share = 1 / Double(levels.count)
                    return .score(
                        value: Double(levels.count - 1) / 2,
                        probabilities: Array(repeating: share, count: levels.count),
                        legend: levels,
                        confidence: 0
                    )
                }
            }
            return DecisionSession.Response(modelID: "threshold", answers: answers)
        }
    }
}

@Suite("DecisionOptions")
struct DecisionOptionsTests {
    // MARK: - Custom options

    @Test func customOptionsStoreAndReadByModelType() {
        var options = DecisionOptions()
        #expect(options[custom: ThresholdDecisionModel.self] == nil)

        options[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)
        #expect(options[custom: ThresholdDecisionModel.self] == .init(probability: 0.9))

        options[custom: ThresholdDecisionModel.self]?.probability = 0.2
        #expect(options[custom: ThresholdDecisionModel.self]?.probability == 0.2)

        options[custom: ThresholdDecisionModel.self] = nil
        #expect(options[custom: ThresholdDecisionModel.self] == nil)
    }

    @Test func customOptionsReturnNilForAnotherModelType() {
        var options = DecisionOptions()
        options[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)

        #expect(options[custom: SystemOneDecisionModel.self] == nil)
        #expect(options[custom: FixedDecisionModel.self] == nil)

        options[custom: SystemOneDecisionModel.self] = .init(extraBody: ["seed": 1])
        #expect(options[custom: ThresholdDecisionModel.self] == .init(probability: 0.9))
        #expect(options[custom: SystemOneDecisionModel.self] == .init(extraBody: ["seed": 1]))
    }

    @Test func optionsCompareCustomValues() {
        var a = DecisionOptions(cacheKey: "k")
        var b = DecisionOptions(cacheKey: "k")
        #expect(a == b)
        #expect(a.hashValue == b.hashValue)

        a[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)
        #expect(a != b)

        b[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)
        #expect(a == b)
        #expect(a.hashValue == b.hashValue)

        b[custom: ThresholdDecisionModel.self] = .init(probability: 0.1)
        #expect(a != b)
        #expect(Set([a, b]).count == 2)
    }

    // MARK: - Session

    @Test func sessionOptionsReachExecutor() async throws {
        let model = ThresholdDecisionModel()
        var options = DecisionOptions(cacheKey: "prefix-1", userID: "user-1")
        options[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)
        let session = DecisionSession(model: model, state: "ticket", options: options)
        #expect(session.options == options)

        let p = try await session.probability(of: "Urgent?")

        #expect(p == 0.9)
        let request = try #require(model.received.withLock { $0.first })
        #expect(request.options == options)
        #expect(request.state == "ticket")
        #expect(request.questions == [.binary(instructions: "Urgent?")])
    }

    @Test func sessionWithoutOptionsSendsDefaultOptions() async throws {
        let model = ThresholdDecisionModel()
        let session = DecisionSession(model: model, state: "ticket")

        _ = try await session.decide(.binary(instructions: "Urgent?"))

        let request = try #require(model.received.withLock { $0.first })
        #expect(request.options == DecisionOptions())
    }

    @Test func callOptionsReplaceSessionOptions() async throws {
        let model = ThresholdDecisionModel()
        var sessionOptions = DecisionOptions(cacheKey: "prefix-1")
        sessionOptions[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)
        let session = DecisionSession(model: model, state: "ticket", options: sessionOptions)
        let callOptions = DecisionOptions(userID: "user-2")

        let p = try await session.probability(of: "Urgent?", options: callOptions)

        // The call's options replace the session's options; they are not merged.
        #expect(p == 0.5)
        let request = try #require(model.received.withLock { $0.first })
        #expect(request.options == callOptions)
        #expect(request.options.cacheKey == nil)
        #expect(request.options[custom: ThresholdDecisionModel.self] == nil)
    }

    @Test func everySessionMethodPassesCallOptions() async throws {
        let model = ThresholdDecisionModel()
        let session = DecisionSession(model: model, state: "ticket", options: DecisionOptions(cacheKey: "session"))
        let options = DecisionOptions(cacheKey: "call")

        _ = try await session.decide([.binary(instructions: "A?")], options: options)
        _ = try await session.decide(.binary(instructions: "B?"), options: options)
        _ = try await session.probability(of: "C?", options: options)
        _ = try await session.choice("D?", from: Department.self, options: options)
        _ = try await session.choice("E?", from: [ChoiceOption("x"), ChoiceOption("y")], options: options)
        _ = try await session.score("F?", levels: ["low", "high"], options: options)

        let cacheKeys = model.received.withLock { $0.map(\.options.cacheKey) }
        #expect(cacheKeys == Array(repeating: "call", count: 6))
    }

    @Test func everySessionMethodUsesSessionOptionsByDefault() async throws {
        let model = ThresholdDecisionModel()
        let session = DecisionSession(model: model, state: "ticket", options: DecisionOptions(cacheKey: "session"))

        _ = try await session.decide([.binary(instructions: "A?")])
        _ = try await session.decide(.binary(instructions: "B?"))
        _ = try await session.probability(of: "C?")
        _ = try await session.choice("D?", from: Department.self)
        _ = try await session.choice("E?", from: [ChoiceOption("x"), ChoiceOption("y")])
        _ = try await session.score("F?", levels: ["low", "high"])

        let cacheKeys = model.received.withLock { $0.map(\.options.cacheKey) }
        #expect(cacheKeys == Array(repeating: "session", count: 6))
    }

    @Test func externalExecutorAnswersThroughSession() async throws {
        let model = ThresholdDecisionModel()
        let session = DecisionSession(model: model, state: .json(["id": 7]))

        let response = try await session.decide([
            .binary(instructions: "A?"),
            .choice(instructions: "B?", options: [ChoiceOption("x"), ChoiceOption("y")]),
            .score(instructions: "C?", levels: ["low", "mid", "high"]),
        ])

        #expect(response.modelID == "threshold")
        #expect(response.answers.count == 3)
        let request = try #require(model.received.withLock { $0.first })
        #expect(request.state == .json(["id": 7]))
        #expect(request.questions.count == 3)
    }

    @Test func requestStoresItsValues() {
        var options = DecisionOptions(userID: "u")
        options[custom: ThresholdDecisionModel.self] = .init(probability: 0.3)
        let request = DecisionRequest(state: "s", questions: [.binary(instructions: "A?")], options: options)

        #expect(request.state == "s")
        #expect(request.questions == [.binary(instructions: "A?")])
        #expect(request.options == options)
        #expect(DecisionRequest(state: "s", questions: []).options == DecisionOptions())
    }
}
