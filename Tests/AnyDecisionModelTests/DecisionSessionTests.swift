import Foundation
import Testing

@testable import AnyDecisionModel

/// A model that answers from fixed log-masses and records what it receives.
struct FixedDecisionModel: DecisionModel {
    var logMasses: @Sendable (Question) -> [Double]
    var capabilities = DecisionCapabilities(maximumChoiceOptions: 3, maximumScoreLevels: 3)
    let calls = LockedState<[[Question]]>([])
    let prewarmed = LockedState<[DecisionState]>([])

    var modelID: String { "fixed" }

    func makeExecutor() -> any DecisionModelExecutor {
        Executor(model: self)
    }

    struct Executor: DecisionModelExecutor {
        let model: FixedDecisionModel

        func prewarm(for state: DecisionState) async throws {
            model.prewarmed.withLock { $0.append(state) }
        }

        func decide(_ questions: [Question], about state: DecisionState) async throws
            -> DecisionSession.Response
        {
            model.calls.withLock { $0.append(questions) }
            let answers = questions.map { question in
                let binary = if case .binary = question { true } else { false }
                let p = Calibration.identity.probabilities(logMasses: model.logMasses(question), binary: binary)
                return TokenScoring.answer(for: question, probabilities: p)
            }
            return DecisionSession.Response(modelID: "fixed", answers: answers)
        }
    }
}

enum Department: String, Choosable {
    case billing, technical, sales

    var optionDescription: String? {
        switch self {
        case .billing: "payments, charges, refunds, invoices"
        case .technical: "bugs, outages, errors, login problems"
        case .sales: "pricing questions, upgrades, new purchases"
        }
    }
}

@Suite("DecisionSession")
struct DecisionSessionTests {
    let model = FixedDecisionModel { question in
        switch question {
        case .binary: [log(0.8), log(0.2)]
        case .choice(_, let options): options.indices.map { $0 == 0 ? 0 : -3 }
        case .score(_, let levels): levels.indices.map { $0 == levels.count - 1 ? 0 : -2 }
        }
    }

    @Test func probabilityReturnsDouble() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        let p = try await session.probability(of: "Urgent?")
        #expect(abs(p - 0.8) < 1e-12)
    }

    @Test func enumChoiceMapsCasesInOrder() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        let team = try await session.choice("Which team?", from: Department.self)
        #expect(team.value == .billing)
        #expect(team.distribution.map(\.option) == [.billing, .technical, .sales])
        #expect(team.confidence > 0 && team.confidence < 1)

        let sent = try #require(model.calls.withLock { $0.first?.first })
        guard case .choice(_, let options) = sent else {
            Issue.record("Expected a choice question.")
            return
        }
        #expect(options.first == ChoiceOption("billing", description: "payments, charges, refunds, invoices"))
    }

    @Test func runtimeChoice() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        let result = try await session.choice("Which?", from: [ChoiceOption("x"), ChoiceOption("y")])
        #expect(result.value == "x")
        #expect(abs(result.probability(of: "x") + result.probability(of: "y") - 1) < 1e-12)
    }

    @Test func scoreKeepsLevelsAndDistribution() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        let anger = try await session.score("How frustrated?", levels: ["Calm", "Frustrated", "Very angry"])
        #expect(anger.levels == ["Calm", "Frustrated", "Very angry"])
        #expect(anger.mostLikelyLevelIndex == 2)
        #expect(anger.value > 1.5)
    }

    @Test func batchAnswersFollowQuestionOrder() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        let questions: [Question] = [
            .score(instructions: "How frustrated?", levels: ["Calm", "Frustrated", "Very angry"]),
            .binary(instructions: "Urgent?"),
            .choice(instructions: "Which?", options: [ChoiceOption("x"), ChoiceOption("y")]),
        ]
        let response = try await session.decide(questions)
        #expect(model.calls.withLock { $0 } == [questions])
        #expect(response.answers.count == 3)
        guard case .score = response.answers[0],
            case .binary(let probability) = response.answers[1],
            case .choice(let name, _, _) = response.answers[2]
        else {
            Issue.record("Expected score, binary, and choice answers, in that order.")
            return
        }
        #expect(abs(probability - 0.8) < 1e-12)
        #expect(name == "x")
    }

    @Test func singleQuestionReturnsAnswer() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        let answer = try await session.decide(.binary(instructions: "Urgent?"))
        guard case .binary(let probability) = answer else {
            Issue.record("Expected a binary answer.")
            return
        }
        #expect(abs(probability - 0.8) < 1e-12)
    }

    @Test func prewarmUsesSessionState() async throws {
        let session = DecisionSession(model: model, state: .json(["id": 7]))
        try await session.prewarm()
        #expect(model.prewarmed.withLock { $0 } == [.json(["id": 7])])
    }

    @Test func rejectsInvalidQuestionsBeforeCallingModel() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(.choice(instructions: "x", options: []))
        }
        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(.choice(instructions: "x", options: [ChoiceOption("a"), ChoiceOption("a")]))
        }
        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(.score(instructions: "x", levels: ["only"]))
        }
        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(.binary(instructions: "  "))
        }
        #expect(model.calls.withLock { $0.isEmpty })
    }

    @Test func rejectsQuestionsBeyondCapabilities() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        let options = ["a", "b", "c", "d"].map { ChoiceOption($0) }
        await #expect(
            throws: DecisionError.unsupportedQuestion(
                index: 1,
                reason: "The model supports at most 3 choice options, not 4."
            )
        ) {
            _ = try await session.decide([
                .binary(instructions: "x"),
                .choice(instructions: "x", options: options),
            ])
        }
        #expect(model.calls.withLock { $0.isEmpty })
    }

    @Test func emptyBatchReturnsEmptyResponse() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        let response = try await session.decide([])
        #expect(response.answers.isEmpty)
        #expect(response.modelID == "fixed")
    }

    @Test func cancelledTaskDoesNotCallModel() async throws {
        let session = DecisionSession(model: model, state: "ticket")
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await session.probability(of: "Urgent?")
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(model.calls.withLock { $0.isEmpty })
    }
}

@Suite("Response validation")
struct ResponseValidationTests {
    let questions: [Question] = [
        .binary(instructions: "Odd?"),
        .choice(instructions: "Team?", options: [ChoiceOption("billing"), ChoiceOption("sales")]),
        .score(instructions: "Anger?", levels: ["low", "high"]),
    ]

    func response(_ answers: [Answer]) -> DecisionSession.Response {
        DecisionSession.Response(modelID: "m", answers: answers)
    }

    var validAnswers: [Answer] {
        [
            .binary(probability: 0.9),
            .choice(name: "billing", probabilities: ["billing": 0.7, "sales": 0.3], confidence: 0.1),
            .score(value: 0.4, probabilities: [0.6, 0.4], confidence: 0.03),
        ]
    }

    @Test func acceptsValidResponse() throws {
        try response(validAnswers).validate(against: questions)
    }

    @Test func rejectsMissingAnswer() {
        var answers = validAnswers
        answers.removeLast()
        #expect(throws: DecisionError.invalidResponse("Expected 3 answers, got 2.")) {
            try response(answers).validate(against: questions)
        }
    }

    @Test func rejectsDiagnosticsThatDoNotMatchAnswers() {
        var response = response(validAnswers)
        response.diagnostics = [
            DecisionSession.Diagnostics(allowedAnswerMass: 1, cachedTokenCount: 0, evaluatedTokenCount: 1)
        ]
        #expect(throws: DecisionError.invalidResponse("Expected 3 diagnostics, got 1.")) {
            try response.validate(against: questions)
        }
    }

    @Test func rejectsWrongAnswerType() {
        var answers = validAnswers
        answers[0] = answers[2]
        #expect(throws: DecisionError.invalidResponse("Question at index 0: Expected a binary answer.")) {
            try response(answers).validate(against: questions)
        }
    }

    @Test func rejectsUnknownChoice() {
        var answers = validAnswers
        answers[1] = .choice(name: "legal", probabilities: ["billing": 0.7, "sales": 0.3], confidence: 0)
        #expect(throws: DecisionError.self) { try response(answers).validate(against: questions) }
    }

    @Test func rejectsMismatchedOptions() {
        var answers = validAnswers
        answers[1] = .choice(name: "billing", probabilities: ["billing": 1], confidence: 1)
        #expect(throws: DecisionError.self) { try response(answers).validate(against: questions) }
    }

    @Test func rejectsProbabilitiesThatDoNotSumToOne() {
        var answers = validAnswers
        answers[2] = .score(value: 0.5, probabilities: [0.6, 0.6], confidence: 0)
        #expect(throws: DecisionError.self) { try response(answers).validate(against: questions) }
    }

    @Test func rejectsOutOfRangeProbability() {
        var answers = validAnswers
        answers[0] = .binary(probability: 1.5)
        #expect(throws: DecisionError.self) { try response(answers).validate(against: questions) }
    }

    @Test func rejectsWrongLevelCount() {
        var answers = validAnswers
        answers[2] = .score(value: 1, probabilities: [0.2, 0.3, 0.5], confidence: 0)
        #expect(throws: DecisionError.self) { try response(answers).validate(against: questions) }
    }
}
