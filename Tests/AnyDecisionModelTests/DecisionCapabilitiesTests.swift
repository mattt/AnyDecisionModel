import Foundation
import Testing

@testable import AnyDecisionModel

private let image = DecisionState.Segment.attachment(
    DecisionState.Attachment(imageData: Data([1]), mimeType: "image/png")
)

/// Returns a state with some text and a number of image attachments.
private func state(attachments count: Int) -> DecisionState {
    .segments([.text("ticket")] + Array(repeating: image, count: count))
}

private func options(_ count: Int) -> [ChoiceOption] {
    (0 ..< count).map { ChoiceOption("option \($0)") }
}

private func levels(_ count: Int) -> [String] {
    (0 ..< count).map { "level \($0)" }
}

@Suite("DecisionCapabilities")
struct DecisionCapabilitiesTests {
    @Test func defaultsHaveNoLimits() {
        let capabilities = DecisionCapabilities()
        #expect(capabilities.questionKinds == [.binary, .choice, .score])
        #expect(capabilities.maximumQuestions == nil)
        #expect(capabilities.maximumChoiceOptions == nil)
        #expect(capabilities.maximumScoreLevels == nil)
        #expect(capabilities.supportedAttachmentKinds.isEmpty)
        #expect(capabilities.maximumAttachments == nil)
        #expect(capabilities.probabilitySource == .unspecified)
    }

    @Test func questionKind() {
        #expect(Question.binary(instructions: "x").kind == .binary)
        #expect(Question.choice(instructions: "x", options: options(2)).kind == .choice)
        #expect(Question.score(instructions: "x", levels: levels(2)).kind == .score)
    }

    @Test func validatesQuestionCountAtAndOverLimit() throws {
        let capabilities = DecisionCapabilities(maximumQuestions: 2)
        let question = Question.binary(instructions: "x")
        try capabilities.validate(Array(repeating: question, count: 2), about: "ticket")
        #expect(
            throws: DecisionError.unsupportedQuestion(
                index: 2,
                reason: "The model accepts at most 2 questions in one request, not 3."
            )
        ) {
            try capabilities.validate(Array(repeating: question, count: 3), about: "ticket")
        }
    }

    @Test func validatesChoiceOptionsAtAndOverLimit() throws {
        let capabilities = DecisionCapabilities(maximumChoiceOptions: 3)
        try capabilities.validate([.choice(instructions: "x", options: options(3))], about: "ticket")
        #expect(
            throws: DecisionError.unsupportedQuestion(
                index: 0,
                reason: "The model supports at most 3 choice options, not 4."
            )
        ) {
            try capabilities.validate([.choice(instructions: "x", options: options(4))], about: "ticket")
        }
    }

    @Test func validatesScoreLevelsAtAndOverLimit() throws {
        let capabilities = DecisionCapabilities(maximumScoreLevels: 3)
        try capabilities.validate([.score(instructions: "x", levels: levels(3))], about: "ticket")
        #expect(
            throws: DecisionError.unsupportedQuestion(
                index: 0,
                reason: "The model supports at most 3 score levels, not 4."
            )
        ) {
            try capabilities.validate([.score(instructions: "x", levels: levels(4))], about: "ticket")
        }
    }

    @Test func validatesAttachmentCountAtAndOverLimit() throws {
        let capabilities = DecisionCapabilities(supportedAttachmentKinds: [.image], maximumAttachments: 2)
        try capabilities.validate([], about: state(attachments: 2))
        #expect(throws: DecisionError.unsupportedState("The model supports at most 2 attachments, not 3.")) {
            try capabilities.validate([], about: state(attachments: 3))
        }
    }

    @Test func validatesAttachmentKinds() throws {
        try DecisionCapabilities(supportedAttachmentKinds: [.image]).validate([], about: state(attachments: 1))
        #expect(throws: DecisionError.unsupportedState("The model does not support image attachments.")) {
            try DecisionCapabilities(supportedAttachmentKinds: [.audio]).validate([], about: state(attachments: 1))
        }
    }

    @Test func validatesQuestionKinds() throws {
        let capabilities = DecisionCapabilities(questionKinds: [.binary, .choice])
        let questions: [Question] = [
            .binary(instructions: "x"),
            .choice(instructions: "x", options: options(2)),
        ]
        try capabilities.validate(questions, about: "ticket")
        #expect(
            throws: DecisionError.unsupportedQuestion(
                index: 1,
                reason: "The model does not support score questions."
            )
        ) {
            try capabilities.validate(
                [.binary(instructions: "x"), .score(instructions: "x", levels: levels(2))],
                about: "ticket"
            )
        }
    }

    @Test func validateChecksOnlyLimits() throws {
        // Malformed questions within the limits pass; the session rejects them.
        try DecisionCapabilities().validate(
            [.binary(instructions: " "), .choice(instructions: "x", options: [])],
            about: "ticket"
        )
    }

    @Test func validateReportsStateBeforeQuestionsAndQuestionsBeforeCount() {
        let capabilities = DecisionCapabilities(
            maximumQuestions: 1,
            maximumChoiceOptions: 2,
            maximumAttachments: 0
        )
        let questions: [Question] = [
            .binary(instructions: "x"),
            .choice(instructions: "x", options: options(3)),
        ]
        #expect(throws: DecisionError.unsupportedState("The model does not support image attachments.")) {
            try capabilities.validate(questions, about: state(attachments: 1))
        }
        #expect(
            throws: DecisionError.unsupportedQuestion(
                index: 1,
                reason: "The model supports at most 2 choice options, not 3."
            )
        ) {
            try capabilities.validate(questions, about: "ticket")
        }
    }

    @Test func remoteBackendsDeclareUnspecifiedProbabilities() {
        let gemini = GeminiDecisionModel(apiKey: nil).capabilities
        #expect(gemini.maximumQuestions == 1000)
        #expect(gemini.probabilitySource == .unspecified)
        #expect(JevDecisionModel(apiKey: nil).capabilities.probabilitySource == .unspecified)
        #expect(OllamaDecisionModel(modelID: "nimble").capabilities.probabilitySource == .unspecified)
    }
}

@Suite("DecisionSession capabilities")
struct DecisionSessionCapabilitiesTests {
    func makeModel(_ capabilities: DecisionCapabilities) -> FixedDecisionModel {
        var model = FixedDecisionModel { question in
            switch question {
            case .binary: [0, 0]
            case .choice(_, let options): options.map { _ in 0 }
            case .score(_, let levels): levels.map { _ in 0 }
            }
        }
        model.capabilities = capabilities
        return model
    }

    @Test func rejectsTooManyQuestionsBeforeCallingModel() async throws {
        let model = makeModel(DecisionCapabilities(maximumQuestions: 2))
        let session = DecisionSession(model: model, state: "ticket")
        let question = Question.binary(instructions: "x")
        _ = try await session.decide(Array(repeating: question, count: 2))
        await #expect(
            throws: DecisionError.unsupportedQuestion(
                index: 2,
                reason: "FixedDecisionModel accepts at most 2 questions in one request, not 3."
            )
        ) {
            _ = try await session.decide(Array(repeating: question, count: 3))
        }
        #expect(model.calls.withLock { $0.count } == 1)
    }

    @Test func rejectsUnsupportedQuestionKindBeforeCallingModel() async throws {
        let model = makeModel(DecisionCapabilities(questionKinds: [.binary]))
        let session = DecisionSession(model: model, state: "ticket")
        await #expect(
            throws: DecisionError.unsupportedQuestion(
                index: 0,
                reason: "The model does not support choice questions."
            )
        ) {
            _ = try await session.choice("Which team?", from: Department.self)
        }
        #expect(model.calls.withLock { $0.isEmpty })
    }

    @Test func rejectsTooManyAttachmentsBeforeLoadingModel() async throws {
        let model = makeModel(DecisionCapabilities(supportedAttachmentKinds: [.image], maximumAttachments: 1))
        let session = DecisionSession(model: model, state: state(attachments: 2))
        let error = DecisionError.unsupportedState("FixedDecisionModel supports at most 1 attachment, not 2.")
        await #expect(throws: error) {
            try await session.prewarm()
        }
        await #expect(throws: error) {
            _ = try await session.decide([])
        }
        await #expect(throws: error) {
            _ = try await session.probability(of: "x")
        }
        #expect(model.prewarmed.withLock { $0.isEmpty })
        #expect(model.calls.withLock { $0.isEmpty })
    }

    @Test func keepsOrderOfInvalidAndUnsupportedQuestions() async throws {
        let model = makeModel(DecisionCapabilities(maximumQuestions: 1, maximumChoiceOptions: 2))
        let session = DecisionSession(model: model, state: "ticket")

        // An unsupported question comes before a later invalid question.
        await #expect(
            throws: DecisionError.unsupportedQuestion(
                index: 0,
                reason: "The model supports at most 2 choice options, not 3."
            )
        ) {
            _ = try await session.decide([
                .choice(instructions: "x", options: options(3)),
                .binary(instructions: " "),
            ])
        }

        // An invalid question comes before the question count.
        await #expect(
            throws: DecisionError.invalidQuestion(index: 1, reason: "The instructions are empty.")
        ) {
            _ = try await session.decide([.binary(instructions: "x"), .binary(instructions: " ")])
        }
        #expect(model.calls.withLock { $0.isEmpty })
    }
}
