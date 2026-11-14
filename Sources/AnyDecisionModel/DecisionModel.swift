import Foundation

/// A configured decision backend.
///
/// A model holds configuration only.
/// It creates an executor for each ``DecisionSession``,
/// and the executor does the work.
public protocol DecisionModel: Sendable {
    /// The model identifier reported in responses.
    var modelID: String { get }

    /// The limits and features of the backend.
    var capabilities: DecisionCapabilities { get }

    /// Creates an executor for one session.
    ///
    /// Executors can keep per-session state, such as a cache for the session state.
    /// Creating an executor must be cheap; load resources lazily.
    func makeExecutor() -> any DecisionModelExecutor
}

/// Evaluates batches of decision questions about one state.
///
/// Executors do not need a causal language model or a KV cache.
/// Each question must be answered independently of the other questions in the batch.
public protocol DecisionModelExecutor: Sendable {
    /// Loads resources and precomputes work for a state.
    func prewarm(for state: DecisionState) async throws

    /// Answers a batch of questions about a state.
    ///
    /// - Parameters:
    ///   - questions: The questions.
    ///   - state: The state that the questions are about.
    /// - Returns: A response with one answer for each question, in the order of the questions.
    func decide(_ questions: [Question], about state: DecisionState) async throws
        -> DecisionSession.Response
}

extension DecisionModelExecutor {
    public func prewarm(for state: DecisionState) async throws {}
}

/// The limits and features of a decision backend.
///
/// A session enforces every limit that the capabilities declare
/// before it asks the model to do any work.
/// Call ``validate(_:about:)`` to check a batch without sending it.
public struct DecisionCapabilities: Hashable, Sendable {
    /// The kinds of questions that the model can answer.
    ///
    /// The default is every kind.
    public var questionKinds: Set<DecisionKind>

    /// The largest number of questions in one batch, or `nil` for no local limit.
    public var maximumQuestions: Int?

    /// The largest number of options in a choice question, or `nil` for no local limit.
    public var maximumChoiceOptions: Int?

    /// The largest number of levels in a score question, or `nil` for no local limit.
    public var maximumScoreLevels: Int?

    /// The kinds of attachments that the model can read in a state.
    ///
    /// A session throws ``DecisionError/unsupportedState(_:)``
    /// for a state with an attachment of another kind,
    /// before it asks the model to do any work.
    public var supportedAttachmentKinds: Set<DecisionState.Attachment.Kind>

    /// The largest number of attachments in a state, or `nil` for no local limit.
    public var maximumAttachments: Int?

    /// How the model produces its probabilities.
    public var probabilitySource: ProbabilitySource

    /// Creates a capabilities description.
    ///
    /// - Parameters:
    ///   - questionKinds: The kinds of questions that the model can answer.
    ///   - maximumQuestions: The largest number of questions in one batch,
    ///     or `nil` for no local limit.
    ///   - maximumChoiceOptions: The largest number of options in a choice question,
    ///     or `nil` for no local limit.
    ///   - maximumScoreLevels: The largest number of levels in a score question,
    ///     or `nil` for no local limit.
    ///   - supportedAttachmentKinds: The kinds of attachments that the model can read in a state.
    ///   - maximumAttachments: The largest number of attachments in a state,
    ///     or `nil` for no local limit.
    ///   - probabilitySource: How the model produces its probabilities.
    public init(
        questionKinds: Set<DecisionKind> = [.binary, .choice, .score],
        maximumQuestions: Int? = nil,
        maximumChoiceOptions: Int? = nil,
        maximumScoreLevels: Int? = nil,
        supportedAttachmentKinds: Set<DecisionState.Attachment.Kind> = [],
        maximumAttachments: Int? = nil,
        probabilitySource: ProbabilitySource = .unspecified
    ) {
        self.questionKinds = questionKinds
        self.maximumQuestions = maximumQuestions
        self.maximumChoiceOptions = maximumChoiceOptions
        self.maximumScoreLevels = maximumScoreLevels
        self.supportedAttachmentKinds = supportedAttachmentKinds
        self.maximumAttachments = maximumAttachments
        self.probabilitySource = probabilitySource
    }

    /// Creates a capabilities description with the parameters of version 0.8.
    ///
    /// This initializer keeps references to it,
    /// such as `DecisionCapabilities.init(maximumChoiceOptions:maximumScoreLevels:supportedAttachmentKinds:)`,
    /// compiling.
    /// Calls use the initializer with every parameter.
    ///
    /// - Parameters:
    ///   - maximumChoiceOptions: The largest number of options in a choice question,
    ///     or `nil` for no local limit.
    ///   - maximumScoreLevels: The largest number of levels in a score question,
    ///     or `nil` for no local limit.
    ///   - supportedAttachmentKinds: The kinds of attachments that the model can read in a state.
    @_disfavoredOverload
    public init(
        maximumChoiceOptions: Int?,
        maximumScoreLevels: Int?,
        supportedAttachmentKinds: Set<DecisionState.Attachment.Kind>
    ) {
        self.init(
            questionKinds: [.binary, .choice, .score],
            maximumQuestions: nil,
            maximumChoiceOptions: maximumChoiceOptions,
            maximumScoreLevels: maximumScoreLevels,
            supportedAttachmentKinds: supportedAttachmentKinds,
            maximumAttachments: nil,
            probabilitySource: .unspecified
        )
    }

    /// Checks a batch of questions about a state against the limits.
    ///
    /// This method checks only the limits that the capabilities declare.
    /// A backend can reject a batch for other reasons when it answers,
    /// and ``DecisionSession`` also rejects malformed questions
    /// with ``DecisionError/invalidQuestion(index:reason:)``.
    ///
    /// The checks run in this order:
    /// the kinds of the attachments in the state,
    /// the number of attachments,
    /// the kind, options, and levels of each question in order,
    /// and then the number of questions.
    ///
    /// - Parameters:
    ///   - questions: The questions.
    ///   - state: The state that the questions are about.
    /// - Throws: ``DecisionError/unsupportedState(_:)``
    ///   or ``DecisionError/unsupportedQuestion(index:reason:)``
    ///   for the first limit that the batch exceeds.
    public func validate(_ questions: [Question], about state: DecisionState) throws {
        try validate(questions, about: state, modelDescription: "The model") { _, _ in }
    }

    /// Checks a batch of questions about a state against the limits,
    /// and checks each question with a closure before its limits.
    ///
    /// - Parameters:
    ///   - questions: The questions.
    ///   - state: The state that the questions are about.
    ///   - modelDescription: The name of the model in errors about the state and the number of questions.
    ///   - checkQuestion: A closure that checks one question and its index
    ///     before the question's limits.
    func validate(
        _ questions: [Question],
        about state: DecisionState,
        modelDescription: String,
        checkQuestion: (Question, Int) throws -> Void
    ) throws {
        let attachments = state.attachments
        for attachment in attachments where !supportedAttachmentKinds.contains(attachment.kind) {
            throw DecisionError.unsupportedState(
                "\(modelDescription) does not support \(attachment.kind.rawValue) attachments."
            )
        }
        if let maximum = maximumAttachments, attachments.count > maximum {
            throw DecisionError.unsupportedState(
                "\(modelDescription) supports at most \(maximum) \(maximum == 1 ? "attachment" : "attachments"), "
                    + "not \(attachments.count)."
            )
        }
        for (index, question) in questions.enumerated() {
            try checkQuestion(question, index)
            try check(question, index: index)
        }
        if let maximum = maximumQuestions, questions.count > maximum {
            throw DecisionError.unsupportedQuestion(
                index: maximum,
                reason: "\(modelDescription) accepts at most \(maximum) \(maximum == 1 ? "question" : "questions") "
                    + "in one request, not \(questions.count)."
            )
        }
    }

    /// Checks a question against the limits.
    ///
    /// - Throws: ``DecisionError/unsupportedQuestion(index:reason:)`` if the question exceeds a limit.
    func check(_ question: Question, index: Int) throws {
        guard questionKinds.contains(question.kind) else {
            throw DecisionError.unsupportedQuestion(
                index: index,
                reason: "The model does not support \(question.kind.rawValue) questions."
            )
        }
        switch question {
        case .binary:
            break
        case .choice(_, let options):
            if let maximum = maximumChoiceOptions, options.count > maximum {
                throw DecisionError.unsupportedQuestion(
                    index: index,
                    reason: "The model supports at most \(maximum) choice options, not \(options.count)."
                )
            }
        case .score(_, let levels):
            if let maximum = maximumScoreLevels, levels.count > maximum {
                throw DecisionError.unsupportedQuestion(
                    index: index,
                    reason: "The model supports at most \(maximum) score levels, not \(levels.count)."
                )
            }
        }
    }
}

extension DecisionCapabilities {
    /// How a model produces the probabilities in its answers.
    #if hasAttribute(nonexhaustive)
        @nonexhaustive
    #endif
    public enum ProbabilitySource: String, Hashable, Sendable, Codable {
        /// The model reads probabilities from its output distribution or from a decision head.
        case native

        /// The model was prompted to state a number.
        case estimated

        /// The provider does not say how it produces probabilities.
        case unspecified
    }
}
