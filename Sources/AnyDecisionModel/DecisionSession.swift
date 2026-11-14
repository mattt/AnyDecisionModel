import Foundation

/// A set of decisions about one immutable state.
///
/// Create a session with a model and a state, then ask questions.
/// Questions are independent: answers never become history for later questions.
///
/// A session sends its ``options`` with every request.
/// To use other options for one call, pass `options:` to that call.
/// The value replaces the session's options for the call;
/// the two values are not merged.
/// Each method that takes `options:` has an overload without it,
/// with the signature of version 0.9,
/// which uses the session's options.
///
/// ```swift
/// let model = MLXDecisionModel(modelID: "mlx-community/Qwen3-4B-Instruct-2507-4bit")
/// let session = DecisionSession(model: model, state: .text(ticket))
///
/// let urgent = try await session.probability(of: "Does this convey urgency?")
/// let team = try await session.choice("Which team?", from: Department.self)
/// ```
public final class DecisionSession: Sendable {
    /// The model that answers the questions.
    public let model: any DecisionModel

    /// The state that every question is about.
    public let state: DecisionState

    /// The options that the session sends with each request
    /// that does not pass its own options.
    public let options: DecisionOptions

    private let executor: any DecisionModelExecutor

    /// Creates a session with options.
    ///
    /// - Parameters:
    ///   - model: The model that answers the questions.
    ///   - state: The state that every question is about.
    ///   - options: The options to send with each request
    ///     that does not pass its own options.
    public init(model: any DecisionModel, state: DecisionState, options: DecisionOptions) {
        self.model = model
        self.state = state
        self.options = options
        self.executor = model.makeExecutor()
    }

    /// Creates a session with default options.
    ///
    /// - Parameters:
    ///   - model: The model that answers the questions.
    ///   - state: The state that every question is about.
    public convenience init(model: any DecisionModel, state: DecisionState) {
        self.init(model: model, state: state, options: DecisionOptions())
    }

    /// Checks a batch against the model's capabilities,
    /// and checks that each question is well formed before its limits.
    ///
    /// - Throws: ``DecisionError/unsupportedState(_:)``,
    ///   ``DecisionError/invalidQuestion(index:reason:)``,
    ///   or ``DecisionError/unsupportedQuestion(index:reason:)``
    ///   for the first problem found.
    private func validate(_ questions: [Question]) throws {
        try model.capabilities.validate(
            questions,
            about: state,
            modelDescription: String(describing: type(of: model))
        ) { question, index in
            try question.validate(index: index)
        }
    }

    /// Loads the model and precomputes work for the state.
    ///
    /// Calling this method is optional.
    /// It moves loading cost out of the first question.
    ///
    /// - Throws: ``DecisionError/unsupportedState(_:)``
    ///   if the state exceeds a limit in the model's ``DecisionModel/capabilities``,
    ///   before the model loads.
    public func prewarm() async throws {
        try validate([])
        try await executor.prewarm(for: state)
    }

    /// Answers a batch of independent questions with the session's options.
    ///
    /// The session checks the batch against the model's ``DecisionModel/capabilities``
    /// before it asks the model to do any work.
    ///
    /// - Parameter questions: The questions.
    /// - Returns: A response with one answer for each question, in the order of the questions.
    /// - Throws: ``DecisionError`` if a question is invalid, the batch exceeds a limit,
    ///   or the backend fails,
    ///   or `CancellationError` if the task is cancelled.
    ///   A backend can also throw its own error for an option that it rejects.
    public func decide(_ questions: [Question]) async throws -> Response {
        try await decide(questions, options: options)
    }

    /// Answers a batch of independent questions with options for this request.
    ///
    /// The session checks the batch against the model's ``DecisionModel/capabilities``
    /// before it asks the model to do any work.
    ///
    /// - Parameters:
    ///   - questions: The questions.
    ///   - options: The options for this request.
    ///     They replace the session's ``options``; the two are not merged.
    /// - Returns: A response with one answer for each question, in the order of the questions.
    /// - Throws: ``DecisionError`` if a question is invalid, the batch exceeds a limit,
    ///   or the backend fails,
    ///   or `CancellationError` if the task is cancelled.
    ///   A backend can also throw its own error for an option that it rejects.
    public func decide(_ questions: [Question], options: DecisionOptions) async throws -> Response {
        try validate(questions)
        guard !questions.isEmpty else {
            return Response(modelID: model.modelID, answers: [])
        }
        try Task.checkCancellation()
        let request = DecisionRequest(state: state, questions: questions, options: options)
        let response = try await executor.decide(request)
        try response.validate(against: questions)
        return response
    }

    /// Answers one question with the session's options.
    ///
    /// - Parameter question: The question.
    /// - Returns: The answer.
    /// - Throws: ``DecisionError`` if the question is invalid or the backend fails,
    ///   or `CancellationError` if the task is cancelled.
    public func decide(_ question: Question) async throws -> Answer {
        try await decide(question, options: options)
    }

    /// Answers one question with options for this request.
    ///
    /// - Parameters:
    ///   - question: The question.
    ///   - options: The options for this request.
    ///     They replace the session's ``options``; the two are not merged.
    /// - Returns: The answer.
    /// - Throws: ``DecisionError`` if the question is invalid or the backend fails,
    ///   or `CancellationError` if the task is cancelled.
    public func decide(_ question: Question, options: DecisionOptions) async throws -> Answer {
        try await decide([question], options: options).answers[0]
    }

    /// Returns the probability that the answer to a yes-or-no question is "yes",
    /// with the session's options.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - criteria: Optional descriptions of true and false answers.
    public func probability(of instructions: String, criteria: BinaryCriteria? = nil) async throws -> Double {
        try await probability(of: instructions, criteria: criteria, options: options)
    }

    /// Returns the probability that the answer to a yes-or-no question is "yes",
    /// with options for this request.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - criteria: Optional descriptions of true and false answers.
    ///   - options: The options for this request.
    ///     They replace the session's ``options``; the two are not merged.
    public func probability(
        of instructions: String,
        criteria: BinaryCriteria? = nil,
        options: DecisionOptions
    ) async throws -> Double {
        let answer = try await decide(.binary(instructions: instructions, criteria: criteria), options: options)
        guard case .binary(let probability) = answer else {
            throw DecisionError.invalidResponse("Expected a binary answer.")
        }
        return probability
    }

    /// Selects one case of an enumeration, with the session's options.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - type: The option type. It defaults to the inferred result type.
    public func choice<Option: Choosable>(
        _ instructions: String,
        from type: Option.Type = Option.self
    ) async throws -> Choice<Option> {
        try await choice(instructions, from: type, options: options)
    }

    /// Selects one case of an enumeration, with options for this request.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - type: The option type. It defaults to the inferred result type.
    ///   - options: The options for this request.
    ///     They replace the session's ``options``; the two are not merged.
    public func choice<Option: Choosable>(
        _ instructions: String,
        from type: Option.Type = Option.self,
        options: DecisionOptions
    ) async throws -> Choice<Option> {
        let cases = Array(Option.allCases)
        return try await choice(
            instructions,
            choiceOptions: cases.map { ChoiceOption($0.optionName, description: $0.optionDescription) },
            values: cases,
            options: options
        )
    }

    /// Selects one option from a set defined at run time, with the session's options.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - choiceOptions: The options to choose from, in display order.
    public func choice(_ instructions: String, from choiceOptions: [ChoiceOption]) async throws -> Choice<String> {
        try await choice(instructions, from: choiceOptions, options: options)
    }

    /// Selects one option from a set defined at run time, with options for this request.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - choiceOptions: The options to choose from, in display order.
    ///   - options: The options for this request.
    ///     They replace the session's ``options``; the two are not merged.
    public func choice(
        _ instructions: String,
        from choiceOptions: [ChoiceOption],
        options: DecisionOptions
    ) async throws -> Choice<String> {
        try await choice(
            instructions,
            choiceOptions: choiceOptions,
            values: choiceOptions.map(\.name),
            options: options
        )
    }

    /// Places the state on an ordinal scale, with the session's options.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - levels: Level descriptions, from lowest to highest.
    public func score(_ instructions: String, levels: [String]) async throws -> Score {
        try await score(instructions, levels: levels, options: options)
    }

    /// Places the state on an ordinal scale, with options for this request.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - levels: Level descriptions, from lowest to highest.
    ///   - options: The options for this request.
    ///     They replace the session's ``options``; the two are not merged.
    public func score(
        _ instructions: String,
        levels: [String],
        options: DecisionOptions
    ) async throws -> Score {
        let answer = try await decide(.score(instructions: instructions, levels: levels), options: options)
        guard case .score(let value, let probabilities, _, let confidence) = answer else {
            throw DecisionError.invalidResponse("Expected a score answer.")
        }
        return Score(value: value, probabilities: probabilities, levels: levels, confidence: confidence)
    }

    /// Asks a choice question and pairs each option with a typed value.
    private func choice<Option: Hashable & Sendable>(
        _ instructions: String,
        choiceOptions: [ChoiceOption],
        values: [Option],
        options: DecisionOptions
    ) async throws -> Choice<Option> {
        let answer = try await decide(.choice(instructions: instructions, options: choiceOptions), options: options)
        guard case .choice(let name, let probabilities, let confidence) = answer else {
            throw DecisionError.invalidResponse("Expected a choice answer.")
        }
        return try Choice(
            name: name,
            probabilities: probabilities,
            confidence: confidence,
            options: Array(zip(values, choiceOptions.map(\.name)))
        )
    }
}

extension DecisionSession {
    /// The result of a batch of decision questions.
    public struct Response: Hashable, Sendable {
        /// The identifier of the model that answered.
        public var modelID: String

        /// The answers, in the order of the questions.
        public var answers: [Answer]

        /// Token usage for the request.
        public var usage: Usage

        /// Local diagnostics for each answer, in the order of the questions.
        ///
        /// Remote backends leave this empty.
        public var diagnostics: [Diagnostics]

        /// Creates a response.
        public init(
            modelID: String,
            answers: [Answer],
            usage: Usage = Usage(),
            diagnostics: [Diagnostics] = []
        ) {
            self.modelID = modelID
            self.answers = answers
            self.usage = usage
            self.diagnostics = diagnostics
        }
    }

    /// Token usage for a decision request.
    public struct Usage: Hashable, Sendable {
        /// The number of input tokens that the backend counted.
        ///
        /// Local models count every prompt token, including tokens reused from a cache.
        public var inputTokenCount: Int

        /// The number of output tokens. Decisions generate no text, so this is usually 0.
        public var outputTokenCount: Int

        /// Creates a usage record.
        public init(inputTokenCount: Int = 0, outputTokenCount: Int = 0) {
            self.inputTokenCount = inputTokenCount
            self.outputTokenCount = outputTokenCount
        }
    }

    /// Local diagnostics for one answer.
    public struct Diagnostics: Hashable, Sendable {
        /// The share of next-token probability that falls on the allowed answer tokens,
        /// before normalization.
        ///
        /// A low value means that the model wanted to reply with something else.
        /// It is not a confidence value
        /// and not the probability that the answer is correct.
        public var allowedAnswerMass: Double

        /// The number of prompt tokens read from the session's prefix cache.
        ///
        /// For Core AI models, this is the number of tokens that the engine reused from its cache,
        /// which can include tokens that the prompt shares with the previous prompt after the prefix.
        public var cachedTokenCount: Int

        /// The number of prompt tokens evaluated for this answer.
        public var evaluatedTokenCount: Int

        /// The duration attributed to this answer.
        ///
        /// For MLX models, this is an estimate allocated from the request's total
        /// prompt preparation and scoring time, measured with a continuous clock.
        /// Each answer receives a share proportional to its number of prompt evaluations,
        /// including option-order rotations.
        /// The allocation weights each prompt equally, regardless of its length.
        /// The estimate can change when other questions in the request change.
        /// The measured time includes evaluation of the model's output.
        /// It excludes model loading and filling the session's shared prefix cache.
        ///
        /// For Core AI models, this is the time to render and read the answer's prompts,
        /// including option-order rotations,
        /// plus the time that the engine reports for evaluating them.
        /// It excludes model loading, filling the session's prefix cache,
        /// and time spent waiting while the shared engine evaluates prompts for other sessions.
        ///
        /// The value is zero if no duration was recorded.
        public var duration: Duration

        /// Creates diagnostics for one answer.
        ///
        /// - Parameters:
        ///   - allowedAnswerMass: The probability mass of the allowed answer tokens.
        ///   - cachedTokenCount: The number of prompt tokens read from the prefix cache.
        ///   - evaluatedTokenCount: The number of prompt tokens evaluated for this answer.
        ///   - duration: The duration attributed to this answer, or zero if not recorded.
        public init(
            allowedAnswerMass: Double,
            cachedTokenCount: Int,
            evaluatedTokenCount: Int,
            duration: Duration = .zero
        ) {
            self.allowedAnswerMass = allowedAnswerMass
            self.cachedTokenCount = cachedTokenCount
            self.evaluatedTokenCount = evaluatedTokenCount
            self.duration = duration
        }
    }
}

// MARK: - Validation

extension DecisionSession.Response {
    /// Checks that the response answers every question with a well-formed answer.
    ///
    /// A choice or score distribution can sum to a value near 1
    /// that a rounding of each probability explains;
    /// ``Answer/totalTolerance(count:)`` describes the rule.
    /// The response keeps those probabilities unchanged.
    ///
    /// - Parameter questions: The questions that the request asked.
    /// - Throws: ``DecisionError/invalidResponse(_:)`` for the first problem found.
    func validate(against questions: [Question]) throws {
        guard answers.count == questions.count else {
            throw DecisionError.invalidResponse(
                "Expected \(questions.count) answers, got \(answers.count)."
            )
        }
        guard diagnostics.isEmpty || diagnostics.count == answers.count else {
            throw DecisionError.invalidResponse(
                "Expected \(answers.count) diagnostics, got \(diagnostics.count)."
            )
        }
        for (index, (question, answer)) in zip(questions, answers).enumerated() {
            try answer.validate(for: question, index: index)
        }
    }
}
