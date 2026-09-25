import Foundation

/// A set of decisions about one immutable state.
///
/// Create a session with a model and a state, then ask questions.
/// Questions are independent: answers never become history for later questions.
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

    private let executor: any DecisionModelExecutor

    /// Creates a session.
    ///
    /// - Parameters:
    ///   - model: The model that answers the questions.
    ///   - state: The state that every question is about.
    public init(model: any DecisionModel, state: DecisionState) {
        self.model = model
        self.state = state
        self.executor = model.makeExecutor()
    }

    /// Loads the model and precomputes work for the state.
    ///
    /// Calling this method is optional.
    /// It moves loading cost out of the first question.
    public func prewarm() async throws {
        try await executor.prewarm(for: state)
    }

    /// Answers a batch of independent questions.
    ///
    /// - Parameter questions: The questions.
    /// - Returns: A response with one answer for each question, in the order of the questions.
    /// - Throws: ``DecisionError`` if a question is invalid or the backend fails,
    ///   or `CancellationError` if the task is cancelled.
    public func decide(_ questions: [Question]) async throws -> Response {
        for (index, question) in questions.enumerated() {
            try question.validate(index: index)
            try model.capabilities.check(question, index: index)
        }
        guard !questions.isEmpty else {
            return Response(modelID: model.modelID, answers: [])
        }
        try Task.checkCancellation()
        let response = try await executor.decide(questions, about: state)
        try response.validate(against: questions)
        return response
    }

    /// Answers one question.
    ///
    /// - Parameter question: The question.
    /// - Returns: The answer.
    /// - Throws: ``DecisionError`` if the question is invalid or the backend fails,
    ///   or `CancellationError` if the task is cancelled.
    public func decide(_ question: Question) async throws -> Answer {
        try await decide([question]).answers[0]
    }

    /// Returns the probability that the answer to a yes-or-no question is "yes".
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - criteria: Optional descriptions of true and false answers.
    public func probability(of instructions: String, criteria: BinaryCriteria? = nil) async throws -> Double {
        let answer = try await decide(.binary(instructions: instructions, criteria: criteria))
        guard case .binary(let probability) = answer else {
            throw DecisionError.invalidResponse("Expected a binary answer.")
        }
        return probability
    }

    /// Selects one case of an enumeration.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - type: The option type. It defaults to the inferred result type.
    public func choice<Option: Choosable>(
        _ instructions: String,
        from type: Option.Type = Option.self
    ) async throws -> Choice<Option> {
        let cases = Array(Option.allCases)
        return try await choice(
            instructions,
            options: cases.map { ChoiceOption($0.optionName, description: $0.optionDescription) },
            values: cases
        )
    }

    /// Selects one option from a set defined at run time.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - options: The options, in display order.
    public func choice(_ instructions: String, from options: [ChoiceOption]) async throws -> Choice<String> {
        try await choice(instructions, options: options, values: options.map(\.name))
    }

    /// Places the state on an ordinal scale.
    ///
    /// - Parameters:
    ///   - instructions: The question.
    ///   - levels: Level descriptions, from lowest to highest.
    public func score(_ instructions: String, levels: [String]) async throws -> Score {
        let answer = try await decide(.score(instructions: instructions, levels: levels))
        guard case .score(let value, let probabilities, _, let confidence) = answer else {
            throw DecisionError.invalidResponse("Expected a score answer.")
        }
        return Score(value: value, probabilities: probabilities, levels: levels, confidence: confidence)
    }

    /// Asks a choice question and pairs each option with a typed value.
    private func choice<Option: Hashable & Sendable>(
        _ instructions: String,
        options: [ChoiceOption],
        values: [Option]
    ) async throws -> Choice<Option> {
        let answer = try await decide(.choice(instructions: instructions, options: options))
        guard case .choice(let name, let probabilities, let confidence) = answer else {
            throw DecisionError.invalidResponse("Expected a choice answer.")
        }
        return try Choice(
            name: name,
            probabilities: probabilities,
            confidence: confidence,
            options: Array(zip(values, options.map(\.name)))
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
    /// - Parameters:
    ///   - questions: The questions that the request asked.
    ///   - tolerance: The allowed difference between the probability total and 1.
    /// - Throws: ``DecisionError/invalidResponse(_:)`` for the first problem found.
    func validate(against questions: [Question], tolerance: Double = 0.01) throws {
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
            try answer.validate(for: question, index: index, tolerance: tolerance)
        }
    }
}
