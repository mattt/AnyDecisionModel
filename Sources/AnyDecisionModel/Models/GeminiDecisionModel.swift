import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A decision model that calls the Gemini Decisions API.
///
/// The model sends each batch to `POST /v1beta/decisions`
/// and returns Gemini's probabilities and confidence values unchanged.
/// Gemini clamps the probability of a binary answer to 0.01 through 0.99.
///
/// By default, the API key comes from the `GEMINI_API_KEY` environment variable.
/// The key is sent only in the `x-goog-api-key` header
/// and is never included in descriptions or errors.
///
/// ```swift
/// let model = GeminiDecisionModel()
/// let session = DecisionSession(model: model, state: .text(ticket))
/// let urgent = try await session.probability(of: "Does this convey urgency?")
/// ```
///
/// A text state is sent as the input text.
/// A JSON state is sent as compact JSON text with sorted keys.
public struct GeminiDecisionModel: DecisionModel, CustomStringConvertible {
    /// The default API base URL.
    public static let defaultBaseURL = URL(string: "https://generativelanguage.googleapis.com")!

    /// The environment variable that supplies the API key.
    private static let apiKeyEnvironmentVariable = "GEMINI_API_KEY"

    /// The largest number of questions in one request.
    static let maximumQuestions = 1000

    /// The largest number of options in a choice question, or levels in a score question.
    static let maximumCriteria = 255

    /// The largest number of characters in an option name.
    static let maximumOptionNameLength = 128

    /// The API base URL.
    public let baseURL: URL

    /// The Gemini decision model identifier, such as `gemini-decision-1`.
    public let modelID: String

    private let tokenProvider: @Sendable () -> String?
    private let urlSession: URLSession

    /// Creates a Gemini decision model.
    ///
    /// - Parameters:
    ///   - baseURL: The API base URL.
    ///   - apiKey: The API key, or a closure that returns it.
    ///     Defaults to the value of `GEMINI_API_KEY`.
    ///   - modelID: The Gemini decision model identifier.
    ///   - session: The URL session for requests.
    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String? =
            GeminiDecisionModel.environmentAPIKey(),
        modelID: String = "gemini-decision-1",
        session: URLSession = URLSession(configuration: .default)
    ) {
        self.baseURL = baseURL
        self.tokenProvider = tokenProvider
        self.modelID = modelID
        self.urlSession = session
    }

    /// Returns the API key from the environment, or `nil` if none is set.
    ///
    /// - Parameter environment: The environment to read.
    public static func environmentAPIKey(
        from environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        guard
            let value = environment[apiKeyEnvironmentVariable]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !value.isEmpty
        else { return nil }
        return value
    }

    public var capabilities: DecisionCapabilities {
        DecisionCapabilities(
            maximumChoiceOptions: Self.maximumCriteria,
            maximumScoreLevels: Self.maximumCriteria
        )
    }

    public var description: String {
        "GeminiDecisionModel(modelID: \(modelID), baseURL: \(baseURL.absoluteString))"
    }

    public func makeExecutor() -> any DecisionModelExecutor {
        Executor(model: self)
    }

    private struct Executor: DecisionModelExecutor {
        let model: GeminiDecisionModel

        func decide(_ questions: [Question], about state: DecisionState) async throws
            -> DecisionSession.Response
        {
            try GeminiDecisionModel.check(questions)

            // Gemini keys questions and answers by identifier. Use each question's position.
            let keys = questions.indices.map(String.init)
            let request = GeminiRequest(
                model: model.modelID,
                input: state.jsonValue.promptText,
                questions: Dictionary(
                    uniqueKeysWithValues: zip(keys, questions.map { SystemOneQuestion($0, dialect: .gemini) })
                )
            )
            let response = try await model.send(request)
            if let feedback = response.promptFeedback, let reason = feedback.blockReason {
                throw DecisionError.inputBlocked(reason: reason, message: feedback.blockReasonMessage)
            }
            let answers = try keys.map { key in
                guard let answer = response.answers[key] else {
                    throw DecisionError.invalidResponse("No answer for the question at index \(key).")
                }
                return answer.answer
            }
            return DecisionSession.Response(
                modelID: response.model ?? model.modelID,
                answers: answers,
                usage: DecisionSession.Usage(
                    inputTokenCount: response.usage?.inputTokenCount ?? 0,
                    outputTokenCount: response.usage?.outputTokenCount ?? 0
                )
            )
        }
    }

    /// Checks a batch against the limits of the Gemini Decisions API.
    ///
    /// The session already checks the largest number of options and levels
    /// through ``capabilities``.
    ///
    /// - Throws: ``DecisionError/unsupportedQuestion(index:reason:)``
    ///   for the first question that the API would reject.
    static func check(_ questions: [Question]) throws {
        if questions.count > maximumQuestions {
            throw DecisionError.unsupportedQuestion(
                index: maximumQuestions,
                reason: "Gemini accepts at most \(maximumQuestions) questions in one request, not \(questions.count)."
            )
        }
        for (index, question) in questions.enumerated() {
            switch question {
            case .binary:
                break
            case .choice(_, let options):
                guard options.count >= 2 else {
                    throw DecisionError.unsupportedQuestion(
                        index: index,
                        reason: "Gemini needs at least 2 choice options, not \(options.count)."
                    )
                }
                for option in options {
                    if let reason = invalidOptionNameReason(option.name) {
                        throw DecisionError.unsupportedQuestion(index: index, reason: reason)
                    }
                }
            case .score(_, let levels):
                if levels.contains(where: { $0.isEmpty }) {
                    throw DecisionError.unsupportedQuestion(
                        index: index,
                        reason: "Gemini does not accept an empty score level."
                    )
                }
            }
        }
    }

    /// Returns why the API would reject an option name, or `nil` if it accepts the name.
    ///
    /// An option name must have 1 to 128 characters.
    /// It must not start or end with whitespace,
    /// and it must not contain control characters or spaces other than the ASCII space.
    ///
    /// This check counts Unicode scalars as characters.
    /// A scalar has at most 4 UTF-8 bytes,
    /// so a name that passes also meets Gemini's limit of 512 UTF-8 bytes.
    static func invalidOptionNameReason(_ name: String) -> String? {
        let length = name.unicodeScalars.count
        guard (1 ... maximumOptionNameLength).contains(length) else {
            return "Gemini accepts option names of 1 to \(maximumOptionNameLength) characters, not \(length)."
        }
        if let first = name.unicodeScalars.first, let last = name.unicodeScalars.last,
            first.properties.isWhitespace || last.properties.isWhitespace
        {
            return "The option name \"\(name)\" starts or ends with whitespace."
        }
        for scalar in name.unicodeScalars {
            if scalar.properties.generalCategory == .control {
                return "An option name contains a control character."
            }
            if scalar != " " && scalar.properties.isWhitespace {
                return "The option name \"\(name)\" contains a space other than the ASCII space."
            }
        }
        return nil
    }

    /// Sends a request and returns the decoded response.
    private func send(_ request: GeminiRequest) async throws -> GeminiResponse {
        guard let apiKey = tokenProvider(), !apiKey.isEmpty else {
            throw DecisionError.missingCredentials
        }

        let url = baseURL.appendingPathComponent("v1beta").appendingPathComponent("decisions")
        let body = try JSONEncoder().encode(request)
        let headers = ["x-goog-api-key": apiKey]

        let data: Data
        do {
            (data, _) = try await urlSession.send(.post, url: url, headers: headers, body: body)
        } catch DecisionError.requestFailed(let statusCode, let detail, let responseHeaders) {
            throw Self.error(statusCode: statusCode, detail: detail, headers: responseHeaders)
        }
        do {
            return try JSONDecoder().decode(GeminiResponse.self, from: data)
        } catch {
            throw DecisionError.invalidResponse(String(describing: error))
        }
    }

    /// Maps an HTTP error to a decision error.
    ///
    /// Gemini reports a blocked input as HTTP 400
    /// with a string error code, such as `safety`.
    /// That error becomes ``DecisionError/inputBlocked(reason:message:)``.
    /// Any other error stays ``DecisionError/requestFailed(statusCode:detail:headers:)``
    /// with the original response body.
    static func error(statusCode: Int, detail: String, headers: [String: String]) -> DecisionError {
        if statusCode == 400,
            let body = try? JSONDecoder().decode(GeminiErrorBody.self, from: Data(detail.utf8)),
            let reason = body.error?.code,
            blockReasons.contains(reason)
        {
            return .inputBlocked(reason: reason, message: body.error?.message)
        }
        return .requestFailed(statusCode: statusCode, detail: detail, headers: headers)
    }

    /// The error codes and block reasons that Gemini documents for a blocked input.
    static let blockReasons: Set<String> = [
        "safety",
        "other",
        "blocklist",
        "prohibited_content",
        "image_safety",
        "content_blocked",
    ]
}

// MARK: - Wire format

/// A batch decision request in the Gemini Decisions API wire format.
struct GeminiRequest: Encodable {
    /// The model identifier.
    var model: String

    /// The text that every question is about.
    var input: String

    /// The questions, keyed by question identifier.
    var questions: [String: SystemOneQuestion]
}

/// A batch decision response in the Gemini Decisions API wire format.
struct GeminiResponse: Decodable {
    /// The identifier of the model that answered, if the response includes it.
    var model: String?

    /// The answers, keyed by question identifier.
    var answers: [String: SystemOneAnswer]

    /// Token usage for the request.
    var usage: Usage?

    /// Feedback about the input, present when the input was blocked.
    var promptFeedback: GeminiPromptFeedback?

    struct Usage: Decodable {
        var inputTokenCount: Int
        var outputTokenCount: Int

        private enum CodingKeys: String, CodingKey {
            case inputTokenCount = "total_input_tokens"
            case outputTokenCount = "total_output_tokens"
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            inputTokenCount = try container.decodeIfPresent(Int.self, forKey: .inputTokenCount) ?? 0
            outputTokenCount = try container.decodeIfPresent(Int.self, forKey: .outputTokenCount) ?? 0
        }
    }

    private enum CodingKeys: String, CodingKey {
        case model
        case answers
        case usage
        case promptFeedback = "prompt_feedback"
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        promptFeedback = try container.decodeIfPresent(GeminiPromptFeedback.self, forKey: .promptFeedback)
        usage = try container.decodeIfPresent(Usage.self, forKey: .usage)
        // A blocked input can come back with no answers.
        if promptFeedback?.blockReason != nil {
            answers = (try? container.decodeIfPresent([String: SystemOneAnswer].self, forKey: .answers)) ?? [:]
        } else {
            answers = try container.decode([String: SystemOneAnswer].self, forKey: .answers)
        }
    }
}

/// Feedback about the input of a Gemini Decisions API request.
struct GeminiPromptFeedback: Decodable {
    /// Why the input was blocked, such as `safety`, or `nil` if it was not blocked.
    var blockReason: String?

    /// A description of why the input was blocked.
    var blockReasonMessage: String?

    private enum CodingKeys: String, CodingKey {
        case blockReason = "block_reason"
        case blockReasonMessage = "block_reason_message"
    }
}

/// An error response body from the Gemini Decisions API.
struct GeminiErrorBody: Decodable {
    struct Detail: Decodable {
        /// The error code, if it is a string.
        ///
        /// A blocked input has a string code, such as `safety`.
        /// A numeric code decodes as `nil`.
        var code: String?

        var message: String?

        private enum CodingKeys: String, CodingKey {
            case code
            case message
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            code = try? container.decodeIfPresent(String.self, forKey: .code)
            message = try? container.decodeIfPresent(String.self, forKey: .message)
        }
    }

    var error: Detail?
}
