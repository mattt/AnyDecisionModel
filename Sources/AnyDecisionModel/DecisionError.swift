import Foundation
import HTTPTypes

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// An error from a decision model or session.
///
/// Every error has a ``kind`` that names what went wrong
/// and a ``message`` that describes it.
/// The other fields are set when they apply:
/// ``questionIndex`` for an error about one question,
/// ``reason`` for a provider's code,
/// ``response`` and ``body`` when the provider answered with an error status,
/// and ``underlyingError`` for the error that caused this one.
///
/// Match an error by its kind:
///
/// ```swift
/// do {
///     let urgent = try await session.probability(of: "Does this convey urgency?")
/// } catch let error as DecisionError where error.kind == .rateLimited {
///     try await Task.sleep(for: error.retryAfter ?? .seconds(1))
/// }
/// ```
///
/// A request that fails on the way, such as one the network drops,
/// throws the transport's own error instead: `URLError` from a URL session.
/// So does a request that is cancelled while the URL session sends it.
/// A model that is cancelled at any other point throws `CancellationError`.
///
/// An error keeps some credentials out of ``response`` and ``body``.
/// In ``response``, the values of the headers
/// `Authorization`, `Proxy-Authorization`, `X-Api-Key`, and `X-Goog-Api-Key` are replaced.
/// For a response with a status code outside 200 to 299,
/// the remote backends also replace in ``body``, and so in ``message``,
/// each value that the request sent in one of those headers,
/// and the token after the scheme in a value such as `Bearer token`.
/// Nothing else is redacted:
/// other response headers,
/// text from a successful response, such as a provider's reason for blocking the input,
/// and the message, body, and underlying error of an error that you create
/// are kept as they are.
///
/// ## Migration from version 0.10
///
/// Through version 0.10, `DecisionError` is an enum,
/// and version 0.11 makes it a struct.
/// Code that creates an error compiles unchanged,
/// because each case of the enum is now a static factory or property with the same message.
/// Other code needs these changes:
///
/// - `DecisionError` is not `Equatable` or `Hashable`,
///   because ``underlyingError`` can hold any error.
///   Replace `#expect(throws: DecisionError.missingCredentials)`
///   and `error == .missingCredentials`
///   with a comparison of ``kind``, and of ``message`` if needed.
/// - Replace `catch DecisionError.requestFailed(let status, let detail, let headers)`
///   with `catch let error as DecisionError where error.response != nil`,
///   and read `error.response?.status.code`, ``body``, and `error.response?.headerFields`.
///   An HTTP failure has the kind
///   ``Kind/authenticationFailed``, ``Kind/quotaExceeded``, ``Kind/timedOut``,
///   ``Kind/rateLimited``, or ``Kind/httpError``.
/// - Replace `catch DecisionError.invalidResponse(let detail)`
///   and the same pattern for the other cases
///   with `catch let error as DecisionError where error.kind == .invalidResponse`,
///   and read ``message``.
///   A response body that can't be decoded now has the kind ``Kind/decodingError``,
///   and a blocked input has the kind ``Kind/guardrailViolation``.
/// - Replace `if case .unsupportedQuestion(let index, _) = error`
///   with a check of ``kind`` and ``questionIndex``.
/// - Replace `switch error` over the cases with `switch error.kind`,
///   and add a `default` case.
/// - Read `Retry-After` from ``retryAfter``
///   in place of `headers["retry-after"]`.
/// - Catch ``DecisionError`` with the kind ``Kind/unsupportedOption``
///   in place of ``SystemOneDecisionModel/ConfigurationError``
///   for a request whose ``SystemOneDecisionModel/CustomOptions`` set a reserved key.
/// - Catch `URLError` with the code `cancelled`, as well as `CancellationError`,
///   for a remote model that is cancelled during a request.
/// - Catch ``DecisionError`` with the kind ``Kind/invalidRequest``
///   in place of `EncodingError` for a request body that a remote model can't encode.
///   The `EncodingError` is in ``underlyingError``.
public struct DecisionError: Error, Sendable, CustomStringConvertible, LocalizedError {
    /// What went wrong. More kinds can be added, so a `switch` over them needs a `default`.
    public struct Kind: Hashable, Sendable, CustomStringConvertible {
        public let description: String

        private init(_ description: String) {
            self.description = description
        }

        // MARK: Transport

        /// The request couldn't be sent as given.
        public static let invalidRequest = Kind("Invalid request")
        /// The answer wasn't an HTTP response, or it doesn't match the request.
        public static let invalidResponse = Kind("Invalid response")
        /// The provider answered with an error status that has no more specific kind:
        /// the error has its response and body.
        public static let httpError = Kind("HTTP error")
        /// The provider reported an error in a successful HTTP response.
        public static let providerError = Kind("Provider error")
        /// A successful answer couldn't be read as what was expected.
        public static let decodingError = Kind("Decoding error")

        // MARK: Provider status

        /// The provider rejected the credentials, with HTTP status 401 or 403.
        public static let authenticationFailed = Kind("Authentication failed")
        /// The account has no remaining quota or credit, with HTTP status 402.
        public static let quotaExceeded = Kind("Quota exceeded")
        /// The provider timed out, with HTTP status 408 or 504.
        public static let timedOut = Kind("Timed out")
        /// The provider limits the rate of requests, with HTTP status 429.
        public static let rateLimited = Kind("Rate limited")

        // MARK: Model

        /// The provider's safety system blocked the input.
        public static let guardrailViolation = Kind("Guardrail violation")
        /// The model refused to answer.
        public static let refusal = Kind("Refusal")
        /// The model could not be loaded or used.
        public static let modelUnavailable = Kind("Model unavailable")
        /// The model needs an API key and none was configured.
        public static let missingCredentials = Kind("Missing credentials")

        // MARK: Request

        /// A question is malformed.
        public static let invalidQuestion = Kind("Invalid question")
        /// The model cannot answer a well-formed question,
        /// for example because it has too many options
        /// or because an answer label has no single-token encoding.
        public static let unsupportedQuestion = Kind("Unsupported question")
        /// The model can't read the state,
        /// for example because the state has an attachment of a kind that the model does not support.
        public static let unsupportedState = Kind("Unsupported state")
        /// The request options set a value that the model does not accept.
        public static let unsupportedOption = Kind("Unsupported option")
    }

    /// What went wrong.
    public let kind: Kind

    /// A description of the error.
    public let message: String

    /// The position of the question in its batch, for an error about one question.
    public let questionIndex: Int?

    /// The provider's code for the error, such as `safety`.
    public let reason: String?

    /// The response, headers included, when the provider answered.
    ///
    /// The values of credential headers, such as `Authorization`, are replaced.
    public let response: HTTPResponse?

    /// The response's body, when it was read.
    public let body: Data?

    /// The error that caused this one, such as a decoding error's.
    public let underlyingError: (any Error)?

    /// Creates a decision error.
    ///
    /// Prefer the static factories, such as ``invalidQuestion(index:reason:)``,
    /// for the errors that they cover.
    ///
    /// - Parameters:
    ///   - kind: What went wrong.
    ///   - message: A description of the error.
    ///     If `nil`, the description is made from the kind,
    ///     the response's status code, the body, and the underlying error.
    ///   - questionIndex: The position of the question in its batch, if the error is about one question.
    ///   - reason: The provider's code for the error.
    ///   - response: The response, when the provider answered.
    ///     The values of credential headers, such as `Authorization`, are replaced.
    ///   - body: The response's body.
    ///   - underlyingError: The error that caused this one.
    public init(
        kind: Kind,
        message: String? = nil,
        questionIndex: Int? = nil,
        reason: String? = nil,
        response: HTTPResponse? = nil,
        body: Data? = nil,
        underlyingError: (any Error)? = nil
    ) {
        self.kind = kind
        self.questionIndex = questionIndex
        self.reason = reason
        self.response = response.map(Self.redactingCredentials)
        self.body = body
        self.underlyingError = underlyingError
        self.message =
            message
            ?? Self.defaultMessage(kind: kind, response: response, body: body, underlyingError: underlyingError)
    }

    public var description: String { message }

    public var errorDescription: String? { message }

    private static func defaultMessage(
        kind: Kind,
        response: HTTPResponse?,
        body: Data?,
        underlyingError: (any Error)?
    ) -> String {
        var description = kind.description
        if let response { description += " (Status \(response.status.code))" }
        if let body, let text = String(data: body, encoding: .utf8), !text.isEmpty { description += ": \(text)" }
        if let underlyingError { description += ": \(underlyingError)" }
        return description
    }
}

// MARK: - Retry

extension DecisionError {
    /// The HTTP status codes that a provider uses for a temporary failure.
    static let retryableStatusCodes: Set<Int> = [429, 503, 529]

    /// Whether the same request can succeed if it is sent again.
    ///
    /// This value is advice to callers and to models that wrap other models.
    /// It is `true` for the kinds ``Kind/rateLimited`` and ``Kind/timedOut``
    /// and for HTTP status 503 and 529.
    /// A ``RetryPolicy`` does not read it:
    /// the policy's ``RetryPolicy/retryableStatusCodes`` decide.
    public var isRetryable: Bool {
        if kind == .rateLimited || kind == .timedOut {
            return true
        }
        return response.map { Self.retryableStatusCodes.contains($0.status.code) } ?? false
    }

    /// How long the provider asks the caller to wait before the next request.
    ///
    /// This value is read from the `Retry-After` header of ``response``,
    /// in seconds or as an HTTP date measured from now.
    /// It is never negative: a date in the past gives zero.
    /// It is `nil` if there is no response, no header, or a header that can't be read.
    public var retryAfter: Duration? {
        retryAfter(now: Date())
    }

    /// Returns the `Retry-After` delay, measuring an HTTP date from `now`.
    func retryAfter(now: Date) -> Duration? {
        response?.headerFields[.retryAfter].flatMap { Self.retryAfter(from: $0, now: now) }
    }

    /// The longest `Retry-After` delay, in seconds.
    ///
    /// A longer value is reduced to this one,
    /// so that a very large header can't overflow a `Duration`.
    static let maximumRetryAfterSeconds: Double = 1e12

    /// Returns the delay in a `Retry-After` header value,
    /// or `nil` if the value is not a number of seconds or an HTTP date.
    ///
    /// - Parameters:
    ///   - value: The header value.
    ///   - now: The time against which a date is measured.
    static func retryAfter(from value: String, now: Date) -> Duration? {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        let seconds: Double
        if let number = Double(trimmed), number.isFinite {
            seconds = number
        } else {
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(identifier: "GMT")
            formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
            guard let date = formatter.date(from: trimmed) else { return nil }
            seconds = date.timeIntervalSince(now)
        }
        return .seconds(min(max(seconds, 0), maximumRetryAfterSeconds))
    }
}

// MARK: - Factories

extension DecisionError {
    /// Returns an error for a malformed question.
    ///
    /// - Parameters:
    ///   - index: The position of the question in its batch.
    ///   - reason: A description of the problem.
    public static func invalidQuestion(index: Int, reason: String) -> DecisionError {
        DecisionError(
            kind: .invalidQuestion,
            message: "Invalid question at index \(index): \(reason)",
            questionIndex: index
        )
    }

    /// Returns an error for a well-formed question that the model cannot answer,
    /// for example because it has too many options
    /// or because an answer label has no single-token encoding.
    ///
    /// - Parameters:
    ///   - index: The position of the question in its batch.
    ///   - reason: A description of the problem.
    public static func unsupportedQuestion(index: Int, reason: String) -> DecisionError {
        DecisionError(
            kind: .unsupportedQuestion,
            message: "Unsupported question at index \(index): \(reason)",
            questionIndex: index
        )
    }

    /// Returns an error for a state that the model can't read,
    /// for example because the state has an attachment of a kind that the model does not support.
    ///
    /// - Parameter reason: A description of the problem.
    public static func unsupportedState(_ reason: String) -> DecisionError {
        DecisionError(kind: .unsupportedState, message: "Unsupported state: \(reason)")
    }

    /// The error for a model that needs an API key when none was configured.
    public static let missingCredentials = DecisionError(
        kind: .missingCredentials,
        message: "No API key is configured."
    )

    /// Returns the error for an HTTP response with a status code outside 200 to 299.
    ///
    /// The status code sets the error's kind:
    ///
    /// | Status | Kind |
    /// | --- | --- |
    /// | 401, 403 | ``Kind/authenticationFailed`` |
    /// | 402 | ``Kind/quotaExceeded`` |
    /// | 408, 504 | ``Kind/timedOut`` |
    /// | 429 | ``Kind/rateLimited`` |
    /// | Other | ``Kind/httpError`` |
    ///
    /// The message is the same for every status code.
    ///
    /// - Parameters:
    ///   - statusCode: The HTTP status code, from 100 to 999.
    ///   - detail: The response body as text.
    ///   - headers: The response headers.
    ///     The values of credential headers, such as `Authorization`, are replaced.
    public static func requestFailed(
        statusCode: Int,
        detail: String,
        headers: [String: String]
    ) -> DecisionError {
        var fields = HTTPFields()
        for name in headers.keys.sorted() {
            guard let fieldName = HTTPField.Name(name), let value = headers[name] else { continue }
            fields.append(HTTPField(name: fieldName, value: value))
        }
        return DecisionError(
            response: HTTPResponse(status: .init(code: statusCode), headerFields: fields),
            body: Data(detail.utf8)
        )
    }

    /// Returns an error for a provider whose safety system blocked the input.
    ///
    /// The error has the kind ``Kind/guardrailViolation``.
    ///
    /// - Parameters:
    ///   - reason: The provider's code, such as `safety`.
    ///   - message: The provider's description, if any.
    public static func inputBlocked(reason: String, message: String?) -> DecisionError {
        let description: String
        if let message, !message.isEmpty {
            description = "The input was blocked (\(reason)): \(message)"
        } else {
            description = "The input was blocked (\(reason))."
        }
        return DecisionError(kind: .guardrailViolation, message: description, reason: reason)
    }

    /// Returns an error for a model that refused to answer.
    ///
    /// - Parameters:
    ///   - reason: The provider's code, if any.
    ///   - message: The provider's description.
    public static func refusal(reason: String? = nil, message: String) -> DecisionError {
        DecisionError(kind: .refusal, message: "The model refused to answer: \(message)", reason: reason)
    }

    /// Returns an error for request options that the model does not accept.
    ///
    /// - Parameter detail: A description of the problem.
    public static func unsupportedOption(_ detail: String) -> DecisionError {
        DecisionError(kind: .unsupportedOption, message: "Unsupported option: \(detail)")
    }

    /// Returns an error for an answer that isn't an HTTP response or doesn't match the request.
    ///
    /// - Parameter detail: A description of the problem.
    public static func invalidResponse(_ detail: String) -> DecisionError {
        DecisionError(kind: .invalidResponse, message: "Invalid response: \(detail)")
    }

    /// Returns an error for an answer to one question that doesn't match the request.
    ///
    /// The message is the same as ``invalidResponse(_:)`` gives for `detail`.
    ///
    /// - Parameters:
    ///   - detail: A description of the problem.
    ///   - questionIndex: The position of the question in its batch.
    static func invalidResponse(_ detail: String, questionIndex: Int) -> DecisionError {
        DecisionError(kind: .invalidResponse, message: "Invalid response: \(detail)", questionIndex: questionIndex)
    }

    /// Returns an error for a model that could not be loaded or used.
    ///
    /// - Parameter detail: A description of the problem.
    public static func modelUnavailable(_ detail: String) -> DecisionError {
        DecisionError(kind: .modelUnavailable, message: "Model unavailable: \(detail)")
    }
}

// MARK: - HTTP

extension DecisionError {
    /// Creates the error for an HTTP response with a status code outside 200 to 299.
    ///
    /// The status code sets the kind,
    /// as ``requestFailed(statusCode:detail:headers:)`` describes.
    init(response: HTTPResponse, body: Data) {
        let kind: Kind
        switch response.status.code {
        case 401, 403: kind = .authenticationFailed
        case 402: kind = .quotaExceeded
        case 408, 504: kind = .timedOut
        case 429: kind = .rateLimited
        default: kind = .httpError
        }
        let detail = String(data: body, encoding: .utf8) ?? "Invalid response"
        self.init(
            kind: kind,
            message: "HTTP error (status \(response.status.code)): \(detail)",
            response: response,
            body: body
        )
    }

    /// Returns the error for a request body that can't be encoded.
    static func encodingFailed(_ error: any Error) -> DecisionError {
        DecisionError(kind: .invalidRequest, message: "Invalid request: \(error)", underlyingError: error)
    }

    /// Returns the error for a successful response body that can't be decoded.
    ///
    /// The message is the one that versions 0.9 and 0.10 used.
    static func decodingFailed(_ error: any Error) -> DecisionError {
        DecisionError(kind: .decodingError, message: "Invalid response: \(error)", underlyingError: error)
    }

    /// The value that replaces a credential header value.
    static let redactedValue = "<redacted>"

    /// The names of headers that carry credentials.
    static let credentialHeaders: [HTTPField.Name] = [
        .authorization,
        .proxyAuthorization,
        HTTPField.Name("X-Api-Key")!,
        HTTPField.Name("X-Goog-Api-Key")!,
    ]

    /// Returns a response with the values of credential headers replaced.
    static func redactingCredentials(_ response: HTTPResponse) -> HTTPResponse {
        var response = response
        for name in credentialHeaders where response.headerFields[name] != nil {
            let count = response.headerFields[fields: name].count
            response.headerFields[fields: name] = Array(
                repeating: HTTPField(name: name, value: redactedValue),
                count: count
            )
        }
        return response
    }
}
