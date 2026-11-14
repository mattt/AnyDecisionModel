import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// An error from a decision model or session.
///
/// Every error has a ``code`` that names the kind of failure
/// and a ``message`` that describes it.
/// The other fields are set when they apply:
/// ``questionIndex`` for an error about one question,
/// ``reason`` for a provider's code,
/// ``retryAfter`` for a server that asks the caller to wait,
/// and ``response`` for an HTTP failure.
///
/// Match an error by its code:
///
/// ```swift
/// do {
///     let urgent = try await session.probability(of: "Does this convey urgency?")
/// } catch let error as DecisionError where error.code == .rateLimited {
///     print("Retry after \(error.retryAfter ?? .seconds(1)).")
/// }
/// ```
///
/// Throw sites use the static factories,
/// such as ``invalidQuestion(index:reason:)`` and ``missingCredentials``.
/// Two errors are equal when all of their fields are equal,
/// so `#expect(throws: DecisionError.missingCredentials)` matches the thrown error.
///
/// Cancellation is reported as `CancellationError`, not as a `DecisionError`.
///
/// No error contains an API key.
/// HTTP errors replace the values of credential headers,
/// such as `Authorization`,
/// and remove the request's credentials from the response body.
///
/// ## Migration from version 0.9
///
/// In version 0.9, `DecisionError` is an enum.
/// Code that creates an error compiles unchanged,
/// because each 0.9 case is now a static factory or property with the same message.
/// Code that matches cases does not compile:
///
/// - Replace `catch DecisionError.requestFailed(let status, _, _)`
///   with `catch let error as DecisionError where error.response != nil`,
///   and read `error.response?.statusCode`.
///   An HTTP failure can now have the code
///   ``Code/authenticationFailed``, ``Code/quotaExceeded``, ``Code/timedOut``,
///   ``Code/rateLimited``, or ``Code/requestFailed``.
/// - Replace `catch DecisionError.invalidResponse(let detail)`
///   and similar patterns for the other cases
///   with `catch let error as DecisionError where error.code == .invalidResponse`,
///   and read ``message``.
/// - Replace `if case .unsupportedQuestion(let index, _) = error`
///   with a check of ``code`` and ``questionIndex``.
/// - Replace `switch error` over the cases with `switch error.code`
///   and add `@unknown default`.
/// - Read `Retry-After` from ``retryAfter``
///   in place of `headers["retry-after"]`.
/// - Catch `DecisionError` with the code ``Code/timedOut``
///   or ``Code/transportFailed`` in place of `URLError`.
///   A cancelled request still throws `CancellationError`.
/// - Catch `DecisionError` with the code ``Code/unsupportedOption``
///   in place of ``SystemOneDecisionModel/ConfigurationError``
///   for a request whose ``SystemOneDecisionModel/CustomOptions`` set a reserved key.
public struct DecisionError: Error, Hashable, Sendable, Codable {
    /// A kind of decision error.
    ///
    /// New codes can be added in a later version.
    #if hasAttribute(nonexhaustive)
        @nonexhaustive
    #endif
    public enum Code: String, Hashable, Sendable, Codable {
        /// A question is malformed.
        case invalidQuestion

        /// The model cannot answer a well-formed question,
        /// for example because it has too many options
        /// or because an answer label has no single-token encoding.
        case unsupportedQuestion

        /// The model can't read the state,
        /// for example because the state has an attachment of a kind that the model does not support.
        case unsupportedState

        /// The model needs an API key and none was configured.
        case missingCredentials

        /// The server rejected the credentials, with HTTP status 401 or 403.
        case authenticationFailed

        /// The server limits the rate of requests, with HTTP status 429.
        ///
        /// ``DecisionError/retryAfter`` is the delay from the `Retry-After` header, if any.
        case rateLimited

        /// The account has no remaining quota or credit, with HTTP status 402.
        case quotaExceeded

        /// The request timed out,
        /// in the URL session or with HTTP status 408 or 504.
        case timedOut

        /// The request failed before the server sent a response,
        /// for example because the connection was lost.
        case transportFailed

        /// The server returned another status code outside 200 to 299.
        case requestFailed

        /// The backend refused to answer questions about the state,
        /// for example because of a safety filter.
        case inputBlocked

        /// The model refused to answer.
        case refused

        /// The request options set a value that the model does not accept.
        case unsupportedOption

        /// The backend returned a response that does not match the request.
        case invalidResponse

        /// The model could not be loaded or used.
        case modelUnavailable
    }

    /// An HTTP response with a status code outside 200 to 299.
    public struct HTTPResponse: Hashable, Sendable, Codable {
        /// The HTTP status code.
        public var statusCode: Int

        /// The response headers, with lowercase names.
        ///
        /// The values of credential headers, such as `Authorization`, are replaced.
        public var headers: [String: String]

        /// The response body as text.
        public var body: String

        /// Creates an HTTP response.
        ///
        /// - Parameters:
        ///   - statusCode: The HTTP status code.
        ///   - headers: The response headers.
        ///     Names are stored in lowercase,
        ///     and the values of credential headers are replaced.
        ///   - body: The response body as text.
        public init(statusCode: Int, headers: [String: String] = [:], body: String = "") {
            self.statusCode = statusCode
            self.headers = Self.redacting(headers)
            self.body = body
        }

        private enum CodingKeys: String, CodingKey {
            case statusCode, headers, body
        }

        /// Decodes a response and replaces the values of credential headers.
        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                statusCode: try container.decode(Int.self, forKey: .statusCode),
                headers: try container.decode([String: String].self, forKey: .headers),
                body: try container.decode(String.self, forKey: .body)
            )
        }

        /// The value that replaces a credential header value.
        static let redactedValue = "<redacted>"

        /// The lowercase names of headers that carry credentials.
        static let credentialHeaders: Set<String> = [
            "authorization",
            "proxy-authorization",
            "x-api-key",
            "x-goog-api-key",
        ]

        /// Returns headers with lowercase names and credential values replaced.
        static func redacting(_ headers: [String: String]) -> [String: String] {
            var result: [String: String] = [:]
            for (name, value) in headers {
                let name = name.lowercased()
                result[name] = credentialHeaders.contains(name) ? redactedValue : value
            }
            return result
        }
    }

    /// The kind of error.
    public var code: Code

    /// A description of the error.
    public var message: String

    /// The position of the question in its batch, for an error about one question.
    public var questionIndex: Int?

    /// The provider's code for the error, such as `safety`.
    public var reason: String?

    /// Whether the same request can succeed if it is sent again.
    ///
    /// This value is the backend's advice to callers and to models that wrap other models.
    /// It is `true` for HTTP status 429, 503, and 529,
    /// and for the codes ``Code/timedOut`` and ``Code/transportFailed``,
    /// unless the transport failure can't change, such as a malformed URL.
    /// A ``RetryPolicy`` does not read it:
    /// the policy's ``RetryPolicy/retryableStatusCodes`` decide.
    public var isRetryable: Bool

    /// How long the server asks the caller to wait before the next request.
    ///
    /// This value comes from the `Retry-After` header,
    /// in seconds or as an HTTP date,
    /// and is never negative.
    /// A date in the past gives zero.
    public var retryAfter: Duration?

    /// The HTTP response, for an error from a server's status code.
    public var response: HTTPResponse?

    /// Creates a decision error.
    ///
    /// Prefer the static factories, such as ``invalidQuestion(index:reason:)``,
    /// for the errors that they cover.
    ///
    /// - Parameters:
    ///   - code: The kind of error.
    ///   - message: A description of the error.
    ///   - questionIndex: The position of the question in its batch, if the error is about one question.
    ///   - reason: The provider's code for the error.
    ///   - isRetryable: Whether the same request can succeed if it is sent again.
    ///     If `nil`, the value is `true` for ``Code/rateLimited``, ``Code/timedOut``,
    ///     ``Code/transportFailed``, and HTTP status 503 and 529.
    ///   - retryAfter: How long the server asks the caller to wait.
    ///   - response: The HTTP response.
    public init(
        code: Code,
        message: String,
        questionIndex: Int? = nil,
        reason: String? = nil,
        isRetryable: Bool? = nil,
        retryAfter: Duration? = nil,
        response: HTTPResponse? = nil
    ) {
        self.code = code
        self.message = message
        self.questionIndex = questionIndex
        self.reason = reason
        self.isRetryable =
            isRetryable ?? Self.isRetryableByDefault(code: code, statusCode: response?.statusCode)
        self.retryAfter = retryAfter
        self.response = response
    }

    /// The HTTP status codes that a server uses for a temporary failure.
    static let retryableStatusCodes: Set<Int> = [429, 503, 529]

    private static func isRetryableByDefault(code: Code, statusCode: Int?) -> Bool {
        switch code {
        case .rateLimited, .timedOut, .transportFailed:
            return true
        default:
            return statusCode.map(retryableStatusCodes.contains) ?? false
        }
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
            code: .invalidQuestion,
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
            code: .unsupportedQuestion,
            message: "Unsupported question at index \(index): \(reason)",
            questionIndex: index
        )
    }

    /// Returns an error for a state that the model can't read,
    /// for example because the state has an attachment of a kind that the model does not support.
    ///
    /// - Parameter reason: A description of the problem.
    public static func unsupportedState(_ reason: String) -> DecisionError {
        DecisionError(code: .unsupportedState, message: "Unsupported state: \(reason)")
    }

    /// The error for a model that needs an API key when none was configured.
    public static let missingCredentials = DecisionError(
        code: .missingCredentials,
        message: "No API key is configured."
    )

    /// Returns the error for an HTTP response with a status code outside 200 to 299.
    ///
    /// The status code sets the error's code:
    ///
    /// | Status | Code |
    /// | --- | --- |
    /// | 401, 403 | ``Code/authenticationFailed`` |
    /// | 402 | ``Code/quotaExceeded`` |
    /// | 408, 504 | ``Code/timedOut`` |
    /// | 429 | ``Code/rateLimited`` |
    /// | Other | ``Code/requestFailed`` |
    ///
    /// The error keeps the status code, headers, and body in ``response``,
    /// and ``retryAfter`` holds the delay from a `Retry-After` header with any status code.
    /// The message is the same for every status code.
    ///
    /// - Parameters:
    ///   - statusCode: The HTTP status code.
    ///   - detail: The response body as text.
    ///   - headers: The response headers.
    ///     The values of credential headers, such as `Authorization`, are replaced.
    public static func requestFailed(
        statusCode: Int,
        detail: String,
        headers: [String: String]
    ) -> DecisionError {
        DecisionError(statusCode: statusCode, detail: detail, headers: headers, now: Date())
    }

    /// Returns an error for a backend that refused to answer questions about the state,
    /// for example because of a safety filter.
    ///
    /// - Parameters:
    ///   - reason: The backend's code, such as `safety`.
    ///   - message: The backend's description, if any.
    public static func inputBlocked(reason: String, message: String?) -> DecisionError {
        let description: String
        if let message, !message.isEmpty {
            description = "The input was blocked (\(reason)): \(message)"
        } else {
            description = "The input was blocked (\(reason))."
        }
        return DecisionError(code: .inputBlocked, message: description, reason: reason)
    }

    /// Returns an error for a model that refused to answer.
    ///
    /// - Parameters:
    ///   - reason: The backend's code, if any.
    ///   - message: The backend's description.
    public static func refused(reason: String? = nil, message: String) -> DecisionError {
        DecisionError(code: .refused, message: "The model refused to answer: \(message)", reason: reason)
    }

    /// Returns an error for request options that the model does not accept.
    ///
    /// - Parameter detail: A description of the problem.
    public static func unsupportedOption(_ detail: String) -> DecisionError {
        DecisionError(code: .unsupportedOption, message: "Unsupported option: \(detail)")
    }

    /// Returns an error for a response that does not match the request.
    ///
    /// - Parameter detail: A description of the problem.
    public static func invalidResponse(_ detail: String) -> DecisionError {
        DecisionError(code: .invalidResponse, message: "Invalid response: \(detail)")
    }

    /// Returns an error for a model that could not be loaded or used.
    ///
    /// - Parameter detail: A description of the problem.
    public static func modelUnavailable(_ detail: String) -> DecisionError {
        DecisionError(code: .modelUnavailable, message: "Model unavailable: \(detail)")
    }
}

// MARK: - HTTP and URL errors

extension DecisionError {
    /// Creates the error for an HTTP response with a status code outside 200 to 299.
    ///
    /// - Parameter now: The time against which a `Retry-After` date is measured.
    init(statusCode: Int, detail: String, headers: [String: String], now: Date) {
        let response = HTTPResponse(statusCode: statusCode, headers: headers, body: detail)
        let code: Code
        switch statusCode {
        case 401, 403: code = .authenticationFailed
        case 402: code = .quotaExceeded
        case 408, 504: code = .timedOut
        case 429: code = .rateLimited
        default: code = .requestFailed
        }
        self.init(
            code: code,
            message: "HTTP error (status \(statusCode)): \(detail)",
            retryAfter: response.headers["retry-after"].flatMap { Self.retryAfter(from: $0, now: now) },
            response: response
        )
    }

    /// Creates the error for a URL session failure other than cancellation.
    ///
    /// `URLError.timedOut` becomes ``Code/timedOut``,
    /// and every other code becomes ``Code/transportFailed``.
    init(_ error: URLError) {
        if error.code == .timedOut {
            self.init(code: .timedOut, message: "The request timed out.")
            return
        }
        self.init(
            code: .transportFailed,
            message: "Transport error (URLError \(error.code.rawValue)): \(error.localizedDescription)",
            isRetryable: !Self.permanentURLErrorCodes.contains(error.code)
        )
    }

    /// The URL error codes for a request that fails the same way every time.
    private static let permanentURLErrorCodes: Set<URLError.Code> = [
        .badURL,
        .unsupportedURL,
        .appTransportSecurityRequiresSecureConnection,
    ]

    /// The longest `Retry-After` delay that an error keeps, in seconds.
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

// MARK: - LocalizedError, CustomStringConvertible

extension DecisionError: LocalizedError, CustomStringConvertible {
    public var errorDescription: String? { message }

    public var description: String { message }
}
