import Foundation

/// An error from a decision model or session.
///
/// Cancellation is reported as `CancellationError`, not as a `DecisionError`.
public enum DecisionError: Error, Hashable, Sendable {
    /// A question is malformed.
    ///
    /// The index is the position of the question in its batch.
    case invalidQuestion(index: Int, reason: String)

    /// The model cannot answer a well-formed question,
    /// for example because it has too many options
    /// or because an answer label has no single-token encoding.
    ///
    /// The index is the position of the question in its batch.
    case unsupportedQuestion(index: Int, reason: String)

    /// The model can't read the state,
    /// for example because the state has an attachment of a kind that the model does not support.
    case unsupportedState(String)

    /// The model needs an API key and none was configured.
    case missingCredentials

    /// The server returned a status code outside 200 to 299.
    ///
    /// The headers are kept so that callers can read values such as `Retry-After`.
    case requestFailed(statusCode: Int, detail: String, headers: [String: String])

    /// The backend refused to answer questions about the state,
    /// for example because of a safety filter.
    ///
    /// The reason is the backend's code, such as `safety`.
    /// The message, if any, is the backend's description.
    case inputBlocked(reason: String, message: String?)

    /// The backend returned a response that does not match the request.
    case invalidResponse(String)

    /// The model could not be loaded or used.
    case modelUnavailable(String)
}

// MARK: - LocalizedError, CustomStringConvertible

extension DecisionError: LocalizedError, CustomStringConvertible {
    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .invalidQuestion(let index, let reason):
            return "Invalid question at index \(index): \(reason)"
        case .unsupportedQuestion(let index, let reason):
            return "Unsupported question at index \(index): \(reason)"
        case .unsupportedState(let reason):
            return "Unsupported state: \(reason)"
        case .missingCredentials:
            return "No API key is configured."
        case .requestFailed(let statusCode, let detail, _):
            return "HTTP error (status \(statusCode)): \(detail)"
        case .inputBlocked(let reason, let message):
            if let message, !message.isEmpty {
                return "The input was blocked (\(reason)): \(message)"
            }
            return "The input was blocked (\(reason))."
        case .invalidResponse(let detail):
            return "Invalid response: \(detail)"
        case .modelUnavailable(let detail):
            return "Model unavailable: \(detail)"
        }
    }
}
