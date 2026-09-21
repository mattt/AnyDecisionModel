import Foundation

/// The input that a decision session asks questions about.
///
/// A state is immutable for the lifetime of a session.
/// Local models can reuse the computation for the state across questions.
public enum DecisionState: Hashable, Sendable {
    /// A text state.
    case text(String)

    /// A structured state, such as a JSON object or array.
    case json(JSONValue)

    /// The state as a JSON value.
    var jsonValue: JSONValue {
        switch self {
        case .text(let text):
            return .string(text)
        case .json(let value):
            return value
        }
    }
}

// MARK: - Codable

extension DecisionState: Codable {
    public init(from decoder: any Decoder) throws {
        let value = try JSONValue(from: decoder)
        if case .string(let text) = value {
            self = .text(text)
        } else {
            self = .json(value)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        try jsonValue.encode(to: encoder)
    }
}

// MARK: - ExpressibleByStringLiteral

extension DecisionState: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self = .text(value)
    }
}
