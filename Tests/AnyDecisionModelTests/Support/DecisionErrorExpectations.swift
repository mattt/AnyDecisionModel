import HTTPTypes
import Testing

@testable import AnyDecisionModel

/// Returns whether an error is a decision error with the kind, message, and question index of another.
///
/// `DecisionError` is not `Equatable`, because its underlying error can be any error,
/// so tests compare these fields.
func isDecisionError(_ error: any Error, like expected: DecisionError) -> Bool {
    guard let error = error as? DecisionError else { return false }
    return error.kind == expected.kind
        && error.message == expected.message
        && error.questionIndex == expected.questionIndex
}

/// Expects `body` to throw a decision error with the kind, message, and question index of `expected`.
func expectDecisionError<R>(
    _ expected: DecisionError,
    sourceLocation: SourceLocation = #_sourceLocation,
    performing body: () throws -> R
) {
    #expect(sourceLocation: sourceLocation) {
        try body()
    } throws: { error in
        isDecisionError(error, like: expected)
    }
}

/// Expects `body` to throw a decision error with the kind, message, and question index of `expected`.
func expectDecisionError<R>(
    _ expected: DecisionError,
    sourceLocation: SourceLocation = #_sourceLocation,
    performing body: () async throws -> R
) async {
    await #expect(sourceLocation: sourceLocation) {
        try await body()
    } throws: { error in
        isDecisionError(error, like: expected)
    }
}

extension DecisionError {
    /// The response body as text.
    var bodyText: String? {
        body.flatMap { String(data: $0, encoding: .utf8) }
    }

    /// Returns the value of a response header.
    func header(_ name: String) -> String? {
        HTTPField.Name(name).flatMap { response?.headerFields[$0] }
    }
}
