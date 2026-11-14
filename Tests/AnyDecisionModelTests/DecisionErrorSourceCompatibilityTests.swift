import Foundation
import Testing

// This file imports the module without `@testable`,
// so it compiles only against the public API, as a client does.
import AnyDecisionModel

/// Checks that code that creates a version 0.9 `DecisionError` still compiles,
/// including references to the 0.9 cases as function values,
/// and that equality checks against those values still match.
@Suite("DecisionError source compatibility")
struct DecisionErrorSourceCompatibilityTests {
    @Test func versionZeroNineThrowSitesCompile() {
        func make(_ index: Int) -> DecisionError {
            switch index {
            case 0: return .invalidQuestion(index: 1, reason: "Empty.")
            case 1: return .unsupportedQuestion(index: 2, reason: "Too many.")
            case 2: return .unsupportedState("No images.")
            case 3: return .missingCredentials
            case 4: return .requestFailed(statusCode: 500, detail: "boom", headers: ["X-Id": "a"])
            case 5: return .inputBlocked(reason: "safety", message: nil)
            case 6: return .invalidResponse("Not JSON.")
            default: return .modelUnavailable("Missing.")
            }
        }

        let codes = (0 ..< 8).map { make($0).code }
        #expect(
            codes == [
                .invalidQuestion, .unsupportedQuestion, .unsupportedState, .missingCredentials,
                .requestFailed, .inputBlocked, .invalidResponse, .modelUnavailable,
            ]
        )
    }

    @Test func versionZeroNineThrowStatementsCompile() {
        #expect(throws: DecisionError.missingCredentials) { throw DecisionError.missingCredentials }
        #expect(throws: DecisionError.invalidQuestion(index: 0, reason: "Empty.")) {
            throw DecisionError.invalidQuestion(index: 0, reason: "Empty.")
        }
        #expect(throws: DecisionError.requestFailed(statusCode: 429, detail: "", headers: [:])) {
            throw DecisionError.requestFailed(statusCode: 429, detail: "", headers: [:])
        }
        #expect(throws: DecisionError.inputBlocked(reason: "safety", message: "Blocked.")) {
            throw DecisionError.inputBlocked(reason: "safety", message: "Blocked.")
        }
        #expect(throws: DecisionError.self) { throw DecisionError.unsupportedState("x") }
    }

    @Test func versionZeroNineEqualityChecksMatch() {
        let error: any Error = DecisionError.missingCredentials
        #expect(error as? DecisionError == .missingCredentials)
        #expect(error as? DecisionError != .invalidResponse("x"))
        #expect(DecisionError.invalidResponse("x") == DecisionError.invalidResponse("x"))
        #expect(DecisionError.unsupportedState("x") != DecisionError.unsupportedState("y"))
    }

    /// References with a type annotation resolve to the 0.9 factories.
    @Test func versionZeroNineCasesWorkAsTypedFunctionValues() {
        let invalidQuestion: (Int, String) -> DecisionError = DecisionError.invalidQuestion(index:reason:)
        let unsupportedQuestion: (Int, String) -> DecisionError = DecisionError.unsupportedQuestion
        let unsupportedState: (String) -> DecisionError = DecisionError.unsupportedState
        let requestFailed: (Int, String, [String: String]) -> DecisionError =
            DecisionError.requestFailed(statusCode:detail:headers:)
        let inputBlocked: (String, String?) -> DecisionError = DecisionError.inputBlocked
        let invalidResponse: (String) -> DecisionError = DecisionError.invalidResponse(_:)
        let modelUnavailable: (String) -> DecisionError = DecisionError.modelUnavailable
        let missingCredentials: DecisionError = DecisionError.missingCredentials

        #expect(invalidQuestion(1, "a") == .invalidQuestion(index: 1, reason: "a"))
        #expect(unsupportedQuestion(1, "a") == .unsupportedQuestion(index: 1, reason: "a"))
        #expect(unsupportedState("a") == .unsupportedState("a"))
        #expect(requestFailed(404, "a", [:]) == .requestFailed(statusCode: 404, detail: "a", headers: [:]))
        #expect(inputBlocked("safety", nil) == .inputBlocked(reason: "safety", message: nil))
        #expect(invalidResponse("a") == .invalidResponse("a"))
        #expect(modelUnavailable("a") == .modelUnavailable("a"))
        #expect(missingCredentials == .missingCredentials)
        #expect(["a", "b"].map(DecisionError.invalidResponse) == [.invalidResponse("a"), .invalidResponse("b")])
    }

    /// References without a type annotation resolve to the 0.9 factories.
    @Test func versionZeroNineCasesWorkAsInferredFunctionValues() {
        let invalidQuestion = DecisionError.invalidQuestion
        let invalidQuestionByName = DecisionError.invalidQuestion(index:reason:)
        let unsupportedQuestion = DecisionError.unsupportedQuestion
        let unsupportedState = DecisionError.unsupportedState
        let requestFailed = DecisionError.requestFailed
        let inputBlocked = DecisionError.inputBlocked(reason:message:)
        let invalidResponse = DecisionError.invalidResponse
        let modelUnavailable = DecisionError.modelUnavailable
        let missingCredentials = DecisionError.missingCredentials

        #expect(invalidQuestion(1, "a") == invalidQuestionByName(1, "a"))
        #expect(unsupportedQuestion(1, "a").questionIndex == 1)
        #expect(unsupportedState("a").code == .unsupportedState)
        #expect(requestFailed(429, "a", ["Retry-After": "1"]).code == .rateLimited)
        #expect(inputBlocked("safety", "Blocked.").reason == "safety")
        #expect(invalidResponse("a").message == "Invalid response: a")
        #expect(modelUnavailable("a").message == "Model unavailable: a")
        #expect(missingCredentials.message == "No API key is configured.")
    }

    /// The replacements for 0.9 patterns that match cases.
    @Test func versionZeroTenMatchingCompiles() {
        func classify(_ error: any Error) -> String {
            do {
                throw error
            } catch let error as DecisionError where error.code == .rateLimited {
                return "rate limited \(error.response?.statusCode ?? 0) \(error.retryAfter ?? .zero)"
            } catch let error as DecisionError where error.response != nil {
                return "status \(error.response?.statusCode ?? 0)"
            } catch let error as DecisionError {
                switch error.code {
                case .invalidQuestion, .unsupportedQuestion:
                    return "question \(error.questionIndex ?? -1)"
                case .missingCredentials:
                    return "credentials"
                default:
                    return "other"
                }
            } catch {
                return "unknown"
            }
        }

        #expect(
            classify(DecisionError.requestFailed(statusCode: 429, detail: "", headers: ["Retry-After": "2"]))
                == "rate limited 429 2.0 seconds"
        )
        #expect(classify(DecisionError.requestFailed(statusCode: 500, detail: "", headers: [:])) == "status 500")
        #expect(classify(DecisionError.unsupportedQuestion(index: 3, reason: "")) == "question 3")
        #expect(classify(DecisionError.missingCredentials) == "credentials")
        #expect(classify(DecisionError.modelUnavailable("")) == "other")
        #expect(classify(CancellationError()) == "unknown")
    }
}
