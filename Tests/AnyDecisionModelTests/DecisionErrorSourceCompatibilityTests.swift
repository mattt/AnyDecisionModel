import Foundation
import Testing

// This file imports the module without `@testable`,
// so it compiles only against the public API, as a client does.
import AnyDecisionModel

/// Checks that code that creates a version 0.9 `DecisionError` still compiles,
/// including references to the 0.9 cases as function values,
/// and that the 0.9 spellings of `DecisionKind` and `ProbabilitySource` still compile.
/// Version 0.10 has the same cases and spellings,
/// so these checks cover it too.
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

        let kinds = (0 ..< 8).map { make($0).kind }
        #expect(
            kinds == [
                .invalidQuestion, .unsupportedQuestion, .unsupportedState, .missingCredentials,
                .httpError, .guardrailViolation, .invalidResponse, .modelUnavailable,
            ]
        )
    }

    @Test func versionZeroNineThrowStatementsCompile() {
        #expect(throws: DecisionError.self) { throw DecisionError.missingCredentials }
        #expect(throws: DecisionError.self) { throw DecisionError.invalidQuestion(index: 0, reason: "Empty.") }
        #expect(throws: DecisionError.self) {
            throw DecisionError.requestFailed(statusCode: 429, detail: "", headers: [:])
        }
        #expect(throws: DecisionError.self) { throw DecisionError.inputBlocked(reason: "safety", message: "Blocked.") }
        #expect(throws: DecisionError.self) { throw DecisionError.unsupportedState("x") }
    }

    /// The replacement for `#expect(throws: DecisionError.missingCredentials)`,
    /// which no longer compiles because `DecisionError` is not `Equatable`.
    @Test func kindComparisonReplacesEquality() {
        #expect {
            throw DecisionError.missingCredentials
        } throws: { error in
            (error as? DecisionError)?.kind == .missingCredentials
        }
        let error: any Error = DecisionError.invalidResponse("x")
        #expect((error as? DecisionError)?.kind == .invalidResponse)
        #expect((error as? DecisionError)?.message == DecisionError.invalidResponse("x").message)
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

        #expect(invalidQuestion(1, "a").questionIndex == 1)
        #expect(unsupportedQuestion(1, "a").kind == .unsupportedQuestion)
        #expect(unsupportedState("a").message == "Unsupported state: a")
        #expect(requestFailed(404, "a", [:]).response?.status.code == 404)
        #expect(inputBlocked("safety", nil).reason == "safety")
        #expect(invalidResponse("a").kind == .invalidResponse)
        #expect(modelUnavailable("a").kind == .modelUnavailable)
        #expect(missingCredentials.kind == .missingCredentials)
        #expect(
            ["a", "b"].map(DecisionError.invalidResponse).map(\.message) == [
                "Invalid response: a", "Invalid response: b",
            ]
        )
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

        #expect(invalidQuestion(1, "a").message == invalidQuestionByName(1, "a").message)
        #expect(unsupportedQuestion(1, "a").questionIndex == 1)
        #expect(unsupportedState("a").kind == .unsupportedState)
        #expect(requestFailed(429, "a", ["Retry-After": "1"]).kind == .rateLimited)
        #expect(inputBlocked("safety", "Blocked.").reason == "safety")
        #expect(invalidResponse("a").message == "Invalid response: a")
        #expect(modelUnavailable("a").message == "Model unavailable: a")
        #expect(missingCredentials.message == "No API key is configured.")
    }

    /// The replacements for 0.9 patterns that match cases.
    @Test func versionZeroElevenMatchingCompiles() {
        func classify(_ error: any Error) -> String {
            do {
                throw error
            } catch let error as DecisionError where error.kind == .rateLimited {
                return "rate limited \(error.response?.status.code ?? 0) \(error.retryAfter ?? .zero)"
            } catch let error as DecisionError where error.response != nil {
                return "status \(error.response?.status.code ?? 0)"
            } catch let error as DecisionError {
                switch error.kind {
                case .invalidQuestion, .unsupportedQuestion:
                    return "question \(error.questionIndex ?? -1)"
                case .missingCredentials:
                    return "credentials"
                default:
                    return "other"
                }
            } catch let error as URLError {
                return "transport \(error.code.rawValue)"
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
        #expect(classify(URLError(.timedOut)) == "transport \(URLError.Code.timedOut.rawValue)")
        #expect(classify(CancellationError()) == "unknown")
    }

    /// Version 0.9 spellings of the kinds and sources compile,
    /// including `switch` statements with a `default` case and `rawValue`.
    @Test func versionZeroNineKindAndSourceSpellingsCompile() {
        func name(_ kind: DecisionKind) -> String {
            switch kind {
            case .binary: return "yes or no"
            case .choice: return "one of many"
            case .score: return "a level"
            default: return kind.rawValue
            }
        }
        #expect([DecisionKind.binary, .choice, .score].map(name) == ["yes or no", "one of many", "a level"])
        #expect(Question.binary(instructions: "A?").kind == .binary)
        #expect(DecisionKind(rawValue: "score") == .score)
        let kinds: Set<DecisionKind> = [.binary, .choice]
        #expect(DecisionCapabilities(questionKinds: kinds).questionKinds.contains(.choice))

        let source: DecisionCapabilities.ProbabilitySource = .native
        #expect(source.rawValue == "native")
        #expect(DecisionCapabilities.ProbabilitySource(rawValue: "estimated") == .estimated)
        #expect(DecisionCapabilities(probabilitySource: .unspecified).probabilitySource == .unspecified)
    }
}
