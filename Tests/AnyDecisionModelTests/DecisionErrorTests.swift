import Foundation
import HTTPTypes
import Testing

@testable import AnyDecisionModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

@Suite("DecisionError")
struct DecisionErrorTests {
    // MARK: - Status codes

    @Test(
        arguments: [
            (401, DecisionError.Kind.authenticationFailed),
            (403, .authenticationFailed),
            (402, .quotaExceeded),
            (408, .timedOut),
            (504, .timedOut),
            (429, .rateLimited),
            (300, .httpError),
            (400, .httpError),
            (404, .httpError),
            (409, .httpError),
            (422, .httpError),
            (500, .httpError),
            (502, .httpError),
            (503, .httpError),
            (529, .httpError),
        ]
    )
    func mapsStatusCode(statusCode: Int, kind: DecisionError.Kind) {
        let error = DecisionError.requestFailed(
            statusCode: statusCode,
            detail: #"{"error": "nope"}"#,
            headers: ["X-Request-Id": "abc"]
        )

        #expect(error.kind == kind)
        #expect(error.message == #"HTTP error (status \#(statusCode)): {"error": "nope"}"#)
        #expect(error.response?.status.code == statusCode)
        #expect(error.bodyText == #"{"error": "nope"}"#)
        #expect(error.header("x-request-id") == "abc")
        #expect(error.questionIndex == nil)
        #expect(error.reason == nil)
        #expect(error.underlyingError == nil)
        #expect(error.retryAfter == nil)
    }

    @Test func everyStatusCodeKeepsTheResponse() {
        for statusCode in 100 ..< 600 {
            let error = DecisionError.requestFailed(statusCode: statusCode, detail: "body", headers: [:])
            #expect(error.response?.status.code == statusCode)
            #expect(error.body == Data("body".utf8))
            #expect(error.message == "HTTP error (status \(statusCode)): body")
        }
    }

    // MARK: - Retry-After

    @Test func rateLimitReadsRetryAfterSeconds() {
        let error = DecisionError.requestFailed(statusCode: 429, detail: "", headers: ["Retry-After": "7"])
        #expect(error.kind == .rateLimited)
        #expect(error.retryAfter == .seconds(7))
        #expect(error.header("retry-after") == "7")
    }

    @Test func retryAfterReadsFractionalSeconds() {
        #expect(DecisionError.retryAfter(from: " 1.5 ", now: Date()) == .milliseconds(1_500))
    }

    @Test func retryAfterReadsHTTPDate() {
        let now = Date(timeIntervalSince1970: 1_445_412_480)
        let error = DecisionError.requestFailed(
            statusCode: 429,
            detail: "",
            headers: ["Retry-After": "Wed, 21 Oct 2015 07:28:05 GMT"]
        )
        #expect(error.retryAfter(now: now) == .seconds(5))
        // A date that has passed asks for no wait.
        #expect(error.retryAfter == .zero)
    }

    @Test func retryAfterInThePastIsZero() {
        let now = Date(timeIntervalSince1970: 1_445_412_480)
        #expect(DecisionError.retryAfter(from: "Wed, 21 Oct 2015 07:27:00 GMT", now: now) == .zero)
        #expect(DecisionError.retryAfter(from: "-3", now: now) == .zero)
    }

    @Test(arguments: ["soon", "", "nan", "inf", "Wednesday"])
    func invalidRetryAfterIsNil(value: String) {
        #expect(DecisionError.retryAfter(from: value, now: Date()) == nil)
        let error = DecisionError.requestFailed(statusCode: 429, detail: "", headers: ["Retry-After": value])
        #expect(error.retryAfter == nil)
    }

    @Test func largeRetryAfterIsLimited() {
        let retryAfter = DecisionError.retryAfter(from: "1e300", now: Date())
        #expect(retryAfter == .seconds(DecisionError.maximumRetryAfterSeconds))
    }

    /// A `Retry-After` header on a status other than 429, such as 503, is read,
    /// because the retry policy honors it for every retryable status code.
    @Test func otherStatusCodesReadRetryAfter() {
        let error = DecisionError.requestFailed(statusCode: 503, detail: "", headers: ["Retry-After": "2"])
        #expect(error.kind == .httpError)
        #expect(error.retryAfter == .seconds(2))
    }

    @Test func errorWithoutResponseHasNoRetryAfter() {
        #expect(DecisionError(kind: .rateLimited).retryAfter == nil)
    }

    // MARK: - isRetryable

    @Test func isRetryableForTemporaryFailures() {
        for statusCode in 100 ..< 600 {
            let error = DecisionError.requestFailed(statusCode: statusCode, detail: "", headers: [:])
            let expected = [408, 429, 503, 504, 529].contains(statusCode)
            #expect(error.isRetryable == expected, "status \(statusCode)")
        }
        #expect(DecisionError(kind: .rateLimited).isRetryable)
        #expect(DecisionError(kind: .timedOut).isRetryable)
        #expect(!DecisionError(kind: .httpError).isRetryable)
        #expect(!DecisionError.missingCredentials.isRetryable)
        #expect(!DecisionError.invalidResponse("x").isRetryable)
        #expect(!DecisionError.inputBlocked(reason: "safety", message: nil).isRetryable)
    }

    // MARK: - Kinds

    @Test func kindDescriptions() {
        let kinds: [(DecisionError.Kind, String)] = [
            (.invalidRequest, "Invalid request"),
            (.invalidResponse, "Invalid response"),
            (.httpError, "HTTP error"),
            (.providerError, "Provider error"),
            (.decodingError, "Decoding error"),
            (.authenticationFailed, "Authentication failed"),
            (.quotaExceeded, "Quota exceeded"),
            (.timedOut, "Timed out"),
            (.rateLimited, "Rate limited"),
            (.guardrailViolation, "Guardrail violation"),
            (.refusal, "Refusal"),
            (.modelUnavailable, "Model unavailable"),
            (.missingCredentials, "Missing credentials"),
            (.invalidQuestion, "Invalid question"),
            (.unsupportedQuestion, "Unsupported question"),
            (.unsupportedState, "Unsupported state"),
            (.unsupportedOption, "Unsupported option"),
        ]
        for (kind, description) in kinds {
            #expect(kind.description == description)
        }
        #expect(Set(kinds.map(\.0)).count == kinds.count)
    }

    // MARK: - Messages and descriptions

    @Test func versionZeroNineMessages() {
        let cases: [(DecisionError, DecisionError.Kind, String)] = [
            (.invalidQuestion(index: 2, reason: "Empty."), .invalidQuestion, "Invalid question at index 2: Empty."),
            (
                .unsupportedQuestion(index: 1, reason: "Too many."), .unsupportedQuestion,
                "Unsupported question at index 1: Too many."
            ),
            (.unsupportedState("No images."), .unsupportedState, "Unsupported state: No images."),
            (.missingCredentials, .missingCredentials, "No API key is configured."),
            (
                .requestFailed(statusCode: 500, detail: "boom", headers: [:]), .httpError,
                "HTTP error (status 500): boom"
            ),
            (
                .inputBlocked(reason: "safety", message: "Blocked."), .guardrailViolation,
                "The input was blocked (safety): Blocked."
            ),
            (.inputBlocked(reason: "safety", message: nil), .guardrailViolation, "The input was blocked (safety)."),
            (.inputBlocked(reason: "safety", message: ""), .guardrailViolation, "The input was blocked (safety)."),
            (.invalidResponse("Not JSON."), .invalidResponse, "Invalid response: Not JSON."),
            (.modelUnavailable("Missing."), .modelUnavailable, "Model unavailable: Missing."),
        ]
        for (error, kind, message) in cases {
            #expect(error.kind == kind)
            #expect(error.message == message)
            #expect(error.description == message)
            #expect(error.errorDescription == message)
            #expect(String(describing: error) == message)
            #expect((error as any Error).localizedDescription == message)
        }
    }

    @Test func defaultMessageFollowsAnyLanguageModel() {
        let underlying = URLError(.cannotDecodeContentData)
        #expect(DecisionError(kind: .providerError).message == "Provider error")
        #expect(
            DecisionError(
                kind: .httpError,
                response: HTTPResponse(status: .badGateway),
                body: Data("upstream".utf8)
            ).message == "HTTP error (Status 502): upstream"
        )
        #expect(
            DecisionError(kind: .decodingError, underlyingError: underlying).message
                == "Decoding error: \(underlying as any Error)"
        )
    }

    @Test func factoriesSetFields() {
        let invalid = DecisionError.invalidQuestion(index: 3, reason: "Empty.")
        #expect(invalid.kind == .invalidQuestion)
        #expect(invalid.questionIndex == 3)

        let unsupported = DecisionError.unsupportedQuestion(index: 4, reason: "Too many.")
        #expect(unsupported.kind == .unsupportedQuestion)
        #expect(unsupported.questionIndex == 4)

        let blocked = DecisionError.inputBlocked(reason: "safety", message: "Blocked.")
        #expect(blocked.kind == .guardrailViolation)
        #expect(blocked.reason == "safety")

        let refusal = DecisionError.refusal(reason: "content_filter", message: "No.")
        #expect(refusal.kind == .refusal)
        #expect(refusal.reason == "content_filter")
        #expect(refusal.message == "The model refused to answer: No.")

        let option = DecisionError.unsupportedOption("No seed.")
        #expect(option.kind == .unsupportedOption)
        #expect(option.message == "Unsupported option: No seed.")
    }

    @Test func codingFailuresKeepTheUnderlyingError() {
        let underlying = URLError(.cannotDecodeContentData)
        let decoding = DecisionError.decodingFailed(underlying)
        #expect(decoding.kind == .decodingError)
        #expect(decoding.message == "Invalid response: \(underlying as any Error)")
        #expect((decoding.underlyingError as? URLError)?.code == .cannotDecodeContentData)

        let encoding = DecisionError.encodingFailed(underlying)
        #expect(encoding.kind == .invalidRequest)
        #expect((encoding.underlyingError as? URLError)?.code == .cannotDecodeContentData)
    }

    // MARK: - Credentials

    @Test func requestFailedRedactsCredentialHeaders() {
        let error = DecisionError.requestFailed(
            statusCode: 401,
            detail: "Unauthorized",
            headers: [
                "Authorization": "Bearer sk-secret",
                "Proxy-Authorization": "Basic sk-secret",
                "X-Api-Key": "sk-secret",
                "x-goog-api-key": "sk-secret",
                "X-Request-Id": "abc",
            ]
        )
        #expect(error.header("authorization") == "<redacted>")
        #expect(error.header("proxy-authorization") == "<redacted>")
        #expect(error.header("x-api-key") == "<redacted>")
        #expect(error.header("x-goog-api-key") == "<redacted>")
        #expect(error.header("x-request-id") == "abc")
        #expect(!String(reflecting: error).contains("sk-secret"))
    }

    @Test func initializerRedactsCredentialHeaders() {
        var fields = HTTPFields()
        fields[.authorization] = "Bearer sk-secret"
        fields.append(HTTPField(name: .authorization, value: "Bearer sk-other"))
        let error = DecisionError(kind: .httpError, response: HTTPResponse(status: .unauthorized, headerFields: fields))
        #expect(error.response?.headerFields[fields: .authorization].map(\.value) == ["<redacted>", "<redacted>"])
    }

    @Test func redactsCredentialsInBody() {
        let headers = ["Authorization": "Bearer sk-secret", "Accept": "application/json"]
        func redact(_ text: String, _ headers: [String: String]) -> String {
            String(decoding: HTTP.redactingCredentials(in: Data(text.utf8), requestHeaders: headers), as: UTF8.self)
        }
        #expect(
            redact("Bad key: sk-secret (Bearer sk-secret) sk-secret", headers)
                == "Bad key: <redacted> (<redacted>) <redacted>"
        )
        #expect(redact("application/json", headers) == "application/json")
        #expect(redact("key g-123", ["x-goog-api-key": "g-123"]) == "key <redacted>")
        #expect(redact("", headers) == "")
    }

    @Test func systemOneErrorsDoNotContainTheAPIKey() async throws {
        let apiKey = "sk-live-0123456789"
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: #"{"error": "Incorrect API key provided: \#(apiKey)"}"#,
            statusCode: 401,
            headers: ["Authorization": "Bearer \(apiKey)", "WWW-Authenticate": "Bearer"]
        )
        let model = try SystemOneDecisionModel(
            baseURL: host.baseURL,
            modelID: "m",
            apiKey: apiKey,
            session: StubURLProtocol.makeSession(),
            retryPolicy: .never
        )
        let session = DecisionSession(model: model, state: "state")

        do {
            _ = try await session.probability(of: "Question?")
            Issue.record("Expected an error.")
        } catch let error as DecisionError {
            #expect(error.kind == .authenticationFailed)
            expectNoSecret(apiKey, in: error)
            #expect(error.bodyText == #"{"error": "Incorrect API key provided: <redacted>"}"#)
        }
        #expect(host.recorded.first?.headers["Authorization"] == "Bearer \(apiKey)")
    }

    @Test func geminiErrorsDoNotContainTheAPIKey() async throws {
        let apiKey = "AIza-0123456789"
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: #"{"error": {"code": 400, "message": "API key \#(apiKey) is not valid."}}"#,
            statusCode: 400,
            headers: ["X-Goog-Api-Key": apiKey]
        )
        let model = GeminiDecisionModel(baseURL: host.baseURL, apiKey: apiKey, session: StubURLProtocol.makeSession())
        let session = DecisionSession(model: model, state: "state")

        do {
            _ = try await session.probability(of: "Question?")
            Issue.record("Expected an error.")
        } catch let error as DecisionError {
            #expect(error.kind == .httpError)
            expectNoSecret(apiKey, in: error)
        }
    }

    private func expectNoSecret(_ secret: String, in error: DecisionError) {
        #expect(!error.message.contains(secret))
        #expect(!error.description.contains(secret))
        #expect(!(error as any Error).localizedDescription.contains(secret))
        #expect(!String(reflecting: error).contains(secret))
        #expect(error.response?.headerFields.contains { $0.value.contains(secret) } == false)
        #expect(error.bodyText?.contains(secret) == false)
    }
}

// MARK: - Open kinds

@Suite("DecisionKind and ProbabilitySource")
struct OpenKindTests {
    @Test func decisionKindNamesAndDescriptions() {
        #expect([DecisionKind.binary, .choice, .score].map(\.rawValue) == ["binary", "choice", "score"])
        #expect([DecisionKind.binary, .choice, .score].map(\.description) == ["binary", "choice", "score"])
        #expect(DecisionKind(rawValue: "choice") == .choice)
        #expect(DecisionKind(rawValue: "ranking")?.rawValue == "ranking")
        #expect(DecisionKind(rawValue: "ranking") != .choice)
    }

    @Test func probabilitySourceNamesAndDescriptions() {
        typealias Source = DecisionCapabilities.ProbabilitySource
        #expect([Source.native, .estimated, .unspecified].map(\.rawValue) == ["native", "estimated", "unspecified"])
        #expect(Source.native.description == "native")
        #expect(Source(rawValue: "estimated") == .estimated)
        #expect(Source(rawValue: "guessed")?.rawValue == "guessed")
        #expect(Source(rawValue: "guessed") != .estimated)
    }

    @Test func encodeAsStrings() throws {
        #expect(try JSONEncoder().encode([DecisionKind.binary]) == Data(#"["binary"]"#.utf8))
        #expect(
            try JSONEncoder().encode([DecisionCapabilities.ProbabilitySource.native]) == Data(#"["native"]"#.utf8)
        )
    }

    @Test func roundTripKnownAndUnknownNames() throws {
        let kinds = try JSONDecoder().decode([DecisionKind].self, from: Data(#"["score", "ranking"]"#.utf8))
        #expect(kinds[0] == .score)
        #expect(kinds[1].rawValue == "ranking")
        #expect(try JSONEncoder().encode(kinds) == Data(#"["score","ranking"]"#.utf8))
        // A decoded value, known or not, is created again from its raw value.
        #expect(kinds.map { DecisionKind(rawValue: $0.rawValue) } == kinds)

        let source = try JSONDecoder().decode(
            DecisionCapabilities.ProbabilitySource.self,
            from: Data(#""calibrated""#.utf8)
        )
        #expect(source.rawValue == "calibrated")
        #expect(source != .native)
        #expect(DecisionCapabilities.ProbabilitySource(rawValue: source.rawValue) == source)
    }
}
