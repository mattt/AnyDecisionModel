import Foundation
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
            (401, DecisionError.Code.authenticationFailed),
            (403, .authenticationFailed),
            (402, .quotaExceeded),
            (408, .timedOut),
            (504, .timedOut),
            (429, .rateLimited),
            (300, .requestFailed),
            (400, .requestFailed),
            (404, .requestFailed),
            (409, .requestFailed),
            (422, .requestFailed),
            (500, .requestFailed),
            (502, .requestFailed),
            (503, .requestFailed),
            (529, .requestFailed),
        ]
    )
    func mapsStatusCode(statusCode: Int, code: DecisionError.Code) {
        let error = DecisionError.requestFailed(
            statusCode: statusCode,
            detail: #"{"error": "nope"}"#,
            headers: ["X-Request-Id": "abc"]
        )

        #expect(error.code == code)
        #expect(error.message == #"HTTP error (status \#(statusCode)): {"error": "nope"}"#)
        #expect(error.response?.statusCode == statusCode)
        #expect(error.response?.body == #"{"error": "nope"}"#)
        #expect(error.response?.headers == ["x-request-id": "abc"])
        #expect(error.questionIndex == nil)
        #expect(error.reason == nil)
        #expect(error.retryAfter == nil)
    }

    @Test func everyStatusCodeKeepsTheResponse() {
        for statusCode in 100 ..< 600 {
            let error = DecisionError.requestFailed(statusCode: statusCode, detail: "body", headers: [:])
            #expect(error.response == DecisionError.HTTPResponse(statusCode: statusCode, body: "body"))
            #expect(error.message == "HTTP error (status \(statusCode)): body")
        }
    }

    // MARK: - Retry-After

    @Test func rateLimitReadsRetryAfterSeconds() {
        let error = DecisionError.requestFailed(statusCode: 429, detail: "", headers: ["Retry-After": " 7 "])
        #expect(error.code == .rateLimited)
        #expect(error.retryAfter == .seconds(7))
        #expect(error.response?.headers["retry-after"] == " 7 ")
    }

    @Test func retryAfterReadsFractionalSeconds() {
        #expect(DecisionError.retryAfter(from: "1.5", now: Date()) == .milliseconds(1_500))
    }

    @Test func retryAfterReadsHTTPDate() {
        let now = Date(timeIntervalSince1970: 1_445_412_480)
        let error = DecisionError(
            statusCode: 429,
            detail: "",
            headers: ["retry-after": "Wed, 21 Oct 2015 07:28:05 GMT"],
            now: now
        )
        #expect(error.retryAfter == .seconds(5))
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

    /// A `Retry-After` header on a status other than 429, such as 503, is kept,
    /// because the retry policy reads it for every retryable status code.
    @Test func otherStatusCodesReadRetryAfter() {
        let error = DecisionError.requestFailed(statusCode: 503, detail: "", headers: ["Retry-After": "2"])
        #expect(error.code == .requestFailed)
        #expect(error.retryAfter == .seconds(2))
    }

    // MARK: - isRetryable

    @Test func isRetryableForTemporaryFailures() {
        for statusCode in 100 ..< 600 {
            let error = DecisionError.requestFailed(statusCode: statusCode, detail: "", headers: [:])
            let expected = [408, 429, 503, 504, 529].contains(statusCode)
            #expect(error.isRetryable == expected, "status \(statusCode)")
        }
        #expect(DecisionError(URLError(.timedOut)).isRetryable)
        #expect(DecisionError(URLError(.networkConnectionLost)).isRetryable)
        #expect(!DecisionError(URLError(.badURL)).isRetryable)
        #expect(!DecisionError(URLError(.unsupportedURL)).isRetryable)
        #expect(!DecisionError.missingCredentials.isRetryable)
        #expect(!DecisionError.invalidResponse("x").isRetryable)
        #expect(!DecisionError.inputBlocked(reason: "safety", message: nil).isRetryable)
    }

    @Test func initializerDefaultsIsRetryableByCode() {
        #expect(DecisionError(code: .rateLimited, message: "").isRetryable)
        #expect(DecisionError(code: .timedOut, message: "").isRetryable)
        #expect(DecisionError(code: .transportFailed, message: "").isRetryable)
        #expect(!DecisionError(code: .requestFailed, message: "").isRetryable)
        #expect(
            DecisionError(
                code: .requestFailed,
                message: "",
                response: DecisionError.HTTPResponse(statusCode: 503)
            ).isRetryable
        )
        #expect(!DecisionError(code: .rateLimited, message: "", isRetryable: false).isRetryable)
        #expect(DecisionError(code: .invalidResponse, message: "", isRetryable: true).isRetryable)
    }

    // MARK: - URL errors

    @Test func mapsURLErrors() {
        let timeout = DecisionError(URLError(.timedOut))
        #expect(timeout.code == .timedOut)
        #expect(timeout.message == "The request timed out.")
        #expect(timeout.response == nil)

        let lost = DecisionError(URLError(.networkConnectionLost))
        #expect(lost.code == .transportFailed)
        #expect(lost.message.hasPrefix("Transport error (URLError \(URLError.Code.networkConnectionLost.rawValue)): "))
        #expect(lost.response == nil)
    }

    // MARK: - Messages and descriptions

    @Test func versionZeroNineMessages() {
        let cases: [(DecisionError, String)] = [
            (.invalidQuestion(index: 2, reason: "Empty."), "Invalid question at index 2: Empty."),
            (.unsupportedQuestion(index: 1, reason: "Too many."), "Unsupported question at index 1: Too many."),
            (.unsupportedState("No images."), "Unsupported state: No images."),
            (.missingCredentials, "No API key is configured."),
            (.requestFailed(statusCode: 500, detail: "boom", headers: [:]), "HTTP error (status 500): boom"),
            (.inputBlocked(reason: "safety", message: "Blocked."), "The input was blocked (safety): Blocked."),
            (.inputBlocked(reason: "safety", message: nil), "The input was blocked (safety)."),
            (.inputBlocked(reason: "safety", message: ""), "The input was blocked (safety)."),
            (.invalidResponse("Not JSON."), "Invalid response: Not JSON."),
            (.modelUnavailable("Missing."), "Model unavailable: Missing."),
        ]
        for (error, message) in cases {
            #expect(error.message == message)
            #expect(error.description == message)
            #expect(error.errorDescription == message)
            #expect(String(describing: error) == message)
            #expect((error as any Error).localizedDescription == message)
        }
    }

    @Test func factoriesSetFields() {
        let invalid = DecisionError.invalidQuestion(index: 3, reason: "Empty.")
        #expect(invalid.code == .invalidQuestion)
        #expect(invalid.questionIndex == 3)

        let unsupported = DecisionError.unsupportedQuestion(index: 4, reason: "Too many.")
        #expect(unsupported.code == .unsupportedQuestion)
        #expect(unsupported.questionIndex == 4)

        let blocked = DecisionError.inputBlocked(reason: "safety", message: "Blocked.")
        #expect(blocked.code == .inputBlocked)
        #expect(blocked.reason == "safety")
        #expect(blocked.response == nil)

        let refused = DecisionError.refused(reason: "content_filter", message: "No.")
        #expect(refused.code == .refused)
        #expect(refused.reason == "content_filter")
        #expect(refused.message == "The model refused to answer: No.")

        let option = DecisionError.unsupportedOption("No seed.")
        #expect(option.code == .unsupportedOption)
        #expect(option.message == "Unsupported option: No seed.")
    }

    // MARK: - Codable

    @Test(
        arguments: [
            DecisionError.missingCredentials,
            .invalidQuestion(index: 1, reason: "Empty."),
            .inputBlocked(reason: "safety", message: nil),
            .requestFailed(statusCode: 429, detail: "slow", headers: ["Retry-After": "3", "X-Id": "a"]),
            DecisionError(URLError(.timedOut)),
            DecisionError(
                code: .refused,
                message: "No.",
                questionIndex: 0,
                reason: "policy",
                isRetryable: true,
                retryAfter: .milliseconds(1_250),
                response: DecisionError.HTTPResponse(statusCode: 418, headers: ["a": "b"], body: "{}")
            ),
        ]
    )
    func codableRoundTrip(error: DecisionError) throws {
        let data = try JSONEncoder().encode(error)
        #expect(try JSONDecoder().decode(DecisionError.self, from: data) == error)
    }

    @Test func codeEncodesAsString() throws {
        let data = try JSONEncoder().encode(DecisionError.missingCredentials)
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["code"] as? String == "missingCredentials")
        #expect(object["message"] as? String == "No API key is configured.")
        #expect(object["isRetryable"] as? Bool == false)
    }

    @Test func decodingRedactsCredentialHeaders() throws {
        let json = #"{"statusCode": 401, "headers": {"Authorization": "Bearer sk-secret"}, "body": ""}"#
        let response = try JSONDecoder().decode(DecisionError.HTTPResponse.self, from: Data(json.utf8))
        #expect(response.headers == ["authorization": "<redacted>"])
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
        #expect(
            error.response?.headers == [
                "authorization": "<redacted>",
                "proxy-authorization": "<redacted>",
                "x-api-key": "<redacted>",
                "x-goog-api-key": "<redacted>",
                "x-request-id": "abc",
            ]
        )
        #expect(!String(reflecting: error).contains("sk-secret"))
    }

    @Test func redactsCredentialsInText() {
        let headers = ["Authorization": "Bearer sk-secret", "Accept": "application/json"]
        #expect(
            HTTP.redactingCredentials(in: "Bad key: sk-secret (Bearer sk-secret)", requestHeaders: headers)
                == "Bad key: <redacted> (<redacted>)"
        )
        #expect(HTTP.redactingCredentials(in: "application/json", requestHeaders: headers) == "application/json")
        #expect(
            HTTP.redactingCredentials(in: "key g-123", requestHeaders: ["x-goog-api-key": "g-123"])
                == "key <redacted>"
        )
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
            #expect(error.code == .authenticationFailed)
            try expectNoSecret(apiKey, in: error)
            #expect(error.response?.body == #"{"error": "Incorrect API key provided: <redacted>"}"#)
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
            #expect(error.code == .requestFailed)
            try expectNoSecret(apiKey, in: error)
        }
    }

    private func expectNoSecret(_ secret: String, in error: DecisionError) throws {
        #expect(!error.message.contains(secret))
        #expect(!error.description.contains(secret))
        #expect(!(error as any Error).localizedDescription.contains(secret))
        #expect(!String(reflecting: error).contains(secret))
        #expect(error.response?.headers.values.contains { $0.contains(secret) } == false)
        #expect(error.response?.body.contains(secret) == false)
        let json = try JSONEncoder().encode(error)
        #expect(String(decoding: json, as: UTF8.self).contains(secret) == false)
    }

    // MARK: - Equality

    @Test func equalityComparesEveryField() {
        #expect(DecisionError.missingCredentials == .missingCredentials)
        #expect(DecisionError.invalidQuestion(index: 1, reason: "a") != .invalidQuestion(index: 2, reason: "a"))
        #expect(DecisionError.invalidQuestion(index: 1, reason: "a") != .unsupportedQuestion(index: 1, reason: "a"))
        var error = DecisionError.missingCredentials
        error.isRetryable = true
        #expect(error != .missingCredentials)
        #expect(Set([DecisionError.missingCredentials, .missingCredentials]).count == 1)
    }
}
