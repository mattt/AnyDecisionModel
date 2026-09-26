import Foundation
import Testing

@testable import AnyDecisionModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

private let okResponse = """
    {"model": "jev-1.13.0", "answers": {"0": {"noul": 0.75}}, "usage": {"input_tokens": 42, "output_tokens": 0}}
    """

private let fastRetries = JevDecisionModel.RetryPolicy(
    strategy: .constant(duration: 0.001),
    maximumInterval: 0.01,
    maximumRetries: 3
)

private func makeModel(
    _ host: StubURLProtocol.Host,
    apiKey: String? = "test-key",
    retryPolicy: JevDecisionModel.RetryPolicy = fastRetries
) -> JevDecisionModel {
    JevDecisionModel(
        baseURL: host.baseURL,
        apiKey: apiKey,
        session: StubURLProtocol.makeSession(),
        retryPolicy: retryPolicy
    )
}

/// Sends questions through a Jev model and returns the request body.
private func requestBody(
    state: DecisionState = "state",
    questions: [Question]
) async throws -> [String: Any] {
    let host = StubURLProtocol.Host()
    // The response has no answers, so the call throws after the request is recorded.
    host.enqueue(json: #"{"model": "m", "answers": {}}"#)
    _ = try? await DecisionSession(model: makeModel(host), state: state).decide(questions)
    let recorded = try #require(host.recorded.first)
    return try #require(try JSONSerialization.jsonObject(with: recorded.body) as? [String: Any])
}

private func requestQuestion(_ question: Question) async throws -> [String: Any] {
    let body = try await requestBody(questions: [question])
    let questions = try #require(body["questions"] as? [String: [String: Any]])
    return try #require(questions["0"])
}

@Suite("JevDecisionModel")
struct JevDecisionModelTests {
    @Test func sendsRequestAndDecodesAnswer() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: makeModel(host), state: .text("A 6 sided die rolled a 3."))

        let probability = try await session.probability(of: "Is the number rolled odd?")
        #expect(probability == 0.75)

        let recorded = try #require(host.recorded.first)
        #expect(recorded.headers["Authorization"] == "Bearer test-key")
        let body = try #require(try JSONSerialization.jsonObject(with: recorded.body) as? [String: Any])
        #expect(body["model"] as? String == "jev-latest")
        #expect(body["state"] as? String == "A 6 sided die rolled a 3.")
        let questions = try #require(body["questions"] as? [String: [String: Any]])
        #expect(questions["0"]?["type"] as? String == "noul")
    }

    @Test func batchResponseIncludesModelAndUsage() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"model": "jev-1.13.0",
                 "answers": {
                   "0": {"choice": "billing", "probabilities": {"billing": 0.8, "sales": 0.2}, "confidence": 0.65},
                   "1": {"noul": 0.9}
                 },
                 "usage": {"input_tokens": 99, "output_tokens": 0}}
                """
        )
        let session = DecisionSession(model: makeModel(host), state: "ticket")
        let response = try await session.decide([
            .choice(instructions: "Team?", options: [ChoiceOption("billing"), ChoiceOption("sales")]),
            .binary(instructions: "Urgent?"),
        ])
        #expect(response.modelID == "jev-1.13.0")
        #expect(response.usage.inputTokenCount == 99)
        guard case .choice(_, _, let confidence) = response.answers[0] else {
            Issue.record("Expected a choice answer.")
            return
        }
        // Remote confidence is kept as it is.
        #expect(confidence == 0.65)
        #expect(response.answers[1] == .binary(probability: 0.9))
    }

    @Test func typedChoicePreservesRemoteConfidence() async throws {
        enum Coin: String, Choosable {
            case heads, tails
        }
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"model": "jev", "answers": {"0": {"choice": "tails", "probabilities": {"heads": 0.4, "tails": 0.6}, "confidence": 0.02}}}
                """
        )
        let session = DecisionSession(model: makeModel(host), state: "I flipped a coin.")
        let result = try await session.choice("How did the coin land?", from: Coin.self)
        #expect(result.value == .tails)
        #expect(result.probability(of: .heads) == 0.4)
        #expect(result.distribution.map(\.option) == [.heads, .tails])
        #expect(result.confidence == 0.02)
    }

    // MARK: - Wire format

    @Test func binaryQuestionUsesNoulWireName() async throws {
        let object = try await requestQuestion(.binary(instructions: "Is the number odd?"))
        #expect(object["type"] as? String == "noul")
        #expect(object["instructions"] as? String == "Is the number odd?")
        #expect(object["criteria"] == nil)
    }

    @Test func binaryCriteriaUseTrueAndFalseKeys() async throws {
        let object = try await requestQuestion(
            .binary(
                instructions: "Is it urgent?",
                criteria: BinaryCriteria(whenTrue: "needs action today", whenFalse: "can wait")
            )
        )
        let criteria = try #require(object["criteria"] as? [String: String])
        #expect(criteria == ["true": "needs action today", "false": "can wait"])
    }

    @Test func choiceCriteriaMapOptionsToDescriptionsOrNull() async throws {
        let object = try await requestQuestion(
            .choice(
                instructions: "Route this ticket.",
                options: [ChoiceOption("billing", description: "payments"), ChoiceOption("sales")]
            )
        )
        #expect(object["type"] as? String == "choice")
        let criteria = try #require(object["criteria"] as? [String: Any])
        #expect(criteria["billing"] as? String == "payments")
        #expect(criteria["sales"] is NSNull)
    }

    @Test func scoreCriteriaAreAnOrderedArray() async throws {
        let object = try await requestQuestion(
            .score(instructions: "How frustrated?", levels: ["Calm", "Frustrated", "Very angry"])
        )
        #expect(object["type"] as? String == "score")
        #expect(object["criteria"] as? [String] == ["Calm", "Frustrated", "Very angry"])
    }

    @Test func structuredStateIsSentAsJSON() async throws {
        let body = try await requestBody(
            state: .json(["ticket": "charged twice", "priority": 2, "tags": ["billing"]]),
            questions: [.binary(instructions: "Urgent?")]
        )
        #expect(body["model"] as? String == "jev-latest")

        let state = try #require(body["state"] as? [String: Any])
        #expect(state["ticket"] as? String == "charged twice")
        #expect(state["priority"] as? Int == 2)
        #expect(state["tags"] as? [String] == ["billing"])
    }

    @Test func textStateIsSentAsString() async throws {
        let body = try await requestBody(state: .text("hello"), questions: [.binary(instructions: "x")])
        #expect(body["state"] as? String == "hello")
    }

    @Test func questionsAreKeyedByPosition() async throws {
        let body = try await requestBody(
            questions: [
                .binary(instructions: "First?"),
                .score(instructions: "Second?", levels: ["low", "high"]),
            ]
        )
        let questions = try #require(body["questions"] as? [String: [String: Any]])
        #expect(questions.keys.sorted() == ["0", "1"])
        #expect(questions["0"]?["instructions"] as? String == "First?")
        #expect(questions["1"]?["instructions"] as? String == "Second?")
    }

    @Test func decodesJevResponse() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {
                  "model": "jev-1.13.0",
                  "answers": {
                    "2": {"score": 1.5, "legend": ["Calm", "Frustrated", "Very angry"],
                          "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "confidence": 0.4},
                    "0": {"noul": 0.97},
                    "1": {"choice": "billing", "probabilities": {"billing": 0.9, "sales": 0.1}, "confidence": 0.8}
                  },
                  "usage": {"input_tokens": 120, "output_tokens": 0},
                  "latency_ms": 31
                }
                """
        )
        let session = DecisionSession(model: makeModel(host), state: "state")
        let response = try await session.decide([
            .binary(instructions: "Odd?"),
            .choice(instructions: "Team?", options: [ChoiceOption("billing"), ChoiceOption("sales")]),
            .score(instructions: "Anger?", levels: ["Calm", "Frustrated", "Very angry"]),
        ])
        #expect(response.modelID == "jev-1.13.0")
        #expect(response.usage == DecisionSession.Usage(inputTokenCount: 120, outputTokenCount: 0))
        #expect(
            response.answers == [
                .binary(probability: 0.97),
                .choice(name: "billing", probabilities: ["billing": 0.9, "sales": 0.1], confidence: 0.8),
                .score(
                    value: 1.5,
                    probabilities: [0.1, 0.3, 0.6],
                    legend: ["Calm", "Frustrated", "Very angry"],
                    confidence: 0.4
                ),
            ]
        )
        #expect(response.diagnostics.isEmpty)
    }

    @Test func ignoresLegendThatIsNotAListOfStrings() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"model": "m", "answers": {"0": {"score": 0.5, "legend": [{"level": 0}, {"level": 1}],
                                                 "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0}}}
                """
        )
        let session = DecisionSession(model: makeModel(host), state: "state")
        let answer = try await session.decide(.score(instructions: "x", levels: ["low", "high"]))
        #expect(answer == .score(value: 0.5, probabilities: [0.5, 0.5], legend: nil, confidence: 0))
    }

    @Test func missingUsageDecodesAsZero() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"model": "m", "answers": {"0": {"noul": 0.5}}}"#)
        let session = DecisionSession(model: makeModel(host), state: "state")
        let response = try await session.decide([.binary(instructions: "x")])
        #expect(response.usage == DecisionSession.Usage())
    }

    @Test(
        arguments: [
            #"{"model": "m", "answers": {"0": {"probabilities": {}}}}"#,
            #"{"model": "m", "answers": {"0": {"noul": "high"}}}"#,
            #"{"model": "m", "answers": {"0": {"score": 1, "probabilities": {"1": 0.5, "2": 0.5}, "confidence": 0}}}"#,
            #"{"model": "m", "answers": {"0": {"choice": "a", "probabilities": {"a": 1}}}}"#,
            #"{"answers": {}}"#,
        ]
    )
    func rejectsResponsesThatDoNotDecode(json: String) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: json)
        let session = DecisionSession(model: makeModel(host), state: "state")
        do {
            _ = try await session.decide(.binary(instructions: "x"))
            Issue.record("Expected an error.")
        } catch DecisionError.invalidResponse(let detail) {
            // The response fails to decode, before it is checked against the question.
            #expect(!detail.hasPrefix("Question"))
        }
    }

    // MARK: - Errors and retries

    @Test(arguments: [401, 422, 500])
    func reportsHTTPErrorsWithoutRetry(statusCode: Int) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"error": "nope"}"#, statusCode: statusCode, headers: ["X-Request-Id": "abc"])
        let session = DecisionSession(model: makeModel(host), state: "state")

        do {
            _ = try await session.probability(of: "Question?")
            Issue.record("Expected an error.")
        } catch let DecisionError.requestFailed(code, detail, headers) {
            #expect(code == statusCode)
            #expect(detail.contains("nope"))
            #expect(headers["x-request-id"] == "abc")
        }
        #expect(host.recorded.count == 1)
    }

    @Test(arguments: [429, 503, 529])
    func retriesRateLimitAndOverload(statusCode: Int) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: "{}", statusCode: statusCode, headers: ["Retry-After": "0"])
        host.enqueue(json: "{}", statusCode: statusCode)
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: makeModel(host), state: "state")

        let probability = try await session.probability(of: "Question?")
        #expect(probability == 0.75)
        #expect(host.recorded.count == 3)
    }

    @Test func stopsAfterMaximumRetries() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"error": "first"}"#, statusCode: 529, headers: ["Retry-After": "1"])
        host.enqueue(json: #"{"error": "second"}"#, statusCode: 529, headers: ["Retry-After": "1"])
        host.enqueue(
            json: #"{"error": "final"}"#,
            statusCode: 529,
            headers: ["Retry-After": "1", "X-Attempt": "final"]
        )
        host.enqueue(json: okResponse)
        let policy = JevDecisionModel.RetryPolicy(
            strategy: .constant(duration: 0.001),
            maximumInterval: 0.001,
            maximumRetries: 2
        )
        let model = makeModel(host, retryPolicy: policy)
        let session = DecisionSession(model: model, state: "state")

        do {
            _ = try await session.probability(of: "Question?")
            Issue.record("Expected an error.")
        } catch let DecisionError.requestFailed(statusCode, detail, headers) {
            #expect(statusCode == 529)
            #expect(detail == #"{"error": "final"}"#)
            #expect(headers["retry-after"] == "1")
            #expect(headers["x-attempt"] == "final")
        }
        #expect(host.recorded.count == 3)
    }

    @Test func eachRequestUsesIndependentRetrier() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: "{}", statusCode: 429)
        host.enqueue(json: okResponse)
        host.enqueue(json: "{}", statusCode: 429)
        host.enqueue(json: okResponse)
        let policy = JevDecisionModel.RetryPolicy(
            strategy: .constant(duration: 0),
            maximumRetries: 1
        )
        let session = DecisionSession(
            model: makeModel(host, retryPolicy: policy),
            state: "state"
        )

        _ = try await session.probability(of: "First question?")
        _ = try await session.probability(of: "Second question?")

        #expect(host.recorded.count == 4)
    }

    @Test func neverPolicyDoesNotRetry() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"error": "busy"}"#, statusCode: 429)
        host.enqueue(json: okResponse)
        let session = DecisionSession(
            model: makeModel(host, retryPolicy: .never),
            state: "state"
        )

        await #expect(throws: DecisionError.self) {
            _ = try await session.probability(of: "Question?")
        }
        #expect(host.recorded.count == 1)
    }

    @Test func retriesCustomStatusCode() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: "{}", statusCode: 502)
        host.enqueue(json: okResponse)
        let policy = JevDecisionModel.RetryPolicy(
            strategy: .constant(duration: 0),
            maximumRetries: 1,
            retryableStatusCodes: [502]
        )
        let session = DecisionSession(
            model: makeModel(host, retryPolicy: policy),
            state: "state"
        )

        let probability = try await session.probability(of: "Question?")
        #expect(probability == 0.75)
        #expect(host.recorded.count == 2)
    }

    @Test func retryAfterSecondsReplacesGeneratedDelay() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: "{}", statusCode: 429, headers: ["Retry-After": "0"])
        host.enqueue(json: okResponse)
        let policy = JevDecisionModel.RetryPolicy(
            strategy: .constant(duration: 60),
            maximumRetries: 1
        )
        let session = DecisionSession(
            model: makeModel(host, retryPolicy: policy),
            state: "state"
        )

        let probability = try await session.probability(of: "Question?")
        #expect(probability == 0.75)
        #expect(host.recorded.count == 2)
    }

    @Test func retryAfterHTTPDateReplacesGeneratedDelay() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: "{}",
            statusCode: 529,
            headers: ["Retry-After": "Wed, 21 Oct 2015 07:27:00 GMT"]
        )
        host.enqueue(json: okResponse)
        let policy = JevDecisionModel.RetryPolicy(
            strategy: .constant(duration: 60),
            maximumRetries: 1
        )
        let session = DecisionSession(
            model: makeModel(host, retryPolicy: policy),
            state: "state"
        )

        let probability = try await session.probability(of: "Question?")
        #expect(probability == 0.75)
        #expect(host.recorded.count == 2)
    }

    @Test func rejectsMalformedResponseBody() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: "not json")
        let session = DecisionSession(model: makeModel(host), state: "state")
        do {
            _ = try await session.probability(of: "Question?")
            Issue.record("Expected an error.")
        } catch DecisionError.invalidResponse {
        }
    }

    @Test func rejectsResponseThatDoesNotMatchQuestions() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"model": "jev", "answers": {"other": {"noul": 0.5}}}"#)
        let session = DecisionSession(model: makeModel(host), state: "state")
        await #expect(throws: DecisionError.invalidResponse("No answer for the question at index 0.")) {
            _ = try await session.probability(of: "Question?")
        }
    }

    @Test func missingCredentialsFailBeforeRequest() async throws {
        let host = StubURLProtocol.Host()
        let session = DecisionSession(model: makeModel(host, apiKey: nil), state: "state")
        await #expect(throws: DecisionError.missingCredentials) {
            _ = try await session.probability(of: "Question?")
        }
        #expect(host.recorded.isEmpty)
    }

    @Test func cancellationPropagates() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse, delay: 5)
        let session = DecisionSession(model: makeModel(host), state: "state")

        let task = Task { try await session.probability(of: "Question?") }
        try await Task.sleep(nanoseconds: 50_000_000)
        task.cancel()
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
    }

    @Test func cancellationStopsRetryBackoff() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: "{}", statusCode: 429, headers: ["Retry-After": "10"])
        let policy = JevDecisionModel.RetryPolicy(
            strategy: .constant(duration: 10),
            maximumInterval: 10,
            maximumRetries: 3
        )
        let model = makeModel(host, retryPolicy: policy)
        let session = DecisionSession(model: model, state: "state")

        let task = Task { try await session.probability(of: "Question?") }
        try await Task.sleep(nanoseconds: 100_000_000)
        task.cancel()
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
        #expect(host.recorded.count == 1)
    }

    @Test func environmentKeyPrefersTypesafeAIVariable() {
        #expect(
            JevDecisionModel.environmentAPIKey(from: ["TYPESAFE_AI_API_KEY": "a", "TYPESAFE_API_KEY": "b"]) == "a"
        )
        #expect(JevDecisionModel.environmentAPIKey(from: ["TYPESAFE_API_KEY": "b"]) == "b")
        #expect(JevDecisionModel.environmentAPIKey(from: ["TYPESAFE_AI_API_KEY": " ", "TYPESAFE_API_KEY": "b"]) == "b")
        #expect(JevDecisionModel.environmentAPIKey(from: [:]) == nil)
    }

    @Test func descriptionOmitsAPIKey() {
        let model = JevDecisionModel(apiKey: "secret-value")
        #expect(!String(describing: model).contains("secret-value"))
        #expect(!String(reflecting: model).contains("secret-value"))
    }
}
