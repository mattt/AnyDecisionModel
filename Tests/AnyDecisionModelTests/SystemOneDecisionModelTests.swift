import Foundation
import Testing

@testable import AnyDecisionModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

private let okResponse = """
    {"model": "served-model", "answers": {"0": {"type": "noul", "noul": 0.75}},
     "usage": {"input_tokens": 42, "output_tokens": 1}}
    """

private let fastRetries = RetryPolicy(
    strategy: .constant(duration: 0.001),
    maximumInterval: 0.01,
    maximumRetries: 3
)

private func makeModel(
    _ host: StubURLProtocol.Host,
    path: String = SystemOneDecisionModel.defaultPath,
    apiKey: String? = nil,
    headers: [String: String] = [:],
    capabilities: DecisionCapabilities = DecisionCapabilities(),
    extraBody: [String: JSONValue] = [:],
    retryPolicy: RetryPolicy = fastRetries
) throws -> SystemOneDecisionModel {
    try SystemOneDecisionModel(
        baseURL: host.baseURL,
        path: path,
        modelID: "my-model",
        apiKey: apiKey,
        headers: headers,
        capabilities: capabilities,
        extraBody: extraBody,
        session: StubURLProtocol.makeSession(),
        retryPolicy: retryPolicy
    )
}

private func requestBody(_ recorded: StubURLProtocol.Recorded) throws -> [String: Any] {
    try #require(try JSONSerialization.jsonObject(with: recorded.body) as? [String: Any])
}

/// Sends one batch through a model and returns the recorded request.
///
/// The response has no answers, so the call throws after the request is recorded.
private func record(
    _ questions: [Question],
    about state: DecisionState,
    with makeModel: (StubURLProtocol.Host) -> any DecisionModel
) async throws -> StubURLProtocol.Recorded {
    let host = StubURLProtocol.Host()
    host.enqueue(json: #"{"model": "m", "answers": {}}"#)
    _ = try? await DecisionSession(model: makeModel(host), state: state).decide(questions)
    return try #require(host.recorded.first)
}

@Suite("SystemOneDecisionModel")
struct SystemOneDecisionModelTests {
    // MARK: - Request

    @Test func sendsRequestToDefaultPath() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: try makeModel(host), state: "ticket")

        let response = try await session.decide([.binary(instructions: "Is this urgent?")])

        #expect(response.modelID == "served-model")
        #expect(response.answers == [.binary(probability: 0.75)])
        #expect(response.usage == DecisionSession.Usage(inputTokenCount: 42, outputTokenCount: 1))

        let recorded = try #require(host.recorded.first)
        #expect(recorded.url?.path == "/v1/systemone")
        #expect(recorded.headers["Authorization"] == nil)
        #expect(recorded.headers["Content-Type"] == "application/json")
        let body = try requestBody(recorded)
        #expect(Set(body.keys) == ["model", "state", "questions"])
        #expect(body["model"] as? String == "my-model")
        #expect(body["state"] as? String == "ticket")
    }

    @Test(arguments: ["api/alpha/decisions", "/api/alpha/decisions/", "api//alpha/decisions"])
    func sendsRequestToCustomPath(path: String) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: try makeModel(host, path: path), state: "ticket")

        _ = try await session.probability(of: "Is this urgent?")

        let recorded = try #require(host.recorded.first)
        #expect(recorded.url?.absoluteString == "\(host.baseURL.absoluteString)/api/alpha/decisions")
    }

    @Test func sendsAPIKeyAsBearerToken() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let model = try makeModel(host, apiKey: "secret-value")
        let session = DecisionSession(model: model, state: "ticket")

        _ = try await session.probability(of: "Is this urgent?")

        let recorded = try #require(host.recorded.first)
        #expect(recorded.headers["Authorization"] == "Bearer secret-value")
        #expect(recorded.url?.absoluteString.contains("secret-value") == false)
        #expect(!recorded.body.contains(Data("secret-value".utf8)))
        #expect(!String(describing: model).contains("secret-value"))
        #expect(!String(reflecting: model).contains("secret-value"))
    }

    @Test func emptyAPIKeySendsNoAuthorization() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: try makeModel(host, apiKey: ""), state: "ticket")

        _ = try await session.probability(of: "Is this urgent?")

        let recorded = try #require(host.recorded.first)
        #expect(recorded.headers["Authorization"] == nil)
    }

    @Test func sendsExtraHeaders() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let model = try makeModel(
            host,
            apiKey: "key",
            headers: ["traceparent": "00-abc-def-01", "X-Title": "Example"]
        )
        let session = DecisionSession(model: model, state: "ticket")

        _ = try await session.probability(of: "Is this urgent?")

        let recorded = try #require(host.recorded.first)
        // FoundationNetworking can change the letter case of header names.
        let headers = Dictionary(uniqueKeysWithValues: recorded.headers.map { ($0.key.lowercased(), $0.value) })
        #expect(headers["traceparent"] == "00-abc-def-01")
        #expect(headers["x-title"] == "Example")
        #expect(recorded.headers["Authorization"] == "Bearer key")
        #expect(recorded.headers["Content-Type"] == "application/json")
    }

    @Test func mergesExtraBody() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let model = try makeModel(
            host,
            extraBody: [
                "cache_salt": "tenant-1",
                "provider": ["order": ["a", "b"]],
                "temperature": 0,
            ]
        )
        let session = DecisionSession(model: model, state: .json(["id": 7]))

        _ = try await session.probability(of: "Is this urgent?")

        let body = try requestBody(try #require(host.recorded.first))
        #expect(Set(body.keys) == ["model", "state", "questions", "cache_salt", "provider", "temperature"])
        #expect(body["model"] as? String == "my-model")
        #expect((body["state"] as? [String: Any])?["id"] as? Int == 7)
        #expect(body["cache_salt"] as? String == "tenant-1")
        #expect((body["provider"] as? [String: Any])?["order"] as? [String] == ["a", "b"])
        #expect((body["temperature"] as? NSNumber)?.doubleValue == 0)
    }

    @Test func requestEncodingIgnoresReservedExtraBodyFields() throws {
        // The initializer rejects these keys.
        // The encoder also skips them, so that no path can replace a core field.
        let request = SystemOneRequest(
            model: "my-model",
            state: "ticket",
            questions: ["0": SystemOneQuestion(.binary(instructions: "Q?"))],
            extraBody: ["model": "other", "state": "other", "questions": [:], "seed": 1]
        )
        let body = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        #expect(body["model"] as? String == "my-model")
        #expect(body["state"] as? String == "ticket")
        #expect((body["questions"] as? [String: Any])?.keys.sorted() == ["0"])
        #expect((body["seed"] as? NSNumber)?.intValue == 1)
    }

    @Test func rejectsStateWithAttachments() async throws {
        let host = StubURLProtocol.Host()
        let state = DecisionState.segments([
            .text("Look at this."),
            .attachment(DecisionState.Attachment(imageData: Data([0x89]), mimeType: "image/png")),
        ])
        let session = DecisionSession(model: try makeModel(host), state: state)

        let expected = DecisionError.unsupportedState("SystemOneDecisionModel does not support image attachments.")
        await #expect(throws: expected) {
            _ = try await session.probability(of: "Is this urgent?")
        }
        #expect(host.recorded.isEmpty)
    }

    // MARK: - Reserved fields

    @Test func findsReservedBodyFields() {
        #expect(SystemOneDecisionModel.reservedBodyField(in: [:]) == nil)
        #expect(SystemOneDecisionModel.reservedBodyField(in: ["keep_alive": 300, "Model": "x"]) == nil)
        #expect(SystemOneDecisionModel.reservedBodyField(in: ["model": "x"]) == "model")
        #expect(SystemOneDecisionModel.reservedBodyField(in: ["state": "x"]) == "state")
        #expect(SystemOneDecisionModel.reservedBodyField(in: ["questions": [:], "seed": 1]) == "questions")
    }

    @Test func findsReservedHeadersInAnyCase() {
        #expect(SystemOneDecisionModel.reservedHeader(in: [:]) == nil)
        #expect(SystemOneDecisionModel.reservedHeader(in: ["traceparent": "x", "Accept-Language": "en"]) == nil)
        #expect(SystemOneDecisionModel.reservedHeader(in: ["Authorization": "x"]) == "Authorization")
        #expect(SystemOneDecisionModel.reservedHeader(in: ["authorization": "x"]) == "authorization")
        #expect(SystemOneDecisionModel.reservedHeader(in: ["CONTENT-TYPE": "x"]) == "CONTENT-TYPE")
    }

    @Test(arguments: ["model", "state", "questions"])
    func initializerThrowsForReservedBodyField(key: String) {
        #expect(throws: SystemOneDecisionModel.ConfigurationError(kind: .reservedBodyField, key: key)) {
            _ = try SystemOneDecisionModel(
                baseURL: URL(string: "https://example.invalid")!,
                modelID: "m",
                extraBody: [key: "x", "seed": 1]
            )
        }
    }

    @Test(arguments: ["Authorization", "authorization", "Content-Type", "CONTENT-TYPE"])
    func initializerThrowsForReservedHeader(name: String) {
        #expect(throws: SystemOneDecisionModel.ConfigurationError(kind: .reservedHeader, key: name)) {
            _ = try SystemOneDecisionModel(
                baseURL: URL(string: "https://example.invalid")!,
                modelID: "m",
                headers: [name: "x", "traceparent": "y"]
            )
        }
    }

    @Test func configurationErrorNamesKey() {
        let body = SystemOneDecisionModel.ConfigurationError(kind: .reservedBodyField, key: "model")
        let header = SystemOneDecisionModel.ConfigurationError(kind: .reservedHeader, key: "authorization")
        #expect(body.description == #"extraBody cannot set the reserved field "model"."#)
        #expect(header.localizedDescription == #"headers cannot set the reserved header "authorization"."#)
    }

    @Test func initializerAcceptsUnreservedKeys() throws {
        let model = try SystemOneDecisionModel(
            baseURL: URL(string: "https://example.invalid")!,
            modelID: "m",
            headers: ["Accept-Language": "en", "X-Model": "y"],
            extraBody: ["Model": "x", "keep_alive": 300]
        )
        #expect(model.headers == ["Accept-Language": "en", "X-Model": "y"])
        #expect(model.extraBody == ["Model": "x", "keep_alive": 300])
    }

    // MARK: - Capabilities

    @Test func sessionEnforcesCapabilitiesBeforeRequest() async throws {
        let host = StubURLProtocol.Host()
        let capabilities = DecisionCapabilities(
            maximumQuestions: 2,
            maximumChoiceOptions: 2,
            maximumScoreLevels: 3
        )
        let model = try makeModel(host, capabilities: capabilities)
        #expect(model.capabilities == capabilities)
        let session = DecisionSession(model: model, state: "ticket")

        await #expect(throws: DecisionError.self) {
            _ = try await session.decide([
                .binary(instructions: "A?"), .binary(instructions: "B?"), .binary(instructions: "C?"),
            ])
        }
        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(
                .choice(instructions: "Team?", options: [ChoiceOption("a"), ChoiceOption("b"), ChoiceOption("c")])
            )
        }
        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(.score(instructions: "How?", levels: ["1", "2", "3", "4"]))
        }
        #expect(host.recorded.isEmpty)
    }

    @Test func defaultCapabilitiesHaveNoLimits() throws {
        let model = try SystemOneDecisionModel(baseURL: URL(string: "https://example.invalid")!, modelID: "m")
        #expect(model.capabilities == DecisionCapabilities())
        #expect(model.capabilities.maximumQuestions == nil)
        #expect(model.capabilities.probabilitySource == .unspecified)
    }

    @Test func sessionRejectsUnsupportedQuestionKind() async throws {
        let host = StubURLProtocol.Host()
        let model = try makeModel(host, capabilities: DecisionCapabilities(questionKinds: [.binary]))
        let session = DecisionSession(model: model, state: "ticket")

        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(.score(instructions: "How?", levels: ["1", "2"]))
        }
        #expect(host.recorded.isEmpty)
    }

    // MARK: - Retries

    @Test(arguments: [429, 503, 529])
    func retriesEligibleStatusCodes(statusCode: Int) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"error": "busy"}"#, statusCode: statusCode)
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: try makeModel(host), state: "ticket")

        let probability = try await session.probability(of: "Is this urgent?")

        #expect(probability == 0.75)
        #expect(host.recorded.count == 2)
    }

    @Test func neverPolicyDoesNotRetry() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"error": "busy"}"#, statusCode: 429)
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: try makeModel(host, retryPolicy: .never), state: "ticket")

        do {
            _ = try await session.probability(of: "Is this urgent?")
            Issue.record("Expected the request to fail.")
        } catch let DecisionError.requestFailed(statusCode, _, _) {
            #expect(statusCode == 429)
        }
        #expect(host.recorded.count == 1)
    }

    @Test func doesNotRetryClientErrors() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"error": "bad"}"#, statusCode: 400)
        let session = DecisionSession(model: try makeModel(host), state: "ticket")

        await #expect(throws: DecisionError.self) {
            _ = try await session.probability(of: "Is this urgent?")
        }
        #expect(host.recorded.count == 1)
    }

    // MARK: - Dialect additions

    @Test func ignoresConfidenceOnBinaryAnswer() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: #"{"model": "m", "answers": {"0": {"type": "noul", "noul": 0.8, "confidence": 0.4}}}"#
        )
        let session = DecisionSession(model: try makeModel(host), state: "ticket")

        let response = try await session.decide([.binary(instructions: "Is this urgent?")])

        #expect(response.answers == [.binary(probability: 0.8)])
    }

    @Test func ignoresLabelMassAndUsageAdditions() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"model": "m",
                 "answers": {
                   "0": {"type": "noul", "noul": 0.3, "label_mass": 0.97},
                   "1": {"type": "choice", "choice": "b", "probabilities": {"a": 0.4, "b": 0.6},
                         "confidence": 0.03, "label_mass": 0.88}
                 },
                 "usage": {"input_tokens": 120, "output_tokens": 1, "cached_tokens": 96,
                           "cost": 0.00012, "cost_details": {"upstream": 0.0001}}}
                """
        )
        let session = DecisionSession(model: try makeModel(host), state: "ticket")

        let response = try await session.decide([
            .binary(instructions: "Is this urgent?"),
            .choice(instructions: "Which?", options: [ChoiceOption("a"), ChoiceOption("b")]),
        ])

        #expect(
            response.answers == [
                .binary(probability: 0.3),
                .choice(name: "b", probabilities: ["a": 0.4, "b": 0.6], confidence: 0.03),
            ]
        )
        #expect(response.usage == DecisionSession.Usage(inputTokenCount: 120, outputTokenCount: 1))
        #expect(response.diagnostics.isEmpty)
    }

    @Test(
        arguments: [
            #"{"type": "refusal", "refusal": "I can't help with that."}"#,
            #"{"refusal": "I can't help with that."}"#,
            #"{"type": "refusal"}"#,
        ]
    )
    func refusalThrowsInvalidResponse(answer: String) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: #"{"model": "m", "answers": {"0": {"type": "noul", "noul": 0.5}, "1": \#(answer)}}"#
        )
        let session = DecisionSession(model: try makeModel(host), state: "ticket")

        do {
            _ = try await session.decide([.binary(instructions: "A?"), .binary(instructions: "B?")])
            Issue.record("Expected an error.")
        } catch DecisionError.invalidResponse {
        }
    }

    @Test func unknownAnswerTypeThrowsInvalidResponse() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: #"{"model": "m", "answers": {"0": {"type": "box", "box": [0.1, 0.2, 0.3, 0.4]}}}"#
        )
        let session = DecisionSession(model: try makeModel(host), state: "ticket")

        do {
            _ = try await session.decide(.binary(instructions: "Where is the dog?"))
            Issue.record("Expected an error.")
        } catch DecisionError.invalidResponse {
        }
    }

    // MARK: - Configurations

    @Test func jevFactoryMatchesJevDecisionModel() {
        let model = SystemOneDecisionModel.jev(apiKey: "key")
        #expect(model.baseURL == JevDecisionModel.defaultBaseURL)
        #expect(model.path == "v1/systemone")
        #expect(model.modelID == "jev-latest")
        #expect(model.headers.isEmpty)
        #expect(model.extraBody.isEmpty)
        #expect(model.retryPolicy == .default)
        #expect(model.requiresAPIKey)
        #expect(model.capabilities == JevDecisionModel(apiKey: "key").capabilities)
    }

    @Test func ollamaFactoryMatchesOllamaDecisionModel() {
        let model = SystemOneDecisionModel.ollama(modelID: "nimble", apiKey: nil)
        #expect(model.baseURL == OllamaDecisionModel.defaultBaseURL)
        #expect(model.path == "v1/systemone")
        #expect(model.modelID == "nimble")
        #expect(model.headers.isEmpty)
        #expect(model.extraBody.isEmpty)
        #expect(model.retryPolicy == .never)
        #expect(!model.requiresAPIKey)
        #expect(model.capabilities == OllamaDecisionModel(modelID: "nimble", apiKey: nil).capabilities)

        let keptAlive = SystemOneDecisionModel.ollama(modelID: "nimble", apiKey: nil, keepAlive: -1)
        #expect(keptAlive.extraBody == ["keep_alive": -1])
    }

    @Test func jevFactoryRequiresAPIKey() async throws {
        let host = StubURLProtocol.Host()
        let model = SystemOneDecisionModel.jev(
            baseURL: host.baseURL,
            apiKey: nil,
            session: StubURLProtocol.makeSession()
        )
        let session = DecisionSession(model: model, state: "ticket")

        await #expect(throws: DecisionError.missingCredentials) {
            _ = try await session.probability(of: "Is this urgent?")
        }
        #expect(host.recorded.isEmpty)
    }

    @Test func ollamaFactoryDoesNotRetry() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"error": "busy"}"#, statusCode: 503)
        host.enqueue(json: okResponse)
        let model = SystemOneDecisionModel.ollama(
            modelID: "nimble",
            baseURL: host.baseURL,
            apiKey: nil,
            session: StubURLProtocol.makeSession()
        )
        let session = DecisionSession(model: model, state: "ticket")

        await #expect(throws: DecisionError.self) {
            _ = try await session.probability(of: "Is this urgent?")
        }
        #expect(host.recorded.count == 1)
    }

    static let batches: [(DecisionState, [Question])] = [
        ("A 6 sided die rolled a 3.", [.binary(instructions: "Is the number odd?")]),
        (
            "Ticket: \"refund\" / urgent ✓",
            [
                .binary(
                    instructions: "Is it urgent?",
                    criteria: BinaryCriteria(whenTrue: "needs action today", whenFalse: "can wait")
                ),
                .choice(
                    instructions: "Which team?",
                    options: [
                        ChoiceOption("billing", description: "payments"),
                        ChoiceOption("sales"),
                        ChoiceOption("bug", description: "errors"),
                    ]
                ),
            ]
        ),
        (
            .json(["id": 7, "tags": ["a", "b"], "customer": ["tier": "gold", "since": 2019]]),
            [
                .score(instructions: "How urgent?", levels: ["Routine", "Soon", "Immediate"]),
                .binary(instructions: "A?"),
                .binary(instructions: "B?"),
            ]
        ),
    ]

    @Test(arguments: 0 ..< batches.count)
    func jevFactorySendsSameBytesAsJevDecisionModel(index: Int) async throws {
        let (state, questions) = Self.batches[index]
        let named = try await record(questions, about: state) { host in
            JevDecisionModel(baseURL: host.baseURL, apiKey: "key", session: StubURLProtocol.makeSession())
        }
        let factory = try await record(questions, about: state) { host in
            SystemOneDecisionModel.jev(baseURL: host.baseURL, apiKey: "key", session: StubURLProtocol.makeSession())
        }

        #expect(named.body == factory.body)
        #expect(named.headers == factory.headers)
        #expect(named.url?.path == factory.url?.path)
    }

    @Test(arguments: 0 ..< batches.count, [nil, 0, 300, -1, 0.5] as [TimeInterval?])
    func ollamaFactorySendsSameBytesAsOllamaDecisionModel(index: Int, keepAlive: TimeInterval?) async throws {
        let (state, questions) = Self.batches[index]
        let named = try await record(questions, about: state) { host in
            OllamaDecisionModel(
                modelID: "nimble",
                baseURL: host.baseURL,
                apiKey: "key",
                keepAlive: keepAlive,
                session: StubURLProtocol.makeSession()
            )
        }
        let factory = try await record(questions, about: state) { host in
            SystemOneDecisionModel.ollama(
                modelID: "nimble",
                baseURL: host.baseURL,
                apiKey: "key",
                keepAlive: keepAlive,
                session: StubURLProtocol.makeSession()
            )
        }

        #expect(named.body == factory.body)
        #expect(named.headers == factory.headers)
        #expect(named.url?.path == factory.url?.path)
    }

    // MARK: - Fixtures

    // These requests are written by hand from version 0.8's Jev and Ollama backends,
    // with the object keys in sorted order.
    // Version 0.8 sent the same fields and values,
    // with the keys in an order that could change between requests.

    /// The headers of a recorded request, with lowercase names.
    ///
    /// FoundationNetworking can change the letter case of header names.
    private func lowercasedHeaders(_ recorded: StubURLProtocol.Recorded) -> [String: String] {
        Dictionary(uniqueKeysWithValues: recorded.headers.map { ($0.key.lowercased(), $0.value) })
    }

    /// Records one request from each model builder
    /// and checks its URL, headers, and body bytes.
    private func expectRequest(
        _ questions: [Question],
        about state: DecisionState,
        from builders: [(StubURLProtocol.Host) -> any DecisionModel],
        headers expectedHeaders: [String: String],
        body expectedBody: String
    ) async throws {
        for build in builders {
            let host = StubURLProtocol.Host()
            host.enqueue(json: #"{"model": "m", "answers": {}}"#)
            _ = try? await DecisionSession(model: build(host), state: state).decide(questions)
            let recorded = try #require(host.recorded.first)
            #expect(recorded.url?.absoluteString == "https://\(host.name)/v1/systemone")
            // URLSession adds `Content-Length` for the body that the model sets.
            var headers = lowercasedHeaders(recorded)
            if let length = headers.removeValue(forKey: "content-length") {
                #expect(length == String(recorded.body.count))
            }
            #expect(headers == expectedHeaders)
            #expect(String(decoding: recorded.body, as: UTF8.self) == expectedBody)
        }
    }

    @Test func jevRequestMatchesFixture() async throws {
        try await expectRequest(
            [
                .choice(
                    instructions: "Which team?",
                    options: [ChoiceOption("billing", description: "payments"), ChoiceOption("sales")]
                ),
                .binary(
                    instructions: "Urgent?",
                    criteria: BinaryCriteria(whenTrue: "today", whenFalse: "can wait")
                ),
                .score(instructions: "How urgent?", levels: ["Low", "High"]),
            ],
            about: "Refund / urgent",
            from: [
                { JevDecisionModel(baseURL: $0.baseURL, apiKey: "test-key", session: StubURLProtocol.makeSession()) },
                {
                    SystemOneDecisionModel.jev(
                        baseURL: $0.baseURL,
                        apiKey: "test-key",
                        session: StubURLProtocol.makeSession()
                    )
                },
            ],
            headers: [
                "accept": "application/json",
                "authorization": "Bearer test-key",
                "content-type": "application/json",
            ],
            body: #"{"model":"jev-latest","questions":{"#
                + #""0":{"criteria":{"billing":"payments","sales":null},"instructions":"Which team?","type":"choice"},"#
                + #""1":{"criteria":{"false":"can wait","true":"today"},"instructions":"Urgent?","type":"noul"},"#
                + #""2":{"criteria":["Low","High"],"instructions":"How urgent?","type":"score"}"#
                + #"},"state":"Refund \/ urgent"}"#
        )
    }

    @Test func ollamaLocalRequestMatchesFixture() async throws {
        try await expectRequest(
            [.binary(instructions: "Is this urgent?")],
            about: .json(["id": 7, "tier": "gold"]),
            from: [
                {
                    OllamaDecisionModel(
                        modelID: "nimble",
                        baseURL: $0.baseURL,
                        apiKey: nil,
                        keepAlive: 300,
                        session: StubURLProtocol.makeSession()
                    )
                },
                {
                    SystemOneDecisionModel.ollama(
                        modelID: "nimble",
                        baseURL: $0.baseURL,
                        apiKey: nil,
                        keepAlive: 300,
                        session: StubURLProtocol.makeSession()
                    )
                },
            ],
            headers: [
                "accept": "application/json",
                "content-type": "application/json",
            ],
            body: #"{"keep_alive":300,"model":"nimble","#
                + #""questions":{"0":{"instructions":"Is this urgent?","type":"noul"}},"#
                + #""state":{"id":7,"tier":"gold"}}"#
        )
    }

    @Test func ollamaCloudRequestMatchesFixture() async throws {
        try await expectRequest(
            [.score(instructions: "How urgent?", levels: ["Low", "High"])],
            about: "ticket",
            from: [
                {
                    OllamaDecisionModel(
                        modelID: "clef:cloud",
                        baseURL: $0.baseURL,
                        apiKey: "ollama-key",
                        keepAlive: 0.5,
                        session: StubURLProtocol.makeSession()
                    )
                },
                {
                    SystemOneDecisionModel.ollama(
                        modelID: "clef:cloud",
                        baseURL: $0.baseURL,
                        apiKey: "ollama-key",
                        keepAlive: 0.5,
                        session: StubURLProtocol.makeSession()
                    )
                },
            ],
            headers: [
                "accept": "application/json",
                "authorization": "Bearer ollama-key",
                "content-type": "application/json",
            ],
            body: #"{"keep_alive":0.5,"model":"clef:cloud","#
                + #""questions":{"0":{"criteria":["Low","High"],"instructions":"How urgent?","type":"score"}},"#
                + #""state":"ticket"}"#
        )
    }
}

// MARK: - Request options

extension SystemOneDecisionModelTests {
    private func systemOneOptions(
        extraBody: [String: JSONValue] = [:],
        extraHeaders: [String: String] = [:]
    ) -> DecisionOptions {
        var options = DecisionOptions()
        options[custom: SystemOneDecisionModel.self] = .init(extraBody: extraBody, extraHeaders: extraHeaders)
        return options
    }

    @Test func requestOptionsOverrideDefaultsWithSameKey() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let model = try makeModel(
            host,
            headers: ["X-Title": "Example", "traceparent": "00-default-01"],
            extraBody: ["keep_alive": 300, "provider": ["order": ["a"]]]
        )
        let session = DecisionSession(model: model, state: "ticket")

        _ = try await session.decide(
            [.binary(instructions: "Is this urgent?")],
            options: systemOneOptions(
                extraBody: ["provider": ["order": ["b"]], "session_id": "s-1"],
                extraHeaders: ["traceparent": "00-request-01"]
            )
        )

        let recorded = try #require(host.recorded.first)
        let body = try requestBody(recorded)
        #expect(body["keep_alive"] as? Int == 300)
        #expect((body["provider"] as? [String: Any])?["order"] as? [String] == ["b"])
        #expect(body["session_id"] as? String == "s-1")
        #expect(body["model"] as? String == "my-model")
        let headers = Dictionary(uniqueKeysWithValues: recorded.headers.map { ($0.key.lowercased(), $0.value) })
        #expect(headers["x-title"] == "Example")
        #expect(headers["traceparent"] == "00-request-01")
    }

    @Test func sessionOptionsReachSystemOneRequest() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let session = DecisionSession(
            model: try makeModel(host),
            state: "ticket",
            options: systemOneOptions(extraBody: ["session_id": "s-2"])
        )

        _ = try await session.probability(of: "Is this urgent?")

        let body = try requestBody(try #require(host.recorded.first))
        #expect(body["session_id"] as? String == "s-2")
    }

    @Test func requestHeaderReplacesDefaultHeaderInAnyCase() throws {
        var body: [String: JSONValue] = [:]
        var headers = ["X-Trace": "default", "X-Title": "Example"]
        let options = SystemOneDecisionModel.CustomOptions(extraHeaders: ["x-trace": "request"])

        try options.apply(to: &body, headers: &headers)

        #expect(headers == ["x-trace": "request", "X-Title": "Example"])
    }

    @Test func requestHeaderInOtherCaseReachesServer() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: try makeModel(host, headers: ["X-Trace": "default"]), state: "ticket")

        _ = try await session.decide(
            [.binary(instructions: "Is this urgent?")],
            options: systemOneOptions(extraHeaders: ["x-trace": "request"])
        )

        let recorded = try #require(host.recorded.first)
        let traces = recorded.headers.filter { $0.key.lowercased() == "x-trace" }
        #expect(traces.map(\.value) == ["request"])
    }

    @Test(arguments: ["model", "state", "questions"])
    func requestOptionsRejectReservedBodyFieldBeforeRequest(key: String) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: try makeModel(host), state: "ticket")

        await #expect(throws: SystemOneDecisionModel.ConfigurationError(kind: .reservedBodyField, key: key)) {
            _ = try await session.decide(
                [.binary(instructions: "Is this urgent?")],
                options: systemOneOptions(extraBody: [key: "x", "seed": 1])
            )
        }
        #expect(host.recorded.isEmpty)
    }

    @Test(arguments: ["Authorization", "authorization", "Content-Type", "CONTENT-TYPE"])
    func requestOptionsRejectReservedHeaderBeforeRequest(name: String) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: okResponse)
        let session = DecisionSession(model: try makeModel(host, apiKey: "secret"), state: "ticket")

        await #expect(throws: SystemOneDecisionModel.ConfigurationError(kind: .reservedHeader, key: name)) {
            _ = try await session.decide(
                [.binary(instructions: "Is this urgent?")],
                options: systemOneOptions(extraHeaders: [name: "x", "traceparent": "y"])
            )
        }
        #expect(host.recorded.isEmpty)
    }

    @Test func requestOptionsForAnotherModelTypeAreIgnored() async throws {
        var options = DecisionOptions(cacheKey: "c", userID: "u")
        options[custom: ThresholdDecisionModel.self] = .init(probability: 0.9)
        let build = { (host: StubURLProtocol.Host) -> any DecisionModel in
            try! makeModel(host, headers: ["X-Title": "Example"], extraBody: ["keep_alive": 300])
        }
        let questions: [Question] = [.binary(instructions: "Is this urgent?")]
        let plain = try await record(questions, about: "ticket", with: build)

        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"model": "m", "answers": {}}"#)
        _ = try? await DecisionSession(model: build(host), state: "ticket", options: options).decide(questions)
        let withOptions = try #require(host.recorded.first)

        #expect(withOptions.body == plain.body)
        #expect(withOptions.headers == plain.headers)
    }

    @Test(arguments: 0 ..< batches.count)
    func jevAndOllamaIgnoreSystemOneRequestOptions(index: Int) async throws {
        let (state, questions) = Self.batches[index]
        let options = systemOneOptions(extraBody: ["seed": 1], extraHeaders: ["X-Title": "Example"])
        let builders: [(StubURLProtocol.Host) -> any DecisionModel] = [
            { host in JevDecisionModel(baseURL: host.baseURL, apiKey: "key", session: StubURLProtocol.makeSession()) },
            { host in
                OllamaDecisionModel(
                    modelID: "nimble",
                    baseURL: host.baseURL,
                    apiKey: "key",
                    keepAlive: 300,
                    session: StubURLProtocol.makeSession()
                )
            },
        ]
        for build in builders {
            let plain = try await record(questions, about: state, with: build)
            let host = StubURLProtocol.Host()
            host.enqueue(json: #"{"model": "m", "answers": {}}"#)
            _ = try? await DecisionSession(model: build(host), state: state, options: options).decide(questions)
            let withOptions = try #require(host.recorded.first)

            #expect(withOptions.body == plain.body)
            #expect(withOptions.headers == plain.headers)
        }
    }
}
