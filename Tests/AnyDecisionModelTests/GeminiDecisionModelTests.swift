import Foundation
import Testing

@testable import AnyDecisionModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

private func makeModel(
    _ host: StubURLProtocol.Host,
    apiKey: String? = "test-key"
) -> GeminiDecisionModel {
    GeminiDecisionModel(
        baseURL: host.baseURL,
        apiKey: apiKey,
        session: StubURLProtocol.makeSession()
    )
}

/// Sends questions through a Gemini model and returns the request body.
private func requestBody(
    state: DecisionState = "state",
    questions: [Question]
) async throws -> [String: Any] {
    let host = StubURLProtocol.Host()
    // The response has no answers, so the call throws after the request is recorded.
    host.enqueue(json: #"{"answers": {}}"#)
    _ = try? await DecisionSession(model: makeModel(host), state: state).decide(questions)
    let recorded = try #require(host.recorded.first)
    return try #require(try JSONSerialization.jsonObject(with: recorded.body) as? [String: Any])
}

private func requestQuestion(_ question: Question) async throws -> [String: Any] {
    let body = try await requestBody(questions: [question])
    let questions = try #require(body["questions"] as? [String: [String: Any]])
    return try #require(questions["0"])
}

/// A response with answers to a binary, a choice, and a score question, keyed "0" to "2".
private let batchResponse = """
    {
      "answers": {
        "0": {"type": "noul", "noul": 0.93, "confidence": 0.93},
        "1": {"type": "choice", "choice": "billing",
              "probabilities": {"billing": 0.85, "technical": 0.1, "sales": 0.05},
              "confidence": 0.85},
        "2": {"type": "score", "score": 1.4,
              "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
              "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
              "confidence": 0.5}
      },
      "usage": {
        "total_input_tokens": 512,
        "input_tokens_by_modality": [{"modality": "text", "tokens": 512}],
        "total_output_tokens": 0,
        "total_tokens": 512
      }
    }
    """

private let teamOptions = [
    ChoiceOption("billing", description: "payments, charges, refunds"),
    ChoiceOption("technical"),
    ChoiceOption("sales", description: "pricing questions"),
]

private let angerLevels = ["Calm", "Frustrated", "Very angry"]

@Suite("GeminiDecisionModel")
struct GeminiDecisionModelTests {
    // MARK: - Requests

    @Test func sendsRequestToDecisionsEndpointWithAPIKeyHeader() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"answers": {"0": {"type": "noul", "noul": 0.75, "confidence": 0.75}}}"#)
        let session = DecisionSession(model: makeModel(host), state: .text("A 6 sided die rolled a 3."))

        let probability = try await session.probability(of: "Is the number rolled odd?")
        #expect(probability == 0.75)

        let recorded = try #require(host.recorded.first)
        // FoundationNetworking on Linux changes the case of header names.
        let headers = Dictionary(uniqueKeysWithValues: recorded.headers.map { ($0.key.lowercased(), $0.value) })
        #expect(headers["x-goog-api-key"] == "test-key")
        #expect(headers["authorization"] == nil)
        #expect(headers["content-type"] == "application/json")

        let body = try #require(try JSONSerialization.jsonObject(with: recorded.body) as? [String: Any])
        #expect(body["model"] as? String == "gemini-decision-1")
        #expect(body["input"] as? String == "A 6 sided die rolled a 3.")
        #expect(body["state"] == nil)
    }

    @Test func requestURLIsDecisionsEndpointWithoutAPIKey() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"answers": {"0": {"type": "noul", "noul": 0.5, "confidence": 0.5}}}"#)
        let session = DecisionSession(model: makeModel(host), state: "state")
        _ = try await session.probability(of: "Question?")

        let url = try #require(host.recorded.first?.url)
        #expect(url.path == "/v1beta/decisions")
        #expect(url.query == nil)
        #expect(!url.absoluteString.contains("test-key"))
    }

    @Test func binaryQuestionUsesNoulAndDescriptionKeys() async throws {
        let object = try await requestQuestion(
            .binary(
                instructions: "Urgent?",
                criteria: BinaryCriteria(whenTrue: "Needs action today", whenFalse: "Can wait")
            )
        )
        #expect(object["type"] as? String == "noul")
        #expect(object["instructions"] as? String == "Urgent?")
        let criteria = try #require(object["criteria"] as? [String: Any])
        #expect(criteria.keys.sorted() == ["false_description", "true_description"])
        #expect(criteria["true_description"] as? String == "Needs action today")
        #expect(criteria["false_description"] as? String == "Can wait")
    }

    @Test func binaryQuestionWithoutCriteriaOmitsThem() async throws {
        let object = try await requestQuestion(.binary(instructions: "Urgent?"))
        #expect(object["criteria"] == nil)
    }

    @Test func choiceCriteriaMapOptionsToDescriptionsOrEmptyString() async throws {
        let object = try await requestQuestion(.choice(instructions: "Team?", options: teamOptions))
        #expect(object["type"] as? String == "choice")
        let criteria = try #require(object["criteria"] as? [String: Any])
        #expect(criteria["billing"] as? String == "payments, charges, refunds")
        #expect(criteria["technical"] as? String == "")
        #expect(criteria["sales"] as? String == "pricing questions")
    }

    @Test func scoreCriteriaAreAnOrderedArray() async throws {
        let object = try await requestQuestion(.score(instructions: "Anger?", levels: angerLevels))
        #expect(object["type"] as? String == "score")
        #expect(object["criteria"] as? [String] == angerLevels)
    }

    @Test func structuredStateIsSentAsJSONText() async throws {
        let body = try await requestBody(
            state: .json(["ticket": "charged twice", "priority": 2, "tags": ["billing"]]),
            questions: [.binary(instructions: "Urgent?")]
        )
        #expect(body["input"] as? String == #"{"priority":2,"tags":["billing"],"ticket":"charged twice"}"#)
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

    // MARK: - Responses

    @Test func decodesAnswersAndUsage() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: batchResponse)
        let session = DecisionSession(model: makeModel(host), state: "ticket")

        let response = try await session.decide([
            .binary(instructions: "Urgent?"),
            .choice(instructions: "Team?", options: teamOptions),
            .score(instructions: "Anger?", levels: angerLevels),
        ])

        #expect(response.modelID == "gemini-decision-1")
        #expect(response.usage == DecisionSession.Usage(inputTokenCount: 512, outputTokenCount: 0))
        #expect(
            response.answers == [
                .binary(probability: 0.93),
                .choice(
                    name: "billing",
                    probabilities: ["billing": 0.85, "technical": 0.1, "sales": 0.05],
                    confidence: 0.85
                ),
                .score(value: 1.4, probabilities: [0.1, 0.4, 0.5], legend: angerLevels, confidence: 0.5),
            ]
        )
        #expect(response.diagnostics.isEmpty)
    }

    @Test func missingUsageDecodesAsZero() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"answers": {"0": {"type": "noul", "noul": 0.5, "confidence": 0.5}}}"#)
        let session = DecisionSession(model: makeModel(host), state: "state")
        let response = try await session.decide([.binary(instructions: "x")])
        #expect(response.usage == DecisionSession.Usage())
    }

    @Test func batchOrderDoesNotChangeAnswers() async throws {
        let questions: [Question] = [
            .binary(instructions: "Urgent?"),
            .choice(instructions: "Team?", options: teamOptions),
            .score(instructions: "Anger?", levels: angerLevels),
        ]
        let answers: [String] = [
            #"{"type": "noul", "noul": 0.93, "confidence": 0.93}"#,
            #"""
            {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.85, "technical": 0.1, "sales": 0.05}, "confidence": 0.85}
            """#,
            #"""
            {"type": "score", "score": 1.4, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
             "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"}, "confidence": 0.5}
            """#,
        ]

        func run(_ order: [Int]) async throws -> [Answer] {
            let host = StubURLProtocol.Host()
            // Return the answers in reverse key order to check that keys, not order, matter.
            let entries = order.indices.reversed().map { position in
                "\"\(position)\": \(answers[order[position]])"
            }
            host.enqueue(json: #"{"answers": {"# + entries.joined(separator: ", ") + "}}")
            let session = DecisionSession(model: makeModel(host), state: "ticket")
            let response = try await session.decide(order.map { questions[$0] })

            let recorded = try #require(host.recorded.first)
            let body = try #require(try JSONSerialization.jsonObject(with: recorded.body) as? [String: Any])
            let sent = try #require(body["questions"] as? [String: [String: Any]])
            for (position, index) in order.enumerated() {
                #expect(sent[String(position)]?["instructions"] as? String == questions[index].instructions)
            }

            // Put the answers back in the original question order.
            var byQuestion = [Answer?](repeating: nil, count: questions.count)
            for (position, index) in order.enumerated() {
                byQuestion[index] = response.answers[position]
            }
            return byQuestion.compactMap { $0 }
        }

        let forward = try await run([0, 1, 2])
        let shuffled = try await run([2, 0, 1])
        #expect(forward.count == 3)
        #expect(forward == shuffled)
    }

    // MARK: - Blocked input and errors

    @Test func blockedInputWithPromptFeedback() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"answers": {},
                 "prompt_feedback": {"block_reason": "safety",
                                     "block_reason_message": "The input was blocked for safety."},
                 "usage": {"total_input_tokens": 12, "total_output_tokens": 0, "total_tokens": 12}}
                """
        )
        let session = DecisionSession(model: makeModel(host), state: "state")
        await #expect(
            throws: DecisionError.inputBlocked(reason: "safety", message: "The input was blocked for safety.")
        ) {
            _ = try await session.probability(of: "Question?")
        }
    }

    @Test func blockedInputWithPromptFeedbackAndNoAnswers() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"prompt_feedback": {"block_reason": "blocklist"}}"#)
        let session = DecisionSession(model: makeModel(host), state: "state")
        await #expect(throws: DecisionError.inputBlocked(reason: "blocklist", message: nil)) {
            _ = try await session.probability(of: "Question?")
        }
    }

    @Test func blockedInputWithHTTP400() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"error": {"code": "safety",
                           "message": "Request blocked due to safety violations (harmful content)."}}
                """,
            statusCode: 400
        )
        let session = DecisionSession(model: makeModel(host), state: "state")
        await #expect(
            throws: DecisionError.inputBlocked(
                reason: "safety",
                message: "Request blocked due to safety violations (harmful content)."
            )
        ) {
            _ = try await session.probability(of: "Question?")
        }
    }

    @Test(arguments: [400, 401, 429, 500])
    func reportsHTTPErrors(statusCode: Int) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: #"{"error": {"code": \#(statusCode), "message": "Too many questions."}}"#,
            statusCode: statusCode,
            headers: ["X-Request-Id": "abc"]
        )
        let session = DecisionSession(model: makeModel(host), state: "state")

        do {
            _ = try await session.probability(of: "Question?")
            Issue.record("Expected an error.")
        } catch let DecisionError.requestFailed(code, detail, headers) {
            #expect(code == statusCode)
            #expect(detail.contains("Too many questions."))
            #expect(!detail.contains("test-key"))
            #expect(headers["x-request-id"] == "abc")
        }
        #expect(host.recorded.count == 1)
    }

    @Test func missingAnswer() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"answers": {"other": {"type": "noul", "noul": 0.5, "confidence": 0.5}}}"#)
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

    // MARK: - Limits

    @Test func rejectsTooManyQuestionsBeforeRequest() async throws {
        let host = StubURLProtocol.Host()
        let session = DecisionSession(model: makeModel(host), state: "state")
        let questions = Array(repeating: Question.binary(instructions: "x"), count: 1001)
        do {
            _ = try await session.decide(questions)
            Issue.record("Expected an error.")
        } catch DecisionError.unsupportedQuestion(let index, _) {
            #expect(index == 1000)
        }
        #expect(host.recorded.isEmpty)
    }

    @Test func rejectsChoiceWithOneOption() async throws {
        let host = StubURLProtocol.Host()
        let session = DecisionSession(model: makeModel(host), state: "state")
        do {
            _ = try await session.decide(.choice(instructions: "x", options: [ChoiceOption("only")]))
            Issue.record("Expected an error.")
        } catch DecisionError.unsupportedQuestion(let index, _) {
            #expect(index == 0)
        }
        #expect(host.recorded.isEmpty)
    }

    @Test func rejectsTooManyOptionsAndLevels() async throws {
        let host = StubURLProtocol.Host()
        let session = DecisionSession(model: makeModel(host), state: "state")
        let options = (0 ..< 256).map { ChoiceOption("option\($0)") }
        let levels = (0 ..< 256).map { "level \($0)" }

        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(.choice(instructions: "x", options: options))
        }
        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(.score(instructions: "x", levels: levels))
        }
        await #expect(throws: DecisionError.self) {
            _ = try await session.decide(.score(instructions: "x", levels: ["low", ""]))
        }
        #expect(host.recorded.isEmpty)
    }

    @Test(
        arguments: [
            "",
            String(repeating: "a", count: 129),
            " leading",
            "trailing ",
            "tab\tinside",
            "no\u{00A0}break",
        ]
    )
    func rejectsInvalidOptionNames(name: String) {
        #expect(GeminiDecisionModel.invalidOptionNameReason(name) != nil)
    }

    @Test func acceptsValidOptionNames() {
        #expect(GeminiDecisionModel.invalidOptionNameReason("billing") == nil)
        #expect(GeminiDecisionModel.invalidOptionNameReason("other team") == nil)
        #expect(GeminiDecisionModel.invalidOptionNameReason(String(repeating: "a", count: 128)) == nil)
        // 128 four-byte characters are 512 bytes, the limit.
        #expect(GeminiDecisionModel.invalidOptionNameReason(String(repeating: "\u{1F600}", count: 128)) == nil)
    }

    // MARK: - Attachments

    @Test(arguments: DecisionState.Attachment.Kind.allCases)
    func sendsInlineAttachmentAsBase64Block(kind: DecisionState.Attachment.Kind) async throws {
        let mimeType = try #require(GeminiDecisionModel.supportedMIMETypes[kind]?.sorted().first)
        let bytes = Data([0x01, 0x02, 0x03, 0xFF])
        let body = try await requestBody(
            state: .segments([
                .text("Look at this:"),
                .attachment(DecisionState.Attachment(kind: kind, source: .data(bytes, mimeType: mimeType))),
            ]),
            questions: [.binary(instructions: "Urgent?")]
        )
        let input = try #require(body["input"] as? [[String: String]])
        #expect(
            input == [
                ["type": "text", "text": "Look at this:"],
                ["type": kind.rawValue, "data": bytes.base64EncodedString(), "mime_type": mimeType],
            ]
        )
    }

    @Test(arguments: DecisionState.Attachment.Kind.allCases)
    func sendsURLAttachmentAsURIBlock(kind: DecisionState.Attachment.Kind) async throws {
        let mimeType = try #require(GeminiDecisionModel.supportedMIMETypes[kind]?.sorted().first)
        let url = URL(string: "https://example.com/media/file?sig=abc")!
        let body = try await requestBody(
            state: .segments([
                .attachment(DecisionState.Attachment(kind: kind, source: .url(url, mimeType: mimeType)))
            ]),
            questions: [.binary(instructions: "Urgent?")]
        )
        let input = try #require(body["input"] as? [[String: String]])
        #expect(input == [["type": kind.rawValue, "uri": url.absoluteString, "mime_type": mimeType]])
    }

    @Test func joinsAdjacentTextSegmentsWithNewline() async throws {
        let image = DecisionState.Attachment(imageData: Data([1]), mimeType: "image/png")
        let body = try await requestBody(
            state: .segments([
                .text("Ticket 1"),
                .text(""),
                .text("Ticket 2"),
                .attachment(image),
                .text(""),
                .attachment(image),
                .text("End"),
            ]),
            questions: [.binary(instructions: "Urgent?")]
        )
        let input = try #require(body["input"] as? [[String: String]])
        #expect(input.map { $0["type"] } == ["text", "image", "image", "text"])
        #expect(input[0]["text"] == "Ticket 1\nTicket 2")
        #expect(input[3]["text"] == "End")
    }

    @Test func textOnlySegmentsAreSentAsString() async throws {
        let body = try await requestBody(
            state: .segments([.text("first"), .text("second")]),
            questions: [.binary(instructions: "Urgent?")]
        )
        #expect(body["input"] as? String == "first\nsecond")
    }

    @Test(
        arguments: [
            DecisionState.Attachment(imageData: Data([1]), mimeType: "image/svg+xml"),
            DecisionState.Attachment(imageData: Data([1]), mimeType: "application/octet-stream"),
            DecisionState.Attachment(audioData: Data([1]), mimeType: "video/mp4"),
            DecisionState.Attachment(documentData: Data([1]), mimeType: "text/csv"),
            DecisionState.Attachment(imageURL: URL(string: "http://example.com/a.png")!, mimeType: "image/png"),
            DecisionState.Attachment(imageURL: URL(fileURLWithPath: "/tmp/a.png"), mimeType: "image/png"),
        ]
    )
    func rejectsUnsupportedAttachmentsBeforeRequest(attachment: DecisionState.Attachment) async throws {
        let host = StubURLProtocol.Host()
        let session = DecisionSession(model: makeModel(host), state: .segments([.attachment(attachment)]))
        do {
            _ = try await session.probability(of: "Urgent?")
            Issue.record("Expected an error.")
        } catch DecisionError.unsupportedState {
        }
        #expect(host.recorded.isEmpty)
    }

    @Test func acceptsMIMETypeWithParametersAndCase() throws {
        try GeminiDecisionModel.check(
            .segments([
                .attachment(DecisionState.Attachment(audioData: Data([1]), mimeType: "audio/L16; rate=16000")),
                .attachment(DecisionState.Attachment(imageData: Data([1]), mimeType: "IMAGE/PNG")),
            ])
        )
    }

    @Test func checksInlineSizeLimits() throws {
        let megabyte = 1024 * 1024
        func state(_ attachments: DecisionState.Attachment...) -> DecisionState {
            .segments(attachments.map { .attachment($0) })
        }

        try GeminiDecisionModel.check(
            state(DecisionState.Attachment(videoData: Data(count: 100 * megabyte), mimeType: "video/mp4"))
        )
        #expect(throws: DecisionError.self) {
            try GeminiDecisionModel.check(
                state(
                    DecisionState.Attachment(videoData: Data(count: 60 * megabyte), mimeType: "video/mp4"),
                    DecisionState.Attachment(imageData: Data(count: 40 * megabyte + 1), mimeType: "image/png")
                )
            )
        }

        try GeminiDecisionModel.check(state(DecisionState.Attachment(documentData: Data(count: 50 * megabyte))))
        #expect(throws: DecisionError.self) {
            try GeminiDecisionModel.check(
                state(
                    DecisionState.Attachment(documentData: Data(count: 25 * megabyte)),
                    DecisionState.Attachment(documentData: Data(count: 25 * megabyte + 1))
                )
            )
        }

        // URLs do not count toward the inline limits.
        try GeminiDecisionModel.check(
            state(
                DecisionState.Attachment(videoData: Data(count: 100 * megabyte), mimeType: "video/mp4"),
                DecisionState.Attachment(
                    videoURL: URL(string: "https://example.com/v.mp4")!,
                    mimeType: "video/mp4"
                )
            )
        )
    }

    // MARK: - Configuration

    @Test func environmentAPIKey() {
        #expect(GeminiDecisionModel.environmentAPIKey(from: ["GEMINI_API_KEY": " abc "]) == "abc")
        #expect(GeminiDecisionModel.environmentAPIKey(from: ["GEMINI_API_KEY": "  "]) == nil)
        #expect(GeminiDecisionModel.environmentAPIKey(from: [:]) == nil)
    }

    @Test func descriptionOmitsAPIKey() {
        let model = GeminiDecisionModel(apiKey: "secret-value")
        #expect(!String(describing: model).contains("secret-value"))
        #expect(!String(reflecting: model).contains("secret-value"))
        #expect(model.modelID == "gemini-decision-1")
        #expect(model.baseURL == URL(string: "https://generativelanguage.googleapis.com")!)
    }
}

// MARK: - Live

@Suite(
    "GeminiDecisionModel live",
    .enabled(if: GeminiDecisionModel.environmentAPIKey() != nil)
)
struct GeminiDecisionModelLiveTests {
    @Test func binaryChoiceAndScore() async throws {
        let session = DecisionSession(
            model: GeminiDecisionModel(),
            state: .text("Our checkout has returned 500 errors since 9am. Customers cannot pay.")
        )

        let response = try await session.decide([
            .binary(instructions: "Is this ticket about a software error?"),
            .choice(instructions: "Which team should handle this ticket?", options: teamOptions),
            .score(instructions: "How urgent is this ticket?", levels: ["Routine", "Soon", "Immediate"]),
        ])

        guard case .binary(let probability) = response.answers[0] else {
            Issue.record("Expected a binary answer.")
            return
        }
        #expect((0.01 ... 0.99).contains(probability))
        #expect(response.usage.inputTokenCount > 0)
    }

    @Test func image() async throws {
        // A 16 by 16 pixel PNG image filled with red.
        let png = try #require(
            Data(
                base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR42mP4z8BAEmIY1TCqYfhqAACQ"
                    + "+f8B8u7oVwAAAABJRU5ErkJggg=="
            )
        )
        let session = DecisionSession(
            model: GeminiDecisionModel(),
            state: .segments([
                .text("An image:"),
                .attachment(DecisionState.Attachment(imageData: png, mimeType: "image/png")),
            ])
        )

        let red = try await session.probability(of: "Is the image red?")
        #expect(red > 0.5)
    }
}
