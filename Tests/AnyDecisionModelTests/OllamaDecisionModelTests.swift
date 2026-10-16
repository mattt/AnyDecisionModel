import Foundation
import Testing

@testable import AnyDecisionModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

private func makeModel(
    _ host: StubURLProtocol.Host,
    apiKey: String? = nil,
    modelID: String = "nimble",
    keepAlive: TimeInterval? = nil
) -> OllamaDecisionModel {
    OllamaDecisionModel(
        modelID: modelID,
        baseURL: host.baseURL,
        apiKey: apiKey,
        keepAlive: keepAlive,
        session: StubURLProtocol.makeSession()
    )
}

private func requestBody(_ host: StubURLProtocol.Host) throws -> [String: Any] {
    let recorded = try #require(host.recorded.first)
    return try #require(try JSONSerialization.jsonObject(with: recorded.body) as? [String: Any])
}

@Suite("OllamaDecisionModel")
struct OllamaDecisionModelTests {
    @Test func sendsRequestAndDecodesAnswers() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"model": "nimble",
                 "answers": {
                   "0": {"type": "choice", "choice": "bug",
                         "probabilities": {"billing": 0.0125, "bug": 0.9781, "account": 0.0094},
                         "confidence": 0.8906},
                   "1": {"type": "noul", "noul": 0.959}
                 },
                 "usage": {"input_tokens": 174, "output_tokens": 1}}
                """
        )
        let session = DecisionSession(
            model: makeModel(host),
            state: .text("Our checkout has returned 500 errors since 9am.")
        )

        let response = try await session.decide([
            .choice(
                instructions: "Which label fits this ticket?",
                options: [
                    ChoiceOption("billing", description: "Payments and refunds"),
                    ChoiceOption("bug", description: "Software errors"),
                    ChoiceOption("account"),
                ]
            ),
            .binary(instructions: "Is the customer requesting a refund?"),
        ])

        #expect(response.modelID == "nimble")
        #expect(response.usage.inputTokenCount == 174)
        #expect(response.usage.outputTokenCount == 1)
        #expect(
            response.answers[0]
                == .choice(
                    name: "bug",
                    probabilities: ["billing": 0.0125, "bug": 0.9781, "account": 0.0094],
                    confidence: 0.8906
                )
        )
        #expect(response.answers[1] == .binary(probability: 0.959))

        let recorded = try #require(host.recorded.first)
        #expect(recorded.headers["Authorization"] == nil)

        let body = try requestBody(host)
        #expect(body["model"] as? String == "nimble")
        #expect(body["state"] as? String == "Our checkout has returned 500 errors since 9am.")
        #expect(body["keep_alive"] == nil)
        let questions = try #require(body["questions"] as? [String: [String: Any]])
        #expect(questions["0"]?["type"] as? String == "choice")
        #expect(questions["1"]?["type"] as? String == "noul")
        let criteria = try #require(questions["0"]?["criteria"] as? [String: Any])
        #expect(criteria["billing"] as? String == "Payments and refunds")
        #expect(criteria["account"] is NSNull)
    }

    @Test func decodesScoreLegendObject() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"model": "nimble",
                 "answers": {
                   "0": {"type": "score", "score": 0.8308,
                         "legend": {"0": "Routine", "1": "Soon", "2": "Immediate"},
                         "probabilities": {"0": 0.25, "1": 0.6692, "2": 0.0808},
                         "confidence": 0.1234}
                 },
                 "usage": {"input_tokens": 200, "output_tokens": 2}}
                """
        )
        let session = DecisionSession(model: makeModel(host), state: "ticket")

        let response = try await session.decide([
            .score(instructions: "How urgent is this?", levels: ["Routine", "Soon", "Immediate"])
        ])

        #expect(
            response.answers[0]
                == .score(
                    value: 0.8308,
                    probabilities: [0.25, 0.6692, 0.0808],
                    legend: ["Routine", "Soon", "Immediate"],
                    confidence: 0.1234
                )
        )
    }

    @Test func sendsAPIKeyAndKeepAlive() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"model": "clef:cloud", "answers": {"0": {"type": "noul", "noul": 0.5}},
                 "usage": {"input_tokens": 10, "output_tokens": 1}}
                """
        )
        let model = makeModel(host, apiKey: "test-key", modelID: "clef:cloud", keepAlive: 300)
        let session = DecisionSession(model: model, state: "ticket")

        let probability = try await session.probability(of: "Is this urgent?")
        #expect(probability == 0.5)

        let recorded = try #require(host.recorded.first)
        #expect(recorded.headers["Authorization"] == "Bearer test-key")
        #expect(!model.description.contains("test-key"))

        let body = try requestBody(host)
        #expect(body["model"] as? String == "clef:cloud")
        #expect((body["keep_alive"] as? NSNumber)?.doubleValue == 300)
    }

    @Test func environmentAPIKey() {
        #expect(OllamaDecisionModel.environmentAPIKey(from: ["OLLAMA_API_KEY": " abc "]) == "abc")
        #expect(OllamaDecisionModel.environmentAPIKey(from: ["OLLAMA_API_KEY": "  "]) == nil)
        #expect(OllamaDecisionModel.environmentAPIKey(from: [:]) == nil)
    }

    @Test func errorResponse() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"error": "model 'nope' not found"}"#, statusCode: 404)
        let session = DecisionSession(model: makeModel(host, modelID: "nope"), state: "ticket")

        do {
            _ = try await session.probability(of: "Is this urgent?")
            Issue.record("Expected the request to fail.")
        } catch let DecisionError.requestFailed(statusCode, detail, _) {
            #expect(statusCode == 404)
            #expect(detail.contains("not found"))
        }
    }

    @Test func missingAnswer() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(
            json: """
                {"model": "nimble", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}}
                """
        )
        let session = DecisionSession(model: makeModel(host), state: "ticket")

        await #expect(throws: DecisionError.invalidResponse("No answer for the question at index 0.")) {
            _ = try await session.probability(of: "Is this urgent?")
        }
    }
}
