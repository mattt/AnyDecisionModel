import Foundation
import Testing

@testable import AnyDecisionModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

#if canImport(CoreGraphics) && canImport(ImageIO)
    import CoreGraphics
#endif

private let pngData = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
private let photoURL = URL(string: "https://example.com/photo.jpg")!

private func encode(_ state: DecisionState) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return String(decoding: try encoder.encode(state), as: UTF8.self)
}

private func roundTrip(_ state: DecisionState) throws -> DecisionState {
    try JSONDecoder().decode(DecisionState.self, from: JSONEncoder().encode(state))
}

@Suite("DecisionState")
struct DecisionStateTests {
    // MARK: - Codable

    @Test func textAndJSONEncodeAsBareValues() throws {
        #expect(try encode(.text("hello")) == #""hello""#)
        #expect(try encode(.json(["id": 7, "tags": ["a"]])) == #"{"id":7,"tags":["a"]}"#)
        #expect(try encode(.json(["$segments": 1, "other": 2])) == #"{"$segments":1,"other":2}"#)
        #expect(try roundTrip(.text("hello")) == .text("hello"))
        #expect(try roundTrip(.json(["id": 7])) == .json(["id": 7]))
    }

    @Test func segmentsEncodeWithTags() throws {
        let state = DecisionState.segments([
            .text("A photo:"),
            .attachment(DecisionState.Attachment(imageData: pngData, mimeType: "image/png")),
            .attachment(DecisionState.Attachment(imageURL: photoURL, mimeType: "image/jpeg")),
        ])
        let object = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(state)) as? [String: Any]
        )
        #expect(object.keys.sorted() == ["$segments"])
        let segments = try #require(object["$segments"] as? [[String: Any]])
        #expect(segments.count == 3)
        #expect(segments[0]["kind"] as? String == "text")
        #expect(segments[0]["text"] as? String == "A photo:")

        #expect(segments[1]["kind"] as? String == "attachment")
        let inline = try #require(segments[1]["attachment"] as? [String: Any])
        #expect(inline["kind"] as? String == "image")
        let inlineSource = try #require(inline["source"] as? [String: String])
        #expect(inlineSource == ["kind": "data", "data": pngData.base64EncodedString(), "mimeType": "image/png"])

        let linked = try #require(segments[2]["attachment"] as? [String: Any])
        let linkedSource = try #require(linked["source"] as? [String: String])
        #expect(linkedSource == ["kind": "url", "url": photoURL.absoluteString, "mimeType": "image/jpeg"])

        #expect(try roundTrip(state) == state)
    }

    @Test(arguments: DecisionState.Attachment.Kind.allCases)
    func everyAttachmentKindRoundTrips(kind: DecisionState.Attachment.Kind) throws {
        let state = DecisionState.segments([
            .attachment(DecisionState.Attachment(kind: kind, source: .data(pngData, mimeType: "x/y"))),
            .attachment(DecisionState.Attachment(kind: kind, source: .url(photoURL, mimeType: "x/y"))),
        ])
        #expect(try roundTrip(state) == state)
    }

    @Test func textOnlySegmentsStaySegments() throws {
        let state = DecisionState.segments([.text("a"), .text("b")])
        #expect(try roundTrip(state) == state)
    }

    @Test(
        arguments: [
            JSONValue.object(["$segments": .array([])]),
            JSONValue.object(["$segments": .array([.object(["kind": "text", "text": "x"])])]),
            JSONValue.object(["$json": 1]),
            JSONValue.object(["$json": .object(["$json": 1])]),
        ]
    )
    func jsonThatLooksTaggedRoundTripsAsJSON(value: JSONValue) throws {
        let state = DecisionState.json(value)
        #expect(try roundTrip(state) == state)
        #expect(try encode(state).hasPrefix(#"{"$json":"#))
    }

    @Test func unknownSegmentKindDecodesAsJSON() throws {
        let json = #"{"$segments": [{"kind": "sticker"}]}"#
        let state = try JSONDecoder().decode(DecisionState.self, from: Data(json.utf8))
        #expect(state == .json(["$segments": [["kind": "sticker"]]]))
    }

    // MARK: - Data encoded by 0.7.0

    // AnyDecisionModel 0.7.0 encoded every JSON state as its bare value.

    @Test(
        arguments: [
            (#"{"$segments":1}"#, DecisionState.json(["$segments": 1])),
            (#"{"$segments":"x"}"#, DecisionState.json(["$segments": "x"])),
            (#"{"$segments":1,"other":2}"#, DecisionState.json(["$segments": 1, "other": 2])),
        ]
    )
    func legacySegmentsKeyDecodesAsJSON(json: String, expected: DecisionState) throws {
        #expect(try JSONDecoder().decode(DecisionState.self, from: Data(json.utf8)) == expected)
    }

    @Test func legacyJSONKeyDecodesAsEscapedValue() throws {
        // This is the documented difference from 0.7.0, which decoded this object as itself.
        let state = try JSONDecoder().decode(DecisionState.self, from: Data(#"{"$json":1}"#.utf8))
        #expect(state == .json(1))
    }

    // MARK: - Equality

    @Test func attachmentEquality() {
        let image = DecisionState.Attachment(imageData: pngData, mimeType: "image/png")
        #expect(image == DecisionState.Attachment(imageData: Data(pngData), mimeType: "image/png"))
        #expect(image.hashValue == DecisionState.Attachment(imageData: Data(pngData), mimeType: "image/png").hashValue)
        #expect(image != DecisionState.Attachment(imageData: pngData, mimeType: "image/jpeg"))
        #expect(image != DecisionState.Attachment(kind: .document, source: .data(pngData, mimeType: "image/png")))
        #expect(
            DecisionState.Attachment(imageURL: photoURL, mimeType: "image/jpeg")
                != DecisionState.Attachment(imageData: pngData, mimeType: "image/jpeg")
        )
        #expect(DecisionState.segments([.text("a")]) != DecisionState.text("a"))
    }

    @Test func documentDefaultsToPDF() {
        #expect(DecisionState.Attachment(documentData: pngData).mimeType == "application/pdf")
        #expect(DecisionState.Attachment(documentURL: photoURL).mimeType == "application/pdf")
    }

    // MARK: - Prompt text

    @Test func promptTextIsUnchangedForTextAndJSON() throws {
        #expect(try DecisionState.text("hello").textOrJSONValue().promptText == "hello")
        #expect(try DecisionState.json(["b": 1, "a": 2]).textOrJSONValue().promptText == #"{"a":2,"b":1}"#)
        #expect(try DecisionState.json(["$json": 1]).textOrJSONValue() == .object(["$json": 1]))
    }

    @Test func textOnlySegmentsReadAsJoinedText() throws {
        let state = DecisionState.segments([.text("first"), .text("second")])
        #expect(try state.textOrJSONValue().promptText == "first\nsecond")
        #expect(state.attachments.isEmpty)
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
        @Test func cgImageEncodesAsPNG() throws {
            let context = try #require(
                CGContext(
                    data: nil,
                    width: 2,
                    height: 2,
                    bitsPerComponent: 8,
                    bytesPerRow: 0,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                )
            )
            let image = try #require(context.makeImage())
            let attachment = try DecisionState.Attachment(image)
            #expect(attachment.kind == .image)
            #expect(attachment.mimeType == "image/png")
            guard case .data(let data, _) = attachment.source else {
                Issue.record("Expected inline data.")
                return
            }
            #expect(data.prefix(8) == pngData)
        }
    #endif
}

// MARK: - Backends without attachment support

@Suite("DecisionState attachments in other backends")
struct UnsupportedAttachmentTests {
    private let state = DecisionState.segments([
        .text("A photo:"),
        .attachment(DecisionState.Attachment(imageData: pngData, mimeType: "image/png")),
    ])

    @Test func ollamaRejectsAttachmentsBeforeRequest() async throws {
        let host = StubURLProtocol.Host()
        let model = OllamaDecisionModel(
            modelID: "nimble",
            baseURL: host.baseURL,
            session: StubURLProtocol.makeSession()
        )
        let session = DecisionSession(model: model, state: state)

        await #expect(
            throws: DecisionError.unsupportedState("OllamaDecisionModel does not support image attachments.")
        ) {
            _ = try await session.probability(of: "Is this urgent?")
        }
        await #expect(throws: DecisionError.self) {
            try await session.prewarm()
        }
        #expect(host.recorded.isEmpty)
    }

    @Test(arguments: ["jev", "ollama"])
    func executorRejectsAttachmentsWithoutSession(backend: String) async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"model": "m", "answers": {"0": {"noul": 0.5}}}"#)
        let model: any DecisionModel =
            backend == "jev"
            ? JevDecisionModel(baseURL: host.baseURL, apiKey: "key", session: StubURLProtocol.makeSession())
            : OllamaDecisionModel(modelID: "nimble", baseURL: host.baseURL, session: StubURLProtocol.makeSession())

        do {
            _ = try await model.makeExecutor().decide([.binary(instructions: "Is this urgent?")], about: state)
            Issue.record("Expected an error.")
        } catch DecisionError.unsupportedState {
        }
        #expect(host.recorded.isEmpty)
    }

    @Test func promptsRejectAttachments() {
        #expect(throws: DecisionError.self) {
            _ = try TokenScoring.userMessage(state: state, question: .binary(instructions: "Urgent?"))
        }
        #expect(throws: DecisionError.self) {
            _ = try JointSchemaEncoding.prefixTokens(for: state) { _ in [] }
        }
    }

    @Test func jevSendsTextOnlySegmentsAsString() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"model": "jev", "answers": {"0": {"noul": 0.5}}}"#)
        let model = JevDecisionModel(baseURL: host.baseURL, apiKey: "key", session: StubURLProtocol.makeSession())
        let session = DecisionSession(model: model, state: .segments([.text("first"), .text("second")]))

        _ = try await session.probability(of: "Is this urgent?")

        let recorded = try #require(host.recorded.first)
        let body = try #require(try JSONSerialization.jsonObject(with: recorded.body) as? [String: Any])
        #expect(body["state"] as? String == "first\nsecond")
    }

    @Test func jevSendsTaggedLookingJSONUnchanged() async throws {
        let host = StubURLProtocol.Host()
        host.enqueue(json: #"{"model": "jev", "answers": {"0": {"noul": 0.5}}}"#)
        let model = JevDecisionModel(baseURL: host.baseURL, apiKey: "key", session: StubURLProtocol.makeSession())
        let session = DecisionSession(model: model, state: .json(["$json": 1]))

        _ = try await session.probability(of: "Is this urgent?")

        let recorded = try #require(host.recorded.first)
        let body = try #require(try JSONSerialization.jsonObject(with: recorded.body) as? [String: Any])
        let state = try #require(body["state"] as? [String: Any])
        #expect(state["$json"] as? Int == 1)
    }
}
