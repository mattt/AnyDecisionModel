import Foundation
import Testing

// This file imports the module without `@testable`,
// so it compiles only against the public API, as a client does.
import AnyDecisionModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// Checks that code written against the version 0.9 `SystemOneDecisionModel` initializer
/// still compiles with the same meaning.
@Suite("SystemOneDecisionModel source compatibility")
struct SystemOneDecisionModelCompatibilityTests {
    let baseURL = URL(string: "https://example.invalid")!

    @Test func versionZeroNineUnqualifiedReferenceCompiles() throws {
        let makeModel = SystemOneDecisionModel.init
        let model = try makeModel(
            baseURL,
            "v1/systemone",
            "m",
            nil,
            [:],
            DecisionCapabilities(),
            [:],
            URLSession(configuration: .default),
            .default
        )
        #expect(model.imageEncoding == nil)
        #expect(model.capabilities.supportedAttachmentKinds.isEmpty)
    }

    @Test func versionZeroNineTypedReferenceCompiles() throws {
        let makeModel:
            (
                URL, String, String, @escaping @autoclosure @Sendable () -> String?, [String: String],
                DecisionCapabilities, [String: JSONValue], URLSession, RetryPolicy
            ) throws -> SystemOneDecisionModel = SystemOneDecisionModel.init
        let model = try makeModel(
            baseURL,
            "v1/systemone",
            "m",
            "key",
            [:],
            DecisionCapabilities(maximumQuestions: 4),
            ["seed": 1],
            URLSession(configuration: .default),
            .never
        )
        #expect(model.imageEncoding == nil)
        #expect(model.capabilities.maximumQuestions == 4)
        #expect(model.extraBody == ["seed": 1])
    }

    @Test func versionZeroNineLabeledReferenceCompiles() throws {
        let makeModel = SystemOneDecisionModel.init(
            baseURL:
            path:
            modelID:
            apiKey:
            headers:
            capabilities:
            extraBody:
            session:
            retryPolicy:
        )
        let model = try makeModel(
            baseURL,
            "v1/systemone",
            "m",
            nil,
            [:],
            DecisionCapabilities(),
            [:],
            URLSession(configuration: .default),
            .default
        )
        #expect(model.imageEncoding == nil)
    }

    @Test func callsWithoutImageEncodingSendNoAttachments() throws {
        let model = try SystemOneDecisionModel(baseURL: baseURL, modelID: "m")
        #expect(model.imageEncoding == nil)
        #expect(model.capabilities == DecisionCapabilities())
    }

    @Test func callsWithImageEncodingUseTheNewInitializer() throws {
        let model = try SystemOneDecisionModel(baseURL: baseURL, modelID: "m", imageEncoding: .dataURL)
        #expect(model.imageEncoding == .dataURL)
        #expect(model.capabilities.supportedAttachmentKinds == [.image])

        let makeModel = SystemOneDecisionModel.init(
            baseURL:
            path:
            modelID:
            apiKey:
            headers:
            capabilities:
            imageEncoding:
            extraBody:
            session:
            retryPolicy:
        )
        let base64 = try makeModel(
            baseURL,
            "v1/systemone",
            "m",
            nil,
            [:],
            DecisionCapabilities(),
            .base64,
            [:],
            URLSession(configuration: .default),
            .default
        )
        #expect(base64.imageEncoding == .base64)
    }
}
