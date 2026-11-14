import Foundation
import Testing

// This file imports the module without `@testable`,
// so it compiles only against the public API, as a client does.
import AnyDecisionModel

@Suite("DecisionCapabilities source compatibility")
struct DecisionCapabilitiesCompatibilityTests {
    @Test func versionEightInitializerReferenceCompiles() {
        let makeCapabilities = DecisionCapabilities.init(
            maximumChoiceOptions:
            maximumScoreLevels:
            supportedAttachmentKinds:
        )
        let capabilities = makeCapabilities(3, 4, [.image])
        #expect(capabilities.maximumChoiceOptions == 3)
        #expect(capabilities.maximumScoreLevels == 4)
        #expect(capabilities.supportedAttachmentKinds == [.image])
        #expect(capabilities.questionKinds == [.binary, .choice, .score])
        #expect(capabilities.maximumQuestions == nil)
        #expect(capabilities.maximumAttachments == nil)
        #expect(capabilities.probabilitySource == .unspecified)
    }

    @Test func ordinaryCallsCompile() {
        let empty = DecisionCapabilities()
        let options = DecisionCapabilities(maximumChoiceOptions: 3)
        let all = DecisionCapabilities(
            maximumChoiceOptions: 3,
            maximumScoreLevels: 4,
            supportedAttachmentKinds: [.image]
        )
        let new = DecisionCapabilities(maximumQuestions: 2, probabilitySource: .native)
        #expect(
            empty
                == DecisionCapabilities(
                    maximumChoiceOptions: nil,
                    maximumScoreLevels: nil,
                    supportedAttachmentKinds: []
                )
        )
        #expect(options.maximumChoiceOptions == 3)
        #expect(all.supportedAttachmentKinds == [.image])
        #expect(new.maximumQuestions == 2)
        #expect(new.probabilitySource == .native)
    }
}
