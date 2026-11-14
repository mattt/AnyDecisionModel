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

    @Test func versionEightUnqualifiedReferenceCompiles() {
        let makeCapabilities = DecisionCapabilities.init
        let capabilities = makeCapabilities(3, 4, [])
        #expect(capabilities == expected(maximumChoiceOptions: 3, maximumScoreLevels: 4))
    }

    /// Returns capabilities with every parameter given,
    /// using the 0.8 defaults for the parameters that 0.8 doesn't have.
    func expected(
        maximumChoiceOptions: Int? = nil,
        maximumScoreLevels: Int? = nil,
        supportedAttachmentKinds: Set<DecisionState.Attachment.Kind> = []
    ) -> DecisionCapabilities {
        DecisionCapabilities(
            questionKinds: [.binary, .choice, .score],
            maximumQuestions: nil,
            maximumChoiceOptions: maximumChoiceOptions,
            maximumScoreLevels: maximumScoreLevels,
            supportedAttachmentKinds: supportedAttachmentKinds,
            maximumAttachments: nil,
            probabilitySource: .unspecified
        )
    }

    @Test func versionEightCallsKeepTheirValues() {
        #expect(DecisionCapabilities() == expected())
        #expect(DecisionCapabilities(maximumChoiceOptions: 3) == expected(maximumChoiceOptions: 3))
        #expect(DecisionCapabilities(maximumScoreLevels: 4) == expected(maximumScoreLevels: 4))
        #expect(
            DecisionCapabilities(supportedAttachmentKinds: [.image])
                == expected(supportedAttachmentKinds: [.image])
        )
        #expect(
            DecisionCapabilities(maximumChoiceOptions: 3, maximumScoreLevels: 4)
                == expected(maximumChoiceOptions: 3, maximumScoreLevels: 4)
        )
        #expect(
            DecisionCapabilities(maximumChoiceOptions: 3, supportedAttachmentKinds: [.image])
                == expected(maximumChoiceOptions: 3, supportedAttachmentKinds: [.image])
        )
        #expect(
            DecisionCapabilities(maximumScoreLevels: 4, supportedAttachmentKinds: [.image])
                == expected(maximumScoreLevels: 4, supportedAttachmentKinds: [.image])
        )
        #expect(
            DecisionCapabilities(maximumChoiceOptions: 3, maximumScoreLevels: 4, supportedAttachmentKinds: [.image])
                == expected(maximumChoiceOptions: 3, maximumScoreLevels: 4, supportedAttachmentKinds: [.image])
        )
    }

    @Test func callsWithNewParametersUseTheNewInitializer() {
        #expect(DecisionCapabilities(questionKinds: [.binary]).questionKinds == [.binary])
        #expect(DecisionCapabilities(maximumQuestions: 2).maximumQuestions == 2)
        #expect(DecisionCapabilities(maximumAttachments: 1).maximumAttachments == 1)
        #expect(DecisionCapabilities(probabilitySource: .native).probabilitySource == .native)

        let mixed = DecisionCapabilities(
            maximumQuestions: 2,
            maximumChoiceOptions: 3,
            maximumScoreLevels: 4,
            supportedAttachmentKinds: [.image],
            probabilitySource: .native
        )
        #expect(mixed.maximumQuestions == 2)
        #expect(mixed.maximumChoiceOptions == 3)
        #expect(mixed.maximumScoreLevels == 4)
        #expect(mixed.supportedAttachmentKinds == [.image])
        #expect(mixed.probabilitySource == .native)
    }
}
