import Foundation
import Testing

// This file imports the module without `@testable`,
// so it compiles only against the public API, as a client does.
import AnyDecisionModel

#if MLX
    import HuggingFace

    @Suite("MLXDecisionModel source compatibility")
    struct MLXDecisionModelCompatibilityTests {
        typealias Initializer = (
            String, HubClient?, URL?, Calibration?, Bool, Bool, Bool, String, Int
        ) -> MLXDecisionModel

        @Test func typedReferenceToTheVersionNineInitializerCompiles() {
            let makeModel: Initializer = MLXDecisionModel.init
            let model = makeModel("model", nil, nil, nil, true, true, true, "System", 3)
            #expect(model.modelID == "model")
            #expect(model.rotationDebiasing)
            #expect(model.prefixCaching)
            #expect(model.closedThinkFallback)
            #expect(model.systemPrompt == "System")
            #expect(model.maximumBatchSize == 3)
            #expect(model.promptFormat == .automatic)
        }

        @Test func untypedReferencesToTheVersionNineInitializerCompile() {
            let labeled = MLXDecisionModel.init(
                modelID:
                hub:
                directory:
                calibration:
                rotationDebiasing:
                prefixCaching:
                closedThinkFallback:
                systemPrompt:
                maximumBatchSize:
            )
            let model = labeled("model", nil, nil, nil, false, false, false, "System", 4)
            #expect(model.maximumBatchSize == 4)
            #expect(model.promptFormat == .automatic)

            let unlabeled = MLXDecisionModel.init
            let other = unlabeled("other", nil, nil, nil, false, true, false, "System", 2)
            #expect(other.prefixCaching)
            #expect(other.promptFormat == .automatic)
        }

        @Test func callsWithoutAFormatUseTheAutomaticFormat() {
            #expect(MLXDecisionModel(modelID: "model").promptFormat == .automatic)
            #expect(MLXDecisionModel(modelID: "model", prefixCaching: true).promptFormat == .automatic)
            #expect(MLXDecisionModel(modelID: "model", promptFormat: .standard).promptFormat == .standard)
        }
    }
#endif
