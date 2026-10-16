import Foundation
import Testing

@testable import AnyDecisionModel

#if Llama
    @Suite("LlamaDecisionModel basics")
    struct LlamaDecisionModelBasicsTests {
        @Test func initialization() {
            let model = LlamaDecisionModel(modelPath: "/path/to/nimble-q4.gguf")
            #expect(model.modelPath == "/path/to/nimble-q4.gguf")
            #expect(model.modelID == "nimble-q4")
            #expect(model.gpuLayers == LlamaDecisionModel.defaultGPULayerCount)
            #expect(model.capabilities.maximumChoiceOptions == 26)
            #expect(model.capabilities.maximumScoreLevels == 10)
            #expect(!model.isLoaded)

            let cpuOnly = LlamaDecisionModel(modelPath: "/path/to/model.gguf", gpuLayers: 0)
            #expect(cpuOnly.gpuLayers == 0)
        }

        @Test func missingModelFile() async throws {
            let model = LlamaDecisionModel(modelPath: "/nonexistent/model.gguf")
            let session = DecisionSession(model: model, state: "ticket")

            await #expect(throws: DecisionError.self) {
                _ = try await session.probability(of: "Is this urgent?")
            }
        }

        @Test func tokenLogProbabilities() {
            // Label "yes" has tokens 0 and 1; label "no" has token 2.
            let logits: [Float] = [2, 0, 1, -1]
            let result = LlamaDecisionModel.tokenLogProbabilities(logits: logits, tokenMap: [[0, 1], [2]])

            let denominator = log(exp(2.0) + exp(0.0) + exp(1.0) + exp(-1.0))
            #expect(abs(result[0][0] - (2 - denominator)) < 1e-9)
            #expect(abs(result[0][1] - (0 - denominator)) < 1e-9)
            #expect(abs(result[1][0] - (1 - denominator)) < 1e-9)
        }
    }

    @Suite(
        "LlamaDecisionModel inference",
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] != nil)
    )
    struct LlamaDecisionModelInferenceTests {
        let model = LlamaDecisionModel(
            modelPath: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"]!
        )

        @Test func binaryChoiceAndScore() async throws {
            let session = DecisionSession(
                model: model,
                state: .text("Our checkout has returned 500 errors since 9am. Customers cannot pay.")
            )

            let probability = try await session.probability(
                of: "Is this ticket about a software error?"
            )
            #expect((0 ... 1).contains(probability))

            let response = try await session.decide([
                .choice(
                    instructions: "Which label fits this ticket?",
                    options: [
                        ChoiceOption("billing", description: "Payments and refunds"),
                        ChoiceOption("bug", description: "Software errors"),
                    ]
                ),
                .score(
                    instructions: "How urgent is this ticket?",
                    levels: ["Routine", "Soon", "Immediate"]
                ),
            ])
            #expect(response.answers.count == 2)
            #expect(response.usage.inputTokenCount > 0)

            guard case .choice(let name, let probabilities, _) = response.answers[0] else {
                Issue.record("Expected a choice answer.")
                return
            }
            #expect(probabilities.keys.contains(name))

            guard case .score(let value, let scoreProbabilities, _, _) = response.answers[1] else {
                Issue.record("Expected a score answer.")
                return
            }
            #expect(scoreProbabilities.count == 3)
            #expect((0 ... 2).contains(value))
        }

        @Test func promptLongerThanBatchSize() async throws {
            var model = self.model
            model.batchSize = 32
            let filler = Array(
                repeating: "The quick brown fox jumps over the lazy dog.",
                count: 30
            ).joined(separator: " ")
            let session = DecisionSession(model: model, state: .text(filler))

            let probability = try await session.probability(of: "Is this text about an animal?")
            #expect((0 ... 1).contains(probability))
        }
    }
#endif
