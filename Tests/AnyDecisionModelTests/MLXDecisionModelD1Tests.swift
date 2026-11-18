import Foundation
import Testing

@testable import AnyDecisionModel

#if MLX
    import MLXLMCommon

    // d1 integration tests download and run Liquid AI's d1-3B 8-bit (3.72 GB).
    // Set `ENABLE_MLX_D1_TESTS=1` to run them,
    // and `MLX_D1_MODEL_DIRECTORY` to load the files from a local directory.
    // The reference answers come from `scripts/d1-fixtures.py`,
    // which runs mlx-vlm at a pinned revision of the model.

    private let d1ModelID = "LiquidAI/d1-3B-MLX-8bit"

    private func makeD1Model(
        prefixCaching: Bool = false,
        rotationDebiasing: Bool = false,
        promptFormat: MLXDecisionModel.PromptFormat = .automatic
    ) -> MLXDecisionModel {
        MLXDecisionModel(
            modelID: d1ModelID,
            directory: ProcessInfo.processInfo.environment["MLX_D1_MODEL_DIRECTORY"].map {
                URL(fileURLWithPath: $0)
            },
            rotationDebiasing: rotationDebiasing,
            prefixCaching: prefixCaching,
            promptFormat: promptFormat
        )
    }

    /// Returns the probabilities in an answer as a flat array, in answer order.
    private func answerProbabilities(_ answer: Answer, question: Question) -> [Double] {
        switch (answer, question) {
        case (.binary(let probability), _):
            return [probability, 1 - probability]
        case (.choice(_, let probabilities, _), .choice(_, let options)):
            return options.map { probabilities[$0.name] ?? .nan }
        case (.score(_, let probabilities, _, _), _):
            return probabilities
        default:
            return []
        }
    }

    private func maximumDifference(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count else { return .infinity }
        return zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    private func argmax(_ values: [Double]) -> Int {
        values.indices.max { values[$0] < values[$1] } ?? -1
    }

    @Suite("MLXDecisionModel d1 configuration")
    struct MLXDecisionModelD1ConfigurationTests {
        @Test func formatDefaultsToAutomatic() {
            #expect(MLXDecisionModel(modelID: d1ModelID).promptFormat == .automatic)
            #expect(MLXDecisionModel(modelID: d1ModelID, promptFormat: .d1LFM2).promptFormat == .d1LFM2)
        }

        @Test func automaticFormatFollowsTheConfiguration() {
            let automatic = MLXDecisionModel(modelID: d1ModelID)
            #expect(automatic.resolvedFormat(declaresD1LFM2: true) == .d1LFM2)
            #expect(automatic.resolvedFormat(declaresD1LFM2: false) == .standard)
            let standard = MLXDecisionModel(modelID: d1ModelID, promptFormat: .standard)
            #expect(standard.resolvedFormat(declaresD1LFM2: true) == .standard)
            let d1 = MLXDecisionModel(modelID: d1ModelID, promptFormat: .d1LFM2)
            #expect(d1.resolvedFormat(declaresD1LFM2: false) == .d1LFM2)
        }

        @Test func lfm2VLConfigurationsGainTheKeysThatMLXVLMReads() throws {
            let configuration: [String: Any] = [
                "model_type": "lfm2_vl",
                "text_config": [
                    "intermediate_size": 10752,
                    "rope_parameters": ["rope_theta": 1_000_000.0, "rope_type": "default"],
                ],
            ]
            let data = try JSONSerialization.data(withJSONObject: configuration)
            let amended = try #require(amendedLFM2VLConfiguration(data))
            let object = try #require(JSONSerialization.jsonObject(with: amended) as? [String: Any])
            let text = try #require(object["text_config"] as? [String: Any])
            #expect(text["block_ff_dim"] as? Int == 10752)
            #expect(text["rope_theta"] as? Double == 1_000_000)
            #expect(text["intermediate_size"] as? Int == 10752)
            #expect(object["model_type"] as? String == "lfm2_vl")

            // A configuration that already has the keys stays as it is.
            #expect(amendedLFM2VLConfiguration(amended) == nil)
            let legacy = ["text_config": ["block_ff_dim": 12288, "intermediate_size": 1]]
            #expect(amendedLFM2VLConfiguration(try JSONSerialization.data(withJSONObject: legacy)) == nil)
        }

        @Test func lfm2VLLoadsThroughMLXVLMWithoutTheD1Entry() throws {
            func configuration(_ object: [String: Any]) throws -> Data {
                try JSONSerialization.data(withJSONObject: object)
            }
            let withoutAutoMap = try configuration(["model_type": "lfm2_vl"])
            #expect(ModelArchitecture(configuration: withoutAutoMap) == .lfm2VL)
            #expect(!D1LFM2Prompt.isD1LFM2Configuration(withoutAutoMap))

            let d1 = try configuration(["model_type": "lfm2_vl", "auto_map": ["AutoModel": "modeling_d1.D1Model"]])
            #expect(ModelArchitecture(configuration: d1) == .lfm2VL)
            #expect(ModelArchitecture(configuration: try configuration(["model_type": "qwen3"])) == .languageModel)
            #expect(ModelArchitecture(configuration: try configuration(["model_type": "lfm2"])) == .languageModel)
            #expect(ModelArchitecture(configuration: Data("not json".utf8)) == .languageModel)

            // Without the entry, `.automatic` uses the standard prompt and `.d1LFM2` uses d1's.
            #expect(MLXDecisionModel(modelID: d1ModelID).resolvedFormat(declaresD1LFM2: false) == .standard)
            #expect(
                MLXDecisionModel(modelID: d1ModelID, promptFormat: .d1LFM2).resolvedFormat(declaresD1LFM2: false)
                    == .d1LFM2
            )
        }

        @Test func capabilitiesDoNotDependOnTheFormat() {
            for format in [MLXDecisionModel.PromptFormat.automatic, .standard, .d1LFM2] {
                let capabilities = MLXDecisionModel(modelID: d1ModelID, promptFormat: format).capabilities
                #expect(capabilities.maximumChoiceOptions == 26)
                #expect(capabilities.maximumScoreLevels == 10)
            }
        }

        @Test func attachmentsAreRejectedBeforeTheModelLoads() async throws {
            let model = MLXDecisionModel(
                modelID: "example/not-a-model",
                directory: URL(fileURLWithPath: "/nonexistent"),
                promptFormat: .d1LFM2
            )
            let state = DecisionState.segments([
                .text("A photo:"),
                .attachment(DecisionState.Attachment(imageData: Data([0xFF]), mimeType: "image/jpeg")),
            ])
            let session = DecisionSession(model: model, state: state)
            await #expect {
                _ = try await session.probability(of: "Is it a cat?")
            } throws: { error in
                guard case .unsupportedState = error as? DecisionError else { return false }
                return true
            }
            #expect(!model.isLoaded)
        }
    }

    @Suite("MLXDecisionModel d1", .enabled(if: shouldRunMLXD1Tests), .serialized)
    struct MLXDecisionModelD1Tests {
        @Test func detectsD1FromTheConfiguration() async throws {
            let loaded = try await makeD1Model().loadModel()
            #expect(loaded.declaresD1LFM2)
            #expect(
                MLXDecisionModel(modelID: d1ModelID).resolvedFormat(declaresD1LFM2: loaded.declaresD1LFM2) == .d1LFM2
            )
        }

        @Test func promptTokensMatchTheReference() async throws {
            let fixture = try D1Fixture.load()
            let container = try await makeD1Model().loadModel().container
            let tokenizer = await container.tokenizer
            #expect(tokenizer.bosToken == "<|startoftext|>")
            let encode = { (text: String) in tokenizer.encode(text: text, addSpecialTokens: false) }
            for testCase in fixture.cases {
                let rendered = try D1LFM2Prompt.render(testCase.question.question, index: 0, encode: encode)
                let prompt = try D1LFM2Prompt.prompt(
                    state: testCase.state.state,
                    question: rendered,
                    bos: tokenizer.bosToken ?? ""
                )
                #expect(prompt == testCase.prompt, "\(testCase.name)")
                #expect(encode(prompt) == testCase.inputIDs, "\(testCase.name)")
                #expect(rendered.tokenGroups == testCase.readoutIDs, "\(testCase.name)")
            }
        }

        /// Swift and mlx-vlm run the same weights in float32 on the same prompt tokens,
        /// so they differ only in how the GPU orders and rounds its sums.
        /// The largest measured difference was 2.2e-6 on an M-series Mac.
        /// The tolerance allows for GPUs that multiply float32 matrices with less precision,
        /// and stays well below the 0.013 that one wrong prompt token caused.
        static let float32Tolerance = 1e-3

        /// mlx-vlm's model as it loads computes in float16,
        /// which moved probabilities by up to 0.0008 from its float32 run on these cases.
        /// The tolerance allows about three times that.
        static let float16Tolerance = 2.5e-3

        @Test func answersMatchTheReference() async throws {
            let fixture = try D1Fixture.load()
            let model = makeD1Model()
            var largest = (float32: 0.0, float16: 0.0)
            for testCase in fixture.cases {
                let question = testCase.question.question
                let session = DecisionSession(model: model, state: testCase.state.state)
                let response = try await session.decide([question])
                let probabilities = answerProbabilities(response.answers[0], question: question)

                let float32Difference = maximumDifference(probabilities, testCase.float32Probabilities)
                let float16Difference = maximumDifference(probabilities, testCase.probabilities)
                largest = (max(largest.float32, float32Difference), max(largest.float16, float16Difference))
                #expect(float32Difference <= Self.float32Tolerance, "\(testCase.name): \(probabilities)")
                #expect(float16Difference <= Self.float16Tolerance, "\(testCase.name): \(probabilities)")
                #expect(argmax(probabilities) == argmax(testCase.probabilities), "\(testCase.name)")
                #expect(response.usage.inputTokenCount == testCase.inputIDs.count, "\(testCase.name)")
            }
            print("d1 largest differences from the float32 and float16 references: \(largest)")
        }

        @Test func prefixCachingAndQuestionBatchesKeepTheAnswers() async throws {
            let fixture = try D1Fixture.load()
            let groups = Dictionary(grouping: fixture.cases) { $0.state.state }
            var largest = 0.0
            for (state, cases) in groups {
                let questions = cases.map(\.question.question)
                for prefixCaching in [false, true] {
                    let session = DecisionSession(model: makeD1Model(prefixCaching: prefixCaching), state: state)
                    let response = try await session.decide(questions)
                    for (index, testCase) in cases.enumerated() {
                        let probabilities = answerProbabilities(response.answers[index], question: questions[index])
                        largest = max(largest, maximumDifference(probabilities, testCase.float32Probabilities))
                        #expect(
                            maximumDifference(probabilities, testCase.float32Probabilities)
                                <= Self.float32Tolerance,
                            "\(testCase.name), prefix caching \(prefixCaching): \(probabilities)"
                        )
                    }
                    if prefixCaching, cases.count > 1 {
                        #expect(response.diagnostics.contains { $0.cachedTokenCount > 0 })
                    }
                }
            }
            print("d1 largest difference with batches and prefix caching: \(largest)")
        }

        @Test func rotationDebiasingAveragesRotations() async throws {
            let fixture = try D1Fixture.load()
            let testCase = try #require(fixture.cases.first { $0.name == "ticket team" })
            let question = testCase.question.question
            guard case .choice(let instructions, let options) = question else {
                Issue.record("The case is not a choice.")
                return
            }
            let state = testCase.state.state
            let session = DecisionSession(model: makeD1Model(rotationDebiasing: true), state: state)
            let response = try await session.decide([question])
            let probabilities = answerProbabilities(response.answers[0], question: question)

            // Ask each rotation of the options separately, and average by option name.
            let plain = DecisionSession(model: makeD1Model(), state: state)
            var expected = [Double](repeating: 0, count: options.count)
            for start in options.indices {
                let rotated = Array(options[start...] + options[..<start])
                let answer = try await plain.decide([.choice(instructions: instructions, options: rotated)]).answers[0]
                guard case .choice(_, let byName, _) = answer else { continue }
                for (index, option) in options.enumerated() {
                    expected[index] += (byName[option.name] ?? .nan) / Double(options.count)
                }
            }
            #expect(maximumDifference(probabilities, expected) <= 1e-5, "\(probabilities) vs \(expected)")
            // The rotations differ, so their average differs from the first rotation alone.
            #expect(maximumDifference(probabilities, testCase.float32Probabilities) > 1e-3)
            #expect(abs(probabilities.reduce(0, +) - 1) < 1e-9)
        }

        @Test func explicitD1FormatRunsAnExportWithoutTheD1Entry() async throws {
            let fixture = try D1Fixture.load()
            _ = try await makeD1Model().loadModel()
            let environment = ProcessInfo.processInfo.environment
            let cache =
                environment["HF_HUB_CACHE"].map { URL(fileURLWithPath: $0) }
                ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
            let snapshot =
                environment["MLX_D1_MODEL_DIRECTORY"].map { URL(fileURLWithPath: $0) }
                ?? cache.appendingPathComponent("models--LiquidAI--d1-3B-MLX-8bit/snapshots/\(fixture.revision)")
            let configurationURL = snapshot.appendingPathComponent("config.json")
            try #require(FileManager.default.fileExists(atPath: configurationURL.path))

            // Link the export with a config.json that has no `auto_map` entry.
            let directory = try makeLinkedDirectory(to: snapshot, excluding: ["config.json"])
            defer { try? FileManager.default.removeItem(at: directory) }
            var configuration = try #require(
                JSONSerialization.jsonObject(with: Data(contentsOf: configurationURL)) as? [String: Any]
            )
            configuration["auto_map"] = nil
            try JSONSerialization.data(withJSONObject: configuration)
                .write(to: directory.appendingPathComponent("config.json"))

            let model = MLXDecisionModel(modelID: d1ModelID, directory: directory, promptFormat: .d1LFM2)
            defer { model.removeFromCache() }
            let loaded = try await model.loadModel()
            #expect(!loaded.declaresD1LFM2)

            let testCase = try #require(fixture.cases.first { $0.name == "ticket team" })
            let question = testCase.question.question
            let response = try await DecisionSession(model: model, state: testCase.state.state).decide([question])
            let probabilities = answerProbabilities(response.answers[0], question: question)
            #expect(maximumDifference(probabilities, testCase.float32Probabilities) <= Self.float32Tolerance)
        }
    }
#endif
