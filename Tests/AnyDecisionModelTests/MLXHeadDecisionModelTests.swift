import Foundation
import Testing

@testable import AnyDecisionModel

#if MLX
    import MLX
    import MLXLMCommon

    // Integration tests download and run Clef-flash 4-bit (6.2 GB).
    // Set `ENABLE_MLX_HEAD_TESTS=1` to run them,
    // and `MLX_HEAD_MODEL_DIRECTORY` to load the files from a local directory.
    // The reference answers come from `scripts/joint-schema-fixtures.py model`
    // at a pinned revision of the model.

    private let modelID = "mlx-community/clef-flash-4bit"

    private func makeModel(prefixCaching: Bool = true, calibration: Calibration? = nil) -> MLXHeadDecisionModel {
        MLXHeadDecisionModel(
            modelID: modelID,
            directory: ProcessInfo.processInfo.environment["MLX_HEAD_MODEL_DIRECTORY"].map {
                URL(fileURLWithPath: $0)
            },
            head: .jointSchema,
            calibration: calibration,
            prefixCaching: prefixCaching
        )
    }

    /// Answers from the reference implementation for single questions.
    private struct ModelFixture: Decodable {
        struct Case: Decodable {
            var name: String
            var state: StateSpec
            var question: QuestionSpec
            var inputIDs: [Int]
            var probabilities: [Double]

            enum CodingKeys: String, CodingKey {
                case name, state, question, probabilities
                case inputIDs = "input_ids"
            }
        }

        var cases: [Case]

        static func load() throws -> ModelFixture {
            try JSONDecoder().decode(ModelFixture.self, from: fixtureData("clef-flash-4bit.json"))
        }
    }

    private let ticket =
        "I was charged twice this month and nobody has answered my emails in a week. Fix this NOW."

    private let departments = [
        ChoiceOption("technical", description: "bugs, outages, errors, login problems"),
        ChoiceOption("billing", description: "payments, charges, refunds, invoices"),
        ChoiceOption("sales", description: "pricing questions, upgrades, new purchases"),
    ]

    private let questions: [Question] = [
        .binary(instructions: "Is the customer asking for a refund?"),
        .choice(instructions: "Route this support ticket.", options: departments),
        .score(instructions: "How frustrated is the customer?", levels: ["Calm", "Frustrated", "Very angry"]),
    ]

    /// Returns the probabilities in an answer as a flat array, in answer order.
    private func probabilities(_ answer: Answer, options: [ChoiceOption] = departments) -> [Double] {
        switch answer {
        case .binary(let probability): return [probability, 1 - probability]
        case .choice(_, let probabilities, _): return options.map { probabilities[$0.name] ?? .nan }
        case .score(_, let probabilities, _, _): return probabilities
        }
    }

    private func options(of question: Question) -> [ChoiceOption] {
        if case .choice(_, let options) = question { return options }
        return []
    }

    private func maximumDifference(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count else { return .infinity }
        return zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    /// Returns how far MLX's peak memory rises above its starting active memory during an operation.
    private func peakMemoryGrowth(_ operation: () async throws -> Void) async throws -> Int {
        Memory.clearCache()
        let start = Memory.activeMemory
        Memory.peakMemory = 0
        try await operation()
        return Memory.peakMemory - start
    }

    /// A downloader that records the file patterns it is asked for and returns one directory.
    private final class RecordingDownloader: Downloader {
        let directory: URL
        private let requests = Locked<[[String]]>([])

        init(directory: URL) {
            self.directory = directory
        }

        var requestedPatterns: [[String]] { requests.withLock { $0 } }

        func download(
            id: String,
            revision: String?,
            matching patterns: [String],
            useLatest: Bool,
            progressHandler: @Sendable @escaping (Progress) -> Void
        ) async throws -> URL {
            requests.withLock { $0.append(patterns) }
            return directory
        }
    }

    private func makeTemporaryDirectory(containing names: [String]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXHeadDecisionModelTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for name in names {
            FileManager.default.createFile(atPath: directory.appendingPathComponent(name).path, contents: Data())
        }
        return directory
    }

    @Suite("MLXHeadDecisionModel configuration")
    struct MLXHeadDecisionModelConfigurationTests {
        @Test func defaults() {
            let model = MLXHeadDecisionModel(modelID: "example/model", head: .jointSchema)
            #expect(model.modelID == "example/model")
            #expect(model.head == .jointSchema)
            #expect(model.hub == nil)
            #expect(model.directory == nil)
            #expect(model.calibration == nil)
            #expect(model.prefixCaching == true)
        }

        @Test func capabilitiesHaveNoOptionOrLevelLimits() {
            let capabilities = MLXHeadDecisionModel(modelID: "example/model", head: .jointSchema).capabilities
            #expect(capabilities.maximumChoiceOptions == nil)
            #expect(capabilities.maximumScoreLevels == nil)
        }

        @Test func downloadsStopBeforeTheWeightsWhenHeadFilesAreMissing() async throws {
            let directory = try makeTemporaryDirectory(containing: ["config.json"])
            defer { try? FileManager.default.removeItem(at: directory) }
            let downloader = RecordingDownloader(directory: directory)
            let model = MLXHeadDecisionModel(modelID: "example/language-model", head: .jointSchema)

            await #expect {
                _ = try await model.modelDirectory(using: downloader, requiring: JointSchemaHead.fileNames)
            } throws: { error in
                guard case DecisionError.modelUnavailable(let detail) = error else { return false }
                return detail.contains("joint_head.safetensors")
            }
            #expect(downloader.requestedPatterns == [JointSchemaHead.fileNames])
        }

        @Test func downloadsTheWeightsAfterFindingHeadFiles() async throws {
            let directory = try makeTemporaryDirectory(containing: JointSchemaHead.fileNames)
            defer { try? FileManager.default.removeItem(at: directory) }
            let downloader = RecordingDownloader(directory: directory)
            let model = MLXHeadDecisionModel(modelID: "example/decision-model", head: .jointSchema)

            let url = try await model.modelDirectory(using: downloader, requiring: JointSchemaHead.fileNames)
            #expect(url == directory)
            #expect(
                downloader.requestedPatterns == [JointSchemaHead.fileNames, ["*.safetensors", "*.json", "*.jinja"]]
            )
        }

        @Test func cacheKeysSeparateDirectoriesAndModels() {
            let hub = MLXHeadDecisionModel(modelID: "example/model", head: .jointSchema)
            let local = MLXHeadDecisionModel(
                modelID: "example/model",
                directory: URL(fileURLWithPath: "/tmp/model"),
                head: .jointSchema
            )
            #expect(hub.cacheKey != local.cacheKey)
            #expect(!hub.isLoaded)
        }
    }

    @Suite("MLXHeadDecisionModel", .enabled(if: shouldRunMLXHeadTests), .serialized)
    struct MLXHeadDecisionModelTests {
        @Test func promptTokensMatchTheReference() async throws {
            let fixture = try ModelFixture.load()
            let loaded = try await makeModel().load(JointSchemaHead.self)
            let tokenizer = await loaded.container.tokenizer
            let encode = { (text: String) in tokenizer.encode(text: text, addSpecialTokens: false) }
            for item in fixture.cases {
                let prefix = JointSchemaHead.prefixTokens(for: item.state.state, encode: encode)
                let (tokens, _) = JointSchemaHead.prompt(for: item.question.question, encode: encode)
                #expect(prefix + tokens == item.inputIDs, Comment(rawValue: item.name))
            }
        }

        @Test(arguments: [true, false])
        func probabilitiesMatchTheReference(prefixCaching: Bool) async throws {
            let fixture = try ModelFixture.load()
            let model = makeModel(prefixCaching: prefixCaching)
            for item in fixture.cases {
                let question = item.question.question
                let session = DecisionSession(model: model, state: item.state.state)
                let answer = try await session.decide(question)
                let difference = maximumDifference(
                    probabilities(answer, options: options(of: question)),
                    item.probabilities
                )
                #expect(difference < 0.02, "\(item.name): \(difference)")
            }
        }

        @Test func cachedAndUncachedResultsMatch() async throws {
            let cached = try await DecisionSession(model: makeModel(prefixCaching: true), state: .text(ticket))
                .decide(questions)
            let uncached = try await DecisionSession(model: makeModel(prefixCaching: false), state: .text(ticket))
                .decide(questions)
            for index in questions.indices {
                let difference = maximumDifference(
                    probabilities(cached.answers[index]),
                    probabilities(uncached.answers[index])
                )
                #expect(difference < 0.01, "question \(index): \(difference)")
                #expect(cached.diagnostics[index].cachedTokenCount > 0)
                #expect(uncached.diagnostics[index].cachedTokenCount == 0)
                #expect(
                    cached.diagnostics[index].cachedTokenCount + cached.diagnostics[index].evaluatedTokenCount
                        == uncached.diagnostics[index].evaluatedTokenCount
                )
            }
            #expect(cached.usage.inputTokenCount == uncached.usage.inputTokenCount)
        }

        @Test(arguments: [true, false])
        func questionsAreIndependentOfBatchAndOrder(prefixCaching: Bool) async throws {
            let model = makeModel(prefixCaching: prefixCaching)
            var alone: [[Double]] = []
            for question in questions {
                let answer = try await DecisionSession(model: model, state: .text(ticket)).decide(question)
                alone.append(probabilities(answer))
            }

            let together = try await DecisionSession(model: model, state: .text(ticket)).decide(questions)
            let reversed = try await DecisionSession(model: model, state: .text(ticket))
                .decide(questions.reversed())
            for index in questions.indices {
                #expect(maximumDifference(probabilities(together.answers[index]), alone[index]) < 1e-6)
                let answer = reversed.answers[questions.count - 1 - index]
                #expect(maximumDifference(probabilities(answer), alone[index]) < 1e-6)
            }
        }

        @Test func reorderedOptionsKeepProbabilitiesByName() async throws {
            let session = DecisionSession(model: makeModel(), state: .text(ticket))
            let forward = try await session.choice("Route this support ticket.", from: departments)
            let reversed = try await session.choice("Route this support ticket.", from: departments.reversed())
            #expect(forward.value == "billing")
            #expect(reversed.value == "billing")
            for option in departments.map(\.name) {
                #expect(abs(forward.probability(of: option) - reversed.probability(of: option)) < 1e-6)
            }
        }

        @Test func diagnosticsAndUsage() async throws {
            let response = try await DecisionSession(model: makeModel(), state: .text(ticket)).decide(questions)
            #expect(response.modelID == modelID)
            #expect(response.usage.outputTokenCount == 0)
            #expect(response.diagnostics.count == questions.count)
            var total = 0
            for diagnostics in response.diagnostics {
                #expect(diagnostics.allowedAnswerMass == 1)
                #expect(diagnostics.evaluatedTokenCount > 0)
                #expect(diagnostics.duration > .zero)
                total += diagnostics.cachedTokenCount + diagnostics.evaluatedTokenCount
            }
            #expect(response.usage.inputTokenCount == total)
        }

        @Test func choicesAcceptMoreThan26Options() async throws {
            let options = (1 ... 30).map { ChoiceOption("option \($0)") }
            let session = DecisionSession(model: makeModel(), state: "The answer is option 17.")
            let answer = try await session.choice("Which option is the answer?", from: options)
            #expect(answer.distribution.count == 30)
            #expect(answer.value == "option 17")
        }

        @Test func calibrationSoftensProbabilities() async throws {
            let state: DecisionState = "A 6 sided die rolled a 4."
            let raw = try await DecisionSession(model: makeModel(), state: state)
                .probability(of: "Is the number rolled odd?")
            let soft = try await DecisionSession(
                model: makeModel(calibration: Calibration(temperature: 4)),
                state: state
            ).probability(of: "Is the number rolled odd?")
            #expect(soft > raw)
            #expect(soft < 0.5)
        }

        @Test func promptsLongerThanTheLimitFailBeforeInference() async throws {
            let state = DecisionState.text(String(repeating: "word ", count: 17_000))
            let session = DecisionSession(model: makeModel(), state: state)
            await #expect {
                _ = try await session.decide(questions)
            } throws: { error in
                guard case DecisionError.unsupportedQuestion(let index, _) = error else { return false }
                return index == 0
            }
        }

        @Test func prewarmSkipsStatesLongerThanTheLimit() async throws {
            let model = makeModel()
            _ = try await DecisionSession(model: model, state: "state").decide(questions[0])

            let session = DecisionSession(model: model, state: .text(String(repeating: "word ", count: 17_000)))
            let growth = try await peakMemoryGrowth { try await session.prewarm() }
            #expect(growth < 1 << 30, "\(growth) bytes")
            await #expect(throws: DecisionError.self) { _ = try await session.decide(questions[0]) }
        }

        @Test(arguments: [true, false])
        func longPromptsUseBoundedMemory(prefixCaching: Bool) async throws {
            let model = makeModel(prefixCaching: prefixCaching)
            _ = try await DecisionSession(model: model, state: "state").decide(questions[0])

            // About 15,000 tokens, below the limit of 16,384.
            let session = DecisionSession(model: model, state: .text(String(repeating: "word ", count: 15_000)))
            let growth = try await peakMemoryGrowth { _ = try await session.decide(questions[0]) }
            #expect(growth < 3 << 30, "\(growth) bytes")
        }

        @Test func modelsWithoutHeadFilesAreUnavailable() async throws {
            let source = try await makeModel().modelDirectory(requiring: JointSchemaHead.fileNames)
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("MLXHeadDecisionModelTests-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            for name in try FileManager.default.contentsOfDirectory(atPath: source.path)
            where name != "joint_head.safetensors" {
                try FileManager.default.createSymbolicLink(
                    at: directory.appendingPathComponent(name),
                    withDestinationURL: source.appendingPathComponent(name)
                )
            }

            let model = MLXHeadDecisionModel(modelID: modelID, directory: directory, head: .jointSchema)
            let session = DecisionSession(model: model, state: "state")
            await #expect {
                _ = try await session.decide(questions[0])
            } throws: { error in
                guard case DecisionError.modelUnavailable(let detail) = error else { return false }
                return detail.contains("joint_head.safetensors")
            }
        }

        @Test func hubModelsWithoutHeadFilesAreUnavailable() async throws {
            let model = MLXHeadDecisionModel(modelID: "mlx-community/Qwen3-0.6B-4bit", head: .jointSchema)
            let session = DecisionSession(model: model, state: "state")
            await #expect {
                _ = try await session.decide(questions[0])
            } throws: { error in
                guard case DecisionError.modelUnavailable(let detail) = error else { return false }
                return detail.contains("joint_head.safetensors")
            }
        }

        @Test func concurrentSessionsAreIsolated() async throws {
            let model = makeModel()
            let states: [DecisionState] = [
                "A 6 sided die rolled a 3.",
                "A 6 sided die rolled a 4.",
                "A 6 sided die rolled a 5.",
                "A 6 sided die rolled a 6.",
            ]
            let question = Question.binary(instructions: "Is the number rolled odd?")

            var sequential: [Double] = []
            for state in states {
                sequential.append(
                    probabilities(try await DecisionSession(model: model, state: state).decide(question))[0]
                )
            }

            let concurrent = try await withThrowingTaskGroup(of: (Int, Double).self) { group in
                for (index, state) in states.enumerated() {
                    group.addTask {
                        let session = DecisionSession(model: model, state: state)
                        var last = 0.0
                        for _ in 0 ..< 3 {
                            last = probabilities(try await session.decide(question))[0]
                        }
                        return (index, last)
                    }
                }
                var results = [Double](repeating: .nan, count: states.count)
                for try await (index, value) in group {
                    results[index] = value
                }
                return results
            }
            #expect(maximumDifference(concurrent, sequential) < 1e-6)
            // The reference gives 0.96, 0.01, 0.86, and 0.01.
            #expect(sequential[0] > 0.5 && sequential[2] > 0.5, "\(sequential)")
            #expect(sequential[1] < 0.5 && sequential[3] < 0.5, "\(sequential)")
        }

        @Test func cacheReportsLoadedModels() async throws {
            let model = makeModel()
            _ = try await DecisionSession(model: model, state: "state").decide(questions[0])
            #expect(model.isLoaded)
            model.removeFromCache()
            #expect(!model.isLoaded)
        }

        @Test func cancellationStopsEvaluation() async throws {
            let session = DecisionSession(model: makeModel(), state: .text(ticket))
            try await session.prewarm()
            let task = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await session.decide(questions)
            }
            await #expect(throws: CancellationError.self) { _ = try await task.value }
        }
    }
#endif
