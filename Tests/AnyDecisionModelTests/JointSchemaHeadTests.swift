import Foundation
import Testing

@testable import AnyDecisionModel

#if MLX
    import MLX
    import MLXNN

    /// A small random head and its logits from the reference `JointSchemaHead` in `clef_mlx.py`,
    /// written by `scripts/joint-schema-fixtures.py head`.
    private struct HeadFixture: Decodable {
        struct Field: Decodable {
            var typeID: Int
            var questionSpan: [Int]
            var optionSpans: [[Int]]

            enum CodingKeys: String, CodingKey {
                case typeID = "type_id"
                case questionSpan = "question_span"
                case optionSpans = "option_spans"
            }

            var spans: JointSchemaHead.FieldSpans {
                JointSchemaHead.FieldSpans(
                    typeID: typeID,
                    questionSpan: questionSpan[0] ..< questionSpan[1],
                    optionSpans: optionSpans.map { $0[0] ..< $0[1] }
                )
            }
        }

        var config: JSONValue
        var fields: [Field]
        var arrays: [String: MLXArray] = [:]

        enum CodingKeys: String, CodingKey {
            case config, fields
        }

        static func load() throws -> HeadFixture {
            var fixture = try JSONDecoder().decode(HeadFixture.self, from: fixtureData("head.json"))
            let url = try #require(
                Bundle.module.url(
                    forResource: "head",
                    withExtension: "safetensors",
                    subdirectory: "Fixtures/JointSchema"
                )
            )
            fixture.arrays = try loadArrays(url: url)
            return fixture
        }

        /// The head weights under their original names.
        var weights: [String: MLXArray] {
            Dictionary(
                uniqueKeysWithValues: arrays.compactMap { key, value in
                    key.hasPrefix("head.") ? (String(key.dropFirst("head.".count)), value) : nil
                }
            )
        }

        /// Writes the head files as a model directory has them.
        func writeModelDirectory() throws -> URL {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("JointSchemaHeadTests-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try JSONEncoder().encode(config).write(to: directory.appendingPathComponent("joint_head_config.json"))
            try save(arrays: weights, url: directory.appendingPathComponent("joint_head.safetensors"))
            return directory
        }
    }

    private func maximumDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    @Suite("JointSchemaHead", .enabled(if: shouldRunMLXTests || shouldRunMLXHeadTests))
    struct JointSchemaHeadTests {
        @Test func logitsMatchTheReference() throws {
            let fixture = try HeadFixture.load()
            let directory = try fixture.writeModelDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }

            let head = try JointSchemaHead.load(from: directory)
            let lexical = try #require(fixture.arrays["lexical"])
            let logits = head(
                hidden: try #require(fixture.arrays["hidden"]),
                inputIDs: try #require(fixture.arrays["input_ids"]),
                fields: fixture.fields.map(\.spans),
                lexical: { lexical[$0] }
            )

            #expect(logits.count == fixture.fields.count)
            for (index, value) in logits.enumerated() {
                let expected = try #require(fixture.arrays["logits.\(index)"])
                #expect(value.shape == expected.shape)
                #expect(maximumDifference(value, expected) < 1e-4, "field \(index)")
            }
        }

        @Test func loadingRejectsMissingWeights() throws {
            let fixture = try HeadFixture.load()
            let directory = try fixture.writeModelDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }

            var weights = fixture.weights
            weights["residual_gate"] = nil
            try save(arrays: weights, url: directory.appendingPathComponent("joint_head.safetensors"))
            #expect(throws: (any Error).self) { _ = try JointSchemaHead.load(from: directory) }
        }

        @Test func answerLogitsFollowTheQuestionOrder() throws {
            let fixture = try HeadFixture.load()
            let directory = try fixture.writeModelDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let head = try JointSchemaHead.load(from: directory)

            // The choice field, as a prompt that starts after a prefix of 10 tokens.
            let spans = fixture.fields[1].spans
            let prefixLength = 10
            let field = JointSchemaEncoding.Field(
                tokens: [],
                typeID: spans.typeID,
                questionSpan: (spans.questionSpan.lowerBound - prefixLength)
                    ..< (spans.questionSpan.upperBound - prefixLength),
                optionSpans: spans.optionSpans.map {
                    ($0.lowerBound - prefixLength) ..< ($0.upperBound - prefixLength)
                },
                answerIndices: [2, 0, 1]
            )
            let embedding = OutputEmbedding(Linear(weight: try #require(fixture.arrays["lexical"])))
            let logits = head.logits(
                hidden: try #require(fixture.arrays["hidden"]),
                tokens: try #require(fixture.arrays["input_ids"]),
                prefixLength: prefixLength,
                prompt: field,
                outputEmbedding: embedding
            )

            // The fixture's logits come from all three fields together,
            // and the head's decoder layers attend across fields, so score this field alone.
            let lexical = try #require(fixture.arrays["lexical"])
            let expected = head(
                hidden: try #require(fixture.arrays["hidden"]),
                inputIDs: try #require(fixture.arrays["input_ids"]),
                fields: [spans],
                lexical: { lexical[$0] }
            )[0].asArray(Float.self)
            #expect(logits.count == 3)
            for (position, answerIndex) in field.answerIndices.enumerated() {
                #expect(abs(logits[answerIndex] - expected[position]) < 1e-4)
            }
        }

        @Test func quantizedOutputEmbeddingRowsMatchTheDequantizedMatrix() throws {
            let fixture = try HeadFixture.load()
            let linear = Linear(weight: try #require(fixture.arrays["lexical"]))
            let quantized = QuantizedLinear(linear, groupSize: 32, bits: 4)
            let ids = MLXArray([3, 0, 63, 3].map(Int32.init))

            let all = dequantized(
                quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: 32,
                bits: 4
            )
            let rows = OutputEmbedding(quantized).rows(ids)
            #expect(rows.shape == [4, 32])
            #expect(maximumDifference(rows, all[ids]) == 0)
            #expect(maximumDifference(OutputEmbedding(linear).rows(ids), linear.weight[ids]) == 0)
        }
    }
#endif
