import Foundation
import Testing

@testable import AnyDecisionModel

/// Prompts and spans from the reference `encode_record` in `clef_mlx.py`,
/// written by `scripts/joint-schema-fixtures.py encoding`.
///
/// The reference ran with a tokenizer that maps each Unicode scalar to its value,
/// so the expected tokens are the prompt text itself.
struct EncodingFixture: Decodable {
    struct Case: Decodable {
        var name: String
        var state: StateSpec
        var question: QuestionSpec
        var inputIDs: [Int]
        var prefixLength: Int
        var typeID: Int
        var questionSpan: [Int]
        var optionSpans: [[Int]]
        var answerIndices: [Int]

        enum CodingKeys: String, CodingKey {
            case name, state, question
            case inputIDs = "input_ids"
            case prefixLength = "prefix_length"
            case typeID = "type_id"
            case questionSpan = "question_span"
            case optionSpans = "option_spans"
            case answerIndices = "answer_indices"
        }
    }

    struct Render: Decodable {
        var value: JSONValue
        var text: String
    }

    var revision: String
    var cases: [Case]
    var renders: [Render]

    static func load() throws -> EncodingFixture {
        try JSONDecoder().decode(EncodingFixture.self, from: fixtureData("encoding.json"))
    }
}

/// A state in a fixture: text or JSON.
struct StateSpec: Decodable {
    var text: String?
    var json: JSONValue?

    var state: DecisionState {
        if let text { return .text(text) }
        return .json(json ?? .null)
    }
}

/// A question in a fixture, in AnyDecisionModel terms.
struct QuestionSpec: Decodable {
    struct Criteria: Decodable {
        var whenTrue: String
        var whenFalse: String
    }

    struct Option: Decodable {
        var name: String
        var description: String?
    }

    var type: String
    var instructions: String
    var criteria: Criteria?
    var options: [Option]?
    var levels: [String]?

    var question: Question {
        switch type {
        case "binary":
            return .binary(
                instructions: instructions,
                criteria: criteria.map { BinaryCriteria(whenTrue: $0.whenTrue, whenFalse: $0.whenFalse) }
            )
        case "choice":
            return .choice(
                instructions: instructions,
                options: (options ?? []).map { ChoiceOption($0.name, description: $0.description) }
            )
        default:
            return .score(instructions: instructions, levels: levels ?? [])
        }
    }
}

/// Returns the contents of a file in `Fixtures/JointSchema`.
func fixtureData(_ name: String) throws -> Data {
    let url = try #require(
        Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures/JointSchema")
    )
    return try Data(contentsOf: url)
}

/// Tokenizes text as its Unicode scalar values, like the tokenizer that wrote the fixtures.
func scalarTokens(_ text: String) -> [Int] {
    text.unicodeScalars.map { Int($0.value) }
}

@Suite("JointSchemaEncoding")
struct JointSchemaEncodingTests {
    @Test func promptsMatchTheReference() throws {
        let fixture = try EncodingFixture.load()
        #expect(!fixture.cases.isEmpty)
        for item in fixture.cases {
            let prefix = JointSchemaEncoding.prefixTokens(for: item.state.state, encode: scalarTokens)
            let field = JointSchemaEncoding.field(for: item.question.question, encode: scalarTokens)
            let comment = Comment(rawValue: item.name)

            #expect(prefix.count == item.prefixLength, comment)
            #expect(prefix + field.tokens == item.inputIDs, comment)
            #expect(field.typeID == item.typeID, comment)

            let offset = prefix.count
            #expect(
                [field.questionSpan.lowerBound + offset, field.questionSpan.upperBound + offset] == item.questionSpan,
                comment
            )
            #expect(
                field.optionSpans.map { [$0.lowerBound + offset, $0.upperBound + offset] } == item.optionSpans,
                comment
            )
            #expect(field.answerIndices == item.answerIndices, comment)
        }
    }

    @Test func jsonRendersLikePython() throws {
        let fixture = try EncodingFixture.load()
        #expect(!fixture.renders.isEmpty)
        for render in fixture.renders {
            #expect(JointSchemaEncoding.render(render.value) == render.text)
        }
    }

    @Test func nonFiniteNumbersRenderLikePython() {
        #expect(
            JointSchemaEncoding.render([.number(.nan), .number(.infinity), .number(-.infinity)])
                == "[NaN,Infinity,-Infinity]"
        )
    }

    @Test func promptLimitMatchesTheReferenceDefault() {
        #expect(JointSchemaEncoding.maximumLength == 16_384)
    }
}
