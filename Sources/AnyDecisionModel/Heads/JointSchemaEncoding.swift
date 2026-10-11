import Foundation

/// Builds prompts for the joint schema head of Cloudflare's Clef models.
///
/// The format follows `encode_record` in the reference implementation, `clef_mlx.py`.
/// A prompt is the system prompt, the state, a schema with one field for each question,
/// and a closing suffix.
/// Each segment is tokenized separately and the tokens are concatenated,
/// so the prompt for a state always starts with the same tokens.
///
/// Every prompt here has one field, so that the answer to a question
/// does not depend on the other questions in a batch.
/// The field always has the same ID, ``questionID``.
enum JointSchemaEncoding {
    /// The largest number of tokens in a prompt, the reference default.
    static let maximumLength = 16_384

    /// The question ID in every prompt.
    static let questionID = "question"

    /// The system prompt that the head was trained with.
    static let systemPrompt =
        "Read the complete state and schema. Decide every field jointly. "
        + "Each answer must be exactly one of that field's allowed options."

    /// The schema field for one question, followed by the closing suffix.
    struct Field: Hashable, Sendable {
        /// The tokens of the field and the closing suffix.
        var tokens: [Int]

        /// The question type: 0 for binary, 1 for choice, 2 for score.
        var typeID: Int

        /// The positions of the instruction tokens in ``tokens``.
        var questionSpan: Range<Int>

        /// The positions of each option's tokens in ``tokens``, in the order the head scores them.
        var optionSpans: [Range<Int>]

        /// For each option in head order, the index of the matching answer in the question.
        var answerIndices: [Int]
    }

    /// Returns the tokens that every prompt about a state starts with.
    ///
    /// - Parameters:
    ///   - state: The state.
    ///   - encode: A function that tokenizes text without adding special tokens.
    static func prefixTokens(for state: DecisionState, encode: (String) -> [Int]) -> [Int] {
        encode("<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\nSTATE:\n")
            + encode(render(state.jsonValue))
            + encode("\n\nSCHEMA FIELDS:\n")
    }

    /// Returns the tokens that follow the prefix in the prompt for a question.
    ///
    /// - Parameters:
    ///   - question: The question.
    ///   - encode: A function that tokenizes text without adding special tokens.
    static func field(for question: Question, encode: (String) -> [Int]) -> Field {
        let (type, typeID, options) = schemaOptions(for: question)

        var tokens = encode("\nFIELD 1\nID: \(questionID)\nTYPE: \(type)\nINSTRUCTION: ")
        let questionStart = tokens.count
        tokens += encode(question.instructions)
        let questionSpan = questionStart ..< tokens.count
        tokens += encode("\nALLOWED OPTIONS:\n")

        var optionSpans: [Range<Int>] = []
        for (position, option) in options.enumerated() {
            tokens += encode("OPTION \(position + 1): ")
            let optionStart = tokens.count
            var semantics: [String: JSONValue] = ["option_id": .string(option.id)]
            if let description = option.description {
                semantics["description"] = .string(description)
            }
            tokens += encode(render(.object(semantics)))
            optionSpans.append(optionStart ..< tokens.count)
            tokens += encode("\n")
        }
        tokens += encode("END FIELD\n")
        tokens += encode("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")

        return Field(
            tokens: tokens,
            typeID: typeID,
            questionSpan: questionSpan,
            optionSpans: optionSpans,
            answerIndices: options.map(\.answerIndex)
        )
    }

    /// One allowed option of a schema field.
    private struct SchemaOption {
        var id: String
        var description: String?
        var answerIndex: Int
    }

    /// Returns the schema type, its ID, and the options of a question, in head order.
    private static func schemaOptions(for question: Question) -> (String, Int, [SchemaOption]) {
        switch question {
        case .binary(_, let criteria):
            return (
                "noul", 0,
                [
                    SchemaOption(
                        id: "true",
                        description: criteria?.whenTrue ?? "The proposition is true or the answer is yes.",
                        answerIndex: 0
                    ),
                    SchemaOption(
                        id: "false",
                        description: criteria?.whenFalse ?? "The proposition is false or the answer is no.",
                        answerIndex: 1
                    ),
                ]
            )
        case .choice(_, let options):
            // The reference sorts options by name, comparing Unicode scalar values.
            let sorted = options.enumerated().sorted { precedes($0.element.name, $1.element.name) }
            return (
                "choice", 1,
                sorted.map {
                    SchemaOption(id: $0.element.name, description: $0.element.description, answerIndex: $0.offset)
                }
            )
        case .score(_, let levels):
            return (
                "score", 2,
                levels.enumerated().map {
                    SchemaOption(id: String($0.offset), description: $0.element, answerIndex: $0.offset)
                }
            )
        }
    }

    // MARK: - Rendering

    /// Returns the text of a value as the reference renders it.
    ///
    /// Strings are used as they are.
    /// Other values are rendered like Python's
    /// `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
    /// Numbers are stored as `Double`,
    /// so integral values with a magnitude below 2^53 render as integers.
    static func render(_ value: JSONValue) -> String {
        if case .string(let text) = value {
            return text
        }
        var output = ""
        appendJSON(value, to: &output)
        return output
    }

    private static func appendJSON(_ value: JSONValue, to output: inout String) {
        switch value {
        case .null:
            output += "null"
        case .bool(let value):
            output += value ? "true" : "false"
        case .number(let value):
            output += pythonNumber(value)
        case .string(let value):
            appendString(value, to: &output)
        case .array(let values):
            output += "["
            for (index, element) in values.enumerated() {
                if index > 0 { output += "," }
                appendJSON(element, to: &output)
            }
            output += "]"
        case .object(let members):
            output += "{"
            for (index, key) in members.keys.sorted(by: precedes).enumerated() {
                if index > 0 { output += "," }
                appendString(key, to: &output)
                output += ":"
                appendJSON(members[key]!, to: &output)
            }
            output += "}"
        }
    }

    /// Returns a number as Python's `json.dumps` writes it.
    private static func pythonNumber(_ value: Double) -> String {
        if value.isNaN { return "NaN" }
        if value.isInfinite { return value < 0 ? "-Infinity" : "Infinity" }
        if value == 0, value.sign == .minus { return "-0.0" }
        if value == value.rounded(), abs(value) < 0x1p53 {
            return String(Int64(value))
        }
        // Swift and Python both write the shortest representation that round-trips,
        // with the same switch to exponent notation.
        return value.description
    }

    /// Appends a quoted string, escaped as Python's `json.dumps` escapes it with `ensure_ascii=False`.
    private static func appendString(_ value: String, to output: inout String) {
        output += "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": output += "\\\""
            case "\\": output += "\\\\"
            case "\n": output += "\\n"
            case "\r": output += "\\r"
            case "\t": output += "\\t"
            case "\u{08}": output += "\\b"
            case "\u{0C}": output += "\\f"
            case _ where scalar.value < 0x20:
                let hex = String(scalar.value, radix: 16)
                output += "\\u" + String(repeating: "0", count: 4 - hex.count) + hex
            default:
                output.unicodeScalars.append(scalar)
            }
        }
        output += "\""
    }

    /// Returns whether `a` sorts before `b` when comparing Unicode scalar values, as Python sorts strings.
    private static func precedes(_ a: String, _ b: String) -> Bool {
        a.unicodeScalars.lexicographicallyPrecedes(b.unicodeScalars) { $0.value < $1.value }
    }
}
