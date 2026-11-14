import Foundation

// MARK: - Request

/// A batch decision request in the System One `/v1/systemone` wire format.
///
/// Jev, Ollama, and other System One servers implement this format,
/// so their backends share these types.
struct SystemOneRequest: Encodable {
    /// The model identifier.
    var model: String

    /// The state that every question is about.
    var state: DecisionState

    /// The questions, keyed by question identifier.
    var questions: [String: SystemOneQuestion]

    /// Extra top-level fields, such as Ollama's `keep_alive`.
    ///
    /// The fields are encoded after the core fields.
    /// A key that is also a core field is ignored,
    /// so that it cannot replace the core field.
    var extraBody: [String: JSONValue] = [:]

    private struct FieldKey: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }

        init(_ stringValue: String) {
            self.stringValue = stringValue
        }

        init?(stringValue: String) {
            self.init(stringValue)
        }

        init?(intValue: Int) {
            return nil
        }

        static let model = FieldKey("model")
        static let state = FieldKey("state")
        static let questions = FieldKey("questions")
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: FieldKey.self)
        try container.encode(model, forKey: .model)
        // Encode the JSON value, not the state's tagged `Codable` form,
        // so that every JSON state is sent unchanged.
        // A state with attachments throws, so that it is never uploaded.
        try container.encode(state.textOrJSONValue(), forKey: .state)
        try container.encode(questions, forKey: .questions)
        for (key, value) in extraBody where !SystemOneDecisionModel.reservedBodyFields.contains(key) {
            try container.encode(value, forKey: FieldKey(key))
        }
    }
}

/// A question in the System One wire format.
///
/// The Gemini Decisions API uses the same question format,
/// except for the names of the binary criteria keys
/// and the value for an option without a description.
struct SystemOneQuestion: Encodable {
    /// A variant of the question wire format.
    enum Dialect {
        /// System One, as Jev and Ollama implement it.
        ///
        /// Binary criteria use the keys `true` and `false`,
        /// and an option without a description is `null`.
        case systemOne

        /// The Gemini Decisions API.
        ///
        /// Binary criteria use the keys `true_description` and `false_description`,
        /// and an option without a description is an empty string.
        case gemini
    }

    var question: Question
    var dialect: Dialect

    init(_ question: Question) {
        self.init(question, dialect: .systemOne)
    }

    init(_ question: Question, dialect: Dialect) {
        self.question = question
        self.dialect = dialect
    }

    /// The wire name for the question type.
    private var typeName: String {
        switch question {
        case .binary: return "noul"
        case .choice: return "choice"
        case .score: return "score"
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case instructions
        case criteria
    }

    private enum BinaryCriteriaKeys: String, CodingKey {
        case whenTrue = "true"
        case whenFalse = "false"
    }

    private enum GeminiBinaryCriteriaKeys: String, CodingKey {
        case whenTrue = "true_description"
        case whenFalse = "false_description"
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(typeName, forKey: .type)

        switch question {
        case .binary(let instructions, let criteria):
            try container.encode(instructions, forKey: .instructions)
            guard let criteria else { break }
            switch dialect {
            case .systemOne:
                var nested = container.nestedContainer(keyedBy: BinaryCriteriaKeys.self, forKey: .criteria)
                try nested.encode(criteria.whenTrue, forKey: .whenTrue)
                try nested.encode(criteria.whenFalse, forKey: .whenFalse)
            case .gemini:
                var nested = container.nestedContainer(keyedBy: GeminiBinaryCriteriaKeys.self, forKey: .criteria)
                try nested.encode(criteria.whenTrue, forKey: .whenTrue)
                try nested.encode(criteria.whenFalse, forKey: .whenFalse)
            }
        case .choice(let instructions, let options):
            try container.encode(instructions, forKey: .instructions)
            switch dialect {
            case .systemOne:
                var criteria: [String: String?] = [:]
                for option in options {
                    criteria[option.name] = .some(option.description)
                }
                try container.encode(criteria, forKey: .criteria)
            case .gemini:
                var criteria: [String: String] = [:]
                for option in options {
                    criteria[option.name] = option.description ?? ""
                }
                try container.encode(criteria, forKey: .criteria)
            }
        case .score(let instructions, let levels):
            try container.encode(instructions, forKey: .instructions)
            try container.encode(levels, forKey: .criteria)
        }
    }
}

// MARK: - Response

/// A batch decision response in the System One `/v1/systemone` wire format.
///
/// Some servers add fields that version 0.8 has no place for:
/// a `confidence` on a `noul` answer,
/// a `label_mass` on an answer,
/// and `cached_tokens` and `cost` in `usage`.
/// The decoder ignores these fields and every other unknown field.
/// A `refusal` answer or an answer of an unknown type, such as `box`,
/// fails to decode.
struct SystemOneResponse: Decodable {
    /// The identifier of the model that answered.
    var model: String

    /// The answers, keyed by question identifier.
    var answers: [String: SystemOneAnswer]

    /// Token usage for the request.
    var usage: Usage?

    struct Usage: Decodable {
        var inputTokenCount: Int
        var outputTokenCount: Int

        private enum CodingKeys: String, CodingKey {
            case inputTokenCount = "input_tokens"
            case outputTokenCount = "output_tokens"
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            inputTokenCount = try container.decodeIfPresent(Int.self, forKey: .inputTokenCount) ?? 0
            outputTokenCount = try container.decodeIfPresent(Int.self, forKey: .outputTokenCount) ?? 0
        }
    }
}

/// An answer in the System One wire format.
struct SystemOneAnswer: Decodable {
    var answer: Answer

    private enum CodingKeys: String, CodingKey {
        case type
        case noul
        case choice
        case score
        case probabilities
        case legend
        case confidence
        case refusal
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decodeIfPresent(String.self, forKey: .type)

        if type == "noul" || (type == nil && container.contains(.noul)) {
            let probability = try container.decode(Double.self, forKey: .noul)
            answer = .binary(probability: probability)
        } else if type == "choice" || (type == nil && container.contains(.choice)) {
            answer = .choice(
                name: try container.decode(String.self, forKey: .choice),
                probabilities: try container.decode([String: Double].self, forKey: .probabilities),
                confidence: try container.decode(Double.self, forKey: .confidence)
            )
        } else if type == "score" || (type == nil && container.contains(.score)) {
            let keyed = try container.decode([String: Double].self, forKey: .probabilities)
            var probabilities: [Double] = []
            for index in 0 ..< keyed.count {
                guard let value = keyed[String(index)] else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .probabilities,
                        in: container,
                        debugDescription: "Score probabilities must be keyed by level index from 0."
                    )
                }
                probabilities.append(value)
            }
            answer = .score(
                value: try container.decode(Double.self, forKey: .score),
                probabilities: probabilities,
                legend: Self.decodeLegend(from: container),
                confidence: try container.decode(Double.self, forKey: .confidence)
            )
        } else if type == "refusal" || (type == nil && container.contains(.refusal)) {
            // Some servers refuse one question and answer the others.
            // Version 0.8 has no way to report one failed question,
            // so a refusal fails the whole response.
            let message = (try? container.decodeIfPresent(String.self, forKey: .refusal)) ?? nil
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: message.map { "The model refused to answer: \($0)" }
                        ?? "The model refused to answer."
                )
            )
        } else if let type {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "The answer type \"\(type)\" is not supported."
                )
            )
        } else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "The answer has no \"noul\", \"choice\", or \"score\" field."
                )
            )
        }
    }

    /// Decodes a score legend.
    ///
    /// Jev returns the legend as a list of level descriptions.
    /// Ollama returns it as an object keyed by level index.
    /// The legend is informational,
    /// so a legend in neither form decodes as `nil`.
    private static func decodeLegend(from container: KeyedDecodingContainer<CodingKeys>) -> [String]? {
        if let list = try? container.decode([String].self, forKey: .legend) {
            return list
        }
        guard let keyed = try? container.decode([String: String].self, forKey: .legend) else {
            return nil
        }
        var legend: [String] = []
        for index in 0 ..< keyed.count {
            guard let value = keyed[String(index)] else { return nil }
            legend.append(value)
        }
        return legend
    }
}
