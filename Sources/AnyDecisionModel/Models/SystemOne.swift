import Foundation

// MARK: - Request

/// A batch decision request in the System One `/v1/systemone` wire format.
///
/// Jev and Ollama both implement this format,
/// so their backends share these types.
struct SystemOneRequest: Encodable {
    /// The model identifier.
    var model: String

    /// The state that every question is about.
    var state: DecisionState

    /// The questions, keyed by question identifier.
    var questions: [String: SystemOneQuestion]

    /// How long to keep the model loaded after the request, in seconds.
    ///
    /// Only Ollama reads this field. `nil` omits it.
    var keepAlive: TimeInterval?

    private enum CodingKeys: String, CodingKey {
        case model
        case state
        case questions
        case keepAlive = "keep_alive"
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(model, forKey: .model)
        try container.encode(state, forKey: .state)
        try container.encode(questions, forKey: .questions)
        try container.encodeIfPresent(keepAlive, forKey: .keepAlive)
    }
}

/// A question in the System One wire format.
struct SystemOneQuestion: Encodable {
    var question: Question

    init(_ question: Question) {
        self.question = question
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

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(typeName, forKey: .type)

        switch question {
        case .binary(let instructions, let criteria):
            try container.encode(instructions, forKey: .instructions)
            if let criteria {
                var nested = container.nestedContainer(keyedBy: BinaryCriteriaKeys.self, forKey: .criteria)
                try nested.encode(criteria.whenTrue, forKey: .whenTrue)
                try nested.encode(criteria.whenFalse, forKey: .whenFalse)
            }
        case .choice(let instructions, let options):
            try container.encode(instructions, forKey: .instructions)
            var criteria: [String: String?] = [:]
            for option in options {
                criteria[option.name] = .some(option.description)
            }
            try container.encode(criteria, forKey: .criteria)
        case .score(let instructions, let levels):
            try container.encode(instructions, forKey: .instructions)
            try container.encode(levels, forKey: .criteria)
        }
    }
}

// MARK: - Response

/// A batch decision response in the System One `/v1/systemone` wire format.
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
