import Foundation

/// Builds prompts and reads answers for Liquid AI's d1 decision models on LFM2-VL, such as d1-3B.
///
/// This follows `prompt.py` in the
/// [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) repository
/// with its defaults:
/// no system turn, every state written as the value it is,
/// each option written as its code and its description, and no calibration.
/// A JSON state can differ from `prompt.py`'s in two ways that `JSONValue` causes:
/// object keys are sorted, and whole numbers below 2^53 are written as integers.
/// See ``stateBlock(_:)``.
/// The model reads each answer from the next-token distribution
/// at the end of the prompt.
/// Each answer scores the largest log-probability among its tokens,
/// and a softmax over those scores gives the answer probabilities.
///
/// The helpers here do not depend on MLX, so they can be tested with synthetic logits.
enum D1LFM2Prompt {
    /// The beginning of a chat turn.
    static let turnStart = "<|im_start|>"

    /// The end of a chat turn.
    static let turnEnd = "<|im_end|>"

    /// The texts whose tokens are read for "yes", in order.
    static let yesForms = ["yes", "Yes", "YES"]

    /// The texts whose tokens are read for "no", in order.
    static let noForms = ["no", "No", "NO"]

    /// The codes that a choice option takes, in order,
    /// when its own code is not a single token or is taken.
    ///
    /// Capital letters, then the numbers 00 to 99, then small letters,
    /// then `#0` to `#199`, then every pair of capital letters.
    static let fallbackCodes: [String] = {
        // Swift 6.2 can't type-check these parts as one chained expression in reasonable time,
        // so each part has its own type.
        let capitals: [String] = letters(from: "A", through: "Z")
        let numbers: [String] = (0 ..< 100).map(twoDigitCode)
        let smallLetters: [String] = letters(from: "a", through: "z")
        let hashes: [String] = (0 ..< 200).map { "#\($0)" }
        let pairs: [String] = capitals.flatMap { first in capitals.map { second in first + second } }

        var codes: [String] = []
        codes.append(contentsOf: capitals)
        codes.append(contentsOf: numbers)
        codes.append(contentsOf: smallLetters)
        codes.append(contentsOf: hashes)
        codes.append(contentsOf: pairs)
        return codes
    }()

    /// A question as d1 reads it:
    /// the text between the state and the end of the user turn,
    /// and the token IDs read for each answer.
    struct RenderedQuestion: Hashable, Sendable {
        /// The question, its options or levels, and the reply instructions.
        let text: String

        /// For each answer, in the order of the rendered options,
        /// the token IDs whose largest log-probability is the answer's score.
        let tokenGroups: [[Int]]
    }

    /// Returns whether a model's `config.json` declares a d1 model that this prompt fits.
    ///
    /// This matches an LFM2-VL model (`model_type` `lfm2_vl`)
    /// whose `auto_map.AutoModel` ends in `.D1Model`, as mlx-vlm and llama.cpp detect d1.
    /// d1 models on other backbones, such as LFM3, use a different prompt, so they don't match.
    static func isD1LFM2Configuration(_ data: Data) -> Bool {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            object["model_type"] as? String == "lfm2_vl",
            let autoMap = object["auto_map"] as? [String: Any],
            let autoModel = autoMap["AutoModel"] as? String
        else { return false }
        return autoModel.hasSuffix(".D1Model")
    }

    // MARK: - Prompt

    /// Returns the complete prompt for a question about a state.
    ///
    /// - Parameters:
    ///   - state: The state.
    ///   - question: The question, rendered by ``render(_:optionOrder:index:encode:)``.
    ///   - bos: The tokenizer's beginning-of-sequence token, or an empty string.
    /// - Throws: ``DecisionError/unsupportedState(_:)`` if the state has attachments.
    static func prompt(state: DecisionState, question: RenderedQuestion, bos: String) throws -> String {
        try prefix(state: state, bos: bos) + suffix(question: question)
    }

    /// Returns the part of the prompt that every question about a state shares.
    ///
    /// A top-level null state is d1's state of `None`,
    /// so the question follows the start of the user turn directly,
    /// with no state block and no `QUESTION:` line.
    /// A null inside an array or object is written as `null`.
    ///
    /// - Throws: ``DecisionError/unsupportedState(_:)`` if the state has attachments.
    static func prefix(state: DecisionState, bos: String) throws -> String {
        let value = try state.textOrJSONValue()
        let body = value == .null ? "" : "\(stateBlock(value))\nQUESTION:\n"
        return "\(bos)\(turnStart)user\n\(body)"
    }

    /// Returns the part of the prompt after the state, up to the answer.
    static func suffix(question: RenderedQuestion) -> String {
        "\(question.text)\(turnEnd)\n\(turnStart)assistant\n"
    }

    /// Returns the state as d1 reads it, followed by a blank line.
    ///
    /// A string is written as it is.
    /// Any other value is written as JSON with an indent of two spaces,
    /// as Python's `json.dumps(state, ensure_ascii=False, indent=2)` writes it,
    /// with two differences that come from `JSONValue`.
    /// `JSONValue` objects have no key order, so object keys are sorted by code point.
    /// `JSONValue` keeps every number as a `Double`,
    /// so ``pythonNumber(_:)`` writes whole numbers below 2^53 as integers,
    /// where Python keeps a `float` such as `27.0` or `-0.0` as it is.
    static func stateBlock(_ value: JSONValue) -> String {
        if case .string(let text) = value {
            return "\(text)\n\n"
        }
        return pythonJSON(value) + "\n\n"
    }

    // MARK: - Questions

    /// Renders a question and finds the tokens read for its answers.
    ///
    /// - Parameters:
    ///   - question: The question.
    ///   - optionOrder: For choice questions, the order in which to list the options,
    ///     as indices into the question's options. Pass `nil` for the question's order.
    ///   - index: The position of the question in its batch, used in errors.
    ///   - encode: A function that tokenizes text without special tokens.
    /// - Throws: ``DecisionError/unsupportedQuestion(index:reason:)``
    ///   if an answer has no single-token form.
    static func render(
        _ question: Question,
        optionOrder: [Int]? = nil,
        index: Int,
        encode: (String) -> [Int]
    ) throws -> RenderedQuestion {
        switch question {
        case .binary(let instructions, let criteria):
            let yes = singleTokenIDs(of: yesForms, encode: encode)
            let no = singleTokenIDs(of: noForms, encode: encode)
            guard !yes.isEmpty, !no.isEmpty else {
                throw DecisionError.unsupportedQuestion(
                    index: index,
                    reason: "The tokenizer has no single-token \"yes\" or \"no\"."
                )
            }
            var text = instructions
            if let criteria {
                text += "\nYes: \(criteria.whenTrue)\nNo: \(criteria.whenFalse)"
            }
            return RenderedQuestion(text: text + "\n\nReply with yes or no only.", tokenGroups: [yes, no])

        case .choice(let instructions, let options):
            let ordered = (optionOrder ?? Array(options.indices)).map { options[$0] }
            let codes = try aliases(for: ordered.map(\.name), index: index, encode: encode)
            var lines: [String] = []
            var groups: [[Int]] = []
            for (option, alias) in zip(ordered, codes) {
                // Python's `description or name` also skips an empty description.
                let description = option.description.flatMap { $0.isEmpty ? nil : $0 }
                lines.append("\(alias.code) \(description ?? option.name.replacingOccurrences(of: "_", with: " "))")
                let spaced = singleTokenIDs(of: [" \(alias.code)"], encode: encode)
                groups.append([alias.token] + spaced.filter { $0 != alias.token })
            }
            let text =
                "\(instructions)\n\nOptions:\n\(lines.joined(separator: "\n"))\n\n"
                + "Reply with the option code only."
            return RenderedQuestion(text: text, tokenGroups: groups)

        case .score(let instructions, let levels):
            var groups: [[Int]] = []
            for level in levels.indices {
                let ids = singleTokenIDs(of: [String(level)], encode: encode)
                guard !ids.isEmpty else {
                    throw DecisionError.unsupportedQuestion(
                        index: index,
                        reason: "The score level \(level) has no single-token digit."
                    )
                }
                groups.append(ids)
            }
            let legend = levels.enumerated().map { "\($0) \($1)" }.joined(separator: "\n")
            let text =
                "\(instructions)\n\n\(legend)\n\n"
                + "Reply with a single digit 0-\(levels.count - 1) only."
            return RenderedQuestion(text: text, tokenGroups: groups)
        }
    }

    /// Returns the code that d1 gives each choice option before checking the tokenizer.
    ///
    /// When every option name, without surrounding whitespace, is a single letter,
    /// the names are their own codes.
    /// Otherwise, up to 26 options are coded A to Z, and more options are coded 00, 01, and so on.
    static func optionCodes(for names: [String]) -> [String] {
        let labels = names.map(strippingPythonWhitespace)
        if !labels.isEmpty, labels.allSatisfy(isSingleLetter) {
            return labels
        }
        if labels.count <= 26 {
            return Array(TokenScoring.choiceLabels.prefix(labels.count))
        }
        return labels.indices.map(twoDigitCode)
    }

    /// Gives every choice option a distinct code that is a single token.
    ///
    /// An option whose code from ``optionCodes(for:)`` is not a single token, or is taken,
    /// gets the first code from ``fallbackCodes`` that is a single token and not taken.
    ///
    /// - Returns: For each option, its code and that code's token ID.
    /// - Throws: ``DecisionError/unsupportedQuestion(index:reason:)``
    ///   if no single-token code is left for an option.
    static func aliases(
        for names: [String],
        index: Int,
        encode: (String) -> [Int]
    ) throws -> [(code: String, token: Int)] {
        var used = Set<Int>()
        var aliases: [(code: String, token: Int)] = []

        func take(_ code: String) -> Bool {
            let ids = encode(code)
            guard ids.count == 1, !used.contains(ids[0]) else { return false }
            aliases.append((code, ids[0]))
            used.insert(ids[0])
            return true
        }

        for code in optionCodes(for: names) {
            if take(code) { continue }
            guard fallbackCodes.contains(where: take) else {
                throw DecisionError.unsupportedQuestion(
                    index: index,
                    reason: "No single-token option code is left for \(names.count) options."
                )
            }
        }
        return aliases
    }

    // MARK: - Readout

    /// Scores each answer by the largest log-probability among its tokens.
    ///
    /// - Parameter tokenLogProbabilities: For each answer,
    ///   the log-probabilities of its tokens under the full next-token distribution.
    /// - Returns: The score of each answer,
    ///   and the total probability of all the answers' tokens.
    static func scores(_ tokenLogProbabilities: [[Double]]) -> (scores: [Double], allowedAnswerMass: Double) {
        let scores = tokenLogProbabilities.map { $0.max() ?? -.infinity }
        let coverage = exp(tokenLogProbabilities.joined().logSumExp())
        return (scores, min(max(coverage, 0), 1))
    }

    // MARK: - Helpers

    /// Returns the token ID of each text that is a single token, without repeats, in order.
    private static func singleTokenIDs(of texts: [String], encode: (String) -> [Int]) -> [Int] {
        var ids: [Int] = []
        for text in texts {
            let encoded = encode(text)
            if encoded.count == 1, !ids.contains(encoded[0]) {
                ids.append(encoded[0])
            }
        }
        return ids
    }

    /// Returns each letter from one Unicode scalar through another, in order.
    private static func letters(from first: UnicodeScalar, through last: UnicodeScalar) -> [String] {
        (first.value ... last.value).compactMap { UnicodeScalar($0).map(String.init) }
    }

    private static func twoDigitCode(_ number: Int) -> String {
        number < 10 ? "0\(number)" : String(number)
    }

    /// Whether a string is one Unicode scalar that Python's `str.isalpha()` accepts.
    private static func isSingleLetter(_ string: String) -> Bool {
        let scalars = string.unicodeScalars
        guard scalars.count == 1, let scalar = scalars.first else { return false }
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
            return true
        default:
            return false
        }
    }

    /// Removes leading and trailing whitespace, as Python's `str.strip()` does.
    private static func strippingPythonWhitespace(_ string: String) -> String {
        // Python's `str.isspace()` also accepts the information separators U+001C to U+001F.
        func isSpace(_ scalar: UnicodeScalar) -> Bool {
            scalar.properties.isWhitespace || (0x1C ... 0x1F).contains(scalar.value)
        }
        let scalars = string.unicodeScalars
        guard let start = scalars.firstIndex(where: { !isSpace($0) }),
            let end = scalars.lastIndex(where: { !isSpace($0) })
        else { return "" }
        return String(scalars[start ... end])
    }

    // MARK: - JSON

    /// Encodes a JSON value as Python's `json.dumps(value, ensure_ascii=False, indent=2)` does,
    /// with object keys sorted by code point and numbers written by ``pythonNumber(_:)``.
    static func pythonJSON(_ value: JSONValue, level: Int = 0) -> String {
        let indent = String(repeating: "  ", count: level + 1)
        let closingIndent = String(repeating: "  ", count: level)
        switch value {
        case .null:
            return "null"
        case .bool(let value):
            return value ? "true" : "false"
        case .number(let value):
            return pythonNumber(value)
        case .string(let value):
            return pythonString(value)
        case .array(let elements):
            guard !elements.isEmpty else { return "[]" }
            let items = elements.map { indent + pythonJSON($0, level: level + 1) }
            return "[\n" + items.joined(separator: ",\n") + "\n\(closingIndent)]"
        case .object(let members):
            guard !members.isEmpty else { return "{}" }
            let keys = members.keys.sorted { $0.unicodeScalars.lexicographicallyPrecedes($1.unicodeScalars) }
            let items = keys.map { key in
                "\(indent)\(pythonString(key)): \(pythonJSON(members[key]!, level: level + 1))"
            }
            return "{\n" + items.joined(separator: ",\n") + "\n\(closingIndent)}"
        }
    }

    /// Encodes a string as a JSON string literal,
    /// escaping only what Python's JSON encoder escapes with `ensure_ascii=False`.
    static func pythonString(_ string: String) -> String {
        var result = "\""
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            case "\u{08}": result += "\\b"
            case "\u{0C}": result += "\\f"
            case _ where scalar.value < 0x20:
                let hex = String(scalar.value, radix: 16)
                result += "\\u" + String(repeating: "0", count: 4 - hex.count) + hex
            default:
                result.unicodeScalars.append(scalar)
            }
        }
        return result + "\""
    }

    /// Encodes a number as Python writes it in JSON, as far as a `Double` allows.
    ///
    /// `JSONValue` keeps every number as a `Double`,
    /// so it can't tell a Python `int` from a `float` with the same value.
    /// A whole number below 2^53 in magnitude is written as an `int`:
    /// `27.0` as `27` and `-0.0` as `0`, where Python writes `27.0` and `-0.0` for a `float`.
    /// Any other number is written as Python's `repr` of a `float`, such as `89.5` or `1e-05`,
    /// so the integer 2^53 is written as `9007199254740992.0`.
    static func pythonNumber(_ value: Double) -> String {
        if value.isNaN { return "NaN" }
        if value.isInfinite { return value < 0 ? "-Infinity" : "Infinity" }
        if value == value.rounded(), abs(value) < 0x1p53 {
            return String(Int64(value))
        }

        // Swift and Python both print the shortest digits that round-trip.
        // Read those digits and the decimal exponent from Swift's text,
        // then lay them out as Python's `repr` does.
        let text = String(abs(value))
        let (mantissa, exponent): (Substring, Int) =
            if let e = text.firstIndex(where: { $0 == "e" || $0 == "E" }) {
                (text[..<e], Int(text[text.index(after: e)...])!)
            } else {
                (text[...], 0)
            }
        let point = mantissa.firstIndex(of: ".") ?? mantissa.endIndex
        let integerDigits = mantissa[..<point]
        let fractionDigits = point == mantissa.endIndex ? "" : mantissa[mantissa.index(after: point)...]
        var digits = String(integerDigits + fractionDigits)
        // The position of the decimal point, counted from the start of `digits`.
        var decimalPoint = integerDigits.count + exponent
        while digits.count > 1, digits.first == "0" {
            digits.removeFirst()
            decimalPoint -= 1
        }
        while digits.count > 1, digits.last == "0" {
            digits.removeLast()
        }

        // Each layout is interpolated from typed parts,
        // because Swift 6.2 is slow to type-check long chains of `+`.
        let sign: String = value < 0 ? "-" : ""
        if -4 < decimalPoint, decimalPoint <= 16 {
            if decimalPoint <= 0 {
                let zeros = String(repeating: "0", count: -decimalPoint)
                return "\(sign)0.\(zeros)\(digits)"
            }
            if decimalPoint >= digits.count {
                let zeros = String(repeating: "0", count: decimalPoint - digits.count)
                return "\(sign)\(digits)\(zeros).0"
            }
            let split = digits.index(digits.startIndex, offsetBy: decimalPoint)
            let integerPart = String(digits[..<split])
            let fractionPart = String(digits[split...])
            return "\(sign)\(integerPart).\(fractionPart)"
        }
        let leadingDigit = String(digits.first!)
        let fraction = String(digits.dropFirst())
        let fractionText: String = fraction.isEmpty ? "" : ".\(fraction)"
        let power = decimalPoint - 1
        let powerSign: String = power < 0 ? "-" : "+"
        let powerText: String = abs(power) < 10 ? "0\(abs(power))" : String(abs(power))
        return "\(sign)\(leadingDigit)\(fractionText)e\(powerSign)\(powerText)"
    }
}
