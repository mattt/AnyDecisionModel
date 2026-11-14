import Foundation
import Testing

@testable import AnyDecisionModel

/// The d1 answers that mlx-vlm returns for single questions.
///
/// `scripts/d1-fixtures.py` writes the fixture from a pinned revision of `LiquidAI/d1-3B-MLX-8bit`.
struct D1Fixture: Decodable {
    struct Case: Decodable {
        var name: String
        var state: StateSpec
        var question: QuestionSpec
        var prompt: String
        var inputIDs: [Int]
        var readoutIDs: [[Int]]
        /// The probabilities from mlx-vlm's model as it loads, with float16 activations.
        var probabilities: [Double]
        /// The probabilities after mlx-vlm's half-precision parameters are converted to float32.
        var float32Probabilities: [Double]

        enum CodingKeys: String, CodingKey {
            case name, state, question, prompt, probabilities
            case inputIDs = "input_ids"
            case readoutIDs = "readout_ids"
            case float32Probabilities = "float32_probabilities"
        }
    }

    var repository: String
    var revision: String
    var cases: [Case]

    static func load() throws -> D1Fixture {
        try JSONDecoder().decode(
            D1Fixture.self,
            from: fixtureData("d1-3b-mlx-8bit.json", subdirectory: "Fixtures/D1")
        )
    }
}

/// d1-3B's beginning-of-sequence token.
private let bos = "<|startoftext|>"

/// The d1-3B token IDs of the answer forms in the fixture.
private let vocabulary: [String: Int] = [
    "yes": 11683, "Yes": 12447, "no": 2243, "No": 4547, "NO": 19598,
    "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46,
    " A": 334, " B": 378, " C": 340, " D": 388, " E": 407, " F": 410,
    "a": 73, "b": 74, "c": 75, "d": 76,
    " a": 267, " b": 293, " c": 280, " d": 277,
    "0": 24, "1": 25, "2": 26, "3": 27, "4": 28, "5": 29, "6": 30,
]

/// Tokenizes the answer forms as d1-3B does, and any other text as several tokens.
private func encode(_ text: String) -> [Int] {
    if let id = vocabulary[text] { return [id] }
    return [1_000_000, 1_000_001] + text.unicodeScalars.map { Int($0.value) }
}

@Suite("D1LFM2Prompt")
struct D1LFM2PromptTests {
    @Test func promptsAndReadoutTokensMatchTheReference() throws {
        let fixture = try D1Fixture.load()
        #expect(fixture.cases.count >= 5)
        for testCase in fixture.cases {
            let question = testCase.question.question
            let rendered = try D1LFM2Prompt.render(question, index: 0, encode: encode)
            let prompt = try D1LFM2Prompt.prompt(state: testCase.state.state, question: rendered, bos: bos)
            #expect(prompt == testCase.prompt, "\(testCase.name)")
            #expect(rendered.tokenGroups == testCase.readoutIDs, "\(testCase.name)")
        }
    }

    @Test func promptLayout() throws {
        let rendered = try D1LFM2Prompt.render(
            .binary(
                instructions: "Is it late?",
                criteria: BinaryCriteria(whenTrue: "after noon", whenFalse: "before noon")
            ),
            index: 0,
            encode: encode
        )
        let prompt = try D1LFM2Prompt.prompt(state: .text("It is 3 PM."), question: rendered, bos: bos)
        #expect(
            prompt
                == "<|startoftext|><|im_start|>user\nIt is 3 PM.\n\n\nQUESTION:\n"
                + "Is it late?\nYes: after noon\nNo: before noon\n\nReply with yes or no only."
                + "<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    @Test func choiceLinesUseDescriptionsOrNames() throws {
        let rendered = try D1LFM2Prompt.render(
            .choice(
                instructions: "Pick one.",
                options: [
                    ChoiceOption("track_order", description: "find an order"),
                    ChoiceOption("cancel_order"),
                    ChoiceOption("other", description: ""),
                ]
            ),
            index: 0,
            encode: encode
        )
        #expect(
            rendered.text
                == "Pick one.\n\nOptions:\nA find an order\nB cancel order\nC other\n\n"
                + "Reply with the option code only."
        )
        #expect(rendered.tokenGroups == [[41, 334], [42, 378], [43, 340]])
    }

    @Test func scoreLegend() throws {
        let rendered = try D1LFM2Prompt.render(
            .score(instructions: "How bad?", levels: ["Fine", "Bad", "Awful"]),
            index: 0,
            encode: encode
        )
        #expect(rendered.text == "How bad?\n\n0 Fine\n1 Bad\n2 Awful\n\nReply with a single digit 0-2 only.")
        #expect(rendered.tokenGroups == [[24], [25], [26]])
    }

    @Test func rotatedChoicesUseCodesForTheirPositions() throws {
        let options = [ChoiceOption("billing"), ChoiceOption("technical"), ChoiceOption("sales")]
        let positional = try D1LFM2Prompt.render(
            .choice(instructions: "Team?", options: options),
            optionOrder: [1, 2, 0],
            index: 0,
            encode: encode
        )
        #expect(positional.text.contains("A technical\nB sales\nC billing"))
        #expect(positional.tokenGroups == [[41, 334], [42, 378], [43, 340]])

        // Letter names are their own codes, so they keep their tokens in any order.
        let letters = [ChoiceOption("a", description: "one"), ChoiceOption("b", description: "two")]
        let native = try D1LFM2Prompt.render(
            .choice(instructions: "Which?", options: letters),
            optionOrder: [1, 0],
            index: 0,
            encode: encode
        )
        #expect(native.text.contains("b two\na one"))
        #expect(native.tokenGroups == [[74, 293], [73, 267]])
    }

    @Test func optionCodes() {
        #expect(D1LFM2Prompt.optionCodes(for: ["a", "b", "c"]) == ["a", "b", "c"])
        #expect(D1LFM2Prompt.optionCodes(for: [" x ", "Y\t"]) == ["x", "Y"])
        #expect(D1LFM2Prompt.optionCodes(for: ["é", "ß"]) == ["é", "ß"])
        #expect(D1LFM2Prompt.optionCodes(for: ["1", "2"]) == ["A", "B"])
        #expect(D1LFM2Prompt.optionCodes(for: ["a", "bb"]) == ["A", "B"])
        #expect(D1LFM2Prompt.optionCodes(for: ["billing", "sales"]) == ["A", "B"])
        let many = (0 ..< 27).map { "option \($0)" }
        #expect(D1LFM2Prompt.optionCodes(for: many).prefix(3) == ["00", "01", "02"])
        #expect(D1LFM2Prompt.optionCodes(for: many).last == "26")
    }

    /// Builds `prompt.py`'s `_FALLBACK_POOL` with plain loops.
    @Test func fallbackCodesMatchTheReferencePool() {
        var expected: [String] = []
        for scalar in UnicodeScalar("A").value ... UnicodeScalar("Z").value {
            expected.append(String(UnicodeScalar(scalar)!))
        }
        for number in 0 ..< 100 {
            expected.append(String(format: "%02d", number))
        }
        for scalar in UnicodeScalar("a").value ... UnicodeScalar("z").value {
            expected.append(String(UnicodeScalar(scalar)!))
        }
        for number in 0 ..< 200 {
            expected.append("#\(number)")
        }
        for first in UnicodeScalar("A").value ... UnicodeScalar("Z").value {
            for second in UnicodeScalar("A").value ... UnicodeScalar("Z").value {
                expected.append(String(UnicodeScalar(first)!) + String(UnicodeScalar(second)!))
            }
        }
        #expect(expected.count == 1_028)
        #expect(D1LFM2Prompt.fallbackCodes == expected)
    }

    @Test func aliasesFallBackWhenACodeIsNotOneTokenOrIsTaken() throws {
        // "A" is several tokens, and "b" and "B" share a token.
        let encode: (String) -> [Int] = { text in
            switch text {
            case "A": [1, 2]
            case "b", "B": [7]
            default: text.unicodeScalars.count == 1 ? [Int(text.unicodeScalars.first!.value)] : [0, 0]
            }
        }
        let aliases = try D1LFM2Prompt.aliases(for: ["xx", "yy", "zz"], index: 0, encode: encode)
        #expect(aliases.map(\.code) == ["B", "C", "D"])  // "A" fails, so the first option falls back to "B".

        let letters = try D1LFM2Prompt.aliases(for: ["b", "B"], index: 0, encode: encode)
        #expect(letters.map(\.code) == ["b", "C"])
        #expect(letters.map(\.token) == [7, 67])
    }

    @Test func aliasesThrowWhenNoCodeIsLeft() {
        expectDecisionError(
            DecisionError.unsupportedQuestion(
                index: 2,
                reason: "No single-token option code is left for 2 options."
            )
        ) {
            try D1LFM2Prompt.aliases(for: ["xx", "yy"], index: 2, encode: { _ in [1] })
        }
    }

    @Test func questionsThrowWithoutSingleTokenAnswers() {
        #expect(throws: DecisionError.self) {
            try D1LFM2Prompt.render(.binary(instructions: "Yes?"), index: 0, encode: { _ in [1, 2] })
        }
        expectDecisionError(
            DecisionError.unsupportedQuestion(
                index: 1,
                reason: "The score level 0 has no single-token digit."
            )
        ) {
            try D1LFM2Prompt.render(
                .score(instructions: "How?", levels: ["a", "b"]),
                index: 1,
                encode: { _ in [1, 2] }
            )
        }
    }

    @Test func readoutTakesTheBestTokenOfEachAnswer() {
        let (scores, mass) = D1LFM2Prompt.scores([[-1, -3], [-2], [-4, -0.5]])
        #expect(scores == [-1, -2, -0.5])
        let expectedMass = [-1, -3, -2, -4, -0.5].reduce(0) { $0 + exp($1) }
        #expect(abs(mass - min(expectedMass, 1)) < 1e-12)

        // The readout is a softmax over the scores, which the identity calibration gives.
        let probabilities = Calibration.identity.probabilities(logMasses: [-1, -2], binary: false)
        let expected = [exp(-1.0), exp(-2.0)].map { $0 / (exp(-1.0) + exp(-2.0)) }
        #expect(zip(probabilities, expected).allSatisfy { abs($0 - $1) < 1e-12 })
        let binary = Calibration.identity.probabilities(logMasses: [-1, -2], binary: true)
        #expect(zip(binary, expected).allSatisfy { abs($0 - $1) < 1e-12 })
    }

    /// A top-level null is d1's state of `None`, which has no state block and no `QUESTION:` line.
    /// The expected prompts come from `prompt.py`'s `render` with `bos="<|startoftext|>"`.
    @Test func nullStatesMatchTheReference() throws {
        let binary = try D1LFM2Prompt.render(.binary(instructions: "Is it late?"), index: 0, encode: encode)
        #expect(
            try D1LFM2Prompt.prompt(state: .json(.null), question: binary, bos: bos)
                == "<|startoftext|><|im_start|>user\nIs it late?\n\nReply with yes or no only."
                + "<|im_end|>\n<|im_start|>assistant\n"
        )
        #expect(
            try D1LFM2Prompt.prompt(state: .json(["a": .null]), question: binary, bos: bos)
                == "<|startoftext|><|im_start|>user\n{\n  \"a\": null\n}\n\n\nQUESTION:\n"
                + "Is it late?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n"
        )
        let score = try D1LFM2Prompt.render(.score(instructions: "How?", levels: ["a", "b"]), index: 0, encode: encode)
        #expect(
            try D1LFM2Prompt.prompt(state: .json([.null]), question: score, bos: bos)
                == "<|startoftext|><|im_start|>user\n[\n  null\n]\n\n\nQUESTION:\n"
                + "How?\n\n0 a\n1 b\n\nReply with a single digit 0-1 only.<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    @Test func stateBlocks() {
        #expect(D1LFM2Prompt.stateBlock(.string("Hello\n")) == "Hello\n\n\n")
        #expect(D1LFM2Prompt.stateBlock(.object([:])) == "{}\n\n")
        #expect(D1LFM2Prompt.stateBlock(.array([])) == "[]\n\n")
        #expect(D1LFM2Prompt.stateBlock(.null) == "null\n\n")
        #expect(
            D1LFM2Prompt.stateBlock(["b": [1, ["c": []]], "a": "x/y", "é": true])
                == "{\n  \"a\": \"x/y\",\n  \"b\": [\n    1,\n    {\n      \"c\": []\n    }\n  ],\n  \"é\": true\n}\n\n"
        )
    }

    @Test func stringsEscapeLikePython() {
        #expect(D1LFM2Prompt.pythonString("Zoë \"quoted\" \\ /") == #""Zoë \"quoted\" \\ /""#)
        #expect(
            D1LFM2Prompt.pythonString("\n\r\t\u{08}\u{0C}\u{01}\u{1F}\u{7F}") == #""\n\r\t\b\f\u0001\u001f\#u{7F}""#
        )
        #expect(D1LFM2Prompt.pythonString("\u{2028}😀") == "\"\u{2028}😀\"")
    }

    @Test(arguments: [
        (27.0, "27"), (-3.0, "-3"), (0.0, "0"), (-0.0, "0"), (89.5, "89.5"), (-0.25, "-0.25"),
        (0.1, "0.1"), (0.0001, "0.0001"), (0.00001, "1e-05"), (1.5e-7, "1.5e-07"),
        (123456789.125, "123456789.125"), (1e16, "1e+16"), (1.5e300, "1.5e+300"),
        (9007199254740992.0, "9007199254740992.0"), (4503599627370495.5, "4503599627370495.5"),
        (Double.nan, "NaN"), (Double.infinity, "Infinity"), (-Double.infinity, "-Infinity"),
    ])
    func numbersMatchPython(value: Double, expected: String) {
        #expect(D1LFM2Prompt.pythonNumber(value) == expected)
    }

    @Test func detectsD1Configurations() {
        func configuration(_ object: [String: Any]) -> Data {
            try! JSONSerialization.data(withJSONObject: object)
        }
        let d1: [String: Any] = ["model_type": "lfm2_vl", "auto_map": ["AutoModel": "modeling_d1.D1Model"]]
        #expect(D1LFM2Prompt.isD1LFM2Configuration(configuration(d1)))
        #expect(!D1LFM2Prompt.isD1LFM2Configuration(configuration(["model_type": "lfm2_vl"])))
        #expect(
            !D1LFM2Prompt.isD1LFM2Configuration(
                configuration(["model_type": "lfm3_vl", "auto_map": ["AutoModel": "modeling_d1.D1Model"]])
            )
        )
        #expect(
            !D1LFM2Prompt.isD1LFM2Configuration(
                configuration(["model_type": "lfm2_vl", "auto_map": ["AutoModel": "modeling.OtherD1Model"]])
            )
        )
        #expect(!D1LFM2Prompt.isD1LFM2Configuration(configuration(["model_type": "qwen3"])))
        #expect(!D1LFM2Prompt.isD1LFM2Configuration(Data("not json".utf8)))
    }
}
