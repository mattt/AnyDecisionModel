# AnyDecisionModel

AnyDecisionModel is a Swift package for typed decisions:
yes-or-no probabilities, choices among options, and scores on ordinal scales.
The API follows the model-and-session pattern of
Apple's Foundation Models framework and
[AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel).

```swift
import AnyDecisionModel

enum Department: String, Choosable {
    case billing, technical, sales
}

let ticket = "I was charged twice for my order, and now the app logs me out when I try to ask for a refund."

let model = MLXDecisionModel(modelID: "mlx-community/Qwen3-4B-Instruct-2507-4bit")
let session = DecisionSession(model: model, state: .text(ticket))

let urgent = try await session.probability(of: "Does this convey urgency?")
let team = try await session.choice("Which team?", from: Department.self)
let anger = try await session.score(
    "How frustrated?",
    levels: ["Calm", "Frustrated", "Very angry"]
)
```

`urgent` is a `Double`.
`team` is a `Choice<Department>` with a value,
a probability for each option, and a confidence value.
`anger` is a `Score` with the expected level,
a probability for each level, and a confidence value.

Six backends are available:

- [x] [MLX](#mlx) runs a small language model on Apple silicon
  and reads each decision from next-token probabilities in one forward pass.
- [x] [MLX decision heads](#mlx-decision-heads) run a model with a trained decision head,
  such as Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef).
- [x] [Core AI](#core-ai) does the same as MLX with a Core AI model on macOS 27 and iOS 27.
- [x] [llama.cpp](#llamacpp) does the same as MLX with a GGUF model through llama.cpp.
- [x] [Jev](#jev) calls TypeSafe's Jev through the [System One API](https://docs.typesafe.ai/api),
  or any server that implements the same API.
- [x] [Ollama](#ollama) calls a System One model in
  [Ollama](https://docs.ollama.com/capabilities/decision),
  locally or in Ollama's cloud.

## Requirements

- Swift 6.2+
- macOS 26+ / iOS 26+ / Linux (core library, Jev, and Ollama backends only)
- Ollama: Ollama v0.35.0 or later (v0.40.5 or later for cloud models)
- MLX: Apple silicon and the Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain`)
- Core AI: Xcode 27 to build, macOS 27 or iOS 27 to run
- llama.cpp: a GGUF model file with a chat template that llama.cpp supports

## Installation

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnyDecisionModel", from: "0.5.0")
]
```

Without traits, the package has no third-party dependencies
and provides the core types and the Jev and Ollama backends.
Enable the `MLX`, `CoreAI`, or `Llama` trait for the local backends:

```swift
.package(
    url: "https://github.com/mattt/AnyDecisionModel",
    from: "0.5.0",
    traits: ["MLX", "CoreAI", "Llama"]
)
```

## Usage

### Sessions and questions

A `DecisionSession` holds one immutable state, which can be text or JSON.
Every question is about that state, and questions are independent:
an answer never becomes input to a later question.
Call `prewarm()` to load the model and process the state before the first question.

### Choices

To get a choice back as one of your own types, conform an enumeration to `Choosable`.
The cases become the options, in declaration order.
`String` raw values become the option names;
other types implement `optionName`.
Implement `optionDescription` to tell the model when to select each option:

```swift
enum Department: String, Choosable {
    case billing, technical, sales

    var optionDescription: String? {
        switch self {
        case .billing: "payments, charges, refunds, invoices"
        case .technical: "bugs, outages, errors, login problems"
        case .sales: "pricing questions, upgrades, new purchases"
        }
    }
}

let team = try await session.choice("Which team?", from: Department.self)

team.value                    // A Department, such as .billing
team.probability(of: .sales)  // The probability of one option
team.distribution             // Every option with its probability
```

When the options are known only at run time,
pass an array of `ChoiceOption` values instead and get a `Choice<String>` back.

### Batches

To ask several questions in one call, pass an array to `decide(_:)`:

```swift
let response = try await session.decide([
    .binary(instructions: "Does this convey urgency?"),
    .choice(
        instructions: "Which team should handle this ticket?",
        options: [
            ChoiceOption("billing", description: "payments, charges, refunds"),
            ChoiceOption("technical", description: "bugs, outages, login problems"),
        ]
    ),
    .score(instructions: "How frustrated?", levels: ["Calm", "Frustrated", "Very angry"]),
])

response.answers[1]   // The Answer to the choice question
response.usage        // Input and output token counts
response.diagnostics  // Local diagnostics, in the order of the questions
```

Answers come back in the order of the questions,
and the position of a question in the batch does not change its result.

## Backends

### MLX

`MLXDecisionModel` loads models through
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
and downloads them from the Hugging Face Hub.
Pass `directory:` to load a model from disk.

For each question, the model builds a prompt with the state, the question,
and a one-token label for each allowed answer,
then reads the probability of each label from a single forward pass.
Binary questions use "yes" and "no",
choices use A to Z, and scores use 0 to 9.
So a choice can have at most 26 options and a score at most 10 levels.

Language models tend to prefer an option because of its position in the list.
Set `rotationDebiasing: true` to ask each choice question once per rotation of the options
and average the probabilities.
A choice with _n_ options then costs _n_ prompt evaluations.

Prefix caching is off by default.
With `prefixCaching: true`, each session evaluates its state once
and each question runs on a copy of that cache.
Compare cached and uncached results for your model before enabling it.

> [!NOTE]
> The `MLX` trait pins mlx-swift-lm to 3.31.
> An app can't use this trait together with a package that requires mlx-swift-lm 3.32,
> such as AnyLanguageModel 1.0 with its `MLX` trait.

### MLX decision heads

`MLXHeadDecisionModel` runs a model that has its own trained decision head.
It reads each decision from the head rather than from next-token probabilities,
so it has no limit on the number of options or levels.
It uses the `MLX` trait.

```swift
let model = MLXHeadDecisionModel(
    modelID: "mlx-community/clef-flash-4bit",
    head: .jointSchema
)
```

The only head format is `.jointSchema`,
the joint schema head of Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) models.
[`mlx-community/clef-flash-4bit`](https://huggingface.co/mlx-community/clef-flash-4bit)
is a 6.2 GB download, and its model card recommends a Mac with at least 16 GB of memory.
Clef can read images and video, but this model reads text and JSON states only.
A prompt can have at most 16,384 tokens;
a longer prompt throws `DecisionError.unsupportedQuestion` before inference.
Prefix caching is on by default.

### Core AI

`CoreAIDecisionModel` runs a Core AI language bundle through
[coreai-kit](https://github.com/john-rocky/coreai-kit).
It uses the `CoreAI` trait and is available only on macOS 27 and iOS 27:

```swift
if #available(macOS 27, iOS 27, *) {
    let model = CoreAIDecisionModel(modelID: "qwen3-4b")
    let session = DecisionSession(model: model, state: .text(ticket))
    let urgent = try await session.probability(of: "Does this convey urgency?")
}
```

Pass a chat model's identifier from the coreai-kit catalog with `modelID:`;
`qwen3-0.6b` is a 352 MB download.
Pass `bundle:` to load a bundle that is already on disk.
The model uses the same prompt, labels, limits, and options as the MLX backend.

If a file imports both `AnyDecisionModel` and `CoreAIKit`,
write `AnyDecisionModel.DecisionError`,
because both modules declare a `DecisionError`.

### llama.cpp

`LlamaDecisionModel` runs a GGUF model through
[llama.swift](https://github.com/mattt/llama.swift),
which packages llama.cpp for Swift.
It uses the `Llama` trait:

```swift
let model = LlamaDecisionModel(modelPath: "/path/to/model.gguf")
let session = DecisionSession(model: model, state: .text(ticket))
let urgent = try await session.probability(of: "Does this convey urgency?")
```

Pass the path of a GGUF file with `modelPath:`.
The prompt uses the chat template embedded in the GGUF file,
applied by `llama_chat_apply_template`,
which supports a predefined list of templates rather than full Jinja.
The model uses the same prompt, labels, limits, and options as the MLX backend,
except prefix caching:
each session clears its context between questions,
so the state is evaluated again for every question.
Pass `gpuLayers: 0` for CPU-only execution,
and `closedThinkFallback: true` for reasoning models
whose templates open a thinking block.

### Jev

`JevDecisionModel` sends each batch to `POST /v1/systemone`.
The API key comes from `TYPESAFE_AI_API_KEY`
(or `TYPESAFE_API_KEY`), or you can pass it to the initializer.
The model retries HTTP 429, 503, and 529 with exponential backoff
and honors `Retry-After`;
pass `retryPolicy: .never` to disable retries.
Jev's probabilities and confidence values are returned unchanged.

The model works with any server that implements the System One API,
such as [Kev](https://github.com/jaredpalmer/kev)
or [Ollaya](https://github.com/ollaya-dev/ollaya).
Pass the server's URL, a placeholder API key, and its model name:

```swift
let model = JevDecisionModel(
    baseURL: URL(string: "http://127.0.0.1:11435")!,
    apiKey: "local",
    modelID: "laya"
)
```

The API key must not be empty, even when the server ignores it.

### Ollama

`OllamaDecisionModel` sends each batch to `POST /v1/systemone` on an Ollama server.
Pass the name of a decision model, such as
[Nimble](https://ollama.com/library/nimble):

```swift
let model = OllamaDecisionModel(modelID: "nimble")
let session = DecisionSession(model: model, state: .text(ticket))
let urgent = try await session.probability(of: "Does this convey urgency?")
```

Local requests need no API key.
For Ollama's cloud, set `OLLAMA_API_KEY` (or pass `apiKey:`)
and point `baseURL:` at `https://ollama.com`,
or add `:cloud` to the model name to go through a local server.
Pass `keepAlive:` to control how long the model stays loaded, in seconds.
Ollama's probabilities and confidence values are returned unchanged.

## Calibration and confidence

Raw probabilities from language models are often overconfident.
Local backends accept a `Calibration` that applies a temperature
to every answer and a bias to binary answers:

```swift
var model = MLXDecisionModel(modelID: "mlx-community/Qwen3-4B-Instruct-2507-4bit")
model.calibration = Calibration(temperature: 6.5, binaryBias: 1.0)
```

Fit a calibration on data that is separate from the data you evaluate.
The values above come from one historical fit and are an example,
not a recommended setting.

Local choice and score answers report confidence as
$1 - H(p) / \log n$,
the normalized complement of the
[entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))
of the answer probabilities.
This measures how concentrated the distribution is,
not the probability that the answer is correct.
The model can be confident and wrong,
so measure the error rate on your own data before you choose a threshold.
Jev and other System One servers compute confidence with their own formulas,
so do not compare confidence values between backends.

Each local answer also reports an allowed-answer mass in `DecisionSession.Diagnostics`:
the share of next-token probability that fell on the answer labels before normalization.
A low value means that the model wanted to reply with something else.

## Limitations

- Wire compatibility with Jev and Ollama is tested against fixtures, not live services.
- Calibration was fitted on two tasks.
  Whether it transfers to other tasks is untested.
- Do not rely on typed output alone to resist injected instructions.
- Inference is not optimized beyond prefix caching.

## Development

The repository uses [mise](https://mise.jdx.dev/) tasks:

| Task | Description |
| --- | --- |
| `mise run build` | Build the core library. |
| `mise run build:mlx` | Build with the MLX trait. |
| `mise run build:coreai` | Build with the CoreAI trait. |
| `mise run build:llama` | Build with the Llama trait. |
| `mise run test` | Run the tests without MLX. |
| `mise run test:mlx` | Run the MLX integration tests. Downloads Qwen3 4B. |
| `mise run test:mlx:head` | Run the MLX decision head integration tests. Downloads Clef-flash 4-bit. |
| `mise run test:coreai` | Run the Core AI integration tests. Downloads Qwen3 0.6B. |
| `mise run test:llama` | Run the tests with the Llama trait. Set `LLAMA_MODEL_PATH` to a GGUF file for inference tests. |

MLX tasks need Xcode's Swift 6.4 or later and the Metal compiler.
Core AI tasks need Xcode 27, and their integration tests run only on macOS 27.
Set `MLX_HEAD_MODEL_DIRECTORY` or `COREAI_MODEL_BUNDLE` to test a model from a local directory.

## Related projects

- [AnyDecisionModel-Examples](https://github.com/mattt/AnyDecisionModel-Examples):
  a ticket-routing command-line tool, Chess and Emoji Magnet macOS apps,
  and evaluation tools for this package.
- [Kev](https://github.com/jaredpalmer/kev):
  small Jev-like decision models with training code, evaluation data, and a System One server.
- [SemIf](https://github.com/TheoLeeCJ/SemIf) and
  [jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration):
  test sets for decision models, with published Jev results.

## Acknowledgments

The HTTP helpers, Linux request coordination, lock, test stub,
and model-loading patterns are adapted from
[AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel),
licensed under the Apache License, Version 2.0.
