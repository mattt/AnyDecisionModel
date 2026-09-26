# AnyDecisionModel

AnyDecisionModel is a Swift package for typed decisions:
yes-or-no probabilities, choices among options, and scores on ordinal scales.
It has two backends.
`MLXDecisionModel` runs a small language model on Apple silicon
and reads each decision from next-token probabilities in one forward pass, with no text generation.
`JevDecisionModel` calls TypeSafe's Jev through the [System One API](https://docs.typesafe.ai/api),
or any server that implements the same API.

The API follows the model-and-session pattern of Apple's Foundation Models framework
and of [AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel).

```swift
import AnyDecisionModel

enum Department: String, Choosable {
    case billing, technical, sales
}

let ticket = "I was charged twice for my order, and now the app logs me out when I try to ask for a refund."

var model = MLXDecisionModel(
    modelID: "mlx-community/Qwen3-4B-Instruct-2507-4bit"
)
// Raw probabilities from language models are often overconfident; a calibration softens them.
model.calibration = Calibration(temperature: 6.5, binaryBias: 1.0)
let session = DecisionSession(model: model, state: .text(ticket))

let urgent = try await session.probability(of: "Does this convey urgency?")
let team = try await session.choice("Which team?", from: Department.self)
let anger = try await session.score(
    "How frustrated?",
    levels: ["Calm", "Frustrated", "Very angry"]
)
```

`urgent` is a `Double`.
`team` is a `Choice<Department>` with a value, a probability for each option, and a confidence value.
`anger` is a `Score` with the expected level, a probability for each level, and a confidence value.

The ticket describes a billing problem and a technical problem.
With the [calibration](#probabilities-calibration-and-confidence) above,
`team.distribution` is 0.75 for billing, 0.18 for technical, and 0.07 for sales,
so your code can route the ticket to billing and also flag it for the technical team.

## Requirements

- Swift 6.2 or later.
- macOS 14 or later, or iOS 17 or later.
  The core library and the Jev backend also build on Linux.
- For MLX: Apple silicon and the Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain`).

## Installation

Add the package and enable the `MLX` trait if you want the local backend:

```swift
.package(
    url: "https://github.com/mattt/AnyDecisionModel",
    from: "0.3.0",
    traits: ["MLX"]
)
```

Without the trait, the package has no third-party dependencies at run time
and provides the core types and the Jev backend.

## Sessions and questions

A `DecisionSession` holds one immutable state, which can be text or JSON.
Every question is about that state, and questions are independent:
an answer never becomes input to a later question.
Call `prewarm()` to load the model and process the state before the first question.

To get a choice back as one of your own types, conform an enumeration to `Choosable`.
The protocol builds on `CaseIterable`, so the cases of the enumeration become the options,
in declaration order.
An enumeration with `String` raw values uses each raw value as the option name;
other types implement `optionName`, which must be unique for each case.
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

team.value                    // A Department, such as .billing.
team.probability(of: .sales)  // The probability of one option.
team.distribution             // Every option with its probability.
```

The model sees the option names and descriptions, not the Swift cases.
The session maps the answer back to a case,
so `team.value` is a `Department` that you can use in an exhaustive `switch`.
When the options are known only at run time,
pass an array of `ChoiceOption` values to `choice(_:from:)` and get a `Choice<String>` back.

The typed helpers ask one question at a time.
To ask several questions together, pass an array to `decide(_:)`:

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

response.modelID      // The model identifier.
response.usage        // Input and output token counts.
response.answers[1]   // The Answer to the choice question.
response.diagnostics  // Local diagnostics, in the order of the questions.
```

Answers come back in the order of the questions.
The position of a question in the batch does not change its result.
To ask one question and get its `Answer`, pass a single `Question` to `decide(_:)`.
Instructions, criteria, option descriptions, and score levels are strings.
A state can be text or structured JSON (`JSONValue`).

Binary answers have no confidence value;
`Answer.binary(probability:)` holds the probability as a `Double`.
Errors are `DecisionError` values; cancellation is reported as `CancellationError`.

## Backends

### MLX

`MLXDecisionModel` loads models through
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) and
[swift-huggingface](https://github.com/huggingface/swift-huggingface).
It downloads and caches files from the Hugging Face Hub,
or loads them from a local directory with `directory:`.
Pass a `HubClient` with `hub:` to control downloads.

For each question, the model builds a prompt that contains the state, the question,
and a short label for each allowed answer.
It then runs one forward pass and reads the probability of every possible next token.
The probability of an answer is the total probability of the tokens that spell its label,
such as "yes", "Yes", and " yes",
divided by the total for all the allowed answers.
Binary questions use the labels "yes" and "no".
Choice options are labeled A to Z, so a choice can have at most 26 options.
Score levels are labeled 0 to 9, so a score can have at most 10 levels.
The model checks every label's token mapping before it runs inference.
The model passes `enable_thinking: false` to the chat template.

> [!NOTE]
> For reasoning models whose templates ignore this flag,
> set `closedThinkFallback: true`.
> This appends `<think>\n\n</think>\n\n` when the tokenizer has both markers
> and the decoded prompt tail does not contain `</think>`.
> The option is off by default.
> Enable it only for models with a thinking mode;
> the default Qwen3 Instruct model retains these markers but has no thinking mode.

Sessions share loaded weights.
Prefix caching is off by default.
mlx-swift 0.31.6 and earlier include MLX 0.31.1,
whose Metal attention kernel applies the causal mask incorrectly
for some split-prefill shapes.
MLX 0.31.2 fixes this
([ml-explore/mlx#3271](https://github.com/ml-explore/mlx/pull/3271)).
With `prefixCaching: true`, each session keeps a cache for the state prefix,
and each question runs on a copy of that cache.
For models whose layers use plain key-value caches, such as Qwen3,
the model passes an explicit causal mask array instead of the affected symbolic mask.
Check cached and uncached results for your model and workload before enabling it.

When `decide(_:)` receives several questions or a choice with rotation debiasing,
the model evaluates their prompts in batches,
with up to `maximumBatchSize` prompts in one forward pass.
The default is 16; set it to 1 to evaluate each prompt separately.
Batching applies to models whose layers use plain key-value caches;
other models evaluate each prompt separately.
Only the session's prefix cache shares evaluation across prompts.
The model does not split a prompt at a prefix shared with other questions:
that split can change probabilities through rounding.

Language models tend to prefer an option because of its position in the list,
for example the option labeled A.
To reduce this position bias, set `rotationDebiasing: true`.
The model then asks each choice question once for every rotation of the option order,
so that each option appears in each position once,
and averages the probabilities.
A choice with _n_ options requires _n_ prompt evaluations.
Several of these evaluations can share a batched forward pass.
This setting is off by default.

Each answer's `DecisionSession.Diagnostics` reports the allowed-answer mass:
the share of next-token probability that falls on the allowed answer tokens before normalization.
A low value means that the model wanted to reply with something else.
This value is not a confidence value and not a probability of being correct.

### Jev

`JevDecisionModel` sends each batch to `POST /v1/systemone`.
It uses Foundation's `URLSession` and has no third-party dependencies.
The Jev wire format is private to this model;
binary questions go out under Jev's wire name, `noul`.
The API key comes from `TYPESAFE_AI_API_KEY`, or from `TYPESAFE_API_KEY` if the first is not set;
you can also pass it to the initializer.
The key is sent only in the `Authorization` header
and does not appear in descriptions or errors.
The model retries HTTP 429, 503, and 529 with bounded exponential backoff
and honors `Retry-After`.
The default policy retries five times, starting at 0.5 seconds and doubling
each delay up to 30 seconds:

```swift
let model = JevDecisionModel(
    retryPolicy: .init(
        strategy: .exponential(base: 0.5, multiplier: 2, jitter: 0),
        timeout: nil,
        maximumInterval: 30,
        maximumRetries: 5,
        retryableStatusCodes: [429, 503, 529]
    )
)
```

Use `.never` to disable retries.
Set `timeout`, `maximumInterval`, or `maximumRetries` to `nil` to disable that limit.
Jev's probabilities and confidence values are returned unchanged.
Jev accepts at most 255 options for each choice.

The model works with any server that implements the System One API.
[Kev](https://github.com/jaredpalmer/kev) is a family of small open-weights decision models
that you can train and serve on your own machine.
To use a local Kev server, pass its URL, a placeholder API key, and its model name:

```swift
let model = JevDecisionModel(
    baseURL: URL(string: "http://127.0.0.1:8009")!,
    apiKey: "local",
    modelID: "kev-latest"
)
```

The API key must not be empty, even when the server ignores it.

## Probabilities, calibration, and confidence

Local probabilities are raw by default.
Raw probabilities from language models are often overconfident.
A `Calibration` applies a temperature to the log-probabilities of every primitive
and a bias to binary log-odds only:

```swift
var model = MLXDecisionModel()
model.calibration = Calibration(temperature: 6.5, binaryBias: 1.0)
```

Fit a calibration on data that is separate from the data you evaluate.
The values above come from a historical fit and are an example,
not a recommended setting for other tasks.
See the companion's
[calibration results](https://github.com/mattt/AnyDecisionModel-Examples/blob/main/docs/historical-results.md)
for the datasets, sample sizes, and held-out measurements.

Local choice and score answers report confidence as the normalized entropy complement,
$1 - H(p) / \log n$,
where $H(p) = -\sum_i p_i \log p_i$ is the [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))
of the answer probabilities
and $n$ is the number of options.
A question that has one option has confidence 1.
This value measures how concentrated the distribution is.
It is not the probability that the answer is correct.
Jev computes its confidence remotely and does not publish the formula.
Other System One servers use their own formulas,
so do not compare confidence values between backends.

## Examples and evaluations

The private companion repository,
[AnyDecisionModel-Examples](https://github.com/mattt/AnyDecisionModel-Examples),
contains the ticket-routing command-line example and evaluation tools.
The example routes tickets with a typed choice
and sends tickets below a confidence threshold to a person.
It never switches to a cloud model on its own.

See the companion's
[evaluation methodology](https://github.com/mattt/AnyDecisionModel-Examples/blob/main/docs/evaluation.md)
for commands, datasets, fixture provenance, and report formats.
The [historical results](https://github.com/mattt/AnyDecisionModel-Examples/blob/main/docs/historical-results.md)
preserve the measurements from 2026-09-19 with their conditions and limitations.
Access requires permission to view the private repository.

## Limitations

- Wire compatibility with Jev is tested against fixtures, not against the live service.
  Live Jev benchmarking and publication are deferred
  until the applicable account terms are clarified
  ([TypeSafe agreement](https://typesafe.ai/legal/mca)).
  Wire compatibility says nothing about model quality.
- Calibration was fitted on two tasks. Whether it transfers to other tasks is untested.
- Local confidence measures concentration. The model can be confident and wrong.
  Measure the error rate on your own data before you choose a confidence threshold.
- Do not rely on typed output alone to resist injected instructions.
  See the [historical robustness results](https://github.com/mattt/AnyDecisionModel-Examples/blob/main/docs/historical-results.md)
  for the observed failures and fixture limitations.
- The MLX backend limits choices to 26 options and scores to 10 levels.
  These limits come from its single-token answer labels.
  Other models report their own limits.
- Inference is not optimized beyond prefix caching.
  Further optimization waits until profiling justifies it.

## Not yet implemented

- A native MLX port of Laya (ModernBERT encoder, type embeddings, transformer head,
  option scorer, preprocessing, and calibration),
  validated against reference fixtures from a pinned upstream checkpoint.
  For background on this kind of model,
  see [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked).
- A DeBERTa zero-shot baseline, which waits for a native implementation.
- A chess example. Unrestricted chess play needs more choice labels than A to Z.

## Development

The repository uses [mise](https://mise.jdx.dev/) tasks:

| Task | Description |
| --- | --- |
| `mise run setup` | Resolve dependencies for the library. |
| `mise run build` | Build the core library. |
| `mise run build:mlx` | Build with the MLX trait. |
| `mise run build:ios` | Compile the core library for iOS. |
| `mise run build:linux` | Build and test the core library in a Linux container. |
| `mise run test` | Run the library tests without MLX. |
| `mise run test:mlx` | Run the tests with MLX integration tests. Downloads Qwen3 4B. |

Setup is safe to repeat and does not download models or datasets.
The hidden tasks in `mise.toml` select Xcode's Swift
on macOS and `swift` from `PATH` on other platforms.
This keeps the compiler and selected Apple SDK compatible.

MLX tasks use Xcode's Swift 6.4 or later
and put the Metal compiler on `PATH`.
SwiftPM compiles MLX's Metal shaders with the Swift Build engine,
which is the default build system in Swift 6.4.
MLX integration tests run only when `ENABLE_MLX_TESTS` is set.

## Related projects

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
