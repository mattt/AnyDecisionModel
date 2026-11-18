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

Decision models have no single API.
The package supports two wire formats and two local methods.

**System One** is the wire format of TypeSafe's
[System One API](https://docs.typesafe.ai/api):
a `POST /v1/systemone` request with a `state`
and a map of named `noul` (yes-or-no), `choice`, and `score` questions.
Three backends send it:

- [Jev](#jev) calls TypeSafe's Jev.
- [Ollama](#ollama) calls a decision model in
  [Ollama](https://docs.ollama.com/capabilities/decision),
  locally or in Ollama's cloud.
- [System One](#system-one) calls any other server that implements the format,
  with a configurable path, headers, limits, and request fields.

System One also has `box` questions, which locate a region of an image.
The package does not send them.

**Gemini Decisions** is the format of Google's
[Gemini Decisions API](https://aistudio.google.com/docs/decisions-api).
The [Gemini](#gemini) backend sends it,
and is the only backend that reads images, audio, video, and documents.

**Next-token scoring** runs a small language model on the device
and reads each decision from the next-token probabilities of the allowed answers.
[MLX](#mlx) does this on Apple silicon,
also with Liquid AI's [d1](#liquid-ai-d1) decision model,
[Core AI](#core-ai) with a Core AI model on macOS 27 and iOS 27,
and [llama.cpp](#llamacpp) with a GGUF model.

**Decision heads** are trained output layers that a model carries for decisions.
[MLX decision heads](#mlx-decision-heads) run such a model,
such as Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef).

Other providers document System One routes:
[vLLM](https://docs.vllm.ai/en/v0.32.0/serving/online_serving/structured_decisions/)
and [SGLang](https://lmsysorg.mintlify.app/docs/supported-models/decision_models.md)
at `/v1/systemone`,
[Liquid](https://www.liquid.ai/en/api) at `/decisions/v1/systemone`,
and [OpenRouter](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request.md)
at `/api/alpha/decisions`, which its documentation marks as alpha.
The package is not tested against them.
To try one, set the path and limits of the [System One](#system-one) backend.

The package has no backend for two other interfaces:
OpenAI's [Decisions API](https://developers.openai.com/api/reference/go/resources/decisions/index.md),
which has its own format with `predicate`, `choice`, and `score` questions,
and Google's on-device
[MediaPipe Decision Maker](https://developers.google.com/edge/mediapipe/solutions/decision/decision_maker).

## Requirements

- Swift 6.2+
- macOS 26+ / iOS 26+ / Linux (core library, Jev, Ollama, System One, and Gemini backends only)
- Ollama: Ollama v0.35.0 or later (v0.40.5 or later for cloud models)
- MLX: Apple silicon and the Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain`)
- Core AI: Xcode 27 to build, macOS 27 or iOS 27 to run
- llama.cpp: a GGUF model file with a chat template that llama.cpp supports

## Installation

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnyDecisionModel", from: "0.10.0")
]
```

Without traits, the package depends only on Apple's
[swift-http-types](https://github.com/apple/swift-http-types),
which `DecisionError` uses for HTTP responses,
and provides the core types and the Jev, Ollama, System One, and Gemini backends.
Enable the `MLX`, `CoreAI`, or `Llama` trait for the local backends:

```swift
.package(
    url: "https://github.com/mattt/AnyDecisionModel",
    from: "0.10.0",
    traits: ["MLX", "CoreAI", "Llama"]
)
```

## Usage

### Sessions and questions

A `DecisionSession` holds one immutable state,
which can be text, JSON, or a list of segments of text and attachments.
Attachments are images, audio, video, and documents,
from data or a URL, each with a MIME type.
The Gemini backend reads attachments of every kind,
and the System One backend can send images
if you set its [image encoding](#images).
The other backends throw `DecisionError.unsupportedState` for a state with attachments,
and a session throws that error before the model does any work.
Every question is about that state, and questions are independent:
an answer never becomes input to a later question.
Call `prewarm()` to load the model and process the state before the first question.

### Choices

To get a choice back as one of your own types, conform an enumeration to `Choosable`.
The cases become the options, in declaration order.
A choice needs at least two options with unique, non-empty names,
so the enumeration needs at least two cases.
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

A choice with fewer than two options decides nothing,
so a session throws `DecisionError.invalidQuestion` for it
before the model does any work.
A score also needs at least two levels.
Version 0.10 and earlier accepted a choice with one option.
To migrate, remove a question that has only one possible answer,
or add the alternatives that the model should consider.

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

A model's `capabilities` declare its limits,
such as the kinds of questions it answers,
the number of questions, options, levels, and attachments,
and the kinds of attachments it reads.
The session checks each batch against them before the model does any work,
and `capabilities.validate(_:about:)` does the same check without a session.

### Request options

A session sends a `DecisionOptions` value with each request.
Pass options to the session's initializer to use them for every request,
or pass `options:` to `decide`, `probability(of:)`, `choice`, or `score` for one call.
The options of a call replace the session's options for that call;
the two values are not merged.

```swift
var options = DecisionOptions(cacheKey: "triage-v2", userID: hashedAccountID)
options[custom: SystemOneDecisionModel.self] = .init(
    extraBody: ["session_id": "triage-7"]
)
let session = DecisionSession(model: model, state: .text(ticket), options: options)

let urgent = try await session.probability(of: "Does this convey urgency?")
let team = try await session.choice("Which team?", from: Department.self, options: DecisionOptions())
```

`cacheKey` and `userID` are hints that a backend can ignore without a change to an answer.
No backend in this package reads them yet.
An option that changes answers or routing is specific to one type of model.
Set it with the `options[custom:]` subscript, keyed by the model type.
Only an executor of that type reads it,
and the subscript returns `nil` for a type that has no value.
A model without custom options has `Never` as its `CustomOptions` type.

### Errors

A model or session throws a `DecisionError`
when a question, state, or option is invalid, or when the model or its provider fails.
The error follows AnyLanguageModel's `HTTPTransportError`:
its `kind` names what went wrong,
and `message` describes it.
Other fields are set when they apply:
`questionIndex` for an error about one question,
`reason` for a provider's code such as `safety`,
`response` and `body` when the provider answered with an error status,
and `underlyingError` for the error that caused this one.
More kinds can be added, so a `switch` over them needs a `default`.

```swift
do {
    let urgent = try await session.probability(of: "Does this convey urgency?")
} catch let error as DecisionError where error.kind == .rateLimited {
    try await Task.sleep(for: error.retryAfter ?? .seconds(1))
} catch let error as DecisionError {
    print(error.kind, error.message)
}
```

The remote backends map HTTP status codes in the same way:
401 and 403 to `authenticationFailed`, 402 to `quotaExceeded`,
408 and 504 to `timedOut`, 429 to `rateLimited`,
and every other status to `httpError`.
`response` is an `HTTPResponse` from swift-http-types.
`retryAfter` reads the response's `Retry-After` header, in seconds or as an HTTP date.
`isRetryable` is `true` for `rateLimited`, `timedOut`, and HTTP 503 and 529;
a model that retries reads its `RetryPolicy` instead,
whose `retryableStatusCodes` decide.
A request that fails on the way, such as one the network drops,
throws the URL session's own `URLError`,
and so does a request that is cancelled while the URL session sends it.
A model that is cancelled at any other point throws `CancellationError`.
An error keeps some credentials out of its response and body.
The values of the response headers
`Authorization`, `Proxy-Authorization`, `X-Api-Key`, and `X-Goog-Api-Key` are replaced.
For a response with an error status,
the remote backends also replace each value that the request sent in one of those headers,
and the token after a scheme such as `Bearer`,
wherever it appears in the body and so in the message.
Nothing else is redacted:
other response headers,
text from a successful response, such as a provider's reason for blocking the input,
and the fields of an error that you create are kept as they are.

#### Migrating from version 0.10

Through version 0.10, `DecisionError` is an enum,
and version 0.11 makes it a struct.
Code that creates an error,
such as `throw DecisionError.invalidQuestion(index: 0, reason: "Empty.")`,
compiles unchanged, with the same message.
Other code needs these changes:

- `DecisionError` is not `Equatable` or `Hashable`,
  because `underlyingError` can hold any error.
  `#expect(throws: DecisionError.missingCredentials)` and `error == .missingCredentials`
  become a comparison of `error.kind`, and of `error.message` if needed.
- `catch DecisionError.requestFailed(let status, let detail, let headers)` becomes
  `catch let error as DecisionError where error.response != nil`,
  with `error.response?.status.code`, `error.body`, and `error.response?.headerFields`.
  The kind of an HTTP error is one of
  `authenticationFailed`, `quotaExceeded`, `timedOut`, `rateLimited`, and `httpError`.
- `catch DecisionError.invalidResponse(let detail)`,
  and the same pattern for the other cases, becomes
  `catch let error as DecisionError where error.kind == .invalidResponse`,
  with `error.message`.
  A response body that can't be decoded now has the kind `decodingError`,
  and `inputBlocked(reason:message:)` creates the kind `guardrailViolation`.
- `case .unsupportedQuestion(let index, _)` becomes
  a check of `error.kind` and `error.questionIndex`.
- `switch error` over the cases becomes `switch error.kind`, with a `default` case.
- `headers["retry-after"]` becomes `error.retryAfter`, a `Duration`.
- A cancelled request to a remote model can throw `URLError` with the code `cancelled`,
  where version 0.10 threw `CancellationError`.
- `catch let error as EncodingError` for a request body that a remote model can't encode
  becomes a check for the kind `invalidRequest`,
  with the `EncodingError` in `error.underlyingError`.
- `catch let error as SystemOneDecisionModel.ConfigurationError` for a request's options
  becomes a check for the kind `unsupportedOption`.
  The `SystemOneDecisionModel` initializer still throws `ConfigurationError`.
- `DecisionKind` and `DecisionCapabilities.ProbabilitySource` are now structs
  with static members, such as `.binary` and `.native`.
  Those spellings, `rawValue`, and `init?(rawValue:)` still compile,
  and both still encode as strings.
  `init?(rawValue:)` now accepts a name that the package does not know,
  as decoding does, where the enum returned `nil`.
  A `switch` over either needs a `default` case in place of `@unknown default`.

## Backends

### MLX

`MLXDecisionModel` loads models through
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
and downloads them from the Hugging Face Hub.
Pass `directory:` to load a model from disk.

For each question, the model builds a prompt with the state, the question,
and a one-token label for each allowed answer,
then reads the probability of each label as the next token after the prompt.
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

The model computes in float32.
When it loads a model, it converts the half-precision parameters to float32,
so that prefix caching and batching don't change the rounding of results.
This makes Qwen3 4B 4-bit 11% larger
and doubles the size of an unquantized half-precision model.

#### Liquid AI d1

`MLXDecisionModel` also runs Liquid AI's
[d1-3B](https://huggingface.co/LiquidAI/d1-3B) decision model,
which is built on LFM2-VL
and reads answers from next-token probabilities with its own prompt:

```swift
let model = MLXDecisionModel(modelID: "LiquidAI/d1-3B-MLX-8bit")
```

[`LiquidAI/d1-3B-MLX-8bit`](https://huggingface.co/LiquidAI/d1-3B-MLX-8bit)
is a 3.72 GB download.
The model reads `config.json` when it loads,
and uses d1's prompt and readout
when `model_type` is `lfm2_vl` and `auto_map.AutoModel` ends in `.D1Model`,
as mlx-vlm and llama.cpp do.
Pass `promptFormat: .d1LFM2` to use them for a model without that entry,
or `promptFormat: .standard` to use the package's prompt.
Other models keep the package's prompt.
A model whose `model_type` is `lfm2_vl` loads through mlx-swift-lm's `MLXVLM` library
with or without the entry;
every other model loads through `MLXLLM`.

With the `.d1LFM2` format, the model follows `prompt.py` in the d1-3B repository:
no system prompt,
the state and the question in d1's own layout,
d1's option codes,
and for each answer the largest log-probability among its token variants,
followed by a softmax.
`systemPrompt` and `closedThinkFallback` don't apply.
A JSON state can differ from what `prompt.py` writes in two ways,
because of how `JSONValue` stores values:

- d1 orders the keys of a JSON object as given,
  but `JSONValue` objects have no key order,
  so the model writes a JSON state with sorted keys.
- `JSONValue` keeps every number as a `Double`,
  so the model writes a whole number below 2^53 as an integer,
  such as `27.0` as `27` and `-0.0` as `0`.
  Python's JSON encoder keeps a `float` such as `27.0` as it is,
  and writes the integer 2^53 without the `.0` that the model adds.

A top-level JSON null is d1's empty state:
the prompt has the question only, with no state and no `QUESTION:` line.
Confidence for choices and scores is the entropy measure of the other local models,
not d1's probability of the top answer.

For nine questions over text and JSON states,
including a choice with six options, scores with five and seven levels,
and options without descriptions,
the probabilities match mlx-vlm 0.8.0 on the same 8-bit weights
within 3e-6 when mlx-vlm also computes in float32,
and within 0.001 with mlx-vlm's float16 computation.
The JSON states in these cases have sorted keys,
and their whole numbers are integers,
so the two differences above don't apply to them.
Prefix caching gives the same answers.
These results need swift-transformers 1.4.0 or later:
earlier versions split some runs of newlines differently from Hugging Face's tokenizers,
which changes d1's prompt tokens and moved one probability by 0.013.
d1's layers don't all use plain key-value caches,
so the model evaluates d1 prompts one at a time rather than in batches.

The d1 support has these limits:

- d1 reads images, but this model reads text and JSON states only.
  A state with attachments throws `DecisionError.unsupportedState`.
- `prompt.py` codes choices with more than 26 options as 00, 01, and so on.
  The model allows at most 26 options, so it doesn't use those codes.
- d1's question types are `noul`, `choice`, and `score`,
  which match binary, choice, and score questions.
  `prompt.py` also writes a state of images alone, with no text,
  which this model can't send.
- The model loads d1 through mlx-swift-lm's `MLXVLM` library,
  so the vision tower stays in memory, unused.
- Only the 8-bit export was tested.
- Liquid AI's d1 models on LFM3, such as d1-2B and d1-Omni-2B,
  use a different prompt and readout, which `.d1LFM2` doesn't follow,
  and `.automatic` doesn't select it for them.

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

llama.cpp's Metal backend stops a process with an assertion
if the process exits while a model has layers on the GPU,
and a decision that is still running at exit can crash on the GPU or the CPU.
So `LlamaDecisionModel` registers an exit handler when a load ends.
The handler stops new loads and decisions,
waits up to three seconds for running work to stop at its next batch of prompt tokens,
and then frees every loaded model and session context.
After the handler starts,
loads and decisions throw `DecisionError.modelUnavailable`.

The handler is best effort.
The process can still stop with the assertion in these cases:

- It exits during the first load,
  because the handler is registered when a load ends.
- It exits from inside a load, a decision, or the release of a model or a session,
  for example from a llama.cpp callback.
  On the thread that runs that work,
  the handler returns at once and frees nothing.
  On a thread that llama.cpp or another library owns,
  the handler waits the full three seconds and then frees nothing.
- Work is still running after three seconds.
  The handler then frees nothing.
  On the CPU, one batch of `batchSize` prompt tokens can take longer;
  a smaller batch size shortens the wait.
- It exits without running exit handlers,
  for example with `_exit` or after a crash.

If freeing the models takes more than ten seconds,
for example because another library's log callback blocks,
the handler aborts the process, so that the exit can't hang.
A crash report for this abort shows
`LlamaLifecycle.abortAfterExitCleanupTimeout()` in its backtrace,
which first tries to write one line to standard error,
but aborts whether or not the line can be written.
If the cleanup ends just after the limit,
the cleanup's own thread can abort first,
without the line,
and the backtrace shows `LlamaLifecycle.abortAfterLateExitCleanup()`.

To be certain that no model is loaded when the process exits,
wait until every call to a session that uses a `LlamaDecisionModel` returns,
start no new calls,
release every such session,
and then call `LlamaDecisionModel.removeAllFromCache()`
or `removeFromCache()` on each model.
A session keeps its weights until it is released,
and neither call waits for a load that is still running.
After the handler starts, both calls still release the cache's references;
weights that the handler freed are not freed again.

To be certain that the assertion can't stop the process,
even with a model loaded on the GPU,
set the environment variable `GGML_METAL_NO_RESIDENCY` to any value
before the first model loads:
in the process's environment,
or with `setenv("GGML_METAL_NO_RESIDENCY", "1", 1)` before the first load,
because this package starts llama.cpp's backend at the first load.
If another library in the process uses llama.cpp,
set the variable before that library starts llama.cpp with `llama_backend_init`.
The Metal backend then doesn't use residency sets,
so the assertion has nothing to count.
Residency sets keep a model's Metal buffers wired for a keep-alive period;
without them, the OS can reclaim that memory while the GPU is idle,
and the next decision waits for it again.
This package doesn't set the variable.
A decision that is still running at exit can still crash.

`LlamaDecisionModel.logLevel` sets the minimum level of the llama.cpp messages
that are written to standard error, and you can set it at any time.
The model installs its log callback once, before its first call to llama.cpp.
If AnyLanguageModel's `LlamaLanguageModel` or AnyEmbeddingModel's `LlamaEmbeddingModel`
runs in the same process,
each package installs its own callback and its own exit handler.
llama.cpp keeps one callback for the process,
so the callback that was installed last decides which messages appear,
and the other packages' log levels have no effect.
Each exit handler waits only for its own package's work
and frees only its own package's models,
so an exit can wait up to three seconds for each package,
and a model that another package still has on the GPU can still cause the assertion.

### Jev

`SystemOneDecisionModel.jev(…)` returns a model
that sends each batch to `POST /v1/systemone`.
The API key comes from `TYPESAFE_AI_API_KEY`
(or `TYPESAFE_API_KEY`), or you can pass it as `apiKey:`.
The model retries HTTP 429, 503, and 529 with exponential backoff
and honors `Retry-After`;
pass `retryPolicy: .never` to disable retries.
Jev's probabilities and confidence values are returned unchanged.
If the server rounds the probabilities,
they can sum to slightly more or less than 1.

The model works with any server that implements the System One API,
such as [Kev](https://github.com/jaredpalmer/kev)
or [Ollaya](https://github.com/ollaya-dev/ollaya).
Pass the server's URL, a placeholder API key, and its model name:

```swift
let model = SystemOneDecisionModel.jev(
    baseURL: URL(string: "http://127.0.0.1:11435")!,
    apiKey: "local",
    modelID: "laya"
)
```

The API key must not be empty, even when the server ignores it.
The [`SystemOneDecisionModel` initializer](#system-one) needs no placeholder key.

### Ollama

`SystemOneDecisionModel.ollama(…)` returns a model
that sends each batch to `POST /v1/systemone` on an Ollama server.
Pass the name of a decision model, such as
[Nimble](https://ollama.com/library/nimble):

```swift
let model = SystemOneDecisionModel.ollama(modelID: "nimble")
let session = DecisionSession(model: model, state: .text(ticket))
let urgent = try await session.probability(of: "Does this convey urgency?")
```

Local requests need no API key.
For Ollama's cloud, set `OLLAMA_API_KEY` (or pass `apiKey:`)
and point `baseURL:` at `https://ollama.com`,
or add `:cloud` to the model name to go through a local server.
Pass `keepAlive:` to control how long the model stays loaded, in seconds.
Ollama's probabilities and confidence values are returned unchanged.
If the server rounds the probabilities,
they can sum to slightly more or less than 1.

### System One

`SystemOneDecisionModel` works with servers that implement the System One API.
Pass the server's URL and model name,
and the path, API key, headers, limits, and extra request fields that the server needs:

```swift
let model = try SystemOneDecisionModel(
    baseURL: URL(string: "http://127.0.0.1:8000")!,
    path: "v1/systemone",
    modelID: "my-model",
    apiKey: ProcessInfo.processInfo.environment["MY_API_KEY"],
    headers: ["traceparent": traceparent],
    capabilities: DecisionCapabilities(maximumQuestions: 64, maximumChoiceOptions: 26),
    extraBody: ["cache_salt": "tenant-1"]
)
```

The model sends the API key, if any, as a bearer token.
The session checks each batch against `capabilities` before it sends a request.
`extraBody` cannot set `model`, `state`, `questions`, or `images`,
and `headers` cannot set `Authorization` or `Content-Type`;
the initializer throws `SystemOneDecisionModel.ConfigurationError`,
which names the key, if they do.
These headers and body fields are the model's defaults.
A request can add to them or override them with
`options[custom: SystemOneDecisionModel.self]`,
which holds `extraBody` and `extraHeaders`:

```swift
var options = DecisionOptions()
options[custom: SystemOneDecisionModel.self] = .init(
    extraBody: ["cache_salt": "tenant-2"],
    extraHeaders: ["traceparent": requestTraceparent]
)
let urgent = try await session.probability(of: "Does this convey urgency?", options: options)
```

A request option overrides a default with the same top-level body key,
or a header with the same name in any letter case.
The other defaults stay.
Request options have the same reserved keys,
and the request throws a `DecisionError` with the kind `unsupportedOption`
before it is sent if one is set.
Every `SystemOneDecisionModel` reads the same request options,
so put a field that belongs to one server, such as provider routing, on the model.
The model retries like the Jev model; pass `retryPolicy: .never` to disable retries.

#### Migrating from `JevDecisionModel` and `OllamaDecisionModel`

Version 0.11 deprecates `JevDecisionModel` and `OllamaDecisionModel`,
and version 1.0 removes them.
`SystemOneDecisionModel.jev(…)` and `SystemOneDecisionModel.ollama(…)`
take the same arguments and send the same requests:

```swift
// Before
let jev = JevDecisionModel(modelID: "jev-latest")
let ollama = OllamaDecisionModel(modelID: "nimble", keepAlive: 300)

// After
let jev = SystemOneDecisionModel.jev(modelID: "jev-latest")
let ollama = SystemOneDecisionModel.ollama(modelID: "nimble", keepAlive: 300)
```

The returned models read request options for `SystemOneDecisionModel`,
which the deprecated types ignore.
Version 1.0 also removes the types' `defaultBaseURL` and `environmentAPIKey(from:)`.
To read an API key yourself,
read `TYPESAFE_AI_API_KEY` (or `TYPESAFE_API_KEY`) for Jev
and `OLLAMA_API_KEY` for Ollama.

#### Images

By default, `SystemOneDecisionModel` sends text and JSON states only,
and a session rejects a state with attachments before it sends a request.
To send images, set `imageEncoding`.
The model then sends the images of a segments state
in a top-level `images` array, in segment order,
and sends the text segments, joined by newlines, as the state.
Empty text segments are left out of a state with images;
a state of text segments only keeps them.
The servers differ in the form of each image:

- `.dataURL` sends a data URL, such as `data:image/jpeg;base64,/9j/4AAQ…`.
  llama.cpp's `llama-server` and Liquid's API document this form.
- `.base64` sends a base64-encoded string with no prefix, such as `/9j/4AAQ…`.
  Ollama documents this form.

For example, this model asks a local `llama-server` about a picture:

```swift
// llama-server -hf LiquidAI/d1-Omni-2B-GGUF:Q8_0 -b 8192 -ub 8192
let model = try SystemOneDecisionModel(
    baseURL: URL(string: "http://127.0.0.1:9931")!,
    modelID: "LiquidAI/d1-Omni-2B-GGUF:Q8_0",
    imageEncoding: .dataURL
)
let state = DecisionState.segments([
    .text("A photo from a customer's damage claim:"),
    .attachment(DecisionState.Attachment(imageData: jpeg, mimeType: "image/jpeg")),
])
let session = DecisionSession(model: model, state: state)
let damaged = try await session.probability(of: "Is the item damaged?")
```

This configuration was checked by hand with llama.cpp 1.0.0 and that model,
with one JPEG image, a yes-or-no question, and a choice question.
Images have not been run against Liquid's API or Ollama.
llama.cpp places every image before the state in the prompt,
and returns HTTP status 501 if the model or its projector does not read images;
the model throws a `DecisionError` with the kind `httpError` for that response.

A model with an image encoding declares `.image` in `capabilities.supportedAttachmentKinds`,
and a model without one declares no attachment kinds,
whatever kinds the `capabilities` argument lists.
The session rejects audio, video, and document attachments before it sends a request,
and the model rejects an image with a URL source or with no data,
because these servers read image bytes only.
The model checks typed attachments only.
Fields that a caller adds in `extraBody` or inside a JSON state,
such as llama.cpp's `files` or image parts in chat messages,
go to the server unchecked.
The model does not send TypeSafe's `assets` form of images,
audio or video, or images in question instructions,
and it does not ask `box` questions.
`SystemOneDecisionModel.jev(…)` and `SystemOneDecisionModel.ollama(…)` send no image attachments.

### Gemini

`GeminiDecisionModel` sends each batch to `POST /v1beta/decisions`
on the [Gemini Decisions API](https://aistudio.google.com/docs/decisions-api).
The API key comes from `GEMINI_API_KEY`, or you can pass it to the initializer:

```swift
let model = GeminiDecisionModel()
let session = DecisionSession(model: model, state: .text(ticket))
let urgent = try await session.probability(of: "Does this convey urgency?")
```

The model defaults to `gemini-decision-1`.
A text state is sent as the input text,
and a JSON state is sent as compact JSON text with sorted keys.
A request can have at most 1,000 questions,
a choice must have 2 to 255 options,
and a score must have 2 to 255 non-empty levels.
The model checks these limits and Gemini's rules for option names before it sends a request.
If Gemini blocks the input, the model throws a `DecisionError` with the kind `guardrailViolation`.

To ask about images, audio, video, or PDF documents,
pass a state of text and attachment segments:

```swift
let state = DecisionState.segments([
    .text("A photo from a customer's damage claim:"),
    .attachment(DecisionState.Attachment(imageData: jpeg, mimeType: "image/jpeg")),
    .attachment(DecisionState.Attachment(documentURL: receiptURL)),
])
let session = DecisionSession(model: GeminiDecisionModel(), state: state)
let damaged = try await session.probability(of: "Is the item damaged?")
```

The model sends the segments in order as Gemini content blocks.
Gemini joins adjacent text blocks with no separator,
so the model joins adjacent text segments with a newline.
Attachment URLs must use HTTPS,
and the model checks MIME types against Gemini's documented types,
which include only PDF for documents.
Inline data can total at most 100 MB in one request, and each inline PDF can be at most 50 MB.
The model rejects an empty input, including text with only whitespace,
and an attachment with no data.
Only Gemini checks the context window and the sizes of files at URLs.

Gemini's probabilities and confidence values are returned unchanged.
Gemini clamps the probability of a binary answer to 0.01 through 0.99,
so a binary answer is never 0 or 1.
Gemini also returns a confidence value for binary answers,
which the model drops because `Answer.binary` has no confidence value.
The model does not retry failed requests.

### Custom backends

To add a backend, conform a type to `DecisionModel`
and return an executor that conforms to `DecisionModelExecutor`.
The executor gets each batch as a `DecisionRequest`,
which holds the state, the questions, and the options:

```swift
struct MyDecisionModel: DecisionModel {
    struct CustomOptions: CustomDecisionOptions {
        var temperature: Double?
    }

    var modelID: String { "my-model" }
    var capabilities: DecisionCapabilities { DecisionCapabilities() }

    func makeExecutor() -> any DecisionModelExecutor { Executor() }

    struct Executor: DecisionModelExecutor {
        func decide(_ request: DecisionRequest) async throws -> DecisionSession.Response {
            let temperature = request.options[custom: MyDecisionModel.self]?.temperature
            // Answer request.questions about request.state.
        }
    }
}
```

Omit `CustomOptions` if the model has no options of its own.

To migrate an executor from version 0.9,
replace `decide(_ questions: [Question], about state: DecisionState)`
with `decide(_ request: DecisionRequest)`,
and read `request.questions` and `request.state`.
The package has no adapter between the two requirements,
so an executor with the old method no longer conforms.
Callers need no change:
each `DecisionSession` method and the initializer keep their 0.9 signatures,
which use the session's options,
so calls and references to them as function values compile unchanged.

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
Jev, other System One servers, and Gemini compute confidence with their own formulas,
so do not compare confidence values between backends.

Each local answer also reports an allowed-answer mass in `DecisionSession.Diagnostics`:
the share of next-token probability that fell on the answer labels before normalization.
A low value means that the model wanted to reply with something else.

## Limitations

- Wire compatibility with Jev, Ollama, and Gemini is tested against fixtures, not live services.
  A live Gemini test runs when `GEMINI_API_KEY` is set.
- Calibration was fitted on two tasks.
  Whether it transfers to other tasks is untested.
- Do not rely on typed output alone to resist injected instructions.
- Inference is not optimized beyond prefix caching.

## Development

The repository uses [mise](https://mise.jdx.dev/) tasks:

| Task | Description |
| --- | --- |
| `mise run setup` | Resolve package dependencies. |
| `mise run lint` | Check Swift formatting. |
| `mise run format` | Format Swift files. |
| `mise run build:ios` | Build for iOS. |
| `mise run build:linux` | Build and test in Docker on Linux. |
| `mise run build` | Build the core library. |
| `mise run build:mlx` | Build with the MLX trait. |
| `mise run build:coreai` | Build with the CoreAI trait. |
| `mise run build:llama` | Build with the Llama trait. |
| `mise run test` | Run the tests without MLX. |
| `mise run test:mlx` | Run the MLX integration tests. Downloads Qwen3 4B. |
| `mise run test:mlx:head` | Run the MLX decision head integration tests. Downloads Clef-flash 4-bit. |
| `mise run test:mlx:d1` | Run the Liquid AI d1 integration tests. Downloads d1-3B 8-bit. |
| `mise run test:coreai` | Run the Core AI integration tests. Downloads Qwen3 0.6B. |
| `mise run test:llama` | Run the tests with the Llama trait. Set `LLAMA_MODEL_PATH` to a GGUF file for inference tests. |

MLX tasks need Xcode's Swift 6.4 or later and the Metal compiler.
Core AI tasks need Xcode 27, and their integration tests run only on macOS 27.
Set `MLX_HEAD_MODEL_DIRECTORY`, `MLX_D1_MODEL_DIRECTORY`, or `COREAI_MODEL_BUNDLE`
to test a model from a local directory.

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
