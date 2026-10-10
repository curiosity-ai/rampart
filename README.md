# Rampart for .NET

On-device PII detection and reversible redaction for .NET 10 — a native C# port of
[National Design Studio's Rampart](https://github.com/nationaldesignstudio/rampart)
([model](https://huggingface.co/nationaldesignstudio/rampart)): the
deterministic recognizers, the 4-bit MiniLM token classifier and the default-deny placeholder
policy, with the published model embedded in the assembly.

No ONNX Runtime, no Python, no network: the encoder is evaluated directly from the published
`model_q4.onnx` weights with SIMD kernels, and every stage is checked against the original —
layer by layer against ONNX Runtime, and end to end against the original JavaScript package.

```csharp
using Rampart;

var guard = new ChatGuard();                     // one per conversation

var safe = guard.Protect("My name is Alex Rivera and my SSN is 472-81-0094.");
// safe.Text         → "My name is [GIVEN_NAME_1] [SURNAME_1] and my SSN is [SSN_1]."
// safe.Placeholders → [GIVEN_NAME_1], [SURNAME_1], [SSN_1]

var reply = await llm.CompleteAsync(safe.Text);  // the provider never sees the real values
Console.WriteLine(guard.Reveal(reply));          // "[GIVEN_NAME_1]" → "Alex"

await foreach (var chunk in guard.RevealAsync(llm.StreamAsync(safe.Text)))
    Console.Write(chunk);                        // placeholders split across chunks are handled
```

```
dotnet add package Rampart
```

[`Rampart/README.md`](Rampart/README.md) is the package's own readme, also shown on NuGet. It has
samples for `IChatClient` round trips, streaming, multi-turn conversations, detection, log redaction,
policy, other languages and ASP.NET Core.

## What it does

```
user message
  → heuristics      SSN (structural rules), credit card (Luhn), email, URL, IPv4/IPv6/MAC
  → premask         structured spans become sentinels ([SSN], [EMAIL], …) before the model sees the text
  → classifier      MiniLM-L6-H384, 35 BIO labels: names, phone, IDs, address parts, …
  → repair          bridge low-confidence fragments, join name connectors, rescue particles (De La Croix)
  → merge + policy  disjoint spans; default-deny: everything is redacted except CITY, STATE, ZIP_CODE
  → session table   stable typed placeholders, the same value → the same token on every turn
assistant reply
  → rehydrate       placeholders back to the real values, also for streamed replies
```

| Layer | Labels |
|---|---|
| Deterministic (premasked) | `SSN`, `CREDIT_CARD`, `EMAIL`, `URL`, `IP_ADDRESS` |
| Model, redacted by default | `GIVEN_NAME`, `SURNAME`, `PHONE`, `TAX_ID`, `BANK_ACCOUNT`, `ROUTING_NUMBER`, `GOVERNMENT_ID`, `PASSPORT`, `DRIVERS_LICENSE`, `BUILDING_NUMBER`, `STREET_NAME`, `SECONDARY_ADDRESS` |
| Model, kept by default | `CITY`, `STATE`, `ZIP_CODE` |

Languages: English, Spanish, French, German, Italian, Portuguese, Dutch (Latin script). Read the
[model card](https://huggingface.co/nationaldesignstudio/rampart) for the evaluation and its
limitations before relying on it: recall on non-Latin-script names is low, structured government IDs
without checksums are model-only, and adversarial inputs (zero-width characters, prompt injection)
can evade detection. Rampart is harm reduction, not a security boundary.

## API

| Type | Purpose |
|---|---|
| `ChatGuard` | Per-conversation filter: `Protect`, `Reveal`, `RevealAsync`, `CreateStreamingReveal`, `ProtectReply`. Not thread-safe; create one per conversation (cheap — share a `RampartAnalyzer`). |
| `RampartAnalyzer` | Stateless, thread-safe detection: `Detect` (raw spans from both layers) and `Analyze` (merged, kept labels included). |
| `Redactor` | One-way redaction for logs and traces: label tag, mask, remove or SHA-256. |
| `RampartOptions` | `KeepLabels`, `Aliases` (e.g. `GIVEN_NAME → NAME`), `MinScore` (0.4), `HeuristicsOnly`, a custom `Ner` detector, a custom `Model`. |
| `RampartModel` | The encoder. `RampartModel.Default` is the embedded model, loaded once and shared; `ComputeLogits(ids)` for raw access; `LoadFromDirectory` for a local copy of the Hugging Face repo. |
| `WordPieceTokenizer`, `TokenClassifier`, `NerDetector`, `Heuristics`, `Policy`, `Premask`, `SessionEntityTable`, `StreamingReveal` | The individual stages, public so each can be used or replaced on its own. |
| `ITokenClassifier` / `DelegateTokenClassifier`, `INerDetector` / `DelegateNerDetector` | The reference's plug points: run `NerDetector`'s windowing and span repair over another classifier (raw BIO rows or aggregated `entity_group` rows, optional tokenizer), or give the guard a different detector. |

```csharp
// Detection only, e.g. for an audit log or a Presidio-style analyzer.
var analyzer = new RampartAnalyzer();
var text = "You can reach Maria Lopez at (415) 555-0132, she lives in Boston.";
foreach (var span in analyzer.Analyze(text))
    Console.WriteLine($"{span.Label.ToName(),-12} {span.Score:0.00} \"{span.Text}\" [{span.Start}, {span.End})");
// GIVEN_NAME   0.98 "Maria" [14, 19)
// SURNAME      0.97 "Lopez" [20, 25)
// PHONE        1.00 "(415) 555-0132" [29, 43)
// CITY         0.98 "Boston" [58, 64)

// One-way redaction.
analyzer.Redact(text);                      // "You can reach [GIVEN_NAME] [SURNAME] at [PHONE], she lives in Boston."
analyzer.Redact(text, RedactionMode.Mask);  // "You can reach ***** ***** at **************, she lives in Boston."

// Policy: keep nothing, and collapse name parts into one alias.
var strict = new ChatGuard(new RampartOptions
{
    KeepLabels = [],
    Aliases = new Dictionary<PiiLabel, string> { [PiiLabel.GivenName] = "NAME", [PiiLabel.Surname] = "NAME" },
});
```

Offsets in `PiiSpan` are UTF-16 code units, `Start` inclusive and `End` exclusive.

### Relation to Catalyst.Presidio

[Catalyst.Presidio](https://github.com/curiosity-ai/catalyst/tree/master/Catalyst.Presidio) is a
pattern-based analyzer / anonymizer / cloaker built on Catalyst's tokenizer and spotters. This
library plays the same three roles with a learned model behind them and no dependency on Catalyst:

| Catalyst.Presidio | Rampart |
|---|---|
| `PresidioAnalyzer.Analyze` → `RecognizerResult` | `RampartAnalyzer.Analyze` → `PiiSpan` (end exclusive) |
| `PresidioAnonymizer.Anonymize` (mask / replace / redact / hash) | `Redactor.Redact` (`Mask` / `Label` / `Remove` / `Hash`) |
| `PresidioCloaker.CloakAsync` (pseudonyms + rehydration, streaming) | `ChatGuard.Protect` / `Reveal` / `RevealAsync` |

## How the model runs

The shipped graph is a BERT encoder (6 layers, hidden 384, 12 heads, FFN 1536, 19,730-piece vocab)
exported with 4-bit `MatMulNBits` linears (block 32, scale per block, zero point 8) and uint8
`DequantizeLinear` embeddings. `RampartModel` reads the ONNX protobuf directly, binds the tensors
by graph structure — nothing is re-quantized or converted — and evaluates:

- **Embeddings**: word + token type, then + position, exactly the graph's order; LayerNorm.
- **Linears**: the 4-bit weights stay 4-bit (~5 MB for the model, so they stay in the last-level
  cache between calls). At load they are only *permuted* into panel order (bit-exact); each call
  dequantizes one 32-column panel at a time with a few SIMD instructions and multiplies every token
  against it with a register-blocked GEMM (12×32 block in 24 zmm accumulators on AVX-512, 6×16 on
  AVX2/NEON; broadcast-FMA, no horizontal sums).
- **Attention**: Q and K each scaled by `√(1/√32)` as in the graph; Kᵀ and V packed per head; each
  (head, block of queries) task runs scores, softmax and the weighted sum of V in an L2-sized
  scratch buffer, so the `[heads, T, T]` probabilities never go to memory.
- **FFN**: exact-erf GELU (vectorized rational approximation, |error| < 5e-7); residual adds fused
  into the LayerNorms; softmax with a fast vectorized `exp` (< 2.5e-7 relative error).
- **Classifier**: argmax over 35 labels, scores from a float32 softmax as transformers.js computes it.
- **Threads**: a dedicated pool of `cores − 1` workers that spin briefly between the ~40 parallel
  regions of a pass, so short inputs gain from parallelism too. `RampartModelOptions.MaxDegreeOfParallelism = 1`
  runs everything on the calling thread (for servers that parallelize across requests); a call that
  finds the pool busy with another inference runs inline rather than queueing.

Tokenization reproduces the transformers.js tokenizer the reference runs (special-token splitting,
`BertNormalizer` cleaning, CJK padding, lowercasing, NFD accent stripping, `BertPreTokenizer`,
greedy WordPiece with `##` continuations and the 100-character limit).

### Performance

Median latency on a 4-core cloud Xeon (AVX-512) against ONNX Runtime 1.23 with full graph
optimization on the same model. All four engines ran in one process, interleaved
(`samples/Rampart.Bench`). Full sweep, environment, calibration and AVX-512 A/B are in
[BENCHMARK-RESULT.md](samples/Rampart.Bench/BENCHMARK-RESULT.md):

| Tokens | Rampart C#, 1 thread | ONNX Runtime, 1 thread | Rampart C#, 4 threads | ONNX Runtime, 4 threads (spinning) |
|---:|---:|---:|---:|---:|
| 32 (a chat message) | 9.9 ms | 13.3 ms | 4.6 ms | 6.4 ms |
| 128 | 37.9 ms | 37.2 ms | 14.5 ms | 17.1 ms |
| 256 | 86.5 ms | 79.8 ms | 30.8 ms | 29.2 ms |
| 512 (one full window) | 196 ms | 208 ms | 65 ms | 80 ms |

Short inputs are ~25% faster than ORT. From 128 tokens up, one thread is at parity (±8%, within the
VM's run-to-run noise), and four threads are faster at every length but 256.
`ChatGuard.Protect` on the chat message, everything included, takes ~4.5 ms (~10.7 ms on one thread).
Loading the embedded model takes 0.3–0.5 s. Inputs longer than one window are scanned in overlapping
500-token windows.

### Allocations

Every buffer on the hot path comes from `ArrayPool<T>.Shared`, and the per-character Unicode folds
are cached, so after warm-up (bytes per call, all threads, `--alloc`):

| Call | Allocated |
|---|---|
| `RampartModel.ComputeLogits(ids, Span<float> logits)` — any length, 1 or 4 threads | 0 B |
| `WordPieceTokenizer.CountTokens` | 0 B |
| `RampartAnalyzer.Detect` (chat message) | ~0.7 KB — the returned spans, their text and the list |
| `ChatGuard.Protect` (chat message) | ~1.2 KB — the above plus the scrubbed string and placeholders |

## Validation

The port was developed against the original source,
[nationaldesignstudio/rampart](https://github.com/nationaldesignstudio/rampart) (commit `95798af`,
package version 0.1.3). The test suite (`tests/Rampart.Tests`, 800+ tests) checks it three ways:

1. **ONNX Runtime running the published graph** (test-only dependency). The graph is re-encoded
   with every intermediate tensor exposed as an output, and the C# engine traces the same tensors
   under the same names. **69 tensors per input** — embeddings, Q/K/V, attention probabilities,
   context, both residual blocks, pre- and post-GELU activations, logits — are compared for 14
   inputs and a full 512-token window. The worst deviation is ~6e-6 relative to the tensor's peak
   (float32 summation order); the bound is 5e-5. Logits are also compared against ORT's fully
   optimized graph on all 93 fixture inputs, window by window, including argmax agreement.
2. **The original JavaScript pipeline**, run from a clone of the source repository on its pinned
   runtime (transformers.js 3.7.5 + onnxruntime-node) over 93 inputs (the package's 58 public eval cases plus
   multilingual, structured-ID, Unicode, special-token, long and edge cases) and a multi-turn
   conversation. Per case the fixture records tokens, the raw per-token classifier output, heuristic
   spans, the premasked text, NER spans, merged detections and the final placeholdered text; the C#
   port reproduces all of them (scores within 1e-4). See `tools/reference/`.
3. **The upstream unit tests, ported** (`tests/Rampart.Tests/Upstream/`): `heuristics`, `premask`,
   `guard`, `pipeline`, `ner-classifier` and `public-api-e2e` from the source repo's `__tests__/`, one
   C# test per upstream case, with the same mock classifiers and detectors (aggregated and BIO rows,
   mocked tokenizers and token counts for the windowing). Only `bench-score` is not ported: it tests
   the offline benchmark-scoring scripts, not the runtime.

The package's own strict gate is enforced too: on its public eval the port scores private-term
recall 74/76 and public-term retention 78/79 — identical to the original, whose two leaks are the
two exotic IDs its comments call out (`AGY-2026-009871`, `1EG4-TE5-MK73`).

The embedded files are checked byte-for-byte against the published SHA-256 hashes, which are also
the Git LFS object ids of `model/` in the source repository.

### Deliberate differences from the reference

The port reproduces the reference's behaviour, quirks included: for example, the particle rescue that
keeps "De La Croix" together also pulls any capitalized word of up to four letters right before a
name into it, so "I'm Alex" and "Call Maria" each redact as one GIVEN_NAME (over-redaction, not a leak). One
reference bug is fixed rather than reproduced: the reference's `foldForModel` records one offset per
code point while its folded string grows by two UTF-16 units for an emoji or other astral character,
so every NER span after such a character is shifted — it leaks part of the name ("A[GIVEN_NAME_1]")
and can emit a span with an undefined end. The port keeps the offsets aligned; the
`x-unicode` fixture case is excluded from the span parity tests and pinned by
`AstralCharactersDoNotShiftOffsets` instead.

## Build and test

```bash
dotnet build -c Release
dotnet test --project tests/Rampart.Tests/Rampart.Tests.csproj -c Release
dotnet run -c Release --project samples/Rampart.Bench
dotnet run -c Release --project samples/Rampart.Redact     # interactive redactor, like upstream's `bun run redact`
```

Requires the .NET 10 SDK. Tests use xunit v3 on Microsoft.Testing.Platform (`global.json`).

## Publishing

[`.devops/azure-pipelines.yml`](.devops/azure-pipelines.yml) is the Azure DevOps pipeline. On every
push to `main` it builds the solution and runs the whole test suite. Only if that passes does it
pack `Rampart` with a CalVer version (`yy.M.<buildId>`) and push it to nuget.org through the
`nuget-curiosity-org` service connection. Pull request builds run the build and tests and stop
there.

## Licenses

- The code in this repository is MIT licensed — see [`LICENSE`](LICENSE).
- The Rampart model, tokenizer vocabulary and configuration embedded in the library
  (`Rampart/Resources/`) are © 2026 National Design Studio, licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see [`LICENSE-MODEL`](LICENSE-MODEL)
  and [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md). The model was trained on the
  [AI4Privacy OpenPII](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) corpus (CC BY 4.0).
- The public eval cases in the test fixture, and the upstream tests ported under `tests/Rampart.Tests/Upstream/`, come from [nationaldesignstudio/rampart](https://github.com/nationaldesignstudio/rampart) (CC BY 4.0).

Cite the model as: National Design Studio (2026), *Rampart*, https://huggingface.co/nationaldesignstudio/rampart.
