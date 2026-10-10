# Rampart for .NET

On-device PII detection and reversible redaction for .NET 10 — a native C# port of
[National Design Studio's Rampart](https://huggingface.co/nationaldesignstudio/rampart): the
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
- **Linears**: 4-bit weights unpacked with SIMD nibble shuffles into transposed float panels, then a
  register-blocked GEMM (6×32 block on AVX-512, 4×16 on AVX2/NEON, FMA, no horizontal sums).
  By default the panels are built once at load (~42 MB); `RampartModelOptions.KeepWeightsQuantized`
  keeps ~5 MB of 4-bit weights and dequantizes per call instead, with identical results.
- **Attention**: Q and K each scaled by `√(1/√64)` as in the graph; per (head, block of queries)
  tasks that run scores, softmax and the weighted sum of V while the block is in cache.
- **FFN**: exact-erf GELU (vectorized rational approximation, |error| < 5e-7).
- **Classifier**: argmax over 35 labels, scores from a float32 softmax as transformers.js computes it.

Tokenization reproduces the transformers.js tokenizer the reference runs (special-token splitting,
`BertNormalizer` cleaning, CJK padding, lowercasing, NFD accent stripping, `BertPreTokenizer`,
greedy WordPiece with `##` continuations and the 100-character limit).

### Performance

Median latency on a 4-core cloud Xeon (AVX-512), weights expanded, against ONNX Runtime 1.23 with
full graph optimization on the same model (`samples/Rampart.Bench`):

| Input | Rampart C# | ONNX Runtime |
|---|---|---|
| chat message (30 tokens) | ~10.5 ms | ~9.8 ms |
| 128 tokens | ~35 ms | ~22 ms |
| 512 tokens (one full window) | ~140 ms | ~91 ms |

`ChatGuard.Protect` on the chat message, everything included, takes ~13 ms. Loading the embedded
model takes ~0.1–0.2 s. Inputs longer than one window are scanned in overlapping 500-token windows.

## Validation

The test suite (`tests/Rampart.Tests`, 700+ tests) checks the port against two references:

1. **ONNX Runtime running the published graph** (test-only dependency). The graph is re-encoded
   with every intermediate tensor exposed as an output, and the C# engine traces the same tensors
   under the same names. **69 tensors per input** — embeddings, Q/K/V, attention probabilities,
   context, both residual blocks, pre- and post-GELU activations, logits — are compared for 14
   inputs and a full 512-token window. The worst deviation is ~6e-6 relative to the tensor's peak
   (float32 summation order); the bound is 5e-5. Logits are also compared against ORT's fully
   optimized graph on all 93 fixture inputs, window by window, including argmax agreement.
2. **The original JavaScript pipeline** — `@nationaldesignstudio/rampart` 0.1.3 on transformers.js
   3.7.5 + onnxruntime-node — run over 93 inputs (the package's 58 public eval cases plus
   multilingual, structured-ID, Unicode, special-token, long and edge cases) and a multi-turn
   conversation. Per case the fixture records tokens, the raw per-token classifier output, heuristic
   spans, the premasked text, NER spans, merged detections and the final placeholdered text; the C#
   port reproduces all of them (scores within 1e-4). See `tools/reference/`.

The package's own strict gate is enforced too: on its public eval the port scores private-term
recall 74/76 and public-term retention 78/79 — identical to the original, whose two leaks are the
two exotic IDs its comments call out (`AGY-2026-009871`, `1EG4-TE5-MK73`).

The embedded files are checked byte-for-byte against the published SHA-256 hashes.

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
```

Requires the .NET 10 SDK. Tests use xunit v3 on Microsoft.Testing.Platform (`global.json`).

## Licenses

- The code in this repository is MIT licensed — see [`LICENSE`](LICENSE).
- The Rampart model, tokenizer vocabulary and configuration embedded in the library
  (`Rampart/Resources/`) are © 2026 National Design Studio, licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see [`LICENSE-MODEL`](LICENSE-MODEL)
  and [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md). The model was trained on the
  [AI4Privacy OpenPII](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) corpus (CC BY 4.0).
- The public eval cases in the test fixture come from the `@nationaldesignstudio/rampart` package (CC BY 4.0).

Cite the model as: National Design Studio (2026), *Rampart*, https://huggingface.co/nationaldesignstudio/rampart.
