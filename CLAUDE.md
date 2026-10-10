# CLAUDE.md

Rampart for .NET: a native C# port of National Design Studio's Rampart PII filter
(https://huggingface.co/nationaldesignstudio/rampart and the `@nationaldesignstudio/rampart` npm
package). Deterministic recognizers + a 4-bit MiniLM token classifier + default-deny placeholder
policy. The published model is embedded in the assembly and run without ONNX Runtime.

## Layout

- `Rampart/` — the library (net10.0, package id `Rampart`).
  - `Resources/` — `model_q4.onnx`, `vocab.txt`, `config.json`, `LICENSE` (CC BY 4.0), byte-for-byte
    the Hugging Face files. Embedded as `Rampart.model_q4.onnx`, `Rampart.vocab.txt`, `Rampart.config.json`.
  - `src/Onnx/` — minimal protobuf reader for ONNX `ModelProto` (nodes, initializers, `Constant`s).
  - `src/Model/` — `RampartWeights` (binds tensors by graph structure/names), `RampartModel` (forward
    pass), `Q4Linear` (MatMulNBits weights), `Gemm` (SIMD micro-kernels), `Kernels` (LayerNorm, GELU/erf, softmax).
  - `src/Tokenization/` — `WordPieceTokenizer` (mirrors the transformers.js BERT tokenizer).
  - `src/Ner/` — `TokenClassifier` (the transformers.js token-classification pipeline), `NerDetector`
    (port of `detectNer`: windows, folding, BIO merge, span repair).
  - `src/Detection/` — `Heuristics`, `Validators`, `Policy` (merge + keep-set), `Premask`.
  - `src/Session/` — `SessionEntityTable` (placeholders), `StreamingReveal`.
  - `src/` — `ChatGuard`, `RampartAnalyzer`, `Redactor`, `Types` (labels, `PiiSpan`).
- `tests/Rampart.Tests/` — xunit v3 suite; `Fixtures/reference.json` is generated, never hand-edited.
- `tools/reference/` — `generate-fixtures.ts`, which runs the ORIGINAL npm package to produce the fixture.
- `samples/Rampart.Bench/` — latency vs ONNX Runtime and a per-stage profile (`-- --profile`).

## Build & test

```bash
dotnet build -c Release
dotnet test --project tests/Rampart.Tests/Rampart.Tests.csproj -c Release
dotnet run -c Release --project samples/Rampart.Bench [-- --profile]
```

xunit v3 runs on Microsoft.Testing.Platform (`global.json`), so `dotnet test` takes `--project`.
The built test assembly is also directly runnable: `dotnet tests/Rampart.Tests/bin/Release/net10.0/Rampart.Tests.dll -class "*LayerParity*"`.

The model is a plain git blob (14.7 MB), not Git LFS, so a fresh clone can run every test.

## The port is defined by parity — keep it that way

This library is a port. Its behaviour is defined by the reference, and the tests prove it:

- `LayerParityTests` — ONNX Runtime runs the embedded graph with every intermediate tensor added as
  a graph output (`OnnxReference.AddGraphOutputs`); `RampartModel.ComputeLogits(ids, IForwardTrace)`
  records the same tensors under the same ONNX names (`OnnxNames`). Bounds: max error 5e-5 of the
  tensor's peak, RMS 2e-5 (observed ~6e-6 / ~2.5e-6). If you change a kernel, these must still pass;
  never loosen a bound to make a change pass — find the numerical difference.
- `LogitsParityTests` — logits vs ORT's fully optimized graph on every fixture input.
- `PipelineParityTests` — tokens, classifier rows, heuristics, premask, NER spans, merged spans and
  the protected text vs the original JavaScript pipeline, case by case.
- `PublicEvalTests` — the upstream strict gate (recall/retention ≥ 95%) and equality with the
  reference's own score.

Rules that follow from that:

- **Reproduce the reference, quirks included.** The JS is the spec: stable sorts (`OrderBy`, never
  `List.Sort`, where JS sorts), JS regex semantics (ASCII `\d`/`\w`/`\b`, JS whitespace for `\s`, `\z`
  for an un-flagged `$`), JS `toLowerCase` (`WordPieceTokenizer.JsLowerCase`), float32 softmax
  (`TokenClassifier.Float32Softmax`), UTF-16 offsets. A behaviour that looks wrong but matches the
  reference stays, unless it leaks PII.
- **A deliberate divergence is documented and tested.** Today there is one: astral characters do not
  shift NER offsets (`PipelineParityTests.KnownDivergences`, README "Deliberate differences"). Adding
  another needs the same: an entry in `KnownDivergences` with the reason, a test pinning the new
  behaviour, and a README note.
- **Weights are read, never converted.** `RampartWeights.Bind` takes tensors from the ONNX file as
  shipped; `ResourceTests` pins the SHA-256 of every embedded file. To update the model, replace the
  files in `Rampart/Resources/`, update those hashes, regenerate the fixture and run the suite.

## Regenerating the reference fixture

Needs bun (or Node with TypeScript stripping) and a local copy of the Hugging Face repo:

```bash
cd tools/reference
npm install                      # @nationaldesignstudio/rampart@0.1.3, @huggingface/transformers@3.7.5
bun generate-fixtures.ts /path/to/rampart-hf-clone
```

It refuses to run unless the clone's `model_q4.onnx` is byte-identical to the embedded one. Add new
inputs to `tools/reference/corpus.ts`, never to the JSON.

## Performance notes

- Default load expands 4-bit weights into transposed float panels (~42 MB) once;
  `RampartModelOptions.KeepWeightsQuantized` keeps ~5 MB and dequantizes per call (same results;
  `GemmTests.QuantizedAndExpandedWeightsAgree`).
- `Gemm.Multiply` is `C = A·B` with a broadcast-FMA micro-kernel (6×32 on AVX-512 — chosen on
  `Avx512F.IsSupported`, not `Vector512.IsHardwareAccelerated`, which is false on many Xeons — and
  4×16 on 256-bit). Do not go back to a dot-product (`A·Bᵀ`) kernel: with head dim 64 the horizontal
  sums dominated attention.
- Parallelism: linears split into (96-token block × 32-column panel) tasks; attention into
  (head × 48-query block) tasks; LayerNorm/GELU split only for long sequences.
- Benchmark after warm-up (tiered JIT) and with ORT spinning disabled; the bench does both.

## Licensing

Code: MIT (`LICENSE`). Model files: CC BY 4.0, © National Design Studio (`LICENSE-MODEL`,
`Rampart/Resources/LICENSE`, `THIRD-PARTY-NOTICES.md`) — keep the attribution when moving or
repackaging them. The NuGet package ships both licenses (`MIT AND CC-BY-4.0`).
