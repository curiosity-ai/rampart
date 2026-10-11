# CLAUDE.md

Rampart for .NET: a native C# port of National Design Studio's Rampart PII filter — source
https://github.com/nationaldesignstudio/rampart, model https://huggingface.co/nationaldesignstudio/rampart.
Deterministic recognizers + a 4-bit MiniLM token classifier + default-deny placeholder policy. The published model is embedded in the assembly and run without ONNX Runtime.

## Layout

- `Rampart/` — the library (net10.0, package id `Rampart`).
  - `README.md` — the package readme (NuGet page): consumer samples. The repo-root README covers the
    port, validation and benchmarks.
  - `Resources/` — `model_q4.onnx`, `vocab.txt`, `config.json`, `LICENSE` (CC BY 4.0), byte-for-byte
    the Hugging Face files. Embedded as `Rampart.model_q4.onnx`, `Rampart.vocab.txt`, `Rampart.config.json`.
  - `src/Onnx/` — minimal protobuf reader for ONNX `ModelProto` (nodes, initializers, `Constant`s).
  - `src/Model/` — `RampartWeights` (binds tensors by graph structure/names), `RampartModel` (forward
    pass), `Q4Linear` (MatMulNBits weights in panel order), `Gemm` (SIMD micro-kernels), `Kernels` (LayerNorm,
    GELU/erf, softmax/exp), `Parallelism` (spinning worker pool).
  - `src/Buffers/` — `PooledList<T>`, the ArrayPool-backed scratch list of every hot path.
  - `src/Tokenization/` — `WordPieceTokenizer` (mirrors the transformers.js BERT tokenizer).
  - `src/Ner/` — `TokenClassifier` (the transformers.js token-classification pipeline), `NerDetector`
    (port of `detectNer`: windows, folding, BIO merge, span repair).
  - `src/Detection/` — `Heuristics`, `Validators`, `Policy` (merge + keep-set), `Premask`.
  - `src/Session/` — `SessionEntityTable` (placeholders), `StreamingReveal`.
  - `src/` — the public API: `RampartAnalyzer` (+ `RampartOptions`), `RampartAnonymizer`, `RampartCloaker`,
    `RecognizerResult`; and the internal `ChatGuard` (port of upstream `guard.ts`) and `Types` (labels, `PiiSpan`).
- `tests/Rampart.Tests/` — xunit v3 suite; `Fixtures/reference.json` is generated, never hand-edited.
  `Upstream/` holds the ports of the source repo's `__tests__/*.test.ts`, one C# file per suite.
- `tools/reference/` — `generate-fixtures.ts`, which runs the ORIGINAL source (a clone) to produce the fixture.
- `samples/Rampart.Bench/` — latency sweep vs ONNX Runtime, per-stage profile, kernel and allocation
  modes; `BENCHMARK-RESULT.md` holds the published numbers.
- `samples/Rampart.Redact/` — interactive redactor, the counterpart of upstream's `cli/redact.ts`.
- `.devops/azure-pipelines.yml` — Azure DevOps: build, test, then pack and push `Rampart` to nuget.org
  (CalVer `yy.M.<buildId>`, service connection `nuget-curiosity-org`) on pushes to `main`; PR builds
  stop after the tests.

## Working against the original source

Keep a clone of the original next to this repo and read it when touching any ported stage; the C#
mirrors its file layout closely (`src/heuristics.ts` → `Detection/Heuristics.cs`, `src/ner/classifier.ts`
→ `Ner/NerDetector.cs` + `Ner/TokenClassifier.cs`, `src/session.ts` → `Session/SessionEntityTable.cs`, …).

```bash
git clone https://github.com/nationaldesignstudio/rampart ../nationaldesignstudio/rampart
cd ../nationaldesignstudio/rampart
bun install --frozen-lockfile
git lfs pull              # model/; if LFS is unavailable, copy the Hugging Face files in (same bytes)
bunx vitest run           # upstream's own unit tests
bun eval/run-public-eval.ts --strict
```

`model/` in the source repo is Git LFS; its object ids are the SHA-256 hashes `ResourceTests` pins, so
the Hugging Face download is a valid substitute when LFS is not reachable.

When upstream changes:
1. Diff the new commit's `src/`, `index.ts`, `eval/public-cases.ts` and `__tests__/` against the one
   recorded in the fixture (`"source"` in `reference.json`).
2. Port the changes, and port new or changed upstream tests into `tests/Rampart.Tests/Upstream/`.
3. Regenerate the fixture from the new commit (below) and run the whole suite.

## Build & test

```bash
dotnet build -c Release
dotnet test --project tests/Rampart.Tests/Rampart.Tests.csproj -c Release
dotnet run -c Release --project samples/Rampart.Bench [-- --profile]
```

xunit v3 runs on Microsoft.Testing.Platform (`global.json`), so `dotnet test` takes `--project`.
The built test assembly is also directly runnable: `dotnet tests/Rampart.Tests/bin/Release/net10.0/Rampart.Tests.dll -class "*LayerParity*"`.

The model is a plain git blob (14.7 MB), not Git LFS, so a fresh clone can run every test.

`Rampart/README.md` ships in the package, so its samples are the public face of the API: when a public
signature changes, update them in the same change. Every snippet compiles, and the output in its
comments is what the code prints. Check both by pasting them into a scratch console app that
references `Rampart/Rampart.csproj`. Pick sample inputs the model handles cleanly, and state its
misses under "Limitations" rather than hiding them.

## The public API is the Catalyst.Presidio flow

Consumers use Rampart the way they use Catalyst.Presidio, and only that way:
`RampartAnalyzer.Analyze(text)` → `List<RecognizerResult>` → `RampartAnonymizer.Anonymize(text, results,
operator)` (one way) or `RampartCloaker.CloakAsync(text, results, action)` (placeholders around a service
call, plus a streaming overload). `RecognizerResult` has Catalyst.Presidio's shape (`End` inclusive,
`float Score`, string `EntityType`) and the anonymizer its signature and operators, so code moves between the two.

- **Everything else is internal**: tokenizer, classifier, NER, heuristics, premask, policy, session table,
  streaming reveal, `ChatGuard`, the model's `ComputeLogits`. Tests and the bench reach them through
  `InternalsVisibleTo`. Do not make a stage public, and do not add integration helpers (DI registration,
  `IChatClient` wrappers and the like) to the package.
- The internal `PiiSpan` is end-exclusive like the reference; convert at the boundary (`End - 1` out,
  `End + 1` in).
- `PipelineParityTests.AnalyzeThenCloak*` pins the public flow to the reference: what `CloakAsync` sends
  must equal upstream `guard.protect` on every fixture case and conversation.
- `Analyze(text, AnalyzeOptions?)` is the one place behaviour may depart from the reference, and only
  when the caller asks:
  - `ExpandCapitalizedParticles = false` skips `NerDetector.RescueCapitalizedParticles`.
  - `Hints` (entity type → values, compiled once into `Detection/HintSet`) adds every hint occurrence the
    layers missed, with score 1, overriding the keep-set. It also extends spans over hint words in the
    repair loop (`ExtendOverHintWords`): a leftward pass first, then a rightward one, with the name
    types sharing words.
  
  Default options must stay byte-identical to the reference (`AnalyzeOptionsTests.DefaultOptionsAreTheReference`).
  `ChatGuard`, the parity tests and `Detect(string)` use the defaults.

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

Needs bun and the source clone above (installed, `model/` hydrated):

```bash
bun tools/reference/generate-fixtures.ts ../nationaldesignstudio/rampart [model-dir]
```

It imports the clone's `index.ts`, `eval/public-cases.ts` and its own `@huggingface/transformers`, refuses
to run unless the model is byte-identical to the embedded one, and records the source commit in the
fixture. Add new inputs to `tools/reference/corpus.ts`, never to the JSON.

## Performance and allocation notes

Measure before and after any change to the engine; the VM-class machines this runs on are noisy
(±10%), so repeat runs. Every mode of `samples/Rampart.Bench` first prints the machine and a
calibration (GEMM GFLOPS best/median, memory read/copy GB/s, CPU steal). Compare a result only with
one whose calibration is in the same band; `--env` prints just that block. Modes:
default (sweep 32–512 tokens, C# vs ORT, 1 thread and all cores), `--single` / `--multi` (one pair),
`--avx512-ab` (C# with `DOTNET_EnableAVX512=0`, then default, in child processes; on .NET 10 the
per-ISA `DOTNET_EnableAVX512F` is ignored), `--profile [--short] [--single]` (time per stage),
`--gemm` / `--attention` (kernel throughput), `--alloc [--single]` and `--alloc-parts` (bytes per call).

- **Compare engines in one process, interleaved** (what the default mode does), never in separate
  runs: the VM gets rescheduled between runs, and a 10% shift would read as a win or a regression.
  Re-publish `BENCHMARK-RESULT.md` when the numbers move.
- **Weights stay 4-bit** (~5 MB, cache-resident), permuted at load into panel order — bit-exact, not
  re-quantized (`GemmTests.PanelOrderedWeightsDecodeToTheShippedNibbles`). Expanding them to float
  (42 MB) was tried and is slower: it no longer fits the L3 and streams from DRAM on every call.
- **`Gemm.Multiply`** is `C = A·B` with a broadcast-FMA micro-kernel specialised per row count by a
  generic struct parameter (12×32 on AVX-512 — chosen on `Avx512F.IsSupported`, not
  `Vector512.IsHardwareAccelerated`, which is false on many Xeons — and 6×16 on 256-bit). Column
  remainders run the vector kernel on a zero-padded panel. Do not go back to a dot-product (`A·Bᵀ`)
  kernel (horizontal sums), and do not fuse an epilogue with many constants (GELU) into the kernel:
  the hoisted constants push the 24 accumulators into spills.
- **Attention never materialises `[heads, T, T]`** except for a trace that captures tensors; each task
  works in a pooled L2-sized scratch. Writing the full tensor made the scores GEMM memory-bound
  (23 GFLOPS instead of 95). Head dim is 32.
- **Parallelism** goes through `Parallelism.For` and a struct `IParallelJob` — never `Parallel.For`
  with a lambda: a lambda that captures a method-scope variable allocates its closure on every call,
  even when the parallel branch is not taken. The pool's workers spin ~0.5 ms before parking; a call
  that finds the pool busy runs inline. Single-threaded, linears iterate tile-major over all tokens.
- **Allocation-free hot path**: `PooledList<T>` (pass by `ref`, dispose once) for every scratch
  buffer; the tokenizer and the NER window path work on spans; per-code-point Unicode folds are cached
  (`string.Normalize`/`TryNormalize` allocate); `Heuristics` scans the two backtracking rules by hand
  (`HeuristicsScannerTests` fuzzes them against the regexes). New code on these paths should keep
  `--alloc` at 0 B for `ComputeLogits` and only result objects for the pipeline.
- Benchmark after a time-based warm-up (the tiered JIT recompiles in the background for a while), and
  time with `Stopwatch.GetTimestamp` rather than `StartNew` (a `Stopwatch` is 40 B, charged to the
  engine under test). The bench does both.

## Licensing

Code: MIT (`LICENSE`). Model files: CC BY 4.0, © National Design Studio (`LICENSE-MODEL`,
`Rampart/Resources/LICENSE`, `THIRD-PARTY-NOTICES.md`) — keep the attribution when moving or
repackaging them. The NuGet package ships both licenses (`MIT AND CC-BY-4.0`).
