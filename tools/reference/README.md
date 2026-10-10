# Reference fixtures

`generate-fixtures.ts` runs the **original** Rampart source — a clone of
https://github.com/nationaldesignstudio/rampart on its own pinned runtime (`bun.lock`:
`@huggingface/transformers` 3.7.5 with onnxruntime-node, CPU) — over the source repo's public eval
cases plus `corpus.ts`, and writes `tests/Rampart.Tests/Fixtures/reference.json`. The C#
`PipelineParityTests`, `LogitsParityTests`, `PublicEvalTests` and `Upstream/PublicApiE2eTests`
read it.

```bash
git clone https://github.com/nationaldesignstudio/rampart ../rampart-src
(cd ../rampart-src && bun install --frozen-lockfile && git lfs pull)
bun tools/reference/generate-fixtures.ts ../rampart-src            # model from ../rampart-src/model
bun tools/reference/generate-fixtures.ts ../rampart-src /tmp/hf    # or from a Hugging Face download
```

The script checks that the model's `onnx/model_q4.onnx` is byte-identical to
`Rampart/Resources/model_q4.onnx`, and records the source commit, package version and
transformers.js version in the fixture. Per case it records the WordPiece tokens, the raw
token-classification rows, heuristic spans, the premasked text, NER spans on that text, all detected
spans, the merged spans and the `ChatGuard.protect` result; per conversation, each turn's protected
text and revealed reply.
