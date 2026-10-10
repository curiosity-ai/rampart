# Reference fixtures

`generate-fixtures.ts` runs the **original** Rampart pipeline — `@nationaldesignstudio/rampart`
0.1.3 on `@huggingface/transformers` 3.7.5 with onnxruntime-node (CPU) — over the package's public
eval cases plus `corpus.ts`, and writes `tests/Rampart.Tests/Fixtures/reference.json`. The C#
`PipelineParityTests`, `LogitsParityTests` and `PublicEvalTests` compare against it.

```bash
git clone https://huggingface.co/nationaldesignstudio/rampart /tmp/rampart-hf   # or download the files
cd tools/reference
npm install
bun generate-fixtures.ts /tmp/rampart-hf
```

The script checks that the clone's `onnx/model_q4.onnx` is byte-identical to
`Rampart/Resources/model_q4.onnx` before it runs. Per case it records the WordPiece tokens, the raw
token-classification rows, heuristic spans, the premasked text, NER spans on that text, all detected
spans, the merged spans and the `ChatGuard.protect` result; per conversation, each turn's protected
text and revealed reply.
