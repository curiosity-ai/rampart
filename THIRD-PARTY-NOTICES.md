# Third-party notices

## Rampart model (embedded)

The files in `Rampart/Resources/` — `model_q4.onnx`, `vocab.txt` and `config.json` — are the
Rampart PII token-classification model published at
https://huggingface.co/nationaldesignstudio/rampart, included unmodified.

- Copyright 2026 National Design Studio
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/ — full text in [`LICENSE-MODEL`](LICENSE-MODEL)
  and `Rampart/Resources/LICENSE`.
- Base model: `nreimers/MiniLM-L6-H384-uncased`, fine-tuned by National Design Studio.
- Training data includes the AI4Privacy `pii-masking-openpii-1.5m` corpus (CC BY 4.0),
  https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m.

No changes were made to the model files. The C# code in this repository is an independent
re-implementation of the inference pipeline of the `@nationaldesignstudio/rampart` package
(CC BY 4.0); its behaviour follows that package, which is attributed here.

## Tests

`tests/Rampart.Tests/Fixtures/reference.json` contains outputs of the Rampart source
(https://github.com/nationaldesignstudio/rampart, package version 0.1.3) and the public evaluation
cases from its `eval/public-cases.ts`. The tests under `tests/Rampart.Tests/Upstream/` are C# ports
of its `__tests__/` suites. © National Design Studio, CC BY 4.0.
