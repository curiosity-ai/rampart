/**
 * Generates tests/Rampart.Tests/Fixtures/reference.json by running the ORIGINAL Rampart pipeline
 * (@nationaldesignstudio/rampart 0.1.3 on transformers.js 3.7.5 + onnxruntime-node) over a corpus.
 *
 *   cd tools/reference && npm install && bun generate-fixtures.ts <model-dir>
 *
 * <model-dir> is a local copy of https://huggingface.co/nationaldesignstudio/rampart (it must hold
 * onnx/model_q4.onnx, tokenizer.json, tokenizer_config.json, config.json). The script checks that its
 * model_q4.onnx is byte-identical to the resource the C# library embeds.
 */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, symlinkSync, writeFileSync, existsSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { env } from "@huggingface/transformers";
import {
  ChatGuard,
  detectHeuristics,
  detectNer,
  loadNerClassifier,
  premask,
  projectMaskedSpan,
  mergeSpans,
} from "@nationaldesignstudio/rampart";
import { PUBLIC_E2E_CASES } from "./node_modules/@nationaldesignstudio/rampart/eval/public-cases.ts";
import { EXTRA_CASES, CONVERSATIONS, longText } from "./corpus.ts";

const here = dirname(new URL(import.meta.url).pathname);
const repo = resolve(here, "../..");
const modelDir = resolve(process.argv[2] ?? join(here, "model"));
const sha = (p: string) => createHash("sha256").update(readFileSync(p)).digest("hex");
const embedded = join(repo, "Rampart/Resources/model_q4.onnx");
if (sha(join(modelDir, "onnx/model_q4.onnx")) !== sha(embedded)) {
  throw new Error("model_q4.onnx differs from the resource embedded in the C# library");
}

// transformers.js resolves `<localModelPath>/<model id>`; point a scratch root at the model dir.
const root = join(tmpdir(), "rampart-fixture-models");
rmSync(root, { recursive: true, force: true });
mkdirSync(join(root, "nationaldesignstudio"), { recursive: true });
symlinkSync(modelDir, join(root, "nationaldesignstudio", "rampart"));
env.localModelPath = root + "/";
env.allowRemoteModels = false;

const classifier = await loadNerClassifier({ device: "cpu" });

async function detectAll(text: string) {
  const heuristic = detectHeuristics(text);
  const map = premask(text, heuristic);
  const masked = await detectNer(map.masked, classifier);
  const contextual = masked.map((s) => projectMaskedSpan(s, text, map)).filter((s) => s !== null);
  return { heuristic, masked: map.masked, nerOnMasked: masked, detected: [...heuristic, ...contextual] };
}

const texts: { id: string; text: string; privateTerms?: unknown; publicTerms?: unknown }[] = [
  ...PUBLIC_E2E_CASES.map((c) => ({ id: c.id, text: c.input, privateTerms: c.privateTerms, publicTerms: c.publicTerms })),
  ...EXTRA_CASES,
  { id: "long-windowed", text: longText() },
];

const cases = [];
for (const c of texts) {
  const inferText = c.text.replaceAll("-", " ");
  const { heuristic, masked, nerOnMasked, detected } = await detectAll(c.text);
  const guard = new ChatGuard({ ner: (t) => detectNer(t, classifier) });
  const protectedResult = await guard.protect(c.text);
  cases.push({
    id: c.id,
    text: c.text,
    privateTerms: c.privateTerms,
    publicTerms: c.publicTerms,
    tokens: classifier.tokenize!(c.text),
    tokenCount: classifier.countTokens!(c.text),
    // Raw pipeline rows on the hyphen-folded text (what detectNer hands the classifier), only when it fits one window.
    entities: classifier.countTokens!(inferText) <= 500 ? await classifier(inferText) : null,
    heuristic,
    masked,
    nerOnMasked,
    detected,
    merged: mergeSpans(detected),
    protected: protectedResult,
  });
}

const conversations = [];
for (const conv of CONVERSATIONS) {
  const guard = new ChatGuard({ ner: (t) => detectNer(t, classifier) });
  const turns = [];
  for (const turn of conv.turns) {
    const result = await guard.protect(turn.user);
    turns.push({ user: turn.user, protected: result, reply: turn.reply, revealed: guard.reveal(turn.reply) });
  }
  conversations.push({ id: conv.id, turns });
}

const out = join(repo, "tests/Rampart.Tests/Fixtures/reference.json");
writeFileSync(
  out,
  JSON.stringify(
    {
      generator: "tools/reference/generate-fixtures.ts",
      package: "@nationaldesignstudio/rampart@0.1.3",
      transformers: "@huggingface/transformers@3.7.5",
      model_sha256: sha(embedded),
      cases,
      conversations,
    },
    null,
    1,
  ) + "\n",
);
console.log(`wrote ${cases.length} cases and ${conversations.length} conversations to ${out}`);
