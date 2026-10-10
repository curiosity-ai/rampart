/**
 * Generates tests/Rampart.Tests/Fixtures/reference.json by running the ORIGINAL Rampart source
 * (https://github.com/nationaldesignstudio/rampart) over the corpus.
 *
 *   git clone https://github.com/nationaldesignstudio/rampart ../rampart-src
 *   (cd ../rampart-src && bun install --frozen-lockfile && git lfs pull)
 *   bun tools/reference/generate-fixtures.ts ../rampart-src [model-dir]
 *
 * The pipeline, the public eval cases and @huggingface/transformers all come from the clone (its own
 * bun.lock pins the runtime), so the fixture is exactly what that commit computes. [model-dir]
 * defaults to <clone>/model and must hold onnx/model_q4.onnx byte-identical to the resource the C#
 * library embeds.
 */
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, symlinkSync, writeFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { EXTRA_CASES, CONVERSATIONS, longText } from "./corpus.ts";

const here = dirname(new URL(import.meta.url).pathname);
const repo = resolve(here, "../..");
if (process.argv[2] === undefined) throw new Error("usage: bun generate-fixtures.ts <rampart-source-clone> [model-dir]");
const source = resolve(process.argv[2]);
const modelDir = resolve(process.argv[3] ?? join(source, "model"));
const sha = (p: string) => createHash("sha256").update(readFileSync(p)).digest("hex");
const embedded = join(repo, "Rampart/Resources/model_q4.onnx");
if (sha(join(modelDir, "onnx/model_q4.onnx")) !== sha(embedded)) {
  throw new Error(`${modelDir}/onnx/model_q4.onnx is not the model the C# library embeds (missing git lfs pull?)`);
}
const commit = execFileSync("git", ["-C", source, "rev-parse", "HEAD"]).toString().trim();
const sourcePackage = JSON.parse(readFileSync(join(source, "package.json"), "utf8"));

// Resolve transformers.js from the clone so `env` below is the very instance its pipeline uses.
const { env } = await import(Bun.resolveSync("@huggingface/transformers", source));
const transformersVersion = JSON.parse(
  readFileSync(join(dirname(Bun.resolveSync("@huggingface/transformers/package.json", source)), "package.json"), "utf8"),
).version;
const { ChatGuard, detectHeuristics, detectNer, loadNerClassifier, premask, projectMaskedSpan, mergeSpans } = await import(
  join(source, "index.ts")
);
const { PUBLIC_E2E_CASES } = await import(join(source, "eval/public-cases.ts"));

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
  const contextual = masked.map((s: any) => projectMaskedSpan(s, text, map)).filter((s: any) => s !== null);
  return { heuristic, masked: map.masked, nerOnMasked: masked, detected: [...heuristic, ...contextual] };
}

const texts: { id: string; text: string; privateTerms?: unknown; publicTerms?: unknown }[] = [
  ...PUBLIC_E2E_CASES.map((c: any) => ({ id: c.id, text: c.input, privateTerms: c.privateTerms, publicTerms: c.publicTerms })),
  ...EXTRA_CASES,
  { id: "long-windowed", text: longText() },
];

const cases = [];
for (const c of texts) {
  const inferText = c.text.replaceAll("-", " ");
  const { heuristic, masked, nerOnMasked, detected } = await detectAll(c.text);
  const guard = new ChatGuard({ ner: (t: string) => detectNer(t, classifier) });
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
  const guard = new ChatGuard({ ner: (t: string) => detectNer(t, classifier) });
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
      source: `https://github.com/nationaldesignstudio/rampart@${commit}`,
      package: `${sourcePackage.name}@${sourcePackage.version}`,
      transformers: `@huggingface/transformers@${transformersVersion}`,
      model_sha256: sha(embedded),
      cases,
      conversations,
    },
    null,
    1,
  ) + "\n",
);
console.log(`wrote ${cases.length} cases and ${conversations.length} conversations to ${out}`);
