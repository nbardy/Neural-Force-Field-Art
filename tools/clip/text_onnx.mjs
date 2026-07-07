// tools/clip/text_onnx.mjs — MobileCLIP-S0 TEXT encoder via ONNX Runtime (CPU).
//
// Run:  node tools/clip/text_onnx.mjs ["a prompt"]        (default: "a photo of a cat")
//
// The vision half is a fused WGSL kernel (real-time, per-frame, on-GPU — see
// README.md). The TEXT half runs ONCE per prompt change, so a stock runtime is
// fine here: no hand-written kernels, just tokenize → ORT-CPU → 512-d embedding.
// The embedding is what the vision encoder's output is scored against (cosine
// loss), so it must live in the SAME 512-d MobileCLIP-S0 space.
//
// Pipeline:
//   prompt → CLIP tokenizer (77-tok context) → int64 input_ids → text_model → text_embeds[512]
//
// Model: Xenova/mobileclip_s0 onnx/text_model_fp16.onnx (85 MB fp16 weights,
// but int64 input / float32 output — onnxruntime-node handles the fp16 interior
// transparently). GOTCHA: this fp16 export only loads at
// graphOptimizationLevel:"basic" — "all"/"disabled" both hit an ORT
// SimplifiedLayerNormFusion bug ("InsertedPrecisionFreeCast … does not exist").
//
// Tokenizer: CLIPTokenizer from the same repo (tokenizer.json +
// tokenizer_config.json), driven by @huggingface/transformers AutoTokenizer
// pinned to LOCAL files (env.allowRemoteModels=false). Verified empirically:
// BOS=49406, EOT=49407, and it pads to 77 with token id 0 after the EOT
// (tokenizer_config pad_token="!" — id 0 in the CLIP byte-level vocab, i.e. the
// "pad with zeros" convention). model_max_length=77.
//
// All model/tokenizer files live under models/mobileclip_s0/ (gitignored,
// auto-downloaded from HF if missing). Output fixture:
//   models/mobileclip_s0/fixtures/text_embeds_test.json  {prompt: [512 floats]}
import { writeFileSync, mkdirSync, existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";
import ort from "onnxruntime-node";
import { AutoTokenizer, env } from "@huggingface/transformers";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const MODEL_DIR = join(ROOT, "models", "mobileclip_s0");
const FIXTURE_DIR = join(MODEL_DIR, "fixtures");
const TEXT_MODEL = join(MODEL_DIR, "onnx", "text_model_fp16.onnx");
const HF = "https://huggingface.co/Xenova/mobileclip_s0/resolve/main";
const CTX = 77; // CLIP text context length (tokenizer_config model_max_length)

// One-time fetch of tokenizer + fp16 text model into the gitignored models dir.
for (const rel of ["tokenizer.json", "tokenizer_config.json", "onnx/text_model_fp16.onnx"]) {
  const dst = join(MODEL_DIR, rel);
  if (existsSync(dst)) continue;
  mkdirSync(dirname(dst), { recursive: true });
  console.error(`fetching ${rel} …`);
  execSync(`curl -sfL -o "${dst}" "${HF}/${rel}"`, { stdio: "inherit" });
}

// --- tokenizer (local files only; no network at runtime) ---
env.allowRemoteModels = false;
env.localModelPath = join(ROOT, "models"); // from_pretrained("mobileclip_s0") → models/mobileclip_s0/
const tokenizer = await AutoTokenizer.from_pretrained("mobileclip_s0");

// Returns { ids: number[77], tensor: ort.Tensor(int64,[1,77]) }.
// padding:"max_length" + truncation pins every prompt to the 77-tok context and
// zero-pads after the EOT, matching what the model was trained/exported with.
function tokenize(prompt) {
  const enc = tokenizer(prompt, { padding: "max_length", max_length: CTX, truncation: true });
  const data = enc.input_ids.data; // BigInt64Array, dims [1, 77]
  return {
    ids: Array.from(data, Number),
    tensor: new ort.Tensor("int64", data, enc.input_ids.dims),
  };
}

// --- text model (see header re: graphOptimizationLevel) ---
const session = await ort.InferenceSession.create(TEXT_MODEL, { graphOptimizationLevel: "basic" });

// prompt → Float32Array(512). Embeddings are NOT L2-normalized by the model.
async function embed(prompt) {
  const { ids, tensor } = tokenize(prompt);
  const out = await session.run({ input_ids: tensor });
  const d = out.text_embeds.data; // Float32Array(512)
  return { ids, vec: new Float32Array(d) }; // copy off ORT's backing buffer
}

const l2 = (v) => Math.hypot(...v);
const cos = (a, b) => {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return dot / Math.sqrt(na * nb);
};

// Golden IMAGE embedding — a random-noise 256×256 image encoded by the vision
// model (onnx_forward.mjs). Node pools small (<4 KB) Buffers, so .buffer is the
// shared pool, NOT this file's bytes: ALWAYS view via (buffer, byteOffset,
// length/4) or Float32Array reads pool garbage. (ort_web_bench.mjs gotcha.)
const gbuf = readFileSync(join(FIXTURE_DIR, "image_embeds_512.f32.bin"));
const imageEmbed = new Float32Array(gbuf.buffer, gbuf.byteOffset, gbuf.length / 4);

// --- run ---
const SANITY = ["a photo of a cat", "a photo of a dog", "a diagram of a neural network"];
const argPrompt = process.argv.slice(2).join(" ") || SANITY[0];

// Compute every embedding we need once (dedup the arg prompt against SANITY).
const prompts = [...new Set([argPrompt, ...SANITY])];
const E = new Map();
for (const p of prompts) E.set(p, await embed(p));

// 1) argv prompt: token ids, first 8 embedding values, L2 norm.
{
  const { ids, vec } = E.get(argPrompt);
  const nTok = ids.findIndex((x) => x === 49407) + 1 || ids.length; // through EOT
  console.log(`model: ${TEXT_MODEL}`);
  console.log(`prompt: "${argPrompt}"`);
  console.log(`token ids (${nTok} real + ${CTX - nTok} zero-pad → ${CTX}): [${ids.slice(0, nTok).join(", ")}, 0, …]`);
  console.log(`embedding: [1, ${vec.length}]  L2=${l2(vec).toFixed(4)}`);
  console.log(`  first 8: ${[...vec.slice(0, 8)].map((x) => x.toFixed(5)).join(", ")}`);
}

// 2) semantic sanity — text↔text cosine. cat–dog MUST beat cat–diagram.
const [cat, dog, diagram] = SANITY.map((p) => E.get(p).vec);
const catDog = cos(cat, dog);
const catDiagram = cos(cat, diagram);
console.log("\ntext↔text cosine (sanity):");
console.log(`  cat  vs dog      = ${catDog.toFixed(4)}`);
console.log(`  cat  vs diagram  = ${catDiagram.toFixed(4)}`);
console.log(`  dog  vs diagram  = ${cos(dog, diagram).toFixed(4)}`);
console.log(`  → cat–dog ${catDog > catDiagram ? ">" : "≤"} cat–diagram  ` +
  `(${catDog > catDiagram ? "PASS" : "FAIL"})`);

// 3) text↔image cosine vs the golden random-noise image (expect low/similar).
console.log("\ntext↔image cosine (vs golden random-noise image):");
for (const p of SANITY) console.log(`  ${p.padEnd(30)} = ${cos(E.get(p).vec, imageEmbed).toFixed(4)}`);

// 4) write the three sanity embeddings for later use.
mkdirSync(FIXTURE_DIR, { recursive: true });
const dump = Object.fromEntries(SANITY.map((p) => [p, Array.from(E.get(p).vec)]));
const out = join(FIXTURE_DIR, "text_embeds_test.json");
writeFileSync(out, JSON.stringify(dump));
console.log(`\nfixture → ${out}`);

if (catDog <= catDiagram) process.exitCode = 1; // sanity gate
