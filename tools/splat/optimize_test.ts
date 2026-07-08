/**
 * Headless end-to-end gate for the prompt→splats optimization core
 * (src/splat/optimize.ts) on a REAL Dawn/Metal adapter via bun-webgpu — the
 * whole loop (splat raster → CLIP forward → −cos loss → CLIP backward → splat
 * backward → Adam) wired on ONE device, with NO browser and NO ORT.
 *
 *   bun tools/splat/optimize_test.ts [steps=80] [G=200000]
 *
 * Prereqs (all regenerable, gitignored under models/mobileclip_s0/):
 *   node tools/clip/onnx_forward.mjs
 *   uv run --with onnx --with numpy python tools/clip/compile_plan.py --train
 *   node tools/clip/text_onnx.mjs "a photo of a cat"   # writes text_embeds_test.json
 *
 * The gate: optimizing 200K random splats toward the REAL "a photo of a cat"
 * text embedding must MAKE cos(image_embed, text) GO UP (i.e. −cos, the loss,
 * goes down). This is the single decisive proof that every seam — the two
 * 768 KB image/grad copies, the frozen-weight CLIP backward, the splat
 * backward, and Adam — is wired correctly and end-to-end differentiable.
 * Before/after PNGs are written so the drift toward the prompt is eyeballable.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { SplatOptimizer, cosine, LEGIBLE_G } from "../../src/splat/optimize";
import type { TrainPlan } from "../../src/clip/vision";
import { writePNG } from "./scene";

setupGlobals();

// Defaults reflect the tuned LEGIBLE regime (structure in ~20 steps), not the
// old 200K noise gate. tools/splat/tune.ts is the knob-sweep harness.
const STEPS = Number(process.argv[2] ?? 20);
const G = Number(process.argv[3] ?? LEGIBLE_G);
const PROMPT = "a photo of a cat";

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const OUT_DIR = "/tmp";

const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = new Float32Array(readFileSync(join(MODEL_DIR, "weights_train.bin")).buffer);
const textAll = JSON.parse(
  readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")
) as Record<string, number[]>;
const textEmbed = new Float32Array(textAll[PROMPT]);
if (textEmbed.length !== plan.textDim) {
  throw new Error(`text embed ${textEmbed.length} != plan.textDim ${plan.textDim}`);
}

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: any = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const t0 = performance.now();
const opt = await SplatOptimizer.create(device, plan, weights, { G });
console.log(`SplatOptimizer(${G} splats) built in ${(performance.now() - t0).toFixed(0)} ms`);
opt.setPrompt(textEmbed);

// control prompts: the image should end up MORE like the cat than these
const controls = Object.keys(textAll).filter((k) => k !== PROMPT);

async function cosToPrompt(): Promise<number> {
  const emb = await opt.currentEmbedding();
  return cosine(emb, textEmbed);
}

const before = await opt.renderImage();
writePNG(join(OUT_DIR, "splat_prompt_before.png"), before, opt.side, opt.side);
const cos0 = await cosToPrompt();
console.log(`\nstep   0: cos("${PROMPT}") = ${cos0.toFixed(5)}`);

// warm up the pipelines (Metal lazy-JIT — same rule as the clip suites)
for (let i = 0; i < 3; i++) opt.step();

const tStep0 = performance.now();
let cosLast = cos0;
for (let s = 1; s <= STEPS; s++) {
  opt.step();
  if (s % 10 === 0 || s === STEPS) {
    cosLast = await cosToPrompt();
    console.log(`step ${String(s).padStart(3)}: cos = ${cosLast.toFixed(5)}  (Δ ${(cosLast - cos0 >= 0 ? "+" : "")}${(cosLast - cos0).toFixed(5)})`);
  }
}
const msPerStep = (performance.now() - tStep0) / (STEPS + 3);

const after = await opt.renderImage();
writePNG(join(OUT_DIR, "splat_prompt_after.png"), after, opt.side, opt.side);

// final embedding vs all prompts (should rank the target highest, or at least
// have risen against it more than the controls)
const finalEmb = await opt.currentEmbedding();
console.log(`\nfinal cosines:`);
console.log(`  "${PROMPT}" (target): ${cosine(finalEmb, textEmbed).toFixed(5)}  [was ${cos0.toFixed(5)}]`);
for (const k of controls) {
  console.log(`  "${k}" (control): ${cosine(finalEmb, new Float32Array(textAll[k])).toFixed(5)}`);
}

console.log(`\nbench: ${msPerStep.toFixed(1)} ms/optimize-step (raster fwd+bwd+adam + CLIP fwd+loss+bwd, ${G} splats)`);
console.log(`PNGs: ${join(OUT_DIR, "splat_prompt_before.png")}  →  ${join(OUT_DIR, "splat_prompt_after.png")}`);

const improved = cosLast - cos0;
if (improved > 0.01) {
  console.log(`\nGATE PASS: cos rose ${improved.toFixed(4)} (> 0.01) — the loop optimizes toward the prompt.`);
} else {
  console.error(`\nGATE FAIL: cos rose only ${improved.toFixed(4)} (need > 0.01).`);
  process.exit(1);
}
