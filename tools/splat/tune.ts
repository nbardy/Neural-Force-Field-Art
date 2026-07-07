/**
 * tune — fast iteration harness for dialing in a LEGIBLE CLIP-splat config
 * (structure, not a per-pixel noise average, in ~20 steps). Runs the real
 * SplatOptimizer headless on Metal and dumps a PNG so the result is eyeballable.
 *
 *   G=12000 STEPS=20 SCALE=9 OPACITY=0.4 COLOR=1.2 \
 *   LR_MEAN=1.5 LR_SCALE=0.06 LR_THETA=0.08 LR_COLOR=0.12 LR_OP=0.06 \
 *   PROMPT="a photo of a cat" TAG=a bun tools/splat/tune.ts
 *
 * Writes /tmp/tune_<TAG>.png and prints the cosine trajectory. Every knob has a
 * LEGIBLE_* default so bare `bun tools/splat/tune.ts` runs the current defaults.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { SplatOptimizer, cosine, LEGIBLE_INIT, LEGIBLE_LRS, LEGIBLE_G } from "../../src/splat/optimize";
import type { TrainPlan } from "../../src/clip/vision";
import { writePNG } from "./scene";

setupGlobals();

const num = (k: string, d: number) => (process.env[k] !== undefined ? Number(process.env[k]) : d);
const G = num("G", LEGIBLE_G);
const STEPS = num("STEPS", 20);
const SEED = num("SEED", 1);
const PROMPT = process.env.PROMPT ?? "a photo of a cat";
const TAG = process.env.TAG ?? "a";
const init = {
  scale: num("SCALE", LEGIBLE_INIT.scale),
  scaleJitter: num("SCALE_JITTER", LEGIBLE_INIT.scaleJitter),
  opacityRaw: num("OPACITY", LEGIBLE_INIT.opacityRaw),
  colorSpread: num("COLOR", LEGIBLE_INIT.colorSpread),
};
const lrs = {
  mean: num("LR_MEAN", LEGIBLE_LRS.mean),
  logScale: num("LR_SCALE", LEGIBLE_LRS.logScale),
  theta: num("LR_THETA", LEGIBLE_LRS.theta),
  color: num("LR_COLOR", LEGIBLE_LRS.color),
  opacity: num("LR_OP", LEGIBLE_LRS.opacity),
};

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = new Float32Array(readFileSync(join(MODEL_DIR, "weights_train.bin")).buffer);
const textAll = JSON.parse(readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")) as Record<string, number[]>;
const text = new Float32Array(textAll[PROMPT] ?? textAll["a photo of a cat"]);

const device: any = await (await (navigator as any).gpu.requestAdapter()).requestDevice();
console.log(`G=${G} steps=${STEPS} scale=${init.scale} op=${init.opacityRaw} color=${init.colorSpread}`);
console.log(`lrs=${JSON.stringify(lrs)}  prompt="${PROMPT}"`);

const opt = await SplatOptimizer.create(device, plan, weights, { G, seed: SEED, init, lrs });
opt.setPrompt(text);

const cos0 = cosine(await opt.currentEmbedding(), text);
for (let i = 0; i < 2; i++) opt.step(); // warm pipelines
// re-init so warmup steps don't count (fresh params, same seed)
opt.raster.setParams(
  (await import("../../src/splat/optimize")).randomSplats(G, SEED, init)
);
opt.raster.zeroAdamState();

for (let s = 1; s <= STEPS; s++) {
  opt.step();
  if (s % 5 === 0 || s === STEPS) {
    const c = cosine(await opt.currentEmbedding(), text);
    console.log(`  step ${String(s).padStart(3)}: cos ${c.toFixed(4)}  (Δ ${(c - cos0 >= 0 ? "+" : "")}${(c - cos0).toFixed(4)})`);
  }
}

const img = await opt.renderImage();
// legibility proxy: per-channel spatial stddev (higher = more structure, less
// flat/noise-averaged). Also print mean so we can see it's not all one colour.
const HW = 256 * 256;
let report = "";
for (let c = 0; c < 3; c++) {
  let m = 0;
  for (let i = 0; i < HW; i++) m += img[c * HW + i];
  m /= HW;
  let v = 0;
  for (let i = 0; i < HW; i++) v += (img[c * HW + i] - m) ** 2;
  report += `${["R", "G", "B"][c]} mean ${m.toFixed(3)} std ${Math.sqrt(v / HW).toFixed(3)}  `;
}
console.log(report);
const path = `/tmp/tune_${TAG}.png`;
writePNG(path, img, 256, 256);
console.log(`PNG: ${path}`);
