/**
 * Compare density-aware 2D splat initializers under equal RGB and Feature8
 * optimization. Run serially on one device:
 *
 *   bun tools/splat/init_sweep.ts [G=12000] [steps=20]
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { SplatOptimizer, cosine, type SplatInit } from "../../src/splat/optimize";
import { FeaturePainterOptimizer } from "../../src/splat/feature_optimize";
import type { TrainPlan } from "../../src/clip/vision";

setupGlobals();

const SIDE = 256;
const G = Number(process.argv[2] ?? 12_000);
const STEPS = Number(process.argv[3] ?? 20);
const SEED = Number(process.env.SEED ?? 1);
const ONLY_CASE = process.env.CASE ?? "all";
const MODE = process.env.MODE ?? "both";
const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = new Float32Array(readFileSync(join(MODEL_DIR, "weights_train.bin")).buffer);
const text = new Float32Array((JSON.parse(readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")) as Record<string, number[]>)["a photo of a cat"]);
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice({
  requiredFeatures: adapter.features.has("subgroups" as GPUFeatureName) ? ["subgroups" as GPUFeatureName] : [],
});

function logit(p: number): number {
  return Math.log(p / (1 - p));
}

/** Approximate uniform-cloud optical depth: G * alpha * 2*pi*s^2 / HW. */
function opacityForOpticalDepth(splatCount: number, scale: number, depth: number): number {
  const alpha = Math.max(0.002, Math.min(0.95, depth * SIDE * SIDE / (splatCount * 2 * Math.PI * scale * scale)));
  return logit(alpha);
}

const base: Required<SplatInit> = {
  scale: 9,
  scaleJitter: 0.35,
  opacityRaw: 0.4,
  colorSpread: 1.2,
};
const cases: Array<{ name: string; init: SplatInit }> = [
  { name: "legacy-s9-a0.60", init: base },
  {
    name: "density4-s9",
    init: { ...base, opacityRaw: opacityForOpticalDepth(G, 9, 4) },
  },
  {
    name: "density4-s5",
    init: { ...base, scale: 5, opacityRaw: opacityForOpticalDepth(G, 5, 4) },
  },
  {
    name: "density4-s4",
    init: { ...base, scale: 4, opacityRaw: opacityForOpticalDepth(G, 4, 4) },
  },
  {
    name: "density4-s3",
    init: { ...base, scale: 3, opacityRaw: opacityForOpticalDepth(G, 3, 4) },
  },
];

async function warmRGB(): Promise<void> {
  const opt = await SplatOptimizer.create(device, plan, weights, { G, seed: SEED });
  opt.setPrompt(text);
  for (let i = 0; i < 3; i++) opt.step();
  await device.queue.onSubmittedWorkDone();
  opt.destroy();
}

async function warmFeature(): Promise<void> {
  const opt = await FeaturePainterOptimizer.create(device, plan, weights, { G, seed: SEED });
  opt.setPrompt(text);
  for (let i = 0; i < 3; i++) opt.step();
  await device.queue.onSubmittedWorkDone();
  opt.destroy();
}

async function runRGB(name: string, init: SplatInit): Promise<void> {
  const opt = await SplatOptimizer.create(device, plan, weights, { G, seed: SEED, init });
  opt.setPrompt(text);
  const before = cosine(await opt.currentEmbedding(), text);
  for (let i = 0; i < STEPS; i++) opt.step();
  await device.queue.onSubmittedWorkDone();
  const after = cosine(await opt.currentEmbedding(), text);
  await opt.renderImage();
  const tile = await opt.raster.readTileTelemetry();
  console.log(`RGB      ${name.padEnd(18)} cos ${before.toFixed(5)} -> ${after.toFixed(5)}  bins ${tile.meanCount.toFixed(1)}  stop ${tile.meanStop.toFixed(1)}  overflow ${tile.overflowTiles}`);
  opt.destroy();
}

async function runFeature(name: string, init: SplatInit): Promise<void> {
  const opt = await FeaturePainterOptimizer.create(device, plan, weights, { G, seed: SEED, init });
  opt.setPrompt(text);
  const before = cosine(await opt.currentEmbedding(), text);
  for (let i = 0; i < STEPS; i++) opt.step();
  await device.queue.onSubmittedWorkDone();
  const after = cosine(await opt.currentEmbedding(), text);
  await opt.renderImage();
  const tile = await opt.raster.readTileTelemetry();
  console.log(`Feature8 ${name.padEnd(18)} cos ${before.toFixed(5)} -> ${after.toFixed(5)}  bins ${tile.meanCount.toFixed(1)}  stop ${tile.meanStop.toFixed(1)}  overflow ${tile.overflowTiles}`);
  opt.destroy();
}

const selectedCases = ONLY_CASE === "all" ? cases : cases.filter((testCase) => testCase.name === ONLY_CASE);
if (selectedCases.length === 0) throw new Error(`unknown CASE=${ONLY_CASE}`);
if (MODE !== "rgb" && MODE !== "feature" && MODE !== "both") throw new Error(`unknown MODE=${MODE}`);

if (MODE === "rgb" || MODE === "both") await warmRGB();
if (MODE === "feature" || MODE === "both") await warmFeature();
console.log(`init sweep: G=${G}, seed=${SEED}, mode=${MODE}, true optimizer steps=${STEPS}`);
for (const testCase of selectedCases) {
  if (MODE === "rgb" || MODE === "both") await runRGB(testCase.name, testCase.init);
  if (MODE === "feature" || MODE === "both") await runFeature(testCase.name, testCase.init);
}
