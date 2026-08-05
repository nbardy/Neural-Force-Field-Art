/**
 * Fair RGB-vs-Feature8 optimizer-step timing on one WebGPU device.
 *
 * Each sample restores the same initialized state, runs a fixed number of
 * complete optimizer steps, and waits for the shared queue. Samples alternate
 * ordering to reduce interference from background GPU work.
 *
 *   bun tools/splat/feature_speed_compare.ts [G=12000] [steps=10] [samples=3]
 *   INIT=density4-s5 bun tools/splat/feature_speed_compare.ts
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { SplatOptimizer, type SplatInit } from "../../src/splat/optimize";
import { FeaturePainterOptimizer } from "../../src/splat/feature_optimize";
import type { TrainPlan } from "../../src/clip/vision";

setupGlobals();

const SIDE = 256;
const G = Number(process.argv[2] ?? 12_000);
const STEPS = Number(process.argv[3] ?? 10);
const SAMPLES = Number(process.argv[4] ?? 3);
const INIT = process.env.INIT ?? "legacy";
const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = new Float32Array(readFileSync(join(MODEL_DIR, "weights_train.bin")).buffer);
const text = new Float32Array((JSON.parse(readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")) as Record<string, number[]>)["a photo of a cat"]);
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice({
  requiredFeatures: adapter.features.has("subgroups" as GPUFeatureName) ? ["subgroups" as GPUFeatureName] : [],
});

function logit(p: number): number { return Math.log(p / (1 - p)); }
function densityInit(scale: number): SplatInit {
  const targetDepth = 4;
  const alpha = Math.max(0.002, Math.min(0.95, targetDepth * SIDE * SIDE / (G * 2 * Math.PI * scale * scale)));
  return { scale, scaleJitter: 0.35, opacityRaw: logit(alpha), colorSpread: 1.2 };
}
const densityMatch = /^density4-s(\d+(?:\.\d+)?)$/.exec(INIT);
const init: SplatInit = INIT === "legacy"
  ? { scale: 9, scaleJitter: 0.35, opacityRaw: 0.4, colorSpread: 1.2 }
  : densityMatch
    ? densityInit(Number(densityMatch[1]))
    : (() => { throw new Error(`unknown INIT=${INIT}`); })();

async function warmRGB(): Promise<void> {
  const opt = await SplatOptimizer.create(device, plan, weights, { G, init, seed: 1 });
  opt.setPrompt(text);
  for (let i = 0; i < 3; i++) opt.step();
  await device.queue.onSubmittedWorkDone();
  opt.destroy();
}
async function warmFeature(): Promise<void> {
  const opt = await FeaturePainterOptimizer.create(device, plan, weights, { G, init, seed: 1 });
  opt.setPrompt(text);
  for (let i = 0; i < 3; i++) opt.step();
  await device.queue.onSubmittedWorkDone();
  opt.destroy();
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
}
function fmt(values: number[]): string {
  return `${values.map((value) => value.toFixed(1)).join(", ")} ms; median ${median(values).toFixed(1)} ms`;
}

await warmRGB();
await warmFeature();

const rgb = await SplatOptimizer.create(device, plan, weights, { G, init, seed: 1 });
const feature = await FeaturePainterOptimizer.create(device, plan, weights, { G, init, seed: 1 });
rgb.setPrompt(text);
feature.setPrompt(text);
const rgbInitial = await rgb.raster.readParams();
const featureInitial = {
  geometry: await feature.raster.readParams(),
  features: await feature.raster.readFeatureParams(),
  decoder: await feature.raster.readDecoderParams(),
};

async function timeRGB(): Promise<number> {
  rgb.raster.setParams(rgbInitial);
  rgb.raster.zeroAdamState();
  await device.queue.onSubmittedWorkDone();
  const started = performance.now();
  for (let step = 0; step < STEPS; step++) rgb.step();
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - started) / STEPS;
}
async function timeFeature(): Promise<number> {
  feature.raster.setParams(featureInitial.geometry);
  feature.raster.setFeatureParams(featureInitial.features);
  feature.raster.setDecoderParams(featureInitial.decoder);
  feature.raster.zeroAdamState();
  await device.queue.onSubmittedWorkDone();
  const started = performance.now();
  for (let step = 0; step < STEPS; step++) feature.step();
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - started) / STEPS;
}

const rgbTimes: number[] = [];
const featureTimes: number[] = [];
for (let sample = 0; sample < SAMPLES; sample++) {
  if ((sample & 1) === 0) {
    rgbTimes.push(await timeRGB());
    featureTimes.push(await timeFeature());
  } else {
    featureTimes.push(await timeFeature());
    rgbTimes.push(await timeRGB());
  }
}

const rgbMedian = median(rgbTimes);
const featureMedian = median(featureTimes);
console.log(`speed compare: G=${G}, init=${INIT}, ${STEPS} steps/sample, ${SAMPLES} alternating samples`);
console.log(`RGB:      ${fmt(rgbTimes)}`);
console.log(`Feature8: ${fmt(featureTimes)}`);
console.log(`Feature8 / RGB median: ${(featureMedian / rgbMedian).toFixed(2)}x`);
rgb.destroy();
feature.destroy();
