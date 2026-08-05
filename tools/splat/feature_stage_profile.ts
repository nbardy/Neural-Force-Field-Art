/**
 * GPU-timestamp profile for the complete Feature8 optimization step.
 *
 *   INIT=density4-s5 RUNS=3 bun tools/splat/feature_stage_profile.ts [G=12000]
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { FeaturePainterOptimizer } from "../../src/splat/feature_optimize";
import { LEGIBLE_LRS, type SplatInit } from "../../src/splat/optimize";
import type { TrainPlan } from "../../src/clip/vision";
import type { FeaturePainterTimestampWrites } from "../../src/splat/feature_painter";

setupGlobals();

const SIDE = 256;
const IMG_BYTES = 3 * SIDE * SIDE * 4;
const G = Number(process.argv[2] ?? 12_000);
const RUNS = Number(process.env.RUNS ?? 3);
const INIT = process.env.INIT ?? "density4-s5";
const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const plan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8")) as TrainPlan;
const weightBytes = readFileSync(join(MODEL_DIR, "weights_train.bin"));
const weights = new Float32Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength / 4).slice();
const text = new Float32Array((JSON.parse(readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")) as Record<string, number[]>)["a photo of a cat"]);

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("feature_stage_profile: no WebGPU adapter");
const requiredFeatures: GPUFeatureName[] = ["subgroups", "timestamp-query"].filter((feature) => adapter.features.has(feature));
if (requiredFeatures.length !== 2) throw new Error("feature_stage_profile: adapter must support subgroups and timestamp-query");
const device = await adapter.requestDevice({ requiredFeatures });

function logit(p: number): number { return Math.log(p / (1 - p)); }
function densityInit(scale: number): SplatInit {
  const targetDepth = 4;
  const alpha = Math.max(0.002, Math.min(0.95, targetDepth * SIDE * SIDE / (G * 2 * Math.PI * scale * scale)));
  return { scale, scaleJitter: 0.35, opacityRaw: logit(alpha), colorSpread: 1.2 };
}
const densityMatch = /^density4-s(\d+(?:\.\d+)?)$/.exec(INIT);
const init: SplatInit = densityMatch
  ? densityInit(Number(densityMatch[1]))
  : INIT === "legacy"
    ? { scale: 9, scaleJitter: 0.35, opacityRaw: 0.4, colorSpread: 1.2 }
    : (() => { throw new Error(`unknown INIT=${INIT}`); })();

const opt = await FeaturePainterOptimizer.create(device, plan, weights, { G, init, seed: 1 });
opt.setPrompt(text);
for (let i = 0; i < 3; i++) opt.step();
await device.queue.onSubmittedWorkDone();

const querySet = device.createQuerySet({ type: "timestamp", count: 8 });
const resolve = device.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const read = device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const writes = (begin: number, end: number): FeaturePainterTimestampWrites => ({
  querySet,
  beginningOfPassWriteIndex: begin,
  endOfPassWriteIndex: end,
});
const rows: number[][] = [];

for (let run = 0; run < RUNS; run++) {
  const encoder = device.createCommandEncoder();
  opt.raster.recordForward(encoder, writes(0, 1));
  encoder.copyBufferToBuffer(opt.raster.image, 0, opt.trainer.inputBuffer, 0, IMG_BYTES);
  opt.trainer.encode(encoder, { backward: true, timestampWrites: writes(2, 3) });
  encoder.copyBufferToBuffer(opt.trainer.inputGradBuffer, 0, opt.raster.gradImage, 0, IMG_BYTES);
  opt.raster.recordBackward(encoder, writes(4, 5));
  opt.raster.recordAdam(encoder, run + 4, LEGIBLE_LRS, undefined, writes(6, 7));
  encoder.resolveQuerySet(querySet, 0, 8, resolve, 0);
  encoder.copyBufferToBuffer(resolve, 0, read, 0, 64);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const timestamps = new BigUint64Array(read.getMappedRange().slice(0));
  read.unmap();
  rows.push([0, 2, 4, 6].map((index) => Number(timestamps[index + 1] - timestamps[index]) / 1e6));
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}
const names = ["raster fwd", "CLIP", "raster bwd", "Adam"];
const medians = names.map((_, index) => median(rows.map((row) => row[index])));
const total = medians.reduce((sum, value) => sum + value, 0);
console.log(`feature timestamp profile: G=${G}, init=${INIT}, ${RUNS} GPU samples`);
for (let run = 0; run < rows.length; run++) {
  console.log(`sample ${String(run + 1).padStart(2, " ")}  ${rows[run].map((value) => value.toFixed(2)).join(" / ")} ms`);
}
for (let i = 0; i < names.length; i++) {
  console.log(`${names[i].padEnd(11)} ${medians[i].toFixed(2)} ms  ${(100 * medians[i] / total).toFixed(1)}%`);
}
console.log(`${"total".padEnd(11)} ${total.toFixed(2)} ms`);

querySet.destroy();
resolve.destroy();
read.destroy();
opt.destroy();
