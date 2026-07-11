/** Integrated CLIP smoke for the experimental anisotropic optimizer. */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import type { TrainPlan } from "../../src/clip/vision";
import { Splat3DAnisotropicOptimizer } from "../../src/splat3d_aniso";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("aniso optimizer: no WebGPU adapter");
const device: GPUDevice = await adapter.requestDevice();
const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const bytes = readFileSync(join(MODEL_DIR, "weights_train.bin"));
const weights = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
const optimizer = await Splat3DAnisotropicOptimizer.create(device, plan, weights, {
  G: 256,
  cap: 256,
  seed: 7,
  viewSampler: "random",
  clipBatchSize: 3,
  convergence: {
    centerWeight: 0.002,
    radiusWeight: 0.004,
    targetRadius: 1.15,
    smallRadiusWeight: 0.02,
    smallRadius: 0.024,
    adaptiveRelocation: true,
    adaptationInterval: 1,
    adaptationFraction: 0.02,
    minRadius: 0.016,
    maxRadius: 0.16,
  },
});
optimizer.setViewPrompts(optimizer.cameras.map((_camera, view) => embedding(view, plan.textDim)));
const before = await optimizer.currentEmbedding(0);
optimizer.step(0, 3);
await device.queue.onSubmittedWorkDone();
const after = await optimizer.currentEmbedding(0);
const params = await optimizer.raster.readParams();
const telemetry = await optimizer.raster.readTileTelemetry();
let embeddingDelta = 0;
for (let i = 0; i < before.length; i++) embeddingDelta = Math.max(embeddingDelta, Math.abs(before[i] - after[i]));
let axisRatioSum = 0;
let rotatedSplats = 0;
for (let g = 0; g < optimizer.raster.dims.G; g++) {
  const scaleBase = 3 * optimizer.raster.dims.G + 3 * g;
  const minScale = Math.min(params[scaleBase], params[scaleBase + 1], params[scaleBase + 2]);
  const maxScale = Math.max(params[scaleBase], params[scaleBase + 1], params[scaleBase + 2]);
  axisRatioSum += Math.exp(maxScale - minScale);
  const quaternionBase = 6 * optimizer.raster.dims.G + 4 * g;
  if (Math.hypot(params[quaternionBase], params[quaternionBase + 1], params[quaternionBase + 2]) > 1e-3) {
    rotatedSplats++;
  }
}
const axisRatioMean = axisRatioSum / optimizer.raster.dims.G;
for (let g = 0; g < 4; g++) params[13 * optimizer.raster.dims.G + g] = -9;
optimizer.raster.setParams(params);
const adaptation = await optimizer.adaptSplatsIfDue(true);
console.log(
  `embedding delta=${embeddingDelta.toExponential(3)} tile=${telemetry.maxCount}/${telemetry.cap} ` +
  `axis-ratio=${axisRatioMean.toFixed(2)} rotated=${rotatedSplats}/${optimizer.raster.dims.G} ` +
  `adapt=${adaptation?.relocationCount ?? 0} batch=${optimizer.clipBatchSize}`
);
optimizer.destroy();

if (
  !params.every(Number.isFinite) ||
  !after.every(Number.isFinite) ||
  embeddingDelta <= 1e-8 ||
  telemetry.overflowTiles ||
  axisRatioMean < 1.5 ||
  rotatedSplats < 0.95 * optimizer.raster.dims.G ||
  optimizer.clipBatchSize !== 3 ||
  !adaptation ||
  adaptation.relocationCount < 1
) {
  throw new Error("GATE FAIL: anisotropic CLIP optimizer is invalid");
}
console.log("GATE PASS: anisotropic raster is connected to CLIP forward/backward and Adam.");

function embedding(seed: number, dim: number): Float32Array {
  const output = new Float32Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    const value = Math.sin((seed + 1) * 1.7 + i * 0.13);
    output[i] = value;
    norm += value * value;
  }
  const scale = 11 / Math.sqrt(norm);
  for (let i = 0; i < dim; i++) output[i] *= scale;
  return output;
}
