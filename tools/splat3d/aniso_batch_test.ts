/** Batch-3 parity and throughput gate for the live anisotropic optimizer. */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import type { TrainPlan } from "../../src/clip/vision";
import { Splat3DAnisotropicOptimizer } from "../../src/splat3d_aniso";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("aniso batch: no WebGPU adapter");
const device = await adapter.requestDevice();
const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const bytes = readFileSync(join(MODEL_DIR, "weights_train.bin"));
const weights = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
const G = Math.max(1, Number(process.env.G ?? 512) | 0);
const STEPS = Math.max(1, Number(process.env.STEPS ?? 4) | 0);
let cap = 256;
while (cap < G && cap < 4096) cap *= 2;
const common = { G, cap, seed: 17, viewSampler: "epoch" as const };

const single = await Splat3DAnisotropicOptimizer.create(device, plan, weights, {
  ...common,
  clipBatchSize: 1,
});
const batched = await Splat3DAnisotropicOptimizer.create(device, plan, weights, {
  ...common,
  clipBatchSize: 3,
});
const prompts = single.cameras.map((_camera, view) => embedding(view, plan.textDim));
single.setViewPrompts(prompts);
batched.setViewPrompts(prompts);

single.step(0, 3);
await device.queue.onSubmittedWorkDone();
batched.step(0, 3);
await device.queue.onSubmittedWorkDone();
const [singleParams, batchParams] = await Promise.all([
  single.raster.readParams(),
  batched.raster.readParams(),
]);
let maxAbs = 0;
let meanAbs = 0;
for (let i = 0; i < singleParams.length; i++) {
  const delta = Math.abs(singleParams[i] - batchParams[i]);
  maxAbs = Math.max(maxAbs, delta);
  meanAbs += delta;
}
meanAbs /= singleParams.length;

const singleMs = await timedSteps(single, STEPS);
const batchMs = await timedSteps(batched, STEPS);
const profile = await batched.profileStep(0, 3);
console.log(
  `params max=${maxAbs.toExponential(3)} mean=${meanAbs.toExponential(3)} ` +
  `single=${singleMs.toFixed(2)}ms batch3=${batchMs.toFixed(2)}ms speedup=${(singleMs / batchMs).toFixed(2)}x`
);
console.log(
  `profile total=${profile.total.toFixed(2)}ms fwd=${profile.rasterFwd.toFixed(2)}ms ` +
  `clip=${profile.clipBatch.toFixed(2)}ms replay=${profile.rasterReplay.toFixed(2)}ms ` +
  `bwd=${profile.rasterBwd.toFixed(2)}ms reg=${profile.regularizer.toFixed(2)}ms ` +
  `adam=${profile.adam.toFixed(2)}ms display=${profile.display.toFixed(2)}ms`
);

single.destroy();
batched.destroy();

if (!singleParams.every(Number.isFinite) || !batchParams.every(Number.isFinite)) {
  throw new Error("GATE FAIL: non-finite anisotropic batch parameters");
}
if (maxAbs > 2e-3 || meanAbs > 2e-5) {
  throw new Error(`GATE FAIL: batch-3 parameter parity max=${maxAbs} mean=${meanAbs}`);
}
if (!(profile.clipBatch > 0 && profile.rasterFwd > 0 && profile.rasterReplay > 0 && profile.rasterBwd > 0)) {
  throw new Error("GATE FAIL: anisotropic profile did not split batch/raster phases");
}
console.log("GATE PASS: anisotropic batch-3 matches three single CLIP passes.");

async function timedSteps(optimizer: Splat3DAnisotropicOptimizer, count: number): Promise<number> {
  optimizer.step(0, 3);
  await device.queue.onSubmittedWorkDone();
  const start = performance.now();
  for (let step = 0; step < count; step++) {
    optimizer.step(0, 3);
    await device.queue.onSubmittedWorkDone();
  }
  return (performance.now() - start) / count;
}

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
