/**
 * Batch-major CLIP forward+backward bench.
 *
 * This verifies the actual optimizer-relevant path: dL/dpixels for a batch of
 * images/text embeddings using one batched dispatch list.
 *
 *   BATCH=3 RUNS=3 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
 *   BATCH=9 RUNS=2 WARMUP=2 bun tools/clip/batch_major_train_bench.ts
 *   FUSE_RESIDUAL_BWD_PW=1 BATCH=3 RUNS=3 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import { VisionTrainer, type TrainPlan } from "../../src/clip/vision";
import { BatchMajorVisionTrainer } from "../../src/clip/vision_batch";
import type { WeightPrecision } from "../../src/clip/vision_wgsl";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const BATCH = Number(process.env.BATCH ?? 3);
const RUNS = Number(process.env.RUNS ?? 3);
const WARMUP = Number(process.env.WARMUP ?? 3);
const SHARED_W_FWD_STEPS = parseStepSet(process.env.SHARED_W_FWD_STEPS ?? "");
const STEM_SPATIAL_BWD = process.env.STEM_SPATIAL_BWD === "1";
const FUSE_PW_GELU = process.env.FUSE_PW_GELU === "1";
const FUSE_GELU_BWD_PW = process.env.FUSE_GELU_BWD_PW === "1";
const FUSE_RESIDUAL_BWD_PW = process.env.FUSE_RESIDUAL_BWD_PW === "1";
const PRECISION: WeightPrecision = process.env.PRECISION === "f16" ? "f16" : "f32";
const WEIGHTS_FILE =
  process.env.WEIGHTS ?? (PRECISION === "f16" ? "weights_train_f16.bin" : "weights_train.bin");

function f32File(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
}

function f16File(path: string): Uint16Array {
  const b = readFileSync(path);
  return new Uint16Array(b.buffer, b.byteOffset, b.byteLength / 2).slice();
}

function parseStepSet(src: string): ReadonlySet<number> {
  const steps = src
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number(part))
    .filter((n) => Number.isFinite(n))
    .map((n) => n | 0);
  return new Set(steps);
}

function textEmbedding(seed: number, dim: number): Float32Array {
  const out = new Float32Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    const v = Math.sin((seed + 1) * 12.9898 + i * 78.233) * 0.5 +
      Math.cos((seed + 3) * 4.1414 + i * 17.17) * 0.5;
    out[i] = v;
    norm += v * v;
  }
  const scale = 11 / Math.sqrt(norm);
  for (let i = 0; i < dim; i++) out[i] *= scale;
  return out;
}

function inputVariant(base: Float32Array, lane: number): Float32Array {
  if (lane === 0) return base.slice();
  const out = new Float32Array(base.length);
  const shift = (lane % 7) * 0.003;
  for (let i = 0; i < base.length; i++) {
    const wave = Math.sin((i + 1) * (lane + 1) * 0.0001) * 0.002;
    out[i] = Math.min(1, Math.max(0, base[i] + shift + wave));
  }
  return out;
}

async function readback(
  device: GPUDevice,
  buf: GPUBuffer,
  floats: number,
  srcOffset = 0
): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: 1 | 8 });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(buf, srcOffset, staging, 0, floats * 4);
  device.queue.submit([e.finish()]);
  await staging.mapAsync(1);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

function compare(got: Float32Array, want: Float32Array): { maxDiff: number; rel: number; cos: number } {
  let maxDiff = 0;
  let scale = 1e-8;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < want.length; i++) {
    const a = got[i];
    const b = want[i];
    maxDiff = Math.max(maxDiff, Math.abs(a - b));
    scale = Math.max(scale, Math.abs(b));
    dot += a * b;
    na += a * a;
    nb += b * b;
  }
  return { maxDiff, rel: maxDiff / scale, cos: dot / Math.sqrt(na * nb) };
}

function finiteSummary(xs: Float32Array): string {
  let finite = 0;
  let maxAbs = 0;
  for (const x of xs) {
    if (Number.isFinite(x)) finite++;
    maxAbs = Math.max(maxAbs, Math.abs(x));
  }
  return `${finite}/${xs.length} finite, maxAbs=${maxAbs.toExponential(2)}`;
}

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const f16Supported = adapter.features.has("shader-f16");
if (PRECISION === "f16" && !f16Supported) {
  throw new Error("batch_major_train_bench: PRECISION=f16 requested but adapter lacks shader-f16");
}
const device: GPUDevice = await adapter.requestDevice({
  requiredFeatures: PRECISION === "f16" ? (["shader-f16"] as GPUFeatureName[]) : [],
});
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = PRECISION === "f16"
  ? f16File(join(MODEL_DIR, WEIGHTS_FILE))
  : f32File(join(MODEL_DIR, WEIGHTS_FILE));
const input = f32File(join(MODEL_DIR, "fixtures", "input_1x3x256x256.f32.bin"));
const inputs = Array.from({ length: BATCH }, (_unused, lane) => inputVariant(input, lane));
const texts = Array.from({ length: BATCH }, (_unused, lane) => textEmbedding(lane, plan.textDim));
const inputFloats = plan.inputShape[0] * plan.inputShape[1] * plan.inputShape[2];

console.log(
  `batch-major train: batch=${BATCH}, runs=${RUNS}, warmup=${WARMUP}, ` +
    `precision=${PRECISION}, weights=${WEIGHTS_FILE}, dispatches=${plan.steps.length}+${plan.backward.length}` +
    (SHARED_W_FWD_STEPS.size ? `, sharedWForwardSteps=${[...SHARED_W_FWD_STEPS].join(",")}` : "") +
    (STEM_SPATIAL_BWD ? `, stemSpatialBwd=1` : "") +
    (FUSE_PW_GELU ? `, fusePointwiseGeluForward=1` : "") +
    (FUSE_GELU_BWD_PW ? `, fuseGeluBwdIntoPw=1` : "") +
    (FUSE_RESIDUAL_BWD_PW ? `, fuseResidualBwdIntoPw=1` : "")
);

let t0 = performance.now();
const single = await VisionTrainer.create(device, plan, weights, {
  weightPrecision: PRECISION,
  stemSpatialBwd: STEM_SPATIAL_BWD,
  fuseGeluBwdIntoPw: FUSE_GELU_BWD_PW,
  fuseResidualBwdIntoPw: FUSE_RESIDUAL_BWD_PW,
});
console.log(`single compile+allocate: ${(performance.now() - t0).toFixed(0)} ms`);

single.writeInput(input);
single.writeText(texts[0]);
for (let i = 0; i < WARMUP; i++) {
  for (let lane = 0; lane < BATCH; lane++) single.run({ backward: true });
}
let sample = await readback(device, single.inputGradBuffer, 16);
console.log(`  separate warm: ${finiteSummary(sample)}`);
let cpu0 = performance.now();
for (let i = 0; i < RUNS; i++) {
  for (let lane = 0; lane < BATCH; lane++) single.run({ backward: true });
}
let cpuMs = (performance.now() - cpu0) / RUNS;
sample = await readback(device, single.inputGradBuffer, 16);
let wallMs = (performance.now() - cpu0) / RUNS;
console.log(`  separate final: ${finiteSummary(sample)}`);
console.log(
  `separate   : ${wallMs.toFixed(2)} ms/batch · ${(wallMs / BATCH).toFixed(2)} ms/image ` +
    `· CPU encode+submit ${cpuMs.toFixed(2)} ms/batch`
);

t0 = performance.now();
const batch = await BatchMajorVisionTrainer.create(device, plan, weights, BATCH, {
  weightPrecision: PRECISION,
  sharedWForwardSteps: SHARED_W_FWD_STEPS,
  stemSpatialBwd: STEM_SPATIAL_BWD,
  fusePointwiseGeluForward: FUSE_PW_GELU,
  fuseGeluBwdIntoPw: FUSE_GELU_BWD_PW,
  fuseResidualBwdIntoPw: FUSE_RESIDUAL_BWD_PW,
});
console.log(`batch compile+allocate : ${(performance.now() - t0).toFixed(0)} ms`);

for (let lane = 0; lane < BATCH; lane++) {
  batch.writeInput(lane, inputs[lane]);
  batch.writeText(lane, texts[lane]);
}
batch.run({ backward: true });
await readback(device, batch.inputGradBuffer, 16, batch.inputGradOffsetBytes(0));

let parityFailures = 0;
for (let lane = 0; lane < BATCH; lane++) {
  single.writeInput(inputs[lane]);
  single.writeText(texts[lane]);
  single.run({ backward: true });
  const got = await readback(device, batch.inputGradBuffer, inputFloats, batch.inputGradOffsetBytes(lane));
  const want = await readback(device, single.inputGradBuffer, inputFloats);
  const c = compare(got, want);
  const ok = c.rel < 2e-3 && c.cos > 0.999;
  if (!ok) parityFailures++;
  console.log(
    `${ok ? "PASS" : "FAIL"} grad parity lane ${lane}: cos=${c.cos.toFixed(6)} ` +
      `relLinf=${c.rel.toExponential(2)} maxDiff=${c.maxDiff.toExponential(2)}`
  );
}
if (parityFailures) process.exit(1);

single.slotBuffers.forEach((b) => b.destroy());
single.weightsBuffer.destroy();
single.textBuffer.destroy();

for (let lane = 0; lane < BATCH; lane++) {
  batch.writeInput(lane, input);
  batch.writeText(lane, texts[0]);
}
for (let i = 0; i < WARMUP; i++) batch.run({ backward: true });
sample = await readback(device, batch.inputGradBuffer, 16, batch.inputGradOffsetBytes(0));
console.log(`  batch-major warm: ${finiteSummary(sample)}`);
cpu0 = performance.now();
for (let i = 0; i < RUNS; i++) batch.run({ backward: true });
cpuMs = (performance.now() - cpu0) / RUNS;
sample = await readback(device, batch.inputGradBuffer, 16, batch.inputGradOffsetBytes(0));
wallMs = (performance.now() - cpu0) / RUNS;
console.log(`  batch-major final: ${finiteSummary(sample)}`);
console.log(
  `batch-major: ${wallMs.toFixed(2)} ms/batch · ${(wallMs / BATCH).toFixed(2)} ms/image ` +
    `· CPU encode+submit ${cpuMs.toFixed(2)} ms/batch`
);

batch.destroy();
