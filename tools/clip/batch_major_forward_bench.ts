/**
 * Batch-major CLIP forward bench.
 *
 * This is the first true batch-dimension test: each generated forward kernel
 * dispatches `workgroups.z = BATCH`, with activation slots laid out as
 * `[batch][slotFloats]`.
 *
 *   BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/batch_major_forward_bench.ts
 *   PLAN=plan_train.json BATCH=9 RUNS=3 WARMUP=2 bun tools/clip/batch_major_forward_bench.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import { VisionEncoder, type VisionPlan } from "../../src/clip/vision";
import { BatchMajorVisionEncoder } from "../../src/clip/vision_batch";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const BATCH = Number(process.env.BATCH ?? 3);
const RUNS = Number(process.env.RUNS ?? 10);
const WARMUP = Number(process.env.WARMUP ?? 3);
const PLAN_FILE = process.env.PLAN ?? "plan.json";
const WEIGHTS_FILE = process.env.WEIGHTS ?? (PLAN_FILE.includes("train") ? "weights_train.bin" : "weights.bin");

function f32File(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
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
  let scale = 1e-6;
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
const device: GPUDevice = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const plan: VisionPlan = JSON.parse(readFileSync(join(MODEL_DIR, PLAN_FILE), "utf8"));
const weights = f32File(join(MODEL_DIR, WEIGHTS_FILE));
const input = f32File(join(MODEL_DIR, "fixtures", "input_1x3x256x256.f32.bin"));
const inputs = Array.from({ length: BATCH }, (_unused, lane) => inputVariant(input, lane));

console.log(
  `batch-major forward: batch=${BATCH}, plan=${PLAN_FILE}, runs=${RUNS}, warmup=${WARMUP}, ` +
    `steps=${plan.steps.length}`
);

let t0 = performance.now();
const single = await VisionEncoder.create(device, plan, weights);
console.log(`single compile+allocate: ${(performance.now() - t0).toFixed(0)} ms`);

// Put the same image into every lane for the timing section so the standard
// encoder can benchmark B forwards without CPU upload cost in the timed loop.
single.writeInput(input);
for (let i = 0; i < WARMUP; i++) {
  for (let lane = 0; lane < BATCH; lane++) single.run();
}
let sample = await readback(device, single.outputBuffer, 16);
console.log(`  separate warm: ${finiteSummary(sample)}`);
let cpu0 = performance.now();
for (let i = 0; i < RUNS; i++) {
  for (let lane = 0; lane < BATCH; lane++) single.run();
}
let cpuMs = (performance.now() - cpu0) / RUNS;
sample = await readback(device, single.outputBuffer, 16);
let wallMs = (performance.now() - cpu0) / RUNS;
console.log(`  separate final: ${finiteSummary(sample)}`);
console.log(
  `separate   : ${wallMs.toFixed(2)} ms/batch · ${(wallMs / BATCH).toFixed(2)} ms/image ` +
    `· CPU encode+submit ${cpuMs.toFixed(2)} ms/batch`
);

t0 = performance.now();
const batch = await BatchMajorVisionEncoder.create(device, plan, weights, BATCH);
console.log(`batch compile+allocate : ${(performance.now() - t0).toFixed(0)} ms`);

for (let lane = 0; lane < BATCH; lane++) batch.writeInput(lane, inputs[lane]);
batch.run();
await readback(device, batch.outputBuffer, Math.min(16, plan.embedDim), batch.outputOffsetBytes(0));

let parityFailures = 0;
for (let lane = 0; lane < BATCH; lane++) {
  single.writeInput(inputs[lane]);
  single.run();
  const got = await readback(device, batch.outputBuffer, plan.embedDim, batch.outputOffsetBytes(lane));
  const want = await readback(device, single.outputBuffer, plan.embedDim);
  const c = compare(got, want);
  const ok = c.rel < 2e-3 && c.cos > 0.999;
  if (!ok) parityFailures++;
  console.log(
    `${ok ? "PASS" : "FAIL"} parity lane ${lane}: cos=${c.cos.toFixed(6)} ` +
      `relLinf=${c.rel.toExponential(2)} maxDiff=${c.maxDiff.toExponential(2)}`
  );
}
if (parityFailures) process.exit(1);

single.slotBuffers.forEach((b) => b.destroy());
single.weightsBuffer.destroy();
for (let lane = 0; lane < BATCH; lane++) batch.writeInput(lane, input);

for (let i = 0; i < WARMUP; i++) batch.run();
sample = await readback(device, batch.outputBuffer, 16, batch.outputOffsetBytes(0));
console.log(`  batch-major warm: ${finiteSummary(sample)}`);
cpu0 = performance.now();
for (let i = 0; i < RUNS; i++) batch.run();
cpuMs = (performance.now() - cpu0) / RUNS;
sample = await readback(device, batch.outputBuffer, 16, batch.outputOffsetBytes(0));
wallMs = (performance.now() - cpu0) / RUNS;
console.log(`  batch-major final: ${finiteSummary(sample)}`);
console.log(
  `batch-major: ${wallMs.toFixed(2)} ms/batch · ${(wallMs / BATCH).toFixed(2)} ms/image ` +
    `· CPU encode+submit ${cpuMs.toFixed(2)} ms/batch`
);

batch.destroy();
