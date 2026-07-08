/**
 * Isolated CLIP batch scheduling bench.
 *
 * This measures the current "replicated activations, shared weights" batcher
 * before doing the larger true batch-major shader fork.
 *
 *   BATCH=3 MODE=backward bun tools/clip/batch_bench.ts
 *   BATCH=9 MODE=forward  RUNS=10 WARMUP=3 bun tools/clip/batch_bench.ts
 *
 * Schedules:
 *   separate   : one submit per lane, full CLIP each time
 *   lane-major : one submit, lane 0 full CLIP, lane 1 full CLIP, ...
 *   step-major : one submit, CLIP step 0 for all lanes, CLIP step 1 for all lanes, ...
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import { ReplicatedBatchVisionTrainer, type BatchSchedule } from "../../src/clip/vision_batch";
import type { TrainPlan } from "../../src/clip/vision";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const BATCH = Number(process.env.BATCH ?? 3);
const RUNS = Number(process.env.RUNS ?? 8);
const WARMUP = Number(process.env.WARMUP ?? 2);
const MODE = process.env.MODE === "forward" ? "forward" : "backward";
const BACKWARD = MODE !== "forward";

function f32File(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
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
  // Keep the loss numerically boring and nonzero. CLIP text embeddings in the
  // app are not normalized, so scale back to a similar L2 magnitude.
  const scale = 11 / Math.sqrt(norm);
  for (let i = 0; i < dim; i++) out[i] *= scale;
  return out;
}

async function readback(
  device: GPUDevice,
  buf: GPUBuffer,
  floats: number
): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: 1 | 8 });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
  device.queue.submit([e.finish()]);
  await staging.mapAsync(1);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
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

const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = f32File(join(MODEL_DIR, "weights_train.bin"));
const input = f32File(join(MODEL_DIR, "fixtures", "input_1x3x256x256.f32.bin"));

console.log(
  `batch bench: batch=${BATCH}, mode=${MODE}, runs=${RUNS}, warmup=${WARMUP}, ` +
    `dispatches=${plan.steps.length}+${plan.backward.length}`
);

const t0 = performance.now();
const clip = await ReplicatedBatchVisionTrainer.create(device, plan, weights, BATCH);
console.log(`compile+allocate: ${(performance.now() - t0).toFixed(0)} ms`);

for (let lane = 0; lane < BATCH; lane++) {
  clip.writeInput(lane, input);
  clip.writeText(lane, textEmbedding(lane, plan.textDim));
}

async function syncAndCheck(label: string): Promise<void> {
  const out = await readback(
    device,
    BACKWARD ? clip.inputGradBuffer(0) : clip.outputBuffer(0),
    BACKWARD ? 16 : Math.min(16, plan.embedDim)
  );
  console.log(`  ${label}: ${finiteSummary(out)}`);
}

async function benchSeparate(): Promise<void> {
  for (let i = 0; i < WARMUP; i++) {
    for (let lane = 0; lane < BATCH; lane++) clip.runLane(lane, { backward: BACKWARD });
  }
  await syncAndCheck("separate warm");
  const cpu0 = performance.now();
  for (let i = 0; i < RUNS; i++) {
    for (let lane = 0; lane < BATCH; lane++) clip.runLane(lane, { backward: BACKWARD });
  }
  const cpuMs = (performance.now() - cpu0) / RUNS;
  await syncAndCheck("separate final");
  const wallMs = (performance.now() - cpu0) / RUNS;
  console.log(
    `separate  : ${wallMs.toFixed(2)} ms/batch · ${(wallMs / BATCH).toFixed(2)} ms/image ` +
      `· CPU encode+submit ${cpuMs.toFixed(2)} ms/batch`
  );
}

async function benchBatch(schedule: BatchSchedule): Promise<void> {
  for (let i = 0; i < WARMUP; i++) clip.run({ backward: BACKWARD, schedule });
  await syncAndCheck(`${schedule} warm`);
  const cpu0 = performance.now();
  for (let i = 0; i < RUNS; i++) clip.run({ backward: BACKWARD, schedule });
  const cpuMs = (performance.now() - cpu0) / RUNS;
  await syncAndCheck(`${schedule} final`);
  const wallMs = (performance.now() - cpu0) / RUNS;
  console.log(
    `${schedule.padEnd(10)}: ${wallMs.toFixed(2)} ms/batch · ${(wallMs / BATCH).toFixed(2)} ms/image ` +
      `· CPU encode+submit ${cpuMs.toFixed(2)} ms/batch`
  );
}

await benchSeparate();
await benchBatch("lane-major");
await benchBatch("step-major");

clip.destroy();
