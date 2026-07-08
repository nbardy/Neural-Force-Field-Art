/**
 * Shared-W pointwise batch microbench.
 *
 * Compares two compact pointwise kernels for one CLIP ConvStep:
 *   z-batch  : workgroups.z = B, each lane stages W independently
 *   shared-W : local_invocation_id.z = B, one workgroup stages W once
 *
 *   BATCH=3 STEP_INDEX=57 RUNS=30 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
 *   BATCH=2 STEP_INDEX=8  RUNS=30 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import type { ConvStep, DispatchSpec, VisionPlan } from "../../src/clip/vision_wgsl";
import {
  pointwiseSharedWBatchDispatch,
  pointwiseZBatchDispatch,
} from "../../src/clip/vision_batch_pointwise";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const BATCH = Number(process.env.BATCH ?? 3);
const RUNS = Number(process.env.RUNS ?? 30);
const WARMUP = Number(process.env.WARMUP ?? 10);
const STEP_INDEX = Number(process.env.STEP_INDEX ?? 57);

const USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };

function f32File(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
}

function data(n: number, seed: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.sin((i + 1) * 0.013 + seed * 1.7) * 0.5 +
      Math.cos((i + 3) * 0.037 + seed * 0.31) * 0.25;
  }
  return out;
}

async function pipeline(device: GPUDevice, spec: DispatchSpec): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: spec.code });
  const pipe = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const err = await device.popErrorScope();
  if (err) throw new Error(`${spec.label}: ${err.message}\n${spec.code}`);
  return pipe;
}

function buffer(device: GPUDevice, floats: Float32Array | number): GPUBuffer {
  const n = typeof floats === "number" ? floats : floats.length;
  const b = device.createBuffer({
    size: Math.max(4, n * 4),
    usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
  });
  if (floats instanceof Float32Array) {
    device.queue.writeBuffer(b, 0, floats as unknown as BufferSource);
  }
  return b;
}

async function readback(device: GPUDevice, buf: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: USAGE.MAP_READ | USAGE.COPY_DST });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
  device.queue.submit([e.finish()]);
  await staging.mapAsync(USAGE.MAP_READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

function compare(got: Float32Array, want: Float32Array): { maxDiff: number; rel: number } {
  let maxDiff = 0;
  let scale = 1e-6;
  for (let i = 0; i < want.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(got[i] - want[i]));
    scale = Math.max(scale, Math.abs(want[i]));
  }
  return { maxDiff, rel: maxDiff / scale };
}

function encodeRun(
  device: GPUDevice,
  pipe: GPUComputePipeline,
  bind: GPUBindGroup,
  spec: DispatchSpec
): void {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(...spec.workgroups);
  pass.end();
  device.queue.submit([enc.finish()]);
}

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: GPUDevice = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const plan: VisionPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const step = plan.steps[STEP_INDEX] as ConvStep;
if (!step || step.kind !== "conv" || step.variant !== "pointwise") {
  throw new Error(`STEP_INDEX=${STEP_INDEX} is not a pointwise conv step`);
}
const weights = f32File(join(MODEL_DIR, "weights_train.bin"));
const P = step.outH * step.outW;
const srcFloats = BATCH * step.cin * P;
const dstFloats = BATCH * step.cout * P;
const hasRes = step.residual !== null;

console.log(
  `pointwise batch: B=${BATCH}, step=${STEP_INDEX}, ${step.cin}->${step.cout} ` +
    `@${step.outH}x${step.outW}, residual=${hasRes}, runs=${RUNS}, warmup=${WARMUP}`
);

const src = data(srcFloats, 1);
const res = hasRes ? data(dstFloats, 2) : new Float32Array(1);
const zDst = new Float32Array(dstFloats);
const sharedDst = new Float32Array(dstFloats);

const weightsBuf = buffer(device, weights);
const srcBuf = buffer(device, src);
const resBuf = buffer(device, res);
const zDstBuf = buffer(device, zDst);
const sharedDstBuf = buffer(device, sharedDst);

const zSpec = pointwiseZBatchDispatch(step, BATCH);
const sharedSpec = pointwiseSharedWBatchDispatch(step, BATCH);
const zPipe = await pipeline(device, zSpec);
const sharedPipe = await pipeline(device, sharedSpec);

function bindFor(pipe: GPUComputePipeline, dst: GPUBuffer, spec: DispatchSpec): GPUBindGroup {
  const entries = [
    { binding: 0, resource: { buffer: weightsBuf } },
    { binding: 1, resource: { buffer: srcBuf } },
    { binding: 2, resource: { buffer: dst } },
  ];
  if (hasRes) entries.push({ binding: 3, resource: { buffer: resBuf } });
  return device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: entries.slice(0, spec.buffers.length),
  });
}

const zBind = bindFor(zPipe, zDstBuf, zSpec);
const sharedBind = bindFor(sharedPipe, sharedDstBuf, sharedSpec);

encodeRun(device, zPipe, zBind, zSpec);
encodeRun(device, sharedPipe, sharedBind, sharedSpec);
let got = await readback(device, sharedDstBuf, dstFloats);
let want = await readback(device, zDstBuf, dstFloats);
let cmp = compare(got, want);
console.log(`parity: relLinf=${cmp.rel.toExponential(2)} maxDiff=${cmp.maxDiff.toExponential(2)}`);
if (cmp.rel > 1e-6) process.exit(1);

for (let i = 0; i < WARMUP; i++) encodeRun(device, zPipe, zBind, zSpec);
await readback(device, zDstBuf, 4);
let t0 = performance.now();
for (let i = 0; i < RUNS; i++) encodeRun(device, zPipe, zBind, zSpec);
let cpuMs = (performance.now() - t0) / RUNS;
await readback(device, zDstBuf, 4);
let wallMs = (performance.now() - t0) / RUNS;
console.log(`z-batch : ${wallMs.toFixed(3)} ms · CPU encode+submit ${cpuMs.toFixed(3)} ms`);

for (let i = 0; i < WARMUP; i++) encodeRun(device, sharedPipe, sharedBind, sharedSpec);
await readback(device, sharedDstBuf, 4);
t0 = performance.now();
for (let i = 0; i < RUNS; i++) encodeRun(device, sharedPipe, sharedBind, sharedSpec);
cpuMs = (performance.now() - t0) / RUNS;
await readback(device, sharedDstBuf, 4);
wallMs = (performance.now() - t0) / RUNS;
console.log(`shared-W: ${wallMs.toFixed(3)} ms · CPU encode+submit ${cpuMs.toFixed(3)} ms`);

got = await readback(device, sharedDstBuf, dstFloats);
want = await readback(device, zDstBuf, dstFloats);
cmp = compare(got, want);
console.log(`final parity: relLinf=${cmp.rel.toExponential(2)} maxDiff=${cmp.maxDiff.toExponential(2)}`);
if (cmp.rel > 1e-6) process.exit(1);

for (const b of [weightsBuf, srcBuf, resBuf, zDstBuf, sharedDstBuf]) b.destroy();
