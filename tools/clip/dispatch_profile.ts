/**
 * Isolated CLIP dispatch profiler.
 *
 * This times each generated WGSL dispatch with warmed split submits. It is not a
 * perfect full-chain GPU timestamp profile, but it gives a concrete kernel
 * ranking before we attempt shared-W pointwise, spatial_bwd, f16, or fusion.
 *
 *   MODE=train BATCH=1 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
 *   MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
 *   MODE=forward PLAN=plan.json RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts
 *   CSV=1 MODE=train BATCH=1 bun tools/clip/dispatch_profile.ts > /tmp/clip.csv
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import {
  planDispatches,
  type BufferRef,
  type DispatchSpec,
  type VisionPlan,
} from "../../src/clip/vision_wgsl";
import { planBwdDispatches, type TrainPlan } from "../../src/clip/vision_bwd_wgsl";
import { batchForwardDispatches, batchTrainDispatches } from "../../src/clip/vision_batch_wgsl";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const MODE = (process.env.MODE ?? "train").toLowerCase();
const BATCH = Number(process.env.BATCH ?? 1);
const RUNS = Number(process.env.RUNS ?? 3);
const WARMUP = Number(process.env.WARMUP ?? 1);
const CSV = process.env.CSV === "1";
const STEM_SPATIAL_BWD = process.env.STEM_SPATIAL_BWD === "1";
const PLAN_FILE =
  process.env.PLAN ?? (MODE === "forward" ? "plan.json" : "plan_train.json");
const WEIGHTS_FILE = process.env.WEIGHTS ?? (PLAN_FILE.includes("train") ? "weights_train.bin" : "weights.bin");

interface Row {
  index: number;
  phase: "forward" | "backward";
  group: string;
  label: string;
  workgroups: [number, number, number];
  ms: number;
}

function f32File(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
}

async function makePipeline(device: GPUDevice, spec: DispatchSpec): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: spec.code });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const err = await device.popErrorScope();
  if (err) throw new Error(`dispatch_profile: '${spec.label}' invalid: ${err.message}\n${spec.code}`);
  return pipeline;
}

function groupLabel(label: string): string {
  if (label.startsWith("pw_bwd")) return "pw_bwd";
  if (label.startsWith("pw ")) return "pw";
  if (label.startsWith("spatial_bwd")) return "spatial_bwd";
  if (label.startsWith("conv")) return "conv";
  if (label.startsWith("gelu_bwd")) return "gelu_bwd";
  if (label.startsWith("gelu ")) return "gelu";
  if (label.startsWith("residual_bwd")) return "residual_bwd";
  if (label.startsWith("attn_core_bwd")) return "attn_core_bwd";
  if (label.startsWith("attn.core")) return "attn.core";
  if (label.startsWith("se_bwd")) return "se_bwd";
  if (label.startsWith("se ")) return "se";
  if (label.startsWith("head_bwd")) return "head_bwd";
  if (label.startsWith("head ")) return "head";
  if (label.startsWith("loss_bwd")) return "loss_bwd";
  return label.split(/\s+/)[0] || "other";
}

function median(xs: number[]): number {
  const ys = xs.slice().sort((a, b) => a - b);
  return ys[Math.floor(ys.length / 2)] ?? 0;
}

function sumWorkgroups(w: [number, number, number]): number {
  return w[0] * w[1] * w[2];
}

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: GPUDevice = await adapter.requestDevice();
const info = adapter.info ?? {};
if (!CSV) console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const plan = JSON.parse(readFileSync(join(MODEL_DIR, PLAN_FILE), "utf8")) as TrainPlan;
const weights = f32File(join(MODEL_DIR, WEIGHTS_FILE));
if (weights.length !== plan.weightsFloats) {
  throw new Error(`weights ${weights.length} != plan ${plan.weightsFloats}`);
}

let specs: DispatchSpec[];
let fwdCount: number;
if (BATCH > 1) {
  if (MODE === "forward") {
    specs = batchForwardDispatches(plan as VisionPlan, BATCH);
    fwdCount = specs.length;
  } else {
    const out = batchTrainDispatches(plan, BATCH, { stemSpatialBwd: STEM_SPATIAL_BWD });
    specs = out.specs;
    fwdCount = out.fwdCount;
  }
} else if (MODE === "forward") {
  specs = planDispatches(plan);
  fwdCount = specs.length;
} else {
  const fwd = planDispatches(plan);
  const bwd = planBwdDispatches(plan, { stemSpatialBwd: STEM_SPATIAL_BWD });
  specs = [...fwd, ...bwd];
  fwdCount = fwd.length;
}

const weightsBuffer = device.createBuffer({
  label: "dispatch-profile-weights",
  size: weights.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(weightsBuffer, 0, weights as unknown as BufferSource);

const textBuffer = device.createBuffer({
  label: "dispatch-profile-text",
  size: plan.textDim * Math.max(1, BATCH) * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const text = new Float32Array(plan.textDim * Math.max(1, BATCH));
for (let i = 0; i < text.length; i++) text[i] = Math.sin(i * 0.017) * 0.1;
device.queue.writeBuffer(textBuffer, 0, text as unknown as BufferSource);

const slotBuffers = plan.slots.map((floats, slot) => {
  const data = new Float32Array(floats * Math.max(1, BATCH));
  for (let i = 0; i < data.length; i += 97) data[i] = Math.sin((slot + 1) * 0.13 + i * 0.001) * 0.01;
  const buf = device.createBuffer({
    label: `dispatch-profile-slot-${slot}`,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buf, 0, data as unknown as BufferSource);
  return buf;
});

function resolve(ref: BufferRef): GPUBuffer {
  if (ref.kind === "weights") return weightsBuffer;
  if (ref.kind === "text") return textBuffer;
  return slotBuffers[ref.slot];
}

if (!CSV) {
  console.log(
    `dispatch profile: mode=${MODE}, plan=${PLAN_FILE}, batch=${BATCH}, ` +
      `dispatches=${specs.length}, runs=${RUNS}, warmup=${WARMUP}` +
      (STEM_SPATIAL_BWD ? `, stemSpatialBwd=1` : "")
  );
}

const rows: Row[] = [];
for (let i = 0; i < specs.length; i++) {
  const spec = specs[i];
  const pipeline = await makePipeline(device, spec);
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: spec.buffers.map((ref, binding) => ({ binding, resource: { buffer: resolve(ref) } })),
  });
  const runOnce = async (): Promise<number> => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(...spec.workgroups);
    pass.end();
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - t0;
  };
  for (let w = 0; w < WARMUP; w++) await runOnce();
  const times: number[] = [];
  for (let r = 0; r < RUNS; r++) times.push(await runOnce());
  rows.push({
    index: i,
    phase: i < fwdCount ? "forward" : "backward",
    group: groupLabel(spec.label),
    label: spec.label,
    workgroups: spec.workgroups,
    ms: median(times),
  });
}

if (CSV) {
  console.log("index,phase,group,label,wgx,wgy,wgz,workgroups,median_ms");
  for (const r of rows) {
    console.log(
      [
        r.index,
        r.phase,
        r.group,
        JSON.stringify(r.label),
        r.workgroups[0],
        r.workgroups[1],
        r.workgroups[2],
        sumWorkgroups(r.workgroups),
        r.ms.toFixed(4),
      ].join(",")
    );
  }
} else {
  const total = rows.reduce((s, r) => s + r.ms, 0);
  const byGroup = new Map<string, number>();
  for (const r of rows) byGroup.set(r.group, (byGroup.get(r.group) ?? 0) + r.ms);
  console.log("\nTop dispatches:");
  for (const r of rows.slice().sort((a, b) => b.ms - a.ms).slice(0, 25)) {
    console.log(
      `${r.ms.toFixed(3).padStart(8)} ms  ${(100 * r.ms / Math.max(total, 1e-6)).toFixed(1).padStart(5)}%  ` +
        `${r.phase.padEnd(8)}  ${r.label}  wg=${r.workgroups.join("x")}`
    );
  }
  console.log("\nGroups:");
  for (const [group, ms] of [...byGroup.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`${ms.toFixed(3).padStart(8)} ms  ${(100 * ms / Math.max(total, 1e-6)).toFixed(1).padStart(5)}%  ${group}`);
  }
  console.log(`\nTotal isolated median sum: ${total.toFixed(3)} ms`);
}

weightsBuffer.destroy();
textBuffer.destroy();
for (const b of slotBuffers) b.destroy();
