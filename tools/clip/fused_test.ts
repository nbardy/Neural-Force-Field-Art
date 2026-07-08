/**
 * Headless verification + bench for the fused WGSL MobileCLIP-S0 vision
 * encoder (src/clip/vision_wgsl.ts + vision.ts) on a REAL Dawn/Metal adapter
 * via bun-webgpu — same harness pattern as tools/kernel_test.ts.
 *
 *   bun tools/clip/fused_test.ts            # per-step verify + bench
 *   FAST=1 bun tools/clip/fused_test.ts     # final embedding + bench only
 *
 * Prereqs (all regenerable, gitignored under models/mobileclip_s0/):
 *   node tools/clip/onnx_forward.mjs                       # fixture input
 *   uv run --with onnx --with numpy python tools/clip/compile_plan.py
 *   uv run --with onnx --with numpy --with onnxruntime python tools/clip/dump_refs.py
 *
 * The oracle is ORT CPU fp32 (tools/clip/dump_refs.py): every plan step's
 * output is compared at a relative-L∞ tolerance; the FIRST divergent step is
 * the broken kernel. Baseline to beat: ORT CPU ≈ 60 ms/forward (4 threads).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { VisionEncoder, type VisionPlan } from "../../src/clip/vision";
import type { PointwiseTileVariant } from "../../src/clip/vision_wgsl";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const REL_TOL = 2e-3;   // relative L∞ per step (fp32 reassociation + erf poly)

// PLAN=plan_train.json verifies the split-GELU train plan (weights_train.bin,
// refs_train/) — the SAME per-step oracle, so gate 3 proves the train forward
// matches ORT before any backward runs (spec §3).
const PLAN_FILE = process.env.PLAN ?? "plan.json";
const IS_TRAIN = PLAN_FILE.includes("train");
const WEIGHTS_FILE = process.env.WEIGHTS ?? (IS_TRAIN ? "weights_train.bin" : "weights.bin");
const REFS_DIR = process.env.REFS ?? (IS_TRAIN ? "refs_train" : "refs");
const POINTWISE_TILE_VARIANT: PointwiseTileVariant =
  process.env.PW_TILE_VARIANT === "rect8x16" || process.env.POINTWISE_TILE_VARIANT === "rect8x16"
    ? "rect8x16"
    : "default";
const POINTWISE_TILE_STEPS = parseStepSet(process.env.PW_TILE_STEPS ?? process.env.POINTWISE_TILE_STEPS ?? "");

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

const plan: VisionPlan = JSON.parse(readFileSync(join(MODEL_DIR, PLAN_FILE), "utf8"));
const weights = new Float32Array(readFileSync(join(MODEL_DIR, WEIGHTS_FILE)).buffer);
const input = new Float32Array(
  readFileSync(join(MODEL_DIR, "fixtures", "input_1x3x256x256.f32.bin")).buffer
);
const manifest: { step: number; ref: string | null; shape: number[] | null; file: string | null }[] =
  JSON.parse(readFileSync(join(MODEL_DIR, REFS_DIR, "manifest.json"), "utf8"));

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: any = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const t0 = performance.now();
const enc = await VisionEncoder.create(device, plan, weights, {
  pointwiseTileVariant: POINTWISE_TILE_VARIANT,
  pointwiseTileSteps: POINTWISE_TILE_STEPS,
});
console.log(
  `pipelines: ${plan.steps.length} steps compiled in ${(performance.now() - t0).toFixed(0)} ms` +
    (POINTWISE_TILE_VARIANT !== "default" ? ` pointwiseTileVariant=${POINTWISE_TILE_VARIANT}` : "") +
    (POINTWISE_TILE_STEPS.size ? ` pointwiseTileSteps=${[...POINTWISE_TILE_STEPS].join(",")}` : "") +
    "\n"
);
enc.writeInput(input);

async function readback(buf: any, floats: number): Promise<Float32Array> {
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

function compare(got: Float32Array, want: Float32Array): { maxDiff: number; scale: number } {
  let maxDiff = 0;
  let scale = 1e-6;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > maxDiff) maxDiff = d;
    const a = Math.abs(want[i]);
    if (a > scale) scale = a;
  }
  return { maxDiff, scale };
}

// ---------------------------------------------------------------------------
// per-step verification (bisection: first bad step = broken kernel)
// ---------------------------------------------------------------------------
const counts = enc.stepDispatchCounts();
let failures = 0;
if (!process.env.FAST) {
  let dispatchEnd = 0;
  for (let i = 0; i < plan.steps.length; i++) {
    dispatchEnd += counts[i];
    const step = plan.steps[i];
    const m = manifest[i];
    if (m.file === null) continue; // attention internals: our layout, no ONNX ref
    const floats = (m.shape as number[]).reduce((a, b) => a * b, 1);
    const cmd = device.createCommandEncoder();
    enc.encode(cmd, dispatchEnd);
    device.queue.submit([cmd.finish()]);
    const got = await readback(enc.slotBuffers[step.dst], floats);
    const want = new Float32Array(
      readFileSync(join(MODEL_DIR, REFS_DIR, m.file)).buffer
    );
    const { maxDiff, scale } = compare(got, want);
    const rel = maxDiff / scale;
    const ok = rel < REL_TOL;
    if (!ok) failures++;
    const tag =
      step.kind === "conv" ? `conv:${step.variant}` : step.kind;
    console.log(
      `${ok ? "PASS" : "FAIL"} [${String(i).padStart(3)}] ${tag.padEnd(14)} ` +
        `relL∞ ${rel.toExponential(2)}  (maxDiff ${maxDiff.toExponential(2)}, scale ${scale.toFixed(3)})` +
        (ok ? "" : `  ← ${step.name}`)
    );
    if (!ok && failures >= 3) {
      console.log("3 failures — stopping (fix the first one; the rest inherit its garbage)");
      process.exit(1);
    }
  }
  if (failures) process.exit(1);
}

// ---------------------------------------------------------------------------
// final embedding vs ORT
// ---------------------------------------------------------------------------
enc.run();
const embeds = await readback(enc.outputBuffer, plan.embedDim);
const refEmb = new Float32Array(
  readFileSync(join(MODEL_DIR, "fixtures", "image_embeds_512.f32.bin")).buffer
);
let dot = 0, na = 0, nb = 0;
for (let i = 0; i < plan.embedDim; i++) {
  dot += embeds[i] * refEmb[i];
  na += embeds[i] ** 2;
  nb += refEmb[i] ** 2;
}
const cos = dot / Math.sqrt(na * nb);
const { maxDiff, scale } = compare(embeds, refEmb);
console.log(`\nembedding: cosine vs ORT = ${cos.toFixed(6)}  relL∞ ${(maxDiff / scale).toExponential(2)}`);
console.log(`  first 8: ${[...embeds.slice(0, 8)].map((x) => x.toFixed(5)).join(", ")}`);
if (cos < 0.999) {
  console.error("FAIL: embedding cosine < 0.999");
  process.exit(1);
}

// ---------------------------------------------------------------------------
// bench — full forward, per-frame submit pattern (shared GPU: ±30%, read trends)
// Warmup matters: Metal JITs each pipeline lazily on FIRST USE, so an
// under-warmed bench reads 30–50% high (10-run FAST benches showed ~10 ms
// where steady state is ~6.6 ms). 10 warm forwards + a sync before timing.
// ---------------------------------------------------------------------------
for (let i = 0; i < 10; i++) enc.run();
await readback(enc.outputBuffer, 4);
const RUNS = Number(process.env.BENCH_RUNS ?? 30);
const tEnc0 = performance.now();
for (let i = 0; i < RUNS; i++) enc.run();
const encodeMs = (performance.now() - tEnc0) / RUNS;
await readback(enc.outputBuffer, 4); // sync
const wall = (performance.now() - tEnc0) / RUNS;
console.log(
  `\nbench (${RUNS} runs): ${wall.toFixed(2)} ms/forward GPU-inclusive · ` +
    `${encodeMs.toFixed(2)} ms CPU encode+submit`
);
console.log(`vs ORT CPU baseline ≈60 ms → ${(60 / wall).toFixed(1)}×`);
console.log("\nALL PASS");
