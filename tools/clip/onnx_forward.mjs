// tools/clip/onnx_forward.mjs — MobileCLIP-S0 vision encoder ONNX baseline.
//
// Run:  node tools/clip/onnx_forward.mjs [runs=20]
//
// The ORT-CPU forward pass is the CORRECTNESS ORACLE for the fused WGSL port
// (same pattern as tfjs ↔ advect kernel in tools/kernel_test.ts): it runs the
// encoder on a deterministic synthetic image and writes the input + embedding
// fixture that tools/clip tests compare against. It also prints per-run
// timings — that's the "standard runtime" number the fused WGSL pass has to
// beat (browser ORT-WebGPU is dispatch-bound on top of this; see HANDOFF.md §2).
//
// Input is NCHW [1,3,256,256] in [0,1] (preprocessor: rescale 1/255 only —
// MobileCLIP does NO mean/std normalization, see preprocessor_config.json).
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ort from "onnxruntime-node";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const MODEL = join(ROOT, "models", "mobileclip_s0", "vision_model.onnx");
const FIXTURE_DIR = join(ROOT, "models", "mobileclip_s0", "fixtures");
const RUNS = Number(process.argv[2] ?? 20);

// Same PCG hash the WGSL kernels use (advect_wgsl.ts) — the fixture input is
// reproducible from the index alone, no RNG state to serialize.
const pcg = (v) => {
  v = Math.imul(v, 747796405) + 2891336453;
  const t = Math.imul((v >>> ((v >>> 28) + 4)) ^ v, 277803737);
  return ((t >>> 22) ^ t) >>> 0;
};
const rand01 = (u) => u * 2.3283064365386963e-10;

export function syntheticImage() {
  const n = 3 * 256 * 256;
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = rand01(pcg(i));
  return data;
}

const input = syntheticImage();
const session = await ort.InferenceSession.create(MODEL, {
  // Single-threaded intra-op: a stable, conservative CPU baseline (the
  // shared-GPU ±30% caveat in HANDOFF.md applies to multi-threaded CPU too).
  intraOpNumThreads: 4,
  graphOptimizationLevel: "all",
});

const feeds = {
  pixel_values: new ort.Tensor("float32", input, [1, 3, 256, 256]),
};

// warmup
for (let i = 0; i < 3; i++) await session.run(feeds);

const times = [];
let out;
for (let i = 0; i < RUNS; i++) {
  const t0 = performance.now();
  out = await session.run(feeds);
  times.push(performance.now() - t0);
}
const embeds = out.image_embeds.data;

times.sort((a, b) => a - b);
const mean = times.reduce((a, b) => a + b, 0) / times.length;
const med = times[Math.floor(times.length / 2)];

const norm = Math.hypot(...embeds);
console.log(`model: ${MODEL}`);
console.log(`embedding: [${out.image_embeds.dims}]  L2=${norm.toFixed(4)}`);
console.log(`  first 8: ${[...embeds.slice(0, 8)].map((x) => x.toFixed(5)).join(", ")}`);
console.log(
  `ORT CPU (${RUNS} runs, 4 threads): mean ${mean.toFixed(1)} ms · ` +
    `median ${med.toFixed(1)} ms · min ${times[0].toFixed(1)} ms`
);

mkdirSync(FIXTURE_DIR, { recursive: true });
writeFileSync(join(FIXTURE_DIR, "input_1x3x256x256.f32.bin"), Buffer.from(input.buffer));
writeFileSync(
  join(FIXTURE_DIR, "image_embeds_512.f32.bin"),
  Buffer.from(embeds.buffer, embeds.byteOffset, embeds.byteLength)
);
console.log(`fixtures → ${FIXTURE_DIR}`);
