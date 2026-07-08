/**
 * Integrated 3D splat optimizer step benchmark.
 *
 * Measures the real Splat3DOptimizer schedule, including raster, CLIP, raster
 * backward, Adam, and display render. Use this to compare the current single
 * CLIP path against the conservative batch-major CLIP path.
 *
 *   CLIP_BATCH=1 VIEWS=3 RUNS=10 WARMUP=3 bun tools/splat3d/step_bench.ts
 *   CLIP_BATCH=3 VIEWS=3 RUNS=10 WARMUP=3 bun tools/splat3d/step_bench.ts
 *   STEM_SPATIAL_BWD=0 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   FUSE_PW_GELU=0 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import type { TrainPlan } from "../../src/clip/vision";
import { LEGIBLE_3D_G, Splat3DOptimizer, randomSplats3D } from "../../src/splat3d/optimize";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const RUNS = Number(process.env.RUNS ?? 10);
const WARMUP = Number(process.env.WARMUP ?? 3);
const VIEWS = Number(process.env.VIEWS ?? 3);
const CLIP_BATCH = Number(process.env.CLIP_BATCH ?? 1);
const SEED = Number(process.env.SEED ?? 1);
const G = Number(process.env.G ?? LEGIBLE_3D_G);
const STEM_SPATIAL_BWD = process.env.STEM_SPATIAL_BWD !== "0";
const FUSE_PW_GELU = process.env.FUSE_PW_GELU !== "0";

function f32File(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
}

function textEmbedding(seed: number, dim: number): Float32Array {
  const out = new Float32Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    const v =
      Math.sin((seed + 1) * 12.9898 + i * 78.233) * 0.5 +
      Math.cos((seed + 3) * 4.1414 + i * 17.17) * 0.5;
    out[i] = v;
    norm += v * v;
  }
  const scale = 11 / Math.sqrt(norm || 1);
  for (let i = 0; i < dim; i++) out[i] *= scale;
  return out;
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
const initParams = randomSplats3D(G, SEED);

const compileStart = performance.now();
const opt = await Splat3DOptimizer.create(device, plan, weights, {
  G,
  seed: SEED,
  initParams,
  clipBatchSize: CLIP_BATCH,
  stemSpatialBwd: STEM_SPATIAL_BWD,
  fusePointwiseGeluForward: FUSE_PW_GELU,
});
console.log(
  `splat3d step bench: G=${G}, views=${VIEWS}/${opt.cameras.length}, ` +
    `clipBatch=${opt.clipBatchSize}, runs=${RUNS}, warmup=${WARMUP}, ` +
    `stemSpatialBwd=${STEM_SPATIAL_BWD ? 1 : 0}, ` +
    `fusePointwiseGeluForward=${FUSE_PW_GELU ? 1 : 0}, ` +
    `compile+allocate=${(performance.now() - compileStart).toFixed(0)} ms`
);

opt.setViewPrompts(opt.cameras.map((_camera, i) => textEmbedding(i, plan.textDim)));

for (let i = 0; i < WARMUP; i++) {
  opt.step(0, VIEWS);
  await device.queue.onSubmittedWorkDone();
}

const t0 = performance.now();
for (let i = 0; i < RUNS; i++) {
  opt.step(0, VIEWS);
  await device.queue.onSubmittedWorkDone();
}
const avg = (performance.now() - t0) / Math.max(1, RUNS);
console.log(`normal step avg: ${avg.toFixed(2)} ms`);

const profile = await opt.profileStep(0, VIEWS);
console.log(
  `profile: total=${profile.total.toFixed(2)} ms ` +
    `rasterFwd=${profile.rasterFwd.toFixed(2)} ` +
    `rasterReplay=${profile.rasterReplay.toFixed(2)} ` +
    `rasterBwd=${profile.rasterBwd.toFixed(2)} ` +
    `clipFwd=${profile.clipFwd.toFixed(2)} ` +
    `clipBwd=${profile.clipBwd.toFixed(2)} ` +
    `clipBatch=${profile.clipBatch.toFixed(2)} ` +
    `adam=${profile.adam.toFixed(2)} display=${profile.display.toFixed(2)}`
);

opt.destroy();
