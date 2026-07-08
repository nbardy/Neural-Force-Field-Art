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
 *   SHARED_W_FWD_STEPS=10,15,24,34,49 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   FUSE_GELU_BWD_PW=1 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   FUSE_RESIDUAL_BWD_PW=1 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   SINGLE_PASS_RASTER_FWD=1 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   VIEW_LANE_RASTER_FWD=1 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   VIEW_LANE_RASTER_BWD=1 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   CAP=1024 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 bun tools/splat3d/step_bench.ts
 *   GRID_DIRECT_RASTER=1 CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 bun tools/splat3d/step_bench.ts
 *   CLIP_REFRESH_INTERVAL=2 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   CLIP_REFRESH_INTERVAL=4 CLIP_CACHED_LR_SCALE=0.25 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   BACKGROUND_MODE=dark_random ALPHA_REG=weak BOUNDS_REG=weak CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 *   TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import type { TrainPlan } from "../../src/clip/vision";
import type { PointwiseTileVariant, WeightPrecision } from "../../src/clip/vision_wgsl";
import {
  LEGIBLE_3D_G,
  Splat3DOptimizer,
  randomSplats3D,
  type Splat3DBackgroundMode,
  type Splat3DConvergenceConfig,
  type Splat3DViewSampler,
} from "../../src/splat3d/optimize";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const RUNS = Number(process.env.RUNS ?? 10);
const WARMUP = Number(process.env.WARMUP ?? 3);
const VIEWS = Number(process.env.VIEWS ?? 3);
const CLIP_BATCH = Number(process.env.CLIP_BATCH ?? 1);
const CLIP_LAYOUT = process.env.CLIP_LAYOUT === "grid9_close2" ? "grid9_close2" : "per_view";
const VIEW_SAMPLER: Splat3DViewSampler = process.env.VIEW_SAMPLER === "random" ? "random" : "epoch";
const SEED = Number(process.env.SEED ?? 1);
const G = Number(process.env.G ?? LEGIBLE_3D_G);
const CAP = Number(process.env.CAP ?? 2048);
const STEM_SPATIAL_BWD = process.env.STEM_SPATIAL_BWD !== "0";
const SPATIAL_BWD_VARIANT =
  process.env.SPATIAL_BWD_VARIANT === "generic"
    ? "generic"
    : process.env.SPATIAL_BWD_VARIANT === "depthwise4"
      ? "depthwise4"
      : undefined;
const FUSE_PW_GELU = process.env.FUSE_PW_GELU !== "0";
const FUSE_GELU_BWD_PW =
  process.env.FUSE_GELU_BWD_PW === "1" ? true : process.env.FUSE_GELU_BWD_PW === "0" ? false : undefined;
const FUSE_RESIDUAL_BWD_PW =
  process.env.FUSE_RESIDUAL_BWD_PW === "1" ? true : process.env.FUSE_RESIDUAL_BWD_PW === "0" ? false : undefined;
const SINGLE_PASS_RASTER_FWD = process.env.SINGLE_PASS_RASTER_FWD === "1";
const VIEW_LANE_RASTER_FWD = process.env.VIEW_LANE_RASTER_FWD === "1";
const VIEW_LANE_RASTER_BWD = process.env.VIEW_LANE_RASTER_BWD === "1";
const GRID_DIRECT_RASTER = process.env.GRID_DIRECT_RASTER === "1";
const CLIP_REFRESH_INTERVAL = Math.max(1, Number(process.env.CLIP_REFRESH_INTERVAL ?? 1) | 0);
const CLIP_CACHED_LR_SCALE = normalizeCachedLrScale(Number(process.env.CLIP_CACHED_LR_SCALE ?? 1));
const BACKGROUND_MODE: Splat3DBackgroundMode =
  process.env.BACKGROUND_MODE === "dark_random" || process.env.BACKGROUND_MODE === "curriculum"
    ? process.env.BACKGROUND_MODE
    : "black";
const ALPHA_REG = process.env.ALPHA_REG === "medium" ? "medium" : process.env.ALPHA_REG === "weak" ? "weak" : "off";
const BOUNDS_REG = process.env.BOUNDS_REG === "medium" ? "medium" : process.env.BOUNDS_REG === "weak" ? "weak" : "off";
const CONVERGENCE: Splat3DConvergenceConfig = {
  backgroundMode: BACKGROUND_MODE,
  opacitySparsity: ALPHA_REG === "medium" ? 0.03 : ALPHA_REG === "weak" ? 0.01 : 0,
  centerWeight: BOUNDS_REG === "medium" ? 0.006 : BOUNDS_REG === "weak" ? 0.002 : 0,
  radiusWeight: BOUNDS_REG === "medium" ? 0.012 : BOUNDS_REG === "weak" ? 0.004 : 0,
  targetRadius: 1.15,
};
const SHARED_W_FWD_STEPS = parseStepSet(process.env.SHARED_W_FWD_STEPS ?? "");
const POINTWISE_TILE_VARIANT: PointwiseTileVariant =
  process.env.PW_TILE_VARIANT === "rect8x16" || process.env.POINTWISE_TILE_VARIANT === "rect8x16"
    ? "rect8x16"
    : "default";
const POINTWISE_TILE_STEPS = parseStepSet(process.env.PW_TILE_STEPS ?? process.env.POINTWISE_TILE_STEPS ?? "");
const TIMESTAMP = process.env.TIMESTAMP === "1";
const CLIP_PRECISION: WeightPrecision =
  process.env.CLIP_PRECISION === "f16" || process.env.PRECISION === "f16" ? "f16" : "f32";
const WEIGHTS_FILE =
  process.env.WEIGHTS ?? (CLIP_PRECISION === "f16" ? "weights_train_f16.bin" : "weights_train.bin");

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

function normalizeCachedLrScale(value: number): number {
  return Number.isFinite(value) ? Math.max(0, value) : 1;
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
const timestampSupported = adapter.features.has("timestamp-query");
const f16Supported = adapter.features.has("shader-f16");
if (CLIP_PRECISION === "f16" && !f16Supported) {
  throw new Error("splat3d step bench: CLIP_PRECISION=f16 requested but adapter lacks shader-f16");
}
const requiredFeatures: GPUFeatureName[] = [];
if (TIMESTAMP && timestampSupported) requiredFeatures.push("timestamp-query" as GPUFeatureName);
if (CLIP_PRECISION === "f16") requiredFeatures.push("shader-f16" as GPUFeatureName);
const device: GPUDevice = await adapter.requestDevice({ requiredFeatures });
const useTimestamps = TIMESTAMP && device.features.has("timestamp-query");
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);
if (TIMESTAMP && !useTimestamps) {
  console.log("splat3d step bench: timestamp-query unavailable, falling back to split-submit wall time");
}

const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = CLIP_PRECISION === "f16"
  ? f16File(join(MODEL_DIR, WEIGHTS_FILE))
  : f32File(join(MODEL_DIR, WEIGHTS_FILE));
const initParams = randomSplats3D(G, SEED);

const compileStart = performance.now();
const opt = await Splat3DOptimizer.create(device, plan, weights, {
  G,
  cap: CAP,
  seed: SEED,
  initParams,
  clipBatchSize: CLIP_BATCH,
  clipLayout: CLIP_LAYOUT,
  viewSampler: VIEW_SAMPLER,
  clipWeightPrecision: CLIP_PRECISION,
  pointwiseTileVariant: POINTWISE_TILE_VARIANT,
  pointwiseTileSteps: POINTWISE_TILE_STEPS,
  stemSpatialBwd: STEM_SPATIAL_BWD,
  spatialBwdVariant: SPATIAL_BWD_VARIANT,
  fusePointwiseGeluForward: FUSE_PW_GELU,
  fuseGeluBwdIntoPw: FUSE_GELU_BWD_PW,
  fuseResidualBwdIntoPw: FUSE_RESIDUAL_BWD_PW,
  singlePassBatchRasterForward: SINGLE_PASS_RASTER_FWD,
  viewLaneBatchRasterForward: VIEW_LANE_RASTER_FWD,
  viewLaneBatchRasterBackward: VIEW_LANE_RASTER_BWD,
  gridDirectRaster: GRID_DIRECT_RASTER,
  sharedWForwardSteps: SHARED_W_FWD_STEPS,
  clipRefreshInterval: CLIP_REFRESH_INTERVAL,
  cachedLrScale: CLIP_CACHED_LR_SCALE,
  convergence: CONVERGENCE,
});
console.log(
  `splat3d step bench: G=${G}, views=${VIEWS}/${opt.cameras.length}, ` +
    `clipBatch=${opt.clipBatchSize}, clipLayout=${opt.clipLayout}, clipPrecision=${CLIP_PRECISION}, ` +
    `viewSampler=${opt.viewSampler}, weights=${WEIGHTS_FILE}, cap=${CAP}, runs=${RUNS}, warmup=${WARMUP}, ` +
    `stemSpatialBwd=${STEM_SPATIAL_BWD ? 1 : 0}, ` +
    `spatialBwdVariant=${SPATIAL_BWD_VARIANT ?? "default"}, ` +
    `pointwiseTileVariant=${POINTWISE_TILE_VARIANT}, ` +
    (POINTWISE_TILE_STEPS.size ? `pointwiseTileSteps=${[...POINTWISE_TILE_STEPS].join(",")}, ` : "") +
    `fusePointwiseGeluForward=${FUSE_PW_GELU ? 1 : 0}, ` +
    `fuseGeluBwdIntoPw=${FUSE_GELU_BWD_PW === undefined ? "default" : FUSE_GELU_BWD_PW ? 1 : 0}, ` +
    `fuseResidualBwdIntoPw=${FUSE_RESIDUAL_BWD_PW === undefined ? "default" : FUSE_RESIDUAL_BWD_PW ? 1 : 0}, ` +
    (SHARED_W_FWD_STEPS.size ? `sharedWForwardSteps=${[...SHARED_W_FWD_STEPS].join(",")}, ` : "") +
    `singlePassBatchRasterForward=${SINGLE_PASS_RASTER_FWD ? 1 : 0}, ` +
    `viewLaneBatchRasterForward=${VIEW_LANE_RASTER_FWD ? 1 : 0}, ` +
    `viewLaneBatchRasterBackward=${VIEW_LANE_RASTER_BWD ? 1 : 0}, ` +
    `gridDirectRaster=${GRID_DIRECT_RASTER ? 1 : 0}, ` +
    `clipRefreshInterval=${CLIP_REFRESH_INTERVAL}, ` +
    `cachedLrScale=${CLIP_CACHED_LR_SCALE}, ` +
    `backgroundMode=${BACKGROUND_MODE}, alphaReg=${ALPHA_REG}, boundsReg=${BOUNDS_REG}, ` +
    `timing=${useTimestamps ? "gpu-timestamp" : "split-submit-wall"}, ` +
    `compile+allocate=${(performance.now() - compileStart).toFixed(0)} ms`
);

opt.setViewPrompts(opt.cameras.map((_camera, i) => textEmbedding(i, plan.textDim)));
if (CLIP_LAYOUT === "grid9_close2") {
  opt.setGridPrompt(textEmbedding(99, plan.textDim));
}

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

const profile = await opt.profileStep(0, VIEWS, { gpuTimestamps: useTimestamps });
console.log(
  `profile: total=${profile.total.toFixed(2)} ms ` +
    `rasterFwd=${profile.rasterFwd.toFixed(2)} ` +
    `rasterReplay=${profile.rasterReplay.toFixed(2)} ` +
    `rasterBwd=${profile.rasterBwd.toFixed(2)} ` +
    `clipFwd=${profile.clipFwd.toFixed(2)} ` +
    `clipBwd=${profile.clipBwd.toFixed(2)} ` +
    `clipBatch=${profile.clipBatch.toFixed(2)} ` +
    `regularizer=${profile.regularizer.toFixed(2)} ` +
    `adam=${profile.adam.toFixed(2)} display=${profile.display.toFixed(2)} ` +
    `timing=${profile.timing}`
);

opt.destroy();
