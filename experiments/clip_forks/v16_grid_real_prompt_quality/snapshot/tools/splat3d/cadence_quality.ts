/**
 * Fixed-budget quality probe for CLIP gradient cadence.
 *
 * This complements step_bench.ts. Step timing can show that cached
 * `dL/dimage` cadence is faster, but this tool asks the more important
 * question: after the same wall-clock optimization budget, did the splats still
 * move toward the full frozen CLIP teacher across all nine views?
 *
 *   BUDGET_MS=5000 CONFIGS=base=1,cache2=2,cache4=4 bun tools/splat3d/cadence_quality.ts
 *   BUDGET_MS=5000 CONFIGS=base=1,cache4=4,cache4lr25=4:0.25 bun tools/splat3d/cadence_quality.ts
 *   RUN_STEPS=80 CONFIGS=base=1,cache4=4 bun tools/splat3d/cadence_quality.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import type { TrainPlan } from "../../src/clip/vision";
import type { WeightPrecision } from "../../src/clip/vision_wgsl";
import {
  LEGIBLE_3D_G,
  Splat3DOptimizer,
  randomSplats3D,
  type Splat3DViewSampler,
} from "../../src/splat3d/optimize";
import { writePNG } from "../splat/scene";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const OUT_DIR = process.env.OUT_DIR ?? "/tmp/nffa_cadence_quality";
const CONFIGS = parseConfigs(process.env.CONFIGS ?? "base=1,cache2=2,cache4=4");
const BUDGET_MS = Number(process.env.BUDGET_MS ?? 5000);
const RUN_STEPS = Number(process.env.RUN_STEPS ?? 0) | 0;
const VIEWS = Number(process.env.VIEWS ?? 3);
const CLIP_BATCH = Number(process.env.CLIP_BATCH ?? 3);
const VIEW_SAMPLER: Splat3DViewSampler = process.env.VIEW_SAMPLER === "random" ? "random" : "epoch";
const SEED = Number(process.env.SEED ?? 1);
const G = Number(process.env.G ?? LEGIBLE_3D_G);
const CAP = Number(process.env.CAP ?? 2048);
const CLIP_PRECISION: WeightPrecision =
  process.env.CLIP_PRECISION === "f16" || process.env.PRECISION === "f16" ? "f16" : "f32";
const WEIGHTS_FILE =
  process.env.WEIGHTS ?? (CLIP_PRECISION === "f16" ? "weights_train_f16.bin" : "weights_train.bin");
const STEM_SPATIAL_BWD = process.env.STEM_SPATIAL_BWD !== "0";
const SPATIAL_BWD_VARIANT = process.env.SPATIAL_BWD_VARIANT === "depthwise4" ? "depthwise4" : undefined;
const FUSE_PW_GELU = process.env.FUSE_PW_GELU !== "0";
const FUSE_GELU_BWD_PW = process.env.FUSE_GELU_BWD_PW === "1";
const FUSE_RESIDUAL_BWD_PW = process.env.FUSE_RESIDUAL_BWD_PW === "1";
const SINGLE_PASS_RASTER_FWD = process.env.SINGLE_PASS_RASTER_FWD === "1";
const VIEW_LANE_RASTER_FWD = process.env.VIEW_LANE_RASTER_FWD === "1";
const VIEW_LANE_RASTER_BWD = process.env.VIEW_LANE_RASTER_BWD === "1";

interface CadenceConfig {
  label: string;
  clipRefreshInterval: number;
  cachedLrScale: number;
}

interface CadenceResult {
  label: string;
  clipRefreshInterval: number;
  cachedLrScale: number;
  steps: number;
  trainMs: number;
  stepsPerSecond: number;
  meanCos: number;
  minCos: number;
  maxCos: number;
  viewCos: number[];
  sheetPath: string;
}

function parseConfigs(src: string): CadenceConfig[] {
  const configs = src
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [labelRaw, intervalRaw] = part.includes("=") ? part.split("=") : [`refresh${part}`, part];
      const [refreshRaw, scaleRaw] = intervalRaw.split(":");
      const parsed = Number(refreshRaw);
      if (!Number.isFinite(parsed)) throw new Error(`cadence_quality: bad config '${part}'`);
      const interval = Math.max(1, parsed | 0);
      const scaleParsed = scaleRaw === undefined ? 1 : Number(scaleRaw);
      const cachedLrScale = Number.isFinite(scaleParsed) ? Math.max(0, scaleParsed) : 1;
      return { label: labelRaw.trim(), clipRefreshInterval: interval, cachedLrScale };
    });
  if (!configs.length) throw new Error("cadence_quality: CONFIGS produced no configs");
  return configs;
}

function f32File(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
}

function f16File(path: string): Uint16Array {
  const b = readFileSync(path);
  return new Uint16Array(b.buffer, b.byteOffset, b.byteLength / 2).slice();
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

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let aa = 0;
  let bb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return dot / Math.sqrt(Math.max(aa * bb, 1e-20));
}

function makeContactSheet(images: Float32Array[], side: number): Float32Array {
  const cols = 3;
  const rows = 3;
  const outW = side * cols;
  const outH = side * rows;
  const out = new Float32Array(3 * outW * outH);
  const srcHW = side * side;
  const dstHW = outW * outH;
  for (let view = 0; view < images.length; view++) {
    const img = images[view];
    const col = view % cols;
    const row = Math.floor(view / cols);
    for (let y = 0; y < side; y++) {
      const dstY = row * side + y;
      for (let x = 0; x < side; x++) {
        const dstX = col * side + x;
        const srcI = y * side + x;
        const dstI = dstY * outW + dstX;
        for (let c = 0; c < 3; c++) {
          out[c * dstHW + dstI] = img[c * srcHW + srcI];
        }
      }
    }
  }
  return out;
}

async function evaluateAllViews(
  opt: Splat3DOptimizer,
  prompts: Float32Array[],
  label: string
): Promise<Omit<CadenceResult, "label" | "clipRefreshInterval" | "steps" | "trainMs" | "stepsPerSecond">> {
  const viewCos: number[] = [];
  const images: Float32Array[] = [];
  for (let view = 0; view < opt.cameras.length; view++) {
    const emb = await opt.currentEmbedding(view);
    viewCos.push(cosine(emb, prompts[view]));
    images.push(await opt.renderView(view));
  }
  const meanCos = viewCos.reduce((sum, v) => sum + v, 0) / Math.max(1, viewCos.length);
  const minCos = Math.min(...viewCos);
  const maxCos = Math.max(...viewCos);
  const sheetPath = join(OUT_DIR, `${label}_views.png`);
  writePNG(sheetPath, makeContactSheet(images, opt.side), opt.side * 3, opt.side * 3);
  return { meanCos, minCos, maxCos, viewCos, sheetPath };
}

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const f16Supported = adapter.features.has("shader-f16");
if (CLIP_PRECISION === "f16" && !f16Supported) {
  throw new Error("cadence_quality: CLIP_PRECISION=f16 requested but adapter lacks shader-f16");
}
const requiredFeatures: GPUFeatureName[] = CLIP_PRECISION === "f16" ? ["shader-f16" as GPUFeatureName] : [];
const device: GPUDevice = await adapter.requestDevice({ requiredFeatures });
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

mkdirSync(OUT_DIR, { recursive: true });

const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = CLIP_PRECISION === "f16"
  ? f16File(join(MODEL_DIR, WEIGHTS_FILE))
  : f32File(join(MODEL_DIR, WEIGHTS_FILE));
const initParams = randomSplats3D(G, SEED);
const prompts = Array.from({ length: 9 }, (_unused, i) => textEmbedding(i, plan.textDim));
const results: CadenceResult[] = [];

console.log(
  `cadence_quality: G=${G}, views=${VIEWS}, clipBatch=${CLIP_BATCH}, viewSampler=${VIEW_SAMPLER}, ` +
    `weights=${WEIGHTS_FILE}, cap=${CAP}, budgetMs=${RUN_STEPS > 0 ? "fixedSteps" : BUDGET_MS}, ` +
    `runSteps=${RUN_STEPS}, configs=${CONFIGS.map((c) => `${c.label}:${c.clipRefreshInterval}:lr${c.cachedLrScale}`).join(",")}`
);

for (const cfg of CONFIGS) {
  const compileStart = performance.now();
  const opt = await Splat3DOptimizer.create(device, plan, weights, {
    G,
    cap: CAP,
    seed: SEED,
    initParams,
    clipBatchSize: CLIP_BATCH,
    clipLayout: "per_view",
    viewSampler: VIEW_SAMPLER,
    clipWeightPrecision: CLIP_PRECISION,
    stemSpatialBwd: STEM_SPATIAL_BWD,
    spatialBwdVariant: SPATIAL_BWD_VARIANT,
    fusePointwiseGeluForward: FUSE_PW_GELU,
    fuseGeluBwdIntoPw: FUSE_GELU_BWD_PW,
    fuseResidualBwdIntoPw: FUSE_RESIDUAL_BWD_PW,
    singlePassBatchRasterForward: SINGLE_PASS_RASTER_FWD,
    viewLaneBatchRasterForward: VIEW_LANE_RASTER_FWD,
    viewLaneBatchRasterBackward: VIEW_LANE_RASTER_BWD,
    clipRefreshInterval: cfg.clipRefreshInterval,
    cachedLrScale: cfg.cachedLrScale,
  });
  opt.setViewPrompts(prompts);
  await device.queue.onSubmittedWorkDone();

  let steps = 0;
  const trainStart = performance.now();
  if (RUN_STEPS > 0) {
    for (; steps < RUN_STEPS; steps++) {
      opt.step(0, VIEWS);
      await device.queue.onSubmittedWorkDone();
    }
  } else {
    do {
      opt.step(0, VIEWS);
      await device.queue.onSubmittedWorkDone();
      steps++;
    } while (performance.now() - trainStart < BUDGET_MS);
  }
  const trainMs = performance.now() - trainStart;
  const quality = await evaluateAllViews(opt, prompts, cfg.label);
  opt.destroy();

  const result: CadenceResult = {
    label: cfg.label,
    clipRefreshInterval: cfg.clipRefreshInterval,
    cachedLrScale: cfg.cachedLrScale,
    steps,
    trainMs,
    stepsPerSecond: (steps * 1000) / Math.max(trainMs, 1e-6),
    ...quality,
  };
  results.push(result);
  console.log(
    `result ${cfg.label}: refresh=${cfg.clipRefreshInterval} cachedLrScale=${cfg.cachedLrScale} steps=${steps} ` +
      `train=${trainMs.toFixed(0)}ms compile=${(trainStart - compileStart).toFixed(0)}ms ` +
      `stepRate=${result.stepsPerSecond.toFixed(2)}/s meanCos=${result.meanCos.toFixed(5)} ` +
      `minCos=${result.minCos.toFixed(5)} maxCos=${result.maxCos.toFixed(5)} sheet=${result.sheetPath}`
  );
}

const jsonPath = join(OUT_DIR, "cadence_quality.json");
writeFileSync(
  jsonPath,
  JSON.stringify(
    {
      date: new Date().toISOString(),
      adapter: info,
      config: {
        G,
        cap: CAP,
        seed: SEED,
        views: VIEWS,
        clipBatch: CLIP_BATCH,
        viewSampler: VIEW_SAMPLER,
        clipPrecision: CLIP_PRECISION,
        weightsFile: WEIGHTS_FILE,
        budgetMs: RUN_STEPS > 0 ? null : BUDGET_MS,
        runSteps: RUN_STEPS > 0 ? RUN_STEPS : null,
      },
      results,
    },
    null,
    2
  )
);
console.log(`JSON: ${jsonPath}`);
