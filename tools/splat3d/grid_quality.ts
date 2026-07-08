/**
 * Real-prompt fixed-budget quality probe for 3D grid/contact-sheet CLIP.
 *
 * This judges training schedules by full per-view 256x256 MobileCLIP teacher
 * scores, not by the grid/contact-sheet lane loss. Text embeddings come from
 * the real MobileCLIP text ONNX model on CPU; image embeddings and training use
 * the repo's WebGPU vision/raster path.
 *
 *   BUDGET_MS=5000 PROMPT="a photo of a cat" CONFIGS=base3=3:3,full9=9:3,grid80=9:3:grid9:directgrid bun tools/splat3d/grid_quality.ts
 *   TRIALS=3 RUN_STEPS=2 CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid bun tools/splat3d/grid_quality.ts
 */
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import ort from "onnxruntime-node";
import { AutoTokenizer, env } from "@huggingface/transformers";
import type { TrainPlan } from "../../src/clip/vision";
import type { WeightPrecision } from "../../src/clip/vision_wgsl";
import { buildGrid9Prompt, buildViewPrompt, type Grid9PromptMode } from "../../src/splat3d/cameras";
import {
  LEGIBLE_3D_G,
  Splat3DOptimizer,
  randomSplats3D,
  type Splat3DClipLayout,
  type Splat3DViewSampler,
} from "../../src/splat3d/optimize";
import { writePNG } from "../splat/scene";

setupGlobals();

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const MODEL_DIR = join(ROOT, "models", "mobileclip_s0");
const TEXT_MODEL = join(MODEL_DIR, "onnx", "text_model_fp16.onnx");
const HF = "https://huggingface.co/Xenova/mobileclip_s0/resolve/main";
const CTX = 77;

const OUT_DIR = process.env.OUT_DIR ?? "/tmp/nffa_grid_quality";
const PROMPT = process.env.PROMPT ?? "a photo of a cat";
const CONFIGS = parseConfigs(process.env.CONFIGS ?? "base3=3:3,full9=9:3,grid80=9:3:grid9:directgrid");
const BUDGET_MS = Number(process.env.BUDGET_MS ?? 5000);
const RUN_STEPS = Number(process.env.RUN_STEPS ?? 0) | 0;
const TRIALS = Math.max(1, Number(process.env.TRIALS ?? 1) | 0);
const SEED = Number(process.env.SEED ?? 1);
const G = Number(process.env.G ?? LEGIBLE_3D_G);
const CAP = Number(process.env.CAP ?? 2048);
const VIEW_SAMPLER: Splat3DViewSampler = process.env.VIEW_SAMPLER === "random" ? "random" : "epoch";
const BLACK_BG_TEXT = process.env.BLACK_BG_TEXT !== "0";
const CLIP_PRECISION: WeightPrecision =
  process.env.CLIP_PRECISION === "f16" || process.env.PRECISION === "f16" ? "f16" : "f32";
const WEIGHTS_FILE =
  process.env.WEIGHTS ?? (CLIP_PRECISION === "f16" ? "weights_train_f16.bin" : "weights_train.bin");

interface GridQualityConfig {
  label: string;
  views: number;
  clipBatch: number;
  clipLayout: Splat3DClipLayout;
  gridDirectRaster: boolean;
  gridPromptMode: Grid9PromptMode;
}

interface QualityEval {
  meanCos: number;
  minCos: number;
  maxCos: number;
  viewCos: number[];
  sheetPath: string;
}

interface BaselineEval extends QualityEval {
  trial: number;
  seed: number;
}

interface GridQualityResult extends QualityEval {
  label: string;
  trial: number;
  seed: number;
  views: number;
  clipBatch: number;
  clipLayout: Splat3DClipLayout;
  gridDirectRaster: boolean;
  gridPromptMode: Grid9PromptMode;
  steps: number;
  trainMs: number;
  compileMs: number;
  stepsPerSecond: number;
  meanDelta: number;
  minDelta: number;
}

interface GridQualitySummary {
  label: string;
  trials: number;
  medianSteps: number;
  medianTrainMs: number;
  medianCompileMs: number;
  medianStepsPerSecond: number;
  medianMeanCos: number;
  medianMeanDelta: number;
  medianMinCos: number;
  medianMinDelta: number;
}

function parseConfigs(src: string): GridQualityConfig[] {
  const configs = src
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [labelRaw, bodyRaw] = part.includes("=") ? part.split("=") : [part, part];
      const [viewsRaw, batchRaw, ...tokens] = bodyRaw.split(/[:x]/);
      const views = Number(viewsRaw);
      const clipBatch = Number(batchRaw);
      if (!Number.isFinite(views) || !Number.isFinite(clipBatch)) {
        throw new Error(`grid_quality: bad config '${part}', expected views:clipBatch[:tokens]`);
      }
      const clipLayout = tokens.includes("grid9") || tokens.includes("grid9_close2") ? "grid9_close2" : "per_view";
      const gridDirectRaster = tokens.includes("directgrid") || tokens.includes("grid80") || tokens.includes("direct");
      const gridPromptMode = tokens.includes("same") ? "same" : "contact_sheet";
      return {
        label: labelRaw.trim(),
        views: views | 0,
        clipBatch: clipBatch | 0,
        clipLayout,
        gridDirectRaster,
        gridPromptMode,
      };
    });
  if (!configs.length) throw new Error("grid_quality: CONFIGS produced no configs");
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

function ensureTextAssets(): void {
  for (const rel of ["tokenizer.json", "tokenizer_config.json", "onnx/text_model_fp16.onnx"]) {
    const dst = join(MODEL_DIR, rel);
    if (existsSync(dst)) continue;
    mkdirSync(dirname(dst), { recursive: true });
    console.error(`grid_quality: fetching ${rel}`);
    execSync(`curl -sfL -o "${dst}" "${HF}/${rel}"`, { stdio: "inherit" });
  }
}

async function createTextEmbedder(): Promise<(prompt: string) => Promise<Float32Array>> {
  ensureTextAssets();
  env.allowRemoteModels = false;
  env.localModelPath = join(ROOT, "models");
  const tokenizer = await AutoTokenizer.from_pretrained("mobileclip_s0");
  const session = await ort.InferenceSession.create(TEXT_MODEL, { graphOptimizationLevel: "basic" });
  const cache = new Map<string, Promise<Float32Array>>();
  return (prompt: string): Promise<Float32Array> => {
    const key = prompt.trim();
    let cached = cache.get(key);
    if (!cached) {
      cached = (async () => {
        const enc = tokenizer(key, { padding: "max_length", max_length: CTX, truncation: true });
        const out = await session.run({ input_ids: new ort.Tensor("int64", enc.input_ids.data, enc.input_ids.dims) });
        return new Float32Array(out.text_embeds.data as Float32Array);
      })();
      cache.set(key, cached);
    }
    return cached;
  };
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
  for (let view = 0; view < Math.min(images.length, cols * rows); view++) {
    const img = images[view];
    const col = view % cols;
    const row = Math.floor(view / cols);
    for (let y = 0; y < side; y++) {
      const dstY = row * side + y;
      for (let x = 0; x < side; x++) {
        const dstX = col * side + x;
        const srcI = y * side + x;
        const dstI = dstY * outW + dstX;
        for (let c = 0; c < 3; c++) out[c * dstHW + dstI] = img[c * srcHW + srcI];
      }
    }
  }
  return out;
}

async function evaluateAllViews(
  opt: Splat3DOptimizer,
  prompts: Float32Array[],
  label: string
): Promise<QualityEval> {
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

function configSummary(config: GridQualityConfig): string {
  const tokens = [
    `${config.views}:${config.clipBatch}`,
    config.clipLayout === "grid9_close2" ? "grid9" : "per_view",
    config.gridDirectRaster ? "directgrid" : "",
    config.gridPromptMode === "same" ? "same" : "",
  ].filter(Boolean);
  return `${config.label}=${tokens.join(":")}`;
}

function median(values: number[]): number {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) * 0.5;
}

function summarizeResults(results: GridQualityResult[]): GridQualitySummary[] {
  const labels = [...new Set(results.map((r) => r.label))];
  return labels.map((label) => {
    const rows = results.filter((r) => r.label === label);
    return {
      label,
      trials: rows.length,
      medianSteps: median(rows.map((r) => r.steps)),
      medianTrainMs: median(rows.map((r) => r.trainMs)),
      medianCompileMs: median(rows.map((r) => r.compileMs)),
      medianStepsPerSecond: median(rows.map((r) => r.stepsPerSecond)),
      medianMeanCos: median(rows.map((r) => r.meanCos)),
      medianMeanDelta: median(rows.map((r) => r.meanDelta)),
      medianMinCos: median(rows.map((r) => r.minCos)),
      medianMinDelta: median(rows.map((r) => r.minDelta)),
    };
  });
}

function trialOrder(configs: GridQualityConfig[], trial: number): GridQualityConfig[] {
  if (configs.length <= 1 || TRIALS <= 1) return configs;
  const offset = trial % configs.length;
  return configs.slice(offset).concat(configs.slice(0, offset));
}

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const f16Supported = adapter.features.has("shader-f16");
if (CLIP_PRECISION === "f16" && !f16Supported) {
  throw new Error("grid_quality: CLIP_PRECISION=f16 requested but adapter lacks shader-f16");
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
const embedText = await createTextEmbedder();
const promptInitParams = randomSplats3D(G, SEED);

const promptShell = await Splat3DOptimizer.create(device, plan, weights, {
  G,
  cap: CAP,
  seed: SEED,
  initParams: promptInitParams,
  clipBatchSize: 3,
});
const viewPromptTexts = promptShell.cameras.map((camera) => buildViewPrompt(PROMPT, camera, BLACK_BG_TEXT));
promptShell.destroy();

console.log(`grid_quality: prompt="${PROMPT}" blackBgText=${BLACK_BG_TEXT ? 1 : 0}`);
console.log(
  `grid_quality: configs=${CONFIGS.map(configSummary).join(",")} trials=${TRIALS} ` +
    `budgetMs=${RUN_STEPS > 0 ? "fixedSteps" : BUDGET_MS} runSteps=${RUN_STEPS}`
);

const viewEmbeds: Float32Array[] = [];
for (let i = 0; i < viewPromptTexts.length; i++) {
  console.log(`encode view ${i + 1}/${viewPromptTexts.length}: ${viewPromptTexts[i]}`);
  viewEmbeds.push(await embedText(viewPromptTexts[i]));
}

const results: GridQualityResult[] = [];
const baselines: BaselineEval[] = [];

for (let trial = 0; trial < TRIALS; trial++) {
  const trialSeed = SEED + trial;
  const initParams = randomSplats3D(G, trialSeed);
  let baselineEval: QualityEval | null = null;
  const configs = trialOrder(CONFIGS, trial);
  console.log(`trial ${trial + 1}/${TRIALS}: seed=${trialSeed} order=${configs.map((c) => c.label).join(",")}`);

  for (const cfg of configs) {
    const compileStart = performance.now();
    const opt = await Splat3DOptimizer.create(device, plan, weights, {
      G,
      cap: CAP,
      seed: trialSeed,
      initParams,
      clipBatchSize: cfg.clipBatch,
      clipLayout: cfg.clipLayout,
      viewSampler: VIEW_SAMPLER,
      clipWeightPrecision: CLIP_PRECISION,
      gridDirectRaster: cfg.gridDirectRaster,
    });
    opt.setViewPrompts(viewEmbeds);
    let gridPromptText: string | null = null;
    if (cfg.clipLayout === "grid9_close2") {
      gridPromptText = buildGrid9Prompt(PROMPT, BLACK_BG_TEXT, cfg.gridPromptMode);
      console.log(`encode grid ${cfg.label}: ${gridPromptText}`);
      opt.setGridPrompt(await embedText(gridPromptText));
    }
    await device.queue.onSubmittedWorkDone();
    if (!baselineEval) {
      const initialLabel = TRIALS > 1 ? `initial_t${trial}` : "initial";
      baselineEval = await evaluateAllViews(opt, viewEmbeds, initialLabel);
      baselines.push({ trial, seed: trialSeed, ...baselineEval });
    }

    let steps = 0;
    const trainStart = performance.now();
    if (RUN_STEPS > 0) {
      for (; steps < RUN_STEPS; steps++) {
        opt.step(0, cfg.views);
        await device.queue.onSubmittedWorkDone();
      }
    } else {
      do {
        opt.step(0, cfg.views);
        await device.queue.onSubmittedWorkDone();
        steps++;
      } while (performance.now() - trainStart < BUDGET_MS);
    }
    const trainMs = performance.now() - trainStart;
    const evalLabel = TRIALS > 1 ? `${cfg.label}_t${trial}` : cfg.label;
    const quality = await evaluateAllViews(opt, viewEmbeds, evalLabel);
    opt.destroy();
    const result: GridQualityResult = {
      label: cfg.label,
      trial,
      seed: trialSeed,
      views: cfg.views,
      clipBatch: cfg.clipBatch,
      clipLayout: cfg.clipLayout,
      gridDirectRaster: cfg.gridDirectRaster,
      gridPromptMode: cfg.gridPromptMode,
      steps,
      trainMs,
      compileMs: trainStart - compileStart,
      stepsPerSecond: (steps * 1000) / Math.max(trainMs, 1e-6),
      meanDelta: quality.meanCos - baselineEval.meanCos,
      minDelta: quality.minCos - baselineEval.minCos,
      ...quality,
    };
    results.push(result);
    console.log(
      `result ${cfg.label} t${trial}: ${cfg.views}/${cfg.clipBatch} layout=${cfg.clipLayout} direct=${cfg.gridDirectRaster ? 1 : 0} ` +
        `gridText=${cfg.gridPromptMode} steps=${steps} train=${trainMs.toFixed(0)}ms ` +
        `compile=${result.compileMs.toFixed(0)}ms stepRate=${result.stepsPerSecond.toFixed(2)}/s ` +
        `meanCos=${result.meanCos.toFixed(5)} delta=${result.meanDelta.toFixed(5)} ` +
        `minCos=${result.minCos.toFixed(5)} sheet=${result.sheetPath}`
    );
  }
}

const summary = summarizeResults(results);
console.log("summary medians:");
for (const row of summary) {
  console.log(
    `summary ${row.label}: trials=${row.trials} steps=${row.medianSteps.toFixed(0)} ` +
      `stepRate=${row.medianStepsPerSecond.toFixed(2)}/s meanCos=${row.medianMeanCos.toFixed(5)} ` +
      `meanDelta=${row.medianMeanDelta.toFixed(5)} minCos=${row.medianMinCos.toFixed(5)}`
  );
}

const jsonPath = join(OUT_DIR, "grid_quality.json");
writeFileSync(
  jsonPath,
  JSON.stringify(
    {
      date: new Date().toISOString(),
      adapter: info,
      prompt: PROMPT,
      blackBgText: BLACK_BG_TEXT,
      viewPromptTexts,
      config: {
        G,
        cap: CAP,
        seed: SEED,
        trials: TRIALS,
        viewSampler: VIEW_SAMPLER,
        clipPrecision: CLIP_PRECISION,
        weightsFile: WEIGHTS_FILE,
        budgetMs: RUN_STEPS > 0 ? null : BUDGET_MS,
        runSteps: RUN_STEPS > 0 ? RUN_STEPS : null,
      },
      baseline: baselines[0] ?? null,
      baselines,
      summary,
      results,
    },
    null,
    2
  )
);
console.log(`JSON: ${jsonPath}`);
