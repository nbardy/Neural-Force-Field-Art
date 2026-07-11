/**
 * Sequential real-prompt quality gate for the live 3D convergence priors.
 *
 * Both variants start from byte-identical splat parameters and use the same
 * MobileCLIP text embeddings. The convergence run enables weak ray compactness,
 * coarse-to-fine screen-space smoothing, and staged geometry/appearance rates.
 * A procedural training background is opt-in through BACKGROUND_MODE.
 *
 *   bun tools/splat3d/convergence_quality.ts
 *   STEPS=100 PROMPT="a carved jade owl" bun tools/splat3d/convergence_quality.ts
 *   STEPS=100 BACKGROUND_MODE=blurred_noise bun tools/splat3d/convergence_quality.ts
 *   WALL_MS=5000 bun tools/splat3d/convergence_quality.ts
 */
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import ort from "onnxruntime-node";
import { AutoTokenizer, env } from "@huggingface/transformers";
import type { TrainPlan } from "../../src/clip/vision";
import { buildViewPrompt, DEFAULT_3D_CAMERAS } from "../../src/splat3d/cameras";
import {
  randomSplats3D,
  Splat3DOptimizer,
  type Splat3DBackgroundMode,
  type Splat3DConvergenceConfig,
} from "../../src/splat3d/optimize";
import { PARAM_STRIDE_3D } from "../../src/splat3d/raster_wgsl";
import { writePNG } from "../splat/scene";

setupGlobals();

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const MODEL_DIR = join(ROOT, "models", "mobileclip_s0");
const TEXT_MODEL = join(MODEL_DIR, "onnx", "text_model_fp16.onnx");
const HF = "https://huggingface.co/Xenova/mobileclip_s0/resolve/main";
const CONTEXT_LENGTH = 77;

const PROMPT = process.env.PROMPT?.trim() || "a photo of a cat";
const STEPS = positiveInteger(process.env.STEPS, 24);
const WALL_MS = nonNegativeNumber(process.env.WALL_MS, 0);
const G = positiveInteger(process.env.G, 1024);
const CAP = positiveInteger(process.env.CAP, Math.min(2048, Math.max(512, G)));
const SEED = integer(process.env.SEED, 1);
const VIEWS_PER_STEP = positiveInteger(process.env.VIEWS, 3);
const RAY_WEIGHT = nonNegativeNumber(process.env.RAY_WEIGHT, 0.02);
const MIP_SMOOTHING = process.env.MIP_SMOOTHING !== "0";
const STAGED_RATES = process.env.STAGED_RATES !== "0";
const BLACK_BG_TEXT = process.env.BLACK_BG_TEXT !== "0";
const OUT_DIR = process.env.OUT_DIR ?? "/tmp/nffa_convergence_quality";
const BACKGROUND_MODE = parseBackgroundMode(process.env.BACKGROUND_MODE);

type VariantName = "base" | "convergence";

interface CloudStats {
  opacityMean: number;
  opacityP10: number;
  opacityP90: number;
  radiusMean: number;
  radiusP10: number;
  radiusP90: number;
  spreadRms: number;
  center: [number, number, number];
}

interface QualityStats {
  meanCos: number;
  minCos: number;
  maxCos: number;
  viewCos: number[];
  sheetPath: string;
}

interface VariantResult extends QualityStats {
  variant: VariantName;
  steps: number;
  trainMs: number;
  stepsPerSecond: number;
  cloud: CloudStats;
  convergence: Splat3DConvergenceConfig;
}

function integer(value: string | undefined, fallback: number): number {
  if (value === undefined || value.trim() === "") return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`convergence_quality: expected integer, got '${value}'`);
  return parsed | 0;
}

function positiveInteger(value: string | undefined, fallback: number): number {
  const parsed = integer(value, fallback);
  if (parsed <= 0) throw new Error(`convergence_quality: expected a positive integer, got '${value}'`);
  return parsed;
}

function nonNegativeNumber(value: string | undefined, fallback: number): number {
  if (value === undefined || value.trim() === "") return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`convergence_quality: expected a non-negative number, got '${value}'`);
  }
  return parsed;
}

function parseBackgroundMode(value: string | undefined): Splat3DBackgroundMode {
  if (value === undefined || value === "" || value === "black") return "black";
  if (value === "blurred_noise" || value === "checkerboard" || value === "fourier" || value === "curriculum") {
    return value;
  }
  throw new Error(
    `convergence_quality: BACKGROUND_MODE must be black, blurred_noise, checkerboard, fourier, or curriculum; got '${value}'`
  );
}

function f32File(path: string): Float32Array {
  const bytes = readFileSync(path);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}

function ensureTextAssets(): void {
  for (const relative of ["tokenizer.json", "tokenizer_config.json", "onnx/text_model_fp16.onnx"]) {
    const destination = join(MODEL_DIR, relative);
    if (existsSync(destination)) continue;
    mkdirSync(dirname(destination), { recursive: true });
    console.error(`convergence_quality: fetching ${relative}`);
    execSync(`curl -sfL -o "${destination}" "${HF}/${relative}"`, { stdio: "inherit" });
  }
}

async function createTextEmbedder(): Promise<(text: string) => Promise<Float32Array>> {
  ensureTextAssets();
  env.allowRemoteModels = false;
  env.localModelPath = join(ROOT, "models");
  const tokenizer = await AutoTokenizer.from_pretrained("mobileclip_s0");
  const session = await ort.InferenceSession.create(TEXT_MODEL, { graphOptimizationLevel: "basic" });
  const cache = new Map<string, Promise<Float32Array>>();
  return (text: string): Promise<Float32Array> => {
    const key = text.trim();
    const existing = cache.get(key);
    if (existing) return existing;
    const pending = (async () => {
      const encoded = tokenizer(key, {
        padding: "max_length",
        max_length: CONTEXT_LENGTH,
        truncation: true,
      });
      const output = await session.run({
        input_ids: new ort.Tensor("int64", encoded.input_ids.data, encoded.input_ids.dims),
      });
      return new Float32Array(output.text_embeds.data as Float32Array);
    })();
    cache.set(key, pending);
    return pending;
  };
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let aa = 0;
  let bb = 0;
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i++) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return dot / Math.sqrt(Math.max(aa * bb, 1e-20));
}

function sigmoid(value: number): number {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

function percentile(sorted: Float32Array, fraction: number): number {
  return sorted[Math.floor((sorted.length - 1) * fraction)];
}

function summarizeCloud(params: Float32Array): CloudStats {
  if (params.length % PARAM_STRIDE_3D !== 0) {
    throw new Error(`convergence_quality: parameter length ${params.length} is not divisible by ${PARAM_STRIDE_3D}`);
  }
  const splats = params.length / PARAM_STRIDE_3D;
  const opacities = new Float32Array(splats);
  const radii = new Float32Array(splats);
  const center: [number, number, number] = [0, 0, 0];
  let opacityMean = 0;
  let radiusMean = 0;
  for (let g = 0; g < splats; g++) {
    center[0] += params[g * 3 + 0];
    center[1] += params[g * 3 + 1];
    center[2] += params[g * 3 + 2];
    const opacity = sigmoid(params[7 * splats + g]);
    const radius = Math.exp(params[3 * splats + g]);
    opacities[g] = opacity;
    radii[g] = radius;
    opacityMean += opacity;
    radiusMean += radius;
  }
  center[0] /= splats;
  center[1] /= splats;
  center[2] /= splats;
  opacityMean /= splats;
  radiusMean /= splats;
  let spread2 = 0;
  for (let g = 0; g < splats; g++) {
    const dx = params[g * 3 + 0] - center[0];
    const dy = params[g * 3 + 1] - center[1];
    const dz = params[g * 3 + 2] - center[2];
    spread2 += dx * dx + dy * dy + dz * dz;
  }
  opacities.sort();
  radii.sort();
  return {
    opacityMean,
    opacityP10: percentile(opacities, 0.1),
    opacityP90: percentile(opacities, 0.9),
    radiusMean,
    radiusP10: percentile(radii, 0.1),
    radiusP90: percentile(radii, 0.9),
    spreadRms: Math.sqrt(spread2 / splats),
    center,
  };
}

function makeContactSheet(images: Float32Array[], side: number): Float32Array {
  const outputSide = side * 3;
  const output = new Float32Array(3 * outputSide * outputSide);
  const srcPixels = side * side;
  const dstPixels = outputSide * outputSide;
  for (let view = 0; view < Math.min(9, images.length); view++) {
    const source = images[view];
    const offsetX = (view % 3) * side;
    const offsetY = Math.floor(view / 3) * side;
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        const sourceIndex = y * side + x;
        const destinationIndex = (offsetY + y) * outputSide + offsetX + x;
        for (let channel = 0; channel < 3; channel++) {
          output[channel * dstPixels + destinationIndex] = source[channel * srcPixels + sourceIndex];
        }
      }
    }
  }
  return output;
}

async function evaluate(
  optimizer: Splat3DOptimizer,
  textEmbeddings: Float32Array[],
  variant: VariantName
): Promise<QualityStats> {
  const viewCos: number[] = [];
  const images: Float32Array[] = [];
  for (let view = 0; view < optimizer.cameras.length; view++) {
    const imageEmbedding = await optimizer.currentEmbedding(view);
    viewCos.push(cosine(imageEmbedding, textEmbeddings[view]));
    images.push(await optimizer.renderView(view));
  }
  await optimizer.device.queue.onSubmittedWorkDone();
  const meanCos = viewCos.reduce((sum, value) => sum + value, 0) / viewCos.length;
  const sheetPath = join(OUT_DIR, `${variant}_views.png`);
  writePNG(sheetPath, makeContactSheet(images, optimizer.side), optimizer.side * 3, optimizer.side * 3);
  return {
    meanCos,
    minCos: Math.min(...viewCos),
    maxCos: Math.max(...viewCos),
    viewCos,
    sheetPath,
  };
}

const convergencePreset: Splat3DConvergenceConfig = {
  backgroundMode: BACKGROUND_MODE,
  rayDistortionWeight: RAY_WEIGHT,
  mipSmoothing: MIP_SMOOTHING,
  mipVarianceStart: 4,
  mipVarianceEnd: 0.0625,
  mipAnnealSteps: 500,
  stagedOptimization: STAGED_RATES,
  geometryWarmupSteps: 250,
  geometryDecaySteps: 1000,
  geometryFinalScale: 0.2,
  appearanceWarmupScale: 0.35,
};

const variants: Array<{ name: VariantName; convergence: Splat3DConvergenceConfig }> = [
  { name: "base", convergence: { backgroundMode: "black" } },
  { name: "convergence", convergence: convergencePreset },
];

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device = await adapter.requestDevice();
const adapterInfo = adapter.info ?? {};
mkdirSync(OUT_DIR, { recursive: true });

const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = f32File(join(MODEL_DIR, "weights_train.bin"));
const embedText = await createTextEmbedder();
const promptTexts = DEFAULT_3D_CAMERAS.map((camera) => buildViewPrompt(PROMPT, camera, BLACK_BG_TEXT));
const promptEmbeddings: Float32Array[] = [];

console.log(`adapter: ${adapterInfo.vendor ?? "?"} ${adapterInfo.architecture ?? "?"}`);
console.log(`convergence_quality: prompt="${PROMPT}" G=${G} cap=${CAP} viewsPerStep=${VIEWS_PER_STEP}`);
console.log(
  `convergence_quality: budget=${WALL_MS > 0 ? `${WALL_MS}ms wall` : `${STEPS} steps`} ` +
    `rayWeight=${RAY_WEIGHT} mip=${MIP_SMOOTHING ? 1 : 0} staged=${STAGED_RATES ? 1 : 0} ` +
    `background=${BACKGROUND_MODE}`
);
for (let view = 0; view < promptTexts.length; view++) {
  console.log(`encode view ${view + 1}/9: ${promptTexts[view]}`);
  promptEmbeddings.push(await embedText(promptTexts[view]));
}

const initialParams = randomSplats3D(G, SEED);
const initialCloud = summarizeCloud(initialParams);
const results: VariantResult[] = [];

for (const variant of variants) {
  await device.queue.onSubmittedWorkDone();
  console.log(`run ${variant.name}: create`);
  const optimizer = await Splat3DOptimizer.create(device, plan, weights, {
    G,
    cap: CAP,
    seed: SEED,
    initParams: initialParams.slice(),
    clipBatchSize: 3,
    clipLayout: "per_view",
    viewSampler: "epoch",
    convergence: variant.convergence,
  });
  optimizer.setViewPrompts(promptEmbeddings);
  await device.queue.onSubmittedWorkDone();

  let completedSteps = 0;
  const trainStart = performance.now();
  do {
    optimizer.step(0, VIEWS_PER_STEP);
    await device.queue.onSubmittedWorkDone();
    completedSteps++;
  } while (WALL_MS > 0 ? performance.now() - trainStart < WALL_MS : completedSteps < STEPS);
  const trainMs = performance.now() - trainStart;

  const quality = await evaluate(optimizer, promptEmbeddings, variant.name);
  const cloud = summarizeCloud(await optimizer.raster.readParams());
  const result: VariantResult = {
    variant: variant.name,
    steps: completedSteps,
    trainMs,
    stepsPerSecond: (completedSteps * 1000) / Math.max(trainMs, 1e-6),
    cloud,
    convergence: variant.convergence,
    ...quality,
  };
  results.push(result);
  console.log(
    `result ${variant.name}: steps=${completedSteps} train=${trainMs.toFixed(0)}ms ` +
      `rate=${result.stepsPerSecond.toFixed(2)}/s meanCos=${result.meanCos.toFixed(5)} ` +
      `minCos=${result.minCos.toFixed(5)} opacity=${cloud.opacityMean.toFixed(4)} ` +
      `radius=${cloud.radiusMean.toFixed(4)} spread=${cloud.spreadRms.toFixed(4)}`
  );
  console.log(`views ${variant.name}: ${result.viewCos.map((value) => value.toFixed(5)).join(" ")}`);

  optimizer.destroy();
  await device.queue.onSubmittedWorkDone();
}

const base = results.find((result) => result.variant === "base")!;
const convergence = results.find((result) => result.variant === "convergence")!;
const comparison = {
  meanCosDelta: convergence.meanCos - base.meanCos,
  minCosDelta: convergence.minCos - base.minCos,
  opacityMeanDelta: convergence.cloud.opacityMean - base.cloud.opacityMean,
  radiusMeanDelta: convergence.cloud.radiusMean - base.cloud.radiusMean,
  spreadRmsDelta: convergence.cloud.spreadRms - base.cloud.spreadRms,
};

console.log(
  `comparison convergence-base: meanCos=${signed(comparison.meanCosDelta, 5)} ` +
    `minCos=${signed(comparison.minCosDelta, 5)} opacity=${signed(comparison.opacityMeanDelta, 4)} ` +
    `radius=${signed(comparison.radiusMeanDelta, 4)} spread=${signed(comparison.spreadRmsDelta, 4)}`
);

const jsonPath = join(OUT_DIR, "convergence_quality.json");
writeFileSync(
  jsonPath,
  JSON.stringify(
    {
      date: new Date().toISOString(),
      adapter: adapterInfo,
      prompt: PROMPT,
      promptTexts,
      config: {
        steps: WALL_MS > 0 ? null : STEPS,
        wallMs: WALL_MS > 0 ? WALL_MS : null,
        G,
        cap: CAP,
        seed: SEED,
        viewsPerStep: VIEWS_PER_STEP,
        blackBackgroundText: BLACK_BG_TEXT,
        convergencePreset,
      },
      initialCloud,
      comparison,
      results,
    },
    null,
    2
  )
);
console.log(`JSON: ${jsonPath}`);

function signed(value: number, digits: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}
