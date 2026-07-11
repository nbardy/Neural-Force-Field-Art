/** Sequential real-prompt quality gate for the full-3D anisotropic default. */
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import ort from "onnxruntime-node";
import { AutoTokenizer, env } from "@huggingface/transformers";
import type { TrainPlan } from "../../src/clip/vision";
import { buildViewPrompt, camerasForFraming } from "../../src/splat3d/cameras";
import { Splat3DAnisotropicOptimizer } from "../../src/splat3d_aniso";
import { writePNG } from "../splat/scene";

setupGlobals();

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const MODEL_DIR = join(ROOT, "models", "mobileclip_s0");
const TEXT_MODEL = join(MODEL_DIR, "onnx", "text_model_fp16.onnx");
const HF = "https://huggingface.co/Xenova/mobileclip_s0/resolve/main";
const PROMPT = process.env.PROMPT?.trim() || "a photo of a cat";
const STEPS = Math.max(1, Number(process.env.STEPS ?? 120) | 0);
const G = Math.max(1, Number(process.env.G ?? 1024) | 0);
const SEED = Number(process.env.SEED ?? 1) | 0;
const OUT_DIR = process.env.OUT_DIR ?? "/tmp/nffa_aniso_quality";
const cameras = camerasForFraming("zoom_out");

ensureTextAssets();
mkdirSync(OUT_DIR, { recursive: true });
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("aniso quality: no WebGPU adapter");
const device = await adapter.requestDevice();
const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weightBytes = readFileSync(join(MODEL_DIR, "weights_train.bin"));
const weights = new Float32Array(
  weightBytes.buffer,
  weightBytes.byteOffset,
  weightBytes.byteLength / 4
).slice();
const embedText = await createTextEmbedder();
const promptTexts = cameras.map((camera) => buildViewPrompt(PROMPT, camera, "centered"));
const prompts: Float32Array[] = [];
for (let view = 0; view < promptTexts.length; view++) {
  console.log(`encode ${view + 1}/${promptTexts.length}: ${promptTexts[view]}`);
  prompts.push(await embedText(promptTexts[view]));
}

const variants = [
  { name: "aniso_base", convergence: {} },
  {
    name: "full3d",
    convergence: {
      centerWeight: 0.002,
      radiusWeight: 0.004,
      targetRadius: 1.15,
      smallRadiusWeight: 0.02,
      smallRadius: 0.024,
      minRadius: 0.016,
      maxRadius: 0.16,
      adaptiveRelocation: true,
      adaptationInterval: 200,
      adaptationFraction: 0.01,
    },
  },
] as const;
const results: any[] = [];
for (const variant of variants) {
  console.log(`create ${variant.name}`);
  const optimizer = await Splat3DAnisotropicOptimizer.create(device, plan, weights, {
    G,
    cap: defaultCap(G),
    seed: SEED,
    cameras,
    viewSampler: "random",
    clipBatchSize: 3,
    lrs: { position: 0.03, logScale: 0.018, quaternion: 0.01, color: 0.04, opacity: 0.025 },
    convergence: variant.convergence,
  });
  optimizer.setViewPrompts(prompts);
  await device.queue.onSubmittedWorkDone();
  const start = performance.now();
  for (let step = 0; step < STEPS; step++) {
    optimizer.step(0, 3);
    await device.queue.onSubmittedWorkDone();
    if ((step + 1) % 200 === 0) await optimizer.adaptSplatsIfDue();
  }
  const trainMs = performance.now() - start;
  const images: Float32Array[] = [];
  const viewCos: number[] = [];
  for (let view = 0; view < cameras.length; view++) {
    images.push(await optimizer.renderView(view));
    viewCos.push(cosine(await optimizer.currentEmbedding(view), prompts[view]));
  }
  const sheetPath = join(OUT_DIR, `${variant.name}.png`);
  writePNG(sheetPath, contactSheet(images, optimizer.side), optimizer.side * 3, optimizer.side * 3);
  const cloud = summarize(await optimizer.raster.readParams(), G);
  const tile = await optimizer.raster.readTileTelemetry();
  const meanCos = viewCos.reduce((sum, value) => sum + value, 0) / viewCos.length;
  const result = {
    name: variant.name,
    steps: STEPS,
    trainMs,
    stepsPerSecond: (1000 * STEPS) / trainMs,
    meanCos,
    minCos: Math.min(...viewCos),
    viewCos,
    cloud,
    adaptation: optimizer.adaptationDiagnostics,
    tile,
    sheetPath,
  };
  results.push(result);
  console.log(
    `${variant.name}: ${result.stepsPerSecond.toFixed(2)}/s mean=${meanCos.toFixed(5)} ` +
      `min=${result.minCos.toFixed(5)} opacity=${cloud.opacityMean.toFixed(3)} ` +
      `radius=${cloud.radiusMean.toFixed(4)} ratio=${cloud.axisRatioMean.toFixed(2)} ` +
      `spread=${cloud.spreadRms.toFixed(3)} adapt=${optimizer.adaptationDiagnostics?.relocationCount ?? 0} ` +
      `density=${optimizer.adaptationDiagnostics?.densityVisiblePixels ?? 0}px ` +
      `tile=${tile.maxCount}/${tile.cap} overflow=${tile.overflowTiles}`
  );
  if (tile.overflowTiles > 0) throw new Error(`${variant.name}: tile overflow in ${tile.overflowTiles} tiles`);
  console.log(`views: ${viewCos.map((value) => value.toFixed(5)).join(" ")}`);
  optimizer.destroy();
  await device.queue.onSubmittedWorkDone();
}

const base = results[0];
const full = results[1];
console.log(
  `full-base: mean=${signed(full.meanCos - base.meanCos, 5)} ` +
    `min=${signed(full.minCos - base.minCos, 5)} opacity=${signed(full.cloud.opacityMean - base.cloud.opacityMean, 4)} ` +
    `radius=${signed(full.cloud.radiusMean - base.cloud.radiusMean, 4)} spread=${signed(full.cloud.spreadRms - base.cloud.spreadRms, 4)}`
);
const jsonPath = join(OUT_DIR, "results.json");
writeFileSync(jsonPath, JSON.stringify({ prompt: PROMPT, G, steps: STEPS, promptTexts, results }, null, 2));
console.log(`RESULTS ${jsonPath}`);

function ensureTextAssets(): void {
  for (const relative of ["tokenizer.json", "tokenizer_config.json", "onnx/text_model_fp16.onnx"]) {
    const destination = join(MODEL_DIR, relative);
    if (existsSync(destination)) continue;
    mkdirSync(dirname(destination), { recursive: true });
    execSync(`curl -sfL -o "${destination}" "${HF}/${relative}"`, { stdio: "inherit" });
  }
}

async function createTextEmbedder(): Promise<(text: string) => Promise<Float32Array>> {
  env.allowRemoteModels = false;
  env.localModelPath = join(ROOT, "models");
  const tokenizer = await AutoTokenizer.from_pretrained("mobileclip_s0");
  const session = await ort.InferenceSession.create(TEXT_MODEL, { graphOptimizationLevel: "basic" });
  return async (text: string): Promise<Float32Array> => {
    const encoded = tokenizer(text, { padding: "max_length", max_length: 77, truncation: true });
    const output = await session.run({
      input_ids: new ort.Tensor("int64", encoded.input_ids.data, encoded.input_ids.dims),
    });
    return new Float32Array(output.text_embeds.data as Float32Array);
  };
}

function summarize(params: Float32Array, splats: number) {
  let opacityMean = 0;
  let radiusMean = 0;
  let axisRatioMean = 0;
  const center = [0, 0, 0];
  for (let g = 0; g < splats; g++) {
    center[0] += params[3 * g];
    center[1] += params[3 * g + 1];
    center[2] += params[3 * g + 2];
    const base = 3 * splats + 3 * g;
    const logs = [params[base], params[base + 1], params[base + 2]];
    radiusMean += Math.exp((logs[0] + logs[1] + logs[2]) / 3);
    axisRatioMean += Math.exp(Math.max(...logs) - Math.min(...logs));
    opacityMean += sigmoid(params[13 * splats + g]);
  }
  center[0] /= splats;
  center[1] /= splats;
  center[2] /= splats;
  let spread2 = 0;
  for (let g = 0; g < splats; g++) {
    spread2 +=
      (params[3 * g] - center[0]) ** 2 +
      (params[3 * g + 1] - center[1]) ** 2 +
      (params[3 * g + 2] - center[2]) ** 2;
  }
  return {
    opacityMean: opacityMean / splats,
    radiusMean: radiusMean / splats,
    axisRatioMean: axisRatioMean / splats,
    spreadRms: Math.sqrt(spread2 / splats),
    center,
  };
}

function contactSheet(images: Float32Array[], side: number): Float32Array {
  const outputSide = side * 3;
  const output = new Float32Array(3 * outputSide * outputSide);
  const sourcePixels = side * side;
  const outputPixels = outputSide * outputSide;
  for (let view = 0; view < Math.min(9, images.length); view++) {
    const offsetX = (view % 3) * side;
    const offsetY = Math.floor(view / 3) * side;
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        const source = y * side + x;
        const destination = (offsetY + y) * outputSide + offsetX + x;
        for (let channel = 0; channel < 3; channel++) {
          output[channel * outputPixels + destination] = images[view][channel * sourcePixels + source];
        }
      }
    }
  }
  return output;
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return dot / Math.sqrt(Math.max(aa * bb, 1e-20));
}

function sigmoid(value: number): number {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

function signed(value: number, digits: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function defaultCap(splats: number): number {
  let cap = 256;
  while (cap < splats && cap < 4096) cap *= 2;
  return cap;
}
