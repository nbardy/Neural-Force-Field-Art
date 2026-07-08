/**
 * Compare f16-weight CLIP against the current f32-weight CLIP on the same
 * input/text pair. This is the correctness gate for the weights-only f16 fork.
 *
 *   bun tools/clip/pack_f16_weights.ts
 *   bun tools/clip/f16_compare.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import { VisionTrainer, type TrainPlan } from "../../src/clip/vision";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const STRICT = process.env.STRICT !== "0";

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

async function readback(device: GPUDevice, buf: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

function compare(a: Float32Array, b: Float32Array): {
  maxDiff: number;
  rel: number;
  cos: number;
  normA: number;
  normB: number;
} {
  let maxDiff = 0;
  let scale = 1e-8;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    maxDiff = Math.max(maxDiff, Math.abs(x - y));
    scale = Math.max(scale, Math.abs(y));
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return { maxDiff, rel: maxDiff / scale, cos: dot / Math.sqrt(na * nb), normA: Math.sqrt(na), normB: Math.sqrt(nb) };
}

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("f16_compare: no WebGPU adapter");
if (!adapter.features.has("shader-f16")) {
  throw new Error("f16_compare: adapter lacks shader-f16");
}
const device: GPUDevice = await adapter.requestDevice({
  requiredFeatures: ["shader-f16"] as GPUFeatureName[],
});
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const input = f32File(join(MODEL_DIR, "fixtures", "input_1x3x256x256.f32.bin"));
const text = textEmbedding(0, plan.textDim);
const f32Weights = f32File(join(MODEL_DIR, "weights_train.bin"));
const f16Weights = f16File(join(MODEL_DIR, "weights_train_f16.bin"));
const inputFloats = plan.inputShape[0] * plan.inputShape[1] * plan.inputShape[2];

const f32 = await VisionTrainer.create(device, plan, f32Weights, {
  stemSpatialBwd: true,
  weightPrecision: "f32",
});
const f16 = await VisionTrainer.create(device, plan, f16Weights, {
  stemSpatialBwd: true,
  weightPrecision: "f16",
});

f32.writeInput(input);
f32.writeText(text);
f32.run({ backward: true });

f16.writeInput(input);
f16.writeText(text);
f16.run({ backward: true });

await device.queue.onSubmittedWorkDone();

const f32Embed = await readback(device, f32.outputBuffer, plan.embedDim);
const f16Embed = await readback(device, f16.outputBuffer, plan.embedDim);
const f32Grad = await readback(device, f32.inputGradBuffer, inputFloats);
const f16Grad = await readback(device, f16.inputGradBuffer, inputFloats);

const embed = compare(f16Embed, f32Embed);
const grad = compare(f16Grad, f32Grad);
console.log(
  `embedding: cos=${embed.cos.toFixed(8)} relLinf=${embed.rel.toExponential(3)} ` +
    `maxDiff=${embed.maxDiff.toExponential(3)} norm(f16/f32)=` +
    `${embed.normA.toExponential(3)}/${embed.normB.toExponential(3)}`
);
console.log(
  `inputGrad : cos=${grad.cos.toFixed(8)} relLinf=${grad.rel.toExponential(3)} ` +
    `maxDiff=${grad.maxDiff.toExponential(3)} norm(f16/f32)=` +
    `${grad.normA.toExponential(3)}/${grad.normB.toExponential(3)}`
);

f32.destroy();
f16.destroy();

if (STRICT && (embed.cos < 0.9995 || grad.cos < 0.995)) {
  throw new Error("f16_compare: f16 weights failed cosine gates");
}
