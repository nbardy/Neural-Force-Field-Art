/**
 * Exact optimizer parity gate for retained 80px grid raster state.
 *
 *   bun tools/splat3d/grid_retain_parity.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import type { TrainPlan } from "../../src/clip/vision";
import { randomSplats3D, Splat3DOptimizer } from "../../src/splat3d/optimize";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const G = 512;
const SEED = 17;

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: GPUDevice = await adapter.requestDevice();
const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weightsFile = readFileSync(join(MODEL_DIR, "weights_train.bin"));
const weights = new Float32Array(
  weightsFile.buffer,
  weightsFile.byteOffset,
  weightsFile.byteLength / Float32Array.BYTES_PER_ELEMENT
).slice();
const initParams = randomSplats3D(G, SEED);
for (let g = 0; g < 40; g++) initParams[7 * G + g] = -8;

const replay = await run(false);
const retained = await run(true);

let maxAbs = 0;
let meanAbs = 0;
for (let i = 0; i < replay.length; i++) {
  const d = Math.abs(replay[i] - retained[i]);
  maxAbs = Math.max(maxAbs, d);
  meanAbs += d;
}
meanAbs /= replay.length;

console.log(`retained/replay params: max=${maxAbs.toExponential(3)} mean=${meanAbs.toExponential(3)}`);
if (maxAbs > 2e-6) {
  console.error("GATE FAIL: retained grid state changes the optimizer result.");
  process.exit(1);
}
console.log("GATE PASS: retained grid state matches forward replay.");

async function run(retainGridCellState: boolean): Promise<Float32Array> {
  const opt = await Splat3DOptimizer.create(device, plan, weights, {
    G,
    cap: 512,
    seed: SEED,
    initParams,
    clipBatchSize: 3,
    clipLayout: "grid9_close2",
    gridDirectRaster: true,
    gridRasterSide: 80,
    retainGridCellState,
    convergence: {
      backgroundMode: "curriculum",
      coverageWeight: 0.05,
      transmittanceStart: 0.88,
      transmittanceEnd: 0.88,
      rayDistortionWeight: 0.1,
      mipSmoothing: true,
      adaptiveRelocation: true,
      adaptationFraction: 0.01,
    },
  });
  opt.setViewPrompts(opt.cameras.map((_camera, i) => textEmbedding(i, plan.textDim)));
  opt.setGridPrompt(textEmbedding(99, plan.textDim));
  opt.step(0, 9);
  await device.queue.onSubmittedWorkDone();
  const adaptation = await opt.adaptSplatsIfDue(true);
  if (!adaptation || adaptation.relocationCount === 0) {
    throw new Error("grid retained parity: adaptive relocation did not exercise any splats");
  }
  console.log(`${retainGridCellState ? "retained" : "replay"} adaptive relocations=${adaptation.relocationCount}`);
  const params = await opt.raster.readParams();
  opt.destroy();
  return params;
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
