/**
 * Focused smoke test for CLIP_LAYOUT=grid9_close2.
 *
 * Verifies the contact-sheet lane is populated and its gutters remain black.
 *
 *   bun tools/splat3d/grid9_close2_test.ts
 *   GRID_DIRECT_RASTER=1 bun tools/splat3d/grid9_close2_test.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { setupGlobals } from "bun-webgpu";
import type { TrainPlan } from "../../src/clip/vision";
import { Splat3DOptimizer } from "../../src/splat3d/optimize";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const SIDE = 256;
const HW = SIDE * SIDE;
const CELL = 80;
const GUTTER = 8;
const IMAGE_FLOATS = 3 * HW;
const GRID_DIRECT_RASTER = process.env.GRID_DIRECT_RASTER === "1";

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

async function readFloats(device: GPUDevice, buf: GPUBuffer, floats: number, offset = 0): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, offset, staging, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

function cellIndex(x: number, y: number): number {
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const x0 = col * (CELL + GUTTER);
      const y0 = row * (CELL + GUTTER);
      if (x >= x0 && x < x0 + CELL && y >= y0 && y < y0 + CELL) return row * 3 + col;
    }
  }
  return -1;
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
const opt = await Splat3DOptimizer.create(device, plan, weights, {
  G: 4096,
  seed: 1,
  clipBatchSize: 3,
  clipLayout: "grid9_close2",
  gridDirectRaster: GRID_DIRECT_RASTER,
});
opt.setViewPrompts(opt.cameras.map((_camera, i) => textEmbedding(i, plan.textDim)));
opt.setGridPrompt(textEmbedding(99, plan.textDim));
opt.step(0, 9);
await device.queue.onSubmittedWorkDone();

if (!opt.batchTrainer) throw new Error("grid9_close2_test: missing batch trainer");
const lane0 = await readFloats(
  device,
  opt.batchTrainer.inputBuffer,
  IMAGE_FLOATS,
  opt.batchTrainer.slotOffsetBytes(0, opt.batchTrainer.plan.inputSlot)
);

const cellMax = new Float32Array(9);
let gutterMax = 0;
let finite = 0;
for (let c = 0; c < 3; c++) {
  for (let y = 0; y < SIDE; y++) {
    for (let x = 0; x < SIDE; x++) {
      const v = lane0[c * HW + y * SIDE + x];
      if (Number.isFinite(v)) finite++;
      const a = Math.abs(v);
      const cell = cellIndex(x, y);
      if (cell >= 0) cellMax[cell] = Math.max(cellMax[cell], a);
      else gutterMax = Math.max(gutterMax, a);
    }
  }
}

let failures = 0;
for (let i = 0; i < 9; i++) {
  if (cellMax[i] <= 1e-6) {
    console.error(`FAIL cell ${i} max=${cellMax[i]}`);
    failures++;
  }
}
if (gutterMax !== 0) {
  console.error(`FAIL gutter max=${gutterMax}`);
  failures++;
}
if (finite !== lane0.length) {
  console.error(`FAIL finite=${finite}/${lane0.length}`);
  failures++;
}

console.log(`grid9_close2 lane0: cells=[${Array.from(cellMax).map((v) => v.toFixed(3)).join(", ")}] gutter=${gutterMax}`);
opt.destroy();
if (failures) process.exit(1);
console.log("PASS grid9_close2 contact sheet");
