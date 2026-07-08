/**
 * Parity gate for the 3D view-lane raster forward ablation.
 *
 * It renders the same views two ways:
 *   1. existing per-view `recordForward(..., laneIO)`
 *   2. new `recordBatchForward(..., views)`
 *
 * Then it applies identical synthetic image gradients through both the existing
 * per-lane backward path and the batched tile-backward path, comparing rendered
 * images and raw splat gradients.
 *
 *   bun tools/splat3d/raster_batch_forward_test.ts
 */
import { setupGlobals } from "bun-webgpu";
import { DEFAULT_3D_CAMERAS, prepareCamera } from "../../src/splat3d/cameras";
import { Raster3DEngine } from "../../src/splat3d/raster";
import { LEGIBLE_3D_INIT, randomSplats3D } from "../../src/splat3d/optimize";
import { PARAM_STRIDE_3D } from "../../src/splat3d/raster_wgsl";

setupGlobals();

const U = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };
const SIDE = 256;
const G = Number(process.env.G ?? 1024);
const LANES = Number(process.env.LANES ?? 3);
const VIEWS = Array.from({ length: LANES }, (_unused, i) => i);
const IMG_FLOATS = 3 * SIDE * SIDE;

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: GPUDevice = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

function makeBuffer(label: string, floats: number, extra = 0): GPUBuffer {
  return device.createBuffer({ label, size: floats * 4, usage: U.STORAGE | extra });
}

async function readFloats(buffer: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: U.MAP_READ | U.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, staging, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

function syntheticGrad(floats: number): Float32Array {
  const out = new Float32Array(floats);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.sin(i * 0.017 + 1.3) * 0.01 + Math.cos(i * 0.003 + 0.7) * 0.004;
  }
  return out;
}

function diffStats(a: Float32Array, b: Float32Array): { maxAbs: number; meanAbs: number } {
  if (a.length !== b.length) throw new Error(`diff length mismatch ${a.length} != ${b.length}`);
  let maxAbs = 0;
  let sumAbs = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    maxAbs = Math.max(maxAbs, d);
    sumAbs += d;
  }
  return { maxAbs, meanAbs: sumAbs / Math.max(1, a.length) };
}

async function runSeparateBackward(
  raster: Raster3DEngine,
  state: Awaited<ReturnType<Raster3DEngine["createBatchForwardState"]>>
): Promise<Float32Array> {
  const enc = device.createCommandEncoder();
  raster.recordClearRawGrad(enc);
  for (let lane = 0; lane < VIEWS.length; lane++) {
    raster.recordBackwardAdd(enc, VIEWS[lane], state.ios[lane]);
  }
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  return readFloats(raster.gradRaw, G * PARAM_STRIDE_3D);
}

async function runBatchBackward(
  raster: Raster3DEngine,
  state: Awaited<ReturnType<Raster3DEngine["createBatchForwardState"]>>
): Promise<Float32Array> {
  const enc = device.createCommandEncoder();
  raster.recordClearRawGrad(enc);
  raster.recordBatchBackwardAdd(enc, state, VIEWS);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  return readFloats(raster.gradRaw, G * PARAM_STRIDE_3D);
}

const cameras = DEFAULT_3D_CAMERAS.map((camera) => prepareCamera(camera, SIDE));
const raster = await Raster3DEngine.create(device, {
  H: SIDE,
  W: SIDE,
  G,
  cap: 2048,
  cameras,
  bg: [0, 0, 0],
});
raster.setParams(randomSplats3D(G, 7, LEGIBLE_3D_INIT));

const imageBuffer = makeBuffer("splat3d-batch-forward-test-image", IMG_FLOATS * LANES, U.COPY_SRC);
const gradBuffer = makeBuffer("splat3d-batch-forward-test-grad", IMG_FLOATS * LANES, U.COPY_DST);
device.queue.writeBuffer(gradBuffer, 0, syntheticGrad(IMG_FLOATS * LANES) as unknown as BufferSource);

const imageOffsets = Array.from({ length: LANES }, (_unused, lane) => lane * IMG_FLOATS * 4);
const state = await raster.createBatchForwardState({
  lanes: LANES,
  imageBuffer,
  imageOffsets,
  gradBuffer,
  gradOffsets: imageOffsets,
});

{
  const enc = device.createCommandEncoder();
  for (let lane = 0; lane < VIEWS.length; lane++) {
    raster.recordForward(enc, VIEWS[lane], state.ios[lane]);
  }
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const separateImage = await readFloats(imageBuffer, IMG_FLOATS * LANES);
const separateGrad = await runSeparateBackward(raster, state);

{
  const enc = device.createCommandEncoder();
  raster.recordBatchForward(enc, state, VIEWS);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const batchImage = await readFloats(imageBuffer, IMG_FLOATS * LANES);
const batchGrad = await runSeparateBackward(raster, state);
const batchBackwardGrad = await runBatchBackward(raster, state);

const imageDiff = diffStats(separateImage, batchImage);
const gradDiff = diffStats(separateGrad, batchGrad);
const batchBackwardDiff = diffStats(separateGrad, batchBackwardGrad);
console.log(`image diff: max=${imageDiff.maxAbs.toExponential(3)} mean=${imageDiff.meanAbs.toExponential(3)}`);
console.log(`grad diff:  max=${gradDiff.maxAbs.toExponential(3)} mean=${gradDiff.meanAbs.toExponential(3)}`);
console.log(
  `batch backward diff: max=${batchBackwardDiff.maxAbs.toExponential(3)} mean=${batchBackwardDiff.meanAbs.toExponential(3)}`
);

raster.destroy();
imageBuffer.destroy();
gradBuffer.destroy();

if (imageDiff.maxAbs > 2e-5 || gradDiff.maxAbs > 2e-3 || batchBackwardDiff.maxAbs > 2e-3) {
  console.error("GATE FAIL: view-lane raster path does not match separate per-view path.");
  process.exit(1);
}
console.log("GATE PASS: view-lane raster forward/backward matches separate per-view path.");
