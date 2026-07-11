/**
 * Regression gate for the 3D centroid regularizer.
 *
 * The centering gradient must be identical for every splat. A gradient based
 * on each individual position contracts the cloud into the tiny-ball failure.
 *
 *   bun tools/splat3d/regularizer_test.ts
 */
import { setupGlobals } from "bun-webgpu";
import { DEFAULT_3D_CAMERAS, prepareCamera } from "../../src/splat3d/cameras";
import { Raster3DEngine, type Raster3DRegularizerOptions } from "../../src/splat3d/raster";
import { CENTER_SUM_SCALE_3D, PARAM_STRIDE_3D } from "../../src/splat3d/raster_wgsl";

setupGlobals();

const U = { MAP_READ: 1, COPY_DST: 8 };
const SIDE = 32;
const G = 256;
const WEIGHT = 0.1;
const EPS = 2e-6;

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: GPUDevice = await adapter.requestDevice();
const camera = prepareCamera(DEFAULT_3D_CAMERAS[0], SIDE);
const raster = await Raster3DEngine.create(device, {
  H: SIDE,
  W: SIDE,
  G,
  cap: 256,
  cameras: [camera],
  bg: [0, 0, 0],
});

const params = makeParams();
raster.setParams(params);

const opts: Raster3DRegularizerOptions = {
  centerWeight: WEIGHT,
  radiusWeight: 0,
  targetRadius: 1.15,
  opacitySparsity: 0,
  smallRadiusWeight: 0,
  smallRadius: 0.024,
  radiusBandWeight: 0,
  minRadius: 0.016,
  maxRadius: 0.16,
};

const enc = device.createCommandEncoder();
raster.recordClearRawGrad(enc);
raster.recordRegularizerAdd(enc, opts);
device.queue.submit([enc.finish()]);
await device.queue.onSubmittedWorkDone();

const grad = await readFloats(raster.gradRaw, G * PARAM_STRIDE_3D);
const center = quantizedCenter(params);
const expected: [number, number, number] = [
  2 * WEIGHT * center[0],
  2 * WEIGHT * center[1],
  2 * WEIGHT * center[2],
];

let maxPositionError = 0;
let maxPairwiseGradientDelta = 0;
for (let g = 0; g < G; g++) {
  for (let axis = 0; axis < 3; axis++) {
    const actual = grad[g * 3 + axis];
    maxPositionError = Math.max(maxPositionError, Math.abs(actual - expected[axis]));
    maxPairwiseGradientDelta = Math.max(maxPairwiseGradientDelta, Math.abs(actual - grad[axis]));
  }
}

let maxOtherGradient = 0;
for (let i = 3 * G; i < grad.length; i++) maxOtherGradient = Math.max(maxOtherGradient, Math.abs(grad[i]));

console.log(`quantized centroid: ${center.map((v) => v.toFixed(6)).join(", ")}`);
console.log(`expected gradient:  ${expected.map((v) => v.toFixed(6)).join(", ")}`);
console.log(`max position error: ${maxPositionError.toExponential(3)}`);
console.log(`max pair delta:     ${maxPairwiseGradientDelta.toExponential(3)}`);
console.log(`max other gradient: ${maxOtherGradient.toExponential(3)}`);

raster.destroy();

if (maxPositionError > EPS || maxPairwiseGradientDelta > EPS || maxOtherGradient > EPS) {
  console.error("GATE FAIL: centroid regularizer can contract or corrupt the splat cloud.");
  process.exit(1);
}
console.log("GATE PASS: centering translates the cloud without contracting its spread.");

function makeParams(): Float32Array {
  const out = new Float32Array(G * PARAM_STRIDE_3D);
  for (let g = 0; g < G; g++) {
    const t = g / (G - 1);
    out[g * 3 + 0] = 0.37 + (t - 0.5) * 1.2;
    out[g * 3 + 1] = -0.21 + Math.sin(g * 0.31) * 0.42;
    out[g * 3 + 2] = 0.13 + Math.cos(g * 0.17) * 0.33;
    out[3 * G + g] = Math.log(0.05);
    out[7 * G + g] = 0;
  }
  out[(G - 1) * 3 + 0] = 40;
  out[(G - 1) * 3 + 1] = -30;
  out[(G - 1) * 3 + 2] = 20;
  out[7 * G + (G - 1)] = -12;
  return out;
}

function quantizedCenter(params: Float32Array): [number, number, number] {
  const sum = [0, 0, 0];
  let massSum = 0;
  for (let g = 0; g < G; g++) {
    const mass = 1 / (1 + Math.exp(-params[7 * G + g]));
    for (let axis = 0; axis < 3; axis++) {
      sum[axis] += Math.round(params[g * 3 + axis] * mass * CENTER_SUM_SCALE_3D);
    }
    massSum += Math.round(mass * CENTER_SUM_SCALE_3D);
  }
  return [
    sum[0] / massSum,
    sum[1] / massSum,
    sum[2] / massSum,
  ];
}

async function readFloats(buffer: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: U.MAP_READ | U.COPY_DST });
  const read = device.createCommandEncoder();
  read.copyBufferToBuffer(buffer, 0, staging, 0, floats * 4);
  device.queue.submit([read.finish()]);
  await staging.mapAsync(1);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}
