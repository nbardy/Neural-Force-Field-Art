/**
 * Standalone anisotropic raster gate.
 *
 *   bun tools/splat3d/aniso_raster_test.ts
 */
import { setupGlobals } from "bun-webgpu";
import { DEFAULT_3D_CAMERAS, prepareCamera } from "../../src/splat3d/cameras";
import { Raster3DEngine } from "../../src/splat3d/raster";
import { PARAM_STRIDE_3D } from "../../src/splat3d/raster_wgsl";
import { ANISO_PARAM_STRIDE_3D } from "../../src/splat3d_aniso/layout";
import { AnisotropicRaster3DEngine } from "../../src/splat3d_aniso/raster_engine";

setupGlobals();

const SIDE = 64;
const G = 12;
const CAP = 512;
const IMAGE_WORDS = 3 * SIDE * SIDE;

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("aniso_raster_test: no WebGPU adapter");
const device = await adapter.requestDevice();
const camera = prepareCamera(DEFAULT_3D_CAMERAS[1], SIDE);

function makeIsotropicParams(): Float32Array {
  const params = new Float32Array(G * PARAM_STRIDE_3D);
  for (let g = 0; g < G; g++) {
    const column = (g % 4) - 1.5;
    const row = Math.floor(g / 4) - 1;
    params[3 * g + 0] = column * 0.21;
    params[3 * g + 1] = row * 0.21;
    params[3 * g + 2] = ((g * 7) % 5 - 2) * 0.035;
    params[3 * G + g] = Math.log(0.095 + 0.007 * (g % 3));
    params[4 * G + 3 * g + 0] = Math.sin(g * 0.71) * 0.9;
    params[4 * G + 3 * g + 1] = Math.cos(g * 0.53) * 0.8;
    params[4 * G + 3 * g + 2] = Math.sin(g * 0.37 + 1.1) * 0.7;
    params[7 * G + g] = 0.4 + 0.08 * (g % 4);
  }
  return params;
}

function liftIsotropicParams(params: Float32Array): Float32Array {
  const lifted = new Float32Array(G * ANISO_PARAM_STRIDE_3D);
  lifted.set(params.subarray(0, 3 * G), 0);
  for (let g = 0; g < G; g++) {
    const radius = params[3 * G + g];
    lifted[3 * G + 3 * g + 0] = radius;
    lifted[3 * G + 3 * g + 1] = radius;
    lifted[3 * G + 3 * g + 2] = radius;
    lifted[6 * G + 4 * g + 0] = 0;
    lifted[6 * G + 4 * g + 1] = 0;
    lifted[6 * G + 4 * g + 2] = 0;
    lifted[6 * G + 4 * g + 3] = 1;
    lifted[10 * G + 3 * g + 0] = params[4 * G + 3 * g + 0];
    lifted[10 * G + 3 * g + 1] = params[4 * G + 3 * g + 1];
    lifted[10 * G + 3 * g + 2] = params[4 * G + 3 * g + 2];
    lifted[13 * G + g] = params[7 * G + g];
  }
  return lifted;
}

function syntheticGradient(): Float32Array {
  const gradient = new Float32Array(IMAGE_WORDS);
  for (let i = 0; i < gradient.length; i++) {
    gradient[i] = 0.006 * Math.sin(i * 0.031 + 0.4) + 0.003 * Math.cos(i * 0.007 - 0.2);
  }
  return gradient;
}

function diffStats(a: Float32Array, b: Float32Array): { max: number; mean: number } {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} != ${b.length}`);
  let max = 0;
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    const delta = Math.abs(a[i] - b[i]);
    max = Math.max(max, delta);
    total += delta;
  }
  return { max, mean: total / Math.max(1, a.length) };
}

function selected(values: Float32Array, offset: number, length: number): Float32Array {
  return values.slice(offset, offset + length);
}

function summedScaleGrad(values: Float32Array): Float32Array {
  const out = new Float32Array(G);
  for (let g = 0; g < G; g++) {
    out[g] = values[3 * G + 3 * g] + values[3 * G + 3 * g + 1] + values[3 * G + 3 * g + 2];
  }
  return out;
}

const iso = await Raster3DEngine.create(device, {
  H: SIDE,
  W: SIDE,
  G,
  cap: CAP,
  cameras: [camera],
  bg: [0, 0, 0],
});
const aniso = await AnisotropicRaster3DEngine.create(device, {
  H: SIDE,
  W: SIDE,
  G,
  cap: CAP,
  cameras: [camera],
  bg: [0, 0, 0],
  projectionMode: "legacy-affine",
});

const isoParams = makeIsotropicParams();
const anisoParams = liftIsotropicParams(isoParams);
const imageGradient = syntheticGradient();
iso.setParams(isoParams);
aniso.setParams(anisoParams);
device.queue.writeBuffer(iso.gradImage, 0, imageGradient as unknown as BufferSource);
device.queue.writeBuffer(aniso.gradImage, 0, imageGradient as unknown as BufferSource);

{
  const enc = device.createCommandEncoder();
  iso.recordForward(enc, 0);
  aniso.recordForward(enc, 0);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const isoImage = await iso.readImage();
const anisoImage = await aniso.readImage();
const imageDiff = diffStats(isoImage, anisoImage);

{
  const enc = device.createCommandEncoder();
  iso.recordClearRawGrad(enc);
  aniso.recordClearRawGrad(enc);
  iso.recordBackwardAdd(enc, 0);
  aniso.recordBackwardAdd(enc, 0);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const isoGrad = await iso.readRawGrad();
const anisoGrad = await aniso.readRawGrad();
const positionDiff = diffStats(selected(isoGrad, 0, 3 * G), selected(anisoGrad, 0, 3 * G));
const scaleDiff = diffStats(selected(isoGrad, 3 * G, G), summedScaleGrad(anisoGrad));
const colorDiff = diffStats(selected(isoGrad, 4 * G, 3 * G), selected(anisoGrad, 10 * G, 3 * G));
const opacityDiff = diffStats(selected(isoGrad, 7 * G, G), selected(anisoGrad, 13 * G, G));
let quaternionMagnitude = 0;
for (const value of selected(anisoGrad, 6 * G, 4 * G)) quaternionMagnitude = Math.max(quaternionMagnitude, Math.abs(value));

console.log(`isotropic image:    max=${imageDiff.max.toExponential(3)} mean=${imageDiff.mean.toExponential(3)}`);
console.log(`isotropic position: max=${positionDiff.max.toExponential(3)} mean=${positionDiff.mean.toExponential(3)}`);
console.log(`isotropic scale:    max=${scaleDiff.max.toExponential(3)} mean=${scaleDiff.mean.toExponential(3)}`);
console.log(`isotropic color:    max=${colorDiff.max.toExponential(3)} mean=${colorDiff.mean.toExponential(3)}`);
console.log(`isotropic opacity:  max=${opacityDiff.max.toExponential(3)} mean=${opacityDiff.mean.toExponential(3)}`);
console.log(`isotropic quaternion max=${quaternionMagnitude.toExponential(3)}`);

if (
  imageDiff.max > 2e-5 ||
  positionDiff.max > 2e-3 ||
  scaleDiff.max > 2e-3 ||
  colorDiff.max > 2e-3 ||
  opacityDiff.max > 2e-3 ||
  quaternionMagnitude > 2e-3
) {
  throw new Error("GATE FAIL: anisotropic raster does not reproduce the isotropic raster");
}

aniso.clearDensityStats();
await device.queue.onSubmittedWorkDone();
{
  const enc = device.createCommandEncoder();
  aniso.recordClearRawGrad(enc);
  aniso.recordForward(enc, 0);
  aniso.recordBackwardAdd(enc, 0, true);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const densityStats = await aniso.readDensityStats();
const densityGrad = await aniso.readRawGrad();
const densityParity = diffStats(anisoGrad, densityGrad);
const totalVisiblePixels = densityStats.visiblePixels.reduce((sum, value) => sum + value, 0);
const maxAbsScreenGradient = Math.max(...densityStats.absScreenGradient);
console.log(
  `density stats: visible=${totalVisiblePixels} maxAbsScreenGrad=${maxAbsScreenGradient.toExponential(3)} ` +
    `rawGradMaxDiff=${densityParity.max.toExponential(3)}`
);
if (totalVisiblePixels === 0 || maxAbsScreenGradient === 0 || densityParity.max > 1e-7) {
  throw new Error("GATE FAIL: density-control statistics are missing or alter the real gradient");
}

function makeAnisotropicParams(): Float32Array {
  const params = liftIsotropicParams(isoParams);
  params[3 * G + 0] = Math.log(0.16);
  params[3 * G + 1] = Math.log(0.065);
  params[3 * G + 2] = Math.log(0.09);
  params[6 * G + 0] = 0.13;
  params[6 * G + 1] = -0.24;
  params[6 * G + 2] = 0.31;
  params[6 * G + 3] = 0.91;
  return params;
}

async function renderLoss(params: Float32Array): Promise<number> {
  aniso.setParams(params);
  aniso.runForward(0);
  await device.queue.onSubmittedWorkDone();
  const image = await aniso.readImage();
  let loss = 0;
  for (let i = 0; i < image.length; i++) loss += image[i] * imageGradient[i];
  return loss;
}

const genuineParams = makeAnisotropicParams();
aniso.setParams(genuineParams);
{
  const enc = device.createCommandEncoder();
  aniso.recordForward(enc, 0);
  aniso.recordClearRawGrad(enc);
  aniso.recordBackwardAdd(enc, 0);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const analytic = await aniso.readRawGrad();
const finiteDifferenceIndices = [0, 3 * G, 6 * G + 1, 10 * G, 13 * G];
const names = ["position.x", "logScale.x", "quaternion.y", "color.r", "opacity"];
let worstRelative = 0;
for (let sample = 0; sample < finiteDifferenceIndices.length; sample++) {
  const index = finiteDifferenceIndices[sample];
  const epsilon = index < 3 * G ? 2e-4 : index >= 6 * G && index < 10 * G ? 2e-3 : 1e-3;
  const plus = genuineParams.slice(); plus[index] += epsilon;
  const minus = genuineParams.slice(); minus[index] -= epsilon;
  const numeric = ((await renderLoss(plus)) - (await renderLoss(minus))) / (2 * epsilon);
  const absolute = Math.abs(analytic[index] - numeric);
  const relative = absolute / Math.max(2e-4, Math.abs(analytic[index]), Math.abs(numeric));
  worstRelative = Math.max(worstRelative, relative);
  console.log(
    `finite diff ${names[sample].padEnd(12)} analytic=${analytic[index].toExponential(5)} numeric=${numeric.toExponential(5)} rel=${relative.toExponential(3)}`
  );
}

aniso.setParams(genuineParams);
if (worstRelative > 0.08) {
  throw new Error(`GATE FAIL: anisotropic raster finite difference relative error ${worstRelative}`);
}

const beforeAdam = await aniso.readParams();
{
  const enc = device.createCommandEncoder();
  aniso.recordAdam(enc, 1, { position: 1e-3, logScale: 1e-3, quaternion: 1e-3, color: 1e-3, opacity: 1e-3 });
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const afterAdam = await aniso.readParams();
for (const index of finiteDifferenceIndices) {
  if (analytic[index] !== 0 && beforeAdam[index] === afterAdam[index]) {
    throw new Error(`GATE FAIL: Adam did not update parameter ${index}`);
  }
}

const telemetry = await aniso.readTileTelemetry();
console.log(`tiles: pairs=${telemetry.totalPairs} max=${telemetry.maxCount}/${telemetry.cap} overflow=${telemetry.overflowPairs}`);
if (telemetry.overflowPairs !== 0) throw new Error("GATE FAIL: parity scene overflowed tile storage");

iso.destroy();
aniso.destroy();
console.log("GATE PASS: anisotropic conic raster forward/backward, finite differences, and Adam are valid.");
