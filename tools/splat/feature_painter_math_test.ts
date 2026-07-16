/**
 * Real-Metal contract checks for the compact fused Feature8 painter.
 *
 * 1. A zero residual decoder must reproduce ordinary RGB splatting exactly.
 * 2. The local-coordinate route must agree with a finite-difference theta
 *    derivative when it is made active by a nonzero latent decoder weight.
 */
import { setupGlobals } from "bun-webgpu";
import { FeaturePainterEngine } from "../../src/splat/feature_painter";
import { DECODER_PARAM_COUNT, FEATURE_STRIDE } from "../../src/splat/feature_painter_wgsl";
import { randomDecoder, randomFeatures } from "../../src/splat/feature_optimize";
import { RasterEngine } from "../../src/splat/raster";
import { PARAM_STRIDE, type RasterConfig } from "../../src/splat/raster_wgsl";

setupGlobals();

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();

function sceneParams(G: number, side: number): Float32Array {
  const p = new Float32Array(G * PARAM_STRIDE);
  const logScale = 2 * G;
  const theta = 4 * G;
  const color = 5 * G;
  const opacity = 8 * G;
  for (let g = 0; g < G; g++) {
    p[g * 2] = 4 + ((g * 19) % (side - 8));
    p[g * 2 + 1] = 4 + ((g * 11) % (side - 8));
    p[logScale + g * 2] = Math.log(2.0 + (g % 3));
    p[logScale + g * 2 + 1] = Math.log(1.5 + ((g + 1) % 4));
    p[theta + g] = g * 0.37;
    p[color + g * 3] = -0.5 + (g % 5) * 0.25;
    p[color + g * 3 + 1] = 0.3 - (g % 4) * 0.2;
    p[color + g * 3 + 2] = -0.2 + (g % 6) * 0.17;
    p[opacity + g] = 0.2 + (g % 4) * 0.15;
  }
  return p;
}

function maxAbs(a: Float32Array, b: Float32Array): number {
  let out = 0;
  for (let i = 0; i < a.length; i++) out = Math.max(out, Math.abs(a[i] - b[i]));
  return out;
}

function dot(a: Float32Array, b: Float32Array): number {
  let out = 0;
  for (let i = 0; i < a.length; i++) out += a[i] * b[i];
  return out;
}

const parityCfg: RasterConfig = { H: 64, W: 64, G: 19, cap: 128, bg: [0.2, 0.35, 0.5] };
const rgb = await RasterEngine.create(device, parityCfg);
const feature = await FeaturePainterEngine.create(device, parityCfg);
const parityParams = sceneParams(parityCfg.G, parityCfg.W);
rgb.setParams(parityParams);
feature.setParams(parityParams);
feature.setFeatureParams(randomFeatures(parityCfg.G, 19));
feature.setDecoderParams(randomDecoder(19));
rgb.runForward();
feature.runForward();
await device.queue.onSubmittedWorkDone();
const rgbImage = await rgb.readImage();
const featureImage = await feature.readImage();
const parityError = maxAbs(rgbImage, featureImage);
console.log(`feature8 production-init RGB parity max abs: ${parityError.toExponential(3)}`);
if (parityError > 2e-6) throw new Error(`Feature8 zero-residual parity drift ${parityError}`);
const parityUpstream = new Float32Array(3 * parityCfg.H * parityCfg.W);
for (let i = 0; i < parityUpstream.length; i++) parityUpstream[i] = Math.sin(i * 0.017) * 0.07;
rgb.setGradImage(parityUpstream);
feature.setGradImage(parityUpstream);
const parityBackward = device.createCommandEncoder();
rgb.recordForward(parityBackward); rgb.recordBackward(parityBackward);
feature.recordForward(parityBackward); feature.recordBackward(parityBackward);
device.queue.submit([parityBackward.finish()]);
await device.queue.onSubmittedWorkDone();
const rgbGradient = await rgb.readGradRaw();
const featureGradient = await feature.readGeometryGradient();
const parityGradientError = maxAbs(rgbGradient, featureGradient);
console.log(`feature8 RGB gradient parity max abs: ${parityGradientError.toExponential(3)}`);
if (parityGradientError > 3e-5) throw new Error(`Feature8 zero-residual gradient drift ${parityGradientError}`);
rgb.destroy();
feature.destroy();

const gradCfg: RasterConfig = { H: 16, W: 16, G: 1, cap: 16, bg: [0.2, 0.3, 0.4] };
const local = await FeaturePainterEngine.create(device, gradCfg);
const params = new Float32Array(PARAM_STRIDE);
params[0] = 8.1; params[1] = 7.6;
params[2] = Math.log(3.6); params[3] = Math.log(1.8);
params[4] = 0.43;
params[5] = 0.2; params[6] = -0.3; params[7] = 0.5;
params[8] = 0.7;
const extras = new Float32Array(FEATURE_STRIDE);
extras[0] = 0.13;
extras[5] = 0.31; // local-x for latent 0
extras[10] = -0.22; // local-y for latent 0
const decoder = new Float32Array(DECODER_PARAM_COUNT);
decoder[3] = 0.8;
decoder[11] = -0.25;
decoder[19] = 0.35;
const upstream = new Float32Array(3 * gradCfg.H * gradCfg.W);
for (let y = 0; y < gradCfg.H; y++) {
  for (let x = 0; x < gradCfg.W; x++) {
    const i = y * gradCfg.W + x;
    upstream[i] = 0.11 + 0.013 * x - 0.008 * y;
    upstream[gradCfg.H * gradCfg.W + i] = -0.06 + 0.007 * y;
    upstream[2 * gradCfg.H * gradCfg.W + i] = 0.04 - 0.005 * x;
  }
}
local.setParams(params);
local.setFeatureParams(extras);
local.setDecoderParams(decoder);
local.setGradImage(upstream);

async function analyticGradient(index: number): Promise<number> {
  const backward = device.createCommandEncoder();
  local.recordForward(backward);
  local.recordBackward(backward);
  device.queue.submit([backward.finish()]);
  await device.queue.onSubmittedWorkDone();
  return (await local.readGeometryGradient())[index];
}

async function lossAt(index: number, value: number): Promise<number> {
  const trial = params.slice();
  trial[index] = value;
  local.setParams(trial);
  local.runForward();
  const image = await local.readImage();
  return dot(image, upstream);
}

const epsilon = 1e-4;
async function numericGradient(index: number): Promise<number> {
  return ((await lossAt(index, params[index] + epsilon)) - (await lossAt(index, params[index] - epsilon))) / (2 * epsilon);
}

const savedAx = extras[5]; const savedAy = extras[10];
extras[5] = 0; extras[10] = 0;
local.setFeatureParams(extras);
const noLocalAnalytic = await analyticGradient(4);
const noLocalNumeric = await numericGradient(4);
console.log(`feature8 alpha-only theta: analytic=${noLocalAnalytic.toExponential(5)} numeric=${noLocalNumeric.toExponential(5)}`);
const alphaAnalytic = await analyticGradient(8);
const alphaNumeric = await numericGradient(8);
console.log(`feature8 alpha-only opacity: analytic=${alphaAnalytic.toExponential(5)} numeric=${alphaNumeric.toExponential(5)}`);
extras[5] = savedAx; extras[10] = savedAy;
local.setFeatureParams(extras);
const analytic = await analyticGradient(4);
const numeric = await numericGradient(4);
const relative = Math.abs(analytic - numeric) / Math.max(1e-4, Math.abs(analytic), Math.abs(numeric));
console.log(`feature8 local theta grad: analytic=${analytic.toExponential(5)} numeric=${numeric.toExponential(5)} rel=${relative.toFixed(4)}`);
if (!Number.isFinite(analytic) || !Number.isFinite(numeric) || relative > 0.08) {
  throw new Error(`Feature8 local-coordinate gradient check failed (relative ${relative})`);
}
local.destroy();
console.log("feature8 math gate: PASS");
