/**
 * bun-webgpu correctness gate for the feature32 residual colorizer.
 *
 *   bun tools/splat3d/feature_colorizer_test.ts
 */
import { setupGlobals } from "bun-webgpu";
import {
  FEATURE32_BIAS_FLOATS,
  FEATURE32_CHANNELS,
  FEATURE32_RGB_CHANNELS,
  FEATURE32_WEIGHT_FLOATS,
  Feature32Colorizer,
} from "../../src/splat3d_feature";

setupGlobals();

const U = { MAP_READ: 1, COPY_DST: 8 };
const WIDTH = 4;
const HEIGHT = 2;
const PIXELS = WIDTH * HEIGHT;
const BATCH = 2;
const SCALE = 0.1;
const FORWARD_EPS = 2e-6;
const GRAD_ABS_EPS = 8e-5;
const GRAD_REL_EPS = 1.5e-3;

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: GPUDevice = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const colorizer = await Feature32Colorizer.create(device, {
  width: WIDTH,
  height: HEIGHT,
  batch: BATCH,
  residualScale: SCALE,
  label: "feature32-test",
});
const io = colorizer.createOwnedIO("feature32-test-io");

const features = deterministic(colorizer.shape.featureFloats, 0.61, 0.13);
device.queue.writeBuffer(io.features, 0, features as unknown as BufferSource);

// Gate 1: zero residual must preserve the RGB-logit skip exactly through the
// sigmoid color boundary, independent of channels 3..31.
{
  const encoder = device.createCommandEncoder();
  colorizer.recordForward(encoder, io.state);
  device.queue.submit([encoder.finish()]);
  const got = await readFloats(io.rgb, colorizer.shape.rgbFloats);
  const expected = zeroResidualReference(features);
  const maxError = maxAbsDiff(got, expected);
  console.log(`zero-residual RGB parity max abs: ${maxError.toExponential(3)}`);
  if (maxError > FORWARD_EPS) fail(`zero-residual RGB parity exceeded ${FORWARD_EPS}`);
}

// Gate 2: run a nonzero forward/backward and compare every returned gradient
// against central finite differences of an independent JS scalar loss.
const weights = deterministic(FEATURE32_WEIGHT_FLOATS, 0.075, 0.71);
const bias = new Float32Array([0.04, -0.025, 0.015, 0]);
const rgbGrad = deterministic(colorizer.shape.rgbFloats, 0.17, 1.19);
colorizer.setParameters(weights, bias);
device.queue.writeBuffer(io.features, 0, features as unknown as BufferSource);
device.queue.writeBuffer(io.rgbGrad, 0, rgbGrad as unknown as BufferSource);

{
  const encoder = device.createCommandEncoder();
  colorizer.recordForward(encoder, io.state);
  colorizer.recordBackward(encoder, io.state);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

const [rgb, featureGrad, weightGrad, biasGrad] = await Promise.all([
  readFloats(io.rgb, colorizer.shape.rgbFloats),
  readFloats(io.featureGrad, colorizer.shape.featureFloats),
  readFloats(colorizer.weightGrad, FEATURE32_WEIGHT_FLOATS),
  readFloats(colorizer.biasGrad, FEATURE32_BIAS_FLOATS),
]);
const expectedRgb = forwardReference(features, weights, bias);
const nonzeroForwardError = maxAbsDiff(rgb, expectedRgb);
console.log(`nonzero forward parity max abs:    ${nonzeroForwardError.toExponential(3)}`);
if (nonzeroForwardError > FORWARD_EPS) fail(`nonzero forward parity exceeded ${FORWARD_EPS}`);

const featureCheck = finiteDifferenceAll(features, (candidate) => loss(candidate, weights, bias, rgbGrad));
const weightCheck = finiteDifferenceAll(weights, (candidate) => loss(features, candidate, bias, rgbGrad));
const bias3 = bias.subarray(0, FEATURE32_RGB_CHANNELS);
const biasCheck = finiteDifferenceAll(bias3, (candidate) => loss(features, weights, candidate, rgbGrad));

const featureStats = gradientStats(featureGrad, featureCheck);
const weightStats = gradientStats(weightGrad, weightCheck);
const biasStats = gradientStats(biasGrad.subarray(0, FEATURE32_RGB_CHANNELS), biasCheck);
console.log(formatStats("feature grad finite diff", featureStats));
console.log(formatStats("weight grad finite diff", weightStats));
console.log(formatStats("bias grad finite diff", biasStats));
if (Math.abs(biasGrad[3]) > 1e-8) fail(`packed bias padding gradient is ${biasGrad[3]}, expected 0`);
for (const [label, stats] of [
  ["feature", featureStats],
  ["weight", weightStats],
  ["bias", biasStats],
] as const) {
  if (stats.violations > 0) {
    fail(`${label} gradient finite difference exceeded abs=${GRAD_ABS_EPS}, rel=${GRAD_REL_EPS}`);
  }
}

io.destroy();
colorizer.destroy();
console.log("GATE PASS: feature32 colorizer forward and backward are correct.");

function deterministic(length: number, scale: number, phase: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = Math.sin(i * 0.173 + phase) * scale + Math.cos(i * 0.037 - phase) * scale * 0.31;
  }
  return out;
}

async function readFloats(buffer: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: U.MAP_READ | U.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, floats * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(U.MAP_READ);
  const output = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return output;
}

function featureIndex(image: number, channel: number, pixel: number): number {
  return (image * FEATURE32_CHANNELS + channel) * PIXELS + pixel;
}

function rgbIndex(image: number, channel: number, pixel: number): number {
  return (image * FEATURE32_RGB_CHANNELS + channel) * PIXELS + pixel;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function zeroResidualReference(input: Float32Array): Float32Array {
  const output = new Float32Array(BATCH * FEATURE32_RGB_CHANNELS * PIXELS);
  for (let image = 0; image < BATCH; image++) {
    for (let channel = 0; channel < FEATURE32_RGB_CHANNELS; channel++) {
      for (let pixel = 0; pixel < PIXELS; pixel++) {
        output[rgbIndex(image, channel, pixel)] = sigmoid(input[featureIndex(image, channel, pixel)]);
      }
    }
  }
  return output;
}

function forwardReference(
  input: Float32Array,
  matrix: Float32Array,
  offset: ArrayLike<number>
): Float32Array {
  const output = new Float32Array(BATCH * FEATURE32_RGB_CHANNELS * PIXELS);
  for (let image = 0; image < BATCH; image++) {
    for (let outputChannel = 0; outputChannel < FEATURE32_RGB_CHANNELS; outputChannel++) {
      for (let pixel = 0; pixel < PIXELS; pixel++) {
        let residual = offset[outputChannel];
        for (let inputChannel = 0; inputChannel < FEATURE32_CHANNELS; inputChannel++) {
          residual += matrix[outputChannel * FEATURE32_CHANNELS + inputChannel]
            * input[featureIndex(image, inputChannel, pixel)];
        }
        const logit = input[featureIndex(image, outputChannel, pixel)] + SCALE * residual;
        output[rgbIndex(image, outputChannel, pixel)] = sigmoid(logit);
      }
    }
  }
  return output;
}

function loss(
  input: Float32Array,
  matrix: Float32Array,
  offset: ArrayLike<number>,
  upstream: Float32Array
): number {
  const output = forwardReference(input, matrix, offset);
  let value = 0;
  for (let i = 0; i < output.length; i++) value += output[i] * upstream[i];
  return value;
}

function finiteDifferenceAll(values: Float32Array, evaluate: (candidate: Float32Array) => number): Float64Array {
  const result = new Float64Array(values.length);
  const candidate = values.slice();
  const epsilon = 1e-2;
  for (let i = 0; i < candidate.length; i++) {
    const original = candidate[i];
    const plus = Math.fround(original + epsilon);
    const minus = Math.fround(original - epsilon);
    candidate[i] = plus;
    const plusLoss = evaluate(candidate);
    candidate[i] = minus;
    const minusLoss = evaluate(candidate);
    candidate[i] = original;
    result[i] = (plusLoss - minusLoss) / (plus - minus);
  }
  return result;
}

function maxAbsDiff(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < expected.length; i++) max = Math.max(max, Math.abs(actual[i] - expected[i]));
  return max;
}

interface GradientStats {
  maxAbs: number;
  maxRel: number;
  worst: number;
  violations: number;
}

function gradientStats(actual: ArrayLike<number>, expected: ArrayLike<number>): GradientStats {
  let maxAbs = 0;
  let maxRel = 0;
  let worst = 0;
  let violations = 0;
  for (let i = 0; i < expected.length; i++) {
    const abs = Math.abs(actual[i] - expected[i]);
    const rel = abs / Math.max(1e-5, Math.abs(expected[i]));
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = i;
    }
    maxRel = Math.max(maxRel, rel);
    if (abs > GRAD_ABS_EPS && rel > GRAD_REL_EPS) violations++;
  }
  return { maxAbs, maxRel, worst, violations };
}

function formatStats(label: string, stats: GradientStats): string {
  return (
    `${label.padEnd(28)} max abs=${stats.maxAbs.toExponential(3)} ` +
    `max rel=${stats.maxRel.toExponential(3)} worst=${stats.worst} violations=${stats.violations}`
  );
}

function fail(message: string): never {
  io.destroy();
  colorizer.destroy();
  console.error(`GATE FAIL: ${message}`);
  process.exit(1);
}
