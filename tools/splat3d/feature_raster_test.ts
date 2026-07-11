/**
 * bun-webgpu correctness gate for the feature32 reference splat raster.
 *
 *   bun tools/splat3d/feature_raster_test.ts
 */
import { setupGlobals } from "bun-webgpu";
import { Feature32ReferenceRaster } from "../../src/splat3d_feature/feature_raster";

setupGlobals();

const U = { MAP_READ: 1, COPY_DST: 8 };
const WIDTH = 4;
const HEIGHT = 2;
const PIXELS = WIDTH * HEIGHT;
const SPLATS = 3;
const CHANNELS = 32;
const MIN_RADIUS = 0.25;
const MAX_OPACITY = 0.99;
const FORWARD_EPS = 3e-6;
const GRAD_ABS_EPS = 2.5e-4;
const GRAD_REL_EPS = 3e-3;

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: GPUDevice = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const raster = await Feature32ReferenceRaster.create(device, {
  width: WIDTH,
  height: HEIGHT,
  splats: SPLATS,
  minRadius: MIN_RADIUS,
  maxOpacity: MAX_OPACITY,
  label: "feature-raster-test",
});
const io = raster.createOwnedIO();

const geometry = new Float32Array([
  1.15, 1.25, Math.log(0.82), -0.35,
  2.75, 0.85, Math.log(1.18), 0.42,
  2.10, 1.80, Math.log(0.63), -0.08,
]);
const splatFeatures = deterministic(SPLATS * CHANNELS, 0.42, 0.31);
const background = deterministic(CHANNELS, 0.11, 0.93);
const upstream = deterministic(CHANNELS * PIXELS, 0.19, 1.27);
const order = new Uint32Array([1, 0, 2]);

device.queue.writeBuffer(io.geometry, 0, geometry as unknown as BufferSource);
device.queue.writeBuffer(io.splatFeatures, 0, splatFeatures as unknown as BufferSource);
device.queue.writeBuffer(io.sortedIds, 0, order as unknown as BufferSource);
device.queue.writeBuffer(io.background, 0, background as unknown as BufferSource);
device.queue.writeBuffer(io.imageFeatureGrad, 0, upstream as unknown as BufferSource);

{
  const encoder = device.createCommandEncoder();
  raster.recordForward(encoder, io.state);
  raster.recordBackward(encoder, io.state);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

const [image, geometryGrad, splatFeatureGrad] = await Promise.all([
  readFloats(io.imageFeatures, CHANNELS * PIXELS),
  readFloats(io.geometryGrad, geometry.length),
  readFloats(io.splatFeatureGrad, splatFeatures.length),
]);
const expectedImage = forwardReference(geometry, splatFeatures, order, background);
const forwardError = maxAbsDiff(image, expectedImage);
console.log(`forward parity max abs:             ${forwardError.toExponential(3)}`);
if (forwardError > FORWARD_EPS) fail(`forward parity exceeded ${FORWARD_EPS}`);

const geometryCheck = finiteDifferenceAll(
  geometry,
  (candidate) => loss(candidate, splatFeatures, order, background, upstream),
  2e-3
);
const featureCheck = finiteDifferenceAll(
  splatFeatures,
  (candidate) => loss(geometry, candidate, order, background, upstream),
  2e-3
);
const geometryStats = gradientStats(geometryGrad, geometryCheck);
const featureStats = gradientStats(splatFeatureGrad, featureCheck);
console.log(formatStats("geometry grad finite diff", geometryStats));
console.log(formatStats("feature grad finite diff", featureStats));
if (geometryStats.violations > 0) {
  fail(`geometry gradient finite difference exceeded abs=${GRAD_ABS_EPS}, rel=${GRAD_REL_EPS}`);
}
if (featureStats.violations > 0) {
  fail(`feature gradient finite difference exceeded abs=${GRAD_ABS_EPS}, rel=${GRAD_REL_EPS}`);
}

// The image layout is the colorizer contract: channel-planar [32][H*W].
for (let channel = 0; channel < CHANNELS; channel++) {
  const first = image[channel * PIXELS];
  if (!Number.isFinite(first)) fail(`channel ${channel} does not begin at a finite planar offset`);
}

io.destroy();
console.log("GATE PASS: feature32 raster compositing and all input gradients are correct.");

function deterministic(length: number, scale: number, phase: number): Float32Array {
  const output = new Float32Array(length);
  for (let index = 0; index < length; index++) {
    output[index] =
      Math.sin(index * 0.173 + phase) * scale + Math.cos(index * 0.047 - phase) * scale * 0.29;
  }
  return output;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function imageIndex(channel: number, pixel: number): number {
  return channel * PIXELS + pixel;
}

function featureIndex(splat: number, channel: number): number {
  return splat * CHANNELS + channel;
}

function alphaAt(packed: ArrayLike<number>, splat: number, x: number, y: number): number {
  const base = splat * 4;
  const dx = x + 0.5 - packed[base];
  const dy = y + 0.5 - packed[base + 1];
  const radius = Math.max(Math.exp(packed[base + 2]), MIN_RADIUS);
  const gaussian = Math.exp(-0.5 * (dx * dx + dy * dy) / (radius * radius));
  return MAX_OPACITY * sigmoid(packed[base + 3]) * gaussian;
}

function forwardReference(
  packedGeometry: Float32Array,
  features: Float32Array,
  sortedIds: Uint32Array,
  bg: Float32Array
): Float64Array {
  const output = new Float64Array(CHANNELS * PIXELS);
  for (let pixel = 0; pixel < PIXELS; pixel++) {
    const x = pixel % WIDTH;
    const y = Math.floor(pixel / WIDTH);
    let transmittance = 1;
    for (const splat of sortedIds) {
      const alpha = alphaAt(packedGeometry, splat, x, y);
      const weight = transmittance * alpha;
      for (let channel = 0; channel < CHANNELS; channel++) {
        output[imageIndex(channel, pixel)] += weight * features[featureIndex(splat, channel)];
      }
      transmittance *= 1 - alpha;
    }
    for (let channel = 0; channel < CHANNELS; channel++) {
      output[imageIndex(channel, pixel)] += transmittance * bg[channel];
    }
  }
  return output;
}

function loss(
  packedGeometry: Float32Array,
  features: Float32Array,
  sortedIds: Uint32Array,
  bg: Float32Array,
  gradient: Float32Array
): number {
  const output = forwardReference(packedGeometry, features, sortedIds, bg);
  let value = 0;
  for (let index = 0; index < output.length; index++) value += output[index] * gradient[index];
  return value;
}

function finiteDifferenceAll(
  values: Float32Array,
  evaluate: (candidate: Float32Array) => number,
  epsilon: number
): Float64Array {
  const result = new Float64Array(values.length);
  const candidate = values.slice();
  for (let index = 0; index < candidate.length; index++) {
    const original = candidate[index];
    const plus = Math.fround(original + epsilon);
    const minus = Math.fround(original - epsilon);
    candidate[index] = plus;
    const plusLoss = evaluate(candidate);
    candidate[index] = minus;
    const minusLoss = evaluate(candidate);
    candidate[index] = original;
    result[index] = (plusLoss - minusLoss) / (plus - minus);
  }
  return result;
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

function maxAbsDiff(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index < expected.length; index++) {
    maximum = Math.max(maximum, Math.abs(actual[index] - expected[index]));
  }
  return maximum;
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
  for (let index = 0; index < expected.length; index++) {
    const absolute = Math.abs(actual[index] - expected[index]);
    const relative = absolute / Math.max(1e-5, Math.abs(expected[index]));
    if (absolute > maxAbs) {
      maxAbs = absolute;
      worst = index;
    }
    maxRel = Math.max(maxRel, relative);
    if (absolute > GRAD_ABS_EPS && relative > GRAD_REL_EPS) violations++;
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
  console.error(`GATE FAIL: ${message}`);
  process.exit(1);
}
