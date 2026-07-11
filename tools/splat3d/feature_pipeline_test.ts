/** End-to-end gate for feature raster -> colorizer -> backward. */
import { setupGlobals } from "bun-webgpu";
import { Feature32Colorizer, Feature32ReferenceRaster } from "../../src/splat3d_feature";

setupGlobals();

const U = { MAP_READ: 1, COPY_DST: 8 };
const W = 8;
const H = 8;
const PIXELS = W * H;
const SPLATS = 3;
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("feature pipeline: no WebGPU adapter");
const device: GPUDevice = await adapter.requestDevice();

const raster = await Feature32ReferenceRaster.create(device, { width: W, height: H, splats: SPLATS });
const rasterIO = raster.createOwnedIO();
const colorizer = await Feature32Colorizer.create(device, { width: W, height: H });
const colorIO = colorizer.createOwnedIO();
const linkedColorIO = colorizer.createIOState({
  features: rasterIO.imageFeatures,
  rgb: colorIO.rgb,
  rgbGrad: colorIO.rgbGrad,
  featureGrad: rasterIO.imageFeatureGrad,
});

const geometry = new Float32Array([
  2.5, 3.5, Math.log(1.8), 1.2,
  5.5, 3.0, Math.log(1.5), 0.8,
  4.0, 6.0, Math.log(1.2), 0.4,
]);
const features = new Float32Array(SPLATS * 32);
for (let splat = 0; splat < SPLATS; splat++) {
  for (let channel = 0; channel < 32; channel++) {
    features[splat * 32 + channel] = Math.sin((splat + 1) * (channel + 2) * 0.13) * 0.7;
  }
}
const rgbGrad = new Float32Array(3 * PIXELS);
for (let i = 0; i < rgbGrad.length; i++) rgbGrad[i] = Math.cos(i * 0.11) * 0.02;
device.queue.writeBuffer(rasterIO.geometry, 0, geometry);
device.queue.writeBuffer(rasterIO.splatFeatures, 0, features);
device.queue.writeBuffer(rasterIO.background, 0, new Float32Array(32));
device.queue.writeBuffer(colorIO.rgbGrad, 0, rgbGrad);
raster.setIdentityOrder(rasterIO);

const enc = device.createCommandEncoder();
raster.recordForward(enc, rasterIO.state);
colorizer.recordForward(enc, linkedColorIO);
colorizer.recordBackward(enc, linkedColorIO);
raster.recordBackward(enc, rasterIO.state);
device.queue.submit([enc.finish()]);
await device.queue.onSubmittedWorkDone();

const rgb = await readFloats(colorIO.rgb, 3 * PIXELS);
const geometryGrad = await readFloats(rasterIO.geometryGrad, 4 * SPLATS);
const featureGrad = await readFloats(rasterIO.splatFeatureGrad, 32 * SPLATS);
const weightGrad = await readFloats(colorizer.weightGrad, 96);
const stats = {
  rgb: maxAbs(rgb),
  geometryGrad: maxAbs(geometryGrad),
  featureGrad: maxAbs(featureGrad),
  weightGrad: maxAbs(weightGrad),
};
console.log(
  `rgb=${stats.rgb.toExponential(3)} geometryGrad=${stats.geometryGrad.toExponential(3)} ` +
    `featureGrad=${stats.featureGrad.toExponential(3)} weightGrad=${stats.weightGrad.toExponential(3)}`
);

colorIO.destroy();
rasterIO.destroy();
colorizer.destroy();

if (!rgb.every(Number.isFinite) || !geometryGrad.every(Number.isFinite) || !featureGrad.every(Number.isFinite)) {
  throw new Error("GATE FAIL: feature pipeline produced non-finite values");
}
if (stats.rgb <= 0 || stats.geometryGrad <= 1e-8 || stats.featureGrad <= 1e-8 || stats.weightGrad <= 1e-8) {
  throw new Error("GATE FAIL: feature pipeline did not propagate a required gradient");
}
console.log("GATE PASS: feature raster, colorizer, and complete backward chain are connected.");

function maxAbs(values: Float32Array): number {
  let max = 0;
  for (const value of values) max = Math.max(max, Math.abs(value));
  return max;
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
