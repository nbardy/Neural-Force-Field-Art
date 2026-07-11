/**
 * WebGPU gate for the Dream Fields mean-transmittance and ray-distortion losses.
 *
 *   bun tools/splat3d/transmittance_distortion_test.ts
 */
import { setupGlobals } from "bun-webgpu";
import { DEFAULT_3D_CAMERAS, prepareCamera } from "../../src/splat3d/cameras";
import { Raster3DEngine } from "../../src/splat3d/raster";
import { PARAM_STRIDE_3D } from "../../src/splat3d/raster_wgsl";

setupGlobals();

const U = { MAP_READ: 1, COPY_DST: 8 };
const SIDE = 32;
const G = 256;

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device: GPUDevice = await adapter.requestDevice();
const camera = prepareCamera(DEFAULT_3D_CAMERAS[1], SIDE);
const raster = await Raster3DEngine.create(device, {
  H: SIDE,
  W: SIDE,
  G,
  cap: 256,
  cameras: [camera],
  bg: [0, 0, 0],
  dynamicBg: true,
  dynamicBgTexture: true,
  backgroundTextureMode: "checkerboard",
  dynamicCoverage: true,
  dynamicTransmittance: true,
  dynamicEntropy: true,
  dynamicFootprint: true,
});
raster.setScreenVariance(1.5);

const params = makeParams();
raster.setParams(params);
device.queue.writeBuffer(raster.gradImage, 0, new Float32Array(3 * SIDE * SIDE));

async function gradients(
  transmittanceWeight: number,
  targetTransmittance: number,
  rayDistortionWeight: number,
  rayEntropyWeight = 0
): Promise<Float32Array> {
  raster.setCoverageRegularizer({
    transmittanceWeight,
    targetTransmittance,
    rayDistortionWeight,
    rayEntropyWeight,
    rayEntropyMask: 0.05,
  });
  const enc = device.createCommandEncoder();
  raster.recordBackgroundGenerate(enc, 7, 0);
  raster.recordClearRawGrad(enc);
  raster.recordForward(enc, 0);
  raster.recordBackwardAdd(enc, 0);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  return readFloats(raster.gradRaw, G * PARAM_STRIDE_3D);
}

const inactive = await gradients(1, 0, 0);
const transmittance = await gradients(1, 1, 0);
const distortion = await gradients(0, 0, 1);
const entropy = await gradients(0, 0, 0, 1);
const inactiveMax = maxAbs(inactive);
const transOpacity = transmittance[7 * G] + transmittance[7 * G + 1];
const distortionPosition = maxAbs(distortion.subarray(0, 3 * G));
const entropyOpacity = maxAbs(entropy.subarray(7 * G, 8 * G));
const finite = [inactive, transmittance, distortion, entropy].every((values) => values.every(Number.isFinite));

console.log(`inactive max:       ${inactiveMax.toExponential(3)}`);
console.log(`trans opacity sum:  ${transOpacity.toExponential(3)}`);
console.log(`dist position max:  ${distortionPosition.toExponential(3)}`);
console.log(`entropy opacity max:${entropyOpacity.toExponential(3)}`);

raster.destroy();

if (
  !finite ||
  inactiveMax > 1e-8 ||
  !(transOpacity > 0) ||
  !(distortionPosition > 1e-8) ||
  !(entropyOpacity > 1e-8)
) {
  console.error("GATE FAIL: transmittance/distortion loss is invalid or fails its one-sided gate.");
  process.exit(1);
}
console.log("GATE PASS: one-sided transmittance and ray-distortion gradients are active and finite.");

function makeParams(): Float32Array {
  const out = new Float32Array(G * PARAM_STRIDE_3D);
  for (let g = 0; g < G; g++) {
    out[g * 3 + 0] = g < 2 ? 0 : 100;
    out[g * 3 + 1] = 0;
    out[g * 3 + 2] = g === 0 ? -0.18 : g === 1 ? 0.18 : 100;
    out[3 * G + g] = Math.log(g < 2 ? 0.12 : 0.01);
    out[4 * G + g * 3 + 0] = 0;
    out[4 * G + g * 3 + 1] = 0;
    out[4 * G + g * 3 + 2] = 0;
    out[7 * G + g] = g < 2 ? 1.5 : -12;
  }
  return out;
}

function maxAbs(values: Float32Array): number {
  let max = 0;
  for (const value of values) max = Math.max(max, Math.abs(value));
  return max;
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
