/**
 * Metal regression gate for anisotropic cloud regularizers.
 *
 *   bun tools/splat3d/aniso_regularizer_test.ts
 */
import { setupGlobals } from "bun-webgpu";
import { DEFAULT_3D_CAMERAS, prepareCamera } from "../../src/splat3d/cameras";
import { ANISO_PARAM_STRIDE_3D } from "../../src/splat3d_aniso/layout";
import {
  AnisotropicRaster3DEngine,
  type AnisotropicRaster3DRegularizerOptions,
} from "../../src/splat3d_aniso/raster_engine";
import { ANISO_CENTER_SUM_SCALE_3D } from "../../src/splat3d_aniso/raster_wgsl";

setupGlobals();

const U = { MAP_READ: 1, COPY_DST: 8 };
const SIDE = 32;
const G = 6;
const EPS = 3e-6;

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("aniso_regularizer_test: no WebGPU adapter");
const device = await adapter.requestDevice();
const raster = await AnisotropicRaster3DEngine.create(device, {
  H: SIDE,
  W: SIDE,
  G,
  cap: 256,
  cameras: [prepareCamera(DEFAULT_3D_CAMERAS[0], SIDE)],
  bg: [0, 0, 0],
});

const params = makeParams();
const opts: AnisotropicRaster3DRegularizerOptions = {
  centerWeight: 0.07,
  radiusWeight: 0.11,
  targetRadius: 0.9,
  opacitySparsity: 0.013,
  smallRadiusWeight: 0.19,
  smallRadius: 0.065,
  radiusBandWeight: 0.17,
  minRadius: 0.05,
  maxRadius: 0.15,
};
raster.setParams(params);

{
  const enc = device.createCommandEncoder();
  raster.recordClearRawGrad(enc);
  raster.recordRegularizerAdd(enc, opts);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}

const actual = await raster.readRawGrad();
const expected = expectedGradient(params, opts);
let maxError = 0;
let maxAxisDelta = 0;
let maxUntouched = 0;
for (let i = 0; i < actual.length; i++) maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
for (let g = 0; g < G; g++) {
  const base = 3 * G + 3 * g;
  maxAxisDelta = Math.max(
    maxAxisDelta,
    Math.abs(actual[base] - actual[base + 1]),
    Math.abs(actual[base] - actual[base + 2])
  );
  for (let i = 6 * G + 4 * g; i < 6 * G + 4 * g + 4; i++) {
    maxUntouched = Math.max(maxUntouched, Math.abs(actual[i]));
  }
  for (let i = 10 * G + 3 * g; i < 10 * G + 3 * g + 3; i++) {
    maxUntouched = Math.max(maxUntouched, Math.abs(actual[i]));
  }
}

console.log(`regularizer max error:       ${maxError.toExponential(3)}`);
console.log(`log-scale axis delta:        ${maxAxisDelta.toExponential(3)}`);
console.log(`quaternion/color max grad:   ${maxUntouched.toExponential(3)}`);
if (maxError > EPS || maxAxisDelta > EPS || maxUntouched > EPS) {
  raster.destroy();
  throw new Error("GATE FAIL: anisotropic regularizer gradients do not match the CPU reference");
}

// A selective reset must clear all five SoA Adam groups for one splat only.
{
  const enc = device.createCommandEncoder();
  raster.recordAdam(enc, 1, {
    position: 1e-3,
    logScale: 1e-3,
    quaternion: 1e-3,
    color: 1e-3,
    opacity: 1e-3,
  });
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
raster.resetAdamForSplats([1, -1, G]);
await device.queue.onSubmittedWorkDone();
let resetMagnitude = 0;
let retainedMagnitude = 0;
for (const moment of await Promise.all([
  readFloats(raster.mBuf, G * ANISO_PARAM_STRIDE_3D),
  readFloats(raster.vBuf, G * ANISO_PARAM_STRIDE_3D),
])) {
  for (const segment of [
    { offset: 0, components: 3 },
    { offset: 3 * G, components: 3 },
    { offset: 6 * G, components: 4 },
    { offset: 10 * G, components: 3 },
    { offset: 13 * G, components: 1 },
  ]) {
    for (let c = 0; c < segment.components; c++) {
      resetMagnitude = Math.max(resetMagnitude, Math.abs(moment[segment.offset + segment.components + c]));
      retainedMagnitude = Math.max(retainedMagnitude, Math.abs(moment[segment.offset + c]));
    }
  }
}
console.log(`reset splat Adam magnitude:  ${resetMagnitude.toExponential(3)}`);
console.log(`retained Adam magnitude:     ${retainedMagnitude.toExponential(3)}`);
raster.destroy();

if (resetMagnitude !== 0 || retainedMagnitude === 0) {
  throw new Error("GATE FAIL: selective anisotropic Adam reset is invalid");
}
console.log("GATE PASS: anisotropic centroid, bounds, opacity, geometric-radius, and Adam reset are valid.");

function makeParams(): Float32Array {
  const out = new Float32Array(G * ANISO_PARAM_STRIDE_3D);
  const radii = [0.035, 0.052, 0.08, 0.14, 0.18, 0.24];
  const opacityRaw = [-1.2, -0.4, 0, 0.5, 1.1, 1.8];
  for (let g = 0; g < G; g++) {
    out[3 * g + 0] = 0.22 + 0.31 * g;
    out[3 * g + 1] = -0.18 + 0.09 * g;
    out[3 * g + 2] = 0.11 - 0.06 * g;
    const logRadius = Math.log(radii[g]);
    out[3 * G + 3 * g + 0] = logRadius - 0.37;
    out[3 * G + 3 * g + 1] = logRadius + 0.08;
    out[3 * G + 3 * g + 2] = logRadius + 0.29;
    out[6 * G + 4 * g + 3] = 1;
    out[13 * G + g] = opacityRaw[g];
  }
  return out;
}

function expectedGradient(
  values: Float32Array,
  regularizer: AnisotropicRaster3DRegularizerOptions
): Float32Array {
  const out = new Float32Array(values.length);
  const centerNumerator = [0, 0, 0];
  let centerMass = 0;
  for (let g = 0; g < G; g++) {
    const opacity = sigmoid(values[13 * G + g]);
    for (let axis = 0; axis < 3; axis++) {
      centerNumerator[axis] += Math.round(values[3 * g + axis] * opacity * ANISO_CENTER_SUM_SCALE_3D);
    }
    centerMass += Math.round(opacity * ANISO_CENTER_SUM_SCALE_3D);
  }
  const center = centerNumerator.map((value) => value / Math.max(centerMass, 1));

  for (let g = 0; g < G; g++) {
    const px = values[3 * g];
    const py = values[3 * g + 1];
    const pz = values[3 * g + 2];
    const positionRadius = Math.hypot(px, py, pz);
    const outside = Math.max(0, positionRadius - Math.max(regularizer.targetRadius, 1e-8));
    const radialScale = 2 * regularizer.radiusWeight * outside / Math.max(positionRadius, 1e-8);
    out[3 * g] = 2 * regularizer.centerWeight * center[0] + radialScale * px;
    out[3 * g + 1] = 2 * regularizer.centerWeight * center[1] + radialScale * py;
    out[3 * g + 2] = 2 * regularizer.centerWeight * center[2] + radialScale * pz;

    const opacity = sigmoid(values[13 * G + g]);
    const scaleBase = 3 * G + 3 * g;
    const radius = Math.exp((values[scaleBase] + values[scaleBase + 1] + values[scaleBase + 2]) / 3);
    const small = Math.max(0, Math.max(regularizer.smallRadius, 1e-8) - radius);
    const under = Math.max(0, Math.max(regularizer.minRadius, 1e-8) - radius);
    const maxRadius = Math.max(regularizer.maxRadius, Math.max(regularizer.minRadius, 1e-8) + 1e-8);
    const over = Math.max(0, radius - maxRadius);
    const radiusDerivative = -2 * regularizer.smallRadiusWeight * opacity * opacity * small
      + regularizer.radiusBandWeight * (-2 * under + 2 * over);
    const perAxis = radiusDerivative * radius / 3;
    out[scaleBase] = perAxis;
    out[scaleBase + 1] = perAxis;
    out[scaleBase + 2] = perAxis;
    out[13 * G + g] = regularizer.opacitySparsity * opacity * (1 - opacity)
      + 2 * regularizer.smallRadiusWeight * small * small * opacity * opacity * (1 - opacity);
  }
  return out;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

async function readFloats(buffer: GPUBuffer, words: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: words * 4, usage: U.MAP_READ | U.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, staging, 0, words * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1);
  const values = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return values;
}
