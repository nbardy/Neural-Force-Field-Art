/**
 * Standalone anisotropic 3D Gaussian projection gate.
 *
 *   bun tools/splat3d/aniso_projection_test.ts
 *
 * Covers SoA layout, legacy isotropic parity, float64 finite differences for
 * position/log-scale/quaternion, WGSL compilation, and WGSL-vs-CPU values.
 */
import { setupGlobals } from "bun-webgpu";
import {
  ANISO_PARAM_STRIDE_3D,
  anisotropicParamSegments3D,
  anisotropicProjectionWGSL,
  backwardAnisotropicProjection,
  packAnisotropicSplats3D,
  projectAnisotropicGaussian,
  unpackAnisotropicSplat3D,
  type AnisotropicProjectionCamera,
  type AnisotropicProjectionParams,
  type AnisotropicProjectionSettings,
  type AnisotropicProjectionUpstream,
  type AnisotropicSplat3D,
  type CovarianceProjectionMode,
  type Vec3,
} from "../../src/splat3d_aniso";

let failures = 0;
function check(ok: boolean, message: string): void {
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${message}`);
}

const dot3 = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub3 = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize3 = (value: Vec3): Vec3 => {
  const norm = Math.hypot(...value);
  return [value[0] / norm, value[1] / norm, value[2] / norm];
};

function makeCamera(eye: Vec3, target: Vec3, focalPx = 231.25): AnisotropicProjectionCamera {
  const forward = normalize3(sub3(target, eye));
  const right = normalize3(cross3(forward, [0, 1, 0]));
  const up = normalize3(cross3(right, forward));
  return { eye, right, up, forward, focalPx, centerPx: [128, 128], near: 0.2 };
}

function close(actual: number, expected: number, atol = 1e-10, rtol = 1e-10): boolean {
  return Math.abs(actual - expected) <= atol + rtol * Math.max(Math.abs(actual), Math.abs(expected));
}

// Parameter layout remains SoA, matching the current optimizer's segment style.
{
  const splats: AnisotropicSplat3D[] = [
    {
      position: [1, 2, 3],
      logScale: [4, 5, 6],
      quaternion: [7, 8, 9, 10],
      color: [11, 12, 13],
      opacity: 14,
    },
    {
      position: [-1, -2, -3],
      logScale: [-4, -5, -6],
      quaternion: [-7, -8, -9, -10],
      color: [-11, -12, -13],
      opacity: -14,
    },
  ];
  const packed = packAnisotropicSplats3D(splats);
  const segments = anisotropicParamSegments3D(splats.length);
  const roundTrip = unpackAnisotropicSplat3D(packed, splats.length, 1);
  check(packed.length === ANISO_PARAM_STRIDE_3D * splats.length, "14-float anisotropic parameter stride");
  check(
    segments.map((segment) => `${segment.name}:${segment.offset}:${segment.length}`).join("|") ===
      "position:0:6|logScale:6:6|quaternion:12:8|color:20:6|opacity:26:2",
    "SoA segment order is position3 + logScale3 + quaternion4 + color3 + opacity1"
  );
  check(
    JSON.stringify(roundTrip) === JSON.stringify(splats[1]),
    "parameter pack/unpack round trip"
  );
}

// Equal scales under the compatibility projection recover the current scalar
// radius/conic exactly, including for an off-axis center.
{
  const camera: AnisotropicProjectionCamera = {
    eye: [0, 0, 3],
    right: [1, 0, 0],
    up: [0, 1, 0],
    forward: [0, 0, -1],
    focalPx: 274.51,
    centerPx: [128, 128],
    near: 0.2,
  };
  const radius = 0.075;
  const params: AnisotropicProjectionParams = {
    position: [0.43, -0.27, 0.18],
    logScale: [Math.log(radius), Math.log(radius), Math.log(radius)],
    quaternion: [0, 0, 0, 7],
  };
  const projected = projectAnisotropicGaussian(params, camera, { mode: "legacy-affine" });
  const relative = sub3(params.position, camera.eye);
  const vx = dot3(relative, camera.right);
  const vy = dot3(relative, camera.up);
  const vz = Math.max(dot3(relative, camera.forward), camera.near);
  const radiusPx = Math.max(camera.focalPx * radius / vz, 0.25);
  const expectedInvR2 = 1 / (radiusPx * radiusPx);
  check(
    close(projected.meanPx[0], 128 + camera.focalPx * vx / vz) &&
      close(projected.meanPx[1], 128 - camera.focalPx * vy / vz),
    "isotropic mean matches current projection"
  );
  check(
    close(projected.conic[0], expectedInvR2) &&
      close(projected.conic[1], 0, 1e-12, 0) &&
      close(projected.conic[2], expectedInvR2),
    "equal scales + identity rotation match current projected radius/conic off axis above the 0.25px floor"
  );
  check(
    close(Math.hypot(...projected.normalizedQuaternion), 1),
    "raw quaternion is normalized before covariance construction"
  );
}

function makeRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function loss(
  params: AnisotropicProjectionParams,
  camera: AnisotropicProjectionCamera,
  settings: Partial<AnisotropicProjectionSettings>,
  upstream: AnisotropicProjectionUpstream
): number {
  const projected = projectAnisotropicGaussian(params, camera, settings);
  return (
    dot3([projected.meanPx[0], projected.meanPx[1], 0], [upstream.meanPx[0], upstream.meanPx[1], 0]) +
    dot3(projected.conic, upstream.conic)
  );
}

function finiteDifferenceGate(mode: CovarianceProjectionMode): { maxScaled: number; maxRelative: number } {
  const rng = makeRng(mode === "legacy-affine" ? 0xa11507 : 0x3d65a5);
  const camera = makeCamera([1.35, -0.62, 3.1], [-0.08, 0.12, -0.04]);
  const settings: Partial<AnisotropicProjectionSettings> = {
    mode,
    screenVariancePx2: 0.04,
  };
  let maxScaled = 0;
  let maxRelative = 0;
  for (let sample = 0; sample < 64; sample++) {
    const params: AnisotropicProjectionParams = {
      position: [(rng() - 0.5) * 0.8, (rng() - 0.5) * 0.8, (rng() - 0.5) * 0.8],
      logScale: [
        Math.log(0.04 + 0.16 * rng()),
        Math.log(0.04 + 0.16 * rng()),
        Math.log(0.04 + 0.16 * rng()),
      ],
      quaternion: [rng() - 0.5, rng() - 0.5, rng() - 0.5, rng() - 0.5],
    };
    const upstream: AnisotropicProjectionUpstream = {
      meanPx: [(rng() - 0.5) * 0.5, (rng() - 0.5) * 0.5],
      conic: [(rng() - 0.5) * 0.5, (rng() - 0.5) * 0.5, (rng() - 0.5) * 0.5],
    };
    const analytic = backwardAnisotropicProjection(params, camera, upstream, settings);
    const groups = [
      ["position", params.position, analytic.position],
      ["logScale", params.logScale, analytic.logScale],
      ["quaternion", params.quaternion, analytic.quaternion],
    ] as const;
    for (const [name, raw, gradient] of groups) {
      for (let component = 0; component < raw.length; component++) {
        const epsilon = name === "position" ? 2e-6 : 1e-5;
        const plus: AnisotropicProjectionParams = {
          position: [...params.position],
          logScale: [...params.logScale],
          quaternion: [...params.quaternion],
        };
        const minus: AnisotropicProjectionParams = {
          position: [...params.position],
          logScale: [...params.logScale],
          quaternion: [...params.quaternion],
        };
        (plus[name] as number[])[component] += epsilon;
        (minus[name] as number[])[component] -= epsilon;
        const finite = (loss(plus, camera, settings, upstream) - loss(minus, camera, settings, upstream)) / (2 * epsilon);
        const absolute = Math.abs(gradient[component] - finite);
        const magnitude = Math.max(Math.abs(gradient[component]), Math.abs(finite));
        maxScaled = Math.max(maxScaled, absolute / (1 + magnitude));
        if (magnitude > 1e-5) {
          maxRelative = Math.max(maxRelative, absolute / (Math.abs(gradient[component]) + Math.abs(finite)));
        }
      }
    }
  }
  return { maxScaled, maxRelative };
}

for (const mode of ["legacy-affine", "perspective-jacobian"] as const) {
  const result = finiteDifferenceGate(mode);
  check(
    result.maxScaled < 2e-6 && result.maxRelative < 2e-5,
    `${mode} backward finite differences: scaled=${result.maxScaled.toExponential(2)}, relative=${result.maxRelative.toExponential(2)}`
  );
}

const gpuCamera = makeCamera([1.35, -0.62, 3.1], [-0.08, 0.12, -0.04]);
const gpuParams: AnisotropicProjectionParams = {
  position: [0.17, -0.22, 0.08],
  logScale: [Math.log(0.083), Math.log(0.137), Math.log(0.052)],
  quaternion: [0.31, -0.27, 0.18, 0.74],
};
const gpuUpstream: AnisotropicProjectionUpstream = {
  meanPx: [0.13, -0.21],
  conic: [0.17, -0.09, 0.23],
};

function wgslFloat(value: number): string {
  const text = Math.fround(value).toString();
  return /[.eE]/.test(text) ? text : `${text}.0`;
}
const wgslVec2 = (value: readonly number[]): string => `vec2f(${value.map(wgslFloat).join(", ")})`;
const wgslVec3 = (value: readonly number[]): string => `vec3f(${value.map(wgslFloat).join(", ")})`;
const wgslVec4 = (value: readonly number[]): string => `vec4f(${value.map(wgslFloat).join(", ")})`;

function gpuTestWGSL(mode: CovarianceProjectionMode): string {
  return /* wgsl */ `
${anisotropicProjectionWGSL({ mode })}
@group(0) @binding(0) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(1)
fn main() {
  let camera = AnisoCamera(
    ${wgslVec3(gpuCamera.eye)},
    ${wgslVec3(gpuCamera.right)},
    ${wgslVec3(gpuCamera.up)},
    ${wgslVec3(gpuCamera.forward)},
    ${wgslFloat(gpuCamera.focalPx)},
    ${wgslVec2(gpuCamera.centerPx)},
    ${wgslFloat(gpuCamera.near)}
  );
  let settings = AnisoProjectionSettings(0.01, 0.45, 0.04, 1e-12, 1e-12);
  let position = ${wgslVec3(gpuParams.position)};
  let log_scale = ${wgslVec3(gpuParams.logScale)};
  let quaternion = ${wgslVec4(gpuParams.quaternion)};
  let upstream = AnisoProjectionUpstream(${wgslVec2(gpuUpstream.meanPx)}, ${wgslVec3(gpuUpstream.conic)});
  let projected = anisoProject(position, log_scale, quaternion, camera, settings);
  let gradient = anisoProjectBackward(position, log_scale, quaternion, camera, settings, upstream);
  output[0] = projected.mean_px.x; output[1] = projected.mean_px.y;
  output[2] = projected.covariance.x; output[3] = projected.covariance.y; output[4] = projected.covariance.z;
  output[5] = projected.conic.x; output[6] = projected.conic.y; output[7] = projected.conic.z;
  output[8] = projected.normalized_quaternion.x; output[9] = projected.normalized_quaternion.y;
  output[10] = projected.normalized_quaternion.z; output[11] = projected.normalized_quaternion.w;
  output[12] = gradient.position.x; output[13] = gradient.position.y; output[14] = gradient.position.z;
  output[15] = gradient.log_scale.x; output[16] = gradient.log_scale.y; output[17] = gradient.log_scale.z;
  output[18] = gradient.quaternion.x; output[19] = gradient.quaternion.y;
  output[20] = gradient.quaternion.z; output[21] = gradient.quaternion.w;
}
`;
}

async function compileAndRunWGSL(device: any, mode: CovarianceProjectionMode): Promise<void> {
  const settings: Partial<AnisotropicProjectionSettings> = { mode, screenVariancePx2: 0.04 };
  const projected = projectAnisotropicGaussian(gpuParams, gpuCamera, settings);
  const gradient = backwardAnisotropicProjection(gpuParams, gpuCamera, gpuUpstream, settings);
  const expected = [
    ...projected.meanPx,
    ...projected.covariance,
    ...projected.conic,
    ...projected.normalizedQuaternion,
    ...gradient.position,
    ...gradient.logScale,
    ...gradient.quaternion,
  ];
  let pipeline: any;
  device.pushErrorScope("validation");
  try {
    const module = device.createShaderModule({ code: gpuTestWGSL(mode) });
    pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  } catch (error) {
    check(false, `${mode} WGSL pipeline creation threw: ${String(error)}`);
  }
  const validationError = await device.popErrorScope();
  if (validationError || !pipeline) {
    check(false, `${mode} WGSL compilation: ${validationError?.message ?? "pipeline unavailable"}`);
    return;
  }
  check(true, `${mode} WGSL compiles`);

  const usage = (globalThis as any).GPUBufferUsage;
  const mapMode = (globalThis as any).GPUMapMode;
  const output = device.createBuffer({ size: expected.length * 4, usage: usage.STORAGE | usage.COPY_SRC });
  const readback = device.createBuffer({ size: expected.length * 4, usage: usage.COPY_DST | usage.MAP_READ });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: output } }],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, expected.length * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(mapMode.READ);
  const actual = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  let worst = 0;
  for (let index = 0; index < expected.length; index++) {
    worst = Math.max(worst, Math.abs(actual[index] - expected[index]) / (1 + Math.abs(expected[index])));
  }
  check(worst < 3e-5, `${mode} WGSL matches CPU reference: worst scaled error ${worst.toExponential(2)}`);
  output.destroy();
  readback.destroy();
}

setupGlobals();
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.log("SKIP  WGSL compile/run: bun-webgpu found no adapter; CPU finite-difference gate completed");
} else {
  const device = await adapter.requestDevice();
  for (const mode of ["legacy-affine", "perspective-jacobian"] as const) {
    await compileAndRunWGSL(device, mode);
  }
  device.destroy?.();
}

if (failures > 0) {
  console.error(`\n${failures} anisotropic projection check(s) failed`);
  process.exit(1);
}
console.log("\nAll anisotropic projection checks passed");
