import type { PreparedCamera3D } from "../splat3d/cameras";
import type { CovarianceProjectionMode } from "./projection";
import { anisotropicProjectionWGSL } from "./projection_wgsl";

export const ANISO_TILE_3D = 16;
export const ANISO_DERIVED_STRIDE_3D = 12;
export const ANISO_ALPHA_THRESHOLD_3D = 1 / 255;
export const ANISO_MAX_ALPHA_3D = 0.99;
export const ANISO_TRANSMITTANCE_CUTOFF_3D = 1e-4;
export const ANISO_REGULARIZER_UNIFORM_BYTES_3D = 64;
export const ANISO_CENTER_SUM_SCALE_3D = 16384;

export interface AnisotropicRaster3DConfig {
  H: number;
  W: number;
  G: number;
  cap: number;
  bg?: [number, number, number];
  near?: number;
  far?: number;
  gradScale?: number;
  projectionMode?: CovarianceProjectionMode;
  minScale?: number;
  maxScale?: number;
  screenVariancePx2?: number;
}

export interface AnisotropicRaster3DDims {
  H: number;
  W: number;
  G: number;
  cap: number;
  tilesX: number;
  tilesY: number;
  numTiles: number;
  bg: [number, number, number];
  near: number;
  far: number;
  gradScale: number;
  projectionMode: CovarianceProjectionMode;
  minScale: number;
  maxScale: number;
  screenVariancePx2: number;
}

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`splat3d_aniso_raster: ${message}`);
}

function fl(value: number): string {
  assert(Number.isFinite(value), `non-finite WGSL literal ${value}`);
  const text = value.toString();
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

const uu = (value: number): string => `${value >>> 0}u`;
const v3 = (value: [number, number, number]): string =>
  `vec3f(${fl(value[0])}, ${fl(value[1])}, ${fl(value[2])})`;

export function resolveAnisotropicRaster3DDims(cfg: AnisotropicRaster3DConfig): AnisotropicRaster3DDims {
  assert(Number.isInteger(cfg.H) && Number.isInteger(cfg.W) && cfg.H > 0 && cfg.W > 0, "invalid image size");
  assert(cfg.H % ANISO_TILE_3D === 0 && cfg.W % ANISO_TILE_3D === 0, "H and W must be multiples of 16");
  assert(Number.isInteger(cfg.G) && cfg.G > 0, "G must be a positive integer");
  assert(cfg.cap >= 256 && (cfg.cap & (cfg.cap - 1)) === 0, "cap must be a power of two >= 256");
  assert(cfg.cap * 4 <= 16384, "cap exceeds the 16KB workgroup-storage budget");
  const minScale = cfg.minScale ?? 0.01;
  const maxScale = cfg.maxScale ?? 0.45;
  assert(minScale > 0 && maxScale > minScale, "scales require 0 < minScale < maxScale");
  const tilesX = cfg.W / ANISO_TILE_3D;
  const tilesY = cfg.H / ANISO_TILE_3D;
  return {
    H: cfg.H,
    W: cfg.W,
    G: cfg.G,
    cap: cfg.cap,
    tilesX,
    tilesY,
    numTiles: tilesX * tilesY,
    bg: cfg.bg ?? [0, 0, 0],
    near: cfg.near ?? 0.2,
    far: cfg.far ?? 12,
    gradScale: cfg.gradScale ?? 65536,
    projectionMode: cfg.projectionMode ?? "legacy-affine",
    minScale,
    maxScale,
    screenVariancePx2: Math.max(0, cfg.screenVariancePx2 ?? 0),
  };
}

function segments(d: AnisotropicRaster3DDims) {
  return {
    position: 0,
    logScale: 3 * d.G,
    quaternion: 6 * d.G,
    color: 10 * d.G,
    opacity: 13 * d.G,
  };
}

function cameraBlock(camera: PreparedCamera3D, d: AnisotropicRaster3DDims): string {
  return /* wgsl */ `
const CAMERA = AnisoCamera(
  ${v3(camera.eye)},
  ${v3(camera.right)},
  ${v3(camera.cameraUp)},
  ${v3(camera.forward)},
  ${fl(camera.focalPx)},
  vec2f(${fl(d.W * 0.5)}, ${fl(d.H * 0.5)}),
  ${fl(d.near)}
);
const SETTINGS = AnisoProjectionSettings(
  ${fl(d.minScale)},
  ${fl(d.maxScale)},
  ${fl(d.screenVariancePx2)},
  1e-12,
  1e-12
);
`;
}

const SIGMOID = /* wgsl */ `fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }`;

export function anisotropicPrepShader3D(cfg: AnisotropicRaster3DConfig, camera: PreparedCamera3D): string {
  const d = resolveAnisotropicRaster3DDims(cfg);
  const s = segments(d);
  return /* wgsl */ `
${SIGMOID}
${anisotropicProjectionWGSL({ mode: d.projectionMode })}
${cameraBlock(camera, d)}
@group(0) @binding(0) var<storage, read> params : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }
  let position = vec3f(
    params[${uu(s.position)} + 3u * g + 0u],
    params[${uu(s.position)} + 3u * g + 1u],
    params[${uu(s.position)} + 3u * g + 2u]
  );
  let logScale = vec3f(
    params[${uu(s.logScale)} + 3u * g + 0u],
    params[${uu(s.logScale)} + 3u * g + 1u],
    params[${uu(s.logScale)} + 3u * g + 2u]
  );
  let quaternion = vec4f(
    params[${uu(s.quaternion)} + 4u * g + 0u],
    params[${uu(s.quaternion)} + 4u * g + 1u],
    params[${uu(s.quaternion)} + 4u * g + 2u],
    params[${uu(s.quaternion)} + 4u * g + 3u]
  );
  let projected = anisoProject(position, logScale, quaternion, CAMERA, SETTINGS);
  let b = g * ${uu(ANISO_DERIVED_STRIDE_3D)};
  derived[b + 0u] = projected.mean_px.x;
  derived[b + 1u] = projected.mean_px.y;
  derived[b + 2u] = projected.conic.x;
  derived[b + 3u] = projected.conic.y;
  derived[b + 4u] = projected.conic.z;
  derived[b + 5u] = projected.camera_position.z;
  derived[b + 6u] = sigmoid1(params[${uu(s.color)} + 3u * g + 0u]);
  derived[b + 7u] = sigmoid1(params[${uu(s.color)} + 3u * g + 1u]);
  derived[b + 8u] = sigmoid1(params[${uu(s.color)} + 3u * g + 2u]);
  derived[b + 9u] = sigmoid1(params[${uu(s.opacity)} + g]);
  derived[b + 10u] = projected.covariance.x;
  derived[b + 11u] = projected.covariance.z;
}
`;
}

export function anisotropicEmitShader3D(cfg: AnisotropicRaster3DConfig): string {
  const d = resolveAnisotropicRaster3DDims(cfg);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> derived : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }
  let b = g * ${uu(ANISO_DERIVED_STRIDE_3D)};
  let depth = derived[b + 5u];
  if (depth <= ${fl(d.near)} || depth >= ${fl(d.far)}) { return; }
  let opacity = derived[b + 9u];
  if (opacity <= ${fl(ANISO_ALPHA_THRESHOLD_3D)}) { return; }
  let ratio = max(${fl(ANISO_ALPHA_THRESHOLD_3D)} / max(opacity, 1e-8), 1e-8);
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let radiusX = sqrt(max(tau * derived[b + 10u], 0.0));
  let radiusY = sqrt(max(tau * derived[b + 11u], 0.0));
  let sx = derived[b + 0u];
  let sy = derived[b + 1u];
  let x0 = max(0, i32(floor(sx - radiusX - 0.5)));
  let x1 = min(${d.W - 1}, i32(ceil(sx + radiusX - 0.5)));
  let y0 = max(0, i32(floor(sy - radiusY - 0.5)));
  let y1 = min(${d.H - 1}, i32(ceil(sy + radiusY - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }
  let tx0 = x0 / ${ANISO_TILE_3D}; let tx1 = x1 / ${ANISO_TILE_3D};
  let ty0 = y0 / ${ANISO_TILE_3D}; let ty1 = y1 / ${ANISO_TILE_3D};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let tile = u32(ty * ${d.tilesX} + tx);
      let slot = atomicAdd(&tileCounts[tile], 1u);
      if (slot < ${uu(d.cap)}) { binnedIds[tile * ${uu(d.cap)} + slot] = g; }
    }
  }
}
`;
}

export function anisotropicForwardShader3D(cfg: AnisotropicRaster3DConfig): string {
  const d = resolveAnisotropicRaster3DDims(cfg);
  const hw = d.H * d.W;
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds : array<u32>;
@group(0) @binding(2) var<storage, read> derived : array<f32>;
@group(0) @binding(3) var<storage, read_write> image : array<f32>;
@group(0) @binding(4) var<storage, read_write> tileStop : array<u32>;
var<workgroup> shIds : array<u32, ${d.cap}>;

fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u) - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}
fn idGreater(a : u32, b : u32) -> bool {
  if (a == 0xffffffffu) { return b != 0xffffffffu; }
  if (b == 0xffffffffu) { return false; }
  let za = derived[a * ${uu(ANISO_DERIVED_STRIDE_3D)} + 5u];
  let zb = derived[b * ${uu(ANISO_DERIVED_STRIDE_3D)} + 5u];
  if (za == zb) { return a > b; }
  return za > zb;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) tid : u32) {
  let tile = wg.x;
  if (tile >= ${uu(d.numTiles)}) { return; }
  let count = min(tileCounts[tile], ${uu(d.cap)});
  let start = tile * ${uu(d.cap)};
  let sortN = nextPow2(count);
  for (var i = tid; i < sortN; i = i + 256u) {
    shIds[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  workgroupBarrier();
  var k = 2u;
  loop {
    if (k > sortN) { break; }
    var j = k >> 1u;
    loop {
      if (j == 0u) { break; }
      let pairs = sortN >> 1u;
      for (var pair = tid; pair < pairs; pair = pair + 256u) {
        let pos = 2u * j * (pair / j) + pair % j;
        let other = pos + j;
        let ascending = (pos & k) == 0u;
        let a = shIds[pos]; let b = shIds[other];
        let swap = select(idGreater(b, a), idGreater(a, b), ascending);
        if (swap) { shIds[pos] = b; shIds[other] = a; }
      }
      workgroupBarrier();
      j >>= 1u;
    }
    k <<= 1u;
  }
  for (var i = tid; i < count; i = i + 256u) { binnedIds[start + i] = shIds[i]; }
  workgroupBarrier();

  let tileX = tile % ${uu(d.tilesX)};
  let tileY = tile / ${uu(d.tilesX)};
  let x = tileX * ${ANISO_TILE_3D}u + tid % ${ANISO_TILE_3D}u;
  let y = tileY * ${ANISO_TILE_3D}u + tid / ${ANISO_TILE_3D}u;
  var localStop = 0u;
  if (x < ${uu(d.W)} && y < ${uu(d.H)}) {
    let px = f32(x) + 0.5; let py = f32(y) + 0.5;
    var r = 0.0; var g = 0.0; var b = 0.0; var T = 1.0;
    for (var i = 0u; i < count; i++) {
      let base = shIds[i] * ${uu(ANISO_DERIVED_STRIDE_3D)};
      let dx = px - derived[base + 0u]; let dy = py - derived[base + 1u];
      let power = -0.5 * (derived[base + 2u] * dx * dx + 2.0 * derived[base + 3u] * dx * dy + derived[base + 4u] * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let alpha = min(${fl(ANISO_MAX_ALPHA_3D)}, derived[base + 9u] * exp(power));
      if (alpha < ${fl(ANISO_ALPHA_THRESHOLD_3D)}) { continue; }
      let weight = T * alpha;
      r += weight * derived[base + 6u]; g += weight * derived[base + 7u]; b += weight * derived[base + 8u];
      T *= 1.0 - alpha;
      if (T < ${fl(ANISO_TRANSMITTANCE_CUTOFF_3D)}) { break; }
    }
    let pixel = y * ${uu(d.W)} + x;
    image[0u * ${uu(hw)} + pixel] = r + T * ${fl(d.bg[0])};
    image[1u * ${uu(hw)} + pixel] = g + T * ${fl(d.bg[1])};
    image[2u * ${uu(hw)} + pixel] = b + T * ${fl(d.bg[2])};
  }
  workgroupBarrier();
  shIds[tid] = localStop;
  workgroupBarrier();
  var offset = 128u;
  loop {
    if (offset == 0u) { break; }
    if (tid < offset) { shIds[tid] = max(shIds[tid], shIds[tid + offset]); }
    workgroupBarrier();
    offset >>= 1u;
  }
  if (tid == 0u) { tileStop[tile] = shIds[0]; }
}
`;
}

export function anisotropicBackwardShader3D(cfg: AnisotropicRaster3DConfig): string {
  const d = resolveAnisotropicRaster3DDims(cfg);
  const hw = d.H * d.W;
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> gradImage : array<f32>;
@group(0) @binding(1) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read> binnedIds : array<u32>;
@group(0) @binding(3) var<storage, read> tileStop : array<u32>;
@group(0) @binding(4) var<storage, read> derived : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad : array<atomic<i32>>;
var<workgroup> shIds : array<u32, ${d.cap}>;
fn fixadd(base : u32, slot : u32, value : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(value * ${fl(d.gradScale)}), -2.14e9, 2.14e9)));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) tid : u32) {
  let tile = wg.x;
  if (tile >= ${uu(d.numTiles)}) { return; }
  let count = min(tileCounts[tile], ${uu(d.cap)});
  let stop = min(count, tileStop[tile]);
  let start = tile * ${uu(d.cap)};
  for (var i = tid; i < stop; i += 256u) { shIds[i] = binnedIds[start + i]; }
  workgroupBarrier();
  let tileX = tile % ${uu(d.tilesX)}; let tileY = tile / ${uu(d.tilesX)};
  let x = tileX * ${ANISO_TILE_3D}u + tid % ${ANISO_TILE_3D}u;
  let y = tileY * ${ANISO_TILE_3D}u + tid / ${ANISO_TILE_3D}u;
  if (x >= ${uu(d.W)} || y >= ${uu(d.H)}) { return; }
  let px = f32(x) + 0.5; let py = f32(y) + 0.5;
  let pixel = y * ${uu(d.W)} + x;
  let go = vec3f(
    gradImage[0u * ${uu(hw)} + pixel],
    gradImage[1u * ${uu(hw)} + pixel],
    gradImage[2u * ${uu(hw)} + pixel]
  );
  var T = 1.0; var end = stop;
  for (var i = 0u; i < stop; i++) {
    let base = shIds[i] * ${uu(ANISO_DERIVED_STRIDE_3D)};
    let dx = px - derived[base + 0u]; let dy = py - derived[base + 1u];
    let power = -0.5 * (derived[base + 2u] * dx * dx + 2.0 * derived[base + 3u] * dx * dy + derived[base + 4u] * dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${fl(ANISO_MAX_ALPHA_3D)}, derived[base + 9u] * exp(power));
    if (alpha < ${fl(ANISO_ALPHA_THRESHOLD_3D)}) { continue; }
    T *= 1.0 - alpha;
    if (T < ${fl(ANISO_TRANSMITTANCE_CUTOFF_3D)}) { end = i + 1u; break; }
  }
  var currentT = T;
  var gT = dot(go, ${v3(d.bg)});
  for (var ii = i32(end) - 1; ii >= 0; ii--) {
    let base = shIds[u32(ii)] * ${uu(ANISO_DERIVED_STRIDE_3D)};
    let dx = px - derived[base + 0u]; let dy = py - derived[base + 1u];
    let a = derived[base + 2u]; let cross = derived[base + 3u]; let c = derived[base + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * cross * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let opacity = derived[base + 9u];
    let raw = opacity * exp(power);
    let alpha = min(${fl(ANISO_MAX_ALPHA_3D)}, raw);
    if (alpha < ${fl(ANISO_ALPHA_THRESHOLD_3D)}) { continue; }
    let previousT = currentT / max(1.0 - alpha, 1e-8);
    let color = vec3f(derived[base + 6u], derived[base + 7u], derived[base + 8u]);
    let dotgc = dot(go, color);
    let gAlpha = previousT * (dotgc - gT);
    fixadd(base, 6u, go.x * previousT * alpha);
    fixadd(base, 7u, go.y * previousT * alpha);
    fixadd(base, 8u, go.z * previousT * alpha);
    let gate = select(0.0, 1.0, raw < ${fl(ANISO_MAX_ALPHA_3D)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    fixadd(base, 0u, gPower * (a * dx + cross * dy));
    fixadd(base, 1u, gPower * (cross * dx + c * dy));
    fixadd(base, 2u, gPower * (-0.5 * dx * dx));
    fixadd(base, 3u, gPower * (-dx * dy));
    fixadd(base, 4u, gPower * (-0.5 * dy * dy));
    fixadd(base, 9u, gRaw * raw / max(opacity, 1e-8));
    gT = alpha * dotgc + (1.0 - alpha) * gT;
    currentT = previousT;
  }
}
`;
}

export function anisotropicChainShader3D(cfg: AnisotropicRaster3DConfig, camera: PreparedCamera3D): string {
  const d = resolveAnisotropicRaster3DDims(cfg);
  const s = segments(d);
  return /* wgsl */ `
${anisotropicProjectionWGSL({ mode: d.projectionMode })}
${cameraBlock(camera, d)}
@group(0) @binding(0) var<storage, read> accGrad : array<i32>;
@group(0) @binding(1) var<storage, read> derived : array<f32>;
@group(0) @binding(2) var<storage, read> params : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }
  let b = g * ${uu(ANISO_DERIVED_STRIDE_3D)};
  let inv = ${fl(1 / d.gradScale)};
  let position = vec3f(
    params[${uu(s.position)} + 3u * g + 0u], params[${uu(s.position)} + 3u * g + 1u], params[${uu(s.position)} + 3u * g + 2u]
  );
  let logScale = vec3f(
    params[${uu(s.logScale)} + 3u * g + 0u], params[${uu(s.logScale)} + 3u * g + 1u], params[${uu(s.logScale)} + 3u * g + 2u]
  );
  let quaternion = vec4f(
    params[${uu(s.quaternion)} + 4u * g + 0u], params[${uu(s.quaternion)} + 4u * g + 1u],
    params[${uu(s.quaternion)} + 4u * g + 2u], params[${uu(s.quaternion)} + 4u * g + 3u]
  );
  let upstream = AnisoProjectionUpstream(
    vec2f(f32(accGrad[b + 0u]), f32(accGrad[b + 1u])) * inv,
    vec3f(f32(accGrad[b + 2u]), f32(accGrad[b + 3u]), f32(accGrad[b + 4u])) * inv
  );
  let gradient = anisoProjectBackward(position, logScale, quaternion, CAMERA, SETTINGS, upstream);
  gradRaw[${uu(s.position)} + 3u * g + 0u] += gradient.position.x;
  gradRaw[${uu(s.position)} + 3u * g + 1u] += gradient.position.y;
  gradRaw[${uu(s.position)} + 3u * g + 2u] += gradient.position.z;
  gradRaw[${uu(s.logScale)} + 3u * g + 0u] += gradient.log_scale.x;
  gradRaw[${uu(s.logScale)} + 3u * g + 1u] += gradient.log_scale.y;
  gradRaw[${uu(s.logScale)} + 3u * g + 2u] += gradient.log_scale.z;
  gradRaw[${uu(s.quaternion)} + 4u * g + 0u] += gradient.quaternion.x;
  gradRaw[${uu(s.quaternion)} + 4u * g + 1u] += gradient.quaternion.y;
  gradRaw[${uu(s.quaternion)} + 4u * g + 2u] += gradient.quaternion.z;
  gradRaw[${uu(s.quaternion)} + 4u * g + 3u] += gradient.quaternion.w;
  let color = vec3f(derived[b + 6u], derived[b + 7u], derived[b + 8u]);
  let colorGradient = vec3f(f32(accGrad[b + 6u]), f32(accGrad[b + 7u]), f32(accGrad[b + 8u])) * inv;
  gradRaw[${uu(s.color)} + 3u * g + 0u] += colorGradient.x * color.x * (1.0 - color.x);
  gradRaw[${uu(s.color)} + 3u * g + 1u] += colorGradient.y * color.y * (1.0 - color.y);
  gradRaw[${uu(s.color)} + 3u * g + 2u] += colorGradient.z * color.z * (1.0 - color.z);
  let opacity = derived[b + 9u];
  gradRaw[${uu(s.opacity)} + g] += f32(accGrad[b + 9u]) * inv * opacity * (1.0 - opacity);
}
`;
}

export function anisotropicClearShader3D(words: number): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> buffer : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x < ${uu(words)}) { buffer[gid.x] = 0u; }
}
`;
}

export function anisotropicCenterReduceShader3D(cfg: AnisotropicRaster3DConfig): string {
  const d = resolveAnisotropicRaster3DDims(cfg);
  const s = segments(d);
  return /* wgsl */ `
${SIGMOID}
@group(0) @binding(0) var<storage, read> params : array<f32>;
@group(0) @binding(1) var<storage, read_write> centerSum : array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }
  let base = ${uu(s.position)} + 3u * g;
  let mass = sigmoid1(params[${uu(s.opacity)} + g]);
  atomicAdd(&centerSum[0], i32(round(params[base + 0u] * mass * ${fl(ANISO_CENTER_SUM_SCALE_3D)})));
  atomicAdd(&centerSum[1], i32(round(params[base + 1u] * mass * ${fl(ANISO_CENTER_SUM_SCALE_3D)})));
  atomicAdd(&centerSum[2], i32(round(params[base + 2u] * mass * ${fl(ANISO_CENTER_SUM_SCALE_3D)})));
  atomicAdd(&centerSum[3], i32(round(mass * ${fl(ANISO_CENTER_SUM_SCALE_3D)})));
}
`;
}

export function anisotropicRegularizerShader3D(cfg: AnisotropicRaster3DConfig): string {
  const d = resolveAnisotropicRaster3DDims(cfg);
  const s = segments(d);
  return /* wgsl */ `
${SIGMOID}
struct RegU {
  centerWeight      : f32,
  radiusWeight      : f32,
  targetRadius      : f32,
  opacitySparsity   : f32,
  smallRadiusWeight : f32,
  smallRadius       : f32,
  radiusBandWeight  : f32,
  minRadius         : f32,
  maxRadius         : f32,
  _pad0             : f32,
  _pad1             : f32,
  _pad2             : f32,
};
@group(0) @binding(0) var<uniform> u : RegU;
@group(0) @binding(1) var<storage, read> params : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradRaw : array<f32>;
@group(0) @binding(3) var<storage, read_write> centerSum : array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }

  let pxIdx = ${uu(s.position)} + 3u * g + 0u;
  let pyIdx = ${uu(s.position)} + 3u * g + 1u;
  let pzIdx = ${uu(s.position)} + 3u * g + 2u;
  let p = vec3f(params[pxIdx], params[pyIdx], params[pzIdx]);
  let positionRadius = length(p);
  let invPositionRadius = 1.0 / max(positionRadius, 1e-8);
  let outside = max(0.0, positionRadius - max(u.targetRadius, 1e-8));
  let centerScale = 1.0 / max(f32(atomicLoad(&centerSum[3])), 1.0);
  let center = vec3f(
    f32(atomicLoad(&centerSum[0])),
    f32(atomicLoad(&centerSum[1])),
    f32(atomicLoad(&centerSum[2]))
  ) * centerScale;
  let gp = 2.0 * u.centerWeight * center
    + (2.0 * u.radiusWeight * outside * invPositionRadius) * p;
  gradRaw[pxIdx] += gp.x;
  gradRaw[pyIdx] += gp.y;
  gradRaw[pzIdx] += gp.z;

  let opacityIdx = ${uu(s.opacity)} + g;
  let opacity = sigmoid1(params[opacityIdx]);
  gradRaw[opacityIdx] += u.opacitySparsity * opacity * (1.0 - opacity);

  let scaleBase = ${uu(s.logScale)} + 3u * g;
  let meanLogScale = (
    params[scaleBase + 0u] + params[scaleBase + 1u] + params[scaleBase + 2u]
  ) / 3.0;
  let radius = exp(meanLogScale);
  let small = max(0.0, max(u.smallRadius, 1e-8) - radius);
  let smallLossGrad = u.smallRadiusWeight * small * small;
  gradRaw[opacityIdx] += 2.0 * smallLossGrad * opacity * opacity * (1.0 - opacity);

  let minRadius = max(u.minRadius, 1e-8);
  let maxRadius = max(u.maxRadius, minRadius + 1e-8);
  let under = max(0.0, minRadius - radius);
  let over = max(0.0, radius - maxRadius);
  let radiusLossDerivative = -2.0 * u.smallRadiusWeight * opacity * opacity * small
    + u.radiusBandWeight * (-2.0 * under + 2.0 * over);
  let perAxisLogScaleGradient = radiusLossDerivative * radius / 3.0;
  gradRaw[scaleBase + 0u] += perAxisLogScaleGradient;
  gradRaw[scaleBase + 1u] += perAxisLogScaleGradient;
  gradRaw[scaleBase + 2u] += perAxisLogScaleGradient;
}
`;
}
