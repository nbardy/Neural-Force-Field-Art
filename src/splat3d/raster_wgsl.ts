import type { PreparedCamera3D } from "./cameras";

export const TILE = 16;
export const PARAM_STRIDE_3D = 8;
export const DERIVED_STRIDE_3D = 11;
export const CAMERA_STRIDE_3D = 16;
export const ALPHA_THRESHOLD = 1.0 / 255.0;
export const MAX_ALPHA = 0.99;
export const TRANSMITTANCE_CUTOFF = 1e-4;
export const EPS = 1e-8;
export const RADIUS_MIN = 0.01;
export const RADIUS_MAX = 0.45;

export interface Raster3DConfig {
  H: number;
  W: number;
  G: number;
  cap: number;
  bg?: [number, number, number];
  dynamicBg?: boolean;
  near?: number;
  far?: number;
  gradScale?: number;
}

export interface Raster3DDims {
  H: number;
  W: number;
  G: number;
  cap: number;
  tilesX: number;
  tilesY: number;
  numTiles: number;
  bg: [number, number, number];
  dynamicBg: boolean;
  near: number;
  far: number;
  gradScale: number;
}

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`raster3d_wgsl: ${msg}`);
}

function fl(x: number): string {
  assert(Number.isFinite(x), `non-finite literal ${x}`);
  let s = x.toString();
  if (!/[.eE]/.test(s)) s += ".0";
  return s;
}

const uu = (x: number): string => `${x >>> 0}u`;
const v3 = (v: [number, number, number]): string => `vec3f(${fl(v[0])}, ${fl(v[1])}, ${fl(v[2])})`;

export function resolveDims3D(cfg: Raster3DConfig): Raster3DDims {
  assert(cfg.H > 0 && cfg.W > 0 && cfg.G > 0, "H,W,G must be positive");
  assert(cfg.H % TILE === 0 && cfg.W % TILE === 0, `H,W must be multiples of ${TILE}`);
  assert((cfg.cap & (cfg.cap - 1)) === 0 && cfg.cap > 0, "cap must be a power of two");
  assert(cfg.cap * 4 <= 16384, `cap*4 (${cfg.cap * 4}B) exceeds 16KB workgroup storage`);
  return {
    H: cfg.H,
    W: cfg.W,
    G: cfg.G,
    cap: cfg.cap,
    tilesX: cfg.W / TILE,
    tilesY: cfg.H / TILE,
    numTiles: (cfg.W / TILE) * (cfg.H / TILE),
    bg: cfg.bg ?? [0, 0, 0],
    dynamicBg: cfg.dynamicBg ?? false,
    near: cfg.near ?? 0.2,
    far: cfg.far ?? 12,
    gradScale: cfg.gradScale ?? 65536,
  };
}

function seg(d: Raster3DDims) {
  return {
    position: 0,
    logRadius: 3 * d.G,
    colorRaw: 4 * d.G,
    opacityRaw: 7 * d.G,
  };
}

const SIGMOID = /* wgsl */ `fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }`;

function bgUniformDecl(binding: number): string {
  return /* wgsl */ `
struct BgU {
  rgb : vec3f,
  _pad : f32,
};
@group(0) @binding(${binding}) var<uniform> bgU : BgU;
`;
}

function bgExpr(d: Raster3DDims, channel: 0 | 1 | 2): string {
  if (!d.dynamicBg) return fl(d.bg[channel]);
  return channel === 0 ? "bgU.rgb.x" : channel === 1 ? "bgU.rgb.y" : "bgU.rgb.z";
}

function cameraBlock(cam: PreparedCamera3D): string {
  return /* wgsl */ `
const CAM_EYE = ${v3(cam.eye)};
const CAM_RIGHT = ${v3(cam.right)};
const CAM_UP = ${v3(cam.cameraUp)};
const CAM_FWD = ${v3(cam.forward)};
const FOCAL_PX = ${fl(cam.focalPx)};
`;
}

function cameraLoadBlock(): string {
  return /* wgsl */ `
fn cameraBase(view : u32) -> u32 {
  return view * ${uu(CAMERA_STRIDE_3D)};
}

fn cameraEye(view : u32) -> vec3f {
  let b = cameraBase(view);
  return vec3f(cameras[b + 0u], cameras[b + 1u], cameras[b + 2u]);
}

fn cameraRight(view : u32) -> vec3f {
  let b = cameraBase(view);
  return vec3f(cameras[b + 3u], cameras[b + 4u], cameras[b + 5u]);
}

fn cameraUp(view : u32) -> vec3f {
  let b = cameraBase(view);
  return vec3f(cameras[b + 6u], cameras[b + 7u], cameras[b + 8u]);
}

fn cameraFwd(view : u32) -> vec3f {
  let b = cameraBase(view);
  return vec3f(cameras[b + 9u], cameras[b + 10u], cameras[b + 11u]);
}

fn cameraFocalPx(view : u32) -> f32 {
  return cameras[cameraBase(view) + 12u];
}
`;
}

export function prepShader3D(cfg: Raster3DConfig, cam: PreparedCamera3D): string {
  const d = resolveDims3D(cfg);
  const s = seg(d);
  return /* wgsl */ `
${SIGMOID}
${cameraBlock(cam)}
@group(0) @binding(0) var<storage, read>       params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }

  let p = vec3f(
    params[${uu(s.position)} + g * 3u + 0u],
    params[${uu(s.position)} + g * 3u + 1u],
    params[${uu(s.position)} + g * 3u + 2u]
  );
  let w = p - CAM_EYE;
  let vx = dot(w, CAM_RIGHT);
  let vy = dot(w, CAM_UP);
  let vz = dot(w, CAM_FWD);
  let safeZ = max(vz, ${fl(d.near)});
  let radiusWorld = clamp(exp(params[${uu(s.logRadius)} + g]), ${fl(RADIUS_MIN)}, ${fl(RADIUS_MAX)});
  let radiusPx = max(FOCAL_PX * radiusWorld / safeZ, 0.25);
  let invR2 = 1.0 / max(radiusPx * radiusPx, ${fl(EPS)});
  let sx = ${fl(d.W * 0.5)} + FOCAL_PX * (vx / safeZ);
  let sy = ${fl(d.H * 0.5)} - FOCAL_PX * (vy / safeZ);

  let base = g * ${uu(DERIVED_STRIDE_3D)};
  derived[base + 0u] = sx;
  derived[base + 1u] = sy;
  derived[base + 2u] = invR2;
  derived[base + 3u] = vz;
  derived[base + 4u] = vx;
  derived[base + 5u] = vy;
  derived[base + 6u] = safeZ;
  derived[base + 7u] = sigmoid1(params[${uu(s.colorRaw)} + g * 3u + 0u]);
  derived[base + 8u] = sigmoid1(params[${uu(s.colorRaw)} + g * 3u + 1u]);
  derived[base + 9u] = sigmoid1(params[${uu(s.colorRaw)} + g * 3u + 2u]);
  derived[base + 10u] = sigmoid1(params[${uu(s.opacityRaw)} + g]);
}
`;
}

export function prepBatchShader3D(cfg: Raster3DConfig): string {
  const d = resolveDims3D(cfg);
  const s = seg(d);
  return /* wgsl */ `
${SIGMOID}
@group(0) @binding(0) var<storage, read>       params      : array<f32>;
@group(0) @binding(1) var<storage, read>       cameras     : array<f32>;
@group(0) @binding(2) var<storage, read>       activeViews : array<u32>;
@group(0) @binding(3) var<storage, read_write> derived     : array<f32>;

${cameraLoadBlock()}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  let lane = gid.z;
  if (g >= ${uu(d.G)}) { return; }
  let view = activeViews[lane];
  let eye = cameraEye(view);
  let right = cameraRight(view);
  let up = cameraUp(view);
  let fwd = cameraFwd(view);
  let focalPx = cameraFocalPx(view);

  let p = vec3f(
    params[${uu(s.position)} + g * 3u + 0u],
    params[${uu(s.position)} + g * 3u + 1u],
    params[${uu(s.position)} + g * 3u + 2u]
  );
  let w = p - eye;
  let vx = dot(w, right);
  let vy = dot(w, up);
  let vz = dot(w, fwd);
  let safeZ = max(vz, ${fl(d.near)});
  let radiusWorld = clamp(exp(params[${uu(s.logRadius)} + g]), ${fl(RADIUS_MIN)}, ${fl(RADIUS_MAX)});
  let radiusPx = max(focalPx * radiusWorld / safeZ, 0.25);
  let invR2 = 1.0 / max(radiusPx * radiusPx, ${fl(EPS)});
  let sx = ${fl(d.W * 0.5)} + focalPx * (vx / safeZ);
  let sy = ${fl(d.H * 0.5)} - focalPx * (vy / safeZ);

  let base = lane * ${uu(d.G * DERIVED_STRIDE_3D)} + g * ${uu(DERIVED_STRIDE_3D)};
  derived[base + 0u] = sx;
  derived[base + 1u] = sy;
  derived[base + 2u] = invR2;
  derived[base + 3u] = vz;
  derived[base + 4u] = vx;
  derived[base + 5u] = vy;
  derived[base + 6u] = safeZ;
  derived[base + 7u] = sigmoid1(params[${uu(s.colorRaw)} + g * 3u + 0u]);
  derived[base + 8u] = sigmoid1(params[${uu(s.colorRaw)} + g * 3u + 1u]);
  derived[base + 9u] = sigmoid1(params[${uu(s.colorRaw)} + g * 3u + 2u]);
  derived[base + 10u] = sigmoid1(params[${uu(s.opacityRaw)} + g]);
}
`;
}

export function emitShader3D(cfg: Raster3DConfig): string {
  const d = resolveDims3D(cfg);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       derived    : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds  : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }
  let base = g * ${uu(DERIVED_STRIDE_3D)};
  let depth = derived[base + 3u];
  if (depth <= ${fl(d.near)} || depth >= ${fl(d.far)}) { return; }
  let op = derived[base + 10u];
  if (op <= ${fl(ALPHA_THRESHOLD)}) { return; }
  let ratio = max(${fl(ALPHA_THRESHOLD)} / max(op, ${fl(EPS)}), ${fl(EPS)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let sx = derived[base + 0u];
  let sy = derived[base + 1u];
  let invR2 = max(derived[base + 2u], ${fl(EPS)});
  let radius = sqrt(tau / invR2);
  let x0 = max(0, i32(floor(sx - radius - 0.5)));
  let x1 = min(${d.W - 1}, i32(ceil(sx + radius - 0.5)));
  let y0 = max(0, i32(floor(sy - radius - 0.5)));
  let y1 = min(${d.H - 1}, i32(ceil(sy + radius - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tx0 = x0 / ${TILE}; let tx1 = x1 / ${TILE};
  let ty0 = y0 / ${TILE}; let ty1 = y1 / ${TILE};
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

export function emitBatchShader3D(cfg: Raster3DConfig): string {
  const d = resolveDims3D(cfg);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       derived    : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds  : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  let lane = gid.z;
  if (g >= ${uu(d.G)}) { return; }
  let derivedBase = lane * ${uu(d.G * DERIVED_STRIDE_3D)} + g * ${uu(DERIVED_STRIDE_3D)};
  let depth = derived[derivedBase + 3u];
  if (depth <= ${fl(d.near)} || depth >= ${fl(d.far)}) { return; }
  let op = derived[derivedBase + 10u];
  if (op <= ${fl(ALPHA_THRESHOLD)}) { return; }
  let ratio = max(${fl(ALPHA_THRESHOLD)} / max(op, ${fl(EPS)}), ${fl(EPS)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let sx = derived[derivedBase + 0u];
  let sy = derived[derivedBase + 1u];
  let invR2 = max(derived[derivedBase + 2u], ${fl(EPS)});
  let radius = sqrt(tau / invR2);
  let x0 = max(0, i32(floor(sx - radius - 0.5)));
  let x1 = min(${d.W - 1}, i32(ceil(sx + radius - 0.5)));
  let y0 = max(0, i32(floor(sy - radius - 0.5)));
  let y1 = min(${d.H - 1}, i32(ceil(sy + radius - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tileCountsBase = lane * ${uu(d.numTiles)};
  let binnedBase = lane * ${uu(d.numTiles * d.cap)};
  let tx0 = x0 / ${TILE}; let tx1 = x1 / ${TILE};
  let ty0 = y0 / ${TILE}; let ty1 = y1 / ${TILE};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let tile = u32(ty * ${d.tilesX} + tx);
      let slot = atomicAdd(&tileCounts[tileCountsBase + tile], 1u);
      if (slot < ${uu(d.cap)}) { binnedIds[binnedBase + tile * ${uu(d.cap)} + slot] = g; }
    }
  }
}
`;
}

export function forwardShader3D(cfg: Raster3DConfig): string {
  const d = resolveDims3D(cfg);
  const HW = d.H * d.W;
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;
${d.dynamicBg ? bgUniformDecl(5) : ""}

var<workgroup> sh_ids     : array<u32, ${d.cap}>;
var<workgroup> sh_maxstop : atomic<u32>;

fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u); v = v - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

fn idGreater(a : u32, b : u32) -> bool {
  if (a == 0xffffffffu) { return b != 0xffffffffu; }
  if (b == 0xffffffffu) { return false; }
  let za = derived[a * ${uu(DERIVED_STRIDE_3D)} + 3u];
  let zb = derived[b * ${uu(DERIVED_STRIDE_3D)} + 3u];
  if (za == zb) { return a > b; }
  return za > zb;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${uu(d.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${uu(d.cap)});
  let start = tileId * ${uu(d.cap)};
  let sortN = nextPow2(count);

  for (var i = tid; i < sortN; i = i + 256u) {
    sh_ids[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  if (tid == 0u) { atomicStore(&sh_maxstop, 0u); }
  workgroupBarrier();

  var k = 2u;
  loop {
    if (k > sortN) { break; }
    var j = k >> 1u;
    loop {
      if (j == 0u) { break; }
      let nPairs = sortN >> 1u;
      for (var pair = tid; pair < nPairs; pair = pair + 256u) {
        let pos = 2u * j * (pair / j) + (pair % j);
        let ixj = pos + j;
        let asc = (pos & k) == 0u;
        let va = sh_ids[pos];
        let vb = sh_ids[ixj];
        let swapAsc = idGreater(va, vb);
        let swapDesc = idGreater(vb, va);
        if ((asc && swapAsc) || (!asc && swapDesc)) { sh_ids[pos] = vb; sh_ids[ixj] = va; }
      }
      workgroupBarrier();
      j = j >> 1u;
    }
    k = k << 1u;
  }

  for (var i = tid; i < count; i = i + 256u) { binnedIds[start + i] = sh_ids[i]; }
  workgroupBarrier();

  let tileX = tileId % ${uu(d.tilesX)};
  let tileY = tileId / ${uu(d.tilesX)};
  let x = tileX * ${TILE}u + (tid % ${TILE}u);
  let y = tileY * ${TILE}u + (tid / ${TILE}u);
  var localStop = 0u;
  if (x < ${uu(d.W)} && y < ${uu(d.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var accR = 0.0; var accG = 0.0; var accB = 0.0; var T = 1.0;
    for (var i = 0u; i < count; i = i + 1u) {
      let gg = sh_ids[i];
      let b = gg * ${uu(DERIVED_STRIDE_3D)};
      let dx = pxc - derived[b + 0u];
      let dy = pyc - derived[b + 1u];
      let invR2 = derived[b + 2u];
      let power = -0.5 * invR2 * (dx * dx + dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[b + 10u] * exp(power);
      let alpha = min(${fl(MAX_ALPHA)}, raw);
      if (alpha < ${fl(ALPHA_THRESHOLD)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b + 7u];
      accG = accG + w * derived[b + 8u];
      accB = accB + w * derived[b + 9u];
      T = T * (1.0 - alpha);
      if (T < ${fl(TRANSMITTANCE_CUTOFF)}) { break; }
    }
    let pix = y * ${uu(d.W)} + x;
    image[0u * ${uu(HW)} + pix] = accR + T * ${bgExpr(d, 0)};
    image[1u * ${uu(HW)} + pix] = accG + T * ${bgExpr(d, 1)};
    image[2u * ${uu(HW)} + pix] = accB + T * ${bgExpr(d, 2)};
  }
  atomicMax(&sh_maxstop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&sh_maxstop); }
}
`;
}

export function forwardBatchShader3D(cfg: Raster3DConfig): string {
  const d = resolveDims3D(cfg);
  const HW = d.H * d.W;
  const IMG = 3 * HW;
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;
${d.dynamicBg ? bgUniformDecl(5) : ""}

var<workgroup> sh_ids     : array<u32, ${d.cap}>;
var<workgroup> sh_maxstop : atomic<u32>;

fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u); v = v - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

fn derivedBase(lane : u32, g : u32) -> u32 {
  return lane * ${uu(d.G * DERIVED_STRIDE_3D)} + g * ${uu(DERIVED_STRIDE_3D)};
}

fn idGreater(lane : u32, a : u32, b : u32) -> bool {
  if (a == 0xffffffffu) { return b != 0xffffffffu; }
  if (b == 0xffffffffu) { return false; }
  let za = derived[derivedBase(lane, a) + 3u];
  let zb = derived[derivedBase(lane, b) + 3u];
  if (za == zb) { return a > b; }
  return za > zb;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  let lane = wg.z;
  if (tileId >= ${uu(d.numTiles)}) { return; }
  let tileCountsBase = lane * ${uu(d.numTiles)};
  let binnedBase = lane * ${uu(d.numTiles * d.cap)};
  let tileStopBase = lane * ${uu(d.numTiles)};
  let imageBase = lane * ${uu(IMG)};
  let count = min(tileCounts[tileCountsBase + tileId], ${uu(d.cap)});
  let start = binnedBase + tileId * ${uu(d.cap)};
  let sortN = nextPow2(count);

  for (var i = tid; i < sortN; i = i + 256u) {
    sh_ids[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  if (tid == 0u) { atomicStore(&sh_maxstop, 0u); }
  workgroupBarrier();

  var k = 2u;
  loop {
    if (k > sortN) { break; }
    var j = k >> 1u;
    loop {
      if (j == 0u) { break; }
      let nPairs = sortN >> 1u;
      for (var pair = tid; pair < nPairs; pair = pair + 256u) {
        let pos = 2u * j * (pair / j) + (pair % j);
        let ixj = pos + j;
        let asc = (pos & k) == 0u;
        let va = sh_ids[pos];
        let vb = sh_ids[ixj];
        let swapAsc = idGreater(lane, va, vb);
        let swapDesc = idGreater(lane, vb, va);
        if ((asc && swapAsc) || (!asc && swapDesc)) { sh_ids[pos] = vb; sh_ids[ixj] = va; }
      }
      workgroupBarrier();
      j = j >> 1u;
    }
    k = k << 1u;
  }

  for (var i = tid; i < count; i = i + 256u) { binnedIds[start + i] = sh_ids[i]; }
  workgroupBarrier();

  let tileX = tileId % ${uu(d.tilesX)};
  let tileY = tileId / ${uu(d.tilesX)};
  let x = tileX * ${TILE}u + (tid % ${TILE}u);
  let y = tileY * ${TILE}u + (tid / ${TILE}u);
  var localStop = 0u;
  if (x < ${uu(d.W)} && y < ${uu(d.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var accR = 0.0; var accG = 0.0; var accB = 0.0; var T = 1.0;
    for (var i = 0u; i < count; i = i + 1u) {
      let gg = sh_ids[i];
      let b = derivedBase(lane, gg);
      let dx = pxc - derived[b + 0u];
      let dy = pyc - derived[b + 1u];
      let invR2 = derived[b + 2u];
      let power = -0.5 * invR2 * (dx * dx + dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[b + 10u] * exp(power);
      let alpha = min(${fl(MAX_ALPHA)}, raw);
      if (alpha < ${fl(ALPHA_THRESHOLD)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b + 7u];
      accG = accG + w * derived[b + 8u];
      accB = accB + w * derived[b + 9u];
      T = T * (1.0 - alpha);
      if (T < ${fl(TRANSMITTANCE_CUTOFF)}) { break; }
    }
    let pix = y * ${uu(d.W)} + x;
    image[imageBase + 0u * ${uu(HW)} + pix] = accR + T * ${bgExpr(d, 0)};
    image[imageBase + 1u * ${uu(HW)} + pix] = accG + T * ${bgExpr(d, 1)};
    image[imageBase + 2u * ${uu(HW)} + pix] = accB + T * ${bgExpr(d, 2)};
  }
  atomicMax(&sh_maxstop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileStopBase + tileId] = atomicLoad(&sh_maxstop); }
}
`;
}

export function backwardShader3D(cfg: Raster3DConfig): string {
  const d = resolveDims3D(cfg);
  const HW = d.H * d.W;
  const SC = fl(d.gradScale);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;
${d.dynamicBg ? bgUniformDecl(6) : ""}

var<workgroup> sh_ids : array<u32, ${d.cap}>;

fn fixadd(base : u32, slot : u32, v : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(v * ${SC}), -2.14e9, 2.14e9)));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${uu(d.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${uu(d.cap)});
  let stopc = min(count, tileStop[tileId]);
  let start = tileId * ${uu(d.cap)};
  for (var i = tid; i < stopc; i = i + 256u) { sh_ids[i] = binnedIds[start + i]; }
  workgroupBarrier();

  let tileX = tileId % ${uu(d.tilesX)};
  let tileY = tileId / ${uu(d.tilesX)};
  let x = tileX * ${TILE}u + (tid % ${TILE}u);
  let y = tileY * ${TILE}u + (tid / ${TILE}u);
  if (x >= ${uu(d.W)} || y >= ${uu(d.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${uu(d.W)} + x;
  let goR = gradImage[0u * ${uu(HW)} + pix];
  let goG = gradImage[1u * ${uu(HW)} + pix];
  let goB = gradImage[2u * ${uu(HW)} + pix];

  var T = 1.0;
  var endi = stopc;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b = gg * ${uu(DERIVED_STRIDE_3D)};
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let power = -0.5 * derived[b + 2u] * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${fl(MAX_ALPHA)}, derived[b + 10u] * exp(power));
    if (alpha < ${fl(ALPHA_THRESHOLD)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${fl(TRANSMITTANCE_CUTOFF)}) { endi = i + 1u; break; }
  }

  var Tcur = T;
  var gT = goR * ${bgExpr(d, 0)} + goG * ${bgExpr(d, 1)} + goB * ${bgExpr(d, 2)};
  for (var ii = i32(endi) - 1; ii >= 0; ii = ii - 1) {
    let gg = sh_ids[u32(ii)];
    let b = gg * ${uu(DERIVED_STRIDE_3D)};
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let invR2 = derived[b + 2u];
    let power = -0.5 * invR2 * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let op = derived[b + 10u];
    let raw = op * exp(power);
    let alpha = min(${fl(MAX_ALPHA)}, raw);
    if (alpha < ${fl(ALPHA_THRESHOLD)}) { continue; }
    let denom = max(1.0 - alpha, ${fl(EPS)});
    let Tprev = Tcur / denom;
    let cR = derived[b + 7u]; let cG = derived[b + 8u]; let cB = derived[b + 9u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b, 7u, goR * Tprev * alpha);
    fixadd(b, 8u, goG * Tprev * alpha);
    fixadd(b, 9u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${fl(MAX_ALPHA)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-invR2 * dx);
    let gdy = gPower * (-invR2 * dy);
    fixadd(b, 0u, -gdx);
    fixadd(b, 1u, -gdy);
    fixadd(b, 2u, gPower * (-0.5) * (dx * dx + dy * dy));
    fixadd(b, 10u, gRaw * (raw / max(op, ${fl(EPS)})));

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`;
}

export function backwardBatchShader3D(cfg: Raster3DConfig): string {
  const d = resolveDims3D(cfg);
  const HW = d.H * d.W;
  const IMG = 3 * HW;
  const SC = fl(d.gradScale);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;
${d.dynamicBg ? bgUniformDecl(6) : ""}

var<workgroup> sh_ids : array<u32, ${d.cap}>;

fn derivedBase(lane : u32, g : u32) -> u32 {
  return lane * ${uu(d.G * DERIVED_STRIDE_3D)} + g * ${uu(DERIVED_STRIDE_3D)};
}

fn fixadd(base : u32, slot : u32, v : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(v * ${SC}), -2.14e9, 2.14e9)));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  let lane = wg.z;
  if (tileId >= ${uu(d.numTiles)}) { return; }
  let tileCountsBase = lane * ${uu(d.numTiles)};
  let binnedBase = lane * ${uu(d.numTiles * d.cap)};
  let tileStopBase = lane * ${uu(d.numTiles)};
  let gradImageBase = lane * ${uu(IMG)};
  let count = min(tileCounts[tileCountsBase + tileId], ${uu(d.cap)});
  let stopc = min(count, tileStop[tileStopBase + tileId]);
  let start = binnedBase + tileId * ${uu(d.cap)};
  for (var i = tid; i < stopc; i = i + 256u) { sh_ids[i] = binnedIds[start + i]; }
  workgroupBarrier();

  let tileX = tileId % ${uu(d.tilesX)};
  let tileY = tileId / ${uu(d.tilesX)};
  let x = tileX * ${TILE}u + (tid % ${TILE}u);
  let y = tileY * ${TILE}u + (tid / ${TILE}u);
  if (x >= ${uu(d.W)} || y >= ${uu(d.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${uu(d.W)} + x;
  let goR = gradImage[gradImageBase + 0u * ${uu(HW)} + pix];
  let goG = gradImage[gradImageBase + 1u * ${uu(HW)} + pix];
  let goB = gradImage[gradImageBase + 2u * ${uu(HW)} + pix];

  var T = 1.0;
  var endi = stopc;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b = derivedBase(lane, gg);
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let power = -0.5 * derived[b + 2u] * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${fl(MAX_ALPHA)}, derived[b + 10u] * exp(power));
    if (alpha < ${fl(ALPHA_THRESHOLD)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${fl(TRANSMITTANCE_CUTOFF)}) { endi = i + 1u; break; }
  }

  var Tcur = T;
  var gT = goR * ${bgExpr(d, 0)} + goG * ${bgExpr(d, 1)} + goB * ${bgExpr(d, 2)};
  for (var ii = i32(endi) - 1; ii >= 0; ii = ii - 1) {
    let gg = sh_ids[u32(ii)];
    let b = derivedBase(lane, gg);
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let invR2 = derived[b + 2u];
    let power = -0.5 * invR2 * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let op = derived[b + 10u];
    let raw = op * exp(power);
    let alpha = min(${fl(MAX_ALPHA)}, raw);
    if (alpha < ${fl(ALPHA_THRESHOLD)}) { continue; }
    let denom = max(1.0 - alpha, ${fl(EPS)});
    let Tprev = Tcur / denom;
    let cR = derived[b + 7u]; let cG = derived[b + 8u]; let cB = derived[b + 9u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b, 7u, goR * Tprev * alpha);
    fixadd(b, 8u, goG * Tprev * alpha);
    fixadd(b, 9u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${fl(MAX_ALPHA)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-invR2 * dx);
    let gdy = gPower * (-invR2 * dy);
    fixadd(b, 0u, -gdx);
    fixadd(b, 1u, -gdy);
    fixadd(b, 2u, gPower * (-0.5) * (dx * dx + dy * dy));
    fixadd(b, 10u, gRaw * (raw / max(op, ${fl(EPS)})));

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`;
}

export function chainAddShader3D(cfg: Raster3DConfig, cam: PreparedCamera3D): string {
  const d = resolveDims3D(cfg);
  const s = seg(d);
  const INV = fl(1.0 / d.gradScale);
  return /* wgsl */ `
${cameraBlock(cam)}
@group(0) @binding(0) var<storage, read>       accGrad : array<i32>;
@group(0) @binding(1) var<storage, read>       derived : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }
  let b = g * ${uu(DERIVED_STRIDE_3D)};
  let invScale = ${INV};
  let gsx = f32(accGrad[b + 0u]) * invScale;
  let gsy = f32(accGrad[b + 1u]) * invScale;
  let gInv = f32(accGrad[b + 2u]) * invScale;
  let gc0 = f32(accGrad[b + 7u]) * invScale;
  let gc1 = f32(accGrad[b + 8u]) * invScale;
  let gc2 = f32(accGrad[b + 9u]) * invScale;
  let gop = f32(accGrad[b + 10u]) * invScale;

  let vx = derived[b + 4u];
  let vy = derived[b + 5u];
  let vz = max(derived[b + 6u], ${fl(d.near)});
  let invR2 = derived[b + 2u];
  let invZ = 1.0 / vz;
  let invZ2 = invZ * invZ;
  let gvx = gsx * FOCAL_PX * invZ;
  let gvy = -gsy * FOCAL_PX * invZ;
  let gvz = gsx * (-FOCAL_PX * vx * invZ2) + gsy * (FOCAL_PX * vy * invZ2) + gInv * (2.0 * invR2 * invZ);
  let gp = CAM_RIGHT * gvx + CAM_UP * gvy + CAM_FWD * gvz;

  gradRaw[${uu(s.position)} + g * 3u + 0u] = gradRaw[${uu(s.position)} + g * 3u + 0u] + gp.x;
  gradRaw[${uu(s.position)} + g * 3u + 1u] = gradRaw[${uu(s.position)} + g * 3u + 1u] + gp.y;
  gradRaw[${uu(s.position)} + g * 3u + 2u] = gradRaw[${uu(s.position)} + g * 3u + 2u] + gp.z;

  let lr = params[${uu(s.logRadius)} + g];
  let er = exp(lr);
  let gateR = select(0.0, 1.0, er > ${fl(RADIUS_MIN)} && er < ${fl(RADIUS_MAX)});
  gradRaw[${uu(s.logRadius)} + g] = gradRaw[${uu(s.logRadius)} + g] + gInv * (-2.0 * invR2) * gateR;

  let col0 = derived[b + 7u]; let col1 = derived[b + 8u]; let col2 = derived[b + 9u];
  let opv = derived[b + 10u];
  gradRaw[${uu(s.colorRaw)} + g * 3u + 0u] = gradRaw[${uu(s.colorRaw)} + g * 3u + 0u] + gc0 * col0 * (1.0 - col0);
  gradRaw[${uu(s.colorRaw)} + g * 3u + 1u] = gradRaw[${uu(s.colorRaw)} + g * 3u + 1u] + gc1 * col1 * (1.0 - col1);
  gradRaw[${uu(s.colorRaw)} + g * 3u + 2u] = gradRaw[${uu(s.colorRaw)} + g * 3u + 2u] + gc2 * col2 * (1.0 - col2);
  gradRaw[${uu(s.opacityRaw)} + g] = gradRaw[${uu(s.opacityRaw)} + g] + gop * opv * (1.0 - opv);
}
`;
}

export function clearShader3D(n: number): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= ${uu(n)}) { return; }
  buf[gid.x] = 0u;
}
`;
}

export const REGULARIZER_UNIFORM_BYTES_3D = 32;

export function regularizerShader3D(cfg: Raster3DConfig): string {
  const d = resolveDims3D(cfg);
  const s = seg(d);
  return /* wgsl */ `
${SIGMOID}
struct RegU {
  centerWeight   : f32,
  radiusWeight   : f32,
  targetRadius   : f32,
  opacitySparsity: f32,
  _pad0          : f32,
  _pad1          : f32,
  _pad2          : f32,
  _pad3          : f32,
};
@group(0) @binding(0) var<uniform>             u       : RegU;
@group(0) @binding(1) var<storage, read>       params  : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }

  let pxIdx = ${uu(s.position)} + g * 3u + 0u;
  let pyIdx = ${uu(s.position)} + g * 3u + 1u;
  let pzIdx = ${uu(s.position)} + g * 3u + 2u;
  let p = vec3f(params[pxIdx], params[pyIdx], params[pzIdx]);
  let r = length(p);
  let invR = 1.0 / max(r, ${fl(EPS)});
  let outside = max(0.0, r - max(u.targetRadius, ${fl(EPS)}));
  let gp = (2.0 * u.centerWeight) * p + (2.0 * u.radiusWeight * outside * invR) * p;
  gradRaw[pxIdx] = gradRaw[pxIdx] + gp.x;
  gradRaw[pyIdx] = gradRaw[pyIdx] + gp.y;
  gradRaw[pzIdx] = gradRaw[pzIdx] + gp.z;

  let opIdx = ${uu(s.opacityRaw)} + g;
  let op = sigmoid1(params[opIdx]);
  gradRaw[opIdx] = gradRaw[opIdx] + u.opacitySparsity * op * (1.0 - op);
}
`;
}

export function paramSegments3D(G: number): { name: string; offset: number; length: number }[] {
  return [
    { name: "position", offset: 0, length: 3 * G },
    { name: "logRadius", offset: 3 * G, length: G },
    { name: "color", offset: 4 * G, length: 3 * G },
    { name: "opacity", offset: 7 * G, length: G },
  ];
}
