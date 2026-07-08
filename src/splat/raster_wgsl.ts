/**
 * raster_wgsl — PURE WGSL codegen for the 2D Gaussian-splat rasterizer
 * (forward + backward) with the reparameterized OUR data model.
 *
 * Zero imports on purpose (mirrors src/clip/vision_wgsl.ts): every shape /
 * threshold / offset is baked into the shader source, so there are no uniforms
 * in the raster kernels (Adam gets a tiny uniform for per-step hyperparams).
 * Each returned string is one self-contained compute module, testable headless
 * under bun-webgpu (tools/splat/raster_test.ts) against a float64 JS reference.
 *
 * ---------------------------------------------------------------------------
 * DATA MODEL (ours — see docs/splat_raster_spec.md; differs from the Metal
 * reference which trains the conic directly). Adam updates the RAW params:
 *
 *   params[G*9] SoA segments (one GPUBuffer):
 *     mean       [0,   2G)   g*2 + {0,1}      px
 *     logScale   [2G,  4G)   2G + g*2 + {0,1}
 *     theta      [4G,  5G)   4G + g
 *     colorRaw   [5G,  8G)   5G + g*3 + {0,1,2}
 *     opacityRaw [8G,  9G)   8G + g
 *   gradRaw, m, v share this SoA layout.
 *
 * Reparameterization (computed in the `prep` kernel; its Jacobian is applied
 * once per splat in the `chain` kernel — NOT inside the per-pixel backward, so
 * the per-pixel backward stays byte-for-byte the reference recurrence and the
 * Jacobian is unit-testable in isolation):
 *     scale   = clamp(exp(logScale), 0.3, 64)      px
 *     ix,iy   = 1/scale.x^2 , 1/scale.y^2
 *     conic a = cos^2 ix + sin^2 iy
 *           b = cos sin (ix - iy)
 *           c = sin^2 ix + cos^2 iy            (inverse of R diag(s^2) R^T)
 *     color   = sigmoid(colorRaw)
 *     opacity = sigmoid(opacityRaw)
 *
 *   derived[G*9] AoS stride 9 (one GPUBuffer, produced by `prep`, consumed by
 *   every raster kernel so binding count stays <=8/stage):
 *     g*9 + {0=mx,1=my, 2=a,3=b,4=c, 5=cR,6=cG,7=cB, 8=opacity}
 *
 * Gradient accumulation is fixed-point atomicAdd<i32> into `accGrad[G*9]`
 * (AoS parallel to `derived`: {mx,my, a,b,c, cR,cG,cB, op}) with a documented
 * scale GRAD_SCALE. WGSL has no f32 atomicAdd; i32 fixed-point is the simplest
 * correct scheme and the scale nearly cancels in Adam (m/sqrt(v) is scale-
 * invariant). `chain` divides by GRAD_SCALE when it reads accGrad. Overflow is
 * guarded by clamping the fixed-point value into i32 range.
 *
 * Output image: NCHW planar f32 [3][H][W] in ~[0,1], out[ch*H*W + y*W + x]
 * (binds directly as the CLIP encoder input slot later).
 * ---------------------------------------------------------------------------
 */

// Algorithm thresholds (baked as literals — fixed by the alpha/visibility math,
// matching the Metal reference gsplat_fast_kernels.metal / v11 fixedbin).
export const TILE = 16; // 16x16 tile == 256 pixels == 256 threads/workgroup
export const ALPHA_THRESHOLD = 1.0 / 255.0; // visible iff alpha >= 1/255
export const MAX_ALPHA = 0.99; // alpha = min(0.99, raw_alpha)
export const TRANSMITTANCE_CUTOFF = 1e-4; // early-out when T < 1e-4
export const EPS = 1e-8;
export const SCALE_MIN = 0.3;
export const SCALE_MAX = 64.0;

/** Per-splat stride of the AoS `derived` / `accGrad` buffers. */
export const DERIVED_STRIDE = 9;
/** Per-splat float count of the SoA `params` / `gradRaw` / `m` / `v` buffers. */
export const PARAM_STRIDE = 9;

export interface RasterConfig {
  H: number;
  W: number;
  G: number;
  /** per-tile capacity of the fixed-stride bins; MUST be a power of two. */
  cap: number;
  /** background colour composited as accum + T*bg (default 0.5 gray). */
  bg?: [number, number, number];
  /** fixed-point scale for the i32 gradient accumulator. */
  gradScale?: number;
}

export interface RasterDims {
  H: number;
  W: number;
  G: number;
  cap: number;
  tilesX: number;
  tilesY: number;
  numTiles: number;
  bg: [number, number, number];
  gradScale: number;
}

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`raster_wgsl: ${msg}`);
}

/** WGSL f32 literal — always has a '.' or exponent so it is not parsed as int. */
function fl(x: number): string {
  assert(Number.isFinite(x), `non-finite literal ${x}`);
  let s = x.toString();
  if (!/[.eE]/.test(s)) s += ".0";
  return s;
}
const uu = (x: number): string => `${x >>> 0}u`;

export function resolveDims(cfg: RasterConfig): RasterDims {
  assert(cfg.H > 0 && cfg.W > 0 && cfg.G > 0, "H,W,G must be positive");
  assert(cfg.H % TILE === 0 && cfg.W % TILE === 0, `H,W must be multiples of ${TILE}`);
  assert((cfg.cap & (cfg.cap - 1)) === 0 && cfg.cap > 0, "cap must be a power of two");
  assert(cfg.cap * 4 <= 16384, `cap*4 (${cfg.cap * 4}B) exceeds 16KB workgroup storage`);
  const tilesX = cfg.W / TILE;
  const tilesY = cfg.H / TILE;
  return {
    H: cfg.H,
    W: cfg.W,
    G: cfg.G,
    cap: cfg.cap,
    tilesX,
    tilesY,
    numTiles: tilesX * tilesY,
    bg: cfg.bg ?? [0.5, 0.5, 0.5],
    gradScale: cfg.gradScale ?? 65536,
  };
}

// ---------------------------------------------------------------------------
// Shared WGSL fragments (inlined per kernel — each module stays standalone)
// ---------------------------------------------------------------------------

/** Segment base offsets into the SoA params/gradRaw/m/v buffers. */
function seg(d: RasterDims) {
  return {
    mean: 0,
    logScale: 2 * d.G,
    theta: 4 * d.G,
    colorRaw: 5 * d.G,
    opacityRaw: 8 * d.G,
  };
}

const SIGMOID = /* wgsl */ `fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }`;

// ---------------------------------------------------------------------------
// 1) prep — thread/splat: raw params -> derived (mean, conic, color, opacity).
//    The single place the reparameterization forward is computed.
// ---------------------------------------------------------------------------
export function prepShader(cfg: RasterConfig): string {
  const d = resolveDims(cfg);
  const s = seg(d);
  return /* wgsl */ `
${SIGMOID}
@group(0) @binding(0) var<storage, read>       params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }
  let mx  = params[${uu(s.mean)} + g * 2u + 0u];
  let my  = params[${uu(s.mean)} + g * 2u + 1u];
  let lsx = params[${uu(s.logScale)} + g * 2u + 0u];
  let lsy = params[${uu(s.logScale)} + g * 2u + 1u];
  let th  = params[${uu(s.theta)} + g];
  let cr0 = params[${uu(s.colorRaw)} + g * 3u + 0u];
  let cr1 = params[${uu(s.colorRaw)} + g * 3u + 1u];
  let cr2 = params[${uu(s.colorRaw)} + g * 3u + 2u];
  let opr = params[${uu(s.opacityRaw)} + g];

  let sx = clamp(exp(lsx), ${fl(SCALE_MIN)}, ${fl(SCALE_MAX)});
  let sy = clamp(exp(lsy), ${fl(SCALE_MIN)}, ${fl(SCALE_MAX)});
  let ix = 1.0 / (sx * sx);
  let iy = 1.0 / (sy * sy);
  let cs = cos(th);
  let sn = sin(th);

  let base = g * ${uu(DERIVED_STRIDE)};
  derived[base + 0u] = mx;
  derived[base + 1u] = my;
  derived[base + 2u] = cs * cs * ix + sn * sn * iy;           // conic a
  derived[base + 3u] = cs * sn * (ix - iy);                   // conic b
  derived[base + 4u] = sn * sn * ix + cs * cs * iy;           // conic c
  derived[base + 5u] = sigmoid1(cr0);
  derived[base + 6u] = sigmoid1(cr1);
  derived[base + 7u] = sigmoid1(cr2);
  derived[base + 8u] = sigmoid1(opr);
}
`;
}

// ---------------------------------------------------------------------------
// 2) emit — thread/splat: fixedbin binning (v11 style, no prefix sum, no CPU
//    readback). Atomic cursor per tile into constant-stride bins tile*cap.
//    Merges count+emit: tileCounts is the cursor (cleared each step); a splat
//    whose slot >= cap is dropped (graceful overflow). The forward re-sorts by
//    index so the emit order is irrelevant to the result (deterministic).
// ---------------------------------------------------------------------------
export function emitShader(cfg: RasterConfig): string {
  const d = resolveDims(cfg);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       derived    : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds  : array<u32>;

// EXACT ellipse-vs-rect test (Metal reference ellipse_intersects_rect): the
// tile intersects the alpha-support ellipse {q(d) <= tau} iff the min of the
// quadratic form over the rect is <= tau. Checks corners + edge extrema.
fn ellipse_hit(mx : f32, my : f32, a : f32, b : f32, c : f32, tau : f32,
               rx0 : f32, ry0 : f32, rx1 : f32, ry1 : f32) -> bool {
  let dx0 = rx0 - mx; let dx1 = rx1 - mx;
  let dy0 = ry0 - my; let dy1 = ry1 - my;
  if (mx >= rx0 && mx <= rx1 && my >= ry0 && my <= ry1) { return true; }
  var qmin = 1e30;
  qmin = min(qmin, a * dx0 * dx0 + 2.0 * b * dx0 * dy0 + c * dy0 * dy0);
  qmin = min(qmin, a * dx0 * dx0 + 2.0 * b * dx0 * dy1 + c * dy1 * dy1);
  qmin = min(qmin, a * dx1 * dx1 + 2.0 * b * dx1 * dy0 + c * dy0 * dy0);
  qmin = min(qmin, a * dx1 * dx1 + 2.0 * b * dx1 * dy1 + c * dy1 * dy1);
  if (c > 1e-8) {
    var dy = clamp(-(b / c) * dx0, dy0, dy1);
    qmin = min(qmin, a * dx0 * dx0 + 2.0 * b * dx0 * dy + c * dy * dy);
    dy = clamp(-(b / c) * dx1, dy0, dy1);
    qmin = min(qmin, a * dx1 * dx1 + 2.0 * b * dx1 * dy + c * dy * dy);
  }
  if (a > 1e-8) {
    var dx = clamp(-(b / a) * dy0, dx0, dx1);
    qmin = min(qmin, a * dx * dx + 2.0 * b * dx * dy0 + c * dy0 * dy0);
    dx = clamp(-(b / a) * dy1, dx0, dx1);
    qmin = min(qmin, a * dx * dx + 2.0 * b * dx * dy1 + c * dy1 * dy1);
  }
  return qmin <= tau;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }
  let base = g * ${uu(DERIVED_STRIDE)};
  let op = derived[base + 8u];
  if (op <= ${fl(ALPHA_THRESHOLD)}) { return; }
  let ratio = max(${fl(ALPHA_THRESHOLD)} / max(op, ${fl(EPS)}), ${fl(EPS)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let mx = derived[base + 0u]; let my = derived[base + 1u];
  let a  = derived[base + 2u]; let b  = derived[base + 3u]; let c = derived[base + 4u];
  let det = max(a * c - b * b, ${fl(EPS)});
  let hx = sqrt(max(tau * c / det, 0.0));
  let hy = sqrt(max(tau * a / det, 0.0));
  let x0 = max(0, i32(floor(mx - hx - 0.5)));
  let x1 = min(${d.W - 1}, i32(ceil(mx + hx - 0.5)));
  let y0 = max(0, i32(floor(my - hy - 0.5)));
  let y1 = min(${d.H - 1}, i32(ceil(my + hy - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tx0 = x0 / ${TILE}; let tx1 = x1 / ${TILE};
  let ty0 = y0 / ${TILE}; let ty1 = y1 / ${TILE};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    let ry0 = f32(ty * ${TILE}) + 0.5;
    let ry1 = min(f32(${d.H - 1}) + 0.5, f32((ty + 1) * ${TILE} - 1) + 0.5);
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let rx0 = f32(tx * ${TILE}) + 0.5;
      let rx1 = min(f32(${d.W - 1}) + 0.5, f32((tx + 1) * ${TILE} - 1) + 0.5);
      if (ellipse_hit(mx, my, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        let tile = u32(ty * ${d.tilesX} + tx);
        let slot = atomicAdd(&tileCounts[tile], 1u);
        if (slot < ${uu(d.cap)}) { binnedIds[tile * ${uu(d.cap)} + slot] = g; }
      }
    }
  }
}
`;
}

// ---------------------------------------------------------------------------
// 3) forward — 1 workgroup(256)/tile, one thread per pixel. Stage tile ids in
//    shared, bitonic-sort ASCENDING (recovers painter order == splat index
//    order; there is no depth), write the sorted ids back so the backward can
//    skip re-sorting, front-to-back composite with early-out, save
//    tileStop = max visible-prefix length (bounds the backward replay).
// ---------------------------------------------------------------------------
export function forwardShader(cfg: RasterConfig): string {
  const d = resolveDims(cfg);
  const HW = d.H * d.W;
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;  // NCHW planar
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;

var<workgroup> sh_ids     : array<u32, ${d.cap}>;
var<workgroup> sh_maxstop : atomic<u32>;

fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u); v = v - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${uu(d.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${uu(d.cap)});
  let start = tileId * ${uu(d.cap)};
  let sortN = nextPow2(count);

  // stage ids + pad to power of two with sentinel 0xffffffff (sorts to the end)
  for (var i = tid; i < sortN; i = i + 256u) {
    sh_ids[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  if (tid == 0u) { atomicStore(&sh_maxstop, 0u); }
  workgroupBarrier();

  // bitonic sort ascending — 256-thread strided variant (v11 shape)
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
        if ((va > vb) == asc) { sh_ids[pos] = vb; sh_ids[ixj] = va; }
      }
      workgroupBarrier();
      j = j >> 1u;
    }
    k = k << 1u;
  }

  // write sorted ids back (backward reuses them without re-sorting)
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
      let b3 = gg * ${uu(DERIVED_STRIDE)};
      let dx = pxc - derived[b3 + 0u];
      let dy = pyc - derived[b3 + 1u];
      let a  = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
      let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[b3 + 8u] * exp(power);
      let alpha = min(${fl(MAX_ALPHA)}, raw);
      if (alpha < ${fl(ALPHA_THRESHOLD)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b3 + 5u];
      accG = accG + w * derived[b3 + 6u];
      accB = accB + w * derived[b3 + 7u];
      T = T * (1.0 - alpha);
      if (T < ${fl(TRANSMITTANCE_CUTOFF)}) { break; }
    }
    let pix = y * ${uu(d.W)} + x;
    image[0u * ${uu(HW)} + pix] = accR + T * ${fl(d.bg[0])};
    image[1u * ${uu(HW)} + pix] = accG + T * ${fl(d.bg[1])};
    image[2u * ${uu(HW)} + pix] = accB + T * ${fl(d.bg[2])};
  }
  atomicMax(&sh_maxstop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&sh_maxstop); }
}
`;
}

// ---------------------------------------------------------------------------
// 4) backward — 1 workgroup(256)/tile, one thread per pixel. Replays the
//    visible prefix (bounded by tileStop) to recover T_final and end_i, then
//    walks BACK-TO-FRONT reconstructing per-splat grads with T_prev = T_cur/
//    (1-alpha). Accumulates DERIVED-space grads (mean, conic, color, opacity)
//    into accGrad via fixed-point atomicAdd<i32> — byte-for-byte the Metal
//    reference recurrence. NO barriers in the per-pixel loop, so the uniformity
//    rule is satisfied trivially (each pixel's end_i gates only its own loop).
// ---------------------------------------------------------------------------
export function backwardShader(cfg: RasterConfig): string {
  const d = resolveDims(cfg);
  const HW = d.H * d.W;
  const SC = fl(d.gradScale);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;  // NCHW planar
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;

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
  workgroupBarrier();  // only barrier; everything below is per-pixel (uniformity safe)

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

  // phase A: replay to recover T_final and the stop index end_i
  var T = 1.0;
  var endi = stopc;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b3 = gg * ${uu(DERIVED_STRIDE)};
    let dx = pxc - derived[b3 + 0u];
    let dy = pyc - derived[b3 + 1u];
    let a = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${fl(MAX_ALPHA)}, derived[b3 + 8u] * exp(power));
    if (alpha < ${fl(ALPHA_THRESHOLD)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${fl(TRANSMITTANCE_CUTOFF)}) { endi = i + 1u; break; }
  }

  // phase B: back-to-front recurrence
  var Tcur = T;
  var gT = goR * ${fl(d.bg[0])} + goG * ${fl(d.bg[1])} + goB * ${fl(d.bg[2])};
  for (var ii = i32(endi) - 1; ii >= 0; ii = ii - 1) {
    let gg = sh_ids[u32(ii)];
    let b3 = gg * ${uu(DERIVED_STRIDE)};
    let dx = pxc - derived[b3 + 0u];
    let dy = pyc - derived[b3 + 1u];
    let a = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let op = derived[b3 + 8u];
    let raw = op * exp(power);
    let alpha = min(${fl(MAX_ALPHA)}, raw);
    if (alpha < ${fl(ALPHA_THRESHOLD)}) { continue; }
    let denom = max(1.0 - alpha, ${fl(EPS)});
    let Tprev = Tcur / denom;
    let cR = derived[b3 + 5u]; let cG = derived[b3 + 6u]; let cB = derived[b3 + 7u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b3, 5u, goR * Tprev * alpha);
    fixadd(b3, 6u, goG * Tprev * alpha);
    fixadd(b3, 7u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${fl(MAX_ALPHA)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-(a * dx + b * dy));
    let gdy = gPower * (-(b * dx + c * dy));
    fixadd(b3, 2u, gPower * (-0.5) * dx * dx);   // g_a
    fixadd(b3, 3u, gPower * (-1.0) * dx * dy);   // g_b
    fixadd(b3, 4u, gPower * (-0.5) * dy * dy);   // g_c
    fixadd(b3, 0u, -gdx);                        // g_mean.x
    fixadd(b3, 1u, -gdy);                        // g_mean.y
    fixadd(b3, 8u, gRaw * (raw / max(op, ${fl(EPS)})));  // g_opacity

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`;
}

// ---------------------------------------------------------------------------
// 5) chain — thread/splat: convert DERIVED-space grads (accGrad, i32 fixed-
//    point) to RAW-space grads (gradRaw, f32 SoA). This is the reparam
//    Jacobian, applied ONCE per splat. Verified against a float64 JS reference
//    on a single splat before the full gradcheck (docs derivation-care note).
//
//    conic (a,b,c)(ix,iy,theta):
//      g_ix = g_a cos^2 + g_b cos sin + g_c sin^2
//      g_iy = g_a sin^2 - g_b cos sin + g_c cos^2
//      g_theta = (ix-iy) [ (cos^2-sin^2) g_b + 2 cos sin (g_c - g_a) ]
//    ix = 1/scale.x^2, scale.x = clamp(exp(lsx)):  dix/dlsx = -2 ix  (unclamped)
//      g_lsx = g_ix * (-2 ix) * gateX
//    sigmoid reparams: g_colorRaw = g_color color(1-color); same for opacity.
//    mean has no reparam (passes through).
// ---------------------------------------------------------------------------
export function chainShader(cfg: RasterConfig): string {
  const d = resolveDims(cfg);
  const s = seg(d);
  const INV = fl(1.0 / d.gradScale);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read>       accGrad : array<i32>;   // fixed-point
@group(0) @binding(1) var<storage, read>       derived : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${uu(d.G)}) { return; }
  let b3 = g * ${uu(DERIVED_STRIDE)};
  let inv = ${INV};
  let gmx = f32(accGrad[b3 + 0u]) * inv;
  let gmy = f32(accGrad[b3 + 1u]) * inv;
  let gA  = f32(accGrad[b3 + 2u]) * inv;
  let gB  = f32(accGrad[b3 + 3u]) * inv;
  let gC  = f32(accGrad[b3 + 4u]) * inv;
  let gc0 = f32(accGrad[b3 + 5u]) * inv;
  let gc1 = f32(accGrad[b3 + 6u]) * inv;
  let gc2 = f32(accGrad[b3 + 7u]) * inv;
  let gop = f32(accGrad[b3 + 8u]) * inv;

  let lsx = params[${uu(s.logScale)} + g * 2u + 0u];
  let lsy = params[${uu(s.logScale)} + g * 2u + 1u];
  let th  = params[${uu(s.theta)} + g];
  let ex = exp(lsx); let ey = exp(lsy);
  let sx = clamp(ex, ${fl(SCALE_MIN)}, ${fl(SCALE_MAX)});
  let sy = clamp(ey, ${fl(SCALE_MIN)}, ${fl(SCALE_MAX)});
  let gateX = select(0.0, 1.0, ex > ${fl(SCALE_MIN)} && ex < ${fl(SCALE_MAX)});
  let gateY = select(0.0, 1.0, ey > ${fl(SCALE_MIN)} && ey < ${fl(SCALE_MAX)});
  let ix = 1.0 / (sx * sx);
  let iy = 1.0 / (sy * sy);
  let cs = cos(th); let sn = sin(th);

  let gix = gA * cs * cs + gB * cs * sn + gC * sn * sn;
  let giy = gA * sn * sn - gB * cs * sn + gC * cs * cs;
  let glsx = gix * (-2.0 * ix) * gateX;
  let glsy = giy * (-2.0 * iy) * gateY;
  let D = ix - iy;
  let gth = D * ((cs * cs - sn * sn) * gB + 2.0 * cs * sn * (gC - gA));

  let col0 = derived[b3 + 5u]; let col1 = derived[b3 + 6u]; let col2 = derived[b3 + 7u];
  let opv  = derived[b3 + 8u];

  gradRaw[${uu(s.mean)} + g * 2u + 0u] = gmx;
  gradRaw[${uu(s.mean)} + g * 2u + 1u] = gmy;
  gradRaw[${uu(s.logScale)} + g * 2u + 0u] = glsx;
  gradRaw[${uu(s.logScale)} + g * 2u + 1u] = glsy;
  gradRaw[${uu(s.theta)} + g] = gth;
  gradRaw[${uu(s.colorRaw)} + g * 3u + 0u] = gc0 * col0 * (1.0 - col0);
  gradRaw[${uu(s.colorRaw)} + g * 3u + 1u] = gc1 * col1 * (1.0 - col1);
  gradRaw[${uu(s.colorRaw)} + g * 3u + 2u] = gc2 * col2 * (1.0 - col2);
  gradRaw[${uu(s.opacityRaw)} + g] = gop * opv * (1.0 - opv);
}
`;
}

// ---------------------------------------------------------------------------
// 6) clear — thread/element: zero a storage buffer viewed as array<u32>
//    (works for the i32 accGrad and the u32 tileCounts; 0 bits == 0 either way).
// ---------------------------------------------------------------------------
export function clearShader(n: number): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= ${uu(n)}) { return; }
  buf[gid.x] = 0u;
}
`;
}

/** Segment offsets for the Adam driver (matches seg()). */
export function paramSegments(G: number): { name: string; offset: number; length: number }[] {
  return [
    { name: "mean", offset: 0, length: 2 * G },
    { name: "logScale", offset: 2 * G, length: 2 * G },
    { name: "theta", offset: 4 * G, length: G },
    { name: "color", offset: 5 * G, length: 3 * G },
    { name: "opacity", offset: 8 * G, length: G },
  ];
}
