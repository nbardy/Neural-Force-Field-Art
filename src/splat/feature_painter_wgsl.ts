/**
 * Specialized fused Feature8 splat shaders.
 *
 * The old experiment materialized a 32-channel feature image and a 32-channel
 * image gradient. This path retains the RGB raster as an exact skip baseline,
 * carries only five latent channels, and evaluates the residual decoder in the
 * same tile passes. See docs/FEATURE_PAINTER_FUSED_DECISION.md.
 */
import {
  ALPHA_THRESHOLD,
  EPS,
  MAX_ALPHA,
  SCALE_MAX,
  SCALE_MIN,
  TILE,
  TRANSMITTANCE_CUTOFF,
  type RasterConfig,
  resolveDims,
} from "./raster_wgsl";

/** Extra per-splat values: z[5], local-x coefficients[5], local-y coefficients[5]. */
export const FEATURE_LATENT_CHANNELS = 5;
export const FEATURE_STRIDE = FEATURE_LATENT_CHANNELS * 3;
/** Composited feature vector: RGB skip values plus five latent values. */
export const FEATURE_DIM = 3 + FEATURE_LATENT_CHANNELS;
/** 3x8 decoder matrix plus 3 output biases. */
export const DECODER_PARAM_COUNT = FEATURE_DIM * 3 + 3;
export const DECODER_RESIDUAL_SCALE = 0.1;

/** Derived RGB-raster state plus an exact local coordinate frame. */
export const FEATURE_STATE_STRIDE = 15;
const STATE_MEAN_X = 0;
const STATE_MEAN_Y = 1;
const STATE_CONIC_A = 2;
const STATE_CONIC_B = 3;
const STATE_CONIC_C = 4;
const STATE_RGB_R = 5;
const STATE_RGB_G = 6;
const STATE_RGB_B = 7;
const STATE_OPACITY = 8;
const STATE_COS = 9;
const STATE_SIN = 10;
const STATE_INV_SX = 11;
const STATE_INV_SY = 12;
const STATE_SCALE_GATE_X = 13;
const STATE_SCALE_GATE_Y = 14;

/** One packed fixed-point accumulator for all splat and decoder gradients. */
export const FEATURE_ACC_DERIVED_OFFSET = 0;
export const FEATURE_ACC_EXTRA_OFFSET = 9;
export const FEATURE_ACC_LOCAL_RAW_OFFSET = FEATURE_ACC_EXTRA_OFFSET + FEATURE_STRIDE;
export const FEATURE_ACC_STRIDE = FEATURE_ACC_LOCAL_RAW_OFFSET + 5;

const f = (x: number): string => /[.eE]/.test(String(x)) ? String(x) : `${x}.0`;
const u = (x: number): string => `${x >>> 0}u`;

function common(cfg: RasterConfig) {
  const d = resolveDims(cfg);
  const decoderOffset = d.G * FEATURE_ACC_STRIDE;
  return {
    d,
    hw: d.H * d.W,
    decoderOffset,
    code: /* wgsl */ `
const STATE_STRIDE : u32 = ${u(FEATURE_STATE_STRIDE)};
const FEATURE_STRIDE : u32 = ${u(FEATURE_STRIDE)};
const FEATURE_DIM : u32 = ${u(FEATURE_DIM)};
const ACC_EXTRA_OFFSET : u32 = ${u(FEATURE_ACC_EXTRA_OFFSET)};
const ACC_LOCAL_RAW_OFFSET : u32 = ${u(FEATURE_ACC_LOCAL_RAW_OFFSET)};
const ACC_STRIDE : u32 = ${u(FEATURE_ACC_STRIDE)};
const DECODER_OFFSET : u32 = ${u(decoderOffset)};
const RESIDUAL_SCALE : f32 = ${f(DECODER_RESIDUAL_SCALE)};

fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn logit1(x : f32) -> f32 { let y = clamp(x, ${f(EPS)}, ${f(1 - EPS)}); return log(y / (1.0 - y)); }
fn fixadd(dst : ptr<storage, array<atomic<i32>>, read_write>, index : u32, v : f32) {
  atomicAdd(&(*dst)[index], i32(clamp(round(v * ${f(d.gradScale)}), -2.14e9, 2.14e9)));
}
`,
  };
}

/** Raw splat parameters -> RGB derived state plus local-frame coefficients. */
export function featurePrepShader(cfg: RasterConfig): string {
  const { d, code } = common(cfg);
  const mean = 0;
  const logScale = 2 * d.G;
  const theta = 4 * d.G;
  const colorRaw = 5 * d.G;
  const opacityRaw = 8 * d.G;
  return /* wgsl */ `
${code}
@group(0) @binding(0) var<storage, read> params : array<f32>;
@group(0) @binding(1) var<storage, read_write> state : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${u(d.G)}) { return; }
  let mx = params[${u(mean)} + g * 2u];
  let my = params[${u(mean)} + g * 2u + 1u];
  let lsx = params[${u(logScale)} + g * 2u];
  let lsy = params[${u(logScale)} + g * 2u + 1u];
  let th = params[${u(theta)} + g];
  let ex = exp(lsx);
  let ey = exp(lsy);
  let sx = clamp(ex, ${f(SCALE_MIN)}, ${f(SCALE_MAX)});
  let sy = clamp(ey, ${f(SCALE_MIN)}, ${f(SCALE_MAX)});
  let invSx = 1.0 / sx;
  let invSy = 1.0 / sy;
  let ix = invSx * invSx;
  let iy = invSy * invSy;
  let cs = cos(th);
  let sn = sin(th);
  let b = g * STATE_STRIDE;
  state[b + ${u(STATE_MEAN_X)}] = mx;
  state[b + ${u(STATE_MEAN_Y)}] = my;
  state[b + ${u(STATE_CONIC_A)}] = cs * cs * ix + sn * sn * iy;
  state[b + ${u(STATE_CONIC_B)}] = cs * sn * (ix - iy);
  state[b + ${u(STATE_CONIC_C)}] = sn * sn * ix + cs * cs * iy;
  state[b + ${u(STATE_RGB_R)}] = sigmoid1(params[${u(colorRaw)} + g * 3u]);
  state[b + ${u(STATE_RGB_G)}] = sigmoid1(params[${u(colorRaw)} + g * 3u + 1u]);
  state[b + ${u(STATE_RGB_B)}] = sigmoid1(params[${u(colorRaw)} + g * 3u + 2u]);
  state[b + ${u(STATE_OPACITY)}] = sigmoid1(params[${u(opacityRaw)} + g]);
  state[b + ${u(STATE_COS)}] = cs;
  state[b + ${u(STATE_SIN)}] = sn;
  state[b + ${u(STATE_INV_SX)}] = invSx;
  state[b + ${u(STATE_INV_SY)}] = invSy;
  state[b + ${u(STATE_SCALE_GATE_X)}] = select(0.0, 1.0, ex > ${f(SCALE_MIN)} && ex < ${f(SCALE_MAX)});
  state[b + ${u(STATE_SCALE_GATE_Y)}] = select(0.0, 1.0, ey > ${f(SCALE_MIN)} && ey < ${f(SCALE_MAX)});
}
`;
}

/** Fixed-bin emitter matching the RGB raster, with the wider Feature8 state. */
export function featureEmitShader(cfg: RasterConfig): string {
  const { d, code } = common(cfg);
  return /* wgsl */ `
${code}
@group(0) @binding(0) var<storage, read> state : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds : array<u32>;

fn ellipseHit(mx : f32, my : f32, a : f32, b : f32, c : f32, tau : f32,
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
  if (g >= ${u(d.G)}) { return; }
  let s = g * STATE_STRIDE;
  let opacity = state[s + ${u(STATE_OPACITY)}];
  if (opacity <= ${f(ALPHA_THRESHOLD)}) { return; }
  let ratio = max(${f(ALPHA_THRESHOLD)} / max(opacity, ${f(EPS)}), ${f(EPS)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }
  let mx = state[s + ${u(STATE_MEAN_X)}];
  let my = state[s + ${u(STATE_MEAN_Y)}];
  let a = state[s + ${u(STATE_CONIC_A)}];
  let b = state[s + ${u(STATE_CONIC_B)}];
  let c = state[s + ${u(STATE_CONIC_C)}];
  let det = max(a * c - b * b, ${f(EPS)});
  let hx = sqrt(max(tau * c / det, 0.0));
  let hy = sqrt(max(tau * a / det, 0.0));
  let x0 = max(0, i32(floor(mx - hx - 0.5)));
  let x1 = min(${d.W - 1}, i32(ceil(mx + hx - 0.5)));
  let y0 = max(0, i32(floor(my - hy - 0.5)));
  let y1 = min(${d.H - 1}, i32(ceil(my + hy - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }
  let tx0 = x0 / ${TILE}; let tx1 = x1 / ${TILE};
  let ty0 = y0 / ${TILE}; let ty1 = y1 / ${TILE};
  for (var ty = ty0; ty <= ty1; ty++) {
    let ry0 = f32(ty * ${TILE}) + 0.5;
    let ry1 = min(f32(${d.H - 1}) + 0.5, f32((ty + 1) * ${TILE} - 1) + 0.5);
    for (var tx = tx0; tx <= tx1; tx++) {
      let rx0 = f32(tx * ${TILE}) + 0.5;
      let rx1 = min(f32(${d.W - 1}) + 0.5, f32((tx + 1) * ${TILE} - 1) + 0.5);
      if (ellipseHit(mx, my, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        let tile = u32(ty * ${d.tilesX} + tx);
        let slot = atomicAdd(&tileCounts[tile], 1u);
        if (slot < ${u(d.cap)}) { binnedIds[tile * ${u(d.cap)} + slot] = g; }
      }
    }
  }
}
`;
}

/** Alpha-composite Feature8 and decode RGB in one tile shader. */
export function featureForwardShader(cfg: RasterConfig): string {
  const { d, hw, code } = common(cfg);
  return /* wgsl */ `
${code}
@group(0) @binding(0) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds : array<u32>;
@group(0) @binding(2) var<storage, read> state : array<f32>;
@group(0) @binding(3) var<storage, read> features : array<f32>;
@group(0) @binding(4) var<storage, read> decoder : array<f32>;
@group(0) @binding(5) var<storage, read_write> image : array<f32>;
@group(0) @binding(6) var<storage, read_write> tileStop : array<u32>;

var<workgroup> shIds : array<u32, ${u(d.cap)}>;
var<workgroup> shMaxStop : atomic<u32>;
fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u) - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${u(d.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${u(d.cap)});
  let start = tileId * ${u(d.cap)};
  let sortN = nextPow2(count);
  for (var i = tid; i < sortN; i += 256u) {
    shIds[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  if (tid == 0u) { atomicStore(&shMaxStop, 0u); }
  workgroupBarrier();
  var k = 2u;
  loop {
    if (k > sortN) { break; }
    var j = k >> 1u;
    loop {
      if (j == 0u) { break; }
      let pairs = sortN >> 1u;
      for (var pair = tid; pair < pairs; pair += 256u) {
        let pos = 2u * j * (pair / j) + (pair % j);
        let ixj = pos + j;
        let asc = (pos & k) == 0u;
        let va = shIds[pos]; let vb = shIds[ixj];
        if ((va > vb) == asc) { shIds[pos] = vb; shIds[ixj] = va; }
      }
      workgroupBarrier();
      j >>= 1u;
    }
    k <<= 1u;
  }
  for (var i = tid; i < count; i += 256u) { binnedIds[start + i] = shIds[i]; }
  workgroupBarrier();

  let tileX = tileId % ${u(d.tilesX)};
  let tileY = tileId / ${u(d.tilesX)};
  let x = tileX * ${u(TILE)} + (tid % ${u(TILE)});
  let y = tileY * ${u(TILE)} + (tid / ${u(TILE)});
  var localStop = 0u;
  if (x < ${u(d.W)} && y < ${u(d.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var baseR = 0.0; var baseG = 0.0; var baseB = 0.0;
    var l0 = 0.0; var l1 = 0.0; var l2 = 0.0; var l3 = 0.0; var l4 = 0.0;
    var T = 1.0;
    for (var i = 0u; i < count; i++) {
      let g = shIds[i]; let s = g * STATE_STRIDE;
      let dx = pxc - state[s + ${u(STATE_MEAN_X)}];
      let dy = pyc - state[s + ${u(STATE_MEAN_Y)}];
      let a = state[s + ${u(STATE_CONIC_A)}];
      let b = state[s + ${u(STATE_CONIC_B)}];
      let c = state[s + ${u(STATE_CONIC_C)}];
      let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = state[s + ${u(STATE_OPACITY)}] * exp(power);
      let alpha = min(${f(MAX_ALPHA)}, raw);
      if (alpha < ${f(ALPHA_THRESHOLD)}) { continue; }
      let cs = state[s + ${u(STATE_COS)}]; let sn = state[s + ${u(STATE_SIN)}];
      let ux = clamp((cs * dx + sn * dy) * state[s + ${u(STATE_INV_SX)}], -3.0, 3.0);
      let uy = clamp((-sn * dx + cs * dy) * state[s + ${u(STATE_INV_SY)}], -3.0, 3.0);
      let e = g * FEATURE_STRIDE;
      let w = T * alpha;
      baseR += w * state[s + ${u(STATE_RGB_R)}];
      baseG += w * state[s + ${u(STATE_RGB_G)}];
      baseB += w * state[s + ${u(STATE_RGB_B)}];
      l0 += w * (features[e] + ux * features[e + 5u] + uy * features[e + 10u]);
      l1 += w * (features[e + 1u] + ux * features[e + 6u] + uy * features[e + 11u]);
      l2 += w * (features[e + 2u] + ux * features[e + 7u] + uy * features[e + 12u]);
      l3 += w * (features[e + 3u] + ux * features[e + 8u] + uy * features[e + 13u]);
      l4 += w * (features[e + 4u] + ux * features[e + 9u] + uy * features[e + 14u]);
      T *= 1.0 - alpha;
      if (T < ${f(TRANSMITTANCE_CUTOFF)}) { break; }
    }
    baseR += T * ${f(d.bg[0])};
    baseG += T * ${f(d.bg[1])};
    baseB += T * ${f(d.bg[2])};
    let rR = decoder[24u] + decoder[0u] * baseR + decoder[1u] * baseG + decoder[2u] * baseB + decoder[3u] * l0 + decoder[4u] * l1 + decoder[5u] * l2 + decoder[6u] * l3 + decoder[7u] * l4;
    let rG = decoder[25u] + decoder[8u] * baseR + decoder[9u] * baseG + decoder[10u] * baseB + decoder[11u] * l0 + decoder[12u] * l1 + decoder[13u] * l2 + decoder[14u] * l3 + decoder[15u] * l4;
    let rB = decoder[26u] + decoder[16u] * baseR + decoder[17u] * baseG + decoder[18u] * baseB + decoder[19u] * l0 + decoder[20u] * l1 + decoder[21u] * l2 + decoder[22u] * l3 + decoder[23u] * l4;
    let pixel = y * ${u(d.W)} + x;
    image[pixel] = sigmoid1(logit1(baseR) + RESIDUAL_SCALE * rR);
    image[${u(hw)} + pixel] = sigmoid1(logit1(baseG) + RESIDUAL_SCALE * rG);
    image[${u(2 * hw)} + pixel] = sigmoid1(logit1(baseB) + RESIDUAL_SCALE * rB);
  }
  atomicMax(&shMaxStop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&shMaxStop); }
}
`;
}

/**
 * Fused RGB-gradient -> Feature8 VJP -> splat backward.
 *
 * The production path reduces f32 contributions in hardware subgroups and
 * quantizes only each subgroup/tile partial before its device atomic. It emits
 * vastly fewer device atomics than a splat-pixel-hit reduction while retaining
 * the fixed-point accumulator used by the geometry and Adam chains.
 */
export function featureBackwardShader(cfg: RasterConfig): string {
  const { d, hw, code } = common(cfg);
  const backwardLanes = 64;
  const pixelsPerLane = (TILE * TILE) / backwardLanes;
  return /* wgsl */ `
enable subgroups;
${code}
@group(0) @binding(0) var<storage, read> gradImage : array<f32>;
@group(0) @binding(1) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read> binnedIds : array<u32>;
@group(0) @binding(3) var<storage, read> tileStop : array<u32>;
@group(0) @binding(4) var<storage, read> state : array<f32>;
@group(0) @binding(5) var<storage, read> features : array<f32>;
@group(0) @binding(6) var<storage, read> decoder : array<f32>;
@group(0) @binding(7) var<storage, read_write> acc : array<atomic<i32>>;

fn qgrad(v : f32) -> i32 {
  return i32(clamp(round(v * ${f(d.gradScale)}), -2.14e9, 2.14e9));
}

// A 16x16 tile has 256 pixels. Each lane owns a compile-time number of pixels,
// so state and feature payloads are loaded once per lane/splat and reductions
// never need a workgroup barrier. Every subgroup leader writes one partial tile
// sum to the fixed-point buffer.
const PIXELS_PER_LANE : u32 = ${u(pixelsPerLane)};

@compute @workgroup_size(${u(backwardLanes)})
fn main(
  @builtin(workgroup_id) wg : vec3u,
  @builtin(local_invocation_index) tid : u32,
  @builtin(subgroup_invocation_id) lane : u32,
) {
  let tileId = wg.x;
  if (tileId >= ${u(d.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${u(d.cap)});
  let stopc = min(count, tileStop[tileId]);
  let start = tileId * ${u(d.cap)};
  let tileX = tileId % ${u(d.tilesX)};
  let tileY = tileId / ${u(d.tilesX)};
  var valid : array<bool, ${u(pixelsPerLane)}>;
  var pixel : array<u32, ${u(pixelsPerLane)}>;
  var pxc : array<f32, ${u(pixelsPerLane)}>;
  var pyc : array<f32, ${u(pixelsPerLane)}>;
  var endi : array<u32, ${u(pixelsPerLane)}>;
  var baseR : array<f32, ${u(pixelsPerLane)}>; var baseG : array<f32, ${u(pixelsPerLane)}>; var baseB : array<f32, ${u(pixelsPerLane)}>;
  var l0 : array<f32, ${u(pixelsPerLane)}>; var l1 : array<f32, ${u(pixelsPerLane)}>; var l2 : array<f32, ${u(pixelsPerLane)}>; var l3 : array<f32, ${u(pixelsPerLane)}>; var l4 : array<f32, ${u(pixelsPerLane)}>;
  var T : array<f32, ${u(pixelsPerLane)}>;
  var gBaseR : array<f32, ${u(pixelsPerLane)}>; var gBaseG : array<f32, ${u(pixelsPerLane)}>; var gBaseB : array<f32, ${u(pixelsPerLane)}>;
  var gL0 : array<f32, ${u(pixelsPerLane)}>; var gL1 : array<f32, ${u(pixelsPerLane)}>; var gL2 : array<f32, ${u(pixelsPerLane)}>; var gL3 : array<f32, ${u(pixelsPerLane)}>; var gL4 : array<f32, ${u(pixelsPerLane)}>;
  var gT : array<f32, ${u(pixelsPerLane)}>;
  for (var p = 0u; p < PIXELS_PER_LANE; p++) {
    let localPixel = tid + ${u(backwardLanes)} * p;
    let x = tileX * ${u(TILE)} + (localPixel % ${u(TILE)});
    let y = tileY * ${u(TILE)} + (localPixel / ${u(TILE)});
    valid[p] = x < ${u(d.W)} && y < ${u(d.H)};
    pixel[p] = y * ${u(d.W)} + x;
    pxc[p] = f32(x) + 0.5;
    pyc[p] = f32(y) + 0.5;
    endi[p] = stopc;
    baseR[p] = 0.0; baseG[p] = 0.0; baseB[p] = 0.0;
    l0[p] = 0.0; l1[p] = 0.0; l2[p] = 0.0; l3[p] = 0.0; l4[p] = 0.0;
    T[p] = 1.0;
    gBaseR[p] = 0.0; gBaseG[p] = 0.0; gBaseB[p] = 0.0;
    gL0[p] = 0.0; gL1[p] = 0.0; gL2[p] = 0.0; gL3[p] = 0.0; gL4[p] = 0.0;
    gT[p] = 0.0;
  }

  // Replay the forward pass four pixels at a time. The payload is loaded once
  // per lane/splat, not once per pixel, and stopped pixels remain masked.
  for (var i = 0u; i < stopc; i++) {
    let g = binnedIds[start + i]; let s = g * STATE_STRIDE; let e = g * FEATURE_STRIDE;
    let mx = state[s + ${u(STATE_MEAN_X)}]; let my = state[s + ${u(STATE_MEAN_Y)}];
    let a = state[s + ${u(STATE_CONIC_A)}]; let b = state[s + ${u(STATE_CONIC_B)}]; let c = state[s + ${u(STATE_CONIC_C)}];
    let opacity = state[s + ${u(STATE_OPACITY)}]; let cs = state[s + ${u(STATE_COS)}]; let sn = state[s + ${u(STATE_SIN)}];
    let invSx = state[s + ${u(STATE_INV_SX)}]; let invSy = state[s + ${u(STATE_INV_SY)}];
    let cR = state[s + ${u(STATE_RGB_R)}]; let cG = state[s + ${u(STATE_RGB_G)}]; let cB = state[s + ${u(STATE_RGB_B)}];
    let f0 = features[e]; let f1 = features[e + 1u]; let f2 = features[e + 2u]; let f3 = features[e + 3u]; let f4 = features[e + 4u];
    let fx0 = features[e + 5u]; let fx1 = features[e + 6u]; let fx2 = features[e + 7u]; let fx3 = features[e + 8u]; let fx4 = features[e + 9u];
    let fy0 = features[e + 10u]; let fy1 = features[e + 11u]; let fy2 = features[e + 12u]; let fy3 = features[e + 13u]; let fy4 = features[e + 14u];
    for (var p = 0u; p < PIXELS_PER_LANE; p++) {
      if (valid[p] && i < endi[p]) {
        let dx = pxc[p] - mx; let dy = pyc[p] - my;
        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
        if (power <= 0.0) {
          let alpha = min(${f(MAX_ALPHA)}, opacity * exp(power));
          if (alpha >= ${f(ALPHA_THRESHOLD)}) {
            let ux = clamp((cs * dx + sn * dy) * invSx, -3.0, 3.0);
            let uy = clamp((-sn * dx + cs * dy) * invSy, -3.0, 3.0);
            let w = T[p] * alpha;
            baseR[p] += w * cR; baseG[p] += w * cG; baseB[p] += w * cB;
            l0[p] += w * (f0 + ux * fx0 + uy * fy0);
            l1[p] += w * (f1 + ux * fx1 + uy * fy1);
            l2[p] += w * (f2 + ux * fx2 + uy * fy2);
            l3[p] += w * (f3 + ux * fx3 + uy * fy3);
            l4[p] += w * (f4 + ux * fx4 + uy * fy4);
            T[p] *= 1.0 - alpha;
            if (T[p] < ${f(TRANSMITTANCE_CUTOFF)}) { endi[p] = i + 1u; }
          }
        }
      }
    }
  }

  // Feature decoder VJP, accumulated locally over the four pixels before the
  // subgroup reduction. There is no feature image or feature-image gradient.
  var ld0 = vec4<f32>(0.0); var ld1 = vec4<f32>(0.0); var ld2 = vec4<f32>(0.0);
  var ld3 = vec4<f32>(0.0); var ld4 = vec4<f32>(0.0); var ld5 = vec4<f32>(0.0); var ld6 = vec4<f32>(0.0);
  for (var p = 0u; p < PIXELS_PER_LANE; p++) {
    if (valid[p]) {
      baseR[p] += T[p] * ${f(d.bg[0])}; baseG[p] += T[p] * ${f(d.bg[1])}; baseB[p] += T[p] * ${f(d.bg[2])};
      let goR = gradImage[pixel[p]]; let goG = gradImage[${u(hw)} + pixel[p]]; let goB = gradImage[${u(2 * hw)} + pixel[p]];
      let rR = decoder[24u] + decoder[0u] * baseR[p] + decoder[1u] * baseG[p] + decoder[2u] * baseB[p] + decoder[3u] * l0[p] + decoder[4u] * l1[p] + decoder[5u] * l2[p] + decoder[6u] * l3[p] + decoder[7u] * l4[p];
      let rG = decoder[25u] + decoder[8u] * baseR[p] + decoder[9u] * baseG[p] + decoder[10u] * baseB[p] + decoder[11u] * l0[p] + decoder[12u] * l1[p] + decoder[13u] * l2[p] + decoder[14u] * l3[p] + decoder[15u] * l4[p];
      let rB = decoder[26u] + decoder[16u] * baseR[p] + decoder[17u] * baseG[p] + decoder[18u] * baseB[p] + decoder[19u] * l0[p] + decoder[20u] * l1[p] + decoder[21u] * l2[p] + decoder[22u] * l3[p] + decoder[23u] * l4[p];
      let outR = sigmoid1(logit1(baseR[p]) + RESIDUAL_SCALE * rR);
      let outG = sigmoid1(logit1(baseG[p]) + RESIDUAL_SCALE * rG);
      let outB = sigmoid1(logit1(baseB[p]) + RESIDUAL_SCALE * rB);
      let dzR = goR * outR * (1.0 - outR); let dzG = goG * outG * (1.0 - outG); let dzB = goB * outB * (1.0 - outB);
      let baseRc = clamp(baseR[p], ${f(EPS)}, ${f(1 - EPS)}); let baseGc = clamp(baseG[p], ${f(EPS)}, ${f(1 - EPS)}); let baseBc = clamp(baseB[p], ${f(EPS)}, ${f(1 - EPS)});
      gBaseR[p] = dzR / max(baseRc * (1.0 - baseRc), ${f(EPS)}) + RESIDUAL_SCALE * (dzR * decoder[0u] + dzG * decoder[8u] + dzB * decoder[16u]);
      gBaseG[p] = dzG / max(baseGc * (1.0 - baseGc), ${f(EPS)}) + RESIDUAL_SCALE * (dzR * decoder[1u] + dzG * decoder[9u] + dzB * decoder[17u]);
      gBaseB[p] = dzB / max(baseBc * (1.0 - baseBc), ${f(EPS)}) + RESIDUAL_SCALE * (dzR * decoder[2u] + dzG * decoder[10u] + dzB * decoder[18u]);
      gL0[p] = RESIDUAL_SCALE * (dzR * decoder[3u] + dzG * decoder[11u] + dzB * decoder[19u]);
      gL1[p] = RESIDUAL_SCALE * (dzR * decoder[4u] + dzG * decoder[12u] + dzB * decoder[20u]);
      gL2[p] = RESIDUAL_SCALE * (dzR * decoder[5u] + dzG * decoder[13u] + dzB * decoder[21u]);
      gL3[p] = RESIDUAL_SCALE * (dzR * decoder[6u] + dzG * decoder[14u] + dzB * decoder[22u]);
      gL4[p] = RESIDUAL_SCALE * (dzR * decoder[7u] + dzG * decoder[15u] + dzB * decoder[23u]);
      gT[p] = gBaseR[p] * ${f(d.bg[0])} + gBaseG[p] * ${f(d.bg[1])} + gBaseB[p] * ${f(d.bg[2])};
      ld0 += vec4<f32>(RESIDUAL_SCALE * dzR * baseR[p], RESIDUAL_SCALE * dzR * baseG[p], RESIDUAL_SCALE * dzR * baseB[p], RESIDUAL_SCALE * dzR * l0[p]);
      ld1 += vec4<f32>(RESIDUAL_SCALE * dzR * l1[p], RESIDUAL_SCALE * dzR * l2[p], RESIDUAL_SCALE * dzR * l3[p], RESIDUAL_SCALE * dzR * l4[p]);
      ld2 += vec4<f32>(RESIDUAL_SCALE * dzG * baseR[p], RESIDUAL_SCALE * dzG * baseG[p], RESIDUAL_SCALE * dzG * baseB[p], RESIDUAL_SCALE * dzG * l0[p]);
      ld3 += vec4<f32>(RESIDUAL_SCALE * dzG * l1[p], RESIDUAL_SCALE * dzG * l2[p], RESIDUAL_SCALE * dzG * l3[p], RESIDUAL_SCALE * dzG * l4[p]);
      ld4 += vec4<f32>(RESIDUAL_SCALE * dzB * baseR[p], RESIDUAL_SCALE * dzB * baseG[p], RESIDUAL_SCALE * dzB * baseB[p], RESIDUAL_SCALE * dzB * l0[p]);
      ld5 += vec4<f32>(RESIDUAL_SCALE * dzB * l1[p], RESIDUAL_SCALE * dzB * l2[p], RESIDUAL_SCALE * dzB * l3[p], RESIDUAL_SCALE * dzB * l4[p]);
      ld6 += vec4<f32>(RESIDUAL_SCALE * dzR, RESIDUAL_SCALE * dzG, RESIDUAL_SCALE * dzB, 0.0);
    }
  }
  let td0 = subgroupAdd(ld0); let td1 = subgroupAdd(ld1); let td2 = subgroupAdd(ld2); let td3 = subgroupAdd(ld3);
  let td4 = subgroupAdd(ld4); let td5 = subgroupAdd(ld5); let td6 = subgroupAdd(ld6);
  if (lane == 0u) {
    atomicAdd(&acc[DECODER_OFFSET], qgrad(td0.x)); atomicAdd(&acc[DECODER_OFFSET + 1u], qgrad(td0.y)); atomicAdd(&acc[DECODER_OFFSET + 2u], qgrad(td0.z)); atomicAdd(&acc[DECODER_OFFSET + 3u], qgrad(td0.w));
    atomicAdd(&acc[DECODER_OFFSET + 4u], qgrad(td1.x)); atomicAdd(&acc[DECODER_OFFSET + 5u], qgrad(td1.y)); atomicAdd(&acc[DECODER_OFFSET + 6u], qgrad(td1.z)); atomicAdd(&acc[DECODER_OFFSET + 7u], qgrad(td1.w));
    atomicAdd(&acc[DECODER_OFFSET + 8u], qgrad(td2.x)); atomicAdd(&acc[DECODER_OFFSET + 9u], qgrad(td2.y)); atomicAdd(&acc[DECODER_OFFSET + 10u], qgrad(td2.z)); atomicAdd(&acc[DECODER_OFFSET + 11u], qgrad(td2.w));
    atomicAdd(&acc[DECODER_OFFSET + 12u], qgrad(td3.x)); atomicAdd(&acc[DECODER_OFFSET + 13u], qgrad(td3.y)); atomicAdd(&acc[DECODER_OFFSET + 14u], qgrad(td3.z)); atomicAdd(&acc[DECODER_OFFSET + 15u], qgrad(td3.w));
    atomicAdd(&acc[DECODER_OFFSET + 16u], qgrad(td4.x)); atomicAdd(&acc[DECODER_OFFSET + 17u], qgrad(td4.y)); atomicAdd(&acc[DECODER_OFFSET + 18u], qgrad(td4.z)); atomicAdd(&acc[DECODER_OFFSET + 19u], qgrad(td4.w));
    atomicAdd(&acc[DECODER_OFFSET + 20u], qgrad(td5.x)); atomicAdd(&acc[DECODER_OFFSET + 21u], qgrad(td5.y)); atomicAdd(&acc[DECODER_OFFSET + 22u], qgrad(td5.z)); atomicAdd(&acc[DECODER_OFFSET + 23u], qgrad(td5.w));
    atomicAdd(&acc[DECODER_OFFSET + 24u], qgrad(td6.x)); atomicAdd(&acc[DECODER_OFFSET + 25u], qgrad(td6.y)); atomicAdd(&acc[DECODER_OFFSET + 26u], qgrad(td6.z));
  }

  // Reverse alpha recurrence. Each lane accumulates its four pixels, then its
  // hardware subgroup emits the tile partial. No cross-subgroup shared state
  // and no per-splat barrier are required.
  for (var ii = i32(stopc) - 1; ii >= 0; ii--) {
    var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0);
    var v3 = vec4<f32>(0.0); var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0);
    var v6 = vec4<f32>(0.0); var v7 = 0.0;
    let g = binnedIds[start + u32(ii)]; let s = g * STATE_STRIDE; let e = g * FEATURE_STRIDE;
    let mx = state[s + ${u(STATE_MEAN_X)}]; let my = state[s + ${u(STATE_MEAN_Y)}];
    let a = state[s + ${u(STATE_CONIC_A)}]; let b = state[s + ${u(STATE_CONIC_B)}]; let c = state[s + ${u(STATE_CONIC_C)}];
    let opacity = state[s + ${u(STATE_OPACITY)}]; let cs = state[s + ${u(STATE_COS)}]; let sn = state[s + ${u(STATE_SIN)}];
    let invSx = state[s + ${u(STATE_INV_SX)}]; let invSy = state[s + ${u(STATE_INV_SY)}];
    let cR = state[s + ${u(STATE_RGB_R)}]; let cG = state[s + ${u(STATE_RGB_G)}]; let cB = state[s + ${u(STATE_RGB_B)}];
    let scaleGateX = state[s + ${u(STATE_SCALE_GATE_X)}]; let scaleGateY = state[s + ${u(STATE_SCALE_GATE_Y)}];
    let f0 = features[e]; let f1 = features[e + 1u]; let f2 = features[e + 2u]; let f3 = features[e + 3u]; let f4 = features[e + 4u];
    let fx0 = features[e + 5u]; let fx1 = features[e + 6u]; let fx2 = features[e + 7u]; let fx3 = features[e + 8u]; let fx4 = features[e + 9u];
    let fy0 = features[e + 10u]; let fy1 = features[e + 11u]; let fy2 = features[e + 12u]; let fy3 = features[e + 13u]; let fy4 = features[e + 14u];
    for (var p = 0u; p < PIXELS_PER_LANE; p++) {
      if (valid[p] && u32(ii) < endi[p]) {
        let dx = pxc[p] - mx; let dy = pyc[p] - my;
        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
        if (power <= 0.0) {
          let raw = opacity * exp(power); let alpha = min(${f(MAX_ALPHA)}, raw);
          if (alpha >= ${f(ALPHA_THRESHOLD)}) {
            let denom = max(1.0 - alpha, ${f(EPS)}); let Tprev = T[p] / denom;
            let uxRaw = (cs * dx + sn * dy) * invSx; let uyRaw = (-sn * dx + cs * dy) * invSy;
            let ux = clamp(uxRaw, -3.0, 3.0); let uy = clamp(uyRaw, -3.0, 3.0);
            let z0 = f0 + ux * fx0 + uy * fy0; let z1 = f1 + ux * fx1 + uy * fy1; let z2 = f2 + ux * fx2 + uy * fy2;
            let z3 = f3 + ux * fx3 + uy * fy3; let z4 = f4 + ux * fx4 + uy * fy4;
            let dotPayload = gBaseR[p] * cR + gBaseG[p] * cG + gBaseB[p] * cB + gL0[p] * z0 + gL1[p] * z1 + gL2[p] * z2 + gL3[p] * z3 + gL4[p] * z4;
            let gAlpha = Tprev * (dotPayload - gT[p]); let w = Tprev * alpha;
            let gUx = select(0.0, w * (gL0[p] * fx0 + gL1[p] * fx1 + gL2[p] * fx2 + gL3[p] * fx3 + gL4[p] * fx4), uxRaw > -3.0 && uxRaw < 3.0);
            let gUy = select(0.0, w * (gL0[p] * fy0 + gL1[p] * fy1 + gL2[p] * fy2 + gL3[p] * fy3 + gL4[p] * fy4), uyRaw > -3.0 && uyRaw < 3.0);
            let gRaw = gAlpha * select(0.0, 1.0, raw < ${f(MAX_ALPHA)}); let gPower = gRaw * raw;
            let gdx = gPower * (-(a * dx + b * dy)); let gdy = gPower * (-(b * dx + c * dy));
            v0 += vec4<f32>(-gdx, -gdy, gPower * (-0.5) * dx * dx, gPower * (-1.0) * dx * dy);
            v1 += vec4<f32>(gPower * (-0.5) * dy * dy, gBaseR[p] * w, gBaseG[p] * w, gBaseB[p] * w);
            v2 += vec4<f32>(gRaw * (raw / max(opacity, ${f(EPS)})), gL0[p] * w, gL1[p] * w, gL2[p] * w);
            v3 += vec4<f32>(gL3[p] * w, gL4[p] * w, gL0[p] * w * ux, gL1[p] * w * ux);
            v4 += vec4<f32>(gL2[p] * w * ux, gL3[p] * w * ux, gL4[p] * w * ux, gL0[p] * w * uy);
            v5 += vec4<f32>(gL1[p] * w * uy, gL2[p] * w * uy, gL3[p] * w * uy, gL4[p] * w * uy);
            v6 += vec4<f32>(gUx * (-cs * invSx) + gUy * (sn * invSy), gUx * (-sn * invSx) + gUy * (-cs * invSy), gUx * (-uxRaw) * scaleGateX, gUy * (-uyRaw) * scaleGateY);
            v7 += gUx * ((-sn * dx + cs * dy) * invSx) + gUy * ((-cs * dx - sn * dy) * invSy);
            gT[p] = alpha * dotPayload + (1.0 - alpha) * gT[p];
            T[p] = Tprev;
          }
        }
      }
    }
    let r0 = subgroupAdd(v0);
    let r1 = subgroupAdd(v1);
    let r2 = subgroupAdd(v2);
    let r3 = subgroupAdd(v3);
    let r4 = subgroupAdd(v4);
    let r5 = subgroupAdd(v5);
    let r6 = subgroupAdd(v6);
    let r7 = subgroupAdd(v7);
    if (lane == 0u) {
      let ab = g * ACC_STRIDE;
      atomicAdd(&acc[ab], qgrad(r0.x)); atomicAdd(&acc[ab + 1u], qgrad(r0.y)); atomicAdd(&acc[ab + 2u], qgrad(r0.z)); atomicAdd(&acc[ab + 3u], qgrad(r0.w));
      atomicAdd(&acc[ab + 4u], qgrad(r1.x)); atomicAdd(&acc[ab + 5u], qgrad(r1.y)); atomicAdd(&acc[ab + 6u], qgrad(r1.z)); atomicAdd(&acc[ab + 7u], qgrad(r1.w));
      atomicAdd(&acc[ab + 8u], qgrad(r2.x)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET], qgrad(r2.y)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 1u], qgrad(r2.z)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 2u], qgrad(r2.w));
      atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 3u], qgrad(r3.x)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 4u], qgrad(r3.y)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 5u], qgrad(r3.z)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 6u], qgrad(r3.w));
      atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 7u], qgrad(r4.x)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 8u], qgrad(r4.y)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 9u], qgrad(r4.z)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 10u], qgrad(r4.w));
      atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 11u], qgrad(r5.x)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 12u], qgrad(r5.y)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 13u], qgrad(r5.z)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 14u], qgrad(r5.w));
      atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET], qgrad(r6.x)); atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET + 1u], qgrad(r6.y)); atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET + 2u], qgrad(r6.z)); atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET + 3u], qgrad(r6.w));
      atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET + 4u], qgrad(r7));
    }
  }
}
`;
}

/** Derived/conic gradients plus direct local-frame gradients -> raw splat grads. */
export function featureGeometryChainShader(cfg: RasterConfig): string {
  const { d, code } = common(cfg);
  const mean = 0;
  const logScale = 2 * d.G;
  const theta = 4 * d.G;
  const colorRaw = 5 * d.G;
  const opacityRaw = 8 * d.G;
  return /* wgsl */ `
${code}
@group(0) @binding(0) var<storage, read> acc : array<i32>;
@group(0) @binding(1) var<storage, read> state : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${u(d.G)}) { return; }
  let ab = g * ACC_STRIDE;
  let sb = g * STATE_STRIDE;
  let inv = ${f(1 / d.gradScale)};
  let gmx = f32(acc[ab]) * inv;
  let gmy = f32(acc[ab + 1u]) * inv;
  let gA = f32(acc[ab + 2u]) * inv;
  let gB = f32(acc[ab + 3u]) * inv;
  let gC = f32(acc[ab + 4u]) * inv;
  let gc0 = f32(acc[ab + 5u]) * inv;
  let gc1 = f32(acc[ab + 6u]) * inv;
  let gc2 = f32(acc[ab + 7u]) * inv;
  let gop = f32(acc[ab + 8u]) * inv;
  let invSx = state[sb + ${u(STATE_INV_SX)}]; let invSy = state[sb + ${u(STATE_INV_SY)}];
  let ix = invSx * invSx; let iy = invSy * invSy;
  let cs = state[sb + ${u(STATE_COS)}]; let sn = state[sb + ${u(STATE_SIN)}];
  let gix = gA * cs * cs + gB * cs * sn + gC * sn * sn;
  let giy = gA * sn * sn - gB * cs * sn + gC * cs * cs;
  let glsx = gix * (-2.0 * ix) * state[sb + ${u(STATE_SCALE_GATE_X)}];
  let glsy = giy * (-2.0 * iy) * state[sb + ${u(STATE_SCALE_GATE_Y)}];
  let gth = (ix - iy) * ((cs * cs - sn * sn) * gB + 2.0 * cs * sn * (gC - gA));
  let lmx = f32(acc[ab + ACC_LOCAL_RAW_OFFSET]) * inv;
  let lmy = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 1u]) * inv;
  let llsx = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 2u]) * inv;
  let llsy = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 3u]) * inv;
  let lth = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 4u]) * inv;
  let color0 = state[sb + ${u(STATE_RGB_R)}]; let color1 = state[sb + ${u(STATE_RGB_G)}]; let color2 = state[sb + ${u(STATE_RGB_B)}];
  let opacity = state[sb + ${u(STATE_OPACITY)}];
  gradRaw[${u(mean)} + g * 2u] = gmx + lmx;
  gradRaw[${u(mean)} + g * 2u + 1u] = gmy + lmy;
  gradRaw[${u(logScale)} + g * 2u] = glsx + llsx;
  gradRaw[${u(logScale)} + g * 2u + 1u] = glsy + llsy;
  gradRaw[${u(theta)} + g] = gth + lth;
  gradRaw[${u(colorRaw)} + g * 3u] = gc0 * color0 * (1.0 - color0);
  gradRaw[${u(colorRaw)} + g * 3u + 1u] = gc1 * color1 * (1.0 - color1);
  gradRaw[${u(colorRaw)} + g * 3u + 2u] = gc2 * color2 * (1.0 - color2);
  gradRaw[${u(opacityRaw)} + g] = gop * opacity * (1.0 - opacity);
}
`;
}

/** Packed accumulator -> feature, direct geometry, and decoder gradients. */
export function featureChainShader(cfg: RasterConfig): string {
  const { d, code } = common(cfg);
  return /* wgsl */ `
${code}
@group(0) @binding(0) var<storage, read> acc : array<i32>;
@group(0) @binding(1) var<storage, read_write> gradFeatures : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradDecoder : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  let inv = ${f(1 / d.gradScale)};
  if (i < ${u(d.G)}) {
    let g = i;
    let ab = g * ACC_STRIDE + ACC_EXTRA_OFFSET;
    let fb = g * FEATURE_STRIDE;
    for (var ch = 0u; ch < FEATURE_STRIDE; ch++) { gradFeatures[fb + ch] = f32(acc[ab + ch]) * inv; }
  }
  if (i < ${u(DECODER_PARAM_COUNT)}) { gradDecoder[i] = f32(acc[DECODER_OFFSET + i]) * inv; }
}
`;
}

export const FEATURE_CHAIN_WORK_ITEMS = (cfg: RasterConfig): number =>
  Math.max(resolveDims(cfg).G, DECODER_PARAM_COUNT);
