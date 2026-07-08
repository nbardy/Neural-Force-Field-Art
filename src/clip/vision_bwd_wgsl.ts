/**
 * vision_bwd_wgsl — PURE WGSL codegen for the fused MobileCLIP-S0 vision
 * encoder BACKWARD pass: dL/dpixels with all WEIGHTS FROZEN (no dW anywhere).
 * This gradient feeds the Gaussian-splat rasterizer so splats can optimize a
 * canvas toward a text prompt (docs/clip_backward_spec.md).
 *
 * Same pure data→string discipline as vision_wgsl.ts: κ (compile_plan.py
 * --train) emits plan_train.json with a `backward:[...]` reverse step list;
 * this module turns each backward entry into ONE specialized compute dispatch.
 * Every shape/offset is a baked literal. The tiled pointwise matmul is SHARED
 * with the forward (pw_bwd reuses pointwiseTiledMain reading the transposed
 * weights), and the erf/GELU constants are imported so forward and backward
 * cannot drift.
 *
 * Grad-slot convention (κ): grad(activation slot s) = nActSlots + s. A grad
 * slot's FIRST writer overwrites; later writers ADD (`accumulate:true`, set by
 * κ from forward-reader order) — no global zero-fill. Saved activations read
 * for recompute (se input, gelu pre-activation, attn qkv, embed) live in their
 * original activation slots (never freed in train mode).
 */
/// <reference types="@webgpu/types" />
import {
  GELU,
  weightsDecl,
  assertStep,
  assertPointwiseTiles,
  PW_TILE_DECLS,
  pointwiseTiledMain,
  type DispatchSpec,
  type BufferRef,
  type VisionPlan,
} from "./vision_wgsl";

// ---------------------------------------------------------------------------
// Canonical backward types (mirror plan_train.json `backward` — κ is the only
// producer). Every entry names its grad src (dY) / dst (dX) SLOTS, any saved
// activation slot, weight offsets, and the accumulate flag.
// ---------------------------------------------------------------------------

/** L = −cos(embed, text). Reads saved embed + the per-prompt text buffer,
 *  writes dL/dembed. d(−cos)/de = −(t/(|e||t|) − cos·e/|e|²). */
export interface LossBwdStep {
  kind: "loss_bwd"; name: string;
  embed: number; dX: number; dim: number; accumulate: boolean;
}

/** GAP backward (1/P broadcast) + matmul backward reading W in STORED
 *  orientation: dX[ci][p] = (1/P)·Σ_co W[ci][co]·dEmb[co]. */
export interface HeadBwdStep {
  kind: "head_bwd"; name: string;
  cin: number; cout: number; h: number; w: number;
  wOff: number; dY: number; dX: number; accumulate: boolean;
}

/** dX = dY ⊙ gelu'(x_pre); gelu'(x)=Φ(x)+x·φ(x). Reads the saved pre-activation. */
export interface GeluBwdStep {
  kind: "gelu_bwd"; name: string;
  n: number; pre: number; dY: number; dX: number; accumulate: boolean;
}

/** dX = Wᵀ·dY via the tiled pointwise kernel over the transposed weights
 *  (γ layer-scale folded into wOffT by κ). cin=reduction(=fwd cout),
 *  cout=output(=fwd cin). No bias, no activation. */
export interface PwBwdStep {
  kind: "pw_bwd"; name: string;
  cin: number; cout: number; outH: number; outW: number;
  wOffT: number; dY: number; dX: number; accumulate: boolean;
}

/** Routes the (unscaled) fused-conv output gradient into the residual
 *  producer's grad slot: grad[res] (+)= dOut. */
export interface ResidualBwdStep {
  kind: "residual_bwd"; name: string;
  n: number; dY: number; dX: number; accumulate: boolean;
}

/** Gather-form conv backward, one thread per INPUT pixel:
 *  dX[ci][iy][ix] = Σ_{co∈group} Σ_{ky,kx valid} W[co][ci_local][ky][kx]·dY[co][oy][ox],
 *  oy=(iy+pad−ky)/stride (divisible & in-bounds). Depthwise = cpg=1 case;
 *  stride∈{1,2} baked (stride-2 = parity check). */
export interface SpatialBwdStep {
  kind: "spatial_bwd"; name: string;
  cin: number; cout: number; k: number; stride: number; pad: number;
  groups: number; h: number; w: number; outH: number; outW: number;
  wOff: number; dY: number; dX: number; accumulate: boolean;
}

/** SE gate backward in one workgroup: recompute gap/mid/scale from the saved
 *  input, then dX = dY⊙scale + (1/P)·(fc1ᵀ relu' fc2ᵀ σ' Σ_p dY·x) broadcast. */
export interface SeBwdStep {
  kind: "se_bwd"; name: string;
  c: number; cmid: number; h: number; w: number;
  w1Off: number; b1Off: number; w2Off: number; b2Off: number;
  dY: number; savedSrc: number; dX: number; accumulate: boolean;
}

/** MHSA backward, one workgroup per head. Recomputes softmax from saved qkv
 *  (probs NOT stored), then dV=pᵀdO, dP=dOVᵀ, dS=p(dP−Σp·dP), dQ=dS·K, dK=dSᵀQ.
 *  Q is already 1/√d-scaled upstream (folded in qkv weights) → no extra scale.
 *  Writes d_qkv channel-planar (same layout as the forward qkv slot). */
export interface AttnCoreBwdStep {
  kind: "attn_core_bwd"; name: string;
  c: number; heads: number; hd: number; nTok: number;
  dY: number; savedQkv: number; dX: number; accumulate: boolean;
}

export type BwdStep =
  | LossBwdStep | HeadBwdStep | GeluBwdStep | PwBwdStep
  | ResidualBwdStep | SpatialBwdStep | SeBwdStep | AttnCoreBwdStep;

/** The train plan is the forward VisionPlan plus these fields. */
export interface TrainPlan extends VisionPlan {
  backward: BwdStep[];
  nActSlots: number;
  inputGradSlot: number;
  embedSlot: number;
  embedGradSlot: number;
  textDim: number;
}

export interface BwdDispatchOptions {
  stemSpatialBwd?: boolean;
}

// ---------------------------------------------------------------------------
// Shared fragments
// ---------------------------------------------------------------------------

/** gelu'(x) = Φ(x) + x·φ(x), Φ(x)=0.5(1+erf(x/√2)), φ(x)=exp(−x²/2)/√(2π).
 *  Uses the EXACT forward erf4 (imported GELU) so the two never desync. */
const GELU_GRAD = /* wgsl */ `
fn geluGrad4(x : vec4f) -> vec4f {
  let cdf = 0.5 * (vec4f(1.0) + erf4(x * 0.7071067811865476));
  let pdf = 0.3989422804014327 * exp(-0.5 * x * x);   // 1/sqrt(2π)
  return cdf + x * pdf;
}`;

const gradSlot = (s: number): BufferRef => ({ kind: "slot", slot: s });

// ---------------------------------------------------------------------------
// pw_bwd — dX = Wᵀ·dY, tiled pointwise kernel over the transposed weights.
// ---------------------------------------------------------------------------

function pwBwd(s: PwBwdStep): DispatchSpec {
  const P = s.outH * s.outW;
  assertPointwiseTiles(s.name, s.cin, s.cout, P, s.wOffT);
  const P4 = P / 4;
  const store = (j: number): string =>
    s.accumulate ? `dst[(co + ${j}u) * ${P4}u + p4] + acc${j}` : `acc${j}`;
  const code = /* wgsl */ `
${weightsDecl(0)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;         // dY  [Cin][P4]
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;   // dX  [Cout][P4]
${PW_TILE_DECLS}
${pointwiseTiledMain({
    cin: s.cin, cout: s.cout, P4, wOff: s.wOffT,
    init: () => `vec4f(0.0)`,
    store,
  })}`;
  return {
    label: `pw_bwd ${s.cin}->${s.cout} @${s.outH}x${s.outW}${s.accumulate ? " +=" : ""}`,
    code,
    workgroups: [P4 / 8, s.cout / 32, 1],
    buffers: [{ kind: "weights" }, gradSlot(s.dY), gradSlot(s.dX)],
  };
}

// ---------------------------------------------------------------------------
// residual_bwd — grad[res] (+)= dOut (elementwise vec4 copy/add).
// ---------------------------------------------------------------------------

function residualBwd(s: ResidualBwdStep): DispatchSpec {
  assertStep(s.n % 4 === 0, `${s.name}: residual n%4 != 0`);
  const n4 = s.n / 4;
  const val = s.accumulate ? `dst[i] + src[i]` : `src[i]`;
  const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src : array<vec4f>;         // dOut
@group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;   // grad[res]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${n4}u) { return; }
  dst[i] = ${val};
}`;
  return {
    label: `residual_bwd n${s.n}${s.accumulate ? " +=" : ""}`,
    code,
    workgroups: [Math.ceil(n4 / 64), 1, 1],
    buffers: [gradSlot(s.dY), gradSlot(s.dX)],
  };
}

// ---------------------------------------------------------------------------
// gelu_bwd — dX = dY ⊙ gelu'(x_pre), one thread per quad.
// ---------------------------------------------------------------------------

function geluBwd(s: GeluBwdStep): DispatchSpec {
  assertStep(s.n % 4 === 0, `${s.name}: gelu_bwd n%4 != 0`);
  const n4 = s.n / 4;
  const val = s.accumulate ? `dst[i] + g` : `g`;
  const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> dy : array<vec4f>;
@group(0) @binding(1) var<storage, read> pre : array<vec4f>;         // saved pre-activation
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${GELU}
${GELU_GRAD}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${n4}u) { return; }
  let g = dy[i] * geluGrad4(pre[i]);
  dst[i] = ${val};
}`;
  return {
    label: `gelu_bwd n${s.n}${s.accumulate ? " +=" : ""}`,
    code,
    workgroups: [Math.ceil(n4 / 64), 1, 1],
    buffers: [gradSlot(s.dY), gradSlot(s.pre), gradSlot(s.dX)],
  };
}

// ---------------------------------------------------------------------------
// spatial_bwd — gather conv backward, one workgroup per input channel, one
// thread per input pixel. Mirror of the forward spatialConv. Stride∈{1,2}
// baked (stride-2 = parity check on iy+pad−ky). Depthwise = cpg=1 special case.
// ---------------------------------------------------------------------------

function spatialBwd(s: SpatialBwdStep): DispatchSpec {
  assertStep(s.stride === 1 || s.stride === 2, `${s.name}: stride ${s.stride} not in {1,2}`);
  const cpg = s.cin / s.groups;          // input channels per group (1 or 3)
  const cpgOut = s.cout / s.groups;      // output channels per group
  assertStep(Number.isInteger(cpg) && Number.isInteger(cpgOut), `${s.name}: bad groups`);
  const WK = cpg * s.k * s.k;            // one cout's weight footprint
  const P = s.h * s.w;
  const outP = s.outH * s.outW;
  // oy/ox from the input coord: stride-1 is identity; stride-2 needs the tap to
  // land on an even offset (divisible by 2) — a baked parity check + shift.
  const solve = (t: string, o: string): string =>
    s.stride === 1
      ? `let ${o} = ${t};`
      : `if ((${t} & 1) != 0) { continue; } let ${o} = ${t} >> 1;`;
  const store = s.accumulate ? `dx[o] + acc` : `acc`;
  const code = /* wgsl */ `
${weightsDecl(0)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // [Cout][outH][outW]
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // [Cin][H][W]
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let ci = gid.y;
  let p = gid.x;
  if (p >= ${P}u) { return; }
  let iy = i32(p / ${s.w}u);
  let ix = i32(p % ${s.w}u);
  let grp = ci / ${cpg}u;
  let ci_local = ci - grp * ${cpg}u;
  var acc = 0.0;
  for (var col = 0u; col < ${cpgOut}u; col = col + 1u) {
    let co = grp * ${cpgOut}u + col;
    let wbase = ${s.wOff}u + co * ${WK}u + ci_local * ${s.k * s.k}u;
    let dybase = co * ${outP}u;
    for (var ky = 0; ky < ${s.k}; ky = ky + 1) {
      let ty = iy + ${s.pad} - ky;
      ${solve("ty", "oy")}
      if (oy < 0 || oy >= ${s.outH}) { continue; }
      let rowW = wbase + u32(ky) * ${s.k}u;
      let rowY = dybase + u32(oy) * ${s.outW}u;
      for (var kx = 0; kx < ${s.k}; kx = kx + 1) {
        let tx = ix + ${s.pad} - kx;
        ${solve("tx", "ox")}
        if (ox < 0 || ox >= ${s.outW}) { continue; }
        acc = fma(W(rowW + u32(kx)), dy[rowY + u32(ox)], acc);
      }
    }
  }
  let o = ci * ${P}u + u32(iy) * ${s.w}u + u32(ix);
  dx[o] = ${store};
}`;
  return {
    label: `spatial_bwd k${s.k}s${s.stride} ${s.cin}<-${s.cout} g${s.groups} @${s.h}x${s.w}${s.accumulate ? " +=" : ""}`,
    code,
    workgroups: [Math.ceil(P / 64), s.cin, 1],
    buffers: [{ kind: "weights" }, gradSlot(s.dY), gradSlot(s.dX)],
  };
}

function isStemSpatialBwd(s: SpatialBwdStep): boolean {
  return s.cin === 3 &&
    s.cout === 64 &&
    s.k === 3 &&
    s.stride === 2 &&
    s.pad === 1 &&
    s.groups === 1 &&
    s.h === 256 &&
    s.w === 256 &&
    s.outH === 128 &&
    s.outW === 128 &&
    !s.accumulate;
}

function spatialBwdStem4(s: SpatialBwdStep): DispatchSpec {
  assertStep(isStemSpatialBwd(s), `${s.name}: stem spatial_bwd specialization received wrong shape`);
  const P = s.h * s.w;
  const Q = P / 4;
  const outP = s.outH * s.outW;
  const addKxs = (rowY: string, wbase: string): string => /* wgsl */ `
      {
        var d3 = 0.0;
        if (oxBase + 2u < 128u) {
          d3 = dy[${rowY} + oxBase + 2u];
        }
        acc = fma(vec4f(W(${wbase} + 0u)), vec4f(0.0, dy[${rowY} + oxBase + 1u], 0.0, d3), acc);
        acc = fma(vec4f(W(${wbase} + 1u)), vec4f(dy[${rowY} + oxBase], 0.0, dy[${rowY} + oxBase + 1u], 0.0), acc);
        acc = fma(vec4f(W(${wbase} + 2u)), vec4f(0.0, dy[${rowY} + oxBase], 0.0, dy[${rowY} + oxBase + 1u]), acc);
      }`;
  const code = /* wgsl */ `
${weightsDecl(0)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // [64][128][128]
@group(0) @binding(2) var<storage, read_write> dx : array<vec4f>;    // [3][256][64 vec4s]

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let ci = gid.y;
  let q = gid.x;
  if (q >= ${Q}u) { return; }
  let iy = q / 64u;
  let ix0 = (q - iy * 64u) * 4u;
  let oxBase = ix0 >> 1u;
  var acc = vec4f(0.0);

  if ((iy & 1u) == 0u) {
    let oy = iy >> 1u;
    for (var co = 0u; co < 64u; co = co + 1u) {
      let rowY = co * ${outP}u + oy * 128u;
      let wbase = ${s.wOff}u + co * 27u + ci * 9u + 3u; // ky = 1
${addKxs("rowY", "wbase")}
    }
  } else {
    for (var co = 0u; co < 64u; co = co + 1u) {
      if (iy < 255u) {
        let oy0 = (iy + 1u) >> 1u;
        let rowY0 = co * ${outP}u + oy0 * 128u;
        let wbase0 = ${s.wOff}u + co * 27u + ci * 9u; // ky = 0
${addKxs("rowY0", "wbase0")}
      }
      let oy2 = (iy - 1u) >> 1u;
      let rowY2 = co * ${outP}u + oy2 * 128u;
      let wbase2 = ${s.wOff}u + co * 27u + ci * 9u + 6u; // ky = 2
${addKxs("rowY2", "wbase2")}
    }
  }

  dx[ci * ${Q}u + q] = acc;
}`;
  return {
    label: `spatial_bwd_stem4 k3s2 3<-64 g1 @256x256`,
    code,
    workgroups: [Math.ceil(Q / 64), 3, 1],
    buffers: [{ kind: "weights" }, gradSlot(s.dY), gradSlot(s.dX)],
  };
}

// ---------------------------------------------------------------------------
// se_bwd — full SE gate backward in one workgroup. Recompute gap/mid/scale from
// the SAVED input (cheaper than saving them), then the standard SE chain rule.
// ---------------------------------------------------------------------------

const SE_WG = 256;

function seBwd(s: SeBwdStep): DispatchSpec {
  const P = s.h * s.w;
  assertStep(s.c <= 2048 && s.cmid <= 512, `${s.name}: SE dims exceed shared-memory plan`);
  const dxVal = s.accumulate ? `dx[i] + v` : `v`;
  const code = /* wgsl */ `
${weightsDecl(0)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // grad[se out]
@group(0) @binding(2) var<storage, read> src : array<f32>;           // saved se input
@group(0) @binding(3) var<storage, read_write> dx : array<f32>;      // grad[se in]
// tmp holds gap (steps 1-2) then dL/dpre2 (steps 3-4) — disjoint lifetimes, so
// one array of size c instead of two keeps workgroup memory ≤ 16KB even at c=1024.
var<workgroup> tmp  : array<f32, ${s.c}>;
var<workgroup> mid  : array<f32, ${s.cmid}>;   // relu(pre1)
var<workgroup> scl  : array<f32, ${s.c}>;      // sigmoid gate
var<workgroup> gp1  : array<f32, ${s.cmid}>;   // dL/dpre1
var<workgroup> ggap : array<f32, ${s.c}>;      // dL/dgap
@compute @workgroup_size(${SE_WG})
fn main(@builtin(local_invocation_index) li : u32) {
  // 1. GAP (recompute) → tmp
  for (var c = li; c < ${s.c}u; c = c + ${SE_WG}u) {
    var sum = 0.0;
    for (var p = 0u; p < ${P}u; p = p + 1u) { sum = sum + src[c * ${P}u + p]; }
    tmp[c] = sum / ${P}.0;
  }
  workgroupBarrier();
  // 2. fc1 + relu (recompute) — reads gap from tmp
  for (var m = li; m < ${s.cmid}u; m = m + ${SE_WG}u) {
    var sum = W(${s.b1Off}u + m);
    for (var c = 0u; c < ${s.c}u; c = c + 1u) {
      sum = fma(tmp[c], W(${s.w1Off}u + m * ${s.c}u + c), sum);
    }
    mid[m] = max(sum, 0.0);
  }
  workgroupBarrier();
  // 3. fc2 + sigmoid (recompute), and dL/dpre2 = (Σ_p dY·x)·σ'(pre2) → tmp (gap dead)
  for (var c = li; c < ${s.c}u; c = c + ${SE_WG}u) {
    var pre2 = W(${s.b2Off}u + c);
    for (var m = 0u; m < ${s.cmid}u; m = m + 1u) {
      pre2 = fma(mid[m], W(${s.w2Off}u + c * ${s.cmid}u + m), pre2);
    }
    let sc = 1.0 / (1.0 + exp(-pre2));
    scl[c] = sc;
    var gscl = 0.0;   // dL/dscl[c] = Σ_p dY[c][p]·x[c][p]
    for (var p = 0u; p < ${P}u; p = p + 1u) {
      gscl = fma(dy[c * ${P}u + p], src[c * ${P}u + p], gscl);
    }
    tmp[c] = gscl * sc * (1.0 - sc);
  }
  workgroupBarrier();
  // 4. dL/dmid = fc2ᵀ·dpre2 ; dL/dpre1 = relu'·dmid  (dpre2 in tmp)
  for (var m = li; m < ${s.cmid}u; m = m + ${SE_WG}u) {
    var gm = 0.0;
    for (var c = 0u; c < ${s.c}u; c = c + 1u) {
      gm = fma(tmp[c], W(${s.w2Off}u + c * ${s.cmid}u + m), gm);
    }
    gp1[m] = select(0.0, gm, mid[m] > 0.0);
  }
  workgroupBarrier();
  // 5. dL/dgap = fc1ᵀ·dpre1
  for (var c = li; c < ${s.c}u; c = c + ${SE_WG}u) {
    var gg = 0.0;
    for (var m = 0u; m < ${s.cmid}u; m = m + 1u) {
      gg = fma(gp1[m], W(${s.w1Off}u + m * ${s.c}u + c), gg);
    }
    ggap[c] = gg;
  }
  workgroupBarrier();
  // 6. dX = dY⊙scale + (1/P)·ggap broadcast
  for (var i = li; i < ${s.c * P}u; i = i + ${SE_WG}u) {
    let c = i / ${P}u;
    let v = dy[i] * scl[c] + ggap[c] / ${P}.0;
    dx[i] = ${dxVal};
  }
}`;
  return {
    label: `se_bwd c${s.c} mid${s.cmid} @${s.h}x${s.w}${s.accumulate ? " +=" : ""}`,
    code,
    workgroups: [1, 1, 1],
    buffers: [{ kind: "weights" }, gradSlot(s.dY), gradSlot(s.savedSrc), gradSlot(s.dX)],
  };
}

// ---------------------------------------------------------------------------
// head_bwd — GAP backward (1/P broadcast) ∘ matmul backward (W stored orient).
// ---------------------------------------------------------------------------

const HEAD_WG = 256;

function headBwd(s: HeadBwdStep): DispatchSpec {
  const P = s.h * s.w;
  const dxVal = s.accumulate ? `dx[o] + v` : `v`;
  const code = /* wgsl */ `
${weightsDecl(0)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // dEmb [Cout]
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // grad[head src] [Cin][P]
var<workgroup> dgap : array<f32, ${s.cin}>;
@compute @workgroup_size(${HEAD_WG})
fn main(@builtin(local_invocation_index) li : u32) {
  // dgap[ci] = Σ_co W[ci][co]·dEmb[co]   (W packed [Cin][Cout], stored orientation)
  for (var ci = li; ci < ${s.cin}u; ci = ci + ${HEAD_WG}u) {
    var acc = 0.0;
    for (var co = 0u; co < ${s.cout}u; co = co + 1u) {
      acc = fma(W(${s.wOff}u + ci * ${s.cout}u + co), dy[co], acc);
    }
    dgap[ci] = acc / ${P}.0;   // GAP backward: 1/P broadcast
  }
  workgroupBarrier();
  for (var o = li; o < ${s.cin * P}u; o = o + ${HEAD_WG}u) {
    let v = dgap[o / ${P}u];
    dx[o] = ${dxVal};
  }
}`;
  return {
    label: `head_bwd ${s.cout}->${s.cin}${s.accumulate ? " +=" : ""}`,
    code,
    workgroups: [1, 1, 1],
    buffers: [{ kind: "weights" }, gradSlot(s.dY), gradSlot(s.dX)],
  };
}

// ---------------------------------------------------------------------------
// loss_bwd — L = −cos(e,t). One workgroup reduction, then per-element grad.
// ---------------------------------------------------------------------------

const LOSS_WG = 256;

function lossBwd(s: LossBwdStep): DispatchSpec {
  const dxVal = s.accumulate ? `dx[k] + g` : `g`;
  const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> e : array<f32>;             // saved embed
@group(0) @binding(1) var<storage, read> t : array<f32>;             // text embedding
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // dL/dembed
var<workgroup> pe : array<f32, ${LOSS_WG}>;   // partial Σ e²
var<workgroup> pt : array<f32, ${LOSS_WG}>;   // partial Σ t²
var<workgroup> pd : array<f32, ${LOSS_WG}>;   // partial Σ e·t
var<workgroup> ne : f32;
var<workgroup> nt : f32;
var<workgroup> cosv : f32;
@compute @workgroup_size(${LOSS_WG})
fn main(@builtin(local_invocation_index) li : u32) {
  var se = 0.0; var st = 0.0; var sd = 0.0;
  for (var k = li; k < ${s.dim}u; k = k + ${LOSS_WG}u) {
    let ev = e[k]; let tv = t[k];
    se = se + ev * ev; st = st + tv * tv; sd = sd + ev * tv;
  }
  pe[li] = se; pt[li] = st; pd[li] = sd;
  workgroupBarrier();
  for (var stride = ${LOSS_WG}u / 2u; stride > 0u; stride = stride >> 1u) {
    if (li < stride) {
      pe[li] = pe[li] + pe[li + stride];
      pt[li] = pt[li] + pt[li + stride];
      pd[li] = pd[li] + pd[li + stride];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    ne = sqrt(max(pe[0], 1e-20));
    nt = sqrt(max(pt[0], 1e-20));
    cosv = pd[0] / (ne * nt);
  }
  workgroupBarrier();
  // d(−cos)/de_k = −( t_k/(|e||t|) − cos·e_k/|e|² )
  let invET = 1.0 / (ne * nt);
  let cosOverE2 = cosv / (ne * ne);
  for (var k = li; k < ${s.dim}u; k = k + ${LOSS_WG}u) {
    let g = -(t[k] * invET - cosOverE2 * e[k]);
    dx[k] = ${dxVal};
  }
}`;
  return {
    label: `loss_bwd -cos dim${s.dim}`,
    code,
    workgroups: [1, 1, 1],
    buffers: [gradSlot(s.embed), { kind: "text" }, gradSlot(s.dX)],
  };
}

// ---------------------------------------------------------------------------
// attn_core_bwd — MHSA backward, one workgroup per head, one thread per token.
// Two phases: (1) thread = query i recomputes softmax row from saved qkv,
// writes dQ_i and stashes per-row scalars (max, denom, rowdot) in shared mem;
// (2) thread = key/value j accumulates dV_j = Σ_i p_ij·dO_i and
// dK_j = Σ_i dS_ij·q_i, re-reading q_i / dO_i from global. Only 3·nTok floats
// of shared memory — no nTok×nTok materialization.
// ---------------------------------------------------------------------------

function attnCoreBwd(s: AttnCoreBwdStep): DispatchSpec {
  const { c, heads, hd, nTok } = s;
  assertStep(c === heads * hd, `${s.name}: c != heads*hd`);
  const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> qkv : array<f32>;           // saved [3C][nTok] planar
@group(0) @binding(1) var<storage, read> dO : array<f32>;            // grad[attnOut] [C][nTok]
@group(0) @binding(2) var<storage, read_write> dQKV : array<f32>;    // grad[qkv] [3C][nTok]
var<workgroup> mrow : array<f32, ${nTok}>;   // per-query softmax max
var<workgroup> drow : array<f32, ${nTok}>;   // per-query softmax denom
var<workgroup> rdot : array<f32, ${nTok}>;   // per-query Σ_k p_ik·dP_ik
@compute @workgroup_size(${nTok})
fn main(@builtin(local_invocation_index) tid : u32,
        @builtin(workgroup_id) wid : vec3u) {
  let head = wid.x;
  let qCh = head * ${hd}u;              // q channels [qCh, qCh+hd)
  let kCh = ${c}u + head * ${hd}u;
  let vCh = ${2 * c}u + head * ${hd}u;

  // ---- phase 1: thread = query i ----
  let i = tid;
  var qi : array<f32, ${hd}>;
  var dOi : array<f32, ${hd}>;
  for (var d = 0u; d < ${hd}u; d = d + 1u) {
    qi[d]  = qkv[(qCh + d) * ${nTok}u + i];
    dOi[d] = dO[(qCh + d) * ${nTok}u + i];
  }
  var p  : array<f32, ${nTok}>;
  var dP : array<f32, ${nTok}>;
  var mx = -3.0e38;
  for (var j = 0u; j < ${nTok}u; j = j + 1u) {
    var sc = 0.0;
    for (var d = 0u; d < ${hd}u; d = d + 1u) { sc = fma(qi[d], qkv[(kCh + d) * ${nTok}u + j], sc); }
    p[j] = sc;
    mx = max(mx, sc);
  }
  var den = 0.0;
  for (var j = 0u; j < ${nTok}u; j = j + 1u) { let e = exp(p[j] - mx); p[j] = e; den = den + e; }
  let inv = 1.0 / den;
  var rd = 0.0;
  for (var j = 0u; j < ${nTok}u; j = j + 1u) {
    p[j] = p[j] * inv;                                  // p_ij
    var dpj = 0.0;
    for (var d = 0u; d < ${hd}u; d = d + 1u) { dpj = fma(dOi[d], qkv[(vCh + d) * ${nTok}u + j], dpj); }
    dP[j] = dpj;                                        // dP_ij = Σ_d dO_i·V_j
    rd = fma(p[j], dpj, rd);                            // Σ_k p_ik·dP_ik
  }
  // dQ_i = Σ_j dS_ij·K_j,  dS_ij = p_ij(dP_ij − rd)
  for (var d = 0u; d < ${hd}u; d = d + 1u) {
    var acc = 0.0;
    for (var j = 0u; j < ${nTok}u; j = j + 1u) {
      let ds = p[j] * (dP[j] - rd);
      acc = fma(ds, qkv[(kCh + d) * ${nTok}u + j], acc);
    }
    dQKV[(qCh + d) * ${nTok}u + i] = acc;
  }
  mrow[i] = mx; drow[i] = den; rdot[i] = rd;
  workgroupBarrier();

  // ---- phase 2: thread = key/value token j ----
  let j = tid;
  var kj : array<f32, ${hd}>;
  var vj : array<f32, ${hd}>;
  for (var d = 0u; d < ${hd}u; d = d + 1u) {
    kj[d] = qkv[(kCh + d) * ${nTok}u + j];
    vj[d] = qkv[(vCh + d) * ${nTok}u + j];
  }
  var dV : array<f32, ${hd}>;
  var dK : array<f32, ${hd}>;
  for (var d = 0u; d < ${hd}u; d = d + 1u) { dV[d] = 0.0; dK[d] = 0.0; }
  for (var ii = 0u; ii < ${nTok}u; ii = ii + 1u) {
    // recompute p_ij and dP_ij for this (query ii, key j)
    var sc = 0.0;
    var dpij = 0.0;
    for (var d = 0u; d < ${hd}u; d = d + 1u) {
      let qv = qkv[(qCh + d) * ${nTok}u + ii];
      sc = fma(qv, kj[d], sc);
      dpij = fma(dO[(qCh + d) * ${nTok}u + ii], vj[d], dpij);
    }
    let pij = exp(sc - mrow[ii]) / drow[ii];
    let dsij = pij * (dpij - rdot[ii]);
    for (var d = 0u; d < ${hd}u; d = d + 1u) {
      dV[d] = fma(pij, dO[(qCh + d) * ${nTok}u + ii], dV[d]);
      dK[d] = fma(dsij, qkv[(qCh + d) * ${nTok}u + ii], dK[d]);
    }
  }
  for (var d = 0u; d < ${hd}u; d = d + 1u) {
    dQKV[(kCh + d) * ${nTok}u + j] = dK[d];
    dQKV[(vCh + d) * ${nTok}u + j] = dV[d];
  }
}`;
  return {
    label: `attn_core_bwd h${heads} n${nTok}`,
    code,
    workgroups: [heads, 1, 1],
    buffers: [gradSlot(s.savedQkv), gradSlot(s.dY), gradSlot(s.dX)],
  };
}

// ---------------------------------------------------------------------------
// Thin dispatcher — backward kind → emitter (one clean handler each).
// ---------------------------------------------------------------------------

export function bwdStepDispatch(step: BwdStep, opts: BwdDispatchOptions = {}): DispatchSpec {
  switch (step.kind) {
    case "loss_bwd":      return lossBwd(step);
    case "head_bwd":      return headBwd(step);
    case "gelu_bwd":      return geluBwd(step);
    case "pw_bwd":        return pwBwd(step);
    case "residual_bwd":  return residualBwd(step);
    case "spatial_bwd":   return opts.stemSpatialBwd && isStemSpatialBwd(step) ? spatialBwdStem4(step) : spatialBwd(step);
    case "se_bwd":        return seBwd(step);
    case "attn_core_bwd": return attnCoreBwd(step);
  }
}

/** All backward dispatches (loss head + reverse step list), in execution order. */
export function planBwdDispatches(plan: TrainPlan, opts: BwdDispatchOptions = {}): DispatchSpec[] {
  return plan.backward.map((step) => bwdStepDispatch(step, opts));
}
