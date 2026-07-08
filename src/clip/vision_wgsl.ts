/**
 * vision_wgsl — PURE WGSL codegen for the fused MobileCLIP-S0 vision encoder.
 *
 * The ONNX graph (518 nodes) is canonicalized offline by tools/clip/
 * compile_plan.py (κ) into a typed plan of FOUR step kinds — conv / se /
 * attention / head — plus one packed f32 weights blob. This module turns each
 * plan step into 1–3 fully-specialized compute dispatches: every shape,
 * stride and weight offset is baked into the shader source, so there are NO
 * uniforms and no runtime branching on structure. ~100 dispatches replace the
 * ~500 ONNX-op graph a standard runtime would issue (HANDOFF.md §2: cost is
 * dispatch-bound, and the CPU side of a pre-encoded pass is near-zero).
 *
 * ZERO imports on purpose: pure data → string, testable headless under
 * bun-webgpu (tools/clip/fused_test.ts) against per-step ORT goldens.
 *
 * Layout contract with κ (must not drift — the test suite pins it):
 *   activations : NCHW planar f32, batch 1 — x[c][y][x] at c*H*W + y*W + x
 *   pointwise W : TRANSPOSED [Cin][Cout] (vec4 over 4 consecutive couts)
 *   depthwise W : [C][k*k];  general W: [Cout][cpg][k][k]
 *   qkv scratch : [part(q,k,v)][head][token][dim] — column o of the ONNX qkv
 *                 matmul is exactly (part*C + head*32 + dim), so the matmul
 *                 writes it with no shuffle; q is pre-scaled by 1/sqrt(hd)
 *   attn scratch: [token][C] (token-major so proj reads rows contiguously)
 *   GELU        : exact-erf form (Abramowitz–Stegun 7.1.26, |err| < 1.5e-7),
 *                 matching the ONNX x·0.5·(1+erf(x/√2)) decomposition
 */

// ---------------------------------------------------------------------------
// Canonical types (mirror plan.json — κ is the only producer)
// ---------------------------------------------------------------------------

export interface ConvStep {
  kind: "conv";
  variant: "pointwise" | "depthwise" | "general";
  name: string;
  cin: number; cout: number; k: number; stride: number; pad: number;
  groups: number; h: number; w: number; outH: number; outW: number;
  src: number; dst: number; wOff: number; bOff: number;
  act: "none" | "gelu";
  residual: number | null;
  layerScaleOff: number | null;
  /** ONNX tensor to verify against; null for attention internals (their
   *  layout is ours, not ONNX's — the block-output step covers them). */
  ref: string | null;
}

export interface SeStep {
  kind: "se";
  name: string;
  c: number; cmid: number; h: number; w: number;
  src: number; dst: number;
  w1Off: number; b1Off: number; w2Off: number; b2Off: number;
  act: "none" | "gelu";
  ref: string;
}

/** softmax(QKᵀ)V per head. qkv (src) and attnOut (dst) are channel-planar:
 *  qkv channel o = part*C + head*hd + d; attnOut channel = head*hd + d. The
 *  surrounding matmuls are plain pointwise ConvSteps (BN + q-scale folded
 *  into their weights by κ). */
export interface AttnCoreStep {
  kind: "attn_core";
  name: string;
  c: number; heads: number; hd: number; nTok: number;
  src: number; dst: number;
  ref: string | null;
}

export interface HeadStep {
  kind: "head";
  name: string;
  cin: number; cout: number; h: number; w: number;
  src: number; dst: number; wOff: number;
  ref: string;
}

/** Train-mode split-GELU: a standalone elementwise activation. Inference fuses
 *  GELU into the conv/se `act` flag; the train plan splits it out so the
 *  pre-activation is saved for gelu-backward (docs/clip_backward_spec.md §1). */
export interface GeluStep {
  kind: "gelu";
  name: string;
  src: number; dst: number; n: number;
  ref: string;
}

export type VisionStep = ConvStep | SeStep | AttnCoreStep | HeadStep | GeluStep;

export interface VisionPlan {
  model: string;
  inputSlot: number;
  inputShape: [number, number, number];
  outputSlot: number;
  embedDim: number;
  slots: number[];          // sizes in floats
  weightsFloats: number;
  steps: VisionStep[];
}

/** One generated compute dispatch: shader + grid + ordered buffer bindings.
 *  "text" is the per-prompt target text embedding (backward loss head only). */
export type BufferRef =
  | { kind: "weights" }
  | { kind: "slot"; slot: number }
  | { kind: "text" };

export interface DispatchSpec {
  label: string;
  code: string;
  workgroups: [number, number, number];
  /** binding i in the shader = buffers[i] */
  buffers: BufferRef[];
}

export type WeightPrecision = "f32" | "f16";
export type PointwiseTileVariant = "default" | "rect8x16";

export interface DispatchOptions {
  weightPrecision?: WeightPrecision;
  pointwiseTileVariant?: PointwiseTileVariant;
  pointwiseTileSteps?: ReadonlySet<number>;
}

// ---------------------------------------------------------------------------
// Shared WGSL fragments
// ---------------------------------------------------------------------------

/** Weight storage precision is a runtime/compiler option; math stays f32. */
export const weightsDecl = (binding: number, precision: WeightPrecision = "f32") =>
  precision === "f16"
    ? `enable f16;\n` +
      `@group(0) @binding(${binding}) var<storage, read> weights : array<vec4<f16>>;\n` +
      `fn W(i : u32) -> f32 { return f32(weights[i >> 2u][i & 3u]); }\n` +
      `fn W4(i : u32) -> vec4f { return vec4f(weights[i]); }`
    : `@group(0) @binding(${binding}) var<storage, read> weights : array<vec4f>;\n` +
      `fn W(i : u32) -> f32 { return weights[i >> 2u][i & 3u]; }\n` +
      `fn W4(i : u32) -> vec4f { return weights[i]; }`;

/** erf via Abramowitz–Stegun 7.1.26 (|abs err| ≤ 1.5e-7 — well inside the
 *  1e-3 verification tolerance); gelu matches ONNX's exact-erf decomposition.
 *  Exported so the backward file (gelu-backward needs the same erf) shares the
 *  EXACT constants — a drift would desync forward/backward silently. */
export const GELU = /* wgsl */ `
fn erf1(x : f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
fn gelu1(x : f32) -> f32 { return 0.5 * x * (1.0 + erf1(x * 0.7071067811865476)); }
fn erf4(x : vec4f) -> vec4f {
  let s = sign(x);
  let a = abs(x);
  let t = vec4f(1.0) / (vec4f(1.0) + 0.3275911 * a);
  let y = vec4f(1.0) - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
fn gelu4(x : vec4f) -> vec4f { return 0.5 * x * (vec4f(1.0) + erf4(x * 0.7071067811865476)); }
`;

export function assertStep(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`vision_wgsl: ${msg}`);
}

// ---------------------------------------------------------------------------
// conv:pointwise — 1x1, groups=1: a [P,Cin]·[Cin,Cout] matmul, the FLOPs bulk
// of the network (fc1/fc2 of every ConvFFN). Classic shared-memory tiling:
// each workgroup (8×8 threads) owns a 32-pixel × 32-cout tile; x and W are
// staged through workgroup memory in 32-deep ci chunks, so each global float
// is read ONCE per tile instead of once per thread (~8× traffic cut vs the
// naive version, which was memory-bound at multi-GB/forward). Every plan
// shape satisfies cin%32==0, cout%32==0, P%32==0 — asserted loudly; a model
// that violates it needs a new handler, not a hidden slow path.
// ---------------------------------------------------------------------------

/** Tile decls shared by the forward pointwise and pw_bwd (32 ci × 8 pixel-quads
 *  for xS, 32 ci × 8 cout-quads for wS, 4KB each). */
export const PW_TILE_DECLS = /* wgsl */ `
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 256>;`;

export const PW_RECT8X16_TILE_DECLS = /* wgsl */ `
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 512>;`;

/** The shared tiled-matmul body: out[co][p] = Σ_ci src[ci][p]·W[ci*cout+co].
 *  Produces acc0..acc3 (vec4 = 4 pixels × 4 couts) then stores. `init` seeds
 *  each acc (bias for fwd, 0 for bwd); `store` maps acc{j} → the value written
 *  to dst[(co+j)*P4+p4] (gelu/residual epilogue for fwd, add-into for bwd
 *  accumulate). Requires src(binding 1, array<vec4f> [Cin][P4]), dst(binding 2),
 *  weights(binding 0), and PW_TILE_DECLS in scope. */
export function pointwiseTiledMain(o: {
  cin: number; cout: number; P4: number; wOff: number;
  init: (j: number) => string;
  store: (j: number) => string;
  loadSrc?: (index: string) => string;
  extraStore?: (j: number) => string;
}): string {
  const extra = (j: number) => o.extraStore ? `\n  ${o.extraStore(j)}` : "";
  const loadSrc = o.loadSrc ? o.loadSrc("srcIndex") : "src[srcIndex]";
  return /* wgsl */ `
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let p4 = wid.x * 8u + lid.x;          // this thread's pixel-quad
  let co = (wid.y * 8u + lid.y) * 4u;   // this thread's first cout
  let p4base = wid.x * 8u;
  let cobase = wid.y * 32u;
  var acc0 = ${o.init(0)};
  var acc1 = ${o.init(1)};
  var acc2 = ${o.init(2)};
  var acc3 = ${o.init(3)};
  for (var ci0 = 0u; ci0 < ${o.cin}u; ci0 = ci0 + 32u) {
    // stage: 256 vec4s each of x and W, 4 per thread
    for (var t = li; t < 256u; t = t + 64u) {
      let ci = t >> 3u;
      let lane = t & 7u;
      let srcIndex = (ci0 + ci) * ${o.P4}u + p4base + lane;
      xS[t] = ${loadSrc};
      wS[t] = W4((${o.wOff}u + (ci0 + ci) * ${o.cout}u + cobase + lane * 4u) / 4u);
    }
    workgroupBarrier();
    for (var ci = 0u; ci < 32u; ci = ci + 1u) {
      let xv = xS[ci * 8u + lid.x];
      let wv = wS[ci * 8u + lid.y];
      acc0 = fma(vec4f(wv.x), xv, acc0);
      acc1 = fma(vec4f(wv.y), xv, acc1);
      acc2 = fma(vec4f(wv.z), xv, acc2);
      acc3 = fma(vec4f(wv.w), xv, acc3);
    }
    workgroupBarrier();
  }
  dst[co * ${o.P4}u + p4] = ${o.store(0)};${extra(0)}
  dst[(co + 1u) * ${o.P4}u + p4] = ${o.store(1)};${extra(1)}
  dst[(co + 2u) * ${o.P4}u + p4] = ${o.store(2)};${extra(2)}
  dst[(co + 3u) * ${o.P4}u + p4] = ${o.store(3)};${extra(3)}
}`;
}

/** Assert a pointwise-shaped step satisfies the tile constraints. Shared so the
 *  backward reuses the SAME loud guard (a violating shape needs a handler). */
export function assertPointwiseTiles(name: string, cin: number, cout: number, P: number, wOff: number): void {
  assertStep(
    P % 32 === 0 && cout % 32 === 0 && cin % 32 === 0,
    `${name}: tiled pointwise needs P%32==0 && cout%32==0 && cin%32==0 (got P=${P} cin=${cin} cout=${cout})`
  );
  assertStep(wOff % 4 === 0, `${name}: wOff not 16B-aligned`);
}

function useRect8x16Pointwise(opts: DispatchOptions, stepIndex?: number): boolean {
  if (opts.pointwiseTileVariant !== "rect8x16") return false;
  const steps = opts.pointwiseTileSteps;
  if (!steps?.size) return true;
  return stepIndex !== undefined && steps.has(stepIndex);
}

function assertPointwiseRect8x16(name: string, cin: number, cout: number, P: number, wOff: number): void {
  assertPointwiseTiles(name, cin, cout, P, wOff);
  assertStep(cout % 64 === 0, `${name}: rect8x16 pointwise needs cout%64==0 (got cout=${cout})`);
}

function pointwiseRect8x16Main(o: {
  cin: number; cout: number; P4: number; wOff: number;
  init: (j: number) => string;
  store: (j: number) => string;
  extraStore?: (j: number) => string;
}): string {
  const extra = (j: number) => o.extraStore ? `\n  ${o.extraStore(j)}` : "";
  return /* wgsl */ `
@compute @workgroup_size(8, 16)
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let p4 = wid.x * 8u + lid.x;          // this thread's pixel-quad
  let co = (wid.y * 16u + lid.y) * 4u;  // this thread's first cout
  let p4base = wid.x * 8u;
  let cobase = wid.y * 64u;
  var acc0 = ${o.init(0)};
  var acc1 = ${o.init(1)};
  var acc2 = ${o.init(2)};
  var acc3 = ${o.init(3)};
  for (var ci0 = 0u; ci0 < ${o.cin}u; ci0 = ci0 + 32u) {
    // stage: x tile is 32 ci x 8 pixel-quads; W tile is 32 ci x 16 cout-quads
    for (var t = li; t < 256u; t = t + 128u) {
      let ci = t >> 3u;
      let lane = t & 7u;
      xS[t] = src[(ci0 + ci) * ${o.P4}u + p4base + lane];
    }
    for (var t = li; t < 512u; t = t + 128u) {
      let ci = t >> 4u;
      let lane = t & 15u;
      wS[t] = W4((${o.wOff}u + (ci0 + ci) * ${o.cout}u + cobase + lane * 4u) / 4u);
    }
    workgroupBarrier();
    for (var ci = 0u; ci < 32u; ci = ci + 1u) {
      let xv = xS[ci * 8u + lid.x];
      let wv = wS[ci * 16u + lid.y];
      acc0 = fma(vec4f(wv.x), xv, acc0);
      acc1 = fma(vec4f(wv.y), xv, acc1);
      acc2 = fma(vec4f(wv.z), xv, acc2);
      acc3 = fma(vec4f(wv.w), xv, acc3);
    }
    workgroupBarrier();
  }
  dst[co * ${o.P4}u + p4] = ${o.store(0)};${extra(0)}
  dst[(co + 1u) * ${o.P4}u + p4] = ${o.store(1)};${extra(1)}
  dst[(co + 2u) * ${o.P4}u + p4] = ${o.store(2)};${extra(2)}
  dst[(co + 3u) * ${o.P4}u + p4] = ${o.store(3)};${extra(3)}
}`;
}

function pointwise(s: ConvStep, opts: DispatchOptions = {}, stepIndex?: number): DispatchSpec {
  if (useRect8x16Pointwise(opts, stepIndex)) return pointwiseRect8x16(s, opts);
  const P = s.outH * s.outW;
  assertPointwiseTiles(s.name, s.cin, s.cout, P, s.wOff);
  const P4 = P / 4;
  const hasRes = s.residual !== null;
  assertStep((s.layerScaleOff !== null) === hasRes, `${s.name}: layerScale without residual`);
  const buffers: BufferRef[] = [
    { kind: "weights" },
    { kind: "slot", slot: s.src },
    { kind: "slot", slot: s.dst },
  ];
  if (hasRes) buffers.push({ kind: "slot", slot: s.residual as number });

  const post = (j: number): string => {
    const a = s.act === "gelu" ? `gelu4(acc${j})` : `acc${j}`;
    if (!hasRes) return a;
    return `res[(co + ${j}u) * ${P4}u + p4] + vec4f(W(${s.layerScaleOff}u + co + ${j}u)) * ${a}`;
  };

  const code = /* wgsl */ `
${weightsDecl(0, opts.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${hasRes ? `@group(0) @binding(3) var<storage, read> res : array<vec4f>;` : ``}
${GELU}
${PW_TILE_DECLS}
${pointwiseTiledMain({
    cin: s.cin, cout: s.cout, P4, wOff: s.wOff,
    init: (j) => `vec4f(W(${s.bOff}u + co + ${j}u))`,
    store: post,
  })}`;
  return {
    label: `pw ${s.cin}->${s.cout} @${s.outH}x${s.outW}`,
    code,
    workgroups: [P4 / 8, s.cout / 32, 1],
    buffers,
  };
}

function pointwiseRect8x16(s: ConvStep, opts: DispatchOptions = {}): DispatchSpec {
  const P = s.outH * s.outW;
  assertPointwiseRect8x16(s.name, s.cin, s.cout, P, s.wOff);
  const P4 = P / 4;
  const hasRes = s.residual !== null;
  assertStep((s.layerScaleOff !== null) === hasRes, `${s.name}: layerScale without residual`);
  const buffers: BufferRef[] = [
    { kind: "weights" },
    { kind: "slot", slot: s.src },
    { kind: "slot", slot: s.dst },
  ];
  if (hasRes) buffers.push({ kind: "slot", slot: s.residual as number });

  const post = (j: number): string => {
    const a = s.act === "gelu" ? `gelu4(acc${j})` : `acc${j}`;
    if (!hasRes) return a;
    return `res[(co + ${j}u) * ${P4}u + p4] + vec4f(W(${s.layerScaleOff}u + co + ${j}u)) * ${a}`;
  };

  const code = /* wgsl */ `
${weightsDecl(0, opts.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${hasRes ? `@group(0) @binding(3) var<storage, read> res : array<vec4f>;` : ``}
${GELU}
${PW_RECT8X16_TILE_DECLS}
${pointwiseRect8x16Main({
    cin: s.cin, cout: s.cout, P4, wOff: s.wOff,
    init: (j) => `vec4f(W(${s.bOff}u + co + ${j}u))`,
    store: post,
  })}`;
  return {
    label: `pw rect8x16 ${s.cin}->${s.cout} @${s.outH}x${s.outW}`,
    code,
    workgroups: [P4 / 8, s.cout / 64, 1],
    buffers,
  };
}

export function pointwiseFusedGelu(
  s: ConvStep,
  gelu: GeluStep,
  opts: DispatchOptions = {},
  stepIndex?: number
): DispatchSpec {
  if (useRect8x16Pointwise(opts, stepIndex)) return pointwiseRect8x16FusedGelu(s, gelu, opts);
  assertStep(s.variant === "pointwise", `${s.name}: fused GELU only supports pointwise conv`);
  assertStep(s.act === "none", `${s.name}: fused GELU expects split train-mode conv`);
  assertStep(s.residual === null && s.layerScaleOff === null, `${s.name}: fused GELU does not support residual epilogues`);
  assertStep(gelu.src === s.dst, `${s.name}: fused GELU src slot ${gelu.src} != conv dst ${s.dst}`);
  const P = s.outH * s.outW;
  assertStep(gelu.n === s.cout * P, `${s.name}: fused GELU n=${gelu.n} != cout*P=${s.cout * P}`);
  assertPointwiseTiles(s.name, s.cin, s.cout, P, s.wOff);
  const P4 = P / 4;
  const code = /* wgsl */ `
${weightsDecl(0, opts.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> geluDst : array<vec4f>;
${GELU}
${PW_TILE_DECLS}
${pointwiseTiledMain({
    cin: s.cin, cout: s.cout, P4, wOff: s.wOff,
    init: (j) => `vec4f(W(${s.bOff}u + co + ${j}u))`,
    store: (j) => `acc${j}`,
    extraStore: (j) => `geluDst[(co + ${j}u) * ${P4}u + p4] = gelu4(acc${j});`,
  })}`;
  return {
    label: `pw+gelu ${s.cin}->${s.cout} @${s.outH}x${s.outW}`,
    code,
    workgroups: [P4 / 8, s.cout / 32, 1],
    buffers: [
      { kind: "weights" },
      { kind: "slot", slot: s.src },
      { kind: "slot", slot: s.dst },
      { kind: "slot", slot: gelu.dst },
    ],
  };
}

function pointwiseRect8x16FusedGelu(
  s: ConvStep,
  gelu: GeluStep,
  opts: DispatchOptions = {}
): DispatchSpec {
  assertStep(s.variant === "pointwise", `${s.name}: fused GELU only supports pointwise conv`);
  assertStep(s.act === "none", `${s.name}: fused GELU expects split train-mode conv`);
  assertStep(s.residual === null && s.layerScaleOff === null, `${s.name}: fused GELU does not support residual epilogues`);
  assertStep(gelu.src === s.dst, `${s.name}: fused GELU src slot ${gelu.src} != conv dst ${s.dst}`);
  const P = s.outH * s.outW;
  assertStep(gelu.n === s.cout * P, `${s.name}: fused GELU n=${gelu.n} != cout*P=${s.cout * P}`);
  assertPointwiseRect8x16(s.name, s.cin, s.cout, P, s.wOff);
  const P4 = P / 4;
  const code = /* wgsl */ `
${weightsDecl(0, opts.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> geluDst : array<vec4f>;
${GELU}
${PW_RECT8X16_TILE_DECLS}
${pointwiseRect8x16Main({
    cin: s.cin, cout: s.cout, P4, wOff: s.wOff,
    init: (j) => `vec4f(W(${s.bOff}u + co + ${j}u))`,
    store: (j) => `acc${j}`,
    extraStore: (j) => `geluDst[(co + ${j}u) * ${P4}u + p4] = gelu4(acc${j});`,
  })}`;
  return {
    label: `pw+gelu rect8x16 ${s.cin}->${s.cout} @${s.outH}x${s.outW}`,
    code,
    workgroups: [P4 / 8, s.cout / 64, 1],
    buffers: [
      { kind: "weights" },
      { kind: "slot", slot: s.src },
      { kind: "slot", slot: s.dst },
      { kind: "slot", slot: gelu.dst },
    ],
  };
}

// ---------------------------------------------------------------------------
// conv:depthwise — k∈{3,7}, groups=C. Thread = one output pixel of one channel.
// ---------------------------------------------------------------------------

function spatialConv(s: ConvStep, opts: DispatchOptions = {}): DispatchSpec {
  assertStep(s.residual === null && s.layerScaleOff === null,
    `${s.name}: spatial conv never carries residual in this plan`);
  assertStep(s.outW % 4 === 0, `${s.name}: spatial tiling needs outW%4==0`);
  const P = s.outH * s.outW;
  const Q = P / 4;                          // horizontal 4-pixel tiles
  const K = s.k, S = s.stride, PAD = s.pad;
  const NIN = 3 * S + K;                    // input cols feeding one tile row
  const cpg = s.cin / s.groups;             // input channels per group (1 or 3)
  const cpgOut = s.cout / s.groups;
  const WK = cpg * K * K;                   // this cout's weight footprint
  assertStep(Number.isInteger(cpg) && Number.isInteger(cpgOut), `${s.name}: bad groups`);
  assertStep(WK <= 64, `${s.name}: weight tile ${WK} exceeds one staging round`);
  const act = (e: string) => (s.act === "gelu" ? `gelu1(${e})` : e);
  // Interior tap loops are FULLY UNROLLED at codegen with literal indices:
  // a private row array indexed by a loop variable spills to device scratch
  // on Metal (dynamic indexing), which cost ~2× on the dw7 layers. Unrolled,
  // every row value is a named register.
  const interior: string[] = [];
  for (let c = 0; c < cpg; c++) {
    interior.push(`    { let base = (ci0 + ${c}u) * ${s.h * s.w}u;`);
    for (let ky = 0; ky < K; ky++) {
      interior.push(`      { let rowBase = base + u32(iy0 + ${ky}) * ${s.w}u + u32(ix0);`);
      for (let i = 0; i < NIN; i++) {
        interior.push(`        let r${i} = src[rowBase + ${i}u];`);
      }
      for (let kx = 0; kx < K; kx++) {
        interior.push(
          `        acc = fma(vec4f(r${kx}, r${S + kx}, r${2 * S + kx}, r${3 * S + kx}), ` +
          `vec4f(wk[${c * K * K + ky * K + kx}u]), acc);`
        );
      }
      interior.push(`      }`);
    }
    interior.push(`    }`);
  }
  // One workgroup = one output channel: its cpg·k·k weights are staged once
  // in shared memory (depthwise is just cpg=1). Each thread produces 4
  // horizontal pixels, loading each input row segment once (NIN loads for
  // 4·K taps ≈ 2.8× fewer for k=7). Interior tiles (the vast majority) take
  // a single-branch unchecked path.
  const code = /* wgsl */ `
${weightsDecl(0, opts.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
${GELU}
var<workgroup> wk : array<f32, ${WK}>;
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let co = gid.y;
  if (li < ${WK}u) { wk[li] = W(${s.wOff}u + co * ${WK}u + li); }
  workgroupBarrier();
  let q = gid.x;
  if (q >= ${Q}u) { return; }
  let oy = i32(q / ${s.outW / 4}u);
  let ox0 = i32(q % ${s.outW / 4}u) * 4;
  let ci0 = (co / ${cpgOut}u) * ${cpg}u;   // first input channel of co's group
  let iy0 = oy * ${S} - ${PAD};
  let ix0 = ox0 * ${S} - ${PAD};
  var acc = vec4f(W(${s.bOff}u + co));
  if (iy0 >= 0 && iy0 + ${K} <= ${s.h} && ix0 >= 0 && ix0 + ${NIN} <= ${s.w}) {
    // interior: every tap in bounds, unchecked unrolled register loads
${interior.join("\n")}
  } else {
    // border: per-tap bounds checks (zero padding)
    for (var c = 0u; c < ${cpg}u; c = c + 1u) {
      let base = (ci0 + c) * ${s.h * s.w}u;
      for (var ky = 0; ky < ${K}; ky = ky + 1) {
        let iy = iy0 + ky;
        if (iy < 0 || iy >= ${s.h}) { continue; }
        let rowBase = base + u32(iy) * ${s.w}u;
        for (var kx = 0; kx < ${K}; kx = kx + 1) {
          let wv = wk[c * ${K * K}u + u32(ky * ${K} + kx)];
          var xv = vec4f(0.0);
          for (var j = 0; j < 4; j = j + 1) {
            let ix = ix0 + j * ${S} + kx;
            if (ix >= 0 && ix < ${s.w}) { xv[j] = src[rowBase + u32(ix)]; }
          }
          acc = fma(xv, vec4f(wv), acc);
        }
      }
    }
  }
  let out = co * ${P}u + u32(oy) * ${s.outW}u + u32(ox0);
  dst[out] = ${act("acc.x")};
  dst[out + 1u] = ${act("acc.y")};
  dst[out + 2u] = ${act("acc.z")};
  dst[out + 3u] = ${act("acc.w")};
}`;
  return {
    label: `conv${s.k} ${s.cin}->${s.cout} g${s.groups} @${s.outH}x${s.outW}`,
    code,
    workgroups: [Math.ceil(Q / 64), s.cout, 1],
    buffers: [
      { kind: "weights" },
      { kind: "slot", slot: s.src },
      { kind: "slot", slot: s.dst },
    ],
  };
}

// ---------------------------------------------------------------------------
// se — the whole gate in ONE dispatch, one workgroup: GAP → fc1+relu →
// fc2+sigmoid → scale (+ fused gelu). Tensors are tiny (≤1024ch × ≤16x16).
// ---------------------------------------------------------------------------

const SE_WG = 256;

function seStep(s: SeStep, opts: DispatchOptions = {}): DispatchSpec {
  const P = s.h * s.w;
  assertStep(s.c <= 2048 && s.cmid <= 512, `${s.name}: SE dims exceed shared-memory plan`);
  const act = (e: string) => (s.act === "gelu" ? `gelu1(${e})` : e);
  const code = /* wgsl */ `
${weightsDecl(0, opts.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
${GELU}
var<workgroup> gap : array<f32, ${s.c}>;
var<workgroup> mid : array<f32, ${s.cmid}>;
var<workgroup> scl : array<f32, ${s.c}>;
@compute @workgroup_size(${SE_WG})
fn main(@builtin(local_invocation_index) li : u32) {
  for (var c = li; c < ${s.c}u; c = c + ${SE_WG}u) {
    var sum = 0.0;
    for (var p = 0u; p < ${P}u; p = p + 1u) { sum = sum + src[c * ${P}u + p]; }
    gap[c] = sum / ${P}.0;
  }
  workgroupBarrier();
  for (var m = li; m < ${s.cmid}u; m = m + ${SE_WG}u) {
    var sum = W(${s.b1Off}u + m);
    for (var c = 0u; c < ${s.c}u; c = c + 1u) {
      sum = fma(gap[c], W(${s.w1Off}u + m * ${s.c}u + c), sum);
    }
    mid[m] = max(sum, 0.0);
  }
  workgroupBarrier();
  for (var c = li; c < ${s.c}u; c = c + ${SE_WG}u) {
    var sum = W(${s.b2Off}u + c);
    for (var m = 0u; m < ${s.cmid}u; m = m + 1u) {
      sum = fma(mid[m], W(${s.w2Off}u + c * ${s.cmid}u + m), sum);
    }
    scl[c] = 1.0 / (1.0 + exp(-sum));
  }
  workgroupBarrier();
  for (var i = li; i < ${s.c * P}u; i = i + ${SE_WG}u) {
    dst[i] = ${act(`src[i] * scl[i / ${P}u]`)};
  }
}`;
  return {
    label: `se c${s.c} mid${s.cmid} @${s.h}x${s.w}`,
    code,
    workgroups: [1, 1, 1],
    buffers: [
      { kind: "weights" },
      { kind: "slot", slot: s.src },
      { kind: "slot", slot: s.dst },
    ],
  };
}

// ---------------------------------------------------------------------------
// attn_core — per-head softmax(QKᵀ)V, K then V staged through shared memory.
// The qkv and proj matmuls around it are ordinary pointwise ConvSteps (κ
// folds BN + q-scale into the qkv weights), so this kernel is pure attention:
// one workgroup per head, one thread per query token, score row in registers.
// ---------------------------------------------------------------------------

function attnCore(s: AttnCoreStep): DispatchSpec {
  const { nTok, hd, heads, c } = s;
  const D4 = hd / 4;               // vec4 tiles per head-dim
  const KV4 = (nTok * hd) / 4;     // one head's K (or V) in vec4s
  assertStep(c === heads * hd, `${s.name}: c != heads*hd`);
  assertStep(nTok <= 256 && KV4 * 16 <= 16384, `${s.name}: K/V won't fit shared memory`);
  // channel-planar addressing: channel o at token n sits at o*nTok + n
  const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> qkv : array<f32>;
@group(0) @binding(1) var<storage, read_write> attnOut : array<f32>;
var<workgroup> kv : array<vec4f, ${KV4}>;   // K, then reused for V; [j][d4]
@compute @workgroup_size(${nTok})
fn main(@builtin(local_invocation_index) i : u32,
        @builtin(workgroup_id) wid : vec3u) {
  let head = wid.x;
  let qCh = head * ${hd}u;                      // q channels [qCh, qCh+hd)
  let kCh = ${c}u + head * ${hd}u;
  let vCh = ${2 * c}u + head * ${hd}u;
  // gather this thread's query row into registers (one-time strided reads)
  var q : array<vec4f, ${D4}>;
  for (var d4 = 0u; d4 < ${D4}u; d4 = d4 + 1u) {
    q[d4] = vec4f(
      qkv[(qCh + d4 * 4u) * ${nTok}u + i],
      qkv[(qCh + d4 * 4u + 1u) * ${nTok}u + i],
      qkv[(qCh + d4 * 4u + 2u) * ${nTok}u + i],
      qkv[(qCh + d4 * 4u + 3u) * ${nTok}u + i]);
  }
  for (var t = i; t < ${KV4}u; t = t + ${nTok}u) {
    let j = t / ${D4}u;
    let d = (t % ${D4}u) * 4u;
    kv[t] = vec4f(
      qkv[(kCh + d) * ${nTok}u + j],
      qkv[(kCh + d + 1u) * ${nTok}u + j],
      qkv[(kCh + d + 2u) * ${nTok}u + j],
      qkv[(kCh + d + 3u) * ${nTok}u + j]);
  }
  workgroupBarrier();
  var p : array<f32, ${nTok}>;   // row i of the score matrix, private
  var m = -3.0e38;
  for (var j = 0u; j < ${nTok}u; j = j + 1u) {
    var sv = vec4f(0.0);
    for (var d4 = 0u; d4 < ${D4}u; d4 = d4 + 1u) {
      sv = fma(q[d4], kv[j * ${D4}u + d4], sv);
    }
    let sc = sv.x + sv.y + sv.z + sv.w;
    p[j] = sc;
    m = max(m, sc);
  }
  var sum = 0.0;
  for (var j = 0u; j < ${nTok}u; j = j + 1u) {
    let e = exp(p[j] - m);
    p[j] = e;
    sum = sum + e;
  }
  let inv = 1.0 / sum;
  workgroupBarrier();   // everyone done with K before it becomes V
  for (var t = i; t < ${KV4}u; t = t + ${nTok}u) {
    let j = t / ${D4}u;
    let d = (t % ${D4}u) * 4u;
    kv[t] = vec4f(
      qkv[(vCh + d) * ${nTok}u + j],
      qkv[(vCh + d + 1u) * ${nTok}u + j],
      qkv[(vCh + d + 2u) * ${nTok}u + j],
      qkv[(vCh + d + 3u) * ${nTok}u + j]);
  }
  workgroupBarrier();
  var acc : array<vec4f, ${D4}>;
  for (var j = 0u; j < ${nTok}u; j = j + 1u) {
    let wgt = p[j] * inv;
    for (var d4 = 0u; d4 < ${D4}u; d4 = d4 + 1u) {
      acc[d4] = fma(vec4f(wgt), kv[j * ${D4}u + d4], acc[d4]);
    }
  }
  // attnOut is channel-planar [head*hd + d][n] — pointwise-conv input layout
  for (var d4 = 0u; d4 < ${D4}u; d4 = d4 + 1u) {
    attnOut[(qCh + d4 * 4u) * ${nTok}u + i] = acc[d4].x;
    attnOut[(qCh + d4 * 4u + 1u) * ${nTok}u + i] = acc[d4].y;
    attnOut[(qCh + d4 * 4u + 2u) * ${nTok}u + i] = acc[d4].z;
    attnOut[(qCh + d4 * 4u + 3u) * ${nTok}u + i] = acc[d4].w;
  }
}`;
  return {
    label: `attn.core h${heads} n${nTok}`,
    code,
    workgroups: [heads, 1, 1],
    buffers: [
      { kind: "slot", slot: s.src },
      { kind: "slot", slot: s.dst },
    ],
  };
}

// ---------------------------------------------------------------------------
// head — GAP over 8x8 + the 1024→512 projection, one workgroup.
// ---------------------------------------------------------------------------

const HEAD_WG = 256;

function headStep(s: HeadStep, opts: DispatchOptions = {}): DispatchSpec {
  const P = s.h * s.w;
  const code = /* wgsl */ `
${weightsDecl(0, opts.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
var<workgroup> gap : array<f32, ${s.cin}>;
@compute @workgroup_size(${HEAD_WG})
fn main(@builtin(local_invocation_index) li : u32) {
  for (var ci = li; ci < ${s.cin}u; ci = ci + ${HEAD_WG}u) {
    var sum = 0.0;
    for (var p = 0u; p < ${P}u; p = p + 1u) { sum = sum + src[ci * ${P}u + p]; }
    gap[ci] = sum / ${P}.0;
  }
  workgroupBarrier();
  for (var co = li; co < ${s.cout}u; co = co + ${HEAD_WG}u) {
    var acc = 0.0;
    for (var ci = 0u; ci < ${s.cin}u; ci = ci + 1u) {
      acc = fma(gap[ci], W(${s.wOff}u + ci * ${s.cout}u + co), acc);
    }
    dst[co] = acc;
  }
}`;
  return {
    label: `head ${s.cin}->${s.cout}`,
    code,
    workgroups: [1, 1, 1],
    buffers: [
      { kind: "weights" },
      { kind: "slot", slot: s.src },
      { kind: "slot", slot: s.dst },
    ],
  };
}

// ---------------------------------------------------------------------------
// gelu — train-mode standalone elementwise activation (vec4, one thread / quad).
// ---------------------------------------------------------------------------

const GELU_WG = 64;

function geluStep(s: GeluStep): DispatchSpec {
  assertStep(s.n % 4 === 0, `${s.name}: gelu n%4 != 0`);
  const n4 = s.n / 4;
  const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;
${GELU}
@compute @workgroup_size(${GELU_WG})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${n4}u) { return; }
  dst[i] = gelu4(src[i]);
}`;
  return {
    label: `gelu n${s.n}`,
    code,
    workgroups: [Math.ceil(n4 / GELU_WG), 1, 1],
    buffers: [
      { kind: "slot", slot: s.src },
      { kind: "slot", slot: s.dst },
    ],
  };
}

// ---------------------------------------------------------------------------
// Thin dispatchers — step kind → dispatch list; conv variant → emitter.
// ---------------------------------------------------------------------------

function convDispatches(s: ConvStep, opts: DispatchOptions = {}, stepIndex?: number): DispatchSpec[] {
  switch (s.variant) {
    case "pointwise": return [pointwise(s, opts, stepIndex)];
    case "depthwise": return [spatialConv(s, opts)];  // depthwise = spatial, cpg=1
    case "general":   return [spatialConv(s, opts)];
  }
}

export function stepDispatches(step: VisionStep, opts: DispatchOptions = {}, stepIndex?: number): DispatchSpec[] {
  switch (step.kind) {
    case "conv":      return convDispatches(step, opts, stepIndex);
    case "se":        return [seStep(step, opts)];
    case "attn_core": return [attnCore(step)];
    case "head":      return [headStep(step, opts)];
    case "gelu":      return [geluStep(step)];
  }
}

/** All dispatches for a full forward pass, in execution order. */
export function planDispatches(plan: VisionPlan, opts: DispatchOptions = {}): DispatchSpec[] {
  return plan.steps.flatMap((step, index) => stepDispatches(step, opts, index));
}
