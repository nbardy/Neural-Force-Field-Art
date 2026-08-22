/**
 * pixel_disc_wgsl — PURE WGSL codegen for four pixel-space GAN kinds.
 *
 * Reverse-mode only (no JVP). Soft bilinear splat + conv→codebook critic;
 * generator reward chains through virtual positions into field weights via
 * extGrads (same seam as the relational adversary).
 *
 * Kinds (CPU oracle src/core/gan/pixel_disc.ts, spec docs/PIXEL_DISC.md):
 *   vec-field   per-cell 2-vector vs F(cell center)
 *   next-frame  D0 → predict D1
 *   real-fake   GAP+MLP logit, real vs random-position fake
 *   inpaint     masked completion on random block
 *
 * densPack layout (4 × nCell f32, nCell = G²):
 *   [0, n)     dens   — current working density
 *   [n, 2n)    dDens  — generator ∂L/∂D (VJP output)
 *   [2n, 3n)   auxA   — D0 / forceX / fake dens (kind-dependent)
 *   [3n, 4n)   auxB   — forceY / inpaint mask (0|1 f32)
 *
 * Trainer must allocate densPack with densPackFloats(G) floats (4×G²).
 * See PIXEL_DISC_KIND_PASSES for per-kind pass order.
 *
 * criticDisc/criticGen are workgroup-parallel (one cell per invocation for the
 * activations, one WEIGHT per invocation for the gradients), dispatched as a
 * SINGLE workgroup so the phases can be separated by barriers. Their per-cell
 * workspace lives in the tail of `scratch` — see pixelScratchBytes.
 */

import { type FieldLayout, type HeadSpec } from "./advect_wgsl";
import {
  emitEncode,
  emitFwdStore,
  emitBwdStore,
  trainScratchLayout,
  COMMON,
} from "./train_wgsl";
import { f32lit } from "./ad/emit_wgsl";
import {
  PIXEL_DISC_SOFT_EPS,
  PIXEL_DISC_DEFAULTS,
  pixelDiscWeightCount,
  type PixelDiscDims,
  type PixelGanKind,
} from "../../core/gan/pixel_disc";

export const PIXEL_DISC_WG = 256;
export const PIXEL_DISC_MAX_BATCH = 512;
/** Fixed-point scale for atomic density accumulation (energy 1 ≡ SCALE). */
export const PIXEL_DISC_DENS_SCALE = 1e6;

const DENS_FLOOR = 1e-3;

/**
 * `critMeta` stats region — the tail after grads|m|v, based at 3·nWeights.
 *
 * Every named slot here is written by SOME kernel. That invariant is the whole
 * point: the readback used to parse SIX floats (`discLoss, genLoss, predX,
 * predY, meanFx, meanFy`) while no kernel ever wrote slots 4-5, so `meanFx`/
 * `meanFy` reported whatever the buffer happened to hold. Slots are added here
 * only together with the kernel write that fills them.
 *
 * Two slots are CONDITIONAL — `genLoss` is written only when the generator pass
 * runs, and `targetFx/targetFy` only by the vec-field kind — so the trainer
 * clears this whole region every step ({@link pixelStatsFloats} bytes). A
 * conditional slot must never be able to report a previous step's value as the
 * current one; that was the second half of the same defect.
 *
 * FORWARD COMPATIBILITY — docs/PLAN_MULTI_GUESS_MODULARIZATION.md §3f wants
 * per-guess win counters in this region so multi-guess collapse is detectable
 * rather than silent. They go at {@link PIXEL_STATS_WIN_BASE} and up, which
 * makes adding them a PARAMETER change (`pixelStatsFloats(guesses)`) instead of
 * a re-layout. Keep the named slots CONTIGUOUS from 0 — PIXEL_STATS_WIN_BASE is
 * derived from this object's size, so a gap would alias a counter onto a scalar.
 */
export const PIXEL_STATS = {
  /** Critic loss. Written by criticDisc, every kind, every step. */
  discLoss: 0,
  /** Generator loss. Written by criticGen — only when the gen pass runs. */
  genLoss: 1,
  /**
   * vec-field ONLY: F(centre of cell 0) — the TARGET force the head is fitting,
   * read straight out of auxA/auxB. It is NOT the head's prediction; the old
   * `predX`/`predY` names claimed it was.
   */
  targetFx: 2,
  targetFy: 3,
} as const;

/**
 * First per-guess win-counter slot. DERIVED from PIXEL_STATS so adding a named
 * scalar is a one-place edit — the counters just shift up with it.
 */
export const PIXEL_STATS_WIN_BASE = Object.keys(PIXEL_STATS).length;

/**
 * Floats in the stats region. `winCounters` is 0 today — there are no guesses
 * yet — and becomes `dims.guesses` when §3f lands, with no other layout change.
 */
export function pixelStatsFloats(winCounters = 0): number {
  return PIXEL_STATS_WIN_BASE + winCounters;
}

export interface PixelDiscShaderOpts {
  dims: PixelDiscDims;
  batchCap: number;
  /** `blend` | 0 | 1 — mirrors adversary fieldLane. */
  fieldLane?: "blend" | 0 | 1;
}

/** Trainer pass orchestration notes (host owns encodeStep). */
export const PIXEL_DISC_KIND_PASSES: Record<PixelGanKind, string> = {
  "vec-field":
    "clearDens → sampleAndSplat → densToFloat → criticDisc → discAdam;" +
    " gen: clearDensGen → virtualSplat → densToFloat → criticGen → vjp → fieldGrad",
  "next-frame":
    "clearDens → sampleAndSplat → densToFloat → copyDensToAux → clearDensGen →" +
    " virtualSplat → densToFloat → criticDisc → discAdam;" +
    " gen: clearDensGen → virtualSplat → densToFloat → criticGen → vjp → fieldGrad",
  "real-fake":
    "clearDens → sampleAndSplat → densToFloat → clearAtomics → fakeSplat →" +
    " densToFloatFake → criticDisc → discAdam;" +
    " gen: clearDensGen → virtualSplat → criticGen → vjp → fieldGrad",
  inpaint:
    "clearDens → sampleAndSplat → densToFloat → criticDisc → discAdam;" +
    " gen: clearDensGen → virtualSplat → densToFloat → criticGen → vjp → fieldGrad",
};

/**
 * WGSL f32 literal. ALIAS, not a reimplementation
 * (docs/PLAN_MULTI_GUESS_MODULARIZATION.md §2a): the body that used to live
 * here was the third independent copy of "WGSL needs a decimal point", and the
 * only hardened one is `f32lit` — it carries the 2026-07-27 exponent-formatting
 * fix (`1e-10` emitted as `0.1`) that the copies never had. Short name kept
 * because it is interpolated inline throughout the emitted shader.
 */
const fl = f32lit;

export function resolvePixelDims(
  partial?: Partial<PixelDiscDims> & { kind?: PixelGanKind }
): PixelDiscDims {
  return {
    G: partial?.G ?? PIXEL_DISC_DEFAULTS.G,
    E: partial?.E ?? PIXEL_DISC_DEFAULTS.E,
    K: partial?.K ?? PIXEL_DISC_DEFAULTS.K,
    hidden: partial?.hidden ?? PIXEL_DISC_DEFAULTS.hidden,
    dt: partial?.dt ?? PIXEL_DISC_DEFAULTS.dt,
    kind: partial?.kind ?? "vec-field",
  };
}

/**
 * Whether this field shape has a fused pixel-critic codegen — as DATA, so the
 * host gate and the constructor cannot drift.
 *
 * The constructor's job is to refuse an unsupported field loudly
 * ({@link validatePixelDiscFusion}); the host's job is to decide whether to
 * build a critic at all, and it needs the same answer WITHOUT throwing, plus
 * the reason, so it can say out loud why a piece's declared critic is off.
 * Two hand-copied ladders is how those two answers drift apart.
 */
export type PixelDiscFusion =
  | { readonly tag: "ok" }
  | { readonly tag: "unsupported"; readonly reason: string };

export function classifyPixelDiscFusion(field: FieldLayout): PixelDiscFusion {
  if (field.spec.kind !== "helmholtz" && field.spec.kind !== "agree-disagree") {
    return {
      tag: "unsupported",
      reason: `needs a two-head neural field (got ${field.spec.kind})`,
    };
  }
  if (field.classes > 0) {
    return { tag: "unsupported", reason: "class-aware fields not supported yet" };
  }
  if (field.encoding.kind === "hashgrid") {
    return { tag: "unsupported", reason: "hashgrid encoding not supported yet" };
  }
  return { tag: "ok" };
}

export function validatePixelDiscFusion(field: FieldLayout): void {
  const fusion = classifyPixelDiscFusion(field);
  if (fusion.tag === "unsupported") {
    throw new Error(`pixel_disc: ${fusion.reason}`);
  }
}

/** f32 slots in densPack (4 slices × G²). Trainer: densPack = mk(densPackFloats(G)*4). */
export function densPackFloats(G: number): number {
  return 4 * G * G;
}

/** Bytes for densPack f32 buffer (atomics densI32 allocated separately). */
export function pixelDensBytes(G: number): number {
  return densPackFloats(G) * 4;
}

export function pixelParticleScratchFloats(field: FieldLayout): number {
  const sl = trainScratchLayout(field, 1);
  return 8 + sl.encStore + sl.siteBlk;
}

/** f32s of critic field-eval workspace per grid cell (encoding + site block). */
export function pixelCritSiteStride(field: FieldLayout): number {
  const sl = trainScratchLayout(field, 1);
  return sl.encStore + sl.siteBlk;
}

/**
 * `scratch` sizing. Three regions, in the order the shader indexes them:
 *   [0, batchCap·partStride)   per-particle blocks
 *   +8                          pad
 *   + sites·critSiteStride      critic field-eval sites — `fillForceGrid` runs
 *                               ONE CELL PER INVOCATION, so vec-field needs one
 *                               per cell or the parallel pass races on scratch
 *   + critWorkFloats            per-cell critic workspace (cFeat/cSoft/…)
 *
 * Takes `dims` rather than a caller-supplied site count on purpose: the shader
 * derives the critic-work base from the same {@link pixelCritSites}, and a
 * hand-passed number is exactly how that base and this size drift apart.
 */
export function pixelScratchBytes(
  field: FieldLayout,
  batchCap: number,
  dims: PixelDiscDims
): number {
  const partStride = pixelParticleScratchFloats(field);
  return (
    batchCap * partStride +
    8 +
    pixelCritSites(dims) * pixelCritSiteStride(field) +
    pixelCritWorkFloats(dims)
  ) * 4;
}

export type PixelWeightLayout =
  | {
      kind: "vec-field";
      convW: number;
      convB: number;
      code: number;
      headW: number;
      headB: number;
      total: number;
    }
  | {
      kind: "next-frame" | "inpaint";
      convW: number;
      convB: number;
      code: number;
      headW: number;
      headB: number;
      total: number;
    }
  | {
      kind: "real-fake";
      convW: number;
      convB: number;
      code: number;
      mlp0W: number;
      mlp0B: number;
      mlp1W: number;
      mlp1B: number;
      total: number;
    };

/** Packed critic weight offsets — matches packPixelDiscWeights / headFloats(kind). */
export function pixelWeightLayout(d: PixelDiscDims): PixelWeightLayout {
  let o = 0;
  const convW = o;
  o += d.E * 9;
  const convB = o;
  o += d.E;
  const code = o;
  o += d.K * d.E;
  switch (d.kind) {
    case "vec-field": {
      const headW = o;
      o += 2 * d.K;
      const headB = o;
      o += 2;
      return { kind: "vec-field", convW, convB, code, headW, headB, total: o };
    }
    case "next-frame":
    case "inpaint": {
      const headW = o;
      o += d.K;
      const headB = o;
      o += 1;
      return { kind: d.kind, convW, convB, code, headW, headB, total: o };
    }
    case "real-fake": {
      const mlp0W = o;
      o += d.hidden * d.K;
      const mlp0B = o;
      o += d.hidden;
      const mlp1W = o;
      o += d.hidden;
      const mlp1B = o;
      o += 1;
      return { kind: "real-fake", convW, convB, code, mlp0W, mlp0B, mlp1W, mlp1B, total: o };
    }
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

/** mulberry32 — matches CPU oracle (one sample, state = seed). */
function emitHashRng(): string {
  return /* wgsl */ `
fn mulberry32(seed : u32) -> f32 {
  var a = seed + 0x6d2b79f5u;
  var t = (a ^ (a >> 15u)) * (1u | a);
  t = t + (t ^ (t >> 7u)) * (61u | t);
  t = t ^ (t >> 14u);
  return f32(t) / 4294967296.0;
}
`;
}

/**
 * Cross-invocation sync point for the critic kernels.
 *
 * BOTH barriers, always. `workgroupBarrier()` orders workgroup memory (the
 * reduction scratch and the GAP/MLP activations); `storageBarrier()` orders the
 * storage buffer the per-cell workspace lives in. Dropping the storage half is
 * the silent-wrong-answer bug: execution still synchronises, so nothing hangs
 * and nothing errors — a lane just reads a neighbour's stale cFeat.
 */
const BAR = /* wgsl */ `workgroupBarrier();
  storageBarrier();`;

/**
 * Per-cell critic workspace, carved out of the tail of `scratch` (floats,
 * relative to the critic-work base that gets baked into the accessors).
 *
 * These used to be function-scope PRIVATE arrays in a `@workgroup_size(1)`
 * kernel: cFeat[E·G²] + cSoft[K·G²] (+ dSoft, gD, gW) held by ONE lane — ~24 KB
 * of spilled thread-local memory per invocation with no latency hiding, which is
 * where 19–22 ms/step at G=16 went. All of it is per-CELL data, so it belongs in
 * storage indexed by cell and shared by the whole workgroup.
 *
 * It deliberately does NOT live in `var<workgroup>`: cSoft alone is K·G²·4 =
 * 16 KB at the gallery config and `maxComputeWorkgroupStorageSize` is 16384
 * bytes on mobile. Only small cross-cell scalars go in workgroup memory
 * (see {@link pixelCritWorkgroupBytes}).
 */
interface CritWorkLayout {
  /** cFeat[E·nCell] — post-ReLU conv features. */
  feat: number;
  /** cSoft[K·nCell] — per-cell softmax over the codebook. */
  soft: number;
  /** dSoft[K·nCell] in; overwritten IN PLACE by dLogit[K·nCell]. */
  dsoft: number;
  /** dFeat[E·nCell], stored already masked by the ReLU (i.e. `gf`). */
  dfeat: number;
  /** 2·nCell per-cell head grads (dvx,dvy for vec-field; dPred otherwise). */
  hg: number;
  total: number;
}

function critWorkLayout(d: PixelDiscDims): CritWorkLayout {
  const n = d.G * d.G;
  const feat = 0;
  const soft = feat + d.E * n;
  const dsoft = soft + d.K * n;
  const dfeat = dsoft + d.K * n;
  const hg = dfeat + d.E * n;
  return { feat, soft, dsoft, dfeat, hg, total: hg + 2 * n };
}

/** f32s of per-cell critic workspace appended to `scratch`. */
export function pixelCritWorkFloats(d: PixelDiscDims): number {
  return critWorkLayout(d).total;
}

/**
 * Critic field-eval sites to reserve in `scratch`. `fillForceGrid` runs ONE
 * CELL PER INVOCATION, so vec-field needs a site block per cell; the other
 * kinds have no force grid and pay for one.
 */
export function pixelCritSites(d: PixelDiscDims): number {
  return d.kind === "vec-field" ? d.G * d.G : 1;
}

/**
 * Bytes of `var<workgroup>` the critic kernels declare. MUST stay ≤ 16384 —
 * that is `maxComputeWorkgroupStorageSize` on mobile, and exceeding it is a
 * pipeline-creation failure on exactly the devices this parallelisation was
 * for. Nothing that scales with G² may move into workgroup memory.
 */
export function pixelCritWorkgroupBytes(d: PixelDiscDims): number {
  const reduce = PIXEL_DISC_WG * 4;
  const gapMlp = d.kind === "real-fake" ? (2 * d.K + 2 * d.hidden) * 4 : 0;
  return reduce + gapMlp;
}

/** Storage accessors for the per-cell workspace, with the base baked in. */
function emitCritAccessors(d: PixelDiscDims, base: number): string {
  const cw = critWorkLayout(d);
  const at = (off: number) => `${base + off}u`;
  return /* wgsl */ `
fn cFeatAt(e : u32, c : u32) -> f32 { return scratch[${at(cw.feat)} + e * N_CELL + c]; }
fn setCFeat(e : u32, c : u32, v : f32) { scratch[${at(cw.feat)} + e * N_CELL + c] = v; }
fn cSoftAt(k : u32, c : u32) -> f32 { return scratch[${at(cw.soft)} + k * N_CELL + c]; }
fn setCSoft(k : u32, c : u32, v : f32) { scratch[${at(cw.soft)} + k * N_CELL + c] = v; }
fn dSoftAt(k : u32, c : u32) -> f32 { return scratch[${at(cw.dsoft)} + k * N_CELL + c]; }
fn setDSoft(k : u32, c : u32, v : f32) { scratch[${at(cw.dsoft)} + k * N_CELL + c] = v; }
fn gFeatAt(e : u32, c : u32) -> f32 { return scratch[${at(cw.dfeat)} + e * N_CELL + c]; }
fn setGFeat(e : u32, c : u32, v : f32) { scratch[${at(cw.dfeat)} + e * N_CELL + c] = v; }
fn hGradAt(j : u32, c : u32) -> f32 { return scratch[${at(cw.hg)} + j * N_CELL + c]; }
fn setHGrad(j : u32, c : u32, v : f32) { scratch[${at(cw.hg)} + j * N_CELL + c] = v; }
`;
}

/**
 * Workgroup sum reduction.
 *
 * MUST be called from UNIFORM control flow by EVERY invocation — it contains
 * barriers, so an `if (cell >= N_CELL) { return; }` guard above it is a hang,
 * not an optimisation. That is also why every cell loop in these kernels is a
 * grid-stride loop instead of a one-cell-per-thread guard.
 *
 * The trailing barrier is not redundant: without it a fast lane can start the
 * NEXT wgSum's `wgRed[tid]` store while a slow lane still reads `wgRed[0]`.
 */
function emitWgReduce(): string {
  return /* wgsl */ `
var<workgroup> wgRed : array<f32, ${PIXEL_DISC_WG}>;
fn wgSum(tid : u32, v : f32) -> f32 {
  wgRed[tid] = v;
  ${BAR}
  for (var s = ${PIXEL_DISC_WG / 2}u; s > 0u; s = s >> 1u) {
    if (tid < s) { wgRed[tid] = wgRed[tid] + wgRed[tid + s]; }
    workgroupBarrier();
  }
  let r = wgRed[0];
  workgroupBarrier();
  return r;
}
`;
}

/** Zero the critMeta gradient slice so every later write can just accumulate. */
function emitZeroGrads(nW: number): string {
  return /* wgsl */ `
  for (var w = tid; w < ${nW}u; w = w + ${PIXEL_DISC_WG}u) { critMeta[w] = 0.0; }
  ${BAR}
`;
}

/**
 * conv3×3 → ReLU → codebook softmax, ONE CELL PER INVOCATION (grid-strided, so
 * G² may exceed the workgroup). The softmax needs only this cell's features, so
 * it fuses into the same loop with no barrier in between.
 */
function emitFwdPar(d: PixelDiscDims, densOff: string): string {
  const { G, E, K } = d;
  const wl = pixelWeightLayout(d);
  return /* wgsl */ `
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    let cy = c / G;
    let cx = c % G;
    var feat : array<f32, ${E}>;
    for (var e = 0u; e < ${E}u; e = e + 1u) {
      var s = critW[${wl.convB}u + e];
      for (var dy = 0u; dy < 3u; dy = dy + 1u) {
        let yy = u32(clamp(i32(cy) + i32(dy) - 1, 0, ${G - 1}));
        for (var dx = 0u; dx < 3u; dx = dx + 1u) {
          let xx = u32(clamp(i32(cx) + i32(dx) - 1, 0, ${G - 1}));
          s = s + critW[${wl.convW}u + e * 9u + dy * 3u + dx] * densPack[${densOff} + yy * G + xx];
        }
      }
      let a = max(s, 0.0);
      feat[e] = a;
      setCFeat(e, c, a);
    }
    var logits : array<f32, ${K}>;
    var maxL = -1e30;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      var s2 = 0.0;
      for (var e = 0u; e < ${E}u; e = e + 1u) {
        s2 = s2 + feat[e] * critW[${wl.code}u + k * ${E}u + e];
      }
      logits[k] = s2;
      maxL = max(maxL, s2);
    }
    var sum = 0.0;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      let v = exp(logits[k] - maxL);
      logits[k] = v;
      sum = sum + v;
    }
    let inv = 1.0 / sum;
    for (var k = 0u; k < ${K}u; k = k + 1u) { setCSoft(k, c, logits[k] * inv); }
  }
  ${BAR}
`;
}

/**
 * Softmax VJP + codebook VJP, one cell per invocation. dSoft is overwritten in
 * place by dLogit (the `dot` that needs the old values is computed first), and
 * dFeat is stored already multiplied by the ReLU mask — so the weight-gradient
 * and gD passes downstream read one array, `gFeatAt`, and never re-check it.
 */
function emitBwdActPar(d: PixelDiscDims): string {
  const { E, K } = d;
  const wl = pixelWeightLayout(d);
  return /* wgsl */ `
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    var dot = 0.0;
    for (var k = 0u; k < ${K}u; k = k + 1u) { dot = dot + cSoftAt(k, c) * dSoftAt(k, c); }
    var dl : array<f32, ${K}>;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      let v = cSoftAt(k, c) * (dSoftAt(k, c) - dot);
      dl[k] = v;
      setDSoft(k, c, v);
    }
    for (var e = 0u; e < ${E}u; e = e + 1u) {
      var g = 0.0;
      for (var k = 0u; k < ${K}u; k = k + 1u) {
        g = g + dl[k] * critW[${wl.code}u + k * ${E}u + e];
      }
      setGFeat(e, c, select(0.0, g, cFeatAt(e, c) > 0.0));
    }
  }
  ${BAR}
`;
}

/**
 * Backbone WEIGHT gradients — ONE WEIGHT PER INVOCATION, each summing over all
 * cells. This is why the kernel needs no gW partials and no atomics: every
 * output index has exactly one owning invocation, and its reduction is a plain
 * in-order loop over cells, so the sum order matches the old serial kernel.
 */
function emitBackboneWeightGrads(d: PixelDiscDims, densOff: string): string {
  const { G, E, K } = d;
  const wl = pixelWeightLayout(d);
  return /* wgsl */ `
  for (var w = tid; w < ${K * E}u; w = w + ${PIXEL_DISC_WG}u) {
    let k = w / ${E}u;
    let e = w % ${E}u;
    var s = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) { s = s + dSoftAt(k, c) * cFeatAt(e, c); }
    critMeta[${wl.code}u + w] = critMeta[${wl.code}u + w] + s;
  }
  for (var e = tid; e < ${E}u; e = e + ${PIXEL_DISC_WG}u) {
    var s = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) { s = s + gFeatAt(e, c); }
    critMeta[${wl.convB}u + e] = critMeta[${wl.convB}u + e] + s;
  }
  for (var w = tid; w < ${E * 9}u; w = w + ${PIXEL_DISC_WG}u) {
    let e = w / 9u;
    let tap = w % 9u;
    let dy = tap / 3u;
    let dx = tap % 3u;
    var s = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) {
      let yy = u32(clamp(i32(c / G) + i32(dy) - 1, 0, ${G - 1}));
      let xx = u32(clamp(i32(c % G) + i32(dx) - 1, 0, ${G - 1}));
      s = s + gFeatAt(e, c) * densPack[${densOff} + yy * G + xx];
    }
    critMeta[${wl.convW}u + w] = critMeta[${wl.convW}u + w] + s;
  }
  ${BAR}
`;
}

/**
 * ∂L/∂D at one cell — the GATHER form of the conv backward.
 *
 * The serial kernel SCATTERED: `gD[clamp(y+dy-1)][clamp(x+dx-1)] += gf[e][y][x]·W`.
 * A scatter races the moment cells are spread across invocations, so this
 * inverts it. For target row Y the contributing source rows are the contiguous
 * range [lo,hi] with
 *   lo = (Y == 0)   ? 0   : Y-dy+1
 *   hi = (Y == G-1) ? G-1 : Y-dy+1
 * and the range is EMPTY when lo > hi. Those two edge cases are exactly the
 * clamp's edge replication: a border cell receives every tap the clamp folded
 * onto it (Y=0 takes y∈{0,1} at dy=0 and nothing at dy=2), an interior cell
 * takes the single source Y-dy+1. Columns are identical by separability.
 */
function emitGdGather(d: PixelDiscDims): string {
  const { G, E } = d;
  const wl = pixelWeightLayout(d);
  return /* wgsl */ `
fn gdAt(c : u32) -> f32 {
  let cy = i32(c / G);
  let cx = i32(c % G);
  var acc = 0.0;
  for (var e = 0u; e < ${E}u; e = e + 1u) {
    for (var dy = 0u; dy < 3u; dy = dy + 1u) {
      let ylo = select(cy - i32(dy) + 1, 0, cy == 0);
      let yhi = select(cy - i32(dy) + 1, ${G - 1}, cy == ${G - 1});
      for (var dx = 0u; dx < 3u; dx = dx + 1u) {
        let xlo = select(cx - i32(dx) + 1, 0, cx == 0);
        let xhi = select(cx - i32(dx) + 1, ${G - 1}, cx == ${G - 1});
        let wv = critW[${wl.convW}u + e * 9u + dy * 3u + dx];
        for (var y = ylo; y <= yhi; y = y + 1) {
          for (var x = xlo; x <= xhi; x = x + 1) {
            acc = acc + wv * gFeatAt(e, u32(y) * G + u32(x));
          }
        }
      }
    }
  }
  return acc;
}
`;
}

/** GAP → MLP forward (real-fake). Leaves `logit` uniform across the workgroup. */
function emitGapMlpFwdPar(d: PixelDiscDims): string {
  const { K, hidden } = d;
  const wl = pixelWeightLayout(d);
  if (wl.kind !== "real-fake") throw new Error("GAP MLP only for real-fake");
  return /* wgsl */ `
  for (var k = tid; k < ${K}u; k = k + ${PIXEL_DISC_WG}u) {
    var s = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) { s = s + cSoftAt(k, c); }
    wgZ[k] = s / f32(N_CELL);
  }
  ${BAR}
  for (var i = tid; i < ${hidden}u; i = i + ${PIXEL_DISC_WG}u) {
    var s = critW[${wl.mlp0B}u + i];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      s = s + critW[${wl.mlp0W}u + i * ${K}u + k] * wgZ[k];
    }
    wgH[i] = tanh(s);
  }
  ${BAR}
  var logit = critW[${wl.mlp1B}u];
  for (var i = 0u; i < ${hidden}u; i = i + 1u) {
    logit = logit + critW[${wl.mlp1W}u + i] * wgH[i];
  }
`;
}

/**
 * MLP backward (real-fake). `dLogit` is uniform, so each MLP weight gradient is
 * owned by one invocation; `dz` reduces over `hidden`, and the resulting
 * per-code constant is broadcast into every cell's dSoft.
 *
 * `wantGrads=false` drops the weight writes entirely: criticGen only ever
 * produces ∂L/∂D — its gW was computed and then discarded by the old kernel.
 */
function emitGapMlpBwdPar(
  d: PixelDiscDims,
  dLogitExpr: string,
  wantGrads: boolean
): string {
  const { K, hidden } = d;
  const wl = pixelWeightLayout(d);
  if (wl.kind !== "real-fake") throw new Error("GAP MLP only for real-fake");
  const wGrad1B = wantGrads
    ? `  if (tid == 0u) { critMeta[${wl.mlp1B}u] = critMeta[${wl.mlp1B}u] + dLogit; }`
    : "";
  const wGradI = wantGrads
    ? `    critMeta[${wl.mlp1W}u + i] = critMeta[${wl.mlp1W}u + i] + dLogit * wgH[i];
    critMeta[${wl.mlp0B}u + i] = critMeta[${wl.mlp0B}u + i] + pre;`
    : "";
  const wGrad0W = wantGrads
    ? `  for (var w = tid; w < ${hidden * K}u; w = w + ${PIXEL_DISC_WG}u) {
    critMeta[${wl.mlp0W}u + w] = critMeta[${wl.mlp0W}u + w] + wgPre[w / ${K}u] * wgZ[w % ${K}u];
  }`
    : "";
  return /* wgsl */ `
  let dLogit = ${dLogitExpr};
${wGrad1B}
  for (var i = tid; i < ${hidden}u; i = i + ${PIXEL_DISC_WG}u) {
    let pre = (dLogit * critW[${wl.mlp1W}u + i]) * (1.0 - wgH[i] * wgH[i]);
    wgPre[i] = pre;
${wGradI}
  }
  ${BAR}
${wGrad0W}
  for (var k = tid; k < ${K}u; k = k + ${PIXEL_DISC_WG}u) {
    var dzk = 0.0;
    for (var i = 0u; i < ${hidden}u; i = i + 1u) {
      dzk = dzk + wgPre[i] * critW[${wl.mlp0W}u + i * ${K}u + k];
    }
    wgDz[k] = dzk / f32(N_CELL);
  }
  ${BAR}
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    for (var k = 0u; k < ${K}u; k = k + 1u) { setDSoft(k, c, wgDz[k]); }
  }
  ${BAR}
`;
}

/**
 * Random block mask + the masked conv input, one cell per invocation. The rect
 * is a pure function of (maskSeed, cell), so every lane derives the same block
 * the old single-thread `buildInpaintMask` wrote cell by cell.
 */
function emitInpaintMaskPar(d: PixelDiscDims): string {
  const G = d.G;
  return /* wgsl */ `
  let bw = max(2u, u32(floor(f32(${G}u) * 0.5)));
  let bh = max(2u, u32(floor(f32(${G}u) * 0.5)));
  let mx0 = u32(floor(mulberry32(uni.maskSeed) * f32(${G}u - bw + 1u)));
  let my0 = u32(floor(mulberry32(uni.maskSeed + 17u) * f32(${G}u - bh + 1u)));
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    let cy = c / G;
    let cx = c % G;
    let inM = cy >= my0 && cy < my0 + bh && cx >= mx0 && cx < mx0 + bw;
    set_auxB(c, select(0.0, 1.0, inM));
    densPack[2u * N_CELL + c] = dens_at(c) * select(1.0, 0.0, inM);
  }
  ${BAR}
`;
}

function emitBceGradLogit(): string {
  return /* wgsl */ `
fn bceGradLogit(logit : f32, tgt : f32) -> vec2f {
  let x = clamp(logit, -20.0, 20.0);
  let p = 1.0 / (1.0 + exp(-x));
  let loss = -(tgt * log(p + 1e-8) + (1.0 - tgt) * log(1.0 - p + 1e-8));
  return vec2f(loss, p - tgt);
}
`;
}
/**
 * Per-cell target field F(cell_center) for `vec-field`, ONE CELL PER INVOCATION.
 *
 * This used to be a plain `fn evalForceGrid()` called from `criticDisc`, which
 * is `@compute @workgroup_size(1)` — so a single GPU thread evaluated the whole
 * neural field (all heads, every weight) at all G² cell centers, serially. At
 * the gallery config that is 256 × 3328 MACs on one lane and measured ~24 ms of
 * the ~36 ms pixel-GAN step, i.e. the dominant cost of the piece. It is a pure
 * map with no cross-cell dependency, so it belongs in its own parallel pass.
 *
 * Writes auxA/auxB, which `criticGen` later reads as its targets. Nothing
 * between the two passes touches aux (clearDensGen only clears dens/dDens), so
 * one dispatch per step ahead of criticDisc is enough.
 */
function emitVecFieldForceGrid(critFwdExpr: string, encInit: string): string {
  return /* wgsl */ `
@compute @workgroup_size(${PIXEL_DISC_WG})
fn fillForceGrid(@builtin(global_invocation_id) gid : vec3u) {
  let cell = gid.x;
  if (cell >= N_CELL) { return; }
  let uk = vec2f(
    (f32(cell % G) + 0.5) / f32(G),
    (f32(cell / G) + 0.5) / f32(G)
  );
  ${encInit}
  let F = ${critFwdExpr};
  set_auxA(cell, F.x);
  set_auxB(cell, F.y);
}
`;
}

/**
 * criticDisc — cell-parallel forward + head, then WEIGHT-parallel gradients.
 *
 * `gD` is deliberately absent: the serial kernel computed it here and never
 * read it (only criticGen turns ∂L/∂D into a generator signal), so the whole
 * conv-input gradient is dead work on the disc side.
 */
function emitCriticDiscBody(d: PixelDiscDims, metaStats: number): string {
  const { K } = d;
  const nW = pixelDiscWeightCount(d);
  const wl = pixelWeightLayout(d);
  const dens0 = "0u";
  const densAux = "2u * N_CELL";

  switch (d.kind) {
    case "vec-field": {
      if (wl.kind !== "vec-field") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitZeroGrads(nW)}
  ${emitFwdPar(d, dens0)}
  var lAct = 0.0;
  var lR = 0.0;
  var lBx = 0.0;
  var lBy = 0.0;
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    // !(dens < FLOOR), not dens >= FLOOR: the serial kernel skipped with
    // "if (dens < FLOOR) { continue; }", and the two disagree on NaN.
    let act = !(dens_at(c) < ${fl(DENS_FLOOR)});
    var vx = critW[${wl.headB}u];
    var vy = critW[${wl.headB}u + 1u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      let q = cSoftAt(k, c);
      vx = vx + critW[${wl.headW}u + k] * q;
      vy = vy + critW[${wl.headW}u + ${K}u + k] * q;
    }
    let ex = vx - auxA(c);
    let ey = vy - auxB(c);
    let r = sqrt(ex * ex + ey * ey + SOFT_EPS2);
    let s = 1.0 / r;
    let dvx = select(0.0, ex * s, act);
    let dvy = select(0.0, ey * s, act);
    lAct = lAct + select(0.0, 1.0, act);
    lR = lR + select(0.0, r, act);
    lBx = lBx + dvx;
    lBy = lBy + dvy;
    setHGrad(0u, c, dvx);
    setHGrad(1u, c, dvy);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      setDSoft(k, c, dvx * critW[${wl.headW}u + k] + dvy * critW[${wl.headW}u + ${K}u + k]);
    }
  }
  let nActive = wgSum(tid, lAct);
  let sumR = wgSum(tid, lR);
  let sumBx = wgSum(tid, lBx);
  let sumBy = wgSum(tid, lBy);
  let activeN = max(nActive, 1.0);
  if (tid == 0u) {
    critMeta[${metaStats + PIXEL_STATS.discLoss}u] = sumR / activeN;
    // auxA/auxB hold the TARGET force at each cell centre (fillForceGrid), so
    // these two are F(cell 0), not a prediction — see PIXEL_STATS.targetFx.
    critMeta[${metaStats + PIXEL_STATS.targetFx}u] = auxA(0u);
    critMeta[${metaStats + PIXEL_STATS.targetFy}u] = auxB(0u);
    critMeta[${wl.headB}u] = critMeta[${wl.headB}u] + sumBx / activeN;
    critMeta[${wl.headB}u + 1u] = critMeta[${wl.headB}u + 1u] + sumBy / activeN;
  }
  for (var k = tid; k < ${K}u; k = k + ${PIXEL_DISC_WG}u) {
    var sx = 0.0;
    var sy = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) {
      let q = cSoftAt(k, c);
      sx = sx + hGradAt(0u, c) * q;
      sy = sy + hGradAt(1u, c) * q;
    }
    critMeta[${wl.headW}u + k] = critMeta[${wl.headW}u + k] + sx / activeN;
    critMeta[${wl.headW}u + ${K}u + k] =
      critMeta[${wl.headW}u + ${K}u + k] + sy / activeN;
  }
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    for (var k = 0u; k < ${K}u; k = k + 1u) { setDSoft(k, c, dSoftAt(k, c) / activeN); }
  }
  ${BAR}
  ${emitBwdActPar(d)}
  ${emitBackboneWeightGrads(d, dens0)}
`;
    }
    case "next-frame": {
      if (wl.kind !== "next-frame") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitZeroGrads(nW)}
  ${emitFwdPar(d, densAux)}
  var lR = 0.0;
  var lB = 0.0;
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    var pred = critW[${wl.headB}u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      pred = pred + critW[${wl.headW}u + k] * cSoftAt(k, c);
    }
    let diff = pred - dens_at(c);
    let r = sqrt(diff * diff + SOFT_EPS2);
    lR = lR + r;
    let dPred = diff / r / f32(N_CELL);
    lB = lB + dPred;
    setHGrad(0u, c, dPred);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      setDSoft(k, c, dPred * critW[${wl.headW}u + k]);
    }
  }
  let sumR = wgSum(tid, lR);
  let sumB = wgSum(tid, lB);
  if (tid == 0u) {
    critMeta[${metaStats + PIXEL_STATS.discLoss}u] = sumR / f32(N_CELL);
    critMeta[${wl.headB}u] = critMeta[${wl.headB}u] + sumB;
  }
  for (var k = tid; k < ${K}u; k = k + ${PIXEL_DISC_WG}u) {
    var s = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) { s = s + hGradAt(0u, c) * cSoftAt(k, c); }
    critMeta[${wl.headW}u + k] = critMeta[${wl.headW}u + k] + s;
  }
  ${BAR}
  ${emitBwdActPar(d)}
  ${emitBackboneWeightGrads(d, densAux)}
`;
    }
    case "real-fake": {
      if (wl.kind !== "real-fake") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitZeroGrads(nW)}
  var discLoss = 0.0;
  {
    ${emitFwdPar(d, dens0)}
    ${emitGapMlpFwdPar(d)}
    let bce = bceGradLogit(logit, 1.0);
    discLoss = discLoss + bce.x;
    ${emitGapMlpBwdPar(d, "bce.y", true)}
    ${emitBwdActPar(d)}
    ${emitBackboneWeightGrads(d, dens0)}
  }
  {
    ${emitFwdPar(d, densAux)}
    ${emitGapMlpFwdPar(d)}
    let bce = bceGradLogit(logit, 0.0);
    discLoss = discLoss + bce.x;
    ${emitGapMlpBwdPar(d, "bce.y", true)}
    ${emitBwdActPar(d)}
    ${emitBackboneWeightGrads(d, densAux)}
  }
  if (tid == 0u) { critMeta[${metaStats + PIXEL_STATS.discLoss}u] = discLoss; }
`;
    }
    case "inpaint": {
      if (wl.kind !== "inpaint") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitZeroGrads(nW)}
  ${emitInpaintMaskPar(d)}
  ${emitFwdPar(d, densAux)}
  var lM = 0.0;
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    lM = lM + select(0.0, 1.0, auxB(c) > 0.5);
  }
  let nMask = max(wgSum(tid, lM), 1.0);
  var lR = 0.0;
  var lB = 0.0;
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    let m = auxB(c) > 0.5;
    var pred = critW[${wl.headB}u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      pred = pred + critW[${wl.headW}u + k] * cSoftAt(k, c);
    }
    let diff = pred - dens_at(c);
    let r = sqrt(diff * diff + SOFT_EPS2);
    lR = lR + select(0.0, r, m);
    let dPred = select(0.0, (diff / r) / nMask, m);
    lB = lB + dPred;
    setHGrad(0u, c, dPred);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      setDSoft(k, c, dPred * critW[${wl.headW}u + k]);
    }
  }
  let sumR = wgSum(tid, lR);
  let sumB = wgSum(tid, lB);
  if (tid == 0u) {
    critMeta[${metaStats + PIXEL_STATS.discLoss}u] = sumR / nMask;
    critMeta[${wl.headB}u] = critMeta[${wl.headB}u] + sumB;
  }
  for (var k = tid; k < ${K}u; k = k + ${PIXEL_DISC_WG}u) {
    var s = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) { s = s + hGradAt(0u, c) * cSoftAt(k, c); }
    critMeta[${wl.headW}u + k] = critMeta[${wl.headW}u + k] + s;
  }
  ${BAR}
  ${emitBwdActPar(d)}
  ${emitBackboneWeightGrads(d, densAux)}
`;
    }
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

/**
 * criticGen — cell-parallel forward + head, then ∂L/∂D into densPack's dDens.
 *
 * Mirror image of criticDisc: `gW` is deliberately absent because the serial
 * kernel computed the critic weight gradient here and then discarded it (only
 * criticDisc feeds discAdam).
 */
function emitCriticGenBody(d: PixelDiscDims, metaStats: number): string {
  const { K } = d;
  const wl = pixelWeightLayout(d);
  const dens0 = "0u";
  const densAux = "2u * N_CELL";

  switch (d.kind) {
    case "vec-field": {
      if (wl.kind !== "vec-field") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitFwdPar(d, dens0)}
  var lAct = 0.0;
  var lR = 0.0;
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    let act = !(dens_at(c) < ${fl(DENS_FLOOR)});
    var vx = critW[${wl.headB}u];
    var vy = critW[${wl.headB}u + 1u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      let q = cSoftAt(k, c);
      vx = vx + critW[${wl.headW}u + k] * q;
      vy = vy + critW[${wl.headW}u + ${K}u + k] * q;
    }
    let ex = vx - auxA(c);
    let ey = vy - auxB(c);
    let r = sqrt(ex * ex + ey * ey + SOFT_EPS2);
    let s = uni.genSign / r;
    let dvx = select(0.0, ex * s, act);
    let dvy = select(0.0, ey * s, act);
    lAct = lAct + select(0.0, 1.0, act);
    lR = lR + select(0.0, r, act);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      setDSoft(k, c, dvx * critW[${wl.headW}u + k] + dvy * critW[${wl.headW}u + ${K}u + k]);
    }
  }
  let nActive = wgSum(tid, lAct);
  let sumR = wgSum(tid, lR);
  let activeN = max(nActive, 1.0);
  if (tid == 0u) { critMeta[${metaStats + PIXEL_STATS.genLoss}u] = uni.genSign * (sumR / activeN); }
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    for (var k = 0u; k < ${K}u; k = k + 1u) { setDSoft(k, c, dSoftAt(k, c) / activeN); }
  }
  ${BAR}
  ${emitBwdActPar(d)}
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) { set_d_dens(c, gdAt(c)); }
`;
    }
    case "next-frame": {
      if (wl.kind !== "next-frame") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitFwdPar(d, densAux)}
  var lR = 0.0;
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    var pred = critW[${wl.headB}u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      pred = pred + critW[${wl.headW}u + k] * cSoftAt(k, c);
    }
    let diff = pred - dens_at(c);
    let r = sqrt(diff * diff + SOFT_EPS2);
    lR = lR + r;
    add_d_dens(c, uni.genSign * (-diff / r) / f32(N_CELL));
  }
  let sumR = wgSum(tid, lR);
  if (tid == 0u) { critMeta[${metaStats + PIXEL_STATS.genLoss}u] = uni.genSign * (sumR / f32(N_CELL)); }
`;
    }
    case "real-fake": {
      if (wl.kind !== "real-fake") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitFwdPar(d, dens0)}
  ${emitGapMlpFwdPar(d)}
  if (tid == 0u) { critMeta[${metaStats + PIXEL_STATS.genLoss}u] = uni.genSign * (-logit); }
  ${emitGapMlpBwdPar(d, "uni.genSign * -1.0", false)}
  ${emitBwdActPar(d)}
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) { set_d_dens(c, gdAt(c)); }
`;
    }
    case "inpaint": {
      if (wl.kind !== "inpaint") throw new Error("layout mismatch");
      return /* wgsl */ `
  var lM = 0.0;
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    lM = lM + select(0.0, 1.0, auxB(c) > 0.5);
  }
  let nMask = max(wgSum(tid, lM), 1.0);
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    densPack[${densAux} + c] = dens_at(c) * (1.0 - auxB(c));
  }
  ${BAR}
  ${emitFwdPar(d, densAux)}
  var lR = 0.0;
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    let m = auxB(c) > 0.5;
    var pred = critW[${wl.headB}u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      pred = pred + critW[${wl.headW}u + k] * cSoftAt(k, c);
    }
    let diff = pred - dens_at(c);
    let r = sqrt(diff * diff + SOFT_EPS2);
    lR = lR + select(0.0, r, m);
    let scale = uni.genSign / r / nMask;
    set_d_dens(c, select(0.0, -diff * scale, m));
    let dPred = select(0.0, diff * scale, m);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      setDSoft(k, c, dPred * critW[${wl.headW}u + k]);
    }
  }
  let sumR = wgSum(tid, lR);
  if (tid == 0u) { critMeta[${metaStats + PIXEL_STATS.genLoss}u] = uni.genSign * (sumR / nMask); }
  ${emitBwdActPar(d)}
  // MASKED cells already took their gradient straight from the residual; only
  // the CONTEXT cells reach D through the conv, because its input is D·(1−mask).
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    let g = gdAt(c);
    if (auxB(c) <= 0.5) { add_d_dens(c, g); }
  }
`;
    }
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

/**
 * Main fused module: particle sample + field + density + critic + disc Adam +
 * gen density path + fieldGrad into extGrads.
 *
 * Bindings (8 storage + uniform):
 *  0 uni, 1 fieldW, 2 critW, 3 scratch, 4 densI32, 5 densPack, 6 critMeta, 7 extGrads, 8 partPos
 */
export function pixelDiscShader(
  field: FieldLayout,
  opts: PixelDiscShaderOpts
): string {
  validatePixelDiscFusion(field);
  const d = opts.dims;
  const batchCap = Math.min(opts.batchCap, PIXEL_DISC_MAX_BATCH);
  const fieldLane = opts.fieldLane ?? "blend";
  const sl = trainScratchLayout(field, 1);
  const heads = field.spec.heads as HeadSpec[];
  const enc = field.encoding;
  const partStride = pixelParticleScratchFloats(field);
  const nCell = d.G * d.G;
  const nW = pixelDiscWeightCount(d);
  const maxW = Math.max(
    2,
    sl.encDim,
    ...heads.flatMap((h) => h.layers.map((L) => Math.max(L.outSize, L.inSize)))
  );

  const oPos = 0;
  const oF = 2;
  const oPos2 = 4;
  const oDF = 6;
  const oEnc = 8;
  const oField = 8 + sl.encStore;

  // Critic field-eval workspace, indexed BY CELL: fillForceGrid runs one cell
  // per invocation, so every cell needs its own encoding + site block. These
  // bases are only referenced from inside fillForceGrid, where `cell` is bound.
  const critWorkspace = batchCap * partStride;
  const critSiteStride = sl.encStore + sl.siteBlk;
  const critSite = `(${critWorkspace + 8}u + cell * ${critSiteStride}u)`;
  const critEncBase = critSite;
  // Per-cell critic workspace lives right after the field-eval sites. Same
  // pixelCritSites() the allocator uses — see pixelScratchBytes.
  const critWorkBase =
    critWorkspace + 8 + pixelCritSites(d) * critSiteStride;
  const critFieldBase = (h: number) =>
    `${critSite} + ${sl.encStore}u + ${h === 0 ? 0 : sl.headBlk[0]}u`;

  const fieldBase = (h: number) =>
    `pBase + ${oField}u + ${h === 0 ? 0 : sl.headBlk[0]}u`;
  const encBase = () => `pBase + ${oEnc}u`;
  const fwdCall = (h: number) =>
    enc.kind === "raw"
      ? `fwd_head_${h}(uk, ${fieldBase(h)}, 0u)`
      : `fwd_head_${h}(${encBase()}, ${fieldBase(h)})`;
  const critFwdCall = (h: number) =>
    enc.kind === "raw"
      ? `fwd_head_${h}(uk, ${critFieldBase(h)}, 0u)`
      : `fwd_head_${h}(${critEncBase}, ${critFieldBase(h)})`;
  const bwdCall = (h: number, dExpr: string) =>
    enc.kind === "raw"
      ? `bwd_head_${h}(${dExpr}, ${fieldBase(h)})`
      : `bwd_head_${h}(${dExpr}, ${fieldBase(h)}, ${encBase()})`;
  const fieldForward =
    fieldLane === "blend"
      ? `(1.0 - uni.alpha) * ${fwdCall(0)} + uni.alpha * ${fwdCall(1)}`
      : fwdCall(fieldLane);
  const critFieldForward =
    fieldLane === "blend"
      ? `(1.0 - uni.alpha) * ${critFwdCall(0)} + uni.alpha * ${critFwdCall(1)}`
      : critFwdCall(fieldLane);
  const fieldBackward =
    fieldLane === "blend"
      ? `let _du0 = ${bwdCall(0, "dSig * (1.0 - uni.alpha)")};
    let _du1 = ${bwdCall(1, "dSig * uni.alpha")};`
      : `let _duLane = ${bwdCall(fieldLane, "dSig")};`;

  const metaGrads = 0;
  const metaM = nW;
  const metaV = 2 * nW;
  const metaStats = 3 * nW;

  const fwdStores = heads
    .map((h, i) => emitFwdStore(i, h, sl, maxW, enc))
    .join("\n");
  const bwdStores = heads
    .map((h, i) => emitBwdStore(i, h, sl, maxW, enc, i === 0 ? "seed" : "accumulate"))
    .join("\n");
  const encodeFn = enc.kind === "raw" ? "" : emitEncode(enc);
  const critEncInit =
    enc.kind === "raw" ? "" : `encodeSite(uk, ${critEncBase});`;

  const fieldBlocks: string[] = [];
  for (const seg of field.segments) {
    if (seg.role === "grid") continue;
    const h = seg.head;
    if (fieldLane !== "blend" && h !== fieldLane) continue;
    const l = seg.layer;
    const L = heads[h].layers[l];
    const start = seg.floatOffset;
    const end = seg.floatOffset + seg.floatLength;
    const headOff = h === 0 ? 0 : sl.headBlk[0];
    const fb = `pBase + ${oField}u + ${headOff}u`;
    const aIn =
      l === 0
        ? enc.kind === "raw"
          ? `scratch[pBase + ${oPos}u + i]`
          : `scratch[pBase + ${oEnc}u + i]`
        : `scratch[${fb} + ${sl.aOff[h][l - 1]}u + i]`;
    if (seg.role === "kernel") {
      fieldBlocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let local = t - ${start}u;
    let i = local / ${L.outSize}u;
    let j = local % ${L.outSize}u;
    for (var s = 0u; s < uni.b; s = s + 1u) {
      let pBase = s * ${partStride}u;
      g = g + ${aIn} * scratch[${fb} + ${sl.dOff[h][l]}u + j];
    }
  }`);
    } else {
      fieldBlocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let j = t - ${start}u;
    for (var s = 0u; s < uni.b; s = s + 1u) {
      let pBase = s * ${partStride}u;
      g = g + scratch[${fb} + ${sl.dOff[h][l]}u + j];
    }
  }`);
    }
  }

  const hashFns = emitHashRng();
  const bceFn = d.kind === "real-fake" ? emitBceGradLogit() : "";
  const forceGridFn =
    d.kind === "vec-field"
      ? emitVecFieldForceGrid(critFieldForward, critEncInit)
      : "";
  const critAccessors = emitCritAccessors(d, critWorkBase);
  const wgReduceFn = emitWgReduce();
  // next-frame's generator gradient reaches D directly (its conv input is D₀,
  // a constant w.r.t. the generated frame), so it is the one kind with no
  // conv-backward gather.
  const gdFn = d.kind === "next-frame" ? "" : emitGdGather(d);
  // Cross-cell scalars only — everything that scales with G² stays in storage.
  const gapWgVars =
    d.kind === "real-fake"
      ? /* wgsl */ `
var<workgroup> wgZ : array<f32, ${d.K}>;
var<workgroup> wgH : array<f32, ${d.hidden}>;
var<workgroup> wgPre : array<f32, ${d.hidden}>;
var<workgroup> wgDz : array<f32, ${d.K}>;
`
      : "";
  const wgBytes = pixelCritWorkgroupBytes(d);
  if (wgBytes > 16384) {
    throw new Error(
      `pixel_disc: ${wgBytes}B of workgroup storage exceeds the 16384B mobile limit`
    );
  }

  const criticDiscBody = emitCriticDiscBody(d, metaStats);
  const criticGenBody = emitCriticGenBody(d, metaStats);

  return /* wgsl */ `
struct Uni {
  width : f32,
  height : f32,
  alpha : f32,
  dt : f32,
  b : u32,
  partCount : u32,
  cursor : u32,
  applyDisc : u32,
  lr : f32,
  beta1 : f32,
  beta2 : f32,
  eps : f32,
  adamT : u32,
  genSign : f32,
  maskSeed : u32,
  pad1 : u32,
};
@group(0) @binding(0) var<uniform> uni : Uni;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> critW : array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch : array<f32>;
@group(0) @binding(4) var<storage, read_write> densI32 : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> densPack : array<f32>;
@group(0) @binding(6) var<storage, read_write> critMeta : array<f32>;
@group(0) @binding(7) var<storage, read_write> extGrads : array<f32>;
@group(0) @binding(8) var<storage, read> partPos : array<f32>;

const N_CELL : u32 = ${nCell}u;
const G : u32 = ${d.G}u;
const DENS_SCALE : f32 = ${fl(PIXEL_DISC_DENS_SCALE)};
const SOFT_EPS2 : f32 = ${fl(PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS)};

${COMMON}

// NaN =/= itself; +/-Inf exceeds the largest finite f32. Same helper the
// relational adversary emits (adversary_wgsl.ts) so the shared extGrads seam is
// guarded identically on both sides — see fieldGrad below.
fn isFiniteF(x : f32) -> bool {
  return x == x && abs(x) <= 3.402823466e+38;
}
${encodeFn}
${fwdStores}
${bwdStores}
${hashFns}
${bceFn}

fn dens_at(i : u32) -> f32 { return densPack[i]; }
fn set_dens(i : u32, v : f32) { densPack[i] = v; }
fn d_dens(i : u32) -> f32 { return densPack[N_CELL + i]; }
fn set_d_dens(i : u32, v : f32) { densPack[N_CELL + i] = v; }
fn add_d_dens(i : u32, v : f32) { densPack[N_CELL + i] = densPack[N_CELL + i] + v; }
fn auxA(i : u32) -> f32 { return densPack[2u * N_CELL + i]; }
fn set_auxA(i : u32, v : f32) { densPack[2u * N_CELL + i] = v; }
fn auxB(i : u32) -> f32 { return densPack[3u * N_CELL + i]; }
fn set_auxB(i : u32, v : f32) { densPack[3u * N_CELL + i] = v; }

${critAccessors}
${wgReduceFn}
${gapWgVars}
${gdFn}
${forceGridFn}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn clearDens(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  atomicStore(&densI32[i], 0);
  densPack[i] = 0.0;
  densPack[N_CELL + i] = 0.0;
}

/** Zero densI32 only — keeps densPack dens (real) for real-fake fake splat. */
@compute @workgroup_size(${PIXEL_DISC_WG})
fn clearAtomics(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  atomicStore(&densI32[i], 0);
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn sampleAndSplat(@builtin(global_invocation_id) gid : vec3u) {
  let s = gid.x;
  if (s >= uni.b) { return; }
  let pBase = s * ${partStride}u;
  let idx = (uni.cursor + s) % max(uni.partCount, 1u);
  let px = partPos[idx * 2u];
  let py = partPos[idx * 2u + 1u];
  let uk = vec2f(px / uni.width, py / uni.height);
  scratch[pBase + ${oPos}u] = uk.x;
  scratch[pBase + ${oPos}u + 1u] = uk.y;
  ${enc.kind === "raw" ? "" : `encodeSite(uk, ${encBase()});`}
  let F = ${fieldForward};
  scratch[pBase + ${oF}u] = F.x;
  scratch[pBase + ${oF}u + 1u] = F.y;

  let scale = f32(G);
  var fx = uk.x * scale - 0.5;
  var fy = uk.y * scale - 0.5;
  fx = clamp(fx, 0.0, f32(G) - 1.0000001);
  fy = clamp(fy, 0.0, f32(G) - 1.0000001);
  let ix = u32(floor(fx));
  let iy = u32(floor(fy));
  let tx = fx - f32(ix);
  let ty = fy - f32(iy);
  let ix1 = min(ix + 1u, G - 1u);
  let iy1 = min(iy + 1u, G - 1u);
  let mass = f32(G * G) / f32(uni.b);
  atomicAdd(&densI32[iy * G + ix], i32((1.0 - tx) * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy * G + ix1], i32(tx * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix], i32((1.0 - tx) * ty * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix1], i32(tx * ty * mass * DENS_SCALE));
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn densToFloat(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  densPack[i] = f32(atomicLoad(&densI32[i])) / DENS_SCALE;
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn copyDensToAux(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  set_auxA(i, dens_at(i));
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn fakeSplat(@builtin(global_invocation_id) gid : vec3u) {
  let s = gid.x;
  if (s >= uni.b) { return; }
  let seed = uni.maskSeed + s * 2654435761u;
  let rx = mulberry32(seed);
  let ry = mulberry32(seed + 1013904223u);
  let scale = f32(G);
  var fx = rx * scale - 0.5;
  var fy = ry * scale - 0.5;
  fx = clamp(fx, 0.0, f32(G) - 1.0000001);
  fy = clamp(fy, 0.0, f32(G) - 1.0000001);
  let ix = u32(floor(fx));
  let iy = u32(floor(fy));
  let tx = fx - f32(ix);
  let ty = fy - f32(iy);
  let ix1 = min(ix + 1u, G - 1u);
  let iy1 = min(iy + 1u, G - 1u);
  let mass = f32(G * G) / f32(uni.b);
  atomicAdd(&densI32[iy * G + ix], i32((1.0 - tx) * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy * G + ix1], i32(tx * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix], i32((1.0 - tx) * ty * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix1], i32(tx * ty * mass * DENS_SCALE));
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn densToFloatFake(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  set_auxA(i, f32(atomicLoad(&densI32[i])) / DENS_SCALE);
}

/**
 * One workgroup, every cell — dispatched as ONE workgroup so the phases can be
 * separated by workgroupBarrier/storageBarrier instead of extra dispatches.
 *
 * Every loop over cells is grid-strided rather than an out-of-range early
 * return for a hard reason: the barriers below (and inside wgSum) must be
 * reached by every invocation, so no lane may leave early.
 */
@compute @workgroup_size(${PIXEL_DISC_WG})
fn criticDisc(@builtin(local_invocation_index) tid : u32) {
  ${criticDiscBody}
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn discAdam(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x;
  if (t >= ${nW}u) { return; }
  let g = critMeta[${metaGrads}u + t];
  if (uni.applyDisc == 0u) { return; }
  let mm = uni.beta1 * critMeta[${metaM}u + t] + (1.0 - uni.beta1) * g;
  let vv = uni.beta2 * critMeta[${metaV}u + t] + (1.0 - uni.beta2) * g * g;
  critMeta[${metaM}u + t] = mm;
  critMeta[${metaV}u + t] = vv;
  let tf_ = f32(uni.adamT);
  let mhat = mm / (1.0 - pow(uni.beta1, tf_));
  let vhat = vv / (1.0 - pow(uni.beta2, tf_));
  critW[t] = critW[t] - uni.lr * mhat / (sqrt(vhat) + uni.eps);
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn clearDensGen(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  atomicStore(&densI32[i], 0);
  densPack[i] = 0.0;
  densPack[N_CELL + i] = 0.0;
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn virtualSplat(@builtin(global_invocation_id) gid : vec3u) {
  let s = gid.x;
  if (s >= uni.b) { return; }
  let pBase = s * ${partStride}u;
  let uk = vec2f(scratch[pBase + ${oPos}u], scratch[pBase + ${oPos}u + 1u]);
  let F = vec2f(scratch[pBase + ${oF}u], scratch[pBase + ${oF}u + 1u]);
  let up = uk + uni.dt * F;
  scratch[pBase + ${oPos2}u] = up.x;
  scratch[pBase + ${oPos2}u + 1u] = up.y;
  let scale = f32(G);
  var fx = up.x * scale - 0.5;
  var fy = up.y * scale - 0.5;
  fx = clamp(fx, 0.0, f32(G) - 1.0000001);
  fy = clamp(fy, 0.0, f32(G) - 1.0000001);
  let ix = u32(floor(fx));
  let iy = u32(floor(fy));
  let tx = fx - f32(ix);
  let ty = fy - f32(iy);
  let ix1 = min(ix + 1u, G - 1u);
  let iy1 = min(iy + 1u, G - 1u);
  let mass = f32(G * G) / f32(uni.b);
  atomicAdd(&densI32[iy * G + ix], i32((1.0 - tx) * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy * G + ix1], i32(tx * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix], i32((1.0 - tx) * ty * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix1], i32(tx * ty * mass * DENS_SCALE));
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn criticGen(@builtin(local_invocation_index) tid : u32) {
  for (var i = tid; i < N_CELL; i = i + ${PIXEL_DISC_WG}u) {
    densPack[i] = f32(atomicLoad(&densI32[i])) / DENS_SCALE;
    densPack[N_CELL + i] = 0.0;
  }
  ${BAR}
  ${criticGenBody}
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn densityVjpAndFieldBwd(@builtin(global_invocation_id) gid : vec3u) {
  let s = gid.x;
  if (s >= uni.b) { return; }
  let pBase = s * ${partStride}u;
  let up = vec2f(scratch[pBase + ${oPos2}u], scratch[pBase + ${oPos2}u + 1u]);
  let scale = f32(G);
  var fx = up.x * scale - 0.5;
  var fy = up.y * scale - 0.5;
  let insideX = fx > 0.0 && fx < f32(G) - 1.0;
  let insideY = fy > 0.0 && fy < f32(G) - 1.0;
  fx = clamp(fx, 0.0, f32(G) - 1.0000001);
  fy = clamp(fy, 0.0, f32(G) - 1.0000001);
  let ix = u32(floor(fx));
  let iy = u32(floor(fy));
  let tx = fx - f32(ix);
  let ty = fy - f32(iy);
  let ix1 = min(ix + 1u, G - 1u);
  let iy1 = min(iy + 1u, G - 1u);
  let mass = f32(G * G) / f32(uni.b);
  let g00 = densPack[N_CELL + iy * G + ix];
  let g10 = densPack[N_CELL + iy * G + ix1];
  let g01 = densPack[N_CELL + iy1 * G + ix];
  let g11 = densPack[N_CELL + iy1 * G + ix1];
  var dfx = mass * (-(1.0 - ty) * g00 + (1.0 - ty) * g10 - ty * g01 + ty * g11);
  var dfy = mass * (-(1.0 - tx) * g00 - tx * g10 + (1.0 - tx) * g01 + tx * g11);
  if (!insideX) { dfx = 0.0; }
  if (!insideY) { dfy = 0.0; }
  let dPosX = dfx * scale;
  let dPosY = dfy * scale;
  let dSig = vec2f(uni.dt * dPosX, uni.dt * dPosY);
  scratch[pBase + ${oDF}u] = dSig.x;
  scratch[pBase + ${oDF}u + 1u] = dSig.y;
  ${fieldBackward}
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn fieldGrad(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x;
  if (t >= ${field.totalFloats}u) { return; }
  var g = 0.0;
${fieldBlocks.join("\n")}
  // Drop a non-finite gradient before it can reach the field's Adam — the guard
  // the relational adversary already applies at this IDENTICAL seam
  // (adversary_wgsl.ts fieldGrad; both sums land in train_wgsl's pass B).
  //
  // It matters MORE here. Every Pixel piece ships ZERO_FIELD_LOSS
  // (docs/PIXEL_DISC.md), which makes the pixel critic the SOLE gradient into
  // the field by construction: there is no field loss upstream to dilute one
  // poisoned float, so a single NaN turns the entire field NaN on the next step
  // and it never recovers.
  extGrads[t] = select(0.0, g, isFiniteF(g));
}
`;
}
