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

import {
  CLASS_SALT,
  encodingPlanes,
  type FieldLayout,
  type HeadSpec,
} from "./advect_wgsl";
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
  pixelDiscWeightCount,
  headFloatsPerGuess,
  pixelGuessCount,
  guessesOf,
  resolvePixelDims as resolvePixelDimsCore,
  validatePixelDims,
  type PixelDiscDims,
  type PixelGanKind,
} from "../../core/gan/pixel_disc";
import { wtaScalars } from "../../core/gan/wta";

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
 * Floats in the stats region: the named scalars plus one WIN COUNTER per guess.
 */
export function pixelStatsFloats(winCounters = 0): number {
  return PIXEL_STATS_WIN_BASE + winCounters;
}

/**
 * Win counters this kind's `criticDisc` actually writes — plan §3f, the reason
 * the stats region was built with a parameter.
 *
 * A guesses knob without this ships a feature whose failure mode is INVISIBLE:
 * a starved guess receives only the ε loser share, never moves, never wins, and
 * K silently degrades to 1 while the loss looks fine. The counter is the only
 * thing that distinguishes "a mixture" from "one head and K−1 passengers".
 *
 * `real-fake` gets ZERO counters, not a counter that is always 1: it has no
 * per-cell residual and therefore no winner (see `validatePixelDims`), and a
 * stats slot no kernel writes is precisely the defect PIXEL_STATS documents.
 */
export function pixelWinCounters(d: PixelDiscDims): number {
  return d.kind === "real-fake" ? 0 : pixelGuessCount(d);
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

/**
 * ALIAS of the oracle's κ, not a second one. This used to be a hand-copied
 * duplicate of `resolvePixelDims` from src/core/gan/pixel_disc.ts; with guesses
 * in `PixelDiscDims` that copy would be a second place to forget
 * {@link validatePixelDims}, i.e. a way for the GPU trainer to build a shader
 * for a (kind, guesses) pair the CPU oracle refuses.
 */
export const resolvePixelDims: (
  partial?: Partial<PixelDiscDims> & { kind?: PixelGanKind }
) => PixelDiscDims = resolvePixelDimsCore;

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
  // The FAMILY gate is the relational adversary's, verbatim
  // (validateAdversaryFusion, adversary_wgsl.ts) rather than a second ladder:
  // one-hot channels widen head 1's layer-0 input and this shader's field
  // backward has no counterpart for those rows, while a family-planed hashgrid
  // only moves the grid's cell index and therefore rides the dEnc machinery
  // the generator reward already uses. `classes > 0` is NOT the question —
  // asking it refused `familyHashgrid` for the same reason it refused the
  // genuinely-unsupported one-hot route.
  switch (field.family.tag) {
    case "none":
    case "grid-plane":
      break;
    case "onehot":
      return {
        tag: "unsupported",
        reason:
          "one-hot class channels widen head 1's layer-0 input and the pixel " +
          "critic's field backward has no counterpart for those rows — use a " +
          "family-planed hashgrid field",
      };
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

/**
 * ONE per-particle scratch block, as data — the offsets the shader bakes in and
 * the stride {@link pixelParticleScratchFloats} allocates come from the same
 * object, so a new region cannot be indexed without also being reserved.
 *
 * The hand-rolled `oEnc = 8` / `oField = 8 + encStore` pair this replaced had
 * no room between them for hashgrid's per-site `dL/dEnc` block, which is the
 * layout half of why the pixel critic refused hashgrid at all
 * (docs/PLAN_PIXEL_GENERATOR_ARCH.md §2a).
 *
 * Same discipline as `trainScratchLayout`/`advScratchLayout`: `encStore` and
 * `dEncStore` are 0 for the encodings that do not use them, so the raw and
 * fourier strides — and therefore every already-shipped pixel shader — are
 * byte-identical to the pre-hashgrid emitter. `oCls` is appended LAST for the
 * same reason: a family-planed field grows the block at the end rather than
 * moving any existing offset.
 */
export interface PixelPartLayout {
  /** normalized sample position u (2) */
  oPos: number;
  /** F(u) (2) */
  oF: number;
  /** virtual advected position u + dt·F (2) */
  oPos2: number;
  /** dL/dF, the density VJP's output (2) */
  oDF: number;
  /** γ(u) — encoded layouts only (`encStore`) */
  oEnc: number;
  /** dL/dγ(u) — hashgrid only (`dEncStore`), read by fieldGrad's grid scatter */
  oDEnc: number;
  /** per-head activation/δ blocks (`siteBlk`) */
  oField: number;
  /**
   * Family label as f32 — family-planed grids only. Stored rather than
   * re-derived in fieldGrad for the reason advScratchLayout gives: a SECOND
   * copy of `pcg(i ^ CLASS_SALT) % C` is how a particle gets advected by one
   * family's plane and scattered into another's.
   */
  oCls: number;
  /** floats per particle */
  stride: number;
}

export function pixelPartLayout(field: FieldLayout): PixelPartLayout {
  const sl = trainScratchLayout(field, 1);
  const oPos = 0;
  const oF = 2;
  const oPos2 = 4;
  const oDF = 6;
  const oEnc = 8;
  const oDEnc = oEnc + sl.encStore;
  const oField = oDEnc + sl.dEncStore;
  const oCls = oField + sl.siteBlk;
  const stride = oCls + (encodingPlanes(field.encoding) > 1 ? 1 : 0);
  return { oPos, oF, oPos2, oDF, oEnc, oDEnc, oField, oCls, stride };
}

export function pixelParticleScratchFloats(field: FieldLayout): number {
  return pixelPartLayout(field).stride;
}

/**
 * f32s of critic field-eval workspace per grid cell (encoding + site block).
 *
 * NO dEnc region, deliberately, even on a hashgrid field: the only writer of
 * these blocks is `fillForceGrid`, which evaluates F at cell centres FORWARD
 * only. Its output is vec-field's TARGET (auxA/auxB), a constant on the tape —
 * `criticGen` differentiates the critic's prediction, never the target — so no
 * `bwd_head_*` call is ever made against a critic site and nothing would ever
 * read a dEnc block here. Reserving one would be dead scratch proportional to
 * G², which is the one allocation in this file that scales with the grid.
 */
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
  return (pixelCritWorkBase(field, batchCap, dims) + pixelCritWorkFloats(dims)) * 4;
}

/**
 * Every head offset below is GUESS 0's. Guess `j` lives at `offset + j·headStride`
 * — the head is guess-major and the `convW | convB | code | head` packing is
 * otherwise unchanged (plan §3a).
 */
interface PixelHeadStride {
  /** Floats between consecutive guesses' head slices. */
  headStride: number;
  /** Guess count. `1` unless `dims.guesses` says otherwise. */
  guesses: number;
}

export type PixelWeightLayout = PixelHeadStride &
  (
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
      }
  );

/** Packed critic weight offsets — matches packPixelDiscWeights / headFloats(kind). */
export function pixelWeightLayout(d: PixelDiscDims): PixelWeightLayout {
  let o = 0;
  const convW = o;
  o += d.E * 9;
  const convB = o;
  o += d.E;
  const code = o;
  o += d.K * d.E;
  const guesses = pixelGuessCount(d);
  const headStride = headFloatsPerGuess(d);
  const total = o + guesses * headStride;
  const stride = { headStride, guesses };
  switch (d.kind) {
    case "vec-field": {
      const headW = o;
      o += 2 * d.K;
      const headB = o;
      return { kind: "vec-field", convW, convB, code, headW, headB, total, ...stride };
    }
    case "next-frame":
    case "inpaint": {
      const headW = o;
      o += d.K;
      const headB = o;
      return { kind: d.kind, convW, convB, code, headW, headB, total, ...stride };
    }
    case "real-fake": {
      const mlp0W = o;
      o += d.hidden * d.K;
      const mlp0B = o;
      o += d.hidden;
      const mlp1W = o;
      o += d.hidden;
      const mlp1B = o;
      return {
        kind: "real-fake",
        convW,
        convB,
        code,
        mlp0W,
        mlp0B,
        mlp1W,
        mlp1B,
        total,
        ...stride,
      };
    }
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

/**
 * The relaxed-WTA fold's compile-time constants, as WGSL literals.
 *
 * The two scalars come from {@link wtaScalars} — this emitter does NOT re-derive
 * `ε/(k−1)`, which is the whole point of src/core/gan/wta.ts existing. `single`
 * yields `{winner: 1, loser: 0}` as CONSTANTS, so the emitted fold collapses to
 * exactly today's single-head arithmetic (`w·r` with `w = 1.0`) rather than to a
 * division by `k−1 = 0`.
 */
interface GuessEmit {
  /**
   * Guess count, baked as a loop bound.
   *
   * AT P = 1 THE EMITTED ARITHMETIC IS THE PRE-GUESSES ARITHMETIC, term for
   * term: `winner = 1`, `loser = 0`, the fold runs once, and every extra
   * operation is a multiply by an exact 1.0 or an add of an exact 0.0. Measured
   * 2026-08-22 against HEAD's emitter, one step at a fresh critic, comparing
   * `critMeta[0..nW)` float by float:
   *
   *   G=16 E=8 K=16 (the SHIPPED config)  vec-field / next-frame / inpaint:
   *     0 floats differ — bit-identical.
   *   G=8 E=4 K=8 (the small test config):
   *     inpaint    0 floats differ
   *     next-frame 20/81 differ, ≤ 4 ulp
   *     vec-field  36/90 differ, ≤ 27 ulp, max |Δ| 1.5e-8 against a scale of 0.88
   *
   * The small-config deltas are the Metal compiler contracting and reassociating
   * an ARITHMETICALLY IDENTICAL expression differently once the K and E loops are
   * short enough to fully unroll — not a semantic change; the shipped config is
   * exact. They matter only because 60 warm Adam steps amplify a 1-ulp gradient
   * difference into the sixth decimal of the equivalence test's printed numbers.
   * Do not chase them by special-casing P === 1 into a second emission path: that
   * would leave the single-guess path unexercised by every multi-guess test.
   */
  P: number;
  /** Head stride in floats, as a WGSL u32 literal suffixless number. */
  stride: number;
  /** `1 − ε` as an f32 literal. */
  winW: string;
  /** `ε/(P−1)` as an f32 literal. */
  loserW: string;
}

function guessEmit(d: PixelDiscDims): GuessEmit {
  const g = guessesOf(d);
  const { winner, loser } = wtaScalars(g);
  return {
    P: pixelGuessCount(d),
    stride: headFloatsPerGuess(d),
    winW: fl(winner),
    loserW: fl(loser),
  };
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
  /**
   * Per-cell head grads, ALREADY multiplied by the guess's WTA weight:
   * `hGradSlots(d)·nCell` floats — 2 per guess for vec-field (dvx,dvy), 1 per
   * guess otherwise (dPred), none for real-fake (its head grads reduce through
   * the MLP, not per cell).
   */
  hg: number;
  /** winIdx[nCell] as f32, or 0 slots where the kind has no per-cell winner. */
  win: number;
  total: number;
}

/** Per-cell head-gradient slots, one set per guess. */
function hGradSlots(d: PixelDiscDims): number {
  const P = pixelGuessCount(d);
  switch (d.kind) {
    case "vec-field":
      return 2 * P;
    case "next-frame":
    case "inpaint":
      return P;
    case "real-fake":
      return 0;
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

/** 1 winner slot per cell, except for the kind that has no per-cell winner. */
function winSlots(d: PixelDiscDims): number {
  return d.kind === "real-fake" ? 0 : 1;
}

function critWorkLayout(d: PixelDiscDims): CritWorkLayout {
  const n = d.G * d.G;
  const feat = 0;
  const soft = feat + d.E * n;
  const dsoft = soft + d.K * n;
  const dfeat = dsoft + d.K * n;
  const hg = dfeat + d.E * n;
  const win = hg + hGradSlots(d) * n;
  return { feat, soft, dsoft, dfeat, hg, win, total: win + winSlots(d) * n };
}

/** f32s of per-cell critic workspace appended to `scratch`. */
export function pixelCritWorkFloats(d: PixelDiscDims): number {
  return critWorkLayout(d).total;
}

/**
 * Float index of `criticDisc`'s per-cell winning-guess array inside `scratch`,
 * absolute. `null` for real-fake, which has no per-cell winner.
 *
 * §3d wanted `var winIdx : array<u32, nCell>` because the critics were
 * `@workgroup_size(1)` and private memory was the binding constraint. They are
 * cell-PARALLEL now, so the winner never has to outlive the cell iteration that
 * computes it and the math needs no array at all. It is still written, for one
 * reason: `tools/pixel_disc_equiv_test.ts` compares the SELECTION cell by cell
 * against the CPU oracle, and a winner mismatch can partially cancel in the
 * summed gradient. nCell f32 of observability.
 */
export function pixelCritWinBase(
  field: FieldLayout,
  batchCap: number,
  d: PixelDiscDims
): number | null {
  if (d.kind === "real-fake") return null;
  return pixelCritWorkBase(field, batchCap, d) + critWorkLayout(d).win;
}

/**
 * Float index where the per-cell critic workspace starts inside `scratch`:
 * after the per-particle blocks, the 8-float pad, and the field-eval sites.
 * ONE definition, used by both {@link pixelScratchBytes} and the shader's baked
 * accessors, so the allocation and the indexing cannot drift.
 */
export function pixelCritWorkBase(
  field: FieldLayout,
  batchCap: number,
  d: PixelDiscDims
): number {
  return (
    batchCap * pixelParticleScratchFloats(field) +
    8 +
    pixelCritSites(d) * pixelCritSiteStride(field)
  );
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
${
    winSlots(d) === 0
      ? ""
      : `fn setWinIdx(c : u32, j : u32) { scratch[${at(cw.win)} + c] = f32(j); }`
  }
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
function emitGapMlpFwdPar(d: PixelDiscDims, age = "0.0", era = "0.0"): string {
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
  if (tid == 0u) {
    wgZ[0] = wgZ[0] + ${age};
    ${K > 1 ? `wgZ[1] = wgZ[1] + ${era};` : ""}
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
  const g = guessEmit(d);
  const winBase = metaStats + PIXEL_STATS_WIN_BASE;
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
  var lBx : array<f32, ${g.P}>;
  var lBy : array<f32, ${g.P}>;
  var lWin : array<f32, ${g.P}>;
  for (var j = 0u; j < ${g.P}u; j = j + 1u) { lBx[j] = 0.0; lBy[j] = 0.0; lWin[j] = 0.0; }
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    // !(dens < FLOOR), not dens >= FLOOR: the serial kernel skipped with
    // "if (dens < FLOOR) { continue; }", and the two disagree on NaN.
    let act = !(dens_at(c) < ${fl(DENS_FLOOR)});
    let tgx = auxA(c);
    let tgy = auxB(c);
    // The TRUNK above ran once; every guess reads the same cSoft (plan §3a).
    var ex : array<f32, ${g.P}>;
    var ey : array<f32, ${g.P}>;
    var rr : array<f32, ${g.P}>;
    var win = 0u;
    var best = 0.0;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let hw = ${wl.headW}u + j * ${g.stride}u;
      let hb = ${wl.headB}u + j * ${g.stride}u;
      var vx = critW[hb];
      var vy = critW[hb + 1u];
      for (var k = 0u; k < ${K}u; k = k + 1u) {
        let q = cSoftAt(k, c);
        vx = vx + critW[hw + k] * q;
        vy = vy + critW[hw + ${K}u + k] * q;
      }
      let dx = vx - tgx;
      let dy = vy - tgy;
      let r = sqrt(dx * dx + dy * dy + SOFT_EPS2);
      ex[j] = dx;
      ey[j] = dy;
      rr[j] = r;
      // STRICT <, ascending: a later equal residual does not displace the
      // incumbent, i.e. ties route to the LOWEST guess index (wta.ts).
      if (j == 0u || r < best) { best = r; win = j; }
    }
    setWinIdx(c, win);
    // §3g: gated by the SAME predicate that gates the residual. An inactive cell
    // still has a mathematical argmin, and counting it would make guess 0 look
    // dominant on a mostly-empty density grid — the collapse detector would lie.
    lWin[win] = lWin[win] + select(0.0, 1.0, act);
    var payoff = 0.0;
    var dvx : array<f32, ${g.P}>;
    var dvy : array<f32, ${g.P}>;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      // CONSTANT w.r.t. the gradient — selection is off-tape.
      let wj = select(${g.loserW}, ${g.winW}, j == win);
      payoff = payoff + wj * rr[j];
      let s = 1.0 / rr[j];
      let gx = select(0.0, wj * (ex[j] * s), act);
      let gy = select(0.0, wj * (ey[j] * s), act);
      dvx[j] = gx;
      dvy[j] = gy;
      setHGrad(2u * j, c, gx);
      setHGrad(2u * j + 1u, c, gy);
      lBx[j] = lBx[j] + gx;
      lBy[j] = lBy[j] + gy;
    }
    lAct = lAct + select(0.0, 1.0, act);
    lR = lR + select(0.0, payoff, act);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      var acc = 0.0;
      for (var j = 0u; j < ${g.P}u; j = j + 1u) {
        let hw = ${wl.headW}u + j * ${g.stride}u;
        acc = acc + dvx[j] * critW[hw + k] + dvy[j] * critW[hw + ${K}u + k];
      }
      setDSoft(k, c, acc);
    }
  }
  let nActive = wgSum(tid, lAct);
  let sumR = wgSum(tid, lR);
  // wgSum contains barriers, so these loops must stay UNIFORM — the bound is a
  // baked literal and no lane leaves early, which is what makes that safe.
  var sumBx : array<f32, ${g.P}>;
  var sumBy : array<f32, ${g.P}>;
  var winTot : array<f32, ${g.P}>;
  for (var j = 0u; j < ${g.P}u; j = j + 1u) {
    sumBx[j] = wgSum(tid, lBx[j]);
    sumBy[j] = wgSum(tid, lBy[j]);
    winTot[j] = wgSum(tid, lWin[j]);
  }
  let activeN = max(nActive, 1.0);
  if (tid == 0u) {
    critMeta[${metaStats + PIXEL_STATS.discLoss}u] = sumR / activeN;
    // auxA/auxB hold the TARGET force at each cell centre (fillForceGrid), so
    // these two are F(cell 0), not a prediction — see PIXEL_STATS.targetFx.
    critMeta[${metaStats + PIXEL_STATS.targetFx}u] = auxA(0u);
    critMeta[${metaStats + PIXEL_STATS.targetFy}u] = auxB(0u);
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let hb = ${wl.headB}u + j * ${g.stride}u;
      critMeta[hb] = critMeta[hb] + sumBx[j] / activeN;
      critMeta[hb + 1u] = critMeta[hb + 1u] + sumBy[j] / activeN;
      critMeta[${winBase}u + j] = winTot[j];
    }
  }
  for (var t = tid; t < ${g.P * K}u; t = t + ${PIXEL_DISC_WG}u) {
    let j = t / ${K}u;
    let k = t % ${K}u;
    let hw = ${wl.headW}u + j * ${g.stride}u;
    var sx = 0.0;
    var sy = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) {
      let q = cSoftAt(k, c);
      sx = sx + hGradAt(2u * j, c) * q;
      sy = sy + hGradAt(2u * j + 1u, c) * q;
    }
    critMeta[hw + k] = critMeta[hw + k] + sx / activeN;
    critMeta[hw + ${K}u + k] = critMeta[hw + ${K}u + k] + sy / activeN;
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
  var lB : array<f32, ${g.P}>;
  var lWin : array<f32, ${g.P}>;
  for (var j = 0u; j < ${g.P}u; j = j + 1u) { lB[j] = 0.0; lWin[j] = 0.0; }
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    let tgt = dens_at(c);
    var dd : array<f32, ${g.P}>;
    var rr : array<f32, ${g.P}>;
    var win = 0u;
    var best = 0.0;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let hw = ${wl.headW}u + j * ${g.stride}u;
      var pred = critW[${wl.headB}u + j * ${g.stride}u];
      for (var k = 0u; k < ${K}u; k = k + 1u) {
        pred = pred + critW[hw + k] * cSoftAt(k, c);
      }
      let diff = pred - tgt;
      let r = sqrt(diff * diff + SOFT_EPS2);
      dd[j] = diff;
      rr[j] = r;
      if (j == 0u || r < best) { best = r; win = j; }
    }
    setWinIdx(c, win);
    // §3g: next-frame's activity predicate is "all cells", so every winner counts.
    lWin[win] = lWin[win] + 1.0;
    var payoff = 0.0;
    var dp : array<f32, ${g.P}>;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let wj = select(${g.loserW}, ${g.winW}, j == win);
      payoff = payoff + wj * rr[j];
      let dPred = wj * (dd[j] / rr[j] / f32(N_CELL));
      dp[j] = dPred;
      setHGrad(j, c, dPred);
      lB[j] = lB[j] + dPred;
    }
    lR = lR + payoff;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      var acc = 0.0;
      for (var j = 0u; j < ${g.P}u; j = j + 1u) {
        acc = acc + dp[j] * critW[${wl.headW}u + j * ${g.stride}u + k];
      }
      setDSoft(k, c, acc);
    }
  }
  let sumR = wgSum(tid, lR);
  var sumB : array<f32, ${g.P}>;
  var winTot : array<f32, ${g.P}>;
  for (var j = 0u; j < ${g.P}u; j = j + 1u) {
    sumB[j] = wgSum(tid, lB[j]);
    winTot[j] = wgSum(tid, lWin[j]);
  }
  if (tid == 0u) {
    critMeta[${metaStats + PIXEL_STATS.discLoss}u] = sumR / f32(N_CELL);
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let hb = ${wl.headB}u + j * ${g.stride}u;
      critMeta[hb] = critMeta[hb] + sumB[j];
      critMeta[${winBase}u + j] = winTot[j];
    }
  }
  for (var t = tid; t < ${g.P * K}u; t = t + ${PIXEL_DISC_WG}u) {
    let j = t / ${K}u;
    let k = t % ${K}u;
    var s = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) { s = s + hGradAt(j, c) * cSoftAt(k, c); }
    let wi = ${wl.headW}u + j * ${g.stride}u + k;
    critMeta[wi] = critMeta[wi] + s;
  }
  ${BAR}
  ${emitBwdActPar(d)}
  ${emitBackboneWeightGrads(d, densAux)}
`;
    }
    case "real-fake": {
      if (wl.kind !== "real-fake") throw new Error("layout mismatch");
      // NO GUESS FOLD and no win counters, by decision rather than by omission:
      // `validatePixelDims` (src/core/gan/pixel_disc.ts) rejects guesses > 1 for
      // this kind, so `wl.guesses` is 1 here and the head below is one head. The
      // reasoning — BCE against a LABEL has one right answer, the generator pass
      // has no winner to inherit, and there is no per-cell predicate to gate a
      // §3g counter with — is written out at that validator.
      return /* wgsl */ `
  ${emitZeroGrads(nW)}
  var discLoss = 0.0;
  {
    ${emitFwdPar(d, dens0)}
    ${emitGapMlpFwdPar(d, "0.0", "1.0")}
    let bce = bceGradLogit(logit, 1.0);
    discLoss = discLoss + bce.x;
    ${emitGapMlpBwdPar(d, "bce.y", true)}
    ${emitBwdActPar(d)}
    ${emitBackboneWeightGrads(d, dens0)}
  }
  {
    ${emitFwdPar(d, densAux)}
    ${emitGapMlpFwdPar(d, "uni.fakeAge", "uni.fakeEra")}
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
  var lB : array<f32, ${g.P}>;
  var lWin : array<f32, ${g.P}>;
  for (var j = 0u; j < ${g.P}u; j = j + 1u) { lB[j] = 0.0; lWin[j] = 0.0; }
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    let m = auxB(c) > 0.5;
    let tgt = dens_at(c);
    var dd : array<f32, ${g.P}>;
    var rr : array<f32, ${g.P}>;
    var win = 0u;
    var best = 0.0;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let hw = ${wl.headW}u + j * ${g.stride}u;
      var pred = critW[${wl.headB}u + j * ${g.stride}u];
      for (var k = 0u; k < ${K}u; k = k + 1u) {
        pred = pred + critW[hw + k] * cSoftAt(k, c);
      }
      let diff = pred - tgt;
      let r = sqrt(diff * diff + SOFT_EPS2);
      dd[j] = diff;
      rr[j] = r;
      if (j == 0u || r < best) { best = r; win = j; }
    }
    setWinIdx(c, win);
    // §3g: inpaint's activity predicate is the MASK — the same one that gates
    // the residual below. An unmasked cell contributes no loss, so its argmin is
    // not a win.
    lWin[win] = lWin[win] + select(0.0, 1.0, m);
    var payoff = 0.0;
    var dp : array<f32, ${g.P}>;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let wj = select(${g.loserW}, ${g.winW}, j == win);
      payoff = payoff + wj * rr[j];
      let dPred = select(0.0, wj * ((dd[j] / rr[j]) / nMask), m);
      dp[j] = dPred;
      setHGrad(j, c, dPred);
      lB[j] = lB[j] + dPred;
    }
    lR = lR + select(0.0, payoff, m);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      var acc = 0.0;
      for (var j = 0u; j < ${g.P}u; j = j + 1u) {
        acc = acc + dp[j] * critW[${wl.headW}u + j * ${g.stride}u + k];
      }
      setDSoft(k, c, acc);
    }
  }
  let sumR = wgSum(tid, lR);
  var sumB : array<f32, ${g.P}>;
  var winTot : array<f32, ${g.P}>;
  for (var j = 0u; j < ${g.P}u; j = j + 1u) {
    sumB[j] = wgSum(tid, lB[j]);
    winTot[j] = wgSum(tid, lWin[j]);
  }
  if (tid == 0u) {
    critMeta[${metaStats + PIXEL_STATS.discLoss}u] = sumR / nMask;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let hb = ${wl.headB}u + j * ${g.stride}u;
      critMeta[hb] = critMeta[hb] + sumB[j];
      critMeta[${winBase}u + j] = winTot[j];
    }
  }
  for (var t = tid; t < ${g.P * K}u; t = t + ${PIXEL_DISC_WG}u) {
    let j = t / ${K}u;
    let k = t % ${K}u;
    var s = 0.0;
    for (var c = 0u; c < N_CELL; c = c + 1u) { s = s + hGradAt(j, c) * cSoftAt(k, c); }
    let wi = ${wl.headW}u + j * ${g.stride}u + k;
    critMeta[wi] = critMeta[wi] + s;
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
  const g = guessEmit(d);
  const dens0 = "0u";
  const densAux = "2u * N_CELL";

  switch (d.kind) {
    case "vec-field": {
      if (wl.kind !== "vec-field") throw new Error("layout mismatch");
      // The gen pass RE-SELECTS on its own density (the virtual splat), exactly
      // as the CPU oracle's gen block does; the disc pass's winner is not
      // carried over — the two passes see different grids.
      return /* wgsl */ `
  ${emitFwdPar(d, dens0)}
  var lAct = 0.0;
  var lR = 0.0;
  for (var c = tid; c < N_CELL; c = c + ${PIXEL_DISC_WG}u) {
    let act = !(dens_at(c) < ${fl(DENS_FLOOR)});
    let tgx = auxA(c);
    let tgy = auxB(c);
    var ex : array<f32, ${g.P}>;
    var ey : array<f32, ${g.P}>;
    var rr : array<f32, ${g.P}>;
    var win = 0u;
    var best = 0.0;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let hw = ${wl.headW}u + j * ${g.stride}u;
      let hb = ${wl.headB}u + j * ${g.stride}u;
      var vx = critW[hb];
      var vy = critW[hb + 1u];
      for (var k = 0u; k < ${K}u; k = k + 1u) {
        let q = cSoftAt(k, c);
        vx = vx + critW[hw + k] * q;
        vy = vy + critW[hw + ${K}u + k] * q;
      }
      let dx = vx - tgx;
      let dy = vy - tgy;
      let r = sqrt(dx * dx + dy * dy + SOFT_EPS2);
      ex[j] = dx;
      ey[j] = dy;
      rr[j] = r;
      if (j == 0u || r < best) { best = r; win = j; }
    }
    var payoff = 0.0;
    var dvx : array<f32, ${g.P}>;
    var dvy : array<f32, ${g.P}>;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let wj = select(${g.loserW}, ${g.winW}, j == win);
      payoff = payoff + wj * rr[j];
      let s = uni.genSign / rr[j];
      dvx[j] = select(0.0, wj * (ex[j] * s), act);
      dvy[j] = select(0.0, wj * (ey[j] * s), act);
    }
    lAct = lAct + select(0.0, 1.0, act);
    lR = lR + select(0.0, payoff, act);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      var acc = 0.0;
      for (var j = 0u; j < ${g.P}u; j = j + 1u) {
        let hw = ${wl.headW}u + j * ${g.stride}u;
        acc = acc + dvx[j] * critW[hw + k] + dvy[j] * critW[hw + ${K}u + k];
      }
      setDSoft(k, c, acc);
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
    let tgt = dens_at(c);
    var dd : array<f32, ${g.P}>;
    var rr : array<f32, ${g.P}>;
    var win = 0u;
    var best = 0.0;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let hw = ${wl.headW}u + j * ${g.stride}u;
      var pred = critW[${wl.headB}u + j * ${g.stride}u];
      for (var k = 0u; k < ${K}u; k = k + 1u) {
        pred = pred + critW[hw + k] * cSoftAt(k, c);
      }
      let diff = pred - tgt;
      let r = sqrt(diff * diff + SOFT_EPS2);
      dd[j] = diff;
      rr[j] = r;
      if (j == 0u || r < best) { best = r; win = j; }
    }
    var payoff = 0.0;
    var gd = 0.0;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let wj = select(${g.loserW}, ${g.winW}, j == win);
      payoff = payoff + wj * rr[j];
      gd = gd + wj * (uni.genSign * (-dd[j] / rr[j]) / f32(N_CELL));
    }
    lR = lR + payoff;
    add_d_dens(c, gd);
  }
  let sumR = wgSum(tid, lR);
  if (tid == 0u) { critMeta[${metaStats + PIXEL_STATS.genLoss}u] = uni.genSign * (sumR / f32(N_CELL)); }
`;
    }
    case "real-fake": {
      if (wl.kind !== "real-fake") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitFwdPar(d, dens0)}
  ${emitGapMlpFwdPar(d, "0.0", "1.0")}
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
    let tgt = dens_at(c);
    var dd : array<f32, ${g.P}>;
    var rr : array<f32, ${g.P}>;
    var win = 0u;
    var best = 0.0;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let hw = ${wl.headW}u + j * ${g.stride}u;
      var pred = critW[${wl.headB}u + j * ${g.stride}u];
      for (var k = 0u; k < ${K}u; k = k + 1u) {
        pred = pred + critW[hw + k] * cSoftAt(k, c);
      }
      let diff = pred - tgt;
      let r = sqrt(diff * diff + SOFT_EPS2);
      dd[j] = diff;
      rr[j] = r;
      if (j == 0u || r < best) { best = r; win = j; }
    }
    var payoff = 0.0;
    var gd = 0.0;
    var dp : array<f32, ${g.P}>;
    for (var j = 0u; j < ${g.P}u; j = j + 1u) {
      let wj = select(${g.loserW}, ${g.winW}, j == win);
      payoff = payoff + wj * rr[j];
      let scale = wj * (uni.genSign / rr[j] / nMask);
      gd = gd + select(0.0, -dd[j] * scale, m);
      dp[j] = select(0.0, dd[j] * scale, m);
    }
    lR = lR + select(0.0, payoff, m);
    set_d_dens(c, gd);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      var acc = 0.0;
      for (var j = 0u; j < ${g.P}u; j = j + 1u) {
        acc = acc + dp[j] * critW[${wl.headW}u + j * ${g.stride}u + k];
      }
      setDSoft(k, c, acc);
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
  // κ for the (kind, guesses) pair, before a single index is baked. A shader
  // built for a pair the CPU oracle refuses would be unverifiable by
  // construction — tools/pixel_disc_equiv_test.ts could not produce the other
  // side of the comparison.
  validatePixelDims(d);
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

  const { oPos, oF, oPos2, oDF, oEnc, oDEnc, oField, oCls } =
    pixelPartLayout(field);

  /**
   * FAMILY-PLANED grid: the label picks the feature plane, so it is live in
   * `encodeSite`, in `bwd_head_*` and in fieldGrad's grid scatter. `planed` is
   * false for every other field, and each `clsArg`/`planeTerm` below then
   * expands to the empty string — that is what keeps the classless generated
   * text character-for-character what it was.
   */
  const planed = encodingPlanes(field.encoding) > 1;
  // vec-field's target is F evaluated at CELL CENTRES, and a cell centre has no
  // family. On a C-plane field there are C different F's there and no principled
  // way to pick one, so this is a refusal rather than a silent `cls = 0` (which
  // would train the critic against family 0's field and look perfectly healthy).
  // Thrown at the same κ boundary as validatePixelDims' real-fake+guesses
  // refusal: it is a (kind, field) question, not the field-only question
  // classifyPixelDiscFusion answers for the host.
  if (planed && d.kind === "vec-field") {
    throw new Error(
      "pixel_disc: vec-field on a family-planed field — its target is F at " +
        "cell centres, and a cell centre has no family, so there is no one " +
        "field to fit. Use next-frame / real-fake / inpaint on a planed field."
    );
  }

  // Critic field-eval workspace, indexed BY CELL: fillForceGrid runs one cell
  // per invocation, so every cell needs its own encoding + site block. These
  // bases are only referenced from inside fillForceGrid, where `cell` is bound.
  const critWorkspace = batchCap * partStride;
  const critSiteStride = sl.encStore + sl.siteBlk;
  const critSite = `(${critWorkspace + 8}u + cell * ${critSiteStride}u)`;
  const critEncBase = critSite;
  // Per-cell critic workspace lives right after the field-eval sites. Same
  // helper the allocator uses — see pixelScratchBytes.
  const critWorkBase = pixelCritWorkBase(field, batchCap, d);
  const critFieldBase = (h: number) =>
    `${critSite} + ${sl.encStore}u + ${h === 0 ? 0 : sl.headBlk[0]}u`;

  const fieldBase = (h: number) =>
    `pBase + ${oField}u + ${h === 0 ? 0 : sl.headBlk[0]}u`;
  const encBase = () => `pBase + ${oEnc}u`;
  const dEncBase = () => `pBase + ${oDEnc}u`;
  /** `, pcls` on a family-planed grid; empty everywhere else. */
  const clsArg = planed ? `, pcls` : ``;
  const fwdCall = (h: number) =>
    enc.kind === "raw"
      ? `fwd_head_${h}(uk, ${fieldBase(h)}, 0u)`
      : `fwd_head_${h}(${encBase()}, ${fieldBase(h)})`;
  const critFwdCall = (h: number) =>
    enc.kind === "raw"
      ? `fwd_head_${h}(uk, ${critFieldBase(h)}, 0u)`
      : `fwd_head_${h}(${critEncBase}, ${critFieldBase(h)})`;
  // One call shape per encoding kind — emitBwdStore's signature is
  // type-directed the same way (train_wgsl.ts), and the adversary's bwdCall is
  // the same three-armed dispatch. hashgrid takes the stored SITE POSITION
  // (its backward recomputes the corner geometry from it rather than storing
  // four indices) and the dEnc block base, plus the family label when planed.
  const bwdCall = (h: number, dExpr: string) =>
    enc.kind === "raw"
      ? `bwd_head_${h}(${dExpr}, ${fieldBase(h)})`
      : enc.kind === "fourier"
      ? `bwd_head_${h}(${dExpr}, ${fieldBase(h)}, ${encBase()})`
      : `bwd_head_${h}(${dExpr}, ${fieldBase(h)}, ukIn, ${dEncBase()}${clsArg})`;
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
  // Which field head SEEDS the shared per-site dL/dEnc block (hashgrid only).
  // Keyed on the LANE, not on the head index: `fieldBackward` calls both heads
  // on the blend lane and exactly ONE head on a direct lane, so a `fieldLane: 1`
  // game with a head-0-seeds assumption would `+=` into a block nobody wrote —
  // reading whatever the previous step left there. Unreachable while the pixel
  // critic was always "blend" AND hashgrid was refused; reachable the moment
  // either changes (docs/PLAN_PIXEL_GENERATOR_ARCH.md §2a, and the same
  // reasoning the adversary's `dEncSeedHead` carries).
  const dEncSeedHead = fieldLane === "blend" ? 0 : fieldLane;
  const bwdStores = heads
    .map((h, i) =>
      emitBwdStore(i, h, sl, maxW, enc, i === dEncSeedHead ? "seed" : "accumulate")
    )
    .join("\n");
  const encodeFn = enc.kind === "raw" ? "" : emitEncode(enc);
  const critEncInit =
    enc.kind === "raw" ? "" : `encodeSite(uk, ${critEncBase});`;
  /**
   * The particle's family, derived where the particle index is live and STORED,
   * exactly as the advect kernel and both trainers derive it
   * (`pcg(i ^ CLASS_SALT) % C`). fieldGrad reads the stored label rather than
   * re-deriving it: this shader's fieldGrad is indexed by WEIGHT, not by
   * particle, so a second copy of the derivation there is how a particle gets
   * encoded through one plane and scattered into another.
   */
  const clsDerive = planed
    ? `
  let pcls = pcg(idx ^ ${CLASS_SALT}u) % ${field.classes}u;
  scratch[pBase + ${oCls}u] = f32(pcls);`
    : ``;
  /**
   * Backward-site prelude: what hashgrid's `bwd_head_*` reads beyond scratch.
   *
   * Both of these (and `clsDerive` above) carry their OWN leading newline and
   * expand to the empty string otherwise, so the raw and fourier shaders come
   * out byte-identical to the pre-hashgrid emitter rather than gaining a blank
   * line — the same discipline the collapsing scratch offsets follow.
   */
  const bwdSitePrelude =
    enc.kind === "hashgrid"
      ? `
  let ukIn = vec2f(scratch[pBase + ${oPos}u], scratch[pBase + ${oPos}u + 1u]);${
          planed ? `\n  let pcls = u32(scratch[pBase + ${oCls}u]);` : ``
        }`
      : ``;

  const fieldBlocks: string[] = [];
  for (const seg of field.segments) {
    if (seg.role === "grid") {
      // THE hashgrid feature table (field weights offset 0): thread = one grid
      // float (cell, feature). This block used to be `continue` — the table was
      // simply skipped, which is not a missing feature but a SILENT one: every
      // grid float's extGrad stayed exactly 0, the encoding's trainable
      // parameters received no generator reward at all, and nothing anywhere
      // reported it (docs/PLAN_PIXEL_GENERATOR_ARCH.md §2c). Extended by
      // tools/pixel_disc_equiv_test.ts §4, which asserts the GRID slice against
      // tfjs autograd specifically because a test on the MLP floats alone would
      // have passed against `continue`.
      //
      // Gather-side, transliterated from the adversary's fieldGrad and
      // train_wgsl's pass B: each grid float scans the batch and claims the
      // bilinear corners that land on its cell, so there is exactly ONE writer
      // per grid float and no atomics. At the clamp border ix1 == ix makes TWO
      // corner tests match the same cell — both add, which is what tfjs's
      // oneHotᵀ scatter does with coincident indices.
      //
      // ABOVE the fieldLane skip below, deliberately: the table is shared by
      // both field heads and lane isolation already happened upstream — only
      // the active lane's bwd_head wrote dEnc.
      if (enc.kind !== "hashgrid") {
        throw new Error(
          `pixel_disc: grid segment on a ${enc.kind} encoding — layout is inconsistent`
        );
      }
      const { gridSize: gs, features: F } = enc;
      // Family plane: a particle only ever reads ITS OWN plane, so a grid float
      // in plane c must ignore every particle whose family is not c. Folding
      // `cls · gs²` into the cell comparison does that with no extra branch. A
      // missing term still runs and still trains — it just trains the wrong
      // family's features, so §4b of the equiv test gates plane support directly.
      const planeTerm = planed ? `pcls * ${gs * gs}u + ` : ``;
      const clsRead = planed ? `\n      let pcls = u32(scratch[pBase + ${oCls}u]);` : ``;
      fieldBlocks.push(`
  if (t >= ${seg.floatOffset}u && t < ${seg.floatOffset + seg.floatLength}u) {
    let cell = (t - ${seg.floatOffset}u) / ${F}u;
    let f = (t - ${seg.floatOffset}u) % ${F}u;
    for (var s = 0u; s < uni.b; s = s + 1u) {
      let pBase = s * ${partStride}u;${clsRead}
      let ux = scratch[pBase + ${oPos}u];
      let uy = scratch[pBase + ${oPos}u + 1u];
      let gxf = clamp(ux, 0.0, 1.0) * ${(gs - 1).toFixed(1)};
      let gyf = clamp(uy, 0.0, 1.0) * ${(gs - 1).toFixed(1)};
      let ix = u32(floor(gxf)); let iy = u32(floor(gyf));
      let fx = gxf - floor(gxf); let fy = gyf - floor(gyf);
      let ix1 = min(ix + 1u, ${gs - 1}u); let iy1 = min(iy + 1u, ${gs - 1}u);
      var wsum = 0.0;
      if (${planeTerm}iy * ${gs}u + ix == cell) { wsum = wsum + (1.0 - fx) * (1.0 - fy); }
      if (${planeTerm}iy * ${gs}u + ix1 == cell) { wsum = wsum + fx * (1.0 - fy); }
      if (${planeTerm}iy1 * ${gs}u + ix == cell) { wsum = wsum + (1.0 - fx) * fy; }
      if (${planeTerm}iy1 * ${gs}u + ix1 == cell) { wsum = wsum + fx * fy; }
      g = g + wsum * scratch[pBase + ${oDEnc}u + f];
    }
  }`);
      continue;
    }
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
  fakeAge : f32,
  fakeEra : f32,
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
  scratch[pBase + ${oPos}u + 1u] = uk.y;${clsDerive}
  ${enc.kind === "raw" ? "" : `encodeSite(uk, ${encBase()}${clsArg});`}
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
  scratch[pBase + ${oDF}u + 1u] = dSig.y;${bwdSitePrelude}
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
