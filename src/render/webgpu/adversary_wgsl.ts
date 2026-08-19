/**
 * adversary_wgsl — PURE WGSL codegen for the FUSED relaxed-WTA adversary
 * (the port of src/core/gan/adversary.ts, oracle-gated by
 * src/render/webgpu/ad/losses.ts::wtaTerm + tools/ad_wta_test.ts).
 *
 * Two generated modules replace the whole tfjs adversary step (which cost
 * 19-32 ms at B=512 on this machine — K optimizer.minimize closures with K
 * pipeline-stalling readbacks):
 *
 *   PASS A (one thread = one tuple, B threads):
 *     sample m particle indices (stateless pcg, its own mixing constant) →
 *     gather positions/velocities → evaluate the LIVE raw field output F(x) →
 *     select the configured target (raw force, or the exact pre-border
 *     post-update velocity including forceMagnitude, friction and component
 *     clipping) →
 *     tuple encoding κ (point | relational pair variants | tri, matching
 *     adversary.ts to the epsilon) → K adversary head forwards → residuals →
 *     first-argmin winner (strict <, = tf.argMin's lowest-index tie rule) →
 *     relaxed weights → per-head DISCRIMINATOR backward (δ's to scratch,
 *     seeded w_j/B — the relaxed weights are constants w.r.t. the gradient,
 *     exactly the tfjs trainStep semantics B1 verified) → GENERATOR backward:
 *     apply the configured game (zero-sum raw/soft-angle/relative-scale, or
 *     direction-adversarial + scale-cooperative scale-hold with its positive
 *     batch-energy anchor), walk dL/dy through the encoding/physics target to
 *     the member field signals and through the FIELD heads to
 *     per-layer δ's (also to scratch). Per-tuple surprise is scattered into
 *     packed raw/per-unit N-capacity planes (surprise_points.ts renders a
 *     selected plane) and
 *     the batch stats (disc loss, raw payoff, RMS‖y‖, pairwise head spread,
 *     closest-pair spread, win counts) are workgroup-reduced into partials.
 *     A `finalize` entry combines them. The packed surprise output contains
 *     two planes: the unchanged raw payoff and a visualization-only
 *     payoff/‖y‖ diagnostic.
 *
 *   PASS B (two entries):
 *     advOpt   — thread = adversary weight float: dW = Σ_tuples aIn⊗δ, then
 *                in-place Adam on the ADVERSARY buffer (own segment, own
 *                moments — the fused analogue of "one Adam per head, own
 *                varList": Adam state is per-parameter, and the adversary
 *                buffer is disjoint from the field's by construction).
 *     fieldGrad— thread = FIELD weight float: dL_gen/dW = Σ_(tuple,member)
 *                aIn⊗δ_field from the adversary scratch, written to extGrads.
 *                The field's own pass B (train_wgsl.ts, extGrad:true) adds
 *                extGrads[t] into its gradient before its Adam — that is how
 *                the generator reward joins the field's backward.
 *
 * WHY THE GENERATOR BACKWARD IS SMALL. u (pair separation, tri side lengths,
 * point position and, for post-velocity, incoming normalized velocity) is
 * sampled data, so du/dW_field ≡ 0. Force targets map directly from F(x).
 * Post-velocity targets additionally apply the exact diagonal Jacobian of
 * clamp((v + forceMagnitude·F)·friction, ±maxVelocity)/maxVelocity; saturated
 * components therefore contribute zero. Soft-angle uses the exact smooth S²
 * chord derivative, and local scale differentiates raw log-magnitude
 * contrasts without the historical 1/||q|| quotient. Verified against the IR
 * oracle, tfjs, and finite differences in tools/{ad_,train_}wta_test.ts.
 *
 * SCOPE (loud throws, no silent fallback):
 *   - field must be helmholtz / agree-disagree, classes == 0. Every encoding
 *     (raw | fourier | hashgrid) is supported: hashgrid stores per-member
 *     dL/dEnc in its own scratch block and fieldGrad carries the gather-side
 *     grid block transliterated from train_wgsl's field pass B. Class-aware
 *     fields remain unsupported — an orthogonal gap (layer-0 one-hot channels
 *     have no encoded-input counterpart).
 *     NOTE the PIXEL critic (pixel_disc_wgsl.ts) still refuses hashgrid; this
 *     port covers the RELATIONAL adversary only.
 *   - adversary head activations: selu/tanh/sigmoid/linear (no sin — its
 *     backward needs a pre-act checkpoint the adversary layout doesn't carry).
 *
 * ANTI-COLLAPSE PRESSURE ({@link FusedGamePressure}) rides the same generator
 * seam: its dL/dF is added into `dSig` before the field backward, so it reaches
 * every encoding through the SAME dEnc/bwd machinery the reward already uses
 * and pass B needs no pressure-specific code. `none` emits zero text.
 */

import {
  CLASS_SALT,
  encodingDim,
  encodingPlanes,
  type Encoding,
  type FamilyRoute,
  type FieldLayout,
  type HeadSpec,
  type AdversaryLayout,
} from "./advect_wgsl";
import {
  COMMON,
  emitEncode,
  emitFwdStore,
  emitBwdStore,
  trainScratchLayout,
  type TrainScratchLayout,
} from "./train_wgsl";

export const ADV_WG = 128;
export const ADV_WG_B = 64;
export const ADV_MAX_BATCH = 4096;
/** pcg mixing constant for the tuple index stream — distinct from the batch
 *  (2654435769), particle-sample (3266489917) and class (2166136261) streams. */
/** Legacy export retained for callers compiled against the first fused
 * adversary. Live tuple sampling is now a deterministic rotating coverage
 * window, not a replacement PCG stream. */
export const ADV_IDX_MIX = 2447445653;
/** == SOFT_EPS in adversary.ts / WTA_SOFT_EPS in ad/losses.ts, squared (it
 *  sits INSIDE the sqrt radicand). Also used as FRAME_EPS² (same value there). */
export const ADV_SOFT_EPS2 = 1e-12;
/** Mirrors core/gan/adversary.ts DIRECTION_ACTIVE_FLOOR. This is both the
 * semantic dead-zone for directionless targets and a 1e3 cap on the exact
 * target-normalization Jacobian. */
export const ADV_DIRECTION_ACTIVE_FLOOR = 1e-3;
/** Triangle side-length gaps at or below this threshold have no unique
 * permutation-invariant canonical vertex order. Such tuples are explicit
 * inactive data, never label-tie-broken training samples. */
export const ADV_TRI_TIE_EPS = 1e-5;
/** Mirrors core/gan/adversary.ts QUAD_ANCHOR_ACTIVE_FLOOR. */
export const ADV_QUAD_ANCHOR_ACTIVE_FLOOR = 1e-5;
/**
 * Stats buffer: finalized values occupy [0, 7+k):
 * [discLoss, rawPayoff, batchRms, meanPairSpread, closestPairSpread, wins...,
 *  energyRms, energyActive]. Per-workgroup partials begin at 32 with stride
 *  7+k.
 *
 * 32 is deliberately above the largest legal finalized prefix (7+16=23).
 * The old base 16 overlapped finalized win slots for legal K=12..16.
 */
export const ADV_STATS_BASE = 32;

/**
 * ANTI-COLLAPSE PRESSURE — the fused twin of `directionOrderLoss`
 * (src/core/losses/isotropy.ts) and of main.ts's `GamePressure`.
 *
 * WHY IT EXISTS (measured, agent_notes/2026-08-17_120215_KST_collapse_fix.md):
 * the soft-angle payoff embeds a 2-vector as ψτ(z)=(z,τ)/√(‖z‖²+τ²), so the
 * ZERO vector is the sphere's north pole and sits chord √2 ≈ 1.414 from EVERY
 * equatorial prediction, while a genuinely direction-varied target only earns
 * the K-way quantization error (≈ 0.44 at K=4/ε=0.05). The generator therefore
 * has a 3.15× incentive to drive the encoded target to zero, which on the pair
 * observer IS a spatially constant — laminar — field. This term prices that.
 *
 * L = polar·‖M₁‖² + nematic·‖M₂‖²  over the batch's soft-unit directions
 *     u = F/√(‖F‖²+τ²),  M₁ = mean(u),  M₂ = mean(uₓ²−u_y², 2uₓu_y)
 *
 * `none` emits ZERO WGSL text, so every pressure-free shader — i.e. every
 * shader this codebase compiled before the term existed — stays byte-identical.
 * It is a compile-time variant, never a runtime branch in the hot loop.
 */
export type FusedGamePressure =
  /** Shipped pre-2026-08-17 behaviour: the game alone steers the field. */
  | { readonly tag: "none" }
  /** `polar` alone is escapable by ±F₀ counter-streaming sheets (measured
   *  R₂ = 0.95), so `nematic` is part of the SAME variant, not a second mode. */
  | {
      readonly tag: "anti-collapse";
      readonly polar: number;
      readonly nematic: number;
      /** Direction softener; normally the objective's own soft-angle τ. It is
       *  also the ε that floors the √(‖F‖²+τ²) radicand — no separate guard. */
      readonly tau: number;
    };

function checkedPressure(p: FusedGamePressure | undefined): FusedGamePressure {
  const value: FusedGamePressure = p ?? { tag: "none" };
  switch (value.tag) {
    case "none":
      return value;
    case "anti-collapse": {
      const finiteNonNeg = (x: number) => Number.isFinite(x) && x >= 0;
      if (!finiteNonNeg(value.polar) || !finiteNonNeg(value.nematic)) {
        throw new Error(
          `adversary: pressure weights must be finite and >= 0, got ` +
            `polar=${value.polar} nematic=${value.nematic}`
        );
      }
      if (!(Number.isFinite(value.tau) && value.tau > 0)) {
        throw new Error(
          `adversary: pressure tau must be finite and > 0, got ${value.tau}`
        );
      }
      return value;
    }
    default: {
      const unhandled: never = value;
      throw new Error(
        `adversary: unhandled pressure ${JSON.stringify(unhandled)}`
      );
    }
  }
}

/**
 * Stats buffer geometry. δ over the pressure variant — the four direction
 * moments are appended at [7+k, 11+k) ONLY under anti-collapse pressure, so a
 * pressure-free adversary keeps the historical 7+k ABI byte-for-byte.
 *
 * The moments are stored (not just R₁) because pass A's BACKWARD needs all
 * four as batch constants: dL/du depends on M₁ₓ, M₁ᵧ, M₂c and M₂s separately.
 * The host derives R₁ = ‖M₁‖ and R₂ = ‖M₂‖ from them.
 */
/**
 * PER-FAMILY PAYOFF INSTRUMENT.
 *
 * `off` is a real state, not a zero: on a classless field there is no family to
 * attribute a payoff to, and on an m > 1 observer a tuple can MIX families, so
 * there is no honest attribution either — inventing a bucketing rule there
 * would put a number on the HUD that nothing computes. See
 * {@link familyInstrument}, which is the only place this is decided.
 */
export type FamilyInstrument =
  | { readonly tag: "off" }
  | { readonly tag: "per-family"; readonly count: number };

/** κ: the instrument exists exactly when a tuple has ONE unambiguous family. */
export function familyInstrument(
  family: FamilyRoute,
  m: number
): FamilyInstrument {
  switch (family.tag) {
    case "none":
    case "onehot":
      return { tag: "off" };
    case "grid-plane":
      return m === 1 ? { tag: "per-family", count: family.count } : { tag: "off" };
  }
}

/** Reduction slots the instrument adds: Σpayoff and count, per family. */
function familySlots(instrument: FamilyInstrument): number {
  return instrument.tag === "off" ? 0 : 2 * instrument.count;
}

export function advStatsLayout(
  k: number,
  pressure: FusedGamePressure,
  family: FamilyInstrument = { tag: "off" }
): {
  finalized: number;
  pstride: number;
  momentOff: number;
  familyOff: number;
} {
  const base = 7 + k;
  const momentOff = base;
  const afterPressure = (() => {
    switch (pressure.tag) {
      case "none":
        return base;
      case "anti-collapse":
        return base + 4;
      default: {
        const unhandled: never = pressure;
        throw new Error(
          `adversary: unhandled pressure ${JSON.stringify(unhandled)}`
        );
      }
    }
  })();
  const familyOff = afterPressure;
  const finalized = afterPressure + familySlots(family);
  return { finalized, pstride: finalized, momentOff, familyOff };
}

export type TupleTag =
  | "point"
  | "pair"
  | "pair-rotation"
  | "pair-rotation-scale-raw"
  | "pair-rotation-scale-adjusted"
  | "tri"
  | "quad-labelled";

/**
 * Position-difference policy for relational tuple observers. The host must
 * derive this from the live boundary mode: wrap -> periodic, bounce/reset ->
 * euclidean. It is required shader configuration, never an implicit default.
 */
export type ObserverGeometry = "periodic" | "euclidean";

/** Which field output defines this predictor game. `blend` preserves the
 * historical field; lanes 0/1 are the independent Agree+Disagree generators. */
export type FieldLane = "blend" | 0 | 1;

/** Physical quantity observed by the predictor. These tags intentionally
 * mirror core/gan/adversary.ts without importing tfjs into this pure codegen
 * module. */
export type FusedAdversaryTarget =
  | { readonly tag: "force" }
  | { readonly tag: "post-velocity" };

/** Objective specialization. Every numeric field is compiled into WGSL: a
 * live change rebuilds the tiny adversary pipelines, while the five-dispatch
 * runtime path stays unchanged. */
export type FusedAdversaryLoss =
  | { readonly tag: "raw-vector" }
  | { readonly tag: "soft-angle"; readonly tau: number }
  | {
      readonly tag: "angle-relative-scale";
      readonly tau: number;
      readonly scaleWeight: number;
      readonly energyWeight: number;
      readonly energyTarget: number;
    }
  | {
      readonly tag: "angle-scale-hold";
      readonly tau: number;
      readonly scaleWeight: number;
      readonly energyWeight: number;
      readonly energyTarget: number;
    };

/** Private compatibility mode. It is reachable only when an old caller omits
 * `loss` while selecting the historical adjusted tuple tag. Explicit new
 * objective modes always receive RAW canonical signals and never execute the
 * old q/|q| tangent-floor path. */
type ResolvedAdversaryLoss =
  | FusedAdversaryLoss
  | { readonly tag: "legacy-adjusted" };

export interface FusedObjectiveDims {
  readonly m: number;
  /** Predictor context width. Post-velocity point adds incoming v/maxVel. */
  readonly du: number;
  /** Canonical 2-vector target width. */
  readonly vectorDy: number;
  /** Local relative-log-scale descriptor width. */
  readonly scaleDy: number;
  /** Predictor output / stored target width. */
  readonly outDy: number;
}

/** WGSL f32 literal — integers get a decimal point so `select(0.0, 1.0, …)`
 *  stays f32 (ε=0 made the relaxed weights emit as abstract-int and the
 *  shader failed to compile — caught by tools/train_wta_test.ts §2). */
const flit = (v: number): string => {
  if (!Number.isFinite(v)) throw new Error(`adversary: non-finite literal ${v}`);
  return Number.isInteger(v) ? v.toFixed(1) : String(v);
};

/** δ: tuple tag → (m, du, dy) — mirrors adversary.ts encodingDims. */
export function tupleDims(tag: TupleTag): { m: number; du: number; dy: number } {
  switch (tag) {
    case "point": return { m: 1, du: 2, dy: 2 };
    case "pair":
    case "pair-rotation":
    case "pair-rotation-scale-raw":
    case "pair-rotation-scale-adjusted":
      return { m: 2, du: 1, dy: 2 };
    case "tri": return { m: 3, du: 3, dy: 6 };
    case "quad-labelled": return { m: 4, du: 6, dy: 8 };
  }
}

export function adjustedTuple(tag: TupleTag): boolean {
  return tag === "pair-rotation-scale-adjusted";
}

const DEFAULT_TARGET: FusedAdversaryTarget = { tag: "force" };
const DEFAULT_LOSS: FusedAdversaryLoss = { tag: "raw-vector" };

function checkedTarget(
  target: FusedAdversaryTarget | undefined
): FusedAdversaryTarget {
  const value = target ?? DEFAULT_TARGET;
  if (value.tag !== "force" && value.tag !== "post-velocity") {
    throw new Error(`adversary: unsupported target ${String((value as { tag?: unknown }).tag)}`);
  }
  return value;
}

function checkedLoss(
  tag: TupleTag,
  loss: FusedAdversaryLoss | undefined
): ResolvedAdversaryLoss {
  // This narrow seam keeps existing constructors/tests bit-compatible. It is
  // not used by the new objective system: every new UI/config path supplies an
  // explicit loss.
  if (loss === undefined && adjustedTuple(tag)) return { tag: "legacy-adjusted" };
  const value = loss ?? DEFAULT_LOSS;
  switch (value.tag) {
    case "raw-vector":
      return value;
    case "soft-angle":
      if (!(Number.isFinite(value.tau) && value.tau > 0)) {
        throw new Error(`adversary: soft-angle tau must be finite and > 0, got ${value.tau}`);
      }
      return value;
    case "angle-relative-scale":
    case "angle-scale-hold": {
      const scaleWeight = value.scaleWeight;
      if (!(Number.isFinite(value.tau) && value.tau > 0)) {
        throw new Error(`adversary: ${value.tag} tau must be finite and > 0, got ${value.tau}`);
      }
      if (!(Number.isFinite(scaleWeight) && scaleWeight >= 0)) {
        throw new Error(
          `adversary: ${value.tag} scaleWeight must be finite and >= 0, got ${scaleWeight}`
        );
      }
      if (!(Number.isFinite(value.energyWeight) && value.energyWeight >= 0)) {
        throw new Error(
          `adversary: ${value.tag} energyWeight must be finite and >= 0, got ${value.energyWeight}`
        );
      }
      if (!(Number.isFinite(value.energyTarget) && value.energyTarget > 0)) {
        throw new Error(
          `adversary: ${value.tag} energyTarget must be finite and > 0, got ${value.energyTarget}`
        );
      }
      return value;
    }
  }
}

/** Joint observer/target/loss shape. Relative scale is local to a tuple, so it
 * is deliberately unavailable to point samples. Post-update velocity is also
 * point-only until a rotation-equivariant clip/context contract exists. */
export function fusedObjectiveDims(
  tag: TupleTag,
  target: FusedAdversaryTarget = DEFAULT_TARGET,
  loss: FusedAdversaryLoss = DEFAULT_LOSS
): FusedObjectiveDims {
  const base = tupleDims(tag);
  if (target.tag === "post-velocity" && tag !== "point") {
    throw new Error(
      `adversary: post-velocity target is point-only; relational ${tag} hides ` +
        `incoming velocity and component clipping is not rotation-equivariant`
    );
  }
  const scaleMode =
    loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold";
  if (scaleMode && tag === "point") {
    throw new Error(
      `adversary: ${loss.tag} needs a relational tuple; point has no local scale contrast`
    );
  }
  if (scaleMode && tag === "pair-rotation-scale-adjusted") {
    // The tag is only a context alias in explicit objective mode. Its target
    // remains raw below; accepting it here is intentional.
  }
  const scaleDy = scaleMode
    ? tag === "tri"
      ? 3
      : tag === "quad-labelled"
        ? 4
        : 1
    : 0;
  const du = target.tag === "post-velocity" ? 4 : base.du;
  return {
    m: base.m,
    du,
    vectorDy: base.dy,
    scaleDy,
    outDy: base.dy + scaleDy,
  };
}

// ---------------------------------------------------------------------------
// Scratch layout (floats, per tuple block)
// ---------------------------------------------------------------------------

export interface AdvScratchLayout {
  tag: TupleTag;
  m: number;
  k: number;
  du: number;
  /** Total predictor output width, retained as `dy` for packed-layout ABI
   * compatibility with AdversaryLayout. */
  dy: number;
  vectorDy: number;
  scaleDy: number;
  /** member particle indices as f32 (exact below 2^24 ≫ MAX_PARTICLES) */
  idxOff: number;      // m floats
  surOff: number;      // 1: per-tuple shared relaxed-WTA payoff
  winOff: number;      // 1: winner head index as f32
  uOff: number;        // du: encoded context (adversary layer-0 a_in)
  yOff: number;        // dy: encoded target
  siteInOff: number;   // m·2: normalized member positions pn
  /** field-encoding store (fourier/hashgrid: m·encDim; raw: 0 — offsets collapse) */
  encOff: number;
  /** per-member dL/dEnc store (hashgrid: m·encDim; raw/fourier: 0 — collapses).
   *  Written by the field heads' backward, read by fieldGrad's grid block. */
  dEncOff: number;
  /** per member, field head blocks [head0 | head1] (train_wgsl offsets) */
  fieldSiteOff: number;
  /** per adversary head: [activations][deltas] */
  advOff: number;
  stride: number;
  fieldSiteBlk: number;
  /** the field's per-(site,head) sub-layout — aOff/dOff/pOff/headBlk reused */
  fieldSl: TrainScratchLayout;
  advBlk: number;
  /**
   * Per-member family label as f32 (m floats on a family-planed grid, 0
   * otherwise — so every classless layout's stride is unchanged and its
   * generated shaders stay byte-identical).
   *
   * Stored rather than re-hashed in pass B on purpose: pass B has no `pcg`,
   * and a SECOND copy of the derivation is exactly the kind of drift that
   * would advect a particle with one family's field and score it against
   * another's. Appended last so no pre-existing offset moves.
   */
  clsOff: number;
  /** shared across heads (identical dims): per-layer act / delta offsets */
  advAOff: number[];
  advDOff: number[];
}

export function advScratchLayout(
  field: FieldLayout,
  advL: AdversaryLayout,
  tag: TupleTag,
  target: FusedAdversaryTarget = DEFAULT_TARGET,
  loss: FusedAdversaryLoss | undefined = undefined
): AdvScratchLayout {
  const resolvedLoss = checkedLoss(tag, loss);
  const dims =
    resolvedLoss.tag === "legacy-adjusted"
      ? { ...tupleDims(tag), vectorDy: 2, scaleDy: 0, outDy: 2 }
      : fusedObjectiveDims(tag, checkedTarget(target), resolvedLoss);
  const { m, du, vectorDy, scaleDy, outDy: dy } = dims;
  if (advL.du !== du || advL.dy !== dy) {
    throw new Error(
      `adversary: layout io (${advL.du}→${advL.dy}) does not match encoding ` +
        `'${tag}' objective (${du}→${dy})`
    );
  }
  const fieldSl = trainScratchLayout(field, 1);
  const outs = advL.heads[0].layers.map((L) => L.outSize);
  const total = outs.reduce((a, b) => a + b, 0);
  const advAOff: number[] = [];
  const advDOff: number[] = [];
  {
    let o = 0;
    for (const s of outs) { advAOff.push(o); o += s; }
    let d = total;
    for (const s of outs) { advDOff.push(d); d += s; }
  }
  const advBlk = 2 * total;
  const idxOff = 0;
  const surOff = idxOff + m;
  const winOff = surOff + 1;
  const uOff = winOff + 1;
  const yOff = uOff + du;
  const siteInOff = yOff + dy;
  const encOff = siteInOff + 2 * m;
  // Same discipline as trainScratchLayout: encStore/dEncStore are 0 for the
  // encodings that do not use them, so raw and fourier strides — and every
  // generated shader built on them — stay byte-identical to the pre-hashgrid
  // adversary (kernel_test's f32 codegen guard depends on that).
  const dEncOff = encOff + m * fieldSl.encStore;
  const fieldSiteOff = dEncOff + m * fieldSl.dEncStore;
  const advOff = fieldSiteOff + m * fieldSl.siteBlk;
  const clsOff = advOff + advL.k * advBlk;
  const stride = clsOff + (encodingPlanes(field.encoding) > 1 ? m : 0);
  return {
    tag, m, k: advL.k, du, dy, vectorDy, scaleDy,
    idxOff, surOff, winOff, uOff, yOff, siteInOff, encOff, dEncOff, fieldSiteOff,
    advOff, clsOff, stride, fieldSiteBlk: fieldSl.siteBlk, fieldSl, advBlk,
    advAOff, advDOff,
  };
}

export function advScratchBytes(
  field: FieldLayout,
  advL: AdversaryLayout,
  tag: TupleTag,
  batchCap: number,
  target: FusedAdversaryTarget = DEFAULT_TARGET,
  loss: FusedAdversaryLoss | undefined = undefined
): number {
  return advScratchLayout(field, advL, tag, target, loss).stride * batchCap * 4;
}

// ---------------------------------------------------------------------------
// κ — the one support-validation site for the fused adversary
// ---------------------------------------------------------------------------

export function validateAdversaryFusion(field: FieldLayout, advL: AdversaryLayout): void {
  if (field.spec.kind !== "helmholtz" && field.spec.kind !== "agree-disagree") {
    throw new Error(
      "adversary: fused adversary needs a two-head neural field " +
        `(got ${field.spec.kind})`
    );
  }
  switch (field.family.tag) {
    case "none":
    case "grid-plane":
      break;
    case "onehot":
      // One-hot channels widen head 1's layer-0 input, and the adversary's
      // field backward has no counterpart for those rows. The family-planed
      // hashgrid needs none: the label only moves the grid's cell index, so it
      // rides the dEnc machinery the reward already uses.
      throw new Error(
        "adversary: one-hot class channels are not supported by the fused " +
          "adversary — use a family-planed hashgrid field"
      );
  }
  for (const h of advL.heads) {
    for (const L of h.layers) {
      if (L.activation === "sin") {
        throw new Error(
          "adversary: sin activations need a pre-act checkpoint the adversary " +
            "scratch does not carry — use selu/tanh/sigmoid/linear"
        );
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Adversary-head codegen (n-dim output — the field emitters hardcode vec2f)
// ---------------------------------------------------------------------------

/** φ' from the POST-activation (sin rejected at validate time). */
function advActDeriv(act: string, postAct: string): string {
  switch (act) {
    case "selu": return `seluD(${postAct})`;
    case "tanh": return `tanhD(${postAct})`;
    case "sigmoid": return `sigmoidD(${postAct})`;
    default: return `1.0`;
  }
}

function emitAdvFwd(j: number, head: HeadSpec, sl: AdvScratchLayout, maxW: number): string {
  const lines: string[] = [];
  lines.push(`fn adv_fwd_${j}(uBase : u32, base : u32) {`);
  lines.push(`  var cur : array<f32, ${maxW}>;`);
  lines.push(`  var nxt : array<f32, ${maxW}>;`);
  lines.push(`  for (var i = 0u; i < ${sl.du}u; i = i + 1u) { cur[i] = scratch[uBase + i]; }`);
  head.layers.forEach((L, l) => {
    lines.push(`  // layer ${l}: ${L.inSize} -> ${L.outSize} (${L.activation})`);
    lines.push(`  for (var o = 0u; o < ${L.outSize}u; o = o + 1u) {`);
    lines.push(`    var s = advW[${L.biasOffset}u + o];`);
    lines.push(`    for (var i = 0u; i < ${L.inSize}u; i = i + 1u) {`);
    lines.push(`      s = s + cur[i] * advW[${L.weightOffset}u + i * ${L.outSize}u + o];`);
    lines.push(`    }`);
    const a =
      L.activation === "selu" ? "selu(s)"
      : L.activation === "tanh" ? "tanh(s)"
      : L.activation === "sigmoid" ? "sigmoid_(s)"
      : "s";
    lines.push(`    let a = ${a};`);
    lines.push(`    scratch[base + ${sl.advAOff[l]}u + o] = a;`);
    lines.push(`    nxt[o] = a;`);
    lines.push(`  }`);
    lines.push(`  for (var o = 0u; o < ${L.outSize}u; o = o + 1u) { cur[o] = nxt[o]; }`);
  });
  lines.push(`}`);
  return lines.join("\n");
}

/**
 * Backward for adversary head j. The CALLER pre-stores dL/d(pred_o) into the
 * last layer's δ slot; this walks it down, multiplying activation derivatives
 * (post-act trick) and writing every layer's δ. No input gradient: u is a
 * function of member positions only, which are constants here.
 */
function emitAdvBwd(j: number, head: HeadSpec, sl: AdvScratchLayout, maxW: number): string {
  const nL = head.layers.length;
  const lines: string[] = [];
  lines.push(`fn adv_bwd_${j}(base : u32) {`);
  lines.push(`  var dcur : array<f32, ${maxW}>;`);
  lines.push(`  var dprev : array<f32, ${maxW}>;`);
  {
    const L = head.layers[nL - 1];
    lines.push(`  for (var o = 0u; o < ${L.outSize}u; o = o + 1u) {`);
    lines.push(`    let a = scratch[base + ${sl.advAOff[nL - 1]}u + o];`);
    lines.push(`    let d = scratch[base + ${sl.advDOff[nL - 1]}u + o] * ${advActDeriv(L.activation, "a")};`);
    lines.push(`    dcur[o] = d;`);
    lines.push(`    scratch[base + ${sl.advDOff[nL - 1]}u + o] = d;`);
    lines.push(`  }`);
  }
  for (let l = nL - 2; l >= 0; l--) {
    const Lnext = head.layers[l + 1];
    const L = head.layers[l];
    lines.push(`  for (var i = 0u; i < ${L.outSize}u; i = i + 1u) {`);
    lines.push(`    var s = 0.0;`);
    lines.push(`    for (var o = 0u; o < ${Lnext.outSize}u; o = o + 1u) {`);
    lines.push(`      s = s + advW[${Lnext.weightOffset}u + i * ${Lnext.outSize}u + o] * dcur[o];`);
    lines.push(`    }`);
    lines.push(`    let a = scratch[base + ${sl.advAOff[l]}u + i];`);
    lines.push(`    dprev[i] = s * ${advActDeriv(L.activation, "a")};`);
    lines.push(`    scratch[base + ${sl.advDOff[l]}u + i] = dprev[i];`);
    lines.push(`  }`);
    lines.push(`  for (var i = 0u; i < ${L.outSize}u; i = i + 1u) { dcur[i] = dprev[i]; }`);
  }
  lines.push(`}`);
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Tuple-encoding codegen — one handler per tag, forward + generator backward
// as ONE straight-line sequence (bwd reuses the fwd's frame locals).
// ---------------------------------------------------------------------------

/**
 * Forward contract: fills `uvec[du]` / `yvec[dy]` from `pn[m]` (normalized
 * positions) and `sig[m]` (raw field outputs), leaving frame locals in
 * scope for the backward. Backward contract: consumes `dLdy[dy]`, fills
 * `dSig[m]` (dL/d raw field output). Positions are constants — no dL/dpn.
 */
function encodeFwdBwd(
  tag: TupleTag,
  observerGeometry: ObserverGeometry,
  target: FusedAdversaryTarget,
  loss: ResolvedAdversaryLoss,
  vectorDy: number,
  scaleDy: number
): { fwd: string; bwd: string } {
  const E2 = "1e-12"; // FRAME_EPS² — same softening as adversary.ts
  const scaleMode =
    loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold";
  const energyWeight = scaleMode ? loss.energyWeight : 0;
  const energyTarget = scaleMode ? loss.energyTarget : 1;
  const scaleRatio2 = 1e-6; // REL_SCALE_SOFT_RATIO² (1e-3²)
  const energyFloor2 = 1e-16;
  if (tag === "point") {
    if (scaleMode) {
      throw new Error(`adversary: ${loss.tag} has no local point scale contrast`);
    }
    return {
      fwd: `
    uvec[0] = pn[0].x; uvec[1] = pn[0].y;
    ${target.tag === "post-velocity" ? "uvec[2] = vn[0].x; uvec[3] = vn[0].y;" : ""}
    yvec[0] = sig[0].x; yvec[1] = sig[0].y;
    let tupleEnergy = dot(sig[0], sig[0]);
    let targetActive = true;`,
      bwd: `
    dSig[0] = vec2f(dLdy[0], dLdy[1]);`,
    };
  }
  const isPair =
    tag === "pair" ||
    tag === "pair-rotation" ||
    tag === "pair-rotation-scale-raw" ||
    tag === "pair-rotation-scale-adjusted";
  if (isPair && loss.tag !== "legacy-adjusted") {
    const keepSeparation = tag === "pair" || tag === "pair-rotation";
    const scaleFwd = scaleMode ? `
    // Swap-invariant local pair magnitude contrast. Using the two raw member
    // magnitudes avoids the identically-equal |(+q,-q)| centered-pair trap.
    let prM0 = dot(sig[0], sig[0]);
    let prM1 = dot(sig[1], sig[1]);
    let tupleEnergy = 0.5 * (prM0 + prM1);
    let prScaleActive = tupleEnergy > ${flit(energyFloor2)};
    let prSafeE = select(1.0, tupleEnergy, prScaleActive);
    let prScaleDen0 = prM0 + ${flit(scaleRatio2)} * prSafeE;
    let prScaleDen1 = prM1 + ${flit(scaleRatio2)} * prSafeE;
    let prScaleDiff = 0.5 * (log(prScaleDen0) - log(prScaleDen1));
    yvec[${vectorDy}] = select(
      0.0,
      sqrt(prScaleDiff * prScaleDiff + ${ADV_SOFT_EPS2}) - 1e-6,
      prScaleActive
    );
    targetActive = targetActive && prScaleActive;` : `
    let tupleEnergy = 0.5 * (dot(sig[0], sig[0]) + dot(sig[1], sig[1]));`;
    const scaleBwd = scaleMode ? `
    if (targetActive) {
      let prDz = dLdy[${vectorDy}];
      let prDContrast = prDz * prScaleDiff /
        sqrt(prScaleDiff * prScaleDiff + ${ADV_SOFT_EPS2});
      let prDA0 = prDContrast;
      let prDA1 = -prDContrast;
      let prCoupled = ${flit(scaleRatio2 / 2)} *
        (prDA0 / prScaleDen0 + prDA1 / prScaleDen1);
      dSig[0] = dSig[0] +
        sig[0] * (prDA0 / prScaleDen0 + prCoupled);
      dSig[1] = dSig[1] +
        sig[1] * (prDA1 / prScaleDen1 + prCoupled);
    }
    if (targetActive) {
      dSig[0] = dSig[0] + 0.5 * energyAnchorSeed * sig[0];
      dSig[1] = dSig[1] + 0.5 * energyAnchorSeed * sig[1];
    }` : "";
    return {
      fwd: `
    let prD = observerDelta2(pn[1] - pn[0]);
    let prR = sqrt(dot(prD, prD) + ${E2});
    let prE1 = prD / prR;
    let prE2 = vec2f(-prE1.y, prE1.x);
    let prDel = sig[1] - sig[0];
    uvec[0] = ${keepSeparation ? "prR" : "1.0"};
    yvec[0] = dot(prDel, prE1);
    yvec[1] = dot(prDel, prE2);
    var targetActive = true;
${scaleFwd}`,
      bwd: `
    let prDDel = dLdy[0] * prE1 + dLdy[1] * prE2;
    dSig[1] = prDDel;
    dSig[0] = -prDDel;
${scaleBwd}`,
    };
  }
  if (tag === "pair-rotation-scale-adjusted" && loss.tag === "legacy-adjusted") {
    return {
      fwd: `
    let prD = observerDelta2(pn[1] - pn[0]);
    let prR = sqrt(dot(prD, prD) + ${E2});
    let prE1 = prD / prR;
    let prE2 = vec2f(-prE1.y, prE1.x);
    let prDel = sig[1] - sig[0];
    let prQ = vec2f(dot(prDel, prE1), dot(prDel, prE2));
    let prQ2 = dot(prQ, prQ);
    let targetActive = prQ2 > ${flit(ADV_DIRECTION_ACTIVE_FLOOR ** 2)};
    let prQNorm = sqrt(max(prQ2, ${flit(ADV_DIRECTION_ACTIVE_FLOOR ** 2)}));
    let prY = select(vec2f(0.0), prQ / prQNorm, targetActive);
    uvec[0] = 1.0;
    yvec[0] = prY.x;
    yvec[1] = prY.y;
    let tupleEnergy = prQ2;`,
      bwd: `
    var prDQ = vec2f(0.0);
    if (targetActive) {
      let prDY = vec2f(dLdy[0], dLdy[1]);
      // Exact transpose Jacobian of q/||q||. Its dot with q is zero
      // (up to f32 roundoff), closing the uniform-scale reward direction.
      prDQ = (prDY - prY * dot(prY, prDY)) / prQNorm;
    }
    let prDDel = prDQ.x * prE1 + prDQ.y * prE2;
    dSig[1] = prDDel;
      dSig[0] = -prDDel;`,
    };
  }
  if (tag === "quad-labelled") {
    // Labels are semantic: member 0 is the anchor, member 1 defines the frame,
    // and members 2/3 retain their supplied order. This is deliberately not a
    // set canonicalization.
    const scaleFwd = scaleMode ? `
    var qlScaleDen : array<f32, 4>;
    var qlScaleA : array<f32, 4>;
    var tupleEnergy = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
      tupleEnergy = tupleEnergy + dot(sig[i], sig[i]) / 4.0;
    }
    let qlScaleActive = tupleEnergy > ${flit(energyFloor2)};
    let qlSafeE = select(1.0, tupleEnergy, qlScaleActive);
    var qlMeanA = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
      qlScaleDen[i] = dot(sig[i], sig[i]) +
        ${flit(scaleRatio2)} * qlSafeE;
      qlScaleA[i] = 0.5 * log(qlScaleDen[i]);
      qlMeanA = qlMeanA + qlScaleA[i] / 4.0;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
      yvec[${vectorDy}u + i] =
        select(0.0, qlScaleA[i] - qlMeanA, qlScaleActive);
    }
    targetActive = targetActive && qlScaleActive;` : `
    var tupleEnergy = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
      tupleEnergy = tupleEnergy + dot(sig[i], sig[i]) / 4.0;
    }`;
    const scaleBwd = scaleMode ? `
    if (targetActive) {
      var qlDA : array<f32, 4>;
      var qlMeanDZ = 0.0;
      for (var i = 0u; i < 4u; i = i + 1u) {
        qlMeanDZ = qlMeanDZ + dLdy[${vectorDy}u + i] / 4.0;
      }
      var qlCoupled = 0.0;
      for (var i = 0u; i < 4u; i = i + 1u) {
        qlDA[i] = dLdy[${vectorDy}u + i] - qlMeanDZ;
        qlCoupled = qlCoupled + qlDA[i] / qlScaleDen[i];
      }
      for (var i = 0u; i < 4u; i = i + 1u) {
        let qlScaleGrad = sig[i] *
          (qlDA[i] / qlScaleDen[i] +
           ${flit(scaleRatio2 / 4)} * qlCoupled);
        dSig[i] = dSig[i] + qlScaleGrad;
      }
    }
    if (targetActive) {
      for (var i = 0u; i < 4u; i = i + 1u) {
        dSig[i] = dSig[i] + 0.25 * energyAnchorSeed * sig[i];
      }
    }` : "";
    return {
      fwd: `
    var qlD : array<vec2f, 3>;
    qlD[0] = observerDelta2(pn[1] - pn[0]);
    qlD[1] = observerDelta2(pn[2] - pn[0]);
    qlD[2] = observerDelta2(pn[3] - pn[0]);
    let qlAnchor2 = dot(qlD[0], qlD[0]);
    let qlSafeR = sqrt(qlAnchor2 + ${E2});
    let qlE1 = qlD[0] / qlSafeR;
    let qlE2 = vec2f(-qlE1.y, qlE1.x);
    uvec[0] = dot(qlD[0], qlE1); uvec[1] = dot(qlD[0], qlE2);
    uvec[2] = dot(qlD[1], qlE1); uvec[3] = dot(qlD[1], qlE2);
    uvec[4] = dot(qlD[2], qlE1); uvec[5] = dot(qlD[2], qlE2);

    let qlMean = (sig[0] + sig[1] + sig[2] + sig[3]) / 4.0;
    var qlRel : array<vec2f, 4>;
    qlRel[0] = sig[0] - qlMean; qlRel[1] = sig[1] - qlMean;
    qlRel[2] = sig[2] - qlMean; qlRel[3] = sig[3] - qlMean;

    // encodeQuadLabelled zeros y from the TRUE anchor magnitude. residualFor
    // independently derives activity from the encoded first relative vector;
    // preserve that subtle two-stage tfjs contract exactly.
    let qlEncodeActive =
      qlAnchor2 > ${flit(ADV_QUAD_ANCHOR_ACTIVE_FLOOR ** 2)};
    var targetActive =
      uvec[0] * uvec[0] + uvec[1] * uvec[1] >
        ${flit(ADV_QUAD_ANCHOR_ACTIVE_FLOOR ** 2)};
    yvec[0] = select(0.0, dot(qlRel[0], qlE1), qlEncodeActive);
    yvec[1] = select(0.0, dot(qlRel[0], qlE2), qlEncodeActive);
    yvec[2] = select(0.0, dot(qlRel[1], qlE1), qlEncodeActive);
    yvec[3] = select(0.0, dot(qlRel[1], qlE2), qlEncodeActive);
    yvec[4] = select(0.0, dot(qlRel[2], qlE1), qlEncodeActive);
    yvec[5] = select(0.0, dot(qlRel[2], qlE2), qlEncodeActive);
    yvec[6] = select(0.0, dot(qlRel[3], qlE1), qlEncodeActive);
    yvec[7] = select(0.0, dot(qlRel[3], qlE2), qlEncodeActive);
${scaleFwd}`,
      bwd: `
    var qlDRel : array<vec2f, 4>;
    qlDRel[0] = dLdy[0] * qlE1 + dLdy[1] * qlE2;
    qlDRel[1] = dLdy[2] * qlE1 + dLdy[3] * qlE2;
    qlDRel[2] = dLdy[4] * qlE1 + dLdy[5] * qlE2;
    qlDRel[3] = dLdy[6] * qlE1 + dLdy[7] * qlE2;
    let qlDSum = (qlDRel[0] + qlDRel[1] + qlDRel[2] + qlDRel[3]) / 4.0;
    dSig[0] = qlDRel[0] - qlDSum;
    dSig[1] = qlDRel[1] - qlDSum;
    dSig[2] = qlDRel[2] - qlDSum;
    dSig[3] = qlDRel[3] - qlDSum;
${scaleBwd}`,
    };
  }
  // tri — mirrors encodeTri step by step (see adversary.ts docstring):
  // opposite side lengths (own min-images), descending rank with stable
  // lower-label tie break, chart at the canonical vertex A, centroid frame,
  // centered raw field signals projected on (ê1, ê2).
  if (tag === "tri") {
    const scaleFwd = scaleMode ? `
    var trScaleDen : array<f32, 3>;
    var trScaleA : array<f32, 3>;
    var tupleEnergy = 0.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
      tupleEnergy = tupleEnergy + dot(sig[i], sig[i]) / 3.0;
    }
    let trScaleActive = tupleEnergy > ${flit(energyFloor2)};
    let trSafeE = select(1.0, tupleEnergy, trScaleActive);
    var trMeanA = 0.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
      trScaleDen[i] = dot(sig[i], sig[i]) +
        ${flit(scaleRatio2)} * trSafeE;
      trScaleA[i] = 0.5 * log(trScaleDen[i]);
      trMeanA = trMeanA + trScaleA[i] / 3.0;
    }
    yvec[${vectorDy}] =
      select(0.0, trScaleA[trA] - trMeanA, trScaleActive);
    yvec[${vectorDy + 1}] =
      select(0.0, trScaleA[trB] - trMeanA, trScaleActive);
    yvec[${vectorDy + 2}] =
      select(0.0, trScaleA[trC] - trMeanA, trScaleActive);
    targetActive = targetActive && trScaleActive;` : `
    var tupleEnergy = 0.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
      tupleEnergy = tupleEnergy + dot(sig[i], sig[i]) / 3.0;
    }`;
    const scaleBwd = scaleMode ? `
    if (targetActive) {
      var trDZ : array<f32, 3>;
      trDZ[trA] = dLdy[${vectorDy}];
      trDZ[trB] = dLdy[${vectorDy + 1}];
      trDZ[trC] = dLdy[${vectorDy + 2}];
      let trMeanDZ = (trDZ[0] + trDZ[1] + trDZ[2]) / 3.0;
      var trDA : array<f32, 3>;
      var trCoupled = 0.0;
      for (var i = 0u; i < 3u; i = i + 1u) {
        trDA[i] = trDZ[i] - trMeanDZ;
        trCoupled = trCoupled + trDA[i] / trScaleDen[i];
      }
      for (var i = 0u; i < 3u; i = i + 1u) {
        let trScaleGrad = sig[i] *
          (trDA[i] / trScaleDen[i] +
           ${flit(scaleRatio2 / 3)} * trCoupled);
        dSig[i] = dSig[i] + trScaleGrad;
      }
    }
    if (targetActive) {
      for (var i = 0u; i < 3u; i = i + 1u) {
        dSig[i] = dSig[i] + (energyAnchorSeed / 3.0) * sig[i];
      }
    }` : "";
    return {
    fwd: `
    var trS : array<f32, 3>;
    {
      let d12 = observerDelta2(pn[2] - pn[1]);
      let d02 = observerDelta2(pn[2] - pn[0]);
      let d01 = observerDelta2(pn[1] - pn[0]);
      trS[0] = sqrt(dot(d12, d12) + ${E2});
      trS[1] = sqrt(dot(d02, d02) + ${E2});
      trS[2] = sqrt(dot(d01, d01) + ${E2});
    }
    // rank_i = #{j: s_j > s_i} + #{j<i: s_j == s_i}  (distinct in {0,1,2})
    var trRank : array<u32, 3>;
    trRank[0] = b2u(trS[1] > trS[0]) + b2u(trS[2] > trS[0]);
    trRank[1] = b2u(trS[0] > trS[1]) + b2u(trS[2] > trS[1]) + b2u(trS[0] == trS[1]);
    trRank[2] = b2u(trS[0] > trS[2]) + b2u(trS[1] > trS[2]) + b2u(trS[0] == trS[2]) + b2u(trS[1] == trS[2]);
    var trPerm : array<u32, 3>;
    trPerm[trRank[0]] = 0u; trPerm[trRank[1]] = 1u; trPerm[trRank[2]] = 2u;
    let trA = trPerm[0]; let trB = trPerm[1]; let trC = trPerm[2];
    let trDAB = observerDelta2(pn[trB] - pn[trA]);
    let trDAC = observerDelta2(pn[trC] - pn[trA]);
    let trCen = (trDAB + trDAC) / 3.0;
    let trVA = -trCen;
    let trLen = sqrt(dot(trVA, trVA) + ${E2});
    let trE1 = trVA / trLen;
    let trE2 = vec2f(-trE1.y, trE1.x);
    let trDbar = (sig[0] + sig[1] + sig[2]) / 3.0;
    var trRel : array<vec2f, 3>;
    trRel[0] = sig[0] - trDbar; trRel[1] = sig[1] - trDbar; trRel[2] = sig[2] - trDbar;
    let trMinGap = min(
      abs(trS[0] - trS[1]),
      min(abs(trS[0] - trS[2]), abs(trS[1] - trS[2]))
    );
    var targetActive = trMinGap > ${ADV_TRI_TIE_EPS};
    uvec[0] = trS[trA]; uvec[1] = trS[trB]; uvec[2] = trS[trC];
    yvec[0] = dot(trRel[trA], trE1); yvec[1] = dot(trRel[trA], trE2);
    yvec[2] = dot(trRel[trB], trE1); yvec[3] = dot(trRel[trB], trE2);
    yvec[4] = dot(trRel[trC], trE1); yvec[5] = dot(trRel[trC], trE2);
${scaleFwd}`,
    bwd: `
    var trDRel : array<vec2f, 3>;
    trDRel[trA] = dLdy[0] * trE1 + dLdy[1] * trE2;
    trDRel[trB] = dLdy[2] * trE1 + dLdy[3] * trE2;
    trDRel[trC] = dLdy[4] * trE1 + dLdy[5] * trE2;
    let trDSum = (trDRel[0] + trDRel[1] + trDRel[2]) / 3.0;
    dSig[0] = trDRel[0] - trDSum;
    dSig[1] = trDRel[1] - trDSum;
    dSig[2] = trDRel[2] - trDSum;
${scaleBwd}`,
    };
  }
  const unhandled: never = tag;
  throw new Error(`adversary: unhandled tuple encoding ${String(unhandled)}`);
}

// ---------------------------------------------------------------------------
// PASS A — advFwd + finalize
// ---------------------------------------------------------------------------

export interface AdvShaderOpts {
  tag: TupleTag;
  relaxEps: number;
  /** Omitted fields retain the historical raw-force/raw-vector ABI. The old
   * adjusted tuple tag alone selects its private compatibility objective. */
  target?: FusedAdversaryTarget;
  loss?: FusedAdversaryLoss;
  /** Required relational geometry; must match the live boundary mode. */
  observerGeometry: ObserverGeometry;
  /** Defaults to blend for compatibility. Direct lanes never read alpha and
   * backpropagate into exactly one field head. */
  fieldLane?: FieldLane;
  /** Omitted resolves to `none`, which emits zero WGSL — see
   *  {@link FusedGamePressure}. */
  pressure?: FusedGamePressure;
}

function checkedFieldLane(lane: FieldLane | undefined): FieldLane {
  const value = lane ?? "blend";
  if (value !== "blend" && value !== 0 && value !== 1) {
    throw new Error(`adversary: invalid fieldLane ${String(value)}`);
  }
  return value;
}

function checkedObserverGeometry(
  geometry: ObserverGeometry
): ObserverGeometry {
  if (geometry !== "periodic" && geometry !== "euclidean") {
    throw new Error(
      `adversary: observerGeometry must be explicitly periodic or euclidean, got ` +
        `${String(geometry)}`
    );
  }
  return geometry;
}

/**
 * Pass A bindings (8 storage — exactly the WebGPU default per-stage limit):
 *   0 uniform UAdv
 *   1 ro  weights   — FIELD weights (name matches the reused field codegen)
 *   2 ro  advW      — adversary weights
 *   3 rw  scratch
 *   4 rw  stats     — [0..31] finalized/reserved, then workgroup partials
 *   5 ro  partPos   6 ro partVel
 *   7 rw  surprise  — two packed N-capacity planes: raw, per-unit-signal
 *   8 ro  tuples    — uploaded member states (m·4 floats/tuple), source==0
 */
export function adversaryPassAShader(
  field: FieldLayout,
  advL: AdversaryLayout,
  opts: AdvShaderOpts
): string {
  validateAdversaryFusion(field, advL);
  const { tag, relaxEps } = opts;
  const target = checkedTarget(opts.target);
  const loss = checkedLoss(tag, opts.loss);
  const fieldLane = checkedFieldLane(opts.fieldLane);
  const observerGeometry = checkedObserverGeometry(opts.observerGeometry);
  const pressure = checkedPressure(opts.pressure);
  const relaxUpper = advL.k <= 1 ? 1 : (advL.k - 1) / advL.k;
  if (!(relaxEps >= 0 && relaxEps < relaxUpper)) {
    throw new Error(
      `adversary: relaxEps ${relaxEps} outside [0, ${relaxUpper}) for k=${advL.k}`
    );
  }
  const sl = advScratchLayout(field, advL, tag, target, opts.loss);
  const { m, k, du, dy, vectorDy, scaleDy } = sl;
  // FAMILY. `planed` says the field's grid is indexed by (family, y, x), so
  // every member needs its label; `instrument` says a tuple has ONE family, so
  // the payoff can be attributed. They are independent: a pair observer on a
  // planed field still trains correctly, it just cannot be charted per family.
  const planed = encodingPlanes(field.encoding) > 1;
  const instrument = familyInstrument(field.family, m);
  const famCount = instrument.tag === "per-family" ? instrument.count : 0;
  if (k >= 2 && relaxEps === 0) {
    // permitted — hard WTA is a real (collapsing) variant; the app never ships it
  }
  const heads = field.spec.heads as HeadSpec[];
  const enc = field.encoding;
  const fsl = sl.fieldSl;
  const maxWField = Math.max(
    2, fsl.encDim,
    ...heads.flatMap((h) => h.layers.map((L) => Math.max(L.outSize, L.inSize)))
  );
  const maxWAdv = Math.max(
    du, dy,
    ...advL.heads[0].layers.map((L) => Math.max(L.outSize, L.inSize))
  );
  const WG = ADV_WG;
  const STRIDE = sl.stride;
  // Legacy finalized slots [0..5+k) stay stable. Energy RMS/active count are
  // appended at 5+k / 6+k; partials carry the same two extra reductions. Under
  // anti-collapse pressure four direction moments follow at [7+k, 11+k).
  const statsL = advStatsLayout(k, pressure, instrument);
  const PSTRIDE = statsL.pstride;
  if (statsL.finalized > ADV_STATS_BASE) {
    throw new Error(
      `adversary: finalized stats prefix ${statsL.finalized} would overlap the ` +
        `partials at ${ADV_STATS_BASE} (k=${k})`
    );
  }
  const MOM = statsL.momentOff;
  const loserW = k >= 2 ? relaxEps / (k - 1) : 0;
  const winW = k >= 2 ? 1 - relaxEps : 1;
  const explicitObjective = loss.tag !== "legacy-adjusted";
  const legacyAngular = loss.tag === "legacy-adjusted";
  const softAngular =
    explicitObjective && loss.tag !== "raw-vector";
  const scaleMode =
    loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold";
  const scaleWeight =
    scaleMode ? loss.scaleWeight : 0;
  const energyWeight = scaleMode ? loss.energyWeight : 0;
  const energyTarget = scaleMode ? loss.energyTarget : 1;
  const tau = softAngular ? loss.tau : 1;
  const encFB = encodeFwdBwd(
    tag,
    observerGeometry,
    target,
    loss,
    vectorDy,
    scaleDy
  );

  // field head call helpers on the adversary scratch (member t = site index)
  const fieldBase = (site: string, h: number) =>
    `sBase + ${sl.fieldSiteOff}u + (${site}) * ${sl.fieldSiteBlk}u + ${h === 0 ? 0 : fsl.headBlk[0]}u`;
  const encBase = (site: string) => `sBase + ${sl.encOff}u + (${site}) * ${fsl.encDim}u`;
  const dEncBase = (site: string) => `sBase + ${sl.dEncOff}u + (${site}) * ${fsl.encDim}u`;
  /** `, mcls[site]` on a family-planed grid; empty everywhere else. */
  const clsArg = (site: string) => (planed ? `, mcls[${site}]` : ``);
  const encodeAt = (uExpr: string, site: string) =>
    enc.kind === "raw"
      ? ``
      : `encodeSite(${uExpr}, ${encBase(site)}${clsArg(site)});\n      `;
  const fwdCall = (h: number, uExpr: string, site: string) =>
    enc.kind === "raw"
      ? `fwd_head_${h}(${uExpr}, ${fieldBase(site, h)}, 0u)`
      : `fwd_head_${h}(${encBase(site)}, ${fieldBase(site, h)})`;
  // One call shape per encoding kind — the emitBwdStore signature is
  // type-directed the same way (train_wgsl.ts). `pn[site]` is the normalized
  // member position, live at every backward site; hashgrid's backward
  // recomputes its corner geometry from it rather than storing four indices.
  const bwdCall = (h: number, dExpr: string, site: string) =>
    enc.kind === "raw"
      ? `bwd_head_${h}(${dExpr}, ${fieldBase(site, h)})`
      : enc.kind === "fourier"
      ? `bwd_head_${h}(${dExpr}, ${fieldBase(site, h)}, ${encBase(site)})`
      : `bwd_head_${h}(${dExpr}, ${fieldBase(site, h)}, pn[${site}], ` +
        `${dEncBase(site)}${clsArg(site)})`;
  // Which field head SEEDS the shared per-site dL/dEnc block (hashgrid only).
  // `fieldBackward` calls both heads on the blend lane and exactly one head on
  // a direct lane, so the seed is the lane, NOT head 0: a lane-1 game (Agree +
  // Disagree lane B) would otherwise `+=` into a block nobody wrote.
  const dEncSeedHead = fieldLane === "blend" ? 0 : fieldLane;
  const fieldForward = (site: string) =>
    fieldLane === "blend"
      ? `(1.0 - u.alpha) * ${fwdCall(0, "uk", site)} + u.alpha * ${fwdCall(1, "uk", site)}`
      : fwdCall(fieldLane, "uk", site);
  const fieldBackward = (site: string) =>
    fieldLane === "blend"
      ? `let _du0 = ${bwdCall(0, "dSig[t] * (1.0 - u.alpha)", site)};
      let _du1 = ${bwdCall(1, "dSig[t] * u.alpha", site)};`
      : `let _duLane = ${bwdCall(fieldLane, "dSig[t]", site)};`;
  const advBase = (j: string) => `sBase + ${sl.advOff}u + (${j}) * ${sl.advBlk}u`;
  const aOutLast = sl.advAOff[advL.heads[0].layers.length - 1];
  const dOutLast = sl.advDOff[advL.heads[0].layers.length - 1];
  const pairCount = (k * (k - 1)) / 2;
  const signalForward =
    target.tag === "post-velocity"
      ? `
      let safeMaxVel = max(u.maxVel, 1e-12);
      vn[t] = mvel[t] / safeMaxVel;
      let preV = (mvel[t] + u.forceMag * F) * u.friction;
      let clippedV = clamp(preV, vec2f(-u.maxVel), vec2f(u.maxVel));
      sig[t] = clippedV / safeMaxVel;
      let physJac = u.forceMag * u.friction / safeMaxVel;
      sigJac[t] = physJac * vec2f(
        select(0.0, 1.0, abs(preV.x) < u.maxVel),
        select(0.0, 1.0, abs(preV.y) < u.maxVel)
      );`
      : `
      vn[t] = vec2f(0.0);
      sig[t] = F;
      sigJac[t] = vec2f(1.0);`;
  const signalBackward =
    target.tag === "post-velocity"
      ? `
    for (var t = 0u; t < ${m}u; t = t + 1u) {
      dSig[t] = dSig[t] * sigJac[t];
    }`
      : "";

  // ---- anti-collapse pressure fragments (all "" when pressure is none, so
  // the emitted text is byte-identical to the pre-pressure adversary) --------
  //
  // Batch-coupled, and solved exactly the way the scale-hold energy anchor
  // already is: the moments are reduced + finalized by the PRE-D forward and
  // read back as CONSTANTS by the post-D forward, whose deltas are the ones
  // fieldGrad consumes. Field weights and the sampled tuples are identical
  // across the two forwards (only the predictor updates in between), so the
  // constants are the exact moments of the batch being differentiated — this
  // is not a one-step-stale approximation.
  const usePressure = pressure.tag === "anti-collapse";
  const polarW = pressure.tag === "anti-collapse" ? pressure.polar : 0;
  const nematicW = pressure.tag === "anti-collapse" ? pressure.nematic : 0;
  // τ² IS the radicand floor (τ > 0 is enforced by checkedPressure), so
  // sqrt(dot(F,F) + τ²) needs no extra ε and matches directionOrderLoss's
  // `.add(tau*tau).sqrt()` bit for bit.
  const pressureTau2 = pressure.tag === "anti-collapse" ? pressure.tau * pressure.tau : 0;
  const pressureStatDecl = usePressure ? `\n  var statMom = vec4f(0.0);` : "";
  const pressureRawDecl = usePressure ? `\n    var fRaw : array<vec2f, ${m}>;` : "";
  // Raw F(x) — NOT sig: on a post-velocity target sig is a velocity, and the
  // collapse this prices is a fact about the FIELD's directions.
  const pressureCapture = usePressure
    ? `\n      fRaw[t] = select(vec2f(0.0), F, isFiniteF(F.x) && isFiniteF(F.y));`
    : "";
  const pressureMoments = usePressure
    ? `
    // Every member contributes, target-active or not: tuple validity is a
    // property of the observer, while the direction field exists regardless.
    for (var t = 0u; t < ${m}u; t = t + 1u) {
      let pf = fRaw[t];
      let ps = sqrt(dot(pf, pf) + ${flit(pressureTau2)});
      let pu = pf / ps;
      statMom = statMom + vec4f(
        pu.x, pu.y, pu.x * pu.x - pu.y * pu.y, 2.0 * pu.x * pu.y
      );
    }`
    : "";
  const pressureBwd = usePressure
    ? `
    // dL/du = (2·wP·M₁ + 2·wN·(2·M₂c·uₓ + 2·M₂s·u_y, −2·M₂c·u_y + 2·M₂s·uₓ))/N
    // dL/dF = (I − uuᵀ)/s · dL/du — one reciprocal and one projection per
    // member, no extra field evaluation and no second-order term.
    let pM1 = vec2f(stats[${MOM}u], stats[${MOM + 1}u]);
    let pM2 = vec2f(stats[${MOM + 2}u], stats[${MOM + 3}u]);
    let pN = f32(max(u.b, 1u)) * ${flit(m)};
    let pdM1 = 2.0 * ${flit(polarW)} * pM1 / pN;
    let pdM2 = 2.0 * ${flit(nematicW)} * pM2 / pN;
    for (var t = 0u; t < ${m}u; t = t + 1u) {
      let pf = fRaw[t];
      let ps = sqrt(dot(pf, pf) + ${flit(pressureTau2)});
      let pu = pf / ps;
      let pdU = vec2f(
        pdM1.x + 2.0 * pdM2.x * pu.x + 2.0 * pdM2.y * pu.y,
        pdM1.y - 2.0 * pdM2.x * pu.y + 2.0 * pdM2.y * pu.x
      );
      let pdF = (pdU - pu * dot(pu, pdU)) / ps;
      dSig[t] = dSig[t] +
        select(vec2f(0.0), pdF, isFiniteF(pdF.x) && isFiniteF(pdF.y));
    }`
    : "";
  const pressureReduce = usePressure
    ? `
  workgroupBarrier();
  red4[tid] = statMom;
  workgroupBarrier();
  stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red4[tid] = red4[tid] + red4[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) {
    stats[pb + ${MOM}u] = red4[0].x;
    stats[pb + ${MOM + 1}u] = red4[0].y;
    stats[pb + ${MOM + 2}u] = red4[0].z;
    stats[pb + ${MOM + 3}u] = red4[0].w;
  }`
    : "";
  const pressureRedDecl = usePressure
    ? `\nvar<workgroup> red4 : array<vec4f, ${WG}>;`
    : "";

  // ---- per-family payoff instrument (all "" when off, so a classless or
  // multi-member game emits the pre-family text verbatim) -------------------
  //
  // `statFam[c]` is this thread's tuple payoff when its family is c and 0
  // otherwise; `statFamN[c]` is the matching indicator. Reducing BOTH is what
  // makes the chart a mean rather than a sum — family sizes are a hash of the
  // particle index, so they are only equal in expectation, and dividing three
  // unequal sums by one batch size would draw three curves that differ by
  // sampling noise alone.
  const FAM = statsL.familyOff;
  const familyStatDecl =
    famCount > 0
      ? `
  var statFam : array<f32, ${famCount}>;` +
        `
  var statFamN : array<f32, ${famCount}>;` +
        `
  for (var c = 0u; c < ${famCount}u; c = c + 1u) { statFam[c] = 0.0; statFamN[c] = 0.0; }`
      : "";
  const familyAccum =
    famCount > 0
      ? `
    // m == 1 is what makes this attribution exact (familyInstrument).
    let famId = min(mcls[0], ${famCount - 1}u);
    statFam[famId] = sur;
    statFamN[famId] = 1.0;`
      : "";
  const familyReduce =
    famCount > 0
      ? Array.from({ length: 2 * famCount }, (_, i) => {
          const c = i >> 1;
          const src = i % 2 === 0 ? `statFam[${c}]` : `statFamN[${c}]`;
          return `
  workgroupBarrier();
  red[tid] = ${src};
  workgroupBarrier();
  stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) { stats[pb + ${FAM + i}u] = red[0]; }`;
        }).join("")
      : "";
  const familyFinalDecl =
    famCount > 0
      ? `
  var sfam : array<f32, ${2 * famCount}>;` +
        `
  for (var i = 0u; i < ${2 * famCount}u; i = i + 1u) { sfam[i] = 0.0; }`
      : "";
  const familyFinalAcc =
    famCount > 0
      ? `
    for (var i = 0u; i < ${2 * famCount}u; i = i + 1u) {
      sfam[i] = sfam[i] + stats[pb + ${FAM}u + i];
    }`
      : "";
  const familyFinalWrite =
    famCount > 0
      ? `
  // MEAN payoff per family, then the family's tuple count. An EMPTY family
  // (no sampled particle carried that label this batch) writes 0 payoff and a
  // 0 count — the host reads the count and reports "no sample", never a 0 that
  // would draw as "this family is perfectly predicted".
  for (var c = 0u; c < ${famCount}u; c = c + 1u) {
    let n = sfam[2u * c + 1u];
    stats[${FAM}u + 2u * c] = select(0.0, sfam[2u * c] / n, n > 0.0);
    stats[${FAM}u + 2u * c + 1u] = n;
  }`
      : "";
  const pressureFinalDecl = usePressure ? `\n  var smom = vec4f(0.0);` : "";
  const pressureFinalAcc = usePressure
    ? `
    smom = smom + vec4f(
      stats[pb + ${MOM}u], stats[pb + ${MOM + 1}u],
      stats[pb + ${MOM + 2}u], stats[pb + ${MOM + 3}u]
    );`
    : "";
  const pressureFinalWrite = usePressure
    ? `
  // Divided here so the backward reads the MEANS directly. R₁ = ‖(x,y)‖ and
  // R₂ = ‖(z,w)‖ are derived on the host (AdvStats.directionOrder).
  let momN = bf * ${flit(m)};
  stats[${MOM}u] = smom.x / momN;
  stats[${MOM + 1}u] = smom.y / momN;
  stats[${MOM + 2}u] = smom.z / momN;
  stats[${MOM + 3}u] = smom.w / momN;`
    : "";

  // K-unrolled fragments
  const advFwdCalls = Array.from({ length: k }, (_, j) =>
    `    adv_fwd_${j}(sBase + ${sl.uOff}u, ${advBase(`${j}u`)});`
  ).join("\n");
  const softVectorCount = vectorDy / 2;
  const softResidForHead = (j: number): string => `
    {
      let ab = ${advBase(`${j}u`)};
      var angleResid = 0.0;
      for (var v = 0u; v < ${softVectorCount}u; v = v + 1u) {
        let o = 2u * v;
        let p = vec2f(
          scratch[ab + ${aOutLast}u + o],
          scratch[ab + ${aOutLast}u + o + 1u]
        );
        angleResid = angleResid +
          softSphereChord(p, vec2f(yvec[o], yvec[o + 1u])) /
          ${flit(softVectorCount)};
      }
      ${scaleMode ? `
      var scaleSS = 0.0;
      for (var q = 0u; q < ${scaleDy}u; q = q + 1u) {
        let e = scratch[ab + ${aOutLast + vectorDy}u + q] -
          yvec[${vectorDy}u + q];
        scaleSS = scaleSS + e * e / ${flit(scaleDy)};
      }
      let scaleResid = sqrt(scaleSS + ${ADV_SOFT_EPS2}) - 1e-6;
      resid[${j}] = select(
        0.0, angleResid + ${flit(scaleWeight)} * scaleResid, targetActive
      );` : `
      resid[${j}] = select(0.0, angleResid, targetActive);`}
    }`;
  const legacyResidForHead = (j: number): string => `
    {
      let ab = ${advBase(`${j}u`)};
      let pred = vec2f(
        scratch[ab + ${aOutLast}u],
        scratch[ab + ${aOutLast}u + 1u]
      );
      let pden = sqrt(dot(pred, pred) + ${ADV_SOFT_EPS2});
      let phat = pred / pden;
      let d = phat - vec2f(yvec[0], yvec[1]);
      resid[${j}] = select(
        0.0,
        sqrt(dot(d, d) + ${ADV_SOFT_EPS2}) - 1e-6,
        targetActive
      );
    }`;
  const rawResidForHead = (j: number): string => `
    {
      var ss = ${ADV_SOFT_EPS2};
      let ab = ${advBase(`${j}u`)};
      for (var o = 0u; o < ${dy}u; o = o + 1u) {
        let d = scratch[ab + ${aOutLast}u + o] - yvec[o];
        ss = ss + d * d;
      }
      let r = select(0.0, sqrt(ss), targetActive);
      // Non-finite residuals must not enter WTA/Adam — one NaN zeroes the batch.
      resid[${j}] = select(0.0, r, isFiniteF(r));
    }`;
  const residCalc = Array.from({ length: k }, (_, j) =>
    softAngular
      ? softResidForHead(j)
      : legacyAngular
        ? legacyResidForHead(j)
        : rawResidForHead(j)
  ).join("");

  const softDiscForHead = (j: number): string => `
    {
      let ab = ${advBase(`${j}u`)};
      let wj = ${k === 1 ? "1.0" : `select(${flit(loserW)}, ${flit(winW)}, win == ${j}u)`};
      for (var v = 0u; v < ${softVectorCount}u; v = v + 1u) {
        let o = 2u * v;
        let p = vec2f(
          scratch[ab + ${aOutLast}u + o],
          scratch[ab + ${aOutLast}u + o + 1u]
        );
        var dp = vec2f(0.0);
        if (targetActive) {
          dp = u.discSeed * wj / ${flit(softVectorCount)} *
            softSphereGradP(p, vec2f(yvec[o], yvec[o + 1u]));
        }
        scratch[ab + ${dOutLast}u + o] = dp.x;
        scratch[ab + ${dOutLast}u + o + 1u] = dp.y;
      }
      ${scaleMode ? `
      var scaleSS = 0.0;
      for (var q = 0u; q < ${scaleDy}u; q = q + 1u) {
        let e = scratch[ab + ${aOutLast + vectorDy}u + q] -
          yvec[${vectorDy}u + q];
        scaleSS = scaleSS + e * e / ${flit(scaleDy)};
      }
      let scaleDen = sqrt(scaleSS + ${ADV_SOFT_EPS2});
      for (var q = 0u; q < ${scaleDy}u; q = q + 1u) {
        var dp = 0.0;
        if (targetActive) {
          dp = u.discSeed * wj * ${flit(scaleWeight)} *
            (scratch[ab + ${aOutLast + vectorDy}u + q] -
             yvec[${vectorDy}u + q]) /
            (${flit(scaleDy)} * scaleDen);
        }
        scratch[ab + ${dOutLast + vectorDy}u + q] = dp;
      }` : ""}
      adv_bwd_${j}(ab);
    }`;
  const legacyDiscForHead = (j: number): string => `
    {
      let ab = ${advBase(`${j}u`)};
      let wj = ${k === 1 ? "1.0" : `select(${flit(loserW)}, ${flit(winW)}, win == ${j}u)`};
      let pred = vec2f(
        scratch[ab + ${aOutLast}u],
        scratch[ab + ${aOutLast}u + 1u]
      );
      let pden = sqrt(dot(pred, pred) + ${ADV_SOFT_EPS2});
      let phat = pred / pden;
      var dr = vec2f(0.0);
      if (targetActive) {
        dr = u.discSeed * wj *
          (phat - vec2f(yvec[0], yvec[1])) / (resid[${j}] + 1e-6);
      }
      let dp = dr / pden - pred * (dot(pred, dr) / (pden * pden * pden));
      scratch[ab + ${dOutLast}u] = dp.x;
      scratch[ab + ${dOutLast}u + 1u] = dp.y;
      adv_bwd_${j}(ab);
    }`;
  const rawDiscForHead = (j: number): string => `
    {
      let ab = ${advBase(`${j}u`)};
      let wj = ${k === 1 ? "1.0" : `select(${flit(loserW)}, ${flit(winW)}, win == ${j}u)`};
      for (var o = 0u; o < ${dy}u; o = o + 1u) {
        let pred = scratch[ab + ${aOutLast}u + o];
        var dp = 0.0;
        if (targetActive) {
          dp = u.discSeed * wj * (pred - yvec[o]) / resid[${j}];
        }
        scratch[ab + ${dOutLast}u + o] = dp;
      }
      adv_bwd_${j}(ab);
    }`;
  const discBwd = Array.from({ length: k }, (_, j) =>
    softAngular
      ? softDiscForHead(j)
      : legacyAngular
        ? legacyDiscForHead(j)
        : rawDiscForHead(j)
  ).join("");
  const headSpreadCalc =
    k === 1
      ? ""
      : Array.from({ length: k }, (_, i) =>
          Array.from({ length: k - i - 1 }, (_, q) => {
            const j = i + q + 1;
            if (legacyAngular) {
              return `
    {
      let ha = ${advBase(`${i}u`)};
      let hb = ${advBase(`${j}u`)};
      let pa = vec2f(
        scratch[ha + ${aOutLast}u],
        scratch[ha + ${aOutLast}u + 1u]
      );
      let pb = vec2f(
        scratch[hb + ${aOutLast}u],
        scratch[hb + ${aOutLast}u + 1u]
      );
      let ua = pa / sqrt(dot(pa, pa) + ${ADV_SOFT_EPS2});
      let ub = pb / sqrt(dot(pb, pb) + ${ADV_SOFT_EPS2});
      let hd = length(ua - ub);
      statHeadMean = statHeadMean + hd;
      statHeadMin = min(statHeadMin, hd);
    }`;
            }
            if (softAngular) {
              return `
    {
      let ha = ${advBase(`${i}u`)};
      let hb = ${advBase(`${j}u`)};
      var hd = 0.0;
      for (var v = 0u; v < ${softVectorCount}u; v = v + 1u) {
        let o = 2u * v;
        let pa = vec2f(
          scratch[ha + ${aOutLast}u + o],
          scratch[ha + ${aOutLast}u + o + 1u]
        );
        let pb = vec2f(
          scratch[hb + ${aOutLast}u + o],
          scratch[hb + ${aOutLast}u + o + 1u]
        );
        hd = hd + softSphereChord(pa, pb) / ${flit(softVectorCount)};
      }
      ${scaleMode ? `
      var hss = 0.0;
      for (var q = 0u; q < ${scaleDy}u; q = q + 1u) {
        let d = scratch[ha + ${aOutLast + vectorDy}u + q] -
          scratch[hb + ${aOutLast + vectorDy}u + q];
        hss = hss + d * d / ${flit(scaleDy)};
      }
      hd = hd + ${flit(scaleWeight)} * sqrt(hss + ${ADV_SOFT_EPS2});` : ""}
      statHeadMean = statHeadMean + hd;
      statHeadMin = min(statHeadMin, hd);
    }`;
            }
            return `
    {
      let ha = ${advBase(`${i}u`)};
      let hb = ${advBase(`${j}u`)};
      var hd2 = 0.0;
      for (var o = 0u; o < ${dy}u; o = o + 1u) {
        let d = scratch[ha + ${aOutLast}u + o] -
          scratch[hb + ${aOutLast}u + o];
        hd2 = hd2 + d * d;
      }
      let hd = sqrt(hd2);
      statHeadMean = statHeadMean + hd;
      statHeadMin = min(statHeadMin, hd);
    }`;
          }).join("")
        ).join("");
  const generatorBwd = softAngular ? `
      if (targetActive) {
        for (var v = 0u; v < ${softVectorCount}u; v = v + 1u) {
          let o = 2u * v;
          let p = vec2f(
            scratch[ab + ${aOutLast}u + o],
            scratch[ab + ${aOutLast}u + o + 1u]
          );
          let gy = softSphereGradY(p, vec2f(yvec[o], yvec[o + 1u]));
          // u.genSeed > 0 is disagree: minimize -payoff. A negative seed is
          // the agreeing lane: minimize +payoff.
          let dg = -u.genSeed * wj / ${flit(softVectorCount)} * gy;
          dLdy[o] = dLdy[o] + dg.x;
          dLdy[o + 1u] = dLdy[o + 1u] + dg.y;
        }
        ${scaleMode ? `
        var scaleSS = 0.0;
        for (var q = 0u; q < ${scaleDy}u; q = q + 1u) {
          let e = scratch[ab + ${aOutLast + vectorDy}u + q] -
            yvec[${vectorDy}u + q];
          scaleSS = scaleSS + e * e / ${flit(scaleDy)};
        }
        let scaleDen = sqrt(scaleSS + ${ADV_SOFT_EPS2});
        for (var q = 0u; q < ${scaleDy}u; q = q + 1u) {
          let p = scratch[ab + ${aOutLast + vectorDy}u + q];
          let y = yvec[${vectorDy}u + q];
          ${loss.tag === "angle-scale-hold"
            ? `// Scale is cooperative for BOTH field roles in scale-hold.
          dLdy[${vectorDy}u + q] = dLdy[${vectorDy}u + q] +
            abs(u.genSeed) * wj * ${flit(scaleWeight)} * (y - p) /
            (${flit(scaleDy)} * scaleDen);`
            : `dLdy[${vectorDy}u + q] = dLdy[${vectorDy}u + q] +
            u.genSeed * wj * ${flit(scaleWeight)} * (p - y) /
            (${flit(scaleDy)} * scaleDen);`}
        }` : ""}
      }` : legacyAngular ? `
      let pred = vec2f(
        scratch[ab + ${aOutLast}u],
        scratch[ab + ${aOutLast}u + 1u]
      );
      let pden = sqrt(dot(pred, pred) + ${ADV_SOFT_EPS2});
      let phat = pred / pden;
      if (targetActive) {
        let d = u.genSeed * wj *
          (phat - vec2f(yvec[0], yvec[1])) / (resid[j] + 1e-6);
        dLdy[0] = dLdy[0] + d.x;
        dLdy[1] = dLdy[1] + d.y;
      }` : `
      if (targetActive) {
        for (var o = 0u; o < ${dy}u; o = o + 1u) {
          let pred = scratch[ab + ${aOutLast}u + o];
          dLdy[o] = dLdy[o] + u.genSeed * wj *
            (pred - yvec[o]) / resid[j];
        }
      }`;

  return /* wgsl */ `
struct UAdv {
  res : vec2f,
  forceMag : f32,
  friction : f32,
  maxVel : f32,
  alpha : f32,
  genSeed : f32,   // positive coefficient on -sharedPayoff; 0 = generator off
  discSeed : f32,  // 1/B
  b : u32,
  partCount : u32,
  sampleOffset : u32, // first live-particle index in this coverage window
  source : u32,    // 0 = uploaded tuples, 2 = live particles
  wgCount : u32,
  surpriseStride : u32, // floats between packed raw and per-unit planes
  pad1 : u32,
};
@group(0) @binding(0) var<uniform> u : UAdv;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read> advW : array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch : array<f32>;
@group(0) @binding(4) var<storage, read_write> stats : array<f32>;
@group(0) @binding(5) var<storage, read> partPos : array<vec2f>;
@group(0) @binding(6) var<storage, read> partVel : array<vec2f>;
@group(0) @binding(7) var<storage, read_write> surprise : array<f32>;
@group(0) @binding(8) var<storage, read> tuples : array<f32>;

${COMMON}

fn observerDelta2(a : vec2f) -> vec2f {
  return ${observerGeometry === "periodic" ? "a - round(a)" : "a"};
}
fn b2u(c : bool) -> u32 { return select(0u, 1u, c); }
fn isFiniteF(x : f32) -> bool {
  // NaN ≠ itself; ±Inf exceeds the largest finite f32.
  return x == x && abs(x) <= 3.402823466e+38;
}
${softAngular ? `
fn softSphereEmbed(a : vec2f) -> vec3f {
  let q = vec3f(a, ${flit(tau)});
  return q / sqrt(dot(q, q));
}
fn softSphereChord(a : vec2f, b : vec2f) -> f32 {
  let d = softSphereEmbed(a) - softSphereEmbed(b);
  return sqrt(dot(d, d) + ${ADV_SOFT_EPS2}) - 1e-6;
}
fn softSphereGradP(a : vec2f, b : vec2f) -> vec2f {
  let qa = vec3f(a, ${flit(tau)});
  let qb = vec3f(b, ${flit(tau)});
  let ra = sqrt(dot(qa, qa));
  let ea = qa / ra;
  let eb = qb / sqrt(dot(qb, qb));
  let d = ea - eb;
  let gd = d / sqrt(dot(d, d) + ${ADV_SOFT_EPS2});
  return (gd.xy - ea.xy * dot(ea, gd)) / ra;
}
fn softSphereGradY(a : vec2f, b : vec2f) -> vec2f {
  let qa = vec3f(a, ${flit(tau)});
  let qb = vec3f(b, ${flit(tau)});
  let ea = qa / sqrt(dot(qa, qa));
  let rb = sqrt(dot(qb, qb));
  let eb = qb / rb;
  let d = eb - ea;
  let gd = d / sqrt(dot(d, d) + ${ADV_SOFT_EPS2});
  return (gd.xy - eb.xy * dot(eb, gd)) / rb;
}
` : ""}

${enc.kind === "raw" ? "" : emitEncode(enc) + "\n"}
${heads.map((h, i) => emitFwdStore(i, h, fsl, maxWField, enc)).join("\n\n")}

${heads.map((h, i) => emitBwdStore(i, h, fsl, maxWField, enc, i === dEncSeedHead ? "seed" : "accumulate")).join("\n\n")}

${advL.heads.map((h, j) => emitAdvFwd(j, h, sl, maxWAdv)).join("\n\n")}

${advL.heads.map((h, j) => emitAdvBwd(j, h, sl, maxWAdv)).join("\n\n")}

var<workgroup> red : array<f32, ${WG}>;${pressureRedDecl}
var<workgroup> winCnt : array<atomic<u32>, ${k}>;

@compute @workgroup_size(${WG})
fn advFwd(@builtin(global_invocation_id) gid : vec3u,
          @builtin(local_invocation_index) tid : u32,
          @builtin(workgroup_id) wgid : vec3u) {
  if (tid < ${k}u) { atomicStore(&winCnt[tid], 0u); }
  workgroupBarrier();

  let s = gid.x;
  let res = u.res;
  var statW = 0.0;
  var statS = 0.0;
  var statY2 = 0.0;
  var statHeadMean = 0.0;
  var statHeadMin = 0.0;
  var statEnergy = 0.0;
  var statActive = 0.0;${pressureStatDecl}${familyStatDecl}
  if (s < u.b) {
    let sBase = s * ${STRIDE}u;

    // ---- sample / load the m member states -------------------------------
    var mpos : array<vec2f, ${m}>;
    var mvel : array<vec2f, ${m}>;
    var midx : array<u32, ${m}>;
    if (u.source == 2u) {
      for (var t = 0u; t < ${m}u; t = t + 1u) {
        // Host caps B so B*m <= partCount. Therefore every scatter target in
        // this dispatch is unique, including when the window wraps, and the
        // non-atomic surprise write below is race-free.
        midx[t] = (u.sampleOffset + s * ${m}u + t) % max(u.partCount, 1u);
        mpos[t] = partPos[midx[t]];
        mvel[t] = partVel[midx[t]];
      }
    } else {
      let tb = s * ${m * 4}u;
      for (var t = 0u; t < ${m}u; t = t + 1u) {
        mpos[t] = vec2f(tuples[tb + t * 4u], tuples[tb + t * 4u + 1u]);
        mvel[t] = vec2f(tuples[tb + t * 4u + 2u], tuples[tb + t * 4u + 3u]);
        midx[t] = s * ${m}u + t;
      }
    }
    for (var t = 0u; t < ${m}u; t = t + 1u) {
      scratch[sBase + ${sl.idxOff}u + t] = f32(midx[t]);
    }${
      planed
        ? `
    // FAMILY LABEL — never stored in a particle buffer, always derived, with
    // the SAME salt and modulus the advect kernel and the renderer use. If
    // these three derivations ever disagree the cloud is advected by one
    // family's field and coloured as another, silently.
    var mcls : array<u32, ${m}>;
    for (var t = 0u; t < ${m}u; t = t + 1u) {
      mcls[t] = pcg(midx[t] ^ ${CLASS_SALT}u) % ${field.classes}u;
      scratch[sBase + ${sl.clsOff}u + t] = f32(mcls[t]);
    }`
        : ""
    }

    // ---- selected target signal per member. Force mode is raw F(x).
    // Post-velocity is the exact normalized, post-force/post-friction/post-clip
    // state BEFORE borders/reset; its clip Jacobian is retained for field bwd.
    var pn : array<vec2f, ${m}>;
    var vn : array<vec2f, ${m}>;
    var sig : array<vec2f, ${m}>;
    var sigJac : array<vec2f, ${m}>;${pressureRawDecl}
    for (var t = 0u; t < ${m}u; t = t + 1u) {
      let uk = mpos[t] / res;
      pn[t] = uk;
      scratch[sBase + ${sl.siteInOff}u + t * 2u] = uk.x;
      scratch[sBase + ${sl.siteInOff}u + t * 2u + 1u] = uk.y;
      ${encodeAt("uk", "t")}let F = ${fieldForward("t")};${pressureCapture}
${signalForward}
      // A nonfinite field signal (SELU Inf·0 before the clamp landed, or a
      // future encoding) must not enter κ or the WTA residual.
      if (!isFiniteF(sig[t].x) || !isFiniteF(sig[t].y)) {
        sig[t] = vec2f(0.0);
        sigJac[t] = vec2f(0.0);
      }
    }
${pressureMoments}
    // ---- tuple encoding κ -------------------------------------------------
    var uvec : array<f32, ${du}>;
    var yvec : array<f32, ${dy}>;
${encFB.fwd}
    for (var i = 0u; i < ${du}u; i = i + 1u) {
      let ui = uvec[i];
      uvec[i] = select(0.0, ui, isFiniteF(ui));
      scratch[sBase + ${sl.uOff}u + i] = uvec[i];
    }
    for (var o = 0u; o < ${dy}u; o = o + 1u) {
      let yo = yvec[o];
      yvec[o] = select(0.0, yo, isFiniteF(yo));
      scratch[sBase + ${sl.yOff}u + o] = yvec[o];
    }

    // ---- K adversary heads: forward, residuals, winner -------------------
${advFwdCalls}
    // Head geometry is a property of the predictors at this context, not of
    // target validity. It therefore includes directionless/ambiguous targets.
    ${k > 1 ? "statHeadMin = 1e30;" : ""}
${headSpreadCalc}
    ${k > 1 ? `statHeadMean = statHeadMean / ${flit(pairCount)};` : ""}
    var resid : array<f32, ${k}>;${residCalc}
    var win = 0u;
    var best = resid[0];
    for (var j = 1u; j < ${k}u; j = j + 1u) {
      if (resid[j] < best) { win = j; best = resid[j]; }
    }
    var weighted = 0.0;
    ${k === 1
      ? `weighted = resid[0];`
      : `for (var j = 0u; j < ${k}u; j = j + 1u) {
      weighted = weighted + select(${flit(loserW)}, ${flit(winW)}, j == win) * resid[j];
    }`}
    // Shared discriminator score. Raw, soft-angle, and relative-scale modes
    // are zero-sum on this scalar. Scale-hold is intentionally general-sum:
    // the field reverses the direction term but cooperates on relative scale,
    // with a separate positive energy anchor applied below.
    let sur = select(0.0, weighted, isFiniteF(weighted));
    scratch[sBase + ${sl.surOff}u] = sur;
    scratch[sBase + ${sl.winOff}u] = f32(win);
    var targetNorm2 = 0.0;
    for (var o = 0u; o < ${vectorDy}u; o = o + 1u) {
      let y = yvec[o];
      if (isFiniteF(y)) { targetNorm2 = targetNorm2 + y * y; }
    }
    var perUnitSignal = 0.0;
    // This branch is intentionally exact: an inactive/directionless target is
    // not "100% surprising" merely because both numerator and denominator use
    // epsilon. The diagnostic never feeds scratch, stats, or either gradient.
    if (
      targetActive &&
      targetNorm2 > ${flit(ADV_DIRECTION_ACTIVE_FLOOR ** 2)}
    ) {
      perUnitSignal =
        sur / max(sqrt(targetNorm2), ${flit(ADV_DIRECTION_ACTIVE_FLOOR)});
    }
    for (var t = 0u; t < ${m}u; t = t + 1u) {
      surprise[midx[t]] = sur;
      surprise[u.surpriseStride + midx[t]] = perUnitSignal;
    }

    // ---- DISCRIMINATOR backward (δ's for advOpt; weights are tape
    //      constants — see relaxedWtaSum semantics, verified in B1) ---------
${discBwd}

    // ---- GENERATOR backward. Winner weights are tape constants. Raw,
    // soft-angle and relative-scale reverse the discriminator score;
    // scale-hold reverses direction but cooperates on scale. ------------------
    var dLdy : array<f32, ${dy}>;
    for (var o = 0u; o < ${dy}u; o = o + 1u) { dLdy[o] = 0.0; }
    for (var j = 0u; j < ${k}u; j = j + 1u) {
      let ab = ${advBase("j")};
      let wj = ${k === 1 ? "1.0" : `select(${flit(loserW)}, ${flit(winW)}, j == win)`};
${generatorBwd}
    }
    ${scaleMode ? `
    // Finalized by the pre-D forward. The first forward also executes this
    // code, but its field deltas are deliberately discarded before fieldGrad.
    // finalize already guarantees sqrt(meanEnergy + 1e-16) >= 1e-8.
    let energyRms = stats[${5 + k}u];
    let energyActive = max(stats[${6 + k}u], 1.0);
    let energyAnchorSeed =
      abs(u.genSeed) * f32(u.b) / energyActive *
      2.0 * ${flit(energyWeight)} *
      (energyRms - ${flit(energyTarget)}) / energyRms;` : `
    let energyAnchorSeed = 0.0;`}
    var dSig : array<vec2f, ${m}>;
${encFB.bwd}
${signalBackward}${pressureBwd}
    for (var t = 0u; t < ${m}u; t = t + 1u) {
      ${fieldBackward("t")}
    }

    statW = weighted;
    statS = sur;${familyAccum}
    if (targetActive) {
      // Guard the RMS reductions: a single nonfinite y-component used to make
      // batchRms/energyRms NaN even after resid was zeroed, which the HUD
      // surfaces as surprise NaN via the reward-scale EMA path.
      if (isFiniteF(targetNorm2)) { statY2 = targetNorm2; }
      if (isFiniteF(tupleEnergy)) { statEnergy = tupleEnergy; }
      statActive = 1.0;
      atomicAdd(&winCnt[win], 1u);
    }
  }

  // ---- workgroup reduction → partials ------------------------------------
  workgroupBarrier();
  let pb = ${ADV_STATS_BASE}u + wgid.x * ${PSTRIDE}u;
  red[tid] = statW;
  workgroupBarrier();
  var stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) { stats[pb] = red[0]; }
  workgroupBarrier();
  red[tid] = statS;
  workgroupBarrier();
  stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) { stats[pb + 1u] = red[0]; }
  workgroupBarrier();
  red[tid] = statY2;
  workgroupBarrier();
  stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) {
    stats[pb + 2u] = red[0];
  }
  workgroupBarrier();
  red[tid] = statHeadMean;
  workgroupBarrier();
  stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) { stats[pb + 3u] = red[0]; }
  workgroupBarrier();
  red[tid] = statHeadMin;
  workgroupBarrier();
  stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) {
    stats[pb + 4u] = red[0];
    for (var j = 0u; j < ${k}u; j = j + 1u) {
      stats[pb + 5u + j] = f32(atomicLoad(&winCnt[j]));
    }
  }
  workgroupBarrier();
  red[tid] = statEnergy;
  workgroupBarrier();
  stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) { stats[pb + ${5 + k}u] = red[0]; }
  workgroupBarrier();
  red[tid] = statActive;
  workgroupBarrier();
  stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) { stats[pb + ${6 + k}u] = red[0]; }${pressureReduce}${familyReduce}
}

// finalize: legacy five scalars + win counts, then energy RMS/active count.
@compute @workgroup_size(1)
fn advFinalize() {
  var sw = 0.0;
  var ss = 0.0;
  var sy = 0.0;
  var shMean = 0.0;
  var shMin = 0.0;
  var se = 0.0;
  var sa = 0.0;${pressureFinalDecl}${familyFinalDecl}
  var wins : array<f32, ${k}>;
  for (var j = 0u; j < ${k}u; j = j + 1u) { wins[j] = 0.0; }
  for (var wg = 0u; wg < u.wgCount; wg = wg + 1u) {
    let pb = ${ADV_STATS_BASE}u + wg * ${PSTRIDE}u;
    sw = sw + stats[pb];
    ss = ss + stats[pb + 1u];
    sy = sy + stats[pb + 2u];
    shMean = shMean + stats[pb + 3u];
    shMin = shMin + stats[pb + 4u];
    for (var j = 0u; j < ${k}u; j = j + 1u) { wins[j] = wins[j] + stats[pb + 5u + j]; }
    se = se + stats[pb + ${5 + k}u];
    sa = sa + stats[pb + ${6 + k}u];${pressureFinalAcc}${familyFinalAcc}
  }
  let bf = f32(max(u.b, 1u));
  stats[0] = sw / bf;             // discriminator loss  (mean weighted residual)
  stats[1] = ss / bf;             // mean raw shared payoff
  stats[2] = sqrt(sy / bf);       // batch RMS‖y‖ (feeds the host EMA)
  stats[3] = shMean / bf;         // mean over contexts of mean head-pair distance
  stats[4] = shMin / bf;          // mean over contexts of closest head-pair distance
  for (var j = 0u; j < ${k}u; j = j + 1u) { stats[5u + j] = wins[j]; }
  // Match the core/AD energy anchor exactly. This is deliberately 1e-16,
  // distinct from the 1e-12 squared soft norm used by chord residuals.
  stats[${5 + k}u] = sqrt(se / max(sa, 1.0) + 1e-16);
  stats[${6 + k}u] = sa;${pressureFinalWrite}${familyFinalWrite}
}
`;
}

// ---------------------------------------------------------------------------
// PASS B — advOpt (adversary Adam) + fieldGrad (generator reward assembly)
// ---------------------------------------------------------------------------

/**
 * Bindings (both entries share one explicit layout):
 *   0 uniform UAdvB { lr, beta1, beta2, eps, t, apply, b, pad }
 *   1 rw advW   2 ro scratch   3 rw advGrads   4 rw adamM   5 rw adamV
 *   6 rw extGrads (FIELD-weight-length; consumed by train_wgsl pass B)
 *
 * THE ANTI-COLLAPSE PRESSURE NEEDS NO CODE HERE. Pass A seeds it into the same
 * `dSig` the reward uses, so by the time this entry reads the scratch it is
 * assembling ∇(genSeed·reward + pressure) — which is also why the term reaches
 * fourier and hashgrid encodings through the machinery that already existed.
 * The only pressure-dependent byte in this shader is one comment line, emitted
 * because "genSeed == 0 ⇒ exact zeros" stops being true once a pressure is
 * declared; a pressure-free call still emits the historical text verbatim.
 */
export function adversaryPassBShader(
  field: FieldLayout,
  advL: AdversaryLayout,
  opts: AdvShaderOpts
): string {
  validateAdversaryFusion(field, advL);
  checkedObserverGeometry(opts.observerGeometry);
  const zeroSeedNote =
    checkedPressure(opts.pressure).tag === "none"
      ? ` When\n// genSeed == 0 every δ is 0 and this writes exact zeros — no stale reward.`
      : `\n// L_gen = genSeed·reward + anti-collapse pressure (seeded into the same dSig),\n` +
        `// so genSeed == 0 leaves the PRESSURE alone steering the field — which is\n` +
        `// exactly what an explicit ?advWeight=0 on a pressured piece should mean.`;
  const sl = advScratchLayout(
    field,
    advL,
    opts.tag,
    checkedTarget(opts.target),
    opts.loss
  );
  const { m } = sl;
  const fsl = sl.fieldSl;
  const heads = field.spec.heads as HeadSpec[];
  const enc = field.encoding;
  const fieldLane = checkedFieldLane(opts.fieldLane);
  const STRIDE = sl.stride;

  // --- adversary weight blocks (thread = adversary weight float) -----------
  const advBlocks: string[] = [];
  for (const seg of advL.segments) {
    const j = seg.head;
    const l = seg.layer;
    const L = advL.heads[j].layers[l];
    const start = seg.floatOffset;
    const end = seg.floatOffset + seg.floatLength;
    const ab = `sBase + ${sl.advOff}u + ${j * sl.advBlk}u`;
    const aIn =
      l === 0
        ? `scratch[sBase + ${sl.uOff}u + i]`
        : `scratch[${ab} + ${sl.advAOff[l - 1]}u + i]`;
    if (seg.role === "kernel") {
      advBlocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let local = t - ${start}u;
    let i = local / ${L.outSize}u;
    let o = local % ${L.outSize}u;
    for (var s = 0u; s < ub.b; s = s + 1u) {
      let sBase = s * ${STRIDE}u;
      g = g + ${aIn} * scratch[${ab} + ${sl.advDOff[l]}u + o];
    }
  }`);
    } else {
      advBlocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let o = t - ${start}u;
    for (var s = 0u; s < ub.b; s = s + 1u) {
      let sBase = s * ${STRIDE}u;
      g = g + scratch[${ab} + ${sl.advDOff[l]}u + o];
    }
  }`);
    }
  }

  // --- field weight blocks (thread = FIELD weight float → extGrads) --------
  const fieldBlocks: string[] = [];
  for (const seg of field.segments) {
    if (seg.role === "grid") {
      // hashgrid feature table (field weights offset 0): thread = one grid
      // float (cell, feature). Transliterated from train_wgsl's field pass B
      // (same gather-side formulation: each grid float scans the (tuple,
      // member) sites and claims the bilinear corners that land on its cell,
      // so there is exactly ONE writer per grid float and no atomics). At the
      // clamp border ix1==ix makes TWO corners match — both add, matching tfjs
      // summing coincident scatters.
      //
      // NOTE the ordering: this branch is deliberately ABOVE the fieldLane
      // skip below. The grid table is shared by both field heads, and lane
      // isolation has already happened upstream — only the active lane's
      // bwd_head wrote dEnc.
      if (enc.kind !== "hashgrid") {
        throw new Error(
          `adversary: grid segment on a ${enc.kind} encoding — layout is inconsistent`
        );
      }
      const { gridSize: gs, features: F } = enc;
      // Family plane. The cell comparison carries `cls · gs²`, so a grid float
      // in plane c collects gradient ONLY from members labelled c — which is
      // what makes the three families separately trainable. Dropping the term
      // does not crash and does not NaN: it silently trains one shared plane
      // and the families converge to identical fields, so
      // tools/family_grid_test.ts gates plane isolation explicitly.
      const planeTerm =
        encodingPlanes(enc) > 1 ? `u32(scratch[sBase + ${sl.clsOff}u + site]) * ${gs * gs}u + ` : ``;
      fieldBlocks.push(`
  if (t >= ${seg.floatOffset}u && t < ${seg.floatOffset + seg.floatLength}u) {
    let cell = (t - ${seg.floatOffset}u) / ${F}u;
    let f = (t - ${seg.floatOffset}u) % ${F}u;
    for (var s = 0u; s < ub.b; s = s + 1u) {
      let sBase = s * ${STRIDE}u;
      for (var site = 0u; site < ${m}u; site = site + 1u) {
        let ux = scratch[sBase + ${sl.siteInOff}u + site * 2u];
        let uy = scratch[sBase + ${sl.siteInOff}u + site * 2u + 1u];
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
        g = g + wsum * scratch[sBase + ${sl.dEncOff}u + site * ${fsl.encDim}u + f];
      }
    }
  }`);
      continue;
    }
    const h = seg.head;
    // Direct lanes are structurally isolated. No block means g remains exact
    // zero for every packed weight owned by the inactive field head.
    if (fieldLane !== "blend" && h !== fieldLane) continue;
    const l = seg.layer;
    const L = heads[h].layers[l];
    const start = seg.floatOffset;
    const end = seg.floatOffset + seg.floatLength;
    const headOff = h === 0 ? 0 : fsl.headBlk[0];
    const fb = `sBase + ${sl.fieldSiteOff}u + site * ${sl.fieldSiteBlk}u + ${headOff}u`;
    const aIn =
      l === 0
        ? enc.kind === "raw"
          ? `scratch[sBase + ${sl.siteInOff}u + site * 2u + i]`
          : `scratch[sBase + ${sl.encOff}u + site * ${fsl.encDim}u + i]`
        : `scratch[${fb} + ${fsl.aOff[h][l - 1]}u + i]`;
    if (seg.role === "kernel") {
      fieldBlocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let local = t - ${start}u;
    let i = local / ${L.outSize}u;
    let j = local % ${L.outSize}u;
    for (var s = 0u; s < ub.b; s = s + 1u) {
      let sBase = s * ${STRIDE}u;
      for (var site = 0u; site < ${m}u; site = site + 1u) {
        g = g + ${aIn} * scratch[${fb} + ${fsl.dOff[h][l]}u + j];
      }
    }
  }`);
    } else {
      fieldBlocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let j = t - ${start}u;
    for (var s = 0u; s < ub.b; s = s + 1u) {
      let sBase = s * ${STRIDE}u;
      for (var site = 0u; site < ${m}u; site = site + 1u) {
        g = g + scratch[${fb} + ${fsl.dOff[h][l]}u + j];
      }
    }
  }`);
    }
  }

  return /* wgsl */ `
struct UAdvB {
  lr : f32,
  beta1 : f32,
  beta2 : f32,
  eps : f32,
  t : u32,
  apply : u32,
  b : u32,
  pad : u32,
};
@group(0) @binding(0) var<uniform> ub : UAdvB;
@group(0) @binding(1) var<storage, read_write> advW : array<f32>;
@group(0) @binding(2) var<storage, read> scratch : array<f32>;
@group(0) @binding(3) var<storage, read_write> advGrads : array<f32>;
@group(0) @binding(4) var<storage, read_write> adamM : array<f32>;
@group(0) @binding(5) var<storage, read_write> adamV : array<f32>;
@group(0) @binding(6) var<storage, read_write> extGrads : array<f32>;

fn isFiniteF(x : f32) -> bool {
  return x == x && abs(x) <= 3.402823466e+38;
}

// thread = one packed ADVERSARY weight float: dW = Σ_tuples aIn⊗δ, then Adam
// on the adversary's own buffer/moments (the field optimizer never sees them).
@compute @workgroup_size(${ADV_WG_B})
fn advOpt(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x;
  if (t >= ${advL.totalFloats}u) { return; }
  var g = 0.0;
${advBlocks.join("\n")}
  // Drop non-finite grads before they enter Adam: Inf/Inf in the bias-corrected
  // ratio is the fused tip-over that paints surprise NaN on the live Quad piece.
  g = select(0.0, g, isFiniteF(g));
  advGrads[t] = g;

  if (ub.apply == 1u) {
    let mm = ub.beta1 * adamM[t] + (1.0 - ub.beta1) * g;
    let vv = ub.beta2 * adamV[t] + (1.0 - ub.beta2) * g * g;
    adamM[t] = mm;
    adamV[t] = vv;
    let tf_ = f32(ub.t);
    let mhat = mm / (1.0 - pow(ub.beta1, tf_));
    let vhat = vv / (1.0 - pow(ub.beta2, tf_));
    let step = ub.lr * mhat / (sqrt(vhat) + ub.eps);
    let next = advW[t] - step;
    advW[t] = select(advW[t], next, isFiniteF(next));
  }
}

// thread = one packed FIELD weight float: the generator reward's gradient
// dL_gen/dW = Σ_(tuple, member) aIn⊗δ_field, OVERWRITTEN each step into
// extGrads. The field's pass B (extGrad:true) adds it before its Adam.${zeroSeedNote}
@compute @workgroup_size(${ADV_WG_B})
fn fieldGrad(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x;
  if (t >= ${field.totalFloats}u) { return; }
  var g = 0.0;
${fieldBlocks.join("\n")}
  extGrads[t] = select(0.0, g, isFiniteF(g));
}
`;
}
