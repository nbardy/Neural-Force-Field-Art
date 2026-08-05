import * as tf from "@tensorflow/tfjs";

/**
 * ADVERSARY — a predictor that tries to guess what the field will do next, and a
 * generator reward equal to how badly it fails.
 * ============================================================================
 *
 * ## Why the old `PredictorEnsemble` had to go (deleted, was src/core/gan/ensemble.ts)
 *
 * That design mapped absolute position x -> displacement Δx and rewarded the
 * generator for ENSEMBLE VARIANCE across k predictors. It is mathematically
 * degenerate for this system: the target is
 *
 *     F(x)
 *
 * i.e. an EXACT, deterministic, realizable function of the very same 2-vector
 * that is fed in. Every predictor has enough capacity to fit F, so all k
 * converge to the same function, the ensemble variance → 0, and the generator
 * signal dies. Its "noisy-TV immunity" docstring described a property that is
 * vacuous when the conditional P(Y|U) is a point mass. It also had zero callers
 * anywhere in the repo, so there was nothing to migrate.
 *
 * ## What replaces it: residual distortion under a quantized predictor
 *
 * Keep the adversarial framing, but make the prediction problem GENUINELY
 * one-to-many, and measure the irreducible residual instead of the variance.
 *
 *   - A **tuple encoding** κ maps a tuple of particles to (u, y): a context and
 *     a target. For `pair` we canonicalize by the Euclidean group, which THROWS
 *     AWAY absolute position and orientation — so a single u corresponds to many
 *     different y across the domain. P(Y|U) is then a real, multimodal
 *     distribution, not a point mass.
 *   - A **predictor** g guesses y from u.
 *   - The discriminator and generator use the SAME relaxed-WTA payoff. The
 *     discriminator minimizes it; the generator minimizes its negative. There
 *     is no second, silently different "generator surprise" objective.
 *
 * The strict API consumes the generator's RAW FORCE/SIGNAL `F(x)`, not a
 * one-step transition. That distinction is load-bearing: a transition also
 * depends on incoming velocity, friction, clipping, boundary handling, and
 * resets. If the predictor sees only position while the target includes those
 * hidden variables, even the point control is artificially one-to-many.
 * `sampleSignal`/`encodeSignal` are therefore the canonical game. Explicit
 * `sampleTransition`/`encodeTransition` methods remain only as legacy
 * visualization/test adapters and are named so they cannot be mistaken for
 * the strict game.
 *
 * Two variants, and only one of them is expected to work:
 *
 *   **VARIANT A `single`** — one predictor, `loss = ‖y − g(u)‖`. This is the
 *   CONTROL. A single head cannot represent a multimodal conditional, so it
 *   settles in the empty space between the modes: the residual is the full
 *   spread of P(Y|U) and the "surprise" signal carries no information about
 *   WHICH mode a tuple is in. It is built honestly here so the collapse is
 *   observable — do not tune it.
 *
 *   MEASURED SUBTLETY, worth knowing before reading any collapse number: the
 *   loss is the NORM `‖·‖`, not the squared norm, so the population minimizer is
 *   the GEOMETRIC MEDIAN of P(Y|U), not the mean. For the exact norm and two
 *   equally likely modes every point on the joining segment is a minimizer.
 *   This implementation uses `sqrt(||e||² + SOFT_EPS²)` for finite gradients,
 *   which very slightly rounds that flat set and selects the midpoint; at the
 *   fixture's O(1) scale the difference is below float32 resolution. Therefore
 *   tests may use the half-separation as a numerical approximation, but must
 *   not claim the softened objective is mathematically exactly flat.
 *
 *   **VARIANT B `wta`** — K guesses with a relaxed-WTA payoff
 *   (Multiple Choice Learning / winner-take-all). Only the closest head is
 *   trained most strongly on a given sample, so the heads QUANTIZE P(Y|U) —
 *   they spread out and cover the modes instead of averaging them. The strict
 *   game payoff includes the winner plus the named relaxed loser shares on both
 *   D and G sides: "even this K-head quantizer cannot pin me down."
 *
 * ## RELAXED WTA is mandatory, not a refinement
 *
 * Plain WTA has a well known failure: **head collapse**. A head that never wins
 * receives exactly zero gradient, so it never moves, so it keeps never winning.
 * K silently degrades to 1 and the loss looks fine while the model has stopped
 * being a mixture. The fix (Rupprecht et al., "relaxed-WTA") is to give the
 * winner weight `1 − ε` and let the `K − 1` losers share `ε`, so every head
 * always receives some gradient. Default ε = 0.05.
 *
 * Additionally every `trainStep` records PER-HEAD WIN COUNTS, so collapse is
 * DETECTABLE rather than silent — see `winStats()` and `collapsed()`.
 *
 * ## Structural notes
 *
 *  - The winner-take-all min is **per-sample**, not batch-coupled: it reduces
 *    over the K heads inside one sample. That matters for the eventual fused
 *    port — unlike the isotropy covariance it needs NO two-pass fold, and it is
 *    directly expressible in the scalar AD IR by folding `g.min` over a
 *    compile-time K loop (the exact shape of `spiralTerm`,
 *    src/render/webgpu/ad/losses.ts:65).
 *  - Adversary parameters are held STRICTLY SEPARATE from the generator field.
 *    Each predictor gets its own Adam restricted to its own `varList` — see the
 *    comment in `trainStep`.
 *  - This module is the tfjs REFERENCE implementation. It is the oracle the
 *    fused WGSL port must match numerically; it is deliberately readable rather
 *    than fast (it does K small GPU→CPU readbacks per step).
 */

/* ══════════════════════════════════════════════════════════════════════════
   0. Canonical domain types.  D = ⊕ᵢ Dᵢ
   ══════════════════════════════════════════════════════════════════════════ */

/** Which prediction game the adversary plays. */
export type AdversaryKind =
  /** One predictor, one guess. The frozen-generator realizability control. */
  | { readonly tag: "single" }
  /**
   * K guesses, relaxed winner-take-all. Winner dominance requires
   * `0 <= relaxEps < (k - 1) / k`; larger values give each loser more weight
   * than the nominal winner and are rejected.
   */
  | { readonly tag: "wta"; readonly k: number; readonly relaxEps: number };

/** What physical quantity the predictor is asked to predict. */
export type AdversaryTarget =
  /**
   * The raw learned field output F(x), before physical force scaling,
   * integration, clipping, borders, or reset. This is the strict historical
   * game and remains the compatibility default.
   */
  | { readonly tag: "force" }
  /**
   * The caller-supplied normalized post-update velocity. This target is point
   * only: its predictor context is [x, v] so incoming momentum is observed
   * rather than manufactured as hidden-state surprise.
   */
  | { readonly tag: "post-velocity" };

/**
 * How a prediction error is scored. Target source and loss are deliberately
 * orthogonal: e.g. both force and post-velocity may use the same soft-angle
 * objective.
 */
export type AdversaryLoss =
  /** Softened Euclidean vector distance. Deliberate amplitude-cheat control. */
  | { readonly tag: "raw-vector" }
  /**
   * Mean chord distance between per-vector S² embeddings
   * ψτ(z)=(zx,zy,τ)/sqrt(||z||²+τ²). This is a real smooth scalar objective,
   * not a surrogate backward.
   */
  | { readonly tag: "soft-angle"; readonly tau: number }
  /**
   * Soft angle plus a local, homogeneous relative log-magnitude descriptor.
   * D minimizes and disagreeing G maximizes both terms. The generator also
   * pays a fixed energy anchor so variance cannot be bought by global blow-up.
   */
  | {
      readonly tag: "angle-relative-scale";
      readonly tau: number;
      readonly scaleWeight: number;
      readonly energyWeight: number;
      readonly energyTarget: number;
    }
  /**
   * D predicts both angle and local relative scale. Disagreeing G maximizes
   * angular error but COOPERATES on scale prediction, plus a fixed energy
   * anchor. This is a named general-sum game, not one shared zero-sum scalar.
   */
  | {
      readonly tag: "angle-scale-hold";
      readonly tau: number;
      readonly scaleWeight: number;
      readonly energyWeight: number;
      readonly energyTarget: number;
    };

/** Canonical defaults used by gallery/UI ingestion and reference tests. */
export const DEFAULT_ADVERSARY_TARGET: AdversaryTarget = { tag: "force" };
export const DEFAULT_ADVERSARY_LOSS: AdversaryLoss = { tag: "raw-vector" };
export const DEFAULT_SOFT_ANGLE_TAU = 0.05;

/**
 * Geometry used by relational observers when subtracting particle positions.
 *
 * This is deliberately independent from the tuple encoding. A wrapped
 * particle domain is a torus, so positions across opposite edges are nearby
 * (`periodic`). Bounce and edge-reset domains are ordinary bounded rectangles,
 * so opposite edges are far apart (`euclidean`). Callers must choose this from
 * the live boundary mode; there is no implicit default at construction.
 */
export type ObserverGeometry = "periodic" | "euclidean";

/** How a tuple of particles becomes a (context, target) pair. */
export type TupleEncoding =
  /** m = 1. u = absolute normalized position, y = its force/signal. */
  | { readonly tag: "point" }
  /**
   * Legacy spelling of `pair-rotation`. Kept while the production WGSL/UI
   * migrates; new code should use the explicit name.
   */
  | { readonly tag: "pair" }
  /**
   * m = 2. Translation/rotation quotient: u=[separation], y=relative signal in
   * the co-rotating pair frame. Scale remains observable and rewardable.
   */
  | { readonly tag: "pair-rotation" }
  /**
   * m = 2 negative control. The context discards separation (u=[1]) but the
   * target remains raw, so increasing signal magnitude still buys payoff.
   */
  | { readonly tag: "pair-rotation-scale-raw" }
  /**
   * Legacy spelling of the same scale-blind pair context/raw canonical target
   * as `pair-rotation-scale-raw`. It no longer changes or normalizes `y`.
   * `defaultAdversaryConfig` maps this spelling to `soft-angle`; explicit
   * callers may choose any objective. Kept only for URL/config compatibility.
   */
  | { readonly tag: "pair-rotation-scale-adjusted" }
  /** m = 3. u = the three side lengths in canonical order, y = per-vertex
   *  centered signals in the canonical triangle frame. See {@link encodeTri}. */
  | { readonly tag: "tri" }
  /**
   * m = 4 with EXPLICIT labels (not a set). Member 0 is the anchor and member 1
   * defines the co-rotating frame. u contains the three labelled relative
   * coordinates (members 1..3) in that frame; y contains four mean-centered
   * signal vectors in the same frame.
   *
   * This closes the four-point experiment without pretending a generic
   * permutation-invariant canonical labelling exists. Reordering members changes
   * the sample by design. Tuples whose anchor pair is near-coincident are marked
   * inactive and contribute exact zero.
   */
  | { readonly tag: "quad-labelled" };

/** Arity and widths implied by an encoding. */
export interface EncodingDims {
  /** Points per tuple. */
  readonly m: number;
  /** Width of the context vector u. */
  readonly du: number;
  /** Width of the target vector y. */
  readonly dy: number;
}

/** Predictor widths after an objective adds auxiliary scale channels. */
export interface ObjectiveDims extends EncodingDims {
  /** Local relative-scale descriptor width (0 for every other loss). */
  readonly ds: number;
  /** Actual final Dense width: dy + ds. */
  readonly out: number;
}

/**
 * One batch of encoded tuples. `u`/`y` are DIFFERENTIABLE w.r.t. the positions
 * they were built from — that is how the generator receives its reward.
 * The caller owns these tensors; free them with {@link disposeTupleSample}.
 */
export interface TupleSample {
  /** [B, du] context. */
  readonly u: tf.Tensor2D;
  /** [B, dy] target. */
  readonly y: tf.Tensor2D;
  /**
   * Local homogeneous relative-scale target [B,ds]. Present exactly for
   * `angle-relative-scale`; absent is a named configuration error there.
   */
  readonly relativeScale?: tf.Tensor2D;
  /** Mean squared member-signal norm [B], used by generator energy anchors. */
  readonly tupleEnergy?: tf.Tensor1D;
  /**
   * Stop-gradient 0/1 validity for the objective. It combines geometric chart
   * validity with the all-zero relative-scale inactivity rule.
   */
  readonly objectiveActivity?: tf.Tensor1D;
  /** [B * m] particle indices, row-major per tuple. Lets surprise map back to particles. */
  readonly idx: Int32Array;
  /** Number of tuples. */
  readonly b: number;
  /** Points per tuple. */
  readonly m: number;
}

/** Tensor inputs for the explicit target-source API. All coordinates are normalized. */
export type AdversaryTargetBatch =
  | {
      readonly tag: "force";
      readonly pos: tf.Tensor2D;
      readonly force: tf.Tensor2D;
    }
  | {
      readonly tag: "post-velocity";
      readonly pos: tf.Tensor2D;
      /** Current velocity normalized by the caller's declared velocity scale. */
      readonly velocity: tf.Tensor2D;
      /** Post-update velocity in the same normalized units. */
      readonly nextVelocity: tf.Tensor2D;
    };

/** Full adversary configuration. Every field is required on purpose — see
 *  {@link defaultAdversaryConfig} for a starting point to spread-override. */
export interface AdversaryConfig {
  readonly kind: AdversaryKind;
  readonly encoding: TupleEncoding;
  readonly target: AdversaryTarget;
  readonly loss: AdversaryLoss;
  /**
   * Position-difference policy for pair/tri/quad observers. Must agree with
   * the live particle boundary (`wrap` -> periodic, bounce/reset -> euclidean).
   * Point encoding is numerically unchanged, but still records the policy so
   * switching tuple encodings cannot reveal an undeclared default.
   */
  readonly observerGeometry: ObserverGeometry;
  /** B — tuples drawn per step. Cost is O(B), independent of the particle count N. */
  readonly batchTuples: number;
  /** Width of the first SELU layer of each predictor. */
  readonly hiddenUnits: number;
  /** Width of the second SELU layer of each predictor. */
  readonly featureDim: number;
  /** Adam learning rate for the predictor (discriminator) update. */
  readonly learningRate: number;
  /** Seed for tuple sampling AND for per-head weight init (heads must differ). */
  readonly seed: number;
}

/** What one discriminator update reports back. */
export interface TrainReport {
  /** Mean weighted predictor loss over the batch (the value actually minimized). */
  readonly loss: number;
  /**
   * [k] how many ACTIVE tuples each head won. Sums to the active tuple count
   * (<= B): an undefined triangle/quad chart or inactive all-zero scale tuple
   * has no winner, matching the fused kernel's `if (targetActive)` accounting.
   */
  readonly winCounts: readonly number[];
}

/**
 * Result of {@link Adversary.headSpread} — the head-collapse DISAMBIGUATOR.
 *
 * A skewed win histogram alone cannot tell two very different states apart:
 *
 *   - spread ≈ the mode separation of P(Y|U) + skewed wins  →  DATA-LIMITED.
 *     There are fewer modes than heads; the surplus heads park elsewhere and
 *     rarely win, but the winners are DISTINCT predictors. K > modes is benign
 *     (the mixture still quantizes everything there is to quantize).
 *   - spread ≈ 0 + skewed wins  →  OPTIMIZER PILEUP. The heads have converged
 *     to (nearly) the same function; effective K = 1 regardless of the
 *     configured K, and the WTA machinery is dead weight. Pathology.
 *
 * `meanPair` is the aggregate statistic to read first (a pileup drives it to
 * ~0); `minPair` additionally exposes the closest pair — it goes small even in
 * the benign state when two surplus heads share a parking spot, so a small
 * `minPair` with a healthy `meanPair` is NOT collapse.
 *
 * A NAMED sum, not a nullable: one head has no pairs, and "no pairs exist" is
 * a different fact from "pairs exist and are zero apart".
 */
export type HeadSpread =
  /** kind = single: pairwise spread is undefined, and that is data, not NaN. */
  | { readonly tag: "single-head" }
  | {
      readonly tag: "spread";
      /** Mean over contexts of the mean pairwise distance between head outputs. */
      readonly meanPair: number;
      /** Mean over contexts of the MINIMUM pairwise distance (closest pair). */
      readonly minPair: number;
      /** k·(k−1)/2 — how many pairs the aggregates cover. */
      readonly pairs: number;
    };

/**
 * Per-particle surprise, scattered from per-tuple values.
 *
 * `hits[i] === 0` means particle i was NOT in any sampled tuple this step. That
 * is reported as data rather than silently defaulted to 0.0 surprise, because
 * "unsampled" and "unsurprising" are different facts and a renderer that
 * conflates them draws a lie. With B ≪ N most particles are unhit on any given
 * frame; the intended consumer keeps an EMA and only updates where `hits > 0`.
 */
export interface ScatteredSurprise {
  /** [n] sum of surprise contributions per particle. */
  readonly sum: Float32Array;
  /** [n] how many tuples touched each particle. */
  readonly hits: Int32Array;
  /** Particle count. */
  readonly n: number;
}

/** Thrown by the single canonicalization boundary when a config is not valid. */
export class AdversaryConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AdversaryConfigError";
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   1. Thin dispatchers (δ) — each may ONLY select a handler.
   ══════════════════════════════════════════════════════════════════════════ */

function assertNever(x: never, what: string): never {
  throw new AdversaryConfigError(`${what}: unhandled variant ${JSON.stringify(x)}`);
}

/** δ: AdversaryKind → head count. */
export function headCount(kind: AdversaryKind): number {
  switch (kind.tag) {
    case "single":
      return 1;
    case "wta":
      return kind.k;
    default:
      return assertNever(kind, "headCount");
  }
}

/** δ: TupleEncoding → arity/widths. */
export function encodingDims(encoding: TupleEncoding): EncodingDims {
  switch (encoding.tag) {
    case "point":
      return { m: 1, du: 2, dy: 2 };
    case "pair":
    case "pair-rotation":
      return { m: 2, du: 1, dy: 2 };
    case "pair-rotation-scale-raw":
    case "pair-rotation-scale-adjusted":
      // Dense layers require a non-empty input; the constant one is the
      // explicit representation of an empty similarity-invariant context.
      return { m: 2, du: 1, dy: 2 };
    case "tri":
      return { m: 3, du: 3, dy: 6 };
    case "quad-labelled":
      return { m: 4, du: 6, dy: 8 };
    default:
      return assertNever(encoding, "encodingDims");
  }
}

/** Width of the local relative-scale descriptor for one tuple. */
export function relativeScaleDim(encoding: TupleEncoding): number {
  switch (encoding.tag) {
    case "point":
      throw new AdversaryConfigError(
        "angle-relative-scale needs at least two member signals; point tuples have no local contrast"
      );
    case "pair":
    case "pair-rotation":
    case "pair-rotation-scale-raw":
    case "pair-rotation-scale-adjusted":
      // The legacy "adjusted" tag is now only a compatibility spelling for
      // the same scale-blind pair geometry/raw canonical vector target. Scale
      // handling belongs to the explicit loss, where the unencoded member
      // signals are still available.
      return 1;
    case "tri":
      return 3;
    case "quad-labelled":
      return 4;
    default:
      return assertNever(encoding, "relativeScaleDim");
  }
}

/**
 * Context/target/output dimensions implied jointly by observer, target, loss.
 * This is the single shape dispatcher the tfjs and fused implementations share.
 */
export function objectiveDims(
  encoding: TupleEncoding,
  target: AdversaryTarget,
  loss: AdversaryLoss
): ObjectiveDims {
  const base = encodingDims(encoding);
  if (target.tag === "post-velocity") {
    if (encoding.tag !== "point") {
      throw new AdversaryConfigError(
        `post-velocity target is point-only because relational contexts do not ` +
          `observe incoming velocity; got ${encoding.tag}`
      );
    }
    const ds =
      loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold"
        ? relativeScaleDim(encoding)
        : 0;
    return { m: 1, du: 4, dy: 2, ds, out: 2 + ds };
  }
  const ds =
    loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold"
      ? relativeScaleDim(encoding)
      : 0;
  return { ...base, ds, out: base.dy + ds };
}

/* ══════════════════════════════════════════════════════════════════════════
   2. Numerical guards shared by every handler
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * Softening constant placed INSIDE every sqrt radicand.
 *
 * `d/dx sqrt(0) = ∞`, and a residual of exactly zero is not a pathological
 * input here — it is the state a perfectly-fitting predictor reaches, i.e.
 * precisely where the adversary spends its time. An outer `+ε` after the sqrt
 * does NOT fix this. Same lesson, same fix, as src/core/losses/chaos.ts:50-55.
 */
export const SOFT_EPS = 1e-6;

/**
 * Legacy per-unit visualization activity floor. The exact S² training
 * objectives do not branch at this threshold and do not divide by ||z||.
 * Exported while the shipped WGSL per-unit display and older probes migrate.
 */
export const DIRECTION_ACTIVE_FLOOR = 1e-3;

/** @deprecated The exact S² objective uses its explicit `tau` instead. */
export const PREDICTOR_DIRECTION_EPS = 1e-6;

/**
 * Relative-scale logs use a softener proportional to the tuple's own energy:
 * log sqrt(||s_i||² + REL_SCALE_SOFT_RATIO²·mean_j||s_j||²).
 * Uniform signal scaling multiplies every radicand by c², so centering removes
 * the resulting +log(c) exactly. A fixed epsilon would break that property.
 */
export const REL_SCALE_SOFT_RATIO = 1e-3;
/** A genuinely all-zero tuple has no relative magnitude contrast. */
export const REL_SCALE_ALL_ZERO_FLOOR = 1e-8;
/** Smooth absolute-value softener for the pair's swap-invariant log contrast. */
export const REL_SCALE_ABS_EPS = 1e-6;
/** Numerical softener for the generator's RMS-energy anchor. */
export const ENERGY_ANCHOR_EPS = 1e-8;

/**
 * Frame-degeneracy softener for the pair encoding.
 *
 * The pair frame is `ê∥ = d / ‖d‖`, undefined at d = 0. Rather than reject
 * coincident pairs (a data-dependent branch, and a silent one) the separation is
 * DEFINED as `r = sqrt(‖d‖² + FRAME_EPS²)` and the frame as `d / r`. This is
 * part of the encoding, not a fallback: it is smooth everywhere, the frame
 * shrinks continuously to the zero vector as the points coincide, and both
 * target components therefore go to 0 there. Cost: r is biased upward by
 * O(FRAME_EPS²/2r) — negligible against real separations in [0,1] torus units.
 */
const FRAME_EPS = 1e-6;
/** Below this separation a labelled quad has no stable anchor frame. */
export const QUAD_ANCHOR_ACTIVE_FLOOR = 1e-5;
/** Triangles with side gaps at/below this floor have no unique vertex order. */
export const TRI_SIDE_TIE_FLOOR = 1e-5;

/* ══════════════════════════════════════════════════════════════════════════
   3. Tuple sampling + encoding handlers (κ: Raw → canonical (u, y))
   ══════════════════════════════════════════════════════════════════════════ */

/** Deterministic PRNG — the repo's standard (tools/grad_reference.ts). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Minimum-image reduction on the unit torus: maps a raw coordinate difference
 * in (−1, 1) to the shortest representative in [−0.5, 0.5]. Relational
 * encoders reach this only through `observerDelta`, so bounce/reset observers
 * cannot accidentally inherit wrap geometry.
 *
 * SUBTLE AND LOAD-BEARING. Particle positions are wrapped with a floored mod, so
 * a pair straddling the periodic seam has a raw difference of ≈ ±1 despite
 * being nearby. The legacy transition adapter has the same issue for a particle
 * that wraps between frames. `tf.round` has zero gradient in tfjs, so
 * `a − round(a)` is gradient-identity in `a` — differentiability is preserved.
 */
/* ─── generator-reward normalization ─────────────────────────────────────── */

/**
 * EMA constant for the legacy raw-reward scale telemetry (~50-step horizon).
 * It smooths per-batch noise; it is not the mathematical scale fix (see
 * `generatorLoss` and the adjusted encoding).
 */
const RMS_EMA_LAMBDA = 0.02;
/** Floor for the normalizer so a near-frozen field cannot divide by ~0. */
const RMS_FLOOR = 1e-6;

/**
 * Reward-normalizer state. A NAMED sum, not a nullable number: "unseeded" is a
 * real state (no `trainStep` has observed a batch yet — exactly the first
 * frame) and is handled deliberately in `generatorLoss`, never defaulted
 * silently.
 */
export type RewardScale = { tag: "unseeded" } | { tag: "seeded"; rms: number };

function minImage(a: tf.Tensor2D): tf.Tensor2D {
  return a.sub(a.round()) as tf.Tensor2D;
}

/** Apply the caller-selected geometry to a raw normalized-position delta. */
function observerDelta(
  a: tf.Tensor2D,
  geometry: ObserverGeometry
): tf.Tensor2D {
  switch (geometry) {
    case "periodic":
      return minImage(a);
    case "euclidean":
      return a;
    default:
      return assertNever(geometry, "observerDelta");
  }
}

/**
 * Draw `b` tuples of `m` DISTINCT particle indices out of `n`.
 * O(b·m) — independent of n, which is the whole point (N is up to 1e6).
 */
function sampleIndices(n: number, b: number, m: number, rnd: () => number): Int32Array {
  if (n < m) {
    throw new AdversaryConfigError(`sampleIndices: need at least m=${m} particles, got n=${n}`);
  }
  const idx = new Int32Array(b * m);
  const taken: number[] = [];
  for (let t = 0; t < b; t++) {
    taken.length = 0;
    for (let j = 0; j < m; j++) {
      // Member j is drawn uniformly from the n−j slots not yet taken, then
      // shifted past each already-taken index at or below it. This is exact
      // sampling-without-replacement (no rejection loop, so no worst-case
      // blowup when n is close to m), and it is uniform over the remaining set.
      let v = Math.floor(rnd() * (n - j));
      for (const tk of taken) if (v >= tk) v++;
      idx[t * m + j] = v;
      taken.push(v);
      taken.sort((p, q) => p - q);
    }
  }
  return idx;
}

/** Column `c` of an [B,2] tensor as [B]. */
function col(t: tf.Tensor2D, c: number): tf.Tensor1D {
  return t.slice([0, c], [-1, 1]).reshape([-1]) as tf.Tensor1D;
}

/** Hard activity decision with an explicit zero derivative. tfjs's Greater op
 * has no registered gradient, so merely relying on "comparisons are
 * nondifferentiable" makes backprop throw. This custom boundary states the
 * intended derivative instead. */
const hardPositiveActivity = tf.customGrad((margin: tf.Tensor) => {
  const value = margin.greater(0).toFloat();
  return {
    value,
    gradFunc: () => tf.zerosLike(margin),
  };
});

/** Raw Euclidean residual, with the softener inside sqrt for finite gradients. */
function euclideanResidual(pred: tf.Tensor2D, target: tf.Tensor2D): tf.Tensor1D {
  return pred
    .sub(target)
    .square()
    .sum(1)
    .add(SOFT_EPS * SOFT_EPS)
    .sqrt() as tf.Tensor1D;
}

/**
 * Exact smooth embedding R²→S² used by the new angular objectives:
 *
 *   ψτ(z) = (zx, zy, τ) / sqrt(zx² + zy² + τ²).
 *
 * Every pair of columns is embedded independently. Unlike z/||z|| this is
 * defined at zero and its exact Jacobian is bounded by O(1/τ).
 */
export function softSphericalEmbedding(z: tf.Tensor2D, tau: number): tf.Tensor2D {
  if (!(Number.isFinite(tau) && tau > 0)) {
    throw new AdversaryConfigError(`soft spherical tau must be finite and > 0, got ${tau}`);
  }
  if (z.shape[1] % 2 !== 0) {
    throw new AdversaryConfigError(
      `soft spherical embedding needs 2-vector blocks, got width ${z.shape[1]}`
    );
  }
  const blocks: tf.Tensor2D[] = [];
  for (let o = 0; o < z.shape[1]; o += 2) {
    const xy = z.slice([0, o], [-1, 2]) as tf.Tensor2D;
    const den = xy
      .square()
      .sum(1)
      .add(tau * tau)
      .sqrt()
      .reshape([-1, 1]) as tf.Tensor2D;
    blocks.push(
      tf.concat(
        [xy.div(den) as tf.Tensor2D, tf.fill([z.shape[0], 1], tau).div(den) as tf.Tensor2D],
        1
      ) as tf.Tensor2D
    );
  }
  return tf.concat(blocks, 1) as tf.Tensor2D;
}

/** Mean per-2-vector chord distance between exact soft-spherical embeddings. */
export function softAngleResidual(
  pred: tf.Tensor2D,
  target: tf.Tensor2D,
  tau: number
): tf.Tensor1D {
  if (pred.shape[1] !== target.shape[1] || pred.shape[1] % 2 !== 0) {
    throw new AdversaryConfigError(
      `soft angle needs matching even widths, got pred=${pred.shape[1]} target=${target.shape[1]}`
    );
  }
  const p = softSphericalEmbedding(pred, tau);
  const y = softSphericalEmbedding(target, tau);
  const perBlock: tf.Tensor1D[] = [];
  for (let o = 0; o < p.shape[1]; o += 3) {
    perBlock.push(
      p
        .slice([0, o], [-1, 3])
        .sub(y.slice([0, o], [-1, 3]))
        .square()
        .sum(1)
        .add(SOFT_EPS * SOFT_EPS)
        .sqrt()
        .sub(SOFT_EPS) as tf.Tensor1D
    );
  }
  return tf.stack(perBlock, 1).mean(1) as tf.Tensor1D;
}

function scaleResidual(pred: tf.Tensor2D, target: tf.Tensor2D): tf.Tensor1D {
  return pred
    .sub(target)
    .square()
    .mean(1)
    .add(SOFT_EPS * SOFT_EPS)
    .sqrt()
    .sub(SOFT_EPS) as tf.Tensor1D;
}

/**
 * The exact per-row activity predicate used by both payoff masking and win
 * accounting. Keeping one handler prevents an inactive row from having zero
 * payoff/gradient yet still being reported as an argMin tie won by head 0.
 */
function tupleActivity(
  encoding: TupleEncoding,
  target: tf.Tensor2D,
  context: tf.Tensor2D
): tf.Tensor1D {
  if (encoding.tag === "quad-labelled") {
    const anchor = context.slice([0, 0], [-1, 2]) as tf.Tensor2D;
    const margin = anchor
      .square()
      .sum(1)
      .sub(QUAD_ANCHOR_ACTIVE_FLOOR * QUAD_ANCHOR_ACTIVE_FLOOR);
    return hardPositiveActivity(margin) as tf.Tensor1D;
  }
  if (encoding.tag === "tri") {
    // u is descending side length. A unique order exists iff BOTH adjacent
    // gaps exceed the named tie floor.
    const gap01 = context
      .slice([0, 0], [-1, 1])
      .sub(context.slice([0, 1], [-1, 1]))
      .reshape([-1]) as tf.Tensor1D;
    const gap12 = context
      .slice([0, 1], [-1, 1])
      .sub(context.slice([0, 2], [-1, 1]))
      .reshape([-1]) as tf.Tensor1D;
    const margin = tf.minimum(gap01, gap12).sub(TRI_SIDE_TIE_FLOOR);
    return hardPositiveActivity(margin) as tf.Tensor1D;
  }
  return tf.ones([target.shape[0]]) as tf.Tensor1D;
}

function residualFor(
  encoding: TupleEncoding,
  pred: tf.Tensor2D,
  target: tf.Tensor2D,
  context: tf.Tensor2D
): tf.Tensor1D {
  const base = euclideanResidual(pred, target);
  return encoding.tag === "tri" || encoding.tag === "quad-labelled"
    ? (base.mul(tupleActivity(encoding, target, context)) as tf.Tensor1D)
    : base;
}

function lossNeedsEnergy(
  loss: AdversaryLoss
): loss is Extract<
  AdversaryLoss,
  { tag: "angle-relative-scale" | "angle-scale-hold" }
> {
  return loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold";
}

function objectiveActivityFor(cfg: AdversaryConfig, s: TupleSample): tf.Tensor1D {
  if (s.objectiveActivity) return s.objectiveActivity;
  return tupleActivity(cfg.encoding, s.y, s.u);
}

export type AdversaryObjectiveTerms = {
  readonly angle: tf.Tensor1D;
  readonly scale: tf.Tensor1D;
  readonly discriminator: tf.Tensor1D;
};

/** Per-head named terms under an explicit objective. */
function objectiveTermsFor(
  cfg: AdversaryConfig,
  dims: ObjectiveDims,
  pred: tf.Tensor2D,
  s: TupleSample
): AdversaryObjectiveTerms {
  // The legacy adjusted encoding is a compatibility spelling for scale-blind
  // pair geometry with the same raw canonical target. Every objective,
  // including raw-vector, is selected explicitly here.
  if (cfg.loss.tag === "raw-vector") {
    const raw = residualFor(cfg.encoding, pred, s.y, s.u);
    return { angle: raw, scale: tf.zerosLike(raw), discriminator: raw };
  }

  const predVector = pred.slice([0, 0], [-1, dims.dy]) as tf.Tensor2D;
  const activity = objectiveActivityFor(cfg, s);
  const angle = softAngleResidual(predVector, s.y, cfg.loss.tau).mul(
    activity
  ) as tf.Tensor1D;
  let scale = tf.zerosLike(angle) as tf.Tensor1D;
  if (
    cfg.loss.tag === "angle-relative-scale" ||
    cfg.loss.tag === "angle-scale-hold"
  ) {
    if (!s.relativeScale || dims.ds <= 0) {
      throw new AdversaryConfigError(
        `${cfg.loss.tag} sample is missing its local relative-scale target`
      );
    }
    const predScale = pred.slice([0, dims.dy], [-1, dims.ds]) as tf.Tensor2D;
    scale = scaleResidual(predScale, s.relativeScale).mul(activity) as tf.Tensor1D;
  }
  return {
    angle,
    scale,
    discriminator: angle.add(
      scale.mul(cfg.loss.tag === "soft-angle" ? 0 : cfg.loss.scaleWeight)
    ) as tf.Tensor1D,
  };
}

/** Per-head generator game term before WTA weighting and the energy anchor. */
export function generatorObjectiveTerm(
  loss: AdversaryLoss,
  terms: AdversaryObjectiveTerms
): tf.Tensor1D {
  switch (loss.tag) {
    case "raw-vector":
    case "soft-angle":
    case "angle-relative-scale":
      return terms.discriminator.neg() as tf.Tensor1D;
    case "angle-scale-hold":
      return terms.angle
        .neg()
        .add(terms.scale.mul(loss.scaleWeight)) as tf.Tensor1D;
    default:
      return assertNever(loss, "generatorObjectiveTerm");
  }
}

/** Per-head discriminator residual under an explicit objective. */
function objectiveResidualFor(
  cfg: AdversaryConfig,
  dims: ObjectiveDims,
  pred: tf.Tensor2D,
  s: TupleSample
): tf.Tensor1D {
  return objectiveTermsFor(cfg, dims, pred, s).discriminator;
}

/**
 * Generator-only fixed-energy anchor. The target is an RMS member-vector norm;
 * the anchor is evaluated only over active tuples. Above the target its radial
 * derivative points inward; below it points outward.
 */
export function energyAnchorLoss(s: TupleSample, loss: AdversaryLoss): tf.Scalar {
  if (!lossNeedsEnergy(loss)) return tf.scalar(0);
  if (!s.tupleEnergy) {
    throw new AdversaryConfigError(`${loss.tag} sample is missing tupleEnergy`);
  }
  const active = s.objectiveActivity ?? tf.onesLike(s.tupleEnergy);
  const activeCount = tf.maximum(active.sum(), tf.scalar(1));
  const meanEnergy = s.tupleEnergy.mul(active).sum().div(activeCount);
  const rms = meanEnergy.add(ENERGY_ANCHOR_EPS * ENERGY_ANCHOR_EPS).sqrt();
  return rms
    .sub(loss.energyTarget)
    .square()
    .mul(loss.energyWeight)
    .asScalar();
}

/**
 * ENCODING HANDLER — `point` (m = 1). NO canonicalization, deliberately.
 *
 *   u = x                       (absolute normalized position, 2-D)
 *   y = F(x)                    (that particle's raw signal, 2-D)
 *
 * This is the DEGENERATE control. y is an exact deterministic function of u
 * (`Δx = F(x)·dt`), so P(Y|U) is a point mass, any predictor with enough
 * capacity drives the residual to ~0, and the generator reward dies. It is kept
 * as a selectable encoding precisely so that degeneracy is measurable side by
 * side with `pair` rather than asserted in a comment.
 */
function encodePointSignal(
  pos: tf.Tensor2D,
  signal: tf.Tensor2D,
  i0: tf.Tensor1D
): [tf.Tensor2D, tf.Tensor2D] {
  const x = pos.gather(i0, 0) as tf.Tensor2D; // [B,2]
  const y = signal.gather(i0, 0) as tf.Tensor2D; // [B,2]
  return [x, y];
}

/**
 * ENCODING HANDLER — `pair` (m = 2). SE(2)-canonicalized.
 *
 * Given particles (x₁, x₂) and their raw signals (F₁, F₂):
 *
 *   d   = observerDelta(x₂ − x₁)            separation vector
 *   r   = sqrt(‖d‖² + FRAME_EPS²)           separation scalar
 *   ê∥  = d / r                             along-axis unit vector
 *   ê⊥  = rot90(ê∥) = (−ê∥.y, ê∥.x)         perpendicular
 *   δ   = F₂ − F₁                           relative signal
 *
 *   u = [ r ]                               context, 1-D
 *   y = [ δ·ê∥ , δ·ê⊥ ]                     target, 2-D
 *
 * ### Invariances ACTUALLY achieved (no more, no less)
 *
 *  ✔ **Translation** of the whole tuple: exact. Only differences enter.
 *  ✔ **Rotation** of the tuple together with its signals: exact. The frame
 *    co-rotates, so both components of y are unchanged.
 *  ✔ **Point swap** x₁ ↔ x₂: exact. d → −d flips BOTH ê∥ and ê⊥, and δ → −δ, so
 *    the two dot products are unchanged. There is no arbitrary ordering choice
 *    baked into the encoding.
 *  ✔ **Boundary topology:** periodic geometry is exact across the torus seam;
 *    Euclidean geometry deliberately treats opposite bounce/reset edges as far.
 *
 *  ✘ **Reflection** is NOT an invariance: a mirror flips the handedness of ê⊥ and
 *    therefore the sign of y[1]. The encoding is reflection-EQUIVARIANT. This is
 *    intentional — chirality is exactly the thing a curl-carrying field has, and
 *    erasing it would throw away the most interesting structure.
 *  ✘ **Scale** is NOT an invariance: r appears explicitly as the context. Also
 *    intentional; the adversary is supposed to learn scale-dependent structure.
 *
 * ### Why this creates GENUINE ambiguity (the whole point)
 *
 * `r` is the COMPLETE invariant of two labelled points under the Euclidean
 * group — it is all that survives canonicalization. Since the underlying field F
 * is not itself SE(2)-invariant, tuples with identical r sitting at different
 * places/orientations in the domain have genuinely different y. So P(Y|U = r) is
 * a real, generally multimodal distribution and NOT a point mass. Feeding
 * absolute positions and predicting absolute signals — for any m — would
 * still be exactly realizable and would still collapse; it is the canonicalization,
 * not the tuple size, that manufactures the ambiguity.
 */
function encodePairSignal(
  pos: tf.Tensor2D,
  signal: tf.Tensor2D,
  i0: tf.Tensor1D,
  i1: tf.Tensor1D,
  geometry: ObserverGeometry,
  encoding: Extract<
    TupleEncoding,
    {
      tag:
        | "pair"
        | "pair-rotation"
        | "pair-rotation-scale-raw"
        | "pair-rotation-scale-adjusted";
    }
  >
): [tf.Tensor2D, tf.Tensor2D] {
  const x0 = pos.gather(i0, 0) as tf.Tensor2D;
  const x1 = pos.gather(i1, 0) as tf.Tensor2D;
  const s0 = signal.gather(i0, 0) as tf.Tensor2D;
  const s1 = signal.gather(i1, 0) as tf.Tensor2D;

  const d = observerDelta(x1.sub(x0) as tf.Tensor2D, geometry); // [B,2]
  const r = d.square().sum(1).add(FRAME_EPS * FRAME_EPS).sqrt() as tf.Tensor1D; // [B]

  const ax = col(d, 0).div(r) as tf.Tensor1D; // ê∥.x
  const ay = col(d, 1).div(r) as tf.Tensor1D; // ê∥.y
  // ê⊥ = rot90(ê∥). Fixed handedness — see the reflection note above.
  const px = ay.neg() as tf.Tensor1D;
  const py = ax;

  const del = s1.sub(s0) as tf.Tensor2D; // relative force/signal δ [B,2]
  const dx = col(del, 0);
  const dy = col(del, 1);

  const along = dx.mul(ax).add(dy.mul(ay)) as tf.Tensor1D;
  const perp = dx.mul(px).add(dy.mul(py)) as tf.Tensor1D;

  const keepSeparation = encoding.tag === "pair" || encoding.tag === "pair-rotation";
  const u = keepSeparation
    ? (r.reshape([-1, 1]) as tf.Tensor2D)
    : (tf.ones([r.shape[0], 1]) as tf.Tensor2D);
  // Every observer emits the RAW canonical relative signal. Direction/scale is
  // an objective decision now, not an encoding side effect. In particular the
  // legacy `pair-rotation-scale-adjusted` spelling is only a scale-blind
  // context alias; exact S² soft-angle loss replaces its old hard quotient.
  return [u, tf.stack([along, perp], 1) as tf.Tensor2D];
}

type TriOrder = {
  readonly sides: tf.Tensor1D[];
  readonly weights: tf.Tensor1D[][];
};

/** Canonical triangle vertex order shared by geometry and scale descriptors. */
function triCanonicalOrder(
  x: tf.Tensor2D[],
  geometry: ObserverGeometry
): TriOrder {
  const side = (a: tf.Tensor2D, b: tf.Tensor2D): tf.Tensor1D =>
    observerDelta(b.sub(a) as tf.Tensor2D, geometry)
      .square()
      .sum(1)
      .add(FRAME_EPS * FRAME_EPS)
      .sqrt() as tf.Tensor1D;
  const sides = [side(x[1], x[2]), side(x[0], x[2]), side(x[0], x[1])];
  const gt = (a: tf.Tensor1D, b: tf.Tensor1D) =>
    tf.greater(a, b).toFloat() as tf.Tensor1D;
  const eq = (a: tf.Tensor1D, b: tf.Tensor1D) =>
    tf.equal(a, b).toFloat() as tf.Tensor1D;
  const rank = [
    gt(sides[1], sides[0]).add(gt(sides[2], sides[0])) as tf.Tensor1D,
    gt(sides[0], sides[1])
      .add(gt(sides[2], sides[1]))
      .add(eq(sides[0], sides[1])) as tf.Tensor1D,
    gt(sides[0], sides[2])
      .add(gt(sides[1], sides[2]))
      .add(eq(sides[0], sides[2]))
      .add(eq(sides[1], sides[2])) as tf.Tensor1D,
  ];
  const weights = rank.map((rk) => {
    const oh = tf.oneHot(rk.toInt() as tf.Tensor1D, 3).toFloat() as tf.Tensor2D;
    return [col(oh, 0), col(oh, 1), col(oh, 2)];
  });
  return { sides, weights };
}

function triPick2(rows: tf.Tensor2D[], weights: tf.Tensor1D[][], r: number): tf.Tensor2D {
  return rows[0]
    .mul(weights[0][r].reshape([-1, 1]))
    .add(rows[1].mul(weights[1][r].reshape([-1, 1])))
    .add(rows[2].mul(weights[2][r].reshape([-1, 1]))) as tf.Tensor2D;
}

function triPick1(rows: tf.Tensor1D[], weights: tf.Tensor1D[][], r: number): tf.Tensor1D {
  return rows[0]
    .mul(weights[0][r])
    .add(rows[1].mul(weights[1][r]))
    .add(rows[2].mul(weights[2][r])) as tf.Tensor1D;
}

/**
 * ENCODING HANDLER — `tri` (m = 3). E(2)-canonicalized triangle.
 *
 * ### Canonicalization, step by step (this is the spec — the tests mirror it)
 *
 * Given labelled vertices (x₀, x₁, x₂) and their signals (F₀, F₁, F₂):
 *
 *  1. **Opposite side lengths**, each through its OWN observer-geometry delta
 *     so the construction is symmetric in the labels (no base-vertex leak):
 *       s₀ = ‖Δg(x₂ − x₁)‖   (side opposite vertex 0)
 *       s₁ = ‖Δg(x₂ − x₀)‖   (opposite vertex 1)
 *       s₂ = ‖Δg(x₁ − x₀)‖   (opposite vertex 2)
 *     each softened as sqrt(‖d‖² + FRAME_EPS²), same as the pair encoding.
 *
 *  2. **Canonical vertex order** = vertices sorted by DESCENDING opposite side
 *     length. This works precisely because m = 3 gives a 1:1 vertex ↔ opposite
 *     side correspondence, so "sort the sides" IS "sort the vertices". Call the
 *     ordered vertices A, B, C (A faces the longest side).
 *
 *     **TIE CASE — dropped, not label-broken.** If either adjacent sorted-side
 *     gap is <= TRI_SIDE_TIE_FLOOR, there is no unique vertex order. The context
 *     still reports the invariant sorted side lengths, but y and the full
 *     discriminator/generator payoff are masked to exact zero. This includes
 *     exact isosceles/equilateral triangles and a named near-tie band, so labels
 *     can never leak into a supposedly permutation-invariant target.
 *
 *  3. **Chart + centering.** Embed in the plane from the canonical base vertex:
 *     p_A = 0, p_B = Δg(x_B − x_A), p_C = Δg(x_C − x_A); centroid
 *     c = (p_B + p_C)/3; centered vertices v_i = p_i − c. For tuples that span
 *     more than half the domain the per-edge min images of step 1 and this
 *     chart can disagree (the triangle wraps the torus and has no consistent
 *     Euclidean embedding); the encoding stays deterministic and
 *     permutation-invariant because the base is the CANONICAL vertex, but the
 *     Euclidean-invariance claims below are for non-wrapping tuples — the same
 *     regime the pair encoding's §3 gates test.
 *
 *  4. **Canonical frame.** ê₁ = v_A / sqrt(‖v_A‖² + FRAME_EPS²) (centroid → A
 *     direction), ê₂ = rot90(ê₁). Fixed handedness, exactly like `pair`.
 *     v_A = 0 (A at the centroid) only for degenerate collinear tuples; the
 *     FRAME_EPS softening makes the frame shrink smoothly to zero there rather
 *     than branching.
 *
 *  5. **Context** u = [s_A, s_B, s_C] — the opposite side lengths in canonical
 *     (descending) order, du = 3. This is the COMPLETE congruence invariant of
 *     the unordered triangle under the FULL Euclidean group E(2), reflections
 *     included: side lengths determine a triangle up to isometry. It is
 *     deliberately blind to chirality — like `pair`, orientation information
 *     lives in the TARGET (equivariantly), not the context.
 *
 *  6. **Target** y = [δ_A·ê₁, δ_A·ê₂, δ_B·ê₁, δ_B·ê₂, δ_C·ê₁, δ_C·ê₂], dy = 6,
 *     where δᵢ = Fᵢ − mean_j Fⱼ is the per-vertex relative raw field signal
 *     (mean-centered by construction; the three δ sum to
 *     zero, a known linear redundancy kept for readability).
 *
 * ### Invariances ACTUALLY achieved (no more, no less)
 *
 *  ✔ Translation is exact within the selected geometry. Periodic geometry also
 *    handles torus translations across the seam; Euclidean geometry treats the
 *    two bounded-domain edges as distinct, as bounce/reset require.
 *  ✔ Rotation of the tuple together with its signals: exact away from the
 *    seam (the frame co-rotates).
 *  ✔ Vertex permutation: exact off the tie set of step 2 (all six orderings of
 *    a scalene triangle encode identically).
 *  ✘ Reflection is NOT an invariance of y: u is reflection-invariant (side
 *    lengths), but a mirror flips the handedness of ê₂ and therefore the sign
 *    of every ê₂-component y[1], y[3], y[5], while the ê₁-components are
 *    unchanged. Reflection-EQUIVARIANT, same contract as `pair` — chirality is
 *    the signal, do not erase it.
 *  ✘ Scale is NOT an invariance: the side lengths are the context, on purpose.
 *
 * ### Why m = 4 is NOT implemented (the ambiguity that blocks it)
 *
 * The m = 3 ordering is safe because an exact key tie (two equal opposite
 * sides) IMPLIES a symmetry of the point set exchanging the tied vertices, so
 * either resolution is geometrically equivalent. For m = 4 that implication
 * FAILS: no distance-derived vertex key (opposite sides no longer exist; sums
 * of distances, sorted distance profiles, etc.) makes ties imply a
 * configuration symmetry — two vertices of a generic quad can share any such
 * key with NO isometry relating them, so the tie set is no longer "symmetric
 * configurations where the choice is harmless" but an arbitrary surface where
 * index tie-breaking changes y discontinuously in a way no documented
 * equivariance covers. Additionally side lengths stop being a complete
 * congruence invariant at m = 4 (a rhombus and a kite-with-equal-sides share
 * them), so the honest context would need diagonals, whose assignment depends
 * on that same broken ordering — the reflection/ordering interplay. Until
 * there is a canonicalization whose tie classes are all symmetry classes,
 * m = 4 would be a silent-fallback machine and is deliberately absent.
 *
 * ### Implementation note: the sort is a stop-gradient permutation
 *
 * The ranks are computed with `tf.greater`/`tf.equal` (no gradient) and applied
 * as one-hot mixing weights — constants on the tape, exactly the
 * `oneHot(argMin)` trick used by {@link weightsWta}. Gradients flow through the
 * selected VALUES (positions, signals) with the permutation held fixed,
 * which is the correct subgradient of a piecewise-constant ordering.
 */
function encodeTri(
  pos: tf.Tensor2D,
  signal: tf.Tensor2D,
  i0: tf.Tensor1D,
  i1: tf.Tensor1D,
  i2: tf.Tensor1D,
  geometry: ObserverGeometry
): [tf.Tensor2D, tf.Tensor2D] {
  const x = [i0, i1, i2].map((i) => pos.gather(i, 0) as tf.Tensor2D); // 3 × [B,2]
  const sig = [i0, i1, i2].map((i) => signal.gather(i, 0) as tf.Tensor2D);

  const { sides: s, weights: w } = triCanonicalOrder(x, geometry);
  const pick2 = (rows: tf.Tensor2D[], r: number) => triPick2(rows, w, r);
  const pick1 = (vals: tf.Tensor1D[], r: number) => triPick1(vals, w, r);

  // Step 3 — chart based at the canonical vertex A, centroid-centered.
  const xA = pick2(x, 0);
  const xB = pick2(x, 1);
  const xC = pick2(x, 2);
  const dAB = observerDelta(xB.sub(xA) as tf.Tensor2D, geometry);
  const dAC = observerDelta(xC.sub(xA) as tf.Tensor2D, geometry);
  const cen = dAB.add(dAC).div(3) as tf.Tensor2D; // centroid (chart coords; p_A = 0)

  // Step 4 — frame from centroid → A.
  const vA = cen.neg() as tf.Tensor2D;
  const lenA = vA.square().sum(1).add(FRAME_EPS * FRAME_EPS).sqrt() as tf.Tensor1D;
  const e1x = col(vA, 0).div(lenA) as tf.Tensor1D;
  const e1y = col(vA, 1).div(lenA) as tf.Tensor1D;
  const e2x = e1y.neg() as tf.Tensor1D; // ê₂ = rot90(ê₁), fixed handedness
  const e2y = e1x;

  // Step 6 — centered per-vertex signals, canonically ordered, projected.
  const disp = sig;
  const dbar = disp[0].add(disp[1]).add(disp[2]).div(3) as tf.Tensor2D;
  const rel = disp.map((d) => d.sub(dbar) as tf.Tensor2D);
  const proj = (v: tf.Tensor2D): [tf.Tensor1D, tf.Tensor1D] => [
    col(v, 0).mul(e1x).add(col(v, 1).mul(e1y)) as tf.Tensor1D,
    col(v, 0).mul(e2x).add(col(v, 1).mul(e2y)) as tf.Tensor1D,
  ];
  const [alongA, perpA] = proj(pick2(rel, 0));
  const [alongB, perpB] = proj(pick2(rel, 1));
  const [alongC, perpC] = proj(pick2(rel, 2));

  // Step 5 — context: sides in canonical (descending) order.
  const u = tf.stack([pick1(s, 0), pick1(s, 1), pick1(s, 2)], 1) as tf.Tensor2D; // [B,3]
  const rawY = tf.stack([alongA, perpA, alongB, perpB, alongC, perpC], 1) as tf.Tensor2D; // [B,6]
  const gap01 = pick1(s, 0).sub(pick1(s, 1)) as tf.Tensor1D;
  const gap12 = pick1(s, 1).sub(pick1(s, 2)) as tf.Tensor1D;
  const margin = tf.minimum(gap01, gap12).sub(TRI_SIDE_TIE_FLOOR);
  const active = hardPositiveActivity(margin).reshape([-1, 1]);
  const y = rawY.mul(active) as tf.Tensor2D;
  return [u, y];
}

/**
 * ENCODING HANDLER — `quad-labelled` (m = 4).
 *
 * Labels are semantic and preserved:
 *
 *   d_j = observerDelta(x_j - x_0), j=1..3
 *   e_1 = d_1 / ||d_1||, e_2 = rot90(e_1)
 *   u   = [proj(d_1), proj(d_2), proj(d_3)]                 (6-D)
 *   y   = [proj(F_j - mean_i F_i)]_{j=0..3}                (8-D)
 *
 * The quotient removes global translation and rotation. It deliberately does
 * NOT remove scale and does NOT claim permutation invariance. Member 0 is the
 * anchor; member 1 defines the chart. `residualFor` masks samples with
 * ||d_1|| <= QUAD_ANCHOR_ACTIVE_FLOOR to exact zero, because their frame is not
 * identifiable.
 */
function encodeQuadLabelled(
  pos: tf.Tensor2D,
  signal: tf.Tensor2D,
  i0: tf.Tensor1D,
  i1: tf.Tensor1D,
  i2: tf.Tensor1D,
  i3: tf.Tensor1D,
  geometry: ObserverGeometry
): [tf.Tensor2D, tf.Tensor2D] {
  const idx = [i0, i1, i2, i3];
  const x = idx.map((i) => pos.gather(i, 0) as tf.Tensor2D);
  const sig = idx.map((i) => signal.gather(i, 0) as tf.Tensor2D);
  const rel = [1, 2, 3].map((j) =>
    observerDelta(x[j].sub(x[0]) as tf.Tensor2D, geometry)
  );

  // Use the true anchor magnitude for activity semantics; FRAME_EPS is only the
  // safe denominator for the forward chart at/near coincidence.
  const anchorMag2 = rel[0].square().sum(1) as tf.Tensor1D;
  const safeR = anchorMag2.add(FRAME_EPS * FRAME_EPS).sqrt() as tf.Tensor1D;
  const e1x = col(rel[0], 0).div(safeR) as tf.Tensor1D;
  const e1y = col(rel[0], 1).div(safeR) as tf.Tensor1D;
  const e2x = e1y.neg() as tf.Tensor1D;
  const e2y = e1x;
  const project = (v: tf.Tensor2D): [tf.Tensor1D, tf.Tensor1D] => [
    col(v, 0).mul(e1x).add(col(v, 1).mul(e1y)) as tf.Tensor1D,
    col(v, 0).mul(e2x).add(col(v, 1).mul(e2y)) as tf.Tensor1D,
  ];

  const coords = rel.flatMap((d) => project(d));
  const u = tf.stack(coords, 1) as tf.Tensor2D;

  const mean = sig[0].add(sig[1]).add(sig[2]).add(sig[3]).div(4) as tf.Tensor2D;
  const targets = sig.flatMap((f) => project(f.sub(mean) as tf.Tensor2D));
  const rawY = tf.stack(targets, 1) as tf.Tensor2D;

  // Zero the encoded target on inactive rows as data hygiene. The residual mask
  // is still load-bearing: a random predictor at u=0 would otherwise be charged.
  const active = hardPositiveActivity(
    anchorMag2.sub(QUAD_ANCHOR_ACTIVE_FLOOR * QUAD_ANCHOR_ACTIVE_FLOOR)
  ).reshape([-1, 1]);
  const y = rawY.mul(active) as tf.Tensor2D;
  return [u, y];
}

type ScaleDescriptor = {
  readonly target: tf.Tensor2D;
  readonly energy: tf.Tensor1D;
  readonly active: tf.Tensor1D;
};

/**
 * Local relative log-magnitude descriptor from UNENCODED member signals.
 *
 * The log softener is homogeneous in tuple energy, hence target(c·signals) is
 * unchanged for every c>0 (away only from the explicitly inactive all-zero
 * tuple). No batch statistic or stop-gradient approximation is involved.
 */
function relativeScaleDescriptor(
  encoding: TupleEncoding,
  memberPos: tf.Tensor2D[],
  memberSignal: tf.Tensor2D[],
  geometry: ObserverGeometry
): ScaleDescriptor {
  const m = memberSignal.length;
  const mag2 = memberSignal.map((s) => s.square().sum(1) as tf.Tensor1D);
  const energy = tf.addN(mag2).div(m) as tf.Tensor1D;
  const active = hardPositiveActivity(
    energy.sub(REL_SCALE_ALL_ZERO_FLOOR * REL_SCALE_ALL_ZERO_FLOOR)
  ) as tf.Tensor1D;
  // Select a positive safe energy before log. The all-zero branch is masked
  // after descriptor construction, avoiding log(0) and 0·Infinity.
  const safeEnergy = tf.where(
    active.cast("bool"),
    energy,
    tf.onesLike(energy)
  ) as tf.Tensor1D;
  const soft2 = safeEnergy.mul(REL_SCALE_SOFT_RATIO * REL_SCALE_SOFT_RATIO);
  const logs = mag2.map(
    (r2) => r2.add(soft2).log().mul(0.5) as tf.Tensor1D
  );

  let target: tf.Tensor2D;
  switch (encoding.tag) {
    case "point":
      throw new AdversaryConfigError(
        "angle-relative-scale/angle-scale-hold require a relational tuple"
      );
    case "pair":
    case "pair-rotation":
    case "pair-rotation-scale-raw":
    case "pair-rotation-scale-adjusted": {
      const contrast = logs[0]
        .sub(logs[1])
        .square()
        .add(REL_SCALE_ABS_EPS * REL_SCALE_ABS_EPS)
        .sqrt()
        .sub(REL_SCALE_ABS_EPS)
        .mul(active);
      target = contrast.reshape([-1, 1]) as tf.Tensor2D;
      break;
    }
    case "tri": {
      const { weights } = triCanonicalOrder(memberPos, geometry);
      const ordered = [0, 1, 2].map((r) => triPick1(logs, weights, r));
      const mean = tf.addN(ordered).div(3) as tf.Tensor1D;
      target = tf
        .stack(ordered.map((v) => v.sub(mean).mul(active)), 1) as tf.Tensor2D;
      break;
    }
    case "quad-labelled": {
      const mean = tf.addN(logs).div(4) as tf.Tensor1D;
      target = tf
        .stack(logs.map((v) => v.sub(mean).mul(active)), 1) as tf.Tensor2D;
      break;
    }
    default:
      return assertNever(encoding, "relativeScaleDescriptor");
  }
  return { target, energy, active };
}

/** Free the tensors owned by a {@link TupleSample}. */
export function disposeTupleSample(s: TupleSample): void {
  s.u.dispose();
  s.y.dispose();
  s.relativeScale?.dispose();
  s.tupleEnergy?.dispose();
  s.objectiveActivity?.dispose();
}

/* ══════════════════════════════════════════════════════════════════════════
   4. Winner weighting handlers — the ONLY variant-dependent computation
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * HANDLER — `single`. One head, weight 1. (k === 1 is guaranteed by the
 * constructor, so there is nothing to select and nothing to branch on.)
 */
function weightsSingle(resid: tf.Tensor2D): tf.Tensor2D {
  return tf.onesLike(resid) as tf.Tensor2D;
}

/**
 * HANDLER — `wta`, relaxed. Winner gets `1 − ε`; the `k − 1` losers SHARE `ε`,
 * i.e. `ε/(k−1)` each. ε = 0 is exact winner-take-all and is what collapses.
 *
 * The weights are CONSTANTS with respect to the gradient. `tf.argMin` returns
 * int32 and has no gradient, so `oneHot(argMin(...))` is a natural
 * stop-gradient — no detach op needed, and no hand-written argmax either. This
 * mirrors how the AD IR gets WTA for free by folding `g.min`
 * (`spiralTerm`, src/render/webgpu/ad/losses.ts:65): the min's reverse rule
 * routes the gradient to the winning branch on its own.
 */
function weightsWta(k: number, relaxEps: number, resid: tf.Tensor2D): tf.Tensor2D {
  const win = tf.oneHot(tf.argMin(resid, 1), k).toFloat() as tf.Tensor2D;
  const loserShare = relaxEps / (k - 1);
  return win.mul(1 - relaxEps - loserShare).add(loserShare) as tf.Tensor2D;
}

/** δ: AdversaryKind → weighting handler. Trivial body; all work is in the handlers. */
function residualWeights(kind: AdversaryKind, resid: tf.Tensor2D): tf.Tensor2D {
  switch (kind.tag) {
    case "single":
      return weightsSingle(resid);
    case "wta":
      return weightsWta(kind.k, kind.relaxEps, resid);
    default:
      return assertNever(kind, "residualWeights");
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   5. Config canonicalization (the single ingestion point, κ)
   ══════════════════════════════════════════════════════════════════════════ */

/** A sane starting configuration; spread-override the fields you care about. */
export function defaultAdversaryConfig(
  kind: AdversaryKind,
  encoding: TupleEncoding,
  observerGeometry: ObserverGeometry
): AdversaryConfig {
  return {
    kind,
    encoding,
    target: DEFAULT_ADVERSARY_TARGET,
    // The old "adjusted" observer spelling now means scale-blind context plus
    // the exact smooth S² objective. Other encodings keep raw-vector as the
    // compatibility default.
    loss:
      encoding.tag === "pair-rotation-scale-adjusted"
        ? { tag: "soft-angle", tau: DEFAULT_SOFT_ANGLE_TAU }
        : DEFAULT_ADVERSARY_LOSS,
    observerGeometry,
    batchTuples: 512,
    hiddenUnits: 32,
    featureDim: 16,
    learningRate: 1e-3,
    seed: 1234,
  };
}

/**
 * Validate ONCE, here. Everything downstream may assume a canonical config and
 * performs no further checking — no "just in case" guards in the handlers.
 */
function validate(cfg: AdversaryConfig): void {
  // Shape validation also rejects relational post-velocity and point local
  // relative-scale objectives at the canonical ingestion boundary.
  objectiveDims(cfg.encoding, cfg.target, cfg.loss);
  if (cfg.observerGeometry !== "periodic" && cfg.observerGeometry !== "euclidean") {
    throw new AdversaryConfigError(
      `observerGeometry must be explicitly "periodic" or "euclidean", got ` +
        `${JSON.stringify(cfg.observerGeometry)}`
    );
  }
  if (!Number.isInteger(cfg.batchTuples) || cfg.batchTuples < 1) {
    throw new AdversaryConfigError(`batchTuples must be a positive integer, got ${cfg.batchTuples}`);
  }
  if (!(cfg.learningRate > 0)) {
    throw new AdversaryConfigError(`learningRate must be > 0, got ${cfg.learningRate}`);
  }
  switch (cfg.loss.tag) {
    case "raw-vector":
      break;
    case "soft-angle":
      if (!(Number.isFinite(cfg.loss.tau) && cfg.loss.tau > 0)) {
        throw new AdversaryConfigError(`soft-angle tau must be finite and > 0, got ${cfg.loss.tau}`);
      }
      break;
    case "angle-relative-scale":
    case "angle-scale-hold":
      if (!(Number.isFinite(cfg.loss.tau) && cfg.loss.tau > 0)) {
        throw new AdversaryConfigError(`${cfg.loss.tag} tau must be finite and > 0, got ${cfg.loss.tau}`);
      }
      if (!(Number.isFinite(cfg.loss.scaleWeight) && cfg.loss.scaleWeight >= 0)) {
        throw new AdversaryConfigError(
          `${cfg.loss.tag} scaleWeight must be finite and >= 0, got ${cfg.loss.scaleWeight}`
        );
      }
      if (!(Number.isFinite(cfg.loss.energyWeight) && cfg.loss.energyWeight >= 0)) {
        throw new AdversaryConfigError(
          `${cfg.loss.tag} energyWeight must be finite and >= 0, got ${cfg.loss.energyWeight}`
        );
      }
      if (!(Number.isFinite(cfg.loss.energyTarget) && cfg.loss.energyTarget > 0)) {
        throw new AdversaryConfigError(
          `${cfg.loss.tag} energyTarget must be finite and > 0, got ${cfg.loss.energyTarget}`
        );
      }
      break;
    default:
      assertNever(cfg.loss, "validate loss");
  }
  if (cfg.kind.tag === "wta") {
    if (!Number.isInteger(cfg.kind.k) || cfg.kind.k < 2) {
      throw new AdversaryConfigError(
        `wta requires an integer k >= 2 (k = 1 IS variant "single"), got ${cfg.kind.k}`
      );
    }
    const winnerDominanceLimit = (cfg.kind.k - 1) / cfg.kind.k;
    if (!(cfg.kind.relaxEps >= 0 && cfg.kind.relaxEps < winnerDominanceLimit)) {
      throw new AdversaryConfigError(
        `relaxEps must be in [0, (k-1)/k) = [0, ${winnerDominanceLimit}) ` +
          `so the winner remains individually dominant, got ${cfg.kind.relaxEps}`
      );
    }
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   6. The adversary
   ══════════════════════════════════════════════════════════════════════════ */

export class Adversary {
  readonly cfg: AdversaryConfig;
  readonly k: number;
  readonly dims: ObjectiveDims;

  /** The k predictor heads: u [B,du] -> vector guess + optional scale channels. */
  private readonly heads: tf.Sequential[];
  /** One Adam PER HEAD — see the comment in `trainStep`. */
  private readonly optimizers: tf.Optimizer[];
  /** Live predictor/discriminator step size. `cfg` remains the immutable
   * construction record; this value tracks UI changes. */
  private learningRateValue: number;
  /** Cumulative per-head win counts since construction / `resetWinCounts()`. */
  private readonly wins: number[];
  private readonly rnd: () => number;
  /** EMA of RMS‖y‖ over observed batches — the generator-reward normalizer. */
  private rmsState: RewardScale = { tag: "unseeded" };

  constructor(cfg: AdversaryConfig) {
    validate(cfg);
    this.cfg = cfg;
    this.k = headCount(cfg.kind);
    this.dims = objectiveDims(cfg.encoding, cfg.target, cfg.loss);
    this.rnd = mulberry32(cfg.seed);
    this.wins = new Array(this.k).fill(0);
    this.learningRateValue = cfg.learningRate;

    this.heads = [];
    this.optimizers = [];
    for (let j = 0; j < this.k; j++) {
      const net = tf.sequential();
      // Distinct SEEDED init per head. Symmetry breaking is not cosmetic here:
      // identical heads produce identical residuals, argMin routes every tie to
      // head 0 (tfjs breaks ties toward the lowest index), and the mixture is
      // dead on arrival. Seeding also keeps the whole thing reproducible.
      const seedOf = (layer: number) => cfg.seed * 7919 + j * 101 + layer;
      net.add(
        tf.layers.dense({
          units: cfg.hiddenUnits,
          activation: "selu",
          inputShape: [this.dims.du],
          kernelInitializer: tf.initializers.glorotNormal({ seed: seedOf(0) }),
        })
      );
      net.add(
        tf.layers.dense({
          units: cfg.featureDim,
          activation: "selu",
          kernelInitializer: tf.initializers.glorotNormal({ seed: seedOf(1) }),
        })
      );
      // Linear head — a raw signal is an unbounded real vector, do not squash it.
      net.add(
        tf.layers.dense({
          units: this.dims.out,
          kernelInitializer: tf.initializers.glorotNormal({ seed: seedOf(2) }),
        })
      );
      this.heads.push(net);
      this.optimizers.push(tf.train.adam(cfg.learningRate));
    }
  }

  /** Current predictor/discriminator learning rate. */
  get learningRate(): number {
    return this.learningRateValue;
  }

  /** Live tfjs fallback LR update.
   *
   * TensorFlow.js exposes Adam's learningRate only as a protected field, so
   * mutating it through an unsafe cast would couple us to an implementation
   * detail. Rebuilding each head's optimizer is the supported, leak-free
   * operation: predictor weights and win/RMS telemetry stay intact, while Adam
   * moments restart deliberately. The fused path keeps its GPU moments and
   * reads LR from a uniform instead.
   */
  setLearningRate(learningRate: number): void {
    if (!(Number.isFinite(learningRate) && learningRate > 0)) {
      throw new AdversaryConfigError(
        `learningRate must be finite and > 0, got ${learningRate}`
      );
    }
    if (learningRate === this.learningRateValue) return;
    for (let j = 0; j < this.optimizers.length; j++) {
      this.optimizers[j].dispose();
      this.optimizers[j] = tf.train.adam(learningRate);
    }
    this.learningRateValue = learningRate;
  }

  /* ─── sampling ─────────────────────────────────────────────────────── */

  /** Draw tuples for the configured explicit target source. */
  sampleTarget(batch: AdversaryTargetBatch): TupleSample {
    const n = batch.pos.shape[0];
    const idx = sampleIndices(n, this.cfg.batchTuples, this.dims.m, this.rnd);
    return this.encodeTarget(batch, idx);
  }

  /**
   * Encode an explicit tuple set for the configured target source.
   *
   * FORCE consumes raw F(x). POST-VELOCITY consumes normalized current and
   * next velocities and exposes current velocity in the point context [x,v].
   */
  encodeTarget(batch: AdversaryTargetBatch, idx: Int32Array): TupleSample {
    if (batch.tag !== this.cfg.target.tag) {
      throw new AdversaryConfigError(
        `target batch ${batch.tag} does not match configured target ${this.cfg.target.tag}`
      );
    }
    if (batch.tag === "force") {
      return this.encodeForceTarget(batch.pos, batch.force, idx);
    }
    if (this.cfg.encoding.tag !== "point") {
      throw new AdversaryConfigError(
        `post-velocity target is point-only, got ${this.cfg.encoding.tag}`
      );
    }
    const { pos, velocity, nextVelocity } = batch;
    for (const [name, value] of [
      ["pos", pos],
      ["velocity", velocity],
      ["nextVelocity", nextVelocity],
    ] as const) {
      if (
        value.rank !== 2 ||
        value.shape[1] !== 2 ||
        value.shape[0] !== pos.shape[0]
      ) {
        throw new AdversaryConfigError(
          `post-velocity ${name} must match pos [N,2], got [${value.shape}]`
        );
      }
    }
    if (idx.length % this.dims.m !== 0) {
      throw new AdversaryConfigError(
        `encodeTarget: idx length ${idx.length} is not a multiple of m=${this.dims.m}`
      );
    }
    const b = idx.length;
    const encoded = tf.tidy(() => {
      const ii = tf.tensor1d(idx, "int32");
      const x = pos.gather(ii, 0) as tf.Tensor2D;
      const v = velocity.gather(ii, 0) as tf.Tensor2D;
      const y = nextVelocity.gather(ii, 0) as tf.Tensor2D;
      return { u: tf.concat([x, v], 1) as tf.Tensor2D, y };
    });
    return { ...encoded, idx, b, m: 1 };
  }

  /**
   * STRICT GAME: draw tuples and encode a raw generator signal.
   *
   * @param pos  [N,2] NORMALIZED positions on the unit torus (coords in [0,1)).
   * @param signal [N,2] raw generator output `F(x)` at those positions.
   * @returns a {@link TupleSample}; caller must {@link disposeTupleSample} it.
   *
   * Incoming velocity, friction, clipping, border mode, and resets are
   * deliberately absent. Thus the point control is the realizable map x→F(x),
   * and relational ambiguity comes only from the declared quotient.
   */
  sampleSignal(pos: tf.Tensor2D, signal: tf.Tensor2D): TupleSample {
    if (this.cfg.target.tag !== "force") {
      throw new AdversaryConfigError(
        `sampleSignal is the force compatibility API; configured target is ${this.cfg.target.tag}`
      );
    }
    return this.sampleTarget({ tag: "force", pos, force: signal });
  }

  /**
   * STRICT GAME: encode an EXPLICIT tuple set from raw force/signal values.
   * Split out from {@link sampleSignal} because re-encoding the same tuples is
   * exactly what the
   * generator update needs (and what an invariance test needs): otherwise a
   * fresh random draw is confounded with the effect being measured.
   *
   * @param idx [B*m] particle indices, row-major per tuple.
   */
  encodeSignal(pos: tf.Tensor2D, signal: tf.Tensor2D, idx: Int32Array): TupleSample {
    if (this.cfg.target.tag !== "force") {
      throw new AdversaryConfigError(
        `encodeSignal is the force compatibility API; configured target is ${this.cfg.target.tag}`
      );
    }
    return this.encodeTarget({ tag: "force", pos, force: signal }, idx);
  }

  private encodeForceTarget(
    pos: tf.Tensor2D,
    signal: tf.Tensor2D,
    idx: Int32Array
  ): TupleSample {
    const m = this.dims.m;
    if (idx.length % m !== 0) {
      throw new AdversaryConfigError(
        `encodeSignal: idx length ${idx.length} is not a multiple of m=${m}`
      );
    }
    if (
      pos.rank !== 2 ||
      signal.rank !== 2 ||
      pos.shape[1] !== 2 ||
      signal.shape[1] !== 2 ||
      pos.shape[0] !== signal.shape[0]
    ) {
      throw new AdversaryConfigError(
        `encodeSignal: expected matching [N,2] position/signal tensors, got ` +
          `[${pos.shape}] and [${signal.shape}]`
      );
    }
    const b = idx.length / m;
    const encoded = tf.tidy(() => {
      const gather = (j: number) => {
        const slice = new Int32Array(b);
        for (let t = 0; t < b; t++) slice[t] = idx[t * m + j];
        return tf.tensor1d(slice, "int32");
      };
      let pair: [tf.Tensor2D, tf.Tensor2D];
      switch (this.cfg.encoding.tag) {
        case "point":
          pair = encodePointSignal(pos, signal, gather(0));
          break;
        case "pair":
        case "pair-rotation":
        case "pair-rotation-scale-raw":
        case "pair-rotation-scale-adjusted":
          pair = encodePairSignal(
            pos,
            signal,
            gather(0),
            gather(1),
            this.cfg.observerGeometry,
            this.cfg.encoding
          );
          break;
        case "tri":
          pair = encodeTri(
            pos,
            signal,
            gather(0),
            gather(1),
            gather(2),
            this.cfg.observerGeometry
          );
          break;
        case "quad-labelled":
          pair = encodeQuadLabelled(
            pos,
            signal,
            gather(0),
            gather(1),
            gather(2),
            gather(3),
            this.cfg.observerGeometry
          );
          break;
        default:
          return assertNever(this.cfg.encoding, "encodeSignal");
      }
      const [u, y] = pair;
      if (
        this.cfg.loss.tag !== "angle-relative-scale" &&
        this.cfg.loss.tag !== "angle-scale-hold"
      ) {
        return { u, y };
      }
      const memberIdx = Array.from({ length: m }, (_, j) => gather(j));
      const memberPos = memberIdx.map((i) => pos.gather(i, 0) as tf.Tensor2D);
      const memberSignal = memberIdx.map(
        (i) => signal.gather(i, 0) as tf.Tensor2D
      );
      const scale = relativeScaleDescriptor(
        this.cfg.encoding,
        memberPos,
        memberSignal,
        this.cfg.observerGeometry
      );
      const geometryActive = tupleActivity(this.cfg.encoding, y, u);
      const objectiveActivity = geometryActive.mul(scale.active) as tf.Tensor1D;
      return {
        u,
        y,
        relativeScale: scale.target,
        tupleEnergy: scale.energy,
        objectiveActivity,
      };
    });
    return { ...encoded, idx, b, m };
  }

  /**
   * LEGACY TRANSITION ADAPTER. Converts wrapped next positions to displacement
   * signals, then delegates to {@link encodeSignal}. Do not use this for the
   * strict adversarial control: next-state displacement can contain hidden
   * velocity, clipping, boundary, and reset state.
   */
  encodeTransition(pos: tf.Tensor2D, next: tf.Tensor2D, idx: Int32Array): TupleSample {
    const signal = tf.tidy(() =>
      observerDelta(next.sub(pos) as tf.Tensor2D, this.cfg.observerGeometry)
    );
    try {
      return this.encodeSignal(pos, signal, idx);
    } finally {
      signal.dispose();
    }
  }

  /** Draw tuples through the explicitly named legacy transition adapter. */
  sampleTransition(pos: tf.Tensor2D, next: tf.Tensor2D): TupleSample {
    const idx = sampleIndices(pos.shape[0], this.cfg.batchTuples, this.dims.m, this.rnd);
    return this.encodeTransition(pos, next, idx);
  }

  /**
   * @deprecated Use {@link sampleSignal} for the strict game, or explicitly use
   * {@link sampleTransition} when testing/visualizing wrapped transitions.
   */
  sample(pos: tf.Tensor2D, next: tf.Tensor2D): TupleSample {
    return this.sampleTransition(pos, next);
  }

  /**
   * @deprecated Use {@link encodeSignal} for the strict game, or explicitly use
   * {@link encodeTransition} for legacy transition data.
   */
  encode(pos: tf.Tensor2D, next: tf.Tensor2D, idx: Int32Array): TupleSample {
    return this.encodeTransition(pos, next, idx);
  }

  /* ─── prediction / residuals ───────────────────────────────────────── */

  /**
   * [k, B, dy] — each head's guess. Differentiable w.r.t. `u`.
   *
   * `apply`, NOT `predict`. `LayersModel.predict()` defaults to `batchSize: 32`
   * and silently slices the input into sub-batches, running the forward pass
   * ⌈B/32⌉ times and concatenating — 8× the work at our B = 256 and a large
   * constant even when it is "correct". `apply()` is the single-shot,
   * tape-recording forward. (The deleted `PredictorEnsemble` used `predict`
   * everywhere, including inside its optimizer closure.)
   */
  predictHeads(u: tf.Tensor2D): tf.Tensor3D {
    return tf.tidy(
      () =>
        tf.stack(
          this.heads.map((h) => h.apply(u) as tf.Tensor2D),
          0
        ) as tf.Tensor3D
    );
  }

  /**
   * [B, k] — per-head residual.
   *
   * Raw modes use softened Euclidean distance. The adjusted scale-quotient uses
   * an angular chord distance between exact unit target direction and normalized
   * predictor direction; inactive zero targets return exact zero.
   */
  residuals(s: TupleSample): tf.Tensor2D {
    return tf.tidy(() => {
      const per = this.heads.map((h) => {
        const pred = h.apply(s.u) as tf.Tensor2D;
        return objectiveResidualFor(this.cfg, this.dims, pred, s);
      });
      return tf.stack(per, 1) as tf.Tensor2D;
    });
  }

  /* ─── the two exported signals ─────────────────────────────────────── */

  /**
   * PER-SAMPLE STRICT PAYOFF: `[B]`, the same relaxed-WTA weighted residual
   * used by the discriminator.
   *
   * The discriminator minimizes `mean(payoff)` and the disagreeing generator
   * minimizes `-mean(payoff)`: one scalar game, opposite signs. For ε=0 this is
   * the winner residual; for ε>0 the loser terms deliberately participate on
   * BOTH sides rather than defining a separate generator objective.
   */
  payoff(s: TupleSample): tf.Tensor1D {
    return tf.tidy(() => {
      const resid = this.residuals(s);
      return resid.mul(residualWeights(this.cfg.kind, resid)).sum(1) as tf.Tensor1D;
    });
  }

  /** Compatibility/render spelling for {@link payoff}. */
  surprise(s: TupleSample): tf.Tensor1D {
    return this.payoff(s);
  }

  /** Pure nearest-head residual, exposed for quantization diagnostics only. */
  winnerResidual(s: TupleSample): tf.Tensor1D {
    return tf.tidy(() => this.residuals(s).min(1) as tf.Tensor1D);
  }

  /**
   * PER-SAMPLE PAYOFF DIVIDED BY THAT SAMPLE'S OWN SIGNAL MAGNITUDE:
   * `payoff / ‖y‖`, `[B]`, with inactive rows exactly zero.
   *
   * WHAT IT IS FOR. The raw residual is a DISTANCE, so it correlates with how
   * large the tuple's signal is: a large raw force in a predictable field can
   * out-score a small force in genuinely ambiguous flow, and a render mode
   * coloured by the raw value is partly a magnitude meter. Dividing by the tuple's
   * own ‖y‖ asks the different question — "what fraction of THIS signal was
   * unpredictable" — which is scale-invariant per tuple: multiply one tuple's
   * signal by c (with the heads following) and this number is unchanged.
   *
   * THIS IS NOT THE TRAINING NORMALIZER, and the two must not be conflated.
   * {@link generatorLoss} divides by a GLOBAL, tape-constant EMA of RMS‖y‖
   * (`RewardScale`, λ=0.02): one number for the whole batch, held fixed on the
   * tape so it is a stop-gradient. It stabilizes telemetry/tuning but, as the
   * generatorLoss docs state, does not remove the current radial gradient. The
   * quantity here is PER TUPLE, is never on a
   * tape, and never enters the reward — it exists only so the RENDER channel
   * stops reporting magnitude. Changing one has no effect on the other.
   *
   * A zero-magnitude tuple has no defined unit scale and returns exact zero,
   * matching the inactive-target convention rather than inventing a ratio.
   */
  payoffPerUnitSignal(s: TupleSample): tf.Tensor1D {
    return tf.tidy(() => {
      const mag = s.y.square().sum(1).sqrt() as tf.Tensor1D;
      const active = mag.greater(DIRECTION_ACTIVE_FLOOR).toFloat() as tf.Tensor1D;
      const safe = tf.where(active.cast("bool"), mag, tf.onesLike(mag)) as tf.Tensor1D;
      return this.surprise(s).div(safe).mul(active) as tf.Tensor1D;
    });
  }

  /** @deprecated Render compatibility alias for {@link payoffPerUnitSignal}. */
  surprisePerUnitMotion(s: TupleSample): tf.Tensor1D {
    return this.payoffPerUnitSignal(s);
  }

  /**
   * GENERATOR term: the NEGATIVE of the discriminator's shared payoff.
   *
   * Raw reward is in signal units, so its EMA divisor makes telemetry and
   * long-horizon tuning less scale-sensitive. It does NOT remove the current
   * tape's radial gradient and is not presented as doing so. `{tag:"unseeded"}`
   * (before the first trainStep) uses scale 1 deliberately rather than fabricating
   * a prior.
   *
   * Raw modes retain the legacy EMA magnitude normalizer as telemetry/backward
   * compatibility, but it is NOT a proof of scale invariance: it is constant on
   * one tape and therefore leaves an instantaneous radial gradient. The
   * `pair-rotation-scale-adjusted` mode is the strict fix: its target and
   * predictor are direction-normalized inside the differentiable graph, so
   * scaling an active raw signal by c>0 leaves both payoff value and radial
   * derivative unchanged/zero. Adjusted mode consequently uses scale 1 here.
   *
   * The generator MINIMIZES its loss, so the shared payoff enters negated.
   * Add this (weighted) into the generator's `computeLoss`. The field is then
   * pushed toward dynamics that stay unpredictable even when the adversary is
   * allowed K guesses — a quantization-resistance objective, not a
   * variance-chasing one.
   *
   * The predictor weights are frozen here NOT by a stop-gradient but structurally:
   * the generator's optimizer owns only the field's variables, so nothing in this
   * graph can update a head.
   */
  generatorLoss(s: TupleSample): tf.Scalar {
    // The normalizer enters as a plain JS number — a CONSTANT on the tape by
    // construction, i.e. exactly the stop-gradient an EMA normalizer needs.
    const scale =
      this.cfg.loss.tag !== "raw-vector" || this.rmsState.tag === "unseeded"
        ? 1
        : Math.max(this.rmsState.rms, RMS_FLOOR);
    return tf.tidy(() => {
      if (this.cfg.loss.tag !== "angle-scale-hold") {
        return this.payoff(s)
          .mean()
          .div(scale)
          .neg()
          .add(energyAnchorLoss(s, this.cfg.loss))
          .asScalar();
      }
      // General-sum hold game: winner weights are selected by D's combined
      // angle+scale residual, but G opposes angle and cooperates on scale.
      const terms = this.heads.map((h) =>
        objectiveTermsFor(this.cfg, this.dims, h.apply(s.u) as tf.Tensor2D, s)
      );
      const d = tf.stack(terms.map((t) => t.discriminator), 1) as tf.Tensor2D;
      const weights = residualWeights(this.cfg.kind, d);
      const gTerms = tf.stack(
        terms.map((t) => generatorObjectiveTerm(this.cfg.loss, t)),
        1
      ) as tf.Tensor2D;
      return gTerms
        .mul(weights)
        .sum(1)
        .mean()
        .add(energyAnchorLoss(s, this.cfg.loss))
        .asScalar();
    });
  }

  /** Reward-normalizer telemetry — the named state, not a nullable number. */
  rewardScaleState(): RewardScale {
    return this.rmsState;
  }

  /* ─── discriminator update ─────────────────────────────────────────── */

  /**
   * DISCRIMINATOR update. Fit the heads to this batch of (u, y).
   *
   * `u` and `y` are DATA here — the tape only reaches predictor weights, so no
   * gradient can leak back into the generator field.
   *
   * ADVERSARY PARAMS ARE STRICTLY SEPARATE FROM THE GENERATOR'S. Each head is
   * stepped by its OWN Adam restricted to its OWN `varList`. Two reasons, both
   * load-bearing:
   *   1. it makes the generator/discriminator parameter split structural rather
   *      than a consequence of accidental tape pruning, and
   *   2. per-head optimizer state is what keeps the heads decorrelated — shared
   *      Adam moments would couple them and undo the quantization.
   * (This is the one thing the deleted `PredictorEnsemble` got right; the
   * approach and its rationale are preserved verbatim in spirit.)
   *
   * The relaxed-WTA weights are computed ONCE from the pre-update residuals and
   * are constants, so minimizing `mean(w_j · d_j)` per head is exactly
   * equivalent to minimizing the full weighted objective w.r.t. that head.
  */
  trainStep(s: TupleSample): TrainReport {
    const [wCols, winMeta] = tf.tidy(() => {
      const resid = this.residuals(s);
      const w = residualWeights(this.cfg.kind, resid);
      const argmin = tf.argMin(resid, 1).toFloat() as tf.Tensor1D;
      const active = objectiveActivityFor(this.cfg, s);
      // Pack winner+activity into ONE readback. Invalid rows still have a
      // mathematical argMin tie at head 0, but active=0 excludes that sentinel
      // from both this report and cumulative collapse telemetry.
      const meta = tf.stack([argmin, active], 1) as tf.Tensor2D;
      return [tf.unstack(w, 1) as tf.Tensor1D[], meta];
    });

    // Histogram on the CPU rather than via oneHot: tf.oneHot rejects depth < 2,
    // which would force a k === 1 special case into what is otherwise one path.
    // Only active rows own a winner (same rule as fused targetActive).
    const winCounts = new Array(this.k).fill(0);
    const meta = winMeta.dataSync();
    winMeta.dispose();
    for (let t = 0; t < s.b; t++) {
      if (meta[2 * t + 1] > 0.5) winCounts[meta[2 * t]]++;
    }
    for (let j = 0; j < this.k; j++) this.wins[j] += winCounts[j];

    const costs: tf.Scalar[] = [];
    for (let j = 0; j < this.k; j++) {
      const net = this.heads[j];
      const varList = net.trainableWeights.map((wv) => (wv as unknown as { val: tf.Variable }).val);
      costs.push(
        this.optimizers[j].minimize(
          () =>
            tf.tidy(() => {
              const pred = net.apply(s.u) as tf.Tensor2D;
              const d = objectiveResidualFor(this.cfg, this.dims, pred, s);
              return d.mul(wCols[j]).mean().asScalar();
            }),
          true,
          varList
        ) as tf.Scalar
      );
    }
    for (const c of wCols) c.dispose();

    // ONE readback for all k costs + the batch RMS‖y‖. The per-head
    // `cost.dataSync()` this replaces was k separate full-pipeline stalls per
    // frame — on the webgpu backend each sync read stalls the queue, and at
    // k=8 that was the dominant chunk of the HUD's `learn` time.
    const packed = tf.tidy(() =>
      tf.stack([...costs, tf.sqrt(s.y.square().sum(1).mean()) as tf.Scalar])
    );
    const vals = packed.dataSync();
    packed.dispose();
    for (const c of costs) c.dispose();

    let total = 0;
    for (let j = 0; j < this.k; j++) total += vals[j];

    // Reward-normalizer update (data side, no tape). First observation SEEDS
    // the EMA outright — blending from a fabricated prior would be a silent
    // fallback (see RewardScale).
    const batchRms = vals[this.k];
    this.rmsState =
      this.rmsState.tag === "unseeded"
        ? { tag: "seeded", rms: batchRms }
        : {
            tag: "seeded",
            rms: this.rmsState.rms + RMS_EMA_LAMBDA * (batchRms - this.rmsState.rms),
          };

    return { loss: total, winCounts };
  }

  /* ─── collapse detection ───────────────────────────────────────────── */

  /** Cumulative per-head win counts and their fractions. */
  winStats(): { counts: readonly number[]; fractions: readonly number[]; total: number } {
    const total = this.wins.reduce((a, x) => a + x, 0);
    const denom = total === 0 ? 1 : total;
    return { counts: this.wins.slice(), fractions: this.wins.map((c) => c / denom), total };
  }

  /**
   * HEAD-COLLAPSE TRIPWIRE. True when any head has won less than `floor` of the
   * tuples seen so far — i.e. the mixture has silently degraded and the effective
   * K is smaller than the configured K. A uniform mixture gives 1/k per head, so
   * a sensible floor is a fraction of that (default: one tenth of uniform).
   * Meaningless before any training, hence the `total === 0` guard.
   */
  collapsed(floor: number = 0.1 / this.k): boolean {
    const { fractions, total } = this.winStats();
    if (total === 0) return false;
    return fractions.some((f) => f < floor);
  }

  resetWinCounts(): void {
    this.wins.fill(0);
  }

  /**
   * Pairwise spread of the head predictions over a probe batch of contexts —
   * the collapse DISAMBIGUATOR (see {@link HeadSpread} for how to read it
   * against the win histogram). δ: dispatch on kind only; the work is in
   * {@link spreadOverHeads}.
   *
   * Cheap by design: one forward pass per head over the probe batch and ONE
   * `dataSync` for both aggregates. Intended cadence is ~1 Hz from the HUD,
   * with `u` taken from the most recent {@link TupleSample} (`s.u`) — the
   * spread is a property of the heads, so any recent batch of real contexts
   * serves as a probe.
   */
  headSpread(u: tf.Tensor2D): HeadSpread {
    switch (this.cfg.kind.tag) {
      case "single":
        return { tag: "single-head" };
      case "wta":
        return this.spreadOverHeads(u);
      default:
        return assertNever(this.cfg.kind, "headSpread");
    }
  }

  /** HANDLER — `wta` spread. Per-context pairwise distances, aggregated. */
  private spreadOverHeads(u: tf.Tensor2D): HeadSpread {
    const packed = tf.tidy(() => {
      const preds = this.heads.map((h) => h.apply(u) as tf.Tensor2D);
      const dists: tf.Tensor1D[] = [];
      for (let i = 0; i < this.k; i++) {
        for (let j = i + 1; j < this.k; j++) {
          const diff = preds[i].sub(preds[j]) as tf.Tensor2D;
          dists.push(diff.square().sum(1).sqrt() as tf.Tensor1D); // [B]
        }
      }
      const D = tf.stack(dists, 1) as tf.Tensor2D; // [B, pairs]
      // mean-over-pairs then mean-over-contexts == global mean; min is per-context.
      return tf.stack([D.mean() as tf.Scalar, D.min(1).mean() as tf.Scalar]);
    });
    const v = packed.dataSync();
    packed.dispose();
    return { tag: "spread", meanPair: v[0], minPair: v[1], pairs: (this.k * (this.k - 1)) / 2 };
  }

  /* ─── cleanup ──────────────────────────────────────────────────────── */

  /** Release every head's weights and every optimizer's state. */
  dispose(): void {
    for (const net of this.heads) net.dispose();
    for (const opt of this.optimizers) opt.dispose();
  }
}

/** Free-function cleanup, for symmetry with the rest of the core API. */
export function disposeAdversary(a: Adversary): void {
  a.dispose();
}

/* ══════════════════════════════════════════════════════════════════════════
   7. Mapping per-tuple surprise back onto particles (for render modes)
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * Scatter per-tuple surprise onto the particles that formed each tuple.
 *
 * Every member of a tuple receives the SAME value: the surprise is a property of
 * the tuple's joint configuration, and splitting it between members would invent
 * an attribution the encoding does not contain.
 *
 * @param s        the sample the surprise was computed from.
 * @param surprise [B] per-tuple values (i.e. `await adversary.surprise(s).data()`).
 * @param n        particle count.
 */
export function scatterSurprise(s: TupleSample, surprise: Float32Array, n: number): ScatteredSurprise {
  if (surprise.length !== s.b) {
    throw new AdversaryConfigError(`scatterSurprise: expected ${s.b} values, got ${surprise.length}`);
  }
  const sum = new Float32Array(n);
  const hits = new Int32Array(n);
  for (let t = 0; t < s.b; t++) {
    const v = surprise[t];
    for (let j = 0; j < s.m; j++) {
      const p = s.idx[t * s.m + j];
      sum[p] += v;
      hits[p] += 1;
    }
  }
  return { sum, hits, n };
}
