/**
 * `window.__nffHealth` — the structured, ~1 Hz health snapshot.
 *
 * WHY THIS EXISTS. Every previous headless gate in this repo read the HUD as
 * TEXT and regexed numbers back out of it. That is how the 2026-08-17 soak
 * flake happened (`agent_notes/2026-08-17_soak_flake_attribution.md`): a
 * formatter change, a locale, an `toExponential(2)` rounding a value across a
 * threshold, and a gate passes or fails for reasons that have nothing to do
 * with the artwork. Everything here is an EXACT float on the wire, and
 * `tools/health_audit.mjs` is forbidden from parsing the HUD.
 *
 * WHAT IT IS NOT. Nothing in this snapshot is optimized. On adversarial and
 * pixel-critic pieces these numbers are OBSERVED — they never enter `extGrads`,
 * a loss, or an optimizer. A healthy adversarial game should raise the
 * structure metrics indirectly; that is the entire experiment, and a metric
 * wired into the objective would stop being evidence of it. (The one piece that
 * optimizes structure DIRECTLY is "Neural Field · Max Structure", and it says
 * so in its own comment.)
 *
 * ABSENT ≠ ZERO. Every field that can be genuinely unavailable is `null`, never
 * a plausible-looking 0: an unmeasured R₁ reported as 0 reads as "perfectly
 * isotropic", which is the single most flattering lie this instrument could
 * tell about a collapsed field.
 */

import type { FieldMetrics } from "./render/webgpu/field_probe";

/** Which trainer produced the numbers. Shipped pieces are all `fused`. */
export type TrainerKind = "fused" | "tfjs";

/**
 * Adversary block. `trainer` selects which fields are present, so consumers
 * dispatch on it instead of probing for undefined.
 */
export type AdvHealth =
  | {
      readonly trainer: "fused";
      /** Mean shared relaxed-WTA payoff, AFTER the isFiniteF gate. */
      readonly payoff: number;
      /**
       * The same payoff BEFORE the gate. `payoffUngated !== payoff` is the
       * fused adversary's ONLY nonfinite canary: some tuple's payoff went
       * nonfinite and the gate silently replaced it with 0. See
       * `AdvStats.payoffUngated`.
       */
      readonly payoffUngated: number;
      /** Historical alias of `payoff` — the same zero-sum scalar. Kept because
       *  the HUD and every existing probe call it "surprise". */
      readonly surprise: number;
      /** Polar order ‖mean u‖ over the training batch; `null` = UNMEASURED
       *  (no anti-collapse pressure compiled, so the moments are never
       *  reduced). Not 0 — 0 means "perfectly isotropic". */
      readonly r1: number | null;
      /** Nematic order; same null contract as `r1`. */
      readonly r2: number | null;
      /** Batch RMS‖y‖ of the encoded target. */
      readonly batchRms: number;
      /** ACTIVE tuples each head won this batch (length k). */
      readonly heads: readonly number[];
    }
  | {
      readonly trainer: "tfjs";
      readonly payoff: number;
      readonly surprise: number;
      readonly r1: number | null;
      readonly r2: number | null;
      /** Head win FRACTIONS (the tfjs oracle reports no raw counts). */
      readonly heads: readonly number[];
    };

/** Pixel critic block, present only while a fused pixel critic is running. */
export interface PixelHealth {
  /** Critic (discriminator) loss. */
  readonly dLoss: number;
  /** Generator-side loss the critic pays the field. */
  readonly gLoss: number;
  /** L2 norm of the critic's external field-gradient buffer, or `null` when the
   *  1 Hz readback has not landed yet. */
  readonly extGradNorm: number | null;
}

/**
 * Field-grid block. Definitions are `tools/collapse_probe.ts::diagnostics`
 * verbatim; see `src/render/webgpu/field_probe.ts` for the measurement.
 *
 * WHAT EACH ONE DETECTS (thresholds measured on this repo — collapse note §5,
 * pressure note §6):
 *   ac  < ~1e-4 sustained → DEAD FIELD. The spatially varying mode is gone; the
 *                           cloud is being pushed by one global constant.
 *   dc/ac > ~3            → the laminar end state (measured 3.7 at full
 *                           collapse, 0.25 with the pressure working).
 *   satFrac > ~0.3        → FROZEN GENERATOR: both tanh components pinned past
 *                           ±0.9 over a third of the domain, so the field
 *                           cannot move even where the gradient wants it to
 *                           (measured 0.46 on the collapsed point observer).
 *   okuboWeiss < 0        → vortex-dominated (the good look). > 0 is
 *                           strain/shear-dominated.
 *   r1 > ~0.5 sustained   → LAMINAR COLLAPSE: the field's directions have
 *                           converged on one global heading. Measured on the
 *                           default piece: 0.99 with the anti-collapse
 *                           pressure off, 0.01–0.10 with it on.
 *   r2 > ~0.5 with low r1 → the ± escape from a polar-only penalty: two
 *                           counter-streaming sheets, which reads on screen as
 *                           the same laminar streaks (measured 0.95).
 *
 * `r1`/`r2` here are the GRID measurement and are present on EVERY piece,
 * adversary or not — unlike `AdvHealth.r1`, which is the training-BATCH twin
 * and is `null` unless anti-collapse pressure is compiled. Same statistic
 * (same τ), two sample populations; they are not interchangeable.
 */
export type FieldHealthBlock = FieldMetrics;

export interface HealthSnapshot {
  /** Gallery piece NAME — the identity convention this repo commits to. */
  readonly piece: string;
  readonly frame: number;
  /** Seconds since the loop started. */
  readonly t: number;
  readonly fps: number;
  /** EMA of the train-step wall time in ms (fused: encode cost; tfjs: real). */
  readonly learnMs: number;
  /** tfjs backend name — `webgpu` on every shipped path. */
  readonly backend: string;
  readonly trainer: TrainerKind;
  /** `null` ⇔ this piece has no adversary. */
  readonly adv: AdvHealth | null;
  /** `null` ⇔ the ~1 Hz field probe has not landed a sample yet. */
  readonly field: FieldHealthBlock | null;
  /** `null` ⇔ this piece has no pixel critic. */
  readonly pixel: PixelHealth | null;
}

/** Publication cadence. Deliberately ~1 Hz: the field probe is 32²×5 field
 *  evaluations plus a readback, which has no business in a 16 ms frame. */
export const HEALTH_PERIOD_MS = 1000;

/** Grid resolution for the live field probe. 32² (1024 sites) versus
 *  collapse_probe's 64²: same statistic, 4× cheaper, and the AC/DC split is a
 *  domain average that converges long before 1024 samples. */
export const HEALTH_GRID_N = 32;

/** The global the headless auditor reads. Typed here so both sides agree. */
export interface HealthWindow {
  __nffHealth?: HealthSnapshot;
}

/** Euclidean norm, NaN-transparent (a NaN anywhere makes the result NaN, which
 *  is exactly what a nonfinite-gradient audit needs to see). */
export function l2Norm(values: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i] * values[i];
  return Math.sqrt(sum);
}
