/**
 * WTA — the relaxed winner-take-all SPEC, in one place, for every backend.
 * ============================================================================
 *
 * ## Why this module exists
 *
 * The relaxed-WTA *reduction* is implemented three times, and that is correct:
 *
 *   - tfjs reference  — `weightsWta`, src/core/gan/adversary.ts, `oneHot(argMin)`
 *     over a `Tensor2D [B,k]`;
 *   - scalar AD IR    — `relaxedWtaWeights`, src/render/webgpu/ad/losses.ts,
 *     winner built as a product of `gt` nodes so its vjp is structurally zero;
 *   - fused WGSL      — src/render/webgpu/adversary_wgsl.ts, a runtime loop over
 *     a compile-time `k` with `select(loser, winner, j == win)`.
 *
 * Tensors, IR nodes and shader strings are genuinely different representations;
 * merging them would cost kernel time and buy nothing, and they are already
 * cross-verified (tools/ad_wta_test.ts §2 gates tfjs ≡ IR, tools/train_wta_test.ts
 * §2 gates kernel ≡ IR on real Metal).
 *
 * What was duplicated is the **spec**: the config type, the bound on ε, the two
 * weight scalars, and the tie rule. Three files independently derived
 *
 *     loser  = ε / (K − 1)
 *     winner = 1 − ε
 *     w_j    = loser + (j == first_argmin ? winner − loser : 0)
 *     payoff = Σ_j w_j · d_j
 *
 * That is small, pure, and belongs here. The reductions stay where they are.
 *
 * ## The `k === 1` division trap — the reason the arithmetic moved
 *
 * `loser = ε/(K−1)` divides by zero at K = 1. Each backend was safe only
 * BY VALIDATION ORDER: `Adversary`'s constructor rejects `k < 2` before
 * `weightsWta` ever runs, and `adversary_wgsl` guards with `k >= 2 ? … : 0`
 * at each of its two literal sites. Lift that arithmetic out standalone and it
 * silently yields `Infinity` — a weight of Infinity on every loser, which is
 * not a crash, it is a wrong loss.
 *
 * So the guard here is STRUCTURAL, not conventional: `k = 1` is not a `wta`
 * with a small K, it is the separate `single` variant, whose handler returns
 * `{ winner: 1, loser: 0 }` as CONSTANTS and never touches the division. The
 * only path that can reach `ε/(k−1)` is the `wta` handler, and the dispatcher
 * runs {@link validateGuessKind} — which requires an integer `k >= 2` — before
 * it can select that handler. There is no argument you can pass to
 * {@link wtaScalars} that divides by zero.
 *
 * ## The tie rule is an invariant, not a convention
 *
 * When two heads produce the exact same residual, the winner is the one with
 * the **LOWEST head index**. Every backend must honor this or the three
 * implementations disagree on ties:
 *
 *   - tfjs `argMin` returns the first minimum, so `oneHot(argMin(resid, 1))`
 *     already routes ties low;
 *   - the IR's `g.min` reverse rule sends the gradient to its FIRST argument on
 *     a tie, and `relaxedWtaWeights` builds the indicator as
 *     `Π_{i<j}[resid_i > resid_j] · Π_{i>j}[resid_j ≤ resid_i]` — strict on the
 *     left, non-strict on the right, which is exactly "first argmin";
 *   - the WGSL loop uses `if (resid[j] < best)` ascending from `j = 0`, so a
 *     later equal residual does not displace the incumbent.
 *
 * Exact float ties are not hypothetical here: an inactive/masked target
 * multiplies every head's residual by 0, and then ALL k residuals tie.
 *
 * ## Winner dominance is what bounds ε
 *
 * Relaxed WTA (Rupprecht et al.) exists because plain WTA collapses: a head
 * that never wins gets exactly zero gradient, so it never moves, so it keeps
 * never winning, and K silently degrades to 1 while the loss looks fine. Giving
 * the K−1 losers a share of ε keeps every head alive. But the relaxation is
 * only meaningful while the winner still individually dominates each loser:
 *
 *     1 − ε > ε/(K−1)   ⟺   ε < (K−1)/K
 *
 * At ε = (K−1)/K all K weights equal 1/K and the objective is a plain mean —
 * no selection at all; above it the "winner" is the LEAST weighted head and the
 * term is actively inverted. Hence the closed-open bound `0 <= ε < (K−1)/K`
 * ({@link winnerDominanceLimit}), rejected once, here. ε = 0 is permitted: hard
 * WTA is a real (collapsing) variant that the tests exercise deliberately.
 *
 * ## Structural notes
 *
 *  - Zero dependencies, no tfjs, no GPU. Importable from a shader emitter, a
 *    tensor reference implementation, and a pure-JS oracle alike.
 *  - The scalars are the TOTAL weights: `winner` is the winning head's whole
 *    weight (1 − ε), not the bonus over the loser share. Backends that want the
 *    additive form (`loser + win_j · bonus`) compute `winner − loser`
 *    themselves; that subtraction is exact in float and keeps this interface
 *    from having two nearly-identical fields to confuse.
 *  - The weights are CONSTANTS with respect to the gradient in all three
 *    backends — selection is off-tape. Nothing here is differentiable, and
 *    nothing here should become differentiable.
 */

/* ══════════════════════════════════════════════════════════════════════════
   0. Canonical domain types.  D = ⊕ᵢ Dᵢ
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * How many guesses the predictor is allowed, and how they are weighted.
 *
 * This is the type formerly spelled `AdversaryKind` (src/core/gan/adversary.ts,
 * which re-exports the old name). It moved because the pixel critics consume it
 * too, and "Adversary" is the wrong noun for a head-count policy.
 */
export type GuessKind =
  /** One predictor, one guess. The frozen-generator realizability control. */
  | { readonly tag: "single" }
  /**
   * K guesses, relaxed winner-take-all. Winner dominance requires
   * `0 <= relaxEps < (k - 1) / k`; larger values give each loser more weight
   * than the nominal winner and are rejected.
   */
  | { readonly tag: "wta"; readonly k: number; readonly relaxEps: number };

/**
 * The two TOTAL per-head weights of a relaxed-WTA payoff.
 *
 *     w_j = loser + (j == first_argmin ? winner - loser : 0)
 *
 * so `Σ_j w_j = winner + (k − 1)·loser = 1` for every valid {@link GuessKind}.
 */
export interface WtaScalars {
  /** Weight of the single winning head — `1 − ε` (`1` for `single`). */
  readonly winner: number;
  /** Weight of EACH losing head — `ε/(k−1)` (`0` for `single`). */
  readonly loser: number;
}

/** Thrown when a {@link GuessKind} is not canonical. */
export class GuessKindError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GuessKindError";
  }
}

function assertNever(x: never, what: string): never {
  throw new GuessKindError(`${what}: unhandled variant ${JSON.stringify(x)}`);
}

/* ══════════════════════════════════════════════════════════════════════════
   1. Validation (κ) — the ONE place the ε bound and the k ≥ 2 rule live.
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * The exclusive upper bound on ε for K guesses: `(K−1)/K`. At or above it the
 * winner no longer individually dominates a loser. Exported so a UI slider and
 * an error message can quote the same number.
 */
export function winnerDominanceLimit(k: number): number {
  return (k - 1) / k;
}

/** HANDLER — `single`. A head count of one has nothing left to constrain. */
function validateSingle(): void {}

/** HANDLER — `wta`. Integer `k >= 2`, and ε strictly under winner dominance. */
function validateWta(k: number, relaxEps: number): void {
  if (!Number.isInteger(k) || k < 2) {
    throw new GuessKindError(
      `wta requires an integer k >= 2 (k = 1 IS variant "single"), got ${k}`
    );
  }
  const limit = winnerDominanceLimit(k);
  if (!(relaxEps >= 0 && relaxEps < limit)) {
    throw new GuessKindError(
      `relaxEps must be in [0, (k-1)/k) = [0, ${limit}) ` +
        `so the winner remains individually dominant, got ${relaxEps}`
    );
  }
}

/** δ: GuessKind → validator. Throws {@link GuessKindError}; returns void. */
export function validateGuessKind(kind: GuessKind): void {
  switch (kind.tag) {
    case "single":
      return validateSingle();
    case "wta":
      return validateWta(kind.k, kind.relaxEps);
    default:
      return assertNever(kind, "validateGuessKind");
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   2. Handlers — one clean path each, no structural branching.
   ══════════════════════════════════════════════════════════════════════════ */

/** δ: GuessKind → head count. */
export function headCount(kind: GuessKind): number {
  switch (kind.tag) {
    case "single":
      return 1;
    case "wta":
      return kind.k;
    default:
      return assertNever(kind, "headCount");
  }
}

/**
 * HANDLER — `single`. Constants. NOTE what is absent: no `ε/(k−1)`, no `k` at
 * all. This handler is the whole reason `k === 1` cannot produce `Infinity`.
 */
function scalarsSingle(): WtaScalars {
  return { winner: 1, loser: 0 };
}

/**
 * HANDLER — `wta`, relaxed. Winner takes `1 − ε`; the `k − 1` losers SHARE ε.
 * Reached only after {@link validateWta} has established `k >= 2`, so the
 * division is total.
 */
function scalarsWta(k: number, relaxEps: number): WtaScalars {
  return { winner: 1 - relaxEps, loser: relaxEps / (k - 1) };
}

/**
 * δ: GuessKind → the two payoff scalars.
 *
 * Validates FIRST, deliberately: the validation is what makes `k >= 2` a
 * precondition of the only handler that divides, rather than an invariant some
 * caller elsewhere is trusted to have checked. It is four comparisons on a path
 * that then does tensor work or emits a shader — the cost is not measurable.
 */
export function wtaScalars(kind: GuessKind): WtaScalars {
  validateGuessKind(kind);
  switch (kind.tag) {
    case "single":
      return scalarsSingle();
    case "wta":
      return scalarsWta(kind.k, kind.relaxEps);
    default:
      return assertNever(kind, "wtaScalars");
  }
}
