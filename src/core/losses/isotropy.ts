import * as tf from "@tensorflow/tfjs";

/**
 * Scale-invariant anisotropy penalty on a batch of force vectors.
 *
 * Builds the 2x2 force covariance C = (Fᵀ·F) / N over the particle batch and
 * penalises how far it is from isotropic. A field is isotropic when its x- and
 * y-energy are equal (C00 == C11) and the cross term vanishes (C01 == 0).
 *
 * The penalty is normalised by the total energy (trace)² so it is invariant to
 * the overall force scale — the model cannot cheat by shrinking every force to
 * make the raw covariance small.
 *
 *   loss = ((C00 - C11)² + 4·C01²) / (C00 + C11 + eps)²
 *
 * loss ∈ [0, 1]: 0 for a perfectly isotropic field, → 1 for a fully degenerate
 * (rank-1 / one-directional) field.
 *
 * @param force [N, 2] force vectors for the particle batch.
 * @returns Differentiable scalar anisotropy penalty.
 */
export function isotropyLoss(force: tf.Tensor2D): tf.Scalar {
  return tf.tidy(() => {
    const eps = 1e-6;
    const n = force.shape[0];

    // C = Fᵀ·F / N  (2x2)
    const cov = force.transpose().matMul(force).div(tf.scalar(n)) as tf.Tensor2D;

    const c00 = cov.slice([0, 0], [1, 1]);
    const c01 = cov.slice([0, 1], [1, 1]);
    const c11 = cov.slice([1, 1], [1, 1]);

    const diff = c00.sub(c11).square();
    const cross = c01.square().mul(4);
    const trace = c00.add(c11).add(eps);

    return diff.add(cross).div(trace.square()).asScalar();
  });
}

/**
 * DIRECTIONAL ORDER PARAMETERS of a force batch — the anti-collapse pressure
 * for adversarial pieces.
 *
 * {@link isotropyLoss} is an ENERGY statistic: the covariance is weighted by
 * ‖F‖², so a handful of large forces decide the verdict and a field whose
 * DIRECTIONS have all aligned at small magnitude scores as isotropic. What the
 * adversary games collapse into is a direction fact, so it is measured on
 * directions:
 *
 *     u_i = F_i / sqrt(‖F_i‖² + τ²)            soft unit vector, exact at 0
 *     R₁  = ‖mean_i u_i‖                       POLAR order  — one global flow
 *     R₂  = ‖mean_i (u_x²−u_y², 2·u_x·u_y)‖    NEMATIC order — ± aligned streaks
 *
 * Both are 0 for an isotropic direction field and 1 for a fully ordered one.
 * The loss is `polarWeight·R₁² + nematicWeight·R₂²`.
 *
 * WHY BOTH, MEASURED (tools/collapse_probe.ts, pair preset, 250 steps): with
 * the polar term alone at λ=0.05 the generator drove R₁ to 0.004 and R₂ to
 * 0.95 — it simply split the domain into counter-streaming ±F₀ sheets, which
 * reads on screen as exactly the same laminar streaks. The nematic term is not
 * a refinement; without it the penalty has a free escape.
 *
 * The double-angle vector is deliberately left UNNORMALIZED (it carries |u|²,
 * which is ≤ 1 and → 0 as ‖F‖ → τ). Dividing by |u|² would restore an exact
 * unit circle at the cost of a 0/0 at F = 0, and the softened magnitude is the
 * honest statement that a near-zero force has no reliable direction.
 *
 * Batch-coupled: one mean over the batch, then one broadcast on the backward.
 * That is the same fold shape as the isotropy covariance (see `isotropyFold`,
 * src/render/webgpu/ad/losses.ts), which is how it becomes fusable.
 *
 * @param force [N,2] RAW field output F(x) — not the physically scaled force.
 * @param tau   Direction softener. Use the objective's own soft-angle τ.
 */
export function directionOrderLoss(
  force: tf.Tensor2D,
  tau: number,
  polarWeight: number,
  nematicWeight: number
): tf.Scalar {
  checkedTau(tau, "directionOrderLoss");
  return tf.tidy(() => {
    const { ux, uy } = softUnit(force, tau);
    const polar = ux.mean().square().add(uy.mean().square());
    const c2 = ux.square().sub(uy.square()) as tf.Tensor1D;
    const s2 = ux.mul(uy).mul(2) as tf.Tensor1D;
    const nematic = c2.mean().square().add(s2.mean().square());
    return polar.mul(polarWeight).add(nematic.mul(nematicWeight)).asScalar();
  });
}

/**
 * The two ORDER PARAMETERS behind {@link directionOrderLoss}, as plain numbers
 * for telemetry: `R₁ = ‖mean u‖` (polar — one global flow) and
 * `R₂ = ‖mean(uₓ²−u_y², 2uₓu_y)‖` (nematic — ± aligned streaks). Both are 0 for
 * an isotropic direction field and 1 for a fully ordered one, and
 * `loss = polar·R₁² + nematic·R₂²` exactly, so the HUD number and the term the
 * generator pays are the SAME statistic rather than two similar ones.
 *
 * The fused trainer reports the identical pair from the moments its pass-A
 * reduction already accumulates (AdvStats.directionOrder); this is the tfjs
 * path's twin so both trainers feed one HUD field.
 *
 * Costs one dataSync — call it on the telemetry path, not inside a tape.
 */
export function directionOrderParameters(
  force: tf.Tensor2D,
  tau: number
): { r1: number; r2: number } {
  checkedTau(tau, "directionOrderParameters");
  const packed = tf.tidy(() => {
    const { ux, uy } = softUnit(force, tau);
    return tf.stack([
      ux.mean(),
      uy.mean(),
      ux.square().sub(uy.square()).mean(),
      ux.mul(uy).mul(2).mean(),
    ]);
  });
  const m = packed.dataSync();
  packed.dispose();
  return { r1: Math.hypot(m[0], m[1]), r2: Math.hypot(m[2], m[3]) };
}

function checkedTau(tau: number, where: string): void {
  if (!(Number.isFinite(tau) && tau > 0)) {
    throw new Error(`${where}: tau must be finite and > 0, got ${tau}`);
  }
}

/** u = F / sqrt(‖F‖² + τ²), split into components. τ² IS the radicand floor. */
function softUnit(
  force: tf.Tensor2D,
  tau: number
): { ux: tf.Tensor1D; uy: tf.Tensor1D } {
  const mag = force
    .square()
    .sum(1)
    .add(tau * tau)
    .sqrt()
    .reshape([-1, 1]) as tf.Tensor2D;
  const u = force.div(mag) as tf.Tensor2D;
  return {
    ux: u.slice([0, 0], [-1, 1]).reshape([-1]) as tf.Tensor1D,
    uy: u.slice([0, 1], [-1, 1]).reshape([-1]) as tf.Tensor1D,
  };
}
