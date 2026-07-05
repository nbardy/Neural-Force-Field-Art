import * as tf from "@tensorflow/tfjs";

/**
 * Local finite-time Lyapunov proxy — rewards sensitive dependence.
 *
 * Chaotic flows amplify tiny differences in initial conditions: two nearby
 * particles diverge exponentially. We approximate the local largest Lyapunov
 * exponent by jittering each position by a small deterministic offset `delta`
 * (‖delta‖ = eps) and measuring how much the force changes:
 *
 *   separation = ‖F(pos + delta) − F(pos)‖ / (eps + 1e-9)
 *   loss       = −mean( log(separation + 1e-6) )
 *
 * `separation` is a one-step, finite-difference stand-in for the directional
 * derivative magnitude ‖∂F/∂delta‖ — i.e. a *local finite-time* Lyapunov proxy,
 * NOT a true asymptotic exponent (no trajectory integration, single step).
 *
 * We return the NEGATIVE log so that MINIMISING this loss MAXIMISES sensitivity:
 * nearby points are pushed to feel different forces, encouraging mixing / chaos.
 * The log gives diminishing returns, keeping the term from dominating once the
 * field is already locally expansive.
 *
 * The jitter is deterministic (not random) so the loss is repeatable across
 * steps and gradients are stable: the batch is split in half — the first half
 * is nudged +eps in x, the second half +eps in y — covering both axes.
 *
 * @param field Differentiable vector field p[N,2] → F[N,2].
 * @param pos   [N, 2] sample positions.
 * @param eps   Jitter magnitude (default 1e-3).
 * @returns Differentiable scalar; lower ⇒ more locally chaotic.
 */
export function chaosLoss(
  field: (p: tf.Tensor2D) => tf.Tensor2D,
  pos: tf.Tensor2D,
  eps = 1e-3
): tf.Scalar {
  return tf.tidy(() => {
    const n = pos.shape[0];

    // Deterministic per-particle offset: +eps in x for the first half of the
    // batch, +eps in y for the second half, so both axes are probed each step.
    const half = Math.floor(n / 2);
    const dx = tf.tensor2d([[eps, 0]]).tile([half, 1]);
    const dy = tf.tensor2d([[0, eps]]).tile([n - half, 1]);
    const delta = dx.concat(dy, 0) as tf.Tensor2D;

    const f0 = field(pos);
    const f1 = field(pos.add(delta) as tf.Tensor2D);

    // Epsilon INSIDE the radicand: d/dx sqrt(0) is infinite, and f1≈f0 happens
    // exactly where the field goes locally flat/saturated — the state chaosLoss
    // pushes away from. Without this the sqrt gradient is NaN/Inf and poisons
    // every field weight through Adam. The outer `.add(1e-6)` before `.log()`
    // does NOT help: the NaN originates upstream, in the sqrt gradient itself.
    // (Mirrors spiralLoss in main.ts, which guards its sqrt with `.add(1e-4)`.)
    const sep = f1
      .sub(f0)
      .square()
      .sum(1)
      .add(1e-12)
      .sqrt()
      .div(tf.scalar(eps + 1e-9));

    return sep.add(1e-6).log().mean().neg().asScalar();
  });
}
