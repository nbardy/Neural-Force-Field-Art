import * as tf from "@tensorflow/tfjs";

/**
 * ε for the constant-mode fraction, in the SAME units as the forces handed in
 * (i.e. Fs = F·forceMagnitude on the physical path).
 *
 * Must equal `STRUCT_EPS` in `src/render/webgpu/train_wgsl.ts` — the fused term
 * and this oracle are only comparable if the constant is literally the same
 * number, and `tools/train_struct_test.ts` imports both and asserts it.
 */
export const STRUCTURE_EPS = 1e-8;

/**
 * CONSTANT-MODE FRACTION of a force batch — the "make structure, not push"
 * objective. Minimize it.
 *
 *     dc²  = ‖mean_i F_i‖²                 the spatially CONSTANT mode
 *     ac²  = mean_i ‖F_i − mean F‖²        the spatially VARYING mode
 *     L    = (dc² + ε) / (ac² + dc² + ε)   because ac² + dc² = mean ‖F‖² exactly
 *
 * so `1 − L` is the normalized structure `ac²/(ac²+dc²)`. Minimizing L is
 * maximizing that fraction.
 *
 * WHY THE RATIO AND NOT `ac²` ITSELF. Raw AC is homogeneous of degree 1 in F:
 * scaling every force by c scales ac by c. Maximizing it therefore has a pure
 * AMPLITUDE CHEAT — the field grows until tanh saturates and never acquires a
 * single new spatial feature (that end state is `satFrac` in the health
 * snapshot, measured at 0.46 on the collapsed point-observer baseline). The
 * ratio is invariant under F → cF, so the only way to move it is to trade
 * constant push for spatial variation.
 *
 * WHY ε IS IN BOTH NUMERATOR AND DENOMINATOR. With ε in the denominator only,
 * the dead field F ≡ 0 scores L = 0/ε = 0 — a PERFECT score — and the term
 * would drive the field to zero to collect it. With ε in both, F ≡ 0 scores
 * L = 1, the worst possible value. This is the same trap `directionOrderLoss`
 * closes with its τ, and it is the one real bug the first draft of this loss
 * had.
 *
 * Range is exactly [0,1]: Jensen gives ‖mean F‖² ≤ mean‖F‖².
 *
 * @param force [N, 2] force vectors for the batch, in physical units.
 * @returns Differentiable scalar in [0,1]; 0 = pure structure, 1 = pure DC.
 */
export function constantModeFraction(force: tf.Tensor2D): tf.Scalar {
  return tf.tidy(() => {
    const mean = force.mean(0) as tf.Tensor1D;
    const dc2 = mean.square().sum();
    const ms = force.square().sum(1).mean();
    return dc2.add(STRUCTURE_EPS).div(ms.add(STRUCTURE_EPS)).asScalar();
  });
}

/** The AC/DC split as plain numbers (telemetry — dataSyncs, not differentiable).
 *  Same definitions as `tools/collapse_probe.ts::diagnostics`. */
export function acDcSplit(force: tf.Tensor2D): { ac: number; dc: number } {
  const [ac, dc] = tf.tidy(() => {
    const mean = force.mean(0) as tf.Tensor1D;
    return tf.stack([
      force.sub(mean.reshape([1, 2])).square().sum(1).mean().sqrt(),
      mean.square().sum().sqrt(),
    ]);
  }).dataSync();
  return { ac, dc };
}
