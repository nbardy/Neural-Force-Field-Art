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
