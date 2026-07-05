import * as tf from "@tensorflow/tfjs";

/**
 * Finite-difference divergence penalty — drives measure preservation.
 *
 * Estimates ∇·F = ∂Fx/∂x + ∂Fy/∂y at each sample position via central
 * differences, then penalises the squared divergence:
 *
 *   loss = mean( (∇·F)² )
 *
 * A divergence-free (solenoidal) field neither compresses nor inflates volumes
 * of particles, so minimising this keeps the flow from collapsing to a point or
 * blowing up to the boundaries — it stays incompressible / area-preserving.
 *
 * The field is sampled four times (±h in x, ±h in y). The step `h` is in the
 * same coordinate units as `pos`; pick it small relative to feature scale but
 * large enough to avoid catastrophic cancellation.
 *
 * @param field Differentiable vector field p[N,2] → F[N,2].
 * @param pos   [N, 2] sample positions.
 * @param h     Central-difference step (default 1e-2).
 * @returns Differentiable scalar mean-squared divergence.
 */
export function divergencePenalty(
  field: (p: tf.Tensor2D) => tf.Tensor2D,
  pos: tf.Tensor2D,
  h = 1e-2
): tf.Scalar {
  return tf.tidy(() => {
    const ex = tf.tensor2d([[h, 0]]);
    const ey = tf.tensor2d([[0, h]]);

    // Central differences: sample the field on either side of each point.
    const fxp = field(pos.add(ex) as tf.Tensor2D);
    const fxm = field(pos.sub(ex) as tf.Tensor2D);
    const fyp = field(pos.add(ey) as tf.Tensor2D);
    const fym = field(pos.sub(ey) as tf.Tensor2D);

    const inv = tf.scalar(1 / (2 * h));

    // ∂Fx/∂x = (Fx(x+h) - Fx(x-h)) / 2h ; column 0 of the x-perturbed samples.
    const dFxdx = fxp.slice([0, 0], [-1, 1]).sub(fxm.slice([0, 0], [-1, 1])).mul(inv);
    // ∂Fy/∂y = (Fy(y+h) - Fy(y-h)) / 2h ; column 1 of the y-perturbed samples.
    const dFydy = fyp.slice([0, 1], [-1, 1]).sub(fym.slice([0, 1], [-1, 1])).mul(inv);

    const div = dFxdx.add(dFydy);
    return div.square().mean().asScalar();
  });
}
