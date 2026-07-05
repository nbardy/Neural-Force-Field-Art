import * as tf from "@tensorflow/tfjs";

/**
 * Power-spectrum slope penalty — best-effort turbulence-cascade shaping.
 *
 * Fully-developed 2D turbulence has a power spectrum that decays as a power law
 * E(k) ∝ k^slope (the Kolmogorov −5/3 inertial range in the energy-cascade
 * regime). We reward the force field for having a similar self-similar,
 * scale-filling structure by matching the slope of its log power spectrum.
 *
 * ── LIMITATION ──────────────────────────────────────────────────────────────
 * tfjs has NO 2D FFT and no true radial averaging. This is a PROXY: we take the
 * 1D FFT along every row and every column (`tf.spectral.fft`, which transforms
 * the innermost axis), average |FFT|² across rows and across columns, and treat
 * the resulting per-frequency power as a stand-in for the radial spectrum P(|k|).
 * This captures axis-aligned scaling but ignores true isotropic radial binning
 * and diagonal wavenumbers. It is a shaping regulariser, not a physics-accurate
 * spectral measurement. Treat as OPTIONAL.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Penalty (scale-invariant in overall power): we regress log P(k) against
 * targetSlope·log k with the intercept free, and penalise the residual
 * variance — so only the *slope* is constrained, not the absolute energy:
 *
 *   r(k)  = log P(k) − targetSlope·log k
 *   loss  = Var_k( r )        (variance over the retained wavenumbers)
 *
 * loss = 0 ⇔ log P(k) is exactly an affine function of log k with the target
 * slope (a clean power law at the requested exponent).
 *
 * @param fieldGrid Force field sampled on a grid: `[G, G, 2]` vector field
 *   (magnitude is taken) or `[G, G]` scalar magnitude field.
 * @param targetSlope Desired log-log spectral slope (default −5/3 ≈ −1.667).
 * @returns Differentiable scalar; lower ⇒ spectrum closer to k^targetSlope.
 */
export function spectrumLoss(
  fieldGrid: tf.Tensor,
  targetSlope = -5 / 3
): tf.Scalar {
  return tf.tidy(() => {
    // κ: canonicalise to a [G, G] real magnitude grid (done once).
    const mag = toMagnitudeGrid(fieldGrid);
    const g = mag.shape[0];

    // Retain wavenumbers k = 1 .. floor(G/2); skip DC (k=0, undefined log k) and
    // the aliased upper half of the FFT (symmetric for real input).
    const half = Math.floor(g / 2);
    const zeros = tf.zerosLike(mag);

    // Row spectrum: FFT each row (innermost axis), power = |FFT|², average rows.
    const rowPower = tf.abs(tf.spectral.fft(tf.complex(mag, zeros))).square().mean(0);
    // Column spectrum: transpose so columns become the innermost axis.
    const magT = mag.transpose();
    const colPower = tf
      .abs(tf.spectral.fft(tf.complex(magT, tf.zerosLike(magT))))
      .square()
      .mean(0);

    // Combined axis-aligned power proxy per frequency bin.
    const power = rowPower.add(colPower).slice([1], [half]); // drop DC, keep 1..half

    // Wavenumbers k = 1..half.
    const k = tf.range(1, half + 1, 1, "float32");
    const logK = k.log();
    const logP = power.add(1e-8).log();

    // Residual after removing the target-slope trend; penalise its variance so
    // absolute power (intercept) is free and only the slope is constrained.
    const resid = logP.sub(logK.mul(targetSlope));
    const centred = resid.sub(resid.mean());
    return centred.square().mean().asScalar();
  });
}

/**
 * Reduce a grid to a real [G, G] magnitude field.
 * `[G, G, 2]` → per-cell vector norm; `[G, G]` → passed through unchanged.
 * Dispatch is on rank only (one canonicalisation step, no downstream checks).
 */
function toMagnitudeGrid(grid: tf.Tensor): tf.Tensor2D {
  if (grid.rank === 3) {
    return grid.square().sum(-1).sqrt() as tf.Tensor2D;
  }
  return grid as tf.Tensor2D;
}
