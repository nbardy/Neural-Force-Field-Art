/**
 * Differentiable loss library for the neural force field.
 *
 * Every LOSS export returns a `tf.Scalar`, is wrapped in `tf.tidy`, and is safe
 * to call inside `optimizer.minimize` — gradients flow back through the physics
 * chain into the field weights. Compose them with weighted sums.
 *
 *   isotropyLoss     — scale-invariant force-covariance anisotropy penalty
 *   directionOrderLoss — polar + nematic DIRECTION order (adversary anti-collapse)
 *   directionOrderParameters — the SAME statistic as plain numbers (telemetry,
 *                      not differentiable — it dataSyncs)
 *   divergencePenalty — finite-difference ∇·F (measure preservation)
 *   chaosLoss        — local finite-time Lyapunov proxy (rewards sensitivity)
 *   spectrumLoss     — best-effort power-law spectral slope shaping (optional)
 */
export {
  isotropyLoss,
  directionOrderLoss,
  directionOrderParameters,
} from "./isotropy";
export { divergencePenalty } from "./divergence";
export { chaosLoss } from "./chaos";
export { spectrumLoss } from "./spectrum";
