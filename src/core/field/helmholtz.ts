/**
 * Helmholtz decomposition force field.
 * ===================================================================
 *
 * The Helmholtz (Hodge) decomposition states that any sufficiently smooth
 * 2D vector field F can be written as the sum of a CURL-FREE part and a
 * DIVERGENCE-FREE part:
 *
 *     F = ∇φ            +          (∂ψ/∂y, -∂ψ/∂x)
 *         └─ gradient ─┘          └──── curl of stream function ────┘
 *          curl-free                    divergence-free
 *         (potential)                  (solenoidal / area-preserving)
 *
 * We learn the two SCALAR potentials φ and ψ with small neural nets and
 * DERIVE the vector field from them. This buys two structural guarantees
 * *by construction* (not by training luck):
 *
 *   1. ∇φ is irrotational (curl(∇φ) ≡ 0). It has sources/sinks, so it
 *      pulls particles toward basins of φ — the PREDICTABLE / ORDER lane.
 *
 *   2. curl(ψ) = (∂ψ/∂y, -∂ψ/∂x) is divergence-free (div(curl ψ) ≡ 0).
 *      A divergence-free flow is AREA-PRESERVING: particles swirl and mix
 *      but never collapse — the UNPREDICTABLE / CHAOS lane.
 *
 * The mix knob `alpha ∈ [0,1]` slides between the two:
 *
 *     forces = (1 - alpha) * ∇φ  +  alpha * curl(ψ)
 *
 * so alpha = 0 is pure gradient (order) and alpha = 1 is pure solenoidal
 * (mixing). It is a live, mutable field — an order↔chaos slider.
 *
 * ─────────────────────────────────────────────────────────────────────
 * CRITICAL: 2D CURL SIGN CONVENTION
 * ─────────────────────────────────────────────────────────────────────
 * In 2D a "curl" of a scalar stream function ψ(x, y) is the vector
 *
 *     curl(ψ) = ( +∂ψ/∂y , -∂ψ/∂x )
 *
 * The SECOND component is NEGATED. tf.grad(sum(ψ)) returns the raw
 * gradient [∂ψ/∂x, ∂ψ/∂y]; we must (a) SWAP the two components AND
 * (b) NEGATE the (new) second one. Getting the sign wrong flips the
 * rotation direction and, more importantly, no longer yields a
 * divergence-free field — silently breaking the whole "area-preserving"
 * guarantee. Do not "simplify" this.
 *
 * ─────────────────────────────────────────────────────────────────────
 * DIFFERENTIABILITY
 * ─────────────────────────────────────────────────────────────────────
 * `forces()` computes the fields via `tf.grad` (differentiation w.r.t.
 * the INPUT position). `tf.grad` is itself differentiable in tfjs, so
 * when the integrator runs the physics step inside `optimizer.minimize`,
 * second-order gradients flow from the loss, through the derived vector
 * field, and into φ's / ψ's weights. Nothing here detaches the tape.
 *
 * ─────────────────────────────────────────────────────────────────────
 * INTEGRATOR NOTE
 * ─────────────────────────────────────────────────────────────────────
 * Output is a RAW vector field (roughly O(1) magnitude, can be negative).
 * Unlike the sigmoid MLPs in main.ts, do NOT apply the `(raw - 0.5)`
 * shift — that shift only re-centers a [0,1] sigmoid output. Feed this
 * vector straight into the physics and let `forceMagnitude` scale it:
 *
 *     const forces = field.forces(posNorm).mul(cfg.forceMagnitude);
 *
 * `trainableWeights` exposes the underlying `tf.Variable`s so the
 * integrator may pass them as the `varList` to `optimizer.minimize`.
 */

import * as tf from "@tensorflow/tfjs";

/**
 * A differentiable 2D force field over normalized positions.
 *
 * `forces(posNorm)` maps `[N,2]` normalized positions (each coord in
 * roughly `[0,1]`) to `[N,2]` raw force vectors. The result stays
 * differentiable w.r.t. `trainableWeights` so it can be optimized in place.
 */
export interface ForceField {
  /** Map normalized positions `[N,2]` -> raw force vectors `[N,2]`. */
  forces(posNorm: tf.Tensor2D): tf.Tensor2D;
  /** The learnable variables of the field (for optimizer varLists). */
  readonly trainableWeights: tf.Variable[];
  /** Release all GPU/CPU tensors owned by the field. */
  dispose(): void;
}

/** Config for {@link HelmholtzField}. */
export interface HelmholtzFieldConfig {
  /**
   * Order↔chaos mix in `[0,1]`. `0` = pure gradient ∇φ (predictable),
   * `1` = pure solenoidal curl(ψ) (mixing). Mutable at runtime.
   */
  alpha: number;
  /** Hidden layer widths for BOTH scalar nets. Default `[32, 32]`. */
  hiddenUnits?: number[];
}

/**
 * Build a scalar potential net R^2 -> R: tanh hidden stack -> linear dense(1).
 * Linear output head is essential — a scalar potential must be unbounded so
 * its gradient can point in any direction with any magnitude.
 *
 * The hidden activation MUST be twice-differentiable: the live training path
 * takes a SECOND-ORDER gradient (optimizer.minimize differentiates a loss that
 * itself depends on tf.grad(net) w.r.t. the input, back into the net weights).
 * SELU/ELU throw on the 2nd-order pass — SELU's gradient uses the `Greater` op
 * (no registered gradient), ELU fails on `EluGrad` — so the whole Helmholtz
 * piece dies on frame 1. tanh is smooth everywhere and gives a well-behaved
 * (non-degenerate) potential; relu would also run but its zero second
 * derivative yields a degenerate curl/gradient field. Do NOT switch to selu/elu.
 */
function makeScalarNet(hiddenUnits: number[]): tf.Sequential {
  const net = tf.sequential();
  hiddenUnits.forEach((units, i) => {
    const cfg: any = { units, activation: "tanh" };
    if (i === 0) cfg.inputShape = [2];
    net.add(tf.layers.dense(cfg));
  });
  net.add(tf.layers.dense({ units: 1, activation: "linear" }));
  return net;
}

/**
 * Gradient of the scalar field `net` w.r.t. its input, evaluated at `x`.
 *
 * Because rows of `x` are independent, `∂/∂x Σ_i net(x_i) = [∇net(x_i)]_i`,
 * so summing the scalar outputs and differentiating once yields the per-row
 * gradient `[N,2]` in a single pass. `tf.grad` keeps this differentiable
 * w.r.t. the net's weights (higher-order autodiff).
 */
function scalarGradient(net: tf.Sequential, x: tf.Tensor2D): tf.Tensor2D {
  const sumScalar = (xx: tf.Tensor2D): tf.Scalar =>
    (net.predict(xx) as tf.Tensor).sum() as tf.Scalar;
  return tf.grad(sumScalar)(x) as tf.Tensor2D;
}

/**
 * Helmholtz-decomposed neural force field.
 *
 * Learns two scalar potentials and derives a vector field that is an
 * explicit blend of its curl-free (gradient) and divergence-free (curl)
 * parts. See the file header for the full math and sign convention.
 */
export class HelmholtzField implements ForceField {
  /** Order↔chaos slider in `[0,1]`; mutate freely at runtime. */
  alpha: number;

  /** Scalar potential φ: R^2 -> R. Its gradient is the curl-free lane. */
  private readonly phi: tf.Sequential;
  /** Stream function ψ: R^2 -> R. Its curl is the divergence-free lane. */
  private readonly psi: tf.Sequential;

  private readonly weights: tf.Variable[];

  constructor({ alpha, hiddenUnits = [32, 32] }: HelmholtzFieldConfig) {
    this.alpha = alpha;
    this.phi = makeScalarNet(hiddenUnits);
    this.psi = makeScalarNet(hiddenUnits);

    // LayerVariable.val is the underlying tf.Variable (protected in the
    // typings). We expose the real Variables so the integrator can hand
    // them to optimizer.minimize as an explicit varList.
    const collect = (net: tf.Sequential): tf.Variable[] =>
      net.trainableWeights.map((w) => (w as any).val as tf.Variable);
    this.weights = [...collect(this.phi), ...collect(this.psi)];
  }

  get trainableWeights(): tf.Variable[] {
    return this.weights;
  }

  /**
   * Raw force vectors `[N,2]` at the given normalized positions `[N,2]`.
   *
   *   gradPhi = ∇φ                         (curl-free)
   *   curlPsi = (∂ψ/∂y, -∂ψ/∂x)            (divergence-free — SEE SIGN NOTE)
   *   forces  = (1 - alpha) * gradPhi + alpha * curlPsi
   *
   * Differentiable w.r.t. {@link trainableWeights}. Wrapped in `tf.tidy`
   * to free intermediates; the tape retains what backprop needs.
   */
  forces(posNorm: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      // Curl-free lane: raw gradient IS the force.
      const gradPhi = scalarGradient(this.phi, posNorm); // [N,2]

      // Divergence-free lane. gradPsi = [∂ψ/∂x, ∂ψ/∂y].
      const gradPsi = scalarGradient(this.psi, posNorm); // [N,2]
      const dPsi_dx = gradPsi.slice([0, 0], [-1, 1]); // ∂ψ/∂x
      const dPsi_dy = gradPsi.slice([0, 1], [-1, 1]); // ∂ψ/∂y
      // 2D curl of a stream function: (+∂ψ/∂y, -∂ψ/∂x). NEGATE 2nd comp.
      const curlPsi = tf.concat([dPsi_dy, dPsi_dx.neg()], 1) as tf.Tensor2D;

      const a = this.alpha;
      return gradPhi.mul(1 - a).add(curlPsi.mul(a)) as tf.Tensor2D;
    });
  }

  dispose(): void {
    this.phi.dispose();
    this.psi.dispose();
  }
}
