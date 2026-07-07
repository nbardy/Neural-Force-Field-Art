/**
 * Neural force field — two direct-vector heads (FIRST-ORDER).
 * ===================================================================
 *
 * We keep the Helmholtz *intuition* — a force field is a blend of a
 * curl-free "order" lane and a divergence-free "chaos" lane — but we no
 * longer DERIVE the two lanes as an analytic gradient / curl of scalar
 * potentials. Instead each lane is a small MLP that outputs its 2D force
 * vector DIRECTLY:
 *
 *     g(posNorm) : R^2 -> R^2   "gradient / attracting"  — ORDER lane
 *     r(posNorm) : R^2 -> R^2   "rotational / mixing"    — CHAOS lane
 *
 *     forces = (1 - alpha) * g(posNorm)  +  alpha * r(posNorm)
 *
 * The mix knob `alpha ∈ [0,1]` slides between them (0 = pure `g` / order,
 * 1 = pure `r` / chaos). It is a live, mutable field — an order↔chaos slider.
 *
 * ─────────────────────────────────────────────────────────────────────
 * WHY DIRECT VECTORS INSTEAD OF grad(φ) / curl(ψ)  (the whole point)
 * ─────────────────────────────────────────────────────────────────────
 * The previous design built the force as F = ∇φ + curl(ψ), computing each
 * lane with `tf.grad` of a scalar net w.r.t. the INPUT position. That gave
 * an EXACT divergence-free chaos lane by construction — but at a fatal cost:
 * because `forces()` itself called `tf.grad`, training (optimizer.minimize
 * differentiates the loss w.r.t. the WEIGHTS) had to differentiate THROUGH
 * that inner gradient. That is SECOND-ORDER autograd, measured at
 * ~800–1700 ms/frame — roughly 10x too slow, and it also forced tanh-only
 * hidden layers (SELU/ELU have no registered 2nd-order gradient in tfjs).
 *
 * TRADEOFF (deliberate): we drop the *exact* divergence-free guarantee and
 * instead encourage it SOFTLY. `main.ts`'s `divergencePenalty` loss term
 * measures ∇·F by forward-only finite differences and penalises its square,
 * nudging `r` toward area-preserving mixing during training. In return,
 * `forces()` is a plain FORWARD pass — no `tf.grad` inside — so training is a
 * single FIRST-order backward, identical in cost to the fast MLP pieces
 * (~10x faster). Approximate-and-cheap beats exact-and-unusable here.
 *
 * ─────────────────────────────────────────────────────────────────────
 * DIFFERENTIABILITY
 * ─────────────────────────────────────────────────────────────────────
 * `forces()` only runs `net.predict` (a forward pass) wrapped in `tf.tidy`.
 * Nothing detaches the tape: when the integrator runs the physics step
 * inside `optimizer.minimize`, a single first-order gradient flows from the
 * loss, through the two heads' outputs, into `g`'s and `r`'s weights.
 *
 * ─────────────────────────────────────────────────────────────────────
 * INTEGRATOR NOTE
 * ─────────────────────────────────────────────────────────────────────
 * Each head ends in a `tanh` layer, so the raw force is bounded ~O(1) (and
 * can be negative). Like the sigmoid MLPs in main.ts this is a RAW signed
 * vector, but do NOT apply the `(raw - 0.5)` shift — that only re-centers a
 * [0,1] sigmoid. Feed the vector straight in and let `forceMagnitude` scale:
 *
 *     const forces = field.forces(posNorm).mul(cfg.forceMagnitude);
 *
 * `trainableWeights` exposes the underlying `tf.Variable`s so the integrator
 * may pass them as the `varList` to `optimizer.minimize`.
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
   * Order↔chaos mix in `[0,1]`. `0` = pure `g` (attracting / predictable),
   * `1` = pure `r` (rotational / mixing). Mutable at runtime.
   */
  alpha: number;
  /** Hidden layer widths for BOTH vector heads. Default `[32, 32]`. */
  hiddenUnits?: number[];
  /**
   * Multi-species class count C (default 0 = classless). When C > 0 the
   * CHAOS head `r` takes `[pos, onehot(class)]` (2+C inputs) while the order
   * head `g` stays class-blind. Class-aware fields are FUSED-KERNEL-ONLY:
   * {@link forces} throws, because the tfjs path has no class to feed —
   * training and advection both run in the WGSL kernels (which derive class
   * from the particle index hash).
   */
  classes?: number;
}

/**
 * Build a vector head R^2 -> R^2: SELU hidden stack -> tanh dense(2).
 *
 * SELU hidden layers are fine now that training is first-order only — the
 * previous design banned SELU because its gradient uses the `Greater` op,
 * which has no registered SECOND-order gradient in tfjs and killed the old
 * `tf.grad`-based path. The final `tanh` bounds the output to ~O(1) per
 * component so the raw force magnitude matches the old grad/curl scale and
 * existing `forceMagnitude` configs still read correctly.
 */
function makeVectorNet(hiddenUnits: number[], inputDim = 2): tf.Sequential {
  const net = tf.sequential();
  hiddenUnits.forEach((units, i) => {
    const cfg: any = { units, activation: "selu" };
    if (i === 0) cfg.inputShape = [inputDim];
    net.add(tf.layers.dense(cfg));
  });
  net.add(tf.layers.dense({ units: 2, activation: "tanh" }));
  return net;
}

/**
 * Direct-vector neural force field.
 *
 * Two small MLP heads output the order lane (`g`) and chaos lane (`r`) as
 * 2D vectors directly; `forces()` blends them by `alpha`. The chaos lane's
 * divergence-free character is encouraged SOFTLY by the `divergencePenalty`
 * loss rather than constructed exactly — see the file header for why (the
 * win is single first-order autograd, ~10x faster than the old grad/curl).
 */
export class HelmholtzField implements ForceField {
  /** Order↔chaos slider in `[0,1]`; mutate freely at runtime. */
  alpha: number;
  /** Multi-species class count (0 = classless). Immutable. */
  readonly classes: number;

  /** Order lane: R^2 -> R^2 direct "gradient / attracting" vector. */
  private readonly g: tf.Sequential;
  /** Chaos lane: R^2 -> R^2 direct "rotational / mixing" vector. */
  private readonly r: tf.Sequential;

  private readonly weights: tf.Variable[];

  constructor({ alpha, hiddenUnits = [32, 32], classes = 0 }: HelmholtzFieldConfig) {
    this.alpha = alpha;
    this.classes = classes;
    this.g = makeVectorNet(hiddenUnits, 2);
    this.r = makeVectorNet(hiddenUnits, 2 + classes);

    // LayerVariable.val is the underlying tf.Variable (protected in the
    // typings). We expose the real Variables so the integrator can hand
    // them to optimizer.minimize as an explicit varList.
    const collect = (net: tf.Sequential): tf.Variable[] =>
      net.trainableWeights.map((w) => (w as any).val as tf.Variable);
    this.weights = [...collect(this.g), ...collect(this.r)];
  }

  get trainableWeights(): tf.Variable[] {
    return this.weights;
  }

  /**
   * The two vector heads `[g, r]` — read-only structural access so the fused
   * advect kernel (src/render/webgpu/advect.ts) can generate WGSL matching
   * the live architecture instead of hardcoding dims.
   */
  get heads(): [tf.Sequential, tf.Sequential] {
    return [this.g, this.r];
  }

  /**
   * Raw force vectors `[N,2]` at the given normalized positions `[N,2]`.
   *
   *   g       = order  lane vector (attracting)  — head `g`
   *   r       = chaos  lane vector (rotational)   — head `r`
   *   forces  = (1 - alpha) * g + alpha * r
   *
   * A plain FORWARD pass (no `tf.grad`): differentiable w.r.t.
   * {@link trainableWeights} in a SINGLE first-order backward. Wrapped in
   * `tf.tidy` to free intermediates; the tape retains what backprop needs.
   */
  forces(posNorm: tf.Tensor2D): tf.Tensor2D {
    if (this.classes > 0) {
      throw new Error(
        "HelmholtzField.forces: class-aware fields are fused-kernel-only " +
          "(the tfjs path has no class input) — do not use ?train=tfjs with " +
          "a classes>0 piece."
      );
    }
    return tf.tidy(() => {
      const gVec = this.g.predict(posNorm) as tf.Tensor2D; // order lane [N,2]
      const rVec = this.r.predict(posNorm) as tf.Tensor2D; // chaos lane [N,2]

      const a = this.alpha;
      return gVec.mul(1 - a).add(rVec.mul(a)) as tf.Tensor2D;
    });
  }

  dispose(): void {
    this.g.dispose();
    this.r.dispose();
  }
}
