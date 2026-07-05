import * as tf from "@tensorflow/tfjs";

/**
 * PredictorEnsemble — an RND / ensemble-disagreement adversary that rewards
 * *structured* (reducible) surprise instead of aleatoric noise.
 *
 * ## The noisy-TV problem
 * A naive curiosity signal ("go where prediction error is high") is fooled by a
 * TV showing static: the pixels are unpredictable, error stays high forever, and
 * the agent stares at noise. The error there is *irreducible* (aleatoric) — no
 * amount of learning shrinks it, so it is not real novelty.
 *
 * ## Disagreement fixes it
 * We train `k` independent predictors on the same observed transitions. Their
 * *disagreement* (variance across the ensemble) distinguishes the two regimes:
 *
 *   - **Reducible / structured novelty** — a region the generator has not yet
 *     "explained". Each predictor extrapolates differently, so they DISAGREE.
 *     As they see more data there, they converge and disagreement decays. This
 *     is exactly the surprise worth chasing.
 *   - **Irreducible noise** (the noisy TV) — the target displacement is random.
 *     Every predictor learns the same thing: the conditional MEAN. They all AGREE
 *     it is unpredictable, so disagreement is LOW. Noise is not rewarded.
 *
 * So `disagreement` is high for "unpredictable but structured" states and low for
 * pure noise — "unpredictable but not noise". The generator is rewarded by
 * `generatorReward = -disagreement` (see below), pushing the field toward states
 * the ensemble cannot yet agree on.
 *
 * ## Adversarial (GAN) reading
 * - `trainStep` is the DISCRIMINATOR update: the ensemble fits the generator's
 *   actual dynamics, collapsing disagreement in regions the field keeps visiting.
 * - `generatorReward` is the GENERATOR pull: it seeks states the discriminator
 *   ensemble has not modelled, keeping the motion perpetually novel. The field
 *   flees the ensemble; the ensemble chases the field.
 *
 * The ensemble owns its parameters and its own Adam optimizers. They are kept
 * strictly SEPARATE from the generator field — nothing here is exposed as a
 * `trainableWeights` the generator optimizer could accidentally co-optimize.
 */
export class PredictorEnsemble {
  /** Number of independent predictors in the ensemble. */
  readonly k: number;

  /** The k predictor networks: normalized pos [N,2] -> next-step displacement [N,2]. */
  private readonly predictors: tf.Sequential[];

  /** One Adam optimizer per predictor — the discriminator's own optimizers. */
  private readonly optimizers: tf.Optimizer[];

  /**
   * @param opts.k          Number of predictors (small; 4–8 is plenty). Default 5.
   * @param opts.hiddenUnits Width of the first SELU hidden layer. Default 32.
   * @param opts.featureDim  Width of the second SELU (feature) layer. Default 16.
   * @param opts.learningRate Adam learning rate for the discriminator update. Default 1e-3.
   *
   * Each predictor is a small MLP: dense(hiddenUnits, SELU) -> dense(featureDim,
   * SELU) -> dense(2, linear). The linear head is deliberate: displacement is an
   * unbounded real vector, so no squashing activation on the output.
   */
  constructor({
    k = 5,
    hiddenUnits = 32,
    featureDim = 16,
    learningRate = 1e-3,
  }: {
    k?: number;
    hiddenUnits?: number;
    featureDim?: number;
    learningRate?: number;
  } = {}) {
    this.k = k;
    this.predictors = [];
    this.optimizers = [];

    for (let i = 0; i < k; i++) {
      const net = tf.sequential();
      net.add(
        tf.layers.dense({ units: hiddenUnits, activation: "selu", inputShape: [2] })
      );
      net.add(tf.layers.dense({ units: featureDim, activation: "selu" }));
      // Linear head — displacement is unbounded, do not squash it.
      net.add(tf.layers.dense({ units: 2 }));
      this.predictors.push(net);
      this.optimizers.push(tf.train.adam(learningRate));
    }
  }

  /**
   * DISCRIMINATOR update. Train all k predictors to regress the *observed*
   * displacement `nextPos - pos` from the current position via MSE. Each predictor
   * is optimized independently by its own Adam over ONLY its own weights, so the
   * ensemble members stay decorrelated (that decorrelation is what produces
   * meaningful disagreement in unexplored regions).
   *
   * `pos` / `nextPos` are treated as constants (data) here — gradient flows into
   * the predictor weights only, never back into the generator field.
   *
   * @param pos     [N,2] normalized current positions (generator-produced).
   * @param nextPos [N,2] normalized next-step positions (observed transition).
   * @returns Mean predictor MSE across the ensemble (plain number, for logging).
   */
  trainStep(pos: tf.Tensor2D, nextPos: tf.Tensor2D): number {
    // Observed displacement is the regression target — pure data, no grad.
    const target = tf.tidy(() => nextPos.sub(pos)) as tf.Tensor2D;

    let total = 0;
    for (let i = 0; i < this.k; i++) {
      const net = this.predictors[i];
      const cost = this.optimizers[i].minimize(
        () =>
          tf.tidy(() => {
            const pred = net.predict(pos) as tf.Tensor2D;
            return tf.losses.meanSquaredError(target, pred).asScalar();
          }),
        true
      ) as tf.Scalar;

      total += cost.dataSync()[0];
      cost.dispose();
    }

    target.dispose();
    return total / this.k;
  }

  /**
   * Ensemble disagreement at `pos`: the variance ACROSS the k predictors of their
   * predicted displacement, reduced to a single scalar.
   *
   *   preds   : k × [N,2]  (one prediction per predictor)
   *   var     : [N,2]      per-component variance across the k predictors
   *   perPart : [N]        per-particle variance = trace of the prediction cov
   *   result  : scalar     mean over particles
   *
   * High where the ensemble extrapolates inconsistently (structured, reducible
   * novelty); low where they all collapse to the same conditional mean (either a
   * well-explored region OR pure noise — see class docs on the noisy TV).
   *
   * Differentiable w.r.t. the generator field IF `pos` was produced by it (the
   * predictor weights are held fixed during the generator update because the
   * generator's optimizer only owns the field's variables).
   *
   * @param pos [N,2] normalized positions.
   * @returns Differentiable scalar disagreement.
   */
  disagreement(pos: tf.Tensor2D): tf.Scalar {
    return tf.tidy(() => {
      const preds = this.predictors.map(
        (net) => net.predict(pos) as tf.Tensor2D
      );
      const stacked = tf.stack(preds); // [k, N, 2]
      const mean = stacked.mean(0); // [N, 2]
      const variance = stacked.sub(mean).square().mean(0); // [N, 2]
      const perParticleVar = variance.sum(1); // [N] — trace of prediction covariance
      return perParticleVar.mean().asScalar();
    });
  }

  /**
   * GENERATOR reward = `-disagreement(pos)`.
   *
   * The generator MINIMIZES loss, so returning the negated disagreement makes the
   * field seek states where the ensemble DISAGREES = reducible/structured novelty.
   * Crucially it does NOT chase noise: on aleatoric noise every predictor learns
   * the same conditional mean, they AGREE, disagreement → 0, and there is no pull.
   * That is the noisy-TV immunity — the field is pushed to be "unpredictable but
   * not noise". Add this term (weighted) into the generator's `computeLoss`.
   *
   * @param pos [N,2] normalized positions from the generator physics step.
   * @returns Differentiable scalar reward term (to be MINIMIZED by the generator).
   */
  generatorReward(pos: tf.Tensor2D): tf.Scalar {
    return tf.tidy(() => this.disagreement(pos).neg().asScalar());
  }

  /** Release all predictor weights and optimizer state. */
  dispose(): void {
    for (const net of this.predictors) net.dispose();
    for (const opt of this.optimizers) opt.dispose();
  }
}
