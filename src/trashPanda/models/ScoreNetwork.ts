import * as tf from "@tensorflow/tfjs";
import { Module } from "../types";

/**
 * Score Network for learning a vector field (score function).
 *
 * Takes a 2D position and outputs a 2D force/direction vector.
 * Trained via score matching to approximate ∇log p(x) for a
 * target distribution, then applied as a force field to drive
 * particles toward that distribution.
 */
export class ScoreNetwork implements Module {
  net: tf.Sequential;

  constructor({
    hiddenUnits = [32, 64, 32],
  }: {
    hiddenUnits?: number[];
  } = {}) {
    this.net = tf.sequential();

    for (let i = 0; i < hiddenUnits.length; i++) {
      const config: any = { units: hiddenUnits[i], activation: "selu" };
      if (i === 0) config.inputShape = [2];
      this.net.add(tf.layers.dense(config));
    }

    this.net.add(tf.layers.dense({ units: 2, activation: "tanh" }));
  }

  predict(x: tf.Tensor): tf.Tensor {
    return this.net.predict(x) as tf.Tensor;
  }

  get trainableWeights() {
    return this.net.trainableWeights;
  }
}
