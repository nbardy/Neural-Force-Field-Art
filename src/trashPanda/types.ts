import * as tf from "@tensorflow/tfjs";

/**
 * Base Class to inherit for all models
 *
 * matches nn.Module from pytorch
 */
export interface Module {
  predict(...args: unknown[]): tf.Tensor;
}

export type allModels = tf.LayersModel | tf.Sequential | Module;
