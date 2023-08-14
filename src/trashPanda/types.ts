import * as tf from "@tensorflow/tfjs";

export interface TrashPandaModel {
  predict(inputTensor: tf.Tensor): tf.Tensor;
}
