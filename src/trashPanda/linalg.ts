import * as tf from "@tensorflow/tfjs";

export function normalize(tensor: tf.Tensor) {
  const norm = tensor.norm();
  return tensor.div(norm);
}
