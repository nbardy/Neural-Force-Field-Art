import * as tf from "@tensorflow/tfjs";
import { getEnv } from "./env";

export const assertShape = (tensor: tf.Tensor, shape: number[]) => {
  // if debug
  if (!getEnv().assertShape) {
    return true;
  }

  if (tensor.shape.length !== shape.length) {
    throw new Error(
      `Expected shape length ${shape.length} but got ${tensor.shape.length}`
    );
  }
  for (let i = 0; i < shape.length; i++) {
    if (tensor.shape[i] !== shape[i]) {
      throw new Error(`Expected shape ${shape} but got ${tensor.shape}`);
    }
  }
};
