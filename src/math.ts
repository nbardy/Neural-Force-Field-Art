// Math ML utils implimented in tfjs
import * as tf from "@tensorflow/tfjs";

// Need to implement this
/**
 * Sample the tensor.
 * @param {tf.Tensor} tensor The tensor to sample from.
 * @param {number} k Number of elements to sample.
 * @returns {tf.Tensor} The sampled tensor.
 */
export const sample = (tensor, k) => {
  // Ensure 'k' is valid
  if (k > tensor.shape[0]) {
    throw new Error(
      `'k' should be less than or equal to the first dimension of the tensor. Received k=${k}, tensor's first dimension=${tensor.shape[0]}`
    );
  }

  // Generate random values for each row of tensor (shape: [N])
  const randomValues = tf.randomUniform([tensor.shape[0]]);

  // Get the top k row indices.
  const { indices } = tf.topk(randomValues, k);

  // Use the gathered indices to extract rows from the tensor
  return tensor.gather(indices);
};

export const pairwiseDistanceWithDropout = (X, Y, N) => {
  // Sample N elements from X and Y
  const sampledX = sample(X, N); // Shape: [N, ...]
  const sampledY = sample(Y, N); // Shape: [N, ...]

  // Using broadcasting to get pairwise differences
  // First, we expand dimensions to prepare for broadcasting:
  const expandedX = sampledX.expandDims(1); // Shape: [N, 1, ...]
  const expandedY = sampledY.expandDims(0); // Shape: [1, N, ...]

  // Now, compute the difference:
  const diff = tf.sub(expandedX, expandedY); // Resulting Shape: [N, N, ...]

  // Compute the pairwise distances:
  const axis = -1;

  const distance = tf.sqrt(tf.sum(tf.square(diff), axis)); // Shape: [N, N]

  return distance;
};
