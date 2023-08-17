import * as tf from "@tensorflow/tfjs";

/*
 * Playing with some ideas for geometric embeddings
 */

// note the spehre rotsres the point a random amount.
// the hyper sphere does it seiujd a random point. this is very different, i should implimwnt each method 
// as rotate, or rotate around m centers
// and then have four embeddeing functions
  

/**
 *  Rotated in random spherical directions. (Should maintain length)
 */
export function sphericalRotationEmbedding(inputTensor, numDirections) {
  const dim = inputTensor.shape[1];
  const randomDirections = tf.randomNormal([numDirections, dim, dim]);
  const [q, r] = tf.linalg.qr(randomDirections); // Shape: numDirections x dim x dim

  // Reshape inputTensor to match the dimensions and apply matrix multiplication
  const expandedInput = tf.expandDims(inputTensor, 0); // Shape: 1 x B x dim
  const tiledInput = tf.tile(expandedInput, [numDirections, 1, 1]); // Shape: numDirections x B x dim
  const embeddings = tf.matMul(tiledInput, q, false, true); // Shape: numDirections x B x dim

  // Reshape the result to concatenate along the second axis
  const sphericalEmbeddings = tf.reshape(embeddings, [-1, numDirections * dim]); // Shape: B x (numDirections * dim)

  return sphericalEmbeddings;
}

/**
 * Rotated in full disc sampled m times around a great circle around n hyperspheres.
 *
 * Idk wtf this will do, should look cool.
 */
function hypersphereRotationEmbedding(
  inputTensor: tf.Tensor3D,
  nSamples: number,
  mSamples: number
): tf.Tensor3D {
  const B = inputTensor.shape[0]; // Batch size
  const L = inputTensor.shape[1]; // Sequence length
  const D = inputTensor.shape[2]; // Embedding dimension

  // Flatten input to make computation easier
  const flattenedInput = inputTensor.reshape([B * L, D]);

  // Generate random centers for the hyperspheres
  const centers = tf.randomNormal([nSamples, D]);

  // Compute distance from each original point to each center
  const distanceToCenters = tf.sqrt(
    tf.sum(tf.square(tf.sub(flattenedInput.expandDims(1), centers)), 2)
  ); // Shape: (B * L) x nSamples

  // Compute angles for the great circle rotation
  const angles = tf.linspace(0, 2 * Math.PI, mSamples).tile([nSamples]); // Shape: nSamples * mSamples

  // Compute the 2D rotation matrices
  const cosAngles = tf.cos(angles);
  const sinAngles = tf.sin(angles);
  const rotationMatrices = tf.stack(
    [
      tf.stack([cosAngles, tf.neg(sinAngles)], 1),
      tf.stack([sinAngles, cosAngles], 1),
    ],
    2
  ); // Shape: nSamples * mSamples x 2 x 2

  const centerTiled = centers
    .reshape([nSamples, 1, D])
    .tile([1, mSamples, 1])
    .reshape([nSamples * mSamples, D]);

  // Perform the rotation for each original point around the hyperspheres
  const rotatedPoints = flattenedInput
    .sub(centerTiled)
    .matMul(rotationMatrices)
    .add(centerTiled)
    .reshape([B * L, nSamples, mSamples, D])
    .transpose([0, 1, 3, 2])
    .reshape([B, L, nSamples * mSamples * D]); // Shape: B x L x (nSamples * mSamples * D)

  return rotatedPoints as tf.Tensor3D;
}
