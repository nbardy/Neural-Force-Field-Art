import * as tf from "@tensorflow/tfjs";

/*
 * Playing with some ideas for geometric embeddings
 */

import * as tf from "@tensorflow/tfjs";

/**
 * Rotated in random spherical directions. (Should maintain length)
 */
export function rotateSphericalEmbedding(inputTensor, numDirections) {
  const dim = inputTensor.shape[1];
  const randomDirections = tf.randomNormal([numDirections, dim, dim]); // Shape: numDirections x dim x dim
  const [q, _] = tf.linalg.qr(randomDirections);

  const expandedInput = tf.expandDims(inputTensor, 0); // Shape: 1 x B x dim
  const tiledInput = tf.tile(expandedInput, [numDirections, 1, 1]); // Shape: numDirections x B x dim
  const embeddings = tf.matMul(tiledInput, q, false, true); // Shape: numDirections x B x dim
  const sphericalEmbeddings = tf.reshape(embeddings, [-1, numDirections * dim]); // Shape: B x (numDirections * dim)

  return sphericalEmbeddings;
}

/**
 * Rotated in full disc sampled m times around a great circle around n hyperspheres.
 */
function rotateHypersphereOffsetCenter(
  inputTensor: tf.Tensor3D,
  nSamples: number,
  mSamples: number
): tf.Tensor3D {
  const B = inputTensor.shape[0]; // B x L x D
  const L = inputTensor.shape[1];
  const D = inputTensor.shape[2];

  const flattenedInput = inputTensor.reshape([B * L, D]);
  const centers = tf.randomNormal([nSamples, D]);
  const distanceToCenters = tf.sqrt(
    tf.sum(tf.square(tf.sub(flattenedInput.expandDims(1), centers)), 2)
  ); // Shape: (B * L) x nSamples
  const angles = tf.linspace(0, 2 * Math.PI, mSamples).tile([nSamples]);
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

  const rotatedPoints = flattenedInput
    .sub(centerTiled)
    .matMul(rotationMatrices)
    .add(centerTiled)
    .reshape([B * L, nSamples, mSamples, D])
    .transpose([0, 1, 3, 2])
    .reshape([B, L, nSamples * mSamples * D]); // Shape: B x L x (nSamples * mSamples * D)

  return rotatedPoints as tf.Tensor3D;
}

/**
 * Simple rotation embedding in 3D for hyper sphere
 */
export function rotateHypersphereEmbedding(inputTensor: tf.Tensor3D, angle: number): tf.Tensor3D {
  const cosAngle = tf.scalar(Math.cos(angle));
  const sinAngle = tf.scalar(Math.sin(angle));

  // Rotation matrix for 3D
  const rotationMatrix = tf.tensor([
    [cosAngle, tf.neg(sinAngle), 0],
    [sinAngle, cosAngle, 0],
    [0, 0, 1]
  ]); // Shape: 3 x 3

  // Apply the 3D rotation
  const rotatedPoints = inputTensor.matMul(rotationMatrix); // Shape: B x L x D

  return rotatedPoints as tf.Tensor3D;
}

/**
 * Complex off-center rotation embedding for spherical
 */
export function rotateSphericalOffsetCenter(inputTensor: tf.Tensor3D, offset: tf.Tensor1D, angle: number): tf.Tensor3D {
  const B = inputTensor.shape[0]; // B x L x D
  const L = inputTensor.shape[1];
  const D = inputTensor.shape[2];

  const cosAngle = tf.scalar(Math.cos(angle));
  const sinAngle = tf.scalar(Math.sin(angle));

  // Rotation matrix for 3D
  const rotationMatrix = tf.tensor([
    [cosAngle, tf.neg(sinAngle), 0],
    [sinAngle, cosAngle, 0],
    [0, 0, 1]
  ]); // Shape: 3 x 3

  // Subtract offset, apply rotation, and add offset back
  const offsetExpanded = offset.reshape([1, 1, D]).tile([B, L, 1]); // Shape: B x L x D
  const rotatedPoints = inputTensor.sub(offsetExpanded)
    .matMul(rotationMatrix)
    .add(offsetExpanded); // Shape: B x L x D

  return rotatedPoints as tf.Tensor3D;
}
