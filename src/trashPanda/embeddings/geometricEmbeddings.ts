import * as tf from "@tensorflow/tfjs";

/*
 * Playing with some ideas for geometric embeddings
 * i tink im 
 */


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

/**
 * Rotates the points in the hyperspherical embedding space.
 *
 * @param inputTensor The input tensor with shape B x L x D, where B is the batch size, L is the sequence length, and D is the dimensionality.
 * @param angle The angle of rotation in radians.
 * @param axis The axis around which the points will be rotated. Must be a unit vector with shape D.
 * @returns The rotated tensor with shape B x L x D.
 */
export function rotateHypersphericalEmbedding(inputTensor: tf.Tensor3D, angle: number, axis: tf.Tensor1D): tf.Tensor3D {
  const cosAngle = tf.scalar(Math.cos(angle));
  const sinAngle = tf.scalar(Math.sin(angle));
  const oneMinusCosAngle = tf.scalar(1).sub(cosAngle);

  const axisComponents = tf.unstack(axis);
  const [u, v, w] = axisComponents.map((comp) => comp.reshape([]));

  // Create the rotation matrix using the angle and axis
  const rotationMatrix = tf.stack([
    u.mul(u).mul(oneMinusCosAngle).add(cosAngle),
    u.mul(v).mul(oneMinusCosAngle).sub(w.mul(sinAngle)),
    u.mul(w).mul(oneMinusCosAngle).add(v.mul(sinAngle)),
    v.mul(u).mul(oneMinusCosAngle).add(w.mul(sinAngle)),
    v.mul(v).mul(oneMinusCosAngle).add(cosAngle),
    v.mul(w).mul(oneMinusCosAngle).sub(u.mul(sinAngle)),
    w.mul(u).mul(oneMinusCosAngle).sub(v.mul(sinAngle)),
    w.mul(v).mul(oneMinusCosAngle).add(u.mul(sinAngle)),
    w.mul(w).mul(oneMinusCosAngle).add(cosAngle),
  ]).reshape([3, 3]); // Shape: 3 x 3

  // Apply the rotation matrix to the input tensor
  const rotatedPoints = inputTensor.matMul(rotationMatrix); // Shape: B x L x D

  return rotatedPoints as tf.Tensor3D;
}
