import * as tf from "@tensorflow/tfjs";
import { normalize } from "../linalg";

/*
 * Playing with some ideas for geometric embeddings
 * i tink im 
 */


export function rotateEmbedding2D(
  inputTensor: tf.Tensor2D, // B x 2
  n: number, // Number of random rotation magnitudes
  m: number, // Sample rate around each rotation
  offsetCenterOpt: boolean // Optional random offset center
): tf.Tensor3D {
  const B = inputTensor.shape[0];
  const rotations = tf.linspace(0, 2 * Math.PI, m);
  const randomMagnitudes = tf.randomUniform([n], -1, 1);
  const angles = tf.outerProduct(randomMagnitudes, rotations); // Shape: n x m

  const cosAngles = tf.cos(angles).reshape([n * m]); // n*m
  const sinAngles = tf.sin(angles).reshape([n * m]); // n*m

  const rotationMatrices = tf.stack([
    tf.stack([cosAngles, tf.neg(sinAngles)], 1),
    tf.stack([sinAngles, cosAngles], 1),
  ], 2); // Shape: n*m x 2 x 2

  let offsetCenter = tf.zeros([1, 2]);
  if (offsetCenterOpt) {
    offsetCenter = tf.randomNormal([1, 2]);
  }

  const tiledInput = inputTensor.add(offsetCenter).tile([n * m, 1]); // Shape: n*m*B x 2
  const rotatedPoints = tiledInput.matMul(rotationMatrices); // Shape: n*m*B x 2

  return rotatedPoints.reshape([n, m, B, 2]).transpose([2, 0, 1, 3]); // Shape: B x n x m x 2
}


//old 


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

