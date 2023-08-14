/**
 * Experiments on point embeddings.
 *
 * I am training a neural network on batches of 2D/3D points. BxN
 *
 *  We want to scale up our model, but the number of parameters is limted
 *  by the number of points in the input.
 *
 *  It would be more apropiate to use RL algorithms to fix this, but this is an art project and
 *  we are trying to visualize models not win.
 *
 *  Instead we expand the dimensions of the tensor, we don't want to learn a perfect
 *  algorithm we want to add enough information to the data to allow the model to
 *  learn slowly and watch learning.
 *
 *  Some things I'm trying
 *  1. Hand crafted features(fft, gaussiankernel, etc...)
 *  2. Repeating the point in a spiral
 *  3. Using positional encoding from transformers(rotary embedding)
 *  4. Using MLP to expand the dimensions
 *
 *  Also adding some dilated repeating patterns to duplicate the
 *
 */
import * as tf from "@tensorflow/tfjs";
import { Tensor3D } from "@tensorflow/tfjs";

import { gaussianKernel } from "../kernels";

//
// Basic linear expand
//
function expandNeuralNetwork(inputTensor) {
  // Bx2
  const layer1 = tf.layers
    .dense({ units: 128, activation: "relu" })
    .apply(inputTensor); // Bx128
  const layer2 = tf.layers
    .dense({ units: 256, activation: "relu" })
    .apply(layer1); // Bx256
  const outputTensor = tf.layers.dense({ units: 512 }).apply(layer2); // Bx512
  return outputTensor;
}

function spiralCurve(t: tf.Tensor, inputTensor: tf.Tensor2D, a: number) {
  const cost = tf.cos(t);
  const sint = tf.sin(t);
  const denominator = tf.sqrt(tf.add(1, tf.mul(a ** 2, tf.square(t))));

  const x = tf.div(cost, denominator).expandDims(1);
  const y = tf.div(sint, denominator).expandDims(1);
  const z = tf.div(tf.mul(tf.neg(a), t), denominator).expandDims(1);

  const curvePoints = tf.concat([x, y, z], 1); // Shape: numSamplesx3

  return tf.add(curvePoints, inputTensor); // Adding inputTensor as center
}

//
// Rotary Embedding
//

// lib func
function rotateHalf(x) {
  const xReshaped = tf.reshape(x, [-1, x.shape[1] / 2, 2]);
  const [x1, x2] = tf.split(xReshaped, 2, 2);
  const negativeX2 = tf.neg(x2);
  const rotatedPart = tf.concat([negativeX2, x1], 2);
  const flattenedRotated = tf.reshape(rotatedPart, [-1, x.shape[1]]);
  return flattenedRotated;
}

// lib func
function applyRotaryEmb(freqs, t, startIndex = 0, scale = 1) {
  const rotDim = freqs.shape[1];
  const endIndex = startIndex + rotDim;
  const tLeft = t.slice([0, 0], [-1, startIndex]);
  const tRight = t.slice([0, endIndex], [-1, -1]);
  const tMiddle = t.slice([0, startIndex], [-1, endIndex]);
  const rotated = tMiddle
    .mul(freqs.cos().mul(scale))
    .add(rotateHalf(tMiddle).mul(freqs.sin().mul(scale)));
  return tf.concat([tLeft, rotated, tRight], 1);
}

// Rotary embedding class
// Use this API
export class RotaryEmbedding {
  theta: number;
  freqs: tf.Tensor1D;

  constructor(
    dim,
    customFreqs = null,
    freqsFor = "lang",
    theta = 10000,
    maxFreq = 10
  ) {
    this.theta = theta;
    if (customFreqs) {
      this.freqs = customFreqs;
    } else if (freqsFor === "lang") {
      this.freqs = tf.tensor1d(
        Array.from(
          { length: dim / 2 },
          (_, i) => 1 / Math.pow(theta, (2 * i) / dim)
        )
      );
    } else if (freqsFor === "pixel") {
      this.freqs = tf.linspace(1, maxFreq / 2, dim / 2).mul(Math.PI);
    } else {
      throw new Error(`Unknown modality ${freqsFor}`);
    }
  }

  rotateQueriesOrKeys(t, seqDim = 0, offset = 0) {
    const seqLen = t.shape[seqDim];
    const freqs = tf
      .range(0, seqLen)
      .add(offset)
      .div(this.theta)
      .reshape([-1, 1])
      .tile([1, this.freqs.shape[0]])
      .mul(this.freqs);
    const freqsRotated = tf.concat([freqs.cos(), freqs.sin()], 1);
    return applyRotaryEmb(freqsRotated, t);
  }
}

// Download Image
export const downloadImage = (img: Tensor3D, name: string) => {
  const a = document.createElement("a");
  document.body.appendChild(a);
  a.style.display = "none";
  const url = window.URL.createObjectURL(
    new Blob([img.dataSync()], { type: "application/octet-stream" })
  );
  a.href = url;
  a.download = name;
  a.click();
  window.URL.revokeObjectURL(url);
};

// Show what the weird embeddings look like with certrain transormations
export const printMaxEmbeddingsImageSeries = () => {
  // Example Usage
  const dim = 8;
  const rotaryEmbedding = new RotaryEmbedding(dim);
  const tensor = tf.tensor2d([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]);
  const rotatedTensor = rotaryEmbedding.rotateQueriesOrKeys(tensor); // Shape: Bx8

  // print
  console.log("print simple");
  tensor.print(true);
  rotatedTensor.print(true);

  // Call the function to compute and print the rotary embeddings
  console.log("applyRotaryEmbeddings");

  // Now let's make an image. Let's create rotary embeddings for each x,y coordinate
  // from 400x400 image
  // Then For each embedding Let's print it as a color.

  // Create a 400x400 image
  const imageWidth = 400;
  const imageHeight = 400;
  const imageTensor = tf.linspace(0, 1, imageWidth * imageHeight);
  const imageTensorReshaped = imageTensor.reshape([imageHeight, imageWidth, 1]);
  const imageTensorReshapedTiled = imageTensorReshaped.tile([1, 1, dim]);

  // Create the rotary embeddings for each x,y coordinate
  const imageTensorRotated = rotaryEmbedding.rotateQueriesOrKeys(
    imageTensorReshapedTiled,
    0,
    0
  );

  // Print the image
  const imageTensorRotatedReshaped = imageTensorRotated.reshape([
    imageHeight,
    imageWidth,
    dim,
  ]);

  // Download the image
  downloadImage(imageTensorRotatedReshaped, "rotaryEmbeddings.png");

  // TODO: PrintSphericalPathEmbeddings
  // TODO: PrintSphericalPathEmbeddingsRotated
  // TODO: PrintSphericalPathEmbeddingsRotated + Various features
};

function maxEmbedding(
  inputTensor: tf.Tensor2D,
  numSamples: number,
  a: number,
  finalLayerSize = 512,
  useFFT = true,
  conv2x = true,
  useGaussianKernel = false
) {
  // Input: BxN
  const B = inputTensor.shape[0];
  const N = inputTensor.shape[1];

  // Generate spherical curve points
  const t = tf.linspace(0, 0.7, numSamples);
  const curvePoints = spiralCurve(t, inputTensor, a); // Shape: BxnumSamplesx3

  // + Spherical coords
  const xyz = curvePoints.reshape([B, numSamples, 3]); // Shape: BxSx3
  const r = tf.norm(xyz, "euclidean", -1); // Shape: BxS

  // Options Bx(1|2)xN
  const theta = tf.acos(
    xyz.slice([0, 0, 2], [-1, -1, 1]).div(r.expandDims(-1))
  ); // Shape: BxS
  const phi = tf.atan2(
    xyz.slice([0, 0, 1], [-1, -1, 1]),
    xyz.slice([0, 0, 0], [-1, -1, 1])
  ); // Shape: BxS
  const sphericalCoords = tf.stack([r, theta, phi], -1); // Shape: BxSx3

  // +Samples S times
  // Already in the shape BxSx(1|2)xN

  // +Rotary Embedding 3x
  const frequencies = [5231, 7819, 9212];
  const rotaryEmbeddings = frequencies.map((freq) => {
    const rotary = new RotaryEmbedding(
      sphericalCoords.shape[-1], // Corrected this line
      null,
      "pixel",
      freq
    );
    return rotary.rotateQueriesOrKeys(sphericalCoords); // Corrected this line, Shape: BxSx3
  }); // Shape: 3xBxSx3

  // + 2xConv
  let outputs = tf.stack(rotaryEmbeddings, 1); // Shape: Bx3xSx3
  if (conv2x) {
    outputs = tf.layers
      .conv2d({
        filters: outputs.shape[-1] * 2,
        kernelSize: [1, 1],
        useBias: false,
      })
      .apply(outputs) as tf.Tensor; // Shape: Bx(3)xSx(6)
  }

  // +useFFT|gaussianKernl
  if (useFFT) {
    outputs = tf.spectral.fft(outputs); // Shape: Bx((1|2|3)x(1|2)x(1|2))x3xSxN
  }

  if (useGaussianKernel) {
    const kernelSize = 5; // Can be adjusted
    const sigma = 1.0; // Can be adjusted

    // Create a 1D Gaussian kernel
    const kernel1D = tf
      .range(-kernelSize / 2, kernelSize / 2 + 1)
      .div(kernelSize)
      .mul(-1)
      .square()
      .div(2 * sigma * sigma)
      .exp();
    const normalizedKernel = kernel1D.div(kernel1D.sum());

    // Reshape kernel to 3D
    const reshapedKernel: Tensor3D = normalizedKernel.reshape([
      kernelSize,
      1,
      1,
    ]);

    // Check if outputs is of rank 3
    if (outputs.rank === 3) {
      // Apply the 1D convolution
      outputs = tf.conv1d(
        outputs as tf.Tensor3D,
        reshapedKernel,
        1,
        "same"
      ) as tf.Tensor3D; // Output shape: BxSx(1|2)xN
    } else {
      // Error handling or reshaping logic if needed
    }
  }
  // Final shape BxMx3xSxN , where M = (1,2,4,8,24)

  // Flatten
  let finalOutput = tf.layers.flatten().apply(outputs) as tf.Tensor;
  // Shape: BxK where K = Mx3xSxN
  // range for N = 2,3 S = 24 and M = 1,2,4,8,24, K = 24x3x24x2 = 3456

  // Linear transformation to get the final layer size
  const denseLayer = tf.layers.dense({ units: finalLayerSize });
  finalOutput = denseLayer.apply(finalOutput) as tf.Tensor; // Shape: B x finalLayerSize

  return finalOutput;
}
