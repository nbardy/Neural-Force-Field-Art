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
 *  What worked in the past: (Simple MLP works at low and high dimensions for simple objectives)
 *  Using MLP to expand the dimensions(https://github.com/nbardy/force-field-ml-art/blob/master/src/models.js#L61)
 *
 *  Some things I'm trying now:
 *  1. Hand crafted features(fft, gaussiankernel, etc...)
 *  2. Repeating the point in a spiral
 *  3. Using positional encoding from transformers(rotary embedding)
 *
 */
import * as tf from "@tensorflow/tfjs";
import { input, OptimizerConstructors, Tensor3D } from "@tensorflow/tfjs";
import { RotaryEmbedding } from "./rotaryEmbeddings";

//
// Basic linear expand (Baseline)
export function expandNeuralNetwork(inputTensor) {
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

//
// Max Embedding Transformer
//
// A transformer with all the tricks
//
export function maxEmbeddingTransformer(inputTensor) {
  // Bx2
  return maxEmbeddingTransformer(512);
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

//
// Test functions for printing image series to understand Rotary Embeddings
//
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
