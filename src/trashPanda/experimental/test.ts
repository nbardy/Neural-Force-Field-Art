import * as tf from "@tensorflow/tfjs";
import { input, OptimizerConstructors, Tensor3D } from "@tensorflow/tfjs";
import { RotaryEmbedding } from "../embeddings/rotaryEmbeddings";

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
