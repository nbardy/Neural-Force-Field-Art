import * as tf from "@tensorflow/tfjs";
import {
  input,
  OptimizerConstructors,
  Tensor,
  Tensor3D,
} from "@tensorflow/tfjs";
import { RotaryEmbedding } from "../embeddings/rotaryEmbeddings";

export const downloadImage = (imageTensor: tf.Tensor3D, filename: string) => {
  const canvas = document.createElement("canvas");
  const [height, width, channels] = imageTensor.shape;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;

  const imageData = ctx.createImageData(width, height);
  const data = imageTensor.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;

    imageData.data[j + 0] = Math.round(255 * data[i * 3 + 0]);
    imageData.data[j + 1] = Math.round(255 * data[i * 3 + 1]);
    imageData.data[j + 2] = Math.round(255 * data[i * 3 + 2]);
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  const a = document.createElement("a");
  a.href = canvas.toDataURL();
  a.download = filename;
  a.click();
};

// Test functions for printing image series to understand Rotary Embeddings
//
// Show what the weird embeddings look like with certrain transormations
export const printMaxEmbeddingsImageSeries = () => {
  // Example Usage
  const dim = 8;
  const rotaryEmbedding = new RotaryEmbedding({ dim });
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
  downloadImage(
    imageTensorRotatedReshaped as tf.Tensor3D,
    "rotaryEmbeddings.png"
  );

  // TODO: PrintSphericalPathEmbeddings
  // TODO: PrintSphericalPathEmbeddingsRotated
  // TODO: PrintSphericalPathEmbeddingsRotated + Various features
};
