import * as tf from "@tensorflow/tfjs";
import { Tensor3D } from "@tensorflow/tfjs";
import { Transformer } from "../blocks/clipTransformer";
import { RotaryEmbedding } from "../embeddings/rotaryEmbeddings";
import { TrashPandaModel } from "../types";

export class MaxEmbeddingTransformer implements TrashPandaModel {
  embeddingLayer: MaxPointEmbeddingLayer;
  name: string;
  transformer: Transformer;

  constructor(args: { modelDim: number; depth: number }) {
    const { modelDim, depth } = args;

    this.name = "maxEmbeddingTransformer";
    this.embeddingLayer = new MaxPointEmbeddingLayer({
      numSamples: 24,
      a: 1, // TODO: What is this?
      NSize: 2, // TODO: Hookup points
      finalLayerSize: 512,
      useGaussianKernel: true,
      useFFT: true,
    });

    this.transformer = new Transformer({
      depth,
      modelDim,
    });
  }

  predict(inputTensor: tf.Tensor2D) {
    const embedding = this.embeddingLayer.predict(inputTensor);
    const transformerOutput = this.transformer.predict(embedding);
    return transformerOutput;
  }
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

type Dense = ReturnType<typeof tf.layers.dense>;

class MaxPointEmbeddingLayer {
  // stores layers
  layers = [];
  useGaussianKernel = false;
  useFFT = true;
  finalLayerSize = 512;
  a = 1;
  numSamples = 100;
  denseLayer: Dense;
  conv2dlayer: ReturnType<typeof tf.layers.conv2d> | null = null;

  //constructorA
  constructor({
    numSamples = 24,
    a,
    NSize = 2, // 2D or 3D Point
    finalLayerSize = 512,
    useFFT = true,
    conv2x = true,
    useGaussianKernel = false,
  }) {
    // params
    const denseLayer = tf.layers.dense({ units: finalLayerSize });
    this.denseLayer = denseLayer;

    if (conv2x) {
      const conv2dlayer = tf.layers.conv2d({
        filters: NSize * 2,
        kernelSize: [1, 1],
        useBias: false,
      });
      this.conv2dlayer = conv2dlayer;
    }

    //hyperParams
    this.numSamples = numSamples;
    this.a = a;
    this.finalLayerSize = finalLayerSize;
    this.useFFT = useFFT;
    this.useGaussianKernel = useGaussianKernel;
  }

  predict(inputTensor: tf.Tensor2D) {
    const { numSamples, a, conv2dlayer, useFFT, useGaussianKernel } = this;
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
    if (this.conv2dlayer) {
      outputs = this.conv2dlayer.apply(outputs) as tf.Tensor; // Shape: Bx(3)xSx3
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
    finalOutput = this.denseLayer.apply(finalOutput) as tf.Tensor; // Shape: B x finalLayerSize

    return finalOutput;
  }
}
