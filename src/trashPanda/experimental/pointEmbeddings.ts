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
import {
  input,
  OptimizerConstructors,
  Sequential,
  Tensor,
  Tensor3D,
} from "@tensorflow/tfjs";
import { RotaryEmbedding } from "../embeddings/rotaryEmbeddings";
import { Transformer } from "../models/clipTransformer";
import { RotaryTransformer } from "../models/rotaryEmbeddingTransformer";
import { TrashPandaModel } from "../types";

//

class LinearExpandDims implements TrashPandaModel {
  layer: Sequential;
  constructor(args: {
    inputDim: number;
    hiddenDim: number;
    outputDim: number;
    depth: number;
  }) {
    const { inputDim, outputDim } = args;
    this.layer = tf.sequential();
    this.layer.add(tf.layers.dense({ units: inputDim, activation: "relu" }));
    for (let i = 0; i < args.depth; i++) {
      this.layer.add(
        tf.layers.dense({ units: args.hiddenDim, activation: "relu" })
      );
    }
    this.layer.add(tf.layers.dense({ units: outputDim, activation: "relu" }));
  }
  predict(input: Tensor) {
    return this.layer.apply(input);
  }
}

//
// Max Embedding Transformer
//
// A transformer with all the tricks
export class PointEmbeddingTransformer {
  pointEmbeddingLayer: Sequential;
  transformer: Transformer | RotaryTransformer;

  constructor(args: {
    modelDim: number;
    depth: number;
    useRotaryTransformer?: boolean;
  }) {
    // Create Layers
    this.transformer = args.useRotaryTransformer
      ? new RotaryTransformer(args)
      : new Transformer(args);
    this.pointEmbeddingLayer = new LinearExpandDims({
      inputDim: 2,
      hiddenDim: 64,
      outputDim: args.modelDim,
      depth: 2,
    });
  }

  // Inference
  predict(input: Tensor) {
    const x = this.pointEmbeddingLayer.predict(input);
    return this.transformer.predict(x);
  }
}
