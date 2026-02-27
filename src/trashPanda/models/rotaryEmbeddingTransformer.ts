import * as tf from "@tensorflow/tfjs";
import { LayersModel } from "@tensorflow/tfjs";
import { RotaryEmbedding } from "../embeddings/rotaryEmbeddings";

import { MultiHeadAttention } from "../layers/multiheadattention";
import { Module } from "../types";

class ResidualAttentionBlock {
  attn: any; // BxCxWxH
  ln_1: any; // BxCxWxH
  mlp: any; // BxCxWxH
  ln_2: any; // BxCxWxH

  rotaryEmbedding: RotaryEmbedding;

  constructor({
    modelDim: modelDim = 64,
    attnHeads: attnHeads = 8,
    attnHeadDim: attnHeadDim = 64,
    dropout = 0,
  }: {
    modelDim?: number;
    attnHeads?: number;
    attnHeadDim?: number;
    dropout?: number;
  } = {}) {
    this.attn = new MultiHeadAttention({
      heads: attnHeads,
      dim: modelDim,
      dimHead: attnHeadDim,
      dropout: dropout,
    });

    this.rotaryEmbedding = new RotaryEmbedding({
      dim: modelDim,
    });

    this.ln_1 = tf.layers.layerNormalization({ axis: -1 });
    this.mlp = tf.sequential({
      layers: [
        tf.layers.dense({ units: modelDim * 4, activation: "relu" }),
        tf.layers.dense({ units: modelDim }),
      ],
    });
    this.ln_2 = tf.layers.layerNormalization({ axis: -1 });
  }

  attention(x: tf.Tensor) {
    const withRotaryEmbedding = this.rotaryEmbedding.rotateQueriesOrKeys(x);
    return this.attn.apply([x, x, x]); // BxSxD
  }

  forward(x: tf.Tensor) {
    let y = this.ln_1.apply(x);
    y = this.attention(y);
    x = tf.add(x, y); // BxSxD

    y = this.ln_2.apply(x);
    y = this.mlp.apply(y);
    x = tf.add(x, y); // BxSxD

    return x;
  }
}

// Decoder only transformer used for GPT/CLIP
export class RotaryTransformer implements Module {
  layers; // BxSxD

  constructor({
    modelDim = 64,
    attnHeads = 8,
    attnHeadDim = 64,
    dropout = 0,
    depth: depth = 6,
  }: {
    modelDim?: number;
    attnHeads?: number;
    attnHeadDim?: number;
    dropout?: number;
    depth?: number;
  } = {}) {
    const layers = [];
    for (let i = 0; i < depth; i++) {
      this.layers.push(
        new ResidualAttentionBlock({
          modelDim,
          attnHeads,
          attnHeadDim,
          dropout: dropout,
        })
      );
    }

    this.layers = tf.sequential({ layers: layers });
  }

  predict(x: tf.Tensor) {
    return this.layers.apply(x);
  }
}
