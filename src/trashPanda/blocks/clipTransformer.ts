import * as tf from "@tensorflow/tfjs";
import { LayersModel } from "@tensorflow/tfjs";

import { MultiHeadAttention } from "../layers/multiheadattention";

class ResidualAttentionBlock {
  attn: any; // BxCxWxH
  ln_1: any; // BxCxWxH
  mlp: any; // BxCxWxH
  ln_2: any; // BxCxWxH

  constructor({
    d_model = 64,
    attn_heads = 8,
    attn_head_dim = 64,
    dropout = 0,
  }: {
    d_model?: number;
    attn_heads?: number;
    attn_head_dim?: number;
    dropout?: number;
  } = {}) {
    this.attn = new MultiHeadAttention({
      heads: attn_heads,
      dim: d_model,
      dimHead: attn_head_dim,
      dropout: dropout,
    });

    this.ln_1 = tf.layers.layerNormalization({ axis: -1 });
    this.mlp = tf.sequential({
      layers: [
        tf.layers.dense({ units: d_model * 4, activation: "relu" }),
        tf.layers.dense({ units: d_model }),
      ],
    });
    this.ln_2 = tf.layers.layerNormalization({ axis: -1 });
  }

  attention(x: tf.Tensor) {
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
export class Transformer {
  layers; // BxSxD

  constructor({
    d_model = 64,
    attn_heads = 8,
    attn_head_dim = 64,
    dropout = 0,
    num_layers = 6,
  }: {
    d_model?: number;
    attn_heads?: number;
    attn_head_dim?: number;
    dropout?: number;
    num_layers?: number;
  } = {}) {
    const layers = [];
    for (let i = 0; i < num_layers; i++) {
      this.layers.push(
        new ResidualAttentionBlock({
          d_model: d_model,
          attn_heads: attn_heads,
          attn_head_dim: attn_head_dim,
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
