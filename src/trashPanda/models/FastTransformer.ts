import * as tf from "@tensorflow/tfjs";
import { LayersModel } from "@tensorflow/tfjs";
import { RotaryEmbedding } from "../embeddings/rotaryEmbeddings";

import { MultiHeadAttention } from "../layers/multiheadattention";
import { TrashPandaModel } from "../types";

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
export class FastTransformer implements TrashPandaModel {
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

    // TODO: Create windowed transformer that accepts the small tiles as a batch.
    // Use rotary transformer
    const compressedDim = modelDim/32;
    self.global_window_transformer = RotaryTransformer({modelDim: compressedDim})
    // cross attention to make final oatch
    self.cross_attn_w1 = CrossAttention({dim: compressedDim});
    self.cross_attn_w2 = CrossAttention({dim: compressedDim]);
    self.seld_attn_w1 = SelfAttention({dim: compressedData]);

    // TODO; modify the residual block to accept global token
    // It should also compute windowed attention early with global residual windowed cross attention.
    // and then later dds layers of global self attention, the cross attention with the last full layer reisdual.
    
    const layers = [];
    const residual = x;

    // W blocks are just repeated windowed sttention with global residual cross attention.
    // self blocks are globak self sttention with a cross attention st the end of the stack.
    // 

    const layers = [ "8xW", "1xSelf", "8xW", "2xSelfCrossLastSelf", "8xW", "4xSelfCrossLast", "4xW", "2xSelfCross", "4xW", "1xSelfCross"]

    const early_global_selfs = 4;
    for(let i = 0; i < early_global_selfs) {
        const block = new ResidualAttentionBlock({
          modelDim,
          attnHeads,
          attnHeadDim,
          dropout: dropout,
        })
      layers.push(block)
     }

    const early_windows = 8;
    for(let i = 0; i < early_windows) {
       const block = new WindowWithGlobalAttentionBlock({
          modelDim,
          attnHeads,
          attnHeadDim,
          contextDim: compressDim,
          dropout: dropout,
        })
      layers.push(block)
    }

    this.earlyWindows = seq;

    this.layers = layers;
  }

  predict(x: tf.Tensor) {
      // create 2d conv to downsample.
      // flatten to 1d
      // 1dconv to 1/4 the size.
      // split to four windows.
      // self attention across each as a batch.
      // cross attention with the global attuon
    let dx = this.downsample.apply(x)
    dx = tf.flatten(dx)
    dx = tf.1dconvshrinklayer.apply(x)
    let wdx = tf.split(dx, 4, axis=1)
    wdx = self.global_window_transformer.predict(wdx);
    const [wdx1, wdx2, wdx3, wdx4] = wdx;
    let global = self.cross_attn_w1(wdx1, wdx2) + self.cross_ttn_w2(wdx3, wdx4)

    // todo eval stack as seq, but maintain a full global residual around the fast local layers and cross attention to it sfter the self sttentuon laywrs.
    // for the fast windowed layers just drop them with the small windows fast, then fast cross attention on the windows siht the global resid.
    
    return this.layers.apply(x);
  }
}
