import * as tf from "@tensorflow/tfjs";

// inspired by lucidrains' implementation in his flash attention repo
export class MultiHeadAttention {
  dim: number;
  heads: number;
  dimHead: number;
  innerDim: number;

  // params
  toQ: any;
  toKV: any;
  toOut: any;

  constructor({ dim, heads = 8, dimHead = 64, dropout = 0.0, causal = false }) {
    this.dim = dim;
    this.heads = heads;
    this.dimHead = dimHead;
    this.innerDim = heads * dimHead;
    this.toQ = tf.layers.dense({ units: this.innerDim, useBias: false });
    this.toKV = tf.layers.dense({ units: this.innerDim * 2, useBias: false });
    this.toOut = tf.layers.dense({ units: dim, useBias: false });
  }

  attention(q, k, v, attnBias = null) {
    // Scale
    const scale = Math.pow(q.shape[q.shape.length - 1], -0.5);
    q = q.mul(scale);

    // Similarity
    let sim = tf.matMul(q, k, false, true);

    // Bias handling
    if (attnBias !== null) {
      sim = sim.add(attnBias);
    }

    sim = sim.sub(sim.max(-1, true));

    const attn = tf.softmax(sim, -1);
    const out = tf.matMul(attn, v); // BxHxWxD

    return out;
  }

  forward(x, context = null, mask = null, attnBias = null) {
    const h = this.heads;
    context = context === null ? x : context;

    let q = this.toQ.apply(x);
    let [k, v] = tf.split(this.toKV.apply(context), 2, -1);

    [q, k, v] = [q, k, v].map((t) =>
      tf.reshape(t, [-1, h, t.shape[1], this.dimHead])
    );

    const out = this.attention(q, k, v, attnBias);
    const outShape = out.shape[1]!;

    const result = tf.reshape(out, [-1, outShape, h * this.dimHead]);

    return this.toOut.apply(result);
  }
}
