import * as tf from "@tensorflow/tfjs";
import { pt } from "../../physics/updateParticles";

//
// Rotary Embedding implementation inspired heavily by lucidrain's rotary embedding implementation
// https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
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

function applyRotaryEmb(freqs, t, startIndex = 0, scale = 1) {
  const rotDim = freqs.shape[1];
  const endIndex = startIndex + rotDim;

  // [Change] Expand dimensions to make them 3D for consistency
  const tLeft = t.slice([0, 0], [-1, startIndex]).expandDims(2); // BxCx1
  const tRight = t.slice([0, endIndex], [-1, -1]).expandDims(2); // BxCx1
  const tMiddle = t.slice([0, startIndex], [-1, endIndex]).expandDims(2); // BxCx1

  pt("tMiddle", tMiddle);

  // Apply rotary embeddings
  const scaledFreqs = freqs.cos().mul(scale).expandDims(2); // BxCx1

  pt("scaledFreqs", scaledFreqs);

  // [Change] Squeeze the last dimension of tMiddle for rotateHalf and then expand it back
  const rotated = tMiddle
    .squeeze(2) // BxC
    .mul(scaledFreqs) // BxCx1
    .add(rotateHalf(tMiddle.squeeze(2)).expandDims(2).mul(scaledFreqs.sin())); // BxCx1

  pt("rotated", rotated);
  pt("tLeft", tLeft);

  // [Change] All tensors are now 3D and can be concatenated along axis 1
  return tf.concat([tLeft, rotated, tRight], 1); // BxCx1
}

// Rotary embedding class
// Use this API
// RotaryEmbedding class
export class RotaryEmbedding {
  private theta: number;
  private freqs: tf.Variable;
  private cache: { [key: string]: tf.Tensor } = {};
  private learnedFreq: boolean;

  constructor({
    dim = 512,
    customFreqs = null,
    freqsFor = "lang",
    theta = 10000,
    maxFreq = 10,
    learnedFreq = false, // New parameter to control whether freqs are trainable
  }) {
    this.theta = theta;
    this.learnedFreq = learnedFreq; // Store the setting

    let initFreqs;
    if (customFreqs) {
      initFreqs = customFreqs;
    } else if (freqsFor === "lang") {
      const arr = Array.from(
        { length: dim / 2 },
        (_, i) => 1 / Math.pow(theta, (2 * i) / dim)
      );
      initFreqs = tf.tensor1d(arr);
    } else if (freqsFor === "pixel") {
      initFreqs = tf.linspace(1, maxFreq / 2, dim / 2).mul(Math.PI);
    } else {
      throw new Error(`Unknown modality ${freqsFor}`);
    }

    // Make freqs a trainable variable if learnedFreq is true
    this.freqs = this.learnedFreq ? tf.variable(initFreqs) : initFreqs;
  }

  private getSeqPos(
    seqLen: number,
    dtype: "float32" | "int32",
    offset = 0
  ): tf.Tensor {
    return tf.range(0, seqLen, 1, dtype).add(offset).div(this.theta);
  }

  rotateQueriesOrKeys(t: tf.Tensor, seqDim = 0, offset = 0): tf.Tensor {
    const seqLen = t.shape[seqDim];
    const freqs = this.getSeqPos(seqLen, t.dtype as "float32" | "int32", offset)
      .reshape([-1, 1])
      .tile([1, this.freqs.shape[0]])
      .mul(this.freqs);

    const freqsRotated = tf.concat([freqs.cos(), freqs.sin()], 1);
    return applyRotaryEmb(freqsRotated, t);
  }
}
