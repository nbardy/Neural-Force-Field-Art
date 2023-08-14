import * as tf from "@tensorflow/tfjs";

//
// Rotary Embedding implementation inspired heavily by lucidrain's rotary embedding implementation
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

// lib func
function applyRotaryEmb(freqs, t, startIndex = 0, scale = 1) {
  const rotDim = freqs.shape[1];
  const endIndex = startIndex + rotDim;
  const tLeft = t.slice([0, 0], [-1, startIndex]);
  const tRight = t.slice([0, endIndex], [-1, -1]);
  const tMiddle = t.slice([0, startIndex], [-1, endIndex]);
  const rotated = tMiddle
    .mul(freqs.cos().mul(scale))
    .add(rotateHalf(tMiddle).mul(freqs.sin().mul(scale)));
  return tf.concat([tLeft, rotated, tRight], 1);
}

// Rotary embedding class
// Use this API
export class RotaryEmbedding {
  theta: number;
  freqs: tf.Tensor1D;

  constructor(
    dim,
    customFreqs = null,
    freqsFor = "lang",
    theta = 10000,
    maxFreq = 10
  ) {
    this.theta = theta;
    if (customFreqs) {
      this.freqs = customFreqs;
    } else if (freqsFor === "lang") {
      this.freqs = tf.tensor1d(
        Array.from(
          { length: dim / 2 },
          (_, i) => 1 / Math.pow(theta, (2 * i) / dim)
        )
      );
    } else if (freqsFor === "pixel") {
      this.freqs = tf.linspace(1, maxFreq / 2, dim / 2).mul(Math.PI);
    } else {
      throw new Error(`Unknown modality ${freqsFor}`);
    }
  }

  rotateQueriesOrKeys(t, seqDim = 0, offset = 0) {
    const seqLen = t.shape[seqDim];
    const freqs = tf
      .range(0, seqLen)
      .add(offset)
      .div(this.theta)
      .reshape([-1, 1])
      .tile([1, this.freqs.shape[0]])
      .mul(this.freqs);
    const freqsRotated = tf.concat([freqs.cos(), freqs.sin()], 1);
    return applyRotaryEmb(freqsRotated, t);
  }
}
