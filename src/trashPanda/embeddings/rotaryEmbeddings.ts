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
  private theta: number;
  private freqs: tf.Tensor1D;

  constructor({
    dim = 512,
    customFreqs = null,
    freqsFor = "lang",
    theta = 10000,
    maxFreq = 10,
    // TODO: Is this right? I don't think so.
    // Should anyhting be trainable
    // freqsTrainable = true,
  }) {
    this.theta = theta;
    let initFreqs;

    if (customFreqs) {
      initFreqs = customFreqs;
    } else if (freqsFor === "lang") {
      // creataes init
      initFreqs = tf.tensor1d(
        Array.from(
          { length: dim / 2 },
          (_, i) => 1 / Math.pow(theta, (2 * i) / dim)
        )
      );
    } else if (freqsFor === "pixel") {
      // this.freqs = tf.linspace(1, maxFreq / 2, dim / 2).mul(Math.PI);
      //  as parameter
      initFreqs = tf.linspace(1, maxFreq / 2, dim / 2).mul(Math.PI);
    } else {
      throw new Error(`Unknown modality ${freqsFor}`);
    }

    // TODO: What is the number 1 last arg
    // A tensor type or something wtf!?
    // this.freqs = new tf.Variable(initFreqs, true, "freqs", 1);
    this.freqs = initFreqs;
  }

  rotateQueriesOrKeys(t, seqDim = 0, offset = 0): tf.Tensor {
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
