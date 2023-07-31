import * as tf from "@tensorflow/tfjs";

function exists(val) {
  return val !== undefined && val !== null;
}

function rotateHalf(x) {
  let shape = x.shape;
  let d = shape[shape.length - 2];
  let r = 2;

  let [x1, x2] = tf.split(x, [d, d], -2);

  let stack = tf.stack([-x2, x1], -1);
  return tf.reshape(stack, shape);
}

function applyRotaryEmb(freqs, t, startIndex = 0, scale = 1) {
  const tDevice = t.device || "CPU";
  freqs = freqs.clone().moveToDevice(tDevice);

  let rotDim = freqs.shape[freqs.shape.length - 1];
  let endIndex = startIndex + rotDim;
  if (rotDim > t.shape[t.shape.length - 1]) {
    throw new Error(
      `Feature dimension ${
        t.shape[t.shape.length - 1]
      } is not of sufficient size to rotate in all the positions ${rotDim}`
    );
  }

  let tLeft = t.slice(
    [0, 0, 0, 0, 0, startIndex],
    [-1, -1, -1, -1, -1, -startIndex]
  );
  t = t.slice(
    [0, 0, 0, 0, 0, startIndex],
    [-1, -1, -1, -1, -1, endIndex - startIndex]
  );
  let tRight = t.slice(
    [0, 0, 0, 0, 0, endIndex],
    [-1, -1, -1, -1, -1, -endIndex]
  );

  let rotated = t
    .mul(freqs.cos().mul(scale))
    .add(rotateHalf(t).mul(freqs.sin().mul(scale)));

  return tf.concat([tLeft, rotated, tRight], -1);
}

// Main Rotary Embedding Class

class RotaryEmbedding {
  constructor(params) {
    this.dim = params.dim;
    this.customFreqs = params.customFreqs;
    this.freqsFor = params.freqsFor || "lang";
    this.theta = params.theta || 10000;
    this.maxFreq = params.maxFreq || 10;
    this.numFreqs = params.numFreqs || 1;
    this.learnedFreq = params.learnedFreq || false;
    this.useXpos = params.useXpos || false;
    this.xposScaleBase = params.xposScaleBase || 512;
    this.interpolateFactor = params.interpolateFactor || 1;

    if (exists(this.customFreqs)) {
      this.freqs = tf.tensor(this.customFreqs);
    } else if (this.freqsFor === "lang") {
      this.freqs = tf.div(
        1,
        tf.pow(this.theta, tf.range(0, this.dim, 2).div(this.dim))
      );
    } else if (this.freqsFor === "pixel") {
      this.freqs = tf.linspace(1, this.maxFreq / 2, this.dim / 2).mul(Math.PI);
    } else if (this.freqsFor === "constant") {
      this.freqs = tf.ones([this.numFreqs]);
    } else {
      throw new Error(`Unknown modality ${this.freqsFor}`);
    }

    if (this.learnedFreq) {
      this.freqs = tf.variable(this.freqs);
    }

    this.cache = {};

    if (!this.useXpos) {
      this.scale = null;
    } else {
      this.scale = tf.pow(
        this.xposScaleBase,
        tf
          .range(0, this.dim, 2)
          .add(0.4 * this.dim)
          .div(1.4 * this.dim)
      );
    }
  }

  getSeqPos(seqLen) {
    return tf.range(0, seqLen).div(this.interpolateFactor);
  }

  rotateQueriesOrKeys(t, seqDim = -2, offset = 0) {
    if (this.useXpos) {
      throw new Error(
        "You must use `.rotateQueriesAndKeys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"
      );
    }

    let seqLen = t.shape[seqDim];
    let freqs = this.forward(() => this.getSeqPos(seqLen), `freqs:${seqLen}`);

    return applyRotaryEmb(freqs, t);
  }

  rotateQueriesAndKeys(q, k, seqDim = -2) {
    if (!this.useXpos) {
      throw new Error("You must set `useXpos` to true to use this method");
    }

    let seqLen = q.shape[seqDim];
    let seq = this.getSeqPos(seqLen);
    let freqs = this.forward(() => seq, `freqs:${seqLen}`);
    let scale = this.getScale(() => seq, `scale:${seqLen}`);

    let rotatedQ = applyRotaryEmb(freqs, q, 0, scale);
    let rotatedK = applyRotaryEmb(freqs, k, 0, tf.div(1, scale));

    return [rotatedQ, rotatedK];
  }

  getScale(t) {
    if (!this.useXpos) {
      throw new Error("You must set `useXpos` to true to use this method");
    }

    if (typeof t === "function") {
      t = t();
    }

    let scale = tf.ones(t.shape);
    if (this.useXpos) {
      let power = tf.sub(t, t.shape[0] / 2).div(this.xposScaleBase);
      scale = tf.pow(this.scale, power.reshape([-1, 1]));
      scale = tf.concat([scale, scale], -1);
    }

    return scale;
  }

  forward(t) {
    if (typeof t === "function") {
      t = t();
    }

    return tf.mul(t, this.freqs).reshape([-1, 2]);
  }
}

// Example of using the RotaryEmbedding
// const rotary = new RotaryEmbedding({ dim: 512 });
// const inputTensor = tf.ones([10, 10, 512]);
// const rotated = rotary.rotateQueriesOrKeys(inputTensor);
