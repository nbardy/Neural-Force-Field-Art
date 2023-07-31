import * as tf from "@tensorflow/tfjs";

// Tensor type definition for better code readability
type Tensor = tf.Tensor;

// helper functions
// const exists = (val: any): boolean => val !== null && val !== undefined;
// typeguard
const exists = <T>(val: T | null | undefined): val is T =>
  val !== null && val !== undefined;

const defaultVal = (val: any, d: any) => (exists(val) ? val : d);

// regular attention
const attention = (
  q: Tensor, // [batch_size, seq_len, heads, head_dim]
  k: Tensor, // [batch_size, seq_len, heads, head_dim]
  v: Tensor, // [batch_size, seq_len, heads, head_dim]
  mask: Tensor | null = null, // [batch_size, seq_len, seq_len] or null
  causal: boolean = false,
  attnBias: Tensor | null = null // [batch_size, seq_len, seq_len] or null
): Tensor => {
  // [batch_size, seq_len, head_dim]
  let scale = Math.sqrt(1.0 / q.shape[q.shape.length - 1]);
  q = q.mul(tf.scalar(scale)); // [batch_size, seq_len, heads, head_dim]

  // Compute dot product between q and k
  let sim = tf.matMul(q, k, false, true); // [batch_size, seq_len, heads, seq_len]

  if (exists(attnBias)) {
    sim = sim.add(attnBias); // [batch_size, seq_len, heads, seq_len]
  }

  if (exists(mask)) {
    sim = tf.where(mask, sim, Number.MIN_SAFE_INTEGER); // [batch_size, seq_len, heads, seq_len]
  }

  if (causal) {
    const maskCausal = tf.linalg.bandPart(tf.onesLike(sim), -1, 0);
    sim = tf.where(maskCausal, sim, Number.MIN_SAFE_INTEGER); // [batch_size, seq_len, heads, seq_len]
  }

  sim = sim.sub(sim.max(-1, true)); // [batch_size, seq_len, heads, seq_len]
  let attn = tf.softmax(sim, -1); // [batch_size, seq_len, heads, seq_len]

  // Compute dot product between attn and v
  let out = tf.matMul(attn, v); // [batch_size, seq_len, heads, head_dim]
  return out;
};

// ... (Due to the complexity, the rest of the code needs to be adapted similarly.)

// main class
class Attention {
  // ypes

  heads: number;
  causal: boolean;
  dropout: number;
  toQ: any;
  toKV: any;
  toOut: any;
  memoryEfficient: boolean;
  qBucketSize: number;
  kBucketSize: number;

  constructor(
    dim,
    heads = 8,
    dimHead = 64,
    dropout = 0.0,
    causal = false,
    memoryEfficient = false,
    qBucketSize = 512,
    kBucketSize = 1024
  ) {
    this.heads = heads;
    this.causal = causal;
    this.dropout = dropout;

    this.toQ = tf.layers.dense({ units: heads * dimHead, useBias: false });
    this.toKV = tf.layers.dense({ units: 2 * heads * dimHead, useBias: false });
    this.toOut = tf.layers.dense({ units: dim, useBias: false });

    this.memoryEfficient = memoryEfficient;
    this.qBucketSize = qBucketSize;
    this.kBucketSize = kBucketSize;
  }

  forward(
    x,
    context = null,
    mask = null,
    attnBias = null,
    memoryEfficient = null,
    qBucketSize = null,
    kBucketSize = null
  ) {
    memoryEfficient = defaultVal(memoryEfficient, this.memoryEfficient);
    qBucketSize = defaultVal(qBucketSize, this.qBucketSize);
    kBucketSize = defaultVal(kBucketSize, this.kBucketSize);

    const h = this.heads;
    context = defaultVal(context, x);

    const q = this.toQ.apply(x);
    const [k, v] = tf.split(this.toKV.apply(context), 2, -1);

    const splitShape = [h, -1];
    const qR = tf.reshape(q, splitShape);
    const kR = tf.reshape(k, splitShape);
    const vR = tf.reshape(v, splitShape);

    const attnFn = !memoryEfficient ? attention : memoryEfficientAttention; // memoryEfficientAttention function needs to be defined similar to attention

    const out = attnFn(
      qR,
      kR,
      vR,
      mask,
      attnBias,
      this.causal,
      qBucketSize,
      kBucketSize,
      this.dropout
    );

    return this.toOut.apply(
      tf.reshape(out, [-1, h * q.shape[q.shape.length - 1]])
    );
  }
}

const l2norm = (tensor: Tensor): Tensor => {
  return tf.div(tensor, tf.norm(tensor, "euclidean", -1, true));
};

const padToMultiple = (
  tensor: Tensor,
  multiple: number,
  dim = -1,
  value = 0
): [boolean, Tensor] => {
  const seqlen = tensor.shape[dim];
  if (seqlen % multiple === 0) return [false, tensor];

  const remainder = multiple - (seqlen % multiple);
  const paddings = Array(tensor.rank).fill([0, 0]);
  paddings[dim] = [0, remainder];
  return [true, tf.pad(tensor, paddings, value)];
};

const lookAround = (
  x: Tensor,
  backward: number = 1,
  forward: number = 0,
  padValue: number = -1,
  dim: number = 2
): Tensor => {
  const t = x.shape[1];
  const paddedX = tf.pad(
    x,
    [
      [0, 0],
      [backward, forward],
      [0, 0],
    ],
    padValue
  );
  let tensors = [];
  for (let ind = 0; ind <= forward + backward; ind++) {
    tensors.push(paddedX.slice([0, ind], [-1, t]));
  }
  return tf.concat(tensors, dim);
};
