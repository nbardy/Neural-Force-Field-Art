import * as tf from "@tensorflow/tfjs";

export function gaussianKernel(
  M: number,
  std: number = 1.0,
  sym: boolean = true,
  dtype: "float32" = "float32"
): tf.Tensor {
  if (std <= 0) {
    throw new Error(
      `Standard deviation must be positive, got: ${std} instead.`
    );
  }

  if (M === 0) {
    return tf.zeros([0], dtype);
  }

  const start = -(M ? (M > 1 && !sym ? M : M - 1) : 0) / 2.0;
  const constant = 1 / (std * Math.sqrt(2));

  // Using tf.linspace to replicate torch.linspace
  const k = tf.linspace(start * constant, (start + (M - 1)) * constant, M);

  // Computing the Gaussian window using the exponential function
  const result = tf.exp(tf.neg(tf.square(k))); // Shape: M

  return result;
}
