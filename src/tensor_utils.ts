import * as tf from "@tensorflow/tfjs";

export function createTiles(
  width: number,
  height: number,
  tileSize: number,
  centerTiles: boolean = false
): tf.Tensor {
  // Calculate grid density based on tile size
  const gridDensityX = width / tileSize;
  const gridDensityY = height / tileSize;

  // Create vectors for x and y coordinates
  let x = tf.linspace(0, 1, gridDensityX);
  let y = tf.linspace(0, 1, gridDensityY);

  // Optionally center tiles by shifting half the distance of a tile
  if (centerTiles) {
    const shiftX = 0.5 * (1 / gridDensityX);
    const shiftY = 0.5 * (1 / gridDensityY);
    x = x.sub(shiftX);
    y = y.sub(shiftY);
  }

  // Create the mesh grid using TensorFlow.js
  const [X, Y] = tf.meshgrid(x, y);

  return tf.stack([X, Y], 2); // BxWxHx2
}
