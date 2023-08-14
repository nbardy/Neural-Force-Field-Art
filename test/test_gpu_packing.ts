import * as tf from "@tensorflow/tfjs"; // Import TensorFlow.js
import {
  extractBufferFromTexture,
  WebGLContextElite as WebGLRenderingContextElite,
} from "../src/quickDraw";

// read buffer data
function readBufferData(
  gl: WebGLRenderingContextElite,
  buffer: WebGLBuffer
): Float32Array {
  // 1. Bind the buffer
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

  // 2. Get the buffer size
  const bufferSize = gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE);

  // 3. Read the data from the buffer into an ArrayBuffer
  const rawData = new ArrayBuffer(bufferSize);
  gl.getBufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(rawData));

  // 4. Return the data as a Float32Array
  return new Float32Array(rawData);
}

function testTensorToGPUDataAndExtraction() {
  // 1. Create a tensor with known data
  const tensor = tf.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
  ]); // Shape: 2x4

  // Get a WebGL rendering context
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl") as WebGLRenderingContextElite;

  // 3. Extract the buffer using the provided function
  const buffer = extractBufferFromTexture(gl, tensor);
  const bufferData = readBufferData(gl, buffer);

  // Convert the tensor data to a standard array for comparison
  const tensorData = tensor.arraySync() as number[][];

  // if number
  if (typeof bufferData[0] === "number") {
    // dump data for debug
    console.debug("bufferData", bufferData);
    throw new Error(" bufferData[0] is number");
  }

  // 5. Compare the buffer data with the original tensor data
  for (let i = 0; i < tensorData.length; i++) {
    for (let j = 0; j < tensorData[i].length; j++) {
      const index = i * tensorData[i].length + j;
      if (bufferData[index] !== tensorData[i][j]) {
        throw new Error(
          `Data mismatch at index ${index}: expected ${tensorData[i][j]}, got ${bufferData[index]}`
        );
      }
    }
  }

  console.log("Test passed!");
}

testTensorToGPUDataAndExtraction(); // Run the test
