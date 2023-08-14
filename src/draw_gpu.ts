import { AgentBatch } from "./types";
import * as twgl from "twgl.js";
import * as tf from "@tensorflow/tfjs";
import { labConversionFunctions, rotateAB, skewL } from "./color";
import { drawCircles, drawTriangles } from "./quickDraw";

/*
 * Colors
 */

const magenta = [255, 0, 255];
const teal = [0, 128, 128];
const aquamarine = [127, 255, 212];

// light and dark with lab
const magnetaSet = [skewL(magenta, 1.4), skewL(magenta, 0.8)];
const tealSet = [skewL(teal, 1.4), skewL(teal, 0.8)];
const aquamarineSet = [skewL(aquamarine, 1.4), skewL(aquamarine, 0.8)];

const pointColorSet = [magnetaSet, tealSet, aquamarineSet];

// slight rotate and shift to something low for the light color
const modelMagenta = skewL(rotateAB(magenta, 0.06), 2.8);
const modelTeal = skewL(rotateAB(teal, 0.03), 1.8);
const modelAquamarine = skewL(rotateAB(aquamarine, -0.04), 2.2);

const modelColorSet = [modelMagenta, modelTeal, modelAquamarine];

export function drawAgents(
  canvas,
  agents: AgentBatch,
  config: { predictField?: boolean }
) {
  const gl = canvas.getContext("webgl");

  // Config Values
  const pointSize = 10;
  const gridDensity = 20; // Tile size for the grid
  const { predictField } = config;
  const { agentModels } = agents;

  if (predictField) {
    // Divide canvas into tiles based on gridDensity
    const tiles = createTiles(canvas.width, canvas.height, gridDensity, true); // Centering tiles
    for (let model of agentModels) {
      const forceUpdates = model.predict(tiles); // Assume the model predicts force updates for each tile

      // Create the triangles with tfjs
      const forceUpdatesMagnitude = tf.norm(forceUpdates, 2, 2); // Calculate the magnitude of the force updates
      drawTriangles(canvas, {
        pos: tiles,
        dir: forceUpdates,
        height: forceUpdatesMagnitude,
      });
    }
  }

  // Draw Agents
  agents.agentPositions.forEach((_, i) => {
    const posColor = pointColorSet[i][0];
    const velColor = pointColorSet[i][1];

    // lighten based on velocity

    // pos circle
    drawCircles(canvas, agents.agentPositions[i], 4, posColor);

    // velocity arrow
    drawTriangles({
      canvas,
      positions: agents.agentPositions[i],
      velocities: agents.agentVelocities[i],
      height: 10,
      velColor,
    });
  });
}

export function exportFromGPU(
  dataId: DataId,
  gl: WebGLRenderingContext,
  options: DataToGPUWebGLOption = {}
): WebGLBuffer {
  // Step 2: Call readToGPU to get the densely packed texture.
  const gpuData = this.readToGPU(dataId, options);

  // Step 3: Create a framebuffer and bind it
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

  // Attach the texture to the framebuffer
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    gpuData.texture,
    0
  );
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error("Framebuffer is not complete");
  }

  // Create a buffer and bind it
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer);
  gl.bufferData(
    gl.PIXEL_PACK_BUFFER,
    gpuData.texShape[0] * gpuData.texShape[1] * 4,
    gl.STATIC_READ
  );

  // Read the pixels from the framebuffer into the buffer
  gl.readPixels(
    0,
    0,
    gpuData.texShape[1],
    gpuData.texShape[0],
    gl.RGBA,
    gl.FLOAT,
    0
  );

  // Unbind the framebuffer and the buffer
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);

  // Step 4: Return the WebGL buffer
  return buffer;
}

function createTiles(
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
