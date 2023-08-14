import { AgentBatch } from "./types";
import * as twgl from "twgl.js";
import * as tf from "@tensorflow/tfjs";
import { labConversionFunctions, rotateAB, skewL } from "./color";
import { drawCircles, drawTriangles } from "./quickDraw";
import { createTiles } from "./tensor_utils";

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
      drawTriangles({
        canvas,
        pos: tiles,
        dir: forceUpdates,
        height: forceUpdatesMagnitude,
      });
    }
  }

  // Draw Agents
  agents.agentPositions.forEach((_, i) => {
    const positions = agents.agentPositions[i];

    const posColor = pointColorSet[i][0];
    const velColor = pointColorSet[i][1];

    // lighten based on velocity

    // pos circle
    drawCircles(canvas, positions, 4, posColor);

    // velocity arrow
    drawTriangles({
      canvas,
      positions: positions,
      velocities: agents.agentVelocities[i],
      height: 10,
      velColor,
    });
  });
}
