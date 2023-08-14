import { AgentBatch } from "../types/all";
import * as twgl from "twgl.js";
import * as tf from "@tensorflow/tfjs";
import { labConversionFunctions, rotateAB, skewL } from "../utils/color";
import { drawCircles, drawTriangles } from "../quickDraw/main";
import { createTiles } from "../utils/tensor_utils";

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
    let tiles = createTiles(canvas.width, canvas.height, gridDensity, true); // Centering tiles
    // flatten to set of poitns from WxHx2 to (W*H)x2
    tiles = tf.reshape(tiles, [-1, 2]);

    for (let i = 0; i < agentModels.length; i++) {
      const model = agentModels[i];
      const modelColor = modelColorSet[i];

      const forceUpdates = model.predict(tiles); // Assume the model predicts force updates for each tile

      // Create the triangles with tfjs
      const forceUpdatesMagnitude = tf.norm(forceUpdates, 2, 1); // Calculate the magnitude of the force updates

      drawTriangles({
        canvas,
        positions: tiles,
        directions: forceUpdates,
        height: forceUpdatesMagnitude,
        background: { color: modelColor },
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
    drawCircles({
      canvas,
      positions,
      radius: 4,
      background: { color: posColor },
    });

    // velocity arrow
    drawTriangles({
      canvas,
      positions: positions,
      directions: agents.agentVelocities[i],
      height: 10,
      background: { color: velColor },
    });
  });
}
