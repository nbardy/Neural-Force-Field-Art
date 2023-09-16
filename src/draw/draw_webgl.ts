import { AgentBatch } from "../types/all";
import * as twgl from "twgl.js";
import * as tf from "@tensorflow/tfjs";
import { labConversionFunctions, rotateAB, skewL } from "../utils/color";
import { drawCircles, drawTriangles } from "../quickDraw/main";
import { createTiles } from "../utils/tensor_utils";

/*
 * Colors
 */

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
