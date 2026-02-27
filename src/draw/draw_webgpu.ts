import { AgentBatch } from "../types/all";
import { labConversionFunctions, rotateAB, skewL } from "../utils/color";
import { createTiles } from "../utils/tensor_utils";
import { SwissGL2 } from "@swissgl/swissgl2"; // Import SwissGL2

// ... (color definitions)

export function drawAgents(
  canvas,
  agents: AgentBatch,
  config: { predictField?: boolean }
) {
  const gl = new SwissGL2(canvas); // Create a new SwissGL2 instance

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

      // Create the triangles with SwissGL2
      const forceUpdatesMagnitude = tf.norm(forceUpdates, 2, 1); // Calculate the magnitude of the force updates

      // TODO: Implement drawTriangles using SwissGL2
    }
  }

  // Draw Agents
  agents.agentPositions.forEach((_, i) => {
    const positions = agents.agentPositions[i];

    const posColor = pointColorSet[i][0];
    const velColor = pointColorSet[i][1];

    // lighten based on velocity

    // pos circle
    // TODO: Implement drawCircles using SwissGL2

    // velocity arrow
    // TODO: Implement drawTriangles using SwissGL2
  });
}
