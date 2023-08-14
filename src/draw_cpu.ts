// import types
import { AgentBatch } from "./types";
import * as tf from "@tensorflow/tfjs";

const colors = ["#ff0000", "#00ff00", "#0000ff"];

/*
 * old deprecated agent drawing code
 */
function drawAgentsCPU(
  canvas,
  agents: AgentBatch,
  config: { predictField?: boolean; model: unknown }
) {
  // debug
  // count of agents

  const ctx = canvas.getContext("2d");

  const { agentPositions } = agents;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  agentPositions.forEach((_, i) => {
    // sync tensors to array
    const positions = agentPositions[i] as tf.Tensor2D;
    // const velocities = agentVelocities[i] as tf.Tensor2D;
    const agents = positions.arraySync() as number[][];

    agents.forEach((agent) => {
      ctx.beginPath();
      ctx.arc(agent[0], agent[1], 10, 0, Math.PI * 2);
      ctx.fillStyle = colors[i];
      ctx.fill();
      ctx.closePath();
    });
  });
}
