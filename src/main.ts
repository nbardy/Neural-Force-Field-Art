import { agentSet } from "./agentSets/set1";

import * as tf from "@tensorflow/tfjs";

import { AgentBatch, AgentSet } from "./types";

const initializeAgents = ({
  width,
  height,
}: {
  width: number;
  height: number;
}): AgentBatch => {
  // Initial positions for agents
  let agent1Positions: tf.Tensor2D = tf.randomUniform([10, 2], 0, width);
  let agent2Positions: tf.Tensor2D = tf.randomUniform([7, 2], 0, height);
  let agent3Positions: tf.Tensor2D = tf.randomUniform([5, 2], 0, width);

  // Initial velocities for agents
  let agent1Velocities: tf.Tensor2D = tf.zerosLike(agent1Positions);
  let agent2Velocities: tf.Tensor2D = tf.zerosLike(agent2Positions);
  let agent3Velocities: tf.Tensor2D = tf.zerosLike(agent3Positions);

  return {
    agentPositions: [agent1Positions, agent2Positions, agent3Positions],
    agentVelocities: [agent1Velocities, agent2Velocities, agent3Velocities],
  };
};

// Colors for agents
const colors = ["yellow", "blue", "green"];

// SGD Optimizers for each agent model
const learningRate = 0.01;

function drawAgents(canvas, agents: AgentBatch) {
  const ctx = canvas.getContext("2d");

  const { agentPositions } = agents;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  agentPositions.forEach((positions, i) => {
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

// let currentAgents = initializeAgents();

export function startLoop(canvas: HTMLCanvasElement) {
  let agentState = initializeAgents(canvas);

  // Create optimizers
  const optimizers = [
    tf.train.sgd(learningRate),
    tf.train.sgd(learningRate),
    tf.train.sgd(learningRate),
  ];

  const { agentModels, rewardFunctions } = agentSet;

  function mainLoop(canvas: HTMLCanvasElement) {
    const { agentPositions, agentVelocities } = agentState;

    // Starts empty to  be filled with updated agents
    const updatedAgents: AgentBatch = {
      agentPositions: [],
      agentVelocities: [],
    };

    /**
     * Setup functions
     */

    // Process an agent at index i
    const processAgent = (positions, i) => {
      const model = agentModels[i];
      const velocities = agentVelocities[i];

      // Predict forces to apply to the agents
      const forces = model.predict(positions) as tf.Tensor2D;

      // Apply the forces to velocities and update positions
      const newV: tf.Tensor2D = velocities.add(forces);
      const newP: tf.Tensor2D = positions.add(velocities);

      // Update the agents
      const newAgents = {
        agentPositions: newP,
        agentVelocities: newV,
      };

      // update state (Set i)
      updatedAgents.agentPositions[i] = newAgents.agentPositions;
      updatedAgents.agentVelocities[i] = newAgents.agentVelocities;

      optimizers[i].minimize(() => {
        return rewardFunctions[i](agentState);
      });
    };

    const cleanTensors = (_, i) => {
      // clean up old tensors
      agentPositions[i].dispose();
      agentVelocities[i].dispose();
    };

    /**
     * Run Loop
     */

    agentPositions.forEach(processAgent);
    // Cleanup old tensors
    agentPositions.forEach(cleanTensors);

    drawAgents(canvas, updatedAgents);

    requestAnimationFrame(() => {
      console.log("frame request");
      mainLoop(canvas);
    });
  }

  console.log("starting loop");
  mainLoop(canvas);
}
