import { agentSet } from "./agentSets/set1";

import { timeout } from "./utils";

import * as tf from "@tensorflow/tfjs";

import { AgentBatch, AgentSet } from "./types";
import { updateParticles2 } from "./updateParticles";
import { Rank } from "@tensorflow/tfjs";

// x,y

const debugPauseTime = 2000;

// Colors for agents
const colors = ["yellow", "blue", "green"];

// SGD Optimizers for each agent model
const learningRate = 0.0001;

function drawAgents(canvas, agents: AgentBatch) {
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

// let currentAgents = initializeAgents();

export function startLoop(canvas: HTMLCanvasElement) {
  // fullscreen
  const width = window.innerWidth;
  const height = window.innerHeight;

  canvas.width = width;
  canvas.height = height;

  let agentState = initializeAgents({ width, height });

  // Create optimizers
  const optimizers = [
    tf.train.adam(learningRate),
    tf.train.adam(learningRate),
    tf.train.adam(learningRate),
  ];

  const { agentModels, rewardFunctions } = agentSet;

  function mainLoop(canvas: HTMLCanvasElement) {
    let { agentPositions, agentVelocities } = agentState;

    // Starts empty to  be filled with updated agents
    const updatedAgents: AgentBatch = {
      agentPositions: [],
      agentVelocities: [],
    };

    /**
     * Setup functions
     */

    drawAgents(canvas, agentState);

    // Process an agent at index i
    // This will optimize it and update the state
    const processAgent = (_, i) => {
      const model = agentModels[i];
      optimizers[i].minimize(() => {
        const velocities = agentVelocities[i];
        const positions = agentPositions[i];
        // Predict forces to apply to the agents
        // const forces = model.predict(positions) as tf.Tensor2D;

        // // Apply the forces to velocities and update positions
        // const newV: tf.Tensor2D = tf.add(velocities, forces);
        // const newP: tf.Tensor2D = tf.add(positions, velocities);

        const { newV, newP } = updateParticles2(
          [positions, velocities],
          model,
          { width, height }
        );

        // Make a new agent state with the non-i agents sets the same
        // but with the updated i agent
        // Then use this input to the reward function

        // keep, will de cleaned after draw and update
        tf.keep(newP);
        tf.keep(newV);

        // add to updated agents
        // console.log("adding to updated agents", i);
        // console.log(updatedAgents.agentPositions);
        updatedAgents.agentPositions[i] = newP;
        updatedAgents.agentVelocities[i] = newV;

        const rewardState: AgentBatch = agentState;

        // note: we map over positions to get an iterator of the right size
        // but we act on pos and vel at the same time
        rewardState.agentPositions.map((_, j) => {
          // set in rewardState
          if (i === j) {
            rewardState.agentPositions[j] = newP;
            rewardState.agentVelocities[j] = newV;
          }
        });

        const reward = rewardFunctions[i](rewardState, i);

        return reward;
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

    // TODO: Debug why this doesn't work
    // agentPositions.forEach(cleanTensors);

    agentState.agentPositions = updatedAgents.agentPositions;
    agentState.agentVelocities = updatedAgents.agentVelocities;

    // log all before and after tensors

    requestAnimationFrame(async () => {
      // await timeout(10 * 20);
      console.log("frame request");
      mainLoop(canvas);
    });
  }

  console.log("starting loop");
  mainLoop(canvas);
}

function pt(name, tensor) {
  console.log(name);
  tensor.print(true);
}
