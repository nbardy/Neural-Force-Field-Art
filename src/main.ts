/**
 * The main loop kicks off training and adds drawing each frame
 */
import { agentConfig } from "./agentSets/set1";

import { timeout } from "./utils/utils";

import * as tf from "@tensorflow/tfjs";

import { AgentBatch, AgentSet } from "./types/all";
import { updateParticles2 } from "./physics/updateParticles";

import { drawAgents } from "./draw/draw_gpu";

import { printMaxEmbeddingsImageSeries } from "./trashPanda/experimental/pointEmbeddings";

console.log("testing");
printMaxEmbeddingsImageSeries();

// Print debugger
const predictField = true;
const debugPauseTime = 2000;

// Colors for agents
const colors = ["yellow", "blue", "green"];

// SGD Optimizers for each agent model
const learningRate = 0.0001;

// let currentAgents = initializeAgents();

export function startLoop(canvas: HTMLCanvasElement) {
  // fullscreen
  const width = window.innerWidth;
  const height = window.innerHeight;

  canvas.width = width;
  canvas.height = height;

  const { agentModels, rewardFunctions, initializeAgents } = agentConfig;

  let agentState = initializeAgents({ width, height });

  // Create optimizers
  const optimizers = [
    tf.train.adam(learningRate),
    tf.train.adam(learningRate),
    tf.train.adam(learningRate),
  ];

  function mainLoop(canvas: HTMLCanvasElement) {
    let { agentPositions, agentVelocities } = agentState;

    // Starts empty to  be filled with updated agents
    const updatedAgents: AgentBatch = {
      agentPositions: [],
      agentVelocities: [],
      agentModels,
    };

    /**
     * Setup functions
     */

    drawAgents(canvas, agentState, { predictField });

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
