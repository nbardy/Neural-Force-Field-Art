// This is the first set of agents. It contains 3 agents.
// Each agent model only get's to see the position of it's own agent positions.

// However the reward function reacts to other agents. So the model
// should act more like a classic rule based system where the agents
// will learn policies that will act mostly on a turn by turn basis.

// This a set of test agents to get the projec skeleton working.
// Followup sets should allow the models to see the other agents, this will
// allow the agents to learn policies that will act more intelligently with
// respect to the behavior of the other agents. Moving the other agents information
// to model input should allow the agents to learn policies that are predictive
// of the behavior of the other agents types.

import * as tf from "@tensorflow/tfjs";
import { pairwiseDistanceWithDropout } from "../utils/math";
import { AgentBatch, AgentSet } from "../types/all";

export function createModel() {
  const model = tf.sequential();

  // Input N x 2

  // Input layer
  model.add(
    tf.layers.dense({ units: 8, activation: "sigmoid", inputShape: [2] }) // N x 8
  );

  // Hidden layers
  model.add(tf.layers.dense({ units: 64, activation: "sigmoid" })); // N x 64
  model.add(tf.layers.dense({ units: 64, activation: "sigmoid" })); // N x 64
  model.add(tf.layers.dense({ units: 64, activation: "sigmoid" })); // N x 64
  model.add(tf.layers.dense({ units: 128, activation: "sigmoid" }));
  model.add(tf.layers.dense({ units: 64, activation: "sigmoid" })); // N x 64
  model.add(tf.layers.dense({ units: 64, activation: "sigmoid" })); // N x 64

  // Output layer
  model.add(tf.layers.dense({ units: 2, activation: "sigmoid" }));

  return model;
}

export const agentModel1 = createModel();

export function calculateReward1(agentBatch, i) {
  const agent1Positions = agentBatch.agentPositions[0];

  // split in half by first dim
  const index = Math.floor(agent1Positions.shape[0] / 2);

  const firstHalf = agent1Positions.slice([0, 0], [index, 2]);
  const secondHalf = agent1Positions.slice([index, 0], [index, 2]);

  const distances = pairwiseDistanceWithDropout(firstHalf, secondHalf, 5);
  let v = distances.mean().neg();

  return v.asScalar();
}

// Agent models
const agentModels = [agentModel1];

// Reward functions
const rewardFunctions = [calculateReward1];

const totalDim = 2;

const count1 = 1000;

const initializeAgents = ({
  width,
  height,
}: {
  width: number;
  height: number;
}): AgentBatch => {
  // Initial positions for agents
  let agent1Positions: tf.Tensor2D = tf.randomUniform(
    [count1, totalDim],
    0,
    width
  );

  // random like
  let agent1Velocities: tf.Tensor2D = tf.randomUniform(
    [count1, totalDim],
    -1,
    1
  );

  return {
    agentPositions: [agent1Positions],
    agentVelocities: [agent1Velocities],
    agentModels,
  };
};

export const agentSet: AgentSet = {
  agentModels,
  rewardFunctions,
  initializeAgents,
};
