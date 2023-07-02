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
import { AgentBatch, AgentSet } from "../types";

export function createModel() {
  const model = tf.sequential();

  // Input layer
  model.add(
    tf.layers.dense({ units: 64, activation: "relu", inputShape: [2] })
  );

  // Hidden layers
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));

  // Output layer
  model.add(tf.layers.dense({ units: 2 }));

  // Compile the model
  model.compile({ loss: "meanSquaredError", optimizer: "adam" });

  return model;
}

export const agentModel1 = createModel();
export const agentModel2 = createModel();
export const agentModel3 = createModel();

export function calculateReward1(agentBatch: AgentBatch) {
  // Take the first and second agent positions from the batch
  const agent1Positions = agentBatch.agentPositions[0];
  const agent2Positions = agentBatch.agentPositions[1];

  // calculate the Euclidean distance between all pairs of agent 1 and agent 2
  const distances = agent1Positions.sub(agent2Positions).norm("euclidean", 1);
  const v = distances.mean().neg(); // maximize for closeness is same as minimize the negative distance

  // return scalar reward
  return v.asScalar();
}

export function calculateReward2(agentBatch: AgentBatch) {
  const agent1Positions = agentBatch.agentPositions[0];
  const agent2Positions = agentBatch.agentPositions[1];

  // calculate the distance between each agent 2 and all other agent 2s
  const distances2 = agent2Positions.sub(agent2Positions).norm("euclidean", 1);
  // calculate the distance between agent 2 and agent 1
  const distances1 = agent2Positions.sub(agent1Positions).norm("euclidean", 1);
  // reward is minimizing the closeness to agent 2s and maximizing distance from agent 1
  const v = distances2.mean().add(distances1.mean().neg());

  // return scalar reward
  return v.asScalar();
}

export function calculateReward3(agentBatch: AgentBatch) {
  const agent1Positions = agentBatch.agentPositions[0];
  const agent2Positions = agentBatch.agentPositions[1];
  const agent3Positions = agentBatch.agentPositions[2];

  // calculate the squared distance between agent 3 and agent 1
  const distances1 = agent3Positions
    .sub(agent1Positions)
    .norm("euclidean", 1)
    .square();
  // calculate the squared distance between agent 3 and agent 2
  const distances2 = agent3Positions
    .sub(agent2Positions)
    .norm("euclidean", 1)
    .square();
  // reward is the sum of both squared distances
  const v = distances1.mean().add(distances2.mean());

  return v.asScalar();
}

// Agent models
const agentModels = [agentModel1, agentModel2, agentModel3];

// Reward functions
const rewardFunctions = [calculateReward1, calculateReward2, calculateReward3];

export const agentSet: AgentSet = {
  agentModels,
  rewardFunctions,
};
