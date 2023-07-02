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

// Squared distance function for pairwise calculation
function squared_dist(A, B) {
  // dimensions
  const m = A.shape[0];
  const n = B.shape[0];

  const A_square = tf.matMul(A, A, false, true).sum(1).expandDims(1);
  const B_square = tf
    .matMul(B, B, false, true)
    .sum(1)
    .expandDims()
    .tile([m, 1]);

  const AB = tf.matMul(A, B, true, false);

  return A_square.sub(AB.mul(2)).add(B_square);
}

export function calculateReward1(agentBatch) {
  const agent1Positions = agentBatch.agentPositions[0];
  const agent2Positions = agentBatch.agentPositions[1];

  const distances = squared_dist(agent1Positions, agent2Positions).sqrt();
  const v = distances.mean().neg();

  return v.asScalar();
}

export function calculateReward2(agentBatch) {
  const agent1Positions = agentBatch.agentPositions[0];
  const agent2Positions = agentBatch.agentPositions[1];

  const distances2 = squared_dist(agent2Positions, agent2Positions).sqrt();
  const distances1 = squared_dist(agent2Positions, agent1Positions).sqrt();

  const v = distances2.mean().add(distances1.mean().neg());

  return v.asScalar();
}

export function calculateReward3(agentBatch) {
  const agent1Positions = agentBatch.agentPositions[0];
  const agent2Positions = agentBatch.agentPositions[1];
  const agent3Positions = agentBatch.agentPositions[2];

  const distances1 = squared_dist(agent3Positions, agent1Positions)
    .sqrt()
    .square();
  const distances2 = squared_dist(agent3Positions, agent2Positions)
    .sqrt()
    .square();

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
