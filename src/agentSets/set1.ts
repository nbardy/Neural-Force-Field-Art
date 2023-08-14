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
import { Rank } from "@tensorflow/tfjs";
import { pairwiseDistanceWithDropout } from "../math";
import { AgentBatch, AgentSet as AgentConfig } from "../types";

export function createModel() {
  const model = tf.sequential();

  // Input layer
  model.add(
    tf.layers.dense({ units: 64, activation: "sigmoid", inputShape: [2] })
  );

  // Hidden layers
  model.add(tf.layers.dense({ units: 128, activation: "sigmoid" }));
  model.add(tf.layers.dense({ units: 128, activation: "sigmoid" }));

  // Output layer
  model.add(tf.layers.dense({ units: 2, activation: "sigmoid" }));

  return model;
}

export const agentModel1 = createModel();
export const agentModel2 = createModel();
export const agentModel3 = createModel();

function squared_dist(A: tf.Tensor2D, B: tf.Tensor2D) {
  const norm_A = A.square().sum(1).expandDims(1); // Shape becomes [m, 1]
  const norm_B = B.square().sum(1).expandDims(0); // Shape becomes [1, n]

  const dists = norm_A.add(norm_B).sub(tf.mul(2, A.matMul(B, false, true)));

  return dists;
}

export function calculateReward1(agentBatch, i) {
  const newV = agentBatch.agentVelocities[i];

  console.log("reward 1");
  const agent1Positions = agentBatch.agentPositions[0];
  const agent2Positions = agentBatch.agentPositions[1];
  const agent3Positions = agentBatch.agentPositions[2];

  const distances = pairwiseDistanceWithDropout(
    agent1Positions,
    agent3Positions,
    5
  );
  let v = distances.mean().neg();

  return v.asScalar();
}

export function calculateReward2(agentBatch) {
  // TODO: Something is wrong with this reward function. The model optimized with it will return NaNs

  // return tf.mean(agentBatch.agentVelocities[1]).asScalar();

  const agent1Positions = agentBatch.agentPositions[0];
  const agent2Positions = agentBatch.agentPositions[1];

  const d = pairwiseDistanceWithDropout(agent1Positions, agent2Positions, 6);

  return d.mean().asScalar();
}

export function calculateReward3(agentBatch) {
  // return tf.mean(agentBatch.agentVelocities[2]).asScalar();

  const agent1Positions = agentBatch.agentPositions[0];
  const agent2Positions = agentBatch.agentPositions[1];
  const agent3Positions = agentBatch.agentPositions[2];

  const distances1 = pairwiseDistanceWithDropout(
    agent3Positions,
    agent1Positions,
    5
  );
  const distances2 = pairwiseDistanceWithDropout(
    agent3Positions,
    agent2Positions,
    5
  );

  const v = distances1.mean().add(distances2.mean());

  return v.asScalar();
}

// Agent models
const agentModels = [agentModel1, agentModel2, agentModel3];

// Reward functions
const rewardFunctions = [calculateReward1, calculateReward2, calculateReward3];

const totalDim = 2;

const count1 = 100;
const count2 = 70;
const count3 = 50;

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
  let agent2Positions: tf.Tensor2D = tf.randomUniform(
    [count2, totalDim],
    0,
    height
  );
  let agent3Positions: tf.Tensor2D = tf.randomUniform(
    [count3, totalDim],
    0,
    width
  );

  // Initial velocities for agents
  // let agent1Velocities: tf.Tensor2D = tf.zerosLike(agent1Positions);
  // let agent2Velocities: tf.Tensor2D = tf.zerosLike(agent2Positions);
  // let agent3Velocities: tf.Tensor2D = tf.zerosLike(agent3Positions);

  // instead of zeros, do 0-1, then multiply by 2, then subtract 1
  const scale = <T extends tf.Tensor<Rank>>(x: T): T =>
    tf.add(tf.mul(x, 2), -1) as T;

  let agent1Velocities: tf.Tensor2D = scale(
    tf.randomUniform(agent1Positions.shape)
  );

  let agent2Velocities: tf.Tensor2D = scale(
    tf.randomUniform(agent2Positions.shape)
  );

  let agent3Velocities: tf.Tensor2D = scale(
    tf.randomUniform(agent3Positions.shape)
  );

  return {
    agentPositions: [agent1Positions, agent2Positions, agent3Positions],
    agentVelocities: [agent1Velocities, agent2Velocities, agent3Velocities],
  };
};

export const agentConfig: AgentConfig = {
  agentModels,
  rewardFunctions,
  initializeAgents,
};
