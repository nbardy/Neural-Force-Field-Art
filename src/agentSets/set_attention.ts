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
import { pairwiseDistanceWithDropout } from "../utils/math";
import { Transformer } from "../trashPanda/blocks/clipTransformer";
import { AgentBatch, AgentSet as AgentConfig } from "../types";

const middleStrength1 = tf.scalar(2.0);
const middleStrength2 = tf.scalar(2.0);
const middleStrength3 = tf.scalar(2.0);

export const agentModel1 = new Transformer({ num_layers: 512 });
export const agentModel2 = new Transformer({ num_layers: 64 });
export const agentModel3 = new Transformer({ num_layers: 12 });

const agentModels = [agentModel1, agentModel2, agentModel3];

function squared_dist(A: tf.Tensor2D, B: tf.Tensor2D) {
  const norm_A = A.square().sum(1).expandDims(1); // Shape becomes [m, 1]
  const norm_B = B.square().sum(1).expandDims(0); // Shape becomes [1, n]

  const dists = norm_A.add(norm_B).sub(tf.mul(2, A.matMul(B, false, true)));

  return dists;
}

// calculates if the agents in batch i are close to the center, position: 0.5, 0.5
export function middleReward(positions, i) {
  const center = tf.tensor2d([[0.5, 0.5]]);
  const distances = squared_dist(positions, center);

  return distances.mean().asScalar();
}

export function repelAndMiddle(agentBatch, selfI, enemiesI) {
  const agentPositionsSelf = agentBatch.agentPositions[selfI];

  const enemiesPositions = enemiesI.map((i) => agentBatch.agentPositions[i]);
  const enemiesDistances = enemiesPositions
    .map((enemyPositions) =>
      pairwiseDistanceWithDropout(agentPositionsSelf, enemyPositions, 24)
    )
    .map((distances) => distances.mean());

  const distanceEnemies = tf.stack(enemiesDistances).mean();
  const distanceToMiddle = middleReward(agentBatch, agentPositionsSelf);

  return middleStrength1.mul(distanceToMiddle).add(distanceEnemies).asScalar();
}

export const calculateReward1 = (agentBatch) =>
  repelAndMiddle(agentBatch, 0, [1, 2]);
export const calculateReward2 = (agentBatch) =>
  repelAndMiddle(agentBatch, 1, [0, 2]);
export const calculateReward3 = (agentBatch) =>
  repelAndMiddle(agentBatch, 2, [0, 1]);

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
    agentModels: agentModels,
  };
};

export const agentConfig: AgentConfig = {
  agentModels,
  rewardFunctions,
  initializeAgents,
};
