import * as tf from "@tensorflow/tfjs";
import { pairwiseDistanceWithDropout } from "../utils/math";

const middleStrength1 = tf.scalar(2.0)

export function squared_dist(A: tf.Tensor2D, B: tf.Tensor2D) {
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
