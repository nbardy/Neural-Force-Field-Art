import * as tf from "@tensorflow/tfjs";

export interface AgentBatch {
  agentPositions: tf.Tensor2D[];
  agentVelocities: tf.Tensor2D[];
}

// type
export type AgentSet = {
  agentModels: tf.Sequential[];
  rewardFunctions: ((agentBatch: AgentBatch) => tf.Scalar)[];
};
