import * as tf from "@tensorflow/tfjs";
import * as TrashPanda from "./trashPanda";

export type ModelType = tf.LayersModel | tf.Sequential | TrashPanda.Transformer;
export interface AgentBatch {
  agentPositions: tf.Tensor2D[];
  agentVelocities: tf.Tensor2D[];
  agentModels: ModelType[];
}

type T1 = (agentBatch: AgentBatch, i: number) => tf.Scalar;
type T2 = (agentBatch: AgentBatch) => tf.Scalar;

type RewardFn = T1 | T2;

// type
export type AgentSet = {
  agentModels: tf.Sequential[];
  rewardFunctions: RewardFn[];
  initializeAgents: (args: { width: number; height: number }) => AgentBatch;
};
