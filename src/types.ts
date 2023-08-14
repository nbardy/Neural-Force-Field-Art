import * as tf from "@tensorflow/tfjs";
import { models, Model } from "./trashPanda";

type TModel = typeof models.Transformer;

export type MLModel = tf.LayersModel | tf.Sequential | Model;

export interface AgentBatch {
  agentPositions: tf.Tensor2D[];
  agentVelocities: tf.Tensor2D[];
  agentModels: MLModel[];
}

type T1 = (agentBatch: AgentBatch, i: number) => tf.Scalar;
type T2 = (agentBatch: AgentBatch) => tf.Scalar;

type RewardFn = T1 | T2;

// type
export type AgentSet = {
  agentModels: MLModel[];
  rewardFunctions: RewardFn[];
  initializeAgents: (args: { width: number; height: number }) => AgentBatch;
};
