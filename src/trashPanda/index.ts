import { normalize } from "./linalg";
import { Transformer } from "./models/clipTransformer";
import { ScoreNetwork } from "./models/ScoreNetwork";
import { MultiHeadAttention } from "./layers/MultiheadAttention";

export const linalg = {
  normalize,
};

export const models = {
  Transformer,
  ScoreNetwork,
};

export const attention = {
  MultiHeadAttention,
};

export type Model = Transformer | ScoreNetwork;
