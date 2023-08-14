import { normalize } from "./linalg";
import { Transformer } from "./blocks/clipTransformer";
import { MultiHeadAttention } from "./layers/multiheadattention";

export const linalg = {
  normalize,
};

export const models = {
  Transformer,
};

export const attention = {
  MultiHeadAttention,
};

// All model types
export type Model = Transformer;
