import { normalize } from "./linalg";
import { Transformer } from "./blocks/clipTransformer";
import { MultiHeadAttention } from "./layers/multiheadattention";

export const trashPanda = {
  linalg: {
    normalize,
  },
  models: {
    Transformer,
  },
  attention: {
    MultiHeadAttention,
  },
};

// default
export default trashPanda;
