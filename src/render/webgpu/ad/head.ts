/**
 * MLP head, built in the IR. `wName`/`bName` return the leaf NAME for each
 * weight/bias — that indirection is the seam that lets the SAME builder serve
 * both numeric verification (names → env values) and WGSL emission (names →
 * `weights[offset]` expressions). See docs/PLAN_AD_IR_BACKWARD_CODEGEN.md §4.
 *
 * Weights are row-major kernels (tfjs Dense convention): W[l][i][j] multiplies
 * input i into output j; bias b[l][j]. Fully unrolled — the codegen unrolls too.
 */
import { Graph, type Node } from "./ir";

export interface HeadDim {
  inSize: number;
  outSize: number;
  act: "selu" | "tanh" | "sigmoid" | "sin";
}

export function buildHead(
  g: Graph,
  dims: HeadDim[],
  input: Node[],
  wName: (l: number, i: number, j: number) => string,
  bName: (l: number, j: number) => string
): Node[] {
  let cur = input;
  dims.forEach((L, l) => {
    const nxt: Node[] = [];
    for (let j = 0; j < L.outSize; j++) {
      let s = g.input(bName(l, j));
      for (let i = 0; i < L.inSize; i++) {
        s = g.add(s, g.mul(cur[i], g.input(wName(l, i, j))));
      }
      nxt.push(g.act(L.act, s));
    }
    cur = nxt;
  });
  return cur;
}
