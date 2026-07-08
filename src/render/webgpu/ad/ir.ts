/**
 * Minimal expression-graph IR for source-to-source autodiff.
 *
 * Nodes form a DAG. Each op has ONE eval handler (eval.ts) and ONE reverse rule
 * (autodiff.ts). The reverse rule BUILDS more IR — the adjoint of a graph is
 * itself a graph (JAX-style `grad` returns a new function). One consequence we
 * rely on: the SAME rules drive both numeric gradient-checking (interp) and
 * WGSL emission, so no per-op derivative is written twice. See
 * docs/PLAN_AD_IR_BACKWARD_CODEGEN.md.
 *
 * Scalars only, deliberately: the trainer codegen unrolls every loop at compile
 * time (sizes are known), so a vector op buys nothing here — an unrolled scalar
 * DAG is exactly the shape the WGSL emitter wants.
 */

// D = ⊕ᵢ Dᵢ — the op sum type. `select(c,a,b)` = c > 0 ? a : b (WGSL `select`
// argument order is (b,a,cond); the emitter flips it). `gt(a,b)` yields 1/0.
export type Node =
  | { id: number; op: "const"; value: number; inputs: [] }
  | { id: number; op: "input"; name: string; inputs: [] }
  | { id: number; op: "add"; inputs: [Node, Node] }
  | { id: number; op: "sub"; inputs: [Node, Node] }
  | { id: number; op: "mul"; inputs: [Node, Node] }
  | { id: number; op: "div"; inputs: [Node, Node] }
  | { id: number; op: "min"; inputs: [Node, Node] }
  | { id: number; op: "max"; inputs: [Node, Node] }
  | { id: number; op: "gt"; inputs: [Node, Node] }
  | { id: number; op: "atan2"; inputs: [Node, Node] }   // atan2(y, x), WGSL order
  | { id: number; op: "mod"; inputs: [Node, Node] }     // floored mod; grad ≡ identity on dividend
  | { id: number; op: "neg"; inputs: [Node] }
  | { id: number; op: "sin"; inputs: [Node] }
  | { id: number; op: "cos"; inputs: [Node] }
  | { id: number; op: "tanh"; inputs: [Node] }
  | { id: number; op: "sigmoid"; inputs: [Node] }
  | { id: number; op: "selu"; inputs: [Node] }
  | { id: number; op: "exp"; inputs: [Node] }
  | { id: number; op: "log"; inputs: [Node] }
  | { id: number; op: "sqrt"; inputs: [Node] }
  | { id: number; op: "abs"; inputs: [Node] }
  | { id: number; op: "select"; inputs: [Node, Node, Node] };

// SELU constants (tfjs / Keras defaults) — kept here so eval and the derivative
// rule read the identical numbers.
export const SELU_ALPHA = 1.6732632423543772;
export const SELU_SCALE = 1.0507009873554805;

/**
 * Graph builder. Holds the SSA counter and memoizes `input(name)` so every
 * reference to the same named leaf is the SAME node — required for reverse mode
 * to accumulate ∂L/∂name across all its uses.
 */
export class Graph {
  private next = 0;
  private inputs = new Map<string, Node>();

  // Internal factory. `Omit<Node,"id">` distributes over the union to only the
  // common fields, so the literal below is typed loosely and cast — the public
  // builder methods are what enforce correct per-op shapes.
  private mk(n: { op: Node["op"]; inputs: Node[]; [k: string]: unknown }): Node {
    return { id: this.next++, ...n } as Node;
  }

  const(value: number): Node {
    return this.mk({ op: "const", value, inputs: [] });
  }
  input(name: string): Node {
    const hit = this.inputs.get(name);
    if (hit) return hit;
    const n = this.mk({ op: "input", name, inputs: [] });
    this.inputs.set(name, n);
    return n;
  }

  add(a: Node, b: Node): Node { return this.mk({ op: "add", inputs: [a, b] }); }
  sub(a: Node, b: Node): Node { return this.mk({ op: "sub", inputs: [a, b] }); }
  mul(a: Node, b: Node): Node { return this.mk({ op: "mul", inputs: [a, b] }); }
  div(a: Node, b: Node): Node { return this.mk({ op: "div", inputs: [a, b] }); }
  min(a: Node, b: Node): Node { return this.mk({ op: "min", inputs: [a, b] }); }
  max(a: Node, b: Node): Node { return this.mk({ op: "max", inputs: [a, b] }); }
  gt(a: Node, b: Node): Node { return this.mk({ op: "gt", inputs: [a, b] }); }
  atan2(y: Node, x: Node): Node { return this.mk({ op: "atan2", inputs: [y, x] }); }
  mod(a: Node, m: Node): Node { return this.mk({ op: "mod", inputs: [a, m] }); }

  neg(a: Node): Node { return this.mk({ op: "neg", inputs: [a] }); }
  sin(a: Node): Node { return this.mk({ op: "sin", inputs: [a] }); }
  cos(a: Node): Node { return this.mk({ op: "cos", inputs: [a] }); }
  tanh(a: Node): Node { return this.mk({ op: "tanh", inputs: [a] }); }
  sigmoid(a: Node): Node { return this.mk({ op: "sigmoid", inputs: [a] }); }
  selu(a: Node): Node { return this.mk({ op: "selu", inputs: [a] }); }
  exp(a: Node): Node { return this.mk({ op: "exp", inputs: [a] }); }
  log(a: Node): Node { return this.mk({ op: "log", inputs: [a] }); }
  sqrt(a: Node): Node { return this.mk({ op: "sqrt", inputs: [a] }); }
  abs(a: Node): Node { return this.mk({ op: "abs", inputs: [a] }); }
  select(cond: Node, a: Node, b: Node): Node {
    return this.mk({ op: "select", inputs: [cond, a, b] });
  }

  /** activation by name — the dispatch a field-type declaration goes through. */
  act(name: "selu" | "tanh" | "sigmoid" | "sin", x: Node): Node {
    return name === "selu" ? this.selu(x)
      : name === "tanh" ? this.tanh(x)
      : name === "sigmoid" ? this.sigmoid(x)
      : this.sin(x);
  }

  /** sum a list (left fold); empty → const 0. */
  sum(xs: Node[]): Node {
    if (xs.length === 0) return this.const(0);
    return xs.reduce((a, b) => this.add(a, b));
  }
}

/** Post-order (inputs-before-node) topological walk of everything under `root`. */
export function topo(root: Node): Node[] {
  const seen = new Set<number>();
  const out: Node[] = [];
  const visit = (n: Node) => {
    if (seen.has(n.id)) return;
    seen.add(n.id);
    for (const c of n.inputs) visit(c);
    out.push(n);
  };
  visit(root);
  return out;
}
