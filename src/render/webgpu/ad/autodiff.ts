/**
 * eval + reverse-mode AD over the IR (ir.ts).
 *
 * Two thin dispatchers, one clean handler per op:
 *   evalNode — numeric forward (memoized over the DAG). Used to gradient-check.
 *   vjp      — the reverse rule: given a node and its output-adjoint (a Node),
 *              return each input's adjoint contribution (also Nodes). The rules
 *              build IR, so `grad` produces a gradient GRAPH, differentiable and
 *              emittable exactly like the forward.
 *
 * The activation rules read the POST-activation node `n` where the derivative
 * allows it (tanh'=1−a², sigmoid'=a(1−a), selu from y, sqrt/exp from y) — the
 * "derivative-from-value" reuse that keeps the backward transcendental-free.
 * `sin` is the exception: its rule references cos(a) of the PRE-activation, which
 * is exactly why SIREN needs the pre-activation checkpointed. That fact falls
 * out of the rule; nobody hand-tracks it. See docs/PLAN_AD_IR_BACKWARD_CODEGEN.md.
 */
import { type Node, type Graph, topo, SELU_ALPHA, SELU_SCALE } from "./ir";

// --- numeric forward -------------------------------------------------------
export function evalNode(root: Node, env: Record<string, number>): number {
  return evalNodes([root], env)[0];
}

/**
 * Evaluate many roots sharing ONE cache — essential when the roots are gradient
 * graphs over a common forward (thousands of ∂L/∂wᵢ that share all the forward
 * and adjoint nodes): the shared subgraph is computed once, not once per root.
 */
export function evalNodes(roots: Node[], env: Record<string, number>): number[] {
  const cache = new Map<number, number>();
  const ev = (n: Node): number => {
    const hit = cache.get(n.id);
    if (hit !== undefined) return hit;
    const v = evalOp(n, ev, env);
    cache.set(n.id, v);
    return v;
  };
  return roots.map(ev);
}

function evalOp(n: Node, ev: (n: Node) => number, env: Record<string, number>): number {
  switch (n.op) {
    case "const": return n.value;
    case "input": {
      const v = env[n.name];
      if (v === undefined) throw new Error(`AD eval: missing input '${n.name}'`);
      return v;
    }
    case "add": return ev(n.inputs[0]) + ev(n.inputs[1]);
    case "sub": return ev(n.inputs[0]) - ev(n.inputs[1]);
    case "mul": return ev(n.inputs[0]) * ev(n.inputs[1]);
    case "div": return ev(n.inputs[0]) / ev(n.inputs[1]);
    case "min": return Math.min(ev(n.inputs[0]), ev(n.inputs[1]));
    case "max": return Math.max(ev(n.inputs[0]), ev(n.inputs[1]));
    case "gt": return ev(n.inputs[0]) > ev(n.inputs[1]) ? 1 : 0;
    case "atan2": return Math.atan2(ev(n.inputs[0]), ev(n.inputs[1]));
    case "mod": {
      const a = ev(n.inputs[0]), m = ev(n.inputs[1]);
      return a - Math.floor(a / m) * m;    // floored mod (matches tf.mod / WGSL)
    }
    case "neg": return -ev(n.inputs[0]);
    case "sin": return Math.sin(ev(n.inputs[0]));
    case "cos": return Math.cos(ev(n.inputs[0]));
    case "tanh": return Math.tanh(ev(n.inputs[0]));
    case "sigmoid": return 1 / (1 + Math.exp(-ev(n.inputs[0])));
    case "selu": {
      const x = ev(n.inputs[0]);
      return SELU_SCALE * (x > 0 ? x : SELU_ALPHA * (Math.exp(x) - 1));
    }
    case "exp": return Math.exp(ev(n.inputs[0]));
    case "log": return Math.log(ev(n.inputs[0]));
    case "sqrt": return Math.sqrt(ev(n.inputs[0]));
    case "abs": return Math.abs(ev(n.inputs[0]));
    case "select": return ev(n.inputs[0]) > 0 ? ev(n.inputs[1]) : ev(n.inputs[2]);
  }
}

// --- reverse rules ---------------------------------------------------------
// vjp(n, gr) returns the adjoint contribution for each of n.inputs, aligned by
// index. Leaves (const/input) have no inputs → []. `gr` is ∂L/∂n.
function vjp(g: Graph, n: Node, gr: Node): Node[] {
  const c = (v: number) => g.const(v);
  switch (n.op) {
    case "const":
    case "input":
      return [];
    case "add": return [gr, gr];
    case "sub": return [gr, g.neg(gr)];
    case "mul": {
      const [a, b] = n.inputs;
      return [g.mul(gr, b), g.mul(gr, a)];
    }
    case "div": {
      const [a, b] = n.inputs;
      // d/da = gr/b ; d/db = -gr·a/b²
      return [g.div(gr, b), g.neg(g.div(g.mul(gr, a), g.mul(b, b)))];
    }
    case "min": {
      const [a, b] = n.inputs;
      const aGtB = g.gt(a, b);                       // ties (a≤b) route to a
      return [g.select(aGtB, c(0), gr), g.select(aGtB, gr, c(0))];
    }
    case "max": {
      const [a, b] = n.inputs;
      const bGtA = g.gt(b, a);                       // ties (a≥b) route to a
      return [g.select(bGtA, c(0), gr), g.select(bGtA, gr, c(0))];
    }
    case "gt": return [c(0), c(0)];                  // step: zero gradient (mask)
    case "atan2": {
      // atan2(y,x): ∂/∂y = x/(y²+x²), ∂/∂x = −y/(y²+x²)
      const [y, x] = n.inputs;
      const r2 = g.add(g.mul(y, y), g.mul(x, x));
      return [g.div(g.mul(gr, x), r2), g.neg(g.div(g.mul(gr, y), r2))];
    }
    // floored mod: value wraps, but ∂/∂dividend = 1 (the floor step is
    // piecewise-constant). Matches tf.mod — the physics wrap stays differentiable.
    case "mod": return [gr, c(0)];
    case "neg": return [g.neg(gr)];
    case "sin": return [g.mul(gr, g.cos(n.inputs[0]))];   // needs cos(pre-act)
    case "cos": return [g.mul(gr, g.neg(g.sin(n.inputs[0])))];
    case "tanh": return [g.mul(gr, g.sub(c(1), g.mul(n, n)))];        // 1 − a²
    case "sigmoid": return [g.mul(gr, g.mul(n, g.sub(c(1), n)))];     // a(1−a)
    case "selu": {
      // a>0 → SCALE ; a≤0 → SCALE·ALPHA·eᵃ = y + SCALE·ALPHA  (from post-act y=n)
      const a = n.inputs[0];
      const d = g.select(g.gt(a, c(0)), c(SELU_SCALE), g.add(n, c(SELU_SCALE * SELU_ALPHA)));
      return [g.mul(gr, d)];
    }
    case "exp": return [g.mul(gr, n)];               // eᵃ = y
    case "log": return [g.div(gr, n.inputs[0])];
    case "sqrt": return [g.div(gr, g.mul(c(2), n))]; // 1/(2√a) = 1/(2y)
    case "abs": return [g.mul(gr, g.select(g.gt(n.inputs[0], c(0)), c(1), c(-1)))];
    case "select": {
      // cond gates which input receives gr; cond itself is non-diff.
      const [cond, , ] = n.inputs;
      return [c(0), g.select(cond, gr, c(0)), g.select(cond, c(0), gr)];
    }
  }
}

// --- forward-mode (JVP) rules ------------------------------------------------
// One tangent rule per op: given the node and its inputs' tangents, build the
// tangent node. Same derivative-from-value reuse as vjp (tanh'=1−a², exp'=y…),
// and the SAME tie conventions for min/max, so forward and reverse agree at
// kinks. Rules build IR — a JVP of a graph is a graph, so reverse-mode over a
// JVP output (reverse-over-forward) works with no extra machinery.
function jvpOp(g: Graph, n: Node, t: (x: Node) => Node): Node {
  const c = (v: number) => g.const(v);
  switch (n.op) {
    case "const": return c(0);
    case "input": return c(0);                       // unseeded leaf
    case "add": return g.add(t(n.inputs[0]), t(n.inputs[1]));
    case "sub": return g.sub(t(n.inputs[0]), t(n.inputs[1]));
    case "mul": {
      const [a, b] = n.inputs;
      return g.add(g.mul(t(a), b), g.mul(a, t(b)));
    }
    case "div": {
      // d(a/b) = (ta − n·tb)/b, reusing the quotient node n = a/b
      const b = n.inputs[1];
      return g.div(g.sub(t(n.inputs[0]), g.mul(n, t(b))), b);
    }
    case "min": {
      const [a, b] = n.inputs;
      return g.select(g.gt(a, b), t(b), t(a));       // ties (a≤b) route to a
    }
    case "max": {
      const [a, b] = n.inputs;
      return g.select(g.gt(b, a), t(b), t(a));       // ties (a≥b) route to a
    }
    case "gt": return c(0);
    case "atan2": {
      // d atan2(y,x) = (x·ty − y·tx)/(x²+y²)
      const [y, x] = n.inputs;
      const r2 = g.add(g.mul(y, y), g.mul(x, x));
      return g.div(g.sub(g.mul(x, t(y)), g.mul(y, t(x))), r2);
    }
    case "mod": return t(n.inputs[0]);               // floored mod: d/d dividend = 1
    case "neg": return g.neg(t(n.inputs[0]));
    case "sin": return g.mul(t(n.inputs[0]), g.cos(n.inputs[0]));
    case "cos": return g.neg(g.mul(t(n.inputs[0]), g.sin(n.inputs[0])));
    case "tanh": return g.mul(t(n.inputs[0]), g.sub(c(1), g.mul(n, n)));
    case "sigmoid": return g.mul(t(n.inputs[0]), g.mul(n, g.sub(c(1), n)));
    case "selu": {
      const a = n.inputs[0];
      const d = g.select(g.gt(a, c(0)), c(SELU_SCALE), g.add(n, c(SELU_SCALE * SELU_ALPHA)));
      return g.mul(t(n.inputs[0]), d);
    }
    case "exp": return g.mul(t(n.inputs[0]), n);
    case "log": return g.div(t(n.inputs[0]), n.inputs[0]);
    case "sqrt": return g.div(t(n.inputs[0]), g.mul(c(2), n));
    case "abs": return g.mul(t(n.inputs[0]), g.select(g.gt(n.inputs[0], c(0)), c(1), c(-1)));
    case "select": return g.select(n.inputs[0], t(n.inputs[1]), t(n.inputs[2]));
  }
}

/**
 * Forward-mode JVP: tangent graphs for `roots`, given seed tangents keyed by
 * NODE ID. Seeding by id (not name) is deliberate — it allows seeding DERIVED
 * nodes, e.g. the probe position pn = pos_K/res, which yields the spatial
 * jacobian ∂F/∂pn at the probe point (the exact replacement for the finite-
 * diff chaos/divergence probes — docs/PLAN_AD_IR_BACKWARD_CODEGEN.md §5.1).
 * A seed OVERRIDES the node's propagated tangent; unseeded leaves get 0.
 */
export function jvp(g: Graph, roots: Node[], seeds: Map<number, Node>): Node[] {
  const tan = new Map<number, Node>();
  const visit = (n: Node): Node => {
    const hit = tan.get(n.id);
    if (hit) return hit;
    const seeded = seeds.get(n.id);
    const tn = seeded ?? jvpOp(g, n, visit);
    tan.set(n.id, tn);
    return tn;
  };
  return roots.map(visit);
}

/**
 * Reverse-mode: ∂output/∂name for each requested input name, as gradient GRAPHS.
 * `seed` is ∂L/∂output (default 1 for a scalar-loss output).
 */
export function grad(
  g: Graph,
  output: Node,
  wrt: string[],
  seed?: Node
): Record<string, Node> {
  const order = topo(output);                        // inputs before node
  const adj = new Map<number, Node>();
  adj.set(output.id, seed ?? g.const(1));
  for (let i = order.length - 1; i >= 0; i--) {
    const n = order[i];
    const gr = adj.get(n.id);
    if (!gr) continue;
    const contribs = vjp(g, n, gr);
    n.inputs.forEach((inp, k) => {
      const prev = adj.get(inp.id);
      adj.set(inp.id, prev ? g.add(prev, contribs[k]) : contribs[k]);
    });
  }
  const out: Record<string, Node> = {};
  for (const name of wrt) {
    const leaf = g.input(name);                      // memoized — same node id
    out[name] = adj.get(leaf.id) ?? g.const(0);
  }
  return out;
}
