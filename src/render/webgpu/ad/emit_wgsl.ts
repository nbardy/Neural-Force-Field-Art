/**
 * IR → WGSL. Emits each interior node once as an SSA `let`, in topological
 * order. Because the IR is a DAG (the builder and the vjp rules share nodes),
 * one-let-per-node IS common-subexpression elimination — no separate CSE pass
 * needed; the sharing is structural. Consts and inputs are inlined.
 *
 * `input` nodes emit their bare name, so the caller is responsible for having
 * those names in WGSL scope (function params, prior `let`s, or scratch reads).
 * That is the seam where this generated math body plugs into the hand-written
 * harness (bindings/reduction/Adam). See docs/PLAN_AD_IR_BACKWARD_CODEGEN.md.
 */
import { type Node, SELU_ALPHA, SELU_SCALE } from "./ir";

/** f32 literal that always carries a decimal point (so WGSL types it as f32). */
export function f32lit(v: number): string {
  if (!Number.isFinite(v)) throw new Error(`AD emit: non-finite const ${v}`);
  if (Number.isInteger(v)) return `${v}.0`;
  return v.toPrecision(9).replace(/0+$/, "").replace(/\.$/, ".0");
}

// Walk stops at `stored` nodes: they are checkpoints already in scope (read from
// scratch), so we neither recurse into them nor emit their definition. This is
// how the backward avoids recomputing the forward — mark the post-activations
// (and any pre-activation a rule demanded) as stored and they become leaves.
function topoMany(roots: Node[], stored: Map<number, string>): Node[] {
  const seen = new Set<number>();
  const out: Node[] = [];
  const visit = (n: Node) => {
    if (seen.has(n.id) || stored.has(n.id)) return;
    seen.add(n.id);
    for (const c of n.inputs) visit(c);
    out.push(n);
  };
  for (const r of roots) visit(r);
  return out;
}

// one clean handler per op: node + child-referencer → WGSL expression string.
function exprOf(n: Node, ref: (n: Node) => string): string {
  // a()/b() are only reached in branches whose op has that arity.
  const ins = n.inputs as Node[];
  const a = () => ref(ins[0]);
  const b = () => ref(ins[1]);
  switch (n.op) {
    case "const": return f32lit(n.value);
    case "input": return n.name;
    case "add": return `(${a()} + ${b()})`;
    case "sub": return `(${a()} - ${b()})`;
    case "mul": return `(${a()} * ${b()})`;
    case "div": return `(${a()} / ${b()})`;
    case "min": return `min(${a()}, ${b()})`;
    case "max": return `max(${a()}, ${b()})`;
    case "gt": return `select(0.0, 1.0, ${a()} > ${b()})`;
    case "atan2": return `atan2(${a()}, ${b()})`;
    case "mod": return `(${a()} - floor(${a()} / ${b()}) * ${b()})`;
    case "neg": return `(-${a()})`;
    case "sin": return `sin(${a()})`;
    case "cos": return `cos(${a()})`;
    case "tanh": return `tanh(${a()})`;
    case "sigmoid": return `(1.0 / (1.0 + exp(-${a()})))`;
    case "selu": return `(${f32lit(SELU_SCALE)} * select(${f32lit(SELU_ALPHA)} * (exp(${a()}) - 1.0), ${a()}, ${a()} > 0.0))`;
    case "exp": return `exp(${a()})`;
    case "log": return `log(${a()})`;
    case "sqrt": return `sqrt(${a()})`;
    case "abs": return `abs(${a()})`;
    // select(cond,x,y) = cond>0 ? x : y  → WGSL select(falseVal, trueVal, cond)
    case "select": return `select(${ref(n.inputs[2])}, ${ref(n.inputs[1])}, ${ref(n.inputs[0])} > 0.0)`;
  }
}

export interface EmitRoot { node: Node; out: string }
export interface EmitOpts {
  indent?: string;
  /** node id → WGSL name of a value already in scope (a scratch checkpoint). */
  stored?: Map<number, string>;
}

/** Emit WGSL statements computing each root into its `out` lvalue. */
export function emitWGSL(roots: EmitRoot[], opts: EmitOpts = {}): string {
  const indent = opts.indent ?? "  ";
  const stored = opts.stored ?? new Map<number, string>();
  const order = topoMany(roots.map((r) => r.node), stored);
  const varName = new Map<number, string>();
  const ref = (n: Node): string =>
    stored.has(n.id) ? stored.get(n.id)!
      : n.op === "const" ? f32lit(n.value)
        : n.op === "input" ? n.name
          : varName.get(n.id)!;
  const lines: string[] = [];
  for (const n of order) {
    if (n.op === "const" || n.op === "input") continue;
    const v = `v${n.id}`;
    varName.set(n.id, v);
    lines.push(`${indent}let ${v} = ${exprOf(n, ref)};`);
  }
  for (const r of roots) lines.push(`${indent}${r.out} = ${ref(r.node)};`);
  return lines.join("\n");
}

/** Count transcendental calls in emitted WGSL — used by architectural-invariant tests. */
export function countTranscendentals(wgsl: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const fn of ["sin", "cos", "tanh", "exp", "log", "sqrt"]) {
    out[fn] = (wgsl.match(new RegExp(`\\b${fn}\\(`, "g")) ?? []).length;
  }
  return out;
}
