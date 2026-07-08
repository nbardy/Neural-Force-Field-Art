/**
 * AD engine verification (M0 + M1): reverse-mode gradients vs central finite
 * differences. Pure JS/TS, no GPU — this proves the source-to-source autodiff
 * (src/render/webgpu/ad) computes the EXACT analytical gradient before any of it
 * is wired into the WGSL trainer. See docs/PLAN_AD_IR_BACKWARD_CODEGEN.md.
 *
 *   bun tools/ad_test.ts
 *
 * Checks:
 *   1. per-op: each op's reverse rule matches FD (incl. selu<0, sin, min/max,
 *      select, sqrt/log/div — the pieces the losses and SIREN need).
 *   2. head:   a full standard 3-layer head (selu·selu·tanh) — reverse grad vs
 *      FD over ALL weights+biases+inputs, reported as cosine similarity + max
 *      rel error (the same metric the fused trainer's tfjs oracle uses; target
 *      cos = 1.0).
 */
import { Graph, type Node } from "../src/render/webgpu/ad/ir";
import { evalNode, grad } from "../src/render/webgpu/ad/autodiff";
import { emitWGSL, countTranscendentals } from "../src/render/webgpu/ad/emit_wgsl";

let failures = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
  if (!cond) failures++;
};

// deterministic PRNG so runs reproduce
let _s = 0x2545f491;
const rnd = () => {
  _s ^= _s << 13; _s ^= _s >>> 17; _s ^= _s << 5; _s |= 0;
  return ((_s >>> 0) / 0xffffffff) * 2 - 1; // [-1,1)
};

const EPS = 1e-4;
/** central finite difference of `f` w.r.t. env[name] */
function fd(f: () => number, env: Record<string, number>, name: string): number {
  const x0 = env[name];
  env[name] = x0 + EPS; const hi = f();
  env[name] = x0 - EPS; const lo = f();
  env[name] = x0;
  return (hi - lo) / (2 * EPS);
}

// --- 1. per-op reverse-vs-FD -----------------------------------------------
// Each entry builds an output from named inputs; we check every input's grad.
type OpCase = { name: string; build: (g: Graph) => Node; inputs: string[]; env: Record<string, number> };
const unary = (op: keyof Graph, at: number): OpCase => ({
  name: `${String(op)}(${at})`,
  build: (g) => (g[op] as (n: Node) => Node)(g.input("x")),
  inputs: ["x"],
  env: { x: at },
});
const cases: OpCase[] = [
  unary("sin", 0.7), unary("cos", -1.3), unary("tanh", 0.9),
  unary("sigmoid", -0.4), unary("selu", 0.8), unary("selu", -0.8), // both branches
  unary("exp", 0.3), unary("log", 1.7), unary("sqrt", 2.3), unary("abs", -1.1),
  unary("neg", 0.5),
  { name: "add", build: (g) => g.add(g.input("a"), g.input("b")), inputs: ["a", "b"], env: { a: 1.2, b: -0.7 } },
  { name: "sub", build: (g) => g.sub(g.input("a"), g.input("b")), inputs: ["a", "b"], env: { a: 1.2, b: -0.7 } },
  { name: "mul", build: (g) => g.mul(g.input("a"), g.input("b")), inputs: ["a", "b"], env: { a: 1.2, b: -0.7 } },
  { name: "div", build: (g) => g.div(g.input("a"), g.input("b")), inputs: ["a", "b"], env: { a: 1.2, b: -0.7 } },
  { name: "min", build: (g) => g.min(g.input("a"), g.input("b")), inputs: ["a", "b"], env: { a: 0.3, b: 0.9 } },
  { name: "max", build: (g) => g.max(g.input("a"), g.input("b")), inputs: ["a", "b"], env: { a: 0.3, b: 0.9 } },
  { name: "select-true", build: (g) => g.select(g.const(1), g.input("a"), g.input("b")), inputs: ["a", "b"], env: { a: 2.1, b: -0.5 } },
  { name: "select-false", build: (g) => g.select(g.const(-1), g.input("a"), g.input("b")), inputs: ["a", "b"], env: { a: 2.1, b: -0.5 } },
  // a composite that stresses the chain: log(sqrt(x²+y²)) — the chaos loss shape
  {
    name: "log(sqrt(x²+y²))",
    build: (g) => g.log(g.sqrt(g.add(g.mul(g.input("x"), g.input("x")), g.mul(g.input("y"), g.input("y"))))),
    inputs: ["x", "y"],
    env: { x: 0.6, y: -0.8 },
  },
];

for (const cse of cases) {
  const g = new Graph();
  const out = cse.build(g);
  const gr = grad(g, out, cse.inputs);
  let worst = 0;
  for (const nm of cse.inputs) {
    const rev = evalNode(gr[nm], cse.env);
    const num = fd(() => evalNode(out, cse.env), cse.env, nm);
    worst = Math.max(worst, Math.abs(rev - num) / (Math.abs(num) + 1e-6));
  }
  ok(worst < 2e-3, `op ${cse.name.padEnd(18)} reverse≈FD  (max rel err ${worst.toExponential(2)})`);
}

// --- 2. full standard head: reverse vs FD over every param -----------------
interface LayerDim { inSize: number; outSize: number; act: "selu" | "tanh" | "sigmoid" | "sin" }
// row-major weights: W[l][i][j] at name W${l}_${i}_${j}; bias b${l}_${j}
function buildHead(g: Graph, dims: LayerDim[]): Node[] {
  let cur: Node[] = Array.from({ length: dims[0].inSize }, (_, i) => g.input(`x${i}`));
  dims.forEach((L, l) => {
    const nxt: Node[] = [];
    for (let j = 0; j < L.outSize; j++) {
      let s = g.input(`b${l}_${j}`);
      for (let i = 0; i < L.inSize; i++) {
        s = g.add(s, g.mul(cur[i], g.input(`W${l}_${i}_${j}`)));
      }
      nxt.push(g.act(L.act, s));
    }
    cur = nxt;
  });
  return cur;
}

function headParamNames(dims: LayerDim[]): string[] {
  const names: string[] = [];
  for (let i = 0; i < dims[0].inSize; i++) names.push(`x${i}`);
  dims.forEach((L, l) => {
    for (let j = 0; j < L.outSize; j++) {
      names.push(`b${l}_${j}`);
      for (let i = 0; i < L.inSize; i++) names.push(`W${l}_${i}_${j}`);
    }
  });
  return names;
}

// standard helmholtz head shape: 2 → 8 → 8 → 2, selu·selu·tanh
const dims: LayerDim[] = [
  { inSize: 2, outSize: 8, act: "selu" },
  { inSize: 8, outSize: 8, act: "selu" },
  { inSize: 8, outSize: 2, act: "tanh" },
];

function checkHead(dims: LayerDim[], label: string) {
  const g = new Graph();
  const out = buildHead(g, dims);
  // scalar loss L = Σ_o c_o · out_o  (arbitrary covector = arbitrary dOut seed)
  const c = out.map(() => rnd());
  let loss: Node = g.const(0);
  out.forEach((o, i) => { loss = g.add(loss, g.mul(g.const(c[i]), o)); });

  const names = headParamNames(dims);
  const env: Record<string, number> = {};
  for (const nm of names) env[nm] = rnd() * 0.7;

  const gr = grad(g, loss, names);
  const rev: number[] = [];
  const num: number[] = [];
  for (const nm of names) {
    rev.push(evalNode(gr[nm], env));
    num.push(fd(() => evalNode(loss, env), env, nm));
  }
  // cosine similarity + max rel error — the fused-trainer oracle metric
  let dot = 0, nr = 0, nn = 0, maxRel = 0;
  for (let i = 0; i < rev.length; i++) {
    dot += rev[i] * num[i]; nr += rev[i] * rev[i]; nn += num[i] * num[i];
    maxRel = Math.max(maxRel, Math.abs(rev[i] - num[i]) / (Math.abs(num[i]) + 1e-4));
  }
  const cos = dot / (Math.sqrt(nr) * Math.sqrt(nn) + 1e-30);
  ok(cos > 0.99999 && maxRel < 5e-3,
    `${label}: ${names.length} params, cos=${cos.toFixed(7)}, maxRel=${maxRel.toExponential(2)}`);
}

checkHead(dims, "standard head selu·selu·tanh");
// SIREN head — proves sin's rule (cos-of-pre-act) is correct end-to-end
checkHead(
  [{ inSize: 2, outSize: 8, act: "sin" }, { inSize: 8, outSize: 8, act: "sin" }, { inSize: 8, outSize: 2, act: "tanh" }],
  "SIREN head sin·sin·tanh"
);
// sigmoid head — the third activation option
checkHead(
  [{ inSize: 2, outSize: 6, act: "sigmoid" }, { inSize: 6, outSize: 2, act: "tanh" }],
  "sigmoid head"
);

// --- 3. WGSL emit + checkpoint mechanism (architectural invariants) ---------
// The backward reads stored post-activations from scratch (checkpoints) rather
// than recomputing the forward. These checks prove that mechanism, and prove the
// SIREN storage requirement surfaces from the sin rule itself — not by hand.
{
  // tanh: dS = dOut·(1 − a²). With post-act `a` stored → transcendental-free.
  const g = new Graph();
  const s = g.input("s");
  const a = g.tanh(s);
  const gS = grad(g, a, ["s"], g.input("dOut"))["s"];
  const withCkpt = emitWGSL([{ node: gS, out: "dS" }], { stored: new Map([[a.id, "aStored"]]) });
  const noCkpt = emitWGSL([{ node: gS, out: "dS" }]);
  const tc = countTranscendentals(withCkpt);
  const transTotal = Object.values(tc).reduce((x, y) => x + y, 0);
  ok(transTotal === 0, `tanh backward w/ checkpoint is transcendental-free (${transTotal} calls)`);
  ok(countTranscendentals(noCkpt).tanh === 1, `tanh backward w/o checkpoint recomputes tanh (shows why we store)`);
}
{
  // sin: dS = dOut·cos(s). cos is NOT recoverable from post-act sin(s), so the
  // backward MUST contain cos of the PRE-activation even with the post-act
  // stored — this is the SIREN checkpoint requirement, surfaced by the rule.
  const g = new Graph();
  const s = g.input("s");
  const a = g.sin(s);
  const gS = grad(g, a, ["s"], g.input("dOut"))["s"];
  const code = emitWGSL([{ node: gS, out: "dS" }], { stored: new Map([[a.id, "aStored"]]) });
  const tc = countTranscendentals(code);
  ok(tc.cos === 1 && tc.sin === 0,
    `SIREN backward forces cos(pre-act) even with post-act stored (cos=${tc.cos}, sin=${tc.sin}) — the storage requirement, mechanical`);
}
{
  // show it's real, tight WGSL: emit a full standard head's input-gradient
  // (bwd_head's du) with all post-activations checkpointed, print line count.
  const g = new Graph();
  const out = buildHead(g, dims);
  // input grads du = dL/dx, seeded by an arbitrary dOut covector on the 2 outputs
  const grads = grad(
    g,
    g.sum(out.map((o, i) => g.mul(g.input(`dOut${i}`), o))),
    ["x0", "x1"]
  );
  // checkpoint every activation output (what scratch actually holds)
  const stored = new Map<number, string>();
  let ci = 0;
  const markActs = (n: Node, seen = new Set<number>()) => {
    if (seen.has(n.id)) return; seen.add(n.id);
    for (const c of n.inputs) markActs(c, seen);
    if (["selu", "tanh", "sigmoid", "sin"].includes(n.op)) stored.set(n.id, `A${ci++}`);
  };
  markActs(out[0]); markActs(out[1]);
  const code = emitWGSL(
    [{ node: grads.x0, out: "du.x" }, { node: grads.x1, out: "du.y" }],
    { stored }
  );
  const lineCount = code.split("\n").length;
  const tc = countTranscendentals(code);
  const transTotal = Object.values(tc).reduce((x, y) => x + y, 0);
  ok(transTotal === 0,
    `standard head backward (${lineCount} WGSL lines) is transcendental-free with activations checkpointed`);
  console.log("\n  --- generated standard-head input-gradient WGSL (excerpt) ---");
  console.log(code.split("\n").slice(0, 8).map((l) => "  " + l).join("\n"));
  console.log(`  ... (${lineCount} lines total)\n`);
}

console.log(failures === 0 ? "\nALL AD CHECKS PASS" : `\n${failures} AD CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
