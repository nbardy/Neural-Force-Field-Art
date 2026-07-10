/**
 * M5 verification: forward-mode JVP through the AD IR (plan §5.1).
 *
 *   bun tools/ad_jvp_test.ts
 *
 * Checks:
 *   1. per-op JVP ≈ central finite differences (every differentiable op)
 *   2. forward ≡ reverse: the head jacobian via JVP (columns) matches the
 *      SAME jacobian via reverse-mode grad (rows) — two independent orderings
 *      of the exact chain rule, so they must agree to fp noise
 *   3. probe convergence: the FD chaos/divergence terms → the exact JVP
 *      terms as h → 0 (each h decade shrinks the gap ~10× — O(h) truncation)
 *   4. training-gradient convergence: ∂L/∂w of the FD-probe loss → ∂L/∂w of
 *      the exact-probe loss as h → 0 (reverse-over-forward works)
 */
import { Graph, type Node } from "../src/render/webgpu/ad/ir";
import { evalNode, evalNodes, grad, jvp } from "../src/render/webgpu/ad/autodiff";
import { buildHead, type HeadDim } from "../src/render/webgpu/ad/head";
import { buildSample, wName, bName, type RolloutCfg } from "../src/render/webgpu/ad/rollout";

let failures = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
  if (!cond) failures++;
};

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- 1: per-op JVP vs central FD --------------------------------------------
{
  const EPS = 1e-5;
  const unary: [string, (g: Graph, x: Node) => Node, number[]][] = [
    ["neg", (g, x) => g.neg(x), [0.7, -1.3]],
    ["sin", (g, x) => g.sin(x), [0.7, -1.3]],
    ["cos", (g, x) => g.cos(x), [0.7, -1.3]],
    ["tanh", (g, x) => g.tanh(x), [0.7, -1.3]],
    ["sigmoid", (g, x) => g.sigmoid(x), [0.7, -1.3]],
    ["selu", (g, x) => g.selu(x), [0.7, -1.3]],
    ["exp", (g, x) => g.exp(x), [0.7, -1.3]],
    ["log", (g, x) => g.log(x), [0.7, 2.1]],
    ["sqrt", (g, x) => g.sqrt(x), [0.7, 2.1]],
    ["abs", (g, x) => g.abs(x), [0.7, -1.3]],
  ];
  let worst = 0;
  for (const [name, build, pts] of unary) {
    for (const v of pts) {
      const g = new Graph();
      const x = g.input("x");
      const y = build(g, x);
      const [ty] = jvp(g, [y], new Map([[x.id, g.const(1)]]));
      const got = evalNode(ty, { x: v });
      const fd =
        (evalNode(y, { x: v + EPS }) - evalNode(y, { x: v - EPS })) / (2 * EPS);
      worst = Math.max(worst, Math.abs(got - fd));
    }
    void name;
  }
  const binary: [string, (g: Graph, a: Node, b: Node) => Node, [number, number][]][] = [
    ["add", (g, a, b) => g.add(a, b), [[0.7, -1.3]]],
    ["sub", (g, a, b) => g.sub(a, b), [[0.7, -1.3]]],
    ["mul", (g, a, b) => g.mul(a, b), [[0.7, -1.3]]],
    ["div", (g, a, b) => g.div(a, b), [[0.7, -1.3]]],
    ["min", (g, a, b) => g.min(a, b), [[0.7, -1.3], [-0.4, 0.9]]],
    ["max", (g, a, b) => g.max(a, b), [[0.7, -1.3], [-0.4, 0.9]]],
    ["atan2", (g, a, b) => g.atan2(a, b), [[0.7, -1.3], [-0.4, 0.9]]],
    // mod's divisor tangent is DEFINED as 0 (tf.mod convention; the divisor is
    // always the constant [W,H] here) — probe the dividend direction only.
    ["mod", (g, a, b) => g.mod(a, b), [[7.3, 2.0]]],
  ];
  for (const [name, build, pts] of binary) {
    const db = name === "mod" ? 0 : 0.5; // direction (1, db)
    for (const [va, vb] of pts) {
      const g = new Graph();
      const a = g.input("a"), b = g.input("b");
      const y = build(g, a, b);
      const [ty] = jvp(g, [y], new Map([[a.id, g.const(1)], [b.id, g.const(db)]]));
      const got = evalNode(ty, { a: va, b: vb });
      const f = (da: number, dbb: number) => evalNode(y, { a: va + da, b: vb + dbb });
      const fd = (f(EPS, db * EPS) - f(-EPS, -db * EPS)) / (2 * EPS);
      worst = Math.max(worst, Math.abs(got - fd));
    }
  }
  ok(worst < 1e-6, `per-op JVP ≈ central FD (worst |Δ| ${worst.toExponential(2)})`);
}

// --- 2: forward ≡ reverse jacobian on real heads -----------------------------
for (const [label, dims] of [
  ["standard selu·selu·tanh", [
    { inSize: 2, outSize: 8, act: "selu" },
    { inSize: 8, outSize: 8, act: "selu" },
    { inSize: 8, outSize: 2, act: "tanh" },
  ]],
  ["SIREN sin·sin·tanh", [
    { inSize: 2, outSize: 8, act: "sin" },
    { inSize: 8, outSize: 8, act: "sin" },
    { inSize: 8, outSize: 2, act: "tanh" },
  ]],
] as [string, HeadDim[]][]) {
  const g = new Graph();
  const rnd = mulberry32(1234);
  const env: Record<string, number> = { x: 0.37, y: 0.81 };
  const w = (l: number, i: number, j: number) => `w_${l}_${i}_${j}`;
  const b = (l: number, j: number) => `b_${l}_${j}`;
  dims.forEach((L, l) => {
    for (let i = 0; i < L.inSize; i++)
      for (let j = 0; j < L.outSize; j++) env[w(l, i, j)] = (rnd() - 0.5) * 1.6;
    for (let j = 0; j < L.outSize; j++) env[b(l, j)] = (rnd() - 0.5) * 0.4;
  });
  const xi = g.input("x"), yi = g.input("y");
  const out = buildHead(g, dims, [xi, yi], w, b);
  // forward: columns J·x̂, J·ŷ
  const [JxxF, JyxF] = jvp(g, out, new Map([[xi.id, g.const(1)], [yi.id, g.const(0)]]));
  const [JxyF, JyyF] = jvp(g, out, new Map([[xi.id, g.const(0)], [yi.id, g.const(1)]]));
  // reverse: rows ∇Fx, ∇Fy
  const gFx = grad(g, out[0], ["x", "y"]);
  const gFy = grad(g, out[1], ["x", "y"]);
  const vals = evalNodes(
    [JxxF, JyxF, JxyF, JyyF, gFx.x, gFx.y, gFy.x, gFy.y],
    env
  );
  const d = Math.max(
    Math.abs(vals[0] - vals[4]), // ∂Fx/∂x
    Math.abs(vals[1] - vals[6]), // ∂Fy/∂x
    Math.abs(vals[2] - vals[5]), // ∂Fx/∂y
    Math.abs(vals[3] - vals[7])  // ∂Fy/∂y
  );
  ok(d < 1e-12, `${label}: JVP columns ≡ reverse rows (max |Δ| ${d.toExponential(2)})`);
}

// --- 3+4: exact probes = the h→0 limit of the FD probes ----------------------
{
  const dims: HeadDim[] = [
    { inSize: 2, outSize: 8, act: "selu" },
    { inSize: 8, outSize: 8, act: "selu" },
    { inSize: 8, outSize: 2, act: "tanh" },
  ];
  const base: RolloutCfg = {
    K: 1, alpha: 0.7, hh: 1e-2,
    forceMag: 3.5, friction: 0.99, maxVel: 26,
    W: 800, H: 600, spiralTurns: 3,
    wChaos: 1.0, wDiv: 0.5, wSpiral: 0.00002, wIso: 1.0,
    N: 4,
  };
  const rnd = mulberry32(42);
  const env: Record<string, number> = { dC00: 0.01, dC11: -0.02, dC01: 0.005 };
  for (const h of [0, 1]) {
    dims.forEach((L, l) => {
      for (let i = 0; i < L.inSize; i++)
        for (let j = 0; j < L.outSize; j++) env[wName(h)(l, i, j)] = (rnd() - 0.5) * 1.2;
      for (let j = 0; j < L.outSize; j++) env[bName(h)(l, j)] = (rnd() - 0.5) * 1.2;
    });
  }
  const names: string[] = [];
  for (const h of [0, 1])
    dims.forEach((L, l) => {
      for (let i = 0; i < L.inSize; i++)
        for (let j = 0; j < L.outSize; j++) names.push(wName(h)(l, i, j));
      for (let j = 0; j < L.outSize; j++) names.push(bName(h)(l, j));
    });

  const gJ = new Graph();
  const exact = buildSample(gJ, { ...base, probes: "jvp" }, dims, dims);
  const exactGrads = grad(gJ, exact.L, names);
  const exactRoots = [exact.chaos, exact.div, ...names.map((n) => exactGrads[n])];

  const pts: [number, number][] = [[123.4, 456.7], [700.1, 88.8], [12.3, 590.0], [401.5, 300.2]];
  let prevTermErr = Infinity;
  let prevCosGap = Infinity;
  let termShrinks = true;
  let gradImproves = true;
  for (const hh of [1e-2, 1e-3, 1e-4]) {
    const gF = new Graph();
    const fd = buildSample(gF, { ...base, hh }, dims, dims);
    const fdGrads = grad(gF, fd.L, names);
    const fdRoots = [fd.chaos, fd.div, ...names.map((n) => fdGrads[n])];
    let termErr = 0;
    let dot = 0, nf = 0, ne = 0;
    for (const [px, py] of pts) {
      env.p0x = px; env.p0y = py;
      const ev = evalNodes(exactRoots, env);
      const fv = evalNodes(fdRoots, env);
      termErr = Math.max(
        termErr,
        Math.abs(ev[0] - fv[0]) / (Math.abs(ev[0]) + 1e-9),
        Math.abs(ev[1] - fv[1]) / (Math.abs(ev[1]) + 1e-9)
      );
      for (let i = 2; i < ev.length; i++) {
        dot += ev[i] * fv[i]; ne += ev[i] * ev[i]; nf += fv[i] * fv[i];
      }
    }
    const cos = dot / (Math.sqrt(ne * nf) + 1e-30);
    console.log(
      `        h=${hh.toExponential(0)}: probe-term relΔ ${termErr.toExponential(2)}, grad cos ${cos.toFixed(9)}`
    );
    if (termErr >= prevTermErr) termShrinks = false;
    if (1 - cos >= prevCosGap) gradImproves = false;
    prevTermErr = termErr;
    prevCosGap = 1 - cos;
  }
  ok(termShrinks && prevTermErr < 1e-3,
    `FD chaos/div terms converge to the exact JVP terms as h→0 (final relΔ ${prevTermErr.toExponential(2)})`);
  ok(gradImproves && prevCosGap < 1e-6,
    `FD-loss ∂L/∂w converges to the exact-probe ∂L/∂w (reverse-over-forward; final 1−cos ${prevCosGap.toExponential(2)})`);
}

console.log(failures === 0 ? "\nM5 JVP CHECKS PASS" : `\n${failures} M5 CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
