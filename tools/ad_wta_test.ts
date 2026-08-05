/**
 * WTA adversary IR-oracle verification — src/render/webgpu/ad/losses.ts::wtaTerm.
 *
 *   bun tools/ad_wta_test.ts            # all sections (GPU section needs Metal)
 *   SKIP_GPU=1 bun tools/ad_wta_test.ts # pure-JS sections only
 *
 * Three gates, each falsifiable:
 *
 *  §1 REVERSE vs CENTRAL FD, per encoding width (pair du=1/dy=2, tri du=3/dy=6)
 *     × K ∈ {2,3,4}: gradients of BOTH the discriminator objective (relaxed
 *     weighted sum) and the generator's identical shared payoff, over ALL adversary
 *     params AND the context/target inputs. cos > 0.99999 + maxRel. Plus the
 *     hard-WTA (ε=0) freeze property: a loser head's weight gradients are
 *     EXACTLY zero (the "starved head frozen" fact adversary_test.ts §2
 *     measures in training is here an exact algebraic gate on the graph).
 *
 *  §2 CROSS-CHECK vs THE REAL tfjs Adversary (src/core/gan/adversary.ts) on an
 *     identical fixture: same head weights (read out of the tfjs models into
 *     the IR leaves), same u/y batch. Asserts the tfjs-minimized discriminator
 *     loss and the IR batch mean agree to rel < 1e-5; discriminator gradient
 *     over every head weight and the generator gradient d loss/d y (and /d u)
 *     agree at cos > 0.99999; and the per-tuple argmin winners agree exactly.
 *     THIS is the gate on the relaxation-weight semantics: tfjs computes the
 *     relaxed weights OUTSIDE optimizer.minimize and passes them as data
 *     (constants on the tape); the IR's gt-built weights have zero-gradient
 *     vjp. If either side differentiated through the winner selection and the
 *     other did not, cos would fall off 1.
 *
 *  §3 EMITTED-WGSL-ON-GPU — the gate this repo has never had. Emits the
 *     explicit soft-angle objective's forward AND full backward via emitWGSL,
 *     compiles and dispatches it on
 *     real Metal (bun-webgpu), one thread per tuple, and compares every output
 *     (payoff aliases, all d weighted/d θ, d payoff/d u,y) against the
 *     JS oracle on identical f32 inputs. This closes the f32lit-class hole:
 *     an emitter bug is invisible to every pure-JS test because f32lit and
 *     exprOf run ONLY on the WGSL path (see the f32lit incident note in
 *     tools/ad_test.ts). Also the first-ever proof that IR-emitted WGSL
 *     compiles and runs at all.
 */
import * as tf from "@tensorflow/tfjs";
import { Graph, type Node } from "../src/render/webgpu/ad/ir";
import { evalNodes, grad } from "../src/render/webgpu/ad/autodiff";
import { emitWGSL } from "../src/render/webgpu/ad/emit_wgsl";
import {
  wtaTerm,
  wtaObjectiveTerm,
  softSphereChordTerm,
  localRelativeScaleTerm,
  energyAnchorTerm,
  postVelocityTargetTerm,
  awName,
  abName,
  type WtaMetric,
} from "../src/render/webgpu/ad/losses";
import { type HeadDim } from "../src/render/webgpu/ad/head";
import {
  Adversary,
  defaultAdversaryConfig,
  encodingDims,
  disposeAdversary,
  type TupleEncoding,
  type TupleSample,
} from "../src/core/gan/adversary";

await tf.setBackend("cpu");
await tf.ready();

let failures = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
  if (!cond) failures++;
};

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Canonical adversary parameter-name order: per head, per layer, biases then kernels. */
function paramNames(k: number, dims: HeadDim[]): string[] {
  const names: string[] = [];
  for (let j = 0; j < k; j++) {
    dims.forEach((L, l) => {
      for (let o = 0; o < L.outSize; o++) names.push(abName(j)(l, o));
      for (let i = 0; i < L.inSize; i++)
        for (let o = 0; o < L.outSize; o++) names.push(awName(j)(l, i, o));
    });
  }
  return names;
}

function cosine(a: number[] | Float32Array, b: number[] | Float32Array): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-30);
}

/* ══════════════════════════════════════════════════════════════════════════
   §1  reverse-mode vs central finite differences
   ══════════════════════════════════════════════════════════════════════════ */
console.log("--- 1. wtaTerm reverse-mode vs central FD (all params + u,y) ---");

const FD_EPS = 1e-4;
function fdOf(f: () => number, env: Record<string, number>, name: string): number {
  const x0 = env[name];
  env[name] = x0 + FD_EPS;
  const hi = f();
  env[name] = x0 - FD_EPS;
  const lo = f();
  env[name] = x0;
  return (hi - lo) / (2 * FD_EPS);
}

function fdCheck(
  label: string,
  du: number,
  dy: number,
  k: number,
  eps: number,
  seed: number,
  metric: WtaMetric = "euclidean"
) {
  const dims: HeadDim[] = [
    { inSize: du, outSize: 6, act: "selu" },
    { inSize: 6, outSize: 6, act: "selu" },
    { inSize: 6, outSize: dy, act: "linear" },
  ];
  const g = new Graph();
  const uN = Array.from({ length: du }, (_, i) => g.input(`u${i}`));
  const yN = Array.from({ length: dy }, (_, o) => g.input(`y${o}`));
  const term = wtaTerm(g, uN, yN, dims, k, eps, metric);

  const pNames = paramNames(k, dims);
  const inNames = [...Array.from({ length: du }, (_, i) => `u${i}`), ...Array.from({ length: dy }, (_, o) => `y${o}`)];
  const allNames = [...pNames, ...inNames];

  const rnd = mulberry32(seed);
  const env: Record<string, number> = {};
  for (const nm of pNames) env[nm] = (rnd() * 2 - 1) * 0.7;
  for (let i = 0; i < du; i++) env[`u${i}`] = 0.1 + rnd() * 0.5;
  for (let o = 0; o < dy; o++) env[`y${o}`] = (rnd() * 2 - 1) * 0.3;
  if (metric === "angular") {
    const n = Math.hypot(env.y0, env.y1);
    env.y0 /= n;
    env.y1 /= n;
  }

  for (const [root, what] of [
    [term.weighted, "weighted"],
    [term.surprise, "surprise"],
  ] as const) {
    const gr = grad(g, root, allNames);
    const rev = evalNodes(allNames.map((nm) => gr[nm]), env);
    const num = allNames.map((nm) => fdOf(() => evalNodes([root], env)[0], env, nm));
    let maxRel = 0;
    for (let i = 0; i < rev.length; i++) {
      maxRel = Math.max(maxRel, Math.abs(rev[i] - num[i]) / (Math.abs(num[i]) + 1e-4));
    }
    const cos = cosine(rev, num);
    ok(
      cos > 0.99999 && maxRel < 5e-3,
      `${label} k=${k} eps=${eps} ${what}: ${allNames.length} grads, cos=${cos.toFixed(7)}, maxRel=${maxRel.toExponential(2)}`
    );
  }
}

fdCheck("pair du=1 dy=2", 1, 2, 2, 0.05, 11);
fdCheck("pair du=1 dy=2", 1, 2, 3, 0.05, 12);
fdCheck("pair du=1 dy=2", 1, 2, 4, 0.05, 13);
fdCheck("tri  du=3 dy=6", 3, 6, 2, 0.05, 21);
fdCheck("tri  du=3 dy=6", 3, 6, 3, 0.05, 22);
fdCheck("tri  du=3 dy=6", 3, 6, 4, 0.05, 23);
// hard WTA still has a correct (winner-only) gradient
fdCheck("pair HARD wta", 1, 2, 2, 0.0, 31);
fdCheck("pair adjusted angular", 1, 2, 4, 0.05, 32, "angular");

// Winner dominance is structural: each loser must receive less weight than
// the winner. ε >= (K-1)/K reverses that meaning and is rejected.
{
  const g = new Graph();
  const dims: HeadDim[] = [
    { inSize: 1, outSize: 2, act: "linear" },
    { inSize: 2, outSize: 2, act: "linear" },
  ];
  let rejected = false;
  try {
    wtaTerm(g, [g.input("u")], [g.input("y0"), g.input("y1")], dims, 4, 0.75);
  } catch {
    rejected = true;
  }
  ok(rejected, "relaxEps = (K-1)/K is rejected (winner must dominate each loser)");
}

// Full adjusted target pipeline q -> q/||q|| -> angular payoff. The exact
// target-normalization Jacobian must annihilate a radial/scale perturbation.
{
  const g = new Graph();
  const qx = g.input("qx");
  const qy = g.input("qy");
  const qn = g.sqrt(g.add(g.mul(qx, qx), g.mul(qy, qy)));
  const y = [g.div(qx, qn), g.div(qy, qn)];
  const dims: HeadDim[] = [
    { inSize: 1, outSize: 4, act: "selu" },
    { inSize: 4, outSize: 2, act: "linear" },
  ];
  const term = wtaTerm(g, [g.input("u")], y, dims, 2, 0.05, "angular");
  const dq = grad(g, term.payoff, ["qx", "qy"]);
  const names = paramNames(2, dims);
  const rnd = mulberry32(31337);
  const env: Record<string, number> = { qx: 0.37, qy: -0.21, u: 1 };
  for (const n of names) env[n] = (rnd() * 2 - 1) * 0.4;
  const [gx, gy, base] = evalNodes([dq.qx, dq.qy, term.payoff], env);
  const radial = env.qx * gx + env.qy * gy;
  env.qx *= 7;
  env.qy *= 7;
  const scaled = evalNodes([term.payoff], env)[0];
  ok(Math.abs(radial) < 1e-10, `adjusted payoff radial derivative q·∇q = ${radial.toExponential(2)}`);
  ok(Math.abs(scaled - base) < 1e-12, "adjusted payoff is invariant under positive 7× target scaling");
}

// ε=0 freeze property: the loser head's parameter gradients are EXACTLY zero.
{
  const du = 1,
    dy = 2,
    k = 2;
  const dims: HeadDim[] = [
    { inSize: du, outSize: 6, act: "selu" },
    { inSize: 6, outSize: 6, act: "selu" },
    { inSize: 6, outSize: dy, act: "linear" },
  ];
  const g = new Graph();
  const uN = [g.input("u0")];
  const yN = [g.input("y0"), g.input("y1")];
  const term = wtaTerm(g, uN, yN, dims, k, 0.0);
  const rnd = mulberry32(99);
  const env: Record<string, number> = { u0: 0.4, y0: 0.1, y1: -0.2 };
  for (const nm of paramNames(k, dims)) env[nm] = (rnd() * 2 - 1) * 0.7;

  const rv = evalNodes(term.resid, env);
  const winner = rv[0] <= rv[1] ? 0 : 1;
  const loser = 1 - winner;
  const perHead = (j: number) => paramNames(1, dims).map((nm) => nm.replace(/^(aw|ab)_0_/, `$1_${j}_`));
  const gr = grad(g, term.weighted, [...perHead(0), ...perHead(1)]);
  const loserG = evalNodes(perHead(loser).map((nm) => gr[nm]), env);
  const winG = evalNodes(perHead(winner).map((nm) => gr[nm]), env);
  const grW = grad(g, term.resid[winner], perHead(winner));
  const winRef = evalNodes(perHead(winner).map((nm) => grW[nm]), env);
  const maxLoser = Math.max(...loserG.map(Math.abs));
  let maxDiff = 0;
  for (let i = 0; i < winG.length; i++) maxDiff = Math.max(maxDiff, Math.abs(winG[i] - winRef[i]));
  ok(maxLoser === 0, `eps=0 loser head gradient is EXACTLY zero (max |g| = ${maxLoser})`);
  ok(
    maxDiff < 1e-12,
    `eps=0 winner gradient == d resid_winner/dθ with weight 1 (max diff ${maxDiff.toExponential(2)})`
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   §2  cross-check vs the real tfjs Adversary on an identical fixture
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 2. cross-check vs tfjs Adversary (identical weights + batch) ---");

const SOFT2 = 1e-6 * 1e-6; // SOFT_EPS² — must match adversary.ts SOFT_EPS

function crossCheck(encoding: TupleEncoding, k: number, eps: number, B: number, seed: number) {
  const label = `${encoding.tag} k=${k}`;
  const cfg = {
    ...defaultAdversaryConfig(
      { tag: "wta", k, relaxEps: eps },
      encoding,
      "periodic"
    ),
    batchTuples: B,
    seed,
  };
  const adv = new Adversary(cfg);
  const { m, du, dy } = encodingDims(encoding);

  // fixture: seeded f32 u/y in plausible ranges (values are data — only the
  // widths matter to the loss math)
  const rnd = mulberry32(seed * 7 + 1);
  const uArr = new Float32Array(B * du);
  const yArr = new Float32Array(B * dy);
  for (let i = 0; i < uArr.length; i++) uArr[i] = 0.05 + 0.6 * rnd();
  if (encoding.tag === "tri") {
    // Real tri contexts are descending canonical side lengths. Keep this loss
    // fixture off the explicitly inactive near-tie set.
    for (let t = 0; t < B; t++) {
      const top = 0.55 + 0.05 * rnd();
      uArr[t * 3] = top;
      uArr[t * 3 + 1] = top - 0.12;
      uArr[t * 3 + 2] = top - 0.24;
    }
  }
  for (let i = 0; i < yArr.length; i++) yArr[i] = 0.4 * (rnd() * 2 - 1);
  const uT = tf.tensor2d(uArr, [B, du]);
  const yT = tf.tensor2d(yArr, [B, dy]);
  const s: TupleSample = { u: uT, y: yT, idx: new Int32Array(B * m), b: B, m };

  // --- pull the tfjs head weights into the IR leaf env -------------------
  const heads = (adv as unknown as { heads: tf.Sequential[] }).heads;
  const w0: Record<string, number> = {};
  heads.forEach((h, j) => {
    const ws = h.getWeights(); // [k0,b0,k1,b1,k2,b2], kernels [in,out] row-major
    for (let l = 0; l < 3; l++) {
      const kd = ws[2 * l].dataSync();
      const bd = ws[2 * l + 1].dataSync();
      const [inS, outS] = ws[2 * l].shape as [number, number];
      for (let i = 0; i < inS; i++)
        for (let o = 0; o < outS; o++) w0[awName(j)(l, i, o)] = kd[i * outS + o];
      for (let o = 0; o < outS; o++) w0[abName(j)(l, o)] = bd[o];
    }
  });

  // --- tfjs side ---------------------------------------------------------
  // Relaxed weights computed OUTSIDE the gradient closure from pre-update
  // residuals — mirrors Adversary.trainStep exactly (data on the tape).
  const resid0 = adv.residuals(s);
  const amT = tf.argMin(resid0, 1);
  const winners = amT.dataSync();
  const loserShare = eps / (k - 1);
  const wT = tf
    .oneHot(amT, k)
    .toFloat()
    .mul(1 - eps - loserShare)
    .add(loserShare) as tf.Tensor2D;

  const varList = heads.flatMap((h) =>
    h.trainableWeights.map((wv) => (wv as unknown as { val: tf.Variable }).val)
  );
  const { value: discLossT, grads: discGrads } = tf.variableGrads(() => {
    const per = heads.map((h) => {
      const pred = h.apply(uT) as tf.Tensor2D;
      return pred.sub(yT).square().sum(1).add(SOFT2).sqrt() as tf.Tensor1D;
    });
    return tf.stack(per, 1).mul(wT).sum(1).mean() as tf.Scalar;
  }, varList);
  const discLossTf = discLossT.dataSync()[0];

  // flatten tfjs disc grads into name → value
  const tfGrad: Record<string, number> = {};
  varList.forEach((v, vi) => {
    const j = Math.floor(vi / 6);
    const l = Math.floor((vi % 6) / 2);
    const isKernel = vi % 2 === 0;
    const gv = discGrads[v.name];
    const gd = gv.dataSync();
    if (isKernel) {
      const [inS, outS] = gv.shape as [number, number];
      for (let i = 0; i < inS; i++)
        for (let o = 0; o < outS; o++) tfGrad[awName(j)(l, i, o)] = gd[i * outS + o];
    } else {
      for (let o = 0; o < gd.length; o++) tfGrad[abName(j)(l, o)] = gd[o];
    }
  });

  const surpriseTf = adv.surprise(s).mean().dataSync()[0];

  // Generator maximizes the SAME relaxed-WTA payoff, with winner weights held
  // fixed exactly as on the discriminator tape.
  const genG = tf.grads((uu, yy) => {
    const per = heads.map((h) => {
      const pred = h.apply(uu as tf.Tensor2D) as tf.Tensor2D;
      return pred.sub(yy as tf.Tensor2D).square().sum(1).add(SOFT2).sqrt() as tf.Tensor1D;
    });
    return tf.stack(per, 1).mul(wT).sum(1).mean();
  })([uT, yT]);
  const genDuTf = genG[0].dataSync();
  const genDyTf = genG[1].dataSync();

  // --- IR side ------------------------------------------------------------
  const dims: HeadDim[] = [
    { inSize: du, outSize: cfg.hiddenUnits, act: "selu" },
    { inSize: cfg.hiddenUnits, outSize: cfg.featureDim, act: "selu" },
    { inSize: cfg.featureDim, outSize: dy, act: "linear" },
  ];
  const g = new Graph();
  const uN = Array.from({ length: du }, (_, i) => g.input(`u${i}`));
  const yN = Array.from({ length: dy }, (_, o) => g.input(`y${o}`));
  const term = wtaTerm(g, uN, yN, dims, k, eps);
  const pNames = paramNames(k, dims);
  const gW = grad(g, term.weighted, pNames);
  const inNames = [...Array.from({ length: du }, (_, i) => `u${i}`), ...Array.from({ length: dy }, (_, o) => `y${o}`)];
  const gGen = grad(g, term.payoff, inNames);

  const roots: Node[] = [
    term.weighted,
    term.surprise,
    ...term.resid,
    ...pNames.map((nm) => gW[nm]),
    ...inNames.map((nm) => gGen[nm]),
  ];

  let discLossIr = 0;
  let surpriseIr = 0;
  let winnerMismatch = 0;
  const irGrad = new Float64Array(pNames.length);
  const irGenDu = new Float64Array(B * du);
  const irGenDy = new Float64Array(B * dy);
  const env: Record<string, number> = { ...w0 };
  for (let t = 0; t < B; t++) {
    for (let i = 0; i < du; i++) env[`u${i}`] = uArr[t * du + i];
    for (let o = 0; o < dy; o++) env[`y${o}`] = yArr[t * dy + o];
    const vals = evalNodes(roots, env);
    discLossIr += vals[0] / B;
    surpriseIr += vals[1] / B;
    let am = 0;
    for (let j = 1; j < k; j++) if (vals[2 + j] < vals[2 + am]) am = j;
    if (am !== winners[t]) winnerMismatch++;
    const gBase = 2 + k;
    for (let i = 0; i < pNames.length; i++) irGrad[i] += vals[gBase + i] / B;
    for (let i = 0; i < du; i++) irGenDu[t * du + i] = vals[gBase + pNames.length + i] / B;
    for (let o = 0; o < dy; o++) irGenDy[t * dy + o] = vals[gBase + pNames.length + du + o] / B;
  }

  // --- compare ------------------------------------------------------------
  const relLoss = Math.abs(discLossIr - discLossTf) / Math.abs(discLossTf);
  const relSur = Math.abs(surpriseIr - surpriseTf) / Math.abs(surpriseTf);
  const tfVec = pNames.map((nm) => tfGrad[nm]);
  const cosDisc = cosine(irGrad as unknown as number[], tfVec);
  const cosDy = cosine(irGenDy as unknown as number[], genDyTf);
  const cosDu = cosine(irGenDu as unknown as number[], genDuTf);

  ok(winnerMismatch === 0, `${label}: per-tuple argmin winners agree (${winnerMismatch}/${B} mismatch)`);
  ok(relLoss < 1e-5, `${label}: discriminator loss IR=${discLossIr.toFixed(8)} vs tfjs=${discLossTf.toFixed(8)} (rel ${relLoss.toExponential(2)})`);
  ok(relSur < 1e-5, `${label}: mean surprise IR=${surpriseIr.toFixed(8)} vs tfjs=${surpriseTf.toFixed(8)} (rel ${relSur.toExponential(2)})`);
  ok(cosDisc > 0.99999, `${label}: discriminator grad over ${pNames.length} head weights, cos=${cosDisc.toFixed(7)}`);
  ok(cosDy > 0.99999, `${label}: generator grad d loss/d y (${B * dy} comps), cos=${cosDy.toFixed(7)}`);
  ok(cosDu > 0.99999, `${label}: generator grad d loss/d u (${B * du} comps), cos=${cosDu.toFixed(7)}`);

  resid0.dispose();
  amT.dispose();
  wT.dispose();
  discLossT.dispose();
  Object.values(discGrads).forEach((t) => t.dispose());
  genG.forEach((t) => t.dispose());
  uT.dispose();
  yT.dispose();
  disposeAdversary(adv);
}

crossCheck({ tag: "pair" }, 4, 0.05, 64, 1234);
crossCheck({ tag: "tri" }, 3, 0.05, 64, 4321);

/* ══════════════════════════════════════════════════════════════════════════
   §2b  explicit objectives: exact smooth angle, local scale, energy, post-v
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 2b. explicit objective math + reverse-mode gates ---");

// Exact S² chord is finite at zero and its reverse gradient matches FD without
// the old 1/|q| tangent quotient.
{
  const tau = 0.05;
  const g = new Graph();
  const px = g.input("px"), py = g.input("py");
  const yx = g.input("yx"), yy = g.input("yy");
  const chord = softSphereChordTerm(g, [px, py], [yx, yy], tau);
  const names = ["px", "py", "yx", "yy"];
  for (const env of [
    { px: 0, py: 0, yx: 0, yy: 0 },
    { px: 1e-5, py: -2e-5, yx: 3e-5, yy: 1e-5 },
    { px: 0.3, py: -0.2, yx: -0.1, yy: 0.4 },
  ]) {
    const gr = grad(g, chord, names);
    const rev = evalNodes(names.map((n) => gr[n]), env);
    const num = names.map((n) => fdOf(() => evalNodes([chord], env)[0], env, n));
    const gradScale = Math.max(...rev.map(Math.abs), ...num.map(Math.abs));
    ok(
      rev.every(Number.isFinite) &&
        (gradScale < 1e-10 || cosine(rev, num) > 0.999),
      `soft S² chord finite/FD at |y|=${Math.hypot(env.yx, env.yy).toExponential(1)}`
    );
    const gy = Math.hypot(rev[2], rev[3]);
    ok(gy <= 2 / tau + 1e-8, `soft S² target gradient bounded (${gy.toFixed(4)} <= ${2 / tau})`);
  }
}

// Pair and tri descriptors are exactly homogeneous: value unchanged and the
// full radial directional derivative is zero.
for (const [label, m] of [["pair", 2], ["tri", 3]] as const) {
  const g = new Graph();
  const sig = Array.from({ length: m }, (_, i) => [
    g.input(`s${i}x`),
    g.input(`s${i}y`),
  ] as [Node, Node]);
  const term = localRelativeScaleTerm(g, sig);
  const root = g.sum(term.target);
  // Pair has one nonnegative contrast; centered tri target sums identically
  // to zero, so probe a nontrivial weighted projection for the derivative.
  const projected =
    m === 2
      ? term.target[0]
      : g.sum(term.target.map((z, i) => g.mul(g.const(i + 1), z)));
  const names = sig.flatMap((_, i) => [`s${i}x`, `s${i}y`]);
  const baseVals = [0.21, -0.13, -0.32, 0.27, 0.11, 0.42];
  const evalScale = (c: number) => {
    const env: Record<string, number> = {};
    names.forEach((n, i) => (env[n] = baseVals[i] * c));
    return evalNodes([projected], env)[0];
  };
  const a = evalScale(0.1), b = evalScale(1), c = evalScale(10);
  ok(
    Math.max(Math.abs(a - b), Math.abs(c - b)) < 1e-11,
    `${label} local relative-scale target invariant under 0.1×/10×`
  );
  const env: Record<string, number> = {};
  names.forEach((n, i) => (env[n] = baseVals[i]));
  const gr = grad(g, projected, names);
  const gv = evalNodes(names.map((n) => gr[n]), env);
  const radial = names.reduce((s, n, i) => s + env[n] * gv[i], 0);
  ok(Math.abs(radial) < 1e-10, `${label} radial identity Σs·∇=${radial.toExponential(2)}`);
  // Keep the exact centered-sum identity itself under test, rather than
  // relying only on the weighted projection.
  if (m === 3) ok(Math.abs(evalNodes([root], env)[0]) < 1e-12, "tri centered logs sum to zero");
}

// Reverse-vs-FD for all four requested objective families, including the
// general-sum scale-hold generator scalar.
for (const loss of [
  { tag: "raw-vector" } as const,
  { tag: "soft-angle", tau: 0.05 } as const,
  {
    tag: "angle-relative-scale", tau: 0.05, scaleWeight: 0.4,
    energyWeight: 0.2, energyTarget: 0.35,
  } as const,
  {
    tag: "angle-scale-hold", tau: 0.05, scaleWeight: 0.4,
    energyWeight: 0.2, energyTarget: 0.35,
  } as const,
]) {
  const g = new Graph();
  const scaleN = loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold"
    ? [g.input("ys")]
    : [];
  const out = 2 + scaleN.length;
  const dims: HeadDim[] = [
    { inSize: 1, outSize: 4, act: "selu" },
    { inSize: 4, outSize: out, act: "linear" },
  ];
  const term = wtaObjectiveTerm(
    g,
    [g.input("u")],
    [g.input("yx"), g.input("yy")],
    scaleN,
    dims,
    2,
    0.05,
    loss
  );
  const names = [
    ...paramNames(2, dims),
    "u", "yx", "yy",
    ...(scaleN.length ? ["ys"] : []),
  ];
  const rnd = mulberry32(900 + out + loss.tag.length);
  const env: Record<string, number> = {};
  for (const n of names) env[n] = (rnd() * 2 - 1) * 0.4;
  for (const [root, side] of [
    [term.weighted, "D"],
    [term.generatorGame, "G"],
  ] as const) {
    const gr = grad(g, root, names);
    const rev = evalNodes(names.map((n) => gr[n]), env);
    const num = names.map((n) => fdOf(() => evalNodes([root], env)[0], env, n));
    ok(
      cosine(rev, num) > 0.99999 && rev.every(Number.isFinite),
      `${loss.tag} ${side} exact reverse vs FD (cos ${cosine(rev, num).toFixed(7)})`
    );
  }
}

// Batch energy anchor has the correct restoring sign.
{
  const g = new Graph();
  const e = g.input("e");
  const anchor = energyAnchorTerm(g, [e], [g.const(1)], 0.2, 0.5);
  const de = grad(g, anchor, ["e"]).e;
  const below = evalNodes([de], { e: 0.04 })[0];
  const at = evalNodes([de], { e: 0.25 })[0];
  const above = evalNodes([de], { e: 1.0 })[0];
  ok(below < 0 && Math.abs(at) < 1e-12 && above > 0,
    `energy anchor restoring derivative below/at/above = ${below.toFixed(3)}/${at.toExponential(1)}/${above.toFixed(3)}`);
}

// Post-v includes incoming velocity in the state, matches the physical update,
// and its force derivative is exactly zero once a component saturates.
{
  const g = new Graph();
  const vx = g.input("vx"), vy = g.input("vy");
  const fx = g.input("fx"), fy = g.input("fy");
  const z = postVelocityTargetTerm(g, [vx, vy], [fx, fy], 3, 0.9, 2);
  const d = grad(g, z[0], ["fx"]).fx;
  const unsat = { vx: 0.2, vy: 0, fx: 0.1, fy: 0 };
  const sat = { vx: 1.9, vy: 0, fx: 1, fy: 0 };
  const [zu, gu] = evalNodes([z[0], d], unsat);
  const [zs, gs] = evalNodes([z[0], d], sat);
  ok(Math.abs(zu - 0.225) < 1e-12 && Math.abs(gu - 1.35) < 1e-12,
    `post-v unsaturated value/Jacobian exact (${zu.toFixed(4)}, ${gu.toFixed(4)})`);
  ok(Math.abs(zs - 1) < 1e-12 && Math.abs(gs) < 1e-12,
    `post-v saturated value/Jacobian exact (${zs.toFixed(1)}, ${gs.toExponential(1)})`);
}

/* ══════════════════════════════════════════════════════════════════════════
   §3  emitted WGSL on real Metal vs the JS oracle
   ══════════════════════════════════════════════════════════════════════════ */
if (process.env.SKIP_GPU === "1") {
  console.log("\n--- 3. emitted-WGSL GPU gate SKIPPED (SKIP_GPU=1) ---");
} else {
  console.log("\n--- 3. emitWGSL forward+backward on real Metal vs JS oracle ---");
  const { setupGlobals } = await import("bun-webgpu");
  setupGlobals();
  (globalThis as any).GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128,
  };
  const USAGE = (globalThis as any).GPUBufferUsage;

  const adapter = await (navigator as any).gpu.requestAdapter();
  if (!adapter) {
    console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
    process.exit(1);
  }
  const device: any = await adapter.requestDevice();

  // Pair-shaped explicit SOFT-ANGLE config, K=3, small heads. This deliberately
  // exercises the new S² embedding in emitted WGSL rather than re-testing only
  // the legacy raw Euclidean objective.
  const du = 1, dy = 2, K = 3, EPSR = 0.05, B = 128;
  const dims: HeadDim[] = [
    { inSize: du, outSize: 8, act: "selu" },
    { inSize: 8, outSize: 8, act: "selu" },
    { inSize: 8, outSize: dy, act: "linear" },
  ];
  const g = new Graph();
  const uN = Array.from({ length: du }, (_, i) => g.input(`u${i}`));
  const yN = Array.from({ length: dy }, (_, o) => g.input(`y${o}`));
  const term = wtaObjectiveTerm(
    g, uN, yN, [], dims, K, EPSR, { tag: "soft-angle", tau: 0.05 }
  );
  const pNames = paramNames(K, dims);
  const inNames = [...Array.from({ length: du }, (_, i) => `u${i}`), ...Array.from({ length: dy }, (_, o) => `y${o}`)];
  const gW = grad(g, term.weighted, pNames);
  const gGen = grad(g, term.generatorGame, inNames);

  const P = pNames.length;
  const TW = du + dy;
  const R = 2 + P + inNames.length; // per-thread outputs

  const roots = [
    { node: term.surprise, out: "outBuf[obase + 0u]" },
    { node: term.weighted, out: "outBuf[obase + 1u]" },
    ...pNames.map((nm, i) => ({ node: gW[nm], out: `outBuf[obase + ${2 + i}u]` })),
    ...inNames.map((nm, i) => ({
      node: gGen[nm],
      out: `outBuf[obase + ${2 + P + i}u]`,
    })),
  ];

  // measured sizes (reported — the unroll cost model is emitted lines ≈ nodes)
  const seen = new Set<number>();
  const countWalk = (n: Node) => {
    if (seen.has(n.id)) return;
    seen.add(n.id);
    for (const c of n.inputs) countWalk(c);
  };
  for (const r of roots) countWalk(r.node);
  const nodeCount = seen.size;

  const body = emitWGSL(roots, { indent: "  " });
  const prelude = [
    `  let base = s * ${TW}u;`,
    ...Array.from({ length: du }, (_, i) => `  let u${i} = tup[base + ${i}u];`),
    ...Array.from({ length: dy }, (_, o) => `  let y${o} = tup[base + ${du + o}u];`),
    ...pNames.map((nm, i) => `  let ${nm} = W[${i}u];`),
    `  let obase = s * ${R}u;`,
  ].join("\n");
  const shader = `
@group(0) @binding(0) var<storage, read> tup : array<f32>;
@group(0) @binding(1) var<storage, read> W : array<f32>;
@group(0) @binding(2) var<storage, read_write> outBuf : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let s = gid.x;
  if (s >= ${B}u) { return; }
${prelude}
${body}
}
`;
  const lineCount = shader.split("\n").length;
  console.log(
    `      emitted: ${nodeCount} IR nodes -> ${lineCount} WGSL lines (${P} weight grads + ${inNames.length} input grads + fwd, per thread)`
  );

  // seeded f32 fixture
  const rnd = mulberry32(777);
  const tupArr = new Float32Array(B * TW);
  for (let t = 0; t < B; t++) {
    for (let i = 0; i < du; i++) tupArr[t * TW + i] = 0.05 + 0.6 * rnd();
    for (let o = 0; o < dy; o++) tupArr[t * TW + du + o] = 0.4 * (rnd() * 2 - 1);
  }
  const wArr = new Float32Array(P);
  for (let i = 0; i < P; i++) wArr[i] = (rnd() * 2 - 1) * 0.7;

  // JS oracle (f64 eval of the SAME graph on the SAME f32-quantized inputs)
  const oracle = new Float64Array(B * R);
  {
    const env: Record<string, number> = {};
    pNames.forEach((nm, i) => (env[nm] = wArr[i]));
    const rootNodes = roots.map((r) => r.node);
    for (let t = 0; t < B; t++) {
      for (let i = 0; i < du; i++) env[`u${i}`] = tupArr[t * TW + i];
      for (let o = 0; o < dy; o++) env[`y${o}`] = tupArr[t * TW + du + o];
      const vals = evalNodes(rootNodes, env);
      for (let r = 0; r < R; r++) oracle[t * R + r] = vals[r];
    }
  }

  // GPU dispatch
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: shader });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const verr = await device.popErrorScope();
  ok(verr === null, `emitted shader compiles on Metal (${verr ? String(verr.message) : "no validation error"})`);

  const mkBuf = (arr: Float32Array, usage: number) => {
    const buf = device.createBuffer({ size: Math.max(arr.byteLength, 16), usage });
    device.queue.writeBuffer(buf, 0, arr);
    return buf;
  };
  const tupBuf = mkBuf(tupArr, USAGE.STORAGE | USAGE.COPY_DST);
  const wBuf = mkBuf(wArr, USAGE.STORAGE | USAGE.COPY_DST);
  const outBytes = B * R * 4;
  const outBuf = device.createBuffer({ size: outBytes, usage: USAGE.STORAGE | USAGE.COPY_SRC });
  const staging = device.createBuffer({ size: outBytes, usage: USAGE.MAP_READ | USAGE.COPY_DST });

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: tupBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: outBuf } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(B / 64));
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, staging, 0, outBytes);
  const t0 = performance.now();
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1);
  const gpuMs = performance.now() - t0;
  const gpu = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();

  // compare per category, rel to the category's value scale (f32 vs f64)
  const cats: Array<[string, number, number]> = [
    ["forward (surprise+weighted)", 0, 2],
    ["d weighted / d headWeights", 2, 2 + P],
    ["d generatorGame / d (u,y)", 2 + P, R],
  ];
  for (const [cname, lo, hi] of cats) {
    let scale = 0;
    for (let t = 0; t < B; t++)
      for (let r = lo; r < hi; r++) scale = Math.max(scale, Math.abs(oracle[t * R + r]));
    let worstRel = 0;
    let worstFloored = 0;
    for (let t = 0; t < B; t++) {
      for (let r = lo; r < hi; r++) {
        const j = oracle[t * R + r];
        const d = Math.abs(gpu[t * R + r] - j);
        worstRel = Math.max(worstRel, d / scale);
        worstFloored = Math.max(worstFloored, d / (Math.abs(j) + 1e-6));
      }
    }
    ok(
      worstRel < 1e-5,
      `GPU ≡ JS oracle: ${cname} (${(hi - lo) * B} values, worst rel ${worstRel.toExponential(2)} of scale ${scale.toExponential(2)}, elementwise ${worstFloored.toExponential(2)})`
    );
  }
  console.log(`      submit→mapped ${gpuMs.toFixed(1)} ms for B=${B} threads (incl. readback)`);

  tupBuf.destroy();
  wBuf.destroy();
  outBuf.destroy();
  staging.destroy();
}

console.log(failures === 0 ? "\nALL WTA ORACLE CHECKS PASS" : `\n${failures} WTA CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
