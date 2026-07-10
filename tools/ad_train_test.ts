/**
 * M4 verification: the source-to-source AD engine reproduces the ENTIRE shipped
 * trainer gradient. Builds the full per-sample loss graph L_ad (K-step rollout +
 * chaos/div/spiral + isotropy fold) in the IR, runs reverse-mode `grad`, sums
 * over the fixture's batch, and compares to tools/fixtures/grad_ref.json — the
 * tfjs-autograd oracle. Pure JS, no GPU. See docs/PLAN_AD_IR_BACKWARD_CODEGEN.md.
 *
 *   bun tools/ad_train_test.ts
 *
 * A cos≈1.0 here means one reverse pass over the IR = the ~400 hand-written
 * lines of δ-chains + BPTT recurrence + loss seeds in train_wgsl.ts, for FREE.
 */
import { readFileSync } from "node:fs";
import { Graph, type Node } from "../src/render/webgpu/ad/ir";
import { evalNodes, grad } from "../src/render/webgpu/ad/autodiff";
import { isotropyTerm } from "../src/render/webgpu/ad/losses";
import { buildSample, wName, bName, type RolloutCfg } from "../src/render/webgpu/ad/rollout";
import type { HeadDim } from "../src/render/webgpu/ad/head";

let failures = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
  if (!cond) failures++;
};

interface FixVar { name: string; shape: number[]; values: number[] }
const FIX = process.env.FIX ?? "./fixtures/grad_ref.json";
const fix = JSON.parse(
  readFileSync(new URL(FIX, import.meta.url), "utf8")
) as { meta: any; variables: FixVar[]; batch: number[]; loss: number; grads: FixVar[] };

const m = fix.meta;
const model: string = m.model ?? "standard";
if (model === "hashgrid") {
  // the hashgrid gather is data-dependent — not expressible in the static
  // scalar IR (see rollout.ts). Its fused backward is verified on Metal vs
  // the tfjs fixture by tools/train_types_test.ts instead.
  console.log("hashgrid fixture: skipped (IR oracle is siren/fourier/standard only)");
  process.exit(0);
}
const lc = m.loss_constants;
const cfg: RolloutCfg = {
  K: m.K, alpha: m.alpha, hh: m.HH,
  forceMag: m.forceMagnitude, friction: m.friction, maxVel: m.maxVelocity,
  W: m.W, H: m.H, spiralTurns: 3,
  wChaos: lc.W_CHAOS, wDiv: lc.W_DIV, wSpiral: lc.W_SPIRAL, wIso: lc.W_ISO,
  N: m.N,
  encoding: model === "fourier"
    ? { kind: "fourier", octaves: m.fourierOctaves ?? 4 }
    : { kind: "raw" },
};
console.log(`fixture: model=${model} K=${m.K} N=${m.N} classes=${m.classes} loss=${fix.loss}\n`);
if (m.classes !== 0) { console.log("(classless fixture only for now)"); }

// --- head dims from variable shapes (vars 0..5 = head0/g, 6..11 = head1/r) ---
const hiddenAct = model === "siren" ? ("sin" as const) : ("selu" as const);
function headDims(vars: FixVar[]): HeadDim[] {
  // kernels are the even-indexed vars; hidden act by model type, output tanh
  const kernels = [vars[0], vars[2], vars[4]];
  return kernels.map((k, l) => ({ inSize: k.shape[0], outSize: k.shape[1], act: l === 2 ? "tanh" : hiddenAct }));
}
const dimsG = headDims(fix.variables.slice(0, 6));
const dimsR = headDims(fix.variables.slice(6, 12));

// --- env: unpack fixture weights into leaf names ----------------------------
const env: Record<string, number> = {};
function loadHead(vars: FixVar[], h: number, dims: HeadDim[]) {
  for (let l = 0; l < 3; l++) {
    const ker = vars[l * 2], bia = vars[l * 2 + 1];
    const [inS, outS] = ker.shape;
    for (let i = 0; i < inS; i++)
      for (let j = 0; j < outS; j++)
        env[wName(h)(l, i, j)] = ker.values[i * outS + j];
    for (let j = 0; j < outS; j++) env[bName(h)(l, j)] = bia.values[j];
  }
  void dims;
}
loadHead(fix.variables.slice(0, 6), 0, dimsG);
loadHead(fix.variables.slice(6, 12), 1, dimsR);

// --- build the per-sample loss graph ONCE -----------------------------------
const g = new Graph();
const sample = buildSample(g, cfg, dimsG, dimsR);
const NK = m.N * m.K;

// --- PASS 1: batch covariance → dC (via AD on isotropyTerm) -----------------
const FsFlat: Node[] = sample.Fs.flatMap(([x, y]) => [x, y]);
let sxx = 0, syy = 0, sxy = 0;
for (let s = 0; s < m.N; s++) {
  env.p0x = fix.batch[2 * s]; env.p0y = fix.batch[2 * s + 1];
  const fsv = evalNodes(FsFlat, env);
  for (let k = 0; k < m.K; k++) {
    const fx = fsv[2 * k], fy = fsv[2 * k + 1];
    sxx += fx * fx; syy += fy * fy; sxy += fx * fy;
  }
}
const C00 = sxx / NK, C11 = syy / NK, C01 = sxy / NK;
// dC = ∂Liso/∂C via the IR itself (proves finalize's closed forms too)
const gi = new Graph();
const iso = isotropyTerm(gi, gi.input("C00"), gi.input("C11"), gi.input("C01"));
const dcg = grad(gi, iso, ["C00", "C11", "C01"]);
const cEnv = { C00, C11, C01 };
const [dC00, dC11, dC01] = evalNodes([dcg.C00, dcg.C11, dcg.C01], cEnv);
// cross-check against the hand-written closed forms in train_wgsl.ts finalize
{
  const S = C00 + C11 + 1e-6, D = C00 - C11, Liso = (D * D + 4 * C01 * C01) / (S * S);
  const cf00 = 2 * D / (S * S) - 2 * Liso / S;
  const cf11 = -2 * D / (S * S) - 2 * Liso / S;
  const cf01 = 8 * C01 / (S * S);
  const closeC = Math.max(Math.abs(dC00 - cf00), Math.abs(dC11 - cf11), Math.abs(dC01 - cf01));
  ok(closeC < 1e-9, `dC via IR ≡ finalize closed forms (max Δ ${closeC.toExponential(2)})`);
}
env.dC00 = dC00; env.dC11 = dC11; env.dC01 = dC01;

// --- PASS 2: reverse-mode grad summed over the batch ------------------------
const names: string[] = [];
const order: { name: string; fixIdx: number; local: number }[] = [];
// build the flat gradient vector in fixture (trainableWeights) order
fix.grads.forEach((gv, fixIdx) => {
  const h = fixIdx < 6 ? 0 : 1;
  const li = fixIdx % 6;               // 0=k0,1=b0,2=k1,3=b1,4=k2,5=b2
  const l = li >> 1;
  const isKernel = (li & 1) === 0;
  if (isKernel) {
    const [inS, outS] = gv.shape;
    for (let i = 0; i < inS; i++) for (let j = 0; j < outS; j++) {
      names.push(wName(h)(l, i, j)); order.push({ name: wName(h)(l, i, j), fixIdx, local: i * outS + j });
    }
  } else {
    const [outS] = gv.shape;
    for (let j = 0; j < outS; j++) {
      names.push(bName(h)(l, j)); order.push({ name: bName(h)(l, j), fixIdx, local: j });
    }
  }
});
const gradGraphs = grad(g, sample.L, names);
const gradRoots = names.map((nm) => gradGraphs[nm]);
const accum = new Float64Array(names.length);
let lossChaos = 0, lossDiv = 0, lossSpiral = 0;
const lossRoots = [sample.chaos, sample.div, sample.spiral];
for (let s = 0; s < m.N; s++) {
  env.p0x = fix.batch[2 * s]; env.p0y = fix.batch[2 * s + 1];
  const vals = evalNodes([...gradRoots, ...lossRoots], env);
  for (let i = 0; i < names.length; i++) accum[i] += vals[i];
  lossChaos += vals[names.length]; lossDiv += vals[names.length + 1]; lossSpiral += vals[names.length + 2];
}

// --- compare gradient to the fixture (cosine + max rel error) ---------------
const revVec: number[] = [];
const refVec: number[] = [];
order.forEach((o, i) => {
  revVec.push(accum[i]);
  refVec.push(fix.grads[o.fixIdx].values[o.local]);
});
let dot = 0, nr = 0, nn = 0, maxRel = 0;
for (let i = 0; i < revVec.length; i++) {
  dot += revVec[i] * refVec[i]; nr += revVec[i] * revVec[i]; nn += refVec[i] * refVec[i];
  maxRel = Math.max(maxRel, Math.abs(revVec[i] - refVec[i]) / (Math.abs(refVec[i]) + 1e-4));
}
const cos = dot / (Math.sqrt(nr) * Math.sqrt(nn) + 1e-30);
ok(cos > 0.99999, `∂L/∂w vs tfjs fixture: ${revVec.length} weights, cos=${cos.toFixed(7)}, maxRel=${maxRel.toExponential(2)}`);

// --- reconstruct the loss and compare to the fixture ------------------------
const S = C00 + C11 + 1e-6, D = C00 - C11, Liso = (D * D + 4 * C01 * C01) / (S * S);
const lossRecon =
  cfg.wChaos * (lossChaos / m.N) +
  cfg.wDiv * (lossDiv / m.N) +
  cfg.wSpiral * (lossSpiral / m.N) +
  cfg.wIso * Liso;
const lossRel = Math.abs(lossRecon - fix.loss) / (Math.abs(fix.loss) + 1e-9);
ok(lossRel < 1e-4, `loss reconstruction: ${lossRecon.toFixed(6)} vs fixture ${fix.loss.toFixed(6)} (rel ${lossRel.toExponential(2)})`);

console.log(failures === 0 ? "\nM4 AD-TRAIN CHECKS PASS" : `\n${failures} M4 CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
