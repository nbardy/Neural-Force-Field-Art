/**
 * Spiral cover oracle — locks the Chamfer ↑ contract without WebGPU:
 *   all mass at one spiral locus  ⇒  high loss
 *   mass on arc-length samples    ⇒  low loss
 * Also checks AD IR coverTerm + reverse-mode pulls the winning particle.
 *
 *   bun tools/cover_oracle_test.ts
 */
import { Graph } from "../src/render/webgpu/ad/ir";
import { coverTerm, type V2 } from "../src/render/webgpu/ad/losses";
import { evalNode, grad } from "../src/render/webgpu/ad/autodiff";

const TURNS = 3;
const THETA_MAX = TURNS * 2 * Math.PI;
const ARC_SKIP = 0.08;
const M = 32;
const W = 800;
const H = 600;
const CX = W / 2;
const CY = H / 2;
const MAX_R = Math.min(W, H) * 0.38;

function arcLen(theta: number): number {
  const s = Math.sqrt(1 + theta * theta);
  return 0.5 * (theta * s + Math.asinh(theta));
}

function thetaFromArc(target: number, lo0: number, hi0: number): number {
  let lo = lo0;
  let hi = hi0;
  for (let i = 0; i < 48; i++) {
    const mid = 0.5 * (lo + hi);
    if (arcLen(mid) - arcLen(lo0) < target) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

function spiralSamples(): Array<[number, number]> {
  const total = arcLen(THETA_MAX);
  const skip = ARC_SKIP * total;
  const usable = total - skip;
  const thetaMin = thetaFromArc(skip, 0, THETA_MAX);
  const out: Array<[number, number]> = [];
  for (let i = 0; i < M; i++) {
    const theta = thetaFromArc(usable * ((i + 0.5) / M), thetaMin, THETA_MAX);
    const r = MAX_R * (theta / THETA_MAX);
    out.push([CX + r * Math.cos(theta), CY + r * Math.sin(theta)]);
  }
  return out;
}

function coverLossCPU(
  positions: Array<[number, number]>,
  samples: Array<[number, number]>
): number {
  const scale = Math.max(MAX_R * MAX_R, 1);
  let acc = 0;
  for (const s of samples) {
    let best = Infinity;
    for (const p of positions) {
      const dx = p[0] - s[0];
      const dy = p[1] - s[1];
      best = Math.min(best, dx * dx + dy * dy);
    }
    acc += best;
  }
  return acc / (samples.length * scale);
}

const samples = spiralSamples();
const onSpiral: Array<[number, number]> = samples.map(([x, y]) => [x, y]);
const clustered: Array<[number, number]> = Array.from({ length: M }, () => [
  samples[0][0],
  samples[0][1],
]);

const Lcluster = coverLossCPU(clustered, samples);
const Lfill = coverLossCPU(onSpiral, samples);
if (!(Lcluster > 5 * Lfill)) {
  throw new Error(
    `cover oracle: cluster ${Lcluster} should be ≫ fill ${Lfill}`
  );
}
console.log(
  `ok cover CPU  cluster=${Lcluster.toExponential(3)}  fill=${Lfill.toExponential(3)}`
);

// IR: 2 particles, 2 samples — winner of sample0 is p0; nudging p0 toward s0
// should decrease loss (negative grad · (s0-p0) direction... grad is 2(p-s)/…).
const g = new Graph();
const p0: V2 = [g.input("p0x"), g.input("p0y")];
const p1: V2 = [g.input("p1x"), g.input("p1y")];
const s0: V2 = [g.const(samples[0][0]), g.const(samples[0][1])];
const s1: V2 = [g.const(samples[Math.floor(M / 2)][0]), g.const(samples[Math.floor(M / 2)][1])];
const L = coverTerm(g, [p0, p1], [s0, s1]);
const env = {
  p0x: samples[0][0] + 40,
  p0y: samples[0][1],
  p1x: samples[Math.floor(M / 2)][0],
  p1y: samples[Math.floor(M / 2)][1],
};
const d = grad(g, L, ["p0x", "p0y"]);
const gx = evalNode(d.p0x, env);
const gy = evalNode(d.p0y, env);
const toward = (samples[0][0] - env.p0x) * gx + (samples[0][1] - env.p0y) * gy;
if (!(toward < 0)) {
  throw new Error(
    `cover IR grad should pull p0 toward s0 (toward·grad=${toward}, g=${gx},${gy})`
  );
}
console.log(`ok cover IR   grad pulls winner toward sample (dot=${toward.toExponential(3)})`);
console.log("cover_oracle_test: PASS");
