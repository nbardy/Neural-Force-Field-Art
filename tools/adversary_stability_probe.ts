/**
 * Focused fused co-training stability probe.
 *
 * This reproduces the production standalone adversary loop with a shared field
 * buffer, one adversary update followed by one field Adam update, and reports
 * the first buffer that becomes non-finite. It is the long-running regression
 * gate for the ZERO-loss codegen, adjusted-direction floor, and G/D LR policy.
 *
 *   TAG=pair-rotation-scale-adjusted STEPS=3000 bun tools/adversary_stability_probe.ts
 *   TAG=pair-rotation-scale-raw FIELD_LR=0.008 STEPS=3000 bun tools/adversary_stability_probe.ts
 */
import { setupGlobals } from "bun-webgpu";
import {
  layoutField,
  type LayerDims,
} from "../src/render/webgpu/advect_wgsl";
import {
  AdversaryTrainer,
  packAdversaryInit,
} from "../src/render/webgpu/adversary_train";
import {
  advScratchLayout,
  type TupleTag,
} from "../src/render/webgpu/adversary_wgsl";
import { FusedTrainer } from "../src/render/webgpu/train";
import type { FieldLossSpec } from "../src/render/webgpu/train_wgsl";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device: GPUDevice = await adapter.requestDevice();

const TAG = (process.env.TAG ?? "pair-rotation-scale-adjusted") as TupleTag;
const STEPS = Number(process.env.STEPS ?? 3000);
const FIELD_LR = Number(process.env.FIELD_LR ?? 0.001);
const DISC_LR = Number(process.env.DISC_LR ?? 0.003);
const WEIGHT = Number(process.env.WEIGHT ?? 0.015);
const B = Number(process.env.B ?? 512);
const K = Number(process.env.K ?? 4);
const N = Number(process.env.N ?? 20000);
const REPORT = Number(process.env.REPORT ?? 25);
const PHYS = {
  width: 800,
  height: 600,
  forceMagnitude: 3.4,
  friction: 0.988,
  maxVelocity: 24,
};
const ZERO: FieldLossSpec = {
  W_CHAOS: 0,
  W_ISO: 0,
  W_DIV: 0,
  W_SPIRAL: 0,
  W_COVER: 0, W_CENTER: 0,
  HH: 1e-2,
  SPIRAL_TURNS: 3,
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
function gaussian(rnd: () => number): number {
  const u1 = Math.max(rnd(), 1e-12);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * rnd());
}
function dims(): LayerDims[] {
  return [
    { inSize: 2, outSize: 32, activation: "selu" },
    { inSize: 32, outSize: 32, activation: "selu" },
    { inSize: 32, outSize: 2, activation: "tanh" },
  ];
}
const layout = layoutField("helmholtz", [dims(), dims()]);
const fieldInit = new Float32Array(layout.totalFloats);
{
  const rnd = mulberry32(101);
  const heads = layout.spec.heads;
  for (const seg of layout.segments) {
    if (seg.role !== "kernel") continue;
    const L = heads[seg.head].layers[seg.layer];
    const std = Math.sqrt(2 / (L.inSize + L.outSize));
    for (let i = 0; i < seg.floatLength; i++) {
      fieldInit[seg.floatOffset + i] = gaussian(rnd) * std;
    }
  }
}
const fieldBuf = device.createBuffer({
  size: layout.totalFloats * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(fieldBuf, 0, fieldInit);

const posBuf = device.createBuffer({
  size: N * 8,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const velBuf = device.createBuffer({
  size: N * 8,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
{
  const rnd = mulberry32(102);
  const p = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    p[2 * i] = rnd() * PHYS.width;
    p[2 * i + 1] = rnd() * PHYS.height;
  }
  device.queue.writeBuffer(posBuf, 0, p);
}

const adv = new AdversaryTrainer(device, layout, {
  observerGeometry: "periodic",
  tag: TAG,
  k: K,
  relaxEps: 0.05,
  hiddenUnits: 32,
  featureDim: 16,
  batchCap: B,
  fieldWeightsBuffer: fieldBuf,
  particleCount: N,
  seed: 103,
});
adv.uploadAdvWeights(packAdversaryInit(adv.advL, 103));
adv.setParticleBuffers(posBuf, velBuf, N);
const field = new FusedTrainer(device, layout, {
  batchCap: B,
  weightsBuffer: fieldBuf,
  extGradBuffer: adv.extGradsBuf,
  loss: ZERO,
});

type Summary = {
  max: number;
  rms: number;
  firstBad: number;
};
function summarize(a: ArrayLike<number>): Summary {
  let max = 0;
  let ss = 0;
  let firstBad = -1;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!Number.isFinite(v) && firstBad < 0) firstBad = i;
    if (Number.isFinite(v)) {
      max = Math.max(max, Math.abs(v));
      ss += v * v;
    }
  }
  return { max, rms: Math.sqrt(ss / Math.max(1, a.length)), firstBad };
}
function fmt(s: Summary): string {
  return `max=${s.max.toExponential(3)} rms=${s.rms.toExponential(3)} bad=${s.firstBad}`;
}
function quantile(xs: number[], q: number): number {
  xs.sort((a, b) => a - b);
  return xs[Math.min(xs.length - 1, Math.floor(q * xs.length))] ?? NaN;
}
function pairGeometry(scratch: Float32Array): string {
  if (!TAG.startsWith("pair")) return "";
  const sl = advScratchLayout(layout, adv.advL, TAG);
  const heads = layout.spec.heads;
  const q: number[] = [];
  const pn: number[] = [];
  const lastAdv = sl.advAOff[adv.advL.heads[0].layers.length - 1];
  for (let s = 0; s < B; s++) {
    const sb = s * sl.stride;
    const sig: [number, number][] = [];
    for (let site = 0; site < 2; site++) {
      const vals: [number, number][] = [];
      for (let h = 0; h < 2; h++) {
        const headOff = h === 0 ? 0 : sl.fieldSl.headBlk[0];
        const last = heads[h].layers.length - 1;
        const out =
          sb +
          sl.fieldSiteOff +
          site * sl.fieldSiteBlk +
          headOff +
          sl.fieldSl.aOff[h][last];
        vals.push([scratch[out], scratch[out + 1]]);
      }
      sig.push([
        0.45 * vals[0][0] + 0.55 * vals[1][0],
        0.45 * vals[0][1] + 0.55 * vals[1][1],
      ]);
    }
    q.push(Math.hypot(sig[1][0] - sig[0][0], sig[1][1] - sig[0][1]));
    for (let j = 0; j < K; j++) {
      const ab = sb + sl.advOff + j * sl.advBlk + lastAdv;
      pn.push(Math.hypot(scratch[ab], scratch[ab + 1]));
    }
  }
  return (
    ` q[min/p1/p50]=${Math.min(...q).toExponential(2)}/` +
    `${quantile(q, 0.01).toExponential(2)}/${quantile(q, 0.5).toExponential(2)}` +
    ` p[min/p1/p50]=${Math.min(...pn).toExponential(2)}/` +
    `${quantile(pn, 0.01).toExponential(2)}/${quantile(pn, 0.5).toExponential(2)}`
  );
}

console.log(
  `stability tag=${TAG} steps=${STEPS} B=${B} K=${K} fieldLR=${FIELD_LR} ` +
    `discLR=${DISC_LR} weight=${WEIGHT}`
);
let failed = false;
for (let step = 0; step < STEPS; step++) {
  const seed = adv.genSeed(WEIGHT, 1, B);
  if (!Number.isFinite(seed)) {
    console.log(`FIRST NONFINITE step=${step} stage=host.genSeed value=${seed}`);
    failed = true;
    break;
  }
  adv.step(PHYS, {
    b: B,
    alpha: 0.55,
    lr: DISC_LR,
    seed: step,
    source: "particles",
    genSeed: seed,
    applyDisc: true,
  });
  const stats = await adv.readStats();
  const ag = await adv.readAdvGrads();
  const aw = await adv.readAdvWeights();
  const eg = await adv.readExtGrads();
  const before = await field.readWeights();
  const stageA = [
    ["stats", summarize([stats.payoffUngated, stats.surprise, stats.batchRms])],
    ["advGrad", summarize(ag)],
    ["advWeight", summarize(aw)],
    ["extGrad", summarize(eg)],
    ["fieldWeight.pre", summarize(before)],
  ] as const;
  const badA = stageA.find(([, s]) => s.firstBad >= 0);
  if (badA) {
    console.log(
      `FIRST NONFINITE step=${step} stage=${badA[0]} ${fmt(badA[1])} ` +
        `stats=${JSON.stringify(stats)} seed=${seed}`
    );
    failed = true;
    break;
  }

  field.step(PHYS, {
    n: B,
    alpha: 0.55,
    lr: FIELD_LR,
    seed: step,
    source: "random",
    apply: true,
  });
  const fg = await field.readGrads();
  const fw = await field.readWeights();
  const fgs = summarize(fg);
  const fws = summarize(fw);
  if (fgs.firstBad >= 0 || fws.firstBad >= 0) {
    console.log(
      `FIRST NONFINITE step=${step} stage=${fgs.firstBad >= 0 ? "fieldGrad" : "fieldWeight.post"} ` +
        `fieldGrad(${fmt(fgs)}) fieldWeight(${fmt(fws)}) extGrad(${fmt(summarize(eg))}) ` +
        `stats=${JSON.stringify(stats)} seed=${seed}`
    );
    failed = true;
    break;
  }
  if (step % REPORT === 0 || step === STEPS - 1) {
    const scratch = await adv.readScratch(advScratchLayout(layout, adv.advL, TAG).stride * B);
    console.log(
      `step=${step} seed=${seed.toExponential(3)} loss=${stats.payoffUngated.toExponential(3)} ` +
        `sur=${stats.surprise.toExponential(3)} yRms=${stats.batchRms.toExponential(3)} ` +
        `FW(${fmt(fws)}) FG(${fmt(fgs)}) AW(${fmt(summarize(aw))}) ` +
        `AG(${fmt(summarize(ag))}) EG(${fmt(summarize(eg))})${pairGeometry(scratch)}`
    );
  }
}
console.log(failed ? "STABILITY PROBE FAILED" : "STABILITY PROBE FINITE");

adv.destroy();
field.destroy();
posBuf.destroy();
velBuf.destroy();
fieldBuf.destroy();
process.exit(failed ? 1 : 0);
