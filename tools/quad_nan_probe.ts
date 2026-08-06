/**
 * Particle-coupled Quad WTA NaN repro.
 *
 * The offline adversary_stability_probe stays finite because it never advects
 * particles — live clouds concentrate and that is where Quad blows. This probe
 * mirrors the gallery loop: adversary D→G on particles, field Adam on particles
 * with extGrads, then fused advect.
 *
 *   bun tools/quad_nan_probe.ts
 *   STEPS=5000 REPORT=30 bun tools/quad_nan_probe.ts
 */
import { setupGlobals } from "bun-webgpu";
import {
  layoutField,
  type LayerDims,
} from "../src/render/webgpu/advect_wgsl";
import { advectShader } from "../src/render/webgpu/advect_wgsl";
import {
  AdversaryTrainer,
  packAdversaryInit,
} from "../src/render/webgpu/adversary_train";
import { FusedTrainer } from "../src/render/webgpu/train";
import type { FieldLossSpec } from "../src/render/webgpu/train_wgsl";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };
(globalThis as any).GPUShaderStage ??= { COMPUTE: 4 };

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device: GPUDevice = await adapter.requestDevice();

const STEPS = Number(process.env.STEPS ?? 4000);
const REPORT = Number(process.env.REPORT ?? 30);
const B = Number(process.env.B ?? 256);
const K = Number(process.env.K ?? 6);
const N = Number(process.env.N ?? 20000);
const FIELD_LR = Number(process.env.FIELD_LR ?? 0.001);
const DISC_LR = Number(process.env.DISC_LR ?? 0.003);
const WEIGHT = Number(process.env.WEIGHT ?? 0.012);
const DRIVE = Number(process.env.DRIVE ?? 0.65);
const TAG = (process.env.TAG ?? "quad-labelled") as
  | "quad-labelled"
  | "tri"
  | "pair"
  | "point";
const FINE_AFTER = Number(process.env.FINE_AFTER ?? -1);
const FRICTION = 0.97;
const MAX_VEL = 24;
const FORCE = (DRIVE * MAX_VEL * (1 - FRICTION)) / FRICTION;
const PHYS = {
  width: 800,
  height: 600,
  forceMagnitude: FORCE,
  friction: FRICTION,
  maxVelocity: MAX_VEL,
};
const ZERO: FieldLossSpec = {
  W_CHAOS: 0,
  W_ISO: 0,
  W_DIV: 0,
  W_SPIRAL: 0,
  W_COVER: 0,
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
  for (const seg of layout.segments) {
    if (seg.role !== "kernel") continue;
    const L = layout.spec.heads[seg.head].layers[seg.layer];
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

// Minimal advect (no tfjs) — same WGSL the live AdvectKernel uses.
const advectPipe = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: device.createShaderModule({
      code: advectShader(layout, {
        stageWeights: layout.totalFloats * 4 <= 16384,
        precision: "f32",
        border: { tag: "wrap" },
      }),
    }),
    entryPoint: "main",
  },
});
const uniData = new ArrayBuffer(48);
const uniF = new Float32Array(uniData);
const uniU = new Uint32Array(uniData);
uniF[0] = PHYS.width;
uniF[1] = PHYS.height;
uniF[2] = PHYS.forceMagnitude;
uniF[3] = PHYS.friction;
uniF[4] = PHYS.maxVelocity;
uniF[6] = 0.003; // resetRate
const uniBuf = device.createBuffer({
  size: 48,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const advectBind = device.createBindGroup({
  layout: advectPipe.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uniBuf } },
    { binding: 1, resource: { buffer: fieldBuf } },
    { binding: 2, resource: { buffer: posBuf } },
    { binding: 3, resource: { buffer: velBuf } },
  ],
});
function advect(seed: number, alpha: number): void {
  uniF[5] = alpha;
  uniU[7] = seed >>> 0;
  uniU[8] = N;
  device.queue.writeBuffer(uniBuf, 0, uniData);
  const enc = device.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(advectPipe);
  p.setBindGroup(0, advectBind);
  p.dispatchWorkgroups(Math.ceil(N / 64));
  p.end();
  device.queue.submit([enc.finish()]);
}

const adv = new AdversaryTrainer(device, layout, {
  observerGeometry: "periodic",
  tag: TAG,
  k: K,
  relaxEps: 0.05,
  hiddenUnits: 32,
  featureDim: 16,
  batchCap: Math.max(B, 1024),
  fieldWeightsBuffer: fieldBuf,
  particleCount: N,
  seed: 103,
});
adv.uploadAdvWeights(packAdversaryInit(adv.advL, 103));
adv.setParticleBuffers(posBuf, velBuf, N);
const field = new FusedTrainer(device, layout, {
  batchCap: Math.max(B, 1024),
  weightsBuffer: fieldBuf,
  extGradBuffer: adv.extGradsBuf,
  loss: ZERO,
});
field.setParticleBuffers(posBuf, velBuf, N);

type Summary = { max: number; rms: number; firstBad: number };
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

console.log(
  `coupled tag=${TAG} steps=${STEPS} B=${B} K=${K} N=${N} ` +
    `fieldLR=${FIELD_LR} discLR=${DISC_LR} weight=${WEIGHT} drive=${DRIVE} ` +
    `force=${FORCE.toFixed(4)}`
);

let failed = false;
for (let step = 0; step < STEPS; step++) {
  // Warm cloud a few frames before training, matching browser startup feel.
  if (step < 5) {
    advect(step, 0.55);
    continue;
  }

  const seed = adv.genSeed(WEIGHT, 1, B);
  adv.step(PHYS, {
    b: B,
    alpha: 0.55,
    lr: DISC_LR,
    seed: step,
    source: "particles",
    genSeed: seed,
    applyDisc: true,
  });
  const fine = FINE_AFTER >= 0 && step >= FINE_AFTER;
  if (fine) {
    const stats = await adv.readStats();
    const aw = summarize(await adv.readAdvWeights());
    const eg = summarize(await adv.readExtGrads());
    const fw = summarize(await field.readWeights());
    if (
      !Number.isFinite(stats.discLoss) ||
      aw.firstBad >= 0 ||
      eg.firstBad >= 0 ||
      fw.firstBad >= 0
    ) {
      console.log(
        `AFTER ADV step=${step} loss=${stats.discLoss} ` +
          `FW(${fmt(fw)}) AW(${fmt(aw)}) EG(${fmt(eg)}) wins=${stats.winCounts.join(",")}`
      );
      failed = true;
      break;
    }
  }
  field.step(PHYS, {
    n: B,
    alpha: 0.55,
    lr: FIELD_LR,
    seed: step,
    source: "particles",
    apply: true,
  });
  if (fine) {
    const fw = summarize(await field.readWeights());
    const fg = summarize(await field.readGrads());
    if (fw.firstBad >= 0 || fg.firstBad >= 0) {
      console.log(
        `AFTER FIELD step=${step} FW(${fmt(fw)}) FG(${fmt(fg)})`
      );
      failed = true;
      break;
    }
  }
  advect(step, 0.55);

  if (fine || step % REPORT === 0 || step === STEPS - 1) {
    const stats = await adv.readStats();
    const ag = summarize(await adv.readAdvGrads());
    const aw = summarize(await adv.readAdvWeights());
    const eg = summarize(await adv.readExtGrads());
    const fw = summarize(await field.readWeights());
    const fg = summarize(await field.readGrads());
    const stages = [
      ["stats", summarize([stats.discLoss, stats.surprise, stats.batchRms])],
      ["advGrad", ag],
      ["advWeight", aw],
      ["extGrad", eg],
      ["fieldGrad", fg],
      ["fieldWeight", fw],
    ] as const;
    const bad = stages.find(([, s]) => s.firstBad >= 0);
    if (fine || step % REPORT === 0 || bad) {
      console.log(
        `step=${step} seed=${seed.toExponential(3)} loss=${Number.isFinite(stats.discLoss) ? stats.discLoss.toExponential(3) : String(stats.discLoss)} ` +
          `sur=${Number.isFinite(stats.surprise) ? stats.surprise.toExponential(3) : String(stats.surprise)} ` +
          `yRms=${Number.isFinite(stats.batchRms) ? stats.batchRms.toExponential(3) : String(stats.batchRms)} ` +
          `wins=${stats.winCounts.join(",")} ` +
          `FW(${fmt(fw)}) FG(${fmt(fg)}) AW(${fmt(aw)}) AG(${fmt(ag)}) EG(${fmt(eg)})`
      );
    }
    if (bad) {
      console.log(
        `FIRST NONFINITE step=${step} stage=${bad[0]} ${fmt(bad[1])} ` +
          `stats=${JSON.stringify(stats)}`
      );
      failed = true;
      break;
    }
  }
}

console.log(failed ? "QUAD COUPLED PROBE FAILED" : "QUAD COUPLED PROBE FINITE");
adv.destroy();
field.destroy();
posBuf.destroy();
velBuf.destroy();
fieldBuf.destroy();
uniBuf.destroy();
process.exit(failed ? 1 : 0);
