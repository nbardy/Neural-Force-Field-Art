/**
 * How much AUTHORITY does the pixel adversary actually have over the field?
 *
 *   bun tools/pixel_disc_authority_probe.ts
 *
 * Pass B computes `g = fieldLossGrad + extGrad0` and hands g to Adam. So the
 * pixel GAN can only bend the artwork in proportion to ‖extGrad‖ / ‖fieldGrad‖.
 * This measures both on the SHIPPED gallery config (G=16 E=8 K=16 hidden=32,
 * ARCH.dualFourier, fieldLoss W_CHAOS .2 / W_ISO .6 / W_DIV .1, weight 0.04).
 */
import { setupGlobals } from "bun-webgpu";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };
(globalThis as any).GPUShaderStage ??= { COMPUTE: 4 };

const GBU = (globalThis as any).GPUBufferUsage;

import { layoutField } from "../src/render/webgpu/advect_wgsl";
import { PixelDiscTrainer } from "../src/render/webgpu/pixel_disc_train";
import { FusedTrainer } from "../src/render/webgpu/train";
import type { PixelGanKind } from "../src/core/gan/pixel_disc";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const l2 = (a: Float32Array) => Math.sqrt(a.reduce((s, x) => s + x * x, 0));

const OCTAVES = 4, ENC_DIM = 2 + 4 * OCTAVES;
const KINDS: PixelGanKind[] = ["vec-field", "next-frame", "real-fake", "inpaint"];
const GEN_WEIGHT: Record<PixelGanKind, number> = {
  "vec-field": 0.04, "next-frame": 0.04, "real-fake": 0.03, inpaint: 0.04,
};
const FIELD_LOSS = {
  W_CHAOS: 0.2, W_ISO: 0.6, W_DIV: 0.1, W_SPIRAL: 0, HH: 1e-2, SPIRAL_TURNS: 3,
};

const headDims = [
  { inSize: ENC_DIM, outSize: 32, activation: "selu" as const },
  { inSize: 32, outSize: 32, activation: "selu" as const },
  { inSize: 32, outSize: 2, activation: "tanh" as const },
];
const layout = layoutField("helmholtz", [headDims, headDims], {
  encoding: { kind: "fourier", octaves: OCTAVES },
});

const adapter = await (navigator as any).gpu.requestAdapter();
const device = await adapter.requestDevice();

const rnd = mulberry32(9);
const weights = new Float32Array(layout.totalFloats);
for (let i = 0; i < weights.length; i++) weights[i] = (rnd() - 0.5) * 0.1;

const N = 80000;
const VIEWPORTS: Array<[string, number, number]> = [
  ["phone 390x844", 390, 844],
  ["desktop 1280x800", 1280, 800],
];
const LOSSES: Array<[string, typeof FIELD_LOSS]> = [
  ["shipped fieldLoss", FIELD_LOSS],
  ["ZERO_FIELD_LOSS", { W_CHAOS: 0, W_ISO: 0, W_DIV: 0, W_SPIRAL: 0, HH: 1e-2, SPIRAL_TURNS: 3 }],
];

for (const [vpName, W, H] of VIEWPORTS) {
for (const [lossName, loss] of LOSSES) {
const pos = new Float32Array(N * 2);
const vel = new Float32Array(N * 2);
for (let i = 0; i < N; i++) {
  pos[i * 2] = W * rnd();
  pos[i * 2 + 1] = H * rnd();
}
const posBuf = device.createBuffer({ size: pos.byteLength, usage: GBU.STORAGE | GBU.COPY_DST });
device.queue.writeBuffer(posBuf, 0, pos);
const velBuf = device.createBuffer({ size: vel.byteLength, usage: GBU.STORAGE | GBU.COPY_DST });
device.queue.writeBuffer(velBuf, 0, vel);

const phys = { width: W, height: H, forceMagnitude: 24 * (1 - 0.97) * 0.65, friction: 0.97, maxVelocity: 24 };

console.log(`\n### ${vpName} · ${lossName}`);
console.log("kind         ‖fieldGrad‖   ‖extGrad‖    ratio      adversary share");
console.log("-".repeat(72));

for (const kind of KINDS) {
  const wBuf = device.createBuffer({
    size: weights.byteLength, usage: GBU.STORAGE | GBU.COPY_DST | GBU.COPY_SRC,
  });
  device.queue.writeBuffer(wBuf, 0, weights);

  const pd = new PixelDiscTrainer(device, layout, {
    fieldWeightsBuffer: wBuf,
    dims: { kind, G: 16, E: 8, K: 16, hidden: 32, dt: 0.15 },
    batchCap: 512, seed: 20260805,
  });
  pd.setParticleBuffers(posBuf, velBuf, N);
  const trainer = new FusedTrainer(device, layout, {
    weightsBuffer: wBuf, batchCap: 1024, kSteps: 1,
    extGradBuffers: [pd.extGradsBuf], loss, border: { tag: "wrap" },
  });
  trainer.uploadWeights(weights);
  trainer.setParticleBuffers(posBuf, velBuf, N);

  // Warm the critic the way the loop does, so we compare a settled adversary.
  let fg = 0, eg = 0;
  for (let step = 0; step < 12; step++) {
    const enc = device.createCommandEncoder();
    pd.encodeStep(enc, {
      b: 256, alpha: 0.55, lr: 1e-3, genWeight: GEN_WEIGHT[kind],
      applyDisc: true, width: W, height: H, maskSeed: step,
    });
    trainer.encodeStep(enc, phys, {
      n: 256, alpha: 0.55, lr: 1.5e-3, seed: step, source: "particles",
    });
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (step >= 8) {
      fg += l2(await trainer.readGrads());
      eg += l2(await pd.readExtGrads());
    }
  }
  fg /= 4; eg /= 4;
  const ratio = eg / fg;
  console.log(
    `${kind.padEnd(12)} ${fg.toExponential(3)}   ${eg.toExponential(3)}   ` +
      `${ratio.toExponential(2)}   ${(100 * ratio / (1 + ratio)).toFixed(4)}%`
  );
  pd.destroy(); trainer.destroy?.(); wBuf.destroy();
}
posBuf.destroy(); velBuf.destroy();
}
}
