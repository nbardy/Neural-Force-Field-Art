/**
 * What does one pixel-GAN step actually COST?
 *
 *   bun tools/pixel_disc_cost_probe.ts
 *
 * The four Pixel pieces run at ~0.3 FPS, which on a phone reads as "frozen on
 * the previous artwork". This times PixelDiscTrainer.encodeStep in isolation
 * across dims, so the blame lands on a specific kernel shape rather than on
 * "mobile".
 *
 * Suspect: criticDisc / criticGen are `@compute @workgroup_size(1)` — ONE GPU
 * thread walks all G² cells, and holds cFeat[E·G²] + cSoft[K·G²] (+ dSoft, gD,
 * gW in the gen pass) as function-scope private arrays. Cost grows with G²·(E+K)
 * in both work AND per-thread private memory, and none of it is parallel.
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

const OCTAVES = 4, ENC_DIM = 2 + 4 * OCTAVES;
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
const wBuf = device.createBuffer({
  size: weights.byteLength, usage: GBU.STORAGE | GBU.COPY_DST | GBU.COPY_SRC,
});
device.queue.writeBuffer(wBuf, 0, weights);

const N = 80000, W = 390, H = 844;
const pos = new Float32Array(N * 2);
for (let i = 0; i < N; i++) { pos[i * 2] = W * rnd(); pos[i * 2 + 1] = H * rnd(); }
const posBuf = device.createBuffer({ size: pos.byteLength, usage: GBU.STORAGE | GBU.COPY_DST });
device.queue.writeBuffer(posBuf, 0, pos);
const velBuf = device.createBuffer({ size: pos.byteLength, usage: GBU.STORAGE | GBU.COPY_DST });

const CONFIGS = [
  { label: "GALLERY  G=8  E=4 K=8  h=16", G: 8, E: 4, K: 8, hidden: 16, gate: true },
  { label: "was      G=16 E=8 K=16 h=32", G: 16, E: 8, K: 16, hidden: 32, gate: false },
  { label: "         G=12 E=8 K=16 h=32", G: 12, E: 8, K: 16, hidden: 32, gate: false },
];

/**
 * A pixel step must fit inside the frame with advect + render, which alone
 * measure ~16 ms at 80k particles. Anything above this is what "frozen on
 * mobile" looked like: 36 ms/step at the old G=16 gallery config.
 */
const BUDGET_MS = 12;
let failures = 0;
const KINDS: PixelGanKind[] = ["vec-field", "next-frame", "real-fake", "inpaint"];

console.log(`particles=${N}  b=256  field=${layout.totalFloats} floats (dualFourier)\n`);
console.log("config                        kind         ms/step   private KB");
console.log("-".repeat(72));

for (const c of CONFIGS) {
  for (const kind of KINDS) {
    const trainer = new PixelDiscTrainer(device, layout, {
      fieldWeightsBuffer: wBuf,
      dims: { kind, G: c.G, E: c.E, K: c.K, hidden: c.hidden, dt: 0.15 },
      batchCap: 512, seed: 20260805,
    });
    trainer.setParticleBuffers(posBuf, velBuf, N);
    const step = async (i: number) => {
      const enc = device.createCommandEncoder();
      trainer.encodeStep(enc, {
        b: 256, alpha: 0.55, lr: 1e-3, genWeight: 0.04,
        applyDisc: true, width: W, height: H, maskSeed: i,
      });
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    };
    await step(0); // warm: shader compile + first-use allocation
    // Median of per-step times: the mean was swamped by GPU contention noise
    // (a browser tab rendering the same piece moved it 2-3x run to run).
    const REPS = 15;
    const times: number[] = [];
    for (let i = 1; i <= REPS; i++) {
      const s0 = performance.now();
      await step(i);
      times.push(performance.now() - s0);
    }
    times.sort((a, b) => a - b);
    const ms = times[Math.floor(times.length / 2)];
    const privKB = ((c.E + c.K) * c.G * c.G * 4) / 1024;
    const bad = c.gate && ms > BUDGET_MS;
    if (bad) failures++;
    console.log(
      `${bad ? "FAIL" : c.gate ? " ok " : "    "} ${c.label}  ${kind.padEnd(11)}  ` +
        `${ms.toFixed(1).padStart(7)}   ${privKB.toFixed(1).padStart(6)}`
    );
    trainer.destroy();
  }
  console.log("");
}

console.log(
  failures === 0
    ? `ALL PASS — every shipped kind under ${BUDGET_MS} ms/step`
    : `${failures} FAILURE(S) — a shipped pixel piece exceeds the frame budget`
);
process.exit(failures === 0 ? 0 : 1);
