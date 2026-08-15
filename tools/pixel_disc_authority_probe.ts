/**
 * Does the pixel adversary actually STEER the field?
 *
 *   bun tools/pixel_disc_authority_probe.ts
 *
 * REGRESSION GUARD for the "Pixel GANs do nothing" incident. Everything about
 * that bug was invisible to the existing suite: every pass dispatched, every
 * gradient was finite, tools/pixel_disc_test.ts passed. The four Pixel pieces
 * simply also shipped a structural fieldLoss, and pass B applies the SUM
 * (train_wgsl.ts: `g = g + extGrad0[t]`, then Adam) — so the critic owned
 * ~0.006% of each update and the artwork was pure W_ISO.
 *
 * What this measures, per kind, at the SHIPPED gallery config (G=16 E=8 K=16
 * hidden=32 on ARCH.dualFourier — tools/pixel_disc_test.ts only covers
 * G=8 E=4 K=8 on a tiny raw field, which is why it never saw this):
 *
 *   ‖extGrad‖ / ‖grads‖   where `grads` is the TOTAL applied gradient
 *                         (FusedTrainer.readGrads returns post-sum `g`)
 *
 * A game piece whose critic is the sole driver scores 1.0. Anything near zero
 * means the adversary is being outvoted and the piece is decorative.
 *
 * Also sweeps phone vs desktop viewport, because the bug was first reported as
 * mobile-only: the splat normalizes by width/height, so the ratio is identical
 * at 390×844 and 1280×800. If those two columns ever diverge, something in the
 * pixel path has picked up a real viewport dependence.
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
import type { FieldLossSpec } from "../src/render/webgpu/train_wgsl";
import type { PixelGanKind } from "../src/core/gan/pixel_disc";

/** Below this share of the applied gradient the critic cannot move the art. */
const MIN_AUTHORITY = 0.5;

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
// Mirrors the GALLERY entries. Keep in sync if the pieces are retuned.
const GEN_WEIGHT: Record<PixelGanKind, number> = {
  "vec-field": 0.04, "next-frame": 0.04, "real-fake": 0.03, inpaint: 0.04,
};
const ZERO: FieldLossSpec = {
  W_CHAOS: 0, W_ISO: 0, W_DIV: 0, W_SPIRAL: 0, HH: 1e-2, SPIRAL_TURNS: 3,
} as FieldLossSpec;
/** The loss the four pieces used to ship — kept so the probe shows the delta. */
const REGRESSED: FieldLossSpec = {
  W_CHAOS: 0.2, W_ISO: 0.6, W_DIV: 0.1, W_SPIRAL: 0, HH: 1e-2, SPIRAL_TURNS: 3,
} as FieldLossSpec;

const headDims = [
  { inSize: ENC_DIM, outSize: 32, activation: "selu" as const },
  { inSize: 32, outSize: 32, activation: "selu" as const },
  { inSize: 32, outSize: 2, activation: "tanh" as const },
];
const layout = layoutField("helmholtz", [headDims, headDims], {
  encoding: { kind: "fourier", octaves: OCTAVES },
});

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) { console.log("no WebGPU adapter"); process.exit(1); }
const device = await adapter.requestDevice();

const rnd = mulberry32(9);
const weights = new Float32Array(layout.totalFloats);
for (let i = 0; i < weights.length; i++) weights[i] = (rnd() - 0.5) * 0.1;

const N = 80000;
const VIEWPORTS: Array<[string, number, number]> = [
  ["phone 390x844", 390, 844],
  ["desktop 1280x800", 1280, 800],
];
const CASES: Array<[string, FieldLossSpec, boolean]> = [
  // label, loss, enforced
  ["ZERO_FIELD_LOSS (shipped)", ZERO, true],
  ["W_CHAOS/W_ISO/W_DIV (the 2026-08 regression)", REGRESSED, false],
];

let failures = 0;
console.log(`weights=${layout.totalFloats} particles=${N}\n`);

for (const [vpName, W, H] of VIEWPORTS) {
  const pos = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    pos[i * 2] = W * rnd();
    pos[i * 2 + 1] = H * rnd();
  }
  const posBuf = device.createBuffer({ size: pos.byteLength, usage: GBU.STORAGE | GBU.COPY_DST });
  device.queue.writeBuffer(posBuf, 0, pos);
  const velBuf = device.createBuffer({ size: pos.byteLength, usage: GBU.STORAGE | GBU.COPY_DST });
  const phys = {
    width: W, height: H,
    forceMagnitude: 24 * (1 - 0.97) * 0.65, friction: 0.97, maxVelocity: 24,
  };

  for (const [caseName, loss, enforced] of CASES) {
    console.log(`### ${vpName} · ${caseName}`);
    console.log("  kind          ‖extGrad‖    ‖grads‖(total)   authority");
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

      // Warm the critic the way the loop does, then average a few steps.
      let tot = 0, ext = 0, samples = 0;
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
          tot += l2(await trainer.readGrads());
          ext += l2(await pd.readExtGrads());
          samples++;
        }
      }
      tot /= samples; ext /= samples;
      const authority = tot > 0 ? ext / tot : 0;
      const bad = enforced && !(authority >= MIN_AUTHORITY);
      if (bad) failures++;
      console.log(
        `  ${bad ? "FAIL" : " ok "} ${kind.padEnd(11)} ${ext.toExponential(3)}    ` +
          `${tot.toExponential(3)}      ${(100 * authority).toFixed(4)}%`
      );
      pd.destroy(); trainer.destroy?.(); wBuf.destroy();
    }
    console.log("");
  }
  posBuf.destroy(); velBuf.destroy();
}

console.log(
  failures === 0
    ? `ALL PASS — critic authority ≥ ${100 * MIN_AUTHORITY}% on every shipped kind`
    : `${failures} FAILURE(S) — a pixel piece's critic is being outvoted; check its fieldLoss`
);
process.exit(failures === 0 ? 0 : 1);
