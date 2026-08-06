/**
 * Pixel GAN verification — four kinds.
 *
 *   bun tools/pixel_disc_test.ts
 *
 * §1 Soft splat + VJP vs FD
 * §2 Each kind: disc loss drops under Adam; gen dF finite
 * §3 (GPU) One encodeStep per kind produces finite extGrads
 */
import {
  softSplatDensity,
  softSplatVjp,
  softResidual,
  softResidualGrad,
  initPixelDiscWeights,
  pixelDiscStep,
  cellCenters,
  PIXEL_DISC_DEFAULTS,
  type PixelDiscDims,
  type PixelGanKind,
} from "../src/core/gan/pixel_disc";

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

const KINDS: PixelGanKind[] = [
  "vec-field",
  "next-frame",
  "real-fake",
  "inpaint",
];

function baseDims(kind: PixelGanKind): PixelDiscDims {
  return {
    kind,
    G: 8,
    E: 4,
    K: 8,
    hidden: 8,
    dt: 0.05,
  };
}

console.log("=== pixel_disc CPU oracle (4 kinds) ===");
console.log(
  `defaults G=${PIXEL_DISC_DEFAULTS.G} E=${PIXEL_DISC_DEFAULTS.E} K=${PIXEL_DISC_DEFAULTS.K}`
);

// ── §1 soft splat ──────────────────────────────────────────────────────────
{
  console.log("\n§1 soft splat + VJP");
  const B = 16;
  const G = 8;
  const rnd = mulberry32(1);
  const pos = new Float64Array(B * 2);
  for (let i = 0; i < B * 2; i++) pos[i] = 0.15 + 0.7 * rnd();
  const D = softSplatDensity(pos, B, G);
  let mass = 0;
  for (let i = 0; i < D.length; i++) mass += D[i];
  ok(Math.abs(mass - G * G) < 1e-6, `mass conserved ${mass} ≈ ${G * G}`);

  const dD = new Float64Array(G * G);
  for (let i = 0; i < dD.length; i++) dD[i] = rnd() * 2 - 1;
  const dPos = new Float64Array(B * 2);
  softSplatVjp(pos, dD, B, G, dPos);

  const EPS = 1e-5;
  let maxRel = 0;
  let checked = 0;
  for (let i = 0; i < B * 2; i++) {
    const p0 = pos[i];
    const x = pos[(i >> 1) << 1];
    const y = pos[((i >> 1) << 1) + 1];
    if (x * G - 0.5 <= EPS || x * G - 0.5 >= G - 1 - EPS) continue;
    if (y * G - 0.5 <= EPS || y * G - 0.5 >= G - 1 - EPS) continue;
    pos[i] = p0 + EPS;
    const Dhi = softSplatDensity(pos, B, G);
    pos[i] = p0 - EPS;
    const Dlo = softSplatDensity(pos, B, G);
    pos[i] = p0;
    let fd = 0;
    for (let c = 0; c < G * G; c++) fd += (dD[c] * (Dhi[c] - Dlo[c])) / (2 * EPS);
    const an = dPos[i];
    const rel = Math.abs(an - fd) / (1e-8 + Math.abs(fd));
    maxRel = Math.max(maxRel, rel);
    checked++;
  }
  ok(
    checked > 8 && maxRel < 2e-3,
    `splat VJP vs FD maxRel=${maxRel.toExponential(2)} (n=${checked})`
  );

  const sg = softResidualGrad(0.5, -0.2, 0.1, 0.3);
  ok(
    Math.abs(sg.r - softResidual(0.5, -0.2, 0.1, 0.3)) < 1e-12,
    "soft residual self-consistent"
  );
}

// ── §2 per-kind disc descent + gen dF ──────────────────────────────────────
{
  console.log("\n§2 disc descent + gen dF (all kinds)");
  for (const kind of KINDS) {
    const dims = baseDims(kind);
    const B = 32;
    const rnd = mulberry32(3 + kind.length);
    const pos = new Float64Array(B * 2);
    const forces = new Float64Array(B * 2);
    for (let i = 0; i < B; i++) {
      pos[i * 2] = 0.2 + 0.6 * rnd();
      pos[i * 2 + 1] = 0.2 + 0.6 * rnd();
      forces[i * 2] = rnd() * 2 - 1;
      forces[i * 2 + 1] = rnd() * 2 - 1;
    }
    const forceGrid =
      kind === "vec-field"
        ? (() => {
            const g = new Float64Array(dims.G * dims.G * 2);
            const ctr = cellCenters(dims.G);
            for (let c = 0; c < dims.G * dims.G; c++) {
              // smooth target field
              g[c * 2] = Math.sin(6 * Math.PI * ctr[c * 2]) * 0.3;
              g[c * 2 + 1] = Math.cos(6 * Math.PI * ctr[c * 2 + 1]) * 0.3;
            }
            return g;
          })()
        : undefined;
    const fakePos =
      kind === "real-fake"
        ? (() => {
            const p = new Float64Array(B * 2);
            for (let i = 0; i < B * 2; i++) p[i] = rnd();
            return p;
          })()
        : undefined;

    const w = initPixelDiscWeights(dims, 7);
    const losses: number[] = [];
    const steps = kind === "inpaint" || kind === "real-fake" ? 150 : 80;
    const lr = kind === "inpaint" || kind === "real-fake" ? 0.08 : 0.03;
    for (let t = 0; t < steps; t++) {
      const r = pixelDiscStep(pos, forces, w, dims, {
        applyDisc: true,
        genSign: 0,
        lr,
        forceGrid,
        fakePos,
        // Fixed mask so inpaint has a stationary target while the critic learns.
        maskSeed: 7,
      });
      losses.push(r.discLoss);
    }
    const drop = losses[losses.length - 1] < losses[0] * 0.95;
    ok(
      drop,
      `${kind} discLoss ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`
    );

    const rGen = pixelDiscStep(pos, forces, w, dims, {
      applyDisc: false,
      genSign: -1,
      lr: 0,
      forceGrid,
      fakePos,
      maskSeed: 99,
    });
    let dFnorm = 0;
    for (let i = 0; i < rGen.dF.length; i++) dFnorm += rGen.dF[i] * rGen.dF[i];
    ok(
      Number.isFinite(dFnorm) && dFnorm > 0,
      `${kind} gen dF ‖·‖²=${dFnorm.toExponential(2)}`
    );
  }
}

// ── §3 GPU smoke per kind ──────────────────────────────────────────────────
{
  console.log("\n§3 GPU smoke (optional)");
  try {
    const { setupGlobals } = await import("bun-webgpu");
    setupGlobals();
    (globalThis as any).GPUBufferUsage ??= {
      MAP_READ: 1,
      MAP_WRITE: 2,
      COPY_SRC: 4,
      COPY_DST: 8,
      UNIFORM: 64,
      STORAGE: 128,
    };
    (globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };
    (globalThis as any).GPUShaderStage ??= { COMPUTE: 4 };

    const adapter = await (navigator as any).gpu.requestAdapter();
    if (!adapter) {
      console.log("  skip  no WebGPU adapter");
    } else {
      const device = await adapter.requestDevice();
      const { layoutField } = await import("../src/render/webgpu/advect_wgsl");
      const { PixelDiscTrainer } = await import(
        "../src/render/webgpu/pixel_disc_train"
      );
      const fieldDims = (inSize: number) => [
        { inSize, outSize: 16, activation: "selu" as const },
        { inSize: 16, outSize: 16, activation: "selu" as const },
        { inSize: 16, outSize: 2, activation: "tanh" as const },
      ];
      const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
      const weights = new Float32Array(layout.totalFloats);
      const rnd = mulberry32(9);
      for (let i = 0; i < weights.length; i++) weights[i] = (rnd() - 0.5) * 0.1;

      const wBuf = device.createBuffer({
        size: weights.byteLength,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
      device.queue.writeBuffer(wBuf, 0, weights);

      const N = 256;
      const pos = new Float32Array(N * 2);
      for (let i = 0; i < N; i++) {
        pos[i * 2] = 100 + 600 * rnd();
        pos[i * 2 + 1] = 80 + 440 * rnd();
      }
      const posBuf = device.createBuffer({
        size: pos.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(posBuf, 0, pos);
      const velBuf = device.createBuffer({
        size: pos.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      for (const kind of KINDS) {
        device.pushErrorScope("validation");
        const trainer = new PixelDiscTrainer(device, layout, {
          fieldWeightsBuffer: wBuf,
          dims: { kind, G: 8, E: 4, K: 8, hidden: 8, dt: 0.05 },
          batchCap: 128,
          seed: 99,
        });
        const shaderErr = await device.popErrorScope();
        if (shaderErr) {
          ok(
            false,
            `${kind} shader: ${shaderErr.message.slice(0, 160)}`
          );
          trainer.destroy();
          continue;
        }
        trainer.setParticleBuffers(posBuf, velBuf, N);
        const enc = device.createCommandEncoder();
        trainer.encodeStep(enc, {
          b: 64,
          alpha: 0.5,
          lr: 1e-3,
          genWeight: 1.0,
          width: 800,
          height: 600,
          maskSeed: 42,
        });
        device.pushErrorScope("validation");
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const submitErr = await device.popErrorScope();
        ok(
          !submitErr,
          submitErr
            ? `${kind} submit: ${submitErr.message.slice(0, 120)}`
            : `${kind} submit ok`
        );
        const ext = await trainer.readExtGrads();
        let nrm = 0;
        let finite = true;
        for (let i = 0; i < ext.length; i++) {
          if (!Number.isFinite(ext[i])) finite = false;
          nrm += ext[i] * ext[i];
        }
        ok(
          finite && nrm > 1e-20,
          `${kind} GPU extGrads ‖·‖²=${nrm.toExponential(2)}`
        );
        trainer.destroy();
      }
      wBuf.destroy();
      posBuf.destroy();
      velBuf.destroy();
      device.destroy?.();
    }
  } catch (e) {
    console.log(
      `  skip  GPU path errored: ${(e as Error).message?.slice(0, 160)}`
    );
  }
}

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
