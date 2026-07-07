/**
 * TRAJECTORY-WINDOW EQUIVALENCE test: proves that with source:"particles" and
 * kSteps=K, the FusedTrainer's imagined K-step rollout from a sampled
 * particle's CURRENT state is EXACTLY the trajectory the AdvectKernel produces
 * for that particle over the next K frames (same weights, same physics, both
 * fp32) — up to (a) random resets (disabled here: resetRate=0) and (b) tiny
 * fp32 op-order differences between the two MLP evaluators (advect: unrolled
 * vec4 staged weights; trainer: looped scalar).
 *
 *   bun tools/window_test.ts
 *
 * Consequence: `?rollout=K&trainEvery=K` (alias `?window=K`) IS trajectory-
 * window training in leading form — no recording machinery, no wasted
 * forwards. This is the proof behind closing the "recorded (true-trailing)
 * windows" branch in docs/PLAN_SPECIES_AND_BATCHES.md §1.
 *
 * Checks:
 *   1. every sampled batch start (batchBuf) matches EXACTLY ONE live particle
 *      bit-for-bit, and the scratch header's pos_0 equals it bit-for-bit
 *   2. velocities enter the rollout: scratch velPre_1 == (v0 + Fs_0)·friction
 *      from the matched particle's REAL velocity, and pos_1 differs from a
 *      vel-0 prediction whenever the particle was moving
 *   3. after K=6 real advect frames, each matched particle's position equals
 *      the sample's scratch pos_K within 0.05 px on the torus
 */
import { setupGlobals } from "bun-webgpu";
setupGlobals();

// Adapter shims for tfjs 4.10 under bun-webgpu — copied verbatim from
// tools/integration_test.ts (proven):
//  1. tfjs calls adapter.requestAdapterInfo(), removed from the spec; Dawn
//     exposes .info.
//  2. bun-webgpu's requestDevice validator rejects feature names it doesn't
//     know — retry-strip any feature the wrapper refuses to pack.
{
  const gpu = (navigator as any).gpu;
  const origReq = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (o?: any) => {
    const a = await origReq(o);
    if (!a) return a;
    if (!a.requestAdapterInfo) {
      a.requestAdapterInfo = async () => a.info ?? {};
    }
    const origDev = a.requestDevice.bind(a);
    a.requestDevice = async (desc?: any) => {
      const d = desc
        ? { ...desc, requiredFeatures: [...(desc.requiredFeatures ?? [])].flat(Infinity) }
        : desc;
      if (d) delete d.requiredLimits;
      for (;;) {
        try {
          return await origDev(d);
        } catch (e: any) {
          const m = /Invalid feature required: (\S+)/.exec(String(e?.message ?? e));
          if (!m || !d?.requiredFeatures?.includes(m[1])) throw e;
          d.requiredFeatures = d.requiredFeatures.filter((f: string) => f !== m[1]);
        }
      }
    };
    return a;
  };
}

const tf = await import("@tensorflow/tfjs");
(globalThis as any).window ??= globalThis;
(globalThis as any).GPUBuffer ??= class GPUBufferShim {};
(globalThis as any).GPUTexture ??= class GPUTextureShim {};
await import("@tensorflow/tfjs-backend-webgpu");
const { HelmholtzField } = await import("../src/core/field/helmholtz");
const { AdvectKernel } = await import("../src/render/webgpu/advect");
const { FusedTrainer } = await import("../src/render/webgpu/train");
const { trainScratchLayout } = await import("../src/render/webgpu/train_wgsl");

await tf.setBackend("webgpu");
await tf.ready();
if (tf.getBackend() !== "webgpu") {
  console.error(`FATAL: backend is ${tf.getBackend()}, not webgpu`);
  process.exit(1);
}
console.log("tfjs backend: webgpu (bun-webgpu/Dawn)\n");

const device: any = (tf.backend() as any).device;
let failures = 0;

async function readF32(buf: any, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: 1 | 8 });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

const PHYS = {
  width: 800, height: 600,
  forceMagnitude: 3.5, friction: 0.99, maxVelocity: 26,
  resetRate: 0, // NO teleports — the one thing that breaks the equivalence
};
const N = 2000;
const K = 6;
const BATCH = 64;
const ALPHA = 0.6;

/** torus distance on one coordinate */
const torusD = (a: number, b: number, m: number) => {
  let d = Math.abs(a - b);
  return Math.min(d, m - d);
};

const field = new HelmholtzField({ alpha: ALPHA }); // classless
const kernel = AdvectKernel.fromField(field as any, PHYS, N);
const trainer = new FusedTrainer(device, kernel.layout, {
  weightsBuffer: kernel.weightsBuffer,
  batchCap: 256,
  kSteps: K,
});
trainer.uploadWeights(kernel.packCurrentWeights());
kernel.syncFromTfjs = false; // weights frozen on the GPU, both kernels read them
trainer.setParticleBuffers(kernel.posBuffer, kernel.velBuffer, kernel.count);

// warm up velocities: 3 real frames so v ≠ 0 when the trainer samples
for (let s = 1; s <= 3; s++) kernel.step(s, ALPHA);

// snapshot the EXACT state the trainer will sample (nothing touches the
// buffers between this readback and the trainer step)
const posPre = await readF32(kernel.posBuffer, 2 * N);
const velPre = await readF32(kernel.velBuffer, 2 * N);

// the frozen window: lr=0 + apply=false ⇒ pass B leaves weights untouched
trainer.step(
  { width: PHYS.width, height: PHYS.height, forceMagnitude: PHYS.forceMagnitude,
    friction: PHYS.friction, maxVelocity: PHYS.maxVelocity },
  { n: BATCH, alpha: ALPHA, lr: 0, apply: false, source: "particles", seed: 42 }
);

const sl = trainScratchLayout(kernel.layout, K);
const batch = await readF32((trainer as any).batchBuf, 2 * BATCH);
const scratch = await readF32((trainer as any).scratchBuf, BATCH * sl.sampleStride);

// index pre-step positions for exact-bit matching (f32 → JS number is exact)
const byPos = new Map<string, number[]>();
for (let i = 0; i < N; i++) {
  const key = `${posPre[2 * i]}|${posPre[2 * i + 1]}`;
  const arr = byPos.get(key);
  if (arr) arr.push(i);
  else byPos.set(key, [i]);
}

// 1 — every sample start matches exactly one particle, bit-for-bit, and the
//     scratch header pos_0 is that same value
{
  let matched = 0, ambiguous = 0, headerMismatch = 0;
  const sampleIdx = new Int32Array(BATCH).fill(-1);
  for (let s = 0; s < BATCH; s++) {
    const bx = batch[2 * s], by = batch[2 * s + 1];
    const hit = byPos.get(`${bx}|${by}`);
    if (!hit) continue;
    if (hit.length !== 1) { ambiguous++; continue; }
    sampleIdx[s] = hit[0];
    matched++;
    const base = s * sl.sampleStride + sl.posOff;
    if (scratch[base] !== bx || scratch[base + 1] !== by) headerMismatch++;
  }
  const ok = matched === BATCH && ambiguous === 0 && headerMismatch === 0;
  if (!ok) failures++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  sampling: ${matched}/${BATCH} batch starts match a live ` +
    `particle bit-for-bit (${ambiguous} ambiguous, ${headerMismatch} scratch pos_0 mismatches)`
  );

  // 2 — velocities entered the rollout: velPre_1 == (v0 + Fs_0)·friction from
  //     the matched particle's REAL velocity; and pos_1 must differ from a
  //     vel-0 prediction whenever the particle was moving.
  {
    let maxVelErr = 0, moving = 0, distinguished = 0;
    for (let s = 0; s < BATCH; s++) {
      const idx = sampleIdx[s];
      if (idx < 0) continue;
      const base = s * sl.sampleStride;
      const fsx = scratch[base + sl.fsOff], fsy = scratch[base + sl.fsOff + 1];
      const v0x = velPre[2 * idx], v0y = velPre[2 * idx + 1];
      // positive check: the trainer's stored velPre_1 used the REAL velocity
      const expX = (v0x + fsx) * PHYS.friction;
      const expY = (v0y + fsy) * PHYS.friction;
      maxVelErr = Math.max(
        maxVelErr,
        Math.abs(scratch[base + sl.velPreOff] - expX),
        Math.abs(scratch[base + sl.velPreOff + 1] - expY)
      );
      // negative check: a vel-0 rollout would land somewhere else
      if (Math.hypot(v0x, v0y) > 0.1) {
        moving++;
        const clip = (x: number) =>
          Math.min(PHYS.maxVelocity, Math.max(-PHYS.maxVelocity, x));
        const fmod = (x: number, m: number) => x - Math.floor(x / m) * m;
        const p0x = scratch[base + sl.posOff], p0y = scratch[base + sl.posOff + 1];
        const z1x = fmod(p0x + clip(fsx * PHYS.friction), PHYS.width);
        const z1y = fmod(p0y + clip(fsy * PHYS.friction), PHYS.height);
        const p1x = scratch[base + sl.posOff + 2], p1y = scratch[base + sl.posOff + 3];
        const d = Math.max(
          torusD(p1x, z1x, PHYS.width),
          torusD(p1y, z1y, PHYS.height)
        );
        if (d > 1e-3) distinguished++;
      }
    }
    const ok2 = maxVelErr < 1e-3 && moving > 0 && distinguished === moving;
    if (!ok2) failures++;
    console.log(
      `${ok2 ? "PASS" : "FAIL"}  velocity entry: velPre_1 matches (v0+Fs_0)·fric ` +
      `(maxΔ=${maxVelErr.toExponential(2)}); pos_1 ≠ vel-0 prediction for ` +
      `${distinguished}/${moving} moving samples`
    );
  }

  // 3 — THE EQUIVALENCE: advect the REAL population K frames (resetRate=0 ⇒
  //     no teleports); each matched particle must land on its sample's
  //     scratch pos_K (torus, 0.05 px — fp32 op-order drift over 6 steps)
  for (let s = 1; s <= K; s++) kernel.step(100 + s, ALPHA);
  const posFinal = await readF32(kernel.posBuffer, 2 * N);
  let maxD = 0;
  for (let s = 0; s < BATCH; s++) {
    const idx = sampleIdx[s];
    if (idx < 0) continue;
    const base = s * sl.sampleStride + sl.posOff + 2 * K;
    maxD = Math.max(
      maxD,
      torusD(posFinal[2 * idx], scratch[base], PHYS.width),
      torusD(posFinal[2 * idx + 1], scratch[base + 1], PHYS.height)
    );
  }
  const ok3 = matched === BATCH && maxD < 0.05;
  if (!ok3) failures++;
  console.log(
    `${ok3 ? "PASS" : "FAIL"}  window equivalence: imagined rollout pos_K == real ` +
    `advected position after K=${K} frames (maxΔ=${maxD.toExponential(2)} px, tol 0.05)`
  );
}

trainer.destroy();
kernel.destroy();
field.dispose();

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
