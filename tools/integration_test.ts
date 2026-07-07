/**
 * FULL-SEAM integration test: real tfjs (webgpu backend) + HelmholtzField +
 * AdvectKernel, headless under Bun on a real Metal adapter via bun-webgpu.
 *
 *   bun tools/integration_test.ts
 *
 * This exercises exactly what the browser runs and what tools/kernel_test.ts
 * cannot: the tfjs→kernel weight sync (dataToGPU + copyBufferToBuffer),
 * variable ordering (g head then r head, kernel/bias per layer), activation
 * introspection off live tf.layers, and the uniform/alpha plumbing.
 *
 * Checks:
 *   1. one kernel step from a known snapshot == float64 reference computed
 *      from the weights tfjs reports (dataSync) — proves packing + codegen
 *      agree with the LIVE model, not a hand-built spec
 *   2. after an Adam training step changes the weights, the next kernel step
 *      matches a reference using the NEW weights — proves per-frame sync
 *   3. resize grow/shrink preserves the surviving particles
 */
import { setupGlobals } from "bun-webgpu";
setupGlobals();

// Adapter shims for tfjs 4.10 under bun-webgpu (patched per-instance — there
// is no GPUAdapter global here):
//  1. tfjs calls adapter.requestAdapterInfo(), removed from the spec; Dawn
//     exposes .info (same shim as src/main.ts).
//  2. bun-webgpu's requestDevice validator rejects feature names it doesn't
//     know (e.g. 'bgra8unorm-storage', which tfjs requests when the adapter
//     advertises it) — retry-strip any feature the wrapper refuses to pack.
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
      // .flat(): tfjs 4.10 has a literal bug — it pushes ['bgra8unorm-storage']
      // (an array) INTO requiredFeatures. Browsers stringify-coerce the enum;
      // bun-webgpu's strict validator throws on it.
      const d = desc
        ? { ...desc, requiredFeatures: [...(desc.requiredFeatures ?? [])].flat(Infinity) }
        : desc;
      // bun-webgpu's limit packer also chokes on tfjs's requiredLimits map;
      // Dawn's device DEFAULTS (16KB workgroup storage, 128MB bindings) are
      // ample for these tests, so just drop the request.
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
// tfjs-backend-webgpu's registration gate is `typeof window !== 'undefined'
// && !!navigator.gpu` — satisfy it under Bun (navigator.gpu comes from
// setupGlobals above). Must happen AFTER tfjs-core import so core still
// detects the node platform, BEFORE the backend import which runs the gate.
(globalThis as any).window ??= globalThis;
// tfjs upload paths do `values instanceof GPUBuffer`; bun-webgpu has no such
// global class. A never-matching stand-in gives the correct answer (false)
// for CPU-side values.
(globalThis as any).GPUBuffer ??= class GPUBufferShim {};
(globalThis as any).GPUTexture ??= class GPUTextureShim {};
await import("@tensorflow/tfjs-backend-webgpu");
const { HelmholtzField } = await import("../src/core/field/helmholtz");
const { AdvectKernel } = await import("../src/render/webgpu/advect");

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

// ---- float64 reference of one advect step, weights straight from tfjs ----
const selu = (x: number) =>
  x > 0 ? 1.0507009873554805 * x : 1.7580993408473768 * (Math.exp(x) - 1);

interface SnapLayer { K: Float32Array; B: Float32Array; inS: number; outS: number; act: string; }

/** Async-read a head's weights out of tfjs (webgpu readback) into plain arrays. */
async function snapshotHead(net: any): Promise<SnapLayer[]> {
  const out: SnapLayer[] = [];
  for (const layer of net.layers) {
    const [kT, bT] = layer.getWeights();
    out.push({
      K: (await kT.data()) as Float32Array,
      B: (await bT.data()) as Float32Array,
      inS: kT.shape[0],
      outS: kT.shape[1],
      act: String(layer.getConfig().activation),
    });
  }
  return out;
}

function refHeadEval(head: SnapLayer[], px: number, py: number): [number, number] {
  let act = [px, py];
  for (const L of head) {
    const out: number[] = new Array(L.outS);
    for (let j = 0; j < L.outS; j++) {
      let s = L.B[j];
      for (let i = 0; i < L.inS; i++) s += act[i] * L.K[i * L.outS + j];
      out[j] = L.act === "selu" ? selu(s) : L.act === "tanh" ? Math.tanh(s) : s;
    }
    act = out;
  }
  return [act[0], act[1]];
}

const PHYS = {
  width: 800, height: 600,
  forceMagnitude: 3.5, friction: 0.99, maxVelocity: 26, resetRate: 0,
};

async function refStep(
  field: any, alpha: number,
  pos: Float32Array, vel: Float32Array
): Promise<{ pos: Float64Array; vel: Float64Array }> {
  const [g, r] = await Promise.all(field.heads.map(snapshotHead));
  const n = pos.length / 2;
  const P = new Float64Array(pos);
  const V = new Float64Array(vel);
  const clip = (x: number) => Math.min(PHYS.maxVelocity, Math.max(-PHYS.maxVelocity, x));
  const fmod = (x: number, m: number) => x - Math.floor(x / m) * m;
  for (let i = 0; i < n; i++) {
    const px = P[2 * i] / PHYS.width;
    const py = P[2 * i + 1] / PHYS.height;
    const gv = refHeadEval(g, px, py);
    const rv = refHeadEval(r, px, py);
    const fx = ((1 - alpha) * gv[0] + alpha * rv[0]) * PHYS.forceMagnitude;
    const fy = ((1 - alpha) * gv[1] + alpha * rv[1]) * PHYS.forceMagnitude;
    V[2 * i] = clip((V[2 * i] + fx) * PHYS.friction);
    V[2 * i + 1] = clip((V[2 * i + 1] + fy) * PHYS.friction);
    P[2 * i] = fmod(P[2 * i] + V[2 * i], PHYS.width);
    P[2 * i + 1] = fmod(P[2 * i + 1] + V[2 * i + 1], PHYS.height);
  }
  return { pos: P, vel: V };
}

function compare(
  label: string,
  gPos: Float32Array, gVel: Float32Array,
  ref: { pos: Float64Array; vel: Float64Array }
): void {
  let maxP = 0, maxV = 0;
  for (let i = 0; i < gPos.length; i++) {
    const m = i % 2 === 0 ? PHYS.width : PHYS.height;
    let dp = Math.abs(gPos[i] - ref.pos[i]);
    dp = Math.min(dp, m - dp); // torus seam
    maxP = Math.max(maxP, dp);
    maxV = Math.max(maxV, Math.abs(gVel[i] - ref.vel[i]));
  }
  const ok = maxP < 0.02 && maxV < 0.02;
  if (!ok) failures++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${label} (maxΔpos=${maxP.toExponential(2)}, maxΔvel=${maxV.toExponential(2)})`
  );
}

// --------------------------------------------------------------------------
const N = 500;
const field = new HelmholtzField({ alpha: 0.7 });
const kernel = AdvectKernel.fromField(field as any, PHYS, N);

// 1 — first step matches reference from the live tfjs weights
{
  const pos0 = await readF32(kernel.posBuffer, 2 * N);
  const vel0 = await readF32(kernel.velBuffer, 2 * N);
  const ref = await refStep(field, field.alpha, pos0, vel0);
  kernel.step(1, field.alpha);
  const gPos = await readF32(kernel.posBuffer, 2 * N);
  const gVel = await readF32(kernel.velBuffer, 2 * N);
  compare("initial weights → kernel step matches tfjs weights", gPos, gVel, ref);
}

// 2 — train (Adam step changes every variable), sync must pick it up
{
  const optimizer = tf.train.adam(0.05);
  optimizer.minimize(
    () =>
      tf.tidy(() => {
        const p = tf.randomUniform([64, 2], 0, 1) as any;
        return (field.forces(p) as any).square().mean().asScalar();
      }),
    false,
    field.trainableWeights
  );
  const pos0 = await readF32(kernel.posBuffer, 2 * N);
  const vel0 = await readF32(kernel.velBuffer, 2 * N);
  field.alpha = 0.3; // also exercise the live-alpha uniform
  const ref = await refStep(field, field.alpha, pos0, vel0);
  kernel.step(2, field.alpha);
  const gPos = await readF32(kernel.posBuffer, 2 * N);
  const gVel = await readF32(kernel.velBuffer, 2 * N);
  compare("post-Adam weights + alpha change → kernel matches", gPos, gVel, ref);
  optimizer.dispose();
}

// 3 — resize preserves survivors (grow appends randoms, shrink slices)
{
  const before = await readF32(kernel.posBuffer, 2 * N);
  kernel.setParticleCount(N * 2);
  const grown = await readF32(kernel.posBuffer, 2 * N * 2);
  let same = true;
  for (let i = 0; i < 2 * N; i++) if (grown[i] !== before[i]) same = false;
  let tailInBounds = true;
  for (let i = 2 * N; i < 4 * N; i += 2) {
    if (grown[i] < 0 || grown[i] >= PHYS.width) tailInBounds = false;
    if (grown[i + 1] < 0 || grown[i + 1] >= PHYS.height) tailInBounds = false;
  }
  kernel.setParticleCount(Math.floor(N / 2));
  const shrunk = await readF32(kernel.posBuffer, 2 * Math.floor(N / 2));
  let sameShrunk = true;
  for (let i = 0; i < shrunk.length; i++) if (shrunk[i] !== before[i]) sameShrunk = false;
  const ok = same && tailInBounds && sameShrunk && kernel.count === Math.floor(N / 2);
  if (!ok) failures++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  resize: grow preserves+appends in-bounds, shrink slices (count=${kernel.count})`
  );
}

kernel.destroy();

// 4 — the FUSED-TRAINING wiring main.ts uses: FusedTrainer co-owns the advect
// kernel's weights buffer (syncFromTfjs=false), Adam-updates it in place, and
// the advect step must then move particles with the TRAINED weights — which
// tfjs never saw. Reference is rebuilt from the packed buffer itself.
{
  const { FusedTrainer } = await import("../src/render/webgpu/train");
  const M = 200;
  const kernel2 = AdvectKernel.fromField(field as any, PHYS, M);
  const trainer = new FusedTrainer(device, kernel2.layout, {
    weightsBuffer: kernel2.weightsBuffer,
    batchCap: 256,
  });
  const w0 = kernel2.packCurrentWeights();
  trainer.uploadWeights(w0);
  kernel2.syncFromTfjs = false;

  const tPhys = {
    width: PHYS.width, height: PHYS.height,
    forceMagnitude: PHYS.forceMagnitude, friction: PHYS.friction,
    maxVelocity: PHYS.maxVelocity,
  };
  for (let s = 0; s < 5; s++) {
    trainer.step(tPhys, { n: 128, alpha: 0.5, lr: 0.01, seed: s });
  }
  const wT = await trainer.readWeights();
  let changed = 0;
  for (let i = 0; i < wT.length; i++) {
    if (!Number.isFinite(wT[i])) changed = -1e9;
    if (wT[i] !== w0[i]) changed++;
  }

  // reference advect step from the PACKED trained weights
  const spec: any = kernel2.layout.spec;
  const packedHead = (h: number): SnapLayer[] =>
    spec.heads[h].layers.map((L: any) => ({
      K: wT.slice(L.weightOffset, L.weightOffset + L.inSize * L.outSize),
      B: wT.slice(L.biasOffset, L.biasOffset + L.outSize),
      inS: L.inSize,
      outS: L.outSize,
      act: L.activation,
    }));
  const gh = packedHead(0);
  const rh = packedHead(1);
  const pos0 = await readF32(kernel2.posBuffer, 2 * M);
  const vel0 = await readF32(kernel2.velBuffer, 2 * M);
  const P = new Float64Array(pos0);
  const V = new Float64Array(vel0);
  const clip = (x: number) => Math.min(PHYS.maxVelocity, Math.max(-PHYS.maxVelocity, x));
  const fmod = (x: number, m: number) => x - Math.floor(x / m) * m;
  const A = 0.5;
  for (let i = 0; i < M; i++) {
    const px = P[2 * i] / PHYS.width;
    const py = P[2 * i + 1] / PHYS.height;
    const gv = refHeadEval(gh, px, py);
    const rv = refHeadEval(rh, px, py);
    const fxv = ((1 - A) * gv[0] + A * rv[0]) * PHYS.forceMagnitude;
    const fyv = ((1 - A) * gv[1] + A * rv[1]) * PHYS.forceMagnitude;
    V[2 * i] = clip((V[2 * i] + fxv) * PHYS.friction);
    V[2 * i + 1] = clip((V[2 * i + 1] + fyv) * PHYS.friction);
    P[2 * i] = fmod(P[2 * i] + V[2 * i], PHYS.width);
    P[2 * i + 1] = fmod(P[2 * i + 1] + V[2 * i + 1], PHYS.height);
  }
  kernel2.step(99, A);
  const gPos = await readF32(kernel2.posBuffer, 2 * M);
  const gVel = await readF32(kernel2.velBuffer, 2 * M);
  compare(`fused-train wiring: advect uses trained weights (${changed} floats changed)`, gPos, gVel, { pos: P, vel: V });
  if (changed < wT.length * 0.9) {
    failures++;
    console.log(`FAIL  training barely changed weights (${changed}/${wT.length})`);
  }
  trainer.destroy();
  kernel2.destroy();
}

field.dispose();

// 5 — class-aware field wiring smoke (numerics are pinned by kernel_test's
// classes=3 case + train_test's invariants; this covers the tfjs→kernel seam:
// ingestion of the 2+C-input r head, trainer co-ownership, advect stepping)
{
  const M = 300;
  const fieldC = new HelmholtzField({ alpha: 0.6, classes: 3 });
  const kC = AdvectKernel.fromField(fieldC as any, PHYS, M);
  const { FusedTrainer } = await import("../src/render/webgpu/train");
  const tC = new FusedTrainer(device, kC.layout, {
    weightsBuffer: kC.weightsBuffer,
    batchCap: 256,
  });
  tC.uploadWeights(kC.packCurrentWeights());
  kC.syncFromTfjs = false;
  tC.setParticleBuffers(kC.posBuffer, kC.velBuffer, kC.count);
  const tPhys = {
    width: PHYS.width, height: PHYS.height,
    forceMagnitude: PHYS.forceMagnitude, friction: PHYS.friction,
    maxVelocity: PHYS.maxVelocity,
  };
  for (let s = 0; s < 4; s++) {
    tC.step(tPhys, { n: 128, alpha: 0.6, lr: 0.01, seed: s, source: "particles" });
    kC.step(s, 0.6);
  }
  const pos = await readF32(kC.posBuffer, 2 * M);
  const vel = await readF32(kC.velBuffer, 2 * M);
  const lossC = (await tC.readLoss()).loss;
  let ok = Number.isFinite(lossC);
  let moved = 0;
  for (let i = 0; i < M; i++) {
    if (!Number.isFinite(pos[2 * i]) || pos[2 * i] < 0 || pos[2 * i] >= PHYS.width) ok = false;
    if (!Number.isFinite(pos[2 * i + 1]) || pos[2 * i + 1] < 0 || pos[2 * i + 1] >= PHYS.height) ok = false;
    if (Math.abs(vel[2 * i]) + Math.abs(vel[2 * i + 1]) > 1e-6) moved++;
  }
  if (moved < M * 0.9) ok = false;
  if (!ok) failures++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  classes=3 seam: trained+advected 4 frames (loss=${lossC.toFixed(4)}, ${moved}/${M} moving, in-bounds)`
  );
  tC.destroy();
  kC.destroy();
  fieldC.dispose();
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
