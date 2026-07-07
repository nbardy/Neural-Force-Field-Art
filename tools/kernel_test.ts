/**
 * Headless GPU verification for the fused advect kernel (src/render/webgpu/
 * advect_wgsl.ts), run on a REAL WebGPU adapter (Dawn/Metal) via bun-webgpu —
 * no browser needed. This beats the old "kernels are un-verifiable in this
 * sandbox" constraint from HANDOFF.md.
 *
 *   bun tools/kernel_test.ts            # correctness + 1M-particle bench
 *   BENCH_N=2000000 bun tools/kernel_test.ts
 *
 * What it checks (math/tolerance, real GPU — not mocks):
 *   1. helmholtz [32,32] (the shipped field), weights staged in workgroup mem
 *   2. helmholtz asymmetric [16,48] — guards the AlphaGOJS hardcoded-dim trap
 *   3. legacy mlp sigmoid [32,64] — the -0.5 recenter path
 *   4. deep mlp [64,128,128,64] — weights too big to stage → storage-read path
 *   5. resetRate=1 → every particle respawns in-bounds with zero velocity
 *   6. bench: fused step at 1M particles (per-frame submit pattern)
 *
 * Each numeric case iterates 3 in-place GPU steps and compares against an
 * independent float64 JS reference of the SAME semantics (main.ts
 * physicsForward: normalize → MLP → forceMag → friction → clip → floored wrap).
 */
import { setupGlobals } from "bun-webgpu";
import {
  layoutField,
  advectShader,
  WORKGROUP_SIZE,
  CLASS_SALT,
  type FieldLayout,
  type LayerDims,
  type Activation,
} from "../src/render/webgpu/advect_wgsl";

setupGlobals();

// --------------------------------------------------------------------------
// deterministic RNG (seeded — failures must reproduce)
// --------------------------------------------------------------------------
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --------------------------------------------------------------------------
// float64 reference (independent of the WGSL — same published semantics)
// --------------------------------------------------------------------------
function refAct(a: Activation, x: number): number {
  if (a === "selu")
    return x > 0 ? 1.0507009873554805 * x : 1.7580993408473768 * (Math.exp(x) - 1);
  if (a === "sigmoid") return 1 / (1 + Math.exp(-x));
  if (a === "tanh") return Math.tanh(x);
  if (a === "sin") return Math.sin(x); // SIREN activation
  return x;
}

// JS port of the WGSL pcg hash (class derivation must match bit-for-bit)
function pcgJS(v: number): number {
  const st = (Math.imul(v, 747796405) + 2891336453) >>> 0;
  const t = Math.imul(((st >>> (((st >>> 28) + 4) & 31)) ^ st) >>> 0, 277803737) >>> 0;
  return ((t >>> 22) ^ t) >>> 0;
}

function refHead(
  layout: FieldLayout,
  head: number,
  W: Float32Array,
  px: number,
  py: number,
  cls = 0
): [number, number] {
  const in0 = layout.spec.heads[head].layers[0].inSize;
  // encoded input: fourier γ(p) = [x,y, sin(ωk x),sin(ωk y),cos(ωk x),cos(ωk y)]
  // (same order as helmholtz.ts / the WGSL emitter); else raw [x,y].
  let act: number[];
  let encDim: number;
  if (layout.encoding.kind === "fourier") {
    act = [px, py];
    for (let k = 0; k < layout.encoding.octaves; k++) {
      const w = Math.pow(2, k) * 2 * Math.PI;
      act.push(Math.sin(w * px), Math.sin(w * py), Math.cos(w * px), Math.cos(w * py));
    }
    encDim = 2 + 4 * layout.encoding.octaves;
  } else {
    act = [px, py];
    encDim = 2;
  }
  for (let k = encDim; k < in0; k++) act.push(k - encDim === cls ? 1 : 0);
  for (const L of layout.spec.heads[head].layers) {
    const out: number[] = new Array(L.outSize);
    for (let j = 0; j < L.outSize; j++) {
      let s = W[L.biasOffset + j];
      for (let i = 0; i < L.inSize; i++)
        s += act[i] * W[L.weightOffset + i * L.outSize + j];
      out[j] = refAct(L.activation, s);
    }
    act = out;
  }
  return [act[0], act[1]];
}

function refForce(
  layout: FieldLayout,
  W: Float32Array,
  alpha: number,
  px: number,
  py: number,
  cls = 0
): [number, number] {
  if (layout.spec.kind === "helmholtz") {
    const g = refHead(layout, 0, W, px, py, cls);
    const r = refHead(layout, 1, W, px, py, cls);
    return [(1 - alpha) * g[0] + alpha * r[0], (1 - alpha) * g[1] + alpha * r[1]];
  }
  const m = refHead(layout, 0, W, px, py);
  return [m[0] - 0.5, m[1] - 0.5];
}

interface Phys {
  w: number; h: number;
  forceMag: number; friction: number; maxVel: number;
  alpha: number; resetRate: number;
}

function refStep(
  layout: FieldLayout,
  W: Float32Array,
  P: Phys,
  pos: Float64Array,
  vel: Float64Array
): void {
  const n = pos.length / 2;
  const clip = (x: number) => Math.min(P.maxVel, Math.max(-P.maxVel, x));
  const fmod = (x: number, m: number) => x - Math.floor(x / m) * m;
  for (let i = 0; i < n; i++) {
    const cls = layout.classes > 0 ? pcgJS((i ^ CLASS_SALT) >>> 0) % layout.classes : 0;
    const f = refForce(layout, W, P.alpha, pos[2 * i] / P.w, pos[2 * i + 1] / P.h, cls);
    const vx = clip((vel[2 * i] + f[0] * P.forceMag) * P.friction);
    const vy = clip((vel[2 * i + 1] + f[1] * P.forceMag) * P.friction);
    vel[2 * i] = vx;
    vel[2 * i + 1] = vy;
    pos[2 * i] = fmod(pos[2 * i] + vx, P.w);
    pos[2 * i + 1] = fmod(pos[2 * i + 1] + vy, P.h);
  }
}

// --------------------------------------------------------------------------
// GPU plumbing
// --------------------------------------------------------------------------
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
// Device features are creation-time-only: the f16 cases need "shader-f16"
// requested HERE (mirrors main.ts, which wraps tfjs's requestDevice call).
// Feature-detect so the suite still runs — minus f16 — on adapters without it.
const HAS_F16: boolean = adapter.features?.has?.("shader-f16") ?? false;
const device: any = await adapter.requestDevice(
  HAS_F16 ? { requiredFeatures: ["shader-f16"] } : undefined
);
const info = adapter.info ?? {};
console.log(
  `adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}` +
    `${HAS_F16 ? "" : "  (no shader-f16 — f16 cases will be SKIPPED)"}\n`
);

// bun-webgpu doesn't implement getCompilationInfo — WGSL compile errors
// surface through the validation error scope at pipeline creation instead.
const USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const MAP_READ_MODE = 1;

async function makePipeline(code: string): Promise<any> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const err = await device.popErrorScope();
  if (err) {
    console.error(code);
    throw new Error(`pipeline validation: ${err.message}`);
  }
  return pipeline;
}

function uniData(P: Phys, seed: number, count: number): ArrayBuffer {
  const buf = new ArrayBuffer(48);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = P.w; f[1] = P.h;
  f[2] = P.forceMag; f[3] = P.friction; f[4] = P.maxVel;
  f[5] = P.alpha; f[6] = P.resetRate;
  u[7] = seed >>> 0; u[8] = count;
  return buf;
}

interface GpuSim {
  pipeline: any;
  bind: any;
  uni: any;
  posBuf: any;
  velBuf: any;
  n: number;
}

/** Passed straight through to advectShader (precision included). */
type ShaderOpts = { stageWeights: boolean; unroll?: boolean; precision?: "f32" | "f16" };

async function makeSim(
  layout: FieldLayout,
  shaderOpts: ShaderOpts,
  W: Float32Array,
  pos: Float32Array,
  vel: Float32Array
): Promise<GpuSim> {
  const n = pos.length / 2;
  const pipeline = await makePipeline(advectShader(layout, shaderOpts));
  const uni = device.createBuffer({ size: 48, usage: USAGE.UNIFORM | USAGE.COPY_DST });
  const wBuf = device.createBuffer({ size: W.byteLength, usage: USAGE.STORAGE | USAGE.COPY_DST });
  const mk = (a: Float32Array) => {
    const b = device.createBuffer({
      size: a.byteLength,
      usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
    });
    device.queue.writeBuffer(b, 0, a);
    return b;
  };
  device.queue.writeBuffer(wBuf, 0, W);
  const posBuf = mk(pos);
  const velBuf = mk(vel);
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: posBuf } },
      { binding: 3, resource: { buffer: velBuf } },
    ],
  });
  return { pipeline, bind, uni, posBuf, velBuf, n };
}

function encodeStep(sim: GpuSim): void {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(sim.pipeline);
  pass.setBindGroup(0, sim.bind);
  pass.dispatchWorkgroups(Math.ceil(sim.n / WORKGROUP_SIZE));
  pass.end();
  device.queue.submit([enc.finish()]);
}

async function readback(buf: any, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: floats * 4,
    usage: USAGE.MAP_READ | USAGE.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(MAP_READ_MODE);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

// --------------------------------------------------------------------------
// test cases
// --------------------------------------------------------------------------
const HELM: Phys = { w: 800, h: 600, forceMag: 3.5, friction: 0.99, maxVel: 26, alpha: 0.7, resetRate: 0 };
let failures = 0;

function chain(hidden: number[], hiddenAct: Activation, outAct: Activation, in0 = 2): LayerDims[] {
  const widths = [in0, ...hidden, 2];
  return widths.slice(1).map((outSize, i) => ({
    inSize: widths[i],
    outSize,
    activation: i === widths.length - 2 ? outAct : hiddenAct,
  }));
}

async function numericCase(
  name: string,
  layout: FieldLayout,
  shaderOpts: ShaderOpts,
  P: Phys
): Promise<void> {
  const rnd = mulberry32(1234);
  const W = new Float32Array(layout.totalFloats);
  for (let i = 0; i < W.length; i++) W[i] = (rnd() - 0.5) * 1.2;
  const n = 1000; // non-multiple of 64 → exercises the tail guard
  const pos = new Float32Array(2 * n);
  const vel = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    pos[2 * i] = rnd() * P.w;
    pos[2 * i + 1] = rnd() * P.h;
    vel[2 * i] = (rnd() - 0.5) * 10;
    vel[2 * i + 1] = (rnd() - 0.5) * 10;
  }
  const refPos = Float64Array.from(pos);
  const refVel = Float64Array.from(vel);

  const sim = await makeSim(layout, shaderOpts, W, pos, vel);
  const STEPS = 3;
  device.queue.writeBuffer(sim.uni, 0, uniData(P, 0, n));
  for (let s = 0; s < STEPS; s++) {
    encodeStep(sim);
    refStep(layout, W, P, refPos, refVel);
  }
  const gPos = await readback(sim.posBuf, 2 * n);
  const gVel = await readback(sim.velBuf, 2 * n);

  let maxP = 0, maxV = 0;
  for (let i = 0; i < 2 * n; i++) {
    // wrap seam: 0 and w/h are the same point — compare on the torus
    const m = i % 2 === 0 ? P.w : P.h;
    let dp = Math.abs(gPos[i] - refPos[i]);
    dp = Math.min(dp, m - dp);
    maxP = Math.max(maxP, dp);
    maxV = Math.max(maxV, Math.abs(gVel[i] - refVel[i]));
  }
  // Tolerance derivation:
  //   f32 — 0.02 px: the original bound; also the REGRESSION GUARD that the
  //     f32 codegen stayed byte-identical after the f16 fast path landed (the
  //     printed maxΔ values must match the pre-f16 run exactly).
  //   f16 — weights/activations round to a 10-bit mantissa ⇒ ~5e-4 relative
  //     per value; through 3 layers of ~32-term MACs that compounds to
  //     ~0.3-1% relative force error. |force| ≲ 1 (tanh heads) × forceMag 3.5
  //     ⇒ per-step velocity error up to ~0.03 px, and velocity persists
  //     through friction≈1 so 3 steps stack to ~0.1 px in velocity and
  //     ~0.2 px in integrated position. Bounds: maxΔvel < 0.1, maxΔpos < 0.25
  //     (small headroom on pos for the worst-case particle). Values well
  //     beyond that (≳1 px) would mean a real bug (wrong row, overflow), not
  //     rounding.
  const [tolP, tolV] = shaderOpts.precision === "f16" ? [0.25, 0.1] : [0.02, 0.02];
  const ok = maxP < tolP && maxV < tolV;
  if (!ok) failures++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${name}  (${STEPS} steps, maxΔpos=${maxP.toExponential(2)}, maxΔvel=${maxV.toExponential(2)})`
  );
}

// 1 — the shipped Helmholtz field: two heads 2→32→32→2 (auto: unrolled+vec4)
await numericCase(
  "helmholtz [32,32] unrolled+staged ",
  layoutField("helmholtz", [chain([32, 32], "selu", "tanh"), chain([32, 32], "selu", "tanh")]),
  { stageWeights: true },
  HELM
);

// 1b — SIREN: sin hidden activations (the advect-forward half of the SIREN
//      field type; training runs through tfjs autograd, this verifies the
//      fused forward matches sin-net semantics).
await numericCase(
  "helmholtz SIREN sin [32,32]        ",
  layoutField("helmholtz", [chain([32, 32], "sin", "tanh"), chain([32, 32], "sin", "tanh")]),
  { stageWeights: true },
  HELM
);

// 1c — FOURIER encoding: layer-0 input is γ(p) (2 + 4·octaves = 18 for 4
//      octaves). Forces the looped emitter; the reference applies the same γ.
{
  const oct = 4;
  const inDim = 2 + 4 * oct;
  const fourierChain = (): LayerDims[] => [
    { inSize: inDim, outSize: 32, activation: "selu" },
    { inSize: 32, outSize: 32, activation: "selu" },
    { inSize: 32, outSize: 2, activation: "tanh" },
  ];
  await numericCase(
    "helmholtz FOURIER γ oct=4 [32,32] ",
    layoutField("helmholtz", [fourierChain(), fourierChain()], {
      encoding: { kind: "fourier", octaves: oct },
    }),
    { stageWeights: true },
    HELM
  );
}

// 2 — asymmetric + non-multiple-of-4 width: hardcoded-dim trap + the scalar
//     fallback inside the unrolled emitter (18 % 4 != 0)
await numericCase(
  "helmholtz [18,36] unrolled+staged ",
  layoutField("helmholtz", [chain([18, 36], "selu", "tanh"), chain([18, 36], "selu", "tanh")]),
  { stageWeights: true },
  HELM
);

// 2b — MULTI-SPECIES: classes=3, chaos head takes 2+3 inputs (order head
//      class-blind at 2). Reference derives class via the same pcg hash.
await numericCase(
  "helmholtz classes=3 [32,32]       ",
  layoutField(
    "helmholtz",
    [chain([32, 32], "selu", "tanh"), chain([32, 32], "selu", "tanh", 5)],
    { classes: 3 }
  ),
  { stageWeights: true },
  HELM
);

// 3 — legacy sigmoid MLP (mlpShallow shape) incl. the -0.5 recenter
await numericCase(
  "mlp [32,64] sigmoid unrolled      ",
  layoutField("mlp", [chain([32, 64], "selu", "sigmoid")]),
  { stageWeights: true },
  { ...HELM, alpha: 0, forceMag: 3.0, maxVel: 22, friction: 0.985 }
);

// 4 — mlpDeep shape: 33k MACs → auto-looped; 133KB ≫ 16KB workgroup limit →
//     storage-read path (looped+unstaged, the big-net configuration)
await numericCase(
  "mlp deep [64,128,128,64] looped   ",
  layoutField("mlp", [chain([64, 128, 128, 64], "selu", "sigmoid")]),
  { stageWeights: false },
  { ...HELM, alpha: 0 }
);

// 5 — force the LOOPED emitter on a small staged net (loop+staging combo,
//     plus a tail layer where outSize%4 != 0 inside the loop form)
await numericCase(
  "helmholtz [32,32] looped+staged   ",
  layoutField("helmholtz", [chain([32, 32], "selu", "tanh"), chain([32, 32], "selu", "tanh")]),
  { stageWeights: true, unroll: false },
  HELM
);

// 6 — f16 FAST PATH: same shipped field + the classes=3 multi-species shape,
//     weights staged as vec4<f16>, MACs/accumulation in f16 (activations and
//     physics f32). Compared against the SAME float64 reference as the f32
//     cases, with the wider f16 tolerance (derivation at the check above).
if (HAS_F16) {
  await numericCase(
    "helmholtz [32,32] f16 unrolled    ",
    layoutField("helmholtz", [chain([32, 32], "selu", "tanh"), chain([32, 32], "selu", "tanh")]),
    { stageWeights: true, precision: "f16" },
    HELM
  );
  await numericCase(
    "helmholtz classes=3 [32,32] f16   ",
    layoutField(
      "helmholtz",
      [chain([32, 32], "selu", "tanh"), chain([32, 32], "selu", "tanh", 5)],
      { classes: 3 }
    ),
    { stageWeights: true, precision: "f16" },
    HELM
  );
} else {
  console.log("SKIP  f16 cases — adapter has no shader-f16 feature");
}

// 5 — resetRate=1: every particle must respawn in-bounds with zero velocity
{
  const layout = layoutField("helmholtz", [
    chain([32, 32], "selu", "tanh"),
    chain([32, 32], "selu", "tanh"),
  ]);
  const rnd = mulberry32(99);
  const W = new Float32Array(layout.totalFloats);
  for (let i = 0; i < W.length; i++) W[i] = (rnd() - 0.5) * 1.2;
  const n = 1000;
  const pos = new Float32Array(2 * n).fill(123.456);
  const vel = new Float32Array(2 * n).fill(9.9);
  const sim = await makeSim(layout, { stageWeights: true }, W, pos, vel);
  device.queue.writeBuffer(sim.uni, 0, uniData({ ...HELM, resetRate: 1 }, 7, n));
  encodeStep(sim);
  const gPos = await readback(sim.posBuf, 2 * n);
  const gVel = await readback(sim.velBuf, 2 * n);
  let ok = true;
  let spreadX = new Set<number>();
  for (let i = 0; i < n; i++) {
    if (gPos[2 * i] < 0 || gPos[2 * i] >= HELM.w) ok = false;
    if (gPos[2 * i + 1] < 0 || gPos[2 * i + 1] >= HELM.h) ok = false;
    if (gVel[2 * i] !== 0 || gVel[2 * i + 1] !== 0) ok = false;
    spreadX.add(Math.floor(gPos[2 * i] / 100)); // respawns must actually spread out
  }
  if (spreadX.size < 4) ok = false;
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  reset: respawn in-bounds, vel=0, spread buckets=${spreadX.size}`);
}

// 7 — benchmark: fused advect at 1M particles, per-frame submit pattern.
//     f32 and f16 run back-to-back on identical initial state (fresh buffers
//     per run) so the two numbers are directly comparable.
{
  const N = Number(process.env.BENCH_N ?? 1_000_000);
  const layout = layoutField("helmholtz", [
    chain([32, 32], "selu", "tanh"),
    chain([32, 32], "selu", "tanh"),
  ]);
  const rnd = mulberry32(5);
  const W = new Float32Array(layout.totalFloats);
  for (let i = 0; i < W.length; i++) W[i] = (rnd() - 0.5) * 1.2;
  const pos = new Float32Array(2 * N);
  const vel = new Float32Array(2 * N);
  for (let i = 0; i < 2 * N; i += 2) {
    pos[i] = rnd() * 1920;
    pos[i + 1] = rnd() * 1080;
  }
  const P: Phys = { w: 1920, h: 1080, forceMag: 3.5, friction: 0.99, maxVel: 26, alpha: 0.7, resetRate: 0.01 };

  console.log("");
  const benchOne = async (precision: "f32" | "f16"): Promise<void> => {
    const sim = await makeSim(layout, { stageWeights: true, precision }, W, pos, vel);
    const WARM = 20, TIMED = 200;
    for (let s = 0; s < WARM; s++) {
      device.queue.writeBuffer(sim.uni, 0, uniData(P, s, N));
      encodeStep(sim);
    }
    await readback(sim.posBuf, 2); // settle
    const t0 = performance.now();
    for (let s = 0; s < TIMED; s++) {
      device.queue.writeBuffer(sim.uni, 0, uniData(P, WARM + s, N));
      encodeStep(sim);
    }
    await readback(sim.posBuf, 2); // fence
    const ms = (performance.now() - t0) / TIMED;
    console.log(
      `BENCH  fused advect ${precision} @ ${N.toLocaleString()} particles: ${ms.toFixed(3)} ms/step  (${(1000 / ms).toFixed(0)} steps/s)`
    );
    for (const b of [sim.posBuf, sim.velBuf]) b.destroy();
  };
  await benchOne("f32");
  if (HAS_F16) await benchOne("f16");
  else console.log("BENCH  f16 skipped — adapter has no shader-f16 feature");
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
