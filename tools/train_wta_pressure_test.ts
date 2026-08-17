/**
 * FUSED ANTI-COLLAPSE PRESSURE — parity vs the tfjs `directionOrderLoss`
 * oracle. Real Metal via bun-webgpu. GPU suites are SEQUENTIAL — run nothing
 * else on the GPU.
 *
 *   bun tools/train_wta_pressure_test.ts
 *
 * WHAT IS UNDER TEST. `FusedGamePressure` compiles a batch-coupled term into
 * adversary pass A: four direction moments are workgroup-reduced and finalized
 * by the PRE-D forward, then read back as constants by the post-D forward,
 * whose dL/dF is added into the SAME `dSig` the generator reward uses. The
 * claim this file gates is that the result is, to f32, the gradient of
 *
 *     L = polar·‖mean u‖² + nematic·‖mean(uₓ²−u_y², 2uₓu_y)‖²,
 *     u = F / sqrt(‖F‖² + τ²)
 *
 * over the B·m member field outputs — the exact function
 * `src/core/losses/isotropy.ts::directionOrderLoss` computes, differentiated
 * by `tf.variableGrads` through a real `HelmholtzField`.
 *
 * Sections (each falsifiable):
 *  §1 codegen: `none` emits ZERO pressure text (the byte-identity property);
 *     `anti-collapse` emits the moments; the stats ABI grows by exactly 4 only
 *     under pressure; τ ≤ 0 and negative weights throw.
 *  §2 parity sweep vs tfjs across raw AND hashgrid, point AND pair observers,
 *     including a POST-VELOCITY target — where sig is a velocity and the
 *     pressure must still read RAW F. Pressure isolated with genSeed = 0.
 *  §3 degenerate fields in closed form: F ≡ 0 exactly (unit_τ(0) must be
 *     finite and the gradient exactly 0, not NaN-gated to 0) and F ≡ const ≠ 0
 *     — the collapse this term exists to price, where R₁ = ‖F‖/√(‖F‖²+τ²) and
 *     R₂ = R₁² exactly.
 *  §4 superposition: extGrads(reward+pressure) ≡ extGrads(reward) +
 *     extGrads(pressure), and ≡ tfjs ∇(genSeed·B·generatorLoss + L).
 *  §5 telemetry: AdvStats.directionOrder ≡ directionOrderParameters on the
 *     same batch, and is `unmeasured` (not 0) with no pressure declared.
 */
import { setupGlobals } from "bun-webgpu";
import * as tf from "@tensorflow/tfjs";
import {
  layoutField,
  layoutAdversary,
  type FieldLayout,
  type AdversaryLayout,
  type LayerDims,
} from "../src/render/webgpu/advect_wgsl";
import {
  adversaryPassAShader,
  adversaryPassBShader,
  advStatsLayout,
  type TupleTag,
  type FusedGamePressure,
} from "../src/render/webgpu/adversary_wgsl";
import { AdversaryTrainer } from "../src/render/webgpu/adversary_train";
import { HelmholtzField } from "../src/core/field/helmholtz";
import {
  Adversary,
  defaultAdversaryConfig,
  disposeTupleSample,
} from "../src/core/gan/adversary";
import {
  directionOrderLoss,
  directionOrderParameters,
} from "../src/core/losses";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

await tf.setBackend("cpu");
await tf.ready();

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter");
  process.exit(1);
}
const device: any = await adapter.requestDevice();

let failures = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
  if (!cond) failures++;
};
const throws = (fn: () => unknown, msg: string) => {
  let threw = false;
  try { fn(); } catch (_) { threw = true; }
  ok(threw, msg);
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
function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-30);
}
function compare(ref: ArrayLike<number>, got: ArrayLike<number>) {
  const cos = cosine(ref, got);
  let scale = 1e-30, rel = 0, nzRef = 0, nzGot = 0, support = 0;
  for (let i = 0; i < ref.length; i++) scale = Math.max(scale, Math.abs(ref[i]));
  for (let i = 0; i < ref.length; i++) {
    rel = Math.max(rel, Math.abs(ref[i] - got[i]) / scale);
    const r = Math.abs(ref[i]) > 1e-9 * scale;
    const g = Math.abs(got[i]) > 1e-9 * scale;
    if (r) nzRef++;
    if (g) nzGot++;
    if (r !== g) support++;
  }
  return { cos, rel, scale, nzRef, nzGot, support };
}
const mkStorage = (bytes: number) =>
  device.createBuffer({
    size: Math.max(16, bytes),
    usage: 128 /*STORAGE*/ | 8 /*COPY_DST*/ | 4 /*COPY_SRC*/,
  });

const PHYS = { width: 800, height: 600, forceMagnitude: 3.5, friction: 0.99, maxVelocity: 26 };
const ALPHA = 0.6;
const HID = 8;
const B = 64;
const TAU = 0.05;
const POLAR = 0.05;
const NEMATIC = 0.05;
const PRESSURE: FusedGamePressure = {
  tag: "anti-collapse", polar: POLAR, nematic: NEMATIC, tau: TAU,
};

function fieldDims(encDim: number): LayerDims[] {
  return [
    { inSize: encDim, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: 2, activation: "tanh" },
  ];
}
function makeFieldWeights(layout: FieldLayout, seed: number): Float32Array {
  const rnd = mulberry32(seed);
  const packed = new Float32Array(layout.totalFloats);
  for (const seg of layout.segments) {
    const amp = seg.role === "grid" ? 0.35 : 0.6;
    for (let x = 0; x < seg.floatLength; x++) {
      packed[seg.floatOffset + x] = Math.fround((rnd() * 2 - 1) * amp);
    }
  }
  return packed;
}
function makeAdvWeights(advL: AdversaryLayout, seed: number): Float32Array {
  const rnd = mulberry32(seed);
  const packed = new Float32Array(advL.totalFloats);
  for (const seg of advL.segments) {
    for (let x = 0; x < seg.floatLength; x++) {
      packed[seg.floatOffset + x] = Math.fround((rnd() * 2 - 1) * 0.7);
    }
  }
  return packed;
}
function makeTuples(n: number, m: number, seed: number): Float32Array {
  const rnd = mulberry32(seed);
  const t = new Float32Array(n * m * 4);
  for (let i = 0; i < n * m; i++) {
    t[i * 4] = Math.fround(rnd() * PHYS.width);
    t[i * 4 + 1] = Math.fround(rnd() * PHYS.height);
    t[i * 4 + 2] = Math.fround((rnd() * 2 - 1) * 35);
    t[i * 4 + 3] = Math.fround((rnd() * 2 - 1) * 35);
  }
  return t;
}

/* ══════════════════════════════════════════════════════════════════════════
   §1 codegen — `none` is a ZERO-BYTE variant
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 1. pressure codegen ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)], {
    encoding: { kind: "raw" },
  });
  const advL = layoutAdversary(3, [
    { inSize: 1, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: 2, activation: "linear" },
  ], { du: 1, dy: 2 });
  const base = { tag: "pair" as TupleTag, relaxEps: 0.05, observerGeometry: "periodic" as const };

  const off = adversaryPassAShader(layout, advL, base);
  const offExplicit = adversaryPassAShader(layout, advL, {
    ...base, pressure: { tag: "none" },
  });
  const on = adversaryPassAShader(layout, advL, { ...base, pressure: PRESSURE });

  ok(off === offExplicit,
    "omitted pressure ≡ explicit {tag:'none'} — the default is the named variant");
  ok(!/statMom|fRaw|red4/.test(off),
    "pressure-off pass A contains NO moment code (byte-identity property)");
  ok(/statMom/.test(on) && /fRaw/.test(on) && /red4/.test(on),
    "pressure-on pass A reduces the four direction moments");
  ok(on.length > off.length,
    `pressure-on pass A is larger (${on.length} vs ${off.length} chars)`);

  // Pass B needs no pressure code at all — only its "genSeed == 0" comment
  // changes, because that claim stops being true once a pressure is declared.
  const bOff = adversaryPassBShader(layout, advL, base);
  const bOn = adversaryPassBShader(layout, advL, { ...base, pressure: PRESSURE });
  const strip = (s: string) => s.split("\n").filter((l) => !l.trim().startsWith("//")).join("\n");
  ok(strip(bOff) === strip(bOn),
    "pass B is IDENTICAL modulo comments — the pressure rides the existing dSig seam");

  // stats ABI: +4 finalized/partial slots, and ONLY under pressure.
  const sOff = advStatsLayout(4, { tag: "none" });
  const sOn = advStatsLayout(4, PRESSURE);
  ok(sOff.finalized === 11 && sOff.pstride === 11,
    `pressure-off stats ABI unchanged (finalized ${sOff.finalized}, stride ${sOff.pstride})`);
  ok(sOn.finalized === 15 && sOn.pstride === 15 && sOn.momentOff === 11,
    `pressure-on appends 4 moments at [${sOn.momentOff}, ${sOn.finalized})`);

  throws(() => adversaryPassAShader(layout, advL, {
    ...base, pressure: { tag: "anti-collapse", polar: 0.05, nematic: 0.05, tau: 0 },
  }), "tau = 0 throws (τ² IS the sqrt radicand floor)");
  throws(() => adversaryPassAShader(layout, advL, {
    ...base, pressure: { tag: "anti-collapse", polar: -1, nematic: 0.05, tau: TAU },
  }), "negative polar weight throws");
  throws(() => adversaryPassAShader(layout, advL, {
    ...base, pressure: { tag: "anti-collapse", polar: 0.05, nematic: 0.05, tau: NaN },
  }), "NaN tau throws");
}

/* ══════════════════════════════════════════════════════════════════════════
   §2/§3/§4/§5 — parity harness
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * What the case's weights force the order parameters to be, in closed form.
 *
 * `laminar` is deliberately NOT "R₁ = 1": with every u identical, R₁ = ‖u‖ =
 * ‖F‖/√(‖F‖²+τ²) < 1 and R₂ = ‖u‖² = R₁² exactly. That τ shortfall is the
 * softener doing its job (a near-zero force has no reliable direction), and
 * pinning the two closed forms is a stronger check than "close to 1".
 */
type OrderExpectation =
  | { readonly tag: "free" }
  | { readonly tag: "isotropic" }
  | { readonly tag: "laminar" };

interface Case {
  label: string;
  enc: { kind: "raw" } | { kind: "hashgrid"; gridSize: number; features: number };
  encDim: number;
  tag: TupleTag;
  encTag: any;
  k: number;
  target: { tag: "force" } | { tag: "post-velocity" };
  loss: any;
  seed: number;
  /** Rewrite the packed field weights to force a degenerate field. */
  degenerate?: (packed: Float32Array, layout: FieldLayout) => void;
  expect: OrderExpectation;
}

/** Zero BOTH heads' output layer (kernel + bias). tanh(0) = 0, so F ≡ 0
 *  exactly while every upstream weight still carries gradient. */
function zeroOutputLayer(packed: Float32Array, layout: FieldLayout): void {
  for (const seg of layout.segments) {
    if (seg.layer === 2) packed.fill(0, seg.floatOffset, seg.floatOffset + seg.floatLength);
  }
}
/** Zero the output KERNELS but set the output BIASES to one direction: the
 *  field becomes the constant tanh(b) ≠ 0 — perfectly laminar, R₁ = R₂ = 1. */
function constantField(packed: Float32Array, layout: FieldLayout): void {
  for (const seg of layout.segments) {
    if (seg.layer !== 2) continue;
    if (seg.role === "kernel") {
      packed.fill(0, seg.floatOffset, seg.floatOffset + seg.floatLength);
    } else {
      packed[seg.floatOffset] = 0.4;
      packed[seg.floatOffset + 1] = -0.9;
    }
  }
}

const CASES: Case[] = [
  {
    label: "raw · point · soft-angle",
    enc: { kind: "raw" }, encDim: 2,
    tag: "point", encTag: { tag: "point" }, k: 3,
    target: { tag: "force" }, loss: { tag: "soft-angle", tau: TAU }, seed: 7100,
    expect: { tag: "free" },
  },
  {
    label: "raw · pair-rot-scale-adj · soft-angle (the shipped game)",
    enc: { kind: "raw" }, encDim: 2,
    tag: "pair-rotation-scale-adjusted",
    encTag: { tag: "pair-rotation-scale-adjusted" }, k: 4,
    target: { tag: "force" }, loss: { tag: "soft-angle", tau: TAU }, seed: 7200,
    expect: { tag: "free" },
  },
  {
    label: "hashgrid 16²×4 · pair-rot-scale-adj (the DEFAULT piece)",
    enc: { kind: "hashgrid", gridSize: 16, features: 4 }, encDim: 4,
    tag: "pair-rotation-scale-adjusted",
    encTag: { tag: "pair-rotation-scale-adjusted" }, k: 4,
    target: { tag: "force" }, loss: { tag: "soft-angle", tau: TAU }, seed: 7300,
    expect: { tag: "free" },
  },
  {
    label: "hashgrid 8²×4 · point · soft-angle",
    enc: { kind: "hashgrid", gridSize: 8, features: 4 }, encDim: 4,
    tag: "point", encTag: { tag: "point" }, k: 3,
    target: { tag: "force" }, loss: { tag: "soft-angle", tau: TAU }, seed: 7400,
    expect: { tag: "free" },
  },
  {
    // sig is a VELOCITY here. The pressure must still read raw F — if it read
    // sig, this case would disagree with the oracle by the physics Jacobian.
    label: "raw · point · POST-VELOCITY target (pressure reads raw F, not sig)",
    enc: { kind: "raw" }, encDim: 2,
    tag: "point", encTag: { tag: "point" }, k: 3,
    target: { tag: "post-velocity" }, loss: { tag: "soft-angle", tau: TAU }, seed: 7500,
    expect: { tag: "free" },
  },
  {
    label: "raw · pair · F ≡ 0 EXACTLY (unit_τ(0) must be finite)",
    enc: { kind: "raw" }, encDim: 2,
    tag: "pair", encTag: { tag: "pair" }, k: 3,
    target: { tag: "force" }, loss: { tag: "raw-vector" }, seed: 7600,
    degenerate: zeroOutputLayer, expect: { tag: "isotropic" },
  },
  {
    label: "raw · pair · F ≡ const ≠ 0 (fully laminar: R₁ = ‖u‖, R₂ = R₁²)",
    enc: { kind: "raw" }, encDim: 2,
    tag: "pair", encTag: { tag: "pair" }, k: 3,
    target: { tag: "force" }, loss: { tag: "raw-vector" }, seed: 7700,
    degenerate: constantField, expect: { tag: "laminar" },
  },
  {
    label: "hashgrid 8²×4 · pair · F ≡ 0 EXACTLY",
    enc: { kind: "hashgrid", gridSize: 8, features: 4 }, encDim: 4,
    tag: "pair", encTag: { tag: "pair" }, k: 3,
    target: { tag: "force" }, loss: { tag: "raw-vector" }, seed: 7800,
    degenerate: zeroOutputLayer, expect: { tag: "isotropic" },
  },
];

const GEN_SEED = 1.234;

for (const tc of CASES) {
  console.log(`\n--- ${tc.label} ---`);
  const layout = layoutField(
    "helmholtz",
    [fieldDims(tc.encDim), fieldDims(tc.encDim)],
    { encoding: tc.enc }
  );
  const packed = makeFieldWeights(layout, tc.seed);
  tc.degenerate?.(packed, layout);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, packed);

  const common = {
    tag: tc.tag, target: tc.target, loss: tc.loss,
    k: tc.k, relaxEps: 0.05, observerGeometry: "periodic" as const,
    hiddenUnits: HID, featureDim: HID, batchCap: 256,
    fieldWeightsBuffer: fieldWBuf, seed: tc.seed + 1,
  };
  const pressed = new AdversaryTrainer(device, layout, { ...common, pressure: PRESSURE });
  const plain = new AdversaryTrainer(device, layout, { ...common });
  const m = pressed.m;
  const aw = makeAdvWeights(pressed.advL, tc.seed + 2);
  pressed.uploadAdvWeights(aw);
  plain.uploadAdvWeights(aw);
  const tuples = makeTuples(B, m, tc.seed + 3);
  pressed.uploadTuples(tuples);
  plain.uploadTuples(tuples);

  const runOpts = { b: B, alpha: ALPHA, lr: 1e-3, source: "uploaded" as const, applyDisc: false };
  // (i) pressure ALONE — genSeed 0 kills the reward, so extGrads is pure ∇L.
  pressed.step(PHYS, { ...runOpts, genSeed: 0 });
  const extPressureOnly = await pressed.readExtGrads();
  const statsPressureOnly = await pressed.readStats();
  // (ii) reward alone, from the pressure-free shader.
  plain.step(PHYS, { ...runOpts, genSeed: GEN_SEED });
  const extRewardOnly = await plain.readExtGrads();
  const statsPlain = await plain.readStats();
  // (iii) both, from the pressured shader.
  pressed.step(PHYS, { ...runOpts, genSeed: GEN_SEED });
  const extBoth = await pressed.readExtGrads();

  // ---- the tfjs oracle ----------------------------------------------------
  const field = new HelmholtzField(
    tc.enc.kind === "hashgrid"
      ? {
          alpha: ALPHA, hiddenUnits: [HID, HID], modelType: "hashgrid",
          gridSize: tc.enc.gridSize, gridFeatures: tc.enc.features,
        }
      : { alpha: ALPHA, hiddenUnits: [HID, HID] }
  );
  if (field.trainableWeights.length !== layout.segments.length) {
    throw new Error(
      `${tc.label}: tfjs var count ${field.trainableWeights.length} != segments ${layout.segments.length}`
    );
  }
  field.trainableWeights.forEach((v, i) => {
    const seg = layout.segments[i];
    v.assign(tf.tensor(
      Array.from(packed.slice(seg.floatOffset, seg.floatOffset + seg.floatLength)),
      v.shape
    ));
  });

  const adv = new Adversary({
    ...defaultAdversaryConfig({ tag: "wta", k: tc.k, relaxEps: 0.05 }, tc.encTag, "periodic"),
    target: tc.target, loss: tc.loss,
    hiddenUnits: HID, featureDim: HID, batchTuples: B, seed: tc.seed + 4,
  });
  const advHeads = (adv as unknown as { heads: tf.Sequential[] }).heads;
  advHeads.forEach((net, j) => {
    const ws: tf.Tensor[] = [];
    pressed.advL.heads[j].layers.forEach((L) => {
      ws.push(tf.tensor2d(
        Array.from(aw.slice(L.weightOffset, L.weightOffset + L.inSize * L.outSize)),
        [L.inSize, L.outSize]
      ));
      ws.push(tf.tensor1d(Array.from(aw.slice(L.biasOffset, L.biasOffset + L.outSize))));
    });
    net.setWeights(ws);
  });

  const N = B * m;
  const posArr = new Float32Array(N * 2);
  const velArr = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    posArr[2 * i] = tuples[4 * i] / PHYS.width;
    posArr[2 * i + 1] = tuples[4 * i + 1] / PHYS.height;
    velArr[2 * i] = tuples[4 * i + 2];
    velArr[2 * i + 1] = tuples[4 * i + 3];
  }
  const idx = new Int32Array(N);
  for (let i = 0; i < N; i++) idx[i] = i;
  const posT = tf.tensor2d(posArr, [N, 2]);
  const velT = tf.tensor2d(velArr, [N, 2]);

  const sampleOf = (force: tf.Tensor2D) =>
    tc.target.tag === "force"
      ? adv.encodeTarget({ tag: "force", pos: posT, force }, idx)
      : adv.encodeTarget({
          tag: "post-velocity", pos: posT,
          velocity: velT.div(PHYS.maxVelocity) as tf.Tensor2D,
          nextVelocity: velT
            .add(force.mul(PHYS.forceMagnitude))
            .mul(PHYS.friction)
            .clipByValue(-PHYS.maxVelocity, PHYS.maxVelocity)
            .div(PHYS.maxVelocity) as tf.Tensor2D,
        }, idx);

  const packRef = (grads: Record<string, tf.Tensor>): Float32Array => {
    const ref = new Float32Array(layout.totalFloats);
    field.trainableWeights.forEach((v, i) => {
      const seg = layout.segments[i];
      const a = grads[v.name].dataSync();
      for (let x = 0; x < seg.floatLength; x++) ref[seg.floatOffset + x] = a[x];
    });
    return ref;
  };

  // (a) PRESSURE-ONLY parity. The oracle is directionOrderLoss over ALL B·m
  //     member field outputs — no batch factor: the term's own 1/N is inside.
  {
    const run = tf.variableGrads(
      () => tf.tidy(() =>
        directionOrderLoss(field.forces(posT), TAU, POLAR, NEMATIC)
      ),
      field.trainableWeights
    );
    const ref = packRef(run.grads);
    const r = compare(ref, extPressureOnly);
    if (tc.expect.tag === "isotropic") {
      // At the exact isotropic point M₁ = M₂ = 0, so dL/du = 0 and the true
      // gradient is EXACTLY zero. The falsifiable content is that it is zero
      // and FINITE — a 0/0 in unit_τ would show up as NaN, and the isFiniteF
      // gate would then silently substitute the same 0. So assert the tfjs
      // oracle is finite too, and that no NaN reached the stats.
      let worst = 0, allFinite = true;
      for (let i = 0; i < extPressureOnly.length; i++) {
        worst = Math.max(worst, Math.abs(extPressureOnly[i]));
        if (!Number.isFinite(extPressureOnly[i])) allFinite = false;
      }
      let refFinite = true;
      for (let i = 0; i < ref.length; i++) if (!Number.isFinite(ref[i])) refFinite = false;
      ok(allFinite && refFinite && worst === 0,
        `  F ≡ 0: pressure grad is EXACT zero and finite on BOTH sides ` +
          `(max |g| ${worst}, tfjs finite ${refFinite})`);
      ok(Number.isFinite(run.value.dataSync()[0]) && run.value.dataSync()[0] === 0,
        `  F ≡ 0: oracle loss is exactly 0 (${run.value.dataSync()[0]})`);
    } else {
      ok(r.nzRef > 0 && r.cos > 0.9999 && r.rel < 3e-3,
        `  pressure-only extGrads ≡ ∇directionOrderLoss ` +
          `(cos ${r.cos.toFixed(7)}, scale-rel ${r.rel.toExponential(2)}, ` +
          `${r.nzRef}/${ref.length} nonzero, scale ${r.scale.toExponential(2)})`);
      ok(r.support === 0,
        `  pressure-only support ≡ tfjs (${r.support} mismatches)`);
    }
    run.value.dispose();
    Object.values(run.grads).forEach((t) => t.dispose());
  }

  // (b) TELEMETRY. The fused moments vs the tfjs order parameters, plus the
  //     closed form the degenerate cases pin exactly.
  {
    const forces = field.forces(posT);
    const { r1, r2 } = directionOrderParameters(forces, TAU);
    const f0 = forces.slice([0, 0], [1, 2]).dataSync();
    forces.dispose();
    const d = statsPressureOnly.directionOrder;
    if (d.tag !== "measured") {
      ok(false, "  directionOrder should be MEASURED under declared pressure");
    } else {
      ok(Math.abs(d.r1 - r1) < 2e-5 && Math.abs(d.r2 - r2) < 2e-5,
        `  R₁/R₂ ≡ tfjs (fused ${d.r1.toFixed(6)}/${d.r2.toFixed(6)}, ` +
          `tfjs ${r1.toFixed(6)}/${r2.toFixed(6)})`);
      switch (tc.expect.tag) {
        case "free":
          break;
        case "isotropic":
          ok(d.r1 === 0 && d.r2 === 0,
            `  F ≡ 0 ⇒ R₁ = R₂ = 0 EXACTLY (got ${d.r1}/${d.r2})`);
          break;
        case "laminar": {
          // Every u identical ⇒ R₁ = ‖u‖ and R₂ = ‖u‖². The τ softener is why
          // R₁ < 1: it is exactly ‖F‖/√(‖F‖²+τ²).
          const n = Math.hypot(f0[0], f0[1]);
          const uMag = n / Math.sqrt(n * n + TAU * TAU);
          ok(Math.abs(d.r1 - uMag) < 1e-5 && Math.abs(d.r2 - uMag * uMag) < 1e-5,
            `  F ≡ const ⇒ R₁ = ‖F‖/√(‖F‖²+τ²) = ${uMag.toFixed(6)} and R₂ = R₁² = ` +
              `${(uMag * uMag).toFixed(6)} (got ${d.r1.toFixed(6)}/${d.r2.toFixed(6)})`);
          break;
        }
        default: {
          const unhandled: never = tc.expect;
          throw new Error(`unhandled expectation ${JSON.stringify(unhandled)}`);
        }
      }
    }
    ok(statsPlain.directionOrder.tag === "unmeasured",
      `  the pressure-free trainer reports UNMEASURED, not 0 ` +
        `(${statsPlain.directionOrder.tag})`);
  }

  // (c) SUPERPOSITION. One extGrads buffer carries both objectives; they must
  //     add, because pass B assembles ∇(genSeed·reward + pressure) from one
  //     dSig. A wrong seam would show up as a scale error here even when each
  //     part is individually right.
  {
    const sum = Float32Array.from(extRewardOnly, (v, i) => v + extPressureOnly[i]);
    const r = compare(sum, extBoth);
    ok(r.nzRef > 0 && r.cos > 0.999999 && r.rel < 1e-4,
      `  extGrads(reward+pressure) ≡ extGrads(reward) + extGrads(pressure) ` +
        `(cos ${r.cos.toFixed(7)}, scale-rel ${r.rel.toExponential(2)})`);

    // …and the same sum against ONE tfjs tape over both objectives.
    const run = tf.variableGrads(
      () => tf.tidy(() => {
        const force = field.forces(posT);
        const s = sampleOf(force);
        const reward = adv.generatorLoss(s).mul(GEN_SEED * B).asScalar();
        const pressure = directionOrderLoss(force, TAU, POLAR, NEMATIC);
        return reward.add(pressure).asScalar();
      }),
      field.trainableWeights
    );
    const ref = packRef(run.grads);
    const rr = compare(ref, extBoth);
    ok(rr.nzRef > 0 && rr.cos > 0.9999 && rr.rel < 5e-3,
      `  extGrads(both) ≡ tfjs ∇(genSeed·B·L_gen + L_pressure) ` +
        `(cos ${rr.cos.toFixed(7)}, scale-rel ${rr.rel.toExponential(2)})`);
    run.value.dispose();
    Object.values(run.grads).forEach((t) => t.dispose());
  }

  // (d) The reward path itself must be untouched by the pressure codegen: the
  //     pressured shader with genSeed set and the plain shader must agree on
  //     the reward component (extBoth − extPressureOnly ≡ extRewardOnly).
  {
    const rewardFromPressed = Float32Array.from(extBoth, (v, i) => v - extPressureOnly[i]);
    const r = compare(extRewardOnly, rewardFromPressed);
    ok(r.nzRef > 0 && r.cos > 0.999999 && r.rel < 1e-4,
      `  the reward component is unchanged by compiling the pressure in ` +
        `(cos ${r.cos.toFixed(7)}, scale-rel ${r.rel.toExponential(2)})`);
  }

  posT.dispose();
  velT.dispose();
  adv.dispose();
  field.dispose();
  pressed.destroy();
  plain.destroy();
  fieldWBuf.destroy();
}

/* ══════════════════════════════════════════════════════════════════════════
   §6 stability — the pressured shader over many steps on live-ish data
   ══════════════════════════════════════════════════════════════════════════ */
// NOTE this section is a FINITENESS gate, not an effect measurement: the
// adversary binds the field weights READ-ONLY, so R₁ cannot move here however
// long it runs. The effect gate is tools/collapse_probe.ts, which closes the
// loop through a field optimizer.
console.log("\n--- 6. pressured steps stay finite ---");
{
  const enc = { kind: "hashgrid" as const, gridSize: 16, features: 4 };
  const layout = layoutField("helmholtz", [fieldDims(4), fieldDims(4)], { encoding: enc });
  const packed = makeFieldWeights(layout, 9101);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, packed);
  const trainer = new AdversaryTrainer(device, layout, {
    tag: "pair-rotation-scale-adjusted",
    loss: { tag: "soft-angle", tau: TAU },
    k: 4, relaxEps: 0.05, observerGeometry: "periodic",
    hiddenUnits: HID, featureDim: HID, batchCap: 256,
    fieldWeightsBuffer: fieldWBuf, seed: 9102, pressure: PRESSURE,
  });
  trainer.uploadAdvWeights(makeAdvWeights(trainer.advL, 9103));
  trainer.uploadTuples(makeTuples(128, 2, 9104));
  let finite = true;
  let lastR1 = NaN;
  for (let s = 0; s < 200; s++) {
    trainer.step(PHYS, {
      b: 128, alpha: ALPHA, lr: 3e-3, source: "uploaded", genSeed: 0.015 / 128,
      applyDisc: true,
    });
  }
  const st = await trainer.readStats();
  const ext = await trainer.readExtGrads();
  for (let i = 0; i < ext.length; i++) if (!Number.isFinite(ext[i])) finite = false;
  if (st.directionOrder.tag === "measured") lastR1 = st.directionOrder.r1;
  ok(finite && Number.isFinite(st.surprise) && Number.isFinite(lastR1),
    `200 pressured adversary steps: extGrads finite, payoff ${st.surprise.toFixed(4)}, ` +
      `R₁ ${lastR1.toFixed(4)}`);
  ok(lastR1 >= 0 && lastR1 <= 1.0001, `R₁ stays in [0,1] (${lastR1.toFixed(6)})`);
  trainer.destroy();
  fieldWBuf.destroy();
}

console.log(
  failures === 0
    ? "\nALL FUSED PRESSURE CHECKS PASS"
    : `\n${failures} CHECK(S) FAILED`
);
process.exit(failures === 0 ? 0 : 1);
