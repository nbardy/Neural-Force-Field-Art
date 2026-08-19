/**
 * FUSED ADVERSARY ON A HASHGRID FIELD — parity vs LIVE tfjs autograd.
 * Real Metal via bun-webgpu. GPU suites are SEQUENTIAL — run nothing else
 * on the GPU.
 *
 *   bun tools/train_wta_hashgrid_test.ts
 *
 * WHY A SEPARATE FILE FROM tools/train_wta_test.ts: that suite's §2 oracle is
 * the AD IR (src/render/webgpu/ad/*), and the IR CANNOT express a hashgrid —
 * its gather indices are data-dependent, so `ad/rollout.ts` types `encoding`
 * as raw | fourier only. The only available oracle for the grid backward is a
 * LIVE tfjs graph, i.e. the §3/§3b pattern: a real `HelmholtzField`
 * (modelType "hashgrid", the same `gridInterp` oneHot·matMul the app's
 * `?train=tfjs` route runs) plus the real `Adversary`, differentiated with
 * `tf.variableGrads`.
 *
 * Sections (each falsifiable):
 *  §1 gates: a hashgrid field is ACCEPTED by the fused adversary; hashgrid +
 *     classes and sin predictor heads still throw.
 *  §2 codegen invariants: raw/fourier scratch offsets still COLLAPSE (dEncOff
 *     == fieldSiteOff → stride unchanged → every pre-existing generated shader
 *     is byte-identical) and only the hashgrid shader mentions dEnc.
 *  §3 parity sweep vs live tfjs over 4 configs (production default piece,
 *     a 4×4 grid where nearly every corner collides, a coincident-corner /
 *     clamp-border batch, and the post-velocity target). Asserts u/y, the
 *     DISCRIMINATOR gradient, and the generator extGrads — with the GRID slice
 *     and the MLP slice asserted SEPARATELY plus a support (which cells are
 *     touched) check, because a completely wrong grid block would otherwise
 *     hide inside an aggregate cosine dominated by whichever slice is larger.
 *  §4 lane isolation (the dEnc seeding hazard): fieldLane 0 AND 1. With one
 *     lane emitted only ONE field head's backward runs, so the head that runs
 *     must SEED the shared dEnc block — a head-index assumption would make
 *     lane 1 accumulate into never-written scratch.
 *  §4 the extGrad seam on a hashgrid layout, with a NONZERO field loss so the
 *     field's own grid block and the adversary's both run: grads(with) −
 *     grads(without) ≡ extGrads over ALL totalFloats, grid floats
 *     [0, gs²·F) included. This is the additive-vs-double-count verdict.
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
  advScratchLayout,
  adversaryPassAShader,
  type TupleTag,
} from "../src/render/webgpu/adversary_wgsl";
import { AdversaryTrainer } from "../src/render/webgpu/adversary_train";
import { FusedTrainer } from "../src/render/webgpu/train";
import { HelmholtzField } from "../src/core/field/helmholtz";
import {
  Adversary,
  defaultAdversaryConfig,
  disposeTupleSample,
} from "../src/core/gan/adversary";

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
const mkStorage = (bytes: number) =>
  device.createBuffer({
    size: Math.max(16, bytes),
    usage: 128 /*STORAGE*/ | 8 /*COPY_DST*/ | 4 /*COPY_SRC*/,
  });
async function readBuf(buf: any, floats: number): Promise<Float32Array> {
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

const PHYS = { width: 800, height: 600, forceMagnitude: 3.5, friction: 0.99, maxVelocity: 26 };
const ALPHA = 0.6;
const GEN_SEED = 1.234;
const HID = 8;

/** field head dims: (encDim)→HID→HID→2, selu·selu·tanh — matches
 *  HelmholtzField's makeVectorNet([HID, HID], encDim). */
function fieldDims(encDim: number): LayerDims[] {
  return [
    { inSize: encDim, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: 2, activation: "tanh" },
  ];
}

/** Packed field weights for a layout that MAY carry a grid segment (head=-1,
 *  layer=-1). The shared maker in train_wta_test.ts indexes heads[seg.head]
 *  and cannot see one. Grid values are deliberately O(0.3): a zero table is a
 *  degenerate field whose backward is trivially "finite". */
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
/** B tuples of m members, layout m × [px, py, vx, vy]. */
function makeTuples(B: number, m: number, seed: number): Float32Array {
  const rnd = mulberry32(seed);
  const t = new Float32Array(B * m * 4);
  for (let i = 0; i < B * m; i++) {
    t[i * 4] = Math.fround(rnd() * PHYS.width);
    t[i * 4 + 1] = Math.fround(rnd() * PHYS.height);
    t[i * 4 + 2] = Math.fround((rnd() * 2 - 1) * 35); // maxVel 26 → some clip
    t[i * 4 + 3] = Math.fround((rnd() * 2 - 1) * 35);
  }
  return t;
}
/**
 * The ONLY places the gather-side grid backward can diverge from tfjs's
 * oneHotᵀ scatter, made deliberate (m = 2):
 *   0 — both members at the SAME position: one cell claimed by two sites.
 *   1 — u ≥ 1 on BOTH axes: clamp(u,0,1) is exactly 1.0, so gxf = gs−1,
 *       fx = 0 and ix1 == ix — the SAME cell matches two corner tests and
 *       both must add (train_wgsl.ts documents this; tfjs sums the coincident
 *       scatters). This is also the du clip-mask branch.
 *   2 — u ≤ 0 on both axes: the other clamp border, cell 0.
 *   3 — one axis clamped high, the other interior (mixed corner geometry).
 *   4 — two members in the same cell but not the same point.
 *   5+ — random interior, so the batch is not degenerate.
 */
function makeCornerTuples(B: number, seed: number): Float32Array {
  const rnd = mulberry32(seed);
  const W = PHYS.width, H = PHYS.height;
  const t = new Float32Array(B * 2 * 4);
  const put = (tuple: number, member: number, px: number, py: number) => {
    const o = (tuple * 2 + member) * 4;
    t[o] = Math.fround(px);
    t[o + 1] = Math.fround(py);
    t[o + 2] = Math.fround((rnd() * 2 - 1) * 35);
    t[o + 3] = Math.fround((rnd() * 2 - 1) * 35);
  };
  for (let s = 0; s < B; s++) {
    switch (s % 6) {
      case 0: put(s, 0, 0.37 * W, 0.61 * H); put(s, 1, 0.37 * W, 0.61 * H); break;
      case 1: put(s, 0, 1.25 * W, 1.4 * H);  put(s, 1, 1.02 * W, 1.9 * H);  break;
      case 2: put(s, 0, -0.3 * W, -0.2 * H); put(s, 1, -0.9 * W, -1.1 * H); break;
      case 3: put(s, 0, 1.3 * W, 0.44 * H);  put(s, 1, 0.22 * W, -0.5 * H); break;
      case 4: put(s, 0, 0.501 * W, 0.501 * H); put(s, 1, 0.502 * W, 0.502 * H); break;
      default: put(s, 0, rnd() * W, rnd() * H); put(s, 1, rnd() * W, rnd() * H);
    }
  }
  return t;
}

/** Compare two float vectors the way the rest of the suite does. */
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

/* ══════════════════════════════════════════════════════════════════════════
   §1 fusion gates
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 1. fusion gates ---");
{
  const enc = { kind: "hashgrid" as const, gridSize: 8, features: 4 };
  const layout = layoutField("helmholtz", [fieldDims(4), fieldDims(4)], { encoding: enc });
  const buf = mkStorage(layout.totalFloats * 4);
  let built: AdversaryTrainer | null = null;
  let err = "";
  try {
    built = new AdversaryTrainer(device, layout, {
      tag: "pair", k: 3, relaxEps: 0.05, observerGeometry: "periodic",
      hiddenUnits: HID, featureDim: HID, batchCap: 64,
      fieldWeightsBuffer: buf, seed: 1,
    });
  } catch (e) { err = (e as Error).message; }
  ok(built !== null, `hashgrid field is ACCEPTED by the fused adversary${err ? ` (got: ${err})` : ""}`);
  built?.destroy();

  // Families on a hashgrid ARE supported now, but ONLY through stacked feature
  // planes (`planes: C` — see tools/family_grid_test.ts). Asking for classes
  // without the planes to hold them must still be refused end to end, because
  // the alternative is silently indexing plane 0 for every family.
  throws(
    () => layoutField("helmholtz", [fieldDims(4), fieldDims(4)], { encoding: enc, classes: 3 }),
    "hashgrid + classes without matching planes still throws"
  );
  // sin predictor heads still refused (no pre-act checkpoint in adv scratch)
  throws(
    () => adversaryPassAShader(
      layout,
      layoutAdversary(2, [
        { inSize: 1, outSize: 4, activation: "sin" },
        { inSize: 4, outSize: 4, activation: "selu" },
        { inSize: 4, outSize: 2, activation: "linear" },
      ], { du: 1, dy: 2 }),
      { tag: "pair", relaxEps: 0.05, observerGeometry: "periodic" }
    ),
    "sin predictor activation still throws on a hashgrid field"
  );
  buf.destroy();
}

/* ══════════════════════════════════════════════════════════════════════════
   §2 codegen invariants — raw/fourier are untouched
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 2. codegen invariants (raw/fourier unchanged) ---");
{
  const advL = layoutAdversary(3, [
    { inSize: 1, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: 2, activation: "linear" },
  ], { du: 1, dy: 2 });
  const cases = [
    { label: "raw", enc: { kind: "raw" as const }, dim: 2 },
    { label: "fourier", enc: { kind: "fourier" as const, octaves: 4 }, dim: 18 },
  ];
  for (const c of cases) {
    const layout = layoutField("helmholtz", [fieldDims(c.dim), fieldDims(c.dim)], { encoding: c.enc });
    const sl = advScratchLayout(layout, advL, "pair");
    ok(
      sl.dEncOff === sl.fieldSiteOff,
      `${c.label}: dEnc block is EMPTY — offsets collapse, stride unchanged (${sl.stride} floats)`
    );
    const wgsl = adversaryPassAShader(layout, advL, {
      tag: "pair", relaxEps: 0.05, observerGeometry: "periodic",
    });
    // `dEnc[i]` locals exist on fourier (its jacobian consumes them); what
    // must NOT exist is a dEnc SCRATCH block — that is the hashgrid-only
    // addition, and its absence is what keeps these shaders byte-identical.
    ok(!/dEncBase/.test(wgsl), `${c.label}: generated pass A has no dEnc scratch block`);
  }
  const layout = layoutField("helmholtz", [fieldDims(4), fieldDims(4)], {
    encoding: { kind: "hashgrid", gridSize: 8, features: 4 },
  });
  const sl = advScratchLayout(layout, advL, "pair");
  ok(
    sl.fieldSiteOff - sl.dEncOff === sl.m * 4,
    `hashgrid: dEnc block is m·encDim = ${sl.m * 4} floats`
  );
  const wgsl = adversaryPassAShader(layout, advL, {
    tag: "pair", relaxEps: 0.05, observerGeometry: "periodic",
  });
  ok(/dEncBase/.test(wgsl), "hashgrid: generated pass A stores dL/dEnc");
}

/* ══════════════════════════════════════════════════════════════════════════
   §3 parity vs LIVE tfjs (HelmholtzField hashgrid + real Adversary)
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 3. hashgrid parity vs live tfjs autograd ---");

interface Case {
  label: string;
  gridSize: number;
  features: number;
  tag: TupleTag;
  encTag: any;
  k: number;
  target: { tag: "force" } | { tag: "post-velocity" };
  loss: any;
  batch: (B: number, m: number, seed: number) => Float32Array;
  fieldLane: "blend" | 0 | 1;
  seed: number;
}

const B = 64;
const CASES: Case[] = [
  {
    // The DEFAULT gallery piece's exact game/arch pairing.
    label: "default piece (pair-rot-scale-adj, soft-angle, 32²×4)",
    gridSize: 32, features: 4,
    tag: "pair-rotation-scale-adjusted",
    encTag: { tag: "pair-rotation-scale-adjusted" },
    k: 4, target: { tag: "force" },
    loss: { tag: "soft-angle", tau: 0.05 },
    batch: makeTuples, fieldLane: "blend", seed: 8100,
  },
  {
    // 4×4 grid, 128 sites: nearly every cell is claimed by many sites and
    // corner collisions are the norm, not the exception.
    label: "dense 4²×3 grid (pair, raw-vector)",
    gridSize: 4, features: 3,
    tag: "pair", encTag: { tag: "pair" },
    k: 3, target: { tag: "force" },
    loss: { tag: "raw-vector" },
    batch: makeTuples, fieldLane: "blend", seed: 8200,
  },
  {
    label: "coincident corners + clamp border (8²×4)",
    gridSize: 8, features: 4,
    tag: "pair", encTag: { tag: "pair" },
    k: 3, target: { tag: "force" },
    loss: { tag: "raw-vector" },
    batch: (b, _m, s) => makeCornerTuples(b, s),
    fieldLane: "blend", seed: 8300,
  },
  {
    label: "post-velocity target (point, soft-angle, 16²×4)",
    gridSize: 16, features: 4,
    tag: "point", encTag: { tag: "point" },
    k: 3, target: { tag: "post-velocity" },
    loss: { tag: "soft-angle", tau: 0.05 },
    batch: makeTuples, fieldLane: "blend", seed: 8400,
  },
  {
    // THE dEnc SEEDING HAZARD. One lane ⇒ one field head backward ⇒ that head
    // must SEED the shared dEnc block. Lane 1 is the failing case if the
    // emitter keeps the "head 0 seeds" assumption.
    label: "fieldLane 0 (8²×4)",
    gridSize: 8, features: 4,
    tag: "pair", encTag: { tag: "pair" },
    k: 3, target: { tag: "force" },
    loss: { tag: "raw-vector" },
    batch: makeTuples, fieldLane: 0, seed: 8500,
  },
  {
    label: "fieldLane 1 (8²×4)",
    gridSize: 8, features: 4,
    tag: "pair", encTag: { tag: "pair" },
    k: 3, target: { tag: "force" },
    loss: { tag: "raw-vector" },
    batch: makeTuples, fieldLane: 1, seed: 8600,
  },
];

for (const tc of CASES) {
  const enc = { kind: "hashgrid" as const, gridSize: tc.gridSize, features: tc.features };
  const F = tc.features;
  const layout = layoutField("helmholtz", [fieldDims(F), fieldDims(F)], { encoding: enc });
  const gridSeg = layout.segments.find((s) => s.role === "grid")!;
  const packed = makeFieldWeights(layout, tc.seed);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, packed);

  const trainer = new AdversaryTrainer(device, layout, {
    tag: tc.tag, target: tc.target, loss: tc.loss,
    k: tc.k, relaxEps: 0.05, observerGeometry: "periodic",
    fieldLane: tc.fieldLane,
    hiddenUnits: HID, featureDim: HID, batchCap: 256,
    fieldWeightsBuffer: fieldWBuf, seed: tc.seed + 1,
  });
  const m = (trainer as any).m as number;
  const aw = makeAdvWeights(trainer.advL, tc.seed + 2);
  trainer.uploadAdvWeights(aw);
  const tuples = tc.batch(B, m, tc.seed + 3);
  trainer.uploadTuples(tuples);
  trainer.step(PHYS, {
    b: B, alpha: ALPHA, lr: 1e-3, source: "uploaded",
    genSeed: GEN_SEED, applyDisc: false,
  });
  const sl = advScratchLayout(layout, trainer.advL, trainer.tag, trainer.target, trainer.loss);
  const [scratch, gpuAdv, gpuExt] = await Promise.all([
    readBuf((trainer as any).scratchBuf, B * sl.stride),
    trainer.readAdvGrads(),
    trainer.readExtGrads(),
  ]);

  // ---- the tfjs oracle: the REAL field class, hashgrid path ---------------
  const field = new HelmholtzField({
    alpha: ALPHA, hiddenUnits: [HID, HID], modelType: "hashgrid",
    gridSize: tc.gridSize, gridFeatures: F,
  });
  // trainableWeights order === layout.segments order (grid first, then head 0
  // kernel/bias per layer, then head 1) — asserted, not assumed.
  ok(
    field.trainableWeights.length === layout.segments.length,
    `${tc.label}: tfjs var count ≡ packed segment count (${layout.segments.length})`
  );
  field.trainableWeights.forEach((v, i) => {
    const seg = layout.segments[i];
    const count = v.shape.reduce((a, b) => a * b, 1);
    if (count !== seg.floatLength) {
      throw new Error(
        `${tc.label}: var ${i} has ${count} floats, segment has ${seg.floatLength}`
      );
    }
    v.assign(tf.tensor(
      Array.from(packed.slice(seg.floatOffset, seg.floatOffset + seg.floatLength)),
      v.shape
    ));
  });

  const adv = new Adversary({
    ...defaultAdversaryConfig(
      { tag: "wta", k: tc.k, relaxEps: 0.05 }, tc.encTag, "periodic"
    ),
    target: tc.target,
    loss: tc.loss,
    hiddenUnits: HID,
    featureDim: HID,
    batchTuples: B,
    seed: tc.seed + 4,
  });
  const advHeads = (adv as unknown as { heads: tf.Sequential[] }).heads;
  advHeads.forEach((net, j) => {
    const ws: tf.Tensor[] = [];
    trainer.advL.heads[j].layers.forEach((L) => {
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

  /** the field signal the fused shader computes for this lane */
  const signalOf = (): tf.Tensor2D => {
    if (tc.fieldLane === "blend") return field.forces(posT);
    const [g, r] = field.headForces(posT);
    const keep = tc.fieldLane === 0 ? g : r;
    const drop = tc.fieldLane === 0 ? r : g;
    drop.dispose();
    return keep;
  };
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

  const fieldSignal = signalOf();
  const sample = sampleOf(fieldSignal);

  // (a) context/target parity — proves the hashgrid FORWARD lines up first,
  //     so a gradient mismatch below cannot be blamed on the forward.
  {
    const uTf = sample.u.dataSync();
    const yTf = sample.y.dataSync();
    const scaleTf = sample.relativeScale?.dataSync();
    let maxU = 0, maxY = 0;
    for (let t = 0; t < B; t++) {
      const base = t * sl.stride;
      for (let i = 0; i < sl.du; i++) {
        maxU = Math.max(maxU, Math.abs(uTf[t * sl.du + i] - scratch[base + sl.uOff + i]));
      }
      for (let i = 0; i < sl.vectorDy; i++) {
        maxY = Math.max(maxY, Math.abs(yTf[t * sl.vectorDy + i] - scratch[base + sl.yOff + i]));
      }
      for (let q = 0; q < sl.scaleDy; q++) {
        maxY = Math.max(maxY, Math.abs(
          scaleTf![t * sl.scaleDy + q] - scratch[base + sl.yOff + sl.vectorDy + q]
        ));
      }
    }
    ok(maxU < 2e-5 && maxY < 2e-5,
      `${tc.label}: hashgrid forward → u/y parity (|Δu| ${maxU.toExponential(2)}, |Δy| ${maxY.toExponential(2)})`);
  }

  // (b) DISCRIMINATOR gradient. Structurally independent of the field
  //     encoding — which is exactly why it is worth asserting: a wrong
  //     scratch stride would silently corrupt the advOff reads.
  {
    const advVars = advHeads.flatMap((n) =>
      n.trainableWeights.map((wv) => (wv as unknown as { val: tf.Variable }).val)
    );
    const dRun = tf.variableGrads(() => adv.payoff(sample).mean().asScalar(), advVars);
    const flat: number[] = [];
    advVars.forEach((v) => {
      const a = dRun.grads[v.name].dataSync();
      for (let i = 0; i < a.length; i++) flat.push(a[i]);
    });
    const ref = new Float32Array(trainer.advL.totalFloats);
    let cur = 0;
    for (const seg of trainer.advL.segments) {
      for (let i = 0; i < seg.floatLength; i++) ref[seg.floatOffset + i] = flat[cur++];
    }
    const r = compare(ref, gpuAdv);
    ok(r.cos > 0.99999 && r.rel < 2e-3,
      `${tc.label}: D gradient ≡ tfjs (cos ${r.cos.toFixed(7)}, scale-rel ${r.rel.toExponential(2)})`);
    dRun.value.dispose();
    Object.values(dRun.grads).forEach((t) => t.dispose());
  }

  // (c) GENERATOR reward through the hashgrid — whole vector, then the GRID
  //     slice and the MLP slice SEPARATELY.
  {
    const fieldVars =
      tc.fieldLane === "blend"
        ? field.trainableWeights
        : // one lane ⇒ the other head is not in the graph and tfjs would throw
          [
            field.trainableWeights[0],
            ...field.trainableWeights.slice(
              1 + (tc.fieldLane === 0 ? 0 : 6),
              1 + (tc.fieldLane === 0 ? 6 : 12)
            ),
          ];
    const gRun = tf.variableGrads(() => tf.tidy(() => {
      const force = signalOf();
      const s = sampleOf(force);
      return adv.generatorLoss(s).mul(GEN_SEED * B).asScalar();
    }), fieldVars);
    // map tfjs vars back onto packed offsets
    const ref = new Float32Array(layout.totalFloats);
    fieldVars.forEach((v) => {
      const i = field.trainableWeights.indexOf(v);
      const seg = layout.segments[i];
      const a = gRun.grads[v.name].dataSync();
      for (let x = 0; x < seg.floatLength; x++) ref[seg.floatOffset + x] = a[x];
    });

    const gridLo = gridSeg.floatOffset;
    const gridHi = gridSeg.floatOffset + gridSeg.floatLength;
    const mlpIdx: number[] = [];
    for (const seg of layout.segments) {
      if (seg.role === "grid") continue;
      for (let x = 0; x < seg.floatLength; x++) mlpIdx.push(seg.floatOffset + x);
    }
    const all = compare(
      Array.from(layout.segments.flatMap((seg) =>
        Array.from({ length: seg.floatLength }, (_, x) => ref[seg.floatOffset + x]))),
      Array.from(layout.segments.flatMap((seg) =>
        Array.from({ length: seg.floatLength }, (_, x) => gpuExt[seg.floatOffset + x])))
    );
    const grid = compare(ref.subarray(gridLo, gridHi), gpuExt.subarray(gridLo, gridHi));
    const mlp = compare(mlpIdx.map((i) => ref[i]), mlpIdx.map((i) => gpuExt[i]));

    ok(all.cos > 0.99999 && all.rel < 3e-3,
      `${tc.label}: extGrads (all ${all.nzRef ? "" : ""}${gridSeg.floatLength + mlpIdx.length} floats) ≡ tfjs ` +
        `(cos ${all.cos.toFixed(7)}, scale-rel ${all.rel.toExponential(2)})`);
    ok(grid.nzRef > 0 && grid.cos > 0.99999 && grid.rel < 3e-3,
      `${tc.label}:   ↳ GRID slice [${gridLo},${gridHi}) ≡ tfjs ` +
        `(cos ${grid.cos.toFixed(7)}, scale-rel ${grid.rel.toExponential(2)}, ` +
        `${grid.nzRef} nonzero of ${gridSeg.floatLength})`);
    ok(grid.support === 0,
      `${tc.label}:   ↳ GRID support ≡ tfjs — the SAME cells are touched ` +
        `(${grid.nzRef} tfjs vs ${grid.nzGot} kernel, ${grid.support} mismatches)`);
    ok(mlp.nzRef > 0 && mlp.cos > 0.99999 && mlp.rel < 3e-3,
      `${tc.label}:   ↳ MLP slice (${mlpIdx.length} floats) ≡ tfjs ` +
        `(cos ${mlp.cos.toFixed(7)}, scale-rel ${mlp.rel.toExponential(2)})`);

    if (tc.fieldLane !== "blend") {
      // Lane isolation: the OTHER head's packed floats must be exact zero.
      const off = tc.fieldLane === 0 ? 7 : 1;
      let worst = 0;
      for (let i = off; i < off + 6; i++) {
        const seg = layout.segments[i];
        for (let x = 0; x < seg.floatLength; x++) {
          worst = Math.max(worst, Math.abs(gpuExt[seg.floatOffset + x]));
        }
      }
      ok(worst === 0,
        `${tc.label}:   ↳ inactive field head extGrad is EXACT zero (max |g| ${worst})`);
    }
    gRun.value.dispose();
    Object.values(gRun.grads).forEach((t) => t.dispose());
  }

  disposeTupleSample(sample);
  fieldSignal.dispose();
  posT.dispose();
  velT.dispose();
  adv.dispose();
  field.dispose();
  trainer.destroy();
  fieldWBuf.destroy();
}

/* ══════════════════════════════════════════════════════════════════════════
   §4 the extGrad seam on a hashgrid layout (grid floats included)
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 4. extGrad seam over a hashgrid layout ---");
{
  const enc = { kind: "hashgrid" as const, gridSize: 8, features: 4 };
  const layout = layoutField("helmholtz", [fieldDims(4), fieldDims(4)], { encoding: enc });
  const gridSeg = layout.segments.find((s) => s.role === "grid")!;
  const packed = makeFieldWeights(layout, 9001);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, packed);

  const adv = new AdversaryTrainer(device, layout, {
    tag: "pair", k: 4, relaxEps: 0.05, hiddenUnits: 16, featureDim: HID,
    observerGeometry: "periodic", batchCap: 256,
    fieldWeightsBuffer: fieldWBuf, seed: 9002,
  });
  adv.uploadAdvWeights(makeAdvWeights(adv.advL, 9003));
  adv.uploadTuples(makeTuples(128, 2, 9004));

  // adversary-only training must leave the field buffer — grid table included
  // — bit-identical, exactly as on raw layouts.
  const before = await readBuf(fieldWBuf, layout.totalFloats);
  for (let s = 0; s < 5; s++) {
    adv.step(PHYS, { b: 128, alpha: ALPHA, lr: 3e-3, source: "uploaded", genSeed: 1.0, applyDisc: true });
  }
  const after = await readBuf(fieldWBuf, layout.totalFloats);
  let bitIdentical = true;
  for (let i = 0; i < layout.totalFloats; i++) if (before[i] !== after[i]) bitIdentical = false;
  ok(bitIdentical, "5 hashgrid adversary steps → field weights (grid table incl.) BIT-IDENTICAL");

  const batch = new Float32Array(2 * 128);
  {
    const rnd = mulberry32(9005);
    for (let i = 0; i < 128; i++) {
      batch[2 * i] = rnd() * PHYS.width;
      batch[2 * i + 1] = rnd() * PHYS.height;
    }
  }
  const withExt = new FusedTrainer(device, layout, {
    batchCap: 256, weightsBuffer: fieldWBuf, extGradBuffer: adv.extGradsBuf,
  });
  const noExt = new FusedTrainer(device, layout, { batchCap: 256 });
  noExt.uploadWeights(after);
  withExt.uploadBatch(batch);
  noExt.uploadBatch(batch);

  adv.step(PHYS, { b: 128, alpha: ALPHA, lr: 3e-3, source: "uploaded", genSeed: 0.7, applyDisc: false });
  const ext = await adv.readExtGrads();
  withExt.step(PHYS, { n: 128, alpha: ALPHA, lr: 0.01, source: "uploaded", apply: false });
  const gWith = await withExt.readGrads();
  noExt.step(PHYS, { n: 128, alpha: ALPHA, lr: 0.01, source: "uploaded", apply: false });
  const gWithout = await noExt.readGrads();
  const diff = Float32Array.from(gWith, (v, i) => v - gWithout[i]);
  const seam = compare(ext, diff);
  // ADDITIVE, NOT DOUBLE-COUNTED. Both FusedTrainers here run the DEFAULT
  // structural loss (train_wgsl LOSS: chaos+iso+div+spiral ≠ 0), so the
  // field's OWN hashgrid grid block is live in both — assert that first, or
  // the cancellation below proves nothing. The field's grid gradient comes
  // from its own scratch/batch/objective and the adversary's comes from the
  // adversary's; they are two different loss terms over the same parameters,
  // so pass B summing them is ∇(L_field + L_gen). If it were a double count,
  // the same term would appear twice and the difference below could not be
  // exactly extGrads.
  {
    let ownGrid = 0;
    for (let i = gridSeg.floatOffset; i < gridSeg.floatOffset + gridSeg.floatLength; i++) {
      ownGrid = Math.max(ownGrid, Math.abs(gWithout[i]));
    }
    ok(ownGrid > 0,
      `field's OWN structural grid gradient is live (max |g| ${ownGrid.toExponential(2)}) ` +
        `— the seam test below is not vacuous`);
  }
  ok(seam.cos > 0.9999 && seam.rel < 1e-3,
    `extGrad seam: grads(with) − grads(without) ≡ extGrads over all ${layout.totalFloats} floats ` +
      `(cos ${seam.cos.toFixed(6)}, scale-rel ${seam.rel.toExponential(2)})`);
  const gridSeam = compare(
    ext.subarray(gridSeg.floatOffset, gridSeg.floatOffset + gridSeg.floatLength),
    diff.subarray(gridSeg.floatOffset, gridSeg.floatOffset + gridSeg.floatLength)
  );
  ok(gridSeam.nzRef > 0 && gridSeam.cos > 0.9999 && gridSeam.rel < 1e-3,
    `  ↳ GRID slice reaches the field's Adam through the seam ` +
      `(${gridSeam.nzRef} nonzero, cos ${gridSeam.cos.toFixed(6)})`);

  adv.destroy();
  withExt.destroy();
  noExt.destroy();
  fieldWBuf.destroy();
}

console.log(
  failures === 0
    ? "\nALL HASHGRID ADVERSARY CHECKS PASS"
    : `\n${failures} CHECK(S) FAILED`
);
process.exit(failures === 0 ? 0 : 1);
