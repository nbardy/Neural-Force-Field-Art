/**
 * FUSED ADVERSARY verification — src/render/webgpu/adversary_{wgsl,train}.ts
 * vs the B1 IR oracle (ad/losses.ts::wtaTerm) and tfjs. Real Metal via
 * bun-webgpu. GPU suites are SEQUENTIAL — run nothing else on the GPU.
 *
 *   bun tools/train_wta_test.ts
 *
 * Sections (each falsifiable):
 *  §1 layout gates: layoutAdversary accepts pair(1→2)/tri(3→6) heads and
 *     throws on every malformed config; the FIELD constraints are untouched
 *     (a 3-wide field head still throws).
 *  §2 kernel ≡ IR oracle, per encoding × k (point/pair variants/tri, k=1..4, ε incl 0,
 *     raw + fourier fields): the oracle builds the ENTIRE per-tuple graph in
 *     the AD IR — raw field signal, minImage, the κ
 *     encodings (tri's stop-gradient sort included, via gt/eq products),
 *     wtaTerm — and is itself FD-spot-checked. Gates: u/y/surprise/winner
 *     parity, disc loss, raw/per-unit packed surprise planes, mean/closest
 *     head-spread stats (including adjusted predictor normalization),
 *     adversary weight grads (cos + scale-rel), generator FIELD weight grads
 *     (extGrads) (cos + scale-rel). K=1 uses an explicit sum type; K=16 proves
 *     the finalized stats prefix cannot overlap workgroup partials.
 *  §2c diagnostic semantics: target-inactive contexts still probe head spread,
 *     inactive per-unit values are exact zero, and a controlled 5× target
 *     scaling changes raw payoff 5× while leaving per-unit color invariant.
 *  §3 tfjs end-to-end generator cross-check (pair k=4): tf.variableGrads over
 *     the real field variables of −genSeed·B·mean(adv.payoff(encodeSignal(...)))
 *     with the REAL Adversary's heads — the full tfjs dataflow the fused
 *     kernel replaces — vs extGrads. Plus u/y parity vs Adversary.encode.
 *  §4 discriminator TRAINS fused: 30 steps on a fixed tuple batch strictly
 *     decrease the loss.
 *  §5 separation (structural, verified bit-exact): adversary-only training
 *     leaves field weights BIT-IDENTICAL; field-only training leaves
 *     adversary weights BIT-IDENTICAL; and the extGrad seam adds exactly
 *     dL_gen/dW into the field's pass-B gradient.
 *  §6 deterministic unique live coverage: effective-B caps, wrap, resize
 *     reset/clear, zero-data stale-gradient safety, uploaded-source isolation,
 *     and Agree+Disagree sampling lockstep.
 *  §7 BENCH at B=512 (pair k=4, tri k=6, labelled quad k=4; tfjs-default
 *     head sizes 32/16):
 *     the tfjs number to beat on this machine is 19-32 ms per learn step.
 */
import { setupGlobals } from "bun-webgpu";
import * as tf from "@tensorflow/tfjs";
import { Graph, type Node } from "../src/render/webgpu/ad/ir";
import { evalNodes, grad } from "../src/render/webgpu/ad/autodiff";
import { buildHead, type HeadDim } from "../src/render/webgpu/ad/head";
import { wtaTerm, awName, abName } from "../src/render/webgpu/ad/losses";
import {
  layoutField,
  layoutAdversary,
  type FieldLayout,
  type AdversaryLayout,
  type LayerDims,
  type Encoding,
} from "../src/render/webgpu/advect_wgsl";
import {
  advScratchLayout,
  tupleDims,
  ADV_SOFT_EPS2,
  ADV_DIRECTION_ACTIVE_FLOOR,
  ADV_QUAD_ANCHOR_ACTIVE_FLOOR,
  ADV_STATS_BASE,
  type TupleTag,
} from "../src/render/webgpu/adversary_wgsl";
import { AdversaryTrainer } from "../src/render/webgpu/adversary_train";
import { FusedTrainer } from "../src/render/webgpu/train";
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
  try {
    fn();
  } catch (_) {
    threw = true;
  }
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

function spreadFromScratch(
  scratch: Float32Array,
  sl: ReturnType<typeof advScratchLayout>,
  advL: AdversaryLayout,
  tuples: number
): { meanPair: number; minPair: number; pairs: number } | null {
  const k = advL.k;
  if (k === 1) return null;
  const pairs = (k * (k - 1)) / 2;
  const lastLayer = advL.heads[0].layers.length - 1;
  const outOff = sl.advAOff[lastLayer];
  const angular = sl.tag === "pair-rotation-scale-adjusted";
  let meanPair = 0;
  let minPair = 0;
  for (let t = 0; t < tuples; t++) {
    const base = t * sl.stride;
    const preds: number[][] = [];
    for (let j = 0; j < k; j++) {
      const hb = base + sl.advOff + j * sl.advBlk + outOff;
      const p = Array.from(scratch.subarray(hb, hb + sl.dy));
      if (angular) {
        const denom = Math.sqrt(p[0] * p[0] + p[1] * p[1] + ADV_SOFT_EPS2);
        p[0] /= denom;
        p[1] /= denom;
      }
      preds.push(p);
    }
    let sum = 0;
    let closest = Infinity;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        let d2 = 0;
        for (let o = 0; o < sl.dy; o++) {
          const d = preds[i][o] - preds[j][o];
          d2 += d * d;
        }
        const d = Math.sqrt(d2);
        sum += d;
        closest = Math.min(closest, d);
      }
    }
    meanPair += sum / pairs;
    minPair += closest;
  }
  return {
    meanPair: meanPair / tuples,
    minPair: minPair / tuples,
    pairs,
  };
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

/* ══════════════════════════════════════════════════════════════════════════
   shared fixture machinery
   ══════════════════════════════════════════════════════════════════════════ */

const PHYS = { width: 800, height: 600, forceMagnitude: 3.5, friction: 0.99, maxVelocity: 26 };
const ALPHA = 0.6;

const fieldWName = (h: number) => (l: number, i: number, j: number) => `w_${h}_${l}_${i}_${j}`;
const fieldBName = (h: number) => (l: number, j: number) => `b_${h}_${l}_${j}`;

/** field head dims used throughout: 2 heads, (encDim)→8→8→2 selu·selu·tanh */
function fieldDims(encDim: number): LayerDims[] {
  return [
    { inSize: encDim, outSize: 8, activation: "selu" },
    { inSize: 8, outSize: 8, activation: "selu" },
    { inSize: 8, outSize: 2, activation: "tanh" },
  ];
}
function fieldHeadDimsIR(encDim: number): HeadDim[] {
  return [
    { inSize: encDim, outSize: 8, act: "selu" },
    { inSize: 8, outSize: 8, act: "selu" },
    { inSize: 8, outSize: 2, act: "tanh" },
  ];
}

/** random packed field weights + the name→value env (both from one stream) */
function makeFieldWeights(layout: FieldLayout, seed: number) {
  const rnd = mulberry32(seed);
  const packed = new Float32Array(layout.totalFloats);
  const env: Record<string, number> = {};
  for (const seg of layout.segments) {
    const heads = layout.spec.heads as any[];
    const L = heads[seg.head].layers[seg.layer];
    for (let x = 0; x < seg.floatLength; x++) {
      const v = Math.fround((rnd() * 2 - 1) * 0.6);
      packed[seg.floatOffset + x] = v;
      if (seg.role === "kernel") {
        const i = Math.floor(x / L.outSize);
        const j = x % L.outSize;
        env[fieldWName(seg.head)(seg.layer, i, j)] = v;
      } else {
        env[fieldBName(seg.head)(seg.layer, x)] = v;
      }
    }
  }
  return { packed, env };
}

function makeAdvWeights(advL: AdversaryLayout, seed: number) {
  const rnd = mulberry32(seed);
  const packed = new Float32Array(advL.totalFloats);
  const env: Record<string, number> = {};
  for (const seg of advL.segments) {
    const L = advL.heads[seg.head].layers[seg.layer];
    for (let x = 0; x < seg.floatLength; x++) {
      const v = Math.fround((rnd() * 2 - 1) * 0.7);
      packed[seg.floatOffset + x] = v;
      if (seg.role === "kernel") {
        const i = Math.floor(x / L.outSize);
        const o = x % L.outSize;
        env[awName(seg.head)(seg.layer, i, o)] = v;
      } else {
        env[abName(seg.head)(seg.layer, x)] = v;
      }
    }
  }
  return { packed, env };
}

/** B tuples of m members: pos uniform in the domain, vel with tails past the
 *  clip (mask coverage). Values f32-quantized so GPU and oracle see identical
 *  inputs. Layout per tuple: m × [px, py, vx, vy]. */
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

/* ── the IR oracle: the ENTIRE per-tuple graph ──────────────────────────── */

interface OracleCfg {
  tag: TupleTag;
  observerGeometry?: "periodic" | "euclidean";
  k: number;
  eps: number;
  enc: Encoding; // field encoding
  hidden: number;
  feature: number;
}

type V2N = [Node, Node];

function buildTupleOracle(cfg: OracleCfg) {
  const { m, du, dy } = tupleDims(cfg.tag);
  const g = new Graph();
  const c = (v: number) => g.const(v);
  const encDim = cfg.enc.kind === "fourier" ? 2 + 4 * cfg.enc.octaves : 2;
  const fDimsIR = fieldHeadDimsIR(encDim);

  const encIR = (pn: V2N): Node[] => {
    if (cfg.enc.kind === "raw") return [pn[0], pn[1]];
    const out: Node[] = [pn[0], pn[1]];
    for (let kk = 0; kk < (cfg.enc as any).octaves; kk++) {
      const w = Math.pow(2, kk) * 2 * Math.PI;
      out.push(g.sin(g.mul(c(w), pn[0])));
      out.push(g.sin(g.mul(c(w), pn[1])));
      out.push(g.cos(g.mul(c(w), pn[0])));
      out.push(g.cos(g.mul(c(w), pn[1])));
    }
    return out;
  };
  const delta = (a: Node): Node =>
    cfg.observerGeometry === "euclidean"
      ? a
      : g.sub(g.mod(g.add(a, c(0.5)), c(1)), c(0.5));
  const dot2 = (a: V2N, b: V2N): Node => g.add(g.mul(a[0], b[0]), g.mul(a[1], b[1]));

  // Member positions + raw field outputs. Velocity/physics are deliberately
  // absent: this oracle pins the strict state-independent prediction game.
  const pn: V2N[] = [];
  const sig: V2N[] = [];
  for (let t = 0; t < m; t++) {
    const px = g.input(`m${t}px`);
    const py = g.input(`m${t}py`);
    const pnx = g.div(px, c(PHYS.width));
    const pny = g.div(py, c(PHYS.height));
    const gamma = encIR([pnx, pny]);
    const h0 = buildHead(g, fDimsIR, gamma, fieldWName(0), fieldBName(0));
    const h1 = buildHead(g, fDimsIR, gamma, fieldWName(1), fieldBName(1));
    const Fx = g.add(g.mul(c(1 - ALPHA), h0[0]), g.mul(c(ALPHA), h1[0]));
    const Fy = g.add(g.mul(c(1 - ALPHA), h0[1]), g.mul(c(ALPHA), h1[1]));
    pn.push([pnx, pny]);
    sig.push([Fx, Fy]);
  }

  // κ — tuple encoding
  let u: Node[];
  let y: Node[];
  let valid: Node | undefined;
  const E2 = c(1e-12);
  if (cfg.tag === "point") {
    u = [pn[0][0], pn[0][1]];
    y = [sig[0][0], sig[0][1]];
  } else if (
    cfg.tag === "pair" ||
    cfg.tag === "pair-rotation" ||
    cfg.tag === "pair-rotation-scale-raw" ||
    cfg.tag === "pair-rotation-scale-adjusted"
  ) {
    const dx = delta(g.sub(pn[1][0], pn[0][0]));
    const dyv = delta(g.sub(pn[1][1], pn[0][1]));
    const r = g.sqrt(g.add(g.add(g.mul(dx, dx), g.mul(dyv, dyv)), E2));
    const e1: V2N = [g.div(dx, r), g.div(dyv, r)];
    const e2: V2N = [g.neg(e1[1]), e1[0]];
    const del: V2N = [g.sub(sig[1][0], sig[0][0]), g.sub(sig[1][1], sig[0][1])];
    const q: V2N = [dot2(del, e1), dot2(del, e2)];
    const keepScale = cfg.tag === "pair" || cfg.tag === "pair-rotation";
    u = [keepScale ? r : c(1)];
    if (cfg.tag === "pair-rotation-scale-adjusted") {
      const q2 = dot2(q, q);
      const active = g.gt(q2, c(ADV_DIRECTION_ACTIVE_FLOOR ** 2));
      const qn = g.sqrt(g.max(q2, c(ADV_DIRECTION_ACTIVE_FLOOR ** 2)));
      y = [g.mul(active, g.div(q[0], qn)), g.mul(active, g.div(q[1], qn))];
    } else {
      y = q;
    }
  } else if (cfg.tag === "quad-labelled") {
    const d: V2N[] = [1, 2, 3].map((j) => [
      delta(g.sub(pn[j][0], pn[0][0])),
      delta(g.sub(pn[j][1], pn[0][1])),
    ] as V2N);
    const anchor2 = dot2(d[0], d[0]);
    const safeR = g.sqrt(g.add(anchor2, E2));
    const e1: V2N = [g.div(d[0][0], safeR), g.div(d[0][1], safeR)];
    const e2: V2N = [g.neg(e1[1]), e1[0]];
    const project = (v: V2N): V2N => [dot2(v, e1), dot2(v, e2)];
    u = d.flatMap((v) => project(v));

    const mean: V2N = [
      g.div(
        g.add(g.add(sig[0][0], sig[1][0]), g.add(sig[2][0], sig[3][0])),
        c(4)
      ),
      g.div(
        g.add(g.add(sig[0][1], sig[1][1]), g.add(sig[2][1], sig[3][1])),
        c(4)
      ),
    ];
    const rel: V2N[] = sig.map((v) => [
      g.sub(v[0], mean[0]),
      g.sub(v[1], mean[1]),
    ]);
    const encodeActive = g.gt(
      anchor2,
      c(ADV_QUAD_ANCHOR_ACTIVE_FLOOR ** 2)
    );
    y = rel
      .flatMap((v) => project(v))
      .map((v) => g.mul(encodeActive, v));
    // Match tupleActivity exactly: the residual mask is derived from the first
    // encoded relative vector, independently of encodeQuadLabelled's raw
    // anchor-magnitude data-hygiene mask.
    valid = g.gt(
      g.add(g.mul(u[0], u[0]), g.mul(u[1], u[1])),
      c(ADV_QUAD_ANCHOR_ACTIVE_FLOOR ** 2)
    );
  } else if (cfg.tag === "tri") {
    // tri — stop-gradient sort via gt/eq products (tape constants)
    const side = (a: V2N, b: V2N): Node => {
      const sx = delta(g.sub(b[0], a[0]));
      const sy = delta(g.sub(b[1], a[1]));
      return g.sqrt(g.add(g.add(g.mul(sx, sx), g.mul(sy, sy)), E2));
    };
    const s = [side(pn[1], pn[2]), side(pn[0], pn[2]), side(pn[0], pn[1])];
    const minGap = g.min(
      g.abs(g.sub(s[0], s[1])),
      g.min(g.abs(g.sub(s[0], s[2])), g.abs(g.sub(s[1], s[2])))
    );
    valid = g.gt(minGap, c(1e-5));
    const one = c(1);
    const eq = (a: Node, b: Node) =>
      g.mul(g.sub(one, g.gt(a, b)), g.sub(one, g.gt(b, a)));
    const rank = [
      g.add(g.gt(s[1], s[0]), g.gt(s[2], s[0])),
      g.add(g.add(g.gt(s[0], s[1]), g.gt(s[2], s[1])), eq(s[0], s[1])),
      g.add(g.add(g.add(g.gt(s[0], s[2]), g.gt(s[1], s[2])), eq(s[0], s[2])), eq(s[1], s[2])),
    ];
    const isr = (i: number, r: number) => eq(rank[i], c(r));
    const pick2 = (rows: V2N[], r: number): V2N => {
      let ax: Node = c(0);
      let ay: Node = c(0);
      for (let i = 0; i < 3; i++) {
        ax = g.add(ax, g.mul(isr(i, r), rows[i][0]));
        ay = g.add(ay, g.mul(isr(i, r), rows[i][1]));
      }
      return [ax, ay];
    };
    const pick1 = (vals: Node[], r: number): Node => {
      let a: Node = c(0);
      for (let i = 0; i < 3; i++) a = g.add(a, g.mul(isr(i, r), vals[i]));
      return a;
    };
    const xA = pick2(pn, 0);
    const xB = pick2(pn, 1);
    const xC = pick2(pn, 2);
    const dAB: V2N = [
      delta(g.sub(xB[0], xA[0])),
      delta(g.sub(xB[1], xA[1])),
    ];
    const dAC: V2N = [
      delta(g.sub(xC[0], xA[0])),
      delta(g.sub(xC[1], xA[1])),
    ];
    const cen: V2N = [g.div(g.add(dAB[0], dAC[0]), c(3)), g.div(g.add(dAB[1], dAC[1]), c(3))];
    const vA: V2N = [g.neg(cen[0]), g.neg(cen[1])];
    const lenA = g.sqrt(g.add(dot2(vA, vA), E2));
    const e1: V2N = [g.div(vA[0], lenA), g.div(vA[1], lenA)];
    const e2: V2N = [g.neg(e1[1]), e1[0]];
    const dbar: V2N = [
      g.div(g.add(g.add(sig[0][0], sig[1][0]), sig[2][0]), c(3)),
      g.div(g.add(g.add(sig[0][1], sig[1][1]), sig[2][1]), c(3)),
    ];
    const rel: V2N[] = sig.map((d) => [g.sub(d[0], dbar[0]), g.sub(d[1], dbar[1])] as V2N);
    u = [pick1(s, 0), pick1(s, 1), pick1(s, 2)];
    y = [];
    for (let r = 0; r < 3; r++) {
      const rr = pick2(rel, r);
      y.push(dot2(rr, e1));
      y.push(dot2(rr, e2));
    }
  } else {
    const unhandled: never = cfg.tag;
    throw new Error(`oracle: unhandled tuple encoding ${String(unhandled)}`);
  }

  const advDims: HeadDim[] = [
    { inSize: du, outSize: cfg.hidden, act: "selu" },
    { inSize: cfg.hidden, outSize: cfg.feature, act: "selu" },
    { inSize: cfg.feature, outSize: dy, act: "linear" },
  ];
  const term = wtaTerm(
    g,
    u,
    y,
    advDims,
    cfg.k,
    cfg.eps,
    cfg.tag === "pair-rotation-scale-adjusted" ? "angular" : "euclidean",
    valid
  );
  return { g, u, y, term, m, du, dy, advDims };
}

/** adversary/field param name orders matching the packed buffer offsets */
function advNameOfOffset(advL: AdversaryLayout): (string | null)[] {
  const names: (string | null)[] = new Array(advL.totalFloats).fill(null);
  for (const seg of advL.segments) {
    const L = advL.heads[seg.head].layers[seg.layer];
    for (let x = 0; x < seg.floatLength; x++) {
      names[seg.floatOffset + x] =
        seg.role === "kernel"
          ? awName(seg.head)(seg.layer, Math.floor(x / L.outSize), x % L.outSize)
          : abName(seg.head)(seg.layer, x);
    }
  }
  return names;
}
function fieldNameOfOffset(layout: FieldLayout): (string | null)[] {
  const names: (string | null)[] = new Array(layout.totalFloats).fill(null);
  const heads = layout.spec.heads as any[];
  for (const seg of layout.segments) {
    const L = heads[seg.head].layers[seg.layer];
    for (let x = 0; x < seg.floatLength; x++) {
      names[seg.floatOffset + x] =
        seg.role === "kernel"
          ? fieldWName(seg.head)(seg.layer, Math.floor(x / L.outSize), x % L.outSize)
          : fieldBName(seg.head)(seg.layer, x);
    }
  }
  return names;
}

/* ══════════════════════════════════════════════════════════════════════════
   §1 layout gates
   ══════════════════════════════════════════════════════════════════════════ */
console.log("--- 1. layoutAdversary / validateChain gates ---");
{
  const pairDims: LayerDims[] = [
    { inSize: 1, outSize: 32, activation: "selu" },
    { inSize: 32, outSize: 16, activation: "selu" },
    { inSize: 16, outSize: 2, activation: "linear" },
  ];
  const triDims: LayerDims[] = [
    { inSize: 3, outSize: 32, activation: "selu" },
    { inSize: 32, outSize: 16, activation: "selu" },
    { inSize: 16, outSize: 6, activation: "linear" },
  ];
  const quadDims: LayerDims[] = [
    { inSize: 6, outSize: 32, activation: "selu" },
    { inSize: 32, outSize: 16, activation: "selu" },
    { inSize: 16, outSize: 8, activation: "linear" },
  ];
  const pair = layoutAdversary(4, pairDims, { du: 1, dy: 2 });
  const tri = layoutAdversary(6, triDims, { du: 3, dy: 6 });
  const quad = layoutAdversary(4, quadDims, { du: 6, dy: 8 });
  ok(pair.segments.length === 4 * 6 && pair.heads.length === 4,
    `pair 1→2 k=4 validates (${pair.totalFloats} floats, ${pair.segments.length} segments)`);
  ok(tri.segments.length === 6 * 6 && tri.totalFloats % 4 === 0,
    `tri 3→6 k=6 validates (${tri.totalFloats} floats)`);
  ok(quad.segments.length === 4 * 6 && tupleDims("quad-labelled").m === 4,
    `quad-labelled 6→8 k=4 validates (${quad.totalFloats} floats)`);
  // offsets 16B-aligned and non-overlapping
  let disjoint = true;
  const sorted = [...tri.segments].sort((a, b) => a.floatOffset - b.floatOffset);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].floatOffset < sorted[i - 1].floatOffset + sorted[i - 1].floatLength) disjoint = false;
  }
  ok(disjoint && sorted.every((s) => s.floatOffset % 4 === 0),
    "tri segments disjoint + 4-float aligned");
  ok(
    ADV_STATS_BASE >= 5 + 16,
    `stats base ${ADV_STATS_BASE} is disjoint from largest legal finalized prefix 5+K=${5 + 16}`
  );

  throws(() => layoutAdversary(0, pairDims, { du: 1, dy: 2 }), "k=0 throws");
  throws(() => layoutAdversary(2, pairDims, { du: 3, dy: 2 }), "du mismatch throws");
  throws(() => layoutAdversary(2, pairDims, { du: 1, dy: 6 }), "dy mismatch throws");
  throws(
    () => layoutAdversary(2, [
      { inSize: 1, outSize: 8, activation: "selu" },
      { inSize: 9, outSize: 2, activation: "linear" },
    ], { du: 1, dy: 2 }),
    "broken chain (8→9) throws"
  );
  // FIELD constraints intact: 3-wide field-head output still rejected
  throws(
    () => layoutField("helmholtz", [
      [{ inSize: 2, outSize: 3, activation: "tanh" }],
      [{ inSize: 2, outSize: 2, activation: "tanh" }],
    ]),
    "field head with outSize 3 still throws (generalization, not loosening)"
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   §2 kernel ≡ IR oracle
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 2. fused kernel vs IR oracle (physics+encoding+WTA) ---");

const GEN_SEED = 1.234; // arbitrary nonzero generator seed (uniform value)
const B2 = 64;

async function oracleCheck(label: string, cfg: OracleCfg, seed: number) {
  const { m, du, dy } = tupleDims(cfg.tag);
  const encDim = cfg.enc.kind === "fourier" ? 2 + 4 * cfg.enc.octaves : 2;
  const layout = layoutField("helmholtz", [fieldDims(encDim), fieldDims(encDim)], {
    encoding: cfg.enc,
  });
  const fw = makeFieldWeights(layout, seed * 3 + 1);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);

  const trainer = new AdversaryTrainer(device, layout, {
    tag: cfg.tag, k: cfg.k, relaxEps: cfg.eps,
    observerGeometry: cfg.observerGeometry ?? "periodic",
    hiddenUnits: cfg.hidden, featureDim: cfg.feature,
    batchCap: 256, fieldWeightsBuffer: fieldWBuf, seed,
  });
  const aw = makeAdvWeights(trainer.advL, seed * 5 + 2);
  trainer.uploadAdvWeights(aw.packed);
  const tuples = makeTuples(B2, m, seed * 7 + 3);
  trainer.uploadTuples(tuples);

  trainer.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 1e-3, seed: 0,
    source: "uploaded", genSeed: GEN_SEED, applyDisc: false,
  });

  const sl = advScratchLayout(layout, trainer.advL, cfg.tag);
  const scratch = await readBuf((trainer as any).scratchBuf, B2 * sl.stride);
  const stats = await trainer.readStats();
  const advGrads = await trainer.readAdvGrads();
  const extGrads = await trainer.readExtGrads();
  const rawPlane = await trainer.readSurprise(B2 * m, "raw-payoff");
  const rawAlias = await trainer.readSurprise(B2 * m);
  const perUnitPlane = await trainer.readSurprise(B2 * m, "per-unit-signal");

  // ---- oracle -------------------------------------------------------------
  const o = buildTupleOracle(cfg);
  const advNames = advNameOfOffset(trainer.advL);
  const fieldNames = fieldNameOfOffset(layout);
  const advNameList = advNames.filter((n): n is string => n !== null);
  const fieldNameList = fieldNames.filter((n): n is string => n !== null);
  const gDisc = grad(o.g, o.term.weighted, advNameList);
  const gGen = grad(o.g, o.term.payoff, fieldNameList);
  const roots: Node[] = [
    ...o.u, ...o.y, o.term.surprise, o.term.weighted, ...o.term.resid,
    ...advNameList.map((n) => gDisc[n]),
    ...fieldNameList.map((n) => gGen[n]),
  ];
  const env: Record<string, number> = { ...fw.env, ...aw.env };

  let maxU = 0, maxY = 0, maxSur = 0, winMis = 0;
  let discLossO = 0, surO = 0, y2O = 0;
  const advAcc: Record<string, number> = {};
  const extAcc: Record<string, number> = {};
  for (const n of advNameList) advAcc[n] = 0;
  for (const n of fieldNameList) extAcc[n] = 0;

  for (let t = 0; t < B2; t++) {
    for (let j = 0; j < m; j++) {
      env[`m${j}px`] = tuples[(t * m + j) * 4];
      env[`m${j}py`] = tuples[(t * m + j) * 4 + 1];
      env[`m${j}vx`] = tuples[(t * m + j) * 4 + 2];
      env[`m${j}vy`] = tuples[(t * m + j) * 4 + 3];
    }
    const vals = evalNodes(roots, env);
    const sBase = t * sl.stride;
    for (let i = 0; i < du; i++) {
      maxU = Math.max(maxU, Math.abs(vals[i] - scratch[sBase + sl.uOff + i]));
    }
    for (let i = 0; i < dy; i++) {
      maxY = Math.max(maxY, Math.abs(vals[du + i] - scratch[sBase + sl.yOff + i]));
    }
    const surV = vals[du + dy];
    const wV = vals[du + dy + 1];
    maxSur = Math.max(maxSur, Math.abs(surV - scratch[sBase + sl.surOff]));
    let am = 0;
    for (let j = 1; j < cfg.k; j++) {
      if (vals[du + dy + 2 + j] < vals[du + dy + 2 + am]) am = j;
    }
    if (am !== scratch[sBase + sl.winOff]) winMis++;
    discLossO += wV / B2;
    surO += surV / B2;
    for (let i = 0; i < dy; i++) y2O += vals[du + i] * vals[du + i] / B2;
    const gb = du + dy + 2 + cfg.k;
    advNameList.forEach((n, i) => (advAcc[n] += vals[gb + i] / B2));
    fieldNameList.forEach((n, i) => (extAcc[n] += -GEN_SEED * vals[gb + advNameList.length + i]));
  }

  // FD spot-check of the ORACLE itself (a handful of field weights, gen path)
  {
    const rnd = mulberry32(seed + 99);
    const picks: string[] = [];
    for (let i = 0; i < 6; i++) {
      picks.push(fieldNameList[Math.floor(rnd() * fieldNameList.length)]);
    }
    // one representative tuple
    for (let j = 0; j < m; j++) {
      env[`m${j}px`] = tuples[j * 4];
      env[`m${j}py`] = tuples[j * 4 + 1];
      env[`m${j}vx`] = tuples[j * 4 + 2];
      env[`m${j}vy`] = tuples[j * 4 + 3];
    }
    const EPS = 1e-4;
    let worst = 0;
    for (const n of picks) {
      const rev = evalNodes([gGen[n]], env)[0];
      const x0 = env[n];
      env[n] = x0 + EPS;
      const hi = evalNodes([o.term.payoff], env)[0];
      env[n] = x0 - EPS;
      const lo = evalNodes([o.term.payoff], env)[0];
      env[n] = x0;
      const fd = (hi - lo) / (2 * EPS);
      worst = Math.max(worst, Math.abs(rev - fd) / (Math.abs(fd) + 1e-4));
    }
    ok(worst < 2e-3, `${label}: oracle FD spot-check on 6 field weights (worst rel ${worst.toExponential(2)})`);
  }

  const relLoss = Math.abs(stats.discLoss - discLossO) / (Math.abs(discLossO) + 1e-12);
  const relSur = Math.abs(stats.surprise - surO) / (Math.abs(surO) + 1e-12);
  const rmsO = Math.sqrt(y2O);
  const relRms = Math.abs(stats.batchRms - rmsO) / (rmsO + 1e-12);
  const spreadO = spreadFromScratch(scratch, sl, trainer.advL, B2);
  let maxRawPlane = 0;
  let maxPerUnitPlane = 0;
  let maxRawAlias = 0;
  for (let t = 0; t < B2; t++) {
    const sBase = t * sl.stride;
    const raw = scratch[sBase + sl.surOff];
    let norm2 = 0;
    for (let o = 0; o < dy; o++) {
      const y = scratch[sBase + sl.yOff + o];
      norm2 += y * y;
    }
    const expectedPerUnit =
      raw !== 0 && norm2 > ADV_DIRECTION_ACTIVE_FLOOR ** 2
        ? raw / Math.max(Math.sqrt(norm2), ADV_DIRECTION_ACTIVE_FLOOR)
        : 0;
    for (let member = 0; member < m; member++) {
      const p = t * m + member;
      maxRawPlane = Math.max(maxRawPlane, Math.abs(rawPlane[p] - raw));
      maxRawAlias = Math.max(maxRawAlias, Math.abs(rawAlias[p] - raw));
      maxPerUnitPlane = Math.max(
        maxPerUnitPlane,
        Math.abs(perUnitPlane[p] - expectedPerUnit)
      );
    }
  }

  const advVecO = advNameList.map((n) => advAcc[n]);
  const advVecG: number[] = [];
  advNames.forEach((n, off) => {
    if (n !== null) advVecG.push(advGrads[off]);
  });
  const extVecO = fieldNameList.map((n) => extAcc[n]);
  const extVecG: number[] = [];
  fieldNames.forEach((n, off) => {
    if (n !== null) extVecG.push(extGrads[off]);
  });
  const cosAdv = cosine(advVecO, advVecG);
  const cosExt = cosine(extVecO, extVecG);
  const scaleOf = (v: number[]) => Math.max(...v.map(Math.abs));
  const advScale = scaleOf(advVecO);
  const extScale = scaleOf(extVecO);
  let advRel = 0, extRel = 0;
  for (let i = 0; i < advVecO.length; i++) {
    advRel = Math.max(advRel, Math.abs(advVecO[i] - advVecG[i]) / advScale);
  }
  for (let i = 0; i < extVecO.length; i++) {
    extRel = Math.max(extRel, Math.abs(extVecO[i] - extVecG[i]) / extScale);
  }

  ok(maxU < 1e-5, `${label}: context u parity (max |Δ| ${maxU.toExponential(2)})`);
  ok(maxY < 1e-5, `${label}: target y parity (max |Δ| ${maxY.toExponential(2)})`);
  ok(maxSur < 1e-5 && winMis === 0,
    `${label}: surprise + winner parity (max |Δ| ${maxSur.toExponential(2)}, ${winMis}/${B2} winner mismatch)`);
  ok(relLoss < 1e-5 && relSur < 1e-5 && relRms < 1e-5,
    `${label}: stats parity — disc ${stats.discLoss.toFixed(7)} (rel ${relLoss.toExponential(2)}), ` +
      `sur rel ${relSur.toExponential(2)}, rms rel ${relRms.toExponential(2)}`);
  ok(
    maxRawPlane < 1e-6 &&
      maxRawAlias < 1e-6 &&
      maxPerUnitPlane < 2e-5,
    `${label}: packed surprise planes — raw ${maxRawPlane.toExponential(2)}, ` +
      `compat ${maxRawAlias.toExponential(2)}, per-unit ${maxPerUnitPlane.toExponential(2)}`
  );
  if (spreadO === null) {
    ok(
      stats.headSpread.tag === "single-head",
      `${label}: K=1 reports the explicit single-head sum type`
    );
  } else {
    const meanRel =
      stats.headSpread.tag === "spread"
        ? Math.abs(stats.headSpread.meanPair - spreadO.meanPair) /
          (Math.abs(spreadO.meanPair) + 1e-12)
        : Infinity;
    const minRel =
      stats.headSpread.tag === "spread"
        ? Math.abs(stats.headSpread.minPair - spreadO.minPair) /
          (Math.abs(spreadO.minPair) + 1e-12)
        : Infinity;
    ok(
      stats.headSpread.tag === "spread" &&
        stats.headSpread.pairs === spreadO.pairs &&
        meanRel < 2e-5 &&
        minRel < 2e-5,
      `${label}: head geometry includes mean/closest pairs ` +
        `(rel ${meanRel.toExponential(2)}/${minRel.toExponential(2)}, pairs ${spreadO.pairs})`
    );
  }
  ok(cosAdv > 0.99999 && advRel < 1e-3,
    `${label}: DISC adversary-weight grads (${advVecO.length}) cos=${cosAdv.toFixed(7)} scale-rel ${advRel.toExponential(2)}`);
  ok(cosExt > 0.99999 && extRel < 1e-3,
    `${label}: GEN field-weight grads/extGrads (${extVecO.length}) cos=${cosExt.toFixed(7)} scale-rel ${extRel.toExponential(2)}`);

  trainer.destroy();
  fieldWBuf.destroy();
}

await oracleCheck("point k=2 raw", { tag: "point", k: 2, eps: 0.05, enc: { kind: "raw" }, hidden: 8, feature: 8 }, 11);
await oracleCheck(
  "point k=2 EUCLIDEAN (geometry-independent control)",
  {
    tag: "point",
    observerGeometry: "euclidean",
    k: 2,
    eps: 0.05,
    enc: { kind: "raw" },
    hidden: 8,
    feature: 8,
  },
  111
);
await oracleCheck("pair  k=1 raw (single control)", { tag: "pair", k: 1, eps: 0, enc: { kind: "raw" }, hidden: 8, feature: 8 }, 12);
await oracleCheck("pair  k=4 raw", { tag: "pair", k: 4, eps: 0.05, enc: { kind: "raw" }, hidden: 8, feature: 8 }, 13);
await oracleCheck("pair  rotation explicit", { tag: "pair-rotation", k: 4, eps: 0.05, enc: { kind: "raw" }, hidden: 8, feature: 8 }, 131);
await oracleCheck("pair  rotation+scale RAW control", { tag: "pair-rotation-scale-raw", k: 4, eps: 0.05, enc: { kind: "raw" }, hidden: 8, feature: 8 }, 132);
// The old adjusted q/|q| oracle was retired. Explicit soft-angle and local
// scale objectives are gated below against the new core/fused contract.
await oracleCheck("pair  k=2 HARD eps=0", { tag: "pair", k: 2, eps: 0, enc: { kind: "raw" }, hidden: 8, feature: 8 }, 14);
await oracleCheck("tri   k=3 raw", { tag: "tri", k: 3, eps: 0.05, enc: { kind: "raw" }, hidden: 8, feature: 8 }, 15);
await oracleCheck("quad-labelled k=3 raw", { tag: "quad-labelled", k: 3, eps: 0.05, enc: { kind: "raw" }, hidden: 8, feature: 8 }, 151);
await oracleCheck("pair  k=2 FOURIER field (oct 2)", { tag: "pair", k: 2, eps: 0.05, enc: { kind: "fourier", octaves: 2 }, hidden: 8, feature: 8 }, 16);
await oracleCheck(
  "pair  k=3 EUCLIDEAN observer",
  {
    tag: "pair",
    observerGeometry: "euclidean",
    k: 3,
    eps: 0.05,
    enc: { kind: "raw" },
    hidden: 8,
    feature: 8,
  },
  161
);
await oracleCheck(
  "tri   k=3 EUCLIDEAN observer",
  {
    tag: "tri",
    observerGeometry: "euclidean",
    k: 3,
    eps: 0.05,
    enc: { kind: "raw" },
    hidden: 8,
    feature: 8,
  },
  162
);
await oracleCheck(
  "quad-labelled k=3 EUCLIDEAN observer",
  {
    tag: "quad-labelled",
    observerGeometry: "euclidean",
    k: 3,
    eps: 0.05,
    enc: { kind: "raw" },
    hidden: 8,
    feature: 8,
  },
  163
);

console.log("\n--- 2a0. legal K=16 stats layout has no finalized/partial overlap ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 1640);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const B = 129; // two workgroups, so both partial strides are exercised
  const trainer = new AdversaryTrainer(device, layout, {
    tag: "point",
    observerGeometry: "periodic",
    k: 16,
    relaxEps: 0.05,
    hiddenUnits: 2,
    featureDim: 2,
    batchCap: B,
    fieldWeightsBuffer: fieldWBuf,
    seed: 1641,
  });
  trainer.uploadAdvWeights(makeAdvWeights(trainer.advL, 1642).packed);
  trainer.uploadTuples(makeTuples(B, 1, 1643));
  trainer.step(PHYS, {
    b: B,
    alpha: ALPHA,
    lr: 1e-3,
    source: "uploaded",
    genSeed: GEN_SEED,
    applyDisc: false,
  });
  const sl = advScratchLayout(layout, trainer.advL, "point");
  const [stats, scratch, packed] = await Promise.all([
    trainer.readStats(),
    trainer.readScratch(B * sl.stride),
    readBuf((trainer as any).statsBuf, ADV_STATS_BASE + 2 * (7 + 16)),
  ]);
  const expected = spreadFromScratch(scratch, sl, trainer.advL, B)!;
  const reservedGapZero = packed
    .subarray(7 + 16, ADV_STATS_BASE)
    .every((v) => v === 0);
  ok(
    stats.headSpread.tag === "spread" &&
      stats.headSpread.pairs === 120 &&
      Math.abs(stats.headSpread.meanPair - expected.meanPair) /
        (Math.abs(expected.meanPair) + 1e-12) <
        2e-5 &&
      stats.winCounts.reduce((a, b) => a + b, 0) === B,
    "K=16 final head stats/wins survive two workgroup partials"
  );
  ok(
    reservedGapZero &&
      ADV_STATS_BASE === 32 &&
      ADV_STATS_BASE >= 7 + trainer.k,
    `final prefix [0,${7 + trainer.k}) and partial base ${ADV_STATS_BASE} are physically disjoint`
  );
  trainer.destroy();
  fieldWBuf.destroy();
}

console.log("\n--- 2a00. closest-pair spread distinguishes one piled pair ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 1644);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const trainer = new AdversaryTrainer(device, layout, {
    tag: "point",
    observerGeometry: "periodic",
    k: 3,
    relaxEps: 0.05,
    hiddenUnits: 2,
    featureDim: 2,
    batchCap: B2,
    fieldWeightsBuffer: fieldWBuf,
    seed: 1645,
  });
  // Heads 0/1 are identical zeros. Head 2 is the constant vector (1,-1).
  // Therefore closestPair=0 while meanPair=(0+√2+√2)/3.
  const advW = new Float32Array(trainer.advL.totalFloats);
  const last = trainer.advL.heads[2].layers.at(-1)!;
  advW[last.biasOffset] = 1;
  advW[last.biasOffset + 1] = -1;
  trainer.uploadAdvWeights(advW);
  trainer.uploadTuples(makeTuples(B2, 1, 1646));
  trainer.step(PHYS, {
    b: B2,
    alpha: ALPHA,
    lr: 1e-3,
    source: "uploaded",
    genSeed: 0,
    applyDisc: false,
  });
  const stats = await trainer.readStats();
  const expectedMean = (2 * Math.SQRT2) / 3;
  ok(
    stats.headSpread.tag === "spread" &&
      stats.headSpread.minPair === 0 &&
      Math.abs(stats.headSpread.meanPair - expectedMean) < 2e-6,
    `closest pair remains 0 while mean pair is ${expectedMean.toFixed(6)}`
  );
  trainer.destroy();
  fieldWBuf.destroy();
}

/* ── boundary-aware observer geometry: opposite edges near only on a torus ─ */
console.log("\n--- 2a. explicit periodic vs Euclidean observer geometry ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 164);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const make = (observerGeometry: "periodic" | "euclidean") =>
    new AdversaryTrainer(device, layout, {
      tag: "pair-rotation",
      observerGeometry,
      k: 2,
      relaxEps: 0.05,
      hiddenUnits: 8,
      featureDim: 8,
      batchCap: 1,
      fieldWeightsBuffer: fieldWBuf,
      seed: 165,
    });
  const periodic = make("periodic");
  const euclidean = make("euclidean");
  const aw = makeAdvWeights(periodic.advL, 166).packed;
  periodic.uploadAdvWeights(aw);
  euclidean.uploadAdvWeights(aw);
  // Normalized x positions 0.99 and 0.01 are 0.02 apart on the torus but
  // 0.98 apart in a bounded Euclidean rectangle.
  const seam = new Float32Array([
    0.99 * PHYS.width, 0.5 * PHYS.height, 0, 0,
    0.01 * PHYS.width, 0.5 * PHYS.height, 0, 0,
  ]);
  periodic.uploadTuples(seam);
  euclidean.uploadTuples(seam);
  const run = async (trainer: AdversaryTrainer) => {
    trainer.step(PHYS, {
      b: 1,
      alpha: ALPHA,
      lr: 1e-3,
      source: "uploaded",
      genSeed: GEN_SEED,
      applyDisc: false,
    });
    const sl = advScratchLayout(layout, trainer.advL, trainer.tag);
    return trainer.readScratch(sl.stride);
  };
  const [sp, se] = await Promise.all([run(periodic), run(euclidean)]);
  const sl = advScratchLayout(layout, periodic.advL, periodic.tag);
  const rp = sp[sl.uOff];
  const re = se[sl.uOff];
  ok(
    Math.abs(rp - 0.02) < 2e-6 && Math.abs(re - 0.98) < 2e-6,
    `seam pair is near only for periodic observer (r=${rp.toFixed(6)} vs ${re.toFixed(6)})`
  );
  ok(
    periodic.observerGeometry === "periodic" &&
      euclidean.observerGeometry === "euclidean",
    "trainer records the explicit observer geometry used to compile its shader"
  );
  periodic.destroy();
  euclidean.destroy();
  fieldWBuf.destroy();
}

/* ── strict target independence: velocity and physics are outside the game ─ */
console.log("\n--- 2b. strict force target ignores velocity / physics controls ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 171);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const trainer = new AdversaryTrainer(device, layout, {
    tag: "pair-rotation", k: 4, relaxEps: 0.05,
    observerGeometry: "periodic",
    hiddenUnits: 8, featureDim: 8, batchCap: 256,
    fieldWeightsBuffer: fieldWBuf, seed: 172,
  });
  const aw = makeAdvWeights(trainer.advL, 173);
  trainer.uploadAdvWeights(aw.packed);
  const a = makeTuples(B2, 2, 174);
  const b = a.slice();
  for (let i = 0; i < B2 * 2; i++) {
    b[i * 4 + 2] = Math.fround(1000 - i * 7);
    b[i * 4 + 3] = Math.fround(-500 + i * 11);
  }
  const sl = advScratchLayout(layout, trainer.advL, trainer.tag);
  const run = async (tuples: Float32Array, phys: typeof PHYS) => {
    trainer.uploadTuples(tuples);
    trainer.step(phys, {
      b: B2, alpha: ALPHA, lr: 1e-3, source: "uploaded",
      genSeed: GEN_SEED, applyDisc: false,
    });
    return {
      scratch: await readBuf((trainer as any).scratchBuf, B2 * sl.stride),
      ext: await trainer.readExtGrads(),
    };
  };
  const one = await run(a, PHYS);
  const two = await run(b, {
    width: PHYS.width,
    height: PHYS.height,
    forceMagnitude: 999,
    friction: 0.01,
    maxVelocity: 0.001,
  });
  let stateDiff = 0;
  for (let t = 0; t < B2; t++) {
    const base = t * sl.stride;
    for (let i = 0; i < sl.du; i++) {
      stateDiff = Math.max(stateDiff, Math.abs(one.scratch[base + sl.uOff + i] - two.scratch[base + sl.uOff + i]));
    }
    for (let i = 0; i < sl.dy; i++) {
      stateDiff = Math.max(stateDiff, Math.abs(one.scratch[base + sl.yOff + i] - two.scratch[base + sl.yOff + i]));
    }
    stateDiff = Math.max(stateDiff, Math.abs(one.scratch[base + sl.surOff] - two.scratch[base + sl.surOff]));
  }
  let extDiff = 0;
  for (let i = 0; i < one.ext.length; i++) extDiff = Math.max(extDiff, Math.abs(one.ext[i] - two.ext[i]));
  ok(stateDiff === 0, "u/y/payoff are bit-identical after arbitrary velocity + physics changes");
  ok(extDiff === 0, "field generator gradients are bit-identical after arbitrary velocity + physics changes");
  trainer.destroy();
  fieldWBuf.destroy();
}

/* ── explicit smooth angle + local relative-scale GPU objective ─────────── */
console.log("\n--- 2c. relative-scale target is invariant; energy remains observable ---");
{
  const linear: LayerDims[] = [{ inSize: 2, outSize: 2, activation: "linear" }];
  const layout = layoutField("helmholtz", [linear, linear]);
  const fw = makeFieldWeights(layout, 181);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const loss = {
    tag: "angle-relative-scale" as const,
    tau: 0.05,
    scaleWeight: 0.4,
    energyWeight: 0.2,
    energyTarget: 0.35,
  };
  const trainer = new AdversaryTrainer(device, layout, {
    tag: "pair-rotation-scale-adjusted", k: 4, relaxEps: 0.05,
    target: { tag: "force" },
    loss,
    observerGeometry: "periodic",
    hiddenUnits: 8, featureDim: 8, batchCap: 256,
    fieldWeightsBuffer: fieldWBuf, seed: 182,
  });
  trainer.uploadAdvWeights(makeAdvWeights(trainer.advL, 183).packed);
  const tuples = makeTuples(B2, 2, 184);
  trainer.uploadTuples(tuples);
  const sl = advScratchLayout(
    layout, trainer.advL, trainer.tag, trainer.target, trainer.loss
  );
  trainer.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 1e-3, source: "uploaded",
    genSeed: GEN_SEED, applyDisc: false,
  });
  const baseStats = await trainer.readStats();
  const baseScratch = await readBuf(
    (trainer as any).scratchBuf, B2 * sl.stride
  );
  const baseExt = await trainer.readExtGrads();
  ok(
    baseExt.every(Number.isFinite) && baseExt.some((v) => Math.abs(v) > 1e-8),
    "relative-scale field gradient is finite and active"
  );

  const baseScale = new Float32Array(B2 * sl.scaleDy);
  for (let t = 0; t < B2; t++) {
    for (let q = 0; q < sl.scaleDy; q++) {
      baseScale[t * sl.scaleDy + q] =
        baseScratch[t * sl.stride + sl.yOff + sl.vectorDy + q];
    }
  }

  const scaled = Float32Array.from(fw.packed, (v) => v * 5);
  device.queue.writeBuffer(fieldWBuf, 0, scaled);
  trainer.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 1e-3, source: "uploaded",
    genSeed: GEN_SEED, applyDisc: false,
  });
  const scaledStats = await trainer.readStats();
  const scaledScratch = await readBuf(
    (trainer as any).scratchBuf, B2 * sl.stride
  );
  let scaleTargetDelta = 0;
  for (let t = 0; t < B2; t++) {
    for (let q = 0; q < sl.scaleDy; q++) {
      scaleTargetDelta = Math.max(
        scaleTargetDelta,
        Math.abs(
          baseScale[t * sl.scaleDy + q] -
            scaledScratch[t * sl.stride + sl.yOff + sl.vectorDy + q]
        )
      );
    }
  }
  const energyRatio = scaledStats.energyRms / baseStats.energyRms;
  ok(
    scaleTargetDelta < 2e-5,
    `relative-log-scale target invariant under 5× common signal scale ` +
      `(max |Δ| ${scaleTargetDelta.toExponential(2)})`
  );
  ok(
    Math.abs(energyRatio - 5) < 2e-4,
    `positive energy channel remains scale-observable (ratio ${energyRatio.toFixed(6)})`
  );
  trainer.destroy();
  fieldWBuf.destroy();

  // Exact zero has neither direction nor a meaningful local scale contrast.
  // It must be inactive rather than manufacturing an epsilon direction.
  const zeroBuf = mkStorage(layout.totalFloats * 4); // WebGPU buffers zero-init
  const zero = new AdversaryTrainer(device, layout, {
    tag: "pair-rotation-scale-adjusted", k: 4, relaxEps: 0.05,
    target: { tag: "force" },
    loss,
    observerGeometry: "periodic",
    hiddenUnits: 8, featureDim: 8, batchCap: 256,
    fieldWeightsBuffer: zeroBuf, seed: 185,
  });
  zero.uploadAdvWeights(makeAdvWeights(zero.advL, 186).packed);
  zero.uploadTuples(makeTuples(B2, 2, 187));
  zero.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 1e-3, source: "uploaded",
    genSeed: GEN_SEED, applyDisc: false,
  });
  const zs = await zero.readStats();
  const za = await zero.readAdvGrads();
  const ze = await zero.readExtGrads();
  const zr = await zero.readSurprise(B2 * 2, "raw-payoff");
  const zu = await zero.readSurprise(B2 * 2, "per-unit-signal");
  ok(
    zs.discLoss === 0 && zs.surprise === 0 && zs.batchRms === 0 &&
      zs.energyActive === 0,
    "inactive zero target reports exact zero loss/payoff/RMS and active count"
  );
  ok(
    zs.headSpread.tag === "spread" &&
      zs.headSpread.meanPair > 0 &&
      Number.isFinite(zs.headSpread.minPair),
    "head spread still probes predictor geometry on target-inactive contexts"
  );
  ok(
    zr.every((v) => v === 0) && zu.every((v) => v === 0),
    "inactive target writes exact zero to raw and per-unit surprise planes"
  );
  ok(za.every((v) => v === 0) && ze.every((v) => v === 0),
    "inactive zero target has exact zero D and G gradients"
  );
  zero.destroy();
  zeroBuf.destroy();
}

console.log("\n--- 2c1. per-unit surprise is a scale diagnostic, not a game term ---");
{
  const linear: LayerDims[] = [
    { inSize: 2, outSize: 2, activation: "linear" },
  ];
  const layout = layoutField("helmholtz", [linear, linear]);
  const fieldW = new Float32Array(layout.totalFloats);
  const heads = layout.spec.heads as any[];
  fieldW[heads[0].layers[0].biasOffset] = 0.2;
  fieldW[heads[0].layers[0].biasOffset + 1] = -0.4;
  fieldW[heads[1].layers[0].biasOffset] = 0.7;
  fieldW[heads[1].layers[0].biasOffset + 1] = 0.1;
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fieldW);
  const B = 4;
  const trainer = new AdversaryTrainer(device, layout, {
    tag: "point",
    observerGeometry: "periodic",
    k: 1,
    relaxEps: 0,
    hiddenUnits: 2,
    featureDim: 2,
    batchCap: B,
    fieldWeightsBuffer: fieldWBuf,
    seed: 1880,
  });
  trainer.uploadAdvWeights(new Float32Array(trainer.advL.totalFloats));
  trainer.uploadTuples(makeTuples(B, 1, 1881));
  const run = async () => {
    trainer.step(PHYS, {
      b: B,
      alpha: ALPHA,
      lr: 1e-3,
      source: "uploaded",
      genSeed: GEN_SEED,
      applyDisc: false,
    });
    return Promise.all([
      trainer.readSurprise(B, "raw-payoff"),
      trainer.readSurprise(B, "per-unit-signal"),
      trainer.readStats(),
    ]);
  };
  const [raw1, unit1, stats1] = await run();
  device.queue.writeBuffer(
    fieldWBuf,
    0,
    Float32Array.from(fieldW, (v) => v * 5)
  );
  const [raw5, unit5, stats5] = await run();
  const rawRatio = raw5[0] / raw1[0];
  let maxUnitDelta = 0;
  for (let i = 0; i < B; i++) {
    maxUnitDelta = Math.max(maxUnitDelta, Math.abs(unit5[i] - unit1[i]));
  }
  const rawDesc = trainer.surprisePlane("raw-payoff");
  const unitDesc = trainer.surprisePlane("per-unit-signal");
  ok(
    Math.abs(rawRatio - 5) < 2e-5 &&
      stats5.surprise > 4.99 * stats1.surprise,
    `raw payoff follows a 5× target scale (ratio ${rawRatio.toFixed(6)})`
  );
  ok(
    maxUnitDelta < 2e-6 &&
      unit1.every((v) => Math.abs(v - 1) < 2e-5),
    `per-unit plane stays scale-stable and dimensionless (max Δ ${maxUnitDelta.toExponential(2)})`
  );
  ok(
    rawDesc.buffer === unitDesc.buffer &&
      rawDesc.offsetFloats === 0 &&
      unitDesc.offsetFloats === B &&
      trainer.surpriseBuf === rawDesc.buffer,
    "raw/per-unit descriptors address two packed planes; surpriseBuf remains the raw compatibility alias"
  );
  throws(
    () => (trainer as any).surprisePlane("not-a-metric"),
    "unknown surprise metric throws instead of silently selecting a plane"
  );
  trainer.destroy();
  fieldWBuf.destroy();
}

console.log("\n--- 2d. ambiguous triangle ties are explicit inactive data ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 188);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const trainer = new AdversaryTrainer(device, layout, {
    tag: "tri", k: 3, relaxEps: 0.05,
    observerGeometry: "periodic",
    hiddenUnits: 8, featureDim: 8, batchCap: 256,
    fieldWeightsBuffer: fieldWBuf, seed: 189,
  });
  trainer.uploadAdvWeights(makeAdvWeights(trainer.advL, 190).packed);
  const tuples = new Float32Array(B2 * 3 * 4);
  for (let b = 0; b < B2; b++) {
    // Isosceles in normalized coordinates: sides opposite labels 0 and 1
    // are exactly tied, so their canonical order is not label-independent.
    const pts = [[0.4, 0.4], [0.6, 0.4], [0.5, 0.55]];
    for (let t = 0; t < 3; t++) {
      const o = (b * 3 + t) * 4;
      tuples[o] = pts[t][0] * PHYS.width;
      tuples[o + 1] = pts[t][1] * PHYS.height;
      tuples[o + 2] = b + t;
      tuples[o + 3] = -b - t;
    }
  }
  trainer.uploadTuples(tuples);
  trainer.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 1e-3, source: "uploaded",
    genSeed: GEN_SEED, applyDisc: false,
  });
  const s = await trainer.readStats();
  const ag = await trainer.readAdvGrads();
  const eg = await trainer.readExtGrads();
  ok(s.discLoss === 0 && s.surprise === 0 && s.winCounts.every((n) => n === 0),
    "near-tie triangle contributes zero payoff and zero head wins");
  ok(ag.every((v) => v === 0) && eg.every((v) => v === 0),
    "near-tie triangle contributes exact zero D/G gradients");
  trainer.destroy();
  fieldWBuf.destroy();
}

console.log("\n--- 2d2. labelled quad: torus frame, label semantics, inactive anchor ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 195);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const trainer = new AdversaryTrainer(device, layout, {
    tag: "quad-labelled",
    observerGeometry: "periodic",
    k: 3,
    relaxEps: 0.05,
    hiddenUnits: 8,
    featureDim: 8,
    batchCap: B2,
    fieldWeightsBuffer: fieldWBuf,
    seed: 196,
  });
  trainer.uploadAdvWeights(makeAdvWeights(trainer.advL, 197).packed);
  const sl = advScratchLayout(layout, trainer.advL, "quad-labelled");

  const labelled = new Float32Array(2 * 4 * 4);
  const pts = [
    [0.98, 0.45],
    [0.02, 0.45], // minimum-image anchor is +0.04, not -0.96
    [0.99, 0.60],
    [0.75, 0.40],
  ];
  for (let b = 0; b < 2; b++) {
    const order = b === 0 ? [0, 1, 2, 3] : [0, 1, 3, 2];
    for (let t = 0; t < 4; t++) {
      const member = order[t];
      const o = (b * 4 + t) * 4;
      labelled[o] = pts[member][0] * PHYS.width;
      labelled[o + 1] = pts[member][1] * PHYS.height;
      labelled[o + 2] = member + 0.25;
      labelled[o + 3] = -member - 0.5;
    }
  }
  trainer.uploadTuples(labelled);
  trainer.step(PHYS, {
    b: 2,
    alpha: ALPHA,
    lr: 1e-3,
    source: "uploaded",
    genSeed: GEN_SEED,
    applyDisc: false,
  });
  const scratch = await trainer.readScratch(2 * sl.stride);
  const ua = Array.from(scratch.subarray(sl.uOff, sl.uOff + 6));
  const ub = Array.from(
    scratch.subarray(sl.stride + sl.uOff, sl.stride + sl.uOff + 6)
  );
  const ya = Array.from(scratch.subarray(sl.yOff, sl.yOff + 8));
  const yb = Array.from(
    scratch.subarray(sl.stride + sl.yOff, sl.stride + sl.yOff + 8)
  );
  ok(
    Math.abs(ua[0] - 0.04) < 2e-6 && Math.abs(ua[1]) < 2e-6,
    `quad anchor uses torus minimum image (frame coordinate ${ua[0].toFixed(6)}, ${ua[1].toExponential(1)})`
  );
  const swapU =
    Math.max(
      Math.abs(ua[0] - ub[0]),
      Math.abs(ua[1] - ub[1]),
      Math.abs(ua[2] - ub[4]),
      Math.abs(ua[3] - ub[5]),
      Math.abs(ua[4] - ub[2]),
      Math.abs(ua[5] - ub[3])
    );
  const swapY =
    Math.max(
      Math.abs(ya[0] - yb[0]),
      Math.abs(ya[1] - yb[1]),
      Math.abs(ya[2] - yb[2]),
      Math.abs(ya[3] - yb[3]),
      Math.abs(ya[4] - yb[6]),
      Math.abs(ya[5] - yb[7]),
      Math.abs(ya[6] - yb[4]),
      Math.abs(ya[7] - yb[5])
    );
  ok(
    swapU < 2e-6 && swapY < 2e-6,
    `swapping labels 2↔3 swaps their context/target slots exactly (max Δ ${Math.max(swapU, swapY).toExponential(2)})`
  );
  ok(
    Math.abs(ua[2] - ub[2]) > 1e-3 || Math.abs(ua[3] - ub[3]) > 1e-3,
    "quad is explicitly labelled, not silently permutation-invariant"
  );

  const degenerate = new Float32Array(B2 * 4 * 4);
  for (let b = 0; b < B2; b++) {
    const q = [
      [0.3, 0.3],
      [0.3, 0.3], // anchor pair coincident
      [0.45, 0.4],
      [0.2, 0.55],
    ];
    for (let t = 0; t < 4; t++) {
      const o = (b * 4 + t) * 4;
      degenerate[o] = q[t][0] * PHYS.width;
      degenerate[o + 1] = q[t][1] * PHYS.height;
    }
  }
  trainer.uploadTuples(degenerate);
  trainer.step(PHYS, {
    b: B2,
    alpha: ALPHA,
    lr: 1e-3,
    source: "uploaded",
    genSeed: GEN_SEED,
    applyDisc: false,
  });
  const [stats, advGrad, extGrad, deadScratch] = await Promise.all([
    trainer.readStats(),
    trainer.readAdvGrads(),
    trainer.readExtGrads(),
    trainer.readScratch(B2 * sl.stride),
  ]);
  let yNonzero = false;
  for (let b = 0; b < B2; b++) {
    for (let i = 0; i < 8; i++) {
      if (deadScratch[b * sl.stride + sl.yOff + i] !== 0) yNonzero = true;
    }
  }
  ok(
    stats.discLoss === 0 &&
      stats.surprise === 0 &&
      stats.batchRms === 0 &&
      stats.winCounts.every((n) => n === 0) &&
      !yNonzero,
    "coincident quad anchor gives exact zero y/payoff/RMS/head wins"
  );
  ok(
    advGrad.every((v) => v === 0) && extGrad.every((v) => v === 0),
    "coincident quad anchor gives exact zero discriminator and generator gradients"
  );

  trainer.destroy();
  fieldWBuf.destroy();
}

console.log("\n--- 2e. Agree+Disagree lane/role isolation ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 201);
  // One shared READ-ONLY field buffer, two wholly independent predictors.
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const makeLane = (
    lane: 0 | 1,
    role: "disagree" | "agree",
    seed: number
  ) => new AdversaryTrainer(device, layout, {
    tag: "pair-rotation-scale-adjusted",
    observerGeometry: "euclidean",
    fieldLane: lane,
    generatorRole: role,
    k: 4,
    relaxEps: 0.05,
    hiddenUnits: 8,
    featureDim: 8,
    batchCap: 256,
    fieldWeightsBuffer: fieldWBuf,
    seed,
  });
  const lane0 = makeLane(0, "disagree", 202);
  const lane1 = makeLane(1, "disagree", 203);
  const sharedAdv = makeAdvWeights(lane0.advL, 204).packed;
  const tuples = makeTuples(B2, 2, 205);
  lane0.uploadAdvWeights(sharedAdv);
  lane1.uploadAdvWeights(sharedAdv);
  lane0.uploadTuples(tuples);
  lane1.uploadTuples(tuples);
  const run = async (trainer: AdversaryTrainer, alpha: number) => {
    trainer.step(PHYS, {
      b: B2,
      alpha,
      lr: 1e-3,
      source: "uploaded",
      genSeed: 1,
      applyDisc: false,
    });
    return trainer.readExtGrads();
  };
  const g0a = await run(lane0, 0.07);
  const g0b = await run(lane0, 0.93);
  const g1a = await run(lane1, 0.07);
  const g1b = await run(lane1, 0.93);
  const headMax = (g: Float32Array, head: number) => {
    let v = 0;
    for (const seg of layout.segments) {
      if (seg.head !== head) continue;
      for (let x = 0; x < seg.floatLength; x++) {
        v = Math.max(v, Math.abs(g[seg.floatOffset + x]));
      }
    }
    return v;
  };
  const maxDelta = (a: Float32Array, b: Float32Array) => {
    let v = 0;
    for (let i = 0; i < a.length; i++) v = Math.max(v, Math.abs(a[i] - b[i]));
    return v;
  };
  ok(headMax(g0a, 0) > 0 && headMax(g0a, 1) === 0,
    "lane A writes active head0 gradients and exact zeros to head1");
  ok(headMax(g1a, 1) > 0 && headMax(g1a, 0) === 0,
    "lane B writes active head1 gradients and exact zeros to head0");
  ok(maxDelta(g0a, g0b) === 0 && maxDelta(g1a, g1b) === 0,
    "direct lane ext gradients are bit-identical across blend-alpha changes");

  // Independent ownership: updating A's discriminator cannot mutate B's.
  const bBefore = await lane1.readAdvWeights();
  const aBefore = await lane0.readAdvWeights();
  lane0.step(PHYS, {
    b: B2, alpha: 0.5, lr: 3e-3, source: "uploaded",
    genSeed: 0, applyDisc: true,
  });
  const bAfter = await lane1.readAdvWeights();
  const aAfter = await lane0.readAdvWeights();
  ok(maxDelta(bBefore, bAfter) === 0,
    "training lane A leaves lane B predictor weights bit-identical");
  ok(maxDelta(aBefore, aAfter) > 0,
    "lane A predictor did update (independence gate is not vacuous)");

  // Same lane/data/predictor, opposite NAMED roles. The host supplies the same
  // non-negative magnitude; the trainer owns the mathematical sign.
  const disagree = makeLane(0, "disagree", 206);
  const agree = makeLane(0, "agree", 207);
  const roleAdv = makeAdvWeights(disagree.advL, 208).packed;
  disagree.uploadAdvWeights(roleAdv);
  agree.uploadAdvWeights(roleAdv);
  disagree.uploadTuples(tuples);
  agree.uploadTuples(tuples);
  const gd = await run(disagree, 0.3);
  const ga = await run(agree, 0.3);
  let signErr = 0, signScale = 0;
  for (let i = 0; i < gd.length; i++) {
    signErr = Math.max(signErr, Math.abs(ga[i] + gd[i]));
    signScale = Math.max(signScale, Math.abs(gd[i]));
  }
  ok(signErr / (signScale + 1e-30) < 2e-7,
    `agree extGrad = -disagree extGrad (rel ${(signErr / (signScale + 1e-30)).toExponential(2)})`);
  ok(
    disagree.genSeed(0.4, 99, B2) === agree.genSeed(0.4, 99, B2),
    "callers pass the same non-negative magnitude; named role owns the sign"
  );

  lane0.destroy();
  lane1.destroy();
  disagree.destroy();
  agree.destroy();
  fieldWBuf.destroy();
}

/* ══════════════════════════════════════════════════════════════════════════
   §3 tfjs end-to-end generator cross-check (pair + labelled quad, k=4)
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 3. tfjs end-to-end cross-check (pair + labelled quad, k=4) ---");
for (const tc of [
  { label: "pair", encoding: { tag: "pair" } as const, seed: 71 },
  { label: "quad-labelled", encoding: { tag: "quad-labelled" } as const, seed: 711 },
]) {
  const cfg: OracleCfg = {
    tag: tc.encoding.tag,
    k: 4,
    eps: 0.05,
    enc: { kind: "raw" },
    hidden: 8,
    feature: 8,
  };
  const { m } = tupleDims(cfg.tag);
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, tc.seed);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const trainer = new AdversaryTrainer(device, layout, {
    tag: cfg.tag, k: cfg.k, relaxEps: cfg.eps,
    observerGeometry: cfg.observerGeometry ?? "periodic",
    hiddenUnits: cfg.hidden, featureDim: cfg.feature,
    batchCap: 256, fieldWeightsBuffer: fieldWBuf, seed: tc.seed + 1,
  });
  const aw = makeAdvWeights(trainer.advL, tc.seed + 2);
  trainer.uploadAdvWeights(aw.packed);
  const tuples = makeTuples(B2, m, tc.seed + 3);
  trainer.uploadTuples(tuples);
  trainer.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 1e-3, source: "uploaded",
    genSeed: GEN_SEED, applyDisc: false,
  });
  const sl = advScratchLayout(layout, trainer.advL, cfg.tag);
  const scratch = await readBuf((trainer as any).scratchBuf, B2 * sl.stride);
  const extGrads = await trainer.readExtGrads();

  // tfjs field: two sequential heads with the SAME packed weights
  const mkHead = () => {
    const net = tf.sequential();
    net.add(tf.layers.dense({ units: 8, activation: "selu", inputShape: [2] }));
    net.add(tf.layers.dense({ units: 8, activation: "selu" }));
    net.add(tf.layers.dense({ units: 2, activation: "tanh" }));
    return net;
  };
  const gHead = mkHead();
  const rHead = mkHead();
  const setFrom = (net: tf.Sequential, h: number) => {
    const dims = fieldDims(2);
    const ws: tf.Tensor[] = [];
    dims.forEach((L, l) => {
      const seg = layout.segments.find((s) => s.head === h && s.layer === l && s.role === "kernel")!;
      const bseg = layout.segments.find((s) => s.head === h && s.layer === l && s.role === "bias")!;
      ws.push(tf.tensor2d(Array.from(fw.packed.slice(seg.floatOffset, seg.floatOffset + seg.floatLength)), [L.inSize, L.outSize]));
      ws.push(tf.tensor1d(Array.from(fw.packed.slice(bseg.floatOffset, bseg.floatOffset + bseg.floatLength))));
    });
    net.setWeights(ws);
  };
  setFrom(gHead, 0);
  setFrom(rHead, 1);

  // real Adversary with the SAME head weights (hidden 8 / feature 8)
  const adv = new Adversary({
    ...defaultAdversaryConfig(
      { tag: "wta", k: cfg.k, relaxEps: cfg.eps },
      tc.encoding,
      cfg.observerGeometry ?? "periodic"
    ),
    hiddenUnits: cfg.hidden,
    featureDim: cfg.feature,
    batchTuples: B2,
    seed: tc.seed + 4,
  });
  const advHeads = (adv as unknown as { heads: tf.Sequential[] }).heads;
  advHeads.forEach((net, j) => {
    const ws: tf.Tensor[] = [];
    trainer.advL.heads[j].layers.forEach((L) => {
      ws.push(tf.tensor2d(Array.from(aw.packed.slice(L.weightOffset, L.weightOffset + L.inSize * L.outSize)), [L.inSize, L.outSize]));
      ws.push(tf.tensor1d(Array.from(aw.packed.slice(L.biasOffset, L.biasOffset + L.outSize))));
    });
    net.setWeights(ws);
  });

  // member states → tensors (normalized pos, pixel vel)
  const N = B2 * m;
  const posArr = new Float32Array(N * 2);
  const velArr = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    posArr[2 * i] = tuples[i * 4] / PHYS.width;
    posArr[2 * i + 1] = tuples[i * 4 + 1] / PHYS.height;
    velArr[2 * i] = tuples[i * 4 + 2];
    velArr[2 * i + 1] = tuples[i * 4 + 3];
  }
  const idx = new Int32Array(N);
  for (let i = 0; i < N; i++) idx[i] = i;

  const posT = tf.tensor2d(posArr, [N, 2]);
  const velT = tf.tensor2d(velArr, [N, 2]);
  const resT = tf.tensor2d([[PHYS.width, PHYS.height]]);

  // u/y parity vs the REAL strict Adversary.encodeSignal (values)
  {
    const signal = tf.tidy(() => {
      const F = gHead.apply(posT) as tf.Tensor2D;
      const R = rHead.apply(posT) as tf.Tensor2D;
      return F.mul(1 - ALPHA).add(R.mul(ALPHA)) as tf.Tensor2D;
    });
    const s = adv.encodeSignal(posT, signal, idx);
    const uTf = s.u.dataSync();
    const yTf = s.y.dataSync();
    let mU = 0, mY = 0;
    for (let t = 0; t < B2; t++) {
      const sBase = t * sl.stride;
      for (let i = 0; i < sl.du; i++) {
        mU = Math.max(
          mU,
          Math.abs(uTf[t * sl.du + i] - scratch[sBase + sl.uOff + i])
        );
      }
      for (let i = 0; i < sl.dy; i++) {
        mY = Math.max(
          mY,
          Math.abs(yTf[t * sl.dy + i] - scratch[sBase + sl.yOff + i])
        );
      }
    }
    ok(mU < 1e-5 && mY < 1e-5,
      `${tc.label}: kernel u/y ≡ tfjs Adversary.encodeSignal ` +
        `(max |Δu| ${mU.toExponential(2)}, |Δy| ${mY.toExponential(2)})`);
    s.u.dispose(); s.y.dispose(); signal.dispose();
  }

  // generator gradient over the REAL field variables
  const varList = [gHead, rHead].flatMap((n) =>
    n.trainableWeights.map((wv) => (wv as unknown as { val: tf.Variable }).val)
  );
  const { value, grads } = tf.variableGrads(() => {
    const F = gHead.apply(posT) as tf.Tensor2D;
    const R = rHead.apply(posT) as tf.Tensor2D;
    const signal = F.mul(1 - ALPHA).add(R.mul(ALPHA)) as tf.Tensor2D;
    const s = adv.encodeSignal(posT, signal, idx);
    const payoff = adv.payoff(s);
    return payoff.mean().mul(-GEN_SEED * B2).asScalar();
  }, varList);

  const tfVec: number[] = [];
  varList.forEach((v) => {
    const gv = grads[v.name].dataSync();
    for (let i = 0; i < gv.length; i++) tfVec.push(gv[i]);
  });
  const gpuVec: number[] = [];
  // varList order = head0 (k0,b0,k1,b1,k2,b2) then head1 — same as segments
  for (const seg of layout.segments) {
    for (let x = 0; x < seg.floatLength; x++) gpuVec.push(extGrads[seg.floatOffset + x]);
  }
  const cosTf = cosine(tfVec, gpuVec);
  const scale = Math.max(...tfVec.map(Math.abs));
  let rel = 0;
  for (let i = 0; i < tfVec.length; i++) {
    rel = Math.max(rel, Math.abs(tfVec[i] - gpuVec[i]) / scale);
  }
  ok(cosTf > 0.99999 && rel < 1e-3,
    `${tc.label}: extGrads ≡ tfjs variableGrads over field vars ` +
      `(${tfVec.length} comps, cos=${cosTf.toFixed(7)}, scale-rel ${rel.toExponential(2)})`);

  value.dispose();
  Object.values(grads).forEach((t) => t.dispose());
  posT.dispose(); velT.dispose(); resT.dispose();
  adv.dispose(); gHead.dispose(); rHead.dispose();
  trainer.destroy();
  fieldWBuf.destroy();
}

/* ══════════════════════════════════════════════════════════════════════════
   §3b explicit objective + target parity: production WGSL ≡ real tfjs
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 3b. explicit loss/target production parity vs tfjs ---");
for (const tc of [
  {
    label: "force + soft-angle",
    encoding: { tag: "pair" } as const,
    target: { tag: "force" } as const,
    loss: { tag: "soft-angle", tau: 0.05 } as const,
    seed: 721,
  },
  {
    label: "force + angle-relative-scale",
    encoding: { tag: "pair" } as const,
    target: { tag: "force" } as const,
    loss: {
      tag: "angle-relative-scale", tau: 0.05, scaleWeight: 0.4,
      energyWeight: 0.2, energyTarget: 0.35,
    } as const,
    seed: 731,
  },
  {
    label: "force + angle-scale-hold",
    encoding: { tag: "pair" } as const,
    target: { tag: "force" } as const,
    loss: {
      tag: "angle-scale-hold", tau: 0.05, scaleWeight: 0.4,
      energyWeight: 0.2, energyTarget: 0.35,
    } as const,
    seed: 741,
  },
  {
    label: "post-velocity + soft-angle",
    encoding: { tag: "point" } as const,
    target: { tag: "post-velocity" } as const,
    loss: { tag: "soft-angle", tau: 0.05 } as const,
    seed: 751,
  },
]) {
  const { m } = tupleDims(tc.encoding.tag);
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, tc.seed);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const trainer = new AdversaryTrainer(device, layout, {
    tag: tc.encoding.tag,
    target: tc.target,
    loss: tc.loss,
    k: 3,
    relaxEps: 0.05,
    observerGeometry: "periodic",
    hiddenUnits: 8,
    featureDim: 8,
    batchCap: 256,
    fieldWeightsBuffer: fieldWBuf,
    seed: tc.seed + 1,
  });
  const aw = makeAdvWeights(trainer.advL, tc.seed + 2);
  trainer.uploadAdvWeights(aw.packed);
  const tuples = makeTuples(B2, m, tc.seed + 3);
  trainer.uploadTuples(tuples);
  trainer.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 1e-3, source: "uploaded",
    genSeed: GEN_SEED, applyDisc: false,
  });
  const sl = advScratchLayout(
    layout, trainer.advL, trainer.tag, trainer.target, trainer.loss
  );
  const [scratch, gpuAdv, gpuExt] = await Promise.all([
    readBuf((trainer as any).scratchBuf, B2 * sl.stride),
    trainer.readAdvGrads(),
    trainer.readExtGrads(),
  ]);

  const mkHead = () => {
    const net = tf.sequential();
    net.add(tf.layers.dense({ units: 8, activation: "selu", inputShape: [2] }));
    net.add(tf.layers.dense({ units: 8, activation: "selu" }));
    net.add(tf.layers.dense({ units: 2, activation: "tanh" }));
    return net;
  };
  const gHead = mkHead();
  const rHead = mkHead();
  const setFieldWeights = (net: tf.Sequential, h: number) => {
    const ws: tf.Tensor[] = [];
    fieldDims(2).forEach((L, l) => {
      const w = layout.segments.find(
        (s) => s.head === h && s.layer === l && s.role === "kernel"
      )!;
      const b = layout.segments.find(
        (s) => s.head === h && s.layer === l && s.role === "bias"
      )!;
      ws.push(tf.tensor2d(
        Array.from(fw.packed.slice(w.floatOffset, w.floatOffset + w.floatLength)),
        [L.inSize, L.outSize]
      ));
      ws.push(tf.tensor1d(
        Array.from(fw.packed.slice(b.floatOffset, b.floatOffset + b.floatLength))
      ));
    });
    net.setWeights(ws);
  };
  setFieldWeights(gHead, 0);
  setFieldWeights(rHead, 1);

  const adv = new Adversary({
    ...defaultAdversaryConfig(
      { tag: "wta", k: 3, relaxEps: 0.05 },
      tc.encoding,
      "periodic"
    ),
    target: tc.target,
    loss: tc.loss,
    hiddenUnits: 8,
    featureDim: 8,
    batchTuples: B2,
    seed: tc.seed + 4,
  });
  const advHeads = (adv as unknown as { heads: tf.Sequential[] }).heads;
  advHeads.forEach((net, j) => {
    const ws: tf.Tensor[] = [];
    trainer.advL.heads[j].layers.forEach((L) => {
      ws.push(tf.tensor2d(
        Array.from(
          aw.packed.slice(L.weightOffset, L.weightOffset + L.inSize * L.outSize)
        ),
        [L.inSize, L.outSize]
      ));
      ws.push(tf.tensor1d(
        Array.from(aw.packed.slice(L.biasOffset, L.biasOffset + L.outSize))
      ));
    });
    net.setWeights(ws);
  });

  const N = B2 * m;
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

  const fieldSignal = tf.tidy(() => {
    const a = gHead.apply(posT) as tf.Tensor2D;
    const b = rHead.apply(posT) as tf.Tensor2D;
    return a.mul(1 - ALPHA).add(b.mul(ALPHA)) as tf.Tensor2D;
  });
  let clipUnsat = 0;
  let clipSat = 0;
  const sample =
    tc.target.tag === "force"
      ? adv.encodeTarget({ tag: "force", pos: posT, force: fieldSignal }, idx)
      : (() => {
          const next = tf.tidy(() => {
            const pre = velT
              .add(fieldSignal.mul(PHYS.forceMagnitude))
              .mul(PHYS.friction);
            return pre
              .clipByValue(-PHYS.maxVelocity, PHYS.maxVelocity)
              .div(PHYS.maxVelocity) as tf.Tensor2D;
          });
          const f = fieldSignal.dataSync();
          for (let i = 0; i < N * 2; i++) {
            const pre =
              (velArr[i] + PHYS.forceMagnitude * f[i]) * PHYS.friction;
            if (Math.abs(pre) < PHYS.maxVelocity) clipUnsat++;
            else clipSat++;
          }
          const vNorm = velT.div(PHYS.maxVelocity) as tf.Tensor2D;
          const s = adv.encodeTarget({
            tag: "post-velocity", pos: posT, velocity: vNorm,
            nextVelocity: next,
          }, idx);
          vNorm.dispose();
          next.dispose();
          return s;
        })();

  const uTf = sample.u.dataSync();
  const yTf = sample.y.dataSync();
  const scaleTf = sample.relativeScale?.dataSync();
  let maxU = 0;
  let maxY = 0;
  for (let t = 0; t < B2; t++) {
    const base = t * sl.stride;
    for (let i = 0; i < sl.du; i++) {
      maxU = Math.max(maxU, Math.abs(
        uTf[t * sl.du + i] - scratch[base + sl.uOff + i]
      ));
    }
    for (let i = 0; i < sl.vectorDy; i++) {
      maxY = Math.max(maxY, Math.abs(
        yTf[t * sl.vectorDy + i] - scratch[base + sl.yOff + i]
      ));
    }
    for (let q = 0; q < sl.scaleDy; q++) {
      maxY = Math.max(maxY, Math.abs(
        scaleTf![t * sl.scaleDy + q] -
          scratch[base + sl.yOff + sl.vectorDy + q]
      ));
    }
  }
  ok(
    maxU < 2e-5 && maxY < 2e-5,
    `${tc.label}: context + objective target parity ` +
      `(max |Δu| ${maxU.toExponential(2)}, |Δy| ${maxY.toExponential(2)})`
  );
  // Pin the exact soft-spherical discriminator cotangent at the production
  // scratch seam before reducing it into packed weight gradients.
  let maxOutputDelta = 0;
  const last = trainer.advL.heads[0].layers.length - 1;
  const aOut = sl.advAOff[last];
  const dOut = sl.advDOff[last];
  const loserW = 0.05 / 2;
  for (let t = 0; t < B2; t++) {
    const base = t * sl.stride;
    const winner = Math.trunc(scratch[base + sl.winOff]);
    for (let j = 0; j < 3; j++) {
      const ab = base + sl.advOff + j * sl.advBlk;
      const wj = j === winner ? 0.95 : loserW;
      for (let v = 0; v < sl.vectorDy / 2; v++) {
        const px = scratch[ab + aOut + 2 * v];
        const py = scratch[ab + aOut + 2 * v + 1];
        const yx = scratch[base + sl.yOff + 2 * v];
        const yy = scratch[base + sl.yOff + 2 * v + 1];
        const tau = tc.loss.tau;
        const rp = Math.hypot(px, py, tau);
        const ry = Math.hypot(yx, yy, tau);
        const ep = [px / rp, py / rp, tau / rp];
        const ey = [yx / ry, yy / ry, tau / ry];
        const d = [ep[0] - ey[0], ep[1] - ey[1], ep[2] - ey[2]];
        const den = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + 1e-12);
        const gd = [d[0] / den, d[1] / den, d[2] / den];
        const proj = ep[0] * gd[0] + ep[1] * gd[1] + ep[2] * gd[2];
        const seed = (1 / B2) * wj / (sl.vectorDy / 2);
        const ex = seed * (gd[0] - ep[0] * proj) / rp;
        const eyGrad = seed * (gd[1] - ep[1] * proj) / rp;
        maxOutputDelta = Math.max(
          maxOutputDelta,
          Math.abs(ex - scratch[ab + dOut + 2 * v]),
          Math.abs(eyGrad - scratch[ab + dOut + 2 * v + 1])
        );
      }
    }
  }
  ok(
    maxOutputDelta < 2e-6,
    `${tc.label}: production D output cotangent is exact S² derivative ` +
      `(max |Δ| ${maxOutputDelta.toExponential(2)})`
  );
  if (tc.target.tag === "post-velocity") {
    ok(
      clipUnsat > 0 && clipSat > 0,
      `${tc.label}: fixture covers both clip Jacobian branches ` +
        `(${clipUnsat} unsaturated, ${clipSat} saturated components)`
    );
  }

  const tfResid = adv.residuals(sample);
  const tfWinners = tf.argMin(tfResid, 1).dataSync();
  const tfPayoff = adv.payoff(sample).dataSync();
  let winnerMismatch = 0;
  let payoffDelta = 0;
  for (let t = 0; t < B2; t++) {
    const base = t * sl.stride;
    if (tfWinners[t] !== Math.trunc(scratch[base + sl.winOff])) {
      winnerMismatch++;
    }
    payoffDelta = Math.max(
      payoffDelta,
      Math.abs(tfPayoff[t] - scratch[base + sl.surOff])
    );
  }
  ok(
    winnerMismatch === 0 && payoffDelta < 3e-5,
    `${tc.label}: production residual/winner ≡ tfjs ` +
      `(winner mismatch ${winnerMismatch}/${B2}, max |Δpayoff| ${payoffDelta.toExponential(2)})`
  );
  tfResid.dispose();

  const advVars = advHeads.flatMap((n) =>
    n.trainableWeights.map((wv) => (wv as unknown as { val: tf.Variable }).val)
  );
  const dRun = tf.variableGrads(
    () => adv.payoff(sample).mean().asScalar(),
    advVars
  );
  const tfAdv: number[] = [];
  advVars.forEach((v) => {
    const a = dRun.grads[v.name].dataSync();
    for (let i = 0; i < a.length; i++) tfAdv.push(a[i]);
  });
  // AdversaryLayout 4-float-aligns every segment. tfjs has no padding, so pack
  // its dense variable stream into the production offsets before comparing.
  const tfAdvPacked = new Float32Array(trainer.advL.totalFloats);
  let tfAdvCursor = 0;
  for (const seg of trainer.advL.segments) {
    for (let i = 0; i < seg.floatLength; i++) {
      tfAdvPacked[seg.floatOffset + i] = tfAdv[tfAdvCursor++];
    }
  }
  ok(
    tfAdvCursor === tfAdv.length,
    `${tc.label}: tfjs predictor gradients pack every non-padding float`
  );
  const advCos = cosine(tfAdvPacked, gpuAdv);
  const advScale = Math.max(...tfAdvPacked.map(Math.abs), 1e-30);
  let advRel = 0;
  for (let i = 0; i < tfAdvPacked.length; i++) {
    advRel = Math.max(
      advRel, Math.abs(tfAdvPacked[i] - gpuAdv[i]) / advScale
    );
  }
  ok(
    advCos > 0.99999 && advRel < 2e-3,
    `${tc.label}: discriminator gradient WGSL ≡ tfjs ` +
      `(cos ${advCos.toFixed(7)}, scale-rel ${advRel.toExponential(2)})`
  );

  const fieldVars = [gHead, rHead].flatMap((n) =>
    n.trainableWeights.map((wv) => (wv as unknown as { val: tf.Variable }).val)
  );
  const gRun = tf.variableGrads(() => tf.tidy(() => {
    const a = gHead.apply(posT) as tf.Tensor2D;
    const b = rHead.apply(posT) as tf.Tensor2D;
    const force = a.mul(1 - ALPHA).add(b.mul(ALPHA)) as tf.Tensor2D;
    const s =
      tc.target.tag === "force"
        ? adv.encodeTarget({ tag: "force", pos: posT, force }, idx)
        : (() => {
            const velocity = velT.div(PHYS.maxVelocity) as tf.Tensor2D;
            const nextVelocity = velT
              .add(force.mul(PHYS.forceMagnitude))
              .mul(PHYS.friction)
              .clipByValue(-PHYS.maxVelocity, PHYS.maxVelocity)
              .div(PHYS.maxVelocity) as tf.Tensor2D;
            return adv.encodeTarget({
              tag: "post-velocity", pos: posT, velocity, nextVelocity,
            }, idx);
          })();
    return adv.generatorLoss(s).mul(GEN_SEED * B2).asScalar();
  }), fieldVars);
  const tfExt: number[] = [];
  fieldVars.forEach((v) => {
    const a = gRun.grads[v.name].dataSync();
    for (let i = 0; i < a.length; i++) tfExt.push(a[i]);
  });
  const gpuExtPacked: number[] = [];
  for (const seg of layout.segments) {
    for (let i = 0; i < seg.floatLength; i++) {
      gpuExtPacked.push(gpuExt[seg.floatOffset + i]);
    }
  }
  const extCos = cosine(tfExt, gpuExtPacked);
  const extScale = Math.max(...tfExt.map(Math.abs), 1e-30);
  let extRel = 0;
  for (let i = 0; i < tfExt.length; i++) {
    extRel = Math.max(
      extRel, Math.abs(tfExt[i] - gpuExtPacked[i]) / extScale
    );
  }
  ok(
    extCos > 0.99999 && extRel < 3e-3,
    `${tc.label}: generator gradient WGSL ≡ tfjs ` +
      `(cos ${extCos.toFixed(7)}, scale-rel ${extRel.toExponential(2)})`
  );

  dRun.value.dispose();
  Object.values(dRun.grads).forEach((t) => t.dispose());
  gRun.value.dispose();
  Object.values(gRun.grads).forEach((t) => t.dispose());
  disposeTupleSample(sample);
  fieldSignal.dispose();
  posT.dispose();
  velT.dispose();
  adv.dispose();
  gHead.dispose();
  rHead.dispose();
  trainer.destroy();
  fieldWBuf.destroy();
}

/* ══════════════════════════════════════════════════════════════════════════
   §4 discriminator trains fused
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 4. alternating update order + discriminator training ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 191);
  const bufA = mkStorage(layout.totalFloats * 4);
  const bufB = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(bufA, 0, fw.packed);
  device.queue.writeBuffer(bufB, 0, fw.packed);
  const mk = (buf: GPUBuffer) => new AdversaryTrainer(device, layout, {
    tag: "pair-rotation", k: 4, relaxEps: 0.05,
    observerGeometry: "periodic",
    hiddenUnits: 8, featureDim: 8, batchCap: 256,
    fieldWeightsBuffer: buf, seed: 192,
  });
  const a = mk(bufA);
  const b = mk(bufB);
  const aw = makeAdvWeights(a.advL, 193).packed;
  const tuples = makeTuples(B2, 2, 194);
  a.uploadAdvWeights(aw); b.uploadAdvWeights(aw);
  a.uploadTuples(tuples); b.uploadTuples(tuples);

  // One fused alternating step must equal the explicit sequence
  // discriminator-update-only, then generator-evaluation-only.
  a.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 2e-3, source: "uploaded",
    genSeed: GEN_SEED, applyDisc: true,
  });
  b.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 2e-3, source: "uploaded",
    genSeed: 0, applyDisc: true,
  });
  b.step(PHYS, {
    b: B2, alpha: ALPHA, lr: 2e-3, source: "uploaded",
    genSeed: GEN_SEED, applyDisc: false,
  });
  const ga = await a.readExtGrads();
  const gb = await b.readExtGrads();
  let maxDiff = 0, scale = 0;
  for (let i = 0; i < ga.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(ga[i] - gb[i]));
    scale = Math.max(scale, Math.abs(gb[i]));
  }
  ok(maxDiff / (scale + 1e-30) < 2e-6,
    `fused G uses post-D-update weights (explicit alternating rel ${(maxDiff / (scale + 1e-30)).toExponential(2)})`);
  a.destroy(); b.destroy(); bufA.destroy(); bufB.destroy();
}

{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 41);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);
  const trainer = new AdversaryTrainer(device, layout, {
    tag: "pair", k: 4, relaxEps: 0.05, hiddenUnits: 32, featureDim: 16,
    observerGeometry: "periodic",
    batchCap: 256, fieldWeightsBuffer: fieldWBuf, seed: 42,
  });
  trainer.uploadTuples(makeTuples(256, 2, 43));
  const losses: number[] = [];
  for (let s = 0; s < 30; s++) {
    trainer.step(PHYS, {
      b: 256, alpha: ALPHA, lr: 3e-3, source: "uploaded", genSeed: 0, applyDisc: true,
    });
    losses.push((await trainer.readStats()).discLoss);
  }
  let decreasingSteps = 0;
  for (let i = 1; i < losses.length; i++) if (losses[i] < losses[i - 1]) decreasingSteps++;
  ok(
    losses[29] < losses[0],
    `loss decreased over 30 steps: ${losses[0].toFixed(6)} → ${losses[29].toFixed(6)} (${decreasingSteps}/29 steps decreasing)`
  );
  ok(losses[29] < 0.5 * losses[0],
    `loss more than halved on the fixed batch (${(losses[29] / losses[0]).toFixed(3)}×)`);
  trainer.destroy();
  fieldWBuf.destroy();
}

/* ══════════════════════════════════════════════════════════════════════════
   §5 structural separation + the extGrad seam
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 5. separation: adversary ⊥ field weights + extGrad seam ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 51);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);

  const adv = new AdversaryTrainer(device, layout, {
    tag: "pair", k: 4, relaxEps: 0.05, hiddenUnits: 16, featureDim: 8,
    observerGeometry: "periodic",
    batchCap: 256, fieldWeightsBuffer: fieldWBuf, seed: 52,
  });
  adv.uploadTuples(makeTuples(128, 2, 53));

  // (a) adversary-only training leaves FIELD weights bit-identical
  const fieldBefore = await readBuf(fieldWBuf, layout.totalFloats);
  for (let s = 0; s < 5; s++) {
    adv.step(PHYS, { b: 128, alpha: ALPHA, lr: 3e-3, source: "uploaded", genSeed: 1.0, applyDisc: true });
  }
  const fieldAfter = await readBuf(fieldWBuf, layout.totalFloats);
  let fieldBitIdentical = true;
  for (let i = 0; i < layout.totalFloats; i++) {
    if (fieldBefore[i] !== fieldAfter[i]) fieldBitIdentical = false;
  }
  ok(fieldBitIdentical, "5 adversary train steps (disc Adam + fieldGrad) → field weights BIT-IDENTICAL");

  // (b) field training (with the extGrad seam ACTIVE) leaves ADVERSARY weights
  //     bit-identical
  const fieldTrainer = new FusedTrainer(device, layout, {
    batchCap: 256, weightsBuffer: fieldWBuf, extGradBuffer: adv.extGradsBuf,
  });
  const batch = new Float32Array(2 * 128);
  {
    const rnd = mulberry32(54);
    for (let i = 0; i < 128; i++) {
      batch[2 * i] = rnd() * PHYS.width;
      batch[2 * i + 1] = rnd() * PHYS.height;
    }
  }
  fieldTrainer.uploadBatch(batch);
  const advBefore = await adv.readAdvWeights();
  for (let s = 0; s < 5; s++) {
    fieldTrainer.step(PHYS, { n: 128, alpha: ALPHA, lr: 0.01, source: "uploaded", apply: true });
  }
  const advAfter = await adv.readAdvWeights();
  let advBitIdentical = true;
  for (let i = 0; i < adv.advL.totalFloats; i++) {
    if (advBefore[i] !== advAfter[i]) advBitIdentical = false;
  }
  ok(advBitIdentical, "5 field train steps (extGrad seam bound) → adversary weights BIT-IDENTICAL");
  const fieldMoved = (await readBuf(fieldWBuf, layout.totalFloats)).some((v, i) => v !== fieldAfter[i]);
  ok(fieldMoved, "…and the field DID train (weights moved — the seam is not a dead path)");

  // (c) the extGrad seam adds exactly dL_gen/dW: grads(with) − grads(without) ≈ extGrads
  const trainerNoExt = new FusedTrainer(device, layout, { batchCap: 256 });
  trainerNoExt.uploadWeights(await readBuf(fieldWBuf, layout.totalFloats));
  trainerNoExt.uploadBatch(batch);
  adv.step(PHYS, { b: 128, alpha: ALPHA, lr: 3e-3, source: "uploaded", genSeed: 0.7, applyDisc: false });
  const ext = await adv.readExtGrads();
  fieldTrainer.step(PHYS, { n: 128, alpha: ALPHA, lr: 0.01, source: "uploaded", apply: false });
  const gWith = await fieldTrainer.readGrads();
  trainerNoExt.step(PHYS, { n: 128, alpha: ALPHA, lr: 0.01, source: "uploaded", apply: false });
  const gWithout = await trainerNoExt.readGrads();
  const diff = Array.from(gWith, (v, i) => v - gWithout[i]);
  const cosSeam = cosine(diff, ext);
  const scale = Math.max(...Array.from(ext, Math.abs));
  let relSeam = 0;
  for (let i = 0; i < ext.length; i++) {
    relSeam = Math.max(relSeam, Math.abs(diff[i] - ext[i]) / (scale + 1e-30));
  }
  ok(cosSeam > 0.9999 && relSeam < 1e-3,
    `extGrad seam: grads(with) − grads(without) ≡ extGrads (cos=${cosSeam.toFixed(6)}, scale-rel ${relSeam.toExponential(2)})`);

  adv.destroy();
  fieldTrainer.destroy();
  trainerNoExt.destroy();
  fieldWBuf.destroy();
}

/* ══════════════════════════════════════════════════════════════════════════
   §6 deterministic unique rotating live-particle coverage
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 6. deterministic unique live coverage + zero-data safety ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 601);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);

  const particleBuffers = (n: number, seed: number) => {
    const posBuf = mkStorage(n * 8);
    const velBuf = mkStorage(n * 8);
    const tuples = makeTuples(n, 1, seed);
    const pos = new Float32Array(n * 2);
    const vel = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      pos[2 * i] = tuples[4 * i];
      pos[2 * i + 1] = tuples[4 * i + 1];
      vel[2 * i] = tuples[4 * i + 2];
      vel[2 * i + 1] = tuples[4 * i + 3];
    }
    device.queue.writeBuffer(posBuf, 0, pos);
    device.queue.writeBuffer(velBuf, 0, vel);
    return { posBuf, velBuf };
  };
  const opts = {
    b: 8,
    alpha: ALPHA,
    lr: 2e-3,
    source: "particles" as const,
    genSeed: 0.4,
    applyDisc: true,
  };

  // N=11, m=3: requested B=8 is capped to effectiveB=floor(11/3)=3.
  const tri = new AdversaryTrainer(device, layout, {
    tag: "tri",
    observerGeometry: "periodic",
    k: 3,
    relaxEps: 0.05,
    hiddenUnits: 8,
    featureDim: 6,
    batchCap: 8,
    fieldWeightsBuffer: fieldWBuf,
    particleCount: 11,
    seed: 602,
  });
  const p11 = particleBuffers(11, 603);
  tri.setParticleBuffers(p11.posBuf, p11.velBuf, 11);
  const triSl = advScratchLayout(layout, tri.advL, "tri");

  tri.step(PHYS, opts);
  let scratch = await tri.readScratch(triSl.stride * 3);
  const indices = (s: Float32Array, b: number) => {
    const out: number[] = [];
    for (let q = 0; q < b; q++) {
      const base = q * triSl.stride + triSl.idxOff;
      for (let t = 0; t < 3; t++) out.push(s[base + t]);
    }
    return out;
  };
  const first = indices(scratch, 3);
  let cov = tri.surpriseCoverage();
  ok(
    first.join(",") === "0,1,2,3,4,5,6,7,8" &&
      new Set(first).size === first.length,
    `N=11,m=3,B=8 → effectiveB=3 with unique first window [${first.join(",")}]`
  );
  ok(
    cov.coveredCount === 9 &&
      Math.abs(cov.covered - 9 / 11) < 1e-12 &&
      cov.cursor === 9 &&
      cov.window.start === 0 &&
      cov.window.count === 9 &&
      cov.window.generation === 1,
    `coverage after first step is exactly 9/11 (cursor ${cov.cursor}, generation ${cov.window.generation})`
  );

  tri.step(PHYS, opts);
  scratch = await tri.readScratch(triSl.stride * 3);
  const second = indices(scratch, 3);
  cov = tri.surpriseCoverage();
  ok(
    second.join(",") === "9,10,0,1,2,3,4,5,6" &&
      new Set(second).size === second.length,
    `wrapped window stays unique [${second.join(",")}]`
  );
  ok(
    cov.covered === 1 &&
      cov.coveredCount === 11 &&
      cov.cursor === 7 &&
      cov.window.start === 9 &&
      cov.window.count === 9 &&
      cov.window.generation === 2,
    "second live step reaches truthful 100% cumulative coverage"
  );

  // Uploaded fixtures keep their historical s*m+t indexing and must not move
  // the live cursor/generation.
  const beforeUploaded = tri.surpriseCoverage();
  tri.uploadTuples(makeTuples(2, 3, 604));
  tri.step(PHYS, {
    ...opts,
    b: 2,
    source: "uploaded",
    applyDisc: false,
  });
  await tri.readStats();
  const afterUploaded = tri.surpriseCoverage();
  ok(
    JSON.stringify(afterUploaded) === JSON.stringify(beforeUploaded),
    "uploaded-source verification step leaves live coverage exactly unchanged"
  );

  // Resize/rebind resets the cursor/generation and explicitly clears retained
  // values even when shrinking without reallocating the surprise buffer.
  const p5 = particleBuffers(5, 605);
  tri.setParticleBuffers(p5.posBuf, p5.velBuf, 5);
  const reset5 = tri.surpriseCoverage();
  const [cleared5Raw, cleared5Unit] = await Promise.all([
    tri.readSurprise(5, "raw-payoff"),
    tri.readSurprise(5, "per-unit-signal"),
  ]);
  ok(
    reset5.covered === 0 &&
      reset5.cursor === 0 &&
      reset5.window.count === 0 &&
      reset5.window.generation === 0 &&
      Array.from(cleared5Raw).every((v) => v === 0) &&
      Array.from(cleared5Unit).every((v) => v === 0),
    "shrink/rebind resets exact coverage and clears both surprise planes"
  );
  // Identical binding is a no-op, not a spurious reset.
  tri.step(PHYS, opts);
  await tri.readStats();
  const beforeSameBinding = tri.surpriseCoverage();
  tri.setParticleBuffers(p5.posBuf, p5.velBuf, 5);
  ok(
    JSON.stringify(tri.surpriseCoverage()) === JSON.stringify(beforeSameBinding),
    "repeating the identical particle binding preserves coverage"
  );

  const p13 = particleBuffers(13, 606);
  tri.setParticleBuffers(p13.posBuf, p13.velBuf, 13);
  const [cleared13Raw, cleared13Unit] = await Promise.all([
    tri.readSurprise(13, "raw-payoff"),
    tri.readSurprise(13, "per-unit-signal"),
  ]);
  ok(
    tri.surpriseCoverage().covered === 0 &&
      Array.from(cleared13Raw).every((v) => v === 0) &&
      Array.from(cleared13Unit).every((v) => v === 0),
    "grow/rebind starts a zeroed two-plane coverage epoch"
  );

  // N<m is an explicit no-data step: no Adam advance/weight mutation, and all
  // gradient/stat products are cleared instead of leaking the previous batch.
  const zero = new AdversaryTrainer(device, layout, {
    tag: "tri",
    observerGeometry: "periodic",
    k: 2,
    relaxEps: 0.05,
    hiddenUnits: 8,
    featureDim: 6,
    batchCap: 8,
    fieldWeightsBuffer: fieldWBuf,
    particleCount: 2,
    seed: 607,
  });
  zero.uploadTuples(makeTuples(2, 3, 608));
  zero.step(PHYS, {
    ...opts,
    b: 2,
    source: "uploaded",
    genSeed: 1,
    applyDisc: false,
  });
  const extBeforeZero = await zero.readExtGrads();
  await zero.readStats(); // seed the RMS EMA so the no-data gate is non-vacuous
  ok(
    Array.from(extBeforeZero).some((v) => v !== 0),
    "zero-data fixture first seeds a demonstrably nonzero external gradient"
  );
  const p2 = particleBuffers(2, 609);
  zero.setParticleBuffers(p2.posBuf, p2.velBuf, 2);
  const weightsBeforeZero = await zero.readAdvWeights();
  const scaleBeforeZero = zero.rewardScaleState();
  zero.step(PHYS, { ...opts, genSeed: 1, applyDisc: true });
  const [extAfterZero, advGradAfterZero, weightsAfterZero, zeroStats] = await Promise.all([
    zero.readExtGrads(),
    zero.readAdvGrads(),
    zero.readAdvWeights(),
    zero.readStats(),
  ]);
  ok(
    Array.from(extAfterZero).every((v) => v === 0) &&
      Array.from(advGradAfterZero).every((v) => v === 0) &&
      zeroStats.discLoss === 0 &&
      zeroStats.surprise === 0 &&
      zeroStats.batchRms === 0 &&
      zeroStats.headSpread.tag === "spread" &&
      zeroStats.headSpread.meanPair === 0 &&
      zeroStats.headSpread.minPair === 0 &&
      zeroStats.winCounts.every((v) => v === 0),
    "N<m clears external/adversary gradients and finalized stats"
  );
  ok(
    Array.from(weightsAfterZero).every((v, i) => v === weightsBeforeZero[i]) &&
      JSON.stringify(zero.rewardScaleState()) === JSON.stringify(scaleBeforeZero) &&
      zero.surpriseCoverage().covered === 0,
    "N<m leaves predictor weights, Adam/RMS epoch, and coverage untouched"
  );

  // Agree and disagree own distinct predictors but must see the identical
  // rotating tuple sequence, including after a resize.
  const live = particleBuffers(17, 610);
  const agreeA = new AdversaryTrainer(device, layout, {
    tag: "quad-labelled",
    observerGeometry: "periodic",
    fieldLane: 0,
    generatorRole: "disagree",
    k: 3,
    relaxEps: 0.05,
    hiddenUnits: 8,
    featureDim: 6,
    batchCap: 8,
    fieldWeightsBuffer: fieldWBuf,
    particleCount: 17,
    seed: 611,
  });
  const agreeB = new AdversaryTrainer(device, layout, {
    tag: "quad-labelled",
    observerGeometry: "periodic",
    fieldLane: 1,
    generatorRole: "agree",
    k: 3,
    relaxEps: 0.05,
    hiddenUnits: 8,
    featureDim: 6,
    batchCap: 8,
    fieldWeightsBuffer: fieldWBuf,
    particleCount: 17,
    seed: 612,
  });
  agreeA.setParticleBuffers(live.posBuf, live.velBuf, 17);
  agreeB.setParticleBuffers(live.posBuf, live.velBuf, 17);
  const quadSl = advScratchLayout(layout, agreeA.advL, "quad-labelled");
  const quadIndices = (s: Float32Array, b: number) => {
    const out: number[] = [];
    for (let q = 0; q < b; q++) {
      const base = q * quadSl.stride + quadSl.idxOff;
      out.push(s[base], s[base + 1], s[base + 2], s[base + 3]);
    }
    return out;
  };
  for (let turn = 0; turn < 3; turn++) {
    agreeA.step(PHYS, opts);
    agreeB.step(PHYS, opts);
    const [sa, sb] = await Promise.all([
      agreeA.readScratch(quadSl.stride * 4),
      agreeB.readScratch(quadSl.stride * 4),
    ]);
    ok(
      quadIndices(sa, 4).join(",") === quadIndices(sb, 4).join(",") &&
        JSON.stringify(agreeA.surpriseCoverage()) ===
          JSON.stringify(agreeB.surpriseCoverage()),
      `Agree A/B quad windows remain lockstep at generation ${turn + 1}`
    );
  }
  const live7 = particleBuffers(7, 613);
  agreeA.setParticleBuffers(live7.posBuf, live7.velBuf, 7);
  agreeB.setParticleBuffers(live7.posBuf, live7.velBuf, 7);
  ok(
    JSON.stringify(agreeA.surpriseCoverage()) ===
      JSON.stringify(agreeB.surpriseCoverage()) &&
      agreeA.surpriseCoverage().covered === 0,
    "Agree A/B coverage resets together on resize"
  );

  tri.destroy();
  zero.destroy();
  agreeA.destroy();
  agreeB.destroy();
  for (const p of [p11, p5, p13, p2, live, live7]) {
    p.posBuf.destroy();
    p.velBuf.destroy();
  }
  fieldWBuf.destroy();
}

/* ══════════════════════════════════════════════════════════════════════════
   §7 bench — B=512, tfjs-default head sizes; the tfjs number is 19-32 ms
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 7. bench: fused adversary step, B=512, live-particle source ---");
{
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = makeFieldWeights(layout, 61);
  const fieldWBuf = mkStorage(layout.totalFloats * 4);
  device.queue.writeBuffer(fieldWBuf, 0, fw.packed);

  const NPART = 200000;
  const posBuf = mkStorage(NPART * 8);
  const velBuf = mkStorage(NPART * 8);
  {
    const rnd = mulberry32(62);
    const pos = new Float32Array(NPART * 2);
    const vel = new Float32Array(NPART * 2);
    for (let i = 0; i < NPART; i++) {
      pos[2 * i] = rnd() * PHYS.width;
      pos[2 * i + 1] = rnd() * PHYS.height;
      vel[2 * i] = (rnd() * 2 - 1) * 10;
      vel[2 * i + 1] = (rnd() * 2 - 1) * 10;
    }
    device.queue.writeBuffer(posBuf, 0, pos);
    device.queue.writeBuffer(velBuf, 0, vel);
  }

  async function bench(label: string, tag: TupleTag, k: number): Promise<void> {
    const trainer = new AdversaryTrainer(device, layout, {
      tag, k, relaxEps: 0.05, hiddenUnits: 32, featureDim: 16,
      observerGeometry: "periodic",
      batchCap: 512, fieldWeightsBuffer: fieldWBuf, particleCount: NPART, seed: 63,
    });
    trainer.setParticleBuffers(posBuf, velBuf, NPART);
    const opts = (s: number) => ({
      b: 512, alpha: ALPHA, lr: 3e-3, seed: s, source: "particles" as const,
      genSeed: 1.0, applyDisc: true,
    });
    // warmup (pipeline compile)
    for (let s = 0; s < 5; s++) trainer.step(PHYS, opts(s));
    await trainer.readStats();
    const ITER = 50;
    const t0 = performance.now();
    for (let s = 0; s < ITER; s++) trainer.step(PHYS, opts(100 + s));
    await trainer.readStats(); // syncs the queue
    const ms = (performance.now() - t0) / ITER;
    const st = trainer.lastStats!;
    console.log(
      `      ${label}: ${ms.toFixed(3)} ms/step (B=512, ${ITER} steps, incl. disc Adam + fieldGrad) — ` +
        `disc ${st.discLoss.toFixed(4)}, sur ${st.surprise.toExponential(2)}, wins [${st.winCounts.join(",")}]`
    );
    ok(ms < 19, `${label}: fused step ${ms.toFixed(3)} ms < tfjs's best 19 ms`);
    ok(Number.isFinite(st.discLoss) && Number.isFinite(st.surprise) && st.winCounts.reduce((a, b) => a + b, 0) === 512,
      `${label}: stats finite, win counts sum to B`);
    trainer.destroy();
  }
  await bench("pair k=4 (32/16)", "pair", 4);
  await bench("tri  k=6 (32/16)", "tri", 6);
  await bench("quad-labelled k=4 (32/16)", "quad-labelled", 4);

  posBuf.destroy();
  velBuf.destroy();
  fieldWBuf.destroy();
}

console.log(failures === 0 ? "\nALL FUSED WTA CHECKS PASS" : `\n${failures} FUSED WTA CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
