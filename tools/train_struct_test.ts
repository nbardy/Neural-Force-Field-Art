/**
 * FUSED W_STRUCT (normalized structure) — parity vs a tfjs autograd oracle.
 * Real Metal via bun-webgpu. GPU suites are SEQUENTIAL — run nothing else.
 *
 *   bun tools/train_struct_test.ts
 *
 * WHAT IS UNDER TEST. `FieldLossSpec.W_STRUCT` compiles a batch-coupled term
 * into the field trainer's pass A: the batch's MEAN force vector is reduced
 * into a second partials block, finalize turns it into
 *
 *     L = (‖mean Fs‖² + ε) / (mean ‖Fs‖² + ε),   Fs = F(x)·forceMagnitude
 *
 * plus three broadcast gradient scalars, and bwd adds ∂L/∂Fs_i into the same
 * `dFs` the isotropy term uses. The claim gated here is that the resulting
 * weight gradient is, to f32, `tf.variableGrads` of `constantModeFraction` over
 * a real `HelmholtzField` — including the two degenerate fields where the ε
 * placement decides whether the term is a structure prior or a field-killer.
 *
 * Sections (each falsifiable):
 *  §1 the ε constants of the two implementations are literally the same number,
 *     and `W_STRUCT: 0` emits ZERO structure text (the byte-identity property).
 *  §2 parity sweep vs tfjs across raw / fourier / hashgrid encodings, with
 *     W_STRUCT alone and superposed with W_DIV.
 *  §3 degenerate fields in CLOSED FORM:
 *       - F ≡ const ≠ 0  ⇒  dc² = ms  ⇒  L = 1 exactly (up to ε), ac = 0;
 *       - F ≡ 0          ⇒  L = ε/ε = 1, and the gradient must be FINITE
 *                           (the ε-in-denominator-only variant returns 0 here,
 *                           i.e. a perfect score for a dead field).
 *  §4 the amplitude invariance that is the whole point: scaling forceMagnitude
 *     by c leaves L unchanged, whereas raw AC scales by c.
 */
import { setupGlobals } from "bun-webgpu";
import * as tf from "@tensorflow/tfjs";
import {
  layoutField,
  type FieldLayout,
  type LayerDims,
  type Encoding,
} from "../src/render/webgpu/advect_wgsl";
import {
  trainPassAShader,
  STRUCT_EPS,
  type FieldLossSpec,
} from "../src/render/webgpu/train_wgsl";
import { FusedTrainer } from "../src/render/webgpu/train";
import { HelmholtzField } from "../src/core/field/helmholtz";
import {
  constantModeFraction,
  isotropyLoss,
  STRUCTURE_EPS,
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
function relMax(ref: ArrayLike<number>, got: ArrayLike<number>): number {
  let scale = 1e-30, rel = 0;
  for (let i = 0; i < ref.length; i++) scale = Math.max(scale, Math.abs(ref[i]));
  for (let i = 0; i < ref.length; i++) {
    rel = Math.max(rel, Math.abs(ref[i] - got[i]) / scale);
  }
  return rel;
}

const HID = 12;
const PHYS = {
  width: 800, height: 600,
  forceMagnitude: 24, friction: 0.985, maxVelocity: 26,
};
const ALPHA = 0.55;
const N = 96;

const fieldDims = (inDim: number): LayerDims[] => [
  { inSize: inDim, outSize: HID, activation: "selu" },
  { inSize: HID, outSize: HID, activation: "selu" },
  { inSize: HID, outSize: 2, activation: "tanh" },
];

const BASE: FieldLossSpec = {
  W_CHAOS: 0, W_ISO: 0, W_DIV: 0, W_SPIRAL: 0,
  W_COVER: 0, W_CENTER: 0, HH: 1e-2, SPIRAL_TURNS: 3,
};

/* ══════════════════════════════════════════════════════════════════════════
   §1 constants + codegen
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 1. constants + codegen ---");
{
  ok(
    STRUCT_EPS === STRUCTURE_EPS,
    `WGSL STRUCT_EPS (${STRUCT_EPS}) === tfjs STRUCTURE_EPS (${STRUCTURE_EPS}) — ` +
      `the two sides are only comparable if this is literally the same number`
  );
  const L = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const off = trainPassAShader(L, { loss: { ...BASE, W_DIV: 0.5 } });
  const on = trainPassAShader(L, { loss: { ...BASE, W_DIV: 0.5, W_STRUCT: 0.5 } });
  ok(
    !/red2|accS|Lstruct/.test(off),
    "W_STRUCT: 0 emits NO structure text at all (byte-identity property)"
  );
  ok(
    /red2/.test(on) && /Lstruct/.test(on) && /lossOut\[9\]/.test(on),
    "W_STRUCT: 0.5 emits the second reduction, the loss and the broadcast grads"
  );
  ok(
    trainPassAShader(L, { loss: { ...BASE, W_DIV: 0.5, W_STRUCT: 0 } }) === off,
    "explicit W_STRUCT: 0 ≡ omitted"
  );
  // A W_STRUCT-only spec must NOT take the zero-loss (external-gradient-only)
  // shortcut — that shader is a no-op and would silently train nothing.
  const onlyStruct = trainPassAShader(L, { loss: { ...BASE, W_STRUCT: 1 } });
  ok(
    /Lstruct/.test(onlyStruct) && !/fn fwd\(@builtin\(global_invocation_id\) _gid/.test(onlyStruct),
    "a W_STRUCT-only objective is a REAL pass A, not the zero-loss stub"
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   §2 parity vs tfjs autograd
   ══════════════════════════════════════════════════════════════════════════ */

/** Uniform batch in PIXEL space; the trainer normalizes by res internally. */
function makeBatch(n: number, seed: number): Float32Array {
  const rnd = mulberry32(seed);
  const b = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    b[2 * i] = Math.fround(rnd() * PHYS.width);
    b[2 * i + 1] = Math.fround(rnd() * PHYS.height);
  }
  return b;
}

function tfjsField(enc: Encoding): HelmholtzField {
  return new HelmholtzField(
    enc.kind === "hashgrid"
      ? {
          alpha: ALPHA, hiddenUnits: [HID, HID], modelType: "hashgrid",
          gridSize: enc.gridSize, gridFeatures: enc.features,
        }
      : enc.kind === "fourier"
      ? { alpha: ALPHA, hiddenUnits: [HID, HID], modelType: "fourier", fourierOctaves: enc.octaves }
      : { alpha: ALPHA, hiddenUnits: [HID, HID] }
  );
}

/**
 * How the gradient comparison is judged.
 *
 * A cosine is meaningless when the true gradient is analytically ZERO — which
 * is exactly what the degenerate fields in §3 produce (a constant field is a
 * stationary point of L, and a dead field has no signal at all). Reporting
 * `cos = 0` there would look like a catastrophic failure and reporting
 * `cos = 1` would be impossible; the honest gate is an ABSOLUTE bound on both
 * sides, so the two cases are different verdict types rather than one metric
 * with a special case bolted on.
 */
type GradVerdict =
  | { readonly tag: "parity"; readonly minCos: number; readonly maxRel: number }
  | { readonly tag: "both-zero"; readonly tol: number };

/**
 * One parity case: fused gradient vs `tf.variableGrads` of the SAME functional.
 * `rig` is a closure so degenerate cases (§3) can force a constant field.
 */
async function parityCase(opts: {
  label: string;
  enc: Encoding;
  loss: FieldLossSpec;
  seed: number;
  /** Overwrite the packed weights (and the mirrored tfjs vars) after init. */
  rig?: (packed: Float32Array, layout: FieldLayout) => void;
  /** Extra assertions on the measured loss value. */
  check?: (fused: number, oracle: number) => void;
  forceMagnitude?: number;
  verdict?: GradVerdict;
}): Promise<{ cos: number; rel: number; loss: number }> {
  const { label, enc, loss, seed } = opts;
  const phys = { ...PHYS, forceMagnitude: opts.forceMagnitude ?? PHYS.forceMagnitude };
  const encDim = enc.kind === "raw" ? 2 : enc.kind === "fourier" ? 2 + 4 * enc.octaves : enc.features;
  const layout = layoutField("helmholtz", [fieldDims(encDim), fieldDims(encDim)], {
    encoding: enc,
  });

  const field = tfjsField(enc);
  if (field.trainableWeights.length !== layout.segments.length) {
    throw new Error(
      `${label}: tfjs var count ${field.trainableWeights.length} != segments ${layout.segments.length}`
    );
  }
  // Seed the PACKED buffer, then mirror it into tfjs so both sides start from
  // bit-identical weights (the fixture, in effect).
  const rnd = mulberry32(seed);
  const packed = new Float32Array(layout.totalFloats);
  for (const seg of layout.segments) {
    for (let i = 0; i < seg.floatLength; i++) {
      packed[seg.floatOffset + i] = Math.fround((rnd() * 2 - 1) * 0.6);
    }
  }
  opts.rig?.(packed, layout);
  field.trainableWeights.forEach((v, i) => {
    const seg = layout.segments[i];
    v.assign(
      tf.tensor(
        Array.from(packed.slice(seg.floatOffset, seg.floatOffset + seg.floatLength)),
        v.shape
      )
    );
  });

  const trainer = new FusedTrainer(device, layout, { batchCap: 256, loss });
  trainer.uploadWeights(packed);
  const batch = makeBatch(N, seed + 7);
  trainer.uploadBatch(batch);
  trainer.step(phys, { n: N, alpha: ALPHA, lr: 0, source: "uploaded", apply: false });
  const fusedLoss = await trainer.readLoss();
  const fusedGrads = await trainer.readGrads();

  // ---- oracle -------------------------------------------------------------
  const posPix = tf.tensor2d(Array.from(batch), [N, 2]) as tf.Tensor2D;
  const run = tf.variableGrads(() =>
    tf.tidy(() => {
      const posNorm = posPix.div(tf.tensor2d([[phys.width, phys.height]])) as tf.Tensor2D;
      const force = field.forces(posNorm).mul(phys.forceMagnitude) as tf.Tensor2D;
      let total = constantModeFraction(force).mul(loss.W_STRUCT ?? 0) as tf.Scalar;
      if (loss.W_ISO !== 0) {
        // Isotropy is the superposition partner on purpose: it is the OTHER
        // consumer of the `acc` vec4 reduction in pass A, so a wrong seam
        // between the two shows up here and nowhere else. (Chaos/divergence
        // probe at pos_K, a different measure — not a useful oracle for this.)
        total = total.add(isotropyLoss(force).mul(loss.W_ISO)) as tf.Scalar;
      }
      return total;
    })
  );
  const oracleLoss = run.value.dataSync()[0];
  const ref = new Float32Array(layout.totalFloats);
  field.trainableWeights.forEach((v, i) => {
    const seg = layout.segments[i];
    ref.set(run.grads[v.name].dataSync() as Float32Array, seg.floatOffset);
  });
  const cos = cosine(ref, fusedGrads);
  const rel = relMax(ref, fusedGrads);
  const lossRel = Math.abs(fusedLoss.loss - oracleLoss) / (Math.abs(oracleLoss) + 1e-12);
  const verdict: GradVerdict = opts.verdict ?? { tag: "parity", minCos: 0.9999, maxRel: 5e-3 };

  ok(
    Number.isFinite(fusedLoss.loss) &&
      fusedGrads.every(Number.isFinite) &&
      ref.every(Number.isFinite),
    `${label}: fused loss and BOTH sides' gradients are FINITE`
  );
  ok(
    lossRel < 2e-3,
    `${label}: loss fused=${fusedLoss.loss.toFixed(6)} tfjs=${oracleLoss.toFixed(6)} (rel ${lossRel.toExponential(2)})`
  );
  if (verdict.tag === "parity") {
    ok(
      cos > verdict.minCos && rel < verdict.maxRel,
      `${label}: grads vs tfjs autograd cos=${cos.toFixed(7)} relMax=${rel.toExponential(2)}`
    );
  } else {
    const maxGot = fusedGrads.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
    const maxRef = ref.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
    ok(
      maxGot < verdict.tol && maxRef < verdict.tol,
      `${label}: gradient is ZERO on BOTH sides (fused max |g| ${maxGot.toExponential(2)}, ` +
        `tfjs max |g| ${maxRef.toExponential(2)} < ${verdict.tol.toExponential(0)})`
    );
  }
  if (fusedLoss.structFraction.tag === "measured") {
    // The reported fraction must be the UNWEIGHTED ratio, in [0,1].
    const f = fusedLoss.structFraction.value;
    ok(
      f >= 0 && f <= 1 + 1e-6,
      `${label}: reported constant-mode fraction ${f.toFixed(6)} ∈ [0,1]`
    );
  }
  opts.check?.(fusedLoss.loss, oracleLoss);

  posPix.dispose();
  run.value.dispose();
  Object.values(run.grads).forEach((g) => g.dispose());
  field.trainableWeights.forEach((v) => v.dispose());
  trainer.destroy();
  return { cos, rel, loss: fusedLoss.loss };
}

console.log("\n--- 2. parity vs tfjs autograd ---");
const encodings: { label: string; enc: Encoding }[] = [
  { label: "raw", enc: { kind: "raw" } },
  { label: "fourier×3", enc: { kind: "fourier", octaves: 3 } },
  { label: "hashgrid 16²×4", enc: { kind: "hashgrid", gridSize: 16, features: 4 } },
];
for (const { label, enc } of encodings) {
  await parityCase({
    label: `${label} · struct only (W=0.7)`,
    enc,
    loss: { ...BASE, W_STRUCT: 0.7 },
    seed: 1234,
  });
  await parityCase({
    label: `${label} · struct 0.7 + iso 1.0 (shared reduction)`,
    enc,
    loss: { ...BASE, W_STRUCT: 0.7, W_ISO: 1.0 },
    seed: 4321,
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   §3 degenerate fields, in closed form
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 3. degenerate fields (closed form) ---");

/**
 * Force a constant field: zero every weight matrix, then set the LAST layer's
 * bias of both heads. tanh(b) is then the (constant) output, blended by α to
 * the same constant. Head layout is [W0,b0,W1,b1,W2,b2] per head.
 */
const rigConstant = (bx: number, by: number) =>
  (packed: Float32Array, layout: FieldLayout) => {
    packed.fill(0);
    for (const seg of layout.segments) {
      if (seg.role === "bias" && seg.layer === 2) {
        packed[seg.floatOffset] = bx;
        packed[seg.floatOffset + 1] = by;
      }
    }
  };

{
  const W = 0.7;
  const bx = 0.4;
  const by = -0.25;
  const c = [Math.tanh(bx), Math.tanh(by)];
  await parityCase({
    label: `raw · F ≡ const (tanh ${bx}, tanh ${by}), ac = 0`,
    enc: { kind: "raw" },
    loss: { ...BASE, W_STRUCT: W },
    seed: 99,
    rig: rigConstant(bx, by),
    // A constant field is a STATIONARY POINT of L: ∂L/∂F_i = (2·F̄ − 2L·F_i) /
    // (N(ms+ε)) and F_i ≡ F̄ with L = 1, so every term cancels analytically.
    // Both sides must produce that zero — a cosine here would compare noise.
    verdict: { tag: "both-zero", tol: 1e-5 },
    check: (fused) => {
      // dc² = ms exactly ⇒ L = 1 exactly (ε cancels), so loss = W_STRUCT.
      ok(
        Math.abs(fused - W) < 1e-5,
        `  F ≡ const ⇒ L = 1 exactly (ε cancels): got ${fused.toFixed(7)}, want ${W}`
      );
    },
  });
  // The constant is NOT small: an L of 1 here is a statement about DC, not
  // about amplitude. Raw-variance maximization would be indifferent to this
  // field's DC and would happily grow ‖F‖ instead.
  ok(
    Math.hypot(c[0], c[1]) > 0.3,
    `  the constant field is not itself near zero (|F| = ${Math.hypot(c[0], c[1]).toFixed(4)}) ` +
      `— an L of 1 here is about DC, not about smallness`
  );
}

{
  // F ≡ 0 exactly: L = ε/ε = 1 (the WORST score). With ε in the denominator
  // only this would be 0/ε = 0 — a PERFECT score for a dead field, which is
  // the failure mode this placement exists to prevent. The gradient is exactly
  // 0 on both sides and must be FINITE: a 0/0 inside the ratio would be NaN,
  // and the extGrad `isFiniteF` gates downstream would silently turn that NaN
  // into the same 0, hiding the bug.
  const W = 0.7;
  await parityCase({
    label: "raw · F ≡ 0 exactly",
    enc: { kind: "raw" },
    loss: { ...BASE, W_STRUCT: W },
    seed: 77,
    rig: (packed) => packed.fill(0),
    verdict: { tag: "both-zero", tol: 1e-9 },
    check: (fused) => {
      ok(
        Math.abs(fused - W) < 1e-6,
        `  F ≡ 0 ⇒ L = ε/ε = 1, the WORST score (got ${fused.toExponential(3)}) — ` +
          `ε in the denominator only would score this 0, i.e. perfect`
      );
    },
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   §4 amplitude invariance — the reason this is not raw variance
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 4. amplitude invariance ---");
{
  const loss: FieldLossSpec = { ...BASE, W_STRUCT: 1 };
  const a = await parityCase({
    label: "raw · forceMagnitude 24",
    enc: { kind: "raw" },
    loss,
    seed: 555,
    forceMagnitude: 24,
  });
  const b = await parityCase({
    label: "raw · forceMagnitude 96 (4×)",
    enc: { kind: "raw" },
    loss,
    seed: 555,
    forceMagnitude: 96,
  });
  ok(
    Math.abs(a.loss - b.loss) < 1e-4,
    `L is INVARIANT under F → 4·F: ${a.loss.toFixed(7)} vs ${b.loss.toFixed(7)} — ` +
      `raw AC would have quadrupled, which is exactly the amplitude cheat`
  );
}

console.log(
  failures === 0
    ? "\nALL FUSED W_STRUCT CHECKS PASS"
    : `\n${failures} W_STRUCT CHECK(S) FAILED`
);
process.exit(failures === 0 ? 0 : 1);
