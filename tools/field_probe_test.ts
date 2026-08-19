/**
 * FIELD HEALTH PROBE — does the LIVE instrument measure the same statistic as
 * `tools/collapse_probe.ts`? Real Metal via bun-webgpu. GPU suites are
 * SEQUENTIAL — run nothing else on the GPU.
 *
 *   bun tools/field_probe_test.ts
 *
 * The snapshot's field block is the whole reason the health audit can say
 * "dead field" or "frozen generator" at all, and it was written by transcribing
 * `collapse_probe.ts::diagnostics` into JS. A transcription is exactly the kind
 * of thing that is right until someone edits one of the two copies, so:
 *
 *  §1 CLOSED FORM. Synthetic force fields with hand-computed AC/DC/sat/OW —
 *     a pure constant (ac = 0, dc = |c|), a pure shear (OW = +1), a pure
 *     rotation (OW = −1), and a saturated diagonal (satFrac = 1). These pin
 *     the definitions without reference to any other implementation.
 *  §2 THE REAL GPU PROBE vs the same tfjs field the offline probe uses:
 *     `HelmholtzField.forces` on collapse_probe's own 64² grid, reduced by
 *     collapse_probe's own formulas, against `FieldProbe.sample()` on the
 *     packed weights. Same weights, same field, two entirely separate
 *     evaluators.
 *  §3 the identity `rmsF² = ac² + dc²`, which is what makes `dc/ac` in the HUD
 *     a complete description of the split.
 *
 * DIRECTION ORDER (R₁/R₂) is checked in both places: closed-form in §1 (where
 * the soft-unit softener makes the expected value a number you can write down,
 * not 1), and in §2 against `directionOrderParameters` — the tfjs twin the
 * adversary's own telemetry uses. The two implementations exist so the HUD can
 * put a grid R₁ and a batch R₁ on the same axis; if they ever stop being the
 * same statistic, that comparison is a lie and §2 is what catches it.
 */
import { setupGlobals } from "bun-webgpu";
import * as tf from "@tensorflow/tfjs";
import {
  layoutField,
  fieldProbeShader,
  type LayerDims,
} from "../src/render/webgpu/advect_wgsl";
import {
  FieldProbe,
  fieldMetricsFrom,
  probePoints,
  PROBE_FD_H,
  PROBE_TAU,
} from "../src/render/webgpu/field_probe";
import { HelmholtzField } from "../src/core/field/helmholtz";
import { directionOrderParameters } from "../src/core/losses/isotropy";

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
const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

/* ══════════════════════════════════════════════════════════════════════════
   §1 closed-form fields
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 1. closed-form synthetic fields ---");

/** Evaluate `f` on `probePoints(gridN)` and reduce, i.e. exercise the exact
 *  host reduction the live probe uses, with the GPU replaced by a formula. */
function metricsOf(gridN: number, f: (x: number, y: number) => [number, number]) {
  const pts = probePoints(gridN);
  const out = new Float32Array(pts.length);
  for (let i = 0; i < pts.length / 2; i++) {
    const [fx, fy] = f(pts[2 * i], pts[2 * i + 1]);
    out[2 * i] = fx;
    out[2 * i + 1] = fy;
  }
  return fieldMetricsFrom(out, gridN);
}

{
  // Pure constant: ALL of the energy is DC. This is the collapse, in closed
  // form — and note |c| is large, so an `ac = 0` verdict here is a statement
  // about structure, not about the field being weak.
  const c: [number, number] = [0.6, -0.35];
  const m = metricsOf(32, () => c);
  ok(near(m.ac, 0, 1e-6), `constant field: ac = ${m.ac.toExponential(2)} ≈ 0`);
  ok(near(m.dc, Math.hypot(...c), 1e-6), `constant field: dc = ‖c‖ = ${m.dc.toFixed(6)}`);
  ok(near(m.rmsF, Math.hypot(...c), 1e-6), `constant field: rmsF = ‖c‖ (all DC)`);
  ok(m.satFrac === 0, `constant field below ±0.9: satFrac = 0`);
}
{
  // Pure shear F = (y, 0): ∂Fx/∂y = 1 is the only nonzero derivative, so
  // curl = −1, s2 = +1, ⟨|S|²⟩ = ⟨ω²⟩ = 1 and OW = 0 exactly. This is the
  // knife edge between vortex- and strain-dominated, and it catches a swapped
  // curl/strain sign that a one-sided test would not.
  const m = metricsOf(32, (_x, y) => [y, 0]);
  ok(near(m.okuboWeiss, 0, 1e-3), `pure shear (y,0): OW = ${m.okuboWeiss.toFixed(5)} ≈ 0`);
}
{
  // Pure rotation F = (−y, x): curl = 2, strain = 0 ⇒ OW = −1 (vortex).
  const m = metricsOf(32, (x, y) => [-y, x]);
  ok(near(m.okuboWeiss, -1, 1e-3), `rotation (−y,x): OW = ${m.okuboWeiss.toFixed(5)} = −1 (vortex)`);
}
{
  // Pure extension F = (x, −y): strain = 2, curl = 0 ⇒ OW = +1 (strain).
  const m = metricsOf(32, (x, y) => [x, -y]);
  ok(near(m.okuboWeiss, 1, 1e-3), `extension (x,−y): OW = ${m.okuboWeiss.toFixed(5)} = +1 (strain)`);
}
{
  // Saturated diagonal: BOTH components past ±0.9 everywhere. This is the
  // "frozen generator" state the audit gates on at satFrac > 0.3.
  const m = metricsOf(32, () => [0.97, -0.95]);
  ok(m.satFrac === 1, `saturated diagonal: satFrac = ${m.satFrac} (both components > 0.9)`);
  const half = metricsOf(32, (x) => (x < 0.5 ? [0.97, -0.95] : [0.1, 0.1]));
  ok(
    near(half.satFrac, 0.5, 0.05),
    `half-saturated field: satFrac = ${half.satFrac.toFixed(3)} ≈ 0.5`
  );
}
{
  // DIRECTION ORDER, closed form. A constant field is FULLY converged, and the
  // expected R₁ is NOT 1: the soft unit u = F/√(‖F‖²+τ²) is deliberately short
  // for a weak force, so R₁ = ‖c‖/√(‖c‖²+τ²) exactly. Writing the softened
  // value out is the point — an implementation that normalized hard, or used
  // the wrong τ, passes an "R₁ ≈ 1" assertion and fails this one.
  const c: [number, number] = [0.6, -0.35];
  const n = Math.hypot(...c);
  const m = metricsOf(32, () => c);
  const want1 = n / Math.sqrt(n * n + PROBE_TAU * PROBE_TAU);
  ok(
    near(m.r1, want1, 1e-6),
    `constant field: r1 = ${m.r1.toFixed(6)} = ‖c‖/√(‖c‖²+τ²) = ${want1.toFixed(6)}`
  );
  // |u|² for the same magnitude — the double-angle vector carries it.
  ok(
    near(m.r2, want1 * want1, 1e-6),
    `constant field: r2 = ${m.r2.toFixed(6)} = |u|² = ${(want1 * want1).toFixed(6)}`
  );
}
{
  // The ± ESCAPE, the reason R₂ is measured at all: two counter-streaming
  // sheets. The polar mean cancels exactly (R₁ = 0) while the field is as
  // laminar as it is possible to be, and only R₂ says so. This is the state
  // `directionOrderLoss` measured at R₂ = 0.95 when the polar term ran alone.
  const m = metricsOf(32, (x) => (x < 0.5 ? [0.6, 0] : [-0.6, 0]));
  ok(near(m.r1, 0, 1e-6), `± counter-streaming sheets: r1 = ${m.r1.toExponential(2)} ≈ 0`);
  ok(
    m.r2 > 0.9,
    `± counter-streaming sheets: r2 = ${m.r2.toFixed(4)} > 0.9 — R₁ ALONE CALLS THIS HEALTHY`
  );
}
{
  // A vortex: directions sweep the full circle, so both order parameters
  // vanish. Deliberately UNIT-magnitude-free — the flow is strongest at the
  // rim — because an energy-weighted statistic (isotropyLoss's covariance)
  // would be dominated by the rim and this must not be.
  const m = metricsOf(64, (x, y) => [-(y - 0.5), x - 0.5]);
  ok(near(m.r1, 0, 1e-3), `vortex: r1 = ${m.r1.toExponential(2)} ≈ 0 (isotropic directions)`);
  ok(near(m.r2, 0, 1e-3), `vortex: r2 = ${m.r2.toExponential(2)} ≈ 0`);
}
{
  // A field of near-ZERO force has no reliable direction, and the softener is
  // what stops the instrument from inventing one: at ‖F‖ ≪ τ, R₁ → ‖F‖/τ → 0.
  // Without it a numerically dead field would report R₁ = 1 and the audit
  // would call a dead field "laminar collapse" — the wrong repair, every time.
  const m = metricsOf(32, () => [1e-6, 0]);
  ok(m.r1 < 1e-4, `near-zero field: r1 = ${m.r1.toExponential(2)} — softener refuses to guess`);
}
{
  // Sanity on the stencil itself: a field with a KNOWN gradient must produce
  // that gradient through the central differences at PROBE_FD_H.
  const m = metricsOf(16, (x, y) => [3 * x, 5 * y]);
  // divergence 8, curl 0, s1 = 3−5 = −2, s2 = 0 ⇒ ⟨|S|²⟩ = 4, ⟨ω²⟩ = 0, OW = 1
  ok(near(m.okuboWeiss, 1, 1e-4), `linear (3x,5y): OW = 1 (pure strain), h = ${PROBE_FD_H}`);
}

/* ══════════════════════════════════════════════════════════════════════════
   §2 GPU probe vs the tfjs field on collapse_probe's own grid
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 2. GPU probe vs the tfjs field (collapse_probe's reduction) ---");

const HID = 12;
const dims = (inDim: number): LayerDims[] => [
  { inSize: inDim, outSize: HID, activation: "selu" },
  { inSize: HID, outSize: HID, activation: "selu" },
  { inSize: HID, outSize: 2, activation: "tanh" },
];
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** collapse_probe.ts::diagnostics, verbatim on the AC/DC/sat/OW quantities.
 *  Deliberately written from the tf ops rather than reusing the host reducer —
 *  a shared helper would make this test tautological. */
function oracleMetrics(field: HelmholtzField, gridN: number) {
  return tf.tidy(() => {
    const a = new Float32Array(gridN * gridN * 2);
    for (let j = 0; j < gridN; j++) {
      for (let i = 0; i < gridN; i++) {
        a[2 * (j * gridN + i)] = (i + 0.5) / gridN;
        a[2 * (j * gridN + i) + 1] = (j + 0.5) / gridN;
      }
    }
    const grid = tf.tensor2d(a, [gridN * gridN, 2]) as tf.Tensor2D;
    const shift = (dx: number, dy: number) =>
      field.forces(grid.add(tf.tensor2d([[dx, dy]])) as tf.Tensor2D);
    const f = field.forces(grid);
    const h = PROBE_FD_H;
    const dFdx = shift(h, 0).sub(shift(-h, 0)).div(2 * h) as tf.Tensor2D;
    const dFdy = shift(0, h).sub(shift(0, -h)).div(2 * h) as tf.Tensor2D;
    const cx = (t: tf.Tensor2D, c: number) =>
      t.slice([0, c], [-1, 1]).reshape([-1]) as tf.Tensor1D;
    const curl = cx(dFdx, 1).sub(cx(dFdy, 0)) as tf.Tensor1D;
    const s1 = cx(dFdx, 0).sub(cx(dFdy, 1)) as tf.Tensor1D;
    const s2 = cx(dFdx, 1).add(cx(dFdy, 0)) as tf.Tensor1D;
    const strain2 = s1.square().add(s2.square()) as tf.Tensor1D;
    const msCurl = curl.square().mean();
    const msStrain = strain2.mean();
    const dcVec = f.mean(0) as tf.Tensor1D;
    const ac = f.sub(dcVec.reshape([1, 2])).square().sum(1).mean().sqrt();
    const sat = tf
      .minimum(cx(f, 0).abs(), cx(f, 1).abs())
      .greater(0.9)
      .toFloat()
      .mean();
    const values = tf
      .stack([
        ac,
        dcVec.square().sum().sqrt(),
        f.square().sum(1).mean().sqrt(),
        sat,
        msStrain.sub(msCurl).div(msStrain.add(msCurl).add(1e-12)),
      ])
      .dataSync();
    // R₁/R₂ come from the ADVERSARY's own tfjs implementation, not from a
    // formula rewritten here: the whole claim the HUD makes is that the grid
    // R₁ and the batch R₁ are one statistic, and only a comparison against
    // that exact function can support it.
    const order = directionOrderParameters(f, PROBE_TAU);
    return {
      ac: values[0], dc: values[1], rmsF: values[2],
      satFrac: values[3], okuboWeiss: values[4],
      r1: order.r1, r2: order.r2,
    };
  });
}

/**
 * Two regimes, on purpose. `scale 0.45` gives a structured field (the healthy
 * case, where AC carries most of the energy); `scale 1.6` drives every head
 * into tanh saturation and produces the near-constant field the audit calls
 * DEAD. A parity test that only ever saw one of them could not tell whether
 * the instrument reports structure or merely reports something.
 */
const CASES = [
  { gridN: 32, scale: 0.45, structured: true },
  { gridN: 64, scale: 0.45, structured: true },
  { gridN: 32, scale: 1.6, structured: false },
] as const;

for (const { gridN, scale, structured } of CASES) {
  const layout = layoutField("helmholtz", [dims(2), dims(2)]);
  const field = new HelmholtzField({ alpha: 0.55, hiddenUnits: [HID, HID] });
  const rnd = mulberry32(4242);
  const packed = new Float32Array(layout.totalFloats);
  for (const seg of layout.segments) {
    for (let i = 0; i < seg.floatLength; i++) {
      packed[seg.floatOffset + i] = Math.fround((rnd() * 2 - 1) * scale);
    }
  }
  field.trainableWeights.forEach((v, i) => {
    const seg = layout.segments[i];
    v.assign(
      tf.tensor(
        Array.from(packed.slice(seg.floatOffset, seg.floatOffset + seg.floatLength)),
        v.shape
      )
    );
  });

  const weights = device.createBuffer({
    size: layout.totalFloats * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weights, 0, packed);
  const probe = new FieldProbe(device, layout, weights, gridN);
  const got = await probe.sample(0.55);
  if (!got) throw new Error("probe returned null on its first sample");
  const want = oracleMetrics(field, gridN);

  const label = `grid ${gridN}² w±${scale}`;
  // f32 on two different evaluators (WGSL selu/tanh vs tfjs) over 5·gridN²
  // sites; 1e-4 relative is the honest floor, and every quantity here is O(1).
  const cmp = (name: keyof typeof want, tol: number) =>
    ok(
      near(got[name], want[name], tol),
      `${label} ${String(name)}: probe ${got[name].toFixed(6)} vs collapse_probe ${want[name].toFixed(6)}`
    );
  cmp("ac", 1e-4);
  cmp("dc", 1e-4);
  cmp("rmsF", 1e-4);
  cmp("satFrac", 1e-9);
  // OW is a ratio of finite differences at h = 1/256 on a saturating net; the
  // two evaluators' tanh differ in the last f32 ulp and that amplifies here.
  cmp("okuboWeiss", 2e-3);
  // Same tolerance class as ac/dc: two evaluators, f32, gridN² sites.
  cmp("r1", 1e-4);
  cmp("r2", 1e-4);
  if (structured) {
    ok(
      got.ac > 1e-2 && got.ac > got.dc * 0.1,
      `${label}: this field really HAS structure (ac ${got.ac.toFixed(5)}, ` +
        `dc ${got.dc.toFixed(5)}) — comparing two dead fields would be vacuous`
    );
  } else {
    // The saturated regime IS the collapse, reproduced deliberately: both
    // evaluators must agree that it is dead, and `tools/health_audit.mjs`'s
    // acDead gate (1e-4) must fire on it.
    ok(
      got.ac < 1e-4 && got.dc > 0.5,
      `${label}: saturation ⇒ DEAD FIELD (ac ${got.ac.toExponential(2)} < 1e-4, ` +
        `dc ${got.dc.toFixed(4)}) — the exact state the audit's acDead gate names`
    );
  }

  /* §3 the identity that makes dc/ac a complete description ---------------- */
  const identity = Math.sqrt(got.ac * got.ac + got.dc * got.dc);
  ok(
    near(identity, got.rmsF, 1e-5),
    `${label}: rmsF² = ac² + dc² (${identity.toFixed(6)} vs ${got.rmsF.toFixed(6)})`
  );

  probe.destroy();
  weights.destroy();
  field.trainableWeights.forEach((v) => v.dispose());
}

/* ══════════════════════════════════════════════════════════════════════════
   §4 THE NO-OPTIMIZATION CONSTRAINT, enforced
   ══════════════════════════════════════════════════════════════════════════ */
console.log("\n--- 4. the probe cannot optimize anything ---");
{
  // The health metrics are OBSERVED on adversarial and pixel pieces — they must
  // never contribute to extGrads or any loss. Three independent statements of
  // that, each of which can actually fail:
  //
  //  (a) the SHADER cannot write weights: binding 1 is `var<storage, read>`,
  //      and the module declares exactly the four diagnostics bindings — there
  //      is no extGrad/grad/adam buffer in its interface at all;
  //  (b) the weights buffer is BIT-IDENTICAL across a probe;
  //  (c) `FieldProbe` never touches `AdversaryTrainer`. That one is structural
  //      (it does not import it) and is stated in the note, not here.
  const layout = layoutField("helmholtz", [dims(2), dims(2)]);
  const src = fieldProbeShader(layout);
  ok(
    /@group\(0\) @binding\(1\) var<storage, read> weights/.test(src) &&
      !/read_write>\s*weights/.test(src),
    "(a) the probe binds weights READ-ONLY — it is not physically able to update them"
  );
  const bindings = [...src.matchAll(/@binding\((\d+)\)/g)].map((m) => m[1]);
  ok(
    bindings.join(",") === "0,1,2,3" && !/extGrad|grads|adam/i.test(src),
    `(a) the probe module declares exactly bindings 0..3 (${bindings.join(",")}) and ` +
      `mentions no gradient/Adam buffer`
  );

  const rnd = mulberry32(31337);
  const packed = new Float32Array(layout.totalFloats);
  for (const seg of layout.segments) {
    for (let i = 0; i < seg.floatLength; i++) {
      packed[seg.floatOffset + i] = Math.fround((rnd() * 2 - 1) * 0.5);
    }
  }
  const weights = device.createBuffer({
    size: layout.totalFloats * 4,
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(weights, 0, packed);
  const readWeights = async (): Promise<Float32Array> => {
    const staging = device.createBuffer({
      size: layout.totalFloats * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(weights, 0, staging, 0, layout.totalFloats * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  };
  const before = await readWeights();
  const probe = new FieldProbe(device, layout, weights, 32);
  for (let i = 0; i < 5; i++) await probe.sample(0.5);
  const after = await readWeights();
  let identical = before.length === after.length;
  for (let i = 0; identical && i < before.length; i++) {
    // BIT comparison, not a tolerance: an optimizer step of any size at all is
    // a failure here, and a tolerance would let a tiny one through.
    identical = Object.is(before[i], after[i]);
  }
  ok(
    identical,
    `(b) 5 probe samples leave all ${layout.totalFloats} packed weight floats BIT-IDENTICAL`
  );
  probe.destroy();
  weights.destroy();
}

console.log(
  failures === 0
    ? "\nALL FIELD PROBE CHECKS PASS"
    : `\n${failures} FIELD PROBE CHECK(S) FAILED`
);
process.exit(failures === 0 ? 0 : 1);
