/**
 * Deterministic tfjs-autograd GRADIENT FIXTURE generator.
 * =====================================================================
 *
 *   bun tools/grad_reference.ts   →  writes tools/fixtures/grad_ref.json
 *
 * Purpose: produce a golden {loss, ∂loss/∂weights} record for the shipped
 * "Helmholtz · Chaos" gallery piece so a fused WGSL *training* kernel can be
 * verified numerically against tfjs's own first-order autograd (CPU backend,
 * float32). This is the training-side analogue of tools/kernel_test.ts, which
 * verifies the forward/advect kernel.
 *
 * Determinism contract (why this file is a fixture, not a demo):
 *   - Field weights are OVERWRITTEN from a single mulberry32(42) stream.
 *   - The training batch is drawn from mulberry32(7).
 *   - Nothing else is random: field.forces() is a pure forward pass, the loss
 *     has no dropout/sampling, and tf CPU ops are deterministic. We therefore
 *     run the whole pipeline TWICE from fresh seeds and assert the numeric
 *     payload is bit-identical before writing (see runSelfCheck()).
 *
 * The math below is TRANSCRIBED (not imported) from src/main.ts because that
 * module pulls DOM-flavoured modules (renderers, WebGPU) that will not load
 * headless. Source line ranges are cited at each transcription site so the
 * fixture and the app cannot silently drift. `isotropyLoss` IS a pure-tf
 * export, so it is imported directly (identical code, no transcription risk).
 *
 * Gotchas honoured here (from the app + task brief):
 *   (b) field.forces() is already tanh-bounded/signed — NO (raw-0.5) shift.
 *   (c) tf.mod is FLOORED mod (matches the app's wrap); its grad w.r.t. the
 *       dividend is identity, so the physics wrap is differentiable.
 *   (d) clipByValue has ZERO grad outside its bounds. We keep the clip so the
 *       fixture matches app semantics exactly. Velocity accumulates across the
 *       K rollout steps (vₖ₊₁=(vₖ+F)·friction), but even at K=4 the worst-case
 *       component stays ≈13.65 < maxVelocity=26 (|F|≤3.5, friction=0.99), so the
 *       clip is grad-transparent here — it stays in the graph on purpose.
 *
 * K-step rollout (BPTT): set K>1 to unroll K physics steps INSIDE one gradient
 * tape and back-propagate through all of them. K=1 reproduces the original
 * single-step fixture bit-for-bit (asserted below). Env: K, N, OUT.
 */
import * as tf from "@tensorflow/tfjs";
import { HelmholtzField } from "../src/core/field/helmholtz";
// isotropyLoss is exported and pure-tf — import verbatim rather than re-transcribe.
import { isotropyLoss } from "../src/core/losses/isotropy";
import { mkdirSync, writeFileSync } from "fs";
import { dirname } from "path";

// ---------------------------------------------------------------------------
// deterministic RNG — copied VERBATIM from tools/kernel_test.ts (lines 37-46)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// pcg hash — copied VERBATIM from tools/kernel_test.ts (lines 61-65), which is
// itself the exact JS port of the WGSL pcg in advect_wgsl.ts. Used to derive
// each sample's class from its SLOT index so the fixture's class assignment is
// bit-identical to the fused kernel's uploaded-source derivation.
// ---------------------------------------------------------------------------
function pcgJS(v: number): number {
  const st = (Math.imul(v, 747796405) + 2891336453) >>> 0;
  const t = Math.imul(((st >>> (((st >>> 28) + 4) & 31)) ^ st) >>> 0, 277803737) >>> 0;
  return ((t >>> 22) ^ t) >>> 0;
}
// FNV offset basis — the salt the kernel xors the slot index with before pcg
// (src/render/webgpu/advect_wgsl.ts CLASS_SALT). Kept as a literal here to
// avoid pulling the WGSL module into this pure-tf fixture generator.
const CLASS_SALT = 2166136261;

// ---------------------------------------------------------------------------
// runtime knobs (env-overridable) — everything else stays pinned below
// ---------------------------------------------------------------------------
//   K   — number of physics rollout steps unrolled INSIDE the tape (BPTT).
//         K=1 collapses to the original single-step fixture bit-for-bit.
//   N   — training batch size (main.ts default sampleRate = 256).
//   OUT — output fixture path (relative to cwd = repo root).
const K = Number(process.env.K ?? 1);
const OUT = process.env.OUT ?? "tools/fixtures/grad_ref.json";
//   CLASSES — multi-species class count C (0 = classless, unchanged). When
//         C>0 the CHAOS head r takes [posNorm, onehot(class)] (2+C inputs);
//         the ORDER head g stays 2-input. Per-sample class is derived from the
//         SAMPLE SLOT index (see CLASS_SALT / pcgJS), matching the fused
//         kernel's uploaded-source path. Orthogonal to K (CLASSES with K=1 is
//         the shipped case). field.forces() throws for C>0, so the blend is
//         computed by hand through field.heads (see forceEval below).
const CLASSES = Number(process.env.CLASSES ?? 0);
//   MODEL — field architecture (standard | siren | fourier | hashgrid). The
//         non-standard types use the field's tfjs path (manual sin / γ(p) /
//         gridInterp) — the SAME code the app's ?train=tfjs route runs — so a
//         fixture verifies the fused trainer against real app semantics.
//         classes>0 is standard-only (HelmholtzField throws otherwise).
const MODEL = (process.env.MODEL ?? "standard") as
  | "standard" | "siren" | "fourier" | "hashgrid";
if (!["standard", "siren", "fourier", "hashgrid"].includes(MODEL)) {
  throw new Error(`MODEL '${MODEL}' not one of standard|siren|fourier|hashgrid`);
}

// ---------------------------------------------------------------------------
// pinned constants
// ---------------------------------------------------------------------------
const N = Number(process.env.N ?? 256); // training batch size (main.ts default sampleRate = 256)
const W = 800;
const H = 600;
const ALPHA = 0.7; // HelmholtzField({ alpha: 0.7 })  (main.ts:371)

// "Helmholtz · Chaos" GALLERY entry (main.ts:360-373)
const FORCE_MAGNITUDE = 3.5;
const FRICTION = 0.99;
const MAX_VELOCITY = 26;

// helmholtzChaosLoss() weights + finite-diff step (main.ts:217-221)
const W_CHAOS = 1.0;
const W_ISO = 1.0;
const W_DIV = 0.5;
const W_SPIRAL = 0.00002;
const HH = 1e-2;

// spiralLoss constants (main.ts:158-159)
const SPIRAL_TURNS = 3;
const SPIRAL_MAX_THETA = SPIRAL_TURNS * 2 * Math.PI;

const WEIGHTS_SEED = 42;
const BATCH_SEED = 7;

// ---------------------------------------------------------------------------
// spiralLoss — transcribed from src/main.ts:161-181
// ---------------------------------------------------------------------------
function spiralLoss(pos: tf.Tensor2D, w: number, h: number): tf.Scalar {
  return tf.tidy(() => {
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(w, h) * 0.38;
    const b = maxR / SPIRAL_MAX_THETA;

    const dx = pos.slice([0, 0], [-1, 1]).sub(cx);
    const dy = pos.slice([0, 1], [-1, 1]).sub(cy);
    const r = dx.square().add(dy.square()).add(1e-4).sqrt();
    const phi = tf.atan2(dy, dx);

    let best = tf.fill(r.shape, 1e8) as tf.Tensor;
    for (let k = 0; k <= SPIRAL_TURNS + 1; k++) {
      const theta = phi.add(2 * Math.PI * k);
      const rSpiral = theta.relu().mul(b);
      best = tf.minimum(best, r.sub(rSpiral).square());
    }
    return best.mean().asScalar();
  });
}

// ---------------------------------------------------------------------------
// One full autograd pass. Builds a fresh field, pins its weights + the batch,
// then computes {loss, grads} exactly as one app training step would.
//
// Returns ONLY plain data (numbers/arrays) so the caller can compare two runs
// bit-for-bit and JSON-serialize the result. All tensors are disposed here.
// ---------------------------------------------------------------------------
interface VarRecord {
  name: string;
  shape: number[];
  values: number[];
}
interface Result {
  loss: number;
  batch: number[];
  variables: VarRecord[];
  grads: VarRecord[];
}

function computeAll(): Result {
  // classes defaults to 0 → identical nets / trainableWeights to the classless
  // field, so passing `classes: 0` is bit-for-bit the original construction.
  // modelType "standard" is likewise the original construction; other MODELs
  // use the field's defaults (ω0=6, octaves=4, grid 32²×4). The ω0 init is
  // irrelevant here — weights are OVERWRITTEN from the seed stream below.
  const field = new HelmholtzField({
    alpha: ALPHA,
    classes: CLASSES,
    modelType: MODEL,
  }); // hiddenUnits default [32,32]

  // (3) OVERWRITE every trainable variable from a single shared PRNG stream.
  //     Order = field.trainableWeights order (g head kernels/biases, then r).
  const rndW = mulberry32(WEIGHTS_SEED);
  for (const v of field.trainableWeights) {
    const count = v.shape.reduce((a, b) => a * b, 1);
    const vals = new Array<number>(count);
    for (let i = 0; i < count; i++) vals[i] = (rndW() - 0.5) * 1.2;
    v.assign(tf.tensor(vals, v.shape));
  }

  // (4) Deterministic batch — pixel coords, shape [N,2], row-major (x,y).
  //     Draw order mirrors tools/kernel_test.ts: x then y per particle.
  const rndB = mulberry32(BATCH_SEED);
  const batch = new Array<number>(N * 2);
  for (let i = 0; i < N; i++) {
    batch[2 * i] = rndB() * W;
    batch[2 * i + 1] = rndB() * H;
  }
  const posT = tf.tensor2d(batch, [N, 2]);
  tf.keep(posT);

  // (4b) Per-sample class from the SAMPLE SLOT index — matches the fused
  //      kernel's uploaded-source derivation cls = pcg(s ^ CLASS_SALT) % C.
  //      Build the [N,C] one-hot ONCE (row-major); r's input everywhere is
  //      concat([posNorm, onehot]). Classless (CLASSES=0): no one-hot at all,
  //      and forceEval falls straight through to field.forces() (below), so the
  //      original single-species fixture is reproduced bit-for-bit.
  const onehotN = ((): tf.Tensor2D | null => {
    if (CLASSES === 0) return null;
    const oh = new Array<number>(N * CLASSES).fill(0);
    for (let s = 0; s < N; s++) {
      const cls = pcgJS((s ^ CLASS_SALT) >>> 0) % CLASSES;
      oh[s * CLASSES + cls] = 1;
    }
    return tf.tensor2d(oh, [N, CLASSES]);
  })();
  if (onehotN) tf.keep(onehotN);

  // Class-aware force blend, used at EVERY field-eval site (physics + probes).
  //   classless (C=0): field.forces(pn) verbatim — preserves the K=1 fixture.
  //   class-aware (C>0): field.forces() THROWS (no class input on the tfjs
  //     path), so blend the heads by hand exactly as forces() would —
  //     g(pn)·(1-α) + r([pn, onehot])·α. `oh` is the [M,C] one-hot for pn's M
  //     rows (physics: onehotN; probe [3N,2]: onehotN tiled ×3, same order).
  const [gHead, rHead] = field.heads;
  const forceEval = (pn: tf.Tensor2D, oh: tf.Tensor2D | null): tf.Tensor2D =>
    CLASSES === 0
      ? field.forces(pn)
      : ((gHead.predict(pn) as tf.Tensor2D)
          .mul(1 - ALPHA)
          .add(
            (rHead.predict(tf.concat([pn, oh as tf.Tensor2D], 1)) as tf.Tensor2D).mul(ALPHA)
          ) as tf.Tensor2D);

  // (5) loss(θ) — K-step physicsForward rollout (main.ts:379-410) then
  //     helmholtzChaosLoss() (main.ts:216-272). Everything is inside ONE tape
  //     (single tf.tidy) so variableGrads back-propagates through ALL K steps
  //     (BPTT) w.r.t. the field weights. Intermediate pos/vel are NOT disposed
  //     by hand — tidy tracks them and the gradient tape keeps what it needs;
  //     disposing them would sever the graph before the backward pass.
  const lossFn = (): tf.Scalar =>
    tf.tidy(() => {
      const wh = tf.tensor2d([[W, H]]);

      // --- K-step physicsForward rollout (main.ts:388-409), carrying pos+vel.
      //     pos_0 = batch (pixel coords), vel_0 = zeros. Each step:
      //       F_k        = field.forces(pos_k / [W,H]) * forceMagnitude   (b)
      //       vel_{k+1}  = clip((vel_k + F_k) * friction, ±maxVelocity)   (d)
      //       pos_{k+1}  = (pos_k + vel_{k+1}) mod [W,H]  (FLOORED mod)    (c)
      let pos = posT as tf.Tensor2D; // pos_0
      let vel = tf.zeros([N, 2]) as tf.Tensor2D; // vel_0
      const stepForces: tf.Tensor2D[] = []; // scaled F_0 … F_{K-1}, each [N,2]
      for (let k = 0; k < K; k++) {
        const posNormPhys = pos.div(wh) as tf.Tensor2D;
        // Field path: raw signed output * forceMagnitude (NO -0.5 shift). (b)
        // Class-aware blend when CLASSES>0 (per-sample one-hot); else field.forces().
        const Fk = forceEval(posNormPhys, onehotN).mul(FORCE_MAGNITUDE) as tf.Tensor2D;
        stepForces.push(Fk);
        // (vel+F)*friction, clip ±maxVelocity. (d) clip kept on purpose.
        vel = vel
          .add(Fk)
          .mul(FRICTION)
          .clipByValue(-MAX_VELOCITY, MAX_VELOCITY) as tf.Tensor2D;
        // (pos+vel) mod [w,h] — FLOORED mod, differentiable dividend. (c)
        pos = pos.add(vel).mod(wh) as tf.Tensor2D;
      }
      const newPos = pos; // pos_K — the loss (except isotropy) reads this.

      // --- helmholtzChaosLoss() (main.ts:216-272) --------------------------
      const posNorm = newPos.div(wh) as tf.Tensor2D;
      // SINGLE [3N,2] probe: centre / +x / +y, one field forward then slice.
      const allPos = tf.concat(
        [
          posNorm,
          posNorm.add(tf.tensor2d([[HH, 0]])),
          posNorm.add(tf.tensor2d([[0, HH]])),
        ],
        0
      ) as tf.Tensor2D;
      // The 3N probe rows are [centre; +x; +y], each an N-row block sharing the
      // SAME per-sample class, so the [3N,C] one-hot is onehotN tiled ×3 (same
      // block order as allPos). (K=1, C=0 ⇒ null ⇒ forceEval uses field.forces.)
      const onehot3N =
        onehotN === null ? null : (tf.tile(onehotN, [3, 1]) as tf.Tensor2D);
      const allF = forceEval(allPos, onehot3N);
      const f0 = allF.slice([0, 0], [N, -1]);
      const fx = allF.slice([N, 0], [N, -1]);
      const fy = allF.slice([2 * N, 0], [N, -1]);

      // chaos: local sensitivity (Lyapunov proxy).  (main.ts:252-255)
      const sepx = fx.sub(f0).square().sum(1);
      const sepy = fy.sub(f0).square().sum(1);
      const sep = sepx.add(sepy).add(1e-12).sqrt().div(HH * 1.4142 + 1e-9);
      const chaos = sep.add(1e-6).log().mean().neg();

      // forward-difference divergence sharing the centre.  (main.ts:258-260)
      const dFxdx = fx.slice([0, 0], [-1, 1]).sub(f0.slice([0, 0], [-1, 1])).div(HH);
      const dFydy = fy.slice([0, 1], [-1, 1]).sub(f0.slice([0, 1], [-1, 1])).div(HH);
      const div = dFxdx.add(dFydy).square().mean();

      // isotropy reads the CONCATENATION of ALL K steps' scaled forces — an
      // [N·K, 2] batch — so every step's field output shapes the penalty.
      // (main.ts:262; K=1 ⇒ concat([F_0]) == F_0, identical to the app.)
      const iso = isotropyLoss(tf.concat(stepForces, 0) as tf.Tensor2D);
      const spiral = spiralLoss(newPos, W, H); // (main.ts:263)

      // composite (main.ts:265-270)
      return chaos
        .mul(W_CHAOS)
        .add(iso.mul(W_ISO))
        .add(div.mul(W_DIV))
        .add(spiral.mul(W_SPIRAL))
        .asScalar();
    });

  // (6) gradients, keyed by variable .name → remap to trainableWeights order.
  const { value, grads } = tf.variableGrads(lossFn, field.trainableWeights);
  const loss = value.dataSync()[0];

  const variables: VarRecord[] = field.trainableWeights.map((v) => ({
    name: v.name,
    shape: v.shape.slice(),
    values: Array.from(v.dataSync()),
  }));
  const gradsOut: VarRecord[] = field.trainableWeights.map((v) => {
    const g = grads[v.name];
    if (!g) throw new Error(`no gradient for variable ${v.name}`);
    return { name: v.name, shape: g.shape.slice(), values: Array.from(g.dataSync()) };
  });

  // cleanup
  value.dispose();
  for (const g of Object.values(grads)) g.dispose();
  posT.dispose();
  if (onehotN) onehotN.dispose();
  field.dispose();

  return { loss, batch, variables, grads: gradsOut };
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const l2 = (xs: number[]): number => Math.sqrt(xs.reduce((s, x) => s + x * x, 0));

// bit-identical (===) equality over the NUMERIC payload. Variable NAME strings
// are deliberately excluded: tfjs assigns layer names from a PROCESS-GLOBAL
// counter (dense_Dense1, dense_Dense2, …) that keeps incrementing across fresh
// HelmholtzField constructions, so run #2's names carry higher suffixes. That
// is a naming artifact, not numeric nondeterminism — the shapes, order, and all
// values still match exactly, which is what the kernel test verifies against.
function assertSameNumeric(a: Result, b: Result): void {
  const eqArr = (x: number[], y: number[], where: string) => {
    if (x.length !== y.length) throw new Error(`determinism: length mismatch ${where}`);
    for (let i = 0; i < x.length; i++)
      if (x[i] !== y[i])
        throw new Error(`determinism: value mismatch ${where}[${i}]: ${x[i]} !== ${y[i]}`);
  };
  if (!Object.is(a.loss, b.loss))
    throw new Error(`determinism: loss ${a.loss} !== ${b.loss}`);
  eqArr(a.batch, b.batch, "batch");
  if (a.variables.length !== b.variables.length) throw new Error("determinism: var count");
  if (a.grads.length !== b.grads.length) throw new Error("determinism: grad count");
  for (let i = 0; i < a.variables.length; i++) {
    eqArr(a.variables[i].shape, b.variables[i].shape, `var${i}.shape`);
    eqArr(a.variables[i].values, b.variables[i].values, `var${i}.values`);
    eqArr(a.grads[i].shape, b.grads[i].shape, `grad${i}.shape`);
    eqArr(a.grads[i].values, b.grads[i].values, `grad${i}.values`);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
await tf.setBackend("cpu");
await tf.ready();
console.log(`TF.js backend: ${tf.getBackend()}\n`);

// (9) robustness: two fresh-seed runs must produce a bit-identical numeric payload.
const run1 = computeAll();
const run2 = computeAll();
assertSameNumeric(run1, run2);
console.log("determinism self-check: two fresh-seed runs are bit-identical (numeric payload).\n");

const result = run1;

// (8) sanity asserts BEFORE writing.
if (!Number.isFinite(result.loss)) throw new Error(`loss is not finite: ${result.loss}`);

// (8b) K=1 regression tripwire. With the default batch (N=256) and default
//      seeds, the K-step rollout collapses to the ORIGINAL single-step loss, so
//      the composite must reproduce the originally-shipped grad_ref.json loss
//      bit-close (≤1e-12). Guards the whole K-generalization against drift.
if (MODEL === "standard" && CLASSES === 0 && K === 1 && N === 256 && WEIGHTS_SEED === 42 && BATCH_SEED === 7) {
  const OLD_K1_LOSS = 1.0965325832366943;
  const dLoss = Math.abs(result.loss - OLD_K1_LOSS);
  if (dLoss > 1e-12)
    throw new Error(
      `K=1 regression: loss ${result.loss} != original ${OLD_K1_LOSS} (Δ=${dLoss})`
    );
  console.log(`K=1 regression check: loss reproduces original ${OLD_K1_LOSS} within 1e-12.`);
}

let totalEntries = 0;
let nonzeroEntries = 0;
console.log(`loss = ${result.loss}`);
console.log("grad L2 norms per variable (trainableWeights order):");
for (let i = 0; i < result.variables.length; i++) {
  const v = result.variables[i];
  const g = result.grads[i];
  if (g.shape.length !== v.shape.length || g.shape.some((d, k) => d !== v.shape[k]))
    throw new Error(`grad shape ${g.shape} != var shape ${v.shape} for ${v.name}`);
  // the hashgrid feature table is legitimately SPARSE (only cells the batch's
  // sites touch get gradient) — exclude it from the density tripwire.
  const isGrid = MODEL === "hashgrid" && i === 0;
  if (!isGrid) {
    totalEntries += g.values.length;
    nonzeroEntries += g.values.filter((x) => x !== 0).length;
  }
  console.log(
    `  [${String(i).padStart(2)}] ${v.name.padEnd(24)} shape=[${v.shape.join(",")}]` +
      `  |grad|₂ = ${l2(g.values).toExponential(6)}`
  );
}
const nonzeroFrac = nonzeroEntries / totalEntries;
console.log(
  `\ngradient nonzero fraction: ${(nonzeroFrac * 100).toFixed(2)}% ` +
    `(${nonzeroEntries}/${totalEntries})`
);
if (nonzeroFrac < 0.95)
  throw new Error(`too many zero gradient entries: ${(nonzeroFrac * 100).toFixed(2)}% nonzero < 95%`);

// (7) write the fixture.
const fixture = {
  meta: {
    K,
    N,
    W,
    H,
    model: MODEL,
    // encoding params (field defaults) — the kernel layout must match these
    fourierOctaves: 4,
    gridSize: 32,
    gridFeatures: 4,
    classes: CLASSES,
    alpha: ALPHA,
    forceMagnitude: FORCE_MAGNITUDE,
    friction: FRICTION,
    maxVelocity: MAX_VELOCITY,
    HH,
    weights_seed: WEIGHTS_SEED,
    batch_seed: BATCH_SEED,
    loss_constants: { W_CHAOS, W_ISO, W_DIV, W_SPIRAL },
  },
  variables: result.variables,
  batch: result.batch,
  loss: result.loss,
  grads: result.grads,
};

// OUT is a filesystem path relative to cwd (repo root when run as
// `bun tools/grad_reference.ts`); default keeps the original location.
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(fixture, null, 2));

const numCount =
  result.batch.length +
  1 +
  result.variables.reduce((s, v) => s + v.values.length, 0) +
  result.grads.reduce((s, g) => s + g.values.length, 0);
console.log(`\nwrote ${OUT}  (K=${K}, N=${N}, ~${numCount} numbers)`);
