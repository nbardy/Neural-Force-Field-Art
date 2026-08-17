/**
 * DIRECTIONAL-COLLAPSE PROBE — why adversary pieces settle into laminar streaks.
 *
 *   PRESET=pair  STEPS=1500 bun tools/collapse_probe.ts
 *   PRESET=point STEPS=1500 bun tools/collapse_probe.ts
 *   PRESET=pair  SWIRL=0.02 bun tools/collapse_probe.ts     # the prototype fix
 *
 * This reproduces the TFJS ORACLE loop exactly as `tick` runs it for a
 * standalone adversary piece (src/main.ts ~2860-2945): both the discriminator
 * batch and the generator batch are FRESH UNIFORM positions, there is no
 * advect coupling, and the aesthetic term is ZERO_FIELD_LOSS. That is the real
 * production path for the default piece, whose hashgrid field the fused
 * adversary kernel refuses (`fusedAdvOk`, src/main.ts:3344).
 *
 * WHAT IT MEASURES (the point of the tool — the HUD has none of this):
 *
 *   R1  polar order  ‖mean_i unit_τ(F(x_i))‖ ∈ [0,1].  0 = isotropic
 *       directions, 1 = every force points the same way (laminar).
 *   R2  nematic order ‖mean_i e^{2iθ_i}‖ ∈ [0,1]. Catches ± streaks that R1
 *       cancels out (counter-flowing sheets are still laminar art).
 *   ω,S the local Jacobian split. For J = ∇F,
 *         div  = ∂ₓFₓ + ∂_yF_y            (isotropic expansion)
 *         ω    = ∂ₓF_y − ∂_yFₓ            (vorticity — the swirl)
 *         S    = (∂ₓFₓ − ∂_yF_y, ∂ₓF_y + ∂_yFₓ)   (traceless strain — the shear)
 *       and OW = (rms|S|² − rms ω²)/(rms|S|² + rms ω²) is the normalized
 *       Okubo–Weiss parameter: OW < 0 ⇔ rotation-dominated (vortices),
 *       OW > 0 ⇔ strain-dominated (laminar shear sheets).
 *
 * Jacobians are CENTRAL FINITE DIFFERENCES, not autograd: `forces()` is a
 * plain forward pass on purpose (src/core/field/helmholtz.ts) and taking
 * tf.grad w.r.t. the input inside a training tape would be the second-order
 * path that architecture was built to avoid. Finite differences keep every
 * term FIRST-order differentiable w.r.t. the weights, which is also what makes
 * the swirl prototype below fusable.
 */
import * as tf from "@tensorflow/tfjs";
import {
  Adversary,
  defaultAdversaryConfig,
  disposeTupleSample,
  type AdversaryConfig,
} from "../src/core/gan/adversary";
import { ARCH, createFieldFromArch } from "../src/core/field/arch";
import type { HelmholtzField } from "../src/core/field/helmholtz";

await tf.setBackend("cpu");
await tf.ready();

/* ── knobs ──────────────────────────────────────────────────────────────── */

type PresetTag = "pair" | "point";
const PRESET = (process.env.PRESET ?? "pair") as PresetTag;
const STEPS = Number(process.env.STEPS ?? 1500);
const REPORT = Number(process.env.REPORT ?? 100);
const SEED = Number(process.env.SEED ?? 1234);
const B = Number(process.env.B ?? 512);
const ARCH_TAG = process.env.ARCH ?? (PRESET === "pair" ? "dualHashgrid" : "dualStd");
/** λ for the prototype swirl term (normalized Okubo–Weiss). 0 = current behaviour. */
const SWIRL = Number(process.env.SWIRL ?? 0);
/** λ for the batch-mean-direction penalty ‖mean unit_τ(F)‖². 0 = off. */
const POLAR = Number(process.env.POLAR ?? 0);
/**
 * λ for the NEMATIC companion ‖mean_B e^{2iθ}‖². POLAR alone is defeated by a
 * ±F₀ field (counter-streaming sheets): measured R1 0.004 with R2 0.95.
 */
const NEMATIC = Number(process.env.NEMATIC ?? 0);
/** λ for the one-sided magnitude hinge mean(relu(|F| − MAG_T))². 0 = off. */
const MAG = Number(process.env.MAG ?? 0);
const MAG_T = Number(process.env.MAG_T ?? 0.25);
/**
 * λ for the TWO-SIDED anchor on the ENCODED TARGET magnitude,
 * `λ·(rms‖y‖ − ANCHOR_T)²`. This is the term the pole exploit needs: the
 * soft-angle payoff is maximized by rms‖y‖ → 0 and nothing else opposes it.
 */
const ANCHOR = Number(process.env.ANCHOR ?? 0);
const ANCHOR_T = Number(process.env.ANCHOR_T ?? 0.3);
/** Finite-difference step in NORMALIZED position units. */
const FD_H = Number(process.env.FD_H ?? 1 / 256);

/**
 * The two live configurations under investigation. `pair` is the shipped
 * default piece; `point` is the user's hand-tuned point-observer dock state.
 */
const PRESETS = {
  pair: {
    encoding: { tag: "pair-rotation-scale-adjusted" } as const,
    kind: { tag: "wta", k: 4, relaxEps: 0.05 } as const,
    tau: 0.05,
    weight: 0.015,
    gLR: 0.001,
    dLR: 0.003,
  },
  point: {
    encoding: { tag: "point" } as const,
    kind: { tag: "wta", k: 8, relaxEps: 0.22 } as const,
    tau: 0.05,
    weight: 0.025,
    gLR: 3.1e-4,
    dLR: 1.2e-4,
  },
} as const;

const P = PRESETS[PRESET];
const WEIGHT = Number(process.env.WEIGHT ?? P.weight);
const GLR = Number(process.env.GLR ?? P.gLR);
const DLR = Number(process.env.DLR ?? P.dLR);
const TAU = Number(process.env.TAU ?? P.tau);

/* ── field + adversary ──────────────────────────────────────────────────── */

const archSpec = (ARCH as Record<string, Record<string, unknown>>)[ARCH_TAG];
if (!archSpec) throw new Error(`unknown ARCH ${ARCH_TAG}`);
const field: HelmholtzField = createFieldFromArch({
  ...(archSpec as any),
  alpha: 0.55,
});

const cfg: AdversaryConfig = {
  ...defaultAdversaryConfig(P.kind, P.encoding, "periodic"),
  loss: { tag: "soft-angle", tau: TAU },
  target: { tag: "force" },
  batchTuples: B,
  learningRate: DLR,
  seed: SEED,
};
const adv = new Adversary(cfg);
const gen = tf.train.adam(GLR);
const varList = field.trainableWeights;

/* ── diagnostics ────────────────────────────────────────────────────────── */

/** Fixed 64×64 measurement grid over the normalized domain. Same every call. */
const GRID_N = 64;
const gridXY = (() => {
  const a = new Float32Array(GRID_N * GRID_N * 2);
  for (let j = 0; j < GRID_N; j++) {
    for (let i = 0; i < GRID_N; i++) {
      a[2 * (j * GRID_N + i)] = (i + 0.5) / GRID_N;
      a[2 * (j * GRID_N + i) + 1] = (j + 0.5) / GRID_N;
    }
  }
  return tf.tensor2d(a, [GRID_N * GRID_N, 2]);
})();

type Diagnostics = {
  r1: number;
  r2: number;
  meanMag: number;
  /** ‖mean_grid F‖ — the spatially CONSTANT (DC) mode. */
  dc: number;
  /** rms‖F − mean F‖ — the spatially varying (AC) mode, i.e. the structure. */
  ac: number;
  /** Fraction of grid points with BOTH tanh components past ±0.9·max. */
  sat: number;
  rmsDiv: number;
  rmsCurl: number;
  rmsStrain: number;
  okuboWeiss: number;
};

function diagnostics(): Diagnostics {
  const vals = tf.tidy(() => {
    const shift = (dx: number, dy: number) =>
      field.forces(gridXY.add(tf.tensor2d([[dx, dy]])) as tf.Tensor2D);
    const f = field.forces(gridXY);
    const fxp = shift(FD_H, 0);
    const fxm = shift(-FD_H, 0);
    const fyp = shift(0, FD_H);
    const fym = shift(0, -FD_H);
    const dFdx = fxp.sub(fxm).div(2 * FD_H) as tf.Tensor2D;
    const dFdy = fyp.sub(fym).div(2 * FD_H) as tf.Tensor2D;
    const cx = (t: tf.Tensor2D, c: number) =>
      t.slice([0, c], [-1, 1]).reshape([-1]) as tf.Tensor1D;

    const div = cx(dFdx, 0).add(cx(dFdy, 1)) as tf.Tensor1D;
    const curl = cx(dFdx, 1).sub(cx(dFdy, 0)) as tf.Tensor1D;
    const s1 = cx(dFdx, 0).sub(cx(dFdy, 1)) as tf.Tensor1D;
    const s2 = cx(dFdx, 1).add(cx(dFdy, 0)) as tf.Tensor1D;
    const strain2 = s1.square().add(s2.square()) as tf.Tensor1D;

    // unit_τ so a near-zero force cannot manufacture a direction.
    const mag = f.square().sum(1).add(TAU * TAU).sqrt().reshape([-1, 1]) as tf.Tensor2D;
    const u = f.div(mag) as tf.Tensor2D;
    const ux = cx(u, 0);
    const uy = cx(u, 1);
    const r1 = ux.mean().square().add(uy.mean().square()).sqrt();
    // e^{2iθ} = (ux²−uy², 2·ux·uy) / |u|²; |u| ≈ 1 for |F| ≫ τ.
    const n2 = ux.square().add(uy.square()).add(1e-12) as tf.Tensor1D;
    const c2 = ux.square().sub(uy.square()).div(n2) as tf.Tensor1D;
    const s2n = ux.mul(uy).mul(2).div(n2) as tf.Tensor1D;
    const r2 = c2.mean().square().add(s2n.mean().square()).sqrt();

    const msCurl = curl.square().mean();
    const msStrain = strain2.mean();
    // Head outputs are tanh, so |F| ≤ (1−α)+α = 1 per component. "Saturated"
    // means BOTH components are pinned — the state that reads as an
    // axis-clipped diagonal on screen.
    const dcVec = f.mean(0) as tf.Tensor1D;
    const ac = f.sub(dcVec.reshape([1, 2])).square().sum(1).mean().sqrt();
    const sat = tf
      .minimum(cx(f, 0).abs(), cx(f, 1).abs())
      .greater(0.9)
      .toFloat()
      .mean();
    return tf.stack([
      r1,
      r2,
      f.square().sum(1).sqrt().mean(),
      dcVec.square().sum().sqrt(),
      ac,
      sat,
      div.square().mean().sqrt(),
      msCurl.sqrt(),
      msStrain.sqrt(),
      msStrain.sub(msCurl).div(msStrain.add(msCurl).add(1e-12)),
    ]);
  });
  const a = vals.dataSync();
  vals.dispose();
  return {
    r1: a[0],
    r2: a[1],
    meanMag: a[2],
    dc: a[3],
    ac: a[4],
    sat: a[5],
    rmsDiv: a[6],
    rmsCurl: a[7],
    rmsStrain: a[8],
    okuboWeiss: a[9],
  };
}

/* ── prototype generator regularizers (the fix under test) ──────────────── */

/**
 * BATCH-MEAN-DIRECTION penalty — candidate (a).
 * `λ·‖mean_B unit_τ(F)‖²`. One reduction + broadcast; kills the global
 * constant mode but says nothing about local rotation.
 */
function polarPenalty(f: tf.Tensor2D, lambda: number, nematic: number): tf.Scalar {
  if (lambda === 0 && nematic === 0) return tf.scalar(0);
  return tf.tidy(() => {
    const mag = f.square().sum(1).add(TAU * TAU).sqrt().reshape([-1, 1]) as tf.Tensor2D;
    const u = f.div(mag) as tf.Tensor2D;
    const ux = u.slice([0, 0], [-1, 1]).reshape([-1]) as tf.Tensor1D;
    const uy = u.slice([0, 1], [-1, 1]).reshape([-1]) as tf.Tensor1D;
    const polar = ux.mean().square().add(uy.mean().square()).mul(lambda);
    // e^{2iθ}·|u|² = (ux²−uy², 2·ux·uy). Using the UNNORMALIZED double-angle
    // vector keeps the term smooth at F=0 (it vanishes there) instead of
    // dividing by a norm that goes to zero.
    const c2 = ux.square().sub(uy.square()) as tf.Tensor1D;
    const s2 = ux.mul(uy).mul(2) as tf.Tensor1D;
    const nem = c2.mean().square().add(s2.mean().square()).mul(nematic);
    return polar.add(nem).asScalar();
  });
}

/**
 * MAGNITUDE HINGE — candidate (c). `λ·mean(relu(‖F‖ − t)²)`. A direction-only
 * loss stops caring about ‖F‖ once ‖F‖ ≫ τ, so nothing else in the objective
 * opposes the ratchet into tanh saturation.
 */
function magPenalty(f: tf.Tensor2D, lambda: number): tf.Scalar {
  if (lambda === 0) return tf.scalar(0);
  return tf.tidy(() =>
    f
      .square()
      .sum(1)
      .add(1e-12)
      .sqrt()
      .sub(MAG_T)
      .relu()
      .square()
      .mean()
      .mul(lambda)
      .asScalar()
  );
}

/**
 * NORMALIZED OKUBO–WEISS penalty — candidate (b), the mechanism-targeted one.
 *
 *   OW = (⟨|S|²⟩ − ⟨ω²⟩) / (⟨|S|²⟩ + ⟨ω²⟩ + eps)  ∈ [−1, 1]
 *
 * Minimizing it moves the field from strain-dominated (laminar shear sheets,
 * OW > 0) to rotation-dominated (vortices, OW < 0). Scale-free by
 * construction, so the generator cannot pay for it by shrinking |F|, and it is
 * exactly the quantity the pair game's soft-angle payoff pushes the WRONG way
 * (see the note). Central differences ⇒ four extra FORWARD passes and a
 * first-order backward.
 */
function swirlPenalty(pos: tf.Tensor2D, lambda: number): tf.Scalar {
  if (lambda === 0) return tf.scalar(0);
  return tf.tidy(() => {
    const shift = (dx: number, dy: number) =>
      field.forces(pos.add(tf.tensor2d([[dx, dy]])) as tf.Tensor2D);
    const dFdx = shift(FD_H, 0).sub(shift(-FD_H, 0)).div(2 * FD_H) as tf.Tensor2D;
    const dFdy = shift(0, FD_H).sub(shift(0, -FD_H)).div(2 * FD_H) as tf.Tensor2D;
    const cx = (t: tf.Tensor2D, c: number) =>
      t.slice([0, c], [-1, 1]).reshape([-1]) as tf.Tensor1D;
    const curl2 = cx(dFdx, 1).sub(cx(dFdy, 0)).square().mean();
    const strain2 = cx(dFdx, 0)
      .sub(cx(dFdy, 1))
      .square()
      .add(cx(dFdx, 1).add(cx(dFdy, 0)).square())
      .mean();
    return strain2
      .sub(curl2)
      .div(strain2.add(curl2).add(1e-12))
      .mul(lambda)
      .asScalar();
  });
}

/* ── the loop ───────────────────────────────────────────────────────────── */

console.log(
  `collapse preset=${PRESET} arch=${ARCH_TAG} steps=${STEPS} B=${B} ` +
    `k=${adv.k} eps=${P.kind.relaxEps} tau=${TAU} weight=${WEIGHT} ` +
    `gLR=${GLR} dLR=${DLR} SWIRL=${SWIRL} POLAR=${POLAR} NEMATIC=${NEMATIC} ` +
    `MAG=${MAG}@${MAG_T} ` +
    `ANCHOR=${ANCHOR}@${ANCHOR_T} seed=${SEED}`
);
console.log(
  "step\tR1\tR2\t|F|\tDC\tAC\tsat\trmsCurl\trmsStrain\tOW\tDloss"
);

const rowRnd = (() => {
  let a = (SEED ^ 0x9e3779b9) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
})();
function uniformBatch(): tf.Tensor2D {
  const a = new Float32Array(B * 2);
  for (let i = 0; i < a.length; i++) a[i] = rowRnd();
  return tf.tensor2d(a, [B, 2]);
}

let lastLoss = NaN;
const trace: { step: number; d: Diagnostics; dLoss: number }[] = [];
for (let step = 0; step <= STEPS; step++) {
  // (1a) DISCRIMINATOR on its own detached uniform batch — no tape is open,
  //      matching src/main.ts (1a).
  tf.tidy(() => {
    const pn = uniformBatch();
    const raw = field.forces(pn);
    const s = adv.sampleTarget({ tag: "force", pos: pn, force: raw });
    lastLoss = adv.trainStep(s).loss;
    disposeTupleSample(s);
  });

  // (1b) GENERATOR — varList is the FIELD's weights only.
  const cost = gen.minimize(
    () =>
      tf.tidy(() => {
        const pn = uniformBatch();
        const raw = field.forces(pn);
        const s = adv.sampleTarget({ tag: "force", pos: pn, force: raw });
        const term = adv.generatorLoss(s).mul(WEIGHT).asScalar();
        // Two-sided anchor on the ENCODED target — differentiable through the
        // same `s.y` the payoff uses, so it directly opposes the pole exploit.
        const anchor =
          ANCHOR === 0
            ? tf.scalar(0)
            : (s.y.square().sum(1).mean().add(1e-12).sqrt().sub(ANCHOR_T).square().mul(
                ANCHOR
              ) as tf.Scalar);
        disposeTupleSample(s);
        return term
          .add(anchor)
          .add(polarPenalty(raw, POLAR, NEMATIC))
          .add(magPenalty(raw, MAG))
          .add(swirlPenalty(pn, SWIRL))
          .asScalar();
      }),
    true,
    varList
  ) as tf.Scalar;
  cost.dispose();

  if (step % REPORT === 0 || step === STEPS) {
    const d = diagnostics();
    trace.push({ step, d, dLoss: lastLoss });
    console.log(
      `${step}\t${d.r1.toFixed(4)}\t${d.r2.toFixed(4)}\t${d.meanMag.toFixed(4)}\t` +
        `${d.dc.toFixed(4)}\t${d.ac.toFixed(4)}\t${d.sat.toFixed(3)}\t` +
        `${d.rmsCurl.toFixed(3)}\t${d.rmsStrain.toFixed(3)}\t` +
        `${d.okuboWeiss.toFixed(4)}\t${lastLoss.toExponential(3)}`
    );
  }
}

/**
 * `IMG=<path.ppm>` — direction-field picture, because headless Chrome on this
 * box reports `adapter: null` and cannot render the live WebGPU page at all
 * (the caveat in AGENTS.md). Hue = arg F, value = ‖F‖ percentile. A laminar
 * collapse is one flat colour; genuine vortices are visible colour wheels.
 * Convert with `sips -s format png <path.ppm> --out <path.png>`.
 */
const IMG = process.env.IMG ?? "";
if (IMG) {
  const R = 256;
  const px = new Uint8Array(R * R * 3);
  const f = tf.tidy(() => {
    const a = new Float32Array(R * R * 2);
    for (let j = 0; j < R; j++) {
      for (let i = 0; i < R; i++) {
        a[2 * (j * R + i)] = (i + 0.5) / R;
        a[2 * (j * R + i) + 1] = (j + 0.5) / R;
      }
    }
    return field.forces(tf.tensor2d(a, [R * R, 2])).dataSync();
  });
  const mags: number[] = [];
  for (let i = 0; i < R * R; i++) mags.push(Math.hypot(f[2 * i], f[2 * i + 1]));
  const sorted = mags.slice().sort((a, b) => a - b);
  const hi = sorted[Math.floor(0.98 * sorted.length)] || 1;
  for (let i = 0; i < R * R; i++) {
    const h = ((Math.atan2(f[2 * i + 1], f[2 * i]) / Math.PI + 1) * 3) % 6;
    const v = Math.min(1, mags[i] / hi);
    const x = 1 - Math.abs((h % 2) - 1);
    const rgb =
      h < 1 ? [1, x, 0] : h < 2 ? [x, 1, 0] : h < 3 ? [0, 1, x]
      : h < 4 ? [0, x, 1] : h < 5 ? [x, 0, 1] : [1, 0, x];
    for (let c = 0; c < 3; c++) px[3 * i + c] = Math.round(255 * rgb[c] * (0.25 + 0.75 * v));
  }
  const header = new TextEncoder().encode(`P6\n${R} ${R}\n255\n`);
  const out = new Uint8Array(header.length + px.length);
  out.set(header, 0);
  out.set(px, header.length);
  await Bun.write(IMG, out);
  console.log(`IMAGE ${IMG}`);
}

const first = trace[0];
const last = trace[trace.length - 1];
console.log(
  `SUMMARY preset=${PRESET} SWIRL=${SWIRL} POLAR=${POLAR} NEMATIC=${NEMATIC} MAG=${MAG} ` +
    `ANCHOR=${ANCHOR} tau=${TAU} ` +
    `R1 ${first.d.r1.toFixed(4)}→${last.d.r1.toFixed(4)} ` +
    `R2 ${first.d.r2.toFixed(4)}→${last.d.r2.toFixed(4)} ` +
    `|F| ${first.d.meanMag.toFixed(4)}→${last.d.meanMag.toFixed(4)} ` +
    `AC ${first.d.ac.toFixed(4)}→${last.d.ac.toFixed(4)} ` +
    `DC/AC ${(first.d.dc / first.d.ac).toFixed(3)}→${(last.d.dc / last.d.ac).toFixed(3)} ` +
    `sat ${first.d.sat.toFixed(3)}→${last.d.sat.toFixed(3)} ` +
    `OW ${first.d.okuboWeiss.toFixed(4)}→${last.d.okuboWeiss.toFixed(4)}`
);
