/**
 * ADVERSARY WIRING verification — the app-level coupling, not the adversary math
 * (that is tools/adversary_test.ts).
 *
 *   bun tools/adversary_wire_test.ts
 *
 * What is actually at risk once the adversary is plugged into the gallery loop,
 * and what each section gates:
 *
 *   §1 PARAMETER SEPARATION. The brief's hard requirement: "the adversary update
 *      must NOT co-optimize the generator field, and vice versa". Both
 *      directions are asserted BEHAVIOURALLY (field forces / head predictions on
 *      fixed probes, bit-for-bit) rather than by inspecting varLists, so a
 *      refactor that accidentally widens either optimizer's varList fails here
 *      even if the plumbing still looks right.
 *   §2 REWARD SIGN THROUGH RAW F(x). The strict game observes the learned field
 *      output before forceMagnitude, friction, velocity, clipping or borders.
 *      A sign slip or lost tape produces a predictor that trains while the
 *      generator does nothing.
 *   §3 URL κ. `?advWeight=0` is the specific bug the rest of this file's
 *      `parseFloat(x) || default` idiom would introduce: a legitimate zero
 *      silently rewritten to the default, i.e. the "run the instrument but do
 *      not steer" mode quietly steering.
 *   §4 GALLERY COHERENCE. A piece that declares `renderer: "surprise"` without
 *      an adversary renders by velocity — the name would lie about the picture.
 *   §5 LEAKS. `adversaryTrainStep` cannot use `tf.tidy` (tidy returns a
 *      `TensorContainer`, the step returns a telemetry record), so it disposes by
 *      hand — the classic one-tensor-per-frame leak that only shows up an hour in.
 *   §6 DIAGNOSTIC ROUTING. Raw/per-unit modes are explicit, while the tfjs
 *      reference path cannot silently substitute a particle-transition metric.
 *   §8 PIXEL CRITIC GATE. The fused field trainer's pass B sums at most TWO
 *      external gradients. A piece declaring the Agree+Disagree game AND a
 *      pixel critic wants three, and the old `if (advTrainerB) … else if
 *      (pixelDiscTrainer)` silently dropped the critic's — it trained, logged
 *      "[pixel-disc] FUSED" and dispatched ten passes per frame at 0% of the
 *      field update. Every way of silencing a declared critic must be a NAMED
 *      state with a reason, and that particular combination must not be
 *      configurable at all.
 *
 * Runs on the tfjs CPU backend; no GPU, no DOM.
 */
import * as tf from "@tensorflow/tfjs";
import { readFileSync } from "node:fs";
import {
  GALLERY,
  resolveAdversary,
  resolveColorMode,
  createAdversary,
  adversaryGeneratorTerm,
  adversaryTrainStep,
  classifyHeads,
  combineHeadHealth,
  forceMagnitudeForDrive,
  physicsForward,
  surpriseMetricOf,
  resolvePixelCritic,
  type ArtPieceConfig,
  type TfjsAdversaryRuntime,
} from "../src/main";
import { HelmholtzField } from "../src/core/field/helmholtz";
import { layoutField, type Encoding } from "../src/render/webgpu/advect_wgsl";

let failures = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
  if (!cond) failures++;
};

const W = 800;
const H = 600;
const N = 256;

/** Deterministic PRNG (repo standard). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fixedPositions(seed: number, n: number): tf.Tensor2D {
  const rnd = mulberry32(seed);
  const a = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    a[2 * i] = rnd() * W;
    a[2 * i + 1] = rnd() * H;
  }
  return tf.tensor2d(a, [n, 2]);
}

/** The piece under test: pair encoding, k = 4, so both encodings are exercised
 *  by §1/§2 in their hardest form (du = 1). */
const CFG: ArtPieceConfig = {
  ...GALLERY.find((g) => g.name === "Adversary · Pair WTA K=4")!,
  particleCount: N,
};

/** A field probe that is invariant to everything except the field weights. */
function fieldFingerprint(field: HelmholtzField, probe: tf.Tensor2D): Float32Array {
  return tf.tidy(() => new Float32Array(field.forces(probe).dataSync()));
}

/** A head probe that is invariant to everything except the predictor weights. */
function headFingerprint(rt: TfjsAdversaryRuntime, u: tf.Tensor2D): Float32Array {
  return tf.tidy(() => new Float32Array(rt.adv.predictHeads(u).dataSync()));
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/** Raw learned output on pixel-space positions. Caller owns the tensor. */
function rawSignal(field: HelmholtzField, pos: tf.Tensor2D): tf.Tensor2D {
  return tf.tidy(() => {
    const wh = tf.tensor2d([[W, H]]);
    return field.forces(pos.div(wh) as tf.Tensor2D) as tf.Tensor2D;
  });
}

/** ONE strict generator step, composed on raw F(x) exactly as `tick()` does. */
function generatorStep(
  opt: tf.Optimizer,
  field: HelmholtzField,
  rt: TfjsAdversaryRuntime,
  pos: tf.Tensor2D,
  vel: tf.Tensor2D,
  weightOverride?: number
): void {
  opt.minimize(
    () =>
      tf.tidy(() => {
        const signal = rawSignal(field, pos);
        const adversarial = adversaryGeneratorTerm(
          rt,
          pos,
          vel,
          vel,
          signal,
          W,
          H,
          1
        );
        return adversarial.mul(weightOverride ?? 1).asScalar();
      }),
    false,
    field.trainableWeights
  );
}

async function main(): Promise<void> {
  await tf.setBackend("cpu");
  await tf.ready();
  console.log(`backend ${tf.getBackend()}`);

  // ══════════════════════════════════════════════════════════════════════
  console.log("\n§1 PARAMETER SEPARATION — neither side may move the other");
  // ══════════════════════════════════════════════════════════════════════
  {
    const field = new HelmholtzField({ alpha: 0.6 });
    const rt = createAdversary(
      resolveAdversary(CFG.adversary, new URLSearchParams()),
      128,
      "periodic"
    );
    if (rt.tag !== "on") throw new Error("§1: expected the piece to declare an adversary");

    const pos = fixedPositions(11, N);
    const vel = tf.zeros([N, 2]) as tf.Tensor2D;
    const probe = tf.tidy(() => fixedPositions(77, 64).div(tf.tensor2d([[W, H]])) as tf.Tensor2D);
    const uProbe = tf.tensor2d(
      Float32Array.from({ length: 32 }, (_, i) => 0.02 + i * 0.01),
      [32, 1]
    );

    const field0 = fieldFingerprint(field, probe);
    const head0 = headFingerprint(rt, uProbe);

    // --- discriminator only -------------------------------------------------
    for (let s = 0; s < 6; s++) {
      const signal = rawSignal(field, pos);
      adversaryTrainStep(rt, pos, vel, vel, signal, W, H, 1);
      signal.dispose();
    }
    const fieldAfterD = fieldFingerprint(field, probe);
    const headAfterD = headFingerprint(rt, uProbe);
    ok(
      maxAbsDiff(field0, fieldAfterD) === 0,
      `6 discriminator steps left the field EXACTLY unchanged (max|Δforce| = ${maxAbsDiff(
        field0,
        fieldAfterD
      ).toExponential(2)})`
    );
    ok(
      maxAbsDiff(head0, headAfterD) > 1e-5,
      `...while moving the predictor heads (max|Δpred| = ${maxAbsDiff(
        head0,
        headAfterD
      ).toExponential(2)})`
    );

    // --- generator only -----------------------------------------------------
    const opt = tf.train.adam(0.01);
    const headBeforeG = headFingerprint(rt, uProbe);
    const fieldBeforeG = fieldFingerprint(field, probe);
    for (let s = 0; s < 6; s++) generatorStep(opt, field, rt, pos, vel);
    const headAfterG = headFingerprint(rt, uProbe);
    const fieldAfterG = fieldFingerprint(field, probe);
    ok(
      maxAbsDiff(headBeforeG, headAfterG) === 0,
      `6 generator steps left the predictor heads EXACTLY unchanged (max|Δpred| = ${maxAbsDiff(
        headBeforeG,
        headAfterG
      ).toExponential(2)})`
    );
    ok(
      maxAbsDiff(fieldBeforeG, fieldAfterG) > 1e-6,
      `...while moving the field (max|Δforce| = ${maxAbsDiff(
        fieldBeforeG,
        fieldAfterG
      ).toExponential(2)})`
    );

    opt.dispose();
    rt.adv.dispose();
    field.dispose();
    pos.dispose();
    vel.dispose();
    probe.dispose();
    uProbe.dispose();
  }

  // ══════════════════════════════════════════════════════════════════════
  console.log("\n§2 REWARD SIGN — descending the term must RAISE surprise");
  // ══════════════════════════════════════════════════════════════════════
  {
    const field = new HelmholtzField({ alpha: 0.6 });
    const rt = createAdversary(
      resolveAdversary(CFG.adversary, new URLSearchParams()),
      128,
      "periodic"
    );
    if (rt.tag !== "on") throw new Error("§2: expected an adversary");

    const pos = fixedPositions(23, N);
    const vel = tf.zeros([N, 2]) as tf.Tensor2D;
    const wh = tf.tensor2d([[W, H]]);

    // Let the predictor get a foothold on the untouched field first, otherwise
    // "surprise went up" would just be measuring an untrained head.
    for (let s = 0; s < 40; s++) {
      const signal = rawSignal(field, pos);
      adversaryTrainStep(rt, pos, vel, vel, signal, W, H, 1);
      signal.dispose();
    }

    // FIXED probe tuples, so the before/after numbers are the same measurement.
    // The predictor is FROZEN for the rest of this section: this isolates the
    // generator's gradient path, which is the thing under test.
    const m = rt.adv.dims.m;
    const tuples = 96;
    const idx = new Int32Array(tuples * m);
    const rnd = mulberry32(5);
    for (let t = 0; t < tuples; t++) {
      for (let j = 0; j < m; j++) idx[t * m + j] = Math.floor(rnd() * N);
    }
    const probeSurprise = (): number =>
      tf.tidy(() => {
        const signal = rawSignal(field, pos);
        const s = rt.adv.encodeSignal(
          pos.div(wh) as tf.Tensor2D,
          signal,
          idx
        );
        return rt.adv.surprise(s).mean().dataSync()[0];
      });
    const probeGeneratorLoss = (): tf.Scalar =>
      tf.tidy(() => {
        const signal = rawSignal(field, pos);
        const s = rt.adv.encodeSignal(pos.div(wh) as tf.Tensor2D, signal, idx);
        return rt.adv.generatorLoss(s);
      });

    const before = probeSurprise();
    const vars = field.trainableWeights;
    const { grads } = tf.variableGrads(probeGeneratorLoss, vars);
    const saved = vars.map((v) => v.clone());
    let maxGrad = 0;
    for (const g of Object.values(grads)) {
      const m = g.abs().max().dataSync()[0];
      maxGrad = Math.max(maxGrad, m);
    }
    // Stay inside one WTA/chart branch: this is a directional-derivative gate,
    // not an optimizer-quality test.
    const eta = 1e-7 / Math.max(maxGrad, 1e-12);
    for (let i = 0; i < vars.length; i++) {
      const g = grads[vars[i].name];
      tf.tidy(() => vars[i].assign(saved[i].sub(g.mul(eta))));
    }
    const after = probeSurprise();
    ok(
      after > before,
      `one normalized gradient-descent step raised the SAME fixed-tuple payoff ` +
        `${before.toExponential(7)} -> ${after.toExponential(
        7
      )} (${(after / Math.max(before, 1e-30)).toFixed(2)}x)`
    );

    // Restore and move by the exact opposite directional derivative. This
    // avoids Adam momentum and long-step nonlinearity obscuring the sign gate.
    for (let i = 0; i < vars.length; i++) {
      const g = grads[vars[i].name];
      tf.tidy(() => vars[i].assign(saved[i].add(g.mul(eta))));
    }
    const flipped = probeSurprise();
    ok(
      flipped < before,
      `the equal-size opposite step lowered the SAME payoff ${before.toExponential(
        7
      )} -> ${flipped.toExponential(7)}`
    );

    for (let i = 0; i < vars.length; i++) vars[i].assign(saved[i]);
    for (const x of saved) x.dispose();
    for (const g of Object.values(grads)) g.dispose();
    rt.adv.dispose();
    field.dispose();
    pos.dispose();
    vel.dispose();
    wh.dispose();
  }

  // The live loop passes forceMagnitude as the final physicsForward argument.
  // Keep this as a behavioral gate: Parcel strips TypeScript without
  // type-checking, so an accidentally removed parameter otherwise becomes a
  // browser-only ReferenceError while the fused renderer continues drawing.
  {
    const field = new HelmholtzField({ alpha: 0.6 });
    const pos = fixedPositions(29, 32);
    const vel = tf.randomUniform([32, 2], -2, 2, "float32", 991) as tf.Tensor2D;
    const [unitForce, tripleForce, postVelocityDiff] = tf.tidy(() => {
      const unit = physicsForward(
        pos,
        vel,
        null,
        field,
        CFG,
        W,
        H,
        CFG.maxVelocity,
        { tag: "wrap" },
        1
      );
      const triple = physicsForward(
        pos,
        vel,
        null,
        field,
        CFG,
        W,
        H,
        CFG.maxVelocity,
        { tag: "wrap" },
        3
      );
      const expectedPost = vel
        .add(triple.rawSignal.mul(3))
        .mul(CFG.friction)
        .clipByValue(-CFG.maxVelocity, CFG.maxVelocity);
      return [
        unit.force.abs().mean().dataSync()[0],
        triple.force.abs().mean().dataSync()[0],
        triple.postUpdateVelocity
          .sub(expectedPost)
          .abs()
          .max()
          .dataSync()[0],
      ];
    });
    ok(
      Math.abs(tripleForce / unitForce - 3) < 1e-5,
      `physicsForward consumes its live forceMagnitude argument (${unitForce.toExponential(
        4
      )} -> ${tripleForce.toExponential(4)})`
    );
    ok(
      postVelocityDiff < 1e-7,
      `post-velocity target source is exactly force + friction + clip before borders ` +
        `(maxΔ ${postVelocityDiff.toExponential(2)})`
    );
    const normalizedVelocityDiff = tf.tidy(() => {
      const z = tf.randomUniform([32, 2], -0.5, 0.5, "float32", 992);
      const drive = 0.65;
      const run = (maxV: number) =>
        physicsForward(
          pos,
          z.mul(maxV) as tf.Tensor2D,
          null,
          field,
          CFG,
          W,
          H,
          maxV,
          { tag: "wrap" },
          forceMagnitudeForDrive(drive, maxV, CFG.friction)
        ).postUpdateVelocity.div(maxV);
      return run(10).sub(run(40)).abs().max().dataSync()[0];
    });
    ok(
      normalizedVelocityDiff < 2e-7,
      `normalized post-velocity is invariant when velocity and maxVelocity scale together ` +
        `(maxΔ ${normalizedVelocityDiff.toExponential(2)})`
    );
    field.dispose();
    pos.dispose();
    vel.dispose();
  }

  // ══════════════════════════════════════════════════════════════════════
  console.log("\n§3 URL κ");
  // ══════════════════════════════════════════════════════════════════════
  {
    const piece = CFG.adversary!;
    const zero = resolveAdversary(piece, new URLSearchParams("advWeight=0"));
    ok(
      zero.tag === "on" && zero.weight === 0,
      `?advWeight=0 survives as 0 (got ${zero.tag === "on" ? zero.weight : "off"}) — ` +
        `the "|| default" idiom would have silently restored ${
          piece.tag === "on" ? piece.weight : "?"
        }`
    );
    const off = resolveAdversary(piece, new URLSearchParams("adv=off"));
    ok(off.tag === "off", "?adv=off overrides a piece that declares an adversary");
    const forcedOn = resolveAdversary(undefined, new URLSearchParams("adv=wta"));
    ok(
      forcedOn.tag === "on" && forcedOn.weight === 0.012,
      "?adv=wta on a non-adversary piece uses the current in-range reward default"
    );
    const clampedHigh = resolveAdversary(
      undefined,
      new URLSearchParams("adv=wta&advWeight=1")
    );
    const clampedLow = resolveAdversary(
      undefined,
      new URLSearchParams("adv=wta&advWeight=-1")
    );
    ok(
      clampedHigh.tag === "on" &&
        clampedHigh.weight === 0.05 &&
        clampedLow.tag === "on" &&
        clampedLow.weight === 0,
      "?advWeight is canonicalized to the live reward control range [0, 0.05]"
    );
    const k3 = resolveAdversary(piece, new URLSearchParams("adv=wta&advK=3&advEps=0.2"));
    ok(
      k3.tag === "on" && k3.kind.tag === "wta" && k3.kind.k === 3 && k3.kind.relaxEps === 0.2,
      "?adv=wta&advK=3&advEps=0.2 lands on the kind"
    );
    const m1 = resolveAdversary(piece, new URLSearchParams("advM=1"));
    ok(
      m1.tag === "on" && m1.encoding.tag === "point",
      "?advM=1 overrides a pair piece to the point encoding"
    );
    const kept = resolveAdversary(piece, new URLSearchParams(""));
    ok(
      kept.tag === "on" &&
        kept.encoding.tag === "pair-rotation-scale-adjusted" &&
        kept.kind.tag === "wta",
      "no params keeps the piece's explicit adjusted pair spec"
    );
    const m3 = resolveAdversary(piece, new URLSearchParams("advM=3"));
    ok(
      m3.tag === "on" && m3.encoding.tag === "tri",
      "?advM=3 lands on the tri encoding (was a hard error before tri existed)"
    );
    const m2 = resolveAdversary(piece, new URLSearchParams("advM=2"));
    ok(
      m2.tag === "on" && m2.encoding.tag === "pair-rotation",
      "?advM=2 uses the explicit pair-rotation tag, not the legacy pair spelling"
    );
    const m4 = resolveAdversary(piece, new URLSearchParams("advM=4"));
    ok(
      m4.tag === "on" && m4.encoding.tag === "quad-labelled",
      "?advM=4 selects the supported LABELLED quad (no permutation-invariance claim)"
    );
    const postV = resolveAdversary(
      piece,
      new URLSearchParams(
        "advM=1&advTarget=post-velocity&advLoss=soft-angle&advTau=0.08"
      )
    );
    ok(
      postV.tag === "on" &&
        postV.encoding.tag === "point" &&
        postV.target?.tag === "post-velocity" &&
        postV.loss?.tag === "soft-angle" &&
        postV.loss.tau === 0.08,
      "post-velocity + soft-angle URL contract resolves without hiding incoming velocity"
    );
    const scaleMode = resolveAdversary(
      piece,
      new URLSearchParams(
        "advLoss=angle-scale-hold&advTau=0.06&advScaleWeight=0.7&" +
          "advEnergyWeight=0.2&advEnergyTarget=0.4"
      )
    );
    ok(
      scaleMode.tag === "on" &&
        scaleMode.loss?.tag === "angle-scale-hold" &&
        scaleMode.loss.tau === 0.06 &&
        scaleMode.loss.scaleWeight === 0.7 &&
        scaleMode.loss.energyWeight === 0.2 &&
        scaleMode.loss.energyTarget === 0.4,
      "angle+scale-hold URL parameters retain both prediction terms and fixed energy"
    );
    let threwInvalidObjective = false;
    try {
      resolveAdversary(
        piece,
        new URLSearchParams("advM=2&advTarget=post-velocity")
      );
    } catch (_) {
      threwInvalidObjective = true;
    }
    ok(
      threwInvalidObjective,
      "relational post-velocity throws instead of claiming a false rotation quotient"
    );
    let threwLoss = false;
    try {
      resolveAdversary(piece, new URLSearchParams("advLoss=bogus"));
    } catch (_) {
      threwLoss = true;
    }
    ok(threwLoss, "unknown adversarial loss throws instead of silently changing games");
    // UI policy: URL adversary knobs are GLOBAL across gallery selections.
    // defaultsForPiece re-resolves every selected piece through this same query,
    // then passes those canonical values to startLoop. Without a URL, each piece
    // still keeps its own declared defaults.
    const globalQ = new URLSearchParams("adv=wta&advK=3&advM=4&advEps=0.1");
    const pointPiece = GALLERY.find((g) => g.name === "Adversary · WTA K=8")!;
    const globalPoint = resolveAdversary(pointPiece.adversary, globalQ);
    const globalPair = resolveAdversary(piece, globalQ);
    ok(
      globalPoint.tag === "on" &&
        globalPair.tag === "on" &&
        globalPoint.kind.tag === "wta" &&
        globalPair.kind.tag === "wta" &&
        globalPoint.kind.k === 3 &&
        globalPair.kind.k === 3 &&
        globalPoint.kind.relaxEps === 0.1 &&
        globalPair.kind.relaxEps === 0.1 &&
        globalPoint.encoding.tag === "quad-labelled" &&
        globalPair.encoding.tag === "quad-labelled",
      "global ?advK/?advM/?advEps policy survives gallery selection"
    );
    const pointDefault = resolveAdversary(pointPiece.adversary, new URLSearchParams());
    ok(
      pointDefault.tag === "on" &&
        pointDefault.kind.tag === "wta" &&
        pointDefault.kind.k === 8 &&
        pointDefault.encoding.tag === "point",
      "without URL overrides, gallery selection preserves each piece's declared defaults"
    );
    let threwMode = false;
    try {
      resolveAdversary(piece, new URLSearchParams("adv=bogus"));
    } catch (_) {
      threwMode = true;
    }
    ok(threwMode, '?adv=bogus throws instead of falling through to "off"');
  }

  // ══════════════════════════════════════════════════════════════════════
  console.log("\n§4 GALLERY COHERENCE");
  // ══════════════════════════════════════════════════════════════════════
  {
    const q = new URLSearchParams("");
    let mismatches = 0;
    let advPieces = 0;
    let surprisePieces = 0;
    for (const piece of GALLERY) {
      const spec = resolveAdversary(piece.adversary, q);
      const mode = resolveColorMode(piece, spec, q);
      if (spec.tag === "on") advPieces++;
      if (piece.renderer === "surprise") surprisePieces++;
      if ((piece.renderer === "surprise") !== (mode.tag !== "velocity")) {
        console.log(`      mismatch: ${piece.name} renderer=${piece.renderer} mode=${mode.tag}`);
        mismatches++;
      }
    }
    ok(
      mismatches === 0,
      `every renderer:"surprise" piece resolves to the surprise colour mode ` +
        `(${surprisePieces} such pieces, 0 mismatches)`
    );
    ok(advPieces >= 4, `gallery ships ${advPieces} adversary pieces (>= 4 required)`);
    ok(
      surprisePieces >= 2,
      `gallery ships ${surprisePieces} surprise-rendered pieces (>= 2 required)`
    );
    // Every adversary piece must expose a trainable field/model. Production
    // consumes it in fused WGSL; the explicit tfjs fallback uses the same
    // declaration as its independent reference. A piece with neither would
    // silently train nothing on either path.
    const trainable = GALLERY.filter(
      (p) => resolveAdversary(p.adversary, q).tag === "on"
    ).every((p) => !!p.createField || !!p.createModel);
    ok(trainable, "every adversary piece supplies a field or a model to train");
  }

  // ══════════════════════════════════════════════════════════════════════
  console.log("\n§4b FUSED RUNTIME — zero idle tfjs predictor/Adam tensors");
  // ══════════════════════════════════════════════════════════════════════
  {
    const spec = resolveAdversary(CFG.adversary, new URLSearchParams());
    const agreePiece = GALLERY.find(
      (piece) => piece.name === "Adversary · Agree + Disagree RGB"
    )!;
    const agreeSpec = resolveAdversary(
      agreePiece.adversary,
      new URLSearchParams()
    );
    const baseline = tf.memory().numTensors;
    const fused = createAdversary(spec, 64, "periodic", 3e-3, "fused");
    const fusedAgree = createAdversary(
      agreeSpec,
      64,
      "periodic",
      3e-3,
      "fused"
    );
    const fusedDelta = tf.memory().numTensors - baseline;
    ok(
      fused.tag === "on" &&
        fused.implementation === "fused" &&
        !("adv" in fused) &&
        fusedAgree.tag === "on" &&
        fusedAgree.implementation === "fused" &&
        !("adv" in fusedAgree) &&
        fusedDelta === 0,
      `ordinary + Agree/Disagree fused runtime construction retained ${fusedDelta} tfjs tensors`
    );

    const reference = createAdversary(spec, 64, "periodic", 3e-3);
    const referenceDelta = tf.memory().numTensors - baseline;
    ok(
      reference.tag === "on" &&
        reference.implementation === "tfjs" &&
        referenceDelta > 0,
      `explicit tfjs fallback still constructs its reference game (+${referenceDelta} tensors)`
    );
    if (reference.tag === "on") reference.adv.dispose();
    const disposedDelta = tf.memory().numTensors - baseline;
    ok(
      disposedDelta === 0,
      `disposing the tfjs fallback returns to baseline (net ${disposedDelta})`
    );
  }

  // ══════════════════════════════════════════════════════════════════════
  console.log("\n§5 LEAKS — adversaryTrainStep disposes BY HAND");
  // ══════════════════════════════════════════════════════════════════════
  // `tf.tidy` cannot wrap it: tidy's return type is `TensorContainer` and the
  // step returns a telemetry record. Every intermediate is therefore disposed
  // explicitly, which is exactly the kind of code that leaks one tensor per
  // frame and is invisible until the tab dies twenty minutes in.
  {
    const field = new HelmholtzField({ alpha: 0.6 });
    const rt = createAdversary(
      resolveAdversary(CFG.adversary, new URLSearchParams()),
      64,
      "periodic"
    );
    if (rt.tag !== "on") throw new Error("§5: expected an adversary");
    const pos = fixedPositions(31, N);
    const vel = tf.zeros([N, 2]) as tf.Tensor2D;
    const opt = tf.train.adam(0.005);

    const step = () => {
      const signal = rawSignal(field, pos);
      adversaryTrainStep(rt, pos, vel, vel, signal, W, H, 1);
      signal.dispose();
      generatorStep(opt, field, rt, pos, vel);
    };

    // Warm up first: Adam allocates its moment slots on the first apply, and
    // counting those as a leak would make the gate cry wolf forever.
    for (let i = 0; i < 3; i++) step();
    const base = tf.memory().numTensors;
    for (let i = 0; i < 25; i++) step();
    const delta = tf.memory().numTensors - base;
    ok(delta === 0, `25 full frames (discriminator + generator) net tensor delta = ${delta}`);

    opt.dispose();
    rt.adv.dispose();
    field.dispose();
    pos.dispose();
    vel.dispose();
  }

  // ══════════════════════════════════════════════════════════════════════
  console.log("\n§6 DIAGNOSTIC ROUTING — raw/per-unit explicit, tfjs fallback honest");
  // ══════════════════════════════════════════════════════════════════════
  {
    const raw = resolveColorMode(CFG, CFG.adversary!, new URLSearchParams("color=surprise"));
    const unit = resolveColorMode(
      CFG,
      CFG.adversary!,
      new URLSearchParams("color=surprise-per-unit")
    );
    const legacyUnit = resolveColorMode(
      CFG,
      CFG.adversary!,
      new URLSearchParams("color=surprise&surNorm=1")
    );
    ok(
      surpriseMetricOf(raw) === "raw-payoff",
      "?color=surprise remains the explicit raw-payoff compatibility alias"
    );
    ok(
      surpriseMetricOf(unit) === "per-unit-signal" &&
        surpriseMetricOf(legacyUnit) === "per-unit-signal",
      "?color=surprise-per-unit and legacy ?surNorm=1 select per-unit signal"
    );

    const noAdversary = { ...CFG, adversary: { tag: "off" } as const };
    const rejected = resolveColorMode(
      noAdversary,
      { tag: "off" },
      new URLSearchParams("color=surprise-per-unit")
    );
    ok(rejected.tag === "velocity", "a piece without an adversary rejects cloud diagnostics");

    // Source gate: the removed transition channel observed previous/next particle
    // positions, a different random variable from strict raw F(x). It must never
    // be resurrected as a silent tfjs fallback.
    const mainSource = readFileSync(new URL("../src/main.ts", import.meta.url), "utf8");
    ok(
      !mainSource.includes("class SurpriseChannel") &&
        !mainSource.includes("selectSurpriseTuples"),
      "transition-based SurpriseChannel is absent from the live source"
    );
    ok(
      mainSource.includes("tfjs is an oracle-only training path") &&
        mainSource.includes("raw/per-unit cloud ") &&
        mainSource.includes("diagnostics require the fused adversary"),
      "tfjs fallback rejection is loud and names the missing fused diagnostic"
    );
    ok(
      mainSource.includes("stats.headSpread"),
      "fused telemetry consumes measured predictor spread instead of staying unprobed"
    );
  }
  // ══════════════════════════════════════════════════════════════════════
  console.log("\n§7 TWO-STATE COLLAPSE VERDICT — threshold sits in the measured gap");
  // ══════════════════════════════════════════════════════════════════════
  // classifyHeads' pileup threshold is anchored to tools/adversary_test.ts §10's
  // MEASURED states (k=4, mode sep 2.088, RMS‖y‖ ≈ 1): benign parking read
  // meanPair = 1.0582, engineered pileup read 0.0857 — and BOTH fired the
  // win-share tripwire with the same histogram shape. This gate replays those
  // exact numbers through the classifier, so a future edit that drags
  // PILEUP_SPREAD_FRACTION outside the (0.086, 1.058)·rms gap — or inverts the
  // comparison — misclassifies a measured state and fails here.
  {
    const scale = { tag: "seeded", rms: 1.0 } as const;
    const benign = classifyHeads(
      true,
      { tag: "spread", meanPair: 1.0582, minPair: 0.1331, pairs: 6 },
      scale
    );
    ok(
      benign.tag === "separated-unresolved",
      `separated heads with skewed wins → separated-unresolved, because spread ` +
        `alone cannot prove K exceeds support (got ${benign.tag})`
    );
    const pileup = classifyHeads(
      true,
      { tag: "spread", meanPair: 0.0857, minPair: 0.0184, pairs: 6 },
      scale
    );
    ok(
      pileup.tag === "pileup",
      `measured PILEUP state (meanPair 0.086·rms, wins skewed) → pileup (got ${pileup.tag})`
    );
    // Spread alone must NOT trigger the light: near-identical heads with a
    // HEALTHY win mixture is ordinary early training (similar inits), not
    // collapse — a refactor that classifies by spread without the skew gate
    // would paint every fresh run red.
    const fresh = classifyHeads(
      false,
      { tag: "spread", meanPair: 0.01, minPair: 0.001, pairs: 6 },
      scale
    );
    ok(fresh.tag === "ok", `tiny spread with HEALTHY wins stays ok (got ${fresh.tag})`);
    // Skew before any probe/scale exists must admit ignorance, not guess
    // either verdict.
    const early = classifyHeads(true, { tag: "unprobed" }, scale);
    const unscaled = classifyHeads(
      true,
      { tag: "spread", meanPair: 0.0857, minPair: 0.0184, pairs: 6 },
      { tag: "unseeded" }
    );
    ok(
      early.tag === "unresolved" && unscaled.tag === "unresolved",
      `skew without a probe or without a reward scale → unresolved, never a ` +
        `guessed verdict (got ${early.tag}, ${unscaled.tag})`
    );
    const branchFailure = combineHeadHealth({ tag: "ok" }, pileup);
    const branchUnknown = combineHeadHealth({ tag: "ok" }, benign);
    ok(
      branchFailure.tag === "pileup" && branchUnknown.tag === "separated-unresolved",
      "Agree+Disagree combines branch health conservatively; averaging cannot hide one lane"
    );
  }

  // ══════════════════════════════════════════════════════════════════════
  console.log("\n§8 PIXEL CRITIC GATE — no silently-inert critic");
  // ══════════════════════════════════════════════════════════════════════
  {
    const headDims = (inSize: number) => [
      { inSize, outSize: 16, activation: "selu" as const },
      { inSize: 16, outSize: 16, activation: "selu" as const },
      { inSize: 16, outSize: 2, activation: "tanh" as const },
    ];
    const mkLayout = (encoding: Encoding, classes = 0) => {
      const inSize =
        encoding.kind === "fourier"
          ? 2 + 4 * encoding.octaves
          : encoding.kind === "hashgrid"
          ? encoding.features
          : 2;
      // C > 0: head 0 stays class-blind, head 1 takes the one-hot (FieldLayout).
      return layoutField(
        "helmholtz",
        [headDims(inSize), headDims(inSize + classes)],
        { encoding, classes }
      );
    };
    const fourier = mkLayout({ kind: "fourier", octaves: 4 });
    const spec = { weight: 0.04, kind: "vec-field" as const };
    const baseGates = {
      declared: spec,
      hasField: true,
      fieldLossDeclared: true,
      wantTfjsTrainer: false,
      adversaryOnTfjs: false,
      agreeDisagreeGame: false,
      layout: fourier,
    };

    // The whole point: a supported piece still gets its critic. If this ever
    // flips to "dropped", the four shipped Pixel pieces are running a
    // ZERO_FIELD_LOSS field with NOTHING training it.
    const fused = resolvePixelCritic(baseGates);
    ok(fused.tag === "fused", `supported Pixel piece resolves fused (got ${fused.tag})`);
    ok(
      fused.tag === "fused" && fused.spec === spec,
      "the fused plan CARRIES the approved spec, so the construction site " +
        "cannot re-derive a different one"
    );

    // §8a THE LOUD PATH. Agree+Disagree + pixel critic is not a weaker
    // configuration, it is a non-configuration: three extGrad claimants for two
    // bindings. It must THROW during piece resolution — before any trainer is
    // built and before "[pixel-disc] FUSED" is ever printed.
    let threwCombo = false;
    let comboMsg = "";
    try {
      resolvePixelCritic({ ...baseGates, agreeDisagreeGame: true });
    } catch (e) {
      threwCombo = true;
      comboMsg = String((e as Error).message);
    }
    ok(
      threwCombo,
      "Agree+Disagree game + pixel critic THROWS at resolution (never a " +
        "constructed-but-unbound critic)"
    );
    ok(
      /Agree\+Disagree/.test(comboMsg) && /0%/.test(comboMsg),
      `the throw names the game and the 0%-of-the-update consequence (got: ${comboMsg.slice(0, 60)}…)`
    );
    // It throws on the DECLARATION, not on the runtime path: a piece carrying
    // both is broken even when some other gate would have dropped the critic
    // anyway, and the author must see it.
    let threwOnTfjsToo = false;
    try {
      resolvePixelCritic({
        ...baseGates,
        agreeDisagreeGame: true,
        wantTfjsTrainer: true,
      });
    } catch (_) {
      threwOnTfjsToo = true;
    }
    ok(threwOnTfjsToo, "the combination throws regardless of which trainer would run");

    // §8b EVERY OTHER DROP IS NAMED. Silence is the bug; a reason string is
    // the fix. `?train=tfjs` in particular used to disable the critic with no
    // log at all.
    const dropCases: { name: string; plan: ReturnType<typeof resolvePixelCritic> }[] = [
      { name: "?train=tfjs", plan: resolvePixelCritic({ ...baseGates, wantTfjsTrainer: true }) },
      { name: "no fieldLoss", plan: resolvePixelCritic({ ...baseGates, fieldLossDeclared: false }) },
      { name: "no field", plan: resolvePixelCritic({ ...baseGates, hasField: false }) },
      { name: "adversary on tfjs", plan: resolvePixelCritic({ ...baseGates, adversaryOnTfjs: true }) },
      {
        name: "hashgrid arch (dock override)",
        plan: resolvePixelCritic({
          ...baseGates,
          layout: mkLayout({ kind: "hashgrid", gridSize: 16, features: 4 }),
        }),
      },
      {
        name: "class-aware field",
        plan: resolvePixelCritic({ ...baseGates, layout: mkLayout({ kind: "raw" }, 3) }),
      },
    ];
    const unnamed = dropCases.filter(
      (c) => c.plan.tag !== "dropped" || c.plan.reason.trim().length === 0
    );
    ok(
      unnamed.length === 0,
      `every gate that silences a declared critic returns a REASON ` +
        `(${dropCases.length} gates checked${
          unnamed.length ? `; unnamed: ${unnamed.map((u) => u.name).join(", ")}` : ""
        })`
    );
    const tfjsPlan = dropCases[0].plan;
    ok(
      tfjsPlan.tag === "dropped" && /train=tfjs/.test(tfjsPlan.reason),
      "?train=tfjs names ITSELF as the cause, so the console warn is actionable"
    );
    // A piece with no critic must stay silent — a warn per piece would train
    // readers to ignore the channel that carries the real one.
    ok(
      resolvePixelCritic({ ...baseGates, declared: undefined }).tag === "absent",
      "a piece that declares no critic resolves absent (no warn)"
    );

    // §8c GALLERY COHERENCE. No shipped piece may carry the impossible pair —
    // resolvePixelCritic throwing here IS the failure.
    const q = new URLSearchParams("");
    let brokenPieces = 0;
    let pixelPieces = 0;
    for (const piece of GALLERY) {
      const advSpec = resolveAdversary(piece.adversary, q);
      if (piece.pixelDisc) pixelPieces++;
      try {
        resolvePixelCritic({
          declared: piece.pixelDisc,
          hasField: !!piece.createField || !!piece.fieldArch,
          fieldLossDeclared: piece.fieldLoss !== undefined,
          wantTfjsTrainer: false,
          adversaryOnTfjs: false,
          agreeDisagreeGame: advSpec.tag === "on" && advSpec.game === "agree-disagree",
          layout: fourier,
        });
      } catch (e) {
        console.log(`      broken piece: ${piece.name} — ${(e as Error).message}`);
        brokenPieces++;
      }
    }
    ok(
      brokenPieces === 0,
      `no gallery piece declares both the Agree+Disagree game and a pixel ` +
        `critic (${pixelPieces} pixel pieces checked)`
    );
  }

  console.log(
    `\n${failures === 0 ? "ALL ADVERSARY WIRING CHECKS PASS" : `${failures} FAILURE(S)`}`
  );
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
