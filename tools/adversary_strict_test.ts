/**
 * Focused semantic gates for the strict raw-force adversary and scale quotient.
 *
 *   bun tools/adversary_strict_test.ts
 */
import * as tf from "@tensorflow/tfjs";
import {
  Adversary,
  AdversaryConfigError,
  DIRECTION_ACTIVE_FLOOR,
  defaultAdversaryConfig,
  disposeAdversary,
  disposeTupleSample,
  type TupleEncoding,
  type TupleSample,
} from "../src/core/gan/adversary";

await tf.setBackend("cpu");
await tf.ready();

let failures = 0;
function ok(condition: boolean, message: string): void {
  console.log(`${condition ? "  ok  " : " FAIL "} ${message}`);
  if (!condition) failures++;
}
const maxDelta = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i])));

const pairPos = tf.tensor2d([
  [0.2, 0.3],
  [0.55, 0.42],
]);
const pairForce = tf.tensor2d([
  [0.07, -0.03],
  [0.31, 0.19],
]);
const pairIdx = Int32Array.from([0, 1]);

function encodePair(tag: TupleEncoding["tag"], scale = 1): { u: number[]; y: number[] } {
  const encoding = { tag } as TupleEncoding;
  const adv = new Adversary({
    ...defaultAdversaryConfig({ tag: "single" }, encoding, "periodic"),
    batchTuples: 1,
    seed: 17,
  });
  const signal = pairForce.mul(scale) as tf.Tensor2D;
  const s = adv.encodeSignal(pairPos, signal, pairIdx);
  const out = { u: Array.from(s.u.dataSync()), y: Array.from(s.y.dataSync()) };
  disposeTupleSample(s);
  signal.dispose();
  disposeAdversary(adv);
  return out;
}

console.log("\n§1 strict raw-force signal — hidden transition state is not an input");
{
  const adv = new Adversary({
    ...defaultAdversaryConfig({ tag: "single" }, { tag: "point" }, "periodic"),
    batchTuples: 2,
    seed: 3,
  });
  const strictA = adv.encodeSignal(pairPos, pairForce, Int32Array.from([0, 1]));
  const strictB = adv.encodeSignal(pairPos, pairForce.clone(), Int32Array.from([0, 1]));
  const yA = strictA.y.dataSync();
  const yB = strictB.y.dataSync();
  ok(maxDelta(yA, pairForce.dataSync()) === 0, "point target is exactly raw F(x)");
  ok(maxDelta(yA, yB) === 0, "same (x,F(x)) is deterministic with no hidden velocity input");

  const nextSlow = pairPos.add(tf.tensor2d([[0.001, 0], [0.001, 0]])) as tf.Tensor2D;
  const nextFast = pairPos.add(tf.tensor2d([[0.02, 0], [-0.015, 0]])) as tf.Tensor2D;
  const legacyA = adv.encodeTransition(pairPos, nextSlow, Int32Array.from([0, 1]));
  const legacyB = adv.encodeTransition(pairPos, nextFast, Int32Array.from([0, 1]));
  ok(
    maxDelta(legacyA.y.dataSync(), legacyB.y.dataSync()) > 1e-3,
    "legacy transitions can differ at identical positions, proving why they are not the strict control"
  );
  disposeTupleSample(strictA);
  disposeTupleSample(strictB);
  disposeTupleSample(legacyA);
  disposeTupleSample(legacyB);
  nextSlow.dispose();
  nextFast.dispose();
  disposeAdversary(adv);
}

console.log("\n§2 quotient variants — rotation, scale-raw control, scale-adjusted");
{
  const rot = encodePair("pair-rotation", 1);
  const raw1 = encodePair("pair-rotation-scale-raw", 1);
  const raw7 = encodePair("pair-rotation-scale-raw", 7);
  const adj1 = encodePair("pair-rotation-scale-adjusted", 1);
  const adj7 = encodePair("pair-rotation-scale-adjusted", 7);
  ok(rot.u[0] > 0.1, `rotation-only retains separation context r=${rot.u[0].toFixed(5)}`);
  ok(raw1.u[0] === 1 && adj1.u[0] === 1, "rotation+scale modes expose constant context u=[1]");
  ok(
    maxDelta(raw7.y, raw1.y.map((v) => 7 * v)) < 2e-6,
    "raw scale-quotient is the explicit negative control: target grows 7×"
  );
  ok(
    maxDelta(adj1.y, adj7.y) < 2e-6,
    "adjusted active target value is exactly invariant under positive signal scaling"
  );
  ok(Math.abs(Math.hypot(...adj1.y) - 1) < 2e-6, "adjusted active target is an exact unit direction");
}

console.log("\n§3 adjusted payoff — normalized predictor, inactive zeros, radial gradient");
{
  const adjusted = new Adversary({
    ...defaultAdversaryConfig(
      { tag: "wta", k: 2, relaxEps: 0.05 },
      { tag: "pair-rotation-scale-adjusted" },
      "periodic"
    ),
    batchTuples: 1,
    seed: 29,
  });
  const active = adjusted.encodeSignal(pairPos, pairForce, pairIdx);
  const pred = adjusted.predictHeads(active.u).dataSync();
  const predNorms = [Math.hypot(pred[0], pred[1]), Math.hypot(pred[2], pred[3])];
  ok(
    predNorms.every((n) => n > 0.99 && n <= 1.00001),
    `adjusted public predictions are direction-normalized (${predNorms.map((n) => n.toFixed(6))})`
  );

  const equalForce = tf.tensor2d([
    [0.3, -0.1],
    [0.3, -0.1],
  ]);
  const inactive = adjusted.encodeSignal(pairPos, equalForce, pairIdx);
  const inactiveY = inactive.y.dataSync();
  const inactivePayoff = adjusted.payoff(inactive).dataSync()[0];
  const inactiveGen = adjusted.generatorLoss(inactive).dataSync()[0];
  ok(maxDelta(inactiveY, [0, 0]) === 0, "zero relative force produces an explicit inactive zero target");
  ok(inactivePayoff === 0 && Object.is(inactiveGen, -0), "inactive target contributes exact zero to D and G");
  const inactiveReport = adjusted.trainStep(inactive);
  ok(inactiveReport.loss === 0, "inactive target contributes exact zero discriminator update loss");

  const directionalGrad = (delta: number) => {
    const signal = tf.tensor2d([[0, 0], [delta, 0]]);
    const grad = tf.grad((x: tf.Tensor) =>
      tf.tidy(() => {
        const sample = adjusted.encodeSignal(pairPos, x as tf.Tensor2D, pairIdx);
        return adjusted.generatorLoss(sample);
      })
    )(signal);
    const sample = adjusted.encodeSignal(pairPos, signal, pairIdx);
    const out = {
      payoff: adjusted.payoff(sample).dataSync()[0],
      y: Array.from(sample.y.dataSync()),
      grad: Array.from(grad.dataSync()),
    };
    disposeTupleSample(sample);
    grad.dispose();
    signal.dispose();
    return out;
  };
  const below = directionalGrad(0.5 * DIRECTION_ACTIVE_FLOOR);
  const above = directionalGrad(1.1 * DIRECTION_ACTIVE_FLOOR);
  ok(
    below.payoff === 0 && below.grad.every((v) => v === 0),
    `relative force below ${DIRECTION_ACTIVE_FLOOR} is exact inactive zero`
  );
  const aboveFinite = above.grad.every(Number.isFinite);
  const dqx = above.grad[2] - above.grad[0];
  const dqy = above.grad[3] - above.grad[1];
  // `above.y` is expressed in the pair's canonical frame, whereas `grad` is
  // with respect to the world-frame force samples.  Dotting those two frames
  // (the previous test) manufactured a large false "radial" component.  The
  // perturbation used by directionalGrad is q_world=(delta, 0), so test
  // tangency against that world-frame radial direction.
  const radial = Math.abs(dqx * (1.1 * DIRECTION_ACTIVE_FLOOR) + dqy * 0);
  ok(
    aboveFinite && above.payoff > 0 && radial < 2e-4,
    `just-over-floor target is active, finite and tangent ` +
      `(payoff=${above.payoff.toExponential(3)}, radial=${radial.toExponential(3)})`
  );

  const lossAt = (adv: Adversary, c: tf.Scalar): tf.Scalar =>
    tf.tidy(() => {
      const signal = pairForce.mul(c) as tf.Tensor2D;
      const s = adv.encodeSignal(pairPos, signal, pairIdx);
      return adv.generatorLoss(s);
    });
  const c = tf.scalar(2);
  const adjustedGrad = tf.grad((x: tf.Tensor) => lossAt(adjusted, x.asScalar()))(c).dataSync()[0];
  const adjustedV1 = lossAt(adjusted, tf.scalar(1)).dataSync()[0];
  const adjustedV7 = lossAt(adjusted, tf.scalar(7)).dataSync()[0];
  ok(
    Math.abs(adjustedV1 - adjustedV7) < 2e-6,
    `adjusted payoff value is scale invariant (${adjustedV1.toFixed(7)} vs ${adjustedV7.toFixed(7)})`
  );
  ok(
    Math.abs(adjustedGrad) < 2e-6,
    `adjusted generator radial derivative dL/dc is zero (${adjustedGrad.toExponential(3)})`
  );

  const raw = new Adversary({
    ...defaultAdversaryConfig(
      { tag: "wta", k: 2, relaxEps: 0.05 },
      { tag: "pair-rotation-scale-raw" },
      "periodic"
    ),
    batchTuples: 1,
    seed: 29,
  });
  const rawGrad = tf.grad((x: tf.Tensor) => lossAt(raw, x.asScalar()))(c).dataSync()[0];
  const rawV1 = lossAt(raw, tf.scalar(1)).dataSync()[0];
  const rawV7 = lossAt(raw, tf.scalar(7)).dataSync()[0];
  ok(
    Math.abs(rawV7) > 2 * Math.abs(rawV1),
    `raw negative control buys payoff by scaling (${rawV1.toFixed(5)} → ${rawV7.toFixed(5)})`
  );
  ok(Math.abs(rawGrad) > 1e-3, `raw negative control has radial gradient (${rawGrad.toExponential(3)})`);

  disposeTupleSample(active);
  disposeTupleSample(inactive);
  equalForce.dispose();
  c.dispose();
  disposeAdversary(adjusted);
  disposeAdversary(raw);
}

console.log("\n§4 strict shared relaxed-WTA payoff — one game, opposite signs");
{
  const adv = new Adversary({
    ...defaultAdversaryConfig(
      { tag: "wta", k: 3, relaxEps: 0.1 },
      { tag: "pair-rotation-scale-raw" },
      "periodic"
    ),
    batchTuples: 1,
    seed: 41,
  });
  const s = adv.encodeSignal(pairPos, pairForce, pairIdx);
  const payoff = adv.payoff(s).mean().dataSync()[0];
  const generator = adv.generatorLoss(s).dataSync()[0];
  const report = adv.trainStep(s);
  ok(Math.abs(generator + payoff) < 2e-6, "generator loss is exactly the negative shared payoff");
  ok(
    Math.abs(report.loss - payoff) < 2e-5,
    `discriminator report is the same pre-update shared payoff (${report.loss.toFixed(7)} vs ${payoff.toFixed(7)})`
  );
  disposeTupleSample(s);
  disposeAdversary(adv);
}

console.log("\n§5 relaxed-WTA epsilon — winner must remain individually dominant");
{
  const make = (eps: number) =>
    new Adversary(
      defaultAdversaryConfig(
        { tag: "wta", k: 4, relaxEps: eps },
        { tag: "pair" },
        "periodic"
      )
    );
  const good = make(0.749);
  disposeAdversary(good);
  ok(true, "epsilon just below (K-1)/K is accepted");
  for (const eps of [0.75, 0.9]) {
    let rejected = false;
    try {
      make(eps);
    } catch (err) {
      rejected = err instanceof AdversaryConfigError;
    }
    ok(rejected, `epsilon ${eps} is rejected at/above (K-1)/K`);
  }
}

console.log("\n§6 labelled four-point quotient — honest labels and inactive anchor");
{
  const adv = new Adversary({
    ...defaultAdversaryConfig(
      { tag: "single" },
      { tag: "quad-labelled" },
      "periodic"
    ),
    batchTuples: 1,
    seed: 53,
  });
  const xs = [
    [0.32, 0.41],
    [0.47, 0.45],
    [0.37, 0.61],
    [0.22, 0.55],
  ];
  const fs = [
    [0.05, -0.03],
    [0.14, 0.02],
    [-0.08, 0.11],
    [0.01, -0.07],
  ];
  const encode = (p: number[][], f: number[][], idx = [0, 1, 2, 3]) => {
    const pt = tf.tensor2d(p);
    const ft = tf.tensor2d(f);
    const s = adv.encodeSignal(pt, ft, Int32Array.from(idx));
    const out = {
      u: Array.from(s.u.dataSync()),
      y: Array.from(s.y.dataSync()),
      payoff: adv.payoff(s).dataSync()[0],
    };
    disposeTupleSample(s);
    pt.dispose();
    ft.dispose();
    return out;
  };
  const base = encode(xs, fs);
  ok(base.u.length === 6 && base.y.length === 8, "quad-labelled has u∈R⁶ and y∈R⁸");

  const th = 0.63;
  const rotate = (v: number[]) => [
    v[0] * Math.cos(th) - v[1] * Math.sin(th),
    v[0] * Math.sin(th) + v[1] * Math.cos(th),
  ];
  const rigidP = xs.map((p) => {
    const q = rotate([p[0] - 0.4, p[1] - 0.5]);
    return [q[0] + 0.43, q[1] + 0.48];
  });
  const rigid = encode(rigidP, fs.map(rotate));
  ok(
    maxDelta(base.u, rigid.u) < 2e-6 && maxDelta(base.y, rigid.y) < 2e-6,
    "quad-labelled quotient is invariant under joint translation+rotation"
  );

  const relabelled = encode(xs, fs, [0, 1, 3, 2]);
  ok(
    maxDelta(base.u, relabelled.u) > 1e-3 && maxDelta(base.y, relabelled.y) > 1e-3,
    "quad labels are explicit: swapping members 2/3 changes the encoded sample"
  );

  const coincident = xs.map((p) => p.slice());
  coincident[1] = coincident[0].slice();
  const inactive = encode(coincident, fs);
  ok(
    inactive.y.every((v) => v === 0) && inactive.payoff === 0,
    "near-coincident anchor frame is marked inactive with exact zero target/payoff"
  );
  disposeAdversary(adv);
}

console.log("\n§7 triangle tie policy — ambiguous canonical orders are inactive");
{
  const adv = new Adversary({
    ...defaultAdversaryConfig({ tag: "single" }, { tag: "tri" }, "periodic"),
    batchTuples: 1,
    seed: 61,
  });
  const pos = tf.tensor2d([
    [0.4, 0.4],
    [0.6, 0.4],
    [0.5, 0.55],
  ]); // exact isosceles: sides opposite labels 0 and 1 are equal
  const force = tf.tensor2d([
    [0.13, -0.07],
    [-0.02, 0.16],
    [0.09, 0.04],
  ]);
  const perms = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
  ];
  let referenceU: number[] | undefined;
  let worstU = 0;
  let worstAbsY = 0;
  let worstPayoff = 0;
  for (const p of perms) {
    const s = adv.encodeSignal(pos, force, Int32Array.from(p));
    const u = Array.from(s.u.dataSync());
    const y = Array.from(s.y.dataSync());
    if (referenceU === undefined) referenceU = u;
    else worstU = Math.max(worstU, maxDelta(referenceU, u));
    worstAbsY = Math.max(worstAbsY, ...y.map(Math.abs));
    worstPayoff = Math.max(worstPayoff, Math.abs(adv.payoff(s).dataSync()[0]));
    disposeTupleSample(s);
  }
  ok(worstU < 2e-6, "isosceles context remains permutation-invariant sorted side lengths");
  ok(
    worstAbsY === 0 && worstPayoff === 0,
    "all six isosceles label permutations are dropped with exact zero target/payoff"
  );
  pos.dispose();
  force.dispose();
  disposeAdversary(adv);
}

console.log("\n§8 inactive tuple win accounting — no phantom head-0 wins");
{
  function checkMixedActivity(
    label: string,
    encoding: TupleEncoding,
    positions: number[][],
    forces: number[][],
    activeIdx: number[],
    inactiveIdx: number[]
  ) {
    const adv = new Adversary({
      ...defaultAdversaryConfig(
        { tag: "wta", k: 3, relaxEps: 0.05 },
        encoding,
        "periodic"
      ),
      batchTuples: 2,
      seed: label === "tri" ? 71 : 73,
    });
    const pos = tf.tensor2d(positions);
    const force = tf.tensor2d(forces);
    const mixedIdx = Int32Array.from([...activeIdx, ...inactiveIdx]);

    const mixed = adv.encodeSignal(pos, force, mixedIdx);
    const mixedPayoff = Array.from(adv.payoff(mixed).dataSync());
    const mixedGrad = tf.grad((x: tf.Tensor) =>
      tf.tidy(() => {
        const sample = adv.encodeSignal(pos, x as tf.Tensor2D, mixedIdx);
        return adv.generatorLoss(sample);
      })
    )(force);
    const g = Array.from(mixedGrad.dataSync());
    const activeMax = Math.max(
      ...activeIdx.flatMap((i) => [Math.abs(g[2 * i]), Math.abs(g[2 * i + 1])])
    );
    const inactiveMax = Math.max(
      ...inactiveIdx.flatMap((i) => [Math.abs(g[2 * i]), Math.abs(g[2 * i + 1])])
    );
    ok(
      mixedPayoff[0] > 0 && mixedPayoff[1] === 0,
      `${label} mixed batch keeps active payoff and exact-zero inactive payoff`
    );
    ok(
      activeMax > 0 && inactiveMax === 0,
      `${label} mixed batch keeps active generator gradient and exact-zero inactive gradient`
    );
    const mixedReport = adv.trainStep(mixed);
    ok(
      mixedReport.winCounts.reduce((a, b) => a + b, 0) === 1 &&
        adv.winStats().total === 1,
      `${label} mixed batch records exactly one active winner, not B=2`
    );
    disposeTupleSample(mixed);
    mixedGrad.dispose();

    adv.resetWinCounts();
    const inactive = adv.encodeSignal(pos, force, Int32Array.from(inactiveIdx));
    const inactivePayoff = adv.payoff(inactive).dataSync()[0];
    const inactiveGrad = tf.grad((x: tf.Tensor) =>
      tf.tidy(() => {
        const sample = adv.encodeSignal(pos, x as tf.Tensor2D, Int32Array.from(inactiveIdx));
        return adv.generatorLoss(sample);
      })
    )(force);
    const inactiveReport = adv.trainStep(inactive);
    ok(
      inactivePayoff === 0 &&
        Array.from(inactiveGrad.dataSync()).every((v) => v === 0),
      `${label} all-inactive batch preserves exact-zero payoff and gradient`
    );
    ok(
      inactiveReport.winCounts.every((n) => n === 0) &&
        adv.winStats().total === 0 &&
        !adv.collapsed(),
      `${label} all-inactive batch records no head-0 sentinel and no false collapse`
    );

    disposeTupleSample(inactive);
    inactiveGrad.dispose();
    pos.dispose();
    force.dispose();
    disposeAdversary(adv);
  }

  checkMixedActivity(
    "tri",
    { tag: "tri" },
    [
      [0.10, 0.10], [0.33, 0.12], [0.18, 0.39], // scalene, active
      [0.60, 0.60], [0.80, 0.60], [0.70, 0.75], // isosceles, inactive
    ],
    [
      [0.10, -0.03], [-0.04, 0.16], [0.08, 0.02],
      [0.05, 0.07], [-0.11, 0.03], [0.02, -0.09],
    ],
    [0, 1, 2],
    [3, 4, 5]
  );

  checkMixedActivity(
    "quad",
    { tag: "quad-labelled" },
    [
      [0.15, 0.15], [0.28, 0.17], [0.20, 0.34], [0.08, 0.28], // active
      [0.60, 0.60], [0.60, 0.60], [0.75, 0.67], [0.55, 0.78], // inactive anchor
    ],
    [
      [0.08, -0.02], [-0.06, 0.14], [0.11, 0.04], [-0.03, -0.09],
      [0.02, 0.06], [-0.08, 0.01], [0.13, -0.04], [-0.01, 0.10],
    ],
    [0, 1, 2, 3],
    [4, 5, 6, 7]
  );
}

pairPos.dispose();
pairForce.dispose();

console.log(`\n${failures === 0 ? "ALL STRICT ADVERSARY CHECKS PASS" : `${failures} CHECK(S) FAILED`}`);
process.exit(failures === 0 ? 0 : 1);
