/**
 * Adversary core verification — src/core/gan/adversary.ts.
 *
 *   bun tools/adversary_test.ts
 *
 * No framework, no mocks: real tfjs on the CPU backend, real training loops,
 * real numbers printed on PASS as well as FAIL (repo convention, see
 * tools/ad_test.ts).
 *
 * Runtime ≈ 3 min on an M-series CPU; §6 is most of it.
 *
 * The checks and the specific bug each one catches:
 *
 *  1. THE THESIS (one-to-many fixture). Same context u, two equally likely
 *     targets y₁/y₂. Variant A (`single`) must land in the empty space between
 *     the modes with its residual pinned at half the mode separation. Variant B
 *     (`wta`, K≥2) must place distinct heads near BOTH modes and keep a strictly
 *     lower mean min-distance. Catches: a WTA implementation that silently
 *     averages (e.g. weights applied to the wrong axis, or the argMin taken over
 *     the batch axis instead of the head axis) — the loss would still go down and
 *     nothing else here would notice.
 *
 *     NOTE ON WHAT "COLLAPSE" MEANS HERE. The loss is the NORM ‖·‖, not the
 *     squared norm, so the population minimizer for one head is the GEOMETRIC
 *     MEDIAN, and for two equally likely modes every point on the joining
 *     segment attains the same objective. The objective is therefore FLAT along
 *     that segment and the head parks at an arbitrary point on it — asserting
 *     "converges to the mean" would be asserting a fact about SGD drift, which
 *     is not falsifiable. So the test asserts the two things that ARE pinned by
 *     the math: the head lies ON the segment, and E[‖y − g‖] = ‖y₁ − y₂‖/2.
 *
 *  2. RELAXED-WTA RESCUES A DEAD HEAD. With relaxEps = 0 a head that never wins
 *     receives EXACTLY zero gradient and its weights are frozen forever (silent
 *     K → 1). Asserts that ε = 0 really does freeze it and that ε = 0.05 really
 *     does not. Catches: someone "simplifying" the loser share away, or applying
 *     ε as a floor on the winner weight instead of on the losers.
 *
 *  3. SE(2) INVARIANCE OF THE PAIR ENCODING. Apply a rigid motion (rotation +
 *     translation) to a pair AND its displacements; u and y must be unchanged.
 *     Also asserts point-swap invariance. Catches: a wrong perpendicular sign
 *     convention, or a frame built from absolute rather than relative vectors —
 *     both of which produce plausible-looking numbers and nothing else would fail.
 *
 *  4. TORUS MINIMUM-IMAGE. A particle that wraps across the periodic seam must
 *     produce a small displacement, not one of size ≈1. Catches the phantom-
 *     displacement bug described at `minImage()` in adversary.ts; without the
 *     guard the generator could farm reward simply by pushing particles over the
 *     seam.
 *
 *  5. GENERATOR GRADIENT SIGN. Gradient steps on the POSITIONS (heads frozen)
 *     must INCREASE the mean surprise. Catches a sign flip in `generatorLoss`,
 *     which would make the adversary an ally instead — the loss curve would look
 *     perfectly healthy.
 *
 *  6. ENCODING DEGENERACY, as a convergence ladder on a real deterministic
 *     field. `point` (absolute in, absolute out) must keep shrinking — its error
 *     is REDUCIBLE, which is exactly why the deleted `PredictorEnsemble` died.
 *     `pair` (SE(2)-canonicalized) must plateau — its error is IRREDUCIBLE.
 *     Catches anyone reintroducing absolute position into the pair context.
 *
 *  7. TENSOR LEAKS across a full construct → train → dispose lifecycle. A leak
 *     is invisible frame-to-frame and fatal after a few minutes of running.
 *
 *  8. LEGACY EMA SCALE TELEMETRY — values are roughly normalized, without the
 *     false claim that a tape-constant divisor removes radial generator gradient.
 *
 *  9. minImage GRADIENT CONTRACT — d(a − round(a))/da = 1 is a backend contract.
 *
 * 10. HEAD SPREAD DISAMBIGUATOR. A skewed win histogram has two readings —
 *     benign K > modes vs optimizer pileup — and `headSpread` must separate
 *     them: healthy spread (~mode separation) on a bimodal fixture with k = 4,
 *     ~0 spread on a fixture engineered to pile all heads onto one mode.
 *     Catches: a spread metric computed over the wrong axis (heads vs batch),
 *     which would read "healthy" forever and make the HUD light a liar.
 *
 * 11. TRI ENCODING INVARIANCES, mirroring §3: rotation+translation, all six
 *     vertex permutations, torus translation across the seam, and reflection
 *     EQUIVARIANCE (u and the ê₁ components invariant, every ê₂ component
 *     flips sign). Catches: a base-vertex leak in the chart (permutation
 *     would fail), a broken canonical ordering, or a handedness slip.
 */

import * as tf from "@tensorflow/tfjs";
import {
  Adversary,
  defaultAdversaryConfig,
  disposeAdversary,
  disposeTupleSample,
  scatterSurprise,
  type AdversaryConfig,
  type TupleSample,
} from "../src/core/gan/adversary";

await tf.setBackend("cpu");
await tf.ready();

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

const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/* ══════════════════════════════════════════════════════════════════════════
   §1  THE THESIS — one-to-many fixture
   ══════════════════════════════════════════════════════════════════════════
   u is CONSTANT (r = 0.3 for every tuple). The target is drawn from a two-point
   conditional: y ∈ {MODE_A, MODE_B}, each with probability 1/2. This is exactly
   the "same input, two equally likely targets" case. Nothing about the field is
   involved — this isolates the loss math from the encoding.                */

const U_CONST = 0.3;
const MODE_A = [1.0, 0.0];
const MODE_B = [-1.0, 0.6];
const MODE_MEAN = [(MODE_A[0] + MODE_B[0]) / 2, (MODE_A[1] + MODE_B[1]) / 2];
const MODE_SEP = dist(MODE_A, MODE_B);

/** Synthetic TupleSample: constant u, bimodal y. `idx` is unused by the loss. */
function bimodalBatch(b: number, rnd: () => number): TupleSample {
  const u = new Float32Array(b);
  const y = new Float32Array(b * 2);
  for (let i = 0; i < b; i++) {
    u[i] = U_CONST;
    const m = rnd() < 0.5 ? MODE_A : MODE_B;
    y[i * 2] = m[0];
    y[i * 2 + 1] = m[1];
  }
  return {
    u: tf.tensor2d(u, [b, 1]),
    y: tf.tensor2d(y, [b, 2]),
    idx: new Int32Array(b * 2),
    b,
    m: 2,
  };
}

interface ThesisResult {
  guesses: number[][];
  /** E[min_k ‖y − g_k‖] evaluated on a fresh held-out batch. */
  meanMinDist: number;
  winFractions: readonly number[];
}

function runThesis(cfg: AdversaryConfig, steps: number, seed: number): ThesisResult {
  const adv = new Adversary(cfg);
  const rnd = mulberry32(seed);
  for (let s = 0; s < steps; s++) {
    const batch = bimodalBatch(256, rnd);
    adv.trainStep(batch);
    disposeTupleSample(batch);
  }
  const evalBatch = bimodalBatch(4096, mulberry32(seed + 777));
  // Quantization quality is the PURE nearest-head residual. `surprise()` is the
  // strict shared relaxed-WTA game payoff and intentionally includes loser
  // terms when ε>0, so it is not a min-distance diagnostic.
  const meanMinDist = tf.tidy(() => adv.winnerResidual(evalBatch).mean().dataSync()[0]);
  const guesses = tf.tidy(() => {
    const p = adv.predictHeads(tf.tensor2d([[U_CONST]])); // [k,1,2]
    const flat = Array.from(p.dataSync());
    const out: number[][] = [];
    for (let j = 0; j < adv.k; j++) out.push([flat[j * 2], flat[j * 2 + 1]]);
    return out;
  });
  const winFractions = adv.winStats().fractions;
  disposeTupleSample(evalBatch);
  disposeAdversary(adv);
  return { guesses, meanMinDist, winFractions };
}

console.log("\n§1  ONE-TO-MANY FIXTURE — variant A (single) vs variant B (wta)");
console.log(
  `      modes  A=(${MODE_A}) B=(${MODE_B})  mean=(${MODE_MEAN.map((v) => v.toFixed(3))})  |A−B|=${MODE_SEP.toFixed(4)}`
);

const BASE = {
  ...defaultAdversaryConfig({ tag: "single" }, { tag: "pair" }, "periodic"),
  batchTuples: 256,
  learningRate: 5e-3,
  seed: 11,
};
const STEPS = 900;

/** Distance from p to the segment [a,b], and the clamped parameter t along it. */
function toSegment(p: number[], a: number[], b: number[]): { d: number; t: number } {
  const ab = [b[0] - a[0], b[1] - a[1]];
  const len2 = ab[0] * ab[0] + ab[1] * ab[1];
  const raw = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2;
  const t = Math.max(0, Math.min(1, raw));
  return { d: dist(p, [a[0] + t * ab[0], a[1] + t * ab[1]]), t };
}

const rA = runThesis({ ...BASE, kind: { tag: "single" } }, STEPS, 3);
const dA_mean = dist(rA.guesses[0], MODE_MEAN);
const dA_nearest = Math.min(dist(rA.guesses[0], MODE_A), dist(rA.guesses[0], MODE_B));
const segA = toSegment(rA.guesses[0], MODE_A, MODE_B);
console.log(
  `      A(single):  g=(${rA.guesses[0].map((v) => v.toFixed(4))})  ` +
    `|g−mean|=${dA_mean.toFixed(4)}  |g−nearest mode|=${dA_nearest.toFixed(4)}  ` +
    `dist-to-segment=${segA.d.toFixed(5)} at t=${segA.t.toFixed(3)}  ` +
    `E[min dist]=${rA.meanMinDist.toFixed(4)}`
);

const rB = runThesis({ ...BASE, kind: { tag: "wta", k: 2, relaxEps: 0.05 } }, STEPS, 3);
const cover = (mode: number[]) => Math.min(...rB.guesses.map((g) => dist(g, mode)));
const coverA = cover(MODE_A);
const coverB = cover(MODE_B);
const headSpread = dist(rB.guesses[0], rB.guesses[1]);
console.log(
  `      B(wta k=2): g0=(${rB.guesses[0].map((v) => v.toFixed(4))})  g1=(${rB.guesses[1].map((v) => v.toFixed(4))})`
);
console.log(
  `                  |head−modeA|=${coverA.toFixed(4)}  |head−modeB|=${coverB.toFixed(4)}  ` +
    `head spread=${headSpread.toFixed(4)}  E[min dist]=${rB.meanMinDist.toFixed(4)}  ` +
    `wins=[${rB.winFractions.map((f) => f.toFixed(3))}]`
);

// A lands on the geometric-median SET (the whole segment), which is the
// falsifiable form of "it averaged the modes instead of choosing one".
ok(
  segA.d < 0.02 * MODE_SEP,
  `A lands on the geometric-median segment (dist ${segA.d.toFixed(5)} < ${(0.02 * MODE_SEP).toFixed(5)}, at t=${segA.t.toFixed(3)})`
);
ok(
  dA_nearest > 0.2 * MODE_SEP,
  `A sits in the empty space, not on a mode (|g−nearest| ${dA_nearest.toFixed(4)} > ${(0.2 * MODE_SEP).toFixed(4)})`
);
// A's residual is pinned at half the mode separation — that IS the collapse,
// and unlike its position this number is determined by the math, not by drift.
ok(
  Math.abs(rA.meanMinDist - MODE_SEP / 2) < 0.02 * MODE_SEP,
  `A's residual is pinned at half the mode separation, i.e. it learned nothing about WHICH mode (${rA.meanMinDist.toFixed(4)} vs ${(MODE_SEP / 2).toFixed(4)})`
);
// B quantizes: each mode has a head near it, and the heads are far apart.
ok(coverA < 0.05 * MODE_SEP, `B covers mode A (${coverA.toFixed(4)} < ${(0.05 * MODE_SEP).toFixed(4)})`);
ok(coverB < 0.05 * MODE_SEP, `B covers mode B (${coverB.toFixed(4)} < ${(0.05 * MODE_SEP).toFixed(4)})`);
ok(headSpread > 0.9 * MODE_SEP, `B's heads are distinct (spread ${headSpread.toFixed(4)} > ${(0.9 * MODE_SEP).toFixed(4)})`);
ok(
  rB.meanMinDist < 0.05 * rA.meanMinDist,
  `B retains strictly lower min-distance than A (${rB.meanMinDist.toFixed(4)} < ${(0.05 * rA.meanMinDist).toFixed(4)} = 0.05·${rA.meanMinDist.toFixed(4)})`
);
// Both heads must actually be USED — a 2-head mixture on a 50/50 conditional
// should win about half each. This is the head-collapse tripwire on the fixture
// that matters, as opposed to §2's deliberately-degenerate unimodal one.
ok(
  Math.min(...rB.winFractions) > 0.3,
  `B keeps both heads alive (win fractions [${rB.winFractions.map((f) => f.toFixed(3))}], min > 0.3)`
);

/* ══════════════════════════════════════════════════════════════════════════
   §2  RELAXED-WTA RESCUES A DEAD HEAD
   ══════════════════════════════════════════════════════════════════════════
   Force the pathology instead of waiting for it: k = 3 heads on a UNIMODAL
   target. Only one head can ever be the winner, so with ε = 0 the other two get
   exactly zero gradient and never move. With ε > 0 they must move.            */

console.log("\n§2  HEAD COLLAPSE — plain WTA freezes losers, relaxed WTA does not");

function loserDrift(relaxEps: number): { drift: number; minWinFrac: number } {
  const cfg: AdversaryConfig = {
    ...BASE,
    kind: { tag: "wta", k: 3, relaxEps },
    seed: 5,
  };
  const adv = new Adversary(cfg);
  const rnd = mulberry32(99);
  // Unimodal target so exactly one head can win.
  const unimodal = (b: number): TupleSample => {
    const u = new Float32Array(b).fill(U_CONST);
    const y = new Float32Array(b * 2);
    for (let i = 0; i < b; i++) {
      y[i * 2] = 1.0 + (rnd() - 0.5) * 0.01;
      y[i * 2 + 1] = 0.0 + (rnd() - 0.5) * 0.01;
    }
    return { u: tf.tensor2d(u, [b, 1]), y: tf.tensor2d(y, [b, 2]), idx: new Int32Array(b * 3), b, m: 3 };
  };

  const probe = tf.tensor2d([[U_CONST]]);
  const before = Array.from(adv.predictHeads(probe).dataSync());
  for (let s = 0; s < 400; s++) {
    const batch = unimodal(128);
    adv.trainStep(batch);
    disposeTupleSample(batch);
  }
  const after = Array.from(adv.predictHeads(probe).dataSync());
  probe.dispose();

  const fr = adv.winStats().fractions;
  // Which head never won? Measure ITS movement specifically.
  let loser = 0;
  for (let j = 1; j < 3; j++) if (fr[j] < fr[loser]) loser = j;
  const drift = Math.hypot(after[loser * 2] - before[loser * 2], after[loser * 2 + 1] - before[loser * 2 + 1]);
  const minWinFrac = Math.min(...fr);
  disposeAdversary(adv);
  return { drift, minWinFrac };
}

const plain = loserDrift(0.0);
const relaxed = loserDrift(0.05);
console.log(
  `      ε=0.00 : least-winning head moved ${plain.drift.toExponential(3)}  (min win fraction ${plain.minWinFrac.toFixed(4)})`
);
console.log(
  `      ε=0.05 : least-winning head moved ${relaxed.drift.toExponential(3)}  (min win fraction ${relaxed.minWinFrac.toFixed(4)})`
);
ok(plain.minWinFrac === 0, `plain WTA really does starve a head (min win fraction = ${plain.minWinFrac})`);
ok(plain.drift < 1e-6, `plain WTA freezes the starved head EXACTLY (drift ${plain.drift.toExponential(3)} < 1e-6)`);
ok(relaxed.drift > 1e-3, `relaxed WTA keeps it learning (drift ${relaxed.drift.toExponential(3)} > 1e-3)`);

/* ══════════════════════════════════════════════════════════════════════════
   §3  SE(2) INVARIANCE OF THE PAIR ENCODING
   ══════════════════════════════════════════════════════════════════════════ */

console.log("\n§3  PAIR ENCODING — invariance of (u, y) under the claimed group");

const PAIR_ADV = new Adversary({
  ...defaultAdversaryConfig({ tag: "single" }, { tag: "pair" }, "periodic"),
  batchTuples: 1,
  seed: 4,
});

/** Encode one explicit pair through the real encoder (deterministic index order). */
function encodeOne(x0: number[], x1: number[], n0: number[], n1: number[]): { u: number; y: number[] } {
  const pos = tf.tensor2d([x0, x1]);
  const nxt = tf.tensor2d([n0, n1]);
  const s = PAIR_ADV.encode(pos, nxt, Int32Array.from([0, 1]));
  const uv = s.u.dataSync()[0];
  const yv = Array.from(s.y.dataSync());
  disposeTupleSample(s);
  pos.dispose();
  nxt.dispose();
  return { u: uv, y: yv };
}

const rot = (v: number[], th: number) => [
  v[0] * Math.cos(th) - v[1] * Math.sin(th),
  v[0] * Math.sin(th) + v[1] * Math.cos(th),
];

// Base configuration, kept well away from the torus seam so wrapping is not in
// play here (§4 tests wrapping separately).
const bx0 = [0.40, 0.50];
const bx1 = [0.55, 0.44];
const bd0 = [0.013, -0.007];
const bd1 = [-0.005, 0.011];
const base = encodeOne(bx0, bx1, [bx0[0] + bd0[0], bx0[1] + bd0[1]], [bx1[0] + bd1[0], bx1[1] + bd1[1]]);

// Rigid motion: rotate about (0.5,0.5) by 0.7 rad, then translate by (0.03,-0.02).
const TH = 0.7;
const C = [0.5, 0.5];
const T = [0.03, -0.02];
const move = (p: number[]) => {
  const r = rot([p[0] - C[0], p[1] - C[1]], TH);
  return [r[0] + C[0] + T[0], r[1] + C[1] + T[1]];
};
const mx0 = move(bx0);
const mx1 = move(bx1);
const md0 = rot(bd0, TH); // displacements rotate, do not translate
const md1 = rot(bd1, TH);
const moved = encodeOne(mx0, mx1, [mx0[0] + md0[0], mx0[1] + md0[1]], [mx1[0] + md1[0], mx1[1] + md1[1]]);

const dU = Math.abs(moved.u - base.u);
const dY = dist(moved.y, base.y);
console.log(
  `      base   u=${base.u.toFixed(7)}  y=(${base.y.map((v) => v.toFixed(7))})`
);
console.log(`      rigid  u=${moved.u.toFixed(7)}  y=(${moved.y.map((v) => v.toFixed(7))})   Δu=${dU.toExponential(2)}  Δy=${dY.toExponential(2)}`);
ok(dU < 1e-5, `u invariant under rotation+translation (Δu ${dU.toExponential(2)} < 1e-5)`);
ok(dY < 1e-5, `y invariant under rotation+translation (Δy ${dY.toExponential(2)} < 1e-5)`);

// Point swap.
const swapped = encodeOne(bx1, bx0, [bx1[0] + bd1[0], bx1[1] + bd1[1]], [bx0[0] + bd0[0], bx0[1] + bd0[1]]);
const sU = Math.abs(swapped.u - base.u);
const sY = dist(swapped.y, base.y);
console.log(`      swap   u=${swapped.u.toFixed(7)}  y=(${swapped.y.map((v) => v.toFixed(7))})   Δu=${sU.toExponential(2)}  Δy=${sY.toExponential(2)}`);
ok(sU < 1e-6 && sY < 1e-6, `encoding invariant under point swap (Δu ${sU.toExponential(2)}, Δy ${sY.toExponential(2)})`);

// Reflection is NOT claimed as an invariance — assert the documented
// EQUIVARIANCE instead, so the docstring cannot drift from the code.
const mirror = (p: number[]) => [p[0], 1 - p[1]];
const fx0 = mirror(bx0);
const fx1 = mirror(bx1);
const fd0 = [bd0[0], -bd0[1]];
const fd1 = [bd1[0], -bd1[1]];
const refl = encodeOne(fx0, fx1, [fx0[0] + fd0[0], fx0[1] + fd0[1]], [fx1[0] + fd1[0], fx1[1] + fd1[1]]);
const rAlong = Math.abs(refl.y[0] - base.y[0]);
const rPerp = Math.abs(refl.y[1] + base.y[1]);
console.log(`      mirror u=${refl.u.toFixed(7)}  y=(${refl.y.map((v) => v.toFixed(7))})   |Δalong|=${rAlong.toExponential(2)}  |perp+perp|=${rPerp.toExponential(2)}`);
ok(
  rAlong < 1e-6 && rPerp < 1e-6 && Math.abs(refl.y[1]) > 1e-8,
  `reflection flips ONLY the perpendicular component, as documented (along Δ ${rAlong.toExponential(2)}, perp sum ${rPerp.toExponential(2)})`
);

PAIR_ADV.dispose();

/* ══════════════════════════════════════════════════════════════════════════
   §4  TORUS MINIMUM-IMAGE
   ══════════════════════════════════════════════════════════════════════════ */

console.log("\n§4  TORUS — a particle crossing the seam must not produce a phantom displacement");

const ptAdv = new Adversary({
  ...defaultAdversaryConfig({ tag: "single" }, { tag: "point" }, "periodic"),
  batchTuples: 2,
  seed: 1,
});
// Particle 0 sits at x = 0.995 and moves +0.01, wrapping to 0.005.
// Particle 1 sits mid-domain and moves the same amount without wrapping.
const wrapPos = tf.tensor2d([
  [0.995, 0.5],
  [0.495, 0.5],
]);
const wrapNext = tf.tensor2d([
  [0.005, 0.5],
  [0.505, 0.5],
]);
const wrapSample = ptAdv.sample(wrapPos, wrapNext);
const wy = Array.from(wrapSample.y.dataSync());
const wIdx = Array.from(wrapSample.idx);
const rowOf = (particle: number) => wIdx.indexOf(particle);
const dWrap = Math.abs(wy[rowOf(0) * 2]);
const dNorm = Math.abs(wy[rowOf(1) * 2]);
console.log(`      wrapped particle Δx = ${dWrap.toFixed(6)}   non-wrapped Δx = ${dNorm.toFixed(6)}`);
ok(
  Math.abs(dWrap - 0.01) < 1e-5,
  `wrapped displacement is the short way round (${dWrap.toFixed(6)} ≈ 0.01, NOT ≈0.99)`
);
ok(Math.abs(dWrap - dNorm) < 1e-5, `wrapped and non-wrapped particles agree (${dWrap.toFixed(6)} vs ${dNorm.toFixed(6)})`);

// Surprise scatter: one tuple per particle here, so hits must be exactly 1 each.
const sc = scatterSurprise(wrapSample, Float32Array.from(tf.tidy(() => ptAdv.surprise(wrapSample).dataSync())), 2);
ok(
  sc.hits[0] === 1 && sc.hits[1] === 1,
  `scatterSurprise reports coverage as data (hits = [${sc.hits[0]}, ${sc.hits[1]}])`
);
disposeTupleSample(wrapSample);
wrapPos.dispose();
wrapNext.dispose();
ptAdv.dispose();

/* ══════════════════════════════════════════════════════════════════════════
   §5  GENERATOR GRADIENT SIGN
   ══════════════════════════════════════════════════════════════════════════
   Freeze the heads; take ONE gradient step on the POSITIONS with respect to
   generatorLoss. Mean surprise must go UP — the generator is rewarded for
   being unpredictable, so descending generatorLoss must ascend surprise.     */

console.log("\n§5  GENERATOR — descending generatorLoss must ASCEND surprise");

const gAdv = new Adversary({
  ...defaultAdversaryConfig(
    { tag: "wta", k: 3, relaxEps: 0.05 },
    { tag: "pair" },
    "periodic"
  ),
  batchTuples: 64,
  seed: 21,
});
const N = 256;
const gr = mulberry32(2024);
const posArr = new Float32Array(N * 2);
const nxtArr = new Float32Array(N * 2);
for (let i = 0; i < N * 2; i++) posArr[i] = gr();
for (let i = 0; i < N * 2; i++) nxtArr[i] = posArr[i] + (gr() - 0.5) * 0.02;

const posVar = tf.variable(tf.tensor2d(posArr, [N, 2]));
const nxtVar = tf.variable(tf.tensor2d(nxtArr, [N, 2]));

// Train the heads a little so the surprise landscape is not pure noise.
for (let s = 0; s < 60; s++) {
  const b = gAdv.sample(posVar as tf.Tensor2D, nxtVar as tf.Tensor2D);
  gAdv.trainStep(b);
  disposeTupleSample(b);
}

// FIX the tuple set once, so "surprise went up" is about the positions moving
// and not about a fresh random draw landing somewhere else.
const fixed = gAdv.sample(posVar as tf.Tensor2D, nxtVar as tf.Tensor2D);
const fixedIdx = fixed.idx;
disposeTupleSample(fixed);

const meanSurprise = (): number =>
  tf.tidy(() => {
    const s = gAdv.encode(posVar as tf.Tensor2D, nxtVar as tf.Tensor2D, fixedIdx);
    const v = gAdv.surprise(s).mean().dataSync()[0];
    disposeTupleSample(s);
    return v;
  });

const before = meanSurprise();

const gopt = tf.train.sgd(0.02);
for (let s = 0; s < 40; s++) {
  gopt.minimize(
    () =>
      tf.tidy(() => {
        const b = gAdv.encode(posVar as tf.Tensor2D, nxtVar as tf.Tensor2D, fixedIdx);
        const v = gAdv.generatorLoss(b);
        disposeTupleSample(b);
        return v;
      }),
    false,
    [nxtVar]
  );
}

const after = meanSurprise();
console.log(`      mean surprise before = ${before.toExponential(4)}   after 40 generator steps = ${after.toExponential(4)}`);
ok(after > before * 1.05, `generator ascends surprise (${after.toExponential(4)} > 1.05 × ${before.toExponential(4)})`);

posVar.dispose();
nxtVar.dispose();
gopt.dispose();
disposeAdversary(gAdv);

/* ══════════════════════════════════════════════════════════════════════════
   §6  ENCODING DEGENERACY — the reason `PredictorEnsemble` was deleted
   ══════════════════════════════════════════════════════════════════════════
   Run BOTH encodings against a real deterministic field, F(x) = amp·(sin 2πy,
   cos 2πx), next = wrap(pos + F·dt). This is the same structure as the shipped
   generator: the displacement is an exact function of the position.

   `point` feeds the absolute position and asks for the absolute displacement —
   exactly realizable, so the residual must go to ~0 and the adversary signal
   dies. `pair` canonicalizes by SE(2), which throws away everything except the
   separation r, so the conditional is genuinely spread and the residual must
   stay high. Residuals are reported as a fraction of RMS‖y‖ so the two
   encodings (different target scales) are comparable.

   The gate is a CONVERGENCE LADDER, not a fixed threshold, because the
   interesting distinction is REDUCIBLE vs IRREDUCIBLE error, and a single
   snapshot cannot tell them apart (an undertrained predictor also has a large
   residual). Probing the same run at 3 training budgets does: `point` must keep
   shrinking (the error is reducible — the target is exactly realizable), while
   `pair` must plateau (the error is irreducible — it is the conditional spread
   that survives canonicalization). That distinction is the entire thesis.

   Catches: the exact failure mode of the deleted ensemble. If someone later
   "improves" the pair encoding by feeding absolute positions back in, the pair
   residual starts shrinking too and the plateau assertion fails.              */

console.log("\n§6  ENCODING DEGENERACY — absolute (point) is exactly realizable, SE(2) (pair) is not");

const FN = 4096;
const FAMP = 0.012;
const frnd = mulberry32(31337);
const fPos = new Float32Array(FN * 2);
for (let i = 0; i < FN * 2; i++) fPos[i] = frnd();
const fNext = new Float32Array(FN * 2);
for (let i = 0; i < FN; i++) {
  const x = fPos[i * 2];
  const y = fPos[i * 2 + 1];
  const fx = FAMP * Math.sin(2 * Math.PI * y);
  const fy = FAMP * Math.cos(2 * Math.PI * x);
  fNext[i * 2] = (((x + fx) % 1) + 1) % 1;
  fNext[i * 2 + 1] = (((y + fy) % 1) + 1) % 1;
}
const fieldPos = tf.tensor2d(fPos, [FN, 2]);
const fieldNext = tf.tensor2d(fNext, [FN, 2]);

// Kept deliberately short — this is a CPU-tfjs script and this section is most
// of the suite's runtime. A longer run confirms the trend continues rather than
// bottoming out: at 1000/3000/9000 steps (batch 256, lr 3e-3) the point ladder
// reads 0.2291 → 0.1315 → 0.0984, i.e. still shrinking with no floor in sight,
// while pair/single reads 0.9240 → 0.9184 → 0.8995 (flat). The short ladder
// below reproduces the same qualitative separation in ~1/3 of the time.
const LADDER = [300, 900, 2700] as const;

/**
 * Train ONE adversary on the field and probe its normalized residual
 * E[min_k‖y − g_k‖] / RMS‖y‖ at each cumulative training budget in LADDER.
 */
function residualLadder(
  kind: Parameters<typeof defaultAdversaryConfig>[0],
  enc: Parameters<typeof defaultAdversaryConfig>[1]
): number[] {
  const adv = new Adversary({
    ...defaultAdversaryConfig(kind, enc, "periodic"),
    batchTuples: 128,
    hiddenUnits: 64,
    featureDim: 32,
    learningRate: 8e-3,
    seed: 90210,
  });
  const probe = () =>
    tf.tidy(() => {
      const b = adv.sample(fieldPos, fieldNext);
      const res = adv.winnerResidual(b).mean().dataSync()[0];
      const rms = Math.sqrt(b.y.square().sum(1).mean().dataSync()[0]);
      disposeTupleSample(b);
      return res / rms;
    });
  const out: number[] = [];
  let done = 0;
  for (const budget of LADDER) {
    for (; done < budget; done++) {
      const b = adv.sample(fieldPos, fieldNext);
      adv.trainStep(b);
      disposeTupleSample(b);
    }
    out.push(probe());
  }
  disposeAdversary(adv);
  return out;
}

const ladPoint = residualLadder({ tag: "single" }, { tag: "point" });
const ladPairS = residualLadder({ tag: "single" }, { tag: "pair" });
const ladPairW = residualLadder({ tag: "wta", k: 4, relaxEps: 0.05 }, { tag: "pair" });
// tri rungs: does the RICHER context (3 side lengths vs 1 separation) restore
// reducible structure? Measured answer: NO — the tri context is still E(2)-
// invariant, so the pose ambiguity survives, and the 6-D target (3 vertices'
// relative displacements) spreads the conditional even wider than pair's
// (calibration run 2026-07-27: tri/single 1.0013 → 1.0040 → 0.9607, i.e. flat
// at ~0.96 of RMS‖y‖ where pair/single plateaus at ~0.90; tri/wta4 reaches
// ~0.64 where pair/wta4 reaches ~0.50).
const ladTriS = residualLadder({ tag: "single" }, { tag: "tri" });
const ladTriW = residualLadder({ tag: "wta", k: 4, relaxEps: 0.05 }, { tag: "tri" });
const fmt = (l: number[]) => l.map((v) => v.toFixed(4)).join("  →  ");
console.log(`      steps                 ${LADDER.map((s) => String(s).padStart(6)).join("    ")}`);
console.log(`      point / single      ${fmt(ladPoint)}`);
console.log(`      pair  / single      ${fmt(ladPairS)}`);
console.log(`      pair  / wta k=4     ${fmt(ladPairW)}`);
console.log(`      tri   / single      ${fmt(ladTriS)}`);
console.log(`      tri   / wta k=4     ${fmt(ladTriW)}`);

const strictlyDecreasing = (l: number[]) => l.every((v, i) => i === 0 || v < l[i - 1]);
const shrink = (l: number[]) => l[l.length - 1] / l[0];

ok(
  strictlyDecreasing(ladPoint) && shrink(ladPoint) < 0.7,
  `point encoding's error is REDUCIBLE — it keeps shrinking with training (${fmt(ladPoint)}, final/first = ${shrink(ladPoint).toFixed(3)} < 0.7)`
);
ok(
  shrink(ladPairS) > 0.85,
  `pair encoding's error is IRREDUCIBLE — it plateaus (${fmt(ladPairS)}, final/first = ${shrink(ladPairS).toFixed(3)} > 0.85)`
);
ok(
  ladPairS[ladPairS.length - 1] > 2 * ladPoint[ladPoint.length - 1],
  `the SE(2) canonicalization is what creates the ambiguity (pair ${ladPairS[ladPairS.length - 1].toFixed(4)} > 2 × point ${ladPoint[ladPoint.length - 1].toFixed(4)})`
);
ok(
  ladPairW[ladPairW.length - 1] < 0.8 * ladPairS[ladPairS.length - 1],
  `K=4 heads quantize the conditional better than 1 head averages it (${ladPairW[ladPairW.length - 1].toFixed(4)} < 0.8 × ${ladPairS[ladPairS.length - 1].toFixed(4)})`
);
ok(
  ladPairW[ladPairW.length - 1] > 0.15,
  `...but the residual stays strictly positive — the generator signal survives K guesses (${ladPairW[ladPairW.length - 1].toFixed(4)} > 0.15)`
);
// Tri gates. The load-bearing one is the plateau: if someone "enriches" the tri
// context with anything pose-dependent (absolute positions, the chart frame
// angle, the signed area TOGETHER with absolute orientation, ...) the target
// becomes realizable, the residual starts shrinking like point's, and the
// first assertion fails.
ok(
  shrink(ladTriS) > 0.85,
  `tri encoding's error is IRREDUCIBLE too — richer context does NOT restore realizability (${fmt(ladTriS)}, final/first = ${shrink(ladTriS).toFixed(3)} > 0.85)`
);
ok(
  ladTriS[ladTriS.length - 1] > 2 * ladPoint[ladPoint.length - 1],
  `tri stays canonicalization-ambiguous, not merely undertrained (tri ${ladTriS[ladTriS.length - 1].toFixed(4)} > 2 × point ${ladPoint[ladPoint.length - 1].toFixed(4)})`
);
ok(
  ladTriW[ladTriW.length - 1] < 0.85 * ladTriS[ladTriS.length - 1],
  `K=4 heads quantize the tri conditional too (${ladTriW[ladTriW.length - 1].toFixed(4)} < 0.85 × ${ladTriS[ladTriS.length - 1].toFixed(4)})`
);
ok(
  ladTriW[ladTriW.length - 1] > 0.15,
  `...and the tri generator signal survives K guesses (${ladTriW[ladTriW.length - 1].toFixed(4)} > 0.15)`
);

fieldPos.dispose();
fieldNext.dispose();

/* ══════════════════════════════════════════════════════════════════════════
   §7  TENSOR LEAKS
   ══════════════════════════════════════════════════════════════════════════
   A leak here is invisible in the art (the frame still renders) and fatal after
   a few minutes of running (WebGPU OOM). Assert that a full construct → sample →
   train → surprise → generatorLoss → dispose cycle nets to zero live tensors. */

console.log("\n§7  LEAKS — a full adversary lifecycle must net to zero tensors");

const leakPos = tf.tensor2d(Array.from({ length: 128 }, () => [Math.random(), Math.random()]));
const leakNext = tf.tensor2d(Array.from({ length: 128 }, () => [Math.random(), Math.random()]));
const t0 = tf.memory().numTensors;
for (let rep = 0; rep < 3; rep++) {
  const a = new Adversary({
    ...defaultAdversaryConfig(
      { tag: "wta", k: 4, relaxEps: 0.05 },
      { tag: "pair" },
      "periodic"
    ),
    batchTuples: 32,
    seed: 7 + rep,
  });
  for (let s = 0; s < 5; s++) {
    const b = a.sample(leakPos, leakNext);
    a.trainStep(b);
    tf.tidy(() => a.surprise(b).dataSync());
    tf.tidy(() => a.generatorLoss(b).dataSync());
    tf.tidy(() => a.residuals(b).dataSync());
    tf.tidy(() => a.predictHeads(b.u).dataSync());
    disposeTupleSample(b);
  }
  disposeAdversary(a);
}
const t1 = tf.memory().numTensors;
leakPos.dispose();
leakNext.dispose();
console.log(`      live tensors ${t0} → ${t1}  (Δ = ${t1 - t0})`);
ok(t1 === t0, `no tensor leak across 3 adversary lifecycles (Δ = ${t1 - t0})`);

/* ══════════════════════════════════════════════════════════════════════════ */

console.log("\n§8  LEGACY EMA SCALE — value stabilization only (not a gradient proof)");
// Two identical raw-mode adversaries train on the same conditional structure at
// target scales 1× and 10×. The EMA keeps reported objective VALUES comparable.
// Because the EMA is tape-constant it does NOT remove instantaneous radial
// generator gradient; tools/adversary_strict_test.ts proves that negative
// control and proves the adjusted angular mode's zero radial derivative.
{
  const mkBatch = (b: number, scale: number, rnd: () => number): TupleSample => {
    const u = new Float32Array(b);
    const y = new Float32Array(b * 2);
    for (let i = 0; i < b; i++) {
      u[i] = 0.5;
      const mode = rnd() < 0.5 ? 1 : -1;
      y[2 * i] = scale * 0.05 * mode;
      y[2 * i + 1] = scale * 0.03 * mode;
    }
    return {
      u: tf.tensor2d(u, [b, 1]),
      y: tf.tensor2d(y, [b, 2]),
      idx: new Int32Array(b * 2),
      b,
      m: 2,
    };
  };
  const run = (scale: number) => {
    const adv = new Adversary({
      ...defaultAdversaryConfig(
        { tag: "wta", k: 2, relaxEps: 0.05 },
        { tag: "pair" },
        "periodic"
      ),
      batchTuples: 64,
      learningRate: 5e-3,
      seed: 31,
    });
    const rnd = mulberry32(41);
    for (let s = 0; s < 250; s++) {
      const b = mkBatch(64, scale, rnd);
      adv.trainStep(b);
      disposeTupleSample(b);
    }
    const evalB = mkBatch(1024, scale, mulberry32(99));
    const gen = tf.tidy(() => adv.generatorLoss(evalB).dataSync()[0]);
    const raw = tf.tidy(() => adv.surprise(evalB).mean().dataSync()[0]);
    const st = adv.rewardScaleState();
    disposeTupleSample(evalB);
    disposeAdversary(adv);
    if (st.tag !== "seeded") throw new Error("normalizer unseeded after 250 steps");
    return { gen, raw, rms: st.rms };
  };
  const s1 = run(1);
  const s10 = run(10);
  const rawRatio = s10.raw / s1.raw;
  const genRatio = s10.gen / s1.gen; // both negative → ratio positive
  console.log(
    `      raw surprise ×${rawRatio.toFixed(2)} across a 10× target scale; normalized generator term ×${genRatio.toFixed(2)}  (rms ${s1.rms.toExponential(2)} → ${s10.rms.toExponential(2)})`
  );
  ok(
    rawRatio > 4,
    `raw payoff DOES scale with target magnitude (×${rawRatio.toFixed(2)} for ×10 targets), confirming why raw mode is a negative control`
  );
  ok(
    genRatio > 0.5 && genRatio < 2.0,
    `legacy EMA keeps objective values comparable (×${genRatio.toFixed(2)}, bounds [0.5, 2.0]); this is not a radial-gradient claim`
  );
}

console.log("\n§9  minImage GRADIENT CONTRACT — round must stay gradient-transparent");
// minImage computes a − round(a): a gradient-identity wrap ONLY because tfjs
// registers d(round)/da = 0. That is a BACKEND CONTRACT, not our code — if a
// backend ever ships a different Round gradient, the generator's gradient
// through every wrapped displacement silently changes (flagged in the module
// docs). This pins it loudly.
{
  const g = tf.grad((a: tf.Tensor) => a.sub(a.round()).sum());
  const at = tf.tensor1d([0.4, 0.9, -0.3, 1.51]);
  const got = Array.from(g(at).dataSync());
  at.dispose();
  const worst = Math.max(...got.map((v) => Math.abs(v - 1)));
  ok(worst < 1e-7, `d(a − round(a))/da = 1 exactly (worst |Δ| = ${worst.toExponential(2)})`);
}

/* ══════════════════════════════════════════════════════════════════════════
   §10  HEAD SPREAD — the collapse DISAMBIGUATOR
   ══════════════════════════════════════════════════════════════════════════
   A skewed win histogram alone is ambiguous, and the HUD's red "HEAD COLLAPSE"
   light fires on the histogram. Two states produce it:

     BENIGN / DATA-LIMITED: K > #modes of P(Y|U). Surplus heads rarely win, but
     the winners are DISTINCT predictors → spread ~ mode separation.
     PATHOLOGY / OPTIMIZER PILEUP: heads converged to the same function.
     Effective K = 1 → spread ~ 0.

   Branch 1 reuses the §1 bimodal fixture with k = 4 (twice as many heads as
   modes). Branch 2 engineers a pileup: a UNIMODAL target with ε > 0, so every
   loser is steadily dragged onto the single mode until all four heads
   coincide. Both branches must show a skewed histogram — the spread is the
   ONLY number separating them, which is the whole claim being tested.       */

console.log("\n§10  HEAD SPREAD — same skewed histogram, opposite diagnoses");

function spreadAfter(
  mk: (b: number, rnd: () => number) => TupleSample,
  steps: number
): { meanPair: number; minPair: number; minWinFrac: number; tripwire: boolean } {
  const adv = new Adversary({
    ...BASE,
    kind: { tag: "wta", k: 4, relaxEps: 0.05 },
  });
  const rnd = mulberry32(3);
  for (let s = 0; s < steps; s++) {
    const batch = mk(256, rnd);
    adv.trainStep(batch);
    disposeTupleSample(batch);
  }
  const probe = tf.tensor2d(new Float32Array(32).fill(U_CONST), [32, 1]);
  const sp = adv.headSpread(probe);
  probe.dispose();
  if (sp.tag !== "spread") throw new Error("k=4 adversary must report a spread");
  const minWinFrac = Math.min(...adv.winStats().fractions);
  const tripwire = adv.collapsed();
  disposeAdversary(adv);
  return { meanPair: sp.meanPair, minPair: sp.minPair, minWinFrac, tripwire };
}

/** Unimodal y (single mode at (1,0), tiny jitter) — every head gets pulled in. */
function unimodalBatch(b: number, rnd: () => number): TupleSample {
  const u = new Float32Array(b).fill(U_CONST);
  const y = new Float32Array(b * 2);
  for (let i = 0; i < b; i++) {
    y[i * 2] = 1.0 + (rnd() - 0.5) * 0.01;
    y[i * 2 + 1] = (rnd() - 0.5) * 0.01;
  }
  return { u: tf.tensor2d(u, [b, 1]), y: tf.tensor2d(y, [b, 2]), idx: new Int32Array(b * 2), b, m: 2 };
}

const benign = spreadAfter(bimodalBatch, 900);
const pileup = spreadAfter(unimodalBatch, 900);
console.log(
  `      benign (bimodal, k=4): meanPair=${benign.meanPair.toFixed(4)}  minPair=${benign.minPair.toFixed(4)}  ` +
    `minWinFrac=${benign.minWinFrac.toFixed(3)}  tripwire=${benign.tripwire}  (mode sep ${MODE_SEP.toFixed(3)})`
);
console.log(
  `      pileup (unimodal, k=4): meanPair=${pileup.meanPair.toFixed(4)}  minPair=${pileup.minPair.toFixed(4)}  ` +
    `minWinFrac=${pileup.minWinFrac.toFixed(3)}  tripwire=${pileup.tripwire}`
);

// Both branches look identical to the histogram: wins are skewed either way.
ok(
  benign.minWinFrac < 0.15 && pileup.minWinFrac < 0.15,
  `both branches skew the win histogram (min fractions ${benign.minWinFrac.toFixed(3)} / ${pileup.minWinFrac.toFixed(3)} < 0.15) — the histogram alone cannot disambiguate`
);
// ...and the benign branch even fires the collapsed() tripwire, which is
// exactly why headSpread must exist: the tripwire flags a healthy state.
ok(
  benign.tripwire,
  `benign K>modes DOES fire the win-count tripwire (collapsed() = ${benign.tripwire}) — histogram-only detection over-alarms`
);
ok(
  benign.meanPair > 0.3 * MODE_SEP,
  `DATA-LIMITED reads as healthy spread: meanPair ${benign.meanPair.toFixed(4)} > ${(0.3 * MODE_SEP).toFixed(4)} (0.3 × mode separation)`
);
ok(
  pileup.meanPair < 0.1 * MODE_SEP,
  `OPTIMIZER PILEUP reads as ~zero spread: meanPair ${pileup.meanPair.toFixed(4)} < ${(0.1 * MODE_SEP).toFixed(4)} (0.1 × mode separation)`
);
ok(
  benign.meanPair > 5 * pileup.meanPair,
  `the two diagnoses are separated by a wide margin (${benign.meanPair.toFixed(4)} > 5 × ${pileup.meanPair.toFixed(4)})`
);

/* ══════════════════════════════════════════════════════════════════════════
   §11  TRI ENCODING — invariance gates, mirroring §3
   ══════════════════════════════════════════════════════════════════════════
   Scalene triangle (all sides distinct) so the canonical ordering is off the
   documented tie set; §3's rigid motion and mirror conventions reused.      */

console.log("\n§11  TRI ENCODING — invariance of (u, y) under the claimed group");

const TRI_ADV = new Adversary({
  ...defaultAdversaryConfig({ tag: "single" }, { tag: "tri" }, "periodic"),
  batchTuples: 1,
  seed: 4,
});

/** Encode one explicit triangle; `order` permutes the vertex LABELS fed in. */
function encodeTriOne(
  xs: number[][],
  ds: number[][],
  order: number[] = [0, 1, 2]
): { u: number[]; y: number[] } {
  const pos = tf.tensor2d(order.map((i) => xs[i]));
  const nxt = tf.tensor2d(order.map((i) => [xs[i][0] + ds[i][0], xs[i][1] + ds[i][1]]));
  const s = TRI_ADV.encode(pos, nxt, Int32Array.from([0, 1, 2]));
  const u = Array.from(s.u.dataSync());
  const y = Array.from(s.y.dataSync());
  disposeTupleSample(s);
  pos.dispose();
  nxt.dispose();
  return { u, y };
}

const distV = (a: number[], b: number[]) =>
  Math.sqrt(a.reduce((acc, v, i) => acc + (v - b[i]) * (v - b[i]), 0));

// Scalene triangle away from the seam; wrapping is exercised separately below.
const TX = [
  [0.42, 0.51],
  [0.58, 0.47],
  [0.49, 0.62],
];
const TD = [
  [0.011, -0.006],
  [-0.004, 0.009],
  [0.007, 0.013],
];
const triBase = encodeTriOne(TX, TD);
console.log(`      base   u=(${triBase.u.map((v) => v.toFixed(6))})`);
console.log(`             y=(${triBase.y.map((v) => v.toFixed(6))})`);

// u must come out sorted descending — the canonical order IS the spec.
ok(
  triBase.u[0] > triBase.u[1] && triBase.u[1] > triBase.u[2],
  `context is the sides in canonical descending order (${triBase.u.map((v) => v.toFixed(4))})`
);

// Rigid motion (same rotation+translation as §3).
const triRigid = encodeTriOne(TX.map(move), TD.map((d) => rot(d, TH)));
const tRigU = distV(triRigid.u, triBase.u);
const tRigY = distV(triRigid.y, triBase.y);
console.log(`      rigid  Δu=${tRigU.toExponential(2)}  Δy=${tRigY.toExponential(2)}`);
ok(tRigU < 1e-5, `u invariant under rotation+translation (Δu ${tRigU.toExponential(2)} < 1e-5)`);
ok(tRigY < 1e-5, `y invariant under rotation+translation (Δy ${tRigY.toExponential(2)} < 1e-5)`);

// All six vertex permutations — catches any base-vertex leak in the chart.
let worstPermU = 0;
let worstPermY = 0;
for (const p of [
  [0, 2, 1],
  [1, 0, 2],
  [1, 2, 0],
  [2, 0, 1],
  [2, 1, 0],
]) {
  const e = encodeTriOne(TX, TD, p);
  worstPermU = Math.max(worstPermU, distV(e.u, triBase.u));
  worstPermY = Math.max(worstPermY, distV(e.y, triBase.y));
}
console.log(`      perms  worst Δu=${worstPermU.toExponential(2)}  worst Δy=${worstPermY.toExponential(2)}`);
ok(
  worstPermU < 1e-6 && worstPermY < 1e-6,
  `encoding invariant under all 6 vertex permutations (worst Δu ${worstPermU.toExponential(2)}, Δy ${worstPermY.toExponential(2)})`
);

// Torus translation pushing the triangle across BOTH seams.
const triTorus = encodeTriOne(
  TX.map((p) => [(p[0] + 0.55) % 1, (p[1] + 0.47) % 1]),
  TD
);
const tTorU = distV(triTorus.u, triBase.u);
const tTorY = distV(triTorus.y, triBase.y);
console.log(`      torus  Δu=${tTorU.toExponential(2)}  Δy=${tTorY.toExponential(2)}`);
ok(tTorU < 1e-5 && tTorY < 1e-5, `encoding invariant under torus translation across the seam (Δu ${tTorU.toExponential(2)}, Δy ${tTorY.toExponential(2)})`);

// Reflection EQUIVARIANCE, the documented contract: u and the ê₁ (along)
// components y[0,2,4] are invariant; every ê₂ (perp) component y[1,3,5] flips
// sign. The magnitude check keeps the flip assertion non-vacuous.
const triRefl = encodeTriOne(
  TX.map((p) => [p[0], 1 - p[1]]),
  TD.map((d) => [d[0], -d[1]])
);
const tRefU = distV(triRefl.u, triBase.u);
const worstAlong = Math.max(...[0, 2, 4].map((i) => Math.abs(triRefl.y[i] - triBase.y[i])));
const worstPerpSum = Math.max(...[1, 3, 5].map((i) => Math.abs(triRefl.y[i] + triBase.y[i])));
const minPerpMag = Math.min(...[1, 3, 5].map((i) => Math.abs(triBase.y[i])));
console.log(
  `      mirror Δu=${tRefU.toExponential(2)}  worst |Δalong|=${worstAlong.toExponential(2)}  ` +
    `worst |perp+perp|=${worstPerpSum.toExponential(2)}  min |perp|=${minPerpMag.toExponential(2)}`
);
ok(tRefU < 1e-6, `u is reflection-INVARIANT — side lengths carry no chirality (Δu ${tRefU.toExponential(2)})`);
ok(
  worstAlong < 1e-6 && worstPerpSum < 1e-6 && minPerpMag > 1e-4,
  `reflection flips EXACTLY the three ê₂ components, as documented (|Δalong| ${worstAlong.toExponential(2)}, |perp sums| ${worstPerpSum.toExponential(2)}, min |perp| ${minPerpMag.toExponential(2)})`
);

TRI_ADV.dispose();

/* ══════════════════════════════════════════════════════════════════════════ */

console.log(`\n      tfjs tensors still alive at exit: ${tf.memory().numTensors}`);
console.log(failures === 0 ? "\nALL ADVERSARY CHECKS PASS" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
