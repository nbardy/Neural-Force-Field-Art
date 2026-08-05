/**
 * Reference gates for orthogonal adversary target/objective modes.
 *
 * Run:
 *   bun tools/adversary_objectives_test.ts
 */
import * as tf from "@tensorflow/tfjs";
import {
  Adversary,
  AdversaryConfigError,
  DEFAULT_SOFT_ANGLE_TAU,
  disposeTupleSample,
  energyAnchorLoss,
  generatorObjectiveTerm,
  objectiveDims,
  softAngleResidual,
  softSphericalEmbedding,
  type AdversaryLoss,
  type AdversaryObjectiveTerms,
  type AdversaryTarget,
  type TupleEncoding,
} from "../src/core/gan/adversary";

await tf.setBackend("cpu");
await tf.ready();

let failures = 0;
function ok(condition: boolean, message: string): void {
  if (condition) console.log(`  ✓ ${message}`);
  else {
    console.error(`  ✗ ${message}`);
    failures++;
  }
}
function close(a: number, b: number, tol = 1e-5): boolean {
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tol;
}
function maxDelta(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

function makeAdversary(
  encoding: TupleEncoding,
  loss: AdversaryLoss,
  target: AdversaryTarget = { tag: "force" }
): Adversary {
  return new Adversary({
    kind: { tag: "single" },
    encoding,
    target,
    loss,
    observerGeometry: "euclidean",
    batchTuples: 2,
    hiddenUnits: 8,
    featureDim: 6,
    learningRate: 1e-3,
    seed: 73,
  });
}

const SOFT: AdversaryLoss = { tag: "soft-angle", tau: 0.1 };
const REL: AdversaryLoss = {
  tag: "angle-relative-scale",
  tau: 0.1,
  scaleWeight: 0.4,
  energyWeight: 0.2,
  energyTarget: 1,
};
const HOLD: AdversaryLoss = {
  tag: "angle-scale-hold",
  tau: 0.1,
  scaleWeight: 0.4,
  energyWeight: 0.2,
  energyTarget: 1,
};

console.log("\n§1 exact soft S² embedding and derivative");
{
  const zero = tf.tidy(() =>
    Array.from(softSphericalEmbedding(tf.tensor2d([[0, 0]]), 0.1).dataSync())
  );
  ok(maxDelta(zero, [0, 0, 1]) < 1e-7, "ψτ(0) is the finite north pole (0,0,1)");

  const blocks = tf.tidy(() =>
    Array.from(
      softSphericalEmbedding(tf.tensor2d([[1, 0, 0, 2, -3, 4]]), 0.1).dataSync()
    )
  );
  ok(blocks.length === 9 && blocks.every(Number.isFinite), "embedding is per 2-vector for dy=6");

  const target = tf.tensor2d([[0.2, -0.7]]);
  const f = (x: number): number =>
    tf.tidy(() =>
      softAngleResidual(tf.tensor2d([[x, 0.3]]), target, 0.1).dataSync()[0]
    );
  const x = tf.scalar(0.35);
  const analytic = tf.tidy(() =>
    tf
      .grad((q: tf.Tensor) =>
        softAngleResidual(q.reshape([1, 1]).concat(tf.tensor2d([[0.3]]), 1), target, 0.1).sum()
      )(x)
      .dataSync()[0]
  );
  const h = 1e-3;
  const fd = (f(0.35 + h) - f(0.35 - h)) / (2 * h);
  ok(close(analytic, fd, 2e-3), `autograd is the exact S² chain rule (AD ${analytic.toFixed(6)}, FD ${fd.toFixed(6)})`);

  const near = tf.tidy(() => {
    const z = tf.tensor1d([0, 0]);
    return Array.from(
      tf.grad((q: tf.Tensor) =>
        softAngleResidual(q.reshape([1, 2]), tf.tensor2d([[1, 0]]), 0.1).sum()
      )(z).dataSync()
    );
  });
  ok(
    near.every(Number.isFinite) && Math.hypot(...near) <= 2 / 0.1,
    `near-zero exact gradient is finite and bounded (${Math.hypot(...near).toFixed(3)})`
  );
  target.dispose();
  x.dispose();
}

console.log("\n§2 canonical target/loss shape validation");
{
  ok(objectiveDims({ tag: "point" }, { tag: "force" }, SOFT).du === 2, "force point context is x");
  ok(
    objectiveDims({ tag: "point" }, { tag: "post-velocity" }, SOFT).du === 4,
    "post-velocity point context is [x,v]"
  );
  const pairRel = objectiveDims({ tag: "pair-rotation-scale-raw" }, { tag: "force" }, REL);
  const adjustedRel = objectiveDims(
    { tag: "pair-rotation-scale-adjusted" },
    { tag: "force" },
    REL
  );
  const triHold = objectiveDims({ tag: "tri" }, { tag: "force" }, HOLD);
  const quadRel = objectiveDims({ tag: "quad-labelled" }, { tag: "force" }, REL);
  ok(pairRel.dy === 2 && pairRel.ds === 1 && pairRel.out === 3, "pair rel-scale head is 2+1");
  ok(
    adjustedRel.ds === 1 && adjustedRel.out === 3,
    "legacy adjusted spelling accepts the explicit relative-scale loss"
  );
  ok(triHold.dy === 6 && triHold.ds === 3 && triHold.out === 9, "tri scale-hold head is 6+3");
  ok(quadRel.dy === 8 && quadRel.ds === 4 && quadRel.out === 12, "quad rel-scale head is 8+4");

  const rejects = (fn: () => unknown): boolean => {
    try {
      fn();
      return false;
    } catch (e) {
      return e instanceof AdversaryConfigError;
    }
  };
  ok(
    rejects(() => objectiveDims({ tag: "point" }, { tag: "force" }, REL)),
    "point relative-scale is rejected"
  );
  ok(
    rejects(() =>
      objectiveDims({ tag: "pair" }, { tag: "post-velocity" }, SOFT)
    ),
    "relational post-velocity is rejected"
  );
  ok(
    rejects(() => makeAdversary({ tag: "pair" }, { tag: "soft-angle", tau: 0 })),
    "non-positive soft-angle tau is rejected"
  );
}

console.log("\n§3 local relative scale: invariant, swap-safe, and zero-safe");
{
  const adv = makeAdversary({ tag: "pair-rotation-scale-raw" }, REL);
  const pos = tf.tensor2d([
    [0.2, 0.3],
    [0.75, 0.45],
  ]);
  const base = tf.tensor2d([
    [0.3, 0.4], // norm .5
    [1.2, -0.5], // norm 1.3
  ]);
  const encode = (scale: number, idx = Int32Array.from([0, 1])) => {
    const force = base.mul(scale) as tf.Tensor2D;
    const s = adv.encodeTarget({ tag: "force", pos, force }, idx);
    force.dispose();
    return s;
  };
  const a = encode(1);
  const b = encode(17);
  const swapped = encode(1, Int32Array.from([1, 0]));
  const av = Array.from(a.relativeScale!.dataSync());
  const bv = Array.from(b.relativeScale!.dataSync());
  const sv = Array.from(swapped.relativeScale!.dataSync());
  ok(maxDelta(av, bv) < 2e-5, `uniform 17× scaling leaves local contrast unchanged (Δ ${maxDelta(av, bv).toExponential(2)})`);
  ok(maxDelta(av, sv) < 2e-5, "pair absolute log contrast is swap invariant");

  const radial = tf.tidy(() => {
    const c = tf.scalar(1);
    return tf
      .grad((q: tf.Tensor) => {
        const force = base.mul(q) as tf.Tensor2D;
        const s = adv.encodeTarget({ tag: "force", pos, force }, Int32Array.from([0, 1]));
        const out = s.relativeScale!.sum();
        disposeTupleSample(s);
        return out;
      })(c)
      .dataSync()[0];
  });
  ok(Math.abs(radial) < 3e-5, `uniform radial derivative is zero (${radial.toExponential(2)})`);

  const equalMag = tf.tensor2d([
    [1, 0],
    [0, -1],
  ]);
  const eq = adv.encodeTarget({ tag: "force", pos, force: equalMag }, Int32Array.from([0, 1]));
  ok(Math.abs(eq.relativeScale!.dataSync()[0]) < 1e-7, "constant/max equal magnitudes have zero contrast");

  const zero = tf.zeros([2, 2]) as tf.Tensor2D;
  const zs = adv.encodeTarget({ tag: "force", pos, force: zero }, Int32Array.from([0, 1]));
  ok(
    zs.objectiveActivity!.dataSync()[0] === 0 &&
      zs.relativeScale!.dataSync()[0] === 0,
    "all-zero tuple is explicitly inactive with a finite zero descriptor"
  );
  for (const s of [a, b, swapped, eq, zs]) disposeTupleSample(s);
  pos.dispose();
  base.dispose();
  equalMag.dispose();
  zero.dispose();
  adv.dispose();
}

console.log("\n§4 tri canonical and labelled-quad scale descriptors");
{
  const tri = makeAdversary({ tag: "tri" }, REL);
  const p = tf.tensor2d([
    [0.15, 0.2],
    [0.72, 0.28],
    [0.38, 0.81],
  ]);
  const f = tf.tensor2d([
    [0.3, 0.1],
    [1.1, 0.2],
    [0.2, 0.65],
  ]);
  const t0 = tri.encodeTarget({ tag: "force", pos: p, force: f }, Int32Array.from([0, 1, 2]));
  const t1 = tri.encodeTarget({ tag: "force", pos: p, force: f }, Int32Array.from([2, 0, 1]));
  ok(
    maxDelta(t0.relativeScale!.dataSync(), t1.relativeScale!.dataSync()) < 1e-6,
    "tri relative logs follow the same canonical vertex order"
  );
  const tv = Array.from(t0.relativeScale!.dataSync());
  ok(Math.abs(tv.reduce((x, y) => x + y, 0)) < 1e-6, "tri centered logs sum to zero");

  const quad = makeAdversary({ tag: "quad-labelled" }, HOLD);
  const qp = tf.tensor2d([
    [0.2, 0.2],
    [0.7, 0.25],
    [0.75, 0.75],
    [0.25, 0.7],
  ]);
  const qf = tf.tensor2d([
    [0.2, 0.1],
    [0.4, 0.3],
    [0.8, 0.1],
    [0.3, 1.0],
  ]);
  const qs = quad.encodeTarget(
    { tag: "force", pos: qp, force: qf },
    Int32Array.from([0, 1, 2, 3])
  );
  const qv = Array.from(qs.relativeScale!.dataSync());
  ok(qv.length === 4 && Math.abs(qv.reduce((x, y) => x + y, 0)) < 1e-6, "quad keeps four labelled centered logs");

  disposeTupleSample(t0);
  disposeTupleSample(t1);
  disposeTupleSample(qs);
  p.dispose();
  f.dispose();
  qp.dispose();
  qf.dispose();
  tri.dispose();
  quad.dispose();
}

console.log("\n§5 generator signs and fixed-energy anchor");
{
  const angle = tf.tensor1d([2]);
  const scale = tf.tensor1d([3]);
  const disc = tf.tidy(() => angle.add(scale.mul(0.4)) as tf.Tensor1D);
  const terms: AdversaryObjectiveTerms = { angle, scale, discriminator: disc };
  const relG = tf.tidy(() => generatorObjectiveTerm(REL, terms).dataSync()[0]);
  const holdG = tf.tidy(() => generatorObjectiveTerm(HOLD, terms).dataSync()[0]);
  ok(close(relG, -3.2), "relative-scale G opposes angle and scale jointly");
  ok(close(holdG, -0.8), "scale-hold G opposes angle but cooperates on scale");

  const anchorGrad = (value: number): number =>
    tf.tidy(() => {
      const x = tf.scalar(value);
      return tf
        .grad((q: tf.Tensor) => {
          const sample = {
            u: tf.zeros([1, 1]) as tf.Tensor2D,
            y: tf.zeros([1, 2]) as tf.Tensor2D,
            tupleEnergy: q.square().reshape([1]) as tf.Tensor1D,
            objectiveActivity: tf.ones([1]) as tf.Tensor1D,
            idx: new Int32Array(1),
            b: 1,
            m: 1,
          };
          return energyAnchorLoss(sample, HOLD);
        })(x)
        .dataSync()[0];
    });
  const below = anchorGrad(0.5);
  const above = anchorGrad(2);
  ok(below < 0 && above > 0, `energy anchor pushes outward below target and inward above (${below.toFixed(3)}, ${above.toFixed(3)})`);
  angle.dispose();
  scale.dispose();
  disc.dispose();
}

console.log("\n§6 post-update velocity observes incoming momentum");
{
  const adv = makeAdversary({ tag: "point" }, SOFT, { tag: "post-velocity" });
  const pos = tf.tensor2d([
    [0.4, 0.6],
    [0.4, 0.6],
  ]);
  const vel = tf.tensor2d([
    [0.1, -0.2],
    [-0.8, 0.7],
  ]);
  const next = tf.tensor2d([
    [0.15, -0.1],
    [-0.7, 0.65],
  ]);
  const s = adv.encodeTarget(
    { tag: "post-velocity", pos, velocity: vel, nextVelocity: next },
    Int32Array.from([0, 1])
  );
  const u = Array.from(s.u.dataSync());
  ok(
    s.u.shape[1] === 4 &&
      close(u[0], u[4]) &&
      close(u[1], u[5]) &&
      (!close(u[2], u[6]) || !close(u[3], u[7])),
    "same x with different incoming v produces distinct [x,v] contexts"
  );
  ok(maxDelta(s.y.dataSync(), next.dataSync()) === 0, "target is the supplied normalized vNext exactly");
  let rejected = false;
  try {
    adv.sampleSignal(pos, next);
  } catch (e) {
    rejected = e instanceof AdversaryConfigError;
  }
  ok(rejected, "force compatibility API rejects a post-velocity configuration");
  disposeTupleSample(s);
  pos.dispose();
  vel.dispose();
  next.dispose();
  adv.dispose();
}

console.log("\n§7 compatibility, trainability, and disposal");
{
  const before = tf.memory().numTensors;
  const raw = makeAdversary({ tag: "pair" }, { tag: "raw-vector" });
  const pos = tf.tensor2d([
    [0.1, 0.2],
    [0.6, 0.7],
    [0.25, 0.8],
    [0.9, 0.15],
  ]);
  const force = tf.tensor2d([
    [0.2, 0.3],
    [-0.4, 0.6],
    [0.7, -0.2],
    [-0.1, -0.8],
  ]);
  const s = raw.sampleSignal(pos, force);
  const report = raw.trainStep(s);
  const g = tf.tidy(() => raw.generatorLoss(s).dataSync()[0]);
  ok(s.y.shape[1] === 2 && !s.relativeScale, "sampleSignal preserves force/raw tuple shape");
  ok(Number.isFinite(report.loss) && Number.isFinite(g), "raw compatibility train/generator paths stay finite");
  disposeTupleSample(s);
  pos.dispose();
  force.dispose();
  raw.dispose();
  ok(tf.memory().numTensors === before, `construct/train/dispose returns to tensor baseline ${before}`);
}

ok(DEFAULT_SOFT_ANGLE_TAU > 0, "default soft-angle tau is named and positive");
console.log(failures === 0 ? "\nALL ADVERSARY OBJECTIVE CHECKS PASS" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
