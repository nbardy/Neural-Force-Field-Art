/**
 * Physical-drive + strict-game live-control gate.
 *
 *   bun tools/drive_controls_test.ts
 *
 * This is deliberately CPU/lightweight. The recurrence proof is algebraic;
 * AdvectKernel's live uniform setter is exercised without dispatching a shader,
 * and the production build covers the typed LoopHandle/React wiring.
 */
import * as tf from "@tensorflow/tfjs";
import {
  adversaryBatchSize,
  GALLERY,
  driveForForceMagnitude,
  forceMagnitudeForDrive,
  resolveLiveGameControls,
  resolveTrainBatchSize,
} from "../src/main";
import {
  Adversary,
  defaultAdversaryConfig,
} from "../src/core/gan/adversary";
import { AdvectKernel } from "../src/render/webgpu/advect";

await tf.setBackend("cpu");
await tf.ready();

let failures = 0;
function ok(condition: boolean, message: string): void {
  console.log(`${condition ? "  ok  " : " FAIL "} ${message}`);
  if (!condition) failures++;
}
const close = (a: number, b: number, tol = 1e-7) =>
  Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

const expected = [
  ["Adversary · Single (control)", 0.55],
  ["Adversary · WTA K=8", 0.65],
  ["Adversary · Pair WTA K=4", 0.65],
  ["Adversary · Tri WTA K=6", 0.65],
  ["Adversary · Quad WTA K=6", 0.65],
  ["Adversary · Agree + Disagree RGB", 0.6],
  ["Adversary · Chaos Weave", 0.75],
] as const;

console.log("\n§1 SHIPPED DEFAULTS + INVARIANT BOUND");
for (const [name, wantedDrive] of expected) {
  const piece = GALLERY.find((candidate) => candidate.name === name);
  ok(!!piece, `${name} exists`);
  if (!piece) continue;
  ok(piece.friction === 0.97, `${name}: friction = 0.97`);
  ok(piece.drive === wantedDrive, `${name}: drive = ${wantedDrive}`);
  const wantedMagnitude = forceMagnitudeForDrive(
    wantedDrive,
    piece.maxVelocity,
    piece.friction
  );
  ok(
    close(piece.forceMagnitude, wantedMagnitude),
    `${name}: forceMagnitude=${piece.forceMagnitude.toFixed(9)} follows drive formula`
  );
  ok(
    close(
      driveForForceMagnitude(
        piece.forceMagnitude,
        piece.maxVelocity,
        piece.friction
      ),
      wantedDrive
    ),
    `${name}: forceMagnitude inverse round-trips`
  );

  // Inductive recurrence gate. If |v_t| <= dV and |F_t| <= 1, then:
  // |v_{t+1}| <= f(dV + M) = dV exactly (up to float roundoff).
  const bound = wantedDrive * piece.maxVelocity;
  const nextWorst =
    piece.friction * (bound + piece.forceMagnitude);
  ok(
    nextWorst <= bound + 1e-9,
    `${name}: worst recurrence ${nextWorst.toFixed(9)} <= ${bound.toFixed(9)} px/f`
  );

  // Exercise both force signs for many steps as a regression against an
  // accidental formula/sign change.
  let velocity = 0;
  let peak = 0;
  for (let step = 0; step < 10_000; step++) {
    const force = step % 7 < 4 ? 1 : -1;
    velocity = piece.friction * (velocity + piece.forceMagnitude * force);
    peak = Math.max(peak, Math.abs(velocity));
  }
  ok(
    peak <= bound + 1e-5 && peak < piece.maxVelocity,
    `${name}: 10k adversarial force steps peak ${peak.toFixed(4)} < clip ${piece.maxVelocity}`
  );
}

console.log("\n§2 URL + GALLERY CANONICALIZATION");
const pair = GALLERY.find((piece) => piece.name === "Adversary · Pair WTA K=4")!;
const weave = GALLERY.find((piece) => piece.name === "Adversary · Chaos Weave")!;
const query = new URLSearchParams("drive=0.4&gLR=0.002&dLR=0.007");
for (const piece of [pair, weave]) {
  const live = resolveLiveGameControls(piece, query);
  ok(live.drive === 0.4, `${piece.name}: global ?drive survives gallery rebuild`);
  ok(live.generatorLearningRate === 0.002, `${piece.name}: ?gLR survives gallery rebuild`);
  ok(
    live.discriminatorLearningRate === 0.007,
    `${piece.name}: ?dLR survives gallery rebuild`
  );
  ok(
    close(
      live.forceMagnitude,
      forceMagnitudeForDrive(0.4, piece.maxVelocity, piece.friction)
    ),
    `${piece.name}: URL drive recomputes physical force`
  );
}
const legacy = GALLERY.find((piece) => !piece.adversary || piece.adversary.tag === "off")!;
const legacyLive = resolveLiveGameControls(
  legacy,
  new URLSearchParams("drive=0.1")
);
ok(!legacyLive.driveEnabled, "non-adversary piece does not opt into drive");
ok(
  legacyLive.forceMagnitude === legacy.forceMagnitude,
  "non-adversary forceMagnitude is unchanged by ?drive"
);
const resized = resolveLiveGameControls(pair, new URLSearchParams("drive=0.65"), 40);
ok(
  close(
    resized.forceMagnitude,
    forceMagnitudeForDrive(0.65, 40, pair.friction)
  ),
  "changing maxVelocity preserves drive by recomputing forceMagnitude"
);
ok(
  resolveLiveGameControls(pair, new URLSearchParams("drive=99")).drive === 1,
  "drive URL clamps to proof-preserving [0,1]"
);
ok(adversaryBatchSize(128) === 128, "train B=128 reaches fused adversary as B=128");
ok(adversaryBatchSize(512) === 512, "train B=512 reaches fused adversary as B=512");
ok(
  adversaryBatchSize(4096) === 1024,
  "train B respects the fused adversary's compiled batchCap=1024"
);

// REGRESSION (2026-08-17): the "train B" slider went to 4096 while FusedTrainer
// was built with batchCap 1024. Over-cap reached FusedTrainer.record(), which
// throws — inside the rAF tick, which re-arms itself on its last line, so the
// animation stopped permanently. Over-cap must now resolve to a runnable n and
// SAY it clamped; only non-finite input is fatal.
{
  const under = resolveTrainBatchSize(512, 4096);
  ok(under.tag === "ok" && under.n === 512, "in-cap train B passes through unchanged");
  const over = resolveTrainBatchSize(9000, 4096);
  ok(
    over.tag === "clamped" && over.n === 4096,
    "over-cap train B clamps to the cap instead of reaching record()'s throw"
  );
  ok(
    over.tag === "clamped" && over.requested === 9000 && over.cap === 4096,
    "the clamp carries its provenance (requested + cap) for the warning"
  );
  ok(resolveTrainBatchSize(0.2, 4096).n === 1, "sub-1 train B floors at 1, never 0");
  let nonFiniteRejected = false;
  try {
    resolveTrainBatchSize(Number.NaN, 4096);
  } catch {
    nonFiniteRejected = true;
  }
  ok(nonFiniteRejected, "non-finite train B is a typed error, not a silent default");
}

console.log("\n§3 LIVE UNIFORM + TFJS PREDICTOR LR");
// TypeScript private is runtime-normal: isolate the setter/getter contract
// without constructing tfjs-webgpu or touching a GPU.
const fakeAdvect = Object.create(AdvectKernel.prototype) as AdvectKernel & {
  uniF: Float32Array;
};
Object.defineProperty(fakeAdvect, "uniF", {
  value: new Float32Array(12),
  writable: false,
});
fakeAdvect.setForceMagnitude(0.4824742268);
ok(
  close(fakeAdvect.forceMagnitude, 0.4824742268),
  "AdvectKernel forceMagnitude live uniform setter/getter agree"
);
let setterRejected = false;
try {
  fakeAdvect.setForceMagnitude(Number.NaN);
} catch {
  setterRejected = true;
}
ok(setterRejected, "AdvectKernel rejects a non-finite live force");

const adversary = new Adversary({
  ...defaultAdversaryConfig({ tag: "single" }, { tag: "point" }, "periodic"),
  batchTuples: 8,
  learningRate: 1e-3,
});
const probe = tf.zeros([8, 2]) as tf.Tensor2D;
const beforeTensor = adversary.predictHeads(probe);
const before = new Float32Array(beforeTensor.dataSync());
beforeTensor.dispose();
adversary.setLearningRate(7e-3);
const afterTensor = adversary.predictHeads(probe);
const after = new Float32Array(afterTensor.dataSync());
afterTensor.dispose();
let maxDelta = 0;
for (let i = 0; i < before.length; i++) {
  maxDelta = Math.max(maxDelta, Math.abs(before[i] - after[i]));
}
ok(adversary.learningRate === 7e-3, "tfjs predictor exposes its live D LR");
ok(maxDelta === 0, "changing D LR preserves predictor weights bit-for-bit");
let lrRejected = false;
try {
  adversary.setLearningRate(0);
} catch {
  lrRejected = true;
}
ok(lrRejected, "tfjs predictor rejects non-positive D LR");
probe.dispose();
adversary.dispose();

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL DRIVE CONTROL CHECKS PASS");
process.exit(failures ? 1 : 0);
