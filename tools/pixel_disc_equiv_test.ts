/**
 * Pixel discriminator — CPU oracle ≡ GPU kernel, numerically.
 *
 *   bun tools/pixel_disc_equiv_test.ts
 *
 * §4 of docs/PLAN_MULTI_GUESS_MODULARIZATION.md. `tools/pixel_disc_test.ts`
 * only checks that the CPU discLoss falls and that the GPU extGrads are finite
 * and non-zero; NOTHING there compares a number produced by
 * `src/core/gan/pixel_disc.ts` against the same number produced by
 * `src/render/webgpu/pixel_disc_wgsl.ts`. The two are hand-mirrored across ~10
 * sites (plan §3e) and the multi-guess port adds a min-over-K fold to every one
 * of them, so this is the gate that has to exist first.
 *
 *   §1  shared-constant preconditions (no GPU needed), incl. the multi-guess
 *       head layout and the §3h per-guess init asymmetry
 *   §1b the encoding/family GATES and the scratch block hashgrid needs, also
 *       without a GPU (docs/PLAN_PIXEL_GENERATOR_ARCH.md §2/§2a/§3)
 *   §2  CPU ≡ GPU at the small test config  G=8  E=4  K=8  hidden=8
 *   §3  CPU ≡ GPU at the shipped gallery config G=16 E=8 K=16 hidden=32
 *       (src/main.ts:2403 et al. — PIXEL_DISC_DEFAULTS)
 *   §4  HASHGRID and FAMILY-PLANED generator fields: the same critic
 *       comparisons, plus the GENERATOR SEAM (dL/dF → extGrads) against live
 *       tfjs autograd — the only assertion in this file that can see the field
 *       backward at all (see the §4 banner for why that matters)
 *
 * Compared, per kind (vec-field, next-frame, real-fake, inpaint) × per guess
 * setting (single, k=2 ε=0.05, k=4 ε=0.22):
 *   discLoss          CPU pixelDiscStep().discLoss   vs critMeta[3·nW + 0]
 *   genLoss           CPU pixelDiscStep().genLoss    vs critMeta[3·nW + 1]
 *   discGradPacked    CPU .discGradPacked[0..nW)     vs critMeta[0..nW)
 *   dF                CPU .dF[0..2B)                 vs scratch[pBase + 6..8)
 *   winner            CPU .winIdx[c]                 vs scratch[winBase + c]
 *   win counts        histogram of CPU .winIdx       vs critMeta[3·nW + 4 + j]
 * plus the density itself, as a diagnostic for the fixed-point splat.
 *
 * All of those are ENCODING-BLIND, and knowing that is what §4 is for: the CPU
 * oracle is handed the positions and the FORCES the shader itself computed, so
 * it never evaluates the neural field and every row above stays green no matter
 * what the field forward or backward does. §4 supplies the other oracle.
 *
 * WHY THE WINNER IS COMPARED CELL BY CELL AND NOT ONLY THROUGH THE GRADIENT: a
 * disagreement about which guess won swaps that cell's weights between `1−ε` and
 * `ε/(k−1)`. On a near-tie the payoff barely moves and the two backends' summed
 * gradients can stay inside tolerance while their SELECTIONS differ — which is
 * the one thing the whole multi-guess port is about. The win-count row
 * additionally gates §3g: for vec-field the counts must sum to the ACTIVE cell
 * count, not to G², or inactive cells are being credited to guess 0.
 *
 * `real-fake` runs at `single` only, and §1 asserts that anything else is
 * REFUSED rather than silently degraded — see `validatePixelDims`.
 *
 * HOW THE TWO SIDES ARE FORCED ONTO IDENTICAL INPUTS — every one of these is a
 * way this test would otherwise silently lie:
 *
 *  - WEIGHTS. `PixelDiscTrainer`'s ctor uploads
 *    `packPixelDiscWeights(initPixelDiscWeights(dims, seed))`
 *    (pixel_disc_train.ts:147-151). Rather than assume that, the harness reads
 *    critW back and asserts it byte-equals the same call, then unpacks THOSE
 *    f32 values as the CPU oracle's f64 weights — so both sides start from
 *    bit-identical numbers, not merely from the same seed.
 *  - PARTICLES. `uni.width = uni.height = 1`, so `sampleAndSplat`'s
 *    `uk = partPos/size` is the identity and the uploaded positions ARE the
 *    normalized coordinates. Positions and the field's F are then read back out
 *    of `scratch` (oPos=0, oF=2) and fed to the CPU in sample order rather than
 *    re-derived — `cursor` is 0 and stays 0 because partCount == b.
 *  - vec-field TARGETS. `fillForceGrid` evaluates the neural field at cell
 *    centres into auxA/auxB. Reproducing that on the CPU would mean
 *    reimplementing the whole field forward, so the harness reads auxA/auxB
 *    back and passes them as `forceGrid`.
 *  - real-fake FAKE CLOUD. The two sides genuinely disagree (plan §4): CPU is
 *    one sequential `mulberry32(12345)` stream (pixel_disc.ts:672-677), GPU is
 *    per-particle independent seeds (pixel_disc_wgsl.ts:1084-1086). The harness
 *    replicates the GPU's cloud, injects it through the existing `fakePos` opt,
 *    and then VERIFIES the injection by splatting it and comparing against the
 *    GPU's own Dfake (auxA). Nothing is compared across mismatched clouds.
 *  - inpaint MASK. Same seed does NOT give the same mask — see the comment on
 *    `pickInpaintSeed`. The harness searches for a maskSeed on which the two
 *    mask generators happen to agree and then verifies the mask cell-by-cell
 *    against the GPU's auxB.
 *  - UPDATE RULE. CPU applies plain SGD (pixel_disc.ts:401), GPU applies Adam
 *    (pixel_disc_wgsl.ts:1369-1382), so post-update weights are NOT comparable.
 *    Every COMPARED step therefore runs `applyDisc: false` and compares
 *    GRADIENTS, which also keeps the GPU state idempotent across the two
 *    encodeSteps (no adamStep, no weight change, cursor returns to 0).
 *  - WARM-UP. Before comparing, the critic is trained for a few dozen Adam
 *    steps ON THE GPU and the CPU then adopts those weights wholesale — the CPU
 *    never replays Adam. This is required, not decorative: at a freshly
 *    initialised critic next-frame's generator gradient is identically zero by
 *    construction (see the comment at WARM_STEPS), so a fresh-weights
 *    comparison of dF would be noise against noise.
 *
 *  - GUESS SETTING. `dims.guesses` is passed to BOTH sides, and the head is a
 *    stride, so a k-mismatch would change `pixelDiscWeightCount` and be caught
 *    by the critW byte-equality check before any number is compared.
 */
import {
  PIXEL_DISC_SOFT_EPS,
  PIXEL_DISC_DEFAULTS,
  initPixelDiscWeights,
  packPixelDiscWeights,
  unpackPixelDiscWeights,
  pixelDiscWeightCount,
  pixelDiscStep,
  softSplatDensity,
  makeInpaintMask,
  headFloats,
  headFloatsPerGuess,
  pixelGuessCount,
  type PixelDiscDims,
  type PixelGanKind,
} from "../src/core/gan/pixel_disc";
import { wtaScalars, type GuessKind } from "../src/core/gan/wta";
import {
  CLASS_SALT,
  layoutField,
  type Encoding,
  type FieldLayout,
  type LayerDims,
} from "../src/render/webgpu/advect_wgsl";
import {
  classifyPixelDiscFusion,
  pixelDiscShader,
  pixelWeightLayout,
  pixelPartLayout,
  pixelCritWinBase,
  pixelWinCounters,
  PIXEL_STATS_WIN_BASE,
} from "../src/render/webgpu/pixel_disc_wgsl";

console.log("=== pixel_disc CPU ↔ GPU equivalence ===");

let failures = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
  if (!cond) failures++;
};
const note = (msg: string) => console.log(`  ..    ${msg}`);

// ── tolerances ─────────────────────────────────────────────────────────────
// Set from MEASURED agreement on today's single-guess code, then given headroom
// — not guessed. Worst values over all four kinds × both configs, Apple M-series
// / bun-webgpu, 2026-08-22 (each cell is the worst of the two configs):
//
//   quantity   worst measured                       bound here   headroom
//   discLoss   rel 4.5e-6  (inpaint, G=8)           5e-5         ~6x
//   genLoss    rel 8.5e-6  (real-fake, G=8)         5e-5         ~6x
//   discGrad   rel 7.1e-6  of scale 2.3e-1          1e-4         ~14x
//              cos deficit 1.6e-11                  1e-7         ~6000x
//   dF         rel 2.6e-5  of scale 6.7e-5          2e-4         ~8x
//              cos deficit 9.6e-11                  1e-7         ~1000x
//   density    abs 8.4e-6                           1e-4         ~12x
//
// Two things set that floor, and neither is a disagreement between the
// implementations:
//
//  1. `sampleAndSplat` accumulates density through i32 atomics at
//     DENS_SCALE=1e6 and TRUNCATES each of the four bilinear taps, so the GPU's
//     D sits ~1e-6·(taps per cell) below the CPU's f64 D. Everything
//     downstream inherits that bias — it is why the density row above is a
//     one-sided 8e-6 rather than f32 epsilon.
//  2. f32 cancellation in sums whose result is much smaller than its terms.
//     The loosest single number in the table, real-fake's genLoss, is exactly
//     this: genLoss IS the MLP logit (0.039) summed from ~hidden terms of
//     magnitude ~0.1.
//
// The COSINE bound is the structural one and is deliberately far tighter in
// spirit than the rel bounds, even though its numeric headroom is larger: the
// thing it exists to catch — a hand-mirrored site drifting (wrong index, a
// dropped term, a missing 1/nActive) — moves cosine by 1e-3 to 1, never by
// 1e-8. Both failures this test found while being written showed up as cos
// deficits of 0.78 and 1.0 with plausible-looking magnitudes.
//
// If a future change makes these bounds fail, the first question is which of
// the two floors above moved — NOT whether to widen the bound.
const REL_LOSS_MAX = 5e-5;
const REL_GRAD_MAX = 1e-4;
const REL_DF_MAX = 2e-4;
/**
 * dF bound WHEN THERE IS MORE THAN ONE GUESS. A separate number, not a widened
 * REL_DF_MAX, because a specific floor moves and it is worth naming rather than
 * absorbing — measured 2026-08-22, `min_c min_j r` after 60 warm steps:
 *
 *   config   k=1       k=2       k=4       cells within 10·SOFT_EPS of the knee
 *   G=8      4.95e-2   2.11e-3   2.35e-4   0 → 0 → 8/64
 *   G=16     1.67e-3   1.26e-3   1.77e-4   0 → 0 → 7/256
 *
 * Taking a MINIMUM over k guesses drives the winning residual toward the
 * soft-L1 knee at `r = SOFT_EPS = 1e-4` — that is what a min-distance loss is
 * FOR. But `d(diff/r)/d(diff) = SOFT_EPS²/r³`, which is ~1 out at r = 5e-2 and
 * ~1e4 at r ≈ SOFT_EPS. So on the handful of cells that reach the knee, the f32
 * vs f64 gap in `pred` is amplified four orders of magnitude before it ever
 * reaches the density VJP, whose four bilinear taps then cancel on top of it.
 *
 * Worst observed under this bound: 3.5e-4 (next-frame k=4 ε=0.22, G=8) — 3x
 * headroom. The COSINE bound is NOT relaxed: the structural check still holds at
 * 1-3.6e-9 against COS_MIN, which is what would actually catch a mirrored site
 * drifting.
 */
const REL_DF_MAX_WTA = 1e-3;
const COS_MIN = 1 - 1e-7;
const ABS_DENS_MAX = 1e-4;

/**
 * §4's extGrads bounds — the generator SEAM against tfjs autograd.
 *
 * MEASURED, then given headroom, like every other bound in this file. Worst
 * over all §4 rows (3 field cases × their kinds), Apple M-series / bun-webgpu,
 * 2026-08-22:
 *
 *   quantity          worst measured        bound here   headroom
 *   grid slice rel    2.1e-7                1e-4         ~470x
 *   mlp slice  rel    4.7e-7                1e-4         ~210x
 *   cos deficit       3.9e-14               1e-9         ~25000x
 *
 * Two orders tighter than `tools/train_wta_hashgrid_test.ts` holds the SAME
 * seam for the relational adversary (cos > 0.99999, rel < 3e-3), and that is
 * not a claim of a better kernel: the cotangent fed to the oracle here is the
 * GPU's OWN dL/dF, so the only f32-vs-f64 divergence left is one field
 * backward, where the adversary's number carries its whole predictor chain too.
 *
 * The cosine bound is the structural one. What it exists to catch — a dropped
 * segment, a wrong plane offset, a mirrored index — moves cosine by 1e-3 to 1;
 * a fully-dropped grid slice (defect §2c) takes it to exactly 0.
 */
const COS_MIN_EXT = 1 - 1e-9;
const REL_EXT_MAX = 1e-4;

/**
 * Smallest CPU-side max|dF| the comparison is allowed to be built on. Below
 * this the generator gradient is cancellation noise and "they agree" would be
 * meaningless — see the WARM_STEPS comment in compareKind.
 */
const DF_SCALE_MIN = 1e-6;

const KINDS: PixelGanKind[] = ["vec-field", "next-frame", "real-fake", "inpaint"];

/**
 * The guess settings every kind is compared under.
 *
 * `single` FIRST and always: it is the compatibility checkpoint — at that
 * setting every number below must equal what the pre-guesses code produced, and
 * it does (the whole file's output was diffed against HEAD).
 *
 * Two relaxation levels, deliberately far apart. ε=0.05 is nearly hard WTA, so
 * the winner carries the gradient and a selection disagreement between the two
 * backends shows up as a large gradient error. ε=0.22 at k=4 gives each of the
 * three losers 0.0733 against the winner's 0.78 — still winner-dominant (the
 * bound is 0.75), but now every guess receives real gradient, which is the
 * regime where a WRONG loser weight would otherwise hide inside tolerance.
 */
const GUESS_CASES: { label: string; guesses: GuessKind }[] = [
  { label: "", guesses: { tag: "single" } },
  { label: " k2ε.05", guesses: { tag: "wta", k: 2, relaxEps: 0.05 } },
  { label: " k4ε.22", guesses: { tag: "wta", k: 4, relaxEps: 0.22 } },
];

const SEED = 1234567;
const WARM_LR = 0.02;
/**
 * Warm-up length per kind. Uniform 60 everywhere EXCEPT real-fake, whose critic
 * separates a structured cloud from uniform noise almost immediately: by 60
 * steps its logit saturates near |7|, BCE falls to ~6e-4 and the surviving
 * gradient is ~1.6e-3, small enough that the f32 sigmoid's own rounding
 * dominates (measured at G=16: discLoss rel 5.8e-5 and discGrad cos 1-6.2e-8,
 * an order worse than every other kind, while the absolute gradient error
 * stayed ~1e-6). A saturated critic is a bad place to compare two
 * implementations, not evidence that they disagree — so real-fake stops short
 * of it. If you raise this, expect the tolerances below to stop holding for
 * reasons that have nothing to do with the kernels.
 */
const WARM_STEPS: Record<PixelGanKind, number> = {
  "vec-field": 60,
  "next-frame": 60,
  "real-fake": 10,
  inpaint: 60,
};

/**
 * MULTI-GUESS vec-field warms for half as long, for the same reason real-fake
 * warms for a sixth: past a point the critic is a degenerate place to compare
 * two implementations. Here the degeneracy is a DEAD CONV TRUNK, and more head
 * parameters reach it sooner. Measured at G=8 E=4 (live post-ReLU units out of
 * E·G² = 256, and the resulting max|dF| on the GPU):
 *
 *   warm   k=1              k=2              k=4
 *   10     47 live, 3.2e-4  132 live, 7.9e-4  29 live, 9.1e-4
 *   30      3 live, 2.2e-4   76 live, 2.8e-4   4 live, 3.9e-4
 *   60      3 live, 6.7e-5   78 live, 4.6e-4   1 live, 0.0
 *
 * At k=4/warm=60 exactly one conv unit is still alive and dF is IDENTICALLY
 * ZERO on both sides — they agree perfectly and the comparison proves nothing,
 * which is what DF_SCALE_MIN exists to make loud. Note the trunk is already
 * dying at k=1 (3 live units): this is a property of a 4-channel conv on a 64
 * cell grid under Adam, not of the guess fold. Single-guess keeps 60 so its rows
 * stay byte-comparable against pre-guesses HEAD.
 */
function warmSteps(kind: PixelGanKind, guesses: number): number {
  if (kind === "vec-field" && guesses > 1) return 30;
  return WARM_STEPS[kind];
}

/** CPU oracle's RNG (pixel_disc.ts:108) — a STREAM, state carried between draws. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The SHADER's `mulberry32` (pixel_disc_wgsl.ts:257-265), which is a DIFFERENT
 * function despite the docstring claiming it "matches CPU oracle":
 *
 *   CPU   t = (t + imul(t ^ t>>>7, 61|t)) ^ t;     <- trailing ^ t
 *   WGSL  t = t + (t ^ (t >> 7u)) * (61u | t);     <- no trailing ^ t
 *
 * So `mulberry32(s)` disagrees between the two backends for every s. It is also
 * used differently: the CPU mask draws x0 then y0 from ONE stream, the shader
 * draws from the two independent seeds `maskSeed` and `maskSeed+17`. Both only
 * ever needed to be "some random block" / "some random cloud", so this is not a
 * bug in either — but it does mean `maskSeed` is not a portable input, which is
 * why the harness verifies masks and clouds instead of trusting the seed.
 */
function wgslMulberry32(seed: number): number {
  const a = (seed + 0x6d2b79f5) >>> 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a) >>> 0;
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0;
  t = (t ^ (t >>> 14)) >>> 0;
  return t / 4294967296;
}

/** The shader's fake cloud (pixel_disc_wgsl.ts `fakeSplat`), in normalized coords. */
function gpuFakeCloud(B: number, maskSeed: number): Float64Array {
  const p = new Float64Array(B * 2);
  for (let s = 0; s < B; s++) {
    const seed = (maskSeed + Math.imul(s, 2654435761)) >>> 0;
    p[s * 2] = wgslMulberry32(seed);
    p[s * 2 + 1] = wgslMulberry32((seed + 1013904223) >>> 0);
  }
  return p;
}

/**
 * A maskSeed on which `makeInpaintMask` and `emitInpaintMaskPar` happen to
 * produce the SAME block, so inpaint's gradient can be compared at all. The
 * block is a G/2 × G/2 rect at one of (G/2+1)² origins, so a match turns up
 * within a few hundred seeds. Verified against the GPU's auxB afterwards.
 */
function pickInpaintSeed(G: number): number | null {
  const bw = Math.max(2, Math.floor(G * 0.5));
  const bh = bw;
  const rx = G - bw + 1;
  const ry = G - bh + 1;
  for (let s = 1; s < 20000; s++) {
    const r = mulberry32(s);
    const cx = Math.floor(r() * rx);
    const cy = Math.floor(r() * ry);
    const gx = Math.floor(wgslMulberry32(s) * rx);
    const gy = Math.floor(wgslMulberry32((s + 17) >>> 0) * ry);
    if (cx === gx && cy === gy) return s;
  }
  return null;
}

interface VecCmp {
  n: number;
  scale: number;
  worstRel: number;
  worstElem: number;
  cos: number;
  finite: boolean;
}

function compareVec(cpu: ArrayLike<number>, gpu: ArrayLike<number>, n: number): VecCmp {
  let scale = 0;
  for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(cpu[i]));
  const denom = scale > 0 ? scale : 1;
  let worstRel = 0;
  let worstElem = 0;
  let da = 0;
  let db = 0;
  let dot = 0;
  let finite = true;
  for (let i = 0; i < n; i++) {
    const a = cpu[i];
    const b = gpu[i];
    if (!Number.isFinite(b) || !Number.isFinite(a)) finite = false;
    const d = Math.abs(a - b);
    worstRel = Math.max(worstRel, d / denom);
    worstElem = Math.max(worstElem, d / (Math.abs(a) + 1e-6));
    da += a * a;
    db += b * b;
    dot += a * b;
  }
  const cos = da > 0 && db > 0 ? dot / Math.sqrt(da * db) : 0;
  return { n, scale, worstRel, worstElem, cos, finite };
}

function relScalar(cpu: number, gpu: number): number {
  return Math.abs(cpu - gpu) / Math.max(Math.abs(cpu), 1e-12);
}

const fmt = (x: number) => x.toExponential(2);

// ── §1 shared-constant preconditions ───────────────────────────────────────
{
  console.log("\n§1 shared constants (asserted, not assumed)");
  const d: PixelDiscDims = { kind: "vec-field", G: 8, E: 4, K: 8, hidden: 8, dt: 0.05 };
  const fieldDims = (inSize: number) => [
    { inSize, outSize: 16, activation: "selu" as const },
    { inSize: 16, outSize: 16, activation: "selu" as const },
    { inSize: 16, outSize: 2, activation: "tanh" as const },
  ];
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const src = pixelDiscShader(layout, { dims: d, batchCap: 64, fieldLane: "blend" });

  const m = /const SOFT_EPS2 : f32 = ([0-9eE.+-]+);/.exec(src);
  const want = PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS;
  ok(
    m !== null && Number(m[1]) === want,
    `shader SOFT_EPS2 = ${m ? m[1] : "<not found>"} ≡ PIXEL_DISC_SOFT_EPS² = ${want}`
  );

  // The density floor is a bare literal on both sides (pixel_disc.ts `densFloor`
  // in stepVecField, DENS_FLOOR in pixel_disc_wgsl.ts). Neither is exported, so
  // read them out of the sources rather than hardcoding a third copy here.
  const cpuSrc = await Bun.file(
    new URL("../src/core/gan/pixel_disc.ts", import.meta.url)
  ).text();
  const cpuFloor = /const densFloor = ([0-9eE.+-]+);/.exec(cpuSrc);
  const gpuFloor = /!\(dens_at\(c\) < ([0-9eE.+-]+)\)/.exec(src);
  if (!cpuFloor || !gpuFloor) {
    note(
      `density-floor literals moved (cpu=${cpuFloor ? cpuFloor[1] : "?"} gpu=${
        gpuFloor ? gpuFloor[1] : "?"
      }) — the vec-field activity check below still covers them behaviourally`
    );
  } else {
    ok(
      Number(cpuFloor[1]) === Number(gpuFloor[1]),
      `vec-field density floor ${cpuFloor[1]} (CPU) ≡ ${gpuFloor[1]} (WGSL)`
    );
  }

  for (const kind of KINDS) {
    for (const gc of GUESS_CASES) {
      if (kind === "real-fake" && gc.guesses.tag !== "single") continue;
      const dd: PixelDiscDims = { ...d, kind, guesses: gc.guesses };
      const P = pixelGuessCount(dd);
      const wl = pixelWeightLayout(dd);
      ok(
        wl.total === pixelDiscWeightCount(dd) &&
          wl.headStride === headFloatsPerGuess(dd) &&
          wl.guesses === P &&
          headFloats(dd) === P * headFloatsPerGuess(dd),
        `${kind}${gc.label} layout.total ≡ weightCount = ${pixelDiscWeightCount(dd)}, ` +
          `head = ${P} × ${headFloatsPerGuess(dd)} (guesses are a STRIDE, §3a)`
      );
    }
  }

  // §3h — per-guess init asymmetry. `initPixelDiscWeights` draws from ONE
  // sequential mulberry32 stream, so calling its per-guess filler once per guess
  // yields distinct heads for free. That is a correct outcome reached
  // IMPLICITLY, which is the fragile kind: an "optimization" that initialises
  // one head and broadcasts it produces identical residuals on every cell, every
  // comparison is then an exact tie, the first-argmin rule routes ALL of them to
  // guess 0, and the mixture is dead on arrival with a completely normal loss.
  for (const kind of KINDS) {
    if (kind === "real-fake") continue;
    const dd: PixelDiscDims = { ...d, kind, guesses: { tag: "wta", k: 4, relaxEps: 0.1 } };
    const w = initPixelDiscWeights(dd, SEED);
    const stride = headFloatsPerGuess(dd);
    let minPairDiff = Infinity;
    for (let a = 0; a < 4; a++) {
      for (let b = a + 1; b < 4; b++) {
        let same = 0;
        let diff = 0;
        for (let i = 0; i < stride; i++) {
          const x = w.head[a * stride + i];
          const y = w.head[b * stride + i];
          if (x === y) same++;
          diff = Math.max(diff, Math.abs(x - y));
        }
        // The bias tail is zero in every guess by design, so "identical floats"
        // is expected there and only the weight rows must differ.
        minPairDiff = Math.min(minPairDiff, diff);
        if (same === stride) minPairDiff = 0;
      }
    }
    ok(
      minPairDiff > 1e-6,
      `${kind} §3h init: every pair of the 4 guesses' head slices differs ` +
        `(closest pair max|Δ| ${fmt(minPairDiff)}) — no broadcast`
    );
  }

  // real-fake refuses guesses LOUDLY. A silently-clamped headCount would leave a
  // UI knob that appears to do something and does not — see validatePixelDims
  // for the three reasons (label BCE has one right answer; the generator pass
  // has no winner; no per-cell predicate to gate a §3g counter with).
  {
    const dd: PixelDiscDims = {
      ...d,
      kind: "real-fake",
      guesses: { tag: "wta", k: 2, relaxEps: 0.1 },
    };
    let threw = "";
    try {
      pixelDiscShader(layout, { dims: dd, batchCap: 64, fieldLane: "blend" });
    } catch (e) {
      threw = (e as Error).message;
    }
    ok(
      threw.includes("real-fake"),
      `real-fake + guesses>1 is REFUSED, not degraded: ${threw.slice(0, 90) || "<no throw>"}`
    );
  }

  // The two WTA scalars reach the shader from src/core/gan/wta.ts and are not
  // re-derived there. `ε=0` would emit as an abstract int and fail to compile —
  // the bug `f32lit` was hardened against — so check the literal, not the value.
  for (const gc of GUESS_CASES) {
    if (gc.guesses.tag === "single") continue;
    const dd: PixelDiscDims = { ...d, kind: "vec-field", guesses: gc.guesses };
    const src2 = pixelDiscShader(layout, { dims: dd, batchCap: 64, fieldLane: "blend" });
    const s = wtaScalars(gc.guesses);
    const m2 = /select\(([0-9eE.+-]+), ([0-9eE.+-]+), j == win\)/.exec(src2);
    ok(
      // f32lit prints ~10 significant digits, so compare at f32 precision, not
      // at f64 exactness — the shader constant IS an f32.
      m2 !== null &&
        Math.abs(Number(m2[1]) - s.loser) < 1e-7 * Math.max(s.loser, 1e-3) &&
        Math.abs(Number(m2[2]) - s.winner) < 1e-7 * s.winner,
      `${gc.label.trim()} shader emits select(loser=${m2 ? m2[1] : "?"}, ` +
        `winner=${m2 ? m2[2] : "?"}) ≡ wtaScalars = (${s.loser}, ${s.winner})`
    );
  }
}

// ── §1b encoding/family gates + scratch layout (no GPU) ────────────────────
//
// docs/PLAN_PIXEL_GENERATOR_ARCH.md §2/§3. Three of these would have caught the
// three §2 defects at codegen time; the fourth (the dEnc seed mode) is the §2a
// latent bug, which no shipped configuration can reach and which therefore has
// nothing but a codegen assertion standing under it.
{
  console.log("\n§1b hashgrid + family gates, and the scratch block they need");
  const HID = 16;
  const dims = (inSize: number): LayerDims[] => [
    { inSize, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: 2, activation: "tanh" },
  ];
  const mkLayout = (enc: Encoding, classes = 0) => {
    const dim = enc.kind === "hashgrid" ? enc.features : enc.kind === "fourier" ? 2 + 4 * enc.octaves : 2;
    // The one-hot route is exactly "head 1's layer 0 gets C more input rows",
    // which is the shape the pixel critic's field backward has no counterpart
    // for — so the refused layout has to be built with that width, not with a
    // classless head that would fail validateChain for an unrelated reason.
    const onehot = enc.kind === "raw" ? classes : 0;
    return layoutField("helmholtz", [dims(dim), dims(dim + onehot)], {
      encoding: enc,
      classes,
    });
  };
  const d: PixelDiscDims = { kind: "next-frame", G: 8, E: 4, K: 8, hidden: 8, dt: 0.05 };

  // (1) the gates, as DATA — classifyPixelDiscFusion must keep answering
  //     without throwing, which is the only reason the host can name the piece
  //     whose critic it turned off.
  const gates: { label: string; layout: FieldLayout; want: "ok" | "unsupported" }[] = [
    { label: "raw dual", layout: mkLayout({ kind: "raw" }), want: "ok" },
    { label: "fourier dual", layout: mkLayout({ kind: "fourier", octaves: 4 }), want: "ok" },
    {
      label: "hashgrid dual",
      layout: mkLayout({ kind: "hashgrid", gridSize: 8, features: 4 }),
      want: "ok",
    },
    {
      label: "family-planed hashgrid",
      layout: mkLayout({ kind: "hashgrid", gridSize: 8, features: 4, planes: 3 }, 3),
      want: "ok",
    },
    { label: "one-hot classes (raw)", layout: mkLayout({ kind: "raw" }, 3), want: "unsupported" },
  ];
  for (const g of gates) {
    const f = classifyPixelDiscFusion(g.layout);
    ok(
      f.tag === g.want,
      `gate: ${g.label} → ${f.tag}${f.tag === "unsupported" ? ` (${f.reason.slice(0, 60)}…)` : ""}`
    );
  }

  // (2) SCRATCH. raw/fourier must still COLLAPSE the dEnc block — that is what
  //     keeps every already-shipped pixel shader byte-identical — and hashgrid
  //     must reserve exactly encDim floats for it.
  for (const [label, enc, encStore] of [
    ["raw", { kind: "raw" } as Encoding, 0],
    ["fourier", { kind: "fourier", octaves: 4 } as Encoding, 18],
  ] as const) {
    const pl = pixelPartLayout(mkLayout(enc));
    ok(
      pl.oDEnc === pl.oField && pl.oEnc === 8 && pl.oField === 8 + encStore,
      `${label}: dEnc block EMPTY (oEnc ${pl.oEnc}, oDEnc ${pl.oDEnc}, oField ${pl.oField}) — stride unchanged`
    );
  }
  {
    const F = 4;
    const pl = pixelPartLayout(mkLayout({ kind: "hashgrid", gridSize: 8, features: F }));
    ok(
      pl.oDEnc === 8 + F && pl.oField - pl.oDEnc === F && pl.oCls === pl.stride,
      `hashgrid: dEnc block is ${pl.oField - pl.oDEnc} = features floats, no cls slot (stride ${pl.stride})`
    );
    const plp = pixelPartLayout(
      mkLayout({ kind: "hashgrid", gridSize: 8, features: F, planes: 3 }, 3)
    );
    ok(
      plp.stride - plp.oCls === 1,
      `family-planed hashgrid: one f32 family label, APPENDED (oCls ${plp.oCls} = stride−1)`
    );
  }

  // (3) the emitted backward. Only the hashgrid shader may carry a dEnc scratch
  //     block, and its grid segment must actually be scattered into extGrads —
  //     `fieldGrad` skipping `role: "grid"` is the defect that runs, looks
  //     healthy, and never trains the table (§2c).
  for (const [label, enc] of [
    ["raw", { kind: "raw" } as Encoding],
    ["fourier", { kind: "fourier", octaves: 4 } as Encoding],
  ] as const) {
    const src = pixelDiscShader(mkLayout(enc), { dims: d, batchCap: 64, fieldLane: "blend" });
    ok(!/dEncBase/.test(src), `${label}: emitted shader has no dEnc scratch block`);
  }
  {
    const hgLayout = mkLayout({ kind: "hashgrid", gridSize: 8, features: 4 });
    const src = pixelDiscShader(hgLayout, { dims: d, batchCap: 64, fieldLane: "blend" });
    const gridSeg = hgLayout.segments.find((s) => s.role === "grid")!;
    ok(/dEncBase/.test(src), "hashgrid: emitted shader stores dL/dEnc");
    ok(
      gridSeg.floatOffset === 0 &&
        src.includes(`let cell = (t - ${gridSeg.floatOffset}u) / 4u;`),
      `hashgrid: fieldGrad scatters the grid segment [0, ${gridSeg.floatLength}) — not skipped`
    );

    // §2a. The dEnc block is SHARED by both field heads: one seeds it (`=`) and
    // the rest accumulate (`+=`). With one lane emitted only ONE head runs, so
    // the SEEDING head must be the lane — keying it on the head index makes a
    // fieldLane:1 game accumulate into scratch nobody wrote this step. No
    // shipped piece passes fieldLane, which is exactly why this needs an
    // assertion rather than a runtime failure.
    const seeds = (lane: "blend" | 0 | 1, h: number) => {
      const s = pixelDiscShader(hgLayout, { dims: d, batchCap: 64, fieldLane: lane });
      const body = s.split(`fn bwd_head_${h}(`)[1] ?? "";
      const upTo = body.slice(0, body.indexOf("\nfn "));
      return /scratch\[dEncBase \+ i\] = dEnc\[i\];/.test(upTo);
    };
    ok(
      seeds("blend", 0) && !seeds("blend", 1),
      "hashgrid blend lane: head 0 SEEDS dEnc, head 1 accumulates"
    );
    ok(seeds(0, 0), "hashgrid fieldLane 0: the emitted head SEEDS dEnc");
    ok(
      seeds(1, 1),
      "hashgrid fieldLane 1: the emitted head SEEDS dEnc (head INDEX would not)"
    );
  }

  // (4) vec-field on a planed field is REFUSED, not silently pinned to plane 0.
  //     Its target is F at cell centres and a cell centre has no family; a
  //     `cls = 0` default would fit the critic to family 0's field and look
  //     completely healthy.
  {
    let threw = "";
    try {
      pixelDiscShader(
        mkLayout({ kind: "hashgrid", gridSize: 8, features: 4, planes: 3 }, 3),
        { dims: { ...d, kind: "vec-field" }, batchCap: 64, fieldLane: "blend" }
      );
    } catch (e) {
      threw = (e as Error).message;
    }
    ok(
      threw.includes("cell centre has no family"),
      `vec-field + family-planed field is REFUSED: ${threw.slice(0, 80) || "<no throw>"}`
    );
  }
}

// ── GPU harness ────────────────────────────────────────────────────────────
let device: GPUDevice | null = null;
let PixelDiscTrainerCtor: any = null;
try {
  const { setupGlobals } = await import("bun-webgpu");
  setupGlobals();
  (globalThis as any).GPUBufferUsage ??= {
    MAP_READ: 1,
    MAP_WRITE: 2,
    COPY_SRC: 4,
    COPY_DST: 8,
    UNIFORM: 64,
    STORAGE: 128,
  };
  (globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };
  (globalThis as any).GPUShaderStage ??= { COMPUTE: 4 };
  const adapter = await (navigator as any).gpu.requestAdapter();
  if (adapter) {
    device = await adapter.requestDevice();
    ({ PixelDiscTrainer: PixelDiscTrainerCtor } = await import(
      "../src/render/webgpu/pixel_disc_train"
    ));
  }
} catch (e) {
  note(`WebGPU unavailable: ${(e as Error).message?.slice(0, 160)}`);
}

/**
 * Read the first `n` f32s of a trainer-owned buffer, clamped to what it
 * actually holds. The clamp is not defensive noise: the stats region of
 * `metaBuf` is being resized (plan §3f/§4b wants it to carry per-guess win
 * counters), so this test must not encode its own copy of that size — it needs
 * only `[0, 3·nW)` for the gradients plus whatever stats slots exist.
 */
async function readFloats(dev: GPUDevice, buf: GPUBuffer, n: number): Promise<Float32Array> {
  const cap = typeof buf.size === "number" && buf.size > 0 ? Math.floor(buf.size / 4) : n;
  const want = Math.min(n, cap);
  const staging = dev.createBuffer({
    size: want * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, want * 4);
  dev.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

/**
 * The generator-reward seam, handed everything the field backward consumed.
 *
 * `dSig` is the GPU's OWN dL/dF (scratch oDF), not the CPU's: the point of a
 * seam check is the chain from dL/dF into the field's packed weights, and
 * feeding it the CPU's cotangent would fold the critic's f32/f64 gap into a
 * number that is supposed to isolate the field backward. The two are compared
 * against each other one assertion earlier ("dF"), so nothing goes unchecked.
 */
interface FieldSeam {
  (a: {
    label: string;
    /** normalized sample positions, sample order, [B·2] */
    pos: Float64Array;
    /** F(u) the shader evaluated and splatted, [B·2] */
    gpuF: Float64Array;
    /** dL_gen/dF per sample, [B·2] */
    dSig: Float64Array;
    /** the whole packed extGrads buffer this step wrote */
    ext: Float32Array;
    alpha: number;
  }): Promise<void> | void;
}

/**
 * One kind, one config: run the fused GPU step twice (disc-only, then with the
 * generator pass), reconstruct the exact same inputs on the CPU, and compare.
 */
async function compareKind(
  dev: GPUDevice,
  layout: FieldLayout,
  fieldWeightsBuf: GPUBuffer,
  dims: PixelDiscDims,
  B: number,
  posBuf: GPUBuffer,
  velBuf: GPUBuffer,
  extra: {
    fieldTag?: string;
    seam?: FieldSeam;
    /** Mirrors the adversary's knob; absent ≡ the shipped "blend". */
    fieldLane?: "blend" | 0 | 1;
  } = {}
): Promise<void> {
  const kind = dims.kind;
  const nCell = dims.G * dims.G;
  const nW = pixelDiscWeightCount(dims);
  const pl = pixelPartLayout(layout);
  const partStride = pl.stride;
  const P = pixelGuessCount(dims);
  // One label for every line of this comparison, so a failure names the guess
  // setting it happened under and not just the kind.
  const kl = `${extra.fieldTag ?? ""}${kind}${
    P === 1 ? "" : ` k${P}ε${(dims.guesses as any).relaxEps}`
  }`;
  const winBase = pixelCritWinBase(layout, B, dims);

  let maskSeed = 4242;
  if (kind === "inpaint") {
    const s = pickInpaintSeed(dims.G);
    if (s === null) {
      ok(false, `${kl} no maskSeed makes CPU/WGSL masks agree — cannot compare`);
      return;
    }
    maskSeed = s;
    note(
      `${kl} maskSeed=${maskSeed} searched, not chosen: makeInpaintMask and ` +
        `emitInpaintMaskPar use different RNG streams (see wgslMulberry32)`
    );
  }

  dev.pushErrorScope("validation");
  const trainer = new PixelDiscTrainerCtor(dev, layout, {
    fieldWeightsBuffer: fieldWeightsBuf,
    dims,
    batchCap: B,
    seed: SEED,
    ...(extra.fieldLane === undefined ? {} : { fieldLane: extra.fieldLane }),
  });
  const shaderErr = await dev.popErrorScope();
  if (shaderErr) {
    ok(false, `${kl} shader: ${String(shaderErr.message).slice(0, 160)}`);
    trainer.destroy();
    return;
  }
  trainer.setParticleBuffers(posBuf, velBuf, B);

  const priv = trainer as unknown as {
    metaBuf: GPUBuffer;
    scratchBuf: GPUBuffer;
    densPack: GPUBuffer;
  };

  // Same weights on both sides — read back rather than assume the ctor's upload.
  const critW0 = await trainer.readCriticWeights();
  const expectW = packPixelDiscWeights(initPixelDiscWeights(dims, SEED), dims);
  let wSame = critW0.length === expectW.length;
  for (let i = 0; wSame && i < expectW.length; i++) wSame = critW0[i] === expectW[i];
  ok(wSame, `${kl} GPU critW ≡ initPixelDiscWeights(dims, ${SEED}) (${nW} floats)`);

  /**
   * Warm the critic on the GPU before comparing, then adopt ITS weights as the
   * CPU oracle's. Not cosmetic — a freshly initialised critic is a DEGENERATE
   * comparison point for next-frame:
   *
   *   at init, `pred` ∈ [-0.39, -0.32] while D₁ ∈ [0, 4.6] over every cell, so
   *   `diff = pred - D₁ < 0` everywhere; the soft-L1 derivative `-diff/r`
   *   saturates to +1 uniformly, `gD` becomes a constant field, and the
   *   bilinear splat VJP's four taps cancel exactly. dF is then identically
   *   zero (measured max|dF| = 1.9e-10) and "cpu ≈ gpu" would be comparing
   *   f32 cancellation noise against f64 cancellation noise. DF_SCALE_MIN
   *   below makes that failure loud instead of green.
   *
   * Adopting the GPU's post-Adam weights, rather than replaying Adam on the
   * CPU, is what keeps the two sides bit-identical without the CPU oracle
   * needing an optimizer it does not have (it applies plain SGD).
   */
  const warm = warmSteps(kind, P);
  for (let t = 0; t < warm; t++) {
    const encW = dev.createCommandEncoder();
    trainer.encodeStep(encW, {
      b: B,
      alpha: 0.5,
      lr: WARM_LR,
      genWeight: 0,
      applyDisc: true,
      width: 1,
      height: 1,
      maskSeed,
    });
    dev.queue.submit([encW.finish()]);
  }
  await dev.queue.onSubmittedWorkDone();
  const critW = await trainer.readCriticWeights();
  let wFinite = true;
  let moved = 0;
  for (let i = 0; i < nW; i++) {
    if (!Number.isFinite(critW[i])) wFinite = false;
    moved = Math.max(moved, Math.abs(critW[i] - critW0[i]));
  }
  ok(
    wFinite && moved > 0,
    `${kl} warmed ${warm} Adam steps @ lr=${WARM_LR} (max |Δw| ${fmt(moved)}), finite`
  );
  const w = unpackPixelDiscWeights(critW, dims);

  const stepOpts = {
    b: B,
    alpha: 0.5,
    lr: 1e-3,
    width: 1,
    height: 1,
    maskSeed,
    applyDisc: false, // gradients, not post-update weights (CPU=SGD, GPU=Adam)
  };

  // ── pass A: disc only ────────────────────────────────────────────────────
  dev.pushErrorScope("validation");
  const encA = dev.createCommandEncoder();
  trainer.encodeStep(encA, { ...stepOpts, genWeight: 0 });
  dev.queue.submit([encA.finish()]);
  await dev.queue.onSubmittedWorkDone();
  const errA = await dev.popErrorScope();
  if (errA) {
    ok(false, `${kl} disc pass: ${String(errA.message).slice(0, 160)}`);
    trainer.destroy();
    return;
  }

  const scratchA = await readFloats(dev, priv.scratchBuf, B * partStride);
  const densA = await readFloats(dev, priv.densPack, 4 * nCell);
  const metaA = await readFloats(dev, priv.metaBuf, 3 * nW + 16);
  if (metaA.length < 3 * nW + 2) {
    ok(false, `${kl} metaBuf holds ${metaA.length} f32 — no room for discLoss/genLoss stats`);
    trainer.destroy();
    return;
  }

  // Positions and F in SAMPLE order, straight out of the shader's own scratch.
  const pos = new Float64Array(B * 2);
  const forces = new Float64Array(B * 2);
  for (let s = 0; s < B; s++) {
    pos[s * 2] = scratchA[s * partStride + pl.oPos];
    pos[s * 2 + 1] = scratchA[s * partStride + pl.oPos + 1];
    forces[s * 2] = scratchA[s * partStride + pl.oF];
    forces[s * 2 + 1] = scratchA[s * partStride + pl.oF + 1];
  }

  const auxA = densA.subarray(2 * nCell, 3 * nCell);
  const auxB = densA.subarray(3 * nCell, 4 * nCell);

  let forceGrid: Float64Array | undefined;
  if (kind === "vec-field") {
    forceGrid = new Float64Array(nCell * 2);
    for (let c = 0; c < nCell; c++) {
      forceGrid[c * 2] = auxA[c];
      forceGrid[c * 2 + 1] = auxB[c];
    }
  }

  let fakePos: Float64Array | undefined;
  if (kind === "real-fake") {
    fakePos = gpuFakeCloud(B, maskSeed);
    // Verify the injected cloud IS the shader's cloud before believing any of
    // real-fake's numbers — a mismatch here would make the comparison a lie.
    const dFakeCpu = softSplatDensity(fakePos, B, dims.G);
    let worst = 0;
    for (let c = 0; c < nCell; c++) worst = Math.max(worst, Math.abs(dFakeCpu[c] - auxA[c]));
    ok(
      worst < ABS_DENS_MAX,
      `${kl} injected fakePos reproduces GPU fake cloud (worst |ΔD| ${fmt(worst)})`
    );
  }

  if (kind === "inpaint") {
    const cpuMask = makeInpaintMask(dims.G, maskSeed);
    let bad = 0;
    let nMask = 0;
    for (let c = 0; c < nCell; c++) {
      const g = auxB[c] > 0.5 ? 1 : 0;
      if (g !== cpuMask[c]) bad++;
      nMask += g;
    }
    ok(
      bad === 0,
      `${kl} CPU mask ≡ GPU mask at maskSeed=${maskSeed} (${nMask}/${nCell} cells, ${bad} disagree)`
    );
  }

  const rDisc = pixelDiscStep(pos, forces, w, dims, {
    applyDisc: false,
    genSign: 0,
    lr: 0,
    forceGrid,
    fakePos,
    maskSeed,
  });

  // Density diagnostic: characterises the i32/DENS_SCALE truncation floor that
  // sets the gradient tolerance. next-frame's `dens` slot holds D1, not D0.
  {
    const cpuD =
      kind === "next-frame"
        ? (() => {
            const p2 = new Float64Array(B * 2);
            for (let i = 0; i < B; i++) {
              p2[i * 2] = pos[i * 2] + dims.dt * forces[i * 2];
              p2[i * 2 + 1] = pos[i * 2 + 1] + dims.dt * forces[i * 2 + 1];
            }
            return softSplatDensity(p2, B, dims.G);
          })()
        : softSplatDensity(pos, B, dims.G);
    let worst = 0;
    for (let c = 0; c < nCell; c++) worst = Math.max(worst, Math.abs(cpuD[c] - densA[c]));
    ok(
      worst < ABS_DENS_MAX,
      `${kl} density: worst |ΔD| ${fmt(worst)} (i32 splat @ DENS_SCALE=1e6 truncates)`
    );
    if (kind === "vec-field") {
      // If the two sides ever straddle the floor on some cell the gradients
      // diverge structurally, not numerically — say so before blaming f32.
      let act = 0;
      let disagree = 0;
      let nearest = Infinity;
      for (let c = 0; c < nCell; c++) {
        const a = !(cpuD[c] < 1e-3);
        const b = !(densA[c] < 1e-3);
        if (a) act++;
        if (a !== b) disagree++;
        if (cpuD[c] > 0) nearest = Math.min(nearest, Math.abs(cpuD[c] - 1e-3));
      }
      ok(
        disagree === 0,
        `${kl} activity: ${act}/${nCell} active, ${disagree} cells straddle the floor` +
          ` (closest non-empty cell is ${fmt(nearest)} from it)`
      );
    }
  }

  const gpuDiscLoss = metaA[3 * nW];
  ok(
    relScalar(rDisc.discLoss, gpuDiscLoss) < REL_LOSS_MAX,
    `${kl} discLoss cpu ${rDisc.discLoss.toFixed(6)} gpu ${gpuDiscLoss.toFixed(6)} ` +
      `rel ${fmt(relScalar(rDisc.discLoss, gpuDiscLoss))}`
  );

  const gc = compareVec(rDisc.discGradPacked, metaA, nW);
  ok(
    gc.finite && gc.worstRel < REL_GRAD_MAX && gc.cos > COS_MIN,
    `${kl} discGrad (${nW} floats, worst rel ${fmt(gc.worstRel)} of scale ${fmt(gc.scale)}, ` +
      `elementwise ${fmt(gc.worstElem)}, cos 1-${fmt(1 - gc.cos)})`
  );

  // ── the SELECTION, not just the sum ──────────────────────────────────────
  // A winner disagreement swaps that cell's weights between (1−ε) and ε/(k−1);
  // on a near-tie the summed gradient barely moves, so the sum alone cannot see
  // it. Compared only where the CPU says the cell was ACTIVE (winIdx >= 0) —
  // §3g: an inactive cell has a mathematical argmin that nothing may credit.
  const cpuWin = rDisc.winIdx;
  if (cpuWin && winBase !== null) {
    const winF = await readFloats(dev, priv.scratchBuf, winBase + nCell);
    let bad = 0;
    let counted = 0;
    const hist = new Array<number>(P).fill(0);
    for (let c = 0; c < nCell; c++) {
      if (cpuWin[c] < 0) continue;
      counted++;
      hist[cpuWin[c]]++;
      if (Math.round(winF[winBase + c]) !== cpuWin[c]) bad++;
    }
    ok(
      bad === 0 && counted > 0,
      `${kl} winner: ${counted} active cells, ${bad} disagree; ` +
        `CPU histogram [${hist.join(", ")}]`
    );

    // The GPU's own §3f counters. Equality with the CPU histogram gates BOTH the
    // counting and its §3g gating in one number: if inactive cells were being
    // credited, the GPU's total would be nCell (${nCell}) rather than `counted`.
    const nWin = pixelWinCounters(dims);
    const gpuHist: number[] = [];
    for (let j = 0; j < nWin; j++) gpuHist.push(metaA[3 * nW + PIXEL_STATS_WIN_BASE + j]);
    let sum = 0;
    let same = nWin === P;
    for (let j = 0; j < nWin; j++) {
      sum += gpuHist[j];
      if (gpuHist[j] !== hist[j]) same = false;
    }
    ok(
      same && sum === counted,
      `${kl} §3f win counters [${gpuHist.join(", ")}] ≡ CPU histogram, ` +
        `Σ=${sum} = active cells (not ${nCell})`
    );
  }

  // ── pass B: disc + generator ─────────────────────────────────────────────
  // Idempotent w.r.t. pass A: applyDisc=false leaves critW/Adam untouched and
  // cursor wraps back to 0 because partCount == b.
  dev.pushErrorScope("validation");
  const encB = dev.createCommandEncoder();
  trainer.encodeStep(encB, { ...stepOpts, genWeight: 1 });
  dev.queue.submit([encB.finish()]);
  await dev.queue.onSubmittedWorkDone();
  const errB = await dev.popErrorScope();
  if (errB) {
    ok(false, `${kl} gen pass: ${String(errB.message).slice(0, 160)}`);
    trainer.destroy();
    return;
  }
  const scratchB = await readFloats(dev, priv.scratchBuf, B * partStride);
  const metaB = await readFloats(dev, priv.metaBuf, 3 * nW + 16);

  // encodeStep maps genWeight → genSign = -|genWeight| (pixel_disc_train.ts:299).
  const rGen = pixelDiscStep(pos, forces, w, dims, {
    applyDisc: false,
    genSign: -1,
    lr: 0,
    forceGrid,
    fakePos,
    maskSeed,
  });

  const gpuGenLoss = metaB[3 * nW + 1];
  ok(
    relScalar(rGen.genLoss, gpuGenLoss) < REL_LOSS_MAX,
    `${kl} genLoss  cpu ${rGen.genLoss.toFixed(6)} gpu ${gpuGenLoss.toFixed(6)} ` +
      `rel ${fmt(relScalar(rGen.genLoss, gpuGenLoss))}`
  );

  const gpuDF = new Float64Array(B * 2);
  for (let s = 0; s < B; s++) {
    gpuDF[s * 2] = scratchB[s * partStride + pl.oDF];
    gpuDF[s * 2 + 1] = scratchB[s * partStride + pl.oDF + 1];
  }
  const dc = compareVec(rGen.dF, gpuDF, B * 2);
  // The multi-guess bound is a NAMED separate floor, not a widened one — see
  // REL_DF_MAX_WTA for the residual-knee measurement that sets it.
  const dfBound = P === 1 ? REL_DF_MAX : REL_DF_MAX_WTA;
  ok(
    dc.finite && dc.scale > DF_SCALE_MIN && dc.worstRel < dfBound && dc.cos > COS_MIN,
    `${kl} dF (${B * 2} floats, worst rel ${fmt(dc.worstRel)} of scale ${fmt(dc.scale)}, ` +
      `elementwise ${fmt(dc.worstElem)}, cos 1-${fmt(1 - dc.cos)})`
  );

  // The seam the WHOLE critic exists to drive: dL/dF → the field's packed
  // weights, extGrads. Nothing above touches it — every §2/§3 assertion would
  // stay green with fieldGrad emitting literal zeros for a whole segment, which
  // is exactly how the grid table was silently dropped.
  if (extra.seam) {
    const ext1 = await trainer.readExtGrads();

    // extGrads must be a function of THIS step, not of history. Nothing about
    // the second step differs — applyDisc is false so no weight moved, maskSeed
    // is fixed, and `cursor` wrapped back to 0 — so the two must be BIT-equal.
    //
    // This is the only assertion that can see a per-site scratch block being
    // accumulated into instead of seeded (§2a): the block starts life zeroed, so
    // a single-step test cannot tell `=` from `+=`, and the second step is
    // where a mis-seeded dEnc doubles.
    const encC = dev.createCommandEncoder();
    trainer.encodeStep(encC, { ...stepOpts, genWeight: 1 });
    dev.queue.submit([encC.finish()]);
    await dev.queue.onSubmittedWorkDone();
    const ext2 = await trainer.readExtGrads();
    let differing = 0;
    let worst = 0;
    for (let t = 0; t < ext1.length; t++) {
      if (ext1[t] !== ext2[t]) differing++;
      worst = Math.max(worst, Math.abs(ext1[t] - ext2[t]));
    }
    ok(
      differing === 0,
      `${kl} extGrads is a function of the STEP, not of history: an identical ` +
        `second gen step reproduces it bit for bit (${differing} floats differ, ` +
        `max |Δ| ${fmt(worst)})`
    );

    await extra.seam({
      label: kl,
      pos,
      gpuF: forces,
      dSig: gpuDF,
      ext: ext1,
      alpha: stepOpts.alpha,
    });
  }

  trainer.destroy();
}

async function runConfig(dev: GPUDevice, label: string, base: Omit<PixelDiscDims, "kind">, B: number) {
  console.log(`\n${label}  G=${base.G} E=${base.E} K=${base.K} hidden=${base.hidden} B=${B}`);
  const fieldDims = (inSize: number) => [
    { inSize, outSize: 16, activation: "selu" as const },
    { inSize: 16, outSize: 16, activation: "selu" as const },
    { inSize: 16, outSize: 2, activation: "tanh" as const },
  ];
  const layout = layoutField("helmholtz", [fieldDims(2), fieldDims(2)]);
  const fw = new Float32Array(layout.totalFloats);
  const rnd = mulberry32(20260822);
  for (let i = 0; i < fw.length; i++) fw[i] = (rnd() - 0.5) * 0.6;
  const fwBuf = dev.createBuffer({
    size: fw.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  dev.queue.writeBuffer(fwBuf, 0, fw);

  // width = height = 1 ⇒ sampleAndSplat's uk = partPos, exactly.
  //
  // The cloud fills [0.05,0.72]² and leaves the rest of the grid empty, which
  // is deliberate on both counts. Empty cells exercise vec-field's density
  // floor (the one place the two implementations branch on a value that the
  // fixed-point splat perturbs). Occupied-vs-empty BOUNDARIES are what make
  // next-frame's dF non-degenerate: its residual is a soft L1, so `diff/r`
  // saturates to ±1 and the density VJP's four bilinear taps cancel almost
  // exactly for any particle whose neighbourhood is uniformly on one side.
  // A cloud packed into two compact blobs measured max|dF| ≈ 3e-9 — pure
  // cancellation noise, and a comparison of noise against noise proves nothing.
  const pos = new Float32Array(B * 2);
  for (let i = 0; i < B; i++) {
    pos[i * 2] = 0.05 + 0.67 * rnd();
    pos[i * 2 + 1] = 0.05 + 0.67 * rnd();
  }
  const posBuf = dev.createBuffer({
    size: pos.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(posBuf, 0, pos);
  const velBuf = dev.createBuffer({
    size: pos.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  for (const kind of KINDS) {
    for (const g of GUESS_CASES) {
      // real-fake takes `single` only — §1 asserts the refusal; running it here
      // would just re-assert the throw at 20x the cost.
      if (kind === "real-fake" && g.guesses.tag !== "single") continue;
      await compareKind(
        dev,
        layout,
        fwBuf,
        { ...base, kind, guesses: g.guesses },
        B,
        posBuf,
        velBuf
      );
    }
  }

  fwBuf.destroy();
  posBuf.destroy();
  velBuf.destroy();
}

if (!device) {
  console.log("\n  skip  no WebGPU adapter — the CPU↔GPU gate DID NOT RUN");
  console.log(failures === 0 ? "\nSKIPPED (§1 only)" : `\n${failures} FAILURE(S)`);
  process.exit(failures === 0 ? 0 : 1);
}

await runConfig(device, "§2 small config", { G: 8, E: 4, K: 8, hidden: 8, dt: 0.05 }, 64);
await runConfig(
  device,
  "§3 shipped gallery config (PIXEL_DISC_DEFAULTS, src/main.ts:2403)",
  {
    G: PIXEL_DISC_DEFAULTS.G,
    E: PIXEL_DISC_DEFAULTS.E,
    K: PIXEL_DISC_DEFAULTS.K,
    hidden: PIXEL_DISC_DEFAULTS.hidden,
    dt: PIXEL_DISC_DEFAULTS.dt,
  },
  128
);

// ═══════════════════════════════════════════════════════════════════════════
// §4 HASHGRID-ENCODED generator fields (docs/PLAN_PIXEL_GENERATOR_ARCH.md §2/§3)
// ═══════════════════════════════════════════════════════════════════════════
//
// §2/§3 above compare the CRITIC. They are encoding-blind by construction: the
// oracle is handed the positions and the forces the shader itself computed, so
// every one of them stays green no matter what the field backward does — which
// is precisely why the dropped grid gradient could not have been caught there.
//
// So this section adds the only comparison that can see it: the GENERATOR SEAM,
// dL/dF → extGrads, against LIVE tfjs autograd on the real `HelmholtzField`
// hashgrid path (the same oracle tools/train_wta_hashgrid_test.ts uses for the
// relational adversary — the AD IR cannot express a hashgrid, its gather
// indices being data-dependent). The GRID slice is asserted SEPARATELY from the
// MLP slice, with its support, because:
//
//   - `extGrads == 0` for the whole grid table is what `if (seg.role ===
//     "grid") continue;` produced. It runs, every number is finite, the loss
//     falls, and the encoding's 4096 trainable floats simply never move.
//   - an aggregate cosine over all totalFloats is dominated by whichever slice
//     is larger, so a fully-zero grid slice can hide inside it.
//
// The FORWARD is asserted first (tfjs F vs the shader's own stored F): a
// gradient mismatch is then attributable to the backward rather than to the
// encoder, and the hand-written planed-grid mirror below cannot silently
// disagree with `encodeSite` about which plane it read.
{
  const tf = await import("@tensorflow/tfjs");
  await tf.setBackend("cpu");
  await tf.ready();
  const { HelmholtzField } = await import("../src/core/field/helmholtz");

  const HID = 16;
  const headDims = (inSize: number): LayerDims[] => [
    { inSize, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: HID, activation: "selu" },
    { inSize: HID, outSize: 2, activation: "tanh" },
  ];

  /** The family derivation, on the host. MUST match the WGSL pcg + CLASS_SALT. */
  const pcg = (v: number): number => {
    const s = Math.imul(v >>> 0, 747796405) + 2891336453;
    const t = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737);
    return ((t >>> 22) ^ t) >>> 0;
  };

  /** compareVec plus the SUPPORT sets — which floats each side made nonzero. */
  const supportOf = (ref: ArrayLike<number>, got: ArrayLike<number>) => {
    let scale = 0;
    for (let i = 0; i < ref.length; i++) scale = Math.max(scale, Math.abs(ref[i]));
    let nzRef = 0;
    let nzGot = 0;
    let mismatch = 0;
    for (let i = 0; i < ref.length; i++) {
      const r = Math.abs(ref[i]) > 1e-9 * scale;
      const g = Math.abs(got[i]) > 1e-9 * scale;
      if (r) nzRef++;
      if (g) nzGot++;
      if (r !== g) mismatch++;
    }
    return { nzRef, nzGot, mismatch };
  };

  interface HgCase {
    label: string;
    gridSize: number;
    features: number;
    classes: number;
    kinds: PixelGanKind[];
    /** Extra `(kind, lane)` runs on top of `kinds` × blend. */
    lanes?: { kind: PixelGanKind; lane: 0 | 1 }[];
    G: number;
    B: number;
  }
  const HG_CASES: HgCase[] = [
    {
      // Small grid on purpose: at 8² nearly every cell is claimed by some
      // particle, so "the grid slice is nonzero" is a statement about the
      // whole table rather than about a handful of lucky cells.
      label: "hashgrid 8²×4 · ",
      gridSize: 8,
      features: 4,
      classes: 0,
      kinds: KINDS,
      // THE §2a HAZARD, at runtime. With one lane emitted only ONE field head's
      // backward runs, and that head must SEED the shared dL/dEnc block. Lane 1
      // is the failing case if the emitter keeps a "head 0 seeds" assumption:
      // it would `+=` into scratch nobody wrote this step, so the GRID slice
      // (and only the grid slice) drifts. Nothing shipped passes fieldLane —
      // which is exactly why it needs a test rather than a bug report.
      //
      // NOT next-frame, and the reason is measured rather than aesthetic: at
      // lane 1 its 60-step warmed critic lands ON the soft-L1 knee this file
      // already documents (REL_DF_MAX_WTA — `d(diff/r)/d(diff) = SOFT_EPS²/r³`),
      // where dF rel goes to 2.1e-3 while cosine stays at 1-8e-8 and the SEAM
      // agrees with tfjs to 9.4e-8. Same run at warm 20/40/55 measures 2.7e-7,
      // i.e. it is that one warmed state, not the lane. vec-field additionally
      // drives `fillForceGrid` through the lane, which next-frame does not have.
      lanes: [
        { kind: "vec-field", lane: 0 },
        { kind: "vec-field", lane: 1 },
        { kind: "inpaint", lane: 0 },
        { kind: "inpaint", lane: 1 },
      ],
      G: 8,
      B: 64,
    },
    {
      // ARCH.dualHashgrid's shipped geometry.
      label: "hashgrid 32²×4 (ARCH.dualHashgrid) · ",
      gridSize: 32,
      features: 4,
      classes: 0,
      kinds: ["next-frame"],
      G: 16,
      B: 128,
    },
    {
      // ARCH.familyHashgrid's route: the label picks a feature PLANE. vec-field
      // is absent because it is refused (§1b) — its target is F at cell centres
      // and a cell centre has no family.
      label: "family-planed 8²×4×3 (ARCH.familyHashgrid) · ",
      gridSize: 8,
      features: 4,
      classes: 3,
      kinds: ["next-frame", "inpaint", "real-fake"],
      G: 8,
      B: 96,
    },
  ];

  for (const hc of HG_CASES) {
    const planes = Math.max(hc.classes, 1);
    const enc: Encoding = {
      kind: "hashgrid",
      gridSize: hc.gridSize,
      features: hc.features,
      ...(planes > 1 ? { planes } : {}),
    };
    const layout = layoutField(
      "helmholtz",
      [headDims(hc.features), headDims(hc.features)],
      { encoding: enc, classes: hc.classes }
    );
    const gridSeg = layout.segments.find((s) => s.role === "grid")!;
    console.log(
      `\n§4 ${hc.label}B=${hc.B} G=${hc.G} (${layout.totalFloats} field floats, ` +
        `grid [0,${gridSeg.floatLength}))`
    );

    // Field weights. The grid gets its own amplitude: a table of zeros is a
    // degenerate field whose backward is trivially "finite" and whose forward
    // is constant.
    const rnd = mulberry32(20260822 + hc.gridSize);
    const fw = new Float32Array(layout.totalFloats);
    for (const seg of layout.segments) {
      const amp = seg.role === "grid" ? 0.7 : 0.6;
      for (let x = 0; x < seg.floatLength; x++) {
        fw[seg.floatOffset + x] = Math.fround((rnd() - 0.5) * amp);
      }
    }
    const fwBuf = device.createBuffer({
      size: fw.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(fwBuf, 0, fw);

    const pos = new Float32Array(hc.B * 2);
    for (let i = 0; i < hc.B; i++) {
      pos[i * 2] = 0.05 + 0.67 * rnd();
      pos[i * 2 + 1] = 0.05 + 0.67 * rnd();
    }
    const posBuf = device.createBuffer({
      size: pos.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(posBuf, 0, pos);
    const velBuf = device.createBuffer({
      size: pos.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ---- the tfjs oracle: the REAL field class, hashgrid path -------------
    const field = new HelmholtzField({
      alpha: 0.5,
      hiddenUnits: [HID, HID],
      modelType: "hashgrid",
      gridSize: hc.gridSize,
      gridFeatures: hc.features,
      classes: hc.classes,
    });
    ok(
      field.trainableWeights.length === layout.segments.length,
      `${hc.label}tfjs var count ≡ packed segment count (${layout.segments.length})`
    );
    field.trainableWeights.forEach((v, i) => {
      const seg = layout.segments[i];
      const count = v.shape.reduce((a, b) => a * b, 1);
      if (count !== seg.floatLength) {
        throw new Error(
          `${hc.label}var ${i} has ${count} floats, segment has ${seg.floatLength}`
        );
      }
      v.assign(
        tf.tensor(
          Array.from(fw.slice(seg.floatOffset, seg.floatOffset + seg.floatLength)),
          v.shape
        )
      );
    });

    // `cursor` is 0 and stays 0 (partCount == b), so the shader's particle
    // index IS the sample index and the family it derives is familyOf(s).
    const familyOf = (s: number) =>
      hc.classes > 0 ? pcg(s ^ CLASS_SALT) % hc.classes : 0;
    const clsT =
      hc.classes > 0
        ? tf.tensor2d(
            Array.from({ length: hc.B }, (_, s) => familyOf(s)),
            [hc.B, 1]
          )
        : null;
    const posT = tf.tensor2d(Array.from(pos), [hc.B, 2]);

    /**
     * The family-planed grid lookup, which `HelmholtzField.forces` REFUSES to
     * do (`classes > 0` has no tfjs path — the label never enters that graph).
     * Same shape as its private `gridInterp` with `cls · gs²` folded into the
     * cell index, i.e. the thing `emitEncode` emits. It is a mirror, and a
     * mirror is only worth trusting because the forward parity assertion below
     * compares it against the shader's own F.
     */
    const planedForces = (lane: "blend" | 0 | 1): tf.Tensor2D =>
      tf.tidy(() => {
        const gs = hc.gridSize;
        const gc = posT.clipByValue(0, 1).mul(gs - 1);
        const i0 = gc.floor();
        const f = gc.sub(i0);
        const ix = i0.slice([0, 0], [-1, 1]);
        const iy = i0.slice([0, 1], [-1, 1]);
        const fx = f.slice([0, 0], [-1, 1]);
        const fy = f.slice([0, 1], [-1, 1]);
        const ix1 = ix.add(1).minimum(gs - 1);
        const iy1 = iy.add(1).minimum(gs - 1);
        const plane = clsT!.mul(gs * gs);
        const gather = (jx: tf.Tensor, jy: tf.Tensor) =>
          tf
            .oneHot(plane.add(jy.mul(gs)).add(jx).reshape([-1]).toInt(), planes * gs * gs)
            .matMul(field.grid!) as tf.Tensor2D;
        const encRows = gather(ix, iy)
          .mul(fx.mul(-1).add(1).mul(fy.mul(-1).add(1)))
          .add(gather(ix1, iy).mul(fx.mul(fy.mul(-1).add(1))))
          .add(gather(ix, iy1).mul(fx.mul(-1).add(1).mul(fy)))
          .add(gather(ix1, iy1).mul(fx.mul(fy))) as tf.Tensor2D;
        const nets = field.heads;
        if (lane !== "blend") return nets[lane].predict(encRows) as tf.Tensor2D;
        const gv = nets[0].predict(encRows) as tf.Tensor2D;
        const rv = nets[1].predict(encRows) as tf.Tensor2D;
        return gv.mul(1 - field.alpha).add(rv.mul(field.alpha)) as tf.Tensor2D;
      });
    /** The field signal the fused shader computes for this lane. */
    const forwardOf = (lane: "blend" | 0 | 1): tf.Tensor2D => {
      if (hc.classes > 0) return planedForces(lane);
      if (lane === "blend") return field.forces(posT);
      const heads = field.headForces(posT);
      heads[1 - lane].dispose();
      return heads[lane];
    };
    /**
     * The variables actually in the graph for this lane. A direct lane leaves
     * the other head out entirely, and tfjs throws rather than returning zeros
     * for a variable the loss never touched.
     */
    const varsOf = (lane: "blend" | 0 | 1) => {
      const all = field.trainableWeights;
      if (lane === "blend") return all;
      const perHead = (all.length - 1) / 2;
      return [all[0], ...all.slice(1 + lane * perHead, 1 + (lane + 1) * perHead)];
    };

    const makeSeam = (lane: "blend" | 0 | 1): FieldSeam => async (a) => {
      field.alpha = a.alpha;

      // (a) FORWARD parity. Proves the hashgrid encode + head forward line up
      //     before any gradient is blamed on the backward.
      const fRef = tf.tidy(() => forwardOf(lane).reshape([-1])).dataSync();
      let maxF = 0;
      for (let i = 0; i < hc.B * 2; i++) maxF = Math.max(maxF, Math.abs(fRef[i] - a.gpuF[i]));
      ok(maxF < 2e-5, `${a.label} F(u) ≡ tfjs hashgrid forward (max |ΔF| ${fmt(maxF)})`);

      // (b) THE SEAM. dL_gen/dW_field = ∂(Σ_s dSig_s · F(u_s))/∂W, with dSig
      //     the shader's own cotangent — exactly what `fieldGrad` assembles.
      const dT = tf.tensor2d(Array.from(a.dSig), [hc.B, 2]);
      const run = tf.variableGrads(
        () => tf.tidy(() => forwardOf(lane).mul(dT).sum().asScalar()),
        varsOf(lane)
      );
      // Everything not in this lane's graph stays 0 in the reference, and the
      // kernel must agree — which is the lane-isolation assertion below.
      const ref = new Float32Array(layout.totalFloats);
      varsOf(lane).forEach((v) => {
        const seg = layout.segments[field.trainableWeights.indexOf(v)];
        const arr = run.grads[v.name].dataSync();
        for (let x = 0; x < seg.floatLength; x++) ref[seg.floatOffset + x] = arr[x];
      });
      run.value.dispose();
      Object.values(run.grads).forEach((t) => t.dispose());
      dT.dispose();

      const gLo = gridSeg.floatOffset;
      const gHi = gridSeg.floatOffset + gridSeg.floatLength;
      const mlpIdx: number[] = [];
      for (const seg of layout.segments) {
        if (seg.role === "grid") continue;
        for (let x = 0; x < seg.floatLength; x++) mlpIdx.push(seg.floatOffset + x);
      }
      const grid = compareVec(ref.subarray(gLo, gHi), a.ext.subarray(gLo, gHi), gHi - gLo);
      const gSup = supportOf(ref.subarray(gLo, gHi), a.ext.subarray(gLo, gHi));
      const mlp = compareVec(
        mlpIdx.map((i) => ref[i]),
        mlpIdx.map((i) => a.ext[i]),
        mlpIdx.length
      );

      // THE assertion defect (c) fails: with the grid segment skipped, every
      // one of these floats is exactly 0, so nzGot == 0 and cos == 0 while
      // everything else in this file still passes.
      ok(
        gSup.nzGot > 0 && grid.finite && grid.cos > COS_MIN_EXT && grid.worstRel < REL_EXT_MAX,
        `${a.label} GRID extGrads [${gLo},${gHi}) ≡ tfjs — ${gSup.nzGot} NONZERO ` +
          `(cos 1-${fmt(1 - grid.cos)}, rel ${fmt(grid.worstRel)} of scale ${fmt(grid.scale)})`
      );
      ok(
        gSup.mismatch === 0,
        `${a.label} GRID support ≡ tfjs — the same cells are touched ` +
          `(${gSup.nzRef} tfjs vs ${gSup.nzGot} kernel, ${gSup.mismatch} mismatches)`
      );
      ok(
        mlp.finite && mlp.cos > COS_MIN_EXT && mlp.worstRel < REL_EXT_MAX,
        `${a.label} MLP extGrads (${mlpIdx.length} floats) ≡ tfjs ` +
          `(cos 1-${fmt(1 - mlp.cos)}, rel ${fmt(mlp.worstRel)} of scale ${fmt(mlp.scale)})`
      );

      // A direct lane emits no blocks for the other head, so its floats must be
      // EXACT zero rather than small. This is the §2a hazard's runtime face: the
      // lane that DOES run is also the lane that must SEED the shared dEnc
      // block, and a head-index seed leaves lane 1 accumulating into whatever
      // the previous step left there — which shows up in the GRID slice above,
      // not here.
      if (lane !== "blend") {
        const idle = layout.segments.filter((s) => s.role !== "grid" && s.head === 1 - lane);
        let worst = 0;
        let floats = 0;
        for (const seg of idle) {
          floats += seg.floatLength;
          for (let x = 0; x < seg.floatLength; x++) {
            worst = Math.max(worst, Math.abs(a.ext[seg.floatOffset + x]));
          }
        }
        ok(
          worst === 0,
          `${a.label} idle head ${1 - lane} extGrad is EXACT zero ` +
            `(${floats} floats, max |g| ${worst})`
        );
      }

      // (c) PLANE ISOLATION, derived host-side from the particle index rather
      //     than from the tfjs mirror above: a wrong `cls · gs²` term would be
      //     wrong in BOTH mirrors and cancel. A dropped plane term sends every
      //     family's gradient into plane 0 — planes 1 and 2 go silent and plane
      //     0 collects cells no family-0 particle ever read.
      if (hc.classes > 0) {
        const gs = hc.gridSize;
        const support = new Set<number>();
        const perFamily = new Array<number>(hc.classes).fill(0);
        for (let s = 0; s < hc.B; s++) {
          const fam = familyOf(s);
          perFamily[fam]++;
          const gxf = Math.min(Math.max(a.pos[s * 2], 0), 1) * (gs - 1);
          const gyf = Math.min(Math.max(a.pos[s * 2 + 1], 0), 1) * (gs - 1);
          const ix = Math.floor(gxf);
          const iy = Math.floor(gyf);
          const ix1 = Math.min(ix + 1, gs - 1);
          const iy1 = Math.min(iy + 1, gs - 1);
          const p = fam * gs * gs;
          support.add(p + iy * gs + ix);
          support.add(p + iy * gs + ix1);
          support.add(p + iy1 * gs + ix);
          support.add(p + iy1 * gs + ix1);
        }
        let outside = 0;
        const nzPerPlane = new Array<number>(hc.classes).fill(0);
        for (let cell = 0; cell < planes * gs * gs; cell++) {
          let any = false;
          for (let f = 0; f < hc.features; f++) {
            if (a.ext[gLo + cell * hc.features + f] !== 0) any = true;
          }
          if (!any) continue;
          nzPerPlane[Math.floor(cell / (gs * gs))]++;
          if (!support.has(cell)) outside++;
        }
        ok(
          perFamily.every((n) => n > 0) && nzPerPlane.every((n) => n > 0) && outside === 0,
          `${a.label} plane isolation: families ${perFamily.join("/")} sampled, ` +
            `cells with gradient per plane ${nzPerPlane.join("/")}, ${outside} outside support`
        );
      }
    };

    const runs: { kind: PixelGanKind; lane: "blend" | 0 | 1 }[] = [
      ...hc.kinds.map((kind) => ({ kind, lane: "blend" as const })),
      ...(hc.lanes ?? []),
    ];
    for (const r of runs) {
      await compareKind(
        device,
        layout,
        fwBuf,
        { kind: r.kind, G: hc.G, E: 4, K: 8, hidden: 8, dt: 0.05 },
        hc.B,
        posBuf,
        velBuf,
        {
          fieldTag: `${hc.label}${r.lane === "blend" ? "" : `lane${r.lane} `}`,
          seam: makeSeam(r.lane),
          fieldLane: r.lane,
        }
      );
    }

    posT.dispose();
    clsT?.dispose();
    field.dispose();
    fwBuf.destroy();
    posBuf.destroy();
    velBuf.destroy();
  }
}

device.destroy?.();

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
