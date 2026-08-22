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
 *   §1 shared-constant preconditions (no GPU needed), incl. the multi-guess
 *      head layout and the §3h per-guess init asymmetry
 *   §2 CPU ≡ GPU at the small test config  G=8  E=4  K=8  hidden=8
 *   §3 CPU ≡ GPU at the shipped gallery config G=16 E=8 K=16 hidden=32
 *       (src/main.ts:2403 et al. — PIXEL_DISC_DEFAULTS)
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
import { layoutField } from "../src/render/webgpu/advect_wgsl";
import {
  pixelDiscShader,
  pixelWeightLayout,
  pixelParticleScratchFloats,
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
 * One kind, one config: run the fused GPU step twice (disc-only, then with the
 * generator pass), reconstruct the exact same inputs on the CPU, and compare.
 */
async function compareKind(
  dev: GPUDevice,
  layout: ReturnType<typeof layoutField>,
  fieldWeightsBuf: GPUBuffer,
  dims: PixelDiscDims,
  B: number,
  posBuf: GPUBuffer,
  velBuf: GPUBuffer
): Promise<void> {
  const kind = dims.kind;
  const nCell = dims.G * dims.G;
  const nW = pixelDiscWeightCount(dims);
  const partStride = pixelParticleScratchFloats(layout);
  const P = pixelGuessCount(dims);
  // One label for every line of this comparison, so a failure names the guess
  // setting it happened under and not just the kind.
  const kl = `${kind}${P === 1 ? "" : ` k${P}ε${(dims.guesses as any).relaxEps}`}`;
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
    pos[s * 2] = scratchA[s * partStride + 0];
    pos[s * 2 + 1] = scratchA[s * partStride + 1];
    forces[s * 2] = scratchA[s * partStride + 2];
    forces[s * 2 + 1] = scratchA[s * partStride + 3];
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
    gpuDF[s * 2] = scratchB[s * partStride + 6];
    gpuDF[s * 2 + 1] = scratchB[s * partStride + 7];
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
device.destroy?.();

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
