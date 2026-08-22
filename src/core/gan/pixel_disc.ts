/**
 * Pixel GANs — CPU oracle for four 2D drawing games.
 *
 *   vec-field   density → per-cell 2-vector field vs F(cell centers)
 *   next-frame  density → next density vs splat(pos+dt·F)
 *   real-fake   density → logit; real cloud vs random-position fake
 *   inpaint     masked density → complete drawing; loss on mask only
 *
 * Shared trunk: soft splat → 3×3 conv → soft codebook. Reverse-mode only.
 * Spec: docs/PIXEL_DISC.md
 *
 * ## Multi-guess heads (docs/PLAN_MULTI_GUESS_MODULARIZATION.md §3)
 *
 * Three of the four kinds give the head `headCount(guesses)` GUESSES and score
 * it on a relaxed winner-take-all min: per cell the guess with the smallest
 * residual takes weight `1−ε`, the losers share ε. The spec — the two scalars,
 * the ε bound, the first-argmin tie rule — is `src/core/gan/wta.ts`; this file
 * imports it and never re-derives it.
 *
 * Guesses are a head STRIDE, not a fifth kind (§3a): the four `step*` handlers
 * stay four handlers and each gains an inner fold over guesses, which is a loop
 * inside a fixed kind. The TRUNK IS SHARED — conv and codebook run once per cell
 * into `cache.feat`/`cache.soft` and every guess reads the same features, which
 * is what keeps the GPU mirror inside its frame budget.
 */

import {
  type GuessKind,
  headCount,
  validateGuessKind,
  wtaScalars,
} from "./wta";

export type { GuessKind } from "./wta";

export type PixelGanKind = "vec-field" | "next-frame" | "real-fake" | "inpaint";

export const PIXEL_DISC_SOFT_EPS = 1e-4;
/** GAP tile grid: G must be divisible by this (16 and 32 both work). */
export const PIXEL_DISC_TILES = 4;
export const PIXEL_DISC_DEFAULTS = {
  G: 16,
  E: 8,
  K: 16,
  hidden: 32,
  dt: 0.15,
} as const;

export interface PixelDiscDims {
  G: number;
  E: number;
  /**
   * SOFT-CODEBOOK size — the number of logits the per-cell softmax runs over
   * (see backboneForward). NOT the guess count: that is {@link PixelDiscDims.guesses},
   * and it is deliberately never spelled `K` anywhere in this port, because a
   * second `K` in this struct is the single most likely way the hand-mirrored
   * WGSL goes wrong (plan §3b — every index over there is a baked literal).
   */
  K: number;
  hidden: number;
  dt: number;
  kind: PixelGanKind;
  /**
   * How many guesses the head gets, and how they are weighted (`src/core/gan/wta.ts`).
   *
   * OPTIONAL, and the optionality is NAMED rather than accidental: absence means
   * "this caller predates the knob", which canonicalizes to {@link SINGLE_GUESS}
   * at exactly one place ({@link guessesOf}). Nothing downstream reads the field
   * directly.
   */
  guesses?: GuessKind;
}

/** The default: one head, one guess, weight 1. */
export const SINGLE_GUESS: GuessKind = { tag: "single" };

/** κ — the ONE place an absent `guesses` becomes a canonical {@link GuessKind}. */
export function guessesOf(d: Pick<PixelDiscDims, "guesses">): GuessKind {
  return d.guesses ?? SINGLE_GUESS;
}

/**
 * κ — canonicality of the (kind, guesses) PAIR, which the two types cannot
 * express separately. Throws; never repairs.
 *
 * ## Why `real-fake` refuses guesses (plan §3g, decided rather than defaulted)
 *
 * The other three kinds score a per-cell PREDICTION against a target, and WTA is
 * exactly the tool for "the target may be one of several things; score the
 * predictor on its best guess". `real-fake` has no per-cell prediction at all:
 * it global-average-pools, runs an MLP, and takes BCE against a LABEL. Three
 * separate things break if guesses are bolted onto that MLP output:
 *
 *  1. BCE against a label has ONE correct answer. `min_j BCE(logit_j, target)`
 *     does not select the best-covering mode, it selects the most CONFIDENT
 *     critic and hands it (1−ε) of the gradient — the rich-get-richer dynamic
 *     §3f exists to detect, with none of the mode-coverage benefit that pays for
 *     it elsewhere.
 *  2. The generator pass has no label, so the winner is UNDEFINED there. Its
 *     payoff is `genSign · (−logit)` and `genSign` is negative, so "min over
 *     guesses" flips meaning with the sign. Any pick (min, mean, guess 0) is
 *     arbitrary, and disc and gen would then train different heads.
 *  3. §3g requires the win counter to be gated by the same predicate that gates
 *     the residual. `real-fake` has no per-cell predicate — its whole step is
 *     TWO samples (real, fake) — so its win histogram would be 2 counts/step
 *     against 256 cells/step for every other kind, and the collapse threshold
 *     would mean something different for this kind alone.
 *
 * So the answer is "not at all", and it is enforced LOUDLY here rather than by
 * silently clamping `headCount` to 1, which would leave a UI knob that appears
 * to do something and does not.
 */
export function validatePixelDims(d: PixelDiscDims): void {
  const g = guessesOf(d);
  validateGuessKind(g);
  if (d.kind === "real-fake" && g.tag !== "single") {
    throw new Error(
      `pixel_disc: kind "real-fake" scores a LABEL through BCE, not a per-cell ` +
        `residual, so it has no winner to take all — it accepts only ` +
        `guesses {tag:"single"}, got ${JSON.stringify(g)}`
    );
  }
}

export function resolvePixelDims(
  partial?: Partial<PixelDiscDims> & { kind?: PixelGanKind }
): PixelDiscDims {
  const d: PixelDiscDims = {
    G: partial?.G ?? PIXEL_DISC_DEFAULTS.G,
    E: partial?.E ?? PIXEL_DISC_DEFAULTS.E,
    K: partial?.K ?? PIXEL_DISC_DEFAULTS.K,
    hidden: partial?.hidden ?? PIXEL_DISC_DEFAULTS.hidden,
    dt: partial?.dt ?? PIXEL_DISC_DEFAULTS.dt,
    kind: partial?.kind ?? "vec-field",
    guesses: guessesOf(partial ?? {}),
  };
  validatePixelDims(d);
  return d;
}

/** Guesses this head is replicated across. 1 unless `guesses` says otherwise. */
export function pixelGuessCount(d: PixelDiscDims): number {
  return headCount(guessesOf(d));
}

/** Packed critic weights — layout depends on kind (see pixelDiscWeightCount). */
export interface PixelDiscWeights {
  convW: Float64Array;
  convB: Float64Array;
  code: Float64Array;
  /** Kind head params, packed contiguously after codebook. */
  head: Float64Array;
}

/**
 * Floats in ONE guess's head — the STRIDE between consecutive guesses in the
 * packed head slice. The head is guess-major: guess `j` owns
 * `[j·headFloatsPerGuess, (j+1)·headFloatsPerGuess)`, so `pixelWeightLayout`'s
 * GPU offsets are the guess-0 offsets plus `j · headStride`.
 */
export function headFloatsPerGuess(d: PixelDiscDims): number {
  switch (d.kind) {
    case "vec-field":
      return d.K * 2 + 2; // linear soft→2
    case "next-frame":
    case "inpaint":
      return d.K + 1; // linear soft→1 density
    case "real-fake":
      return d.hidden * d.K + d.hidden + d.hidden + 1; // GAP→MLP→logit
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

/** Total head floats: guesses are a head STRIDE, not a fifth kind (plan §3a). */
export function headFloats(d: PixelDiscDims): number {
  return pixelGuessCount(d) * headFloatsPerGuess(d);
}

export function pixelDiscWeightCount(d: PixelDiscDims): number {
  return d.E * 9 + d.E + d.K * d.E + headFloats(d);
}

export function packPixelDiscWeights(w: PixelDiscWeights, d: PixelDiscDims): Float32Array {
  const out = new Float32Array(pixelDiscWeightCount(d));
  let o = 0;
  out.set(w.convW, o); o += w.convW.length;
  out.set(w.convB, o); o += w.convB.length;
  out.set(w.code, o); o += w.code.length;
  out.set(w.head, o);
  return out;
}

export function unpackPixelDiscWeights(
  packed: ArrayLike<number>,
  d: PixelDiscDims
): PixelDiscWeights {
  const need = pixelDiscWeightCount(d);
  if (packed.length !== need) {
    throw new Error(`pixel_disc: pack length ${packed.length} ≠ ${need}`);
  }
  let o = 0;
  const take = (n: number) => {
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) a[i] = packed[o++];
    return a;
  };
  return {
    convW: take(d.E * 9),
    convB: take(d.E),
    code: take(d.K * d.E),
    head: take(headFloats(d)),
  };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function initPixelDiscWeights(d: PixelDiscDims, seed: number): PixelDiscWeights {
  const rnd = mulberry32(seed);
  const gauss = () => {
    const u1 = Math.max(rnd(), 1e-12);
    const u2 = rnd();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  const fill = (n: number, fanIn: number, fanOut: number) => {
    const std = Math.sqrt(2 / (fanIn + fanOut));
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) a[i] = gauss() * std;
    return a;
  };
  const head = new Float64Array(headFloats(d));
  let o = 0;
  const fillHead = (n: number, fanIn: number, fanOut: number) => {
    const std = Math.sqrt(2 / (fanIn + fanOut));
    for (let i = 0; i < n; i++) head[o++] = gauss() * std;
  };
  /** δ: kind → the filler for ONE guess's head slice. */
  const fillOneGuess = (): void => {
    switch (d.kind) {
      case "vec-field":
        fillHead(d.K * 2, d.K, 2);
        o += 2; // bias zero
        return;
      case "next-frame":
      case "inpaint":
        fillHead(d.K, d.K, 1);
        o += 1;
        return;
      case "real-fake":
        fillHead(d.hidden * d.K, d.K, d.hidden);
        o += d.hidden; // bias0
        fillHead(d.hidden, d.hidden, 1);
        o += 1; // bias1
        return;
      default: {
        const _e: never = d.kind;
        throw new Error(`pixel_disc: bad kind ${_e}`);
      }
    }
  };
  /**
   * PER-GUESS SYMMETRY BREAKING — load-bearing, and currently IMPLICIT, which is
   * the fragile kind (plan §3h).
   *
   * `gauss()` draws from ONE sequential mulberry32 stream whose state advances
   * across calls, so calling `fillOneGuess()` once per guess yields distinct
   * draws for free. That is the whole mechanism: there is no per-guess seed here
   * the way `adversary.ts` derives `cfg.seed·7919 + j·101 + layer`.
   *
   * DO NOT "optimize" this loop into initialize-one-head-and-broadcast, and do
   * not hoist the fill out of the loop. Identical heads produce identical
   * residuals on every cell; every comparison is then an exact tie; the
   * first-argmin tie rule routes ALL of them to guess 0; guesses 1..K−1 receive
   * only the ε loser share of a payoff they never influence, and the mixture is
   * dead on arrival while the loss looks completely normal.
   *
   * `tools/pixel_disc_equiv_test.ts` asserts two guesses' head slices differ.
   */
  const guesses = pixelGuessCount(d);
  for (let j = 0; j < guesses; j++) fillOneGuess();
  return {
    convW: fill(d.E * 9, 9, d.E),
    convB: new Float64Array(d.E),
    code: fill(d.K * d.E, d.E, d.K),
    head,
  };
}

export function softResidual1(a: number, b: number): number {
  const d = a - b;
  return Math.sqrt(d * d + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
}

export function softResidual2(yx: number, yy: number, tx: number, ty: number): number {
  const dx = yx - tx;
  const dy = yy - ty;
  return Math.sqrt(dx * dx + dy * dy + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
}

export function splatMass(B: number, G: number): number {
  return (G * G) / B;
}

export function softSplatDensity(
  pos: ArrayLike<number>,
  B: number,
  G: number,
  out?: Float64Array
): Float64Array {
  const D = out ?? new Float64Array(G * G);
  D.fill(0);
  const scale = G;
  const mass = splatMass(B, G);
  for (let i = 0; i < B; i++) {
    let fx = pos[i * 2] * scale - 0.5;
    let fy = pos[i * 2 + 1] * scale - 0.5;
    fx = Math.min(G - 1.0000001, Math.max(0, fx));
    fy = Math.min(G - 1.0000001, Math.max(0, fy));
    const ix = Math.floor(fx);
    const iy = Math.floor(fy);
    const tx = fx - ix;
    const ty = fy - iy;
    const ix1 = Math.min(ix + 1, G - 1);
    const iy1 = Math.min(iy + 1, G - 1);
    D[iy * G + ix] += (1 - tx) * (1 - ty) * mass;
    D[iy * G + ix1] += tx * (1 - ty) * mass;
    D[iy1 * G + ix] += (1 - tx) * ty * mass;
    D[iy1 * G + ix1] += tx * ty * mass;
  }
  return D;
}

export function softSplatVjp(
  pos: ArrayLike<number>,
  dD: ArrayLike<number>,
  B: number,
  G: number,
  dPos: Float64Array
): void {
  dPos.fill(0);
  const scale = G;
  const mass = splatMass(B, G);
  for (let i = 0; i < B; i++) {
    let fx = pos[i * 2] * scale - 0.5;
    let fy = pos[i * 2 + 1] * scale - 0.5;
    fx = Math.min(G - 1.0000001, Math.max(0, fx));
    fy = Math.min(G - 1.0000001, Math.max(0, fy));
    const ix = Math.floor(fx);
    const iy = Math.floor(fy);
    const tx = fx - ix;
    const ty = fy - iy;
    const ix1 = Math.min(ix + 1, G - 1);
    const iy1 = Math.min(iy + 1, G - 1);
    const g00 = dD[iy * G + ix];
    const g10 = dD[iy * G + ix1];
    const g01 = dD[iy1 * G + ix];
    const g11 = dD[iy1 * G + ix1];
    const dfx = mass * (-(1 - ty) * g00 + (1 - ty) * g10 - ty * g01 + ty * g11);
    const dfy = mass * (-(1 - tx) * g00 - tx * g10 + (1 - tx) * g01 + tx * g11);
    const px = pos[i * 2];
    const py = pos[i * 2 + 1];
    const insideX = px * scale - 0.5 > 0 && px * scale - 0.5 < G - 1;
    const insideY = py * scale - 0.5 > 0 && py * scale - 0.5 < G - 1;
    dPos[i * 2] = insideX ? dfx * scale : 0;
    dPos[i * 2 + 1] = insideY ? dfy * scale : 0;
  }
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

export interface BackboneCache {
  feat: Float64Array;
  soft: Float64Array;
}

export function makeBackboneCache(d: PixelDiscDims): BackboneCache {
  const n = d.G * d.G;
  return {
    feat: new Float64Array(d.E * n),
    soft: new Float64Array(d.K * n),
  };
}

/** Conv3×3 + soft codebook. Writes cache. */
export function backboneForward(
  D: ArrayLike<number>,
  w: PixelDiscWeights,
  d: PixelDiscDims,
  cache: BackboneCache
): void {
  const { G, E, K } = d;
  const nCell = G * G;
  const { feat, soft } = cache;
  for (let e = 0; e < E; e++) {
    const b = w.convB[e];
    for (let y = 0; y < G; y++) {
      for (let x = 0; x < G; x++) {
        let s = b;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = Math.min(G - 1, Math.max(0, y + dy));
          for (let dx = -1; dx <= 1; dx++) {
            const xx = Math.min(G - 1, Math.max(0, x + dx));
            s += w.convW[e * 9 + (dy + 1) * 3 + (dx + 1)] * D[yy * G + xx];
          }
        }
        feat[e * nCell + y * G + x] = relu(s);
      }
    }
  }
  for (let cell = 0; cell < nCell; cell++) {
    let maxL = -Infinity;
    const logits = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let e = 0; e < E; e++) s += feat[e * nCell + cell] * w.code[k * E + e];
      logits[k] = s;
      if (s > maxL) maxL = s;
    }
    let sum = 0;
    for (let k = 0; k < K; k++) {
      const v = Math.exp(logits[k] - maxL);
      soft[k * nCell + cell] = v;
      sum += v;
    }
    const inv = 1 / sum;
    for (let k = 0; k < K; k++) soft[k * nCell + cell] *= inv;
  }
}

export interface CriticGrads {
  convW: Float64Array;
  convB: Float64Array;
  code: Float64Array;
  head: Float64Array;
  dD: Float64Array;
}

export function zeroCriticGrads(d: PixelDiscDims): CriticGrads {
  return {
    convW: new Float64Array(d.E * 9),
    convB: new Float64Array(d.E),
    code: new Float64Array(d.K * d.E),
    head: new Float64Array(headFloats(d)),
    dD: new Float64Array(d.G * d.G),
  };
}

/** VJP through backbone given per-cell dL/dsoft[K]. */
function backboneBackwardFromDSoft(
  dSoft: Float64Array,
  D: ArrayLike<number>,
  w: PixelDiscWeights,
  d: PixelDiscDims,
  cache: BackboneCache,
  g: CriticGrads
): void {
  const { G, E, K } = d;
  const nCell = G * G;
  const { feat, soft } = cache;
  const dFeat = new Float64Array(E * nCell);
  for (let c = 0; c < nCell; c++) {
    let dot = 0;
    for (let k = 0; k < K; k++) dot += soft[k * nCell + c] * dSoft[k * nCell + c];
    for (let k = 0; k < K; k++) {
      const dLogit = soft[k * nCell + c] * (dSoft[k * nCell + c] - dot);
      for (let e = 0; e < E; e++) {
        g.code[k * E + e] += dLogit * feat[e * nCell + c];
        dFeat[e * nCell + c] += dLogit * w.code[k * E + e];
      }
    }
  }
  for (let e = 0; e < E; e++) {
    for (let y = 0; y < G; y++) {
      for (let x = 0; x < G; x++) {
        const idx = e * nCell + y * G + x;
        if (feat[idx] <= 0) continue;
        const gf = dFeat[idx];
        g.convB[e] += gf;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = Math.min(G - 1, Math.max(0, y + dy));
          for (let dx = -1; dx <= 1; dx++) {
            const xx = Math.min(G - 1, Math.max(0, x + dx));
            const wi = e * 9 + (dy + 1) * 3 + (dx + 1);
            g.convW[wi] += gf * D[yy * G + xx];
            g.dD[yy * G + xx] += gf * w.convW[wi];
          }
        }
      }
    }
  }
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function bceGradLogit(logit: number, target: number): { loss: number; dLogit: number } {
  // softplus-style stable BCE
  const p = sigmoid(logit);
  const loss = -(target * Math.log(p + 1e-8) + (1 - target) * Math.log(1 - p + 1e-8));
  return { loss, dLogit: p - target };
}

/** Build a random inpaint mask (axis-aligned block ~25% area). */
export function makeInpaintMask(G: number, seed: number): Uint8Array {
  const rnd = mulberry32(seed);
  const mask = new Uint8Array(G * G);
  const bw = Math.max(2, Math.floor(G * 0.5));
  const bh = Math.max(2, Math.floor(G * 0.5));
  const x0 = Math.floor(rnd() * (G - bw + 1));
  const y0 = Math.floor(rnd() * (G - bh + 1));
  for (let y = y0; y < y0 + bh; y++) {
    for (let x = x0; x < x0 + bw; x++) mask[y * G + x] = 1;
  }
  return mask;
}

export interface PixelDiscStepResult {
  discLoss: number;
  genLoss: number;
  dF: Float64Array;
  discGradPacked: Float32Array;
  /**
   * DISC-pass winning guess per cell, `-1` where the cell was INACTIVE for this
   * kind (plan §3g: an inactive cell still has a mathematical argmin, and
   * counting it would make guess 0 look dominant on a mostly-empty grid and turn
   * §3f's collapse detector into a liar). `null` for `real-fake`, which has no
   * per-cell residual and therefore no winner — absence is the meaning, so it is
   * not stood in for by an array of zeros.
   *
   * Exists so `tools/pixel_disc_equiv_test.ts` can compare the SELECTION against
   * the GPU's, not merely the summed gradient: two backends can disagree about
   * which guess won on a handful of cells and still land within tolerance on the
   * sum, because winner/loser weight swaps partially cancel.
   */
  winIdx: Int32Array | null;
}

/** The compile-time-ish constants of one step's WTA fold. See src/core/gan/wta.ts. */
interface GuessPlan {
  /** Guess count. */
  readonly P: number;
  /** Weight of the winning guess (`1−ε`). */
  readonly winner: number;
  /** Weight of EACH losing guess (`ε/(P−1)`, `0` for `single`). */
  readonly loser: number;
  /** Floats between consecutive guesses' head slices. */
  readonly stride: number;
}

function guessPlan(d: PixelDiscDims): GuessPlan {
  const g = guessesOf(d);
  const { winner, loser } = wtaScalars(g);
  return {
    P: headCount(g),
    winner,
    loser,
    stride: headFloatsPerGuess(d),
  };
}

/**
 * First-argmin over `r[0..P)`. STRICT `<` scanning ascending, so an equal later
 * residual does NOT displace the incumbent — the tie rule all four backends
 * honor (src/core/gan/wta.ts). Exact ties are not hypothetical: identical heads
 * (plan §3h) tie on every cell.
 */
function argMinFirst(r: Float64Array, P: number): number {
  let win = 0;
  let best = r[0];
  for (let j = 1; j < P; j++) {
    if (r[j] < best) {
      best = r[j];
      win = j;
    }
  }
  return win;
}

function applyDiscGrads(
  w: PixelDiscWeights,
  d: PixelDiscDims,
  g: CriticGrads,
  lr: number
): Float32Array {
  const packed = packPixelDiscWeights(
    { convW: g.convW, convB: g.convB, code: g.code, head: g.head },
    d
  );
  if (lr > 0) {
    const cur = packPixelDiscWeights(w, d);
    for (let i = 0; i < cur.length; i++) cur[i] -= lr * packed[i];
    const nw = unpackPixelDiscWeights(cur, d);
    w.convW.set(nw.convW);
    w.convB.set(nw.convB);
    w.code.set(nw.code);
    w.head.set(nw.head);
  }
  return packed;
}

/**
 * One disc (+ optional gen-through-density) step.
 *
 * `forces` — [B*2] F at particle positions (normalized coords in `pos`).
 * `forceGrid` — optional [G*G*2] F at cell centers (required for vec-field).
 * `maskSeed` — inpaint mask seed (frame index).
 */
export function pixelDiscStep(
  pos: Float64Array,
  forces: Float64Array,
  w: PixelDiscWeights,
  d: PixelDiscDims,
  opts: {
    applyDisc: boolean;
    genSign: number;
    lr: number;
    forceGrid?: Float64Array;
    maskSeed?: number;
    fakePos?: Float64Array;
  }
): PixelDiscStepResult {
  // κ once, at the entry to the core: after this the four handlers may assume
  // (kind, guesses) is canonical and never re-check it.
  validatePixelDims(d);
  switch (d.kind) {
    case "vec-field":
      return stepVecField(pos, forces, w, d, opts);
    case "next-frame":
      return stepNextFrame(pos, forces, w, d, opts);
    case "real-fake":
      return stepRealFake(pos, forces, w, d, opts);
    case "inpaint":
      return stepInpaint(pos, forces, w, d, opts);
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

function stepVecField(
  pos: Float64Array,
  forces: Float64Array,
  w: PixelDiscWeights,
  d: PixelDiscDims,
  opts: {
    applyDisc: boolean;
    genSign: number;
    lr: number;
    forceGrid?: Float64Array;
  }
): PixelDiscStepResult {
  const B = pos.length / 2;
  const { G, K } = d;
  const nCell = G * G;
  if (!opts.forceGrid || opts.forceGrid.length !== nCell * 2) {
    throw new Error("pixel_disc vec-field needs forceGrid[G*G*2]");
  }
  const D = softSplatDensity(pos, B, G);
  const cache = makeBackboneCache(d);
  backboneForward(D, w, d, cache);

  // head layout, PER GUESS: W[2,K] row-major then bias[2]; guess j starts at
  // j·stride. The trunk above ran ONCE — every guess reads the same cache.soft.
  const { P, winner, loser, stride } = guessPlan(d);
  let discR = 0;
  let nActive = 0;
  const dSoft = new Float64Array(K * nCell);
  const g = zeroCriticGrads(d);
  const densFloor = 1e-3;
  const winIdx = new Int32Array(nCell).fill(-1);
  const rr = new Float64Array(P);
  const ex = new Float64Array(P);
  const ey = new Float64Array(P);
  for (let c = 0; c < nCell; c++) {
    if (D[c] < densFloor) continue;
    nActive++;
    const tx = opts.forceGrid[c * 2];
    const ty = opts.forceGrid[c * 2 + 1];
    for (let j = 0; j < P; j++) {
      const h = j * stride;
      let vx = w.head[h + 2 * K];
      let vy = w.head[h + 2 * K + 1];
      for (let k = 0; k < K; k++) {
        vx += w.head[h + 0 * K + k] * cache.soft[k * nCell + c];
        vy += w.head[h + 1 * K + k] * cache.soft[k * nCell + c];
      }
      const dx = vx - tx;
      const dy = vy - ty;
      ex[j] = dx;
      ey[j] = dy;
      rr[j] = Math.sqrt(dx * dx + dy * dy + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
    }
    // Selection is OFF-TAPE: `win` and the two weights are constants w.r.t. the
    // gradient, exactly as in the other three backends.
    const win = argMinFirst(rr, P);
    winIdx[c] = win;
    for (let j = 0; j < P; j++) {
      const wj = j === win ? winner : loser;
      const h = j * stride;
      discR += wj * rr[j];
      const s = 1 / rr[j];
      const dvx = ex[j] * s * wj;
      const dvy = ey[j] * s * wj;
      g.head[h + 2 * K] += dvx;
      g.head[h + 2 * K + 1] += dvy;
      for (let k = 0; k < K; k++) {
        g.head[h + 0 * K + k] += dvx * cache.soft[k * nCell + c];
        g.head[h + 1 * K + k] += dvy * cache.soft[k * nCell + c];
        dSoft[k * nCell + c] +=
          dvx * w.head[h + 0 * K + k] + dvy * w.head[h + 1 * K + k];
      }
    }
  }
  const active = Math.max(nActive, 1);
  discR /= active;
  for (let i = 0; i < g.head.length; i++) g.head[i] /= active;
  for (let i = 0; i < dSoft.length; i++) dSoft[i] /= active;
  backboneBackwardFromDSoft(dSoft, D, w, d, cache, g);
  const discGradPacked = applyDiscGrads(w, d, g, opts.applyDisc ? opts.lr : 0);

  // Gen through D(pos')
  let genLoss = 0;
  const dF = new Float64Array(B * 2);
  if (opts.genSign !== 0) {
    const pos2 = new Float64Array(B * 2);
    for (let i = 0; i < B; i++) {
      pos2[i * 2] = pos[i * 2] + d.dt * forces[i * 2];
      pos2[i * 2 + 1] = pos[i * 2 + 1] + d.dt * forces[i * 2 + 1];
    }
    const D2 = softSplatDensity(pos2, B, G);
    const cache2 = makeBackboneCache(d);
    backboneForward(D2, w, d, cache2);
    const g2 = zeroCriticGrads(d);
    const dSoft2 = new Float64Array(K * nCell);
    let genR = 0;
    let n2 = 0;
    for (let c = 0; c < nCell; c++) {
      if (D2[c] < densFloor) continue;
      n2++;
      const tx = opts.forceGrid[c * 2];
      const ty = opts.forceGrid[c * 2 + 1];
      for (let j = 0; j < P; j++) {
        const h = j * stride;
        let vx = 0;
        let vy = 0;
        for (let k = 0; k < K; k++) {
          vx += w.head[h + 0 * K + k] * cache2.soft[k * nCell + c];
          vy += w.head[h + 1 * K + k] * cache2.soft[k * nCell + c];
        }
        vx += w.head[h + 2 * K];
        vy += w.head[h + 2 * K + 1];
        const dx = vx - tx;
        const dy = vy - ty;
        ex[j] = dx;
        ey[j] = dy;
        rr[j] = Math.sqrt(dx * dx + dy * dy + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
      }
      // The gen pass re-selects on ITS OWN density (D2), the same way the GPU's
      // criticGen does — the disc pass's winner is not carried over.
      const win = argMinFirst(rr, P);
      for (let j = 0; j < P; j++) {
        const wj = j === win ? winner : loser;
        const h = j * stride;
        genR += wj * rr[j];
        const s = opts.genSign / rr[j];
        const dvx = ex[j] * s * wj;
        const dvy = ey[j] * s * wj;
        for (let k = 0; k < K; k++) {
          dSoft2[k * nCell + c] +=
            dvx * w.head[h + 0 * K + k] + dvy * w.head[h + 1 * K + k];
        }
      }
    }
    const a2 = Math.max(n2, 1);
    genR /= a2;
    genLoss = opts.genSign * genR;
    for (let i = 0; i < dSoft2.length; i++) dSoft2[i] /= a2;
    backboneBackwardFromDSoft(dSoft2, D2, w, d, cache2, g2);
    const dPos = new Float64Array(B * 2);
    softSplatVjp(pos2, g2.dD, B, G, dPos);
    for (let i = 0; i < B * 2; i++) dF[i] = d.dt * dPos[i];
  }

  return { discLoss: discR, genLoss, dF, discGradPacked, winIdx };
}

function stepNextFrame(
  pos: Float64Array,
  forces: Float64Array,
  w: PixelDiscWeights,
  d: PixelDiscDims,
  opts: { applyDisc: boolean; genSign: number; lr: number }
): PixelDiscStepResult {
  const B = pos.length / 2;
  const { G, K } = d;
  const nCell = G * G;
  const D0 = softSplatDensity(pos, B, G);
  const pos2 = new Float64Array(B * 2);
  for (let i = 0; i < B; i++) {
    pos2[i * 2] = pos[i * 2] + d.dt * forces[i * 2];
    pos2[i * 2 + 1] = pos[i * 2 + 1] + d.dt * forces[i * 2 + 1];
  }
  const D1 = softSplatDensity(pos2, B, G);

  const cache = makeBackboneCache(d);
  backboneForward(D0, w, d, cache);
  const g = zeroCriticGrads(d);
  const dSoft = new Float64Array(K * nCell);
  // Guesses are a head stride; the trunk above ran once and is shared (§3a).
  const { P, winner, loser, stride } = guessPlan(d);
  const rr = new Float64Array(P);
  const dd = new Float64Array(P);
  // EVERY cell is active for next-frame, so every cell's winner counts (§3g).
  const winIdx = new Int32Array(nCell);
  let discR = 0;
  for (let c = 0; c < nCell; c++) {
    for (let j = 0; j < P; j++) {
      const h = j * stride;
      let pred = w.head[h + K];
      for (let k = 0; k < K; k++) pred += w.head[h + k] * cache.soft[k * nCell + c];
      const diff = pred - D1[c];
      dd[j] = diff;
      rr[j] = Math.sqrt(diff * diff + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
    }
    const win = argMinFirst(rr, P);
    winIdx[c] = win;
    for (let j = 0; j < P; j++) {
      const wj = j === win ? winner : loser;
      const h = j * stride;
      discR += wj * rr[j];
      const dPred = (dd[j] / rr[j]) * wj;
      g.head[h + K] += dPred;
      for (let k = 0; k < K; k++) {
        g.head[h + k] += dPred * cache.soft[k * nCell + c];
        dSoft[k * nCell + c] += dPred * w.head[h + k];
      }
    }
  }
  discR /= nCell;
  for (let i = 0; i < g.head.length; i++) g.head[i] /= nCell;
  for (let i = 0; i < dSoft.length; i++) dSoft[i] /= nCell;
  backboneBackwardFromDSoft(dSoft, D0, w, d, cache, g);
  // Also path through D1 target: dR/dD1 = -Σ_j w_j·dPred_j/nCell — for gen via
  // splat(pos'). NOTE: nothing reads dD1 today (the generator block below
  // recomputes the same quantity into `gD`); it is kept in step with the guess
  // fold rather than left as a stale single-guess expression.
  const dD1 = new Float64Array(nCell);
  {
    // recompute dPred for target side
    for (let c = 0; c < nCell; c++) {
      for (let j = 0; j < P; j++) {
        const h = j * stride;
        let pred = w.head[h + K];
        for (let k = 0; k < K; k++) pred += w.head[h + k] * cache.soft[k * nCell + c];
        const diff = pred - D1[c];
        dd[j] = diff;
        rr[j] = Math.sqrt(diff * diff + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
      }
      const win = argMinFirst(rr, P);
      let acc = 0;
      for (let j = 0; j < P; j++) {
        const wj = j === win ? winner : loser;
        acc += (-dd[j] / rr[j]) * wj;
      }
      dD1[c] = acc / nCell;
    }
  }
  const discGradPacked = applyDiscGrads(w, d, g, opts.applyDisc ? opts.lr : 0);

  let genLoss = 0;
  const dF = new Float64Array(B * 2);
  if (opts.genSign !== 0) {
    // Maximize residual: L = genSign * R, R uses D1(pos'). Stopgrad D0 path for gen
    // (critic frozen) — only D1 depends on F.
    const gD = new Float64Array(nCell);
    let genR = 0;
    const cache2 = makeBackboneCache(d);
    backboneForward(D0, w, d, cache2); // frozen critic on D0
    for (let c = 0; c < nCell; c++) {
      for (let j = 0; j < P; j++) {
        const h = j * stride;
        let pred = w.head[h + K];
        for (let k = 0; k < K; k++) pred += w.head[h + k] * cache2.soft[k * nCell + c];
        const diff = pred - D1[c];
        dd[j] = diff;
        rr[j] = Math.sqrt(diff * diff + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
      }
      const win = argMinFirst(rr, P);
      let acc = 0;
      for (let j = 0; j < P; j++) {
        const wj = j === win ? winner : loser;
        genR += wj * rr[j];
        acc += opts.genSign * (-dd[j] / rr[j]) * wj;
      }
      gD[c] = acc / nCell;
    }
    genR /= nCell;
    genLoss = opts.genSign * genR;
    const dPos = new Float64Array(B * 2);
    softSplatVjp(pos2, gD, B, G, dPos);
    for (let i = 0; i < B * 2; i++) dF[i] = d.dt * dPos[i];
  }
  return { discLoss: discR, genLoss, dF, discGradPacked, winIdx };
}

function stepRealFake(
  pos: Float64Array,
  forces: Float64Array,
  w: PixelDiscWeights,
  d: PixelDiscDims,
  opts: {
    applyDisc: boolean;
    genSign: number;
    lr: number;
    fakePos?: Float64Array;
  }
): PixelDiscStepResult {
  // NO GUESS FOLD, deliberately — see validatePixelDims for the three reasons
  // (BCE against a label has one right answer; the gen pass has no winner; there
  // is no per-cell activity predicate to gate a win counter with). `guesses` is
  // canonicalized to "single" for this kind, so the head below is one head.
  const B = pos.length / 2;
  const { G, K, hidden } = d;
  const nCell = G * G;
  const Dreal = softSplatDensity(pos, B, G);
  const fakePos = opts.fakePos ?? (() => {
    const rnd = mulberry32(12345);
    const p = new Float64Array(B * 2);
    for (let i = 0; i < B * 2; i++) p[i] = rnd();
    return p;
  })();
  const Dfake = softSplatDensity(fakePos, B, G);

  const logitOf = (D: Float64Array, cache: BackboneCache, g?: CriticGrads, target?: number) => {
    backboneForward(D, w, d, cache);
    const z = new Float64Array(K);
    for (let c = 0; c < nCell; c++) {
      for (let k = 0; k < K; k++) z[k] += cache.soft[k * nCell + c];
    }
    for (let k = 0; k < K; k++) z[k] /= nCell;
    // head: W0[hidden,K], b0[hidden], W1[hidden], b1
    const h = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      let s = w.head[hidden * K + i];
      for (let k = 0; k < K; k++) s += w.head[i * K + k] * z[k];
      h[i] = Math.tanh(s);
    }
    const w1Off = hidden * K + hidden;
    let logit = w.head[w1Off + hidden];
    for (let i = 0; i < hidden; i++) logit += w.head[w1Off + i] * h[i];

    if (g && target !== undefined) {
      const { loss, dLogit } = bceGradLogit(logit, target);
      // backward MLP
      g.head[w1Off + hidden] += dLogit;
      const dh = new Float64Array(hidden);
      for (let i = 0; i < hidden; i++) {
        g.head[w1Off + i] += dLogit * h[i];
        dh[i] = dLogit * w.head[w1Off + i];
      }
      const dz = new Float64Array(K);
      for (let i = 0; i < hidden; i++) {
        const pre = dh[i] * (1 - h[i] * h[i]);
        g.head[hidden * K + i] += pre;
        for (let k = 0; k < K; k++) {
          g.head[i * K + k] += pre * z[k];
          dz[k] += pre * w.head[i * K + k];
        }
      }
      const dSoft = new Float64Array(K * nCell);
      for (let k = 0; k < K; k++) {
        const gz = dz[k] / nCell;
        for (let c = 0; c < nCell; c++) dSoft[k * nCell + c] = gz;
      }
      backboneBackwardFromDSoft(dSoft, D, w, d, cache, g);
      return { logit, loss };
    }
    return { logit, loss: 0 };
  };

  const cacheR = makeBackboneCache(d);
  const cacheF = makeBackboneCache(d);
  const g = zeroCriticGrads(d);
  const real = logitOf(Dreal, cacheR, g, 1);
  const fake = logitOf(Dfake, cacheF, g, 0);
  const discLoss = real.loss + fake.loss;
  const discGradPacked = applyDiscGrads(w, d, g, opts.applyDisc ? opts.lr : 0);

  let genLoss = 0;
  const dF = new Float64Array(B * 2);
  if (opts.genSign !== 0) {
    // Maximize real logit of D(pos'): L = genSign * (-logit) with genSign=-weight ⇒ max logit
    const pos2 = new Float64Array(B * 2);
    for (let i = 0; i < B; i++) {
      pos2[i * 2] = pos[i * 2] + d.dt * forces[i * 2];
      pos2[i * 2 + 1] = pos[i * 2 + 1] + d.dt * forces[i * 2 + 1];
    }
    const D2 = softSplatDensity(pos2, B, G);
    const cache2 = makeBackboneCache(d);
    const g2 = zeroCriticGrads(d);
    // Manual: want dL/dlogit = genSign * (-1) for L=-logit when genSign=1...
    // We use genSign as the multiplier on (−logit): L_gen = genSign * (−logit).
    backboneForward(D2, w, d, cache2);
    const z = new Float64Array(K);
    for (let c = 0; c < nCell; c++) {
      for (let k = 0; k < K; k++) z[k] += cache2.soft[k * nCell + c];
    }
    for (let k = 0; k < K; k++) z[k] /= nCell;
    const h = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) {
      let s = w.head[hidden * K + i];
      for (let k = 0; k < K; k++) s += w.head[i * K + k] * z[k];
      h[i] = Math.tanh(s);
    }
    const w1Off = hidden * K + hidden;
    let logit = w.head[w1Off + hidden];
    for (let i = 0; i < hidden; i++) logit += w.head[w1Off + i] * h[i];
    genLoss = opts.genSign * -logit;
    const dLogit = opts.genSign * -1;
    const dh = new Float64Array(hidden);
    for (let i = 0; i < hidden; i++) dh[i] = dLogit * w.head[w1Off + i];
    const dz = new Float64Array(K);
    for (let i = 0; i < hidden; i++) {
      const pre = dh[i] * (1 - h[i] * h[i]);
      for (let k = 0; k < K; k++) dz[k] += pre * w.head[i * K + k];
    }
    const dSoft = new Float64Array(K * nCell);
    for (let k = 0; k < K; k++) {
      const gz = dz[k] / nCell;
      for (let c = 0; c < nCell; c++) dSoft[k * nCell + c] = gz;
    }
    backboneBackwardFromDSoft(dSoft, D2, w, d, cache2, g2);
    const dPos = new Float64Array(B * 2);
    softSplatVjp(pos2, g2.dD, B, G, dPos);
    for (let i = 0; i < B * 2; i++) dF[i] = d.dt * dPos[i];
  }
  return { discLoss, genLoss, dF, discGradPacked, winIdx: null };
}

function stepInpaint(
  pos: Float64Array,
  forces: Float64Array,
  w: PixelDiscWeights,
  d: PixelDiscDims,
  opts: { applyDisc: boolean; genSign: number; lr: number; maskSeed?: number }
): PixelDiscStepResult {
  const B = pos.length / 2;
  const { G, K } = d;
  const nCell = G * G;
  const D = softSplatDensity(pos, B, G);
  const mask = makeInpaintMask(G, opts.maskSeed ?? 0);
  const Din = new Float64Array(nCell);
  let nMask = 0;
  for (let c = 0; c < nCell; c++) {
    Din[c] = mask[c] ? 0 : D[c];
    if (mask[c]) nMask++;
  }
  nMask = Math.max(nMask, 1);

  const cache = makeBackboneCache(d);
  backboneForward(Din, w, d, cache);
  const g = zeroCriticGrads(d);
  const dSoft = new Float64Array(K * nCell);
  const { P, winner, loser, stride } = guessPlan(d);
  const rr = new Float64Array(P);
  const dd = new Float64Array(P);
  // -1 on UNMASKED cells: they carry no residual, so they have no winner to
  // count (§3g). Counting their argmin would report guess 0 as dominant on the
  // 75% of the grid the loss never looks at.
  const winIdx = new Int32Array(nCell).fill(-1);
  let discR = 0;
  for (let c = 0; c < nCell; c++) {
    if (!mask[c]) continue;
    for (let j = 0; j < P; j++) {
      const h = j * stride;
      let pred = w.head[h + K];
      for (let k = 0; k < K; k++) pred += w.head[h + k] * cache.soft[k * nCell + c];
      const diff = pred - D[c];
      dd[j] = diff;
      rr[j] = Math.sqrt(diff * diff + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
    }
    const win = argMinFirst(rr, P);
    winIdx[c] = win;
    for (let j = 0; j < P; j++) {
      const wj = j === win ? winner : loser;
      const h = j * stride;
      discR += wj * rr[j];
      const dPred = ((dd[j] / rr[j]) * wj) / nMask;
      g.head[h + K] += dPred;
      for (let k = 0; k < K; k++) {
        g.head[h + k] += dPred * cache.soft[k * nCell + c];
        dSoft[k * nCell + c] += dPred * w.head[h + k];
      }
    }
  }
  discR /= nMask;
  backboneBackwardFromDSoft(dSoft, Din, w, d, cache, g);
  const discGradPacked = applyDiscGrads(w, d, g, opts.applyDisc ? opts.lr : 0);

  let genLoss = 0;
  const dF = new Float64Array(B * 2);
  if (opts.genSign !== 0) {
    const pos2 = new Float64Array(B * 2);
    for (let i = 0; i < B; i++) {
      pos2[i * 2] = pos[i * 2] + d.dt * forces[i * 2];
      pos2[i * 2 + 1] = pos[i * 2 + 1] + d.dt * forces[i * 2 + 1];
    }
    const D2 = softSplatDensity(pos2, B, G);
    const Din2 = new Float64Array(nCell);
    for (let c = 0; c < nCell; c++) Din2[c] = mask[c] ? 0 : D2[c];
    const cache2 = makeBackboneCache(d);
    backboneForward(Din2, w, d, cache2);
    const g2 = zeroCriticGrads(d);
    const dSoft2 = new Float64Array(K * nCell);
    let genR = 0;
    // L = genSign * R; R on mask vs D2. Grad through D2 (target) and through Din2.
    const dD2 = new Float64Array(nCell);
    for (let c = 0; c < nCell; c++) {
      if (!mask[c]) continue;
      for (let j = 0; j < P; j++) {
        const h = j * stride;
        let pred = w.head[h + K];
        for (let k = 0; k < K; k++) pred += w.head[h + k] * cache2.soft[k * nCell + c];
        const diff = pred - D2[c];
        dd[j] = diff;
        rr[j] = Math.sqrt(diff * diff + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
      }
      const win = argMinFirst(rr, P);
      for (let j = 0; j < P; j++) {
        const wj = j === win ? winner : loser;
        const h = j * stride;
        genR += wj * rr[j];
        const scale = (opts.genSign / rr[j] / nMask) * wj;
        // through target D2
        dD2[c] += -dd[j] * scale;
        // through pred ← Din2 (masked input)
        const dPred = dd[j] * scale;
        for (let k = 0; k < K; k++) dSoft2[k * nCell + c] += dPred * w.head[h + k];
      }
    }
    genR /= nMask;
    genLoss = opts.genSign * genR;
    backboneBackwardFromDSoft(dSoft2, Din2, w, d, cache2, g2);
    // Din2 = D2 on visible cells ⇒ add g2.dD into dD2 where mask=0
    for (let c = 0; c < nCell; c++) {
      if (!mask[c]) dD2[c] += g2.dD[c];
    }
    const dPos = new Float64Array(B * 2);
    softSplatVjp(pos2, dD2, B, G, dPos);
    for (let i = 0; i < B * 2; i++) dF[i] = d.dt * dPos[i];
  }
  return { discLoss: discR, genLoss, dF, discGradPacked, winIdx };
}

/** Cell-center coordinates in normalized [0,1]² for vec-field targets. */
export function cellCenters(G: number): Float64Array {
  const out = new Float64Array(G * G * 2);
  for (let y = 0; y < G; y++) {
    for (let x = 0; x < G; x++) {
      const c = y * G + x;
      out[c * 2] = (x + 0.5) / G;
      out[c * 2 + 1] = (y + 0.5) / G;
    }
  }
  return out;
}

// Compatibility aliases used by older tests / trainer imports
export const softResidual = softResidual2;
export function softResidualGrad(
  yx: number,
  yy: number,
  tx: number,
  ty: number
): { dyx: number; dyy: number; dtx: number; dty: number; r: number } {
  const dx = yx - tx;
  const dy = yy - ty;
  const r = Math.sqrt(dx * dx + dy * dy + PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS);
  const s = 1 / r;
  return { dyx: dx * s, dyy: dy * s, dtx: -dx * s, dty: -dy * s, r };
}
