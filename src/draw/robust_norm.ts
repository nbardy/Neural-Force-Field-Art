/**
 * RobustSpan — drift-tracking, outlier-immune normalisation for a per-particle
 * scalar channel (the adversary's "surprise").
 *
 * WHY NOT A HARDCODED RANGE. Surprise is `‖y − ĝ(u)‖` (variant A) or
 * `min_k ‖y − ĝ_k(u)‖` (variant B). Its magnitude moves over ORDERS of magnitude
 * during a run: large while the predictor is untrained, then collapsing as it
 * fits, then re-inflating whenever the generator finds a new ambiguity. Any fixed
 * [0, C] window is either saturated white for the first 200 frames or flat black
 * for the rest of the run.
 *
 * WHY NOT MIN/MAX. A single particle sitting on a field singularity produces a
 * residual 10^5× the median and squashes every other particle into bin 0. The
 * quantity we want to SEE is the bulk of the distribution, so we track empirical
 * percentiles (default P2 / P50 / P98), not extrema.
 *
 * THE ESTIMATOR (chosen deliberately; alternatives and why not, below):
 *   per frame → stride-subsample S values → sort → exact empirical P2/P50/P98
 *             → EMA-blend into the running span with rate λ.
 *
 *   - Stride subsampling is unbiased here because particle INDEX is uncorrelated
 *     with particle STATE: initial positions are independent `Math.random()` draws
 *     and every respawn is a stateless `pcg(index ^ seed·const)` hash
 *     (src/render/webgpu/advect.ts:429-438, advect_wgsl.ts:728-734). There is no
 *     spatial sort anywhere in the repo, so index order carries no structure.
 *   - Sorting 1024 floats is ~10⁴ comparisons — free next to N draw calls, and it
 *     makes the per-frame percentile EXACT rather than an approximation.
 *   - The EMA (λ = 0.06 ≈ 17-frame time constant) is what handles drift. A P²
 *     (Jain–Chlamtac) streaming quantile was the other candidate and was REJECTED:
 *     P² converges to the quantile of the ENTIRE history, so after 2000 frames it
 *     is pinned to the early, untrained scale and stops tracking. An EMA over
 *     exact per-frame percentiles forgets at a controlled rate, which is the
 *     behaviour we actually want.
 *
 * THE ANTI-FAKE-SIGNAL FLOOR — the most important line in this file.
 * When the adversary collapses (variant A is EXPECTED to collapse), every
 * residual goes to ~0 and the P2..P98 spread goes to ~0 with it. A normaliser
 * that divides by the observed spread would then rescale pure numerical noise
 * across the full colour ramp and paint a gorgeous, entirely fictional picture of
 * a working adversary. So the denominator is floored by an ABSOLUTE constant
 * (SPAN_FLOOR, in surprise units): below it the whole cloud maps to one end of
 * the ramp and READS AS COLLAPSED, which is the truth. `collapsed` exposes the
 * same fact as a boolean and the renderer prints the raw lo/hi so the reader can
 * audit the normalisation instead of trusting it.
 */

/** Robust summary of a scalar channel. All three in RAW surprise units. */
export interface Span {
  /** low percentile (default P2) */
  lo: number;
  /** median (P50) — the pivot for diverging colormaps */
  mid: number;
  /** high percentile (default P98) */
  hi: number;
}

/**
 * Absolute floor on the normalisation denominator, in surprise units.
 *
 * Surprise is a distance in NORMALISED position space (positions are divided by
 * the resolution before the predictor sees them, see map_rollout-advect §3), so
 * a P2..P98 spread below 1e-6 means the whole cloud agrees to within a
 * millionth of the canvas — i.e. the predictor has fully absorbed the field and
 * there is no disagreement left to draw. Raising this makes collapse MORE
 * visible; lowering it lets numerical noise be stretched into a fake rainbow.
 */
export const SPAN_FLOOR = 1e-6;

export interface RobustSpanOpts {
  /** max values sorted per frame (default 1024) */
  sample?: number;
  /** EMA rate, 0<λ≤1 (default 0.06 ≈ 17-frame time constant) */
  lambda?: number;
  /** low quantile in [0,1) (default 0.02) */
  loQ?: number;
  /** high quantile in (0,1] (default 0.98) */
  hiQ?: number;
}

/** Exact empirical quantile of an ASCENDING-sorted array, linear interpolation. */
function quantileSorted(sorted: Float64Array, len: number, q: number): number {
  const pos = q * (len - 1);
  const i = Math.floor(pos);
  const j = Math.min(len - 1, i + 1);
  const f = pos - i;
  return sorted[i] * (1 - f) + sorted[j] * f;
}

export class RobustSpan {
  private readonly cap: number;
  private readonly lambda: number;
  private readonly loQ: number;
  private readonly hiQ: number;
  private readonly buf: Float64Array;

  private _span: Span = { lo: 0, mid: 0, hi: 0 };
  private _raw: Span = { lo: 0, mid: 0, hi: 0 };
  private _updates = 0;
  private _rejected = 0;

  constructor(opts: RobustSpanOpts = {}) {
    const cap = opts.sample ?? 1024;
    const lambda = opts.lambda ?? 0.06;
    const loQ = opts.loQ ?? 0.02;
    const hiQ = opts.hiQ ?? 0.98;
    // Ingestion-time validation (κ). Everything downstream assumes these hold,
    // so they are checked ONCE, loudly, instead of defended at every use site.
    if (!(cap >= 2)) throw new Error(`RobustSpan: sample must be >= 2, got ${cap}`);
    if (!(lambda > 0 && lambda <= 1))
      throw new Error(`RobustSpan: lambda must be in (0,1], got ${lambda}`);
    if (!(loQ >= 0 && loQ < hiQ && hiQ <= 1))
      throw new Error(`RobustSpan: need 0 <= loQ < hiQ <= 1, got ${loQ}..${hiQ}`);
    this.cap = cap;
    this.lambda = lambda;
    this.loQ = loQ;
    this.hiQ = hiQ;
    this.buf = new Float64Array(cap);
  }

  /** EMA-smoothed span — what the colormap should use. */
  get span(): Span {
    return this._span;
  }

  /** This frame's UNSMOOTHED percentile estimate — for the HUD / debugging. */
  get raw(): Span {
    return this._raw;
  }

  /** Frames that contributed at least one finite sample. */
  get updates(): number {
    return this._updates;
  }

  /**
   * Non-finite values seen since construction. NaN in the surprise channel means
   * the adversary's loss has blown up (the classic sqrt(0)-gradient incident,
   * see src/core/losses/chaos.ts:50-55). Sorting NaNs would silently corrupt the
   * percentiles, so they are DROPPED and COUNTED — never averaged in, never
   * silently ignored. A nonzero value here should be surfaced by the caller.
   */
  get rejected(): number {
    return this._rejected;
  }

  /** True when the bulk spread has fallen below the absolute floor. */
  get collapsed(): boolean {
    return this._span.hi - this._span.lo < SPAN_FLOOR;
  }

  /**
   * Fold one frame's values in. `n` values are read from `values[0..n-1]`;
   * at most `cap` of them are kept, stride-selected.
   *
   * If a frame contains ZERO finite samples the span is left exactly as it was:
   * that is "no new information", not a fallback — no value is invented, and the
   * `rejected` counter records that it happened.
   */
  update(values: Float32Array, n: number): void {
    const count = Math.min(n, values.length);
    const step = Math.max(1, Math.floor(count / this.cap));
    const buf = this.buf;
    let len = 0;
    for (let i = 0; i < count && len < this.cap; i += step) {
      const v = values[i];
      if (Number.isFinite(v)) buf[len++] = v;
      else this._rejected++;
    }
    if (len === 0) return;

    const view = buf.subarray(0, len);
    view.sort();
    const lo = quantileSorted(buf, len, this.loQ);
    const mid = quantileSorted(buf, len, 0.5);
    const hi = quantileSorted(buf, len, this.hiQ);
    this._raw = { lo, mid, hi };

    // First frame seeds directly — an EMA from a 0-initialised state would spend
    // its whole time constant crawling up from zero and paint the opening
    // seconds of every run as uniformly saturated.
    if (this._updates === 0) {
      this._span = { lo, mid, hi };
    } else {
      const l = this.lambda;
      const s = this._span;
      this._span = {
        lo: s.lo + l * (lo - s.lo),
        mid: s.mid + l * (mid - s.mid),
        hi: s.hi + l * (hi - s.hi),
      };
    }
    this._updates++;
  }
}
