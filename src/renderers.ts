/**
 * Pluggable renderers for the particle art engine.
 *
 * "alpha-fade"   — Paste previous frame faded + draw new particles on top.
 *                  One full-canvas alpha blend (hardware-accelerated by the
 *                  browser compositor even on Canvas 2D). Ghost trails are
 *                  intentional — they never fully fade due to 8-bit alpha
 *                  quantisation, leaving permanent traces that build up
 *                  into the art.
 *
 * "trail-buffer" — Ring buffer of last N positions, full clear each frame.
 *                  No ghost artifacts, precise trail length. Higher memory
 *                  but only touches pixels where particles actually are.
 *
 * "clean"        — Full clear + current particles only. Fastest possible,
 *                  no trails. Good for debugging or fast iteration.
 *
 * "surprise"     — INSTRUMENT, not an effect. Colours each particle by a
 *                  per-particle scalar pulled from a {@link SurpriseSource} —
 *                  the adversary's residual ‖y − ĝ(u)‖ (variant A) or
 *                  min_k‖y − ĝ_k(u)‖ (variant B) — through a perceptual
 *                  colormap with a robust, drift-tracking normalisation.
 *                  Deliberately has NO trails: a faded trail shows a STALE
 *                  surprise value, and the whole point of this mode is that
 *                  variant A's collapse and variant B's persistence are read
 *                  off the screen instead of guessed at.
 */

import type { ArtPieceConfig } from "./main";
import { COLORMAPS, type ColormapName, type RGB } from "./draw/colormap";
import { RobustSpan } from "./draw/robust_norm";

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------
export type RendererType = "alpha-fade" | "trail-buffer" | "clean" | "surprise";

export interface Renderer {
  render(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    positions: number[][],
    velocities: number[][],
    frame: number
  ): void;
  destroy(): void;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Default dot radius in px. Overridable per-piece via cfg.pointSize.
const DEFAULT_POINT_SIZE = 1.3;

// Multiplier for the soft-glow disc drawn under each core dot, and its alpha.
// Cheap 2-pass round splat — a larger, low-alpha arc under a crisp core arc.
// No per-particle gradient objects (those allocate + are slow); just two fills.
const GLOW_RADIUS_MULT = 2.1;
const GLOW_ALPHA = 0.22;

/**
 * Resolve the dot radius for a piece. `pointSize` is an OPTIONAL knob that is
 * not declared on {@link ArtPieceConfig} (renderers.ts must not edit main.ts),
 * so we read it through a narrow structural cast. Absent → DEFAULT_POINT_SIZE.
 */
function pointRadius(cfg: ArtPieceConfig): number {
  const size = (cfg as { pointSize?: number }).pointSize;
  return typeof size === "number" && size > 0 ? size : DEFAULT_POINT_SIZE;
}

// Draw particles as small round soft dots. Velocity → colour (unchanged).
// Each dot = a low-alpha glow disc + a crisp core disc, both round arcs.
function drawParticles(
  ctx: CanvasRenderingContext2D,
  positions: number[][],
  velocities: number[][],
  radius: number
) {
  const TAU = Math.PI * 2;
  const glowR = radius * GLOW_RADIUS_MULT;
  for (let i = 0; i < positions.length; i++) {
    const x = positions[i][0];
    const y = positions[i][1];
    const vx = velocities[i][0];
    const vy = velocities[i][1];
    const speed = Math.sqrt(vx * vx + vy * vy);
    const r = Math.min(255, 80 + Math.abs(vx) * 60) | 0;
    const g = Math.min(255, 40 + Math.abs(vy) * 60) | 0;
    const b = Math.min(255, 120 + speed * 40) | 0;

    // Soft glow: larger, low-alpha disc underneath (baked into the colour so we
    // never touch globalAlpha — one less state write per particle).
    ctx.fillStyle = `rgba(${r},${g},${b},${GLOW_ALPHA})`;
    ctx.beginPath();
    ctx.arc(x, y, glowR, 0, TAU);
    ctx.fill();

    // Crisp core dot on top.
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, TAU);
    ctx.fill();
  }
}

function drawHUD(
  ctx: CanvasRenderingContext2D,
  name: string,
  frame: number,
  n: number
) {
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.font = "12px monospace";
  ctx.fillText(`${name}  frame ${frame}  particles ${n}`, 8, 16);
}

function drawSpiralOverlay(
  ctx: CanvasRenderingContext2D,
  pts: number[][]
) {
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    if (i === 0) ctx.moveTo(pts[i][0], pts[i][1]);
    else ctx.lineTo(pts[i][0], pts[i][1]);
  }
  ctx.strokeStyle = "rgba(100,60,180,0.12)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Alpha-Fade Renderer
//
// How it works: each frame we draw a semi-transparent rect of the background
// colour over the ENTIRE canvas. This blends the old content toward the bg,
// creating a fade. Then we stamp new particles on top. The canvas itself acts
// as the "old buffer" — no extra allocations.
//
// Speed: one hardware-composited fillRect (the browser uses the GPU for
// canvas compositing even in 2D mode) + N small fillRects for particles.
// On a GPU-equipped machine the full-canvas blend is essentially free.
//
// Ghost trails: 8-bit alpha at low values (0.03–0.08) quantises to 0 change
// per frame for faint pixels, so old trails never fully vanish. This is the
// desired artistic effect — persistent traces that accumulate over time.
// ---------------------------------------------------------------------------
class AlphaFadeRenderer implements Renderer {
  private bgStr: string;
  private fadeStr: string;
  private cfg: ArtPieceConfig;
  private spiralPts?: number[][];
  private firstFrame = true;

  constructor(cfg: ArtPieceConfig, spiralPts?: number[][]) {
    this.cfg = cfg;
    this.spiralPts = spiralPts;
    const [r, g, b] = cfg.backgroundColor;
    this.bgStr = `rgb(${r},${g},${b})`;
    this.fadeStr = `rgba(${r},${g},${b},${cfg.alphaBlend})`;
  }

  render(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    positions: number[][],
    velocities: number[][],
    frame: number
  ) {
    if (this.firstFrame) {
      ctx.fillStyle = this.bgStr;
      ctx.fillRect(0, 0, w, h);
      if (this.spiralPts) drawSpiralOverlay(ctx, this.spiralPts);
      this.firstFrame = false;
    }

    // Fade existing content toward background
    ctx.fillStyle = this.fadeStr;
    ctx.fillRect(0, 0, w, h);

    drawParticles(ctx, positions, velocities, pointRadius(this.cfg));
    drawHUD(ctx, this.cfg.name, frame, positions.length);
  }

  destroy() {}
}

// ---------------------------------------------------------------------------
// Trail-Buffer Renderer
//
// Stores last TRAIL_LEN frames of positions in a flat Float32Array ring
// buffer. Each render: full opaque clear → draw trail dots oldest-to-newest
// with decreasing opacity → draw current particles.
//
// Touches only N × TRAIL_LEN pixels (where particles are) instead of the
// full canvas. No ghost artifacts. Precise trail length control.
// ---------------------------------------------------------------------------
const TRAIL_LEN = 20;

class TrailBufferRenderer implements Renderer {
  private trail: Float32Array;
  private head = 0;
  private n: number;
  private bgStr: string;
  private cfg: ArtPieceConfig;
  private spiralPts?: number[][];

  constructor(cfg: ArtPieceConfig, particleCount: number, spiralPts?: number[][]) {
    this.n = particleCount;
    this.cfg = cfg;
    this.spiralPts = spiralPts;
    this.trail = new Float32Array(TRAIL_LEN * particleCount * 2);
    const [r, g, b] = cfg.backgroundColor;
    this.bgStr = `rgb(${r},${g},${b})`;
  }

  render(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    positions: number[][],
    velocities: number[][],
    frame: number
  ) {
    // Push current positions into ring buffer
    const off = this.head * this.n * 2;
    for (let i = 0; i < this.n; i++) {
      this.trail[off + i * 2] = positions[i][0];
      this.trail[off + i * 2 + 1] = positions[i][1];
    }
    this.head = (this.head + 1) % TRAIL_LEN;

    // Full opaque clear
    ctx.fillStyle = this.bgStr;
    ctx.fillRect(0, 0, w, h);

    if (this.spiralPts) drawSpiralOverlay(ctx, this.spiralPts);

    // Trail dots — one globalAlpha change per age level
    const trailFrames = Math.min(frame, TRAIL_LEN);
    ctx.fillStyle = "rgb(140,110,220)";
    for (let age = trailFrames - 1; age >= 0; age--) {
      const slot = ((this.head - 1 - age) + TRAIL_LEN * 100) % TRAIL_LEN;
      const sOff = slot * this.n * 2;
      ctx.globalAlpha = ((trailFrames - age) / trailFrames) * 0.35;
      for (let i = 0; i < this.n; i++) {
        ctx.fillRect(
          this.trail[sOff + i * 2] - 0.5,
          this.trail[sOff + i * 2 + 1] - 0.5,
          1.5, 1.5
        );
      }
    }
    ctx.globalAlpha = 1;

    drawParticles(ctx, positions, velocities, pointRadius(this.cfg));
    drawHUD(ctx, this.cfg.name, frame, positions.length);
  }

  destroy() {
    this.trail = new Float32Array(0);
  }
}

// ---------------------------------------------------------------------------
// Clean Renderer — no trails, just current frame. Fastest.
// ---------------------------------------------------------------------------
class CleanRenderer implements Renderer {
  private bgStr: string;
  private cfg: ArtPieceConfig;
  private spiralPts?: number[][];

  constructor(cfg: ArtPieceConfig, spiralPts?: number[][]) {
    this.cfg = cfg;
    this.spiralPts = spiralPts;
    const [r, g, b] = cfg.backgroundColor;
    this.bgStr = `rgb(${r},${g},${b})`;
  }

  render(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    positions: number[][],
    velocities: number[][],
    frame: number
  ) {
    ctx.fillStyle = this.bgStr;
    ctx.fillRect(0, 0, w, h);
    if (this.spiralPts) drawSpiralOverlay(ctx, this.spiralPts);
    drawParticles(ctx, positions, velocities, pointRadius(this.cfg));
    drawHUD(ctx, this.cfg.name, frame, positions.length);
  }

  destroy() {}
}

// ---------------------------------------------------------------------------
// Surprise Renderer — per-particle scalar → perceptual colormap
//
// PERF, HONESTLY — measured, not guessed. This is the Canvas2D path and it IS a
// per-particle CPU round-trip: `SurpriseSource.read(n)` hands back N floats, so
// whoever computes surprise on the GPU must map N×4 bytes back every frame.
//
// Two separate costs, and only one of them is this file's:
//   (a) THE COLOURING LOOP (normalise + level + 2 fillStyle writes per particle,
//       measured under bun with a stub context, M4): 0.06 ms at N=4096, 0.44 ms
//       at 65k, 1.00 ms at 262k. Linear and cheap — the CSS-string cache below is
//       what makes it so; it was the dominant cost before.
//   (b) THE READBACK + RASTERISATION, which are NOT cheap. N `arc()` calls put
//       Canvas2D's practical ceiling in the low tens of thousands of particles,
//       and `mapAsync` on the surprise buffer is a synchronisation point that
//       costs a frame of latency regardless of size. The shipped hot path runs
//       ~1e6 particles through a GPU splat precisely to avoid both.
// So: this renderer is honest at the particle counts where you sit and READ the
// instrument (thousands), and is not a substitute for the fused path at 1e6.
//
// The fused alternative already exists and needs no refactor of THIS file:
// src/render/webgpu/surprise_points.ts renders straight from a GPU
// `array<f32>` surprise buffer with the same colormap (the WGSL LUT is
// generated from src/draw/colormap.ts, so the two paths agree by construction),
// and src/render/webgpu/surprise_points.ts::GpuSurpriseStats feeds the SAME
// RobustSpan from a 4 KB subsample every 8th frame instead of the full buffer.
// What it needs from upstream is exactly one thing: the adversary must write its
// per-particle residual into a GPUBuffer of N f32 (today no such buffer exists —
// the Adversary (src/core/gan/adversary.ts) is pure tfjs and evaluates a
// sampled batch, not the cloud).
// Until that buffer exists this CPU path is the only way to see the signal, and
// at the particle counts where the adversary is interesting it is fast enough.
// ---------------------------------------------------------------------------

/**
 * Per-particle scalar channel. `read(n)` returns at least `n` RAW (unnormalised)
 * values — normalisation belongs to the renderer, which is the only place that
 * knows the display's dynamic range. Implementations must return a value for
 * every particle; "not computed yet" is the source's problem to represent (as
 * zeros), not a case the renderer branches on.
 */
export interface SurpriseSource {
  read(n: number): Float32Array;
}

export interface SurpriseOpts {
  source: SurpriseSource;
  colormap: ColormapName;
}

/** Quantisation of the ramp into cached CSS colour strings. 64 levels is below
 *  the JND for these ramps and removes ALL per-particle string allocation —
 *  building `rgba(...)` per particle was the single biggest cost in the naive
 *  version. */
const CSS_LEVELS = 64;

/**
 * One extra slot past the ramp, painted a colour NO colormap here produces
 * (pure magenta), reserved for non-finite surprise.
 *
 * BOUNDARY DEFENCE, not decoration. A NaN residual means the adversary's loss
 * blew up (the sqrt(0)-gradient incident, src/core/losses/chaos.ts:50-55). Without
 * this slot `Math.round(NaN)` indexes the ramp array out of bounds, `fillStyle`
 * is assigned `undefined`, Canvas2D IGNORES the assignment, and the particle
 * silently keeps the previous particle's colour — a blown-up run that looks
 * completely healthy. Magenta makes it unmissable.
 */
const NAN_LEVEL = CSS_LEVELS;
const NAN_RGB: RGB = [255, 0, 255];

function cssRamp(colormap: ColormapName, alpha: number): string[] {
  const cm = COLORMAPS[colormap];
  const out: string[] = new Array(CSS_LEVELS + 1);
  for (let i = 0; i < CSS_LEVELS; i++) {
    const c: RGB = cm.ramp(i / (CSS_LEVELS - 1));
    out[i] =
      alpha >= 1
        ? `rgb(${c[0]},${c[1]},${c[2]})`
        : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
  }
  out[NAN_LEVEL] =
    alpha >= 1
      ? `rgb(${NAN_RGB[0]},${NAN_RGB[1]},${NAN_RGB[2]})`
      : `rgba(${NAN_RGB[0]},${NAN_RGB[1]},${NAN_RGB[2]},${alpha})`;
  return out;
}

class SurpriseRenderer implements Renderer {
  private readonly bgStr: string;
  private readonly cfg: ArtPieceConfig;
  private readonly spiralPts?: number[][];
  private readonly source: SurpriseSource;
  private readonly colormap: ColormapName;
  private readonly core: string[];
  private readonly glow: string[];
  private readonly norm = new RobustSpan();

  constructor(cfg: ArtPieceConfig, opts: SurpriseOpts, spiralPts?: number[][]) {
    this.cfg = cfg;
    this.spiralPts = spiralPts;
    this.source = opts.source;
    this.colormap = opts.colormap;
    this.core = cssRamp(opts.colormap, 1);
    this.glow = cssRamp(opts.colormap, GLOW_ALPHA);
    const [r, g, b] = cfg.backgroundColor;
    this.bgStr = `rgb(${r},${g},${b})`;
  }

  render(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    positions: number[][],
    _velocities: number[][],
    frame: number
  ) {
    const n = positions.length;
    const values = this.source.read(n);
    // Contract check at the one place raw values enter (κ). A short array would
    // otherwise read `undefined` past the end, quietly paint the tail magenta,
    // and look like an adversary blow-up instead of a plumbing bug.
    if (values.length < n) {
      throw new Error(
        `SurpriseSource.read(${n}) returned ${values.length} values — ` +
          `the channel must cover every rendered particle.`
      );
    }
    this.norm.update(values, n);
    const span = this.norm.span;
    const cm = COLORMAPS[this.colormap];

    ctx.fillStyle = this.bgStr;
    ctx.fillRect(0, 0, w, h);
    if (this.spiralPts) drawSpiralOverlay(ctx, this.spiralPts);

    const TAU = Math.PI * 2;
    const radius = pointRadius(this.cfg);
    const glowR = radius * GLOW_RADIUS_MULT;
    const top = CSS_LEVELS - 1;
    for (let i = 0; i < n; i++) {
      const x = positions[i][0];
      const y = positions[i][1];
      // `t >= 0` is FALSE for NaN — that comparison is the whole guard, and it
      // canonicalises a foreign float into the level domain without a second
      // pass over the data. See NAN_LEVEL.
      const t = cm.position(span, values[i]);
      const level = t >= 0 ? Math.round(t * top) : NAN_LEVEL;

      ctx.fillStyle = this.glow[level];
      ctx.beginPath();
      ctx.arc(x, y, glowR, 0, TAU);
      ctx.fill();

      ctx.fillStyle = this.core[level];
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, TAU);
      ctx.fill();
    }

    this.drawLegend(ctx, h, span.lo, span.hi);
    drawHUD(ctx, this.cfg.name, frame, n);
    this.drawReadout(ctx);
  }

  /** Colour bar + the RAW percentile endpoints. Printing the endpoints is what
   *  makes the normalisation auditable — a rainbow with lo=1e-9, hi=1.2e-9 is a
   *  picture of numerical noise, and the reader can now see that. */
  private drawLegend(
    ctx: CanvasRenderingContext2D,
    h: number,
    lo: number,
    hi: number
  ) {
    const x0 = 8;
    const y0 = h - 28;
    const barW = 160;
    const barH = 8;
    const step = barW / CSS_LEVELS;
    for (let i = 0; i < CSS_LEVELS; i++) {
      ctx.fillStyle = this.core[i];
      ctx.fillRect(x0 + i * step, y0, step + 1, barH);
    }
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "10px monospace";
    ctx.fillText(fmt(lo), x0, y0 + barH + 11);
    ctx.textAlign = "right";
    ctx.fillText(fmt(hi), x0 + barW, y0 + barH + 11);
    ctx.textAlign = "left";
  }

  private drawReadout(ctx: CanvasRenderingContext2D) {
    const s = this.norm.span;
    const collapsed = this.norm.collapsed;
    ctx.font = "12px monospace";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fillText(
      `surprise[${this.colormap}] p2=${fmt(s.lo)} p50=${fmt(s.mid)} p98=${fmt(s.hi)}`,
      8,
      32
    );
    if (collapsed) {
      // Not a warning about a bug — this is the EXPECTED reading for variant A,
      // and the label is here so a flat frame is never mistaken for a broken one.
      ctx.fillStyle = "rgba(255,120,120,0.85)";
      ctx.fillText("ADVERSARY COLLAPSED (span < floor — colour is not meaningful)", 8, 48);
    }
    if (this.norm.rejected > 0) {
      ctx.fillStyle = "rgba(255,200,120,0.85)";
      ctx.fillText(`non-finite surprise dropped: ${this.norm.rejected}`, 8, 64);
    }
  }

  destroy() {}
}

function fmt(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e-3 && a < 1e4 ? v.toFixed(4) : v.toExponential(2);
}

// ---------------------------------------------------------------------------
// Factory
//
// The `surprise` case needs data the other three do not. Rather than thread an
// optional scalar channel through every renderer (and make each one ask "do I
// have one?"), the requirement is enforced ONCE here, at the ingestion boundary:
// no source → typed error, never a silent fall back to a different renderer.
// Past that point SurpriseRenderer holds a non-optional source and has a single
// straight-line path.
// ---------------------------------------------------------------------------
export function createRenderer(
  type: RendererType,
  cfg: ArtPieceConfig,
  particleCount: number,
  spiralPts?: number[][],
  surprise?: SurpriseOpts
): Renderer {
  switch (type) {
    case "alpha-fade":
      return new AlphaFadeRenderer(cfg, spiralPts);
    case "trail-buffer":
      return new TrailBufferRenderer(cfg, particleCount, spiralPts);
    case "clean":
      return new CleanRenderer(cfg, spiralPts);
    case "surprise":
      if (!surprise) {
        throw new Error(
          `createRenderer("surprise") requires a SurpriseOpts { source, colormap } — ` +
            `piece "${cfg.name}" declared the surprise renderer with no scalar channel.`
        );
      }
      return new SurpriseRenderer(cfg, surprise, spiralPts);
  }
}
