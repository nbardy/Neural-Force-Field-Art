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
 */

import type { ArtPieceConfig } from "./main";

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------
export type RendererType = "alpha-fade" | "trail-buffer" | "clean";

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
// Factory
// ---------------------------------------------------------------------------
export function createRenderer(
  type: RendererType,
  cfg: ArtPieceConfig,
  particleCount: number,
  spiralPts?: number[][]
): Renderer {
  switch (type) {
    case "alpha-fade":
      return new AlphaFadeRenderer(cfg, spiralPts);
    case "trail-buffer":
      return new TrailBufferRenderer(cfg, particleCount, spiralPts);
    case "clean":
      return new CleanRenderer(cfg, spiralPts);
  }
}
