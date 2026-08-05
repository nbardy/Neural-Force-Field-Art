/**
 * Perceptual colormaps for scalar particle channels — pure functions, no DOM,
 * no GPU. This module is the SINGLE SOURCE OF TRUTH for the surprise colouring:
 * the Canvas2D renderer samples `ramp()` directly, and the WGSL renderer emits a
 * LUT generated from the same `ramp()` (src/render/webgpu/surprise_wgsl.ts), so
 * the CPU and GPU paths cannot drift apart. tools/surprise_test.ts pins the
 * agreement numerically.
 *
 * `viridis` and `inferno` are the standard matplotlib ramps, reproduced by the
 * well-known degree-6 polynomial fits (Zucker, shadertoy WlfXRN). Verified
 * against the published anchor colours in tools/surprise_test.ts — viridis(0) =
 * (68,1,84), viridis(1) = (253,231,37), inferno(1) = (252,255,164) — because a
 * single mistyped coefficient produces a map that still LOOKS plausible while
 * being non-monotone in luminance, which is exactly the failure a colormap is
 * supposed to prevent.
 *
 * `coolwarm` is Moreland's bent cool–warm diverging ramp, piecewise-linear
 * through its published control points. It is the map to use when the channel is
 * SIGNED relative to a pivot (e.g. "which particles surprise the predictor MORE
 * than the median particle") — a sequential map cannot show a sign.
 *
 * POLARITY IS DATA, NOT A BRANCH. Each colormap carries its own
 * `position(span, v)` (and the matching WGSL expression), so the render path is
 * always the single line `ramp(position(span, v))` with no "is this diverging?"
 * test anywhere downstream.
 */

import { SPAN_FLOOR, type Span } from "./robust_norm";

export type RGB = readonly [number, number, number];

export type ColormapName = "inferno" | "viridis" | "coolwarm";

export interface Colormap {
  readonly name: ColormapName;
  /** Ramp colour at t ∈ [0,1] → sRGB bytes. t is clamped. */
  ramp(t: number): RGB;
  /** Raw value + robust span → ramp position in [0,1]. */
  position(span: Span, v: number): number;
  /**
   * The WGSL form of {@link position}, as an expression over `v` and the
   * uniform fields `u.lo`, `u.mid`, `u.hi`. Kept next to the TS version so the
   * two are edited together; tools/surprise_test.ts checks the emitted shader
   * still contains it.
   */
  readonly wgslPosition: string;
}

const clamp01 = (t: number): number => (t < 0 ? 0 : t > 1 ? 1 : t);

const byte = (x: number): number => {
  const v = Math.round(clamp01(x) * 255);
  return v;
};

// --- polynomial ramps -------------------------------------------------------
// c[i] = the vec3 coefficient of t^i. Evaluated by Horner-free ascending powers
// to match the WGSL reference implementations these were lifted from.
type Poly = readonly (readonly [number, number, number])[];

const VIRIDIS_POLY: Poly = [
  [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
  [0.1050930431085774, 1.404613529898575, 1.384590162594685],
  [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
  [-4.634230498983486, -5.799100973351585, -19.33244095627987],
  [6.228269936347081, 14.17993336680509, 56.69055260068105],
  [4.776384997670288, -13.74514537774601, -65.35303263337234],
  [-5.435455855934631, 4.645852612178535, 26.3124352495832],
];

const INFERNO_POLY: Poly = [
  [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
  [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
  [11.60249308247187, -3.972853965665698, -15.9423941062914],
  [-41.70399613139459, 17.43639888205313, 44.35414519872813],
  [77.16296278894483, -33.40235894210092, -81.80730925738993],
  [-71.31942824499214, 32.62606426397723, 73.20951985803202],
  [25.13112622477341, -12.24266895238567, -23.07032500287172],
];

// The fits overshoot slightly outside the data range near the endpoints, so the
// clamp in `byte` is load-bearing, not defensive: without it t≈1 emits >255.
function evalPoly(c: Poly, t: number): RGB {
  const x = clamp01(t);
  let p = 1;
  let r = 0;
  let g = 0;
  let b = 0;
  for (let i = 0; i < c.length; i++) {
    r += c[i][0] * p;
    g += c[i][1] * p;
    b += c[i][2] * p;
    p *= x;
  }
  return [byte(r), byte(g), byte(b)];
}

// --- piecewise-linear ramp --------------------------------------------------
// Moreland bent cool–warm, the ParaView/matplotlib "coolwarm" control points.
const COOLWARM_STOPS: readonly RGB[] = [
  [59, 76, 192],
  [98, 130, 234],
  [141, 176, 254],
  [184, 208, 249],
  [221, 221, 221],
  [245, 196, 173],
  [244, 154, 123],
  [222, 96, 77],
  [180, 4, 38],
];

function evalStops(stops: readonly RGB[], t: number): RGB {
  const x = clamp01(t) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

// --- polarity ---------------------------------------------------------------
// Both denominators are floored by SPAN_FLOOR *in raw surprise units*. See the
// long note in robust_norm.ts: this is what stops a collapsed adversary from
// being rescaled into a convincing-looking rainbow.

/** Sequential: lo → 0, hi → 1. */
function unipolar(span: Span, v: number): number {
  return clamp01((v - span.lo) / Math.max(span.hi - span.lo, SPAN_FLOOR));
}

/** Diverging: mid → 0.5, symmetric half-width so the pivot never drifts. */
function bipolar(span: Span, v: number): number {
  const half = Math.max(span.hi - span.mid, span.mid - span.lo, SPAN_FLOOR);
  const s = (v - span.mid) / half;
  return 0.5 + 0.5 * (s < -1 ? -1 : s > 1 ? 1 : s);
}

// DERIVED, never retyped: a WGSL literal that drifts from SPAN_FLOOR would make
// the GPU renderer stretch a collapsed adversary while the CPU one refuses to —
// the same class of latent, test-invisible bug as the f32lit exponent incident
// (src/render/webgpu/ad/emit_wgsl.ts:16-27). `toExponential()` always yields
// exponent form ("1e-6"), which WGSL types as f32 without needing a point.
const FLOOR_WGSL = SPAN_FLOOR.toExponential();

const UNIPOLAR_WGSL = `clamp((v - u.lo) / max(u.hi - u.lo, ${FLOOR_WGSL}), 0.0, 1.0)`;
const BIPOLAR_WGSL =
  `0.5 + 0.5 * clamp((v - u.mid) / ` +
  `max(max(u.hi - u.mid, u.mid - u.lo), ${FLOOR_WGSL}), -1.0, 1.0)`;

// --- the table (this IS the dispatcher: one lookup, zero branching) ---------
export const COLORMAPS: Record<ColormapName, Colormap> = {
  inferno: {
    name: "inferno",
    ramp: (t) => evalPoly(INFERNO_POLY, t),
    position: unipolar,
    wgslPosition: UNIPOLAR_WGSL,
  },
  viridis: {
    name: "viridis",
    ramp: (t) => evalPoly(VIRIDIS_POLY, t),
    position: unipolar,
    wgslPosition: UNIPOLAR_WGSL,
  },
  coolwarm: {
    name: "coolwarm",
    ramp: (t) => evalStops(COOLWARM_STOPS, t),
    position: bipolar,
    wgslPosition: BIPOLAR_WGSL,
  },
};

/**
 * Default stop count for emitted LUTs.
 *
 * 33 is chosen from a measurement, not a guess (tools/surprise_test.ts §3 prints
 * the numbers; the sweep was 17/33/65/129 stops):
 *   - MEAN error vs the exact ramp is 0.70/255 (inferno), 0.54 (viridis),
 *     0.40 (coolwarm) — i.e. sub-quantisation everywhere.
 *   - MAX error is 4/255 for inferno at t ≈ 0.005 and does NOT improve at 65
 *     stops, because it is not interpolation error: inferno's polynomial fit
 *     goes slightly NEGATIVE in blue near t=0, so the exact ramp has a hard
 *     clamp KINK at the black end that no piecewise-linear LUT can follow.
 *     Concretely it is #000004 vs #000407. Doubling the table buys nothing.
 * So: more stops would be cargo cult. The test pins both the mean and that max.
 */
export const LUT_STOPS = 33;

/**
 * Sample `cm` into `stops` evenly spaced RGB triples in [0,1] float form —
 * the representation the WGSL renderer embeds as a `const` array and the
 * Canvas2D renderer bakes into its CSS-colour cache.
 */
export function rampLUT(cm: Colormap, stops: number = LUT_STOPS): Float32Array {
  if (!(stops >= 2)) throw new Error(`rampLUT: need >= 2 stops, got ${stops}`);
  const out = new Float32Array(stops * 3);
  for (let i = 0; i < stops; i++) {
    const c = cm.ramp(i / (stops - 1));
    out[i * 3] = c[0] / 255;
    out[i * 3 + 1] = c[1] / 255;
    out[i * 3 + 2] = c[2] / 255;
  }
  return out;
}

/**
 * Linear interpolation into a {@link rampLUT} — the exact arithmetic the emitted
 * WGSL performs. Kept here so the test suite can prove the GPU path reproduces
 * `cm.ramp()` without a GPU.
 */
export function sampleLUT(lut: Float32Array, t: number): RGB {
  const stops = lut.length / 3;
  const x = clamp01(t) * (stops - 1);
  const i = Math.min(stops - 2, Math.floor(x));
  const f = x - i;
  const a = i * 3;
  const b = (i + 1) * 3;
  return [
    Math.round((lut[a] + (lut[b] - lut[a]) * f) * 255),
    Math.round((lut[a + 1] + (lut[b + 1] - lut[a + 1]) * f) * 255),
    Math.round((lut[a + 2] + (lut[b + 2] - lut[a + 2]) * f) * 255),
  ];
}
