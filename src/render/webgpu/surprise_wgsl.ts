/**
 * WGSL codegen for the surprise point renderer — pure string generation, no
 * tfjs, no device, so it is fully testable under `bun` (tools/surprise_test.ts).
 * Mirrors the advect_wgsl.ts / advect.ts split used everywhere else in this repo.
 *
 * The colour ramp is NOT hand-written here. It is emitted as a `const` array
 * sampled from src/draw/colormap.ts, and the value→ramp-position expression is
 * taken verbatim from that colormap's `wgslPosition`. One source of truth: the
 * GPU cannot drift from the Canvas2D path, and the test proves the emitted LUT
 * reproduces `cm.ramp()` to within a byte.
 *
 * Bindings (group 0):
 *   0  uniform  Uni { resolution, pointSize, lo, mid, hi, gain, surpriseOffset }
 *   1  storage  posBuf : array<f32>   — interleaved xy PIXELS (AdvectKernel.posBuffer)
 *   2  storage  surBuf : array<f32>   — packed N-float surprise planes
 *
 * posBuf is bound exactly as points.ts binds it, so the same
 * `AdvectKernel.posBuffer` can feed either renderer with no repacking.
 */

import { COLORMAPS, rampLUT, LUT_STOPS, type ColormapName } from "../../draw/colormap";
import { f32lit } from "./ad/emit_wgsl";

export interface SurpriseShaderOpts {
  /** LUT resolution (default LUT_STOPS = 33). */
  stops?: number;
}

/** Uniform block size in bytes — seven f32 values + one u32 plane offset. */
export const SURPRISE_UNI_BYTES = 32;

export function surprisePointsShader(
  name: ColormapName,
  opts: SurpriseShaderOpts = {}
): string {
  const cm = COLORMAPS[name];
  const stops = opts.stops ?? LUT_STOPS;
  const lut = rampLUT(cm, stops);
  const entries: string[] = [];
  for (let i = 0; i < stops; i++) {
    entries.push(
      `vec3f(${f32lit(lut[i * 3])}, ${f32lit(lut[i * 3 + 1])}, ${f32lit(lut[i * 3 + 2])})`
    );
  }

  return /* wgsl */ `
// GENERATED from src/draw/colormap.ts — colormap "${cm.name}", ${stops} stops.
// Do not hand-edit the LUT: regenerate by changing the TS ramp.
struct Uni {
  resolution : vec2f,
  pointSize  : f32,
  lo         : f32,
  mid        : f32,
  hi         : f32,
  gain       : f32,
  surpriseOffset : u32,
};

@group(0) @binding(0) var<uniform> u : Uni;
@group(0) @binding(1) var<storage, read> posBuf : array<f32>;
@group(0) @binding(2) var<storage, read> surBuf : array<f32>;

const LUT_N : u32 = ${stops}u;
const LUT : array<vec3f, ${stops}> = array<vec3f, ${stops}>(
  ${entries.join(",\n  ")}
);

// Ramp position — emitted from COLORMAPS["${cm.name}"].wgslPosition so the GPU
// uses byte-identical polarity + span-floor logic to the CPU renderer.
fn position(v : f32) -> f32 {
  return ${cm.wgslPosition};
}

fn ramp(t : f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0) * f32(LUT_N - 1u);
  let i = min(LUT_N - 2u, u32(floor(x)));
  let f = x - f32(i);
  return mix(LUT[i], LUT[i + 1u], f);
}

struct VSOut {
  @builtin(position) clip  : vec4f,
  @location(0)       uv    : vec2f,
  @location(1)       color : vec3f,
};

@vertex
fn vs(@builtin(vertex_index) vid : u32,
      @builtin(instance_index) iid : u32) -> VSOut {
  var corners = array<vec2f, 4>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0));
  let corner = corners[vid];

  let px = posBuf[iid * 2u];
  let py = posBuf[iid * 2u + 1u];

  // pixel (origin top-left) -> clip space (centre, y up) — same mapping as
  // points.ts so the two renderers overlay pixel-exactly.
  var centre = (vec2f(px, py) / u.resolution) * 2.0 - vec2f(1.0, 1.0);
  centre.y = -centre.y;
  let offset = (corner - vec2f(0.5, 0.5)) * u.pointSize / u.resolution * 2.0;

  var out : VSOut;
  out.clip = vec4f(centre + offset, 0.0, 1.0);
  out.uv = corner;
  out.color = ramp(position(surBuf[u.surpriseOffset + iid])) * u.gain;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let d = length(in.uv - vec2f(0.5, 0.5));
  let aa = fwidth(d) * 1.5;
  let alpha = smoothstep(0.5, 0.5 - aa, d);
  if (alpha <= 0.0) { discard; }
  return vec4f(in.color, alpha);
}
`;
}
