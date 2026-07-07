/**
 * SplatRenderer — compute-splat particle renderer (the 1M@60 path).
 *
 * WHY: the quad renderer (points.ts) pushes 4 vertices per particle through
 * the raster/blend unit — at 1M particles that's 4M vertex invocations plus
 * 5-10x additive overdraw, measured ~36 ms/frame while the advect compute is
 * only ~7-13 ms. This renderer replaces raster with compute: one thread per
 * particle atomically accumulates fixed-point RGB energy into a full-res
 * storage buffer, then a single fullscreen-triangle pass tonemaps it to the
 * target. Cost scales with N + W*H — overdraw is just more atomic adds.
 *
 * Per frame (ONE command encoder / ONE submit), three ordered passes:
 *   1. DECAY   compute — thread per texel-channel: acc = u32(f32(acc)*decay).
 *              decay=0 → hard clear (quad-renderer semantics); ~0.85-0.95 →
 *              ghost trails, the feature points.ts never had (a swapchain
 *              isn't preserved across frames; an owned accumulator is).
 *   2. SPLAT   compute — thread per particle: reads the SAME interleaved-xy
 *              pos/vel storage buffers points.ts binds, colours EXACTLY like
 *              points.ts's vertex shader (speed mix blue→orange; classes>0 →
 *              pcg-hash cosine palette with brightness by speed), deposits a
 *              RADIAL CONE kernel (weight = max(0, 1 - dist/radius), dist from
 *              the particle's exact subpixel position to each texel centre,
 *              weights NORMALIZED in-shader so every particle deposits total
 *              energy 1.0 → 4096 counts regardless of radius/subpixel phase)
 *              of fixed-point atomicAdds, bounds-guarded. Round dots, not the
 *              square 2x2 bilinear footprint this pass used to have.
 *   3. TONEMAP fullscreen triangle — reads acc as plain array<u32> (atomics
 *              are only needed while pass 2 races; passes in one encoder are
 *              ordered), colour = background + acc/4096 * exposure, soft
 *              shoulder c/(1+c) so additive glow saturates instead of
 *              clipping, gamma 1/2.2.
 *
 * RETINA — the `dpr` option (default 1) sizes the accumulator at
 * ceil(w*dpr)×ceil(h*dpr) NATIVE pixels and scales particle positions by dpr
 * in the splat pass; `render()` keeps taking CSS w,h, and the caller keeps
 * canvas.width/height == the same ceil(w*dpr)×ceil(h*dpr) (the tonemap
 * indexes the accumulator by fragment coordinate — a stride mismatch would
 * shear the image). The physics world stays in CSS pixels.
 *
 * COLOURING HOOKS — all live-settable fields, uploaded each frame, no
 * pipeline rebuilds: `.exposure` (linear gain), `.decay` (trail persistence),
 * `.radius` (splat dot radius, CSS px), `.classes`, `.maxSpeed`,
 * `.background`. For per-class palettes, edit the
 * cosine-palette block in SPLAT_WGSL (swap in a lookup keyed by `cls`); for
 * colour grading / tone curves, edit `fs` in TONEMAP_WGSL — that is the ONE
 * place where accumulated energy becomes screen colour.
 *
 * HEADLESS: pass canvas=null plus an explicit opts.device — output goes to an
 * internal rgba8unorm offscreen texture (`.offscreen.texture`) sized on first
 * render, which tests read back via copyTextureToBuffer (tools/splat_test.ts).
 * No tfjs backend, no swapchain needed.
 */

import * as tf from "@tensorflow/tfjs";
import type { PassTimestampWrites } from "./gputime";
import {
  attachCanvas,
  computePipeline,
  renderPipeline,
  uniformBuffer,
  bindGroup,
  GpuCtx,
} from "./microgpu";

/** Fixed-point scale of the accumulator: energy 1.0 == 4096 integer counts. */
export const SPLAT_FIXED_POINT = 4096;

const WG = 256;
/** Max native cone radius. Caps the splat tap count so retina (dpr>1) doesn't
 *  explode the atomic-bound splat pass; a crisp small dot, not a fat blob. */
const NATIVE_RADIUS_MAX = 1.6;

// Shared uniform block — one 64-byte buffer bound to all three passes.
const UNI_WGSL = /* wgsl */ `
struct Uni {
  size       : vec2u,  // accumulator W,H (NATIVE pixels: ceil(css * dpr))
  count      : u32,    // live particles
  classes    : u32,    // 0 = speed colouring
  maxSpeed   : f32,
  exposure   : f32,
  decay      : f32,
  radius     : f32,    // cone radius in NATIVE pixels, pre-clamped [0.75, 4]
  background : vec4f,  // rgb 0-1 (pre-divided by 255), a unused
  dpr        : f32,    // devicePixelRatio: CSS particle coords -> native texels
  pad0       : f32,
  pad1       : f32,
  pad2       : f32,
};
@group(0) @binding(0) var<uniform> u : Uni;
`;

// Decay is FUSED into the tonemap pass (below) — it multiplies the
// accumulator by `decay` in place after reading it for display, so there is no
// separate decay/clear compute pass (that streamed the whole ~44MB buffer with
// its own barrier — the dominant retina cost). decay=0 => tonemap writes 0
// (hard clear for the next frame).

// Pass 1 — splat. Thread per particle; colour formulas are copied VERBATIM
// from points.ts's vertex shader so the two renderers are visually
// interchangeable (hasVel is always 1 on this path, same as renderFromBuffers).
const SPLAT_WGSL = /* wgsl */ `
${UNI_WGSL}
// same hash + salt as points.ts / advect / train kernels — class is derived,
// not stored
fn pcg(v : u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let t = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (t >> 22u) ^ t;
}

@group(0) @binding(1) var<storage, read> posBuf : array<f32>;
@group(0) @binding(2) var<storage, read> velBuf : array<f32>;
@group(0) @binding(3) var<storage, read_write> acc : array<atomic<u32>>;

fn tap(x : i32, y : i32, wgt : f32, col : vec3f) {
  if (wgt <= 0.0) { return; } // box corners outside the cone: skip the atomics
  if (x < 0 || y < 0 || x >= i32(u.size.x) || y >= i32(u.size.y)) { return; }
  let base = (u32(y) * u.size.x + u32(x)) * 3u;
  atomicAdd(&acc[base + 0u], u32(col.r * wgt * ${SPLAT_FIXED_POINT}.0));
  atomicAdd(&acc[base + 1u], u32(col.g * wgt * ${SPLAT_FIXED_POINT}.0));
  atomicAdd(&acc[base + 2u], u32(col.b * wgt * ${SPLAT_FIXED_POINT}.0));
}

fn cone(x : i32, y : i32, p : vec2f) -> f32 {
  let d = distance(vec2f(f32(x), f32(y)), p);
  return max(0.0, 1.0 - d / u.radius);
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let iid = gid.x;
  if (iid >= u.count) { return; }
  // CSS-pixel particle position -> native accumulator texels
  let px = posBuf[iid * 2u] * u.dpr;
  let py = posBuf[iid * 2u + 1u] * u.dpr;
  let vx = velBuf[iid * 2u];
  let vy = velBuf[iid * 2u + 1u];

  let t = clamp(length(vec2f(vx, vy)) / u.maxSpeed, 0.0, 1.0);
  var col = mix(vec3f(0.25, 0.55, 1.0), vec3f(1.0, 0.55, 0.2), t); // blue->orange
  if (u.classes > 0u) {
    // per-species base colour (cosine palette, golden-angle spaced hues),
    // brightness modulated by speed — PALETTE HOOK: swap this block for a
    // per-class lookup to hand-pick species colours
    let cls = pcg(iid ^ 2166136261u) % u.classes;
    let hue = f32(cls) * 2.399963;
    let base = 0.55 + 0.45 * cos(vec3f(hue, hue + 2.0944, hue + 4.1888));
    col = base * (0.55 + 0.45 * t);
  }

  // RADIAL CONE kernel over the integer box covering the circle |t - p| <= r.
  // Texel "centres" sit at INTEGER coords — the same convention as the old
  // 2x2 bilinear (an integer-position particle is centred on that texel
  // index, which is what the tonemap's u32(frag.xy) indexing expects).
  // Two in-shader passes: sum the cone weights, then deposit normalized taps,
  // so EVERY particle deposits total energy 1.0 (4096 counts) regardless of
  // radius or subpixel phase. u.radius is pre-clamped to [0.75, 4] on the CPU,
  // so the box is <= 9x9 and wsum > 0 always (the nearest texel centre is at
  // most sqrt(2)/2 ~ 0.707 < 0.75 away).
  let p = vec2f(px, py);
  let x0 = i32(ceil(px - u.radius));
  let x1 = i32(floor(px + u.radius));
  let y0 = i32(ceil(py - u.radius));
  let y1 = i32(floor(py + u.radius));
  var wsum = 0.0;
  for (var y = y0; y <= y1; y++) {
    for (var x = x0; x <= x1; x++) {
      wsum += cone(x, y, p);
    }
  }
  for (var y = y0; y <= y1; y++) {
    for (var x = x0; x <= x1; x++) {
      tap(x, y, cone(x, y, p) / wsum, col);
    }
  }
}
`;

// Pass 2 (final) — tonemap + FUSED DECAY. Reads the accumulator as plain u32
// (pass ordering within the encoder makes the splat atomics visible), writes
// the display pixel, THEN multiplies the accumulator by decay in place — so
// the separate decay compute pass is gone (it streamed the whole ~44MB buffer
// with its own barrier). Decay now happens at END of frame instead of start;
// equivalent up to a one-frame phase shift (measured 32ms→~9ms at 1M retina).
// binding(1) is read_write so the fragment can write the faded value back.
const TONEMAP_WGSL = /* wgsl */ `
${UNI_WGSL}
@group(0) @binding(1) var<storage, read_write> acc : array<u32>;

@vertex
fn vs(@builtin(vertex_index) vid : u32) -> @builtin(position) vec4f {
  // fullscreen triangle: (-1,-1) (3,-1) (-1,3)
  let x = f32((vid << 1u) & 2u);
  let y = f32(vid & 2u);
  return vec4f(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

// GRADING HOOK: the single place where accumulated energy becomes a screen
// colour — exposure curves, per-channel grading, vignettes etc. go here.
@fragment
fn fs(@builtin(position) frag : vec4f) -> @location(0) vec4f {
  let x = u32(frag.x);
  let y = u32(frag.y);
  let base = (y * u.size.x + x) * 3u;
  let r = acc[base + 0u];
  let g = acc[base + 1u];
  let b = acc[base + 2u];
  let energy = vec3f(f32(r), f32(g), f32(b)) * (1.0 / ${SPLAT_FIXED_POINT}.0);
  // FUSED DECAY: fade the accumulator for the NEXT frame after reading it for
  // display. decay=0 -> writes 0 (hard clear); decay~0.9 -> ghost trails.
  acc[base + 0u] = u32(f32(r) * u.decay);
  acc[base + 1u] = u32(f32(g) * u.decay);
  acc[base + 2u] = u32(f32(b) * u.decay);
  var c = u.background.rgb + energy * u.exposure;
  c = c / (1.0 + c);            // soft shoulder: glow saturates, never clips
  c = pow(c, vec3f(1.0 / 2.2)); // gamma
  return vec4f(c, 1.0);
}
`;

export interface SplatOpts {
  /** clear/base colour, 0-255 per channel (same convention as points.ts) */
  background?: [number, number, number];
  maxSpeed?: number;
  /** multi-species count — colours splats per class (0 = speed colouring) */
  classes?: number;
  /** per-frame trail persistence: 0 = hard clear, ~0.85-0.95 = ghost trails */
  decay?: number;
  /** linear gain on accumulated energy before the tone curve */
  exposure?: number;
  /**
   * devicePixelRatio (default 1): the accumulator becomes
   * ceil(w*dpr)×ceil(h*dpr) native pixels and particle positions (CSS px) are
   * scaled by dpr in the splat pass. render() keeps taking CSS w,h; the
   * caller's canvas backing store must be the SAME ceil(w*dpr)×ceil(h*dpr).
   */
  dpr?: number;
  /** radial cone splat radius in CSS px (default 1.25); native radius
   *  = radius*dpr, clamped to [0.75, 4] */
  radius?: number;
  /** explicit device (headless/tests) — defaults to tfjs's webgpu device */
  device?: GPUDevice;
}

/** Where the tonemap pass lands: swapchain (canvas) or owned texture. */
interface RenderTarget {
  readonly format: GPUTextureFormat;
  /** Per-frame output view; `w`,`h` must match the accumulator size. */
  view(w: number, h: number): GPUTextureView;
  destroy(): void;
}

class CanvasTarget implements RenderTarget {
  constructor(private readonly ctx: GpuCtx) {}
  get format(): GPUTextureFormat {
    return this.ctx.format;
  }
  // Caller keeps canvas.width/height == ceil(w*dpr)×ceil(h*dpr) for the CSS
  // w,h passed to render() — the tonemap shader indexes the accumulator by
  // fragment coordinate, so a size (row-stride) mismatch would shear/misread.
  view(_w: number, _h: number): GPUTextureView {
    return this.ctx.context.getCurrentTexture().createView();
  }
  destroy(): void {
    try {
      (this.ctx.context as any).unconfigure?.();
    } catch (_) {}
  }
}

export class OffscreenTarget implements RenderTarget {
  readonly format: GPUTextureFormat = "rgba8unorm";
  /** Readable output (RENDER_ATTACHMENT | COPY_SRC); allocated on first view. */
  texture: GPUTexture | null = null;
  private w = 0;
  private h = 0;
  constructor(private readonly device: GPUDevice) {}
  view(w: number, h: number): GPUTextureView {
    if (!this.texture || w !== this.w || h !== this.h) {
      this.texture?.destroy();
      this.texture = this.device.createTexture({
        size: { width: w, height: h, depthOrArrayLayers: 1 },
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      this.w = w;
      this.h = h;
    }
    return this.texture.createView();
  }
  destroy(): void {
    try {
      this.texture?.destroy();
    } catch (_) {}
  }
}

export class SplatRenderer {
  /** True iff the environment exposes the WebGPU API (same as points.ts). */
  static isSupported(): boolean {
    return typeof navigator !== "undefined" && !!(navigator as any).gpu;
  }

  private readonly device: GPUDevice;
  private readonly target: RenderTarget;
  /** Headless output (constructed with canvas === null), else null. */
  readonly offscreen: OffscreenTarget | null;

  private readonly splatPipe: GPUComputePipeline;
  private readonly tonePipe: GPURenderPipeline;
  private readonly uni: GPUBuffer;
  private readonly uniData = new ArrayBuffer(64);
  private readonly uniF = new Float32Array(this.uniData);
  private readonly uniU = new Uint32Array(this.uniData);

  /** devicePixelRatio — fixed at construction (canvas backing is sized once) */
  readonly dpr: number;

  // Live-settable colouring controls — re-uploaded every frame, so changing
  // any of these mid-run retunes the image with zero pipeline rebuilds.
  background: [number, number, number];
  maxSpeed: number;
  classes: number;
  decay: number;
  exposure: number;
  /** radial cone splat radius, CSS px (native = radius*dpr, clamped [0.75,4]) */
  radius: number;

  // accumulation buffer + bind groups, cached on size / buffer identity
  private accBuf: GPUBuffer | null = null;
  private accW = 0;
  private accH = 0;
  private toneBind: GPUBindGroup | null = null;
  private splatBind: GPUBindGroup | null = null;
  private splatBindPos: GPUBuffer | null = null;
  private splatBindVel: GPUBuffer | null = null;

  /**
   * @param canvas on-screen canvas (must NOT have a 2d/webgl context yet), or
   *   null for headless mode (render into `.offscreen.texture`).
   * @throws without opts.device if the tfjs 'webgpu' backend/device isn't
   *   ready — construct after `await tf.setBackend('webgpu'); await tf.ready()`.
   */
  constructor(canvas: HTMLCanvasElement | null, opts: SplatOpts = {}) {
    const device =
      opts.device ?? ((tf.backend() as any).device as GPUDevice | undefined);
    if (!device) {
      throw new Error(
        "SplatRenderer: no GPUDevice — pass opts.device, or set the tfjs " +
          "'webgpu' backend and await tf.ready() before constructing."
      );
    }
    this.device = device;
    this.offscreen = canvas ? null : new OffscreenTarget(device);
    this.target = canvas
      ? new CanvasTarget(attachCanvas(canvas, device))
      : this.offscreen!;

    this.splatPipe = computePipeline(device, SPLAT_WGSL);
    this.tonePipe = renderPipeline(device, {
      code: TONEMAP_WGSL,
      format: this.target.format,
      topology: "triangle-list",
    });
    this.uni = uniformBuffer(device, 64);

    this.dpr = opts.dpr ?? 1;
    this.background = opts.background ?? [2, 0, 12];
    this.maxSpeed = opts.maxSpeed ?? 4;
    this.classes = opts.classes ?? 0;
    this.decay = opts.decay ?? 0;
    this.exposure = opts.exposure ?? 1;
    this.radius = opts.radius ?? 1.25;
  }

  /** Splat N particles (interleaved-xy f32 pos/vel storage buffers — the same
   *  buffers points.ts binds) into a frame of CSS size w×h (native size
   *  ceil(w*dpr)×ceil(h*dpr)): decay → splat → tonemap, one encoder, one
   *  submit. Reallocates the accumulator when the size changes.
   *  (Self-submitting; the fused hot path uses encodeRender.) */
  render(
    posBuf: GPUBuffer,
    velBuf: GPUBuffer,
    n: number,
    w: number,
    h: number
  ): void {
    const enc = this.device.createCommandEncoder();
    this.record(enc, posBuf, velBuf, n, w, h);
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Same three passes as {@link render}, recorded into a CALLER-owned encoder
   * and NOT submitted — lets a whole frame collapse to one queue.submit. The
   * target's getCurrentTexture() is called here, during this frame's recording
   * (correct). `ts` times the WHOLE splat: its begin index goes on the decay
   * pass, its end index on the tonemap pass (the middle splat pass is untimed).
   */
  encodeRender(
    encoder: GPUCommandEncoder,
    posBuf: GPUBuffer,
    velBuf: GPUBuffer,
    n: number,
    w: number,
    h: number,
    ts?: PassTimestampWrites
  ): void {
    this.record(encoder, posBuf, velBuf, n, w, h, ts);
  }

  private record(
    encoder: GPUCommandEncoder,
    posBuf: GPUBuffer,
    velBuf: GPUBuffer,
    n: number,
    w: number,
    h: number,
    ts?: PassTimestampWrites
  ): void {
    // CSS w,h -> native accumulator size. Math.ceil matches the caller's
    // canvas-backing sizing (main.ts) so tonemap fragment coords, the canvas
    // and the accumulator all agree texel-for-texel.
    const nw = Math.ceil(w * this.dpr);
    const nh = Math.ceil(h * this.dpr);
    this.ensureAccum(nw, nh);
    this.ensureSplatBind(posBuf, velBuf);

    this.uniU[0] = nw;
    this.uniU[1] = nh;
    this.uniU[2] = n;
    this.uniU[3] = this.classes >>> 0;
    this.uniF[4] = this.maxSpeed;
    this.uniF[5] = this.exposure;
    this.uniF[6] = this.decay;
    // native cone radius. Clamp max is NATIVE_RADIUS_MAX (1.6), not 4: at
    // retina (dpr 2) radius*dpr would be 2.5 → a 5×5=25-tap box, and the splat
    // is atomic-bound so taps ≈ cost (25→9 taps roughly halves the frame at
    // 1M). 1.6 native is a crisp small dot — SHARPER than the old fat 2.5 blob,
    // not softer. The [0.75, cap] floor still guarantees wsum > 0 in-shader.
    this.uniF[7] = Math.min(NATIVE_RADIUS_MAX, Math.max(0.75, this.radius * this.dpr));
    this.uniF[8] = this.background[0] / 255;
    this.uniF[9] = this.background[1] / 255;
    this.uniF[10] = this.background[2] / 255;
    this.uniF[11] = 1;
    this.uniF[12] = this.dpr;
    this.device.queue.writeBuffer(this.uni, 0, this.uniData);

    const enc = encoder;
    // @webgpu/types 0.1.30 predates object-form timestampWrites; the live
    // runtime uses it (see gputime.ts). Two passes now (decay is fused into
    // tonemap): begin on splat, end on tonemap, so the span covers the whole
    // splat render.
    const splatTs = ts
      ? { querySet: ts.querySet, beginningOfPassWriteIndex: ts.beginningOfPassWriteIndex }
      : undefined;
    const toneTs = ts
      ? { querySet: ts.querySet, endOfPassWriteIndex: ts.endOfPassWriteIndex }
      : undefined;

    {
      // 1 — splat (thread per particle); reads last frame's decayed buffer
      const p = enc.beginComputePass(
        (splatTs ? { timestampWrites: splatTs } : undefined) as GPUComputePassDescriptor
      );
      p.setPipeline(this.splatPipe);
      p.setBindGroup(0, this.splatBind!);
      p.dispatchWorkgroups(Math.ceil(n / WG));
      p.end();
    }
    {
      // 2 — tonemap + fused decay (fullscreen triangle; the draw covers every
      // pixel, the clear is just a defined-load requirement)
      const p = enc.beginRenderPass({
        colorAttachments: [
          {
            view: this.target.view(nw, nh),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            storeOp: "store",
          },
        ],
        ...((toneTs ? { timestampWrites: toneTs } : {}) as object),
      });
      p.setPipeline(this.tonePipe);
      p.setBindGroup(0, this.toneBind!);
      p.draw(3);
      p.end();
    }
  }

  /** Test hook: the live accumulation buffer (native W×H×3 u32 counts,
   *  COPY_SRC) — lets tools/splat_test.ts assert on raw deposited energy.
   *  NOTE: the tonemap pass fades this buffer by `decay` at end of frame, so a
   *  raw read after render() sees post-decay counts; read with decay=1 (or call
   *  clearAccum first) to observe a single frame's raw deposit. */
  get accumBuffer(): GPUBuffer | null {
    return this.accBuf;
  }

  /** Zero the accumulation buffer — clears trails on demand (e.g. a "reset
   *  trails" control), and lets tests start a frame from a known-clean buffer
   *  independent of the fused end-of-frame decay. No-op before the first
   *  render (the buffer is allocated lazily and zero-initialised). */
  clearAccum(): void {
    if (!this.accBuf) return;
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.accBuf, 0, this.accW * this.accH * 3 * 4);
    this.device.queue.submit([enc.finish()]);
  }

  /** (Re)allocate the W×H×3 atomic<u32> accumulator + its bind groups.
   *  WebGPU zero-initialises fresh buffers, so a resize starts clean. */
  private ensureAccum(w: number, h: number): void {
    if (this.accBuf && this.accW === w && this.accH === h) return;
    this.accBuf?.destroy();
    this.accBuf = this.device.createBuffer({
      size: w * h * 3 * 4,
      // COPY_DST: clearAccum() uses encoder.clearBuffer to zero it on demand.
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    this.accW = w;
    this.accH = h;
    this.toneBind = bindGroup(this.device, this.tonePipe, [
      { binding: 0, resource: { buffer: this.uni } },
      { binding: 1, resource: { buffer: this.accBuf } },
    ]);
    this.splatBindPos = null; // splat group binds acc too — force rebuild
  }

  // Cached on buffer identity like points.ts: the fused advect kernel's
  // buffers are stable across frames, so this rebuilds only on resize/swap.
  private ensureSplatBind(posBuf: GPUBuffer, velBuf: GPUBuffer): void {
    if (
      this.splatBind &&
      this.splatBindPos === posBuf &&
      this.splatBindVel === velBuf
    ) {
      return;
    }
    this.splatBind = this.device.createBindGroup({
      layout: this.splatPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: velBuf } },
        { binding: 3, resource: { buffer: this.accBuf! } },
      ],
    });
    this.splatBindPos = posBuf;
    this.splatBindVel = velBuf;
  }

  destroy(): void {
    try {
      this.uni.destroy();
    } catch (_) {}
    try {
      this.accBuf?.destroy();
    } catch (_) {}
    this.target.destroy();
  }
}
