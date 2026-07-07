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
 *              2x2 bilinear footprint (floor position, fractional weights) of
 *              fixed-point (energy 1.0 → 4096) atomicAdds, bounds-guarded.
 *   3. TONEMAP fullscreen triangle — reads acc as plain array<u32> (atomics
 *              are only needed while pass 2 races; passes in one encoder are
 *              ordered), colour = background + acc/4096 * exposure, soft
 *              shoulder c/(1+c) so additive glow saturates instead of
 *              clipping, gamma 1/2.2.
 *
 * COLOURING HOOKS — all live-settable fields, uploaded each frame, no
 * pipeline rebuilds: `.exposure` (linear gain), `.decay` (trail persistence),
 * `.classes`, `.maxSpeed`, `.background`. For per-class palettes, edit the
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

// Shared uniform block — one 48-byte buffer bound to all three passes.
const UNI_WGSL = /* wgsl */ `
struct Uni {
  size       : vec2u,  // accumulator W,H (pixels)
  count      : u32,    // live particles
  classes    : u32,    // 0 = speed colouring
  maxSpeed   : f32,
  exposure   : f32,
  decay      : f32,
  pad0       : f32,
  background : vec4f,  // rgb 0-1 (pre-divided by 255), a unused
};
@group(0) @binding(0) var<uniform> u : Uni;
`;

// Pass 1 — decay/clear. Thread per texel-channel. The workgroup grid is 2D
// because W*H*3/256 exceeds the 65535 per-dimension dispatch limit at 4K.
const DECAY_WGSL = /* wgsl */ `
${UNI_WGSL}
@group(0) @binding(1) var<storage, read_write> acc : array<u32>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(num_workgroups) nwg : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let i = (wg.y * nwg.x + wg.x) * ${WG}u + li;
  if (i >= u.size.x * u.size.y * 3u) { return; }
  acc[i] = u32(f32(acc[i]) * u.decay); // decay=0 -> hard clear
}
`;

// Pass 2 — splat. Thread per particle; colour formulas are copied VERBATIM
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
  if (x < 0 || y < 0 || x >= i32(u.size.x) || y >= i32(u.size.y)) { return; }
  let base = (u32(y) * u.size.x + u32(x)) * 3u;
  atomicAdd(&acc[base + 0u], u32(col.r * wgt * ${SPLAT_FIXED_POINT}.0));
  atomicAdd(&acc[base + 1u], u32(col.g * wgt * ${SPLAT_FIXED_POINT}.0));
  atomicAdd(&acc[base + 2u], u32(col.b * wgt * ${SPLAT_FIXED_POINT}.0));
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let iid = gid.x;
  if (iid >= u.count) { return; }
  let px = posBuf[iid * 2u];
  let py = posBuf[iid * 2u + 1u];
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

  // 2x2 bilinear footprint: floor position, fractional weights
  let x0 = i32(floor(px));
  let y0 = i32(floor(py));
  let fx = px - floor(px);
  let fy = py - floor(py);
  tap(x0,     y0,     (1.0 - fx) * (1.0 - fy), col);
  tap(x0 + 1, y0,     fx * (1.0 - fy),         col);
  tap(x0,     y0 + 1, (1.0 - fx) * fy,         col);
  tap(x0 + 1, y0 + 1, fx * fy,                 col);
}
`;

// Pass 3 — tonemap. Reads the accumulator as plain u32 (pass ordering within
// the encoder makes the pass-2 atomics visible; no atomic view needed here).
const TONEMAP_WGSL = /* wgsl */ `
${UNI_WGSL}
@group(0) @binding(1) var<storage, read> acc : array<u32>;

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
  let energy = vec3f(f32(acc[base + 0u]), f32(acc[base + 1u]), f32(acc[base + 2u]))
             * (1.0 / ${SPLAT_FIXED_POINT}.0);
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
  // Caller keeps canvas.width/height == the w,h passed to render(), exactly
  // like points.ts — the tonemap shader indexes the accumulator by fragment
  // coordinate, so a size mismatch would read the wrong texels.
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

  private readonly decayPipe: GPUComputePipeline;
  private readonly splatPipe: GPUComputePipeline;
  private readonly tonePipe: GPURenderPipeline;
  private readonly uni: GPUBuffer;
  private readonly uniData = new ArrayBuffer(48);
  private readonly uniF = new Float32Array(this.uniData);
  private readonly uniU = new Uint32Array(this.uniData);

  // Live-settable colouring controls — re-uploaded every frame, so changing
  // any of these mid-run retunes the image with zero pipeline rebuilds.
  background: [number, number, number];
  maxSpeed: number;
  classes: number;
  decay: number;
  exposure: number;

  // accumulation buffer + bind groups, cached on size / buffer identity
  private accBuf: GPUBuffer | null = null;
  private accW = 0;
  private accH = 0;
  private decayBind: GPUBindGroup | null = null;
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

    this.decayPipe = computePipeline(device, DECAY_WGSL);
    this.splatPipe = computePipeline(device, SPLAT_WGSL);
    this.tonePipe = renderPipeline(device, {
      code: TONEMAP_WGSL,
      format: this.target.format,
      topology: "triangle-list",
    });
    this.uni = uniformBuffer(device, 48);

    this.background = opts.background ?? [2, 0, 12];
    this.maxSpeed = opts.maxSpeed ?? 4;
    this.classes = opts.classes ?? 0;
    this.decay = opts.decay ?? 0;
    this.exposure = opts.exposure ?? 1;
  }

  /** Splat N particles (interleaved-xy f32 pos/vel storage buffers — the same
   *  buffers points.ts binds) into a w×h frame: decay → splat → tonemap, one
   *  encoder, one submit. Reallocates the accumulator when w/h changes.
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
    this.ensureAccum(w, h);
    this.ensureSplatBind(posBuf, velBuf);

    this.uniU[0] = w;
    this.uniU[1] = h;
    this.uniU[2] = n;
    this.uniU[3] = this.classes >>> 0;
    this.uniF[4] = this.maxSpeed;
    this.uniF[5] = this.exposure;
    this.uniF[6] = this.decay;
    this.uniF[7] = 0;
    this.uniF[8] = this.background[0] / 255;
    this.uniF[9] = this.background[1] / 255;
    this.uniF[10] = this.background[2] / 255;
    this.uniF[11] = 1;
    this.device.queue.writeBuffer(this.uni, 0, this.uniData);

    const enc = encoder;
    // @webgpu/types 0.1.30 predates object-form timestampWrites; the live
    // runtime uses it (see gputime.ts). Split the caller's begin/end across the
    // decay (begin) and tonemap (end) passes so the span covers the whole splat.
    const decayTs = ts
      ? { querySet: ts.querySet, beginningOfPassWriteIndex: ts.beginningOfPassWriteIndex }
      : undefined;
    const toneTs = ts
      ? { querySet: ts.querySet, endOfPassWriteIndex: ts.endOfPassWriteIndex }
      : undefined;

    {
      // 1 — decay/clear (thread per texel-channel, 2D grid: see DECAY_WGSL)
      const groups = Math.ceil((w * h * 3) / WG);
      const gx = Math.min(groups, 65535);
      const gy = Math.ceil(groups / gx);
      const p = enc.beginComputePass(
        (decayTs ? { timestampWrites: decayTs } : undefined) as GPUComputePassDescriptor
      );
      p.setPipeline(this.decayPipe);
      p.setBindGroup(0, this.decayBind!);
      p.dispatchWorkgroups(gx, gy);
      p.end();
    }
    {
      // 2 — splat (thread per particle)
      const p = enc.beginComputePass();
      p.setPipeline(this.splatPipe);
      p.setBindGroup(0, this.splatBind!);
      p.dispatchWorkgroups(Math.ceil(n / WG));
      p.end();
    }
    {
      // 3 — tonemap (fullscreen triangle; the draw covers every pixel, the
      // clear is just a defined-load requirement)
      const p = enc.beginRenderPass({
        colorAttachments: [
          {
            view: this.target.view(w, h),
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

  /** (Re)allocate the W×H×3 atomic<u32> accumulator + its bind groups.
   *  WebGPU zero-initialises fresh buffers, so a resize starts clean. */
  private ensureAccum(w: number, h: number): void {
    if (this.accBuf && this.accW === w && this.accH === h) return;
    this.accBuf?.destroy();
    this.accBuf = this.device.createBuffer({
      size: w * h * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.accW = w;
    this.accH = h;
    this.decayBind = this.device.createBindGroup({
      layout: this.decayPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.accBuf } },
      ],
    });
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
