/**
 * GpuPointRendererWebGPU — WebGPU-native, zero-copy particle renderer.
 *
 * Positions/velocities are read STRAIGHT from tfjs tensors: on the 'webgpu'
 * backend `tensor.dataToGPU()` returns `{ buffer: GPUBuffer, tensorRef }`, and
 * because we render on tfjs's OWN GPUDevice (tf.backend().device), that buffer
 * is bound directly as a read-only storage buffer — no CPU readback, no copy,
 * and (unlike the WebGL path) no texture-layout inference and no canvas-context
 * poisoning. The vertex shader indexes it by `@builtin(instance_index)` and
 * emits one instanced quad per particle; the fragment shader carves a smooth
 * sub-pixel anti-aliased round dot with additive (glow) blending.
 *
 * v1 clears to the background each frame (no ghost trails yet — trails need an
 * owned accumulation texture since a WebGPU swapchain isn't preserved across
 * frames; that's the v2 refinement once this path is confirmed on real hardware).
 */

import * as tf from "@tensorflow/tfjs";
import type { PassTimestampWrites } from "./gputime";
import {
  attachCanvas,
  renderPipeline,
  uniformBuffer,
  bindGroup,
  renderPass,
  BLEND_ADD,
  GpuCtx,
} from "./microgpu";

const SHADER = /* wgsl */ `
struct Uni {
  resolution : vec2f,
  pointSize  : f32,
  maxSpeed   : f32,
  hasVel     : u32,
  classes    : u32,
  palette    : u32,
  diagnostic : u32,
  diagOffset : u32,
  diagScale  : f32,
  pad        : f32,
};

// same hash + salt as the advect/train kernels — class is derived, not stored
fn pcg(v : u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let t = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (t >> 22u) ^ t;
}
@group(0) @binding(0) var<uniform> u : Uni;
@group(0) @binding(1) var<storage, read> posBuf : array<f32>;
@group(0) @binding(2) var<storage, read> velBuf : array<f32>;
@group(0) @binding(3) var<storage, read> diagnosticBuf : array<f32>;

fn diagnosticColor(iid : u32, velocity : f32) -> vec3f {
  let value = clamp((diagnosticBuf[iid + u.diagOffset] + u.pad) * u.diagScale, 0.0, 1.0);
  let low = min(value * 2.0, 1.0);
  let high = max(value * 2.0 - 1.0, 0.0);
  var col = mix(
    mix(vec3f(0.08, 0.25, 1.0), vec3f(0.9, 0.12, 0.85), low),
    vec3f(1.0, 0.55, 0.08),
    high
  );
  // Diagnostic-only palette selectors mirror the splat path.
  if (u.palette == 5u) { col = vec3f(value * 1.15, value * value * 0.8, value * value * value * 0.35); }
  if (u.palette == 6u) { col = vec3f(0.12 + 0.78 * value, 0.08 + 0.85 * (1.0 - abs(2.0 * value - 1.0)), 0.32 + 0.5 * (1.0 - value)); }
  if (u.palette == 7u) { col = mix(vec3f(0.23, 0.3, 0.95), vec3f(0.95, 0.25, 0.22), value); }
  return col * (0.55 + 0.45 * velocity);
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

  // pixel (origin top-left) -> clip space (centre, y up)
  var centre = (vec2f(px, py) / u.resolution) * 2.0 - vec2f(1.0, 1.0);
  centre.y = -centre.y;
  let offset = (corner - vec2f(0.5, 0.5)) * u.pointSize / u.resolution * 2.0;

  var out : VSOut;
  out.clip = vec4f(centre + offset, 0.0, 1.0);
  out.uv = corner;

  var col = vec3f(1.0, 1.0, 1.0);
  var t = 1.0;
  if (u.hasVel == 1u) {
    let vx = velBuf[iid * 2u];
    let vy = velBuf[iid * 2u + 1u];
    t = clamp(length(vec2f(vx, vy)) / u.maxSpeed, 0.0, 1.0);
    col = mix(vec3f(0.25, 0.55, 1.0), vec3f(1.0, 0.55, 0.2), t); // blue->orange
  }
  if (u.palette == 2u) {
    let role = pcg(iid ^ 2246822519u) % 3u;
    let base = select(
      select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), role == 1u),
      vec3f(0.0, 0.0, 1.0),
      role == 2u
    );
    col = base * (0.55 + 0.45 * t);
  } else if (u.palette == 3u) {
    // rgb-families: EXACT R/G/B keyed on the family label — the same
    // pcg(i ^ CLASS_SALT) % C the advect kernel and both trainers derive. The
    // hue is the instrument here: the per-family payoff chart is only readable
    // if "the red family" on screen is literally family 0.
    let cls = pcg(iid ^ 2166136261u) % u.classes;
    let base = select(
      select(vec3f(1.0, 0.12, 0.12), vec3f(0.12, 1.0, 0.22), cls == 1u),
      vec3f(0.22, 0.35, 1.0),
      cls == 2u
    );
    col = base * (0.55 + 0.45 * t);
  } else if (u.palette == 4u) {
    let group = iid % 3u;
    let base = select(
      select(vec3f(0.12, 0.95, 1.0), vec3f(1.0, 0.18, 0.78), group == 1u),
      vec3f(1.0, 0.72, 0.12),
      group == 2u
    );
    col = base * (0.55 + 0.45 * t);
  } else if (u.classes > 0u) {
    // per-species base colour (cosine palette, golden-angle spaced hues),
    // brightness modulated by speed
    let cls = pcg(iid ^ 2166136261u) % u.classes;
    let hue = f32(cls) * 2.399963;
    let base = 0.55 + 0.45 * cos(vec3f(hue, hue + 2.0944, hue + 4.1888));
    col = base * (0.55 + 0.45 * t);
  }
  if (u.diagnostic != 0u) { col = diagnosticColor(iid, t); }
  out.color = col;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let d = length(in.uv - vec2f(0.5, 0.5));
  let aa = fwidth(d) * 1.5;              // sub-pixel AA width
  let alpha = smoothstep(0.5, 0.5 - aa, d);
  if (alpha <= 0.0) { discard; }
  return vec4f(in.color, alpha);        // additive: colour weighted by alpha
}
`;

export interface GpuPointOpts {
  pointSize?: number;
  background?: [number, number, number];
  maxSpeed?: number;
  /** multi-species count — colours dots per class (0 = speed colouring) */
  classes?: number;
  palette?: "speed" | "species" | "rgb-roles" | "rgb-families" | "optimizer-groups-v1";
  /** Optional per-particle scalar colour diagnostic. Values map to [0,1]. */
  colorDiagnostic?: PointColorDiagnostic;
}

export interface PointColorDiagnostic {
  buffer: GPUBuffer;
  mode: "raw" | "per-unit";
  offsetFloats?: number;
  scale?: number;
  bias?: number;
  colormap?: "inferno" | "viridis" | "coolwarm";
}

export class GpuPointRendererWebGPU {
  /** True iff the browser exposes the WebGPU API. Adapter/device readiness is
   *  confirmed separately by successfully setting the tfjs 'webgpu' backend. */
  static isSupported(): boolean {
    return typeof navigator !== "undefined" && !!(navigator as any).gpu;
  }

  private readonly device: GPUDevice;
  private readonly ctx: GpuCtx;
  private readonly pipeline: GPURenderPipeline;
  private readonly uni: GPUBuffer;
  private readonly uniData = new ArrayBuffer(48);
  private readonly uniF = new Float32Array(this.uniData);
  private readonly uniU = new Uint32Array(this.uniData);
  private readonly bg: [number, number, number];
  private readonly pointSize: number;
  private maxSpeed: number;
  private readonly classes: number;
  private readonly palette: number;
  private colorDiagnostic: PointColorDiagnostic | null;
  private readonly zeroDiagnostic: GPUBuffer;

  /**
   * @param canvas on-screen canvas (must NOT have a 2d/webgl context yet).
   * @throws if the tfjs 'webgpu' backend/device isn't ready — construct only
   *   after `await tf.setBackend('webgpu'); await tf.ready()`.
   */
  constructor(canvas: HTMLCanvasElement, opts: GpuPointOpts = {}) {
    const device = (tf.backend() as any).device as GPUDevice | undefined;
    if (!device) {
      throw new Error(
        "GpuPointRendererWebGPU: tf.backend().device is missing — set the " +
          "'webgpu' backend and await tf.ready() before constructing."
      );
    }
    this.device = device;
    this.ctx = attachCanvas(canvas, device);
    this.pipeline = renderPipeline(device, {
      code: SHADER,
      format: this.ctx.format,
      blend: BLEND_ADD,
      topology: "triangle-strip",
    });
    this.uni = uniformBuffer(device, 48);
    this.pointSize = opts.pointSize ?? 2;
    this.bg = opts.background ?? [2, 0, 12];
    this.maxSpeed = opts.maxSpeed ?? 4;
    this.classes = opts.classes ?? 0;
    this.palette =
      opts.palette === "optimizer-groups-v1"
        ? 4
        : opts.palette === "rgb-families"
        ? 3
        : opts.palette === "rgb-roles"
        ? 2
        : opts.palette === "species"
        ? 1
        : 0;
    this.colorDiagnostic = opts.colorDiagnostic ?? null;
    this.zeroDiagnostic = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.zeroDiagnostic, 0, new Float32Array([0]));
  }

  /** Select a scalar colour source while retaining point/velocity geometry. */
  setColorDiagnostic(diagnostic: PointColorDiagnostic | null): void {
    this.colorDiagnostic = diagnostic;
  }

  /** Live velocity-colour scale; independent of the render pipeline. */
  setMaxSpeed(v: number): void {
    this.maxSpeed = Math.max(1e-6, v);
  }

  // Cached bind group for the raw-buffer path: the fused advect kernel's
  // buffers are stable across frames (unlike per-frame dataToGPU clones), so
  // the group is rebuilt only when the buffers themselves change (resize).
  private bufBind: GPUBindGroup | null = null;
  private bufBindPos: GPUBuffer | null = null;
  private bufBindVel: GPUBuffer | null = null;
  private bufBindDiagnostic: GPUBuffer | null = null;

  /** Draw N round dots straight from raw GPU buffers (interleaved xy f32) —
   *  the zero-copy path for the fused advect kernel's particle state.
   *  (Self-submitting; the fused hot path uses encodeRender.) */
  renderFromBuffers(
    posBuf: GPUBuffer,
    velBuf: GPUBuffer,
    n: number,
    w: number,
    h: number
  ): void {
    const encoder = this.device.createCommandEncoder();
    this.record(encoder, posBuf, velBuf, n, w, h);
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Same render pass as {@link renderFromBuffers}, recorded into a CALLER-owned
   * encoder and NOT submitted — lets a whole frame collapse to one queue.submit.
   * getCurrentTexture() is called here, during this frame's recording (correct).
   * `ts` optionally timestamps the pass.
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
    this.uniF[0] = w;
    this.uniF[1] = h;
    this.uniF[2] = this.pointSize;
    this.uniF[3] = this.maxSpeed;
    this.uniU[4] = 1; // hasVel
    this.uniU[5] = this.classes;
    this.uniU[6] = diagnostic
      ? diagnostic.colormap === "viridis" ? 6 : diagnostic.colormap === "coolwarm" ? 7 : 5
      : this.palette;
    const diagnostic = this.colorDiagnostic;
    this.uniU[7] = diagnostic ? 1 : 0;
    this.uniU[8] = Math.max(0, Math.floor(diagnostic?.offsetFloats ?? 0));
    this.uniF[9] = Number.isFinite(diagnostic?.scale) ? Math.max(0, diagnostic!.scale!) : 1;
    this.uniF[10] = Number.isFinite(diagnostic?.bias) ? diagnostic!.bias! : 0;
    this.device.queue.writeBuffer(this.uni, 0, this.uniData);

    const diagnosticBuffer = diagnostic?.buffer ?? this.zeroDiagnostic;
    if (
      this.bufBindPos !== posBuf ||
      this.bufBindVel !== velBuf ||
      this.bufBindDiagnostic !== diagnosticBuffer
    ) {
      this.bufBind = bindGroup(this.device, this.pipeline, [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: velBuf } },
        { binding: 3, resource: { buffer: diagnosticBuffer } },
      ]);
      this.bufBindPos = posBuf;
      this.bufBindVel = velBuf;
      this.bufBindDiagnostic = diagnosticBuffer;
    }
    const group = this.bufBind!;

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.ctx.context.getCurrentTexture().createView(),
          clearValue: {
            r: this.bg[0] / 255,
            g: this.bg[1] / 255,
            b: this.bg[2] / 255,
            a: 1,
          },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      // @webgpu/types 0.1.30 predates object-form timestampWrites (see gputime).
      ...((ts ? { timestampWrites: ts } : {}) as object),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.draw(4, n);
    pass.end();
  }

  /** Draw `pos` ([N,2] pixel coords) as N round dots, reading positions straight
   *  from GPU memory. `vel` ([N,2]) colours dots by speed. No CPU copy. */
  render(
    pos: tf.Tensor2D,
    vel: tf.Tensor2D | null,
    w: number,
    h: number,
    _frame: number
  ): void {
    const n = pos.shape[0];

    this.uniF[0] = w;
    this.uniF[1] = h;
    this.uniF[2] = this.pointSize;
    this.uniF[3] = this.maxSpeed;
    this.uniU[4] = vel ? 1 : 0; // byte offset 16
    this.uniU[5] = this.classes;
    this.uniU[6] = this.palette;
    const diagnostic = this.colorDiagnostic;
    this.uniU[7] = diagnostic ? 1 : 0;
    this.uniU[8] = Math.max(0, Math.floor(diagnostic?.offsetFloats ?? 0));
    this.uniF[9] = Number.isFinite(diagnostic?.scale) ? Math.max(0, diagnostic!.scale!) : 1;
    this.device.queue.writeBuffer(this.uni, 0, this.uniData);

    // Zero-copy: tfjs tensor GPU buffers, same device -> bindable directly.
    const posGpu = pos.dataToGPU();
    const velGpu = vel ? vel.dataToGPU() : null;

    const group = bindGroup(this.device, this.pipeline, [
      { binding: 0, resource: { buffer: this.uni } },
      { binding: 1, resource: { buffer: (posGpu as any).buffer } },
      { binding: 2, resource: { buffer: ((velGpu ?? posGpu) as any).buffer } },
      { binding: 3, resource: { buffer: diagnostic?.buffer ?? this.zeroDiagnostic } },
    ]);

    renderPass(
      this.ctx,
      [this.bg[0] / 255, this.bg[1] / 255, this.bg[2] / 255, 1],
      (p) => {
        p.setPipeline(this.pipeline);
        p.setBindGroup(0, group);
        p.draw(4, n); // triangle-strip quad, N instances
      }
    );

    // Release the per-frame tensor clones tfjs minted for dataToGPU().
    posGpu.tensorRef.dispose();
    velGpu?.tensorRef.dispose();
  }

  destroy(): void {
    try {
      this.uni.destroy();
    } catch (_) {}
    try {
      (this.ctx.context as any).unconfigure?.();
    } catch (_) {}
    try {
      this.zeroDiagnostic.destroy();
    } catch (_) {}
  }
}
