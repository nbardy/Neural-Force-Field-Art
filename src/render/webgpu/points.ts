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
};
@group(0) @binding(0) var<uniform> u : Uni;
@group(0) @binding(1) var<storage, read> posBuf : array<f32>;
@group(0) @binding(2) var<storage, read> velBuf : array<f32>;

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
  if (u.hasVel == 1u) {
    let vx = velBuf[iid * 2u];
    let vy = velBuf[iid * 2u + 1u];
    let t = clamp(length(vec2f(vx, vy)) / u.maxSpeed, 0.0, 1.0);
    col = mix(vec3f(0.25, 0.55, 1.0), vec3f(1.0, 0.55, 0.2), t); // blue->orange
  }
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
  private readonly uniData = new ArrayBuffer(32);
  private readonly uniF = new Float32Array(this.uniData);
  private readonly uniU = new Uint32Array(this.uniData);
  private readonly bg: [number, number, number];
  private readonly pointSize: number;
  private readonly maxSpeed: number;

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
    this.uni = uniformBuffer(device, 32);
    this.pointSize = opts.pointSize ?? 2;
    this.bg = opts.background ?? [2, 0, 12];
    this.maxSpeed = opts.maxSpeed ?? 4;
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
    this.device.queue.writeBuffer(this.uni, 0, this.uniData);

    // Zero-copy: tfjs tensor GPU buffers, same device -> bindable directly.
    const posGpu = pos.dataToGPU();
    const velGpu = vel ? vel.dataToGPU() : null;

    const group = bindGroup(this.device, this.pipeline, [
      { binding: 0, resource: { buffer: this.uni } },
      { binding: 1, resource: { buffer: (posGpu as any).buffer } },
      { binding: 2, resource: { buffer: ((velGpu ?? posGpu) as any).buffer } },
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
  }
}
