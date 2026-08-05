/**
 * GpuSurpriseRendererWebGPU — the zero-round-trip path for the "surprise"
 * render mode. Positions come from `AdvectKernel.posBuffer` and the per-particle
 * scalar comes from a second `array<f32>` GPU buffer of length N; both are bound
 * read-only and coloured in the vertex shader. Nothing per-particle ever touches
 * the CPU.
 *
 * NORMALISATION IS THE ONLY THING THAT COMES BACK. A percentile needs the whole
 * distribution, which is a reduction the render pass cannot do. Rather than read
 * N floats every frame (see the perf note in src/renderers.ts), {@link
 * GpuSurpriseStats} copies up to 1024 values from the latest freshly-written
 * coverage window once every `every` frames and maps them asynchronously. That
 * is at most 4 KB per 8 frames
 * regardless of N, and it never blocks the frame: the mapped result feeds the
 * RobustSpan a frame or two late, which is invisible at a 17-frame EMA time
 * constant.
 *
 * Window generations are deduplicated and wrapped windows use two GPU copies.
 * Consequently stale zeros (or values retained across a resize) never enter
 * the robust normalizer.
 */

import * as tf from "@tensorflow/tfjs";
import type { PassTimestampWrites } from "./gputime";
import {
  attachCanvas,
  renderPipeline,
  uniformBuffer,
  bindGroup,
  BLEND_ADD,
  GpuCtx,
} from "./microgpu";
import { surprisePointsShader, SURPRISE_UNI_BYTES } from "./surprise_wgsl";
import { RobustSpan, type Span } from "../../draw/robust_norm";
import type { ColormapName } from "../../draw/colormap";

export interface GpuSurpriseOpts {
  colormap: ColormapName;
  pointSize?: number;
  background?: [number, number, number];
  /** brightness multiplier applied after the ramp (additive blend) */
  gain?: number;
}

export class GpuSurpriseRendererWebGPU {
  static isSupported(): boolean {
    return typeof navigator !== "undefined" && !!(navigator as any).gpu;
  }

  private readonly device: GPUDevice;
  private readonly ctx: GpuCtx;
  private readonly pipeline: GPURenderPipeline;
  private readonly uni: GPUBuffer;
  private readonly uniData = new ArrayBuffer(SURPRISE_UNI_BYTES);
  private readonly uniF = new Float32Array(this.uniData);
  private readonly uniU = new Uint32Array(this.uniData);
  private readonly bg: [number, number, number];
  private readonly pointSize: number;
  private readonly gain: number;

  /** Current normalisation window — set by the caller from {@link GpuSurpriseStats}. */
  span: Span = { lo: 0, mid: 0, hi: 1 };

  constructor(canvas: HTMLCanvasElement, opts: GpuSurpriseOpts) {
    const device = (tf.backend() as any).device as GPUDevice | undefined;
    if (!device) {
      throw new Error(
        "GpuSurpriseRendererWebGPU: tf.backend().device is missing — set the " +
          "'webgpu' backend and await tf.ready() before constructing."
      );
    }
    this.device = device;
    this.ctx = attachCanvas(canvas, device);
    this.pipeline = renderPipeline(device, {
      code: surprisePointsShader(opts.colormap),
      format: this.ctx.format,
      blend: BLEND_ADD,
      topology: "triangle-strip",
    });
    this.uni = uniformBuffer(device, SURPRISE_UNI_BYTES);
    this.pointSize = opts.pointSize ?? 2;
    this.bg = opts.background ?? [2, 0, 12];
    this.gain = opts.gain ?? 1;
  }

  private bind: GPUBindGroup | null = null;
  private bindPos: GPUBuffer | null = null;
  private bindSur: GPUBuffer | null = null;

  /** Record the pass into a caller-owned encoder (the fused one-submit path). */
  encodeRender(
    encoder: GPUCommandEncoder,
    posBuf: GPUBuffer,
    surBuf: GPUBuffer,
    n: number,
    w: number,
    h: number,
    ts?: PassTimestampWrites,
    /** First float of the selected packed N-value surprise plane. */
    offsetFloats = 0
  ): void {
    if (!Number.isInteger(offsetFloats) || offsetFloats < 0) {
      throw new Error(
        `GpuSurpriseRendererWebGPU: offsetFloats must be a non-negative integer, got ${offsetFloats}`
      );
    }
    this.uniF[0] = w;
    this.uniF[1] = h;
    this.uniF[2] = this.pointSize;
    this.uniF[3] = this.span.lo;
    this.uniF[4] = this.span.mid;
    this.uniF[5] = this.span.hi;
    this.uniF[6] = this.gain;
    this.uniU[7] = offsetFloats;
    this.device.queue.writeBuffer(this.uni, 0, this.uniData);

    if (this.bindPos !== posBuf || this.bindSur !== surBuf) {
      this.bind = bindGroup(this.device, this.pipeline, [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: surBuf } },
      ]);
      this.bindPos = posBuf;
      this.bindSur = surBuf;
    }

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
    pass.setBindGroup(0, this.bind!);
    pass.draw(4, n);
    pass.end();
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

export interface GpuSurpriseStatsOpts {
  /** values copied per sampling frame (default 1024 = 4 KB) */
  sample?: number;
  /** sample every Nth frame (default 8) */
  every?: number;
}

export interface GpuSurpriseWindow {
  start: number;
  count: number;
  generation: number;
}

/**
 * Async, fixed-cost percentile feed for {@link GpuSurpriseRendererWebGPU}.
 *
 * Cost is O(sample), NOT O(N): one 4 KB buffer-to-buffer copy every `every`
 * frames plus one `mapAsync`. `pending` guards against issuing a second map
 * while one is in flight (mapping an already-mapped buffer is a validation
 * error), so a slow GPU degrades to fewer stat updates rather than to a crash.
 */
export class GpuSurpriseStats {
  norm: RobustSpan;
  private readonly staging: GPUBuffer;
  private readonly sample: number;
  private readonly every: number;
  private pending = false;
  private epoch = 0;
  private lastGeneration = -1;
  private pendingGeneration = -1;
  /** Normalization history belongs to exactly one packed plane. */
  private historyOffset: number | null = null;

  // `device` is a plain parameter, not a `private readonly` field: it is only
  // needed to create the staging buffer, and retaining it as a member kept an
  // unused GPUDevice reference alive on every instance.
  constructor(device: GPUDevice, opts: GpuSurpriseStatsOpts = {}) {
    this.sample = opts.sample ?? 1024;
    if (!Number.isInteger(this.sample) || this.sample < 2) {
      throw new Error(`GpuSurpriseStats: sample must be an integer >= 2, got ${this.sample}`);
    }
    this.every = Math.max(1, opts.every ?? 8);
    this.norm = new RobustSpan({ sample: this.sample });
    this.staging = device.createBuffer({
      size: this.sample * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  /**
   * Record the stats copy into this frame's encoder. Call BEFORE submit; the
   * map is kicked off by {@link afterSubmit}.
   */
  encodeSample(
    encoder: GPUCommandEncoder,
    surBuf: GPUBuffer,
    n: number,
    frame: number,
    window: GpuSurpriseWindow,
    /** First float of the selected packed N-value surprise plane. */
    offsetFloats = 0
  ): boolean {
    if (!Number.isInteger(n) || n < 0) {
      throw new Error(`GpuSurpriseStats: n must be a non-negative integer, got ${n}`);
    }
    if (!Number.isInteger(offsetFloats) || offsetFloats < 0) {
      throw new Error(
        `GpuSurpriseStats: offsetFloats must be a non-negative integer, got ${offsetFloats}`
      );
    }
    if (
      !Number.isInteger(window.start) ||
      !Number.isInteger(window.count) ||
      !Number.isInteger(window.generation) ||
      window.start < 0 ||
      window.count < 0 ||
      window.count > n ||
      (n > 0 && window.start >= n)
    ) {
      throw new Error(`GpuSurpriseStats: invalid fresh window ${JSON.stringify(window)} for n=${n}`);
    }
    // An EMA over two semantically different planes is not meaningful. Plane
    // selection therefore starts a fresh history epoch before deduplication.
    if (this.historyOffset === null) {
      this.historyOffset = offsetFloats;
    } else if (this.historyOffset !== offsetFloats) {
      this.reset();
      this.historyOffset = offsetFloats;
    }
    if (this.pending || frame % this.every !== 0) return false;
    if (n === 0 || window.count === 0) return false;
    if (
      window.generation === this.lastGeneration ||
      window.generation === this.pendingGeneration
    ) {
      return false;
    }
    const count = Math.min(this.sample, window.count);
    const tail = Math.min(count, n - window.start);
    if (tail > 0) {
      encoder.copyBufferToBuffer(
        surBuf,
        (offsetFloats + window.start) * 4,
        this.staging,
        0,
        tail * 4
      );
    }
    const wrapped = count - tail;
    if (wrapped > 0) {
      encoder.copyBufferToBuffer(
        surBuf,
        offsetFloats * 4,
        this.staging,
        tail * 4,
        wrapped * 4
      );
    }
    const bytes = count * 4;
    this.pendingBytes = bytes;
    this.pendingGeneration = window.generation;
    this.pendingEpoch = this.epoch;
    return true;
  }

  private pendingBytes = 0;
  private pendingEpoch = 0;

  /** Kick off the async map for a copy recorded this frame. Never blocks. */
  afterSubmit(recorded: boolean): void {
    if (!recorded || this.pending) return;
    this.pending = true;
    const bytes = this.pendingBytes;
    const generation = this.pendingGeneration;
    const epoch = this.pendingEpoch;
    this.staging
      .mapAsync(GPUMapMode.READ, 0, bytes)
      .then(() => {
        const view = new Float32Array(this.staging.getMappedRange(0, bytes).slice(0));
        this.staging.unmap();
        this.pending = false;
        this.pendingGeneration = -1;
        if (epoch === this.epoch) {
          this.norm.update(view, view.length);
          this.lastGeneration = generation;
        }
      })
      .catch(() => {
        // A lost/destroyed device rejects the map. Clearing `pending` keeps the
        // next frame's attempt legal; the span simply stops updating, which the
        // HUD shows as a frozen lo/hi rather than as an invented range.
        this.pending = false;
        this.pendingGeneration = -1;
      });
  }

  /** Start a new particle-buffer epoch. Any in-flight old readback is ignored. */
  reset(): void {
    this.epoch++;
    this.lastGeneration = -1;
    this.historyOffset = null;
    this.norm = new RobustSpan({ sample: this.sample });
  }

  destroy(): void {
    try {
      this.staging.destroy();
    } catch (_) {}
  }
}
