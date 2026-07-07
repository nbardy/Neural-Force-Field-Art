/**
 * AdvectKernel — the tfjs-coupled half of the fused advect pass.
 *
 * Owns the particle state (pos/vel GPUBuffers — NOT tfjs tensors) and the one
 * fused compute dispatch that replaces the whole tfjs advect stage. tfjs keeps
 * doing `learn` (Adam on a small random batch — it samples random points, so
 * it never needs the particle state at all); each frame the trained weights
 * flow tfjs → kernel as ~10KB of GPU→GPU copies (`dataToGPU` +
 * `copyBufferToBuffer`, zero readback), then ONE dispatch advects every
 * particle, and the renderer draws straight from the kernel's buffers.
 *
 * The WGSL itself is generated per-model by ./advect_wgsl.ts (pure, headless-
 * tested on a real Metal adapter via `bun tools/kernel_test.ts`). Dims are
 * read off the LIVE tf.Sequential layers and cross-checked against the packed
 * layout — a mismatch throws at construction (no silent AlphaGOJS D=8 trap).
 */

import * as tf from "@tensorflow/tfjs";
import {
  layoutField,
  advectShader,
  totalMacs,
  WORKGROUP_SIZE,
  MAX_PARTICLES,
  UNROLL_MAC_LIMIT,
  type Activation,
  type FieldLayout,
  type LayerDims,
} from "./advect_wgsl";
import { computePipeline } from "./microgpu";
import type { PassTimestampWrites } from "./gputime";
import type { HelmholtzField } from "../../core/field/helmholtz";

/** Per-piece physics constants baked into the uniform once at construction. */
export interface AdvectPhysics {
  width: number;
  height: number;
  forceMagnitude: number;
  friction: number;
  maxVelocity: number;
  resetRate: number;
}

interface IngestedHead {
  dims: LayerDims[];
  /** kernel0, bias0, kernel1, bias1, … — pairs 1:1 with layout segments */
  vars: tf.Variable[];
}

/**
 * κ: read a tf.Sequential's dense stack into canonical LayerDims + the
 * underlying tf.Variables. Throws on anything the kernel can't evaluate —
 * never silently skips a layer.
 */
function ingestSequential(net: tf.Sequential, label: string): IngestedHead {
  const dims: LayerDims[] = [];
  const vars: tf.Variable[] = [];
  for (const layer of net.layers) {
    if (layer.getClassName() !== "Dense") {
      throw new Error(
        `advect: layer '${layer.name}' in ${label} is ${layer.getClassName()} — only Dense is supported`
      );
    }
    const lw = layer.trainableWeights;
    if (lw.length !== 2) {
      throw new Error(
        `advect: dense '${layer.name}' in ${label} needs kernel+bias (useBias:true), got ${lw.length} weights`
      );
    }
    const kShape = lw[0].shape;
    const bShape = lw[1].shape;
    const inSize = kShape[0];
    const outSize = kShape[1];
    if (
      kShape.length !== 2 || bShape.length !== 1 ||
      inSize == null || outSize == null || bShape[0] !== outSize
    ) {
      throw new Error(
        `advect: dense '${layer.name}' in ${label} has unexpected shapes kernel=[${kShape}] bias=[${bShape}]`
      );
    }
    // Serialized activation identifier ('selu' | 'tanh' | …); layoutField
    // validates it against the supported set and throws otherwise.
    const act = String(
      (layer.getConfig() as { activation?: string }).activation ?? "linear"
    ) as Activation;
    dims.push({ inSize, outSize, activation: act });
    // LayerVariable.val is the underlying tf.Variable (protected in the
    // typings) — same access trick as HelmholtzField.trainableWeights.
    vars.push(
      (lw[0] as unknown as { val: tf.Variable }).val,
      (lw[1] as unknown as { val: tf.Variable }).val
    );
  }
  return { dims, vars };
}

export class AdvectKernel {
  /** Field path: two vector heads blended by the live alpha uniform. */
  static fromField(
    field: HelmholtzField,
    physics: AdvectPhysics,
    particleCount: number
  ): AdvectKernel {
    const [g, r] = field.heads;
    const hg = ingestSequential(g, "helmholtz.g");
    const hr = ingestSequential(r, "helmholtz.r");
    // Override activations from the FIELD, not the tf config: SIREN builds
    // LINEAR tf layers and applies sin/tanh manually, so the config reads
    // "linear" — the field declares the real hidden activation. Standard is a
    // no-op (selu/tanh already). Hidden layers → hiddenActivation, last → tanh.
    const hidden = field.hiddenActivation;
    const remap = (dims: LayerDims[]) =>
      dims.map((d, i) => ({
        ...d,
        activation: (i === dims.length - 1 ? "tanh" : hidden) as Activation,
      }));
    // classes + encoding come from the field config; layoutField validates
    // that r's live layer-0 input width matches encDim(+classes) (D-trap rule).
    const encoding =
      field.modelType === "fourier"
        ? ({ kind: "fourier", octaves: field.fourierOctaves } as const)
        : field.modelType === "hashgrid"
        ? ({ kind: "hashgrid", gridSize: field.gridSize, features: field.gridFeatures } as const)
        : ({ kind: "raw" } as const);
    const layout = layoutField("helmholtz", [remap(hg.dims), remap(hr.dims)], {
      classes: field.classes ?? 0,
      encoding,
    });
    // hashgrid: the grid tf.Variable is FIRST (matches the "grid" segment at
    // offset 0); then the head variables.
    const gridVar = field.grid ? [field.grid] : [];
    return new AdvectKernel(
      layout,
      [...gridVar, ...hg.vars, ...hr.vars],
      physics,
      particleCount
    );
  }

  /** Legacy path: single sigmoid MLP, output re-centered by -0.5 in-shader. */
  static fromModel(
    model: tf.Sequential,
    physics: AdvectPhysics,
    particleCount: number
  ): AdvectKernel {
    const hm = ingestSequential(model, "mlp");
    return new AdvectKernel(layoutField("mlp", [hm.dims]), hm.vars, physics, particleCount);
  }

  private readonly device: GPUDevice;
  /** Packed-weights layout — share with a FusedTrainer to co-own weights. */
  readonly layout: FieldLayout;
  private readonly vars: tf.Variable[];
  private readonly pipeline: GPUComputePipeline;
  private readonly weightsBuf: GPUBuffer;
  /**
   * When a FusedTrainer updates the shared weights buffer in place, set this
   * false — step() then skips the per-frame tfjs sync entirely (weights are
   * born on the GPU and never touch tfjs again).
   */
  syncFromTfjs = true;
  private readonly uni: GPUBuffer;
  private readonly uniData = new ArrayBuffer(48);
  private readonly uniF = new Float32Array(this.uniData);
  private readonly uniU = new Uint32Array(this.uniData);

  private posBuf!: GPUBuffer;
  private velBuf!: GPUBuffer;
  private bind!: GPUBindGroup;
  private particleCount = 0;

  private constructor(
    layout: FieldLayout,
    vars: tf.Variable[],
    physics: AdvectPhysics,
    particleCount: number
  ) {
    const device = (tf.backend() as unknown as { device?: GPUDevice }).device;
    if (!device) {
      throw new Error(
        "AdvectKernel: tf.backend().device is missing — set the 'webgpu' " +
          "backend and await tf.ready() before constructing."
      );
    }
    this.device = device;
    this.layout = layout;
    this.vars = vars;

    // The layout's segments were derived from the same dims as `vars`; verify
    // the pairing anyway so a future ingestion change fails HERE, loudly.
    if (vars.length !== layout.segments.length) {
      throw new Error(
        `advect: ${vars.length} variables vs ${layout.segments.length} packed segments`
      );
    }
    layout.segments.forEach((seg, i) => {
      const size = vars[i].shape.reduce((a, b) => a * b, 1);
      if (size !== seg.floatLength) {
        throw new Error(
          `advect: variable ${i} has ${size} floats but segment expects ${seg.floatLength}`
        );
      }
    });

    const stageWeights =
      layout.totalFloats * 4 <= device.limits.maxComputeWorkgroupStorageSize;
    // f16 fast path (measured ~9.3 → ~6.5 ms/step @ 1M on Apple Metal): only
    // the unrolled+staged codegen has an f16 variant, and the FEATURE must be
    // on the device — tfjs creates the device and requests only its own
    // features, so main.ts wraps GPUAdapter.requestDevice to append
    // "shader-f16" (device features are fixed at creation time).
    // Fourier (and any non-raw encoding) forces the LOOPED emitter, which has
    // no f16 variant — so it only qualifies for f16 when unrolled AND raw.
    const unrolled =
      totalMacs(layout) <= UNROLL_MAC_LIMIT && layout.encoding.kind === "raw";
    const precision: "f32" | "f16" =
      device.features.has("shader-f16") && stageWeights && unrolled
        ? "f16"
        : "f32";
    this.pipeline = computePipeline(
      device,
      advectShader(layout, { stageWeights, precision })
    );

    // COPY_SRC: a co-owning FusedTrainer reads this buffer back in tests
    // (readWeights) — and future checkpointing will too.
    this.weightsBuf = device.createBuffer({
      size: layout.totalFloats * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.uni = device.createBuffer({
      size: this.uniData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Physics constants live at fixed uniform slots; step() only touches
    // alpha/seed/count. Layout must match the WGSL Uni struct in advect_wgsl.
    this.uniF[0] = physics.width;
    this.uniF[1] = physics.height;
    this.uniF[2] = physics.forceMagnitude;
    this.uniF[3] = physics.friction;
    this.uniF[4] = physics.maxVelocity;
    this.uniF[6] = physics.resetRate;

    this.allocParticles(particleCount);

    console.log(
      `[advect] fused kernel: ${layout.spec.kind}, ${layout.totalFloats} weight ` +
        `floats, ${totalMacs(layout)} MACs/particle, staged=${stageWeights}, ` +
        `unrolled=${unrolled}, precision=${precision}, n=${particleCount}`
    );
  }

  get posBuffer(): GPUBuffer {
    return this.posBuf;
  }
  get velBuffer(): GPUBuffer {
    return this.velBuf;
  }
  get count(): number {
    return this.particleCount;
  }
  /** The packed weights buffer (bind into a FusedTrainer to train in place). */
  get weightsBuffer(): GPUBuffer {
    return this.weightsBuf;
  }

  /**
   * Snapshot the CURRENT tfjs weights into packed-layout order — used to seed
   * a FusedTrainer once before turning syncFromTfjs off. dataSync is free at
   * init time (fresh layer variables are CPU-resident).
   */
  packCurrentWeights(): Float32Array {
    const out = new Float32Array(this.layout.totalFloats);
    this.layout.segments.forEach((seg, i) => {
      out.set(this.vars[i].dataSync() as Float32Array, seg.floatOffset);
    });
    return out;
  }

  /**
   * One frame: sync current tfjs weights into the packed buffer (GPU→GPU) and
   * run the fused advect dispatch over all particles. `seed` varies the reset
   * RNG per frame (frame counter is perfect); `alpha` is the live order↔chaos
   * mix (ignored by 'mlp' kind).
   */
  step(seed: number, alpha: number): void {
    const encoder = this.device.createCommandEncoder();
    const refs = this.recordStep(encoder, seed, alpha);
    this.device.queue.submit([encoder.finish()]);
    // Release the tensor clones tfjs minted for dataToGPU (after submit —
    // the queue keeps the source buffers alive for the in-flight copies).
    for (const t of refs) t.dispose();
  }

  /**
   * Same weight-sync + advect pass as {@link step}, recorded into a
   * CALLER-owned encoder and NOT submitted — lets a whole frame collapse to one
   * queue.submit. Returns the tfjs dataToGPU tensor clones (empty when
   * syncFromTfjs is off); the CALLER must dispose them AFTER its submit so the
   * source buffers survive the in-flight copies. `ts` optionally timestamps the
   * advect pass.
   */
  encodeStep(
    encoder: GPUCommandEncoder,
    seed: number,
    alpha: number,
    ts?: PassTimestampWrites
  ): tf.Tensor[] {
    return this.recordStep(encoder, seed, alpha, ts);
  }

  private recordStep(
    encoder: GPUCommandEncoder,
    seed: number,
    alpha: number,
    ts?: PassTimestampWrites
  ): tf.Tensor[] {
    this.uniF[5] = alpha;
    this.uniU[7] = seed >>> 0;
    this.uniU[8] = this.particleCount;
    this.device.queue.writeBuffer(this.uni, 0, this.uniData);

    // Weights sync: ~10KB per frame, one segment per variable. A tfjs tensor
    // is in one of TWO residency states the API won't let us query directly:
    //  - GPU-resident → dataToGPU() hands us its GPUBuffer; copyBufferToBuffer
    //    into the packed buffer (no readback). dataToGPU may flush tfjs's own
    //    pending submits — fine, ours lands after in queue order.
    //  - CPU-resident → dataToGPU() throws "Data is not on GPU but on CPU".
    //    This is COMMON here, not an edge case: tfjs-webgpu forwards small-
    //    tensor ops to the CPU backend (WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD),
    //    and our weights are tiny, so even Adam-updated values can live CPU-
    //    side. Then dataSync() is a free read and writeBuffer uploads it.
    // Both paths land identical bytes. Found by tools/integration_test.ts.
    const refs: tf.Tensor[] = [];
    for (let i = 0; this.syncFromTfjs && i < this.vars.length; i++) {
      const seg = this.layout.segments[i];
      let gd: { buffer: GPUBuffer; tensorRef: tf.Tensor } | null = null;
      try {
        gd = this.vars[i].dataToGPU() as unknown as {
          buffer: GPUBuffer;
          tensorRef: tf.Tensor;
        };
      } catch (e) {
        if (!String(e).includes("not on GPU")) throw e;
      }
      if (gd) {
        encoder.copyBufferToBuffer(
          gd.buffer,
          0,
          this.weightsBuf,
          seg.floatOffset * 4,
          seg.floatLength * 4
        );
        refs.push(gd.tensorRef);
      } else {
        this.device.queue.writeBuffer(
          this.weightsBuf,
          seg.floatOffset * 4,
          this.vars[i].dataSync() as unknown as BufferSource
        );
      }
    }

    // @webgpu/types 0.1.30 predates the object-form timestampWrites; the live
    // runtime uses it, so cast the descriptor (see gputime.ts).
    const pass = encoder.beginComputePass(
      (ts ? { timestampWrites: ts } : undefined) as GPUComputePassDescriptor
    );
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.dispatchWorkgroups(Math.ceil(this.particleCount / WORKGROUP_SIZE));
    pass.end();

    return refs;
  }

  /**
   * Live-update the random-reset fraction (the uniform is rewritten every
   * step). With particle-sourced training this doubles as the exploration
   * dial: resets inject fresh uniform vel-0 states into the cloud, hence
   * into the training batch.
   */
  setResetRate(r: number): void {
    this.uniF[6] = Math.max(0, Math.min(1, r));
  }

  /**
   * Live resize preserving state: grow appends fresh random particles, shrink
   * slices off the tail — same semantics the old tfjs resize path had.
   */
  setParticleCount(n: number): void {
    const nn = Math.max(1, Math.round(n));
    if (nn === this.particleCount) return;
    const oldPos = this.posBuf;
    const oldVel = this.velBuf;
    const keep = Math.min(nn, this.particleCount);

    this.createParticleBuffers(nn);
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(oldPos, 0, this.posBuf, 0, keep * 8);
    encoder.copyBufferToBuffer(oldVel, 0, this.velBuf, 0, keep * 8);
    this.device.queue.submit([encoder.finish()]);
    if (nn > keep) {
      // new tail: random positions; velocities stay zero (buffers zero-init)
      this.device.queue.writeBuffer(
        this.posBuf,
        keep * 8,
        this.randomPositions(nn - keep) as unknown as BufferSource
      );
    }
    oldPos.destroy();
    oldVel.destroy();
    this.particleCount = nn;
  }

  destroy(): void {
    for (const b of [this.posBuf, this.velBuf, this.weightsBuf, this.uni]) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }

  private randomPositions(n: number): Float32Array {
    const a = new Float32Array(2 * n);
    const w = this.uniF[0];
    const h = this.uniF[1];
    for (let i = 0; i < n; i++) {
      a[2 * i] = Math.random() * w;
      a[2 * i + 1] = Math.random() * h;
    }
    return a;
  }

  private createParticleBuffers(n: number): void {
    if (n > MAX_PARTICLES) {
      throw new Error(`advect: ${n} particles > 1D dispatch cap ${MAX_PARTICLES}`);
    }
    const mk = () =>
      this.device.createBuffer({
        size: n * 8, // vec2f per particle
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
    this.posBuf = mk();
    this.velBuf = mk();
    this.bind = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.weightsBuf } },
        { binding: 2, resource: { buffer: this.posBuf } },
        { binding: 3, resource: { buffer: this.velBuf } },
      ],
    });
  }

  private allocParticles(n: number): void {
    this.createParticleBuffers(n);
    this.device.queue.writeBuffer(
      this.posBuf,
      0,
      this.randomPositions(n) as unknown as BufferSource
    );
    this.particleCount = n;
  }
}
