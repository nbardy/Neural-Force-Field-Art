/**
 * PixelDiscTrainer — fused pixel-space GAN (WebGPU), four kinds.
 *
 * Owns critic weights + Adam, densPack, and extGradsBuf for the field
 * trainer's pass B (same seam as AdversaryTrainer). Oracle:
 * src/core/gan/pixel_disc.ts. Spec: docs/PIXEL_DISC.md.
 */

import type { FieldLayout } from "./advect_wgsl";
import type { PassTimestampWrites } from "./gputime";
import {
  initPixelDiscWeights,
  packPixelDiscWeights,
  pixelDiscWeightCount,
  type PixelDiscDims,
  type PixelGanKind,
} from "../../core/gan/pixel_disc";
import {
  PIXEL_DISC_WG,
  PIXEL_DISC_MAX_BATCH,
  pixelDiscShader,
  pixelScratchBytes,
  densPackFloats,
  resolvePixelDims,
  validatePixelDiscFusion,
  pixelWeightLayout,
} from "./pixel_disc_wgsl";

const ADAM = { beta1: 0.9, beta2: 0.999, eps: 1e-7 } as const;

export interface PixelDiscStats {
  discLoss: number;
  genLoss: number;
  predX: number;
  predY: number;
  meanFx: number;
  meanFy: number;
}

export interface PixelDiscStepOpts {
  b: number;
  alpha: number;
  lr: number;
  /** Non-negative magnitude; disagree uses L_gen = -weight * R (or −logit). */
  genWeight: number;
  applyDisc?: boolean;
  width: number;
  height: number;
  /** Inpaint mask / fake RNG seed (frame index). */
  maskSeed?: number;
}

export class PixelDiscTrainer {
  readonly field: FieldLayout;
  readonly dims: PixelDiscDims;
  readonly kind: PixelGanKind;
  readonly batchCap: number;
  readonly extGradsBuf: GPUBuffer;
  readonly nWeights: number;

  private readonly device: GPUDevice;
  private readonly fieldWeightsBuf: GPUBuffer;
  private readonly critWBuf: GPUBuffer;
  private readonly scratchBuf: GPUBuffer;
  private readonly densI32: GPUBuffer;
  private readonly densPack: GPUBuffer;
  private readonly metaBuf: GPUBuffer;
  private readonly uniBuf: GPUBuffer;
  private readonly uniData = new ArrayBuffer(64);
  private partPos: GPUBuffer | null = null;
  private partDummy: GPUBuffer;
  private partCount = 0;
  private cursor = 0;
  private adamStep = 0;
  private frame = 0;
  lastStats: PixelDiscStats | null = null;

  private readonly pipeClear: GPUComputePipeline;
  private readonly pipeClearAtomics: GPUComputePipeline;
  private readonly pipeSample: GPUComputePipeline;
  private readonly pipeDensF: GPUComputePipeline;
  private readonly pipeCopyAux: GPUComputePipeline;
  private readonly pipeFakeSplat: GPUComputePipeline;
  private readonly pipeDensFake: GPUComputePipeline;
  /** vec-field only: parallel F(cell_center) fill. Null for the other kinds. */
  private readonly pipeForceGrid: GPUComputePipeline | null;
  private readonly pipeCriticDisc: GPUComputePipeline;
  private readonly pipeAdam: GPUComputePipeline;
  private readonly pipeClearGen: GPUComputePipeline;
  private readonly pipeVirtual: GPUComputePipeline;
  private readonly pipeCriticGen: GPUComputePipeline;
  private readonly pipeVjp: GPUComputePipeline;
  private readonly pipeFieldGrad: GPUComputePipeline;
  private readonly bgl: GPUBindGroupLayout;
  private bind: GPUBindGroup;
  private statsStaging: GPUBuffer;
  private statsPending = false;

  constructor(
    device: GPUDevice,
    field: FieldLayout,
    opts: {
      fieldWeightsBuffer: GPUBuffer;
      dims?: Partial<PixelDiscDims> & { kind?: PixelGanKind };
      batchCap?: number;
      fieldLane?: "blend" | 0 | 1;
      seed?: number;
    }
  ) {
    validatePixelDiscFusion(field);
    this.device = device;
    this.field = field;
    this.dims = resolvePixelDims(opts.dims);
    this.kind = this.dims.kind;
    this.batchCap = Math.min(opts.batchCap ?? 512, PIXEL_DISC_MAX_BATCH);
    this.fieldWeightsBuf = opts.fieldWeightsBuffer;
    this.nWeights = pixelDiscWeightCount(this.dims);

    const mk = (bytes: number) =>
      device.createBuffer({
        size: bytes,
        usage:
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

    this.critWBuf = mk(this.nWeights * 4);
    const nCell = this.dims.G * this.dims.G;
    // vec-field runs fillForceGrid one cell per invocation, so it needs nCell
    // critic workspaces; the other kinds have no force grid and keep one.
    this.scratchBuf = mk(
      pixelScratchBytes(field, this.batchCap, this.kind === "vec-field" ? nCell : 1)
    );
    this.densI32 = mk(nCell * 4);
    this.densPack = mk(densPackFloats(this.dims.G) * 4);
    this.metaBuf = mk((this.nWeights * 3 + 8) * 4);
    this.extGradsBuf = mk(field.totalFloats * 4);
    this.uniBuf = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.partDummy = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    this.statsStaging = device.createBuffer({
      size: 8 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const init = packPixelDiscWeights(
      initPixelDiscWeights(this.dims, opts.seed ?? 20260805),
      this.dims
    );
    device.queue.writeBuffer(this.critWBuf, 0, init);

    const shader = pixelDiscShader(field, {
      dims: this.dims,
      batchCap: this.batchCap,
      fieldLane: opts.fieldLane ?? "blend",
    });
    const mod = device.createShaderModule({ code: shader });
    this.bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        {
          binding: 8,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    const mkPipe = (entryPoint: string) =>
      device.createComputePipeline({
        layout,
        compute: { module: mod, entryPoint },
      });
    this.pipeClear = mkPipe("clearDens");
    this.pipeClearAtomics = mkPipe("clearAtomics");
    this.pipeSample = mkPipe("sampleAndSplat");
    this.pipeDensF = mkPipe("densToFloat");
    this.pipeCopyAux = mkPipe("copyDensToAux");
    this.pipeFakeSplat = mkPipe("fakeSplat");
    this.pipeDensFake = mkPipe("densToFloatFake");
    this.pipeForceGrid = this.kind === "vec-field" ? mkPipe("fillForceGrid") : null;
    this.pipeCriticDisc = mkPipe("criticDisc");
    this.pipeAdam = mkPipe("discAdam");
    this.pipeClearGen = mkPipe("clearDensGen");
    this.pipeVirtual = mkPipe("virtualSplat");
    this.pipeCriticGen = mkPipe("criticGen");
    this.pipeVjp = mkPipe("densityVjpAndFieldBwd");
    this.pipeFieldGrad = mkPipe("fieldGrad");

    this.bind = this.makeBind(this.partDummy);
  }

  private makeBind(partPos: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.uniBuf } },
        { binding: 1, resource: { buffer: this.fieldWeightsBuf } },
        { binding: 2, resource: { buffer: this.critWBuf } },
        { binding: 3, resource: { buffer: this.scratchBuf } },
        { binding: 4, resource: { buffer: this.densI32 } },
        { binding: 5, resource: { buffer: this.densPack } },
        { binding: 6, resource: { buffer: this.metaBuf } },
        { binding: 7, resource: { buffer: this.extGradsBuf } },
        { binding: 8, resource: { buffer: partPos } },
      ],
    });
  }

  setParticleBuffers(pos: GPUBuffer, _vel: GPUBuffer, count: number): void {
    this.partPos = pos;
    this.partCount = count;
    this.bind = this.makeBind(pos);
    this.cursor = 0;
  }

  uploadCriticWeights(w: Float32Array): void {
    if (w.length !== this.nWeights) {
      throw new Error(`pixel_disc: expected ${this.nWeights} weights, got ${w.length}`);
    }
    this.device.queue.writeBuffer(this.critWBuf, 0, w);
  }

  async readCriticWeights(): Promise<Float32Array> {
    return this.read(this.critWBuf, this.nWeights);
  }

  async readExtGrads(): Promise<Float32Array> {
    return this.read(this.extGradsBuf, this.field.totalFloats);
  }

  private async read(buf: GPUBuffer, n: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /**
   * One alternating disc→gen step into a caller-owned encoder.
   * Clears and rewrites extGradsBuf every call.
   */
  encodeStep(
    encoder: GPUCommandEncoder,
    o: PixelDiscStepOpts,
    ts?: PassTimestampWrites
  ): void {
    if (!this.partPos) {
      throw new Error("pixel_disc: setParticleBuffers first");
    }
    if (!Number.isInteger(o.b) || o.b <= 0 || o.b > this.batchCap) {
      throw new Error(`pixel_disc: bad b=${o.b}`);
    }
    const b = Math.min(o.b, this.partCount);
    if (b === 0) {
      encoder.clearBuffer(this.extGradsBuf);
      return;
    }
    const applyDisc = o.applyDisc ?? true;
    if (applyDisc) this.adamStep++;
    const maskSeed = (o.maskSeed ?? this.frame++) >>> 0;

    const f = new Float32Array(this.uniData);
    const u = new Uint32Array(this.uniData);
    f[0] = o.width;
    f[1] = o.height;
    f[2] = o.alpha;
    f[3] = this.dims.dt;
    u[4] = b;
    u[5] = this.partCount;
    u[6] = this.cursor;
    u[7] = applyDisc ? 1 : 0;
    f[8] = o.lr;
    f[9] = ADAM.beta1;
    f[10] = ADAM.beta2;
    f[11] = ADAM.eps;
    u[12] = this.adamStep;
    // disagree: maximize residual / real-score ⇒ L_gen = -|weight| · R
    f[13] = o.genWeight === 0 ? 0 : -Math.abs(o.genWeight);
    u[14] = maskSeed;
    u[15] = 0;
    this.device.queue.writeBuffer(this.uniBuf, 0, this.uniData);

    const nCell = this.dims.G * this.dims.G;
    const dispatch = (pipe: GPUComputePipeline, n: number) => {
      const pass = encoder.beginComputePass(
        ts ? { timestampWrites: ts } : undefined
      );
      pass.setPipeline(pipe);
      pass.setBindGroup(0, this.bind);
      pass.dispatchWorkgroups(Math.ceil(Math.max(n, 1) / PIXEL_DISC_WG));
      pass.end();
    };
    const dispatch1 = (pipe: GPUComputePipeline) => {
      const pass = encoder.beginComputePass(
        ts ? { timestampWrites: ts } : undefined
      );
      pass.setPipeline(pipe);
      pass.setBindGroup(0, this.bind);
      pass.dispatchWorkgroups(1);
      pass.end();
    };

    encoder.clearBuffer(this.extGradsBuf);
    dispatch(this.pipeClear, nCell);
    dispatch(this.pipeSample, b);
    dispatch(this.pipeDensF, nCell);

    // Targets for vec-field: one invocation per cell, ahead of the single-thread
    // critic that reads them. Was an inline serial loop inside criticDisc.
    if (this.pipeForceGrid) dispatch(this.pipeForceGrid, nCell);

    switch (this.kind) {
      case "next-frame":
        dispatch(this.pipeCopyAux, nCell);
        dispatch(this.pipeClearGen, nCell);
        dispatch(this.pipeVirtual, b);
        dispatch(this.pipeDensF, nCell);
        break;
      case "real-fake":
        dispatch(this.pipeClearAtomics, nCell);
        dispatch(this.pipeFakeSplat, b);
        dispatch(this.pipeDensFake, nCell);
        break;
      case "vec-field":
      case "inpaint":
        break;
      default: {
        const _e: never = this.kind;
        throw new Error(`pixel_disc: bad kind ${_e}`);
      }
    }

    dispatch1(this.pipeCriticDisc);
    dispatch(this.pipeAdam, this.nWeights);

    if (o.genWeight !== 0) {
      dispatch(this.pipeClearGen, nCell);
      dispatch(this.pipeVirtual, b);
      dispatch1(this.pipeCriticGen);
      dispatch(this.pipeVjp, b);
      dispatch(this.pipeFieldGrad, this.field.totalFloats);
    }

    this.cursor = (this.cursor + b) % Math.max(this.partCount, 1);
  }

  /** Copy stats into the MAP_READ staging buffer. Call before submit. */
  recordStats(encoder: GPUCommandEncoder): boolean {
    if (this.statsPending) return false;
    encoder.copyBufferToBuffer(
      this.metaBuf,
      this.nWeights * 3 * 4,
      this.statsStaging,
      0,
      8 * 4
    );
    return true;
  }

  /** Kick async stats readback AFTER the encoder that recorded them is submitted. */
  afterSubmit(recorded: boolean): void {
    if (!recorded || this.statsPending) return;
    this.statsPending = true;
    this.statsStaging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const s = new Float32Array(this.statsStaging.getMappedRange().slice(0));
        this.statsStaging.unmap();
        this.statsPending = false;
        this.lastStats = {
          discLoss: s[0],
          genLoss: s[1],
          predX: s[2],
          predY: s[3],
          meanFx: s[4],
          meanFy: s[5],
        };
      })
      .catch(() => {
        this.statsPending = false;
      });
  }

  destroy(): void {
    for (const b of [
      this.critWBuf,
      this.scratchBuf,
      this.densI32,
      this.densPack,
      this.metaBuf,
      this.extGradsBuf,
      this.uniBuf,
      this.partDummy,
      this.statsStaging,
    ]) {
      b.destroy();
    }
  }
}

export { resolvePixelDims, pixelWeightLayout };
