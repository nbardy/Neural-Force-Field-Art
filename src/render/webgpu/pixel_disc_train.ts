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
  PIXEL_STATS,
  PIXEL_STATS_WIN_BASE,
  pixelStatsFloats,
  pixelWinCounters,
} from "./pixel_disc_wgsl";

const ADAM = { beta1: 0.9, beta2: 0.999, eps: 1e-7 } as const;

/**
 * The 1 Hz readback of `critMeta`'s stats tail. One field per slot some kernel
 * actually writes — see PIXEL_STATS in pixel_disc_wgsl.ts for the layout and
 * for why `meanFx`/`meanFy` are gone (nothing ever wrote them).
 */
export interface PixelDiscStats {
  discLoss: number;
  genLoss: number;
  /**
   * vec-field ONLY: F(centre of cell 0), the TARGET the critic head is fitting.
   * `null` for the three kinds that HAVE no such quantity — absence is the
   * meaning here, so it is not stood in for by a 0 the caller could mistake for
   * a measurement. Formerly `predX`/`predY`, which named it a prediction.
   */
  targetF: { x: number; y: number } | null;
  /**
   * ACTIVE cells won by each guess this step (plan §3f) — the collapse detector.
   * `winCounts[j] === 0` while the others are large means guess `j` is receiving
   * only the ε loser share, will never move, and will never win: K has silently
   * degraded and the loss will not say so. Inactive cells are excluded (§3g).
   *
   * `null` for `real-fake`, which has no per-cell winner to count — absence is
   * the meaning, not "zero wins".
   */
  winCounts: number[] | null;
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

export interface PixelDiscHistoricalReplayOpts {
  /** Number of detached G×G real-density snapshots retained on the GPU. */
  capacity: number;
  /** Capture one live snapshot every N discriminator steps. */
  captureEvery?: number;
  /** Probability of using an old snapshot as the negative example. */
  probability?: number;
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
  private readonly historyBuf: GPUBuffer | null;
  /**
   * Floats in `critMeta`'s stats tail: the named scalars plus one §3f win
   * counter per guess. An INSTANCE value, not a module constant, because the
   * counter count is a property of `dims` — buffer size, staging size, per-step
   * clear and readback all read it, so they cannot disagree about the length.
   */
  private readonly statsFloats: number;
  private readonly uniData = new ArrayBuffer(64);
  private partPos: GPUBuffer | null = null;
  private partDummy: GPUBuffer;
  private partCount = 0;
  private cursor = 0;
  private historyWrite = 0;
  private historyCount = 0;
  private historyStep = 0;
  private readonly historyCaptureEvery: number;
  private readonly historyProbability: number;
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
      historicalReplay?: PixelDiscHistoricalReplayOpts;
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
    // scratch also carries the critic's per-cell workspace (cFeat/cSoft/dSoft/
    // gf) now that criticDisc/criticGen are workgroup-parallel, plus one
    // field-eval site per cell for vec-field's fillForceGrid. pixelScratchBytes
    // derives both from dims so the sizes cannot drift from the shader's bases.
    this.scratchBuf = mk(pixelScratchBytes(field, this.batchCap, this.dims));
    this.densI32 = mk(nCell * 4);
    this.densPack = mk(densPackFloats(this.dims.G) * 4);
    // grads | adamM | adamV | stats. The stats tail is sized from the shared
    // layout helper so host and shader cannot disagree about its length, and so
    // the per-guess win counters of PLAN_MULTI_GUESS_MODULARIZATION §3f arrive
    // as an argument to pixelStatsFloats rather than as a re-layout here.
    this.statsFloats = pixelStatsFloats(pixelWinCounters(this.dims));
    this.metaBuf = mk((this.nWeights * 3 + this.statsFloats) * 4);
    this.extGradsBuf = mk(field.totalFloats * 4);
    const history = opts.historicalReplay;
    if (history && this.kind !== "real-fake") {
      throw new Error("pixel_disc: historical replay is only supported for real-fake");
    }
    if (history && (!Number.isInteger(history.capacity) || history.capacity <= 0)) {
      throw new Error(`pixel_disc: bad historical replay capacity=${history?.capacity}`);
    }
    if (
      history?.captureEvery !== undefined &&
      (!Number.isInteger(history.captureEvery) || history.captureEvery <= 0)
    ) {
      throw new Error(`pixel_disc: bad historical replay captureEvery=${history.captureEvery}`);
    }
    if (
      history?.probability !== undefined &&
      (!Number.isFinite(history.probability) || history.probability < 0 || history.probability > 1)
    ) {
      throw new Error(`pixel_disc: bad historical replay probability=${history.probability}`);
    }
    const historyCapacity = history?.capacity ?? 0;
    this.historyBuf = history
      ? device.createBuffer({
          size: historyCapacity * nCell * 4,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
      : null;
    this.historyCaptureEvery = history?.captureEvery ?? 1;
    this.historyProbability = history?.probability ?? 1;
    this.uniBuf = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.partDummy = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    this.statsStaging = device.createBuffer({
      size: this.statsFloats * 4,
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
    this.historyWrite = 0;
    this.historyCount = 0;
    this.historyStep = 0;
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
    // Zero the stats tail EVERY step, before anything writes it. `genLoss` is
    // written only when the generator pass runs and `targetFx/Fy` only by the
    // vec-field kind, so without this a conditional slot keeps reporting the
    // last step that happened to fill it as if it were current. The clear spans
    // the whole region, so §3f's win counters inherit the same guarantee.
    encoder.clearBuffer(this.metaBuf, this.nWeights * 3 * 4, this.statsFloats * 4);

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
    /**
     * EXACTLY ONE workgroup — load-bearing, not a leftover from when
     * criticDisc/criticGen were `@workgroup_size(1)`.
     *
     * Both critics are now `@workgroup_size(PIXEL_DISC_WG)` and phase-separated
     * by workgroupBarrier/storageBarrier plus workgroup reductions (softmax
     * normalizers, active-cell counts, summed residuals). Those synchronise
     * within a workgroup only, so dispatching 2+ would silently give each extra
     * workgroup its own partial reductions and let them race on the shared
     * per-cell scratch. The kernels grid-stride over G² instead.
     */
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

    // RealFake's historical population is a detached replay of prior live
    // density images. Capture the current positive example before replacing
    // the auxiliary region with an old negative example for this step.
    const availableHistoryCount = this.historyCount;
    const replayHit = this.historyBuf !== null &&
      availableHistoryCount > 0 &&
      this.historyRandom() < this.historyProbability;
    if (this.historyBuf && this.historyStep % this.historyCaptureEvery === 0) {
      const nCellBytes = nCell * 4;
      encoder.copyBufferToBuffer(
        this.densPack,
        0,
        this.historyBuf,
        this.historyWrite * nCellBytes,
        nCellBytes
      );
      this.historyWrite = (this.historyWrite + 1) % (this.historyBuf.size / nCellBytes);
      this.historyCount = Math.min(this.historyCount + 1, this.historyBuf.size / nCellBytes);
    }

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
        if (replayHit) {
          const nCellBytes = nCell * 4;
          const slot = this.historyRandomInt(availableHistoryCount);
          encoder.copyBufferToBuffer(
            this.historyBuf!,
            slot * nCellBytes,
            this.densPack,
            2 * nCellBytes,
            nCellBytes
          );
        } else {
          dispatch(this.pipeClearAtomics, nCell);
          dispatch(this.pipeFakeSplat, b);
          dispatch(this.pipeDensFake, nCell);
        }
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
    this.historyStep++;
  }

  private historyRandom(): number {
    let x = (this.historyStep * 1664525 + 1013904223) >>> 0;
    x ^= x >>> 16;
    return x / 0x100000000;
  }

  private historyRandomInt(n: number): number {
    let x = (this.historyStep * 22695477 + 1) >>> 0;
    x ^= x >>> 15;
    return n > 0 ? x % n : 0;
  }

  /** Copy stats into the MAP_READ staging buffer. Call before submit. */
  recordStats(encoder: GPUCommandEncoder): boolean {
    if (this.statsPending) return false;
    encoder.copyBufferToBuffer(
      this.metaBuf,
      this.nWeights * 3 * 4,
      this.statsStaging,
      0,
      this.statsFloats * 4
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
        // The one canonicalization point: raw f32 slots -> PixelDiscStats.
        // `kind` is fixed at construction, so this decides once whether the
        // vec-field target slots carry a measurement at all.
        const nWin = pixelWinCounters(this.dims);
        this.lastStats = {
          discLoss: s[PIXEL_STATS.discLoss],
          genLoss: s[PIXEL_STATS.genLoss],
          targetF:
            this.kind === "vec-field"
              ? { x: s[PIXEL_STATS.targetFx], y: s[PIXEL_STATS.targetFy] }
              : null,
          winCounts:
            nWin === 0
              ? null
              : Array.from(s.subarray(PIXEL_STATS_WIN_BASE, PIXEL_STATS_WIN_BASE + nWin)),
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
