/// <reference types="@webgpu/types" />
import { VisionTrainer, type TrainPlan, type WeightArray } from "../clip/vision";
import type { WeightPrecision } from "../clip/vision_wgsl";
import { BatchMajorVisionTrainer } from "../clip/vision_batch";
import { type AdamHyper, DEFAULT_HYPER } from "../splat/adam_wgsl";
import { DEFAULT_3D_CAMERAS, type Camera3D, type PreparedCamera3D, prepareCamera } from "./cameras";
import { Grid9Close2ClipLayout } from "./grid_clip";
import {
  Raster3DEngine,
  type AdamLRs3D,
  DEFAULT_3D_LRS,
  type Raster3DBatchForwardState,
  type Raster3DIOState,
} from "./raster";
import { PARAM_STRIDE_3D } from "./raster_wgsl";

const SIDE = 256;
const U = { COPY_SRC: 4, COPY_DST: 8 };

export interface Splat3DInit {
  radius?: number;
  radiusJitter?: number;
  opacityRaw?: number;
  colorSpread?: number;
  positionSpread?: number;
}

export interface Splat3DOptimizerConfig {
  G?: number;
  cap?: number;
  bg?: [number, number, number];
  seed?: number;
  init?: Splat3DInit;
  initParams?: Float32Array;
  cameras?: Camera3D[];
  lrs?: AdamLRs3D;
  hyper?: AdamHyper;
  clipBatchSize?: number;
  clipLayout?: Splat3DClipLayout;
  clipWeightPrecision?: WeightPrecision;
  stemSpatialBwd?: boolean;
  fusePointwiseGeluForward?: boolean;
  fuseGeluBwdIntoPw?: boolean;
  fuseResidualBwdIntoPw?: boolean;
  singlePassBatchRasterForward?: boolean;
  viewLaneBatchRasterForward?: boolean;
  viewLaneBatchRasterBackward?: boolean;
}

export type Splat3DClipMode = "single" | "batch";
export type Splat3DClipLayout = "per_view" | "grid9_close2";
export type Splat3DStepTimingMode = "split-submit-wall" | "gpu-timestamp";

export interface Splat3DProfileOptions {
  gpuTimestamps?: boolean;
}

export interface Splat3DStepTimings {
  views: number;
  totalViews: number;
  clipMode: Splat3DClipMode;
  clipBatchSize: number;
  timing: Splat3DStepTimingMode;
  total: number;
  clear: number;
  rasterFwd: number;
  rasterReplay: number;
  clipFwd: number;
  clipBwd: number;
  clipBatch: number;
  rasterBwd: number;
  adam: number;
  display: number;
}

export const LEGIBLE_3D_G = 4096;
export const LEGIBLE_3D_INIT: Required<Splat3DInit> = {
  radius: 0.075,
  radiusJitter: 0.35,
  opacityRaw: 0.3,
  colorSpread: 1.2,
  positionSpread: 0.9,
};

export class Splat3DOptimizer {
  readonly device: GPUDevice;
  readonly raster: Raster3DEngine;
  readonly trainer: VisionTrainer;
  readonly batchTrainer: BatchMajorVisionTrainer | null;
  readonly gridClip: Grid9Close2ClipLayout | null;
  readonly cameras: PreparedCamera3D[];
  readonly side = SIDE;
  readonly clipBatchSize: number;
  readonly clipLayout: Splat3DClipLayout;
  readonly batchRasterForward: Raster3DBatchForwardState | null;
  private readonly textBuffers: GPUBuffer[];
  private readonly gridTextBuffer: GPUBuffer | null;
  private readonly singleIO: Raster3DIOState;
  private readonly batchIO: Raster3DIOState[];
  private readonly lrs: AdamLRs3D;
  private readonly hyper: AdamHyper;
  private readonly singlePassBatchRasterForward: boolean;
  private readonly viewLaneBatchRasterForward: boolean;
  private readonly viewLaneBatchRasterBackward: boolean;
  private step_ = 0;
  private hasPrompts = false;
  private rngState = 1;
  private viewOrder: number[] = [];
  private viewCursor = 0;

  static async create(
    device: GPUDevice,
    trainPlan: TrainPlan,
    weights: WeightArray,
    cfg: Splat3DOptimizerConfig = {}
  ): Promise<Splat3DOptimizer> {
    const [ic, ih, iw] = trainPlan.inputShape;
    if (ic !== 3 || ih !== SIDE || iw !== SIDE) {
      throw new Error(`splat3d: CLIP inputShape [${ic},${ih},${iw}] != [3,${SIDE},${SIDE}]`);
    }
    const cameras = (cfg.cameras ?? DEFAULT_3D_CAMERAS).map((c) => prepareCamera(c, SIDE));
    const G = cfg.G ?? LEGIBLE_3D_G;
    const raster = await Raster3DEngine.create(device, {
      H: SIDE,
      W: SIDE,
      G,
      cap: cfg.cap ?? 2048,
      bg: cfg.bg ?? [0, 0, 0],
      cameras,
	    });
    const clipDispatchOptions = {
      weightPrecision: cfg.clipWeightPrecision,
      stemSpatialBwd: cfg.stemSpatialBwd ?? true,
      fusePointwiseGeluForward: cfg.fusePointwiseGeluForward ?? true,
      fuseGeluBwdIntoPw: cfg.fuseGeluBwdIntoPw ?? false,
      fuseResidualBwdIntoPw: cfg.fuseResidualBwdIntoPw ?? false,
    };
    const trainer = await VisionTrainer.create(device, trainPlan, weights, clipDispatchOptions);
    const clipBatchSize = normalizeClipBatchSize(cfg.clipBatchSize);
    const clipLayout = cfg.clipLayout ?? "per_view";
    const batchTrainer =
      clipBatchSize > 1
        ? await BatchMajorVisionTrainer.create(device, trainPlan, weights, clipBatchSize, {
            weightPrecision: cfg.clipWeightPrecision,
            stemSpatialBwd: clipDispatchOptions.stemSpatialBwd,
            fusePointwiseGeluForward: clipDispatchOptions.fusePointwiseGeluForward,
            fuseGeluBwdIntoPw: clipDispatchOptions.fuseGeluBwdIntoPw,
            fuseResidualBwdIntoPw: clipDispatchOptions.fuseResidualBwdIntoPw,
          })
        : null;
    if (clipLayout === "grid9_close2" && !batchTrainer) {
      throw new Error("splat3d: CLIP_LAYOUT=grid9_close2 needs CLIP_BATCH=3");
    }
    if (clipLayout === "grid9_close2" && cameras.length < 9) {
      throw new Error(`splat3d: CLIP_LAYOUT=grid9_close2 needs at least 9 cameras, got ${cameras.length}`);
    }
    raster.setParams(cfg.initParams ?? randomSplats3D(G, cfg.seed ?? 1, cfg.init));
    raster.zeroAdamState();
    const batchRasterForward =
      batchTrainer && ((cfg.viewLaneBatchRasterForward ?? false) || (cfg.viewLaneBatchRasterBackward ?? false))
        ? await raster.createBatchForwardState({
            lanes: batchTrainer.batch,
            imageBuffer: batchTrainer.inputBuffer,
            imageOffsets: Array.from({ length: batchTrainer.batch }, (_unused, lane) =>
              batchTrainer.slotOffsetBytes(lane, batchTrainer.plan.inputSlot)
            ),
            gradBuffer: batchTrainer.inputGradBuffer,
            gradOffsets: Array.from({ length: batchTrainer.batch }, (_unused, lane) =>
              batchTrainer.inputGradOffsetBytes(lane)
            ),
          })
        : null;
    const gridClip =
      clipLayout === "grid9_close2" && batchTrainer
        ? await Grid9Close2ClipLayout.create(device, raster, batchTrainer)
        : null;
    return new Splat3DOptimizer(device, raster, trainer, batchTrainer, gridClip, batchRasterForward, cameras, cfg);
  }

  private constructor(
    device: GPUDevice,
    raster: Raster3DEngine,
    trainer: VisionTrainer,
    batchTrainer: BatchMajorVisionTrainer | null,
    gridClip: Grid9Close2ClipLayout | null,
    batchRasterForward: Raster3DBatchForwardState | null,
    cameras: PreparedCamera3D[],
    cfg: Splat3DOptimizerConfig
  ) {
    this.device = device;
    this.raster = raster;
    this.trainer = trainer;
    this.batchTrainer = batchTrainer;
    this.gridClip = gridClip;
    this.batchRasterForward = batchRasterForward;
    this.cameras = cameras;
    this.clipBatchSize = batchTrainer?.batch ?? 1;
    this.clipLayout = cfg.clipLayout ?? "per_view";
    this.lrs = cfg.lrs ?? DEFAULT_3D_LRS;
    this.hyper = cfg.hyper ?? DEFAULT_HYPER;
    this.singlePassBatchRasterForward = cfg.singlePassBatchRasterForward ?? false;
    this.viewLaneBatchRasterForward = cfg.viewLaneBatchRasterForward ?? false;
    this.viewLaneBatchRasterBackward = cfg.viewLaneBatchRasterBackward ?? false;
    this.rngState = ((cfg.seed ?? 1) ^ 0x9e3779b9) >>> 0 || 1;
    this.textBuffers = cameras.map((_, i) =>
      device.createBuffer({
        label: `splat3d-text-${i}`,
        size: trainer.plan.textDim * 4,
        usage: U.COPY_SRC | U.COPY_DST,
      })
    );
    this.gridTextBuffer = gridClip
      ? device.createBuffer({
          label: "splat3d-grid9-text",
          size: trainer.plan.textDim * 4,
          usage: U.COPY_SRC | U.COPY_DST,
        })
      : null;
    this.singleIO = raster.createIOState(trainer.inputBuffer, 0, trainer.inputGradBuffer, 0);
    this.batchIO =
      batchRasterForward?.ios ??
      (batchTrainer
        ? Array.from({ length: batchTrainer.batch }, (_unused, lane) =>
            raster.createIOState(
              batchTrainer.inputBuffer,
              batchTrainer.slotOffsetBytes(lane, batchTrainer.plan.inputSlot),
              batchTrainer.inputGradBuffer,
              batchTrainer.inputGradOffsetBytes(lane),
              { privateState: true }
            )
          )
        : []);
  }

  setViewPrompts(embeds: Float32Array[]): void {
    if (embeds.length !== this.cameras.length) {
      throw new Error(`splat3d: ${embeds.length} text embeds for ${this.cameras.length} cameras`);
    }
    for (let i = 0; i < embeds.length; i++) {
      if (embeds[i].length !== this.trainer.plan.textDim) {
        throw new Error(`splat3d: view ${i} text ${embeds[i].length} != ${this.trainer.plan.textDim}`);
      }
      this.device.queue.writeBuffer(this.textBuffers[i], 0, embeds[i] as unknown as BufferSource);
    }
    if (this.gridTextBuffer) {
      this.device.queue.writeBuffer(this.gridTextBuffer, 0, embeds[0] as unknown as BufferSource);
    }
    this.hasPrompts = true;
  }

  setGridPrompt(embed: Float32Array): void {
    if (!this.gridTextBuffer) return;
    if (embed.length !== this.trainer.plan.textDim) {
      throw new Error(`splat3d: grid text ${embed.length} != ${this.trainer.plan.textDim}`);
    }
    this.device.queue.writeBuffer(this.gridTextBuffer, 0, embed as unknown as BufferSource);
  }

  step(displayView = 0, viewsPerStep = this.cameras.length): void {
    if (!this.hasPrompts) throw new Error("splat3d: setViewPrompts() before step()");
    const views = this.sampleViews(viewsPerStep);
    const enc = this.device.createCommandEncoder();
    this.raster.recordClearRawGrad(enc);
    this.recordTrainingViews(enc, views);
    this.step_ += 1;
    this.raster.recordAdam(enc, this.step_, this.lrs, this.hyper);
    this.raster.recordForward(enc, displayView);
    this.device.queue.submit([enc.finish()]);
  }

  async profileStep(
    displayView = 0,
    viewsPerStep = this.cameras.length,
    opts: Splat3DProfileOptions = {}
  ): Promise<Splat3DStepTimings> {
    if (!this.hasPrompts) throw new Error("splat3d: setViewPrompts() before profileStep()");
    await this.device.queue.onSubmittedWorkDone();
    const views = this.sampleViews(viewsPerStep);
    const timer = opts.gpuTimestamps ? GpuPassTimer.create(this.device) : null;
    const timings: Splat3DStepTimings = {
      views: views.length,
      totalViews: this.cameras.length,
      clipMode: this.useBatchFor(views) ? "batch" : "single",
      clipBatchSize: this.clipBatchSize,
      timing: timer ? "gpu-timestamp" : "split-submit-wall",
      total: 0,
      clear: 0,
      rasterFwd: 0,
      rasterReplay: 0,
      clipFwd: 0,
      clipBwd: 0,
      clipBatch: 0,
      rasterBwd: 0,
      adam: 0,
      display: 0,
    };
    const totalStart = performance.now();

    try {
      timings.clear += await this.submitTimed((enc, ts) => {
        this.raster.recordClearRawGrad(enc, ts);
      }, timer);

      if (this.useGridLayoutFor(views)) {
        const batch = this.batchTrainer!;
        const gridViews = views.slice(0, 9);
        const closeups = this.grid9CloseupViews(gridViews);
        timings.rasterFwd += await this.profileGrid9Close2Inputs(gridViews, closeups, timer);
        timings.clipBatch += await this.submitTimed((enc, ts) => {
          batch.encode(enc, { backward: true, timestampWrites: ts });
        }, timer);
        const bwd = await this.profileGrid9Close2Backward(gridViews, closeups, timer);
        timings.rasterReplay += bwd.replay;
        timings.rasterBwd += bwd.backward;
      } else if (this.useBatchFor(views)) {
        const batch = this.batchTrainer!;
        for (let start = 0; start < views.length; start += batch.batch) {
          const chunk = views.slice(start, start + batch.batch);
          if (chunk.length < batch.batch) {
            for (const view of chunk) {
              timings.rasterFwd += await this.submitTimed((enc, ts) => this.recordSingleForwardToTrainer(enc, view, ts), timer);
              timings.clipFwd += await this.submitTimed((enc, ts) => this.trainer.encodeForward(enc, ts), timer);
              timings.clipBwd += await this.submitTimed((enc, ts) => this.recordSingleTextAndBackward(enc, view, ts), timer);
              timings.rasterBwd += await this.submitTimed((enc, ts) => this.recordSingleRasterBackward(enc, view, ts), timer);
            }
            continue;
          }
          timings.rasterFwd += await this.profileBatchInputs(chunk, timer);
          timings.clipBatch += await this.submitTimed((enc, ts) => {
            batch.encode(enc, { backward: true, timestampWrites: ts });
          }, timer);
          if (this.viewLaneBatchRasterBackward && this.batchRasterForward && chunk.length > 1) {
            timings.rasterBwd += await this.submitTimed((enc, ts) => {
              this.raster.recordBatchBackwardAdd(enc, this.batchRasterForward!, chunk, ts);
            }, timer);
            continue;
          }
          for (let lane = 0; lane < chunk.length; lane++) {
            const view = chunk[lane];
            const io = this.batchIO[lane];
            timings.rasterBwd += await this.submitTimed((enc, ts) => {
              this.raster.recordBackwardAdd(enc, view, io, ts);
            }, timer);
          }
        }
      } else {
        for (const v of views) {
          timings.rasterFwd += await this.submitTimed((enc, ts) => this.recordSingleForwardToTrainer(enc, v, ts), timer);
          timings.clipFwd += await this.submitTimed((enc, ts) => {
            this.trainer.encodeForward(enc, ts);
          }, timer);
          timings.clipBwd += await this.submitTimed((enc, ts) => this.recordSingleTextAndBackward(enc, v, ts), timer);
          timings.rasterBwd += await this.submitTimed((enc, ts) => this.recordSingleRasterBackward(enc, v, ts), timer);
        }
      }

      this.step_ += 1;
      timings.adam += await this.submitTimed((enc, ts) => {
        this.raster.recordAdam(enc, this.step_, this.lrs, this.hyper, ts);
      }, timer);
      timings.display += await this.submitTimed((enc, ts) => {
        this.raster.recordForward(enc, displayView, undefined, ts);
      }, timer);
      timings.total = timer ? timedTotal(timings) : performance.now() - totalStart;
      return timings;
    } finally {
      timer?.destroy();
    }
  }

  get stepCount(): number {
    return this.step_;
  }

  async renderView(view = 0): Promise<Float32Array> {
    this.raster.runForward(view);
    return this.raster.readImage();
  }

  renderViewToImage(view = 0): void {
    this.raster.runForward(view);
  }

  async currentEmbedding(view = 0): Promise<Float32Array> {
    const enc = this.device.createCommandEncoder();
    this.raster.recordForward(enc, view, this.singleIO);
    this.trainer.encode(enc, { backward: false });
    this.device.queue.submit([enc.finish()]);
    return readFloats(this.device, this.trainer.outputBuffer, this.trainer.plan.embedDim);
  }

  destroy(): void {
    this.raster.destroy();
    this.trainer.destroy();
    this.batchTrainer?.destroy();
    this.gridClip?.destroy();
    this.gridTextBuffer?.destroy();
    for (const b of this.textBuffers) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }

  private useBatchFor(views: number[]): boolean {
    return !!this.batchTrainer && views.length >= this.batchTrainer.batch;
  }

  private useGridLayoutFor(views: number[]): boolean {
    if (this.clipLayout !== "grid9_close2") return false;
    if (!this.batchTrainer || !this.gridClip || !this.gridTextBuffer) {
      throw new Error("splat3d: grid9_close2 layout was not initialized");
    }
    if (views.length < 9) {
      throw new Error(`splat3d: grid9_close2 needs VIEWS=9, got ${views.length}`);
    }
    if (this.batchTrainer.batch < 3) {
      throw new Error(`splat3d: grid9_close2 needs CLIP_BATCH=3, got ${this.batchTrainer.batch}`);
    }
    return true;
  }

  private grid9CloseupViews(gridViews: number[]): [number, number] {
    const n = gridViews.length;
    const a = this.step_ % n;
    return [gridViews[a], gridViews[(a + 4) % n]];
  }

  private recordTrainingViews(enc: GPUCommandEncoder, views: number[]): void {
    if (this.useGridLayoutFor(views)) {
      this.recordGrid9Close2Training(enc, views.slice(0, 9));
      return;
    }
    if (!this.useBatchFor(views)) {
      for (const v of views) this.recordSingleTrainingView(enc, v);
      return;
    }
    const batch = this.batchTrainer!;
    for (let start = 0; start < views.length; start += batch.batch) {
      const chunk = views.slice(start, start + batch.batch);
      if (chunk.length < batch.batch) {
        for (const view of chunk) this.recordSingleTrainingView(enc, view);
        continue;
      }
      this.recordBatchInputs(enc, chunk);
      batch.encode(enc, { backward: true });
      if (this.viewLaneBatchRasterBackward && this.batchRasterForward && chunk.length > 1) {
        this.raster.recordBatchBackwardAdd(enc, this.batchRasterForward, chunk);
        continue;
      }
      for (let lane = 0; lane < chunk.length; lane++) {
        const view = chunk[lane];
        const io = this.batchIO[lane];
        this.raster.recordBackwardAdd(enc, view, io);
      }
    }
  }

  private recordGrid9Close2Training(enc: GPUCommandEncoder, gridViews: number[]): void {
    const batch = this.batchTrainer!;
    const closeups = this.grid9CloseupViews(gridViews);
    this.recordGrid9Close2Inputs(enc, gridViews, closeups);
    batch.encode(enc, { backward: true });
    this.recordGrid9Close2Backward(enc, gridViews, closeups);
  }

  private recordGrid9Close2Inputs(enc: GPUCommandEncoder, gridViews: number[], closeups: [number, number]): void {
    const batch = this.batchTrainer!;
    const grid = this.gridClip!;
    this.recordGrid9Close2TextCopies(enc, closeups);
    grid.clearGridImage(enc);
    for (let cell = 0; cell < 9; cell++) {
      this.raster.recordForward(enc, gridViews[cell], grid.scratchIO);
      grid.recordCopyCell(enc, cell);
    }
    for (let lane = 0; lane < 2; lane++) {
      this.raster.recordForward(enc, closeups[lane], this.batchIO[lane + 1]);
    }
    // The batch variable is intentionally touched here so future edits keep the
    // lane contract visible: lane 0 grid, lanes 1-2 close-ups.
    if (batch.batch < 3) throw new Error("splat3d: grid9_close2 lost its CLIP batch");
  }

  private recordGrid9Close2Backward(enc: GPUCommandEncoder, gridViews: number[], closeups: [number, number]): void {
    const grid = this.gridClip!;
    for (let cell = 0; cell < 9; cell++) {
      grid.clearScratchGrad(enc);
      grid.recordScatterCell(enc, cell);
      this.raster.recordForward(enc, gridViews[cell], grid.scratchIO);
      this.raster.recordBackwardAdd(enc, gridViews[cell], grid.scratchIO);
    }
    for (let lane = 0; lane < 2; lane++) {
      this.raster.recordBackwardAdd(enc, closeups[lane], this.batchIO[lane + 1]);
    }
  }

  private recordGrid9Close2TextCopies(enc: GPUCommandEncoder, closeups: [number, number]): void {
    const batch = this.batchTrainer!;
    const bytes = batch.plan.textDim * 4;
    enc.copyBufferToBuffer(this.gridTextBuffer!, 0, batch.textBuffer, batch.textOffsetBytes(0), bytes);
    for (let lane = 0; lane < 2; lane++) {
      const view = closeups[lane];
      enc.copyBufferToBuffer(this.textBuffers[view], 0, batch.textBuffer, batch.textOffsetBytes(lane + 1), bytes);
    }
  }

  private recordSingleTrainingView(enc: GPUCommandEncoder, view: number): void {
    enc.copyBufferToBuffer(this.textBuffers[view], 0, this.trainer.textBuffer, 0, this.trainer.plan.textDim * 4);
    this.raster.recordForward(enc, view, this.singleIO);
    this.trainer.encode(enc, { backward: true });
    this.raster.recordBackwardAdd(enc, view, this.singleIO);
  }

  private recordBatchInputs(enc: GPUCommandEncoder, views: number[]): void {
    this.recordBatchTextCopies(enc, views);
    if (this.singlePassBatchRasterForward && views.length > 1) {
      this.raster.recordForwards(enc, views, this.batchIO.slice(0, views.length));
      return;
    }
    if (this.viewLaneBatchRasterForward && this.batchRasterForward && views.length > 1) {
      this.raster.recordBatchForward(enc, this.batchRasterForward, views);
      return;
    }
    for (let lane = 0; lane < views.length; lane++) {
      this.raster.recordForward(enc, views[lane], this.batchIO[lane]);
    }
  }

  private async profileBatchInputs(views: number[], timer: GpuPassTimer | null): Promise<number> {
    if (!timer) {
      return this.submitTimed((enc) => this.recordBatchInputs(enc, views));
    }
    const copyEnc = this.device.createCommandEncoder();
    this.recordBatchTextCopies(copyEnc, views);
    this.device.queue.submit([copyEnc.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    if (this.singlePassBatchRasterForward && views.length > 1) {
      return this.submitTimed((enc, ts) => {
        this.raster.recordForwards(enc, views, this.batchIO.slice(0, views.length), ts);
      }, timer);
    }
    if (this.viewLaneBatchRasterForward && this.batchRasterForward && views.length > 1) {
      return this.submitTimed((enc, ts) => {
        this.raster.recordBatchForward(enc, this.batchRasterForward!, views, ts);
      }, timer);
    }
    let ms = 0;
    for (let lane = 0; lane < views.length; lane++) {
      ms += await this.submitTimed((enc, ts) => {
        this.raster.recordForward(enc, views[lane], this.batchIO[lane], ts);
      }, timer);
    }
    return ms;
  }

  private async profileGrid9Close2Inputs(
    gridViews: number[],
    closeups: [number, number],
    timer: GpuPassTimer | null
  ): Promise<number> {
    if (!timer) {
      return this.submitTimed((enc) => this.recordGrid9Close2Inputs(enc, gridViews, closeups));
    }
    const grid = this.gridClip!;
    const setup = this.device.createCommandEncoder();
    this.recordGrid9Close2TextCopies(setup, closeups);
    grid.clearGridImage(setup);
    this.device.queue.submit([setup.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    let ms = 0;
    for (let cell = 0; cell < 9; cell++) {
      ms += await this.submitTimed((enc, ts) => {
        this.raster.recordForward(enc, gridViews[cell], grid.scratchIO, ts);
      }, timer);
      ms += await this.submitTimed((enc, ts) => {
        grid.recordCopyCell(enc, cell, ts);
      }, timer);
    }
    for (let lane = 0; lane < 2; lane++) {
      ms += await this.submitTimed((enc, ts) => {
        this.raster.recordForward(enc, closeups[lane], this.batchIO[lane + 1], ts);
      }, timer);
    }
    return ms;
  }

  private async profileGrid9Close2Backward(
    gridViews: number[],
    closeups: [number, number],
    timer: GpuPassTimer | null
  ): Promise<{ replay: number; backward: number }> {
    if (!timer) {
      return {
        replay: 0,
        backward: await this.submitTimed((enc) => this.recordGrid9Close2Backward(enc, gridViews, closeups)),
      };
    }
    const grid = this.gridClip!;
    let replay = 0;
    let backward = 0;
    for (let cell = 0; cell < 9; cell++) {
      backward += await this.submitTimed((enc, ts) => {
        grid.clearScratchGrad(enc);
        grid.recordScatterCell(enc, cell, ts);
      }, timer);
      replay += await this.submitTimed((enc, ts) => {
        this.raster.recordForward(enc, gridViews[cell], grid.scratchIO, ts);
      }, timer);
      backward += await this.submitTimed((enc, ts) => {
        this.raster.recordBackwardAdd(enc, gridViews[cell], grid.scratchIO, ts);
      }, timer);
    }
    for (let lane = 0; lane < 2; lane++) {
      backward += await this.submitTimed((enc, ts) => {
        this.raster.recordBackwardAdd(enc, closeups[lane], this.batchIO[lane + 1], ts);
      }, timer);
    }
    return { replay, backward };
  }

  private recordBatchTextCopies(enc: GPUCommandEncoder, views: number[]): void {
    const batch = this.batchTrainer!;
    for (let lane = 0; lane < views.length; lane++) {
      const view = views[lane];
      enc.copyBufferToBuffer(this.textBuffers[view], 0, batch.textBuffer, batch.textOffsetBytes(lane), batch.plan.textDim * 4);
    }
  }

  private recordSingleForwardToTrainer(enc: GPUCommandEncoder, view: number, timestampWrites?: PassTimestampWrites): void {
    this.raster.recordForward(enc, view, this.singleIO, timestampWrites);
  }

  private recordSingleTextAndBackward(enc: GPUCommandEncoder, view: number, timestampWrites?: PassTimestampWrites): void {
    enc.copyBufferToBuffer(this.textBuffers[view], 0, this.trainer.textBuffer, 0, this.trainer.plan.textDim * 4);
    this.trainer.encodeBackward(enc, timestampWrites);
  }

  private recordSingleRasterBackward(enc: GPUCommandEncoder, view: number, timestampWrites?: PassTimestampWrites): void {
    this.raster.recordBackwardAdd(enc, view, this.singleIO, timestampWrites);
  }

  private async submitTimed(
    record: (enc: GPUCommandEncoder, timestampWrites?: PassTimestampWrites) => void,
    timer: GpuPassTimer | null = null
  ): Promise<number> {
    if (timer) return timer.time(record);
    const enc = this.device.createCommandEncoder();
    record(enc);
    const t0 = performance.now();
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return performance.now() - t0;
  }

  private sampleViews(viewsPerStep: number): number[] {
    const n = this.cameras.length;
    const k = Math.max(1, Math.min(n, viewsPerStep | 0));
    if (k >= n) return Array.from({ length: n }, (_unused, i) => i);
    const views: number[] = [];
    while (views.length < k) {
      if (this.viewCursor >= this.viewOrder.length) this.shuffleViewOrder();
      views.push(this.viewOrder[this.viewCursor]);
      this.viewCursor += 1;
    }
    return views;
  }

  private shuffleViewOrder(): void {
    this.viewOrder = Array.from({ length: this.cameras.length }, (_unused, i) => i);
    for (let i = this.viewOrder.length - 1; i > 0; i--) {
      const j = this.nextRandomU32() % (i + 1);
      const tmp = this.viewOrder[i];
      this.viewOrder[i] = this.viewOrder[j];
      this.viewOrder[j] = tmp;
    }
    this.viewCursor = 0;
  }

  private nextRandomU32(): number {
    this.rngState = (Math.imul(this.rngState, 1664525) + 1013904223) >>> 0;
    return this.rngState;
  }
}

function normalizeClipBatchSize(value: number | undefined): number {
  const n = Number.isFinite(value) ? value! | 0 : 1;
  return n > 1 ? Math.min(9, n) : 1;
}

function timedTotal(t: Splat3DStepTimings): number {
  return (
    t.clear +
    t.rasterFwd +
    t.rasterReplay +
    t.clipFwd +
    t.clipBwd +
    t.clipBatch +
    t.rasterBwd +
    t.adam +
    t.display
  );
}

type PassTimestampWrites = {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
};

class GpuPassTimer {
  static create(device: GPUDevice): GpuPassTimer | null {
    return device.features.has("timestamp-query") ? new GpuPassTimer(device) : null;
  }

  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly readBuffer: GPUBuffer;

  private constructor(private readonly device: GPUDevice) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    this.resolveBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  async time(record: (enc: GPUCommandEncoder, timestampWrites: PassTimestampWrites) => void): Promise<number> {
    const enc = this.device.createCommandEncoder();
    record(enc, {
      querySet: this.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    });
    enc.resolveQuerySet(this.querySet, 0, 2, this.resolveBuffer, 0);
    enc.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, 16);
    this.device.queue.submit([enc.finish()]);
    await this.readBuffer.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(this.readBuffer.getMappedRange().slice(0));
    this.readBuffer.unmap();
    return Number(ts[1] - ts[0]) / 1e6;
  }

  destroy(): void {
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    this.readBuffer.destroy();
  }
}

export function randomSplats3D(G: number, seed = 1, init: Splat3DInit = {}): Float32Array {
  const radius = init.radius ?? LEGIBLE_3D_INIT.radius;
  const radiusJitter = init.radiusJitter ?? LEGIBLE_3D_INIT.radiusJitter;
  const opacityRaw = init.opacityRaw ?? LEGIBLE_3D_INIT.opacityRaw;
  const colorSpread = init.colorSpread ?? LEGIBLE_3D_INIT.colorSpread;
  const positionSpread = init.positionSpread ?? LEGIBLE_3D_INIT.positionSpread;
  let state = (seed >>> 0) || 1;
  const next = (): number => {
    state = (Math.imul(state, 747796405) + 2891336453) >>> 0;
    let t = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
    t = ((t >>> 22) ^ t) >>> 0;
    return t / 4294967296;
  };
  const normal = (): number => {
    let u = 0;
    let v = 0;
    while (u === 0) u = next();
    while (v === 0) v = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const p = new Float32Array(G * PARAM_STRIDE_3D);
  const posOff = 0;
  const radOff = 3 * G;
  const colOff = 4 * G;
  const opOff = 7 * G;
  const lnRadius = Math.log(radius);
  for (let g = 0; g < G; g++) {
    p[posOff + g * 3 + 0] = (next() * 2 - 1) * positionSpread;
    p[posOff + g * 3 + 1] = (next() * 2 - 1) * positionSpread;
    p[posOff + g * 3 + 2] = (next() * 2 - 1) * positionSpread;
    p[radOff + g] = lnRadius + radiusJitter * normal();
    p[colOff + g * 3 + 0] = colorSpread * normal();
    p[colOff + g * 3 + 1] = colorSpread * normal();
    p[colOff + g * 3 + 2] = colorSpread * normal();
    p[opOff + g] = opacityRaw;
  }
  return p;
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb || 1);
}

async function readFloats(device: GPUDevice, buf: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: 1 | 8 });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}
