/// <reference types="@webgpu/types" />
import { VisionTrainer, type TrainPlan, type WeightArray } from "../clip/vision";
import type { PointwiseTileVariant, WeightPrecision } from "../clip/vision_wgsl";
import { BatchMajorVisionTrainer } from "../clip/vision_batch";
import { type AdamHyper, DEFAULT_HYPER } from "../splat/adam_wgsl";
import {
  DEFAULT_3D_CAMERAS,
  DUAL_GRID_CAMERA_COUNT,
  DUAL_GRID_RANDOM_START,
  DUAL_GRID_ZOOM_START,
  FIXED_GRID_CAMERA_COUNT,
  type Camera3D,
  type PreparedCamera3D,
  prepareCamera,
  sampleWeightedCameraIndices,
} from "./cameras";
import { Grid9Close2ClipLayout } from "./grid_clip";
import type { BackgroundTextureMode } from "./background_textures";
import {
  planFixedBudgetSplatAdaptation,
  type SplatAdaptationDiagnostics,
} from "./adaptive";
import {
  Raster3DEngine,
  type AdamLRs3D,
  DEFAULT_3D_LRS,
  type Raster3DBatchForwardState,
  type Raster3DCoverageOptions,
  type Raster3DIOState,
  type Raster3DRegularizerOptions,
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
  viewSampler?: Splat3DViewSampler;
  clipWeightPrecision?: WeightPrecision;
  pointwiseTileVariant?: PointwiseTileVariant;
  pointwiseTileSteps?: ReadonlySet<number>;
  stemSpatialBwd?: boolean;
  spatialBwdVariant?: "generic" | "depthwise4";
  fusePointwiseGeluForward?: boolean;
  fuseGeluBwdIntoPw?: boolean;
  fuseResidualBwdIntoPw?: boolean;
  singlePassBatchRasterForward?: boolean;
  viewLaneBatchRasterForward?: boolean;
  viewLaneBatchRasterBackward?: boolean;
  gridDirectRaster?: boolean;
  gridRasterSide?: number;
  retainGridCellState?: boolean;
  gridGradientScale?: number;
  randomGridGradientScale?: number;
  sharedWForwardSteps?: ReadonlySet<number>;
  clipRefreshInterval?: number;
  cachedLrScale?: number;
  convergence?: Splat3DConvergenceConfig;
}

export type Splat3DClipMode = "single" | "batch";
export type Splat3DClipLayout = "per_view" | "grid9_close2" | "dual_grid4";
export type Splat3DViewSampler = "epoch" | "random";
export type Splat3DStepTimingMode = "split-submit-wall" | "gpu-timestamp";
export type Splat3DBackgroundMode =
  | "black"
  | "dark_random"
  | "curriculum"
  | "blurred_noise"
  | "checkerboard"
  | "fourier";

export interface Splat3DConvergenceConfig {
  backgroundMode?: Splat3DBackgroundMode;
  centerWeight?: number;
  radiusWeight?: number;
  targetRadius?: number;
  opacitySparsity?: number;
  coverageWeight?: number;
  coverageTarget?: number;
  transmittanceStart?: number;
  transmittanceEnd?: number;
  transmittanceAnnealSteps?: number;
  rayDistortionWeight?: number;
  rayEntropyWeight?: number;
  rayEntropyMask?: number;
  smallRadiusWeight?: number;
  smallRadius?: number;
  radiusBandWeight?: number;
  minRadius?: number;
  maxRadius?: number;
  stagedOptimization?: boolean;
  geometryWarmupSteps?: number;
  geometryDecaySteps?: number;
  geometryFinalScale?: number;
  appearanceWarmupScale?: number;
  adaptiveRelocation?: boolean;
  adaptationInterval?: number;
  adaptationFraction?: number;
  mipSmoothing?: boolean;
  mipVarianceStart?: number;
  mipVarianceEnd?: number;
  mipAnnealSteps?: number;
}

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
  regularizer: number;
  adam: number;
  display: number;
}

interface DualGrid4ViewPlan {
  fixedGrid: number[];
  randomGrid: number[];
  singles: [number, number];
  zooms: [number, number];
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
  readonly randomGridClip: Grid9Close2ClipLayout | null;
  readonly cameras: PreparedCamera3D[];
  readonly side = SIDE;
  readonly clipBatchSize: number;
  readonly clipLayout: Splat3DClipLayout;
  readonly viewSampler: Splat3DViewSampler;
  readonly batchRasterForward: Raster3DBatchForwardState | null;
  private readonly textBuffers: GPUBuffer[];
  private readonly gridTextBuffer: GPUBuffer | null;
  private readonly randomGridTextBuffer: GPUBuffer | null;
  private readonly zoomTextBuffer: GPUBuffer | null;
  private readonly singleIO: Raster3DIOState;
  private readonly batchIO: Raster3DIOState[];
  private readonly lrs: AdamLRs3D;
  private readonly hyper: AdamHyper;
  private readonly singlePassBatchRasterForward: boolean;
  private readonly viewLaneBatchRasterForward: boolean;
  private readonly viewLaneBatchRasterBackward: boolean;
  private readonly gridDirectRaster: boolean;
  private readonly clipRefreshInterval: number;
  private readonly cachedLrScale: number;
  private readonly convergence: Required<Splat3DConvergenceConfig>;
  private step_ = 0;
  private hasPrompts = false;
  private rngState = 1;
  private viewOrder: number[] = [];
  private viewCursor = 0;
  private cachedBatchViews: number[] | null = null;
  private lastAdaptationStep = -1;
  private adaptationDiagnostics_: SplatAdaptationDiagnostics | null = null;

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
    const convergence = normalizeConvergenceConfig(cfg.convergence);
    const backgroundTextureMode = textureModeForBackground(convergence.backgroundMode);
    const raster = await Raster3DEngine.create(device, {
      H: SIDE,
      W: SIDE,
      G,
      cap: cfg.cap ?? defaultRasterCap(G),
      bg: cfg.bg ?? [0, 0, 0],
      dynamicBg: convergence.backgroundMode !== "black",
      dynamicBgTexture: backgroundTextureMode !== undefined,
      backgroundTextureMode,
      backgroundSeed: cfg.seed ?? 1,
      dynamicCoverage:
        convergence.coverageWeight !== 0 ||
        convergence.rayDistortionWeight !== 0 ||
        convergence.rayEntropyWeight !== 0,
      dynamicTransmittance: convergence.coverageWeight !== 0,
      dynamicEntropy: convergence.rayEntropyWeight !== 0,
      dynamicFootprint: convergence.mipSmoothing,
      cameras,
    });
    const clipBatchSize = normalizeClipBatchSize(cfg.clipBatchSize);
    const clipLayout = cfg.clipLayout ?? "per_view";
    const gridRasterSide = normalizeGridRasterSide(cfg.gridRasterSide, cfg.gridDirectRaster);
    const promoteGrid80BackwardStack =
      (clipLayout === "grid9_close2" || clipLayout === "dual_grid4") &&
      (clipLayout === "grid9_close2" ? clipBatchSize === 3 : clipBatchSize === 6) &&
      gridRasterSide === 80 &&
      (cfg.stemSpatialBwd ?? true) === true &&
      (cfg.fusePointwiseGeluForward ?? true) === true &&
      (cfg.clipWeightPrecision ?? "f32") === "f32";
    const effectiveSpatialBwdVariant = cfg.spatialBwdVariant ?? (promoteGrid80BackwardStack ? "depthwise4" : undefined);
    const promoteBackwardLocalFusion = promoteGrid80BackwardStack && effectiveSpatialBwdVariant === "depthwise4";
    const clipDispatchOptions = {
      weightPrecision: cfg.clipWeightPrecision,
      pointwiseTileVariant: cfg.pointwiseTileVariant,
      pointwiseTileSteps: cfg.pointwiseTileSteps,
      stemSpatialBwd: cfg.stemSpatialBwd ?? true,
      spatialBwdVariant: effectiveSpatialBwdVariant,
      fusePointwiseGeluForward: cfg.fusePointwiseGeluForward ?? true,
      fuseGeluBwdIntoPw: cfg.fuseGeluBwdIntoPw ?? promoteBackwardLocalFusion,
      fuseResidualBwdIntoPw: cfg.fuseResidualBwdIntoPw ?? promoteBackwardLocalFusion,
    };
    const trainer = await VisionTrainer.create(device, trainPlan, weights, clipDispatchOptions);
    const batchTrainer =
      clipBatchSize > 1
        ? await BatchMajorVisionTrainer.create(device, trainPlan, weights, clipBatchSize, {
            weightPrecision: cfg.clipWeightPrecision,
            stemSpatialBwd: clipDispatchOptions.stemSpatialBwd,
            spatialBwdVariant: clipDispatchOptions.spatialBwdVariant,
            sharedWForwardSteps: cfg.sharedWForwardSteps,
            fusePointwiseGeluForward: clipDispatchOptions.fusePointwiseGeluForward,
            fuseGeluBwdIntoPw: clipDispatchOptions.fuseGeluBwdIntoPw,
            fuseResidualBwdIntoPw: clipDispatchOptions.fuseResidualBwdIntoPw,
          })
        : null;
    if ((clipLayout === "grid9_close2" || clipLayout === "dual_grid4") && !batchTrainer) {
      throw new Error(`splat3d: CLIP_LAYOUT=${clipLayout} needs batched CLIP`);
    }
    if (clipLayout === "grid9_close2" && cameras.length < 9) {
      throw new Error(`splat3d: CLIP_LAYOUT=grid9_close2 needs at least 9 cameras, got ${cameras.length}`);
    }
    if (clipLayout === "dual_grid4") {
      if (clipBatchSize < 6) {
        throw new Error(`splat3d: CLIP_LAYOUT=dual_grid4 needs CLIP_BATCH=6, got ${clipBatchSize}`);
      }
      if (cameras.length < DUAL_GRID_CAMERA_COUNT) {
        throw new Error(`splat3d: CLIP_LAYOUT=dual_grid4 needs ${DUAL_GRID_CAMERA_COUNT} cameras, got ${cameras.length}`);
      }
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
      (clipLayout === "grid9_close2" || clipLayout === "dual_grid4") && batchTrainer
        ? await Grid9Close2ClipLayout.create(device, raster, batchTrainer, {
            directRaster: cfg.gridDirectRaster ?? false,
            rasterSide: gridRasterSide,
            gridLane: 0,
            retainCellState: cfg.retainGridCellState,
            gradientScale: normalizeGridGradientScale(cfg.gridGradientScale),
            backgroundTextureMode,
            backgroundSeed: cfg.seed ?? 1,
          })
        : null;
    const randomGridClip =
      clipLayout === "dual_grid4" && batchTrainer
        ? await Grid9Close2ClipLayout.create(device, raster, batchTrainer, {
            directRaster: cfg.gridDirectRaster ?? false,
            rasterSide: gridRasterSide,
            gridLane: 1,
            scratchRaster: gridRasterSide !== SIDE ? gridClip?.raster : undefined,
            retainCellState: cfg.retainGridCellState,
            gradientScale: normalizeGridGradientScale(cfg.randomGridGradientScale ?? cfg.gridGradientScale),
            backgroundTextureMode,
            backgroundSeed: (cfg.seed ?? 1) ^ 0x51f15e,
          })
        : null;
    return new Splat3DOptimizer(device, raster, trainer, batchTrainer, gridClip, randomGridClip, batchRasterForward, cameras, cfg);
  }

  private constructor(
    device: GPUDevice,
    raster: Raster3DEngine,
    trainer: VisionTrainer,
    batchTrainer: BatchMajorVisionTrainer | null,
    gridClip: Grid9Close2ClipLayout | null,
    randomGridClip: Grid9Close2ClipLayout | null,
    batchRasterForward: Raster3DBatchForwardState | null,
    cameras: PreparedCamera3D[],
    cfg: Splat3DOptimizerConfig
  ) {
    this.device = device;
    this.raster = raster;
    this.trainer = trainer;
    this.batchTrainer = batchTrainer;
    this.gridClip = gridClip;
    this.randomGridClip = randomGridClip;
    this.batchRasterForward = batchRasterForward;
    this.cameras = cameras;
    this.clipBatchSize = batchTrainer?.batch ?? 1;
    this.clipLayout = cfg.clipLayout ?? "per_view";
    this.viewSampler = cfg.viewSampler ?? "epoch";
    this.lrs = cfg.lrs ?? DEFAULT_3D_LRS;
    this.hyper = cfg.hyper ?? DEFAULT_HYPER;
    this.singlePassBatchRasterForward = cfg.singlePassBatchRasterForward ?? false;
    this.viewLaneBatchRasterForward = cfg.viewLaneBatchRasterForward ?? false;
    this.viewLaneBatchRasterBackward = cfg.viewLaneBatchRasterBackward ?? false;
    this.gridDirectRaster = cfg.gridDirectRaster ?? false;
    this.clipRefreshInterval = Math.max(1, cfg.clipRefreshInterval ?? 1);
    this.cachedLrScale = normalizeCachedLrScale(cfg.cachedLrScale);
    this.convergence = normalizeConvergenceConfig(cfg.convergence);
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
    this.randomGridTextBuffer = randomGridClip
      ? device.createBuffer({
          label: "splat3d-random-grid9-text",
          size: trainer.plan.textDim * 4,
          usage: U.COPY_SRC | U.COPY_DST,
        })
      : null;
    this.zoomTextBuffer =
      this.clipLayout === "dual_grid4"
        ? device.createBuffer({
            label: "splat3d-zoom-text",
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

  setRandomGridPrompt(embed: Float32Array): void {
    if (!this.randomGridTextBuffer) return;
    if (embed.length !== this.trainer.plan.textDim) {
      throw new Error(`splat3d: random grid text ${embed.length} != ${this.trainer.plan.textDim}`);
    }
    this.device.queue.writeBuffer(this.randomGridTextBuffer, 0, embed as unknown as BufferSource);
  }

  setZoomPrompt(embed: Float32Array): void {
    if (!this.zoomTextBuffer) return;
    if (embed.length !== this.trainer.plan.textDim) {
      throw new Error(`splat3d: zoom text ${embed.length} != ${this.trainer.plan.textDim}`);
    }
    this.device.queue.writeBuffer(this.zoomTextBuffer, 0, embed as unknown as BufferSource);
  }

  step(displayView = 0, viewsPerStep = this.cameras.length): void {
    if (!this.hasPrompts) throw new Error("splat3d: setViewPrompts() before step()");
    this.applyTrainingBackground();
    this.applyCoverageRegularizer();
    this.applyFootprintCurriculum();
    const useCached = this.shouldUseCachedBatchStep(viewsPerStep);
    const views = useCached ? this.cachedBatchViews!.slice() : this.sampleViews(viewsPerStep);
    const enc = this.device.createCommandEncoder();
    this.recordBackgroundTextures(enc, this.trainingBackgroundStrength());
    this.raster.recordClearRawGrad(enc);
    if (useCached) {
      this.recordCachedBatchTrainingViews(enc, views);
    } else {
      this.recordTrainingViews(enc, views);
      this.updateCachedBatchViews(views);
    }
    this.recordConvergenceRegularizer(enc);
    this.step_ += 1;
    this.raster.recordAdam(enc, this.step_, this.lrsForStep(useCached), this.hyper);
    this.raster.recordBackgroundGenerate(enc, this.step_, 0, 1);
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
    this.applyTrainingBackground();
    this.applyCoverageRegularizer();
    this.applyFootprintCurriculum();
    const useCached = this.shouldUseCachedBatchStep(viewsPerStep);
    const views = useCached ? this.cachedBatchViews!.slice() : this.sampleViews(viewsPerStep);
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
      regularizer: 0,
      adam: 0,
      display: 0,
    };
    const totalStart = performance.now();

    try {
      if (this.hasTexturedBackground()) {
        const backgroundEncoder = this.device.createCommandEncoder();
        this.recordBackgroundTextures(backgroundEncoder, this.trainingBackgroundStrength());
        this.device.queue.submit([backgroundEncoder.finish()]);
      }
      timings.clear += await this.submitTimed((enc, ts) => {
        this.raster.recordClearRawGrad(enc, ts);
      }, timer);

      if (useCached) {
        timings.rasterFwd += await this.profileCachedBatchInputs(views, timer);
        timings.rasterBwd += await this.profileCachedBatchBackward(views, timer);
      } else if (this.useDualGrid4Layout()) {
        const batch = this.batchTrainer!;
        const plan = this.dualGrid4Views();
        timings.views = plan.fixedGrid.length + plan.randomGrid.length + plan.singles.length + plan.zooms.length;
        timings.rasterFwd += await this.profileDualGrid4Inputs(plan, timer);
        timings.clipBatch += await this.submitTimed((enc, ts) => {
          batch.encode(enc, { backward: true, timestampWrites: ts });
        }, timer);
        const bwd = await this.profileDualGrid4Backward(plan, timer);
        timings.rasterReplay += bwd.replay;
        timings.rasterBwd += bwd.backward;
      } else if (this.useGridLayoutFor(views)) {
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

      if (!useCached) this.updateCachedBatchViews(views);
      if (this.convergenceRegularizerEnabled()) {
        timings.regularizer += await this.submitTimed((enc, ts) => {
          this.recordConvergenceRegularizer(enc, ts);
        }, timer);
      }
      this.step_ += 1;
      timings.adam += await this.submitTimed((enc, ts) => {
        this.raster.recordAdam(enc, this.step_, this.lrsForStep(useCached), this.hyper, ts);
      }, timer);
      this.applyDisplayBackground();
      timings.display += await this.submitTimed((enc, ts) => {
        this.raster.recordBackgroundGenerate(enc, this.step_, 0, 1);
        this.raster.recordForward(enc, displayView, undefined, ts);
      }, timer);
      await this.adaptSplatsIfDue();
      timings.total = timer ? timedTotal(timings) : performance.now() - totalStart;
      return timings;
    } finally {
      timer?.destroy();
    }
  }

  get stepCount(): number {
    return this.step_;
  }

  get adaptationDiagnostics(): SplatAdaptationDiagnostics | null {
    return this.adaptationDiagnostics_;
  }

  async adaptSplatsIfDue(force = false): Promise<SplatAdaptationDiagnostics | null> {
    if (!this.convergence.adaptiveRelocation) return null;
    const interval = Math.max(1, Math.round(this.convergence.adaptationInterval));
    if (!force && this.step_ < interval) return null;
    if (!force && this.lastAdaptationStep >= 0 && this.step_ - this.lastAdaptationStep < interval) return null;
    if (!force && this.lastAdaptationStep === this.step_) return this.adaptationDiagnostics_;
    await this.device.queue.onSubmittedWorkDone();
    const [params, gradients] = await Promise.all([this.raster.readParams(), this.raster.readRawGrad()]);
    const plan = planFixedBudgetSplatAdaptation(params, gradients, {
      maxRelocations: Math.max(1, Math.floor(this.raster.dims.G * this.convergence.adaptationFraction)),
      seed: (this.rngState ^ this.step_) >>> 0,
      deadOpacityThreshold: 0.04,
      minParentOpacity: 0.12,
      splitOffsetScale: 0.55,
    });
    if (plan.changedIndices.length > 0) {
      this.raster.setParams(plan.params);
      this.raster.resetAdamForSplats(plan.changedIndices);
    }
    this.lastAdaptationStep = this.step_;
    this.adaptationDiagnostics_ = plan.diagnostics;
    return plan.diagnostics;
  }

  async renderView(view = 0): Promise<Float32Array> {
    this.applyDisplayBackground();
    this.renderBlackView(view);
    return this.raster.readImage();
  }

  renderViewToImage(view = 0): void {
    this.applyDisplayBackground();
    this.renderBlackView(view);
  }

  async currentEmbedding(view = 0): Promise<Float32Array> {
    this.applyDisplayBackground();
    const enc = this.device.createCommandEncoder();
    this.raster.recordBackgroundGenerate(enc, this.step_, 0, 1);
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
    this.randomGridClip?.destroy();
    this.gridTextBuffer?.destroy();
    this.randomGridTextBuffer?.destroy();
    this.zoomTextBuffer?.destroy();
    for (const b of this.textBuffers) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }

  prepareDisplayFrame(): void {
    this.applyDisplayBackground();
  }

  private useBatchFor(views: number[]): boolean {
    return !!this.batchTrainer && views.length >= this.batchTrainer.batch;
  }

  private renderBlackView(view: number): void {
    const enc = this.device.createCommandEncoder();
    this.raster.recordBackgroundGenerate(enc, this.step_, 0, 1);
    this.raster.recordForward(enc, view);
    this.device.queue.submit([enc.finish()]);
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

  private useDualGrid4Layout(): boolean {
    if (this.clipLayout !== "dual_grid4") return false;
    if (
      !this.batchTrainer ||
      !this.gridClip ||
      !this.randomGridClip ||
      !this.gridTextBuffer ||
      !this.randomGridTextBuffer ||
      !this.zoomTextBuffer
    ) {
      throw new Error("splat3d: dual_grid4 layout was not initialized");
    }
    if (this.batchTrainer.batch < 6) {
      throw new Error(`splat3d: dual_grid4 needs CLIP_BATCH=6, got ${this.batchTrainer.batch}`);
    }
    if (this.cameras.length < DUAL_GRID_CAMERA_COUNT) {
      throw new Error(`splat3d: dual_grid4 needs ${DUAL_GRID_CAMERA_COUNT} cameras, got ${this.cameras.length}`);
    }
    return true;
  }

  private grid9CloseupViews(gridViews: number[]): [number, number] {
    const n = gridViews.length;
    if (this.viewSampler === "random") {
      const a = Math.floor(hash01(this.step_, 101) * n) % n;
      let b = Math.floor(hash01(this.step_, 211) * n) % n;
      if (b === a) b = (b + 4) % n;
      return [gridViews[a], gridViews[b]];
    }
    const a = this.step_ % n;
    return [gridViews[a], gridViews[(a + 4) % n]];
  }

  private recordTrainingViews(enc: GPUCommandEncoder, views: number[]): void {
    if (this.useDualGrid4Layout()) {
      this.recordDualGrid4Training(enc, this.dualGrid4Views());
      return;
    }
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

  private recordCachedBatchTrainingViews(enc: GPUCommandEncoder, views: number[]): void {
    this.recordCachedBatchInputs(enc, views);
    this.recordCachedBatchBackward(enc, views);
  }

  private recordCachedBatchInputs(enc: GPUCommandEncoder, views: number[]): void {
    if (!this.batchTrainer) throw new Error("splat3d: cached CLIP step needs batch trainer");
    if (views.length !== this.batchTrainer.batch) {
      throw new Error(`splat3d: cached CLIP step needs one full batch, got ${views.length}`);
    }
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

  private recordCachedBatchBackward(enc: GPUCommandEncoder, views: number[]): void {
    if (this.viewLaneBatchRasterBackward && this.batchRasterForward && views.length > 1) {
      this.raster.recordBatchBackwardAdd(enc, this.batchRasterForward, views);
      return;
    }
    for (let lane = 0; lane < views.length; lane++) {
      this.raster.recordBackwardAdd(enc, views[lane], this.batchIO[lane]);
    }
  }

  private recordDualGrid4Training(enc: GPUCommandEncoder, plan: DualGrid4ViewPlan): void {
    const batch = this.batchTrainer!;
    this.recordDualGrid4Inputs(enc, plan);
    batch.encode(enc, { backward: true });
    this.recordDualGrid4Backward(enc, plan);
  }

  private recordDualGrid4Inputs(enc: GPUCommandEncoder, plan: DualGrid4ViewPlan): void {
    const batch = this.batchTrainer!;
    const fixedGrid = this.gridClip!;
    const randomGrid = this.randomGridClip!;
    this.recordDualGrid4TextCopies(enc, plan);
    fixedGrid.clearGridImage(enc);
    randomGrid.clearGridImage(enc);
    for (let cell = 0; cell < FIXED_GRID_CAMERA_COUNT; cell++) {
      fixedGrid.raster.recordForward(enc, plan.fixedGrid[cell], fixedGrid.scratchIOForCell(cell));
      fixedGrid.recordCopyCell(enc, cell);
      randomGrid.raster.recordForward(enc, plan.randomGrid[cell], randomGrid.scratchIOForCell(cell));
      randomGrid.recordCopyCell(enc, cell);
    }
    this.raster.recordForward(enc, plan.singles[0], this.batchIO[2]);
    this.raster.recordForward(enc, plan.singles[1], this.batchIO[3]);
    this.raster.recordForward(enc, plan.zooms[0], this.batchIO[4]);
    this.raster.recordForward(enc, plan.zooms[1], this.batchIO[5]);
    if (batch.batch < 6) throw new Error("splat3d: dual_grid4 lost its CLIP batch");
  }

  private recordDualGrid4Backward(enc: GPUCommandEncoder, plan: DualGrid4ViewPlan): void {
    const fixedGrid = this.gridClip!;
    const randomGrid = this.randomGridClip!;
    for (let cell = 0; cell < FIXED_GRID_CAMERA_COUNT; cell++) {
      fixedGrid.clearScratchGrad(enc);
      fixedGrid.recordScatterCell(enc, cell);
      const fixedIO = fixedGrid.scratchIOForCell(cell);
      if (!fixedGrid.retainsCellState) fixedGrid.raster.recordForward(enc, plan.fixedGrid[cell], fixedIO);
      fixedGrid.raster.recordBackwardAdd(enc, plan.fixedGrid[cell], fixedIO);

      randomGrid.clearScratchGrad(enc);
      randomGrid.recordScatterCell(enc, cell);
      const randomIO = randomGrid.scratchIOForCell(cell);
      if (!randomGrid.retainsCellState) randomGrid.raster.recordForward(enc, plan.randomGrid[cell], randomIO);
      randomGrid.raster.recordBackwardAdd(enc, plan.randomGrid[cell], randomIO);
    }
    this.raster.recordBackwardAdd(enc, plan.singles[0], this.batchIO[2]);
    this.raster.recordBackwardAdd(enc, plan.singles[1], this.batchIO[3]);
    this.raster.recordBackwardAdd(enc, plan.zooms[0], this.batchIO[4]);
    this.raster.recordBackwardAdd(enc, plan.zooms[1], this.batchIO[5]);
  }

  private recordDualGrid4TextCopies(enc: GPUCommandEncoder, plan: DualGrid4ViewPlan): void {
    const batch = this.batchTrainer!;
    const bytes = batch.plan.textDim * 4;
    enc.copyBufferToBuffer(this.gridTextBuffer!, 0, batch.textBuffer, batch.textOffsetBytes(0), bytes);
    enc.copyBufferToBuffer(this.randomGridTextBuffer!, 0, batch.textBuffer, batch.textOffsetBytes(1), bytes);
    enc.copyBufferToBuffer(this.textBuffers[plan.singles[0]], 0, batch.textBuffer, batch.textOffsetBytes(2), bytes);
    enc.copyBufferToBuffer(this.textBuffers[plan.singles[1]], 0, batch.textBuffer, batch.textOffsetBytes(3), bytes);
    enc.copyBufferToBuffer(this.zoomTextBuffer!, 0, batch.textBuffer, batch.textOffsetBytes(4), bytes);
    enc.copyBufferToBuffer(this.zoomTextBuffer!, 0, batch.textBuffer, batch.textOffsetBytes(5), bytes);
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
      grid.raster.recordForward(enc, gridViews[cell], grid.scratchIOForCell(cell));
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
      const io = grid.scratchIOForCell(cell);
      if (!grid.retainsCellState) grid.raster.recordForward(enc, gridViews[cell], io);
      grid.raster.recordBackwardAdd(enc, gridViews[cell], io);
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

  private async profileCachedBatchInputs(views: number[], timer: GpuPassTimer | null): Promise<number> {
    if (!timer) {
      return this.submitTimed((enc) => this.recordCachedBatchInputs(enc, views));
    }
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

  private async profileCachedBatchBackward(views: number[], timer: GpuPassTimer | null): Promise<number> {
    if (!timer) {
      return this.submitTimed((enc) => this.recordCachedBatchBackward(enc, views));
    }
    if (this.viewLaneBatchRasterBackward && this.batchRasterForward && views.length > 1) {
      return this.submitTimed((enc, ts) => {
        this.raster.recordBatchBackwardAdd(enc, this.batchRasterForward!, views, ts);
      }, timer);
    }
    let ms = 0;
    for (let lane = 0; lane < views.length; lane++) {
      ms += await this.submitTimed((enc, ts) => {
        this.raster.recordBackwardAdd(enc, views[lane], this.batchIO[lane], ts);
      }, timer);
    }
    return ms;
  }

  private async profileDualGrid4Inputs(plan: DualGrid4ViewPlan, timer: GpuPassTimer | null): Promise<number> {
    if (!timer) {
      return this.submitTimed((enc) => this.recordDualGrid4Inputs(enc, plan));
    }
    const fixedGrid = this.gridClip!;
    const randomGrid = this.randomGridClip!;
    const setup = this.device.createCommandEncoder();
    this.recordDualGrid4TextCopies(setup, plan);
    fixedGrid.clearGridImage(setup);
    randomGrid.clearGridImage(setup);
    this.device.queue.submit([setup.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    let ms = 0;
    for (let cell = 0; cell < FIXED_GRID_CAMERA_COUNT; cell++) {
      ms += await this.submitTimed((enc, ts) => {
        fixedGrid.raster.recordForward(enc, plan.fixedGrid[cell], fixedGrid.scratchIOForCell(cell), ts);
      }, timer);
      ms += await this.submitTimed((enc, ts) => {
        fixedGrid.recordCopyCell(enc, cell, ts);
      }, timer);
      ms += await this.submitTimed((enc, ts) => {
        randomGrid.raster.recordForward(enc, plan.randomGrid[cell], randomGrid.scratchIOForCell(cell), ts);
      }, timer);
      ms += await this.submitTimed((enc, ts) => {
        randomGrid.recordCopyCell(enc, cell, ts);
      }, timer);
    }
    ms += await this.submitTimed((enc, ts) => {
      this.raster.recordForward(enc, plan.singles[0], this.batchIO[2], ts);
    }, timer);
    ms += await this.submitTimed((enc, ts) => {
      this.raster.recordForward(enc, plan.singles[1], this.batchIO[3], ts);
    }, timer);
    ms += await this.submitTimed((enc, ts) => {
      this.raster.recordForward(enc, plan.zooms[0], this.batchIO[4], ts);
    }, timer);
    ms += await this.submitTimed((enc, ts) => {
      this.raster.recordForward(enc, plan.zooms[1], this.batchIO[5], ts);
    }, timer);
    return ms;
  }

  private async profileDualGrid4Backward(
    plan: DualGrid4ViewPlan,
    timer: GpuPassTimer | null
  ): Promise<{ replay: number; backward: number }> {
    if (!timer) {
      return {
        replay: 0,
        backward: await this.submitTimed((enc) => this.recordDualGrid4Backward(enc, plan)),
      };
    }
    const fixedGrid = this.gridClip!;
    const randomGrid = this.randomGridClip!;
    let replay = 0;
    let backward = 0;
    for (let cell = 0; cell < FIXED_GRID_CAMERA_COUNT; cell++) {
      backward += await this.submitTimed((enc, ts) => {
        fixedGrid.clearScratchGrad(enc);
        fixedGrid.recordScatterCell(enc, cell, ts);
      }, timer);
      const fixedIO = fixedGrid.scratchIOForCell(cell);
      if (!fixedGrid.retainsCellState) {
        replay += await this.submitTimed((enc, ts) => {
          fixedGrid.raster.recordForward(enc, plan.fixedGrid[cell], fixedIO, ts);
        }, timer);
      }
      backward += await this.submitTimed((enc, ts) => {
        fixedGrid.raster.recordBackwardAdd(enc, plan.fixedGrid[cell], fixedIO, ts);
      }, timer);

      backward += await this.submitTimed((enc, ts) => {
        randomGrid.clearScratchGrad(enc);
        randomGrid.recordScatterCell(enc, cell, ts);
      }, timer);
      const randomIO = randomGrid.scratchIOForCell(cell);
      if (!randomGrid.retainsCellState) {
        replay += await this.submitTimed((enc, ts) => {
          randomGrid.raster.recordForward(enc, plan.randomGrid[cell], randomIO, ts);
        }, timer);
      }
      backward += await this.submitTimed((enc, ts) => {
        randomGrid.raster.recordBackwardAdd(enc, plan.randomGrid[cell], randomIO, ts);
      }, timer);
    }
    backward += await this.submitTimed((enc, ts) => {
      this.raster.recordBackwardAdd(enc, plan.singles[0], this.batchIO[2], ts);
    }, timer);
    backward += await this.submitTimed((enc, ts) => {
      this.raster.recordBackwardAdd(enc, plan.singles[1], this.batchIO[3], ts);
    }, timer);
    backward += await this.submitTimed((enc, ts) => {
      this.raster.recordBackwardAdd(enc, plan.zooms[0], this.batchIO[4], ts);
    }, timer);
    backward += await this.submitTimed((enc, ts) => {
      this.raster.recordBackwardAdd(enc, plan.zooms[1], this.batchIO[5], ts);
    }, timer);
    return { replay, backward };
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
        grid.raster.recordForward(enc, gridViews[cell], grid.scratchIOForCell(cell), ts);
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
      const io = grid.scratchIOForCell(cell);
      if (!grid.retainsCellState) {
        replay += await this.submitTimed((enc, ts) => {
          grid.raster.recordForward(enc, gridViews[cell], io, ts);
        }, timer);
      }
      backward += await this.submitTimed((enc, ts) => {
        grid.raster.recordBackwardAdd(enc, gridViews[cell], io, ts);
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

  private recordConvergenceRegularizer(enc: GPUCommandEncoder, timestampWrites?: PassTimestampWrites): void {
    if (!this.convergenceRegularizerEnabled()) return;
    this.raster.recordRegularizerAdd(enc, this.regularizerOptions(), timestampWrites);
  }

  private convergenceRegularizerEnabled(): boolean {
    return (
      this.convergence.centerWeight !== 0 ||
      this.convergence.radiusWeight !== 0 ||
      this.convergence.opacitySparsity !== 0 ||
      this.convergence.smallRadiusWeight !== 0 ||
      this.convergence.radiusBandWeight !== 0
    );
  }

  private regularizerOptions(): Raster3DRegularizerOptions {
    return {
      centerWeight: this.convergence.centerWeight,
      radiusWeight: this.convergence.radiusWeight,
      targetRadius: this.convergence.targetRadius,
      opacitySparsity: this.convergence.opacitySparsity,
      smallRadiusWeight: this.convergence.smallRadiusWeight,
      smallRadius: this.convergence.smallRadius,
      radiusBandWeight: this.convergence.radiusBandWeight,
      minRadius: this.convergence.minRadius,
      maxRadius: this.convergence.maxRadius,
    };
  }

  private coverageOptions(): Raster3DCoverageOptions {
    const anneal = Math.max(1, this.convergence.transmittanceAnnealSteps);
    const t = Math.max(0, Math.min(1, this.step_ / anneal));
    const targetTransmittance =
      this.convergence.transmittanceStart +
      (this.convergence.transmittanceEnd - this.convergence.transmittanceStart) * t;
    return {
      transmittanceWeight: this.convergence.coverageWeight,
      targetTransmittance,
      rayDistortionWeight: this.convergence.rayDistortionWeight,
      rayEntropyWeight: this.convergence.rayEntropyWeight,
      rayEntropyMask: this.convergence.rayEntropyMask,
    };
  }

  private applyTrainingBackground(): void {
    this.applyBackground(this.trainingBackground());
  }

  private applyDisplayBackground(): void {
    this.applyBackground([0, 0, 0]);
  }

  private applyCoverageRegularizer(): void {
    const opts = this.coverageOptions();
    this.raster.setCoverageRegularizer(opts);
    if (this.gridClip && this.gridClip.raster !== this.raster) {
      this.gridClip.raster.setCoverageRegularizer(opts);
    }
    if (this.randomGridClip && this.randomGridClip.raster !== this.raster) {
      this.randomGridClip.raster.setCoverageRegularizer(opts);
    }
  }

  private applyFootprintCurriculum(): void {
    if (!this.convergence.mipSmoothing) return;
    const anneal = Math.max(1, this.convergence.mipAnnealSteps);
    const t = Math.max(0, Math.min(1, this.step_ / anneal));
    const variance =
      this.convergence.mipVarianceStart +
      (this.convergence.mipVarianceEnd - this.convergence.mipVarianceStart) * t;
    const rasters = new Set<Raster3DEngine>([this.raster]);
    if (this.gridClip) rasters.add(this.gridClip.raster);
    if (this.randomGridClip) rasters.add(this.randomGridClip.raster);
    for (const raster of rasters) raster.setScreenVariance(variance);
  }

  private applyBackground(rgb: [number, number, number]): void {
    this.raster.setBackground(rgb);
    if (this.gridClip && this.gridClip.raster !== this.raster) {
      this.gridClip.raster.setBackground(rgb);
    }
    if (this.randomGridClip && this.randomGridClip.raster !== this.raster) {
      this.randomGridClip.raster.setBackground(rgb);
    }
  }

  private hasTexturedBackground(): boolean {
    return (
      this.raster.usesTexturedBackground ||
      !!this.gridClip?.raster.usesTexturedBackground ||
      !!this.randomGridClip?.raster.usesTexturedBackground
    );
  }

  private recordBackgroundTextures(enc: GPUCommandEncoder, strength: number): void {
    const rasters = new Set<Raster3DEngine>([this.raster]);
    if (this.gridClip) rasters.add(this.gridClip.raster);
    if (this.randomGridClip) rasters.add(this.randomGridClip.raster);
    for (const raster of rasters) raster.recordBackgroundGenerate(enc, this.step_, strength);
  }

  private trainingBackgroundStrength(): number {
    if (this.convergence.backgroundMode === "curriculum") {
      return this.step_ < 120 ? 0.35 : Math.min(1, 0.35 + (this.step_ - 120) / 380);
    }
    return 1;
  }

  private trainingBackground(): [number, number, number] {
    const mode = this.convergence.backgroundMode;
    if (mode === "black") return [0, 0, 0];
    const moderate = mode === "curriculum" && this.step_ >= 120 && this.step_ % 8 === 0;
    const max = moderate ? 0.28 : 0.09;
    const floor = moderate ? 0.02 : 0;
    return [
      floor + max * hash01(this.step_, 11),
      floor + max * hash01(this.step_, 29),
      floor + max * hash01(this.step_, 47),
    ];
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
    const k = this.normalizedViewCount(viewsPerStep);
    if (k >= n) return Array.from({ length: n }, (_unused, i) => i);
    if (this.viewSampler === "random") return this.sampleRandomViews(k);
    const views: number[] = [];
    while (views.length < k) {
      if (this.viewCursor >= this.viewOrder.length) this.shuffleViewOrder();
      views.push(this.viewOrder[this.viewCursor]);
      this.viewCursor += 1;
    }
    return views;
  }

  private shouldUseCachedBatchStep(viewsPerStep: number): boolean {
    if (this.clipRefreshInterval <= 1) return false;
    if (!this.cachedBatchViews) return false;
    if (this.step_ % this.clipRefreshInterval === 0) return false;
    if (this.clipLayout !== "per_view") return false;
    if (!this.batchTrainer) return false;
    return this.normalizedViewCount(viewsPerStep) === this.cachedBatchViews.length;
  }

  private lrsForStep(useCached: boolean): AdamLRs3D {
    let lrs = this.lrs;
    if (this.convergence.stagedOptimization) {
      const warmup = Math.max(1, this.convergence.geometryWarmupSteps);
      const decay = Math.max(1, this.convergence.geometryDecaySteps);
      const appearanceT = Math.max(0, Math.min(1, this.step_ / warmup));
      const geometryT = Math.max(0, Math.min(1, (this.step_ - warmup) / decay));
      const geometryScale =
        1 + (this.convergence.geometryFinalScale - 1) * geometryT;
      const appearanceScale =
        this.convergence.appearanceWarmupScale +
        (1 - this.convergence.appearanceWarmupScale) * appearanceT;
      lrs = {
        position: this.lrs.position * geometryScale,
        logRadius: this.lrs.logRadius * geometryScale,
        color: this.lrs.color * appearanceScale,
        opacity: this.lrs.opacity * appearanceScale,
      };
    }
    if (!useCached || this.cachedLrScale === 1) return lrs;
    return scaleLrs3D(lrs, this.cachedLrScale);
  }

  private updateCachedBatchViews(views: number[]): void {
    if (this.clipRefreshInterval <= 1 || this.clipLayout !== "per_view" || !this.batchTrainer) {
      this.cachedBatchViews = null;
      return;
    }
    this.cachedBatchViews = views.length === this.batchTrainer.batch ? views.slice() : null;
  }

  private normalizedViewCount(viewsPerStep: number): number {
    const n = this.cameras.length;
    return Math.max(1, Math.min(n, viewsPerStep | 0));
  }

  private dualGrid4Views(): DualGrid4ViewPlan {
    return {
      fixedGrid: Array.from({ length: FIXED_GRID_CAMERA_COUNT }, (_unused, i) => i),
      randomGrid: this.sampleCameraRange(DUAL_GRID_RANDOM_START, FIXED_GRID_CAMERA_COUNT, FIXED_GRID_CAMERA_COUNT, 401),
      singles: this.sampleCameraPair(DUAL_GRID_RANDOM_START, FIXED_GRID_CAMERA_COUNT, 503, 607),
      zooms: this.sampleCameraPair(DUAL_GRID_ZOOM_START, FIXED_GRID_CAMERA_COUNT, 709, 811),
    };
  }

  private sampleCameraRange(start: number, count: number, k: number, salt: number): number[] {
    const pool = Array.from({ length: count }, (_unused, i) => start + i);
    if (this.viewSampler !== "random") {
      const offset = this.step_ % Math.max(1, count);
      return pool.slice(offset).concat(pool.slice(0, offset)).slice(0, k);
    }
    for (let i = 0; i < Math.min(k, pool.length); i++) {
      const r = Math.floor(hash01(this.step_, salt + i * 37) * (pool.length - i));
      const j = i + Math.max(0, Math.min(pool.length - i - 1, r));
      const tmp = pool[i];
      pool[i] = pool[j];
      pool[j] = tmp;
    }
    return pool.slice(0, k);
  }

  private sampleCameraPair(start: number, count: number, saltA: number, saltB: number): [number, number] {
    if (this.viewSampler !== "random") {
      const a = start + (this.step_ % count);
      return [a, start + ((this.step_ + 4) % count)];
    }
    const a = start + (Math.floor(hash01(this.step_, saltA) * count) % count);
    let b = start + (Math.floor(hash01(this.step_, saltB) * count) % count);
    if (b === a) b = start + ((b - start + 4) % count);
    return [a, b];
  }

  private sampleRandomViews(k: number): number[] {
    return sampleWeightedCameraIndices(
      this.cameras,
      k,
      () => this.nextRandomU32() / 4294967296
    );
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

function defaultRasterCap(splats: number): number {
  let cap = 256;
  while (cap < splats && cap < 4096) cap *= 2;
  return cap;
}

function normalizeCachedLrScale(value: number | undefined): number {
  if (value === undefined) return 1;
  if (!Number.isFinite(value)) return 1;
  return Math.max(0, value);
}

function normalizeGridGradientScale(value: number | undefined): number {
  if (value === undefined) return 1;
  if (!Number.isFinite(value)) return 1;
  return Math.max(0, value);
}

function normalizeGridRasterSide(value: number | undefined, directRaster: boolean | undefined): number {
  if (value !== undefined && Number.isFinite(value)) {
    const n = value | 0;
    if (n === 80 || n === SIDE || n === 512) return n;
  }
  return directRaster ? 80 : SIDE;
}

function normalizeConvergenceConfig(cfg: Splat3DConvergenceConfig | undefined): Required<Splat3DConvergenceConfig> {
  const requestedBackground = cfg?.backgroundMode;
  const backgroundMode: Splat3DBackgroundMode =
    requestedBackground === "dark_random" ||
    requestedBackground === "curriculum" ||
    requestedBackground === "blurred_noise" ||
    requestedBackground === "checkerboard" ||
    requestedBackground === "fourier"
      ? requestedBackground
      : "black";
  return {
    backgroundMode,
    centerWeight: finiteNonNegative(cfg?.centerWeight, 0),
    radiusWeight: finiteNonNegative(cfg?.radiusWeight, 0),
    targetRadius: finitePositive(cfg?.targetRadius, 1.15),
    opacitySparsity: finiteNonNegative(cfg?.opacitySparsity, 0),
    coverageWeight: finiteNonNegative(cfg?.coverageWeight, 0),
    coverageTarget: clamp01(cfg?.coverageTarget, 0.18),
    transmittanceStart: clamp01(cfg?.transmittanceStart, 0.4),
    transmittanceEnd: clamp01(
      cfg?.transmittanceEnd,
      cfg?.coverageTarget === undefined ? 0.88 : 1 - clamp01(cfg.coverageTarget, 0.18)
    ),
    transmittanceAnnealSteps: finitePositive(cfg?.transmittanceAnnealSteps, 500),
    rayDistortionWeight: finiteNonNegative(cfg?.rayDistortionWeight, 0),
    rayEntropyWeight: finiteNonNegative(cfg?.rayEntropyWeight, 0),
    rayEntropyMask: finiteNonNegative(cfg?.rayEntropyMask, 0.05),
    smallRadiusWeight: finiteNonNegative(cfg?.smallRadiusWeight, 0),
    smallRadius: finitePositive(cfg?.smallRadius, 0.022),
    radiusBandWeight: finiteNonNegative(cfg?.radiusBandWeight, 0),
    minRadius: finitePositive(cfg?.minRadius, 0.014),
    maxRadius: finitePositive(cfg?.maxRadius, 0.18),
    stagedOptimization: cfg?.stagedOptimization === true,
    geometryWarmupSteps: finitePositive(cfg?.geometryWarmupSteps, 250),
    geometryDecaySteps: finitePositive(cfg?.geometryDecaySteps, 1000),
    geometryFinalScale: finiteNonNegative(cfg?.geometryFinalScale, 0.2),
    appearanceWarmupScale: clamp01(cfg?.appearanceWarmupScale, 0.35),
    adaptiveRelocation: cfg?.adaptiveRelocation === true,
    adaptationInterval: finitePositive(cfg?.adaptationInterval, 200),
    adaptationFraction: clamp01(cfg?.adaptationFraction, 0.01),
    mipSmoothing: cfg?.mipSmoothing === true,
    mipVarianceStart: finiteNonNegative(cfg?.mipVarianceStart, 4),
    mipVarianceEnd: finiteNonNegative(cfg?.mipVarianceEnd, 0.0625),
    mipAnnealSteps: finitePositive(cfg?.mipAnnealSteps, 500),
  };
}

function textureModeForBackground(mode: Splat3DBackgroundMode): BackgroundTextureMode | undefined {
  if (mode === "curriculum" || mode === "blurred_noise") return "blurred_noise";
  if (mode === "checkerboard" || mode === "fourier") return mode;
  return undefined;
}

function finiteNonNegative(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) ? Math.max(0, value) : fallback;
}

function finitePositive(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) ? Math.max(1e-4, value) : fallback;
}

function clamp01(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : fallback;
}

function scaleLrs3D(lrs: AdamLRs3D, scale: number): AdamLRs3D {
  return {
    position: lrs.position * scale,
    logRadius: lrs.logRadius * scale,
    color: lrs.color * scale,
    opacity: lrs.opacity * scale,
  };
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
    t.regularizer +
    t.adam +
    t.display
  );
}

function hash01(step: number, salt: number): number {
  let x = (Math.imul((step + 1) >>> 0, 747796405) + Math.imul(salt >>> 0, 2891336453)) >>> 0;
  x = Math.imul((x >>> ((x >>> 28) + 4)) ^ x, 277803737) >>> 0;
  x = ((x >>> 22) ^ x) >>> 0;
  return x / 4294967296;
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
