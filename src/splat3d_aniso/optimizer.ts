/// <reference types="@webgpu/types" />

import { VisionTrainer, type TrainPlan, type WeightArray } from "../clip/vision";
import { BatchMajorVisionTrainer } from "../clip/vision_batch";
import {
  DEFAULT_3D_CAMERAS,
  prepareCamera,
  sampleWeightedCameraIndices,
  type Camera3D,
  type PreparedCamera3D,
} from "../splat3d/cameras";
import {
  LEGIBLE_3D_G,
  randomSplats3D,
  type Splat3DConvergenceConfig,
  type Splat3DStepTimings,
  type Splat3DViewSampler,
} from "../splat3d/optimize";
import type { SplatAdaptationDiagnostics } from "../splat3d/adaptive";
import { planFixedBudgetAnisotropicSplatAdaptation } from "./adaptive";
import { ANISO_PARAM_STRIDE_3D } from "./layout";
import {
  AnisotropicRaster3DEngine,
  DEFAULT_ANISOTROPIC_3D_LRS,
  type AnisotropicAdamLRs3D,
  type AnisotropicDensityStats3D,
  type AnisotropicRaster3DRegularizerOptions,
} from "./raster_engine";

const U = { MAP_READ: 1, COPY_DST: 8 };
const SIDE = 256;
const IMAGE_BYTES = 3 * SIDE * SIDE * 4;

export interface Splat3DAnisotropicOptimizerConfig {
  G?: number;
  cap?: number;
  seed?: number;
  cameras?: Camera3D[];
  viewSampler?: Splat3DViewSampler;
  lrs?: Partial<AnisotropicAdamLRs3D>;
  clipBatchSize?: number;
  convergence?: Splat3DConvergenceConfig;
  /** Maximum zero-mean log-axis offset used to break spherical initialization symmetry. */
  initialAnisotropy?: number;
  randomInitialRotation?: boolean;
}

export interface AnisotropicInitOptions {
  anisotropy?: number;
  randomRotation?: boolean;
}

interface AnisotropicConvergenceConfig {
  centerWeight: number;
  radiusWeight: number;
  targetRadius: number;
  opacitySparsity: number;
  smallRadiusWeight: number;
  smallRadius: number;
  radiusBandWeight: number;
  minRadius: number;
  maxRadius: number;
  stagedOptimization: boolean;
  geometryWarmupSteps: number;
  geometryDecaySteps: number;
  geometryFinalScale: number;
  appearanceWarmupScale: number;
  adaptiveRelocation: boolean;
  adaptationInterval: number;
  adaptationFraction: number;
}

export class Splat3DAnisotropicOptimizer {
  readonly device: GPUDevice;
  readonly raster: AnisotropicRaster3DEngine;
  readonly trainer: VisionTrainer;
  readonly batchTrainer: BatchMajorVisionTrainer | null;
  readonly cameras: PreparedCamera3D[];
  readonly side = SIDE;
  readonly clipBatchSize: number;
  readonly clipLayout = "per_view" as const;
  readonly viewSampler: Splat3DViewSampler;
  private readonly textBuffers: GPUBuffer[];
  private readonly lrs: AnisotropicAdamLRs3D;
  private readonly convergence: AnisotropicConvergenceConfig;
  private step_ = 0;
  private hasPrompts = false;
  private viewCursor = 0;
  private rng: number;
  private lastAdaptationStep = -1;
  private hasDensityStats = false;
  private adaptationDiagnostics_: SplatAdaptationDiagnostics | null = null;

  static async create(
    device: GPUDevice,
    plan: TrainPlan,
    weights: WeightArray,
    cfg: Splat3DAnisotropicOptimizerConfig = {}
  ): Promise<Splat3DAnisotropicOptimizer> {
    const [channels, height, width] = plan.inputShape;
    if (channels !== 3 || height !== SIDE || width !== SIDE) {
      throw new Error(`splat3d aniso: CLIP input shape must be [3,${SIDE},${SIDE}]`);
    }
    const G = cfg.G ?? LEGIBLE_3D_G;
    const cameras = (cfg.cameras ?? DEFAULT_3D_CAMERAS).map((camera) => prepareCamera(camera, SIDE));
    const raster = await AnisotropicRaster3DEngine.create(device, {
      H: SIDE,
      W: SIDE,
      G,
      cap: cfg.cap ?? defaultCap(G),
      cameras,
      bg: [0, 0, 0],
    });
    raster.setParams(
      isotropicInitAsAnisotropic(G, cfg.seed ?? 1, {
        anisotropy: cfg.initialAnisotropy ?? 0.45,
        randomRotation: cfg.randomInitialRotation ?? true,
      })
    );
    raster.zeroAdamState();
    const trainer = await VisionTrainer.create(device, plan, weights, {
      stemSpatialBwd: true,
      fusePointwiseGeluForward: true,
    });
    const clipBatchSize = normalizeClipBatchSize(cfg.clipBatchSize);
    const batchTrainer =
      clipBatchSize > 1
        ? await BatchMajorVisionTrainer.create(device, plan, weights, clipBatchSize, {
            stemSpatialBwd: true,
            fusePointwiseGeluForward: true,
          })
        : null;
    return new Splat3DAnisotropicOptimizer(device, raster, trainer, batchTrainer, cameras, cfg);
  }

  private constructor(
    device: GPUDevice,
    raster: AnisotropicRaster3DEngine,
    trainer: VisionTrainer,
    batchTrainer: BatchMajorVisionTrainer | null,
    cameras: PreparedCamera3D[],
    cfg: Splat3DAnisotropicOptimizerConfig
  ) {
    this.device = device;
    this.raster = raster;
    this.trainer = trainer;
    this.batchTrainer = batchTrainer;
    this.clipBatchSize = batchTrainer?.batch ?? 1;
    this.cameras = cameras;
    this.viewSampler = cfg.viewSampler ?? "epoch";
    this.rng = ((cfg.seed ?? 1) ^ 0x9e3779b9) >>> 0 || 1;
    this.lrs = { ...DEFAULT_ANISOTROPIC_3D_LRS, ...cfg.lrs };
    this.convergence = normalizeAnisotropicConvergence(cfg.convergence);
    this.textBuffers = cameras.map((_camera, index) =>
      device.createBuffer({
        label: `splat3d-aniso-text-${index}`,
        size: trainer.plan.textDim * 4,
        usage: 4 | 8,
      })
    );
  }

  setViewPrompts(embeddings: Float32Array[]): void {
    if (embeddings.length !== this.cameras.length) {
      throw new Error(`splat3d aniso: expected ${this.cameras.length} view prompts, got ${embeddings.length}`);
    }
    for (let view = 0; view < embeddings.length; view++) {
      if (embeddings[view].length !== this.trainer.plan.textDim) {
        throw new Error(`splat3d aniso: prompt ${view} has wrong embedding size`);
      }
      this.device.queue.writeBuffer(this.textBuffers[view], 0, embeddings[view]);
    }
    this.hasPrompts = true;
  }

  setGridPrompt(_embedding: Float32Array): void {}
  setRandomGridPrompt(_embedding: Float32Array): void {}
  setZoomPrompt(_embedding: Float32Array): void {}

  step(displayView = 0, viewsPerStep = this.cameras.length): void {
    if (!this.hasPrompts) throw new Error("splat3d aniso: setViewPrompts before step");
    const views = this.sampleViews(viewsPerStep);
    const collectDensityStats = this.shouldCaptureDensityStats();
    const encoder = this.device.createCommandEncoder();
    this.raster.recordClearRawGrad(encoder);
    this.recordTrainingViews(encoder, views, collectDensityStats);
    this.recordConvergenceRegularizer(encoder);
    this.step_++;
    this.raster.recordAdam(encoder, this.step_, this.lrsForStep());
    this.raster.recordForward(encoder, displayView);
    this.device.queue.submit([encoder.finish()]);
    this.hasDensityStats ||= collectDensityStats;
  }

  async profileStep(displayView = 0, viewsPerStep = this.cameras.length): Promise<Splat3DStepTimings> {
    if (!this.hasPrompts) throw new Error("splat3d aniso: setViewPrompts before profileStep");
    await this.device.queue.onSubmittedWorkDone();
    const views = this.sampleViews(viewsPerStep);
    const batch = this.batchTrainer;
    if (!batch || views.length !== batch.batch) {
      const start = performance.now();
      this.step(displayView, viewsPerStep);
      await this.device.queue.onSubmittedWorkDone();
      const total = performance.now() - start;
      return this.emptyTimings(views.length, total);
    }

    const start = performance.now();
    const timings = this.emptyTimings(views.length, 0);
    const collectDensityStats = this.shouldCaptureDensityStats();
    timings.clear = await this.submitTimed((encoder) => this.raster.recordClearRawGrad(encoder));
    timings.rasterFwd = await this.submitTimed((encoder) => this.recordBatchInputs(encoder, views));
    timings.clipBatch = await this.submitTimed((encoder) => batch.encode(encoder, { backward: true }));
    for (let lane = 0; lane < views.length; lane++) {
      const view = views[lane];
      timings.rasterReplay += await this.submitTimed((encoder) => {
        encoder.copyBufferToBuffer(
          batch.inputGradBuffer,
          batch.inputGradOffsetBytes(lane),
          this.raster.gradImage,
          0,
          IMAGE_BYTES
        );
        this.raster.recordForward(encoder, view);
      });
      timings.rasterBwd += await this.submitTimed((encoder) =>
        this.raster.recordBackwardAdd(encoder, view, collectDensityStats)
      );
    }
    timings.regularizer = await this.submitTimed((encoder) => this.recordConvergenceRegularizer(encoder));
    this.step_++;
    this.hasDensityStats ||= collectDensityStats;
    timings.adam = await this.submitTimed((encoder) => this.raster.recordAdam(encoder, this.step_, this.lrsForStep()));
    timings.display = await this.submitTimed((encoder) => this.raster.recordForward(encoder, displayView));
    await this.adaptSplatsIfDue();
    const total = performance.now() - start;
    timings.total = total;
    return timings;
  }

  private emptyTimings(views: number, total: number): Splat3DStepTimings {
    return {
      views,
      totalViews: this.cameras.length,
      clipMode: this.batchTrainer ? "batch" : "single",
      clipBatchSize: this.clipBatchSize,
      timing: "split-submit-wall",
      total,
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
    if (!force && this.lastAdaptationStep >= 0 && this.step_ - this.lastAdaptationStep < interval) {
      return null;
    }
    if (!force && this.lastAdaptationStep === this.step_) return this.adaptationDiagnostics_;
    await this.device.queue.onSubmittedWorkDone();
    const densityStats = this.hasDensityStats ? await this.raster.readDensityStats() : null;
    const [params, gradients] = await Promise.all([
      this.raster.readParams(),
      this.raster.readRawGrad(),
    ]);
    const densityControl = densityStats === null ? null : densityControlInputs(densityStats);
    const plan = planFixedBudgetAnisotropicSplatAdaptation(params, gradients, {
      maxRelocations: Math.max(1, Math.floor(this.raster.dims.G * this.convergence.adaptationFraction)),
      seed: (this.rng ^ this.step_) >>> 0,
      deadOpacityThreshold: 0.04,
      minParentOpacity: 0.12,
      minRadius: this.convergence.minRadius,
      maxRadius: this.convergence.maxRadius,
      splitOffsetScale: 0.55,
      selectionNeed: densityControl?.selectionNeed,
      coverage: densityControl?.coverage,
    });
    if (densityStats !== null) {
      plan.diagnostics.densityStatsSampled = true;
      plan.diagnostics.densityVisiblePixels = densityStats.visiblePixels.reduce((sum, value) => sum + value, 0);
      plan.diagnostics.densityMaxScreenGradient = Math.max(...densityStats.absScreenGradient);
    }
    if (plan.changedIndices.length > 0) {
      this.raster.setParams(plan.params);
      this.raster.resetAdamForSplats(plan.changedIndices);
    }
    this.lastAdaptationStep = this.step_;
    if (densityStats !== null) {
      this.raster.clearDensityStats();
      this.hasDensityStats = false;
    }
    this.adaptationDiagnostics_ = plan.diagnostics;
    return plan.diagnostics;
  }

  prepareDisplayFrame(): void {}

  renderViewToImage(view = 0): void {
    this.raster.runForward(view);
  }

  async renderView(view = 0): Promise<Float32Array> {
    this.raster.runForward(view);
    return this.raster.readImage();
  }

  async currentEmbedding(view = 0): Promise<Float32Array> {
    const encoder = this.device.createCommandEncoder();
    this.raster.recordForward(encoder, view);
    encoder.copyBufferToBuffer(this.raster.image, 0, this.trainer.inputBuffer, 0, IMAGE_BYTES);
    this.trainer.encode(encoder, { backward: false });
    this.device.queue.submit([encoder.finish()]);
    return readFloats(this.device, this.trainer.outputBuffer, this.trainer.plan.embedDim);
  }

  destroy(): void {
    this.raster.destroy();
    this.trainer.destroy();
    this.batchTrainer?.destroy();
    for (const buffer of this.textBuffers) buffer.destroy();
  }

  private shouldCaptureDensityStats(): boolean {
    if (!this.convergence.adaptiveRelocation) return false;
    const interval = Math.max(1, Math.round(this.convergence.adaptationInterval));
    return this.step_ + 1 >= interval && (this.step_ + 1) % interval === 0;
  }

  private recordTrainingView(encoder: GPUCommandEncoder, view: number, collectDensityStats = false): void {
    const index = Math.max(0, Math.min(this.cameras.length - 1, view | 0));
    encoder.copyBufferToBuffer(
      this.textBuffers[index],
      0,
      this.trainer.textBuffer,
      0,
      this.trainer.plan.textDim * 4
    );
    this.raster.recordForward(encoder, index);
    encoder.copyBufferToBuffer(this.raster.image, 0, this.trainer.inputBuffer, 0, IMAGE_BYTES);
    this.trainer.encode(encoder, { backward: true });
    encoder.copyBufferToBuffer(this.trainer.inputGradBuffer, 0, this.raster.gradImage, 0, IMAGE_BYTES);
    this.raster.recordBackwardAdd(encoder, index, collectDensityStats);
  }

  private recordTrainingViews(encoder: GPUCommandEncoder, views: number[], collectDensityStats = false): void {
    const batch = this.batchTrainer;
    if (!batch || views.length < batch.batch) {
      for (const view of views) this.recordTrainingView(encoder, view, collectDensityStats);
      return;
    }
    for (let start = 0; start < views.length; start += batch.batch) {
      const chunk = views.slice(start, start + batch.batch);
      if (chunk.length < batch.batch) {
        for (const view of chunk) this.recordTrainingView(encoder, view, collectDensityStats);
        continue;
      }
      this.recordBatchTrainingViews(encoder, chunk, collectDensityStats);
    }
  }

  private recordBatchTrainingViews(encoder: GPUCommandEncoder, views: number[], collectDensityStats = false): void {
    const batch = this.batchTrainer!;
    this.recordBatchInputs(encoder, views);
    batch.encode(encoder, { backward: true });
    this.recordBatchBackward(encoder, views, collectDensityStats);
  }

  private recordBatchInputs(encoder: GPUCommandEncoder, views: number[]): void {
    const batch = this.batchTrainer!;
    const textBytes = batch.plan.textDim * 4;
    for (let lane = 0; lane < views.length; lane++) {
      const view = Math.max(0, Math.min(this.cameras.length - 1, views[lane] | 0));
      encoder.copyBufferToBuffer(this.textBuffers[view], 0, batch.textBuffer, batch.textOffsetBytes(lane), textBytes);
      this.raster.recordForward(encoder, view);
      encoder.copyBufferToBuffer(
        this.raster.image,
        0,
        batch.inputBuffer,
        batch.slotOffsetBytes(lane, batch.plan.inputSlot),
        IMAGE_BYTES
      );
    }
  }

  private recordBatchBackward(encoder: GPUCommandEncoder, views: number[], collectDensityStats = false): void {
    const batch = this.batchTrainer!;
    for (let lane = 0; lane < views.length; lane++) {
      const view = Math.max(0, Math.min(this.cameras.length - 1, views[lane] | 0));
      encoder.copyBufferToBuffer(batch.inputGradBuffer, batch.inputGradOffsetBytes(lane), this.raster.gradImage, 0, IMAGE_BYTES);
      // Shared tile/conic state belongs to the last forward, so replay this lane.
      this.raster.recordForward(encoder, view);
      this.raster.recordBackwardAdd(encoder, view, collectDensityStats);
    }
  }

  private async submitTimed(record: (encoder: GPUCommandEncoder) => void): Promise<number> {
    const encoder = this.device.createCommandEncoder();
    record(encoder);
    const start = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  }

  private lrsForStep(): AnisotropicAdamLRs3D {
    if (!this.convergence.stagedOptimization) return this.lrs;
    const warmup = Math.max(1, this.convergence.geometryWarmupSteps);
    const decay = Math.max(1, this.convergence.geometryDecaySteps);
    const appearanceT = Math.max(0, Math.min(1, this.step_ / warmup));
    const geometryT = Math.max(0, Math.min(1, (this.step_ - warmup) / decay));
    const geometryScale = 1 + (this.convergence.geometryFinalScale - 1) * geometryT;
    const appearanceScale =
      this.convergence.appearanceWarmupScale +
      (1 - this.convergence.appearanceWarmupScale) * appearanceT;
    return {
      position: this.lrs.position * geometryScale,
      logScale: this.lrs.logScale * geometryScale,
      quaternion: this.lrs.quaternion * geometryScale,
      color: this.lrs.color * appearanceScale,
      opacity: this.lrs.opacity * appearanceScale,
    };
  }

  private recordConvergenceRegularizer(encoder: GPUCommandEncoder): void {
    if (!this.convergenceRegularizerEnabled()) return;
    this.raster.recordRegularizerAdd(encoder, this.regularizerOptions());
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

  private regularizerOptions(): AnisotropicRaster3DRegularizerOptions {
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

  private sampleViews(requested: number): number[] {
    const count = Math.max(1, Math.min(this.cameras.length, requested | 0));
    if (count === this.cameras.length) return Array.from({ length: count }, (_unused, view) => view);
    if (this.viewSampler === "epoch") {
      const views = Array.from({ length: count }, (_unused, offset) => (this.viewCursor + offset) % this.cameras.length);
      this.viewCursor = (this.viewCursor + count) % this.cameras.length;
      return views;
    }
    return sampleWeightedCameraIndices(this.cameras, count, () => {
      this.rng ^= this.rng << 13;
      this.rng ^= this.rng >>> 17;
      this.rng ^= this.rng << 5;
      return (this.rng >>> 0) / 4294967296;
    });
  }
}

export function isotropicInitAsAnisotropic(
  G: number,
  seed: number,
  options: AnisotropicInitOptions = {}
): Float32Array {
  const isotropic = randomSplats3D(G, seed);
  const output = new Float32Array(G * ANISO_PARAM_STRIDE_3D);
  const anisotropy = Math.max(0, options.anisotropy ?? 0);
  const randomRotation = options.randomRotation ?? anisotropy > 0;
  const next = seededRandom(seed ^ 0xa511e9b3);
  output.set(isotropic.subarray(0, 3 * G), 0);
  for (let g = 0; g < G; g++) {
    const logRadius = isotropic[3 * G + g];
    const stretch = anisotropy * (0.65 + 0.7 * next());
    const offsets = [-stretch, 0, stretch];
    for (let axis = offsets.length - 1; axis > 0; axis--) {
      const other = Math.floor(next() * (axis + 1));
      [offsets[axis], offsets[other]] = [offsets[other], offsets[axis]];
    }
    output[3 * G + g * 3 + 0] = logRadius + offsets[0];
    output[3 * G + g * 3 + 1] = logRadius + offsets[1];
    output[3 * G + g * 3 + 2] = logRadius + offsets[2];

    if (randomRotation) {
      // Shoemake's uniform unit-quaternion construction, stored as [x, y, z, w].
      const u1 = next();
      const u2 = 2 * Math.PI * next();
      const u3 = 2 * Math.PI * next();
      const a = Math.sqrt(1 - u1);
      const b = Math.sqrt(u1);
      output[6 * G + g * 4 + 0] = a * Math.sin(u2);
      output[6 * G + g * 4 + 1] = a * Math.cos(u2);
      output[6 * G + g * 4 + 2] = b * Math.sin(u3);
      output[6 * G + g * 4 + 3] = b * Math.cos(u3);
    } else {
      output[6 * G + g * 4 + 3] = 1;
    }
  }
  output.set(isotropic.subarray(4 * G, 7 * G), 10 * G);
  output.set(isotropic.subarray(7 * G, 8 * G), 13 * G);
  return output;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function defaultCap(G: number): number {
  let cap = 256;
  while (cap < G && cap < 4096) cap *= 2;
  return cap;
}

function normalizeClipBatchSize(value: number | undefined): number {
  if (value === undefined) return 3;
  if (!Number.isFinite(value)) return 3;
  const batch = value | 0;
  return batch > 1 ? Math.min(9, batch) : 1;
}

function normalizeAnisotropicConvergence(
  cfg: Splat3DConvergenceConfig | undefined
): AnisotropicConvergenceConfig {
  return {
    centerWeight: finiteNonNegative(cfg?.centerWeight, 0),
    radiusWeight: finiteNonNegative(cfg?.radiusWeight, 0),
    targetRadius: finitePositive(cfg?.targetRadius, 1.15),
    opacitySparsity: finiteNonNegative(cfg?.opacitySparsity, 0),
    smallRadiusWeight: finiteNonNegative(cfg?.smallRadiusWeight, 0),
    smallRadius: finitePositive(cfg?.smallRadius, 0.024),
    radiusBandWeight: finiteNonNegative(cfg?.radiusBandWeight, 0),
    minRadius: finitePositive(cfg?.minRadius, 0.016),
    maxRadius: finitePositive(cfg?.maxRadius, 0.16),
    stagedOptimization: cfg?.stagedOptimization === true,
    geometryWarmupSteps: finitePositive(cfg?.geometryWarmupSteps, 250),
    geometryDecaySteps: finitePositive(cfg?.geometryDecaySteps, 1000),
    geometryFinalScale: finiteNonNegative(cfg?.geometryFinalScale, 0.2),
    appearanceWarmupScale: clamp01(cfg?.appearanceWarmupScale, 0.35),
    adaptiveRelocation: cfg?.adaptiveRelocation === true,
    adaptationInterval: finitePositive(cfg?.adaptationInterval, 200),
    adaptationFraction: clamp01(cfg?.adaptationFraction, 0.01),
  };
}

function densityControlInputs(stats: AnisotropicDensityStats3D): {
  selectionNeed: Float32Array;
  coverage: Float32Array;
} {
  const selectionNeed = stats.absScreenGradient.slice();
  const coverage = new Float32Array(selectionNeed.length);
  for (let g = 0; g < coverage.length; g++) {
    const pixels = stats.visiblePixels[g];
    // The AbsGS statistic already sums pixel-level magnitudes, so it already
    // carries Pixel-GS's coverage weighting. This confidence term only rejects
    // one-pixel noise and gently discounts weakly visible candidates.
    coverage[g] = pixels < 4 ? 0 : Math.min(1, Math.sqrt(pixels / 64));
  }
  return { selectionNeed, coverage };
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

async function readFloats(device: GPUDevice, buffer: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: U.MAP_READ | U.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, floats * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(1);
  const output = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return output;
}
