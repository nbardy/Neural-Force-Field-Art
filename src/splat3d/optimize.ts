/// <reference types="@webgpu/types" />
import { VisionTrainer, type TrainPlan } from "../clip/vision";
import { BatchMajorVisionTrainer } from "../clip/vision_batch";
import { type AdamHyper, DEFAULT_HYPER } from "../splat/adam_wgsl";
import { DEFAULT_3D_CAMERAS, type Camera3D, type PreparedCamera3D, prepareCamera } from "./cameras";
import { Raster3DEngine, type AdamLRs3D, DEFAULT_3D_LRS, type Raster3DIOState } from "./raster";
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
  stemSpatialBwd?: boolean;
}

export type Splat3DClipMode = "single" | "batch";

export interface Splat3DStepTimings {
  views: number;
  totalViews: number;
  clipMode: Splat3DClipMode;
  clipBatchSize: number;
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
  readonly cameras: PreparedCamera3D[];
  readonly side = SIDE;
  readonly clipBatchSize: number;
  private readonly textBuffers: GPUBuffer[];
  private readonly singleIO: Raster3DIOState;
  private readonly batchIO: Raster3DIOState[];
  private readonly lrs: AdamLRs3D;
  private readonly hyper: AdamHyper;
  private step_ = 0;
  private hasPrompts = false;
  private rngState = 1;
  private viewOrder: number[] = [];
  private viewCursor = 0;

  static async create(
    device: GPUDevice,
    trainPlan: TrainPlan,
    weights: Float32Array,
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
    const trainer = await VisionTrainer.create(device, trainPlan, weights);
    const clipBatchSize = normalizeClipBatchSize(cfg.clipBatchSize);
    const batchTrainer =
      clipBatchSize > 1
        ? await BatchMajorVisionTrainer.create(device, trainPlan, weights, clipBatchSize, {
            stemSpatialBwd: cfg.stemSpatialBwd ?? true,
          })
        : null;
    raster.setParams(cfg.initParams ?? randomSplats3D(G, cfg.seed ?? 1, cfg.init));
    raster.zeroAdamState();
    return new Splat3DOptimizer(device, raster, trainer, batchTrainer, cameras, cfg);
  }

  private constructor(
    device: GPUDevice,
    raster: Raster3DEngine,
    trainer: VisionTrainer,
    batchTrainer: BatchMajorVisionTrainer | null,
    cameras: PreparedCamera3D[],
    cfg: Splat3DOptimizerConfig
  ) {
    this.device = device;
    this.raster = raster;
    this.trainer = trainer;
    this.batchTrainer = batchTrainer;
    this.cameras = cameras;
    this.clipBatchSize = batchTrainer?.batch ?? 1;
    this.lrs = cfg.lrs ?? DEFAULT_3D_LRS;
    this.hyper = cfg.hyper ?? DEFAULT_HYPER;
    this.rngState = ((cfg.seed ?? 1) ^ 0x9e3779b9) >>> 0 || 1;
    this.textBuffers = cameras.map((_, i) =>
      device.createBuffer({
        label: `splat3d-text-${i}`,
        size: trainer.plan.textDim * 4,
        usage: U.COPY_SRC | U.COPY_DST,
      })
    );
    this.singleIO = raster.createIOState(trainer.inputBuffer, 0, trainer.inputGradBuffer, 0);
    this.batchIO = batchTrainer
      ? Array.from({ length: batchTrainer.batch }, (_unused, lane) =>
          raster.createIOState(
            batchTrainer.inputBuffer,
            batchTrainer.slotOffsetBytes(lane, batchTrainer.plan.inputSlot),
            batchTrainer.inputGradBuffer,
            batchTrainer.inputGradOffsetBytes(lane),
            { privateState: true }
          )
        )
      : [];
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
    this.hasPrompts = true;
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

  async profileStep(displayView = 0, viewsPerStep = this.cameras.length): Promise<Splat3DStepTimings> {
    if (!this.hasPrompts) throw new Error("splat3d: setViewPrompts() before profileStep()");
    await this.device.queue.onSubmittedWorkDone();
    const views = this.sampleViews(viewsPerStep);
    const timings: Splat3DStepTimings = {
      views: views.length,
      totalViews: this.cameras.length,
      clipMode: this.useBatchFor(views) ? "batch" : "single",
      clipBatchSize: this.clipBatchSize,
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

    timings.clear += await this.submitTimed((enc) => {
      this.raster.recordClearRawGrad(enc);
    });

    if (this.useBatchFor(views)) {
      const batch = this.batchTrainer!;
      for (let start = 0; start < views.length; start += batch.batch) {
        const chunk = views.slice(start, start + batch.batch);
        if (chunk.length < batch.batch) {
          for (const view of chunk) {
            timings.rasterFwd += await this.submitTimed((enc) => this.recordSingleForwardToTrainer(enc, view));
            timings.clipFwd += await this.submitTimed((enc) => this.trainer.encodeForward(enc));
            timings.clipBwd += await this.submitTimed((enc) => this.recordSingleTextAndBackward(enc, view));
            timings.rasterBwd += await this.submitTimed((enc) => this.recordSingleRasterBackward(enc, view));
          }
          continue;
        }
        timings.rasterFwd += await this.submitTimed((enc) => this.recordBatchInputs(enc, chunk));
        timings.clipBatch += await this.submitTimed((enc) => batch.encode(enc, { backward: true }));
        for (let lane = 0; lane < chunk.length; lane++) {
          const view = chunk[lane];
          const io = this.batchIO[lane];
          timings.rasterBwd += await this.submitTimed((enc) => {
            this.raster.recordBackwardAdd(enc, view, io);
          });
        }
      }
    } else {
      for (const v of views) {
        timings.rasterFwd += await this.submitTimed((enc) => this.recordSingleForwardToTrainer(enc, v));
        timings.clipFwd += await this.submitTimed((enc) => {
          this.trainer.encodeForward(enc);
        });
        timings.clipBwd += await this.submitTimed((enc) => this.recordSingleTextAndBackward(enc, v));
        timings.rasterBwd += await this.submitTimed((enc) => this.recordSingleRasterBackward(enc, v));
      }
    }

    this.step_ += 1;
    timings.adam += await this.submitTimed((enc) => {
      this.raster.recordAdam(enc, this.step_, this.lrs, this.hyper);
    });
    timings.display += await this.submitTimed((enc) => {
      this.raster.recordForward(enc, displayView);
    });
    timings.total = performance.now() - totalStart;
    return timings;
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
    for (const b of this.textBuffers) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }

  private useBatchFor(views: number[]): boolean {
    return !!this.batchTrainer && views.length >= this.batchTrainer.batch;
  }

  private recordTrainingViews(enc: GPUCommandEncoder, views: number[]): void {
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
      for (let lane = 0; lane < chunk.length; lane++) {
        const view = chunk[lane];
        const io = this.batchIO[lane];
        this.raster.recordBackwardAdd(enc, view, io);
      }
    }
  }

  private recordSingleTrainingView(enc: GPUCommandEncoder, view: number): void {
    enc.copyBufferToBuffer(this.textBuffers[view], 0, this.trainer.textBuffer, 0, this.trainer.plan.textDim * 4);
    this.raster.recordForward(enc, view, this.singleIO);
    this.trainer.encode(enc, { backward: true });
    this.raster.recordBackwardAdd(enc, view, this.singleIO);
  }

  private recordBatchInputs(enc: GPUCommandEncoder, views: number[]): void {
    const batch = this.batchTrainer!;
    for (let lane = 0; lane < views.length; lane++) {
      const view = views[lane];
      enc.copyBufferToBuffer(this.textBuffers[view], 0, batch.textBuffer, batch.textOffsetBytes(lane), batch.plan.textDim * 4);
      this.raster.recordForward(enc, view, this.batchIO[lane]);
    }
  }

  private recordSingleForwardToTrainer(enc: GPUCommandEncoder, view: number): void {
    this.raster.recordForward(enc, view, this.singleIO);
  }

  private recordSingleTextAndBackward(enc: GPUCommandEncoder, view: number): void {
    enc.copyBufferToBuffer(this.textBuffers[view], 0, this.trainer.textBuffer, 0, this.trainer.plan.textDim * 4);
    this.trainer.encodeBackward(enc);
  }

  private recordSingleRasterBackward(enc: GPUCommandEncoder, view: number): void {
    this.raster.recordBackwardAdd(enc, view, this.singleIO);
  }

  private async submitTimed(record: (enc: GPUCommandEncoder) => void): Promise<number> {
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
