/// <reference types="@webgpu/types" />
import { VisionTrainer, type TrainPlan } from "../clip/vision";
import { type AdamHyper, DEFAULT_HYPER } from "../splat/adam_wgsl";
import { DEFAULT_3D_CAMERAS, type Camera3D, type PreparedCamera3D, prepareCamera } from "./cameras";
import { Raster3DEngine, type AdamLRs3D, DEFAULT_3D_LRS } from "./raster";
import { PARAM_STRIDE_3D } from "./raster_wgsl";

const SIDE = 256;
const IMG_BYTES = 3 * SIDE * SIDE * 4;
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
  readonly cameras: PreparedCamera3D[];
  readonly side = SIDE;
  private readonly textBuffers: GPUBuffer[];
  private readonly lrs: AdamLRs3D;
  private readonly hyper: AdamHyper;
  private step_ = 0;
  private hasPrompts = false;

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
    raster.setParams(cfg.initParams ?? randomSplats3D(G, cfg.seed ?? 1, cfg.init));
    raster.zeroAdamState();
    return new Splat3DOptimizer(device, raster, trainer, cameras, cfg);
  }

  private constructor(
    device: GPUDevice,
    raster: Raster3DEngine,
    trainer: VisionTrainer,
    cameras: PreparedCamera3D[],
    cfg: Splat3DOptimizerConfig
  ) {
    this.device = device;
    this.raster = raster;
    this.trainer = trainer;
    this.cameras = cameras;
    this.lrs = cfg.lrs ?? DEFAULT_3D_LRS;
    this.hyper = cfg.hyper ?? DEFAULT_HYPER;
    this.textBuffers = cameras.map((_, i) =>
      device.createBuffer({
        label: `splat3d-text-${i}`,
        size: trainer.plan.textDim * 4,
        usage: U.COPY_SRC | U.COPY_DST,
      })
    );
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

  step(displayView = 0): void {
    if (!this.hasPrompts) throw new Error("splat3d: setViewPrompts() before step()");
    const enc = this.device.createCommandEncoder();
    this.raster.recordClearRawGrad(enc);
    for (let v = 0; v < this.cameras.length; v++) {
      enc.copyBufferToBuffer(this.textBuffers[v], 0, this.trainer.textBuffer, 0, this.trainer.plan.textDim * 4);
      this.raster.recordForward(enc, v);
      enc.copyBufferToBuffer(this.raster.image, 0, this.trainer.inputBuffer, 0, IMG_BYTES);
      this.trainer.encode(enc, { backward: true });
      enc.copyBufferToBuffer(this.trainer.inputGradBuffer, 0, this.raster.gradImage, 0, IMG_BYTES);
      this.raster.recordBackwardAdd(enc, v);
    }
    this.step_ += 1;
    this.raster.recordAdam(enc, this.step_, this.lrs, this.hyper);
    this.raster.recordForward(enc, displayView);
    this.device.queue.submit([enc.finish()]);
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
    this.raster.recordForward(enc, view);
    enc.copyBufferToBuffer(this.raster.image, 0, this.trainer.inputBuffer, 0, IMG_BYTES);
    this.trainer.encode(enc, { backward: false });
    this.device.queue.submit([enc.finish()]);
    return readFloats(this.device, this.trainer.outputBuffer, this.trainer.plan.embedDim);
  }

  destroy(): void {
    this.raster.destroy();
    for (const b of this.textBuffers) {
      try {
        b.destroy();
      } catch (_) {}
    }
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
