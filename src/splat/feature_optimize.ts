/** CLIP-guided optimizer for the experimental 2D Feature Painter. */
/// <reference types="@webgpu/types" />
import { VisionTrainer, type TrainPlan } from "../clip/vision";
import { DECODER_PARAM_COUNT, FEATURE_LATENT_CHANNELS, FEATURE_STRIDE } from "./feature_painter_wgsl";
import { DECODER_LR, FEATURE_LR, FeaturePainterEngine } from "./feature_painter";
import { DEFAULT_HYPER, type AdamHyper, type AdamLRs } from "./adam_wgsl";
import { LEGIBLE_LRS, randomSplats, type SplatInit, type SplatNudgeOptions, nudgeSplatMask, nudgeSplats, cosine } from "./optimize";

const SIDE = 256;
const IMG_BYTES = 3 * SIDE * SIDE * 4;
/** Keep the experimental feature path compact while its denser high-count
 * schedule is evaluated separately. */
export const FEATURE_PAINTER_G = 2048;

/** Lower optical depth leaves interior splats visible and trainable. The old
 * alpha=.60/scale=9 setup had roughly 9.5 expected overlaps per pixel. */
export const FEATURE_PAINTER_INIT: Required<SplatInit> = {
  scale: 7,
  scaleJitter: 0.45,
  opacityRaw: -0.1,
  colorSpread: 1.1,
};

/** The early geometry multiplier below gives centers a real chance to migrate
 * before colour/opacity settle into the first low-frequency CLIP solution. */
export const FEATURE_PAINTER_LRS: AdamLRs = {
  ...LEGIBLE_LRS,
};

export interface FeaturePainterOptimizerConfig {
  G?: number;
  seed?: number;
  init?: SplatInit;
  lrs?: AdamLRs;
  hyper?: AdamHyper;
  featureLR?: number;
  decoderLR?: number;
}

export class FeaturePainterOptimizer {
  readonly device: GPUDevice;
  readonly raster: FeaturePainterEngine;
  readonly trainer: VisionTrainer;
  readonly side = SIDE;
  private step_ = 0;
  private readonly init: SplatInit;
  private readonly lrs: AdamLRs;
  private readonly hyper: AdamHyper;
  private readonly featureLR: number;
  private readonly decoderLR: number;

  static async create(device: GPUDevice, plan: TrainPlan, weights: Float32Array, cfg: FeaturePainterOptimizerConfig = {}) {
    const [channels, height, width] = plan.inputShape;
    if (channels !== 3 || height !== SIDE || width !== SIDE) throw new Error("feature painter requires MobileCLIP 256x256 RGB input");
    const G = cfg.G ?? FEATURE_PAINTER_G;
    const raster = await FeaturePainterEngine.create(device, { H: SIDE, W: SIDE, G, cap: 2048, bg: [0.5, 0.5, 0.5] });
    const trainer = await VisionTrainer.create(device, plan, weights);
    const init = cfg.init ?? FEATURE_PAINTER_INIT;
    raster.setParams(randomSplats(G, cfg.seed ?? 1, init));
    raster.setFeatureParams(randomFeatures(G, cfg.seed ?? 1));
    // The output residual is still exactly zero at boot: z/Ax/Ay and all
    // decoder bias/RGB-skip weights are zero. Nonzero latent columns give the
    // feature field a gradient on its very first optimization step.
    raster.setDecoderParams(randomDecoder(cfg.seed ?? 1));
    raster.zeroAdamState();
    return new FeaturePainterOptimizer(device, raster, trainer, init, cfg);
  }

  private constructor(
    device: GPUDevice,
    raster: FeaturePainterEngine,
    trainer: VisionTrainer,
    init: SplatInit,
    cfg: FeaturePainterOptimizerConfig,
  ) {
    this.device = device;
    this.raster = raster;
    this.trainer = trainer;
    this.init = init;
    this.lrs = cfg.lrs ?? FEATURE_PAINTER_LRS;
    this.hyper = cfg.hyper ?? DEFAULT_HYPER;
    this.featureLR = cfg.featureLR ?? FEATURE_LR;
    this.decoderLR = cfg.decoderLR ?? DECODER_LR;
  }

  setPrompt(text: Float32Array): void {
    this.trainer.writeText(text);
  }

  get stepCount(): number {
    return this.step_;
  }

  step(): void {
    const encoder = this.device.createCommandEncoder();
    this.raster.recordForward(encoder);
    encoder.copyBufferToBuffer(this.raster.image, 0, this.trainer.inputBuffer, 0, IMG_BYTES);
    this.trainer.encode(encoder, { backward: true });
    encoder.copyBufferToBuffer(this.trainer.inputGradBuffer, 0, this.raster.gradImage, 0, IMG_BYTES);
    this.raster.recordBackward(encoder);
    this.step_ += 1;
    this.raster.recordAdam(encoder, this.step_, this.lrsForStep(), this.hyper, undefined, {
      feature: this.featureLR,
      decoder: this.decoderLR,
    });
    this.device.queue.submit([encoder.finish()]);
  }

  /** Start mobile, then progressively settle. This is only an LR schedule, so
   * Adam's fixed beta bias correction remains mathematically valid. */
  private lrsForStep(): AdamLRs {
    const t = Math.max(0, Math.min(1, this.step_ / 180));
    const geometry = 1.6 - 0.6 * t;
    const appearance = 0.55 + 0.45 * t;
    return {
      mean: this.lrs.mean * geometry,
      logScale: this.lrs.logScale * geometry,
      theta: this.lrs.theta * geometry,
      color: this.lrs.color * appearance,
      opacity: this.lrs.opacity * appearance,
    };
  }

  async nudge(opts: SplatNudgeOptions = {}): Promise<void> {
    const G = this.raster.dims.G;
    const seed = opts.seed ?? Date.now();
    const amount = opts.amount ?? 0.12;
    const selection = nudgeSplatMask(G, seed, amount);
    const [params, features] = await Promise.all([
      this.raster.readParams(),
      this.raster.readFeatureParams(),
    ]);
    nudgeSplats(params, G, seed, amount, opts.init ?? this.init, selection);
    for (let g = 0; g < G; g++) {
      if (selection[g] !== 0) features.fill(0, g * FEATURE_STRIDE, (g + 1) * FEATURE_STRIDE);
    }
    this.raster.setParams(params);
    this.raster.setFeatureParams(features);
    this.raster.zeroAdamState();
  }

  async renderImage(): Promise<Float32Array> {
    this.raster.runForward();
    return this.raster.readImage();
  }

  async currentEmbedding(): Promise<Float32Array> {
    const encoder = this.device.createCommandEncoder();
    this.raster.recordForward(encoder);
    encoder.copyBufferToBuffer(this.raster.image, 0, this.trainer.inputBuffer, 0, IMG_BYTES);
    this.trainer.encode(encoder, { backward: false });
    this.device.queue.submit([encoder.finish()]);
    return readFloats(this.device, this.trainer.outputBuffer, this.trainer.plan.embedDim);
  }

  destroy(): void {
    this.raster.destroy();
  }
}

/**
 * Compact Feature8 initialization. RGB remains in the regular splat buffer;
 * only five latent z channels live here. All channels begin at zero, so a
 * fresh renderer exactly matches RGB even though the latent decoder columns
 * are nonzero and can route gradient immediately.
 */
export function randomFeatures(G: number, _seed = 1): Float32Array {
  return new Float32Array(G*FEATURE_STRIDE);
}

/**
 * A zero-output but gradient-active decoder initialization. The RGB-skip
 * columns and bias stay zero; only latent columns are seeded. Because every
 * latent feature starts at zero, this preserves exact RGB image parity while
 * avoiding the one-step feature-gradient dead start of an all-zero decoder.
 */
export function randomDecoder(seed = 1): Float32Array {
  let state=(seed>>>0)||1; const next=()=>{state=(Math.imul(state,747796405)+2891336453)>>>0; let t=Math.imul((state>>>((state>>>28)+4))^state,277803737)>>>0; t=((t>>>22)^t)>>>0; return t/4294967296;};
  const normal=()=>{let a=0,b=0;while(a===0)a=next();while(b===0)b=next();return Math.sqrt(-2*Math.log(a))*Math.cos(2*Math.PI*b);};
  const out=new Float32Array(DECODER_PARAM_COUNT);
  for(let output=0;output<3;output++) {
    for(let channel=0;channel<FEATURE_LATENT_CHANNELS;channel++) {
      out[output*8+3+channel]=0.25*normal();
    }
  }
  return out;
}

async function readFloats(device:GPUDevice, buffer:GPUBuffer, floats:number) { const staging=device.createBuffer({size:floats*4,usage:1|8}); const enc=device.createCommandEncoder(); enc.copyBufferToBuffer(buffer,0,staging,0,floats*4); device.queue.submit([enc.finish()]); await staging.mapAsync(1); const out=new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); staging.destroy(); return out; }
export { cosine };
