/** CLIP-guided optimizer for the experimental 2D Feature Painter. */
/// <reference types="@webgpu/types" />
import { VisionTrainer, type TrainPlan } from "../clip/vision";
import { FEATURE_GROUPS, FEATURE_STRIDE } from "./feature_painter_wgsl";
import { FeaturePainterEngine } from "./feature_painter";
import { LEGIBLE_INIT, LEGIBLE_LRS, randomSplats, type SplatInit, type SplatNudgeOptions, nudgeSplats, cosine } from "./optimize";

const SIDE = 256;
const IMG_BYTES = 3 * SIDE * SIDE * 4;
/** Lower than RGB mode because every visible hit carries 32 channels. */
export const FEATURE_PAINTER_G = 2048;

export class FeaturePainterOptimizer {
  readonly device: GPUDevice;
  readonly raster: FeaturePainterEngine;
  readonly trainer: VisionTrainer;
  readonly side = SIDE;
  private step_ = 0;
  private readonly init?: SplatInit;

  static async create(device: GPUDevice, plan: TrainPlan, weights: Float32Array, cfg: { G?: number; seed?: number; init?: SplatInit } = {}) {
    const [channels, height, width] = plan.inputShape;
    if (channels !== 3 || height !== SIDE || width !== SIDE) throw new Error("feature painter requires MobileCLIP 256x256 RGB input");
    const G = cfg.G ?? FEATURE_PAINTER_G;
    const raster = await FeaturePainterEngine.create(device, { H: SIDE, W: SIDE, G, cap: 2048, bg: [0.5, 0.5, 0.5] });
    const trainer = await VisionTrainer.create(device, plan, weights);
    raster.setParams(randomSplats(G, cfg.seed ?? 1, cfg.init));
    raster.setFeatureParams(randomFeatures(G, cfg.seed ?? 1));
    raster.zeroAdamState();
    return new FeaturePainterOptimizer(device, raster, trainer, cfg.init);
  }

  private constructor(device: GPUDevice, raster: FeaturePainterEngine, trainer: VisionTrainer, init?: SplatInit) { this.device=device; this.raster=raster; this.trainer=trainer; this.init=init; }
  setPrompt(text: Float32Array) { this.trainer.writeText(text); }
  get stepCount() { return this.step_; }
  step() { const enc=this.device.createCommandEncoder(); this.raster.recordForward(enc); enc.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,IMG_BYTES); this.trainer.encode(enc,{backward:true}); enc.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,IMG_BYTES); this.raster.recordBackward(enc); this.step_++; this.raster.recordAdam(enc,this.step_,LEGIBLE_LRS); this.device.queue.submit([enc.finish()]); }
  async nudge(opts: SplatNudgeOptions = {}) { const G=this.raster.dims.G; const params=await this.raster.readParams(); nudgeSplats(params,G,opts.seed??Date.now(),opts.amount??0.24,opts.init??this.init); this.raster.setParams(params); this.raster.zeroAdamState(); }
  async renderImage() { this.raster.runForward(); return this.raster.readImage(); }
  async currentEmbedding() { const enc=this.device.createCommandEncoder(); this.raster.recordForward(enc); enc.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,IMG_BYTES); this.trainer.encode(enc,{backward:false}); this.device.queue.submit([enc.finish()]); return readFloats(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim); }
  destroy() { this.raster.destroy(); }
}

/** Base RGB logits are active at boot; residual channels and local banks start
 * small so the painter begins near the RGB splat behavior but can develop marks. */
export function randomFeatures(G: number, seed = 1): Float32Array {
  let state=(seed>>>0)||1; const next=()=>{state=(Math.imul(state,747796405)+2891336453)>>>0; let t=Math.imul((state>>>((state>>>28)+4))^state,277803737)>>>0; t=((t>>>22)^t)>>>0; return t/4294967296;};
  const normal=()=>{let a=0,b=0;while(a===0)a=next();while(b===0)b=next();return Math.sqrt(-2*Math.log(a))*Math.cos(2*Math.PI*b);};
  const out=new Float32Array(G*FEATURE_STRIDE);
  for(let g=0;g<G;g++) for(let group=0;group<FEATURE_GROUPS;group++) { const base=(g*3)*FEATURE_GROUPS+group; const noise=group===0?1.0:0.04; out[base+0]=noise*normal(); out[base+1]=noise*normal(); out[base+2]=noise*normal(); out[base+3]=group===0?0:0.04*normal(); for(let bank=1;bank<3;bank++) for(let lane=0;lane<4;lane++) out[base+bank*FEATURE_GROUPS+lane]=0.008*normal(); }
  return out;
}

async function readFloats(device:GPUDevice, buffer:GPUBuffer, floats:number) { const staging=device.createBuffer({size:floats*4,usage:1|8}); const enc=device.createCommandEncoder(); enc.copyBufferToBuffer(buffer,0,staging,0,floats*4); device.queue.submit([enc.finish()]); await staging.mapAsync(1); const out=new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); staging.destroy(); return out; }
export { cosine };
