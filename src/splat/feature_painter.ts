/** Runtime owner for the experimental tiled 2D Feature Painter. */
/// <reference types="@webgpu/types" />
import { Feature32Colorizer, type Feature32ColorizerIOState } from "../splat3d_feature/colorizer";
import { adamShader, ADAM_UNIFORM_BYTES, type AdamHyper, type AdamLRs, DEFAULT_HYPER } from "./adam_wgsl";
import { chainShader, clearShader, emitShader, PARAM_STRIDE, prepShader, resolveDims, type RasterConfig, type RasterDims } from "./raster_wgsl";
import { FEATURE_STRIDE, featureBackwardShader, featureChainShader, featureForwardShader } from "./feature_painter_wgsl";

const U = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const WG = 256;
const ceil = (n: number) => Math.ceil(n / WG);

async function compute(device: GPUDevice, code: string, label: string): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ label, code });
  const pipe = device.createComputePipeline({ label, layout: "auto", compute: { module, entryPoint: "main" } });
  const error = await device.popErrorScope();
  if (error) throw new Error(`feature painter ${label}: ${(error as GPUValidationError).message}`);
  return pipe;
}

export interface FeaturePainterConfig extends RasterConfig {}

export class FeaturePainterEngine {
  readonly dims: RasterDims;
  readonly params: GPUBuffer;
  readonly image: GPUBuffer;
  readonly gradImage: GPUBuffer;
  readonly featureParams: GPUBuffer;
  readonly colorizer: Feature32Colorizer;
  private readonly device: GPUDevice;
  private readonly derived: GPUBuffer;
  private readonly accGeom: GPUBuffer;
  private readonly gradGeom: GPUBuffer;
  private readonly geomM: GPUBuffer;
  private readonly geomV: GPUBuffer;
  private readonly accFeature: GPUBuffer;
  private readonly gradFeature: GPUBuffer;
  private readonly featureM: GPUBuffer;
  private readonly featureV: GPUBuffer;
  private readonly featureImage: GPUBuffer;
  private readonly featureImageGrad: GPUBuffer;
  private readonly tileCounts: GPUBuffer;
  private readonly binnedIds: GPUBuffer;
  private readonly tileStop: GPUBuffer;
  private readonly prepPipe: GPUComputePipeline;
  private readonly emitPipe: GPUComputePipeline;
  private readonly forwardPipe: GPUComputePipeline;
  private readonly backwardPipe: GPUComputePipeline;
  private readonly geomChainPipe: GPUComputePipeline;
  private readonly featureChainPipe: GPUComputePipeline;
  private readonly clearBinsPipe: GPUComputePipeline;
  private readonly clearGeomPipe: GPUComputePipeline;
  private readonly clearFeaturePipe: GPUComputePipeline;
  private readonly adamPipe: GPUComputePipeline;
  private readonly prepBind: GPUBindGroup;
  private readonly emitBind: GPUBindGroup;
  private readonly forwardBind: GPUBindGroup;
  private readonly backwardBind: GPUBindGroup;
  private readonly geomChainBind: GPUBindGroup;
  private readonly featureChainBind: GPUBindGroup;
  private readonly clearBinsBind: GPUBindGroup;
  private readonly clearGeomBind: GPUBindGroup;
  private readonly clearFeatureBind: GPUBindGroup;
  private readonly geometryAdamBinds: GPUBuffer[] = [];
  private readonly geometryAdamGroups: GPUBindGroup[] = [];
  private readonly featureAdamUniform: GPUBuffer;
  private readonly featureAdamGroup: GPUBindGroup;
  private readonly colorIO: Feature32ColorizerIOState;

  private constructor(device: GPUDevice, cfg: FeaturePainterConfig, colorizer: Feature32Colorizer) {
    this.device = device; this.dims = resolveDims(cfg); this.colorizer = colorizer;
    const d = this.dims; const g9 = d.G * PARAM_STRIDE; const fN = d.G * FEATURE_STRIDE;
    const storage = (count: number, extra = 0) => device.createBuffer({ size: count * 4, usage: U.STORAGE | extra });
    this.params = storage(g9, U.COPY_SRC | U.COPY_DST); this.derived = storage(g9); this.accGeom = storage(g9, U.COPY_DST); this.gradGeom = storage(g9); this.geomM = storage(g9, U.COPY_DST); this.geomV = storage(g9, U.COPY_DST);
    this.featureParams = storage(fN, U.COPY_SRC | U.COPY_DST); this.accFeature = storage(fN, U.COPY_DST); this.gradFeature = storage(fN); this.featureM = storage(fN, U.COPY_DST); this.featureV = storage(fN, U.COPY_DST);
    this.tileCounts = storage(d.numTiles, U.COPY_DST); this.binnedIds = storage(d.numTiles * d.cap); this.tileStop = storage(d.numTiles);
    this.featureImage = storage(32 * d.H * d.W); this.featureImageGrad = storage(32 * d.H * d.W); this.image = storage(3 * d.H * d.W, U.COPY_SRC); this.gradImage = storage(3 * d.H * d.W, U.COPY_DST);
    this.prepPipe = null as unknown as GPUComputePipeline; this.emitPipe = null as unknown as GPUComputePipeline; this.forwardPipe = null as unknown as GPUComputePipeline; this.backwardPipe = null as unknown as GPUComputePipeline; this.geomChainPipe = null as unknown as GPUComputePipeline; this.featureChainPipe = null as unknown as GPUComputePipeline; this.clearBinsPipe = null as unknown as GPUComputePipeline; this.clearGeomPipe = null as unknown as GPUComputePipeline; this.clearFeaturePipe = null as unknown as GPUComputePipeline; this.adamPipe = null as unknown as GPUComputePipeline;
    this.prepBind = null as unknown as GPUBindGroup; this.emitBind = null as unknown as GPUBindGroup; this.forwardBind = null as unknown as GPUBindGroup; this.backwardBind = null as unknown as GPUBindGroup; this.geomChainBind = null as unknown as GPUBindGroup; this.featureChainBind = null as unknown as GPUBindGroup; this.clearBinsBind = null as unknown as GPUBindGroup; this.clearGeomBind = null as unknown as GPUBindGroup; this.clearFeatureBind = null as unknown as GPUBindGroup;
    this.featureAdamUniform = device.createBuffer({ size: ADAM_UNIFORM_BYTES, usage: U.UNIFORM | U.COPY_DST }); this.featureAdamGroup = null as unknown as GPUBindGroup;
    this.colorIO = colorizer.createIOState({ features: this.featureImage, rgb: this.image, rgbGrad: this.gradImage, featureGrad: this.featureImageGrad });
  }

  static async create(device: GPUDevice, cfg: FeaturePainterConfig): Promise<FeaturePainterEngine> {
    const colorizer = await Feature32Colorizer.create(device, { width: cfg.W, height: cfg.H, batch: 1, label: "feature-painter-colorizer" });
    const e = new FeaturePainterEngine(device, cfg, colorizer); await e.build(cfg); return e;
  }

  private async build(cfg: FeaturePainterConfig): Promise<void> {
    const d = this.dims;
    (this as any).prepPipe = await compute(this.device, prepShader(cfg), "prep"); (this as any).emitPipe = await compute(this.device, emitShader(cfg), "emit"); (this as any).forwardPipe = await compute(this.device, featureForwardShader(cfg), "forward"); (this as any).backwardPipe = await compute(this.device, featureBackwardShader(cfg), "backward"); (this as any).geomChainPipe = await compute(this.device, chainShader(cfg), "geometry-chain"); (this as any).featureChainPipe = await compute(this.device, featureChainShader(cfg), "feature-chain"); (this as any).clearBinsPipe = await compute(this.device, clearShader(d.numTiles), "clear-bins"); (this as any).clearGeomPipe = await compute(this.device, clearShader(d.G * PARAM_STRIDE), "clear-geometry"); (this as any).clearFeaturePipe = await compute(this.device, clearShader(d.G * FEATURE_STRIDE), "clear-feature"); (this as any).adamPipe = await compute(this.device, adamShader(), "adam");
    const bind = (pipe: GPUComputePipeline, buffers: GPUBuffer[]) => this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    (this as any).prepBind = bind(this.prepPipe, [this.params, this.derived]); (this as any).emitBind = bind(this.emitPipe, [this.derived, this.tileCounts, this.binnedIds]); (this as any).forwardBind = bind(this.forwardPipe, [this.tileCounts, this.binnedIds, this.derived, this.featureParams, this.featureImage, this.tileStop]); (this as any).backwardBind = bind(this.backwardPipe, [this.featureImageGrad, this.tileCounts, this.binnedIds, this.tileStop, this.derived, this.featureParams, this.accGeom, this.accFeature]); (this as any).geomChainBind = bind(this.geomChainPipe, [this.accGeom, this.derived, this.params, this.gradGeom]); (this as any).featureChainBind = bind(this.featureChainPipe, [this.accFeature, this.gradFeature]); (this as any).clearBinsBind = bind(this.clearBinsPipe, [this.tileCounts]); (this as any).clearGeomBind = bind(this.clearGeomPipe, [this.accGeom]); (this as any).clearFeatureBind = bind(this.clearFeaturePipe, [this.accFeature]);
    const segments = [{ offset: 0, length: 2*d.G, lr: "mean" }, { offset: 2*d.G, length: 2*d.G, lr: "logScale" }, { offset: 4*d.G, length: d.G, lr: "theta" }, { offset: 5*d.G, length: 3*d.G, lr: "color" }, { offset: 8*d.G, length: d.G, lr: "opacity" }];
    for (const s of segments) { const uni = this.device.createBuffer({ size: ADAM_UNIFORM_BYTES, usage: U.UNIFORM | U.COPY_DST }); this.geometryAdamBinds.push(uni); this.geometryAdamGroups.push(this.device.createBindGroup({ layout: this.adamPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: this.params } }, { binding: 2, resource: { buffer: this.gradGeom } }, { binding: 3, resource: { buffer: this.geomM } }, { binding: 4, resource: { buffer: this.geomV } }] })); }
    (this as any).featureAdamGroup = this.device.createBindGroup({ layout: this.adamPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.featureAdamUniform } }, { binding: 1, resource: { buffer: this.featureParams } }, { binding: 2, resource: { buffer: this.gradFeature } }, { binding: 3, resource: { buffer: this.featureM } }, { binding: 4, resource: { buffer: this.featureV } }] });
  }

  setParams(data: Float32Array) { this.device.queue.writeBuffer(this.params, 0, data as unknown as BufferSource); }
  setFeatureParams(data: Float32Array) { if (data.length !== this.dims.G * FEATURE_STRIDE) throw new Error("feature painter: wrong feature count"); this.device.queue.writeBuffer(this.featureParams, 0, data as unknown as BufferSource); }
  private async read(buffer: GPUBuffer, floats: number): Promise<Float32Array> { const stage=this.device.createBuffer({size:floats*4,usage:U.MAP_READ|U.COPY_DST}); const enc=this.device.createCommandEncoder(); enc.copyBufferToBuffer(buffer,0,stage,0,floats*4); this.device.queue.submit([enc.finish()]); await stage.mapAsync(1); const out=new Float32Array(stage.getMappedRange().slice(0)); stage.unmap(); stage.destroy(); return out; }
  readParams() { return this.read(this.params, this.dims.G * PARAM_STRIDE); }
  readFeatureParams() { return this.read(this.featureParams, this.dims.G * FEATURE_STRIDE); }
  readImage() { return this.read(this.image, 3 * this.dims.H * this.dims.W); }
  zeroAdamState() { const z9 = new Float32Array(this.dims.G * PARAM_STRIDE); const zf = new Float32Array(this.dims.G * FEATURE_STRIDE); this.device.queue.writeBuffer(this.geomM, 0, z9); this.device.queue.writeBuffer(this.geomV, 0, z9); this.device.queue.writeBuffer(this.featureM, 0, zf); this.device.queue.writeBuffer(this.featureV, 0, zf); }
  recordForward(enc: GPUCommandEncoder) { const p = enc.beginComputePass(); p.setPipeline(this.prepPipe); p.setBindGroup(0,this.prepBind); p.dispatchWorkgroups(ceil(this.dims.G)); p.setPipeline(this.clearBinsPipe); p.setBindGroup(0,this.clearBinsBind); p.dispatchWorkgroups(ceil(this.dims.numTiles)); p.setPipeline(this.emitPipe); p.setBindGroup(0,this.emitBind); p.dispatchWorkgroups(ceil(this.dims.G)); p.setPipeline(this.forwardPipe); p.setBindGroup(0,this.forwardBind); p.dispatchWorkgroups(this.dims.numTiles); p.end(); this.colorizer.recordForward(enc,this.colorIO); }
  recordBackward(enc: GPUCommandEncoder) { this.colorizer.recordBackward(enc,this.colorIO); const p=enc.beginComputePass(); p.setPipeline(this.clearGeomPipe); p.setBindGroup(0,this.clearGeomBind); p.dispatchWorkgroups(ceil(this.dims.G*PARAM_STRIDE)); p.setPipeline(this.clearFeaturePipe); p.setBindGroup(0,this.clearFeatureBind); p.dispatchWorkgroups(ceil(this.dims.G*FEATURE_STRIDE)); p.setPipeline(this.backwardPipe); p.setBindGroup(0,this.backwardBind); p.dispatchWorkgroups(this.dims.numTiles); p.setPipeline(this.geomChainPipe); p.setBindGroup(0,this.geomChainBind); p.dispatchWorkgroups(ceil(this.dims.G)); p.setPipeline(this.featureChainPipe); p.setBindGroup(0,this.featureChainBind); p.dispatchWorkgroups(ceil(this.dims.G*FEATURE_STRIDE)); p.end(); }
  recordAdam(enc: GPUCommandEncoder, step:number, lrs:AdamLRs, hyper:AdamHyper=DEFAULT_HYPER) { const segs=[{offset:0,length:2*this.dims.G,lr:lrs.mean},{offset:2*this.dims.G,length:2*this.dims.G,lr:lrs.logScale},{offset:4*this.dims.G,length:this.dims.G,lr:lrs.theta},{offset:5*this.dims.G,length:3*this.dims.G,lr:lrs.color},{offset:8*this.dims.G,length:this.dims.G,lr:lrs.opacity}]; const bc1=1-Math.pow(hyper.beta1,step),bc2=1-Math.pow(hyper.beta2,step); const write=(buf:GPUBuffer,offset:number,count:number,lr:number)=>{const a=new ArrayBuffer(32),u32=new Uint32Array(a),f32=new Float32Array(a);u32[0]=offset;u32[1]=count;f32[2]=lr;f32[3]=hyper.beta1;f32[4]=hyper.beta2;f32[5]=hyper.eps;f32[6]=bc1;f32[7]=bc2;this.device.queue.writeBuffer(buf,0,a);}; segs.forEach((s,i)=>write(this.geometryAdamBinds[i],s.offset,s.length,s.lr)); write(this.featureAdamUniform,0,this.dims.G*FEATURE_STRIDE,0.025); const p=enc.beginComputePass(); p.setPipeline(this.adamPipe); segs.forEach((s,i)=>{p.setBindGroup(0,this.geometryAdamGroups[i]);p.dispatchWorkgroups(ceil(s.length));}); p.setBindGroup(0,this.featureAdamGroup);p.dispatchWorkgroups(ceil(this.dims.G*FEATURE_STRIDE));p.end(); this.colorizer.recordSgd(enc, 0.0025); }
  runForward() { const enc=this.device.createCommandEncoder(); this.recordForward(enc); this.device.queue.submit([enc.finish()]); }
  destroy() { for(const b of [this.params,this.derived,this.accGeom,this.gradGeom,this.geomM,this.geomV,this.featureParams,this.accFeature,this.gradFeature,this.featureM,this.featureV,this.tileCounts,this.binnedIds,this.tileStop,this.featureImage,this.featureImageGrad,this.image,this.gradImage,...this.geometryAdamBinds,this.featureAdamUniform]) try{b.destroy()}catch{} this.colorizer.destroy(); }
}
