import { DEFAULT_HYPER, type AdamHyper, adamShader, ADAM_UNIFORM_BYTES } from "../splat/adam_wgsl";
import type { PreparedCamera3D } from "./cameras";
import {
  Raster3DConfig,
  Raster3DDims,
  resolveDims3D,
  prepShader3D,
  prepBatchShader3D,
  emitShader3D,
  emitBatchShader3D,
  forwardShader3D,
  forwardBatchShader3D,
  backwardShader3D,
  backwardBatchShader3D,
  chainAddShader3D,
  clearShader3D,
  paramSegments3D,
  PARAM_STRIDE_3D,
  DERIVED_STRIDE_3D,
  CAMERA_STRIDE_3D,
} from "./raster_wgsl";

const U = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const WG = 256;
const STORAGE_OFFSET_ALIGN = 256;
const ceil = (n: number) => Math.ceil(n / WG);
type BufferBindingInput = GPUBuffer | { buffer: GPUBuffer; offset?: number; size?: number };

export interface Raster3DEngineConfig extends Raster3DConfig {
  cameras: PreparedCamera3D[];
}

export interface Raster3DIOState {
  prepBind: GPUBindGroup[];
  chainBind: GPUBindGroup[];
  emitBind: GPUBindGroup;
  clearBinsBind: GPUBindGroup;
  clearGradsBind: GPUBindGroup;
  fwdBind: GPUBindGroup;
  bwdBind: GPUBindGroup;
}

export interface Raster3DIOOptions {
  privateState?: boolean;
}

interface Raster3DScratchState {
  derived: BufferBindingInput;
  accGrad: BufferBindingInput;
  tileCounts: BufferBindingInput;
  binnedIds: BufferBindingInput;
  tileStop: BufferBindingInput;
  prepBind: GPUBindGroup[];
  chainBind: GPUBindGroup[];
  emitBind: GPUBindGroup;
  clearBinsBind: GPUBindGroup;
  clearGradsBind: GPUBindGroup;
}

interface Raster3DRawScratchBuffers {
  derived: GPUBuffer;
  tileCounts: GPUBuffer;
  binnedIds: GPUBuffer;
  tileStop: GPUBuffer;
}

export interface Raster3DBatchForwardState {
  lanes: number;
  activeViews: GPUBuffer;
  ios: Raster3DIOState[];
  prepPipe: GPUComputePipeline;
  clearBinsPipe: GPUComputePipeline;
  emitPipe: GPUComputePipeline;
  fwdPipe: GPUComputePipeline;
  clearGradsPipe: GPUComputePipeline;
  bwdPipe: GPUComputePipeline;
  prepBind: GPUBindGroup;
  clearBinsBind: GPUBindGroup;
  emitBind: GPUBindGroup;
  fwdBind: GPUBindGroup;
  clearGradsBind: GPUBindGroup;
  bwdBind: GPUBindGroup;
}

export interface Raster3DBatchForwardOptions {
  lanes: number;
  imageBuffer: GPUBuffer;
  imageOffsets: number[];
  gradBuffer: GPUBuffer;
  gradOffsets: number[];
}

export interface AdamLRs3D {
  position: number;
  logRadius: number;
  color: number;
  opacity: number;
}

export const DEFAULT_3D_LRS: AdamLRs3D = {
  position: 0.025,
  logRadius: 0.01,
  color: 0.08,
  opacity: 0.03,
};

async function makeCompute(device: GPUDevice, code: string, label: string): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const err = await device.popErrorScope();
  if (err) {
    console.error(`--- WGSL that failed (${label}) ---\n${code}`);
    throw new Error(`raster3d pipeline validation (${label}): ${(err as GPUValidationError).message}`);
  }
  return pipeline;
}

export class Raster3DEngine {
  readonly dims: Raster3DDims;
  readonly cameras: PreparedCamera3D[];
  private readonly device: GPUDevice;

  private prepPipe: GPUComputePipeline[] = [];
  private chainPipe: GPUComputePipeline[] = [];
  private emitPipe!: GPUComputePipeline;
  private fwdPipe!: GPUComputePipeline;
  private bwdPipe!: GPUComputePipeline;
  private clearBinsPipe!: GPUComputePipeline;
  private clearGradsPipe!: GPUComputePipeline;
  private clearRawPipe!: GPUComputePipeline;
  private adamPipe!: GPUComputePipeline;

  params!: GPUBuffer;
  derived!: GPUBuffer;
  accGrad!: GPUBuffer;
  gradRaw!: GPUBuffer;
  mBuf!: GPUBuffer;
  vBuf!: GPUBuffer;
  tileCounts!: GPUBuffer;
  binnedIds!: GPUBuffer;
  tileStop!: GPUBuffer;
  image!: GPUBuffer;
  gradImage!: GPUBuffer;
  private cameraBuffer!: GPUBuffer;

  private prepBind: GPUBindGroup[] = [];
  private chainBind: GPUBindGroup[] = [];
  private emitBind!: GPUBindGroup;
  private fwdBind!: GPUBindGroup;
  private bwdBind!: GPUBindGroup;
  private clearBinsBind!: GPUBindGroup;
  private clearGradsBind!: GPUBindGroup;
  private clearRawBind!: GPUBindGroup;
  private adamUni: GPUBuffer[] = [];
  private adamBind: GPUBindGroup[] = [];
  private extraBuffers: GPUBuffer[] = [];

  private constructor(device: GPUDevice, cfg: Raster3DEngineConfig) {
    this.device = device;
    this.dims = resolveDims3D(cfg);
    this.cameras = cfg.cameras;
    if (!this.cameras.length) throw new Error("raster3d: at least one camera is required");
  }

  static async create(device: GPUDevice, cfg: Raster3DEngineConfig): Promise<Raster3DEngine> {
    const e = new Raster3DEngine(device, cfg);
    await e.build(cfg);
    return e;
  }

  private storage(floats: number, extra = 0): GPUBuffer {
    return this.device.createBuffer({ size: floats * 4, usage: U.STORAGE | extra });
  }

  private bindGroup(pipe: GPUComputePipeline, bindings: BufferBindingInput[]): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: bindings.map((resource, binding) => ({
        binding,
        resource: "buffer" in resource ? resource : { buffer: resource },
      })),
    });
  }

  private async build(cfg: Raster3DEngineConfig): Promise<void> {
    const d = this.dims;
    const GP = d.G * PARAM_STRIDE_3D;
    const GD = d.G * DERIVED_STRIDE_3D;

    this.params = this.storage(GP, U.COPY_SRC | U.COPY_DST);
    this.derived = this.storage(GD);
    this.accGrad = this.storage(GD, U.COPY_DST);
    this.gradRaw = this.storage(GP, U.COPY_SRC | U.COPY_DST);
    this.mBuf = this.storage(GP, U.COPY_DST);
    this.vBuf = this.storage(GP, U.COPY_DST);
    this.tileCounts = this.storage(d.numTiles, U.COPY_DST | U.COPY_SRC);
    this.binnedIds = this.storage(d.numTiles * d.cap);
    this.tileStop = this.storage(d.numTiles);
    this.image = this.storage(3 * d.H * d.W, U.COPY_SRC);
    this.gradImage = this.storage(3 * d.H * d.W, U.COPY_DST);
    this.cameraBuffer = this.device.createBuffer({
      label: "splat3d-cameras",
      size: this.cameras.length * CAMERA_STRIDE_3D * 4,
      usage: U.STORAGE | U.COPY_DST,
    });
    this.device.queue.writeBuffer(this.cameraBuffer, 0, serializeCameras3D(this.cameras) as unknown as BufferSource);

    this.prepPipe = await Promise.all(this.cameras.map((cam, i) => makeCompute(this.device, prepShader3D(cfg, cam), `prep-${i}`)));
    this.chainPipe = await Promise.all(
      this.cameras.map((cam, i) => makeCompute(this.device, chainAddShader3D(cfg, cam), `chain-${i}`))
    );
    this.emitPipe = await makeCompute(this.device, emitShader3D(cfg), "emit");
    this.fwdPipe = await makeCompute(this.device, forwardShader3D(cfg), "forward");
    this.bwdPipe = await makeCompute(this.device, backwardShader3D(cfg), "backward");
    this.clearBinsPipe = await makeCompute(this.device, clearShader3D(d.numTiles), "clearBins");
    this.clearGradsPipe = await makeCompute(this.device, clearShader3D(GD), "clearGrads");
    this.clearRawPipe = await makeCompute(this.device, clearShader3D(GP), "clearRawGrad");
    this.adamPipe = await makeCompute(this.device, adamShader(), "adam");

    this.prepBind = this.prepPipe.map((pipe) => this.bindGroup(pipe, [this.params, this.derived]));
    this.chainBind = this.chainPipe.map((pipe) => this.bindGroup(pipe, [this.accGrad, this.derived, this.params, this.gradRaw]));
    this.emitBind = this.bindGroup(this.emitPipe, [this.derived, this.tileCounts, this.binnedIds]);
    this.clearBinsBind = this.bindGroup(this.clearBinsPipe, [this.tileCounts]);
    this.clearGradsBind = this.bindGroup(this.clearGradsPipe, [this.accGrad]);
    this.clearRawBind = this.bindGroup(this.clearRawPipe, [this.gradRaw]);
    const shared = this.sharedScratchState();
    this.fwdBind = this.makeForwardBind(shared, this.image, 0);
    this.bwdBind = this.makeBackwardBind(shared, this.gradImage, 0);

    for (const _ of paramSegments3D(d.G)) {
      const uni = this.device.createBuffer({ size: ADAM_UNIFORM_BYTES, usage: U.UNIFORM | U.COPY_DST });
      this.adamUni.push(uni);
      this.adamBind.push(
        this.device.createBindGroup({
          layout: this.adamPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uni } },
            { binding: 1, resource: { buffer: this.params } },
            { binding: 2, resource: { buffer: this.gradRaw } },
            { binding: 3, resource: { buffer: this.mBuf } },
            { binding: 4, resource: { buffer: this.vBuf } },
          ],
        })
      );
    }
  }

  setParams(data: Float32Array): void {
    if (data.length !== this.dims.G * PARAM_STRIDE_3D) throw new Error("setParams3D: wrong length");
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  zeroAdamState(): void {
    const z = new Float32Array(this.dims.G * PARAM_STRIDE_3D);
    this.device.queue.writeBuffer(this.mBuf, 0, z);
    this.device.queue.writeBuffer(this.vBuf, 0, z);
  }

  private async readFloats(buf: GPUBuffer, floats: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({ size: floats * 4, usage: U.MAP_READ | U.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  readImage(): Promise<Float32Array> {
    return this.readFloats(this.image, 3 * this.dims.H * this.dims.W);
  }

  readParams(): Promise<Float32Array> {
    return this.readFloats(this.params, this.dims.G * PARAM_STRIDE_3D);
  }

  createIOState(
    imageBuffer: GPUBuffer,
    imageOffset: number,
    gradBuffer: GPUBuffer,
    gradOffset: number,
    opts: Raster3DIOOptions = {}
  ): Raster3DIOState {
    this.checkIOBinding("image", imageOffset);
    this.checkIOBinding("grad", gradOffset);
    const scratch = opts.privateState ? this.createPrivateScratchState() : this.sharedScratchState();
    return this.createIOStateForScratch(scratch, imageBuffer, imageOffset, gradBuffer, gradOffset);
  }

  async createBatchForwardState(opts: Raster3DBatchForwardOptions): Promise<Raster3DBatchForwardState> {
    const d = this.dims;
    const lanes = opts.lanes | 0;
    if (lanes < 1 || lanes > this.cameras.length) {
      throw new Error(`raster3d: invalid batch-forward lanes ${opts.lanes}`);
    }
    if (opts.imageOffsets.length !== lanes || opts.gradOffsets.length !== lanes) {
      throw new Error("raster3d: batch-forward offsets must match lane count");
    }
    this.checkContiguousImageOffsets("batch image", opts.imageOffsets);
    this.checkContiguousImageOffsets("batch grad", opts.gradOffsets);

    const raw = this.createRawScratchBuffers(lanes);
    const batchAccGrad = this.storage(d.G * DERIVED_STRIDE_3D * lanes, U.COPY_DST);
    const activeViews = this.device.createBuffer({
      label: "splat3d-batch-active-views",
      size: lanes * 4,
      usage: U.STORAGE | U.COPY_DST,
    });
    this.extraBuffers.push(raw.derived, raw.tileCounts, raw.binnedIds, raw.tileStop, batchAccGrad, activeViews);

    const prepPipe = await makeCompute(this.device, prepBatchShader3D(d), "prep-batch");
    const clearBinsPipe = await makeCompute(this.device, clearShader3D(d.numTiles * lanes), "clearBins-batch");
    const emitPipe = await makeCompute(this.device, emitBatchShader3D(d), "emit-batch");
    const fwdPipe = await makeCompute(this.device, forwardBatchShader3D(d), "forward-batch");
    const clearGradsPipe = await makeCompute(this.device, clearShader3D(d.G * DERIVED_STRIDE_3D * lanes), "clearGrads-batch");
    const bwdPipe = await makeCompute(this.device, backwardBatchShader3D(d), "backward-batch");
    const imageOffset = opts.imageOffsets[0];
    const gradOffset = opts.gradOffsets[0];
    const prepBind = this.bindGroup(prepPipe, [this.params, this.cameraBuffer, activeViews, raw.derived]);
    const clearBinsBind = this.bindGroup(clearBinsPipe, [raw.tileCounts]);
    const emitBind = this.bindGroup(emitPipe, [raw.derived, raw.tileCounts, raw.binnedIds]);
    const fwdBind = this.bindGroup(fwdPipe, [
      raw.tileCounts,
      raw.binnedIds,
      raw.derived,
      { buffer: opts.imageBuffer, offset: imageOffset, size: this.imageByteSize() * lanes },
      raw.tileStop,
    ]);
    const clearGradsBind = this.bindGroup(clearGradsPipe, [batchAccGrad]);
    const bwdBind = this.bindGroup(bwdPipe, [
      { buffer: opts.gradBuffer, offset: gradOffset, size: this.imageByteSize() * lanes },
      raw.tileCounts,
      raw.binnedIds,
      raw.tileStop,
      raw.derived,
      batchAccGrad,
    ]);
    const ios = Array.from({ length: lanes }, (_unused, lane) =>
      this.createIOStateForScratch(
        this.laneScratchState(raw, lane, batchAccGrad),
        opts.imageBuffer,
        opts.imageOffsets[lane],
        opts.gradBuffer,
        opts.gradOffsets[lane]
      )
    );
    return {
      lanes,
      activeViews,
      ios,
      prepPipe,
      clearBinsPipe,
      emitPipe,
      fwdPipe,
      clearGradsPipe,
      bwdPipe,
      prepBind,
      clearBinsBind,
      emitBind,
      fwdBind,
      clearGradsBind,
      bwdBind,
    };
  }

  recordClearRawGrad(enc: GPUCommandEncoder): void {
    const p = enc.beginComputePass();
    p.setPipeline(this.clearRawPipe);
    p.setBindGroup(0, this.clearRawBind);
    p.dispatchWorkgroups(ceil(this.dims.G * PARAM_STRIDE_3D));
    p.end();
  }

  recordForward(enc: GPUCommandEncoder, view = 0, io?: Raster3DIOState): void {
    const p = enc.beginComputePass();
    this.encodeForwardPass(p, view, io);
    p.end();
  }

  recordForwards(enc: GPUCommandEncoder, views: number[], ios: Raster3DIOState[]): void {
    if (views.length !== ios.length) {
      throw new Error(`raster3d: ${views.length} views but ${ios.length} IO states`);
    }
    const p = enc.beginComputePass();
    for (let i = 0; i < views.length; i++) {
      this.encodeForwardPass(p, views[i], ios[i]);
    }
    p.end();
  }

  recordBatchForward(enc: GPUCommandEncoder, state: Raster3DBatchForwardState, views: number[]): void {
    if (views.length < 1 || views.length > state.lanes) {
      throw new Error(`raster3d: ${views.length} batch-forward views for ${state.lanes} lanes`);
    }
    const active = new Uint32Array(state.lanes);
    for (let lane = 0; lane < views.length; lane++) active[lane] = this.viewIndex(views[lane]);
    this.device.queue.writeBuffer(state.activeViews, 0, active as unknown as BufferSource);
    const d = this.dims;
    const p = enc.beginComputePass();
    p.setPipeline(state.prepPipe);
    p.setBindGroup(0, state.prepBind);
    p.dispatchWorkgroups(ceil(d.G), 1, views.length);
    p.setPipeline(state.clearBinsPipe);
    p.setBindGroup(0, state.clearBinsBind);
    p.dispatchWorkgroups(ceil(d.numTiles * views.length));
    p.setPipeline(state.emitPipe);
    p.setBindGroup(0, state.emitBind);
    p.dispatchWorkgroups(ceil(d.G), 1, views.length);
    p.setPipeline(state.fwdPipe);
    p.setBindGroup(0, state.fwdBind);
    p.dispatchWorkgroups(d.numTiles, 1, views.length);
    p.end();
  }

  recordBatchBackwardAdd(enc: GPUCommandEncoder, state: Raster3DBatchForwardState, views: number[]): void {
    if (views.length < 1 || views.length > state.lanes) {
      throw new Error(`raster3d: ${views.length} batch-backward views for ${state.lanes} lanes`);
    }
    const d = this.dims;
    const p = enc.beginComputePass();
    p.setPipeline(state.clearGradsPipe);
    p.setBindGroup(0, state.clearGradsBind);
    p.dispatchWorkgroups(ceil(d.G * DERIVED_STRIDE_3D * views.length));
    p.setPipeline(state.bwdPipe);
    p.setBindGroup(0, state.bwdBind);
    p.dispatchWorkgroups(d.numTiles, 1, views.length);
    for (let lane = 0; lane < views.length; lane++) {
      const v = this.viewIndex(views[lane]);
      p.setPipeline(this.chainPipe[v]);
      p.setBindGroup(0, state.ios[lane].chainBind[v]);
      p.dispatchWorkgroups(ceil(d.G));
    }
    p.end();
  }

  private encodeForwardPass(p: GPUComputePassEncoder, view = 0, io?: Raster3DIOState): void {
    const d = this.dims;
    const v = this.viewIndex(view);
    p.setPipeline(this.prepPipe[v]);
    p.setBindGroup(0, io?.prepBind[v] ?? this.prepBind[v]);
    p.dispatchWorkgroups(ceil(d.G));
    p.setPipeline(this.clearBinsPipe);
    p.setBindGroup(0, io?.clearBinsBind ?? this.clearBinsBind);
    p.dispatchWorkgroups(ceil(d.numTiles));
    p.setPipeline(this.emitPipe);
    p.setBindGroup(0, io?.emitBind ?? this.emitBind);
    p.dispatchWorkgroups(ceil(d.G));
    p.setPipeline(this.fwdPipe);
    p.setBindGroup(0, io?.fwdBind ?? this.fwdBind);
    p.dispatchWorkgroups(d.numTiles);
  }

  recordBackwardAdd(enc: GPUCommandEncoder, view = 0, io?: Raster3DIOState): void {
    const d = this.dims;
    const v = this.viewIndex(view);
    const p = enc.beginComputePass();
    p.setPipeline(this.clearGradsPipe);
    p.setBindGroup(0, io?.clearGradsBind ?? this.clearGradsBind);
    p.dispatchWorkgroups(ceil(d.G * DERIVED_STRIDE_3D));
    p.setPipeline(this.bwdPipe);
    p.setBindGroup(0, io?.bwdBind ?? this.bwdBind);
    p.dispatchWorkgroups(d.numTiles);
    p.setPipeline(this.chainPipe[v]);
    p.setBindGroup(0, io?.chainBind[v] ?? this.chainBind[v]);
    p.dispatchWorkgroups(ceil(d.G));
    p.end();
  }

  recordAdam(
    enc: GPUCommandEncoder,
    step: number,
    lrs: AdamLRs3D = DEFAULT_3D_LRS,
    hyper: AdamHyper = DEFAULT_HYPER
  ): void {
    const segs = paramSegments3D(this.dims.G);
    const lrByName: Record<string, number> = {
      position: lrs.position,
      logRadius: lrs.logRadius,
      color: lrs.color,
      opacity: lrs.opacity,
    };
    const bc1 = 1 - Math.pow(hyper.beta1, step);
    const bc2 = 1 - Math.pow(hyper.beta2, step);
    segs.forEach((s, i) => {
      const buf = new ArrayBuffer(ADAM_UNIFORM_BYTES);
      const u32 = new Uint32Array(buf);
      const f32 = new Float32Array(buf);
      u32[0] = s.offset;
      u32[1] = s.length;
      f32[2] = lrByName[s.name];
      f32[3] = hyper.beta1;
      f32[4] = hyper.beta2;
      f32[5] = hyper.eps;
      f32[6] = bc1;
      f32[7] = bc2;
      this.device.queue.writeBuffer(this.adamUni[i], 0, buf);
    });

    const p = enc.beginComputePass();
    p.setPipeline(this.adamPipe);
    segs.forEach((s, i) => {
      p.setBindGroup(0, this.adamBind[i]);
      p.dispatchWorkgroups(ceil(s.length));
    });
    p.end();
  }

  runForward(view = 0): void {
    const enc = this.device.createCommandEncoder();
    this.recordForward(enc, view);
    this.device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    for (const b of [
      this.params,
      this.derived,
      this.accGrad,
      this.gradRaw,
      this.mBuf,
      this.vBuf,
      this.tileCounts,
      this.binnedIds,
      this.tileStop,
      this.image,
      this.gradImage,
      this.cameraBuffer,
      ...this.extraBuffers,
      ...this.adamUni,
    ]) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }

  private viewIndex(view: number): number {
    return Math.max(0, Math.min(this.cameras.length - 1, view | 0));
  }

  private imageByteSize(): number {
    return 3 * this.dims.H * this.dims.W * 4;
  }

  private sharedScratchState(): Raster3DScratchState {
    return {
      derived: this.derived,
      accGrad: this.accGrad,
      tileCounts: this.tileCounts,
      binnedIds: this.binnedIds,
      tileStop: this.tileStop,
      prepBind: this.prepBind,
      chainBind: this.chainBind,
      emitBind: this.emitBind,
      clearBinsBind: this.clearBinsBind,
      clearGradsBind: this.clearGradsBind,
    };
  }

  private createPrivateScratchState(): Raster3DScratchState {
    const { derived, tileCounts, binnedIds, tileStop } = this.createRawScratchBuffers(1);
    this.extraBuffers.push(derived, tileCounts, binnedIds, tileStop);
    return {
      derived,
      accGrad: this.accGrad,
      tileCounts,
      binnedIds,
      tileStop,
      prepBind: this.prepPipe.map((pipe) => this.bindGroup(pipe, [this.params, derived])),
      chainBind: this.chainPipe.map((pipe) => this.bindGroup(pipe, [this.accGrad, derived, this.params, this.gradRaw])),
      emitBind: this.bindGroup(this.emitPipe, [derived, tileCounts, binnedIds]),
      clearBinsBind: this.bindGroup(this.clearBinsPipe, [tileCounts]),
      clearGradsBind: this.clearGradsBind,
    };
  }

  private createRawScratchBuffers(lanes: number): Raster3DRawScratchBuffers {
    const d = this.dims;
    return {
      derived: this.storage(d.G * DERIVED_STRIDE_3D * lanes),
      tileCounts: this.storage(d.numTiles * lanes),
      binnedIds: this.storage(d.numTiles * d.cap * lanes),
      tileStop: this.storage(d.numTiles * lanes),
    };
  }

  private laneScratchState(raw: Raster3DRawScratchBuffers, lane: number, accGradBuffer: GPUBuffer): Raster3DScratchState {
    const d = this.dims;
    const derivedBytes = d.G * DERIVED_STRIDE_3D * 4;
    const tileBytes = d.numTiles * 4;
    const binnedBytes = d.numTiles * d.cap * 4;
    const derived = this.sliceBinding(raw.derived, lane * derivedBytes, derivedBytes);
    const accGrad = this.sliceBinding(accGradBuffer, lane * derivedBytes, derivedBytes);
    const tileCounts = this.sliceBinding(raw.tileCounts, lane * tileBytes, tileBytes);
    const binnedIds = this.sliceBinding(raw.binnedIds, lane * binnedBytes, binnedBytes);
    const tileStop = this.sliceBinding(raw.tileStop, lane * tileBytes, tileBytes);
    return {
      derived,
      accGrad,
      tileCounts,
      binnedIds,
      tileStop,
      prepBind: this.prepPipe.map((pipe) => this.bindGroup(pipe, [this.params, derived])),
      chainBind: this.chainPipe.map((pipe) => this.bindGroup(pipe, [accGrad, derived, this.params, this.gradRaw])),
      emitBind: this.bindGroup(this.emitPipe, [derived, tileCounts, binnedIds]),
      clearBinsBind: this.bindGroup(this.clearBinsPipe, [tileCounts]),
      clearGradsBind: this.bindGroup(this.clearGradsPipe, [accGrad]),
    };
  }

  private createIOStateForScratch(
    scratch: Raster3DScratchState,
    imageBuffer: GPUBuffer,
    imageOffset: number,
    gradBuffer: GPUBuffer,
    gradOffset: number
  ): Raster3DIOState {
    return {
      prepBind: scratch.prepBind,
      chainBind: scratch.chainBind,
      emitBind: scratch.emitBind,
      clearBinsBind: scratch.clearBinsBind,
      clearGradsBind: scratch.clearGradsBind,
      fwdBind: this.makeForwardBind(scratch, imageBuffer, imageOffset),
      bwdBind: this.makeBackwardBind(scratch, gradBuffer, gradOffset),
    };
  }

  private makeForwardBind(scratch: Raster3DScratchState, imageBuffer: GPUBuffer, imageOffset: number): GPUBindGroup {
    return this.bindGroup(this.fwdPipe, [
      scratch.tileCounts,
      scratch.binnedIds,
      scratch.derived,
      { buffer: imageBuffer, offset: imageOffset, size: this.imageByteSize() },
      scratch.tileStop,
    ]);
  }

  private makeBackwardBind(scratch: Raster3DScratchState, gradBuffer: GPUBuffer, gradOffset: number): GPUBindGroup {
    return this.bindGroup(this.bwdPipe, [
      { buffer: gradBuffer, offset: gradOffset, size: this.imageByteSize() },
      scratch.tileCounts,
      scratch.binnedIds,
      scratch.tileStop,
      scratch.derived,
      scratch.accGrad,
    ]);
  }

  private checkIOBinding(name: string, offset: number): void {
    if (!Number.isInteger(offset) || offset < 0 || offset % STORAGE_OFFSET_ALIGN !== 0) {
      throw new Error(`raster3d: ${name} offset ${offset} must be ${STORAGE_OFFSET_ALIGN}-byte aligned`);
    }
  }

  private checkContiguousImageOffsets(name: string, offsets: number[]): void {
    if (!offsets.length) throw new Error(`raster3d: empty ${name} offsets`);
    const stride = this.imageByteSize();
    for (let i = 0; i < offsets.length; i++) {
      this.checkIOBinding(name, offsets[i]);
      if (offsets[i] !== offsets[0] + i * stride) {
        throw new Error(`raster3d: ${name} offsets must be contiguous image lanes`);
      }
    }
  }

  private sliceBinding(buffer: GPUBuffer, offset: number, size: number): BufferBindingInput {
    this.checkIOBinding("scratch", offset);
    return { buffer, offset, size };
  }
}

function serializeCameras3D(cameras: PreparedCamera3D[]): Float32Array {
  const data = new Float32Array(cameras.length * CAMERA_STRIDE_3D);
  for (let i = 0; i < cameras.length; i++) {
    const cam = cameras[i];
    const o = i * CAMERA_STRIDE_3D;
    data[o + 0] = cam.eye[0];
    data[o + 1] = cam.eye[1];
    data[o + 2] = cam.eye[2];
    data[o + 3] = cam.right[0];
    data[o + 4] = cam.right[1];
    data[o + 5] = cam.right[2];
    data[o + 6] = cam.cameraUp[0];
    data[o + 7] = cam.cameraUp[1];
    data[o + 8] = cam.cameraUp[2];
    data[o + 9] = cam.forward[0];
    data[o + 10] = cam.forward[1];
    data[o + 11] = cam.forward[2];
    data[o + 12] = cam.focalPx;
  }
  return data;
}
