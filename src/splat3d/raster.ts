import { DEFAULT_HYPER, type AdamHyper, adamShader, ADAM_UNIFORM_BYTES } from "../splat/adam_wgsl";
import type { PreparedCamera3D } from "./cameras";
import {
  Raster3DConfig,
  Raster3DDims,
  resolveDims3D,
  prepShader3D,
  emitShader3D,
  forwardShader3D,
  backwardShader3D,
  chainAddShader3D,
  clearShader3D,
  paramSegments3D,
  PARAM_STRIDE_3D,
  DERIVED_STRIDE_3D,
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
  fwdBind: GPUBindGroup;
  bwdBind: GPUBindGroup;
}

export interface Raster3DIOOptions {
  privateState?: boolean;
}

interface Raster3DScratchState {
  derived: GPUBuffer;
  tileCounts: GPUBuffer;
  binnedIds: GPUBuffer;
  tileStop: GPUBuffer;
  prepBind: GPUBindGroup[];
  chainBind: GPUBindGroup[];
  emitBind: GPUBindGroup;
  clearBinsBind: GPUBindGroup;
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
    return {
      prepBind: scratch.prepBind,
      chainBind: scratch.chainBind,
      emitBind: scratch.emitBind,
      clearBinsBind: scratch.clearBinsBind,
      fwdBind: this.makeForwardBind(scratch, imageBuffer, imageOffset),
      bwdBind: this.makeBackwardBind(scratch, gradBuffer, gradOffset),
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
    const d = this.dims;
    const v = this.viewIndex(view);
    const p = enc.beginComputePass();
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
    p.end();
  }

  recordBackwardAdd(enc: GPUCommandEncoder, view = 0, io?: Raster3DIOState): void {
    const d = this.dims;
    const v = this.viewIndex(view);
    const p = enc.beginComputePass();
    p.setPipeline(this.clearGradsPipe);
    p.setBindGroup(0, this.clearGradsBind);
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
      tileCounts: this.tileCounts,
      binnedIds: this.binnedIds,
      tileStop: this.tileStop,
      prepBind: this.prepBind,
      chainBind: this.chainBind,
      emitBind: this.emitBind,
      clearBinsBind: this.clearBinsBind,
    };
  }

  private createPrivateScratchState(): Raster3DScratchState {
    const d = this.dims;
    const derived = this.storage(d.G * DERIVED_STRIDE_3D);
    const tileCounts = this.storage(d.numTiles);
    const binnedIds = this.storage(d.numTiles * d.cap);
    const tileStop = this.storage(d.numTiles);
    this.extraBuffers.push(derived, tileCounts, binnedIds, tileStop);
    return {
      derived,
      tileCounts,
      binnedIds,
      tileStop,
      prepBind: this.prepPipe.map((pipe) => this.bindGroup(pipe, [this.params, derived])),
      chainBind: this.chainPipe.map((pipe) => this.bindGroup(pipe, [this.accGrad, derived, this.params, this.gradRaw])),
      emitBind: this.bindGroup(this.emitPipe, [derived, tileCounts, binnedIds]),
      clearBinsBind: this.bindGroup(this.clearBinsPipe, [tileCounts]),
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
      this.accGrad,
    ]);
  }

  private checkIOBinding(name: string, offset: number): void {
    if (!Number.isInteger(offset) || offset < 0 || offset % STORAGE_OFFSET_ALIGN !== 0) {
      throw new Error(`raster3d: ${name} offset ${offset} must be ${STORAGE_OFFSET_ALIGN}-byte aligned`);
    }
  }
}
