import { ADAM_UNIFORM_BYTES, DEFAULT_HYPER, adamShader, type AdamHyper } from "../splat/adam_wgsl";
import type { PreparedCamera3D } from "../splat3d/cameras";
import { ANISO_PARAM_STRIDE_3D, anisotropicParamSegments3D } from "./layout";
import {
  ANISO_DERIVED_STRIDE_3D,
  ANISO_DENSITY_STAT_SCALE_3D,
  ANISO_REGULARIZER_UNIFORM_BYTES_3D,
  anisotropicBackwardShader3D,
  anisotropicCenterReduceShader3D,
  anisotropicChainShader3D,
  anisotropicClearShader3D,
  anisotropicEmitShader3D,
  anisotropicForwardShader3D,
  anisotropicPrepShader3D,
  anisotropicRegularizerShader3D,
  resolveAnisotropicRaster3DDims,
  type AnisotropicRaster3DConfig,
  type AnisotropicRaster3DDims,
} from "./raster_wgsl";

const U = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const WORKGROUP_SIZE = 256;
const ceilGroups = (items: number): number => Math.ceil(items / WORKGROUP_SIZE);

export interface AnisotropicRaster3DEngineConfig extends AnisotropicRaster3DConfig {
  cameras: PreparedCamera3D[];
  sharedParams?: GPUBuffer;
  sharedGradRaw?: GPUBuffer;
}

export interface AnisotropicAdamLRs3D {
  position: number;
  logScale: number;
  quaternion: number;
  color: number;
  opacity: number;
}

export interface AnisotropicRaster3DRegularizerOptions {
  centerWeight: number;
  radiusWeight: number;
  targetRadius: number;
  opacitySparsity: number;
  smallRadiusWeight: number;
  smallRadius: number;
  radiusBandWeight: number;
  minRadius: number;
  maxRadius: number;
}

export interface AnisotropicTileTelemetry3D {
  cap: number;
  maxCount: number;
  maxStop: number;
  overflowTiles: number;
  overflowPairs: number;
  totalPairs: number;
}

/** Adaptation-only statistics collected by a sampled raster backward pass. */
export interface AnisotropicDensityStats3D {
  /** Sum of per-pixel absolute screen-centre gradients, restored to f32 units. */
  absScreenGradient: Float32Array;
  /** Visible alpha-contributing pixels per splat. */
  visiblePixels: Uint32Array;
}

export const DEFAULT_ANISOTROPIC_3D_LRS: AnisotropicAdamLRs3D = {
  position: 0.025,
  logScale: 0.01,
  quaternion: 0.005,
  color: 0.08,
  opacity: 0.03,
};

async function makeCompute(device: GPUDevice, code: string, label: string): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code, label: `${label}-module` });
  const pipeline = device.createComputePipeline({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const error = await device.popErrorScope();
  if (error) {
    console.error(`--- anisotropic WGSL failure (${label}) ---\n${code}`);
    throw new Error(`anisotropic raster pipeline validation (${label}): ${(error as GPUValidationError).message}`);
  }
  return pipeline;
}

export class AnisotropicRaster3DEngine {
  readonly dims: AnisotropicRaster3DDims;
  readonly cameras: PreparedCamera3D[];

  readonly params: GPUBuffer;
  readonly derived: GPUBuffer;
  readonly accGrad: GPUBuffer;
  readonly gradRaw: GPUBuffer;
  readonly mBuf: GPUBuffer;
  readonly vBuf: GPUBuffer;
  readonly tileCounts: GPUBuffer;
  readonly binnedIds: GPUBuffer;
  readonly tileStop: GPUBuffer;
  readonly densityStats: GPUBuffer;
  readonly image: GPUBuffer;
  readonly gradImage: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly ownsParams: boolean;
  private readonly ownsGradRaw: boolean;
  private prepPipes: GPUComputePipeline[] = [];
  private chainPipes: GPUComputePipeline[] = [];
  private emitPipe!: GPUComputePipeline;
  private forwardPipe!: GPUComputePipeline;
  private backwardPipe!: GPUComputePipeline;
  private densityBackwardPipe!: GPUComputePipeline;
  private clearBinsPipe!: GPUComputePipeline;
  private clearAccGradPipe!: GPUComputePipeline;
  private clearRawGradPipe!: GPUComputePipeline;
  private clearDensityStatsPipe!: GPUComputePipeline;
  private adamPipe!: GPUComputePipeline;
  private centerReducePipe!: GPUComputePipeline;
  private regularizerPipe!: GPUComputePipeline;
  private prepBinds: GPUBindGroup[] = [];
  private chainBinds: GPUBindGroup[] = [];
  private emitBind!: GPUBindGroup;
  private forwardBind!: GPUBindGroup;
  private backwardBind!: GPUBindGroup;
  private densityBackwardBind!: GPUBindGroup;
  private clearBinsBind!: GPUBindGroup;
  private clearAccGradBind!: GPUBindGroup;
  private clearRawGradBind!: GPUBindGroup;
  private clearDensityStatsBind!: GPUBindGroup;
  private adamUniforms: GPUBuffer[] = [];
  private adamBinds: GPUBindGroup[] = [];
  private centerSum!: GPUBuffer;
  private centerReduceBind!: GPUBindGroup;
  private regularizerUniform!: GPUBuffer;
  private regularizerBind!: GPUBindGroup;
  private destroyed = false;

  private constructor(device: GPUDevice, cfg: AnisotropicRaster3DEngineConfig) {
    this.device = device;
    this.dims = resolveAnisotropicRaster3DDims(cfg);
    this.cameras = cfg.cameras;
    if (this.cameras.length === 0) throw new Error("splat3d_aniso_raster: at least one camera is required");

    const paramWords = this.dims.G * ANISO_PARAM_STRIDE_3D;
    const derivedWords = this.dims.G * ANISO_DERIVED_STRIDE_3D;
    this.params = cfg.sharedParams ?? this.storage(paramWords, U.COPY_SRC | U.COPY_DST, "aniso-params");
    this.gradRaw = cfg.sharedGradRaw ?? this.storage(paramWords, U.COPY_SRC | U.COPY_DST, "aniso-grad-raw");
    this.ownsParams = !cfg.sharedParams;
    this.ownsGradRaw = !cfg.sharedGradRaw;
    this.derived = this.storage(derivedWords, 0, "aniso-derived");
    this.accGrad = this.storage(derivedWords, U.COPY_DST, "aniso-acc-grad");
    this.mBuf = this.storage(paramWords, U.COPY_SRC | U.COPY_DST, "aniso-adam-m");
    this.vBuf = this.storage(paramWords, U.COPY_SRC | U.COPY_DST, "aniso-adam-v");
    this.tileCounts = this.storage(this.dims.numTiles, U.COPY_DST | U.COPY_SRC, "aniso-tile-counts");
    this.binnedIds = this.storage(this.dims.numTiles * this.dims.cap, 0, "aniso-binned-ids");
    this.tileStop = this.storage(this.dims.numTiles, U.COPY_SRC, "aniso-tile-stop");
    this.densityStats = this.storage(3 * this.dims.G, U.COPY_SRC, "aniso-density-stats");
    this.image = this.storage(3 * this.dims.H * this.dims.W, U.COPY_SRC, "aniso-image");
    this.gradImage = this.storage(3 * this.dims.H * this.dims.W, U.COPY_DST, "aniso-grad-image");
  }

  static async create(device: GPUDevice, cfg: AnisotropicRaster3DEngineConfig): Promise<AnisotropicRaster3DEngine> {
    const engine = new AnisotropicRaster3DEngine(device, cfg);
    await engine.build(cfg);
    engine.clearDensityStats();
    return engine;
  }

  private storage(words: number, extra: number, label: string): GPUBuffer {
    return this.device.createBuffer({ label, size: words * 4, usage: U.STORAGE | extra });
  }

  private bindGroup(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  private async build(cfg: AnisotropicRaster3DEngineConfig): Promise<void> {
    const d = this.dims;
    this.prepPipes = await Promise.all(
      this.cameras.map((camera, index) => makeCompute(this.device, anisotropicPrepShader3D(cfg, camera), `aniso-prep-${index}`))
    );
    this.chainPipes = await Promise.all(
      this.cameras.map((camera, index) => makeCompute(this.device, anisotropicChainShader3D(cfg, camera), `aniso-chain-${index}`))
    );
    this.emitPipe = await makeCompute(this.device, anisotropicEmitShader3D(cfg), "aniso-emit");
    this.forwardPipe = await makeCompute(this.device, anisotropicForwardShader3D(cfg), "aniso-forward");
    this.backwardPipe = await makeCompute(this.device, anisotropicBackwardShader3D(cfg), "aniso-backward");
    this.densityBackwardPipe = await makeCompute(
      this.device,
      anisotropicBackwardShader3D(cfg, true),
      "aniso-density-backward"
    );
    this.clearBinsPipe = await makeCompute(this.device, anisotropicClearShader3D(d.numTiles), "aniso-clear-bins");
    this.clearAccGradPipe = await makeCompute(
      this.device,
      anisotropicClearShader3D(d.G * ANISO_DERIVED_STRIDE_3D),
      "aniso-clear-acc-grad"
    );
    this.clearRawGradPipe = await makeCompute(
      this.device,
      anisotropicClearShader3D(d.G * ANISO_PARAM_STRIDE_3D),
      "aniso-clear-raw-grad"
    );
    this.clearDensityStatsPipe = await makeCompute(
      this.device,
      anisotropicClearShader3D(3 * d.G),
      "aniso-clear-density-stats"
    );
    this.adamPipe = await makeCompute(this.device, adamShader(), "aniso-adam");
    this.centerReducePipe = await makeCompute(
      this.device,
      anisotropicCenterReduceShader3D(cfg),
      "aniso-center-reduce"
    );
    this.regularizerPipe = await makeCompute(
      this.device,
      anisotropicRegularizerShader3D(cfg),
      "aniso-regularizer"
    );

    this.prepBinds = this.prepPipes.map((pipeline) => this.bindGroup(pipeline, [this.params, this.derived]));
    this.chainBinds = this.chainPipes.map((pipeline) =>
      this.bindGroup(pipeline, [this.accGrad, this.derived, this.params, this.gradRaw])
    );
    this.emitBind = this.bindGroup(this.emitPipe, [this.derived, this.tileCounts, this.binnedIds]);
    this.forwardBind = this.bindGroup(this.forwardPipe, [
      this.tileCounts,
      this.binnedIds,
      this.derived,
      this.image,
      this.tileStop,
    ]);
    this.backwardBind = this.bindGroup(this.backwardPipe, [
      this.gradImage,
      this.tileCounts,
      this.binnedIds,
      this.tileStop,
      this.derived,
      this.accGrad,
    ]);
    this.densityBackwardBind = this.bindGroup(this.densityBackwardPipe, [
      this.gradImage,
      this.tileCounts,
      this.binnedIds,
      this.tileStop,
      this.derived,
      this.accGrad,
      this.densityStats,
    ]);
    this.clearBinsBind = this.bindGroup(this.clearBinsPipe, [this.tileCounts]);
    this.clearAccGradBind = this.bindGroup(this.clearAccGradPipe, [this.accGrad]);
    this.clearRawGradBind = this.bindGroup(this.clearRawGradPipe, [this.gradRaw]);
    this.clearDensityStatsBind = this.bindGroup(this.clearDensityStatsPipe, [this.densityStats]);
    this.centerSum = this.storage(4, U.COPY_DST, "aniso-center-sum");
    this.centerReduceBind = this.bindGroup(this.centerReducePipe, [this.params, this.centerSum]);
    this.regularizerUniform = this.device.createBuffer({
      label: "aniso-regularizer-uniform",
      size: ANISO_REGULARIZER_UNIFORM_BYTES_3D,
      usage: U.UNIFORM | U.COPY_DST,
    });
    this.regularizerBind = this.bindGroup(this.regularizerPipe, [
      this.regularizerUniform,
      this.params,
      this.gradRaw,
      this.centerSum,
    ]);

    for (const _segment of anisotropicParamSegments3D(d.G)) {
      const uniform = this.device.createBuffer({
        label: "aniso-adam-uniform",
        size: ADAM_UNIFORM_BYTES,
        usage: 64 | U.COPY_DST,
      });
      this.adamUniforms.push(uniform);
      this.adamBinds.push(
        this.device.createBindGroup({
          layout: this.adamPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniform } },
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
    if (data.length !== this.dims.G * ANISO_PARAM_STRIDE_3D) {
      throw new Error(`splat3d_aniso_raster: expected ${this.dims.G * ANISO_PARAM_STRIDE_3D} params, got ${data.length}`);
    }
    this.device.queue.writeBuffer(this.params, 0, data as unknown as BufferSource);
  }

  zeroAdamState(): void {
    const zeros = new Float32Array(this.dims.G * ANISO_PARAM_STRIDE_3D);
    this.device.queue.writeBuffer(this.mBuf, 0, zeros as unknown as BufferSource);
    this.device.queue.writeBuffer(this.vBuf, 0, zeros as unknown as BufferSource);
  }

  resetAdamForSplats(indices: ArrayLike<number>): void {
    const G = this.dims.G;
    const segments = anisotropicParamSegments3D(G);
    for (let i = 0; i < indices.length; i++) {
      const g = indices[i] | 0;
      if (g < 0 || g >= G) continue;
      for (const segment of segments) {
        const zeros = new Float32Array(segment.components);
        const wordOffset = segment.offset + segment.components * g;
        this.device.queue.writeBuffer(this.mBuf, wordOffset * 4, zeros);
        this.device.queue.writeBuffer(this.vBuf, wordOffset * 4, zeros);
      }
    }
  }

  recordForward(enc: GPUCommandEncoder, view = 0): void {
    const index = this.viewIndex(view);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.prepPipes[index]);
    pass.setBindGroup(0, this.prepBinds[index]);
    pass.dispatchWorkgroups(ceilGroups(this.dims.G));
    pass.setPipeline(this.clearBinsPipe);
    pass.setBindGroup(0, this.clearBinsBind);
    pass.dispatchWorkgroups(ceilGroups(this.dims.numTiles));
    pass.setPipeline(this.emitPipe);
    pass.setBindGroup(0, this.emitBind);
    pass.dispatchWorkgroups(ceilGroups(this.dims.G));
    pass.setPipeline(this.forwardPipe);
    pass.setBindGroup(0, this.forwardBind);
    pass.dispatchWorkgroups(this.dims.numTiles);
    pass.end();
  }

  recordClearRawGrad(enc: GPUCommandEncoder): void {
    const pass = enc.beginComputePass();
    pass.setPipeline(this.clearRawGradPipe);
    pass.setBindGroup(0, this.clearRawGradBind);
    pass.dispatchWorkgroups(ceilGroups(this.dims.G * ANISO_PARAM_STRIDE_3D));
    pass.end();
  }

  recordBackwardAdd(enc: GPUCommandEncoder, view = 0, collectDensityStats = false): void {
    const index = this.viewIndex(view);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.clearAccGradPipe);
    pass.setBindGroup(0, this.clearAccGradBind);
    pass.dispatchWorkgroups(ceilGroups(this.dims.G * ANISO_DERIVED_STRIDE_3D));
    pass.setPipeline(collectDensityStats ? this.densityBackwardPipe : this.backwardPipe);
    pass.setBindGroup(0, collectDensityStats ? this.densityBackwardBind : this.backwardBind);
    pass.dispatchWorkgroups(this.dims.numTiles);
    pass.setPipeline(this.chainPipes[index]);
    pass.setBindGroup(0, this.chainBinds[index]);
    pass.dispatchWorkgroups(ceilGroups(this.dims.G));
    pass.end();
  }

  clearDensityStats(): void {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.clearDensityStatsPipe);
    pass.setBindGroup(0, this.clearDensityStatsBind);
    pass.dispatchWorkgroups(ceilGroups(3 * this.dims.G));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  recordRegularizerAdd(enc: GPUCommandEncoder, opts: AnisotropicRaster3DRegularizerOptions): void {
    if (!anisotropicRegularizerEnabled(opts)) return;
    const data = new Float32Array(16);
    data[0] = opts.centerWeight;
    data[1] = opts.radiusWeight;
    data[2] = opts.targetRadius;
    data[3] = opts.opacitySparsity;
    data[4] = opts.smallRadiusWeight;
    data[5] = opts.smallRadius;
    data[6] = opts.radiusBandWeight;
    data[7] = opts.minRadius;
    data[8] = opts.maxRadius;
    this.device.queue.writeBuffer(this.regularizerUniform, 0, data);
    if (opts.centerWeight !== 0) enc.clearBuffer(this.centerSum, 0, 16);
    const pass = enc.beginComputePass();
    if (opts.centerWeight !== 0) {
      pass.setPipeline(this.centerReducePipe);
      pass.setBindGroup(0, this.centerReduceBind);
      pass.dispatchWorkgroups(ceilGroups(this.dims.G));
    }
    pass.setPipeline(this.regularizerPipe);
    pass.setBindGroup(0, this.regularizerBind);
    pass.dispatchWorkgroups(ceilGroups(this.dims.G));
    pass.end();
  }

  recordAdam(
    enc: GPUCommandEncoder,
    step: number,
    lrs: AnisotropicAdamLRs3D = DEFAULT_ANISOTROPIC_3D_LRS,
    hyper: AdamHyper = DEFAULT_HYPER
  ): void {
    const segments = anisotropicParamSegments3D(this.dims.G);
    const learningRate: Record<string, number> = {
      position: lrs.position,
      logScale: lrs.logScale,
      quaternion: lrs.quaternion,
      color: lrs.color,
      opacity: lrs.opacity,
    };
    const t = Math.max(1, step);
    const bc1 = 1 - Math.pow(hyper.beta1, t);
    const bc2 = 1 - Math.pow(hyper.beta2, t);
    segments.forEach((segment, index) => {
      const bytes = new ArrayBuffer(ADAM_UNIFORM_BYTES);
      const u32 = new Uint32Array(bytes);
      const f32 = new Float32Array(bytes);
      u32[0] = segment.offset;
      u32[1] = segment.length;
      f32[2] = learningRate[segment.name];
      f32[3] = hyper.beta1;
      f32[4] = hyper.beta2;
      f32[5] = hyper.eps;
      f32[6] = bc1;
      f32[7] = bc2;
      this.device.queue.writeBuffer(this.adamUniforms[index], 0, bytes);
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.adamPipe);
    segments.forEach((segment, index) => {
      pass.setBindGroup(0, this.adamBinds[index]);
      pass.dispatchWorkgroups(ceilGroups(segment.length));
    });
    pass.end();
  }

  runForward(view = 0): void {
    const enc = this.device.createCommandEncoder();
    this.recordForward(enc, view);
    this.device.queue.submit([enc.finish()]);
  }

  readImage(): Promise<Float32Array> {
    return this.readFloats(this.image, 3 * this.dims.H * this.dims.W);
  }

  readParams(): Promise<Float32Array> {
    return this.readFloats(this.params, this.dims.G * ANISO_PARAM_STRIDE_3D);
  }

  readRawGrad(): Promise<Float32Array> {
    return this.readFloats(this.gradRaw, this.dims.G * ANISO_PARAM_STRIDE_3D);
  }

  async readDensityStats(): Promise<AnisotropicDensityStats3D> {
    const values = await this.readU32(this.densityStats, 3 * this.dims.G);
    const absScreenGradient = new Float32Array(this.dims.G);
    const visiblePixels = new Uint32Array(this.dims.G);
    for (let g = 0; g < this.dims.G; g++) {
      const base = 3 * g;
      absScreenGradient[g] = Math.hypot(values[base], values[base + 1]) / ANISO_DENSITY_STAT_SCALE_3D;
      visiblePixels[g] = values[base + 2];
    }
    return { absScreenGradient, visiblePixels };
  }

  async readTileTelemetry(): Promise<AnisotropicTileTelemetry3D> {
    const words = this.dims.numTiles;
    const bytes = words * 4;
    const staging = this.device.createBuffer({ size: bytes * 2, usage: U.MAP_READ | U.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.tileCounts, 0, staging, 0, bytes);
    enc.copyBufferToBuffer(this.tileStop, 0, staging, bytes, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const mapped = staging.getMappedRange();
    const counts = new Uint32Array(mapped.slice(0, bytes));
    const stops = new Uint32Array(mapped.slice(bytes, bytes * 2));
    staging.unmap();
    staging.destroy();
    let maxCount = 0;
    let maxStop = 0;
    let overflowTiles = 0;
    let overflowPairs = 0;
    let totalPairs = 0;
    for (let i = 0; i < words; i++) {
      maxCount = Math.max(maxCount, counts[i]);
      maxStop = Math.max(maxStop, stops[i]);
      totalPairs += counts[i];
      if (counts[i] > this.dims.cap) {
        overflowTiles++;
        overflowPairs += counts[i] - this.dims.cap;
      }
    }
    return { cap: this.dims.cap, maxCount, maxStop, overflowTiles, overflowPairs, totalPairs };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const buffers = [
      this.derived,
      this.accGrad,
      this.mBuf,
      this.vBuf,
      this.tileCounts,
      this.binnedIds,
      this.tileStop,
      this.densityStats,
      this.image,
      this.gradImage,
      this.centerSum,
      this.regularizerUniform,
      ...this.adamUniforms,
    ];
    if (this.ownsParams) buffers.push(this.params);
    if (this.ownsGradRaw) buffers.push(this.gradRaw);
    for (const buffer of buffers) buffer.destroy();
  }

  private viewIndex(view: number): number {
    return Math.max(0, Math.min(this.cameras.length - 1, view | 0));
  }

  private async readFloats(buffer: GPUBuffer, words: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({ size: words * 4, usage: U.MAP_READ | U.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, words * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const values = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return values;
  }

  private async readU32(buffer: GPUBuffer, words: number): Promise<Uint32Array> {
    const staging = this.device.createBuffer({ size: words * 4, usage: U.MAP_READ | U.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, words * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const values = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return values;
  }
}

function anisotropicRegularizerEnabled(opts: AnisotropicRaster3DRegularizerOptions): boolean {
  return (
    opts.centerWeight !== 0 ||
    opts.radiusWeight !== 0 ||
    opts.opacitySparsity !== 0 ||
    opts.smallRadiusWeight !== 0 ||
    opts.radiusBandWeight !== 0
  );
}
