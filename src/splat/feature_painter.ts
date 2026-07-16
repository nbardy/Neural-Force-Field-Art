/** Runtime owner for the fused compact 2D Feature8 painter. */
/// <reference types="@webgpu/types" />
import {
  ADAM_UNIFORM_BYTES,
  adamShader,
  type AdamHyper,
  type AdamLRs,
  DEFAULT_HYPER,
} from "./adam_wgsl";
import {
  clearShader,
  PARAM_STRIDE,
  resolveDims,
  type RasterConfig,
  type RasterDims,
} from "./raster_wgsl";
import {
  DECODER_PARAM_COUNT,
  FEATURE_ACC_STRIDE,
  FEATURE_CHAIN_WORK_ITEMS,
  FEATURE_STATE_STRIDE,
  FEATURE_STRIDE,
  featureBackwardShader,
  featureChainShader,
  featureEmitShader,
  featureForwardShader,
  featureGeometryChainShader,
  featurePrepShader,
} from "./feature_painter_wgsl";

const U = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const WORKGROUP_SIZE = 256;
const workgroups = (count: number): number => Math.ceil(count / WORKGROUP_SIZE);
const FEATURE_LR = 0.025;
const DECODER_LR = 0.03;

type Segment = { offset: number; length: number; lr: number };

async function compute(device: GPUDevice, code: string, label: string): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ label, code });
  const pipeline = device.createComputePipeline({ label, layout: "auto", compute: { module, entryPoint: "main" } });
  const error = await device.popErrorScope();
  if (error) throw new Error(`feature painter ${label}: ${(error as GPUValidationError).message}`);
  return pipeline;
}

function adamUniform(device: GPUDevice): GPUBuffer {
  return device.createBuffer({ size: ADAM_UNIFORM_BYTES, usage: U.UNIFORM | U.COPY_DST });
}

function writeAdam(
  device: GPUDevice,
  buffer: GPUBuffer,
  offset: number,
  count: number,
  lr: number,
  step: number,
  hyper: AdamHyper,
): void {
  const raw = new ArrayBuffer(32);
  const u32 = new Uint32Array(raw);
  const f32 = new Float32Array(raw);
  u32[0] = offset;
  u32[1] = count;
  f32[2] = lr;
  f32[3] = hyper.beta1;
  f32[4] = hyper.beta2;
  f32[5] = hyper.eps;
  f32[6] = 1 - Math.pow(hyper.beta1, step);
  f32[7] = 1 - Math.pow(hyper.beta2, step);
  device.queue.writeBuffer(buffer, 0, raw);
}

export interface FeaturePainterConfig extends RasterConfig {}

/**
 * Compact feature rasterizer with exact RGB-skip initialization.
 *
 * `params` owns normal splat geometry/RGB logits. `featureParams` owns only
 * z/Ax/Ay, and `decoderParams` owns a 3x8 residual projection. All image-space
 * feature work happens in registers inside the tile shaders.
 */
export class FeaturePainterEngine {
  readonly dims: RasterDims;
  readonly params: GPUBuffer;
  readonly featureParams: GPUBuffer;
  readonly decoderParams: GPUBuffer;
  readonly image: GPUBuffer;
  readonly gradImage: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly state: GPUBuffer;
  private readonly tileCounts: GPUBuffer;
  private readonly binnedIds: GPUBuffer;
  private readonly tileStop: GPUBuffer;
  private readonly acc: GPUBuffer;
  private readonly gradGeom: GPUBuffer;
  private readonly geomM: GPUBuffer;
  private readonly geomV: GPUBuffer;
  private readonly gradFeature: GPUBuffer;
  private readonly featureM: GPUBuffer;
  private readonly featureV: GPUBuffer;
  private readonly gradDecoder: GPUBuffer;
  private readonly decoderM: GPUBuffer;
  private readonly decoderV: GPUBuffer;

  private prepPipe!: GPUComputePipeline;
  private emitPipe!: GPUComputePipeline;
  private forwardPipe!: GPUComputePipeline;
  private backwardPipe!: GPUComputePipeline;
  private geometryChainPipe!: GPUComputePipeline;
  private featureChainPipe!: GPUComputePipeline;
  private clearBinsPipe!: GPUComputePipeline;
  private clearAccPipe!: GPUComputePipeline;
  private adamPipe!: GPUComputePipeline;

  private prepBind!: GPUBindGroup;
  private emitBind!: GPUBindGroup;
  private forwardBind!: GPUBindGroup;
  private backwardBind!: GPUBindGroup;
  private geometryChainBind!: GPUBindGroup;
  private featureChainBind!: GPUBindGroup;
  private clearBinsBind!: GPUBindGroup;
  private clearAccBind!: GPUBindGroup;
  private readonly geometryAdamUniforms: GPUBuffer[] = [];
  private readonly geometryAdamGroups: GPUBindGroup[] = [];
  private readonly featureAdamUniform: GPUBuffer;
  private featureAdamGroup!: GPUBindGroup;
  private readonly decoderAdamUniform: GPUBuffer;
  private decoderAdamGroup!: GPUBindGroup;
  private readonly geometrySegments: Segment[];

  private constructor(device: GPUDevice, cfg: FeaturePainterConfig) {
    this.device = device;
    this.dims = resolveDims(cfg);
    const d = this.dims;
    const storage = (count: number, extra = 0): GPUBuffer =>
      device.createBuffer({ size: count * 4, usage: U.STORAGE | extra });
    const geomCount = d.G * PARAM_STRIDE;
    const featureCount = d.G * FEATURE_STRIDE;
    const accCount = d.G * FEATURE_ACC_STRIDE + DECODER_PARAM_COUNT;

    this.params = storage(geomCount, U.COPY_SRC | U.COPY_DST);
    this.featureParams = storage(featureCount, U.COPY_SRC | U.COPY_DST);
    this.decoderParams = storage(DECODER_PARAM_COUNT, U.COPY_SRC | U.COPY_DST);
    this.image = storage(3 * d.H * d.W, U.COPY_SRC);
    this.gradImage = storage(3 * d.H * d.W, U.COPY_DST);
    this.state = storage(d.G * FEATURE_STATE_STRIDE);
    this.tileCounts = storage(d.numTiles, U.COPY_DST);
    this.binnedIds = storage(d.numTiles * d.cap);
    this.tileStop = storage(d.numTiles);
    this.acc = storage(accCount, U.COPY_DST);
    this.gradGeom = storage(geomCount, U.COPY_SRC);
    this.geomM = storage(geomCount, U.COPY_DST);
    this.geomV = storage(geomCount, U.COPY_DST);
    this.gradFeature = storage(featureCount, U.COPY_SRC);
    this.featureM = storage(featureCount, U.COPY_DST);
    this.featureV = storage(featureCount, U.COPY_DST);
    this.gradDecoder = storage(DECODER_PARAM_COUNT, U.COPY_SRC);
    this.decoderM = storage(DECODER_PARAM_COUNT, U.COPY_DST);
    this.decoderV = storage(DECODER_PARAM_COUNT, U.COPY_DST);

    this.featureAdamUniform = adamUniform(device);
    this.decoderAdamUniform = adamUniform(device);
    this.geometrySegments = [
      { offset: 0, length: 2 * d.G, lr: 0 },
      { offset: 2 * d.G, length: 2 * d.G, lr: 0 },
      { offset: 4 * d.G, length: d.G, lr: 0 },
      { offset: 5 * d.G, length: 3 * d.G, lr: 0 },
      { offset: 8 * d.G, length: d.G, lr: 0 },
    ];
  }

  static async create(device: GPUDevice, cfg: FeaturePainterConfig): Promise<FeaturePainterEngine> {
    const engine = new FeaturePainterEngine(device, cfg);
    await engine.build(cfg);
    return engine;
  }

  private async build(cfg: FeaturePainterConfig): Promise<void> {
    const d = this.dims;
    this.prepPipe = await compute(this.device, featurePrepShader(cfg), "feature8-prep");
    this.emitPipe = await compute(this.device, featureEmitShader(cfg), "feature8-emit");
    this.forwardPipe = await compute(this.device, featureForwardShader(cfg), "feature8-forward");
    this.backwardPipe = await compute(this.device, featureBackwardShader(cfg), "feature8-backward");
    this.geometryChainPipe = await compute(this.device, featureGeometryChainShader(cfg), "feature8-geometry-chain");
    this.featureChainPipe = await compute(this.device, featureChainShader(cfg), "feature8-feature-chain");
    this.clearBinsPipe = await compute(this.device, clearShader(d.numTiles), "feature8-clear-bins");
    this.clearAccPipe = await compute(this.device, clearShader(d.G * FEATURE_ACC_STRIDE + DECODER_PARAM_COUNT), "feature8-clear-acc");
    this.adamPipe = await compute(this.device, adamShader(), "feature8-adam");

    const bind = (pipe: GPUComputePipeline, buffers: GPUBuffer[]): GPUBindGroup => this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.prepBind = bind(this.prepPipe, [this.params, this.state]);
    this.emitBind = bind(this.emitPipe, [this.state, this.tileCounts, this.binnedIds]);
    this.forwardBind = bind(this.forwardPipe, [
      this.tileCounts, this.binnedIds, this.state, this.featureParams, this.decoderParams, this.image, this.tileStop,
    ]);
    this.backwardBind = bind(this.backwardPipe, [
      this.gradImage, this.tileCounts, this.binnedIds, this.tileStop, this.state, this.featureParams, this.decoderParams, this.acc,
    ]);
    this.geometryChainBind = bind(this.geometryChainPipe, [this.acc, this.state, this.gradGeom]);
    this.featureChainBind = bind(this.featureChainPipe, [this.acc, this.gradFeature, this.gradDecoder]);
    this.clearBinsBind = bind(this.clearBinsPipe, [this.tileCounts]);
    this.clearAccBind = bind(this.clearAccPipe, [this.acc]);

    const groups = this.geometrySegments.map((segment) => {
      const uniform = adamUniform(this.device);
      this.geometryAdamUniforms.push(uniform);
      return this.device.createBindGroup({
        layout: this.adamPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: this.params } },
          { binding: 2, resource: { buffer: this.gradGeom } },
          { binding: 3, resource: { buffer: this.geomM } },
          { binding: 4, resource: { buffer: this.geomV } },
        ],
      });
    });
    this.geometryAdamGroups.push(...groups);
    this.featureAdamGroup = this.device.createBindGroup({
      layout: this.adamPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.featureAdamUniform } },
        { binding: 1, resource: { buffer: this.featureParams } },
        { binding: 2, resource: { buffer: this.gradFeature } },
        { binding: 3, resource: { buffer: this.featureM } },
        { binding: 4, resource: { buffer: this.featureV } },
      ],
    });
    this.decoderAdamGroup = this.device.createBindGroup({
      layout: this.adamPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.decoderAdamUniform } },
        { binding: 1, resource: { buffer: this.decoderParams } },
        { binding: 2, resource: { buffer: this.gradDecoder } },
        { binding: 3, resource: { buffer: this.decoderM } },
        { binding: 4, resource: { buffer: this.decoderV } },
      ],
    });
  }

  setParams(data: Float32Array): void {
    if (data.length !== this.dims.G * PARAM_STRIDE) throw new Error("feature painter: wrong geometry parameter count");
    this.device.queue.writeBuffer(this.params, 0, data as unknown as BufferSource);
  }

  setFeatureParams(data: Float32Array): void {
    if (data.length !== this.dims.G * FEATURE_STRIDE) throw new Error("feature painter: wrong feature parameter count");
    this.device.queue.writeBuffer(this.featureParams, 0, data as unknown as BufferSource);
  }

  setDecoderParams(data: Float32Array): void {
    if (data.length !== DECODER_PARAM_COUNT) throw new Error("feature painter: wrong decoder parameter count");
    this.device.queue.writeBuffer(this.decoderParams, 0, data as unknown as BufferSource);
  }

  setGradImage(data: Float32Array): void {
    if (data.length !== 3 * this.dims.H * this.dims.W) throw new Error("feature painter: wrong image gradient count");
    this.device.queue.writeBuffer(this.gradImage, 0, data as unknown as BufferSource);
  }

  private async read(buffer: GPUBuffer, floats: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({ size: floats * 4, usage: U.MAP_READ | U.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, floats * 4);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  readParams(): Promise<Float32Array> { return this.read(this.params, this.dims.G * PARAM_STRIDE); }
  readFeatureParams(): Promise<Float32Array> { return this.read(this.featureParams, this.dims.G * FEATURE_STRIDE); }
  readDecoderParams(): Promise<Float32Array> { return this.read(this.decoderParams, DECODER_PARAM_COUNT); }
  readGeometryGradient(): Promise<Float32Array> { return this.read(this.gradGeom, this.dims.G * PARAM_STRIDE); }
  readFeatureGradient(): Promise<Float32Array> { return this.read(this.gradFeature, this.dims.G * FEATURE_STRIDE); }
  readDecoderGradient(): Promise<Float32Array> { return this.read(this.gradDecoder, DECODER_PARAM_COUNT); }
  readImage(): Promise<Float32Array> { return this.read(this.image, 3 * this.dims.H * this.dims.W); }

  zeroAdamState(): void {
    const zeros = (count: number): Float32Array => new Float32Array(count);
    const geom = zeros(this.dims.G * PARAM_STRIDE);
    const feature = zeros(this.dims.G * FEATURE_STRIDE);
    const decoder = zeros(DECODER_PARAM_COUNT);
    this.device.queue.writeBuffer(this.geomM, 0, geom as unknown as BufferSource); this.device.queue.writeBuffer(this.geomV, 0, geom as unknown as BufferSource);
    this.device.queue.writeBuffer(this.featureM, 0, feature as unknown as BufferSource); this.device.queue.writeBuffer(this.featureV, 0, feature as unknown as BufferSource);
    this.device.queue.writeBuffer(this.decoderM, 0, decoder as unknown as BufferSource); this.device.queue.writeBuffer(this.decoderV, 0, decoder as unknown as BufferSource);
  }

  recordForward(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.prepPipe); pass.setBindGroup(0, this.prepBind); pass.dispatchWorkgroups(workgroups(this.dims.G));
    pass.setPipeline(this.clearBinsPipe); pass.setBindGroup(0, this.clearBinsBind); pass.dispatchWorkgroups(workgroups(this.dims.numTiles));
    pass.setPipeline(this.emitPipe); pass.setBindGroup(0, this.emitBind); pass.dispatchWorkgroups(workgroups(this.dims.G));
    pass.setPipeline(this.forwardPipe); pass.setBindGroup(0, this.forwardBind); pass.dispatchWorkgroups(this.dims.numTiles);
    pass.end();
  }

  recordBackward(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.clearAccPipe); pass.setBindGroup(0, this.clearAccBind);
    pass.dispatchWorkgroups(workgroups(this.dims.G * FEATURE_ACC_STRIDE + DECODER_PARAM_COUNT));
    pass.setPipeline(this.backwardPipe); pass.setBindGroup(0, this.backwardBind); pass.dispatchWorkgroups(this.dims.numTiles);
    pass.setPipeline(this.geometryChainPipe); pass.setBindGroup(0, this.geometryChainBind); pass.dispatchWorkgroups(workgroups(this.dims.G));
    pass.setPipeline(this.featureChainPipe); pass.setBindGroup(0, this.featureChainBind); pass.dispatchWorkgroups(workgroups(FEATURE_CHAIN_WORK_ITEMS(this.dims)));
    pass.end();
  }

  recordAdam(encoder: GPUCommandEncoder, step: number, lrs: AdamLRs, hyper: AdamHyper = DEFAULT_HYPER): void {
    const lrsBySegment = [lrs.mean, lrs.logScale, lrs.theta, lrs.color, lrs.opacity];
    this.geometrySegments.forEach((segment, index) => {
      writeAdam(this.device, this.geometryAdamUniforms[index], segment.offset, segment.length, lrsBySegment[index], step, hyper);
    });
    writeAdam(this.device, this.featureAdamUniform, 0, this.dims.G * FEATURE_STRIDE, FEATURE_LR, step, hyper);
    writeAdam(this.device, this.decoderAdamUniform, 0, DECODER_PARAM_COUNT, DECODER_LR, step, hyper);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.adamPipe);
    this.geometrySegments.forEach((segment, index) => {
      pass.setBindGroup(0, this.geometryAdamGroups[index]);
      pass.dispatchWorkgroups(workgroups(segment.length));
    });
    pass.setBindGroup(0, this.featureAdamGroup); pass.dispatchWorkgroups(workgroups(this.dims.G * FEATURE_STRIDE));
    pass.setBindGroup(0, this.decoderAdamGroup); pass.dispatchWorkgroups(workgroups(DECODER_PARAM_COUNT));
    pass.end();
  }

  runForward(): void {
    const encoder = this.device.createCommandEncoder();
    this.recordForward(encoder);
    this.device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    const buffers = [
      this.params, this.featureParams, this.decoderParams, this.image, this.gradImage, this.state,
      this.tileCounts, this.binnedIds, this.tileStop, this.acc, this.gradGeom, this.geomM, this.geomV,
      this.gradFeature, this.featureM, this.featureV, this.gradDecoder, this.decoderM, this.decoderV,
      this.featureAdamUniform, this.decoderAdamUniform, ...this.geometryAdamUniforms,
    ];
    for (const buffer of buffers) {
      try { buffer.destroy(); } catch { /* already released */ }
    }
  }
}
