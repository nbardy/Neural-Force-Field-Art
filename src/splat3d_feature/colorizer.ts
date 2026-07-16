/// <reference types="@webgpu/types" />

import {
  FEATURE32_BIAS_FLOATS,
  FEATURE32_CHANNELS,
  FEATURE32_DEFAULT_RESIDUAL_SCALE,
  FEATURE32_GROUPS,
  FEATURE32_RGB_CHANNELS,
  FEATURE32_WEIGHT_FLOATS,
  FEATURE32_WORKGROUP_SIZE,
  feature32ColorizerFeatureGradShader,
  feature32ColorizerForwardShader,
  feature32ColorizerParameterGradShader,
  feature32ColorizerSgdShader,
} from "./colorizer_wgsl";

const U = { COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };

export interface Feature32ColorizerConfig {
  width: number;
  height: number;
  batch?: number;
  residualScale?: number;
  label?: string;
}

export interface Feature32ColorizerShape {
  width: number;
  height: number;
  pixels: number;
  pixelGroups: number;
  batch: number;
  residualScale: number;
  featureFloats: number;
  rgbFloats: number;
  featureBytes: number;
  rgbBytes: number;
}

export interface Feature32BufferSlice {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

export type Feature32BufferBinding = GPUBuffer | Feature32BufferSlice;

export interface Feature32ColorizerIOBuffers {
  features: Feature32BufferBinding;
  rgb: Feature32BufferBinding;
  rgbGrad: Feature32BufferBinding;
  featureGrad: Feature32BufferBinding;
}

export interface Feature32ResolvedIOBuffers {
  features: GPUBufferBinding;
  rgb: GPUBufferBinding;
  rgbGrad: GPUBufferBinding;
  featureGrad: GPUBufferBinding;
}

export interface Feature32ColorizerIOState {
  readonly buffers: Feature32ResolvedIOBuffers;
  readonly forwardBind: GPUBindGroup;
  readonly featureGradBind: GPUBindGroup;
  readonly parameterGradBind: GPUBindGroup;
}

export interface Feature32ColorizerOwnedIO {
  readonly features: GPUBuffer;
  readonly rgb: GPUBuffer;
  readonly rgbGrad: GPUBuffer;
  readonly featureGrad: GPUBuffer;
  readonly state: Feature32ColorizerIOState;
  destroy(): void;
}

export interface Feature32PassTimestampWrites {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
}

function ceilDiv(value: number, divisor: number): number {
  return Math.ceil(value / divisor);
}

function resolveShape(config: Feature32ColorizerConfig): Feature32ColorizerShape {
  const { width, height } = config;
  const batch = config.batch ?? 1;
  const residualScale = config.residualScale ?? FEATURE32_DEFAULT_RESIDUAL_SCALE;
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error(`feature32 colorizer: invalid image shape ${width}x${height}`);
  }
  if (!Number.isInteger(batch) || batch <= 0) {
    throw new Error(`feature32 colorizer: batch must be a positive integer, got ${batch}`);
  }
  if (!Number.isFinite(residualScale) || residualScale < 0) {
    throw new Error(`feature32 colorizer: residualScale must be finite and non-negative, got ${residualScale}`);
  }
  const pixels = width * height;
  if (pixels % 4 !== 0) {
    throw new Error(`feature32 colorizer: width*height must be divisible by 4 for planar vec4 packing, got ${pixels}`);
  }
  const pixelGroups = pixels / 4;
  const featureFloats = batch * FEATURE32_CHANNELS * pixels;
  const rgbFloats = batch * FEATURE32_RGB_CHANNELS * pixels;
  return {
    width,
    height,
    pixels,
    pixelGroups,
    batch,
    residualScale,
    featureFloats,
    rgbFloats,
    featureBytes: featureFloats * 4,
    rgbBytes: rgbFloats * 4,
  };
}

async function makeCompute(device: GPUDevice, code: string, label: string): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ label: `${label}-shader`, code });
  const pipeline = device.createComputePipeline({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const error = await device.popErrorScope();
  if (error) {
    console.error(`--- WGSL that failed (${label}) ---\n${code}`);
    throw new Error(`feature32 colorizer pipeline validation (${label}): ${(error as GPUValidationError).message}`);
  }
  return pipeline;
}

function beginComputePass(
  encoder: GPUCommandEncoder,
  timestampWrites?: Feature32PassTimestampWrites
): GPUComputePassEncoder {
  return timestampWrites
    ? encoder.beginComputePass({ timestampWrites } as GPUComputePassDescriptor)
    : encoder.beginComputePass();
}

function asFloat32(values: ArrayLike<number>, expected: number, label: string): Float32Array {
  if (values.length !== expected) {
    throw new Error(`feature32 colorizer: ${label} has ${values.length} floats, expected ${expected}`);
  }
  return values instanceof Float32Array ? values : Float32Array.from(values);
}

export class Feature32Colorizer {
  readonly shape: Feature32ColorizerShape;
  readonly weights: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly weightGrad: GPUBuffer;
  readonly biasGrad: GPUBuffer;

  private forwardPipe!: GPUComputePipeline;
  private featureGradPipe!: GPUComputePipeline;
  private parameterGradPipe!: GPUComputePipeline;
  private sgdPipe!: GPUComputePipeline;
  private weightSgdUniform!: GPUBuffer;
  private biasSgdUniform!: GPUBuffer;
  private weightSgdBind!: GPUBindGroup;
  private biasSgdBind!: GPUBindGroup;
  private readonly label: string;

  private constructor(private readonly device: GPUDevice, config: Feature32ColorizerConfig) {
    this.shape = resolveShape(config);
    this.label = config.label ?? "feature32-colorizer";
    const parameterUsage = U.STORAGE | U.COPY_SRC | U.COPY_DST;
    this.weights = device.createBuffer({
      label: `${this.label}-weights`,
      size: FEATURE32_WEIGHT_FLOATS * 4,
      usage: parameterUsage,
    });
    this.bias = device.createBuffer({
      label: `${this.label}-bias`,
      size: FEATURE32_BIAS_FLOATS * 4,
      usage: parameterUsage,
    });
    this.weightGrad = device.createBuffer({
      label: `${this.label}-weight-grad`,
      size: FEATURE32_WEIGHT_FLOATS * 4,
      usage: parameterUsage,
    });
    this.biasGrad = device.createBuffer({
      label: `${this.label}-bias-grad`,
      size: FEATURE32_BIAS_FLOATS * 4,
      usage: parameterUsage,
    });
  }

  static async create(device: GPUDevice, config: Feature32ColorizerConfig): Promise<Feature32Colorizer> {
    const colorizer = new Feature32Colorizer(device, config);
    try {
      await colorizer.build();
      colorizer.zeroParameters();
      colorizer.zeroParameterGradients();
      return colorizer;
    } catch (error) {
      colorizer.destroy();
      throw error;
    }
  }

  private async build(): Promise<void> {
    const shaderConfig = {
      pixels: this.shape.pixels,
      batch: this.shape.batch,
      residualScale: this.shape.residualScale,
    };
    const forwardWorkgroups = ceilDiv(this.shape.pixelGroups * this.shape.batch, FEATURE32_WORKGROUP_SIZE);
    const maxWorkgroups = Number(this.device.limits.maxComputeWorkgroupsPerDimension);
    if (forwardWorkgroups > maxWorkgroups) {
      throw new Error(
        `feature32 colorizer: forward dispatch ${forwardWorkgroups} exceeds device limit ${maxWorkgroups}`
      );
    }
    const maxBindingBytes = Number(this.device.limits.maxStorageBufferBindingSize);
    if (this.shape.featureBytes > maxBindingBytes || this.shape.rgbBytes > maxBindingBytes) {
      throw new Error(
        `feature32 colorizer: buffer binding exceeds device limit ${maxBindingBytes} bytes ` +
          `(features=${this.shape.featureBytes}, rgb=${this.shape.rgbBytes})`
      );
    }
    this.forwardPipe = await makeCompute(
      this.device,
      feature32ColorizerForwardShader(shaderConfig),
      `${this.label}-forward`
    );
    this.featureGradPipe = await makeCompute(
      this.device,
      feature32ColorizerFeatureGradShader(shaderConfig),
      `${this.label}-feature-grad`
    );
    this.parameterGradPipe = await makeCompute(
      this.device,
      feature32ColorizerParameterGradShader(shaderConfig),
      `${this.label}-parameter-grad`
    );
    this.sgdPipe = await makeCompute(this.device, feature32ColorizerSgdShader(), `${this.label}-sgd`);
    this.weightSgdUniform = this.device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.biasSgdUniform = this.device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    const sgdBind = (uniform: GPUBuffer, parameter: GPUBuffer, gradient: GPUBuffer) => this.device.createBindGroup({
      layout: this.sgdPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: parameter } },
        { binding: 2, resource: { buffer: gradient } },
      ],
    });
    this.weightSgdBind = sgdBind(this.weightSgdUniform, this.weights, this.weightGrad);
    this.biasSgdBind = sgdBind(this.biasSgdUniform, this.bias, this.biasGrad);
  }

  private resolveBinding(input: Feature32BufferBinding, requiredBytes: number, label: string): GPUBufferBinding {
    const slice = "buffer" in (input as Feature32BufferSlice) ? (input as Feature32BufferSlice) : { buffer: input as GPUBuffer };
    const offset = slice.offset ?? 0;
    const available = slice.size ?? Number(slice.buffer.size) - offset;
    const alignment = Number(this.device.limits.minStorageBufferOffsetAlignment);
    if (!Number.isInteger(offset) || offset < 0 || offset % alignment !== 0) {
      throw new Error(`feature32 colorizer: ${label} offset ${offset} must be aligned to ${alignment} bytes`);
    }
    if (available < requiredBytes || offset + requiredBytes > Number(slice.buffer.size)) {
      throw new Error(
        `feature32 colorizer: ${label} binding has ${available} bytes at offset ${offset}, needs ${requiredBytes}`
      );
    }
    return { buffer: slice.buffer, offset, size: requiredBytes };
  }

  createIOState(input: Feature32ColorizerIOBuffers): Feature32ColorizerIOState {
    const buffers: Feature32ResolvedIOBuffers = {
      features: this.resolveBinding(input.features, this.shape.featureBytes, "features"),
      rgb: this.resolveBinding(input.rgb, this.shape.rgbBytes, "rgb"),
      rgbGrad: this.resolveBinding(input.rgbGrad, this.shape.rgbBytes, "rgbGrad"),
      featureGrad: this.resolveBinding(input.featureGrad, this.shape.featureBytes, "featureGrad"),
    };
    const bind = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]): GPUBindGroup =>
      this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const parameter = (buffer: GPUBuffer, size: number): GPUBufferBinding => ({ buffer, offset: 0, size });
    return {
      buffers,
      forwardBind: bind(this.forwardPipe, [
        { binding: 0, resource: buffers.features },
        { binding: 1, resource: parameter(this.weights, FEATURE32_WEIGHT_FLOATS * 4) },
        { binding: 2, resource: parameter(this.bias, FEATURE32_BIAS_FLOATS * 4) },
        { binding: 3, resource: buffers.rgb },
      ]),
      featureGradBind: bind(this.featureGradPipe, [
        { binding: 0, resource: buffers.rgbGrad },
        { binding: 1, resource: buffers.rgb },
        { binding: 2, resource: parameter(this.weights, FEATURE32_WEIGHT_FLOATS * 4) },
        { binding: 3, resource: buffers.featureGrad },
      ]),
      parameterGradBind: bind(this.parameterGradPipe, [
        { binding: 0, resource: buffers.features },
        { binding: 1, resource: buffers.rgbGrad },
        { binding: 2, resource: buffers.rgb },
        { binding: 3, resource: parameter(this.weightGrad, FEATURE32_WEIGHT_FLOATS * 4) },
        { binding: 4, resource: parameter(this.biasGrad, FEATURE32_BIAS_FLOATS * 4) },
      ]),
    };
  }

  createOwnedIO(label = `${this.label}-io`): Feature32ColorizerOwnedIO {
    const usage = U.STORAGE | U.COPY_SRC | U.COPY_DST;
    const features = this.device.createBuffer({ label: `${label}-features`, size: this.shape.featureBytes, usage });
    const rgb = this.device.createBuffer({ label: `${label}-rgb`, size: this.shape.rgbBytes, usage });
    const rgbGrad = this.device.createBuffer({ label: `${label}-rgb-grad`, size: this.shape.rgbBytes, usage });
    const featureGrad = this.device.createBuffer({
      label: `${label}-feature-grad`,
      size: this.shape.featureBytes,
      usage,
    });
    const state = this.createIOState({ features, rgb, rgbGrad, featureGrad });
    return {
      features,
      rgb,
      rgbGrad,
      featureGrad,
      state,
      destroy(): void {
        features.destroy();
        rgb.destroy();
        rgbGrad.destroy();
        featureGrad.destroy();
      },
    };
  }

  setParameters(weights: ArrayLike<number>, bias: ArrayLike<number>): void {
    this.device.queue.writeBuffer(
      this.weights,
      0,
      asFloat32(weights, FEATURE32_WEIGHT_FLOATS, "weights") as unknown as BufferSource
    );
    const packedBias = bias.length === FEATURE32_RGB_CHANNELS
      ? new Float32Array([bias[0], bias[1], bias[2], 0])
      : asFloat32(bias, FEATURE32_BIAS_FLOATS, "bias");
    this.device.queue.writeBuffer(this.bias, 0, packedBias as unknown as BufferSource);
  }

  zeroParameters(): void {
    this.device.queue.writeBuffer(this.weights, 0, new Float32Array(FEATURE32_WEIGHT_FLOATS));
    this.device.queue.writeBuffer(this.bias, 0, new Float32Array(FEATURE32_BIAS_FLOATS));
  }

  zeroParameterGradients(): void {
    this.device.queue.writeBuffer(this.weightGrad, 0, new Float32Array(FEATURE32_WEIGHT_FLOATS));
    this.device.queue.writeBuffer(this.biasGrad, 0, new Float32Array(FEATURE32_BIAS_FLOATS));
  }

  recordForward(
    encoder: GPUCommandEncoder,
    io: Feature32ColorizerIOState,
    timestampWrites?: Feature32PassTimestampWrites
  ): void {
    const pass = beginComputePass(encoder, timestampWrites);
    pass.setPipeline(this.forwardPipe);
    pass.setBindGroup(0, io.forwardBind);
    pass.dispatchWorkgroups(ceilDiv(this.shape.pixelGroups * this.shape.batch, FEATURE32_WORKGROUP_SIZE));
    pass.end();
  }

  recordBackward(
    encoder: GPUCommandEncoder,
    io: Feature32ColorizerIOState,
    timestampWrites?: Feature32PassTimestampWrites
  ): void {
    const pass = beginComputePass(encoder, timestampWrites);
    pass.setPipeline(this.featureGradPipe);
    pass.setBindGroup(0, io.featureGradBind);
    pass.dispatchWorkgroups(
      ceilDiv(this.shape.pixelGroups * this.shape.batch, FEATURE32_WORKGROUP_SIZE),
      FEATURE32_GROUPS
    );
    pass.setPipeline(this.parameterGradPipe);
    pass.setBindGroup(0, io.parameterGradBind);
    pass.dispatchWorkgroups(FEATURE32_GROUPS);
    pass.end();
  }

  /** Applies the decoder gradients produced by `recordBackward`. */
  recordSgd(encoder: GPUCommandEncoder, lr: number): void {
    if (!Number.isFinite(lr) || lr < 0) throw new Error(`feature32 colorizer: invalid SGD lr ${lr}`);
    const write = (buffer: GPUBuffer, count: number) => {
      const raw = new ArrayBuffer(16);
      new Uint32Array(raw)[0] = count;
      new Float32Array(raw)[2] = lr;
      this.device.queue.writeBuffer(buffer, 0, raw);
    };
    write(this.weightSgdUniform, FEATURE32_WEIGHT_FLOATS / 4);
    write(this.biasSgdUniform, FEATURE32_BIAS_FLOATS / 4);
    const compute = encoder.beginComputePass();
    compute.setPipeline(this.sgdPipe);
    compute.setBindGroup(0, this.weightSgdBind);
    compute.dispatchWorkgroups(ceilDiv(FEATURE32_WEIGHT_FLOATS / 4, FEATURE32_WORKGROUP_SIZE));
    compute.setBindGroup(0, this.biasSgdBind);
    compute.dispatchWorkgroups(ceilDiv(FEATURE32_BIAS_FLOATS / 4, FEATURE32_WORKGROUP_SIZE));
    compute.end();
  }

  destroy(): void {
    this.weights.destroy();
    this.bias.destroy();
    this.weightGrad.destroy();
    this.biasGrad.destroy();
    this.weightSgdUniform?.destroy();
    this.biasSgdUniform?.destroy();
  }
}
