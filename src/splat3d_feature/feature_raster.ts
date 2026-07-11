/// <reference types="@webgpu/types" />

import {
  FEATURE_RASTER_CHANNELS,
  FEATURE_RASTER_DEFAULT_MAX_OPACITY,
  FEATURE_RASTER_DEFAULT_MIN_RADIUS,
  FEATURE_RASTER_GROUPS,
  FEATURE_RASTER_WORKGROUP_SIZE,
  featureRasterBackwardShader,
  featureRasterForwardShader,
} from "./feature_raster_wgsl";

const U = { COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };

export interface FeatureRasterConfig {
  width: number;
  height: number;
  splats: number;
  minRadius?: number;
  maxOpacity?: number;
  maxCompositePairs?: number;
  label?: string;
}

export interface FeatureRasterShape {
  width: number;
  height: number;
  pixels: number;
  splats: number;
  channels: number;
  featureGroups: number;
  minRadius: number;
  maxOpacity: number;
  compositePairs: number;
  imageFeatureFloats: number;
  splatFeatureFloats: number;
  geometryFloats: number;
}

export interface FeatureRasterBufferSlice {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

export type FeatureRasterBufferBinding = GPUBuffer | FeatureRasterBufferSlice;

export interface FeatureRasterIOBuffers {
  geometry: FeatureRasterBufferBinding;
  splatFeatures: FeatureRasterBufferBinding;
  sortedIds: FeatureRasterBufferBinding;
  background: FeatureRasterBufferBinding;
  imageFeatures: FeatureRasterBufferBinding;
  imageFeatureGrad: FeatureRasterBufferBinding;
  geometryGrad: FeatureRasterBufferBinding;
  splatFeatureGrad: FeatureRasterBufferBinding;
}

export interface FeatureRasterIOState {
  readonly buffers: Readonly<Record<keyof FeatureRasterIOBuffers, GPUBufferBinding>>;
  readonly forwardBind: GPUBindGroup;
  readonly backwardBind: GPUBindGroup;
}

export interface FeatureRasterOwnedIO {
  readonly geometry: GPUBuffer;
  readonly splatFeatures: GPUBuffer;
  readonly sortedIds: GPUBuffer;
  readonly background: GPUBuffer;
  readonly imageFeatures: GPUBuffer;
  readonly imageFeatureGrad: GPUBuffer;
  readonly geometryGrad: GPUBuffer;
  readonly splatFeatureGrad: GPUBuffer;
  readonly state: FeatureRasterIOState;
  destroy(): void;
}

export interface FeatureRasterTimestampWrites {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
}

function ceilDiv(value: number, divisor: number): number {
  return Math.ceil(value / divisor);
}

function bufferBytes(floats: number): number {
  return floats * 4;
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
    throw new Error(`feature raster pipeline validation (${label}): ${(error as GPUValidationError).message}`);
  }
  return pipeline;
}

function beginComputePass(
  encoder: GPUCommandEncoder,
  timestampWrites?: FeatureRasterTimestampWrites
): GPUComputePassEncoder {
  return timestampWrites
    ? encoder.beginComputePass({ timestampWrites } as GPUComputePassDescriptor)
    : encoder.beginComputePass();
}

export class Feature32ReferenceRaster {
  readonly shape: FeatureRasterShape;

  private forwardPipe!: GPUComputePipeline;
  private backwardPipe!: GPUComputePipeline;
  private readonly label: string;

  private constructor(private readonly device: GPUDevice, config: FeatureRasterConfig) {
    const { width, height, splats } = config;
    if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
      throw new Error(`feature raster: invalid image shape ${width}x${height}`);
    }
    if (!Number.isInteger(splats) || splats <= 0) {
      throw new Error(`feature raster: splats must be a positive integer, got ${splats}`);
    }
    const pixels = width * height;
    if (pixels % 4 !== 0) {
      throw new Error(`feature raster: width*height must be divisible by 4 for Feature32Colorizer, got ${pixels}`);
    }
    const minRadius = config.minRadius ?? FEATURE_RASTER_DEFAULT_MIN_RADIUS;
    const maxOpacity = config.maxOpacity ?? FEATURE_RASTER_DEFAULT_MAX_OPACITY;
    const compositePairs = pixels * splats;
    const maxCompositePairs = config.maxCompositePairs ?? 1_048_576;
    if (!Number.isInteger(maxCompositePairs) || maxCompositePairs <= 0) {
      throw new Error(`feature raster: maxCompositePairs must be a positive integer, got ${maxCompositePairs}`);
    }
    if (compositePairs > maxCompositePairs) {
      throw new Error(
        `feature raster: reference backward would visit ${compositePairs} pixel/splat pairs; ` +
          `limit is ${maxCompositePairs}. Use a smaller correctness case or a tiled raster.`
      );
    }
    this.shape = {
      width,
      height,
      pixels,
      splats,
      channels: FEATURE_RASTER_CHANNELS,
      featureGroups: FEATURE_RASTER_GROUPS,
      minRadius,
      maxOpacity,
      compositePairs,
      imageFeatureFloats: FEATURE_RASTER_CHANNELS * pixels,
      splatFeatureFloats: FEATURE_RASTER_CHANNELS * splats,
      geometryFloats: 4 * splats,
    };
    this.label = config.label ?? "feature32-reference-raster";
  }

  static async create(device: GPUDevice, config: FeatureRasterConfig): Promise<Feature32ReferenceRaster> {
    const raster = new Feature32ReferenceRaster(device, config);
    await raster.build();
    return raster;
  }

  private async build(): Promise<void> {
    const shaderConfig = {
      width: this.shape.width,
      height: this.shape.height,
      splats: this.shape.splats,
      minRadius: this.shape.minRadius,
      maxOpacity: this.shape.maxOpacity,
    };
    const forwardWorkgroups = ceilDiv(this.shape.pixels, FEATURE_RASTER_WORKGROUP_SIZE);
    if (forwardWorkgroups > Number(this.device.limits.maxComputeWorkgroupsPerDimension)) {
      throw new Error(`feature raster: forward dispatch ${forwardWorkgroups} exceeds the device limit`);
    }
    const maxBindingBytes = Number(this.device.limits.maxStorageBufferBindingSize);
    const largestBinding = Math.max(
      bufferBytes(this.shape.imageFeatureFloats),
      bufferBytes(this.shape.splatFeatureFloats),
      bufferBytes(this.shape.geometryFloats)
    );
    if (largestBinding > maxBindingBytes) {
      throw new Error(`feature raster: ${largestBinding}-byte binding exceeds device limit ${maxBindingBytes}`);
    }
    this.forwardPipe = await makeCompute(
      this.device,
      featureRasterForwardShader(shaderConfig),
      `${this.label}-forward`
    );
    this.backwardPipe = await makeCompute(
      this.device,
      featureRasterBackwardShader(shaderConfig),
      `${this.label}-backward`
    );
  }

  private resolveBinding(input: FeatureRasterBufferBinding, requiredBytes: number, label: string): GPUBufferBinding {
    const value = input as FeatureRasterBufferSlice;
    const slice = value.buffer ? value : { buffer: input as GPUBuffer };
    const offset = slice.offset ?? 0;
    const available = slice.size ?? Number(slice.buffer.size) - offset;
    const alignment = Number(this.device.limits.minStorageBufferOffsetAlignment);
    if (!Number.isInteger(offset) || offset < 0 || offset % alignment !== 0) {
      throw new Error(`feature raster: ${label} offset ${offset} must be aligned to ${alignment} bytes`);
    }
    if (available < requiredBytes || offset + requiredBytes > Number(slice.buffer.size)) {
      throw new Error(
        `feature raster: ${label} has ${available} bytes at offset ${offset}, needs ${requiredBytes}`
      );
    }
    return { buffer: slice.buffer, offset, size: requiredBytes };
  }

  createIOState(input: FeatureRasterIOBuffers): FeatureRasterIOState {
    const bytes = {
      geometry: bufferBytes(this.shape.geometryFloats),
      splatFeatures: bufferBytes(this.shape.splatFeatureFloats),
      sortedIds: this.shape.splats * 4,
      background: FEATURE_RASTER_CHANNELS * 4,
      imageFeatures: bufferBytes(this.shape.imageFeatureFloats),
      imageFeatureGrad: bufferBytes(this.shape.imageFeatureFloats),
      geometryGrad: bufferBytes(this.shape.geometryFloats),
      splatFeatureGrad: bufferBytes(this.shape.splatFeatureFloats),
    };
    const buffers = Object.fromEntries(
      (Object.keys(bytes) as Array<keyof typeof bytes>).map((key) => [
        key,
        this.resolveBinding(input[key], bytes[key], key),
      ])
    ) as unknown as Record<keyof FeatureRasterIOBuffers, GPUBufferBinding>;
    const bind = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]): GPUBindGroup =>
      this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    return {
      buffers,
      forwardBind: bind(this.forwardPipe, [
        { binding: 0, resource: buffers.geometry },
        { binding: 1, resource: buffers.splatFeatures },
        { binding: 2, resource: buffers.sortedIds },
        { binding: 3, resource: buffers.background },
        { binding: 4, resource: buffers.imageFeatures },
      ]),
      backwardBind: bind(this.backwardPipe, [
        { binding: 0, resource: buffers.geometry },
        { binding: 1, resource: buffers.splatFeatures },
        { binding: 2, resource: buffers.sortedIds },
        { binding: 3, resource: buffers.background },
        { binding: 4, resource: buffers.imageFeatureGrad },
        { binding: 5, resource: buffers.geometryGrad },
        { binding: 6, resource: buffers.splatFeatureGrad },
      ]),
    };
  }

  createOwnedIO(label = `${this.label}-io`): FeatureRasterOwnedIO {
    const usage = U.STORAGE | U.COPY_SRC | U.COPY_DST;
    const make = (suffix: string, size: number): GPUBuffer =>
      this.device.createBuffer({ label: `${label}-${suffix}`, size, usage });
    const geometry = make("geometry", bufferBytes(this.shape.geometryFloats));
    const splatFeatures = make("splat-features", bufferBytes(this.shape.splatFeatureFloats));
    const sortedIds = make("sorted-ids", this.shape.splats * 4);
    const background = make("background", FEATURE_RASTER_CHANNELS * 4);
    const imageFeatures = make("image-features", bufferBytes(this.shape.imageFeatureFloats));
    const imageFeatureGrad = make("image-feature-grad", bufferBytes(this.shape.imageFeatureFloats));
    const geometryGrad = make("geometry-grad", bufferBytes(this.shape.geometryFloats));
    const splatFeatureGrad = make("splat-feature-grad", bufferBytes(this.shape.splatFeatureFloats));
    const state = this.createIOState({
      geometry,
      splatFeatures,
      sortedIds,
      background,
      imageFeatures,
      imageFeatureGrad,
      geometryGrad,
      splatFeatureGrad,
    });
    return {
      geometry,
      splatFeatures,
      sortedIds,
      background,
      imageFeatures,
      imageFeatureGrad,
      geometryGrad,
      splatFeatureGrad,
      state,
      destroy(): void {
        geometry.destroy();
        splatFeatures.destroy();
        sortedIds.destroy();
        background.destroy();
        imageFeatures.destroy();
        imageFeatureGrad.destroy();
        geometryGrad.destroy();
        splatFeatureGrad.destroy();
      },
    };
  }

  setIdentityOrder(io: Pick<FeatureRasterOwnedIO, "sortedIds">): void {
    this.device.queue.writeBuffer(
      io.sortedIds,
      0,
      Uint32Array.from({ length: this.shape.splats }, (_unused, index) => index) as unknown as BufferSource
    );
  }

  recordForward(
    encoder: GPUCommandEncoder,
    io: FeatureRasterIOState,
    timestampWrites?: FeatureRasterTimestampWrites
  ): void {
    const pass = beginComputePass(encoder, timestampWrites);
    pass.setPipeline(this.forwardPipe);
    pass.setBindGroup(0, io.forwardBind);
    pass.dispatchWorkgroups(ceilDiv(this.shape.pixels, FEATURE_RASTER_WORKGROUP_SIZE));
    pass.end();
  }

  recordBackward(
    encoder: GPUCommandEncoder,
    io: FeatureRasterIOState,
    timestampWrites?: FeatureRasterTimestampWrites
  ): void {
    const pass = beginComputePass(encoder, timestampWrites);
    pass.setPipeline(this.backwardPipe);
    pass.setBindGroup(0, io.backwardBind);
    pass.dispatchWorkgroups(1);
    pass.end();
  }
}
