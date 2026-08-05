/**
 * Direct trainable image baseline for the prompt-to-splats page.
 *
 * This deliberately owns no geometry: one raw RGB logit per output pixel is
 * optimized against the same frozen MobileCLIP trainer used by both splat
 * renderers. It is an ablation/control, not a claim that unconstrained pixels
 * are a good image prior. Comparing it with splats tells us whether a failed
 * run is CLIP's objective or the splat representation/optimizer.
 */
/// <reference types="@webgpu/types" />
import { ADAM_UNIFORM_BYTES, adamShader, DEFAULT_HYPER, type AdamHyper } from "./adam_wgsl";
import { VisionTrainer, type TrainPlan } from "../clip/vision";
import { cosine } from "./optimize";

const SIDE = 256;
const PIXELS = 3 * SIDE * SIDE;
const IMG_BYTES = PIXELS * 4;
const U = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const WORKGROUP_SIZE = 256;
const workgroups = (count: number): number => Math.ceil(count / WORKGROUP_SIZE);

/** Adam is intentionally modest: direct pixels have no useful geometry prior. */
export const PIXEL_BUFFER_LR = 0.08;

function pixelForwardShader(): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> raw : array<f32>;
@group(0) @binding(1) var<storage, read_write> image : array<f32>;
fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${PIXELS}u) { return; }
  image[i] = sigmoid1(raw[i]);
}
`;
}

function pixelChainShader(): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> image : array<f32>;
@group(0) @binding(1) var<storage, read> gradImage : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradRaw : array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${PIXELS}u) { return; }
  let rgb = image[i];
  gradRaw[i] = gradImage[i] * rgb * (1.0 - rgb);
}
`;
}

function makeAdamUniform(device: GPUDevice): GPUBuffer {
  return device.createBuffer({ size: ADAM_UNIFORM_BYTES, usage: U.UNIFORM | U.COPY_DST });
}

function writeAdamUniform(device: GPUDevice, buffer: GPUBuffer, step: number, lr: number, hyper: AdamHyper): void {
  const raw = new ArrayBuffer(ADAM_UNIFORM_BYTES);
  const u32 = new Uint32Array(raw);
  const f32 = new Float32Array(raw);
  u32[0] = 0;
  u32[1] = PIXELS;
  f32[2] = lr;
  f32[3] = hyper.beta1;
  f32[4] = hyper.beta2;
  f32[5] = hyper.eps;
  f32[6] = 1 - Math.pow(hyper.beta1, step);
  f32[7] = 1 - Math.pow(hyper.beta2, step);
  device.queue.writeBuffer(buffer, 0, raw);
}

async function compute(device: GPUDevice, code: string, label: string): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ label, code });
  const pipeline = device.createComputePipeline({ label, layout: "auto", compute: { module, entryPoint: "main" } });
  const error = await device.popErrorScope();
  if (error) throw new Error(`pixel buffer ${label}: ${(error as GPUValidationError).message}`);
  return pipeline;
}

/** GPU image parameter buffer consumed by PixelBufferOptimizer. */
export class PixelBufferEngine {
  readonly image: GPUBuffer;
  readonly gradImage: GPUBuffer;
  readonly raw: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly gradRaw: GPUBuffer;
  private readonly m: GPUBuffer;
  private readonly v: GPUBuffer;
  private readonly adamUniform: GPUBuffer;
  private readonly forwardPipe: GPUComputePipeline;
  private readonly chainPipe: GPUComputePipeline;
  private readonly adamPipe: GPUComputePipeline;
  private readonly forwardBind: GPUBindGroup;
  private readonly chainBind: GPUBindGroup;
  private readonly adamBind: GPUBindGroup;

  static async create(device: GPUDevice, seed = 1): Promise<PixelBufferEngine> {
    const raw = device.createBuffer({ size: IMG_BYTES, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    const image = device.createBuffer({ size: IMG_BYTES, usage: U.STORAGE | U.COPY_SRC });
    const gradImage = device.createBuffer({ size: IMG_BYTES, usage: U.STORAGE | U.COPY_DST });
    const gradRaw = device.createBuffer({ size: IMG_BYTES, usage: U.STORAGE | U.COPY_SRC });
    const m = device.createBuffer({ size: IMG_BYTES, usage: U.STORAGE | U.COPY_DST });
    const v = device.createBuffer({ size: IMG_BYTES, usage: U.STORAGE | U.COPY_DST });
    const adamUniform = makeAdamUniform(device);
    const [forwardPipe, chainPipe, adamPipe] = await Promise.all([
      compute(device, pixelForwardShader(), "pixel-buffer-forward"),
      compute(device, pixelChainShader(), "pixel-buffer-chain"),
      compute(device, adamShader(), "pixel-buffer-adam"),
    ]);
    const forwardBind = device.createBindGroup({
      layout: forwardPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: raw } }, { binding: 1, resource: { buffer: image } }],
    });
    const chainBind = device.createBindGroup({
      layout: chainPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: image } },
        { binding: 1, resource: { buffer: gradImage } },
        { binding: 2, resource: { buffer: gradRaw } },
      ],
    });
    const adamBind = device.createBindGroup({
      layout: adamPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: adamUniform } },
        { binding: 1, resource: { buffer: raw } },
        { binding: 2, resource: { buffer: gradRaw } },
        { binding: 3, resource: { buffer: m } },
        { binding: 4, resource: { buffer: v } },
      ],
    });
    const engine = new PixelBufferEngine(device, raw, image, gradImage, gradRaw, m, v, adamUniform, forwardPipe, chainPipe, adamPipe, forwardBind, chainBind, adamBind);
    engine.setRaw(randomPixelLogits(seed));
    engine.zeroAdamState();
    return engine;
  }

  private constructor(
    device: GPUDevice,
    raw: GPUBuffer,
    image: GPUBuffer,
    gradImage: GPUBuffer,
    gradRaw: GPUBuffer,
    m: GPUBuffer,
    v: GPUBuffer,
    adamUniform: GPUBuffer,
    forwardPipe: GPUComputePipeline,
    chainPipe: GPUComputePipeline,
    adamPipe: GPUComputePipeline,
    forwardBind: GPUBindGroup,
    chainBind: GPUBindGroup,
    adamBind: GPUBindGroup,
  ) {
    this.device = device;
    this.raw = raw;
    this.image = image;
    this.gradImage = gradImage;
    this.gradRaw = gradRaw;
    this.m = m;
    this.v = v;
    this.adamUniform = adamUniform;
    this.forwardPipe = forwardPipe;
    this.chainPipe = chainPipe;
    this.adamPipe = adamPipe;
    this.forwardBind = forwardBind;
    this.chainBind = chainBind;
    this.adamBind = adamBind;
  }

  setRaw(data: Float32Array): void {
    if (data.length !== PIXELS) throw new Error("pixel buffer: wrong raw image length");
    this.device.queue.writeBuffer(this.raw, 0, data as unknown as BufferSource);
  }

  zeroAdamState(): void {
    const zeros = new Float32Array(PIXELS);
    this.device.queue.writeBuffer(this.m, 0, zeros as unknown as BufferSource);
    this.device.queue.writeBuffer(this.v, 0, zeros as unknown as BufferSource);
  }

  recordForward(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.forwardPipe);
    pass.setBindGroup(0, this.forwardBind);
    pass.dispatchWorkgroups(workgroups(PIXELS));
    pass.end();
  }

  recordBackward(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.chainPipe);
    pass.setBindGroup(0, this.chainBind);
    pass.dispatchWorkgroups(workgroups(PIXELS));
    pass.end();
  }

  recordAdam(encoder: GPUCommandEncoder, step: number, lr = PIXEL_BUFFER_LR, hyper: AdamHyper = DEFAULT_HYPER): void {
    writeAdamUniform(this.device, this.adamUniform, step, lr, hyper);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.adamPipe);
    pass.setBindGroup(0, this.adamBind);
    pass.dispatchWorkgroups(workgroups(PIXELS));
    pass.end();
  }

  runForward(): void {
    const encoder = this.device.createCommandEncoder();
    this.recordForward(encoder);
    this.device.queue.submit([encoder.finish()]);
  }

  async readImage(): Promise<Float32Array> {
    return this.read(this.image);
  }

  async readRaw(): Promise<Float32Array> {
    return this.read(this.raw);
  }

  private async read(source: GPUBuffer): Promise<Float32Array> {
    const staging = this.device.createBuffer({ size: IMG_BYTES, usage: U.MAP_READ | U.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, staging, 0, IMG_BYTES);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return output;
  }

  destroy(): void {
    for (const buffer of [this.raw, this.image, this.gradImage, this.gradRaw, this.m, this.v, this.adamUniform]) {
      try { buffer.destroy(); } catch { /* already released */ }
    }
  }
}

export class PixelBufferOptimizer {
  readonly device: GPUDevice;
  readonly raster: PixelBufferEngine;
  readonly trainer: VisionTrainer;
  readonly side = SIDE;
  private step_ = 0;

  static async create(device: GPUDevice, plan: TrainPlan, weights: Float32Array, seed = 1): Promise<PixelBufferOptimizer> {
    const [channels, height, width] = plan.inputShape;
    if (channels !== 3 || height !== SIDE || width !== SIDE) {
      throw new Error("pixel buffer requires MobileCLIP 256x256 RGB input");
    }
    const [raster, trainer] = await Promise.all([
      PixelBufferEngine.create(device, seed),
      VisionTrainer.create(device, plan, weights),
    ]);
    return new PixelBufferOptimizer(device, raster, trainer);
  }

  private constructor(device: GPUDevice, raster: PixelBufferEngine, trainer: VisionTrainer) {
    this.device = device;
    this.raster = raster;
    this.trainer = trainer;
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
    this.raster.recordAdam(encoder, this.step_);
    this.device.queue.submit([encoder.finish()]);
  }

  async nudge(seed = Date.now(), amount = 0.12): Promise<void> {
    const raw = await this.raster.readRaw();
    const fresh = randomPixelLogits(seed);
    const chance = Math.max(0, Math.min(1, amount));
    let state = (seed ^ 0x9e3779b9) >>> 0 || 1;
    for (let i = 0; i < raw.length; i++) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      if (state / 4294967296 < chance) raw[i] = fresh[i];
    }
    this.raster.setRaw(raw);
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

/** Near-neutral, low-amplitude noise. The learned tensor is raw RGB logits. */
export function randomPixelLogits(seed = 1): Float32Array {
  const raw = new Float32Array(PIXELS);
  let state = seed >>> 0 || 1;
  const random = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  for (let i = 0; i < raw.length; i++) raw[i] = (random() - 0.5) * 0.16;
  return raw;
}

async function readFloats(device: GPUDevice, source: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: U.MAP_READ | U.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, floats * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const output = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return output;
}

export { cosine };
