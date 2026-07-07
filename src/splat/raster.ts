/**
 * RasterEngine — runtime owner of the 2D Gaussian-splat rasterizer buffers and
 * the prep -> bin -> forward -> backward -> chain -> Adam pipeline. Pure codegen
 * shaders come from src/splat/raster_wgsl.ts and src/splat/adam_wgsl.ts; this
 * class holds the GPU buffers, builds the pipelines (validation error scope, so
 * WGSL errors surface even under bun-webgpu which lacks getCompilationInfo), and
 * exposes record/run pass methods plus upload/readback helpers.
 *
 * Device-agnostic: pass an explicit GPUDevice (bun-webgpu headless or browser).
 *
 * Buffer inventory (all storage buffers <= 6 per shader stage, under the WebGPU
 * default of 8):
 *   params   [G*9] f32  SoA raw params (Adam-updated)          COPY_SRC|DST
 *   derived  [G*9] f32  AoS mean/conic/color/opacity (prep out)
 *   grads accumulate:
 *   accGrad  [G*9] i32  AoS fixed-point derived-space grads    COPY_DST (clear)
 *   gradRaw  [G*9] f32  SoA raw-space grads (chain out)        COPY_SRC
 *   m,v      [G*9] f32  SoA Adam moments                       COPY_DST (zero)
 *   binning:
 *   tileCounts [T] u32  fixedbin cursor / count                COPY_DST
 *   binnedIds  [T*cap] u32
 *   tileStop   [T] u32
 *   images:
 *   image     [3HW] f32 NCHW planar output                     COPY_SRC
 *   gradImage [3HW] f32 NCHW planar dL/dpixels input           COPY_DST
 */

import {
  RasterConfig,
  RasterDims,
  resolveDims,
  prepShader,
  emitShader,
  forwardShader,
  backwardShader,
  chainShader,
  clearShader,
  paramSegments,
  PARAM_STRIDE,
  DERIVED_STRIDE,
} from "./raster_wgsl";
import { adamShader, ADAM_UNIFORM_BYTES, AdamLRs, AdamHyper, DEFAULT_LRS, DEFAULT_HYPER } from "./adam_wgsl";

const U = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const WG = 256;
const ceil = (n: number) => Math.ceil(n / WG);

async function makeCompute(device: GPUDevice, code: string, label: string): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const err = await device.popErrorScope();
  if (err) {
    console.error(`--- WGSL that failed (${label}) ---\n${code}`);
    throw new Error(`raster pipeline validation (${label}): ${(err as GPUValidationError).message}`);
  }
  return pipeline;
}

export class RasterEngine {
  readonly dims: RasterDims;
  private readonly device: GPUDevice;

  // pipelines
  private prepPipe!: GPUComputePipeline;
  private emitPipe!: GPUComputePipeline;
  private fwdPipe!: GPUComputePipeline;
  private bwdPipe!: GPUComputePipeline;
  private chainPipe!: GPUComputePipeline;
  private clearBinsPipe!: GPUComputePipeline;
  private clearGradsPipe!: GPUComputePipeline;
  private adamPipe!: GPUComputePipeline;

  // buffers
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

  // bind groups
  private prepBind!: GPUBindGroup;
  private emitBind!: GPUBindGroup;
  private fwdBind!: GPUBindGroup;
  private bwdBind!: GPUBindGroup;
  private chainBind!: GPUBindGroup;
  private clearBinsBind!: GPUBindGroup;
  private clearGradsBind!: GPUBindGroup;
  // per-group adam uniform buffers + bind groups (one per param group)
  private adamUni: GPUBuffer[] = [];
  private adamBind: GPUBindGroup[] = [];

  private constructor(device: GPUDevice, cfg: RasterConfig) {
    this.device = device;
    this.dims = resolveDims(cfg);
    if (this.dims.numTiles > 65535) throw new Error("raster: numTiles exceeds 1D dispatch limit");
  }

  static async create(device: GPUDevice, cfg: RasterConfig): Promise<RasterEngine> {
    const e = new RasterEngine(device, cfg);
    await e.build(cfg);
    return e;
  }

  private storage(floats: number, extra = 0): GPUBuffer {
    return this.device.createBuffer({ size: floats * 4, usage: U.STORAGE | extra });
  }

  private async build(cfg: RasterConfig): Promise<void> {
    const d = this.dims;
    const G9 = d.G * PARAM_STRIDE; // == d.G * DERIVED_STRIDE (both 9)
    // buffers
    this.params = this.storage(G9, U.COPY_SRC | U.COPY_DST);
    this.derived = this.storage(G9);
    this.accGrad = this.storage(G9, U.COPY_DST | U.COPY_SRC);
    this.gradRaw = this.storage(G9, U.COPY_SRC);
    this.mBuf = this.storage(G9, U.COPY_DST);
    this.vBuf = this.storage(G9, U.COPY_DST);
    this.tileCounts = this.storage(d.numTiles, U.COPY_DST | U.COPY_SRC);
    this.binnedIds = this.storage(d.numTiles * d.cap, U.COPY_SRC);
    this.tileStop = this.storage(d.numTiles, U.COPY_SRC);
    this.image = this.storage(3 * d.H * d.W, U.COPY_SRC);
    this.gradImage = this.storage(3 * d.H * d.W, U.COPY_DST);

    // pipelines
    this.prepPipe = await makeCompute(this.device, prepShader(cfg), "prep");
    this.emitPipe = await makeCompute(this.device, emitShader(cfg), "emit");
    this.fwdPipe = await makeCompute(this.device, forwardShader(cfg), "forward");
    this.bwdPipe = await makeCompute(this.device, backwardShader(cfg), "backward");
    this.chainPipe = await makeCompute(this.device, chainShader(cfg), "chain");
    this.clearBinsPipe = await makeCompute(this.device, clearShader(d.numTiles), "clearBins");
    this.clearGradsPipe = await makeCompute(this.device, clearShader(G9), "clearGrads");
    this.adamPipe = await makeCompute(this.device, adamShader(), "adam");

    const bg = (pipe: GPUComputePipeline, bufs: GPUBuffer[]): GPUBindGroup =>
      this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });

    this.prepBind = bg(this.prepPipe, [this.params, this.derived]);
    this.emitBind = bg(this.emitPipe, [this.derived, this.tileCounts, this.binnedIds]);
    this.fwdBind = bg(this.fwdPipe, [this.tileCounts, this.binnedIds, this.derived, this.image, this.tileStop]);
    this.bwdBind = bg(this.bwdPipe, [
      this.gradImage,
      this.tileCounts,
      this.binnedIds,
      this.tileStop,
      this.derived,
      this.accGrad,
    ]);
    this.chainBind = bg(this.chainPipe, [this.accGrad, this.derived, this.params, this.gradRaw]);
    this.clearBinsBind = bg(this.clearBinsPipe, [this.tileCounts]);
    this.clearGradsBind = bg(this.clearGradsPipe, [this.accGrad]);

    for (const _ of paramSegments(d.G)) {
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

  // ---- uploads / readback ------------------------------------------------
  setParams(data: Float32Array): void {
    if (data.length !== this.dims.G * PARAM_STRIDE) throw new Error("setParams: wrong length");
    this.device.queue.writeBuffer(this.params, 0, data);
  }
  setGradImage(data: Float32Array): void {
    if (data.length !== 3 * this.dims.H * this.dims.W) throw new Error("setGradImage: wrong length");
    this.device.queue.writeBuffer(this.gradImage, 0, data);
  }
  zeroAdamState(): void {
    const z = new Float32Array(this.dims.G * PARAM_STRIDE);
    this.device.queue.writeBuffer(this.mBuf, 0, z);
    this.device.queue.writeBuffer(this.vBuf, 0, z);
  }

  private async readFloats(buf: GPUBuffer, floats: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({ size: floats * 4, usage: U.MAP_READ | U.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(1 /* GPUMapMode.READ */);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }
  readImage(): Promise<Float32Array> {
    return this.readFloats(this.image, 3 * this.dims.H * this.dims.W);
  }
  readParams(): Promise<Float32Array> {
    return this.readFloats(this.params, this.dims.G * PARAM_STRIDE);
  }
  readGradRaw(): Promise<Float32Array> {
    return this.readFloats(this.gradRaw, this.dims.G * PARAM_STRIDE);
  }

  // ---- pass recording ----------------------------------------------------
  /** prep -> clear bins -> emit -> forward. Populates `derived` and `image`. */
  recordForward(enc: GPUCommandEncoder): void {
    const d = this.dims;
    const p = enc.beginComputePass();
    p.setPipeline(this.prepPipe);
    p.setBindGroup(0, this.prepBind);
    p.dispatchWorkgroups(ceil(d.G));
    p.setPipeline(this.clearBinsPipe);
    p.setBindGroup(0, this.clearBinsBind);
    p.dispatchWorkgroups(ceil(d.numTiles));
    p.setPipeline(this.emitPipe);
    p.setBindGroup(0, this.emitBind);
    p.dispatchWorkgroups(ceil(d.G));
    p.setPipeline(this.fwdPipe);
    p.setBindGroup(0, this.fwdBind);
    p.dispatchWorkgroups(d.numTiles);
    p.end();
  }

  /** clear grads -> backward -> chain. Requires a prior recordForward (uses its
   *  sorted binnedIds, tileStop and derived). Reads `gradImage`, writes gradRaw. */
  recordBackward(enc: GPUCommandEncoder): void {
    const d = this.dims;
    const p = enc.beginComputePass();
    p.setPipeline(this.clearGradsPipe);
    p.setBindGroup(0, this.clearGradsBind);
    p.dispatchWorkgroups(ceil(d.G * DERIVED_STRIDE));
    p.setPipeline(this.bwdPipe);
    p.setBindGroup(0, this.bwdBind);
    p.dispatchWorkgroups(d.numTiles);
    p.setPipeline(this.chainPipe);
    p.setBindGroup(0, this.chainBind);
    p.dispatchWorkgroups(ceil(d.G));
    p.end();
  }

  /** Adam over all 5 param groups; call after recordBackward (reads gradRaw). */
  recordAdam(enc: GPUCommandEncoder, step: number, lrs: AdamLRs = DEFAULT_LRS, hyper: AdamHyper = DEFAULT_HYPER): void {
    const segs = paramSegments(this.dims.G);
    const lrByName: Record<string, number> = {
      mean: lrs.mean,
      logScale: lrs.logScale,
      theta: lrs.theta,
      color: lrs.color,
      opacity: lrs.opacity,
    };
    const bc1 = 1 - Math.pow(hyper.beta1, step);
    const bc2 = 1 - Math.pow(hyper.beta2, step);
    // write the 5 uniforms first (queued before the submit that runs `enc`)
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

  // ---- self-submitting convenience wrappers ------------------------------
  runForward(): void {
    const enc = this.device.createCommandEncoder();
    this.recordForward(enc);
    this.device.queue.submit([enc.finish()]);
  }
  runBackward(): void {
    const enc = this.device.createCommandEncoder();
    this.recordBackward(enc);
    this.device.queue.submit([enc.finish()]);
  }
  runAdam(step: number, lrs?: AdamLRs, hyper?: AdamHyper): void {
    const enc = this.device.createCommandEncoder();
    this.recordAdam(enc, step, lrs, hyper);
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
      ...this.adamUni,
    ]) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }
}
