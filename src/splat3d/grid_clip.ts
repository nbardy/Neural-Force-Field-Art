/// <reference types="@webgpu/types" />
import type { BatchMajorVisionTrainer } from "../clip/vision_batch";
import { prepareCamera } from "./cameras";
import { Raster3DEngine, type Raster3DIOState } from "./raster";
import type { BackgroundTextureMode } from "./background_textures";

const U = { COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };
const SIDE = 256;
const HW = SIDE * SIDE;
const IMAGE_FLOATS = 3 * HW;
const IMAGE_BYTES = IMAGE_FLOATS * 4;
const CELL = 80;
const PACKED_CELL_MAX = Math.ceil(SIDE / 3);
const CELL_HW = CELL * CELL;
const CELL_IMAGE_FLOATS = 3 * CELL_HW;
const CELL_IMAGE_BYTES = CELL_IMAGE_FLOATS * 4;
const GUTTER = 8;
const WG = 256;

type PassTimestampWrites = {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
};

interface CellBindings {
  copyPipe: GPUComputePipeline;
  copyBind: GPUBindGroup;
  scatterPipe: GPUComputePipeline;
  scatterBind: GPUBindGroup;
  workItems: number;
}

export interface Grid9Close2ClipOptions {
  directRaster?: boolean;
  rasterSide?: number;
  packedGrid?: boolean;
  gridLane?: number;
  scratchRaster?: Raster3DEngine;
  gradientScale?: number;
  retainCellState?: boolean;
  backgroundTextureMode?: BackgroundTextureMode;
  backgroundSeed?: number;
}

function cellOrigin(cell: number, packed = false): { x: number; y: number; w: number; h: number; maxSide: number } {
  const col = cell % 3;
  const row = Math.floor(cell / 3);
  if (!packed) return { x: col * (CELL + GUTTER), y: row * (CELL + GUTTER), w: CELL, h: CELL, maxSide: CELL };
  const x0 = Math.floor((col * SIDE) / 3);
  const y0 = Math.floor((row * SIDE) / 3);
  const x1 = Math.floor(((col + 1) * SIDE) / 3);
  const y1 = Math.floor(((row + 1) * SIDE) / 3);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0, maxSide: PACKED_CELL_MAX };
}

async function makeCompute(device: GPUDevice, code: string, label: string): Promise<GPUComputePipeline> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const err = await device.popErrorScope();
  if (err) {
    console.error(`--- WGSL that failed (${label}) ---\n${code}`);
    throw new Error(`grid9_close2 pipeline validation (${label}): ${(err as GPUValidationError).message}`);
  }
  return pipeline;
}

function beginComputePass(enc: GPUCommandEncoder, timestampWrites?: PassTimestampWrites): GPUComputePassEncoder {
  return timestampWrites
    ? enc.beginComputePass({ timestampWrites } as GPUComputePassDescriptor)
    : enc.beginComputePass();
}

function fl(x: number): string {
  const v = Number.isFinite(x) ? x : 1;
  let s = v.toString();
  if (!/[.eE]/.test(s)) s += ".0";
  return s;
}

function gridCopyShader(
  cell: number,
  direction: "downsample" | "scatter",
  scratchSide: number,
  gradientScale = 1,
  packedGrid = false
): string {
  const { x, y, w, h, maxSide } = cellOrigin(cell, packedGrid);
  const scratchHW = scratchSide * scratchSide;
  const srcName = direction === "downsample" ? "src" : "gridGrad";
  const dstName = direction === "downsample" ? "gridImage" : "dst";
  const resample =
    scratchSide === CELL
      ? direction === "downsample"
        ? /* wgsl */ `
  let srcPix = cy * ${CELL}u + cx;
  ${dstName}[ch * ${HW}u + dstPix] = ${srcName}[ch * ${scratchHW}u + srcPix];`
        : /* wgsl */ `
  let srcPix = cy * ${CELL}u + cx;
  ${dstName}[ch * ${scratchHW}u + srcPix] = ${srcName}[ch * ${HW}u + dstPix] * ${fl(gradientScale)};`
      : direction === "downsample"
        ? /* wgsl */ `
  let fx = (f32(cx) + 0.5) * ${fl(scratchSide / w)} - 0.5;
  let fy = (f32(cy) + 0.5) * ${fl(scratchSide / h)} - 0.5;
  let x0 = u32(clamp(floor(fx), 0.0, ${fl(scratchSide - 1)}));
  let y0 = u32(clamp(floor(fy), 0.0, ${fl(scratchSide - 1)}));
  let x1 = min(${scratchSide - 1}u, x0 + 1u);
  let y1 = min(${scratchSide - 1}u, y0 + 1u);
  let wx = clamp(fx - f32(x0), 0.0, 1.0);
  let wy = clamp(fy - f32(y0), 0.0, 1.0);
  let base = ch * ${scratchHW}u;
  let v00 = ${srcName}[base + y0 * ${scratchSide}u + x0];
  let v10 = ${srcName}[base + y0 * ${scratchSide}u + x1];
  let v01 = ${srcName}[base + y1 * ${scratchSide}u + x0];
  let v11 = ${srcName}[base + y1 * ${scratchSide}u + x1];
  let vx0 = mix(v00, v10, wx);
  let vx1 = mix(v01, v11, wx);
  ${dstName}[ch * ${HW}u + dstPix] = mix(vx0, vx1, wy);`
        : /* wgsl */ `
  let fx = (f32(cx) + 0.5) * ${fl(scratchSide / w)} - 0.5;
  let fy = (f32(cy) + 0.5) * ${fl(scratchSide / h)} - 0.5;
  let x0 = u32(clamp(floor(fx), 0.0, ${fl(scratchSide - 1)}));
  let y0 = u32(clamp(floor(fy), 0.0, ${fl(scratchSide - 1)}));
  let x1 = min(${scratchSide - 1}u, x0 + 1u);
  let y1 = min(${scratchSide - 1}u, y0 + 1u);
  let wx = clamp(fx - f32(x0), 0.0, 1.0);
  let wy = clamp(fy - f32(y0), 0.0, 1.0);
  let g = ${srcName}[ch * ${HW}u + dstPix] * ${fl(gradientScale)};
  let base = ch * ${scratchHW}u;
  ${dstName}[base + y0 * ${scratchSide}u + x0] = ${dstName}[base + y0 * ${scratchSide}u + x0] + g * (1.0 - wx) * (1.0 - wy);
  ${dstName}[base + y0 * ${scratchSide}u + x1] = ${dstName}[base + y0 * ${scratchSide}u + x1] + g * wx * (1.0 - wy);
  ${dstName}[base + y1 * ${scratchSide}u + x0] = ${dstName}[base + y1 * ${scratchSide}u + x0] + g * (1.0 - wx) * wy;
  ${dstName}[base + y1 * ${scratchSide}u + x1] = ${dstName}[base + y1 * ${scratchSide}u + x1] + g * wx * wy;`;
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> ${srcName} : array<f32>;
@group(0) @binding(1) var<storage, read_write> ${dstName} : array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${3 * maxSide * maxSide}u) { return; }
  let cellPix = i % ${maxSide * maxSide}u;
  let ch = i / ${maxSide * maxSide}u;
  let cx = cellPix % ${maxSide}u;
  let cy = cellPix / ${maxSide}u;
  if (cx >= ${w}u || cy >= ${h}u) { return; }
  let dstPix = (${y}u + cy) * ${SIDE}u + (${x}u + cx);
${resample}
}`;
}

export class Grid9Close2ClipLayout {
  readonly raster: Raster3DEngine;
  readonly scratchIO: Raster3DIOState;
  readonly retainsCellState: boolean;
  private readonly scratchIOs: Raster3DIOState[];
  private readonly scratchImage: GPUBuffer;
  private readonly scratchGrad: GPUBuffer;
  private readonly cells: CellBindings[];
  private readonly scratchImageBytes: number;
  readonly directRaster: boolean;

  private constructor(
    private readonly device: GPUDevice,
    raster: Raster3DEngine,
    scratchImage: GPUBuffer,
    scratchGrad: GPUBuffer,
    private readonly gridImageBuffer: GPUBuffer,
    private readonly gridImageOffset: number,
    cells: CellBindings[],
    directRaster: boolean,
    scratchImageBytes: number,
    private readonly ownsRaster: boolean,
    retainCellState: boolean
  ) {
    this.raster = raster;
    this.scratchImage = scratchImage;
    this.scratchGrad = scratchGrad;
    this.cells = cells;
    this.directRaster = directRaster;
    this.scratchImageBytes = scratchImageBytes;
    this.scratchIO = raster.createIOState(scratchImage, 0, scratchGrad, 0, { privateState: true });
    this.retainsCellState = retainCellState && raster.dims.H === CELL && raster.dims.W === CELL;
    this.scratchIOs = [this.scratchIO];
    while (this.scratchIOs.length < 9) {
      this.scratchIOs.push(
        this.retainsCellState
          ? raster.createIOState(scratchImage, 0, scratchGrad, 0, { privateState: true })
          : this.scratchIO
      );
    }
  }

  static async create(
    device: GPUDevice,
    raster: Raster3DEngine,
    batch: BatchMajorVisionTrainer,
    opts: Grid9Close2ClipOptions = {}
  ): Promise<Grid9Close2ClipLayout> {
    const gridLane = Math.max(0, opts.gridLane ?? 0) | 0;
    if (batch.batch <= gridLane) {
      throw new Error(`grid9_close2: grid lane ${gridLane} outside CLIP batch ${batch.batch}`);
    }
    const scratchSide = normalizeScratchSide(opts.rasterSide, opts.directRaster);
    const packedGrid = opts.packedGrid ?? scratchSide === 512;
    const directRaster = scratchSide !== SIDE;
    const gradientScale = Math.max(0, Number.isFinite(opts.gradientScale) ? opts.gradientScale! : 1);
    const scratchRaster = opts.scratchRaster
      ? opts.scratchRaster
      : directRaster
      ? await Raster3DEngine.create(device, {
          H: scratchSide,
          W: scratchSide,
          G: raster.dims.G,
          cap: raster.dims.cap,
          bg: raster.dims.bg,
          dynamicBg: raster.dims.dynamicBg,
          dynamicBgTexture: opts.backgroundTextureMode !== undefined,
          backgroundTextureMode: opts.backgroundTextureMode,
          backgroundSeed: opts.backgroundSeed,
          dynamicCoverage: raster.dims.dynamicCoverage,
          dynamicTransmittance: raster.dims.dynamicTransmittance,
          dynamicEntropy: raster.dims.dynamicEntropy,
          dynamicFootprint: raster.dims.dynamicFootprint,
          near: raster.dims.near,
          far: raster.dims.far,
          gradScale: raster.dims.gradScale,
          cameras: raster.cameras.map((c) => prepareCamera(c, scratchSide)),
          sharedParams: raster.params,
          sharedGradRaw: raster.gradRaw,
        })
      : raster;
    const ownsRaster = directRaster && !opts.scratchRaster;
    const scratchBytes = 3 * scratchSide * scratchSide * 4;
    const scratchImage = device.createBuffer({
      label: `grid9-close2-scratch-image-lane-${gridLane}`,
      size: scratchBytes,
      usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
    });
    const scratchGrad = device.createBuffer({
      label: `grid9-close2-scratch-grad-lane-${gridLane}`,
      size: scratchBytes,
      usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
    });
    const gridImageOffset = batch.slotOffsetBytes(gridLane, batch.plan.inputSlot);
    const gridGradOffset = batch.inputGradOffsetBytes(gridLane);
    const cells: CellBindings[] = [];
    for (let cell = 0; cell < 9; cell++) {
      const info = cellOrigin(cell, packedGrid);
      const workItems = 3 * info.maxSide * info.maxSide;
      const copyPipe = await makeCompute(device, gridCopyShader(cell, "downsample", scratchSide, 1, packedGrid), `grid-copy-${cell}`);
      const scatterPipe = await makeCompute(
        device,
        gridCopyShader(cell, "scatter", scratchSide, gradientScale, packedGrid),
        `grid-scatter-${cell}`
      );
      const copyBind = device.createBindGroup({
        layout: copyPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: scratchImage, offset: 0, size: scratchBytes } },
          { binding: 1, resource: { buffer: batch.inputBuffer, offset: gridImageOffset, size: IMAGE_BYTES } },
        ],
      });
      const scatterBind = device.createBindGroup({
        layout: scatterPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: batch.inputGradBuffer, offset: gridGradOffset, size: IMAGE_BYTES } },
          { binding: 1, resource: { buffer: scratchGrad, offset: 0, size: scratchBytes } },
        ],
      });
      cells.push({ copyPipe, copyBind, scatterPipe, scatterBind, workItems });
    }
    return new Grid9Close2ClipLayout(
      device,
      scratchRaster,
      scratchImage,
      scratchGrad,
      batch.inputBuffer,
      gridImageOffset,
      cells,
      directRaster,
      scratchBytes,
      ownsRaster,
      opts.retainCellState ?? scratchSide === CELL
    );
  }

  clearGridImage(enc: GPUCommandEncoder): void {
    enc.clearBuffer(this.gridImageBuffer, this.gridImageOffset, IMAGE_BYTES);
  }

  clearScratchGrad(enc: GPUCommandEncoder): void {
    enc.clearBuffer(this.scratchGrad, 0, this.scratchImageBytes);
  }

  recordCopyCell(enc: GPUCommandEncoder, cell: number, timestampWrites?: PassTimestampWrites): void {
    const c = this.cell(cell);
    const p = beginComputePass(enc, timestampWrites);
    p.setPipeline(c.copyPipe);
    p.setBindGroup(0, c.copyBind);
    p.dispatchWorkgroups(Math.ceil(c.workItems / WG));
    p.end();
  }

  recordScatterCell(enc: GPUCommandEncoder, cell: number, timestampWrites?: PassTimestampWrites): void {
    const c = this.cell(cell);
    const p = beginComputePass(enc, timestampWrites);
    p.setPipeline(c.scatterPipe);
    p.setBindGroup(0, c.scatterBind);
    p.dispatchWorkgroups(Math.ceil(c.workItems / WG));
    p.end();
  }

  scratchIOForCell(cell: number): Raster3DIOState {
    const io = this.scratchIOs[cell | 0];
    if (!io) throw new Error(`grid9_close2: bad scratch cell ${cell}`);
    return io;
  }

  destroy(): void {
    this.scratchImage.destroy();
    this.scratchGrad.destroy();
    if (this.directRaster && this.ownsRaster) this.raster.destroy();
  }

  private cell(cell: number): CellBindings {
    const c = this.cells[cell | 0];
    if (!c) throw new Error(`grid9_close2: bad cell ${cell}`);
    return c;
  }

}

function normalizeScratchSide(value: number | undefined, directRaster: boolean | undefined): number {
  const fallback = directRaster ? CELL : SIDE;
  const n = Number.isFinite(value) ? value! | 0 : fallback;
  if (n === CELL || n === SIDE || n === 512) return n;
  throw new Error(`grid9_close2: unsupported raster side ${n}; expected ${CELL}, ${SIDE}, or 512`);
}
