/// <reference types="@webgpu/types" />
import type { BatchMajorVisionTrainer } from "../clip/vision_batch";
import { type Raster3DEngine, type Raster3DIOState } from "./raster";

const U = { COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };
const SIDE = 256;
const HW = SIDE * SIDE;
const IMAGE_FLOATS = 3 * HW;
const IMAGE_BYTES = IMAGE_FLOATS * 4;
const CELL = 80;
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
}

function cellOrigin(cell: number): { x: number; y: number } {
  const col = cell % 3;
  const row = Math.floor(cell / 3);
  return { x: col * (CELL + GUTTER), y: row * (CELL + GUTTER) };
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

function gridCopyShader(cell: number, direction: "downsample" | "scatter"): string {
  const { x, y } = cellOrigin(cell);
  const srcName = direction === "downsample" ? "src" : "gridGrad";
  const dstName = direction === "downsample" ? "gridImage" : "dst";
  const value =
    direction === "downsample"
      ? `${dstName}[ch * ${HW}u + dstPix] = ${srcName}[ch * ${HW}u + srcPix];`
      : `${dstName}[ch * ${HW}u + srcPix] = ${srcName}[ch * ${HW}u + dstPix];`;
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> ${srcName} : array<f32>;
@group(0) @binding(1) var<storage, read_write> ${dstName} : array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${3 * CELL * CELL}u) { return; }
  let cellPix = i % ${CELL * CELL}u;
  let ch = i / ${CELL * CELL}u;
  let cx = cellPix % ${CELL}u;
  let cy = cellPix / ${CELL}u;
  let srcX = min(255u, (cx * 256u + 128u) / ${CELL}u);
  let srcY = min(255u, (cy * 256u + 128u) / ${CELL}u);
  let srcPix = srcY * ${SIDE}u + srcX;
  let dstPix = (${y}u + cy) * ${SIDE}u + (${x}u + cx);
  ${value}
}`;
}

export class Grid9Close2ClipLayout {
  readonly scratchIO: Raster3DIOState;
  private readonly scratchImage: GPUBuffer;
  private readonly scratchGrad: GPUBuffer;
  private readonly cells: CellBindings[];

  private constructor(
    private readonly device: GPUDevice,
    raster: Raster3DEngine,
    scratchImage: GPUBuffer,
    scratchGrad: GPUBuffer,
    private readonly gridImageBuffer: GPUBuffer,
    private readonly gridImageOffset: number,
    cells: CellBindings[]
  ) {
    this.scratchImage = scratchImage;
    this.scratchGrad = scratchGrad;
    this.cells = cells;
    this.scratchIO = raster.createIOState(scratchImage, 0, scratchGrad, 0, { privateState: true });
  }

  static async create(
    device: GPUDevice,
    raster: Raster3DEngine,
    batch: BatchMajorVisionTrainer
  ): Promise<Grid9Close2ClipLayout> {
    if (batch.batch < 3) {
      throw new Error(`grid9_close2: needs CLIP batch >= 3, got ${batch.batch}`);
    }
    const scratchImage = device.createBuffer({
      label: "grid9-close2-scratch-image",
      size: IMAGE_BYTES,
      usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
    });
    const scratchGrad = device.createBuffer({
      label: "grid9-close2-scratch-grad",
      size: IMAGE_BYTES,
      usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
    });
    const gridImageOffset = batch.slotOffsetBytes(0, batch.plan.inputSlot);
    const gridGradOffset = batch.inputGradOffsetBytes(0);
    const cells: CellBindings[] = [];
    for (let cell = 0; cell < 9; cell++) {
      const copyPipe = await makeCompute(device, gridCopyShader(cell, "downsample"), `grid-copy-${cell}`);
      const scatterPipe = await makeCompute(device, gridCopyShader(cell, "scatter"), `grid-scatter-${cell}`);
      const copyBind = device.createBindGroup({
        layout: copyPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: scratchImage, offset: 0, size: IMAGE_BYTES } },
          { binding: 1, resource: { buffer: batch.inputBuffer, offset: gridImageOffset, size: IMAGE_BYTES } },
        ],
      });
      const scatterBind = device.createBindGroup({
        layout: scatterPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: batch.inputGradBuffer, offset: gridGradOffset, size: IMAGE_BYTES } },
          { binding: 1, resource: { buffer: scratchGrad, offset: 0, size: IMAGE_BYTES } },
        ],
      });
      cells.push({ copyPipe, copyBind, scatterPipe, scatterBind });
    }
    return new Grid9Close2ClipLayout(device, raster, scratchImage, scratchGrad, batch.inputBuffer, gridImageOffset, cells);
  }

  clearGridImage(enc: GPUCommandEncoder): void {
    enc.clearBuffer(this.gridImageBuffer, this.gridImageOffset, IMAGE_BYTES);
  }

  clearScratchGrad(enc: GPUCommandEncoder): void {
    enc.clearBuffer(this.scratchGrad, 0, IMAGE_BYTES);
  }

  recordCopyCell(enc: GPUCommandEncoder, cell: number, timestampWrites?: PassTimestampWrites): void {
    const c = this.cell(cell);
    const p = beginComputePass(enc, timestampWrites);
    p.setPipeline(c.copyPipe);
    p.setBindGroup(0, c.copyBind);
    p.dispatchWorkgroups(Math.ceil((3 * CELL * CELL) / WG));
    p.end();
  }

  recordScatterCell(enc: GPUCommandEncoder, cell: number, timestampWrites?: PassTimestampWrites): void {
    const c = this.cell(cell);
    const p = beginComputePass(enc, timestampWrites);
    p.setPipeline(c.scatterPipe);
    p.setBindGroup(0, c.scatterBind);
    p.dispatchWorkgroups(Math.ceil((3 * CELL * CELL) / WG));
    p.end();
  }

  destroy(): void {
    this.scratchImage.destroy();
    this.scratchGrad.destroy();
  }

  private cell(cell: number): CellBindings {
    const c = this.cells[cell | 0];
    if (!c) throw new Error(`grid9_close2: bad cell ${cell}`);
    return c;
  }

}
