/**
 * vision_batch — isolated CLIP batching experiments.
 *
 * This intentionally does NOT replace VisionTrainer. It shares one frozen
 * weights buffer and one compiled pipeline set across N independent activation
 * slot sets, then lets benchmarks compare repeated-image schedules:
 *
 *   separate   : one compute pass/submit per image
 *   lane-major : image 0 full CLIP, image 1 full CLIP, ... in one pass
 *   step-major : step 0 for every image, step 1 for every image, ...
 *
 * step-major is the cheap test for pipeline/cache locality before the heavier
 * true batch-major shader fork that adds a batch dimension to every kernel.
 */
/// <reference types="@webgpu/types" />
import {
  planDispatches,
  type BufferRef,
  type DispatchSpec,
  type VisionPlan,
} from "./vision_wgsl";
import { planBwdDispatches, type TrainPlan } from "./vision_bwd_wgsl";
import { batchForwardDispatches, batchTrainDispatches, type BatchDispatchOptions } from "./vision_batch_wgsl";

const USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };

export type BatchSchedule = "lane-major" | "step-major";

interface BatchEncoded {
  pipeline: GPUComputePipeline;
  binds: GPUBindGroup[];
  workgroups: [number, number, number];
  label: string;
}

interface SingleEncoded {
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  workgroups: [number, number, number];
  label: string;
}

async function compilePipelines(
  device: GPUDevice,
  specs: DispatchSpec[]
): Promise<{ spec: DispatchSpec; pipeline: GPUComputePipeline }[]> {
  const out: { spec: DispatchSpec; pipeline: GPUComputePipeline }[] = [];
  for (const spec of specs) {
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code: spec.code });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const err = await device.popErrorScope();
    if (err) {
      throw new Error(`vision_batch: pipeline '${spec.label}' invalid: ${err.message}\n${spec.code}`);
    }
    out.push({ spec, pipeline });
  }
  return out;
}

function checkLane(lane: number, batch: number): void {
  if (!Number.isInteger(lane) || lane < 0 || lane >= batch) {
    throw new Error(`vision_batch: lane ${lane} outside [0, ${batch})`);
  }
}

/**
 * Shared-weights, replicated-activation CLIP trainer for batch experiments.
 * Each lane has independent activation/grad slots and text embedding buffer.
 */
export class ReplicatedBatchVisionTrainer {
  readonly plan: TrainPlan;
  readonly device: GPUDevice;
  readonly batch: number;
  readonly weightsBuffer: GPUBuffer;
  readonly slotBuffers: GPUBuffer[][];
  readonly textBuffers: GPUBuffer[];
  readonly fwdCount: number;
  private readonly dispatches: BatchEncoded[];

  static async create(
    device: GPUDevice,
    plan: TrainPlan,
    weights: Float32Array,
    batch: number
  ): Promise<ReplicatedBatchVisionTrainer> {
    if (!Number.isInteger(batch) || batch < 1) {
      throw new Error(`vision_batch: invalid batch ${batch}`);
    }
    if (weights.length !== plan.weightsFloats) {
      throw new Error(
        `vision_batch: weights blob ${weights.length} floats != plan ${plan.weightsFloats}`
      );
    }
    const fwd = planDispatches(plan);
    const bwd = planBwdDispatches(plan);
    const built = await compilePipelines(device, [...fwd, ...bwd]);
    return new ReplicatedBatchVisionTrainer(device, plan, weights, batch, built, fwd.length);
  }

  private constructor(
    device: GPUDevice,
    plan: TrainPlan,
    weights: Float32Array,
    batch: number,
    built: { spec: DispatchSpec; pipeline: GPUComputePipeline }[],
    fwdCount: number
  ) {
    this.device = device;
    this.plan = plan;
    this.batch = batch;
    this.fwdCount = fwdCount;

    this.weightsBuffer = device.createBuffer({
      label: "clip-batch-weights",
      size: weights.byteLength,
      usage: USAGE.STORAGE | USAGE.COPY_DST,
    });
    device.queue.writeBuffer(this.weightsBuffer, 0, weights as unknown as BufferSource);

    this.slotBuffers = Array.from({ length: batch }, (_unused, lane) =>
      plan.slots.map((floats, slot) =>
        device.createBuffer({
          label: `clip-batch-lane-${lane}-slot-${slot}`,
          size: floats * 4,
          usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
        })
      )
    );
    this.textBuffers = Array.from({ length: batch }, (_unused, lane) =>
      device.createBuffer({
        label: `clip-batch-lane-${lane}-text`,
        size: plan.textDim * 4,
        usage: USAGE.STORAGE | USAGE.COPY_DST,
      })
    );

    const resolve = (lane: number, ref: BufferRef): GPUBuffer =>
      ref.kind === "weights"
        ? this.weightsBuffer
        : ref.kind === "text"
          ? this.textBuffers[lane]
          : this.slotBuffers[lane][ref.slot];

    this.dispatches = built.map(({ spec, pipeline }) => ({
      pipeline,
      workgroups: spec.workgroups,
      label: spec.label,
      binds: this.slotBuffers.map((_slots, lane) =>
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: spec.buffers.map((ref, i) => ({
            binding: i,
            resource: { buffer: resolve(lane, ref) },
          })),
        })
      ),
    }));
  }

  inputBuffer(lane: number): GPUBuffer {
    checkLane(lane, this.batch);
    return this.slotBuffers[lane][this.plan.inputSlot];
  }

  outputBuffer(lane: number): GPUBuffer {
    checkLane(lane, this.batch);
    return this.slotBuffers[lane][this.plan.outputSlot];
  }

  inputGradBuffer(lane: number): GPUBuffer {
    checkLane(lane, this.batch);
    return this.slotBuffers[lane][this.plan.inputGradSlot];
  }

  writeInput(lane: number, pixels: Float32Array): void {
    checkLane(lane, this.batch);
    const [c, h, w] = this.plan.inputShape;
    if (pixels.length !== c * h * w) {
      throw new Error(`vision_batch: input ${pixels.length} != ${c * h * w}`);
    }
    this.device.queue.writeBuffer(this.inputBuffer(lane), 0, pixels as unknown as BufferSource);
  }

  writeText(lane: number, text: Float32Array): void {
    checkLane(lane, this.batch);
    if (text.length !== this.plan.textDim) {
      throw new Error(`vision_batch: text ${text.length} != ${this.plan.textDim}`);
    }
    this.device.queue.writeBuffer(this.textBuffers[lane], 0, text as unknown as BufferSource);
  }

  encodeLane(
    encoder: GPUCommandEncoder,
    lane: number,
    opts: { backward?: boolean } = {}
  ): void {
    checkLane(lane, this.batch);
    const limit = opts.backward === false ? this.fwdCount : this.dispatches.length;
    const pass = encoder.beginComputePass();
    for (let i = 0; i < limit; i++) {
      const d = this.dispatches[i];
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.binds[lane]);
      pass.dispatchWorkgroups(...d.workgroups);
    }
    pass.end();
  }

  encode(
    encoder: GPUCommandEncoder,
    opts: { backward?: boolean; schedule?: BatchSchedule } = {}
  ): void {
    const limit = opts.backward === false ? this.fwdCount : this.dispatches.length;
    const schedule = opts.schedule ?? "step-major";
    const pass = encoder.beginComputePass();
    if (schedule === "lane-major") {
      for (let lane = 0; lane < this.batch; lane++) {
        for (let i = 0; i < limit; i++) {
          const d = this.dispatches[i];
          pass.setPipeline(d.pipeline);
          pass.setBindGroup(0, d.binds[lane]);
          pass.dispatchWorkgroups(...d.workgroups);
        }
      }
    } else {
      for (let i = 0; i < limit; i++) {
        const d = this.dispatches[i];
        pass.setPipeline(d.pipeline);
        for (let lane = 0; lane < this.batch; lane++) {
          pass.setBindGroup(0, d.binds[lane]);
          pass.dispatchWorkgroups(...d.workgroups);
        }
      }
    }
    pass.end();
  }

  runLane(lane: number, opts: { backward?: boolean } = {}): void {
    const enc = this.device.createCommandEncoder();
    this.encodeLane(enc, lane, opts);
    this.device.queue.submit([enc.finish()]);
  }

  run(opts: { backward?: boolean; schedule?: BatchSchedule } = {}): void {
    const enc = this.device.createCommandEncoder();
    this.encode(enc, opts);
    this.device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    this.weightsBuffer.destroy();
    for (const lane of this.slotBuffers) {
      for (const b of lane) b.destroy();
    }
    for (const b of this.textBuffers) b.destroy();
  }
}

/**
 * True batch-major forward encoder. Activation slots are one buffer per logical
 * slot, sized `batch * slotFloats`, and each generated shader runs all lanes by
 * dispatching z workgroups. This is the forward-only proof before batch
 * backward and shared-W pointwise kernels.
 */
export class BatchMajorVisionEncoder {
  readonly plan: VisionPlan;
  readonly device: GPUDevice;
  readonly batch: number;
  readonly weightsBuffer: GPUBuffer;
  readonly slotBuffers: GPUBuffer[];
  private readonly dispatches: SingleEncoded[];

  static async create(
    device: GPUDevice,
    plan: VisionPlan,
    weights: Float32Array,
    batch: number,
    opts: BatchDispatchOptions = {}
  ): Promise<BatchMajorVisionEncoder> {
    if (!Number.isInteger(batch) || batch < 1) {
      throw new Error(`vision_batch: invalid batch ${batch}`);
    }
    if (weights.length !== plan.weightsFloats) {
      throw new Error(
        `vision_batch: weights blob ${weights.length} floats != plan ${plan.weightsFloats}`
      );
    }
    const built = await compilePipelines(device, batchForwardDispatches(plan, batch, opts));
    return new BatchMajorVisionEncoder(device, plan, weights, batch, built);
  }

  private constructor(
    device: GPUDevice,
    plan: VisionPlan,
    weights: Float32Array,
    batch: number,
    built: { spec: DispatchSpec; pipeline: GPUComputePipeline }[]
  ) {
    this.device = device;
    this.plan = plan;
    this.batch = batch;

    this.weightsBuffer = device.createBuffer({
      label: "clip-batch-major-weights",
      size: weights.byteLength,
      usage: USAGE.STORAGE | USAGE.COPY_DST,
    });
    device.queue.writeBuffer(this.weightsBuffer, 0, weights as unknown as BufferSource);

    this.slotBuffers = plan.slots.map((floats, slot) =>
      device.createBuffer({
        label: `clip-batch-major-slot-${slot}`,
        size: floats * batch * 4,
        usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
      })
    );

    const resolve = (ref: BufferRef): GPUBuffer => {
      if (ref.kind === "weights") return this.weightsBuffer;
      if (ref.kind === "slot") return this.slotBuffers[ref.slot];
      throw new Error("vision_batch: batch-major forward received a text binding");
    };

    this.dispatches = built.map(({ spec, pipeline }) => ({
      pipeline,
      workgroups: spec.workgroups,
      label: spec.label,
      bind: device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: spec.buffers.map((ref, i) => ({
          binding: i,
          resource: { buffer: resolve(ref) },
        })),
      }),
    }));
  }

  get inputBuffer(): GPUBuffer {
    return this.slotBuffers[this.plan.inputSlot];
  }

  get outputBuffer(): GPUBuffer {
    return this.slotBuffers[this.plan.outputSlot];
  }

  slotOffsetBytes(lane: number, slot: number): number {
    checkLane(lane, this.batch);
    return lane * this.plan.slots[slot] * 4;
  }

  outputOffsetBytes(lane: number): number {
    return this.slotOffsetBytes(lane, this.plan.outputSlot);
  }

  writeInput(lane: number, pixels: Float32Array): void {
    checkLane(lane, this.batch);
    const [c, h, w] = this.plan.inputShape;
    if (pixels.length !== c * h * w) {
      throw new Error(`vision_batch: input ${pixels.length} != ${c * h * w}`);
    }
    this.device.queue.writeBuffer(
      this.inputBuffer,
      this.slotOffsetBytes(lane, this.plan.inputSlot),
      pixels as unknown as BufferSource
    );
  }

  encode(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    for (const d of this.dispatches) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bind);
      pass.dispatchWorkgroups(...d.workgroups);
    }
    pass.end();
  }

  run(): void {
    const enc = this.device.createCommandEncoder();
    this.encode(enc);
    this.device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    this.weightsBuffer.destroy();
    for (const b of this.slotBuffers) b.destroy();
  }
}

/**
 * True batch-major forward+backward trainer. This is the gradient-producing
 * counterpart to BatchMajorVisionEncoder and mirrors VisionTrainer's public
 * buffers, except each buffer is laid out as `[batch][slotFloats]`.
 */
export class BatchMajorVisionTrainer {
  readonly plan: TrainPlan;
  readonly device: GPUDevice;
  readonly batch: number;
  readonly weightsBuffer: GPUBuffer;
  readonly slotBuffers: GPUBuffer[];
  readonly textBuffer: GPUBuffer;
  readonly fwdCount: number;
  private readonly dispatches: SingleEncoded[];

  static async create(
    device: GPUDevice,
    plan: TrainPlan,
    weights: Float32Array,
    batch: number,
    opts: BatchDispatchOptions = {}
  ): Promise<BatchMajorVisionTrainer> {
    if (!Number.isInteger(batch) || batch < 1) {
      throw new Error(`vision_batch: invalid batch ${batch}`);
    }
    if (weights.length !== plan.weightsFloats) {
      throw new Error(
        `vision_batch: weights blob ${weights.length} floats != plan ${plan.weightsFloats}`
      );
    }
    const { specs, fwdCount } = batchTrainDispatches(plan, batch, opts);
    const built = await compilePipelines(device, specs);
    return new BatchMajorVisionTrainer(device, plan, weights, batch, built, fwdCount);
  }

  private constructor(
    device: GPUDevice,
    plan: TrainPlan,
    weights: Float32Array,
    batch: number,
    built: { spec: DispatchSpec; pipeline: GPUComputePipeline }[],
    fwdCount: number
  ) {
    this.device = device;
    this.plan = plan;
    this.batch = batch;
    this.fwdCount = fwdCount;

    this.weightsBuffer = device.createBuffer({
      label: "clip-batch-major-train-weights",
      size: weights.byteLength,
      usage: USAGE.STORAGE | USAGE.COPY_DST,
    });
    device.queue.writeBuffer(this.weightsBuffer, 0, weights as unknown as BufferSource);

    this.textBuffer = device.createBuffer({
      label: "clip-batch-major-text",
      size: plan.textDim * batch * 4,
      usage: USAGE.STORAGE | USAGE.COPY_DST,
    });

    this.slotBuffers = plan.slots.map((floats, slot) =>
      device.createBuffer({
        label: `clip-batch-major-train-slot-${slot}`,
        size: floats * batch * 4,
        usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
      })
    );

    const resolve = (ref: BufferRef): GPUBuffer => {
      if (ref.kind === "weights") return this.weightsBuffer;
      if (ref.kind === "text") return this.textBuffer;
      return this.slotBuffers[ref.slot];
    };

    this.dispatches = built.map(({ spec, pipeline }) => ({
      pipeline,
      workgroups: spec.workgroups,
      label: spec.label,
      bind: device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: spec.buffers.map((ref, i) => ({
          binding: i,
          resource: { buffer: resolve(ref) },
        })),
      }),
    }));
  }

  get inputBuffer(): GPUBuffer {
    return this.slotBuffers[this.plan.inputSlot];
  }

  get outputBuffer(): GPUBuffer {
    return this.slotBuffers[this.plan.outputSlot];
  }

  get inputGradBuffer(): GPUBuffer {
    return this.slotBuffers[this.plan.inputGradSlot];
  }

  slotOffsetBytes(lane: number, slot: number): number {
    checkLane(lane, this.batch);
    return lane * this.plan.slots[slot] * 4;
  }

  outputOffsetBytes(lane: number): number {
    return this.slotOffsetBytes(lane, this.plan.outputSlot);
  }

  inputGradOffsetBytes(lane: number): number {
    return this.slotOffsetBytes(lane, this.plan.inputGradSlot);
  }

  textOffsetBytes(lane: number): number {
    checkLane(lane, this.batch);
    return lane * this.plan.textDim * 4;
  }

  writeInput(lane: number, pixels: Float32Array): void {
    checkLane(lane, this.batch);
    const [c, h, w] = this.plan.inputShape;
    if (pixels.length !== c * h * w) {
      throw new Error(`vision_batch: input ${pixels.length} != ${c * h * w}`);
    }
    this.device.queue.writeBuffer(
      this.inputBuffer,
      this.slotOffsetBytes(lane, this.plan.inputSlot),
      pixels as unknown as BufferSource
    );
  }

  writeText(lane: number, text: Float32Array): void {
    checkLane(lane, this.batch);
    if (text.length !== this.plan.textDim) {
      throw new Error(`vision_batch: text ${text.length} != ${this.plan.textDim}`);
    }
    this.device.queue.writeBuffer(
      this.textBuffer,
      this.textOffsetBytes(lane),
      text as unknown as BufferSource
    );
  }

  encode(encoder: GPUCommandEncoder, opts: { backward?: boolean } = {}): void {
    const limit = opts.backward === false ? this.fwdCount : this.dispatches.length;
    const pass = encoder.beginComputePass();
    for (let i = 0; i < limit; i++) {
      const d = this.dispatches[i];
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bind);
      pass.dispatchWorkgroups(...d.workgroups);
    }
    pass.end();
  }

  run(opts: { backward?: boolean } = {}): void {
    const enc = this.device.createCommandEncoder();
    this.encode(enc, opts);
    this.device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    this.weightsBuffer.destroy();
    this.textBuffer.destroy();
    for (const b of this.slotBuffers) b.destroy();
  }
}
