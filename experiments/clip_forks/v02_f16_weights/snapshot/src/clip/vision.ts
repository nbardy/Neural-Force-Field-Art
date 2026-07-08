/**
 * vision — the runtime half of the fused MobileCLIP-S0 vision encoder.
 *
 * Owns the GPU resources for a compiled plan (tools/clip/compile_plan.py):
 * one packed weights buffer, one storage buffer per activation slot, and one
 * fully-specialized compute pipeline per generated dispatch
 * (src/clip/vision_wgsl.ts). A forward pass encodes ~100 dispatches into a
 * SINGLE compute pass on one command encoder — the per-frame CPU cost is just
 * that encoding (WebGPU command buffers are single-use, so "record once" means
 * re-encoding a fixed dispatch list, which is microseconds — not ORT's
 * per-op JS graph walk).
 *
 * Device-agnostic on purpose: runs identically under bun-webgpu (Dawn/Metal,
 * headless — tools/clip/fused_test.ts) and in the browser. No tfjs coupling.
 */
/// <reference types="@webgpu/types" />
// (transitive dep of tfjs-backend-webgpu — advect.ts gets these types via its
// tfjs import; this file deliberately imports no tfjs, so reference directly)
import {
  planDispatches,
  type DispatchSpec,
  type BufferRef,
  type VisionPlan,
} from "./vision_wgsl";
import { planBwdDispatches, type TrainPlan } from "./vision_bwd_wgsl";

export type { VisionPlan, TrainPlan };

const USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };

type PassTimestampWrites = {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
};

interface Encoded {
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  workgroups: [number, number, number];
  label: string;
}

function beginComputePass(enc: GPUCommandEncoder, timestampWrites?: PassTimestampWrites): GPUComputePassEncoder {
  return timestampWrites
    ? enc.beginComputePass({ timestampWrites } as GPUComputePassDescriptor)
    : enc.beginComputePass();
}

export class VisionEncoder {
  readonly plan: VisionPlan;
  readonly device: GPUDevice;
  readonly slotBuffers: GPUBuffer[];
  readonly weightsBuffer: GPUBuffer;
  private dispatches: Encoded[] = [];

  /**
   * Async factory (pipeline validation is async). `weights` must be the
   * packed blob from compile_plan.py — its length is checked against the
   * plan loudly; a mismatched pair cannot run.
   */
  static async create(
    device: GPUDevice,
    plan: VisionPlan,
    weights: Float32Array
  ): Promise<VisionEncoder> {
    if (weights.length !== plan.weightsFloats) {
      throw new Error(
        `vision: weights blob ${weights.length} floats != plan ${plan.weightsFloats}`
      );
    }
    return new VisionEncoder(device, plan, weights, await buildPipelines(device, plan));
  }

  private constructor(
    device: GPUDevice,
    plan: VisionPlan,
    weights: Float32Array,
    built: { spec: DispatchSpec; pipeline: GPUComputePipeline }[]
  ) {
    this.device = device;
    this.plan = plan;
    this.weightsBuffer = device.createBuffer({
      size: weights.byteLength,
      usage: USAGE.STORAGE | USAGE.COPY_DST,
    });
    device.queue.writeBuffer(this.weightsBuffer, 0, weights as unknown as BufferSource);
    this.slotBuffers = plan.slots.map((floats, i) =>
      device.createBuffer({
        label: `clip-slot-${i}`,
        size: floats * 4,
        usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
      })
    );
    this.dispatches = built.map(({ spec, pipeline }) => ({
      pipeline,
      workgroups: spec.workgroups,
      label: spec.label,
      bind: this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: spec.buffers.map((ref, i) => ({
          binding: i,
          resource: {
            // Forward-only: weights + activation slots. A 'text' ref only
            // appears in the backward loss head, which lives in VisionTrainer;
            // seeing one here is a wiring bug, so fail loudly (no silent path).
            buffer:
              ref.kind === "weights"
                ? this.weightsBuffer
                : ref.kind === "slot"
                  ? this.slotBuffers[ref.slot]
                  : (() => {
                      throw new Error(
                        "vision: forward encoder received a 'text' binding (loss head belongs to VisionTrainer)"
                      );
                    })(),
          },
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

  /** Upload an NCHW [3,256,256] image in [0,1]. */
  writeInput(pixels: Float32Array): void {
    const [c, h, w] = this.plan.inputShape;
    if (pixels.length !== c * h * w) {
      throw new Error(`vision: input ${pixels.length} != ${c * h * w}`);
    }
    this.device.queue.writeBuffer(this.inputBuffer, 0, pixels as unknown as BufferSource);
  }

  /**
   * Encode the whole forward (optionally only the first `stepLimit` plan
   * steps — the per-step verification hook) into one compute pass.
   */
  encode(
    encoder: GPUCommandEncoder,
    dispatchLimit = this.dispatches.length,
    timestampWrites?: PassTimestampWrites
  ): void {
    // One compute pass for the whole forward — WebGPU guarantees storage
    // write visibility BETWEEN dispatches in a pass (each dispatch is its own
    // usage scope), verified on Dawn/Metal by the per-step suite.
    const pass = beginComputePass(encoder, timestampWrites);
    for (let i = 0; i < dispatchLimit; i++) {
      const d = this.dispatches[i];
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bind);
      pass.dispatchWorkgroups(...d.workgroups);
    }
    pass.end();
  }

  /** Submit one full forward. */
  run(): void {
    const enc = this.device.createCommandEncoder();
    this.encode(enc);
    this.device.queue.submit([enc.finish()]);
  }

  /** Dispatch count per plan step (test bisection needs the mapping).
   *  Every step kind is exactly one dispatch since attention became
   *  pointwise-conv + attn_core + pointwise-conv plan steps. */
  stepDispatchCounts(): number[] {
    return this.plan.steps.map(() => 1);
  }
}

async function buildPipelines(
  device: GPUDevice,
  plan: VisionPlan
): Promise<{ spec: DispatchSpec; pipeline: GPUComputePipeline }[]> {
  return compilePipelines(device, planDispatches(plan));
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
      throw new Error(`vision: pipeline '${spec.label}' invalid: ${err.message}\n${spec.code}`);
    }
    out.push({ spec, pipeline });
  }
  return out;
}

/**
 * VisionTrainer — the runtime for the fused backward. Owns activation AND grad
 * slot buffers (plan.slots is 2×: [0,nAct) activations, [nAct,2nAct) grads),
 * the packed weights (with transposed pointwise copies), and a per-prompt text
 * buffer. Encodes forward + loss head + backward as ONE compute pass; the
 * input gradient (dL/dpixels) lands in slot `plan.inputGradSlot`.
 *
 * Weights FROZEN — no dW, no optimizer here (spec non-goals).
 */
export class VisionTrainer {
  readonly plan: TrainPlan;
  readonly device: GPUDevice;
  readonly slotBuffers: GPUBuffer[];
  readonly weightsBuffer: GPUBuffer;
  readonly textBuffer: GPUBuffer;
  private dispatches: Encoded[] = [];
  readonly fwdCount: number;   // dispatch index where backward begins

  static async create(
    device: GPUDevice,
    plan: TrainPlan,
    weights: Float32Array
  ): Promise<VisionTrainer> {
    if (weights.length !== plan.weightsFloats) {
      throw new Error(
        `vision: weights blob ${weights.length} floats != plan ${plan.weightsFloats}`
      );
    }
    const fwd = planDispatches(plan);
    const bwd = planBwdDispatches(plan);
    const built = await compilePipelines(device, [...fwd, ...bwd]);
    return new VisionTrainer(device, plan, weights, built, fwd.length);
  }

  private constructor(
    device: GPUDevice,
    plan: TrainPlan,
    weights: Float32Array,
    built: { spec: DispatchSpec; pipeline: GPUComputePipeline }[],
    fwdCount: number
  ) {
    this.device = device;
    this.plan = plan;
    this.fwdCount = fwdCount;
    this.weightsBuffer = device.createBuffer({
      size: weights.byteLength,
      usage: USAGE.STORAGE | USAGE.COPY_DST,
    });
    device.queue.writeBuffer(this.weightsBuffer, 0, weights as unknown as BufferSource);
    this.textBuffer = device.createBuffer({
      size: plan.textDim * 4,
      usage: USAGE.STORAGE | USAGE.COPY_DST,
    });
    this.slotBuffers = plan.slots.map((floats, i) =>
      device.createBuffer({
        label: `clip-tslot-${i}`,
        size: floats * 4,
        usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
      })
    );
    const resolve = (ref: BufferRef): GPUBuffer =>
      ref.kind === "weights"
        ? this.weightsBuffer
        : ref.kind === "text"
          ? this.textBuffer
          : this.slotBuffers[ref.slot];
    this.dispatches = built.map(({ spec, pipeline }) => ({
      pipeline,
      workgroups: spec.workgroups,
      label: spec.label,
      bind: this.device.createBindGroup({
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

  writeInput(pixels: Float32Array): void {
    const [c, h, w] = this.plan.inputShape;
    if (pixels.length !== c * h * w) {
      throw new Error(`vision: input ${pixels.length} != ${c * h * w}`);
    }
    this.device.queue.writeBuffer(this.inputBuffer, 0, pixels as unknown as BufferSource);
  }

  /** Target text embedding for the −cos loss (uploaded per prompt change). */
  writeText(text: Float32Array): void {
    if (text.length !== this.plan.textDim) {
      throw new Error(`vision: text ${text.length} != ${this.plan.textDim}`);
    }
    this.device.queue.writeBuffer(this.textBuffer, 0, text as unknown as BufferSource);
  }

  /** Encode forward, then (optionally) the loss head + backward, one pass. */
  encode(
    encoder: GPUCommandEncoder,
    opts: { backward?: boolean; timestampWrites?: PassTimestampWrites } = {}
  ): void {
    const limit = opts.backward === false ? this.fwdCount : this.dispatches.length;
    const pass = beginComputePass(encoder, opts.timestampWrites);
    for (let i = 0; i < limit; i++) {
      const d = this.dispatches[i];
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bind);
      pass.dispatchWorkgroups(...d.workgroups);
    }
    pass.end();
  }

  /** Encode only the verified forward pass, preserving activations for backward. */
  encodeForward(encoder: GPUCommandEncoder, timestampWrites?: PassTimestampWrites): void {
    const pass = beginComputePass(encoder, timestampWrites);
    for (let i = 0; i < this.fwdCount; i++) {
      const d = this.dispatches[i];
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bind);
      pass.dispatchWorkgroups(...d.workgroups);
    }
    pass.end();
  }

  /** Encode only the loss head + backward. Requires a prior forward. */
  encodeBackward(encoder: GPUCommandEncoder, timestampWrites?: PassTimestampWrites): void {
    const pass = beginComputePass(encoder, timestampWrites);
    for (let i = this.fwdCount; i < this.dispatches.length; i++) {
      const d = this.dispatches[i];
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bind);
      pass.dispatchWorkgroups(...d.workgroups);
    }
    pass.end();
  }

  /** Forward + backward. dL/dpixels is left in `inputGradBuffer`. */
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
