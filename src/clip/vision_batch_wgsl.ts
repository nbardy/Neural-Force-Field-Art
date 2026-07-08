/**
 * vision_batch_wgsl — batch-major forward fork for the fused CLIP encoder.
 *
 * Iteration 2 of the batching work keeps the proven per-step kernels and adds
 * a real batch dimension by dispatching workgroups in z. Every activation slot
 * is allocated as `[batch][slotFloats]`; each shader indexes lane
 * `workgroup_id.z` by adding a per-binding base offset. Weights remain shared.
 *
 * This is deliberately forward-only. Backward needs the same treatment once
 * the forward batch lane proves useful.
 */
/// <reference types="@webgpu/types" />
import {
  planDispatches,
  pointwiseFusedGelu,
  stepDispatches,
  type BufferRef,
  type ConvStep,
  type DispatchSpec,
  type GeluStep,
  type VisionPlan,
} from "./vision_wgsl";
import { planBwdDispatches, type BwdDispatchOptions, type TrainPlan } from "./vision_bwd_wgsl";
import { pointwiseSharedWBatchForwardDispatch } from "./vision_batch_pointwise";

interface BatchBinding {
  name: string;
  elem: "f32" | "vec4f";
  strideFloats: number;
}

export interface BatchDispatchOptions extends BwdDispatchOptions {
  sharedWForwardSteps?: ReadonlySet<number>;
  fusePointwiseGeluForward?: boolean;
}

function mainSignature(code: string): { start: number; openBrace: number; signature: string } {
  const start = code.indexOf("fn main(");
  if (start < 0) throw new Error("vision_batch_wgsl: missing fn main");
  const openBrace = code.indexOf("{", start);
  if (openBrace < 0) throw new Error("vision_batch_wgsl: missing main body");
  return { start, openBrace, signature: code.slice(start, openBrace) };
}

function batchBindings(plan: VisionPlan | TrainPlan, spec: DispatchSpec): BatchBinding[] {
  const out: BatchBinding[] = [];
  for (let i = 0; i < spec.buffers.length; i++) {
    const ref = spec.buffers[i];
    if (ref.kind !== "slot" && ref.kind !== "text") continue;
    const re = new RegExp(
      `@group\\(0\\)\\s*@binding\\(${i}\\)\\s*var<storage,[^>]+>\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*:\\s*array<([^>]+)>`
    );
    const m = spec.code.match(re);
    if (!m) {
      throw new Error(`vision_batch_wgsl: could not find slot binding ${i} in ${spec.label}`);
    }
    const elem = m[2].trim();
    if (elem !== "f32" && elem !== "vec4f") {
      throw new Error(`vision_batch_wgsl: unsupported array<${elem}> in ${spec.label}`);
    }
    const strideFloats = ref.kind === "slot" ? plan.slots[ref.slot] : (plan as TrainPlan).textDim;
    if (!Number.isFinite(strideFloats)) {
      throw new Error(`vision_batch_wgsl: text binding in ${spec.label} needs a TrainPlan`);
    }
    if (elem === "vec4f" && strideFloats % 4 !== 0) {
      throw new Error(
        `vision_batch_wgsl: ${spec.label} binding ${i} has ${strideFloats} floats, not vec4-aligned`
      );
    }
    out.push({ name: m[1], elem, strideFloats });
  }
  return out;
}

function addWorkgroupIdIfNeeded(code: string): { code: string; batchExpr: string } {
  const sig = mainSignature(code);
  const existing = sig.signature.match(/@builtin\(workgroup_id\)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*vec3u/);
  if (existing) return { code, batchExpr: `${existing[1]}.z` };

  const injectedSig = sig.signature.replace(/\)\s*$/, ",\n        @builtin(workgroup_id) batchWid : vec3u)");
  return {
    code: code.slice(0, sig.start) + injectedSig + code.slice(sig.openBrace),
    batchExpr: "batchWid.z",
  };
}

function addBatchOffsets(plan: VisionPlan, spec: DispatchSpec): string {
  const bindings = batchBindings(plan, spec);
  if (bindings.length === 0) return spec.code;

  const withWid = addWorkgroupIdIfNeeded(spec.code);
  const sig = mainSignature(withWid.code);
  const unique = new Map<string, BatchBinding>();
  for (const b of bindings) {
    if (unique.has(b.name)) {
      throw new Error(`vision_batch_wgsl: duplicate slot variable '${b.name}' in ${spec.label}`);
    }
    unique.set(b.name, b);
  }

  const baseLines = [
    `  let batchLane = ${withWid.batchExpr};`,
    ...bindings.map((b) => {
      const stride = b.elem === "vec4f" ? b.strideFloats / 4 : b.strideFloats;
      return `  let batchBase_${b.name} = batchLane * ${stride}u;`;
    }),
  ];

  let code = withWid.code.slice(0, sig.openBrace + 1) +
    "\n" + baseLines.join("\n") +
    withWid.code.slice(sig.openBrace + 1);

  for (const b of bindings) {
    const re = new RegExp(`\\b${b.name}\\[`, "g");
    code = code.replace(re, `${b.name}[batchBase_${b.name} + `);
  }
  return code;
}

function forwardDispatches(plan: VisionPlan, batch: number, opts: BatchDispatchOptions): DispatchSpec[] {
  const specs: DispatchSpec[] = [];
  for (let index = 0; index < plan.steps.length; index++) {
    const step = plan.steps[index];
    if (
      opts.sharedWForwardSteps?.has(index) &&
      step.kind === "conv" &&
      step.variant === "pointwise"
    ) {
      specs.push(pointwiseSharedWBatchForwardDispatch(plan, step as ConvStep, batch, opts.weightPrecision));
      continue;
    }
    const next = plan.steps[index + 1];
    if (
      opts.fusePointwiseGeluForward &&
      step.kind === "conv" &&
      step.variant === "pointwise" &&
      next?.kind === "gelu" &&
      next.src === step.dst
    ) {
      specs.push(batchSpec(plan, pointwiseFusedGelu(step as ConvStep, next as GeluStep, opts, index), batch));
      index += 1;
      continue;
    }
    for (const spec of stepDispatches(step, opts, index)) {
      if (spec.workgroups[2] !== 1) {
        throw new Error(`vision_batch_wgsl: ${spec.label} already uses workgroup z=${spec.workgroups[2]}`);
      }
      specs.push({
        ...spec,
        code: addBatchOffsets(plan, spec),
        workgroups: [spec.workgroups[0], spec.workgroups[1], batch] as [number, number, number],
      });
    }
  }
  return specs;
}

function batchSpec(plan: VisionPlan, spec: DispatchSpec, batch: number): DispatchSpec {
  if (spec.workgroups[2] !== 1) {
    throw new Error(`vision_batch_wgsl: ${spec.label} already uses workgroup z=${spec.workgroups[2]}`);
  }
  return {
    ...spec,
    code: addBatchOffsets(plan, spec),
    workgroups: [spec.workgroups[0], spec.workgroups[1], batch] as [number, number, number],
  };
}

export function batchForwardDispatches(
  plan: VisionPlan,
  batch: number,
  opts: BatchDispatchOptions = {}
): DispatchSpec[] {
  if (!Number.isInteger(batch) || batch < 1) {
    throw new Error(`vision_batch_wgsl: invalid batch ${batch}`);
  }
  if (!opts.sharedWForwardSteps?.size && !opts.fusePointwiseGeluForward) {
    return planDispatches(plan, opts).map((spec) => batchSpec(plan, spec, batch));
  }
  return forwardDispatches(plan, batch, opts);
}

export function batchTrainDispatches(
  plan: TrainPlan,
  batch: number,
  opts: BatchDispatchOptions = {}
): {
  specs: DispatchSpec[];
  fwdCount: number;
} {
  if (!Number.isInteger(batch) || batch < 1) {
    throw new Error(`vision_batch_wgsl: invalid batch ${batch}`);
  }
  const fwd = opts.sharedWForwardSteps?.size || opts.fusePointwiseGeluForward
    ? forwardDispatches(plan, batch, opts)
    : planDispatches(plan, opts).map((spec) => batchSpec(plan, spec, batch));
  const bwd = planBwdDispatches(plan, opts).map((spec) => batchSpec(plan, spec, batch));
  const all = [...fwd, ...bwd];
  return { specs: all, fwdCount: fwd.length };
}

export type { BufferRef, DispatchSpec, VisionPlan };
