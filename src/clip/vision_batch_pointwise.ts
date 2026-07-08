/**
 * vision_batch_pointwise — isolated shared-W pointwise batch experiments.
 *
 * The batch-major CLIP fork runs one workgroup per image lane, so each lane
 * reloads the same 32x32 W tile. This microkernel instead puts batch lanes in
 * local_invocation_id.z: one workgroup stages W once, while each lane stages
 * its own X tile. B=2 and B=3 fit the 16 KB workgroup-memory limit.
 */
/// <reference types="@webgpu/types" />
import {
  GELU,
  assertPointwiseTiles,
  weightsDecl,
  type ConvStep,
  type DispatchSpec,
} from "./vision_wgsl";

function pointwiseBuffers(hasRes: boolean): DispatchSpec["buffers"] {
  const buffers: DispatchSpec["buffers"] = [
    { kind: "weights" },
    { kind: "slot", slot: 0 },
    { kind: "slot", slot: 1 },
  ];
  if (hasRes) buffers.push({ kind: "slot", slot: 2 });
  return buffers;
}

function postExpr(s: ConvStep, P4: number, j: number, value: string, batched = true): string {
  const a = s.act === "gelu" ? `gelu4(${value})` : value;
  if (s.residual === null) return a;
  const resAt = batched
    ? `res[resBase + (co + ${j}u) * ${P4}u + p4]`
    : `res[(co + ${j}u) * ${P4}u + p4]`;
  return `${resAt} + vec4f(W(${s.layerScaleOff}u + co + ${j}u)) * ${a}`;
}

/** Baseline z-batch pointwise: same math as the normal pointwise kernel, but
 * buffers are compact `[batch][tensor]` instead of full CLIP slot buffers. */
export function pointwiseZBatchDispatch(s: ConvStep, batch: number): DispatchSpec {
  if (!Number.isInteger(batch) || batch < 1) {
    throw new Error(`pointwise_zbatch: invalid batch ${batch}`);
  }
  const P = s.outH * s.outW;
  assertPointwiseTiles(s.name, s.cin, s.cout, P, s.wOff);
  const P4 = P / 4;
  const srcStride = s.cin * P4;
  const dstStride = s.cout * P4;
  const hasRes = s.residual !== null;
  const code = /* wgsl */ `
${weightsDecl(0)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${hasRes ? `@group(0) @binding(3) var<storage, read> res : array<vec4f>;` : ``}
${GELU}
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 256>;
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let lane = wid.z;
  let srcBase = lane * ${srcStride}u;
  let dstBase = lane * ${dstStride}u;
  let resBase = lane * ${dstStride}u;
  let p4 = wid.x * 8u + lid.x;
  let co = (wid.y * 8u + lid.y) * 4u;
  let p4base = wid.x * 8u;
  let cobase = wid.y * 32u;
  var acc0 = vec4f(W(${s.bOff}u + co));
  var acc1 = vec4f(W(${s.bOff}u + co + 1u));
  var acc2 = vec4f(W(${s.bOff}u + co + 2u));
  var acc3 = vec4f(W(${s.bOff}u + co + 3u));
  for (var ci0 = 0u; ci0 < ${s.cin}u; ci0 = ci0 + 32u) {
    for (var t = li; t < 256u; t = t + 64u) {
      let ci = t >> 3u;
      let px = t & 7u;
      xS[t] = src[srcBase + (ci0 + ci) * ${P4}u + p4base + px];
      wS[t] = W4((${s.wOff}u + (ci0 + ci) * ${s.cout}u + cobase + px * 4u) / 4u);
    }
    workgroupBarrier();
    for (var ci = 0u; ci < 32u; ci = ci + 1u) {
      let xv = xS[ci * 8u + lid.x];
      let wv = wS[ci * 8u + lid.y];
      acc0 = fma(vec4f(wv.x), xv, acc0);
      acc1 = fma(vec4f(wv.y), xv, acc1);
      acc2 = fma(vec4f(wv.z), xv, acc2);
      acc3 = fma(vec4f(wv.w), xv, acc3);
    }
    workgroupBarrier();
  }
  dst[dstBase + co * ${P4}u + p4] = ${postExpr(s, P4, 0, "acc0")};
  dst[dstBase + (co + 1u) * ${P4}u + p4] = ${postExpr(s, P4, 1, "acc1")};
  dst[dstBase + (co + 2u) * ${P4}u + p4] = ${postExpr(s, P4, 2, "acc2")};
  dst[dstBase + (co + 3u) * ${P4}u + p4] = ${postExpr(s, P4, 3, "acc3")};
}`;
  return {
    label: `pw-zbatch B${batch} ${s.cin}->${s.cout} @${s.outH}x${s.outW}`,
    code,
    workgroups: [P4 / 8, s.cout / 32, batch],
    buffers: pointwiseBuffers(hasRes),
  };
}

/** Shared-W batch pointwise: B lanes inside the same workgroup, one W tile. */
export function pointwiseSharedWBatchDispatch(s: ConvStep, batch: number): DispatchSpec {
  if (!Number.isInteger(batch) || batch < 1 || batch > 3) {
    throw new Error(`pointwise_shared_w: batch ${batch} outside [1, 3]`);
  }
  const P = s.outH * s.outW;
  assertPointwiseTiles(s.name, s.cin, s.cout, P, s.wOff);
  const P4 = P / 4;
  const srcStride = s.cin * P4;
  const dstStride = s.cout * P4;
  const hasRes = s.residual !== null;
  const code = /* wgsl */ `
${weightsDecl(0)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${hasRes ? `@group(0) @binding(3) var<storage, read> res : array<vec4f>;` : ``}
${GELU}
var<workgroup> xS : array<vec4f, ${256 * batch}>;
var<workgroup> wS : array<vec4f, 256>;
@compute @workgroup_size(8, 8, ${batch})
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u) {
  let lane = lid.z;
  let li = lid.y * 8u + lid.x;
  let xTile = lane * 256u;
  let srcBase = lane * ${srcStride}u;
  let dstBase = lane * ${dstStride}u;
  let resBase = lane * ${dstStride}u;
  let p4 = wid.x * 8u + lid.x;
  let co = (wid.y * 8u + lid.y) * 4u;
  let p4base = wid.x * 8u;
  let cobase = wid.y * 32u;
  var acc0 = vec4f(W(${s.bOff}u + co));
  var acc1 = vec4f(W(${s.bOff}u + co + 1u));
  var acc2 = vec4f(W(${s.bOff}u + co + 2u));
  var acc3 = vec4f(W(${s.bOff}u + co + 3u));
  for (var ci0 = 0u; ci0 < ${s.cin}u; ci0 = ci0 + 32u) {
    for (var t = li; t < 256u; t = t + 64u) {
      let ci = t >> 3u;
      let px = t & 7u;
      xS[xTile + t] = src[srcBase + (ci0 + ci) * ${P4}u + p4base + px];
      if (lane == 0u) {
        wS[t] = W4((${s.wOff}u + (ci0 + ci) * ${s.cout}u + cobase + px * 4u) / 4u);
      }
    }
    workgroupBarrier();
    for (var ci = 0u; ci < 32u; ci = ci + 1u) {
      let xv = xS[xTile + ci * 8u + lid.x];
      let wv = wS[ci * 8u + lid.y];
      acc0 = fma(vec4f(wv.x), xv, acc0);
      acc1 = fma(vec4f(wv.y), xv, acc1);
      acc2 = fma(vec4f(wv.z), xv, acc2);
      acc3 = fma(vec4f(wv.w), xv, acc3);
    }
    workgroupBarrier();
  }
  dst[dstBase + co * ${P4}u + p4] = ${postExpr(s, P4, 0, "acc0")};
  dst[dstBase + (co + 1u) * ${P4}u + p4] = ${postExpr(s, P4, 1, "acc1")};
  dst[dstBase + (co + 2u) * ${P4}u + p4] = ${postExpr(s, P4, 2, "acc2")};
  dst[dstBase + (co + 3u) * ${P4}u + p4] = ${postExpr(s, P4, 3, "acc3")};
}`;
  return {
    label: `pw-shared-w B${batch} ${s.cin}->${s.cout} @${s.outH}x${s.outW}`,
    code,
    workgroups: [P4 / 8, s.cout / 32, 1],
    buffers: pointwiseBuffers(hasRes),
  };
}
