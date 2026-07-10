!// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles
function(e,r,t,i,a){/* eslint-disable no-undef */var u="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof self?self:"undefined"!=typeof window?window:"undefined"!=typeof global?global:{},o="function"==typeof u[i]&&u[i],c=o.cache||{},s="undefined"!=typeof module&&"function"==typeof module.require&&module.require.bind(module);function n(r,t){if(!c[r]){if(!e[r]){// if we cannot find the module within our internal map or
// cache jump to the current global require ie. the last bundle
// that was added to the page.
var a="function"==typeof u[i]&&u[i];if(!t&&a)return a(r,!0);// If there are other bundles on this page the require from the
// previous one is saved to 'previousRequire'. Repeat this as
// many times as there are bundles until the module is found or
// we exhaust the require chain.
if(o)return o(r,!0);// Try the node require function if it exists.
if(s&&"string"==typeof r)return s(r);var d=Error("Cannot find module '"+r+"'");throw d.code="MODULE_NOT_FOUND",d}f.resolve=function(t){var i=e[r][1][t];return null!=i?i:t},f.cache={};var l=c[r]=new n.Module(r);e[r][0].call(l.exports,f,l,l.exports,this)}return c[r].exports;function f(e){var r=f.resolve(e);return!1===r?{}:n(r)}}n.isParcelRequire=!0,n.Module=function(e){this.id=e,this.bundle=n,this.exports={}},n.modules=e,n.cache=c,n.parent=o,n.register=function(r,t){e[r]=[function(e,r){r.exports=t},{}]},Object.defineProperty(n,"root",{get:function(){return u[i]}}),u[i]=n;for(var d=0;d<r.length;d++)n(r[d]);if(t){// Expose entry point to Node, AMD or browser globals
// Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
var l=n(t);"object"==typeof exports&&"undefined"!=typeof module?module.exports=l:"function"==typeof define&&define.amd?define(function(){return l}):a&&(this[a]=l)}}({kfWkJ:[function(e,r,t){/**
 * adam_wgsl — PURE WGSL codegen for the fused Adam optimizer over the raster's
 * raw parameter buffers (src/splat/raster_wgsl.ts data model).
 *
 * One generic kernel, thread/param over a contiguous SoA segment [offset,
 * offset+count) of the shared params/grad/m/v buffers. It is dispatched once
 * per parameter GROUP (mean, logScale, theta, color, opacity) with that group's
 * learning rate — so each dispatch has ONE learning rate and there is no
 * per-thread group lookup (the group distinction is data in the uniform, not
 * structural branching in the shader). Bias-corrected update; the caller passes
 * bc1 = 1-beta1^t and bc2 = 1-beta2^t so t need not be an integer in the shader.
 *
 * The i32 fixed-point gradient scale is already undone by the raster `chain`
 * kernel, so `grad` here is a true f32 raw-parameter gradient — Adam applies no
 * further scaling. (Adam's m/sqrt(v) is scale-invariant anyway; unscaling in
 * `chain` keeps this kernel a plain textbook Adam.)
 *
 * Zero imports; the uniform layout is the ONLY contract with the runtime.
 *//** Adam uniform: 8 x 4 bytes = 32 bytes (std140-safe: all scalars). */var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"ADAM_UNIFORM_BYTES",()=>a),i.export(t,"adamShader",()=>u),i.export(t,"DEFAULT_LRS",()=>o),i.export(t,"DEFAULT_HYPER",()=>c);let a=32;function u(){return/* wgsl */`
struct AdamU {
  offset : u32,
  count  : u32,
  lr     : f32,
  beta1  : f32,
  beta2  : f32,
  eps    : f32,
  bc1    : f32,   // 1 - beta1^t
  bc2    : f32,   // 1 - beta2^t
};
@group(0) @binding(0) var<uniform>              u      : AdamU;
@group(0) @binding(1) var<storage, read_write>  params : array<f32>;
@group(0) @binding(2) var<storage, read>        grad   : array<f32>;
@group(0) @binding(3) var<storage, read_write>  mBuf   : array<f32>;
@group(0) @binding(4) var<storage, read_write>  vBuf   : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= u.count) { return; }
  let idx = u.offset + i;
  let g = grad[idx];
  let m = u.beta1 * mBuf[idx] + (1.0 - u.beta1) * g;
  let v = u.beta2 * vBuf[idx] + (1.0 - u.beta2) * g * g;
  mBuf[idx] = m;
  vBuf[idx] = v;
  let mhat = m / u.bc1;
  let vhat = v / u.bc2;
  params[idx] = params[idx] - u.lr * mhat / (sqrt(vhat) + u.eps);
}
`}let o={mean:.01,logScale:.005,theta:.005,color:.005,opacity:.005},c={beta1:.9,beta2:.999,eps:1e-8}},{"@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],lNzsi:[function(e,r,t){/**
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
 *//// <reference types="@webgpu/types" />
// (transitive dep of tfjs-backend-webgpu — advect.ts gets these types via its
// tfjs import; this file deliberately imports no tfjs, so reference directly)
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"VisionEncoder",()=>n),/**
 * VisionTrainer — the runtime for the fused backward. Owns activation AND grad
 * slot buffers (plan.slots is 2×: [0,nAct) activations, [nAct,2nAct) grads),
 * the packed weights (with transposed pointwise copies), and a per-prompt text
 * buffer. Encodes forward + loss head + backward as ONE compute pass; the
 * input gradient (dL/dpixels) lands in slot `plan.inputGradSlot`.
 *
 * Weights FROZEN — no dW, no optimizer here (spec non-goals).
 */i.export(t,"VisionTrainer",()=>f);var a=e("./vision_wgsl"),u=e("./vision_bwd_wgsl");let o={COPY_SRC:4,COPY_DST:8,STORAGE:128};function c(e,r,t){if(r.length!==t)throw Error(`${e}: weights blob ${r.length} scalars != plan ${t}`)}function s(e,r){return r?e.beginComputePass({timestampWrites:r}):e.beginComputePass()}class n{/**
   * Async factory (pipeline validation is async). `weights` must be the
   * packed blob from compile_plan.py — its length is checked against the
   * plan loudly; a mismatched pair cannot run.
   */static async create(e,r,t,i={}){return c("vision",t,r.weightsFloats),new n(e,r,t,await d(e,r,i))}constructor(e,r,t,i){this.dispatches=[],this.device=e,this.plan=r,this.weightsBuffer=e.createBuffer({size:t.byteLength,usage:o.STORAGE|o.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,t),this.slotBuffers=r.slots.map((r,t)=>e.createBuffer({label:`clip-slot-${t}`,size:4*r,usage:o.STORAGE|o.COPY_DST|o.COPY_SRC})),this.dispatches=i.map(({spec:e,pipeline:r})=>({pipeline:r,workgroups:e.workgroups,label:e.label,bind:this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:e.buffers.map((e,r)=>({binding:r,resource:{// Forward-only: weights + activation slots. A 'text' ref only
        // appears in the backward loss head, which lives in VisionTrainer;
        // seeing one here is a wiring bug, so fail loudly (no silent path).
        buffer:"weights"===e.kind?this.weightsBuffer:"slot"===e.kind?this.slotBuffers[e.slot]:(()=>{throw Error("vision: forward encoder received a 'text' binding (loss head belongs to VisionTrainer)")})()}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}/** Upload an NCHW [3,256,256] image in [0,1]. */writeInput(e){let[r,t,i]=this.plan.inputShape;if(e.length!==r*t*i)throw Error(`vision: input ${e.length} != ${r*t*i}`);this.device.queue.writeBuffer(this.inputBuffer,0,e)}/**
   * Encode the whole forward (optionally only the first `stepLimit` plan
   * steps — the per-step verification hook) into one compute pass.
   */encode(e,r=this.dispatches.length,t){// One compute pass for the whole forward — WebGPU guarantees storage
// write visibility BETWEEN dispatches in a pass (each dispatch is its own
// usage scope), verified on Dawn/Metal by the per-step suite.
let i=s(e,t);for(let e=0;e<r;e++){let r=this.dispatches[e];i.setPipeline(r.pipeline),i.setBindGroup(0,r.bind),i.dispatchWorkgroups(...r.workgroups)}i.end()}/** Submit one full forward. */run(){let e=this.device.createCommandEncoder();this.encode(e),this.device.queue.submit([e.finish()])}/** Dispatch count per plan step (test bisection needs the mapping).
   *  Every step kind is exactly one dispatch since attention became
   *  pointwise-conv + attn_core + pointwise-conv plan steps. */stepDispatchCounts(){return this.plan.steps.map(()=>1)}}async function d(e,r,t={}){return l(e,(0,a.planDispatches)(r,t))}async function l(e,r){let t=[];for(let i of r){e.pushErrorScope("validation");let r=e.createShaderModule({code:i.code}),a=e.createComputePipeline({layout:"auto",compute:{module:r,entryPoint:"main"}}),u=await e.popErrorScope();if(u)throw Error(`vision: pipeline '${i.label}' invalid: ${u.message}
${i.code}`);t.push({spec:i,pipeline:a})}return t}class f{static async create(e,r,t,i={}){c("vision",t,r.weightsFloats);let o=(0,a.planDispatches)(r,i),s=(0,u.planBwdDispatches)(r,i),n=await l(e,[...o,...s]);return new f(e,r,t,n,o.length)}constructor(e,r,t,i,a){this.dispatches=[],this.device=e,this.plan=r,this.fwdCount=a,this.weightsBuffer=e.createBuffer({size:t.byteLength,usage:o.STORAGE|o.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,t),this.textBuffer=e.createBuffer({size:4*r.textDim,usage:o.STORAGE|o.COPY_DST}),this.slotBuffers=r.slots.map((r,t)=>e.createBuffer({label:`clip-tslot-${t}`,size:4*r,usage:o.STORAGE|o.COPY_DST|o.COPY_SRC}));let u=e=>"weights"===e.kind?this.weightsBuffer:"text"===e.kind?this.textBuffer:this.slotBuffers[e.slot];this.dispatches=i.map(({spec:e,pipeline:r})=>({pipeline:r,workgroups:e.workgroups,label:e.label,bind:this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:e.buffers.map((e,r)=>({binding:r,resource:{buffer:u(e)}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}get inputGradBuffer(){return this.slotBuffers[this.plan.inputGradSlot]}writeInput(e){let[r,t,i]=this.plan.inputShape;if(e.length!==r*t*i)throw Error(`vision: input ${e.length} != ${r*t*i}`);this.device.queue.writeBuffer(this.inputBuffer,0,e)}/** Target text embedding for the −cos loss (uploaded per prompt change). */writeText(e){if(e.length!==this.plan.textDim)throw Error(`vision: text ${e.length} != ${this.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffer,0,e)}/** Encode forward, then (optionally) the loss head + backward, one pass. */encode(e,r={}){let t=!1===r.backward?this.fwdCount:this.dispatches.length,i=s(e,r.timestampWrites);for(let e=0;e<t;e++){let r=this.dispatches[e];i.setPipeline(r.pipeline),i.setBindGroup(0,r.bind),i.dispatchWorkgroups(...r.workgroups)}i.end()}/** Encode only the verified forward pass, preserving activations for backward. */encodeForward(e,r){let t=s(e,r);for(let e=0;e<this.fwdCount;e++){let r=this.dispatches[e];t.setPipeline(r.pipeline),t.setBindGroup(0,r.bind),t.dispatchWorkgroups(...r.workgroups)}t.end()}/** Encode only the loss head + backward. Requires a prior forward. */encodeBackward(e,r){let t=s(e,r);for(let e=this.fwdCount;e<this.dispatches.length;e++){let r=this.dispatches[e];t.setPipeline(r.pipeline),t.setBindGroup(0,r.bind),t.dispatchWorkgroups(...r.workgroups)}t.end()}/** Forward + backward. dL/dpixels is left in `inputGradBuffer`. */run(e={}){let r=this.device.createCommandEncoder();this.encode(r,e),this.device.queue.submit([r.finish()])}destroy(){for(let e of(this.weightsBuffer.destroy(),this.textBuffer.destroy(),this.slotBuffers))e.destroy()}}},{"./vision_wgsl":"oFDUc","./vision_bwd_wgsl":"2Oqph","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],oFDUc:[function(e,r,t){/**
 * vision_wgsl — PURE WGSL codegen for the fused MobileCLIP-S0 vision encoder.
 *
 * The ONNX graph (518 nodes) is canonicalized offline by tools/clip/
 * compile_plan.py (κ) into a typed plan of FOUR step kinds — conv / se /
 * attention / head — plus one packed f32 weights blob. This module turns each
 * plan step into 1–3 fully-specialized compute dispatches: every shape,
 * stride and weight offset is baked into the shader source, so there are NO
 * uniforms and no runtime branching on structure. ~100 dispatches replace the
 * ~500 ONNX-op graph a standard runtime would issue (HANDOFF.md §2: cost is
 * dispatch-bound, and the CPU side of a pre-encoded pass is near-zero).
 *
 * ZERO imports on purpose: pure data → string, testable headless under
 * bun-webgpu (tools/clip/fused_test.ts) against per-step ORT goldens.
 *
 * Layout contract with κ (must not drift — the test suite pins it):
 *   activations : NCHW planar f32, batch 1 — x[c][y][x] at c*H*W + y*W + x
 *   pointwise W : TRANSPOSED [Cin][Cout] (vec4 over 4 consecutive couts)
 *   depthwise W : [C][k*k];  general W: [Cout][cpg][k][k]
 *   qkv scratch : [part(q,k,v)][head][token][dim] — column o of the ONNX qkv
 *                 matmul is exactly (part*C + head*32 + dim), so the matmul
 *                 writes it with no shuffle; q is pre-scaled by 1/sqrt(hd)
 *   attn scratch: [token][C] (token-major so proj reads rows contiguously)
 *   GELU        : exact-erf form (Abramowitz–Stegun 7.1.26, |err| < 1.5e-7),
 *                 matching the ONNX x·0.5·(1+erf(x/√2)) decomposition
 */// ---------------------------------------------------------------------------
// Canonical types (mirror plan.json — κ is the only producer)
// ---------------------------------------------------------------------------
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"weightsDecl",()=>a),i.export(t,"GELU",()=>u),i.export(t,"assertStep",()=>o),i.export(t,"PW_TILE_DECLS",()=>c),i.export(t,"PW_RECT8X16_TILE_DECLS",()=>s),/** The shared tiled-matmul body: out[co][p] = Σ_ci src[ci][p]·W[ci*cout+co].
 *  Produces acc0..acc3 (vec4 = 4 pixels × 4 couts) then stores. `init` seeds
 *  each acc (bias for fwd, 0 for bwd); `store` maps acc{j} → the value written
 *  to dst[(co+j)*P4+p4] (gelu/residual epilogue for fwd, add-into for bwd
 *  accumulate). Requires src(binding 1, array<vec4f> [Cin][P4]), dst(binding 2),
 *  weights(binding 0), and PW_TILE_DECLS in scope. */i.export(t,"pointwiseTiledMain",()=>n),/** Assert a pointwise-shaped step satisfies the tile constraints. Shared so the
 *  backward reuses the SAME loud guard (a violating shape needs a handler). */i.export(t,"assertPointwiseTiles",()=>d),i.export(t,"pointwiseFusedGelu",()=>$),i.export(t,"stepDispatches",()=>g),/** All dispatches for a full forward pass, in execution order. */i.export(t,"planDispatches",()=>v);let a=(e,r="f32")=>"f16"===r?`enable f16;
@group(0) @binding(${e}) var<storage, read> weights : array<vec4<f16>>;
fn W(i : u32) -> f32 { return f32(weights[i >> 2u][i & 3u]); }
fn W4(i : u32) -> vec4f { return vec4f(weights[i]); }`:`@group(0) @binding(${e}) var<storage, read> weights : array<vec4f>;
fn W(i : u32) -> f32 { return weights[i >> 2u][i & 3u]; }
fn W4(i : u32) -> vec4f { return weights[i]; }`,u=/* wgsl */`
fn erf1(x : f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
fn gelu1(x : f32) -> f32 { return 0.5 * x * (1.0 + erf1(x * 0.7071067811865476)); }
fn erf4(x : vec4f) -> vec4f {
  let s = sign(x);
  let a = abs(x);
  let t = vec4f(1.0) / (vec4f(1.0) + 0.3275911 * a);
  let y = vec4f(1.0) - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
fn gelu4(x : vec4f) -> vec4f { return 0.5 * x * (vec4f(1.0) + erf4(x * 0.7071067811865476)); }
`;function o(e,r){if(!e)throw Error(`vision_wgsl: ${r}`)}let c=/* wgsl */`
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 256>;`,s=/* wgsl */`
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 512>;`;function n(e){let r=r=>e.extraStore?`
  ${e.extraStore(r)}`:"",t=e.loadSrc?e.loadSrc("srcIndex"):"src[srcIndex]";return/* wgsl */`
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let p4 = wid.x * 8u + lid.x;          // this thread's pixel-quad
  let co = (wid.y * 8u + lid.y) * 4u;   // this thread's first cout
  let p4base = wid.x * 8u;
  let cobase = wid.y * 32u;
  var acc0 = ${e.init(0)};
  var acc1 = ${e.init(1)};
  var acc2 = ${e.init(2)};
  var acc3 = ${e.init(3)};
  for (var ci0 = 0u; ci0 < ${e.cin}u; ci0 = ci0 + 32u) {
    // stage: 256 vec4s each of x and W, 4 per thread
    for (var t = li; t < 256u; t = t + 64u) {
      let ci = t >> 3u;
      let lane = t & 7u;
      let srcIndex = (ci0 + ci) * ${e.P4}u + p4base + lane;
      xS[t] = ${t};
      wS[t] = W4((${e.wOff}u + (ci0 + ci) * ${e.cout}u + cobase + lane * 4u) / 4u);
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
  dst[co * ${e.P4}u + p4] = ${e.store(0)};${r(0)}
  dst[(co + 1u) * ${e.P4}u + p4] = ${e.store(1)};${r(1)}
  dst[(co + 2u) * ${e.P4}u + p4] = ${e.store(2)};${r(2)}
  dst[(co + 3u) * ${e.P4}u + p4] = ${e.store(3)};${r(3)}
}`}function d(e,r,t,i,a){o(i%32==0&&t%32==0&&r%32==0,`${e}: tiled pointwise needs P%32==0 && cout%32==0 && cin%32==0 (got P=${i} cin=${r} cout=${t})`),o(a%4==0,`${e}: wOff not 16B-aligned`)}function l(e,r){if("rect8x16"!==e.pointwiseTileVariant)return!1;let t=e.pointwiseTileSteps;return!t?.size||void 0!==r&&t.has(r)}function f(e,r,t,i,a){d(e,r,t,i,a),o(t%64==0,`${e}: rect8x16 pointwise needs cout%64==0 (got cout=${t})`)}function p(e){let r=r=>e.extraStore?`
  ${e.extraStore(r)}`:"";return/* wgsl */`
@compute @workgroup_size(8, 16)
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let p4 = wid.x * 8u + lid.x;          // this thread's pixel-quad
  let co = (wid.y * 16u + lid.y) * 4u;  // this thread's first cout
  let p4base = wid.x * 8u;
  let cobase = wid.y * 64u;
  var acc0 = ${e.init(0)};
  var acc1 = ${e.init(1)};
  var acc2 = ${e.init(2)};
  var acc3 = ${e.init(3)};
  for (var ci0 = 0u; ci0 < ${e.cin}u; ci0 = ci0 + 32u) {
    // stage: x tile is 32 ci x 8 pixel-quads; W tile is 32 ci x 16 cout-quads
    for (var t = li; t < 256u; t = t + 128u) {
      let ci = t >> 3u;
      let lane = t & 7u;
      xS[t] = src[(ci0 + ci) * ${e.P4}u + p4base + lane];
    }
    for (var t = li; t < 512u; t = t + 128u) {
      let ci = t >> 4u;
      let lane = t & 15u;
      wS[t] = W4((${e.wOff}u + (ci0 + ci) * ${e.cout}u + cobase + lane * 4u) / 4u);
    }
    workgroupBarrier();
    for (var ci = 0u; ci < 32u; ci = ci + 1u) {
      let xv = xS[ci * 8u + lid.x];
      let wv = wS[ci * 16u + lid.y];
      acc0 = fma(vec4f(wv.x), xv, acc0);
      acc1 = fma(vec4f(wv.y), xv, acc1);
      acc2 = fma(vec4f(wv.z), xv, acc2);
      acc3 = fma(vec4f(wv.w), xv, acc3);
    }
    workgroupBarrier();
  }
  dst[co * ${e.P4}u + p4] = ${e.store(0)};${r(0)}
  dst[(co + 1u) * ${e.P4}u + p4] = ${e.store(1)};${r(1)}
  dst[(co + 2u) * ${e.P4}u + p4] = ${e.store(2)};${r(2)}
  dst[(co + 3u) * ${e.P4}u + p4] = ${e.store(3)};${r(3)}
}`}function $(e,r,t={},i){if(l(t,i))return function(e,r,t={}){o("pointwise"===e.variant,`${e.name}: fused GELU only supports pointwise conv`),o("none"===e.act,`${e.name}: fused GELU expects split train-mode conv`),o(null===e.residual&&null===e.layerScaleOff,`${e.name}: fused GELU does not support residual epilogues`),o(r.src===e.dst,`${e.name}: fused GELU src slot ${r.src} != conv dst ${e.dst}`);let i=e.outH*e.outW;o(r.n===e.cout*i,`${e.name}: fused GELU n=${r.n} != cout*P=${e.cout*i}`),f(e.name,e.cin,e.cout,i,e.wOff);let c=i/4,n=/* wgsl */`
${a(0,t.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> geluDst : array<vec4f>;
${u}
${s}
${p({cin:e.cin,cout:e.cout,P4:c,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:e=>`acc${e}`,extraStore:e=>`geluDst[(co + ${e}u) * ${c}u + p4] = gelu4(acc${e});`})}`;return{label:`pw+gelu rect8x16 ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:n,workgroups:[c/8,e.cout/64,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst},{kind:"slot",slot:r.dst}]}}(e,r,t);o("pointwise"===e.variant,`${e.name}: fused GELU only supports pointwise conv`),o("none"===e.act,`${e.name}: fused GELU expects split train-mode conv`),o(null===e.residual&&null===e.layerScaleOff,`${e.name}: fused GELU does not support residual epilogues`),o(r.src===e.dst,`${e.name}: fused GELU src slot ${r.src} != conv dst ${e.dst}`);let $=e.outH*e.outW;o(r.n===e.cout*$,`${e.name}: fused GELU n=${r.n} != cout*P=${e.cout*$}`),d(e.name,e.cin,e.cout,$,e.wOff);let g=$/4,v=/* wgsl */`
${a(0,t.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> geluDst : array<vec4f>;
${u}
${c}
${n({cin:e.cin,cout:e.cout,P4:g,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:e=>`acc${e}`,extraStore:e=>`geluDst[(co + ${e}u) * ${g}u + p4] = gelu4(acc${e});`})}`;return{label:`pw+gelu ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:v,workgroups:[g/8,e.cout/32,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst},{kind:"slot",slot:r.dst}]}}function g(e,r={},t){switch(e.kind){case"conv":return(// ---------------------------------------------------------------------------
// Thin dispatchers — step kind → dispatch list; conv variant → emitter.
// ---------------------------------------------------------------------------
function(e,r={},t){switch(e.variant){case"pointwise":return[function(e,r={},t){if(l(r,t))return function(e,r={}){let t=e.outH*e.outW;f(e.name,e.cin,e.cout,t,e.wOff);let i=t/4,c=null!==e.residual;o(null!==e.layerScaleOff===c,`${e.name}: layerScale without residual`);let n=[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}];c&&n.push({kind:"slot",slot:e.residual});let d=/* wgsl */`
${a(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${c?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${u}
${s}
${p({cin:e.cin,cout:e.cout,P4:i,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:r=>{let t="gelu"===e.act?`gelu4(acc${r})`:`acc${r}`;return c?`res[(co + ${r}u) * ${i}u + p4] + vec4f(W(${e.layerScaleOff}u + co + ${r}u)) * ${t}`:t}})}`;return{label:`pw rect8x16 ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:d,workgroups:[i/8,e.cout/64,1],buffers:n}}(e,r);let i=e.outH*e.outW;d(e.name,e.cin,e.cout,i,e.wOff);let $=i/4,g=null!==e.residual;o(null!==e.layerScaleOff===g,`${e.name}: layerScale without residual`);let v=[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}];g&&v.push({kind:"slot",slot:e.residual});let w=/* wgsl */`
${a(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${g?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${u}
${c}
${n({cin:e.cin,cout:e.cout,P4:$,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:r=>{let t="gelu"===e.act?`gelu4(acc${r})`:`acc${r}`;return g?`res[(co + ${r}u) * ${$}u + p4] + vec4f(W(${e.layerScaleOff}u + co + ${r}u)) * ${t}`:t}})}`;return{label:`pw ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:w,workgroups:[$/8,e.cout/32,1],buffers:v}}(e,r,t)];case"depthwise":case"general":return[// ---------------------------------------------------------------------------
// conv:depthwise — k∈{3,7}, groups=C. Thread = one output pixel of one channel.
// ---------------------------------------------------------------------------
function(e,r={}){o(null===e.residual&&null===e.layerScaleOff,`${e.name}: spatial conv never carries residual in this plan`),o(e.outW%4==0,`${e.name}: spatial tiling needs outW%4==0`);let t=e.outH*e.outW,i=t/4,c=e.k,s=e.stride,n=e.pad,d=3*s+c,l=e.cin/e.groups,f=e.cout/e.groups,p=l*c*c;o(Number.isInteger(l)&&Number.isInteger(f),`${e.name}: bad groups`),o(p<=64,`${e.name}: weight tile ${p} exceeds one staging round`);let $=r=>"gelu"===e.act?`gelu1(${r})`:r,g=[];for(let r=0;r<l;r++){g.push(`    { let base = (ci0 + ${r}u) * ${e.h*e.w}u;`);for(let t=0;t<c;t++){g.push(`      { let rowBase = base + u32(iy0 + ${t}) * ${e.w}u + u32(ix0);`);for(let e=0;e<d;e++)g.push(`        let r${e} = src[rowBase + ${e}u];`);for(let e=0;e<c;e++)g.push(`        acc = fma(vec4f(r${e}, r${s+e}, r${2*s+e}, r${3*s+e}), vec4f(wk[${r*c*c+t*c+e}u]), acc);`);g.push("      }")}g.push("    }")}// One workgroup = one output channel: its cpg·k·k weights are staged once
// in shared memory (depthwise is just cpg=1). Each thread produces 4
// horizontal pixels, loading each input row segment once (NIN loads for
// 4·K taps ≈ 2.8× fewer for k=7). Interior tiles (the vast majority) take
// a single-branch unchecked path.
let v=/* wgsl */`
${a(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
${u}
var<workgroup> wk : array<f32, ${p}>;
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let co = gid.y;
  if (li < ${p}u) { wk[li] = W(${e.wOff}u + co * ${p}u + li); }
  workgroupBarrier();
  let q = gid.x;
  if (q >= ${i}u) { return; }
  let oy = i32(q / ${e.outW/4}u);
  let ox0 = i32(q % ${e.outW/4}u) * 4;
  let ci0 = (co / ${f}u) * ${l}u;   // first input channel of co's group
  let iy0 = oy * ${s} - ${n};
  let ix0 = ox0 * ${s} - ${n};
  var acc = vec4f(W(${e.bOff}u + co));
  if (iy0 >= 0 && iy0 + ${c} <= ${e.h} && ix0 >= 0 && ix0 + ${d} <= ${e.w}) {
    // interior: every tap in bounds, unchecked unrolled register loads
${g.join("\n")}
  } else {
    // border: per-tap bounds checks (zero padding)
    for (var c = 0u; c < ${l}u; c = c + 1u) {
      let base = (ci0 + c) * ${e.h*e.w}u;
      for (var ky = 0; ky < ${c}; ky = ky + 1) {
        let iy = iy0 + ky;
        if (iy < 0 || iy >= ${e.h}) { continue; }
        let rowBase = base + u32(iy) * ${e.w}u;
        for (var kx = 0; kx < ${c}; kx = kx + 1) {
          let wv = wk[c * ${c*c}u + u32(ky * ${c} + kx)];
          var xv = vec4f(0.0);
          for (var j = 0; j < 4; j = j + 1) {
            let ix = ix0 + j * ${s} + kx;
            if (ix >= 0 && ix < ${e.w}) { xv[j] = src[rowBase + u32(ix)]; }
          }
          acc = fma(xv, vec4f(wv), acc);
        }
      }
    }
  }
  let out = co * ${t}u + u32(oy) * ${e.outW}u + u32(ox0);
  dst[out] = ${$("acc.x")};
  dst[out + 1u] = ${$("acc.y")};
  dst[out + 2u] = ${$("acc.z")};
  dst[out + 3u] = ${$("acc.w")};
}`;return{label:`conv${e.k} ${e.cin}->${e.cout} g${e.groups} @${e.outH}x${e.outW}`,code:v,workgroups:[Math.ceil(i/64),e.cout,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e,r)];// depthwise = spatial, cpg=1
}}(e,r,t));case"se":return[function(e,r={}){var t;let i=e.h*e.w;o(e.c<=2048&&e.cmid<=512,`${e.name}: SE dims exceed shared-memory plan`);let c=/* wgsl */`
${a(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
${u}
var<workgroup> gap : array<f32, ${e.c}>;
var<workgroup> mid : array<f32, ${e.cmid}>;
var<workgroup> scl : array<f32, ${e.c}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li : u32) {
  for (var c = li; c < ${e.c}u; c = c + 256u) {
    var sum = 0.0;
    for (var p = 0u; p < ${i}u; p = p + 1u) { sum = sum + src[c * ${i}u + p]; }
    gap[c] = sum / ${i}.0;
  }
  workgroupBarrier();
  for (var m = li; m < ${e.cmid}u; m = m + 256u) {
    var sum = W(${e.b1Off}u + m);
    for (var c = 0u; c < ${e.c}u; c = c + 1u) {
      sum = fma(gap[c], W(${e.w1Off}u + m * ${e.c}u + c), sum);
    }
    mid[m] = max(sum, 0.0);
  }
  workgroupBarrier();
  for (var c = li; c < ${e.c}u; c = c + 256u) {
    var sum = W(${e.b2Off}u + c);
    for (var m = 0u; m < ${e.cmid}u; m = m + 1u) {
      sum = fma(mid[m], W(${e.w2Off}u + c * ${e.cmid}u + m), sum);
    }
    scl[c] = 1.0 / (1.0 + exp(-sum));
  }
  workgroupBarrier();
  for (var i = li; i < ${e.c*i}u; i = i + 256u) {
    dst[i] = ${(t=`src[i] * scl[i / ${i}u]`,"gelu"===e.act?`gelu1(${t})`:t)};
  }
}`;return{label:`se c${e.c} mid${e.cmid} @${e.h}x${e.w}`,code:c,workgroups:[1,1,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e,r)];case"attn_core":return[// ---------------------------------------------------------------------------
// attn_core — per-head softmax(QKᵀ)V, K then V staged through shared memory.
// The qkv and proj matmuls around it are ordinary pointwise ConvSteps (κ
// folds BN + q-scale into the qkv weights), so this kernel is pure attention:
// one workgroup per head, one thread per query token, score row in registers.
// ---------------------------------------------------------------------------
function(e){let{nTok:r,hd:t,heads:i,c:a}=e,u=t/4,c=r*t/4;o(a===i*t,`${e.name}: c != heads*hd`),o(r<=256&&16*c<=16384,`${e.name}: K/V won't fit shared memory`);// channel-planar addressing: channel o at token n sits at o*nTok + n
let s=/* wgsl */`
@group(0) @binding(0) var<storage, read> qkv : array<f32>;
@group(0) @binding(1) var<storage, read_write> attnOut : array<f32>;
var<workgroup> kv : array<vec4f, ${c}>;   // K, then reused for V; [j][d4]
@compute @workgroup_size(${r})
fn main(@builtin(local_invocation_index) i : u32,
        @builtin(workgroup_id) wid : vec3u) {
  let head = wid.x;
  let qCh = head * ${t}u;                      // q channels [qCh, qCh+hd)
  let kCh = ${a}u + head * ${t}u;
  let vCh = ${2*a}u + head * ${t}u;
  // gather this thread's query row into registers (one-time strided reads)
  var q : array<vec4f, ${u}>;
  for (var d4 = 0u; d4 < ${u}u; d4 = d4 + 1u) {
    q[d4] = vec4f(
      qkv[(qCh + d4 * 4u) * ${r}u + i],
      qkv[(qCh + d4 * 4u + 1u) * ${r}u + i],
      qkv[(qCh + d4 * 4u + 2u) * ${r}u + i],
      qkv[(qCh + d4 * 4u + 3u) * ${r}u + i]);
  }
  for (var t = i; t < ${c}u; t = t + ${r}u) {
    let j = t / ${u}u;
    let d = (t % ${u}u) * 4u;
    kv[t] = vec4f(
      qkv[(kCh + d) * ${r}u + j],
      qkv[(kCh + d + 1u) * ${r}u + j],
      qkv[(kCh + d + 2u) * ${r}u + j],
      qkv[(kCh + d + 3u) * ${r}u + j]);
  }
  workgroupBarrier();
  var p : array<f32, ${r}>;   // row i of the score matrix, private
  var m = -3.0e38;
  for (var j = 0u; j < ${r}u; j = j + 1u) {
    var sv = vec4f(0.0);
    for (var d4 = 0u; d4 < ${u}u; d4 = d4 + 1u) {
      sv = fma(q[d4], kv[j * ${u}u + d4], sv);
    }
    let sc = sv.x + sv.y + sv.z + sv.w;
    p[j] = sc;
    m = max(m, sc);
  }
  var sum = 0.0;
  for (var j = 0u; j < ${r}u; j = j + 1u) {
    let e = exp(p[j] - m);
    p[j] = e;
    sum = sum + e;
  }
  let inv = 1.0 / sum;
  workgroupBarrier();   // everyone done with K before it becomes V
  for (var t = i; t < ${c}u; t = t + ${r}u) {
    let j = t / ${u}u;
    let d = (t % ${u}u) * 4u;
    kv[t] = vec4f(
      qkv[(vCh + d) * ${r}u + j],
      qkv[(vCh + d + 1u) * ${r}u + j],
      qkv[(vCh + d + 2u) * ${r}u + j],
      qkv[(vCh + d + 3u) * ${r}u + j]);
  }
  workgroupBarrier();
  var acc : array<vec4f, ${u}>;
  for (var j = 0u; j < ${r}u; j = j + 1u) {
    let wgt = p[j] * inv;
    for (var d4 = 0u; d4 < ${u}u; d4 = d4 + 1u) {
      acc[d4] = fma(vec4f(wgt), kv[j * ${u}u + d4], acc[d4]);
    }
  }
  // attnOut is channel-planar [head*hd + d][n] — pointwise-conv input layout
  for (var d4 = 0u; d4 < ${u}u; d4 = d4 + 1u) {
    attnOut[(qCh + d4 * 4u) * ${r}u + i] = acc[d4].x;
    attnOut[(qCh + d4 * 4u + 1u) * ${r}u + i] = acc[d4].y;
    attnOut[(qCh + d4 * 4u + 2u) * ${r}u + i] = acc[d4].z;
    attnOut[(qCh + d4 * 4u + 3u) * ${r}u + i] = acc[d4].w;
  }
}`;return{label:`attn.core h${i} n${r}`,code:s,workgroups:[i,1,1],buffers:[{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)];case"head":return[function(e,r={}){let t=e.h*e.w,i=/* wgsl */`
${a(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
var<workgroup> gap : array<f32, ${e.cin}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li : u32) {
  for (var ci = li; ci < ${e.cin}u; ci = ci + 256u) {
    var sum = 0.0;
    for (var p = 0u; p < ${t}u; p = p + 1u) { sum = sum + src[ci * ${t}u + p]; }
    gap[ci] = sum / ${t}.0;
  }
  workgroupBarrier();
  for (var co = li; co < ${e.cout}u; co = co + 256u) {
    var acc = 0.0;
    for (var ci = 0u; ci < ${e.cin}u; ci = ci + 1u) {
      acc = fma(gap[ci], W(${e.wOff}u + ci * ${e.cout}u + co), acc);
    }
    dst[co] = acc;
  }
}`;return{label:`head ${e.cin}->${e.cout}`,code:i,workgroups:[1,1,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e,r)];case"gelu":return[function(e){o(e.n%4==0,`${e.name}: gelu n%4 != 0`);let r=e.n/4,t=/* wgsl */`
@group(0) @binding(0) var<storage, read> src : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;
${u}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${r}u) { return; }
  dst[i] = gelu4(src[i]);
}`;return{label:`gelu n${e.n}`,code:t,workgroups:[Math.ceil(r/64),1,1],buffers:[{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)]}}function v(e,r={}){return e.steps.flatMap((e,t)=>g(e,r,t))}},{"@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],"2Oqph":[function(e,r,t){/**
 * vision_bwd_wgsl — PURE WGSL codegen for the fused MobileCLIP-S0 vision
 * encoder BACKWARD pass: dL/dpixels with all WEIGHTS FROZEN (no dW anywhere).
 * This gradient feeds the Gaussian-splat rasterizer so splats can optimize a
 * canvas toward a text prompt (docs/clip_backward_spec.md).
 *
 * Same pure data→string discipline as vision_wgsl.ts: κ (compile_plan.py
 * --train) emits plan_train.json with a `backward:[...]` reverse step list;
 * this module turns each backward entry into ONE specialized compute dispatch.
 * Every shape/offset is a baked literal. The tiled pointwise matmul is SHARED
 * with the forward (pw_bwd reuses pointwiseTiledMain reading the transposed
 * weights), and the erf/GELU constants are imported so forward and backward
 * cannot drift.
 *
 * Grad-slot convention (κ): grad(activation slot s) = nActSlots + s. A grad
 * slot's FIRST writer overwrites; later writers ADD (`accumulate:true`, set by
 * κ from forward-reader order) — no global zero-fill. Saved activations read
 * for recompute (se input, gelu pre-activation, attn qkv, embed) live in their
 * original activation slots (never freed in train mode).
 *//// <reference types="@webgpu/types" />
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),// ---------------------------------------------------------------------------
// Thin dispatcher — backward kind → emitter (one clean handler each).
// ---------------------------------------------------------------------------
i.export(t,"bwdStepDispatch",()=>l),/** All backward dispatches (loss head + reverse step list), in execution order. */i.export(t,"planBwdDispatches",()=>f);var a=e("./vision_wgsl");// ---------------------------------------------------------------------------
// Shared fragments
// ---------------------------------------------------------------------------
/** gelu'(x) = Φ(x) + x·φ(x), Φ(x)=0.5(1+erf(x/√2)), φ(x)=exp(−x²/2)/√(2π).
 *  Uses the EXACT forward erf4 (imported GELU) so the two never desync. */let u=/* wgsl */`
fn geluGrad4(x : vec4f) -> vec4f {
  let cdf = 0.5 * (vec4f(1.0) + erf4(x * 0.7071067811865476));
  let pdf = 0.3989422804014327 * exp(-0.5 * x * x);   // 1/sqrt(2π)
  return cdf + x * pdf;
}`,o=e=>({kind:"slot",slot:e});function c(e,r){return!!r&&"pw_bwd"===r.kind&&!e.accumulate&&e.dX===r.dY&&e.n===r.cin*r.outH*r.outW}function s(e,r){return!!r&&"pw_bwd"===r.kind&&e.dY===r.dY&&e.n===r.cin*r.outH*r.outW&&r.cout>=r.cin}function n(e){return(1===e.stride||2===e.stride)&&e.groups===e.cin&&e.cout===e.cin&&e.h%4==0&&e.w%4==0&&e.outW>0}function d(e){return 3===e.cin&&64===e.cout&&3===e.k&&2===e.stride&&1===e.pad&&1===e.groups&&256===e.h&&256===e.w&&128===e.outH&&128===e.outW&&!e.accumulate}function l(e,r={}){switch(e.kind){case"loss_bwd":return function(e){let r=e.accumulate?"dx[k] + g":"g",t=/* wgsl */`
@group(0) @binding(0) var<storage, read> e : array<f32>;             // saved embed
@group(0) @binding(1) var<storage, read> t : array<f32>;             // text embedding
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // dL/dembed
var<workgroup> pe : array<f32, 256>;   // partial Σ e\xb2
var<workgroup> pt : array<f32, 256>;   // partial Σ t\xb2
var<workgroup> pd : array<f32, 256>;   // partial Σ e\xb7t
var<workgroup> ne : f32;
var<workgroup> nt : f32;
var<workgroup> cosv : f32;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li : u32) {
  var se = 0.0; var st = 0.0; var sd = 0.0;
  for (var k = li; k < ${e.dim}u; k = k + 256u) {
    let ev = e[k]; let tv = t[k];
    se = se + ev * ev; st = st + tv * tv; sd = sd + ev * tv;
  }
  pe[li] = se; pt[li] = st; pd[li] = sd;
  workgroupBarrier();
  for (var stride = 256u / 2u; stride > 0u; stride = stride >> 1u) {
    if (li < stride) {
      pe[li] = pe[li] + pe[li + stride];
      pt[li] = pt[li] + pt[li + stride];
      pd[li] = pd[li] + pd[li + stride];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    ne = sqrt(max(pe[0], 1e-20));
    nt = sqrt(max(pt[0], 1e-20));
    cosv = pd[0] / (ne * nt);
  }
  workgroupBarrier();
  // d(−cos)/de_k = −( t_k/(|e||t|) − cos\xb7e_k/|e|\xb2 )
  let invET = 1.0 / (ne * nt);
  let cosOverE2 = cosv / (ne * ne);
  for (var k = li; k < ${e.dim}u; k = k + 256u) {
    let g = -(t[k] * invET - cosOverE2 * e[k]);
    dx[k] = ${r};
  }
}`;return{label:`loss_bwd -cos dim${e.dim}`,code:t,workgroups:[1,1,1],buffers:[o(e.embed),{kind:"text"},o(e.dX)]}}(e);case"head_bwd":return function(e,r={}){let t=e.h*e.w,i=e.accumulate?"dx[o] + v":"v",u=/* wgsl */`
${(0,a.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // dEmb [Cout]
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // grad[head src] [Cin][P]
var<workgroup> dgap : array<f32, ${e.cin}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li : u32) {
  // dgap[ci] = Σ_co W[ci][co]\xb7dEmb[co]   (W packed [Cin][Cout], stored orientation)
  for (var ci = li; ci < ${e.cin}u; ci = ci + 256u) {
    var acc = 0.0;
    for (var co = 0u; co < ${e.cout}u; co = co + 1u) {
      acc = fma(W(${e.wOff}u + ci * ${e.cout}u + co), dy[co], acc);
    }
    dgap[ci] = acc / ${t}.0;   // GAP backward: 1/P broadcast
  }
  workgroupBarrier();
  for (var o = li; o < ${e.cin*t}u; o = o + 256u) {
    let v = dgap[o / ${t}u];
    dx[o] = ${i};
  }
}`;return{label:`head_bwd ${e.cout}->${e.cin}${e.accumulate?" +=":""}`,code:u,workgroups:[1,1,1],buffers:[{kind:"weights"},o(e.dY),o(e.dX)]}}(e,r);case"gelu_bwd":return(// ---------------------------------------------------------------------------
// gelu_bwd — dX = dY ⊙ gelu'(x_pre), one thread per quad.
// ---------------------------------------------------------------------------
function(e){(0,a.assertStep)(e.n%4==0,`${e.name}: gelu_bwd n%4 != 0`);let r=e.n/4,t=e.accumulate?"dst[i] + g":"g",i=/* wgsl */`
@group(0) @binding(0) var<storage, read> dy : array<vec4f>;
@group(0) @binding(1) var<storage, read> pre : array<vec4f>;         // saved pre-activation
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${a.GELU}
${u}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${r}u) { return; }
  let g = dy[i] * geluGrad4(pre[i]);
  dst[i] = ${t};
}`;return{label:`gelu_bwd n${e.n}${e.accumulate?" +=":""}`,code:i,workgroups:[Math.ceil(r/64),1,1],buffers:[o(e.dY),o(e.pre),o(e.dX)]}}(e));case"pw_bwd":return(// ---------------------------------------------------------------------------
// pw_bwd — dX = Wᵀ·dY, tiled pointwise kernel over the transposed weights.
// ---------------------------------------------------------------------------
function(e,r={}){let t=e.outH*e.outW;(0,a.assertPointwiseTiles)(e.name,e.cin,e.cout,t,e.wOffT);let i=t/4,u=/* wgsl */`
${(0,a.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;         // dY  [Cin][P4]
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;   // dX  [Cout][P4]
${a.PW_TILE_DECLS}
${(0,a.pointwiseTiledMain)({cin:e.cin,cout:e.cout,P4:i,wOff:e.wOffT,init:()=>"vec4f(0.0)",store:r=>e.accumulate?`dst[(co + ${r}u) * ${i}u + p4] + acc${r}`:`acc${r}`})}`;return{label:`pw_bwd ${e.cin}->${e.cout} @${e.outH}x${e.outW}${e.accumulate?" +=":""}`,code:u,workgroups:[i/8,e.cout/32,1],buffers:[{kind:"weights"},o(e.dY),o(e.dX)]}}(e,r));case"residual_bwd":return(// ---------------------------------------------------------------------------
// residual_bwd — grad[res] (+)= dOut (elementwise vec4 copy/add).
// ---------------------------------------------------------------------------
function(e){(0,a.assertStep)(e.n%4==0,`${e.name}: residual n%4 != 0`);let r=e.n/4,t=e.accumulate?"dst[i] + src[i]":"src[i]",i=/* wgsl */`
@group(0) @binding(0) var<storage, read> src : array<vec4f>;         // dOut
@group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;   // grad[res]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${r}u) { return; }
  dst[i] = ${t};
}`;return{label:`residual_bwd n${e.n}${e.accumulate?" +=":""}`,code:i,workgroups:[Math.ceil(r/64),1,1],buffers:[o(e.dY),o(e.dX)]}}(e));case"spatial_bwd":if(r.stemSpatialBwd&&d(e))return function(e,r={}){(0,a.assertStep)(d(e),`${e.name}: stem spatial_bwd specialization received wrong shape`);let t=e.h*e.w,i=t/4,u=e.outH*e.outW,c=(e,r)=>/* wgsl */`
      {
        var d3 = 0.0;
        if (oxBase + 2u < 128u) {
          d3 = dy[${e} + oxBase + 2u];
        }
        acc = fma(vec4f(W(${r} + 0u)), vec4f(0.0, dy[${e} + oxBase + 1u], 0.0, d3), acc);
        acc = fma(vec4f(W(${r} + 1u)), vec4f(dy[${e} + oxBase], 0.0, dy[${e} + oxBase + 1u], 0.0), acc);
        acc = fma(vec4f(W(${r} + 2u)), vec4f(0.0, dy[${e} + oxBase], 0.0, dy[${e} + oxBase + 1u]), acc);
      }`,s=/* wgsl */`
${(0,a.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // [64][128][128]
@group(0) @binding(2) var<storage, read_write> dx : array<vec4f>;    // [3][256][64 vec4s]

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let ci = gid.y;
  let q = gid.x;
  if (q >= ${i}u) { return; }
  let iy = q / 64u;
  let ix0 = (q - iy * 64u) * 4u;
  let oxBase = ix0 >> 1u;
  var acc = vec4f(0.0);

  if ((iy & 1u) == 0u) {
    let oy = iy >> 1u;
    for (var co = 0u; co < 64u; co = co + 1u) {
      let rowY = co * ${u}u + oy * 128u;
      let wbase = ${e.wOff}u + co * 27u + ci * 9u + 3u; // ky = 1
${c("rowY","wbase")}
    }
  } else {
    for (var co = 0u; co < 64u; co = co + 1u) {
      if (iy < 255u) {
        let oy0 = (iy + 1u) >> 1u;
        let rowY0 = co * ${u}u + oy0 * 128u;
        let wbase0 = ${e.wOff}u + co * 27u + ci * 9u; // ky = 0
${c("rowY0","wbase0")}
      }
      let oy2 = (iy - 1u) >> 1u;
      let rowY2 = co * ${u}u + oy2 * 128u;
      let wbase2 = ${e.wOff}u + co * 27u + ci * 9u + 6u; // ky = 2
${c("rowY2","wbase2")}
    }
  }

  dx[ci * ${i}u + q] = acc;
}`;return{label:"spatial_bwd_stem4 k3s2 3<-64 g1 @256x256",code:s,workgroups:[Math.ceil(i/64),3,1],buffers:[{kind:"weights"},o(e.dY),o(e.dX)]}}(e,r);if("depthwise4"===r.spatialBwdVariant&&n(e))return function(e,r={}){(0,a.assertStep)(n(e),`${e.name}: depthwise4 spatial_bwd received wrong shape`);let t=e.h*e.w,i=t/4,u=e.outH*e.outW,c=e.accumulate?`dx[ci * ${i}u + q] + acc`:"acc",s=1===e.stride?/* wgsl */`
  let oy = ty;
  if (oy < 0 || oy >= ${e.outH}) { continue; }`:/* wgsl */`
  if ((ty & 1) != 0) { continue; }
  let oy = ty >> 1;
  if (oy < 0 || oy >= ${e.outH}) { continue; }`,d=r=>1===e.stride?/* wgsl */`
      {
        let ox${r} = tx${r};
        if (ox${r} >= 0 && ox${r} < ${e.outW}) { y${r} = dy[rowY + u32(ox${r})]; }
      }`:/* wgsl */`
      {
        if ((tx${r} & 1) == 0) {
          let ox${r} = tx${r} >> 1;
          if (ox${r} >= 0 && ox${r} < ${e.outW}) { y${r} = dy[rowY + u32(ox${r})]; }
        }
      }`,l=/* wgsl */`
${(0,a.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // [C][outH][outW]
@group(0) @binding(2) var<storage, read_write> dx : array<vec4f>;    // [C][H*W/4]
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let ci = gid.y;
  let q = gid.x;
  if (q >= ${i}u) { return; }
  let iy = i32(q / ${e.w/4}u);
  let ix0 = i32((q % ${e.w/4}u) * 4u);
  let wbase = ${e.wOff}u + ci * ${e.k*e.k}u;
  let dybase = ci * ${u}u;
  var acc = vec4f(0.0);
  for (var ky = 0; ky < ${e.k}; ky = ky + 1) {
    let ty = iy + ${e.pad} - ky;
${s}
    let rowY = dybase + u32(oy) * ${e.outW}u;
    let rowW = wbase + u32(ky) * ${e.k}u;
    for (var kx = 0; kx < ${e.k}; kx = kx + 1) {
      let kxi = i32(kx);
      let tx0 = ix0 + ${e.pad} - kxi;
      let tx1 = tx0 + 1;
      let tx2 = tx0 + 2;
      let tx3 = tx0 + 3;
      var y0 = 0.0;
      var y1 = 0.0;
      var y2 = 0.0;
      var y3 = 0.0;
${d(0)}
${d(1)}
${d(2)}
${d(3)}
      acc = fma(vec4f(W(rowW + u32(kx))), vec4f(y0, y1, y2, y3), acc);
    }
  }
  dx[ci * ${i}u + q] = ${c};
}`;return{label:`spatial_bwd_dw4 k${e.k}s${e.stride} ${e.cin}<-${e.cout} g${e.groups} @${e.h}x${e.w}${e.accumulate?" +=":""}`,code:l,workgroups:[Math.ceil(i/64),e.cin,1],buffers:[{kind:"weights"},o(e.dY),o(e.dX)]}}(e,r);return(// ---------------------------------------------------------------------------
// spatial_bwd — gather conv backward, one workgroup per input channel, one
// thread per input pixel. Mirror of the forward spatialConv. Stride∈{1,2}
// baked (stride-2 = parity check on iy+pad−ky). Depthwise = cpg=1 special case.
// ---------------------------------------------------------------------------
function(e,r={}){(0,a.assertStep)(1===e.stride||2===e.stride,`${e.name}: stride ${e.stride} not in {1,2}`);let t=e.cin/e.groups,i=e.cout/e.groups;// input channels per group (1 or 3)
(0,a.assertStep)(Number.isInteger(t)&&Number.isInteger(i),`${e.name}: bad groups`);let u=t*e.k*e.k,c=e.h*e.w,s=e.outH*e.outW,n=(r,t)=>1===e.stride?`let ${t} = ${r};`:`if ((${r} & 1) != 0) { continue; } let ${t} = ${r} >> 1;`,d=e.accumulate?"dx[o] + acc":"acc",l=/* wgsl */`
${(0,a.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // [Cout][outH][outW]
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // [Cin][H][W]
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let ci = gid.y;
  let p = gid.x;
  if (p >= ${c}u) { return; }
  let iy = i32(p / ${e.w}u);
  let ix = i32(p % ${e.w}u);
  let grp = ci / ${t}u;
  let ci_local = ci - grp * ${t}u;
  var acc = 0.0;
  for (var col = 0u; col < ${i}u; col = col + 1u) {
    let co = grp * ${i}u + col;
    let wbase = ${e.wOff}u + co * ${u}u + ci_local * ${e.k*e.k}u;
    let dybase = co * ${s}u;
    for (var ky = 0; ky < ${e.k}; ky = ky + 1) {
      let ty = iy + ${e.pad} - ky;
      ${n("ty","oy")}
      if (oy < 0 || oy >= ${e.outH}) { continue; }
      let rowW = wbase + u32(ky) * ${e.k}u;
      let rowY = dybase + u32(oy) * ${e.outW}u;
      for (var kx = 0; kx < ${e.k}; kx = kx + 1) {
        let tx = ix + ${e.pad} - kx;
        ${n("tx","ox")}
        if (ox < 0 || ox >= ${e.outW}) { continue; }
        acc = fma(W(rowW + u32(kx)), dy[rowY + u32(ox)], acc);
      }
    }
  }
  let o = ci * ${c}u + u32(iy) * ${e.w}u + u32(ix);
  dx[o] = ${d};
}`;// one cout's weight footprint
return{label:`spatial_bwd k${e.k}s${e.stride} ${e.cin}<-${e.cout} g${e.groups} @${e.h}x${e.w}${e.accumulate?" +=":""}`,code:l,workgroups:[Math.ceil(c/64),e.cin,1],buffers:[{kind:"weights"},o(e.dY),o(e.dX)]}}(e,r));case"se_bwd":return function(e,r={}){let t=e.h*e.w;(0,a.assertStep)(e.c<=2048&&e.cmid<=512,`${e.name}: SE dims exceed shared-memory plan`);let i=e.accumulate?"dx[i] + v":"v",u=/* wgsl */`
${(0,a.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // grad[se out]
@group(0) @binding(2) var<storage, read> src : array<f32>;           // saved se input
@group(0) @binding(3) var<storage, read_write> dx : array<f32>;      // grad[se in]
// tmp holds gap (steps 1-2) then dL/dpre2 (steps 3-4) — disjoint lifetimes, so
// one array of size c instead of two keeps workgroup memory ≤ 16KB even at c=1024.
var<workgroup> tmp  : array<f32, ${e.c}>;
var<workgroup> mid  : array<f32, ${e.cmid}>;   // relu(pre1)
var<workgroup> scl  : array<f32, ${e.c}>;      // sigmoid gate
var<workgroup> gp1  : array<f32, ${e.cmid}>;   // dL/dpre1
var<workgroup> ggap : array<f32, ${e.c}>;      // dL/dgap
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li : u32) {
  // 1. GAP (recompute) → tmp
  for (var c = li; c < ${e.c}u; c = c + 256u) {
    var sum = 0.0;
    for (var p = 0u; p < ${t}u; p = p + 1u) { sum = sum + src[c * ${t}u + p]; }
    tmp[c] = sum / ${t}.0;
  }
  workgroupBarrier();
  // 2. fc1 + relu (recompute) — reads gap from tmp
  for (var m = li; m < ${e.cmid}u; m = m + 256u) {
    var sum = W(${e.b1Off}u + m);
    for (var c = 0u; c < ${e.c}u; c = c + 1u) {
      sum = fma(tmp[c], W(${e.w1Off}u + m * ${e.c}u + c), sum);
    }
    mid[m] = max(sum, 0.0);
  }
  workgroupBarrier();
  // 3. fc2 + sigmoid (recompute), and dL/dpre2 = (Σ_p dY\xb7x)\xb7σ'(pre2) → tmp (gap dead)
  for (var c = li; c < ${e.c}u; c = c + 256u) {
    var pre2 = W(${e.b2Off}u + c);
    for (var m = 0u; m < ${e.cmid}u; m = m + 1u) {
      pre2 = fma(mid[m], W(${e.w2Off}u + c * ${e.cmid}u + m), pre2);
    }
    let sc = 1.0 / (1.0 + exp(-pre2));
    scl[c] = sc;
    var gscl = 0.0;   // dL/dscl[c] = Σ_p dY[c][p]\xb7x[c][p]
    for (var p = 0u; p < ${t}u; p = p + 1u) {
      gscl = fma(dy[c * ${t}u + p], src[c * ${t}u + p], gscl);
    }
    tmp[c] = gscl * sc * (1.0 - sc);
  }
  workgroupBarrier();
  // 4. dL/dmid = fc2ᵀ\xb7dpre2 ; dL/dpre1 = relu'\xb7dmid  (dpre2 in tmp)
  for (var m = li; m < ${e.cmid}u; m = m + 256u) {
    var gm = 0.0;
    for (var c = 0u; c < ${e.c}u; c = c + 1u) {
      gm = fma(tmp[c], W(${e.w2Off}u + c * ${e.cmid}u + m), gm);
    }
    gp1[m] = select(0.0, gm, mid[m] > 0.0);
  }
  workgroupBarrier();
  // 5. dL/dgap = fc1ᵀ\xb7dpre1
  for (var c = li; c < ${e.c}u; c = c + 256u) {
    var gg = 0.0;
    for (var m = 0u; m < ${e.cmid}u; m = m + 1u) {
      gg = fma(gp1[m], W(${e.w1Off}u + m * ${e.c}u + c), gg);
    }
    ggap[c] = gg;
  }
  workgroupBarrier();
  // 6. dX = dY⊙scale + (1/P)\xb7ggap broadcast
  for (var i = li; i < ${e.c*t}u; i = i + 256u) {
    let c = i / ${t}u;
    let v = dy[i] * scl[c] + ggap[c] / ${t}.0;
    dx[i] = ${i};
  }
}`;return{label:`se_bwd c${e.c} mid${e.cmid} @${e.h}x${e.w}${e.accumulate?" +=":""}`,code:u,workgroups:[1,1,1],buffers:[{kind:"weights"},o(e.dY),o(e.savedSrc),o(e.dX)]}}(e,r);case"attn_core_bwd":return(// ---------------------------------------------------------------------------
// attn_core_bwd — MHSA backward, one workgroup per head, one thread per token.
// Two phases: (1) thread = query i recomputes softmax row from saved qkv,
// writes dQ_i and stashes per-row scalars (max, denom, rowdot) in shared mem;
// (2) thread = key/value j accumulates dV_j = Σ_i p_ij·dO_i and
// dK_j = Σ_i dS_ij·q_i, re-reading q_i / dO_i from global. Only 3·nTok floats
// of shared memory — no nTok×nTok materialization.
// ---------------------------------------------------------------------------
function(e){let{c:r,heads:t,hd:i,nTok:u}=e;(0,a.assertStep)(r===t*i,`${e.name}: c != heads*hd`);let c=/* wgsl */`
@group(0) @binding(0) var<storage, read> qkv : array<f32>;           // saved [3C][nTok] planar
@group(0) @binding(1) var<storage, read> dO : array<f32>;            // grad[attnOut] [C][nTok]
@group(0) @binding(2) var<storage, read_write> dQKV : array<f32>;    // grad[qkv] [3C][nTok]
var<workgroup> mrow : array<f32, ${u}>;   // per-query softmax max
var<workgroup> drow : array<f32, ${u}>;   // per-query softmax denom
var<workgroup> rdot : array<f32, ${u}>;   // per-query Σ_k p_ik\xb7dP_ik
@compute @workgroup_size(${u})
fn main(@builtin(local_invocation_index) tid : u32,
        @builtin(workgroup_id) wid : vec3u) {
  let head = wid.x;
  let qCh = head * ${i}u;              // q channels [qCh, qCh+hd)
  let kCh = ${r}u + head * ${i}u;
  let vCh = ${2*r}u + head * ${i}u;

  // ---- phase 1: thread = query i ----
  let i = tid;
  var qi : array<f32, ${i}>;
  var dOi : array<f32, ${i}>;
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    qi[d]  = qkv[(qCh + d) * ${u}u + i];
    dOi[d] = dO[(qCh + d) * ${u}u + i];
  }
  var p  : array<f32, ${u}>;
  var dP : array<f32, ${u}>;
  var mx = -3.0e38;
  for (var j = 0u; j < ${u}u; j = j + 1u) {
    var sc = 0.0;
    for (var d = 0u; d < ${i}u; d = d + 1u) { sc = fma(qi[d], qkv[(kCh + d) * ${u}u + j], sc); }
    p[j] = sc;
    mx = max(mx, sc);
  }
  var den = 0.0;
  for (var j = 0u; j < ${u}u; j = j + 1u) { let e = exp(p[j] - mx); p[j] = e; den = den + e; }
  let inv = 1.0 / den;
  var rd = 0.0;
  for (var j = 0u; j < ${u}u; j = j + 1u) {
    p[j] = p[j] * inv;                                  // p_ij
    var dpj = 0.0;
    for (var d = 0u; d < ${i}u; d = d + 1u) { dpj = fma(dOi[d], qkv[(vCh + d) * ${u}u + j], dpj); }
    dP[j] = dpj;                                        // dP_ij = Σ_d dO_i\xb7V_j
    rd = fma(p[j], dpj, rd);                            // Σ_k p_ik\xb7dP_ik
  }
  // dQ_i = Σ_j dS_ij\xb7K_j,  dS_ij = p_ij(dP_ij − rd)
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    var acc = 0.0;
    for (var j = 0u; j < ${u}u; j = j + 1u) {
      let ds = p[j] * (dP[j] - rd);
      acc = fma(ds, qkv[(kCh + d) * ${u}u + j], acc);
    }
    dQKV[(qCh + d) * ${u}u + i] = acc;
  }
  mrow[i] = mx; drow[i] = den; rdot[i] = rd;
  workgroupBarrier();

  // ---- phase 2: thread = key/value token j ----
  let j = tid;
  var kj : array<f32, ${i}>;
  var vj : array<f32, ${i}>;
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    kj[d] = qkv[(kCh + d) * ${u}u + j];
    vj[d] = qkv[(vCh + d) * ${u}u + j];
  }
  var dV : array<f32, ${i}>;
  var dK : array<f32, ${i}>;
  for (var d = 0u; d < ${i}u; d = d + 1u) { dV[d] = 0.0; dK[d] = 0.0; }
  for (var ii = 0u; ii < ${u}u; ii = ii + 1u) {
    // recompute p_ij and dP_ij for this (query ii, key j)
    var sc = 0.0;
    var dpij = 0.0;
    for (var d = 0u; d < ${i}u; d = d + 1u) {
      let qv = qkv[(qCh + d) * ${u}u + ii];
      sc = fma(qv, kj[d], sc);
      dpij = fma(dO[(qCh + d) * ${u}u + ii], vj[d], dpij);
    }
    let pij = exp(sc - mrow[ii]) / drow[ii];
    let dsij = pij * (dpij - rdot[ii]);
    for (var d = 0u; d < ${i}u; d = d + 1u) {
      dV[d] = fma(pij, dO[(qCh + d) * ${u}u + ii], dV[d]);
      dK[d] = fma(dsij, qkv[(qCh + d) * ${u}u + ii], dK[d]);
    }
  }
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    dQKV[(kCh + d) * ${u}u + j] = dK[d];
    dQKV[(vCh + d) * ${u}u + j] = dV[d];
  }
}`;return{label:`attn_core_bwd h${t} n${u}`,code:c,workgroups:[t,1,1],buffers:[o(e.savedQkv),o(e.dY),o(e.dX)]}}(e))}}function f(e,r={}){if(!r.fuseGeluBwdIntoPw&&!r.fuseResidualBwdIntoPw)return e.backward.map(e=>l(e,r));let t=[];for(let i=0;i<e.backward.length;i++){let n=e.backward[i],d=e.backward[i+1];if(r.fuseResidualBwdIntoPw&&"residual_bwd"===n.kind&&s(n,d)){t.push(function(e,r,t={}){(0,a.assertStep)(s(e,r),`${e.name}: cannot fuse residual copy into ${r.name}`);let i=r.outH*r.outW;(0,a.assertPointwiseTiles)(r.name,r.cin,r.cout,i,r.wOffT);let u=i/4,c=/* wgsl */`
${(0,a.weightsDecl)(0,t.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;            // dY [Cin][P4]
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;      // dX [Cout][P4]
@group(0) @binding(3) var<storage, read_write> resDst : array<vec4f>;   // residual grad [Cin][P4]
${a.PW_TILE_DECLS}
${(0,a.pointwiseTiledMain)({cin:r.cin,cout:r.cout,P4:u,wOff:r.wOffT,init:()=>"vec4f(0.0)",store:e=>r.accumulate?`dst[(co + ${e}u) * ${u}u + p4] + acc${e}`:`acc${e}`,extraStore:t=>{let i=`co + ${t}u`,a=`(${i}) * ${u}u + p4`,o=e.accumulate?`resDst[${a}] + src[${a}]`:`src[${a}]`;return`if (${i} < ${r.cin}u) { resDst[${a}] = ${o}; }`}})}`;return{label:`pw_bwd+residual ${r.cin}->${r.cout} @${r.outH}x${r.outW}${r.accumulate?" +=":""}${e.accumulate?" res+=":""}`,code:c,workgroups:[u/8,r.cout/32,1],buffers:[{kind:"weights"},o(r.dY),o(r.dX),o(e.dX)]}}(n,d,r)),i+=1;continue}if(r.fuseGeluBwdIntoPw&&"gelu_bwd"===n.kind&&c(n,d)){t.push(function(e,r,t={}){(0,a.assertStep)(c(e,r),`${e.name}: cannot fuse GELU backward into ${r.name}`);let i=r.outH*r.outW;(0,a.assertPointwiseTiles)(r.name,r.cin,r.cout,i,r.wOffT);let s=i/4,n=/* wgsl */`
${(0,a.weightsDecl)(0,t.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;         // dY before GELU derivative [Cin][P4]
@group(0) @binding(2) var<storage, read> pre : array<vec4f>;         // saved GELU pre-activation [Cin][P4]
@group(0) @binding(3) var<storage, read_write> dst : array<vec4f>;   // dX [Cout][P4]
${a.GELU}
${u}
${a.PW_TILE_DECLS}
${(0,a.pointwiseTiledMain)({cin:r.cin,cout:r.cout,P4:s,wOff:r.wOffT,init:()=>"vec4f(0.0)",loadSrc:e=>`src[${e}] * geluGrad4(pre[${e}])`,store:e=>r.accumulate?`dst[(co + ${e}u) * ${s}u + p4] + acc${e}`:`acc${e}`})}`;return{label:`pw_bwd+gelu ${r.cin}->${r.cout} @${r.outH}x${r.outW}${r.accumulate?" +=":""}`,code:n,workgroups:[s/8,r.cout/32,1],buffers:[{kind:"weights"},o(e.dY),{kind:"slot",slot:e.pre},o(r.dX)]}}(n,d,r)),i+=1;continue}t.push(l(n,r))}return t}},{"./vision_wgsl":"oFDUc","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],"3CXuq":[function(e,r,t){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"loadClipTrainAssets",()=>o);var a=e("./fetch_progress");let u="https://huggingface.co/Nbardy/nff-clip-splat-weights/resolve/main/";async function o(e){let r=[];for(let t of function(){let e=new URLSearchParams(location.search).get("modelBase"),r=[];return e&&r.push(e.endsWith("/")?e:`${e}/`),["localhost","127.0.0.1"].includes(location.hostname)&&(// Parcel serves /models as an HTML fallback; try it first for repo-root
// static servers, then try the known local model server used in dev.
r.push("/models/mobileclip_s0/"),r.push(`http://${location.hostname}:8799/models/mobileclip_s0/`)),r.push(u),[...new Set(r)]}()){let i=t===u?" from HF":t.includes(":8799/")?" from local model server":t.startsWith("http")?` from ${t}`:"";try{e(`fetching CLIP plan${i}...`);let r=await c(t),u=await (0,a.fetchArrayBufferWithProgress)(t+"weights_train.bin",r=>{e((0,a.formatProgress)(`loading CLIP weights${i}`,r))});return{plan:r,weights:new Float32Array(u),base:t}}catch(e){r.push(`${t}: ${e?.message??e}`)}}throw Error(`could not load CLIP train assets:
${r.join("\n")}`)}async function c(e){let r=await fetch(e+"plan_train.json");if(!r.ok)throw Error(`plan_train.json fetch ${r.status}`);let t=await r.text(),i=t.trimStart().slice(0,80);if(i.startsWith("<!DOCTYPE")||i.startsWith("<html"))throw Error("plan_train.json returned HTML instead of JSON");try{return JSON.parse(t)}catch(e){throw Error(`plan_train.json invalid JSON: ${e?.message??e}`)}}},{"./fetch_progress":"8QffC","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],"8QffC":[function(e,r,t){/**
 * fetch_progress — stream a fetch() body and report download progress so the
 * pages can show a loading bar for the 82 MB CLIP weights. Content-Length gives
 * the total (a CORS-safelisted header, so it's readable even from the HF
 * cross-origin fetch); the stream reader gives bytes-so-far. Falls back to a
 * plain arrayBuffer() if the body isn't streamable, and reports total=0 when
 * Content-Length is absent (caller then shows an indeterminate readout).
 *
 * Shared by src/splat_page.ts and src/splat3d_page.ts (same loader).
 */var i=e("@parcel/transformer-js/src/esmodule-helpers.js");async function a(e,r,t){let i=performance.now(),a=await fetch(e,t);if(!a.ok)throw Error(`fetch ${a.status} ${e}`);let u=Number(a.headers.get("content-length"))||0,o=a.body?.getReader();if(!o){let e=await a.arrayBuffer();return r({received:e.byteLength,total:e.byteLength||u,elapsedMs:performance.now()-i}),e}let c=[],s=0;for(;;){let{done:e,value:t}=await o.read();if(e)break;c.push(t),r({received:s+=t.byteLength,total:u,elapsedMs:performance.now()-i})}let n=new Uint8Array(s),d=0;for(let e of c)n.set(e,d),d+=e.byteLength;return n.buffer}function u(e,r){let t=(r.received/1e6).toFixed(1),i=(r.elapsedMs/1e3).toFixed(1),a=r.elapsedMs>0?(r.received/(r.elapsedMs/1e3)/1e6).toFixed(1):"0.0";if(r.total>0){let u=Math.min(100,Math.round(r.received/r.total*100)),o=(r.total/1e6).toFixed(0),c=Math.round(u/100*16),s="█".repeat(c)+"░".repeat(16-c);return`${e}  [${s}] ${u}%  \xb7  ${t}/${o} MB  \xb7  ${i}s  \xb7  ${a} MB/s`}return`${e}  ${t} MB  \xb7  ${i}s  \xb7  ${a} MB/s`}i.defineInteropFlag(t),i.export(t,"fetchArrayBufferWithProgress",()=>a),/** Compact text bar: "loading CLIP weights [████░░░░] 52% · 43/82 MB · 3.1s · 14 MB/s". */i.export(t,"formatProgress",()=>u)},{"@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}]},[],null,"parcelRequire924a")//# sourceMappingURL=splat3d.79422fb4.js.map
;
//# sourceMappingURL=splat3d.79422fb4.js.map
