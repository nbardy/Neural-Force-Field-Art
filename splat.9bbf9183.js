!function(e,r,t,i){var a="u">typeof globalThis?globalThis:"u">typeof self?self:"u">typeof window?window:"u">typeof global?global:{},u="function"==typeof a[t]&&a[t],o=u.i||{},c=u.cache||{},s="u">typeof module&&"function"==typeof module.require&&module.require.bind(module);function n(r,o){if(!c[r]){if(!e[r]){if(i[r])return i[r];var d="function"==typeof a[t]&&a[t];if(!o&&d)return d(r,!0);if(u)return u(r,!0);if(s&&"string"==typeof r)return s(r);var l=Error("Cannot find module '"+r+"'");throw l.code="MODULE_NOT_FOUND",l}p.resolve=function(t){var i=e[r][1][t];return null!=i?i:t},p.cache={};var f=c[r]=new n.Module(r);e[r][0].call(f.exports,p,f,f.exports,a)}return c[r].exports;function p(e){var r=p.resolve(e);if(!1===r)return{};if(Array.isArray(r)){var t={__esModule:!0};return r.forEach(function(e){var r=e[0],i=e[1],a=e[2]||e[0],u=n(i);"*"===r?Object.keys(u).forEach(function(e){"default"===e||"__esModule"===e||Object.prototype.hasOwnProperty.call(t,e)||Object.defineProperty(t,e,{enumerable:!0,get:function(){return u[e]}})}):"*"===a?Object.defineProperty(t,r,{enumerable:!0,value:u}):Object.defineProperty(t,r,{enumerable:!0,get:function(){return"default"===a?u.__esModule?u.default:u:u[a]}})}),t}return n(r)}}n.isParcelRequire=!0,n.Module=function(e){this.id=e,this.bundle=n,this.require=s,this.exports={}},n.modules=e,n.cache=c,n.parent=u,n.distDir=void 0,n.publicUrl=void 0,n.devServer=void 0,n.i=o,n.register=function(r,t){e[r]=[function(e,r){r.exports=t},{}]},Object.defineProperty(n,"root",{get:function(){return a[t]}}),a[t]=n;for(var d=0;d<r.length;d++)n(r[d])}({bbLCC:[function(e,r,t,i){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(t),a.export(t,"ADAM_UNIFORM_BYTES",()=>u),a.export(t,"adamShader",()=>o),a.export(t,"DEFAULT_LRS",()=>c),a.export(t,"DEFAULT_HYPER",()=>s);let u=32;function o(){return`
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
`}let c={mean:.01,logScale:.005,theta:.005,color:.005,opacity:.005},s={beta1:.9,beta2:.999,eps:1e-8}},{"@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],"3gu6C":[function(e,r,t,i){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(t),a.export(t,"VisionEncoder",()=>n),a.export(t,"VisionTrainer",()=>f);var u=e("./vision_wgsl"),o=e("./vision_bwd_wgsl");function c(e,r,t){if(r.length!==t)throw Error(`${e}: weights blob ${r.length} scalars != plan ${t}`)}function s(e,r){return r?e.beginComputePass({timestampWrites:r}):e.beginComputePass()}class n{static async create(e,r,t,i={}){return c("vision",t,r.weightsFloats),new n(e,r,t,await d(e,r,i))}constructor(e,r,t,i){this.dispatches=[],this.device=e,this.plan=r,this.weightsBuffer=e.createBuffer({size:t.byteLength,usage:136}),e.queue.writeBuffer(this.weightsBuffer,0,t),this.slotBuffers=r.slots.map((r,t)=>e.createBuffer({label:`clip-slot-${t}`,size:4*r,usage:140})),this.dispatches=i.map(({spec:e,pipeline:r})=>({pipeline:r,workgroups:e.workgroups,label:e.label,bind:this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:e.buffers.map((e,r)=>({binding:r,resource:{buffer:"weights"===e.kind?this.weightsBuffer:"slot"===e.kind?this.slotBuffers[e.slot]:(()=>{throw Error("vision: forward encoder received a 'text' binding (loss head belongs to VisionTrainer)")})()}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}writeInput(e){let[r,t,i]=this.plan.inputShape;if(e.length!==r*t*i)throw Error(`vision: input ${e.length} != ${r*t*i}`);this.device.queue.writeBuffer(this.inputBuffer,0,e)}encode(e,r=this.dispatches.length,t){let i=s(e,t);for(let e=0;e<r;e++){let r=this.dispatches[e];i.setPipeline(r.pipeline),i.setBindGroup(0,r.bind),i.dispatchWorkgroups(...r.workgroups)}i.end()}run(){let e=this.device.createCommandEncoder();this.encode(e),this.device.queue.submit([e.finish()])}stepDispatchCounts(){return this.plan.steps.map(()=>1)}}async function d(e,r,t={}){return l(e,(0,u.planDispatches)(r,t))}async function l(e,r){let t=[];for(let i of r){e.pushErrorScope("validation");let r=e.createShaderModule({code:i.code}),a=e.createComputePipeline({layout:"auto",compute:{module:r,entryPoint:"main"}}),u=await e.popErrorScope();if(u)throw Error(`vision: pipeline '${i.label}' invalid: ${u.message}
${i.code}`);t.push({spec:i,pipeline:a})}return t}class f{static async create(e,r,t,i={}){c("vision",t,r.weightsFloats);let a=(0,u.planDispatches)(r,i),s=(0,o.planBwdDispatches)(r,i),n=await l(e,[...a,...s]);return new f(e,r,t,n,a.length)}constructor(e,r,t,i,a){this.dispatches=[],this.device=e,this.plan=r,this.fwdCount=a,this.weightsBuffer=e.createBuffer({size:t.byteLength,usage:136}),e.queue.writeBuffer(this.weightsBuffer,0,t),this.textBuffer=e.createBuffer({size:4*r.textDim,usage:136}),this.slotBuffers=r.slots.map((r,t)=>e.createBuffer({label:`clip-tslot-${t}`,size:4*r,usage:140}));let u=e=>"weights"===e.kind?this.weightsBuffer:"text"===e.kind?this.textBuffer:this.slotBuffers[e.slot];this.dispatches=i.map(({spec:e,pipeline:r})=>({pipeline:r,workgroups:e.workgroups,label:e.label,bind:this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:e.buffers.map((e,r)=>({binding:r,resource:{buffer:u(e)}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}get inputGradBuffer(){return this.slotBuffers[this.plan.inputGradSlot]}writeInput(e){let[r,t,i]=this.plan.inputShape;if(e.length!==r*t*i)throw Error(`vision: input ${e.length} != ${r*t*i}`);this.device.queue.writeBuffer(this.inputBuffer,0,e)}writeText(e){if(e.length!==this.plan.textDim)throw Error(`vision: text ${e.length} != ${this.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffer,0,e)}encode(e,r={}){let t=!1===r.backward?this.fwdCount:this.dispatches.length,i=s(e,r.timestampWrites);for(let e=0;e<t;e++){let r=this.dispatches[e];i.setPipeline(r.pipeline),i.setBindGroup(0,r.bind),i.dispatchWorkgroups(...r.workgroups)}i.end()}encodeForward(e,r){let t=s(e,r);for(let e=0;e<this.fwdCount;e++){let r=this.dispatches[e];t.setPipeline(r.pipeline),t.setBindGroup(0,r.bind),t.dispatchWorkgroups(...r.workgroups)}t.end()}encodeBackward(e,r){let t=s(e,r);for(let e=this.fwdCount;e<this.dispatches.length;e++){let r=this.dispatches[e];t.setPipeline(r.pipeline),t.setBindGroup(0,r.bind),t.dispatchWorkgroups(...r.workgroups)}t.end()}run(e={}){let r=this.device.createCommandEncoder();this.encode(r,e),this.device.queue.submit([r.finish()])}destroy(){for(let e of(this.weightsBuffer.destroy(),this.textBuffer.destroy(),this.slotBuffers))e.destroy()}}},{"./vision_wgsl":"jaeEI","./vision_bwd_wgsl":"6k6vK","@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],jaeEI:[function(e,r,t,i){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(t),a.export(t,"weightsDecl",()=>u),a.export(t,"GELU",()=>o),a.export(t,"assertStep",()=>c),a.export(t,"PW_TILE_DECLS",()=>s),a.export(t,"PW_RECT8X16_TILE_DECLS",()=>n),a.export(t,"pointwiseTiledMain",()=>d),a.export(t,"assertPointwiseTiles",()=>l),a.export(t,"pointwiseFusedGelu",()=>g),a.export(t,"stepDispatches",()=>v),a.export(t,"planDispatches",()=>w);let u=(e,r="f32")=>"f16"===r?`enable f16;
@group(0) @binding(${e}) var<storage, read> weights : array<vec4<f16>>;
fn W(i : u32) -> f32 { return f32(weights[i >> 2u][i & 3u]); }
fn W4(i : u32) -> vec4f { return vec4f(weights[i]); }`:`@group(0) @binding(${e}) var<storage, read> weights : array<vec4f>;
fn W(i : u32) -> f32 { return weights[i >> 2u][i & 3u]; }
fn W4(i : u32) -> vec4f { return weights[i]; }`,o=`
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
`;function c(e,r){if(!e)throw Error(`vision_wgsl: ${r}`)}let s=`
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 256>;`,n=`
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 512>;`;function d(e){let r=r=>e.extraStore?`
  ${e.extraStore(r)}`:"",t=e.loadSrc?e.loadSrc("srcIndex"):"src[srcIndex]";return`
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
}`}function l(e,r,t,i,a){c(i%32==0&&t%32==0&&r%32==0,`${e}: tiled pointwise needs P%32==0 && cout%32==0 && cin%32==0 (got P=${i} cin=${r} cout=${t})`),c(a%4==0,`${e}: wOff not 16B-aligned`)}function f(e,r){if("rect8x16"!==e.pointwiseTileVariant)return!1;let t=e.pointwiseTileSteps;return!t?.size||void 0!==r&&t.has(r)}function p(e,r,t,i,a){l(e,r,t,i,a),c(t%64==0,`${e}: rect8x16 pointwise needs cout%64==0 (got cout=${t})`)}function $(e){let r=r=>e.extraStore?`
  ${e.extraStore(r)}`:"";return`
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
}`}function g(e,r,t={},i){if(f(t,i))return function(e,r,t={}){c("pointwise"===e.variant,`${e.name}: fused GELU only supports pointwise conv`),c("none"===e.act,`${e.name}: fused GELU expects split train-mode conv`),c(null===e.residual&&null===e.layerScaleOff,`${e.name}: fused GELU does not support residual epilogues`),c(r.src===e.dst,`${e.name}: fused GELU src slot ${r.src} != conv dst ${e.dst}`);let i=e.outH*e.outW;c(r.n===e.cout*i,`${e.name}: fused GELU n=${r.n} != cout*P=${e.cout*i}`),p(e.name,e.cin,e.cout,i,e.wOff);let a=i/4,s=`
${u(0,t.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> geluDst : array<vec4f>;
${o}
${n}
${$({cin:e.cin,cout:e.cout,P4:a,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:e=>`acc${e}`,extraStore:e=>`geluDst[(co + ${e}u) * ${a}u + p4] = gelu4(acc${e});`})}`;return{label:`pw+gelu rect8x16 ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:s,workgroups:[a/8,e.cout/64,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst},{kind:"slot",slot:r.dst}]}}(e,r,t);c("pointwise"===e.variant,`${e.name}: fused GELU only supports pointwise conv`),c("none"===e.act,`${e.name}: fused GELU expects split train-mode conv`),c(null===e.residual&&null===e.layerScaleOff,`${e.name}: fused GELU does not support residual epilogues`),c(r.src===e.dst,`${e.name}: fused GELU src slot ${r.src} != conv dst ${e.dst}`);let a=e.outH*e.outW;c(r.n===e.cout*a,`${e.name}: fused GELU n=${r.n} != cout*P=${e.cout*a}`),l(e.name,e.cin,e.cout,a,e.wOff);let v=a/4,w=`
${u(0,t.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> geluDst : array<vec4f>;
${o}
${s}
${d({cin:e.cin,cout:e.cout,P4:v,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:e=>`acc${e}`,extraStore:e=>`geluDst[(co + ${e}u) * ${v}u + p4] = gelu4(acc${e});`})}`;return{label:`pw+gelu ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:w,workgroups:[v/8,e.cout/32,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst},{kind:"slot",slot:r.dst}]}}function v(e,r={},t){switch(e.kind){case"conv":return function(e,r={},t){switch(e.variant){case"pointwise":return[function(e,r={},t){if(f(r,t))return function(e,r={}){let t=e.outH*e.outW;p(e.name,e.cin,e.cout,t,e.wOff);let i=t/4,a=null!==e.residual;c(null!==e.layerScaleOff===a,`${e.name}: layerScale without residual`);let s=[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}];a&&s.push({kind:"slot",slot:e.residual});let d=`
${u(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${a?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${o}
${n}
${$({cin:e.cin,cout:e.cout,P4:i,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:r=>{let t="gelu"===e.act?`gelu4(acc${r})`:`acc${r}`;return a?`res[(co + ${r}u) * ${i}u + p4] + vec4f(W(${e.layerScaleOff}u + co + ${r}u)) * ${t}`:t}})}`;return{label:`pw rect8x16 ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:d,workgroups:[i/8,e.cout/64,1],buffers:s}}(e,r);let i=e.outH*e.outW;l(e.name,e.cin,e.cout,i,e.wOff);let a=i/4,g=null!==e.residual;c(null!==e.layerScaleOff===g,`${e.name}: layerScale without residual`);let v=[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}];g&&v.push({kind:"slot",slot:e.residual});let w=`
${u(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${g?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${o}
${s}
${d({cin:e.cin,cout:e.cout,P4:a,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:r=>{let t="gelu"===e.act?`gelu4(acc${r})`:`acc${r}`;return g?`res[(co + ${r}u) * ${a}u + p4] + vec4f(W(${e.layerScaleOff}u + co + ${r}u)) * ${t}`:t}})}`;return{label:`pw ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:w,workgroups:[a/8,e.cout/32,1],buffers:v}}(e,r,t)];case"depthwise":case"general":return[function(e,r={}){c(null===e.residual&&null===e.layerScaleOff,`${e.name}: spatial conv never carries residual in this plan`),c(e.outW%4==0,`${e.name}: spatial tiling needs outW%4==0`);let t=e.outH*e.outW,i=t/4,a=e.k,s=e.stride,n=e.pad,d=3*s+a,l=e.cin/e.groups,f=e.cout/e.groups,p=l*a*a;c(Number.isInteger(l)&&Number.isInteger(f),`${e.name}: bad groups`),c(p<=64,`${e.name}: weight tile ${p} exceeds one staging round`);let $=r=>"gelu"===e.act?`gelu1(${r})`:r,g=[];for(let r=0;r<l;r++){g.push(`    { let base = (ci0 + ${r}u) * ${e.h*e.w}u;`);for(let t=0;t<a;t++){g.push(`      { let rowBase = base + u32(iy0 + ${t}) * ${e.w}u + u32(ix0);`);for(let e=0;e<d;e++)g.push(`        let r${e} = src[rowBase + ${e}u];`);for(let e=0;e<a;e++)g.push(`        acc = fma(vec4f(r${e}, r${s+e}, r${2*s+e}, r${3*s+e}), vec4f(wk[${r*a*a+t*a+e}u]), acc);`);g.push("      }")}g.push("    }")}let v=`
${u(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
${o}
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
  if (iy0 >= 0 && iy0 + ${a} <= ${e.h} && ix0 >= 0 && ix0 + ${d} <= ${e.w}) {
    // interior: every tap in bounds, unchecked unrolled register loads
${g.join("\n")}
  } else {
    // border: per-tap bounds checks (zero padding)
    for (var c = 0u; c < ${l}u; c = c + 1u) {
      let base = (ci0 + c) * ${e.h*e.w}u;
      for (var ky = 0; ky < ${a}; ky = ky + 1) {
        let iy = iy0 + ky;
        if (iy < 0 || iy >= ${e.h}) { continue; }
        let rowBase = base + u32(iy) * ${e.w}u;
        for (var kx = 0; kx < ${a}; kx = kx + 1) {
          let wv = wk[c * ${a*a}u + u32(ky * ${a} + kx)];
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
}`;return{label:`conv${e.k} ${e.cin}->${e.cout} g${e.groups} @${e.outH}x${e.outW}`,code:v,workgroups:[Math.ceil(i/64),e.cout,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e,r)]}}(e,r,t);case"se":return[function(e,r={}){let t,i=e.h*e.w;c(e.c<=2048&&e.cmid<=512,`${e.name}: SE dims exceed shared-memory plan`);let a=`
${u(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
${o}
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
}`;return{label:`se c${e.c} mid${e.cmid} @${e.h}x${e.w}`,code:a,workgroups:[1,1,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e,r)];case"attn_core":return[function(e){let{nTok:r,hd:t,heads:i,c:a}=e,u=t/4,o=r*t/4;c(a===i*t,`${e.name}: c != heads*hd`),c(r<=256&&16*o<=16384,`${e.name}: K/V won't fit shared memory`);let s=`
@group(0) @binding(0) var<storage, read> qkv : array<f32>;
@group(0) @binding(1) var<storage, read_write> attnOut : array<f32>;
var<workgroup> kv : array<vec4f, ${o}>;   // K, then reused for V; [j][d4]
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
  for (var t = i; t < ${o}u; t = t + ${r}u) {
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
  for (var t = i; t < ${o}u; t = t + ${r}u) {
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
  // attnOut is channel-planar [head*hd + d][n] \u{2014} pointwise-conv input layout
  for (var d4 = 0u; d4 < ${u}u; d4 = d4 + 1u) {
    attnOut[(qCh + d4 * 4u) * ${r}u + i] = acc[d4].x;
    attnOut[(qCh + d4 * 4u + 1u) * ${r}u + i] = acc[d4].y;
    attnOut[(qCh + d4 * 4u + 2u) * ${r}u + i] = acc[d4].z;
    attnOut[(qCh + d4 * 4u + 3u) * ${r}u + i] = acc[d4].w;
  }
}`;return{label:`attn.core h${i} n${r}`,code:s,workgroups:[i,1,1],buffers:[{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)];case"head":return[function(e,r={}){let t=e.h*e.w,i=`
${u(0,r.weightPrecision)}
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
}`;return{label:`head ${e.cin}->${e.cout}`,code:i,workgroups:[1,1,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e,r)];case"gelu":let i,a;return[(c(e.n%4==0,`${e.name}: gelu n%4 != 0`),i=e.n/4,a=`
@group(0) @binding(0) var<storage, read> src : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;
${o}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${i}u) { return; }
  dst[i] = gelu4(src[i]);
}`,{label:`gelu n${e.n}`,code:a,workgroups:[Math.ceil(i/64),1,1],buffers:[{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]})]}}function w(e,r={}){return e.steps.flatMap((e,t)=>v(e,r,t))}},{"@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],"6k6vK":[function(e,r,t,i){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(t),a.export(t,"bwdStepDispatch",()=>f),a.export(t,"planBwdDispatches",()=>p);var u=e("./vision_wgsl");let o=`
fn geluGrad4(x : vec4f) -> vec4f {
  let cdf = 0.5 * (vec4f(1.0) + erf4(x * 0.7071067811865476));
  let pdf = 0.3989422804014327 * exp(-0.5 * x * x);   // 1/sqrt(2\u{3C0})
  return cdf + x * pdf;
}`,c=e=>({kind:"slot",slot:e});function s(e,r){return!!r&&"pw_bwd"===r.kind&&!e.accumulate&&e.dX===r.dY&&e.n===r.cin*r.outH*r.outW}function n(e,r){return!!r&&"pw_bwd"===r.kind&&e.dY===r.dY&&e.n===r.cin*r.outH*r.outW&&r.cout>=r.cin}function d(e){return(1===e.stride||2===e.stride)&&e.groups===e.cin&&e.cout===e.cin&&e.h%4==0&&e.w%4==0&&e.outW>0}function l(e){return 3===e.cin&&64===e.cout&&3===e.k&&2===e.stride&&1===e.pad&&1===e.groups&&256===e.h&&256===e.w&&128===e.outH&&128===e.outW&&!e.accumulate}function f(e,r={}){switch(e.kind){case"loss_bwd":let t,i;return t=e.accumulate?"dx[k] + g":"g",i=`
@group(0) @binding(0) var<storage, read> e : array<f32>;             // saved embed
@group(0) @binding(1) var<storage, read> t : array<f32>;             // text embedding
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // dL/dembed
var<workgroup> pe : array<f32, 256>;   // partial \u{3A3} e\xb2
var<workgroup> pt : array<f32, 256>;   // partial \u{3A3} t\xb2
var<workgroup> pd : array<f32, 256>;   // partial \u{3A3} e\xb7t
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
  // d(\u{2212}cos)/de_k = \u{2212}( t_k/(|e||t|) \u{2212} cos\xb7e_k/|e|\xb2 )
  let invET = 1.0 / (ne * nt);
  let cosOverE2 = cosv / (ne * ne);
  for (var k = li; k < ${e.dim}u; k = k + 256u) {
    let g = -(t[k] * invET - cosOverE2 * e[k]);
    dx[k] = ${t};
  }
}`,{label:`loss_bwd -cos dim${e.dim}`,code:i,workgroups:[1,1,1],buffers:[c(e.embed),{kind:"text"},c(e.dX)]};case"head_bwd":return function(e,r={}){let t=e.h*e.w,i=e.accumulate?"dx[o] + v":"v",a=`
${(0,u.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // dEmb [Cout]
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // grad[head src] [Cin][P]
var<workgroup> dgap : array<f32, ${e.cin}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li : u32) {
  // dgap[ci] = \u{3A3}_co W[ci][co]\xb7dEmb[co]   (W packed [Cin][Cout], stored orientation)
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
}`;return{label:`head_bwd ${e.cout}->${e.cin}${e.accumulate?" +=":""}`,code:a,workgroups:[1,1,1],buffers:[{kind:"weights"},c(e.dY),c(e.dX)]}}(e,r);case"gelu_bwd":let a,s,n;return(0,u.assertStep)(e.n%4==0,`${e.name}: gelu_bwd n%4 != 0`),a=e.n/4,s=e.accumulate?"dst[i] + g":"g",n=`
@group(0) @binding(0) var<storage, read> dy : array<vec4f>;
@group(0) @binding(1) var<storage, read> pre : array<vec4f>;         // saved pre-activation
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${u.GELU}
${o}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${a}u) { return; }
  let g = dy[i] * geluGrad4(pre[i]);
  dst[i] = ${s};
}`,{label:`gelu_bwd n${e.n}${e.accumulate?" +=":""}`,code:n,workgroups:[Math.ceil(a/64),1,1],buffers:[c(e.dY),c(e.pre),c(e.dX)]};case"pw_bwd":return function(e,r={}){let t=e.outH*e.outW;(0,u.assertPointwiseTiles)(e.name,e.cin,e.cout,t,e.wOffT);let i=t/4,a=`
${(0,u.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;         // dY  [Cin][P4]
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;   // dX  [Cout][P4]
${u.PW_TILE_DECLS}
${(0,u.pointwiseTiledMain)({cin:e.cin,cout:e.cout,P4:i,wOff:e.wOffT,init:()=>"vec4f(0.0)",store:r=>e.accumulate?`dst[(co + ${r}u) * ${i}u + p4] + acc${r}`:`acc${r}`})}`;return{label:`pw_bwd ${e.cin}->${e.cout} @${e.outH}x${e.outW}${e.accumulate?" +=":""}`,code:a,workgroups:[i/8,e.cout/32,1],buffers:[{kind:"weights"},c(e.dY),c(e.dX)]}}(e,r);case"residual_bwd":let p,$,g;return(0,u.assertStep)(e.n%4==0,`${e.name}: residual n%4 != 0`),p=e.n/4,$=e.accumulate?"dst[i] + src[i]":"src[i]",g=`
@group(0) @binding(0) var<storage, read> src : array<vec4f>;         // dOut
@group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;   // grad[res]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${p}u) { return; }
  dst[i] = ${$};
}`,{label:`residual_bwd n${e.n}${e.accumulate?" +=":""}`,code:g,workgroups:[Math.ceil(p/64),1,1],buffers:[c(e.dY),c(e.dX)]};case"spatial_bwd":if(r.stemSpatialBwd&&l(e))return function(e,r={}){(0,u.assertStep)(l(e),`${e.name}: stem spatial_bwd specialization received wrong shape`);let t=e.h*e.w/4,i=e.outH*e.outW,a=(e,r)=>`
      {
        var d3 = 0.0;
        if (oxBase + 2u < 128u) {
          d3 = dy[${e} + oxBase + 2u];
        }
        acc = fma(vec4f(W(${r} + 0u)), vec4f(0.0, dy[${e} + oxBase + 1u], 0.0, d3), acc);
        acc = fma(vec4f(W(${r} + 1u)), vec4f(dy[${e} + oxBase], 0.0, dy[${e} + oxBase + 1u], 0.0), acc);
        acc = fma(vec4f(W(${r} + 2u)), vec4f(0.0, dy[${e} + oxBase], 0.0, dy[${e} + oxBase + 1u]), acc);
      }`;return{label:"spatial_bwd_stem4 k3s2 3<-64 g1 @256x256",code:`
${(0,u.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // [64][128][128]
@group(0) @binding(2) var<storage, read_write> dx : array<vec4f>;    // [3][256][64 vec4s]

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let ci = gid.y;
  let q = gid.x;
  if (q >= ${t}u) { return; }
  let iy = q / 64u;
  let ix0 = (q - iy * 64u) * 4u;
  let oxBase = ix0 >> 1u;
  var acc = vec4f(0.0);

  if ((iy & 1u) == 0u) {
    let oy = iy >> 1u;
    for (var co = 0u; co < 64u; co = co + 1u) {
      let rowY = co * ${i}u + oy * 128u;
      let wbase = ${e.wOff}u + co * 27u + ci * 9u + 3u; // ky = 1
${a("rowY","wbase")}
    }
  } else {
    for (var co = 0u; co < 64u; co = co + 1u) {
      if (iy < 255u) {
        let oy0 = (iy + 1u) >> 1u;
        let rowY0 = co * ${i}u + oy0 * 128u;
        let wbase0 = ${e.wOff}u + co * 27u + ci * 9u; // ky = 0
${a("rowY0","wbase0")}
      }
      let oy2 = (iy - 1u) >> 1u;
      let rowY2 = co * ${i}u + oy2 * 128u;
      let wbase2 = ${e.wOff}u + co * 27u + ci * 9u + 6u; // ky = 2
${a("rowY2","wbase2")}
    }
  }

  dx[ci * ${t}u + q] = acc;
}`,workgroups:[Math.ceil(t/64),3,1],buffers:[{kind:"weights"},c(e.dY),c(e.dX)]}}(e,r);if("depthwise4"===r.spatialBwdVariant&&d(e))return function(e,r={}){(0,u.assertStep)(d(e),`${e.name}: depthwise4 spatial_bwd received wrong shape`);let t=e.h*e.w/4,i=e.outH*e.outW,a=e.accumulate?`dx[ci * ${t}u + q] + acc`:"acc",o=1===e.stride?`
  let oy = ty;
  if (oy < 0 || oy >= ${e.outH}) { continue; }`:`
  if ((ty & 1) != 0) { continue; }
  let oy = ty >> 1;
  if (oy < 0 || oy >= ${e.outH}) { continue; }`,s=r=>1===e.stride?`
      {
        let ox${r} = tx${r};
        if (ox${r} >= 0 && ox${r} < ${e.outW}) { y${r} = dy[rowY + u32(ox${r})]; }
      }`:`
      {
        if ((tx${r} & 1) == 0) {
          let ox${r} = tx${r} >> 1;
          if (ox${r} >= 0 && ox${r} < ${e.outW}) { y${r} = dy[rowY + u32(ox${r})]; }
        }
      }`,n=`
${(0,u.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // [C][outH][outW]
@group(0) @binding(2) var<storage, read_write> dx : array<vec4f>;    // [C][H*W/4]
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let ci = gid.y;
  let q = gid.x;
  if (q >= ${t}u) { return; }
  let iy = i32(q / ${e.w/4}u);
  let ix0 = i32((q % ${e.w/4}u) * 4u);
  let wbase = ${e.wOff}u + ci * ${e.k*e.k}u;
  let dybase = ci * ${i}u;
  var acc = vec4f(0.0);
  for (var ky = 0; ky < ${e.k}; ky = ky + 1) {
    let ty = iy + ${e.pad} - ky;
${o}
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
${s(0)}
${s(1)}
${s(2)}
${s(3)}
      acc = fma(vec4f(W(rowW + u32(kx))), vec4f(y0, y1, y2, y3), acc);
    }
  }
  dx[ci * ${t}u + q] = ${a};
}`;return{label:`spatial_bwd_dw4 k${e.k}s${e.stride} ${e.cin}<-${e.cout} g${e.groups} @${e.h}x${e.w}${e.accumulate?" +=":""}`,code:n,workgroups:[Math.ceil(t/64),e.cin,1],buffers:[{kind:"weights"},c(e.dY),c(e.dX)]}}(e,r);return function(e,r={}){(0,u.assertStep)(1===e.stride||2===e.stride,`${e.name}: stride ${e.stride} not in {1,2}`);let t=e.cin/e.groups,i=e.cout/e.groups;(0,u.assertStep)(Number.isInteger(t)&&Number.isInteger(i),`${e.name}: bad groups`);let a=t*e.k*e.k,o=e.h*e.w,s=e.outH*e.outW,n=(r,t)=>1===e.stride?`let ${t} = ${r};`:`if ((${r} & 1) != 0) { continue; } let ${t} = ${r} >> 1;`,d=e.accumulate?"dx[o] + acc":"acc",l=`
${(0,u.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // [Cout][outH][outW]
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // [Cin][H][W]
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let ci = gid.y;
  let p = gid.x;
  if (p >= ${o}u) { return; }
  let iy = i32(p / ${e.w}u);
  let ix = i32(p % ${e.w}u);
  let grp = ci / ${t}u;
  let ci_local = ci - grp * ${t}u;
  var acc = 0.0;
  for (var col = 0u; col < ${i}u; col = col + 1u) {
    let co = grp * ${i}u + col;
    let wbase = ${e.wOff}u + co * ${a}u + ci_local * ${e.k*e.k}u;
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
  let o = ci * ${o}u + u32(iy) * ${e.w}u + u32(ix);
  dx[o] = ${d};
}`;return{label:`spatial_bwd k${e.k}s${e.stride} ${e.cin}<-${e.cout} g${e.groups} @${e.h}x${e.w}${e.accumulate?" +=":""}`,code:l,workgroups:[Math.ceil(o/64),e.cin,1],buffers:[{kind:"weights"},c(e.dY),c(e.dX)]}}(e,r);case"se_bwd":return function(e,r={}){let t=e.h*e.w;(0,u.assertStep)(e.c<=2048&&e.cmid<=512,`${e.name}: SE dims exceed shared-memory plan`);let i=e.accumulate?"dx[i] + v":"v",a=`
${(0,u.weightsDecl)(0,r.weightPrecision)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // grad[se out]
@group(0) @binding(2) var<storage, read> src : array<f32>;           // saved se input
@group(0) @binding(3) var<storage, read_write> dx : array<f32>;      // grad[se in]
// tmp holds gap (steps 1-2) then dL/dpre2 (steps 3-4) \u{2014} disjoint lifetimes, so
// one array of size c instead of two keeps workgroup memory \u{2264} 16KB even at c=1024.
var<workgroup> tmp  : array<f32, ${e.c}>;
var<workgroup> mid  : array<f32, ${e.cmid}>;   // relu(pre1)
var<workgroup> scl  : array<f32, ${e.c}>;      // sigmoid gate
var<workgroup> gp1  : array<f32, ${e.cmid}>;   // dL/dpre1
var<workgroup> ggap : array<f32, ${e.c}>;      // dL/dgap
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li : u32) {
  // 1. GAP (recompute) \u{2192} tmp
  for (var c = li; c < ${e.c}u; c = c + 256u) {
    var sum = 0.0;
    for (var p = 0u; p < ${t}u; p = p + 1u) { sum = sum + src[c * ${t}u + p]; }
    tmp[c] = sum / ${t}.0;
  }
  workgroupBarrier();
  // 2. fc1 + relu (recompute) \u{2014} reads gap from tmp
  for (var m = li; m < ${e.cmid}u; m = m + 256u) {
    var sum = W(${e.b1Off}u + m);
    for (var c = 0u; c < ${e.c}u; c = c + 1u) {
      sum = fma(tmp[c], W(${e.w1Off}u + m * ${e.c}u + c), sum);
    }
    mid[m] = max(sum, 0.0);
  }
  workgroupBarrier();
  // 3. fc2 + sigmoid (recompute), and dL/dpre2 = (\u{3A3}_p dY\xb7x)\xb7\u{3C3}'(pre2) \u{2192} tmp (gap dead)
  for (var c = li; c < ${e.c}u; c = c + 256u) {
    var pre2 = W(${e.b2Off}u + c);
    for (var m = 0u; m < ${e.cmid}u; m = m + 1u) {
      pre2 = fma(mid[m], W(${e.w2Off}u + c * ${e.cmid}u + m), pre2);
    }
    let sc = 1.0 / (1.0 + exp(-pre2));
    scl[c] = sc;
    var gscl = 0.0;   // dL/dscl[c] = \u{3A3}_p dY[c][p]\xb7x[c][p]
    for (var p = 0u; p < ${t}u; p = p + 1u) {
      gscl = fma(dy[c * ${t}u + p], src[c * ${t}u + p], gscl);
    }
    tmp[c] = gscl * sc * (1.0 - sc);
  }
  workgroupBarrier();
  // 4. dL/dmid = fc2\u{1D40}\xb7dpre2 ; dL/dpre1 = relu'\xb7dmid  (dpre2 in tmp)
  for (var m = li; m < ${e.cmid}u; m = m + 256u) {
    var gm = 0.0;
    for (var c = 0u; c < ${e.c}u; c = c + 1u) {
      gm = fma(tmp[c], W(${e.w2Off}u + c * ${e.cmid}u + m), gm);
    }
    gp1[m] = select(0.0, gm, mid[m] > 0.0);
  }
  workgroupBarrier();
  // 5. dL/dgap = fc1\u{1D40}\xb7dpre1
  for (var c = li; c < ${e.c}u; c = c + 256u) {
    var gg = 0.0;
    for (var m = 0u; m < ${e.cmid}u; m = m + 1u) {
      gg = fma(gp1[m], W(${e.w1Off}u + m * ${e.c}u + c), gg);
    }
    ggap[c] = gg;
  }
  workgroupBarrier();
  // 6. dX = dY\u{2299}scale + (1/P)\xb7ggap broadcast
  for (var i = li; i < ${e.c*t}u; i = i + 256u) {
    let c = i / ${t}u;
    let v = dy[i] * scl[c] + ggap[c] / ${t}.0;
    dx[i] = ${i};
  }
}`;return{label:`se_bwd c${e.c} mid${e.cmid} @${e.h}x${e.w}${e.accumulate?" +=":""}`,code:a,workgroups:[1,1,1],buffers:[{kind:"weights"},c(e.dY),c(e.savedSrc),c(e.dX)]}}(e,r);case"attn_core_bwd":return function(e){let{c:r,heads:t,hd:i,nTok:a}=e;(0,u.assertStep)(r===t*i,`${e.name}: c != heads*hd`);let o=`
@group(0) @binding(0) var<storage, read> qkv : array<f32>;           // saved [3C][nTok] planar
@group(0) @binding(1) var<storage, read> dO : array<f32>;            // grad[attnOut] [C][nTok]
@group(0) @binding(2) var<storage, read_write> dQKV : array<f32>;    // grad[qkv] [3C][nTok]
var<workgroup> mrow : array<f32, ${a}>;   // per-query softmax max
var<workgroup> drow : array<f32, ${a}>;   // per-query softmax denom
var<workgroup> rdot : array<f32, ${a}>;   // per-query \u{3A3}_k p_ik\xb7dP_ik
@compute @workgroup_size(${a})
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
    qi[d]  = qkv[(qCh + d) * ${a}u + i];
    dOi[d] = dO[(qCh + d) * ${a}u + i];
  }
  var p  : array<f32, ${a}>;
  var dP : array<f32, ${a}>;
  var mx = -3.0e38;
  for (var j = 0u; j < ${a}u; j = j + 1u) {
    var sc = 0.0;
    for (var d = 0u; d < ${i}u; d = d + 1u) { sc = fma(qi[d], qkv[(kCh + d) * ${a}u + j], sc); }
    p[j] = sc;
    mx = max(mx, sc);
  }
  var den = 0.0;
  for (var j = 0u; j < ${a}u; j = j + 1u) { let e = exp(p[j] - mx); p[j] = e; den = den + e; }
  let inv = 1.0 / den;
  var rd = 0.0;
  for (var j = 0u; j < ${a}u; j = j + 1u) {
    p[j] = p[j] * inv;                                  // p_ij
    var dpj = 0.0;
    for (var d = 0u; d < ${i}u; d = d + 1u) { dpj = fma(dOi[d], qkv[(vCh + d) * ${a}u + j], dpj); }
    dP[j] = dpj;                                        // dP_ij = \u{3A3}_d dO_i\xb7V_j
    rd = fma(p[j], dpj, rd);                            // \u{3A3}_k p_ik\xb7dP_ik
  }
  // dQ_i = \u{3A3}_j dS_ij\xb7K_j,  dS_ij = p_ij(dP_ij \u{2212} rd)
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    var acc = 0.0;
    for (var j = 0u; j < ${a}u; j = j + 1u) {
      let ds = p[j] * (dP[j] - rd);
      acc = fma(ds, qkv[(kCh + d) * ${a}u + j], acc);
    }
    dQKV[(qCh + d) * ${a}u + i] = acc;
  }
  mrow[i] = mx; drow[i] = den; rdot[i] = rd;
  workgroupBarrier();

  // ---- phase 2: thread = key/value token j ----
  let j = tid;
  var kj : array<f32, ${i}>;
  var vj : array<f32, ${i}>;
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    kj[d] = qkv[(kCh + d) * ${a}u + j];
    vj[d] = qkv[(vCh + d) * ${a}u + j];
  }
  var dV : array<f32, ${i}>;
  var dK : array<f32, ${i}>;
  for (var d = 0u; d < ${i}u; d = d + 1u) { dV[d] = 0.0; dK[d] = 0.0; }
  for (var ii = 0u; ii < ${a}u; ii = ii + 1u) {
    // recompute p_ij and dP_ij for this (query ii, key j)
    var sc = 0.0;
    var dpij = 0.0;
    for (var d = 0u; d < ${i}u; d = d + 1u) {
      let qv = qkv[(qCh + d) * ${a}u + ii];
      sc = fma(qv, kj[d], sc);
      dpij = fma(dO[(qCh + d) * ${a}u + ii], vj[d], dpij);
    }
    let pij = exp(sc - mrow[ii]) / drow[ii];
    let dsij = pij * (dpij - rdot[ii]);
    for (var d = 0u; d < ${i}u; d = d + 1u) {
      dV[d] = fma(pij, dO[(qCh + d) * ${a}u + ii], dV[d]);
      dK[d] = fma(dsij, qkv[(qCh + d) * ${a}u + ii], dK[d]);
    }
  }
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    dQKV[(kCh + d) * ${a}u + j] = dK[d];
    dQKV[(vCh + d) * ${a}u + j] = dV[d];
  }
}`;return{label:`attn_core_bwd h${t} n${a}`,code:o,workgroups:[t,1,1],buffers:[c(e.savedQkv),c(e.dY),c(e.dX)]}}(e)}}function p(e,r={}){if(!r.fuseGeluBwdIntoPw&&!r.fuseResidualBwdIntoPw)return e.backward.map(e=>f(e,r));let t=[];for(let i=0;i<e.backward.length;i++){let a=e.backward[i],d=e.backward[i+1];if(r.fuseResidualBwdIntoPw&&"residual_bwd"===a.kind&&n(a,d)){t.push(function(e,r,t={}){(0,u.assertStep)(n(e,r),`${e.name}: cannot fuse residual copy into ${r.name}`);let i=r.outH*r.outW;(0,u.assertPointwiseTiles)(r.name,r.cin,r.cout,i,r.wOffT);let a=i/4,o=`
${(0,u.weightsDecl)(0,t.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;            // dY [Cin][P4]
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;      // dX [Cout][P4]
@group(0) @binding(3) var<storage, read_write> resDst : array<vec4f>;   // residual grad [Cin][P4]
${u.PW_TILE_DECLS}
${(0,u.pointwiseTiledMain)({cin:r.cin,cout:r.cout,P4:a,wOff:r.wOffT,init:()=>"vec4f(0.0)",store:e=>r.accumulate?`dst[(co + ${e}u) * ${a}u + p4] + acc${e}`:`acc${e}`,extraStore:t=>{let i=`co + ${t}u`,u=`(${i}) * ${a}u + p4`,o=e.accumulate?`resDst[${u}] + src[${u}]`:`src[${u}]`;return`if (${i} < ${r.cin}u) { resDst[${u}] = ${o}; }`}})}`;return{label:`pw_bwd+residual ${r.cin}->${r.cout} @${r.outH}x${r.outW}${r.accumulate?" +=":""}${e.accumulate?" res+=":""}`,code:o,workgroups:[a/8,r.cout/32,1],buffers:[{kind:"weights"},c(r.dY),c(r.dX),c(e.dX)]}}(a,d,r)),i+=1;continue}if(r.fuseGeluBwdIntoPw&&"gelu_bwd"===a.kind&&s(a,d)){t.push(function(e,r,t={}){(0,u.assertStep)(s(e,r),`${e.name}: cannot fuse GELU backward into ${r.name}`);let i=r.outH*r.outW;(0,u.assertPointwiseTiles)(r.name,r.cin,r.cout,i,r.wOffT);let a=i/4,n=`
${(0,u.weightsDecl)(0,t.weightPrecision)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;         // dY before GELU derivative [Cin][P4]
@group(0) @binding(2) var<storage, read> pre : array<vec4f>;         // saved GELU pre-activation [Cin][P4]
@group(0) @binding(3) var<storage, read_write> dst : array<vec4f>;   // dX [Cout][P4]
${u.GELU}
${o}
${u.PW_TILE_DECLS}
${(0,u.pointwiseTiledMain)({cin:r.cin,cout:r.cout,P4:a,wOff:r.wOffT,init:()=>"vec4f(0.0)",loadSrc:e=>`src[${e}] * geluGrad4(pre[${e}])`,store:e=>r.accumulate?`dst[(co + ${e}u) * ${a}u + p4] + acc${e}`:`acc${e}`})}`;return{label:`pw_bwd+gelu ${r.cin}->${r.cout} @${r.outH}x${r.outW}${r.accumulate?" +=":""}`,code:n,workgroups:[a/8,r.cout/32,1],buffers:[{kind:"weights"},c(e.dY),{kind:"slot",slot:e.pre},c(r.dX)]}}(a,d,r)),i+=1;continue}t.push(f(a,r))}return t}},{"./vision_wgsl":"jaeEI","@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],j8tuj:[function(e,r,t,i){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(t),a.export(t,"loadClipTrainAssets",()=>c);var u=e("./fetch_progress");let o="https://huggingface.co/Nbardy/nff-clip-splat-weights/resolve/main/";async function c(e){var r,t;let i,a,c=[];for(let n of(i=new URLSearchParams(location.search).get("modelBase"),a=[],i&&a.push((r=i).endsWith("/")?r:`${r}/`),["localhost","127.0.0.1"].includes(location.hostname)&&(a.push("/models/mobileclip_s0/"),a.push(`http://${location.hostname}:8799/models/mobileclip_s0/`)),a.push(o),[...new Set(a)])){let r=(t=n)===o?" from HF":t.includes(":8799/")?" from local model server":t.startsWith("http")?` from ${t}`:"";try{e(`fetching CLIP plan${r}...`);let t=await s(n),i=await (0,u.fetchArrayBufferWithProgress)(n+"weights_train.bin",t=>{e((0,u.formatProgress)(`loading CLIP weights${r}`,t))});return{plan:t,weights:new Float32Array(i),base:n}}catch(e){c.push(`${n}: ${e?.message??e}`)}}throw Error(`could not load CLIP train assets:
${c.join("\n")}`)}async function s(e){let r=await fetch(e+"plan_train.json");if(!r.ok)throw Error(`plan_train.json fetch ${r.status}`);let t=await r.text(),i=t.trimStart().slice(0,80);if(i.startsWith("<!DOCTYPE")||i.startsWith("<html"))throw Error("plan_train.json returned HTML instead of JSON");try{return JSON.parse(t)}catch(e){throw Error(`plan_train.json invalid JSON: ${e?.message??e}`)}}},{"./fetch_progress":"cabmD","@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],cabmD:[function(e,r,t,i){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");async function u(e,r,t){let i=performance.now(),a=await fetch(e,t);if(!a.ok)throw Error(`fetch ${a.status} ${e}`);let u=Number(a.headers.get("content-length"))||0,o=a.body?.getReader();if(!o){let e=await a.arrayBuffer();return r({received:e.byteLength,total:e.byteLength||u,elapsedMs:performance.now()-i}),e}let c=[],s=0;for(;;){let{done:e,value:t}=await o.read();if(e)break;c.push(t),r({received:s+=t.byteLength,total:u,elapsedMs:performance.now()-i})}let n=new Uint8Array(s),d=0;for(let e of c)n.set(e,d),d+=e.byteLength;return n.buffer}function o(e,r){let t=(r.received/1e6).toFixed(1),i=(r.elapsedMs/1e3).toFixed(1),a=r.elapsedMs>0?(r.received/(r.elapsedMs/1e3)/1e6).toFixed(1):"0.0";if(r.total>0){let u=Math.min(100,Math.round(r.received/r.total*100)),o=(r.total/1e6).toFixed(0),c=Math.round(u/100*16),s="█".repeat(c)+"░".repeat(16-c);return`${e}  [${s}] ${u}%  \xb7  ${t}/${o} MB  \xb7  ${i}s  \xb7  ${a} MB/s`}return`${e}  ${t} MB  \xb7  ${i}s  \xb7  ${a} MB/s`}a.defineInteropFlag(t),a.export(t,"fetchArrayBufferWithProgress",()=>u),a.export(t,"formatProgress",()=>o)},{"@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}]},[],"parcelRequire924a",{});
//# sourceMappingURL=splat.9bbf9183.js.map
