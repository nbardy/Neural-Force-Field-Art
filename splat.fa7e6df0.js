!// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles
function(e,t,i,r,a){/* eslint-disable no-undef */var s="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof self?self:"undefined"!=typeof window?window:"undefined"!=typeof global?global:{},n="function"==typeof s[r]&&s[r],o=n.cache||{},d="undefined"!=typeof module&&"function"==typeof module.require&&module.require.bind(module);function l(t,i){if(!o[t]){if(!e[t]){// if we cannot find the module within our internal map or
// cache jump to the current global require ie. the last bundle
// that was added to the page.
var a="function"==typeof s[r]&&s[r];if(!i&&a)return a(t,!0);// If there are other bundles on this page the require from the
// previous one is saved to 'previousRequire'. Repeat this as
// many times as there are bundles until the module is found or
// we exhaust the require chain.
if(n)return n(t,!0);// Try the node require function if it exists.
if(d&&"string"==typeof t)return d(t);var u=Error("Cannot find module '"+t+"'");throw u.code="MODULE_NOT_FOUND",u}p.resolve=function(i){var r=e[t][1][i];return null!=r?r:i},p.cache={};var c=o[t]=new l.Module(t);e[t][0].call(c.exports,p,c,c.exports,this)}return o[t].exports;function p(e){var t=p.resolve(e);return!1===t?{}:l(t)}}l.isParcelRequire=!0,l.Module=function(e){this.id=e,this.bundle=l,this.exports={}},l.modules=e,l.cache=o,l.parent=n,l.register=function(t,i){e[t]=[function(e,t){t.exports=i},{}]},Object.defineProperty(l,"root",{get:function(){return s[r]}}),s[r]=l;for(var u=0;u<t.length;u++)l(t[u]);if(i){// Expose entry point to Node, AMD or browser globals
// Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
var c=l(i);"object"==typeof exports&&"undefined"!=typeof module?module.exports=c:"function"==typeof define&&define.amd?define(function(){return c}):a&&(this[a]=c)}}({"7i9mK":[function(e,t,i){let r,a,s,n,o,d,l;/**
 * splat_page — the browser wrap for the prompt→splats optimizer (Task #7,
 * phase 2). The differentiable core (src/splat/optimize.ts → SplatOptimizer) is
 * DONE and verified headless; this file only wires it to the DOM on ONE
 * GPUDevice we create here: WebGPU boot + canvas context, in-browser CLIP text
 * encoding (transformers.js), a storage-buffer blit shader, and the rAF
 * optimize loop.
 *
 * ── How to run the page ──────────────────────────────────────────────────────
 *   1. Regenerate the vision train-weights (once; gitignored under models/):
 *        uv run --with onnx --with numpy python tools/clip/compile_plan.py --train
 *      → models/mobileclip_s0/{plan_train.json, weights_train.bin (82 MB)}
 *   2. Build the page (relative asset URLs so it serves under /dist/):
 *        npx parcel build --no-scope-hoist --public-url ./ src/splat.html
 *   3. Serve the repo root (so the page, the 82 MB weights, and everything are
 *      same-origin — the text model + tokenizer + transformers.js load from the
 *      HF hub / jsdelivr CDN at runtime, cached by the browser):
 *        node tools/splat/serve.mjs            # → http://localhost:8799
 *   4. Open http://localhost:8799/dist/splat.html
 *
 * The puppeteer acceptance gate drives exactly this:
 *        node tools/splat/page_smoke.mjs
 *
 * ── Why the weights are fetched, not bundled ─────────────────────────────────
 * models/ is gitignored and outside src/, so parcel neither serves nor bundles
 * it (bundling 82 MB through parcel is a non-starter). The page fetch()es the
 * plan (JSON) + weights (arrayBuffer) from /models/… on the static server that
 * hosts /dist. Only the CLIP VISION train-weights need local serving; the text
 * model + tokenizer come from the HF hub via transformers.js.
 *//// <reference types="@webgpu/types" />
var u=e("./splat/optimize"),c=e("./splat/model_assets");let p={gpu:!!navigator.gpu,ready:!1,running:!1,step:0,cos:null,initialCos:null,error:null,phase:"boot"};window.__splat=p;// ── DOM ──────────────────────────────────────────────────────────────────────
let g=document.getElementById("splat"),h=document.getElementById("prompt"),m=document.getElementById("optimize"),f=document.getElementById("nudge"),y=document.getElementById("reset"),v=document.getElementById("readout"),x=document.getElementById("notice");function b(e){x.textContent=e}function w(e){p.error=e,p.phase="error",b(e),v.textContent="—",// eslint-disable-next-line no-console
console.error("[splat_page]",e)}function $(){p.step=o?o.stepCount:0;let e=[`step ${p.step}`];if(null!==p.cos){let t=p.initialCos??p.cos,i=p.cos-t;e.push(`cos ${p.cos.toFixed(4)}`),e.push(`init ${t.toFixed(4)}`),e.push(`Δ ${i>=0?"+":""}${i.toFixed(4)}`)}p.phase&&"run"!==p.phase&&e.push(`(${p.phase})`),v.textContent=e.join("  \xb7  ")}let _=1,P=null,B=/* wgsl */`
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
  // one oversized triangle covering the whole clip volume (no vertex buffer)
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  return vec4<f32>(p[vi], 0.0, 1.0);
}

@group(0) @binding(0) var<storage, read> img : array<f32>;

@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
  // raster.image is NCHW planar [3][256][256], img[c*HW + y*256 + x].
  // Framebuffer origin is top-left (y down); raster row 0 is the top row (matches
  // the headless PNG dumps), so no Y flip.
  let x : u32 = u32(pos.x);
  let y : u32 = u32(pos.y);
  let HW : u32 = 65536u;
  let i : u32 = y * 256u + x;
  return vec4<f32>(img[i], img[HW + i], img[2u * HW + i], 1.0);
}
`;async function S(){r.pushErrorScope("validation");let e=r.createShaderModule({code:B});d=r.createRenderPipeline({layout:"auto",vertex:{module:e,entryPoint:"vs"},fragment:{module:e,entryPoint:"fs",targets:[{format:l}]},primitive:{topology:"triangle-list"}});let t=await r.popErrorScope();if(t)throw Error(`blit pipeline invalid: ${t.message}`)}function C(){P=r.createBindGroup({layout:d.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:o.raster.image}}]})}// Hide the specifier behind a Function-constructor indirection so the BUNDLER
// leaves it alone and the BROWSER does a genuine native dynamic import of the
// CDN URL. A plain `import(TF_URL)` gets rewritten into a parcel module helper
// that would try to resolve the URL as a local bundle.
let R=Function("u","return import(u)"),E=null,I=null;async function T(e){if(I)return;let t=await R("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm");t.env.allowRemoteModels=!0;let i="Nbardy/nff-clip-splat-weights",r=t=>{if("progress"===t.status&&t.total){let i=Math.round(t.progress??t.loaded/t.total*100),r=Math.round(i/100*16),a="█".repeat(r)+"░".repeat(16-r);e?.(`loading text encoder  [${a}] ${i}%  \xb7  ${(t.loaded/1e6).toFixed(1)}/${(t.total/1e6).toFixed(0)} MB`)}};// self-hosted alongside the vision weights
E=await t.AutoTokenizer.from_pretrained(i,{progress_callback:r}),I=await t.CLIPTextModelWithProjection.from_pretrained(i,{dtype:"fp16",device:"wasm",session_options:{graphOptimizationLevel:"basic"},progress_callback:r})}async function G(e){await T();let t=await E(e,{padding:"max_length",max_length:77,truncation:!0}),i=await I(t),r=i.text_embeds.data,a=new Float32Array(512);for(let e=0;e<512;e++)a[e]=r[e];return a}// ── Optimize loop ─────────────────────────────────────────────────────────────
let A=null,k=0,M=!1,q=!1;async function D(){if(A&&!M&&!q){M=!0;try{let e=await o.currentEmbedding(),t=(0,u.cosine)(e,A);p.cos=t,null===p.initialCos&&(p.initialCos=t),$()}finally{M=!1}}}function W(){p.running&&A&&(// 2 optimize steps/frame keeps the page responsive; each step is one submit
// (raster fwd → CLIP fwd+loss+bwd → raster bwd → Adam). At LEGIBLE_G≈12K
// splats a step is cheap, so the loop stays smooth.
o.step(),o.step(),k+=2,p.step=o.stepCount,k>=14&&(k=0,D())),function(){if(!P)return;let e=r.createCommandEncoder(),t=e.beginRenderPass({colorAttachments:[{view:a.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});t.setPipeline(d),t.setBindGroup(0,P),t.draw(3),t.end(),r.queue.submit([e.finish()])}(),requestAnimationFrame(W)}async function F(){if(!p.ready)return;let e=h.value.trim()||"a photo of a cat";m.disabled=!0,p.phase="encoding",p.running=!1,b("encoding prompt (first use downloads the text model — slow)…"),$();try{let t=await G(e);A=t,o.setPrompt(t);// Baseline cos on the CURRENT splats — this is the "initial" the gate checks
// the run rises above.
let i=await o.currentEmbedding();p.initialCos=(0,u.cosine)(i,t),p.cos=p.initialCos,k=0,b(""),p.phase="run",p.running=!0,$()}catch(e){w(`text encode failed: ${e?.message??e}`)}finally{m.disabled=!1}}async function O(){if(!p.ready)return;p.running=!1,A=null,p.cos=null,p.initialCos=null,p.phase="reset",_+=1;let e=o;o=await (0,u.SplatOptimizer).create(r,s,n,{seed:_}),e.destroy(),C(),await o.renderImage(),h.value="",p.step=0,b(""),$()}async function L(){if(!p.ready||q)return;q=!0;let e=p.running;p.running=!1,p.phase="nudge",f.disabled=!0,_+=1,$();try{if(await o.nudge({seed:_}),await o.renderImage(),k=0,A){let e=await o.currentEmbedding();p.cos=(0,u.cosine)(e,A)}p.phase=e&&A?"run":"idle",p.running=e&&!!A,b(""),$()}catch(e){w(`nudge failed: ${e?.message??e}`)}finally{q=!1,f.disabled=!1}}// ── Boot ─────────────────────────────────────────────────────────────────────
async function j(){if(!navigator.gpu){w("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled."),m.disabled=!0,f.disabled=!0,y.disabled=!0;return}p.phase="adapter";let e=await navigator.gpu.requestAdapter();if(!e){w("no WebGPU adapter available.");return}r=await e.requestDevice(),r.addEventListener?.("uncapturederror",e=>{// eslint-disable-next-line no-console
console.error("[webgpu]",e.error?.message??e.error)}),a=g.getContext("webgpu"),l=navigator.gpu.getPreferredCanvasFormat(),a.configure({device:r,format:l,alphaMode:"opaque"}),p.phase="weights";try{let e=await (0,c.loadClipTrainAssets)(e=>{v.textContent=e});s=e.plan,n=e.weights}catch(e){return w(e?.message??String(e))}p.phase="optimizer",v.textContent="building optimizer…",await S(),o=await (0,u.SplatOptimizer).create(r,s,n,{seed:_}),C(),await o.renderImage(),// Preload the text encoder at boot (with its own progress bar) so the first
// Optimize is instant instead of stalling on an 84 MB download.
p.phase="textmodel",await T(e=>{v.textContent=e}),p.ready=!0,p.phase="idle",m.disabled=!1,f.disabled=!1,y.disabled=!1,b(""),$(),requestAnimationFrame(W)}m.addEventListener("click",()=>void F()),f.addEventListener("click",()=>void L()),y.addEventListener("click",()=>void O()),h.addEventListener("keydown",e=>{"Enter"===e.key&&F()}),j().catch(e=>w(`boot failed: ${e?.message??e}`))},{"./splat/optimize":"nZSdJ","./splat/model_assets":"3CXuq"}],nZSdJ:[function(e,t,i){/**
 * optimize — SplatOptimizer: the prompt→splats optimization core (Task #7).
 *
 * Wires the two independently-verified halves on ONE shared GPUDevice, fully
 * GPU-resident:
 *   RasterEngine  (src/splat/raster.ts) — 2D Gaussian splats → NCHW image, and
 *     the differentiable backward + fused Adam on the raw splat params.
 *   VisionTrainer (src/clip/vision.ts)  — MobileCLIP-S0 forward + −cos(embed,
 *     text) loss + hand-written backward → dL/dpixels.
 *
 * One optimize step is ONE command submit:
 *   raster forward (splats → image)
 *   copy image → CLIP input slot                 (768 KB, identical NCHW bytes)
 *   CLIP forward + loss + backward → dL/dpixels
 *   copy dL/dpixels → raster gradImage           (768 KB)
 *   raster backward → raw-param grads
 *   raster Adam → params updated
 * Nothing round-trips to CPU in the hot loop. The copies are legal byte-for-byte
 * blits because the raster output and the CLIP input are BOTH 256×256 NCHW
 * planar f32 in [0,1] by construction (asserted in create()).
 *
 * Device-agnostic: verified headless under bun-webgpu (tools/splat/
 * optimize_test.ts proves −cos actually decreases) before the browser page
 * (src/splat_page.ts) wraps it with ORT-web text encoding + a canvas.
 *//// <reference types="@webgpu/types" />
var r=e("@parcel/transformer-js/src/esmodule-helpers.js");r.defineInteropFlag(i),r.export(i,"LEGIBLE_G",()=>d),r.export(i,"LEGIBLE_INIT",()=>l),r.export(i,"LEGIBLE_LRS",()=>u),r.export(i,"DEFAULT_NUDGE_AMOUNT",()=>c),r.export(i,"SplatOptimizer",()=>p),/** cos(a, b) — the metric the page shows and the test gates on. */r.export(i,"cosine",()=>g),// ---------------------------------------------------------------------------
// Deterministic random splat init (browser-safe: no node imports). Conventional
// 2D-splat start — spread over the canvas, small translucent Gaussians, mid
// colours the optimizer pushes around. SoA layout matches raster_wgsl.ts:
// [mean 2G][logScale 2G][theta G][colorRaw 3G][opacityRaw G], per-splat
// interleaved within each segment.
// ---------------------------------------------------------------------------
r.export(i,"randomSplats",()=>h),r.export(i,"nudgeSplats",()=>m);var a=e("./raster"),s=e("./raster_wgsl"),n=e("../clip/vision"),o=e("./adam_wgsl");let d=12e3,l={scale:9,scaleJitter:.35,opacityRaw:.4,colorSpread:1.2},u={mean:1.5,logScale:.06,theta:.08,color:.12,opacity:.06},c=.18;class p{static async create(e,t,i,r={}){let[s,o,l]=t.inputShape;if(3!==s||256!==o||256!==l)throw Error(`optimize: CLIP inputShape [${s},${o},${l}] != [3,256,256] — the raster→CLIP copy assumes matching NCHW dims`);let u=r.G??d,c=r.cap??2048,g=await (0,a.RasterEngine).create(e,{H:256,W:256,G:u,cap:c,bg:r.bg??[.5,.5,.5]}),m=await (0,n.VisionTrainer).create(e,t,i);return g.setParams(r.initParams??h(u,r.seed??1,r.init)),g.zeroAdamState(),new p(e,g,m,r)}constructor(e,t,i,r){this.side=256,this.step_=0,this.device=e,this.raster=t,this.trainer=i,this.lrs=r.lrs??u,this.hyper=r.hyper??o.DEFAULT_HYPER,this.init=r.init}/** Target text embedding (raw, un-normalized — the −cos loss normalizes it).
   *  Call on every prompt change; cheap (a 2 KB buffer write). */setPrompt(e){this.trainer.writeText(e)}/** One optimization step: forward → CLIP loss → backward → Adam, ONE submit. */step(){let e=this.device.createCommandEncoder();this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!0}),e.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackward(e),this.step_+=1,this.raster.recordAdam(e,this.step_,this.lrs,this.hyper),this.device.queue.submit([e.finish()])}get stepCount(){return this.step_}/** Partial re-randomization of the current splats. Unlike Reset, this keeps
   *  the optimizer, CLIP resources, prompt, step count, and Adam buffers alive. */async nudge(e={}){let t=this.raster.dims.G,i=await this.raster.readParams();m(i,t,e.seed??Date.now(),e.amount??c,e.init??this.init),this.raster.setParams(i)}/** Render the current splats without training; leaves the image on the GPU
   *  and returns it (NCHW planar [3][256][256]) for display / metrics. */async renderImage(){return this.raster.runForward(),this.raster.readImage()}/** CLIP embedding of the current splat image (forward-only). The page can use
   *  this to show live cosine similarity to the prompt; the test uses it to
   *  prove the loss decreases. */async currentEmbedding(){let e=this.device.createCommandEncoder();return this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!1}),this.device.queue.submit([e.finish()]),y(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){this.raster.destroy()}}function g(e,t){let i=0,r=0,a=0;for(let s=0;s<e.length;s++)i+=e[s]*t[s],r+=e[s]*e[s],a+=t[s]*t[s];return i/Math.sqrt(r*a||1)}function h(e,t=1,i={}){let r=i.scale??l.scale,a=i.scaleJitter??l.scaleJitter,n=i.opacityRaw??l.opacityRaw,o=i.colorSpread??l.colorSpread,d=t>>>0||1,u=()=>{let e=Math.imul((d=Math.imul(d,747796405)+2891336453>>>0)>>>(d>>>28)+4^d,277803737)>>>0;return(e=(e>>>22^e)>>>0)/4294967296},c=()=>{let e=0,t=0;for(;0===e;)e=u();for(;0===t;)t=u();return Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*t)},p=new Float32Array(e*s.PARAM_STRIDE),g=2*e,h=4*e,m=5*e,f=8*e,y=Math.log(r);for(let t=0;t<e;t++)p[0+2*t+0]=256*u(),p[0+2*t+1]=256*u(),p[g+2*t+0]=y+a*c(),p[g+2*t+1]=y+a*c(),p[h+t]=u()*Math.PI*2,p[m+3*t+0]=o*c(),p[m+3*t+1]=o*c(),p[m+3*t+2]=o*c(),p[f+t]=n;return p}function m(e,t,i=1,r=c,a={}){if(e.length!==t*s.PARAM_STRIDE)throw Error("nudgeSplats: wrong param length");let n=f(r,0,1);if(0===n)return e;let o=h(t,i,a),d=2*t,l=4*t,u=5*t,p=8*t,g=Math.log(.3),m=Math.log(64);for(let i=0;i<t;i++){var y,v,x,b,w,$,_,P;let t=0+2*i;e[t+0]=f((y=e[t+0],y+(o[t+0]-y)*n),0,256),e[t+1]=f((v=e[t+1],v+(o[t+1]-v)*n),0,256);let r=d+2*i;e[r+0]=f((x=e[r+0],x+(o[r+0]-x)*n),g,m),e[r+1]=f((b=e[r+1],b+(o[r+1]-b)*n),g,m);let a=l+i;e[a]=function(e,t,i){let r=2*Math.PI;return e+(((t-e+Math.PI)%r+r)%r-Math.PI)*i}(e[a],o[a],n);let s=u+3*i;e[s+0]=(w=e[s+0],w+(o[s+0]-w)*n),e[s+1]=($=e[s+1],$+(o[s+1]-$)*n),e[s+2]=(_=e[s+2],_+(o[s+2]-_)*n);let c=p+i;e[c]=(P=e[c],P+(o[c]-P)*n)}return e}function f(e,t,i){return Math.max(t,Math.min(i,e))}// small readback helper (kept local — RasterEngine's is private, and the CLIP
// output buffer isn't one of RasterEngine's).
async function y(e,t,i){let r=e.createBuffer({size:4*i,usage:9/*COPY_DST*/}),a=e.createCommandEncoder();a.copyBufferToBuffer(t,0,r,0,4*i),e.queue.submit([a.finish()]),await r.mapAsync(1);let s=new Float32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),s}},{"./raster":"5D8U0","./raster_wgsl":"6IBEA","../clip/vision":"lNzsi","./adam_wgsl":"kfWkJ","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],"5D8U0":[function(e,t,i){/**
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
 */var r=e("@parcel/transformer-js/src/esmodule-helpers.js");r.defineInteropFlag(i),r.export(i,"RasterEngine",()=>l);var a=e("./raster_wgsl"),s=e("./adam_wgsl");let n={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},o=e=>Math.ceil(e/256);async function d(e,t,i){e.pushErrorScope("validation");let r=e.createShaderModule({code:t}),a=e.createComputePipeline({layout:"auto",compute:{module:r,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw console.error(`--- WGSL that failed (${i}) ---
${t}`),Error(`raster pipeline validation (${i}): ${s.message}`);return a}class l{constructor(e,t){if(// per-group adam uniform buffers + bind groups (one per param group)
this.adamUni=[],this.adamBind=[],this.device=e,this.dims=(0,a.resolveDims)(t),this.dims.numTiles>65535)throw Error("raster: numTiles exceeds 1D dispatch limit")}static async create(e,t){let i=new l(e,t);return await i.build(t),i}storage(e,t=0){return this.device.createBuffer({size:4*e,usage:n.STORAGE|t})}async build(e){let t=this.dims,i=t.G*a.PARAM_STRIDE;// buffers
this.params=this.storage(i,n.COPY_SRC|n.COPY_DST),this.derived=this.storage(i),this.accGrad=this.storage(i,n.COPY_DST|n.COPY_SRC),this.gradRaw=this.storage(i,n.COPY_SRC),this.mBuf=this.storage(i,n.COPY_DST),this.vBuf=this.storage(i,n.COPY_DST),this.tileCounts=this.storage(t.numTiles,n.COPY_DST|n.COPY_SRC),this.binnedIds=this.storage(t.numTiles*t.cap,n.COPY_SRC),this.tileStop=this.storage(t.numTiles,n.COPY_SRC),this.image=this.storage(3*t.H*t.W,n.COPY_SRC),this.gradImage=this.storage(3*t.H*t.W,n.COPY_DST),// pipelines
this.prepPipe=await d(this.device,(0,a.prepShader)(e),"prep"),this.emitPipe=await d(this.device,(0,a.emitShader)(e),"emit"),this.fwdPipe=await d(this.device,(0,a.forwardShader)(e),"forward"),this.bwdPipe=await d(this.device,(0,a.backwardShader)(e),"backward"),this.chainPipe=await d(this.device,(0,a.chainShader)(e),"chain"),this.clearBinsPipe=await d(this.device,(0,a.clearShader)(t.numTiles),"clearBins"),this.clearGradsPipe=await d(this.device,(0,a.clearShader)(i),"clearGrads"),this.adamPipe=await d(this.device,(0,s.adamShader)(),"adam");let r=(e,t)=>this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:t.map((e,t)=>({binding:t,resource:{buffer:e}}))});for(let e of(this.prepBind=r(this.prepPipe,[this.params,this.derived]),this.emitBind=r(this.emitPipe,[this.derived,this.tileCounts,this.binnedIds]),this.fwdBind=r(this.fwdPipe,[this.tileCounts,this.binnedIds,this.derived,this.image,this.tileStop]),this.bwdBind=r(this.bwdPipe,[this.gradImage,this.tileCounts,this.binnedIds,this.tileStop,this.derived,this.accGrad]),this.chainBind=r(this.chainPipe,[this.accGrad,this.derived,this.params,this.gradRaw]),this.clearBinsBind=r(this.clearBinsPipe,[this.tileCounts]),this.clearGradsBind=r(this.clearGradsPipe,[this.accGrad]),(0,a.paramSegments)(t.G))){let e=this.device.createBuffer({size:s.ADAM_UNIFORM_BYTES,usage:n.UNIFORM|n.COPY_DST});this.adamUni.push(e),this.adamBind.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradRaw}},{binding:3,resource:{buffer:this.mBuf}},{binding:4,resource:{buffer:this.vBuf}}]}))}}// ---- uploads / readback ------------------------------------------------
setParams(e){if(e.length!==this.dims.G*a.PARAM_STRIDE)throw Error("setParams: wrong length");this.device.queue.writeBuffer(this.params,0,e)}setGradImage(e){if(e.length!==3*this.dims.H*this.dims.W)throw Error("setGradImage: wrong length");this.device.queue.writeBuffer(this.gradImage,0,e)}zeroAdamState(){let e=new Float32Array(this.dims.G*a.PARAM_STRIDE);this.device.queue.writeBuffer(this.mBuf,0,e),this.device.queue.writeBuffer(this.vBuf,0,e)}async readFloats(e,t){let i=this.device.createBuffer({size:4*t,usage:n.MAP_READ|n.COPY_DST}),r=this.device.createCommandEncoder();r.copyBufferToBuffer(e,0,i,0,4*t),this.device.queue.submit([r.finish()]),await i.mapAsync(1/* GPUMapMode.READ */);let a=new Float32Array(i.getMappedRange().slice(0));return i.unmap(),i.destroy(),a}readImage(){return this.readFloats(this.image,3*this.dims.H*this.dims.W)}readParams(){return this.readFloats(this.params,this.dims.G*a.PARAM_STRIDE)}readGradRaw(){return this.readFloats(this.gradRaw,this.dims.G*a.PARAM_STRIDE)}// ---- pass recording ----------------------------------------------------
/** prep -> clear bins -> emit -> forward. Populates `derived` and `image`. */recordForward(e){let t=this.dims,i=e.beginComputePass();i.setPipeline(this.prepPipe),i.setBindGroup(0,this.prepBind),i.dispatchWorkgroups(o(t.G)),i.setPipeline(this.clearBinsPipe),i.setBindGroup(0,this.clearBinsBind),i.dispatchWorkgroups(o(t.numTiles)),i.setPipeline(this.emitPipe),i.setBindGroup(0,this.emitBind),i.dispatchWorkgroups(o(t.G)),i.setPipeline(this.fwdPipe),i.setBindGroup(0,this.fwdBind),i.dispatchWorkgroups(t.numTiles),i.end()}/** clear grads -> backward -> chain. Requires a prior recordForward (uses its
   *  sorted binnedIds, tileStop and derived). Reads `gradImage`, writes gradRaw. */recordBackward(e){let t=this.dims,i=e.beginComputePass();i.setPipeline(this.clearGradsPipe),i.setBindGroup(0,this.clearGradsBind),i.dispatchWorkgroups(o(t.G*a.DERIVED_STRIDE)),i.setPipeline(this.bwdPipe),i.setBindGroup(0,this.bwdBind),i.dispatchWorkgroups(t.numTiles),i.setPipeline(this.chainPipe),i.setBindGroup(0,this.chainBind),i.dispatchWorkgroups(o(t.G)),i.end()}/** Adam over all 5 param groups; call after recordBackward (reads gradRaw). */recordAdam(e,t,i=s.DEFAULT_LRS,r=s.DEFAULT_HYPER){let n=(0,a.paramSegments)(this.dims.G),d={mean:i.mean,logScale:i.logScale,theta:i.theta,color:i.color,opacity:i.opacity},l=1-Math.pow(r.beta1,t),u=1-Math.pow(r.beta2,t);// write the 5 uniforms first (queued before the submit that runs `enc`)
n.forEach((e,t)=>{let i=new ArrayBuffer(s.ADAM_UNIFORM_BYTES),a=new Uint32Array(i),n=new Float32Array(i);a[0]=e.offset,a[1]=e.length,n[2]=d[e.name],n[3]=r.beta1,n[4]=r.beta2,n[5]=r.eps,n[6]=l,n[7]=u,this.device.queue.writeBuffer(this.adamUni[t],0,i)});let c=e.beginComputePass();c.setPipeline(this.adamPipe),n.forEach((e,t)=>{c.setBindGroup(0,this.adamBind[t]),c.dispatchWorkgroups(o(e.length))}),c.end()}// ---- self-submitting convenience wrappers ------------------------------
runForward(){let e=this.device.createCommandEncoder();this.recordForward(e),this.device.queue.submit([e.finish()])}runBackward(){let e=this.device.createCommandEncoder();this.recordBackward(e),this.device.queue.submit([e.finish()])}runAdam(e,t,i){let r=this.device.createCommandEncoder();this.recordAdam(r,e,t,i),this.device.queue.submit([r.finish()])}destroy(){for(let e of[this.params,this.derived,this.accGrad,this.gradRaw,this.mBuf,this.vBuf,this.tileCounts,this.binnedIds,this.tileStop,this.image,this.gradImage,...this.adamUni])try{e.destroy()}catch(e){}}}},{"./raster_wgsl":"6IBEA","./adam_wgsl":"kfWkJ","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],"6IBEA":[function(e,t,i){/**
 * raster_wgsl — PURE WGSL codegen for the 2D Gaussian-splat rasterizer
 * (forward + backward) with the reparameterized OUR data model.
 *
 * Zero imports on purpose (mirrors src/clip/vision_wgsl.ts): every shape /
 * threshold / offset is baked into the shader source, so there are no uniforms
 * in the raster kernels (Adam gets a tiny uniform for per-step hyperparams).
 * Each returned string is one self-contained compute module, testable headless
 * under bun-webgpu (tools/splat/raster_test.ts) against a float64 JS reference.
 *
 * ---------------------------------------------------------------------------
 * DATA MODEL (ours — see docs/splat_raster_spec.md; differs from the Metal
 * reference which trains the conic directly). Adam updates the RAW params:
 *
 *   params[G*9] SoA segments (one GPUBuffer):
 *     mean       [0,   2G)   g*2 + {0,1}      px
 *     logScale   [2G,  4G)   2G + g*2 + {0,1}
 *     theta      [4G,  5G)   4G + g
 *     colorRaw   [5G,  8G)   5G + g*3 + {0,1,2}
 *     opacityRaw [8G,  9G)   8G + g
 *   gradRaw, m, v share this SoA layout.
 *
 * Reparameterization (computed in the `prep` kernel; its Jacobian is applied
 * once per splat in the `chain` kernel — NOT inside the per-pixel backward, so
 * the per-pixel backward stays byte-for-byte the reference recurrence and the
 * Jacobian is unit-testable in isolation):
 *     scale   = clamp(exp(logScale), 0.3, 64)      px
 *     ix,iy   = 1/scale.x^2 , 1/scale.y^2
 *     conic a = cos^2 ix + sin^2 iy
 *           b = cos sin (ix - iy)
 *           c = sin^2 ix + cos^2 iy            (inverse of R diag(s^2) R^T)
 *     color   = sigmoid(colorRaw)
 *     opacity = sigmoid(opacityRaw)
 *
 *   derived[G*9] AoS stride 9 (one GPUBuffer, produced by `prep`, consumed by
 *   every raster kernel so binding count stays <=8/stage):
 *     g*9 + {0=mx,1=my, 2=a,3=b,4=c, 5=cR,6=cG,7=cB, 8=opacity}
 *
 * Gradient accumulation is fixed-point atomicAdd<i32> into `accGrad[G*9]`
 * (AoS parallel to `derived`: {mx,my, a,b,c, cR,cG,cB, op}) with a documented
 * scale GRAD_SCALE. WGSL has no f32 atomicAdd; i32 fixed-point is the simplest
 * correct scheme and the scale nearly cancels in Adam (m/sqrt(v) is scale-
 * invariant). `chain` divides by GRAD_SCALE when it reads accGrad. Overflow is
 * guarded by clamping the fixed-point value into i32 range.
 *
 * Output image: NCHW planar f32 [3][H][W] in ~[0,1], out[ch*H*W + y*W + x]
 * (binds directly as the CLIP encoder input slot later).
 * ---------------------------------------------------------------------------
 */// Algorithm thresholds (baked as literals — fixed by the alpha/visibility math,
// matching the Metal reference gsplat_fast_kernels.metal / v11 fixedbin).
var r=e("@parcel/transformer-js/src/esmodule-helpers.js");r.defineInteropFlag(i),r.export(i,"TILE",()=>a),r.export(i,"ALPHA_THRESHOLD",()=>s),r.export(i,"MAX_ALPHA",()=>n),r.export(i,"TRANSMITTANCE_CUTOFF",()=>o),r.export(i,"EPS",()=>d),r.export(i,"SCALE_MIN",()=>l),r.export(i,"SCALE_MAX",()=>u),r.export(i,"DERIVED_STRIDE",()=>c),r.export(i,"PARAM_STRIDE",()=>p),r.export(i,"resolveDims",()=>f),// ---------------------------------------------------------------------------
// 1) prep — thread/splat: raw params -> derived (mean, conic, color, opacity).
//    The single place the reparameterization forward is computed.
// ---------------------------------------------------------------------------
r.export(i,"prepShader",()=>v),// ---------------------------------------------------------------------------
// 2) emit — thread/splat: fixedbin binning (v11 style, no prefix sum, no CPU
//    readback). Atomic cursor per tile into constant-stride bins tile*cap.
//    Merges count+emit: tileCounts is the cursor (cleared each step); a splat
//    whose slot >= cap is dropped (graceful overflow). The forward re-sorts by
//    index so the emit order is irrelevant to the result (deterministic).
// ---------------------------------------------------------------------------
r.export(i,"emitShader",()=>x),// ---------------------------------------------------------------------------
// 3) forward — 1 workgroup(256)/tile, one thread per pixel. Stage tile ids in
//    shared, bitonic-sort ASCENDING (recovers painter order == splat index
//    order; there is no depth), write the sorted ids back so the backward can
//    skip re-sorting, front-to-back composite with early-out, save
//    tileStop = max visible-prefix length (bounds the backward replay).
// ---------------------------------------------------------------------------
r.export(i,"forwardShader",()=>b),// ---------------------------------------------------------------------------
// 4) backward — 1 workgroup(256)/tile, one thread per pixel. Replays the
//    visible prefix (bounded by tileStop) to recover T_final and end_i, then
//    walks BACK-TO-FRONT reconstructing per-splat grads with T_prev = T_cur/
//    (1-alpha). Accumulates DERIVED-space grads (mean, conic, color, opacity)
//    into accGrad via fixed-point atomicAdd<i32> — byte-for-byte the Metal
//    reference recurrence. NO barriers in the per-pixel loop, so the uniformity
//    rule is satisfied trivially (each pixel's end_i gates only its own loop).
// ---------------------------------------------------------------------------
r.export(i,"backwardShader",()=>w),// ---------------------------------------------------------------------------
// 5) chain — thread/splat: convert DERIVED-space grads (accGrad, i32 fixed-
//    point) to RAW-space grads (gradRaw, f32 SoA). This is the reparam
//    Jacobian, applied ONCE per splat. Verified against a float64 JS reference
//    on a single splat before the full gradcheck (docs derivation-care note).
//
//    conic (a,b,c)(ix,iy,theta):
//      g_ix = g_a cos^2 + g_b cos sin + g_c sin^2
//      g_iy = g_a sin^2 - g_b cos sin + g_c cos^2
//      g_theta = (ix-iy) [ (cos^2-sin^2) g_b + 2 cos sin (g_c - g_a) ]
//    ix = 1/scale.x^2, scale.x = clamp(exp(lsx)):  dix/dlsx = -2 ix  (unclamped)
//      g_lsx = g_ix * (-2 ix) * gateX
//    sigmoid reparams: g_colorRaw = g_color color(1-color); same for opacity.
//    mean has no reparam (passes through).
// ---------------------------------------------------------------------------
r.export(i,"chainShader",()=>$),// ---------------------------------------------------------------------------
// 6) clear — thread/element: zero a storage buffer viewed as array<u32>
//    (works for the i32 accGrad and the u32 tileCounts; 0 bits == 0 either way).
// ---------------------------------------------------------------------------
r.export(i,"clearShader",()=>_),/** Segment offsets for the Adam driver (matches seg()). */r.export(i,"paramSegments",()=>P);let a=16,s=1/255,n=.99,o=1e-4,d=1e-8,l=.3,u=64,c=9,p=9;// 16x16 tile == 256 pixels == 256 threads/workgroup
function g(e,t){if(!e)throw Error(`raster_wgsl: ${t}`)}/** WGSL f32 literal — always has a '.' or exponent so it is not parsed as int. */function h(e){g(Number.isFinite(e),`non-finite literal ${e}`);let t=e.toString();return/[.eE]/.test(t)||(t+=".0"),t}let m=e=>`${e>>>0}u`;function f(e){g(e.H>0&&e.W>0&&e.G>0,"H,W,G must be positive"),g(e.H%a==0&&e.W%a==0,`H,W must be multiples of ${a}`),g((e.cap&e.cap-1)==0&&e.cap>0,"cap must be a power of two"),g(4*e.cap<=16384,`cap*4 (${4*e.cap}B) exceeds 16KB workgroup storage`);let t=e.W/a,i=e.H/a;return{H:e.H,W:e.W,G:e.G,cap:e.cap,tilesX:t,tilesY:i,numTiles:t*i,bg:e.bg??[.5,.5,.5],gradScale:e.gradScale??65536}}// ---------------------------------------------------------------------------
// Shared WGSL fragments (inlined per kernel — each module stays standalone)
// ---------------------------------------------------------------------------
/** Segment base offsets into the SoA params/gradRaw/m/v buffers. */function y(e){return{mean:0,logScale:2*e.G,theta:4*e.G,colorRaw:5*e.G,opacityRaw:8*e.G}}function v(e){let t=f(e),i=y(t);return/* wgsl */`
fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@group(0) @binding(0) var<storage, read>       params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${m(t.G)}) { return; }
  let mx  = params[${m(i.mean)} + g * 2u + 0u];
  let my  = params[${m(i.mean)} + g * 2u + 1u];
  let lsx = params[${m(i.logScale)} + g * 2u + 0u];
  let lsy = params[${m(i.logScale)} + g * 2u + 1u];
  let th  = params[${m(i.theta)} + g];
  let cr0 = params[${m(i.colorRaw)} + g * 3u + 0u];
  let cr1 = params[${m(i.colorRaw)} + g * 3u + 1u];
  let cr2 = params[${m(i.colorRaw)} + g * 3u + 2u];
  let opr = params[${m(i.opacityRaw)} + g];

  let sx = clamp(exp(lsx), ${h(l)}, ${h(u)});
  let sy = clamp(exp(lsy), ${h(l)}, ${h(u)});
  let ix = 1.0 / (sx * sx);
  let iy = 1.0 / (sy * sy);
  let cs = cos(th);
  let sn = sin(th);

  let base = g * ${m(c)};
  derived[base + 0u] = mx;
  derived[base + 1u] = my;
  derived[base + 2u] = cs * cs * ix + sn * sn * iy;           // conic a
  derived[base + 3u] = cs * sn * (ix - iy);                   // conic b
  derived[base + 4u] = sn * sn * ix + cs * cs * iy;           // conic c
  derived[base + 5u] = sigmoid1(cr0);
  derived[base + 6u] = sigmoid1(cr1);
  derived[base + 7u] = sigmoid1(cr2);
  derived[base + 8u] = sigmoid1(opr);
}
`}function x(e){let t=f(e);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       derived    : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds  : array<u32>;

// EXACT ellipse-vs-rect test (Metal reference ellipse_intersects_rect): the
// tile intersects the alpha-support ellipse {q(d) <= tau} iff the min of the
// quadratic form over the rect is <= tau. Checks corners + edge extrema.
fn ellipse_hit(mx : f32, my : f32, a : f32, b : f32, c : f32, tau : f32,
               rx0 : f32, ry0 : f32, rx1 : f32, ry1 : f32) -> bool {
  let dx0 = rx0 - mx; let dx1 = rx1 - mx;
  let dy0 = ry0 - my; let dy1 = ry1 - my;
  if (mx >= rx0 && mx <= rx1 && my >= ry0 && my <= ry1) { return true; }
  var qmin = 1e30;
  qmin = min(qmin, a * dx0 * dx0 + 2.0 * b * dx0 * dy0 + c * dy0 * dy0);
  qmin = min(qmin, a * dx0 * dx0 + 2.0 * b * dx0 * dy1 + c * dy1 * dy1);
  qmin = min(qmin, a * dx1 * dx1 + 2.0 * b * dx1 * dy0 + c * dy0 * dy0);
  qmin = min(qmin, a * dx1 * dx1 + 2.0 * b * dx1 * dy1 + c * dy1 * dy1);
  if (c > 1e-8) {
    var dy = clamp(-(b / c) * dx0, dy0, dy1);
    qmin = min(qmin, a * dx0 * dx0 + 2.0 * b * dx0 * dy + c * dy * dy);
    dy = clamp(-(b / c) * dx1, dy0, dy1);
    qmin = min(qmin, a * dx1 * dx1 + 2.0 * b * dx1 * dy + c * dy * dy);
  }
  if (a > 1e-8) {
    var dx = clamp(-(b / a) * dy0, dx0, dx1);
    qmin = min(qmin, a * dx * dx + 2.0 * b * dx * dy0 + c * dy0 * dy0);
    dx = clamp(-(b / a) * dy1, dx0, dx1);
    qmin = min(qmin, a * dx * dx + 2.0 * b * dx * dy1 + c * dy1 * dy1);
  }
  return qmin <= tau;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${m(t.G)}) { return; }
  let base = g * ${m(c)};
  let op = derived[base + 8u];
  if (op <= ${h(s)}) { return; }
  let ratio = max(${h(s)} / max(op, ${h(d)}), ${h(d)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let mx = derived[base + 0u]; let my = derived[base + 1u];
  let a  = derived[base + 2u]; let b  = derived[base + 3u]; let c = derived[base + 4u];
  let det = max(a * c - b * b, ${h(d)});
  let hx = sqrt(max(tau * c / det, 0.0));
  let hy = sqrt(max(tau * a / det, 0.0));
  let x0 = max(0, i32(floor(mx - hx - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(mx + hx - 0.5)));
  let y0 = max(0, i32(floor(my - hy - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(my + hy - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tx0 = x0 / ${a}; let tx1 = x1 / ${a};
  let ty0 = y0 / ${a}; let ty1 = y1 / ${a};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    let ry0 = f32(ty * ${a}) + 0.5;
    let ry1 = min(f32(${t.H-1}) + 0.5, f32((ty + 1) * ${a} - 1) + 0.5);
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let rx0 = f32(tx * ${a}) + 0.5;
      let rx1 = min(f32(${t.W-1}) + 0.5, f32((tx + 1) * ${a} - 1) + 0.5);
      if (ellipse_hit(mx, my, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        let tile = u32(ty * ${t.tilesX} + tx);
        let slot = atomicAdd(&tileCounts[tile], 1u);
        if (slot < ${m(t.cap)}) { binnedIds[tile * ${m(t.cap)} + slot] = g; }
      }
    }
  }
}
`}function b(e){let t=f(e),i=t.H*t.W;return/* wgsl */`
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;  // NCHW planar
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;

var<workgroup> sh_ids     : array<u32, ${t.cap}>;
var<workgroup> sh_maxstop : atomic<u32>;

fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u); v = v - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${m(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${m(t.cap)});
  let start = tileId * ${m(t.cap)};
  let sortN = nextPow2(count);

  // stage ids + pad to power of two with sentinel 0xffffffff (sorts to the end)
  for (var i = tid; i < sortN; i = i + 256u) {
    sh_ids[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  if (tid == 0u) { atomicStore(&sh_maxstop, 0u); }
  workgroupBarrier();

  // bitonic sort ascending — 256-thread strided variant (v11 shape)
  var k = 2u;
  loop {
    if (k > sortN) { break; }
    var j = k >> 1u;
    loop {
      if (j == 0u) { break; }
      let nPairs = sortN >> 1u;
      for (var pair = tid; pair < nPairs; pair = pair + 256u) {
        let pos = 2u * j * (pair / j) + (pair % j);
        let ixj = pos + j;
        let asc = (pos & k) == 0u;
        let va = sh_ids[pos];
        let vb = sh_ids[ixj];
        if ((va > vb) == asc) { sh_ids[pos] = vb; sh_ids[ixj] = va; }
      }
      workgroupBarrier();
      j = j >> 1u;
    }
    k = k << 1u;
  }

  // write sorted ids back (backward reuses them without re-sorting)
  for (var i = tid; i < count; i = i + 256u) { binnedIds[start + i] = sh_ids[i]; }
  workgroupBarrier();

  let tileX = tileId % ${m(t.tilesX)};
  let tileY = tileId / ${m(t.tilesX)};
  let x = tileX * ${a}u + (tid % ${a}u);
  let y = tileY * ${a}u + (tid / ${a}u);
  var localStop = 0u;
  if (x < ${m(t.W)} && y < ${m(t.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var accR = 0.0; var accG = 0.0; var accB = 0.0; var T = 1.0;
    for (var i = 0u; i < count; i = i + 1u) {
      let gg = sh_ids[i];
      let b3 = gg * ${m(c)};
      let dx = pxc - derived[b3 + 0u];
      let dy = pyc - derived[b3 + 1u];
      let a  = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
      let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[b3 + 8u] * exp(power);
      let alpha = min(${h(n)}, raw);
      if (alpha < ${h(s)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b3 + 5u];
      accG = accG + w * derived[b3 + 6u];
      accB = accB + w * derived[b3 + 7u];
      T = T * (1.0 - alpha);
      if (T < ${h(o)}) { break; }
    }
    let pix = y * ${m(t.W)} + x;
    image[0u * ${m(i)} + pix] = accR + T * ${h(t.bg[0])};
    image[1u * ${m(i)} + pix] = accG + T * ${h(t.bg[1])};
    image[2u * ${m(i)} + pix] = accB + T * ${h(t.bg[2])};
  }
  atomicMax(&sh_maxstop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&sh_maxstop); }
}
`}function w(e){let t=f(e),i=t.H*t.W,r=h(t.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;  // NCHW planar
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;

var<workgroup> sh_ids : array<u32, ${t.cap}>;

fn fixadd(base : u32, slot : u32, v : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(v * ${r}), -2.14e9, 2.14e9)));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${m(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${m(t.cap)});
  let stopc = min(count, tileStop[tileId]);
  let start = tileId * ${m(t.cap)};
  for (var i = tid; i < stopc; i = i + 256u) { sh_ids[i] = binnedIds[start + i]; }
  workgroupBarrier();  // only barrier; everything below is per-pixel (uniformity safe)

  let tileX = tileId % ${m(t.tilesX)};
  let tileY = tileId / ${m(t.tilesX)};
  let x = tileX * ${a}u + (tid % ${a}u);
  let y = tileY * ${a}u + (tid / ${a}u);
  if (x >= ${m(t.W)} || y >= ${m(t.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${m(t.W)} + x;
  let goR = gradImage[0u * ${m(i)} + pix];
  let goG = gradImage[1u * ${m(i)} + pix];
  let goB = gradImage[2u * ${m(i)} + pix];

  // phase A: replay to recover T_final and the stop index end_i
  var T = 1.0;
  var endi = stopc;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b3 = gg * ${m(c)};
    let dx = pxc - derived[b3 + 0u];
    let dy = pyc - derived[b3 + 1u];
    let a = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${h(n)}, derived[b3 + 8u] * exp(power));
    if (alpha < ${h(s)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${h(o)}) { endi = i + 1u; break; }
  }

  // phase B: back-to-front recurrence
  var Tcur = T;
  var gT = goR * ${h(t.bg[0])} + goG * ${h(t.bg[1])} + goB * ${h(t.bg[2])};
  for (var ii = i32(endi) - 1; ii >= 0; ii = ii - 1) {
    let gg = sh_ids[u32(ii)];
    let b3 = gg * ${m(c)};
    let dx = pxc - derived[b3 + 0u];
    let dy = pyc - derived[b3 + 1u];
    let a = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let op = derived[b3 + 8u];
    let raw = op * exp(power);
    let alpha = min(${h(n)}, raw);
    if (alpha < ${h(s)}) { continue; }
    let denom = max(1.0 - alpha, ${h(d)});
    let Tprev = Tcur / denom;
    let cR = derived[b3 + 5u]; let cG = derived[b3 + 6u]; let cB = derived[b3 + 7u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b3, 5u, goR * Tprev * alpha);
    fixadd(b3, 6u, goG * Tprev * alpha);
    fixadd(b3, 7u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${h(n)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-(a * dx + b * dy));
    let gdy = gPower * (-(b * dx + c * dy));
    fixadd(b3, 2u, gPower * (-0.5) * dx * dx);   // g_a
    fixadd(b3, 3u, gPower * (-1.0) * dx * dy);   // g_b
    fixadd(b3, 4u, gPower * (-0.5) * dy * dy);   // g_c
    fixadd(b3, 0u, -gdx);                        // g_mean.x
    fixadd(b3, 1u, -gdy);                        // g_mean.y
    fixadd(b3, 8u, gRaw * (raw / max(op, ${h(d)})));  // g_opacity

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`}function $(e){let t=f(e),i=y(t),r=h(1/t.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       accGrad : array<i32>;   // fixed-point
@group(0) @binding(1) var<storage, read>       derived : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${m(t.G)}) { return; }
  let b3 = g * ${m(c)};
  let inv = ${r};
  let gmx = f32(accGrad[b3 + 0u]) * inv;
  let gmy = f32(accGrad[b3 + 1u]) * inv;
  let gA  = f32(accGrad[b3 + 2u]) * inv;
  let gB  = f32(accGrad[b3 + 3u]) * inv;
  let gC  = f32(accGrad[b3 + 4u]) * inv;
  let gc0 = f32(accGrad[b3 + 5u]) * inv;
  let gc1 = f32(accGrad[b3 + 6u]) * inv;
  let gc2 = f32(accGrad[b3 + 7u]) * inv;
  let gop = f32(accGrad[b3 + 8u]) * inv;

  let lsx = params[${m(i.logScale)} + g * 2u + 0u];
  let lsy = params[${m(i.logScale)} + g * 2u + 1u];
  let th  = params[${m(i.theta)} + g];
  let ex = exp(lsx); let ey = exp(lsy);
  let sx = clamp(ex, ${h(l)}, ${h(u)});
  let sy = clamp(ey, ${h(l)}, ${h(u)});
  let gateX = select(0.0, 1.0, ex > ${h(l)} && ex < ${h(u)});
  let gateY = select(0.0, 1.0, ey > ${h(l)} && ey < ${h(u)});
  let ix = 1.0 / (sx * sx);
  let iy = 1.0 / (sy * sy);
  let cs = cos(th); let sn = sin(th);

  let gix = gA * cs * cs + gB * cs * sn + gC * sn * sn;
  let giy = gA * sn * sn - gB * cs * sn + gC * cs * cs;
  let glsx = gix * (-2.0 * ix) * gateX;
  let glsy = giy * (-2.0 * iy) * gateY;
  let D = ix - iy;
  let gth = D * ((cs * cs - sn * sn) * gB + 2.0 * cs * sn * (gC - gA));

  let col0 = derived[b3 + 5u]; let col1 = derived[b3 + 6u]; let col2 = derived[b3 + 7u];
  let opv  = derived[b3 + 8u];

  gradRaw[${m(i.mean)} + g * 2u + 0u] = gmx;
  gradRaw[${m(i.mean)} + g * 2u + 1u] = gmy;
  gradRaw[${m(i.logScale)} + g * 2u + 0u] = glsx;
  gradRaw[${m(i.logScale)} + g * 2u + 1u] = glsy;
  gradRaw[${m(i.theta)} + g] = gth;
  gradRaw[${m(i.colorRaw)} + g * 3u + 0u] = gc0 * col0 * (1.0 - col0);
  gradRaw[${m(i.colorRaw)} + g * 3u + 1u] = gc1 * col1 * (1.0 - col1);
  gradRaw[${m(i.colorRaw)} + g * 3u + 2u] = gc2 * col2 * (1.0 - col2);
  gradRaw[${m(i.opacityRaw)} + g] = gop * opv * (1.0 - opv);
}
`}function _(e){return/* wgsl */`
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= ${m(e)}) { return; }
  buf[gid.x] = 0u;
}
`}function P(e){return[{name:"mean",offset:0,length:2*e},{name:"logScale",offset:2*e,length:2*e},{name:"theta",offset:4*e,length:e},{name:"color",offset:5*e,length:3*e},{name:"opacity",offset:8*e,length:e}]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],"4C2Su":[function(e,t,i){i.interopDefault=function(e){return e&&e.__esModule?e:{default:e}},i.defineInteropFlag=function(e){Object.defineProperty(e,"__esModule",{value:!0})},i.exportAll=function(e,t){return Object.keys(e).forEach(function(i){"default"===i||"__esModule"===i||t.hasOwnProperty(i)||Object.defineProperty(t,i,{enumerable:!0,get:function(){return e[i]}})}),t},i.export=function(e,t,i){Object.defineProperty(e,t,{enumerable:!0,get:i})}},{}]},["7i9mK"],"7i9mK","parcelRequire924a")//# sourceMappingURL=splat.fa7e6df0.js.map
;
//# sourceMappingURL=splat.fa7e6df0.js.map
