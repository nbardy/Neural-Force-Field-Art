!// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles
function(e,t,r,a,i){/* eslint-disable no-undef */var s="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof self?self:"undefined"!=typeof window?window:"undefined"!=typeof global?global:{},d="function"==typeof s[a]&&s[a],o=d.cache||{},n="undefined"!=typeof module&&"function"==typeof module.require&&module.require.bind(module);function u(t,r){if(!o[t]){if(!e[t]){// if we cannot find the module within our internal map or
// cache jump to the current global require ie. the last bundle
// that was added to the page.
var i="function"==typeof s[a]&&s[a];if(!r&&i)return i(t,!0);// If there are other bundles on this page the require from the
// previous one is saved to 'previousRequire'. Repeat this as
// many times as there are bundles until the module is found or
// we exhaust the require chain.
if(d)return d(t,!0);// Try the node require function if it exists.
if(n&&"string"==typeof t)return n(t);var l=Error("Cannot find module '"+t+"'");throw l.code="MODULE_NOT_FOUND",l}p.resolve=function(r){var a=e[t][1][r];return null!=a?a:r},p.cache={};var c=o[t]=new u.Module(t);e[t][0].call(c.exports,p,c,c.exports,this)}return o[t].exports;function p(e){var t=p.resolve(e);return!1===t?{}:u(t)}}u.isParcelRequire=!0,u.Module=function(e){this.id=e,this.bundle=u,this.exports={}},u.modules=e,u.cache=o,u.parent=d,u.register=function(t,r){e[t]=[function(e,t){t.exports=r},{}]},Object.defineProperty(u,"root",{get:function(){return s[a]}}),s[a]=u;for(var l=0;l<t.length;l++)u(t[l]);if(r){// Expose entry point to Node, AMD or browser globals
// Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
var c=u(r);"object"==typeof exports&&"undefined"!=typeof module?module.exports=c:"function"==typeof define&&define.amd?define(function(){return c}):i&&(this[i]=c)}}({"7i9mK":[function(e,t,r){let a,i,s,d,o,n,u;/**
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
var l=e("./splat/optimize"),c=e("./splat/feature_optimize"),p=e("./splat/pixel_optimize"),g=e("./splat/model_assets");let f={gpu:!!navigator.gpu,ready:!1,running:!1,step:0,cos:null,initialCos:null,error:null,phase:"boot"};window.__splat=f;// ── DOM ──────────────────────────────────────────────────────────────────────
let m=document.getElementById("splat"),h=document.getElementById("prompt"),y=document.getElementById("optimize"),b=document.getElementById("nudge"),x=document.getElementById("reset"),E=document.getElementById("representation"),v=document.getElementById("auto-explore"),A=document.getElementById("readout"),_=document.getElementById("notice");function w(e){_.textContent=e}function S(e){f.error=e,f.phase="error",w(e),A.textContent="—",// eslint-disable-next-line no-console
console.error("[splat_page]",e)}function R(){f.step=o?o.stepCount:0;let e=[`step ${f.step}`];if(null!==f.cos){let t=f.initialCos??f.cos,r=f.cos-t;e.push(`cos ${f.cos.toFixed(4)}`),e.push(`init ${t.toFixed(4)}`),e.push(`Δ ${r>=0?"+":""}${r.toFixed(4)}`)}f.phase&&"run"!==f.phase&&e.push(`(${f.phase})`),A.textContent=e.join("  \xb7  ")}let T=1,C=null,$=/* wgsl */`
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
`;async function P(){a.pushErrorScope("validation");let e=a.createShaderModule({code:$});n=a.createRenderPipeline({layout:"auto",vertex:{module:e,entryPoint:"vs"},fragment:{module:e,entryPoint:"fs",targets:[{format:u}]},primitive:{topology:"triangle-list"}});let t=await a.popErrorScope();if(t)throw Error(`blit pipeline invalid: ${t.message}`)}function B(){C=a.createBindGroup({layout:n.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:o.raster.image}}]})}// Hide the specifier behind a Function-constructor indirection so the BUNDLER
// leaves it alone and the BROWSER does a genuine native dynamic import of the
// CDN URL. A plain `import(TF_URL)` gets rewritten into a parcel module helper
// that would try to resolve the URL as a local bundle.
let D=Function("u","return import(u)"),I=null,L=null;async function F(e){if(L)return;let t=await D("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm");t.env.allowRemoteModels=!0;let r="Nbardy/nff-clip-splat-weights",a=t=>{if("progress"===t.status&&t.total){let r=Math.round(t.progress??t.loaded/t.total*100),a=Math.round(r/100*16),i="█".repeat(a)+"░".repeat(16-a);e?.(`loading text encoder  [${i}] ${r}%  \xb7  ${(t.loaded/1e6).toFixed(1)}/${(t.total/1e6).toFixed(0)} MB`)}};// self-hosted alongside the vision weights
I=await t.AutoTokenizer.from_pretrained(r,{progress_callback:a}),L=await t.CLIPTextModelWithProjection.from_pretrained(r,{dtype:"fp16",device:"wasm",session_options:{graphOptimizationLevel:"basic"},progress_callback:a})}async function G(e){await F();let t=await I(e,{padding:"max_length",max_length:77,truncation:!0}),r=await L(t),a=r.text_embeds.data,i=new Float32Array(512);for(let e=0;e<512;e++)i[e]=a[e];return i}// ── Optimize loop ─────────────────────────────────────────────────────────────
let O=null,U=0,M=!1,q=!1,z=0,k=[{step:96,amount:.12},{step:260,amount:.06}];async function W(){if(O&&!M&&!q){M=!0;try{let e=await o.currentEmbedding(),t=(0,l.cosine)(e,O);f.cos=t,null===f.initialCos&&(f.initialCos=t),R()}finally{M=!1}}}function Y(){f.running&&O&&(// 2 optimize steps/frame keeps the page responsive; each step is one submit
// (raster fwd → CLIP fwd+loss+bwd → raster bwd → Adam). At LEGIBLE_G≈12K
// splats a step is cheap, so the loop stays smooth.
o.step(),o.step(),U+=2,f.step=o.stepCount,U>=14&&(U=0,W()),function(){if(!v.checked||q||o instanceof p.PixelBufferOptimizer)return;let e=k[z];e&&!(o.stepCount<e.step)&&(z+=1,X({amount:e.amount,automatic:!0}))}()),function(){if(!C)return;let e=a.createCommandEncoder(),t=e.beginRenderPass({colorAttachments:[{view:i.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});t.setPipeline(n),t.setBindGroup(0,C),t.draw(3),t.end(),a.queue.submit([e.finish()])}(),requestAnimationFrame(Y)}async function N(){if(!f.ready)return;let e=h.value.trim()||"a photo of a cat";y.disabled=!0,f.phase="encoding",f.running=!1,w("encoding prompt (first use downloads the text model — slow)…"),R();try{let t=await G(e);O=t,o.setPrompt(t);// Baseline cos on the CURRENT splats — this is the "initial" the gate checks
// the run rises above.
let r=await o.currentEmbedding();f.initialCos=(0,l.cosine)(r,t),f.cos=f.initialCos,U=0,z=0,w(""),f.phase="run",f.running=!0,R()}catch(e){S(`text encode failed: ${e?.message??e}`)}finally{y.disabled=!1}}async function H(){if(!f.ready)return;f.running=!1,O=null,f.cos=null,f.initialCos=null,f.phase="reset",z=0,T+=1;let e=o;try{o=await j(),e.destroy(),B(),await o.renderImage(),h.value="",f.step=0,f.phase="idle",w(""),R()}catch(e){S(`reset failed: ${e?.message??e}`)}}async function j(){return"feature"===E.value?(0,c.FeaturePainterOptimizer).create(a,s,d,{seed:T}):"pixels"===E.value?(0,p.PixelBufferOptimizer).create(a,s,d,T):(0,l.SplatOptimizer).create(a,s,d,{seed:T})}async function X(e={}){if(!f.ready||q)return;q=!0;let t=f.running;f.running=!1,f.phase=e.automatic?"explore":"nudge",b.disabled=!0,T+=1,R();try{if(o instanceof p.PixelBufferOptimizer?await o.nudge(T,e.amount):await o.nudge({seed:T,amount:e.amount}),await o.renderImage(),U=0,O){let e=await o.currentEmbedding();f.cos=(0,l.cosine)(e,O)}f.phase=t&&O?"run":"idle",f.running=t&&!!O,w(""),R()}catch(e){S(`nudge failed: ${e?.message??e}`)}finally{q=!1,b.disabled=!1}}// ── Boot ─────────────────────────────────────────────────────────────────────
async function V(){if(!navigator.gpu){S("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled."),y.disabled=!0,b.disabled=!0,x.disabled=!0;return}f.phase="adapter";let e=await navigator.gpu.requestAdapter();if(!e){S("no WebGPU adapter available.");return}if(!e.features.has("subgroups")){S("this build requires WebGPU subgroup reductions.");return}a=await e.requestDevice({requiredFeatures:["subgroups"]}),a.addEventListener?.("uncapturederror",e=>{// eslint-disable-next-line no-console
console.error("[webgpu]",e.error?.message??e.error)}),i=m.getContext("webgpu"),u=navigator.gpu.getPreferredCanvasFormat(),i.configure({device:a,format:u,alphaMode:"opaque"}),f.phase="weights";try{let e=await (0,g.loadClipTrainAssets)(e=>{A.textContent=e});s=e.plan,d=e.weights}catch(e){return S(e?.message??String(e))}f.phase="optimizer",A.textContent="building optimizer…",await P(),o=await j(),B(),await o.renderImage(),// Preload the text encoder at boot (with its own progress bar) so the first
// Optimize is instant instead of stalling on an 84 MB download.
f.phase="textmodel",await F(e=>{A.textContent=e}),f.ready=!0,f.phase="idle",y.disabled=!1,b.disabled=!1,x.disabled=!1,w(""),R(),requestAnimationFrame(Y)}y.addEventListener("click",()=>void N()),b.addEventListener("click",()=>void X()),x.addEventListener("click",()=>void H()),E.addEventListener("change",()=>void H()),h.addEventListener("keydown",e=>{"Enter"===e.key&&N()}),V().catch(e=>S(`boot failed: ${e?.message??e}`))},{"./splat/optimize":"nZSdJ","./splat/feature_optimize":"dCIvw","./splat/pixel_optimize":"drkYM","./splat/model_assets":"3CXuq"}],nZSdJ:[function(e,t,r){/**
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
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"LEGIBLE_G",()=>n),a.export(r,"LEGIBLE_INIT",()=>u),a.export(r,"LEGIBLE_LRS",()=>l),a.export(r,"DEFAULT_NUDGE_AMOUNT",()=>c),/** Deterministically select which full splat candidates a nudge replaces. */a.export(r,"nudgeSplatMask",()=>p),a.export(r,"SplatOptimizer",()=>g),/** cos(a, b) — the metric the page shows and the test gates on. */a.export(r,"cosine",()=>f),// ---------------------------------------------------------------------------
// Deterministic random splat init (browser-safe: no node imports). Conventional
// 2D-splat start — spread over the canvas, small translucent Gaussians, mid
// colours the optimizer pushes around. SoA layout matches raster_wgsl.ts:
// [mean 2G][logScale 2G][theta G][colorRaw 3G][opacityRaw G], per-splat
// interleaved within each segment.
// ---------------------------------------------------------------------------
a.export(r,"randomSplats",()=>m),a.export(r,"nudgeSplats",()=>h);var i=e("./raster"),s=e("./raster_wgsl"),d=e("../clip/vision"),o=e("./adam_wgsl");let n=12e3,u={scale:9,scaleJitter:.35,opacityRaw:.4,colorSpread:1.2},l={mean:1.5,logScale:.06,theta:.08,color:.12,opacity:.06},c=.18;function p(e,t=1,r=c){let a=Math.max(0,Math.min(1,r)),i=new Uint32Array(e),s=(2246822507^t)>>>0||1;for(let t=0;t<e;t++)s=Math.imul(s,1664525)+1013904223>>>0,i[t]=s/4294967296<a?1:0;return i}class g{static async create(e,t,r,a={}){let[s,o,u]=t.inputShape;if(3!==s||256!==o||256!==u)throw Error(`optimize: CLIP inputShape [${s},${o},${u}] != [3,256,256] — the raster→CLIP copy assumes matching NCHW dims`);let l=a.G??n,c=a.cap??2048,p=await (0,i.RasterEngine).create(e,{H:256,W:256,G:l,cap:c,bg:a.bg??[.5,.5,.5]}),f=await (0,d.VisionTrainer).create(e,t,r);return p.setParams(a.initParams??m(l,a.seed??1,a.init)),p.zeroAdamState(),new g(e,p,f,a)}constructor(e,t,r,a){this.side=256,this.step_=0,this.device=e,this.raster=t,this.trainer=r,this.lrs=a.lrs??l,this.hyper=a.hyper??o.DEFAULT_HYPER,this.init=a.init}/** Target text embedding (raw, un-normalized — the −cos loss normalizes it).
   *  Call on every prompt change; cheap (a 2 KB buffer write). */setPrompt(e){this.trainer.writeText(e)}/** One optimization step: forward → CLIP loss → backward → Adam, ONE submit. */step(){let e=this.device.createCommandEncoder();this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!0}),e.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackward(e),this.step_+=1,this.raster.recordAdam(e,this.step_,this.lrs,this.hyper),this.device.queue.submit([e.finish()])}get stepCount(){return this.step_}/** Sparse candidate replacement of the current splats. The CLIP resources and
 * prompt stay alive, but Adam's old momentum is deliberately discarded:
 * retaining it makes fresh candidates drift straight back to the prior basin.
 */async nudge(e={}){let t=this.raster.dims.G,r=await this.raster.readParams();h(r,t,e.seed??Date.now(),e.amount??c,e.init??this.init),this.raster.setParams(r),this.raster.zeroAdamState()}/** Render the current splats without training; leaves the image on the GPU
   *  and returns it (NCHW planar [3][256][256]) for display / metrics. */async renderImage(){return this.raster.runForward(),this.raster.readImage()}/** CLIP embedding of the current splat image (forward-only). The page can use
   *  this to show live cosine similarity to the prompt; the test uses it to
   *  prove the loss decreases. */async currentEmbedding(){let e=this.device.createCommandEncoder();return this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!1}),this.device.queue.submit([e.finish()]),y(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){this.raster.destroy()}}function f(e,t){let r=0,a=0,i=0;for(let s=0;s<e.length;s++)r+=e[s]*t[s],a+=e[s]*e[s],i+=t[s]*t[s];return r/Math.sqrt(a*i||1)}function m(e,t=1,r={}){let a=r.scale??u.scale,i=r.scaleJitter??u.scaleJitter,d=r.opacityRaw??u.opacityRaw,o=r.colorSpread??u.colorSpread,n=t>>>0||1,l=()=>{let e=Math.imul((n=Math.imul(n,747796405)+2891336453>>>0)>>>(n>>>28)+4^n,277803737)>>>0;return(e=(e>>>22^e)>>>0)/4294967296},c=()=>{let e=0,t=0;for(;0===e;)e=l();for(;0===t;)t=l();return Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*t)},p=new Float32Array(e*s.PARAM_STRIDE),g=2*e,f=4*e,m=5*e,h=8*e,y=Math.log(a);for(let t=0;t<e;t++)p[0+2*t+0]=256*l(),p[0+2*t+1]=256*l(),p[g+2*t+0]=y+i*c(),p[g+2*t+1]=y+i*c(),p[f+t]=l()*Math.PI*2,p[m+3*t+0]=o*c(),p[m+3*t+1]=o*c(),p[m+3*t+2]=o*c(),p[h+t]=d;return p}function h(e,t,r=1,a=c,i={},d=p(t,r,a)){if(e.length!==t*s.PARAM_STRIDE)throw Error("nudgeSplats: wrong param length");if(d.length!==t)throw Error("nudgeSplats: wrong selection length");let o=m(t,r,i),n=[[0,2],[2*t,2],[4*t,1],[5*t,3],[8*t,1]];// Replacing a subset creates actual new candidates in unused parts of the
// canvas. Interpolating every splat only makes the same configuration wiggle.
for(let r=0;r<t;r++)if(0!==d[r])for(let[t,a]of n){let i=t+r*a;e.set(o.subarray(i,i+a),i)}return e}// small readback helper (kept local — RasterEngine's is private, and the CLIP
// output buffer isn't one of RasterEngine's).
async function y(e,t,r){let a=e.createBuffer({size:4*r,usage:9/*COPY_DST*/}),i=e.createCommandEncoder();i.copyBufferToBuffer(t,0,a,0,4*r),e.queue.submit([i.finish()]),await a.mapAsync(1);let s=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),a.destroy(),s}},{"./raster":"5D8U0","./raster_wgsl":"6IBEA","../clip/vision":"lNzsi","./adam_wgsl":"kfWkJ","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"5D8U0":[function(e,t,r){/**
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
 */var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"RasterEngine",()=>u);var i=e("./raster_wgsl"),s=e("./adam_wgsl");let d={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},o=e=>Math.ceil(e/256);async function n(e,t,r){e.pushErrorScope("validation");let a=e.createShaderModule({code:t}),i=e.createComputePipeline({layout:"auto",compute:{module:a,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw console.error(`--- WGSL that failed (${r}) ---
${t}`),Error(`raster pipeline validation (${r}): ${s.message}`);return i}class u{constructor(e,t){if(// per-group adam uniform buffers + bind groups (one per param group)
this.adamUni=[],this.adamBind=[],this.device=e,this.dims=(0,i.resolveDims)(t),this.dims.numTiles>65535)throw Error("raster: numTiles exceeds 1D dispatch limit")}static async create(e,t){let r=new u(e,t);return await r.build(t),r}storage(e,t=0){return this.device.createBuffer({size:4*e,usage:d.STORAGE|t})}async build(e){let t=this.dims,r=t.G*i.PARAM_STRIDE;// buffers
this.params=this.storage(r,d.COPY_SRC|d.COPY_DST),this.derived=this.storage(r),this.accGrad=this.storage(r,d.COPY_DST|d.COPY_SRC),this.gradRaw=this.storage(r,d.COPY_SRC),this.mBuf=this.storage(r,d.COPY_DST),this.vBuf=this.storage(r,d.COPY_DST),this.tileCounts=this.storage(t.numTiles,d.COPY_DST|d.COPY_SRC),this.binnedIds=this.storage(t.numTiles*t.cap,d.COPY_SRC),this.tileStop=this.storage(t.numTiles,d.COPY_SRC),this.image=this.storage(3*t.H*t.W,d.COPY_SRC),this.gradImage=this.storage(3*t.H*t.W,d.COPY_DST),// pipelines
this.prepPipe=await n(this.device,(0,i.prepShader)(e),"prep"),this.emitPipe=await n(this.device,(0,i.emitShader)(e),"emit"),this.fwdPipe=await n(this.device,(0,i.forwardShader)(e),"forward"),this.bwdPipe=await n(this.device,(0,i.backwardShader)(e),"backward"),this.chainPipe=await n(this.device,(0,i.chainShader)(e),"chain"),this.clearBinsPipe=await n(this.device,(0,i.clearShader)(t.numTiles),"clearBins"),this.clearGradsPipe=await n(this.device,(0,i.clearShader)(r),"clearGrads"),this.adamPipe=await n(this.device,(0,s.adamShader)(),"adam");let a=(e,t)=>this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:t.map((e,t)=>({binding:t,resource:{buffer:e}}))});for(let e of(this.prepBind=a(this.prepPipe,[this.params,this.derived]),this.emitBind=a(this.emitPipe,[this.derived,this.tileCounts,this.binnedIds]),this.fwdBind=a(this.fwdPipe,[this.tileCounts,this.binnedIds,this.derived,this.image,this.tileStop]),this.bwdBind=a(this.bwdPipe,[this.gradImage,this.tileCounts,this.binnedIds,this.tileStop,this.derived,this.accGrad]),this.chainBind=a(this.chainPipe,[this.accGrad,this.derived,this.params,this.gradRaw]),this.clearBinsBind=a(this.clearBinsPipe,[this.tileCounts]),this.clearGradsBind=a(this.clearGradsPipe,[this.accGrad]),(0,i.paramSegments)(t.G))){let e=this.device.createBuffer({size:s.ADAM_UNIFORM_BYTES,usage:d.UNIFORM|d.COPY_DST});this.adamUni.push(e),this.adamBind.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradRaw}},{binding:3,resource:{buffer:this.mBuf}},{binding:4,resource:{buffer:this.vBuf}}]}))}}// ---- uploads / readback ------------------------------------------------
setParams(e){if(e.length!==this.dims.G*i.PARAM_STRIDE)throw Error("setParams: wrong length");this.device.queue.writeBuffer(this.params,0,e)}setGradImage(e){if(e.length!==3*this.dims.H*this.dims.W)throw Error("setGradImage: wrong length");this.device.queue.writeBuffer(this.gradImage,0,e)}zeroAdamState(){let e=new Float32Array(this.dims.G*i.PARAM_STRIDE);this.device.queue.writeBuffer(this.mBuf,0,e),this.device.queue.writeBuffer(this.vBuf,0,e)}async readFloats(e,t){let r=this.device.createBuffer({size:4*t,usage:d.MAP_READ|d.COPY_DST}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(1/* GPUMapMode.READ */);let i=new Float32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}async readU32(e,t){let r=this.device.createBuffer({size:4*t,usage:d.MAP_READ|d.COPY_DST}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(1/* GPUMapMode.READ */);let i=new Uint32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}readImage(){return this.readFloats(this.image,3*this.dims.H*this.dims.W)}readParams(){return this.readFloats(this.params,this.dims.G*i.PARAM_STRIDE)}readGradRaw(){return this.readFloats(this.gradRaw,this.dims.G*i.PARAM_STRIDE)}async readTileTelemetry(){let e=await this.readU32(this.tileCounts,this.dims.numTiles),t=await this.readU32(this.tileStop,this.dims.numTiles),r=0,a=0,i=0,s=0,d=0,o=0;for(let n=0;n<this.dims.numTiles;n++){let u=e[n],l=t[n];r+=u,a+=l,i=Math.max(i,u),s=Math.max(s,l),u>this.dims.cap&&(d++,o+=u-this.dims.cap)}return{meanCount:r/this.dims.numTiles,maxCount:i,meanStop:a/this.dims.numTiles,maxStop:s,overflowTiles:d,overflowEntries:o}}// ---- pass recording ----------------------------------------------------
/** prep -> clear bins -> emit -> forward. Populates `derived` and `image`. */recordForward(e){let t=this.dims,r=e.beginComputePass();r.setPipeline(this.prepPipe),r.setBindGroup(0,this.prepBind),r.dispatchWorkgroups(o(t.G)),r.setPipeline(this.clearBinsPipe),r.setBindGroup(0,this.clearBinsBind),r.dispatchWorkgroups(o(t.numTiles)),r.setPipeline(this.emitPipe),r.setBindGroup(0,this.emitBind),r.dispatchWorkgroups(o(t.G)),r.setPipeline(this.fwdPipe),r.setBindGroup(0,this.fwdBind),r.dispatchWorkgroups(t.numTiles),r.end()}/** clear grads -> backward -> chain. Requires a prior recordForward (uses its
   *  sorted binnedIds, tileStop and derived). Reads `gradImage`, writes gradRaw. */recordBackward(e){let t=this.dims,r=e.beginComputePass();r.setPipeline(this.clearGradsPipe),r.setBindGroup(0,this.clearGradsBind),r.dispatchWorkgroups(o(t.G*i.DERIVED_STRIDE)),r.setPipeline(this.bwdPipe),r.setBindGroup(0,this.bwdBind),r.dispatchWorkgroups(t.numTiles),r.setPipeline(this.chainPipe),r.setBindGroup(0,this.chainBind),r.dispatchWorkgroups(o(t.G)),r.end()}/** Adam over all 5 param groups; call after recordBackward (reads gradRaw). */recordAdam(e,t,r=s.DEFAULT_LRS,a=s.DEFAULT_HYPER){let d=(0,i.paramSegments)(this.dims.G),n={mean:r.mean,logScale:r.logScale,theta:r.theta,color:r.color,opacity:r.opacity},u=1-Math.pow(a.beta1,t),l=1-Math.pow(a.beta2,t);// write the 5 uniforms first (queued before the submit that runs `enc`)
d.forEach((e,t)=>{let r=new ArrayBuffer(s.ADAM_UNIFORM_BYTES),i=new Uint32Array(r),d=new Float32Array(r);i[0]=e.offset,i[1]=e.length,d[2]=n[e.name],d[3]=a.beta1,d[4]=a.beta2,d[5]=a.eps,d[6]=u,d[7]=l,this.device.queue.writeBuffer(this.adamUni[t],0,r)});let c=e.beginComputePass();c.setPipeline(this.adamPipe),d.forEach((e,t)=>{c.setBindGroup(0,this.adamBind[t]),c.dispatchWorkgroups(o(e.length))}),c.end()}// ---- self-submitting convenience wrappers ------------------------------
runForward(){let e=this.device.createCommandEncoder();this.recordForward(e),this.device.queue.submit([e.finish()])}runBackward(){let e=this.device.createCommandEncoder();this.recordBackward(e),this.device.queue.submit([e.finish()])}runAdam(e,t,r){let a=this.device.createCommandEncoder();this.recordAdam(a,e,t,r),this.device.queue.submit([a.finish()])}destroy(){for(let e of[this.params,this.derived,this.accGrad,this.gradRaw,this.mBuf,this.vBuf,this.tileCounts,this.binnedIds,this.tileStop,this.image,this.gradImage,...this.adamUni])try{e.destroy()}catch(e){}}}},{"./raster_wgsl":"6IBEA","./adam_wgsl":"kfWkJ","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"6IBEA":[function(e,t,r){/**
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
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"TILE",()=>i),a.export(r,"ALPHA_THRESHOLD",()=>s),a.export(r,"MAX_ALPHA",()=>d),a.export(r,"TRANSMITTANCE_CUTOFF",()=>o),a.export(r,"EPS",()=>n),a.export(r,"SCALE_MIN",()=>u),a.export(r,"SCALE_MAX",()=>l),a.export(r,"DERIVED_STRIDE",()=>c),a.export(r,"PARAM_STRIDE",()=>p),a.export(r,"resolveDims",()=>h),// ---------------------------------------------------------------------------
// 1) prep — thread/splat: raw params -> derived (mean, conic, color, opacity).
//    The single place the reparameterization forward is computed.
// ---------------------------------------------------------------------------
a.export(r,"prepShader",()=>b),// ---------------------------------------------------------------------------
// 2) emit — thread/splat: fixedbin binning (v11 style, no prefix sum, no CPU
//    readback). Atomic cursor per tile into constant-stride bins tile*cap.
//    Merges count+emit: tileCounts is the cursor (cleared each step); a splat
//    whose slot >= cap is dropped (graceful overflow). The forward re-sorts by
//    index so the emit order is irrelevant to the result (deterministic).
// ---------------------------------------------------------------------------
a.export(r,"emitShader",()=>x),// ---------------------------------------------------------------------------
// 3) forward — 1 workgroup(256)/tile, one thread per pixel. Stage tile ids in
//    shared, bitonic-sort ASCENDING (recovers painter order == splat index
//    order; there is no depth), write the sorted ids back so the backward can
//    skip re-sorting, front-to-back composite with early-out, save
//    tileStop = max visible-prefix length (bounds the backward replay).
// ---------------------------------------------------------------------------
a.export(r,"forwardShader",()=>E),// ---------------------------------------------------------------------------
// 4) backward — 1 workgroup(256)/tile, one thread per pixel. Replays the
//    visible prefix (bounded by tileStop) to recover T_final and end_i, then
//    walks BACK-TO-FRONT reconstructing per-splat grads with T_prev = T_cur/
//    (1-alpha). Accumulates DERIVED-space grads (mean, conic, color, opacity)
//    into accGrad via fixed-point atomicAdd<i32> — byte-for-byte the Metal
//    reference recurrence. NO barriers in the per-pixel loop, so the uniformity
//    rule is satisfied trivially (each pixel's end_i gates only its own loop).
// ---------------------------------------------------------------------------
a.export(r,"backwardShader",()=>v),// ---------------------------------------------------------------------------
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
a.export(r,"chainShader",()=>A),// ---------------------------------------------------------------------------
// 6) clear — thread/element: zero a storage buffer viewed as array<u32>
//    (works for the i32 accGrad and the u32 tileCounts; 0 bits == 0 either way).
// ---------------------------------------------------------------------------
a.export(r,"clearShader",()=>_),/** Segment offsets for the Adam driver (matches seg()). */a.export(r,"paramSegments",()=>w);let i=16,s=1/255,d=.99,o=1e-4,n=1e-8,u=.3,l=64,c=9,p=9;// 16x16 tile == 256 pixels == 256 threads/workgroup
function g(e,t){if(!e)throw Error(`raster_wgsl: ${t}`)}/** WGSL f32 literal — always has a '.' or exponent so it is not parsed as int. */function f(e){g(Number.isFinite(e),`non-finite literal ${e}`);let t=e.toString();return/[.eE]/.test(t)||(t+=".0"),t}let m=e=>`${e>>>0}u`;function h(e){g(e.H>0&&e.W>0&&e.G>0,"H,W,G must be positive"),g(e.H%i==0&&e.W%i==0,`H,W must be multiples of ${i}`),g((e.cap&e.cap-1)==0&&e.cap>0,"cap must be a power of two"),g(4*e.cap<=16384,`cap*4 (${4*e.cap}B) exceeds 16KB workgroup storage`);let t=e.W/i,r=e.H/i;return{H:e.H,W:e.W,G:e.G,cap:e.cap,tilesX:t,tilesY:r,numTiles:t*r,bg:e.bg??[.5,.5,.5],gradScale:e.gradScale??65536}}// ---------------------------------------------------------------------------
// Shared WGSL fragments (inlined per kernel — each module stays standalone)
// ---------------------------------------------------------------------------
/** Segment base offsets into the SoA params/gradRaw/m/v buffers. */function y(e){return{mean:0,logScale:2*e.G,theta:4*e.G,colorRaw:5*e.G,opacityRaw:8*e.G}}function b(e){let t=h(e),r=y(t);return/* wgsl */`
fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@group(0) @binding(0) var<storage, read>       params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${m(t.G)}) { return; }
  let mx  = params[${m(r.mean)} + g * 2u + 0u];
  let my  = params[${m(r.mean)} + g * 2u + 1u];
  let lsx = params[${m(r.logScale)} + g * 2u + 0u];
  let lsy = params[${m(r.logScale)} + g * 2u + 1u];
  let th  = params[${m(r.theta)} + g];
  let cr0 = params[${m(r.colorRaw)} + g * 3u + 0u];
  let cr1 = params[${m(r.colorRaw)} + g * 3u + 1u];
  let cr2 = params[${m(r.colorRaw)} + g * 3u + 2u];
  let opr = params[${m(r.opacityRaw)} + g];

  let sx = clamp(exp(lsx), ${f(u)}, ${f(l)});
  let sy = clamp(exp(lsy), ${f(u)}, ${f(l)});
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
`}function x(e){let t=h(e);return/* wgsl */`
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
  if (op <= ${f(s)}) { return; }
  let ratio = max(${f(s)} / max(op, ${f(n)}), ${f(n)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let mx = derived[base + 0u]; let my = derived[base + 1u];
  let a  = derived[base + 2u]; let b  = derived[base + 3u]; let c = derived[base + 4u];
  let det = max(a * c - b * b, ${f(n)});
  let hx = sqrt(max(tau * c / det, 0.0));
  let hy = sqrt(max(tau * a / det, 0.0));
  let x0 = max(0, i32(floor(mx - hx - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(mx + hx - 0.5)));
  let y0 = max(0, i32(floor(my - hy - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(my + hy - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tx0 = x0 / ${i}; let tx1 = x1 / ${i};
  let ty0 = y0 / ${i}; let ty1 = y1 / ${i};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    let ry0 = f32(ty * ${i}) + 0.5;
    let ry1 = min(f32(${t.H-1}) + 0.5, f32((ty + 1) * ${i} - 1) + 0.5);
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let rx0 = f32(tx * ${i}) + 0.5;
      let rx1 = min(f32(${t.W-1}) + 0.5, f32((tx + 1) * ${i} - 1) + 0.5);
      if (ellipse_hit(mx, my, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        let tile = u32(ty * ${t.tilesX} + tx);
        let slot = atomicAdd(&tileCounts[tile], 1u);
        if (slot < ${m(t.cap)}) { binnedIds[tile * ${m(t.cap)} + slot] = g; }
      }
    }
  }
}
`}function E(e){let t=h(e),r=t.H*t.W;return/* wgsl */`
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
  let x = tileX * ${i}u + (tid % ${i}u);
  let y = tileY * ${i}u + (tid / ${i}u);
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
      let alpha = min(${f(d)}, raw);
      if (alpha < ${f(s)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b3 + 5u];
      accG = accG + w * derived[b3 + 6u];
      accB = accB + w * derived[b3 + 7u];
      T = T * (1.0 - alpha);
      if (T < ${f(o)}) { break; }
    }
    let pix = y * ${m(t.W)} + x;
    image[0u * ${m(r)} + pix] = accR + T * ${f(t.bg[0])};
    image[1u * ${m(r)} + pix] = accG + T * ${f(t.bg[1])};
    image[2u * ${m(r)} + pix] = accB + T * ${f(t.bg[2])};
  }
  atomicMax(&sh_maxstop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&sh_maxstop); }
}
`}function v(e){let t=h(e),r=t.H*t.W,a=f(t.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;  // NCHW planar
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;

var<workgroup> sh_ids : array<u32, ${t.cap}>;

fn fixadd(base : u32, slot : u32, v : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(v * ${a}), -2.14e9, 2.14e9)));
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
  let x = tileX * ${i}u + (tid % ${i}u);
  let y = tileY * ${i}u + (tid / ${i}u);
  if (x >= ${m(t.W)} || y >= ${m(t.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${m(t.W)} + x;
  let goR = gradImage[0u * ${m(r)} + pix];
  let goG = gradImage[1u * ${m(r)} + pix];
  let goB = gradImage[2u * ${m(r)} + pix];

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
    let alpha = min(${f(d)}, derived[b3 + 8u] * exp(power));
    if (alpha < ${f(s)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${f(o)}) { endi = i + 1u; break; }
  }

  // phase B: back-to-front recurrence
  var Tcur = T;
  var gT = goR * ${f(t.bg[0])} + goG * ${f(t.bg[1])} + goB * ${f(t.bg[2])};
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
    let alpha = min(${f(d)}, raw);
    if (alpha < ${f(s)}) { continue; }
    let denom = max(1.0 - alpha, ${f(n)});
    let Tprev = Tcur / denom;
    let cR = derived[b3 + 5u]; let cG = derived[b3 + 6u]; let cB = derived[b3 + 7u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b3, 5u, goR * Tprev * alpha);
    fixadd(b3, 6u, goG * Tprev * alpha);
    fixadd(b3, 7u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${f(d)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-(a * dx + b * dy));
    let gdy = gPower * (-(b * dx + c * dy));
    fixadd(b3, 2u, gPower * (-0.5) * dx * dx);   // g_a
    fixadd(b3, 3u, gPower * (-1.0) * dx * dy);   // g_b
    fixadd(b3, 4u, gPower * (-0.5) * dy * dy);   // g_c
    fixadd(b3, 0u, -gdx);                        // g_mean.x
    fixadd(b3, 1u, -gdy);                        // g_mean.y
    fixadd(b3, 8u, gRaw * (raw / max(op, ${f(n)})));  // g_opacity

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`}function A(e){let t=h(e),r=y(t),a=f(1/t.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       accGrad : array<i32>;   // fixed-point
@group(0) @binding(1) var<storage, read>       derived : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${m(t.G)}) { return; }
  let b3 = g * ${m(c)};
  let inv = ${a};
  let gmx = f32(accGrad[b3 + 0u]) * inv;
  let gmy = f32(accGrad[b3 + 1u]) * inv;
  let gA  = f32(accGrad[b3 + 2u]) * inv;
  let gB  = f32(accGrad[b3 + 3u]) * inv;
  let gC  = f32(accGrad[b3 + 4u]) * inv;
  let gc0 = f32(accGrad[b3 + 5u]) * inv;
  let gc1 = f32(accGrad[b3 + 6u]) * inv;
  let gc2 = f32(accGrad[b3 + 7u]) * inv;
  let gop = f32(accGrad[b3 + 8u]) * inv;

  let lsx = params[${m(r.logScale)} + g * 2u + 0u];
  let lsy = params[${m(r.logScale)} + g * 2u + 1u];
  let th  = params[${m(r.theta)} + g];
  let ex = exp(lsx); let ey = exp(lsy);
  let sx = clamp(ex, ${f(u)}, ${f(l)});
  let sy = clamp(ey, ${f(u)}, ${f(l)});
  let gateX = select(0.0, 1.0, ex > ${f(u)} && ex < ${f(l)});
  let gateY = select(0.0, 1.0, ey > ${f(u)} && ey < ${f(l)});
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

  gradRaw[${m(r.mean)} + g * 2u + 0u] = gmx;
  gradRaw[${m(r.mean)} + g * 2u + 1u] = gmy;
  gradRaw[${m(r.logScale)} + g * 2u + 0u] = glsx;
  gradRaw[${m(r.logScale)} + g * 2u + 1u] = glsy;
  gradRaw[${m(r.theta)} + g] = gth;
  gradRaw[${m(r.colorRaw)} + g * 3u + 0u] = gc0 * col0 * (1.0 - col0);
  gradRaw[${m(r.colorRaw)} + g * 3u + 1u] = gc1 * col1 * (1.0 - col1);
  gradRaw[${m(r.colorRaw)} + g * 3u + 2u] = gc2 * col2 * (1.0 - col2);
  gradRaw[${m(r.opacityRaw)} + g] = gop * opv * (1.0 - opv);
}
`}function _(e){return/* wgsl */`
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= ${m(e)}) { return; }
  buf[gid.x] = 0u;
}
`}function w(e){return[{name:"mean",offset:0,length:2*e},{name:"logScale",offset:2*e,length:2*e},{name:"theta",offset:4*e,length:e},{name:"color",offset:5*e,length:3*e},{name:"opacity",offset:8*e,length:e}]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],k3151:[function(e,t,r){r.interopDefault=function(e){return e&&e.__esModule?e:{default:e}},r.defineInteropFlag=function(e){Object.defineProperty(e,"__esModule",{value:!0})},r.exportAll=function(e,t){return Object.keys(e).forEach(function(r){"default"===r||"__esModule"===r||t.hasOwnProperty(r)||Object.defineProperty(t,r,{enumerable:!0,get:function(){return e[r]}})}),t},r.export=function(e,t,r){Object.defineProperty(e,t,{enumerable:!0,get:r})}},{}],dCIvw:[function(e,t,r){/** CLIP-guided optimizer for the experimental 2D Feature Painter. *//// <reference types="@webgpu/types" />
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"FEATURE_PAINTER_G",()=>u),a.export(r,"FEATURE_PAINTER_INIT",()=>l),a.export(r,"FEATURE_PAINTER_LRS",()=>c),a.export(r,"FeaturePainterOptimizer",()=>p),/**
 * Compact Feature8 initialization. RGB remains in the regular splat buffer;
 * only five latent z channels live here. All channels begin at zero, so a
 * fresh renderer exactly matches RGB even though the latent decoder columns
 * are nonzero and can route gradient immediately.
 */a.export(r,"randomFeatures",()=>g),/**
 * A zero-output but gradient-active decoder initialization. The RGB-skip
 * columns and bias stay zero; only latent columns are seeded. Because every
 * latent feature starts at zero, this preserves exact RGB image parity while
 * avoiding the one-step feature-gradient dead start of an all-zero decoder.
 */a.export(r,"randomDecoder",()=>f),a.export(r,"cosine",()=>n.cosine);var i=e("../clip/vision"),s=e("./feature_painter_wgsl"),d=e("./feature_painter"),o=e("./adam_wgsl"),n=e("./optimize");let u=2048,l={scale:7,scaleJitter:.45,opacityRaw:-.1,colorSpread:1.1},c={...n.LEGIBLE_LRS};class p{static async create(e,t,r,a={}){let[s,o,c]=t.inputShape;if(3!==s||256!==o||256!==c)throw Error("feature painter requires MobileCLIP 256x256 RGB input");let m=a.G??u,h=await (0,d.FeaturePainterEngine).create(e,{H:256,W:256,G:m,cap:2048,bg:[.5,.5,.5]}),y=await (0,i.VisionTrainer).create(e,t,r),b=a.init??l;return h.setParams((0,n.randomSplats)(m,a.seed??1,b)),h.setFeatureParams(g(m,a.seed??1)),// The output residual is still exactly zero at boot: z/Ax/Ay and all
// decoder bias/RGB-skip weights are zero. Nonzero latent columns give the
// feature field a gradient on its very first optimization step.
h.setDecoderParams(f(a.seed??1)),h.zeroAdamState(),new p(e,h,y,b,a)}constructor(e,t,r,a,i){this.side=256,this.step_=0,this.device=e,this.raster=t,this.trainer=r,this.init=a,this.lrs=i.lrs??c,this.hyper=i.hyper??o.DEFAULT_HYPER,this.featureLR=i.featureLR??d.FEATURE_LR,this.decoderLR=i.decoderLR??d.DECODER_LR}setPrompt(e){this.trainer.writeText(e)}get stepCount(){return this.step_}step(){let e=this.device.createCommandEncoder();this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!0}),e.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackward(e),this.step_+=1,this.raster.recordAdam(e,this.step_,this.lrsForStep(),this.hyper,void 0,{feature:this.featureLR,decoder:this.decoderLR}),this.device.queue.submit([e.finish()])}/** Start mobile, then progressively settle. This is only an LR schedule, so
   * Adam's fixed beta bias correction remains mathematically valid. */lrsForStep(){let e=Math.max(0,Math.min(1,this.step_/180)),t=1.6-.6*e,r=.55+.45*e;return{mean:this.lrs.mean*t,logScale:this.lrs.logScale*t,theta:this.lrs.theta*t,color:this.lrs.color*r,opacity:this.lrs.opacity*r}}async nudge(e={}){let t=this.raster.dims.G,r=e.seed??Date.now(),a=e.amount??.12,i=(0,n.nudgeSplatMask)(t,r,a),[d,o]=await Promise.all([this.raster.readParams(),this.raster.readFeatureParams()]);(0,n.nudgeSplats)(d,t,r,a,e.init??this.init,i);for(let e=0;e<t;e++)0!==i[e]&&o.fill(0,e*s.FEATURE_STRIDE,(e+1)*s.FEATURE_STRIDE);this.raster.setParams(d),this.raster.setFeatureParams(o),this.raster.zeroAdamState()}async renderImage(){return this.raster.runForward(),this.raster.readImage()}async currentEmbedding(){let e=this.device.createCommandEncoder();return this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!1}),this.device.queue.submit([e.finish()]),m(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){this.raster.destroy()}}function g(e,t=1){return new Float32Array(e*s.FEATURE_STRIDE)}function f(e=1){let t=e>>>0||1,r=()=>{let e=Math.imul((t=Math.imul(t,747796405)+2891336453>>>0)>>>(t>>>28)+4^t,277803737)>>>0;return(e=(e>>>22^e)>>>0)/4294967296},a=()=>{let e=0,t=0;for(;0===e;)e=r();for(;0===t;)t=r();return Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*t)},i=new Float32Array(s.DECODER_PARAM_COUNT);for(let e=0;e<3;e++)for(let t=0;t<s.FEATURE_LATENT_CHANNELS;t++)i[8*e+3+t]=.25*a();return i}async function m(e,t,r){let a=e.createBuffer({size:4*r,usage:9}),i=e.createCommandEncoder();i.copyBufferToBuffer(t,0,a,0,4*r),e.queue.submit([i.finish()]),await a.mapAsync(1);let s=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),a.destroy(),s}},{"../clip/vision":"lNzsi","./feature_painter_wgsl":"4Oo98","./feature_painter":"jHKeo","./adam_wgsl":"kfWkJ","./optimize":"nZSdJ","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"4Oo98":[function(e,t,r){/**
 * Specialized fused Feature8 splat shaders.
 *
 * The old experiment materialized a 32-channel feature image and a 32-channel
 * image gradient. This path retains the RGB raster as an exact skip baseline,
 * carries only five latent channels, and evaluates the residual decoder in the
 * same tile passes. See docs/FEATURE_PAINTER_FUSED_DECISION.md.
 */var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"FEATURE_LATENT_CHANNELS",()=>s),a.export(r,"FEATURE_STRIDE",()=>d),a.export(r,"FEATURE_DIM",()=>o),a.export(r,"DECODER_PARAM_COUNT",()=>n),a.export(r,"DECODER_RESIDUAL_SCALE",()=>u),a.export(r,"FEATURE_STATE_STRIDE",()=>l),a.export(r,"FEATURE_ACC_DERIVED_OFFSET",()=>c),a.export(r,"FEATURE_ACC_EXTRA_OFFSET",()=>p),a.export(r,"FEATURE_ACC_LOCAL_RAW_OFFSET",()=>g),a.export(r,"FEATURE_ACC_STRIDE",()=>f),/** Raw splat parameters -> RGB derived state plus local-frame coefficients. */a.export(r,"featurePrepShader",()=>b),/** Fixed-bin emitter matching the RGB raster, with the wider Feature8 state. */a.export(r,"featureEmitShader",()=>x),/** Alpha-composite Feature8 and decode RGB in one tile shader. */a.export(r,"featureForwardShader",()=>E),/**
 * Fused RGB-gradient -> Feature8 VJP -> splat backward.
 *
 * The production path reduces f32 contributions in hardware subgroups and
 * quantizes only each subgroup/tile partial before its device atomic. It emits
 * vastly fewer device atomics than a splat-pixel-hit reduction while retaining
 * the fixed-point accumulator used by the geometry and Adam chains.
 */a.export(r,"featureBackwardShader",()=>v),/** Derived/conic gradients plus direct local-frame gradients -> raw splat grads. */a.export(r,"featureGeometryChainShader",()=>A),/** Packed accumulator -> feature, direct geometry, and decoder gradients. */a.export(r,"featureChainShader",()=>_),a.export(r,"FEATURE_CHAIN_WORK_ITEMS",()=>w);var i=e("./raster_wgsl");let s=5,d=3*s,o=3+s,n=3*o+3,u=.1,l=15,c=0,p=9,g=p+d,f=g+5,m=e=>/[.eE]/.test(String(e))?String(e):`${e}.0`,h=e=>`${e>>>0}u`;function y(e){let t=(0,i.resolveDims)(e),r=t.G*f;return{d:t,hw:t.H*t.W,decoderOffset:r,code:/* wgsl */`
const STATE_STRIDE : u32 = ${h(l)};
const FEATURE_STRIDE : u32 = ${h(d)};
const FEATURE_DIM : u32 = ${h(o)};
const ACC_EXTRA_OFFSET : u32 = ${h(p)};
const ACC_LOCAL_RAW_OFFSET : u32 = ${h(g)};
const ACC_STRIDE : u32 = ${h(f)};
const DECODER_OFFSET : u32 = ${h(r)};
const RESIDUAL_SCALE : f32 = ${m(u)};

fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn logit1(x : f32) -> f32 { let y = clamp(x, ${m(i.EPS)}, ${m(1-i.EPS)}); return log(y / (1.0 - y)); }
fn fixadd(dst : ptr<storage, array<atomic<i32>>, read_write>, index : u32, v : f32) {
  atomicAdd(&(*dst)[index], i32(clamp(round(v * ${m(t.gradScale)}), -2.14e9, 2.14e9)));
}
`}}function b(e){let{d:t,code:r}=y(e),a=2*t.G,s=4*t.G,d=5*t.G,o=8*t.G;return/* wgsl */`
${r}
@group(0) @binding(0) var<storage, read> params : array<f32>;
@group(0) @binding(1) var<storage, read_write> state : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${h(t.G)}) { return; }
  let mx = params[${h(0)} + g * 2u];
  let my = params[${h(0)} + g * 2u + 1u];
  let lsx = params[${h(a)} + g * 2u];
  let lsy = params[${h(a)} + g * 2u + 1u];
  let th = params[${h(s)} + g];
  let ex = exp(lsx);
  let ey = exp(lsy);
  let sx = clamp(ex, ${m(i.SCALE_MIN)}, ${m(i.SCALE_MAX)});
  let sy = clamp(ey, ${m(i.SCALE_MIN)}, ${m(i.SCALE_MAX)});
  let invSx = 1.0 / sx;
  let invSy = 1.0 / sy;
  let ix = invSx * invSx;
  let iy = invSy * invSy;
  let cs = cos(th);
  let sn = sin(th);
  let b = g * STATE_STRIDE;
  state[b + ${h(0)}] = mx;
  state[b + ${h(1)}] = my;
  state[b + ${h(2)}] = cs * cs * ix + sn * sn * iy;
  state[b + ${h(3)}] = cs * sn * (ix - iy);
  state[b + ${h(4)}] = sn * sn * ix + cs * cs * iy;
  state[b + ${h(5)}] = sigmoid1(params[${h(d)} + g * 3u]);
  state[b + ${h(6)}] = sigmoid1(params[${h(d)} + g * 3u + 1u]);
  state[b + ${h(7)}] = sigmoid1(params[${h(d)} + g * 3u + 2u]);
  state[b + ${h(8)}] = sigmoid1(params[${h(o)} + g]);
  state[b + ${h(9)}] = cs;
  state[b + ${h(10)}] = sn;
  state[b + ${h(11)}] = invSx;
  state[b + ${h(12)}] = invSy;
  state[b + ${h(13)}] = select(0.0, 1.0, ex > ${m(i.SCALE_MIN)} && ex < ${m(i.SCALE_MAX)});
  state[b + ${h(14)}] = select(0.0, 1.0, ey > ${m(i.SCALE_MIN)} && ey < ${m(i.SCALE_MAX)});
}
`}function x(e){let{d:t,code:r}=y(e);return/* wgsl */`
${r}
@group(0) @binding(0) var<storage, read> state : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds : array<u32>;

fn ellipseHit(mx : f32, my : f32, a : f32, b : f32, c : f32, tau : f32,
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
  if (g >= ${h(t.G)}) { return; }
  let s = g * STATE_STRIDE;
  let opacity = state[s + ${h(8)}];
  if (opacity <= ${m(i.ALPHA_THRESHOLD)}) { return; }
  let ratio = max(${m(i.ALPHA_THRESHOLD)} / max(opacity, ${m(i.EPS)}), ${m(i.EPS)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }
  let mx = state[s + ${h(0)}];
  let my = state[s + ${h(1)}];
  let a = state[s + ${h(2)}];
  let b = state[s + ${h(3)}];
  let c = state[s + ${h(4)}];
  let det = max(a * c - b * b, ${m(i.EPS)});
  let hx = sqrt(max(tau * c / det, 0.0));
  let hy = sqrt(max(tau * a / det, 0.0));
  let x0 = max(0, i32(floor(mx - hx - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(mx + hx - 0.5)));
  let y0 = max(0, i32(floor(my - hy - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(my + hy - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }
  let tx0 = x0 / ${i.TILE}; let tx1 = x1 / ${i.TILE};
  let ty0 = y0 / ${i.TILE}; let ty1 = y1 / ${i.TILE};
  for (var ty = ty0; ty <= ty1; ty++) {
    let ry0 = f32(ty * ${i.TILE}) + 0.5;
    let ry1 = min(f32(${t.H-1}) + 0.5, f32((ty + 1) * ${i.TILE} - 1) + 0.5);
    for (var tx = tx0; tx <= tx1; tx++) {
      let rx0 = f32(tx * ${i.TILE}) + 0.5;
      let rx1 = min(f32(${t.W-1}) + 0.5, f32((tx + 1) * ${i.TILE} - 1) + 0.5);
      if (ellipseHit(mx, my, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        let tile = u32(ty * ${t.tilesX} + tx);
        let slot = atomicAdd(&tileCounts[tile], 1u);
        if (slot < ${h(t.cap)}) { binnedIds[tile * ${h(t.cap)} + slot] = g; }
      }
    }
  }
}
`}function E(e){let{d:t,hw:r,code:a}=y(e);return/* wgsl */`
${a}
@group(0) @binding(0) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds : array<u32>;
@group(0) @binding(2) var<storage, read> state : array<f32>;
@group(0) @binding(3) var<storage, read> features : array<f32>;
@group(0) @binding(4) var<storage, read> decoder : array<f32>;
@group(0) @binding(5) var<storage, read_write> image : array<f32>;
@group(0) @binding(6) var<storage, read_write> tileStop : array<u32>;

var<workgroup> shIds : array<u32, ${h(t.cap)}>;
var<workgroup> shMaxStop : atomic<u32>;
fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u) - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${h(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${h(t.cap)});
  let start = tileId * ${h(t.cap)};
  let sortN = nextPow2(count);
  for (var i = tid; i < sortN; i += 256u) {
    shIds[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  if (tid == 0u) { atomicStore(&shMaxStop, 0u); }
  workgroupBarrier();
  var k = 2u;
  loop {
    if (k > sortN) { break; }
    var j = k >> 1u;
    loop {
      if (j == 0u) { break; }
      let pairs = sortN >> 1u;
      for (var pair = tid; pair < pairs; pair += 256u) {
        let pos = 2u * j * (pair / j) + (pair % j);
        let ixj = pos + j;
        let asc = (pos & k) == 0u;
        let va = shIds[pos]; let vb = shIds[ixj];
        if ((va > vb) == asc) { shIds[pos] = vb; shIds[ixj] = va; }
      }
      workgroupBarrier();
      j >>= 1u;
    }
    k <<= 1u;
  }
  for (var i = tid; i < count; i += 256u) { binnedIds[start + i] = shIds[i]; }
  workgroupBarrier();

  let tileX = tileId % ${h(t.tilesX)};
  let tileY = tileId / ${h(t.tilesX)};
  let x = tileX * ${h(i.TILE)} + (tid % ${h(i.TILE)});
  let y = tileY * ${h(i.TILE)} + (tid / ${h(i.TILE)});
  var localStop = 0u;
  if (x < ${h(t.W)} && y < ${h(t.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var baseR = 0.0; var baseG = 0.0; var baseB = 0.0;
    var l0 = 0.0; var l1 = 0.0; var l2 = 0.0; var l3 = 0.0; var l4 = 0.0;
    var T = 1.0;
    for (var i = 0u; i < count; i++) {
      let g = shIds[i]; let s = g * STATE_STRIDE;
      let dx = pxc - state[s + ${h(0)}];
      let dy = pyc - state[s + ${h(1)}];
      let a = state[s + ${h(2)}];
      let b = state[s + ${h(3)}];
      let c = state[s + ${h(4)}];
      let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = state[s + ${h(8)}] * exp(power);
      let alpha = min(${m(i.MAX_ALPHA)}, raw);
      if (alpha < ${m(i.ALPHA_THRESHOLD)}) { continue; }
      let cs = state[s + ${h(9)}]; let sn = state[s + ${h(10)}];
      let ux = clamp((cs * dx + sn * dy) * state[s + ${h(11)}], -3.0, 3.0);
      let uy = clamp((-sn * dx + cs * dy) * state[s + ${h(12)}], -3.0, 3.0);
      let e = g * FEATURE_STRIDE;
      let w = T * alpha;
      baseR += w * state[s + ${h(5)}];
      baseG += w * state[s + ${h(6)}];
      baseB += w * state[s + ${h(7)}];
      l0 += w * (features[e] + ux * features[e + 5u] + uy * features[e + 10u]);
      l1 += w * (features[e + 1u] + ux * features[e + 6u] + uy * features[e + 11u]);
      l2 += w * (features[e + 2u] + ux * features[e + 7u] + uy * features[e + 12u]);
      l3 += w * (features[e + 3u] + ux * features[e + 8u] + uy * features[e + 13u]);
      l4 += w * (features[e + 4u] + ux * features[e + 9u] + uy * features[e + 14u]);
      T *= 1.0 - alpha;
      if (T < ${m(i.TRANSMITTANCE_CUTOFF)}) { break; }
    }
    baseR += T * ${m(t.bg[0])};
    baseG += T * ${m(t.bg[1])};
    baseB += T * ${m(t.bg[2])};
    let rR = decoder[24u] + decoder[0u] * baseR + decoder[1u] * baseG + decoder[2u] * baseB + decoder[3u] * l0 + decoder[4u] * l1 + decoder[5u] * l2 + decoder[6u] * l3 + decoder[7u] * l4;
    let rG = decoder[25u] + decoder[8u] * baseR + decoder[9u] * baseG + decoder[10u] * baseB + decoder[11u] * l0 + decoder[12u] * l1 + decoder[13u] * l2 + decoder[14u] * l3 + decoder[15u] * l4;
    let rB = decoder[26u] + decoder[16u] * baseR + decoder[17u] * baseG + decoder[18u] * baseB + decoder[19u] * l0 + decoder[20u] * l1 + decoder[21u] * l2 + decoder[22u] * l3 + decoder[23u] * l4;
    let pixel = y * ${h(t.W)} + x;
    image[pixel] = sigmoid1(logit1(baseR) + RESIDUAL_SCALE * rR);
    image[${h(r)} + pixel] = sigmoid1(logit1(baseG) + RESIDUAL_SCALE * rG);
    image[${h(2*r)} + pixel] = sigmoid1(logit1(baseB) + RESIDUAL_SCALE * rB);
  }
  atomicMax(&shMaxStop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&shMaxStop); }
}
`}function v(e){let{d:t,hw:r,code:a}=y(e),s=i.TILE*i.TILE/64;return/* wgsl */`
enable subgroups;
${a}
@group(0) @binding(0) var<storage, read> gradImage : array<f32>;
@group(0) @binding(1) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read> binnedIds : array<u32>;
@group(0) @binding(3) var<storage, read> tileStop : array<u32>;
@group(0) @binding(4) var<storage, read> state : array<f32>;
@group(0) @binding(5) var<storage, read> features : array<f32>;
@group(0) @binding(6) var<storage, read> decoder : array<f32>;
@group(0) @binding(7) var<storage, read_write> acc : array<atomic<i32>>;

fn qgrad(v : f32) -> i32 {
  return i32(clamp(round(v * ${m(t.gradScale)}), -2.14e9, 2.14e9));
}

// A 16x16 tile has 256 pixels. Each lane owns a compile-time number of pixels,
// so state and feature payloads are loaded once per lane/splat and reductions
// never need a workgroup barrier. Every subgroup leader writes one partial tile
// sum to the fixed-point buffer.
const PIXELS_PER_LANE : u32 = ${h(s)};

@compute @workgroup_size(${h(64)})
fn main(
  @builtin(workgroup_id) wg : vec3u,
  @builtin(local_invocation_index) tid : u32,
  @builtin(subgroup_invocation_id) lane : u32,
) {
  let tileId = wg.x;
  if (tileId >= ${h(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${h(t.cap)});
  let stopc = min(count, tileStop[tileId]);
  let start = tileId * ${h(t.cap)};
  let tileX = tileId % ${h(t.tilesX)};
  let tileY = tileId / ${h(t.tilesX)};
  var valid : array<bool, ${h(s)}>;
  var pixel : array<u32, ${h(s)}>;
  var pxc : array<f32, ${h(s)}>;
  var pyc : array<f32, ${h(s)}>;
  var endi : array<u32, ${h(s)}>;
  var baseR : array<f32, ${h(s)}>; var baseG : array<f32, ${h(s)}>; var baseB : array<f32, ${h(s)}>;
  var l0 : array<f32, ${h(s)}>; var l1 : array<f32, ${h(s)}>; var l2 : array<f32, ${h(s)}>; var l3 : array<f32, ${h(s)}>; var l4 : array<f32, ${h(s)}>;
  var T : array<f32, ${h(s)}>;
  var gBaseR : array<f32, ${h(s)}>; var gBaseG : array<f32, ${h(s)}>; var gBaseB : array<f32, ${h(s)}>;
  var gL0 : array<f32, ${h(s)}>; var gL1 : array<f32, ${h(s)}>; var gL2 : array<f32, ${h(s)}>; var gL3 : array<f32, ${h(s)}>; var gL4 : array<f32, ${h(s)}>;
  var gT : array<f32, ${h(s)}>;
  for (var p = 0u; p < PIXELS_PER_LANE; p++) {
    let localPixel = tid + ${h(64)} * p;
    let x = tileX * ${h(i.TILE)} + (localPixel % ${h(i.TILE)});
    let y = tileY * ${h(i.TILE)} + (localPixel / ${h(i.TILE)});
    valid[p] = x < ${h(t.W)} && y < ${h(t.H)};
    pixel[p] = y * ${h(t.W)} + x;
    pxc[p] = f32(x) + 0.5;
    pyc[p] = f32(y) + 0.5;
    endi[p] = stopc;
    baseR[p] = 0.0; baseG[p] = 0.0; baseB[p] = 0.0;
    l0[p] = 0.0; l1[p] = 0.0; l2[p] = 0.0; l3[p] = 0.0; l4[p] = 0.0;
    T[p] = 1.0;
    gBaseR[p] = 0.0; gBaseG[p] = 0.0; gBaseB[p] = 0.0;
    gL0[p] = 0.0; gL1[p] = 0.0; gL2[p] = 0.0; gL3[p] = 0.0; gL4[p] = 0.0;
    gT[p] = 0.0;
  }

  // Replay the forward pass four pixels at a time. The payload is loaded once
  // per lane/splat, not once per pixel, and stopped pixels remain masked.
  for (var i = 0u; i < stopc; i++) {
    let g = binnedIds[start + i]; let s = g * STATE_STRIDE; let e = g * FEATURE_STRIDE;
    let mx = state[s + ${h(0)}]; let my = state[s + ${h(1)}];
    let a = state[s + ${h(2)}]; let b = state[s + ${h(3)}]; let c = state[s + ${h(4)}];
    let opacity = state[s + ${h(8)}]; let cs = state[s + ${h(9)}]; let sn = state[s + ${h(10)}];
    let invSx = state[s + ${h(11)}]; let invSy = state[s + ${h(12)}];
    let cR = state[s + ${h(5)}]; let cG = state[s + ${h(6)}]; let cB = state[s + ${h(7)}];
    let f0 = features[e]; let f1 = features[e + 1u]; let f2 = features[e + 2u]; let f3 = features[e + 3u]; let f4 = features[e + 4u];
    let fx0 = features[e + 5u]; let fx1 = features[e + 6u]; let fx2 = features[e + 7u]; let fx3 = features[e + 8u]; let fx4 = features[e + 9u];
    let fy0 = features[e + 10u]; let fy1 = features[e + 11u]; let fy2 = features[e + 12u]; let fy3 = features[e + 13u]; let fy4 = features[e + 14u];
    for (var p = 0u; p < PIXELS_PER_LANE; p++) {
      if (valid[p] && i < endi[p]) {
        let dx = pxc[p] - mx; let dy = pyc[p] - my;
        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
        if (power <= 0.0) {
          let alpha = min(${m(i.MAX_ALPHA)}, opacity * exp(power));
          if (alpha >= ${m(i.ALPHA_THRESHOLD)}) {
            let ux = clamp((cs * dx + sn * dy) * invSx, -3.0, 3.0);
            let uy = clamp((-sn * dx + cs * dy) * invSy, -3.0, 3.0);
            let w = T[p] * alpha;
            baseR[p] += w * cR; baseG[p] += w * cG; baseB[p] += w * cB;
            l0[p] += w * (f0 + ux * fx0 + uy * fy0);
            l1[p] += w * (f1 + ux * fx1 + uy * fy1);
            l2[p] += w * (f2 + ux * fx2 + uy * fy2);
            l3[p] += w * (f3 + ux * fx3 + uy * fy3);
            l4[p] += w * (f4 + ux * fx4 + uy * fy4);
            T[p] *= 1.0 - alpha;
            if (T[p] < ${m(i.TRANSMITTANCE_CUTOFF)}) { endi[p] = i + 1u; }
          }
        }
      }
    }
  }

  // Feature decoder VJP, accumulated locally over the four pixels before the
  // subgroup reduction. There is no feature image or feature-image gradient.
  var ld0 = vec4<f32>(0.0); var ld1 = vec4<f32>(0.0); var ld2 = vec4<f32>(0.0);
  var ld3 = vec4<f32>(0.0); var ld4 = vec4<f32>(0.0); var ld5 = vec4<f32>(0.0); var ld6 = vec4<f32>(0.0);
  for (var p = 0u; p < PIXELS_PER_LANE; p++) {
    if (valid[p]) {
      baseR[p] += T[p] * ${m(t.bg[0])}; baseG[p] += T[p] * ${m(t.bg[1])}; baseB[p] += T[p] * ${m(t.bg[2])};
      let goR = gradImage[pixel[p]]; let goG = gradImage[${h(r)} + pixel[p]]; let goB = gradImage[${h(2*r)} + pixel[p]];
      let rR = decoder[24u] + decoder[0u] * baseR[p] + decoder[1u] * baseG[p] + decoder[2u] * baseB[p] + decoder[3u] * l0[p] + decoder[4u] * l1[p] + decoder[5u] * l2[p] + decoder[6u] * l3[p] + decoder[7u] * l4[p];
      let rG = decoder[25u] + decoder[8u] * baseR[p] + decoder[9u] * baseG[p] + decoder[10u] * baseB[p] + decoder[11u] * l0[p] + decoder[12u] * l1[p] + decoder[13u] * l2[p] + decoder[14u] * l3[p] + decoder[15u] * l4[p];
      let rB = decoder[26u] + decoder[16u] * baseR[p] + decoder[17u] * baseG[p] + decoder[18u] * baseB[p] + decoder[19u] * l0[p] + decoder[20u] * l1[p] + decoder[21u] * l2[p] + decoder[22u] * l3[p] + decoder[23u] * l4[p];
      let outR = sigmoid1(logit1(baseR[p]) + RESIDUAL_SCALE * rR);
      let outG = sigmoid1(logit1(baseG[p]) + RESIDUAL_SCALE * rG);
      let outB = sigmoid1(logit1(baseB[p]) + RESIDUAL_SCALE * rB);
      let dzR = goR * outR * (1.0 - outR); let dzG = goG * outG * (1.0 - outG); let dzB = goB * outB * (1.0 - outB);
      let baseRc = clamp(baseR[p], ${m(i.EPS)}, ${m(1-i.EPS)}); let baseGc = clamp(baseG[p], ${m(i.EPS)}, ${m(1-i.EPS)}); let baseBc = clamp(baseB[p], ${m(i.EPS)}, ${m(1-i.EPS)});
      gBaseR[p] = dzR / max(baseRc * (1.0 - baseRc), ${m(i.EPS)}) + RESIDUAL_SCALE * (dzR * decoder[0u] + dzG * decoder[8u] + dzB * decoder[16u]);
      gBaseG[p] = dzG / max(baseGc * (1.0 - baseGc), ${m(i.EPS)}) + RESIDUAL_SCALE * (dzR * decoder[1u] + dzG * decoder[9u] + dzB * decoder[17u]);
      gBaseB[p] = dzB / max(baseBc * (1.0 - baseBc), ${m(i.EPS)}) + RESIDUAL_SCALE * (dzR * decoder[2u] + dzG * decoder[10u] + dzB * decoder[18u]);
      gL0[p] = RESIDUAL_SCALE * (dzR * decoder[3u] + dzG * decoder[11u] + dzB * decoder[19u]);
      gL1[p] = RESIDUAL_SCALE * (dzR * decoder[4u] + dzG * decoder[12u] + dzB * decoder[20u]);
      gL2[p] = RESIDUAL_SCALE * (dzR * decoder[5u] + dzG * decoder[13u] + dzB * decoder[21u]);
      gL3[p] = RESIDUAL_SCALE * (dzR * decoder[6u] + dzG * decoder[14u] + dzB * decoder[22u]);
      gL4[p] = RESIDUAL_SCALE * (dzR * decoder[7u] + dzG * decoder[15u] + dzB * decoder[23u]);
      gT[p] = gBaseR[p] * ${m(t.bg[0])} + gBaseG[p] * ${m(t.bg[1])} + gBaseB[p] * ${m(t.bg[2])};
      ld0 += vec4<f32>(RESIDUAL_SCALE * dzR * baseR[p], RESIDUAL_SCALE * dzR * baseG[p], RESIDUAL_SCALE * dzR * baseB[p], RESIDUAL_SCALE * dzR * l0[p]);
      ld1 += vec4<f32>(RESIDUAL_SCALE * dzR * l1[p], RESIDUAL_SCALE * dzR * l2[p], RESIDUAL_SCALE * dzR * l3[p], RESIDUAL_SCALE * dzR * l4[p]);
      ld2 += vec4<f32>(RESIDUAL_SCALE * dzG * baseR[p], RESIDUAL_SCALE * dzG * baseG[p], RESIDUAL_SCALE * dzG * baseB[p], RESIDUAL_SCALE * dzG * l0[p]);
      ld3 += vec4<f32>(RESIDUAL_SCALE * dzG * l1[p], RESIDUAL_SCALE * dzG * l2[p], RESIDUAL_SCALE * dzG * l3[p], RESIDUAL_SCALE * dzG * l4[p]);
      ld4 += vec4<f32>(RESIDUAL_SCALE * dzB * baseR[p], RESIDUAL_SCALE * dzB * baseG[p], RESIDUAL_SCALE * dzB * baseB[p], RESIDUAL_SCALE * dzB * l0[p]);
      ld5 += vec4<f32>(RESIDUAL_SCALE * dzB * l1[p], RESIDUAL_SCALE * dzB * l2[p], RESIDUAL_SCALE * dzB * l3[p], RESIDUAL_SCALE * dzB * l4[p]);
      ld6 += vec4<f32>(RESIDUAL_SCALE * dzR, RESIDUAL_SCALE * dzG, RESIDUAL_SCALE * dzB, 0.0);
    }
  }
  let td0 = subgroupAdd(ld0); let td1 = subgroupAdd(ld1); let td2 = subgroupAdd(ld2); let td3 = subgroupAdd(ld3);
  let td4 = subgroupAdd(ld4); let td5 = subgroupAdd(ld5); let td6 = subgroupAdd(ld6);
  if (lane == 0u) {
    atomicAdd(&acc[DECODER_OFFSET], qgrad(td0.x)); atomicAdd(&acc[DECODER_OFFSET + 1u], qgrad(td0.y)); atomicAdd(&acc[DECODER_OFFSET + 2u], qgrad(td0.z)); atomicAdd(&acc[DECODER_OFFSET + 3u], qgrad(td0.w));
    atomicAdd(&acc[DECODER_OFFSET + 4u], qgrad(td1.x)); atomicAdd(&acc[DECODER_OFFSET + 5u], qgrad(td1.y)); atomicAdd(&acc[DECODER_OFFSET + 6u], qgrad(td1.z)); atomicAdd(&acc[DECODER_OFFSET + 7u], qgrad(td1.w));
    atomicAdd(&acc[DECODER_OFFSET + 8u], qgrad(td2.x)); atomicAdd(&acc[DECODER_OFFSET + 9u], qgrad(td2.y)); atomicAdd(&acc[DECODER_OFFSET + 10u], qgrad(td2.z)); atomicAdd(&acc[DECODER_OFFSET + 11u], qgrad(td2.w));
    atomicAdd(&acc[DECODER_OFFSET + 12u], qgrad(td3.x)); atomicAdd(&acc[DECODER_OFFSET + 13u], qgrad(td3.y)); atomicAdd(&acc[DECODER_OFFSET + 14u], qgrad(td3.z)); atomicAdd(&acc[DECODER_OFFSET + 15u], qgrad(td3.w));
    atomicAdd(&acc[DECODER_OFFSET + 16u], qgrad(td4.x)); atomicAdd(&acc[DECODER_OFFSET + 17u], qgrad(td4.y)); atomicAdd(&acc[DECODER_OFFSET + 18u], qgrad(td4.z)); atomicAdd(&acc[DECODER_OFFSET + 19u], qgrad(td4.w));
    atomicAdd(&acc[DECODER_OFFSET + 20u], qgrad(td5.x)); atomicAdd(&acc[DECODER_OFFSET + 21u], qgrad(td5.y)); atomicAdd(&acc[DECODER_OFFSET + 22u], qgrad(td5.z)); atomicAdd(&acc[DECODER_OFFSET + 23u], qgrad(td5.w));
    atomicAdd(&acc[DECODER_OFFSET + 24u], qgrad(td6.x)); atomicAdd(&acc[DECODER_OFFSET + 25u], qgrad(td6.y)); atomicAdd(&acc[DECODER_OFFSET + 26u], qgrad(td6.z));
  }

  // Reverse alpha recurrence. Each lane accumulates its four pixels, then its
  // hardware subgroup emits the tile partial. No cross-subgroup shared state
  // and no per-splat barrier are required.
  for (var ii = i32(stopc) - 1; ii >= 0; ii--) {
    var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0);
    var v3 = vec4<f32>(0.0); var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0);
    var v6 = vec4<f32>(0.0); var v7 = 0.0;
    let g = binnedIds[start + u32(ii)]; let s = g * STATE_STRIDE; let e = g * FEATURE_STRIDE;
    let mx = state[s + ${h(0)}]; let my = state[s + ${h(1)}];
    let a = state[s + ${h(2)}]; let b = state[s + ${h(3)}]; let c = state[s + ${h(4)}];
    let opacity = state[s + ${h(8)}]; let cs = state[s + ${h(9)}]; let sn = state[s + ${h(10)}];
    let invSx = state[s + ${h(11)}]; let invSy = state[s + ${h(12)}];
    let cR = state[s + ${h(5)}]; let cG = state[s + ${h(6)}]; let cB = state[s + ${h(7)}];
    let scaleGateX = state[s + ${h(13)}]; let scaleGateY = state[s + ${h(14)}];
    let f0 = features[e]; let f1 = features[e + 1u]; let f2 = features[e + 2u]; let f3 = features[e + 3u]; let f4 = features[e + 4u];
    let fx0 = features[e + 5u]; let fx1 = features[e + 6u]; let fx2 = features[e + 7u]; let fx3 = features[e + 8u]; let fx4 = features[e + 9u];
    let fy0 = features[e + 10u]; let fy1 = features[e + 11u]; let fy2 = features[e + 12u]; let fy3 = features[e + 13u]; let fy4 = features[e + 14u];
    for (var p = 0u; p < PIXELS_PER_LANE; p++) {
      if (valid[p] && u32(ii) < endi[p]) {
        let dx = pxc[p] - mx; let dy = pyc[p] - my;
        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
        if (power <= 0.0) {
          let raw = opacity * exp(power); let alpha = min(${m(i.MAX_ALPHA)}, raw);
          if (alpha >= ${m(i.ALPHA_THRESHOLD)}) {
            let denom = max(1.0 - alpha, ${m(i.EPS)}); let Tprev = T[p] / denom;
            let uxRaw = (cs * dx + sn * dy) * invSx; let uyRaw = (-sn * dx + cs * dy) * invSy;
            let ux = clamp(uxRaw, -3.0, 3.0); let uy = clamp(uyRaw, -3.0, 3.0);
            let z0 = f0 + ux * fx0 + uy * fy0; let z1 = f1 + ux * fx1 + uy * fy1; let z2 = f2 + ux * fx2 + uy * fy2;
            let z3 = f3 + ux * fx3 + uy * fy3; let z4 = f4 + ux * fx4 + uy * fy4;
            let dotPayload = gBaseR[p] * cR + gBaseG[p] * cG + gBaseB[p] * cB + gL0[p] * z0 + gL1[p] * z1 + gL2[p] * z2 + gL3[p] * z3 + gL4[p] * z4;
            let gAlpha = Tprev * (dotPayload - gT[p]); let w = Tprev * alpha;
            let gUx = select(0.0, w * (gL0[p] * fx0 + gL1[p] * fx1 + gL2[p] * fx2 + gL3[p] * fx3 + gL4[p] * fx4), uxRaw > -3.0 && uxRaw < 3.0);
            let gUy = select(0.0, w * (gL0[p] * fy0 + gL1[p] * fy1 + gL2[p] * fy2 + gL3[p] * fy3 + gL4[p] * fy4), uyRaw > -3.0 && uyRaw < 3.0);
            let gRaw = gAlpha * select(0.0, 1.0, raw < ${m(i.MAX_ALPHA)}); let gPower = gRaw * raw;
            let gdx = gPower * (-(a * dx + b * dy)); let gdy = gPower * (-(b * dx + c * dy));
            v0 += vec4<f32>(-gdx, -gdy, gPower * (-0.5) * dx * dx, gPower * (-1.0) * dx * dy);
            v1 += vec4<f32>(gPower * (-0.5) * dy * dy, gBaseR[p] * w, gBaseG[p] * w, gBaseB[p] * w);
            v2 += vec4<f32>(gRaw * (raw / max(opacity, ${m(i.EPS)})), gL0[p] * w, gL1[p] * w, gL2[p] * w);
            v3 += vec4<f32>(gL3[p] * w, gL4[p] * w, gL0[p] * w * ux, gL1[p] * w * ux);
            v4 += vec4<f32>(gL2[p] * w * ux, gL3[p] * w * ux, gL4[p] * w * ux, gL0[p] * w * uy);
            v5 += vec4<f32>(gL1[p] * w * uy, gL2[p] * w * uy, gL3[p] * w * uy, gL4[p] * w * uy);
            v6 += vec4<f32>(gUx * (-cs * invSx) + gUy * (sn * invSy), gUx * (-sn * invSx) + gUy * (-cs * invSy), gUx * (-uxRaw) * scaleGateX, gUy * (-uyRaw) * scaleGateY);
            v7 += gUx * ((-sn * dx + cs * dy) * invSx) + gUy * ((-cs * dx - sn * dy) * invSy);
            gT[p] = alpha * dotPayload + (1.0 - alpha) * gT[p];
            T[p] = Tprev;
          }
        }
      }
    }
    let r0 = subgroupAdd(v0);
    let r1 = subgroupAdd(v1);
    let r2 = subgroupAdd(v2);
    let r3 = subgroupAdd(v3);
    let r4 = subgroupAdd(v4);
    let r5 = subgroupAdd(v5);
    let r6 = subgroupAdd(v6);
    let r7 = subgroupAdd(v7);
    if (lane == 0u) {
      let ab = g * ACC_STRIDE;
      atomicAdd(&acc[ab], qgrad(r0.x)); atomicAdd(&acc[ab + 1u], qgrad(r0.y)); atomicAdd(&acc[ab + 2u], qgrad(r0.z)); atomicAdd(&acc[ab + 3u], qgrad(r0.w));
      atomicAdd(&acc[ab + 4u], qgrad(r1.x)); atomicAdd(&acc[ab + 5u], qgrad(r1.y)); atomicAdd(&acc[ab + 6u], qgrad(r1.z)); atomicAdd(&acc[ab + 7u], qgrad(r1.w));
      atomicAdd(&acc[ab + 8u], qgrad(r2.x)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET], qgrad(r2.y)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 1u], qgrad(r2.z)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 2u], qgrad(r2.w));
      atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 3u], qgrad(r3.x)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 4u], qgrad(r3.y)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 5u], qgrad(r3.z)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 6u], qgrad(r3.w));
      atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 7u], qgrad(r4.x)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 8u], qgrad(r4.y)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 9u], qgrad(r4.z)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 10u], qgrad(r4.w));
      atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 11u], qgrad(r5.x)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 12u], qgrad(r5.y)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 13u], qgrad(r5.z)); atomicAdd(&acc[ab + ACC_EXTRA_OFFSET + 14u], qgrad(r5.w));
      atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET], qgrad(r6.x)); atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET + 1u], qgrad(r6.y)); atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET + 2u], qgrad(r6.z)); atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET + 3u], qgrad(r6.w));
      atomicAdd(&acc[ab + ACC_LOCAL_RAW_OFFSET + 4u], qgrad(r7));
    }
  }
}
`}function A(e){let{d:t,code:r}=y(e),a=2*t.G,i=4*t.G,s=5*t.G,d=8*t.G;return/* wgsl */`
${r}
@group(0) @binding(0) var<storage, read> acc : array<i32>;
@group(0) @binding(1) var<storage, read> state : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${h(t.G)}) { return; }
  let ab = g * ACC_STRIDE;
  let sb = g * STATE_STRIDE;
  let inv = ${m(1/t.gradScale)};
  let gmx = f32(acc[ab]) * inv;
  let gmy = f32(acc[ab + 1u]) * inv;
  let gA = f32(acc[ab + 2u]) * inv;
  let gB = f32(acc[ab + 3u]) * inv;
  let gC = f32(acc[ab + 4u]) * inv;
  let gc0 = f32(acc[ab + 5u]) * inv;
  let gc1 = f32(acc[ab + 6u]) * inv;
  let gc2 = f32(acc[ab + 7u]) * inv;
  let gop = f32(acc[ab + 8u]) * inv;
  let invSx = state[sb + ${h(11)}]; let invSy = state[sb + ${h(12)}];
  let ix = invSx * invSx; let iy = invSy * invSy;
  let cs = state[sb + ${h(9)}]; let sn = state[sb + ${h(10)}];
  let gix = gA * cs * cs + gB * cs * sn + gC * sn * sn;
  let giy = gA * sn * sn - gB * cs * sn + gC * cs * cs;
  let glsx = gix * (-2.0 * ix) * state[sb + ${h(13)}];
  let glsy = giy * (-2.0 * iy) * state[sb + ${h(14)}];
  let gth = (ix - iy) * ((cs * cs - sn * sn) * gB + 2.0 * cs * sn * (gC - gA));
  let lmx = f32(acc[ab + ACC_LOCAL_RAW_OFFSET]) * inv;
  let lmy = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 1u]) * inv;
  let llsx = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 2u]) * inv;
  let llsy = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 3u]) * inv;
  let lth = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 4u]) * inv;
  let color0 = state[sb + ${h(5)}]; let color1 = state[sb + ${h(6)}]; let color2 = state[sb + ${h(7)}];
  let opacity = state[sb + ${h(8)}];
  gradRaw[${h(0)} + g * 2u] = gmx + lmx;
  gradRaw[${h(0)} + g * 2u + 1u] = gmy + lmy;
  gradRaw[${h(a)} + g * 2u] = glsx + llsx;
  gradRaw[${h(a)} + g * 2u + 1u] = glsy + llsy;
  gradRaw[${h(i)} + g] = gth + lth;
  gradRaw[${h(s)} + g * 3u] = gc0 * color0 * (1.0 - color0);
  gradRaw[${h(s)} + g * 3u + 1u] = gc1 * color1 * (1.0 - color1);
  gradRaw[${h(s)} + g * 3u + 2u] = gc2 * color2 * (1.0 - color2);
  gradRaw[${h(d)} + g] = gop * opacity * (1.0 - opacity);
}
`}function _(e){let{d:t,code:r}=y(e);return/* wgsl */`
${r}
@group(0) @binding(0) var<storage, read> acc : array<i32>;
@group(0) @binding(1) var<storage, read_write> gradFeatures : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradDecoder : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  let inv = ${m(1/t.gradScale)};
  if (i < ${h(t.G)}) {
    let g = i;
    let ab = g * ACC_STRIDE + ACC_EXTRA_OFFSET;
    let fb = g * FEATURE_STRIDE;
    for (var ch = 0u; ch < FEATURE_STRIDE; ch++) { gradFeatures[fb + ch] = f32(acc[ab + ch]) * inv; }
  }
  if (i < ${h(n)}) { gradDecoder[i] = f32(acc[DECODER_OFFSET + i]) * inv; }
}
`}let w=e=>Math.max((0,i.resolveDims)(e).G,n)},{"./raster_wgsl":"6IBEA","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],jHKeo:[function(e,t,r){/** Runtime owner for the fused compact 2D Feature8 painter. *//// <reference types="@webgpu/types" />
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"FEATURE_LR",()=>u),a.export(r,"DECODER_LR",()=>l),/**
 * Compact feature rasterizer with exact RGB-skip initialization.
 *
 * `params` owns normal splat geometry/RGB logits. `featureParams` owns only
 * z/Ax/Ay, and `decoderParams` owns a 3x8 residual projection. All image-space
 * feature work happens in registers inside the tile shaders.
 */a.export(r,"FeaturePainterEngine",()=>h);var i=e("./adam_wgsl"),s=e("./raster_wgsl"),d=e("./feature_painter_wgsl");let o={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},n=e=>Math.ceil(e/256),u=.025,l=.03,c=[{offset:3,length:5},{offset:11,length:5},{offset:19,length:5}];function p(e,t){return t?e.beginComputePass({timestampWrites:t}):e.beginComputePass()}async function g(e,t,r){e.pushErrorScope("validation");let a=e.createShaderModule({label:r,code:t}),i=e.createComputePipeline({label:r,layout:"auto",compute:{module:a,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw Error(`feature painter ${r}: ${s.message}`);return i}function f(e){return e.createBuffer({size:i.ADAM_UNIFORM_BYTES,usage:o.UNIFORM|o.COPY_DST})}function m(e,t,r,a,i,s,d){let o=new ArrayBuffer(32),n=new Uint32Array(o),u=new Float32Array(o);n[0]=r,n[1]=a,u[2]=i,u[3]=d.beta1,u[4]=d.beta2,u[5]=d.eps,u[6]=1-Math.pow(d.beta1,s),u[7]=1-Math.pow(d.beta2,s),e.queue.writeBuffer(t,0,o)}class h{constructor(e,t){this.geometryAdamUniforms=[],this.geometryAdamGroups=[],this.decoderAdamUniforms=[],this.decoderAdamGroups=[],this.device=e,this.dims=(0,s.resolveDims)(t);let r=this.dims,a=(t,r=0)=>e.createBuffer({size:4*t,usage:o.STORAGE|r}),i=r.G*s.PARAM_STRIDE,n=r.G*d.FEATURE_STRIDE,u=r.G*d.FEATURE_ACC_STRIDE+d.DECODER_PARAM_COUNT;this.params=a(i,o.COPY_SRC|o.COPY_DST),this.featureParams=a(n,o.COPY_SRC|o.COPY_DST),this.decoderParams=a(d.DECODER_PARAM_COUNT,o.COPY_SRC|o.COPY_DST),this.image=a(3*r.H*r.W,o.COPY_SRC),this.gradImage=a(3*r.H*r.W,o.COPY_DST),this.state=a(r.G*d.FEATURE_STATE_STRIDE),this.tileCounts=a(r.numTiles,o.COPY_DST|o.COPY_SRC),this.binnedIds=a(r.numTiles*r.cap),this.tileStop=a(r.numTiles,o.COPY_SRC),this.acc=a(u,o.COPY_DST),this.gradGeom=a(i,o.COPY_SRC),this.geomM=a(i,o.COPY_DST),this.geomV=a(i,o.COPY_DST),this.gradFeature=a(n,o.COPY_SRC),this.featureM=a(n,o.COPY_DST),this.featureV=a(n,o.COPY_DST),this.gradDecoder=a(d.DECODER_PARAM_COUNT,o.COPY_SRC),this.decoderM=a(d.DECODER_PARAM_COUNT,o.COPY_DST),this.decoderV=a(d.DECODER_PARAM_COUNT,o.COPY_DST),this.featureAdamUniform=f(e),this.geometrySegments=[{offset:0,length:2*r.G,lr:0},{offset:2*r.G,length:2*r.G,lr:0},{offset:4*r.G,length:r.G,lr:0},{offset:5*r.G,length:3*r.G,lr:0},{offset:8*r.G,length:r.G,lr:0}]}static async create(e,t){if(!e.features.has("subgroups"))throw Error("Feature Painter requires the WebGPU subgroups feature on this build");let r=new h(e,t);return await r.build(t),r}async build(e){let t=this.dims;this.prepPipe=await g(this.device,(0,d.featurePrepShader)(e),"feature8-prep"),this.emitPipe=await g(this.device,(0,d.featureEmitShader)(e),"feature8-emit"),this.forwardPipe=await g(this.device,(0,d.featureForwardShader)(e),"feature8-forward"),this.backwardPipe=await g(this.device,(0,d.featureBackwardShader)(e),"feature8-backward-subgroups"),this.geometryChainPipe=await g(this.device,(0,d.featureGeometryChainShader)(e),"feature8-geometry-chain"),this.featureChainPipe=await g(this.device,(0,d.featureChainShader)(e),"feature8-feature-chain"),this.clearBinsPipe=await g(this.device,(0,s.clearShader)(t.numTiles),"feature8-clear-bins"),this.clearAccPipe=await g(this.device,(0,s.clearShader)(t.G*d.FEATURE_ACC_STRIDE+d.DECODER_PARAM_COUNT),"feature8-clear-acc"),this.adamPipe=await g(this.device,(0,i.adamShader)(),"feature8-adam");let r=(e,t)=>this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:t.map((e,t)=>({binding:t,resource:{buffer:e}}))});this.prepBind=r(this.prepPipe,[this.params,this.state]),this.emitBind=r(this.emitPipe,[this.state,this.tileCounts,this.binnedIds]),this.forwardBind=r(this.forwardPipe,[this.tileCounts,this.binnedIds,this.state,this.featureParams,this.decoderParams,this.image,this.tileStop]),this.backwardBind=r(this.backwardPipe,[this.gradImage,this.tileCounts,this.binnedIds,this.tileStop,this.state,this.featureParams,this.decoderParams,this.acc]),this.geometryChainBind=r(this.geometryChainPipe,[this.acc,this.state,this.gradGeom]),this.featureChainBind=r(this.featureChainPipe,[this.acc,this.gradFeature,this.gradDecoder]),this.clearBinsBind=r(this.clearBinsPipe,[this.tileCounts]),this.clearAccBind=r(this.clearAccPipe,[this.acc]);let a=this.geometrySegments.map(e=>{let t=f(this.device);return this.geometryAdamUniforms.push(t),this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradGeom}},{binding:3,resource:{buffer:this.geomM}},{binding:4,resource:{buffer:this.geomV}}]})});for(let e of(this.geometryAdamGroups.push(...a),this.featureAdamGroup=this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.featureAdamUniform}},{binding:1,resource:{buffer:this.featureParams}},{binding:2,resource:{buffer:this.gradFeature}},{binding:3,resource:{buffer:this.featureM}},{binding:4,resource:{buffer:this.featureV}}]}),c)){let e=f(this.device);this.decoderAdamUniforms.push(e),this.decoderAdamGroups.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.decoderParams}},{binding:2,resource:{buffer:this.gradDecoder}},{binding:3,resource:{buffer:this.decoderM}},{binding:4,resource:{buffer:this.decoderV}}]}))}}setParams(e){if(e.length!==this.dims.G*s.PARAM_STRIDE)throw Error("feature painter: wrong geometry parameter count");this.device.queue.writeBuffer(this.params,0,e)}setFeatureParams(e){if(e.length!==this.dims.G*d.FEATURE_STRIDE)throw Error("feature painter: wrong feature parameter count");this.device.queue.writeBuffer(this.featureParams,0,e)}setDecoderParams(e){if(e.length!==d.DECODER_PARAM_COUNT)throw Error("feature painter: wrong decoder parameter count");this.device.queue.writeBuffer(this.decoderParams,0,e)}setGradImage(e){if(e.length!==3*this.dims.H*this.dims.W)throw Error("feature painter: wrong image gradient count");this.device.queue.writeBuffer(this.gradImage,0,e)}async read(e,t){let r=this.device.createBuffer({size:4*t,usage:o.MAP_READ|o.COPY_DST}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(GPUMapMode.READ);let i=new Float32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}async readU32(e,t){let r=this.device.createBuffer({size:4*t,usage:o.MAP_READ|o.COPY_DST}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(GPUMapMode.READ);let i=new Uint32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}readParams(){return this.read(this.params,this.dims.G*s.PARAM_STRIDE)}readFeatureParams(){return this.read(this.featureParams,this.dims.G*d.FEATURE_STRIDE)}readDecoderParams(){return this.read(this.decoderParams,d.DECODER_PARAM_COUNT)}readGeometryGradient(){return this.read(this.gradGeom,this.dims.G*s.PARAM_STRIDE)}readFeatureGradient(){return this.read(this.gradFeature,this.dims.G*d.FEATURE_STRIDE)}readDecoderGradient(){return this.read(this.gradDecoder,d.DECODER_PARAM_COUNT)}readImage(){return this.read(this.image,3*this.dims.H*this.dims.W)}async readTileTelemetry(){let e=await this.readU32(this.tileCounts,this.dims.numTiles),t=await this.readU32(this.tileStop,this.dims.numTiles),r=0,a=0,i=0,s=0,d=0,o=0;for(let n=0;n<this.dims.numTiles;n++){let u=e[n],l=t[n];r+=u,a+=l,i=Math.max(i,u),s=Math.max(s,l),u>this.dims.cap&&(d++,o+=u-this.dims.cap)}return{meanCount:r/this.dims.numTiles,maxCount:i,meanStop:a/this.dims.numTiles,maxStop:s,overflowTiles:d,overflowEntries:o}}zeroAdamState(){let e=e=>new Float32Array(e),t=e(this.dims.G*s.PARAM_STRIDE),r=e(this.dims.G*d.FEATURE_STRIDE),a=e(d.DECODER_PARAM_COUNT);this.device.queue.writeBuffer(this.geomM,0,t),this.device.queue.writeBuffer(this.geomV,0,t),this.device.queue.writeBuffer(this.featureM,0,r),this.device.queue.writeBuffer(this.featureV,0,r),this.device.queue.writeBuffer(this.decoderM,0,a),this.device.queue.writeBuffer(this.decoderV,0,a)}recordForward(e,t){let r=p(e,t);r.setPipeline(this.prepPipe),r.setBindGroup(0,this.prepBind),r.dispatchWorkgroups(n(this.dims.G)),r.setPipeline(this.clearBinsPipe),r.setBindGroup(0,this.clearBinsBind),r.dispatchWorkgroups(n(this.dims.numTiles)),r.setPipeline(this.emitPipe),r.setBindGroup(0,this.emitBind),r.dispatchWorkgroups(n(this.dims.G)),r.setPipeline(this.forwardPipe),r.setBindGroup(0,this.forwardBind),r.dispatchWorkgroups(this.dims.numTiles),r.end()}recordBackward(e,t){let r=p(e,t);r.setPipeline(this.clearAccPipe),r.setBindGroup(0,this.clearAccBind),r.dispatchWorkgroups(n(this.dims.G*d.FEATURE_ACC_STRIDE+d.DECODER_PARAM_COUNT)),r.setPipeline(this.backwardPipe),r.setBindGroup(0,this.backwardBind),r.dispatchWorkgroups(this.dims.numTiles),r.setPipeline(this.geometryChainPipe),r.setBindGroup(0,this.geometryChainBind),r.dispatchWorkgroups(n(this.dims.G)),r.setPipeline(this.featureChainPipe),r.setBindGroup(0,this.featureChainBind),r.dispatchWorkgroups(n((0,d.FEATURE_CHAIN_WORK_ITEMS)(this.dims))),r.end()}recordAdam(e,t,r,a=i.DEFAULT_HYPER,s,o={}){let g=[r.mean,r.logScale,r.theta,r.color,r.opacity];this.geometrySegments.forEach((e,r)=>{m(this.device,this.geometryAdamUniforms[r],e.offset,e.length,g[r],t,a)}),m(this.device,this.featureAdamUniform,0,this.dims.G*d.FEATURE_STRIDE,o.feature??u,t,a),c.forEach((e,r)=>{m(this.device,this.decoderAdamUniforms[r],e.offset,e.length,o.decoder??l,t,a)});let f=p(e,s);f.setPipeline(this.adamPipe),this.geometrySegments.forEach((e,t)=>{f.setBindGroup(0,this.geometryAdamGroups[t]),f.dispatchWorkgroups(n(e.length))}),f.setBindGroup(0,this.featureAdamGroup),f.dispatchWorkgroups(n(this.dims.G*d.FEATURE_STRIDE)),this.decoderAdamGroups.forEach((e,t)=>{f.setBindGroup(0,e),f.dispatchWorkgroups(n(c[t].length))}),f.end()}runForward(){let e=this.device.createCommandEncoder();this.recordForward(e),this.device.queue.submit([e.finish()])}destroy(){let e=[this.params,this.featureParams,this.decoderParams,this.image,this.gradImage,this.state,this.tileCounts,this.binnedIds,this.tileStop,this.acc,this.gradGeom,this.geomM,this.geomV,this.gradFeature,this.featureM,this.featureV,this.gradDecoder,this.decoderM,this.decoderV,this.featureAdamUniform,...this.decoderAdamUniforms,...this.geometryAdamUniforms];for(let t of e)try{t.destroy()}catch{}}}},{"./adam_wgsl":"kfWkJ","./raster_wgsl":"6IBEA","./feature_painter_wgsl":"4Oo98","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],drkYM:[function(e,t,r){/**
 * Direct trainable image baseline for the prompt-to-splats page.
 *
 * This deliberately owns no geometry: one raw RGB logit per output pixel is
 * optimized against the same frozen MobileCLIP trainer used by both splat
 * renderers. It is an ablation/control, not a claim that unconstrained pixels
 * are a good image prior. Comparing it with splats tells us whether a failed
 * run is CLIP's objective or the splat representation/optimizer.
 *//// <reference types="@webgpu/types" />
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"PIXEL_BUFFER_LR",()=>u),/** GPU image parameter buffer consumed by PixelBufferOptimizer. */a.export(r,"PixelBufferEngine",()=>c),a.export(r,"PixelBufferOptimizer",()=>p),/** Near-neutral, low-amplitude noise. The learned tensor is raw RGB logits. */a.export(r,"randomPixelLogits",()=>g),a.export(r,"cosine",()=>d.cosine);var i=e("./adam_wgsl"),s=e("../clip/vision"),d=e("./optimize");let o={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},n=e=>Math.ceil(e/256),u=.08;async function l(e,t,r){e.pushErrorScope("validation");let a=e.createShaderModule({label:r,code:t}),i=e.createComputePipeline({label:r,layout:"auto",compute:{module:a,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw Error(`pixel buffer ${r}: ${s.message}`);return i}class c{static async create(e,t=1){let r=e.createBuffer({size:786432,usage:o.STORAGE|o.COPY_SRC|o.COPY_DST}),a=e.createBuffer({size:786432,usage:o.STORAGE|o.COPY_SRC}),s=e.createBuffer({size:786432,usage:o.STORAGE|o.COPY_DST}),d=e.createBuffer({size:786432,usage:o.STORAGE|o.COPY_SRC}),n=e.createBuffer({size:786432,usage:o.STORAGE|o.COPY_DST}),u=e.createBuffer({size:786432,usage:o.STORAGE|o.COPY_DST}),p=e.createBuffer({size:i.ADAM_UNIFORM_BYTES,usage:o.UNIFORM|o.COPY_DST}),[f,m,h]=await Promise.all([l(e,/* wgsl */`
@group(0) @binding(0) var<storage, read> raw : array<f32>;
@group(0) @binding(1) var<storage, read_write> image : array<f32>;
fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= 196608u) { return; }
  image[i] = sigmoid1(raw[i]);
}
`,"pixel-buffer-forward"),l(e,/* wgsl */`
@group(0) @binding(0) var<storage, read> image : array<f32>;
@group(0) @binding(1) var<storage, read> gradImage : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradRaw : array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= 196608u) { return; }
  let rgb = image[i];
  gradRaw[i] = gradImage[i] * rgb * (1.0 - rgb);
}
`,"pixel-buffer-chain"),l(e,(0,i.adamShader)(),"pixel-buffer-adam")]),y=e.createBindGroup({layout:f.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:a}}]}),b=e.createBindGroup({layout:m.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:a}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:d}}]}),x=e.createBindGroup({layout:h.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:r}},{binding:2,resource:{buffer:d}},{binding:3,resource:{buffer:n}},{binding:4,resource:{buffer:u}}]}),E=new c(e,r,a,s,d,n,u,p,f,m,h,y,b,x);return E.setRaw(g(t)),E.zeroAdamState(),E}constructor(e,t,r,a,i,s,d,o,n,u,l,c,p,g){this.device=e,this.raw=t,this.image=r,this.gradImage=a,this.gradRaw=i,this.m=s,this.v=d,this.adamUniform=o,this.forwardPipe=n,this.chainPipe=u,this.adamPipe=l,this.forwardBind=c,this.chainBind=p,this.adamBind=g}setRaw(e){if(196608!==e.length)throw Error("pixel buffer: wrong raw image length");this.device.queue.writeBuffer(this.raw,0,e)}zeroAdamState(){let e=new Float32Array(196608);this.device.queue.writeBuffer(this.m,0,e),this.device.queue.writeBuffer(this.v,0,e)}recordForward(e){let t=e.beginComputePass();t.setPipeline(this.forwardPipe),t.setBindGroup(0,this.forwardBind),t.dispatchWorkgroups(n(196608)),t.end()}recordBackward(e){let t=e.beginComputePass();t.setPipeline(this.chainPipe),t.setBindGroup(0,this.chainBind),t.dispatchWorkgroups(n(196608)),t.end()}recordAdam(e,t,r=u,a=i.DEFAULT_HYPER){!function(e,t,r,a,s){let d=new ArrayBuffer(i.ADAM_UNIFORM_BYTES),o=new Uint32Array(d),n=new Float32Array(d);o[0]=0,o[1]=196608,n[2]=a,n[3]=s.beta1,n[4]=s.beta2,n[5]=s.eps,n[6]=1-Math.pow(s.beta1,r),n[7]=1-Math.pow(s.beta2,r),e.queue.writeBuffer(t,0,d)}(this.device,this.adamUniform,t,r,a);let s=e.beginComputePass();s.setPipeline(this.adamPipe),s.setBindGroup(0,this.adamBind),s.dispatchWorkgroups(n(196608)),s.end()}runForward(){let e=this.device.createCommandEncoder();this.recordForward(e),this.device.queue.submit([e.finish()])}async readImage(){return this.read(this.image)}async readRaw(){return this.read(this.raw)}async read(e){let t=this.device.createBuffer({size:786432,usage:o.MAP_READ|o.COPY_DST}),r=this.device.createCommandEncoder();r.copyBufferToBuffer(e,0,t,0,786432),this.device.queue.submit([r.finish()]),await t.mapAsync(GPUMapMode.READ);let a=new Float32Array(t.getMappedRange().slice(0));return t.unmap(),t.destroy(),a}destroy(){for(let e of[this.raw,this.image,this.gradImage,this.gradRaw,this.m,this.v,this.adamUniform])try{e.destroy()}catch{}}}class p{static async create(e,t,r,a=1){let[i,d,o]=t.inputShape;if(3!==i||256!==d||256!==o)throw Error("pixel buffer requires MobileCLIP 256x256 RGB input");let[n,u]=await Promise.all([c.create(e,a),(0,s.VisionTrainer).create(e,t,r)]);return new p(e,n,u)}constructor(e,t,r){this.side=256,this.step_=0,this.device=e,this.raster=t,this.trainer=r}setPrompt(e){this.trainer.writeText(e)}get stepCount(){return this.step_}step(){let e=this.device.createCommandEncoder();this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!0}),e.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackward(e),this.step_+=1,this.raster.recordAdam(e,this.step_),this.device.queue.submit([e.finish()])}async nudge(e=Date.now(),t=.12){let r=await this.raster.readRaw(),a=g(e),i=Math.max(0,Math.min(1,t)),s=(2654435769^e)>>>0||1;for(let e=0;e<r.length;e++)(s=Math.imul(s,1664525)+1013904223>>>0)/4294967296<i&&(r[e]=a[e]);this.raster.setRaw(r),this.raster.zeroAdamState()}async renderImage(){return this.raster.runForward(),this.raster.readImage()}async currentEmbedding(){let e=this.device.createCommandEncoder();return this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!1}),this.device.queue.submit([e.finish()]),f(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){this.raster.destroy()}}function g(e=1){let t=new Float32Array(196608),r=e>>>0||1,a=()=>(r=Math.imul(r,1664525)+1013904223>>>0)/4294967296;for(let e=0;e<t.length;e++)t[e]=(a()-.5)*.16;return t}async function f(e,t,r){let a=e.createBuffer({size:4*r,usage:o.MAP_READ|o.COPY_DST}),i=e.createCommandEncoder();i.copyBufferToBuffer(t,0,a,0,4*r),e.queue.submit([i.finish()]),await a.mapAsync(GPUMapMode.READ);let s=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),a.destroy(),s}},{"./adam_wgsl":"kfWkJ","../clip/vision":"lNzsi","./optimize":"nZSdJ","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}]},["7i9mK"],"7i9mK","parcelRequire924a")//# sourceMappingURL=splat.75d61259.js.map
;
//# sourceMappingURL=splat.75d61259.js.map
