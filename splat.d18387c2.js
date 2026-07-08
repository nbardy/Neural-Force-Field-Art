!// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles
function(e,r,t,i,a){/* eslint-disable no-undef */var o="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof self?self:"undefined"!=typeof window?window:"undefined"!=typeof global?global:{},s="function"==typeof o[i]&&o[i],u=s.cache||{},n="undefined"!=typeof module&&"function"==typeof module.require&&module.require.bind(module);function d(r,t){if(!u[r]){if(!e[r]){// if we cannot find the module within our internal map or
// cache jump to the current global require ie. the last bundle
// that was added to the page.
var a="function"==typeof o[i]&&o[i];if(!t&&a)return a(r,!0);// If there are other bundles on this page the require from the
// previous one is saved to 'previousRequire'. Repeat this as
// many times as there are bundles until the module is found or
// we exhaust the require chain.
if(s)return s(r,!0);// Try the node require function if it exists.
if(n&&"string"==typeof r)return n(r);var c=Error("Cannot find module '"+r+"'");throw c.code="MODULE_NOT_FOUND",c}p.resolve=function(t){var i=e[r][1][t];return null!=i?i:t},p.cache={};var l=u[r]=new d.Module(r);e[r][0].call(l.exports,p,l,l.exports,this)}return u[r].exports;function p(e){var r=p.resolve(e);return!1===r?{}:d(r)}}d.isParcelRequire=!0,d.Module=function(e){this.id=e,this.bundle=d,this.exports={}},d.modules=e,d.cache=u,d.parent=s,d.register=function(r,t){e[r]=[function(e,r){r.exports=t},{}]},Object.defineProperty(d,"root",{get:function(){return o[i]}}),o[i]=d;for(var c=0;c<r.length;c++)d(r[c]);if(t){// Expose entry point to Node, AMD or browser globals
// Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
var l=d(t);"object"==typeof exports&&"undefined"!=typeof module?module.exports=l:"function"==typeof define&&define.amd?define(function(){return l}):a&&(this[a]=l)}}({"7i9mK":[function(e,r,t){let i,a,o,s,u,n,d;/**
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
var c=e("./splat/optimize"),l=e("./splat/fetch_progress");let p={gpu:!!navigator.gpu,ready:!1,running:!1,step:0,cos:null,initialCos:null,error:null,phase:"boot"};window.__splat=p;// ── DOM ──────────────────────────────────────────────────────────────────────
let f=document.getElementById("splat"),g=document.getElementById("prompt"),h=document.getElementById("optimize"),m=document.getElementById("nudge"),v=document.getElementById("reset"),$=document.getElementById("readout"),w=document.getElementById("notice");function b(e){w.textContent=e}function x(e){p.error=e,p.phase="error",b(e),$.textContent="—",// eslint-disable-next-line no-console
console.error("[splat_page]",e)}function y(){p.step=u?u.stepCount:0;let e=[`step ${p.step}`];if(null!==p.cos){let r=p.initialCos??p.cos,t=p.cos-r;e.push(`cos ${p.cos.toFixed(4)}`),e.push(`init ${r.toFixed(4)}`),e.push(`Δ ${t>=0?"+":""}${t.toFixed(4)}`)}p.phase&&"run"!==p.phase&&e.push(`(${p.phase})`),$.textContent=e.join("  \xb7  ")}let k=1,_=null,B=/* wgsl */`
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
`;async function C(){i.pushErrorScope("validation");let e=i.createShaderModule({code:B});n=i.createRenderPipeline({layout:"auto",vertex:{module:e,entryPoint:"vs"},fragment:{module:e,entryPoint:"fs",targets:[{format:d}]},primitive:{topology:"triangle-list"}});let r=await i.popErrorScope();if(r)throw Error(`blit pipeline invalid: ${r.message}`)}function P(){_=i.createBindGroup({layout:n.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:u.raster.image}}]})}// Hide the specifier behind a Function-constructor indirection so the BUNDLER
// leaves it alone and the BROWSER does a genuine native dynamic import of the
// CDN URL. A plain `import(TF_URL)` gets rewritten into a parcel module helper
// that would try to resolve the URL as a local bundle.
let S=Function("u","return import(u)"),j=null,E=null;async function q(e){if(E)return;let r=await S("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm");r.env.allowRemoteModels=!0;let t="Nbardy/nff-clip-splat-weights",i=r=>{if("progress"===r.status&&r.total){let t=Math.round(r.progress??r.loaded/r.total*100),i=Math.round(t/100*16),a="█".repeat(i)+"░".repeat(16-i);e?.(`loading text encoder  [${a}] ${t}%  \xb7  ${(r.loaded/1e6).toFixed(1)}/${(r.total/1e6).toFixed(0)} MB`)}};// self-hosted alongside the vision weights
j=await r.AutoTokenizer.from_pretrained(t,{progress_callback:i}),E=await r.CLIPTextModelWithProjection.from_pretrained(t,{dtype:"fp16",device:"wasm",session_options:{graphOptimizationLevel:"basic"},progress_callback:i})}async function T(e){await q();let r=await j(e,{padding:"max_length",max_length:77,truncation:!0}),t=await E(r),i=t.text_embeds.data,a=new Float32Array(512);for(let e=0;e<512;e++)a[e]=i[e];return a}// ── Optimize loop ─────────────────────────────────────────────────────────────
let G=null,R=0,I=!1,O=!1;async function A(){if(G&&!I&&!O){I=!0;try{let e=await u.currentEmbedding(),r=(0,c.cosine)(e,G);p.cos=r,null===p.initialCos&&(p.initialCos=r),y()}finally{I=!1}}}function W(){p.running&&G&&(// 2 optimize steps/frame keeps the page responsive; each step is one submit
// (raster fwd → CLIP fwd+loss+bwd → raster bwd → Adam). At LEGIBLE_G≈12K
// splats a step is cheap, so the loop stays smooth.
u.step(),u.step(),R+=2,p.step=u.stepCount,R>=14&&(R=0,A())),function(){if(!_)return;let e=i.createCommandEncoder(),r=e.beginRenderPass({colorAttachments:[{view:a.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});r.setPipeline(n),r.setBindGroup(0,_),r.draw(3),r.end(),i.queue.submit([e.finish()])}(),requestAnimationFrame(W)}async function M(){if(!p.ready)return;let e=g.value.trim()||"a photo of a cat";h.disabled=!0,p.phase="encoding",p.running=!1,b("encoding prompt (first use downloads the text model — slow)…"),y();try{let r=await T(e);G=r,u.setPrompt(r);// Baseline cos on the CURRENT splats — this is the "initial" the gate checks
// the run rises above.
let t=await u.currentEmbedding();p.initialCos=(0,c.cosine)(t,r),p.cos=p.initialCos,R=0,b(""),p.phase="run",p.running=!0,y()}catch(e){x(`text encode failed: ${e?.message??e}`)}finally{h.disabled=!1}}async function D(){if(!p.ready)return;p.running=!1,G=null,p.cos=null,p.initialCos=null,p.phase="reset",k+=1;let e=u;u=await (0,c.SplatOptimizer).create(i,o,s,{seed:k}),e.destroy(),P(),await u.renderImage(),g.value="",p.step=0,b(""),y()}async function F(){if(!p.ready||O)return;O=!0;let e=p.running;p.running=!1,p.phase="nudge",m.disabled=!0,k+=1,y();try{if(await u.nudge({seed:k}),await u.renderImage(),R=0,G){let e=await u.currentEmbedding();p.cos=(0,c.cosine)(e,G)}p.phase=e&&G?"run":"idle",p.running=e&&!!G,b(""),y()}catch(e){x(`nudge failed: ${e?.message??e}`)}finally{O=!1,m.disabled=!1}}// ── Boot ─────────────────────────────────────────────────────────────────────
async function L(){let e;if(!navigator.gpu){x("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled."),h.disabled=!0,m.disabled=!0,v.disabled=!0;return}p.phase="adapter";let r=await navigator.gpu.requestAdapter();if(!r){x("no WebGPU adapter available.");return}i=await r.requestDevice(),i.addEventListener?.("uncapturederror",e=>{// eslint-disable-next-line no-console
console.error("[webgpu]",e.error?.message??e.error)}),a=f.getContext("webgpu"),d=navigator.gpu.getPreferredCanvasFormat(),a.configure({device:i,format:d,alphaMode:"opaque"}),p.phase="weights";// Prod (GitHub Pages) fetches the packed weights from the HF Hub: GitHub
// release assets send no CORS header, HF does — same host the text model
// loads from (upload via tools/splat/upload_weights.py). Local dev uses the
// fast same-origin static server (tools/splat/serve.mjs) so iteration
// doesn't re-pull 82 MB over the network.
let t=["localhost","127.0.0.1"].includes(location.hostname),n=t?"/models/mobileclip_s0/":"https://huggingface.co/Nbardy/nff-clip-splat-weights/resolve/main/",g=t?"":" from HF";$.textContent=`fetching CLIP plan${g}…`;let w=await fetch(n+"plan_train.json");if(!w.ok)return x(`plan_train.json fetch ${w.status} from ${n}`);o=await w.json();try{e=await (0,l.fetchArrayBufferWithProgress)(n+"weights_train.bin",e=>{$.textContent=(0,l.formatProgress)(`loading CLIP weights${g}`,e)})}catch(e){return x(`weights_train.bin fetch failed from ${n}: ${e?.message??e}`)}s=new Float32Array(e),p.phase="optimizer",$.textContent="building optimizer…",await C(),u=await (0,c.SplatOptimizer).create(i,o,s,{seed:k}),P(),await u.renderImage(),// Preload the text encoder at boot (with its own progress bar) so the first
// Optimize is instant instead of stalling on an 84 MB download.
p.phase="textmodel",await q(e=>{$.textContent=e}),p.ready=!0,p.phase="idle",h.disabled=!1,m.disabled=!1,v.disabled=!1,b(""),y(),requestAnimationFrame(W)}h.addEventListener("click",()=>void M()),m.addEventListener("click",()=>void F()),v.addEventListener("click",()=>void D()),g.addEventListener("keydown",e=>{"Enter"===e.key&&M()}),L().catch(e=>x(`boot failed: ${e?.message??e}`))},{"./splat/optimize":"nZSdJ","./splat/fetch_progress":"8QffC"}],nZSdJ:[function(e,r,t){/**
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
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"LEGIBLE_G",()=>n),i.export(t,"LEGIBLE_INIT",()=>d),i.export(t,"LEGIBLE_LRS",()=>c),i.export(t,"DEFAULT_NUDGE_AMOUNT",()=>l),i.export(t,"SplatOptimizer",()=>p),/** cos(a, b) — the metric the page shows and the test gates on. */i.export(t,"cosine",()=>f),// ---------------------------------------------------------------------------
// Deterministic random splat init (browser-safe: no node imports). Conventional
// 2D-splat start — spread over the canvas, small translucent Gaussians, mid
// colours the optimizer pushes around. SoA layout matches raster_wgsl.ts:
// [mean 2G][logScale 2G][theta G][colorRaw 3G][opacityRaw G], per-splat
// interleaved within each segment.
// ---------------------------------------------------------------------------
i.export(t,"randomSplats",()=>g),i.export(t,"nudgeSplats",()=>h);var a=e("./raster"),o=e("./raster_wgsl"),s=e("../clip/vision"),u=e("./adam_wgsl");let n=12e3,d={scale:9,scaleJitter:.35,opacityRaw:.4,colorSpread:1.2},c={mean:1.5,logScale:.06,theta:.08,color:.12,opacity:.06},l=.18;class p{static async create(e,r,t,i={}){let[o,u,d]=r.inputShape;if(3!==o||256!==u||256!==d)throw Error(`optimize: CLIP inputShape [${o},${u},${d}] != [3,256,256] — the raster→CLIP copy assumes matching NCHW dims`);let c=i.G??n,l=i.cap??2048,f=await (0,a.RasterEngine).create(e,{H:256,W:256,G:c,cap:l,bg:i.bg??[.5,.5,.5]}),h=await (0,s.VisionTrainer).create(e,r,t);return f.setParams(i.initParams??g(c,i.seed??1,i.init)),f.zeroAdamState(),new p(e,f,h,i)}constructor(e,r,t,i){this.side=256,this.step_=0,this.device=e,this.raster=r,this.trainer=t,this.lrs=i.lrs??c,this.hyper=i.hyper??u.DEFAULT_HYPER,this.init=i.init}/** Target text embedding (raw, un-normalized — the −cos loss normalizes it).
   *  Call on every prompt change; cheap (a 2 KB buffer write). */setPrompt(e){this.trainer.writeText(e)}/** One optimization step: forward → CLIP loss → backward → Adam, ONE submit. */step(){let e=this.device.createCommandEncoder();this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!0}),e.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackward(e),this.step_+=1,this.raster.recordAdam(e,this.step_,this.lrs,this.hyper),this.device.queue.submit([e.finish()])}get stepCount(){return this.step_}/** Partial re-randomization of the current splats. Unlike Reset, this keeps
   *  the optimizer, CLIP resources, prompt, step count, and Adam buffers alive. */async nudge(e={}){let r=this.raster.dims.G,t=await this.raster.readParams();h(t,r,e.seed??Date.now(),e.amount??l,e.init??this.init),this.raster.setParams(t)}/** Render the current splats without training; leaves the image on the GPU
   *  and returns it (NCHW planar [3][256][256]) for display / metrics. */async renderImage(){return this.raster.runForward(),this.raster.readImage()}/** CLIP embedding of the current splat image (forward-only). The page can use
   *  this to show live cosine similarity to the prompt; the test uses it to
   *  prove the loss decreases. */async currentEmbedding(){let e=this.device.createCommandEncoder();return this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!1}),this.device.queue.submit([e.finish()]),v(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){this.raster.destroy()}}function f(e,r){let t=0,i=0,a=0;for(let o=0;o<e.length;o++)t+=e[o]*r[o],i+=e[o]*e[o],a+=r[o]*r[o];return t/Math.sqrt(i*a||1)}function g(e,r=1,t={}){let i=t.scale??d.scale,a=t.scaleJitter??d.scaleJitter,s=t.opacityRaw??d.opacityRaw,u=t.colorSpread??d.colorSpread,n=r>>>0||1,c=()=>{let e=Math.imul((n=Math.imul(n,747796405)+2891336453>>>0)>>>(n>>>28)+4^n,277803737)>>>0;return(e=(e>>>22^e)>>>0)/4294967296},l=()=>{let e=0,r=0;for(;0===e;)e=c();for(;0===r;)r=c();return Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*r)},p=new Float32Array(e*o.PARAM_STRIDE),f=2*e,g=4*e,h=5*e,m=8*e,v=Math.log(i);for(let r=0;r<e;r++)p[0+2*r+0]=256*c(),p[0+2*r+1]=256*c(),p[f+2*r+0]=v+a*l(),p[f+2*r+1]=v+a*l(),p[g+r]=c()*Math.PI*2,p[h+3*r+0]=u*l(),p[h+3*r+1]=u*l(),p[h+3*r+2]=u*l(),p[m+r]=s;return p}function h(e,r,t=1,i=l,a={}){if(e.length!==r*o.PARAM_STRIDE)throw Error("nudgeSplats: wrong param length");let s=m(i,0,1);if(0===s)return e;let u=g(r,t,a),n=2*r,d=4*r,c=5*r,p=8*r,f=Math.log(.3),h=Math.log(64);for(let t=0;t<r;t++){var v,$,w,b,x,y,k,_;let r=0+2*t;e[r+0]=m((v=e[r+0],v+(u[r+0]-v)*s),0,256),e[r+1]=m(($=e[r+1],$+(u[r+1]-$)*s),0,256);let i=n+2*t;e[i+0]=m((w=e[i+0],w+(u[i+0]-w)*s),f,h),e[i+1]=m((b=e[i+1],b+(u[i+1]-b)*s),f,h);let a=d+t;e[a]=function(e,r,t){let i=2*Math.PI;return e+(((r-e+Math.PI)%i+i)%i-Math.PI)*t}(e[a],u[a],s);let o=c+3*t;e[o+0]=(x=e[o+0],x+(u[o+0]-x)*s),e[o+1]=(y=e[o+1],y+(u[o+1]-y)*s),e[o+2]=(k=e[o+2],k+(u[o+2]-k)*s);let l=p+t;e[l]=(_=e[l],_+(u[l]-_)*s)}return e}function m(e,r,t){return Math.max(r,Math.min(t,e))}// small readback helper (kept local — RasterEngine's is private, and the CLIP
// output buffer isn't one of RasterEngine's).
async function v(e,r,t){let i=e.createBuffer({size:4*t,usage:9/*COPY_DST*/}),a=e.createCommandEncoder();a.copyBufferToBuffer(r,0,i,0,4*t),e.queue.submit([a.finish()]),await i.mapAsync(1);let o=new Float32Array(i.getMappedRange().slice(0));return i.unmap(),i.destroy(),o}},{"./raster":"5D8U0","./raster_wgsl":"6IBEA","../clip/vision":"lNzsi","./adam_wgsl":"kfWkJ","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"5D8U0":[function(e,r,t){/**
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
 */var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"RasterEngine",()=>d);var a=e("./raster_wgsl"),o=e("./adam_wgsl");let s={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},u=e=>Math.ceil(e/256);async function n(e,r,t){e.pushErrorScope("validation");let i=e.createShaderModule({code:r}),a=e.createComputePipeline({layout:"auto",compute:{module:i,entryPoint:"main"}}),o=await e.popErrorScope();if(o)throw console.error(`--- WGSL that failed (${t}) ---
${r}`),Error(`raster pipeline validation (${t}): ${o.message}`);return a}class d{constructor(e,r){if(// per-group adam uniform buffers + bind groups (one per param group)
this.adamUni=[],this.adamBind=[],this.device=e,this.dims=(0,a.resolveDims)(r),this.dims.numTiles>65535)throw Error("raster: numTiles exceeds 1D dispatch limit")}static async create(e,r){let t=new d(e,r);return await t.build(r),t}storage(e,r=0){return this.device.createBuffer({size:4*e,usage:s.STORAGE|r})}async build(e){let r=this.dims,t=r.G*a.PARAM_STRIDE;// buffers
this.params=this.storage(t,s.COPY_SRC|s.COPY_DST),this.derived=this.storage(t),this.accGrad=this.storage(t,s.COPY_DST|s.COPY_SRC),this.gradRaw=this.storage(t,s.COPY_SRC),this.mBuf=this.storage(t,s.COPY_DST),this.vBuf=this.storage(t,s.COPY_DST),this.tileCounts=this.storage(r.numTiles,s.COPY_DST|s.COPY_SRC),this.binnedIds=this.storage(r.numTiles*r.cap,s.COPY_SRC),this.tileStop=this.storage(r.numTiles,s.COPY_SRC),this.image=this.storage(3*r.H*r.W,s.COPY_SRC),this.gradImage=this.storage(3*r.H*r.W,s.COPY_DST),// pipelines
this.prepPipe=await n(this.device,(0,a.prepShader)(e),"prep"),this.emitPipe=await n(this.device,(0,a.emitShader)(e),"emit"),this.fwdPipe=await n(this.device,(0,a.forwardShader)(e),"forward"),this.bwdPipe=await n(this.device,(0,a.backwardShader)(e),"backward"),this.chainPipe=await n(this.device,(0,a.chainShader)(e),"chain"),this.clearBinsPipe=await n(this.device,(0,a.clearShader)(r.numTiles),"clearBins"),this.clearGradsPipe=await n(this.device,(0,a.clearShader)(t),"clearGrads"),this.adamPipe=await n(this.device,(0,o.adamShader)(),"adam");let i=(e,r)=>this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:r.map((e,r)=>({binding:r,resource:{buffer:e}}))});for(let e of(this.prepBind=i(this.prepPipe,[this.params,this.derived]),this.emitBind=i(this.emitPipe,[this.derived,this.tileCounts,this.binnedIds]),this.fwdBind=i(this.fwdPipe,[this.tileCounts,this.binnedIds,this.derived,this.image,this.tileStop]),this.bwdBind=i(this.bwdPipe,[this.gradImage,this.tileCounts,this.binnedIds,this.tileStop,this.derived,this.accGrad]),this.chainBind=i(this.chainPipe,[this.accGrad,this.derived,this.params,this.gradRaw]),this.clearBinsBind=i(this.clearBinsPipe,[this.tileCounts]),this.clearGradsBind=i(this.clearGradsPipe,[this.accGrad]),(0,a.paramSegments)(r.G))){let e=this.device.createBuffer({size:o.ADAM_UNIFORM_BYTES,usage:s.UNIFORM|s.COPY_DST});this.adamUni.push(e),this.adamBind.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradRaw}},{binding:3,resource:{buffer:this.mBuf}},{binding:4,resource:{buffer:this.vBuf}}]}))}}// ---- uploads / readback ------------------------------------------------
setParams(e){if(e.length!==this.dims.G*a.PARAM_STRIDE)throw Error("setParams: wrong length");this.device.queue.writeBuffer(this.params,0,e)}setGradImage(e){if(e.length!==3*this.dims.H*this.dims.W)throw Error("setGradImage: wrong length");this.device.queue.writeBuffer(this.gradImage,0,e)}zeroAdamState(){let e=new Float32Array(this.dims.G*a.PARAM_STRIDE);this.device.queue.writeBuffer(this.mBuf,0,e),this.device.queue.writeBuffer(this.vBuf,0,e)}async readFloats(e,r){let t=this.device.createBuffer({size:4*r,usage:s.MAP_READ|s.COPY_DST}),i=this.device.createCommandEncoder();i.copyBufferToBuffer(e,0,t,0,4*r),this.device.queue.submit([i.finish()]),await t.mapAsync(1/* GPUMapMode.READ */);let a=new Float32Array(t.getMappedRange().slice(0));return t.unmap(),t.destroy(),a}readImage(){return this.readFloats(this.image,3*this.dims.H*this.dims.W)}readParams(){return this.readFloats(this.params,this.dims.G*a.PARAM_STRIDE)}readGradRaw(){return this.readFloats(this.gradRaw,this.dims.G*a.PARAM_STRIDE)}// ---- pass recording ----------------------------------------------------
/** prep -> clear bins -> emit -> forward. Populates `derived` and `image`. */recordForward(e){let r=this.dims,t=e.beginComputePass();t.setPipeline(this.prepPipe),t.setBindGroup(0,this.prepBind),t.dispatchWorkgroups(u(r.G)),t.setPipeline(this.clearBinsPipe),t.setBindGroup(0,this.clearBinsBind),t.dispatchWorkgroups(u(r.numTiles)),t.setPipeline(this.emitPipe),t.setBindGroup(0,this.emitBind),t.dispatchWorkgroups(u(r.G)),t.setPipeline(this.fwdPipe),t.setBindGroup(0,this.fwdBind),t.dispatchWorkgroups(r.numTiles),t.end()}/** clear grads -> backward -> chain. Requires a prior recordForward (uses its
   *  sorted binnedIds, tileStop and derived). Reads `gradImage`, writes gradRaw. */recordBackward(e){let r=this.dims,t=e.beginComputePass();t.setPipeline(this.clearGradsPipe),t.setBindGroup(0,this.clearGradsBind),t.dispatchWorkgroups(u(r.G*a.DERIVED_STRIDE)),t.setPipeline(this.bwdPipe),t.setBindGroup(0,this.bwdBind),t.dispatchWorkgroups(r.numTiles),t.setPipeline(this.chainPipe),t.setBindGroup(0,this.chainBind),t.dispatchWorkgroups(u(r.G)),t.end()}/** Adam over all 5 param groups; call after recordBackward (reads gradRaw). */recordAdam(e,r,t=o.DEFAULT_LRS,i=o.DEFAULT_HYPER){let s=(0,a.paramSegments)(this.dims.G),n={mean:t.mean,logScale:t.logScale,theta:t.theta,color:t.color,opacity:t.opacity},d=1-Math.pow(i.beta1,r),c=1-Math.pow(i.beta2,r);// write the 5 uniforms first (queued before the submit that runs `enc`)
s.forEach((e,r)=>{let t=new ArrayBuffer(o.ADAM_UNIFORM_BYTES),a=new Uint32Array(t),s=new Float32Array(t);a[0]=e.offset,a[1]=e.length,s[2]=n[e.name],s[3]=i.beta1,s[4]=i.beta2,s[5]=i.eps,s[6]=d,s[7]=c,this.device.queue.writeBuffer(this.adamUni[r],0,t)});let l=e.beginComputePass();l.setPipeline(this.adamPipe),s.forEach((e,r)=>{l.setBindGroup(0,this.adamBind[r]),l.dispatchWorkgroups(u(e.length))}),l.end()}// ---- self-submitting convenience wrappers ------------------------------
runForward(){let e=this.device.createCommandEncoder();this.recordForward(e),this.device.queue.submit([e.finish()])}runBackward(){let e=this.device.createCommandEncoder();this.recordBackward(e),this.device.queue.submit([e.finish()])}runAdam(e,r,t){let i=this.device.createCommandEncoder();this.recordAdam(i,e,r,t),this.device.queue.submit([i.finish()])}destroy(){for(let e of[this.params,this.derived,this.accGrad,this.gradRaw,this.mBuf,this.vBuf,this.tileCounts,this.binnedIds,this.tileStop,this.image,this.gradImage,...this.adamUni])try{e.destroy()}catch(e){}}}},{"./raster_wgsl":"6IBEA","./adam_wgsl":"kfWkJ","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"6IBEA":[function(e,r,t){/**
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
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"TILE",()=>a),i.export(t,"ALPHA_THRESHOLD",()=>o),i.export(t,"MAX_ALPHA",()=>s),i.export(t,"TRANSMITTANCE_CUTOFF",()=>u),i.export(t,"EPS",()=>n),i.export(t,"SCALE_MIN",()=>d),i.export(t,"SCALE_MAX",()=>c),i.export(t,"DERIVED_STRIDE",()=>l),i.export(t,"PARAM_STRIDE",()=>p),i.export(t,"resolveDims",()=>m),// ---------------------------------------------------------------------------
// 1) prep — thread/splat: raw params -> derived (mean, conic, color, opacity).
//    The single place the reparameterization forward is computed.
// ---------------------------------------------------------------------------
i.export(t,"prepShader",()=>$),// ---------------------------------------------------------------------------
// 2) emit — thread/splat: fixedbin binning (v11 style, no prefix sum, no CPU
//    readback). Atomic cursor per tile into constant-stride bins tile*cap.
//    Merges count+emit: tileCounts is the cursor (cleared each step); a splat
//    whose slot >= cap is dropped (graceful overflow). The forward re-sorts by
//    index so the emit order is irrelevant to the result (deterministic).
// ---------------------------------------------------------------------------
i.export(t,"emitShader",()=>w),// ---------------------------------------------------------------------------
// 3) forward — 1 workgroup(256)/tile, one thread per pixel. Stage tile ids in
//    shared, bitonic-sort ASCENDING (recovers painter order == splat index
//    order; there is no depth), write the sorted ids back so the backward can
//    skip re-sorting, front-to-back composite with early-out, save
//    tileStop = max visible-prefix length (bounds the backward replay).
// ---------------------------------------------------------------------------
i.export(t,"forwardShader",()=>b),// ---------------------------------------------------------------------------
// 4) backward — 1 workgroup(256)/tile, one thread per pixel. Replays the
//    visible prefix (bounded by tileStop) to recover T_final and end_i, then
//    walks BACK-TO-FRONT reconstructing per-splat grads with T_prev = T_cur/
//    (1-alpha). Accumulates DERIVED-space grads (mean, conic, color, opacity)
//    into accGrad via fixed-point atomicAdd<i32> — byte-for-byte the Metal
//    reference recurrence. NO barriers in the per-pixel loop, so the uniformity
//    rule is satisfied trivially (each pixel's end_i gates only its own loop).
// ---------------------------------------------------------------------------
i.export(t,"backwardShader",()=>x),// ---------------------------------------------------------------------------
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
i.export(t,"chainShader",()=>y),// ---------------------------------------------------------------------------
// 6) clear — thread/element: zero a storage buffer viewed as array<u32>
//    (works for the i32 accGrad and the u32 tileCounts; 0 bits == 0 either way).
// ---------------------------------------------------------------------------
i.export(t,"clearShader",()=>k),/** Segment offsets for the Adam driver (matches seg()). */i.export(t,"paramSegments",()=>_);let a=16,o=1/255,s=.99,u=1e-4,n=1e-8,d=.3,c=64,l=9,p=9;// 16x16 tile == 256 pixels == 256 threads/workgroup
function f(e,r){if(!e)throw Error(`raster_wgsl: ${r}`)}/** WGSL f32 literal — always has a '.' or exponent so it is not parsed as int. */function g(e){f(Number.isFinite(e),`non-finite literal ${e}`);let r=e.toString();return/[.eE]/.test(r)||(r+=".0"),r}let h=e=>`${e>>>0}u`;function m(e){f(e.H>0&&e.W>0&&e.G>0,"H,W,G must be positive"),f(e.H%a==0&&e.W%a==0,`H,W must be multiples of ${a}`),f((e.cap&e.cap-1)==0&&e.cap>0,"cap must be a power of two"),f(4*e.cap<=16384,`cap*4 (${4*e.cap}B) exceeds 16KB workgroup storage`);let r=e.W/a,t=e.H/a;return{H:e.H,W:e.W,G:e.G,cap:e.cap,tilesX:r,tilesY:t,numTiles:r*t,bg:e.bg??[.5,.5,.5],gradScale:e.gradScale??65536}}// ---------------------------------------------------------------------------
// Shared WGSL fragments (inlined per kernel — each module stays standalone)
// ---------------------------------------------------------------------------
/** Segment base offsets into the SoA params/gradRaw/m/v buffers. */function v(e){return{mean:0,logScale:2*e.G,theta:4*e.G,colorRaw:5*e.G,opacityRaw:8*e.G}}function $(e){let r=m(e),t=v(r);return/* wgsl */`
fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@group(0) @binding(0) var<storage, read>       params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${h(r.G)}) { return; }
  let mx  = params[${h(t.mean)} + g * 2u + 0u];
  let my  = params[${h(t.mean)} + g * 2u + 1u];
  let lsx = params[${h(t.logScale)} + g * 2u + 0u];
  let lsy = params[${h(t.logScale)} + g * 2u + 1u];
  let th  = params[${h(t.theta)} + g];
  let cr0 = params[${h(t.colorRaw)} + g * 3u + 0u];
  let cr1 = params[${h(t.colorRaw)} + g * 3u + 1u];
  let cr2 = params[${h(t.colorRaw)} + g * 3u + 2u];
  let opr = params[${h(t.opacityRaw)} + g];

  let sx = clamp(exp(lsx), ${g(d)}, ${g(c)});
  let sy = clamp(exp(lsy), ${g(d)}, ${g(c)});
  let ix = 1.0 / (sx * sx);
  let iy = 1.0 / (sy * sy);
  let cs = cos(th);
  let sn = sin(th);

  let base = g * ${h(l)};
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
`}function w(e){let r=m(e);return/* wgsl */`
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
  if (g >= ${h(r.G)}) { return; }
  let base = g * ${h(l)};
  let op = derived[base + 8u];
  if (op <= ${g(o)}) { return; }
  let ratio = max(${g(o)} / max(op, ${g(n)}), ${g(n)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let mx = derived[base + 0u]; let my = derived[base + 1u];
  let a  = derived[base + 2u]; let b  = derived[base + 3u]; let c = derived[base + 4u];
  let det = max(a * c - b * b, ${g(n)});
  let hx = sqrt(max(tau * c / det, 0.0));
  let hy = sqrt(max(tau * a / det, 0.0));
  let x0 = max(0, i32(floor(mx - hx - 0.5)));
  let x1 = min(${r.W-1}, i32(ceil(mx + hx - 0.5)));
  let y0 = max(0, i32(floor(my - hy - 0.5)));
  let y1 = min(${r.H-1}, i32(ceil(my + hy - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tx0 = x0 / ${a}; let tx1 = x1 / ${a};
  let ty0 = y0 / ${a}; let ty1 = y1 / ${a};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    let ry0 = f32(ty * ${a}) + 0.5;
    let ry1 = min(f32(${r.H-1}) + 0.5, f32((ty + 1) * ${a} - 1) + 0.5);
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let rx0 = f32(tx * ${a}) + 0.5;
      let rx1 = min(f32(${r.W-1}) + 0.5, f32((tx + 1) * ${a} - 1) + 0.5);
      if (ellipse_hit(mx, my, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        let tile = u32(ty * ${r.tilesX} + tx);
        let slot = atomicAdd(&tileCounts[tile], 1u);
        if (slot < ${h(r.cap)}) { binnedIds[tile * ${h(r.cap)} + slot] = g; }
      }
    }
  }
}
`}function b(e){let r=m(e),t=r.H*r.W;return/* wgsl */`
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;  // NCHW planar
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;

var<workgroup> sh_ids     : array<u32, ${r.cap}>;
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
  if (tileId >= ${h(r.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${h(r.cap)});
  let start = tileId * ${h(r.cap)};
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

  let tileX = tileId % ${h(r.tilesX)};
  let tileY = tileId / ${h(r.tilesX)};
  let x = tileX * ${a}u + (tid % ${a}u);
  let y = tileY * ${a}u + (tid / ${a}u);
  var localStop = 0u;
  if (x < ${h(r.W)} && y < ${h(r.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var accR = 0.0; var accG = 0.0; var accB = 0.0; var T = 1.0;
    for (var i = 0u; i < count; i = i + 1u) {
      let gg = sh_ids[i];
      let b3 = gg * ${h(l)};
      let dx = pxc - derived[b3 + 0u];
      let dy = pyc - derived[b3 + 1u];
      let a  = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
      let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[b3 + 8u] * exp(power);
      let alpha = min(${g(s)}, raw);
      if (alpha < ${g(o)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b3 + 5u];
      accG = accG + w * derived[b3 + 6u];
      accB = accB + w * derived[b3 + 7u];
      T = T * (1.0 - alpha);
      if (T < ${g(u)}) { break; }
    }
    let pix = y * ${h(r.W)} + x;
    image[0u * ${h(t)} + pix] = accR + T * ${g(r.bg[0])};
    image[1u * ${h(t)} + pix] = accG + T * ${g(r.bg[1])};
    image[2u * ${h(t)} + pix] = accB + T * ${g(r.bg[2])};
  }
  atomicMax(&sh_maxstop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&sh_maxstop); }
}
`}function x(e){let r=m(e),t=r.H*r.W,i=g(r.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;  // NCHW planar
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;

var<workgroup> sh_ids : array<u32, ${r.cap}>;

fn fixadd(base : u32, slot : u32, v : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(v * ${i}), -2.14e9, 2.14e9)));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${h(r.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${h(r.cap)});
  let stopc = min(count, tileStop[tileId]);
  let start = tileId * ${h(r.cap)};
  for (var i = tid; i < stopc; i = i + 256u) { sh_ids[i] = binnedIds[start + i]; }
  workgroupBarrier();  // only barrier; everything below is per-pixel (uniformity safe)

  let tileX = tileId % ${h(r.tilesX)};
  let tileY = tileId / ${h(r.tilesX)};
  let x = tileX * ${a}u + (tid % ${a}u);
  let y = tileY * ${a}u + (tid / ${a}u);
  if (x >= ${h(r.W)} || y >= ${h(r.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${h(r.W)} + x;
  let goR = gradImage[0u * ${h(t)} + pix];
  let goG = gradImage[1u * ${h(t)} + pix];
  let goB = gradImage[2u * ${h(t)} + pix];

  // phase A: replay to recover T_final and the stop index end_i
  var T = 1.0;
  var endi = stopc;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b3 = gg * ${h(l)};
    let dx = pxc - derived[b3 + 0u];
    let dy = pyc - derived[b3 + 1u];
    let a = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${g(s)}, derived[b3 + 8u] * exp(power));
    if (alpha < ${g(o)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${g(u)}) { endi = i + 1u; break; }
  }

  // phase B: back-to-front recurrence
  var Tcur = T;
  var gT = goR * ${g(r.bg[0])} + goG * ${g(r.bg[1])} + goB * ${g(r.bg[2])};
  for (var ii = i32(endi) - 1; ii >= 0; ii = ii - 1) {
    let gg = sh_ids[u32(ii)];
    let b3 = gg * ${h(l)};
    let dx = pxc - derived[b3 + 0u];
    let dy = pyc - derived[b3 + 1u];
    let a = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let op = derived[b3 + 8u];
    let raw = op * exp(power);
    let alpha = min(${g(s)}, raw);
    if (alpha < ${g(o)}) { continue; }
    let denom = max(1.0 - alpha, ${g(n)});
    let Tprev = Tcur / denom;
    let cR = derived[b3 + 5u]; let cG = derived[b3 + 6u]; let cB = derived[b3 + 7u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b3, 5u, goR * Tprev * alpha);
    fixadd(b3, 6u, goG * Tprev * alpha);
    fixadd(b3, 7u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${g(s)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-(a * dx + b * dy));
    let gdy = gPower * (-(b * dx + c * dy));
    fixadd(b3, 2u, gPower * (-0.5) * dx * dx);   // g_a
    fixadd(b3, 3u, gPower * (-1.0) * dx * dy);   // g_b
    fixadd(b3, 4u, gPower * (-0.5) * dy * dy);   // g_c
    fixadd(b3, 0u, -gdx);                        // g_mean.x
    fixadd(b3, 1u, -gdy);                        // g_mean.y
    fixadd(b3, 8u, gRaw * (raw / max(op, ${g(n)})));  // g_opacity

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`}function y(e){let r=m(e),t=v(r),i=g(1/r.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       accGrad : array<i32>;   // fixed-point
@group(0) @binding(1) var<storage, read>       derived : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${h(r.G)}) { return; }
  let b3 = g * ${h(l)};
  let inv = ${i};
  let gmx = f32(accGrad[b3 + 0u]) * inv;
  let gmy = f32(accGrad[b3 + 1u]) * inv;
  let gA  = f32(accGrad[b3 + 2u]) * inv;
  let gB  = f32(accGrad[b3 + 3u]) * inv;
  let gC  = f32(accGrad[b3 + 4u]) * inv;
  let gc0 = f32(accGrad[b3 + 5u]) * inv;
  let gc1 = f32(accGrad[b3 + 6u]) * inv;
  let gc2 = f32(accGrad[b3 + 7u]) * inv;
  let gop = f32(accGrad[b3 + 8u]) * inv;

  let lsx = params[${h(t.logScale)} + g * 2u + 0u];
  let lsy = params[${h(t.logScale)} + g * 2u + 1u];
  let th  = params[${h(t.theta)} + g];
  let ex = exp(lsx); let ey = exp(lsy);
  let sx = clamp(ex, ${g(d)}, ${g(c)});
  let sy = clamp(ey, ${g(d)}, ${g(c)});
  let gateX = select(0.0, 1.0, ex > ${g(d)} && ex < ${g(c)});
  let gateY = select(0.0, 1.0, ey > ${g(d)} && ey < ${g(c)});
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

  gradRaw[${h(t.mean)} + g * 2u + 0u] = gmx;
  gradRaw[${h(t.mean)} + g * 2u + 1u] = gmy;
  gradRaw[${h(t.logScale)} + g * 2u + 0u] = glsx;
  gradRaw[${h(t.logScale)} + g * 2u + 1u] = glsy;
  gradRaw[${h(t.theta)} + g] = gth;
  gradRaw[${h(t.colorRaw)} + g * 3u + 0u] = gc0 * col0 * (1.0 - col0);
  gradRaw[${h(t.colorRaw)} + g * 3u + 1u] = gc1 * col1 * (1.0 - col1);
  gradRaw[${h(t.colorRaw)} + g * 3u + 2u] = gc2 * col2 * (1.0 - col2);
  gradRaw[${h(t.opacityRaw)} + g] = gop * opv * (1.0 - opv);
}
`}function k(e){return/* wgsl */`
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= ${h(e)}) { return; }
  buf[gid.x] = 0u;
}
`}function _(e){return[{name:"mean",offset:0,length:2*e},{name:"logScale",offset:2*e,length:2*e},{name:"theta",offset:4*e,length:e},{name:"color",offset:5*e,length:3*e},{name:"opacity",offset:8*e,length:e}]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],k3151:[function(e,r,t){t.interopDefault=function(e){return e&&e.__esModule?e:{default:e}},t.defineInteropFlag=function(e){Object.defineProperty(e,"__esModule",{value:!0})},t.exportAll=function(e,r){return Object.keys(e).forEach(function(t){"default"===t||"__esModule"===t||r.hasOwnProperty(t)||Object.defineProperty(r,t,{enumerable:!0,get:function(){return e[t]}})}),r},t.export=function(e,r,t){Object.defineProperty(e,r,{enumerable:!0,get:t})}},{}],kfWkJ:[function(e,r,t){/**
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
 *//** Adam uniform: 8 x 4 bytes = 32 bytes (std140-safe: all scalars). */var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"ADAM_UNIFORM_BYTES",()=>a),i.export(t,"adamShader",()=>o),i.export(t,"DEFAULT_LRS",()=>s),i.export(t,"DEFAULT_HYPER",()=>u);let a=32;function o(){return/* wgsl */`
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
`}let s={mean:.01,logScale:.005,theta:.005,color:.005,opacity:.005},u={beta1:.9,beta2:.999,eps:1e-8}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],lNzsi:[function(e,r,t){/**
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
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"VisionEncoder",()=>u),/**
 * VisionTrainer — the runtime for the fused backward. Owns activation AND grad
 * slot buffers (plan.slots is 2×: [0,nAct) activations, [nAct,2nAct) grads),
 * the packed weights (with transposed pointwise copies), and a per-prompt text
 * buffer. Encodes forward + loss head + backward as ONE compute pass; the
 * input gradient (dL/dpixels) lands in slot `plan.inputGradSlot`.
 *
 * Weights FROZEN — no dW, no optimizer here (spec non-goals).
 */i.export(t,"VisionTrainer",()=>c);var a=e("./vision_wgsl"),o=e("./vision_bwd_wgsl");let s={COPY_SRC:4,COPY_DST:8,STORAGE:128};class u{/**
   * Async factory (pipeline validation is async). `weights` must be the
   * packed blob from compile_plan.py — its length is checked against the
   * plan loudly; a mismatched pair cannot run.
   */static async create(e,r,t){if(t.length!==r.weightsFloats)throw Error(`vision: weights blob ${t.length} floats != plan ${r.weightsFloats}`);return new u(e,r,t,await n(e,r))}constructor(e,r,t,i){this.dispatches=[],this.device=e,this.plan=r,this.weightsBuffer=e.createBuffer({size:t.byteLength,usage:s.STORAGE|s.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,t),this.slotBuffers=r.slots.map((r,t)=>e.createBuffer({label:`clip-slot-${t}`,size:4*r,usage:s.STORAGE|s.COPY_DST|s.COPY_SRC})),this.dispatches=i.map(({spec:e,pipeline:r})=>({pipeline:r,workgroups:e.workgroups,label:e.label,bind:this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:e.buffers.map((e,r)=>({binding:r,resource:{// Forward-only: weights + activation slots. A 'text' ref only
        // appears in the backward loss head, which lives in VisionTrainer;
        // seeing one here is a wiring bug, so fail loudly (no silent path).
        buffer:"weights"===e.kind?this.weightsBuffer:"slot"===e.kind?this.slotBuffers[e.slot]:(()=>{throw Error("vision: forward encoder received a 'text' binding (loss head belongs to VisionTrainer)")})()}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}/** Upload an NCHW [3,256,256] image in [0,1]. */writeInput(e){let[r,t,i]=this.plan.inputShape;if(e.length!==r*t*i)throw Error(`vision: input ${e.length} != ${r*t*i}`);this.device.queue.writeBuffer(this.inputBuffer,0,e)}/**
   * Encode the whole forward (optionally only the first `stepLimit` plan
   * steps — the per-step verification hook) into one compute pass.
   */encode(e,r=this.dispatches.length){// One compute pass for the whole forward — WebGPU guarantees storage
// write visibility BETWEEN dispatches in a pass (each dispatch is its own
// usage scope), verified on Dawn/Metal by the per-step suite.
let t=e.beginComputePass();for(let e=0;e<r;e++){let r=this.dispatches[e];t.setPipeline(r.pipeline),t.setBindGroup(0,r.bind),t.dispatchWorkgroups(...r.workgroups)}t.end()}/** Submit one full forward. */run(){let e=this.device.createCommandEncoder();this.encode(e),this.device.queue.submit([e.finish()])}/** Dispatch count per plan step (test bisection needs the mapping).
   *  Every step kind is exactly one dispatch since attention became
   *  pointwise-conv + attn_core + pointwise-conv plan steps. */stepDispatchCounts(){return this.plan.steps.map(()=>1)}}async function n(e,r){return d(e,(0,a.planDispatches)(r))}async function d(e,r){let t=[];for(let i of r){e.pushErrorScope("validation");let r=e.createShaderModule({code:i.code}),a=e.createComputePipeline({layout:"auto",compute:{module:r,entryPoint:"main"}}),o=await e.popErrorScope();if(o)throw Error(`vision: pipeline '${i.label}' invalid: ${o.message}
${i.code}`);t.push({spec:i,pipeline:a})}return t}class c{static async create(e,r,t){if(t.length!==r.weightsFloats)throw Error(`vision: weights blob ${t.length} floats != plan ${r.weightsFloats}`);let i=(0,a.planDispatches)(r),s=(0,o.planBwdDispatches)(r),u=await d(e,[...i,...s]);return new c(e,r,t,u,i.length)}constructor(e,r,t,i,a){this.dispatches=[],this.device=e,this.plan=r,this.fwdCount=a,this.weightsBuffer=e.createBuffer({size:t.byteLength,usage:s.STORAGE|s.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,t),this.textBuffer=e.createBuffer({size:4*r.textDim,usage:s.STORAGE|s.COPY_DST}),this.slotBuffers=r.slots.map((r,t)=>e.createBuffer({label:`clip-tslot-${t}`,size:4*r,usage:s.STORAGE|s.COPY_DST|s.COPY_SRC}));let o=e=>"weights"===e.kind?this.weightsBuffer:"text"===e.kind?this.textBuffer:this.slotBuffers[e.slot];this.dispatches=i.map(({spec:e,pipeline:r})=>({pipeline:r,workgroups:e.workgroups,label:e.label,bind:this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:e.buffers.map((e,r)=>({binding:r,resource:{buffer:o(e)}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}get inputGradBuffer(){return this.slotBuffers[this.plan.inputGradSlot]}writeInput(e){let[r,t,i]=this.plan.inputShape;if(e.length!==r*t*i)throw Error(`vision: input ${e.length} != ${r*t*i}`);this.device.queue.writeBuffer(this.inputBuffer,0,e)}/** Target text embedding for the −cos loss (uploaded per prompt change). */writeText(e){if(e.length!==this.plan.textDim)throw Error(`vision: text ${e.length} != ${this.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffer,0,e)}/** Encode forward, then (optionally) the loss head + backward, one pass. */encode(e,r={}){let t=!1===r.backward?this.fwdCount:this.dispatches.length,i=e.beginComputePass();for(let e=0;e<t;e++){let r=this.dispatches[e];i.setPipeline(r.pipeline),i.setBindGroup(0,r.bind),i.dispatchWorkgroups(...r.workgroups)}i.end()}/** Encode only the verified forward pass, preserving activations for backward. */encodeForward(e){let r=e.beginComputePass();for(let e=0;e<this.fwdCount;e++){let t=this.dispatches[e];r.setPipeline(t.pipeline),r.setBindGroup(0,t.bind),r.dispatchWorkgroups(...t.workgroups)}r.end()}/** Encode only the loss head + backward. Requires a prior forward. */encodeBackward(e){let r=e.beginComputePass();for(let e=this.fwdCount;e<this.dispatches.length;e++){let t=this.dispatches[e];r.setPipeline(t.pipeline),r.setBindGroup(0,t.bind),r.dispatchWorkgroups(...t.workgroups)}r.end()}/** Forward + backward. dL/dpixels is left in `inputGradBuffer`. */run(e={}){let r=this.device.createCommandEncoder();this.encode(r,e),this.device.queue.submit([r.finish()])}}},{"./vision_wgsl":"oFDUc","./vision_bwd_wgsl":"2Oqph","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],oFDUc:[function(e,r,t){/**
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
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"weightsDecl",()=>a),i.export(t,"GELU",()=>o),i.export(t,"assertStep",()=>s),i.export(t,"PW_TILE_DECLS",()=>u),/** The shared tiled-matmul body: out[co][p] = Σ_ci src[ci][p]·W[ci*cout+co].
 *  Produces acc0..acc3 (vec4 = 4 pixels × 4 couts) then stores. `init` seeds
 *  each acc (bias for fwd, 0 for bwd); `store` maps acc{j} → the value written
 *  to dst[(co+j)*P4+p4] (gelu/residual epilogue for fwd, add-into for bwd
 *  accumulate). Requires src(binding 1, array<vec4f> [Cin][P4]), dst(binding 2),
 *  weights(binding 0), and PW_TILE_DECLS in scope. */i.export(t,"pointwiseTiledMain",()=>n),/** Assert a pointwise-shaped step satisfies the tile constraints. Shared so the
 *  backward reuses the SAME loud guard (a violating shape needs a handler). */i.export(t,"assertPointwiseTiles",()=>d),i.export(t,"stepDispatches",()=>c),/** All dispatches for a full forward pass, in execution order. */i.export(t,"planDispatches",()=>l);let a=e=>`@group(0) @binding(${e}) var<storage, read> weights : array<vec4f>;
fn W(i : u32) -> f32 { return weights[i >> 2u][i & 3u]; }
fn W4(i : u32) -> vec4f { return weights[i]; }`,o=/* wgsl */`
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
`;function s(e,r){if(!e)throw Error(`vision_wgsl: ${r}`)}let u=/* wgsl */`
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 256>;`;function n(e){return/* wgsl */`
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
      xS[t] = src[(ci0 + ci) * ${e.P4}u + p4base + lane];
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
  dst[co * ${e.P4}u + p4] = ${e.store(0)};
  dst[(co + 1u) * ${e.P4}u + p4] = ${e.store(1)};
  dst[(co + 2u) * ${e.P4}u + p4] = ${e.store(2)};
  dst[(co + 3u) * ${e.P4}u + p4] = ${e.store(3)};
}`}function d(e,r,t,i,a){s(i%32==0&&t%32==0&&r%32==0,`${e}: tiled pointwise needs P%32==0 && cout%32==0 && cin%32==0 (got P=${i} cin=${r} cout=${t})`),s(a%4==0,`${e}: wOff not 16B-aligned`)}function c(e){switch(e.kind){case"conv":return(// ---------------------------------------------------------------------------
// Thin dispatchers — step kind → dispatch list; conv variant → emitter.
// ---------------------------------------------------------------------------
function(e){switch(e.variant){case"pointwise":return[function(e){let r=e.outH*e.outW;d(e.name,e.cin,e.cout,r,e.wOff);let t=r/4,i=null!==e.residual;s(null!==e.layerScaleOff===i,`${e.name}: layerScale without residual`);let c=[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}];i&&c.push({kind:"slot",slot:e.residual});let l=/* wgsl */`
${a(0)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${i?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${o}
${u}
${n({cin:e.cin,cout:e.cout,P4:t,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:r=>{let a="gelu"===e.act?`gelu4(acc${r})`:`acc${r}`;return i?`res[(co + ${r}u) * ${t}u + p4] + vec4f(W(${e.layerScaleOff}u + co + ${r}u)) * ${a}`:a}})}`;return{label:`pw ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:l,workgroups:[t/8,e.cout/32,1],buffers:c}}(e)];case"depthwise":case"general":return[// ---------------------------------------------------------------------------
// conv:depthwise — k∈{3,7}, groups=C. Thread = one output pixel of one channel.
// ---------------------------------------------------------------------------
function(e){s(null===e.residual&&null===e.layerScaleOff,`${e.name}: spatial conv never carries residual in this plan`),s(e.outW%4==0,`${e.name}: spatial tiling needs outW%4==0`);let r=e.outH*e.outW,t=r/4,i=e.k,u=e.stride,n=e.pad,d=3*u+i,c=e.cin/e.groups,l=e.cout/e.groups,p=c*i*i;s(Number.isInteger(c)&&Number.isInteger(l),`${e.name}: bad groups`),s(p<=64,`${e.name}: weight tile ${p} exceeds one staging round`);let f=r=>"gelu"===e.act?`gelu1(${r})`:r,g=[];for(let r=0;r<c;r++){g.push(`    { let base = (ci0 + ${r}u) * ${e.h*e.w}u;`);for(let t=0;t<i;t++){g.push(`      { let rowBase = base + u32(iy0 + ${t}) * ${e.w}u + u32(ix0);`);for(let e=0;e<d;e++)g.push(`        let r${e} = src[rowBase + ${e}u];`);for(let e=0;e<i;e++)g.push(`        acc = fma(vec4f(r${e}, r${u+e}, r${2*u+e}, r${3*u+e}), vec4f(wk[${r*i*i+t*i+e}u]), acc);`);g.push("      }")}g.push("    }")}// One workgroup = one output channel: its cpg·k·k weights are staged once
// in shared memory (depthwise is just cpg=1). Each thread produces 4
// horizontal pixels, loading each input row segment once (NIN loads for
// 4·K taps ≈ 2.8× fewer for k=7). Interior tiles (the vast majority) take
// a single-branch unchecked path.
let h=/* wgsl */`
${a(0)}
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
  if (q >= ${t}u) { return; }
  let oy = i32(q / ${e.outW/4}u);
  let ox0 = i32(q % ${e.outW/4}u) * 4;
  let ci0 = (co / ${l}u) * ${c}u;   // first input channel of co's group
  let iy0 = oy * ${u} - ${n};
  let ix0 = ox0 * ${u} - ${n};
  var acc = vec4f(W(${e.bOff}u + co));
  if (iy0 >= 0 && iy0 + ${i} <= ${e.h} && ix0 >= 0 && ix0 + ${d} <= ${e.w}) {
    // interior: every tap in bounds, unchecked unrolled register loads
${g.join("\n")}
  } else {
    // border: per-tap bounds checks (zero padding)
    for (var c = 0u; c < ${c}u; c = c + 1u) {
      let base = (ci0 + c) * ${e.h*e.w}u;
      for (var ky = 0; ky < ${i}; ky = ky + 1) {
        let iy = iy0 + ky;
        if (iy < 0 || iy >= ${e.h}) { continue; }
        let rowBase = base + u32(iy) * ${e.w}u;
        for (var kx = 0; kx < ${i}; kx = kx + 1) {
          let wv = wk[c * ${i*i}u + u32(ky * ${i} + kx)];
          var xv = vec4f(0.0);
          for (var j = 0; j < 4; j = j + 1) {
            let ix = ix0 + j * ${u} + kx;
            if (ix >= 0 && ix < ${e.w}) { xv[j] = src[rowBase + u32(ix)]; }
          }
          acc = fma(xv, vec4f(wv), acc);
        }
      }
    }
  }
  let out = co * ${r}u + u32(oy) * ${e.outW}u + u32(ox0);
  dst[out] = ${f("acc.x")};
  dst[out + 1u] = ${f("acc.y")};
  dst[out + 2u] = ${f("acc.z")};
  dst[out + 3u] = ${f("acc.w")};
}`;return{label:`conv${e.k} ${e.cin}->${e.cout} g${e.groups} @${e.outH}x${e.outW}`,code:h,workgroups:[Math.ceil(t/64),e.cout,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)];// depthwise = spatial, cpg=1
}}(e));case"se":return[function(e){var r;let t=e.h*e.w;s(e.c<=2048&&e.cmid<=512,`${e.name}: SE dims exceed shared-memory plan`);let i=/* wgsl */`
${a(0)}
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
    for (var p = 0u; p < ${t}u; p = p + 1u) { sum = sum + src[c * ${t}u + p]; }
    gap[c] = sum / ${t}.0;
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
  for (var i = li; i < ${e.c*t}u; i = i + 256u) {
    dst[i] = ${(r=`src[i] * scl[i / ${t}u]`,"gelu"===e.act?`gelu1(${r})`:r)};
  }
}`;return{label:`se c${e.c} mid${e.cmid} @${e.h}x${e.w}`,code:i,workgroups:[1,1,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)];case"attn_core":return[// ---------------------------------------------------------------------------
// attn_core — per-head softmax(QKᵀ)V, K then V staged through shared memory.
// The qkv and proj matmuls around it are ordinary pointwise ConvSteps (κ
// folds BN + q-scale into the qkv weights), so this kernel is pure attention:
// one workgroup per head, one thread per query token, score row in registers.
// ---------------------------------------------------------------------------
function(e){let{nTok:r,hd:t,heads:i,c:a}=e,o=t/4,u=r*t/4;s(a===i*t,`${e.name}: c != heads*hd`),s(r<=256&&16*u<=16384,`${e.name}: K/V won't fit shared memory`);// channel-planar addressing: channel o at token n sits at o*nTok + n
let n=/* wgsl */`
@group(0) @binding(0) var<storage, read> qkv : array<f32>;
@group(0) @binding(1) var<storage, read_write> attnOut : array<f32>;
var<workgroup> kv : array<vec4f, ${u}>;   // K, then reused for V; [j][d4]
@compute @workgroup_size(${r})
fn main(@builtin(local_invocation_index) i : u32,
        @builtin(workgroup_id) wid : vec3u) {
  let head = wid.x;
  let qCh = head * ${t}u;                      // q channels [qCh, qCh+hd)
  let kCh = ${a}u + head * ${t}u;
  let vCh = ${2*a}u + head * ${t}u;
  // gather this thread's query row into registers (one-time strided reads)
  var q : array<vec4f, ${o}>;
  for (var d4 = 0u; d4 < ${o}u; d4 = d4 + 1u) {
    q[d4] = vec4f(
      qkv[(qCh + d4 * 4u) * ${r}u + i],
      qkv[(qCh + d4 * 4u + 1u) * ${r}u + i],
      qkv[(qCh + d4 * 4u + 2u) * ${r}u + i],
      qkv[(qCh + d4 * 4u + 3u) * ${r}u + i]);
  }
  for (var t = i; t < ${u}u; t = t + ${r}u) {
    let j = t / ${o}u;
    let d = (t % ${o}u) * 4u;
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
    for (var d4 = 0u; d4 < ${o}u; d4 = d4 + 1u) {
      sv = fma(q[d4], kv[j * ${o}u + d4], sv);
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
  for (var t = i; t < ${u}u; t = t + ${r}u) {
    let j = t / ${o}u;
    let d = (t % ${o}u) * 4u;
    kv[t] = vec4f(
      qkv[(vCh + d) * ${r}u + j],
      qkv[(vCh + d + 1u) * ${r}u + j],
      qkv[(vCh + d + 2u) * ${r}u + j],
      qkv[(vCh + d + 3u) * ${r}u + j]);
  }
  workgroupBarrier();
  var acc : array<vec4f, ${o}>;
  for (var j = 0u; j < ${r}u; j = j + 1u) {
    let wgt = p[j] * inv;
    for (var d4 = 0u; d4 < ${o}u; d4 = d4 + 1u) {
      acc[d4] = fma(vec4f(wgt), kv[j * ${o}u + d4], acc[d4]);
    }
  }
  // attnOut is channel-planar [head*hd + d][n] — pointwise-conv input layout
  for (var d4 = 0u; d4 < ${o}u; d4 = d4 + 1u) {
    attnOut[(qCh + d4 * 4u) * ${r}u + i] = acc[d4].x;
    attnOut[(qCh + d4 * 4u + 1u) * ${r}u + i] = acc[d4].y;
    attnOut[(qCh + d4 * 4u + 2u) * ${r}u + i] = acc[d4].z;
    attnOut[(qCh + d4 * 4u + 3u) * ${r}u + i] = acc[d4].w;
  }
}`;return{label:`attn.core h${i} n${r}`,code:n,workgroups:[i,1,1],buffers:[{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)];case"head":return[function(e){let r=e.h*e.w,t=/* wgsl */`
${a(0)}
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
var<workgroup> gap : array<f32, ${e.cin}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li : u32) {
  for (var ci = li; ci < ${e.cin}u; ci = ci + 256u) {
    var sum = 0.0;
    for (var p = 0u; p < ${r}u; p = p + 1u) { sum = sum + src[ci * ${r}u + p]; }
    gap[ci] = sum / ${r}.0;
  }
  workgroupBarrier();
  for (var co = li; co < ${e.cout}u; co = co + 256u) {
    var acc = 0.0;
    for (var ci = 0u; ci < ${e.cin}u; ci = ci + 1u) {
      acc = fma(gap[ci], W(${e.wOff}u + ci * ${e.cout}u + co), acc);
    }
    dst[co] = acc;
  }
}`;return{label:`head ${e.cin}->${e.cout}`,code:t,workgroups:[1,1,1],buffers:[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)];case"gelu":return[function(e){s(e.n%4==0,`${e.name}: gelu n%4 != 0`);let r=e.n/4,t=/* wgsl */`
@group(0) @binding(0) var<storage, read> src : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;
${o}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${r}u) { return; }
  dst[i] = gelu4(src[i]);
}`;return{label:`gelu n${e.n}`,code:t,workgroups:[Math.ceil(r/64),1,1],buffers:[{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)]}}function l(e){return e.steps.flatMap(c)}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"2Oqph":[function(e,r,t){/**
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
i.export(t,"bwdStepDispatch",()=>u),/** All backward dispatches (loss head + reverse step list), in execution order. */i.export(t,"planBwdDispatches",()=>n);var a=e("./vision_wgsl");// ---------------------------------------------------------------------------
// Shared fragments
// ---------------------------------------------------------------------------
/** gelu'(x) = Φ(x) + x·φ(x), Φ(x)=0.5(1+erf(x/√2)), φ(x)=exp(−x²/2)/√(2π).
 *  Uses the EXACT forward erf4 (imported GELU) so the two never desync. */let o=/* wgsl */`
fn geluGrad4(x : vec4f) -> vec4f {
  let cdf = 0.5 * (vec4f(1.0) + erf4(x * 0.7071067811865476));
  let pdf = 0.3989422804014327 * exp(-0.5 * x * x);   // 1/sqrt(2π)
  return cdf + x * pdf;
}`,s=e=>({kind:"slot",slot:e});function u(e){switch(e.kind){case"loss_bwd":return function(e){let r=e.accumulate?"dx[k] + g":"g",t=/* wgsl */`
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
}`;return{label:`loss_bwd -cos dim${e.dim}`,code:t,workgroups:[1,1,1],buffers:[s(e.embed),{kind:"text"},s(e.dX)]}}(e);case"head_bwd":return function(e){let r=e.h*e.w,t=e.accumulate?"dx[o] + v":"v",i=/* wgsl */`
${(0,a.weightsDecl)(0)}
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
    dgap[ci] = acc / ${r}.0;   // GAP backward: 1/P broadcast
  }
  workgroupBarrier();
  for (var o = li; o < ${e.cin*r}u; o = o + 256u) {
    let v = dgap[o / ${r}u];
    dx[o] = ${t};
  }
}`;return{label:`head_bwd ${e.cout}->${e.cin}${e.accumulate?" +=":""}`,code:i,workgroups:[1,1,1],buffers:[{kind:"weights"},s(e.dY),s(e.dX)]}}(e);case"gelu_bwd":return(// ---------------------------------------------------------------------------
// gelu_bwd — dX = dY ⊙ gelu'(x_pre), one thread per quad.
// ---------------------------------------------------------------------------
function(e){(0,a.assertStep)(e.n%4==0,`${e.name}: gelu_bwd n%4 != 0`);let r=e.n/4,t=e.accumulate?"dst[i] + g":"g",i=/* wgsl */`
@group(0) @binding(0) var<storage, read> dy : array<vec4f>;
@group(0) @binding(1) var<storage, read> pre : array<vec4f>;         // saved pre-activation
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${a.GELU}
${o}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${r}u) { return; }
  let g = dy[i] * geluGrad4(pre[i]);
  dst[i] = ${t};
}`;return{label:`gelu_bwd n${e.n}${e.accumulate?" +=":""}`,code:i,workgroups:[Math.ceil(r/64),1,1],buffers:[s(e.dY),s(e.pre),s(e.dX)]}}(e));case"pw_bwd":return(// ---------------------------------------------------------------------------
// pw_bwd — dX = Wᵀ·dY, tiled pointwise kernel over the transposed weights.
// ---------------------------------------------------------------------------
function(e){let r=e.outH*e.outW;(0,a.assertPointwiseTiles)(e.name,e.cin,e.cout,r,e.wOffT);let t=r/4,i=/* wgsl */`
${(0,a.weightsDecl)(0)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;         // dY  [Cin][P4]
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;   // dX  [Cout][P4]
${a.PW_TILE_DECLS}
${(0,a.pointwiseTiledMain)({cin:e.cin,cout:e.cout,P4:t,wOff:e.wOffT,init:()=>"vec4f(0.0)",store:r=>e.accumulate?`dst[(co + ${r}u) * ${t}u + p4] + acc${r}`:`acc${r}`})}`;return{label:`pw_bwd ${e.cin}->${e.cout} @${e.outH}x${e.outW}${e.accumulate?" +=":""}`,code:i,workgroups:[t/8,e.cout/32,1],buffers:[{kind:"weights"},s(e.dY),s(e.dX)]}}(e));case"residual_bwd":return(// ---------------------------------------------------------------------------
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
}`;return{label:`residual_bwd n${e.n}${e.accumulate?" +=":""}`,code:i,workgroups:[Math.ceil(r/64),1,1],buffers:[s(e.dY),s(e.dX)]}}(e));case"spatial_bwd":return(// ---------------------------------------------------------------------------
// spatial_bwd — gather conv backward, one workgroup per input channel, one
// thread per input pixel. Mirror of the forward spatialConv. Stride∈{1,2}
// baked (stride-2 = parity check on iy+pad−ky). Depthwise = cpg=1 special case.
// ---------------------------------------------------------------------------
function(e){(0,a.assertStep)(1===e.stride||2===e.stride,`${e.name}: stride ${e.stride} not in {1,2}`);let r=e.cin/e.groups,t=e.cout/e.groups;// input channels per group (1 or 3)
(0,a.assertStep)(Number.isInteger(r)&&Number.isInteger(t),`${e.name}: bad groups`);let i=r*e.k*e.k,o=e.h*e.w,u=e.outH*e.outW,n=(r,t)=>1===e.stride?`let ${t} = ${r};`:`if ((${r} & 1) != 0) { continue; } let ${t} = ${r} >> 1;`,d=e.accumulate?"dx[o] + acc":"acc",c=/* wgsl */`
${(0,a.weightsDecl)(0)}
@group(0) @binding(1) var<storage, read> dy : array<f32>;            // [Cout][outH][outW]
@group(0) @binding(2) var<storage, read_write> dx : array<f32>;      // [Cin][H][W]
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let ci = gid.y;
  let p = gid.x;
  if (p >= ${o}u) { return; }
  let iy = i32(p / ${e.w}u);
  let ix = i32(p % ${e.w}u);
  let grp = ci / ${r}u;
  let ci_local = ci - grp * ${r}u;
  var acc = 0.0;
  for (var col = 0u; col < ${t}u; col = col + 1u) {
    let co = grp * ${t}u + col;
    let wbase = ${e.wOff}u + co * ${i}u + ci_local * ${e.k*e.k}u;
    let dybase = co * ${u}u;
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
}`;// one cout's weight footprint
return{label:`spatial_bwd k${e.k}s${e.stride} ${e.cin}<-${e.cout} g${e.groups} @${e.h}x${e.w}${e.accumulate?" +=":""}`,code:c,workgroups:[Math.ceil(o/64),e.cin,1],buffers:[{kind:"weights"},s(e.dY),s(e.dX)]}}(e));case"se_bwd":return function(e){let r=e.h*e.w;(0,a.assertStep)(e.c<=2048&&e.cmid<=512,`${e.name}: SE dims exceed shared-memory plan`);let t=e.accumulate?"dx[i] + v":"v",i=/* wgsl */`
${(0,a.weightsDecl)(0)}
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
    for (var p = 0u; p < ${r}u; p = p + 1u) { sum = sum + src[c * ${r}u + p]; }
    tmp[c] = sum / ${r}.0;
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
    for (var p = 0u; p < ${r}u; p = p + 1u) {
      gscl = fma(dy[c * ${r}u + p], src[c * ${r}u + p], gscl);
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
  for (var i = li; i < ${e.c*r}u; i = i + 256u) {
    let c = i / ${r}u;
    let v = dy[i] * scl[c] + ggap[c] / ${r}.0;
    dx[i] = ${t};
  }
}`;return{label:`se_bwd c${e.c} mid${e.cmid} @${e.h}x${e.w}${e.accumulate?" +=":""}`,code:i,workgroups:[1,1,1],buffers:[{kind:"weights"},s(e.dY),s(e.savedSrc),s(e.dX)]}}(e);case"attn_core_bwd":return(// ---------------------------------------------------------------------------
// attn_core_bwd — MHSA backward, one workgroup per head, one thread per token.
// Two phases: (1) thread = query i recomputes softmax row from saved qkv,
// writes dQ_i and stashes per-row scalars (max, denom, rowdot) in shared mem;
// (2) thread = key/value j accumulates dV_j = Σ_i p_ij·dO_i and
// dK_j = Σ_i dS_ij·q_i, re-reading q_i / dO_i from global. Only 3·nTok floats
// of shared memory — no nTok×nTok materialization.
// ---------------------------------------------------------------------------
function(e){let{c:r,heads:t,hd:i,nTok:o}=e;(0,a.assertStep)(r===t*i,`${e.name}: c != heads*hd`);let u=/* wgsl */`
@group(0) @binding(0) var<storage, read> qkv : array<f32>;           // saved [3C][nTok] planar
@group(0) @binding(1) var<storage, read> dO : array<f32>;            // grad[attnOut] [C][nTok]
@group(0) @binding(2) var<storage, read_write> dQKV : array<f32>;    // grad[qkv] [3C][nTok]
var<workgroup> mrow : array<f32, ${o}>;   // per-query softmax max
var<workgroup> drow : array<f32, ${o}>;   // per-query softmax denom
var<workgroup> rdot : array<f32, ${o}>;   // per-query Σ_k p_ik\xb7dP_ik
@compute @workgroup_size(${o})
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
    qi[d]  = qkv[(qCh + d) * ${o}u + i];
    dOi[d] = dO[(qCh + d) * ${o}u + i];
  }
  var p  : array<f32, ${o}>;
  var dP : array<f32, ${o}>;
  var mx = -3.0e38;
  for (var j = 0u; j < ${o}u; j = j + 1u) {
    var sc = 0.0;
    for (var d = 0u; d < ${i}u; d = d + 1u) { sc = fma(qi[d], qkv[(kCh + d) * ${o}u + j], sc); }
    p[j] = sc;
    mx = max(mx, sc);
  }
  var den = 0.0;
  for (var j = 0u; j < ${o}u; j = j + 1u) { let e = exp(p[j] - mx); p[j] = e; den = den + e; }
  let inv = 1.0 / den;
  var rd = 0.0;
  for (var j = 0u; j < ${o}u; j = j + 1u) {
    p[j] = p[j] * inv;                                  // p_ij
    var dpj = 0.0;
    for (var d = 0u; d < ${i}u; d = d + 1u) { dpj = fma(dOi[d], qkv[(vCh + d) * ${o}u + j], dpj); }
    dP[j] = dpj;                                        // dP_ij = Σ_d dO_i\xb7V_j
    rd = fma(p[j], dpj, rd);                            // Σ_k p_ik\xb7dP_ik
  }
  // dQ_i = Σ_j dS_ij\xb7K_j,  dS_ij = p_ij(dP_ij − rd)
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    var acc = 0.0;
    for (var j = 0u; j < ${o}u; j = j + 1u) {
      let ds = p[j] * (dP[j] - rd);
      acc = fma(ds, qkv[(kCh + d) * ${o}u + j], acc);
    }
    dQKV[(qCh + d) * ${o}u + i] = acc;
  }
  mrow[i] = mx; drow[i] = den; rdot[i] = rd;
  workgroupBarrier();

  // ---- phase 2: thread = key/value token j ----
  let j = tid;
  var kj : array<f32, ${i}>;
  var vj : array<f32, ${i}>;
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    kj[d] = qkv[(kCh + d) * ${o}u + j];
    vj[d] = qkv[(vCh + d) * ${o}u + j];
  }
  var dV : array<f32, ${i}>;
  var dK : array<f32, ${i}>;
  for (var d = 0u; d < ${i}u; d = d + 1u) { dV[d] = 0.0; dK[d] = 0.0; }
  for (var ii = 0u; ii < ${o}u; ii = ii + 1u) {
    // recompute p_ij and dP_ij for this (query ii, key j)
    var sc = 0.0;
    var dpij = 0.0;
    for (var d = 0u; d < ${i}u; d = d + 1u) {
      let qv = qkv[(qCh + d) * ${o}u + ii];
      sc = fma(qv, kj[d], sc);
      dpij = fma(dO[(qCh + d) * ${o}u + ii], vj[d], dpij);
    }
    let pij = exp(sc - mrow[ii]) / drow[ii];
    let dsij = pij * (dpij - rdot[ii]);
    for (var d = 0u; d < ${i}u; d = d + 1u) {
      dV[d] = fma(pij, dO[(qCh + d) * ${o}u + ii], dV[d]);
      dK[d] = fma(dsij, qkv[(qCh + d) * ${o}u + ii], dK[d]);
    }
  }
  for (var d = 0u; d < ${i}u; d = d + 1u) {
    dQKV[(kCh + d) * ${o}u + j] = dK[d];
    dQKV[(vCh + d) * ${o}u + j] = dV[d];
  }
}`;return{label:`attn_core_bwd h${t} n${o}`,code:u,workgroups:[t,1,1],buffers:[s(e.savedQkv),s(e.dY),s(e.dX)]}}(e))}}function n(e){return e.backward.map(u)}},{"./vision_wgsl":"oFDUc","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"8QffC":[function(e,r,t){/**
 * fetch_progress — stream a fetch() body and report download progress so the
 * pages can show a loading bar for the 82 MB CLIP weights. Content-Length gives
 * the total (a CORS-safelisted header, so it's readable even from the HF
 * cross-origin fetch); the stream reader gives bytes-so-far. Falls back to a
 * plain arrayBuffer() if the body isn't streamable, and reports total=0 when
 * Content-Length is absent (caller then shows an indeterminate readout).
 *
 * Shared by src/splat_page.ts and src/splat3d_page.ts (same loader).
 */var i=e("@parcel/transformer-js/src/esmodule-helpers.js");async function a(e,r,t){let i=performance.now(),a=await fetch(e,t);if(!a.ok)throw Error(`fetch ${a.status} ${e}`);let o=Number(a.headers.get("content-length"))||0,s=a.body?.getReader();if(!s){let e=await a.arrayBuffer();return r({received:e.byteLength,total:e.byteLength||o,elapsedMs:performance.now()-i}),e}let u=[],n=0;for(;;){let{done:e,value:t}=await s.read();if(e)break;u.push(t),r({received:n+=t.byteLength,total:o,elapsedMs:performance.now()-i})}let d=new Uint8Array(n),c=0;for(let e of u)d.set(e,c),c+=e.byteLength;return d.buffer}function o(e,r){let t=(r.received/1e6).toFixed(1),i=(r.elapsedMs/1e3).toFixed(1),a=r.elapsedMs>0?(r.received/(r.elapsedMs/1e3)/1e6).toFixed(1):"0.0";if(r.total>0){let o=Math.min(100,Math.round(r.received/r.total*100)),s=(r.total/1e6).toFixed(0),u=Math.round(o/100*16),n="█".repeat(u)+"░".repeat(16-u);return`${e}  [${n}] ${o}%  \xb7  ${t}/${s} MB  \xb7  ${i}s  \xb7  ${a} MB/s`}return`${e}  ${t} MB  \xb7  ${i}s  \xb7  ${a} MB/s`}i.defineInteropFlag(t),i.export(t,"fetchArrayBufferWithProgress",()=>a),/** Compact text bar: "loading CLIP weights [████░░░░] 52% · 43/82 MB · 3.1s · 14 MB/s". */i.export(t,"formatProgress",()=>o)},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}]},["7i9mK"],"7i9mK","parcelRequire924a")//# sourceMappingURL=splat.d18387c2.js.map
;
//# sourceMappingURL=splat.d18387c2.js.map
