!// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles
function(e,r,t,i,a){/* eslint-disable no-undef */var o="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof self?self:"undefined"!=typeof window?window:"undefined"!=typeof global?global:{},s="function"==typeof o[i]&&o[i],n=s.cache||{},u="undefined"!=typeof module&&"function"==typeof module.require&&module.require.bind(module);function d(r,t){if(!n[r]){if(!e[r]){// if we cannot find the module within our internal map or
// cache jump to the current global require ie. the last bundle
// that was added to the page.
var a="function"==typeof o[i]&&o[i];if(!t&&a)return a(r,!0);// If there are other bundles on this page the require from the
// previous one is saved to 'previousRequire'. Repeat this as
// many times as there are bundles until the module is found or
// we exhaust the require chain.
if(s)return s(r,!0);// Try the node require function if it exists.
if(u&&"string"==typeof r)return u(r);var l=Error("Cannot find module '"+r+"'");throw l.code="MODULE_NOT_FOUND",l}p.resolve=function(t){var i=e[r][1][t];return null!=i?i:t},p.cache={};var c=n[r]=new d.Module(r);e[r][0].call(c.exports,p,c,c.exports,this)}return n[r].exports;function p(e){var r=p.resolve(e);return!1===r?{}:d(r)}}d.isParcelRequire=!0,d.Module=function(e){this.id=e,this.bundle=d,this.exports={}},d.modules=e,d.cache=n,d.parent=s,d.register=function(r,t){e[r]=[function(e,r){r.exports=t},{}]},Object.defineProperty(d,"root",{get:function(){return o[i]}}),o[i]=d;for(var l=0;l<r.length;l++)d(r[l]);if(t){// Expose entry point to Node, AMD or browser globals
// Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
var c=d(t);"object"==typeof exports&&"undefined"!=typeof module?module.exports=c:"function"==typeof define&&define.amd?define(function(){return c}):a&&(this[a]=c)}}({e5WXe:[function(e,r,t){let i,a,o,s,n,u;/// <reference types="@webgpu/types" />
var d=e("./splat3d/cameras"),l=e("./splat3d/optimize"),c=e("./splat/fetch_progress");let p={gpu:!!navigator.gpu,ready:!1,running:!1,step:0,view:0,cos:null,initialCos:null,error:null,phase:"boot",promptMode:"camera",blackBgText:!0,profiling:!1};window.__splat3d=p;let f=document.getElementById("grid"),g=document.getElementById("prompt"),h=document.getElementById("view"),m=document.getElementById("promptMode"),v=document.getElementById("bgTextMode"),w=document.getElementById("optimize"),$=document.getElementById("reset"),b=document.getElementById("readout"),x=document.getElementById("notice"),y=document.getElementById("timings");function k(e){x.textContent=e}function _(e){p.error=e,p.phase="error",k(e),b.textContent="—",console.error("[splat3d_page]",e)}function B(){p.step=s?s.stepCount:0;let e=s?.cameras[S]?.name??"view",r=[`step ${p.step}`,e];if(r.push("camera"===p.promptMode?"camera text":"same text"),p.blackBgText&&r.push("black bg"),null!==p.cos){let e=p.initialCos??p.cos,t=p.cos-e;r.push(`cos ${p.cos.toFixed(4)}`),r.push(`init ${e.toFixed(4)}`),r.push(`Δ ${t>=0?"+":""}${t.toFixed(4)}`)}p.phase&&"run"!==p.phase&&r.push(`(${p.phase})`),b.textContent=r.join("  \xb7  ")}function C(){if(!R){y.textContent="sampled wall profile waiting...";return}let e=R,r=Math.max(e.total,.001),t=(e,t)=>`${e.padEnd(11)} ${t.toFixed(1).padStart(6)} ms ${(100*t/r).toFixed(0).padStart(3)}%`;y.textContent=[`sampled wall step ${p.step}`,`${e.views} views \xb7 split submit path`,t("opt total",e.total),t("raster",e.rasterFwd+e.rasterBwd),t("  fwd",e.rasterFwd),t("  bwd",e.rasterBwd),t("clip",e.clipFwd+e.clipBwd),t("  fwd",e.clipFwd),t("  bwd",e.clipBwd),t("adam",e.adam),t("display",e.display),t("clear",e.clear),`sample every ${E} steps`].join("\n")}let P=1,S=0,j=!1,R=null,T=!1,E=30,D=null,A=[],G=[],O=/* wgsl */`
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
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
  let x : u32 = u32(pos.x);
  let y : u32 = u32(pos.y);
  let HW : u32 = 65536u;
  let i : u32 = y * 256u + x;
  return vec4<f32>(img[i], img[HW + i], img[2u * HW + i], 1.0);
}
`;async function I(){i.pushErrorScope("validation");let e=i.createShaderModule({code:O});n=i.createRenderPipeline({layout:"auto",vertex:{module:e,entryPoint:"vs"},fragment:{module:e,entryPoint:"fs",targets:[{format:u}]},primitive:{topology:"triangle-list"}});let r=await i.popErrorScope();if(r)throw Error(`blit pipeline invalid: ${r.message}`)}function q(){D=i.createBindGroup({layout:n.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:s.raster.image}}]})}let M=Function("u","return import(u)"),W=null,F=null;async function L(e){if(F)return;let r=await M("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm");r.env.allowRemoteModels=!0;let t="Nbardy/nff-clip-splat-weights",i=r=>{if("progress"===r.status&&r.total){let t=Math.round(r.progress??r.loaded/r.total*100),i=Math.round(t/100*16),a="█".repeat(i)+"░".repeat(16-i);e?.(`loading text encoder  [${a}] ${t}%  \xb7  ${(r.loaded/1e6).toFixed(1)}/${(r.total/1e6).toFixed(0)} MB`)}};// self-hosted alongside the vision weights
W=await r.AutoTokenizer.from_pretrained(t,{progress_callback:i}),F=await r.CLIPTextModelWithProjection.from_pretrained(t,{dtype:"fp16",device:"wasm",session_options:{graphOptimizationLevel:"basic"},progress_callback:i})}async function z(e){await L();let r=await W(e,{padding:"max_length",max_length:77,truncation:!0}),t=await F(r),i=t.text_embeds.data,a=new Float32Array(512);for(let e=0;e<512;e++)a[e]=i[e];return a}let Y=null,H=0,U=!1;async function X(){if(!Y||T)return;let e=s,r=S;T=!0,p.profiling=!0,p.phase="profile",B();try{let t=await e.profileStep(r);if(e!==s||!p.running)return;R=t,p.step=e.stepCount,H+=1,j=!0,C(),H>=3&&(H=0,V())}catch(e){_(`profile step failed: ${e?.message??e}`)}finally{p.profiling=!1,"profile"===p.phase&&(p.phase=p.running?"run":"idle"),T=!1,B()}}async function V(){if(Y&&!U){U=!0;try{let e=await s.currentEmbedding(S),r=(0,l.cosine)(e,Y[S]);p.cos=r,null===p.initialCos&&(p.initialCos=r),B()}finally{U=!1}}}function N(){if(p.running&&Y&&!T){let e=s.stepCount>0&&s.stepCount%E==0;e?X():(s.step(S),j=!0,H+=1,p.step=s.stepCount,H>=3&&(H=0,V()))}j&&(function(){if(!D||!A.length)return;let e=i.createCommandEncoder();for(let r=0;r<A.length;r++)s.raster.recordForward(e,r),function(e,r){if(!D)return;let t=e.beginRenderPass({colorAttachments:[{view:r.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});t.setPipeline(n),t.setBindGroup(0,D),t.draw(3),t.end()}(e,A[r]);i.queue.submit([e.finish()])}(),j=!1),requestAnimationFrame(N)}async function K(){if(!p.ready||T)return;let e=g.value.trim()||"a photo of a cat";w.disabled=!0,$.disabled=!0,h.disabled=!0,m.disabled=!0,v.disabled=!0,p.running=!1,p.phase="encoding",p.cos=null,p.initialCos=null,R=null,p.promptMode="same"===m.value?"same":"camera",p.blackBgText="none"!==v.value,C(),B();try{let r=[];for(let t=0;t<s.cameras.length;t++){k(`encoding prompt ${t+1}/${s.cameras.length}…`);let i="camera"===p.promptMode?(0,d.buildViewPrompt)(e,s.cameras[t],p.blackBgText):(0,d.buildBasePrompt)(e,p.blackBgText);r.push(await z(i))}Y=r,s.setViewPrompts(r);let t=await s.currentEmbedding(S);p.initialCos=(0,l.cosine)(t,r[S]),p.cos=p.initialCos,H=0,k(""),p.phase="run",p.running=!0,j=!0,B()}catch(e){_(`text encode failed: ${e?.message??e}`)}finally{w.disabled=!1,$.disabled=!1,h.disabled=!1,m.disabled=!1,v.disabled=!1}}async function Z(){if(!p.ready)return;if(T){k("wait for profiling sample to finish before reset");return}p.running=!1,Y=null,p.cos=null,p.initialCos=null,R=null,p.phase="reset",P+=1;let e=s;s=await (0,l.Splat3DOptimizer).create(i,a,o,{seed:P}),e.destroy(),q(),j=!0,p.step=0,k(""),C(),B()}async function Q(){J(Math.max(0,h.selectedIndex)),p.ready&&(p.cos=null,p.initialCos=null,Y&&V(),B())}function J(e){S=Math.max(0,Math.min(s?s.cameras.length-1:0,0|e)),p.view=S,h.selectedIndex=S;for(let e=0;e<G.length;e++)G[e].classList.toggle("active",e===S)}function ee(){p.promptMode="same"===m.value?"same":"camera",p.blackBgText="none"!==v.value,R=null,Y&&(p.running=!1,Y=null,p.cos=null,p.initialCos=null,p.phase="idle",k("")),C(),B()}async function er(){let e;if(!navigator.gpu){_("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled."),w.disabled=!0,$.disabled=!0,h.disabled=!0,m.disabled=!0,v.disabled=!0;return}p.phase="adapter";let r=await navigator.gpu.requestAdapter();if(!r)return _("no WebGPU adapter available.");i=await r.requestDevice(),i.addEventListener?.("uncapturederror",e=>{console.error("[webgpu]",e.error?.message??e.error)}),u=navigator.gpu.getPreferredCanvasFormat(),p.phase="weights";// Prod (GitHub Pages) fetches the packed weights from the HF Hub: GitHub
// release assets send no CORS header, HF does — same host the text model
// loads from (upload via tools/splat/upload_weights.py). Local dev uses the
// fast same-origin static server (tools/splat/serve.mjs). Same loader as the
// 2D page (src/splat_page.ts) — both share the one CLIP weights repo.
let t=["localhost","127.0.0.1"].includes(location.hostname),n=t?"/models/mobileclip_s0/":"https://huggingface.co/Nbardy/nff-clip-splat-weights/resolve/main/",d=t?"":" from HF";b.textContent=`fetching CLIP plan${d}…`;let g=await fetch(n+"plan_train.json");if(!g.ok)return _(`plan_train.json fetch ${g.status} from ${n}`);a=await g.json();try{e=await (0,c.fetchArrayBufferWithProgress)(n+"weights_train.bin",e=>{b.textContent=(0,c.formatProgress)(`loading CLIP weights${d}`,e)})}catch(e){return _(`weights_train.bin fetch failed from ${n}: ${e?.message??e}`)}o=new Float32Array(e),p.phase="optimizer",b.textContent="building 3D optimizer…",await I(),s=await (0,l.Splat3DOptimizer).create(i,a,o,{seed:P}),function(){h.textContent="",f.textContent="",A=[],G=[];for(let e=0;e<s.cameras.length;e++){let r=s.cameras[e],t=document.createElement("option");t.value=r.name,t.textContent=r.name,h.appendChild(t);let a=document.createElement("div");a.className="tile";let o=document.createElement("canvas");o.className="view",o.width=256,o.height=256;let n=document.createElement("div");n.className="label",n.textContent=r.name,a.append(o,n),a.addEventListener("click",()=>{J(e),p.cos=null,p.initialCos=null,Y&&V(),B()});let d=o.getContext("webgpu");d.configure({device:i,format:u,alphaMode:"opaque"}),f.appendChild(a),A.push(d),G.push(a)}J(S)}(),q(),j=!0,// Preload the text encoder at boot (with its own progress bar) so the first
// Optimize is instant instead of stalling on an 84 MB download (× the 9 views).
p.phase="textmodel",await L(e=>{b.textContent=e}),p.ready=!0,p.phase="idle",w.disabled=!1,$.disabled=!1,h.disabled=!1,m.disabled=!1,v.disabled=!1,k(""),B(),requestAnimationFrame(N)}w.addEventListener("click",()=>void K()),$.addEventListener("click",()=>void Z()),h.addEventListener("change",()=>void Q()),m.addEventListener("change",ee),v.addEventListener("change",ee),g.addEventListener("keydown",e=>{"Enter"===e.key&&K()}),er().catch(e=>_(`boot failed: ${e?.message??e}`))},{"./splat3d/cameras":"3Dl1Z","./splat3d/optimize":"dTqrt","./splat/fetch_progress":"8QffC"}],"3Dl1Z":[function(e,r,t){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"DEFAULT_3D_CAMERAS",()=>a),i.export(t,"BLACK_BACKGROUND_PROMPT",()=>o),i.export(t,"buildBasePrompt",()=>s),i.export(t,"buildViewPrompt",()=>n),i.export(t,"prepareCamera",()=>u);let a=[{name:"top",promptSuffix:"a top-down camera angle",eye:[0,3.3,0],target:[0,0,0],up:[0,0,-1]},{name:"front",promptSuffix:"a front-facing camera angle",eye:[0,0,3],target:[0,0,0]},{name:"right",promptSuffix:"a camera angle from the right side",eye:[3,0,0],target:[0,0,0]},{name:"back",promptSuffix:"a camera angle from behind",eye:[0,0,-3],target:[0,0,0]},{name:"left",promptSuffix:"a camera angle from the left side",eye:[-3,0,0],target:[0,0,0]},{name:"front-left-high",promptSuffix:"an elevated 45 degree camera angle from the front left looking down",eye:[-2.16,1.7,2.16],target:[0,0,0]},{name:"front-right-high",promptSuffix:"an elevated 45 degree camera angle from the front right looking down",eye:[2.16,1.7,2.16],target:[0,0,0]},{name:"back-right-low",promptSuffix:"a low 45 degree camera angle from the rear right looking up",eye:[2.16,-1.3,-2.16],target:[0,0,0]},{name:"back-left-low",promptSuffix:"a low 45 degree camera angle from the rear left looking up",eye:[-2.16,-1.3,-2.16],target:[0,0,0]}],o="on a black background";function s(e,r=!0){let t=e.trim()||"a photo of a cat";return!r||/\bblack background\b/i.test(t)?t:`${t}, ${o}`}function n(e,r,t=!0){return s(`${e.trim()||"a photo of a cat"}, ${r.promptSuffix}`,t)}function u(e,r){var t,i;let a=c((t=e.target,i=e.eye,[t[0]-i[0],t[1]-i[1],t[2]-i[2]])),o=e.up??[0,1,0],s=c(d(a,o));1e-5>l(s)&&(s=c(d(a,[0,0,1])));let n=c(d(s,a)),u=(e.fovYDeg??50)*Math.PI/180;return{...e,right:s,cameraUp:n,forward:a,focalPx:.5*r/Math.tan(.5*u)}}function d(e,r){return[e[1]*r[2]-e[2]*r[1],e[2]*r[0]-e[0]*r[2],e[0]*r[1]-e[1]*r[0]]}function l(e){return Math.hypot(e[0],e[1],e[2])}function c(e){let r=1/Math.max(l(e),1e-9);return[e[0]*r,e[1]*r,e[2]*r]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],k3151:[function(e,r,t){t.interopDefault=function(e){return e&&e.__esModule?e:{default:e}},t.defineInteropFlag=function(e){Object.defineProperty(e,"__esModule",{value:!0})},t.exportAll=function(e,r){return Object.keys(e).forEach(function(t){"default"===t||"__esModule"===t||r.hasOwnProperty(t)||Object.defineProperty(r,t,{enumerable:!0,get:function(){return e[t]}})}),r},t.export=function(e,r,t){Object.defineProperty(e,r,{enumerable:!0,get:t})}},{}],dTqrt:[function(e,r,t){/// <reference types="@webgpu/types" />
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"LEGIBLE_3D_G",()=>d),i.export(t,"LEGIBLE_3D_INIT",()=>l),i.export(t,"Splat3DOptimizer",()=>c),i.export(t,"randomSplats3D",()=>p),i.export(t,"cosine",()=>f);var a=e("../clip/vision"),o=e("../splat/adam_wgsl"),s=e("./cameras"),n=e("./raster"),u=e("./raster_wgsl");let d=4096,l={radius:.075,radiusJitter:.35,opacityRaw:.3,colorSpread:1.2,positionSpread:.9};class c{static async create(e,r,t,i={}){let[o,u,l]=r.inputShape;if(3!==o||256!==u||256!==l)throw Error(`splat3d: CLIP inputShape [${o},${u},${l}] != [3,256,256]`);let f=(i.cameras??(0,s.DEFAULT_3D_CAMERAS)).map(e=>(0,s.prepareCamera)(e,256)),g=i.G??d,h=await (0,n.Raster3DEngine).create(e,{H:256,W:256,G:g,cap:i.cap??2048,bg:i.bg??[0,0,0],cameras:f}),m=await (0,a.VisionTrainer).create(e,r,t);return h.setParams(i.initParams??p(g,i.seed??1,i.init)),h.zeroAdamState(),new c(e,h,m,f,i)}constructor(e,r,t,i,a){this.side=256,this.step_=0,this.hasPrompts=!1,this.device=e,this.raster=r,this.trainer=t,this.cameras=i,this.lrs=a.lrs??n.DEFAULT_3D_LRS,this.hyper=a.hyper??o.DEFAULT_HYPER,this.textBuffers=i.map((r,i)=>e.createBuffer({label:`splat3d-text-${i}`,size:4*t.plan.textDim,usage:12}))}setViewPrompts(e){if(e.length!==this.cameras.length)throw Error(`splat3d: ${e.length} text embeds for ${this.cameras.length} cameras`);for(let r=0;r<e.length;r++){if(e[r].length!==this.trainer.plan.textDim)throw Error(`splat3d: view ${r} text ${e[r].length} != ${this.trainer.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffers[r],0,e[r])}this.hasPrompts=!0}step(e=0){if(!this.hasPrompts)throw Error("splat3d: setViewPrompts() before step()");let r=this.device.createCommandEncoder();this.raster.recordClearRawGrad(r);for(let e=0;e<this.cameras.length;e++)r.copyBufferToBuffer(this.textBuffers[e],0,this.trainer.textBuffer,0,4*this.trainer.plan.textDim),this.raster.recordForward(r,e),r.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(r,{backward:!0}),r.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackwardAdd(r,e);this.step_+=1,this.raster.recordAdam(r,this.step_,this.lrs,this.hyper),this.raster.recordForward(r,e),this.device.queue.submit([r.finish()])}async profileStep(e=0){if(!this.hasPrompts)throw Error("splat3d: setViewPrompts() before profileStep()");await this.device.queue.onSubmittedWorkDone();let r={views:this.cameras.length,total:0,clear:0,rasterFwd:0,clipFwd:0,clipBwd:0,rasterBwd:0,adam:0,display:0},t=performance.now();r.clear+=await this.submitTimed(e=>{this.raster.recordClearRawGrad(e)});for(let e=0;e<this.cameras.length;e++)r.rasterFwd+=await this.submitTimed(r=>{this.raster.recordForward(r,e),r.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432)}),r.clipFwd+=await this.submitTimed(e=>{this.trainer.encodeForward(e)}),r.clipBwd+=await this.submitTimed(r=>{r.copyBufferToBuffer(this.textBuffers[e],0,this.trainer.textBuffer,0,4*this.trainer.plan.textDim),this.trainer.encodeBackward(r)}),r.rasterBwd+=await this.submitTimed(r=>{r.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackwardAdd(r,e)});return this.step_+=1,r.adam+=await this.submitTimed(e=>{this.raster.recordAdam(e,this.step_,this.lrs,this.hyper)}),r.display+=await this.submitTimed(r=>{this.raster.recordForward(r,e)}),r.total=performance.now()-t,r}get stepCount(){return this.step_}async renderView(e=0){return this.raster.runForward(e),this.raster.readImage()}renderViewToImage(e=0){this.raster.runForward(e)}async currentEmbedding(e=0){let r=this.device.createCommandEncoder();return this.raster.recordForward(r,e),r.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(r,{backward:!1}),this.device.queue.submit([r.finish()]),g(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){for(let e of(this.raster.destroy(),this.textBuffers))try{e.destroy()}catch(e){}}async submitTimed(e){let r=this.device.createCommandEncoder();e(r);let t=performance.now();return this.device.queue.submit([r.finish()]),await this.device.queue.onSubmittedWorkDone(),performance.now()-t}}function p(e,r=1,t={}){let i=t.radius??l.radius,a=t.radiusJitter??l.radiusJitter,o=t.opacityRaw??l.opacityRaw,s=t.colorSpread??l.colorSpread,n=t.positionSpread??l.positionSpread,d=r>>>0||1,c=()=>{let e=Math.imul((d=Math.imul(d,747796405)+2891336453>>>0)>>>(d>>>28)+4^d,277803737)>>>0;return(e=(e>>>22^e)>>>0)/4294967296},p=()=>{let e=0,r=0;for(;0===e;)e=c();for(;0===r;)r=c();return Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*r)},f=new Float32Array(e*u.PARAM_STRIDE_3D),g=3*e,h=4*e,m=7*e,v=Math.log(i);for(let r=0;r<e;r++)f[0+3*r+0]=(2*c()-1)*n,f[0+3*r+1]=(2*c()-1)*n,f[0+3*r+2]=(2*c()-1)*n,f[g+r]=v+a*p(),f[h+3*r+0]=s*p(),f[h+3*r+1]=s*p(),f[h+3*r+2]=s*p(),f[m+r]=o;return f}function f(e,r){let t=0,i=0,a=0;for(let o=0;o<e.length;o++)t+=e[o]*r[o],i+=e[o]*e[o],a+=r[o]*r[o];return t/Math.sqrt(i*a||1)}async function g(e,r,t){let i=e.createBuffer({size:4*t,usage:9}),a=e.createCommandEncoder();a.copyBufferToBuffer(r,0,i,0,4*t),e.queue.submit([a.finish()]),await i.mapAsync(1);let o=new Float32Array(i.getMappedRange().slice(0));return i.unmap(),i.destroy(),o}},{"../clip/vision":"lNzsi","../splat/adam_wgsl":"kfWkJ","./cameras":"3Dl1Z","./raster":"AoYYi","./raster_wgsl":"fuyeU","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],lNzsi:[function(e,r,t){/**
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
 */i.export(t,"VisionTrainer",()=>l);var a=e("./vision_wgsl"),o=e("./vision_bwd_wgsl");let s={COPY_SRC:4,COPY_DST:8,STORAGE:128};class n{/**
   * Async factory (pipeline validation is async). `weights` must be the
   * packed blob from compile_plan.py — its length is checked against the
   * plan loudly; a mismatched pair cannot run.
   */static async create(e,r,t){if(t.length!==r.weightsFloats)throw Error(`vision: weights blob ${t.length} floats != plan ${r.weightsFloats}`);return new n(e,r,t,await u(e,r))}constructor(e,r,t,i){this.dispatches=[],this.device=e,this.plan=r,this.weightsBuffer=e.createBuffer({size:t.byteLength,usage:s.STORAGE|s.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,t),this.slotBuffers=r.slots.map((r,t)=>e.createBuffer({label:`clip-slot-${t}`,size:4*r,usage:s.STORAGE|s.COPY_DST|s.COPY_SRC})),this.dispatches=i.map(({spec:e,pipeline:r})=>({pipeline:r,workgroups:e.workgroups,label:e.label,bind:this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:e.buffers.map((e,r)=>({binding:r,resource:{// Forward-only: weights + activation slots. A 'text' ref only
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
   *  pointwise-conv + attn_core + pointwise-conv plan steps. */stepDispatchCounts(){return this.plan.steps.map(()=>1)}}async function u(e,r){return d(e,(0,a.planDispatches)(r))}async function d(e,r){let t=[];for(let i of r){e.pushErrorScope("validation");let r=e.createShaderModule({code:i.code}),a=e.createComputePipeline({layout:"auto",compute:{module:r,entryPoint:"main"}}),o=await e.popErrorScope();if(o)throw Error(`vision: pipeline '${i.label}' invalid: ${o.message}
${i.code}`);t.push({spec:i,pipeline:a})}return t}class l{static async create(e,r,t){if(t.length!==r.weightsFloats)throw Error(`vision: weights blob ${t.length} floats != plan ${r.weightsFloats}`);let i=(0,a.planDispatches)(r),s=(0,o.planBwdDispatches)(r),n=await d(e,[...i,...s]);return new l(e,r,t,n,i.length)}constructor(e,r,t,i,a){this.dispatches=[],this.device=e,this.plan=r,this.fwdCount=a,this.weightsBuffer=e.createBuffer({size:t.byteLength,usage:s.STORAGE|s.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,t),this.textBuffer=e.createBuffer({size:4*r.textDim,usage:s.STORAGE|s.COPY_DST}),this.slotBuffers=r.slots.map((r,t)=>e.createBuffer({label:`clip-tslot-${t}`,size:4*r,usage:s.STORAGE|s.COPY_DST|s.COPY_SRC}));let o=e=>"weights"===e.kind?this.weightsBuffer:"text"===e.kind?this.textBuffer:this.slotBuffers[e.slot];this.dispatches=i.map(({spec:e,pipeline:r})=>({pipeline:r,workgroups:e.workgroups,label:e.label,bind:this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:e.buffers.map((e,r)=>({binding:r,resource:{buffer:o(e)}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}get inputGradBuffer(){return this.slotBuffers[this.plan.inputGradSlot]}writeInput(e){let[r,t,i]=this.plan.inputShape;if(e.length!==r*t*i)throw Error(`vision: input ${e.length} != ${r*t*i}`);this.device.queue.writeBuffer(this.inputBuffer,0,e)}/** Target text embedding for the −cos loss (uploaded per prompt change). */writeText(e){if(e.length!==this.plan.textDim)throw Error(`vision: text ${e.length} != ${this.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffer,0,e)}/** Encode forward, then (optionally) the loss head + backward, one pass. */encode(e,r={}){let t=!1===r.backward?this.fwdCount:this.dispatches.length,i=e.beginComputePass();for(let e=0;e<t;e++){let r=this.dispatches[e];i.setPipeline(r.pipeline),i.setBindGroup(0,r.bind),i.dispatchWorkgroups(...r.workgroups)}i.end()}/** Encode only the verified forward pass, preserving activations for backward. */encodeForward(e){let r=e.beginComputePass();for(let e=0;e<this.fwdCount;e++){let t=this.dispatches[e];r.setPipeline(t.pipeline),r.setBindGroup(0,t.bind),r.dispatchWorkgroups(...t.workgroups)}r.end()}/** Encode only the loss head + backward. Requires a prior forward. */encodeBackward(e){let r=e.beginComputePass();for(let e=this.fwdCount;e<this.dispatches.length;e++){let t=this.dispatches[e];r.setPipeline(t.pipeline),r.setBindGroup(0,t.bind),r.dispatchWorkgroups(...t.workgroups)}r.end()}/** Forward + backward. dL/dpixels is left in `inputGradBuffer`. */run(e={}){let r=this.device.createCommandEncoder();this.encode(r,e),this.device.queue.submit([r.finish()])}}},{"./vision_wgsl":"oFDUc","./vision_bwd_wgsl":"2Oqph","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],oFDUc:[function(e,r,t){/**
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
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"weightsDecl",()=>a),i.export(t,"GELU",()=>o),i.export(t,"assertStep",()=>s),i.export(t,"PW_TILE_DECLS",()=>n),/** The shared tiled-matmul body: out[co][p] = Σ_ci src[ci][p]·W[ci*cout+co].
 *  Produces acc0..acc3 (vec4 = 4 pixels × 4 couts) then stores. `init` seeds
 *  each acc (bias for fwd, 0 for bwd); `store` maps acc{j} → the value written
 *  to dst[(co+j)*P4+p4] (gelu/residual epilogue for fwd, add-into for bwd
 *  accumulate). Requires src(binding 1, array<vec4f> [Cin][P4]), dst(binding 2),
 *  weights(binding 0), and PW_TILE_DECLS in scope. */i.export(t,"pointwiseTiledMain",()=>u),/** Assert a pointwise-shaped step satisfies the tile constraints. Shared so the
 *  backward reuses the SAME loud guard (a violating shape needs a handler). */i.export(t,"assertPointwiseTiles",()=>d),i.export(t,"stepDispatches",()=>l),/** All dispatches for a full forward pass, in execution order. */i.export(t,"planDispatches",()=>c);let a=e=>`@group(0) @binding(${e}) var<storage, read> weights : array<vec4f>;
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
`;function s(e,r){if(!e)throw Error(`vision_wgsl: ${r}`)}let n=/* wgsl */`
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 256>;`;function u(e){return/* wgsl */`
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
}`}function d(e,r,t,i,a){s(i%32==0&&t%32==0&&r%32==0,`${e}: tiled pointwise needs P%32==0 && cout%32==0 && cin%32==0 (got P=${i} cin=${r} cout=${t})`),s(a%4==0,`${e}: wOff not 16B-aligned`)}function l(e){switch(e.kind){case"conv":return(// ---------------------------------------------------------------------------
// Thin dispatchers — step kind → dispatch list; conv variant → emitter.
// ---------------------------------------------------------------------------
function(e){switch(e.variant){case"pointwise":return[function(e){let r=e.outH*e.outW;d(e.name,e.cin,e.cout,r,e.wOff);let t=r/4,i=null!==e.residual;s(null!==e.layerScaleOff===i,`${e.name}: layerScale without residual`);let l=[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}];i&&l.push({kind:"slot",slot:e.residual});let c=/* wgsl */`
${a(0)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${i?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${o}
${n}
${u({cin:e.cin,cout:e.cout,P4:t,wOff:e.wOff,init:r=>`vec4f(W(${e.bOff}u + co + ${r}u))`,store:r=>{let a="gelu"===e.act?`gelu4(acc${r})`:`acc${r}`;return i?`res[(co + ${r}u) * ${t}u + p4] + vec4f(W(${e.layerScaleOff}u + co + ${r}u)) * ${a}`:a}})}`;return{label:`pw ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:c,workgroups:[t/8,e.cout/32,1],buffers:l}}(e)];case"depthwise":case"general":return[// ---------------------------------------------------------------------------
// conv:depthwise — k∈{3,7}, groups=C. Thread = one output pixel of one channel.
// ---------------------------------------------------------------------------
function(e){s(null===e.residual&&null===e.layerScaleOff,`${e.name}: spatial conv never carries residual in this plan`),s(e.outW%4==0,`${e.name}: spatial tiling needs outW%4==0`);let r=e.outH*e.outW,t=r/4,i=e.k,n=e.stride,u=e.pad,d=3*n+i,l=e.cin/e.groups,c=e.cout/e.groups,p=l*i*i;s(Number.isInteger(l)&&Number.isInteger(c),`${e.name}: bad groups`),s(p<=64,`${e.name}: weight tile ${p} exceeds one staging round`);let f=r=>"gelu"===e.act?`gelu1(${r})`:r,g=[];for(let r=0;r<l;r++){g.push(`    { let base = (ci0 + ${r}u) * ${e.h*e.w}u;`);for(let t=0;t<i;t++){g.push(`      { let rowBase = base + u32(iy0 + ${t}) * ${e.w}u + u32(ix0);`);for(let e=0;e<d;e++)g.push(`        let r${e} = src[rowBase + ${e}u];`);for(let e=0;e<i;e++)g.push(`        acc = fma(vec4f(r${e}, r${n+e}, r${2*n+e}, r${3*n+e}), vec4f(wk[${r*i*i+t*i+e}u]), acc);`);g.push("      }")}g.push("    }")}// One workgroup = one output channel: its cpg·k·k weights are staged once
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
  let ci0 = (co / ${c}u) * ${l}u;   // first input channel of co's group
  let iy0 = oy * ${n} - ${u};
  let ix0 = ox0 * ${n} - ${u};
  var acc = vec4f(W(${e.bOff}u + co));
  if (iy0 >= 0 && iy0 + ${i} <= ${e.h} && ix0 >= 0 && ix0 + ${d} <= ${e.w}) {
    // interior: every tap in bounds, unchecked unrolled register loads
${g.join("\n")}
  } else {
    // border: per-tap bounds checks (zero padding)
    for (var c = 0u; c < ${l}u; c = c + 1u) {
      let base = (ci0 + c) * ${e.h*e.w}u;
      for (var ky = 0; ky < ${i}; ky = ky + 1) {
        let iy = iy0 + ky;
        if (iy < 0 || iy >= ${e.h}) { continue; }
        let rowBase = base + u32(iy) * ${e.w}u;
        for (var kx = 0; kx < ${i}; kx = kx + 1) {
          let wv = wk[c * ${i*i}u + u32(ky * ${i} + kx)];
          var xv = vec4f(0.0);
          for (var j = 0; j < 4; j = j + 1) {
            let ix = ix0 + j * ${n} + kx;
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
function(e){let{nTok:r,hd:t,heads:i,c:a}=e,o=t/4,n=r*t/4;s(a===i*t,`${e.name}: c != heads*hd`),s(r<=256&&16*n<=16384,`${e.name}: K/V won't fit shared memory`);// channel-planar addressing: channel o at token n sits at o*nTok + n
let u=/* wgsl */`
@group(0) @binding(0) var<storage, read> qkv : array<f32>;
@group(0) @binding(1) var<storage, read_write> attnOut : array<f32>;
var<workgroup> kv : array<vec4f, ${n}>;   // K, then reused for V; [j][d4]
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
  for (var t = i; t < ${n}u; t = t + ${r}u) {
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
  for (var t = i; t < ${n}u; t = t + ${r}u) {
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
}`;return{label:`attn.core h${i} n${r}`,code:u,workgroups:[i,1,1],buffers:[{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)];case"head":return[function(e){let r=e.h*e.w,t=/* wgsl */`
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
}`;return{label:`gelu n${e.n}`,code:t,workgroups:[Math.ceil(r/64),1,1],buffers:[{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}]}}(e)]}}function c(e){return e.steps.flatMap(l)}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"2Oqph":[function(e,r,t){/**
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
i.export(t,"bwdStepDispatch",()=>n),/** All backward dispatches (loss head + reverse step list), in execution order. */i.export(t,"planBwdDispatches",()=>u);var a=e("./vision_wgsl");// ---------------------------------------------------------------------------
// Shared fragments
// ---------------------------------------------------------------------------
/** gelu'(x) = Φ(x) + x·φ(x), Φ(x)=0.5(1+erf(x/√2)), φ(x)=exp(−x²/2)/√(2π).
 *  Uses the EXACT forward erf4 (imported GELU) so the two never desync. */let o=/* wgsl */`
fn geluGrad4(x : vec4f) -> vec4f {
  let cdf = 0.5 * (vec4f(1.0) + erf4(x * 0.7071067811865476));
  let pdf = 0.3989422804014327 * exp(-0.5 * x * x);   // 1/sqrt(2π)
  return cdf + x * pdf;
}`,s=e=>({kind:"slot",slot:e});function n(e){switch(e.kind){case"loss_bwd":return function(e){let r=e.accumulate?"dx[k] + g":"g",t=/* wgsl */`
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
(0,a.assertStep)(Number.isInteger(r)&&Number.isInteger(t),`${e.name}: bad groups`);let i=r*e.k*e.k,o=e.h*e.w,n=e.outH*e.outW,u=(r,t)=>1===e.stride?`let ${t} = ${r};`:`if ((${r} & 1) != 0) { continue; } let ${t} = ${r} >> 1;`,d=e.accumulate?"dx[o] + acc":"acc",l=/* wgsl */`
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
    let dybase = co * ${n}u;
    for (var ky = 0; ky < ${e.k}; ky = ky + 1) {
      let ty = iy + ${e.pad} - ky;
      ${u("ty","oy")}
      if (oy < 0 || oy >= ${e.outH}) { continue; }
      let rowW = wbase + u32(ky) * ${e.k}u;
      let rowY = dybase + u32(oy) * ${e.outW}u;
      for (var kx = 0; kx < ${e.k}; kx = kx + 1) {
        let tx = ix + ${e.pad} - kx;
        ${u("tx","ox")}
        if (ox < 0 || ox >= ${e.outW}) { continue; }
        acc = fma(W(rowW + u32(kx)), dy[rowY + u32(ox)], acc);
      }
    }
  }
  let o = ci * ${o}u + u32(iy) * ${e.w}u + u32(ix);
  dx[o] = ${d};
}`;// one cout's weight footprint
return{label:`spatial_bwd k${e.k}s${e.stride} ${e.cin}<-${e.cout} g${e.groups} @${e.h}x${e.w}${e.accumulate?" +=":""}`,code:l,workgroups:[Math.ceil(o/64),e.cin,1],buffers:[{kind:"weights"},s(e.dY),s(e.dX)]}}(e));case"se_bwd":return function(e){let r=e.h*e.w;(0,a.assertStep)(e.c<=2048&&e.cmid<=512,`${e.name}: SE dims exceed shared-memory plan`);let t=e.accumulate?"dx[i] + v":"v",i=/* wgsl */`
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
function(e){let{c:r,heads:t,hd:i,nTok:o}=e;(0,a.assertStep)(r===t*i,`${e.name}: c != heads*hd`);let n=/* wgsl */`
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
}`;return{label:`attn_core_bwd h${t} n${o}`,code:n,workgroups:[t,1,1],buffers:[s(e.savedQkv),s(e.dY),s(e.dX)]}}(e))}}function u(e){return e.backward.map(n)}},{"./vision_wgsl":"oFDUc","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],kfWkJ:[function(e,r,t){/**
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
 *//** Adam uniform: 8 x 4 bytes = 32 bytes (std140-safe: all scalars). */var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"ADAM_UNIFORM_BYTES",()=>a),i.export(t,"adamShader",()=>o),i.export(t,"DEFAULT_LRS",()=>s),i.export(t,"DEFAULT_HYPER",()=>n);let a=32;function o(){return/* wgsl */`
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
`}let s={mean:.01,logScale:.005,theta:.005,color:.005,opacity:.005},n={beta1:.9,beta2:.999,eps:1e-8}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],AoYYi:[function(e,r,t){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"DEFAULT_3D_LRS",()=>u),i.export(t,"Raster3DEngine",()=>l);var a=e("../splat/adam_wgsl"),o=e("./raster_wgsl");let s={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},n=e=>Math.ceil(e/256),u={position:.025,logRadius:.01,color:.08,opacity:.03};async function d(e,r,t){e.pushErrorScope("validation");let i=e.createShaderModule({code:r}),a=e.createComputePipeline({layout:"auto",compute:{module:i,entryPoint:"main"}}),o=await e.popErrorScope();if(o)throw console.error(`--- WGSL that failed (${t}) ---
${r}`),Error(`raster3d pipeline validation (${t}): ${o.message}`);return a}class l{constructor(e,r){if(this.prepPipe=[],this.chainPipe=[],this.prepBind=[],this.chainBind=[],this.adamUni=[],this.adamBind=[],this.device=e,this.dims=(0,o.resolveDims3D)(r),this.cameras=r.cameras,!this.cameras.length)throw Error("raster3d: at least one camera is required")}static async create(e,r){let t=new l(e,r);return await t.build(r),t}storage(e,r=0){return this.device.createBuffer({size:4*e,usage:s.STORAGE|r})}async build(e){let r=this.dims,t=r.G*o.PARAM_STRIDE_3D,i=r.G*o.DERIVED_STRIDE_3D;this.params=this.storage(t,s.COPY_SRC|s.COPY_DST),this.derived=this.storage(i),this.accGrad=this.storage(i,s.COPY_DST),this.gradRaw=this.storage(t,s.COPY_SRC|s.COPY_DST),this.mBuf=this.storage(t,s.COPY_DST),this.vBuf=this.storage(t,s.COPY_DST),this.tileCounts=this.storage(r.numTiles,s.COPY_DST|s.COPY_SRC),this.binnedIds=this.storage(r.numTiles*r.cap),this.tileStop=this.storage(r.numTiles),this.image=this.storage(3*r.H*r.W,s.COPY_SRC),this.gradImage=this.storage(3*r.H*r.W,s.COPY_DST),this.prepPipe=await Promise.all(this.cameras.map((r,t)=>d(this.device,(0,o.prepShader3D)(e,r),`prep-${t}`))),this.chainPipe=await Promise.all(this.cameras.map((r,t)=>d(this.device,(0,o.chainAddShader3D)(e,r),`chain-${t}`))),this.emitPipe=await d(this.device,(0,o.emitShader3D)(e),"emit"),this.fwdPipe=await d(this.device,(0,o.forwardShader3D)(e),"forward"),this.bwdPipe=await d(this.device,(0,o.backwardShader3D)(e),"backward"),this.clearBinsPipe=await d(this.device,(0,o.clearShader3D)(r.numTiles),"clearBins"),this.clearGradsPipe=await d(this.device,(0,o.clearShader3D)(i),"clearGrads"),this.clearRawPipe=await d(this.device,(0,o.clearShader3D)(t),"clearRawGrad"),this.adamPipe=await d(this.device,(0,a.adamShader)(),"adam");let n=(e,r)=>this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:r.map((e,r)=>({binding:r,resource:{buffer:e}}))});for(let e of(this.prepBind=this.prepPipe.map(e=>n(e,[this.params,this.derived])),this.chainBind=this.chainPipe.map(e=>n(e,[this.accGrad,this.derived,this.params,this.gradRaw])),this.emitBind=n(this.emitPipe,[this.derived,this.tileCounts,this.binnedIds]),this.fwdBind=n(this.fwdPipe,[this.tileCounts,this.binnedIds,this.derived,this.image,this.tileStop]),this.bwdBind=n(this.bwdPipe,[this.gradImage,this.tileCounts,this.binnedIds,this.tileStop,this.derived,this.accGrad]),this.clearBinsBind=n(this.clearBinsPipe,[this.tileCounts]),this.clearGradsBind=n(this.clearGradsPipe,[this.accGrad]),this.clearRawBind=n(this.clearRawPipe,[this.gradRaw]),(0,o.paramSegments3D)(r.G))){let e=this.device.createBuffer({size:a.ADAM_UNIFORM_BYTES,usage:s.UNIFORM|s.COPY_DST});this.adamUni.push(e),this.adamBind.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradRaw}},{binding:3,resource:{buffer:this.mBuf}},{binding:4,resource:{buffer:this.vBuf}}]}))}}setParams(e){if(e.length!==this.dims.G*o.PARAM_STRIDE_3D)throw Error("setParams3D: wrong length");this.device.queue.writeBuffer(this.params,0,e)}zeroAdamState(){let e=new Float32Array(this.dims.G*o.PARAM_STRIDE_3D);this.device.queue.writeBuffer(this.mBuf,0,e),this.device.queue.writeBuffer(this.vBuf,0,e)}async readFloats(e,r){let t=this.device.createBuffer({size:4*r,usage:s.MAP_READ|s.COPY_DST}),i=this.device.createCommandEncoder();i.copyBufferToBuffer(e,0,t,0,4*r),this.device.queue.submit([i.finish()]),await t.mapAsync(1);let a=new Float32Array(t.getMappedRange().slice(0));return t.unmap(),t.destroy(),a}readImage(){return this.readFloats(this.image,3*this.dims.H*this.dims.W)}readParams(){return this.readFloats(this.params,this.dims.G*o.PARAM_STRIDE_3D)}recordClearRawGrad(e){let r=e.beginComputePass();r.setPipeline(this.clearRawPipe),r.setBindGroup(0,this.clearRawBind),r.dispatchWorkgroups(n(this.dims.G*o.PARAM_STRIDE_3D)),r.end()}recordForward(e,r=0){let t=this.dims,i=this.viewIndex(r),a=e.beginComputePass();a.setPipeline(this.prepPipe[i]),a.setBindGroup(0,this.prepBind[i]),a.dispatchWorkgroups(n(t.G)),a.setPipeline(this.clearBinsPipe),a.setBindGroup(0,this.clearBinsBind),a.dispatchWorkgroups(n(t.numTiles)),a.setPipeline(this.emitPipe),a.setBindGroup(0,this.emitBind),a.dispatchWorkgroups(n(t.G)),a.setPipeline(this.fwdPipe),a.setBindGroup(0,this.fwdBind),a.dispatchWorkgroups(t.numTiles),a.end()}recordBackwardAdd(e,r=0){let t=this.dims,i=this.viewIndex(r),a=e.beginComputePass();a.setPipeline(this.clearGradsPipe),a.setBindGroup(0,this.clearGradsBind),a.dispatchWorkgroups(n(t.G*o.DERIVED_STRIDE_3D)),a.setPipeline(this.bwdPipe),a.setBindGroup(0,this.bwdBind),a.dispatchWorkgroups(t.numTiles),a.setPipeline(this.chainPipe[i]),a.setBindGroup(0,this.chainBind[i]),a.dispatchWorkgroups(n(t.G)),a.end()}recordAdam(e,r,t=u,i=a.DEFAULT_HYPER){let s=(0,o.paramSegments3D)(this.dims.G),d={position:t.position,logRadius:t.logRadius,color:t.color,opacity:t.opacity},l=1-Math.pow(i.beta1,r),c=1-Math.pow(i.beta2,r);s.forEach((e,r)=>{let t=new ArrayBuffer(a.ADAM_UNIFORM_BYTES),o=new Uint32Array(t),s=new Float32Array(t);o[0]=e.offset,o[1]=e.length,s[2]=d[e.name],s[3]=i.beta1,s[4]=i.beta2,s[5]=i.eps,s[6]=l,s[7]=c,this.device.queue.writeBuffer(this.adamUni[r],0,t)});let p=e.beginComputePass();p.setPipeline(this.adamPipe),s.forEach((e,r)=>{p.setBindGroup(0,this.adamBind[r]),p.dispatchWorkgroups(n(e.length))}),p.end()}runForward(e=0){let r=this.device.createCommandEncoder();this.recordForward(r,e),this.device.queue.submit([r.finish()])}destroy(){for(let e of[this.params,this.derived,this.accGrad,this.gradRaw,this.mBuf,this.vBuf,this.tileCounts,this.binnedIds,this.tileStop,this.image,this.gradImage,...this.adamUni])try{e.destroy()}catch(e){}}viewIndex(e){return Math.max(0,Math.min(this.cameras.length-1,0|e))}}},{"../splat/adam_wgsl":"kfWkJ","./raster_wgsl":"fuyeU","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],fuyeU:[function(e,r,t){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(t),i.export(t,"TILE",()=>a),i.export(t,"PARAM_STRIDE_3D",()=>o),i.export(t,"DERIVED_STRIDE_3D",()=>s),i.export(t,"ALPHA_THRESHOLD",()=>n),i.export(t,"MAX_ALPHA",()=>u),i.export(t,"TRANSMITTANCE_CUTOFF",()=>d),i.export(t,"EPS",()=>l),i.export(t,"RADIUS_MIN",()=>c),i.export(t,"RADIUS_MAX",()=>p),i.export(t,"resolveDims3D",()=>v),i.export(t,"prepShader3D",()=>b),i.export(t,"emitShader3D",()=>x),i.export(t,"forwardShader3D",()=>y),i.export(t,"backwardShader3D",()=>k),i.export(t,"chainAddShader3D",()=>_),i.export(t,"clearShader3D",()=>B),i.export(t,"paramSegments3D",()=>C);let a=16,o=8,s=11,n=1/255,u=.99,d=1e-4,l=1e-8,c=.01,p=.45;function f(e,r){if(!e)throw Error(`raster3d_wgsl: ${r}`)}function g(e){f(Number.isFinite(e),`non-finite literal ${e}`);let r=e.toString();return/[.eE]/.test(r)||(r+=".0"),r}let h=e=>`${e>>>0}u`,m=e=>`vec3f(${g(e[0])}, ${g(e[1])}, ${g(e[2])})`;function v(e){return f(e.H>0&&e.W>0&&e.G>0,"H,W,G must be positive"),f(e.H%a==0&&e.W%a==0,`H,W must be multiples of ${a}`),f((e.cap&e.cap-1)==0&&e.cap>0,"cap must be a power of two"),f(4*e.cap<=16384,`cap*4 (${4*e.cap}B) exceeds 16KB workgroup storage`),{H:e.H,W:e.W,G:e.G,cap:e.cap,tilesX:e.W/a,tilesY:e.H/a,numTiles:e.W/a*(e.H/a),bg:e.bg??[0,0,0],near:e.near??.2,far:e.far??12,gradScale:e.gradScale??65536}}function w(e){return{position:0,logRadius:3*e.G,colorRaw:4*e.G,opacityRaw:7*e.G}}function $(e){return/* wgsl */`
const CAM_EYE = ${m(e.eye)};
const CAM_RIGHT = ${m(e.right)};
const CAM_UP = ${m(e.cameraUp)};
const CAM_FWD = ${m(e.forward)};
const FOCAL_PX = ${g(e.focalPx)};
`}function b(e,r){let t=v(e),i=w(t);return/* wgsl */`
fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
${$(r)}
@group(0) @binding(0) var<storage, read>       params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${h(t.G)}) { return; }

  let p = vec3f(
    params[${h(i.position)} + g * 3u + 0u],
    params[${h(i.position)} + g * 3u + 1u],
    params[${h(i.position)} + g * 3u + 2u]
  );
  let w = p - CAM_EYE;
  let vx = dot(w, CAM_RIGHT);
  let vy = dot(w, CAM_UP);
  let vz = dot(w, CAM_FWD);
  let safeZ = max(vz, ${g(t.near)});
  let radiusWorld = clamp(exp(params[${h(i.logRadius)} + g]), ${g(c)}, ${g(p)});
  let radiusPx = max(FOCAL_PX * radiusWorld / safeZ, 0.25);
  let invR2 = 1.0 / max(radiusPx * radiusPx, ${g(l)});
  let sx = ${g(.5*t.W)} + FOCAL_PX * (vx / safeZ);
  let sy = ${g(.5*t.H)} - FOCAL_PX * (vy / safeZ);

  let base = g * ${h(s)};
  derived[base + 0u] = sx;
  derived[base + 1u] = sy;
  derived[base + 2u] = invR2;
  derived[base + 3u] = vz;
  derived[base + 4u] = vx;
  derived[base + 5u] = vy;
  derived[base + 6u] = safeZ;
  derived[base + 7u] = sigmoid1(params[${h(i.colorRaw)} + g * 3u + 0u]);
  derived[base + 8u] = sigmoid1(params[${h(i.colorRaw)} + g * 3u + 1u]);
  derived[base + 9u] = sigmoid1(params[${h(i.colorRaw)} + g * 3u + 2u]);
  derived[base + 10u] = sigmoid1(params[${h(i.opacityRaw)} + g]);
}
`}function x(e){let r=v(e);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       derived    : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds  : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${h(r.G)}) { return; }
  let base = g * ${h(s)};
  let depth = derived[base + 3u];
  if (depth <= ${g(r.near)} || depth >= ${g(r.far)}) { return; }
  let op = derived[base + 10u];
  if (op <= ${g(n)}) { return; }
  let ratio = max(${g(n)} / max(op, ${g(l)}), ${g(l)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let sx = derived[base + 0u];
  let sy = derived[base + 1u];
  let invR2 = max(derived[base + 2u], ${g(l)});
  let radius = sqrt(tau / invR2);
  let x0 = max(0, i32(floor(sx - radius - 0.5)));
  let x1 = min(${r.W-1}, i32(ceil(sx + radius - 0.5)));
  let y0 = max(0, i32(floor(sy - radius - 0.5)));
  let y1 = min(${r.H-1}, i32(ceil(sy + radius - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tx0 = x0 / ${a}; let tx1 = x1 / ${a};
  let ty0 = y0 / ${a}; let ty1 = y1 / ${a};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let tile = u32(ty * ${r.tilesX} + tx);
      let slot = atomicAdd(&tileCounts[tile], 1u);
      if (slot < ${h(r.cap)}) { binnedIds[tile * ${h(r.cap)} + slot] = g; }
    }
  }
}
`}function y(e){let r=v(e),t=r.H*r.W;return/* wgsl */`
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;

var<workgroup> sh_ids     : array<u32, ${r.cap}>;
var<workgroup> sh_maxstop : atomic<u32>;

fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u); v = v - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

fn idGreater(a : u32, b : u32) -> bool {
  if (a == 0xffffffffu) { return b != 0xffffffffu; }
  if (b == 0xffffffffu) { return false; }
  let za = derived[a * ${h(s)} + 3u];
  let zb = derived[b * ${h(s)} + 3u];
  if (za == zb) { return a > b; }
  return za > zb;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${h(r.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${h(r.cap)});
  let start = tileId * ${h(r.cap)};
  let sortN = nextPow2(count);

  for (var i = tid; i < sortN; i = i + 256u) {
    sh_ids[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  if (tid == 0u) { atomicStore(&sh_maxstop, 0u); }
  workgroupBarrier();

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
        let swapAsc = idGreater(va, vb);
        let swapDesc = idGreater(vb, va);
        if ((asc && swapAsc) || (!asc && swapDesc)) { sh_ids[pos] = vb; sh_ids[ixj] = va; }
      }
      workgroupBarrier();
      j = j >> 1u;
    }
    k = k << 1u;
  }

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
      let b = gg * ${h(s)};
      let dx = pxc - derived[b + 0u];
      let dy = pyc - derived[b + 1u];
      let invR2 = derived[b + 2u];
      let power = -0.5 * invR2 * (dx * dx + dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[b + 10u] * exp(power);
      let alpha = min(${g(u)}, raw);
      if (alpha < ${g(n)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b + 7u];
      accG = accG + w * derived[b + 8u];
      accB = accB + w * derived[b + 9u];
      T = T * (1.0 - alpha);
      if (T < ${g(d)}) { break; }
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
`}function k(e){let r=v(e),t=r.H*r.W,i=g(r.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;
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
  workgroupBarrier();

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

  var T = 1.0;
  var endi = stopc;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b = gg * ${h(s)};
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let power = -0.5 * derived[b + 2u] * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${g(u)}, derived[b + 10u] * exp(power));
    if (alpha < ${g(n)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${g(d)}) { endi = i + 1u; break; }
  }

  var Tcur = T;
  var gT = goR * ${g(r.bg[0])} + goG * ${g(r.bg[1])} + goB * ${g(r.bg[2])};
  for (var ii = i32(endi) - 1; ii >= 0; ii = ii - 1) {
    let gg = sh_ids[u32(ii)];
    let b = gg * ${h(s)};
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let invR2 = derived[b + 2u];
    let power = -0.5 * invR2 * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let op = derived[b + 10u];
    let raw = op * exp(power);
    let alpha = min(${g(u)}, raw);
    if (alpha < ${g(n)}) { continue; }
    let denom = max(1.0 - alpha, ${g(l)});
    let Tprev = Tcur / denom;
    let cR = derived[b + 7u]; let cG = derived[b + 8u]; let cB = derived[b + 9u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b, 7u, goR * Tprev * alpha);
    fixadd(b, 8u, goG * Tprev * alpha);
    fixadd(b, 9u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${g(u)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-invR2 * dx);
    let gdy = gPower * (-invR2 * dy);
    fixadd(b, 0u, -gdx);
    fixadd(b, 1u, -gdy);
    fixadd(b, 2u, gPower * (-0.5) * (dx * dx + dy * dy));
    fixadd(b, 10u, gRaw * (raw / max(op, ${g(l)})));

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`}function _(e,r){let t=v(e),i=w(t),a=g(1/t.gradScale);return/* wgsl */`
${$(r)}
@group(0) @binding(0) var<storage, read>       accGrad : array<i32>;
@group(0) @binding(1) var<storage, read>       derived : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${h(t.G)}) { return; }
  let b = g * ${h(s)};
  let invScale = ${a};
  let gsx = f32(accGrad[b + 0u]) * invScale;
  let gsy = f32(accGrad[b + 1u]) * invScale;
  let gInv = f32(accGrad[b + 2u]) * invScale;
  let gc0 = f32(accGrad[b + 7u]) * invScale;
  let gc1 = f32(accGrad[b + 8u]) * invScale;
  let gc2 = f32(accGrad[b + 9u]) * invScale;
  let gop = f32(accGrad[b + 10u]) * invScale;

  let vx = derived[b + 4u];
  let vy = derived[b + 5u];
  let vz = max(derived[b + 6u], ${g(t.near)});
  let invR2 = derived[b + 2u];
  let invZ = 1.0 / vz;
  let invZ2 = invZ * invZ;
  let gvx = gsx * FOCAL_PX * invZ;
  let gvy = -gsy * FOCAL_PX * invZ;
  let gvz = gsx * (-FOCAL_PX * vx * invZ2) + gsy * (FOCAL_PX * vy * invZ2) + gInv * (2.0 * invR2 * invZ);
  let gp = CAM_RIGHT * gvx + CAM_UP * gvy + CAM_FWD * gvz;

  gradRaw[${h(i.position)} + g * 3u + 0u] = gradRaw[${h(i.position)} + g * 3u + 0u] + gp.x;
  gradRaw[${h(i.position)} + g * 3u + 1u] = gradRaw[${h(i.position)} + g * 3u + 1u] + gp.y;
  gradRaw[${h(i.position)} + g * 3u + 2u] = gradRaw[${h(i.position)} + g * 3u + 2u] + gp.z;

  let lr = params[${h(i.logRadius)} + g];
  let er = exp(lr);
  let gateR = select(0.0, 1.0, er > ${g(c)} && er < ${g(p)});
  gradRaw[${h(i.logRadius)} + g] = gradRaw[${h(i.logRadius)} + g] + gInv * (-2.0 * invR2) * gateR;

  let col0 = derived[b + 7u]; let col1 = derived[b + 8u]; let col2 = derived[b + 9u];
  let opv = derived[b + 10u];
  gradRaw[${h(i.colorRaw)} + g * 3u + 0u] = gradRaw[${h(i.colorRaw)} + g * 3u + 0u] + gc0 * col0 * (1.0 - col0);
  gradRaw[${h(i.colorRaw)} + g * 3u + 1u] = gradRaw[${h(i.colorRaw)} + g * 3u + 1u] + gc1 * col1 * (1.0 - col1);
  gradRaw[${h(i.colorRaw)} + g * 3u + 2u] = gradRaw[${h(i.colorRaw)} + g * 3u + 2u] + gc2 * col2 * (1.0 - col2);
  gradRaw[${h(i.opacityRaw)} + g] = gradRaw[${h(i.opacityRaw)} + g] + gop * opv * (1.0 - opv);
}
`}function B(e){return/* wgsl */`
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= ${h(e)}) { return; }
  buf[gid.x] = 0u;
}
`}function C(e){return[{name:"position",offset:0,length:3*e},{name:"logRadius",offset:3*e,length:e},{name:"color",offset:4*e,length:3*e},{name:"opacity",offset:7*e,length:e}]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"8QffC":[function(e,r,t){/**
 * fetch_progress — stream a fetch() body and report download progress so the
 * pages can show a loading bar for the 82 MB CLIP weights. Content-Length gives
 * the total (a CORS-safelisted header, so it's readable even from the HF
 * cross-origin fetch); the stream reader gives bytes-so-far. Falls back to a
 * plain arrayBuffer() if the body isn't streamable, and reports total=0 when
 * Content-Length is absent (caller then shows an indeterminate readout).
 *
 * Shared by src/splat_page.ts and src/splat3d_page.ts (same loader).
 */var i=e("@parcel/transformer-js/src/esmodule-helpers.js");async function a(e,r,t){let i=performance.now(),a=await fetch(e,t);if(!a.ok)throw Error(`fetch ${a.status} ${e}`);let o=Number(a.headers.get("content-length"))||0,s=a.body?.getReader();if(!s){let e=await a.arrayBuffer();return r({received:e.byteLength,total:e.byteLength||o,elapsedMs:performance.now()-i}),e}let n=[],u=0;for(;;){let{done:e,value:t}=await s.read();if(e)break;n.push(t),r({received:u+=t.byteLength,total:o,elapsedMs:performance.now()-i})}let d=new Uint8Array(u),l=0;for(let e of n)d.set(e,l),l+=e.byteLength;return d.buffer}function o(e,r){let t=(r.received/1e6).toFixed(1),i=(r.elapsedMs/1e3).toFixed(1),a=r.elapsedMs>0?(r.received/(r.elapsedMs/1e3)/1e6).toFixed(1):"0.0";if(r.total>0){let o=Math.min(100,Math.round(r.received/r.total*100)),s=(r.total/1e6).toFixed(0),n=Math.round(o/100*16),u="█".repeat(n)+"░".repeat(16-n);return`${e}  [${u}] ${o}%  \xb7  ${t}/${s} MB  \xb7  ${i}s  \xb7  ${a} MB/s`}return`${e}  ${t} MB  \xb7  ${i}s  \xb7  ${a} MB/s`}i.defineInteropFlag(t),i.export(t,"fetchArrayBufferWithProgress",()=>a),/** Compact text bar: "loading CLIP weights [████░░░░] 52% · 43/82 MB · 3.1s · 14 MB/s". */i.export(t,"formatProgress",()=>o)},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}]},["e5WXe"],"e5WXe","parcelRequire924a")//# sourceMappingURL=splat3d.0f8c6b60.js.map
;
//# sourceMappingURL=splat3d.0f8c6b60.js.map
