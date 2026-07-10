!// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles
function(e,t,r,i,a){/* eslint-disable no-undef */var s="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof self?self:"undefined"!=typeof window?window:"undefined"!=typeof global?global:{},n="function"==typeof s[i]&&s[i],o=n.cache||{},d="undefined"!=typeof module&&"function"==typeof module.require&&module.require.bind(module);function l(t,r){if(!o[t]){if(!e[t]){// if we cannot find the module within our internal map or
// cache jump to the current global require ie. the last bundle
// that was added to the page.
var a="function"==typeof s[i]&&s[i];if(!r&&a)return a(t,!0);// If there are other bundles on this page the require from the
// previous one is saved to 'previousRequire'. Repeat this as
// many times as there are bundles until the module is found or
// we exhaust the require chain.
if(n)return n(t,!0);// Try the node require function if it exists.
if(d&&"string"==typeof t)return d(t);var u=Error("Cannot find module '"+t+"'");throw u.code="MODULE_NOT_FOUND",u}h.resolve=function(r){var i=e[t][1][r];return null!=i?i:r},h.cache={};var c=o[t]=new l.Module(t);e[t][0].call(c.exports,h,c,c.exports,this)}return o[t].exports;function h(e){var t=h.resolve(e);return!1===t?{}:l(t)}}l.isParcelRequire=!0,l.Module=function(e){this.id=e,this.bundle=l,this.exports={}},l.modules=e,l.cache=o,l.parent=n,l.register=function(t,r){e[t]=[function(e,t){t.exports=r},{}]},Object.defineProperty(l,"root",{get:function(){return s[i]}}),s[i]=l;for(var u=0;u<t.length;u++)l(t[u]);if(r){// Expose entry point to Node, AMD or browser globals
// Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
var c=l(r);"object"==typeof exports&&"undefined"!=typeof module?module.exports=c:"function"==typeof define&&define.amd?define(function(){return c}):a&&(this[a]=c)}}({e5WXe:[function(e,t,r){let i,a,s,n,o,d;/// <reference types="@webgpu/types" />
var l=e("./splat3d/cameras"),u=e("./splat3d/optimize"),c=e("./splat/model_assets");let h={gpu:!!navigator.gpu,ready:!1,running:!1,step:0,view:0,cos:null,initialCos:null,error:null,phase:"boot",qualityPreset:"dream",promptMode:"coarse",gridPromptMode:"contact_sheet",bgPromptMode:"centered",backgroundMode:"curriculum",alphaReg:"weak",boundsReg:"weak",coverageReg:"weak",splatReg:"tiny",framingMode:"zoom_out",profiling:!1,viewsPerStep:9,viewSampler:"random",clipBatchSize:3,clipLayout:"grid9_close2",gridDirectRaster:!0};window.__splat3d=h;let p=document.getElementById("grid"),g=document.getElementById("prompt"),f=document.getElementById("qualityPreset"),m=document.getElementById("view"),w=document.getElementById("promptMode"),v=document.getElementById("bgTextMode"),b=document.getElementById("backgroundMode"),B=document.getElementById("alphaReg"),y=document.getElementById("boundsReg"),x=document.getElementById("coverageReg"),$=document.getElementById("splatReg"),R=document.getElementById("framingMode"),_=document.getElementById("viewBatch"),S=document.getElementById("viewSampler"),P=document.getElementById("clipMode"),C=document.getElementById("clipLayout"),k=document.getElementById("gridPromptMode"),T=document.getElementById("gridRasterMode"),I=document.getElementById("optimize"),G=document.getElementById("reset"),E=document.getElementById("readout"),D=document.getElementById("notice"),O=document.getElementById("timings");function F(e){D.textContent=e}function A(e){h.error=e,h.phase="error",F(e),E.textContent="—",console.error("[splat3d_page]",e)}function z(){h.step=n?n.stepCount:0;let e=n?.cameras[L]?.name??"view",t=[`step ${h.step}`,e];if(t.push("dream"===h.qualityPreset?"dream-ish":"fast"===h.qualityPreset?"fast base":"manual"),n&&t.push(`${h.viewsPerStep}/${n.cameras.length} views`),"random"===h.viewSampler&&t.push("random"),t.push(h.clipBatchSize>1?`clip x${h.clipBatchSize}`:"clip x1"),"grid9_close2"===h.clipLayout&&t.push("grid+2"),"grid9_close2"===h.clipLayout){let e="same"===h.gridPromptMode?"grid=same text":"literal_v2"===h.gridPromptMode?"object grid text":"literal"===h.gridPromptMode?"literal grid text":"grid text";t.push(e),h.gridDirectRaster&&t.push("80px grid raster")}if(t.push("camera"===h.promptMode?"camera text":"coarse"===h.promptMode?"coarse text":"same text"),"black"===h.bgPromptMode&&t.push("black bg"),"centered"===h.bgPromptMode&&t.push("centered bg"),"black"!==h.backgroundMode&&t.push("curriculum"===h.backgroundMode?"bg curriculum":"dark random bg"),"off"!==h.alphaReg&&t.push(`alpha ${h.alphaReg}`),"off"!==h.boundsReg&&t.push(`bounds ${h.boundsReg}`),"off"!==h.coverageReg&&t.push(`coverage ${h.coverageReg}`),"off"!==h.splatReg&&t.push("band"===h.splatReg?"scale band":"anti-tiny"),"zoom_out"===h.framingMode&&t.push("zoom out"),null!==h.cos){let e=h.initialCos??h.cos,r=h.cos-e;t.push(`cos ${h.cos.toFixed(4)}`),t.push(`init ${e.toFixed(4)}`),t.push(`Δ ${r>=0?"+":""}${r.toFixed(4)}`)}h.phase&&"run"!==h.phase&&t.push(`(${h.phase})`),E.textContent=t.join("  \xb7  ")}function M(){if(!j){O.textContent="sampled wall profile waiting...";return}let e=j,t=Math.max(e.total,.001),r=(e,r)=>`${e.padEnd(11)} ${r.toFixed(1).padStart(6)} ms ${(100*r/t).toFixed(0).padStart(3)}%`,i=[`${"gpu-timestamp"===e.timing?"sampled GPU step":"sampled wall step"} ${h.step}`,`${e.views}/${e.totalViews} views \xb7 ${h.viewSampler} \xb7 ${"batch"===e.clipMode?`batch CLIP x${e.clipBatchSize}`:"single CLIP"} \xb7 ${e.timing}`,r("opt total",e.total),r("raster",e.rasterFwd+e.rasterReplay+e.rasterBwd),r("  fwd",e.rasterFwd)];e.rasterReplay>0&&i.push(r("  replay",e.rasterReplay)),i.push(r("  bwd",e.rasterBwd)),"batch"===e.clipMode?i.push(r("clip batch",e.clipBatch)):i.push(r("clip",e.clipFwd+e.clipBwd),r("  fwd",e.clipFwd),r("  bwd",e.clipBwd)),e.regularizer>0&&i.push(r("reg",e.regularizer)),i.push(r("adam",e.adam),r("display",e.display),r("clear",e.clear),`sample every ${q} steps`),O.textContent=i.join("\n")}let W=1,L=0,U=!1,j=null,Y=!1,q=30,V=null,N=[],H=[];function X(){if("grid9_close2"===K())return 3;let e=Number(P.value);return Number.isFinite(e)&&e>1?Math.min(9,0|e):1}function Z(){return"fast"===f.value?"fast":"manual"===f.value?"manual":"dream"}function J(){if("dream"===Z())return{position:.035,logRadius:.018,color:.035,opacity:.025}}function K(){return"grid9_close2"===C.value?"grid9_close2":"per_view"}function Q(){if("grid9_close2"===K())return 9;let e=Number(_.value),t=n?.cameras.length??9;return Number.isFinite(e)?Math.max(1,Math.min(t,0|e)):3}function ee(){return"random"===S.value?"random":"epoch"}function et(){return"same"===k.value?"same":"literal_v2"===k.value?"literal_v2":"literal"===k.value?"literal":"contact_sheet"}function er(){return"direct80"===T.value}function ei(){return"same"===w.value?"same":"coarse"===w.value?"coarse":"camera"}function ea(){return"none"===v.value?"none":"centered"===v.value?"centered":"black"}function es(){return"dark_random"===b.value?"dark_random":"curriculum"===b.value?"curriculum":"black"}function en(){return"medium"===B.value?"medium":"weak"===B.value?"weak":"off"}function eo(){return"medium"===y.value?"medium":"weak"===y.value?"weak":"off"}function ed(){return"medium"===x.value?"medium":"weak"===x.value?"weak":"off"}function el(){return"band"===$.value?"band":"tiny"===$.value?"tiny":"off"}function eu(){let e=en(),t=eo(),r=ed(),i=el();return{backgroundMode:es(),opacitySparsity:"medium"===e?.03:"weak"===e?.01:0,centerWeight:"medium"===t?.006:"weak"===t?.002:0,radiusWeight:"medium"===t?.012:"weak"===t?.004:0,targetRadius:1.15,coverageWeight:"medium"===r?24:"weak"===r?8:0,coverageTarget:"medium"===r?.24:.18,smallRadiusWeight:"band"===i?.035:"tiny"===i?.02:0,smallRadius:.024,radiusBandWeight:"band"===i?.012:0,minRadius:.016,maxRadius:.16}}let ec=!1;function eh(e){if("manual"!==e){ec=!0;try{if("dream"===e){w.value="coarse",v.value="centered",b.value="curriculum",B.value="weak",y.value="weak",x.value="weak",$.value="tiny",R.value="zoom_out",_.value="9",S.value="random",C.value="grid9_close2",P.value="3",k.value="literal_v2",T.value="direct80";return}w.value="camera",v.value="black",b.value="black",B.value="off",y.value="off",x.value="off",$.value="off",R.value="normal",_.value="3",S.value="epoch",C.value="per_view",P.value="3"}finally{ec=!1}}}function ep(){ec||"manual"===f.value||(f.value="manual",h.qualityPreset="manual")}function eg(){h.promptMode=ei(),h.gridPromptMode=et(),h.gridDirectRaster=er(),h.bgPromptMode=ea()}function ef(){h.qualityPreset=Z(),h.backgroundMode=es(),h.alphaReg=en(),h.boundsReg=eo(),h.coverageReg=ed(),h.splatReg=el(),h.framingMode="zoom_out"===R.value?"zoom_out":"normal"}function em(){let e="grid9_close2"===K();e&&(P.value="3",_.value="9")}function ew(e){let t="grid9_close2"===K();I.disabled=e,G.disabled=e,f.disabled=e,m.disabled=e,w.disabled=e,v.disabled=e,b.disabled=e,B.disabled=e,y.disabled=e,x.disabled=e,$.disabled=e,R.disabled=e,C.disabled=e,k.disabled=e||!t,T.disabled=e||!t,_.disabled=e||t,S.disabled=e,P.disabled=e||t}async function ev(e,t){h.phase=t,h.clipLayout=K(),h.gridPromptMode=et(),h.gridDirectRaster=er(),h.viewsPerStep=Q(),h.viewSampler=ee(),h.clipBatchSize=X(),eg(),ef(),z();let r=n;n=await (0,u.Splat3DOptimizer).create(i,a,s,{seed:e,clipBatchSize:h.clipBatchSize,clipLayout:h.clipLayout,viewSampler:h.viewSampler,gridDirectRaster:h.gridDirectRaster,lrs:J(),convergence:eu(),cameras:(0,l.camerasForFraming)(h.framingMode)}),h.clipLayout=n.clipLayout,h.clipBatchSize=n.clipBatchSize,h.viewSampler=n.viewSampler,r?.destroy(),ej(),ey(),U=!0,h.step=0,j=null,M(),z()}let eb=/* wgsl */`
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
`;async function eB(){i.pushErrorScope("validation");let e=i.createShaderModule({code:eb});o=i.createRenderPipeline({layout:"auto",vertex:{module:e,entryPoint:"vs"},fragment:{module:e,entryPoint:"fs",targets:[{format:d}]},primitive:{topology:"triangle-list"}});let t=await i.popErrorScope();if(t)throw Error(`blit pipeline invalid: ${t.message}`)}function ey(){V=i.createBindGroup({layout:o.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:n.raster.image}}]})}let ex=Function("u","return import(u)"),e$=null,eR=null,e_=new Map;async function eS(e){if(eR)return;let t=await ex("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm");t.env.allowRemoteModels=!0;let r="Nbardy/nff-clip-splat-weights",i=t=>{if("progress"===t.status&&t.total){let r=Math.round(t.progress??t.loaded/t.total*100),i=Math.round(r/100*16),a="█".repeat(i)+"░".repeat(16-i);e?.(`loading text encoder  [${a}] ${r}%  \xb7  ${(t.loaded/1e6).toFixed(1)}/${(t.total/1e6).toFixed(0)} MB`)}};// self-hosted alongside the vision weights
e$=await t.AutoTokenizer.from_pretrained(r,{progress_callback:i}),eR=await t.CLIPTextModelWithProjection.from_pretrained(r,{dtype:"fp16",device:"wasm",session_options:{graphOptimizationLevel:"basic"},progress_callback:i})}async function eP(e){await eS();let t=await e$(e,{padding:"max_length",max_length:77,truncation:!0}),r=await eR(t),i=r.text_embeds.data,a=new Float32Array(512);for(let e=0;e<512;e++)a[e]=i[e];return a}function eC(e){let t=e.trim(),r=e_.get(t);return r||(r=eP(t).catch(e=>{throw e_.delete(t),e}),e_.set(t,r)),r}let ek=null,eT=0,eI=!1;async function eG(){if(!ek||Y)return;let e=n,t=L,r=h.viewsPerStep;Y=!0,h.profiling=!0,h.phase="profile",z();try{let i=await e.profileStep(t,r);if(e!==n||!h.running)return;j=i,h.step=e.stepCount,eT+=1,U=!0,M(),eT>=3&&(eT=0,eE())}catch(e){A(`profile step failed: ${e?.message??e}`)}finally{h.profiling=!1,"profile"===h.phase&&(h.phase=h.running?"run":"idle"),Y=!1,z()}}async function eE(){if(ek&&!eI){eI=!0;try{let e=await n.currentEmbedding(L),t=(0,u.cosine)(e,ek[L]);h.cos=t,null===h.initialCos&&(h.initialCos=t),z()}finally{eI=!1}}}function eD(){if(h.running&&ek&&!Y){let e=n.stepCount>0&&n.stepCount%q==0;e?eG():(n.step(L,h.viewsPerStep),U=!0,eT+=1,h.step=n.stepCount,eT>=3&&(eT=0,eE()))}U&&(function(){if(!V||!N.length)return;n.prepareDisplayFrame();let e=i.createCommandEncoder();for(let t=0;t<N.length;t++)n.raster.recordForward(e,t),function(e,t){if(!V)return;let r=e.beginRenderPass({colorAttachments:[{view:t.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});r.setPipeline(o),r.setBindGroup(0,V),r.draw(3),r.end()}(e,N[t]);i.queue.submit([e.finish()])}(),U=!1),requestAnimationFrame(eD)}async function eO(){if(!h.ready||Y)return;em();let e=g.value.trim()||"a photo of a cat";ew(!0),h.running=!1,h.phase="encoding",h.cos=null,h.initialCos=null,j=null,h.clipLayout=K(),h.gridPromptMode=et(),h.gridDirectRaster=er(),h.viewsPerStep=Q(),h.viewSampler=ee(),h.clipBatchSize=X(),h.promptMode=ei(),h.bgPromptMode=ea(),ef(),M(),z();try{let t=[];if("same"===h.promptMode){F("encoding prompt 1/1...");let r=await eC((0,l.buildBasePrompt)(e,h.bgPromptMode));for(let e=0;e<n.cameras.length;e++)t.push(r)}else for(let r=0;r<n.cameras.length;r++){F(`encoding prompt ${r+1}/${n.cameras.length}...`);let i="coarse"===h.promptMode?(0,l.buildCoarseViewPrompt)(e,n.cameras[r],h.bgPromptMode):(0,l.buildViewPrompt)(e,n.cameras[r],h.bgPromptMode);t.push(await eC(i))}ek=t,n.setViewPrompts(t),"grid9_close2"===h.clipLayout&&(F("encoding grid prompt..."),n.setGridPrompt(await eC((0,l.buildGrid9Prompt)(e,h.bgPromptMode,h.gridPromptMode))));let r=await n.currentEmbedding(L);h.initialCos=(0,u.cosine)(r,t[L]),h.cos=h.initialCos,eT=0,F(""),h.phase="run",h.running=!0,U=!0,z()}catch(e){A(`text encode failed: ${e?.message??e}`)}finally{ew(!1)}}async function eF(){if(h.ready){if(Y){F("wait for profiling sample to finish before reset");return}h.running=!1,ek=null,h.cos=null,h.initialCos=null,j=null,h.phase="reset",W+=1,await ev(W,"reset"),h.phase="idle",F(""),z()}}async function eA(){ez(Math.max(0,m.selectedIndex)),h.ready&&(h.cos=null,h.initialCos=null,ek&&eE(),z())}function ez(e){L=Math.max(0,Math.min(n?n.cameras.length-1:0,0|e)),h.view=L,m.selectedIndex=L;for(let e=0;e<H.length;e++)H[e].classList.toggle("active",e===L)}function eM(){ep(),eg(),j=null,ek&&(h.running=!1,ek=null,h.cos=null,h.initialCos=null,h.phase="idle",F("")),M(),z()}async function eW(){if(h.ready){if(Y){F("wait for profiling sample to finish before changing convergence settings"),b.value=h.backgroundMode,B.value=h.alphaReg,y.value=h.boundsReg,x.value=h.coverageReg,$.value=h.splatReg,R.value=h.framingMode;return}ep(),ef(),h.running=!1,ek=null,h.cos=null,h.initialCos=null,j=null,ew(!0);try{await ev(W,"convergence"),F(""),h.phase="idle"}catch(e){A(`convergence settings change failed: ${e?.message??e}`)}finally{ew(!1),z()}}}async function eL(){let e=h.qualityPreset,t=Z();if(h.qualityPreset=t,!h.ready){eh(t),eg(),ef(),h.viewsPerStep=Q(),h.viewSampler=ee(),h.clipBatchSize=X(),z();return}if(Y){F("wait for profiling sample to finish before changing quality preset"),f.value=e,h.qualityPreset=e;return}eh(t),em(),h.running=!1,ek=null,h.cos=null,h.initialCos=null,j=null,ew(!0);try{await ev(W,"preset"),F(""),h.phase="idle"}catch(e){A(`quality preset change failed: ${e?.message??e}`)}finally{ew(!1),z()}}async function eU(){if(h.ready){if(em(),Y){F("wait for profiling sample to finish before changing CLIP settings"),P.value=String(h.clipBatchSize),C.value=h.clipLayout,S.value=h.viewSampler,em();return}ep(),h.running=!1,ek=null,h.cos=null,h.initialCos=null,j=null,ew(!0);try{await ev(W,"optimizer"),F(""),h.phase="idle"}catch(e){A(`clip settings change failed: ${e?.message??e}`)}finally{ew(!1),z()}}}function ej(){m.textContent="",p.textContent="",N=[],H=[];for(let e=0;e<n.cameras.length;e++){let t=n.cameras[e],r=document.createElement("option");r.value=t.name,r.textContent=t.name,m.appendChild(r);let a=document.createElement("div");a.className="tile";let s=document.createElement("canvas");s.className="view",s.width=256,s.height=256;let o=document.createElement("div");o.className="label",o.textContent=t.name,a.append(s,o),a.addEventListener("click",()=>{ez(e),h.cos=null,h.initialCos=null,ek&&eE(),z()});let l=s.getContext("webgpu");l.configure({device:i,format:d,alphaMode:"opaque"}),p.appendChild(a),N.push(l),H.push(a)}ez(L)}async function eY(){if(!navigator.gpu){A("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled."),ew(!0);return}h.phase="adapter";let e=await navigator.gpu.requestAdapter();if(!e)return A("no WebGPU adapter available.");i=await e.requestDevice(),i.addEventListener?.("uncapturederror",e=>{console.error("[webgpu]",e.error?.message??e.error)}),d=navigator.gpu.getPreferredCanvasFormat(),h.phase="weights";try{let e=await (0,c.loadClipTrainAssets)(e=>{E.textContent=e});a=e.plan,s=e.weights}catch(e){return A(e?.message??String(e))}h.phase="optimizer",E.textContent="building 3D optimizer…",await eB(),eh(Z()),em(),eg(),ef(),n=await (0,u.Splat3DOptimizer).create(i,a,s,{seed:W,clipBatchSize:X(),clipLayout:K(),viewSampler:ee(),gridDirectRaster:er(),lrs:J(),convergence:eu(),cameras:(0,l.camerasForFraming)(h.framingMode)}),h.clipLayout=n.clipLayout,h.viewsPerStep=Q(),h.viewSampler=n.viewSampler,h.clipBatchSize=n.clipBatchSize,h.gridPromptMode=et(),h.gridDirectRaster=er(),ej(),ey(),U=!0,// Preload the text encoder at boot (with its own progress bar) so the first
// Optimize is instant instead of stalling on an 84 MB download (× the 9 views).
h.phase="textmodel",await eS(e=>{E.textContent=e}),h.ready=!0,h.phase="idle",ew(!1),F(""),z(),requestAnimationFrame(eD)}I.addEventListener("click",()=>void eO()),G.addEventListener("click",()=>void eF()),m.addEventListener("change",()=>void eA()),f.addEventListener("change",()=>void eL()),w.addEventListener("change",eM),v.addEventListener("change",eM),b.addEventListener("change",()=>void eW()),B.addEventListener("change",()=>void eW()),y.addEventListener("change",()=>void eW()),x.addEventListener("change",()=>void eW()),$.addEventListener("change",()=>void eW()),R.addEventListener("change",()=>void eW()),_.addEventListener("change",function(){ep(),em(),h.viewsPerStep=Q(),j=null,M(),z()}),S.addEventListener("change",()=>void eU()),P.addEventListener("change",()=>void eU()),C.addEventListener("change",()=>void eU()),k.addEventListener("change",eM),T.addEventListener("change",()=>void eU()),g.addEventListener("keydown",e=>{"Enter"===e.key&&eO()}),eY().catch(e=>A(`boot failed: ${e?.message??e}`))},{"./splat3d/cameras":"3Dl1Z","./splat3d/optimize":"dTqrt","./splat/model_assets":"3CXuq"}],"3Dl1Z":[function(e,t,r){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"DEFAULT_3D_CAMERAS",()=>a),i.export(r,"BLACK_BACKGROUND_PROMPT",()=>s),i.export(r,"CENTERED_BLACK_BACKGROUND_PROMPT",()=>n),i.export(r,"normalizeBackgroundPromptMode",()=>o),i.export(r,"buildBasePrompt",()=>d),i.export(r,"buildViewPrompt",()=>l),i.export(r,"buildCoarseViewPrompt",()=>u),i.export(r,"buildGrid9Prompt",()=>c),i.export(r,"camerasForFraming",()=>h),i.export(r,"prepareCamera",()=>p);let a=[{name:"top",promptSuffix:"a top-down camera angle",eye:[0,3.3,0],target:[0,0,0],up:[0,0,-1]},{name:"front",promptSuffix:"a front-facing camera angle",eye:[0,0,3],target:[0,0,0]},{name:"right",promptSuffix:"a camera angle from the right side",eye:[3,0,0],target:[0,0,0]},{name:"back",promptSuffix:"a camera angle from behind",eye:[0,0,-3],target:[0,0,0]},{name:"left",promptSuffix:"a camera angle from the left side",eye:[-3,0,0],target:[0,0,0]},{name:"front-left-high",promptSuffix:"an elevated 45 degree camera angle from the front left looking down",eye:[-2.16,1.7,2.16],target:[0,0,0]},{name:"front-right-high",promptSuffix:"an elevated 45 degree camera angle from the front right looking down",eye:[2.16,1.7,2.16],target:[0,0,0]},{name:"back-right-low",promptSuffix:"a low 45 degree camera angle from the rear right looking up",eye:[2.16,-1.3,-2.16],target:[0,0,0]},{name:"back-left-low",promptSuffix:"a low 45 degree camera angle from the rear left looking up",eye:[-2.16,-1.3,-2.16],target:[0,0,0]}],s="on a black background",n="centered on a black background";function o(e=!0){return!0===e?"black":!1===e?"none":e}function d(e,t=!0){let r=e.trim()||"a photo of a cat",i=o(t);if("none"===i||/\bblack background\b/i.test(r))return r;let a="centered"===i?n:s;return`${r}, ${a}`}function l(e,t,r=!0){return d(`${e.trim()||"a photo of a cat"}, ${t.promptSuffix}`,r)}function u(e,t,r=!0){return d(`${e.trim()||"a photo of a cat"}, ${function(e){switch(e.name){case"top":return"a top-down view";case"front":return"a front view";case"back":return"a back view";case"left":case"right":return"a side view";default:return e.eye[1]>=0?"an elevated side view looking down":"a low side view looking up"}}(t)}`,r)}function c(e,t=!0,r="contact_sheet"){if("same"===r)return d(e,t);let i=e.trim()||"a photo of a cat",a=o(t),s="none"===a||/\bblack background\b/i.test(i)?"":", centered on a black background",n="top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up";return"literal_v2"===r?`a grid of 9 different camera angles of the same object, the object is centered, and the object is ${i}${s}`:"literal"===r?`a 3x3 grid showing ${i} from 9 different camera angles${s}. The 9 panels show the same subject in reading order: ${n}`:`a 3x3 image grid showing the same subject, ${i}, from nine different camera angles${s}: `+n}function h(e){return"zoom_out"!==e?a:a.map(e=>({...e,eye:[1.25*e.eye[0],1.25*e.eye[1],1.25*e.eye[2]],fovYDeg:Math.max(e.fovYDeg??50,56)}))}function p(e,t){var r,i;let a=m((r=e.target,i=e.eye,[r[0]-i[0],r[1]-i[1],r[2]-i[2]])),s=e.up??[0,1,0],n=m(g(a,s));1e-5>f(n)&&(n=m(g(a,[0,0,1])));let o=m(g(n,a)),d=(e.fovYDeg??50)*Math.PI/180;return{...e,right:n,cameraUp:o,forward:a,focalPx:.5*t/Math.tan(.5*d)}}function g(e,t){return[e[1]*t[2]-e[2]*t[1],e[2]*t[0]-e[0]*t[2],e[0]*t[1]-e[1]*t[0]]}function f(e){return Math.hypot(e[0],e[1],e[2])}function m(e){let t=1/Math.max(f(e),1e-9);return[e[0]*t,e[1]*t,e[2]*t]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],"4C2Su":[function(e,t,r){r.interopDefault=function(e){return e&&e.__esModule?e:{default:e}},r.defineInteropFlag=function(e){Object.defineProperty(e,"__esModule",{value:!0})},r.exportAll=function(e,t){return Object.keys(e).forEach(function(r){"default"===r||"__esModule"===r||t.hasOwnProperty(r)||Object.defineProperty(t,r,{enumerable:!0,get:function(){return e[r]}})}),t},r.export=function(e,t,r){Object.defineProperty(e,t,{enumerable:!0,get:r})}},{}],dTqrt:[function(e,t,r){/// <reference types="@webgpu/types" />
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"LEGIBLE_3D_G",()=>h),i.export(r,"LEGIBLE_3D_INIT",()=>p),i.export(r,"Splat3DOptimizer",()=>g),i.export(r,"randomSplats3D",()=>B),i.export(r,"cosine",()=>y);var a=e("../clip/vision"),s=e("../clip/vision_batch"),n=e("../splat/adam_wgsl"),o=e("./cameras"),d=e("./grid_clip"),l=e("./raster"),u=e("./raster_wgsl");let c={COPY_SRC:4,COPY_DST:8},h=4096,p={radius:.075,radiusJitter:.35,opacityRaw:.3,colorSpread:1.2,positionSpread:.9};class g{static async create(e,t,r,i={}){let[n,u,c]=t.inputShape;if(3!==n||256!==u||256!==c)throw Error(`splat3d: CLIP inputShape [${n},${u},${c}] != [3,256,256]`);let p=(i.cameras??(0,o.DEFAULT_3D_CAMERAS)).map(e=>(0,o.prepareCamera)(e,256)),m=i.G??h,w=f(i.convergence),v=await (0,l.Raster3DEngine).create(e,{H:256,W:256,G:m,cap:i.cap??2048,bg:i.bg??[0,0,0],dynamicBg:"black"!==w.backgroundMode,dynamicCoverage:0!==w.coverageWeight,cameras:p}),b=function(e){let t=Number.isFinite(e)?0|e:1;return t>1?Math.min(9,t):1}(i.clipBatchSize),y=i.clipLayout??"per_view",x="grid9_close2"===y&&3===b&&(i.gridDirectRaster??!1)===!0&&(i.stemSpatialBwd??!0)===!0&&(i.fusePointwiseGeluForward??!0)===!0&&(i.clipWeightPrecision??"f32")==="f32",$=i.spatialBwdVariant??(x?"depthwise4":void 0),R=x&&"depthwise4"===$,_={weightPrecision:i.clipWeightPrecision,pointwiseTileVariant:i.pointwiseTileVariant,pointwiseTileSteps:i.pointwiseTileSteps,stemSpatialBwd:i.stemSpatialBwd??!0,spatialBwdVariant:$,fusePointwiseGeluForward:i.fusePointwiseGeluForward??!0,fuseGeluBwdIntoPw:i.fuseGeluBwdIntoPw??R,fuseResidualBwdIntoPw:i.fuseResidualBwdIntoPw??R},S=await (0,a.VisionTrainer).create(e,t,r,_),P=b>1?await (0,s.BatchMajorVisionTrainer).create(e,t,r,b,{weightPrecision:i.clipWeightPrecision,stemSpatialBwd:_.stemSpatialBwd,spatialBwdVariant:_.spatialBwdVariant,sharedWForwardSteps:i.sharedWForwardSteps,fusePointwiseGeluForward:_.fusePointwiseGeluForward,fuseGeluBwdIntoPw:_.fuseGeluBwdIntoPw,fuseResidualBwdIntoPw:_.fuseResidualBwdIntoPw}):null;if("grid9_close2"===y&&!P)throw Error("splat3d: CLIP_LAYOUT=grid9_close2 needs CLIP_BATCH=3");if("grid9_close2"===y&&p.length<9)throw Error(`splat3d: CLIP_LAYOUT=grid9_close2 needs at least 9 cameras, got ${p.length}`);v.setParams(i.initParams??B(m,i.seed??1,i.init)),v.zeroAdamState();let C=P&&(i.viewLaneBatchRasterForward||i.viewLaneBatchRasterBackward)?await v.createBatchForwardState({lanes:P.batch,imageBuffer:P.inputBuffer,imageOffsets:Array.from({length:P.batch},(e,t)=>P.slotOffsetBytes(t,P.plan.inputSlot)),gradBuffer:P.inputGradBuffer,gradOffsets:Array.from({length:P.batch},(e,t)=>P.inputGradOffsetBytes(t))}):null,k="grid9_close2"===y&&P?await (0,d.Grid9Close2ClipLayout).create(e,v,P,{directRaster:i.gridDirectRaster??!1}):null;return new g(e,v,S,P,k,C,p,i)}constructor(e,t,r,i,a,s,o,d){var u;this.side=256,this.step_=0,this.hasPrompts=!1,this.rngState=1,this.viewOrder=[],this.viewCursor=0,this.cachedBatchViews=null,this.device=e,this.raster=t,this.trainer=r,this.batchTrainer=i,this.gridClip=a,this.batchRasterForward=s,this.cameras=o,this.clipBatchSize=i?.batch??1,this.clipLayout=d.clipLayout??"per_view",this.viewSampler=d.viewSampler??"epoch",this.lrs=d.lrs??l.DEFAULT_3D_LRS,this.hyper=d.hyper??n.DEFAULT_HYPER,this.singlePassBatchRasterForward=d.singlePassBatchRasterForward??!1,this.viewLaneBatchRasterForward=d.viewLaneBatchRasterForward??!1,this.viewLaneBatchRasterBackward=d.viewLaneBatchRasterBackward??!1,this.gridDirectRaster=d.gridDirectRaster??!1,this.clipRefreshInterval=Math.max(1,d.clipRefreshInterval??1),this.cachedLrScale=void 0!==(u=d.cachedLrScale)&&Number.isFinite(u)?Math.max(0,u):1,this.convergence=f(d.convergence),this.rngState=((d.seed??1)^2654435769)>>>0||1,this.textBuffers=o.map((t,i)=>e.createBuffer({label:`splat3d-text-${i}`,size:4*r.plan.textDim,usage:c.COPY_SRC|c.COPY_DST})),this.gridTextBuffer=a?e.createBuffer({label:"splat3d-grid9-text",size:4*r.plan.textDim,usage:c.COPY_SRC|c.COPY_DST}):null,this.singleIO=t.createIOState(r.inputBuffer,0,r.inputGradBuffer,0),this.batchIO=s?.ios??(i?Array.from({length:i.batch},(e,r)=>t.createIOState(i.inputBuffer,i.slotOffsetBytes(r,i.plan.inputSlot),i.inputGradBuffer,i.inputGradOffsetBytes(r),{privateState:!0})):[])}setViewPrompts(e){if(e.length!==this.cameras.length)throw Error(`splat3d: ${e.length} text embeds for ${this.cameras.length} cameras`);for(let t=0;t<e.length;t++){if(e[t].length!==this.trainer.plan.textDim)throw Error(`splat3d: view ${t} text ${e[t].length} != ${this.trainer.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffers[t],0,e[t])}this.gridTextBuffer&&this.device.queue.writeBuffer(this.gridTextBuffer,0,e[0]),this.hasPrompts=!0}setGridPrompt(e){if(this.gridTextBuffer){if(e.length!==this.trainer.plan.textDim)throw Error(`splat3d: grid text ${e.length} != ${this.trainer.plan.textDim}`);this.device.queue.writeBuffer(this.gridTextBuffer,0,e)}}step(e=0,t=this.cameras.length){if(!this.hasPrompts)throw Error("splat3d: setViewPrompts() before step()");this.applyTrainingBackground(),this.applyCoverageRegularizer();let r=this.shouldUseCachedBatchStep(t),i=r?this.cachedBatchViews.slice():this.sampleViews(t),a=this.device.createCommandEncoder();this.raster.recordClearRawGrad(a),r?this.recordCachedBatchTrainingViews(a,i):(this.recordTrainingViews(a,i),this.updateCachedBatchViews(i)),this.recordConvergenceRegularizer(a),this.step_+=1,this.raster.recordAdam(a,this.step_,this.lrsForStep(r),this.hyper),this.raster.recordForward(a,e),this.device.queue.submit([a.finish()])}async profileStep(e=0,t=this.cameras.length,r={}){if(!this.hasPrompts)throw Error("splat3d: setViewPrompts() before profileStep()");await this.device.queue.onSubmittedWorkDone(),this.applyTrainingBackground(),this.applyCoverageRegularizer();let i=this.shouldUseCachedBatchStep(t),a=i?this.cachedBatchViews.slice():this.sampleViews(t),s=r.gpuTimestamps?b.create(this.device):null,n={views:a.length,totalViews:this.cameras.length,clipMode:this.useBatchFor(a)?"batch":"single",clipBatchSize:this.clipBatchSize,timing:s?"gpu-timestamp":"split-submit-wall",total:0,clear:0,rasterFwd:0,rasterReplay:0,clipFwd:0,clipBwd:0,clipBatch:0,rasterBwd:0,regularizer:0,adam:0,display:0},o=performance.now();try{if(n.clear+=await this.submitTimed((e,t)=>{this.raster.recordClearRawGrad(e,t)},s),i)n.rasterFwd+=await this.profileCachedBatchInputs(a,s),n.rasterBwd+=await this.profileCachedBatchBackward(a,s);else if(this.useGridLayoutFor(a)){let e=this.batchTrainer,t=a.slice(0,9),r=this.grid9CloseupViews(t);n.rasterFwd+=await this.profileGrid9Close2Inputs(t,r,s),n.clipBatch+=await this.submitTimed((t,r)=>{e.encode(t,{backward:!0,timestampWrites:r})},s);let i=await this.profileGrid9Close2Backward(t,r,s);n.rasterReplay+=i.replay,n.rasterBwd+=i.backward}else if(this.useBatchFor(a)){let e=this.batchTrainer;for(let t=0;t<a.length;t+=e.batch){let r=a.slice(t,t+e.batch);if(r.length<e.batch){for(let e of r)n.rasterFwd+=await this.submitTimed((t,r)=>this.recordSingleForwardToTrainer(t,e,r),s),n.clipFwd+=await this.submitTimed((e,t)=>this.trainer.encodeForward(e,t),s),n.clipBwd+=await this.submitTimed((t,r)=>this.recordSingleTextAndBackward(t,e,r),s),n.rasterBwd+=await this.submitTimed((t,r)=>this.recordSingleRasterBackward(t,e,r),s);continue}if(n.rasterFwd+=await this.profileBatchInputs(r,s),n.clipBatch+=await this.submitTimed((t,r)=>{e.encode(t,{backward:!0,timestampWrites:r})},s),this.viewLaneBatchRasterBackward&&this.batchRasterForward&&r.length>1){n.rasterBwd+=await this.submitTimed((e,t)=>{this.raster.recordBatchBackwardAdd(e,this.batchRasterForward,r,t)},s);continue}for(let e=0;e<r.length;e++){let t=r[e],i=this.batchIO[e];n.rasterBwd+=await this.submitTimed((e,r)=>{this.raster.recordBackwardAdd(e,t,i,r)},s)}}}else for(let e of a)n.rasterFwd+=await this.submitTimed((t,r)=>this.recordSingleForwardToTrainer(t,e,r),s),n.clipFwd+=await this.submitTimed((e,t)=>{this.trainer.encodeForward(e,t)},s),n.clipBwd+=await this.submitTimed((t,r)=>this.recordSingleTextAndBackward(t,e,r),s),n.rasterBwd+=await this.submitTimed((t,r)=>this.recordSingleRasterBackward(t,e,r),s);return i||this.updateCachedBatchViews(a),this.convergenceRegularizerEnabled()&&(n.regularizer+=await this.submitTimed((e,t)=>{this.recordConvergenceRegularizer(e,t)},s)),this.step_+=1,n.adam+=await this.submitTimed((e,t)=>{this.raster.recordAdam(e,this.step_,this.lrsForStep(i),this.hyper,t)},s),this.applyDisplayBackground(),n.display+=await this.submitTimed((t,r)=>{this.raster.recordForward(t,e,void 0,r)},s),n.total=s?n.clear+n.rasterFwd+n.rasterReplay+n.clipFwd+n.clipBwd+n.clipBatch+n.rasterBwd+n.regularizer+n.adam+n.display:performance.now()-o,n}finally{s?.destroy()}}get stepCount(){return this.step_}async renderView(e=0){return this.applyDisplayBackground(),this.raster.runForward(e),this.raster.readImage()}renderViewToImage(e=0){this.applyDisplayBackground(),this.raster.runForward(e)}async currentEmbedding(e=0){this.applyDisplayBackground();let t=this.device.createCommandEncoder();return this.raster.recordForward(t,e,this.singleIO),this.trainer.encode(t,{backward:!1}),this.device.queue.submit([t.finish()]),x(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){for(let e of(this.raster.destroy(),this.trainer.destroy(),this.batchTrainer?.destroy(),this.gridClip?.destroy(),this.gridTextBuffer?.destroy(),this.textBuffers))try{e.destroy()}catch(e){}}prepareDisplayFrame(){this.applyDisplayBackground()}useBatchFor(e){return!!this.batchTrainer&&e.length>=this.batchTrainer.batch}useGridLayoutFor(e){if("grid9_close2"!==this.clipLayout)return!1;if(!this.batchTrainer||!this.gridClip||!this.gridTextBuffer)throw Error("splat3d: grid9_close2 layout was not initialized");if(e.length<9)throw Error(`splat3d: grid9_close2 needs VIEWS=9, got ${e.length}`);if(this.batchTrainer.batch<3)throw Error(`splat3d: grid9_close2 needs CLIP_BATCH=3, got ${this.batchTrainer.batch}`);return!0}grid9CloseupViews(e){let t=e.length;if("random"===this.viewSampler){let r=Math.floor(v(this.step_,101)*t)%t,i=Math.floor(v(this.step_,211)*t)%t;return i===r&&(i=(i+4)%t),[e[r],e[i]]}let r=this.step_%t;return[e[r],e[(r+4)%t]]}recordTrainingViews(e,t){if(this.useGridLayoutFor(t)){this.recordGrid9Close2Training(e,t.slice(0,9));return}if(!this.useBatchFor(t)){for(let r of t)this.recordSingleTrainingView(e,r);return}let r=this.batchTrainer;for(let i=0;i<t.length;i+=r.batch){let a=t.slice(i,i+r.batch);if(a.length<r.batch){for(let t of a)this.recordSingleTrainingView(e,t);continue}if(this.recordBatchInputs(e,a),r.encode(e,{backward:!0}),this.viewLaneBatchRasterBackward&&this.batchRasterForward&&a.length>1){this.raster.recordBatchBackwardAdd(e,this.batchRasterForward,a);continue}for(let t=0;t<a.length;t++){let r=a[t],i=this.batchIO[t];this.raster.recordBackwardAdd(e,r,i)}}}recordCachedBatchTrainingViews(e,t){this.recordCachedBatchInputs(e,t),this.recordCachedBatchBackward(e,t)}recordCachedBatchInputs(e,t){if(!this.batchTrainer)throw Error("splat3d: cached CLIP step needs batch trainer");if(t.length!==this.batchTrainer.batch)throw Error(`splat3d: cached CLIP step needs one full batch, got ${t.length}`);if(this.singlePassBatchRasterForward&&t.length>1){this.raster.recordForwards(e,t,this.batchIO.slice(0,t.length));return}if(this.viewLaneBatchRasterForward&&this.batchRasterForward&&t.length>1){this.raster.recordBatchForward(e,this.batchRasterForward,t);return}for(let r=0;r<t.length;r++)this.raster.recordForward(e,t[r],this.batchIO[r])}recordCachedBatchBackward(e,t){if(this.viewLaneBatchRasterBackward&&this.batchRasterForward&&t.length>1){this.raster.recordBatchBackwardAdd(e,this.batchRasterForward,t);return}for(let r=0;r<t.length;r++)this.raster.recordBackwardAdd(e,t[r],this.batchIO[r])}recordGrid9Close2Training(e,t){let r=this.batchTrainer,i=this.grid9CloseupViews(t);this.recordGrid9Close2Inputs(e,t,i),r.encode(e,{backward:!0}),this.recordGrid9Close2Backward(e,t,i)}recordGrid9Close2Inputs(e,t,r){let i=this.batchTrainer,a=this.gridClip;this.recordGrid9Close2TextCopies(e,r),a.clearGridImage(e);for(let r=0;r<9;r++)a.raster.recordForward(e,t[r],a.scratchIO),a.recordCopyCell(e,r);for(let t=0;t<2;t++)this.raster.recordForward(e,r[t],this.batchIO[t+1]);// The batch variable is intentionally touched here so future edits keep the
// lane contract visible: lane 0 grid, lanes 1-2 close-ups.
if(i.batch<3)throw Error("splat3d: grid9_close2 lost its CLIP batch")}recordGrid9Close2Backward(e,t,r){let i=this.gridClip;for(let r=0;r<9;r++)i.clearScratchGrad(e),i.recordScatterCell(e,r),i.raster.recordForward(e,t[r],i.scratchIO),i.raster.recordBackwardAdd(e,t[r],i.scratchIO);for(let t=0;t<2;t++)this.raster.recordBackwardAdd(e,r[t],this.batchIO[t+1])}recordGrid9Close2TextCopies(e,t){let r=this.batchTrainer,i=4*r.plan.textDim;e.copyBufferToBuffer(this.gridTextBuffer,0,r.textBuffer,r.textOffsetBytes(0),i);for(let a=0;a<2;a++){let s=t[a];e.copyBufferToBuffer(this.textBuffers[s],0,r.textBuffer,r.textOffsetBytes(a+1),i)}}recordSingleTrainingView(e,t){e.copyBufferToBuffer(this.textBuffers[t],0,this.trainer.textBuffer,0,4*this.trainer.plan.textDim),this.raster.recordForward(e,t,this.singleIO),this.trainer.encode(e,{backward:!0}),this.raster.recordBackwardAdd(e,t,this.singleIO)}recordBatchInputs(e,t){if(this.recordBatchTextCopies(e,t),this.singlePassBatchRasterForward&&t.length>1){this.raster.recordForwards(e,t,this.batchIO.slice(0,t.length));return}if(this.viewLaneBatchRasterForward&&this.batchRasterForward&&t.length>1){this.raster.recordBatchForward(e,this.batchRasterForward,t);return}for(let r=0;r<t.length;r++)this.raster.recordForward(e,t[r],this.batchIO[r])}async profileBatchInputs(e,t){if(!t)return this.submitTimed(t=>this.recordBatchInputs(t,e));let r=this.device.createCommandEncoder();if(this.recordBatchTextCopies(r,e),this.device.queue.submit([r.finish()]),await this.device.queue.onSubmittedWorkDone(),this.singlePassBatchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordForwards(t,e,this.batchIO.slice(0,e.length),r)},t);if(this.viewLaneBatchRasterForward&&this.batchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordBatchForward(t,this.batchRasterForward,e,r)},t);let i=0;for(let r=0;r<e.length;r++)i+=await this.submitTimed((t,i)=>{this.raster.recordForward(t,e[r],this.batchIO[r],i)},t);return i}async profileCachedBatchInputs(e,t){if(!t)return this.submitTimed(t=>this.recordCachedBatchInputs(t,e));if(this.singlePassBatchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordForwards(t,e,this.batchIO.slice(0,e.length),r)},t);if(this.viewLaneBatchRasterForward&&this.batchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordBatchForward(t,this.batchRasterForward,e,r)},t);let r=0;for(let i=0;i<e.length;i++)r+=await this.submitTimed((t,r)=>{this.raster.recordForward(t,e[i],this.batchIO[i],r)},t);return r}async profileCachedBatchBackward(e,t){if(!t)return this.submitTimed(t=>this.recordCachedBatchBackward(t,e));if(this.viewLaneBatchRasterBackward&&this.batchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordBatchBackwardAdd(t,this.batchRasterForward,e,r)},t);let r=0;for(let i=0;i<e.length;i++)r+=await this.submitTimed((t,r)=>{this.raster.recordBackwardAdd(t,e[i],this.batchIO[i],r)},t);return r}async profileGrid9Close2Inputs(e,t,r){if(!r)return this.submitTimed(r=>this.recordGrid9Close2Inputs(r,e,t));let i=this.gridClip,a=this.device.createCommandEncoder();this.recordGrid9Close2TextCopies(a,t),i.clearGridImage(a),this.device.queue.submit([a.finish()]),await this.device.queue.onSubmittedWorkDone();let s=0;for(let t=0;t<9;t++)s+=await this.submitTimed((r,a)=>{i.raster.recordForward(r,e[t],i.scratchIO,a)},r)+await this.submitTimed((e,r)=>{i.recordCopyCell(e,t,r)},r);for(let e=0;e<2;e++)s+=await this.submitTimed((r,i)=>{this.raster.recordForward(r,t[e],this.batchIO[e+1],i)},r);return s}async profileGrid9Close2Backward(e,t,r){if(!r)return{replay:0,backward:await this.submitTimed(r=>this.recordGrid9Close2Backward(r,e,t))};let i=this.gridClip,a=0,s=0;for(let t=0;t<9;t++)s+=await this.submitTimed((e,r)=>{i.clearScratchGrad(e),i.recordScatterCell(e,t,r)},r),a+=await this.submitTimed((r,a)=>{i.raster.recordForward(r,e[t],i.scratchIO,a)},r),s+=await this.submitTimed((r,a)=>{i.raster.recordBackwardAdd(r,e[t],i.scratchIO,a)},r);for(let e=0;e<2;e++)s+=await this.submitTimed((r,i)=>{this.raster.recordBackwardAdd(r,t[e],this.batchIO[e+1],i)},r);return{replay:a,backward:s}}recordBatchTextCopies(e,t){let r=this.batchTrainer;for(let i=0;i<t.length;i++){let a=t[i];e.copyBufferToBuffer(this.textBuffers[a],0,r.textBuffer,r.textOffsetBytes(i),4*r.plan.textDim)}}recordSingleForwardToTrainer(e,t,r){this.raster.recordForward(e,t,this.singleIO,r)}recordSingleTextAndBackward(e,t,r){e.copyBufferToBuffer(this.textBuffers[t],0,this.trainer.textBuffer,0,4*this.trainer.plan.textDim),this.trainer.encodeBackward(e,r)}recordSingleRasterBackward(e,t,r){this.raster.recordBackwardAdd(e,t,this.singleIO,r)}recordConvergenceRegularizer(e,t){this.convergenceRegularizerEnabled()&&this.raster.recordRegularizerAdd(e,this.regularizerOptions(),t)}convergenceRegularizerEnabled(){return 0!==this.convergence.centerWeight||0!==this.convergence.radiusWeight||0!==this.convergence.opacitySparsity||0!==this.convergence.smallRadiusWeight||0!==this.convergence.radiusBandWeight}regularizerOptions(){return{centerWeight:this.convergence.centerWeight,radiusWeight:this.convergence.radiusWeight,targetRadius:this.convergence.targetRadius,opacitySparsity:this.convergence.opacitySparsity,smallRadiusWeight:this.convergence.smallRadiusWeight,smallRadius:this.convergence.smallRadius,radiusBandWeight:this.convergence.radiusBandWeight,minRadius:this.convergence.minRadius,maxRadius:this.convergence.maxRadius}}coverageOptions(){return{weight:this.convergence.coverageWeight,targetAlpha:this.convergence.coverageTarget}}applyTrainingBackground(){this.applyBackground(this.trainingBackground())}applyDisplayBackground(){this.applyBackground([0,0,0])}applyCoverageRegularizer(){let e=this.coverageOptions();this.raster.setCoverageRegularizer(e),this.gridClip&&this.gridClip.raster!==this.raster&&this.gridClip.raster.setCoverageRegularizer(e)}applyBackground(e){this.raster.setBackground(e),this.gridClip&&this.gridClip.raster!==this.raster&&this.gridClip.raster.setBackground(e)}trainingBackground(){let e=this.convergence.backgroundMode;if("black"===e)return[0,0,0];let t="curriculum"===e&&this.step_>=120&&this.step_%8==0,r=t?.28:.09,i=t?.02:0;return[i+r*v(this.step_,11),i+r*v(this.step_,29),i+r*v(this.step_,47)]}async submitTimed(e,t=null){if(t)return t.time(e);let r=this.device.createCommandEncoder();e(r);let i=performance.now();return this.device.queue.submit([r.finish()]),await this.device.queue.onSubmittedWorkDone(),performance.now()-i}sampleViews(e){let t=this.cameras.length,r=this.normalizedViewCount(e);if(r>=t)return Array.from({length:t},(e,t)=>t);if("random"===this.viewSampler)return this.sampleRandomViews(r);let i=[];for(;i.length<r;)this.viewCursor>=this.viewOrder.length&&this.shuffleViewOrder(),i.push(this.viewOrder[this.viewCursor]),this.viewCursor+=1;return i}shouldUseCachedBatchStep(e){return!(this.clipRefreshInterval<=1)&&!!this.cachedBatchViews&&this.step_%this.clipRefreshInterval!=0&&"per_view"===this.clipLayout&&!!this.batchTrainer&&this.normalizedViewCount(e)===this.cachedBatchViews.length}lrsForStep(e){var t,r;return e&&1!==this.cachedLrScale?(t=this.lrs,r=this.cachedLrScale,{position:t.position*r,logRadius:t.logRadius*r,color:t.color*r,opacity:t.opacity*r}):this.lrs}updateCachedBatchViews(e){if(this.clipRefreshInterval<=1||"per_view"!==this.clipLayout||!this.batchTrainer){this.cachedBatchViews=null;return}this.cachedBatchViews=e.length===this.batchTrainer.batch?e.slice():null}normalizedViewCount(e){let t=this.cameras.length;return Math.max(1,Math.min(t,0|e))}sampleRandomViews(e){let t=Array.from({length:this.cameras.length},(e,t)=>t);for(let r=0;r<e;r++){let e=r+this.nextRandomU32()%(t.length-r),i=t[r];t[r]=t[e],t[e]=i}return t.slice(0,e)}shuffleViewOrder(){this.viewOrder=Array.from({length:this.cameras.length},(e,t)=>t);for(let e=this.viewOrder.length-1;e>0;e--){let t=this.nextRandomU32()%(e+1),r=this.viewOrder[e];this.viewOrder[e]=this.viewOrder[t],this.viewOrder[t]=r}this.viewCursor=0}nextRandomU32(){return this.rngState=Math.imul(this.rngState,1664525)+1013904223>>>0,this.rngState}}function f(e){var t;let r=e?.backgroundMode==="dark_random"||e?.backgroundMode==="curriculum"?e.backgroundMode:"black";return{backgroundMode:r,centerWeight:m(e?.centerWeight,0),radiusWeight:m(e?.radiusWeight,0),targetRadius:w(e?.targetRadius,1.15),opacitySparsity:m(e?.opacitySparsity,0),coverageWeight:m(e?.coverageWeight,0),coverageTarget:void 0!==(t=e?.coverageTarget)&&Number.isFinite(t)?Math.max(0,Math.min(1,t)):.18,smallRadiusWeight:m(e?.smallRadiusWeight,0),smallRadius:w(e?.smallRadius,.022),radiusBandWeight:m(e?.radiusBandWeight,0),minRadius:w(e?.minRadius,.014),maxRadius:w(e?.maxRadius,.18)}}function m(e,t){return void 0!==e&&Number.isFinite(e)?Math.max(0,e):t}function w(e,t){return void 0!==e&&Number.isFinite(e)?Math.max(1e-4,e):t}function v(e,t){let r=Math.imul(e+1>>>0,747796405)+Math.imul(t>>>0,2891336453)>>>0;return(r=((r=Math.imul(r>>>(r>>>28)+4^r,277803737)>>>0)>>>22^r)>>>0)/4294967296}class b{static create(e){return e.features.has("timestamp-query")?new b(e):null}constructor(e){this.device=e,this.querySet=e.createQuerySet({type:"timestamp",count:2}),this.resolveBuffer=e.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),this.readBuffer=e.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})}async time(e){let t=this.device.createCommandEncoder();e(t,{querySet:this.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}),t.resolveQuerySet(this.querySet,0,2,this.resolveBuffer,0),t.copyBufferToBuffer(this.resolveBuffer,0,this.readBuffer,0,16),this.device.queue.submit([t.finish()]),await this.readBuffer.mapAsync(GPUMapMode.READ);let r=new BigUint64Array(this.readBuffer.getMappedRange().slice(0));return this.readBuffer.unmap(),Number(r[1]-r[0])/1e6}destroy(){this.querySet.destroy(),this.resolveBuffer.destroy(),this.readBuffer.destroy()}}function B(e,t=1,r={}){let i=r.radius??p.radius,a=r.radiusJitter??p.radiusJitter,s=r.opacityRaw??p.opacityRaw,n=r.colorSpread??p.colorSpread,o=r.positionSpread??p.positionSpread,d=t>>>0||1,l=()=>{let e=Math.imul((d=Math.imul(d,747796405)+2891336453>>>0)>>>(d>>>28)+4^d,277803737)>>>0;return(e=(e>>>22^e)>>>0)/4294967296},c=()=>{let e=0,t=0;for(;0===e;)e=l();for(;0===t;)t=l();return Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*t)},h=new Float32Array(e*u.PARAM_STRIDE_3D),g=3*e,f=4*e,m=7*e,w=Math.log(i);for(let t=0;t<e;t++)h[0+3*t+0]=(2*l()-1)*o,h[0+3*t+1]=(2*l()-1)*o,h[0+3*t+2]=(2*l()-1)*o,h[g+t]=w+a*c(),h[f+3*t+0]=n*c(),h[f+3*t+1]=n*c(),h[f+3*t+2]=n*c(),h[m+t]=s;return h}function y(e,t){let r=0,i=0,a=0;for(let s=0;s<e.length;s++)r+=e[s]*t[s],i+=e[s]*e[s],a+=t[s]*t[s];return r/Math.sqrt(i*a||1)}async function x(e,t,r){let i=e.createBuffer({size:4*r,usage:9}),a=e.createCommandEncoder();a.copyBufferToBuffer(t,0,i,0,4*r),e.queue.submit([a.finish()]),await i.mapAsync(1);let s=new Float32Array(i.getMappedRange().slice(0));return i.unmap(),i.destroy(),s}},{"../clip/vision":"lNzsi","../clip/vision_batch":"bWgJv","../splat/adam_wgsl":"kfWkJ","./cameras":"3Dl1Z","./grid_clip":"7V9cb","./raster":"AoYYi","./raster_wgsl":"fuyeU","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],bWgJv:[function(e,t,r){/**
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
 *//// <reference types="@webgpu/types" />
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),/**
 * Shared-weights, replicated-activation CLIP trainer for batch experiments.
 * Each lane has independent activation/grad slots and text embedding buffer.
 */i.export(r,"ReplicatedBatchVisionTrainer",()=>h),/**
 * True batch-major forward encoder. Activation slots are one buffer per logical
 * slot, sized `batch * slotFloats`, and each generated shader runs all lanes by
 * dispatching z workgroups. This is the forward-only proof before batch
 * backward and shared-W pointwise kernels.
 */i.export(r,"BatchMajorVisionEncoder",()=>p),/**
 * True batch-major forward+backward trainer. This is the gradient-producing
 * counterpart to BatchMajorVisionEncoder and mirrors VisionTrainer's public
 * buffers, except each buffer is laid out as `[batch][slotFloats]`.
 */i.export(r,"BatchMajorVisionTrainer",()=>g);var a=e("./vision_wgsl"),s=e("./vision_bwd_wgsl"),n=e("./vision_batch_wgsl");let o={COPY_SRC:4,COPY_DST:8,STORAGE:128};function d(e,t){return t?e.beginComputePass({timestampWrites:t}):e.beginComputePass()}async function l(e,t){let r=[];for(let i of t){e.pushErrorScope("validation");let t=e.createShaderModule({code:i.code}),a=e.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw Error(`vision_batch: pipeline '${i.label}' invalid: ${s.message}
${i.code}`);r.push({spec:i,pipeline:a})}return r}function u(e,t){if(!Number.isInteger(e)||e<0||e>=t)throw Error(`vision_batch: lane ${e} outside [0, ${t})`)}function c(e,t){if(e.length!==t)throw Error(`vision_batch: weights blob ${e.length} scalars != plan ${t}`)}class h{static async create(e,t,r,i,n={}){if(!Number.isInteger(i)||i<1)throw Error(`vision_batch: invalid batch ${i}`);c(r,t.weightsFloats);let o=(0,a.planDispatches)(t,n),d=(0,s.planBwdDispatches)(t,n),u=await l(e,[...o,...d]);return new h(e,t,r,i,u,o.length)}constructor(e,t,r,i,a,s){this.device=e,this.plan=t,this.batch=i,this.fwdCount=s,this.weightsBuffer=e.createBuffer({label:"clip-batch-weights",size:r.byteLength,usage:o.STORAGE|o.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,r),this.slotBuffers=Array.from({length:i},(r,i)=>t.slots.map((t,r)=>e.createBuffer({label:`clip-batch-lane-${i}-slot-${r}`,size:4*t,usage:o.STORAGE|o.COPY_DST|o.COPY_SRC}))),this.textBuffers=Array.from({length:i},(r,i)=>e.createBuffer({label:`clip-batch-lane-${i}-text`,size:4*t.textDim,usage:o.STORAGE|o.COPY_DST}));let n=(e,t)=>"weights"===t.kind?this.weightsBuffer:"text"===t.kind?this.textBuffers[e]:this.slotBuffers[e][t.slot];this.dispatches=a.map(({spec:t,pipeline:r})=>({pipeline:r,workgroups:t.workgroups,label:t.label,binds:this.slotBuffers.map((i,a)=>e.createBindGroup({layout:r.getBindGroupLayout(0),entries:t.buffers.map((e,t)=>({binding:t,resource:{buffer:n(a,e)}}))}))}))}inputBuffer(e){return u(e,this.batch),this.slotBuffers[e][this.plan.inputSlot]}outputBuffer(e){return u(e,this.batch),this.slotBuffers[e][this.plan.outputSlot]}inputGradBuffer(e){return u(e,this.batch),this.slotBuffers[e][this.plan.inputGradSlot]}writeInput(e,t){u(e,this.batch);let[r,i,a]=this.plan.inputShape;if(t.length!==r*i*a)throw Error(`vision_batch: input ${t.length} != ${r*i*a}`);this.device.queue.writeBuffer(this.inputBuffer(e),0,t)}writeText(e,t){if(u(e,this.batch),t.length!==this.plan.textDim)throw Error(`vision_batch: text ${t.length} != ${this.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffers[e],0,t)}encodeLane(e,t,r={}){u(t,this.batch);let i=!1===r.backward?this.fwdCount:this.dispatches.length,a=d(e,r.timestampWrites);for(let e=0;e<i;e++){let r=this.dispatches[e];a.setPipeline(r.pipeline),a.setBindGroup(0,r.binds[t]),a.dispatchWorkgroups(...r.workgroups)}a.end()}encode(e,t={}){let r=!1===t.backward?this.fwdCount:this.dispatches.length,i=t.schedule??"step-major",a=d(e,t.timestampWrites);if("lane-major"===i)for(let e=0;e<this.batch;e++)for(let t=0;t<r;t++){let r=this.dispatches[t];a.setPipeline(r.pipeline),a.setBindGroup(0,r.binds[e]),a.dispatchWorkgroups(...r.workgroups)}else for(let e=0;e<r;e++){let t=this.dispatches[e];a.setPipeline(t.pipeline);for(let e=0;e<this.batch;e++)a.setBindGroup(0,t.binds[e]),a.dispatchWorkgroups(...t.workgroups)}a.end()}runLane(e,t={}){let r=this.device.createCommandEncoder();this.encodeLane(r,e,t),this.device.queue.submit([r.finish()])}run(e={}){let t=this.device.createCommandEncoder();this.encode(t,e),this.device.queue.submit([t.finish()])}destroy(){for(let e of(this.weightsBuffer.destroy(),this.slotBuffers))for(let t of e)t.destroy();for(let e of this.textBuffers)e.destroy()}}class p{static async create(e,t,r,i,a={}){if(!Number.isInteger(i)||i<1)throw Error(`vision_batch: invalid batch ${i}`);c(r,t.weightsFloats);let s=await l(e,(0,n.batchForwardDispatches)(t,i,a));return new p(e,t,r,i,s)}constructor(e,t,r,i,a){this.device=e,this.plan=t,this.batch=i,this.weightsBuffer=e.createBuffer({label:"clip-batch-major-weights",size:r.byteLength,usage:o.STORAGE|o.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,r),this.slotBuffers=t.slots.map((t,r)=>e.createBuffer({label:`clip-batch-major-slot-${r}`,size:t*i*4,usage:o.STORAGE|o.COPY_DST|o.COPY_SRC}));let s=e=>{if("weights"===e.kind)return this.weightsBuffer;if("slot"===e.kind)return this.slotBuffers[e.slot];throw Error("vision_batch: batch-major forward received a text binding")};this.dispatches=a.map(({spec:t,pipeline:r})=>({pipeline:r,workgroups:t.workgroups,label:t.label,bind:e.createBindGroup({layout:r.getBindGroupLayout(0),entries:t.buffers.map((e,t)=>({binding:t,resource:{buffer:s(e)}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}slotOffsetBytes(e,t){return u(e,this.batch),e*this.plan.slots[t]*4}outputOffsetBytes(e){return this.slotOffsetBytes(e,this.plan.outputSlot)}writeInput(e,t){u(e,this.batch);let[r,i,a]=this.plan.inputShape;if(t.length!==r*i*a)throw Error(`vision_batch: input ${t.length} != ${r*i*a}`);this.device.queue.writeBuffer(this.inputBuffer,this.slotOffsetBytes(e,this.plan.inputSlot),t)}encode(e,t){let r=d(e,t);for(let e of this.dispatches)r.setPipeline(e.pipeline),r.setBindGroup(0,e.bind),r.dispatchWorkgroups(...e.workgroups);r.end()}run(){let e=this.device.createCommandEncoder();this.encode(e),this.device.queue.submit([e.finish()])}destroy(){for(let e of(this.weightsBuffer.destroy(),this.slotBuffers))e.destroy()}}class g{static async create(e,t,r,i,a={}){if(!Number.isInteger(i)||i<1)throw Error(`vision_batch: invalid batch ${i}`);c(r,t.weightsFloats);let{specs:s,fwdCount:o}=(0,n.batchTrainDispatches)(t,i,a),d=await l(e,s);return new g(e,t,r,i,d,o)}constructor(e,t,r,i,a,s){this.device=e,this.plan=t,this.batch=i,this.fwdCount=s,this.weightsBuffer=e.createBuffer({label:"clip-batch-major-train-weights",size:r.byteLength,usage:o.STORAGE|o.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,r),this.textBuffer=e.createBuffer({label:"clip-batch-major-text",size:t.textDim*i*4,usage:o.STORAGE|o.COPY_DST}),this.slotBuffers=t.slots.map((t,r)=>e.createBuffer({label:`clip-batch-major-train-slot-${r}`,size:t*i*4,usage:o.STORAGE|o.COPY_DST|o.COPY_SRC}));let n=e=>"weights"===e.kind?this.weightsBuffer:"text"===e.kind?this.textBuffer:this.slotBuffers[e.slot];this.dispatches=a.map(({spec:t,pipeline:r})=>({pipeline:r,workgroups:t.workgroups,label:t.label,bind:e.createBindGroup({layout:r.getBindGroupLayout(0),entries:t.buffers.map((e,t)=>({binding:t,resource:{buffer:n(e)}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}get inputGradBuffer(){return this.slotBuffers[this.plan.inputGradSlot]}slotOffsetBytes(e,t){return u(e,this.batch),e*this.plan.slots[t]*4}outputOffsetBytes(e){return this.slotOffsetBytes(e,this.plan.outputSlot)}inputGradOffsetBytes(e){return this.slotOffsetBytes(e,this.plan.inputGradSlot)}textOffsetBytes(e){return u(e,this.batch),e*this.plan.textDim*4}writeInput(e,t){u(e,this.batch);let[r,i,a]=this.plan.inputShape;if(t.length!==r*i*a)throw Error(`vision_batch: input ${t.length} != ${r*i*a}`);this.device.queue.writeBuffer(this.inputBuffer,this.slotOffsetBytes(e,this.plan.inputSlot),t)}writeText(e,t){if(u(e,this.batch),t.length!==this.plan.textDim)throw Error(`vision_batch: text ${t.length} != ${this.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffer,this.textOffsetBytes(e),t)}encode(e,t={}){let r=!1===t.backward?this.fwdCount:this.dispatches.length,i=d(e,t.timestampWrites);for(let e=0;e<r;e++){let t=this.dispatches[e];i.setPipeline(t.pipeline),i.setBindGroup(0,t.bind),i.dispatchWorkgroups(...t.workgroups)}i.end()}run(e={}){let t=this.device.createCommandEncoder();this.encode(t,e),this.device.queue.submit([t.finish()])}destroy(){for(let e of(this.weightsBuffer.destroy(),this.textBuffer.destroy(),this.slotBuffers))e.destroy()}}},{"./vision_wgsl":"oFDUc","./vision_bwd_wgsl":"2Oqph","./vision_batch_wgsl":"eNMEN","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],eNMEN:[function(e,t,r){/**
 * vision_batch_wgsl — batch-major forward fork for the fused CLIP encoder.
 *
 * Iteration 2 of the batching work keeps the proven per-step kernels and adds
 * a real batch dimension by dispatching workgroups in z. Every activation slot
 * is allocated as `[batch][slotFloats]`; each shader indexes lane
 * `workgroup_id.z` by adding a per-binding base offset. Weights remain shared.
 *
 * This is deliberately forward-only. Backward needs the same treatment once
 * the forward batch lane proves useful.
 *//// <reference types="@webgpu/types" />
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"batchForwardDispatches",()=>c),i.export(r,"batchTrainDispatches",()=>h);var a=e("./vision_wgsl"),s=e("./vision_bwd_wgsl"),n=e("./vision_batch_pointwise");function o(e){let t=e.indexOf("fn main(");if(t<0)throw Error("vision_batch_wgsl: missing fn main");let r=e.indexOf("{",t);if(r<0)throw Error("vision_batch_wgsl: missing main body");return{start:t,openBrace:r,signature:e.slice(t,r)}}function d(e,t){let r=function(e,t){let r=[];for(let i=0;i<t.buffers.length;i++){let a=t.buffers[i];if("slot"!==a.kind&&"text"!==a.kind)continue;let s=RegExp(`@group\\(0\\)\\s*@binding\\(${i}\\)\\s*var<storage,[^>]+>\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*:\\s*array<([^>]+)>`),n=t.code.match(s);if(!n)throw Error(`vision_batch_wgsl: could not find slot binding ${i} in ${t.label}`);let o=n[2].trim();if("f32"!==o&&"vec4f"!==o)throw Error(`vision_batch_wgsl: unsupported array<${o}> in ${t.label}`);let d="slot"===a.kind?e.slots[a.slot]:e.textDim;if(!Number.isFinite(d))throw Error(`vision_batch_wgsl: text binding in ${t.label} needs a TrainPlan`);if("vec4f"===o&&d%4!=0)throw Error(`vision_batch_wgsl: ${t.label} binding ${i} has ${d} floats, not vec4-aligned`);r.push({name:n[1],elem:o,strideFloats:d})}return r}(e,t);if(0===r.length)return t.code;let i=function(e){let t=o(e),r=t.signature.match(/@builtin\(workgroup_id\)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*vec3u/);if(r)return{code:e,batchExpr:`${r[1]}.z`};let i=t.signature.replace(/\)\s*$/,",\n        @builtin(workgroup_id) batchWid : vec3u)");return{code:e.slice(0,t.start)+i+e.slice(t.openBrace),batchExpr:"batchWid.z"}}(t.code),a=o(i.code),s=new Map;for(let e of r){if(s.has(e.name))throw Error(`vision_batch_wgsl: duplicate slot variable '${e.name}' in ${t.label}`);s.set(e.name,e)}let n=[`  let batchLane = ${i.batchExpr};`,...r.map(e=>{let t="vec4f"===e.elem?e.strideFloats/4:e.strideFloats;return`  let batchBase_${e.name} = batchLane * ${t}u;`})],d=i.code.slice(0,a.openBrace+1)+"\n"+n.join("\n")+i.code.slice(a.openBrace+1);for(let e of r){let t=RegExp(`\\b${e.name}\\[`,"g");d=d.replace(t,`${e.name}[batchBase_${e.name} + `)}return d}function l(e,t,r){let i=[];for(let s=0;s<e.steps.length;s++){let o=e.steps[s];if(r.sharedWForwardSteps?.has(s)&&"conv"===o.kind&&"pointwise"===o.variant){i.push((0,n.pointwiseSharedWBatchForwardDispatch)(e,o,t,r.weightPrecision));continue}let l=e.steps[s+1];if(r.fusePointwiseGeluForward&&"conv"===o.kind&&"pointwise"===o.variant&&l?.kind==="gelu"&&l.src===o.dst){i.push(u(e,(0,a.pointwiseFusedGelu)(o,l,r,s),t)),s+=1;continue}for(let n of(0,a.stepDispatches)(o,r,s)){if(1!==n.workgroups[2])throw Error(`vision_batch_wgsl: ${n.label} already uses workgroup z=${n.workgroups[2]}`);i.push({...n,code:d(e,n),workgroups:[n.workgroups[0],n.workgroups[1],t]})}}return i}function u(e,t,r){if(1!==t.workgroups[2])throw Error(`vision_batch_wgsl: ${t.label} already uses workgroup z=${t.workgroups[2]}`);return{...t,code:d(e,t),workgroups:[t.workgroups[0],t.workgroups[1],r]}}function c(e,t,r={}){if(!Number.isInteger(t)||t<1)throw Error(`vision_batch_wgsl: invalid batch ${t}`);return r.sharedWForwardSteps?.size||r.fusePointwiseGeluForward?l(e,t,r):(0,a.planDispatches)(e,r).map(r=>u(e,r,t))}function h(e,t,r={}){if(!Number.isInteger(t)||t<1)throw Error(`vision_batch_wgsl: invalid batch ${t}`);let i=r.sharedWForwardSteps?.size||r.fusePointwiseGeluForward?l(e,t,r):(0,a.planDispatches)(e,r).map(r=>u(e,r,t)),n=(0,s.planBwdDispatches)(e,r).map(r=>u(e,r,t)),o=[...i,...n];return{specs:o,fwdCount:i.length}}},{"./vision_wgsl":"oFDUc","./vision_bwd_wgsl":"2Oqph","./vision_batch_pointwise":"92gmd","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],"92gmd":[function(e,t,r){/**
 * vision_batch_pointwise — isolated shared-W pointwise batch experiments.
 *
 * The batch-major CLIP fork runs one workgroup per image lane, so each lane
 * reloads the same 32x32 W tile. This microkernel instead puts batch lanes in
 * local_invocation_id.z: one workgroup stages W once, while each lane stages
 * its own X tile. B=2 and B=3 fit the 16 KB workgroup-memory limit.
 *//// <reference types="@webgpu/types" />
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),/** Baseline z-batch pointwise: same math as the normal pointwise kernel, but
 * buffers are compact `[batch][tensor]` instead of full CLIP slot buffers. */i.export(r,"pointwiseZBatchDispatch",()=>d),/** Shared-W batch pointwise: B lanes inside the same workgroup, one W tile. */i.export(r,"pointwiseSharedWBatchDispatch",()=>l),/** Production shared-W forward pointwise: real CLIP slots laid out as
 * `[batch][slotFloats]`, with batch lanes inside one workgroup. */i.export(r,"pointwiseSharedWBatchForwardDispatch",()=>u);var a=e("./vision_wgsl");function s(e){let t=[{kind:"weights"},{kind:"slot",slot:0},{kind:"slot",slot:1}];return e&&t.push({kind:"slot",slot:2}),t}function n(e,t,r){let i=e.slots[t];if(i%4!=0)throw Error(`pointwise_shared_w: ${r} slot ${t} has non-vec4 stride ${i}`);return i/4}function o(e,t,r,i,a=!0){let s="gelu"===e.act?`gelu4(${i})`:i;if(null===e.residual)return s;let n=a?`res[resBase + (co + ${r}u) * ${t}u + p4]`:`res[(co + ${r}u) * ${t}u + p4]`;return`${n} + vec4f(W(${e.layerScaleOff}u + co + ${r}u)) * ${s}`}function d(e,t,r="f32"){if(!Number.isInteger(t)||t<1)throw Error(`pointwise_zbatch: invalid batch ${t}`);let i=e.outH*e.outW;(0,a.assertPointwiseTiles)(e.name,e.cin,e.cout,i,e.wOff);let n=i/4,d=e.cin*n,l=e.cout*n,u=null!==e.residual,c=/* wgsl */`
${(0,a.weightsDecl)(0,r)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${u?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${a.GELU}
var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 256>;
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let lane = wid.z;
  let srcBase = lane * ${d}u;
  let dstBase = lane * ${l}u;
  let resBase = lane * ${l}u;
  let p4 = wid.x * 8u + lid.x;
  let co = (wid.y * 8u + lid.y) * 4u;
  let p4base = wid.x * 8u;
  let cobase = wid.y * 32u;
  var acc0 = vec4f(W(${e.bOff}u + co));
  var acc1 = vec4f(W(${e.bOff}u + co + 1u));
  var acc2 = vec4f(W(${e.bOff}u + co + 2u));
  var acc3 = vec4f(W(${e.bOff}u + co + 3u));
  for (var ci0 = 0u; ci0 < ${e.cin}u; ci0 = ci0 + 32u) {
    for (var t = li; t < 256u; t = t + 64u) {
      let ci = t >> 3u;
      let px = t & 7u;
      xS[t] = src[srcBase + (ci0 + ci) * ${n}u + p4base + px];
      wS[t] = W4((${e.wOff}u + (ci0 + ci) * ${e.cout}u + cobase + px * 4u) / 4u);
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
  dst[dstBase + co * ${n}u + p4] = ${o(e,n,0,"acc0")};
  dst[dstBase + (co + 1u) * ${n}u + p4] = ${o(e,n,1,"acc1")};
  dst[dstBase + (co + 2u) * ${n}u + p4] = ${o(e,n,2,"acc2")};
  dst[dstBase + (co + 3u) * ${n}u + p4] = ${o(e,n,3,"acc3")};
}`;return{label:`pw-zbatch B${t} ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:c,workgroups:[n/8,e.cout/32,t],buffers:s(u)}}function l(e,t,r="f32"){if(!Number.isInteger(t)||t<1||t>3)throw Error(`pointwise_shared_w: batch ${t} outside [1, 3]`);let i=e.outH*e.outW;(0,a.assertPointwiseTiles)(e.name,e.cin,e.cout,i,e.wOff);let n=i/4,d=e.cin*n,l=e.cout*n,u=null!==e.residual,c=/* wgsl */`
${(0,a.weightsDecl)(0,r)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${u?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${a.GELU}
var<workgroup> xS : array<vec4f, ${256*t}>;
var<workgroup> wS : array<vec4f, 256>;
@compute @workgroup_size(8, 8, ${t})
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u) {
  let lane = lid.z;
  let li = lid.y * 8u + lid.x;
  let xTile = lane * 256u;
  let srcBase = lane * ${d}u;
  let dstBase = lane * ${l}u;
  let resBase = lane * ${l}u;
  let p4 = wid.x * 8u + lid.x;
  let co = (wid.y * 8u + lid.y) * 4u;
  let p4base = wid.x * 8u;
  let cobase = wid.y * 32u;
  var acc0 = vec4f(W(${e.bOff}u + co));
  var acc1 = vec4f(W(${e.bOff}u + co + 1u));
  var acc2 = vec4f(W(${e.bOff}u + co + 2u));
  var acc3 = vec4f(W(${e.bOff}u + co + 3u));
  for (var ci0 = 0u; ci0 < ${e.cin}u; ci0 = ci0 + 32u) {
    for (var t = li; t < 256u; t = t + 64u) {
      let ci = t >> 3u;
      let px = t & 7u;
      xS[xTile + t] = src[srcBase + (ci0 + ci) * ${n}u + p4base + px];
      if (lane == 0u) {
        wS[t] = W4((${e.wOff}u + (ci0 + ci) * ${e.cout}u + cobase + px * 4u) / 4u);
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
  dst[dstBase + co * ${n}u + p4] = ${o(e,n,0,"acc0")};
  dst[dstBase + (co + 1u) * ${n}u + p4] = ${o(e,n,1,"acc1")};
  dst[dstBase + (co + 2u) * ${n}u + p4] = ${o(e,n,2,"acc2")};
  dst[dstBase + (co + 3u) * ${n}u + p4] = ${o(e,n,3,"acc3")};
}`;return{label:`pw-shared-w B${t} ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:c,workgroups:[n/8,e.cout/32,1],buffers:s(u)}}function u(e,t,r,i="f32"){if(!Number.isInteger(r)||r<1||r>3)throw Error(`pointwise_shared_w_forward: batch ${r} outside [1, 3]`);let s=t.outH*t.outW;(0,a.assertPointwiseTiles)(t.name,t.cin,t.cout,s,t.wOff);let d=s/4,l=n(e,t.src,"src"),u=n(e,t.dst,"dst"),c=null!==t.residual,h=c?n(e,t.residual,"residual"):u,p=/* wgsl */`
${(0,a.weightsDecl)(0,i)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${c?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${a.GELU}
var<workgroup> xS : array<vec4f, ${256*r}>;
var<workgroup> wS : array<vec4f, 256>;
@compute @workgroup_size(8, 8, ${r})
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u) {
  let lane = lid.z;
  let li = lid.y * 8u + lid.x;
  let xTile = lane * 256u;
  let srcBase = lane * ${l}u;
  let dstBase = lane * ${u}u;
  let resBase = lane * ${h}u;
  let p4 = wid.x * 8u + lid.x;
  let co = (wid.y * 8u + lid.y) * 4u;
  let p4base = wid.x * 8u;
  let cobase = wid.y * 32u;
  var acc0 = vec4f(W(${t.bOff}u + co));
  var acc1 = vec4f(W(${t.bOff}u + co + 1u));
  var acc2 = vec4f(W(${t.bOff}u + co + 2u));
  var acc3 = vec4f(W(${t.bOff}u + co + 3u));
  for (var ci0 = 0u; ci0 < ${t.cin}u; ci0 = ci0 + 32u) {
    for (var t = li; t < 256u; t = t + 64u) {
      let ci = t >> 3u;
      let px = t & 7u;
      xS[xTile + t] = src[srcBase + (ci0 + ci) * ${d}u + p4base + px];
      if (lane == 0u) {
        wS[t] = W4((${t.wOff}u + (ci0 + ci) * ${t.cout}u + cobase + px * 4u) / 4u);
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
  dst[dstBase + co * ${d}u + p4] = ${o(t,d,0,"acc0")};
  dst[dstBase + (co + 1u) * ${d}u + p4] = ${o(t,d,1,"acc1")};
  dst[dstBase + (co + 2u) * ${d}u + p4] = ${o(t,d,2,"acc2")};
  dst[dstBase + (co + 3u) * ${d}u + p4] = ${o(t,d,3,"acc3")};
}`;return{label:`pw-shared-w-fwd B${r} ${t.cin}->${t.cout} @${t.outH}x${t.outW}`,code:p,workgroups:[d/8,t.cout/32,1],buffers:function(e){let t=[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}];return null!==e.residual&&t.push({kind:"slot",slot:e.residual}),t}(t)}}},{"./vision_wgsl":"oFDUc","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],"7V9cb":[function(e,t,r){/// <reference types="@webgpu/types" />
var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"Grid9Close2ClipLayout",()=>u);var a=e("./cameras"),s=e("./raster");let n={COPY_SRC:4,COPY_DST:8,STORAGE:128};async function o(e,t,r){e.pushErrorScope("validation");let i=e.createShaderModule({code:t}),a=e.createComputePipeline({layout:"auto",compute:{module:i,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw console.error(`--- WGSL that failed (${r}) ---
${t}`),Error(`grid9_close2 pipeline validation (${r}): ${s.message}`);return a}function d(e,t){return t?e.beginComputePass({timestampWrites:t}):e.beginComputePass()}function l(e,t,r){let{x:i,y:a}={x:e%3*88,y:88*Math.floor(e/3)},s=r*r,n="downsample"===t?"src":"gridGrad",o="downsample"===t?"gridImage":"dst",d="downsample"===t?`${o}[ch * 65536u + dstPix] = ${n}[ch * ${s}u + srcPix];`:`${o}[ch * ${s}u + srcPix] = ${n}[ch * 65536u + dstPix];`,l=80===r?"let srcPix = cy * 80u + cx;":/* wgsl */`
  let srcX = min(255u, (cx * 256u + 128u) / 80u);
  let srcY = min(255u, (cy * 256u + 128u) / 80u);
  let srcPix = srcY * 256u + srcX;`;return/* wgsl */`
@group(0) @binding(0) var<storage, read> ${n} : array<f32>;
@group(0) @binding(1) var<storage, read_write> ${o} : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= 19200u) { return; }
  let cellPix = i % 6400u;
  let ch = i / 6400u;
  let cx = cellPix % 80u;
  let cy = cellPix / 80u;
${l}
  let dstPix = (${a}u + cy) * 256u + (${i}u + cx);
  ${d}
}`}class u{constructor(e,t,r,i,a,s,n,o){this.device=e,this.gridImageBuffer=a,this.gridImageOffset=s,this.raster=t,this.scratchImage=r,this.scratchGrad=i,this.cells=n,this.directRaster=o,this.scratchImageBytes=o?76800:786432,this.scratchIO=t.createIOState(r,0,i,0,{privateState:!0})}static async create(e,t,r,i={}){if(r.batch<3)throw Error(`grid9_close2: needs CLIP batch >= 3, got ${r.batch}`);let d=!!i.directRaster,c=d?await (0,s.Raster3DEngine).create(e,{H:80,W:80,G:t.dims.G,cap:t.dims.cap,bg:t.dims.bg,dynamicBg:t.dims.dynamicBg,dynamicCoverage:t.dims.dynamicCoverage,near:t.dims.near,far:t.dims.far,gradScale:t.dims.gradScale,cameras:t.cameras.map(e=>(0,a.prepareCamera)(e,80)),sharedParams:t.params,sharedGradRaw:t.gradRaw}):t,h=d?76800:786432,p=e.createBuffer({label:"grid9-close2-scratch-image",size:h,usage:n.STORAGE|n.COPY_SRC|n.COPY_DST}),g=e.createBuffer({label:"grid9-close2-scratch-grad",size:h,usage:n.STORAGE|n.COPY_SRC|n.COPY_DST}),f=r.slotOffsetBytes(0,r.plan.inputSlot),m=r.inputGradOffsetBytes(0),w=[];for(let t=0;t<9;t++){let i=await o(e,l(t,"downsample",d?80:256),`grid-copy-${t}`),a=await o(e,l(t,"scatter",d?80:256),`grid-scatter-${t}`),s=e.createBindGroup({layout:i.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:p,offset:0,size:h}},{binding:1,resource:{buffer:r.inputBuffer,offset:f,size:786432}}]}),n=e.createBindGroup({layout:a.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r.inputGradBuffer,offset:m,size:786432}},{binding:1,resource:{buffer:g,offset:0,size:h}}]});w.push({copyPipe:i,copyBind:s,scatterPipe:a,scatterBind:n})}return new u(e,c,p,g,r.inputBuffer,f,w,d)}clearGridImage(e){e.clearBuffer(this.gridImageBuffer,this.gridImageOffset,786432)}clearScratchGrad(e){e.clearBuffer(this.scratchGrad,0,this.scratchImageBytes)}recordCopyCell(e,t,r){let i=this.cell(t),a=d(e,r);a.setPipeline(i.copyPipe),a.setBindGroup(0,i.copyBind),a.dispatchWorkgroups(Math.ceil(75)),a.end()}recordScatterCell(e,t,r){let i=this.cell(t),a=d(e,r);a.setPipeline(i.scatterPipe),a.setBindGroup(0,i.scatterBind),a.dispatchWorkgroups(Math.ceil(75)),a.end()}destroy(){this.scratchImage.destroy(),this.scratchGrad.destroy(),this.directRaster&&this.raster.destroy()}cell(e){let t=this.cells[0|e];if(!t)throw Error(`grid9_close2: bad cell ${e}`);return t}}},{"./cameras":"3Dl1Z","./raster":"AoYYi","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],AoYYi:[function(e,t,r){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"DEFAULT_3D_LRS",()=>d),i.export(r,"Raster3DEngine",()=>c);var a=e("../splat/adam_wgsl"),s=e("./raster_wgsl");let n={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},o=e=>Math.ceil(e/256),d={position:.025,logRadius:.01,color:.08,opacity:.03};async function l(e,t,r){e.pushErrorScope("validation");let i=e.createShaderModule({code:t}),a=e.createComputePipeline({layout:"auto",compute:{module:i,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw console.error(`--- WGSL that failed (${r}) ---
${t}`),Error(`raster3d pipeline validation (${r}): ${s.message}`);return a}function u(e,t){return t?e.beginComputePass({timestampWrites:t}):e.beginComputePass()}class c{constructor(e,t){if(this.ownsParams=!0,this.ownsGradRaw=!0,this.prepPipe=[],this.chainPipe=[],this.bgUni=null,this.coverageUni=null,this.prepBind=[],this.chainBind=[],this.adamUni=[],this.adamBind=[],this.extraBuffers=[],this.device=e,this.dims=(0,s.resolveDims3D)(t),this.cameras=t.cameras,!this.cameras.length)throw Error("raster3d: at least one camera is required")}static async create(e,t){let r=new c(e,t);return await r.build(t),r}storage(e,t=0){return this.device.createBuffer({size:4*e,usage:n.STORAGE|t})}bindGroup(e,t){return this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:t.map((e,t)=>({binding:t,resource:"buffer"in e?e:{buffer:e}}))})}async build(e){let t=this.dims,r=t.G*s.PARAM_STRIDE_3D,i=t.G*s.DERIVED_STRIDE_3D;this.params=e.sharedParams??this.storage(r,n.COPY_SRC|n.COPY_DST),this.ownsParams=!e.sharedParams,this.derived=this.storage(i),this.accGrad=this.storage(i,n.COPY_DST),this.gradRaw=e.sharedGradRaw??this.storage(r,n.COPY_SRC|n.COPY_DST),this.ownsGradRaw=!e.sharedGradRaw,this.mBuf=this.storage(r,n.COPY_DST),this.vBuf=this.storage(r,n.COPY_DST),this.tileCounts=this.storage(t.numTiles,n.COPY_DST|n.COPY_SRC),this.binnedIds=this.storage(t.numTiles*t.cap),this.tileStop=this.storage(t.numTiles,n.COPY_SRC),this.image=this.storage(3*t.H*t.W,n.COPY_SRC),this.gradImage=this.storage(3*t.H*t.W,n.COPY_DST),this.cameraBuffer=this.device.createBuffer({label:"splat3d-cameras",size:this.cameras.length*s.CAMERA_STRIDE_3D*4,usage:n.STORAGE|n.COPY_DST}),this.device.queue.writeBuffer(this.cameraBuffer,0,function(e){let t=new Float32Array(e.length*s.CAMERA_STRIDE_3D);for(let r=0;r<e.length;r++){let i=e[r],a=r*s.CAMERA_STRIDE_3D;t[a+0]=i.eye[0],t[a+1]=i.eye[1],t[a+2]=i.eye[2],t[a+3]=i.right[0],t[a+4]=i.right[1],t[a+5]=i.right[2],t[a+6]=i.cameraUp[0],t[a+7]=i.cameraUp[1],t[a+8]=i.cameraUp[2],t[a+9]=i.forward[0],t[a+10]=i.forward[1],t[a+11]=i.forward[2],t[a+12]=i.focalPx}return t}(this.cameras)),t.dynamicBg&&(this.bgUni=this.device.createBuffer({label:"splat3d-background",size:16,usage:n.UNIFORM|n.COPY_DST}),this.setBackground(t.bg)),t.dynamicCoverage&&(this.coverageUni=this.device.createBuffer({label:"splat3d-coverage",size:s.COVERAGE_UNIFORM_BYTES_3D,usage:n.UNIFORM|n.COPY_DST}),this.setCoverageRegularizer({weight:0,targetAlpha:.18})),this.prepPipe=await Promise.all(this.cameras.map((t,r)=>l(this.device,(0,s.prepShader3D)(e,t),`prep-${r}`))),this.chainPipe=await Promise.all(this.cameras.map((t,r)=>l(this.device,(0,s.chainAddShader3D)(e,t),`chain-${r}`))),this.emitPipe=await l(this.device,(0,s.emitShader3D)(e),"emit"),this.fwdPipe=await l(this.device,(0,s.forwardShader3D)(e),"forward"),this.bwdPipe=await l(this.device,(0,s.backwardShader3D)(e),"backward"),this.clearBinsPipe=await l(this.device,(0,s.clearShader3D)(t.numTiles),"clearBins"),this.clearGradsPipe=await l(this.device,(0,s.clearShader3D)(i),"clearGrads"),this.clearRawPipe=await l(this.device,(0,s.clearShader3D)(r),"clearRawGrad"),this.adamPipe=await l(this.device,(0,a.adamShader)(),"adam"),this.regularizerPipe=await l(this.device,(0,s.regularizerShader3D)(e),"regularizer"),this.prepBind=this.prepPipe.map(e=>this.bindGroup(e,[this.params,this.derived])),this.chainBind=this.chainPipe.map(e=>this.bindGroup(e,[this.accGrad,this.derived,this.params,this.gradRaw])),this.emitBind=this.bindGroup(this.emitPipe,[this.derived,this.tileCounts,this.binnedIds]),this.clearBinsBind=this.bindGroup(this.clearBinsPipe,[this.tileCounts]),this.clearGradsBind=this.bindGroup(this.clearGradsPipe,[this.accGrad]),this.clearRawBind=this.bindGroup(this.clearRawPipe,[this.gradRaw]),this.regularizerUni=this.device.createBuffer({label:"splat3d-regularizer-uniform",size:s.REGULARIZER_UNIFORM_BYTES_3D,usage:n.UNIFORM|n.COPY_DST}),this.regularizerBind=this.bindGroup(this.regularizerPipe,[this.regularizerUni,this.params,this.gradRaw]);let o=this.sharedScratchState();for(let e of(this.fwdBind=this.makeForwardBind(o,this.image,0),this.bwdBind=this.makeBackwardBind(o,this.gradImage,0),(0,s.paramSegments3D)(t.G))){let e=this.device.createBuffer({size:a.ADAM_UNIFORM_BYTES,usage:n.UNIFORM|n.COPY_DST});this.adamUni.push(e),this.adamBind.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradRaw}},{binding:3,resource:{buffer:this.mBuf}},{binding:4,resource:{buffer:this.vBuf}}]}))}}setParams(e){if(e.length!==this.dims.G*s.PARAM_STRIDE_3D)throw Error("setParams3D: wrong length");this.device.queue.writeBuffer(this.params,0,e)}zeroAdamState(){let e=new Float32Array(this.dims.G*s.PARAM_STRIDE_3D);this.device.queue.writeBuffer(this.mBuf,0,e),this.device.queue.writeBuffer(this.vBuf,0,e)}setBackground(e){if(!this.bgUni)return;let t=new Float32Array([e[0],e[1],e[2],0]);this.device.queue.writeBuffer(this.bgUni,0,t)}setCoverageRegularizer(e){if(!this.coverageUni)return;let t=new Float32Array([e.weight,e.targetAlpha,0,0]);this.device.queue.writeBuffer(this.coverageUni,0,t)}async readFloats(e,t){let r=this.device.createBuffer({size:4*t,usage:n.MAP_READ|n.COPY_DST}),i=this.device.createCommandEncoder();i.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([i.finish()]),await r.mapAsync(1);let a=new Float32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),a}readImage(){return this.readFloats(this.image,3*this.dims.H*this.dims.W)}readParams(){return this.readFloats(this.params,this.dims.G*s.PARAM_STRIDE_3D)}createIOState(e,t,r,i,a={}){this.checkIOBinding("image",t),this.checkIOBinding("grad",i);let s=a.privateState?this.createPrivateScratchState():this.sharedScratchState();return this.createIOStateForScratch(s,e,t,r,i)}async createBatchForwardState(e){let t=this.dims,r=0|e.lanes;if(r<1||r>this.cameras.length)throw Error(`raster3d: invalid batch-forward lanes ${e.lanes}`);if(e.imageOffsets.length!==r||e.gradOffsets.length!==r)throw Error("raster3d: batch-forward offsets must match lane count");this.checkContiguousImageOffsets("batch image",e.imageOffsets),this.checkContiguousImageOffsets("batch grad",e.gradOffsets);let i=this.createRawScratchBuffers(r),a=this.storage(t.G*s.DERIVED_STRIDE_3D*r,n.COPY_DST),o=this.device.createBuffer({label:"splat3d-batch-active-views",size:4*r,usage:n.STORAGE|n.COPY_DST});this.extraBuffers.push(i.derived,i.tileCounts,i.binnedIds,i.tileStop,a,o);let d=await l(this.device,(0,s.prepBatchShader3D)(t),"prep-batch"),u=await l(this.device,(0,s.clearShader3D)(t.numTiles*r),"clearBins-batch"),c=await l(this.device,(0,s.emitBatchShader3D)(t),"emit-batch"),h=await l(this.device,(0,s.forwardBatchShader3D)(t),"forward-batch"),p=await l(this.device,(0,s.clearShader3D)(t.G*s.DERIVED_STRIDE_3D*r),"clearGrads-batch"),g=await l(this.device,(0,s.backwardBatchShader3D)(t),"backward-batch"),f=e.imageOffsets[0],m=e.gradOffsets[0],w=this.bindGroup(d,[this.params,this.cameraBuffer,o,i.derived]),v=this.bindGroup(u,[i.tileCounts]),b=this.bindGroup(c,[i.derived,i.tileCounts,i.binnedIds]),B=[i.tileCounts,i.binnedIds,i.derived,{buffer:e.imageBuffer,offset:f,size:this.imageByteSize()*r},i.tileStop];this.bgUni&&B.push(this.bgUni);let y=this.bindGroup(h,B),x=this.bindGroup(p,[a]),$=[{buffer:e.gradBuffer,offset:m,size:this.imageByteSize()*r},i.tileCounts,i.binnedIds,i.tileStop,i.derived,a];this.bgUni&&$.push(this.bgUni),this.coverageUni&&$.push(this.coverageUni);let R=this.bindGroup(g,$),_=Array.from({length:r},(t,r)=>this.createIOStateForScratch(this.laneScratchState(i,r,a),e.imageBuffer,e.imageOffsets[r],e.gradBuffer,e.gradOffsets[r]));return{lanes:r,activeViews:o,ios:_,prepPipe:d,clearBinsPipe:u,emitPipe:c,fwdPipe:h,clearGradsPipe:p,bwdPipe:g,prepBind:w,clearBinsBind:v,emitBind:b,fwdBind:y,clearGradsBind:x,bwdBind:R}}recordClearRawGrad(e,t){let r=u(e,t);r.setPipeline(this.clearRawPipe),r.setBindGroup(0,this.clearRawBind),r.dispatchWorkgroups(o(this.dims.G*s.PARAM_STRIDE_3D)),r.end()}recordRegularizerAdd(e,t,r){if(!(0!==t.centerWeight||0!==t.radiusWeight||0!==t.opacitySparsity||0!==t.smallRadiusWeight||0!==t.radiusBandWeight))return;let i=new Float32Array(16);i[0]=t.centerWeight,i[1]=t.radiusWeight,i[2]=t.targetRadius,i[3]=t.opacitySparsity,i[4]=t.smallRadiusWeight,i[5]=t.smallRadius,i[6]=t.radiusBandWeight,i[7]=t.minRadius,i[8]=t.maxRadius,this.device.queue.writeBuffer(this.regularizerUni,0,i);let a=u(e,r);a.setPipeline(this.regularizerPipe),a.setBindGroup(0,this.regularizerBind),a.dispatchWorkgroups(o(this.dims.G)),a.end()}recordForward(e,t=0,r,i){let a=u(e,i);this.encodeForwardPass(a,t,r),a.end()}recordForwards(e,t,r,i){if(t.length!==r.length)throw Error(`raster3d: ${t.length} views but ${r.length} IO states`);let a=u(e,i);for(let e=0;e<t.length;e++)this.encodeForwardPass(a,t[e],r[e]);a.end()}recordBatchForward(e,t,r,i){if(r.length<1||r.length>t.lanes)throw Error(`raster3d: ${r.length} batch-forward views for ${t.lanes} lanes`);let a=new Uint32Array(t.lanes);for(let e=0;e<r.length;e++)a[e]=this.viewIndex(r[e]);this.device.queue.writeBuffer(t.activeViews,0,a);let s=this.dims,n=u(e,i);n.setPipeline(t.prepPipe),n.setBindGroup(0,t.prepBind),n.dispatchWorkgroups(o(s.G),1,r.length),n.setPipeline(t.clearBinsPipe),n.setBindGroup(0,t.clearBinsBind),n.dispatchWorkgroups(o(s.numTiles*r.length)),n.setPipeline(t.emitPipe),n.setBindGroup(0,t.emitBind),n.dispatchWorkgroups(o(s.G),1,r.length),n.setPipeline(t.fwdPipe),n.setBindGroup(0,t.fwdBind),n.dispatchWorkgroups(s.numTiles,1,r.length),n.end()}recordBatchBackwardAdd(e,t,r,i){if(r.length<1||r.length>t.lanes)throw Error(`raster3d: ${r.length} batch-backward views for ${t.lanes} lanes`);let a=this.dims,n=u(e,i);n.setPipeline(t.clearGradsPipe),n.setBindGroup(0,t.clearGradsBind),n.dispatchWorkgroups(o(a.G*s.DERIVED_STRIDE_3D*r.length)),n.setPipeline(t.bwdPipe),n.setBindGroup(0,t.bwdBind),n.dispatchWorkgroups(a.numTiles,1,r.length);for(let e=0;e<r.length;e++){let i=this.viewIndex(r[e]);n.setPipeline(this.chainPipe[i]),n.setBindGroup(0,t.ios[e].chainBind[i]),n.dispatchWorkgroups(o(a.G))}n.end()}encodeForwardPass(e,t=0,r){let i=this.dims,a=this.viewIndex(t);e.setPipeline(this.prepPipe[a]),e.setBindGroup(0,r?.prepBind[a]??this.prepBind[a]),e.dispatchWorkgroups(o(i.G)),e.setPipeline(this.clearBinsPipe),e.setBindGroup(0,r?.clearBinsBind??this.clearBinsBind),e.dispatchWorkgroups(o(i.numTiles)),e.setPipeline(this.emitPipe),e.setBindGroup(0,r?.emitBind??this.emitBind),e.dispatchWorkgroups(o(i.G)),e.setPipeline(this.fwdPipe),e.setBindGroup(0,r?.fwdBind??this.fwdBind),e.dispatchWorkgroups(i.numTiles)}recordBackwardAdd(e,t=0,r,i){let a=this.dims,n=this.viewIndex(t),d=u(e,i);d.setPipeline(this.clearGradsPipe),d.setBindGroup(0,r?.clearGradsBind??this.clearGradsBind),d.dispatchWorkgroups(o(a.G*s.DERIVED_STRIDE_3D)),d.setPipeline(this.bwdPipe),d.setBindGroup(0,r?.bwdBind??this.bwdBind),d.dispatchWorkgroups(a.numTiles),d.setPipeline(this.chainPipe[n]),d.setBindGroup(0,r?.chainBind[n]??this.chainBind[n]),d.dispatchWorkgroups(o(a.G)),d.end()}recordAdam(e,t,r=d,i=a.DEFAULT_HYPER,n){let l=(0,s.paramSegments3D)(this.dims.G),c={position:r.position,logRadius:r.logRadius,color:r.color,opacity:r.opacity},h=1-Math.pow(i.beta1,t),p=1-Math.pow(i.beta2,t);l.forEach((e,t)=>{let r=new ArrayBuffer(a.ADAM_UNIFORM_BYTES),s=new Uint32Array(r),n=new Float32Array(r);s[0]=e.offset,s[1]=e.length,n[2]=c[e.name],n[3]=i.beta1,n[4]=i.beta2,n[5]=i.eps,n[6]=h,n[7]=p,this.device.queue.writeBuffer(this.adamUni[t],0,r)});let g=u(e,n);g.setPipeline(this.adamPipe),l.forEach((e,t)=>{g.setBindGroup(0,this.adamBind[t]),g.dispatchWorkgroups(o(e.length))}),g.end()}runForward(e=0){let t=this.device.createCommandEncoder();this.recordForward(t,e),this.device.queue.submit([t.finish()])}destroy(){let e=[this.derived,this.accGrad,this.mBuf,this.vBuf,this.tileCounts,this.binnedIds,this.tileStop,this.image,this.gradImage,this.cameraBuffer,...this.extraBuffers,...this.adamUni,this.regularizerUni];for(let t of(this.bgUni&&e.push(this.bgUni),this.coverageUni&&e.push(this.coverageUni),this.ownsParams&&e.push(this.params),this.ownsGradRaw&&e.push(this.gradRaw),e))try{t.destroy()}catch(e){}}viewIndex(e){return Math.max(0,Math.min(this.cameras.length-1,0|e))}imageByteSize(){return 3*this.dims.H*this.dims.W*4}sharedScratchState(){return{derived:this.derived,accGrad:this.accGrad,tileCounts:this.tileCounts,binnedIds:this.binnedIds,tileStop:this.tileStop,prepBind:this.prepBind,chainBind:this.chainBind,emitBind:this.emitBind,clearBinsBind:this.clearBinsBind,clearGradsBind:this.clearGradsBind}}createPrivateScratchState(){let{derived:e,tileCounts:t,binnedIds:r,tileStop:i}=this.createRawScratchBuffers(1);return this.extraBuffers.push(e,t,r,i),{derived:e,accGrad:this.accGrad,tileCounts:t,binnedIds:r,tileStop:i,prepBind:this.prepPipe.map(t=>this.bindGroup(t,[this.params,e])),chainBind:this.chainPipe.map(t=>this.bindGroup(t,[this.accGrad,e,this.params,this.gradRaw])),emitBind:this.bindGroup(this.emitPipe,[e,t,r]),clearBinsBind:this.bindGroup(this.clearBinsPipe,[t]),clearGradsBind:this.clearGradsBind}}createRawScratchBuffers(e){let t=this.dims;return{derived:this.storage(t.G*s.DERIVED_STRIDE_3D*e),tileCounts:this.storage(t.numTiles*e),binnedIds:this.storage(t.numTiles*t.cap*e),tileStop:this.storage(t.numTiles*e)}}laneScratchState(e,t,r){let i=this.dims,a=i.G*s.DERIVED_STRIDE_3D*4,n=4*i.numTiles,o=i.numTiles*i.cap*4,d=this.sliceBinding(e.derived,t*a,a),l=this.sliceBinding(r,t*a,a),u=this.sliceBinding(e.tileCounts,t*n,n),c=this.sliceBinding(e.binnedIds,t*o,o),h=this.sliceBinding(e.tileStop,t*n,n);return{derived:d,accGrad:l,tileCounts:u,binnedIds:c,tileStop:h,prepBind:this.prepPipe.map(e=>this.bindGroup(e,[this.params,d])),chainBind:this.chainPipe.map(e=>this.bindGroup(e,[l,d,this.params,this.gradRaw])),emitBind:this.bindGroup(this.emitPipe,[d,u,c]),clearBinsBind:this.bindGroup(this.clearBinsPipe,[u]),clearGradsBind:this.bindGroup(this.clearGradsPipe,[l])}}createIOStateForScratch(e,t,r,i,a){return{prepBind:e.prepBind,chainBind:e.chainBind,emitBind:e.emitBind,clearBinsBind:e.clearBinsBind,clearGradsBind:e.clearGradsBind,fwdBind:this.makeForwardBind(e,t,r),bwdBind:this.makeBackwardBind(e,i,a)}}makeForwardBind(e,t,r){let i=[e.tileCounts,e.binnedIds,e.derived,{buffer:t,offset:r,size:this.imageByteSize()},e.tileStop];return this.bgUni&&i.push(this.bgUni),this.bindGroup(this.fwdPipe,i)}makeBackwardBind(e,t,r){let i=[{buffer:t,offset:r,size:this.imageByteSize()},e.tileCounts,e.binnedIds,e.tileStop,e.derived,e.accGrad];return this.bgUni&&i.push(this.bgUni),this.coverageUni&&i.push(this.coverageUni),this.bindGroup(this.bwdPipe,i)}checkIOBinding(e,t){if(!Number.isInteger(t)||t<0||t%256!=0)throw Error(`raster3d: ${e} offset ${t} must be 256-byte aligned`)}checkContiguousImageOffsets(e,t){if(!t.length)throw Error(`raster3d: empty ${e} offsets`);let r=this.imageByteSize();for(let i=0;i<t.length;i++)if(this.checkIOBinding(e,t[i]),t[i]!==t[0]+i*r)throw Error(`raster3d: ${e} offsets must be contiguous image lanes`)}sliceBinding(e,t,r){return this.checkIOBinding("scratch",t),{buffer:e,offset:t,size:r}}}},{"../splat/adam_wgsl":"kfWkJ","./raster_wgsl":"fuyeU","@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}],fuyeU:[function(e,t,r){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"TILE",()=>a),i.export(r,"PARAM_STRIDE_3D",()=>s),i.export(r,"DERIVED_STRIDE_3D",()=>n),i.export(r,"CAMERA_STRIDE_3D",()=>o),i.export(r,"COVERAGE_UNIFORM_BYTES_3D",()=>d),i.export(r,"ALPHA_THRESHOLD",()=>l),i.export(r,"MAX_ALPHA",()=>u),i.export(r,"TRANSMITTANCE_CUTOFF",()=>c),i.export(r,"EPS",()=>h),i.export(r,"RADIUS_MIN",()=>p),i.export(r,"RADIUS_MAX",()=>g),i.export(r,"resolveDims3D",()=>b),i.export(r,"prepShader3D",()=>P),i.export(r,"prepBatchShader3D",()=>C),i.export(r,"emitShader3D",()=>k),i.export(r,"emitBatchShader3D",()=>T),i.export(r,"forwardShader3D",()=>I),i.export(r,"forwardBatchShader3D",()=>G),i.export(r,"backwardShader3D",()=>E),i.export(r,"backwardBatchShader3D",()=>D),i.export(r,"chainAddShader3D",()=>O),i.export(r,"clearShader3D",()=>F),i.export(r,"REGULARIZER_UNIFORM_BYTES_3D",()=>A),i.export(r,"regularizerShader3D",()=>z),i.export(r,"paramSegments3D",()=>M);let a=16,s=8,n=11,o=16,d=16,l=1/255,u=.99,c=1e-4,h=1e-8,p=.01,g=.45;function f(e,t){if(!e)throw Error(`raster3d_wgsl: ${t}`)}function m(e){f(Number.isFinite(e),`non-finite literal ${e}`);let t=e.toString();return/[.eE]/.test(t)||(t+=".0"),t}let w=e=>`${e>>>0}u`,v=e=>`vec3f(${m(e[0])}, ${m(e[1])}, ${m(e[2])})`;function b(e){return f(e.H>0&&e.W>0&&e.G>0,"H,W,G must be positive"),f(e.H%a==0&&e.W%a==0,`H,W must be multiples of ${a}`),f((e.cap&e.cap-1)==0&&e.cap>0,"cap must be a power of two"),f(4*e.cap<=16384,`cap*4 (${4*e.cap}B) exceeds 16KB workgroup storage`),{H:e.H,W:e.W,G:e.G,cap:e.cap,tilesX:e.W/a,tilesY:e.H/a,numTiles:e.W/a*(e.H/a),bg:e.bg??[0,0,0],dynamicBg:e.dynamicBg??!1,dynamicCoverage:e.dynamicCoverage??!1,near:e.near??.2,far:e.far??12,gradScale:e.gradScale??65536}}function B(e){return{position:0,logRadius:3*e.G,colorRaw:4*e.G,opacityRaw:7*e.G}}let y=/* wgsl */"fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }";function x(e){return/* wgsl */`
struct BgU {
  rgb : vec3f,
  _pad : f32,
};
@group(0) @binding(${e}) var<uniform> bgU : BgU;
`}function $(e,t){return e.dynamicBg?0===t?"bgU.rgb.x":1===t?"bgU.rgb.y":"bgU.rgb.z":m(e.bg[t])}function R(e){return e.dynamicCoverage?/* wgsl */`
struct CoverageU {
  weight      : f32,
  targetAlpha : f32,
  _pad0       : f32,
  _pad1       : f32,
};
@group(0) @binding(${e.dynamicBg?7:6}) var<uniform> coverageU : CoverageU;
`:""}function _(e,t){return e.dynamicCoverage?/* wgsl */`
  let targetT = 1.0 - clamp(coverageU.targetAlpha, 0.0, 1.0);
  gT = gT + coverageU.weight * ${m(2/t)} * (T - targetT);
`:""}function S(e){return/* wgsl */`
const CAM_EYE = ${v(e.eye)};
const CAM_RIGHT = ${v(e.right)};
const CAM_UP = ${v(e.cameraUp)};
const CAM_FWD = ${v(e.forward)};
const FOCAL_PX = ${m(e.focalPx)};
`}function P(e,t){let r=b(e),i=B(r);return/* wgsl */`
${y}
${S(t)}
@group(0) @binding(0) var<storage, read>       params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${w(r.G)}) { return; }

  let p = vec3f(
    params[${w(i.position)} + g * 3u + 0u],
    params[${w(i.position)} + g * 3u + 1u],
    params[${w(i.position)} + g * 3u + 2u]
  );
  let w = p - CAM_EYE;
  let vx = dot(w, CAM_RIGHT);
  let vy = dot(w, CAM_UP);
  let vz = dot(w, CAM_FWD);
  let safeZ = max(vz, ${m(r.near)});
  let radiusWorld = clamp(exp(params[${w(i.logRadius)} + g]), ${m(p)}, ${m(g)});
  let radiusPx = max(FOCAL_PX * radiusWorld / safeZ, 0.25);
  let invR2 = 1.0 / max(radiusPx * radiusPx, ${m(h)});
  let sx = ${m(.5*r.W)} + FOCAL_PX * (vx / safeZ);
  let sy = ${m(.5*r.H)} - FOCAL_PX * (vy / safeZ);

  let base = g * ${w(n)};
  derived[base + 0u] = sx;
  derived[base + 1u] = sy;
  derived[base + 2u] = invR2;
  derived[base + 3u] = vz;
  derived[base + 4u] = vx;
  derived[base + 5u] = vy;
  derived[base + 6u] = safeZ;
  derived[base + 7u] = sigmoid1(params[${w(i.colorRaw)} + g * 3u + 0u]);
  derived[base + 8u] = sigmoid1(params[${w(i.colorRaw)} + g * 3u + 1u]);
  derived[base + 9u] = sigmoid1(params[${w(i.colorRaw)} + g * 3u + 2u]);
  derived[base + 10u] = sigmoid1(params[${w(i.opacityRaw)} + g]);
}
`}function C(e){let t=b(e),r=B(t);return/* wgsl */`
${y}
@group(0) @binding(0) var<storage, read>       params      : array<f32>;
@group(0) @binding(1) var<storage, read>       cameras     : array<f32>;
@group(0) @binding(2) var<storage, read>       activeViews : array<u32>;
@group(0) @binding(3) var<storage, read_write> derived     : array<f32>;


fn cameraBase(view : u32) -> u32 {
  return view * ${w(o)};
}

fn cameraEye(view : u32) -> vec3f {
  let b = cameraBase(view);
  return vec3f(cameras[b + 0u], cameras[b + 1u], cameras[b + 2u]);
}

fn cameraRight(view : u32) -> vec3f {
  let b = cameraBase(view);
  return vec3f(cameras[b + 3u], cameras[b + 4u], cameras[b + 5u]);
}

fn cameraUp(view : u32) -> vec3f {
  let b = cameraBase(view);
  return vec3f(cameras[b + 6u], cameras[b + 7u], cameras[b + 8u]);
}

fn cameraFwd(view : u32) -> vec3f {
  let b = cameraBase(view);
  return vec3f(cameras[b + 9u], cameras[b + 10u], cameras[b + 11u]);
}

fn cameraFocalPx(view : u32) -> f32 {
  return cameras[cameraBase(view) + 12u];
}


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  let lane = gid.z;
  if (g >= ${w(t.G)}) { return; }
  let view = activeViews[lane];
  let eye = cameraEye(view);
  let right = cameraRight(view);
  let up = cameraUp(view);
  let fwd = cameraFwd(view);
  let focalPx = cameraFocalPx(view);

  let p = vec3f(
    params[${w(r.position)} + g * 3u + 0u],
    params[${w(r.position)} + g * 3u + 1u],
    params[${w(r.position)} + g * 3u + 2u]
  );
  let w = p - eye;
  let vx = dot(w, right);
  let vy = dot(w, up);
  let vz = dot(w, fwd);
  let safeZ = max(vz, ${m(t.near)});
  let radiusWorld = clamp(exp(params[${w(r.logRadius)} + g]), ${m(p)}, ${m(g)});
  let radiusPx = max(focalPx * radiusWorld / safeZ, 0.25);
  let invR2 = 1.0 / max(radiusPx * radiusPx, ${m(h)});
  let sx = ${m(.5*t.W)} + focalPx * (vx / safeZ);
  let sy = ${m(.5*t.H)} - focalPx * (vy / safeZ);

  let base = lane * ${w(t.G*n)} + g * ${w(n)};
  derived[base + 0u] = sx;
  derived[base + 1u] = sy;
  derived[base + 2u] = invR2;
  derived[base + 3u] = vz;
  derived[base + 4u] = vx;
  derived[base + 5u] = vy;
  derived[base + 6u] = safeZ;
  derived[base + 7u] = sigmoid1(params[${w(r.colorRaw)} + g * 3u + 0u]);
  derived[base + 8u] = sigmoid1(params[${w(r.colorRaw)} + g * 3u + 1u]);
  derived[base + 9u] = sigmoid1(params[${w(r.colorRaw)} + g * 3u + 2u]);
  derived[base + 10u] = sigmoid1(params[${w(r.opacityRaw)} + g]);
}
`}function k(e){let t=b(e);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       derived    : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds  : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${w(t.G)}) { return; }
  let base = g * ${w(n)};
  let depth = derived[base + 3u];
  if (depth <= ${m(t.near)} || depth >= ${m(t.far)}) { return; }
  let op = derived[base + 10u];
  if (op <= ${m(l)}) { return; }
  let ratio = max(${m(l)} / max(op, ${m(h)}), ${m(h)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let sx = derived[base + 0u];
  let sy = derived[base + 1u];
  let invR2 = max(derived[base + 2u], ${m(h)});
  let radius = sqrt(tau / invR2);
  let x0 = max(0, i32(floor(sx - radius - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(sx + radius - 0.5)));
  let y0 = max(0, i32(floor(sy - radius - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(sy + radius - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tx0 = x0 / ${a}; let tx1 = x1 / ${a};
  let ty0 = y0 / ${a}; let ty1 = y1 / ${a};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let tile = u32(ty * ${t.tilesX} + tx);
      let slot = atomicAdd(&tileCounts[tile], 1u);
      if (slot < ${w(t.cap)}) { binnedIds[tile * ${w(t.cap)} + slot] = g; }
    }
  }
}
`}function T(e){let t=b(e);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       derived    : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds  : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  let lane = gid.z;
  if (g >= ${w(t.G)}) { return; }
  let derivedBase = lane * ${w(t.G*n)} + g * ${w(n)};
  let depth = derived[derivedBase + 3u];
  if (depth <= ${m(t.near)} || depth >= ${m(t.far)}) { return; }
  let op = derived[derivedBase + 10u];
  if (op <= ${m(l)}) { return; }
  let ratio = max(${m(l)} / max(op, ${m(h)}), ${m(h)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let sx = derived[derivedBase + 0u];
  let sy = derived[derivedBase + 1u];
  let invR2 = max(derived[derivedBase + 2u], ${m(h)});
  let radius = sqrt(tau / invR2);
  let x0 = max(0, i32(floor(sx - radius - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(sx + radius - 0.5)));
  let y0 = max(0, i32(floor(sy - radius - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(sy + radius - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tileCountsBase = lane * ${w(t.numTiles)};
  let binnedBase = lane * ${w(t.numTiles*t.cap)};
  let tx0 = x0 / ${a}; let tx1 = x1 / ${a};
  let ty0 = y0 / ${a}; let ty1 = y1 / ${a};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let tile = u32(ty * ${t.tilesX} + tx);
      let slot = atomicAdd(&tileCounts[tileCountsBase + tile], 1u);
      if (slot < ${w(t.cap)}) { binnedIds[binnedBase + tile * ${w(t.cap)} + slot] = g; }
    }
  }
}
`}function I(e){let t=b(e),r=t.H*t.W;return/* wgsl */`
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;
${t.dynamicBg?x(5):""}

var<workgroup> sh_ids     : array<u32, ${t.cap}>;
var<workgroup> sh_maxstop : atomic<u32>;

fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u); v = v - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

fn idGreater(a : u32, b : u32) -> bool {
  if (a == 0xffffffffu) { return b != 0xffffffffu; }
  if (b == 0xffffffffu) { return false; }
  let za = derived[a * ${w(n)} + 3u];
  let zb = derived[b * ${w(n)} + 3u];
  if (za == zb) { return a > b; }
  return za > zb;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${w(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${w(t.cap)});
  let start = tileId * ${w(t.cap)};
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

  let tileX = tileId % ${w(t.tilesX)};
  let tileY = tileId / ${w(t.tilesX)};
  let x = tileX * ${a}u + (tid % ${a}u);
  let y = tileY * ${a}u + (tid / ${a}u);
  var localStop = 0u;
  if (x < ${w(t.W)} && y < ${w(t.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var accR = 0.0; var accG = 0.0; var accB = 0.0; var T = 1.0;
    for (var i = 0u; i < count; i = i + 1u) {
      let gg = sh_ids[i];
      let b = gg * ${w(n)};
      let dx = pxc - derived[b + 0u];
      let dy = pyc - derived[b + 1u];
      let invR2 = derived[b + 2u];
      let power = -0.5 * invR2 * (dx * dx + dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[b + 10u] * exp(power);
      let alpha = min(${m(u)}, raw);
      if (alpha < ${m(l)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b + 7u];
      accG = accG + w * derived[b + 8u];
      accB = accB + w * derived[b + 9u];
      T = T * (1.0 - alpha);
      if (T < ${m(c)}) { break; }
    }
    let pix = y * ${w(t.W)} + x;
    image[0u * ${w(r)} + pix] = accR + T * ${$(t,0)};
    image[1u * ${w(r)} + pix] = accG + T * ${$(t,1)};
    image[2u * ${w(r)} + pix] = accB + T * ${$(t,2)};
  }
  atomicMax(&sh_maxstop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&sh_maxstop); }
}
`}function G(e){let t=b(e),r=t.H*t.W,i=3*r;return/* wgsl */`
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;
${t.dynamicBg?x(5):""}

var<workgroup> sh_ids     : array<u32, ${t.cap}>;
var<workgroup> sh_maxstop : atomic<u32>;

fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u); v = v - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

fn derivedBase(lane : u32, g : u32) -> u32 {
  return lane * ${w(t.G*n)} + g * ${w(n)};
}

fn idGreater(lane : u32, a : u32, b : u32) -> bool {
  if (a == 0xffffffffu) { return b != 0xffffffffu; }
  if (b == 0xffffffffu) { return false; }
  let za = derived[derivedBase(lane, a) + 3u];
  let zb = derived[derivedBase(lane, b) + 3u];
  if (za == zb) { return a > b; }
  return za > zb;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  let lane = wg.z;
  if (tileId >= ${w(t.numTiles)}) { return; }
  let tileCountsBase = lane * ${w(t.numTiles)};
  let binnedBase = lane * ${w(t.numTiles*t.cap)};
  let tileStopBase = lane * ${w(t.numTiles)};
  let imageBase = lane * ${w(i)};
  let count = min(tileCounts[tileCountsBase + tileId], ${w(t.cap)});
  let start = binnedBase + tileId * ${w(t.cap)};
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
        let swapAsc = idGreater(lane, va, vb);
        let swapDesc = idGreater(lane, vb, va);
        if ((asc && swapAsc) || (!asc && swapDesc)) { sh_ids[pos] = vb; sh_ids[ixj] = va; }
      }
      workgroupBarrier();
      j = j >> 1u;
    }
    k = k << 1u;
  }

  for (var i = tid; i < count; i = i + 256u) { binnedIds[start + i] = sh_ids[i]; }
  workgroupBarrier();

  let tileX = tileId % ${w(t.tilesX)};
  let tileY = tileId / ${w(t.tilesX)};
  let x = tileX * ${a}u + (tid % ${a}u);
  let y = tileY * ${a}u + (tid / ${a}u);
  var localStop = 0u;
  if (x < ${w(t.W)} && y < ${w(t.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var accR = 0.0; var accG = 0.0; var accB = 0.0; var T = 1.0;
    for (var i = 0u; i < count; i = i + 1u) {
      let gg = sh_ids[i];
      let b = derivedBase(lane, gg);
      let dx = pxc - derived[b + 0u];
      let dy = pyc - derived[b + 1u];
      let invR2 = derived[b + 2u];
      let power = -0.5 * invR2 * (dx * dx + dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[b + 10u] * exp(power);
      let alpha = min(${m(u)}, raw);
      if (alpha < ${m(l)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b + 7u];
      accG = accG + w * derived[b + 8u];
      accB = accB + w * derived[b + 9u];
      T = T * (1.0 - alpha);
      if (T < ${m(c)}) { break; }
    }
    let pix = y * ${w(t.W)} + x;
    image[imageBase + 0u * ${w(r)} + pix] = accR + T * ${$(t,0)};
    image[imageBase + 1u * ${w(r)} + pix] = accG + T * ${$(t,1)};
    image[imageBase + 2u * ${w(r)} + pix] = accB + T * ${$(t,2)};
  }
  atomicMax(&sh_maxstop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileStopBase + tileId] = atomicLoad(&sh_maxstop); }
}
`}function E(e){let t=b(e),r=t.H*t.W,i=m(t.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;
${t.dynamicBg?x(6):""}
${R(t)}

var<workgroup> sh_ids : array<u32, ${t.cap}>;

fn fixadd(base : u32, slot : u32, v : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(v * ${i}), -2.14e9, 2.14e9)));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${w(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${w(t.cap)});
  let stopc = min(count, tileStop[tileId]);
  let start = tileId * ${w(t.cap)};
  for (var i = tid; i < stopc; i = i + 256u) { sh_ids[i] = binnedIds[start + i]; }
  workgroupBarrier();

  let tileX = tileId % ${w(t.tilesX)};
  let tileY = tileId / ${w(t.tilesX)};
  let x = tileX * ${a}u + (tid % ${a}u);
  let y = tileY * ${a}u + (tid / ${a}u);
  if (x >= ${w(t.W)} || y >= ${w(t.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${w(t.W)} + x;
  let goR = gradImage[0u * ${w(r)} + pix];
  let goG = gradImage[1u * ${w(r)} + pix];
  let goB = gradImage[2u * ${w(r)} + pix];

  var T = 1.0;
  var endi = stopc;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b = gg * ${w(n)};
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let power = -0.5 * derived[b + 2u] * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${m(u)}, derived[b + 10u] * exp(power));
    if (alpha < ${m(l)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${m(c)}) { endi = i + 1u; break; }
  }

  var Tcur = T;
  var gT = goR * ${$(t,0)} + goG * ${$(t,1)} + goB * ${$(t,2)};
${_(t,r)}
  for (var ii = i32(endi) - 1; ii >= 0; ii = ii - 1) {
    let gg = sh_ids[u32(ii)];
    let b = gg * ${w(n)};
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let invR2 = derived[b + 2u];
    let power = -0.5 * invR2 * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let op = derived[b + 10u];
    let raw = op * exp(power);
    let alpha = min(${m(u)}, raw);
    if (alpha < ${m(l)}) { continue; }
    let denom = max(1.0 - alpha, ${m(h)});
    let Tprev = Tcur / denom;
    let cR = derived[b + 7u]; let cG = derived[b + 8u]; let cB = derived[b + 9u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b, 7u, goR * Tprev * alpha);
    fixadd(b, 8u, goG * Tprev * alpha);
    fixadd(b, 9u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${m(u)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-invR2 * dx);
    let gdy = gPower * (-invR2 * dy);
    fixadd(b, 0u, -gdx);
    fixadd(b, 1u, -gdy);
    fixadd(b, 2u, gPower * (-0.5) * (dx * dx + dy * dy));
    fixadd(b, 10u, gRaw * (raw / max(op, ${m(h)})));

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`}function D(e){let t=b(e),r=t.H*t.W,i=3*r,s=m(t.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;
${t.dynamicBg?x(6):""}
${R(t)}

var<workgroup> sh_ids : array<u32, ${t.cap}>;

fn derivedBase(lane : u32, g : u32) -> u32 {
  return lane * ${w(t.G*n)} + g * ${w(n)};
}

fn fixadd(base : u32, slot : u32, v : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(v * ${s}), -2.14e9, 2.14e9)));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  let lane = wg.z;
  if (tileId >= ${w(t.numTiles)}) { return; }
  let tileCountsBase = lane * ${w(t.numTiles)};
  let binnedBase = lane * ${w(t.numTiles*t.cap)};
  let tileStopBase = lane * ${w(t.numTiles)};
  let gradImageBase = lane * ${w(i)};
  let count = min(tileCounts[tileCountsBase + tileId], ${w(t.cap)});
  let stopc = min(count, tileStop[tileStopBase + tileId]);
  let start = binnedBase + tileId * ${w(t.cap)};
  for (var i = tid; i < stopc; i = i + 256u) { sh_ids[i] = binnedIds[start + i]; }
  workgroupBarrier();

  let tileX = tileId % ${w(t.tilesX)};
  let tileY = tileId / ${w(t.tilesX)};
  let x = tileX * ${a}u + (tid % ${a}u);
  let y = tileY * ${a}u + (tid / ${a}u);
  if (x >= ${w(t.W)} || y >= ${w(t.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${w(t.W)} + x;
  let goR = gradImage[gradImageBase + 0u * ${w(r)} + pix];
  let goG = gradImage[gradImageBase + 1u * ${w(r)} + pix];
  let goB = gradImage[gradImageBase + 2u * ${w(r)} + pix];

  var T = 1.0;
  var endi = stopc;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b = derivedBase(lane, gg);
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let power = -0.5 * derived[b + 2u] * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${m(u)}, derived[b + 10u] * exp(power));
    if (alpha < ${m(l)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${m(c)}) { endi = i + 1u; break; }
  }

  var Tcur = T;
  var gT = goR * ${$(t,0)} + goG * ${$(t,1)} + goB * ${$(t,2)};
${_(t,r)}
  for (var ii = i32(endi) - 1; ii >= 0; ii = ii - 1) {
    let gg = sh_ids[u32(ii)];
    let b = derivedBase(lane, gg);
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let invR2 = derived[b + 2u];
    let power = -0.5 * invR2 * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let op = derived[b + 10u];
    let raw = op * exp(power);
    let alpha = min(${m(u)}, raw);
    if (alpha < ${m(l)}) { continue; }
    let denom = max(1.0 - alpha, ${m(h)});
    let Tprev = Tcur / denom;
    let cR = derived[b + 7u]; let cG = derived[b + 8u]; let cB = derived[b + 9u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b, 7u, goR * Tprev * alpha);
    fixadd(b, 8u, goG * Tprev * alpha);
    fixadd(b, 9u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${m(u)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-invR2 * dx);
    let gdy = gPower * (-invR2 * dy);
    fixadd(b, 0u, -gdx);
    fixadd(b, 1u, -gdy);
    fixadd(b, 2u, gPower * (-0.5) * (dx * dx + dy * dy));
    fixadd(b, 10u, gRaw * (raw / max(op, ${m(h)})));

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`}function O(e,t){let r=b(e),i=B(r),a=m(1/r.gradScale);return/* wgsl */`
${S(t)}
@group(0) @binding(0) var<storage, read>       accGrad : array<i32>;
@group(0) @binding(1) var<storage, read>       derived : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${w(r.G)}) { return; }
  let b = g * ${w(n)};
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
  let vz = max(derived[b + 6u], ${m(r.near)});
  let invR2 = derived[b + 2u];
  let invZ = 1.0 / vz;
  let invZ2 = invZ * invZ;
  let gvx = gsx * FOCAL_PX * invZ;
  let gvy = -gsy * FOCAL_PX * invZ;
  let gvz = gsx * (-FOCAL_PX * vx * invZ2) + gsy * (FOCAL_PX * vy * invZ2) + gInv * (2.0 * invR2 * invZ);
  let gp = CAM_RIGHT * gvx + CAM_UP * gvy + CAM_FWD * gvz;

  gradRaw[${w(i.position)} + g * 3u + 0u] = gradRaw[${w(i.position)} + g * 3u + 0u] + gp.x;
  gradRaw[${w(i.position)} + g * 3u + 1u] = gradRaw[${w(i.position)} + g * 3u + 1u] + gp.y;
  gradRaw[${w(i.position)} + g * 3u + 2u] = gradRaw[${w(i.position)} + g * 3u + 2u] + gp.z;

  let lr = params[${w(i.logRadius)} + g];
  let er = exp(lr);
  let gateR = select(0.0, 1.0, er > ${m(p)} && er < ${m(g)});
  gradRaw[${w(i.logRadius)} + g] = gradRaw[${w(i.logRadius)} + g] + gInv * (-2.0 * invR2) * gateR;

  let col0 = derived[b + 7u]; let col1 = derived[b + 8u]; let col2 = derived[b + 9u];
  let opv = derived[b + 10u];
  gradRaw[${w(i.colorRaw)} + g * 3u + 0u] = gradRaw[${w(i.colorRaw)} + g * 3u + 0u] + gc0 * col0 * (1.0 - col0);
  gradRaw[${w(i.colorRaw)} + g * 3u + 1u] = gradRaw[${w(i.colorRaw)} + g * 3u + 1u] + gc1 * col1 * (1.0 - col1);
  gradRaw[${w(i.colorRaw)} + g * 3u + 2u] = gradRaw[${w(i.colorRaw)} + g * 3u + 2u] + gc2 * col2 * (1.0 - col2);
  gradRaw[${w(i.opacityRaw)} + g] = gradRaw[${w(i.opacityRaw)} + g] + gop * opv * (1.0 - opv);
}
`}function F(e){return/* wgsl */`
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= ${w(e)}) { return; }
  buf[gid.x] = 0u;
}
`}let A=64;function z(e){let t=b(e),r=B(t);return/* wgsl */`
${y}
struct RegU {
  centerWeight   : f32,
  radiusWeight   : f32,
  targetRadius   : f32,
  opacitySparsity: f32,
  smallRadiusWeight : f32,
  smallRadius       : f32,
  radiusBandWeight  : f32,
  minRadius         : f32,
  maxRadius         : f32,
  _pad0             : f32,
  _pad1             : f32,
  _pad2             : f32,
};
@group(0) @binding(0) var<uniform>             u       : RegU;
@group(0) @binding(1) var<storage, read>       params  : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${w(t.G)}) { return; }

  let pxIdx = ${w(r.position)} + g * 3u + 0u;
  let pyIdx = ${w(r.position)} + g * 3u + 1u;
  let pzIdx = ${w(r.position)} + g * 3u + 2u;
  let p = vec3f(params[pxIdx], params[pyIdx], params[pzIdx]);
  let r = length(p);
  let invR = 1.0 / max(r, ${m(h)});
  let outside = max(0.0, r - max(u.targetRadius, ${m(h)}));
  let gp = (2.0 * u.centerWeight) * p + (2.0 * u.radiusWeight * outside * invR) * p;
  gradRaw[pxIdx] = gradRaw[pxIdx] + gp.x;
  gradRaw[pyIdx] = gradRaw[pyIdx] + gp.y;
  gradRaw[pzIdx] = gradRaw[pzIdx] + gp.z;

  let opIdx = ${w(r.opacityRaw)} + g;
  let op = sigmoid1(params[opIdx]);
  gradRaw[opIdx] = gradRaw[opIdx] + u.opacitySparsity * op * (1.0 - op);

  let radiusIdx = ${w(r.logRadius)} + g;
  let logRadius = params[radiusIdx];
  let radius = exp(logRadius);
  let small = max(0.0, max(u.smallRadius, ${m(h)}) - radius);
  let smallLossGrad = u.smallRadiusWeight * small * small;
  gradRaw[opIdx] = gradRaw[opIdx] + 2.0 * smallLossGrad * op * op * (1.0 - op);
  gradRaw[radiusIdx] = gradRaw[radiusIdx] - 2.0 * u.smallRadiusWeight * op * op * small * radius;

  let minR = max(u.minRadius, ${m(h)});
  let maxR = max(u.maxRadius, minR + ${m(h)});
  let under = max(0.0, minR - radius);
  let over = max(0.0, radius - maxR);
  let gRadius = u.radiusBandWeight * (-2.0 * under + 2.0 * over);
  gradRaw[radiusIdx] = gradRaw[radiusIdx] + gRadius * radius;
}
`}function M(e){return[{name:"position",offset:0,length:3*e},{name:"logRadius",offset:3*e,length:e},{name:"color",offset:4*e,length:3*e},{name:"opacity",offset:7*e,length:e}]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"4C2Su"}]},["e5WXe"],"e5WXe","parcelRequire924a")//# sourceMappingURL=splat3d.24e8a634.js.map
;
//# sourceMappingURL=splat3d.24e8a634.js.map
