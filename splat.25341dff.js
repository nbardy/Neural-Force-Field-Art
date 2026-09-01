!function(e,t,r,a,i){var s="u">typeof globalThis?globalThis:"u">typeof self?self:"u">typeof window?window:"u">typeof global?global:{},d="function"==typeof s[a]&&s[a],o=d.i||{},u=d.cache||{},n="u">typeof module&&"function"==typeof module.require&&module.require.bind(module);function l(t,r){if(!u[t]){if(!e[t]){if(i[t])return i[t];var o="function"==typeof s[a]&&s[a];if(!r&&o)return o(t,!0);if(d)return d(t,!0);if(n&&"string"==typeof t)return n(t);var c=Error("Cannot find module '"+t+"'");throw c.code="MODULE_NOT_FOUND",c}g.resolve=function(r){var a=e[t][1][r];return null!=a?a:r},g.cache={};var p=u[t]=new l.Module(t);e[t][0].call(p.exports,g,p,p.exports,s)}return u[t].exports;function g(e){var t=g.resolve(e);if(!1===t)return{};if(Array.isArray(t)){var r={__esModule:!0};return t.forEach(function(e){var t=e[0],a=e[1],i=e[2]||e[0],s=l(a);"*"===t?Object.keys(s).forEach(function(e){"default"===e||"__esModule"===e||Object.prototype.hasOwnProperty.call(r,e)||Object.defineProperty(r,e,{enumerable:!0,get:function(){return s[e]}})}):"*"===i?Object.defineProperty(r,t,{enumerable:!0,value:s}):Object.defineProperty(r,t,{enumerable:!0,get:function(){return"default"===i?s.__esModule?s.default:s:s[i]}})}),r}return l(t)}}l.isParcelRequire=!0,l.Module=function(e){this.id=e,this.bundle=l,this.require=n,this.exports={}},l.modules=e,l.cache=u,l.parent=d,l.distDir=void 0,l.publicUrl=void 0,l.devServer=void 0,l.i=o,l.register=function(t,r){e[t]=[function(e,t){t.exports=r},{}]},Object.defineProperty(l,"root",{get:function(){return s[a]}}),s[a]=l;for(var c=0;c<t.length;c++)l(t[c]);if(r){var p=l(r);"object"==typeof exports&&"u">typeof module?module.exports=p:"function"==typeof define&&define.amd&&define(function(){return p})}}({"3k7uM":[function(e,t,r,a){let i,s,d,o,u,n,l;var c=e("./splat/optimize"),p=e("./splat/feature_optimize"),g=e("./splat/pixel_optimize"),f=e("./splat/model_assets");let m={gpu:!!navigator.gpu,ready:!1,running:!1,step:0,cos:null,initialCos:null,error:null,phase:"boot"};window.__splat=m;let h=document.getElementById("splat"),y=document.getElementById("prompt"),b=document.getElementById("optimize"),x=document.getElementById("nudge"),v=document.getElementById("reset"),E=document.getElementById("representation"),w=document.getElementById("auto-explore"),A=document.getElementById("readout"),_=document.getElementById("notice");function R(e){_.textContent=e}function S(e){m.error=e,m.phase="error",R(e),A.textContent="—",console.error("[splat_page]",e)}function T(){m.step=u?u.stepCount:0;let e=[`step ${m.step}`];if(null!==m.cos){let t=m.initialCos??m.cos,r=m.cos-t;e.push(`cos ${m.cos.toFixed(4)}`),e.push(`init ${t.toFixed(4)}`),e.push(`\u{394} ${r>=0?"+":""}${r.toFixed(4)}`)}m.phase&&"run"!==m.phase&&e.push(`(${m.phase})`),A.textContent=e.join("  ·  ")}let $=1,C=null,B=`
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
`;async function P(){i.pushErrorScope("validation");let e=i.createShaderModule({code:B});n=i.createRenderPipeline({layout:"auto",vertex:{module:e,entryPoint:"vs"},fragment:{module:e,entryPoint:"fs",targets:[{format:l}]},primitive:{topology:"triangle-list"}});let t=await i.popErrorScope();if(t)throw Error(`blit pipeline invalid: ${t.message}`)}function L(){C=i.createBindGroup({layout:n.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:u.raster.image}}]})}let I=Function("u","return import(u)"),F=null,G=null;async function D(e){if(G)return;let t=await I("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm");t.env.allowRemoteModels=!0;let r="Nbardy/nff-clip-splat-weights",a=t=>{if("progress"===t.status&&t.total){let r=Math.round(t.progress??t.loaded/t.total*100),a=Math.round(r/100*16),i="█".repeat(a)+"░".repeat(16-a);e?.(`loading text encoder  [${i}] ${r}%  \xb7  ${(t.loaded/1e6).toFixed(1)}/${(t.total/1e6).toFixed(0)} MB`)}};F=await t.AutoTokenizer.from_pretrained(r,{progress_callback:a}),G=await t.CLIPTextModelWithProjection.from_pretrained(r,{dtype:"fp16",device:"wasm",session_options:{graphOptimizationLevel:"basic"},progress_callback:a})}async function O(e){await D();let t=await F(e,{padding:"max_length",max_length:77,truncation:!0}),r=(await G(t)).text_embeds.data,a=new Float32Array(512);for(let e=0;e<512;e++)a[e]=r[e];return a}let U=null,q=0,M=!1,z=!1,k=0,W=[{step:96,amount:.12},{step:260,amount:.06}];async function N(){if(U&&!M&&!z){M=!0;try{let e=await u.currentEmbedding(),t=(0,c.cosine)(e,U);m.cos=t,null===m.initialCos&&(m.initialCos=t),T()}finally{M=!1}}}function H(){m.running&&U&&(u.step(),u.step(),q+=2,m.step=u.stepCount,q>=14&&(q=0,N()),function(){if(!w.checked||z||u instanceof g.PixelBufferOptimizer)return;let e=W[k];e&&!(u.stepCount<e.step)&&(k+=1,V({amount:e.amount,automatic:!0}))}()),function(){if(!C)return;let e=i.createCommandEncoder(),t=e.beginRenderPass({colorAttachments:[{view:s.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});t.setPipeline(n),t.setBindGroup(0,C),t.draw(3),t.end(),i.queue.submit([e.finish()])}(),requestAnimationFrame(H)}async function j(){if(!m.ready)return;let e=y.value.trim()||"a photo of a cat";b.disabled=!0,m.phase="encoding",m.running=!1,R("encoding prompt (first use downloads the text model — slow)…"),T();try{let t=await O(e);U=t,u.setPrompt(t);let r=await u.currentEmbedding();m.initialCos=(0,c.cosine)(r,t),m.cos=m.initialCos,q=0,k=0,R(""),m.phase="run",m.running=!0,T()}catch(e){S(`text encode failed: ${e?.message??e}`)}finally{b.disabled=!1}}async function X(){if(!m.ready)return;m.running=!1,U=null,m.cos=null,m.initialCos=null,m.phase="reset",k=0,$+=1;let e=u;try{u=await Y(),e.destroy(),L(),await u.renderImage(),y.value="",m.step=0,m.phase="idle",R(""),T()}catch(e){S(`reset failed: ${e?.message??e}`)}}async function Y(){return"feature"===E.value?p.FeaturePainterOptimizer.create(i,d,o,{seed:$}):"pixels"===E.value?g.PixelBufferOptimizer.create(i,d,o,$):c.SplatOptimizer.create(i,d,o,{seed:$})}async function V(e={}){if(!m.ready||z)return;z=!0;let t=m.running;m.running=!1,m.phase=e.automatic?"explore":"nudge",x.disabled=!0,$+=1,T();try{if(u instanceof g.PixelBufferOptimizer?await u.nudge($,e.amount):await u.nudge({seed:$,amount:e.amount}),await u.renderImage(),q=0,U){let e=await u.currentEmbedding();m.cos=(0,c.cosine)(e,U)}m.phase=t&&U?"run":"idle",m.running=t&&!!U,R(""),T()}catch(e){S(`nudge failed: ${e?.message??e}`)}finally{z=!1,x.disabled=!1}}async function K(){if(!navigator.gpu){S("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled."),b.disabled=!0,x.disabled=!0,v.disabled=!0;return}m.phase="adapter";let e=await navigator.gpu.requestAdapter();if(!e)return void S("no WebGPU adapter available.");if(!e.features.has("subgroups"))return void S("this build requires WebGPU subgroup reductions.");i=await e.requestDevice({requiredFeatures:["subgroups"]}),i.addEventListener?.("uncapturederror",e=>{console.error("[webgpu]",e.error?.message??e.error)}),s=h.getContext("webgpu"),l=navigator.gpu.getPreferredCanvasFormat(),s.configure({device:i,format:l,alphaMode:"opaque"}),m.phase="weights";try{let e=await (0,f.loadClipTrainAssets)(e=>{A.textContent=e});d=e.plan,o=e.weights}catch(e){return S(e?.message??String(e))}m.phase="optimizer",A.textContent="building optimizer…",await P(),u=await Y(),L(),await u.renderImage(),m.phase="textmodel",await D(e=>{A.textContent=e}),m.ready=!0,m.phase="idle",b.disabled=!1,x.disabled=!1,v.disabled=!1,R(""),T(),requestAnimationFrame(H)}b.addEventListener("click",()=>void j()),x.addEventListener("click",()=>void V()),v.addEventListener("click",()=>void X()),E.addEventListener("change",()=>void X()),y.addEventListener("keydown",e=>{"Enter"===e.key&&j()}),K().catch(e=>S(`boot failed: ${e?.message??e}`))},{"./splat/optimize":"bTsmq","./splat/feature_optimize":"kK6ZN","./splat/pixel_optimize":"ez88J","./splat/model_assets":"j8tuj"}],bTsmq:[function(e,t,r,a){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"LEGIBLE_G",()=>n),i.export(r,"LEGIBLE_INIT",()=>l),i.export(r,"LEGIBLE_LRS",()=>c),i.export(r,"DEFAULT_NUDGE_AMOUNT",()=>p),i.export(r,"nudgeSplatMask",()=>g),i.export(r,"SplatOptimizer",()=>f),i.export(r,"cosine",()=>m),i.export(r,"randomSplats",()=>h),i.export(r,"nudgeSplats",()=>y);var s=e("./raster"),d=e("./raster_wgsl"),o=e("../clip/vision"),u=e("./adam_wgsl");let n=12e3,l={scale:9,scaleJitter:.35,opacityRaw:.4,colorSpread:1.2},c={mean:1.5,logScale:.06,theta:.08,color:.12,opacity:.06},p=.18;function g(e,t=1,r=p){let a=Math.max(0,Math.min(1,r)),i=new Uint32Array(e),s=(0x85ebca6b^t)>>>0||1;for(let t=0;t<e;t++)s=Math.imul(s,1664525)+0x3c6ef35f>>>0,i[t]=+(s/0x100000000<a);return i}class f{static async create(e,t,r,a={}){let[i,d,u]=t.inputShape;if(3!==i||256!==d||256!==u)throw Error(`optimize: CLIP inputShape [${i},${d},${u}] != [3,256,256] \u{2014} the raster→CLIP copy assumes matching NCHW dims`);let l=a.G??n,c=a.cap??2048,p=await s.RasterEngine.create(e,{H:256,W:256,G:l,cap:c,bg:a.bg??[.5,.5,.5]}),g=await o.VisionTrainer.create(e,t,r);return p.setParams(a.initParams??h(l,a.seed??1,a.init)),p.zeroAdamState(),new f(e,p,g,a)}constructor(e,t,r,a){this.side=256,this.step_=0,this.device=e,this.raster=t,this.trainer=r,this.lrs=a.lrs??c,this.hyper=a.hyper??u.DEFAULT_HYPER,this.init=a.init}setPrompt(e){this.trainer.writeText(e)}step(){let e=this.device.createCommandEncoder();this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!0}),e.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackward(e),this.step_+=1,this.raster.recordAdam(e,this.step_,this.lrs,this.hyper),this.device.queue.submit([e.finish()])}get stepCount(){return this.step_}async nudge(e={}){let t=this.raster.dims.G,r=await this.raster.readParams();y(r,t,e.seed??Date.now(),e.amount??p,e.init??this.init),this.raster.setParams(r),this.raster.zeroAdamState()}async renderImage(){return this.raster.runForward(),this.raster.readImage()}async currentEmbedding(){let e=this.device.createCommandEncoder();return this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!1}),this.device.queue.submit([e.finish()]),b(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){this.raster.destroy()}}function m(e,t){let r=0,a=0,i=0;for(let s=0;s<e.length;s++)r+=e[s]*t[s],a+=e[s]*e[s],i+=t[s]*t[s];return r/Math.sqrt(a*i||1)}function h(e,t=1,r={}){let a=r.scale??l.scale,i=r.scaleJitter??l.scaleJitter,s=r.opacityRaw??l.opacityRaw,o=r.colorSpread??l.colorSpread,u=t>>>0||1,n=()=>{let e=Math.imul((u=Math.imul(u,0x2c9277b5)+0xac564b05>>>0)>>>(u>>>28)+4^u,0x108ef2d9)>>>0;return(e=(e>>>22^e)>>>0)/0x100000000},c=()=>{let e=0,t=0;for(;0===e;)e=n();for(;0===t;)t=n();return Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*t)},p=new Float32Array(e*d.PARAM_STRIDE),g=2*e,f=4*e,m=5*e,y=8*e,b=Math.log(a);for(let t=0;t<e;t++)p[0+2*t+0]=256*n(),p[0+2*t+1]=256*n(),p[g+2*t+0]=b+i*c(),p[g+2*t+1]=b+i*c(),p[f+t]=n()*Math.PI*2,p[m+3*t+0]=o*c(),p[m+3*t+1]=o*c(),p[m+3*t+2]=o*c(),p[y+t]=s;return p}function y(e,t,r=1,a=p,i={},s=g(t,r,a)){if(e.length!==t*d.PARAM_STRIDE)throw Error("nudgeSplats: wrong param length");if(s.length!==t)throw Error("nudgeSplats: wrong selection length");let o=h(t,r,i),u=[[0,2],[2*t,2],[4*t,1],[5*t,3],[8*t,1]];for(let r=0;r<t;r++)if(0!==s[r])for(let[t,a]of u){let i=t+r*a;e.set(o.subarray(i,i+a),i)}return e}async function b(e,t,r){let a=e.createBuffer({size:4*r,usage:9}),i=e.createCommandEncoder();i.copyBufferToBuffer(t,0,a,0,4*r),e.queue.submit([i.finish()]),await a.mapAsync(1);let s=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),a.destroy(),s}},{"./raster":"iCWad","./raster_wgsl":"5a6Kr","../clip/vision":"3gu6C","./adam_wgsl":"bbLCC","@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],iCWad:[function(e,t,r,a){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"RasterEngine",()=>n);var s=e("./raster_wgsl"),d=e("./adam_wgsl");let o=e=>Math.ceil(e/256);async function u(e,t,r){e.pushErrorScope("validation");let a=e.createShaderModule({code:t}),i=e.createComputePipeline({layout:"auto",compute:{module:a,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw console.error(`--- WGSL that failed (${r}) ---
${t}`),Error(`raster pipeline validation (${r}): ${s.message}`);return i}class n{constructor(e,t){if(this.adamUni=[],this.adamBind=[],this.device=e,this.dims=(0,s.resolveDims)(t),this.dims.numTiles>65535)throw Error("raster: numTiles exceeds 1D dispatch limit")}static async create(e,t){let r=new n(e,t);return await r.build(t),r}storage(e,t=0){return this.device.createBuffer({size:4*e,usage:128|t})}async build(e){let t=this.dims,r=t.G*s.PARAM_STRIDE;this.params=this.storage(r,12),this.derived=this.storage(r),this.accGrad=this.storage(r,12),this.gradRaw=this.storage(r,4),this.mBuf=this.storage(r,8),this.vBuf=this.storage(r,8),this.tileCounts=this.storage(t.numTiles,12),this.binnedIds=this.storage(t.numTiles*t.cap,4),this.tileStop=this.storage(t.numTiles,4),this.image=this.storage(3*t.H*t.W,4),this.gradImage=this.storage(3*t.H*t.W,8),this.prepPipe=await u(this.device,(0,s.prepShader)(e),"prep"),this.emitPipe=await u(this.device,(0,s.emitShader)(e),"emit"),this.fwdPipe=await u(this.device,(0,s.forwardShader)(e),"forward"),this.bwdPipe=await u(this.device,(0,s.backwardShader)(e),"backward"),this.chainPipe=await u(this.device,(0,s.chainShader)(e),"chain"),this.clearBinsPipe=await u(this.device,(0,s.clearShader)(t.numTiles),"clearBins"),this.clearGradsPipe=await u(this.device,(0,s.clearShader)(r),"clearGrads"),this.adamPipe=await u(this.device,(0,d.adamShader)(),"adam");let a=(e,t)=>this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:t.map((e,t)=>({binding:t,resource:{buffer:e}}))});for(let e of(this.prepBind=a(this.prepPipe,[this.params,this.derived]),this.emitBind=a(this.emitPipe,[this.derived,this.tileCounts,this.binnedIds]),this.fwdBind=a(this.fwdPipe,[this.tileCounts,this.binnedIds,this.derived,this.image,this.tileStop]),this.bwdBind=a(this.bwdPipe,[this.gradImage,this.tileCounts,this.binnedIds,this.tileStop,this.derived,this.accGrad]),this.chainBind=a(this.chainPipe,[this.accGrad,this.derived,this.params,this.gradRaw]),this.clearBinsBind=a(this.clearBinsPipe,[this.tileCounts]),this.clearGradsBind=a(this.clearGradsPipe,[this.accGrad]),(0,s.paramSegments)(t.G))){let e=this.device.createBuffer({size:d.ADAM_UNIFORM_BYTES,usage:72});this.adamUni.push(e),this.adamBind.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradRaw}},{binding:3,resource:{buffer:this.mBuf}},{binding:4,resource:{buffer:this.vBuf}}]}))}}setParams(e){if(e.length!==this.dims.G*s.PARAM_STRIDE)throw Error("setParams: wrong length");this.device.queue.writeBuffer(this.params,0,e)}setGradImage(e){if(e.length!==3*this.dims.H*this.dims.W)throw Error("setGradImage: wrong length");this.device.queue.writeBuffer(this.gradImage,0,e)}zeroAdamState(){let e=new Float32Array(this.dims.G*s.PARAM_STRIDE);this.device.queue.writeBuffer(this.mBuf,0,e),this.device.queue.writeBuffer(this.vBuf,0,e)}async readFloats(e,t){let r=this.device.createBuffer({size:4*t,usage:9}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(1);let i=new Float32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}async readU32(e,t){let r=this.device.createBuffer({size:4*t,usage:9}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(1);let i=new Uint32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}readImage(){return this.readFloats(this.image,3*this.dims.H*this.dims.W)}readParams(){return this.readFloats(this.params,this.dims.G*s.PARAM_STRIDE)}readGradRaw(){return this.readFloats(this.gradRaw,this.dims.G*s.PARAM_STRIDE)}async readTileTelemetry(){let e=await this.readU32(this.tileCounts,this.dims.numTiles),t=await this.readU32(this.tileStop,this.dims.numTiles),r=0,a=0,i=0,s=0,d=0,o=0;for(let u=0;u<this.dims.numTiles;u++){let n=e[u],l=t[u];r+=n,a+=l,i=Math.max(i,n),s=Math.max(s,l),n>this.dims.cap&&(d++,o+=n-this.dims.cap)}return{meanCount:r/this.dims.numTiles,maxCount:i,meanStop:a/this.dims.numTiles,maxStop:s,overflowTiles:d,overflowEntries:o}}recordForward(e){let t=this.dims,r=e.beginComputePass();r.setPipeline(this.prepPipe),r.setBindGroup(0,this.prepBind),r.dispatchWorkgroups(o(t.G)),r.setPipeline(this.clearBinsPipe),r.setBindGroup(0,this.clearBinsBind),r.dispatchWorkgroups(o(t.numTiles)),r.setPipeline(this.emitPipe),r.setBindGroup(0,this.emitBind),r.dispatchWorkgroups(o(t.G)),r.setPipeline(this.fwdPipe),r.setBindGroup(0,this.fwdBind),r.dispatchWorkgroups(t.numTiles),r.end()}recordBackward(e){let t=this.dims,r=e.beginComputePass();r.setPipeline(this.clearGradsPipe),r.setBindGroup(0,this.clearGradsBind),r.dispatchWorkgroups(o(t.G*s.DERIVED_STRIDE)),r.setPipeline(this.bwdPipe),r.setBindGroup(0,this.bwdBind),r.dispatchWorkgroups(t.numTiles),r.setPipeline(this.chainPipe),r.setBindGroup(0,this.chainBind),r.dispatchWorkgroups(o(t.G)),r.end()}recordAdam(e,t,r=d.DEFAULT_LRS,a=d.DEFAULT_HYPER){let i=(0,s.paramSegments)(this.dims.G),u={mean:r.mean,logScale:r.logScale,theta:r.theta,color:r.color,opacity:r.opacity},n=1-Math.pow(a.beta1,t),l=1-Math.pow(a.beta2,t);i.forEach((e,t)=>{let r=new ArrayBuffer(d.ADAM_UNIFORM_BYTES),i=new Uint32Array(r),s=new Float32Array(r);i[0]=e.offset,i[1]=e.length,s[2]=u[e.name],s[3]=a.beta1,s[4]=a.beta2,s[5]=a.eps,s[6]=n,s[7]=l,this.device.queue.writeBuffer(this.adamUni[t],0,r)});let c=e.beginComputePass();c.setPipeline(this.adamPipe),i.forEach((e,t)=>{c.setBindGroup(0,this.adamBind[t]),c.dispatchWorkgroups(o(e.length))}),c.end()}runForward(){let e=this.device.createCommandEncoder();this.recordForward(e),this.device.queue.submit([e.finish()])}runBackward(){let e=this.device.createCommandEncoder();this.recordBackward(e),this.device.queue.submit([e.finish()])}runAdam(e,t,r){let a=this.device.createCommandEncoder();this.recordAdam(a,e,t,r),this.device.queue.submit([a.finish()])}destroy(){for(let e of[this.params,this.derived,this.accGrad,this.gradRaw,this.mBuf,this.vBuf,this.tileCounts,this.binnedIds,this.tileStop,this.image,this.gradImage,...this.adamUni])try{e.destroy()}catch(e){}}}},{"./raster_wgsl":"5a6Kr","./adam_wgsl":"bbLCC","@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],"5a6Kr":[function(e,t,r,a){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"TILE",()=>s),i.export(r,"ALPHA_THRESHOLD",()=>d),i.export(r,"MAX_ALPHA",()=>o),i.export(r,"TRANSMITTANCE_CUTOFF",()=>u),i.export(r,"EPS",()=>n),i.export(r,"SCALE_MIN",()=>l),i.export(r,"SCALE_MAX",()=>c),i.export(r,"DERIVED_STRIDE",()=>p),i.export(r,"PARAM_STRIDE",()=>g),i.export(r,"resolveDims",()=>y),i.export(r,"prepShader",()=>x),i.export(r,"emitShader",()=>v),i.export(r,"forwardShader",()=>E),i.export(r,"backwardShader",()=>w),i.export(r,"chainShader",()=>A),i.export(r,"clearShader",()=>_),i.export(r,"paramSegments",()=>R);let s=16,d=1/255,o=.99,u=1e-4,n=1e-8,l=.3,c=64,p=9,g=9;function f(e,t){if(!e)throw Error(`raster_wgsl: ${t}`)}function m(e){f(Number.isFinite(e),`non-finite literal ${e}`);let t=e.toString();return/[.eE]/.test(t)||(t+=".0"),t}let h=e=>`${e>>>0}u`;function y(e){f(e.H>0&&e.W>0&&e.G>0,"H,W,G must be positive"),f(e.H%s==0&&e.W%s==0,`H,W must be multiples of ${s}`),f((e.cap&e.cap-1)==0&&e.cap>0,"cap must be a power of two"),f(4*e.cap<=16384,`cap*4 (${4*e.cap}B) exceeds 16KB workgroup storage`);let t=e.W/s,r=e.H/s;return{H:e.H,W:e.W,G:e.G,cap:e.cap,tilesX:t,tilesY:r,numTiles:t*r,bg:e.bg??[.5,.5,.5],gradScale:e.gradScale??65536}}function b(e){return{mean:0,logScale:2*e.G,theta:4*e.G,colorRaw:5*e.G,opacityRaw:8*e.G}}function x(e){let t=y(e),r=b(t);return`
fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@group(0) @binding(0) var<storage, read>       params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${h(t.G)}) { return; }
  let mx  = params[${h(r.mean)} + g * 2u + 0u];
  let my  = params[${h(r.mean)} + g * 2u + 1u];
  let lsx = params[${h(r.logScale)} + g * 2u + 0u];
  let lsy = params[${h(r.logScale)} + g * 2u + 1u];
  let th  = params[${h(r.theta)} + g];
  let cr0 = params[${h(r.colorRaw)} + g * 3u + 0u];
  let cr1 = params[${h(r.colorRaw)} + g * 3u + 1u];
  let cr2 = params[${h(r.colorRaw)} + g * 3u + 2u];
  let opr = params[${h(r.opacityRaw)} + g];

  let sx = clamp(exp(lsx), ${m(l)}, ${m(c)});
  let sy = clamp(exp(lsy), ${m(l)}, ${m(c)});
  let ix = 1.0 / (sx * sx);
  let iy = 1.0 / (sy * sy);
  let cs = cos(th);
  let sn = sin(th);

  let base = g * ${h(p)};
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
`}function v(e){let t=y(e);return`
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
  if (g >= ${h(t.G)}) { return; }
  let base = g * ${h(p)};
  let op = derived[base + 8u];
  if (op <= ${m(d)}) { return; }
  let ratio = max(${m(d)} / max(op, ${m(n)}), ${m(n)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let mx = derived[base + 0u]; let my = derived[base + 1u];
  let a  = derived[base + 2u]; let b  = derived[base + 3u]; let c = derived[base + 4u];
  let det = max(a * c - b * b, ${m(n)});
  let hx = sqrt(max(tau * c / det, 0.0));
  let hy = sqrt(max(tau * a / det, 0.0));
  let x0 = max(0, i32(floor(mx - hx - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(mx + hx - 0.5)));
  let y0 = max(0, i32(floor(my - hy - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(my + hy - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tx0 = x0 / ${s}; let tx1 = x1 / ${s};
  let ty0 = y0 / ${s}; let ty1 = y1 / ${s};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    let ry0 = f32(ty * ${s}) + 0.5;
    let ry1 = min(f32(${t.H-1}) + 0.5, f32((ty + 1) * ${s} - 1) + 0.5);
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let rx0 = f32(tx * ${s}) + 0.5;
      let rx1 = min(f32(${t.W-1}) + 0.5, f32((tx + 1) * ${s} - 1) + 0.5);
      if (ellipse_hit(mx, my, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        let tile = u32(ty * ${t.tilesX} + tx);
        let slot = atomicAdd(&tileCounts[tile], 1u);
        if (slot < ${h(t.cap)}) { binnedIds[tile * ${h(t.cap)} + slot] = g; }
      }
    }
  }
}
`}function E(e){let t=y(e),r=t.H*t.W;return`
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
  if (tileId >= ${h(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${h(t.cap)});
  let start = tileId * ${h(t.cap)};
  let sortN = nextPow2(count);

  // stage ids + pad to power of two with sentinel 0xffffffff (sorts to the end)
  for (var i = tid; i < sortN; i = i + 256u) {
    sh_ids[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  if (tid == 0u) { atomicStore(&sh_maxstop, 0u); }
  workgroupBarrier();

  // bitonic sort ascending \u{2014} 256-thread strided variant (v11 shape)
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

  let tileX = tileId % ${h(t.tilesX)};
  let tileY = tileId / ${h(t.tilesX)};
  let x = tileX * ${s}u + (tid % ${s}u);
  let y = tileY * ${s}u + (tid / ${s}u);
  var localStop = 0u;
  if (x < ${h(t.W)} && y < ${h(t.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var accR = 0.0; var accG = 0.0; var accB = 0.0; var T = 1.0;
    for (var i = 0u; i < count; i = i + 1u) {
      let gg = sh_ids[i];
      let b3 = gg * ${h(p)};
      let dx = pxc - derived[b3 + 0u];
      let dy = pyc - derived[b3 + 1u];
      let a  = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
      let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = derived[b3 + 8u] * exp(power);
      let alpha = min(${m(o)}, raw);
      if (alpha < ${m(d)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b3 + 5u];
      accG = accG + w * derived[b3 + 6u];
      accB = accB + w * derived[b3 + 7u];
      T = T * (1.0 - alpha);
      if (T < ${m(u)}) { break; }
    }
    let pix = y * ${h(t.W)} + x;
    image[0u * ${h(r)} + pix] = accR + T * ${m(t.bg[0])};
    image[1u * ${h(r)} + pix] = accG + T * ${m(t.bg[1])};
    image[2u * ${h(r)} + pix] = accB + T * ${m(t.bg[2])};
  }
  atomicMax(&sh_maxstop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&sh_maxstop); }
}
`}function w(e){let t=y(e),r=t.H*t.W,a=m(t.gradScale);return`
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
  if (tileId >= ${h(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${h(t.cap)});
  let stopc = min(count, tileStop[tileId]);
  let start = tileId * ${h(t.cap)};
  for (var i = tid; i < stopc; i = i + 256u) { sh_ids[i] = binnedIds[start + i]; }
  workgroupBarrier();  // only barrier; everything below is per-pixel (uniformity safe)

  let tileX = tileId % ${h(t.tilesX)};
  let tileY = tileId / ${h(t.tilesX)};
  let x = tileX * ${s}u + (tid % ${s}u);
  let y = tileY * ${s}u + (tid / ${s}u);
  if (x >= ${h(t.W)} || y >= ${h(t.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${h(t.W)} + x;
  let goR = gradImage[0u * ${h(r)} + pix];
  let goG = gradImage[1u * ${h(r)} + pix];
  let goB = gradImage[2u * ${h(r)} + pix];

  // phase A: replay to recover T_final and the stop index end_i
  var T = 1.0;
  var endi = stopc;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b3 = gg * ${h(p)};
    let dx = pxc - derived[b3 + 0u];
    let dy = pyc - derived[b3 + 1u];
    let a = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${m(o)}, derived[b3 + 8u] * exp(power));
    if (alpha < ${m(d)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${m(u)}) { endi = i + 1u; break; }
  }

  // phase B: back-to-front recurrence
  var Tcur = T;
  var gT = goR * ${m(t.bg[0])} + goG * ${m(t.bg[1])} + goB * ${m(t.bg[2])};
  for (var ii = i32(endi) - 1; ii >= 0; ii = ii - 1) {
    let gg = sh_ids[u32(ii)];
    let b3 = gg * ${h(p)};
    let dx = pxc - derived[b3 + 0u];
    let dy = pyc - derived[b3 + 1u];
    let a = derived[b3 + 2u]; let b = derived[b3 + 3u]; let c = derived[b3 + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let op = derived[b3 + 8u];
    let raw = op * exp(power);
    let alpha = min(${m(o)}, raw);
    if (alpha < ${m(d)}) { continue; }
    let denom = max(1.0 - alpha, ${m(n)});
    let Tprev = Tcur / denom;
    let cR = derived[b3 + 5u]; let cG = derived[b3 + 6u]; let cB = derived[b3 + 7u];
    let dotgc = goR * cR + goG * cG + goB * cB;
    let gAlpha = Tprev * (dotgc - gT);

    fixadd(b3, 5u, goR * Tprev * alpha);
    fixadd(b3, 6u, goG * Tprev * alpha);
    fixadd(b3, 7u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${m(o)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-(a * dx + b * dy));
    let gdy = gPower * (-(b * dx + c * dy));
    fixadd(b3, 2u, gPower * (-0.5) * dx * dx);   // g_a
    fixadd(b3, 3u, gPower * (-1.0) * dx * dy);   // g_b
    fixadd(b3, 4u, gPower * (-0.5) * dy * dy);   // g_c
    fixadd(b3, 0u, -gdx);                        // g_mean.x
    fixadd(b3, 1u, -gdy);                        // g_mean.y
    fixadd(b3, 8u, gRaw * (raw / max(op, ${m(n)})));  // g_opacity

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
  }
}
`}function A(e){let t=y(e),r=b(t),a=m(1/t.gradScale);return`
@group(0) @binding(0) var<storage, read>       accGrad : array<i32>;   // fixed-point
@group(0) @binding(1) var<storage, read>       derived : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${h(t.G)}) { return; }
  let b3 = g * ${h(p)};
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

  let lsx = params[${h(r.logScale)} + g * 2u + 0u];
  let lsy = params[${h(r.logScale)} + g * 2u + 1u];
  let th  = params[${h(r.theta)} + g];
  let ex = exp(lsx); let ey = exp(lsy);
  let sx = clamp(ex, ${m(l)}, ${m(c)});
  let sy = clamp(ey, ${m(l)}, ${m(c)});
  let gateX = select(0.0, 1.0, ex > ${m(l)} && ex < ${m(c)});
  let gateY = select(0.0, 1.0, ey > ${m(l)} && ey < ${m(c)});
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

  gradRaw[${h(r.mean)} + g * 2u + 0u] = gmx;
  gradRaw[${h(r.mean)} + g * 2u + 1u] = gmy;
  gradRaw[${h(r.logScale)} + g * 2u + 0u] = glsx;
  gradRaw[${h(r.logScale)} + g * 2u + 1u] = glsy;
  gradRaw[${h(r.theta)} + g] = gth;
  gradRaw[${h(r.colorRaw)} + g * 3u + 0u] = gc0 * col0 * (1.0 - col0);
  gradRaw[${h(r.colorRaw)} + g * 3u + 1u] = gc1 * col1 * (1.0 - col1);
  gradRaw[${h(r.colorRaw)} + g * 3u + 2u] = gc2 * col2 * (1.0 - col2);
  gradRaw[${h(r.opacityRaw)} + g] = gop * opv * (1.0 - opv);
}
`}function _(e){return`
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= ${h(e)}) { return; }
  buf[gid.x] = 0u;
}
`}function R(e){return[{name:"mean",offset:0,length:2*e},{name:"logScale",offset:2*e,length:2*e},{name:"theta",offset:4*e,length:e},{name:"color",offset:5*e,length:3*e},{name:"opacity",offset:8*e,length:e}]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],gKN0c:[function(e,t,r,a){r.interopDefault=function(e){return e&&e.__esModule?e:{default:e}},r.defineInteropFlag=function(e){Object.defineProperty(e,"__esModule",{value:!0})},r.exportAll=function(e,t){return Object.keys(e).forEach(function(r){"default"===r||"__esModule"===r||Object.prototype.hasOwnProperty.call(t,r)||Object.defineProperty(t,r,{enumerable:!0,get:function(){return e[r]}})}),t},r.export=function(e,t,r){Object.defineProperty(e,t,{enumerable:!0,get:r})}},{}],kK6ZN:[function(e,t,r,a){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"FEATURE_PAINTER_G",()=>l),i.export(r,"FEATURE_PAINTER_INIT",()=>c),i.export(r,"FEATURE_PAINTER_LRS",()=>p),i.export(r,"FeaturePainterOptimizer",()=>g),i.export(r,"randomFeatures",()=>f),i.export(r,"randomDecoder",()=>m),i.export(r,"cosine",()=>n.cosine);var s=e("../clip/vision"),d=e("./feature_painter_wgsl"),o=e("./feature_painter"),u=e("./adam_wgsl"),n=e("./optimize");let l=2048,c={scale:7,scaleJitter:.45,opacityRaw:-.1,colorSpread:1.1},p={...n.LEGIBLE_LRS};class g{static async create(e,t,r,a={}){let[i,d,u]=t.inputShape;if(3!==i||256!==d||256!==u)throw Error("feature painter requires MobileCLIP 256x256 RGB input");let p=a.G??l,h=await o.FeaturePainterEngine.create(e,{H:256,W:256,G:p,cap:2048,bg:[.5,.5,.5]}),y=await s.VisionTrainer.create(e,t,r),b=a.init??c;return h.setParams((0,n.randomSplats)(p,a.seed??1,b)),h.setFeatureParams(f(p,a.seed??1)),h.setDecoderParams(m(a.seed??1)),h.zeroAdamState(),new g(e,h,y,b,a)}constructor(e,t,r,a,i){this.side=256,this.step_=0,this.device=e,this.raster=t,this.trainer=r,this.init=a,this.lrs=i.lrs??p,this.hyper=i.hyper??u.DEFAULT_HYPER,this.featureLR=i.featureLR??o.FEATURE_LR,this.decoderLR=i.decoderLR??o.DECODER_LR}setPrompt(e){this.trainer.writeText(e)}get stepCount(){return this.step_}step(){let e=this.device.createCommandEncoder();this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!0}),e.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackward(e),this.step_+=1,this.raster.recordAdam(e,this.step_,this.lrsForStep(),this.hyper,void 0,{feature:this.featureLR,decoder:this.decoderLR}),this.device.queue.submit([e.finish()])}lrsForStep(){let e=Math.max(0,Math.min(1,this.step_/180)),t=1.6-.6*e,r=.55+.45*e;return{mean:this.lrs.mean*t,logScale:this.lrs.logScale*t,theta:this.lrs.theta*t,color:this.lrs.color*r,opacity:this.lrs.opacity*r}}async nudge(e={}){let t=this.raster.dims.G,r=e.seed??Date.now(),a=e.amount??.12,i=(0,n.nudgeSplatMask)(t,r,a),[s,o]=await Promise.all([this.raster.readParams(),this.raster.readFeatureParams()]);(0,n.nudgeSplats)(s,t,r,a,e.init??this.init,i);for(let e=0;e<t;e++)0!==i[e]&&o.fill(0,e*d.FEATURE_STRIDE,(e+1)*d.FEATURE_STRIDE);this.raster.setParams(s),this.raster.setFeatureParams(o),this.raster.zeroAdamState()}async renderImage(){return this.raster.runForward(),this.raster.readImage()}async currentEmbedding(){let e=this.device.createCommandEncoder();return this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!1}),this.device.queue.submit([e.finish()]),h(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){this.raster.destroy()}}function f(e,t=1){return new Float32Array(e*d.FEATURE_STRIDE)}function m(e=1){let t=e>>>0||1,r=()=>{let e=Math.imul((t=Math.imul(t,0x2c9277b5)+0xac564b05>>>0)>>>(t>>>28)+4^t,0x108ef2d9)>>>0;return(e=(e>>>22^e)>>>0)/0x100000000},a=()=>{let e=0,t=0;for(;0===e;)e=r();for(;0===t;)t=r();return Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*t)},i=new Float32Array(d.DECODER_PARAM_COUNT);for(let e=0;e<3;e++)for(let t=0;t<d.FEATURE_LATENT_CHANNELS;t++)i[8*e+3+t]=.25*a();return i}async function h(e,t,r){let a=e.createBuffer({size:4*r,usage:9}),i=e.createCommandEncoder();i.copyBufferToBuffer(t,0,a,0,4*r),e.queue.submit([i.finish()]),await a.mapAsync(1);let s=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),a.destroy(),s}},{"../clip/vision":"3gu6C","./feature_painter_wgsl":"2sBBA","./feature_painter":"bMW7z","./adam_wgsl":"bbLCC","./optimize":"bTsmq","@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],"2sBBA":[function(e,t,r,a){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"FEATURE_LATENT_CHANNELS",()=>d),i.export(r,"FEATURE_STRIDE",()=>o),i.export(r,"FEATURE_DIM",()=>u),i.export(r,"DECODER_PARAM_COUNT",()=>n),i.export(r,"DECODER_RESIDUAL_SCALE",()=>l),i.export(r,"FEATURE_STATE_STRIDE",()=>c),i.export(r,"FEATURE_ACC_DERIVED_OFFSET",()=>p),i.export(r,"FEATURE_ACC_EXTRA_OFFSET",()=>g),i.export(r,"FEATURE_ACC_LOCAL_RAW_OFFSET",()=>f),i.export(r,"FEATURE_ACC_STRIDE",()=>m),i.export(r,"featurePrepShader",()=>x),i.export(r,"featureEmitShader",()=>v),i.export(r,"featureForwardShader",()=>E),i.export(r,"featureBackwardShader",()=>w),i.export(r,"featureGeometryChainShader",()=>A),i.export(r,"featureChainShader",()=>_),i.export(r,"FEATURE_CHAIN_WORK_ITEMS",()=>R);var s=e("./raster_wgsl");let d=5,o=15,u=8,n=27,l=.1,c=15,p=0,g=9,f=24,m=29,h=e=>/[.eE]/.test(String(e))?String(e):`${e}.0`,y=e=>`${e>>>0}u`;function b(e){let t=(0,s.resolveDims)(e),r=t.G*m;return{d:t,hw:t.H*t.W,decoderOffset:r,code:`
const STATE_STRIDE : u32 = ${y(c)};
const FEATURE_STRIDE : u32 = ${y(o)};
const FEATURE_DIM : u32 = ${y(u)};
const ACC_EXTRA_OFFSET : u32 = ${y(g)};
const ACC_LOCAL_RAW_OFFSET : u32 = ${y(f)};
const ACC_STRIDE : u32 = ${y(m)};
const DECODER_OFFSET : u32 = ${y(r)};
const RESIDUAL_SCALE : f32 = ${h(l)};

fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn logit1(x : f32) -> f32 { let y = clamp(x, ${h(s.EPS)}, ${h(1-s.EPS)}); return log(y / (1.0 - y)); }
fn fixadd(dst : ptr<storage, array<atomic<i32>>, read_write>, index : u32, v : f32) {
  atomicAdd(&(*dst)[index], i32(clamp(round(v * ${h(t.gradScale)}), -2.14e9, 2.14e9)));
}
`}}function x(e){let{d:t,code:r}=b(e),a=2*t.G,i=4*t.G,d=5*t.G,o=8*t.G;return`
${r}
@group(0) @binding(0) var<storage, read> params : array<f32>;
@group(0) @binding(1) var<storage, read_write> state : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${y(t.G)}) { return; }
  let mx = params[${y(0)} + g * 2u];
  let my = params[${y(0)} + g * 2u + 1u];
  let lsx = params[${y(a)} + g * 2u];
  let lsy = params[${y(a)} + g * 2u + 1u];
  let th = params[${y(i)} + g];
  let ex = exp(lsx);
  let ey = exp(lsy);
  let sx = clamp(ex, ${h(s.SCALE_MIN)}, ${h(s.SCALE_MAX)});
  let sy = clamp(ey, ${h(s.SCALE_MIN)}, ${h(s.SCALE_MAX)});
  let invSx = 1.0 / sx;
  let invSy = 1.0 / sy;
  let ix = invSx * invSx;
  let iy = invSy * invSy;
  let cs = cos(th);
  let sn = sin(th);
  let b = g * STATE_STRIDE;
  state[b + ${y(0)}] = mx;
  state[b + ${y(1)}] = my;
  state[b + ${y(2)}] = cs * cs * ix + sn * sn * iy;
  state[b + ${y(3)}] = cs * sn * (ix - iy);
  state[b + ${y(4)}] = sn * sn * ix + cs * cs * iy;
  state[b + ${y(5)}] = sigmoid1(params[${y(d)} + g * 3u]);
  state[b + ${y(6)}] = sigmoid1(params[${y(d)} + g * 3u + 1u]);
  state[b + ${y(7)}] = sigmoid1(params[${y(d)} + g * 3u + 2u]);
  state[b + ${y(8)}] = sigmoid1(params[${y(o)} + g]);
  state[b + ${y(9)}] = cs;
  state[b + ${y(10)}] = sn;
  state[b + ${y(11)}] = invSx;
  state[b + ${y(12)}] = invSy;
  state[b + ${y(13)}] = select(0.0, 1.0, ex > ${h(s.SCALE_MIN)} && ex < ${h(s.SCALE_MAX)});
  state[b + ${y(14)}] = select(0.0, 1.0, ey > ${h(s.SCALE_MIN)} && ey < ${h(s.SCALE_MAX)});
}
`}function v(e){let{d:t,code:r}=b(e);return`
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
  if (g >= ${y(t.G)}) { return; }
  let s = g * STATE_STRIDE;
  let opacity = state[s + ${y(8)}];
  if (opacity <= ${h(s.ALPHA_THRESHOLD)}) { return; }
  let ratio = max(${h(s.ALPHA_THRESHOLD)} / max(opacity, ${h(s.EPS)}), ${h(s.EPS)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }
  let mx = state[s + ${y(0)}];
  let my = state[s + ${y(1)}];
  let a = state[s + ${y(2)}];
  let b = state[s + ${y(3)}];
  let c = state[s + ${y(4)}];
  let det = max(a * c - b * b, ${h(s.EPS)});
  let hx = sqrt(max(tau * c / det, 0.0));
  let hy = sqrt(max(tau * a / det, 0.0));
  let x0 = max(0, i32(floor(mx - hx - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(mx + hx - 0.5)));
  let y0 = max(0, i32(floor(my - hy - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(my + hy - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }
  let tx0 = x0 / ${s.TILE}; let tx1 = x1 / ${s.TILE};
  let ty0 = y0 / ${s.TILE}; let ty1 = y1 / ${s.TILE};
  for (var ty = ty0; ty <= ty1; ty++) {
    let ry0 = f32(ty * ${s.TILE}) + 0.5;
    let ry1 = min(f32(${t.H-1}) + 0.5, f32((ty + 1) * ${s.TILE} - 1) + 0.5);
    for (var tx = tx0; tx <= tx1; tx++) {
      let rx0 = f32(tx * ${s.TILE}) + 0.5;
      let rx1 = min(f32(${t.W-1}) + 0.5, f32((tx + 1) * ${s.TILE} - 1) + 0.5);
      if (ellipseHit(mx, my, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        let tile = u32(ty * ${t.tilesX} + tx);
        let slot = atomicAdd(&tileCounts[tile], 1u);
        if (slot < ${y(t.cap)}) { binnedIds[tile * ${y(t.cap)} + slot] = g; }
      }
    }
  }
}
`}function E(e){let{d:t,hw:r,code:a}=b(e);return`
${a}
@group(0) @binding(0) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds : array<u32>;
@group(0) @binding(2) var<storage, read> state : array<f32>;
@group(0) @binding(3) var<storage, read> features : array<f32>;
@group(0) @binding(4) var<storage, read> decoder : array<f32>;
@group(0) @binding(5) var<storage, read_write> image : array<f32>;
@group(0) @binding(6) var<storage, read_write> tileStop : array<u32>;

var<workgroup> shIds : array<u32, ${y(t.cap)}>;
var<workgroup> shMaxStop : atomic<u32>;
fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u) - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  if (tileId >= ${y(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${y(t.cap)});
  let start = tileId * ${y(t.cap)};
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

  let tileX = tileId % ${y(t.tilesX)};
  let tileY = tileId / ${y(t.tilesX)};
  let x = tileX * ${y(s.TILE)} + (tid % ${y(s.TILE)});
  let y = tileY * ${y(s.TILE)} + (tid / ${y(s.TILE)});
  var localStop = 0u;
  if (x < ${y(t.W)} && y < ${y(t.H)}) {
    let pxc = f32(x) + 0.5;
    let pyc = f32(y) + 0.5;
    var baseR = 0.0; var baseG = 0.0; var baseB = 0.0;
    var l0 = 0.0; var l1 = 0.0; var l2 = 0.0; var l3 = 0.0; var l4 = 0.0;
    var T = 1.0;
    for (var i = 0u; i < count; i++) {
      let g = shIds[i]; let s = g * STATE_STRIDE;
      let dx = pxc - state[s + ${y(0)}];
      let dy = pyc - state[s + ${y(1)}];
      let a = state[s + ${y(2)}];
      let b = state[s + ${y(3)}];
      let c = state[s + ${y(4)}];
      let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let raw = state[s + ${y(8)}] * exp(power);
      let alpha = min(${h(s.MAX_ALPHA)}, raw);
      if (alpha < ${h(s.ALPHA_THRESHOLD)}) { continue; }
      let cs = state[s + ${y(9)}]; let sn = state[s + ${y(10)}];
      let ux = clamp((cs * dx + sn * dy) * state[s + ${y(11)}], -3.0, 3.0);
      let uy = clamp((-sn * dx + cs * dy) * state[s + ${y(12)}], -3.0, 3.0);
      let e = g * FEATURE_STRIDE;
      let w = T * alpha;
      baseR += w * state[s + ${y(5)}];
      baseG += w * state[s + ${y(6)}];
      baseB += w * state[s + ${y(7)}];
      l0 += w * (features[e] + ux * features[e + 5u] + uy * features[e + 10u]);
      l1 += w * (features[e + 1u] + ux * features[e + 6u] + uy * features[e + 11u]);
      l2 += w * (features[e + 2u] + ux * features[e + 7u] + uy * features[e + 12u]);
      l3 += w * (features[e + 3u] + ux * features[e + 8u] + uy * features[e + 13u]);
      l4 += w * (features[e + 4u] + ux * features[e + 9u] + uy * features[e + 14u]);
      T *= 1.0 - alpha;
      if (T < ${h(s.TRANSMITTANCE_CUTOFF)}) { break; }
    }
    baseR += T * ${h(t.bg[0])};
    baseG += T * ${h(t.bg[1])};
    baseB += T * ${h(t.bg[2])};
    let rR = decoder[24u] + decoder[0u] * baseR + decoder[1u] * baseG + decoder[2u] * baseB + decoder[3u] * l0 + decoder[4u] * l1 + decoder[5u] * l2 + decoder[6u] * l3 + decoder[7u] * l4;
    let rG = decoder[25u] + decoder[8u] * baseR + decoder[9u] * baseG + decoder[10u] * baseB + decoder[11u] * l0 + decoder[12u] * l1 + decoder[13u] * l2 + decoder[14u] * l3 + decoder[15u] * l4;
    let rB = decoder[26u] + decoder[16u] * baseR + decoder[17u] * baseG + decoder[18u] * baseB + decoder[19u] * l0 + decoder[20u] * l1 + decoder[21u] * l2 + decoder[22u] * l3 + decoder[23u] * l4;
    let pixel = y * ${y(t.W)} + x;
    image[pixel] = sigmoid1(logit1(baseR) + RESIDUAL_SCALE * rR);
    image[${y(r)} + pixel] = sigmoid1(logit1(baseG) + RESIDUAL_SCALE * rG);
    image[${y(2*r)} + pixel] = sigmoid1(logit1(baseB) + RESIDUAL_SCALE * rB);
  }
  atomicMax(&shMaxStop, localStop);
  workgroupBarrier();
  if (tid == 0u) { tileStop[tileId] = atomicLoad(&shMaxStop); }
}
`}function w(e){let{d:t,hw:r,code:a}=b(e),i=s.TILE*s.TILE/64;return`
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
  return i32(clamp(round(v * ${h(t.gradScale)}), -2.14e9, 2.14e9));
}

// A 16x16 tile has 256 pixels. Each lane owns a compile-time number of pixels,
// so state and feature payloads are loaded once per lane/splat and reductions
// never need a workgroup barrier. Every subgroup leader writes one partial tile
// sum to the fixed-point buffer.
const PIXELS_PER_LANE : u32 = ${y(i)};

@compute @workgroup_size(${y(64)})
fn main(
  @builtin(workgroup_id) wg : vec3u,
  @builtin(local_invocation_index) tid : u32,
  @builtin(subgroup_invocation_id) lane : u32,
) {
  let tileId = wg.x;
  if (tileId >= ${y(t.numTiles)}) { return; }
  let count = min(tileCounts[tileId], ${y(t.cap)});
  let stopc = min(count, tileStop[tileId]);
  let start = tileId * ${y(t.cap)};
  let tileX = tileId % ${y(t.tilesX)};
  let tileY = tileId / ${y(t.tilesX)};
  var valid : array<bool, ${y(i)}>;
  var pixel : array<u32, ${y(i)}>;
  var pxc : array<f32, ${y(i)}>;
  var pyc : array<f32, ${y(i)}>;
  var endi : array<u32, ${y(i)}>;
  var baseR : array<f32, ${y(i)}>; var baseG : array<f32, ${y(i)}>; var baseB : array<f32, ${y(i)}>;
  var l0 : array<f32, ${y(i)}>; var l1 : array<f32, ${y(i)}>; var l2 : array<f32, ${y(i)}>; var l3 : array<f32, ${y(i)}>; var l4 : array<f32, ${y(i)}>;
  var T : array<f32, ${y(i)}>;
  var gBaseR : array<f32, ${y(i)}>; var gBaseG : array<f32, ${y(i)}>; var gBaseB : array<f32, ${y(i)}>;
  var gL0 : array<f32, ${y(i)}>; var gL1 : array<f32, ${y(i)}>; var gL2 : array<f32, ${y(i)}>; var gL3 : array<f32, ${y(i)}>; var gL4 : array<f32, ${y(i)}>;
  var gT : array<f32, ${y(i)}>;
  for (var p = 0u; p < PIXELS_PER_LANE; p++) {
    let localPixel = tid + ${y(64)} * p;
    let x = tileX * ${y(s.TILE)} + (localPixel % ${y(s.TILE)});
    let y = tileY * ${y(s.TILE)} + (localPixel / ${y(s.TILE)});
    valid[p] = x < ${y(t.W)} && y < ${y(t.H)};
    pixel[p] = y * ${y(t.W)} + x;
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
    let mx = state[s + ${y(0)}]; let my = state[s + ${y(1)}];
    let a = state[s + ${y(2)}]; let b = state[s + ${y(3)}]; let c = state[s + ${y(4)}];
    let opacity = state[s + ${y(8)}]; let cs = state[s + ${y(9)}]; let sn = state[s + ${y(10)}];
    let invSx = state[s + ${y(11)}]; let invSy = state[s + ${y(12)}];
    let cR = state[s + ${y(5)}]; let cG = state[s + ${y(6)}]; let cB = state[s + ${y(7)}];
    let f0 = features[e]; let f1 = features[e + 1u]; let f2 = features[e + 2u]; let f3 = features[e + 3u]; let f4 = features[e + 4u];
    let fx0 = features[e + 5u]; let fx1 = features[e + 6u]; let fx2 = features[e + 7u]; let fx3 = features[e + 8u]; let fx4 = features[e + 9u];
    let fy0 = features[e + 10u]; let fy1 = features[e + 11u]; let fy2 = features[e + 12u]; let fy3 = features[e + 13u]; let fy4 = features[e + 14u];
    for (var p = 0u; p < PIXELS_PER_LANE; p++) {
      if (valid[p] && i < endi[p]) {
        let dx = pxc[p] - mx; let dy = pyc[p] - my;
        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
        if (power <= 0.0) {
          let alpha = min(${h(s.MAX_ALPHA)}, opacity * exp(power));
          if (alpha >= ${h(s.ALPHA_THRESHOLD)}) {
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
            if (T[p] < ${h(s.TRANSMITTANCE_CUTOFF)}) { endi[p] = i + 1u; }
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
      baseR[p] += T[p] * ${h(t.bg[0])}; baseG[p] += T[p] * ${h(t.bg[1])}; baseB[p] += T[p] * ${h(t.bg[2])};
      let goR = gradImage[pixel[p]]; let goG = gradImage[${y(r)} + pixel[p]]; let goB = gradImage[${y(2*r)} + pixel[p]];
      let rR = decoder[24u] + decoder[0u] * baseR[p] + decoder[1u] * baseG[p] + decoder[2u] * baseB[p] + decoder[3u] * l0[p] + decoder[4u] * l1[p] + decoder[5u] * l2[p] + decoder[6u] * l3[p] + decoder[7u] * l4[p];
      let rG = decoder[25u] + decoder[8u] * baseR[p] + decoder[9u] * baseG[p] + decoder[10u] * baseB[p] + decoder[11u] * l0[p] + decoder[12u] * l1[p] + decoder[13u] * l2[p] + decoder[14u] * l3[p] + decoder[15u] * l4[p];
      let rB = decoder[26u] + decoder[16u] * baseR[p] + decoder[17u] * baseG[p] + decoder[18u] * baseB[p] + decoder[19u] * l0[p] + decoder[20u] * l1[p] + decoder[21u] * l2[p] + decoder[22u] * l3[p] + decoder[23u] * l4[p];
      let outR = sigmoid1(logit1(baseR[p]) + RESIDUAL_SCALE * rR);
      let outG = sigmoid1(logit1(baseG[p]) + RESIDUAL_SCALE * rG);
      let outB = sigmoid1(logit1(baseB[p]) + RESIDUAL_SCALE * rB);
      let dzR = goR * outR * (1.0 - outR); let dzG = goG * outG * (1.0 - outG); let dzB = goB * outB * (1.0 - outB);
      let baseRc = clamp(baseR[p], ${h(s.EPS)}, ${h(1-s.EPS)}); let baseGc = clamp(baseG[p], ${h(s.EPS)}, ${h(1-s.EPS)}); let baseBc = clamp(baseB[p], ${h(s.EPS)}, ${h(1-s.EPS)});
      gBaseR[p] = dzR / max(baseRc * (1.0 - baseRc), ${h(s.EPS)}) + RESIDUAL_SCALE * (dzR * decoder[0u] + dzG * decoder[8u] + dzB * decoder[16u]);
      gBaseG[p] = dzG / max(baseGc * (1.0 - baseGc), ${h(s.EPS)}) + RESIDUAL_SCALE * (dzR * decoder[1u] + dzG * decoder[9u] + dzB * decoder[17u]);
      gBaseB[p] = dzB / max(baseBc * (1.0 - baseBc), ${h(s.EPS)}) + RESIDUAL_SCALE * (dzR * decoder[2u] + dzG * decoder[10u] + dzB * decoder[18u]);
      gL0[p] = RESIDUAL_SCALE * (dzR * decoder[3u] + dzG * decoder[11u] + dzB * decoder[19u]);
      gL1[p] = RESIDUAL_SCALE * (dzR * decoder[4u] + dzG * decoder[12u] + dzB * decoder[20u]);
      gL2[p] = RESIDUAL_SCALE * (dzR * decoder[5u] + dzG * decoder[13u] + dzB * decoder[21u]);
      gL3[p] = RESIDUAL_SCALE * (dzR * decoder[6u] + dzG * decoder[14u] + dzB * decoder[22u]);
      gL4[p] = RESIDUAL_SCALE * (dzR * decoder[7u] + dzG * decoder[15u] + dzB * decoder[23u]);
      gT[p] = gBaseR[p] * ${h(t.bg[0])} + gBaseG[p] * ${h(t.bg[1])} + gBaseB[p] * ${h(t.bg[2])};
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
    let mx = state[s + ${y(0)}]; let my = state[s + ${y(1)}];
    let a = state[s + ${y(2)}]; let b = state[s + ${y(3)}]; let c = state[s + ${y(4)}];
    let opacity = state[s + ${y(8)}]; let cs = state[s + ${y(9)}]; let sn = state[s + ${y(10)}];
    let invSx = state[s + ${y(11)}]; let invSy = state[s + ${y(12)}];
    let cR = state[s + ${y(5)}]; let cG = state[s + ${y(6)}]; let cB = state[s + ${y(7)}];
    let scaleGateX = state[s + ${y(13)}]; let scaleGateY = state[s + ${y(14)}];
    let f0 = features[e]; let f1 = features[e + 1u]; let f2 = features[e + 2u]; let f3 = features[e + 3u]; let f4 = features[e + 4u];
    let fx0 = features[e + 5u]; let fx1 = features[e + 6u]; let fx2 = features[e + 7u]; let fx3 = features[e + 8u]; let fx4 = features[e + 9u];
    let fy0 = features[e + 10u]; let fy1 = features[e + 11u]; let fy2 = features[e + 12u]; let fy3 = features[e + 13u]; let fy4 = features[e + 14u];
    for (var p = 0u; p < PIXELS_PER_LANE; p++) {
      if (valid[p] && u32(ii) < endi[p]) {
        let dx = pxc[p] - mx; let dy = pyc[p] - my;
        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
        if (power <= 0.0) {
          let raw = opacity * exp(power); let alpha = min(${h(s.MAX_ALPHA)}, raw);
          if (alpha >= ${h(s.ALPHA_THRESHOLD)}) {
            let denom = max(1.0 - alpha, ${h(s.EPS)}); let Tprev = T[p] / denom;
            let uxRaw = (cs * dx + sn * dy) * invSx; let uyRaw = (-sn * dx + cs * dy) * invSy;
            let ux = clamp(uxRaw, -3.0, 3.0); let uy = clamp(uyRaw, -3.0, 3.0);
            let z0 = f0 + ux * fx0 + uy * fy0; let z1 = f1 + ux * fx1 + uy * fy1; let z2 = f2 + ux * fx2 + uy * fy2;
            let z3 = f3 + ux * fx3 + uy * fy3; let z4 = f4 + ux * fx4 + uy * fy4;
            let dotPayload = gBaseR[p] * cR + gBaseG[p] * cG + gBaseB[p] * cB + gL0[p] * z0 + gL1[p] * z1 + gL2[p] * z2 + gL3[p] * z3 + gL4[p] * z4;
            let gAlpha = Tprev * (dotPayload - gT[p]); let w = Tprev * alpha;
            let gUx = select(0.0, w * (gL0[p] * fx0 + gL1[p] * fx1 + gL2[p] * fx2 + gL3[p] * fx3 + gL4[p] * fx4), uxRaw > -3.0 && uxRaw < 3.0);
            let gUy = select(0.0, w * (gL0[p] * fy0 + gL1[p] * fy1 + gL2[p] * fy2 + gL3[p] * fy3 + gL4[p] * fy4), uyRaw > -3.0 && uyRaw < 3.0);
            let gRaw = gAlpha * select(0.0, 1.0, raw < ${h(s.MAX_ALPHA)}); let gPower = gRaw * raw;
            let gdx = gPower * (-(a * dx + b * dy)); let gdy = gPower * (-(b * dx + c * dy));
            v0 += vec4<f32>(-gdx, -gdy, gPower * (-0.5) * dx * dx, gPower * (-1.0) * dx * dy);
            v1 += vec4<f32>(gPower * (-0.5) * dy * dy, gBaseR[p] * w, gBaseG[p] * w, gBaseB[p] * w);
            v2 += vec4<f32>(gRaw * (raw / max(opacity, ${h(s.EPS)})), gL0[p] * w, gL1[p] * w, gL2[p] * w);
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
`}function A(e){let{d:t,code:r}=b(e),a=2*t.G,i=4*t.G,s=5*t.G,d=8*t.G;return`
${r}
@group(0) @binding(0) var<storage, read> acc : array<i32>;
@group(0) @binding(1) var<storage, read> state : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${y(t.G)}) { return; }
  let ab = g * ACC_STRIDE;
  let sb = g * STATE_STRIDE;
  let inv = ${h(1/t.gradScale)};
  let gmx = f32(acc[ab]) * inv;
  let gmy = f32(acc[ab + 1u]) * inv;
  let gA = f32(acc[ab + 2u]) * inv;
  let gB = f32(acc[ab + 3u]) * inv;
  let gC = f32(acc[ab + 4u]) * inv;
  let gc0 = f32(acc[ab + 5u]) * inv;
  let gc1 = f32(acc[ab + 6u]) * inv;
  let gc2 = f32(acc[ab + 7u]) * inv;
  let gop = f32(acc[ab + 8u]) * inv;
  let invSx = state[sb + ${y(11)}]; let invSy = state[sb + ${y(12)}];
  let ix = invSx * invSx; let iy = invSy * invSy;
  let cs = state[sb + ${y(9)}]; let sn = state[sb + ${y(10)}];
  let gix = gA * cs * cs + gB * cs * sn + gC * sn * sn;
  let giy = gA * sn * sn - gB * cs * sn + gC * cs * cs;
  let glsx = gix * (-2.0 * ix) * state[sb + ${y(13)}];
  let glsy = giy * (-2.0 * iy) * state[sb + ${y(14)}];
  let gth = (ix - iy) * ((cs * cs - sn * sn) * gB + 2.0 * cs * sn * (gC - gA));
  let lmx = f32(acc[ab + ACC_LOCAL_RAW_OFFSET]) * inv;
  let lmy = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 1u]) * inv;
  let llsx = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 2u]) * inv;
  let llsy = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 3u]) * inv;
  let lth = f32(acc[ab + ACC_LOCAL_RAW_OFFSET + 4u]) * inv;
  let color0 = state[sb + ${y(5)}]; let color1 = state[sb + ${y(6)}]; let color2 = state[sb + ${y(7)}];
  let opacity = state[sb + ${y(8)}];
  gradRaw[${y(0)} + g * 2u] = gmx + lmx;
  gradRaw[${y(0)} + g * 2u + 1u] = gmy + lmy;
  gradRaw[${y(a)} + g * 2u] = glsx + llsx;
  gradRaw[${y(a)} + g * 2u + 1u] = glsy + llsy;
  gradRaw[${y(i)} + g] = gth + lth;
  gradRaw[${y(s)} + g * 3u] = gc0 * color0 * (1.0 - color0);
  gradRaw[${y(s)} + g * 3u + 1u] = gc1 * color1 * (1.0 - color1);
  gradRaw[${y(s)} + g * 3u + 2u] = gc2 * color2 * (1.0 - color2);
  gradRaw[${y(d)} + g] = gop * opacity * (1.0 - opacity);
}
`}function _(e){let{d:t,code:r}=b(e);return`
${r}
@group(0) @binding(0) var<storage, read> acc : array<i32>;
@group(0) @binding(1) var<storage, read_write> gradFeatures : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradDecoder : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  let inv = ${h(1/t.gradScale)};
  if (i < ${y(t.G)}) {
    let g = i;
    let ab = g * ACC_STRIDE + ACC_EXTRA_OFFSET;
    let fb = g * FEATURE_STRIDE;
    for (var ch = 0u; ch < FEATURE_STRIDE; ch++) { gradFeatures[fb + ch] = f32(acc[ab + ch]) * inv; }
  }
  if (i < ${y(n)}) { gradDecoder[i] = f32(acc[DECODER_OFFSET + i]) * inv; }
}
`}let R=e=>Math.max((0,s.resolveDims)(e).G,n)},{"./raster_wgsl":"5a6Kr","@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],bMW7z:[function(e,t,r,a){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"FEATURE_LR",()=>n),i.export(r,"DECODER_LR",()=>l),i.export(r,"FeaturePainterEngine",()=>h);var s=e("./adam_wgsl"),d=e("./raster_wgsl"),o=e("./feature_painter_wgsl");let u=e=>Math.ceil(e/256),n=.025,l=.03,c=[{offset:3,length:5},{offset:11,length:5},{offset:19,length:5}];function p(e,t){return t?e.beginComputePass({timestampWrites:t}):e.beginComputePass()}async function g(e,t,r){e.pushErrorScope("validation");let a=e.createShaderModule({label:r,code:t}),i=e.createComputePipeline({label:r,layout:"auto",compute:{module:a,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw Error(`feature painter ${r}: ${s.message}`);return i}function f(e){return e.createBuffer({size:s.ADAM_UNIFORM_BYTES,usage:72})}function m(e,t,r,a,i,s,d){let o=new ArrayBuffer(32),u=new Uint32Array(o),n=new Float32Array(o);u[0]=r,u[1]=a,n[2]=i,n[3]=d.beta1,n[4]=d.beta2,n[5]=d.eps,n[6]=1-Math.pow(d.beta1,s),n[7]=1-Math.pow(d.beta2,s),e.queue.writeBuffer(t,0,o)}class h{constructor(e,t){this.geometryAdamUniforms=[],this.geometryAdamGroups=[],this.decoderAdamUniforms=[],this.decoderAdamGroups=[],this.device=e,this.dims=(0,d.resolveDims)(t);let r=this.dims,a=(t,r=0)=>e.createBuffer({size:4*t,usage:128|r}),i=r.G*d.PARAM_STRIDE,s=r.G*o.FEATURE_STRIDE,u=r.G*o.FEATURE_ACC_STRIDE+o.DECODER_PARAM_COUNT;this.params=a(i,12),this.featureParams=a(s,12),this.decoderParams=a(o.DECODER_PARAM_COUNT,12),this.image=a(3*r.H*r.W,4),this.gradImage=a(3*r.H*r.W,8),this.state=a(r.G*o.FEATURE_STATE_STRIDE),this.tileCounts=a(r.numTiles,12),this.binnedIds=a(r.numTiles*r.cap),this.tileStop=a(r.numTiles,4),this.acc=a(u,8),this.gradGeom=a(i,4),this.geomM=a(i,8),this.geomV=a(i,8),this.gradFeature=a(s,4),this.featureM=a(s,8),this.featureV=a(s,8),this.gradDecoder=a(o.DECODER_PARAM_COUNT,4),this.decoderM=a(o.DECODER_PARAM_COUNT,8),this.decoderV=a(o.DECODER_PARAM_COUNT,8),this.featureAdamUniform=f(e),this.geometrySegments=[{offset:0,length:2*r.G,lr:0},{offset:2*r.G,length:2*r.G,lr:0},{offset:4*r.G,length:r.G,lr:0},{offset:5*r.G,length:3*r.G,lr:0},{offset:8*r.G,length:r.G,lr:0}]}static async create(e,t){if(!e.features.has("subgroups"))throw Error("Feature Painter requires the WebGPU subgroups feature on this build");let r=new h(e,t);return await r.build(t),r}async build(e){let t=this.dims;this.prepPipe=await g(this.device,(0,o.featurePrepShader)(e),"feature8-prep"),this.emitPipe=await g(this.device,(0,o.featureEmitShader)(e),"feature8-emit"),this.forwardPipe=await g(this.device,(0,o.featureForwardShader)(e),"feature8-forward"),this.backwardPipe=await g(this.device,(0,o.featureBackwardShader)(e),"feature8-backward-subgroups"),this.geometryChainPipe=await g(this.device,(0,o.featureGeometryChainShader)(e),"feature8-geometry-chain"),this.featureChainPipe=await g(this.device,(0,o.featureChainShader)(e),"feature8-feature-chain"),this.clearBinsPipe=await g(this.device,(0,d.clearShader)(t.numTiles),"feature8-clear-bins"),this.clearAccPipe=await g(this.device,(0,d.clearShader)(t.G*o.FEATURE_ACC_STRIDE+o.DECODER_PARAM_COUNT),"feature8-clear-acc"),this.adamPipe=await g(this.device,(0,s.adamShader)(),"feature8-adam");let r=(e,t)=>this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:t.map((e,t)=>({binding:t,resource:{buffer:e}}))});this.prepBind=r(this.prepPipe,[this.params,this.state]),this.emitBind=r(this.emitPipe,[this.state,this.tileCounts,this.binnedIds]),this.forwardBind=r(this.forwardPipe,[this.tileCounts,this.binnedIds,this.state,this.featureParams,this.decoderParams,this.image,this.tileStop]),this.backwardBind=r(this.backwardPipe,[this.gradImage,this.tileCounts,this.binnedIds,this.tileStop,this.state,this.featureParams,this.decoderParams,this.acc]),this.geometryChainBind=r(this.geometryChainPipe,[this.acc,this.state,this.gradGeom]),this.featureChainBind=r(this.featureChainPipe,[this.acc,this.gradFeature,this.gradDecoder]),this.clearBinsBind=r(this.clearBinsPipe,[this.tileCounts]),this.clearAccBind=r(this.clearAccPipe,[this.acc]);let a=this.geometrySegments.map(e=>{let t=f(this.device);return this.geometryAdamUniforms.push(t),this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradGeom}},{binding:3,resource:{buffer:this.geomM}},{binding:4,resource:{buffer:this.geomV}}]})});for(let e of(this.geometryAdamGroups.push(...a),this.featureAdamGroup=this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.featureAdamUniform}},{binding:1,resource:{buffer:this.featureParams}},{binding:2,resource:{buffer:this.gradFeature}},{binding:3,resource:{buffer:this.featureM}},{binding:4,resource:{buffer:this.featureV}}]}),c)){let e=f(this.device);this.decoderAdamUniforms.push(e),this.decoderAdamGroups.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.decoderParams}},{binding:2,resource:{buffer:this.gradDecoder}},{binding:3,resource:{buffer:this.decoderM}},{binding:4,resource:{buffer:this.decoderV}}]}))}}setParams(e){if(e.length!==this.dims.G*d.PARAM_STRIDE)throw Error("feature painter: wrong geometry parameter count");this.device.queue.writeBuffer(this.params,0,e)}setFeatureParams(e){if(e.length!==this.dims.G*o.FEATURE_STRIDE)throw Error("feature painter: wrong feature parameter count");this.device.queue.writeBuffer(this.featureParams,0,e)}setDecoderParams(e){if(e.length!==o.DECODER_PARAM_COUNT)throw Error("feature painter: wrong decoder parameter count");this.device.queue.writeBuffer(this.decoderParams,0,e)}setGradImage(e){if(e.length!==3*this.dims.H*this.dims.W)throw Error("feature painter: wrong image gradient count");this.device.queue.writeBuffer(this.gradImage,0,e)}async read(e,t){let r=this.device.createBuffer({size:4*t,usage:9}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(GPUMapMode.READ);let i=new Float32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}async readU32(e,t){let r=this.device.createBuffer({size:4*t,usage:9}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(GPUMapMode.READ);let i=new Uint32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}readParams(){return this.read(this.params,this.dims.G*d.PARAM_STRIDE)}readFeatureParams(){return this.read(this.featureParams,this.dims.G*o.FEATURE_STRIDE)}readDecoderParams(){return this.read(this.decoderParams,o.DECODER_PARAM_COUNT)}readGeometryGradient(){return this.read(this.gradGeom,this.dims.G*d.PARAM_STRIDE)}readFeatureGradient(){return this.read(this.gradFeature,this.dims.G*o.FEATURE_STRIDE)}readDecoderGradient(){return this.read(this.gradDecoder,o.DECODER_PARAM_COUNT)}readImage(){return this.read(this.image,3*this.dims.H*this.dims.W)}async readTileTelemetry(){let e=await this.readU32(this.tileCounts,this.dims.numTiles),t=await this.readU32(this.tileStop,this.dims.numTiles),r=0,a=0,i=0,s=0,d=0,o=0;for(let u=0;u<this.dims.numTiles;u++){let n=e[u],l=t[u];r+=n,a+=l,i=Math.max(i,n),s=Math.max(s,l),n>this.dims.cap&&(d++,o+=n-this.dims.cap)}return{meanCount:r/this.dims.numTiles,maxCount:i,meanStop:a/this.dims.numTiles,maxStop:s,overflowTiles:d,overflowEntries:o}}zeroAdamState(){let e=e=>new Float32Array(e),t=e(this.dims.G*d.PARAM_STRIDE),r=e(this.dims.G*o.FEATURE_STRIDE),a=e(o.DECODER_PARAM_COUNT);this.device.queue.writeBuffer(this.geomM,0,t),this.device.queue.writeBuffer(this.geomV,0,t),this.device.queue.writeBuffer(this.featureM,0,r),this.device.queue.writeBuffer(this.featureV,0,r),this.device.queue.writeBuffer(this.decoderM,0,a),this.device.queue.writeBuffer(this.decoderV,0,a)}recordForward(e,t){let r=p(e,t);r.setPipeline(this.prepPipe),r.setBindGroup(0,this.prepBind),r.dispatchWorkgroups(u(this.dims.G)),r.setPipeline(this.clearBinsPipe),r.setBindGroup(0,this.clearBinsBind),r.dispatchWorkgroups(u(this.dims.numTiles)),r.setPipeline(this.emitPipe),r.setBindGroup(0,this.emitBind),r.dispatchWorkgroups(u(this.dims.G)),r.setPipeline(this.forwardPipe),r.setBindGroup(0,this.forwardBind),r.dispatchWorkgroups(this.dims.numTiles),r.end()}recordBackward(e,t){let r=p(e,t);r.setPipeline(this.clearAccPipe),r.setBindGroup(0,this.clearAccBind),r.dispatchWorkgroups(u(this.dims.G*o.FEATURE_ACC_STRIDE+o.DECODER_PARAM_COUNT)),r.setPipeline(this.backwardPipe),r.setBindGroup(0,this.backwardBind),r.dispatchWorkgroups(this.dims.numTiles),r.setPipeline(this.geometryChainPipe),r.setBindGroup(0,this.geometryChainBind),r.dispatchWorkgroups(u(this.dims.G)),r.setPipeline(this.featureChainPipe),r.setBindGroup(0,this.featureChainBind),r.dispatchWorkgroups(u((0,o.FEATURE_CHAIN_WORK_ITEMS)(this.dims))),r.end()}recordAdam(e,t,r,a=s.DEFAULT_HYPER,i,d={}){let g=[r.mean,r.logScale,r.theta,r.color,r.opacity];this.geometrySegments.forEach((e,r)=>{m(this.device,this.geometryAdamUniforms[r],e.offset,e.length,g[r],t,a)}),m(this.device,this.featureAdamUniform,0,this.dims.G*o.FEATURE_STRIDE,d.feature??n,t,a),c.forEach((e,r)=>{m(this.device,this.decoderAdamUniforms[r],e.offset,e.length,d.decoder??l,t,a)});let f=p(e,i);f.setPipeline(this.adamPipe),this.geometrySegments.forEach((e,t)=>{f.setBindGroup(0,this.geometryAdamGroups[t]),f.dispatchWorkgroups(u(e.length))}),f.setBindGroup(0,this.featureAdamGroup),f.dispatchWorkgroups(u(this.dims.G*o.FEATURE_STRIDE)),this.decoderAdamGroups.forEach((e,t)=>{f.setBindGroup(0,e),f.dispatchWorkgroups(u(c[t].length))}),f.end()}runForward(){let e=this.device.createCommandEncoder();this.recordForward(e),this.device.queue.submit([e.finish()])}destroy(){for(let e of[this.params,this.featureParams,this.decoderParams,this.image,this.gradImage,this.state,this.tileCounts,this.binnedIds,this.tileStop,this.acc,this.gradGeom,this.geomM,this.geomV,this.gradFeature,this.featureM,this.featureV,this.gradDecoder,this.decoderM,this.decoderV,this.featureAdamUniform,...this.decoderAdamUniforms,...this.geometryAdamUniforms])try{e.destroy()}catch{}}}},{"./adam_wgsl":"bbLCC","./raster_wgsl":"5a6Kr","./feature_painter_wgsl":"2sBBA","@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}],ez88J:[function(e,t,r,a){var i=e("@parcel/transformer-js/src/esmodule-helpers.js");i.defineInteropFlag(r),i.export(r,"PIXEL_BUFFER_LR",()=>n),i.export(r,"PixelBufferEngine",()=>c),i.export(r,"PixelBufferOptimizer",()=>p),i.export(r,"randomPixelLogits",()=>g),i.export(r,"cosine",()=>o.cosine);var s=e("./adam_wgsl"),d=e("../clip/vision"),o=e("./optimize");let u=e=>Math.ceil(e/256),n=.08;async function l(e,t,r){e.pushErrorScope("validation");let a=e.createShaderModule({label:r,code:t}),i=e.createComputePipeline({label:r,layout:"auto",compute:{module:a,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw Error(`pixel buffer ${r}: ${s.message}`);return i}class c{static async create(e,t=1){let r=e.createBuffer({size:786432,usage:140}),a=e.createBuffer({size:786432,usage:132}),i=e.createBuffer({size:786432,usage:136}),d=e.createBuffer({size:786432,usage:132}),o=e.createBuffer({size:786432,usage:136}),u=e.createBuffer({size:786432,usage:136}),n=e.createBuffer({size:s.ADAM_UNIFORM_BYTES,usage:72}),[p,f,m]=await Promise.all([l(e,`
@group(0) @binding(0) var<storage, read> raw : array<f32>;
@group(0) @binding(1) var<storage, read_write> image : array<f32>;
fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= 196608u) { return; }
  image[i] = sigmoid1(raw[i]);
}
`,"pixel-buffer-forward"),l(e,`
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
`,"pixel-buffer-chain"),l(e,(0,s.adamShader)(),"pixel-buffer-adam")]),h=e.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:a}}]}),y=e.createBindGroup({layout:f.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:a}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:d}}]}),b=e.createBindGroup({layout:m.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:r}},{binding:2,resource:{buffer:d}},{binding:3,resource:{buffer:o}},{binding:4,resource:{buffer:u}}]}),x=new c(e,r,a,i,d,o,u,n,p,f,m,h,y,b);return x.setRaw(g(t)),x.zeroAdamState(),x}constructor(e,t,r,a,i,s,d,o,u,n,l,c,p,g){this.device=e,this.raw=t,this.image=r,this.gradImage=a,this.gradRaw=i,this.m=s,this.v=d,this.adamUniform=o,this.forwardPipe=u,this.chainPipe=n,this.adamPipe=l,this.forwardBind=c,this.chainBind=p,this.adamBind=g}setRaw(e){if(196608!==e.length)throw Error("pixel buffer: wrong raw image length");this.device.queue.writeBuffer(this.raw,0,e)}zeroAdamState(){let e=new Float32Array(196608);this.device.queue.writeBuffer(this.m,0,e),this.device.queue.writeBuffer(this.v,0,e)}recordForward(e){let t=e.beginComputePass();t.setPipeline(this.forwardPipe),t.setBindGroup(0,this.forwardBind),t.dispatchWorkgroups(u(196608)),t.end()}recordBackward(e){let t=e.beginComputePass();t.setPipeline(this.chainPipe),t.setBindGroup(0,this.chainBind),t.dispatchWorkgroups(u(196608)),t.end()}recordAdam(e,t,r=n,a=s.DEFAULT_HYPER){var i,d;let o,l,c;i=this.device,d=this.adamUniform,l=new Uint32Array(o=new ArrayBuffer(s.ADAM_UNIFORM_BYTES)),c=new Float32Array(o),l[0]=0,l[1]=196608,c[2]=r,c[3]=a.beta1,c[4]=a.beta2,c[5]=a.eps,c[6]=1-Math.pow(a.beta1,t),c[7]=1-Math.pow(a.beta2,t),i.queue.writeBuffer(d,0,o);let p=e.beginComputePass();p.setPipeline(this.adamPipe),p.setBindGroup(0,this.adamBind),p.dispatchWorkgroups(u(196608)),p.end()}runForward(){let e=this.device.createCommandEncoder();this.recordForward(e),this.device.queue.submit([e.finish()])}async readImage(){return this.read(this.image)}async readRaw(){return this.read(this.raw)}async read(e){let t=this.device.createBuffer({size:786432,usage:9}),r=this.device.createCommandEncoder();r.copyBufferToBuffer(e,0,t,0,786432),this.device.queue.submit([r.finish()]),await t.mapAsync(GPUMapMode.READ);let a=new Float32Array(t.getMappedRange().slice(0));return t.unmap(),t.destroy(),a}destroy(){for(let e of[this.raw,this.image,this.gradImage,this.gradRaw,this.m,this.v,this.adamUniform])try{e.destroy()}catch{}}}class p{static async create(e,t,r,a=1){let[i,s,o]=t.inputShape;if(3!==i||256!==s||256!==o)throw Error("pixel buffer requires MobileCLIP 256x256 RGB input");let[u,n]=await Promise.all([c.create(e,a),d.VisionTrainer.create(e,t,r)]);return new p(e,u,n)}constructor(e,t,r){this.side=256,this.step_=0,this.device=e,this.raster=t,this.trainer=r}setPrompt(e){this.trainer.writeText(e)}get stepCount(){return this.step_}step(){let e=this.device.createCommandEncoder();this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!0}),e.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackward(e),this.step_+=1,this.raster.recordAdam(e,this.step_),this.device.queue.submit([e.finish()])}async nudge(e=Date.now(),t=.12){let r=await this.raster.readRaw(),a=g(e),i=Math.max(0,Math.min(1,t)),s=(0x9e3779b9^e)>>>0||1;for(let e=0;e<r.length;e++)(s=Math.imul(s,1664525)+0x3c6ef35f>>>0)/0x100000000<i&&(r[e]=a[e]);this.raster.setRaw(r),this.raster.zeroAdamState()}async renderImage(){return this.raster.runForward(),this.raster.readImage()}async currentEmbedding(){let e=this.device.createCommandEncoder();return this.raster.recordForward(e),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!1}),this.device.queue.submit([e.finish()]),f(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){this.raster.destroy()}}function g(e=1){let t=new Float32Array(196608),r=e>>>0||1,a=()=>(r=Math.imul(r,1664525)+0x3c6ef35f>>>0)/0x100000000;for(let e=0;e<t.length;e++)t[e]=(a()-.5)*.16;return t}async function f(e,t,r){let a=e.createBuffer({size:4*r,usage:9}),i=e.createCommandEncoder();i.copyBufferToBuffer(t,0,a,0,4*r),e.queue.submit([i.finish()]),await a.mapAsync(GPUMapMode.READ);let s=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),a.destroy(),s}},{"./adam_wgsl":"bbLCC","../clip/vision":"3gu6C","./optimize":"bTsmq","@parcel/transformer-js/src/esmodule-helpers.js":"gKN0c"}]},["3k7uM"],"3k7uM","parcelRequire924a",{});
//# sourceMappingURL=splat.25341dff.js.map
