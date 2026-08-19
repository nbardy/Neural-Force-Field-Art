!// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles
function(e,t,r,a,i){/* eslint-disable no-undef */var s="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof self?self:"undefined"!=typeof window?window:"undefined"!=typeof global?global:{},n="function"==typeof s[a]&&s[a],o=n.cache||{},d="undefined"!=typeof module&&"function"==typeof module.require&&module.require.bind(module);function l(t,r){if(!o[t]){if(!e[t]){// if we cannot find the module within our internal map or
// cache jump to the current global require ie. the last bundle
// that was added to the page.
var i="function"==typeof s[a]&&s[a];if(!r&&i)return i(t,!0);// If there are other bundles on this page the require from the
// previous one is saved to 'previousRequire'. Repeat this as
// many times as there are bundles until the module is found or
// we exhaust the require chain.
if(n)return n(t,!0);// Try the node require function if it exists.
if(d&&"string"==typeof t)return d(t);var c=Error("Cannot find module '"+t+"'");throw c.code="MODULE_NOT_FOUND",c}h.resolve=function(r){var a=e[t][1][r];return null!=a?a:r},h.cache={};var u=o[t]=new l.Module(t);e[t][0].call(u.exports,h,u,u.exports,this)}return o[t].exports;function h(e){var t=h.resolve(e);return!1===t?{}:l(t)}}l.isParcelRequire=!0,l.Module=function(e){this.id=e,this.bundle=l,this.exports={}},l.modules=e,l.cache=o,l.parent=n,l.register=function(t,r){e[t]=[function(e,t){t.exports=r},{}]},Object.defineProperty(l,"root",{get:function(){return s[a]}}),s[a]=l;for(var c=0;c<t.length;c++)l(t[c]);if(r){// Expose entry point to Node, AMD or browser globals
// Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
var u=l(r);"object"==typeof exports&&"undefined"!=typeof module?module.exports=u:"function"==typeof define&&define.amd?define(function(){return u}):i&&(this[i]=u)}}({e5WXe:[function(e,t,r){let a,i,s,n,o,d;/// <reference types="@webgpu/types" />
var l=e("./splat3d/cameras"),c=e("./splat3d/optimize"),u=e("./splat3d/raster_wgsl"),h=e("./splat3d_aniso"),p=e("./splat/model_assets");let g={gpu:!!navigator.gpu,ready:!1,running:!1,step:0,view:0,cos:null,initialCos:null,error:null,phase:"boot",qualityPreset:"full3d",representation:"anisotropic",promptMode:"camera",gridPromptMode:"contact_sheet",bgPromptMode:"centered",backgroundMode:"black",alphaReg:"off",boundsReg:"weak",coverageReg:"off",rayReg:"off",entropyReg:"off",stageMode:"joint",adaptiveSplats:!0,mipSmoothing:!1,splatReg:"tiny",framingMode:"zoom_out",profiling:!1,viewsPerStep:3,viewSampler:"random",clipBatchSize:3,clipLayout:"per_view",gridDirectRaster:!0,gridRasterSide:80,gridStrength:"weak"};window.__splat3d=g;let f=document.getElementById("grid"),m=document.getElementById("prompt"),v=document.getElementById("qualityPreset"),w=document.getElementById("representation"),b=document.getElementById("view"),x=document.getElementById("promptMode"),y=document.getElementById("bgTextMode"),_=document.getElementById("backgroundMode"),B=document.getElementById("alphaReg"),S=document.getElementById("boundsReg"),$=document.getElementById("coverageReg"),R=document.getElementById("rayReg"),T=document.getElementById("entropyReg"),P=document.getElementById("stageMode"),C=document.getElementById("adaptiveSplats"),I=document.getElementById("mipSmoothing"),G=document.getElementById("splatReg"),A=document.getElementById("framingMode"),k=document.getElementById("viewBatch"),E=document.getElementById("viewSampler"),D=document.getElementById("clipMode"),M=document.getElementById("clipLayout"),O=document.getElementById("gridPromptMode"),z=document.getElementById("gridRasterMode"),W=document.getElementById("gridStrength"),F=document.getElementById("optimize"),L=document.getElementById("reset"),N=document.getElementById("readout"),j=document.getElementById("notice"),U=document.getElementById("timings");function q(e){j.textContent=e}function Y(e){g.error=e,g.phase="error",q(e),N.textContent="—",console.error("[splat3d_page]",e)}function V(){g.step=n?n.stepCount:0;let e=n?.cameras[Z]?.name??"view",t=[`step ${g.step}`,e];if(t.push("full3d"===g.qualityPreset?"full 3D":"fast"===g.qualityPreset?"fast base":"manual"),t.push("anisotropic"===g.representation?"anisotropic":"isotropic"),n&&t.push(`${g.viewsPerStep}/${n.cameras.length} views`),"random"===g.viewSampler&&t.push("random"),t.push(g.clipBatchSize>1?`clip x${g.clipBatchSize}`:"clip x1"),"grid9_close2"===g.clipLayout&&t.push("grid+2"),"dual_grid4"===g.clipLayout&&t.push("2 grids+4"),"grid9_close2"===g.clipLayout||"dual_grid4"===g.clipLayout){let e="same"===g.gridPromptMode?"grid=same text":"literal_v2"===g.gridPromptMode?"object grid text":"literal"===g.gridPromptMode?"literal grid text":"grid text";t.push(e),t.push(`${g.gridRasterSide}px grid raster`),t.push(`grid ${g.gridStrength}`)}if(t.push("camera"===g.promptMode?"camera text":"coarse"===g.promptMode?"coarse text":"same text"),"black"===g.bgPromptMode&&t.push("black bg"),"centered"===g.bgPromptMode&&t.push("centered bg"),"black"!==g.backgroundMode&&t.push(`${g.backgroundMode.replaceAll("_"," ")} bg`),"off"!==g.alphaReg&&t.push(`alpha ${g.alphaReg}`),"off"!==g.boundsReg&&t.push(`bounds ${g.boundsReg}`),"off"!==g.coverageReg&&t.push(`transmit ${g.coverageReg}`),"off"!==g.rayReg&&t.push(`ray compact ${g.rayReg}`),"off"!==g.entropyReg&&t.push(`ray entropy ${g.entropyReg}`),"staged"===g.stageMode&&t.push("staged rates"),g.adaptiveSplats&&t.push("adaptive splats"),g.mipSmoothing&&t.push("coarse-to-fine"),"off"!==g.splatReg&&t.push("band"===g.splatReg?"scale band":"anti-tiny"),"zoom_out"===g.framingMode&&t.push("zoom out"),null!==g.cos){let e=g.initialCos??g.cos,r=g.cos-e;t.push(`cos ${g.cos.toFixed(4)}`),t.push(`init ${e.toFixed(4)}`),t.push(`Δ ${r>=0?"+":""}${r.toFixed(4)}`)}g.phase&&"run"!==g.phase&&t.push(`(${g.phase})`),N.textContent=t.join("  \xb7  ")}function H(){if(!Q){U.textContent="sampled wall profile waiting...";return}let e=Q,t=Math.max(e.total,.001),r=(e,r)=>`${e.padEnd(11)} ${r.toFixed(1).padStart(6)} ms ${(100*r/t).toFixed(0).padStart(3)}%`,a=[`${"gpu-timestamp"===e.timing?"sampled GPU step":"sampled wall step"} ${g.step}`,`${e.views}/${e.totalViews} views \xb7 ${g.viewSampler} \xb7 ${"batch"===e.clipMode?`batch CLIP x${e.clipBatchSize}`:"single CLIP"} \xb7 ${e.timing}`,r("opt total",e.total),r("raster",e.rasterFwd+e.rasterReplay+e.rasterBwd),r("  fwd",e.rasterFwd)];if(e.rasterReplay>0&&a.push(r("  replay",e.rasterReplay)),a.push(r("  bwd",e.rasterBwd)),"batch"===e.clipMode?a.push(r("clip batch",e.clipBatch)):a.push(r("clip",e.clipFwd+e.clipBwd),r("  fwd",e.clipFwd),r("  bwd",e.clipBwd)),e.regularizer>0&&a.push(r("reg",e.regularizer)),a.push(r("adam",e.adam),r("display",e.display),r("clear",e.clear),`sample every ${er} steps`),K){let e=K;a.push(`tile count   ${e.maxCount}/${e.cap} max`),a.push(e.overflowTiles>0?`OVERFLOW     ${e.overflowTiles} tiles \xb7 ${e.overflowPairs} pairs`:"tile overflow 0")}if(ee){let e=ee;a.push(`opacity      ${e.opacityMean.toFixed(3)} mean \xb7 ${e.opacityP10.toFixed(3)}-${e.opacityP90.toFixed(3)} p10-p90`),a.push(`splat radius ${e.radiusMean.toFixed(4)} \xb7 spread ${e.spreadRms.toFixed(3)}`),"anisotropic"===g.representation&&(a.push(`axis ratio   ${e.axisRatioMean.toFixed(2)} mean \xb7 ${e.axisRatioP90.toFixed(2)} p90`),a.push(`screen ratio ${e.screenAxisRatioMean.toFixed(2)} mean \xb7 ${e.screenAxisRatioP90.toFixed(2)} p90`))}let i=n?.adaptationDiagnostics;i&&(a.push(`adapt splats ${i.relocationCount} moved \xb7 ${i.eligibleDestinations} dead`),i.densityStatsSampled&&a.push(`density sample ${i.densityVisiblePixels??0} px \xb7 abs grad ${(i.densityMaxScreenGradient??0).toExponential(2)}`)),U.textContent=a.join("\n")}let X=1,Z=0,J=!1,Q=null,K=null,ee=null,et=!1,er=30,ea=null,ei=[],es=[];function en(){let e=ed();if("grid9_close2"===e)return 3;if("dual_grid4"===e)return 6;let t=Number(D.value);return Number.isFinite(t)&&t>1?Math.min(9,0|t):1}function eo(){return"fast"===v.value?"fast":"manual"===v.value?"manual":"full3d"}function ed(){return"anisotropic"===el()?"per_view":"dual_grid4"===M.value?"dual_grid4":"grid9_close2"===M.value?"grid9_close2":"per_view"}function el(){return"anisotropic"===w.value?"anisotropic":"isotropic"}function ec(){let e=ed();if("grid9_close2"===e)return 9;if("dual_grid4"===e)return 22;let t=Number(k.value),r=n?.cameras.length??9;return Number.isFinite(t)?Math.max(1,Math.min(r,0|t)):3}function eu(){return"random"===E.value?"random":"epoch"}function eh(){return"same"===O.value?"same":"literal_v2"===O.value?"literal_v2":"literal"===O.value?"literal":"contact_sheet"}function ep(){return"scratch256"!==z.value}function eg(){return"hi512"===z.value?512:"direct80"===z.value?80:256}function ef(){return"off"===W.value?"off":"medium"===W.value?"medium":"full"===W.value?"full":"weak"}function em(){return"same"===x.value?"same":"coarse"===x.value?"coarse":"camera"}function ev(){return"none"===y.value?"none":"centered"===y.value?"centered":"black"}function ew(){return"dark_random"===_.value?"dark_random":"curriculum"===_.value?"curriculum":"blurred_noise"===_.value?"blurred_noise":"checkerboard"===_.value?"checkerboard":"fourier"===_.value?"fourier":"black"}function eb(){return"medium"===B.value?"medium":"weak"===B.value?"weak":"off"}function ex(){return"medium"===S.value?"medium":"weak"===S.value?"weak":"off"}function ey(){return"medium"===$.value?"medium":"weak"===$.value?"weak":"off"}function e_(){return"medium"===R.value?"medium":"weak"===R.value?"weak":"off"}function eB(){return"medium"===T.value?"medium":"weak"===T.value?"weak":"off"}function eS(){return"band"===G.value?"band":"tiny"===G.value?"tiny":"off"}function e$(){let e=eb(),t=ex(),r=ey(),a=e_(),i=eB(),s=eS();return{backgroundMode:ew(),opacitySparsity:"medium"===e?.03:"weak"===e?.01:0,centerWeight:"medium"===t?.006:"weak"===t?.002:0,radiusWeight:"medium"===t?.012:"weak"===t?.004:0,targetRadius:1.15,coverageWeight:"medium"===r?.2:"weak"===r?.05:0,coverageTarget:.12,transmittanceStart:.4,transmittanceEnd:.88,transmittanceAnnealSteps:500,rayDistortionWeight:"medium"===a?.1:"weak"===a?.02:0,rayEntropyWeight:"medium"===i?.05:"weak"===i?.01:0,rayEntropyMask:.05,smallRadiusWeight:"band"===s?.035:"tiny"===s?.02:0,smallRadius:.024,radiusBandWeight:"band"===s?.012:0,minRadius:.016,maxRadius:.16,stagedOptimization:"staged"===P.value,geometryWarmupSteps:250,geometryDecaySteps:1e3,geometryFinalScale:.2,appearanceWarmupScale:.35,adaptiveRelocation:"on"===C.value,adaptationInterval:200,adaptationFraction:.01,mipSmoothing:"on"===I.value,mipVarianceStart:4,mipVarianceEnd:.0625,mipAnnealSteps:500}}let eR=!1;function eT(e){if("manual"!==e){eR=!0;try{if("full3d"===e){w.value="anisotropic",x.value="camera",y.value="centered",_.value="black",B.value="off",S.value="weak",$.value="off",R.value="off",T.value="off",P.value="joint",C.value="on",I.value="off",G.value="tiny",A.value="zoom_out",k.value="3",E.value="random",M.value="per_view",D.value="3",O.value="literal_v2",z.value="direct80",W.value="weak";return}w.value="isotropic",x.value="camera",y.value="black",_.value="black",B.value="off",S.value="off",$.value="off",R.value="off",T.value="off",P.value="joint",C.value="off",I.value="off",G.value="off",A.value="normal",k.value="3",E.value="epoch",M.value="per_view",D.value="3",W.value="full"}finally{eR=!1}}}function eP(){eR||"manual"===v.value||(v.value="manual",g.qualityPreset="manual")}function eC(){g.promptMode=em(),g.gridPromptMode=eh(),g.gridDirectRaster=ep(),g.gridRasterSide=eg(),g.gridStrength=ef(),g.bgPromptMode=ev()}function eI(){g.qualityPreset=eo(),g.representation=el(),g.backgroundMode=ew(),g.alphaReg=eb(),g.boundsReg=ex(),g.coverageReg=ey(),g.rayReg=e_(),g.entropyReg=eB(),g.stageMode="staged"===P.value?"staged":"joint",g.adaptiveSplats="on"===C.value,g.mipSmoothing="on"===I.value,g.splatReg=eS(),g.framingMode="zoom_out"===A.value?"zoom_out":"normal"}function eG(){if("anisotropic"===el()){M.value="per_view",_.value="black",$.value="off",R.value="off",T.value="off",I.value="off";return}let e=ed();"grid9_close2"===e?(D.value="3",k.value="9"):"dual_grid4"===e&&(D.value="6",k.value="9")}function eA(e){let t="grid9_close2"===ed()||"dual_grid4"===ed(),r="anisotropic"===el();F.disabled=e,L.disabled=e,v.disabled=e,w.disabled=e,b.disabled=e,x.disabled=e,y.disabled=e,_.disabled=e||r,B.disabled=e,S.disabled=e,$.disabled=e||r,R.disabled=e||r,T.disabled=e||r,P.disabled=e,C.disabled=e,I.disabled=e||r,G.disabled=e,A.disabled=e,M.disabled=e||r,O.disabled=e||!t,z.disabled=e||!t,W.disabled=e||!t,k.disabled=e||t,E.disabled=e,D.disabled=e||t}async function ek(e,t){g.phase=t,g.clipLayout=ed(),g.gridPromptMode=eh(),g.gridDirectRaster=ep(),g.gridRasterSide=eg(),g.gridStrength=ef(),g.viewsPerStep=ec(),g.viewSampler=eu(),g.clipBatchSize=en(),eC(),eI(),V();let r=n;n=await eE(e),g.clipLayout=n.clipLayout,g.clipBatchSize=n.clipBatchSize,g.viewSampler=n.viewSampler,r?.destroy(),e6(),eO(),J=!0,g.step=0,Q=null,K=null,H(),V()}async function eE(e){if("anisotropic"===el())return(0,h.Splat3DAnisotropicOptimizer).create(a,i,s,{seed:e,clipBatchSize:g.clipBatchSize,viewSampler:eu(),lrs:function(){if("full3d"===eo())return{position:.03,logScale:.018,quaternion:.01,color:.04,opacity:.025}}(),convergence:e$(),cameras:(0,l.camerasForFraming)(g.framingMode)});let t=function(){let e=ef();return"off"===e?{grid:0,randomGrid:0}:"medium"===e?{grid:.5,randomGrid:.35}:"full"===e?{grid:1,randomGrid:1}:{grid:.25,randomGrid:.15}}();return(0,c.Splat3DOptimizer).create(a,i,s,{seed:e,clipBatchSize:g.clipBatchSize,clipLayout:g.clipLayout,viewSampler:g.viewSampler,gridDirectRaster:g.gridDirectRaster,gridRasterSide:g.gridRasterSide,gridGradientScale:t.grid,randomGridGradientScale:t.randomGrid,lrs:function(){if("full3d"===eo())return{position:.035,logRadius:.018,color:.035,opacity:.025}}(),convergence:e$(),cameras:"dual_grid4"===g.clipLayout?(0,l.dualGridCamerasForFraming)(g.framingMode):(0,l.camerasForFraming)(g.framingMode)})}let eD=/* wgsl */`
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
`;async function eM(){a.pushErrorScope("validation");let e=a.createShaderModule({code:eD});o=a.createRenderPipeline({layout:"auto",vertex:{module:e,entryPoint:"vs"},fragment:{module:e,entryPoint:"fs",targets:[{format:d}]},primitive:{topology:"triangle-list"}});let t=await a.popErrorScope();if(t)throw Error(`blit pipeline invalid: ${t.message}`)}function eO(){ea=a.createBindGroup({layout:o.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:n.raster.image}}]})}let ez=Function("u","return import(u)"),eW=null,eF=null,eL=new Map;async function eN(e){if(eF)return;let t=await ez("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm");t.env.allowRemoteModels=!0;let r="Nbardy/nff-clip-splat-weights",a=t=>{if("progress"===t.status&&t.total){let r=Math.round(t.progress??t.loaded/t.total*100),a=Math.round(r/100*16),i="█".repeat(a)+"░".repeat(16-a);e?.(`loading text encoder  [${i}] ${r}%  \xb7  ${(t.loaded/1e6).toFixed(1)}/${(t.total/1e6).toFixed(0)} MB`)}};// self-hosted alongside the vision weights
eW=await t.AutoTokenizer.from_pretrained(r,{progress_callback:a}),eF=await t.CLIPTextModelWithProjection.from_pretrained(r,{dtype:"fp16",device:"wasm",session_options:{graphOptimizationLevel:"basic"},progress_callback:a})}async function ej(e){await eN();let t=await eW(e,{padding:"max_length",max_length:77,truncation:!0}),r=await eF(t),a=r.text_embeds.data,i=new Float32Array(512);for(let e=0;e<512;e++)i[e]=a[e];return i}function eU(e){let t=e.trim(),r=eL.get(t);return r||(r=ej(t).catch(e=>{throw eL.delete(t),e}),eL.set(t,r)),r}let eq=null,eY=0,eV=!1;async function eH(){if(!eq||et)return;let e=n,t=Z,r=g.viewsPerStep;et=!0,g.profiling=!0,g.phase="profile",V();try{let a=await e.profileStep(t,r);if(e!==n||!g.running||(Q=a,K=await e.raster.readTileTelemetry(),ee=function(e,t){let r="anisotropic"===g.representation,a=r?h.ANISO_PARAM_STRIDE_3D:u.PARAM_STRIDE_3D,i=Math.floor(e.length/a),s=[0,0,0],n=new Float32Array(i),o=new Float32Array(i),d=new Float32Array(i),l=0,c=0;for(let a=0;a<i;a++){s[0]+=e[3*a+0],s[1]+=e[3*a+1],s[2]+=e[3*a+2];let u=r?(e[3*i+3*a+0]+e[3*i+3*a+1]+e[3*i+3*a+2])/3:e[3*i+a];if(l+=Math.exp(u),r){let r=3*i+3*a,s=Math.min(e[r],e[r+1],e[r+2]),n=Math.max(e[r],e[r+1],e[r+2]);if(o[a]=Math.exp(n-s),t){let s=6*i+4*a,n=(0,h.projectAnisotropicGaussian)({position:[e[3*a],e[3*a+1],e[3*a+2]],logScale:[e[r],e[r+1],e[r+2]],quaternion:[e[s],e[s+1],e[s+2],e[s+3]]},{eye:t.eye,right:t.right,up:t.cameraUp,forward:t.forward,focalPx:t.focalPx,centerPx:[128,128],near:.2}),[o,l,c]=n.covariance,u=Math.sqrt(Math.max(0,(o-c)**2+4*l*l)),p=Math.max(1e-12,.5*(o+c+u)),g=Math.max(1e-12,.5*(o+c-u));d[a]=Math.sqrt(p/g)}else d[a]=1}else o[a]=1,d[a]=1;let p=e[(r?13:7)*i+a],g=p>=0?1/(1+Math.exp(-p)):Math.exp(p)/(1+Math.exp(p));n[a]=g,c+=g}s[0]/=i,s[1]/=i,s[2]/=i;let p=0;for(let t=0;t<i;t++){let r=e[3*t+0]-s[0],a=e[3*t+1]-s[1],i=e[3*t+2]-s[2];p+=r*r+a*a+i*i}n.sort(),o.sort(),d.sort();let f=0,m=0;for(let e of o)f+=e;for(let e of d)m+=e;return{opacityMean:c/i,opacityP10:n[Math.floor((i-1)*.1)],opacityP90:n[Math.floor((i-1)*.9)],radiusMean:l/i,spreadRms:Math.sqrt(p/i),axisRatioMean:f/i,axisRatioP90:o[Math.floor((i-1)*.9)],screenAxisRatioMean:m/i,screenAxisRatioP90:d[Math.floor((i-1)*.9)]}}(await e.raster.readParams(),e.cameras[t]),e!==n||!g.running))return;g.step=e.stepCount,eY+=1,J=!0,H(),eY>=3&&(eY=0,eX())}catch(e){Y(`profile step failed: ${e?.message??e}`)}finally{g.profiling=!1,"profile"===g.phase&&(g.phase=g.running?"run":"idle"),et=!1,V()}}async function eX(){if(eq&&!eV){eV=!0;try{let e=await n.currentEmbedding(Z),t=(0,c.cosine)(e,eq[Z]);g.cos=t,null===g.initialCos&&(g.initialCos=t),V()}finally{eV=!1}}}function eZ(){if(g.running&&eq&&!et){let e=n.stepCount>0&&n.stepCount%er==0;e?eH():(n.step(Z,g.viewsPerStep),J=!0,eY+=1,g.step=n.stepCount,eY>=3&&(eY=0,eX()))}J&&(function(){if(!ea||!ei.length)return;n.prepareDisplayFrame();let e=a.createCommandEncoder();for(let t=0;t<ei.length;t++)n.raster.recordForward(e,t),function(e,t){if(!ea)return;let r=e.beginRenderPass({colorAttachments:[{view:t.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});r.setPipeline(o),r.setBindGroup(0,ea),r.draw(3),r.end()}(e,ei[t]);a.queue.submit([e.finish()])}(),J=!1),requestAnimationFrame(eZ)}async function eJ(){if(!g.ready||et)return;eG();let e=m.value.trim()||"a photo of a cat";eA(!0),g.running=!1,g.phase="encoding",g.cos=null,g.initialCos=null,Q=null,K=null,g.clipLayout=ed(),g.gridPromptMode=eh(),g.gridDirectRaster=ep(),g.gridRasterSide=eg(),g.gridStrength=ef(),g.viewsPerStep=ec(),g.viewSampler=eu(),g.clipBatchSize=en(),g.promptMode=em(),g.bgPromptMode=ev(),eI(),H(),V();try{let t=[];if("same"===g.promptMode){q("encoding prompt 1/1...");let r=await eU((0,l.buildBasePrompt)(e,g.bgPromptMode));for(let e=0;e<n.cameras.length;e++)t.push(r)}else for(let r=0;r<n.cameras.length;r++){q(`encoding prompt ${r+1}/${n.cameras.length}...`);let a="coarse"===g.promptMode?(0,l.buildCoarseViewPrompt)(e,n.cameras[r],g.bgPromptMode):(0,l.buildViewPrompt)(e,n.cameras[r],g.bgPromptMode);t.push(await eU(a))}eq=t,n.setViewPrompts(t),("grid9_close2"===g.clipLayout||"dual_grid4"===g.clipLayout)&&(q("encoding grid prompt..."),n.setGridPrompt(await eU((0,l.buildGrid9Prompt)(e,g.bgPromptMode,g.gridPromptMode)))),"dual_grid4"===g.clipLayout&&(q("encoding random grid prompt..."),n.setRandomGridPrompt(await eU((0,l.buildRandomGrid9Prompt)(e,g.bgPromptMode))),q("encoding zoom prompt..."),n.setZoomPrompt(await eU((0,l.buildZoomPrompt)(e,g.bgPromptMode))));let r=await n.currentEmbedding(Z);g.initialCos=(0,c.cosine)(r,t[Z]),g.cos=g.initialCos,eY=0,q(""),g.phase="run",g.running=!0,J=!0,V()}catch(e){Y(`text encode failed: ${e?.message??e}`)}finally{eA(!1)}}async function eQ(){if(g.ready){if(et){q("wait for profiling sample to finish before reset");return}g.running=!1,eq=null,g.cos=null,g.initialCos=null,Q=null,K=null,g.phase="reset",X+=1,await ek(X,"reset"),g.phase="idle",q(""),V()}}async function eK(){e0(Math.max(0,b.selectedIndex)),g.ready&&(g.cos=null,g.initialCos=null,eq&&eX(),V())}function e0(e){Z=Math.max(0,Math.min(ei.length?ei.length-1:0,0|e)),g.view=Z,b.selectedIndex=Z;for(let e=0;e<es.length;e++)es[e].classList.toggle("active",e===Z)}function e1(){eP(),eC(),Q=null,K=null,eq&&(g.running=!1,eq=null,g.cos=null,g.initialCos=null,g.phase="idle",q("")),H(),V()}async function e2(){if(g.ready){if(et){q("wait for profiling sample to finish before changing convergence settings"),_.value=g.backgroundMode,B.value=g.alphaReg,S.value=g.boundsReg,$.value=g.coverageReg,R.value=g.rayReg,T.value=g.entropyReg,P.value=g.stageMode,C.value=g.adaptiveSplats?"on":"off",I.value=g.mipSmoothing?"on":"off",w.value=g.representation,G.value=g.splatReg,A.value=g.framingMode;return}eP(),eI(),g.running=!1,eq=null,g.cos=null,g.initialCos=null,Q=null,K=null,eA(!0);try{await ek(X,"convergence"),q(""),g.phase="idle"}catch(e){Y(`convergence settings change failed: ${e?.message??e}`)}finally{eA(!1),V()}}}async function e3(){eG(),await e2()}async function e4(){let e=g.qualityPreset,t=eo();if(g.qualityPreset=t,!g.ready){eT(t),eC(),eI(),g.viewsPerStep=ec(),g.viewSampler=eu(),g.clipBatchSize=en(),V();return}if(et){q("wait for profiling sample to finish before changing quality preset"),v.value=e,g.qualityPreset=e;return}eT(t),eG(),g.running=!1,eq=null,g.cos=null,g.initialCos=null,Q=null,K=null,eA(!0);try{await ek(X,"preset"),q(""),g.phase="idle"}catch(e){Y(`quality preset change failed: ${e?.message??e}`)}finally{eA(!1),V()}}async function e5(){if(g.ready){if(eG(),et){var e;q("wait for profiling sample to finish before changing CLIP settings"),D.value=String(g.clipBatchSize),M.value=g.clipLayout,E.value=g.viewSampler,z.value=512===(e=g.gridRasterSide)?"hi512":80===e?"direct80":"scratch256",W.value=g.gridStrength,eG();return}eP(),g.running=!1,eq=null,g.cos=null,g.initialCos=null,Q=null,K=null,eA(!0);try{await ek(X,"optimizer"),q(""),g.phase="idle"}catch(e){Y(`clip settings change failed: ${e?.message??e}`)}finally{eA(!1),V()}}}function e6(){b.textContent="",f.textContent="",ei=[],es=[];let e=Math.min(9,n.cameras.length);for(let t=0;t<e;t++){let e=n.cameras[t],r=document.createElement("option");r.value=e.name,r.textContent=e.name,b.appendChild(r);let i=document.createElement("div");i.className="tile";let s=document.createElement("canvas");s.className="view",s.width=256,s.height=256;let o=document.createElement("div");o.className="label",o.textContent=e.name,i.append(s,o),i.addEventListener("click",()=>{e0(t),g.cos=null,g.initialCos=null,eq&&eX(),V()});let l=s.getContext("webgpu");l.configure({device:a,format:d,alphaMode:"opaque"}),f.appendChild(i),ei.push(l),es.push(i)}e0(Z)}async function e9(){if(!navigator.gpu){Y("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled."),eA(!0);return}g.phase="adapter";let e=await navigator.gpu.requestAdapter();if(!e)return Y("no WebGPU adapter available.");a=await e.requestDevice(),a.addEventListener?.("uncapturederror",e=>{console.error("[webgpu]",e.error?.message??e.error)}),d=navigator.gpu.getPreferredCanvasFormat(),g.phase="weights";try{let e=await (0,p.loadClipTrainAssets)(e=>{N.textContent=e});i=e.plan,s=e.weights}catch(e){return Y(e?.message??String(e))}g.phase="optimizer",N.textContent="building 3D optimizer…",await eM(),eT(eo()),eG(),eC(),eI(),n=await eE(X),g.clipLayout=n.clipLayout,g.viewsPerStep=ec(),g.viewSampler=n.viewSampler,g.clipBatchSize=n.clipBatchSize,g.gridPromptMode=eh(),g.gridDirectRaster=ep(),g.gridRasterSide=eg(),g.gridStrength=ef(),// Preload the text encoder at boot (with its own progress bar) so the first
// Optimize is instant instead of stalling on an 84 MB download (× the 9 views).
g.phase="textmodel",await eN(e=>{N.textContent=e}),// Do not create black WebGPU canvases while the text encoder is downloading.
// The view grid only appears once the page can immediately render into it.
e6(),eO(),J=!0,g.ready=!0,g.phase="idle",eA(!1),q(""),V(),requestAnimationFrame(eZ)}F.addEventListener("click",()=>void eJ()),L.addEventListener("click",()=>void eQ()),b.addEventListener("change",()=>void eK()),v.addEventListener("change",()=>void e4()),w.addEventListener("change",()=>void e3()),x.addEventListener("change",e1),y.addEventListener("change",e1),_.addEventListener("change",()=>void e2()),B.addEventListener("change",()=>void e2()),S.addEventListener("change",()=>void e2()),$.addEventListener("change",()=>void e2()),R.addEventListener("change",()=>void e2()),T.addEventListener("change",()=>void e2()),P.addEventListener("change",()=>void e2()),C.addEventListener("change",()=>void e2()),I.addEventListener("change",()=>void e2()),G.addEventListener("change",()=>void e2()),A.addEventListener("change",()=>void e2()),k.addEventListener("change",function(){eP(),eG(),g.viewsPerStep=ec(),Q=null,K=null,H(),V()}),E.addEventListener("change",()=>void e5()),D.addEventListener("change",()=>void e5()),M.addEventListener("change",()=>void e5()),O.addEventListener("change",e1),z.addEventListener("change",()=>void e5()),W.addEventListener("change",()=>void e5()),m.addEventListener("keydown",e=>{"Enter"===e.key&&eJ()}),e9().catch(e=>Y(`boot failed: ${e?.message??e}`))},{"./splat3d/cameras":"3Dl1Z","./splat3d/optimize":"dTqrt","./splat3d/raster_wgsl":"fuyeU","./splat3d_aniso":"t4gra","./splat/model_assets":"3CXuq"}],"3Dl1Z":[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"DEFAULT_3D_CAMERAS",()=>i),a.export(r,"BLACK_BACKGROUND_PROMPT",()=>s),a.export(r,"CENTERED_BLACK_BACKGROUND_PROMPT",()=>n),a.export(r,"FIXED_GRID_CAMERA_COUNT",()=>o),a.export(r,"DUAL_GRID_RANDOM_START",()=>d),a.export(r,"DUAL_GRID_ZOOM_START",()=>l),a.export(r,"DUAL_GRID_CAMERA_COUNT",()=>c),a.export(r,"normalizeBackgroundPromptMode",()=>u),a.export(r,"buildBasePrompt",()=>h),a.export(r,"buildViewPrompt",()=>p),a.export(r,"sampleWeightedCameraIndices",()=>g),a.export(r,"buildCoarseViewPrompt",()=>f),a.export(r,"buildGrid9Prompt",()=>m),a.export(r,"buildRandomGrid9Prompt",()=>v),a.export(r,"buildZoomPrompt",()=>w),a.export(r,"camerasForFraming",()=>b),a.export(r,"dualGridCamerasForFraming",()=>x),a.export(r,"prepareCamera",()=>S);let i=[{name:"top",promptSuffix:"a top-down camera angle",promptPrefix:"a directly overhead view of",sampleWeight:3,eye:[0,3.3,0],target:[0,0,0],up:[0,0,-1]},{name:"front",promptSuffix:"a front-facing camera angle",promptPrefix:"a front-on view of",sampleWeight:4,eye:[0,0,3],target:[0,0,0]},{name:"right",promptSuffix:"a camera angle from the right side",promptPrefix:"a right-side view of",sampleWeight:3,eye:[3,0,0],target:[0,0,0]},{name:"back",promptSuffix:"a camera angle from behind",promptPrefix:"a rear view of",sampleWeight:.75,eye:[0,0,-3],target:[0,0,0]},{name:"left",promptSuffix:"a camera angle from the left side",promptPrefix:"a left-side view of",sampleWeight:3,eye:[-3,0,0],target:[0,0,0]},{name:"front-left-high",promptSuffix:"an elevated 45 degree camera angle from the front left looking down",eye:[-2.16,1.7,2.16],target:[0,0,0],sampleWeight:1},{name:"front-right-high",promptSuffix:"an elevated 45 degree camera angle from the front right looking down",eye:[2.16,1.7,2.16],target:[0,0,0],sampleWeight:1},{name:"back-right-low",promptSuffix:"a low 45 degree camera angle from the rear right looking up",eye:[2.16,-1.3,-2.16],target:[0,0,0],sampleWeight:.75},{name:"back-left-low",promptSuffix:"a low 45 degree camera angle from the rear left looking up",eye:[-2.16,-1.3,-2.16],target:[0,0,0],sampleWeight:.75}],s="on a black background",n="centered on a black background",o=9,d=9,l=18,c=27;function u(e=!0){return!0===e?"black":!1===e?"none":e}function h(e,t=!0){let r=e.trim()||"a photo of a cat",a=u(t);if("none"===a||/\bblack background\b/i.test(r))return r;let i="centered"===a?n:s;return`${r}, ${i}`}function p(e,t,r=!0){let a=e.trim()||"a photo of a cat",i=t.promptPrefix?`${t.promptPrefix} ${a}`:`${a}, ${t.promptSuffix}`;return h(i,r)}function g(e,t,r){let a=e.map((e,t)=>t),i=[],s=Math.max(0,Math.min(a.length,0|t));for(;i.length<s;){let t=0;for(let r of a)t+=Math.max(0,e[r].sampleWeight??1);if(!(t>0)){i.push(...a.splice(0,s-i.length));break}let n=r()*t,o=a.length-1;for(let t=0;t<a.length;t++)if((n-=Math.max(0,e[a[t]].sampleWeight??1))<0){o=t;break}i.push(a[o]),a.splice(o,1)}return i}function f(e,t,r=!0){return h(`${e.trim()||"a photo of a cat"}, ${function(e){switch(e.name){case"top":return"a top-down view";case"front":return"a front view";case"back":return"a back view";case"left":case"right":return"a side view";default:return e.eye[1]>=0?"an elevated side view looking down":"a low side view looking up"}}(t)}`,r)}function m(e,t=!0,r="contact_sheet"){if("same"===r)return h(e,t);let a=e.trim()||"a photo of a cat",i=u(t),s="none"===i||/\bblack background\b/i.test(a)?"":", centered on a black background",n="top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up";return"literal_v2"===r?`a grid of 9 different camera angles of the same object, the object is centered, and the object is ${a}${s}`:"literal"===r?`a 3x3 grid showing ${a} from 9 different camera angles${s}. The 9 panels show the same subject in reading order: ${n}`:`a 3x3 image grid showing the same subject, ${a}, from nine different camera angles${s}: `+n}function v(e,t=!0){let r=e.trim()||"a photo of a cat",a=u(t),i="none"===a||/\bblack background\b/i.test(r)?"":", centered on a black background";return`a 3x3 grid of nine varied camera views of the same object, the object is centered, and the object is ${r}${i}`}function w(e,t=!0){let r=e.trim()||"a photo of a cat";return h(`a zoomed-in close-up view of ${r}`,t)}function b(e){return y(i,e)}function x(e){return[...y(i,e),...y(_("random"),e),...y(_("zoom"),e)]}function y(e,t){return"zoom_out"!==t?e.map(e=>({...e})):e.map(e=>({...e,eye:[1.25*e.eye[0],1.25*e.eye[1],1.25*e.eye[2]],fovYDeg:Math.max(e.fovYDeg??50,56)}))}function _(e){let t=[16,-10,29,-24,8,34,-32,21,-18],r="zoom"===e?2.15:3.15,a="zoom"===e?34:50;return[18,63,111,157,206,249,292,334,38].map((i,s)=>{let n=t[s],o=i*Math.PI/180,d=n*Math.PI/180,l=r*Math.cos(d);return{name:`${e}-${String(s+1).padStart(2,"0")}`,promptSuffix:"zoom"===e?`a zoomed-in close-up camera angle ${B(i,n)}`:`a varied camera angle ${B(i,n)}`,eye:[l*Math.sin(o),r*Math.sin(d),l*Math.cos(o)],target:[0,0,0],fovYDeg:a}})}function B(e,t){let r=(e%360+360)%360,a=r<22.5||r>=337.5?"from the front":r<67.5?"from the front right":r<112.5?"from the right side":r<157.5?"from the rear right":r<202.5?"from behind":r<247.5?"from the rear left":r<292.5?"from the left side":"from the front left";return t>12?`${a}, looking slightly down`:t<-12?`${a}, looking slightly up`:a}function S(e,t){var r,a;let i=T((r=e.target,a=e.eye,[r[0]-a[0],r[1]-a[1],r[2]-a[2]])),s=e.up??[0,1,0],n=T($(i,s));1e-5>R(n)&&(n=T($(i,[0,0,1])));let o=T($(n,i)),d=(e.fovYDeg??50)*Math.PI/180;return{...e,right:n,cameraUp:o,forward:i,focalPx:.5*t/Math.tan(.5*d)}}function $(e,t){return[e[1]*t[2]-e[2]*t[1],e[2]*t[0]-e[0]*t[2],e[0]*t[1]-e[1]*t[0]]}function R(e){return Math.hypot(e[0],e[1],e[2])}function T(e){let t=1/Math.max(R(e),1e-9);return[e[0]*t,e[1]*t,e[2]*t]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],k3151:[function(e,t,r){r.interopDefault=function(e){return e&&e.__esModule?e:{default:e}},r.defineInteropFlag=function(e){Object.defineProperty(e,"__esModule",{value:!0})},r.exportAll=function(e,t){return Object.keys(e).forEach(function(r){"default"===r||"__esModule"===r||t.hasOwnProperty(r)||Object.defineProperty(t,r,{enumerable:!0,get:function(){return e[r]}})}),t},r.export=function(e,t,r){Object.defineProperty(e,t,{enumerable:!0,get:r})}},{}],dTqrt:[function(e,t,r){/// <reference types="@webgpu/types" />
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"LEGIBLE_3D_G",()=>p),a.export(r,"LEGIBLE_3D_INIT",()=>g),a.export(r,"Splat3DOptimizer",()=>f),a.export(r,"randomSplats3D",()=>B),a.export(r,"cosine",()=>S);var i=e("../clip/vision"),s=e("../clip/vision_batch"),n=e("../splat/adam_wgsl"),o=e("./cameras"),d=e("./grid_clip"),l=e("./adaptive"),c=e("./raster"),u=e("./raster_wgsl");let h={COPY_SRC:4,COPY_DST:8},p=4096,g={radius:.075,radiusJitter:.35,opacityRaw:.3,colorSpread:1.2,positionSpread:.9};class f{static async create(e,t,r,a={}){var n;let[l,u,h]=t.inputShape;if(3!==l||256!==u||256!==h)throw Error(`splat3d: CLIP inputShape [${l},${u},${h}] != [3,256,256]`);let g=(a.cameras??(0,o.DEFAULT_3D_CAMERAS)).map(e=>(0,o.prepareCamera)(e,256)),w=a.G??p,b=v(a.convergence),x="curriculum"===(n=b.backgroundMode)||"blurred_noise"===n?"blurred_noise":"checkerboard"===n||"fourier"===n?n:void 0,y=await (0,c.Raster3DEngine).create(e,{H:256,W:256,G:w,cap:a.cap??function(e){let t=256;for(;t<e&&t<4096;)t*=2;return t}(w),bg:a.bg??[0,0,0],dynamicBg:"black"!==b.backgroundMode,dynamicBgTexture:void 0!==x,backgroundTextureMode:x,backgroundSeed:a.seed??1,dynamicCoverage:0!==b.coverageWeight||0!==b.rayDistortionWeight||0!==b.rayEntropyWeight,dynamicTransmittance:0!==b.coverageWeight,dynamicEntropy:0!==b.rayEntropyWeight,dynamicFootprint:b.mipSmoothing,cameras:g}),_=function(e){let t=Number.isFinite(e)?0|e:1;return t>1?Math.min(9,t):1}(a.clipBatchSize),S=a.clipLayout??"per_view",$=function(e,t){if(void 0!==e&&Number.isFinite(e)){let t=0|e;if(80===t||256===t||512===t)return t}return t?80:256}(a.gridRasterSide,a.gridDirectRaster),R=("grid9_close2"===S||"dual_grid4"===S)&&("grid9_close2"===S?3===_:6===_)&&80===$&&(a.stemSpatialBwd??!0)===!0&&(a.fusePointwiseGeluForward??!0)===!0&&(a.clipWeightPrecision??"f32")==="f32",T=a.spatialBwdVariant??(R?"depthwise4":void 0),P=R&&"depthwise4"===T,C={weightPrecision:a.clipWeightPrecision,pointwiseTileVariant:a.pointwiseTileVariant,pointwiseTileSteps:a.pointwiseTileSteps,stemSpatialBwd:a.stemSpatialBwd??!0,spatialBwdVariant:T,fusePointwiseGeluForward:a.fusePointwiseGeluForward??!0,fuseGeluBwdIntoPw:a.fuseGeluBwdIntoPw??P,fuseResidualBwdIntoPw:a.fuseResidualBwdIntoPw??P},I=await (0,i.VisionTrainer).create(e,t,r,C),G=_>1?await (0,s.BatchMajorVisionTrainer).create(e,t,r,_,{weightPrecision:a.clipWeightPrecision,stemSpatialBwd:C.stemSpatialBwd,spatialBwdVariant:C.spatialBwdVariant,sharedWForwardSteps:a.sharedWForwardSteps,fusePointwiseGeluForward:C.fusePointwiseGeluForward,fuseGeluBwdIntoPw:C.fuseGeluBwdIntoPw,fuseResidualBwdIntoPw:C.fuseResidualBwdIntoPw}):null;if(("grid9_close2"===S||"dual_grid4"===S)&&!G)throw Error(`splat3d: CLIP_LAYOUT=${S} needs batched CLIP`);if("grid9_close2"===S&&g.length<9)throw Error(`splat3d: CLIP_LAYOUT=grid9_close2 needs at least 9 cameras, got ${g.length}`);if("dual_grid4"===S){if(_<6)throw Error(`splat3d: CLIP_LAYOUT=dual_grid4 needs CLIP_BATCH=6, got ${_}`);if(g.length<o.DUAL_GRID_CAMERA_COUNT)throw Error(`splat3d: CLIP_LAYOUT=dual_grid4 needs ${o.DUAL_GRID_CAMERA_COUNT} cameras, got ${g.length}`)}y.setParams(a.initParams??B(w,a.seed??1,a.init)),y.zeroAdamState();let A=G&&(a.viewLaneBatchRasterForward||a.viewLaneBatchRasterBackward)?await y.createBatchForwardState({lanes:G.batch,imageBuffer:G.inputBuffer,imageOffsets:Array.from({length:G.batch},(e,t)=>G.slotOffsetBytes(t,G.plan.inputSlot)),gradBuffer:G.inputGradBuffer,gradOffsets:Array.from({length:G.batch},(e,t)=>G.inputGradOffsetBytes(t))}):null,k=("grid9_close2"===S||"dual_grid4"===S)&&G?await (0,d.Grid9Close2ClipLayout).create(e,y,G,{directRaster:a.gridDirectRaster??!1,rasterSide:$,gridLane:0,retainCellState:a.retainGridCellState,gradientScale:m(a.gridGradientScale),backgroundTextureMode:x,backgroundSeed:a.seed??1}):null,E="dual_grid4"===S&&G?await (0,d.Grid9Close2ClipLayout).create(e,y,G,{directRaster:a.gridDirectRaster??!1,rasterSide:$,gridLane:1,scratchRaster:256!==$?k?.raster:void 0,retainCellState:a.retainGridCellState,gradientScale:m(a.randomGridGradientScale??a.gridGradientScale),backgroundTextureMode:x,backgroundSeed:(a.seed??1)^5370206}):null;return new f(e,y,I,G,k,E,A,g,a)}constructor(e,t,r,a,i,s,o,d,l){var u;this.side=256,this.step_=0,this.hasPrompts=!1,this.rngState=1,this.viewOrder=[],this.viewCursor=0,this.cachedBatchViews=null,this.lastAdaptationStep=-1,this.adaptationDiagnostics_=null,this.device=e,this.raster=t,this.trainer=r,this.batchTrainer=a,this.gridClip=i,this.randomGridClip=s,this.batchRasterForward=o,this.cameras=d,this.clipBatchSize=a?.batch??1,this.clipLayout=l.clipLayout??"per_view",this.viewSampler=l.viewSampler??"epoch",this.lrs=l.lrs??c.DEFAULT_3D_LRS,this.hyper=l.hyper??n.DEFAULT_HYPER,this.singlePassBatchRasterForward=l.singlePassBatchRasterForward??!1,this.viewLaneBatchRasterForward=l.viewLaneBatchRasterForward??!1,this.viewLaneBatchRasterBackward=l.viewLaneBatchRasterBackward??!1,this.gridDirectRaster=l.gridDirectRaster??!1,this.clipRefreshInterval=Math.max(1,l.clipRefreshInterval??1),this.cachedLrScale=void 0!==(u=l.cachedLrScale)&&Number.isFinite(u)?Math.max(0,u):1,this.convergence=v(l.convergence),this.rngState=((l.seed??1)^2654435769)>>>0||1,this.textBuffers=d.map((t,a)=>e.createBuffer({label:`splat3d-text-${a}`,size:4*r.plan.textDim,usage:h.COPY_SRC|h.COPY_DST})),this.gridTextBuffer=i?e.createBuffer({label:"splat3d-grid9-text",size:4*r.plan.textDim,usage:h.COPY_SRC|h.COPY_DST}):null,this.randomGridTextBuffer=s?e.createBuffer({label:"splat3d-random-grid9-text",size:4*r.plan.textDim,usage:h.COPY_SRC|h.COPY_DST}):null,this.zoomTextBuffer="dual_grid4"===this.clipLayout?e.createBuffer({label:"splat3d-zoom-text",size:4*r.plan.textDim,usage:h.COPY_SRC|h.COPY_DST}):null,this.singleIO=t.createIOState(r.inputBuffer,0,r.inputGradBuffer,0),this.batchIO=o?.ios??(a?Array.from({length:a.batch},(e,r)=>t.createIOState(a.inputBuffer,a.slotOffsetBytes(r,a.plan.inputSlot),a.inputGradBuffer,a.inputGradOffsetBytes(r),{privateState:!0})):[])}setViewPrompts(e){if(e.length!==this.cameras.length)throw Error(`splat3d: ${e.length} text embeds for ${this.cameras.length} cameras`);for(let t=0;t<e.length;t++){if(e[t].length!==this.trainer.plan.textDim)throw Error(`splat3d: view ${t} text ${e[t].length} != ${this.trainer.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffers[t],0,e[t])}this.gridTextBuffer&&this.device.queue.writeBuffer(this.gridTextBuffer,0,e[0]),this.hasPrompts=!0}setGridPrompt(e){if(this.gridTextBuffer){if(e.length!==this.trainer.plan.textDim)throw Error(`splat3d: grid text ${e.length} != ${this.trainer.plan.textDim}`);this.device.queue.writeBuffer(this.gridTextBuffer,0,e)}}setRandomGridPrompt(e){if(this.randomGridTextBuffer){if(e.length!==this.trainer.plan.textDim)throw Error(`splat3d: random grid text ${e.length} != ${this.trainer.plan.textDim}`);this.device.queue.writeBuffer(this.randomGridTextBuffer,0,e)}}setZoomPrompt(e){if(this.zoomTextBuffer){if(e.length!==this.trainer.plan.textDim)throw Error(`splat3d: zoom text ${e.length} != ${this.trainer.plan.textDim}`);this.device.queue.writeBuffer(this.zoomTextBuffer,0,e)}}step(e=0,t=this.cameras.length){if(!this.hasPrompts)throw Error("splat3d: setViewPrompts() before step()");this.applyTrainingBackground(),this.applyCoverageRegularizer(),this.applyFootprintCurriculum();let r=this.shouldUseCachedBatchStep(t),a=r?this.cachedBatchViews.slice():this.sampleViews(t),i=this.device.createCommandEncoder();this.recordBackgroundTextures(i,this.trainingBackgroundStrength()),this.raster.recordClearRawGrad(i),r?this.recordCachedBatchTrainingViews(i,a):(this.recordTrainingViews(i,a),this.updateCachedBatchViews(a)),this.recordConvergenceRegularizer(i),this.step_+=1,this.raster.recordAdam(i,this.step_,this.lrsForStep(r),this.hyper),this.raster.recordBackgroundGenerate(i,this.step_,0,1),this.raster.recordForward(i,e),this.device.queue.submit([i.finish()])}async profileStep(e=0,t=this.cameras.length,r={}){if(!this.hasPrompts)throw Error("splat3d: setViewPrompts() before profileStep()");await this.device.queue.onSubmittedWorkDone(),this.applyTrainingBackground(),this.applyCoverageRegularizer(),this.applyFootprintCurriculum();let a=this.shouldUseCachedBatchStep(t),i=a?this.cachedBatchViews.slice():this.sampleViews(t),s=r.gpuTimestamps?_.create(this.device):null,n={views:i.length,totalViews:this.cameras.length,clipMode:this.useBatchFor(i)?"batch":"single",clipBatchSize:this.clipBatchSize,timing:s?"gpu-timestamp":"split-submit-wall",total:0,clear:0,rasterFwd:0,rasterReplay:0,clipFwd:0,clipBwd:0,clipBatch:0,rasterBwd:0,regularizer:0,adam:0,display:0},o=performance.now();try{if(this.hasTexturedBackground()){let e=this.device.createCommandEncoder();this.recordBackgroundTextures(e,this.trainingBackgroundStrength()),this.device.queue.submit([e.finish()])}if(n.clear+=await this.submitTimed((e,t)=>{this.raster.recordClearRawGrad(e,t)},s),a)n.rasterFwd+=await this.profileCachedBatchInputs(i,s),n.rasterBwd+=await this.profileCachedBatchBackward(i,s);else if(this.useDualGrid4Layout()){let e=this.batchTrainer,t=this.dualGrid4Views();n.views=t.fixedGrid.length+t.randomGrid.length+t.singles.length+t.zooms.length,n.rasterFwd+=await this.profileDualGrid4Inputs(t,s),n.clipBatch+=await this.submitTimed((t,r)=>{e.encode(t,{backward:!0,timestampWrites:r})},s);let r=await this.profileDualGrid4Backward(t,s);n.rasterReplay+=r.replay,n.rasterBwd+=r.backward}else if(this.useGridLayoutFor(i)){let e=this.batchTrainer,t=i.slice(0,9),r=this.grid9CloseupViews(t);n.rasterFwd+=await this.profileGrid9Close2Inputs(t,r,s),n.clipBatch+=await this.submitTimed((t,r)=>{e.encode(t,{backward:!0,timestampWrites:r})},s);let a=await this.profileGrid9Close2Backward(t,r,s);n.rasterReplay+=a.replay,n.rasterBwd+=a.backward}else if(this.useBatchFor(i)){let e=this.batchTrainer;for(let t=0;t<i.length;t+=e.batch){let r=i.slice(t,t+e.batch);if(r.length<e.batch){for(let e of r)n.rasterFwd+=await this.submitTimed((t,r)=>this.recordSingleForwardToTrainer(t,e,r),s),n.clipFwd+=await this.submitTimed((e,t)=>this.trainer.encodeForward(e,t),s),n.clipBwd+=await this.submitTimed((t,r)=>this.recordSingleTextAndBackward(t,e,r),s),n.rasterBwd+=await this.submitTimed((t,r)=>this.recordSingleRasterBackward(t,e,r),s);continue}if(n.rasterFwd+=await this.profileBatchInputs(r,s),n.clipBatch+=await this.submitTimed((t,r)=>{e.encode(t,{backward:!0,timestampWrites:r})},s),this.viewLaneBatchRasterBackward&&this.batchRasterForward&&r.length>1){n.rasterBwd+=await this.submitTimed((e,t)=>{this.raster.recordBatchBackwardAdd(e,this.batchRasterForward,r,t)},s);continue}for(let e=0;e<r.length;e++){let t=r[e],a=this.batchIO[e];n.rasterBwd+=await this.submitTimed((e,r)=>{this.raster.recordBackwardAdd(e,t,a,r)},s)}}}else for(let e of i)n.rasterFwd+=await this.submitTimed((t,r)=>this.recordSingleForwardToTrainer(t,e,r),s),n.clipFwd+=await this.submitTimed((e,t)=>{this.trainer.encodeForward(e,t)},s),n.clipBwd+=await this.submitTimed((t,r)=>this.recordSingleTextAndBackward(t,e,r),s),n.rasterBwd+=await this.submitTimed((t,r)=>this.recordSingleRasterBackward(t,e,r),s);return a||this.updateCachedBatchViews(i),this.convergenceRegularizerEnabled()&&(n.regularizer+=await this.submitTimed((e,t)=>{this.recordConvergenceRegularizer(e,t)},s)),this.step_+=1,n.adam+=await this.submitTimed((e,t)=>{this.raster.recordAdam(e,this.step_,this.lrsForStep(a),this.hyper,t)},s),this.applyDisplayBackground(),n.display+=await this.submitTimed((t,r)=>{this.raster.recordBackgroundGenerate(t,this.step_,0,1),this.raster.recordForward(t,e,void 0,r)},s),await this.adaptSplatsIfDue(),n.total=s?n.clear+n.rasterFwd+n.rasterReplay+n.clipFwd+n.clipBwd+n.clipBatch+n.rasterBwd+n.regularizer+n.adam+n.display:performance.now()-o,n}finally{s?.destroy()}}get stepCount(){return this.step_}get adaptationDiagnostics(){return this.adaptationDiagnostics_}async adaptSplatsIfDue(e=!1){if(!this.convergence.adaptiveRelocation)return null;let t=Math.max(1,Math.round(this.convergence.adaptationInterval));if(!e&&this.step_<t||!e&&this.lastAdaptationStep>=0&&this.step_-this.lastAdaptationStep<t)return null;if(!e&&this.lastAdaptationStep===this.step_)return this.adaptationDiagnostics_;await this.device.queue.onSubmittedWorkDone();let[r,a]=await Promise.all([this.raster.readParams(),this.raster.readRawGrad()]),i=(0,l.planFixedBudgetSplatAdaptation)(r,a,{maxRelocations:Math.max(1,Math.floor(this.raster.dims.G*this.convergence.adaptationFraction)),seed:(this.rngState^this.step_)>>>0,deadOpacityThreshold:.04,minParentOpacity:.12,splitOffsetScale:.55});return i.changedIndices.length>0&&(this.raster.setParams(i.params),this.raster.resetAdamForSplats(i.changedIndices)),this.lastAdaptationStep=this.step_,this.adaptationDiagnostics_=i.diagnostics,i.diagnostics}async renderView(e=0){return this.applyDisplayBackground(),this.renderBlackView(e),this.raster.readImage()}renderViewToImage(e=0){this.applyDisplayBackground(),this.renderBlackView(e)}async currentEmbedding(e=0){this.applyDisplayBackground();let t=this.device.createCommandEncoder();return this.raster.recordBackgroundGenerate(t,this.step_,0,1),this.raster.recordForward(t,e,this.singleIO),this.trainer.encode(t,{backward:!1}),this.device.queue.submit([t.finish()]),$(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){for(let e of(this.raster.destroy(),this.trainer.destroy(),this.batchTrainer?.destroy(),this.gridClip?.destroy(),this.randomGridClip?.destroy(),this.gridTextBuffer?.destroy(),this.randomGridTextBuffer?.destroy(),this.zoomTextBuffer?.destroy(),this.textBuffers))try{e.destroy()}catch(e){}}prepareDisplayFrame(){this.applyDisplayBackground()}useBatchFor(e){return!!this.batchTrainer&&e.length>=this.batchTrainer.batch}renderBlackView(e){let t=this.device.createCommandEncoder();this.raster.recordBackgroundGenerate(t,this.step_,0,1),this.raster.recordForward(t,e),this.device.queue.submit([t.finish()])}useGridLayoutFor(e){if("grid9_close2"!==this.clipLayout)return!1;if(!this.batchTrainer||!this.gridClip||!this.gridTextBuffer)throw Error("splat3d: grid9_close2 layout was not initialized");if(e.length<9)throw Error(`splat3d: grid9_close2 needs VIEWS=9, got ${e.length}`);if(this.batchTrainer.batch<3)throw Error(`splat3d: grid9_close2 needs CLIP_BATCH=3, got ${this.batchTrainer.batch}`);return!0}useDualGrid4Layout(){if("dual_grid4"!==this.clipLayout)return!1;if(!this.batchTrainer||!this.gridClip||!this.randomGridClip||!this.gridTextBuffer||!this.randomGridTextBuffer||!this.zoomTextBuffer)throw Error("splat3d: dual_grid4 layout was not initialized");if(this.batchTrainer.batch<6)throw Error(`splat3d: dual_grid4 needs CLIP_BATCH=6, got ${this.batchTrainer.batch}`);if(this.cameras.length<o.DUAL_GRID_CAMERA_COUNT)throw Error(`splat3d: dual_grid4 needs ${o.DUAL_GRID_CAMERA_COUNT} cameras, got ${this.cameras.length}`);return!0}grid9CloseupViews(e){let t=e.length;if("random"===this.viewSampler){let r=Math.floor(y(this.step_,101)*t)%t,a=Math.floor(y(this.step_,211)*t)%t;return a===r&&(a=(a+4)%t),[e[r],e[a]]}let r=this.step_%t;return[e[r],e[(r+4)%t]]}recordTrainingViews(e,t){if(this.useDualGrid4Layout()){this.recordDualGrid4Training(e,this.dualGrid4Views());return}if(this.useGridLayoutFor(t)){this.recordGrid9Close2Training(e,t.slice(0,9));return}if(!this.useBatchFor(t)){for(let r of t)this.recordSingleTrainingView(e,r);return}let r=this.batchTrainer;for(let a=0;a<t.length;a+=r.batch){let i=t.slice(a,a+r.batch);if(i.length<r.batch){for(let t of i)this.recordSingleTrainingView(e,t);continue}if(this.recordBatchInputs(e,i),r.encode(e,{backward:!0}),this.viewLaneBatchRasterBackward&&this.batchRasterForward&&i.length>1){this.raster.recordBatchBackwardAdd(e,this.batchRasterForward,i);continue}for(let t=0;t<i.length;t++){let r=i[t],a=this.batchIO[t];this.raster.recordBackwardAdd(e,r,a)}}}recordCachedBatchTrainingViews(e,t){this.recordCachedBatchInputs(e,t),this.recordCachedBatchBackward(e,t)}recordCachedBatchInputs(e,t){if(!this.batchTrainer)throw Error("splat3d: cached CLIP step needs batch trainer");if(t.length!==this.batchTrainer.batch)throw Error(`splat3d: cached CLIP step needs one full batch, got ${t.length}`);if(this.singlePassBatchRasterForward&&t.length>1){this.raster.recordForwards(e,t,this.batchIO.slice(0,t.length));return}if(this.viewLaneBatchRasterForward&&this.batchRasterForward&&t.length>1){this.raster.recordBatchForward(e,this.batchRasterForward,t);return}for(let r=0;r<t.length;r++)this.raster.recordForward(e,t[r],this.batchIO[r])}recordCachedBatchBackward(e,t){if(this.viewLaneBatchRasterBackward&&this.batchRasterForward&&t.length>1){this.raster.recordBatchBackwardAdd(e,this.batchRasterForward,t);return}for(let r=0;r<t.length;r++)this.raster.recordBackwardAdd(e,t[r],this.batchIO[r])}recordDualGrid4Training(e,t){let r=this.batchTrainer;this.recordDualGrid4Inputs(e,t),r.encode(e,{backward:!0}),this.recordDualGrid4Backward(e,t)}recordDualGrid4Inputs(e,t){let r=this.batchTrainer,a=this.gridClip,i=this.randomGridClip;this.recordDualGrid4TextCopies(e,t),a.clearGridImage(e),i.clearGridImage(e);for(let r=0;r<o.FIXED_GRID_CAMERA_COUNT;r++)a.raster.recordForward(e,t.fixedGrid[r],a.scratchIOForCell(r)),a.recordCopyCell(e,r),i.raster.recordForward(e,t.randomGrid[r],i.scratchIOForCell(r)),i.recordCopyCell(e,r);if(this.raster.recordForward(e,t.singles[0],this.batchIO[2]),this.raster.recordForward(e,t.singles[1],this.batchIO[3]),this.raster.recordForward(e,t.zooms[0],this.batchIO[4]),this.raster.recordForward(e,t.zooms[1],this.batchIO[5]),r.batch<6)throw Error("splat3d: dual_grid4 lost its CLIP batch")}recordDualGrid4Backward(e,t){let r=this.gridClip,a=this.randomGridClip;for(let i=0;i<o.FIXED_GRID_CAMERA_COUNT;i++){r.clearScratchGrad(e),r.recordScatterCell(e,i);let s=r.scratchIOForCell(i);r.retainsCellState||r.raster.recordForward(e,t.fixedGrid[i],s),r.raster.recordBackwardAdd(e,t.fixedGrid[i],s),a.clearScratchGrad(e),a.recordScatterCell(e,i);let n=a.scratchIOForCell(i);a.retainsCellState||a.raster.recordForward(e,t.randomGrid[i],n),a.raster.recordBackwardAdd(e,t.randomGrid[i],n)}this.raster.recordBackwardAdd(e,t.singles[0],this.batchIO[2]),this.raster.recordBackwardAdd(e,t.singles[1],this.batchIO[3]),this.raster.recordBackwardAdd(e,t.zooms[0],this.batchIO[4]),this.raster.recordBackwardAdd(e,t.zooms[1],this.batchIO[5])}recordDualGrid4TextCopies(e,t){let r=this.batchTrainer,a=4*r.plan.textDim;e.copyBufferToBuffer(this.gridTextBuffer,0,r.textBuffer,r.textOffsetBytes(0),a),e.copyBufferToBuffer(this.randomGridTextBuffer,0,r.textBuffer,r.textOffsetBytes(1),a),e.copyBufferToBuffer(this.textBuffers[t.singles[0]],0,r.textBuffer,r.textOffsetBytes(2),a),e.copyBufferToBuffer(this.textBuffers[t.singles[1]],0,r.textBuffer,r.textOffsetBytes(3),a),e.copyBufferToBuffer(this.zoomTextBuffer,0,r.textBuffer,r.textOffsetBytes(4),a),e.copyBufferToBuffer(this.zoomTextBuffer,0,r.textBuffer,r.textOffsetBytes(5),a)}recordGrid9Close2Training(e,t){let r=this.batchTrainer,a=this.grid9CloseupViews(t);this.recordGrid9Close2Inputs(e,t,a),r.encode(e,{backward:!0}),this.recordGrid9Close2Backward(e,t,a)}recordGrid9Close2Inputs(e,t,r){let a=this.batchTrainer,i=this.gridClip;this.recordGrid9Close2TextCopies(e,r),i.clearGridImage(e);for(let r=0;r<9;r++)i.raster.recordForward(e,t[r],i.scratchIOForCell(r)),i.recordCopyCell(e,r);for(let t=0;t<2;t++)this.raster.recordForward(e,r[t],this.batchIO[t+1]);// The batch variable is intentionally touched here so future edits keep the
// lane contract visible: lane 0 grid, lanes 1-2 close-ups.
if(a.batch<3)throw Error("splat3d: grid9_close2 lost its CLIP batch")}recordGrid9Close2Backward(e,t,r){let a=this.gridClip;for(let r=0;r<9;r++){a.clearScratchGrad(e),a.recordScatterCell(e,r);let i=a.scratchIOForCell(r);a.retainsCellState||a.raster.recordForward(e,t[r],i),a.raster.recordBackwardAdd(e,t[r],i)}for(let t=0;t<2;t++)this.raster.recordBackwardAdd(e,r[t],this.batchIO[t+1])}recordGrid9Close2TextCopies(e,t){let r=this.batchTrainer,a=4*r.plan.textDim;e.copyBufferToBuffer(this.gridTextBuffer,0,r.textBuffer,r.textOffsetBytes(0),a);for(let i=0;i<2;i++){let s=t[i];e.copyBufferToBuffer(this.textBuffers[s],0,r.textBuffer,r.textOffsetBytes(i+1),a)}}recordSingleTrainingView(e,t){e.copyBufferToBuffer(this.textBuffers[t],0,this.trainer.textBuffer,0,4*this.trainer.plan.textDim),this.raster.recordForward(e,t,this.singleIO),this.trainer.encode(e,{backward:!0}),this.raster.recordBackwardAdd(e,t,this.singleIO)}recordBatchInputs(e,t){if(this.recordBatchTextCopies(e,t),this.singlePassBatchRasterForward&&t.length>1){this.raster.recordForwards(e,t,this.batchIO.slice(0,t.length));return}if(this.viewLaneBatchRasterForward&&this.batchRasterForward&&t.length>1){this.raster.recordBatchForward(e,this.batchRasterForward,t);return}for(let r=0;r<t.length;r++)this.raster.recordForward(e,t[r],this.batchIO[r])}async profileBatchInputs(e,t){if(!t)return this.submitTimed(t=>this.recordBatchInputs(t,e));let r=this.device.createCommandEncoder();if(this.recordBatchTextCopies(r,e),this.device.queue.submit([r.finish()]),await this.device.queue.onSubmittedWorkDone(),this.singlePassBatchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordForwards(t,e,this.batchIO.slice(0,e.length),r)},t);if(this.viewLaneBatchRasterForward&&this.batchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordBatchForward(t,this.batchRasterForward,e,r)},t);let a=0;for(let r=0;r<e.length;r++)a+=await this.submitTimed((t,a)=>{this.raster.recordForward(t,e[r],this.batchIO[r],a)},t);return a}async profileCachedBatchInputs(e,t){if(!t)return this.submitTimed(t=>this.recordCachedBatchInputs(t,e));if(this.singlePassBatchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordForwards(t,e,this.batchIO.slice(0,e.length),r)},t);if(this.viewLaneBatchRasterForward&&this.batchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordBatchForward(t,this.batchRasterForward,e,r)},t);let r=0;for(let a=0;a<e.length;a++)r+=await this.submitTimed((t,r)=>{this.raster.recordForward(t,e[a],this.batchIO[a],r)},t);return r}async profileCachedBatchBackward(e,t){if(!t)return this.submitTimed(t=>this.recordCachedBatchBackward(t,e));if(this.viewLaneBatchRasterBackward&&this.batchRasterForward&&e.length>1)return this.submitTimed((t,r)=>{this.raster.recordBatchBackwardAdd(t,this.batchRasterForward,e,r)},t);let r=0;for(let a=0;a<e.length;a++)r+=await this.submitTimed((t,r)=>{this.raster.recordBackwardAdd(t,e[a],this.batchIO[a],r)},t);return r}async profileDualGrid4Inputs(e,t){if(!t)return this.submitTimed(t=>this.recordDualGrid4Inputs(t,e));let r=this.gridClip,a=this.randomGridClip,i=this.device.createCommandEncoder();this.recordDualGrid4TextCopies(i,e),r.clearGridImage(i),a.clearGridImage(i),this.device.queue.submit([i.finish()]),await this.device.queue.onSubmittedWorkDone();let s=0;for(let i=0;i<o.FIXED_GRID_CAMERA_COUNT;i++)s+=await this.submitTimed((t,a)=>{r.raster.recordForward(t,e.fixedGrid[i],r.scratchIOForCell(i),a)},t)+await this.submitTimed((e,t)=>{r.recordCopyCell(e,i,t)},t)+await this.submitTimed((t,r)=>{a.raster.recordForward(t,e.randomGrid[i],a.scratchIOForCell(i),r)},t)+await this.submitTimed((e,t)=>{a.recordCopyCell(e,i,t)},t);return s+(await this.submitTimed((t,r)=>{this.raster.recordForward(t,e.singles[0],this.batchIO[2],r)},t)+await this.submitTimed((t,r)=>{this.raster.recordForward(t,e.singles[1],this.batchIO[3],r)},t)+await this.submitTimed((t,r)=>{this.raster.recordForward(t,e.zooms[0],this.batchIO[4],r)},t)+await this.submitTimed((t,r)=>{this.raster.recordForward(t,e.zooms[1],this.batchIO[5],r)},t))}async profileDualGrid4Backward(e,t){if(!t)return{replay:0,backward:await this.submitTimed(t=>this.recordDualGrid4Backward(t,e))};let r=this.gridClip,a=this.randomGridClip,i=0,s=0;for(let n=0;n<o.FIXED_GRID_CAMERA_COUNT;n++){s+=await this.submitTimed((e,t)=>{r.clearScratchGrad(e),r.recordScatterCell(e,n,t)},t);let o=r.scratchIOForCell(n);r.retainsCellState||(i+=await this.submitTimed((t,a)=>{r.raster.recordForward(t,e.fixedGrid[n],o,a)},t)),s+=await this.submitTimed((t,a)=>{r.raster.recordBackwardAdd(t,e.fixedGrid[n],o,a)},t)+await this.submitTimed((e,t)=>{a.clearScratchGrad(e),a.recordScatterCell(e,n,t)},t);let d=a.scratchIOForCell(n);a.retainsCellState||(i+=await this.submitTimed((t,r)=>{a.raster.recordForward(t,e.randomGrid[n],d,r)},t)),s+=await this.submitTimed((t,r)=>{a.raster.recordBackwardAdd(t,e.randomGrid[n],d,r)},t)}return{replay:i,backward:s+=await this.submitTimed((t,r)=>{this.raster.recordBackwardAdd(t,e.singles[0],this.batchIO[2],r)},t)+await this.submitTimed((t,r)=>{this.raster.recordBackwardAdd(t,e.singles[1],this.batchIO[3],r)},t)+await this.submitTimed((t,r)=>{this.raster.recordBackwardAdd(t,e.zooms[0],this.batchIO[4],r)},t)+await this.submitTimed((t,r)=>{this.raster.recordBackwardAdd(t,e.zooms[1],this.batchIO[5],r)},t)}}async profileGrid9Close2Inputs(e,t,r){if(!r)return this.submitTimed(r=>this.recordGrid9Close2Inputs(r,e,t));let a=this.gridClip,i=this.device.createCommandEncoder();this.recordGrid9Close2TextCopies(i,t),a.clearGridImage(i),this.device.queue.submit([i.finish()]),await this.device.queue.onSubmittedWorkDone();let s=0;for(let t=0;t<9;t++)s+=await this.submitTimed((r,i)=>{a.raster.recordForward(r,e[t],a.scratchIOForCell(t),i)},r)+await this.submitTimed((e,r)=>{a.recordCopyCell(e,t,r)},r);for(let e=0;e<2;e++)s+=await this.submitTimed((r,a)=>{this.raster.recordForward(r,t[e],this.batchIO[e+1],a)},r);return s}async profileGrid9Close2Backward(e,t,r){if(!r)return{replay:0,backward:await this.submitTimed(r=>this.recordGrid9Close2Backward(r,e,t))};let a=this.gridClip,i=0,s=0;for(let t=0;t<9;t++){s+=await this.submitTimed((e,r)=>{a.clearScratchGrad(e),a.recordScatterCell(e,t,r)},r);let n=a.scratchIOForCell(t);a.retainsCellState||(i+=await this.submitTimed((r,i)=>{a.raster.recordForward(r,e[t],n,i)},r)),s+=await this.submitTimed((r,i)=>{a.raster.recordBackwardAdd(r,e[t],n,i)},r)}for(let e=0;e<2;e++)s+=await this.submitTimed((r,a)=>{this.raster.recordBackwardAdd(r,t[e],this.batchIO[e+1],a)},r);return{replay:i,backward:s}}recordBatchTextCopies(e,t){let r=this.batchTrainer;for(let a=0;a<t.length;a++){let i=t[a];e.copyBufferToBuffer(this.textBuffers[i],0,r.textBuffer,r.textOffsetBytes(a),4*r.plan.textDim)}}recordSingleForwardToTrainer(e,t,r){this.raster.recordForward(e,t,this.singleIO,r)}recordSingleTextAndBackward(e,t,r){e.copyBufferToBuffer(this.textBuffers[t],0,this.trainer.textBuffer,0,4*this.trainer.plan.textDim),this.trainer.encodeBackward(e,r)}recordSingleRasterBackward(e,t,r){this.raster.recordBackwardAdd(e,t,this.singleIO,r)}recordConvergenceRegularizer(e,t){this.convergenceRegularizerEnabled()&&this.raster.recordRegularizerAdd(e,this.regularizerOptions(),t)}convergenceRegularizerEnabled(){return 0!==this.convergence.centerWeight||0!==this.convergence.radiusWeight||0!==this.convergence.opacitySparsity||0!==this.convergence.smallRadiusWeight||0!==this.convergence.radiusBandWeight}regularizerOptions(){return{centerWeight:this.convergence.centerWeight,radiusWeight:this.convergence.radiusWeight,targetRadius:this.convergence.targetRadius,opacitySparsity:this.convergence.opacitySparsity,smallRadiusWeight:this.convergence.smallRadiusWeight,smallRadius:this.convergence.smallRadius,radiusBandWeight:this.convergence.radiusBandWeight,minRadius:this.convergence.minRadius,maxRadius:this.convergence.maxRadius}}coverageOptions(){let e=Math.max(1,this.convergence.transmittanceAnnealSteps),t=Math.max(0,Math.min(1,this.step_/e)),r=this.convergence.transmittanceStart+(this.convergence.transmittanceEnd-this.convergence.transmittanceStart)*t;return{transmittanceWeight:this.convergence.coverageWeight,targetTransmittance:r,rayDistortionWeight:this.convergence.rayDistortionWeight,rayEntropyWeight:this.convergence.rayEntropyWeight,rayEntropyMask:this.convergence.rayEntropyMask}}applyTrainingBackground(){this.applyBackground(this.trainingBackground())}applyDisplayBackground(){this.applyBackground([0,0,0])}applyCoverageRegularizer(){let e=this.coverageOptions();this.raster.setCoverageRegularizer(e),this.gridClip&&this.gridClip.raster!==this.raster&&this.gridClip.raster.setCoverageRegularizer(e),this.randomGridClip&&this.randomGridClip.raster!==this.raster&&this.randomGridClip.raster.setCoverageRegularizer(e)}applyFootprintCurriculum(){if(!this.convergence.mipSmoothing)return;let e=Math.max(1,this.convergence.mipAnnealSteps),t=Math.max(0,Math.min(1,this.step_/e)),r=this.convergence.mipVarianceStart+(this.convergence.mipVarianceEnd-this.convergence.mipVarianceStart)*t,a=new Set([this.raster]);for(let e of(this.gridClip&&a.add(this.gridClip.raster),this.randomGridClip&&a.add(this.randomGridClip.raster),a))e.setScreenVariance(r)}applyBackground(e){this.raster.setBackground(e),this.gridClip&&this.gridClip.raster!==this.raster&&this.gridClip.raster.setBackground(e),this.randomGridClip&&this.randomGridClip.raster!==this.raster&&this.randomGridClip.raster.setBackground(e)}hasTexturedBackground(){return this.raster.usesTexturedBackground||!!this.gridClip?.raster.usesTexturedBackground||!!this.randomGridClip?.raster.usesTexturedBackground}recordBackgroundTextures(e,t){let r=new Set([this.raster]);for(let a of(this.gridClip&&r.add(this.gridClip.raster),this.randomGridClip&&r.add(this.randomGridClip.raster),r))a.recordBackgroundGenerate(e,this.step_,t)}trainingBackgroundStrength(){return"curriculum"===this.convergence.backgroundMode?this.step_<120?.35:Math.min(1,.35+(this.step_-120)/380):1}trainingBackground(){let e=this.convergence.backgroundMode;if("black"===e)return[0,0,0];let t="curriculum"===e&&this.step_>=120&&this.step_%8==0,r=t?.28:.09,a=t?.02:0;return[a+r*y(this.step_,11),a+r*y(this.step_,29),a+r*y(this.step_,47)]}async submitTimed(e,t=null){if(t)return t.time(e);let r=this.device.createCommandEncoder();e(r);let a=performance.now();return this.device.queue.submit([r.finish()]),await this.device.queue.onSubmittedWorkDone(),performance.now()-a}sampleViews(e){let t=this.cameras.length,r=this.normalizedViewCount(e);if(r>=t)return Array.from({length:t},(e,t)=>t);if("random"===this.viewSampler)return this.sampleRandomViews(r);let a=[];for(;a.length<r;)this.viewCursor>=this.viewOrder.length&&this.shuffleViewOrder(),a.push(this.viewOrder[this.viewCursor]),this.viewCursor+=1;return a}shouldUseCachedBatchStep(e){return!(this.clipRefreshInterval<=1)&&!!this.cachedBatchViews&&this.step_%this.clipRefreshInterval!=0&&"per_view"===this.clipLayout&&!!this.batchTrainer&&this.normalizedViewCount(e)===this.cachedBatchViews.length}lrsForStep(e){var t,r;let a=this.lrs;if(this.convergence.stagedOptimization){let e=Math.max(1,this.convergence.geometryWarmupSteps),t=Math.max(1,this.convergence.geometryDecaySteps),r=Math.max(0,Math.min(1,this.step_/e)),i=Math.max(0,Math.min(1,(this.step_-e)/t)),s=1+(this.convergence.geometryFinalScale-1)*i,n=this.convergence.appearanceWarmupScale+(1-this.convergence.appearanceWarmupScale)*r;a={position:this.lrs.position*s,logRadius:this.lrs.logRadius*s,color:this.lrs.color*n,opacity:this.lrs.opacity*n}}return e&&1!==this.cachedLrScale?(t=a,r=this.cachedLrScale,{position:t.position*r,logRadius:t.logRadius*r,color:t.color*r,opacity:t.opacity*r}):a}updateCachedBatchViews(e){if(this.clipRefreshInterval<=1||"per_view"!==this.clipLayout||!this.batchTrainer){this.cachedBatchViews=null;return}this.cachedBatchViews=e.length===this.batchTrainer.batch?e.slice():null}normalizedViewCount(e){let t=this.cameras.length;return Math.max(1,Math.min(t,0|e))}dualGrid4Views(){return{fixedGrid:Array.from({length:o.FIXED_GRID_CAMERA_COUNT},(e,t)=>t),randomGrid:this.sampleCameraRange(o.DUAL_GRID_RANDOM_START,o.FIXED_GRID_CAMERA_COUNT,o.FIXED_GRID_CAMERA_COUNT,401),singles:this.sampleCameraPair(o.DUAL_GRID_RANDOM_START,o.FIXED_GRID_CAMERA_COUNT,503,607),zooms:this.sampleCameraPair(o.DUAL_GRID_ZOOM_START,o.FIXED_GRID_CAMERA_COUNT,709,811)}}sampleCameraRange(e,t,r,a){let i=Array.from({length:t},(t,r)=>e+r);if("random"!==this.viewSampler){let e=this.step_%Math.max(1,t);return i.slice(e).concat(i.slice(0,e)).slice(0,r)}for(let e=0;e<Math.min(r,i.length);e++){let t=Math.floor(y(this.step_,a+37*e)*(i.length-e)),r=e+Math.max(0,Math.min(i.length-e-1,t)),s=i[e];i[e]=i[r],i[r]=s}return i.slice(0,r)}sampleCameraPair(e,t,r,a){if("random"!==this.viewSampler){let r=e+this.step_%t;return[r,e+(this.step_+4)%t]}let i=e+Math.floor(y(this.step_,r)*t)%t,s=e+Math.floor(y(this.step_,a)*t)%t;return s===i&&(s=e+(s-e+4)%t),[i,s]}sampleRandomViews(e){return(0,o.sampleWeightedCameraIndices)(this.cameras,e,()=>this.nextRandomU32()/4294967296)}shuffleViewOrder(){this.viewOrder=Array.from({length:this.cameras.length},(e,t)=>t);for(let e=this.viewOrder.length-1;e>0;e--){let t=this.nextRandomU32()%(e+1),r=this.viewOrder[e];this.viewOrder[e]=this.viewOrder[t],this.viewOrder[t]=r}this.viewCursor=0}nextRandomU32(){return this.rngState=Math.imul(this.rngState,1664525)+1013904223>>>0,this.rngState}}function m(e){return void 0!==e&&Number.isFinite(e)?Math.max(0,e):1}function v(e){let t=e?.backgroundMode;return{backgroundMode:"dark_random"===t||"curriculum"===t||"blurred_noise"===t||"checkerboard"===t||"fourier"===t?t:"black",centerWeight:w(e?.centerWeight,0),radiusWeight:w(e?.radiusWeight,0),targetRadius:b(e?.targetRadius,1.15),opacitySparsity:w(e?.opacitySparsity,0),coverageWeight:w(e?.coverageWeight,0),coverageTarget:x(e?.coverageTarget,.18),transmittanceStart:x(e?.transmittanceStart,.4),transmittanceEnd:x(e?.transmittanceEnd,e?.coverageTarget===void 0?.88:1-x(e.coverageTarget,.18)),transmittanceAnnealSteps:b(e?.transmittanceAnnealSteps,500),rayDistortionWeight:w(e?.rayDistortionWeight,0),rayEntropyWeight:w(e?.rayEntropyWeight,0),rayEntropyMask:w(e?.rayEntropyMask,.05),smallRadiusWeight:w(e?.smallRadiusWeight,0),smallRadius:b(e?.smallRadius,.022),radiusBandWeight:w(e?.radiusBandWeight,0),minRadius:b(e?.minRadius,.014),maxRadius:b(e?.maxRadius,.18),stagedOptimization:e?.stagedOptimization===!0,geometryWarmupSteps:b(e?.geometryWarmupSteps,250),geometryDecaySteps:b(e?.geometryDecaySteps,1e3),geometryFinalScale:w(e?.geometryFinalScale,.2),appearanceWarmupScale:x(e?.appearanceWarmupScale,.35),adaptiveRelocation:e?.adaptiveRelocation===!0,adaptationInterval:b(e?.adaptationInterval,200),adaptationFraction:x(e?.adaptationFraction,.01),mipSmoothing:e?.mipSmoothing===!0,mipVarianceStart:w(e?.mipVarianceStart,4),mipVarianceEnd:w(e?.mipVarianceEnd,.0625),mipAnnealSteps:b(e?.mipAnnealSteps,500)}}function w(e,t){return void 0!==e&&Number.isFinite(e)?Math.max(0,e):t}function b(e,t){return void 0!==e&&Number.isFinite(e)?Math.max(1e-4,e):t}function x(e,t){return void 0!==e&&Number.isFinite(e)?Math.max(0,Math.min(1,e)):t}function y(e,t){let r=Math.imul(e+1>>>0,747796405)+Math.imul(t>>>0,2891336453)>>>0;return(r=((r=Math.imul(r>>>(r>>>28)+4^r,277803737)>>>0)>>>22^r)>>>0)/4294967296}class _{static create(e){return e.features.has("timestamp-query")?new _(e):null}constructor(e){this.device=e,this.querySet=e.createQuerySet({type:"timestamp",count:2}),this.resolveBuffer=e.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),this.readBuffer=e.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})}async time(e){let t=this.device.createCommandEncoder();e(t,{querySet:this.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}),t.resolveQuerySet(this.querySet,0,2,this.resolveBuffer,0),t.copyBufferToBuffer(this.resolveBuffer,0,this.readBuffer,0,16),this.device.queue.submit([t.finish()]),await this.readBuffer.mapAsync(GPUMapMode.READ);let r=new BigUint64Array(this.readBuffer.getMappedRange().slice(0));return this.readBuffer.unmap(),Number(r[1]-r[0])/1e6}destroy(){this.querySet.destroy(),this.resolveBuffer.destroy(),this.readBuffer.destroy()}}function B(e,t=1,r={}){let a=r.radius??g.radius,i=r.radiusJitter??g.radiusJitter,s=r.opacityRaw??g.opacityRaw,n=r.colorSpread??g.colorSpread,o=r.positionSpread??g.positionSpread,d=t>>>0||1,l=()=>{let e=Math.imul((d=Math.imul(d,747796405)+2891336453>>>0)>>>(d>>>28)+4^d,277803737)>>>0;return(e=(e>>>22^e)>>>0)/4294967296},c=()=>{let e=0,t=0;for(;0===e;)e=l();for(;0===t;)t=l();return Math.sqrt(-2*Math.log(e))*Math.cos(2*Math.PI*t)},h=new Float32Array(e*u.PARAM_STRIDE_3D),p=3*e,f=4*e,m=7*e,v=Math.log(a);for(let t=0;t<e;t++)h[0+3*t+0]=(2*l()-1)*o,h[0+3*t+1]=(2*l()-1)*o,h[0+3*t+2]=(2*l()-1)*o,h[p+t]=v+i*c(),h[f+3*t+0]=n*c(),h[f+3*t+1]=n*c(),h[f+3*t+2]=n*c(),h[m+t]=s;return h}function S(e,t){let r=0,a=0,i=0;for(let s=0;s<e.length;s++)r+=e[s]*t[s],a+=e[s]*e[s],i+=t[s]*t[s];return r/Math.sqrt(a*i||1)}async function $(e,t,r){let a=e.createBuffer({size:4*r,usage:9}),i=e.createCommandEncoder();i.copyBufferToBuffer(t,0,a,0,4*r),e.queue.submit([i.finish()]),await a.mapAsync(1);let s=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),a.destroy(),s}},{"../clip/vision":"lNzsi","../clip/vision_batch":"bWgJv","../splat/adam_wgsl":"kfWkJ","./cameras":"3Dl1Z","./grid_clip":"7V9cb","./adaptive":"eTMcA","./raster":"AoYYi","./raster_wgsl":"fuyeU","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],bWgJv:[function(e,t,r){/**
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
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),/**
 * Shared-weights, replicated-activation CLIP trainer for batch experiments.
 * Each lane has independent activation/grad slots and text embedding buffer.
 */a.export(r,"ReplicatedBatchVisionTrainer",()=>h),/**
 * True batch-major forward encoder. Activation slots are one buffer per logical
 * slot, sized `batch * slotFloats`, and each generated shader runs all lanes by
 * dispatching z workgroups. This is the forward-only proof before batch
 * backward and shared-W pointwise kernels.
 */a.export(r,"BatchMajorVisionEncoder",()=>p),/**
 * True batch-major forward+backward trainer. This is the gradient-producing
 * counterpart to BatchMajorVisionEncoder and mirrors VisionTrainer's public
 * buffers, except each buffer is laid out as `[batch][slotFloats]`.
 */a.export(r,"BatchMajorVisionTrainer",()=>g);var i=e("./vision_wgsl"),s=e("./vision_bwd_wgsl"),n=e("./vision_batch_wgsl");let o={COPY_SRC:4,COPY_DST:8,STORAGE:128};function d(e,t){return t?e.beginComputePass({timestampWrites:t}):e.beginComputePass()}async function l(e,t){let r=[];for(let a of t){e.pushErrorScope("validation");let t=e.createShaderModule({code:a.code}),i=e.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw Error(`vision_batch: pipeline '${a.label}' invalid: ${s.message}
${a.code}`);r.push({spec:a,pipeline:i})}return r}function c(e,t){if(!Number.isInteger(e)||e<0||e>=t)throw Error(`vision_batch: lane ${e} outside [0, ${t})`)}function u(e,t){if(e.length!==t)throw Error(`vision_batch: weights blob ${e.length} scalars != plan ${t}`)}class h{static async create(e,t,r,a,n={}){if(!Number.isInteger(a)||a<1)throw Error(`vision_batch: invalid batch ${a}`);u(r,t.weightsFloats);let o=(0,i.planDispatches)(t,n),d=(0,s.planBwdDispatches)(t,n),c=await l(e,[...o,...d]);return new h(e,t,r,a,c,o.length)}constructor(e,t,r,a,i,s){this.device=e,this.plan=t,this.batch=a,this.fwdCount=s,this.weightsBuffer=e.createBuffer({label:"clip-batch-weights",size:r.byteLength,usage:o.STORAGE|o.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,r),this.slotBuffers=Array.from({length:a},(r,a)=>t.slots.map((t,r)=>e.createBuffer({label:`clip-batch-lane-${a}-slot-${r}`,size:4*t,usage:o.STORAGE|o.COPY_DST|o.COPY_SRC}))),this.textBuffers=Array.from({length:a},(r,a)=>e.createBuffer({label:`clip-batch-lane-${a}-text`,size:4*t.textDim,usage:o.STORAGE|o.COPY_DST}));let n=(e,t)=>"weights"===t.kind?this.weightsBuffer:"text"===t.kind?this.textBuffers[e]:this.slotBuffers[e][t.slot];this.dispatches=i.map(({spec:t,pipeline:r})=>({pipeline:r,workgroups:t.workgroups,label:t.label,binds:this.slotBuffers.map((a,i)=>e.createBindGroup({layout:r.getBindGroupLayout(0),entries:t.buffers.map((e,t)=>({binding:t,resource:{buffer:n(i,e)}}))}))}))}inputBuffer(e){return c(e,this.batch),this.slotBuffers[e][this.plan.inputSlot]}outputBuffer(e){return c(e,this.batch),this.slotBuffers[e][this.plan.outputSlot]}inputGradBuffer(e){return c(e,this.batch),this.slotBuffers[e][this.plan.inputGradSlot]}writeInput(e,t){c(e,this.batch);let[r,a,i]=this.plan.inputShape;if(t.length!==r*a*i)throw Error(`vision_batch: input ${t.length} != ${r*a*i}`);this.device.queue.writeBuffer(this.inputBuffer(e),0,t)}writeText(e,t){if(c(e,this.batch),t.length!==this.plan.textDim)throw Error(`vision_batch: text ${t.length} != ${this.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffers[e],0,t)}encodeLane(e,t,r={}){c(t,this.batch);let a=!1===r.backward?this.fwdCount:this.dispatches.length,i=d(e,r.timestampWrites);for(let e=0;e<a;e++){let r=this.dispatches[e];i.setPipeline(r.pipeline),i.setBindGroup(0,r.binds[t]),i.dispatchWorkgroups(...r.workgroups)}i.end()}encode(e,t={}){let r=!1===t.backward?this.fwdCount:this.dispatches.length,a=t.schedule??"step-major",i=d(e,t.timestampWrites);if("lane-major"===a)for(let e=0;e<this.batch;e++)for(let t=0;t<r;t++){let r=this.dispatches[t];i.setPipeline(r.pipeline),i.setBindGroup(0,r.binds[e]),i.dispatchWorkgroups(...r.workgroups)}else for(let e=0;e<r;e++){let t=this.dispatches[e];i.setPipeline(t.pipeline);for(let e=0;e<this.batch;e++)i.setBindGroup(0,t.binds[e]),i.dispatchWorkgroups(...t.workgroups)}i.end()}runLane(e,t={}){let r=this.device.createCommandEncoder();this.encodeLane(r,e,t),this.device.queue.submit([r.finish()])}run(e={}){let t=this.device.createCommandEncoder();this.encode(t,e),this.device.queue.submit([t.finish()])}destroy(){for(let e of(this.weightsBuffer.destroy(),this.slotBuffers))for(let t of e)t.destroy();for(let e of this.textBuffers)e.destroy()}}class p{static async create(e,t,r,a,i={}){if(!Number.isInteger(a)||a<1)throw Error(`vision_batch: invalid batch ${a}`);u(r,t.weightsFloats);let s=await l(e,(0,n.batchForwardDispatches)(t,a,i));return new p(e,t,r,a,s)}constructor(e,t,r,a,i){this.device=e,this.plan=t,this.batch=a,this.weightsBuffer=e.createBuffer({label:"clip-batch-major-weights",size:r.byteLength,usage:o.STORAGE|o.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,r),this.slotBuffers=t.slots.map((t,r)=>e.createBuffer({label:`clip-batch-major-slot-${r}`,size:t*a*4,usage:o.STORAGE|o.COPY_DST|o.COPY_SRC}));let s=e=>{if("weights"===e.kind)return this.weightsBuffer;if("slot"===e.kind)return this.slotBuffers[e.slot];throw Error("vision_batch: batch-major forward received a text binding")};this.dispatches=i.map(({spec:t,pipeline:r})=>({pipeline:r,workgroups:t.workgroups,label:t.label,bind:e.createBindGroup({layout:r.getBindGroupLayout(0),entries:t.buffers.map((e,t)=>({binding:t,resource:{buffer:s(e)}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}slotOffsetBytes(e,t){return c(e,this.batch),e*this.plan.slots[t]*4}outputOffsetBytes(e){return this.slotOffsetBytes(e,this.plan.outputSlot)}writeInput(e,t){c(e,this.batch);let[r,a,i]=this.plan.inputShape;if(t.length!==r*a*i)throw Error(`vision_batch: input ${t.length} != ${r*a*i}`);this.device.queue.writeBuffer(this.inputBuffer,this.slotOffsetBytes(e,this.plan.inputSlot),t)}encode(e,t){let r=d(e,t);for(let e of this.dispatches)r.setPipeline(e.pipeline),r.setBindGroup(0,e.bind),r.dispatchWorkgroups(...e.workgroups);r.end()}run(){let e=this.device.createCommandEncoder();this.encode(e),this.device.queue.submit([e.finish()])}destroy(){for(let e of(this.weightsBuffer.destroy(),this.slotBuffers))e.destroy()}}class g{static async create(e,t,r,a,i={}){if(!Number.isInteger(a)||a<1)throw Error(`vision_batch: invalid batch ${a}`);u(r,t.weightsFloats);let{specs:s,fwdCount:o}=(0,n.batchTrainDispatches)(t,a,i),d=await l(e,s);return new g(e,t,r,a,d,o)}constructor(e,t,r,a,i,s){this.device=e,this.plan=t,this.batch=a,this.fwdCount=s,this.weightsBuffer=e.createBuffer({label:"clip-batch-major-train-weights",size:r.byteLength,usage:o.STORAGE|o.COPY_DST}),e.queue.writeBuffer(this.weightsBuffer,0,r),this.textBuffer=e.createBuffer({label:"clip-batch-major-text",size:t.textDim*a*4,usage:o.STORAGE|o.COPY_DST}),this.slotBuffers=t.slots.map((t,r)=>e.createBuffer({label:`clip-batch-major-train-slot-${r}`,size:t*a*4,usage:o.STORAGE|o.COPY_DST|o.COPY_SRC}));let n=e=>"weights"===e.kind?this.weightsBuffer:"text"===e.kind?this.textBuffer:this.slotBuffers[e.slot];this.dispatches=i.map(({spec:t,pipeline:r})=>({pipeline:r,workgroups:t.workgroups,label:t.label,bind:e.createBindGroup({layout:r.getBindGroupLayout(0),entries:t.buffers.map((e,t)=>({binding:t,resource:{buffer:n(e)}}))})}))}get inputBuffer(){return this.slotBuffers[this.plan.inputSlot]}get outputBuffer(){return this.slotBuffers[this.plan.outputSlot]}get inputGradBuffer(){return this.slotBuffers[this.plan.inputGradSlot]}slotOffsetBytes(e,t){return c(e,this.batch),e*this.plan.slots[t]*4}outputOffsetBytes(e){return this.slotOffsetBytes(e,this.plan.outputSlot)}inputGradOffsetBytes(e){return this.slotOffsetBytes(e,this.plan.inputGradSlot)}textOffsetBytes(e){return c(e,this.batch),e*this.plan.textDim*4}writeInput(e,t){c(e,this.batch);let[r,a,i]=this.plan.inputShape;if(t.length!==r*a*i)throw Error(`vision_batch: input ${t.length} != ${r*a*i}`);this.device.queue.writeBuffer(this.inputBuffer,this.slotOffsetBytes(e,this.plan.inputSlot),t)}writeText(e,t){if(c(e,this.batch),t.length!==this.plan.textDim)throw Error(`vision_batch: text ${t.length} != ${this.plan.textDim}`);this.device.queue.writeBuffer(this.textBuffer,this.textOffsetBytes(e),t)}encode(e,t={}){let r=!1===t.backward?this.fwdCount:this.dispatches.length,a=d(e,t.timestampWrites);for(let e=0;e<r;e++){let t=this.dispatches[e];a.setPipeline(t.pipeline),a.setBindGroup(0,t.bind),a.dispatchWorkgroups(...t.workgroups)}a.end()}run(e={}){let t=this.device.createCommandEncoder();this.encode(t,e),this.device.queue.submit([t.finish()])}destroy(){for(let e of(this.weightsBuffer.destroy(),this.textBuffer.destroy(),this.slotBuffers))e.destroy()}}},{"./vision_wgsl":"oFDUc","./vision_bwd_wgsl":"2Oqph","./vision_batch_wgsl":"eNMEN","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],eNMEN:[function(e,t,r){/**
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
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"batchForwardDispatches",()=>u),a.export(r,"batchTrainDispatches",()=>h);var i=e("./vision_wgsl"),s=e("./vision_bwd_wgsl"),n=e("./vision_batch_pointwise");function o(e){let t=e.indexOf("fn main(");if(t<0)throw Error("vision_batch_wgsl: missing fn main");let r=e.indexOf("{",t);if(r<0)throw Error("vision_batch_wgsl: missing main body");return{start:t,openBrace:r,signature:e.slice(t,r)}}function d(e,t){let r=function(e,t){let r=[];for(let a=0;a<t.buffers.length;a++){let i=t.buffers[a];if("slot"!==i.kind&&"text"!==i.kind)continue;let s=RegExp(`@group\\(0\\)\\s*@binding\\(${a}\\)\\s*var<storage,[^>]+>\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*:\\s*array<([^>]+)>`),n=t.code.match(s);if(!n)throw Error(`vision_batch_wgsl: could not find slot binding ${a} in ${t.label}`);let o=n[2].trim();if("f32"!==o&&"vec4f"!==o)throw Error(`vision_batch_wgsl: unsupported array<${o}> in ${t.label}`);let d="slot"===i.kind?e.slots[i.slot]:e.textDim;if(!Number.isFinite(d))throw Error(`vision_batch_wgsl: text binding in ${t.label} needs a TrainPlan`);if("vec4f"===o&&d%4!=0)throw Error(`vision_batch_wgsl: ${t.label} binding ${a} has ${d} floats, not vec4-aligned`);r.push({name:n[1],elem:o,strideFloats:d})}return r}(e,t);if(0===r.length)return t.code;let a=function(e){let t=o(e),r=t.signature.match(/@builtin\(workgroup_id\)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*vec3u/);if(r)return{code:e,batchExpr:`${r[1]}.z`};let a=t.signature.replace(/\)\s*$/,",\n        @builtin(workgroup_id) batchWid : vec3u)");return{code:e.slice(0,t.start)+a+e.slice(t.openBrace),batchExpr:"batchWid.z"}}(t.code),i=o(a.code),s=new Map;for(let e of r){if(s.has(e.name))throw Error(`vision_batch_wgsl: duplicate slot variable '${e.name}' in ${t.label}`);s.set(e.name,e)}let n=[`  let batchLane = ${a.batchExpr};`,...r.map(e=>{let t="vec4f"===e.elem?e.strideFloats/4:e.strideFloats;return`  let batchBase_${e.name} = batchLane * ${t}u;`})],d=a.code.slice(0,i.openBrace+1)+"\n"+n.join("\n")+a.code.slice(i.openBrace+1);for(let e of r){let t=RegExp(`\\b${e.name}\\[`,"g");d=d.replace(t,`${e.name}[batchBase_${e.name} + `)}return d}function l(e,t,r){let a=[];for(let s=0;s<e.steps.length;s++){let o=e.steps[s];if(r.sharedWForwardSteps?.has(s)&&"conv"===o.kind&&"pointwise"===o.variant){a.push((0,n.pointwiseSharedWBatchForwardDispatch)(e,o,t,r.weightPrecision));continue}let l=e.steps[s+1];if(r.fusePointwiseGeluForward&&"conv"===o.kind&&"pointwise"===o.variant&&l?.kind==="gelu"&&l.src===o.dst){a.push(c(e,(0,i.pointwiseFusedGelu)(o,l,r,s),t)),s+=1;continue}for(let n of(0,i.stepDispatches)(o,r,s)){if(1!==n.workgroups[2])throw Error(`vision_batch_wgsl: ${n.label} already uses workgroup z=${n.workgroups[2]}`);a.push({...n,code:d(e,n),workgroups:[n.workgroups[0],n.workgroups[1],t]})}}return a}function c(e,t,r){if(1!==t.workgroups[2])throw Error(`vision_batch_wgsl: ${t.label} already uses workgroup z=${t.workgroups[2]}`);return{...t,code:d(e,t),workgroups:[t.workgroups[0],t.workgroups[1],r]}}function u(e,t,r={}){if(!Number.isInteger(t)||t<1)throw Error(`vision_batch_wgsl: invalid batch ${t}`);return r.sharedWForwardSteps?.size||r.fusePointwiseGeluForward?l(e,t,r):(0,i.planDispatches)(e,r).map(r=>c(e,r,t))}function h(e,t,r={}){if(!Number.isInteger(t)||t<1)throw Error(`vision_batch_wgsl: invalid batch ${t}`);let a=r.sharedWForwardSteps?.size||r.fusePointwiseGeluForward?l(e,t,r):(0,i.planDispatches)(e,r).map(r=>c(e,r,t)),n=(0,s.planBwdDispatches)(e,r).map(r=>c(e,r,t)),o=[...a,...n];return{specs:o,fwdCount:a.length}}},{"./vision_wgsl":"oFDUc","./vision_bwd_wgsl":"2Oqph","./vision_batch_pointwise":"92gmd","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"92gmd":[function(e,t,r){/**
 * vision_batch_pointwise — isolated shared-W pointwise batch experiments.
 *
 * The batch-major CLIP fork runs one workgroup per image lane, so each lane
 * reloads the same 32x32 W tile. This microkernel instead puts batch lanes in
 * local_invocation_id.z: one workgroup stages W once, while each lane stages
 * its own X tile. B=2 and B=3 fit the 16 KB workgroup-memory limit.
 *//// <reference types="@webgpu/types" />
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),/** Baseline z-batch pointwise: same math as the normal pointwise kernel, but
 * buffers are compact `[batch][tensor]` instead of full CLIP slot buffers. */a.export(r,"pointwiseZBatchDispatch",()=>d),/** Shared-W batch pointwise: B lanes inside the same workgroup, one W tile. */a.export(r,"pointwiseSharedWBatchDispatch",()=>l),/** Production shared-W forward pointwise: real CLIP slots laid out as
 * `[batch][slotFloats]`, with batch lanes inside one workgroup. */a.export(r,"pointwiseSharedWBatchForwardDispatch",()=>c);var i=e("./vision_wgsl");function s(e){let t=[{kind:"weights"},{kind:"slot",slot:0},{kind:"slot",slot:1}];return e&&t.push({kind:"slot",slot:2}),t}function n(e,t,r){let a=e.slots[t];if(a%4!=0)throw Error(`pointwise_shared_w: ${r} slot ${t} has non-vec4 stride ${a}`);return a/4}function o(e,t,r,a,i=!0){let s="gelu"===e.act?`gelu4(${a})`:a;if(null===e.residual)return s;let n=i?`res[resBase + (co + ${r}u) * ${t}u + p4]`:`res[(co + ${r}u) * ${t}u + p4]`;return`${n} + vec4f(W(${e.layerScaleOff}u + co + ${r}u)) * ${s}`}function d(e,t,r="f32"){if(!Number.isInteger(t)||t<1)throw Error(`pointwise_zbatch: invalid batch ${t}`);let a=e.outH*e.outW;(0,i.assertPointwiseTiles)(e.name,e.cin,e.cout,a,e.wOff);let n=a/4,d=e.cin*n,l=e.cout*n,c=null!==e.residual,u=/* wgsl */`
${(0,i.weightsDecl)(0,r)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${c?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${i.GELU}
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
}`;return{label:`pw-zbatch B${t} ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:u,workgroups:[n/8,e.cout/32,t],buffers:s(c)}}function l(e,t,r="f32"){if(!Number.isInteger(t)||t<1||t>3)throw Error(`pointwise_shared_w: batch ${t} outside [1, 3]`);let a=e.outH*e.outW;(0,i.assertPointwiseTiles)(e.name,e.cin,e.cout,a,e.wOff);let n=a/4,d=e.cin*n,l=e.cout*n,c=null!==e.residual,u=/* wgsl */`
${(0,i.weightsDecl)(0,r)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${c?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${i.GELU}
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
}`;return{label:`pw-shared-w B${t} ${e.cin}->${e.cout} @${e.outH}x${e.outW}`,code:u,workgroups:[n/8,e.cout/32,1],buffers:s(c)}}function c(e,t,r,a="f32"){if(!Number.isInteger(r)||r<1||r>3)throw Error(`pointwise_shared_w_forward: batch ${r} outside [1, 3]`);let s=t.outH*t.outW;(0,i.assertPointwiseTiles)(t.name,t.cin,t.cout,s,t.wOff);let d=s/4,l=n(e,t.src,"src"),c=n(e,t.dst,"dst"),u=null!==t.residual,h=u?n(e,t.residual,"residual"):c,p=/* wgsl */`
${(0,i.weightsDecl)(0,a)}
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
${u?"@group(0) @binding(3) var<storage, read> res : array<vec4f>;":""}
${i.GELU}
var<workgroup> xS : array<vec4f, ${256*r}>;
var<workgroup> wS : array<vec4f, 256>;
@compute @workgroup_size(8, 8, ${r})
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u) {
  let lane = lid.z;
  let li = lid.y * 8u + lid.x;
  let xTile = lane * 256u;
  let srcBase = lane * ${l}u;
  let dstBase = lane * ${c}u;
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
}`;return{label:`pw-shared-w-fwd B${r} ${t.cin}->${t.cout} @${t.outH}x${t.outW}`,code:p,workgroups:[d/8,t.cout/32,1],buffers:function(e){let t=[{kind:"weights"},{kind:"slot",slot:e.src},{kind:"slot",slot:e.dst}];return null!==e.residual&&t.push({kind:"slot",slot:e.residual}),t}(t)}}},{"./vision_wgsl":"oFDUc","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"7V9cb":[function(e,t,r){/// <reference types="@webgpu/types" />
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"Grid9Close2ClipLayout",()=>p);var i=e("./cameras"),s=e("./raster");let n={COPY_SRC:4,COPY_DST:8,STORAGE:128},o=Math.ceil(256/3);function d(e,t=!1){let r=e%3,a=Math.floor(e/3);if(!t)return{x:88*r,y:88*a,w:80,h:80,maxSide:80};let i=Math.floor(256*r/3),s=Math.floor(256*a/3);return{x:i,y:s,w:Math.floor((r+1)*256/3)-i,h:Math.floor((a+1)*256/3)-s,maxSide:o}}async function l(e,t,r){e.pushErrorScope("validation");let a=e.createShaderModule({code:t}),i=e.createComputePipeline({layout:"auto",compute:{module:a,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw console.error(`--- WGSL that failed (${r}) ---
${t}`),Error(`grid9_close2 pipeline validation (${r}): ${s.message}`);return i}function c(e,t){return t?e.beginComputePass({timestampWrites:t}):e.beginComputePass()}function u(e){let t=Number.isFinite(e)?e:1,r=t.toString();return/[.eE]/.test(r)||(r+=".0"),r}function h(e,t,r,a=1,i=!1){let{x:s,y:n,w:o,h:l,maxSide:c}=d(e,i),h=r*r,p="downsample"===t?"src":"gridGrad",g="downsample"===t?"gridImage":"dst",f=80===r?"downsample"===t?/* wgsl */`
  let srcPix = cy * 80u + cx;
  ${g}[ch * 65536u + dstPix] = ${p}[ch * ${h}u + srcPix];`:/* wgsl */`
  let srcPix = cy * 80u + cx;
  ${g}[ch * ${h}u + srcPix] = ${p}[ch * 65536u + dstPix] * ${u(a)};`:"downsample"===t?/* wgsl */`
  let fx = (f32(cx) + 0.5) * ${u(r/o)} - 0.5;
  let fy = (f32(cy) + 0.5) * ${u(r/l)} - 0.5;
  let x0 = u32(clamp(floor(fx), 0.0, ${u(r-1)}));
  let y0 = u32(clamp(floor(fy), 0.0, ${u(r-1)}));
  let x1 = min(${r-1}u, x0 + 1u);
  let y1 = min(${r-1}u, y0 + 1u);
  let wx = clamp(fx - f32(x0), 0.0, 1.0);
  let wy = clamp(fy - f32(y0), 0.0, 1.0);
  let base = ch * ${h}u;
  let v00 = ${p}[base + y0 * ${r}u + x0];
  let v10 = ${p}[base + y0 * ${r}u + x1];
  let v01 = ${p}[base + y1 * ${r}u + x0];
  let v11 = ${p}[base + y1 * ${r}u + x1];
  let vx0 = mix(v00, v10, wx);
  let vx1 = mix(v01, v11, wx);
  ${g}[ch * 65536u + dstPix] = mix(vx0, vx1, wy);`:/* wgsl */`
  let fx = (f32(cx) + 0.5) * ${u(r/o)} - 0.5;
  let fy = (f32(cy) + 0.5) * ${u(r/l)} - 0.5;
  let x0 = u32(clamp(floor(fx), 0.0, ${u(r-1)}));
  let y0 = u32(clamp(floor(fy), 0.0, ${u(r-1)}));
  let x1 = min(${r-1}u, x0 + 1u);
  let y1 = min(${r-1}u, y0 + 1u);
  let wx = clamp(fx - f32(x0), 0.0, 1.0);
  let wy = clamp(fy - f32(y0), 0.0, 1.0);
  let g = ${p}[ch * 65536u + dstPix] * ${u(a)};
  let base = ch * ${h}u;
  ${g}[base + y0 * ${r}u + x0] = ${g}[base + y0 * ${r}u + x0] + g * (1.0 - wx) * (1.0 - wy);
  ${g}[base + y0 * ${r}u + x1] = ${g}[base + y0 * ${r}u + x1] + g * wx * (1.0 - wy);
  ${g}[base + y1 * ${r}u + x0] = ${g}[base + y1 * ${r}u + x0] + g * (1.0 - wx) * wy;
  ${g}[base + y1 * ${r}u + x1] = ${g}[base + y1 * ${r}u + x1] + g * wx * wy;`;return/* wgsl */`
@group(0) @binding(0) var<storage, read> ${p} : array<f32>;
@group(0) @binding(1) var<storage, read_write> ${g} : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= ${3*c*c}u) { return; }
  let cellPix = i % ${c*c}u;
  let ch = i / ${c*c}u;
  let cx = cellPix % ${c}u;
  let cy = cellPix / ${c}u;
  if (cx >= ${o}u || cy >= ${l}u) { return; }
  let dstPix = (${n}u + cy) * 256u + (${s}u + cx);
${f}
}`}class p{constructor(e,t,r,a,i,s,n,o,d,l,c){for(this.device=e,this.gridImageBuffer=i,this.gridImageOffset=s,this.ownsRaster=l,this.raster=t,this.scratchImage=r,this.scratchGrad=a,this.cells=n,this.directRaster=o,this.scratchImageBytes=d,this.scratchIO=t.createIOState(r,0,a,0,{privateState:!0}),this.retainsCellState=c&&80===t.dims.H&&80===t.dims.W,this.scratchIOs=[this.scratchIO];this.scratchIOs.length<9;)this.scratchIOs.push(this.retainsCellState?t.createIOState(r,0,a,0,{privateState:!0}):this.scratchIO)}static async create(e,t,r,a={}){let o=0|Math.max(0,a.gridLane??0);if(r.batch<=o)throw Error(`grid9_close2: grid lane ${o} outside CLIP batch ${r.batch}`);let c=function(e,t){let r=Number.isFinite(e)?0|e:t?80:256;if(80===r||256===r||512===r)return r;throw Error(`grid9_close2: unsupported raster side ${r}; expected 80, 256, or 512`)}(a.rasterSide,a.directRaster),u=a.packedGrid??512===c,g=256!==c,f=Math.max(0,Number.isFinite(a.gradientScale)?a.gradientScale:1),m=a.scratchRaster?a.scratchRaster:g?await (0,s.Raster3DEngine).create(e,{H:c,W:c,G:t.dims.G,cap:t.dims.cap,bg:t.dims.bg,dynamicBg:t.dims.dynamicBg,dynamicBgTexture:void 0!==a.backgroundTextureMode,backgroundTextureMode:a.backgroundTextureMode,backgroundSeed:a.backgroundSeed,dynamicCoverage:t.dims.dynamicCoverage,dynamicTransmittance:t.dims.dynamicTransmittance,dynamicEntropy:t.dims.dynamicEntropy,dynamicFootprint:t.dims.dynamicFootprint,near:t.dims.near,far:t.dims.far,gradScale:t.dims.gradScale,cameras:t.cameras.map(e=>(0,i.prepareCamera)(e,c)),sharedParams:t.params,sharedGradRaw:t.gradRaw}):t,v=g&&!a.scratchRaster,w=3*c*c*4,b=e.createBuffer({label:`grid9-close2-scratch-image-lane-${o}`,size:w,usage:n.STORAGE|n.COPY_SRC|n.COPY_DST}),x=e.createBuffer({label:`grid9-close2-scratch-grad-lane-${o}`,size:w,usage:n.STORAGE|n.COPY_SRC|n.COPY_DST}),y=r.slotOffsetBytes(o,r.plan.inputSlot),_=r.inputGradOffsetBytes(o),B=[];for(let t=0;t<9;t++){let a=d(t,u),i=3*a.maxSide*a.maxSide,s=await l(e,h(t,"downsample",c,1,u),`grid-copy-${t}`),n=await l(e,h(t,"scatter",c,f,u),`grid-scatter-${t}`),o=e.createBindGroup({layout:s.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:b,offset:0,size:w}},{binding:1,resource:{buffer:r.inputBuffer,offset:y,size:786432}}]}),p=e.createBindGroup({layout:n.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r.inputGradBuffer,offset:_,size:786432}},{binding:1,resource:{buffer:x,offset:0,size:w}}]});B.push({copyPipe:s,copyBind:o,scatterPipe:n,scatterBind:p,workItems:i})}return new p(e,m,b,x,r.inputBuffer,y,B,g,w,v,a.retainCellState??80===c)}clearGridImage(e){e.clearBuffer(this.gridImageBuffer,this.gridImageOffset,786432)}clearScratchGrad(e){e.clearBuffer(this.scratchGrad,0,this.scratchImageBytes)}recordCopyCell(e,t,r){let a=this.cell(t),i=c(e,r);i.setPipeline(a.copyPipe),i.setBindGroup(0,a.copyBind),i.dispatchWorkgroups(Math.ceil(a.workItems/256)),i.end()}recordScatterCell(e,t,r){let a=this.cell(t),i=c(e,r);i.setPipeline(a.scatterPipe),i.setBindGroup(0,a.scatterBind),i.dispatchWorkgroups(Math.ceil(a.workItems/256)),i.end()}scratchIOForCell(e){let t=this.scratchIOs[0|e];if(!t)throw Error(`grid9_close2: bad scratch cell ${e}`);return t}destroy(){this.scratchImage.destroy(),this.scratchGrad.destroy(),this.directRaster&&this.ownsRaster&&this.raster.destroy()}cell(e){let t=this.cells[0|e];if(!t)throw Error(`grid9_close2: bad cell ${e}`);return t}}},{"./cameras":"3Dl1Z","./raster":"AoYYi","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],AoYYi:[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"DEFAULT_3D_LRS",()=>l),a.export(r,"Raster3DEngine",()=>h);var i=e("../splat/adam_wgsl"),s=e("./background_textures"),n=e("./raster_wgsl");let o={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},d=e=>Math.ceil(e/256),l={position:.025,logRadius:.01,color:.08,opacity:.03};async function c(e,t,r){e.pushErrorScope("validation");let a=e.createShaderModule({code:t}),i=e.createComputePipeline({layout:"auto",compute:{module:a,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw console.error(`--- WGSL that failed (${r}) ---
${t}`),Error(`raster3d pipeline validation (${r}): ${s.message}`);return i}function u(e,t){return t?e.beginComputePass({timestampWrites:t}):e.beginComputePass()}class h{constructor(e,t){if(this.ownsParams=!0,this.ownsGradRaw=!0,this.prepPipe=[],this.chainPipe=[],this.transmittancePipe=null,this.bgUni=null,this.backgroundTextureGenerator=null,this.coverageUni=null,this.transmittanceSum=null,this.footprintUni=null,this.prepBind=[],this.chainBind=[],this.transmittanceBind=null,this.adamUni=[],this.adamBind=[],this.extraBuffers=[],this.device=e,this.dims=(0,n.resolveDims3D)(t),this.cameras=t.cameras,!this.cameras.length)throw Error("raster3d: at least one camera is required")}static async create(e,t){let r=new h(e,t);return await r.build(t),r}storage(e,t=0){return this.device.createBuffer({size:4*e,usage:o.STORAGE|t})}bindGroup(e,t){return this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:t.map((e,t)=>({binding:t,resource:"buffer"in e?e:{buffer:e}}))})}async build(e){let t=this.dims,r=t.G*n.PARAM_STRIDE_3D,a=t.G*n.DERIVED_STRIDE_3D;this.params=e.sharedParams??this.storage(r,o.COPY_SRC|o.COPY_DST),this.ownsParams=!e.sharedParams,this.derived=this.storage(a),this.accGrad=this.storage(a,o.COPY_DST),this.gradRaw=e.sharedGradRaw??this.storage(r,o.COPY_SRC|o.COPY_DST),this.ownsGradRaw=!e.sharedGradRaw,this.mBuf=this.storage(r,o.COPY_DST),this.vBuf=this.storage(r,o.COPY_DST),this.tileCounts=this.storage(t.numTiles,o.COPY_DST|o.COPY_SRC),this.binnedIds=this.storage(t.numTiles*t.cap),this.tileStop=this.storage(t.numTiles,o.COPY_SRC),this.image=this.storage(3*t.H*t.W,o.COPY_SRC),this.gradImage=this.storage(3*t.H*t.W,o.COPY_DST),this.cameraBuffer=this.device.createBuffer({label:"splat3d-cameras",size:this.cameras.length*n.CAMERA_STRIDE_3D*4,usage:o.STORAGE|o.COPY_DST}),this.device.queue.writeBuffer(this.cameraBuffer,0,function(e){let t=new Float32Array(e.length*n.CAMERA_STRIDE_3D);for(let r=0;r<e.length;r++){let a=e[r],i=r*n.CAMERA_STRIDE_3D;t[i+0]=a.eye[0],t[i+1]=a.eye[1],t[i+2]=a.eye[2],t[i+3]=a.right[0],t[i+4]=a.right[1],t[i+5]=a.right[2],t[i+6]=a.cameraUp[0],t[i+7]=a.cameraUp[1],t[i+8]=a.cameraUp[2],t[i+9]=a.forward[0],t[i+10]=a.forward[1],t[i+11]=a.forward[2],t[i+12]=a.focalPx}return t}(this.cameras)),e.backgroundTextureMode?this.backgroundTextureGenerator=await (0,s.BackgroundTextureGenerator).create(this.device,{H:t.H,W:t.W,mode:e.backgroundTextureMode,seed:e.backgroundSeed??0}):t.dynamicBg&&(this.bgUni=this.device.createBuffer({label:"splat3d-background",size:16,usage:o.UNIFORM|o.COPY_DST}),this.setBackground(t.bg)),t.dynamicCoverage&&(this.coverageUni=this.device.createBuffer({label:"splat3d-coverage",size:n.COVERAGE_UNIFORM_BYTES_3D,usage:o.UNIFORM|o.COPY_DST}),t.dynamicTransmittance&&(this.transmittanceSum=this.storage(1,o.COPY_DST)),this.setCoverageRegularizer({transmittanceWeight:0,targetTransmittance:.6,rayDistortionWeight:0,rayEntropyWeight:0,rayEntropyMask:.05})),t.dynamicFootprint&&(this.footprintUni=this.device.createBuffer({label:"splat3d-footprint",size:16,usage:o.UNIFORM|o.COPY_DST}),this.setScreenVariance(.0625)),this.prepPipe=await Promise.all(this.cameras.map((t,r)=>c(this.device,(0,n.prepShader3D)(e,t),`prep-${r}`))),this.chainPipe=await Promise.all(this.cameras.map((t,r)=>c(this.device,(0,n.chainAddShader3D)(e,t),`chain-${r}`))),this.emitPipe=await c(this.device,(0,n.emitShader3D)(e),"emit"),this.fwdPipe=await c(this.device,(0,n.forwardShader3D)(e),"forward"),this.transmittancePipe=t.dynamicTransmittance?await c(this.device,(0,n.transmittanceReduceShader3D)(e),"transmittance-reduce"):null,this.bwdPipe=await c(this.device,(0,n.backwardShader3D)(e),"backward"),this.clearBinsPipe=await c(this.device,(0,n.clearShader3D)(t.numTiles),"clearBins"),this.clearGradsPipe=await c(this.device,(0,n.clearShader3D)(a),"clearGrads"),this.clearRawPipe=await c(this.device,(0,n.clearShader3D)(r),"clearRawGrad"),this.adamPipe=await c(this.device,(0,i.adamShader)(),"adam"),this.centerReducePipe=await c(this.device,(0,n.centerReduceShader3D)(e),"center-reduce"),this.regularizerPipe=await c(this.device,(0,n.regularizerShader3D)(e),"regularizer"),this.prepBind=this.prepPipe.map(e=>{let t=[this.params,this.derived];return this.footprintUni&&t.push(this.footprintUni),this.bindGroup(e,t)}),this.chainBind=this.chainPipe.map(e=>this.bindGroup(e,[this.accGrad,this.derived,this.params,this.gradRaw])),this.emitBind=this.bindGroup(this.emitPipe,[this.derived,this.tileCounts,this.binnedIds]),this.clearBinsBind=this.bindGroup(this.clearBinsPipe,[this.tileCounts]),this.clearGradsBind=this.bindGroup(this.clearGradsPipe,[this.accGrad]),this.clearRawBind=this.bindGroup(this.clearRawPipe,[this.gradRaw]),this.centerSum=this.storage(4,o.COPY_DST),this.centerReduceBind=this.bindGroup(this.centerReducePipe,[this.params,this.centerSum]),this.regularizerUni=this.device.createBuffer({label:"splat3d-regularizer-uniform",size:n.REGULARIZER_UNIFORM_BYTES_3D,usage:o.UNIFORM|o.COPY_DST}),this.regularizerBind=this.bindGroup(this.regularizerPipe,[this.regularizerUni,this.params,this.gradRaw,this.centerSum]);let d=this.sharedScratchState();for(let e of(this.fwdBind=this.makeForwardBind(d,this.image,0),this.transmittanceBind=this.makeTransmittanceBind(d),this.bwdBind=this.makeBackwardBind(d,this.gradImage,0),(0,n.paramSegments3D)(t.G))){let e=this.device.createBuffer({size:i.ADAM_UNIFORM_BYTES,usage:o.UNIFORM|o.COPY_DST});this.adamUni.push(e),this.adamBind.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradRaw}},{binding:3,resource:{buffer:this.mBuf}},{binding:4,resource:{buffer:this.vBuf}}]}))}}setParams(e){if(e.length!==this.dims.G*n.PARAM_STRIDE_3D)throw Error("setParams3D: wrong length");this.device.queue.writeBuffer(this.params,0,e)}zeroAdamState(){let e=new Float32Array(this.dims.G*n.PARAM_STRIDE_3D);this.device.queue.writeBuffer(this.mBuf,0,e),this.device.queue.writeBuffer(this.vBuf,0,e)}setBackground(e){if(!this.bgUni)return;let t=new Float32Array([e[0],e[1],e[2],0]);this.device.queue.writeBuffer(this.bgUni,0,t)}recordBackgroundGenerate(e,t,r,a=0){this.backgroundTextureGenerator?.recordGenerate(e,t,r,a)}get usesTexturedBackground(){return null!==this.backgroundTextureGenerator}setCoverageRegularizer(e){if(!this.coverageUni)return;let t=new Float32Array([e.transmittanceWeight,e.targetTransmittance,e.rayDistortionWeight,e.rayEntropyWeight,e.rayEntropyMask,0,0,0]);this.device.queue.writeBuffer(this.coverageUni,0,t)}setScreenVariance(e){if(!this.footprintUni)return;let t=new Float32Array([Math.max(0,e),0,0,0]);this.device.queue.writeBuffer(this.footprintUni,0,t)}async readFloats(e,t){let r=this.device.createBuffer({size:4*t,usage:o.MAP_READ|o.COPY_DST}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(1);let i=new Float32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}readImage(){return this.readFloats(this.image,3*this.dims.H*this.dims.W)}readParams(){return this.readFloats(this.params,this.dims.G*n.PARAM_STRIDE_3D)}readRawGrad(){return this.readFloats(this.gradRaw,this.dims.G*n.PARAM_STRIDE_3D)}resetAdamForSplats(e){let t=this.dims.G,r=new Float32Array(1),a=new Float32Array(3);for(let i=0;i<e.length;i++){let s=0|e[i];if(!(s<0)&&!(s>=t))for(let e of[this.mBuf,this.vBuf])this.device.queue.writeBuffer(e,12*s,a),this.device.queue.writeBuffer(e,(3*t+s)*4,r),this.device.queue.writeBuffer(e,(4*t+3*s)*4,a),this.device.queue.writeBuffer(e,(7*t+s)*4,r)}}async readTileTelemetry(){let e=this.dims.numTiles,t=4*e,r=this.device.createBuffer({size:2*t,usage:o.MAP_READ|o.COPY_DST}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(this.tileCounts,0,r,0,t),a.copyBufferToBuffer(this.tileStop,0,r,t,t),this.device.queue.submit([a.finish()]),await r.mapAsync(1);let i=r.getMappedRange(),s=new Uint32Array(i.slice(0,t)),n=new Uint32Array(i.slice(t,2*t));r.unmap(),r.destroy();let d=0,l=0,c=0,u=0,h=0;for(let t=0;t<e;t++){let e=s[t];d=Math.max(d,e),l=Math.max(l,n[t]),h+=e,e>this.dims.cap&&(c+=1,u+=e-this.dims.cap)}return{cap:this.dims.cap,maxCount:d,maxStop:l,overflowTiles:c,overflowPairs:u,totalPairs:h}}createIOState(e,t,r,a,i={}){this.checkIOBinding("image",t),this.checkIOBinding("grad",a);let s=i.privateState?this.createPrivateScratchState():this.sharedScratchState();return this.createIOStateForScratch(s,e,t,r,a)}async createBatchForwardState(e){let t=this.dims,r=0|e.lanes;if(r<1||r>this.cameras.length)throw Error(`raster3d: invalid batch-forward lanes ${e.lanes}`);if(e.imageOffsets.length!==r||e.gradOffsets.length!==r)throw Error("raster3d: batch-forward offsets must match lane count");this.checkContiguousImageOffsets("batch image",e.imageOffsets),this.checkContiguousImageOffsets("batch grad",e.gradOffsets);let a=this.createRawScratchBuffers(r),i=this.storage(t.G*n.DERIVED_STRIDE_3D*r,o.COPY_DST),s=this.device.createBuffer({label:"splat3d-batch-active-views",size:4*r,usage:o.STORAGE|o.COPY_DST}),d=t.dynamicTransmittance?this.storage(r,o.COPY_DST):null;this.extraBuffers.push(a.derived,a.tileCounts,a.binnedIds,a.tileStop,i,s),d&&this.extraBuffers.push(d);let l=await c(this.device,(0,n.prepBatchShader3D)(t),"prep-batch"),u=await c(this.device,(0,n.clearShader3D)(t.numTiles*r),"clearBins-batch"),h=await c(this.device,(0,n.emitBatchShader3D)(t),"emit-batch"),p=await c(this.device,(0,n.forwardBatchShader3D)(t),"forward-batch"),g=t.dynamicTransmittance?await c(this.device,(0,n.transmittanceReduceShader3D)(t,!0),"transmittance-reduce-batch"):null,f=await c(this.device,(0,n.clearShader3D)(t.G*n.DERIVED_STRIDE_3D*r),"clearGrads-batch"),m=await c(this.device,(0,n.backwardBatchShader3D)(t),"backward-batch"),v=e.imageOffsets[0],w=e.gradOffsets[0],b=[this.params,this.cameraBuffer,s,a.derived];this.footprintUni&&b.push(this.footprintUni);let x=this.bindGroup(l,b),y=this.bindGroup(u,[a.tileCounts]),_=this.bindGroup(h,[a.derived,a.tileCounts,a.binnedIds]),B=[a.tileCounts,a.binnedIds,a.derived,{buffer:e.imageBuffer,offset:v,size:this.imageByteSize()*r},a.tileStop],S=this.backgroundBinding();S&&B.push(S);let $=this.bindGroup(p,B),R=this.bindGroup(f,[i]),T=g&&d?this.bindGroup(g,[a.tileCounts,a.binnedIds,a.tileStop,a.derived,d]):null,P=[{buffer:e.gradBuffer,offset:w,size:this.imageByteSize()*r},a.tileCounts,a.binnedIds,a.tileStop,a.derived,i];S&&P.push(S),this.coverageUni&&P.push(this.coverageUni),d&&P.push(d);let C=this.bindGroup(m,P),I=Array.from({length:r},(t,r)=>this.createIOStateForScratch(this.laneScratchState(a,r,i),e.imageBuffer,e.imageOffsets[r],e.gradBuffer,e.gradOffsets[r]));return{lanes:r,activeViews:s,ios:I,prepPipe:l,clearBinsPipe:u,emitPipe:h,fwdPipe:p,clearGradsPipe:f,transmittancePipe:g,bwdPipe:m,prepBind:x,clearBinsBind:y,emitBind:_,fwdBind:$,clearGradsBind:R,transmittanceBind:T,bwdBind:C,transmittanceSum:d}}recordClearRawGrad(e,t){let r=u(e,t);r.setPipeline(this.clearRawPipe),r.setBindGroup(0,this.clearRawBind),r.dispatchWorkgroups(d(this.dims.G*n.PARAM_STRIDE_3D)),r.end()}recordRegularizerAdd(e,t,r){if(!(0!==t.centerWeight||0!==t.radiusWeight||0!==t.opacitySparsity||0!==t.smallRadiusWeight||0!==t.radiusBandWeight))return;let a=new Float32Array(16);a[0]=t.centerWeight,a[1]=t.radiusWeight,a[2]=t.targetRadius,a[3]=t.opacitySparsity,a[4]=t.smallRadiusWeight,a[5]=t.smallRadius,a[6]=t.radiusBandWeight,a[7]=t.minRadius,a[8]=t.maxRadius,this.device.queue.writeBuffer(this.regularizerUni,0,a),0!==t.centerWeight&&e.clearBuffer(this.centerSum,0,16);let i=u(e,r);0!==t.centerWeight&&(i.setPipeline(this.centerReducePipe),i.setBindGroup(0,this.centerReduceBind),i.dispatchWorkgroups(d(this.dims.G))),i.setPipeline(this.regularizerPipe),i.setBindGroup(0,this.regularizerBind),i.dispatchWorkgroups(d(this.dims.G)),i.end()}recordForward(e,t=0,r,a){let i=u(e,a);this.encodeForwardPass(i,t,r),i.end()}recordForwards(e,t,r,a){if(t.length!==r.length)throw Error(`raster3d: ${t.length} views but ${r.length} IO states`);let i=u(e,a);for(let e=0;e<t.length;e++)this.encodeForwardPass(i,t[e],r[e]);i.end()}recordBatchForward(e,t,r,a){if(r.length<1||r.length>t.lanes)throw Error(`raster3d: ${r.length} batch-forward views for ${t.lanes} lanes`);let i=new Uint32Array(t.lanes);for(let e=0;e<r.length;e++)i[e]=this.viewIndex(r[e]);this.device.queue.writeBuffer(t.activeViews,0,i);let s=this.dims,n=u(e,a);n.setPipeline(t.prepPipe),n.setBindGroup(0,t.prepBind),n.dispatchWorkgroups(d(s.G),1,r.length),n.setPipeline(t.clearBinsPipe),n.setBindGroup(0,t.clearBinsBind),n.dispatchWorkgroups(d(s.numTiles*r.length)),n.setPipeline(t.emitPipe),n.setBindGroup(0,t.emitBind),n.dispatchWorkgroups(d(s.G),1,r.length),n.setPipeline(t.fwdPipe),n.setBindGroup(0,t.fwdBind),n.dispatchWorkgroups(s.numTiles,1,r.length),n.end()}recordBatchBackwardAdd(e,t,r,a){if(r.length<1||r.length>t.lanes)throw Error(`raster3d: ${r.length} batch-backward views for ${t.lanes} lanes`);let i=this.dims;t.transmittanceSum&&e.clearBuffer(t.transmittanceSum,0,4*r.length);let s=u(e,a);t.transmittancePipe&&t.transmittanceBind&&(s.setPipeline(t.transmittancePipe),s.setBindGroup(0,t.transmittanceBind),s.dispatchWorkgroups(i.numTiles,1,r.length)),s.setPipeline(t.clearGradsPipe),s.setBindGroup(0,t.clearGradsBind),s.dispatchWorkgroups(d(i.G*n.DERIVED_STRIDE_3D*r.length)),s.setPipeline(t.bwdPipe),s.setBindGroup(0,t.bwdBind),s.dispatchWorkgroups(i.numTiles,1,r.length);for(let e=0;e<r.length;e++){let a=this.viewIndex(r[e]);s.setPipeline(this.chainPipe[a]),s.setBindGroup(0,t.ios[e].chainBind[a]),s.dispatchWorkgroups(d(i.G))}s.end()}encodeForwardPass(e,t=0,r){let a=this.dims,i=this.viewIndex(t);e.setPipeline(this.prepPipe[i]),e.setBindGroup(0,r?.prepBind[i]??this.prepBind[i]),e.dispatchWorkgroups(d(a.G)),e.setPipeline(this.clearBinsPipe),e.setBindGroup(0,r?.clearBinsBind??this.clearBinsBind),e.dispatchWorkgroups(d(a.numTiles)),e.setPipeline(this.emitPipe),e.setBindGroup(0,r?.emitBind??this.emitBind),e.dispatchWorkgroups(d(a.G)),e.setPipeline(this.fwdPipe),e.setBindGroup(0,r?.fwdBind??this.fwdBind),e.dispatchWorkgroups(a.numTiles)}recordBackwardAdd(e,t=0,r,a){let i=this.dims,s=this.viewIndex(t);this.transmittanceSum&&e.clearBuffer(this.transmittanceSum,0,4);let o=u(e,a),l=r?.transmittanceBind??this.transmittanceBind;this.transmittancePipe&&l&&(o.setPipeline(this.transmittancePipe),o.setBindGroup(0,l),o.dispatchWorkgroups(i.numTiles)),o.setPipeline(this.clearGradsPipe),o.setBindGroup(0,r?.clearGradsBind??this.clearGradsBind),o.dispatchWorkgroups(d(i.G*n.DERIVED_STRIDE_3D)),o.setPipeline(this.bwdPipe),o.setBindGroup(0,r?.bwdBind??this.bwdBind),o.dispatchWorkgroups(i.numTiles),o.setPipeline(this.chainPipe[s]),o.setBindGroup(0,r?.chainBind[s]??this.chainBind[s]),o.dispatchWorkgroups(d(i.G)),o.end()}recordAdam(e,t,r=l,a=i.DEFAULT_HYPER,s){let o=(0,n.paramSegments3D)(this.dims.G),c={position:r.position,logRadius:r.logRadius,color:r.color,opacity:r.opacity},h=1-Math.pow(a.beta1,t),p=1-Math.pow(a.beta2,t);o.forEach((e,t)=>{let r=new ArrayBuffer(i.ADAM_UNIFORM_BYTES),s=new Uint32Array(r),n=new Float32Array(r);s[0]=e.offset,s[1]=e.length,n[2]=c[e.name],n[3]=a.beta1,n[4]=a.beta2,n[5]=a.eps,n[6]=h,n[7]=p,this.device.queue.writeBuffer(this.adamUni[t],0,r)});let g=u(e,s);g.setPipeline(this.adamPipe),o.forEach((e,t)=>{g.setBindGroup(0,this.adamBind[t]),g.dispatchWorkgroups(d(e.length))}),g.end()}runForward(e=0){let t=this.device.createCommandEncoder();this.recordForward(t,e),this.device.queue.submit([t.finish()])}destroy(){let e=[this.derived,this.accGrad,this.mBuf,this.vBuf,this.tileCounts,this.binnedIds,this.tileStop,this.image,this.gradImage,this.cameraBuffer,...this.extraBuffers,...this.adamUni,this.centerSum,this.regularizerUni];for(let t of(this.bgUni&&e.push(this.bgUni),this.coverageUni&&e.push(this.coverageUni),this.transmittanceSum&&e.push(this.transmittanceSum),this.footprintUni&&e.push(this.footprintUni),this.ownsParams&&e.push(this.params),this.ownsGradRaw&&e.push(this.gradRaw),e))try{t.destroy()}catch(e){}this.backgroundTextureGenerator?.destroy()}viewIndex(e){return Math.max(0,Math.min(this.cameras.length-1,0|e))}imageByteSize(){return 3*this.dims.H*this.dims.W*4}sharedScratchState(){return{derived:this.derived,accGrad:this.accGrad,tileCounts:this.tileCounts,binnedIds:this.binnedIds,tileStop:this.tileStop,prepBind:this.prepBind,chainBind:this.chainBind,emitBind:this.emitBind,clearBinsBind:this.clearBinsBind,clearGradsBind:this.clearGradsBind}}createPrivateScratchState(){let{derived:e,tileCounts:t,binnedIds:r,tileStop:a}=this.createRawScratchBuffers(1);return this.extraBuffers.push(e,t,r,a),{derived:e,accGrad:this.accGrad,tileCounts:t,binnedIds:r,tileStop:a,prepBind:this.prepPipe.map(t=>{let r=[this.params,e];return this.footprintUni&&r.push(this.footprintUni),this.bindGroup(t,r)}),chainBind:this.chainPipe.map(t=>this.bindGroup(t,[this.accGrad,e,this.params,this.gradRaw])),emitBind:this.bindGroup(this.emitPipe,[e,t,r]),clearBinsBind:this.bindGroup(this.clearBinsPipe,[t]),clearGradsBind:this.clearGradsBind}}createRawScratchBuffers(e){let t=this.dims;return{derived:this.storage(t.G*n.DERIVED_STRIDE_3D*e),tileCounts:this.storage(t.numTiles*e),binnedIds:this.storage(t.numTiles*t.cap*e),tileStop:this.storage(t.numTiles*e)}}laneScratchState(e,t,r){let a=this.dims,i=a.G*n.DERIVED_STRIDE_3D*4,s=4*a.numTiles,o=a.numTiles*a.cap*4,d=this.sliceBinding(e.derived,t*i,i),l=this.sliceBinding(r,t*i,i),c=this.sliceBinding(e.tileCounts,t*s,s),u=this.sliceBinding(e.binnedIds,t*o,o),h=this.sliceBinding(e.tileStop,t*s,s);return{derived:d,accGrad:l,tileCounts:c,binnedIds:u,tileStop:h,prepBind:this.prepPipe.map(e=>{let t=[this.params,d];return this.footprintUni&&t.push(this.footprintUni),this.bindGroup(e,t)}),chainBind:this.chainPipe.map(e=>this.bindGroup(e,[l,d,this.params,this.gradRaw])),emitBind:this.bindGroup(this.emitPipe,[d,c,u]),clearBinsBind:this.bindGroup(this.clearBinsPipe,[c]),clearGradsBind:this.bindGroup(this.clearGradsPipe,[l])}}createIOStateForScratch(e,t,r,a,i){return{prepBind:e.prepBind,chainBind:e.chainBind,emitBind:e.emitBind,clearBinsBind:e.clearBinsBind,clearGradsBind:e.clearGradsBind,fwdBind:this.makeForwardBind(e,t,r),transmittanceBind:this.makeTransmittanceBind(e),bwdBind:this.makeBackwardBind(e,a,i)}}makeForwardBind(e,t,r){let a=[e.tileCounts,e.binnedIds,e.derived,{buffer:t,offset:r,size:this.imageByteSize()},e.tileStop],i=this.backgroundBinding();return i&&a.push(i),this.bindGroup(this.fwdPipe,a)}makeTransmittanceBind(e){return this.transmittancePipe&&this.transmittanceSum?this.bindGroup(this.transmittancePipe,[e.tileCounts,e.binnedIds,e.tileStop,e.derived,this.transmittanceSum]):null}makeBackwardBind(e,t,r){let a=[{buffer:t,offset:r,size:this.imageByteSize()},e.tileCounts,e.binnedIds,e.tileStop,e.derived,e.accGrad],i=this.backgroundBinding();return i&&a.push(i),this.coverageUni&&a.push(this.coverageUni),this.transmittanceSum&&a.push(this.transmittanceSum),this.bindGroup(this.bwdPipe,a)}backgroundBinding(){return this.backgroundTextureGenerator?.buffer??this.bgUni}checkIOBinding(e,t){if(!Number.isInteger(t)||t<0||t%256!=0)throw Error(`raster3d: ${e} offset ${t} must be 256-byte aligned`)}checkContiguousImageOffsets(e,t){if(!t.length)throw Error(`raster3d: empty ${e} offsets`);let r=this.imageByteSize();for(let a=0;a<t.length;a++)if(this.checkIOBinding(e,t[a]),t[a]!==t[0]+a*r)throw Error(`raster3d: ${e} offsets must be contiguous image lanes`)}sliceBinding(e,t,r){return this.checkIOBinding("scratch",t),{buffer:e,offset:t,size:r}}}},{"../splat/adam_wgsl":"kfWkJ","./background_textures":"8Kv6f","./raster_wgsl":"fuyeU","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"8Kv6f":[function(e,t,r){/// <reference types="@webgpu/types" />
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"BackgroundTextureGenerator",()=>d);let i={COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},s={black:0,dark_solid:1,blurred_noise:2,checkerboard:3,fourier:4};function n(e,t){if(!Number.isInteger(e)||e<0||e>4294967295)throw Error(`background_textures: ${t} must be an unsigned 32-bit integer`);return e>>>0}function o(e,t){if(!Number.isSafeInteger(e)||e<=0)throw Error(`background_textures: ${t} must be a positive integer`);return e}class d{constructor(e,t,r,a,i,s){this.destroyed=!1,this.device=e,this.H=t.H,this.W=t.W,this.mode=t.mode,this.seed=t.seed,this.buffer=r,this.uniformBuffers=a,this.pipeline=i,this.bindGroups=s,this.dispatchX=Math.ceil(t.W/8),this.dispatchY=Math.ceil(t.H/8)}static async create(e,t){let r=o(t.H,"H"),a=o(t.W,"W");if(!Object.prototype.hasOwnProperty.call(s,t.mode))throw Error(`background_textures: unsupported mode ${String(t.mode)}`);let l=n(t.seed??0,"seed"),c=r*a;if(!Number.isSafeInteger(c)||c>Math.floor(1431655765))throw Error("background_textures: H*W is too large for planar u32 indexing");let u=12*c,h=Number(e.limits.maxStorageBufferBindingSize),p=Number(e.limits.maxBufferSize);if(u>h||u>p)throw Error(`background_textures: ${u} byte output exceeds device storage limits`);let g=Math.ceil(a/8),f=Math.ceil(r/8),m=Number(e.limits.maxComputeWorkgroupsPerDimension);if(g>m||f>m)throw Error(`background_textures: ${g}x${f} dispatch exceeds device limits`);let v=t.label??`splat3d-background-${t.mode}`,w=e.createBuffer({label:`${v}-rgb`,size:u,usage:i.STORAGE|i.COPY_SRC|i.COPY_DST}),b=Array.from({length:2},(t,r)=>e.createBuffer({label:`${v}-uniforms-${r}`,size:16,usage:i.UNIFORM|i.COPY_DST}));e.pushErrorScope("validation");let x=e.createShaderModule({label:`${v}-shader`,code:function(e,t,r){let a=Math.min(e,t);return/* wgsl */`
struct GenerateUniforms {
  step     : u32,
  seed     : u32,
  strength : f32,
  _pad     : u32,
};

@group(0) @binding(0) var<uniform> uniforms : GenerateUniforms;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

const WIDTH : u32 = ${t}u;
const HEIGHT : u32 = ${e}u;
const PIXELS : u32 = ${e*t}u;
const MODE : u32 = ${s[r]}u;
const TAU : f32 = 6.283185307179586;
const BLURRED_CELL : f32 = ${Math.max(4,.16*a)};
const CHECKER_CELL : u32 = ${Math.max(2,Math.floor(a/8))}u;

fn hash32(value : u32) -> u32 {
  var x = value;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn keyedHash(a : u32, b : u32, c : u32) -> u32 {
  var h = hash32(uniforms.seed ^ 0xa511e9b3u);
  h = hash32(h ^ hash32(uniforms.step + 0x63d83595u));
  h = hash32(h ^ hash32(a + 0x9e3779b9u));
  h = hash32(h ^ hash32(b + 0x85ebca6bu));
  return hash32(h ^ hash32(c + 0xc2b2ae35u));
}

fn random01(a : u32, b : u32, c : u32) -> f32 {
  return f32(keyedHash(a, b, c) >> 8u) * 0.000000059604644775390625;
}

fn valueNoise(pixel : vec2f, channel : u32, octave : u32) -> f32 {
  let cellSize = max(2.0, BLURRED_CELL / exp2(f32(octave)));
  let offset = vec2f(
    random01(channel, octave, 101u),
    random01(channel, octave, 211u)
  ) * cellSize;
  let p = (pixel + offset) / cellSize;
  let base = vec2u(floor(p));
  let f = fract(p);
  let curve = f * f * (vec2f(3.0) - 2.0 * f);
  let salt = channel * 17u + octave * 131u;
  let n00 = random01(base.x, base.y, salt);
  let n10 = random01(base.x + 1u, base.y, salt);
  let n01 = random01(base.x, base.y + 1u, salt);
  let n11 = random01(base.x + 1u, base.y + 1u, salt);
  return mix(mix(n00, n10, curve.x), mix(n01, n11, curve.x), curve.y);
}

fn blurredNoise(pixel : vec2f, channel : u32) -> f32 {
  var total = 0.0;
  var weight = 0.0;
  var amplitude = 1.0;
  for (var octave = 0u; octave < 3u; octave = octave + 1u) {
    total = total + amplitude * valueNoise(pixel, channel, octave);
    weight = weight + amplitude;
    amplitude = amplitude * 0.5;
  }
  return total / weight;
}

fn fourierValue(pixel : vec2f, channel : u32) -> f32 {
  let p = pixel / ${a}.0;
  var total = 0.0;
  var weight = 0.0;
  for (var wave = 0u; wave < 6u; wave = wave + 1u) {
    let angle = TAU * random01(channel, wave, 307u);
    let direction = vec2f(cos(angle), sin(angle));
    let frequency = 0.75 + 5.25 * random01(channel, wave, 401u);
    let phase = TAU * random01(channel, wave, 503u);
    let amplitude = 1.0 / (1.0 + 0.35 * f32(wave));
    total = total + amplitude * sin(TAU * frequency * dot(p, direction) + phase);
    weight = weight + amplitude;
  }
  return clamp(0.5 + 0.5 * total / weight, 0.0, 1.0);
}

fn darkSolid() -> vec3f {
  return vec3f(
    0.015 + 0.075 * random01(1u, 0u, 601u),
    0.012 + 0.070 * random01(1u, 1u, 601u),
    0.018 + 0.080 * random01(1u, 2u, 601u)
  );
}

fn blurredTexture(pixel : vec2f) -> vec3f {
  let baseNoise = blurredNoise(pixel, 0u);
  let detail = vec3f(
    blurredNoise(pixel, 1u),
    blurredNoise(pixel, 2u),
    blurredNoise(pixel, 3u)
  );
  let value = vec3f(baseNoise) * 0.72 + detail * 0.28;
  return vec3f(0.012, 0.010, 0.014) + value * vec3f(0.17, 0.16, 0.19);
}

fn checkerTexture(coord : vec2u) -> vec3f {
  let offset = vec2u(
    keyedHash(7u, 0u, 701u) % CHECKER_CELL,
    keyedHash(7u, 1u, 701u) % CHECKER_CELL
  );
  let checker = ((coord.x + offset.x) / CHECKER_CELL + (coord.y + offset.y) / CHECKER_CELL) & 1u;
  let low = vec3f(
    0.012 + 0.025 * random01(0u, 0u, 709u),
    0.014 + 0.022 * random01(0u, 1u, 709u),
    0.018 + 0.028 * random01(0u, 2u, 709u)
  );
  let high = vec3f(
    0.105 + 0.080 * random01(1u, 0u, 719u),
    0.095 + 0.075 * random01(1u, 1u, 719u),
    0.115 + 0.085 * random01(1u, 2u, 719u)
  );
  return select(low, high, checker == 1u);
}

fn fourierTexture(pixel : vec2f) -> vec3f {
  let baseWave = fourierValue(pixel, 0u);
  let value = vec3f(
    0.68 * baseWave + 0.32 * fourierValue(pixel, 1u),
    0.72 * baseWave + 0.28 * fourierValue(pixel, 2u),
    0.64 * baseWave + 0.36 * fourierValue(pixel, 3u)
  );
  return vec3f(0.012, 0.010, 0.014) + value * vec3f(0.25, 0.23, 0.28);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= WIDTH || gid.y >= HEIGHT) { return; }

  let index = gid.y * WIDTH + gid.x;
  let pixel = vec2f(gid.xy) + vec2f(0.5);
  var rgb = vec3f(0.0);
  if (MODE == 1u) {
    rgb = darkSolid();
  } else if (MODE == 2u) {
    rgb = blurredTexture(pixel);
  } else if (MODE == 3u) {
    rgb = checkerTexture(gid.xy);
  } else if (MODE == 4u) {
    rgb = fourierTexture(pixel);
  }
  rgb = clamp(rgb * uniforms.strength, vec3f(0.0), vec3f(1.0));

  output[index] = rgb.r;
  output[PIXELS + index] = rgb.g;
  output[2u * PIXELS + index] = rgb.b;
}
`}(r,a,t.mode)}),y=e.createComputePipeline({label:`${v}-pipeline`,layout:"auto",compute:{module:x,entryPoint:"main"}}),_=await e.popErrorScope();if(_){for(let e of(w.destroy(),b))e.destroy();throw Error(`background_textures: pipeline validation failed: ${_.message}`)}let B=b.map((t,r)=>e.createBindGroup({label:`${v}-bind-group-${r}`,layout:y.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:w}}]}));return new d(e,{H:r,W:a,mode:t.mode,seed:l,label:v},w,b,y,B)}recordGenerate(e,t,r,a=0){if(this.destroyed)throw Error("background_textures: generator has been destroyed");let i=n(t,"step");if(!Number.isFinite(r))throw Error("background_textures: strength must be finite");let s=new ArrayBuffer(16),o=new Uint32Array(s),d=new Float32Array(s);if(o[0]=i,o[1]=this.seed,d[2]=Math.max(0,Math.min(1,r)),!Number.isInteger(a)||a<0||a>=this.uniformBuffers.length)throw Error(`background_textures: uniform slot ${a} is outside [0, ${this.uniformBuffers.length})`);this.device.queue.writeBuffer(this.uniformBuffers[a],0,s);let l=e.beginComputePass({label:`splat3d-background-${this.mode}-generate`});l.setPipeline(this.pipeline),l.setBindGroup(0,this.bindGroups[a]),l.dispatchWorkgroups(this.dispatchX,this.dispatchY),l.end()}destroy(){if(!this.destroyed)for(let e of(this.destroyed=!0,this.buffer.destroy(),this.uniformBuffers))e.destroy()}}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],fuyeU:[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"TILE",()=>i),a.export(r,"PARAM_STRIDE_3D",()=>s),a.export(r,"DERIVED_STRIDE_3D",()=>n),a.export(r,"CAMERA_STRIDE_3D",()=>o),a.export(r,"COVERAGE_UNIFORM_BYTES_3D",()=>d),a.export(r,"TRANSMITTANCE_SUM_SCALE_3D",()=>l),a.export(r,"ALPHA_THRESHOLD",()=>c),a.export(r,"MAX_ALPHA",()=>u),a.export(r,"TRANSMITTANCE_CUTOFF",()=>h),a.export(r,"EPS",()=>p),a.export(r,"RADIUS_MIN",()=>g),a.export(r,"RADIUS_MAX",()=>f),a.export(r,"resolveDims3D",()=>x),a.export(r,"prepShader3D",()=>D),a.export(r,"prepBatchShader3D",()=>M),a.export(r,"emitShader3D",()=>O),a.export(r,"emitBatchShader3D",()=>z),a.export(r,"forwardShader3D",()=>W),a.export(r,"forwardBatchShader3D",()=>F),a.export(r,"transmittanceReduceShader3D",()=>L),a.export(r,"backwardShader3D",()=>N),a.export(r,"backwardBatchShader3D",()=>j),a.export(r,"chainAddShader3D",()=>U),a.export(r,"clearShader3D",()=>q),a.export(r,"REGULARIZER_UNIFORM_BYTES_3D",()=>Y),a.export(r,"CENTER_SUM_SCALE_3D",()=>V),a.export(r,"centerReduceShader3D",()=>H),a.export(r,"regularizerShader3D",()=>X),a.export(r,"paramSegments3D",()=>Z);let i=16,s=8,n=11,o=16,d=32,l=4096,c=1/255,u=.99,h=1e-4,p=1e-8,g=.01,f=.45;function m(e,t){if(!e)throw Error(`raster3d_wgsl: ${t}`)}function v(e){m(Number.isFinite(e),`non-finite literal ${e}`);let t=e.toString();return/[.eE]/.test(t)||(t+=".0"),t}let w=e=>`${e>>>0}u`,b=e=>`vec3f(${v(e[0])}, ${v(e[1])}, ${v(e[2])})`;function x(e){return m(e.H>0&&e.W>0&&e.G>0,"H,W,G must be positive"),m(e.H%i==0&&e.W%i==0,`H,W must be multiples of ${i}`),m((e.cap&e.cap-1)==0&&e.cap>0,"cap must be a power of two"),m(e.cap>=256,"cap must be at least one 256-thread tile"),m(4*e.cap<=16384,`cap*4 (${4*e.cap}B) exceeds 16KB workgroup storage`),{H:e.H,W:e.W,G:e.G,cap:e.cap,tilesX:e.W/i,tilesY:e.H/i,numTiles:e.W/i*(e.H/i),bg:e.bg??[0,0,0],dynamicBg:(e.dynamicBg??!1)||(e.dynamicBgTexture??!1),dynamicBgTexture:e.dynamicBgTexture??!1,dynamicCoverage:(e.dynamicCoverage??!1)||(e.dynamicTransmittance??!1)||(e.dynamicEntropy??!1),dynamicTransmittance:e.dynamicTransmittance??!1,dynamicEntropy:e.dynamicEntropy??!1,dynamicFootprint:e.dynamicFootprint??!1,near:e.near??.2,far:e.far??12,gradScale:e.gradScale??65536}}function y(e){return{position:0,logRadius:3*e.G,colorRaw:4*e.G,opacityRaw:7*e.G}}let _=/* wgsl */"fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }";function B(e,t){return e.dynamicFootprint?`
struct FootprintU { variancePx2 : f32, _pad0 : f32, _pad1 : f32, _pad2 : f32 };
@group(0) @binding(${t}) var<uniform> footprintU : FootprintU;
`:""}function S(e,t){return e.dynamicFootprint?`${t} * ${t} + max(footprintU.variancePx2, 0.0)`:`max(${t}, 0.25) * max(${t}, 0.25)`}function $(e,t){return e.dynamicBg?e.dynamicBgTexture?`@group(0) @binding(${t}) var<storage, read> backgroundImage : array<f32>;`:/* wgsl */`
struct BgU {
  rgb : vec3f,
  _pad : f32,
};
@group(0) @binding(${t}) var<uniform> bgU : BgU;
`:""}function R(e,t){return e.dynamicBg?e.dynamicBgTexture?`backgroundImage[${t}u * ${w(e.H*e.W)} + pix]`:0===t?"bgU.rgb.x":1===t?"bgU.rgb.y":"bgU.rgb.z":v(e.bg[t])}function T(e){if(!e.dynamicCoverage)return"";let t=e.dynamicBg?7:6,r=e.dynamicTransmittance?`@group(0) @binding(${t+1}) var<storage, read_write> transmittanceSum : array<atomic<u32>>;`:"";return`
struct CoverageU {
  transmittanceWeight : f32,
  targetTransmittance : f32,
  rayDistortionWeight : f32,
  rayEntropyWeight    : f32,
  rayEntropyMask      : f32,
  _pad0       : f32,
  _pad1       : f32,
  _pad2       : f32,
};
@group(0) @binding(${t}) var<uniform> coverageU : CoverageU;

${r}`}function P(e,t,r="0u"){return e.dynamicTransmittance?/* wgsl */`
  let meanT = f32(atomicLoad(&transmittanceSum[${r}])) * ${v(1/(l*t))};
  if (meanT < clamp(coverageU.targetTransmittance, 0.0, 1.0)) {
    gT = gT - coverageU.transmittanceWeight * ${v(1/t)};
  }
`:""}function C(e,t){return e.dynamicCoverage?`coverageU.rayDistortionWeight * ${v(2/t)}`:"0.0"}function I(e,t){return e.dynamicEntropy?`
    if (totalAlpha > coverageU.rayEntropyMask) {
      let probability = alpha / max(totalAlpha, ${v(p)});
      gAlpha = gAlpha - coverageU.rayEntropyWeight * ${v(1/t)} *
        (log(max(probability, ${v(p)})) + rayEntropy) / max(totalAlpha, ${v(p)});
    }
`:""}function G(e){return e.dynamicEntropy?`
  var totalAlpha = 0.0;
  var totalAlphaLogAlpha = 0.0;`:""}function A(e){return e.dynamicEntropy?`
    totalAlpha = totalAlpha + alpha;
    totalAlphaLogAlpha = totalAlphaLogAlpha + alpha * log(max(alpha, ${v(p)}));`:""}function k(e){return e.dynamicEntropy?`
  let rayEntropy = log(max(totalAlpha, ${v(p)})) -
    totalAlphaLogAlpha / max(totalAlpha, ${v(p)});`:""}function E(e){return/* wgsl */`
const CAM_EYE = ${b(e.eye)};
const CAM_RIGHT = ${b(e.right)};
const CAM_UP = ${b(e.cameraUp)};
const CAM_FWD = ${b(e.forward)};
const FOCAL_PX = ${v(e.focalPx)};
`}function D(e,t){let r=x(e),a=y(r);return/* wgsl */`
${_}
${E(t)}
@group(0) @binding(0) var<storage, read>       params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;
${B(r,2)}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${w(r.G)}) { return; }

  let p = vec3f(
    params[${w(a.position)} + g * 3u + 0u],
    params[${w(a.position)} + g * 3u + 1u],
    params[${w(a.position)} + g * 3u + 2u]
  );
  let w = p - CAM_EYE;
  let vx = dot(w, CAM_RIGHT);
  let vy = dot(w, CAM_UP);
  let vz = dot(w, CAM_FWD);
  let safeZ = max(vz, ${v(r.near)});
  let radiusWorld = clamp(exp(params[${w(a.logRadius)} + g]), ${v(g)}, ${v(f)});
  let radiusPx = FOCAL_PX * radiusWorld / safeZ;
  let invR2 = 1.0 / max(${S(r,"radiusPx")}, ${v(p)});
  let sx = ${v(.5*r.W)} + FOCAL_PX * (vx / safeZ);
  let sy = ${v(.5*r.H)} - FOCAL_PX * (vy / safeZ);

  let base = g * ${w(n)};
  derived[base + 0u] = sx;
  derived[base + 1u] = sy;
  derived[base + 2u] = invR2;
  derived[base + 3u] = vz;
  derived[base + 4u] = vx;
  derived[base + 5u] = vy;
  derived[base + 6u] = safeZ;
  derived[base + 7u] = sigmoid1(params[${w(a.colorRaw)} + g * 3u + 0u]);
  derived[base + 8u] = sigmoid1(params[${w(a.colorRaw)} + g * 3u + 1u]);
  derived[base + 9u] = sigmoid1(params[${w(a.colorRaw)} + g * 3u + 2u]);
  derived[base + 10u] = sigmoid1(params[${w(a.opacityRaw)} + g]);
}
`}function M(e){let t=x(e),r=y(t);return/* wgsl */`
${_}
@group(0) @binding(0) var<storage, read>       params      : array<f32>;
@group(0) @binding(1) var<storage, read>       cameras     : array<f32>;
@group(0) @binding(2) var<storage, read>       activeViews : array<u32>;
@group(0) @binding(3) var<storage, read_write> derived     : array<f32>;
${B(t,4)}


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
  let safeZ = max(vz, ${v(t.near)});
  let radiusWorld = clamp(exp(params[${w(r.logRadius)} + g]), ${v(g)}, ${v(f)});
  let radiusPx = focalPx * radiusWorld / safeZ;
  let invR2 = 1.0 / max(${S(t,"radiusPx")}, ${v(p)});
  let sx = ${v(.5*t.W)} + focalPx * (vx / safeZ);
  let sy = ${v(.5*t.H)} - focalPx * (vy / safeZ);

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
`}function O(e){let t=x(e);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       derived    : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds  : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${w(t.G)}) { return; }
  let base = g * ${w(n)};
  let depth = derived[base + 3u];
  if (depth <= ${v(t.near)} || depth >= ${v(t.far)}) { return; }
  let op = derived[base + 10u];
  if (op <= ${v(c)}) { return; }
  let ratio = max(${v(c)} / max(op, ${v(p)}), ${v(p)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let sx = derived[base + 0u];
  let sy = derived[base + 1u];
  let invR2 = max(derived[base + 2u], ${v(p)});
  let radius = sqrt(tau / invR2);
  let x0 = max(0, i32(floor(sx - radius - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(sx + radius - 0.5)));
  let y0 = max(0, i32(floor(sy - radius - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(sy + radius - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tx0 = x0 / ${i}; let tx1 = x1 / ${i};
  let ty0 = y0 / ${i}; let ty1 = y1 / ${i};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let tile = u32(ty * ${t.tilesX} + tx);
      let slot = atomicAdd(&tileCounts[tile], 1u);
      if (slot < ${w(t.cap)}) { binnedIds[tile * ${w(t.cap)} + slot] = g; }
    }
  }
}
`}function z(e){let t=x(e);return/* wgsl */`
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
  if (depth <= ${v(t.near)} || depth >= ${v(t.far)}) { return; }
  let op = derived[derivedBase + 10u];
  if (op <= ${v(c)}) { return; }
  let ratio = max(${v(c)} / max(op, ${v(p)}), ${v(p)});
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let sx = derived[derivedBase + 0u];
  let sy = derived[derivedBase + 1u];
  let invR2 = max(derived[derivedBase + 2u], ${v(p)});
  let radius = sqrt(tau / invR2);
  let x0 = max(0, i32(floor(sx - radius - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(sx + radius - 0.5)));
  let y0 = max(0, i32(floor(sy - radius - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(sy + radius - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }

  let tileCountsBase = lane * ${w(t.numTiles)};
  let binnedBase = lane * ${w(t.numTiles*t.cap)};
  let tx0 = x0 / ${i}; let tx1 = x1 / ${i};
  let ty0 = y0 / ${i}; let ty1 = y1 / ${i};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let tile = u32(ty * ${t.tilesX} + tx);
      let slot = atomicAdd(&tileCounts[tileCountsBase + tile], 1u);
      if (slot < ${w(t.cap)}) { binnedIds[binnedBase + tile * ${w(t.cap)} + slot] = g; }
    }
  }
}
`}function W(e){let t=x(e),r=t.H*t.W;return/* wgsl */`
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;
${$(t,5)}

var<workgroup> sh_ids : array<u32, ${t.cap}>;

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
  let x = tileX * ${i}u + (tid % ${i}u);
  let y = tileY * ${i}u + (tid / ${i}u);
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
      let alpha = min(${v(u)}, raw);
      if (alpha < ${v(c)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b + 7u];
      accG = accG + w * derived[b + 8u];
      accB = accB + w * derived[b + 9u];
      T = T * (1.0 - alpha);
      if (T < ${v(h)}) { break; }
    }
    let pix = y * ${w(t.W)} + x;
    image[0u * ${w(r)} + pix] = accR + T * ${R(t,0)};
    image[1u * ${w(r)} + pix] = accG + T * ${R(t,1)};
    image[2u * ${w(r)} + pix] = accB + T * ${R(t,2)};
  }
  workgroupBarrier();
  sh_ids[tid] = localStop;
  workgroupBarrier();
  var reduceOffset = 128u;
  loop {
    if (reduceOffset == 0u) { break; }
    if (tid < reduceOffset) { sh_ids[tid] = max(sh_ids[tid], sh_ids[tid + reduceOffset]); }
    workgroupBarrier();
    reduceOffset = reduceOffset >> 1u;
  }
  if (tid == 0u) { tileStop[tileId] = sh_ids[0]; }
}
`}function F(e){let t=x(e),r=t.H*t.W,a=3*r;return/* wgsl */`
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       derived    : array<f32>;
@group(0) @binding(3) var<storage, read_write> image      : array<f32>;
@group(0) @binding(4) var<storage, read_write> tileStop   : array<u32>;
${$(t,5)}

var<workgroup> sh_ids : array<u32, ${t.cap}>;

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
  let imageBase = lane * ${w(a)};
  let count = min(tileCounts[tileCountsBase + tileId], ${w(t.cap)});
  let start = binnedBase + tileId * ${w(t.cap)};
  let sortN = nextPow2(count);

  for (var i = tid; i < sortN; i = i + 256u) {
    sh_ids[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
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
  let x = tileX * ${i}u + (tid % ${i}u);
  let y = tileY * ${i}u + (tid / ${i}u);
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
      let alpha = min(${v(u)}, raw);
      if (alpha < ${v(c)}) { continue; }
      let w = T * alpha;
      accR = accR + w * derived[b + 7u];
      accG = accG + w * derived[b + 8u];
      accB = accB + w * derived[b + 9u];
      T = T * (1.0 - alpha);
      if (T < ${v(h)}) { break; }
    }
    let pix = y * ${w(t.W)} + x;
    image[imageBase + 0u * ${w(r)} + pix] = accR + T * ${R(t,0)};
    image[imageBase + 1u * ${w(r)} + pix] = accG + T * ${R(t,1)};
    image[imageBase + 2u * ${w(r)} + pix] = accB + T * ${R(t,2)};
  }
  workgroupBarrier();
  sh_ids[tid] = localStop;
  workgroupBarrier();
  var reduceOffset = 128u;
  loop {
    if (reduceOffset == 0u) { break; }
    if (tid < reduceOffset) { sh_ids[tid] = max(sh_ids[tid], sh_ids[tid + reduceOffset]); }
    workgroupBarrier();
    reduceOffset = reduceOffset >> 1u;
  }
  if (tid == 0u) { tileStop[tileStopBase + tileId] = sh_ids[0]; }
}
`}function L(e,t=!1){let r=x(e);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(2) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(3) var<storage, read>       derived    : array<f32>;
@group(0) @binding(4) var<storage, read_write> transmittanceSum : array<atomic<u32>>;

var<workgroup> sh_ids : array<u32, ${r.cap}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u,
        @builtin(local_invocation_index) tid : u32) {
  let tileId = wg.x;
  let lane = ${t?"wg.z":"0u"};
  if (tileId >= ${w(r.numTiles)}) { return; }
  let tileBase = lane * ${w(r.numTiles)};
  let idsBase = lane * ${w(r.numTiles*r.cap)};
  let derivedBase = lane * ${w(r.G*n)};
  let count = min(tileCounts[tileBase + tileId], ${w(r.cap)});
  let stopc = min(count, tileStop[tileBase + tileId]);
  let start = idsBase + tileId * ${w(r.cap)};
  for (var i = tid; i < stopc; i = i + 256u) { sh_ids[i] = binnedIds[start + i]; }
  workgroupBarrier();

  let tileX = tileId % ${w(r.tilesX)};
  let tileY = tileId / ${w(r.tilesX)};
  let x = tileX * ${i}u + (tid % ${i}u);
  let y = tileY * ${i}u + (tid / ${i}u);
  if (x >= ${w(r.W)} || y >= ${w(r.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  var T = 1.0;
  for (var i = 0u; i < stopc; i = i + 1u) {
    let b = derivedBase + sh_ids[i] * ${w(n)};
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let power = -0.5 * derived[b + 2u] * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${v(u)}, derived[b + 10u] * exp(power));
    if (alpha < ${v(c)}) { continue; }
    T = T * (1.0 - alpha);
    if (T < ${v(h)}) { break; }
  }
  atomicAdd(&transmittanceSum[lane], u32(round(clamp(T, 0.0, 1.0) * ${v(l)})));
}
`}function N(e){let t=x(e),r=t.H*t.W,a=v(t.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;
${$(t,6)}
${T(t)}

var<workgroup> sh_ids : array<u32, ${t.cap}>;

fn fixadd(base : u32, slot : u32, v : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(v * ${a}), -2.14e9, 2.14e9)));
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
  let x = tileX * ${i}u + (tid % ${i}u);
  let y = tileY * ${i}u + (tid / ${i}u);
  if (x >= ${w(t.W)} || y >= ${w(t.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${w(t.W)} + x;
  let goR = gradImage[0u * ${w(r)} + pix];
  let goG = gradImage[1u * ${w(r)} + pix];
  let goB = gradImage[2u * ${w(r)} + pix];

  var T = 1.0;
  var endi = stopc;
  var totalWeight = 0.0;
  var totalWeightDepth = 0.0;
${G(t)}
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b = gg * ${w(n)};
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let power = -0.5 * derived[b + 2u] * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${v(u)}, derived[b + 10u] * exp(power));
    if (alpha < ${v(c)}) { continue; }
    let weight = T * alpha;
    let z = clamp((derived[b + 6u] - ${v(t.near)}) / ${v(t.far-t.near)}, 0.0, 1.0);
    totalWeight = totalWeight + weight;
    totalWeightDepth = totalWeightDepth + weight * z;
${A(t)}
    T = T * (1.0 - alpha);
    if (T < ${v(h)}) { endi = i + 1u; break; }
  }

  var Tcur = T;
  var gT = goR * ${R(t,0)} + goG * ${R(t,1)} + goB * ${R(t,2)};
${P(t,r)}
  var suffixWeight = 0.0;
  var suffixWeightDepth = 0.0;
${k(t)}
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
    let alpha = min(${v(u)}, raw);
    if (alpha < ${v(c)}) { continue; }
    let denom = max(1.0 - alpha, ${v(p)});
    let Tprev = Tcur / denom;
    let weight = Tprev * alpha;
    let z = clamp((derived[b + 6u] - ${v(t.near)}) / ${v(t.far-t.near)}, 0.0, 1.0);
    let prefixWeight = totalWeight - weight - suffixWeight;
    let prefixWeightDepth = totalWeightDepth - weight * z - suffixWeightDepth;
    let distortionScale = ${C(t,r)};
    let gWeight = distortionScale *
      (z * prefixWeight - prefixWeightDepth + suffixWeightDepth - z * suffixWeight);
    let gDepth = distortionScale * weight * (prefixWeight - suffixWeight) / ${v(t.far-t.near)};
    let cR = derived[b + 7u]; let cG = derived[b + 8u]; let cB = derived[b + 9u];
    let dotgc = goR * cR + goG * cG + goB * cB + gWeight;
    var gAlpha = Tprev * (dotgc - gT);
${I(t,r)}

    fixadd(b, 7u, goR * Tprev * alpha);
    fixadd(b, 8u, goG * Tprev * alpha);
    fixadd(b, 9u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${v(u)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-invR2 * dx);
    let gdy = gPower * (-invR2 * dy);
    fixadd(b, 0u, -gdx);
    fixadd(b, 1u, -gdy);
    fixadd(b, 2u, gPower * (-0.5) * (dx * dx + dy * dy));
    fixadd(b, 6u, gDepth);
    fixadd(b, 10u, gRaw * (raw / max(op, ${v(p)})));

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
    suffixWeight = suffixWeight + weight;
    suffixWeightDepth = suffixWeightDepth + weight * z;
  }
}
`}function j(e){let t=x(e),r=t.H*t.W,a=3*r,s=v(t.gradScale);return/* wgsl */`
@group(0) @binding(0) var<storage, read>       gradImage  : array<f32>;
@group(0) @binding(1) var<storage, read>       tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read>       binnedIds  : array<u32>;
@group(0) @binding(3) var<storage, read>       tileStop   : array<u32>;
@group(0) @binding(4) var<storage, read>       derived    : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad    : array<atomic<i32>>;
${$(t,6)}
${T(t)}

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
  let gradImageBase = lane * ${w(a)};
  let count = min(tileCounts[tileCountsBase + tileId], ${w(t.cap)});
  let stopc = min(count, tileStop[tileStopBase + tileId]);
  let start = binnedBase + tileId * ${w(t.cap)};
  for (var i = tid; i < stopc; i = i + 256u) { sh_ids[i] = binnedIds[start + i]; }
  workgroupBarrier();

  let tileX = tileId % ${w(t.tilesX)};
  let tileY = tileId / ${w(t.tilesX)};
  let x = tileX * ${i}u + (tid % ${i}u);
  let y = tileY * ${i}u + (tid / ${i}u);
  if (x >= ${w(t.W)} || y >= ${w(t.H)}) { return; }
  let pxc = f32(x) + 0.5;
  let pyc = f32(y) + 0.5;
  let pix = y * ${w(t.W)} + x;
  let goR = gradImage[gradImageBase + 0u * ${w(r)} + pix];
  let goG = gradImage[gradImageBase + 1u * ${w(r)} + pix];
  let goB = gradImage[gradImageBase + 2u * ${w(r)} + pix];

  var T = 1.0;
  var endi = stopc;
  var totalWeight = 0.0;
  var totalWeightDepth = 0.0;
${G(t)}
  for (var i = 0u; i < stopc; i = i + 1u) {
    let gg = sh_ids[i];
    let b = derivedBase(lane, gg);
    let dx = pxc - derived[b + 0u];
    let dy = pyc - derived[b + 1u];
    let power = -0.5 * derived[b + 2u] * (dx * dx + dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${v(u)}, derived[b + 10u] * exp(power));
    if (alpha < ${v(c)}) { continue; }
    let weight = T * alpha;
    let z = clamp((derived[b + 6u] - ${v(t.near)}) / ${v(t.far-t.near)}, 0.0, 1.0);
    totalWeight = totalWeight + weight;
    totalWeightDepth = totalWeightDepth + weight * z;
${A(t)}
    T = T * (1.0 - alpha);
    if (T < ${v(h)}) { endi = i + 1u; break; }
  }

  var Tcur = T;
  var gT = goR * ${R(t,0)} + goG * ${R(t,1)} + goB * ${R(t,2)};
${P(t,r,"lane")}
  var suffixWeight = 0.0;
  var suffixWeightDepth = 0.0;
${k(t)}
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
    let alpha = min(${v(u)}, raw);
    if (alpha < ${v(c)}) { continue; }
    let denom = max(1.0 - alpha, ${v(p)});
    let Tprev = Tcur / denom;
    let weight = Tprev * alpha;
    let z = clamp((derived[b + 6u] - ${v(t.near)}) / ${v(t.far-t.near)}, 0.0, 1.0);
    let prefixWeight = totalWeight - weight - suffixWeight;
    let prefixWeightDepth = totalWeightDepth - weight * z - suffixWeightDepth;
    let distortionScale = ${C(t,r)};
    let gWeight = distortionScale *
      (z * prefixWeight - prefixWeightDepth + suffixWeightDepth - z * suffixWeight);
    let gDepth = distortionScale * weight * (prefixWeight - suffixWeight) / ${v(t.far-t.near)};
    let cR = derived[b + 7u]; let cG = derived[b + 8u]; let cB = derived[b + 9u];
    let dotgc = goR * cR + goG * cG + goB * cB + gWeight;
    var gAlpha = Tprev * (dotgc - gT);
${I(t,r)}

    fixadd(b, 7u, goR * Tprev * alpha);
    fixadd(b, 8u, goG * Tprev * alpha);
    fixadd(b, 9u, goB * Tprev * alpha);

    let gate = select(0.0, 1.0, raw < ${v(u)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gdx = gPower * (-invR2 * dx);
    let gdy = gPower * (-invR2 * dy);
    fixadd(b, 0u, -gdx);
    fixadd(b, 1u, -gdy);
    fixadd(b, 2u, gPower * (-0.5) * (dx * dx + dy * dy));
    fixadd(b, 6u, gDepth);
    fixadd(b, 10u, gRaw * (raw / max(op, ${v(p)})));

    gT = alpha * dotgc + (1.0 - alpha) * gT;
    Tcur = Tprev;
    suffixWeight = suffixWeight + weight;
    suffixWeightDepth = suffixWeightDepth + weight * z;
  }
}
`}function U(e,t){let r=x(e),a=y(r),i=v(1/r.gradScale);return/* wgsl */`
${E(t)}
@group(0) @binding(0) var<storage, read>       accGrad : array<i32>;
@group(0) @binding(1) var<storage, read>       derived : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${w(r.G)}) { return; }
  let b = g * ${w(n)};
  let invScale = ${i};
  let gsx = f32(accGrad[b + 0u]) * invScale;
  let gsy = f32(accGrad[b + 1u]) * invScale;
  let gInv = f32(accGrad[b + 2u]) * invScale;
  let gDepth = f32(accGrad[b + 6u]) * invScale;
  let gc0 = f32(accGrad[b + 7u]) * invScale;
  let gc1 = f32(accGrad[b + 8u]) * invScale;
  let gc2 = f32(accGrad[b + 9u]) * invScale;
  let gop = f32(accGrad[b + 10u]) * invScale;

  let vx = derived[b + 4u];
  let vy = derived[b + 5u];
  let vz = max(derived[b + 6u], ${v(r.near)});
  let invR2 = derived[b + 2u];
  let invZ = 1.0 / vz;
  let invZ2 = invZ * invZ;
  let gvx = gsx * FOCAL_PX * invZ;
  let gvy = -gsy * FOCAL_PX * invZ;
  let gvz = gsx * (-FOCAL_PX * vx * invZ2) + gsy * (FOCAL_PX * vy * invZ2) +
    gInv * (2.0 * invR2 * invZ) + gDepth;
  let gp = CAM_RIGHT * gvx + CAM_UP * gvy + CAM_FWD * gvz;

  gradRaw[${w(a.position)} + g * 3u + 0u] = gradRaw[${w(a.position)} + g * 3u + 0u] + gp.x;
  gradRaw[${w(a.position)} + g * 3u + 1u] = gradRaw[${w(a.position)} + g * 3u + 1u] + gp.y;
  gradRaw[${w(a.position)} + g * 3u + 2u] = gradRaw[${w(a.position)} + g * 3u + 2u] + gp.z;

  let lr = params[${w(a.logRadius)} + g];
  let er = exp(lr);
  let rawRadiusPx = FOCAL_PX * er / vz;
  let gateR = select(
    0.0,
    1.0,
    er > ${v(g)} && er < ${v(f)}${r.dynamicFootprint?"":" && rawRadiusPx > 0.25"}
  );
  gradRaw[${w(a.logRadius)} + g] = gradRaw[${w(a.logRadius)} + g] +
    gInv * (-2.0 * rawRadiusPx * rawRadiusPx * invR2 * invR2) * gateR;

  let col0 = derived[b + 7u]; let col1 = derived[b + 8u]; let col2 = derived[b + 9u];
  let opv = derived[b + 10u];
  gradRaw[${w(a.colorRaw)} + g * 3u + 0u] = gradRaw[${w(a.colorRaw)} + g * 3u + 0u] + gc0 * col0 * (1.0 - col0);
  gradRaw[${w(a.colorRaw)} + g * 3u + 1u] = gradRaw[${w(a.colorRaw)} + g * 3u + 1u] + gc1 * col1 * (1.0 - col1);
  gradRaw[${w(a.colorRaw)} + g * 3u + 2u] = gradRaw[${w(a.colorRaw)} + g * 3u + 2u] + gc2 * col2 * (1.0 - col2);
  gradRaw[${w(a.opacityRaw)} + g] = gradRaw[${w(a.opacityRaw)} + g] + gop * opv * (1.0 - opv);
}
`}function q(e){return/* wgsl */`
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= ${w(e)}) { return; }
  buf[gid.x] = 0u;
}
`}let Y=64,V=16384;function H(e){let t=x(e),r=y(t);return/* wgsl */`
${_}
@group(0) @binding(0) var<storage, read>       params    : array<f32>;
@group(0) @binding(1) var<storage, read_write> centerSum : array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${w(t.G)}) { return; }
  let base = ${w(r.position)} + g * 3u;
  let mass = sigmoid1(params[${w(r.opacityRaw)} + g]);
  atomicAdd(&centerSum[0], i32(round(params[base + 0u] * mass * ${v(V)})));
  atomicAdd(&centerSum[1], i32(round(params[base + 1u] * mass * ${v(V)})));
  atomicAdd(&centerSum[2], i32(round(params[base + 2u] * mass * ${v(V)})));
  atomicAdd(&centerSum[3], i32(round(mass * ${v(V)})));
}
`}function X(e){let t=x(e),r=y(t);return/* wgsl */`
${_}
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
@group(0) @binding(3) var<storage, read_write> centerSum : array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${w(t.G)}) { return; }

  let pxIdx = ${w(r.position)} + g * 3u + 0u;
  let pyIdx = ${w(r.position)} + g * 3u + 1u;
  let pzIdx = ${w(r.position)} + g * 3u + 2u;
  let p = vec3f(params[pxIdx], params[pyIdx], params[pzIdx]);
  let r = length(p);
  let invR = 1.0 / max(r, ${v(p)});
  let outside = max(0.0, r - max(u.targetRadius, ${v(p)}));
  let centerScale = 1.0 / max(f32(atomicLoad(&centerSum[3])), 1.0);
  let center = vec3f(
    f32(atomicLoad(&centerSum[0])),
    f32(atomicLoad(&centerSum[1])),
    f32(atomicLoad(&centerSum[2]))
  ) * centerScale;
  let gp = (2.0 * u.centerWeight) * center + (2.0 * u.radiusWeight * outside * invR) * p;
  gradRaw[pxIdx] = gradRaw[pxIdx] + gp.x;
  gradRaw[pyIdx] = gradRaw[pyIdx] + gp.y;
  gradRaw[pzIdx] = gradRaw[pzIdx] + gp.z;

  let opIdx = ${w(r.opacityRaw)} + g;
  let op = sigmoid1(params[opIdx]);
  gradRaw[opIdx] = gradRaw[opIdx] + u.opacitySparsity * op * (1.0 - op);

  let radiusIdx = ${w(r.logRadius)} + g;
  let logRadius = params[radiusIdx];
  let radius = exp(logRadius);
  let small = max(0.0, max(u.smallRadius, ${v(p)}) - radius);
  let smallLossGrad = u.smallRadiusWeight * small * small;
  gradRaw[opIdx] = gradRaw[opIdx] + 2.0 * smallLossGrad * op * op * (1.0 - op);
  gradRaw[radiusIdx] = gradRaw[radiusIdx] - 2.0 * u.smallRadiusWeight * op * op * small * radius;

  let minR = max(u.minRadius, ${v(p)});
  let maxR = max(u.maxRadius, minR + ${v(p)});
  let under = max(0.0, minR - radius);
  let over = max(0.0, radius - maxR);
  let gRadius = u.radiusBandWeight * (-2.0 * under + 2.0 * over);
  gradRaw[radiusIdx] = gradRaw[radiusIdx] + gRadius * radius;
}
`}function Z(e){return[{name:"position",offset:0,length:3*e},{name:"logRadius",offset:3*e,length:e},{name:"color",offset:4*e,length:3*e},{name:"opacity",offset:7*e,length:e}]}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],eTMcA:[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),/**
 * Plans a CPU-side, fixed-count adaptation of isotropic 3D splats.
 *
 * Layout is the current 8G SoA layout:
 * `[position xyz * G, logRadius * G, colorRaw rgb * G, opacityRaw * G]`.
 * The input arrays are never mutated.
 */a.export(r,"planFixedBudgetSplatAdaptation",()=>n);var i=e("./raster_wgsl");let s=2**(-1/3);function n(e,t,r={}){!function(e,t){if(0===e.length||e.length%i.PARAM_STRIDE_3D!=0)throw Error(`splat3d adaptive: params length ${e.length} must be a positive multiple of ${i.PARAM_STRIDE_3D}`);if(t.length!==e.length)throw Error(`splat3d adaptive: raw gradient length ${t.length} != params length ${e.length}`);for(let t=0;t<e.length;t++)if(!Number.isFinite(e[t]))throw Error(`splat3d adaptive: non-finite parameter at offset ${t}`)}(e,t);let a=e.length/i.PARAM_STRIDE_3D,n=function(e,t){let r=e.maxRelocations??Math.max(1,Math.floor(.01*t)),a=e.deadOpacityThreshold??.05,n=e.minParentOpacity??.1,d=e.minParentNeed??0,l=e.minRadius??i.RADIUS_MIN,c=e.maxRadius??i.RADIUS_MAX,u=e.minOpacity??1e-4,h=e.maxOpacity??i.MAX_ALPHA,p=e.splitRadiusScale??s,g=e.splitOffsetScale??.5;if(o(r,0,Number.MAX_SAFE_INTEGER,"maxRelocations"),o(a,0,1,"deadOpacityThreshold"),o(n,0,1,"minParentOpacity"),o(d,0,Number.MAX_VALUE,"minParentNeed"),o(l,Number.MIN_VALUE,Number.MAX_VALUE,"minRadius"),o(c,Number.MIN_VALUE,Number.MAX_VALUE,"maxRadius"),c<=l)throw Error("splat3d adaptive: maxRadius must be greater than minRadius");if(!(u>0&&u<1))throw Error("splat3d adaptive: minOpacity must be in (0, 1)");if(!(h>u&&h<1))throw Error("splat3d adaptive: maxOpacity must be in (minOpacity, 1)");if(!(Number.isFinite(p)&&p>0&&p<=1))throw Error("splat3d adaptive: splitRadiusScale must be in (0, 1]");return o(g,0,100,"splitOffsetScale"),{maxRelocations:Math.min(t,Math.floor(r)),selectionNeed:e.selectionNeed,coverage:e.coverage,deadOpacityThreshold:a,minParentOpacity:n,minParentNeed:d,minRadius:l,maxRadius:c,minOpacity:u,maxOpacity:h,splitRadiusScale:p,splitOffsetScale:g,seed:(e.seed??1)>>>0}}(r,a);if(void 0!==n.coverage&&n.coverage.length!==a)throw Error(`splat3d adaptive: coverage length ${n.coverage.length} != splat count ${a}`);if(void 0!==n.selectionNeed&&n.selectionNeed.length!==a)throw Error(`splat3d adaptive: selectionNeed length ${n.selectionNeed.length} != splat count ${a}`);let y=e.slice(),_=new Set,B=3*a,S=4*a,$=7*a,R=0,T=0;for(let e=0;e<a;e++){let t=B+e,r=Math.exp(y[t]);(r<n.minRadius||r>n.maxRadius)&&(y[t]=p(r,n.minRadius,n.maxRadius),_.add(e),R++);let a=$+e,i=h(y[a]);(i<n.minOpacity||i>n.maxOpacity)&&(y[a]=g(i,n.minOpacity,n.maxOpacity),_.add(e),T++)}let P=m(y,a),C=[];for(let e=0;e<a;e++){let r=c(t[3*e+0]),a=c(t[3*e+1]),i=c(t[3*e+2]),s=void 0===n.selectionNeed?Math.hypot(Math.abs(r),Math.abs(a),Math.abs(i)):u(n.selectionNeed[e]),o=void 0===n.coverage?1:u(n.coverage[e]),d=function(e,t){if(0===e||0===t)return 0;let r=e*t;return Number.isFinite(r)?r:Number.MAX_VALUE}(s,o);C.push({index:e,opacity:h(y[$+e]),gradientMagnitude:s,coverageWeight:o,need:d})}let I=C.filter(e=>e.opacity<=n.deadOpacityThreshold).sort(d),G=new Set(I.map(e=>e.index)),A=C.filter(e=>!G.has(e.index)&&e.opacity>=n.minParentOpacity&&e.need>n.minParentNeed).sort(l),k=Math.min(n.maxRelocations,I.length,A.length),E=[];for(let e=0;e<k;e++){let t=I[e],r=A[e],a=r.index,i=t.index,s=[y[3*a+0],y[3*a+1],y[3*a+2]],o=Math.exp(y[B+a]),d=Math.exp(y[B+i]),l=h(y[$+a]),c=h(y[$+i]),u=f(l,o)+f(c,d),m=x(o*n.splitRadiusScale,n.minRadius,n.maxRadius),R=u/(2*m*m),T=x(1-Math.exp(-R),n.minOpacity,n.maxOpacity),P=p(m,n.minRadius,n.maxRadius),C=g(T,n.minOpacity,n.maxOpacity),G=Math.exp(P),k=function(e,t,r,a){let i=a^Math.imul(e+1,2654435761)^Math.imul(t+1,2246822519)^Math.imul(r+1,3266489917),s=(v(i)+.5)/4294967296,n=(v(668265263^i)+.5)/4294967296,o=2*s-1,d=Math.sqrt(Math.max(0,1-o*o)),l=2*Math.PI*n;return[d*Math.cos(l),d*Math.sin(l),o]}(a,i,e,n.seed),D=G*n.splitOffsetScale,M=[s[0]-k[0]*D,s[1]-k[1]*D,s[2]-k[2]*D],O=[s[0]+k[0]*D,s[1]+k[1]*D,s[2]+k[2]*D];w(y,a,M),w(y,i,O),y[B+a]=P,y[B+i]=P,y[$+a]=C,y[$+i]=C;for(let e=0;e<3;e++)y[S+3*i+e]=y[S+3*a+e];_.add(a),_.add(i),E.push({parentIndex:a,destinationIndex:i,parentNeed:r.need,parentGradientMagnitude:r.gradientMagnitude,parentCoverageWeight:r.coverageWeight,parentPositionBefore:s,parentPositionAfter:b(y,a),childPosition:b(y,i),parentRadiusBefore:o,destinationRadiusBefore:d,radiusAfter:Math.exp(y[B+a]),parentOpacityBefore:l,destinationOpacityBefore:c,opacityAfter:h(y[$+a]),coverageMassBefore:u,coverageMassAfter:2*f(h(y[$+a]),Math.exp(y[B+a]))})}!function(e){for(let t=0;t<e.length;t++)if(!Number.isFinite(e[t]))throw Error(`splat3d adaptive: adaptation produced non-finite parameter at offset ${t}`)}(y);let D=m(y,a),M=A.length>0?A[0].need:0,O=k>0?A[k-1].need:0;return{params:y,changedIndices:Uint32Array.from(Array.from(_).sort((e,t)=>e-t)),relocations:E,diagnostics:{splatCount:a,requestedRelocations:n.maxRelocations,eligibleDestinations:I.length,eligibleParents:A.length,relocationCount:k,radiusClampCount:R,opacityClampCount:T,maxNeed:M,minSelectedNeed:O,coverageMassBefore:P,coverageMassAfter:D,coverageMassRelativeError:Math.abs(D-P)/Math.max(P,1e-12)}}}function o(e,t,r,a){if(!(Number.isFinite(e)&&e>=t&&e<=r))throw Error(`splat3d adaptive: ${a} must be finite and in [${t}, ${r}]`)}function d(e,t){return e.opacity-t.opacity||e.need-t.need||e.index-t.index}function l(e,t){return t.need-e.need||t.gradientMagnitude-e.gradientMagnitude||t.opacity-e.opacity||e.index-t.index}function c(e){return Number.isFinite(e)?e:0}function u(e){return Number.isFinite(e)&&e>0?e:0}function h(e){if(e>=0)return 1/(1+Math.exp(-e));let t=Math.exp(e);return t/(1+t)}function p(e,t,r){let a=Math.min((r-t)*.25,1e-7*Math.max(r,1));return Math.fround(Math.log(x(e,t+a,r-a)))}function g(e,t,r){var a;let i=Math.min((r-t)*.25,1e-7);return Math.fround(Math.log(a=x(e,t+i,r-i))-Math.log1p(-a))}function f(e,t){return-Math.log1p(-e)*t*t}function m(e,t){let r=3*t,a=7*t,i=0;for(let s=0;s<t;s++)i+=f(h(e[a+s]),Math.exp(e[r+s]));return i}function v(e){let t=e>>>0;return((t=Math.imul((t=Math.imul(t^t>>>16,2146121005))^t>>>15,2221713035))^t>>>16)>>>0}function w(e,t,r){e[3*t+0]=r[0],e[3*t+1]=r[1],e[3*t+2]=r[2]}function b(e,t){return[e[3*t+0],e[3*t+1],e[3*t+2]]}function x(e,t,r){return Math.min(r,Math.max(t,e))}},{"./raster_wgsl":"fuyeU","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],t4gra:[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r);var i=e("./layout");a.exportAll(i,r);var s=e("./projection");a.exportAll(s,r);var n=e("./projection_wgsl");a.exportAll(n,r);var o=e("./raster_wgsl");a.exportAll(o,r);var d=e("./raster_engine");a.exportAll(d,r);var l=e("./optimizer");a.exportAll(l,r);var c=e("./adaptive");a.exportAll(c,r)},{"./layout":"d5AMr","./projection":"4agiy","./projection_wgsl":"2WL5b","./raster_wgsl":"jpT8C","./raster_engine":"5Bvz3","./optimizer":"1AKMX","./adaptive":"aSGHB","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],d5AMr:[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"ANISO_PARAM_STRIDE_3D",()=>i),a.export(r,"ANISO_PARAM_COMPONENTS_3D",()=>s),/**
 * SoA segments, compatible with the current 3D optimizer's packing style:
 * position3 | logScale3 | quaternion4 | color3 | opacity1.
 */a.export(r,"anisotropicParamSegments3D",()=>n),a.export(r,"packAnisotropicSplats3D",()=>o),a.export(r,"unpackAnisotropicSplat3D",()=>d);let i=14,s={position:3,logScale:3,quaternion:4,color:3,opacity:1};function n(e){if(!Number.isInteger(e)||e<0)throw Error(`splat3d_aniso: invalid splat count ${e}`);return[{name:"position",offset:0,length:3*e,components:3},{name:"logScale",offset:3*e,length:3*e,components:3},{name:"quaternion",offset:6*e,length:4*e,components:4},{name:"color",offset:10*e,length:3*e,components:3},{name:"opacity",offset:13*e,length:e,components:1}]}function o(e){let t=e.length,r=n(t),a=Object.fromEntries(r.map(e=>[e.name,e.offset])),s=new Float32Array(i*t);for(let r=0;r<t;r++){let t=e[r];s.set(t.position,a.position+3*r),s.set(t.logScale,a.logScale+3*r),s.set(t.quaternion,a.quaternion+4*r),s.set(t.color,a.color+3*r),s[a.opacity+r]=t.opacity}return s}function d(e,t,r){if(!Number.isInteger(r)||r<0||r>=t)throw Error(`splat3d_aniso: splat index ${r} outside [0, ${t})`);if(e.length<i*t)throw Error(`splat3d_aniso: packed length ${e.length} is smaller than ${i*t}`);let a=n(t),s=Object.fromEntries(a.map(e=>[e.name,e.offset])),o=(t,r)=>Array.from({length:r},(r,a)=>e[t+a]);return{position:o(s.position+3*r,3),logScale:o(s.logScale+3*r,3),quaternion:o(s.quaternion+4*r,4),color:o(s.color+3*r,3),opacity:e[s.opacity+r]}}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"4agiy":[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"DEFAULT_ANISOTROPIC_PROJECTION_SETTINGS",()=>i),a.export(r,"normalizeQuaternionXYZW",()=>c),/** Rotation columns for a normalized [x, y, z, w] quaternion. */a.export(r,"quaternionRotationColumnsXYZW",()=>u),a.export(r,"projectAnisotropicGaussian",()=>f),a.export(r,"backwardAnisotropicProjection",()=>m);let i={mode:"legacy-affine",minScale:.01,maxScale:.45,screenVariancePx2:0,quaternionEpsilon:1e-12,determinantEpsilon:1e-12},s=(e,t)=>e[0]*t[0]+e[1]*t[1]+e[2]*t[2],n=(e,t)=>[e[0]*t,e[1]*t,e[2]*t],o=(e,t,r)=>[e[0]+t[0]*r,e[1]+t[1]*r,e[2]+t[2]*r],d=(e,t,r)=>Math.max(t,Math.min(r,e));function l(e){let t={...i,...e};if(!(t.minScale>0)||!(t.maxScale>t.minScale))throw Error("splat3d_aniso: scales require 0 < minScale < maxScale");if(!(t.quaternionEpsilon>0)||!(t.determinantEpsilon>0))throw Error("splat3d_aniso: epsilon values must be positive");if(!(t.screenVariancePx2>=0))throw Error("splat3d_aniso: screenVariancePx2 must be non-negative");return t}function c(e,t=1e-12){let r=Math.max(Math.hypot(e[0],e[1],e[2],e[3]),t);return[e[0]/r,e[1]/r,e[2]/r,e[3]/r]}function u(e){let[t,r,a,i]=e;return[[1-2*(r*r+a*a),2*(t*r+a*i),2*(t*a-r*i)],[2*(t*r-a*i),1-2*(t*t+a*a),2*(r*a+t*i)],[2*(t*a+r*i),2*(r*a-t*i),1-2*(t*t+r*r)]]}function h(e,t){return[s(e,t.right),s(e,t.up),s(e,t.forward)]}function p(e,t){let r=n(t.right,e[0]);return r=o(r,t.up,e[1]),o(r,t.forward,e[2])}function g(e,t,r){let a=l(r),i=[e.position[0]-t.eye[0],e.position[1]-t.eye[1],e.position[2]-t.eye[2]],o=h(i,t),c=Math.max(o[2],t.near),p=o[2]>t.near?1:0,g=function(e,t,r,a){let i=1/r,s=t*i,n=[[s,0,-t*e[0]*i*i],[0,-s,t*e[1]*i*i]];return{mean:n,covariance:"perspective-jacobian"===a?n:[[s,0,0],[0,-s,0]]}}(o,t.focalPx,c,a.mode),f=[t.centerPx[0]+t.focalPx*o[0]/c,t.centerPx[1]-t.focalPx*o[1]/c],m=[Math.exp(e.logScale[0]),Math.exp(e.logScale[1]),Math.exp(e.logScale[2])],v=m.map(e=>d(e,a.minScale,a.maxScale)),w=m.map(e=>e>a.minScale&&e<a.maxScale?1:0),b=Math.hypot(...e.quaternion),x=Math.max(b,a.quaternionEpsilon),y=e.quaternion.map(e=>e/x),_=u(y),B=_.map(e=>h(e,t)),S=B.map((e,t)=>n(e,v[t])),$=S.map(e=>[s(g.covariance[0],e),s(g.covariance[1],e)]),R=a.screenVariancePx2,T=0,P=a.screenVariancePx2;for(let e of $)R+=e[0]*e[0],T+=e[0]*e[1],P+=e[1]*e[1];let C=[R,T,P],I=R*P-T*T,G=Math.max(I,a.determinantEpsilon),A=[P/G,-T/G,R/G];return{meanPx:f,covariance:C,conic:A,cameraPosition:o,scales:v,normalizedQuaternion:y,determinant:G,rawScales:m,scaleGates:w,rotation:_,rotationCamera:B,transformCamera:S,covarianceJacobian:g.covariance,meanJacobian:g.mean,rawDeterminant:I,safeZ:c,zGate:p,quaternionNorm:x,quaternionNormActive:b>a.quaternionEpsilon}}function f(e,t,r){let a=g(e,t,r);return{meanPx:a.meanPx,covariance:a.covariance,conic:a.conic,cameraPosition:a.cameraPosition,scales:a.scales,normalizedQuaternion:a.normalizedQuaternion,determinant:a.determinant}}function m(e,t,r,a){let i=l(a),o=g(e,t,i);if(!(o.rawDeterminant>i.determinantEpsilon))throw Error("splat3d_aniso: covariance determinant floor is active; projection gradient is undefined there");let d=function(e,t){let[r,a,i]=e,[s,n,o]=t,d=.5*n,l=r*s+a*d,c=r*d+a*o;return[-(l*r+c*a),-2*(l*a+c*i),-((a*s+i*d)*a+(a*d+i*o)*i)]}(o.conic,r.conic),[c,u,h]=d,f=o.transformCamera.map(e=>[s(o.covarianceJacobian[0],e),s(o.covarianceJacobian[1],e)]),m=f.map(e=>[2*c*e[0]+u*e[1],u*e[0]+2*h*e[1]]),v=m.map(e=>[o.covarianceJacobian[0][0]*e[0]+o.covarianceJacobian[1][0]*e[1],o.covarianceJacobian[0][1]*e[0]+o.covarianceJacobian[1][1]*e[1],o.covarianceJacobian[0][2]*e[0]+o.covarianceJacobian[1][2]*e[1]]),w=[[0,0,0],[0,0,0]];for(let e=0;e<3;e++)for(let t=0;t<3;t++)w[0][t]+=m[e][0]*o.transformCamera[e][t],w[1][t]+=m[e][1]*o.transformCamera[e][t];let b=o.rotation.map((e,r)=>n(p(v[r],t),o.scales[r])),x=[0,1,2].map(e=>s(v[e],o.rotationCamera[e])*o.rawScales[e]*o.scaleGates[e]),y=function(e,t){let[r,a,i,s]=t,n=e[0][0],o=e[0][1],d=e[0][2],l=e[1][0],c=e[1][1],u=e[1][2],h=e[2][0],p=e[2][1],g=e[2][2];return[2*a*(l+o)+2*i*(h+d)-4*r*(c+g)+2*s*(u-p),-4*a*(n+g)+2*r*(l+o)+2*i*(p+u)+2*s*(h-d),-4*i*(n+c)+2*r*(h+d)+2*a*(p+u)+2*s*(o-l),2*i*(o-l)+2*a*(h-d)+2*r*(u-p)]}(b,o.normalizedQuaternion),_=function(e,t,r,a,i){if(!i)return r.map(e=>e/a);let s=t[0]*r[0]+t[1]*r[1]+t[2]*r[2]+t[3]*r[3];return t.map((e,t)=>(r[t]-e*s)/a)}(e.quaternion,o.normalizedQuaternion,y,o.quaternionNorm,o.quaternionNormActive),B=[o.meanJacobian[0][0]*r.meanPx[0]+o.meanJacobian[1][0]*r.meanPx[1],o.meanJacobian[0][1]*r.meanPx[0]+o.meanJacobian[1][1]*r.meanPx[1],o.meanJacobian[0][2]*r.meanPx[0]+o.meanJacobian[1][2]*r.meanPx[1]],S=1/o.safeZ,$=S*S;return"perspective-jacobian"===i.mode?(B[0]+=w[0][2]*(-t.focalPx*$),B[1]+=w[1][2]*(t.focalPx*$),B[2]+=w[0][0]*(-t.focalPx*$)+w[0][2]*(2*t.focalPx*o.cameraPosition[0]*$*S)+w[1][1]*(t.focalPx*$)+w[1][2]*(-2*t.focalPx*o.cameraPosition[1]*$*S)):B[2]+=w[0][0]*(-t.focalPx*$)+w[1][1]*(t.focalPx*$),B[2]*=o.zGate,{position:p(B,t),logScale:x,quaternion:_}}},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"2WL5b":[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");function i(e={}){let t=e.mode??"legacy-affine",r="perspective-jacobian"===t?/* wgsl */`
  let inv_z = 1.0 / safe_z;
  let inv_z2 = inv_z * inv_z;
  return mat3x2f(
    vec2f(focal_px * inv_z, 0.0),
    vec2f(0.0, -focal_px * inv_z),
    vec2f(-focal_px * camera_position.x * inv_z2, focal_px * camera_position.y * inv_z2)
  );`:/* wgsl */`
  let focal_inv_z = focal_px / safe_z;
  return mat3x2f(
    vec2f(focal_inv_z, 0.0),
    vec2f(0.0, -focal_inv_z),
    vec2f(0.0)
  );`,a="perspective-jacobian"===t?/* wgsl */`
  camera_gradient.x = camera_gradient.x + jacobian_gradient[2].x * (-camera.focal_px * inv_z2);
  camera_gradient.y = camera_gradient.y + jacobian_gradient[2].y * camera.focal_px * inv_z2;
  camera_gradient.z = camera_gradient.z
    + jacobian_gradient[0].x * (-camera.focal_px * inv_z2)
    + jacobian_gradient[2].x * (2.0 * camera.focal_px * projected.camera_position.x * inv_z3)
    + jacobian_gradient[1].y * (camera.focal_px * inv_z2)
    + jacobian_gradient[2].y * (-2.0 * camera.focal_px * projected.camera_position.y * inv_z3);`:/* wgsl */`
  camera_gradient.z = camera_gradient.z
    + jacobian_gradient[0].x * (-camera.focal_px * inv_z2)
    + jacobian_gradient[1].y * (camera.focal_px * inv_z2);`;return/* wgsl */`
struct AnisoCamera {
  eye       : vec3f,
  right     : vec3f,
  up        : vec3f,
  forward   : vec3f,
  focal_px  : f32,
  center_px : vec2f,
  near      : f32,
};

struct AnisoProjectionSettings {
  min_scale            : f32,
  max_scale            : f32,
  screen_variance_px2  : f32,
  quaternion_epsilon   : f32,
  determinant_epsilon  : f32,
};

struct AnisoProjection {
  mean_px               : vec2f,
  covariance            : vec3f,
  conic                 : vec3f,
  camera_position       : vec3f,
  scales                : vec3f,
  normalized_quaternion : vec4f,
  determinant           : f32,
};

struct AnisoProjectionUpstream {
  mean_px : vec2f,
  conic   : vec3f,
};

struct AnisoProjectionGradient {
  position   : vec3f,
  log_scale  : vec3f,
  quaternion : vec4f,
};

fn anisoWorldToCamera(camera : AnisoCamera, vector : vec3f) -> vec3f {
  return vec3f(dot(vector, camera.right), dot(vector, camera.up), dot(vector, camera.forward));
}

fn anisoCameraToWorld(camera : AnisoCamera, vector : vec3f) -> vec3f {
  return camera.right * vector.x + camera.up * vector.y + camera.forward * vector.z;
}

fn anisoNormalizeQuaternion(raw : vec4f, epsilon : f32) -> vec4f {
  return raw * inverseSqrt(max(dot(raw, raw), epsilon * epsilon));
}

// Quaternion order is [x, y, z, w]; the matrix constructor arguments are columns.
fn anisoQuaternionRotation(q : vec4f) -> mat3x3f {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
    vec3f(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
    vec3f(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))
  );
}

// mat3x2f is a 2-row, 3-column matrix. Its columns correspond to camera x/y/z.
fn anisoMeanJacobian(camera_position : vec3f, focal_px : f32, safe_z : f32) -> mat3x2f {
  let inv_z = 1.0 / safe_z;
  let inv_z2 = inv_z * inv_z;
  return mat3x2f(
    vec2f(focal_px * inv_z, 0.0),
    vec2f(0.0, -focal_px * inv_z),
    vec2f(-focal_px * camera_position.x * inv_z2, focal_px * camera_position.y * inv_z2)
  );
}

fn anisoCovarianceJacobian(camera_position : vec3f, focal_px : f32, safe_z : f32) -> mat3x2f {${r}
}

fn anisoProject(
  position : vec3f,
  log_scale : vec3f,
  quaternion : vec4f,
  camera : AnisoCamera,
  settings : AnisoProjectionSettings
) -> AnisoProjection {
  let camera_position = anisoWorldToCamera(camera, position - camera.eye);
  let safe_z = max(camera_position.z, camera.near);
  let mean_px = camera.center_px + vec2f(
    camera.focal_px * camera_position.x / safe_z,
    -camera.focal_px * camera_position.y / safe_z
  );
  let scales = clamp(exp(log_scale), vec3f(settings.min_scale), vec3f(settings.max_scale));
  let normalized_quaternion = anisoNormalizeQuaternion(quaternion, settings.quaternion_epsilon);
  let rotation = anisoQuaternionRotation(normalized_quaternion);
  let transform = mat3x3f(
    anisoWorldToCamera(camera, rotation[0]) * scales.x,
    anisoWorldToCamera(camera, rotation[1]) * scales.y,
    anisoWorldToCamera(camera, rotation[2]) * scales.z
  );
  let jacobian = anisoCovarianceJacobian(camera_position, camera.focal_px, safe_z);
  let axis0 = jacobian * transform[0];
  let axis1 = jacobian * transform[1];
  let axis2 = jacobian * transform[2];
  let covariance = vec3f(
    dot(vec3f(axis0.x, axis1.x, axis2.x), vec3f(axis0.x, axis1.x, axis2.x)) + settings.screen_variance_px2,
    axis0.x * axis0.y + axis1.x * axis1.y + axis2.x * axis2.y,
    dot(vec3f(axis0.y, axis1.y, axis2.y), vec3f(axis0.y, axis1.y, axis2.y)) + settings.screen_variance_px2
  );
  let determinant = max(
    covariance.x * covariance.z - covariance.y * covariance.y,
    settings.determinant_epsilon
  );
  let conic = vec3f(covariance.z, -covariance.y, covariance.x) / determinant;
  return AnisoProjection(
    mean_px,
    covariance,
    conic,
    camera_position,
    scales,
    normalized_quaternion,
    determinant
  );
}

fn anisoConicToCovarianceGradient(conic : vec3f, upstream : vec3f) -> vec3f {
  // The stored off-diagonal appears once in upstream, but twice in a matrix trace.
  let h00 = upstream.x;
  let h01 = 0.5 * upstream.y;
  let h11 = upstream.z;
  let t00 = conic.x * h00 + conic.y * h01;
  let t01 = conic.x * h01 + conic.y * h11;
  let t10 = conic.y * h00 + conic.z * h01;
  let t11 = conic.y * h01 + conic.z * h11;
  return vec3f(
    -(t00 * conic.x + t01 * conic.y),
    -2.0 * (t00 * conic.y + t01 * conic.z),
    -(t10 * conic.y + t11 * conic.z)
  );
}

fn anisoRotationQuaternionGradient(rotation_gradient : mat3x3f, q : vec4f) -> vec4f {
  let g00 = rotation_gradient[0].x; let g10 = rotation_gradient[0].y; let g20 = rotation_gradient[0].z;
  let g01 = rotation_gradient[1].x; let g11 = rotation_gradient[1].y; let g21 = rotation_gradient[1].z;
  let g02 = rotation_gradient[2].x; let g12 = rotation_gradient[2].y; let g22 = rotation_gradient[2].z;
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return vec4f(
    2.0 * y * (g01 + g10) + 2.0 * z * (g02 + g20) - 4.0 * x * (g11 + g22) + 2.0 * w * (g21 - g12),
    -4.0 * y * (g00 + g22) + 2.0 * x * (g01 + g10) + 2.0 * z * (g12 + g21) + 2.0 * w * (g02 - g20),
    -4.0 * z * (g00 + g11) + 2.0 * x * (g02 + g20) + 2.0 * y * (g12 + g21) + 2.0 * w * (g10 - g01),
    2.0 * z * (g10 - g01) + 2.0 * y * (g02 - g20) + 2.0 * x * (g21 - g12)
  );
}

fn anisoNormalizedQuaternionGradient(
  raw : vec4f,
  normalized : vec4f,
  gradient : vec4f,
  epsilon : f32
) -> vec4f {
  let norm2 = dot(raw, raw);
  let norm = sqrt(max(norm2, epsilon * epsilon));
  if (norm2 <= epsilon * epsilon) {
    return gradient / norm;
  }
  return (gradient - normalized * dot(normalized, gradient)) / norm;
}

fn anisoProjectBackward(
  position : vec3f,
  log_scale : vec3f,
  quaternion : vec4f,
  camera : AnisoCamera,
  settings : AnisoProjectionSettings,
  upstream : AnisoProjectionUpstream
) -> AnisoProjectionGradient {
  let projected = anisoProject(position, log_scale, quaternion, camera, settings);
  let safe_z = max(projected.camera_position.z, camera.near);
  let jacobian = anisoCovarianceJacobian(projected.camera_position, camera.focal_px, safe_z);
  let mean_jacobian = anisoMeanJacobian(projected.camera_position, camera.focal_px, safe_z);
  let rotation = anisoQuaternionRotation(projected.normalized_quaternion);
  let rotation_camera = mat3x3f(
    anisoWorldToCamera(camera, rotation[0]),
    anisoWorldToCamera(camera, rotation[1]),
    anisoWorldToCamera(camera, rotation[2])
  );
  let transform = mat3x3f(
    rotation_camera[0] * projected.scales.x,
    rotation_camera[1] * projected.scales.y,
    rotation_camera[2] * projected.scales.z
  );
  let axis0 = jacobian * transform[0];
  let axis1 = jacobian * transform[1];
  let axis2 = jacobian * transform[2];
  let covariance_gradient = anisoConicToCovarianceGradient(projected.conic, upstream.conic);
  let axis_gradient0 = vec2f(
    2.0 * covariance_gradient.x * axis0.x + covariance_gradient.y * axis0.y,
    covariance_gradient.y * axis0.x + 2.0 * covariance_gradient.z * axis0.y
  );
  let axis_gradient1 = vec2f(
    2.0 * covariance_gradient.x * axis1.x + covariance_gradient.y * axis1.y,
    covariance_gradient.y * axis1.x + 2.0 * covariance_gradient.z * axis1.y
  );
  let axis_gradient2 = vec2f(
    2.0 * covariance_gradient.x * axis2.x + covariance_gradient.y * axis2.y,
    covariance_gradient.y * axis2.x + 2.0 * covariance_gradient.z * axis2.y
  );
  let transform_gradient0 = transpose(jacobian) * axis_gradient0;
  let transform_gradient1 = transpose(jacobian) * axis_gradient1;
  let transform_gradient2 = transpose(jacobian) * axis_gradient2;
  let jacobian_gradient = mat3x2f(
    axis_gradient0 * transform[0].x + axis_gradient1 * transform[1].x + axis_gradient2 * transform[2].x,
    axis_gradient0 * transform[0].y + axis_gradient1 * transform[1].y + axis_gradient2 * transform[2].y,
    axis_gradient0 * transform[0].z + axis_gradient1 * transform[1].z + axis_gradient2 * transform[2].z
  );

  let rotation_gradient = mat3x3f(
    anisoCameraToWorld(camera, transform_gradient0) * projected.scales.x,
    anisoCameraToWorld(camera, transform_gradient1) * projected.scales.y,
    anisoCameraToWorld(camera, transform_gradient2) * projected.scales.z
  );
  let raw_scale = exp(log_scale);
  let scale_gate = vec3f(
    select(0.0, 1.0, raw_scale.x > settings.min_scale && raw_scale.x < settings.max_scale),
    select(0.0, 1.0, raw_scale.y > settings.min_scale && raw_scale.y < settings.max_scale),
    select(0.0, 1.0, raw_scale.z > settings.min_scale && raw_scale.z < settings.max_scale)
  );
  let log_scale_gradient = vec3f(
    dot(transform_gradient0, rotation_camera[0]),
    dot(transform_gradient1, rotation_camera[1]),
    dot(transform_gradient2, rotation_camera[2])
  ) * raw_scale * scale_gate;
  let normalized_quaternion_gradient = anisoRotationQuaternionGradient(
    rotation_gradient,
    projected.normalized_quaternion
  );
  let quaternion_gradient = anisoNormalizedQuaternionGradient(
    quaternion,
    projected.normalized_quaternion,
    normalized_quaternion_gradient,
    settings.quaternion_epsilon
  );

  var camera_gradient = transpose(mean_jacobian) * upstream.mean_px;
  let inv_z = 1.0 / safe_z;
  let inv_z2 = inv_z * inv_z;
  let inv_z3 = inv_z2 * inv_z;${a}
  camera_gradient.z = camera_gradient.z * select(0.0, 1.0, projected.camera_position.z > camera.near);

  return AnisoProjectionGradient(
    anisoCameraToWorld(camera, camera_gradient),
    log_scale_gradient,
    quaternion_gradient
  );
}
`}a.defineInteropFlag(r),/**
 * Reusable WGSL math for anisotropic 3D Gaussian projection and its backward
 * chain. The caller owns storage declarations and compute entry points.
 */a.export(r,"anisotropicProjectionWGSL",()=>i)},{"@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],jpT8C:[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"ANISO_TILE_3D",()=>s),a.export(r,"ANISO_DERIVED_STRIDE_3D",()=>n),a.export(r,"ANISO_ALPHA_THRESHOLD_3D",()=>o),a.export(r,"ANISO_MAX_ALPHA_3D",()=>d),a.export(r,"ANISO_TRANSMITTANCE_CUTOFF_3D",()=>l),a.export(r,"ANISO_REGULARIZER_UNIFORM_BYTES_3D",()=>c),a.export(r,"ANISO_CENTER_SUM_SCALE_3D",()=>u),a.export(r,"ANISO_DENSITY_STAT_SCALE_3D",()=>h),a.export(r,"resolveAnisotropicRaster3DDims",()=>v),a.export(r,"anisotropicPrepShader3D",()=>y),a.export(r,"anisotropicEmitShader3D",()=>_),a.export(r,"anisotropicForwardShader3D",()=>B),a.export(r,"anisotropicBackwardShader3D",()=>S),a.export(r,"anisotropicChainShader3D",()=>$),a.export(r,"anisotropicClearShader3D",()=>R),a.export(r,"anisotropicCenterReduceShader3D",()=>T),a.export(r,"anisotropicRegularizerShader3D",()=>P);var i=e("./projection_wgsl");let s=16,n=12,o=1/255,d=.99,l=1e-4,c=64,u=16384,h=65536;function p(e,t){if(!e)throw Error(`splat3d_aniso_raster: ${t}`)}function g(e){p(Number.isFinite(e),`non-finite WGSL literal ${e}`);let t=e.toString();return/[.eE]/.test(t)?t:`${t}.0`}let f=e=>`${e>>>0}u`,m=e=>`vec3f(${g(e[0])}, ${g(e[1])}, ${g(e[2])})`;function v(e){p(Number.isInteger(e.H)&&Number.isInteger(e.W)&&e.H>0&&e.W>0,"invalid image size"),p(e.H%s==0&&e.W%s==0,"H and W must be multiples of 16"),p(Number.isInteger(e.G)&&e.G>0,"G must be a positive integer"),p(e.cap>=256&&(e.cap&e.cap-1)==0,"cap must be a power of two >= 256"),p(4*e.cap<=16384,"cap exceeds the 16KB workgroup-storage budget");let t=e.minScale??.01,r=e.maxScale??.45;p(t>0&&r>t,"scales require 0 < minScale < maxScale");let a=e.W/s,i=e.H/s;return{H:e.H,W:e.W,G:e.G,cap:e.cap,tilesX:a,tilesY:i,numTiles:a*i,bg:e.bg??[0,0,0],near:e.near??.2,far:e.far??12,gradScale:e.gradScale??65536,projectionMode:e.projectionMode??"legacy-affine",minScale:t,maxScale:r,screenVariancePx2:Math.max(0,e.screenVariancePx2??0)}}function w(e){return{position:0,logScale:3*e.G,quaternion:6*e.G,color:10*e.G,opacity:13*e.G}}function b(e,t){return/* wgsl */`
const CAMERA = AnisoCamera(
  ${m(e.eye)},
  ${m(e.right)},
  ${m(e.cameraUp)},
  ${m(e.forward)},
  ${g(e.focalPx)},
  vec2f(${g(.5*t.W)}, ${g(.5*t.H)}),
  ${g(t.near)}
);
const SETTINGS = AnisoProjectionSettings(
  ${g(t.minScale)},
  ${g(t.maxScale)},
  ${g(t.screenVariancePx2)},
  1e-12,
  1e-12
);
`}let x=/* wgsl */"fn sigmoid1(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }";function y(e,t){let r=v(e),a=w(r);return/* wgsl */`
${x}
${(0,i.anisotropicProjectionWGSL)({mode:r.projectionMode})}
${b(t,r)}
@group(0) @binding(0) var<storage, read> params : array<f32>;
@group(0) @binding(1) var<storage, read_write> derived : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${f(r.G)}) { return; }
  let position = vec3f(
    params[${f(a.position)} + 3u * g + 0u],
    params[${f(a.position)} + 3u * g + 1u],
    params[${f(a.position)} + 3u * g + 2u]
  );
  let logScale = vec3f(
    params[${f(a.logScale)} + 3u * g + 0u],
    params[${f(a.logScale)} + 3u * g + 1u],
    params[${f(a.logScale)} + 3u * g + 2u]
  );
  let quaternion = vec4f(
    params[${f(a.quaternion)} + 4u * g + 0u],
    params[${f(a.quaternion)} + 4u * g + 1u],
    params[${f(a.quaternion)} + 4u * g + 2u],
    params[${f(a.quaternion)} + 4u * g + 3u]
  );
  let projected = anisoProject(position, logScale, quaternion, CAMERA, SETTINGS);
  let b = g * ${f(n)};
  derived[b + 0u] = projected.mean_px.x;
  derived[b + 1u] = projected.mean_px.y;
  derived[b + 2u] = projected.conic.x;
  derived[b + 3u] = projected.conic.y;
  derived[b + 4u] = projected.conic.z;
  derived[b + 5u] = projected.camera_position.z;
  derived[b + 6u] = sigmoid1(params[${f(a.color)} + 3u * g + 0u]);
  derived[b + 7u] = sigmoid1(params[${f(a.color)} + 3u * g + 1u]);
  derived[b + 8u] = sigmoid1(params[${f(a.color)} + 3u * g + 2u]);
  derived[b + 9u] = sigmoid1(params[${f(a.opacity)} + g]);
  derived[b + 10u] = projected.covariance.x;
  derived[b + 11u] = projected.covariance.z;
}
`}function _(e){let t=v(e);return/* wgsl */`
@group(0) @binding(0) var<storage, read> derived : array<f32>;
@group(0) @binding(1) var<storage, read_write> tileCounts : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binnedIds : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${f(t.G)}) { return; }
  let b = g * ${f(n)};
  let depth = derived[b + 5u];
  if (depth <= ${g(t.near)} || depth >= ${g(t.far)}) { return; }
  let opacity = derived[b + 9u];
  if (opacity <= ${g(o)}) { return; }
  let ratio = max(${g(o)} / max(opacity, 1e-8), 1e-8);
  let tau = -2.0 * log(ratio);
  if (!(tau > 0.0)) { return; }

  let radiusX = sqrt(max(tau * derived[b + 10u], 0.0));
  let radiusY = sqrt(max(tau * derived[b + 11u], 0.0));
  let sx = derived[b + 0u];
  let sy = derived[b + 1u];
  let x0 = max(0, i32(floor(sx - radiusX - 0.5)));
  let x1 = min(${t.W-1}, i32(ceil(sx + radiusX - 0.5)));
  let y0 = max(0, i32(floor(sy - radiusY - 0.5)));
  let y1 = min(${t.H-1}, i32(ceil(sy + radiusY - 0.5)));
  if (x0 > x1 || y0 > y1) { return; }
  let tx0 = x0 / ${s}; let tx1 = x1 / ${s};
  let ty0 = y0 / ${s}; let ty1 = y1 / ${s};
  for (var ty = ty0; ty <= ty1; ty = ty + 1) {
    for (var tx = tx0; tx <= tx1; tx = tx + 1) {
      let tile = u32(ty * ${t.tilesX} + tx);
      let slot = atomicAdd(&tileCounts[tile], 1u);
      if (slot < ${f(t.cap)}) { binnedIds[tile * ${f(t.cap)} + slot] = g; }
    }
  }
}
`}function B(e){let t=v(e),r=t.H*t.W;return/* wgsl */`
@group(0) @binding(0) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(1) var<storage, read_write> binnedIds : array<u32>;
@group(0) @binding(2) var<storage, read> derived : array<f32>;
@group(0) @binding(3) var<storage, read_write> image : array<f32>;
@group(0) @binding(4) var<storage, read_write> tileStop : array<u32>;
var<workgroup> shIds : array<u32, ${t.cap}>;

fn nextPow2(x : u32) -> u32 {
  var v = max(x, 1u) - 1u;
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}
fn idGreater(a : u32, b : u32) -> bool {
  if (a == 0xffffffffu) { return b != 0xffffffffu; }
  if (b == 0xffffffffu) { return false; }
  let za = derived[a * ${f(n)} + 5u];
  let zb = derived[b * ${f(n)} + 5u];
  if (za == zb) { return a > b; }
  return za > zb;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) tid : u32) {
  let tile = wg.x;
  if (tile >= ${f(t.numTiles)}) { return; }
  let count = min(tileCounts[tile], ${f(t.cap)});
  let start = tile * ${f(t.cap)};
  let sortN = nextPow2(count);
  for (var i = tid; i < sortN; i = i + 256u) {
    shIds[i] = select(0xffffffffu, binnedIds[start + i], i < count);
  }
  workgroupBarrier();
  var k = 2u;
  loop {
    if (k > sortN) { break; }
    var j = k >> 1u;
    loop {
      if (j == 0u) { break; }
      let pairs = sortN >> 1u;
      for (var pair = tid; pair < pairs; pair = pair + 256u) {
        let pos = 2u * j * (pair / j) + pair % j;
        let other = pos + j;
        let ascending = (pos & k) == 0u;
        let a = shIds[pos]; let b = shIds[other];
        let swap = select(idGreater(b, a), idGreater(a, b), ascending);
        if (swap) { shIds[pos] = b; shIds[other] = a; }
      }
      workgroupBarrier();
      j >>= 1u;
    }
    k <<= 1u;
  }
  for (var i = tid; i < count; i = i + 256u) { binnedIds[start + i] = shIds[i]; }
  workgroupBarrier();

  let tileX = tile % ${f(t.tilesX)};
  let tileY = tile / ${f(t.tilesX)};
  let x = tileX * ${s}u + tid % ${s}u;
  let y = tileY * ${s}u + tid / ${s}u;
  var localStop = 0u;
  if (x < ${f(t.W)} && y < ${f(t.H)}) {
    let px = f32(x) + 0.5; let py = f32(y) + 0.5;
    var r = 0.0; var g = 0.0; var b = 0.0; var T = 1.0;
    for (var i = 0u; i < count; i++) {
      let base = shIds[i] * ${f(n)};
      let dx = px - derived[base + 0u]; let dy = py - derived[base + 1u];
      let power = -0.5 * (derived[base + 2u] * dx * dx + 2.0 * derived[base + 3u] * dx * dy + derived[base + 4u] * dy * dy);
      localStop = i + 1u;
      if (power > 0.0) { continue; }
      let alpha = min(${g(d)}, derived[base + 9u] * exp(power));
      if (alpha < ${g(o)}) { continue; }
      let weight = T * alpha;
      r += weight * derived[base + 6u]; g += weight * derived[base + 7u]; b += weight * derived[base + 8u];
      T *= 1.0 - alpha;
      if (T < ${g(l)}) { break; }
    }
    let pixel = y * ${f(t.W)} + x;
    image[0u * ${f(r)} + pixel] = r + T * ${g(t.bg[0])};
    image[1u * ${f(r)} + pixel] = g + T * ${g(t.bg[1])};
    image[2u * ${f(r)} + pixel] = b + T * ${g(t.bg[2])};
  }
  workgroupBarrier();
  shIds[tid] = localStop;
  workgroupBarrier();
  var offset = 128u;
  loop {
    if (offset == 0u) { break; }
    if (tid < offset) { shIds[tid] = max(shIds[tid], shIds[tid + offset]); }
    workgroupBarrier();
    offset >>= 1u;
  }
  if (tid == 0u) { tileStop[tile] = shIds[0]; }
}
`}function S(e,t=!1){let r=v(e),a=r.H*r.W;return/* wgsl */`
@group(0) @binding(0) var<storage, read> gradImage : array<f32>;
@group(0) @binding(1) var<storage, read> tileCounts : array<u32>;
@group(0) @binding(2) var<storage, read> binnedIds : array<u32>;
@group(0) @binding(3) var<storage, read> tileStop : array<u32>;
@group(0) @binding(4) var<storage, read> derived : array<f32>;
@group(0) @binding(5) var<storage, read_write> accGrad : array<atomic<i32>>;
${t?"@group(0) @binding(6) var<storage, read_write> densityStats : array<atomic<u32>>;":""}
var<workgroup> shIds : array<u32, ${r.cap}>;
fn fixadd(base : u32, slot : u32, value : f32) {
  atomicAdd(&accGrad[base + slot], i32(clamp(round(value * ${g(r.gradScale)}), -2.14e9, 2.14e9)));
}
${t?/* wgsl */`
fn densityadd(g : u32, gx : f32, gy : f32) {
  let base = 3u * g;
  let sx = u32(clamp(round(abs(gx) * ${g(h)}), 0.0, 1048576.0));
  let sy = u32(clamp(round(abs(gy) * ${g(h)}), 0.0, 1048576.0));
  atomicAdd(&densityStats[base + 0u], sx);
  atomicAdd(&densityStats[base + 1u], sy);
  atomicAdd(&densityStats[base + 2u], 1u);
}
`:""}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) tid : u32) {
  let tile = wg.x;
  if (tile >= ${f(r.numTiles)}) { return; }
  let count = min(tileCounts[tile], ${f(r.cap)});
  let stop = min(count, tileStop[tile]);
  let start = tile * ${f(r.cap)};
  for (var i = tid; i < stop; i += 256u) { shIds[i] = binnedIds[start + i]; }
  workgroupBarrier();
  let tileX = tile % ${f(r.tilesX)}; let tileY = tile / ${f(r.tilesX)};
  let x = tileX * ${s}u + tid % ${s}u;
  let y = tileY * ${s}u + tid / ${s}u;
  if (x >= ${f(r.W)} || y >= ${f(r.H)}) { return; }
  let px = f32(x) + 0.5; let py = f32(y) + 0.5;
  let pixel = y * ${f(r.W)} + x;
  let go = vec3f(
    gradImage[0u * ${f(a)} + pixel],
    gradImage[1u * ${f(a)} + pixel],
    gradImage[2u * ${f(a)} + pixel]
  );
  var T = 1.0; var end = stop;
  for (var i = 0u; i < stop; i++) {
    let base = shIds[i] * ${f(n)};
    let dx = px - derived[base + 0u]; let dy = py - derived[base + 1u];
    let power = -0.5 * (derived[base + 2u] * dx * dx + 2.0 * derived[base + 3u] * dx * dy + derived[base + 4u] * dy * dy);
    if (power > 0.0) { continue; }
    let alpha = min(${g(d)}, derived[base + 9u] * exp(power));
    if (alpha < ${g(o)}) { continue; }
    T *= 1.0 - alpha;
    if (T < ${g(l)}) { end = i + 1u; break; }
  }
  var currentT = T;
  var gT = dot(go, ${m(r.bg)});
  for (var ii = i32(end) - 1; ii >= 0; ii--) {
    let base = shIds[u32(ii)] * ${f(n)};
    let dx = px - derived[base + 0u]; let dy = py - derived[base + 1u];
    let a = derived[base + 2u]; let cross = derived[base + 3u]; let c = derived[base + 4u];
    let power = -0.5 * (a * dx * dx + 2.0 * cross * dx * dy + c * dy * dy);
    if (power > 0.0) { continue; }
    let opacity = derived[base + 9u];
    let raw = opacity * exp(power);
    let alpha = min(${g(d)}, raw);
    if (alpha < ${g(o)}) { continue; }
    let previousT = currentT / max(1.0 - alpha, 1e-8);
    let color = vec3f(derived[base + 6u], derived[base + 7u], derived[base + 8u]);
    let dotgc = dot(go, color);
    let gAlpha = previousT * (dotgc - gT);
    fixadd(base, 6u, go.x * previousT * alpha);
    fixadd(base, 7u, go.y * previousT * alpha);
    fixadd(base, 8u, go.z * previousT * alpha);
    let gate = select(0.0, 1.0, raw < ${g(d)});
    let gRaw = gAlpha * gate;
    let gPower = gRaw * raw;
    let gCenterX = gPower * (a * dx + cross * dy);
    let gCenterY = gPower * (cross * dx + c * dy);
    fixadd(base, 0u, gCenterX);
    fixadd(base, 1u, gCenterY);
    ${t?"densityadd(shIds[u32(ii)], gCenterX, gCenterY);":""}
    fixadd(base, 2u, gPower * (-0.5 * dx * dx));
    fixadd(base, 3u, gPower * (-dx * dy));
    fixadd(base, 4u, gPower * (-0.5 * dy * dy));
    fixadd(base, 9u, gRaw * raw / max(opacity, 1e-8));
    gT = alpha * dotgc + (1.0 - alpha) * gT;
    currentT = previousT;
  }
}
`}function $(e,t){let r=v(e),a=w(r);return/* wgsl */`
${(0,i.anisotropicProjectionWGSL)({mode:r.projectionMode})}
${b(t,r)}
@group(0) @binding(0) var<storage, read> accGrad : array<i32>;
@group(0) @binding(1) var<storage, read> derived : array<f32>;
@group(0) @binding(2) var<storage, read> params : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradRaw : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${f(r.G)}) { return; }
  let b = g * ${f(n)};
  let inv = ${g(1/r.gradScale)};
  let position = vec3f(
    params[${f(a.position)} + 3u * g + 0u], params[${f(a.position)} + 3u * g + 1u], params[${f(a.position)} + 3u * g + 2u]
  );
  let logScale = vec3f(
    params[${f(a.logScale)} + 3u * g + 0u], params[${f(a.logScale)} + 3u * g + 1u], params[${f(a.logScale)} + 3u * g + 2u]
  );
  let quaternion = vec4f(
    params[${f(a.quaternion)} + 4u * g + 0u], params[${f(a.quaternion)} + 4u * g + 1u],
    params[${f(a.quaternion)} + 4u * g + 2u], params[${f(a.quaternion)} + 4u * g + 3u]
  );
  let upstream = AnisoProjectionUpstream(
    vec2f(f32(accGrad[b + 0u]), f32(accGrad[b + 1u])) * inv,
    vec3f(f32(accGrad[b + 2u]), f32(accGrad[b + 3u]), f32(accGrad[b + 4u])) * inv
  );
  let gradient = anisoProjectBackward(position, logScale, quaternion, CAMERA, SETTINGS, upstream);
  gradRaw[${f(a.position)} + 3u * g + 0u] += gradient.position.x;
  gradRaw[${f(a.position)} + 3u * g + 1u] += gradient.position.y;
  gradRaw[${f(a.position)} + 3u * g + 2u] += gradient.position.z;
  gradRaw[${f(a.logScale)} + 3u * g + 0u] += gradient.log_scale.x;
  gradRaw[${f(a.logScale)} + 3u * g + 1u] += gradient.log_scale.y;
  gradRaw[${f(a.logScale)} + 3u * g + 2u] += gradient.log_scale.z;
  gradRaw[${f(a.quaternion)} + 4u * g + 0u] += gradient.quaternion.x;
  gradRaw[${f(a.quaternion)} + 4u * g + 1u] += gradient.quaternion.y;
  gradRaw[${f(a.quaternion)} + 4u * g + 2u] += gradient.quaternion.z;
  gradRaw[${f(a.quaternion)} + 4u * g + 3u] += gradient.quaternion.w;
  let color = vec3f(derived[b + 6u], derived[b + 7u], derived[b + 8u]);
  let colorGradient = vec3f(f32(accGrad[b + 6u]), f32(accGrad[b + 7u]), f32(accGrad[b + 8u])) * inv;
  gradRaw[${f(a.color)} + 3u * g + 0u] += colorGradient.x * color.x * (1.0 - color.x);
  gradRaw[${f(a.color)} + 3u * g + 1u] += colorGradient.y * color.y * (1.0 - color.y);
  gradRaw[${f(a.color)} + 3u * g + 2u] += colorGradient.z * color.z * (1.0 - color.z);
  let opacity = derived[b + 9u];
  gradRaw[${f(a.opacity)} + g] += f32(accGrad[b + 9u]) * inv * opacity * (1.0 - opacity);
}
`}function R(e){return/* wgsl */`
@group(0) @binding(0) var<storage, read_write> buffer : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x < ${f(e)}) { buffer[gid.x] = 0u; }
}
`}function T(e){let t=v(e),r=w(t);return/* wgsl */`
${x}
@group(0) @binding(0) var<storage, read> params : array<f32>;
@group(0) @binding(1) var<storage, read_write> centerSum : array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${f(t.G)}) { return; }
  let base = ${f(r.position)} + 3u * g;
  let mass = sigmoid1(params[${f(r.opacity)} + g]);
  atomicAdd(&centerSum[0], i32(round(params[base + 0u] * mass * ${g(u)})));
  atomicAdd(&centerSum[1], i32(round(params[base + 1u] * mass * ${g(u)})));
  atomicAdd(&centerSum[2], i32(round(params[base + 2u] * mass * ${g(u)})));
  atomicAdd(&centerSum[3], i32(round(mass * ${g(u)})));
}
`}function P(e){let t=v(e),r=w(t);return/* wgsl */`
${x}
struct RegU {
  centerWeight      : f32,
  radiusWeight      : f32,
  targetRadius      : f32,
  opacitySparsity   : f32,
  smallRadiusWeight : f32,
  smallRadius       : f32,
  radiusBandWeight  : f32,
  minRadius         : f32,
  maxRadius         : f32,
  _pad0             : f32,
  _pad1             : f32,
  _pad2             : f32,
};
@group(0) @binding(0) var<uniform> u : RegU;
@group(0) @binding(1) var<storage, read> params : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradRaw : array<f32>;
@group(0) @binding(3) var<storage, read_write> centerSum : array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let g = gid.x;
  if (g >= ${f(t.G)}) { return; }

  let pxIdx = ${f(r.position)} + 3u * g + 0u;
  let pyIdx = ${f(r.position)} + 3u * g + 1u;
  let pzIdx = ${f(r.position)} + 3u * g + 2u;
  let p = vec3f(params[pxIdx], params[pyIdx], params[pzIdx]);
  let positionRadius = length(p);
  let invPositionRadius = 1.0 / max(positionRadius, 1e-8);
  let outside = max(0.0, positionRadius - max(u.targetRadius, 1e-8));
  let centerScale = 1.0 / max(f32(atomicLoad(&centerSum[3])), 1.0);
  let center = vec3f(
    f32(atomicLoad(&centerSum[0])),
    f32(atomicLoad(&centerSum[1])),
    f32(atomicLoad(&centerSum[2]))
  ) * centerScale;
  let gp = 2.0 * u.centerWeight * center
    + (2.0 * u.radiusWeight * outside * invPositionRadius) * p;
  gradRaw[pxIdx] += gp.x;
  gradRaw[pyIdx] += gp.y;
  gradRaw[pzIdx] += gp.z;

  let opacityIdx = ${f(r.opacity)} + g;
  let opacity = sigmoid1(params[opacityIdx]);
  gradRaw[opacityIdx] += u.opacitySparsity * opacity * (1.0 - opacity);

  let scaleBase = ${f(r.logScale)} + 3u * g;
  let meanLogScale = (
    params[scaleBase + 0u] + params[scaleBase + 1u] + params[scaleBase + 2u]
  ) / 3.0;
  let radius = exp(meanLogScale);
  let small = max(0.0, max(u.smallRadius, 1e-8) - radius);
  let smallLossGrad = u.smallRadiusWeight * small * small;
  gradRaw[opacityIdx] += 2.0 * smallLossGrad * opacity * opacity * (1.0 - opacity);

  let minRadius = max(u.minRadius, 1e-8);
  let maxRadius = max(u.maxRadius, minRadius + 1e-8);
  let under = max(0.0, minRadius - radius);
  let over = max(0.0, radius - maxRadius);
  let radiusLossDerivative = -2.0 * u.smallRadiusWeight * opacity * opacity * small
    + u.radiusBandWeight * (-2.0 * under + 2.0 * over);
  let perAxisLogScaleGradient = radiusLossDerivative * radius / 3.0;
  gradRaw[scaleBase + 0u] += perAxisLogScaleGradient;
  gradRaw[scaleBase + 1u] += perAxisLogScaleGradient;
  gradRaw[scaleBase + 2u] += perAxisLogScaleGradient;
}
`}},{"./projection_wgsl":"2WL5b","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"5Bvz3":[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"DEFAULT_ANISOTROPIC_3D_LRS",()=>l),a.export(r,"AnisotropicRaster3DEngine",()=>u);var i=e("../splat/adam_wgsl"),s=e("./layout"),n=e("./raster_wgsl");let o={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128},d=e=>Math.ceil(e/256),l={position:.025,logScale:.01,quaternion:.005,color:.08,opacity:.03};async function c(e,t,r){e.pushErrorScope("validation");let a=e.createShaderModule({code:t,label:`${r}-module`}),i=e.createComputePipeline({label:r,layout:"auto",compute:{module:a,entryPoint:"main"}}),s=await e.popErrorScope();if(s)throw console.error(`--- anisotropic WGSL failure (${r}) ---
${t}`),Error(`anisotropic raster pipeline validation (${r}): ${s.message}`);return i}class u{constructor(e,t){if(this.prepPipes=[],this.chainPipes=[],this.prepBinds=[],this.chainBinds=[],this.adamUniforms=[],this.adamBinds=[],this.destroyed=!1,this.device=e,this.dims=(0,n.resolveAnisotropicRaster3DDims)(t),this.cameras=t.cameras,0===this.cameras.length)throw Error("splat3d_aniso_raster: at least one camera is required");let r=this.dims.G*s.ANISO_PARAM_STRIDE_3D,a=this.dims.G*n.ANISO_DERIVED_STRIDE_3D;this.params=t.sharedParams??this.storage(r,o.COPY_SRC|o.COPY_DST,"aniso-params"),this.gradRaw=t.sharedGradRaw??this.storage(r,o.COPY_SRC|o.COPY_DST,"aniso-grad-raw"),this.ownsParams=!t.sharedParams,this.ownsGradRaw=!t.sharedGradRaw,this.derived=this.storage(a,0,"aniso-derived"),this.accGrad=this.storage(a,o.COPY_DST,"aniso-acc-grad"),this.mBuf=this.storage(r,o.COPY_SRC|o.COPY_DST,"aniso-adam-m"),this.vBuf=this.storage(r,o.COPY_SRC|o.COPY_DST,"aniso-adam-v"),this.tileCounts=this.storage(this.dims.numTiles,o.COPY_DST|o.COPY_SRC,"aniso-tile-counts"),this.binnedIds=this.storage(this.dims.numTiles*this.dims.cap,0,"aniso-binned-ids"),this.tileStop=this.storage(this.dims.numTiles,o.COPY_SRC,"aniso-tile-stop"),this.densityStats=this.storage(3*this.dims.G,o.COPY_SRC,"aniso-density-stats"),this.image=this.storage(3*this.dims.H*this.dims.W,o.COPY_SRC,"aniso-image"),this.gradImage=this.storage(3*this.dims.H*this.dims.W,o.COPY_DST,"aniso-grad-image")}static async create(e,t){let r=new u(e,t);return await r.build(t),r.clearDensityStats(),r}storage(e,t,r){return this.device.createBuffer({label:r,size:4*e,usage:o.STORAGE|t})}bindGroup(e,t){return this.device.createBindGroup({layout:e.getBindGroupLayout(0),entries:t.map((e,t)=>({binding:t,resource:{buffer:e}}))})}async build(e){let t=this.dims;for(let r of(this.prepPipes=await Promise.all(this.cameras.map((t,r)=>c(this.device,(0,n.anisotropicPrepShader3D)(e,t),`aniso-prep-${r}`))),this.chainPipes=await Promise.all(this.cameras.map((t,r)=>c(this.device,(0,n.anisotropicChainShader3D)(e,t),`aniso-chain-${r}`))),this.emitPipe=await c(this.device,(0,n.anisotropicEmitShader3D)(e),"aniso-emit"),this.forwardPipe=await c(this.device,(0,n.anisotropicForwardShader3D)(e),"aniso-forward"),this.backwardPipe=await c(this.device,(0,n.anisotropicBackwardShader3D)(e),"aniso-backward"),this.densityBackwardPipe=await c(this.device,(0,n.anisotropicBackwardShader3D)(e,!0),"aniso-density-backward"),this.clearBinsPipe=await c(this.device,(0,n.anisotropicClearShader3D)(t.numTiles),"aniso-clear-bins"),this.clearAccGradPipe=await c(this.device,(0,n.anisotropicClearShader3D)(t.G*n.ANISO_DERIVED_STRIDE_3D),"aniso-clear-acc-grad"),this.clearRawGradPipe=await c(this.device,(0,n.anisotropicClearShader3D)(t.G*s.ANISO_PARAM_STRIDE_3D),"aniso-clear-raw-grad"),this.clearDensityStatsPipe=await c(this.device,(0,n.anisotropicClearShader3D)(3*t.G),"aniso-clear-density-stats"),this.adamPipe=await c(this.device,(0,i.adamShader)(),"aniso-adam"),this.centerReducePipe=await c(this.device,(0,n.anisotropicCenterReduceShader3D)(e),"aniso-center-reduce"),this.regularizerPipe=await c(this.device,(0,n.anisotropicRegularizerShader3D)(e),"aniso-regularizer"),this.prepBinds=this.prepPipes.map(e=>this.bindGroup(e,[this.params,this.derived])),this.chainBinds=this.chainPipes.map(e=>this.bindGroup(e,[this.accGrad,this.derived,this.params,this.gradRaw])),this.emitBind=this.bindGroup(this.emitPipe,[this.derived,this.tileCounts,this.binnedIds]),this.forwardBind=this.bindGroup(this.forwardPipe,[this.tileCounts,this.binnedIds,this.derived,this.image,this.tileStop]),this.backwardBind=this.bindGroup(this.backwardPipe,[this.gradImage,this.tileCounts,this.binnedIds,this.tileStop,this.derived,this.accGrad]),this.densityBackwardBind=this.bindGroup(this.densityBackwardPipe,[this.gradImage,this.tileCounts,this.binnedIds,this.tileStop,this.derived,this.accGrad,this.densityStats]),this.clearBinsBind=this.bindGroup(this.clearBinsPipe,[this.tileCounts]),this.clearAccGradBind=this.bindGroup(this.clearAccGradPipe,[this.accGrad]),this.clearRawGradBind=this.bindGroup(this.clearRawGradPipe,[this.gradRaw]),this.clearDensityStatsBind=this.bindGroup(this.clearDensityStatsPipe,[this.densityStats]),this.centerSum=this.storage(4,o.COPY_DST,"aniso-center-sum"),this.centerReduceBind=this.bindGroup(this.centerReducePipe,[this.params,this.centerSum]),this.regularizerUniform=this.device.createBuffer({label:"aniso-regularizer-uniform",size:n.ANISO_REGULARIZER_UNIFORM_BYTES_3D,usage:o.UNIFORM|o.COPY_DST}),this.regularizerBind=this.bindGroup(this.regularizerPipe,[this.regularizerUniform,this.params,this.gradRaw,this.centerSum]),(0,s.anisotropicParamSegments3D)(t.G))){let e=this.device.createBuffer({label:"aniso-adam-uniform",size:i.ADAM_UNIFORM_BYTES,usage:64|o.COPY_DST});this.adamUniforms.push(e),this.adamBinds.push(this.device.createBindGroup({layout:this.adamPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.params}},{binding:2,resource:{buffer:this.gradRaw}},{binding:3,resource:{buffer:this.mBuf}},{binding:4,resource:{buffer:this.vBuf}}]}))}}setParams(e){if(e.length!==this.dims.G*s.ANISO_PARAM_STRIDE_3D)throw Error(`splat3d_aniso_raster: expected ${this.dims.G*s.ANISO_PARAM_STRIDE_3D} params, got ${e.length}`);this.device.queue.writeBuffer(this.params,0,e)}zeroAdamState(){let e=new Float32Array(this.dims.G*s.ANISO_PARAM_STRIDE_3D);this.device.queue.writeBuffer(this.mBuf,0,e),this.device.queue.writeBuffer(this.vBuf,0,e)}resetAdamForSplats(e){let t=this.dims.G,r=(0,s.anisotropicParamSegments3D)(t);for(let a=0;a<e.length;a++){let i=0|e[a];if(!(i<0)&&!(i>=t))for(let e of r){let t=new Float32Array(e.components),r=e.offset+e.components*i;this.device.queue.writeBuffer(this.mBuf,4*r,t),this.device.queue.writeBuffer(this.vBuf,4*r,t)}}}recordForward(e,t=0){let r=this.viewIndex(t),a=e.beginComputePass();a.setPipeline(this.prepPipes[r]),a.setBindGroup(0,this.prepBinds[r]),a.dispatchWorkgroups(d(this.dims.G)),a.setPipeline(this.clearBinsPipe),a.setBindGroup(0,this.clearBinsBind),a.dispatchWorkgroups(d(this.dims.numTiles)),a.setPipeline(this.emitPipe),a.setBindGroup(0,this.emitBind),a.dispatchWorkgroups(d(this.dims.G)),a.setPipeline(this.forwardPipe),a.setBindGroup(0,this.forwardBind),a.dispatchWorkgroups(this.dims.numTiles),a.end()}recordClearRawGrad(e){let t=e.beginComputePass();t.setPipeline(this.clearRawGradPipe),t.setBindGroup(0,this.clearRawGradBind),t.dispatchWorkgroups(d(this.dims.G*s.ANISO_PARAM_STRIDE_3D)),t.end()}recordBackwardAdd(e,t=0,r=!1){let a=this.viewIndex(t),i=e.beginComputePass();i.setPipeline(this.clearAccGradPipe),i.setBindGroup(0,this.clearAccGradBind),i.dispatchWorkgroups(d(this.dims.G*n.ANISO_DERIVED_STRIDE_3D)),i.setPipeline(r?this.densityBackwardPipe:this.backwardPipe),i.setBindGroup(0,r?this.densityBackwardBind:this.backwardBind),i.dispatchWorkgroups(this.dims.numTiles),i.setPipeline(this.chainPipes[a]),i.setBindGroup(0,this.chainBinds[a]),i.dispatchWorkgroups(d(this.dims.G)),i.end()}clearDensityStats(){let e=this.device.createCommandEncoder(),t=e.beginComputePass();t.setPipeline(this.clearDensityStatsPipe),t.setBindGroup(0,this.clearDensityStatsBind),t.dispatchWorkgroups(d(3*this.dims.G)),t.end(),this.device.queue.submit([e.finish()])}recordRegularizerAdd(e,t){if(!(0!==t.centerWeight||0!==t.radiusWeight||0!==t.opacitySparsity||0!==t.smallRadiusWeight||0!==t.radiusBandWeight))return;let r=new Float32Array(16);r[0]=t.centerWeight,r[1]=t.radiusWeight,r[2]=t.targetRadius,r[3]=t.opacitySparsity,r[4]=t.smallRadiusWeight,r[5]=t.smallRadius,r[6]=t.radiusBandWeight,r[7]=t.minRadius,r[8]=t.maxRadius,this.device.queue.writeBuffer(this.regularizerUniform,0,r),0!==t.centerWeight&&e.clearBuffer(this.centerSum,0,16);let a=e.beginComputePass();0!==t.centerWeight&&(a.setPipeline(this.centerReducePipe),a.setBindGroup(0,this.centerReduceBind),a.dispatchWorkgroups(d(this.dims.G))),a.setPipeline(this.regularizerPipe),a.setBindGroup(0,this.regularizerBind),a.dispatchWorkgroups(d(this.dims.G)),a.end()}recordAdam(e,t,r=l,a=i.DEFAULT_HYPER){let n=(0,s.anisotropicParamSegments3D)(this.dims.G),o={position:r.position,logScale:r.logScale,quaternion:r.quaternion,color:r.color,opacity:r.opacity},c=Math.max(1,t),u=1-Math.pow(a.beta1,c),h=1-Math.pow(a.beta2,c);n.forEach((e,t)=>{let r=new ArrayBuffer(i.ADAM_UNIFORM_BYTES),s=new Uint32Array(r),n=new Float32Array(r);s[0]=e.offset,s[1]=e.length,n[2]=o[e.name],n[3]=a.beta1,n[4]=a.beta2,n[5]=a.eps,n[6]=u,n[7]=h,this.device.queue.writeBuffer(this.adamUniforms[t],0,r)});let p=e.beginComputePass();p.setPipeline(this.adamPipe),n.forEach((e,t)=>{p.setBindGroup(0,this.adamBinds[t]),p.dispatchWorkgroups(d(e.length))}),p.end()}runForward(e=0){let t=this.device.createCommandEncoder();this.recordForward(t,e),this.device.queue.submit([t.finish()])}readImage(){return this.readFloats(this.image,3*this.dims.H*this.dims.W)}readParams(){return this.readFloats(this.params,this.dims.G*s.ANISO_PARAM_STRIDE_3D)}readRawGrad(){return this.readFloats(this.gradRaw,this.dims.G*s.ANISO_PARAM_STRIDE_3D)}async readDensityStats(){let e=await this.readU32(this.densityStats,3*this.dims.G),t=new Float32Array(this.dims.G),r=new Uint32Array(this.dims.G);for(let a=0;a<this.dims.G;a++){let i=3*a;t[a]=Math.hypot(e[i],e[i+1])/n.ANISO_DENSITY_STAT_SCALE_3D,r[a]=e[i+2]}return{absScreenGradient:t,visiblePixels:r}}async readTileTelemetry(){let e=this.dims.numTiles,t=4*e,r=this.device.createBuffer({size:2*t,usage:o.MAP_READ|o.COPY_DST}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(this.tileCounts,0,r,0,t),a.copyBufferToBuffer(this.tileStop,0,r,t,t),this.device.queue.submit([a.finish()]),await r.mapAsync(1);let i=r.getMappedRange(),s=new Uint32Array(i.slice(0,t)),n=new Uint32Array(i.slice(t,2*t));r.unmap(),r.destroy();let d=0,l=0,c=0,u=0,h=0;for(let t=0;t<e;t++)d=Math.max(d,s[t]),l=Math.max(l,n[t]),h+=s[t],s[t]>this.dims.cap&&(c++,u+=s[t]-this.dims.cap);return{cap:this.dims.cap,maxCount:d,maxStop:l,overflowTiles:c,overflowPairs:u,totalPairs:h}}destroy(){if(this.destroyed)return;this.destroyed=!0;let e=[this.derived,this.accGrad,this.mBuf,this.vBuf,this.tileCounts,this.binnedIds,this.tileStop,this.densityStats,this.image,this.gradImage,this.centerSum,this.regularizerUniform,...this.adamUniforms];for(let t of(this.ownsParams&&e.push(this.params),this.ownsGradRaw&&e.push(this.gradRaw),e))t.destroy()}viewIndex(e){return Math.max(0,Math.min(this.cameras.length-1,0|e))}async readFloats(e,t){let r=this.device.createBuffer({size:4*t,usage:o.MAP_READ|o.COPY_DST}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(1);let i=new Float32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}async readU32(e,t){let r=this.device.createBuffer({size:4*t,usage:o.MAP_READ|o.COPY_DST}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(e,0,r,0,4*t),this.device.queue.submit([a.finish()]),await r.mapAsync(1);let i=new Uint32Array(r.getMappedRange().slice(0));return r.unmap(),r.destroy(),i}}},{"../splat/adam_wgsl":"kfWkJ","./layout":"d5AMr","./raster_wgsl":"jpT8C","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],"1AKMX":[function(e,t,r){/// <reference types="@webgpu/types" />
var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),a.export(r,"Splat3DAnisotropicOptimizer",()=>u),a.export(r,"isotropicInitAsAnisotropic",()=>h);var i=e("../clip/vision"),s=e("../clip/vision_batch"),n=e("../splat3d/cameras"),o=e("../splat3d/optimize"),d=e("./adaptive"),l=e("./layout"),c=e("./raster_engine");class u{static async create(e,t,r,a={}){let[d,l,p]=t.inputShape;if(3!==d||256!==l||256!==p)throw Error("splat3d aniso: CLIP input shape must be [3,256,256]");let g=a.G??o.LEGIBLE_3D_G,f=(a.cameras??(0,n.DEFAULT_3D_CAMERAS)).map(e=>(0,n.prepareCamera)(e,256)),m=await (0,c.AnisotropicRaster3DEngine).create(e,{H:256,W:256,G:g,cap:a.cap??function(e){let t=256;for(;t<e&&t<4096;)t*=2;return t}(g),cameras:f,bg:[0,0,0]});m.setParams(h(g,a.seed??1,{anisotropy:a.initialAnisotropy??.45,randomRotation:a.randomInitialRotation??!0})),m.zeroAdamState();let v=await (0,i.VisionTrainer).create(e,t,r,{stemSpatialBwd:!0,fusePointwiseGeluForward:!0}),w=function(e){if(void 0===e||!Number.isFinite(e))return 3;let t=0|e;return t>1?Math.min(9,t):1}(a.clipBatchSize),b=w>1?await (0,s.BatchMajorVisionTrainer).create(e,t,r,w,{stemSpatialBwd:!0,fusePointwiseGeluForward:!0}):null;return new u(e,m,v,b,f,a)}constructor(e,t,r,a,i,s){var n;this.side=256,this.clipLayout="per_view",this.step_=0,this.hasPrompts=!1,this.viewCursor=0,this.lastAdaptationStep=-1,this.hasDensityStats=!1,this.adaptationDiagnostics_=null,this.device=e,this.raster=t,this.trainer=r,this.batchTrainer=a,this.clipBatchSize=a?.batch??1,this.cameras=i,this.viewSampler=s.viewSampler??"epoch",this.rng=((s.seed??1)^2654435769)>>>0||1,this.lrs={...c.DEFAULT_ANISOTROPIC_3D_LRS,...s.lrs},this.convergence=(n=s.convergence,{centerWeight:p(n?.centerWeight,0),radiusWeight:p(n?.radiusWeight,0),targetRadius:g(n?.targetRadius,1.15),opacitySparsity:p(n?.opacitySparsity,0),smallRadiusWeight:p(n?.smallRadiusWeight,0),smallRadius:g(n?.smallRadius,.024),radiusBandWeight:p(n?.radiusBandWeight,0),minRadius:g(n?.minRadius,.016),maxRadius:g(n?.maxRadius,.16),stagedOptimization:n?.stagedOptimization===!0,geometryWarmupSteps:g(n?.geometryWarmupSteps,250),geometryDecaySteps:g(n?.geometryDecaySteps,1e3),geometryFinalScale:p(n?.geometryFinalScale,.2),appearanceWarmupScale:f(n?.appearanceWarmupScale,.35),adaptiveRelocation:n?.adaptiveRelocation===!0,adaptationInterval:g(n?.adaptationInterval,200),adaptationFraction:f(n?.adaptationFraction,.01)}),this.textBuffers=i.map((t,a)=>e.createBuffer({label:`splat3d-aniso-text-${a}`,size:4*r.plan.textDim,usage:12}))}setViewPrompts(e){if(e.length!==this.cameras.length)throw Error(`splat3d aniso: expected ${this.cameras.length} view prompts, got ${e.length}`);for(let t=0;t<e.length;t++){if(e[t].length!==this.trainer.plan.textDim)throw Error(`splat3d aniso: prompt ${t} has wrong embedding size`);this.device.queue.writeBuffer(this.textBuffers[t],0,e[t])}this.hasPrompts=!0}setGridPrompt(e){}setRandomGridPrompt(e){}setZoomPrompt(e){}step(e=0,t=this.cameras.length){if(!this.hasPrompts)throw Error("splat3d aniso: setViewPrompts before step");let r=this.sampleViews(t),a=this.shouldCaptureDensityStats(),i=this.device.createCommandEncoder();this.raster.recordClearRawGrad(i),this.recordTrainingViews(i,r,a),this.recordConvergenceRegularizer(i),this.step_++,this.raster.recordAdam(i,this.step_,this.lrsForStep()),this.raster.recordForward(i,e),this.device.queue.submit([i.finish()]),this.hasDensityStats||=a}async profileStep(e=0,t=this.cameras.length){if(!this.hasPrompts)throw Error("splat3d aniso: setViewPrompts before profileStep");await this.device.queue.onSubmittedWorkDone();let r=this.sampleViews(t),a=this.batchTrainer;if(!a||r.length!==a.batch){let a=performance.now();this.step(e,t),await this.device.queue.onSubmittedWorkDone();let i=performance.now()-a;return this.emptyTimings(r.length,i)}let i=performance.now(),s=this.emptyTimings(r.length,0),n=this.shouldCaptureDensityStats();s.clear=await this.submitTimed(e=>this.raster.recordClearRawGrad(e)),s.rasterFwd=await this.submitTimed(e=>this.recordBatchInputs(e,r)),s.clipBatch=await this.submitTimed(e=>a.encode(e,{backward:!0}));for(let e=0;e<r.length;e++){let t=r[e];s.rasterReplay+=await this.submitTimed(r=>{r.copyBufferToBuffer(a.inputGradBuffer,a.inputGradOffsetBytes(e),this.raster.gradImage,0,786432),this.raster.recordForward(r,t)}),s.rasterBwd+=await this.submitTimed(e=>this.raster.recordBackwardAdd(e,t,n))}s.regularizer=await this.submitTimed(e=>this.recordConvergenceRegularizer(e)),this.step_++,this.hasDensityStats||=n,s.adam=await this.submitTimed(e=>this.raster.recordAdam(e,this.step_,this.lrsForStep())),s.display=await this.submitTimed(t=>this.raster.recordForward(t,e)),await this.adaptSplatsIfDue();let o=performance.now()-i;return s.total=o,s}emptyTimings(e,t){return{views:e,totalViews:this.cameras.length,clipMode:this.batchTrainer?"batch":"single",clipBatchSize:this.clipBatchSize,timing:"split-submit-wall",total:t,clear:0,rasterFwd:0,rasterReplay:0,clipFwd:0,clipBwd:0,clipBatch:0,rasterBwd:0,regularizer:0,adam:0,display:0}}get stepCount(){return this.step_}get adaptationDiagnostics(){return this.adaptationDiagnostics_}async adaptSplatsIfDue(e=!1){if(!this.convergence.adaptiveRelocation)return null;let t=Math.max(1,Math.round(this.convergence.adaptationInterval));if(!e&&this.step_<t||!e&&this.lastAdaptationStep>=0&&this.step_-this.lastAdaptationStep<t)return null;if(!e&&this.lastAdaptationStep===this.step_)return this.adaptationDiagnostics_;await this.device.queue.onSubmittedWorkDone();let r=this.hasDensityStats?await this.raster.readDensityStats():null,[a,i]=await Promise.all([this.raster.readParams(),this.raster.readRawGrad()]),s=null===r?null:function(e){let t=e.absScreenGradient.slice(),r=new Float32Array(t.length);for(let t=0;t<r.length;t++){let a=e.visiblePixels[t];// The AbsGS statistic already sums pixel-level magnitudes, so it already
// carries Pixel-GS's coverage weighting. This confidence term only rejects
// one-pixel noise and gently discounts weakly visible candidates.
r[t]=a<4?0:Math.min(1,Math.sqrt(a/64))}return{selectionNeed:t,coverage:r}}(r),n=(0,d.planFixedBudgetAnisotropicSplatAdaptation)(a,i,{maxRelocations:Math.max(1,Math.floor(this.raster.dims.G*this.convergence.adaptationFraction)),seed:(this.rng^this.step_)>>>0,deadOpacityThreshold:.04,minParentOpacity:.12,minRadius:this.convergence.minRadius,maxRadius:this.convergence.maxRadius,splitOffsetScale:.55,selectionNeed:s?.selectionNeed,coverage:s?.coverage});return null!==r&&(n.diagnostics.densityStatsSampled=!0,n.diagnostics.densityVisiblePixels=r.visiblePixels.reduce((e,t)=>e+t,0),n.diagnostics.densityMaxScreenGradient=Math.max(...r.absScreenGradient)),n.changedIndices.length>0&&(this.raster.setParams(n.params),this.raster.resetAdamForSplats(n.changedIndices)),this.lastAdaptationStep=this.step_,null!==r&&(this.raster.clearDensityStats(),this.hasDensityStats=!1),this.adaptationDiagnostics_=n.diagnostics,n.diagnostics}prepareDisplayFrame(){}renderViewToImage(e=0){this.raster.runForward(e)}async renderView(e=0){return this.raster.runForward(e),this.raster.readImage()}async currentEmbedding(e=0){let t=this.device.createCommandEncoder();return this.raster.recordForward(t,e),t.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(t,{backward:!1}),this.device.queue.submit([t.finish()]),m(this.device,this.trainer.outputBuffer,this.trainer.plan.embedDim)}destroy(){for(let e of(this.raster.destroy(),this.trainer.destroy(),this.batchTrainer?.destroy(),this.textBuffers))e.destroy()}shouldCaptureDensityStats(){if(!this.convergence.adaptiveRelocation)return!1;let e=Math.max(1,Math.round(this.convergence.adaptationInterval));return this.step_+1>=e&&(this.step_+1)%e==0}recordTrainingView(e,t,r=!1){let a=Math.max(0,Math.min(this.cameras.length-1,0|t));e.copyBufferToBuffer(this.textBuffers[a],0,this.trainer.textBuffer,0,4*this.trainer.plan.textDim),this.raster.recordForward(e,a),e.copyBufferToBuffer(this.raster.image,0,this.trainer.inputBuffer,0,786432),this.trainer.encode(e,{backward:!0}),e.copyBufferToBuffer(this.trainer.inputGradBuffer,0,this.raster.gradImage,0,786432),this.raster.recordBackwardAdd(e,a,r)}recordTrainingViews(e,t,r=!1){let a=this.batchTrainer;if(!a||t.length<a.batch){for(let a of t)this.recordTrainingView(e,a,r);return}for(let i=0;i<t.length;i+=a.batch){let s=t.slice(i,i+a.batch);if(s.length<a.batch){for(let t of s)this.recordTrainingView(e,t,r);continue}this.recordBatchTrainingViews(e,s,r)}}recordBatchTrainingViews(e,t,r=!1){let a=this.batchTrainer;this.recordBatchInputs(e,t),a.encode(e,{backward:!0}),this.recordBatchBackward(e,t,r)}recordBatchInputs(e,t){let r=this.batchTrainer,a=4*r.plan.textDim;for(let i=0;i<t.length;i++){let s=Math.max(0,Math.min(this.cameras.length-1,0|t[i]));e.copyBufferToBuffer(this.textBuffers[s],0,r.textBuffer,r.textOffsetBytes(i),a),this.raster.recordForward(e,s),e.copyBufferToBuffer(this.raster.image,0,r.inputBuffer,r.slotOffsetBytes(i,r.plan.inputSlot),786432)}}recordBatchBackward(e,t,r=!1){let a=this.batchTrainer;for(let i=0;i<t.length;i++){let s=Math.max(0,Math.min(this.cameras.length-1,0|t[i]));e.copyBufferToBuffer(a.inputGradBuffer,a.inputGradOffsetBytes(i),this.raster.gradImage,0,786432),// Shared tile/conic state belongs to the last forward, so replay this lane.
this.raster.recordForward(e,s),this.raster.recordBackwardAdd(e,s,r)}}async submitTimed(e){let t=this.device.createCommandEncoder();e(t);let r=performance.now();return this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone(),performance.now()-r}lrsForStep(){if(!this.convergence.stagedOptimization)return this.lrs;let e=Math.max(1,this.convergence.geometryWarmupSteps),t=Math.max(1,this.convergence.geometryDecaySteps),r=Math.max(0,Math.min(1,this.step_/e)),a=Math.max(0,Math.min(1,(this.step_-e)/t)),i=1+(this.convergence.geometryFinalScale-1)*a,s=this.convergence.appearanceWarmupScale+(1-this.convergence.appearanceWarmupScale)*r;return{position:this.lrs.position*i,logScale:this.lrs.logScale*i,quaternion:this.lrs.quaternion*i,color:this.lrs.color*s,opacity:this.lrs.opacity*s}}recordConvergenceRegularizer(e){this.convergenceRegularizerEnabled()&&this.raster.recordRegularizerAdd(e,this.regularizerOptions())}convergenceRegularizerEnabled(){return 0!==this.convergence.centerWeight||0!==this.convergence.radiusWeight||0!==this.convergence.opacitySparsity||0!==this.convergence.smallRadiusWeight||0!==this.convergence.radiusBandWeight}regularizerOptions(){return{centerWeight:this.convergence.centerWeight,radiusWeight:this.convergence.radiusWeight,targetRadius:this.convergence.targetRadius,opacitySparsity:this.convergence.opacitySparsity,smallRadiusWeight:this.convergence.smallRadiusWeight,smallRadius:this.convergence.smallRadius,radiusBandWeight:this.convergence.radiusBandWeight,minRadius:this.convergence.minRadius,maxRadius:this.convergence.maxRadius}}sampleViews(e){let t=Math.max(1,Math.min(this.cameras.length,0|e));if(t===this.cameras.length)return Array.from({length:t},(e,t)=>t);if("epoch"===this.viewSampler){let e=Array.from({length:t},(e,t)=>(this.viewCursor+t)%this.cameras.length);return this.viewCursor=(this.viewCursor+t)%this.cameras.length,e}return(0,n.sampleWeightedCameraIndices)(this.cameras,t,()=>(this.rng^=this.rng<<13,this.rng^=this.rng>>>17,this.rng^=this.rng<<5,(this.rng>>>0)/4294967296))}}function h(e,t,r={}){let a;let i=(0,o.randomSplats3D)(e,t),s=new Float32Array(e*l.ANISO_PARAM_STRIDE_3D),n=Math.max(0,r.anisotropy??0),d=r.randomRotation??n>0,c=(a=(2769414579^t)>>>0||1,()=>{let e=a=a+1831565813>>>0;return e=Math.imul(e^e>>>15,1|e),(((e^=e+Math.imul(e^e>>>7,61|e))^e>>>14)>>>0)/4294967296});s.set(i.subarray(0,3*e),0);for(let t=0;t<e;t++){let r=i[3*e+t],a=n*(.65+.7*c()),o=[-a,0,a];for(let e=o.length-1;e>0;e--){let t=Math.floor(c()*(e+1));[o[e],o[t]]=[o[t],o[e]]}if(s[3*e+3*t+0]=r+o[0],s[3*e+3*t+1]=r+o[1],s[3*e+3*t+2]=r+o[2],d){// Shoemake's uniform unit-quaternion construction, stored as [x, y, z, w].
let r=c(),a=2*Math.PI*c(),i=2*Math.PI*c(),n=Math.sqrt(1-r),o=Math.sqrt(r);s[6*e+4*t+0]=n*Math.sin(a),s[6*e+4*t+1]=n*Math.cos(a),s[6*e+4*t+2]=o*Math.sin(i),s[6*e+4*t+3]=o*Math.cos(i)}else s[6*e+4*t+3]=1}return s.set(i.subarray(4*e,7*e),10*e),s.set(i.subarray(7*e,8*e),13*e),s}function p(e,t){return void 0!==e&&Number.isFinite(e)?Math.max(0,e):t}function g(e,t){return void 0!==e&&Number.isFinite(e)?Math.max(1e-4,e):t}function f(e,t){return void 0!==e&&Number.isFinite(e)?Math.max(0,Math.min(1,e)):t}async function m(e,t,r){let a=e.createBuffer({size:4*r,usage:9}),i=e.createCommandEncoder();i.copyBufferToBuffer(t,0,a,0,4*r),e.queue.submit([i.finish()]),await a.mapAsync(1);let s=new Float32Array(a.getMappedRange().slice(0));return a.unmap(),a.destroy(),s}},{"../clip/vision":"lNzsi","../clip/vision_batch":"bWgJv","../splat3d/cameras":"3Dl1Z","../splat3d/optimize":"dTqrt","./adaptive":"aSGHB","./layout":"d5AMr","./raster_engine":"5Bvz3","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}],aSGHB:[function(e,t,r){var a=e("@parcel/transformer-js/src/esmodule-helpers.js");a.defineInteropFlag(r),/**
 * Plans a CPU-side, fixed-count adaptation of anisotropic 3D splats.
 *
 * Dead, low-opacity destinations are replaced with children of the highest
 * coverage-weighted position-gradient parents. A split clones color and
 * orientation, and applies one log-scale delta to every axis so anisotropy is
 * exactly preserved. The input arrays are never mutated.
 */a.export(r,"planFixedBudgetAnisotropicSplatAdaptation",()=>o);var i=e("../splat3d/raster_wgsl"),s=e("./layout");let n=2**(-1/3);function o(e,t,r={}){!function(e,t){if(0===e.length||e.length%s.ANISO_PARAM_STRIDE_3D!=0)throw Error(`splat3d aniso adaptive: params length ${e.length} must be a positive multiple of ${s.ANISO_PARAM_STRIDE_3D}`);if(t.length!==e.length)throw Error(`splat3d aniso adaptive: raw gradient length ${t.length} != params length ${e.length}`);for(let t=0;t<e.length;t++)if(!Number.isFinite(e[t]))throw Error(`splat3d aniso adaptive: non-finite parameter at offset ${t}`)}(e,t);let a=e.length/s.ANISO_PARAM_STRIDE_3D,o=function(e,t){let r=e.maxRelocations??Math.max(1,Math.floor(.01*t)),a=e.deadOpacityThreshold??.05,s=e.minParentOpacity??.1,o=e.minParentNeed??0,d=e.minRadius??i.RADIUS_MIN,l=e.maxRadius??i.RADIUS_MAX,c=e.minOpacity??1e-4,u=e.maxOpacity??i.MAX_ALPHA,h=e.splitRadiusScale??n,p=e.splitOffsetScale??.5;if(y(r,0,Number.MAX_SAFE_INTEGER,"maxRelocations"),y(a,0,1,"deadOpacityThreshold"),y(s,0,1,"minParentOpacity"),y(o,0,Number.MAX_VALUE,"minParentNeed"),y(d,Number.MIN_VALUE,Number.MAX_VALUE,"minRadius"),y(l,Number.MIN_VALUE,Number.MAX_VALUE,"maxRadius"),l<=d)throw Error("splat3d aniso adaptive: maxRadius must exceed minRadius");if(!(c>0&&c<1))throw Error("splat3d aniso adaptive: minOpacity must be in (0, 1)");if(!(u>c&&u<1))throw Error("splat3d aniso adaptive: maxOpacity must be in (minOpacity, 1)");if(!(Number.isFinite(h)&&h>0&&h<=1))throw Error("splat3d aniso adaptive: splitRadiusScale must be in (0, 1]");return y(p,0,100,"splitOffsetScale"),{maxRelocations:Math.min(t,Math.floor(r)),selectionNeed:e.selectionNeed,coverage:e.coverage,deadOpacityThreshold:a,minParentOpacity:s,minParentNeed:o,minScale:d,maxScale:l,minOpacity:c,maxOpacity:u,splitScale:h,splitOffsetScale:p,seed:(e.seed??1)>>>0}}(r,a);if(void 0!==o.coverage&&o.coverage.length!==a)throw Error(`splat3d aniso adaptive: coverage length ${o.coverage.length} != splat count ${a}`);if(void 0!==o.selectionNeed&&o.selectionNeed.length!==a)throw Error(`splat3d aniso adaptive: selectionNeed length ${o.selectionNeed.length} != splat count ${a}`);let B=e.slice(),S=new Set,$=3*a,R=6*a,T=10*a,P=13*a,C=0,I=0;for(let e=0;e<a;e++){let t=b(B,$+3*e),r=u(t,o.minScale,o.maxScale);0!==r&&(x(B,$+3*e,t.map(e=>Math.fround(e+r))),S.add(e),C++);let a=P+e,i=m(B[a]);(i<o.minOpacity||i>o.maxOpacity)&&(B[a]=v(i,o.minOpacity,o.maxOpacity),S.add(e),I++)}let G=c(B,a),A=[];for(let e=0;e<a;e++){let r=g(t[3*e+0]),a=g(t[3*e+1]),i=g(t[3*e+2]),s=void 0===o.selectionNeed?Math.hypot(Math.abs(r),Math.abs(a),Math.abs(i)):f(o.selectionNeed[e]),n=void 0===o.coverage?1:f(o.coverage[e]),d=function(e,t){if(0===e||0===t)return 0;let r=e*t;return Number.isFinite(r)?r:Number.MAX_VALUE}(s,n);A.push({index:e,opacity:m(B[P+e]),gradientMagnitude:s,coverageWeight:n,need:d})}let k=A.filter(e=>e.opacity<=o.deadOpacityThreshold).sort(h),E=new Set(k.map(e=>e.index)),D=A.filter(e=>!E.has(e.index)&&e.opacity>=o.minParentOpacity&&e.need>o.minParentNeed).sort(p),M=Math.min(o.maxRelocations,k.length,D.length),O=[];for(let e=0;e<M;e++){var z;let t=k[e],r=D[e],a=r.index,i=t.index,s=b(B,3*a),n=b(B,$+3*a),c=b(B,$+3*i),h=[B[z=R+4*a],B[z+1],B[z+2],B[z+3]],p=m(B[P+a]),g=m(B[P+i]),f=d(n),y=d(c),C=l(p,f)+l(g,y),I=n.map(e=>e+Math.log(o.splitScale)),G=u(I,o.minScale,o.maxScale),A=I.map(e=>Math.fround(e+G)),E=d(A),M=C/(2*E),W=_(1-Math.exp(-M),o.minOpacity,o.maxOpacity),F=v(W,o.minOpacity,o.maxOpacity),L=Math.sqrt(E),N=function(e,t,r,a){let i=a^Math.imul(e+1,2654435761)^Math.imul(t+1,2246822519)^Math.imul(r+1,3266489917),s=(w(i)+.5)/4294967296,n=(w(668265263^i)+.5)/4294967296,o=2*s-1,d=Math.sqrt(Math.max(0,1-o*o)),l=2*Math.PI*n;return[d*Math.cos(l),d*Math.sin(l),o]}(a,i,e,o.seed),j=L*o.splitOffsetScale,U=[s[0]-N[0]*j,s[1]-N[1]*j,s[2]-N[2]*j],q=[s[0]+N[0]*j,s[1]+N[1]*j,s[2]+N[2]*j];x(B,3*a,U),x(B,3*i,q),x(B,$+3*a,A),x(B,$+3*i,A),x(B,R+4*i,h);for(let e=0;e<3;e++)B[T+3*i+e]=B[T+3*a+e];B[P+a]=F,B[P+i]=F,S.add(a),S.add(i);let Y=m(F),V=2*l(Y,d(A));O.push({parentIndex:a,destinationIndex:i,parentNeed:r.need,parentGradientMagnitude:r.gradientMagnitude,parentCoverageWeight:r.coverageWeight,parentPositionBefore:s,parentPositionAfter:b(B,3*a),childPosition:b(B,3*i),parentLogScaleBefore:n,destinationLogScaleBefore:c,logScaleAfter:b(B,$+3*a),parentQuaternion:h,parentOpacityBefore:p,destinationOpacityBefore:g,opacityAfter:Y,coverageMassBefore:C,coverageMassAfter:V})}!function(e){for(let t=0;t<e.length;t++)if(!Number.isFinite(e[t]))throw Error(`splat3d aniso adaptive: adaptation produced non-finite parameter at offset ${t}`)}(B);let W=c(B,a);return{params:B,changedIndices:Uint32Array.from(Array.from(S).sort((e,t)=>e-t)),relocations:O,diagnostics:{splatCount:a,requestedRelocations:o.maxRelocations,eligibleDestinations:k.length,eligibleParents:D.length,relocationCount:M,radiusClampCount:C,opacityClampCount:I,maxNeed:D.length>0?D[0].need:0,minSelectedNeed:M>0?D[M-1].need:0,coverageMassBefore:G,coverageMassAfter:W,coverageMassRelativeError:Math.abs(W-G)/Math.max(G,1e-12)}}}function d(e){return Math.exp(2/3*(e[0]+e[1]+e[2]))}function l(e,t){return-Math.log1p(-e)*t}function c(e,t){let r=3*t,a=13*t,i=0;for(let s=0;s<t;s++)i+=l(m(e[a+s]),d(b(e,r+3*s)));return i}/** Clamps geometric-mean scale with one shared shift, preserving axis ratios. */function u(e,t,r){let a=(e[0]+e[1]+e[2])/3;return Math.max(Math.log(t),Math.min(Math.log(r),a))-a}function h(e,t){return e.opacity-t.opacity||e.need-t.need||e.index-t.index}function p(e,t){return t.need-e.need||t.gradientMagnitude-e.gradientMagnitude||t.opacity-e.opacity||e.index-t.index}function g(e){return Number.isFinite(e)?e:0}function f(e){return Number.isFinite(e)&&e>0?e:0}function m(e){if(e>=0)return 1/(1+Math.exp(-e));let t=Math.exp(e);return t/(1+t)}function v(e,t,r){var a;let i=Math.min((r-t)*.25,1e-7);return Math.fround(Math.log(a=_(e,t+i,r-i))-Math.log1p(-a))}function w(e){let t=e>>>0;return((t=Math.imul((t=Math.imul(t^t>>>16,2146121005))^t>>>15,2221713035))^t>>>16)>>>0}function b(e,t){return[e[t],e[t+1],e[t+2]]}function x(e,t,r){for(let a=0;a<r.length;a++)e[t+a]=r[a]}function y(e,t,r,a){if(!(Number.isFinite(e)&&e>=t&&e<=r))throw Error(`splat3d aniso adaptive: ${a} must be finite and in [${t}, ${r}]`)}function _(e,t,r){return Math.min(r,Math.max(t,e))}},{"../splat3d/raster_wgsl":"fuyeU","./layout":"d5AMr","@parcel/transformer-js/src/esmodule-helpers.js":"k3151"}]},["e5WXe"],"e5WXe","parcelRequire924a")//# sourceMappingURL=splat3d.4303df36.js.map
;
//# sourceMappingURL=splat3d.4303df36.js.map
