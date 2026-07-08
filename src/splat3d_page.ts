/// <reference types="@webgpu/types" />
import {
  buildBasePrompt,
  buildCoarseViewPrompt,
  buildGrid9Prompt,
  buildViewPrompt,
  camerasForFraming,
  type BackgroundPromptMode,
  type CameraFramingMode,
  type Grid9PromptMode,
  type ViewPromptMode,
} from "./splat3d/cameras";
import {
  Splat3DOptimizer,
  cosine,
  type Splat3DBackgroundMode,
  type Splat3DClipLayout,
  type Splat3DConvergenceConfig,
  type Splat3DStepTimings,
  type Splat3DViewSampler,
} from "./splat3d/optimize";
import { fetchArrayBufferWithProgress, formatProgress } from "./splat/fetch_progress";
import type { TrainPlan } from "./clip/vision";

const SIDE = 256;
const HW = SIDE * SIDE;

interface Status {
  gpu: boolean;
  ready: boolean;
  running: boolean;
  step: number;
  view: number;
  cos: number | null;
  initialCos: number | null;
  error: string | null;
  phase: string;
  promptMode: ViewPromptMode;
  gridPromptMode: Grid9PromptMode;
  bgPromptMode: BackgroundPromptMode;
  backgroundMode: Splat3DBackgroundMode;
  alphaReg: "off" | "weak" | "medium";
  boundsReg: "off" | "weak" | "medium";
  framingMode: CameraFramingMode;
  profiling: boolean;
  viewsPerStep: number;
  viewSampler: Splat3DViewSampler;
  clipBatchSize: number;
  clipLayout: Splat3DClipLayout;
  gridDirectRaster: boolean;
}

const status: Status = {
  gpu: !!navigator.gpu,
  ready: false,
  running: false,
  step: 0,
  view: 0,
  cos: null,
  initialCos: null,
  error: null,
  phase: "boot",
  promptMode: "camera",
  gridPromptMode: "contact_sheet",
  bgPromptMode: "black",
  backgroundMode: "black",
  alphaReg: "off",
  boundsReg: "off",
  framingMode: "normal",
  profiling: false,
  viewsPerStep: 3,
  viewSampler: "epoch",
  clipBatchSize: 3,
  clipLayout: "per_view",
  gridDirectRaster: false,
};
(window as any).__splat3d = status;

const gridEl = document.getElementById("grid") as HTMLDivElement;
const promptInput = document.getElementById("prompt") as HTMLInputElement;
const viewSelect = document.getElementById("view") as HTMLSelectElement;
const promptModeSelect = document.getElementById("promptMode") as HTMLSelectElement;
const bgTextModeSelect = document.getElementById("bgTextMode") as HTMLSelectElement;
const backgroundModeSelect = document.getElementById("backgroundMode") as HTMLSelectElement;
const alphaRegSelect = document.getElementById("alphaReg") as HTMLSelectElement;
const boundsRegSelect = document.getElementById("boundsReg") as HTMLSelectElement;
const framingModeSelect = document.getElementById("framingMode") as HTMLSelectElement;
const viewBatchSelect = document.getElementById("viewBatch") as HTMLSelectElement;
const viewSamplerSelect = document.getElementById("viewSampler") as HTMLSelectElement;
const clipModeSelect = document.getElementById("clipMode") as HTMLSelectElement;
const clipLayoutSelect = document.getElementById("clipLayout") as HTMLSelectElement;
const gridPromptModeSelect = document.getElementById("gridPromptMode") as HTMLSelectElement;
const gridRasterModeSelect = document.getElementById("gridRasterMode") as HTMLSelectElement;
const optimizeBtn = document.getElementById("optimize") as HTMLButtonElement;
const resetBtn = document.getElementById("reset") as HTMLButtonElement;
const readoutEl = document.getElementById("readout") as HTMLDivElement;
const noticeEl = document.getElementById("notice") as HTMLDivElement;
const timingsEl = document.getElementById("timings") as HTMLDivElement;

function setNotice(msg: string): void {
  noticeEl.textContent = msg;
}

function fail(msg: string): void {
  status.error = msg;
  status.phase = "error";
  setNotice(msg);
  readoutEl.textContent = "—";
  console.error("[splat3d_page]", msg);
}

function renderReadout(): void {
  status.step = opt ? opt.stepCount : 0;
  const camera = opt?.cameras[displayView]?.name ?? "view";
  const parts: string[] = [`step ${status.step}`, camera];
  if (opt) parts.push(`${status.viewsPerStep}/${opt.cameras.length} views`);
  if (status.viewSampler === "random") parts.push("random");
  parts.push(status.clipBatchSize > 1 ? `clip x${status.clipBatchSize}` : "clip x1");
  if (status.clipLayout === "grid9_close2") parts.push("grid+2");
  if (status.clipLayout === "grid9_close2") {
    const gridText =
      status.gridPromptMode === "same"
        ? "grid=same text"
        : status.gridPromptMode === "literal_v2"
          ? "object grid text"
        : status.gridPromptMode === "literal"
          ? "literal grid text"
          : "grid text";
    parts.push(gridText);
    if (status.gridDirectRaster) parts.push("80px grid raster");
  }
  parts.push(status.promptMode === "camera" ? "camera text" : status.promptMode === "coarse" ? "coarse text" : "same text");
  if (status.bgPromptMode === "black") parts.push("black bg");
  if (status.bgPromptMode === "centered") parts.push("centered bg");
  if (status.backgroundMode !== "black") parts.push(status.backgroundMode === "curriculum" ? "bg curriculum" : "dark random bg");
  if (status.alphaReg !== "off") parts.push(`alpha ${status.alphaReg}`);
  if (status.boundsReg !== "off") parts.push(`bounds ${status.boundsReg}`);
  if (status.framingMode === "zoom_out") parts.push("zoom out");
  if (status.cos !== null) {
    const init = status.initialCos ?? status.cos;
    const d = status.cos - init;
    parts.push(`cos ${status.cos.toFixed(4)}`);
    parts.push(`init ${init.toFixed(4)}`);
    parts.push(`Δ ${d >= 0 ? "+" : ""}${d.toFixed(4)}`);
  }
  if (status.phase && status.phase !== "run") parts.push(`(${status.phase})`);
  readoutEl.textContent = parts.join("  ·  ");
}

function renderTimings(): void {
  if (!latestTimings) {
    timingsEl.textContent = "sampled wall profile waiting...";
    return;
  }
  const t = latestTimings;
  const total = Math.max(t.total, 0.001);
  const line = (name: string, ms: number): string => {
    const pct = (100 * ms) / total;
    return `${name.padEnd(11)} ${ms.toFixed(1).padStart(6)} ms ${pct.toFixed(0).padStart(3)}%`;
  };
  const lines = [
    `${t.timing === "gpu-timestamp" ? "sampled GPU step" : "sampled wall step"} ${status.step}`,
    `${t.views}/${t.totalViews} views · ${status.viewSampler} · ${t.clipMode === "batch" ? `batch CLIP x${t.clipBatchSize}` : "single CLIP"} · ${t.timing}`,
    line("opt total", t.total),
    line("raster", t.rasterFwd + t.rasterReplay + t.rasterBwd),
    line("  fwd", t.rasterFwd),
  ];
  if (t.rasterReplay > 0) lines.push(line("  replay", t.rasterReplay));
  lines.push(line("  bwd", t.rasterBwd));
  if (t.clipMode === "batch") {
    lines.push(line("clip batch", t.clipBatch));
  } else {
    lines.push(line("clip", t.clipFwd + t.clipBwd), line("  fwd", t.clipFwd), line("  bwd", t.clipBwd));
  }
  if (t.regularizer > 0) lines.push(line("reg", t.regularizer));
  lines.push(line("adam", t.adam), line("display", t.display), line("clear", t.clear), `sample every ${PROFILE_PERIOD} steps`);
  timingsEl.textContent = lines.join("\n");
}

let device!: GPUDevice;
let plan!: TrainPlan;
let weights!: Float32Array;
let opt!: Splat3DOptimizer;
let seed = 1;
let displayView = 0;
let gridDirty = false;
let latestTimings: Splat3DStepTimings | null = null;
let profileBusy = false;
const PROFILE_PERIOD = 30;

let blitPipe!: GPURenderPipeline;
let blitBind: GPUBindGroup | null = null;
let viewCtxs: GPUCanvasContext[] = [];
let viewTiles: HTMLDivElement[] = [];
let canvasFormat!: GPUTextureFormat;

function selectedClipBatchSize(): number {
  if (selectedClipLayout() === "grid9_close2") return 3;
  const n = Number(clipModeSelect.value);
  return Number.isFinite(n) && n > 1 ? Math.min(9, n | 0) : 1;
}

function selectedClipLayout(): Splat3DClipLayout {
  return clipLayoutSelect.value === "grid9_close2" ? "grid9_close2" : "per_view";
}

function selectedViewsPerStep(): number {
  if (selectedClipLayout() === "grid9_close2") return 9;
  const n = Number(viewBatchSelect.value);
  const maxViews = opt?.cameras.length ?? 9;
  return Number.isFinite(n) ? Math.max(1, Math.min(maxViews, n | 0)) : 3;
}

function selectedViewSampler(): Splat3DViewSampler {
  return viewSamplerSelect.value === "random" ? "random" : "epoch";
}

function selectedGridPromptMode(): Grid9PromptMode {
  if (gridPromptModeSelect.value === "same") return "same";
  if (gridPromptModeSelect.value === "literal_v2") return "literal_v2";
  if (gridPromptModeSelect.value === "literal") return "literal";
  return "contact_sheet";
}

function selectedGridDirectRaster(): boolean {
  return gridRasterModeSelect.value === "direct80";
}

function selectedPromptMode(): ViewPromptMode {
  if (promptModeSelect.value === "same") return "same";
  if (promptModeSelect.value === "coarse") return "coarse";
  return "camera";
}

function selectedBgPromptMode(): BackgroundPromptMode {
  if (bgTextModeSelect.value === "none") return "none";
  if (bgTextModeSelect.value === "centered") return "centered";
  return "black";
}

function selectedBackgroundMode(): Splat3DBackgroundMode {
  if (backgroundModeSelect.value === "dark_random") return "dark_random";
  if (backgroundModeSelect.value === "curriculum") return "curriculum";
  return "black";
}

function selectedAlphaReg(): Status["alphaReg"] {
  return alphaRegSelect.value === "medium" ? "medium" : alphaRegSelect.value === "weak" ? "weak" : "off";
}

function selectedBoundsReg(): Status["boundsReg"] {
  return boundsRegSelect.value === "medium" ? "medium" : boundsRegSelect.value === "weak" ? "weak" : "off";
}

function selectedFramingMode(): CameraFramingMode {
  return framingModeSelect.value === "zoom_out" ? "zoom_out" : "normal";
}

function selectedConvergenceConfig(): Splat3DConvergenceConfig {
  const alphaReg = selectedAlphaReg();
  const boundsReg = selectedBoundsReg();
  return {
    backgroundMode: selectedBackgroundMode(),
    opacitySparsity: alphaReg === "medium" ? 0.03 : alphaReg === "weak" ? 0.01 : 0,
    centerWeight: boundsReg === "medium" ? 0.006 : boundsReg === "weak" ? 0.002 : 0,
    radiusWeight: boundsReg === "medium" ? 0.012 : boundsReg === "weak" ? 0.004 : 0,
    targetRadius: 1.15,
  };
}

function syncConvergenceStatus(): void {
  status.backgroundMode = selectedBackgroundMode();
  status.alphaReg = selectedAlphaReg();
  status.boundsReg = selectedBoundsReg();
  status.framingMode = selectedFramingMode();
}

function syncClipLayoutControls(): void {
  const grid = selectedClipLayout() === "grid9_close2";
  if (grid) {
    clipModeSelect.value = "3";
    viewBatchSelect.value = "9";
  }
}

function setControlsDisabled(disabled: boolean): void {
  const grid = selectedClipLayout() === "grid9_close2";
  optimizeBtn.disabled = disabled;
  resetBtn.disabled = disabled;
  viewSelect.disabled = disabled;
  promptModeSelect.disabled = disabled;
  bgTextModeSelect.disabled = disabled;
  backgroundModeSelect.disabled = disabled;
  alphaRegSelect.disabled = disabled;
  boundsRegSelect.disabled = disabled;
  framingModeSelect.disabled = disabled;
  clipLayoutSelect.disabled = disabled;
  gridPromptModeSelect.disabled = disabled || !grid;
  gridRasterModeSelect.disabled = disabled || !grid;
  viewBatchSelect.disabled = disabled || grid;
  viewSamplerSelect.disabled = disabled;
  clipModeSelect.disabled = disabled || grid;
}

async function rebuildOptimizer(nextSeed: number, phase: string): Promise<void> {
  status.phase = phase;
  status.clipLayout = selectedClipLayout();
  status.gridPromptMode = selectedGridPromptMode();
  status.gridDirectRaster = selectedGridDirectRaster();
  status.viewsPerStep = selectedViewsPerStep();
  status.viewSampler = selectedViewSampler();
  status.clipBatchSize = selectedClipBatchSize();
  syncConvergenceStatus();
  renderReadout();
  const old = opt;
  opt = await Splat3DOptimizer.create(device, plan, weights, {
    seed: nextSeed,
    clipBatchSize: status.clipBatchSize,
    clipLayout: status.clipLayout,
    viewSampler: status.viewSampler,
    gridDirectRaster: status.gridDirectRaster,
    convergence: selectedConvergenceConfig(),
    cameras: camerasForFraming(status.framingMode),
  });
  status.clipLayout = opt.clipLayout;
  status.clipBatchSize = opt.clipBatchSize;
  status.viewSampler = opt.viewSampler;
  old?.destroy();
  populateViews();
  rebuildBlitBind();
  gridDirty = true;
  status.step = 0;
  latestTimings = null;
  renderTimings();
  renderReadout();
}

const BLIT_WGSL = /* wgsl */ `
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
  let HW : u32 = ${HW}u;
  let i : u32 = y * ${SIDE}u + x;
  return vec4<f32>(img[i], img[HW + i], img[2u * HW + i], 1.0);
}
`;

async function buildBlitPipeline(): Promise<void> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: BLIT_WGSL });
  blitPipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format: canvasFormat }] },
    primitive: { topology: "triangle-list" },
  });
  const err = await device.popErrorScope();
  if (err) throw new Error(`blit pipeline invalid: ${err.message}`);
}

function rebuildBlitBind(): void {
  blitBind = device.createBindGroup({
    layout: blitPipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: opt.raster.image } }],
  });
}

function recordBlit(enc: GPUCommandEncoder, target: GPUCanvasContext): void {
  if (!blitBind) return;
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: target.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
  });
  pass.setPipeline(blitPipe);
  pass.setBindGroup(0, blitBind);
  pass.draw(3);
  pass.end();
}

function renderGrid(): void {
  if (!blitBind || !viewCtxs.length) return;
  opt.prepareDisplayFrame();
  const enc = device.createCommandEncoder();
  for (let view = 0; view < viewCtxs.length; view++) {
    opt.raster.recordForward(enc, view);
    recordBlit(enc, viewCtxs[view]);
  }
  device.queue.submit([enc.finish()]);
}

const TF_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm";
const nativeImport = new Function("u", "return import(u)") as (u: string) => Promise<any>;
let tokenizer: any = null;
let textModel: any = null;
const promptEmbedCache = new Map<string, Promise<Float32Array>>();

async function loadTextModel(onProgress?: (msg: string) => void): Promise<void> {
  if (textModel) return;
  const tf: any = await nativeImport(TF_URL);
  tf.env.allowRemoteModels = true;
  const id = "Nbardy/nff-clip-splat-weights"; // self-hosted alongside the vision weights
  const progress_callback = (p: any) => {
    if (p.status === "progress" && p.total) {
      const pct = Math.round(p.progress ?? (p.loaded / p.total) * 100);
      const fill = Math.round((pct / 100) * 16);
      const bar = "█".repeat(fill) + "░".repeat(16 - fill);
      onProgress?.(`loading text encoder  [${bar}] ${pct}%  ·  ${(p.loaded / 1e6).toFixed(1)}/${(p.total / 1e6).toFixed(0)} MB`);
    }
  };
  tokenizer = await tf.AutoTokenizer.from_pretrained(id, { progress_callback });
  textModel = await tf.CLIPTextModelWithProjection.from_pretrained(id, {
    dtype: "fp16", // 84 MB, lossless vs fp32
    device: "wasm", // keep text off the shared render GPU
    session_options: { graphOptimizationLevel: "basic" }, // dodges the LayerNormFusion bug
    progress_callback,
  });
}

async function encodePrompt(text: string): Promise<Float32Array> {
  await loadTextModel();
  const enc = await tokenizer(text, {
    padding: "max_length",
    max_length: 77,
    truncation: true,
  });
  const out = await textModel(enc);
  const d = out.text_embeds.data as ArrayLike<number>;
  const vec = new Float32Array(512);
  for (let i = 0; i < 512; i++) vec[i] = d[i];
  return vec;
}

function encodePromptCached(text: string): Promise<Float32Array> {
  const key = text.trim();
  let cached = promptEmbedCache.get(key);
  if (!cached) {
    cached = encodePrompt(key).catch((e) => {
      promptEmbedCache.delete(key);
      throw e;
    });
    promptEmbedCache.set(key, cached);
  }
  return cached;
}

let viewEmbeds: Float32Array[] | null = null;
let stepsSinceReadout = 0;
let cosBusy = false;

async function runProfiledStep(): Promise<void> {
  if (!viewEmbeds || profileBusy) return;
  const profiledOpt = opt;
  const profiledView = displayView;
  const profiledViewsPerStep = status.viewsPerStep;
  profileBusy = true;
  status.profiling = true;
  status.phase = "profile";
  renderReadout();
  try {
    const timings = await profiledOpt.profileStep(profiledView, profiledViewsPerStep);
    if (profiledOpt !== opt || !status.running) return;
    latestTimings = timings;
    status.step = profiledOpt.stepCount;
    stepsSinceReadout += 1;
    gridDirty = true;
    renderTimings();
    if (stepsSinceReadout >= 3) {
      stepsSinceReadout = 0;
      void updateCos();
    }
  } catch (e: any) {
    fail(`profile step failed: ${e?.message ?? e}`);
  } finally {
    status.profiling = false;
    if (status.phase === "profile") status.phase = status.running ? "run" : "idle";
    profileBusy = false;
    renderReadout();
  }
}

async function updateCos(): Promise<void> {
  if (!viewEmbeds || cosBusy) return;
  cosBusy = true;
  try {
    const emb = await opt.currentEmbedding(displayView);
    const c = cosine(emb, viewEmbeds[displayView]);
    status.cos = c;
    if (status.initialCos === null) status.initialCos = c;
    renderReadout();
  } finally {
    cosBusy = false;
  }
}

function frame(): void {
  if (status.running && viewEmbeds && !profileBusy) {
    const shouldProfile = opt.stepCount > 0 && opt.stepCount % PROFILE_PERIOD === 0;
    if (shouldProfile) {
      void runProfiledStep();
    } else {
      opt.step(displayView, status.viewsPerStep);
      gridDirty = true;
      stepsSinceReadout += 1;
      status.step = opt.stepCount;
      if (stepsSinceReadout >= 3) {
        stepsSinceReadout = 0;
        void updateCos();
      }
    }
  }
  if (gridDirty) {
    renderGrid();
    gridDirty = false;
  }
  requestAnimationFrame(frame);
}

async function onOptimize(): Promise<void> {
  if (!status.ready) return;
  if (profileBusy) return;
  syncClipLayoutControls();
  const text = promptInput.value.trim() || "a photo of a cat";
  setControlsDisabled(true);
  status.running = false;
  status.phase = "encoding";
  status.cos = null;
  status.initialCos = null;
  latestTimings = null;
  status.clipLayout = selectedClipLayout();
  status.gridPromptMode = selectedGridPromptMode();
  status.gridDirectRaster = selectedGridDirectRaster();
  status.viewsPerStep = selectedViewsPerStep();
  status.viewSampler = selectedViewSampler();
  status.clipBatchSize = selectedClipBatchSize();
  status.promptMode = selectedPromptMode();
  status.bgPromptMode = selectedBgPromptMode();
  syncConvergenceStatus();
  renderTimings();
  renderReadout();
  try {
    const embeds: Float32Array[] = [];
    if (status.promptMode === "same") {
      setNotice("encoding prompt 1/1...");
      const embed = await encodePromptCached(buildBasePrompt(text, status.bgPromptMode));
      for (let i = 0; i < opt.cameras.length; i++) embeds.push(embed);
    } else {
      for (let i = 0; i < opt.cameras.length; i++) {
        setNotice(`encoding prompt ${i + 1}/${opt.cameras.length}...`);
        const promptText =
          status.promptMode === "coarse"
            ? buildCoarseViewPrompt(text, opt.cameras[i], status.bgPromptMode)
            : buildViewPrompt(text, opt.cameras[i], status.bgPromptMode);
        embeds.push(await encodePromptCached(promptText));
      }
	    }
	    viewEmbeds = embeds;
	    opt.setViewPrompts(embeds);
    if (status.clipLayout === "grid9_close2") {
      setNotice("encoding grid prompt...");
      opt.setGridPrompt(await encodePromptCached(buildGrid9Prompt(text, status.bgPromptMode, status.gridPromptMode)));
    }
	    const e0 = await opt.currentEmbedding(displayView);
    status.initialCos = cosine(e0, embeds[displayView]);
    status.cos = status.initialCos;
    stepsSinceReadout = 0;
    setNotice("");
    status.phase = "run";
    status.running = true;
    gridDirty = true;
    renderReadout();
	  } catch (e: any) {
	    fail(`text encode failed: ${e?.message ?? e}`);
	  } finally {
	    setControlsDisabled(false);
	  }
}

async function onReset(): Promise<void> {
  if (!status.ready) return;
  if (profileBusy) {
    setNotice("wait for profiling sample to finish before reset");
    return;
  }
  status.running = false;
  viewEmbeds = null;
  status.cos = null;
  status.initialCos = null;
  latestTimings = null;
  status.phase = "reset";
  seed += 1;
  await rebuildOptimizer(seed, "reset");
  status.phase = "idle";
  setNotice("");
  renderReadout();
}

async function onViewChange(): Promise<void> {
  setDisplayView(Math.max(0, viewSelect.selectedIndex));
  if (!status.ready) return;
  status.cos = null;
  status.initialCos = null;
  if (viewEmbeds) void updateCos();
  renderReadout();
}

function setDisplayView(view: number): void {
  displayView = Math.max(0, Math.min(opt ? opt.cameras.length - 1 : 0, view | 0));
  status.view = displayView;
  viewSelect.selectedIndex = displayView;
  for (let i = 0; i < viewTiles.length; i++) {
    viewTiles[i].classList.toggle("active", i === displayView);
  }
}

function onPromptModeChange(): void {
  status.promptMode = selectedPromptMode();
  status.gridPromptMode = selectedGridPromptMode();
  status.gridDirectRaster = selectedGridDirectRaster();
  status.bgPromptMode = selectedBgPromptMode();
  latestTimings = null;
  if (viewEmbeds) {
    status.running = false;
    viewEmbeds = null;
    status.cos = null;
    status.initialCos = null;
    status.phase = "idle";
    setNotice("");
  }
  renderTimings();
  renderReadout();
}

async function onConvergenceSettingsChange(): Promise<void> {
  if (!status.ready) return;
  if (profileBusy) {
    setNotice("wait for profiling sample to finish before changing convergence settings");
    backgroundModeSelect.value = status.backgroundMode;
    alphaRegSelect.value = status.alphaReg;
    boundsRegSelect.value = status.boundsReg;
    framingModeSelect.value = status.framingMode;
    return;
  }
  syncConvergenceStatus();
  status.running = false;
  viewEmbeds = null;
  status.cos = null;
  status.initialCos = null;
  latestTimings = null;
  setControlsDisabled(true);
  try {
    await rebuildOptimizer(seed, "convergence");
    setNotice("");
    status.phase = "idle";
  } catch (e: any) {
    fail(`convergence settings change failed: ${e?.message ?? e}`);
  } finally {
    setControlsDisabled(false);
    renderReadout();
  }
}

function onViewBatchChange(): void {
  syncClipLayoutControls();
  status.viewsPerStep = selectedViewsPerStep();
  latestTimings = null;
  renderTimings();
  renderReadout();
}

async function onClipSettingsChange(): Promise<void> {
  if (!status.ready) return;
  syncClipLayoutControls();
  if (profileBusy) {
    setNotice("wait for profiling sample to finish before changing CLIP settings");
    clipModeSelect.value = String(status.clipBatchSize);
    clipLayoutSelect.value = status.clipLayout;
    viewSamplerSelect.value = status.viewSampler;
    syncClipLayoutControls();
    return;
  }
  status.running = false;
  viewEmbeds = null;
  status.cos = null;
  status.initialCos = null;
  latestTimings = null;
  setControlsDisabled(true);
  try {
    await rebuildOptimizer(seed, "optimizer");
    setNotice("");
    status.phase = "idle";
  } catch (e: any) {
    fail(`clip settings change failed: ${e?.message ?? e}`);
  } finally {
    setControlsDisabled(false);
    renderReadout();
  }
}

function populateViews(): void {
  viewSelect.textContent = "";
  gridEl.textContent = "";
  viewCtxs = [];
  viewTiles = [];
  for (let i = 0; i < opt.cameras.length; i++) {
    const camera = opt.cameras[i];
    const option = document.createElement("option");
    option.value = camera.name;
    option.textContent = camera.name;
    viewSelect.appendChild(option);

    const tile = document.createElement("div");
    tile.className = "tile";
    const canvas = document.createElement("canvas");
    canvas.className = "view";
    canvas.width = SIDE;
    canvas.height = SIDE;
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = camera.name;
    tile.append(canvas, label);
    tile.addEventListener("click", () => {
      setDisplayView(i);
      status.cos = null;
      status.initialCos = null;
      if (viewEmbeds) void updateCos();
      renderReadout();
    });
    const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
    ctx.configure({ device, format: canvasFormat, alphaMode: "opaque" });
    gridEl.appendChild(tile);
    viewCtxs.push(ctx);
    viewTiles.push(tile);
  }
  setDisplayView(displayView);
}

async function boot(): Promise<void> {
  if (!navigator.gpu) {
    fail("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled.");
    setControlsDisabled(true);
    return;
  }
  status.phase = "adapter";
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return fail("no WebGPU adapter available.");
  device = await adapter.requestDevice();
  device.addEventListener?.("uncapturederror", (ev: any) => {
    console.error("[webgpu]", ev.error?.message ?? ev.error);
  });

  canvasFormat = navigator.gpu.getPreferredCanvasFormat();

  status.phase = "weights";
  // Prod (GitHub Pages) fetches the packed weights from the HF Hub: GitHub
  // release assets send no CORS header, HF does — same host the text model
  // loads from (upload via tools/splat/upload_weights.py). Local dev uses the
  // fast same-origin static server (tools/splat/serve.mjs). Same loader as the
  // 2D page (src/splat_page.ts) — both share the one CLIP weights repo.
  const isLocal = ["localhost", "127.0.0.1"].includes(location.hostname);
  const MODEL_BASE = isLocal
    ? "/models/mobileclip_s0/"
    : "https://huggingface.co/Nbardy/nff-clip-splat-weights/resolve/main/";
  const srcLabel = isLocal ? "" : " from HF";
  readoutEl.textContent = `fetching CLIP plan${srcLabel}…`;
  const planRes = await fetch(MODEL_BASE + "plan_train.json");
  if (!planRes.ok) return fail(`plan_train.json fetch ${planRes.status} from ${MODEL_BASE}`);
  plan = (await planRes.json()) as TrainPlan;
  // 82 MB weights with a live progress bar (Content-Length + streamed body).
  let wbuf: ArrayBuffer;
  try {
    wbuf = await fetchArrayBufferWithProgress(MODEL_BASE + "weights_train.bin", (p) => {
      readoutEl.textContent = formatProgress(`loading CLIP weights${srcLabel}`, p);
    });
  } catch (e: any) {
    return fail(`weights_train.bin fetch failed from ${MODEL_BASE}: ${e?.message ?? e}`);
  }
  weights = new Float32Array(wbuf);

  status.phase = "optimizer";
  readoutEl.textContent = "building 3D optimizer…";
  await buildBlitPipeline();
  syncClipLayoutControls();
  syncConvergenceStatus();
  opt = await Splat3DOptimizer.create(device, plan, weights, {
    seed,
    clipBatchSize: selectedClipBatchSize(),
    clipLayout: selectedClipLayout(),
    viewSampler: selectedViewSampler(),
    gridDirectRaster: selectedGridDirectRaster(),
    convergence: selectedConvergenceConfig(),
    cameras: camerasForFraming(status.framingMode),
  });
  status.clipLayout = opt.clipLayout;
  status.viewsPerStep = selectedViewsPerStep();
  status.viewSampler = opt.viewSampler;
  status.clipBatchSize = opt.clipBatchSize;
  status.gridPromptMode = selectedGridPromptMode();
  status.gridDirectRaster = selectedGridDirectRaster();
  populateViews();
  rebuildBlitBind();
  gridDirty = true;

  // Preload the text encoder at boot (with its own progress bar) so the first
  // Optimize is instant instead of stalling on an 84 MB download (× the 9 views).
  status.phase = "textmodel";
  await loadTextModel((msg) => { readoutEl.textContent = msg; });

  status.ready = true;
  status.phase = "idle";
  setControlsDisabled(false);
  setNotice("");
  renderReadout();
  requestAnimationFrame(frame);
}

optimizeBtn.addEventListener("click", () => void onOptimize());
resetBtn.addEventListener("click", () => void onReset());
viewSelect.addEventListener("change", () => void onViewChange());
promptModeSelect.addEventListener("change", onPromptModeChange);
bgTextModeSelect.addEventListener("change", onPromptModeChange);
backgroundModeSelect.addEventListener("change", () => void onConvergenceSettingsChange());
alphaRegSelect.addEventListener("change", () => void onConvergenceSettingsChange());
boundsRegSelect.addEventListener("change", () => void onConvergenceSettingsChange());
framingModeSelect.addEventListener("change", () => void onConvergenceSettingsChange());
viewBatchSelect.addEventListener("change", onViewBatchChange);
viewSamplerSelect.addEventListener("change", () => void onClipSettingsChange());
clipModeSelect.addEventListener("change", () => void onClipSettingsChange());
clipLayoutSelect.addEventListener("change", () => void onClipSettingsChange());
gridPromptModeSelect.addEventListener("change", onPromptModeChange);
gridRasterModeSelect.addEventListener("change", () => void onClipSettingsChange());
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") void onOptimize();
});

boot().catch((e) => fail(`boot failed: ${e?.message ?? e}`));
