/// <reference types="@webgpu/types" />
import { buildBasePrompt, buildViewPrompt } from "./splat3d/cameras";
import { Splat3DOptimizer, cosine } from "./splat3d/optimize";
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
  promptMode: "camera" | "same";
  blackBgText: boolean;
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
  blackBgText: true,
};
(window as any).__splat3d = status;

const gridEl = document.getElementById("grid") as HTMLDivElement;
const promptInput = document.getElementById("prompt") as HTMLInputElement;
const viewSelect = document.getElementById("view") as HTMLSelectElement;
const promptModeSelect = document.getElementById("promptMode") as HTMLSelectElement;
const bgTextModeSelect = document.getElementById("bgTextMode") as HTMLSelectElement;
const optimizeBtn = document.getElementById("optimize") as HTMLButtonElement;
const resetBtn = document.getElementById("reset") as HTMLButtonElement;
const readoutEl = document.getElementById("readout") as HTMLDivElement;
const noticeEl = document.getElementById("notice") as HTMLDivElement;

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
  parts.push(status.promptMode === "camera" ? "camera text" : "same text");
  if (status.blackBgText) parts.push("black bg");
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

let device!: GPUDevice;
let plan!: TrainPlan;
let weights!: Float32Array;
let opt!: Splat3DOptimizer;
let seed = 1;
let displayView = 0;
let gridDirty = false;

let blitPipe!: GPURenderPipeline;
let blitBind: GPUBindGroup | null = null;
let viewCtxs: GPUCanvasContext[] = [];
let viewTiles: HTMLDivElement[] = [];
let canvasFormat!: GPUTextureFormat;

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

async function loadTextModel(): Promise<void> {
  if (textModel) return;
  const tf: any = await nativeImport(TF_URL);
  tf.env.allowRemoteModels = true;
  const id = "Xenova/mobileclip_s0";
  tokenizer = await tf.AutoTokenizer.from_pretrained(id);
  textModel = await tf.CLIPTextModelWithProjection.from_pretrained(id, {
    dtype: "fp32",
    device: "wasm",
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

let viewEmbeds: Float32Array[] | null = null;
let stepsSinceReadout = 0;
let cosBusy = false;

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
  if (status.running && viewEmbeds) {
    opt.step(displayView);
    gridDirty = true;
    stepsSinceReadout += 1;
    status.step = opt.stepCount;
    if (stepsSinceReadout >= 3) {
      stepsSinceReadout = 0;
      void updateCos();
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
  const text = promptInput.value.trim() || "a photo of a cat";
  optimizeBtn.disabled = true;
  resetBtn.disabled = true;
  viewSelect.disabled = true;
  promptModeSelect.disabled = true;
  bgTextModeSelect.disabled = true;
  status.running = false;
  status.phase = "encoding";
  status.cos = null;
  status.initialCos = null;
  status.promptMode = promptModeSelect.value === "same" ? "same" : "camera";
  status.blackBgText = bgTextModeSelect.value !== "none";
  renderReadout();
  try {
    const embeds: Float32Array[] = [];
    for (let i = 0; i < opt.cameras.length; i++) {
      setNotice(`encoding prompt ${i + 1}/${opt.cameras.length}…`);
      const prompt =
        status.promptMode === "camera"
          ? buildViewPrompt(text, opt.cameras[i], status.blackBgText)
          : buildBasePrompt(text, status.blackBgText);
      embeds.push(await encodePrompt(prompt));
    }
    viewEmbeds = embeds;
    opt.setViewPrompts(embeds);
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
    optimizeBtn.disabled = false;
    resetBtn.disabled = false;
    viewSelect.disabled = false;
    promptModeSelect.disabled = false;
    bgTextModeSelect.disabled = false;
  }
}

async function onReset(): Promise<void> {
  if (!status.ready) return;
  status.running = false;
  viewEmbeds = null;
  status.cos = null;
  status.initialCos = null;
  status.phase = "reset";
  seed += 1;
  const old = opt;
  opt = await Splat3DOptimizer.create(device, plan, weights, { seed });
  old.destroy();
  rebuildBlitBind();
  gridDirty = true;
  status.step = 0;
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
  status.promptMode = promptModeSelect.value === "same" ? "same" : "camera";
  status.blackBgText = bgTextModeSelect.value !== "none";
  if (viewEmbeds) {
    status.running = false;
    viewEmbeds = null;
    status.cos = null;
    status.initialCos = null;
    status.phase = "idle";
    setNotice("");
  }
  renderReadout();
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
    optimizeBtn.disabled = true;
    resetBtn.disabled = true;
    viewSelect.disabled = true;
    promptModeSelect.disabled = true;
    bgTextModeSelect.disabled = true;
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
  opt = await Splat3DOptimizer.create(device, plan, weights, { seed });
  populateViews();
  rebuildBlitBind();
  gridDirty = true;

  status.ready = true;
  status.phase = "idle";
  optimizeBtn.disabled = false;
  resetBtn.disabled = false;
  viewSelect.disabled = false;
  promptModeSelect.disabled = false;
  bgTextModeSelect.disabled = false;
  setNotice("");
  renderReadout();
  requestAnimationFrame(frame);
}

optimizeBtn.addEventListener("click", () => void onOptimize());
resetBtn.addEventListener("click", () => void onReset());
viewSelect.addEventListener("change", () => void onViewChange());
promptModeSelect.addEventListener("change", onPromptModeChange);
bgTextModeSelect.addEventListener("change", onPromptModeChange);
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") void onOptimize();
});

boot().catch((e) => fail(`boot failed: ${e?.message ?? e}`));
