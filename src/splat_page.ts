/**
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
 */
/// <reference types="@webgpu/types" />
import { SplatOptimizer, cosine } from "./splat/optimize";
import type { TrainPlan } from "./clip/vision";

const SIDE = 256;
const HW = SIDE * SIDE;
// NOTE: we pass NO { G } / LR overrides to SplatOptimizer.create — it defaults to
// the LEGIBLE regime (LEGIBLE_G≈12K large opaque splats + LEGIBLE_LRS) tuned to
// produce a recognizable subject in ~20 steps. Passing G=200_000 reverts to the
// old noise regime (200K tiny translucent splats average to gray). Only `seed`
// is overridden, so Reset re-randomizes.

// Structured status the puppeteer gate reads (window.__splat) — the DOM readout
// is for humans, this is the machine-checkable mirror of the same numbers.
interface Status {
  gpu: boolean;
  ready: boolean;
  running: boolean;
  step: number;
  cos: number | null;
  initialCos: number | null;
  error: string | null;
  phase: string;
}
const status: Status = {
  gpu: !!navigator.gpu,
  ready: false,
  running: false,
  step: 0,
  cos: null,
  initialCos: null,
  error: null,
  phase: "boot",
};
(window as any).__splat = status;

// ── DOM ──────────────────────────────────────────────────────────────────────
const canvas = document.getElementById("splat") as HTMLCanvasElement;
const promptInput = document.getElementById("prompt") as HTMLInputElement;
const optimizeBtn = document.getElementById("optimize") as HTMLButtonElement;
const nudgeBtn = document.getElementById("nudge") as HTMLButtonElement;
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
  // eslint-disable-next-line no-console
  console.error("[splat_page]", msg);
}

function renderReadout(): void {
  status.step = opt ? opt.stepCount : 0;
  const parts: string[] = [`step ${status.step}`];
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

// ── Module state (set during boot) ───────────────────────────────────────────
let device!: GPUDevice;
let ctx!: GPUCanvasContext;
let plan!: TrainPlan;
let weights!: Float32Array;
let opt!: SplatOptimizer;
let seed = 1;

// Blit pipeline (storage-buffer image → canvas). Pipeline built once; the bind
// group is rebuilt whenever the optimizer (hence raster.image) is recreated.
let blitPipe!: GPURenderPipeline;
let blitBind: GPUBindGroup | null = null;
let canvasFormat!: GPUTextureFormat;

const BLIT_WGSL = /* wgsl */ `
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

function blit(): void {
  if (!blitBind) return;
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: ctx.getCurrentTexture().createView(),
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
  device.queue.submit([enc.finish()]);
}

// ── Text encoding (transformers.js, loaded lazily from the CDN) ──────────────
// SAME model/tokenizer as tools/clip/text_onnx.mjs: Xenova/mobileclip_s0,
// 77-token context, pad id 0 after EOT, 512-d float32 output, NOT L2-normalized
// (the −cos loss normalizes; cosine is scale-invariant anyway). We use the fp32
// text_model.onnx (dtype:"fp32") to dodge the fp16 SimplifiedLayerNormFusion ORT
// bug noted in text_onnx.mjs, and device:"wasm" so text stays OFF our shared GPU.
// jsDelivr's /+esm endpoint (NOT the raw dist/*.web.js) — it rebundles the web
// build and REWRITES bare dep specifiers (`onnxruntime-web/webgpu`) into
// resolvable CDN URLs, which the raw file leaves bare and the browser then
// can't resolve. All ORT glue + wasm then load from the CDN too.
const TF_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm";
// Hide the specifier behind a Function-constructor indirection so the BUNDLER
// leaves it alone and the BROWSER does a genuine native dynamic import of the
// CDN URL. A plain `import(TF_URL)` gets rewritten into a parcel module helper
// that would try to resolve the URL as a local bundle.
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

// ── Optimize loop ─────────────────────────────────────────────────────────────
let promptEmbed: Float32Array | null = null;
let stepsSinceReadout = 0;
let cosBusy = false;
let nudgeBusy = false;

async function updateCos(): Promise<void> {
  if (!promptEmbed || cosBusy || nudgeBusy) return;
  cosBusy = true;
  try {
    const emb = await opt.currentEmbedding();
    const c = cosine(emb, promptEmbed);
    status.cos = c;
    if (status.initialCos === null) status.initialCos = c;
    renderReadout();
  } finally {
    cosBusy = false;
  }
}

function frame(): void {
  if (status.running && promptEmbed) {
    // 2 optimize steps/frame keeps the page responsive; each step is one submit
    // (raster fwd → CLIP fwd+loss+bwd → raster bwd → Adam). At LEGIBLE_G≈12K
    // splats a step is cheap, so the loop stays smooth.
    opt.step();
    opt.step();
    stepsSinceReadout += 2;
    status.step = opt.stepCount;
    if (stepsSinceReadout >= 14) {
      stepsSinceReadout = 0;
      void updateCos(); // async readback; don't block the frame on it
    }
  }
  blit();
  requestAnimationFrame(frame);
}

async function onOptimize(): Promise<void> {
  if (!status.ready) return;
  const text = promptInput.value.trim() || "a photo of a cat";
  optimizeBtn.disabled = true;
  status.phase = "encoding";
  status.running = false;
  setNotice("encoding prompt (first use downloads the text model — slow)…");
  renderReadout();
  try {
    const emb = await encodePrompt(text);
    promptEmbed = emb;
    opt.setPrompt(emb);
    // Baseline cos on the CURRENT splats — this is the "initial" the gate checks
    // the run rises above.
    const e0 = await opt.currentEmbedding();
    status.initialCos = cosine(e0, emb);
    status.cos = status.initialCos;
    stepsSinceReadout = 0;
    setNotice("");
    status.phase = "run";
    status.running = true;
    renderReadout();
  } catch (e: any) {
    fail(`text encode failed: ${e?.message ?? e}`);
  } finally {
    optimizeBtn.disabled = false;
  }
}

async function onReset(): Promise<void> {
  if (!status.ready) return;
  status.running = false;
  promptEmbed = null;
  status.cos = null;
  status.initialCos = null;
  status.phase = "reset";
  seed += 1;
  const old = opt;
  opt = await SplatOptimizer.create(device, plan, weights, { seed });
  old.destroy();
  rebuildBlitBind();
  await opt.renderImage(); // repopulate raster.image for the blit
  promptInput.value = "";
  status.step = 0;
  setNotice("");
  renderReadout();
}

async function onNudge(): Promise<void> {
  if (!status.ready || nudgeBusy) return;
  nudgeBusy = true;
  const resume = status.running;
  status.running = false;
  status.phase = "nudge";
  nudgeBtn.disabled = true;
  seed += 1;
  renderReadout();
  try {
    await opt.nudge({ seed });
    await opt.renderImage();
    stepsSinceReadout = 0;
    if (promptEmbed) {
      const emb = await opt.currentEmbedding();
      status.cos = cosine(emb, promptEmbed);
    }
    status.phase = resume && promptEmbed ? "run" : "idle";
    status.running = resume && !!promptEmbed;
    setNotice("");
    renderReadout();
  } catch (e: any) {
    fail(`nudge failed: ${e?.message ?? e}`);
  } finally {
    nudgeBusy = false;
    nudgeBtn.disabled = false;
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────────
async function boot(): Promise<void> {
  if (!navigator.gpu) {
    fail("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled.");
    optimizeBtn.disabled = true;
    nudgeBtn.disabled = true;
    resetBtn.disabled = true;
    return;
  }
  status.phase = "adapter";
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    fail("no WebGPU adapter available.");
    return;
  }
  device = await adapter.requestDevice();
  device.addEventListener?.("uncapturederror", (ev: any) => {
    // eslint-disable-next-line no-console
    console.error("[webgpu]", ev.error?.message ?? ev.error);
  });

  ctx = canvas.getContext("webgpu") as GPUCanvasContext;
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: canvasFormat, alphaMode: "opaque" });

  status.phase = "weights";
  // Prod (GitHub Pages) fetches the packed weights from the HF Hub: GitHub
  // release assets send no CORS header, HF does — same host the text model
  // loads from (upload via tools/splat/upload_weights.py). Local dev uses the
  // fast same-origin static server (tools/splat/serve.mjs) so iteration
  // doesn't re-pull 82 MB over the network.
  const isLocal = ["localhost", "127.0.0.1"].includes(location.hostname);
  const MODEL_BASE = isLocal
    ? "/models/mobileclip_s0/"
    : "https://huggingface.co/Nbardy/nff-clip-splat-weights/resolve/main/";
  readoutEl.textContent = `fetching CLIP vision weights (82 MB)${isLocal ? "" : " from HF"}…`;
  const [planRes, wRes] = await Promise.all([
    fetch(MODEL_BASE + "plan_train.json"),
    fetch(MODEL_BASE + "weights_train.bin"),
  ]);
  if (!planRes.ok) return fail(`plan_train.json fetch ${planRes.status} from ${MODEL_BASE}`);
  if (!wRes.ok) return fail(`weights_train.bin fetch ${wRes.status} from ${MODEL_BASE}`);
  plan = (await planRes.json()) as TrainPlan;
  weights = new Float32Array(await wRes.arrayBuffer());

  status.phase = "optimizer";
  readoutEl.textContent = "building optimizer…";
  await buildBlitPipeline();
  opt = await SplatOptimizer.create(device, plan, weights, { seed });
  rebuildBlitBind();
  await opt.renderImage(); // populate raster.image so the canvas isn't blank pre-optimize

  status.ready = true;
  status.phase = "idle";
  optimizeBtn.disabled = false;
  nudgeBtn.disabled = false;
  resetBtn.disabled = false;
  setNotice("");
  renderReadout();
  requestAnimationFrame(frame);
}

optimizeBtn.addEventListener("click", () => void onOptimize());
nudgeBtn.addEventListener("click", () => void onNudge());
resetBtn.addEventListener("click", () => void onReset());
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") void onOptimize();
});

boot().catch((e) => fail(`boot failed: ${e?.message ?? e}`));
