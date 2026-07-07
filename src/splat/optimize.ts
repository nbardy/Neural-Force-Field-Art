/**
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
 */
/// <reference types="@webgpu/types" />
import { RasterEngine } from "./raster";
import { PARAM_STRIDE } from "./raster_wgsl";
import { VisionTrainer, type TrainPlan } from "../clip/vision";
import { type AdamLRs, type AdamHyper, DEFAULT_HYPER } from "./adam_wgsl";

/** CLIP-S0 fixes the encoder input at 256×256; the splat canvas matches so the
 *  image/grad handoff is a straight buffer copy (no resample kernel needed). */
const SIDE = 256;
const IMG_BYTES = 3 * SIDE * SIDE * 4;

/** Init distribution for the splat cloud. Defaults produce the "legible"
 *  regime: a moderate count of large, fairly opaque Gaussians that form
 *  colour REGIONS (not a per-pixel noise average). See tuneDefaults(). */
export interface SplatInit {
  /** mean Gaussian radius in px (logScale seeded at ln(scale)). */
  scale?: number;
  /** ± lognormal jitter on the radius. */
  scaleJitter?: number;
  /** opacityRaw seed (sigmoid → opacity); 0 → 0.5, +1 → 0.73. */
  opacityRaw?: number;
  /** stddev of the per-channel colorRaw (sigmoid-logit space). */
  colorSpread?: number;
}

export interface SplatOptimizerConfig {
  /** splat count (default 12_000 — the legible regime, not the 200K gate). */
  G?: number;
  /** per-tile fixed-bin capacity, power of two (default 2048). */
  cap?: number;
  /** background colour composited as accum + T·bg (default mid gray). */
  bg?: [number, number, number];
  /** deterministic init RNG seed (default 1). */
  seed?: number;
  /** init distribution (see SplatInit); overridden by initParams if given. */
  init?: SplatInit;
  /** fully-formed packed params [G*9] — overrides init (tuning / restart). */
  initParams?: Float32Array;
  lrs?: AdamLRs;
  hyper?: AdamHyper;
}

/** Tuned defaults for fast, legible CLIP-splat optimization (structure in
 *  ~20 steps): far fewer, larger, more-opaque splats than the 200K gate, and
 *  aggressive LRs so 20 Adam steps actually move colour/position. Dialed in
 *  empirically (tools/splat/tune.ts). */
export const LEGIBLE_G = 12_000;
export const LEGIBLE_INIT: Required<SplatInit> = {
  scale: 9,
  scaleJitter: 0.35,
  opacityRaw: 0.4,
  colorSpread: 1.2,
};
export const LEGIBLE_LRS: AdamLRs = {
  mean: 1.5,
  logScale: 0.06,
  theta: 0.08,
  color: 0.12,
  opacity: 0.06,
};

export interface SplatNudgeOptions {
  /** deterministic random source for the fresh target parameters. */
  seed?: number;
  /** 0 = no-op, 1 = full random reinit; default is a visible but partial nudge. */
  amount?: number;
  /** optional override for the fresh random target distribution. */
  init?: SplatInit;
}

export const DEFAULT_NUDGE_AMOUNT = 0.18;

export class SplatOptimizer {
  readonly device: GPUDevice;
  readonly raster: RasterEngine;
  readonly trainer: VisionTrainer;
  readonly side = SIDE;
  private step_ = 0;
  private readonly lrs: AdamLRs;
  private readonly hyper: AdamHyper;
  private readonly init?: SplatInit;

  static async create(
    device: GPUDevice,
    trainPlan: TrainPlan,
    weights: Float32Array,
    cfg: SplatOptimizerConfig = {}
  ): Promise<SplatOptimizer> {
    const [ic, ih, iw] = trainPlan.inputShape;
    if (ic !== 3 || ih !== SIDE || iw !== SIDE) {
      throw new Error(
        `optimize: CLIP inputShape [${ic},${ih},${iw}] != [3,${SIDE},${SIDE}] — ` +
          `the raster→CLIP copy assumes matching NCHW dims`
      );
    }
    const G = cfg.G ?? LEGIBLE_G;
    const cap = cfg.cap ?? 2048;
    const raster = await RasterEngine.create(device, {
      H: SIDE,
      W: SIDE,
      G,
      cap,
      bg: cfg.bg ?? [0.5, 0.5, 0.5],
    });
    const trainer = await VisionTrainer.create(device, trainPlan, weights);
    raster.setParams(cfg.initParams ?? randomSplats(G, cfg.seed ?? 1, cfg.init));
    raster.zeroAdamState();
    return new SplatOptimizer(device, raster, trainer, cfg);
  }

  private constructor(
    device: GPUDevice,
    raster: RasterEngine,
    trainer: VisionTrainer,
    cfg: SplatOptimizerConfig
  ) {
    this.device = device;
    this.raster = raster;
    this.trainer = trainer;
    this.lrs = cfg.lrs ?? LEGIBLE_LRS;
    this.hyper = cfg.hyper ?? DEFAULT_HYPER;
    this.init = cfg.init;
  }

  /** Target text embedding (raw, un-normalized — the −cos loss normalizes it).
   *  Call on every prompt change; cheap (a 2 KB buffer write). */
  setPrompt(textEmbed: Float32Array): void {
    this.trainer.writeText(textEmbed);
  }

  /** One optimization step: forward → CLIP loss → backward → Adam, ONE submit. */
  step(): void {
    const enc = this.device.createCommandEncoder();
    this.raster.recordForward(enc);
    enc.copyBufferToBuffer(this.raster.image, 0, this.trainer.inputBuffer, 0, IMG_BYTES);
    this.trainer.encode(enc, { backward: true });
    enc.copyBufferToBuffer(this.trainer.inputGradBuffer, 0, this.raster.gradImage, 0, IMG_BYTES);
    this.raster.recordBackward(enc);
    this.step_ += 1; // Adam bias-correction is 1-based
    this.raster.recordAdam(enc, this.step_, this.lrs, this.hyper);
    this.device.queue.submit([enc.finish()]);
  }

  get stepCount(): number {
    return this.step_;
  }

  /** Partial re-randomization of the current splats. Unlike Reset, this keeps
   *  the optimizer, CLIP resources, prompt, step count, and Adam buffers alive. */
  async nudge(opts: SplatNudgeOptions = {}): Promise<void> {
    const G = this.raster.dims.G;
    const params = await this.raster.readParams();
    nudgeSplats(params, G, opts.seed ?? Date.now(), opts.amount ?? DEFAULT_NUDGE_AMOUNT, opts.init ?? this.init);
    this.raster.setParams(params);
  }

  /** Render the current splats without training; leaves the image on the GPU
   *  and returns it (NCHW planar [3][256][256]) for display / metrics. */
  async renderImage(): Promise<Float32Array> {
    this.raster.runForward();
    return this.raster.readImage();
  }

  /** CLIP embedding of the current splat image (forward-only). The page can use
   *  this to show live cosine similarity to the prompt; the test uses it to
   *  prove the loss decreases. */
  async currentEmbedding(): Promise<Float32Array> {
    const enc = this.device.createCommandEncoder();
    this.raster.recordForward(enc);
    enc.copyBufferToBuffer(this.raster.image, 0, this.trainer.inputBuffer, 0, IMG_BYTES);
    this.trainer.encode(enc, { backward: false });
    this.device.queue.submit([enc.finish()]);
    return readFloats(this.device, this.trainer.outputBuffer, this.trainer.plan.embedDim);
  }

  destroy(): void {
    this.raster.destroy();
  }
}

/** cos(a, b) — the metric the page shows and the test gates on. */
export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb || 1);
}

// ---------------------------------------------------------------------------
// Deterministic random splat init (browser-safe: no node imports). Conventional
// 2D-splat start — spread over the canvas, small translucent Gaussians, mid
// colours the optimizer pushes around. SoA layout matches raster_wgsl.ts:
// [mean 2G][logScale 2G][theta G][colorRaw 3G][opacityRaw G], per-splat
// interleaved within each segment.
// ---------------------------------------------------------------------------
export function randomSplats(G: number, seed = 1, init: SplatInit = {}): Float32Array {
  const scale = init.scale ?? LEGIBLE_INIT.scale;
  const scaleJitter = init.scaleJitter ?? LEGIBLE_INIT.scaleJitter;
  const opacityRaw = init.opacityRaw ?? LEGIBLE_INIT.opacityRaw;
  const colorSpread = init.colorSpread ?? LEGIBLE_INIT.colorSpread;
  let state = (seed >>> 0) || 1;
  const next = (): number => {
    state = (Math.imul(state, 747796405) + 2891336453) >>> 0;
    let t = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
    t = ((t >>> 22) ^ t) >>> 0;
    return t / 4294967296;
  };
  const normal = (): number => {
    let u = 0;
    let v = 0;
    while (u === 0) u = next();
    while (v === 0) v = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const p = new Float32Array(G * PARAM_STRIDE);
  const meanOff = 0;
  const lsOff = 2 * G;
  const thetaOff = 4 * G;
  const colOff = 5 * G;
  const opOff = 8 * G;
  const lnScale = Math.log(scale);
  for (let g = 0; g < G; g++) {
    p[meanOff + g * 2 + 0] = next() * SIDE;
    p[meanOff + g * 2 + 1] = next() * SIDE;
    p[lsOff + g * 2 + 0] = lnScale + scaleJitter * normal();
    p[lsOff + g * 2 + 1] = lnScale + scaleJitter * normal();
    p[thetaOff + g] = next() * Math.PI * 2;
    p[colOff + g * 3 + 0] = colorSpread * normal();
    p[colOff + g * 3 + 1] = colorSpread * normal();
    p[colOff + g * 3 + 2] = colorSpread * normal();
    p[opOff + g] = opacityRaw;
  }
  return p;
}

export function nudgeSplats(
  params: Float32Array,
  G: number,
  seed = 1,
  amount = DEFAULT_NUDGE_AMOUNT,
  init: SplatInit = {}
): Float32Array {
  if (params.length !== G * PARAM_STRIDE) throw new Error("nudgeSplats: wrong param length");
  const t = clamp(amount, 0, 1);
  if (t === 0) return params;
  const fresh = randomSplats(G, seed, init);
  const meanOff = 0;
  const lsOff = 2 * G;
  const thetaOff = 4 * G;
  const colOff = 5 * G;
  const opOff = 8 * G;
  const logMin = Math.log(0.3);
  const logMax = Math.log(64);

  for (let g = 0; g < G; g++) {
    const mi = meanOff + g * 2;
    params[mi + 0] = clamp(lerp(params[mi + 0], fresh[mi + 0], t), 0, SIDE);
    params[mi + 1] = clamp(lerp(params[mi + 1], fresh[mi + 1], t), 0, SIDE);

    const li = lsOff + g * 2;
    params[li + 0] = clamp(lerp(params[li + 0], fresh[li + 0], t), logMin, logMax);
    params[li + 1] = clamp(lerp(params[li + 1], fresh[li + 1], t), logMin, logMax);

    const ti = thetaOff + g;
    params[ti] = blendAngle(params[ti], fresh[ti], t);

    const ci = colOff + g * 3;
    params[ci + 0] = lerp(params[ci + 0], fresh[ci + 0], t);
    params[ci + 1] = lerp(params[ci + 1], fresh[ci + 1], t);
    params[ci + 2] = lerp(params[ci + 2], fresh[ci + 2], t);

    const oi = opOff + g;
    params[oi] = lerp(params[oi], fresh[oi], t);
  }
  return params;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function blendAngle(a: number, b: number, t: number): number {
  const tau = Math.PI * 2;
  const d = ((((b - a) + Math.PI) % tau) + tau) % tau - Math.PI;
  return a + d * t;
}

// small readback helper (kept local — RasterEngine's is private, and the CLIP
// output buffer isn't one of RasterEngine's).
async function readFloats(device: GPUDevice, buf: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: 1 /*MAP_READ*/ | 8 /*COPY_DST*/ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}
