/**
 * Neural Force Field Art — Gallery Engine
 *
 * Core algorithm:
 *   1. Neural network predicts force vectors from particle positions
 *   2. Forces are applied as acceleration to particles (velocity + position update)
 *   3. Loss is computed on the resulting positions (spiral distance, center distance)
 *   4. Gradients flow back through the entire physics chain to update model weights
 *   5. Random reset keeps particles exploring instead of collapsing
 *
 * The model DISCOVERS how to move particles creatively while minimising a
 * simple constraint — it is NOT told the answer directly.
 */
import * as tf from "@tensorflow/tfjs";
// Registers the 'webgpu' backend with tfjs. The @tensorflow/tfjs union package
// ships only cpu + webgl backends; without this import tf.setBackend('webgpu')
// throws "Backend name 'webgpu' not found in registry".
import "@tensorflow/tfjs-backend-webgpu";
import { createRenderer, RendererType } from "./renderers";
import { ForceField, HelmholtzField } from "./core/field/helmholtz";
import {
  isotropyLoss,
  divergencePenalty,
  chaosLoss,
} from "./core/losses";
// OPTIONAL zero-copy GPU renderer (perf lane). Imported so it compiles and is
// ready, but only used when a preset sets `gpu: true` (none do by default — it
// needs browser QA). See src/render/gpuPoints.ts.
import { GpuPointRenderer } from "./render/gpuPoints";
import { GpuPointRendererWebGPU } from "./render/webgpu/points";
import { SplatRenderer, SplatStyle } from "./render/webgpu/splat";
import { AdvectKernel } from "./render/webgpu/advect";
import { FusedTrainer } from "./render/webgpu/train";
import { GpuTimer } from "./render/webgpu/gputime";

// tfjs-backend-webgpu 4.10 calls adapter.requestAdapterInfo(), which current
// Chrome removed in favour of the synchronous `adapter.info` property — without
// this shim the webgpu backend fails to init ("requestAdapterInfo is not a
// function"). Safe no-op where WebGPU is absent (GPUAdapter undefined).
{
  const GA = (globalThis as any).GPUAdapter;
  if (GA && !GA.prototype.requestAdapterInfo) {
    GA.prototype.requestAdapterInfo = function () {
      return Promise.resolve((this as any).info ?? {});
    };
  }
  // tfjs also OWNS GPUDevice creation and requests only the features it wants
  // for itself — but device features are fixed at creation time, so anything
  // OUR kernels need must be appended to tfjs's requestDevice call. Wrap it to
  // add "shader-f16" (advect kernel's f16 fast path, see render/webgpu/
  // advect.ts) and "timestamp-query" (upcoming GPU profiling), each only when
  // the adapter reports it, deduped, preserving every other descriptor field.
  if (GA && !(GA.prototype as any).__nffaRequestDeviceWrapped) {
    (GA.prototype as any).__nffaRequestDeviceWrapped = true;
    const origRequestDevice = GA.prototype.requestDevice;
    GA.prototype.requestDevice = function (
      desc?: GPUDeviceDescriptor
    ): Promise<GPUDevice> {
      const extras = ["shader-f16", "timestamp-query"].filter((f) =>
        (this as GPUAdapter).features.has(f as GPUFeatureName)
      ) as GPUFeatureName[];
      const requiredFeatures = [
        ...new Set([...(desc?.requiredFeatures ?? []), ...extras]),
      ];
      return origRequestDevice.call(this, { ...desc, requiredFeatures });
    };
  }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
/**
 * Extra context handed to {@link ArtPieceConfig.computeLoss} on each step.
 * `force` is the per-step force tensor `[N,2]` (already scaled by
 * `forceMagnitude`) that produced the current positions — reused so
 * force-based losses (isotropy) do not recompute the field. `field`, when
 * present, is the live {@link ForceField} so field-sampling losses (chaos,
 * divergence) can probe it at arbitrary positions. Both are `undefined` for
 * the legacy MLP pieces, which ignore this argument entirely.
 */
export interface LossContext {
  force?: tf.Tensor2D;
  field?: ForceField | null;
}

export interface ArtPieceConfig {
  name: string;
  particleCount: number;
  friction: number;
  forceMagnitude: number;
  maxVelocity: number;
  resetRate: number;
  drawRate: number;
  learningRate: number;
  backgroundColor: [number, number, number];
  alphaBlend: number;
  renderer: RendererType;
  /**
   * Legacy path: a sigmoid MLP whose `[0,1]` output is re-centered by
   * `(raw - 0.5)`. Mutually exclusive with {@link createField}.
   */
  createModel?: () => tf.Sequential;
  /**
   * Field path: a {@link ForceField} (e.g. {@link HelmholtzField}) whose raw
   * signed output is used directly (NO `-0.5` shift). Its `trainableWeights`
   * become the optimizer varList. Takes precedence over {@link createModel}.
   */
  createField?: () => ForceField;
  /**
   * OPTIONAL: route rendering through the zero-copy {@link GpuPointRenderer}
   * instead of the Canvas2D renderers. Off by default (needs browser QA); no
   * shipped preset sets it.
   */
  gpu?: boolean;
  computeLoss: (
    pos: tf.Tensor2D,
    w: number,
    h: number,
    ctx?: LossContext
  ) => tf.Scalar;
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------
async function initBackend(preferred?: string) {
  // WHY cpu-first: the model is TINY (2->..->2) evaluated per-point over a big
  // batch. That workload is DISPATCH-bound, not compute-bound — on webgl/webgpu
  // every op is a separate GPU kernel dispatch, and the per-frame arraySync()
  // forces a GPU->CPU readback (a full pipeline flush on webgpu). CPU runs the
  // tiny matmuls inline with zero dispatch/readback and is dramatically faster
  // here (measured: webgpu ~3-15 FPS vs cpu many×). Override with ?backend=webgl.
  const override =
    new URLSearchParams(location.search).get("backend") || preferred;
  const order = override
    ? [override, "cpu", "webgl", "webgpu"]
    : ["cpu", "webgl", "webgpu"];
  for (const b of order) {
    try {
      if (await tf.setBackend(b)) {
        await tf.ready();
        console.log(`TF.js backend: ${tf.getBackend()}`);
        return;
      }
    } catch (_) {}
  }
  throw new Error("No TF.js backend available");
}

// ---------------------------------------------------------------------------
// Model factories
// ---------------------------------------------------------------------------
function mlpShallow(): tf.Sequential {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 32, activation: "selu", inputShape: [2] }));
  m.add(tf.layers.dense({ units: 64, activation: "selu" }));
  m.add(tf.layers.dense({ units: 2, activation: "sigmoid" }));
  return m;
}

function mlpDeep(): tf.Sequential {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 64, activation: "selu", inputShape: [2] }));
  m.add(tf.layers.dense({ units: 128, activation: "selu" }));
  m.add(tf.layers.dense({ units: 128, activation: "selu" }));
  m.add(tf.layers.dense({ units: 64, activation: "selu" }));
  m.add(tf.layers.dense({ units: 2, activation: "sigmoid" }));
  return m;
}

function mlpWide(): tf.Sequential {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 256, activation: "selu", inputShape: [2] }));
  m.add(tf.layers.dense({ units: 2, activation: "sigmoid" }));
  return m;
}

// ---------------------------------------------------------------------------
// Loss functions  (all differentiable through the physics chain)
// ---------------------------------------------------------------------------
const SPIRAL_TURNS = 3;
const SPIRAL_MAX_THETA = SPIRAL_TURNS * 2 * Math.PI;

function spiralLoss(pos: tf.Tensor2D, w: number, h: number): tf.Scalar {
  return tf.tidy(() => {
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(w, h) * 0.38;
    const b = maxR / SPIRAL_MAX_THETA;

    const dx = pos.slice([0, 0], [-1, 1]).sub(cx);
    const dy = pos.slice([0, 1], [-1, 1]).sub(cy);
    const r = dx.square().add(dy.square()).add(1e-4).sqrt();
    const phi = tf.atan2(dy, dx);

    let best = tf.fill(r.shape, 1e8) as tf.Tensor;
    for (let k = 0; k <= SPIRAL_TURNS + 1; k++) {
      const theta = phi.add(2 * Math.PI * k);
      const rSpiral = theta.relu().mul(b);
      best = tf.minimum(best, r.sub(rSpiral).square());
    }
    return best.mean().asScalar();
  });
}

function centerLoss(pos: tf.Tensor2D, w: number, h: number): tf.Scalar {
  return tf.tidy(() => {
    const center = tf.tensor2d([[w / 2, h / 2]]);
    return pos.sub(center).square().sum(1).mean().asScalar();
  });
}

function spiralPlusCenterLoss(
  centerWeight: number
): (pos: tf.Tensor2D, w: number, h: number) => tf.Scalar {
  return (pos, w, h) =>
    tf.tidy(() => {
      const sL = spiralLoss(pos, w, h);
      const cL = centerLoss(pos, w, h);
      return sL.add(cL.mul(centerWeight)).asScalar();
    });
}

/**
 * Composite loss for the Helmholtz field piece: mostly chaos + isotropy, held
 * together by a faint spiral so the mixing has a ghost of structure.
 *
 *   loss = W_CHAOS·chaos + W_ISO·isotropy + W_DIV·divergence + W_SPIRAL·spiral
 *
 * - chaos      maximises local sensitivity (returns −log separation).
 * - isotropy   keeps force energy directionally balanced (reads the SAME
 *              per-step force tensor via {@link LossContext.force}).
 * - divergence lightly pins the flow toward area-preserving.
 * - spiral     a tiny structural anchor (pixel-scale MSE → small weight).
 *
 * Requires `ctx.field` and `ctx.force` (the field piece always supplies them).
 * Weights are artistic knobs — tune freely.
 */
function helmholtzChaosLoss(): ArtPieceConfig["computeLoss"] {
  const W_CHAOS = 1.0;
  const W_ISO = 1.0;
  const W_DIV = 0.5;
  const W_SPIRAL = 0.00002;
  const HH = 1e-2; // finite-diff / jitter step (normalized coords)
  return (pos, w, h, ctx) =>
    tf.tidy(() => {
      const field = ctx!.field!;
      const force = ctx!.force!;
      const posNorm = pos.div(tf.tensor2d([[w, h]])) as tf.Tensor2D;

      // PERF: evaluate the field just 3× — a shared centre f0 plus one +x and one
      // +y neighbour — and reuse them for BOTH the chaos (Lyapunov) and the
      // divergence terms. (Was 7 evals: chaos 2 + divergence 4 + physics 1, each
      // = 2 MLP heads. On the dispatch-bound webgpu backend those tiny ops are
      // the dominant cost of this piece; sharing the centre roughly halves it.)
      // SINGLE (sharded) forward: batch the 3 sample sets — centre, +x, +y —
      // into one [3N,2] tensor so the field runs ONCE (1 set of GPU dispatches
      // instead of 3), then slice. Same math, ~1/3 the dispatch overhead, still
      // one backward pass over the whole graph (first-order autograd).
      const N = posNorm.shape[0];
      const allPos = tf.concat(
        [
          posNorm,
          posNorm.add(tf.tensor2d([[HH, 0]])),
          posNorm.add(tf.tensor2d([[0, HH]])),
        ],
        0
      ) as tf.Tensor2D;
      const allF = field.forces(allPos);
      const f0 = allF.slice([0, 0], [N, -1]);
      const fx = allF.slice([N, 0], [N, -1]);
      const fy = allF.slice([2 * N, 0], [N, -1]);

      // chaos: local sensitivity — how much F changes for a small +x/+y nudge.
      const sepx = fx.sub(f0).square().sum(1);
      const sepy = fy.sub(f0).square().sum(1);
      const sep = sepx.add(sepy).add(1e-12).sqrt().div(HH * 1.4142 + 1e-9);
      const chaos = sep.add(1e-6).log().mean().neg();

      // forward-difference divergence sharing the centre: ∂Fx/∂x + ∂Fy/∂y.
      const dFxdx = fx.slice([0, 0], [-1, 1]).sub(f0.slice([0, 0], [-1, 1])).div(HH);
      const dFydy = fy.slice([0, 1], [-1, 1]).sub(f0.slice([0, 1], [-1, 1])).div(HH);
      const div = dFxdx.add(dFydy).square().mean();

      const iso = isotropyLoss(force);
      const spiral = spiralLoss(pos, w, h);

      return chaos
        .mul(W_CHAOS)
        .add(iso.mul(W_ISO))
        .add(div.mul(W_DIV))
        .add(spiral.mul(W_SPIRAL))
        .asScalar();
    });
}

// ---------------------------------------------------------------------------
// Gallery
// ---------------------------------------------------------------------------
export const GALLERY: ArtPieceConfig[] = [
  {
    name: "Spiral · Ghost",
    particleCount: 1000,
    friction: 0.985,
    forceMagnitude: 3.0,
    maxVelocity: 22,
    resetRate: 0.012,
    drawRate: 2,
    learningRate: 0.01,
    backgroundColor: [12, 0, 34],
    alphaBlend: 0.06,
    renderer: "alpha-fade",
    createModel: mlpShallow,
    computeLoss: spiralPlusCenterLoss(0.00005),
  },
  {
    name: "Spiral · Trails",
    particleCount: 800,
    friction: 0.99,
    forceMagnitude: 3.5,
    maxVelocity: 28,
    resetRate: 0.008,
    drawRate: 2,
    learningRate: 0.008,
    backgroundColor: [4, 0, 18],
    alphaBlend: 0.04,
    renderer: "trail-buffer",
    createModel: mlpDeep,
    computeLoss: spiralPlusCenterLoss(0.0001),
  },
  {
    name: "Vortex · Ghost",
    particleCount: 1200,
    friction: 0.985,
    forceMagnitude: 3.0,
    maxVelocity: 20,
    resetRate: 0.015,
    drawRate: 1,
    learningRate: 0.01,
    backgroundColor: [12, 0, 34],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    createModel: mlpShallow,
    computeLoss: (p, w, h) =>
      tf.tidy(() => centerLoss(p, w, h).mul(0.001).asScalar()),
  },
  {
    name: "Galaxy · Clean",
    particleCount: 1500,
    friction: 0.975,
    forceMagnitude: 4.0,
    maxVelocity: 30,
    resetRate: 0.01,
    drawRate: 3,
    learningRate: 0.005,
    backgroundColor: [2, 0, 12],
    alphaBlend: 0.03,
    renderer: "clean",
    createModel: mlpWide,
    computeLoss: spiralPlusCenterLoss(0.00002),
  },
  {
    name: "Galaxy · Ghost",
    particleCount: 1500,
    friction: 0.975,
    forceMagnitude: 4.0,
    maxVelocity: 30,
    resetRate: 0.01,
    drawRate: 3,
    learningRate: 0.005,
    backgroundColor: [2, 0, 12],
    alphaBlend: 0.03,
    renderer: "alpha-fade",
    createModel: mlpWide,
    computeLoss: spiralPlusCenterLoss(0.00002),
  },
  {
    // Helmholtz-decomposed field driven by a GAN-style chaos objective. The
    // alpha slider (see index.tsx) slides the field between order (∇φ) and
    // chaos (curl ψ) live; alpha starts biased toward mixing.
    // particleCount is 200k (slider to 1M): advection is a single fused WGSL
    // dispatch (see render/webgpu/advect.ts), so count no longer gates FPS.
    name: "Helmholtz · Chaos",
    particleCount: 200000,
    friction: 0.99,
    forceMagnitude: 3.5,
    maxVelocity: 26,
    resetRate: 0.01,
    drawRate: 2,
    learningRate: 0.01,
    backgroundColor: [6, 2, 20],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    createField: () => new HelmholtzField({ alpha: 0.7 }),
    computeLoss: helmholtzChaosLoss(),
  },
  {
    // MULTI-SPECIES: 3 particle classes. The chaos lane r(pos, onehot(class))
    // mixes each species differently; the order lane g(pos) and the isotropy
    // pressure are class-blind. Class = hash of particle index (stable,
    // storage-free); renderer colours by species. FUSED-ONLY (tfjs has no
    // class input — ?train=tfjs is ignored for this piece).
    name: "Helmholtz · Species",
    particleCount: 200000,
    friction: 0.99,
    forceMagnitude: 3.5,
    maxVelocity: 26,
    resetRate: 0.01,
    drawRate: 2,
    learningRate: 0.01,
    backgroundColor: [4, 2, 16],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    createField: () => new HelmholtzField({ alpha: 0.7, classes: 3 }),
    computeLoss: helmholtzChaosLoss(),
  },
  {
    // SIREN field — sinusoidal activations (Sitzmann et al.). Smooth, well-
    // defined spatial derivatives (which the chaos/divergence probe losses
    // measure), and higher-frequency structure than SELU. A SELECTABLE model
    // type to compare against the standard field (same loss, same knobs).
    // Trains FUSED (sin backward checkpoints pre-acts); advects fused too.
    name: "Helmholtz · SIREN",
    particleCount: 200000,
    friction: 0.99,
    forceMagnitude: 3.5,
    maxVelocity: 26,
    resetRate: 0.01,
    drawRate: 2,
    learningRate: 0.005,
    backgroundColor: [6, 2, 20],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    createField: () =>
      new HelmholtzField({ alpha: 0.7, modelType: "siren", sirenOmega0: 6 }),
    computeLoss: helmholtzChaosLoss(),
  },
  {
    // FOURIER-feature field — γ(p) = [x,y,sin/cos(ωk·x/y)] input encoding beats
    // the plain MLP's spectral bias, exposing fine "filigree" spatial detail.
    // ω0/octaves are the artistic dial. Trains FUSED (encoding jacobian reuses
    // the stored sin/cos features); advects fused. SELECTABLE comparison type.
    name: "Helmholtz · Fourier",
    particleCount: 200000,
    friction: 0.99,
    forceMagnitude: 3.5,
    maxVelocity: 26,
    resetRate: 0.01,
    drawRate: 2,
    learningRate: 0.008,
    backgroundColor: [2, 4, 14],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    createField: () =>
      new HelmholtzField({ alpha: 0.7, modelType: "fourier", fourierOctaves: 4 }),
    computeLoss: helmholtzChaosLoss(),
  },
  {
    // HASH-GRID field (Instant-NGP essence, single-level): a learned feature
    // grid (32×32×4) is bilinearly interpolated → tiny MLP. Best detail-per-
    // FLOP; the field reads more "painted texture" than "equation". Trains
    // FUSED (pass B scatters into grid cells); advects via the fused
    // grid-interp kernel. SELECTABLE comparison type.
    name: "Helmholtz · HashGrid",
    particleCount: 200000,
    friction: 0.99,
    forceMagnitude: 3.5,
    maxVelocity: 26,
    resetRate: 0.01,
    drawRate: 2,
    learningRate: 0.01,
    backgroundColor: [10, 4, 14],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    createField: () =>
      new HelmholtzField({
        alpha: 0.7,
        modelType: "hashgrid",
        gridSize: 32,
        gridFeatures: 4,
      }),
    computeLoss: helmholtzChaosLoss(),
  },
];

// ---------------------------------------------------------------------------
// Physics step (inside optimizer.minimize — gradients flow through)
// ---------------------------------------------------------------------------
function physicsForward(
  pos: tf.Tensor2D,
  vel: tf.Tensor2D,
  model: tf.Sequential | null,
  field: ForceField | null,
  cfg: ArtPieceConfig,
  w: number,
  h: number
): { newPos: tf.Tensor2D; newVel: tf.Tensor2D; force: tf.Tensor2D } {
  const posNorm = pos.div(tf.tensor2d([[w, h]])) as tf.Tensor2D;
  // Field path: raw signed output used directly (NO -0.5 shift).
  // MLP path (legacy): sigmoid output re-centered by (raw - 0.5).
  const forces = field
    ? (field.forces(posNorm).mul(cfg.forceMagnitude) as tf.Tensor2D)
    : ((model!.predict(posNorm) as tf.Tensor2D).sub(0.5).mul(cfg.forceMagnitude) as tf.Tensor2D);

  // PERF: clip/wrap the WHOLE [N,2] tensor in one op each instead of
  // slice-x/clip, slice-y/clip, concat (5 ops -> 1) and likewise for the wrap.
  // maxVelocity is symmetric so a single clipByValue covers both axes; mod
  // broadcasts a [1,2] to wrap x by w and y by h at once. ~8 fewer GPU
  // dispatches per call, and this runs twice a frame (learn + advect).
  const clippedVel = vel
    .add(forces)
    .mul(cfg.friction)
    .clipByValue(-cfg.maxVelocity, cfg.maxVelocity) as tf.Tensor2D;

  const wrappedPos = pos
    .add(clippedVel)
    .mod(tf.tensor2d([[w, h]])) as tf.Tensor2D;

  return { newPos: wrappedPos, newVel: clippedVel, force: forces };
}

// Random reset now lives INSIDE the fused advect kernel (PCG hash per
// particle+frame) — see src/render/webgpu/advect_wgsl.ts. The old tfjs
// randomReset (~10 dispatches/frame) is gone with the rest of the tfjs
// advect stage.

// Renderer is now in src/renderers.ts — three implementations:
//   "alpha-fade"   — dual-buffer ghost trails (fast, hardware-composited)
//   "trail-buffer" — ring buffer clean trails (precise, no ghosts)
//   "clean"        — no trails (fastest, debug/iteration)

// ---------------------------------------------------------------------------
// Generate spiral target points in pixel coords (for overlay)
// ---------------------------------------------------------------------------
function spiralPixelPoints(w: number, h: number, n = 600): number[][] {
  const cx = w / 2;
  const cy = h / 2;
  const maxR = Math.min(w, h) * 0.38;
  const b = maxR / SPIRAL_MAX_THETA;
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const theta = (i / n) * SPIRAL_MAX_THETA;
    const r = b * theta;
    pts.push([cx + r * Math.cos(theta), cy + r * Math.sin(theta)]);
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Main simulation loop
// ---------------------------------------------------------------------------
/** Full-screen "needs WebGPU" notice. There is NO Canvas2D/WebGL fallback — by
 *  design (we're WebGPU-only). Shown when the browser has no WebGPU. */
function showWebGPUWarning(): void {
  document.documentElement.style.margin = "0";
  document.body.style.margin = "0";
  const o = document.createElement("div");
  o.style.cssText =
    "position:fixed;inset:0;z-index:10000;display:flex;align-items:center;" +
    "justify-content:center;background:#05010f;color:#cbd5ff;text-align:center;" +
    "font:16px/1.6 ui-monospace,monospace;padding:24px";
  o.innerHTML =
    '<div style="max-width:560px">' +
    '<div style="font-size:44px;margin-bottom:12px">⚡</div>' +
    '<div style="font-size:20px;margin-bottom:10px;color:#fff">This needs WebGPU</div>' +
    '<div style="margin-bottom:16px;color:#94a0c8">Neural Force Field Art runs ' +
    "entirely on the GPU (zero-copy tfjs → WebGPU). Your browser doesn't have " +
    "WebGPU enabled.</div>" +
    '<div><a href="https://caniuse.com/webgpu" target="_blank" ' +
    'style="color:#8ab4ff">Go get WebGPU working →</a> ' +
    '<span style="color:#5b6890">(Chrome / Edge / Safari 18+ / Firefox, latest)</span></div>' +
    "</div>";
  document.body.appendChild(o);
}

/**
 * Per-piece trail defaults for the splat renderer, keyed by the piece's
 * declared renderer style (the splat's decay is now the one real trail
 * mechanism behind all three looks): ghost pieces get soft trails,
 * trail-buffer pieces long streaks, clean pieces none. `?decay=F` overrides.
 */
const SPLAT_DECAY_BY_RENDERER: Record<RendererType, number> = {
  "alpha-fade": 0.94,
  "trail-buffer": 0.97,
  clean: 0,
};

export interface LoopHandle {
  field: HelmholtzField | null;
  getParticleCount(): number;
  setParticleCount(n: number): void;
  getSampleRate(): number;
  setSampleRate(n: number): void;
  /** Live respawn fraction — with particle-sourced training this is also the
   *  exploration dial (resets feed fresh uniform states into the batch). */
  getResetRate(): number;
  setResetRate(r: number): void;
  /** Splat trail persistence (0 = hard clear … 0.99 = long streaks) — wired
   *  straight to SplatRenderer.decay, the "trails" slider's backing knob. */
  getDecay(): number;
  setDecay(d: number): void;
}

export function startLoop(
  canvas: HTMLCanvasElement,
  configIndex: number,
  onReady?: (handle: LoopHandle) => void
): () => void {
  let running = true;
  const cfg = GALLERY[configIndex];
  let particleCount = cfg.particleCount; // rendered/advected particles (live)
  let sampleRate = 256; // points the field trains on per frame (live)
  let resetRate = cfg.resetRate; // respawn fraction (live — see setResetRate)
  // `?window=K` (1..16): trajectory-window training — sets BOTH rollout=K and
  // trainEvery=K. tools/window_test.ts proves the K-step imagined rollout from
  // live particle states IS the next K real frames (maxΔ ≈ 6e-5 px at K=6), so
  // this is true window training with zero recording machinery. 0 = not set.
  const windowK = Math.max(
    0,
    Math.min(
      16,
      parseInt(new URLSearchParams(location.search).get("window") ?? "0", 10) || 0
    )
  );
  // `?trainEvery=N` (default 1): run the fused train step every Nth frame.
  const trainEvery =
    windowK > 0
      ? windowK
      : Math.max(
          1,
          parseInt(new URLSearchParams(location.search).get("trainEvery") ?? "1", 10) || 1
        );

  // WebGPU-only — no Canvas2D/WebGL fallback (by design). Warn + bail if absent.
  if (!GpuPointRendererWebGPU.isSupported()) {
    showWebGPUWarning();
    return () => {};
  }

  // RETINA: the canvas BACKING store is native resolution (devicePixelRatio,
  // capped at 2 — 3x phone DPRs quadruple the accumulator for little gain)
  // while the physics WORLD stays w×h CSS pixels: kernels, trainer and losses
  // are untouched; only the backing store and the splat accumulator scale.
  // Math.ceil keeps canvas.width EXACTLY equal to the splat renderer's native
  // accumulator width (its tonemap indexes the accumulator by fragment coord —
  // a row-stride mismatch would shear the image). The quad renderer needs no
  // change: its vertex math is pos/resolution-relative (incl. pointSize, which
  // is in world units), and its fwidth AA sharpens automatically at native res.
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = window.innerWidth;
  const h = window.innerHeight;
  canvas.width = Math.ceil(w * dpr);
  canvas.height = Math.ceil(h * dpr);

  // Full-screen, no scroll (canvas is inline by default -> descender gap).
  // Explicit CSS width/height pin the element to CSS pixels so the native
  // backing maps 1:1 onto device pixels.
  canvas.style.cssText =
    `display:block;position:fixed;inset:0;width:${w}px;height:${h}px`;
  document.documentElement.style.margin = "0";
  document.body.style.cssText = "margin:0;overflow:hidden;background:#000";

  // ALL tensor/model creation is DEFERRED to the async init below: tfjs throws
  // if you build tensors/models before the highest-priority backend (webgpu)
  // has finished initializing (needs await tf.ready()). Assigned there.
  let field: ForceField | null = null;
  let model: tf.Sequential | null = null;
  let varList: tf.Variable[] | undefined = undefined;
  let optimizer: tf.Optimizer | null = null;
  // Particle state lives in the fused kernel's GPUBuffers, NOT tfjs tensors —
  // training samples random points, so tfjs never touches the particles.
  let advect: AdvectKernel | null = null;
  // Fused trainer (field pieces): analytic backward + Adam in 2 WGSL
  // dispatches, updating the advect kernel's weights buffer IN PLACE — the
  // whole hot path is then GPU-only, tfjs idle. Gradients verified against
  // tfjs autograd (cos=1.0000000) by tools/train_test.ts. `?train=tfjs`
  // falls back to the tfjs optimizer path for A/B comparison.
  let trainer: FusedTrainer | null = null;
  let trainSource: "particles" | "random" = "particles";
  let mixRandom = 0;
  let hudLoss = NaN;
  // The shared GPUDevice (tfjs's) and the optional GPU profiler. Assigned in
  // the async init once the webgpu backend is confirmed; timer is null when the
  // adapter lacks "timestamp-query" (→ HUD falls back to CPU-encode lines).
  let device: GPUDevice | null = null;
  let timer: GpuTimer | null = null;
  let wh: tf.Tensor2D | null = null;
  let renderer: GpuPointRendererWebGPU | null = null;
  // Compute-splat renderer (accumulation buffer + tonemap): ~5 ms at 1M vs
  // ~25 ms for instanced quads (4M verts + additive overdraw). The DEFAULT
  // path at every count — its radial cone kernel gives round dots at low N
  // too (the old 2x2 bilinear looked square, which is why quads used to win
  // below 20k), and its decay gives real ghost trails. `?render=quads`
  // restores the quad path; `?decay=F` overrides decay; `?dot=F` sets the
  // splat radius in CSS px.
  let splat: SplatRenderer | null = null;
  const renderOverride = new URLSearchParams(location.search).get("render");
  const SPLAT_MIN_N = 0;
  const exposureScale =
    parseFloat(new URLSearchParams(location.search).get("exposure") ?? "1") || 1;
  // `?stroke=dot|vel|curl` — splat draw style (default dot, the shipped look).
  // "vel"/"curl" draw per-frame geometric strokes along each particle's
  // backward trajectory, so fast particles (maxVelocity ~26 px/frame vs a
  // ~1.6px dot) read as continuous filaments instead of disconnected dots.
  const strokeParam = new URLSearchParams(location.search).get("stroke");
  const strokeStyle: SplatStyle =
    strokeParam === "vel" || strokeParam === "curl" ? strokeParam : "dot";
  // `?strokeLen=F` — stroke length in FRAMES of travel (default 3, [0.5, 16]).
  // Number.isFinite (not `|| 3`) so an explicit `?strokeLen=0` clamps to the
  // documented 0.5 floor instead of silently becoming the default.
  const strokeLenParam = new URLSearchParams(location.search).get("strokeLen");
  const strokeLenParsed = strokeLenParam !== null ? parseFloat(strokeLenParam) : NaN;
  const strokeLen = Number.isFinite(strokeLenParsed)
    ? Math.max(0.5, Math.min(16, strokeLenParsed))
    : 3;
  // Stroke-style segmented control (dot|vel|curl) — created next to the
  // trails bar once the splat renderer exists; removed in the cleanup closure.
  let strokeUi: HTMLDivElement | null = null;

  let frame = 0;

  // --- Telemetry HUD: FPS + per-stage timing so the bottleneck is visible ----
  const tele = document.createElement("div");
  tele.style.cssText =
    "position:fixed;top:8px;right:8px;z-index:9999;font:11px/1.45 ui-monospace," +
    "monospace;color:#8f8;background:rgba(0,0,0,.6);padding:6px 9px;border-radius:" +
    "5px;white-space:pre;pointer-events:none;letter-spacing:.02em";
  document.body.appendChild(tele);
  const ema = (prev: number, x: number, a = 0.12) =>
    prev === 0 ? x : prev * (1 - a) + x * a;
  let emaFrame = 0,
    emaTrain = 0,
    emaRender = 0,
    lastT = performance.now();

  // Learning is DECOUPLED from motion: we train the field on a small random
  // batch each frame (real-time, cheap) while the FUSED WGSL KERNEL advects
  // ALL particles in ONE compute dispatch (MLP forward + integrate + clip +
  // wrap + random reset — was ~40 tfjs dispatches). Weights flow tfjs→kernel
  // as ~10KB of GPU→GPU copies per frame; particle state never touches tfjs,
  // so particle count scales to 1M+ without touching the train cost.
  async function tick() {
    if (!running || !(optimizer || trainer) || !advect || !wh || !device) return;
    frame++;

    // ONE command encoder for the whole frame. On the FUSED path it records
    // trainer pass A+B, the advect pass, and the render, then submits ONCE —
    // versus the old three-submits-per-frame (train, advect, render). The tfjs
    // legacy path can't share the encoder for its learn (optimizer.minimize
    // does its own internal submits), but advect+render still share this one.
    const enc = device.createCommandEncoder();
    // Per-pass timestamp descriptors (undefined when the profiler is absent):
    // 0/1 rollout(A), 2/3 optim(B), 4/5 advect, 6/7 render.
    const tsA = timer?.writes(0, 1);
    const tsB = timer?.writes(2, 3);
    const tsAdvect = timer?.writes(4, 5);
    const tsRender = timer?.writes(6, 7);

    // (1) LEARN — one gradient step on a SMALL batch.
    //     Fused path (field pieces): 2 WGSL dispatches (analytic backward +
    //     Adam) recorded into `enc`, writing the shared weights buffer in place
    //     — no tfjs, no readback; the advect pass below then reads the freshly
    //     trained weights (pass ordering in one encoder inserts the barrier).
    //     tfjs path (legacy MLP pieces / ?train=tfjs): optimizer.minimize.
    const trainStart = performance.now();
    // `?trainEvery=N`: amortize training over N frames — applies to BOTH the
    // fused path (imagined-rollout batch) and the tfjs path (SIREN/Fourier,
    // whose autograd + encoding learn stage is heavy — this is how a Fourier
    // piece reaches 60fps: train every 2-3 frames, advect every frame).
    if (frame % trainEvery !== 0) {
      // skip training this frame (both paths)
    } else if (trainer) {
      trainer.encodeStep(
        enc,
        {
          width: w,
          height: h,
          forceMagnitude: cfg.forceMagnitude,
          friction: cfg.friction,
          maxVelocity: cfg.maxVelocity,
        },
        {
          n: sampleRate,
          alpha: (field as HelmholtzField).alpha,
          lr: cfg.learningRate,
          seed: frame,
          source: trainSource,
          mixRandom,
        },
        tsA,
        tsB
      );
      if (frame % 30 === 0) {
        trainer
          .readLoss()
          .then((l) => (hudLoss = l.loss))
          .catch(() => {});
      }
    } else {
      optimizer!.minimize(
        () =>
          tf.tidy(() => {
            const tp = tf.randomUniform([sampleRate, 2], 0, 1).mul(
              wh!
            ) as tf.Tensor2D;
            const tv = tf.zeros([sampleRate, 2]) as tf.Tensor2D;
            const r = physicsForward(tp, tv, model, field, cfg, w, h);
            return cfg.computeLoss(r.newPos, w, h, { force: r.force, field });
          }),
        false,
        varList
      );
    }
    const trainMs = performance.now() - trainStart;

    // (2) ADVECT — ONE fused dispatch over ALL particles, recorded into `enc`.
    //     Returns the tfjs weight-sync clones (empty on the fused path); they
    //     must be disposed AFTER submit so their source buffers survive the
    //     in-flight copies.
    const advectRefs = advect.encodeStep(
      enc,
      frame,
      field ? (field as HelmholtzField).alpha : 0,
      tsAdvect
    );

    // (3) RENDER — dots drawn straight from the kernel's particle buffers,
    //     recorded into `enc`.
    let renderMs = 0;
    if (renderer) {
      const r0 = performance.now();
      const useSplat =
        splat !== null &&
        renderOverride !== "quads" &&
        (renderOverride === "splat" || advect.count >= SPLAT_MIN_N);
      if (useSplat) {
        // AUTO-EXPOSURE: accumulated energy scales with particle density and
        // the trail steady-state 1/(1-decay); normalize so the MEAN displayed
        // energy stays constant across counts (attractor hot-spots still
        // bloom through the tonemap shoulder — that's the aesthetic).
        // The mean is over NATIVE texels — w*h CSS px times dpr² — since each
        // particle's 4096 energy spreads over the dpr-scaled accumulator.
        // `?exposure=F` scales the target.
        splat!.exposure =
          (0.35 * w * h * dpr * dpr * (1 - Math.min(splat!.decay, 0.995))) /
          Math.max(1, advect.count) * exposureScale;
        splat!.encodeRender(
          enc,
          advect.posBuffer,
          advect.velBuffer,
          advect.count,
          w,
          h,
          tsRender
        );
      } else {
        renderer.encodeRender(
          enc,
          advect.posBuffer,
          advect.velBuffer,
          advect.count,
          w,
          h,
          tsRender
        );
      }
      renderMs = performance.now() - r0;
    }

    // Resolve the timestamp query set into a staging buffer within THIS frame's
    // encoder (~every 15 frames), then SINGLE submit for the whole frame.
    if (timer) timer.maybeResolve(enc, frame);
    device.queue.submit([enc.finish()]);
    for (const t of advectRefs) t.dispose();
    if (timer) timer.afterSubmit();

    const now = performance.now();
    emaFrame = ema(emaFrame, now - lastT);
    lastT = now;
    emaTrain = ema(emaTrain, trainMs);
    emaRender = ema(emaRender, renderMs);
    if (frame % 6 === 0) {
      const head =
        `${cfg.name}\n` +
        `backend ${tf.getBackend()}  render=${particleCount} train=${sampleRate}\n` +
        `FPS     ${(1000 / emaFrame).toFixed(1)}  (${emaFrame.toFixed(1)} ms)\n`;
      // HONEST TIMINGS. advect + render are OUR GPU passes on EVERY piece, so
      // when the profiler is live we always show their real GPU time — never
      // the CPU-encode time (which is ~0.1ms: the cost of RECORDING commands,
      // not the async GPU work). The fused path also has real GPU rollout/optim
      // passes; the legacy tfjs path's "learn" has no clean GPU span (tfjs owns
      // its submits), so it's shown as CPU wall time, explicitly labelled.
      // Without timestamp-query at all, everything is labelled (cpu-encode) so
      // nothing masquerades as a real render time.
      const gt = timer?.timings;
      let body: string;
      if (gt && trainer) {
        body =
          `rollout ${gt.rollout.toFixed(2)} ms  optim ${gt.optim.toFixed(2)} ms  loss ${hudLoss.toFixed(3)}\n` +
          `advect  ${gt.advect.toFixed(2)} ms  render ${gt.render.toFixed(2)} ms  (gpu)\n`;
      } else if (gt) {
        body =
          `learn   ${emaTrain.toFixed(1)} ms  (cpu·tfjs)\n` +
          `advect  ${gt.advect.toFixed(2)} ms  render ${gt.render.toFixed(2)} ms  (gpu)\n`;
      } else {
        body =
          `learn   ${emaTrain.toFixed(1)} ms${
            trainer ? `  (fused)  loss ${hudLoss.toFixed(3)}` : "  (cpu·tfjs)"
          }\n` + `render  ${emaRender.toFixed(1)} ms  (cpu-encode)\n`;
      }
      tele.textContent = head + body + `tensors ${tf.memory().numTensors}`;
    }

    requestAnimationFrame(tick);
  }

  (async () => {
    // WebGPU backend so tfjs tensors live in GPUBuffers we can render from with
    // zero copy (same GPUDevice). No fallback — warn and bail if it won't init.
    try {
      await tf.setBackend("webgpu");
      await tf.ready();
    } catch (e) {
      console.error("[webgpu] backend init failed", e);
    }
    if (!running) return;
    if (tf.getBackend() !== "webgpu") {
      showWebGPUWarning();
      return;
    }

    // The shared GPUDevice tfjs created — everything (advect/train/render/timer)
    // records onto it. The optional GPU profiler needs the "timestamp-query"
    // feature (main.ts's requestDevice shim appends it when present); when it's
    // absent GpuTimer.create returns null and the HUD keeps its CPU-encode lines.
    device = (tf.backend() as unknown as { device: GPUDevice }).device;
    timer = GpuTimer.create(device);
    console.log(
      timer
        ? "[gputime] timestamp-query active — HUD shows per-pass GPU ms"
        : "[gputime] no timestamp-query feature — HUD uses CPU-encode ms"
    );

    // `?handoff=N`: override tfjs's small-tensor CPU forwarding threshold
    // (WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD; 0 = force every op onto the GPU).
    // Only affects the tfjs learn path (legacy pieces / ?train=tfjs) — pair
    // with the HUD's learn line to A/B it on real hardware.
    const handoff = new URLSearchParams(location.search).get("handoff");
    if (handoff !== null) {
      tf.env().set("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD", parseInt(handoff, 10) || 0);
      console.log(`[tfjs] CPU handoff threshold -> ${parseInt(handoff, 10) || 0}`);
    }

    // Backend is ready — NOW safe to build models/tensors.
    field = cfg.createField ? cfg.createField() : null;
    model = !field && cfg.createModel ? cfg.createModel() : null;
    varList = field ? field.trainableWeights : undefined;
    optimizer = tf.train.adam(cfg.learningRate);
    wh = tf.tensor2d([[w, h]]);
    tf.keep(wh);

    // Fused advect kernel: WGSL is GENERATED from the live model's layer dims
    // (see advect_wgsl.ts) — works for both the field and legacy MLP pieces.
    // Owns pos/vel as raw GPUBuffers; construction throws loudly on any
    // unsupported architecture instead of silently falling back.
    const physics = {
      width: w,
      height: h,
      forceMagnitude: cfg.forceMagnitude,
      friction: cfg.friction,
      maxVelocity: cfg.maxVelocity,
      resetRate: cfg.resetRate,
    };
    advect = field
      ? AdvectKernel.fromField(field as HelmholtzField, physics, particleCount)
      : AdvectKernel.fromModel(model!, physics, particleCount);

    // Field pieces train FUSED by default: the trainer co-owns the advect
    // kernel's weights buffer and Adam-updates it in place — weights never
    // leave the GPU. tfjs remains only as the (idle) blueprint. Legacy MLP
    // pieces keep the tfjs optimizer (their losses aren't in the kernel yet).
    let wantTfjsTrainer =
      new URLSearchParams(location.search).get("train") === "tfjs";
    const fieldClasses = field ? (field as HelmholtzField).classes ?? 0 : 0;
    if (wantTfjsTrainer && fieldClasses > 0) {
      console.warn(
        "[train] ?train=tfjs ignored: class-aware fields are fused-only " +
          "(tfjs has no class input)"
      );
      wantTfjsTrainer = false;
    }
    // ALL field types (standard/siren/fourier/hashgrid) now train FUSED: the
    // trainer codegen handles sin backward (pre-act checkpoint), the fourier
    // encoding jacobian, and the hashgrid interp/scatter — each verified vs a
    // tfjs-autograd fixture on Metal at cos=1.0 (tools/train_types_test.ts).
    // `?train=tfjs` keeps the autograd path selectable for A/B comparison.
    if (field && !wantTfjsTrainer) {
      // `?rollout=K` (1..16, default 1): K-step BPTT rollout — the loss sees
      // how particles FLOW through the field (evolving pos+vel), not just one
      // step. K is compiled into the trainer's WGSL. K=1 ≡ the tfjs loss.
      // `?window=K` overrides this (and trainEvery) — see windowK above.
      const rollout =
        windowK > 0
          ? windowK
          : Math.max(
              1,
              Math.min(
                16,
                parseInt(new URLSearchParams(location.search).get("rollout") ?? "1", 10) || 1
              )
            );
      // Training states come from the LIVE PARTICLE CLOUD by default: real
      // positions AND velocities, denser where the attractors are (that's
      // where the art lives). Coverage/exploration is the reset slider's job —
      // resets continuously inject fresh uniform vel-0 states into the cloud,
      // hence into the batch. `?batch=random` restores the old uniform source.
      if (new URLSearchParams(location.search).get("batch") === "random") {
        trainSource = "random";
      }
      // `?mix=F` (0..1): coverage floor — fraction of the particle-sourced
      // batch replaced by fresh uniform random points each step.
      mixRandom = Math.max(
        0,
        Math.min(
          1,
          parseFloat(new URLSearchParams(location.search).get("mix") ?? "0") || 0
        )
      );
      trainer = new FusedTrainer(device!, advect.layout, {
        weightsBuffer: advect.weightsBuffer,
        batchCap: 1024,
        kSteps: rollout,
      });
      trainer.uploadWeights(advect.packCurrentWeights());
      trainer.setParticleBuffers(advect.posBuffer, advect.velBuffer, advect.count);
      advect.syncFromTfjs = false;
      console.log(
        `[train] fused trainer active (2 dispatches/step, tfjs idle, ` +
        `batch=${trainSource}, rollout=${rollout})`
      );
    }

    try {
      renderer = new GpuPointRendererWebGPU(canvas, {
        pointSize: (cfg as { pointSize?: number }).pointSize ?? 2.5,
        background: cfg.backgroundColor,
        maxSpeed: cfg.maxVelocity,
        classes: field ? (field as HelmholtzField).classes ?? 0 : 0,
      });
      // splat shares the canvas context with the quad renderer (same device/
      // format); its passes only run on frames where it's picked.
      const decayParam = new URLSearchParams(location.search).get("decay");
      const dotParam = new URLSearchParams(location.search).get("dot");
      splat = new SplatRenderer(canvas, {
        background: cfg.backgroundColor,
        maxSpeed: cfg.maxVelocity,
        classes: field ? (field as HelmholtzField).classes ?? 0 : 0,
        dpr,
        // `?dot=F` — radial splat radius in CSS px (default 1.25)
        radius: dotParam !== null ? parseFloat(dotParam) || 1.25 : 1.25,
        // Trails come from the piece's declared renderer style (the splat's
        // decay is the real mechanism behind all of them); `?decay=F` overrides.
        decay:
          decayParam !== null
            ? Math.max(0, Math.min(0.995, parseFloat(decayParam) || 0))
            : SPLAT_DECAY_BY_RENDERER[cfg.renderer],
        // `?stroke=` / `?strokeLen=` — geometric stroke trails (parsed above)
        style: strokeStyle,
        strokeLen,
      });
    } catch (e) {
      console.error("[webgpu] renderer init failed", e);
      showWebGPUWarning();
      return;
    }

    // --- Stroke-style segmented control (dot | vel | curl) -----------------
    // Sits at the right end of the trails-slider bar (the React bar at
    // bottom:42 shrink-wraps its content on the left). Built imperatively
    // here rather than in index.tsx: it drives a renderer-local knob —
    // splat.style is live-switchable (SplatRenderer lazily builds the stroke
    // pipeline on the first "vel"/"curl" frame, so flipping is cheap).
    {
      const paint = (b: HTMLButtonElement, active: boolean): void => {
        b.style.cssText =
          "padding:2px 8px;border-radius:3px;cursor:pointer;font:11px monospace;" +
          (active
            ? "border:1px solid #5b8cff;background:rgba(60,90,200,0.4);color:#dce8ff"
            : "border:1px solid #444;background:rgba(30,30,50,0.6);color:#888");
      };
      strokeUi = document.createElement("div");
      strokeUi.style.cssText =
        "position:fixed;bottom:44px;right:12px;z-index:101;display:flex;gap:4px;" +
        "align-items:center;padding:5px 8px;background:rgba(0,0,0,0.55);" +
        "border-radius:4px;color:#8fbcff;font:12px monospace";
      const label = document.createElement("span");
      label.textContent = "stroke";
      label.style.marginRight = "4px";
      strokeUi.appendChild(label);
      const btns: HTMLButtonElement[] = [];
      (["dot", "vel", "curl"] as SplatStyle[]).forEach((s) => {
        const b = document.createElement("button");
        b.textContent = s;
        b.onclick = () => {
          if (splat) splat.style = s;
          btns.forEach((x) => paint(x, x.textContent === s));
        };
        paint(b, s === strokeStyle);
        btns.push(b);
        strokeUi!.appendChild(b);
      });
      document.body.appendChild(strokeUi);
    }

    // Live controls for the UI: particle count (resizes kernel buffers,
    // preserving state — grow appends, shrink slices), sample rate, trails.
    // Fired AFTER renderer/splat construction so getDecay() reads the live
    // splat default (the trails slider initializes from it).
    if (onReady) {
      onReady({
        field: field ? (field as HelmholtzField) : null,
        getParticleCount: () => particleCount,
        setParticleCount: (n: number) => {
          if (!advect) return;
          advect.setParticleCount(n);
          particleCount = advect.count;
          // resize replaces the pos/vel buffers — refresh the trainer's view
          if (trainer) {
            trainer.setParticleBuffers(advect.posBuffer, advect.velBuffer, advect.count);
          }
        },
        getResetRate: () => resetRate,
        setResetRate: (r: number) => {
          resetRate = Math.max(0, Math.min(0.2, r));
          if (advect) advect.setResetRate(resetRate);
        },
        getSampleRate: () => sampleRate,
        setSampleRate: (n: number) => {
          sampleRate = Math.max(1, Math.round(n));
        },
        getDecay: () => splat!.decay,
        setDecay: (d: number) => {
          if (splat) splat.decay = Math.max(0, Math.min(0.99, d));
        },
      });
    }
    console.log(`starting: ${cfg.name} (webgpu)`);
    tick();
  })();

  return () => {
    running = false;
    if (renderer) renderer.destroy();
    if (splat) splat.destroy?.();
    if (timer) timer.destroy(); // querySet + resolve/staging GPUBuffers
    tele.remove();
    if (strokeUi) strokeUi.remove();
    if (optimizer) optimizer.dispose(); // frees Adam accumulators (leaked on tab switch)
    if (wh) wh.dispose();
    if (trainer) trainer.destroy(); // batch/scratch/grads/adam GPUBuffers
    if (advect) advect.destroy(); // pos/vel/weights GPUBuffers
    if (model) model.dispose();
    if (field) field.dispose();
  };
}
