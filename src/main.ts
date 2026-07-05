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
  return (pos, w, h, ctx) =>
    tf.tidy(() => {
      const field = ctx!.field!;
      const force = ctx!.force!;
      const posNorm = pos.div(tf.tensor2d([[w, h]])) as tf.Tensor2D;
      const fieldFn = (p: tf.Tensor2D): tf.Tensor2D => field.forces(p);

      const chaos = chaosLoss(fieldFn, posNorm);
      const iso = isotropyLoss(force);
      const div = divergencePenalty(fieldFn, posNorm);
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
    name: "Helmholtz · Chaos",
    particleCount: 1200,
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

  const updVel = vel.add(forces).mul(cfg.friction);

  const vx = updVel
    .slice([0, 0], [-1, 1])
    .clipByValue(-cfg.maxVelocity, cfg.maxVelocity);
  const vy = updVel
    .slice([0, 1], [-1, 1])
    .clipByValue(-cfg.maxVelocity, cfg.maxVelocity);
  const clippedVel = vx.concat(vy, 1) as tf.Tensor2D;

  const updPos = pos.add(clippedVel);
  const px = updPos.slice([0, 0], [-1, 1]).mod(tf.scalar(w));
  const py = updPos.slice([0, 1], [-1, 1]).mod(tf.scalar(h));
  const wrappedPos = px.concat(py, 1) as tf.Tensor2D;

  return { newPos: wrappedPos, newVel: clippedVel, force: forces };
}

// ---------------------------------------------------------------------------
// Random reset — respawn fraction of particles at random positions
// ---------------------------------------------------------------------------
function randomReset(
  pos: tf.Tensor2D,
  vel: tf.Tensor2D,
  rate: number,
  w: number,
  h: number
): { pos: tf.Tensor2D; vel: tf.Tensor2D } {
  return tf.tidy(() => {
    const n = pos.shape[0];
    const keep = tf.less(tf.randomUniform([n, 1]), tf.scalar(1 - rate));

    const rx = tf.randomUniform([n, 1], 0, w);
    const ry = tf.randomUniform([n, 1], 0, h);

    const posX = pos.slice([0, 0], [-1, 1]);
    const posY = pos.slice([0, 1], [-1, 1]);

    const newX = tf.where(keep, posX, rx);
    const newY = tf.where(keep, posY, ry);
    const newPos = newX.concat(newY, 1) as tf.Tensor2D;

    const zeroVel = tf.zeros([n, 2]);
    const newVel = tf.where(keep.tile([1, 2]), vel, zeroVel) as tf.Tensor2D;

    return { pos: newPos, vel: newVel };
  }) as { pos: tf.Tensor2D; vel: tf.Tensor2D };
}

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
export function startLoop(
  canvas: HTMLCanvasElement,
  configIndex: number,
  onField?: (field: HelmholtzField | null) => void
): () => void {
  let running = true;
  const cfg = GALLERY[configIndex];

  const w = window.innerWidth;
  const h = window.innerHeight;
  canvas.width = w;
  canvas.height = h;

  // Render path: GPU-resident (WebGL2 + tfjs dataToGPU, ZERO per-frame readback)
  // by default; Canvas2D fallback. A canvas commits to ONE context type, so we
  // decide up front: gl !== null => GPU path (ctx stays null), else Canvas2D.
  // `?gpu=0` forces the Canvas2D fallback (escape hatch if the WebGL renderer
  // misbehaves on a given machine); otherwise GPU is the default.
  const gpuParam = new URLSearchParams(location.search).get("gpu");
  const wantGpu = gpuParam === "0" ? false : cfg.gpu !== false;
  let gl: WebGL2RenderingContext | null = null;
  if (wantGpu) {
    try {
      gl = GpuPointRenderer.registerCanvasWithTf(canvas);
    } catch (_) {
      gl = null;
    }
  }
  const ctx = gl ? null : canvas.getContext("2d");

  // Either a ForceField (new path) or a sigmoid MLP (legacy). Exactly one is
  // constructed per preset; the other stays null and drives the dispatch in
  // physicsForward and the optimizer varList below.
  const field: ForceField | null = cfg.createField ? cfg.createField() : null;
  const model: tf.Sequential | null =
    !field && cfg.createModel ? cfg.createModel() : null;

  // Hand the live field to the caller so a UI control can mutate `alpha`.
  if (onField) onField(field ? (field as HelmholtzField) : null);

  // When using a field, restrict the optimizer to the field's own weights.
  const varList: tf.Variable[] | undefined = field
    ? field.trainableWeights
    : undefined;

  const optimizer = tf.train.adam(cfg.learningRate);

  // Constructed after backend init (needs the ready webgl context). The canvas
  // was already registered with tfjs above when gl !== null.
  let gpuRenderer: GpuPointRenderer | null = null;

  let pos = tf.randomUniform([cfg.particleCount, 2], 0, 1).mul(
    tf.tensor2d([[w, h]])
  ) as tf.Tensor2D;
  let vel = tf.zeros([cfg.particleCount, 2]) as tf.Tensor2D;
  tf.keep(pos);
  tf.keep(vel);

  const spiralPts = spiralPixelPoints(w, h);
  const renderer = gl
    ? null
    : createRenderer(cfg.renderer, cfg, cfg.particleCount, spiralPts);
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
    emaSim = 0,
    emaRender = 0,
    lastT = performance.now();

  // Learning is DECOUPLED from motion: we train the field on a small random
  // batch each frame (real-time, cheap) and advect ALL visible particles
  // forward-only (no gradient tape, no per-frame readback stall). This is what
  // kept the old build "real-time and fast" — the expensive autograd graph
  // never gates the render loop.
  const TRAIN_BATCH = 256;
  const wh = tf.tensor2d([[w, h]]);
  tf.keep(wh);

  async function tick() {
    if (!running) return;
    frame++;

    // (1) LEARN — one gradient step on a SMALL random batch. This is the only
    //     place the second-order autograd graph runs; it's tiny and does not
    //     block the motion below.
    const trainStart = performance.now();
    optimizer.minimize(
      () =>
        tf.tidy(() => {
          const tp = tf.randomUniform([TRAIN_BATCH, 2], 0, 1).mul(
            wh
          ) as tf.Tensor2D;
          const tv = tf.zeros([TRAIN_BATCH, 2]) as tf.Tensor2D;
          const r = physicsForward(tp, tv, model, field, cfg, w, h);
          return cfg.computeLoss(r.newPos, w, h, { force: r.force, field });
        }),
      false,
      varList
    );
    const trainMs = performance.now() - trainStart;

    // (2) ADVECT — move ALL visible particles forward-only (NO gradient tape).
    //     Cheap enough to render many points every frame.
    const simStart = performance.now();
    const stepped = tf.tidy(() => {
      const r = physicsForward(pos, vel, model, field, cfg, w, h);
      return [r.newPos, r.newVel] as [tf.Tensor2D, tf.Tensor2D];
    });
    const oldPos = pos;
    const oldVel = vel;
    const reset = randomReset(stepped[0], stepped[1], cfg.resetRate, w, h);
    pos = reset.pos;
    vel = reset.vel;
    tf.keep(pos);
    tf.keep(vel);
    oldPos.dispose();
    oldVel.dispose();
    stepped[0].dispose();
    stepped[1].dispose();
    const simMs = performance.now() - simStart;

    // (3) RENDER
    let renderMs = 0;
    if (gpuRenderer) {
      const r0 = performance.now();
      gpuRenderer.render(pos, vel, w, h, frame);
      renderMs = performance.now() - r0;
    } else {
      const r0 = performance.now();
      const posArr = pos.arraySync() as number[][];
      const velArr = vel.arraySync() as number[][];
      renderer!.render(ctx!, w, h, posArr, velArr, frame);
      renderMs = performance.now() - r0;
    }

    const now = performance.now();
    emaFrame = ema(emaFrame, now - lastT);
    lastT = now;
    emaTrain = ema(emaTrain, trainMs);
    emaSim = ema(emaSim, simMs);
    emaRender = ema(emaRender, renderMs);
    if (frame % 6 === 0) {
      tele.textContent =
        `${cfg.name}\n` +
        `backend ${tf.getBackend()}  render=${cfg.particleCount} train=${TRAIN_BATCH}\n` +
        `FPS     ${(1000 / emaFrame).toFixed(1)}  (${emaFrame.toFixed(1)} ms)\n` +
        `learn   ${emaTrain.toFixed(1)} ms\n` +
        `advect  ${emaSim.toFixed(1)} ms\n` +
        `render  ${emaRender.toFixed(1)} ms\n` +
        `tensors ${tf.memory().numTensors}`;
    }

    requestAnimationFrame(tick);
  }

  initBackend(gl ? "webgl" : undefined).then(() => {
    // GPU-resident path: build the renderer now that the webgl backend + context
    // are ready. registerCanvasWithTf already bound our on-screen gl to tfjs, so
    // dataToGPU() textures live in the same context we draw with (zero readback).
    if (gl && tf.getBackend() === "webgl") {
      gpuRenderer = new GpuPointRenderer(canvas, {
        pointSize: (cfg as { pointSize?: number }).pointSize ?? 2,
        background: cfg.backgroundColor,
        alphaBlend: cfg.alphaBlend,
      });
    } else if (gl) {
      console.error(
        `[gpu] backend '${tf.getBackend()}' != webgl; a webgl canvas cannot ` +
          `fall back to Canvas2D on the same element.`
      );
    }
    console.log(
      `starting: ${cfg.name} (backend ${tf.getBackend()}, ${gl ? "gpu" : "canvas2d"})`
    );
    tick();
  });

  return () => {
    running = false;
    if (renderer) renderer.destroy();
    if (gpuRenderer) gpuRenderer.destroy();
    tele.remove();
    wh.dispose();
    pos.dispose();
    vel.dispose();
    if (model) model.dispose();
    if (field) field.dispose();
  };
}
