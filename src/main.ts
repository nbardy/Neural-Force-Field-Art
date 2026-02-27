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
import { createRenderer, RendererType, Renderer } from "./renderers";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
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
  createModel: () => tf.Sequential;
  computeLoss: (pos: tf.Tensor2D, w: number, h: number) => tf.Scalar;
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------
async function initBackend() {
  for (const b of ["webgpu", "webgl", "cpu"]) {
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

// ---------------------------------------------------------------------------
// Gallery
// ---------------------------------------------------------------------------
export const GALLERY: ArtPieceConfig[] = [
  {
    name: "Spiral · Ghost",
    particleCount: 1000,
    friction: 0.911,
    forceMagnitude: 1.2,
    maxVelocity: 3.0,
    resetRate: 0.0008,
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
    friction: 0.85,
    forceMagnitude: 1.5,
    maxVelocity: 4.0,
    resetRate: 0.001,
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
    friction: 0.92,
    forceMagnitude: 1.0,
    maxVelocity: 2.5,
    resetRate: 0.002,
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
    friction: 0.80,
    forceMagnitude: 1.8,
    maxVelocity: 5.0,
    resetRate: 0.003,
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
    friction: 0.80,
    forceMagnitude: 1.8,
    maxVelocity: 5.0,
    resetRate: 0.003,
    drawRate: 3,
    learningRate: 0.005,
    backgroundColor: [2, 0, 12],
    alphaBlend: 0.03,
    renderer: "alpha-fade",
    createModel: mlpWide,
    computeLoss: spiralPlusCenterLoss(0.00002),
  },
];

// ---------------------------------------------------------------------------
// Physics step (inside optimizer.minimize — gradients flow through)
// ---------------------------------------------------------------------------
function physicsForward(
  pos: tf.Tensor2D,
  vel: tf.Tensor2D,
  model: tf.Sequential,
  cfg: ArtPieceConfig,
  w: number,
  h: number
): { newPos: tf.Tensor2D; newVel: tf.Tensor2D } {
  const posNorm = pos.div(tf.tensor2d([[w, h]]));
  const raw = model.predict(posNorm) as tf.Tensor2D;
  const forces = raw.sub(0.5).mul(cfg.forceMagnitude);

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

  return { newPos: wrappedPos, newVel: clippedVel };
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
  configIndex: number
): () => void {
  let running = true;
  const cfg = GALLERY[configIndex];
  const ctx = canvas.getContext("2d")!;

  const w = window.innerWidth;
  const h = window.innerHeight;
  canvas.width = w;
  canvas.height = h;

  const model = cfg.createModel();
  const optimizer = tf.train.adam(cfg.learningRate);

  let pos = tf.randomUniform([cfg.particleCount, 2], 0, 1).mul(
    tf.tensor2d([[w, h]])
  ) as tf.Tensor2D;
  let vel = tf.zeros([cfg.particleCount, 2]) as tf.Tensor2D;
  tf.keep(pos);
  tf.keep(vel);

  const spiralPts = spiralPixelPoints(w, h);
  const renderer = createRenderer(cfg.renderer, cfg, cfg.particleCount, spiralPts);
  let frame = 0;

  async function tick() {
    if (!running) return;
    frame++;

    for (let d = 0; d < cfg.drawRate; d++) {
      let newPos: tf.Tensor2D | null = null;
      let newVel: tf.Tensor2D | null = null;

      optimizer.minimize(() => {
        const result = physicsForward(pos, vel, model, cfg, w, h);
        newPos = result.newPos;
        newVel = result.newVel;
        tf.keep(newPos);
        tf.keep(newVel);
        return cfg.computeLoss(newPos, w, h);
      });

      const oldPos = pos;
      const oldVel = vel;

      const reset = randomReset(newPos!, newVel!, cfg.resetRate, w, h);
      pos = reset.pos;
      vel = reset.vel;
      tf.keep(pos);
      tf.keep(vel);

      oldPos.dispose();
      oldVel.dispose();
      newPos!.dispose();
      newVel!.dispose();
    }

    const posArr = pos.arraySync() as number[][];
    const velArr = vel.arraySync() as number[][];
    renderer.render(ctx, w, h, posArr, velArr, frame);

    requestAnimationFrame(tick);
  }

  initBackend().then(() => {
    console.log(`starting: ${cfg.name}`);
    tick();
  });

  return () => {
    running = false;
    renderer.destroy();
    pos.dispose();
    vel.dispose();
    model.dispose();
  };
}
