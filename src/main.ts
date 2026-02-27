/**
 * Diffusion score-matching learner for an Archimedes spiral force field.
 *
 * A ScoreNetwork (trashPanda Module) learns the score function ∇log p(x)
 * for a distribution concentrated on an Archimedes spiral. The learned
 * vector field is applied as a force to a particle system, so particles
 * gradually converge from a random cloud into the spiral shape.
 */
import * as tf from "@tensorflow/tfjs";
import { ScoreNetwork } from "./trashPanda/models/ScoreNetwork";
import { drawSpiralScene } from "./draw/draw_canvas2d";

// ---------------------------------------------------------------------------
// Backend init with graceful fallback
// ---------------------------------------------------------------------------
async function initBackend() {
  for (const backend of ["webgpu", "webgl", "cpu"]) {
    try {
      if (await tf.setBackend(backend)) {
        await tf.ready();
        console.log(`TF.js backend: ${tf.getBackend()}`);
        return;
      }
    } catch (_) {}
  }
  throw new Error("No TF.js backend available");
}

// ---------------------------------------------------------------------------
// Archimedes spiral   r = b·θ   centred at (0.5, 0.5) in [0,1]²
// ---------------------------------------------------------------------------
const SPIRAL_TURNS = 3;
const SPIRAL_MAX_THETA = SPIRAL_TURNS * 2 * Math.PI;
const SPIRAL_B = 0.38 / SPIRAL_MAX_THETA;

function generateSpiralPoints(n: number): number[][] {
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const theta = (i / n) * SPIRAL_MAX_THETA;
    const r = SPIRAL_B * theta;
    pts.push([0.5 + r * Math.cos(theta), 0.5 + r * Math.sin(theta)]);
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Nearest-point score target: direction + magnitude toward spiral
// ---------------------------------------------------------------------------
function nearestSpiralTarget(
  px: number,
  py: number,
  spiralDense: number[][]
): [number, number] {
  let bestD2 = Infinity;
  let bx = px,
    by = py;
  for (let i = 0; i < spiralDense.length; i++) {
    const dx = spiralDense[i][0] - px;
    const dy = spiralDense[i][1] - py;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      bx = spiralDense[i][0];
      by = spiralDense[i][1];
    }
  }
  const dx = bx - px;
  const dy = by - py;
  const dist = Math.sqrt(bestD2);
  if (dist < 1e-6) return [0, 0];
  const mag = 1 - Math.exp(-dist * 10);
  return [(dx / dist) * mag, (dy / dist) * mag];
}

// ---------------------------------------------------------------------------
// Build one training batch: random positions + noisy spiral positions
// ---------------------------------------------------------------------------
function buildBatch(spiralDense: number[][], nRand: number, nNear: number) {
  const positions: number[][] = [];
  const targets: number[][] = [];

  for (let i = 0; i < nRand; i++) {
    const px = Math.random();
    const py = Math.random();
    positions.push([px, py]);
    targets.push(nearestSpiralTarget(px, py, spiralDense));
  }

  for (let i = 0; i < nNear; i++) {
    const sp = spiralDense[(Math.random() * spiralDense.length) | 0];
    const sigma = 0.005 + Math.random() * 0.12;
    const px = sp[0] + (Math.random() - 0.5) * 2 * sigma;
    const py = sp[1] + (Math.random() - 0.5) * 2 * sigma;
    positions.push([px, py]);
    targets.push(nearestSpiralTarget(px, py, spiralDense));
  }

  return { positions, targets };
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
export async function startLoop(canvas: HTMLCanvasElement) {
  await initBackend();

  const width = window.innerWidth;
  const height = window.innerHeight;
  canvas.width = width;
  canvas.height = height;

  const model = new ScoreNetwork({ hiddenUnits: [32, 64, 32] });
  const optimizer = tf.train.adam(0.005);

  const spiralPoints = generateSpiralPoints(1000);

  const PARTICLE_COUNT = 400;
  const pos = tf.variable(tf.randomUniform([PARTICLE_COUNT, 2], 0.08, 0.92));
  const vel = tf.variable(tf.zeros([PARTICLE_COUNT, 2]) as tf.Tensor2D);

  let frame = 0;
  let cachedField: { pos: number[][]; vec: number[][] } | undefined;

  function tick() {
    frame++;

    // --- train ----------------------------------------------------------
    const steps = frame < 60 ? 3 : 1;
    for (let s = 0; s < steps; s++) {
      const { positions, targets } = buildBatch(spiralPoints, 150, 100);
      optimizer.minimize(() =>
        tf.tidy(() => {
          const pT = tf.tensor2d(positions);
          const tT = tf.tensor2d(targets);
          return (model.predict(pT) as tf.Tensor2D)
            .sub(tT)
            .square()
            .mean()
            .asScalar();
        })
      );
    }

    // --- simulate (Langevin dynamics) ------------------------------------
    tf.tidy(() => {
      const forces = model.predict(pos) as tf.Tensor2D;
      const ramp = Math.min(1, frame / 40);
      const forceMag = 0.003 * ramp;
      const friction = 0.93;
      const noiseMag = 0.0012;

      const noise = tf.randomNormal([PARTICLE_COUNT, 2]).mul(noiseMag);
      const nv = vel
        .add(forces.mul(forceMag))
        .add(noise)
        .mul(friction) as tf.Tensor2D;
      const np = pos.add(nv).clipByValue(0.002, 0.998) as tf.Tensor2D;
      vel.assign(nv);
      pos.assign(np);
    });

    // --- draw -----------------------------------------------------------
    const posArr = pos.arraySync() as number[][];
    const velArr = vel.arraySync() as number[][];

    if (frame % 6 === 1) {
      const G = 16;
      const gp: number[][] = [];
      for (let gy = 0; gy < G; gy++)
        for (let gx = 0; gx < G; gx++)
          gp.push([(gx + 0.5) / G, (gy + 0.5) / G]);
      const gT = tf.tensor2d(gp);
      const gF = (model.predict(gT) as tf.Tensor2D).arraySync() as number[][];
      gT.dispose();
      cachedField = { pos: gp, vec: gF };
    }

    drawSpiralScene(
      canvas,
      posArr,
      velArr,
      spiralPoints,
      width,
      height,
      frame,
      cachedField
    );

    requestAnimationFrame(tick);
  }

  console.log("starting spiral diffusion loop");
  tick();
}
