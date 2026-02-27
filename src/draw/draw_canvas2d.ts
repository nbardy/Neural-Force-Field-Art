import * as tf from "@tensorflow/tfjs";
import { AgentBatch } from "../types/all";

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------
function rgb(r: number, g: number, b: number): string {
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}
function rgba(r: number, g: number, b: number, a: number): string {
  return `rgba(${r | 0},${g | 0},${b | 0},${a})`;
}

// Three agent-group palettes
const PALETTE = [
  { core: [255, 60, 255], glow: [255, 60, 255] },   // magenta
  { core: [60, 220, 220], glow: [60, 220, 220] },    // teal
  { core: [120, 255, 200], glow: [120, 255, 200] },   // aquamarine
];

// ---------------------------------------------------------------------------
// Generic agent-batch renderer (kept for backward compat)
// ---------------------------------------------------------------------------
export function drawAgents(
  canvas: HTMLCanvasElement,
  agents: AgentBatch,
  config: { predictField?: boolean }
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.fillStyle = "rgb(12,0,34)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  agents.agentPositions.forEach((posTensor, i) => {
    const positions = (posTensor as tf.Tensor2D).arraySync() as number[][];
    const velocities = (agents.agentVelocities[i] as tf.Tensor2D).arraySync() as number[][];
    const c = PALETTE[i % PALETTE.length];

    for (let j = 0; j < positions.length; j++) {
      const [x, y] = positions[j];
      const [vx, vy] = velocities[j];

      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = rgba(c.glow[0], c.glow[1], c.glow[2], 0.15);
      ctx.fill();

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = rgb(c.core[0], c.core[1], c.core[2]);
      ctx.fill();

      const spd = Math.sqrt(vx * vx + vy * vy);
      if (spd > 0.1) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + vx * 8, y + vy * 8);
        ctx.strokeStyle = rgba(c.glow[0], c.glow[1], c.glow[2], 0.5);
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Spiral diffusion scene renderer
// ---------------------------------------------------------------------------
export function drawSpiralScene(
  canvas: HTMLCanvasElement,
  positions: number[][],
  velocities: number[][],
  spiralTarget: number[][],
  w: number,
  h: number,
  frame: number,
  forceField?: { pos: number[][]; vec: number[][] }
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // --- background -------------------------------------------------------
  ctx.fillStyle = "rgb(8,2,28)";
  ctx.fillRect(0, 0, w, h);

  // --- force field arrows (drawn first, behind everything) --------------
  if (forceField) {
    const { pos: fPos, vec: fVec } = forceField;
    for (let i = 0; i < fPos.length; i++) {
      const fx = fPos[i][0] * w;
      const fy = fPos[i][1] * h;
      const mag = Math.sqrt(fVec[i][0] ** 2 + fVec[i][1] ** 2);
      if (mag < 0.01) continue;
      const scale = 18;
      const dx = (fVec[i][0] / mag) * scale;
      const dy = (fVec[i][1] / mag) * scale;
      const alpha = Math.min(0.35, mag * 0.5);
      ctx.beginPath();
      ctx.moveTo(fx, fy);
      ctx.lineTo(fx + dx, fy + dy);
      ctx.strokeStyle = rgba(80, 50, 160, alpha);
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  }

  // --- target spiral curve ----------------------------------------------
  ctx.beginPath();
  for (let i = 0; i < spiralTarget.length; i++) {
    const sx = spiralTarget[i][0] * w;
    const sy = spiralTarget[i][1] * h;
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.strokeStyle = rgba(140, 90, 255, 0.3);
  ctx.lineWidth = 2;
  ctx.stroke();

  // spiral dots along the curve for visibility
  for (let i = 0; i < spiralTarget.length; i += 25) {
    const sx = spiralTarget[i][0] * w;
    const sy = spiralTarget[i][1] * h;
    ctx.beginPath();
    ctx.arc(sx, sy, 1.5, 0, Math.PI * 2);
    ctx.fillStyle = rgba(140, 90, 255, 0.25);
    ctx.fill();
  }

  // --- particles --------------------------------------------------------
  const n = positions.length;
  for (let i = 0; i < n; i++) {
    const px = positions[i][0] * w;
    const py = positions[i][1] * h;
    const vx = velocities[i][0];
    const vy = velocities[i][1];
    const c = PALETTE[i % PALETTE.length];

    // glow
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = rgba(c.glow[0], c.glow[1], c.glow[2], 0.12);
    ctx.fill();

    // core
    ctx.beginPath();
    ctx.arc(px, py, 2.2, 0, Math.PI * 2);
    ctx.fillStyle = rgb(c.core[0], c.core[1], c.core[2]);
    ctx.fill();

    // velocity tail
    const spd = Math.sqrt(vx * vx + vy * vy);
    if (spd > 0.0005) {
      const sc = w * 6;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + vx * sc, py + vy * sc);
      ctx.strokeStyle = rgba(c.glow[0], c.glow[1], c.glow[2], 0.35);
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
  }

  // --- HUD --------------------------------------------------------------
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = "13px monospace";
  ctx.fillText(`frame ${frame}`, 10, 20);
}
