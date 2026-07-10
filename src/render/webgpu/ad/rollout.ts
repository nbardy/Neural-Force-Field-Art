/**
 * The full per-sample training loss L_ad, built in the IR — a K-step physics
 * rollout of the two-head Helmholtz field, then the composite chaos/div/spiral
 * loss on pos_K plus the batch-isotropy fold. `grad(L_ad, weights)` is the ENTIRE
 * per-sample backward: δ-chains, the K-step BPTT two-adjoint recurrence, and the
 * loss seeds all emerge from one reverse pass. Summed over the batch it equals
 * the hand-written trainer's gradient (verified vs the tfjs fixture at cos≈1).
 *
 * Mirrors tools/grad_reference.ts (the tfjs oracle) exactly: raw field output at
 * the probes, forceMag-scaled forces for physics+isotropy, spiral on pixel
 * pos_K, floored-mod wrap, clip via min/max. See docs/PLAN_AD_IR_BACKWARD_CODEGEN.md.
 */
import { Graph, type Node } from "./ir";
import { buildHead, type HeadDim } from "./head";
import { chaosTerm, divergenceTerm, spiralTerm, isotropyFold, type V2 } from "./losses";

export interface RolloutCfg {
  K: number;
  alpha: number;
  hh: number;
  forceMag: number;
  friction: number;
  maxVel: number;
  W: number;
  H: number;
  spiralTurns: number;
  wChaos: number;
  wDiv: number;
  wSpiral: number;
  wIso: number;
  N: number; // batch size — for the 1/N and 1/(N·K) loss scalings
  /**
   * Input encoding ahead of the heads (default raw). Fourier builds γ(p) in
   * the IR — sin/cos nodes — so its encoding jacobian falls out of the same
   * reverse pass (nothing hand-derived). SIREN needs no entry here: it is raw
   * encoding + act:"sin" layers in the head dims. HashGrid is NOT expressible
   * in this static scalar graph (its gather indices are data-dependent) — its
   * fused backward is verified directly on Metal vs the tfjs fixture instead
   * (tools/train_types_test.ts).
   */
  encoding?: { kind: "raw" } | { kind: "fourier"; octaves: number };
}

/** γ(p) in the IR — [x, y, sin(ωk x), sin(ωk y), cos(ωk x), cos(ωk y)]·k,
 *  ωk = 2^k·2π; feature order matches helmholtz.ts fourierEncode / the WGSL. */
function encodeInput(g: Graph, pn: V2, cfg: RolloutCfg): Node[] {
  const enc = cfg.encoding ?? { kind: "raw" };
  if (enc.kind === "raw") return [pn[0], pn[1]];
  const out: Node[] = [pn[0], pn[1]];
  for (let k = 0; k < enc.octaves; k++) {
    const w = g.const(Math.pow(2, k) * 2 * Math.PI);
    out.push(
      g.sin(g.mul(w, pn[0])),
      g.sin(g.mul(w, pn[1])),
      g.cos(g.mul(w, pn[0])),
      g.cos(g.mul(w, pn[1]))
    );
  }
  return out;
}

// weight/bias leaf names (head h ∈ {0=order/g, 1=chaos/r}), shared across samples
export const wName = (h: number) => (l: number, i: number, j: number) => `w_${h}_${l}_${i}_${j}`;
export const bName = (h: number) => (l: number, j: number) => `b_${h}_${l}_${j}`;

/** blended RAW field force F(pn) = (1−α)·head0(γ(pn)) + α·head1(γ(pn)). No forceMag. */
function blendRaw(
  g: Graph,
  cfg: RolloutCfg,
  dimsG: HeadDim[],
  dimsR: HeadDim[],
  pn: V2,
  alpha: number
): V2 {
  const enc = encodeInput(g, pn, cfg);
  const fg = buildHead(g, dimsG, enc, wName(0), bName(0));
  const fr = buildHead(g, dimsR, enc, wName(1), bName(1));
  const a1 = g.const(1 - alpha), a = g.const(alpha);
  return [
    g.add(g.mul(a1, fg[0]), g.mul(a, fr[0])),
    g.add(g.mul(a1, fg[1]), g.mul(a, fr[1])),
  ];
}

export interface SampleGraph {
  L: Node;           // per-sample loss (all constants folded; Σ over batch = true loss)
  Fs: V2[];          // scaled forces per step — pass 1 reads these for the covariance
  posK: V2;          // final position (pixel), post floored-mod wrap
  chaos: Node;       // unweighted per-sample terms — for loss reconstruction / checks
  div: Node;
  spiral: Node;
}

/**
 * Build one sample's loss graph. Inputs: weights (shared leaves via wName/bName),
 * `p0x`/`p0y` (this sample's start pixel), and `dC00`/`dC11`/`dC01` (the batch
 * covariance gradient, injected as constants after pass 1).
 */
export function buildSample(g: Graph, cfg: RolloutCfg, dimsG: HeadDim[], dimsR: HeadDim[]): SampleGraph {
  let pos: V2 = [g.input("p0x"), g.input("p0y")];
  let vel: V2 = [g.const(0), g.const(0)];
  const W = g.const(cfg.W), H = g.const(cfg.H);
  const fm = g.const(cfg.forceMag), fr = g.const(cfg.friction);
  const vmax = g.const(cfg.maxVel), vmin = g.const(-cfg.maxVel);
  const Fs: V2[] = [];
  for (let k = 0; k < cfg.K; k++) {
    const pn: V2 = [g.div(pos[0], W), g.div(pos[1], H)];
    const fraw = blendRaw(g, cfg, dimsG, dimsR, pn, cfg.alpha);
    const Fk: V2 = [g.mul(fraw[0], fm), g.mul(fraw[1], fm)];
    Fs.push(Fk);
    // vel = clip((vel+F)·friction, ±maxVel)  — clip via min/max
    const vx = g.max(g.min(g.mul(g.add(vel[0], Fk[0]), fr), vmax), vmin);
    const vy = g.max(g.min(g.mul(g.add(vel[1], Fk[1]), fr), vmax), vmin);
    vel = [vx, vy];
    // pos = (pos+vel) mod [W,H] — floored mod (grad ≡ identity on dividend)
    pos = [g.mod(g.add(pos[0], vx), W), g.mod(g.add(pos[1], vy), H)];
  }
  const posK = pos;

  // probes at pn = pos_K/[W,H] (raw field output — matches the fixture)
  const pn: V2 = [g.div(posK[0], W), g.div(posK[1], H)];
  const hh = g.const(cfg.hh);
  const f0 = blendRaw(g, cfg, dimsG, dimsR, pn, cfg.alpha);
  const fx = blendRaw(g, cfg, dimsG, dimsR, [g.add(pn[0], hh), pn[1]], cfg.alpha);
  const fy = blendRaw(g, cfg, dimsG, dimsR, [pn[0], g.add(pn[1], hh)], cfg.alpha);
  const chaos = chaosTerm(g, f0, fx, fy, cfg.hh);
  const div = divergenceTerm(g, f0, fx, fy, cfg.hh);

  // spiral on pos_K in PIXEL coords
  const cx = cfg.W / 2, cy = cfg.H / 2;
  const b = (Math.min(cfg.W, cfg.H) * 0.38) / (cfg.spiralTurns * 2 * Math.PI);
  const spiral = spiralTerm(g, g.sub(posK[0], g.const(cx)), g.sub(posK[1], g.const(cy)), b, cfg.spiralTurns);

  // isotropy fold: dC injected as constants (computed from batch covariance).
  const dC: [Node, Node, Node] = [g.input("dC00"), g.input("dC11"), g.input("dC01")];
  const isoFold = isotropyFold(g, Fs, dC);

  const NK = cfg.N * cfg.K;
  const L = g.sum([
    g.mul(g.const(cfg.wChaos / cfg.N), chaos),
    g.mul(g.const(cfg.wDiv / cfg.N), div),
    g.mul(g.const(cfg.wSpiral / cfg.N), spiral),
    g.mul(g.const(cfg.wIso / NK), isoFold),
  ]);
  return { L, Fs, posK, chaos, div, spiral };
}
