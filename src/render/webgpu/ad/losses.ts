/**
 * The composite loss terms, built in the IR. These mirror `helmholtzChaosLoss`
 * in main.ts and the fused trainer's forward (train_wgsl.ts:454) EXACTLY — same
 * epsilons, same shapes — so the AD gradient reproduces the shipped one.
 *
 * Unweighted: the W_* weights and 1/N means are linear post-multipliers applied
 * where the terms are combined (see rollout.ts / the trainer), not baked in.
 * See docs/PLAN_AD_IR_BACKWARD_CODEGEN.md §3.3.
 */
import { Graph, type Node } from "./ir";
import { buildHead, type HeadDim } from "./head";

export type V2 = [Node, Node];

export const sub2 = (g: Graph, a: V2, b: V2): V2 => [g.sub(a[0], b[0]), g.sub(a[1], b[1])];
export const dot2 = (g: Graph, a: V2, b: V2): Node => g.add(g.mul(a[0], b[0]), g.mul(a[1], b[1]));

/** chaos = −log(sep + 1e-6), sep = |[fx−f0, fy−f0]| / (hh·√2). */
export function chaosTerm(g: Graph, f0: V2, fx: V2, fy: V2, hh: number): Node {
  const dxv = sub2(g, fx, f0);
  const dyv = sub2(g, fy, f0);
  const sq = g.sqrt(g.add(g.add(dot2(g, dxv, dxv), dot2(g, dyv, dyv)), g.const(1e-12)));
  const sep = g.div(sq, g.const(hh * 1.4142 + 1e-9));
  return g.neg(g.log(g.add(sep, g.const(1e-6))));
}

/** divergence = g², g = ((fx.x−f0.x)+(fy.y−f0.y))/hh. */
export function divergenceTerm(g: Graph, f0: V2, fx: V2, fy: V2, hh: number): Node {
  const gd = g.div(g.add(g.sub(fx[0], f0[0]), g.sub(fy[1], f0[1])), g.const(hh));
  return g.mul(gd, gd);
}

/**
 * EXACT-probe chaos (plan §5.1): the h→0 limit of {@link chaosTerm}, with the
 * finite-diff columns replaced by the true jacobian columns Jx=∂F/∂x, Jy=∂F/∂y
 * (from forward-mode `jvp`). In the FD version dxv ≈ h·Jx, so
 * sep = √(|dxv|²+|dyv|²)/(h·√2) → √(|Jx|²+|Jy|²)/√2 — no h, no truncation
 * error, and 2 fewer field evals. NOT the shipped loss (flipping changes
 * training semantics); selectable via RolloutCfg.probes.
 */
export function chaosTermExact(g: Graph, Jx: V2, Jy: V2): Node {
  const sq = g.sqrt(g.add(g.add(dot2(g, Jx, Jx), dot2(g, Jy, Jy)), g.const(1e-12)));
  const sep = g.div(sq, g.const(1.4142));
  return g.neg(g.log(g.add(sep, g.const(1e-6))));
}

/** EXACT-probe divergence: (∇·F)² = (Jx.x + Jy.y)² — the FD term's h→0 limit. */
export function divergenceTermExact(g: Graph, Jx: V2, Jy: V2): Node {
  const gd = g.add(Jx[0], Jy[1]);
  return g.mul(gd, gd);
}

/** isotropy on the batch covariance: Liso = (D²+4C01²)/S², D=C00−C11, S=C00+C11+1e-6. */
export function isotropyTerm(g: Graph, C00: Node, C11: Node, C01: Node): Node {
  const S = g.add(g.add(C00, C11), g.const(1e-6));
  const D = g.sub(C00, C11);
  const num = g.add(g.mul(D, D), g.mul(g.const(4), g.mul(C01, C01)));
  return g.div(num, g.mul(S, S));
}

/**
 * spiral = min_k (r − b·max(θ+2πk, 0))²  (winner-take-all over turns).
 * r = √(dx²+dy²+1e-4), θ = atan2(dy,dx). The min routes gradient to the winning
 * branch — reverse mode handles that via the min rule, no hand-written argmin.
 */
export function spiralTerm(g: Graph, dx: Node, dy: Node, b: number, turns: number): Node {
  const r = g.sqrt(g.add(g.add(g.mul(dx, dx), g.mul(dy, dy)), g.const(1e-4)));
  const phi = g.atan2(dy, dx);
  let best: Node | null = null;
  for (let k = 0; k <= turns + 1; k++) {
    const theta = g.add(phi, g.const(2 * Math.PI * k));
    const diff = g.sub(r, g.mul(g.const(b), g.max(theta, g.const(0))));
    const d = g.mul(diff, diff);
    best = best === null ? d : g.min(best, d);
  }
  return best!;
}

/* ══════════════════════════════════════════════════════════════════════════
   Relaxed winner-take-all adversary — the IR oracle for the fused port of
   src/core/gan/adversary.ts. Per-tuple only: B (the batch of tuples) is the
   THREAD dimension, never an IR dimension, and the 1/B mean plus the
   reward-normalizer 1/EMA[RMS‖y‖] (a tape CONSTANT in the tfjs reference)
   are linear post-multipliers applied at the combination site.
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * Softening constant inside every residual radicand — must equal SOFT_EPS in
 * src/core/gan/adversary.ts (residual = sqrt(Σ(y−g)² + SOFT_EPS²)). d/dx sqrt(0)
 * is ∞ and a zero residual is exactly where a well-fit predictor lives.
 */
export const WTA_SOFT_EPS = 1e-6;
/** Target directions below this squared norm are inactive in the adjusted
 * angular game. Must match adversary.ts and adversary_wgsl.ts. */
/** Mirrors the tfjs and fused kernels. The dead-zone also bounds the quotient
 * Jacobian and is therefore part of the numerical contract, not test trivia. */
export const WTA_DIRECTION_ACTIVE_FLOOR = 1e-3;

export type WtaMetric = "euclidean" | "angular";

/**
 * Adversary weight-leaf names. DISTINCT PREFIX ON PURPOSE: the field's rollout
 * graph owns `w_{h}_{l}_{i}_{j}` / `b_{h}_{l}_{j}` plus `p0x,p0y,dC00,dC11,dC01`
 * (rollout.ts:82-83), and Graph.input memoizes BY NAME — a collision would
 * silently alias adversary parameters onto field parameters.
 *   aw_{j}_{l}_{i}_{o} : head j, layer l, kernel row i (input), col o (output)
 *   ab_{j}_{l}_{o}     : head j, layer l, bias o
 * Row-major tfjs Dense convention, identical to buildHead.
 */
export const awName = (j: number) => (l: number, i: number, o: number) => `aw_${j}_${l}_${i}_${o}`;
export const abName = (j: number) => (l: number, o: number) => `ab_${j}_${l}_${o}`;

/** What {@link wtaTerm} hands back — every piece the fused port needs. */
export interface WtaTerm {
  /** [k][dy] raw head outputs g_j(u) (the head-spread probe reads these). */
  preds: Node[][];
  /** [k] softened residuals ‖y − g_j(u)‖. */
  resid: Node[];
  /** Compatibility/render spelling for {@link payoff}. */
  surprise: Node;
  /** Pure nearest-head residual, retained as a quantization diagnostic only. */
  minResidual: Node;
  /**
   * Σ_j w_j·resid_j — the PER-TUPLE discriminator objective (mean over B of
   * this equals the tfjs trainStep loss Σ_j mean(w_j·d_j)). See the docstring
   * on {@link wtaTerm} for the w_j semantics.
   */
  weighted: Node;
  /**
   * The strict zero-sum payoff shared by both players. The predictor minimizes
   * it; the field minimizes its negation. This is intentionally the SAME
   * relaxed-WTA expression, not the old asymmetric min-only generator reward.
   */
  payoff: Node;
}

/**
 * HANDLER — relaxed-WTA weighting (k ≥ 2). Winner takes 1−ε, the k−1 losers
 * share ε (ε/(k−1) each) — Rupprecht et al. relaxed-WTA, exactly
 * `weightsWta` in src/core/gan/adversary.ts.
 *
 * Winner indicator for head j = "j is the FIRST argmin", matching tf.argMin's
 * lowest-index tie rule:
 *   win_j = Π_{i<j} [resid_i > resid_j] · Π_{i>j} [resid_j ≤ resid_i]
 * built from `gt` nodes, whose reverse rule is ZERO (autodiff.ts:109).
 */
function relaxedWtaWeights(g: Graph, resid: Node[], relaxEps: number): Node[] {
  const k = resid.length;
  const loserShare = relaxEps / (k - 1);
  const winBonus = 1 - relaxEps - loserShare; // w_j = loserShare + win_j·winBonus
  const weights: Node[] = [];
  for (let j = 0; j < k; j++) {
    let win: Node = g.const(1);
    for (let i = 0; i < j; i++) win = g.mul(win, g.gt(resid[i], resid[j]));
    for (let i = j + 1; i < k; i++) win = g.mul(win, g.sub(g.const(1), g.gt(resid[j], resid[i])));
    weights.push(g.add(g.const(loserShare), g.mul(win, g.const(winBonus))));
  }
  return weights;
}

function relaxedWtaSum(g: Graph, resid: Node[], relaxEps: number): Node {
  const weights = relaxedWtaWeights(g, resid, relaxEps);
  return g.sum(resid.map((r, j) => g.mul(weights[j], r)));
}

/**
 * Relaxed winner-take-all multiple-choice term: k head-MLPs on a SHARED
 * context u, each guessing the target y; per-tuple surprise = min_j residual,
 * per-tuple discriminator objective = relaxed-weighted residual sum.
 *
 * Weight leaves are `aw_{j}_{l}_{i}_{o}` / `ab_{j}_{l}_{o}` ({@link awName}).
 * All k heads share `dims` (the shape), never the leaves.
 *
 * ## Relaxation-weight gradient semantics (MATCHED TO tfjs, verified, not guessed)
 *
 * In the reference `Adversary.trainStep` the weights w_j are computed from the
 * PRE-UPDATE residuals in a tidy OUTSIDE `optimizer.minimize`, then `unstack`ed
 * into plain tensors that enter the minimized closure as DATA — they are
 * constants on the tape twice over (computed off-tape, and derived through
 * `argMin`/`oneHot` which have no gradient anyway). So tfjs's gradient of the
 * weighted loss w.r.t. head j is exactly w_j·∂resid_j/∂θ_j, with NO derivative
 * through the winner selection.
 *
 * This graph reproduces that semantics STRUCTURALLY: the winner indicators are
 * products of `gt` nodes and `gt`'s vjp is [0,0], so the reverse pass sees the
 * w_j as constants evaluated at the same (pre-update) parameters — identical
 * values, identical gradient. Ties route to the LOWEST head index, matching
 * tf.argMin. Verified against the real Adversary in tools/ad_wta_test.ts
 * (loss rel < 1e-5, grad cos > 0.99999, per-tuple winner agreement).
 *
 * `minResidual` retains the pure min-fold diagnostic. `payoff`/`surprise` are
 * the relaxed weighted scalar shared by D and G; using one scalar with opposite
 * signs is what makes this a strict adversarial game.
 *
 * kind dispatch mirrors AdversaryKind: k = 1 is the `single` control (weight
 * ≡ 1, no relaxation — relaxEps unused); k ≥ 2 is relaxed WTA.
 */
export function wtaTerm(
  g: Graph,
  u: Node[],
  y: Node[],
  dims: HeadDim[],
  k: number,
  relaxEps: number,
  metric: WtaMetric = "euclidean",
  /** Optional stop-gradient validity mask (0/1), e.g. triangle tie rejection. */
  valid?: Node
): WtaTerm {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error(`wtaTerm: k must be an integer >= 1, got ${k}`);
  }
  const relaxUpper = k <= 1 ? 1 : (k - 1) / k;
  if (!(relaxEps >= 0 && relaxEps < relaxUpper)) {
    throw new Error(
      `wtaTerm: relaxEps must be in [0, ${relaxUpper}) for k=${k}, got ${relaxEps}`
    );
  }
  if (dims.length === 0 || dims[0].inSize !== u.length) {
    throw new Error(
      `wtaTerm: dims[0].inSize ${dims[0]?.inSize} must equal du = ${u.length}`
    );
  }
  if (dims[dims.length - 1].outSize !== y.length) {
    throw new Error(
      `wtaTerm: last outSize ${dims[dims.length - 1].outSize} must equal dy = ${y.length}`
    );
  }
  if (metric === "angular" && y.length !== 2) {
    throw new Error(`wtaTerm: angular metric requires dy = 2, got ${y.length}`);
  }

  const preds: Node[][] = [];
  const resid: Node[] = [];
  const metricActive =
    metric === "angular"
      ? g.gt(dot2(g, y as V2, y as V2), g.const(WTA_DIRECTION_ACTIVE_FLOOR ** 2))
      : g.const(1);
  const targetActive = valid === undefined ? metricActive : g.mul(valid, metricActive);
  for (let j = 0; j < k; j++) {
    const pred = buildHead(g, dims, u, awName(j), abName(j));
    preds.push(pred);
    const compared =
      metric === "angular"
        ? (() => {
            const den = g.sqrt(
              g.add(dot2(g, pred as V2, pred as V2), g.const(WTA_SOFT_EPS ** 2))
            );
            return [g.div(pred[0], den), g.div(pred[1], den)];
          })()
        : pred;
    // ‖y − compared_j(u)‖ softened INSIDE the radicand.
    let ss: Node = g.const(0);
    for (let o = 0; o < y.length; o++) {
      const d = g.sub(compared[o], y[o]);
      ss = g.add(ss, g.mul(d, d));
    }
    const r = g.sqrt(g.add(ss, g.const(WTA_SOFT_EPS * WTA_SOFT_EPS)));
    const metricResidual =
      metric === "angular" ? g.sub(r, g.const(WTA_SOFT_EPS)) : r;
    resid.push(g.mul(targetActive, metricResidual));
  }

  // min-fold over heads, ascending j (ties → earlier, = tf.argMin).
  let minResidual = resid[0];
  for (let j = 1; j < k; j++) minResidual = g.min(minResidual, resid[j]);

  const weighted = k === 1 ? resid[0] : relaxedWtaSum(g, resid, relaxEps);
  const payoff = weighted;
  const surprise = payoff;
  return { preds, resid, surprise, minResidual, weighted, payoff };
}

/* ══════════════════════════════════════════════════════════════════════════
   Explicit fused adversary objectives. These supersede the legacy `angular`
   metric without changing wtaTerm's compatibility ABI.
   ══════════════════════════════════════════════════════════════════════════ */

export type AdversaryObjectiveLoss =
  | { readonly tag: "raw-vector" }
  | { readonly tag: "soft-angle"; readonly tau: number }
  | {
      readonly tag: "angle-relative-scale";
      readonly tau: number;
      readonly scaleWeight: number;
      readonly energyWeight: number;
      readonly energyTarget: number;
    }
  | {
      readonly tag: "angle-scale-hold";
      readonly tau: number;
      readonly scaleWeight: number;
      readonly energyWeight: number;
      readonly energyTarget: number;
    };

export interface WtaObjectiveTerm extends WtaTerm {
  /** Relaxed-WTA weighted directional term before scale composition. */
  angleWeighted: Node;
  /** Relaxed-WTA weighted local relative-scale term (zero when absent). */
  scaleWeighted: Node;
  /**
   * Generator game scalar, excluding the positive energy anchor:
   * ordinary modes = -D; scale-hold = -angle + scaleWeight*scale.
   */
  generatorGame: Node;
}

/** Exact smooth R²→S² embedding ψτ(q)=(qx,qy,τ)/sqrt(|q|²+τ²). */
export function softSphere2(g: Graph, q: V2, tau: number): [Node, Node, Node] {
  if (!(Number.isFinite(tau) && tau > 0)) {
    throw new Error(`softSphere2: tau must be finite and > 0, got ${tau}`);
  }
  const den = g.sqrt(g.add(dot2(g, q, q), g.const(tau * tau)));
  return [g.div(q[0], den), g.div(q[1], den), g.div(g.const(tau), den)];
}

/** Softened chord distance between exact S² embeddings. */
export function softSphereChordTerm(
  g: Graph,
  pred: V2,
  target: V2,
  tau: number
): Node {
  const p = softSphere2(g, pred, tau);
  const y = softSphere2(g, target, tau);
  let ss: Node = g.const(WTA_SOFT_EPS * WTA_SOFT_EPS);
  for (let i = 0; i < 3; i++) {
    const d = g.sub(p[i], y[i]);
    ss = g.add(ss, g.mul(d, d));
  }
  return g.sub(g.sqrt(ss), g.const(WTA_SOFT_EPS));
}

/**
 * Explicit per-tuple WTA objective. `vectorTarget` is a sequence of 2-vectors;
 * `scaleTarget` is the optional local homogeneous log-scale descriptor.
 * Winner assignment always uses D's positive joint residual.
 */
export function wtaObjectiveTerm(
  g: Graph,
  u: Node[],
  vectorTarget: Node[],
  scaleTarget: Node[],
  dims: HeadDim[],
  k: number,
  relaxEps: number,
  loss: AdversaryObjectiveLoss,
  valid?: Node
): WtaObjectiveTerm {
  if (loss.tag === "raw-vector") {
    if (scaleTarget.length !== 0) {
      throw new Error("wtaObjectiveTerm: raw-vector does not accept scale channels");
    }
    const base = wtaTerm(g, u, vectorTarget, dims, k, relaxEps, "euclidean", valid);
    return {
      ...base,
      angleWeighted: base.weighted,
      scaleWeighted: g.const(0),
      generatorGame: g.neg(base.weighted),
    };
  }
  if (vectorTarget.length === 0 || vectorTarget.length % 2 !== 0) {
    throw new Error(
      `wtaObjectiveTerm: vector target width must be positive/even, got ${vectorTarget.length}`
    );
  }
  const scaleMode =
    loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold";
  if (scaleMode !== (scaleTarget.length > 0)) {
    throw new Error(
      `wtaObjectiveTerm: ${loss.tag} scale width mismatch (${scaleTarget.length})`
    );
  }
  const out = vectorTarget.length + scaleTarget.length;
  if (
    dims.length === 0 ||
    dims[0].inSize !== u.length ||
    dims[dims.length - 1].outSize !== out
  ) {
    throw new Error(
      `wtaObjectiveTerm: head io must be ${u.length}→${out}, got ` +
        `${dims[0]?.inSize}→${dims[dims.length - 1]?.outSize}`
    );
  }
  if (!Number.isInteger(k) || k < 1) {
    throw new Error(`wtaObjectiveTerm: k must be an integer >=1, got ${k}`);
  }
  const upper = k <= 1 ? 1 : (k - 1) / k;
  if (!(relaxEps >= 0 && relaxEps < upper)) {
    throw new Error(
      `wtaObjectiveTerm: relaxEps must be in [0,${upper}), got ${relaxEps}`
    );
  }
  const active = valid ?? g.const(1);
  const preds: Node[][] = [];
  const angleResid: Node[] = [];
  const scaleResid: Node[] = [];
  const resid: Node[] = [];
  const nVec = vectorTarget.length / 2;
  for (let j = 0; j < k; j++) {
    const pred = buildHead(g, dims, u, awName(j), abName(j));
    preds.push(pred);
    const chords: Node[] = [];
    for (let v = 0; v < nVec; v++) {
      chords.push(
        softSphereChordTerm(
          g,
          [pred[2 * v], pred[2 * v + 1]],
          [vectorTarget[2 * v], vectorTarget[2 * v + 1]],
          loss.tau
        )
      );
    }
    const angle = g.mul(active, g.div(g.sum(chords), g.const(nVec)));
    angleResid.push(angle);
    let scale: Node = g.const(0);
    if (scaleMode) {
      let ss: Node = g.const(WTA_SOFT_EPS * WTA_SOFT_EPS);
      for (let q = 0; q < scaleTarget.length; q++) {
        const e = g.sub(pred[vectorTarget.length + q], scaleTarget[q]);
        ss = g.add(ss, g.div(g.mul(e, e), g.const(scaleTarget.length)));
      }
      scale = g.mul(
        active,
        g.sub(g.sqrt(ss), g.const(WTA_SOFT_EPS))
      );
    }
    scaleResid.push(scale);
    const scaleWeight = scaleMode ? loss.scaleWeight : 0;
    resid.push(g.add(angle, g.mul(g.const(scaleWeight), scale)));
  }
  let minResidual = resid[0];
  for (let j = 1; j < k; j++) minResidual = g.min(minResidual, resid[j]);
  const weights =
    k === 1 ? [g.const(1)] : relaxedWtaWeights(g, resid, relaxEps);
  const weighted = g.sum(resid.map((r, j) => g.mul(weights[j], r)));
  const angleWeighted = g.sum(
    angleResid.map((r, j) => g.mul(weights[j], r))
  );
  const scaleWeighted = g.sum(
    scaleResid.map((r, j) => g.mul(weights[j], r))
  );
  const generatorGame =
    loss.tag === "angle-scale-hold"
      ? g.add(
          g.neg(angleWeighted),
          g.mul(g.const(loss.scaleWeight), scaleWeighted)
        )
      : g.neg(weighted);
  return {
    preds,
    resid,
    surprise: weighted,
    minResidual,
    weighted,
    payoff: weighted,
    angleWeighted,
    scaleWeighted,
    generatorGame,
  };
}

export const REL_SCALE_RATIO = 1e-3;
export const REL_SCALE_ACTIVE_FLOOR = 1e-8;
export const REL_SCALE_ABS_EPS = 1e-6;

export interface LocalScaleTerm {
  target: Node[];
  energy: Node;
  active: Node;
}

/**
 * Exact local homogeneous relative-scale descriptor from raw member 2-vectors.
 * Pair returns the smooth absolute log-magnitude contrast; m>=3 returns
 * centered log magnitudes. Callers put triangle members in canonical order.
 */
export function localRelativeScaleTerm(
  g: Graph,
  signals: V2[]
): LocalScaleTerm {
  if (signals.length < 2) {
    throw new Error("localRelativeScaleTerm: needs at least two member vectors");
  }
  const mag2 = signals.map((s) => dot2(g, s, s));
  const energy = g.div(g.sum(mag2), g.const(signals.length));
  const active = g.gt(
    energy,
    g.const(REL_SCALE_ACTIVE_FLOOR * REL_SCALE_ACTIVE_FLOOR)
  );
  const safeEnergy = g.select(active, energy, g.const(1));
  const soft = g.mul(
    g.const(REL_SCALE_RATIO * REL_SCALE_RATIO),
    safeEnergy
  );
  const logs = mag2.map((r2) =>
    g.mul(g.const(0.5), g.log(g.add(r2, soft)))
  );
  if (signals.length === 2) {
    const d = g.sub(logs[0], logs[1]);
    const smoothAbs = g.sub(
      g.sqrt(g.add(g.mul(d, d), g.const(REL_SCALE_ABS_EPS ** 2))),
      g.const(REL_SCALE_ABS_EPS)
    );
    return { target: [g.mul(active, smoothAbs)], energy, active };
  }
  const mean = g.div(g.sum(logs), g.const(signals.length));
  return {
    target: logs.map((a) => g.mul(active, g.sub(a, mean))),
    energy,
    active,
  };
}

/** Exact batch RMS-energy anchor used by the tfjs and fused generator paths. */
export function energyAnchorTerm(
  g: Graph,
  energies: Node[],
  activities: Node[],
  weight: number,
  target: number
): Node {
  if (energies.length === 0 || energies.length !== activities.length) {
    throw new Error("energyAnchorTerm: energies/activities must be non-empty and aligned");
  }
  const activeCount = g.max(g.sum(activities), g.const(1));
  const mean = g.div(
    g.sum(energies.map((e, i) => g.mul(e, activities[i]))),
    activeCount
  );
  const rms = g.sqrt(g.add(mean, g.const(1e-16)));
  const d = g.sub(rms, g.const(target));
  return g.mul(g.const(weight), g.mul(d, d));
}

/**
 * Normalized post-force/post-friction/post-clip velocity, before borders/reset.
 * Reverse AD through min/max yields the exact unsaturated clip mask.
 */
export function postVelocityTargetTerm(
  g: Graph,
  velocity: V2,
  force: V2,
  forceMagnitude: number,
  friction: number,
  maxVelocity: number
): V2 {
  if (!(Number.isFinite(maxVelocity) && maxVelocity > 0)) {
    throw new Error(`postVelocityTargetTerm: maxVelocity must be >0, got ${maxVelocity}`);
  }
  const lo = g.const(-maxVelocity);
  const hi = g.const(maxVelocity);
  return [0, 1].map((i) => {
    const pre = g.mul(
      g.add(velocity[i], g.mul(g.const(forceMagnitude), force[i])),
      g.const(friction)
    );
    const clipped = g.min(g.max(pre, lo), hi);
    return g.div(clipped, g.const(maxVelocity));
  }) as V2;
}

/**
 * The per-force isotropy energy folded into a per-sample loss: with the batch
 * covariance gradient dC = (dLiso/dC00, dC11, dC01) treated as CONSTANTS, adding
 * Σ_k [dC00·Fsx² + dC11·Fsy² + dC01·Fsx·Fsy] to the loss makes ∂/∂Fs reproduce
 * the trainer's isotropy seed (train_wgsl.ts:622). This is how batch-coupled
 * isotropy becomes a local, autodiff-able term.
 */
export function isotropyFold(g: Graph, Fs: V2[], dC: [Node, Node, Node]): Node {
  const [d00, d11, d01] = dC;
  let acc: Node = g.const(0);
  for (const [fx, fy] of Fs) {
    acc = g.add(acc, g.add(
      g.add(g.mul(d00, g.mul(fx, fx)), g.mul(d11, g.mul(fy, fy))),
      g.mul(d01, g.mul(fx, fy))
    ));
  }
  return acc;
}
