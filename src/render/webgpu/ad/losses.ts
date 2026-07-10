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
