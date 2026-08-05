# Fused adversary objective implementation

## Goal

Implement the new adversarial target/loss choices in the production fused
WebGPU path while preserving the existing five-dispatch ordering, independent
predictor/field optimizer lanes, `extGrads` seam, and legacy default behavior.

Owned files for this bounded task:

- `src/render/webgpu/adversary_wgsl.ts`
- `src/render/webgpu/adversary_train.ts`
- `src/render/webgpu/ad/losses.ts`
- `tools/train_wta_test.ts`
- `tools/ad_wta_test.ts`

No edits to `main.ts`, `index.tsx`, or `src/core/gan/adversary.ts`; another
agent owns those public interfaces.

## Required target/loss matrix

- Targets: raw `force`; point-state `post-velocity`.
- Losses: `raw-vector`, `soft-angle`, `angle-relative-scale`,
  `angle-scale-hold`.
- `post-velocity` is point-only and must expose incoming normalized velocity
  in the predictor context.
- Relational scale is local to each tuple. Point + relative-scale is rejected.
- The explicit raw-vector mode is the known amplitude-cheat negative control.

## Verified starting facts

- The fused target currently is already raw `F(x)` before force magnitude,
  velocity integration, friction, clipping, borders, or reset.
- The production step ordering is:
  `advFwd(pre-D) -> advFinalize -> advOpt -> advFwd(post-D) -> fieldGrad`.
- Pass A already consumes the implementation's practical storage-binding
  budget. Reuse scratch/stats; do not add a storage binding.
- Particle velocity is already bound, and uploaded tuple rows already contain
  `[px, py, vx, vy]`.
- The current scratch layout assumes encoded target width equals predictor
  output width. Relative-scale prediction requires separating vector target
  width from total predictor output width.
- Current finalized stats occupy `[discLoss,payoff,batchRms,headMean,headMin,
  wins...]`, with partials starting at `ADV_STATS_BASE=32`.

## Intended mathematics

For every 2-vector use the exact soft-spherical embedding

`psi_tau(q) = (qx,qy,tau) / sqrt(||q||^2 + tau^2)`

and chord residual

`||psi_tau(pred)-psi_tau(target)||` (softened at coincidence as needed).
The shader must differentiate this scalar exactly; no custom/surrogate
backward and no hard near-zero quotient.

Relative-scale targets are dimensionless and exactly invariant to uniform
tuple scaling:

- Pair: a nondegenerate within-pair contrast (raw force contrast versus tuple
  RMS), not the degenerate centered magnitudes of `(+q,-q)`.
- Tri/quad: centered softened log member magnitudes.

`angle-relative-scale`: predictor and disagreeing generator oppose each other
on both direction and relative scale, plus a positive fixed energy anchor.

`angle-scale-hold`: predictor still learns its declared prediction terms;
field direction is adversarial, relative scale is cooperative, and the same
positive fixed energy anchor prevents collapse/inflation.

For point `post-velocity`, target the normalized pre-border update and include
incoming normalized velocity in context. Differentiate the physical update
exactly through its clip mask; reject relational post-velocity combinations.

## Open implementation concerns

- The concurrently edited core file currently contains duplicate declarations
  in a few places. Fused code must consume only stable tag spellings and avoid
  depending on unfinished helper signatures until reconciled.
- Need inspect whether the core's `angle-scale-hold` intends a scale predictor
  channel; current type documentation appears to say scale is not in D's
  residual, while the user's requested semantics say scale is cooperative.
  The fused implementation must choose one explicit contract and record it.
- Need keep WGSL specialization compile-time by target/loss to avoid bloating
  every kernel and increasing register pressure.

## Progress

- 2026-07-30 03:35 KST: bounded task started; no production files edited yet.

