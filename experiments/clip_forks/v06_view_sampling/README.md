# v06: Same-Resolution View Sampling

## Goal

Make N-of-K camera supervision explicit without lowering CLIP input resolution.
This fork keeps each selected view as a full `256x256` CLIP image and changes
only which camera views are optimized per step.

## Current Controls

- `1/9`, `2/9`, `3/9`, `5/9`, `9/9` view count in the 3D page.
- `epoch views`: shuffled coverage of all cameras before repeating.
- `random views`: fresh K distinct random cameras each step.
- `clip x1`, `clip x2`, `clip x3`, `clip x9`.

## Tool Flags

```bash
VIEW_SAMPLER=epoch CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
VIEW_SAMPLER=random CLIP_BATCH=2 VIEWS=2 bun tools/splat3d/step_bench.ts
TRIALS=3 CONFIGS=k1=1:1,k2=2:2,k2rand=2:2:random,base=3:3 RUNS=10 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

## Snapshot

The `snapshot/` directory contains the pre-fork versions of:

- `src/splat3d/optimize.ts`
- `src/splat3d_page.ts`
- `src/splat3d.html`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`

Diff live code against this fork:

```bash
node experiments/clip_forks/diff_fork.mjs v06_view_sampling
```

## Quality Gate

Compare schedules after equal wall-clock and equal CLIP-call budgets:

- prompt fidelity;
- cross-view consistency;
- detail level;
- black-background leakage;
- mean/min all-view cosine if available.
