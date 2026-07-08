# v14 Cadence Quality Gate

Date: 2026-07-08

## Hypothesis

`CLIP_REFRESH_INTERVAL` already showed a 2x-class wall-clock step-rate win, but
the speed number alone is not enough. Cached `dL/dimage` is a proxy schedule,
not the same objective as full CLIP every step.

This fork adds a fixed-budget quality harness:

```bash
BUDGET_MS=5000 CONFIGS=base=1,cache2=2,cache4=4 bun tools/splat3d/cadence_quality.ts
```

For each config the tool:

- starts from the same deterministic splat initialization;
- optimizes for the same wall-clock budget;
- evaluates all 9 camera views with the full frozen CLIP image tower;
- reports cosine against the deterministic synthetic per-view prompt embedding;
- writes a 3x3 PNG contact sheet per config and a JSON result file.

The synthetic prompt embedding is the same deterministic test embedding used by
`step_bench.ts`. This is a convergence/regression signal, not a semantic prompt
quality judgment.

## Snapshot

The `snapshot/` directory contains the current optimizer, step benchmark, step
matrix, fork index, and perf notes from the start of this fork.

Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v14_cadence_quality_gate
```

## Measurement

Command:

```bash
BUDGET_MS=5000 CONFIGS=base=1,cache2=2,cache4=4 bun tools/splat3d/cadence_quality.ts
```

Adapter: Apple `metal-3`, `apple-m4`, non-fallback.

Config: `G=4096`, `views=3`, `clipBatch=3`, `cap=2048`, `seed=1`,
`viewSampler=epoch`, f32 weights.

| Variant | Refresh Interval | Steps / 5s | Steps / Sec | Mean Full-Teacher Cos | Min Cos | Max Cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 1 | 88 | 17.59 | 0.09336 | 0.04962 | 0.15112 |
| `cache2` | 2 | 147 | 29.33 | 0.04363 | 0.01030 | 0.07999 |
| `cache4` | 4 | 221 | 44.11 | 0.02805 | -0.01557 | 0.07909 |

Artifacts:

- `results/2026-07-08/cadence_quality.json`
- `results/2026-07-08/base_views.png`
- `results/2026-07-08/cache2_views.png`
- `results/2026-07-08/cache4_views.png`

## Decision

Do not promote naive cached CLIP gradient cadence.

The speedup is real, but the fixed-wall-clock full-teacher score regressed in
this first quality gate. `cache4` ran `2.51x` as many optimizer steps as base,
yet reached only about `30%` of base's mean teacher cosine.

Likely reasons:

- cached image gradients are stale after splat parameters move;
- this fork repeats the cached view set between refreshes, so camera coverage
  advances only on refresh steps;
- the same learning rates may be too high when one gradient is applied multiple
  times.

Next smarter variants, if we return to this lane:

- scale the raster/Adam learning rate on cached steps;
- refresh image gradients every other step only for early coarse stages;
- alternate cached steps with different regularizers instead of repeating the
  same stale view gradient;
- use a periodic full 9-view teacher refresh as the promotion gate.
