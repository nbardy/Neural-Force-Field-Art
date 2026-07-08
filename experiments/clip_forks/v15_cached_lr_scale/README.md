# v15 Cached LR Scale

Date: 2026-07-08

## Hypothesis

The v14 quality gate showed that naive cached `dL/dimage` cadence is fast but
quality-regressive. One likely cause is that stale CLIP image gradients are
applied too aggressively on cached steps.

This fork adds a narrow gate:

```bash
CLIP_CACHED_LR_SCALE=0.25
```

Only cached-gradient steps use the scaled splat Adam learning rates. Refresh
steps keep the normal learning rates. Default behavior is unchanged because the
scale defaults to `1`.

## Snapshot

The `snapshot/` directory contains:

- `src/splat3d/optimize.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `tools/splat3d/cadence_quality.ts`
- `docs/SPLAT3D_PERF_NOTES.md`
- `experiments/clip_forks/README.md`

Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v15_cached_lr_scale
```

## Usage

Single bench:

```bash
CLIP_REFRESH_INTERVAL=4 CLIP_CACHED_LR_SCALE=0.25 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
```

Matrix token:

```bash
TRIALS=3 CONFIGS=base=3:3,cache4=3:3:cache4,cache4lr25=3:3:cache4:lr0.25 RUNS=8 WARMUP=6 bun tools/splat3d/step_matrix.ts
```

Quality token:

```bash
BUDGET_MS=5000 CONFIGS=base=1,cache2=2,cache2lr50=2:0.5,cache4=4,cache4lr25=4:0.25 bun tools/splat3d/cadence_quality.ts
```

## Measurements

Adapter: Apple `metal-3`, `apple-m4`, non-fallback.

Config: `G=4096`, `views=3`, `clipBatch=3`, `cap=2048`, `seed=1`,
`viewSampler=epoch`, f32 weights.

### Fixed-Wall-Clock Full-Teacher Score

```bash
BUDGET_MS=5000 CONFIGS=base=1,cache2=2,cache2lr50=2:0.5,cache4=4,cache4lr25=4:0.25 OUT_DIR=/tmp/nffa_cadence_lr_quality bun tools/splat3d/cadence_quality.ts
```

| Variant | Refresh Interval | Cached LR Scale | Steps / 5s | Steps / Sec | Mean Full-Teacher Cos | Min Cos | Max Cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 1 | 1.00 | 87 | 17.26 | 0.09413 | 0.04752 | 0.14648 |
| `cache2` | 2 | 1.00 | 149 | 29.49 | 0.04559 | 0.00896 | 0.09123 |
| `cache2lr50` | 2 | 0.50 | 142 | 28.37 | 0.05842 | 0.01546 | 0.09793 |
| `cache4` | 4 | 1.00 | 213 | 42.16 | 0.02450 | -0.01104 | 0.07710 |
| `cache4lr25` | 4 | 0.25 | 217 | 43.32 | 0.04000 | 0.00737 | 0.08517 |

Artifacts:

- `results/2026-07-08/cadence_quality.json`
- `results/2026-07-08/base_views.png`
- `results/2026-07-08/cache2_views.png`
- `results/2026-07-08/cache2lr50_views.png`
- `results/2026-07-08/cache4_views.png`
- `results/2026-07-08/cache4lr25_views.png`

### Step Matrix

```bash
TRIALS=3 CONFIGS=base=3:3,cache2=3:3:cache2,cache2lr50=3:3:cache2:lr0.5,cache4=3:3:cache4,cache4lr25=3:3:cache4:lr0.25 RUNS=8 WARMUP=6 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `base` | 53.64 ms | 57.31 ms | 41.94 ms | 13.31 ms |
| `cache2` | 32.90 ms | 56.65 ms | 41.81 ms | 12.76 ms |
| `cache2lr50` | 32.69 ms | 57.28 ms | 41.64 ms | 12.74 ms |
| `cache4` | 22.61 ms | 15.01 ms | 0.00 ms | 12.51 ms |
| `cache4lr25` | 22.75 ms | 14.53 ms | 0.00 ms | 12.68 ms |

## Decision

Do not promote as-is.

Cached-step LR scaling is a useful improvement:

- `cache2lr50` improves mean full-teacher cosine by `28%` over naive `cache2`
  (`0.04559 -> 0.05842`) with no step-speed cost.
- `cache4lr25` improves mean full-teacher cosine by `63%` over naive `cache4`
  (`0.02450 -> 0.04000`) with no step-speed cost.

But both remain far below base (`0.09413`). Stale gradients are part of the
problem, not the whole problem. The next cadence attempt should change camera
coverage or objective semantics, not only step size.
