# v13 Cached CLIP Gradient Cadence

Date: 2026-07-08

## Hypothesis

The optimizer does not update CLIP weights. CLIP is a frozen teacher that
produces `dL/dimage`; raster backward maps that image gradient to splat
parameter gradients, then Adam updates splat parameters.

So an "alternate CLIP step" can still optimize splats if it reuses a recent
full-resolution CLIP image gradient. This fork adds an env-gated benchmark
path:

```text
refresh step: render selected 256px views -> CLIP forward/backward -> cache dL/dimage lanes -> raster backward -> Adam
cached step:  render same selected 256px views -> reuse cached dL/dimage lanes -> raster backward -> Adam
```

This first version is deliberately scoped to `per_view` + one complete
`CLIP_BATCH` chunk, i.e. the default `VIEWS=3 CLIP_BATCH=3` path. Grid/contact
sheet cached gradients need separate semantics and are not included here.

## Snapshot

The `snapshot/` directory contains:

- `src/splat3d/optimize.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `docs/SPLAT3D_PERF_NOTES.md`
- `experiments/clip_forks/README.md`

Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v13_cached_clip_gradient
```

## Runtime Gate

```bash
CLIP_REFRESH_INTERVAL=2 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
CLIP_REFRESH_INTERVAL=4 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
```

Matrix tokens:

```bash
TRIALS=3 CONFIGS=base=3:3,cache2=3:3:cache2,cache4=3:3:cache4 bun tools/splat3d/step_matrix.ts
```

`CLIP_REFRESH_INTERVAL=1` is the default/current behavior.

## Correctness And Scope

This path does not claim mathematical equivalence to full CLIP every step. It
is a proxy/cadence experiment.

What is preserved:

- selected views are still rendered at full `256x256`;
- cached image gradients have the same buffer shape as current CLIP
  `dL/dpixels`;
- skipped steps still run raster forward, raster backward, and Adam;
- CLIP weights remain frozen.

What changes:

- epoch view sampling advances only on refresh steps in this first fork;
- cached steps optimize against stale `dL/dimage`;
- quality must be judged with a fixed-wall-clock full-CLIP teacher score and
  all-view screenshots before promotion.

Unsupported/fallback in this fork:

- `grid9_close2`;
- `VIEWS` larger than one complete batch chunk;
- single-CLIP paths.

Those silently stay on the full-CLIP path because the cache is only populated
when `views.length === clipBatch`.

## Measurements

Clean exact-cycle matrix:

```bash
TRIALS=5 CONFIGS=base=3:3,cache2=3:3:cache2,cache4=3:3:cache4 RUNS=8 WARMUP=6 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `base` | `53.03 ms` | `56.52 ms` | `41.00 ms` | `13.66 ms` |
| `cache2` | `32.60 ms` | `55.58 ms` | `40.96 ms` | `12.38 ms` |
| `cache4` | `22.40 ms` | `14.09 ms` | `0.00 ms` | `12.48 ms` |

Step-time speedups from the normal-step median:

- `cache2`: `1.63x`
- `cache4`: `2.37x`

The sampled profile can land on either a refresh or cached step. The normal
step average over exact refresh cycles is the cadence metric to trust.

Build gate:

```bash
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Result: passed.

## Decision

Keep this as an env-gated benchmark fork. It is the first concrete 2x-class
wall-clock lever that keeps selected CLIP inputs at full `256x256`, but it
cannot be promoted until a quality gate shows that fixed-wall-clock full-teacher
score and all-view screenshots improve or at least hold up.

Next work should add a fixed-wall-clock quality/teacher-score harness for:

- `base`;
- `cache2`;
- `cache4`;
- optional periodic all-view refresh.
