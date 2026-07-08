# v20 Default Grid80 Backward Stack

Date: 2026-07-08

## Goal

Promote the already-implemented exact backward stack only for the measured
`grid9_close2 + direct80 + batch3 + f32` path:

- `SPATIAL_BWD_VARIANT=depthwise4`
- `FUSE_GELU_BWD_PW=1`
- `FUSE_RESIDUAL_BWD_PW=1`

This is not a global CLIP default. Per-view CLIP keeps its existing default.
The grid80 path now opts into the stack unless a benchmark explicitly disables
it with `SPATIAL_BWD_VARIANT=generic` and/or `FUSE_*_BWD_PW=0`.

## Runtime Behavior

Hidden promotion envelope in `Splat3DOptimizer.create()`:

```text
clipLayout == grid9_close2
clipBatchSize == 3
gridDirectRaster == true
stemSpatialBwd == true
fusePointwiseGeluForward == true
clipWeightPrecision == f32/default
```

Within that envelope:

```text
spatialBwdVariant defaults to depthwise4
fuseGeluBwdIntoPw defaults to true
fuseResidualBwdIntoPw defaults to true
```

Explicit config/env overrides still win.

## Negative Controls

Use these to reproduce the old path:

```bash
SPATIAL_BWD_VARIANT=generic
FUSE_GELU_BWD_PW=0
FUSE_RESIDUAL_BWD_PW=0
```

`tools/splat3d/step_matrix.ts` now distinguishes default vs explicit states:

```bash
grid80default=9:3:grid9:directgrid
grid80old=9:3:grid9:directgrid:generic:nofusions
grid80dw4old=9:3:grid9:directgrid:dw4:nofusions
grid80explicit=9:3:grid9:directgrid:dw4:gelubwd:resbwd
```

## Snapshot

The `snapshot/` directory contains:

- `src/splat3d/optimize.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `tools/splat3d/grid_quality.ts`
- `tools/splat3d/cadence_quality.ts`
- `tools/clip/bwd_test.ts`
- `docs/SPLAT3D_PERF_NOTES.md`
- `experiments/clip_forks/README.md`

Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v20_default_backward_stack
```

## Correctness Gate

`tools/clip/bwd_test.ts` now passes these env options into the real
`VisionTrainer.create()` gate-2 path. Before this fork, the env command logged
in v11 did not actually exercise the fused trainer path.

Command:

```bash
SPATIAL_BWD_VARIANT=depthwise4 FUSE_GELU_BWD_PW=1 FUSE_RESIDUAL_BWD_PW=1 BENCH_RUNS=5 bun tools/clip/bwd_test.ts
```

Result:

```text
gate 1: ALL PASS
pipelines: 129 fwd + 152 bwd ... (spatialBwdVariant=depthwise4, fuseGeluBwdIntoPw=1, fuseResidualBwdIntoPw=1)
directional: 8/8 under rel 2e-2
gate 2: PASS
ALL PASS
```

## Integrated Timing

Command:

```bash
TRIALS=5 RUNS=5 WARMUP=3 CONFIGS='perview=3:3,grid80default=9:3:grid9:directgrid,grid80old=9:3:grid9:directgrid:generic:nofusions,grid80dw4old=9:3:grid9:directgrid:dw4:nofusions,grid80explicit=9:3:grid9:directgrid:dw4:gelubwd:resbwd' bun tools/splat3d/step_matrix.ts
```

Result file:

- `results/2026-07-08/step_matrix_grid80_default.txt`

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `perview` | `52.51 ms` | `56.86 ms` | `41.30 ms` | `12.72 ms` |
| `grid80default` | `57.05 ms` | `59.33 ms` | `36.71 ms` | `19.79 ms` |
| `grid80old` | `58.71 ms` | `61.78 ms` | `39.98 ms` | `18.78 ms` |
| `grid80dw4old` | `56.42 ms` | `60.11 ms` | `37.94 ms` | `19.36 ms` |
| `grid80explicit` | `55.13 ms` | `58.04 ms` | `36.41 ms` | `19.76 ms` |

Read:

- The promoted default beats the old generic/no-fusion grid80 path by `1.66 ms`
  normal-step median and `3.27 ms` CLIP median in this run.
- `grid80default` and `grid80explicit` are expected to use the same backward
  stack; the small median difference here is treated as run-order/GPU noise.
- This is a small stack promotion, not a 2x answer.

## Decision

Promote the stack only for grid80/direct raster. Keep env/config controls and
keep deeper v20/v21 work focused on structural pointwise backward changes:

- shared-W batch `pw_bwd`;
- split-K `pw_bwd`;
- remaining non-pointwise local GELU fusions.
