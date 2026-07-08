# v11 Backward Local Fusions

Date: 2026-07-08

## Hypothesis

The earlier `FUSE_GELU_BWD_PW=1` and `FUSE_RESIDUAL_BWD_PW=1` gates were tested
before the current best 3D stack. Re-test them together on:

- `CLIP_LAYOUT=grid9_close2`
- `GRID_DIRECT_RASTER=1`
- `SPATIAL_BWD_VARIANT=depthwise4`
- `CLIP_BATCH=3`
- `VIEWS=9`

The target is not a full 2x CLIP win. The target is to remove legal local
dispatches and scratch-buffer materialization without changing math.

## Snapshot

The `snapshot/` directory contains the relevant live files at fork creation:

- `src/clip/vision_bwd_wgsl.ts`
- `src/clip/vision_batch_wgsl.ts`
- `src/splat3d/optimize.ts`
- `tools/clip/bwd_test.ts`
- `tools/clip/dispatch_profile.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `docs/SPLAT3D_PERF_NOTES.md`
- `experiments/clip_forks/README.md`

Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v11_backward_local_fusions
```

## Correctness Gate

```bash
SPATIAL_BWD_VARIANT=depthwise4 FUSE_GELU_BWD_PW=1 FUSE_RESIDUAL_BWD_PW=1 bun tools/clip/bwd_test.ts
```

Result:

```text
gate 1: ALL PASS
directional: 8/8 under rel 2e-2
gate 2: PASS
ALL PASS
```

## Timestamp Profile

Baseline:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=2 bun tools/clip/dispatch_profile.ts
```

Fused:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 FUSE_GELU_BWD_PW=1 FUSE_RESIDUAL_BWD_PW=1 MODE=train BATCH=3 RUNS=3 WARMUP=2 bun tools/clip/dispatch_profile.ts
```

| Variant | Dispatches | Isolated Timestamp Sum | Main Groups |
| --- | ---: | ---: | --- |
| `depthwise4` | `257` | `86.704 ms` | `pw_bwd`, `conv`, `spatial_bwd`, `pw`, `pw+gelu` |
| `depthwise4 + both fusions` | `211` | `54.591 ms` | `conv`, `spatial_bwd`, `pw`, `pw+gelu`, fused `pw_bwd` groups |

Interpret isolated timestamp sums cautiously; they overstate the integrated
gain. They are still useful because they show the dispatch count and local
materialization reduction are real.

## Integrated Gate

Focused matrix:

```bash
TRIALS=7 CONFIGS=grid80dw4=9:3:grid9:directgrid:dw4,grid80dw4both=9:3:grid9:directgrid:dw4:gelubwd:resbwd RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `grid80 + depthwise4` | `56.20 ms` | `58.33 ms` | `36.41 ms` | `19.51 ms` |
| `grid80 + depthwise4 + both fusions` | `54.63 ms` | `56.81 ms` | `34.91 ms` | `19.29 ms` |

## Decision

Keep the fusions as measured, correctness-passing gates and consider enabling
them for the grid80+depthwise path after visual/interactive testing. This is a
real small CLIP win, not a standalone 2x answer.

The bigger lesson is that legal local fusion can help, but the current 2-4x
path still needs structural work: fewer full CLIP calls per useful progress,
better pointwise/spatial kernels, or a teacher/proxy schedule.
