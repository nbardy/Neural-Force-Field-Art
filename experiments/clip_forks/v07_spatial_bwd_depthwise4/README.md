# v07 Spatial Backward Depthwise4

## Hypothesis

CLIP training spends a meaningful share of time in depthwise `spatial_bwd`
dispatches. The current generic gather shader computes one input pixel per
thread, which repeats nearby `dy` and weight reads across adjacent horizontal
pixels. A depthwise-only `vec4f` lane can compute four adjacent input pixels per
thread and reduce dispatch work for the layers that satisfy:

- `groups === cin`
- `cout === cin`
- `stride` is `1` or `2`
- input height and width are divisible by `4`

This fork is gated by `SPATIAL_BWD_VARIANT=depthwise4`.

## Snapshot

`snapshot/` contains the relevant CLIP backward emitters, 3D optimizer wiring,
and benchmark tools copied from `HEAD` before the fork was edited.

Diff from snapshot:

```bash
node experiments/clip_forks/diff_fork.mjs v07_spatial_bwd_depthwise4
```

## Correctness Gate

```bash
bun tools/clip/bwd_test.ts
SPATIAL_BWD_VARIANT=depthwise4 bun tools/clip/bwd_test.ts
```

## Timing Gate

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 SPATIAL_BWD_VARIANT=depthwise4 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
TRIALS=2 CONFIGS=base=3:3,dw4=3:3:dw4 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

## Result

Correctness passed:

```text
bun tools/clip/bwd_test.ts
gate 1: ALL PASS
gate 2: PASS
ALL PASS
```

Isolated CLIP timestamp profile was mixed:

| Variant | Total Isolated Sum | `spatial_bwd` |
| --- | ---: | ---: |
| baseline | `69.992 ms` | `15.466 ms` / `22.1%` |
| `SPATIAL_BWD_VARIANT=depthwise4` | `75.760 ms` | `13.631 ms` / `18.0%` |

The targeted bucket improved by about `11.9%`, but isolated total got worse in
that run because other buckets moved up. Treat isolated total as noisy for this
small fork.

Integrated 3D step matrix was consistently positive:

```text
TRIALS=5 CONFIGS=base=3:3,dw4=3:3:dw4 RUNS=7 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| baseline | `53.12 ms` | `54.37 ms` | `40.00 ms` | `11.59 ms` |
| `depthwise4` | `49.96 ms` | `51.80 ms` | `38.24 ms` | `11.26 ms` |

Decision: keep gated for now. The integrated win is real enough to preserve and
continue testing, but it is a modest ~6% step win, not a 2x CLIP leap.
