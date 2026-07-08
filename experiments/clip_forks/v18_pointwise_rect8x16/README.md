# v18 Pointwise Rect8x16 Tile

Date: 2026-07-08

## Goal

Try an exact-math pointwise forward tile that gives each workgroup a wider
output-channel tile:

```text
baseline pointwise tile:   8 pixel-quads x  8 cout-quads = 32 cout
rect8x16 pointwise tile:   8 pixel-quads x 16 cout-quads = 64 cout
```

This is not a full CLIP rewrite and not a precision change. It is a gated WGSL
tile variant for selected forward pointwise and `pointwise + GELU` steps.
Default behavior is unchanged.

## Runtime Gate

Environment switches:

```bash
PW_TILE_VARIANT=rect8x16
PW_TILE_STEPS=57,59,62,64
```

Matrix tokens:

```bash
pwrect
pwsteps57-59-62-64
```

The broad allowlist tested in this fork targeted the repeated
`256<->768 @16x16` forward pointwise family:

```text
57,59,62,64,67,69,72,74,77,79,82,84,87,89,92,94,97,99,102,104
```

## Implementation

Primary files:

- `src/clip/vision_wgsl.ts`
- `src/clip/vision_batch_wgsl.ts`
- `src/splat3d/optimize.ts`
- `tools/clip/batch_major_train_bench.ts`
- `tools/clip/batch_major_train_matrix.ts`
- `tools/clip/dispatch_profile.ts`
- `tools/clip/fused_test.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `tools/splat3d/cadence_quality.ts`
- `tools/splat3d/grid_quality.ts`

The new code path adds:

- `PointwiseTileVariant = "default" | "rect8x16"`
- `DispatchOptions.pointwiseTileVariant`
- `DispatchOptions.pointwiseTileSteps`
- `pointwiseRect8x16Main()`
- `pointwiseRect8x16()`
- `pointwiseRect8x16FusedGelu()`

The rect tile uses:

```text
workgroup_size = 8 x 16 = 128 threads
xS              = 256 vec4f
wS              = 512 vec4f
workgroup mem   = 12 KB
workgroups      = [P4 / 8, cout / 64, batch]
```

It keeps the same mathematical outputs and f32 accumulation. It only changes
how selected pointwise forward tiles stage activations and weights.

## Snapshot

The `snapshot/` directory contains the live source files copied before this
fork was edited. Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v18_pointwise_rect8x16
```

## Correctness Gates

Narrow train-forward gate:

```bash
PLAN=plan_train.json PW_TILE_VARIANT=rect8x16 PW_TILE_STEPS=57,59 BENCH_RUNS=1 bun tools/clip/fused_test.ts
```

Result: passed, final embedding cosine `1.000000`.

Broad train-forward gate:

```bash
PLAN=plan_train.json PW_TILE_VARIANT=rect8x16 PW_TILE_STEPS=57,59,62,64,67,69,72,74,77,79,82,84,87,89,92,94,97,99,102,104 BENCH_RUNS=1 bun tools/clip/fused_test.ts
```

Result: passed, final embedding cosine `1.000000`.

Batch-major train gradient gate:

```bash
PW_TILE_VARIANT=rect8x16 PW_TILE_STEPS=57,59,62,64,67,69,72,74,77,79,82,84,87,89,92,94,97,99,102,104 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
```

Result: passed for lanes `0/1/2`, gradient cosine `1.000000`, relLinf `0`.

## CLIP-Only Timing

Command:

```bash
TRIALS=3 RUNS=3 WARMUP=2 BATCH=3 CONFIGS='base=stem,gelu;rect16=stem,gelu,pwrect,pwsteps57-59-62-64-67-69-72-74-77-79-82-84-87-89-92-94-97-99-102-104' bun tools/clip/batch_major_train_matrix.ts
```

| Variant | Batch Median | Range | Image Median |
| --- | ---: | ---: | ---: |
| `base` | `41.63 ms` | `41.15..41.69` | `13.88 ms` |
| `rect16` | `41.43 ms` | `40.69..41.99` | `13.81 ms` |

Read: effectively flat. This is not enough to promote.

## Timestamp Attribution

Same-session isolated timestamp runs, B=3, one warmed run:

| Group | Base | Rect16 |
| --- | ---: | ---: |
| total | `79.17 ms` | `100.73 ms` |
| `pw_bwd` | `21.76 ms` | `21.10 ms` |
| `pw` | `10.29 ms` | `16.38 ms` |
| `pw+gelu` | `9.70 ms` | `16.97 ms` |
| `spatial_bwd` | `16.38 ms` | `15.07 ms` |
| `conv` | `8.72 ms` | `17.69 ms` |

Read: the targeted forward pointwise families got worse in this timestamp
profile. One timestamp run is noisy, but the direction does not support
promotion.

## Integrated 3D Timing

Broad allowlist:

```bash
TRIALS=3 RUNS=5 WARMUP=3 CONFIGS=base=3:3,rect16=3:3:pwrect:pwsteps57-59-62-64-67-69-72-74-77-79-82-84-87-89-92-94-97-99-102-104 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `base` | `55.99 ms` | `58.76 ms` | `42.55 ms` | `13.54 ms` |
| `rect16` | `55.92 ms` | `61.51 ms` | `44.53 ms` | `14.33 ms` |

Narrow first-four allowlist:

```bash
TRIALS=3 RUNS=5 WARMUP=3 CONFIGS=base=3:3,rect4=3:3:pwrect:pwsteps57-59-62-64 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `base` | `54.78 ms` | `61.88 ms` | `41.19 ms` | `14.23 ms` |
| `rect4` | `55.41 ms` | `61.73 ms` | `42.07 ms` | `14.46 ms` |

## Decision

Keep `rect8x16` gated and do not promote.

The fork is still useful because it proves the new tile plumbing, step allowlist
plumbing, and matrix-token plumbing work. But the measured variant is neutral
to worse. The next larger pointwise attempt should not just widen the
workgroup. Better candidates are:

- true dual-output-channel-per-thread accumulation to amortize staged `xS`;
- split-K for selected `pw_bwd` high-channel low-spatial shapes;
- narrow pointwise-only f16 storage or activation experiments with a full
  `dL/dimage` gate.
