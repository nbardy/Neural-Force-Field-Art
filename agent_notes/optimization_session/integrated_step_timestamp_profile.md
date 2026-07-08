# Integrated Step Timestamp Profile

Date: 2026-07-08

## Question

Can the 3D optimizer's sampled profile report real GPU timestamp timings for
the integrated raster -> CLIP -> raster backward -> Adam -> display schedule?

## Change

`tools/splat3d/step_bench.ts` now accepts `TIMESTAMP=1`. When the adapter
supports `timestamp-query`, it requests that feature and asks
`Splat3DOptimizer.profileStep()` to attach timestamp writes to the actual
compute passes being measured.

This is measurement-only. Normal `step()` still uses the same single-submit
training schedule. The browser page still uses split-submit wall profiling
unless a caller explicitly requests GPU timestamps.

## Failed Approach

A first marker-pass idea was tested and rejected. The idea was to insert tiny
timestamped no-op compute passes before and after a segment that records its own
passes. On this Dawn/Metal runtime, empty marker passes returned zero
timestamps, and one-thread marker passes did not measure intervening work:

```text
wall 2.9287080000000003
[ "563173866864640", "563173866864640", "563173866864640", "563173866864640" ] 0 0
```

Conclusion: do not use marker passes as segment brackets here. Timestamp writes
must be attached to the actual compute pass being timed.

## Implementation Detail

The CLIP and raster runtimes now accept optional timestamp descriptors at pass
creation sites:

- `VisionTrainer.encodeForward()`
- `VisionTrainer.encodeBackward()`
- `VisionTrainer.encode()`
- `BatchMajorVisionTrainer.encode()`
- raster forward/backward/batch-forward/batch-backward/Adam pass recorders

For the default batch-raster input path, multiple per-view raster forward passes
are timed separately and summed. This avoids pretending that one timestamp can
span several internal passes.

## Commands

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 VIEW_LANE_RASTER_BWD=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
bun tools/splat3d/raster_batch_forward_test.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

## Result

Default `3/9`, `batch CLIP x3`, promoted CLIP settings:

| Timing Mode | Normal Step Avg | Profile Total | Raster Fwd | Raster Bwd | CLIP Batch | Adam | Display |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU timestamp | `79.80 ms` | `52.82 ms` | `1.31 ms` | `10.03 ms` | `41.03 ms` | `0.00 ms` | `0.46 ms` |
| split-submit wall | `64.72 ms` | `60.13 ms` | `2.07 ms` | `11.41 ms` | `43.91 ms` | `0.34 ms` | `0.93 ms` |

The GPU timestamp profile is the better attribution signal; normal-step wall
time remains the promotion gate because it measures the real app schedule.

View-lane raster backward timestamp smoke:

| Variant | Normal Step Avg | Profile Total | Raster Fwd | Raster Bwd | CLIP Batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | `79.80 ms` | `52.82 ms` | `1.31 ms` | `10.03 ms` | `41.03 ms` |
| `VIEW_LANE_RASTER_BWD=1` | `61.93 ms` | `54.00 ms` | `1.38 ms` | `9.24 ms` | `42.86 ms` |

The raster backward timestamp improvement is under `1 ms` on this default
sample, which supports the earlier decision not to promote view-lane backward.

## Interpretation

For the current default path, CLIP batch remains the dominant integrated GPU
segment. Raster backward is real but secondary. Adam/display/clear are too small
to spend optimization time on right now.
