# 3D Splat Collapse And Raster Replay Session

Date: 2026-07-10

## Findings

1. The hard black center square was tile overflow. At `G=4096`, a concentrated
   scene reached tile count 4096 while cap 2048 dropped 21,387 of 65,659 pairs.
2. Raising cap to 4096 removed the artifact but made the real collapsed-scene
   workload visible. The browser overlay later showed `3730/4096` splats in the
   hottest tile, so per-tile bitonic sort and backward dominated.
3. `centerWeight` was implemented as per-splat L2 shrinkage. It now uses an i32
   atomic centroid reduction and gives every splat the same translation
   gradient. The regression gate reports zero pairwise gradient delta.
4. `coverage weak` was the primary remaining tiny-ball cause. The current loss
   is `mean((coverage - target)^2)` per pixel. It contracts cloud RMS spread
   from `0.9031` to `0.3352` in 40 steps; all regularizers off retains `0.8733`.
5. Dream-ish now defaults to coverage off. The toggle remains for ablation, but
   this formulation should not be described as the final Dream Fields loss.

## Raster Pass Accounting

The default `grid9_close2` objective used 11 training forwards, 9 replay
forwards, 11 backwards, and one display forward per optimizer step. Native
single-forward 3DGS throughput is therefore not a direct comparison.

At 80px, retaining nine independent grid scratch states costs about 5.2 MB and
removes the nine replay forwards. 256px and 512px grid modes keep replay to
avoid large memory growth. `RETAIN_GRID_STATE=0` is the rollback/benchmark
switch.

## Gates

- `bun tools/splat3d/regularizer_test.ts`
- `bun tools/splat3d/grid_retain_parity.ts`
- `GRID_DIRECT_RASTER=1 bun tools/splat3d/grid9_close2_test.ts`
- `bun tools/splat3d/grid9_close2_test.ts`
- `bun tools/splat3d/raster_batch_forward_test.ts`

The retain parity gate reports max parameter difference `0.000e+0`.

## Performance Interpretation

A cold retained sample was `48.56 ms` total: raster forward `3.80`, replay
`0.00`, raster backward `12.45`, CLIP batch `31.85`, display `0.46`. A later
throttled replay sample spent `10.88 ms` in replay. Do not quote the wall-time
difference between those runs as a clean speedup because CLIP timing also
shifted with GPU contention.
