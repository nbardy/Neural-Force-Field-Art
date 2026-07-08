# View-Lane Raster Forward Ablation

Date: 2026-07-08

## Question

Does the exact static multi-view raster path get faster if we move camera
constants into a buffer and dispatch raster forward across
`workgroup_id.z = view lane`?

## Implementation

- Added a compact `16xf32` camera record per prepared camera.
- Added batch `prep`, `emit`, and `forward` WGSL kernels.
- Allocated combined lane-strided scratch for:
  - `derived[lane, splat, slot]`
  - `tileCounts[lane, tile]`
  - `binnedIds[lane, tile, slot]`
  - `tileStop[lane, tile]`
- Kept backward on the existing per-lane path by binding aligned slices of that
  combined scratch.
- Exposed the ablation as `VIEW_LANE_RASTER_FWD=1` and matrix token `viewlane`.

This is an exact scheduler/layout change, not a new splat object and not a
STAR-style sublinear camera-bundle primitive.

## Correctness

Command:

```bash
bun tools/splat3d/raster_batch_forward_test.ts
```

Result:

```text
adapter: apple metal-3
image diff: max=0.000e+0 mean=0.000e+0
grad diff:  max=0.000e+0 mean=0.000e+0
GATE PASS: view-lane raster forward matches separate per-view path.
```

The gradient comparison matters because backward consumes saved `binnedIds`,
`tileStop`, and `derived` from the forward pass. Matching raw splat gradients
means the batched forward's saved lane state is compatible with the existing
backward chain.

## Timing

3-view default chunk:

```bash
TRIALS=3 CONFIGS=base=3:3,viewlane=3:3:viewlane RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

```text
base      normal 52.92 ms  profile 55.90 ms  clip 41.07 ms  raster 12.81 ms
viewlane  normal 53.03 ms  profile 57.23 ms  clip 40.99 ms  raster 13.77 ms
```

9 views with batch size 3:

```bash
TRIALS=2 CONFIGS=base=9:3,viewlane=9:3:viewlane RUNS=3 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

```text
base      normal 154.26 ms  profile 161.91 ms  clip 122.32 ms  raster 36.85 ms
viewlane  normal 152.82 ms  profile 163.51 ms  clip 122.48 ms  raster 38.03 ms
```

## Decision

Do not promote.

The path is exact and useful groundwork, but forward scheduling alone did not
improve the integrated step. The 3-view default was flat on normal step and
worse in sampled profile/raster. The 9-view run had a tiny normal-step median
edge, but sampled profile and raster were still worse.

Next raster work should target a different bottleneck:

- lane-strided `accGrad` and batched backward;
- direct raw-gradient fixed-point atomics;
- overflow telemetry and cap tuning;
- workgroup staging in forward/backward after telemetry shows it is worth it.
