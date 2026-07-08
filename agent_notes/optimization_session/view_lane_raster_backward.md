# View-Lane Raster Backward Ablation

Date: 2026-07-08

## Question

The forward-only view-lane raster path was exact but not faster. Does the idea
become useful if the tile backward pass also runs over `workgroup_id.z = lane`?

## Implementation

- Added `backwardBatchShader3D`.
- Added lane-strided `accGrad` to the view-lane raster state.
- Added per-lane clear bindings so old `recordBackwardAdd()` still clears the
  correct `accGrad` slice when it is given lane-strided state.
- Added `recordBatchBackwardAdd()`:
  - clear active `accGrad` lanes;
  - run one batched tile backward dispatch;
  - run the existing camera-specific `chainAdd` dispatch sequentially per lane.
- Added `VIEW_LANE_RASTER_BWD=1` and matrix token `viewbwd`.

This avoids changing the raw parameter gradient accumulation contract. The risky
part, `chainAdd`, remains ordered and camera-specific.

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
batch backward diff: max=0.000e+0 mean=0.000e+0
GATE PASS: view-lane raster forward/backward matches separate per-view path.
```

There was one useful catch during implementation: lane-sliced backward initially
reused the shared `accGrad` clear binding. That made separate per-lane backward
over the new state accumulate stale lane gradients. Adding per-lane
`clearGradsBind` fixed it.

## Timing

Default 3-view chunk:

```bash
TRIALS=3 CONFIGS=base=3:3,viewbwd=3:3:viewbwd,viewboth=3:3:viewlane:viewbwd RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

```text
base      normal 52.69 ms  profile 56.81 ms  clip 40.50 ms  raster 13.45 ms
viewbwd   normal 53.37 ms  profile 57.37 ms  clip 41.35 ms  raster 12.63 ms
viewboth  normal 53.44 ms  profile 57.60 ms  clip 41.13 ms  raster 12.73 ms
```

All 9 views with batch size 3:

```bash
TRIALS=2 CONFIGS=base=9:3,viewbwd=9:3:viewbwd,viewboth=9:3:viewlane:viewbwd RUNS=3 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

```text
base      normal 157.01 ms  profile 162.59 ms  clip 121.95 ms  raster 37.66 ms
viewbwd   normal 157.41 ms  profile 163.03 ms  clip 124.41 ms  raster 35.00 ms
viewboth  normal 152.42 ms  profile 163.78 ms  clip 123.90 ms  raster 36.14 ms
```

## Decision

Do not promote.

The exact batched backward path is worth keeping because it proves the
lane-strided `accGrad` contract and sometimes lowers sampled raster time. It
does not improve the default 3-view optimizer step, and the 9-view normal-step
win for forward+backward did not appear in split-profile total.

Next raster work should gather better evidence before more scheduling changes:

- overflow count per tile;
- tile occupancy histograms;
- active stop-count histograms;
- per-kernel timestamp queries if WebGPU timestamp support is available;
- only then workgroup staging or raw-gradient atomic rewrites.
