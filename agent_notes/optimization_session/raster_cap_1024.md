# Raster Cap 1024 Ablation

Date: 2026-07-08

## Question

Telemetry showed the default `4096`-splat scene never exceeded `911` tile/splat
pairs per tile under the current initialization. Can we lower the tile capacity
from `2048` to `1024` and gain speed from lower workgroup memory pressure?

## Implementation

- Added `CAP` to `tools/splat3d/step_bench.ts`.
- Added `capNNN` tokens to `tools/splat3d/step_matrix.ts`, for example:

```bash
TRIALS=3 CONFIGS=base=3:3,cap1024=3:3:cap1024 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

No app/default runtime behavior changed.

## Safety

Command:

```bash
CAP=1024 bun tools/splat3d/raster_telemetry.ts
```

Result:

```text
overflowPairs=0/576871 (0.0%)
maxCount=911
maxStop=365
```

So `1024` is exact for the measured initial default scene.

## Timing

Default 3-view chunk:

```bash
TRIALS=3 CONFIGS=base=3:3,cap1024=3:3:cap1024 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

```text
cap 2048  normal 53.00 ms  profile 58.08 ms  clip 41.70 ms  raster 13.56 ms
cap 1024  normal 53.44 ms  profile 56.78 ms  clip 40.53 ms  raster 13.83 ms
```

All 9 views with batch size 3:

```bash
TRIALS=2 CONFIGS=base=9:3,cap1024=9:3:cap1024 RUNS=3 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

```text
cap 2048  normal 156.02 ms  profile 166.22 ms  clip 123.63 ms  raster 38.69 ms
cap 1024  normal 156.21 ms  profile 165.15 ms  clip 123.58 ms  raster 38.41 ms
```

## Decision

Do not promote.

The smaller cap is safe for the measured initial default scene, but it did not
improve integrated normal-step timing. It also gives less headroom if splats
move or radii grow during optimization.

Keep the benchmark controls because they are useful for future memory/capacity
checks.
