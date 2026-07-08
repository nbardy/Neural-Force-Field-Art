# v10 Shared-W Pointwise Forward

## Hypothesis

Batch-major CLIP dispatches pointwise forward kernels with `workgroups.z = B`.
That means each batch lane stages the same weight tile independently. A shared-W
kernel puts the batch lanes inside `local_invocation_id.z`, stages one `W` tile,
and stages separate `X` tiles per lane.

The implementation already existed as `pointwiseSharedWBatchForwardDispatch`.
This fork wires the allowlist into the 3D optimizer bench path so it can be
tested in the real splat schedule.

Gate:

```bash
SHARED_W_FWD_STEPS=10,15,24,34,49
```

Step-matrix token:

```bash
sw10-15-24-34-49
```

## Snapshot

`snapshot/` contains the batch pointwise emitters, optimizer wiring, CLIP/3D
benchmark tools, and notes copied from `HEAD` before this fork was edited.

Diff from snapshot:

```bash
node experiments/clip_forks/diff_fork.mjs v10_shared_w_pointwise_forward
```

## Microbench Scan

Command:

```bash
TRIALS=1 BATCHES=3 STEPS=4,8,10,13,15,18,22,24,27,29,32,34,37,39,42,44,47,49,53,57,59,62,64,67,69,72,74,77,79,82,84,87,89,92,94,97,99,102,104,108,111,113,115,117,118,120,122,124 RUNS=10 WARMUP=3 bun tools/clip/pointwise_batch_matrix.ts
```

Takeaway: shared-W parity passed, but most hot pointwise shapes were flat or
slower. A small residual/flat allowlist looked least bad:

```text
10,15,24,34,49
```

## CLIP-Only Gate

Command:

```bash
TRIALS=5 BATCH=3 RUNS=5 WARMUP=3 CONFIGS='base=stem,gelu;residuals=stem,gelu,10,15,24,34,49' bun tools/clip/batch_major_train_matrix.ts
```

Result:

| Variant | Batch Median | Image Median |
| --- | ---: | ---: |
| baseline | `40.86 ms` | `13.62 ms` |
| residual shared-W | `40.65 ms` | `13.55 ms` |

This is only about a `0.5%` CLIP-only win.

## Integrated 3D Gate

Ordinary `3/9` path:

```bash
TRIALS=5 CONFIGS=base=3:3,swres=3:3:sw10-15-24-34-49 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| baseline | `52.88 ms` | `57.03 ms` | `41.30 ms` | `13.70 ms` |
| shared-W residuals | `53.27 ms` | `57.04 ms` | `40.99 ms` | `13.42 ms` |

Grid80 + depthwise4 stack:

```bash
TRIALS=3 CONFIGS=grid80dw4=9:3:grid9:directgrid:dw4,grid80dw4sw=9:3:grid9:directgrid:dw4:sw10-15-24-34-49 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| grid80 + dw4 | `55.47 ms` | `57.76 ms` | `36.20 ms` | `18.99 ms` |
| grid80 + dw4 + shared-W | `56.41 ms` | `57.86 ms` | `36.02 ms` | `19.45 ms` |

## Decision

Do not promote. Keep the env gate and matrix token because it is useful
diagnostically, but shared-W forward pointwise does not improve integrated 3D
step time on the measured paths.

Reason: reducing repeated W staging is not enough to compensate for larger
3D workgroups, more workgroup memory, and the interaction with the existing
pointwise+GELU fusion. The next pointwise fork should target the base tile shape
or backward pointwise, not this shared-W forward allowlist.
