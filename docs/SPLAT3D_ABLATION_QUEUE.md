# 3D Splat Optimization Ablation Queue

Date: 2026-07-08

## Current Baseline

- 4096 splats, 256px render, 9 cameras.
- Default training is N-of-K shuffled view sampling with `3/9` views per step.
- Current sampled wall profile is split-submit wall time, not GPU timestamp
  attribution.
- Recent measured normal step averages:
  - `9/9`: about 205 ms.
  - `5/9`: about 122 ms.
  - `3/9`: about 70 ms.
- CLIP remains the main sampled cost, so the next large win should reduce CLIP
  calls or batch CLIP more effectively.

## Commit And Measurement Rule

- Commit each landed performance change separately.
- If a trial loses, either revert in its own commit or record it here as
  rejected with the measurement that killed it.
- A local kernel microbench is not enough to promote an app behavior change.
  Promotion needs either optimizer wall-time evidence or a clear enabling reason
  such as a correctness/profiling tool.

## Queue

### 1. Batch-Major CLIP In The 3D Optimizer

Hypothesis: Batching selected camera views through `BatchMajorVisionTrainer`
will reduce 3D optimizer wall time versus repeated single-view CLIP.

Status: attempted and landed behind an opt-in UI toggle. Default remains single
CLIP until the win is more stable for the default `3/9` path.

Implementation notes:

- Add a toggled path in `src/splat3d/optimize.ts`.
- Start with B=3 chunks for the current `3/9` default.
- Keep the existing single-view path as fallback.
- Conservative first path may re-render each view before raster backward, because
  current raster state is single-view.

Verification:

```bash
BATCH=3 RUNS=2 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Promotion gate:

- B=3 batch path beats current 3-view optimizer wall time after raster overhead.
- CLIP gradient parity is preserved in the CLIP bench.
- Page still optimizes and all 9 canvases stay nonblank.

Kill/rework gate:

- If conservative raster replay erases the batch CLIP win, move to per-lane
  raster state before making it default.

Attempt 1 result:

- Implemented conservative batch-major CLIP in `Splat3DOptimizer`.
- Added `single CLIP`, `batch CLIP x3`, and `batch CLIP x9` modes to the 3D
  page.
- Added `tools/splat3d/step_bench.ts`.
- The scheduler falls back to single CLIP when the active view count is smaller
  than the selected batch size, so `batch CLIP x9` is intended for `9/9`.

Commands run:

```bash
CLIP_BATCH=1 VIEWS=3 RUNS=8 WARMUP=4 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=3 RUNS=8 WARMUP=4 bun tools/splat3d/step_bench.ts
CLIP_BATCH=1 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
CLIP_BATCH=9 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
```

Observed results:

| Views | CLIP batch | Normal step avg | Profile total | Note |
| --- | ---: | ---: | ---: | --- |
| 3/9 | 1 | 104.32 ms | 129.47 ms | second sequential run |
| 3/9 | 3 | 93.41 ms | 121.43 ms | modest win, noisy |
| 9/9 | 1 | 381.18 ms | 412.02 ms | same-session matrix |
| 9/9 | 3 | 292.10 ms | 314.28 ms | clear all-view win |
| 9/9 | 9 | 299.93 ms | 330.83 ms | clear win, not better than x3 here |

Interpretation:

- Conservative batch CLIP is useful enough to keep as an ablation toggle.
- It is not strong enough to make default while it still replays raster forward
  before each raster backward.
- The next real promotion path is per-lane raster state, not more partial-chunk
  batching.

### 2. CLIP Dispatch Profiler

Hypothesis: exact dispatch timing will prevent wasted kernel rewrites and show
whether pointwise, `spatial_bwd`, attention, or elementwise kernels are the real
next target.

Status: first isolated profiler landed. A sequential integrated matrix runner
also landed after single-run timing proved noisy.

Implementation notes:

- Add `tools/clip/dispatch_profile.ts`.
- Output CSV/JSON with label, workgroups, plan, batch, mode, and elapsed time.
- Use timestamp queries if available; otherwise use warmed split submits.

Promotion gate:

- Top 80% of CLIP time is attributable by dispatch group.
- Results are stable enough across runs to choose an allowlist.

Attempt 1 result:

- Added `tools/clip/dispatch_profile.ts`.
- It reports warmed split-submit median time per generated WGSL dispatch plus
  grouped totals. This is not full-chain timestamp attribution, but it is enough
  to rank kernel families before rewrites.

Commands run:

```bash
MODE=train BATCH=1 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Observed group totals:

| Batch | Top groups by isolated median sum |
| ---: | --- |
| 1 | `pw` 19.9%, `pw_bwd` 18.7%, `spatial_bwd` 17.1%, `conv` 14.3%, `gelu`/`gelu_bwd` 17.9% combined |
| 3 | `pw` 20.8%, `spatial_bwd` 19.6%, `pw_bwd` 19.5%, `conv` 14.7%, `gelu`/`gelu_bwd` 16.3% combined |

Interpretation:

- Shared-W pointwise and pointwise fusion are plausible, but only as measured
  shape-gated variants.
- `spatial_bwd` is a real target, especially the stem
  `spatial_bwd k3s2 3<-64 @256x256`.
- Attention backward does not justify first-wave optimization. It was about
  1.8%-1.9% of the isolated median sum in the warmed profiles.
- GELU traffic is large enough to keep fusion on the backlog, but after batch
  integration and the highest pointwise/spatial tests.

Attempt 2 result:

- Added `tools/splat3d/step_matrix.ts`.
- It runs `tools/splat3d/step_bench.ts` sequentially for a config matrix and
  summarizes median/min/max normal step, split profile, CLIP, and raster time.
- This is a benchmark-control tool for future ablations, not a runtime speed
  change.

Command run:

```bash
TRIALS=2 CONFIGS=3:1,3:3 RUNS=4 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

Observed run:

| Views | CLIP batch | Median normal | Range | Median profile | Median CLIP | Median raster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3/9 | 1 | 194.68 ms | 163.44-225.91 ms | 188.71 ms | 156.67 ms | 24.76 ms |
| 3/9 | 3 | 143.66 ms | 141.55-145.77 ms | 139.09 ms | 111.31 ms | 23.69 ms |

Interpretation:

- Current machine/GPU state is much slower than the earlier per-lane raster
  baseline, so one-off timings should not be used to judge a shader patch.
- The same-session matrix still supports `batch CLIP x3` over single CLIP for
  the default `3/9` workload.
- Future performance attempts should use `step_matrix.ts` or GPU timestamp
  queries before promotion.

### 3. Raster/CLIP Buffer Aliasing

Hypothesis: removing image and gradient copies between raster and CLIP saves
bandwidth and simplifies the batch path.

Status: landed as an enabling cleanup.

Implementation notes:

- Let raster forward write directly into a CLIP input lane buffer, or let CLIP
  bind the raster image buffer as input.
- Let raster backward read directly from a CLIP input-gradient lane.
- Assert storage-offset alignment before using dynamic offsets.

Promotion gate:

- Old and aliased paths match on one fixed view.
- Integrated wall time improves or the change is required to make batch CLIP
  profitable.

Attempt 1 result:

- Added `Raster3DIOState`, allowing raster forward/backward shaders to bind
  external image and image-gradient buffers with storage offsets.
- `Splat3DOptimizer` now renders directly into `VisionTrainer` /
  `BatchMajorVisionTrainer` input slots and reads gradients directly from CLIP
  input-gradient slots.
- This removes two full-image GPU copies per optimized view.

Commands run:

```bash
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
CLIP_BATCH=1 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Observed same-session run after aliasing:

| Views | CLIP batch | Normal step avg | Profile total | Note |
| --- | ---: | ---: | ---: | --- |
| 3/9 | 1 | 153.73 ms | 180.79 ms | GPU was running slower than earlier baseline |
| 3/9 | 3 | 135.49 ms | 140.68 ms | batch still ahead in same session |

Interpretation:

- Do not credit aliasing with a large standalone speedup; the copy cost is small
  versus CLIP.
- Keep it because it simplifies the next per-lane raster-state patch and removes
  avoidable memory traffic from the correct path.

### 4. Per-Lane Raster State For Batched Views

Hypothesis: storing `derived`, bins, sorted IDs, and tile stops per active view
lets us render several views, run batch CLIP once, and run matching backwards
without re-rendering.

Status: landed for batch CLIP lanes.

Implementation notes:

- Add state buffers shaped by `[batchLane, ...]`.
- Dispatch prep/bin/forward/backward with `workgroup_id.z = lane`.
- Keep depth sorting and compositing per view; do not attempt one cross-view
  sort.

Promotion gate:

- Matches conservative replay path.
- B=3 optimizer step beats single path by a clear margin.

Attempt 1 result:

- Added optional private raster scratch state to `Raster3DIOState`.
- Batch CLIP lanes now own private `derived`, `tileCounts`, `binnedIds`, and
  `tileStop` buffers.
- The optimizer renders selected views into batch CLIP lanes, runs batch CLIP,
  and applies raster backward from the saved lane-local tile state. The replay
  forward pass is gone for complete batch chunks.

Commands run:

```bash
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
CLIP_BATCH=1 VIEWS=3 RUNS=6 WARMUP=3 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=3 RUNS=6 WARMUP=3 bun tools/splat3d/step_bench.ts
CLIP_BATCH=1 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
CLIP_BATCH=9 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
```

Observed sequential run:

| Views | CLIP batch | Normal step avg | Profile total | Raster replay | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| 3/9 | 1 | 80.87 ms | 101.99 ms | 0.00 ms | single baseline |
| 3/9 | 3 | 58.40 ms | 62.39 ms | 0.00 ms | clear win |
| 9/9 | 1 | 195.65 ms | 215.00 ms | 0.00 ms | all-view baseline |
| 9/9 | 3 | 167.23 ms | 176.97 ms | 0.00 ms | best all-view batch mode |
| 9/9 | 9 | 235.48 ms | 295.60 ms | 0.00 ms | still worse than x3 |

Interpretation:

- Promote `batch CLIP x3` as the useful performance ablation for multi-view
  training.
- Keep `batch CLIP x9` as an experiment toggle only; on Apple `metal-3` it still
  loses despite eliminating raster replay.
- The next raster-side ablation should be camera-buffer / z-dispatch or
  workgroup staging, not more replay cleanup.

STAR UVT / world-tube read-through:

- STAR's sublinear win comes from making time a native raster dimension. One
  tube covers many coherent frames through `ma=(u,v,t)` and `q_uvt`.
- Our nine CLIP cameras are a discrete camera bundle, not a smooth time axis.
  Depth order and tile footprint are per view, so a single cross-view sort or
  shared footprint is not correct in general.
- The transferable idea is per-lane tile state plus tile-pair/direct backward.
  The representation-level ideas, such as projective rational tubes or camera
  gauges, require a different primitive contract and should be treated as a
  later research ablation.

Follow-up note:

- `agent_notes/optimization_session/multiview_raster_worldtube_review.md`

Attempt 2 result: single-pass batch raster forward was implemented but not
promoted.

Hypothesis: recording all selected batch-view raster forwards inside one compute
pass would reduce pass/pipeline churn before the larger camera-buffer /
`workgroup_id.z` rewrite.

Implementation:

- `Raster3DEngine.recordForwards()` keeps one compute pass open and encodes
  `prep -> clear bins -> emit -> forward` for each selected view/lane.
- `Splat3DOptimizer` can enable it with `singlePassBatchRasterForward`.
- Bench flags: `SINGLE_PASS_RASTER_FWD=1` and the `rasterpass` matrix token.

Initial matrices were mixed. The rerun that mattered most was after checking
whether the path should become default:

```bash
TRIALS=3 CONFIGS=base=3:3,noraster=3:3:norasterpass RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
TRIALS=2 CONFIGS=base=9:3,noraster=9:3:norasterpass RUNS=3 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

In those reruns, the promoted single-pass path was worse than the explicit old
separate-forward path:

| Views | Path | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | single-pass candidate | `80.13 ms` | `93.50 ms` | `66.33 ms` | `24.31 ms` |
| 3 | separate-forward control | `71.11 ms` | `90.13 ms` | `64.17 ms` | `23.35 ms` |
| 9 | single-pass candidate | `316.73 ms` | `394.44 ms` | `294.43 ms` | `95.46 ms` |
| 9 | separate-forward control | `277.38 ms` | `352.81 ms` | `264.90 ms` | `83.42 ms` |

Decision:

- Keep `recordForwards()` and the `SINGLE_PASS_RASTER_FWD=1` gate as a tool for
  future raster scheduling experiments.
- Do not enable it by default. The result is too noisy and the default-promotion
  reruns favored the existing separate-forward path.
- The next raster attempt should skip this shallow pass-level optimization and
  move to the camera-buffer / view-lane-dispatch design.

Attempt 3 result: view-lane raster forward was implemented but not promoted.

Hypothesis: moving camera constants into a storage buffer and dispatching
forward raster work across `workgroup_id.z = view lane` would reduce baked-camera
pipeline switching and bind churn for batch CLIP chunks.

Implementation:

- Added a compact camera buffer with one `16xf32` record per prepared camera.
- Added batch-specific `prep`, `emit`, and `forward` WGSL kernels that index
  lane-strided `derived`, `tileCounts`, `binnedIds`, `tileStop`, and image
  lanes.
- Added `Raster3DEngine.createBatchForwardState()` so the batched forward owns
  one combined scratch allocation while backward can still bind each lane slice
  through aligned storage offsets.
- Added `Raster3DEngine.recordBatchForward()` and the
  `VIEW_LANE_RASTER_FWD=1` / `viewlane` benchmark gate.
- Added `tools/splat3d/raster_batch_forward_test.ts` to compare images and raw
  gradients against the old separate per-view path.

Correctness:

```bash
bun tools/splat3d/raster_batch_forward_test.ts
```

Result:

```text
image diff: max=0.000e+0 mean=0.000e+0
grad diff:  max=0.000e+0 mean=0.000e+0
GATE PASS
```

Performance:

```bash
TRIALS=3 CONFIGS=base=3:3,viewlane=3:3:viewlane RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
TRIALS=2 CONFIGS=base=9:3,viewlane=9:3:viewlane RUNS=3 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

| Views | Path | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | separate-forward control | `52.92 ms` | `55.90 ms` | `41.07 ms` | `12.81 ms` |
| 3 | view-lane forward | `53.03 ms` | `57.23 ms` | `40.99 ms` | `13.77 ms` |
| 9 | separate-forward control | `154.26 ms` | `161.91 ms` | `122.32 ms` | `36.85 ms` |
| 9 | view-lane forward | `152.82 ms` | `163.51 ms` | `122.48 ms` | `38.03 ms` |

Decision:

- Keep the exact view-lane forward path as a gated ablation and as groundwork
  for a later true batched backward experiment.
- Do not enable it by default. It was exact, but sampled raster/profile timing
  was worse and normal-step wins were flat/noisy.
- The next raster-side work should not keep shaving forward scheduling. If
  raster remains worth attacking, move to lane-strided `accGrad`/batched
  backward or overflow/workgroup-staging telemetry.

Attempt 4 result: view-lane raster backward was implemented but not promoted.

Hypothesis: the costly part of the raster lane path is not forward scheduling
alone but repeated backward tile passes and `accGrad` clears. A lane-strided
`accGrad` plus one `workgroup_id.z = view lane` backward dispatch could reduce
that overhead while keeping `chainAdd` sequential for correctness.

Implementation:

- Added a batched backward WGSL kernel that reads lane-strided CLIP image
  gradients, tile counts, sorted IDs, tile stops, and derived splat data.
- Added a lane-strided `accGrad` buffer to the view-lane raster state.
- `recordBatchBackwardAdd()` clears the active `accGrad` lanes, runs one
  batched tile-backward dispatch, then runs the existing camera-specific
  `chainAdd` dispatch once per lane.
- Exposed with `VIEW_LANE_RASTER_BWD=1` and matrix token `viewbwd`.
- The existing `tools/splat3d/raster_batch_forward_test.ts` now checks both
  view-lane forward and view-lane backward against the separate per-view path.

Correctness:

```bash
bun tools/splat3d/raster_batch_forward_test.ts
```

Result:

```text
image diff: max=0.000e+0 mean=0.000e+0
grad diff:  max=0.000e+0 mean=0.000e+0
batch backward diff: max=0.000e+0 mean=0.000e+0
GATE PASS
```

Performance:

```bash
TRIALS=3 CONFIGS=base=3:3,viewbwd=3:3:viewbwd,viewboth=3:3:viewlane:viewbwd RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
TRIALS=2 CONFIGS=base=9:3,viewbwd=9:3:viewbwd,viewboth=9:3:viewlane:viewbwd RUNS=3 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

| Views | Path | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | separate-forward/backward control | `52.69 ms` | `56.81 ms` | `40.50 ms` | `13.45 ms` |
| 3 | view-lane backward | `53.37 ms` | `57.37 ms` | `41.35 ms` | `12.63 ms` |
| 3 | view-lane forward + backward | `53.44 ms` | `57.60 ms` | `41.13 ms` | `12.73 ms` |
| 9 | separate-forward/backward control | `157.01 ms` | `162.59 ms` | `121.95 ms` | `37.66 ms` |
| 9 | view-lane backward | `157.41 ms` | `163.03 ms` | `124.41 ms` | `35.00 ms` |
| 9 | view-lane forward + backward | `152.42 ms` | `163.78 ms` | `123.90 ms` | `36.14 ms` |

Decision:

- Keep the exact batched backward path and parity test as a gated ablation.
- Do not enable it by default. It reduces sampled raster median, but default
  3-view normal/profile timing regressed, and the 9-view normal-step win for
  forward+backward did not show up in split-profile total.
- The next raster attempt should use telemetry before another scheduler rewrite:
  overflow counts, tile occupancy histograms, and possibly per-kernel timestamp
  queries if available.

### 5. Shape-Gated Shared-W Pointwise

Hypothesis: some pointwise CLIP layers can reuse a staged weight tile across
batch lanes and improve full CLIP train time.

Implementation notes:

- Start with all allowlists off.
- Enable only shapes that are hot in the dispatch profile and win in a full
  CLIP chain.
- Do not apply globally; existing microbench results are mixed.
- `tools/clip/pointwise_batch_matrix.ts` now runs the isolated shared-W
  microbench over shape/batch matrices and reports median ratios.

Promotion gate:

- Full B=3 batch-major train wall time improves by at least 5%.
- Gradient parity still passes.

Attempt 1 result: not promoted.

Commands:

```bash
BATCHES=3 STEPS=8,10,57,59,111,113,115,117 TRIALS=3 RUNS=20 WARMUP=5 bun tools/clip/pointwise_batch_matrix.ts
BATCHES=2 STEPS=8,10,57,59,111,113,115,117 TRIALS=3 RUNS=20 WARMUP=5 bun tools/clip/pointwise_batch_matrix.ts
```

Results:

- B=3 median wins: step `8` (`0.800x`), step `10` (`0.824x`), step `57`
  (`0.947x`), step `111` (`0.774x`), step `115` (`0.883x`).
- B=3 flat/loss: step `59` flat (`1.010x`), step `113` loss (`1.108x`),
  step `117` loss (`1.100x`).
- B=2 cleanest wins: step `8` (`0.838x`) and step `57` (`0.884x`).
- Every row preserved exact parity against the z-batch kernel.

Decision:

- Do not integrate a global shared-W pointwise replacement.
- Keep a later full-plan allowlist ablation for B=3 steps `8`, `10`, `115`,
  maybe `111`, and a separate `2 + 1` schedule test for B=2 steps `8` and `57`.
- Promotion still requires full batch-major CLIP train wall time to improve by
  at least 5%.

Implementation boundary for the next attempt:

- Do not string-patch the generic `batchTrainDispatches()` output. It only sees
  generated `DispatchSpec`s.
- Add an emitter-level selective path that still has `ConvStep` metadata and
  emits production shared-W dispatches with the real source, destination, and
  residual slot refs.
- Start forward-only for the first 64-channel block, measure full B=3 train
  wall time, then add `pw_bwd` only if the forward path does not regress.

Attempt 2 result: implemented but not promoted.

- Added a gated production forward path:
  `pointwiseSharedWBatchForwardDispatch()` plus `sharedWForwardSteps`.
- Default optimizer behavior is unchanged unless a caller passes the allowlist.
- Exact gradient parity passed in `tools/clip/batch_major_train_bench.ts` for
  both baseline and `SHARED_W_FWD_STEPS=8,10,111,115`.

Commands:

```bash
TRIALS=2 RUNS=3 WARMUP=3 CONFIGS='base=;early=8,10;candidates=8,10,111,115' bun tools/clip/batch_major_train_matrix.ts
BATCH=2 TRIALS=2 RUNS=3 WARMUP=3 CONFIGS='base=;b2wins=8,57' bun tools/clip/batch_major_train_matrix.ts
```

Results:

- B=3 baseline median: `75.13 ms`.
- B=3 `8,10` median: `76.36 ms`.
- B=3 `8,10,111,115` median: `76.34 ms`.
- B=2 baseline median: `41.52 ms`.
- B=2 `8,57` median: `41.66 ms`.

Decision:

- Do not enable selective shared-W forward in the app or default optimizer.
- The compact-kernel wins do not survive full-chain train timing.
- Keep the gated path and matrix tool for future variants, but move active work
  to a target that shows up more clearly in full-chain profiling.

### 6. `spatial_bwd` Staging

Hypothesis: selected spatial backward kernels may be bandwidth-bound and benefit
from staged weights or vectorized horizontal pixels.

Promotion gate:

- Dispatch profiler shows `spatial_bwd` is a top contributor.
- `bun tools/clip/bwd_test.ts` passes.
- Full B=3 CLIP train time improves.

Attempt 1 result: rejected.

- Tried staging each `spatial_bwd` workgroup's per-input-channel weight footprint
  into workgroup memory.
- Correctness passed, including per-kernel spatial backward checks and the
  end-to-end directional derivative gate.
- Integrated optimizer timing did not produce a reliable speed win. The first
  staged run was much slower, and subsequent clean-code reruns also showed GPU
  timing instability, so the safe conclusion is "no proven win".

Commands run:

```bash
bun tools/clip/bwd_test.ts
BATCH=3 RUNS=3 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
CLIP_BATCH=3 VIEWS=3 RUNS=6 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Observed integrated runs:

| Variant | 3/9 batch x3 normal step | Profile total | CLIP batch |
| --- | ---: | ---: | ---: |
| per-lane raster baseline | 58.40 ms | 62.39 ms | 46.22 ms |
| staged `spatial_bwd` | 140.80 ms | 140.64 ms | 110.61 ms |
| reverted clean code rerun | 219.63 ms | 163.71 ms | 129.81 ms |

Decision:

- Reverted the staged-weight shader change.
- Do not retry this exact staging shape without GPU timestamp queries or a more
  controlled bench harness. Future `spatial_bwd` work should try a different
  decomposition, such as vectorized horizontal pixels or a special stem kernel,
  and must be gated by the integrated 3D step bench.

Attempt 2 result: profiling gate added.

- Added `tools/clip/spatial_bwd_profile_matrix.ts` to aggregate exact
  `spatial_bwd` dispatch labels across sequential `dispatch_profile.ts` runs.
- This does not change runtime behavior; it ranks candidate kernels for the next
  shader ablation.

Command:

```bash
BATCHES=1,3 TRIALS=2 RUNS=3 WARMUP=1 TOP=12 bun tools/clip/spatial_bwd_profile_matrix.ts
```

Results:

- B=1 total `spatial_bwd` median sum: `22.878 ms`.
- B=3 total `spatial_bwd` median sum: `40.495 ms`.
- Top B=3 single label: `spatial_bwd k3s2 3<-64 g1 @256x256`, median
  `7.667 ms`.
- Next B=3 labels were much smaller: `k7s2 64<-128 @64x64` at `2.381 ms` and
  repeated `k7s1 64<-64 @64x64 +=` at `2.001 ms` median per dispatch.

Decision:

- The next `spatial_bwd` code attempt should specialize the stem or vectorize
  horizontal pixels. It should not retry generic workgroup weight staging.
- Any variant must pass `bun tools/clip/bwd_test.ts` and improve full
  `BATCH=3` batch-major train or integrated 3D step timing before promotion.

Attempt 3 result: promoted for 3D batch optimizer.

- Added `spatial_bwd_stem4`, a stem-only `k3s2 3<-64 @256x256` backward
  specialization that writes four horizontal input pixels per thread.
- Wired the variant through `stemSpatialBwd` batch dispatch options.
- The 3D optimizer enables it by default for `BatchMajorVisionTrainer`; the
  integrated bench can disable it with `STEM_SPATIAL_BWD=0`.

Correctness:

```bash
STEM_SPATIAL_BWD=1 BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
bun tools/clip/bwd_test.ts
```

- Full-model gradient parity versus the generic single-image trainer passed for
  all B=3 lanes: `cos=1.000000`, `relLinf=0.00e+0`.
- Default backward tests still passed.

CLIP-only and dispatch profile:

- Full-chain B=3 CLIP matrix: baseline median `87.69 ms`, stem median
  `64.59 ms`.
- B=3 `spatial_bwd` profile: stem label dropped from `6.867 ms` to `1.195 ms`;
  total spatial backward sum dropped from `36.087 ms` to `30.799 ms`.

Integrated 3D step matrix:

```bash
STEM_SPATIAL_BWD=0 TRIALS=2 CONFIGS=3:3 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
TRIALS=2 CONFIGS=3:3 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| stem off | `95.58 ms` | `116.12 ms` | `86.56 ms` | `26.60 ms` |
| stem on | `72.72 ms` | `90.13 ms` | `65.30 ms` | `22.20 ms` |

Decision:

- Keep `stemSpatialBwd` enabled by default for the 3D batch optimizer.
- Keep `STEM_SPATIAL_BWD=0` as the negative-control path for future benches.

### 7. Pointwise + GELU Forward Fusion

Hypothesis: in train mode, pointwise convs followed by standalone GELU can write
both the saved pre-activation and the GELU output in one dispatch. This removes
24 forward GELU dispatches in the B=3 batch-major train path without changing
backward math.

Status: promoted for the 3D batch optimizer. The fused path is enabled by
default when `clipBatchSize > 1`; `FUSE_PW_GELU=0` is the integrated benchmark
negative control.

Implementation notes:

- `pointwiseFusedGelu` emits a pointwise kernel with two output bindings:
  the original pre-activation slot and the following GELU output slot.
- Batch dispatch generation skips the standalone GELU step only for exact
  pointwise-conv + GELU pairs where `gelu.src === conv.dst`.
- Spatial and SE GELU pairs are unchanged.

Correctness:

```bash
FUSE_PW_GELU=1 STEM_SPATIAL_BWD=1 BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
bun tools/clip/bwd_test.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

- Batch-major gradient parity passed for all B=3 lanes:
  `cos=1.000000`, `relLinf=0.00e+0`.
- Default backward tests and directional derivative gate passed.
- Parcel build passed with the standard Browserslist warning.

CLIP-only timing:

```bash
TRIALS=2 RUNS=3 WARMUP=3 CONFIGS='stem=stem;gelu=stem,gelu' bun tools/clip/batch_major_train_matrix.ts
```

| Variant | B=3 CLIP Train Median |
| --- | ---: |
| stem only | `73.33 ms` |
| stem + pointwise GELU fusion | `68.06 ms` |

Dispatch profile:

```bash
FUSE_PW_GELU=1 STEM_SPATIAL_BWD=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 TOP=16 bun tools/clip/dispatch_profile.ts
```

- Dispatch count dropped from `281` to `257`.
- The remaining standalone `gelu` isolated median sum was `2.194 ms`; the
  fused `pw+gelu` group accounted for `17.046 ms`.

Integrated 3D step matrix:

```bash
FUSE_PW_GELU=0 TRIALS=2 CONFIGS=3:3 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
TRIALS=2 CONFIGS=3:3 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| stem only | `88.99 ms` | `111.50 ms` | `80.68 ms` | `27.82 ms` |
| stem + pointwise GELU fusion | `87.28 ms` | `104.88 ms` | `76.38 ms` | `25.41 ms` |

Decision:

- Enable by default for 3D batch CLIP because it is exact, reduces dispatch
  count, improves B=3 CLIP train median by about `7%`, and improves integrated
  split-profile CLIP median by about `5%`.
- Keep it scoped to pointwise forward GELU. Do not fuse spatial/SE GELUs or
  GELU backward until a separate full-chain-visible gate justifies it.

Attempt 2 result: GELU backward into pointwise backward was implemented but not
promoted.

- `fuseGeluBwdIntoPw` detects adjacent `gelu_bwd` + `pw_bwd` pairs where the
  GELU intermediate has one writer and the following pointwise backward consumes
  it immediately.
- The fused dispatch stages `dY * geluGrad(pre)` directly inside the pointwise
  tile load and skips the standalone `gelu_bwd` dispatch.
- `FUSE_GELU_BWD_PW=1` enables the ablation in CLIP and 3D step benches.

Correctness and CLIP-only timing:

```bash
FUSE_PW_GELU=1 FUSE_GELU_BWD_PW=1 STEM_SPATIAL_BWD=1 BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
TRIALS=2 RUNS=3 WARMUP=3 CONFIGS='default=stem,gelu;gelubwd=stem,gelu,gelubwd' bun tools/clip/batch_major_train_matrix.ts
```

| Variant | B=3 CLIP Train Median |
| --- | ---: |
| default forward GELU fusion | `68.22 ms` |
| + GELU backward fusion | `61.70 ms` |

Dispatch profile:

```bash
FUSE_PW_GELU=1 FUSE_GELU_BWD_PW=1 STEM_SPATIAL_BWD=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 TOP=16 bun tools/clip/dispatch_profile.ts
```

- Dispatch count dropped from `257` to `233`.
- Standalone `gelu_bwd` isolated median sum dropped to `2.191 ms`.

Integrated 3D alternating matrix:

```bash
TRIALS=3 CONFIGS=base=3:3,gelubwd=3:3:gelubwd RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| default | `78.30 ms` | `93.50 ms` | `67.51 ms` | `23.38 ms` |
| `FUSE_GELU_BWD_PW=1` | `76.83 ms` | `94.71 ms` | `67.02 ms` | `24.81 ms` |

Decision:

- Keep the fused backward code and benchmark flag for future experiments.
- Do not enable by default: the CLIP-only matrix won, but the integrated 3D
  split profile was flat to slightly worse.

### 8. F16 Weights With F32 Math

Hypothesis: f16 weights reduce memory traffic and payload size without changing
the core accumulation path.

Promotion gate:

- Feature-gated fallback to f32.
- Embedding cosine versus f32 fused is at least 0.9995.
- Gradient cosine versus f32 fused is at least 0.995.
- B=3 train path improves at least 10%, or payload reduction is explicitly the
  reason for promotion.

### 9. Prompt Embedding Cache

Hypothesis: same-text mode and repeated camera prompts should avoid repeated
text encoder work.

Status: first in-memory cache landed for the 3D page.

Promotion gate:

- Same-text mode encodes one prompt.
- Warm-cache optimize start is below 50 ms for same prompt.
- Camera-mode warm-cache returns all view embeddings below 100 ms.

Attempt 1 result:

- Added an in-memory promise cache in `src/splat3d_page.ts`.
- `same text` mode now encodes one expanded prompt and reuses the embedding for
  all camera views.
- Camera-text mode also reuses cached embeddings when the exact expanded prompt
  repeats across runs.

Remaining:

- Add an explicit browser smoke/readout check for same-text prompt count.
- Consider IndexedDB persistence after the hot-loop work is further along.

### 10. Raster Occupancy Telemetry

Hypothesis: before another raster scheduler or staging rewrite, measure whether
the current bottleneck is tile overflow, high occupancy, or long per-tile
compositing chains.

Status: telemetry tool landed.

Implementation:

- Added `tools/splat3d/raster_telemetry.ts`.
- It renders selected 3D views, reads `tileCounts` and `tileStop`, and reports
  active-tile rate, overflow tiles/pairs, count percentiles, stop percentiles,
  and mean `stop/count`.
- `tileStop` now has `COPY_SRC` usage so tools can inspect it.

Commands:

```bash
bun tools/splat3d/raster_telemetry.ts
G=12000 CAP=2048 bun tools/splat3d/raster_telemetry.ts
```

Default `G=4096`, `CAP=2048` result:

| Metric | Result |
| --- | ---: |
| Aggregate overflow pairs | `0 / 576871` |
| Max tile count | `911` |
| Max tile stop | `365` |
| Worst dropped pair pct | `0.0%` |

High-splat stress `G=12000`, `CAP=2048` result:

| Metric | Result |
| --- | ---: |
| Aggregate overflow pairs | `21440 / 1717219` (`1.2%`) |
| Max tile count | `2678` |
| Max tile stop | `702` |
| Worst dropped pair pct | `3.6%` |

Interpretation:

- Overflow is not a default-workload bottleneck at 4096 splats.
- The current default cap is conservative for the initial 4096-splat scene:
  observed max tile count is below `1024`.
- Overflow starts to matter at much higher splat counts, especially oblique
  high/low views.
- The next measured raster ablation should test a smaller cap such as `1024`
  on the default workload, while keeping high-splat stress telemetry as the
  safety check.

Attempt 2 result: `cap=1024` was tested but not promoted.

Implementation:

- Added `CAP` support to `tools/splat3d/step_bench.ts`.
- Added `capNNN` config tokens to `tools/splat3d/step_matrix.ts`.

Safety check:

```bash
CAP=1024 bun tools/splat3d/raster_telemetry.ts
```

Result: default `G=4096` still had zero overflow:

| Metric | Result |
| --- | ---: |
| Aggregate overflow pairs | `0 / 576871` |
| Max tile count | `911` |
| Max tile stop | `365` |

Performance:

```bash
TRIALS=3 CONFIGS=base=3:3,cap1024=3:3:cap1024 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
TRIALS=2 CONFIGS=base=9:3,cap1024=9:3:cap1024 RUNS=3 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

| Views | Cap | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | `2048` | `53.00 ms` | `58.08 ms` | `41.70 ms` | `13.56 ms` |
| 3 | `1024` | `53.44 ms` | `56.78 ms` | `40.53 ms` | `13.83 ms` |
| 9 | `2048` | `156.02 ms` | `166.22 ms` | `123.63 ms` | `38.69 ms` |
| 9 | `1024` | `156.21 ms` | `165.15 ms` | `123.58 ms` | `38.41 ms` |

Decision:

- Do not change the app/default optimizer cap. The smaller cap is safe for the
  measured initial default scene, but integrated timing is flat/mixed.
- Keep `CAP` and `capNNN` benchmark support for future memory/safety checks.

## Already Tried

### Exact Circle-Vs-Tile Support Pruning

Result: rejected for current default workload.

Reason: mathematically valid, but flat to slightly slower at 256px / 4096
splats because the extra per-tile math outweighed the bin-count reduction.

### Replicated Activation CLIP Batching

Result: rejected for production integration.

Reason: it reduces CPU encode/submit overhead but does not improve GPU wall time.
True batch-major dispatch is the useful path.
