# 3D Splat Performance Notes

Last updated: 2026-07-08

## Current Timing Surface

The 3D page now samples a split-submit wall-time profile every 30 optimizer
steps. The overlay reports:

- raster forward and backward
- CLIP forward and backward
- Adam step
- clear/display overhead
- sampled optimizer-step total

This is intentionally labeled as sampled wall time. It is good enough to decide
whether CLIP or raster is the current bottleneck, but it is not exact GPU
timestamp attribution for the normal single-submit `step()` path.

## Measurements

Default 4096-splat scene, 256px renders, 9 cameras, Bun/WebGPU on 2026-07-08:

| Views / Step | Normal Step Avg | Split Profile Total | Raster Fwd | Raster Bwd | CLIP Fwd | CLIP Bwd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 / 9 | 205.26 ms | 228.20 ms | 8.16 ms | 33.38 ms | 77.77 ms | 102.81 ms |
| 5 / 9 | 122.02 ms | 134.83 ms | 5.38 ms | 19.85 ms | 39.77 ms | 57.79 ms |
| 3 / 9 | 69.94 ms | 75.57 ms | 2.32 ms | 11.20 ms | 24.07 ms | 34.96 ms |

Takeaway: random N-of-K view sampling is a real wall-clock lever. The 3/9 path
is roughly 2.9x faster than 9/9 while preserving stochastic coverage of all
camera prompts over multiple steps. The implementation uses shuffled epochs, so
3/9 still randomizes order but covers all 9 views every 3 optimizer steps. CLIP
remains around 78-79% of sampled wall time, so deeper speedups should attack
CLIP calls/batching/precision before large raster rewrites.

Tried and reverted: exact circle-vs-tile support pruning in emit/binning. It was
mathematically safe, but on this workload it made raster forward/backward flat
to slightly slower, likely because the extra per-tile math outweighed the small
bin-count reduction at 256px / 4096 splats.

## Four-Agent Pass: Main Read

### CLIP

The most likely CLIP bottleneck is still pointwise matmul. Static work is
dominated by forward pointwise and `pw_bwd`, so another clean 4x from kernel
tuning alone is unlikely unless we change precision, reduce CLIP calls, or make
a larger matmul strategy shift.

Next measurements:

- Per-dispatch timing grouped by label/shape for B=1, B=3, and B=9.
- Pointwise forward/backward timing for the repeated 256->768 and 768->256
  blocks.
- Stem `spatial_bwd`, especially the first 256x256 3<-64 kernel.
- Aggregate GELU forward/backward share.
- Attention backward only if timestamps show it above roughly 5-10%.

Most promising CLIP experiments:

- Use batch-major CLIP for multi-view training.
- Try f16 weights/activations with f32 reductions behind feature gating.
- Stage weights in `spatial_bwd`.
- Fuse pointwise/GELU/residual blocks if dispatch and scratch traffic show up.
- Add per-dispatch timestamp-query profiling to the batch benches.

### Raster

The current rasterizer already has two key FasterGS-style ideas: per-tile sorted
IDs are saved in forward, and backward reuses the saved order plus tile stop
counts. The best next raster experiments are smaller and directly applicable:

- Alias raster image/grad buffers with CLIP input/grad buffers to remove two
  full-image copies per view.
- Stage derived splat parameters in workgroup chunks for forward/backward.
- Add tile overflow telemetry before tuning caps or thresholds.
- Benchmark workgroup-local gradient reductions before replacing global fixed
  point atomics.

Do not port the dynamic viewer bucket-sort path into training. It is useful for
preview, but the training backward path needs the saved per-tile order and stop
state.

## Next Decision Rule

Use the overlay first:

- If CLIP dominates, land batch-major training in the 3D optimizer before deep
  shader rewrites.
- If raster dominates, start with buffer aliasing, overflow telemetry, and
  workgroup staging. Re-test circle-tile pruning only at higher resolution or
  larger splat counts.
- If Adam/display/clear are visible but small, leave them alone until CLIP and
  raster are under control.

## Batch-Major CLIP Integration Attempt

The first app-level batch integration is now behind an opt-in 3D page control:

- `single CLIP`
- `batch CLIP x3`
- `batch CLIP x9`

The implementation is conservative. It renders selected views into CLIP batch
lanes, runs one batch-major CLIP train pass, then re-renders each view before
applying that lane's image gradient. This preserves current raster correctness
because the rasterizer still owns only one view's tile bins and sorted order at a
time.

Headless integrated benchmark:

```bash
CLIP_BATCH=1 VIEWS=3 RUNS=8 WARMUP=4 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=3 RUNS=8 WARMUP=4 bun tools/splat3d/step_bench.ts
CLIP_BATCH=1 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
CLIP_BATCH=9 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
```

Representative measurements on Apple `metal-3`:

| Views / Step | CLIP Batch | Normal Step Avg | Split Profile Total | Notes |
| --- | ---: | ---: | ---: | --- |
| 3 / 9 | 1 | 104.32 ms | 129.47 ms | sequential run |
| 3 / 9 | 3 | 93.41 ms | 121.43 ms | modest, noisy win |
| 9 / 9 | 1 | 381.18 ms | 412.02 ms | same-session matrix |
| 9 / 9 | 3 | 292.10 ms | 314.28 ms | clear all-view win |
| 9 / 9 | 9 | 299.93 ms | 330.83 ms | not better than x3 here |

Takeaway: keep batch CLIP as an ablation/screenshot toggle, but do not make it
the default yet. The replayed raster forward eats enough of the CLIP batching win
that the next promotion step should be per-lane raster state or direct
raster/CLIP buffer binding.

## Dispatch Profile Snapshot

`tools/clip/dispatch_profile.ts` now provides warmed isolated dispatch timings:

```bash
MODE=train BATCH=1 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
CSV=1 MODE=train BATCH=3 bun tools/clip/dispatch_profile.ts > /tmp/clip_b3.csv
```

This is a kernel-ranking tool, not exact full-chain attribution. First warmed
results:

| Batch | Dominant Groups |
| ---: | --- |
| 1 | `pw` 19.9%, `pw_bwd` 18.7%, `spatial_bwd` 17.1%, `conv` 14.3% |
| 3 | `pw` 20.8%, `spatial_bwd` 19.6%, `pw_bwd` 19.5%, `conv` 14.7% |

The profiler changes the priority slightly: `spatial_bwd` is at least as
important as pointwise in B=3, while attention backward is too small for the
first wave.

## Prompt Encoding Cache

The 3D page now caches text embeddings by exact expanded prompt. In `same text`
mode it encodes one prompt and reuses the embedding for all 9 views. In camera
text mode, repeated runs of the same prompt reuse the cached camera-specific
embeddings.

This does not change optimizer step speed, but it removes duplicate text tower
work from prompt setup and makes same-text ablations less annoying to run.

## Raster/CLIP Buffer Aliasing

The rasterizer can now bind external image and image-gradient buffers through
`Raster3DIOState`. The 3D optimizer uses this to render directly into CLIP input
slots and read gradients directly from CLIP input-gradient slots.

This removes two full-image copies per optimized view. It is not a large
standalone speedup in current measurements because CLIP dominates, but it is the
right setup for per-lane raster state and batched raster/CLIP scheduling.

## Per-Lane Raster State

Batch CLIP lanes now get private raster scratch state: `derived`, `tileCounts`,
`binnedIds`, and `tileStop`. The optimizer can render a chunk of selected views
into CLIP batch input lanes, run one batch CLIP train pass, then apply each
lane's raster backward using the saved tile state. The replay forward pass is
gone for complete batch chunks.

Sequential benchmark on Apple `metal-3` after this change:

| Views / Step | CLIP Batch | Normal Step Avg | Split Profile Total | Raster Replay |
| --- | ---: | ---: | ---: | ---: |
| 3 / 9 | 1 | 80.87 ms | 101.99 ms | 0.00 ms |
| 3 / 9 | 3 | 58.40 ms | 62.39 ms | 0.00 ms |
| 9 / 9 | 1 | 195.65 ms | 215.00 ms | 0.00 ms |
| 9 / 9 | 3 | 167.23 ms | 176.97 ms | 0.00 ms |
| 9 / 9 | 9 | 235.48 ms | 295.60 ms | 0.00 ms |

Takeaway: `batch CLIP x3` is now the useful multi-view speed mode. `batch CLIP
x9` is still not worth promoting on this hardware.

The 3D page now defaults to `3 / 9 views` plus `batch CLIP x3`. Single CLIP and
batch x9 remain available as explicit ablation toggles.

## Step Matrix Runner

`tools/splat3d/step_matrix.ts` runs `step_bench.ts` sequentially across a config
matrix and reports medians plus min/max ranges. Use it for future performance
promotion decisions when the GPU is noisy:

```bash
TRIALS=2 CONFIGS=3:1,3:3 RUNS=4 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

First control run under a slow/noisy GPU state still showed the same-session
direction: `3/9 batch x3` median normal step `143.66 ms` versus `194.68 ms` for
`3/9 single CLIP`.

## Multi-View Raster Batching Read

STAR UVT/world tubes do not directly turn our 9 fixed CLIP cameras into one
cheap raster. Their speed comes from a primitive whose support is native over
ordered time in one camera gauge, or over a projective rational moving-camera
gauge. Our 9 views have different projection maps, tile support, visibility,
and depth order, so one shared cross-view tile sort would be wrong or too loose.

The near-term raster path is still valuable:

- move camera constants out of baked WGSL into a camera storage/uniform buffer;
- make `derived`, `tileCounts`, `binnedIds`, and `tileStop` lane-strided;
- dispatch prep/emit/forward/backward with `workgroup_id.z = view lane`;
- keep one tile list and one depth order per camera lane.

This should cut CPU/pass/pipeline overhead and bind churn, but it will not remove
the per-view projection, binning, sort, compositing, or backward transmittance
walk. A true STAR-style sublinear multi-camera primitive would be a separate
representation project: projective rational camera bundles or atlas-residual
splats with a new backward chain.

Follow-up note: `agent_notes/optimization_session/static_multiview_worldtube_followup.md`
re-checked the STAR UVT/PRT world-tube math against the current browser raster.
The exact next ablation does not require a new splat object: move camera data
into a buffer and dispatch raster kernels over `workgroup_id.z = view lane`.
The only backward wrinkle is that `accGrad` is currently shared and cleared per
view, so a true batched backward needs lane-strided `accGrad` first, or a larger
fixed-point raw-gradient atomic rewrite later.

## Stem Spatial Backward Specialization

The B=3 dispatch profile made the MobileCLIP stem backward the largest single
CLIP kernel: `spatial_bwd k3s2 3<-64 g1 @256x256`. A stem-only
`spatial_bwd_stem4` variant now computes four horizontal input pixels per
thread and is enabled by default for the 3D batch optimizer.

Same-session integrated matrix:

```bash
STEM_SPATIAL_BWD=0 TRIALS=2 CONFIGS=3:3 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
TRIALS=2 CONFIGS=3:3 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Step Median | Profile Median | CLIP Median |
| --- | ---: | ---: | ---: |
| stem off | `95.58 ms` | `116.12 ms` | `86.56 ms` |
| stem on | `72.72 ms` | `90.13 ms` | `65.30 ms` |

Use `STEM_SPATIAL_BWD=0` as the negative-control path when future CLIP changes
need to compare against the pre-specialization baseline.

## Pointwise GELU Forward Fusion

The B=3 post-stem dispatch profile still showed standalone train-mode GELU
forward/backward as a visible traffic cost. The promoted forward-only fusion
handles the 24 exact pointwise-conv + GELU pairs by writing both the saved
pre-activation slot and the GELU output slot from one pointwise dispatch.
Spatial and SE GELU pairs remain split.

Verification:

```bash
FUSE_PW_GELU=1 STEM_SPATIAL_BWD=1 BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
bun tools/clip/bwd_test.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Same-session B=3 CLIP matrix:

| Variant | B=3 CLIP Train Median |
| --- | ---: |
| stem only | `73.33 ms` |
| stem + pointwise GELU fusion | `68.06 ms` |

Integrated 3D step matrix:

| Variant | Normal Step Median | Profile Median | CLIP Median |
| --- | ---: | ---: | ---: |
| `FUSE_PW_GELU=0` | `88.99 ms` | `111.50 ms` | `80.68 ms` |
| default fused | `87.28 ms` | `104.88 ms` | `76.38 ms` |

The 3D batch optimizer enables this by default. Use `FUSE_PW_GELU=0` in
`tools/splat3d/step_bench.ts` / `step_matrix.ts` as the negative control.
