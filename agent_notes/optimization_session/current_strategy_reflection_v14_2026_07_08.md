# Current Strategy Reflection: v14 Update

Date: 2026-07-08

## What We Have Done

- Built the 3D splat optimizer path with 9 camera views, black background, and
  camera-specific prompt text support.
- Added the 3x3 grid/contact-sheet CLIP schedule plus 2 close-up lanes.
- Added direct grid raster so grid cells render at `80x80` instead of full
  `256x256` scratch images.
- Added per-lane raster scratch state so batch CLIP lanes can keep their own
  forward tile/order state.
- Added an overlay/profile surface for raster forward/backward, CLIP
  forward/backward, Adam, display, and total step cost.
- Added versioned experiment forks `v01` through `v14`, each with notes and
  snapshots for rollback/diff review.
- Added Chrome/Dawn trace capture with `tools/webgpu_trace.mjs`.
- Added cached CLIP gradient cadence and then a fixed-wall-clock quality gate
  that rejected the naive version.

## What Made Big Leaps

- N-of-K view scheduling is the biggest clean speed lever so far. It reduces
  full CLIP work while preserving coverage over time.
- Grid/contact-sheet CLIP is a real schedule lever because it can supervise 9
  views with fewer CLIP lanes, but it changes the signal and must be judged by
  screenshots/teacher score.
- Direct grid raster is a good systems win because it removes wasted full-size
  scratch rendering for small grid cells.
- Cached CLIP gradient cadence produced a real wall-clock step-rate gain:
  `cache4` reached roughly `44.1 steps/sec` versus base `17.6 steps/sec` in the
  v14 gate.

## Where We Are Churning

- Small CLIP shader fusions and pointwise variants are producing incremental
  gains, not the desired 2-4x by themselves.
- Naive cached `dL/dimage` was too stale. In the 5s full-teacher gate, base mean
  cosine was `0.09336`, `cache2` was `0.04363`, and `cache4` was `0.02805`.
- We still do not have Metal shader-counter evidence, so statements about
  memory bandwidth, occupancy, cache misses, and register pressure remain
  hypotheses.

## What Is Going Well

- The experiment trail is now strong. New risky ideas are gated and recorded
  under `experiments/clip_forks/vNN_*` with snapshots.
- Measurements now separate wall-clock speed from quality. v13 showed cadence
  speed; v14 showed naive cadence quality failure.
- The browser trace tool works and captures real Chrome/Dawn command behavior.
- We have enough profiling to know CLIP dominates more than raster on the
  current settings, and pointwise/spatial/conv groups matter more than attention
  backward as first targets.

## What Could Go Better

- Before another deep CLIP shader rewrite, get real GPU-counter profiling from
  full Xcode/Metal tooling if possible.
- For every speed schedule, pair the speed benchmark with a fixed-wall-clock
  quality gate immediately.
- Keep screenshots/contact sheets in the fork result folder, not only `/tmp`,
  so the blog trail survives.
- Use real prompt embeddings for semantic quality gates once the synthetic
  teacher-score harness is stable.

## Is There A Real 2-4x CLIP Speed Path?

There is a real 2x-class wall-clock path, but it is more likely to come from
schedule/proxy design than from one WGSL tweak.

Likely stack:

1. N-of-K random/epoch views.
2. Grid/contact-sheet proxy plus close-up lanes.
3. Smarter cached/proxy steps with lower cached-step learning rate or periodic
   full 9-view teacher refresh.
4. Targeted CLIP kernel work on pointwise/spatial/conv after counter profiling.
5. Selective precision only where input-gradient quality passes.

A pure same-resolution, every-step MobileCLIP train pass getting 4x faster from
only shader cleanup is still unproven. It may have 1.3-2x headroom through
better tiling/fusion/precision, but a defensible 4x claim needs hardware-counter
evidence or a larger algorithmic change.

## Shader Profiling Status

Used:

- WebGPU timestamp-query profiling in CLIP dispatch tools.
- Integrated 3D step profiling.
- Chrome/Dawn trace capture through `tools/webgpu_trace.mjs`.

Not yet used:

- Metal System Trace with full Xcode selected.
- Shader-level memory bandwidth counters.
- Occupancy/register/cache/stall counters.
- Shader source line attribution.

So the honest answer is: we have trace tooling and timing evidence, but not yet
true shader hot-spot counters.
