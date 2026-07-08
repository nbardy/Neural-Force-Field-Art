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

Status: first isolated profiler landed.

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

### 3. Raster/CLIP Buffer Aliasing

Hypothesis: removing image and gradient copies between raster and CLIP saves
bandwidth and simplifies the batch path.

Implementation notes:

- Let raster forward write directly into a CLIP input lane buffer, or let CLIP
  bind the raster image buffer as input.
- Let raster backward read directly from a CLIP input-gradient lane.
- Assert storage-offset alignment before using dynamic offsets.

Promotion gate:

- Old and aliased paths match on one fixed view.
- Integrated wall time improves or the change is required to make batch CLIP
  profitable.

### 4. Per-Lane Raster State For Batched Views

Hypothesis: storing `derived`, bins, sorted IDs, and tile stops per active view
lets us render several views, run batch CLIP once, and run matching backwards
without re-rendering.

Implementation notes:

- Add state buffers shaped by `[batchLane, ...]`.
- Dispatch prep/bin/forward/backward with `workgroup_id.z = lane`.
- Keep depth sorting and compositing per view; do not attempt one cross-view
  sort.

Promotion gate:

- Matches conservative replay path.
- B=3 optimizer step beats single path by a clear margin.

### 5. Shape-Gated Shared-W Pointwise

Hypothesis: some pointwise CLIP layers can reuse a staged weight tile across
batch lanes and improve full CLIP train time.

Implementation notes:

- Start with all allowlists off.
- Enable only shapes that are hot in the dispatch profile and win in a full
  CLIP chain.
- Do not apply globally; existing microbench results are mixed.

Promotion gate:

- Full B=3 batch-major train wall time improves by at least 5%.
- Gradient parity still passes.

### 6. `spatial_bwd` Staging

Hypothesis: selected spatial backward kernels may be bandwidth-bound and benefit
from staged weights or vectorized horizontal pixels.

Promotion gate:

- Dispatch profiler shows `spatial_bwd` is a top contributor.
- `bun tools/clip/bwd_test.ts` passes.
- Full B=3 CLIP train time improves.

### 7. F16 Weights With F32 Math

Hypothesis: f16 weights reduce memory traffic and payload size without changing
the core accumulation path.

Promotion gate:

- Feature-gated fallback to f32.
- Embedding cosine versus f32 fused is at least 0.9995.
- Gradient cosine versus f32 fused is at least 0.995.
- B=3 train path improves at least 10%, or payload reduction is explicitly the
  reason for promotion.

### 8. Prompt Embedding Cache

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

## Already Tried

### Exact Circle-Vs-Tile Support Pruning

Result: rejected for current default workload.

Reason: mathematically valid, but flat to slightly slower at 256px / 4096
splats because the extra per-tile math outweighed the bin-count reduction.

### Replicated Activation CLIP Batching

Result: rejected for production integration.

Reason: it reduces CPU encode/submit overhead but does not improve GPU wall time.
True batch-major dispatch is the useful path.
