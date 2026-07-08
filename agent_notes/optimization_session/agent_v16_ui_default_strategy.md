# Agent v16 UI/default strategy for 3D optimization gates

Date: 2026-07-08

Scope: inspection note only. No source changes are made here.

## Current 3D page surface

The active 3D page is `src/splat3d.html` plus `src/splat3d_page.ts`.

Current controls, in page order:

- `#prompt`: text input, default `a photo of a cat`.
- `#view`: populated from the nine prepared camera names.
- `#promptMode`: `camera text` default, `same text`.
- `#bgTextMode`: `black bg` default, `no bg text`.
- `#viewBatch`: `1/9 view`, `2/9 views`, `3/9 views` default, `5/9 views`, `9/9 views`.
- `#viewSampler`: `epoch views` default, `random views`.
- `#clipMode`: `single CLIP`, `batch CLIP x2`, `batch CLIP x3` default, `batch CLIP x9`.
- `#clipLayout`: `per-view CLIP` default, `3x3 grid + 2`.
- `#gridPromptMode`: `grid prompt` default, `same grid text`; disabled unless grid layout is selected.
- `#gridRasterMode`: `grid raster 256` default, `grid raster 80`; disabled unless grid layout is selected.
- `#optimize` and `#reset`.

Current runtime defaults in `src/splat3d_page.ts`:

- `promptMode = camera`
- `gridPromptMode = contact_sheet`
- `blackBgText = true`
- `viewsPerStep = 3`
- `viewSampler = epoch`
- `clipBatchSize = 3`
- `clipLayout = per_view`
- `gridDirectRaster = false`

The readout already exposes the screenshot-relevant state: step, active camera,
views per step, random sampler marker, `clip xN`, `grid+2`, grid prompt mode,
`80px grid raster`, prompt mode, black background text, cosine/init/delta, and
phase. The timing HUD samples every 30 steps and reports wall-profile buckets.

## Promote or keep as default now

1. Keep the main app default as `per-view CLIP`, `3/9 views`, `epoch views`,
   `batch CLIP x3`.

   This is the right first-screen behavior: it is the fastest stable interactive
   path, still samples all nine cameras across shuffled epochs, and avoids making
   the first screenshot depend on a contact-sheet prompt-quality question.

2. Keep `batch CLIP x3` as the default and keep the CLIP batch selector visible.

   The selector is useful for performance storytelling and debugging. `batch CLIP
   x3` won after per-lane raster state removed replay; `batch CLIP x9` should
   stay available as an ablation, not a default.

3. Make `grid raster 80` the preferred/default grid-mode raster choice, but do
   not make grid layout the global page default yet.

   `grid9_close2 + directgrid` moved normal median from about `87.93 ms` to
   `59.03 ms`, and `grid80 + depthwise4` brought all-view grid supervision close
   to the ordinary `3/9` per-view baseline. Quality still needs visual testing,
   so the safe promotion is: when the user selects `3x3 grid + 2`, default that
   mode to the `80x80` grid raster path while keeping the `256` option as a
   fallback.

4. Promote `SPATIAL_BWD_VARIANT=depthwise4` as a silent internal default for the
   3D batch optimizer, not as UI.

   It passed correctness and repeatedly showed a modest integrated win
   (`53.12 ms -> 49.96 ms` in the focused matrix; also helped the grid80 stack).
   It has no artist-facing meaning, so putting it in the controls would be UI
   noise. Keep a benchmark negative-control path to force generic spatial
   backward.

5. Keep the existing silent defaults: `stemSpatialBwd=true` and
   `fusePointwiseGeluForward=true`.

   These are exact kernel improvements with measured wins. They should remain
   hidden defaults. The user-facing page should talk in terms of views, CLIP
   batches, and grid/contact-sheet supervision, not shader implementation names.

## Keep env-only for now

- `CLIP_PRECISION=f16` / `PRECISION=f16`: do not surface. It halves payload, but
  all-weight f16 failed the input-gradient cosine gate and was slower in the
  recorded integrated timestamp run.
- `SHARED_W_FWD_STEPS`: do not surface or default. Full-chain timing did not
  preserve the microbench wins.
- `FUSE_GELU_BWD_PW=1` and `FUSE_RESIDUAL_BWD_PW=1`: keep env-only for the main
  app path. They are promising as a paired grid80+depthwise stack improvement,
  but earlier per-view integrated results were flat. Re-test visually and
  interactively before making them a hidden grid-mode default.
- `SINGLE_PASS_RASTER_FWD=1`: keep env-only. Exact path, but the separate-forward
  control won the rerun.
- `VIEW_LANE_RASTER_FWD=1` and `VIEW_LANE_RASTER_BWD=1`: keep env-only. They
  improve some raster buckets but did not improve the default 3-view optimizer
  step enough to justify page exposure.
- `CAP` / `capNNN`: keep benchmark-only. `CAP=1024` was safe for the measured
  default scene but performance was flat/mixed and has less safety margin.
- `G`: keep benchmark/dev-only. Splat count changes visual capacity and memory
  pressure; it is not a screenshot progress control yet.
- `TIMESTAMP=1`: keep tool-only. The page HUD's sampled wall profile is enough
  for screenshots; timestamp queries are for bench/profiling tools.
- `STEM_SPATIAL_BWD=0` and `FUSE_PW_GELU=0`: keep as negative controls only.

## Screenshot and blog presentation

Use the UI to show workflow-level progress, not kernel switches.

Recommended screenshot sequence:

1. Boot/default view: random shared 3D splats in the 3x3 camera grid, controls
   showing `3/9 views`, `epoch views`, `batch CLIP x3`, `per-view CLIP`.
2. Default run after enough steps for structure: timing HUD visible, readout
   showing `clip x3`, `camera text`, `black bg`, and cosine improvement.
3. All-view progress screenshot: select `3x3 grid + 2`, `grid prompt`, and
   `grid raster 80`. Caption this as "all nine cameras supervised each step via a
   contact sheet plus two close-up lanes"; do not mention `GRID_DIRECT_RASTER`
   in the app UI.
4. Prompt-quality comparison: `camera text` vs `same text`, and `black bg` vs
   `no bg text`. These are understandable visual ablations.
5. Optional performance appendix image/table: per-view `3/9 batch x3` versus
   `grid80+2 all-view`. Put shader gates in the caption or blog text, not in the
   page controls.

Avoid adding visible checkboxes for f16, GELU/residual fusion, shared-W, raster
lane scheduling, cap, or timestamps. They are engineering gates, not creative
controls. If the page needs reproducible screenshots later, add a query/preset
surface such as `?preset=default3` and `?preset=grid80`, but keep the on-screen UI
focused on the current selectors.

## Files to touch later if implementing

For changing the grid-mode default to `grid raster 80`:

- `src/splat3d.html`: change the selected option in `#gridRasterMode`, or leave
  markup as-is and let page logic switch it when grid layout is selected.
- `src/splat3d_page.ts`: adjust `status.gridDirectRaster`, `selectedGridDirectRaster()`,
  and/or `syncClipLayoutControls()` so `3x3 grid + 2` defaults to direct `80`
  raster while preserving the explicit `grid raster 256` fallback.
- `src/splat3d_page.ts`: no readout work is needed; it already appends
  `80px grid raster`.
- `src/splat3d/optimize.ts` and `src/splat3d/grid_clip.ts`: no logic change is
  needed unless the default is moved into optimizer config instead of page state.

For silently defaulting `depthwise4`:

- `src/splat3d/optimize.ts`: set `spatialBwdVariant` to `depthwise4` by default
  for the 3D batch path, while still allowing a config override to generic.
- `tools/splat3d/step_bench.ts`: keep `SPATIAL_BWD_VARIANT` as the explicit
  positive/negative bench control.
- `tools/splat3d/step_matrix.ts`: keep the `dw4` token and consider adding a
  `generic` or `nodw4` token if the app default changes.
- `docs/SPLAT3D_PERF_NOTES.md` and `docs/SPLAT3D_ABLATION_QUEUE.md`: update the
  baseline description after the default changes.

For possibly making backward local fusions hidden defaults later:

- `src/splat3d/optimize.ts`: decide whether defaults are global or conditional on
  `clipLayout=grid9_close2 && gridDirectRaster && spatialBwdVariant=depthwise4`.
- `src/clip/vision_bwd_wgsl.ts` and `src/clip/vision_batch_wgsl.ts`: existing
  fused emitters are the important implementation surface; only touch if the
  defaulting requires cleaner option plumbing.
- `tools/clip/bwd_test.ts`, `tools/clip/dispatch_profile.ts`,
  `tools/splat3d/step_bench.ts`, and `tools/splat3d/step_matrix.ts`: keep env
  controls for correctness and same-session comparisons.

For screenshot presets, if needed:

- `src/splat3d_page.ts`: parse `URLSearchParams` before `syncClipLayoutControls()`
  and before initial `Splat3DOptimizer.create()`.
- `src/splat3d_page.ts`: apply preset values to existing DOM controls, then reuse
  the current selected-value helpers.
- Optional only: `docs/BLOG_PROGRESS_NOTES.md` for the exact capture recipe.

For f16 UI, if it is ever revisited:

- `src/splat3d_page.ts`: request `shader-f16` only when explicitly selected,
  load `weights_train_f16.bin` as `Uint16Array`, and surface a clear fallback or
  warning.
- `src/splat3d/optimize.ts`, `src/clip/vision.ts`, `src/clip/vision_batch.ts`,
  `src/clip/vision_wgsl.ts`, `src/clip/vision_bwd_wgsl.ts`, and
  `src/clip/vision_batch_wgsl.ts`: keep precision plumbing exact.
- Do not add this until gradient quality and integrated timing both clear the
  recorded gates.
