# Agent 3 Notes - 3x3 Grid CLIP Strategy

Date: 2026-07-08

Scope: evaluated the proposed strategy of rendering nine 3D camera views into
one 256x256 CLIP input as a 3x3 grid, prompting CLIP with a "3x3 grid from nine
different angles" text, and optionally adding two full-resolution close-up
camera losses. I inspected the current 3D optimizer, raster path, camera prompts,
CLIP input layout, batch CLIP path, and benchmark tools. I did not edit runtime
code.

## Short Read

This is worth trying as a gated ablation, but it is not equivalent to the
current nine separate 256x256 view losses.

It preserves the CLIP encoder input shape, because MobileCLIP still receives one
`[3,256,256]` image. It does not preserve per-view resolution or per-view loss
signal. In a practical tiled mosaic each camera view is around `80x80`, not
`256x256`, and the whole 3x3 contact sheet gets one embedding/loss. That makes
it a CLIP-call reduction / all-view-coverage trick, not a true substitute for
nine independent CLIP losses.

The strongest prototype is probably not pure grid-only. The useful ablation is:

- lane 0: one low-res 3x3 grid containing all nine views;
- lanes 1 and 2: two sampled full-resolution 256x256 close-up views;
- one batch-major CLIP x3 train pass.

That keeps the current CLIP batch shape, adds full-scene all-view pressure every
step, and preserves two high-resolution camera gradients.

## Current Code Facts

Current 3D optimizer facts:

- `src/splat3d/optimize.ts` asserts CLIP input shape `[3,256,256]`.
- `Splat3DOptimizer.step()` samples N-of-K views with shuffled coverage.
- Default UI state is `3/9` views per step and `batch CLIP x3`.
- Per-view prompts are encoded into one text embedding per camera.
- Batch-major CLIP stores activation slots as `[batch][slotFloats]`.
- Raster can already render directly into CLIP input lanes and read directly
  from CLIP input-gradient lanes through `Raster3DIOState`.
- Per-lane raster forward state exists for batch chunks, so complete B=3 chunks
  no longer need a replayed forward pass.

Current raster facts:

- Raster dims must be multiples of `TILE=16`.
- Current full-resolution view is `256x256`, which is `16x16=256` tiles.
- A 3x3 grid cannot use exact `85.333px` cells in the existing tiled raster.
- The clean first grid is `cell=80`, `gutter=8`:
  `3 * 80 + 2 * 8 = 256`.
- Each low-res view then has `5x5=25` tiles. All nine cells have 225 view-tiles,
  slightly less pixel/tile work than one full 256 view, but prep/binning still
  happens for nine camera projections.

## 1. Gated Ablation Implementation

Do this as a forked experiment, not by mutating the default raster shaders in
place.

Suggested gate names:

- `clipLayoutMode: "per-view" | "grid9" | "grid9_close2"`
- tool env: `CLIP_LAYOUT=per_view|grid9|grid9_close2`
- optional env: `GRID_CELL=80`, `GRID_GUTTER=8`, `GRID_CLOSEUPS=2`

Recommended fork files:

- `src/splat3d/raster_grid_wgsl.ts`
  Copy/fork from `src/splat3d/raster_wgsl.ts`. Modify projection and image
  addressing for a fixed 3x3 mosaic. Keep the existing raster untouched.

- `src/splat3d/raster_grid.ts`
  Fork the runtime owner from `src/splat3d/raster.ts`, but only keep what the
  grid path needs: grid forward, grid backward-add, clear, and grid scratch
  buffers. This leaves a clear rollback path.

- `src/splat3d/grid_clip.ts`
  New small helper for grid layout, prompt construction, view ordering, and
  close-up sampling. Keep policy here instead of scattering grid math through
  the optimizer.

- `src/splat3d/optimize.ts`
  Add the gated branch only after the forked grid raster compiles. The default
  path should remain `per-view`.

- `src/splat3d_page.ts` and `src/splat3d.html`
  Add a UI select only after tool benchmarks show the mode is stable:
  `per-view`, `grid 9`, `grid 9 + 2 close`.

- `tools/splat3d/step_bench.ts`
  Add `CLIP_LAYOUT` and print it in the benchmark header.

- `tools/splat3d/step_matrix.ts`
  Add config tokens `grid9` and `gridclose2`.

Grid raster shape:

- output buffer is still full CLIP NCHW, `[3,256,256]`;
- clear the full output to black first;
- for each camera cell, project with a camera prepared for `side=80`;
- write local cell pixel `(x,y)` to full image pixel
  `(cellX + x, cellY + y)`;
- read CLIP gradient from the same full-image pixel during grid backward;
- leave the `8px` gutters black and do not backpropagate through them;
- save per-cell/per-view `derived`, `tileCounts`, `binnedIds`, and `tileStop`,
  because depth order and tile membership are still view-specific.

`grid9` schedule:

1. Clear raw splat grads.
2. Raster all nine cameras into one 3x3 CLIP input image.
3. Run one single-image CLIP train pass with the grid prompt.
4. Grid raster backward accumulates gradients from all nine cells.
5. Adam update.
6. Render display view normally.

`grid9_close2` schedule:

1. Clear raw splat grads.
2. Lane 0: raster the 3x3 grid into batch CLIP lane 0.
3. Lanes 1 and 2: raster two sampled full-resolution camera views into batch
   CLIP lanes 1 and 2.
4. Copy/write text embeddings:
   lane 0 gets the grid prompt;
   lanes 1 and 2 get the normal per-camera prompts.
5. Run one batch-major CLIP x3 train pass.
6. Backward lane 0 through grid raster.
7. Backward lanes 1 and 2 through normal full-resolution raster.
8. Adam update.
9. Render display view normally.

Start with equal lane weighting because the current CLIP trainer effectively
adds lane gradients. If the grid dominates or becomes too weak, add a later
`GRID_LOSS_WEIGHT` gate by scaling the grid input gradient before raster
backward.

## 2. Does It Preserve CLIP Resolution And Signal?

It preserves CLIP input resolution only in the narrow sense that the tensor fed
to MobileCLIP is still `[3,256,256]`.

It does not preserve per-view signal:

- current per-view training gives each selected camera a full `256x256` image;
- `grid9` gives each camera about `80x80` pixels;
- one CLIP embedding supervises the whole contact sheet, not each camera cell;
- CLIP can satisfy the text using the most salient cells and ignore weak cells;
- spatial ordering such as "top left is top-down" is a weak CLIP prior unless
  the model has learned contact sheets well.

So if the requirement is "same CLIP model resolution", grid mode qualifies. If
the requirement is "same visual information per camera", it does not.

The `grid9_close2` variant is the compromise: it keeps all-view coverage while
retaining two full-res per-camera losses each step.

## 3. Likely Speed Impact

Expected speed is workload-dependent:

- `grid9` versus current `9/9, batch CLIP x3`:
  likely faster. It replaces three B=3 CLIP train passes with one B=1 CLIP train
  pass and reduces raster pixel work to roughly one full 256 image worth of
  cells. It still pays nine camera prep/bin passes.

- `grid9` versus current default `3/9, batch CLIP x3`:
  not guaranteed faster. It uses one B=1 CLIP pass instead of one B=3 pass, but
  it also projects/bins all nine cameras. It may be a quality/coverage win more
  than a speed win.

- `grid9_close2` versus current default `3/9, batch CLIP x3`:
  probably similar or a little slower in raw step time. It still uses one B=3
  CLIP pass, but swaps one full-res camera for a nine-cell low-res grid. The
  reason to try it is not obvious speed; it is better multi-view pressure per
  step at roughly default CLIP cost.

The first benchmark should judge grid mode on two axes:

- wall time versus `3/9 x3` and `9/9 x3`;
- visual convergence and multi-view consistency after the same wall-clock time.

## 4. Prompt Wording

Use natural camera language, not math coordinates and not hyphenated camera ids.
The exact camera coordinates already live in the raster camera data; the text
encoder is more likely to respond to ordinary phrases.

Recommended grid prompt:

```text
a 3 by 3 contact sheet of the same cat from nine different camera angles on a black background: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, low rear-left view looking up
```

For a user prompt `P`, build:

```text
a 3 by 3 contact sheet of the same P from nine different camera angles on a black background: ...
```

If `P` already contains "on a black background", do not duplicate it.

For `grid9_close2`:

- lane 0 text: the grid/contact-sheet prompt above;
- close-up lanes: existing `buildViewPrompt(base, camera, blackBg)` prompts.

Do not start with numeric text like "camera at azimuth 90 degrees, elevation 0
degrees". CLIP text embeddings usually handle "right side view" and "top-down
view" better than coordinate prose. Keep the numeric camera definition in
`Camera3D.eye/target/up`.

## 5. Expected Quality Risks

Main risks:

- CLIP may optimize for "collage/contact sheet" rather than the object.
- Each view gets only `80x80`, so fine geometry and face/object details are much
  weaker.
- A single grid loss entangles nine camera views. It does not tell the optimizer
  which cell failed.
- Some cells may become ignored if one or two views strongly satisfy the prompt.
- The model may learn black gutters/borders or grid composition instead of 3D
  structure.
- Low rear/upward views are already semantically hard; at `80x80` they may
  become mostly silhouette/texture pressure.
- If grid and close-up losses are equally weighted, the two full-res views may
  dominate detail while the grid only enforces rough coverage.
- If the grid loss is overweighted, it may pull the object toward a small,
  centered, icon-like shape that reads in all cells but lacks detail.

Mitigations to keep in the first prototype:

- black background and black gutters;
- centered object wording in the base prompt if desired;
- two full-res close-up lanes for detail;
- screenshot/visual gates, not only cosine or ms/step;
- keep the existing per-view mode as default until grid quality is proven.

## 6. First Prototype Files And Benchmark Gates

First prototype should be a forked `v1` path with obvious rollback:

```bash
cp src/splat3d/raster_wgsl.ts src/splat3d/raster_grid_wgsl.ts
cp src/splat3d/raster.ts src/splat3d/raster_grid.ts
```

Then trim/modify the fork instead of changing the default shaders.

Minimal code gates:

```bash
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
GRID_CELL=80 GRID_GUTTER=8 CLIP_LAYOUT=grid9 VIEWS=9 CLIP_BATCH=1 RUNS=3 WARMUP=2 TIMESTAMP=1 bun tools/splat3d/step_bench.ts
GRID_CELL=80 GRID_GUTTER=8 CLIP_LAYOUT=grid9_close2 VIEWS=9 CLIP_BATCH=3 RUNS=3 WARMUP=2 TIMESTAMP=1 bun tools/splat3d/step_bench.ts
TRIALS=3 CONFIGS=base=3:3,all9=9:3,grid9=9:1:grid9,gridclose=9:3:gridclose2 RUNS=4 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

Minimal correctness gates:

- grid image has nine nonblank cells and black gutters;
- grid backward produces finite nonzero raw gradients;
- full-res close-up lanes still produce finite nonzero raw gradients;
- no WebGPU validation errors;
- default `per-view` mode produces identical benchmark output before and after
  the grid code is added.

Promotion gates:

- `grid9` should beat `9/9 x3` wall time by a large margin while looking better
  than `3/9 x3` at equal wall-clock time, or it is just a novelty mode.
- `grid9_close2` should match default `3/9 x3` step time within roughly 15-25%
  and visibly improve multi-view consistency after the same wall-clock time.
- Neither grid mode should become default until screenshot evidence shows it is
  not merely learning a low-resolution contact-sheet aesthetic.

Kill gates:

- if grid-only makes small unreadable blobs, keep it as a blog/screenshot
  ablation but do not optimize further;
- if `grid9_close2` is slower and not visibly more multi-view consistent than
  default `3/9 x3`, drop it;
- if CLIP strongly ignores some cells, consider cell-wise CLIP crops later, but
  that becomes multiple CLIP calls again and loses the main speed thesis.

## Recommendation

Prototype `grid9_close2` first if the goal is art quality. Prototype `grid9`
first if the goal is a speed/coverage experiment against full `9/9`.

Do not describe this internally as "same resolution multi-view CLIP". The honest
description is:

> one full-resolution CLIP image containing nine low-resolution views, optionally
> paired with two full-resolution camera losses.

That makes the tradeoff clear and keeps the ablation easy to judge.
