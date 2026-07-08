# Agent v15 Multiview Raster Big Leaps

Date: 2026-07-08

Scope: inspected `src/splat3d/raster.ts`, `src/splat3d/raster_wgsl.ts`,
`src/splat3d/grid_clip.ts`, `src/splat3d/cameras.ts`,
`src/splat3d/optimize.ts`, and the existing optimization notes around
worldtube, static multiview raster, view-lane raster, grid contact sheets,
occupancy telemetry, cap testing, and integrated timestamp profiling.

This is a notes-only pass. No runtime source was changed.

## Current Read

The exact view-lane scheduler idea has already been tried. `Raster3DEngine`
now owns a compact camera buffer through `serializeCameras3D()` and can create
lane-strided raster state in `createBatchForwardState()`. The generated batch
WGSL path uses `activeViews[lane]` and `workgroup_id.z` in:

- `prepBatchShader3D()`
- `emitBatchShader3D()`
- `forwardBatchShader3D()`
- `backwardBatchShader3D()`

The runtime entry points are:

- `Raster3DEngine.recordBatchForward()`
- `Raster3DEngine.recordBatchBackwardAdd()`
- `Raster3DEngine.laneScratchState()`

Correctness passed exactly in `tools/splat3d/raster_batch_forward_test.ts`, but
the prior timing notes say it did not clear the full optimizer promotion bar.
Forward-only view-lane raster was flat or worse, and view-lane backward saved
some sampled raster time but did not improve the default `3/9, CLIP batch x3`
step.

The practical big step that did work is schedule-level, not STAR math:
`grid9_close2` plus `GRID_DIRECT_RASTER=1`. In `Grid9Close2ClipLayout.create()`,
`directRaster` creates a shared-parameter scratch raster at `CELL=80` instead
of `SIDE=256`. That keeps CLIP input at `256x256` but makes the nine grid cells
cheap raster views. Recorded result from `docs/SPLAT3D_PERF_NOTES.md`:

```text
grid9_close2              normal 87.93 ms  raster 46.01 ms
grid9_close2 + directgrid normal 59.03 ms  raster 18.76 ms
grid80 + depthwise4       normal 56.08 ms
grid80 + dw4 + bwd fusions normal 54.63 ms
```

That is the closest thing here to a real multiview leap: all nine cameras are
touched every step at near default `3/9` wall time. It is not the same objective
as nine independent full-resolution CLIP losses, so quality is the open gate.

## What Is Plausible Now

### 1. Make `directgrid` the real multiview branch, then remove its replay/copy tax

Plausible now and probably the best raster-specific target.

Current `grid9_close2` direct raster still has extra structure:

- `recordGrid9Close2Inputs()` renders each grid cell into an `80x80` scratch
  image and then calls `Grid9Close2ClipLayout.recordCopyCell()`.
- `recordGrid9Close2Backward()` scatters CLIP lane-0 gradients back to an
  `80x80` scratch grad, replays that cell forward, then calls
  `recordBackwardAdd()`.

The ambitious exact version is a dedicated grid-cell raster state:

```text
derived[cell, splat]
tileCounts[cell, tile80]
binnedIds[cell, tile80, slot]
tileStop[cell, tile80]
image = CLIP batch lane 0 contact sheet, addressed with cell origin
gradImage = CLIP lane 0 gradient, addressed with the same cell origin
```

Concretely:

- Add a grid-cell variant near `src/splat3d/grid_clip.ts`, not by mutating the
  default full-view path first.
- Reuse the camera-buffer and `workgroup_id.z` lane pattern from
  `Raster3DEngine.createBatchForwardState()`.
- Add viewport-aware forward/backward WGSL that writes/reads
  `cellOrigin(cell) + localPixel`.
- Keep all nine cell forward states until after CLIP backward so grid backward
  does not replay the nine cells.
- Then `gridCopyShader()` and the scatter path become removable for this branch.

Expected win: smaller than the first direct-raster jump, but real if replay,
copy, and scatter are still a meaningful part of `rasterReplay + rasterBwd`.
It also simplifies the mental model: one contact-sheet raster state, one CLIP
lane, no temporary image shuttle.

Benchmark gates:

```bash
GRID_DIRECT_RASTER=1 bun tools/splat3d/grid9_close2_test.ts
bun tools/splat3d/grid9_close2_test.ts
TRIALS=5 CONFIGS=grid=9:3:grid9:directgrid,gridnew=9:3:grid9:directgrid RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
TIMESTAMP=1 GRID_DIRECT_RASTER=1 CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
```

Promotion bar:

- `grid9_close2_test` passes: all cells populated, gutters black.
- Gradient parity against current `directgrid` for fixed synthetic CLIP grads,
  with max raw-gradient diff no worse than the existing raster batch gate.
- At least `8%` lower `rasterReplay + rasterBwd` for `grid9_close2 directgrid`.
- No regression in normal median versus current `grid80 + depthwise4` stack.

### 2. Fuse grid-cell backward into raw parameter-gradient atomics

Plausible now, but riskier than the viewport state cleanup.

Current backward has two stages:

```text
backwardShader3D/backwardBatchShader3D -> accGrad[derived slots]
chainAddShader3D(camera)               -> gradRaw[param slots]
```

For grid cells and view lanes, `recordBatchBackwardAdd()` still runs
camera-specific `chainAdd` sequentially per lane. A direct raw-gradient
backward would compute the chain terms inside the tile backward shader and
atomic-add fixed-point raw parameter gradients directly.

Concrete files/functions:

- `src/splat3d/raster_wgsl.ts`
  - fork `backwardBatchShader3D()`;
  - inline the camera math from `chainAddShader3D()`;
  - write fixed-point atomics into raw param-gradient buffers, not
    `accGrad`.
- `src/splat3d/raster.ts`
  - add a gated bind/state path alongside `recordBatchBackwardAdd()`;
  - preserve the existing path as the correctness oracle.

Why it might win:

- removes `accGrad[lane, splat, derivedSlot]` clear traffic;
- removes one `chainAdd` dispatch per active view/cell;
- removes derived-gradient memory traffic between tile backward and chain.

Why it might lose:

- tile backward already has heavy per-pixel work and atomics;
- inlining chain math increases register pressure;
- direct atomics to raw params may increase contention on color/opacity slots.

Benchmark gates:

```bash
bun tools/splat3d/raster_batch_forward_test.ts
TIMESTAMP=1 VIEW_LANE_RASTER_BWD=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
TRIALS=5 CONFIGS=base=3:3,rawbwd=3:3:viewbwd RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
TRIALS=5 CONFIGS=grid=9:3:grid9:directgrid,gridraw=9:3:grid9:directgrid RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

Promotion bar:

- parity gate against current raw gradients;
- timestamped raster backward down by at least `15%`;
- full normal-step median down by at least `5%` on the grid stack, not only on
  isolated raster timing.

### 3. Treat contact-sheet training as the multiview objective, with periodic full-view refresh

Plausible now and likely higher leverage than another exact raster scheduler.

Static cameras do not make exact per-view raster sublinear, but the optimizer
does not always need the exact nine independent full-res CLIP losses. A useful
training schedule could be:

```text
most steps:  grid9_close2 + directgrid + two full-res closeups
every M:     full 9/9 per-view batch x3 refresh
optional:   default 3/9 per-view steps between refreshes
```

Concrete code locations:

- `Splat3DOptimizer.recordTrainingViews()` chooses the layout branch.
- `Splat3DOptimizer.grid9CloseupViews()` currently uses deterministic rotating
  closeups.
- `src/splat3d/cameras.ts` owns `buildGrid9Prompt()` and per-camera prompts.
- `tools/splat3d/step_matrix.ts` already has `grid9` and `directgrid` tokens.

This is not mathematically equivalent to the baseline loss. It is a cheaper
supervision schedule using real CLIP. The right gate is quality per wall-clock
minute, not gradient parity.

Benchmark/quality gates:

```bash
TRIALS=5 CONFIGS=base=3:3,all9=9:3,grid80=9:3:grid9:directgrid,grid80dw4=9:3:grid9:directgrid:dw4 RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
node tools/smoke.mjs http://localhost:1234/splat3d.html 8000 /tmp
```

Promotion bar:

- wall time within `10%` of current `3/9` default;
- visibly better 9-view consistency after the same wall-clock budget;
- no obvious collapse into "contact sheet texture" instead of the object;
- periodic full-view refresh improves or stabilizes quality enough to justify
  its extra cost.

### 4. Multi-camera prep for shared splat attributes

Plausible now but probably modest.

`prepBatchShader3D()` currently runs per `(splat, lane)`, so color sigmoid,
opacity sigmoid, radius exp/clamp, and param reads are repeated for each lane.
A packed multi-camera prep could run one invocation per splat, compute shared
attributes once, then loop over `B` active cameras and write
`derived[lane, splat]`.

Concrete files:

- `src/splat3d/raster_wgsl.ts`: new `prepPackedViewsShader3D()`.
- `src/splat3d/raster.ts`: alternate prep pipe in `createBatchForwardState()`.

This only helps prep, not binning, sorting, compositing, or CLIP. It should not
be prioritized unless timestamps prove prep is a measurable slice.

Gate:

```bash
TIMESTAMP=1 VIEW_LANE_RASTER_FWD=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
TRIALS=5 CONFIGS=view=3:3:viewlane,packed=3:3:viewlane RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

Promotion bar: only continue if timestamped raster forward drops by at least
`10%` and full normal median does not regress. Otherwise prep is not the wall.

### 5. Tile-private reductions to reduce global atomic pressure

Plausible as a researchy WebGPU kernel experiment, not a quick default.

The backward shaders currently let each pixel thread atomic-add derived
gradients into global `accGrad`. For dense tiles, many pixels hit the same
splat. A more aggressive shader would aggregate contributions for a small
chunk of splats inside the tile workgroup and issue fewer global atomics.

Concrete target:

- `backwardShader3D()` and `backwardBatchShader3D()` in
  `src/splat3d/raster_wgsl.ts`.

Hard part: alpha compositing backward is a per-pixel reverse transmittance walk.
The shader can stage IDs, but reducing 11 gradient slots per splat across 256
pixel threads is workgroup-memory and barrier heavy. It may trade global atomic
pressure for worse occupancy.

Gate:

- first add timestamp sub-attribution or a microbench with synthetic dense
  tiles;
- only implement if raster backward remains above `15 ms` on the target grid
  schedule after the simpler grid cleanup.

## What Is Plausible But Approximate

### 6. Stale bins / stale sorted tile lists

Static cameras make binning topology somewhat stable across optimizer steps, but
the splat positions, radii, opacity, and color still change every Adam update.
Reusing `tileCounts`, `binnedIds`, or depth order for several steps is therefore
not exact.

Still, it is a legitimate big-leap ablation:

```text
refresh bins/sorts every M steps
between refreshes: reuse tile lists, recompute derived, composite/backward
```

This needs a hard quality gate because stale visibility can bias gradients.

Concrete files:

- `Raster3DEngine.recordForward()` currently always runs
  `prep -> clearBins -> emit -> forward`.
- A stale-bin branch would skip `clearBins/emit` and reuse `binnedIds`, but only
  after checking whether `forwardShader3D()` can tolerate updated derived depth
  with old membership and order. It probably cannot be exact because old sorted
  order may be stale.

Gate:

- synthetic image/gradient diff against exact for one-step, two-step, four-step
  stale windows;
- full optimizer visual comparison at same wall-clock;
- kill immediately if max image diff or raw-gradient cosine is unstable.

### 7. Depth-prefix / nearest-K tile culling

Telemetry showed default `G=4096` had no overflow and max tile count below
`1024`, but max `tileStop` was much lower than max count. That suggests much of
the far sorted tail often does not affect compositing.

An approximate branch could keep only a nearest-depth prefix per tile, or use
previous `tileStop + margin` as a cap. This attacks sort/composite work, not
CLIP. It is not exact unless the discarded tail is guaranteed below the
transmittance cutoff for every pixel in the tile.

Concrete target:

- `emitShader3D()` and `emitBatchShader3D()` currently append the first
  `cap` atomic arrivals, not nearest splats.
- `forwardShader3D()` sorts and walks `count`; `tileStop` is only known after
  forward.

Gate:

- start with analysis in `tools/splat3d/raster_telemetry.ts`: report
  `stop/count` by view over optimization steps, not just initialization;
- test exact image/gradient error for `K=384/512/768` nearest-depth prefixes;
- promote only as an opt-in quality/speed mode if wall-clock convergence wins.

## Speculative Math, Not A Browser Patch Yet

The existing worldtube notes are right: STAR UVT gets speed by changing the
primitive contract. It is not just "batch views." Time is an ordered dimension;
our nine cameras are a discrete rig with different projection maps, visibility,
tile footprints, and depth order.

The following ideas are ambitious but speculative. They should not be described
as exact optimizations of the current splat raster.

### A. Camera-bundle splats

Represent each splat's projected center/radius as a low-order rational function
over a camera coordinate, then raster a bundle over `(u, v, camera)`.

Why it is attractive:

- one primitive could cover several nearby camera samples;
- it is the closest analog to STAR PRT / worldtube math.

Why it is speculative here:

- the default rig has top/front/side/back/high/low views, not a smooth camera
  trajectory;
- depth order can swap between views;
- alpha compositing remains camera-specific unless the primitive accepts
  approximation.

Possible narrower version: fit piecewise bundles for camera clusters:
equator views, high front diagonals, low rear diagonals, top. That is a new
primitive and a new backward chain, not a shader refactor.

### B. Atlas-residual multiview splats

Choose one or more canonical screen/depth atlases and store per-view residuals
from atlas coordinates to camera cells. Render the atlas once, warp or gather
into views/contact-sheet cells.

Why it might fit this project:

- the camera rig is fixed and small;
- grid/contact-sheet training already accepts lower effective per-view
  resolution.

Why it is not exact:

- disocclusion and per-view depth order are the hard part;
- a canonical atlas can invent visibility where a camera should see occlusion;
- CLIP gradients would have to be pulled back through the warp and residual
  model.

This is a quality/speed research branch, not a replacement for
`Raster3DEngine`.

### C. Ray-space / Plucker-fiber primitive

Represent density directly over camera rays or ray bundles, then evaluate the
fixed rig by querying the same ray-space object.

Why it is interesting:

- aligns with the "rays as fibers" language;
- could make static camera sets native rather than repeatedly projecting world
  Gaussians.

Why it is speculative:

- the current optimizer owns world-space splat params and Adam updates those
  params directly;
- alpha compositing still needs a per-camera order along each ray;
- converting CLIP image gradients back into world/object parameters is a new
  inverse problem.

## Suggested Priority

1. Use `grid9_close2 + GRID_DIRECT_RASTER=1` as the main all-view candidate.
   Its speedup is already measured; the missing gate is visual convergence.
2. If touching raster code, remove directgrid replay/copy/scatter first with a
   viewport-aware grid-cell batch state.
3. Only after that, test direct raw-gradient atomics for grid/view-lane
   backward.
4. Do not spend more time on shallow view-lane dispatch or `CAP=1024`; both
   already passed correctness but failed the integrated promotion bar.
5. Treat stale bins, nearest-K prefixes, and camera-bundle/worldtube math as
   approximate research branches with explicit quality gates.

