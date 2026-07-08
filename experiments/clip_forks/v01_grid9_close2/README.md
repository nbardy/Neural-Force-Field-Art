# v01_grid9_close2 CLIP layout fork

## Goal

Test `CLIP_LAYOUT=grid9_close2`: keep the CLIP input shape at `[3, 256, 256]`, but spend one CLIP lane on a 3x3 contact sheet containing all 9 camera views, then spend two more CLIP lanes on full-resolution close-up views. This is a signal-efficiency experiment, not a CLIP shader optimization.

The intended batch is:

1. Lane 0: one 256x256 contact sheet with all 9 views.
2. Lane 1: one normal 256x256 render from a sampled close-up camera.
3. Lane 2: one normal 256x256 render from a different sampled close-up camera.

The hypothesis is that lane 0 keeps multi-view consistency pressure alive every step, while lanes 1 and 2 preserve enough full-resolution CLIP detail that the splats do not collapse into a low-res contact-sheet-only solution.

## Gate

Add this as an off-by-default path:

```bash
CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9
```

Rules:

- Require `CLIP_BATCH=3`; fail fast for any other batch size.
- Require at least 9 cameras; fail fast if `DEFAULT_3D_CAMERAS.length < 9`.
- Keep the existing per-view/batch CLIP path unchanged when `CLIP_LAYOUT` is unset or set to `per_view`.
- Keep black background as the render default and keep prompt text explicit about black background.

## Likely Code Files

Implementation should be isolated enough to delete if the ablation loses:

- `src/splat3d/optimize.ts`
  - Add `clipLayout?: "per_view" | "grid9_close2"` to `Splat3DOptimizerConfig`.
  - Route `recordTrainingViews` to the grid path only when `clipLayout === "grid9_close2"`.
  - Add profile timing labels if practical: `gridRasterFwd`, `gridRasterBwd`, `closeupRasterFwd`, `closeupRasterBwd`.
- `src/splat3d/raster.ts`
  - Add a small grid IO/state helper, preferably separate from the existing batch forward state.
  - Reuse existing per-view IO for lanes 1 and 2.
- `src/splat3d/raster_wgsl.ts`
  - Add grid forward/backward shader emitters instead of modifying the normal forward/backward emitters inline.
  - Keep the existing shaders as the correctness reference.
- `src/splat3d/cameras.ts`
  - No camera geometry change expected.
  - Optionally add a single exported `GRID9_PROMPT_SUFFIX`.
- `src/splat3d_page.ts`
  - Add a UI toggle only after the benchmark path works.
  - Suggested label: `3x3 grid + 2 close-ups`.
- `tools/splat3d/step_bench.ts`
  - Read `CLIP_LAYOUT`.
  - Print it in the bench header.
- `tools/splat3d/grid9_close2_test.ts`
  - Add a focused parity/correctness test rather than hiding this inside the main bench.

If this grows large, fork new helper files rather than expanding `optimize.ts`:

- `src/splat3d/grid_clip.ts`
- `src/splat3d/raster_grid.ts`
- `src/splat3d/raster_grid_wgsl.ts`

## Raster Layout

The contact sheet should fill the existing 256x256 CLIP image without resizing CLIP:

```text
cell = 80
gutter = 8
side = 256

x0 = col * (cell + gutter)
y0 = row * (cell + gutter)
```

This gives:

```text
3 * 80 + 2 * 8 = 256
```

Cell mapping:

```text
row 0: top, front, right
row 1: back, left, front-left-high
row 2: front-right-high, back-right-low, back-left-low
```

Each cell renders one camera into its own 80x80 viewport inside lane 0. Gutters and alpha misses are black. Do not write splat color into gutters.

Forward raster options:

- Conservative first implementation: loop 9 views and render each into the lane 0 input buffer with a cell viewport transform.
- Faster later implementation: one grid shader path where each work item knows `cellId`, loads the matching camera, and writes to the proper cell offset.

Backward raster options:

- Conservative first implementation: run 9 cell backward passes that read the lane 0 CLIP gradient from the corresponding cell rectangle and accumulate into shared raw splat gradients.
- Faster later implementation: one grid backward shader with `cellId` in the dispatch domain.

Important correctness detail: lane 0 CLIP gradient outside cells should not create gradients against splats. It may contain CLIP pressure on the black gutters, but the raster backward path should only touch pixels covered by each view cell.

## Prompt Plan

Base prompt should continue to use the black-background toggle:

```text
a photo of a cat, centered on a black background
```

Lane 0 grid prompt:

```text
a 3x3 contact sheet of the same cat from nine camera angles: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up, centered on a black background
```

Lane 1 and lane 2 close-up prompts should use the existing `buildViewPrompt(base, camera, includeBlackBackground)` wording, for example:

```text
a photo of a cat, a camera angle from the right side, on a black background
```

Do not use compact internal labels like `front-left-high` as the only natural-language prompt. They are useful in logs, but CLIP should receive normal visual language like "elevated front-left view looking down".

## View Sampling

For each optimizer step:

- Lane 0 always includes all 9 views.
- Lane 1 and lane 2 sample two distinct cameras.
- Prefer sampling from the same shuffled camera order used by `sampleViews` so this stays comparable to `VIEWS=N`.
- Avoid duplicate close-up cameras in the same step.
- Log the two close-up camera names in debug mode so screenshot runs are reproducible.

Possible first policy:

```text
closeupA = shuffledViews[step % 9]
closeupB = shuffledViews[(step + 4) % 9]
```

The `+4` offset keeps the close-ups separated around the object most of the time.

## Correctness Checks

Add a focused test before benchmarking speed:

```bash
bun tools/splat3d/grid9_close2_test.ts
```

Checks to include:

- Lane 0 writes nonzero RGB into all 9 cells for a small deterministic splat set.
- Lane 0 gutters remain exactly black before CLIP normalization.
- Lane 1 and lane 2 match existing full-resolution per-view raster output for the same camera.
- Grid backward accumulates finite nonzero gradients into splat params.
- Grid backward does not write NaN/Inf gradients.
- With only one active cell, grid backward approximately matches the existing backward path after accounting for the 80x80 viewport transform.
- `CLIP_LAYOUT=per_view` output is byte-identical or numerically identical to the current path.

Screenshot/debug check:

```bash
CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 SEED=1 RUNS=1 WARMUP=0 bun tools/splat3d/step_bench.ts
```

The first implementation should also expose a way to dump the lane 0 image to confirm the 3x3 order visually.

## Perf Metrics

Primary metric:

- End-to-end optimizer step ms with timestamp profiling.

Secondary metrics:

- `rasterFwd`
- `rasterBwd`
- `clipBatch`
- `adam`
- `display`
- total views optimized per step
- subjective image quality after fixed step counts

Expected result shape:

- CLIP time should be close to current `CLIP_BATCH=3`, because CLIP still sees 3 images.
- Raster time may rise because lane 0 renders 9 small views plus 2 full views.
- Quality may improve versus `VIEWS=3` because every step sees all 9 viewpoints.
- Wall clock should be much faster than full `CLIP_BATCH=9` if that path is ever tested, because CLIP is still only batch 3.

## Exact Bench Commands

Baseline current 3-view batch:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=10 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Baseline full 9-view per-view pressure, if supported:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=10 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Grid experiment:

```bash
TIMESTAMP=1 CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 RUNS=10 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Wall-clock smoke without timestamp query:

```bash
CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 RUNS=20 WARMUP=5 bun tools/splat3d/step_bench.ts
```

Cap interaction:

```bash
TIMESTAMP=1 CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 CAP=1024 RUNS=10 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Existing raster gates interaction:

```bash
TIMESTAMP=1 CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 VIEW_LANE_RASTER_FWD=1 VIEW_LANE_RASTER_BWD=1 RUNS=10 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Build check:

```bash
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

## Risks

- The 80x80 cells may be too low-res for CLIP to give useful object-detail gradients.
- CLIP may interpret the contact sheet as a collage instead of a single object with consistent 3D structure.
- Contact-sheet prompt length may dilute the base object prompt.
- Grid gutters may attract unwanted CLIP pressure unless black background and gutter handling are consistent.
- The grid backward path may be easy to get subtly wrong because pixel-to-cell coordinates change the raster derivative scale.
- If raster cost dominates after this change, the layout may be quality-positive but not speed-positive.
- If lane 1 and lane 2 always sample similar angles, the optimizer may overfit those two views while lane 0 provides only weak low-res pressure.

## Recommendation

Build the conservative version first: 9 separate 80x80 cell renders into lane 0 plus two existing 256x256 close-up renders. Do not start with a clever multi-view shader. The first question is whether `grid9_close2` gives better convergence per CLIP batch. If the signal works, then optimize the grid raster into a single shader path.

## Measured First Pass

Implemented gate:

```bash
CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9
```

Implementation notes:

```text
src/splat3d/grid_clip.ts owns the contact-sheet copy/scatter shaders.
Lane 0 is cleared to black, then filled with 9 nearest-sampled 80x80 cells.
Lanes 1 and 2 remain normal 256x256 close-up renders.
Grid backward scatters the lane-0 cell gradient into a scratch full-res gradient.
The raster forward is replayed before each grid-cell backward pass so the binned/tile state matches that camera.
```

Smoke test:

```bash
bun tools/splat3d/grid9_close2_test.ts
```

Result:

```text
PASS grid9_close2 contact sheet
cells=[0.830, 0.774, 0.761, 0.793, 0.807, 0.751, 0.735, 0.760, 0.758]
gutter=0
```

Timestamp comparison, one run, no warmup:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=1 WARMUP=0 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 RUNS=1 WARMUP=0 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=0 bun tools/splat3d/step_bench.ts
```

Results:

```text
9-view per-view batch: total=154.80 ms raster=33.49 ms clipBatch=120.91 ms
grid9_close2:         total=84.93 ms raster=44.04 ms clipBatch=39.98 ms
3-view per-view batch: total=64.29 ms raster=12.91 ms clipBatch=50.86 ms
```

Matrix wrapper:

```bash
TRIALS=1 CONFIGS='base9=9:3,grid=9:3:grid9' RUNS=1 WARMUP=0 bun tools/splat3d/step_matrix.ts
```

Result:

```text
base9 profile=162.86 ms clip=121.39 ms raster=37.82 ms
grid  profile=89.06 ms  clip=40.03 ms  raster=45.86 ms
```

Conclusion: the first conservative grid path is a clear speed win versus full
9-view CLIP because it runs one B=3 CLIP call instead of three. It is still
slower than ordinary 3-view training because it renders/replays all 9 grid views
plus 2 close-ups. Promotion now depends on convergence/image quality, not raw
step speed alone.
