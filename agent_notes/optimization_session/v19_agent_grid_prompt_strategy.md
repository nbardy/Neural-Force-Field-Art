# v19 Agent Grid Prompt Strategy

Date: 2026-07-08

Scope: inspected `src/splat3d/grid_clip.ts`, `src/splat3d/cameras.ts`,
`src/splat3d/optimize.ts`, the relevant browser UI in `src/splat3d_page.ts` and
`src/splat3d.html`, and the `v01`, `v08`, and `v16` fork docs. This is a
notes-only pass. I did not edit source code.

## Short Recommendation

Treat the user's proposed prompt,

```text
3x3 grid of a cat from 9 different angles
```

as a prompt-quality variant on top of the existing `grid9_close2` schedule, not
as a new renderer layout. The current code already keeps the CLIP input at
`[3,256,256]`, puts a 3x3 contact sheet in batch lane 0, and uses batch lanes 1
and 2 for two full-resolution per-view renders.

The best prompt to test first is:

```text
a 3x3 grid of the same cat from 9 different camera angles, centered on a black background. The 9 panels show the cat in reading order: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up
```

Also test the user's minimal literal wording as a control:

```text
a 3x3 grid of a cat from 9 different angles, centered on a black background
```

Keep the two "close-up" lanes as the current full-resolution camera-prompt lanes:

```text
a photo of a cat, a top-down camera angle, on a black background
a photo of a cat, a camera angle from the right side, on a black background
```

Do not add the phrase "close-up" to those lane prompts unless the camera/FOV is
actually changed. In current code they are close-ups only relative to the 80x80
grid cells; they are normal 256x256 renders with the same camera geometry.

## Current Behavior

### Cameras and Prompt Builders

`src/splat3d/cameras.ts` defines nine cameras in this order:

```text
top
front
right
back
left
front-left-high
front-right-high
back-right-low
back-left-low
```

Each camera has a natural-language suffix, for example:

```text
a top-down camera angle
a front-facing camera angle
a camera angle from the right side
an elevated 45 degree camera angle from the front left looking down
```

Current per-view text comes from `buildViewPrompt(base, camera, includeBlackBackground)`.
For the default prompt, the view text list is:

```text
a photo of a cat, a top-down camera angle, on a black background
a photo of a cat, a front-facing camera angle, on a black background
a photo of a cat, a camera angle from the right side, on a black background
a photo of a cat, a camera angle from behind, on a black background
a photo of a cat, a camera angle from the left side, on a black background
a photo of a cat, an elevated 45 degree camera angle from the front left looking down, on a black background
a photo of a cat, an elevated 45 degree camera angle from the front right looking down, on a black background
a photo of a cat, a low 45 degree camera angle from the rear right looking up, on a black background
a photo of a cat, a low 45 degree camera angle from the rear left looking up, on a black background
```

Current grid-lane text comes from `buildGrid9Prompt(base, includeBlackBackground,
"contact_sheet")`. For the default cat prompt it expands to:

```text
a 3x3 image grid showing the same subject, a photo of a cat, from nine different camera angles, centered on a black background: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up
```

There is also `Grid9PromptMode = "contact_sheet" | "same"`. `"same"` makes lane
0 use the normal base prompt instead of a grid/contact-sheet prompt.

### Grid CLIP Layout

`src/splat3d/grid_clip.ts` hard-codes the contact-sheet geometry:

```text
SIDE = 256
CELL = 80
GUTTER = 8
3 * CELL + 2 * GUTTER = 256
```

Lane 0 is still a normal CLIP-sized 256x256 RGB image. The nine panel origins are:

```text
cell 0: x=0,   y=0
cell 1: x=88,  y=0
cell 2: x=176, y=0
cell 3: x=0,   y=88
cell 4: x=88,  y=88
cell 5: x=176, y=88
cell 6: x=0,   y=176
cell 7: x=88,  y=176
cell 8: x=176, y=176
```

The grid helper can operate in two raster modes:

- `directRaster=false`: render each grid view at 256x256 scratch, then nearest
  downsample into its 80x80 cell.
- `directRaster=true`: create an 80x80 scratch raster with cameras prepared for
  side 80, then copy each 80x80 result into the 256x256 lane-0 contact sheet.

Both modes feed CLIP the same 256x256 lane. `directRaster=true` is the speed
candidate from v16 because it lowers grid raster work while preserving CLIP's
native input resolution.

### Optimizer Schedule

`src/splat3d/optimize.ts` exposes:

```text
Splat3DClipLayout = "per_view" | "grid9_close2"
```

`grid9_close2` requires:

```text
CLIP_BATCH=3
VIEWS=9
at least 9 cameras
```

One optimization step records:

```text
lane 0: 256x256 contact sheet containing all nine cameras as 80x80 cells
lane 1: one full-resolution 256x256 render from a selected camera
lane 2: one full-resolution 256x256 render from another selected camera
```

Then a single batch-major CLIP forward/backward pass trains all three lanes.
Backward for the grid lane scatters the lane-0 CLIP gradient cell by cell and
accumulates it into the shared splat gradients. Backward for lanes 1 and 2 uses
the ordinary per-view raster backward path.

Current close-up selection is deterministic:

```text
closeupA = gridViews[step % 9]
closeupB = gridViews[(step + 4) % 9]
```

Because grid mode forces `VIEWS=9`, `gridViews` is usually all cameras in fixed
array order. So "sample 2 close-up views" in the current implementation means a
reproducible rotating pair, not a random pair. This is fine for deterministic
benchmarks. If the desired product behavior is true sampling, add a separate
close-up sampler gate later and do not mix that with the prompt ablation.

### Browser UI

`src/splat3d.html` and `src/splat3d_page.ts` already expose the relevant knobs:

```text
CLIP layout: per-view CLIP | 3x3 grid + 2
grid prompt: grid prompt | same grid text
grid raster: grid raster 256 | grid raster 80
```

When `3x3 grid + 2` is selected, the page forces:

```text
clipBatchSize = 3
viewsPerStep = 9
```

The page then:

1. Encodes the nine per-view prompts with `buildViewPrompt(...)`.
2. Calls `opt.setViewPrompts(...)`.
3. Encodes the lane-0 grid prompt with `buildGrid9Prompt(...)`.
4. Calls `opt.setGridPrompt(...)`.

The readout includes `grid+2`, `grid text` or `grid=same text`, and `80px grid
raster` when applicable, so screenshots can prove which path is active.

## Difference From Existing `grid9_close2`

The proposed user prompt does not change the tensor size, cell geometry, number
of CLIP lanes, or renderer schedule.

It differs from current `grid9_close2` in three narrower ways:

1. The prompt's head noun becomes literal and object-specific.

   Current:

   ```text
   a 3x3 image grid showing the same subject, a photo of a cat, from nine different camera angles...
   ```

   Proposed:

   ```text
   a 3x3 grid of the same cat from 9 different camera angles...
   ```

   This removes the awkward phrase "same subject, a photo of a cat" and makes
   CLIP see "grid of cat" early.

2. The proposed prompt should preserve the reading-order camera list.

   The short user text alone does not tell CLIP which panel corresponds to which
   camera. The current grid layout has a fixed camera order, so the stronger
   prompt should still list:

   ```text
   top-down, front, right, rear, left, elevated front-left, elevated front-right, low rear-right, low rear-left
   ```

3. "Sample 2 close-up views" should be separated from "change prompt text".

   Current code already uses two full-resolution lanes. It rotates the pair
   deterministically. A true random or epoch sampler is a separate schedule
   ablation. For a clean prompt test, hold the current rotating pair constant.

## Prompt Text To Use

### Primary Lane-0 Prompt

Use this for the first real quality test:

```text
a 3x3 grid of the same cat from 9 different camera angles, centered on a black background. The 9 panels show the cat in reading order: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up
```

Reasons:

- It matches the user's wording closely.
- It says "cat" directly instead of nesting the base phrase inside "same
  subject".
- It keeps "same cat" consistency pressure.
- It keeps black-background alignment with the render.
- It preserves the cell order that `grid_clip.ts` actually uses.

### Minimal Literal Control

Use this as a control to see whether the long camera list helps or hurts:

```text
a 3x3 grid of a cat from 9 different angles, centered on a black background
```

This is closest to the user's exact proposal, but it gives CLIP less structure
about the grid panels.

### Current-Prompt Control

Keep the current `contact_sheet` prompt as the control:

```text
a 3x3 image grid showing the same subject, a photo of a cat, from nine different camera angles, centered on a black background: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up
```

### Same-Text Negative Control

Keep the current `same` grid text as a negative/control variant:

```text
a photo of a cat, on a black background
```

If the literal grid prompt cannot beat `same`, the contact-sheet text is not
helping this CLIP tower.

### Close-Up Lane Prompts

Do not change lane 1 and lane 2 text for the prompt test. They should stay as
normal per-view prompts:

```text
a photo of a cat, {camera prompt suffix}, on a black background
```

Examples:

```text
a photo of a cat, a front-facing camera angle, on a black background
a photo of a cat, a low 45 degree camera angle from the rear right looking up, on a black background
```

If a later fork adds actual close-up cameras, then the prompt should say
"close-up". With current cameras, saying "close-up" would describe a crop/zoom
that is not rendered.

## Benchmark And Quality Gate

### What Not To Use As The Prompt Gate

Do not judge prompt wording with `tools/splat3d/step_bench.ts` alone. That tool
uses synthetic deterministic text embeddings, so it is good for timing and
schedule regressions but cannot evaluate a real text prompt.

Do not use the grid-lane CLIP score as the main metric. v16 made the right call:
judge the trained splats by full-resolution per-view MobileCLIP teacher scores
for all nine cameras.

### Current Baseline From v16

v16 measured the real MobileCLIP prompt path with:

```bash
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat" CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid OUT_DIR=/tmp/nffa_grid_quality_cat_trials bun tools/splat3d/grid_quality.ts
```

Median results:

| Variant | Steps / 5s | Steps / Sec | Mean Cos | Mean Delta | Min Cos | Min Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base3` | 91 | 18.02 | 0.25410 | 0.12579 | 0.20980 | 0.09576 |
| `grid80` current prompt | 78 | 15.55 | 0.23802 | 0.11083 | 0.19165 | 0.08511 |

Read:

- Current `grid80` reached about `88.1%` of `base3`'s median mean-cosine
  improvement.
- Current `grid80` reached about `88.9%` of `base3`'s median min-cosine
  improvement.
- This is good enough to keep gated, but not good enough to promote as default.

The proposed prompt should be considered useful only if it closes a meaningful
part of this gap.

### Required Harness Capability

The current `grid_quality.ts` parser supports `contact_sheet` and `same` grid
prompt modes. To test the proposed prompt cleanly, a later source-editing pass
should add either:

```text
GRID_PROMPT_TEXT="..."
```

or new config tokens such as:

```text
literalshort
literalordered
```

Do this in the quality harness first. The browser UI does not need a new visible
option until the prompt wins.

### Correctness Gate

Run before quality comparisons, especially if any prompt-mode plumbing touches
shared grid code:

```bash
bun tools/splat3d/grid9_close2_test.ts
GRID_DIRECT_RASTER=1 bun tools/splat3d/grid9_close2_test.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Expected correctness properties:

- Lane 0 writes nonzero RGB into all nine cells.
- Gutters remain black.
- Direct 80x80 grid raster still fills all cells.
- No NaN or Inf gradients.
- The default `per_view` path remains unchanged.

### Prompt-Quality Gate

Primary prompt-quality run should use fixed steps, because prompt text should not
change runtime. Fixed steps isolates gradient usefulness.

Suggested command after adding prompt-mode support:

```bash
TRIALS=5 RUN_STEPS=80 PROMPT="a photo of a cat" CONFIGS=base3=3:3,grid_current=9:3:grid9:directgrid,grid_literal_short=9:3:grid9:directgrid:literalshort,grid_literal_ordered=9:3:grid9:directgrid:literalordered,grid_same=9:3:grid9:directgrid:same OUT_DIR=/tmp/nffa_v19_grid_prompt_steps bun tools/splat3d/grid_quality.ts
```

Secondary user-facing run should use the v16 fixed wall-clock budget:

```bash
TRIALS=5 BUDGET_MS=5000 PROMPT="a photo of a cat" CONFIGS=base3=3:3,grid_current=9:3:grid9:directgrid,grid_literal_short=9:3:grid9:directgrid:literalshort,grid_literal_ordered=9:3:grid9:directgrid:literalordered,grid_same=9:3:grid9:directgrid:same OUT_DIR=/tmp/nffa_v19_grid_prompt_budget bun tools/splat3d/grid_quality.ts
```

Use `grid80` / `directgrid` for the primary gate because v16 identified it as
the speed candidate. The final CLIP tensor remains 256x256 even when direct grid
raster is enabled.

### Promotion Thresholds

Use median results across trials and inspect per-view scores, not just the mean.

To replace the current `contact_sheet` wording with the proposed ordered prompt:

- `grid_literal_ordered` median `meanDelta` should beat `grid_current` by at
  least `+0.005` absolute.
- `grid_literal_ordered` median `minDelta` should not be worse than
  `grid_current` by more than `0.002` absolute.
- No individual camera should regress versus `grid_current` by more than `0.010`
  in most trials.
- Generated view sheets should not show a single-view overfit, black collapse,
  or loss of side/rear structure.

To promote `grid9_close2 + directgrid + proposed prompt` as a page default over
`base3`, require a stronger gate:

- Median `meanDelta` at least `95%` of `base3`.
- Median `minDelta` at least `95%` of `base3`.
- Median step rate at least `85%` of `base3`.
- Visual sheets show all nine views improving, not only the two full-resolution
  close-up lanes.

The v16 `grid80` result did not clear this stronger default gate. It was around
`88%` of `base3`'s improvement, so the proposed prompt needs to produce a real
quality lift, not merely tie the current prompt.

### Timing Regression Gate

Prompt text should not materially affect optimizer step time after text encoding.
Still run the integrated timing matrix if any code path changes:

```bash
TRIALS=3 CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

Expected outcome:

- `grid80` timing should stay close to the v16 profile shape.
- `clipBatch` should stay near the same batch-3 cost.
- Raster cost should not move due to prompt changes.

## Close-Up Sampling Follow-Up

The user's phrase "plus sample 2 close-up views" is already partly implemented
as two full-resolution lanes, but current selection is deterministic. If a later
agent wants to test true sampling, add it as a separate gate:

```text
gridCloseupSampler = rotating | random | epoch
gridCloseupCount = 0 | 1 | 2
```

Default should remain:

```text
rotating, count=2
```

Benchmark it separately from prompt wording:

```bash
TRIALS=5 RUN_STEPS=80 PROMPT="a photo of a cat" CONFIGS=grid_rotating=9:3:grid9:directgrid:literalordered:rotating,grid_random=9:3:grid9:directgrid:literalordered:random,grid_epoch=9:3:grid9:directgrid:literalordered:epoch OUT_DIR=/tmp/nffa_v19_closeup_sampler bun tools/splat3d/grid_quality.ts
```

Do not test close-up sampler and prompt wording in the same first pass; otherwise
we will not know which change moved quality.

## Final Read

The proposed CLIP contact-sheet prompt is worth testing, but it is not a new
architecture. Current `grid9_close2` already implements the important data path:
same 256x256 CLIP resolution, one all-view 3x3 grid lane, and two full-resolution
detail lanes. The open question is whether a more literal cat/grid prompt gives
MobileCLIP a stronger lane-0 gradient than the v08 `same subject, a photo of a
cat` wording.

The right proof is a real-text `grid_quality.ts` gate against the existing
`contact_sheet` prompt and the `same` negative control, scored by full-resolution
per-view teacher cosines over all nine cameras. If the ordered literal prompt
closes the v16 quality gap without hurting min-view score, it should replace the
current grid prompt. If it only ties, keep the current prompt and move effort to
close-up sampling or periodic full per-view refresh.
