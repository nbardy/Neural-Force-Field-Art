# Agent Notes - Grid Contact-Sheet CLIP Prompt And Schedule

Date: 2026-07-08

Scope: inspected `src/splat3d/cameras.ts`, `src/splat3d/grid_clip.ts`, `src/splat3d/optimize.ts`, `src/splat3d_page.ts`, `src/splat3d.html`, `tools/splat3d/*`, and existing notes. This is a notes-only pass. I did not modify source files.

## Short Answer

We already have the contact-sheet idea wired into the browser UI. Selecting `3x3 grid + 2` makes lane 0 a 256x256 CLIP image containing a 3x3 contact sheet of all nine views, and lanes 1-2 two full-resolution 256x256 close-up views. The browser also already encodes a dedicated grid prompt with "a 3x3 contact sheet ... from nine camera angles" wording.

The important caveat: current `grid9_close2` keeps CLIP resolution at 256, but it does not yet make raster cheap. Each grid cell is currently produced by rendering a normal 256x256 view into scratch, then nearest-downsampling that view into an 80x80 contact-sheet cell. This lowers effective per-view resolution for CLIP, not the raster workload.

## Current `grid9_close2` Behavior

Files:

- `src/splat3d/cameras.ts`
- `src/splat3d/grid_clip.ts`
- `src/splat3d/optimize.ts`
- `src/splat3d_page.ts`
- `src/splat3d.html`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `tools/splat3d/grid9_close2_test.ts`

Current UI:

- `src/splat3d.html` exposes `per-view CLIP` and `3x3 grid + 2`.
- `src/splat3d_page.ts` forces `clipBatch=3` and `viewsPerStep=9` whenever `grid9_close2` is selected.
- The readout adds `grid+2`, so screenshots show the selected schedule.

Current optimizer schedule:

- `Splat3DClipLayout = "per_view" | "grid9_close2"`.
- `grid9_close2` requires a batch trainer and at least nine cameras.
- Lane 0: one 256x256 CLIP input containing all nine views as an 80px-cell contact sheet with 8px gutters.
- Lane 1: one normal 256x256 close-up view.
- Lane 2: one normal 256x256 close-up view.
- Then one batch-major CLIP train pass runs over the three lanes.
- Grid backward scatters the lane-0 CLIP gradient cell-by-cell into scratch, replays the matching view forward, and calls normal raster backward for that view.
- Close-up lanes call normal raster backward from their batch lane gradients.

Current close-up policy:

- `grid9CloseupViews(gridViews)` returns `[gridViews[step % 9], gridViews[(step + 4) % 9]]`.
- Because `grid9_close2` forces `VIEWS=9`, `sampleViews(9)` returns all camera indices in fixed order. So the current close-up pair is a deterministic rotating pair, not random or epoch-shuffled.
- The `viewSampler` UI does not materially affect grid lane membership today, because grid mode always asks for all nine views.

Current contact-sheet geometry:

- `SIDE = 256`.
- `CELL = 80`.
- `GUTTER = 8`.
- `3 * 80 + 2 * 8 = 256`.
- Cell order is the camera array order:
  `top`, `front`, `right`, `back`, `left`, `front-left-high`, `front-right-high`, `back-right-low`, `back-left-low`.

Current performance interpretation from prior notes:

- `grid9_close2` is much faster than full `9/9` per-view CLIP because it uses one B=3 CLIP pass instead of three B=3 passes.
- It is still slower than default `3/9 batch x3` because it renders/replays all nine grid views plus two close-ups.
- Promotion is a convergence-quality question, not an obvious raw-speed question.

## Do We Already Feed A "3x3 Grid Of A Cat From 9 Angles" Prompt?

Yes, in the browser path, when `3x3 grid + 2` is selected.

Current builder in `src/splat3d/cameras.ts`:

```text
{base}, a 3x3 contact sheet of the same subject from nine camera angles: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up, on a black background
```

For the default prompt, the expanded lane-0 text is effectively:

```text
a photo of a cat, a 3x3 contact sheet of the same subject from nine camera angles: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up, on a black background
```

The browser wires this through `src/splat3d_page.ts`: after view prompts are encoded, grid mode calls `buildGrid9Prompt(...)` and writes it with `opt.setGridPrompt(...)`.

Important exception: the headless benches do not encode real text. `tools/splat3d/step_bench.ts` uses deterministic synthetic embeddings for all view prompts and a separate synthetic embedding for the grid prompt. That is correct for timing and schedule tests, but it does not test prompt wording quality.

## Recommended Exact Prompt Wording

Use natural visual language, not camera coordinate math and not internal labels like `front-left-high` by themselves. The camera coordinates already live in `Camera3D.eye/target/up`; CLIP should get phrases like "right side camera angle" and "elevated front-left camera angle looking down".

Recommended lane-0 grid prompt template:

```text
a 3 by 3 contact sheet showing the same subject, {base}, centered on a black background. The nine panels show, in reading order: top-down camera angle, front-facing camera angle, right side camera angle, rear camera angle, left side camera angle, elevated front-left camera angle looking down, elevated front-right camera angle looking down, low rear-right camera angle looking up, and low rear-left camera angle looking up
```

For the default cat prompt:

```text
a 3 by 3 contact sheet showing the same subject, a photo of a cat, centered on a black background. The nine panels show, in reading order: top-down camera angle, front-facing camera angle, right side camera angle, rear camera angle, left side camera angle, elevated front-left camera angle looking down, elevated front-right camera angle looking down, low rear-right camera angle looking up, and low rear-left camera angle looking up
```

Why this is slightly better than current wording:

- "3 by 3" is often tokenized more plainly than `3x3`.
- "showing the same subject" puts consistency pressure before the camera list.
- "centered on a black background" is stronger for our render than only "on a black background".
- "in reading order" gives CLIP a conventional ordering description without requiring coordinate prose.
- It keeps all camera descriptions natural.

Recommended close-up lane prompt wording should stay as ordinary per-view text:

```text
{base}, centered on a black background, {camera prompt suffix}
```

Example:

```text
a photo of a cat, centered on a black background, a camera angle from the right side
```

I would not start with:

```text
camera at azimuth 90 degrees and elevation 0 degrees
```

That is mathematically precise for us but probably weaker for this CLIP text tower than ordinary visual phrasing.

## Should We Sample Two Close-Ups?

Yes. Keep two full-resolution close-up lanes.

Reason:

- The grid lane gives all-view coverage every step, but each camera is only 80x80 to CLIP.
- The two close-up lanes preserve full 256x256 object/detail gradients.
- The B=3 CLIP call is already the useful batch size on this hardware; `grid + 2` fills that batch with one global low-res view signal plus two high-res detail signals.

Current deterministic pair policy is reasonable for reproducible screenshots, but the next quality ablation should expose the policy:

- `rotating`: current `[step % 9, (step + 4) % 9]`.
- `random`: two distinct cameras per step from the seeded RNG.
- `epoch`: two distinct cameras from the existing shuffled camera order.

Do not remove the two close-ups until a pure `grid9` quality test proves the contact sheet is enough. My expectation is that pure `grid9` will be faster but too weak on detail.

## Keeping CLIP At 256 While Lowering Per-View Effective Resolution

Current implementation already keeps the CLIP tensor shape fixed:

```text
CLIP input: [3, 256, 256]
lane 0:     256x256 contact sheet
cell:       80x80
gutters:    8px black
lanes 1-2:  256x256 full-res close-ups
```

This satisfies "CLIP at the same resolution" because MobileCLIP still sees its trained 256x256 input size.

But there are two different meanings of "lower resolution":

1. CLIP-visible lower per-view resolution: already implemented. Each grid view is downsampled into an 80x80 panel inside a 256x256 CLIP image.
2. Raster-compute lower per-view resolution: not implemented yet. Today each grid view is rendered at 256x256 into scratch and then copied/downsampled into an 80x80 cell.

If we want the contact-sheet schedule to be a real speed lever, the next fork should render each grid cell directly into its 80x80 viewport instead of rendering 256x256 then downsampling.

The clean target:

- Keep the final CLIP buffer as `[3,256,256]`.
- Render cell `i` directly to pixel rectangle:
  `x0 = col * (CELL + GUTTER)`, `y0 = row * (CELL + GUTTER)`.
- Prepare camera projection with `side=CELL`, not `side=256`, for grid cells.
- Write color directly into the cell rectangle of lane 0.
- Read CLIP gradient directly from the same cell rectangle during backward.
- Do not rasterize or backprop through gutters.

## Concrete Implementation Plan

This plan is for the next source-editing pass. This note pass did not modify source files.

1. Improve grid prompt wording.

   File: `src/splat3d/cameras.ts`

   Change only `buildGrid9Prompt(...)` first. Keep `buildViewPrompt(...)` compatible. Recommended output should be the exact lane-0 template above. Also consider changing `buildBasePrompt(...)` from "on a black background" to "centered on a black background" behind the existing black-background toggle, because that helps both grid and close-ups.

2. Make close-up sampling explicit.

   File: `src/splat3d/optimize.ts`

   Add a gated config later, for example:

   ```ts
   gridCloseupSampler?: "rotating" | "random" | "epoch";
   gridCloseupCount?: 0 | 1 | 2;
   ```

   Keep default `gridCloseupCount=2` and default sampler `rotating` so existing behavior does not change accidentally. Then ablate `random` and `epoch` for screenshot quality.

3. Preserve the current UI contract but label it more explicitly if needed.

   Files:

   - `src/splat3d.html`
   - `src/splat3d_page.ts`

   Current label `3x3 grid + 2` is accurate. If changing visible text, prefer:

   ```text
   grid prompt + 2 close-ups
   ```

   Keep forcing `CLIP_BATCH=3` and `VIEWS=9` for this layout. That prevents invalid states.

4. Add a true low-res grid raster fork.

   Suggested fork files:

   - `src/splat3d/raster_grid_wgsl.ts`
   - `src/splat3d/raster_grid.ts`
   - `src/splat3d/grid_clip.ts`
   - `src/splat3d/optimize.ts`

   Do not mangle the main raster shaders first. Copy the current raster path into the fork, then make it cell-viewport aware. The first correctness target is exact behavior for `CELL=256` and sane visual behavior for `CELL=80`.

   Implementation shape:

   - `Grid9Close2ClipLayout` continues to own contact-sheet constants and lane contract.
   - New grid raster path renders directly into lane 0 cell rectangles.
   - Backward reads gradients directly from lane 0 cell rectangles.
   - `gridCopyShader` / `gridScatterShader` become removable after direct viewport rendering is proven.

5. Add prompt/schedule bench and debug controls.

   Files:

   - `tools/splat3d/step_bench.ts`
   - `tools/splat3d/step_matrix.ts`
   - `tools/splat3d/grid9_close2_test.ts`

   Timing flags:

   ```bash
   CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9
   GRID_CELL=80
   GRID_CLOSEUPS=2
   GRID_CLOSEUP_SAMPLER=rotating|random|epoch
   ```

   Existing benches use synthetic embeddings, so they are fine for speed. For prompt wording, use browser screenshots or add a separate browser-driven smoke that actually runs `buildGrid9Prompt(...)` through the text encoder.

6. Quality gates before promotion.

   Commands:

   ```bash
   bun tools/splat3d/grid9_close2_test.ts
   TRIALS=3 CONFIGS=base=3:3,all9=9:3,grid=9:3:grid9 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
   npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
   ```

   Visual tests:

   - fixed prompt: `a photo of a cat`;
   - black background on/off;
   - `per-view CLIP` default versus `3x3 grid + 2`;
   - same wall-clock screenshot sequence, not only same step count;
   - inspect all nine views, not only the active view.

## Recommendation

Keep `grid9_close2` as the main contact-sheet schedule. It is the right compromise: one low-res all-view prompt every step plus two full-res close-ups.

The next real improvement is not another prompt toggle by itself. It is a paired fork:

1. tighten the lane-0 contact-sheet text to the exact wording above;
2. render grid cells directly at 80x80 into the 256x256 CLIP lane so the schedule saves raster work too.

That gives us a cleaner blog story as well: same CLIP input resolution, nine-angle global composition, two high-resolution detail anchors, then a later raster fork that turns the contact sheet from a CLIP-call trick into a real compute win.
