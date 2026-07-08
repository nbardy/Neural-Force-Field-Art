# v19 Grid Literal Prompt

Date: 2026-07-08

## Goal

Test the prompt wording idea:

```text
a 3x3 grid of a cat from 9 different angles
```

The renderer/layout does not change. `grid9_close2` already feeds CLIP a normal
`3x256x256` image in lane 0 containing a 3x3 contact sheet, plus two
full-resolution close-up lanes in lanes 1 and 2. This fork only adds a third
grid-prompt mode so the prompt can match that contact-sheet image more directly.

## Runtime Gate

Browser UI:

```text
CLIP layout: 3x3 grid + 2
grid prompt: literal grid
grid raster: grid raster 80
```

Quality harness token:

```bash
CONFIGS='base3=3:3,grid80=9:3:grid9:directgrid,grid80literal=9:3:grid9:directgrid:literal'
```

## Prompt Modes

Existing `contact_sheet` mode:

```text
a 3x3 image grid showing the same subject, a photo of a cat, from nine different camera angles, centered on a black background: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up
```

New `literal` mode:

```text
a 3x3 grid showing a photo of a cat from 9 different camera angles, centered on a black background. The 9 panels show the same subject in reading order: top-down view, front-facing view, right side view, rear view, left side view, elevated front-left view looking down, elevated front-right view looking down, low rear-right view looking up, and low rear-left view looking up
```

`same` mode remains the negative/control lane where the grid contact sheet uses
the same base prompt.

## Snapshot

The `snapshot/` directory contains:

- `src/splat3d/cameras.ts`
- `src/splat3d_page.ts`
- `src/splat3d.html`
- `tools/splat3d/grid_quality.ts`
- `docs/SPLAT3D_PERF_NOTES.md`
- `experiments/clip_forks/README.md`

Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v19_grid_literal_prompt
```

## First Gate

Build:

```bash
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Fixed-step parser/smoke:

```bash
RUN_STEPS=1 TRIALS=1 PROMPT="a photo of a cat" CONFIGS='grid80literal=9:3:grid9:directgrid:literal' OUT_DIR=/tmp/nffa_grid_literal_smoke bun tools/splat3d/grid_quality.ts
```

## Quality Gate To Run Next

Use fixed-wall-clock repeats so GPU contention and compile cost do not dominate
one run:

```bash
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat" CONFIGS='base3=3:3,grid80=9:3:grid9:directgrid,grid80literal=9:3:grid9:directgrid:literal' OUT_DIR=/tmp/nffa_grid_literal_cat bun tools/splat3d/grid_quality.ts
```

Promotion requires `grid80literal` to improve median full per-view teacher
score versus `grid80` without materially reducing step rate, and to close the
gap to `base3` in screenshots.
