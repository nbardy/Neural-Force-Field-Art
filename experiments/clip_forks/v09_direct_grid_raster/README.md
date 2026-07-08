# v09 Direct Grid Raster

## Hypothesis

`grid9_close2` lowered CLIP-visible per-view resolution by packing nine camera
views into one `256x256` CLIP lane, but it still paid full raster cost for the
grid lane: render each view at `256x256`, then downsample to an `80x80` cell.

The bigger speed lever is to render each grid view directly at `80x80`, copy it
into the contact-sheet cell, scatter the CLIP gradient back to the `80x80`
scratch image, and run raster backward at `80x80`.

This keeps CLIP at its trained `256x256` input size while reducing grid-lane
raster work.

Gate:

```bash
GRID_DIRECT_RASTER=1
```

The browser UI exposes the same gate as `grid raster 80` when `3x3 grid + 2` is
selected.

## Implementation

- `Raster3DEngine` can now be created with shared `params` and `gradRaw`
  buffers, without owning their destruction.
- `Grid9Close2ClipLayout` can create an internal `80x80` raster engine with
  cameras prepared at `side=80`.
- The internal raster shares the main splat params and raw gradient buffer, so
  grid gradients still feed the main Adam step.
- The grid lane remains a `256x256` CLIP image with nine `80x80` cells and black
  gutters.
- Close-up lanes still use the normal full-resolution `256x256` raster path.

## Snapshot

`snapshot/` contains the grid, optimizer, raster, page, and benchmark files
copied from `HEAD` before the fork was edited.

Diff from snapshot:

```bash
node experiments/clip_forks/diff_fork.mjs v09_direct_grid_raster
```

## Correctness Gate

```bash
bun tools/splat3d/grid9_close2_test.ts
GRID_DIRECT_RASTER=1 bun tools/splat3d/grid9_close2_test.ts
```

Both modes must populate all nine cells and leave gutters black.

## Timing Gate

```bash
TRIALS=3 CONFIGS=grid=9:3:grid9,grid80=9:3:grid9:directgrid RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
TRIALS=3 CONFIGS=base=3:3,baseDw4=3:3:dw4,grid80=9:3:grid9:directgrid,grid80dw4=9:3:grid9:directgrid:dw4 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

## Result

Correctness passed:

```text
bun tools/splat3d/grid9_close2_test.ts
GRID_DIRECT_RASTER=1 bun tools/splat3d/grid9_close2_test.ts
PASS grid9_close2 contact sheet
```

Direct grid raster gives a large grid-mode speedup:

| Variant | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `grid9_close2` | `87.93 ms` | `89.85 ms` | `41.07 ms` | `46.01 ms` |
| `grid9_close2 + directgrid` | `59.03 ms` | `60.48 ms` | `38.72 ms` | `18.76 ms` |

Stacking the earlier `depthwise4` CLIP fork gets all-view grid supervision close
to the ordinary `3/9` per-view baseline:

| Variant | Normal Step Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `3/9 per-view` | `52.79 ms` | `56.41 ms` | `41.17 ms` | `13.00 ms` |
| `3/9 per-view + dw4` | `49.69 ms` | `54.96 ms` | `39.15 ms` | `12.96 ms` |
| `grid80+2 all-view` | `59.14 ms` | `60.64 ms` | `39.08 ms` | `19.20 ms` |
| `grid80+2 all-view + dw4` | `56.08 ms` | `57.62 ms` | `36.42 ms` | `18.48 ms` |

Decision: keep gated for quality testing. The speed result is strong enough to
make `grid raster 80` the preferred all-view contact-sheet ablation.
