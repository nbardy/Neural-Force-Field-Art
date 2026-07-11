# Fixed-Budget Splat Adaptation Worker

Date: 2026-07-10

## Added API

`src/splat3d/adaptive.ts` exports:

```ts
planFixedBudgetSplatAdaptation(
  params: Float32Array,
  rawGradients: Float32Array,
  options?: SplatAdaptationOptions
): SplatAdaptationPlan
```

The function is CPU-only and pure with respect to its inputs. Both arrays use
the current isotropic `8G` SoA layout: positions `3G`, log radius `G`, raw RGB
`3G`, and raw opacity `G`. It returns a new parameter array at the same length,
sorted unique changed splat indices, per-relocation records, and aggregate
diagnostics.

## Selection And Relocation

- Parent need is the Euclidean norm of componentwise-absolute raw position
  gradients, multiplied by optional per-splat coverage telemetry.
- Destinations are selected by low sigmoid opacity, then low need, then index.
- Parents are selected by high need, then gradient magnitude, opacity, and
  index. Dead candidates cannot also become parents.
- Each parent is used once. A selected destination copies the parent's color,
  while parent and child move symmetrically around the old parent center along
  a seeded deterministic unit direction.
- Both split radii default to `2^(-1/3)` of the parent radius. Opacity is solved
  from optical-density-weighted projected area,
  `-log(1 - opacity) * radius^2`, including the replaced destination's old
  contribution. This approximately preserves composited coverage while
  changing both radius and opacity.
- Radius and sigmoid opacity are clamped to configured finite bounds. Defaults
  use the current raster radius bounds (`0.01` to `0.45`) and maximum alpha
  (`0.99`).

## Integration Notes

There is deliberately no GPU integration in this worker. A caller would need
to read params and raw gradients, run the planner at a chosen cadence, upload
the returned params, and reset Adam moments for every `changedIndices` entry.
The reset matters for both relocated splats and parents whose position,
radius, or opacity changed.

The current raw gradient buffer contains already-accumulated signed gradients.
Taking componentwise absolute values before the norm prevents cross-axis sign
effects but cannot reconstruct AbsGS per-pixel absolute accumulation after
pixel/view contributions have canceled. Supplying per-splat coverage weights
adds the Pixel-GS-style size/visibility signal available to this CPU planner.

## Gate

```bash
bun tools/splat3d/adaptive_test.ts
```

The gate covers fixed parameter count, finite output, radius/opacity bounds,
input immutability, exact reproducibility, coverage-weighted parent ranking,
approximate coverage conservation, and movement of dead splats to the
neighborhood of high-gradient parents.
