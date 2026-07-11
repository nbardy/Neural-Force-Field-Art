# Feature32 Reference Raster

## Scope

This work adds a standalone differentiable 32-channel splat raster. It does not
modify or replace the current RGB tile raster. The implementation is a bounded
correctness oracle for a later tiled feature raster.

## Layout And Math

- Geometry is splat-major `vec4(meanX, meanY, logRadius, opacityLogit)` in pixel
  coordinates.
- Features are splat-major `[splat][32]`, packed as eight `vec4f` values.
- `sortedIds` supplies the front-to-back compositing order.
- Raster output and its upstream gradient are channel-planar `[32][H*W]` f32,
  exactly matching `Feature32Colorizer.features` and `featureGrad`.
- A fixed 32D background is composited through final transmittance.
- Opacity is `0.99 * sigmoid(opacityLogit)`. Radius is
  `max(exp(logRadius), minRadius)`.
- Backward returns gradients for all 32D splat features and packed geometry.

## Integration Contract

Create one `Feature32ReferenceRaster` and one `Feature32Colorizer` with the same
width and height. Bind the raster `imageFeatures` buffer as colorizer
`features`, and bind raster `imageFeatureGrad` as colorizer `featureGrad`.

Command order:

1. `featureRaster.recordForward(...)`
2. `colorizer.recordForward(...)`
3. CLIP forward/backward writes `colorizer.rgbGrad`
4. `colorizer.recordBackward(...)`
5. `featureRaster.recordBackward(...)`

The reference backward is one deterministic GPU invocation. That avoids float
atomics and gives stable finite-difference comparisons, but its work is
`O(H * W * splats * 32)`. `maxCompositePairs` defaults to 1,048,576 and rejects
larger configurations. Production integration needs the existing tile/bin
structure or another conflict-free reduction strategy.

## Verification

Run:

```bash
bun tools/splat3d/feature_raster_test.ts
```

The gate checks forward parity and exhaustive central finite differences for
all per-splat feature, opacity, 2D mean, and log-radius inputs on WebGPU.
