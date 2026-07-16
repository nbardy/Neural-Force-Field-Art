# 2D Feature Painter

The 2D prompt-to-splats page now has an explicit `feature painter` mode beside
the established `RGB splats` baseline. It is an experimental representation,
not the default.

## Contract

Each splat has three 32D banks: a base feature plus local-x and local-y feature
coefficients. At a pixel it emits:

```text
z_i(x) = base_i + u_x(x) * localX_i + u_y(x) * localY_i
```

The feature vectors are source-over alpha composited in the existing tiled
2D raster. A shared, trainable 32-to-RGB residual decoder then emits the image
seen by CLIP. Its RGB skip starts at zero residual, so channels 0-2 provide the
initial RGB-compatible path while the other channels can become useful as the
decoder learns.

The local coordinate is conic-normalized and clamped to `[-3, 3]`. It is
computed before compositing, where an individual splat center is well-defined.
This avoids the invalid design of giving a post-composite pixel an arbitrary
single splat center.

## Deliberate First-Version Limit

The local-appearance derivative into mean, scale, and rotation is stopped.
Geometry still receives the exact alpha-compositing gradient. The base/local
feature banks and decoder train normally. This keeps the initial tiled path
within WebGPU's portable eight-storage-buffer stage limit and avoids adding a
partially verified geometry Jacobian to the live optimizer.

The next representation-level improvement is the full local-frame Jacobian,
including rotation, after a finite-difference gradient check. Do not claim
that this version is a full neural-splat appearance model yet.

## Resource And Quality Envelope

- Feature mode defaults to 2,048 splats; RGB remains 12,000 splats.
- Each feature splat has 96 scalar parameters: 32 base, 32 local-x, 32 local-y.
- Feature images and feature gradients are 32 x 256 x 256 fp32, about 8 MB each.
- Metal smoke, fixed cat prompt, 2,048 splats: `0.17411 -> 0.31624` cosine in
  20 optimizer steps. This proves gradient flow, not final visual superiority.

## Exploration Behavior

`NUDGE` now clears Adam moment state after reseeding parameters. Retaining old
moments made a partial reseed drift toward the previous local minimum. In
feature mode it preserves the learned residual decoder and only clears the
splat/feature Adam state.
