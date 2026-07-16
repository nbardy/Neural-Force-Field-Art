# Fused Feature Painter Decision

## Decision

Replace the first experimental `F=32` feature painter with a specialized,
fused `Feature8` renderer:

```
base(x) = sum_i T_i alpha_i rgb_i + T_final background
latent(x) = sum_i T_i alpha_i (z_i + u_x Axi + u_y Ayi)
rgb(x) = sigmoid(logit(clamp(base(x))) + 0.1 * (b + W [base(x), latent(x)]))
```

`rgb_i` is the ordinary RGB value of splat `i`, obtained by applying sigmoid to
the existing raw `colorRaw` parameter before alpha compositing.  It is not a
feature logit.

The decoder bias and RGB-skip columns begin at zero; its latent columns use a
small deterministic seed while all latent features begin at zero. Therefore
the residual output is zero and the feature path initially emits the same RGB
image as the regular rasterizer at the same splat count and seed. The seeded
latent columns give the feature field a gradient on step one. The learned
decoder is a residual, not a replacement color source.

## Why This Is The Fast Path

The renderer has eight composited values per pixel: RGB base plus five latent
channels.  It does not materialize an `F x H x W` feature image or its gradient.
The CLIP RGB gradient is consumed directly in the tile backward shader, which:

1. recomputes the compact feature field in registers;
2. evaluates the linear decoder VJP in registers;
3. accumulates splat and decoder gradients; and
4. writes no dense feature-gradient image.

The only global decoder state is 27 floats (a 3x8 matrix and 3 biases). Its
per-pixel accumulation stays in workgroup atomics and emits only 27 global
atomic updates per tile, instead of 27 global decoder atomics per pixel. A
shared-array reduction was also measured, but regressed on the target Metal
path; it remains documented as a rejected optimization rather than a default.

## Local Coordinates

For a splat with center `m`, angle `theta`, and scale `(sx, sy)`, use its real
rotated local frame:

```
dx = x - mx; dy = y - my
ux = clamp((cos(theta) dx + sin(theta) dy) / sx, -3, 3)
uy = clamp((-sin(theta) dx + cos(theta) dy) / sy, -3, 3)
```

The backward includes the local-coordinate derivative into center, log scale,
and rotation.  This is what makes `Axi` and `Ayi` genuine local appearance
parameters rather than a stopped-gradient decoration.

## Critical Distinction: Standard Feature Splatting vs Neural Splats

A final pixel can contain many splats, each with a different local coordinate.
Consequently, an MLP cannot correctly consume "the local coordinate of the
pixel" after the splats have already been composited.  There are two different
models:

* **Standard feature splatting** composites features, then runs one decoder per
  pixel.  This implementation uses that model and lets each splat locally
  modulate the features before the composite.
* **Neural splats** run an MLP for every contributing splat-pixel pair on
  `[z_i, ux_i, uy_i]`, then composite the colors.  That is mathematically
  valid, but its MLP and backward execute once per visible hit, so it is not
  the first browser fast path.

## DynaWorld Lessons Used

The local DynaWorld measurements are directionally clear:

* Direct `F32` feature gradients are bandwidth/atomic bound.
* Compact `K=4/8` feature paths were much faster than direct `F32` feature
  splatting at tested sizes.
* Fusing a full scalar colorizer into tile backward was not universally faster;
  the 512px larger cases regressed.
* LayerNorm/RMSNorm and a hidden MLP add a meaningful cost for a tiny decoder.

So this path keeps `F=8`, a no-norm linear residual decoder, and fuses only the
memory-bound image-gradient handoff.  A hidden MLP, PCA visualization replay,
and per-hit neural color are intentionally separate opt-in experiments.

## Rejected Initialization

Do not copy RGB **logits** into feature channels `0:3` and decode them after
the composite.  In general:

```
sigmoid(sum_i w_i logit(rgb_i)) != sum_i w_i rgb_i
```

The parity-preserving RGB skip is the composited RGB image itself, followed by
`logit(base)` only inside the final residual output transform.

## Initial State

* Existing geometry, raw RGB logits, and opacity use the regular splat init.
* Five base latent values start at zero.
* Local-x and local-y coefficients start at zero.
* Decoder bias and RGB-skip columns start at zero; latent decoder columns use
  deterministic `0.25 * N(0, 1)` values.

This makes the feature system visually inert at step zero while routing a
gradient into latent channels immediately.

## Verification Snapshot

The real-Metal `tools/splat/feature_painter_math_test.ts` gate currently gives:

* production-init RGB parity: `1.192e-7` max absolute pixel error;
* RGB gradient parity against the normal raster: `6.054e-6` max absolute raw
  gradient error; and
* local-theta finite difference: `2.8%` relative error at a point away from
  alpha-threshold discontinuities.

In one serial local run at `G=2048` and 60 cat-prompt steps, Feature8 moved
cosine from `0.16917` to `0.35163`, with latent parameter L2 movement `5.97`.
The matched RGB control reached `0.35549`. Feature8 measured `36.8 ms/step`
against RGB's `25.9 ms/step` in that run. Treat those timing values as a local
comparison, not a portable benchmark: GPU contention and browser adapters vary.

The next speed decision should target feature-gradient atomics or a compact
basis lookup, not a hidden decoder MLP. The shared-array decoder-reduction
attempt was measured at about `50.6 ms/step` and was rejected.
