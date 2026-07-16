# 2D Feature Painter

The 2D prompt-to-splats page has an experimental `feature painter` mode beside
the RGB baseline. It is a compact Feature8 renderer, not the earlier 32-channel
prototype.

## Representation

Each splat retains its normal RGB color and alpha. It also owns five latent
values plus five local-x and five local-y coefficients:

```text
latent_i(x) = z_i + ux_i(x) * Ax_i + uy_i(x) * Ay_i
```

The raster alpha-composites RGB and the five latent values. A small 3x8 linear
residual decoder then maps `[composited RGB, composited latent]` to RGB. It is
not a per-hit neural MLP: each splat's local coordinates are applied before
compositing, where that splat's local frame is defined.

## RGB Skip And Initialization

The RGB skip is the ordinary composited RGB image. It is transformed through
`logit(baseRGB)` only immediately before the residual output sigmoid. Thus a
zero residual reproduces the standard rasterizer; raw per-splat RGB logits are
never incorrectly composited as feature values.

At boot, all latent values and local coefficients are zero. Decoder bias and
RGB-skip columns are zero; deterministic latent decoder columns are nonzero.
That preserves exact RGB output while giving feature values a gradient on the
first optimizer step.

## Differentiability And Resources

- The local frame uses true rotation and anisotropic scale, not a conic-axis
  approximation.
- The backward propagates local appearance through center, log scale, and
  rotation, as well as through alpha compositing.
- No dense feature image or dense feature-gradient image is materialized.
- The tile backward consumes CLIP's RGB gradient directly and keeps only a
  27-value workgroup-local decoder gradient reduction.
- Feature mode defaults to 2,048 splats; RGB mode remains 12,000.

Real-Metal checks live in `tools/splat/feature_painter_math_test.ts`:

- production initialization RGB parity;
- RGB gradient parity against the regular rasterizer; and
- a finite-difference local-theta gradient check.

See `docs/FEATURE_PAINTER_FUSED_DECISION.md` for the formula, rejected designs,
and DynaWorld-derived performance rationale.

## Exploration Behavior

`NUDGE` preserves the representation but clears Adam moments after reseeding
geometry. Retaining old moments made a partial reseed drift toward its prior
local basin instead of exploring.
