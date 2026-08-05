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

At boot, all latent values and local coefficients are zero. The RGB skip is a
fixed identity route, not a trainable global colour shortcut: decoder bias and
RGB-skip columns remain zero, while deterministic latent decoder columns are
nonzero. That preserves exact RGB output while giving feature values a gradient
on the first optimizer step.

## Differentiability And Resources

- The local frame uses true rotation and anisotropic scale, not a conic-axis
  approximation.
- The backward propagates local appearance through center, log scale, and
  rotation, as well as through alpha compositing.
- No dense feature image or dense feature-gradient image is materialized.
- The tile backward consumes CLIP's RGB gradient directly. It uses hardware
  subgroup reductions and a 64-lane, four-pixels-per-lane tile schedule, so
  it has no per-splat workgroup barriers or workgroup gradient atomics.
- This is intentionally a subgroup-required path. The live page requests the
  `subgroups` WebGPU feature and reports a clear failure instead of selecting a
  slower fallback shader.
- Feature mode currently defaults to 2,048 splats; RGB mode remains 12,000.
  This is not an intrinsic Feature8 limit. The legacy initializer has fixed
  `scale=9` and `alpha=sigmoid(0.4)`, so its optical depth grows with both
  splat count and area. A high-count mode needs count-aware alpha/scale
  initialization and a late appearance-learning schedule, rather than simply
  reusing the 2k settings.

Real-Metal checks live in `tools/splat/feature_painter_math_test.ts`:

- production initialization RGB parity;
- RGB gradient parity against the regular rasterizer; and
- a finite-difference local-theta gradient check.

See `docs/FEATURE_PAINTER_FUSED_DECISION.md` for the formula, rejected designs,
the measured Metal/Chrome performance rationale, and the remaining ablations.
See `docs/PIXEL_BUFFER_CONTROL.md` for the direct-pixel CLIP control and why a
high CLIP cosine alone is not a useful artistic-quality gate.

## Exploration Behavior

The default starts with lower optical depth than the legacy 2k setup, so
interior splats receive a useful gradient instead of being hidden behind the
first alpha layer. It retains standard Adam after a lower-beta ablation failed
to improve movement, and uses a 180-step geometry-first learning-rate schedule:
centers, scale, and rotation can migrate early, then appearance settles.

`NUDGE` replaces a sparse fraction of full splat candidates, clears the matching
latent payloads, and clears Adam moments. It no longer interpolates every splat
toward a new random position, which only caused the same composition to wiggle.
The optional `explore` control does two sparse refreshes at steps 96 and 260.
It is intentionally off by default while the 1k-step convergence ablation is
running. A short 300-step measurement is not sufficient to choose between fast
local improvement and a restart schedule that may recover later.

Use the trajectory harness rather than the 20-step smoke test for convergence
decisions:

```bash
MODE=default STEPS=1000 OUT_DIR=/tmp/feature-default bun tools/splat/feature_dynamics.ts
MODE=default EXPLORE=1 STEPS=1000 OUT_DIR=/tmp/feature-explore bun tools/splat/feature_dynamics.ts
```
