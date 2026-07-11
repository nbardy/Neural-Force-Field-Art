# 3D Feature Splat Fork

Date: 2026-07-10

## Immediate Opacity Finding

The faded step-800 result is explained by the current `alpha weak` math. It is
an unbounded per-splat opacity penalty, not Dream Fields' clipped global mean
transmittance loss. Isolated mean opacity:

- step 0: `0.574443`
- step 100: `0.097784`
- step 500: `0.008264`
- step 800: `0.003901`

Dream-ish now defaults to alpha off. The correct future loss is
`-min(tau, mean(T))` with `tau` annealed from `0.40` to `0.88` over 500 steps.

## DynaWorld Readout

Relevant local sources:

- `dynaworld/src/train/runtime_types.py`: `F>3` requires a downstream colorizer.
- `dynaworld/src/train/colorize.py`: per-pixel 1x1 feature-to-RGB decoder.
- `dynaworld/src/train/dynamic_gauge_foam.py`: RGB skip plus residual decoder.
- `dynaworld/third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff`:
  register-cached pixel feature gradients and SIMD/threadgroup feature-gradient
  reduction before global atomics.

DynaWorld heavily exercises F16/F32, especially F32. There is no local F22
precedent. Use F32 for eight aligned `vec4` groups.

## Proposed Contract

- UI/rebuild toggle: `RGB | feature32`.
- Fork the WGSL renderer; preserve RGB as reference.
- Per splat: 32 raw features, first three initialized from RGB logits.
- Raster output: 32D composited feature image plus alpha.
- Colorizer v1: zero-initialized residual linear 32-to-3 decoder with direct
  RGB skip. Add hidden width only after the linear gate.
- Backward: RGB CLIP gradient -> colorizer -> rendered features -> splats.
- No view conditioning in v1.
- No direct CLIP intermediate-feature objective in v1.

## Risk

Feature-image memory is manageable, but backward color atomics grow from 3 to
32 channels. A naive WGSL loop could make raster backward dominant again. The
fork needs vec4 loads, cached pixel gradients, and local reductions from its
first benchmarkable version.
