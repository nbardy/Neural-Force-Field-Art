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

The only global decoder state is 27 floats (a 3x8 matrix and 3 biases). The
production backward runs a 64-lane workgroup with four pixels assigned to each
lane. A lane locally accumulates its four-pixel contributions, `subgroupAdd`
reduces those values in hardware, and each subgroup leader emits a fixed-point
partial into the device accumulator. The same schedule reduces splat gradients.

This avoids the old two sources of serialization: per-hit device atomics and
per-splat workgroup barriers. It does not use a scalar fallback. The live page
requires WebGPU `subgroups`; Chrome on the target Apple M4 reports that feature
and compiles/runs the path.

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
* RGB gradient parity against the normal raster: `1.373e-4` max absolute raw
  gradient error; and
* local-theta finite difference: `2.73%` relative error at a point away from
  alpha-threshold discontinuities.

The gradient difference is bounded fixed-point reduction-order noise. The RGB
baseline rounds each pixel contribution; the subgroup implementation sums a
small f32 partial and rounds once per subgroup. The test permits `2e-4` but
keeps the finite-difference check, so this is not an unchecked approximation.

Earlier one-off quality/timing values in this document predated two harness
corrections: the reported optimizer loop included three mutating warm-up steps,
and RGB timing divided by those warm-up steps after timing began. Treat them as
historical debugging evidence, not a Feature8/RGB quality or speed comparison.
The corrected gates use disposable warm-up optimizers, true reported step
counts, and serial medians for timing; GPU contention and browser adapters vary.

## Subgroup Performance Snapshot

`tools/splat/feature_stage_profile.ts` uses GPU timestamps around the complete
Feature8 optimizer step. On the Apple M4 with `G=12000`, `density4-s5`, and a
stable nine-sample run, it reported:

| Stage | Median GPU time |
| --- | ---: |
| Feature raster forward | `1.38 ms` |
| CLIP train pass | `16.78 ms` |
| Feature raster backward | `4.92 ms` |
| Adam | `0.07 ms` |
| Total | `23.13 ms` |

The previous barrier-heavy subgroup reducer measured about `30 ms` raster
backward in a comparable profiling session; an earlier run was near `49 ms`.
Absolute GPU timings move with browser/GPU contention, so the durable result is
the topology change: tile-coarsening plus subgroup reductions made raster
backward smaller than CLIP rather than the dominant stage. Testing the two
nearest occupancy points kept `64 lanes x 4 pixels` as the one production
shader: `128 x 2` and `32 x 8` measured about `7.27 ms` and `7.21 ms` backward
respectively in their nine-sample runs.

The next high-value speed lane is CLIP, not a hidden decoder MLP. Raster-side
work should now focus on candidate-count ablations or a compact basis lookup,
each with an equal-step convergence gate.

## Active-Set Count Check

The original `2048` default came from the discarded Feature32 prototype, whose
dense feature image and gradient were substantially more expensive. The compact
Feature8 rewrite removed that hard resource reason, but a July 16 check found a
separate initializer problem for the current painter dynamics:

| Initialization | Mean / max splats binned per tile | Mean / max tile max-prefix | Tile overflow |
| --- | ---: | ---: | ---: |
| `G=2048` | `143.3 / 196` | `113.9 / 169` | `0` |
| `G=12000` | `797.7 / 969` | `122.5 / 203` | `0` |

The bins do not overflow. Alpha compositing in deterministic splat-ID order
reaches near-zero transmittance after a similar prefix in both cases. The
recorded prefix is each tile's maximum over its pixels, so actual per-pixel
visibility is no greater and is usually lower.

This is **shared with RGB**, not a Feature8-only gradient problem. In a true
20-step `G=12000` RGB run, the telemetry was `796.9` binned and `123.3` prefix
per tile; Feature8 under the same legacy init was `796.7` and `123.0`. At
Feature8 initialization, the RGB/geometry backward has an explicit parity gate
against the standard raster. Extra latent channels are visibility-gated by the
same `T * alpha` weight.

The relevant fix is count-aware density initialization. For a uniform cloud,
the approximate optical depth is:

```text
tau ~= G * alpha * 2*pi*scale^2 / (H*W)
```

For `G=12000`, the legacy `scale=9`, `alpha~=0.60` setting is excessively
opaque. An initial Feature8 ablation using `scale=5` and target `tau=4`
(`alpha~=0.139`, `opacityRaw~= -1.82`) kept all roughly `303` binned splats in
the tile replay. Across three seed-1/2/3 true-20-step runs, its final Feature8
cosines were `0.35005`, `0.32782`, and `0.35681`, versus legacy-init results
`0.33789`, `0.32607`, and `0.30704`. This is promising but not yet enough to
change the live default: one seed regressed after 20 steps, so the next gate is
feature/decoder LR decay or freeze-after-warmup, followed by a 20/60-step
multi-seed comparison.

A later same-seed 50-step check compared two count-aware footprints at `G=12000`:

| Initialization | Final cosine | Mean tile candidates |
| --- | ---: | ---: |
| `density4-s5` | `0.38055` | `315.0` |
| `density4-s3` | `0.38329` | `208.5` |

`s=3` is a credible fast candidate because it preserves the optical-depth
target while reducing active candidates. It is not the live default yet: this
is one seed and still needs the documented multi-seed, equal-step gate.
