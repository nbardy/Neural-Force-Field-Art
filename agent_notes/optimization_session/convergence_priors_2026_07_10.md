# Convergence Priors Session

Date: 2026-07-10

## Live Optimizer Changes

- Replaced the collapsing symmetric per-pixel coverage target with the exact
  clipped Dream Fields global mean-transmittance loss.
- Added point-splat ray distortion in O(N) per ray using total and suffix
  `(weight, weight*depth)` moments in the existing backward scan.
- Added optional normalized-alpha ray entropy as a separate InfoNeRF-inspired
  ablation.
- Added opacity-weighted centroid reduction with a broadcast position gradient.
- Added staged geometry/appearance learning rates without resetting Adam.
- Added a screen-space covariance curriculum (`4.0 -> 0.0625 px^2`) with the
  correct radius derivative through the additive variance.
- Added procedural blurred-noise, checkerboard, and Fourier backgrounds. They
  are generated on GPU; final display renders are regenerated at strength zero.
  Training and display use separate uniform/bind slots so two generation
  dispatches in one command encoder cannot alias to the final host write.
- Added CPU-planned, fixed-budget split/relocate every 200 profiled steps. The
  optimizer resets Adam moments only for changed splat slices.

## Correctness Properties

- All new ray losses are compile-time absent when disabled.
- Mean transmittance is reduced from the exact retained/replayed scratch state
  immediately before each backward pass, so grid cells cannot overwrite one
  another's statistic.
- Separate and batched raster backward remain bit-exact with transmittance and
  distortion enabled.
- Retained and replayed grid modes remain bit-exact with textured backgrounds,
  mip smoothing, ray losses, and a forced adaptive relocation.
- The centroid gate includes a nearly transparent, far-away outlier and verifies
  identical position gradients and zero non-position gradients.

## Sequential Apple Metal Timing

Same 4096-splat, 9-view direct80 grid, CLIP batch 3:

| Mode | Total GPU | Raster fwd+bwd |
| --- | ---: | ---: |
| Base | 47.64 ms | 15.20 ms |
| Ray compact weak | 46.40 ms | 15.07 ms |
| Mip curriculum only | 49.87 ms | 17.18 ms |
| Background curriculum + ray + mip + staged | 48.43 ms | 16.98 ms |

The fresh paired total overhead was 1.7%; raster-only work rose 11.7%. The
normal-step wall sample rose 4.3%. Ray-only no longer runs the global
transmittance reduction, and its remaining cost was below run-to-run CLIP noise.
CLIP remains the largest step component. A transient bun-webgpu shader compile returned a
null-character parse error in one timing process; an immediate sequential rerun
passed and is the number recorded above.

## Representation Fork Status

- `src/splat3d_aniso/`: 14-float SoA layout plus a complete tiled conic RGB
  raster, analytic backward, five-group Adam, finite differences, and isotropic
  forward/back parity.
- `src/splat3d_feature/`: 32D reference compositing raster plus residual RGB
  colorizer and complete backward chain, with exhaustive finite differences.

Anisotropic is wired to CLIP as an experimental live per-view/single-CLIP mode;
it still needs retained/batch/grid IO and convergence regularizers. Feature mode's
reference raster is intentionally serial and still needs 3D camera projection,
tiled binning, and optimizer state. This distinction is important for the blog:
the feature fork is real and tested, but it is not yet a live-page convergence
result.

## Short Real-Prompt Gate

`tools/splat3d/convergence_quality.ts` ran sequential base/prior variants from
identical initialization for 40 steps, `G=1024`, three views per step, real
MobileCLIP text for `a photo of a cat`:

| Prior | Mean cosine delta | Min-view delta |
| --- | ---: | ---: |
| ray compact weak | `+0.00016` | `-0.01648` |
| mip smoothing | `-0.00828` | `-0.03455` |
| staged rates | `-0.01713` | `-0.01363` |
| ray + mip + staged | `-0.01098` | `-0.00259` |

This is one prompt and a short budget, but it is enough to keep every new prior
off in the Dream default. They remain manual ablations for longer/multi-prompt
tests.

## Gates

```bash
npm run build
bun tools/splat3d/transmittance_distortion_test.ts
RAY_REG=1 bun tools/splat3d/raster_batch_forward_test.ts
bun tools/splat3d/grid_retain_parity.ts
bun tools/splat3d/regularizer_test.ts
bun tools/splat3d/background_textures_test.ts
bun tools/splat3d/adaptive_test.ts
bun tools/splat3d/aniso_projection_test.ts
bun tools/splat3d/feature_colorizer_test.ts
```

The headless smoke reached the page but its Chromium build reported no WebGPU
adapter. Real WebGPU execution is covered by the bun-webgpu Apple Metal gates.
