# Standalone anisotropic conic raster fork

Date: 2026-07-10

## Scope

This lane adds a standalone differentiable anisotropic raster without changing the live isotropic optimizer. It reuses the validated projection/Jacobian code in `src/splat3d_aniso/projection_wgsl.ts` and mirrors the core `Raster3DEngine` command order.

New implementation files:

- `src/splat3d_aniso/raster_wgsl.ts`
- `src/splat3d_aniso/raster_engine.ts`
- `tools/splat3d/aniso_raster_test.ts`

## Data layout

Parameters use the existing 14G SoA contract:

1. position: `3G`
2. log scale: `3G`
3. raw quaternion `[x,y,z,w]`: `4G`
4. raw RGB: `3G`
5. raw opacity: `G`

The 12G derived buffer stores screen mean, conic `[a,b,c]`, camera depth, sigmoid RGB/opacity, and the covariance diagonal used for conservative ellipse bounds.

## Raster path

1. Project each 3D Gaussian to a 2D covariance and inverse-covariance conic.
2. Bin with the exact axis-aligned ellipse extent `sqrt(tau * covariance diagonal)` at the alpha cutoff.
3. Sort each tile front to back with the same bitonic workgroup algorithm as the isotropic raster.
4. Composite with `power = -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)`.
5. Reverse alpha compositing to accumulate mean, conic, RGB, and opacity gradients.
6. Chain mean/conic gradients through `anisoProjectBackward` to position, three log scales, and quaternion.

The engine supports multiple camera-specific prep/chain pipelines, raw-gradient accumulation across views, tile telemetry, and five-group Adam updates. It intentionally does not yet implement batch view lanes, dynamic backgrounds, convergence priors, private retained scratch states, or grid integration.

## Verification

`bun tools/splat3d/aniso_raster_test.ts` passes on Apple Metal.

- isotropic forward max error: `8.941e-8`
- isotropic position-gradient max error: `1.986e-5`
- isotropic summed-scale-gradient max error: `6.029e-5`
- isotropic color/opacity max error: below `5e-10`
- isotropic quaternion-gradient max magnitude: `2.163e-10`
- anisotropic finite differences: position, log scale, quaternion, color, and opacity all below `3.1%` relative error; four of five are below `1%`
- tile overflow: zero
- Adam updates all sampled parameter groups

The position finite difference uses a smaller world-space epsilon (`2e-4`) because the raster has deliberate alpha-cutoff and tile-membership discontinuities. At `1e-3`, the sample crossed a support boundary and was not a valid local derivative comparison.

## Integration path

Keep this fork separate until its runtime and convergence are measured. The minimal live integration is a renderer selector that constructs `AnisotropicRaster3DEngine`, initializes all three log scales from the old scalar radius, and initializes quaternions to identity. Batch/grid support should be ported only after the single-view fork has quality evidence, because anisotropic derived and gradient storage is larger and its ellipse bounds can increase tile pressure.

At this checkpoint the fork was standalone. The follow-up below records its
later opt-in live integration; rollback now means selecting isotropic mode, not
deleting files that the page imports.

## Live integration follow-up: break spherical symmetry

The first live version followed the conservative integration plan too literally:
all three axes copied the scalar radius and every quaternion was identity. That
made every primitive a sphere at initialization. Rotation is unidentifiable for
a sphere, so quaternion gradients also begin at zero; the UI could truthfully
say "anisotropic" while showing circles for a long time.

The live anisotropic initializer now applies a shuffled `[-a, 0, +a]` offset to
the three log scales (`a` is centered around `0.45`) and samples a uniform unit
quaternion. The log offsets sum to zero, preserving each old splat's geometric
mean radius/volume scale while producing a roughly `2:1` initial 3D axis ratio.
The Metal CLIP/Adam gate reports:

- mean 3D axis ratio after one step: `2.54`
- initialized projected ratio across all nine cameras: `1.74-1.76` mean,
  `2.31-2.34` p90
- non-identity orientations: `256/256`
- embedding delta: `1.744e-2`
- tile occupancy: `60/256`, zero overflow

The page telemetry now reports both 3D axis ratio and projected screen ratio.
This matters because an elongated 3D Gaussian can still project nearly circular
when its long axis points toward a camera.

Random per-view training now uses weighted sampling without replacement. Front,
left/right side, and directly-overhead cameras carry most of the probability;
rear and oblique cameras remain present at lower probability. Epoch sampling is
unchanged and still gives every camera equal coverage. Canonical CLIP captions
are prefix-shaped (`a front-on view of ...`, `a right-side view of ...`,
`a directly overhead view of ...`) instead of appending camera jargon after the
subject.

## Production default integration

The anisotropic path is no longer a single-CLIP demonstration. It now includes:

- true batch-major CLIP x3 with exact single-pass parameter parity;
- replayed conic/tile state for each batch lane before raster backward;
- split phase profiling for initial raster, CLIP, replay, raster backward,
  regularization, Adam, and display;
- opacity-weighted centroid/bounds regularization;
- geometric-mean anti-tiny and optional scale-band gradients;
- staged geometry/appearance rates as an available control;
- fixed-budget anisotropic relocation with selective Adam reset;
- a default `full 3D` UI preset using camera-prefix text, centered black
  framing, zoom-out, and biased random 3-of-9 views.

The default deliberately leaves transmittance/ray/mip/background experiments
off. The new features are integrated and selectable, but controlled cat gates
did not justify enabling those loss terms. The enabled geometry safeguards did:
at 1,024 splats and 240 equal steps they improved mean cosine by `0.00754`,
worst-view cosine by `0.04074`, and reduced spread by `11.3%`.

One adaptation failure was found and fixed during this gate. Per-axis clamping
cannot preserve anisotropy when a trained axis ratio exceeds the configured
scalar range. Adaptation now clamps only geometric-mean scale through one shared
log shift, preserving every pairwise axis ratio and avoiding a step-200 crash.
