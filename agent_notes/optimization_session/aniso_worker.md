# Standalone anisotropic 3D Gaussian math fork

Date: 2026-07-10

## Scope

The fork lives entirely in `src/splat3d_aniso/`. It does not alter or import the
current rasterizer. `tools/splat3d/aniso_projection_test.ts` is its acceptance
gate.

Raw parameters retain the current optimizer's SoA packing style:

```text
position[3G] | logScale[3G] | quaternion[4G] | color[3G] | opacity[G]
```

The total parameter count is `14 * G`. Quaternions use `[x, y, z, w]` order and
are normalized in forward projection. Color and opacity are layout values only;
an integration can preserve the current sigmoid/logit interpretation.

## Projection math

For world covariance

```text
Sigma_world = R(normalize(q)) diag(exp(logScale)^2) R^T
Sigma_camera = K Sigma_world K^T
```

the module projects with either of two Jacobians:

```text
legacy-affine:
J = [[f/z,    0, 0],
     [  0, -f/z, 0]]

perspective-jacobian:
J = [[f/z,    0, -f*x/z^2],
     [  0, -f/z,  f*y/z^2]]
```

`legacy-affine` is the default because equal scales recover the current
`radiusPx = focalPx * radiusWorld / z` footprint exactly, including off axis,
while the current 0.25 px hard floor is inactive.
`perspective-jacobian` is the standard local pinhole linearization and lets
camera-forward variance affect off-axis footprints.

The 2D covariance and conic use symmetric three-float storage:

```text
Sigma_2D = [s00, s01, s11]
conic    = inverse(Sigma_2D) = [a, b, c]
power    = -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
```

An optional isotropic variance can be added to the 2D covariance for a smooth
screen-space low-pass floor. The default is zero for current-renderer parity.

The analytic backward accepts upstream gradients on projected mean and conic.
It chains through inverse covariance, covariance projection, scale exponentials,
the rotation matrix, quaternion normalization, the projection Jacobian, and the
world-to-camera transform. It returns raw gradients for position, all three log
scales, and all four quaternion components.

## Integration requirements

1. Change the raster parameter count from `8 * G` to `14 * G` and use
   `anisotropicParamSegments3D(G)` for offsets. Adam remains elementwise but its
   parameter count and any group-specific learning rates must cover the new
   scale and quaternion segments.
2. Initialize old isotropic scenes by copying `logRadius` into all three
   `logScale` lanes and setting every quaternion to `[0, 0, 0, 1]`.
3. Embed `anisotropicProjectionWGSL({ mode: "legacy-affine" })` in prep and chain
   shaders. Construct `AnisoCamera` from the existing eye/right/cameraUp/forward,
   focal pixel value, image center, and near plane.
4. Expand derived and accumulated-gradient records by two floats so the scalar
   `invR2` becomes conic `[a,b,c]`. Camera position or the raw inputs must remain
   available to `anisoProjectBackward`.
5. Change raster power to the full quadratic form. Pixel backward must accumulate
   conic gradients proportional to `-0.5 * [dx^2, 2*dx*dy, dy^2]` and mean
   gradients through `C * [dx,dy]`.
6. Change tile bounds from one radius to ellipse extents. At threshold `tau`, the
   exact axis-aligned bounds are `sqrt(tau * covariance.s00)` in x and
   `sqrt(tau * covariance.s11)` in y. Store covariance in derived data or recover
   it from the conic determinant.
7. Update scale regularizers to operate on three axes. Add a quaternion policy:
   normalization already removes radial updates, but resetting near-zero raw
   quaternions to identity avoids the epsilon branch becoming a dead rotation.
8. Keep `legacy-affine` for the first end-to-end parity gate. Treat
   `perspective-jacobian` as a separate quality/stability ablation because it
   intentionally changes off-axis isotropic footprints.

## Verification

Run:

```bash
bun tools/splat3d/aniso_projection_test.ts
```

The gate checks the 14-float layout, off-axis isotropic parity, quaternion
normalization, finite differences for both projection modes, WGSL compilation,
and GPU-vs-CPU forward/backward values when `bun-webgpu` exposes an adapter.
