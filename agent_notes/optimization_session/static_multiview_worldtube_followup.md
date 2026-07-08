# Static Multi-View Raster Batching Follow-Up

Date: 2026-07-08

## Question

Can the 3D CLIP splat optimizer raster many static camera views cheaply by
borrowing the world-tube / STAR UVT idea: rays as fibers, camera as gauge math,
or one object covering several views?

## Files Re-Checked

- `/Users/nicholasbardy/git/gsplats_browser/research_experiments/camera_movement_aware_worldtubes.md`
- `/Users/nicholasbardy/git/gsplats_browser/TODO/star_variable_camera_projection_task.md`
- `/Users/nicholasbardy/git/gsplats_browser/agent_notes/STAR_VARIABLE_CAMERA_HANDOFF_2026_05_12.md`
- `/Users/nicholasbardy/git/gsplats_browser/agent_notes/STAR_UVT_PRT_B1_METAL_FORWARD_2026_05_13.md`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/README.md`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/world_tube.py`
- `src/splat3d/raster_wgsl.ts`
- `src/splat3d/raster.ts`
- `src/splat3d/optimize.ts`

## Read

STAR UVT is not just "batch frames." It changes the primitive contract. A world
tube stores a moving center plus time precision, and projection emits:

```text
ma    = (u, v, t)
q_uvt = (uu, uv, ut, vv, vt, tt)
```

The `ut` and `vt` terms encode screen velocity, so one primitive has native
support over an ordered time interval. The PRT continuation makes the same point
for moving cameras: when camera motion bends the projection, the solution is a
projective rational camera-time tube or a piecewise camera-time segment, not a
loop over frames disguised as STAR.

Our 9 CLIP views are different. They are a small discrete camera set around one
static object, not neighboring samples of one ordered temporal path. A camera
index can be used as a batch lane, but it is not a well-behaved `t` coordinate:
top, front, side, back, high, and low views have different projection maps,
visibility, depth order, and tile support.

## Current Browser Raster Shape

The current 3D renderer bakes camera constants into the prep and chain shaders:

- `prepShader3D(cfg, cam)` projects world splats into one camera's screen gauge.
- `chainAddShader3D(cfg, cam)` maps screen-space derived gradients back to world
  splat params for that same camera.
- `emitShader3D`, `forwardShader3D`, and `backwardShader3D` operate on the
  already-projected `derived` buffer.

`Raster3DEngine.recordForward()` therefore does one view at a time:

```text
prep(view) -> clear bins -> emit -> forward
```

The optimizer's batch path now gives CLIP lanes private raster forward state and
writes rendered views into batch CLIP input lanes. That removed the old replayed
forward for full CLIP batches, but it still records separate raster work for
each camera view.

## Exact Fast Path Without A New Splat Object

The practical exact optimization is a view-lane raster kernel:

```text
cameraBuffer[view]
activeViews[lane]
derived[lane, splat]
tileCounts[lane, tile]
binnedIds[lane, tile, slot]
tileStop[lane, tile]
image[lane, channel, pixel]
```

Then dispatch with `workgroup_id.z = lane`:

```text
prep_batched:     [ceil(G), lanes]
clear_bins:       [ceil(numTiles), lanes]
emit_batched:     [ceil(G), lanes]
forward_batched:  [numTiles, lanes]
backward_batched: [numTiles, lanes]
```

This keeps one normal 3D splat object. It should reduce JS encode overhead,
pipeline switching, bind churn, and some repeated parameter/camera setup. It
does not collapse 9 views into 1 view worth of shader work. Each camera still
needs its own projection, tile list, depth sort, alpha compositing, and backward
transmittance walk.

Backward needs one extra design choice. `accGrad` is currently shared and
cleared per view before chain-add. A true batched backward must either:

- make `accGrad[lane, splat, derivedSlot]` lane-strided, then chain each lane
  into `gradRaw`; or
- replace chain-add with a fixed-point/atomic raw-gradient path.

The first option is safer and closer to current correctness. The second may be
faster later but is a larger numerical and parity risk.

## Why Static Cameras Do Not Give A Free Shared Raster

Static camera poses help scheduling because camera data is constant. They do
not make the projected splats constant because the optimized world params change
every Adam step. Per step, each view still gets different:

- screen center and radius;
- near/far rejection;
- tile footprint;
- depth ordering;
- alpha transmittance chain;
- CLIP image gradient.

Precomputing camera basis is useful. Reusing tile lists across steps is unsafe
unless treated as a separate stale-binning ablation, because splats move and
change radius/opacity during optimization.

## What Would Need A Different Object

A STAR-style sublinear static multi-camera renderer would need a primitive that
lives over camera rays or a camera bundle, not just over world position:

1. Projective camera-bundle splat.
   Store low-order homogeneous projection coefficients over a chosen camera
   path or camera manifold. Good for smooth camera trajectories, awkward for our
   discrete top/front/side/back/high/low set.

2. Atlas-residual splat.
   Choose a small number of screen/depth atlases and store residual projection
   corrections per view. This might fit a fixed 9-camera rig better than one
   global camera gauge, but it is a new render/backward contract.

3. Ray-space / Plucker-fiber primitive.
   Represent density over camera rays directly. This is conceptually aligned
   with "rays as fibers," but exact alpha compositing and view-dependent depth
   order still require per-camera evaluation unless the representation accepts
   approximation.

Those are research projects. They should wait until the exact view-lane raster
path and overflow telemetry are exhausted.

## Recommended Next Ablation

Do not change the splat object yet. Implement camera-buffer plus view-dimension
dispatch as an exact scheduler optimization:

1. Move camera constants from baked WGSL into a compact camera buffer.
2. Add lane-strided raster buffers for `derived`, bins, stops, and `accGrad`.
3. Add batched prep/emit/forward shaders with `workgroup_id.z = lane`.
4. Keep chain-add sequential per lane first for correctness.
5. Only after parity and timing, try batched chain or fixed-point raw-gradient
   atomics.

Expected win: modest but real if raster overhead and pipeline churn matter. It
will not be a 9x raster win; the STAR/PRT-level win requires a different
camera-bundle object.
