# Multi-View Raster Batching And STAR UVT Review

Date: 2026-07-08

## Question

Can the 3D CLIP optimizer render many camera views cheaply by borrowing the
world-tube / STAR UVT idea: rays as fibers, camera as a gauge, one primitive
covering multiple samples instead of one full raster pass per view?

## Files Checked

- `/Users/nicholasbardy/git/gsplats_browser/research_experiments/camera_movement_aware_worldtubes.md`
- `/Users/nicholasbardy/git/gsplats_browser/TODO/star_variable_camera_projection_task.md`
- `/Users/nicholasbardy/git/gsplats_browser/agent_notes/STAR_VARIABLE_CAMERA_HANDOFF_2026_05_12.md`
- `/Users/nicholasbardy/git/gsplats_browser/agent_notes/STAR_UVT_PRT_B1_METAL_FORWARD_2026_05_13.md`
- `/Users/nicholasbardy/git/gsplats_browser/agent_notes/STAR_UVT_PRT_F1A_METAL_ATLAS_BINNING_2026_05_13.md`
- `/Users/nicholasbardy/git/gsplats_browser/agent_notes/STAR_UVT_PRT_D4B_REAL_MULTICAM_RERUN_2026_05_13.md`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/world_tube.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/README.md`
- `src/splat3d/raster_wgsl.ts`
- `src/splat3d/raster.ts`
- `src/splat3d/cameras.ts`

## Read

STAR UVT gets its speed by making time a native raster dimension. One tube stores
an affine center in screen time through `ma = (u, v, t)` plus `q_uvt`, where
the `u-t` and `v-t` precision terms encode projected velocity. The renderer then
bins tile-tube pairs in `(u, v, t)` instead of materializing one splat set per
frame.

That maps cleanly to time because neighboring frames are coherent samples of one
motion path. Our nine CLIP cameras are not equivalent to neighboring times:
top, front, side, and low/high oblique views have different projection maps,
visibility, depth order, and tile footprints. A single physical Gaussian cannot
usually be linear/simple in all camera gauges at once.

The STAR variable-camera notes say the same thing for moving cameras: the naive
per-frame camera loop destroys the win, but fixing it requires a representation
change such as projective rational tubes, camera-gauge tubes, or an atlas. It is
not just a better loop around the existing splat object.

## What Transfers Now

1. Multi-lane raster state.
   The immediate WebGPU win is to make the current view replay unnecessary:
   store `derived`, `tileCounts`, `binnedIds`, and `tileStop` per active batch
   lane, render all selected views into CLIP batch lanes, run batch CLIP once,
   then backward each lane from its saved tile state.

2. Dispatch over a view dimension.
   After per-lane buffers exist, the prep/bin/forward/backward shaders can use
   `workgroup_id.z` as a view lane. Camera data should move from baked WGSL
   constants into a camera storage/uniform buffer. This reduces JS/pass overhead
   and lets one dispatch cover B views, but it still does B view projections and
   B tile lists.

3. Tile-pair backward mindset.
   STAR's useful backward lesson is to keep gradients owned by tile-primitive
   pairs or direct atomics, not by huge per-sample workspaces. Our current
   WebGPU raster backward already follows this spirit with saved per-tile IDs
   and `tileStop`; per-lane state should preserve that.

## What Does Not Transfer Directly

1. UVT as-is.
   UVT's third axis is ordered time. Our camera index axis is a discrete set of
   unrelated gauges. Treating camera index as `t` would make support bounds very
   loose and depth ordering unstable.

2. One cross-view tile sort.
   Depth order is per camera. A splat pair can swap order between front and back
   views. Sorting once across all views would be wrong except for restricted
   scenes.

3. One shared screen footprint.
   A 3D Gaussian's projected radius and center vary substantially by camera.
   A shared conservative footprint would likely over-bin enough to lose the win.

## Longer-Term Representation Options

1. Projective rational camera bundle.
   Store per-camera or low-order camera-bundle homogeneous coefficients for the
   projected center, similar to STAR PRT. This needs a different primitive
   contract and a new backward chain.

2. Learned neutral gauge.
   Choose a `G_star` gauge that minimizes projection curvature across the nine
   training views. This is elegant but research-heavy and not a quick browser
   optimization.

3. View atlas / inverse-homography residuals.
   STAR's atlas-residual direction may be the closest analog for fixed camera
   sets: project views into local atlases or depth bands, then raster residual
   footprints. This is also a new object/render contract.

## Recommendation

Do not rewrite the browser splat object yet. The next concrete ablation should
be per-lane raster state plus view-dimension dispatch. It should answer the
practical question: how much of the current batch-CLIP win is lost only because
we replay raster forward before backward?

If per-lane state leaves raster as the bottleneck, then try camera-buffer
multi-view dispatch and workgroup staging. Only after that should we prototype
a STAR-inspired camera-bundle primitive.
