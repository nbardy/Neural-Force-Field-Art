# CLIP / Raster Experiment Forks

This directory is the trail for larger optimization attempts that should not be
buried as silent edits to the default shaders. Each `vNN_*` folder owns one
hypothesis, one rollback point, and one measurement story.

## Rules

1. Keep default behavior unchanged until a fork wins.
2. Put risky code behind an env/UI gate such as `CLIP_LAYOUT=grid9_close2`,
   `PRECISION=f16`, or `FUSE_RESIDUAL_BWD_PW=1`.
3. Prefer copied variant emitters or small wrapper modules over mutating the hot
   default shader path in place.
4. Record the exact commands, adapter, and before/after numbers in the fork
   folder.
5. Commit each attempt separately, including rejected gates when the result is
   informative.

## Standard Gate

Every performance fork should clear these checks before promotion:

```bash
bun tools/clip/fused_test.ts
bun tools/clip/bwd_test.ts
TIMESTAMP=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Use the gate-specific env vars in addition to the commands above.

## Active Fork Lanes

- `v01_grid9_close2`: one 3x3 contact-sheet CLIP lane plus two full-resolution
  close-up lanes.
- `v02_f16_weights`: f16 CLIP weight storage with f32 activations, gradients,
  and accumulation.
- `v03_residual_bwd_pw_fusion`: fold legal `residual_bwd -> pw_bwd` pairs.
- `v04_pointwise_tile_rewrite`: new pointwise matmul tiling variants.
- `v05_gpu_counter_trace`: Chrome / Xcode / Metal trace plan for real shader
  bottleneck evidence.

