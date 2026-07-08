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

Each active fork contains a `snapshot/` directory with the relevant live source
files copied at the start of the fork. Use that snapshot for cheap diffs and
rollback review; do not import code from `snapshot/` at runtime.

```bash
node experiments/clip_forks/diff_fork.mjs v02_f16_weights
node experiments/clip_forks/diff_fork.mjs v04_pointwise_tile_rewrite
node experiments/clip_forks/diff_fork.mjs v07_spatial_bwd_depthwise4
```

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
- `v06_view_sampling`: same-resolution N-of-K camera schedules.
- `v07_spatial_bwd_depthwise4`: depthwise-only vectorized CLIP spatial
  backward lane.

## Snapshot Coverage

- `v01_grid9_close2`: 3D optimizer/grid compositor and 3D step/grid tests.
- `v02_f16_weights`: CLIP runtime, WGSL emitters, generator, correctness tests,
  and train bench.
- `v03_residual_bwd_pw_fusion`: CLIP WGSL emitters and backward/profile tests.
- `v04_pointwise_tile_rewrite`: CLIP pointwise emitters and pointwise benches.
- `v05_gpu_counter_trace`: integrated 3D profiler and CLIP dispatch profiler.
- `v06_view_sampling`: 3D optimizer/page controls and 3D step benches.
- `v07_spatial_bwd_depthwise4`: CLIP backward emitter, backward tests, 3D
  optimizer wiring, and benchmark env gates.

## Related Notes

- `docs/SPLAT3D_ABLATION_QUEUE.md`
- `docs/SPLAT3D_PERF_NOTES.md`
- `docs/CLIP_BATCHING_NOTES.md`
- `tools/clip/README.md`
- `agent_notes/optimization_session/agent_f16_reality_check.md`
- `agent_notes/optimization_session/agent_pointwise_bottleneck.md`
- `agent_notes/optimization_session/agent_fusion_bigger_leaps.md`
- `agent_notes/optimization_session/agent_grid_clip_strategy.md`
- `agent_notes/optimization_session/agent_gpu_profiler_plan.md`
- `agent_notes/optimization_session/clip_2x_4x_trace_status.md`
