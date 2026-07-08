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
node experiments/clip_forks/diff_fork.mjs v08_grid_contact_sheet_prompt
node experiments/clip_forks/diff_fork.mjs v09_direct_grid_raster
node experiments/clip_forks/diff_fork.mjs v10_shared_w_pointwise_forward
node experiments/clip_forks/diff_fork.mjs v11_backward_local_fusions
node experiments/clip_forks/diff_fork.mjs v12_chrome_dawn_trace
node experiments/clip_forks/diff_fork.mjs v13_cached_clip_gradient
node experiments/clip_forks/diff_fork.mjs v14_cadence_quality_gate
node experiments/clip_forks/diff_fork.mjs v15_cached_lr_scale
node experiments/clip_forks/diff_fork.mjs v16_grid_real_prompt_quality
node experiments/clip_forks/diff_fork.mjs v17_pointwise_roofline
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
- `v08_grid_contact_sheet_prompt`: explicit text prompt for the 3x3 grid CLIP
  lane, with a same-text toggle.
- `v09_direct_grid_raster`: render grid contact-sheet cells at `80x80` instead
  of full `256x256` scratch, while sharing splat params and gradients.
- `v10_shared_w_pointwise_forward`: env-gated shared-weight batch pointwise
  forward allowlist; recorded as no-promote after integrated timing.
- `v11_backward_local_fusions`: refresh of the existing GELU/residual backward
  pointwise fusion gates on the current grid80+depthwise stack.
- `v12_chrome_dawn_trace`: reproducible Chrome/Puppeteer trace helper for
  browser Dawn/WebGPU scheduling evidence.
- `v13_cached_clip_gradient`: benchmark-only CLIP cadence gate that reuses
  cached full-resolution `dL/dimage` on skipped steps.
- `v14_cadence_quality_gate`: fixed-wall-clock full-teacher quality probe for
  cached CLIP gradient cadence.
- `v15_cached_lr_scale`: cached-gradient cadence variant that scales Adam
  learning rates only on cached steps.
- `v16_grid_real_prompt_quality`: real MobileCLIP text-prompt quality gate for
  `grid9_close2 + directgrid` versus the default `3/9` per-view schedule.
- `v17_pointwise_roofline`: static and timestamp-backed report for pointwise
  CLIP math, memory layout, f16 reality, and next exact-math pointwise forks.

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
- `v08_grid_contact_sheet_prompt`: 3D camera prompt builders, grid CLIP page
  controls, and 3D benchmark context.
- `v09_direct_grid_raster`: 3D raster engine, grid CLIP layout, optimizer/page
  controls, and grid/step benchmarks.
- `v10_shared_w_pointwise_forward`: batch pointwise emitters, 3D optimizer
  wiring, CLIP train matrix, and 3D step matrix.
- `v11_backward_local_fusions`: CLIP backward emitters, 3D optimizer wiring,
  backward correctness tests, dispatch profiler, and 3D step matrix.
- `v12_chrome_dawn_trace`: browser trace helper, Chrome trace plan, and
  perf-note context.
- `v13_cached_clip_gradient`: 3D optimizer schedule, 3D step bench/matrix, and
  perf-note context.
- `v14_cadence_quality_gate`: cadence quality harness, 3D optimizer context,
  3D step bench/matrix, and perf-note context.
- `v15_cached_lr_scale`: 3D optimizer schedule hook, 3D step bench/matrix,
  cadence quality harness, and perf-note context.
- `v16_grid_real_prompt_quality`: grid quality harness, prompt builders, 3D
  optimizer context, grid test/bench context, and perf-note context.
- `v17_pointwise_roofline`: CLIP WGSL emitters, dispatch profiler, batch
  pointwise experiments, pointwise report, perf-note context, and prior
  pointwise agent notes.

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
- `agent_notes/optimization_session/current_strategy_reflection_v14_2026_07_08.md`
- `agent_notes/optimization_session/v15_agent_synthesis_2026_07_08.md`
- `agent_notes/optimization_session/nunchaku_clip_approx_2026_07_08.md`
- `agent_notes/optimization_session/pointwise_roofline_v17_2026_07_08.md`
