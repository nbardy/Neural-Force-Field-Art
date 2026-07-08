# 3D Splat Performance Notes

Last updated: 2026-07-08

## Current Timing Surface

The 3D page now samples a split-submit wall-time profile every 30 optimizer
steps. The overlay reports:

- raster forward and backward
- CLIP forward and backward
- Adam step
- clear/display overhead
- sampled optimizer-step total

This is intentionally labeled as sampled wall time. It is good enough to decide
whether CLIP or raster is the current bottleneck, but it is not exact GPU
timestamp attribution for the normal single-submit `step()` path.

## Measurements

Default 4096-splat scene, 256px renders, 9 cameras, Bun/WebGPU on 2026-07-08:

| Views / Step | Normal Step Avg | Split Profile Total | Raster Fwd | Raster Bwd | CLIP Fwd | CLIP Bwd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 / 9 | 205.26 ms | 228.20 ms | 8.16 ms | 33.38 ms | 77.77 ms | 102.81 ms |
| 5 / 9 | 122.02 ms | 134.83 ms | 5.38 ms | 19.85 ms | 39.77 ms | 57.79 ms |
| 3 / 9 | 69.94 ms | 75.57 ms | 2.32 ms | 11.20 ms | 24.07 ms | 34.96 ms |

Takeaway: random N-of-K view sampling is a real wall-clock lever. The 3/9 path
is roughly 2.9x faster than 9/9 while preserving stochastic coverage of all
camera prompts over multiple steps. The implementation uses shuffled epochs, so
3/9 still randomizes order but covers all 9 views every 3 optimizer steps. CLIP
remains around 78-79% of sampled wall time, so deeper speedups should attack
CLIP calls/batching/precision before large raster rewrites.

Tried and reverted: exact circle-vs-tile support pruning in emit/binning. It was
mathematically safe, but on this workload it made raster forward/backward flat
to slightly slower, likely because the extra per-tile math outweighed the small
bin-count reduction at 256px / 4096 splats.

## Four-Agent Pass: Main Read

### CLIP

The most likely CLIP bottleneck is still pointwise matmul. Static work is
dominated by forward pointwise and `pw_bwd`, so another clean 4x from kernel
tuning alone is unlikely unless we change precision, reduce CLIP calls, or make
a larger matmul strategy shift.

Next measurements:

- Per-dispatch timing grouped by label/shape for B=1, B=3, and B=9.
- Pointwise forward/backward timing for the repeated 256->768 and 768->256
  blocks.
- Stem `spatial_bwd`, especially the first 256x256 3<-64 kernel.
- Aggregate GELU forward/backward share.
- Attention backward only if timestamps show it above roughly 5-10%.

Most promising CLIP experiments:

- Use batch-major CLIP for multi-view training.
- Try f16 weights/activations with f32 reductions behind feature gating.
- Stage weights in `spatial_bwd`.
- Fuse pointwise/GELU/residual blocks if dispatch and scratch traffic show up.
- Add per-dispatch timestamp-query profiling to the batch benches.

### Raster

The current rasterizer already has two key FasterGS-style ideas: per-tile sorted
IDs are saved in forward, and backward reuses the saved order plus tile stop
counts. The best next raster experiments are smaller and directly applicable:

- Alias raster image/grad buffers with CLIP input/grad buffers to remove two
  full-image copies per view.
- Stage derived splat parameters in workgroup chunks for forward/backward.
- Add tile overflow telemetry before tuning caps or thresholds.
- Benchmark workgroup-local gradient reductions before replacing global fixed
  point atomics.

Do not port the dynamic viewer bucket-sort path into training. It is useful for
preview, but the training backward path needs the saved per-tile order and stop
state.

## Next Decision Rule

Use the overlay first:

- If CLIP dominates, land batch-major training in the 3D optimizer before deep
  shader rewrites.
- If raster dominates, start with buffer aliasing, overflow telemetry, and
  workgroup staging. Re-test circle-tile pruning only at higher resolution or
  larger splat counts.
- If Adam/display/clear are visible but small, leave them alone until CLIP and
  raster are under control.
