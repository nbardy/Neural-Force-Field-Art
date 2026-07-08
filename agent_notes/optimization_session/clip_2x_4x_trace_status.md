# CLIP 2-4x And Shader Trace Status

Date: 2026-07-08

## Current Answer

The learnings are recorded in `agent_notes/optimization_session/`, especially:

- `current_strategy_reflection.md`
- `agent_gpu_profiler_plan.md`
- `agent_f16_reality_check.md`
- `agent_pointwise_bottleneck.md`
- `clip_timestamp_dispatch_profile.md`
- `agent_grid_clip_strategy.md`

What is still missing is a real GPU-counter capture. We have WebGPU timestamp
profiles and wall-time benches, but we have not yet used Metal shader counters
for memory bandwidth, occupancy, register pressure, cache behavior, or
threadgroup-memory pressure.

The code-level experiment trail is now in `experiments/clip_forks/`. Important
current outcomes:

- `v02_f16_weights`: f16 CLIP weight storage is not promotable yet. It halves
  weight storage, but input-gradient cosine failed the strict gate and timestamp
  speed was slightly worse in the measured run.
- `v06_view_sampling`: N-of-K view sampling is the biggest practical wall-time
  lever so far because it reduces CLIP calls per optimizer step while preserving
  camera coverage over time.
- `v07_spatial_bwd_depthwise4`: vectorizing depthwise spatial backward four
  horizontal pixels at a time passed correctness and improved integrated 3D
  step median `53.12 ms -> 49.96 ms`, but it is still gated because the
  isolated timestamp profile was mixed.

## Tooling Reality

This machine currently has Command Line Tools selected:

```text
/Library/Developer/CommandLineTools
```

`/usr/bin/xctrace` exists on PATH, but `xcrun` cannot find the Xcode GPU tools:

```text
xcrun: error: unable to find utility "xctrace", not a developer tool or in PATH
xcrun: error: unable to find utility "metalperftrace", not a developer tool or in PATH
xcrun: error: unable to find utility "gpudebug", not a developer tool or in PATH
xcrun: error: unable to find utility "metal", not a developer tool or in PATH
```

So the honest status is:

- WebGPU timestamp profiling: yes, used.
- Integrated split-submit wall profiling: yes, used.
- CLIP dispatch ranking: yes, used.
- Browser/Chrome trace: planned, not yet used.
- Metal GPU shader counters: not yet available through the current selected
  developer toolchain.

## Fresh Timing Snapshot

Commands run:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Isolated B=3 CLIP timestamp ranking:

| Group | Time | Share |
| --- | ---: | ---: |
| `pw_bwd` | `25.82 ms` | `23.8%` |
| `spatial_bwd` | `23.79 ms` | `21.9%` |
| `conv` | `20.25 ms` | `18.7%` |
| `pw` | `14.09 ms` | `13.0%` |
| `pw+gelu` | `13.96 ms` | `12.9%` |
| `gelu_bwd` | `3.74 ms` | `3.4%` |
| `attn_core_bwd` | `2.56 ms` | `2.4%` |

Total isolated median timestamp sum: `108.46 ms`.

Integrated wall-time snapshot:

| Mode | Normal Step Avg | Profile Total | CLIP Profile | Raster Bwd Profile |
| --- | ---: | ---: | ---: | ---: |
| `3/9` per-view, CLIP batch 3 | `119.31 ms` | `99.50 ms` | `83.67 ms` | `11.23 ms` |
| `9/9` per-view, CLIP batch 3 | `155.54 ms` | `162.68 ms` | `122.66 ms` | `31.69 ms` |
| `9/9` grid9+2, CLIP batch 3 | `119.35 ms` | `150.19 ms` | `90.61 ms` | `43.19 ms` |

Interpretation:

- CLIP is still the main integrated cost.
- In CLIP, pointwise/backward plus spatial/conv families dominate.
- Attention backward is not the first target.
- Grid9+2 is faster than full `9/9` per-view wall time in this short run, but
  it increases raster work and should be treated as a schedule/signal ablation,
  not a pure CLIP-kernel speedup.

## Is There A Real 2-4x CLIP Path?

A pure same-model, same-resolution, every-step MobileCLIP train pass is unlikely
to get a clean 4x from one small shader tweak. A real 2-4x wall-clock gain is
still plausible, but probably needs stacked changes:

1. Reduce CLIP calls per useful optimizer progress.
   - N-of-K view sampling.
   - Grid9+closeup schedule.
   - Alternate full-view passes and cheaper proxy passes.

2. Revisit precision more selectively.
   - All-weight f16 was tested and did not pass the gradient-quality/speed gate.
   - The remaining possible lane is selective f16 for activations or specific
     read-mostly buffers with f32 accumulation, but this needs tighter
     per-kernel proof before broad edits.

3. Rewrite the hot pointwise/spatial kernels with a profiling target.
   - Current hot groups are `pw_bwd`, `pw`, `pw+gelu`, `spatial_bwd`, and
     `conv`.
   - The next fork should test weight/activation layout and tiling changes, not
     broad in-place shader edits.

4. Cut dispatch/materialization overhead where correctness allows it.
   - Local backward fusions are plausible.
   - Whole-CLIP fusion is not realistic because training needs saved
     intermediate activations and dispatch-boundary synchronization.

5. Consider a cheaper teacher/proxy loop.
   - Use full CLIP as the teacher/checkpoint.
   - Use a smaller learned or hand-built proxy for most inner-loop steps.
   - This is the highest upside and highest research risk.

## Next Profiling Step

Before another deep shader rewrite, run one of these:

1. Install/select full Xcode and rerun the Metal tool checks:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun xctrace list templates | rg -i 'metal|gpu'
xcrun --find metalperftrace
xcrun --find gpudebug
```

2. If full Xcode is not available, use Chrome tracing next:

```bash
mkdir -p /tmp/nffa_trace_profile
open -na "Google Chrome" --args \
  --user-data-dir=/tmp/nffa_trace_profile \
  --enable-unsafe-webgpu \
  --trace-startup='gpu,dawn,disabled-by-default-gpu.dawn,disabled-by-default-gpu.service,devtools.timeline,blink,renderer.scheduler,v8,benchmark' \
  --trace-startup-duration=30 \
  --trace-startup-file=/tmp/nffa_webgpu_trace.json \
  --trace-startup-format=json \
  http://localhost:1234/splat3d.html
```

Chrome trace will not give per-shader bandwidth/occupancy counters, but it can
show queue gaps, command-buffer behavior, browser scheduling, and compilation
events.

## Decision

Keep chasing two tracks:

- Productive schedule wins: N-of-K random/epoch views, grid/contact-sheet CLIP,
  and lower-frequency all-view refreshes. This is the only near-term path with
  realistic 2x+ wall-time potential without changing the CLIP model.
- Narrow shader forks: pointwise backward/forward tiling, conv/depthwise
  spatial variants, and selective precision. The `depthwise4` result shows
  there is real shader headroom, but the size of the win argues for stacked
  5-15% improvements, not a single magic 4x kernel.
