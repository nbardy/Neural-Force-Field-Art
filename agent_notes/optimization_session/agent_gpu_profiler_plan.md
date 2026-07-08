# Agent 5 GPU Profiler Plan

Date: 2026-07-08

Scope: map the real profiling options for the Neural-Force-Field-Art
CLIP/splat optimizer on macOS/WebGPU/Metal. This note is intentionally
measurement-only. It does not propose runtime code edits as part of this task.

## Executive Read

We have enough in-repo timing to know the next optimization should be CLIP-led,
but we do not yet have true shader-counter evidence for why the hot CLIP
dispatches are slow. WebGPU timestamp queries now answer "which pass/dispatch
took milliseconds?" They do not answer memory bandwidth, occupancy, cache
misses, register pressure, or stall reason. For that, the next serious session
needs full Xcode/Metal tooling, not another shader rewrite.

Local machine state checked during this pass:

- WebGPU adapter: `apple`, `metal-3`, `apple-m4`, macOS `15.5`.
- Adapter features include `timestamp-query`, `shader-f16`, `subgroups`, and
  Dawn-native/internal features.
- Installed developer path is `/Library/Developer/CommandLineTools`.
- Full Xcode tools are currently missing from `xcrun`: `xctrace`, `gpudebug`,
  `metal`, `metalperftrace`, and `instruments` were not found.
- Chrome is installed at `/Applications/Google Chrome.app` and reports
  `Google Chrome 150.0.7871.46`.

Practical implication: today we can run strong WebGPU timestamp profiling and
Chrome trace recording. True Metal System Trace, Metal debugger, GPU counters,
and `metalperftrace` need full Xcode installed and selected first.

## 1. What We Have Used Already

### Integrated optimizer timing

`tools/splat3d/step_bench.ts` measures the real 3D optimizer schedule:

- raster forward;
- CLIP forward/backward, including batch-major CLIP;
- raster backward;
- Adam;
- display render.

It supports both split-submit wall profiling and GPU timestamp profiling:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
```

Current recorded takeaway from `docs/SPLAT3D_PERF_NOTES.md` and
`agent_notes/optimization_session/integrated_step_timestamp_profile.md`:

- default `3/9`, `batch CLIP x3` integrated GPU timestamp smoke:
  - profile total about `52.82 ms`;
  - raster forward about `1.31 ms`;
  - raster backward about `10.03 ms`;
  - CLIP batch about `41.03 ms`;
  - Adam/display are tiny.
- view-lane raster backward saved less than `1 ms` of timestamped raster
  backward on the smoke, so shallow raster scheduling is not the next big lever.

### CLIP dispatch ranking

`tools/clip/dispatch_profile.ts` isolates every generated CLIP dispatch and can
use timestamp queries:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts > /tmp/clip_b3_ts.csv
```

Current recorded B=3 timestamp ranking:

- `pw_bwd`: about `26%`;
- `spatial_bwd`: about `19%`;
- `conv`: about `16%`;
- `pw+gelu`: about `15%`;
- `pw`: about `15%`;
- `attn_core_bwd`: about `3%`.

This already says attention backward is not first-wave work.

### Sequential ablation matrix

`tools/splat3d/step_matrix.ts` runs step benches sequentially to avoid parallel
GPU contention:

```bash
TRIALS=3 CONFIGS=base=3:3,alt=3:3:gelubwd RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

This is the promotion gate for runtime changes, because isolated kernel wins
have repeatedly failed to survive the full optimizer path.

### Raster occupancy telemetry

`tools/splat3d/raster_telemetry.ts` reads raster tile counts/stops after real
forwards:

```bash
bun tools/splat3d/raster_telemetry.ts
G=12000 CAP=2048 bun tools/splat3d/raster_telemetry.ts
```

Current recorded result: default `G=4096`, `CAP=2048` has zero overflow and
max tile count below `1024`; `CAP=1024` was safe for the initial scene but not
faster in integrated timing.

### Browser smoke and visual correctness

The repo has browser gates:

- `tools/smoke.mjs` for the main WebGPU page with SwiftShader fallback.
- `tools/splat/page_smoke.mjs` for the 2D prompt-to-splats page on real
  headless Metal Chrome.

They prove pages run and render; they are not shader profilers.

### Existing app-side GPU timer

`src/render/webgpu/gputime.ts` already has a ring-buffer timestamp-query helper
for the older particle-art path. It resolves timestamp query pairs off the hot
path and labels the HUD as real GPU time when supported.

## 2. What True GPU Counters Are Available Locally

### Available immediately

These are available on the current machine without installing anything:

- WebGPU `timestamp-query` in Bun/WebGPU.
- WebGPU `timestamp-query` in Chrome if the browser/device path exposes it.
- Chrome tracing through `chrome://tracing`, `about:tracing`, or startup trace
  flags.
- Chrome `chrome://gpu` for backend sanity checks.
- Chrome DevTools Performance for JS/main-thread and frame-level profiling.

These do not expose shader hardware counters.

### Available after full Xcode install

Install/select full Xcode first:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun xctrace list templates
xcrun --find gpudebug
xcrun --find metalperftrace
```

Expected useful tools:

- Instruments `Metal System Trace`: CPU/GPU timeline, queue behavior, GPU
  utilization, memory timeline, performance state, and over-serialization.
- Xcode Metal Debugger / GPU capture: per-command/per-dispatch inspection and
  GPU counters where Chrome/Dawn capture works.
- `gpudebug`: command-line investigation of GPU traces.
- `metalperftrace`: Apple command-line Metal performance tracing and summary
  tool if present in the installed Xcode version.
- Metal Performance HUD: quick live high-level GPU/FPS counters for a Metal
  process, useful for finding a capture window but not enough for CLIP shader
  optimization.

Apple's current Metal tools page explicitly describes Metal System Trace as a
CPU/GPU/memory timeline and says Xcode performance views include hardware
counters. It also mentions `metalperftrace` for collecting traces and readable
summaries. Exact counter names vary by Xcode and GPU generation, so we should
confirm on the installed tool before writing counter-specific gates.

### Not currently available

The current Command Line Tools install is not enough:

```text
xcrun: error: unable to find utility "xctrace", not a developer tool or in PATH
xcrun: error: unable to find utility "gpudebug", not a developer tool or in PATH
xcrun: error: unable to find utility "metalperftrace", not a developer tool or in PATH
```

So if we want true memory/occupancy/stall evidence, setup is the first task.

## 3. Exact Profiling Workflows

### A. WebGPU timestamp session in Bun

Use this first because it is runnable today and gives the most direct
repo-specific signal.

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_clip_b3_dispatch_ts.csv
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
```

Use this to decide whether the top dispatch groups are stable across B=3 and
full 9-view mode. Do not run several GPU benches in parallel.

What this answers:

- Which CLIP dispatch families dominate GPU time?
- Does integrated CLIP still dominate after current raster/batch changes?
- Is raster backward large enough to justify another raster rewrite?
- Are Adam/display/clear worth touching?

What it cannot answer:

- whether `pw_bwd` is memory-bandwidth bound or occupancy bound;
- whether `spatial_bwd` is stalled on global loads or register pressure;
- whether f16 would improve arithmetic throughput, memory traffic, or neither;
- cache behavior or threadgroup memory bank conflicts.

### B. Chrome trace of the browser page

Build and serve the 3D page:

```bash
npx parcel build --no-scope-hoist --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
```

Launch a clean Chrome instance with WebGPU/Metal and startup tracing:

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" \
  --user-data-dir=/tmp/nffa_trace_profile \
  --use-angle=metal \
  --ignore-gpu-blocklist \
  --enable-webgpu-developer-features \
  --disable-background-timer-throttling \
  --disable-backgrounding-occluded-windows \
  --disable-renderer-backgrounding \
  --trace-startup='gpu,dawn,disabled-by-default-gpu.dawn,disabled-by-default-gpu.service,devtools.timeline,blink,renderer.scheduler,v8,benchmark' \
  --trace-startup-duration=30 \
  --trace-startup-file=/tmp/nffa_webgpu_trace.json \
  --trace-startup-format=json \
  http://localhost:8799/dist/splat3d.html
```

Manual alternative:

1. Open only the target tab and `chrome://tracing`.
2. Press Record.
3. Select categories around `gpu`, `dawn`, `disabled-by-default-gpu.dawn`,
   `disabled-by-default-gpu.service`, `devtools.timeline`, `blink`,
   `renderer.scheduler`, and `v8`.
4. Switch to the splat tab, run a short optimize window, pause two seconds,
   then stop and save.
5. Load the trace in `chrome://tracing` or `https://ui.perfetto.dev`.

What this answers:

- Is Chrome/Dawn leaving gaps between command buffers?
- Is the GPU process or renderer thread CPU-bound while encoding?
- Are pipeline creation/JIT events leaking into measured runs?
- Are maps/readbacks or model fetch/compile operations overlapping training?
- Is the browser throttling or backgrounding the page?

What it cannot answer:

- per-shader memory bandwidth or occupancy;
- exact WGSL line-level stall reason;
- reliable full optimizer GPU time unless combined with WebGPU timestamps.

### C. Chrome sanity checks

Open:

```text
chrome://gpu
```

Confirm:

- WebGPU is enabled;
- ANGLE/backend is Metal;
- no SwiftShader/fallback adapter for real perf runs;
- no obvious GPU blocklist or device-loss messages.

This is setup validation, not profiling.

### D. Metal System Trace through Instruments/xctrace

Prerequisite:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun xctrace list templates | rg 'Metal|GPU'
```

For Chrome, record all processes because WebGPU work lives across the browser,
renderer, and GPU processes:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --all-processes \
  --time-limit 20s \
  --output /tmp/nffa_metal_system.trace
open /tmp/nffa_metal_system.trace
```

Then launch or switch to Chrome and run a short optimizer window during the
20-second recording. Keep the capture narrow: 2 seconds idle, 5-10 seconds of
optimization, 2 seconds idle.

For a Bun/WebGPU command, use `--launch` so the trace owns the target process:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 20s \
  --target-stdout - \
  --output /tmp/nffa_bun_metal_system.trace \
  --launch -- bun tools/splat3d/step_bench.ts
```

Add env vars through `--env` when needed:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 20s \
  --target-stdout - \
  --env TIMESTAMP=1 \
  --env CLIP_BATCH=3 \
  --env VIEWS=3 \
  --env RUNS=10 \
  --env WARMUP=3 \
  --output /tmp/nffa_bun_metal_system.trace \
  --launch -- bun tools/splat3d/step_bench.ts
```

What this answers:

- Is the GPU queue saturated or idle between WebGPU command buffers?
- Are compute dispatches serialized when they could overlap?
- Is there CPU/GPU imbalance from command encoding, queue submits, or readbacks?
- Are memory allocations or large buffer traffic visible at step boundaries?
- Is thermal/performance state changing during noisy benches?

What it only partially answers:

- It may not show nice CLIP dispatch names because Chrome/Dawn translates WGSL
  into Metal under the browser process.
- It is better for timeline/queue/system behavior than for line-level shader
  tuning unless paired with Metal GPU capture/counters.

### E. Xcode Metal debugger / GPU capture

Prerequisite:

```bash
xcrun --find gpudebug
xcrun --find metal
```

Manual workflow:

1. Install/select full Xcode.
2. Launch the target with the smallest repro, ideally a Bun/WebGPU bench rather
   than the full browser page.
3. Capture a GPU frame/trace in Xcode or use `gpudebug` if available.
4. Inspect compute dispatches, resource bindings, shader timeline/counters, and
   memory report.
5. Export screenshots/tables into `agent_notes/optimization_session/`.

Command-line starting point once `gpudebug` exists:

```bash
xcrun gpudebug --help
```

What this answers:

- Per-dispatch hardware counters, when capture works.
- Whether hot CLIP kernels are ALU-bound, memory-bound, occupancy-limited, or
  barrier/synchronization-limited.
- Whether threadgroup memory staging is helping or reducing occupancy.
- Whether f16 changes throughput, bandwidth pressure, or just precision.

Risk:

- Browser WebGPU capture can be brittle because Chrome is multi-process and Dawn
  owns the Metal objects. A Bun/WebGPU microbench may be a better first capture.

### F. Metal Performance HUD

Quick live check after full Xcode/runtime support:

```bash
env METAL_HUD_ENABLED=1 \
  /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --user-data-dir=/tmp/nffa_hud_profile \
  --use-angle=metal \
  --ignore-gpu-blocklist \
  --enable-webgpu-developer-features \
  http://localhost:8799/dist/splat3d.html
```

What this answers:

- Is the app obviously GPU-bound?
- Is frame pacing stable?
- Which short interaction window is worth capturing in Xcode/Instruments?

What it cannot answer:

- CLIP kernel bottleneck details.
- Per-dispatch bandwidth/stall/occupancy.

### G. `metalperftrace`

After full Xcode:

```bash
xcrun --find metalperftrace
xcrun metalperftrace --help
```

Use it only after validating syntax locally. Apple's tools page describes it as
a command-line way to collect Metal performance traces and readable summaries,
but this machine cannot verify the CLI today.

Likely best use:

- run it around `bun tools/splat3d/step_bench.ts`, not the full browser, because
  the target process is simpler;
- compare f32 CLIP versus future f16 CLIP under identical inputs;
- archive the summary in `agent_notes/optimization_session/`.

## 4. Bottleneck Questions By Tool

| Question | Best Tool | Why |
| --- | --- | --- |
| Is CLIP still the integrated bottleneck? | `TIMESTAMP=1 tools/splat3d/step_bench.ts` | Measures actual raster/CLIP/backward/Adam pass spans. |
| Which CLIP dispatch groups are hot? | `TIMESTAMP=1 tools/clip/dispatch_profile.ts` | Gives per-dispatch and grouped GPU-time ranking. |
| Is browser scheduling/queue submit overhead visible? | Chrome trace | Shows renderer/GPU-process/queue gaps and compilation events. |
| Are benches polluted by Metal JIT/pipeline creation? | Chrome trace plus warmup discipline | Trace shows pipeline and browser activity; benches show cold/warm deltas. |
| Is raster limited by tile overflow or long tile lists? | `tools/splat3d/raster_telemetry.ts` | Reads tile-count and tile-stop distributions. |
| Is `pw_bwd` memory-bound or occupancy-bound? | Metal GPU counters / capture | Requires hardware counters; WebGPU timestamps only show duration. |
| Does `spatial_bwd` suffer from global weight loads? | Metal GPU counters plus a targeted microbench | Need bandwidth/cache/occupancy counters around that dispatch. |
| Would CLIP f16 produce a real 2x path? | WebGPU timestamps for wall plus Metal counters for cause | Timestamp proves speed; counters explain bandwidth/ALU impact. |
| Are GPU queues idle because we over-split passes? | Metal System Trace and Chrome trace | Timeline-level view of command buffer gaps and serialization. |
| Are readbacks/maps stalling the loop? | Chrome trace and Metal System Trace | Shows CPU/GPU sync and map/readback timing. |

## 5. Recommended Next Profiling Session Before Code Changes

Do this before another shader optimization:

1. Run a stable WebGPU timestamp baseline:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/dispatch_profile.ts
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/dispatch_profile.ts > agent_notes/optimization_session/clip_b3_dispatch_timestamp_baseline.csv
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=10 WARMUP=5 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

2. Record one Chrome trace of the browser page:

```bash
npx parcel build --no-scope-hoist --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
```

Then launch Chrome with the startup trace command from section B, run a short
optimize window, and save `/tmp/nffa_webgpu_trace.json`.

3. Install/select full Xcode if not already present:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun xctrace list templates | rg 'Metal|GPU'
```

4. Record a Metal System Trace for the Bun benchmark:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 20s \
  --target-stdout - \
  --env TIMESTAMP=1 \
  --env CLIP_BATCH=3 \
  --env VIEWS=3 \
  --env RUNS=10 \
  --env WARMUP=5 \
  --output /tmp/nffa_bun_metal_system.trace \
  --launch -- bun tools/splat3d/step_bench.ts
```

5. Only after that, choose one code experiment:

- If counters show CLIP pointwise backward is memory-bandwidth-bound, try f16
  weights/slots or a deeper pointwise layout change.
- If counters show low occupancy/register pressure in `spatial_bwd`, try a
  shape-specific kernel rather than a generic staging rewrite.
- If Chrome/Metal traces show queue gaps or browser scheduling overhead, work on
  pass/submit structure.
- If none of those are true, do not keep shaving raster schedulers; revisit
  algorithmic CLIP-call reduction such as grid CLIP or N-of-K schedule changes.

## Notes On Current Optimization Claims

The earlier large f16 win was in the particle `advect` path, not the CLIP
vision model. `src/render/webgpu/advect_wgsl.ts` documents a shipped f16 fast
path around `9.3 -> 6.5 ms/step` for the small force-field MLP on Apple Metal.
CLIP f16 remains an ablation item (`docs/SPLAT3D_ABLATION_QUEUE.md` section 8)
with no promoted CLIP f16 implementation yet.

So the honest current state is:

- f16 works and helped in the older force-field advect kernel;
- text ONNX uses an fp16 model internally, but it runs once per prompt and is
  not hot-loop CLIP vision;
- hot-loop MobileCLIP vision training is still f32 WGSL today;
- the local adapter supports `shader-f16`, so CLIP f16 is technically possible
  and should be profiled after the counter/timestamp baseline.

## Source Pointers

Local:

- `docs/SPLAT3D_PERF_NOTES.md`
- `docs/SPLAT3D_ABLATION_QUEUE.md`
- `tools/clip/README.md`
- `tools/clip/dispatch_profile.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `tools/splat3d/raster_telemetry.ts`
- `src/render/webgpu/gputime.ts`
- `agent_notes/optimization_session/integrated_step_timestamp_profile.md`
- `agent_notes/optimization_session/clip_timestamp_dispatch_profile.md`

External references checked:

- WebGPU timestamp query API: https://developer.mozilla.org/en-US/docs/Web/API/GPUQuerySet
- WebGPU compute-pass timestamp fields: https://developer.mozilla.org/en-US/docs/Web/API/GPUCommandEncoder/beginComputePass
- Chrome tracing overview: https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/
- Chrome trace recording workflow: https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/recording-tracing-runs/
- `xctrace` command reference: https://keith.github.io/xcode-man-pages/xctrace.1.html
- Apple Metal tools overview: https://developer.apple.com/metal/tools/
