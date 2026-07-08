# v19 Real GPU Profiling On macOS / Chrome / Metal

Date: 2026-07-08

Scope: audit the profiling tools we can use for
`/Users/nicholasbardy/git/Neural-Force-Field-Art` on the current Mac/Chrome/Metal
stack. This is documentation-only. No runtime/source files were edited.

## Executive Read

Usable now:

- WebGPU timestamp-query profiling in the repo's Bun/WebGPU harnesses.
- Chrome DevTools / Perfetto tracing of the browser page through
  `tools/webgpu_trace.mjs` or Chrome startup tracing.
- Dawn debug toggles through Chrome command-line flags, mainly for shader dumps,
  labels, and capture/debug context.
- `chrome://gpu` and screenshots/smoke probes for backend sanity checks.

Not usable on this machine right now:

- Instruments / `xctrace`.
- Xcode Metal System Trace.
- Xcode Metal Debugger / GPU capture UI.
- `gpudebug`, `gpucapture`, `metal`, and `metalperftrace`.
- Hardware counter proof for memory bandwidth, occupancy, register pressure,
  cache behavior, ALU limiter, barrier stalls, or threadgroup-memory pressure.

Current local check:

```bash
xcode-select -p
```

Output:

```text
/Library/Developer/CommandLineTools
```

```bash
xcrun --find xctrace
xcrun --find instruments
xcrun --find metal
xcrun --find gpudebug
xcrun --find metalperftrace
```

Each returned `xcrun: error: unable to find utility ...`. Full Xcode must be
installed and selected before we can run Apple Metal System Trace or counters.

Chrome is available:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version
```

Output:

```text
Google Chrome 150.0.7871.46
```

The Bun/WebGPU adapter is the real Apple M4 Metal adapter and exposes the
features we need for current timestamp profiling:

```bash
bun -e 'import { setupGlobals } from "bun-webgpu"; setupGlobals(); const gpu = globalThis.navigator?.gpu; console.log("navigator.gpu", !!gpu); const adapter = await gpu.requestAdapter(); console.log("adapter", !!adapter); if (adapter) { console.log("info", JSON.stringify(adapter.info ?? null)); console.log("features", Array.from(adapter.features).sort().join(",")); console.log("timestamp-query", adapter.features.has("timestamp-query")); console.log("shader-f16", adapter.features.has("shader-f16")); console.log("subgroups", adapter.features.has("subgroups")); }'
```

Observed:

```text
navigator.gpu true
adapter true
info {"vendor":"apple","architecture":"metal-3","device":"apple-m4","description":"Metal driver on macOS Version 15.5 (Build 24F74)",...,"isFallbackAdapter":false}
timestamp-query true
shader-f16 true
subgroups true
```

## 1. WebGPU Timestamp Queries

### What to use

Use the repo's existing timestamp paths first. They are the most direct
benchmarking tools available today.

CLIP dispatch ranking:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts > /tmp/nffa_clip_b3_dispatch_ts.csv
```

Integrated optimizer attribution:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_LAYOUT=grid9_close2 GRID_DIRECT_RASTER=1 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Sequential promotion matrix:

```bash
TRIALS=3 CONFIGS=base=3:3,grid=9:3:grid9 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

For source-level variants, run isolated timestamp first and then integrated
step timing. Do not run multiple WebGPU benches in parallel; queue contention
and thermals make the results useless.

### What it can prove

- Which CLIP dispatch families dominate GPU time.
- Whether `pw_bwd`, `pw+gelu`, `pw`, `spatial_bwd`, `conv`, or raster passes
  are worth optimizing.
- Whether a fork reduces elapsed GPU time for the isolated dispatches it
  targets.
- Whether integrated optimizer attribution still says CLIP dominates after
  raster/grid changes.
- Whether a pass is too small to justify more work. Adam/display/clear have
  repeatedly been in that bucket.

### What it cannot prove

- Memory bandwidth saturation.
- Cache hit rate or cache thrash.
- Occupancy.
- Register pressure/spills.
- ALU versus memory versus launch-limited cause.
- Threadgroup-memory bank conflicts.
- Metal compiler line-level behavior.

Interpretation rule: timestamps answer "how long did this pass take?" They do
not answer "why did this shader take that long?"

### Browser caveat

Chrome can expose timestamp queries when the adapter/device support the
`timestamp-query` feature and the page explicitly requests it. The current
`tools/webgpu_trace.mjs` path does not add browser timestamp instrumentation by
itself. It records Chrome/Dawn trace events. The repo's usable timestamp tools
today are the Bun/WebGPU benchmark harnesses above.

Chrome's WebGPU developer-features documentation says timestamp queries measure
GPU command execution and that enabling WebGPU Developer Features disables the
100 microsecond timestamp quantization used for privacy protection during
normal browsing. Use this only for development/profiling.

## 2. Chrome / Perfetto / DevTools Tracing

### Existing helper

`tools/webgpu_trace.mjs` is the reproducible path. It launches Chrome through
Puppeteer with WebGPU/Metal flags, waits for the 3D page to run, records a
DevTools trace, writes a screenshot, and prints before/after app state.

Build and serve the production page from the repo root so `/dist/` assets and
`/models/` weights are same-origin:

```bash
npx parcel build --no-scope-hoist --no-cache --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
GRID9=1 DIRECT_GRID=1 SETUP_TIMEOUT_MS=180000 node tools/webgpu_trace.mjs http://localhost:8799/dist/splat3d.html 12000 /tmp/nffa_trace
```

For a headed run:

```bash
HEADED=1 GRID9=1 DIRECT_GRID=1 SETUP_TIMEOUT_MS=180000 node tools/webgpu_trace.mjs http://localhost:8799/dist/splat3d.html 12000 /tmp/nffa_trace
```

Useful environment variables:

```text
PROMPT="a photo of a cat"
TRACE_PROFILE_DIR=/tmp/nffa_webgpu_trace_profile
CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

Open the resulting JSON in Perfetto or Chrome tracing:

```bash
open https://ui.perfetto.dev
```

or open `chrome://tracing` in Chrome and load the JSON.

The v12 fork recorded this exact working shape:

```text
Mode: grid9_close2, gridDirectRaster=true, views=9, clipBatch=3
Steps: 2 -> 35
Events: 40,331
Top buckets included DawnCommands, Metal backpressure,
DeviceMTL::SubmitPendingCommandBuffer, CommandEncoder::Finish, Queue::Submit.
```

### Startup trace alternative

Use this when Puppeteer is too intrusive or when you want to interact manually:

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" \
  --user-data-dir=/tmp/nffa_trace_profile \
  --use-angle=metal \
  --ignore-gpu-blocklist \
  --enable-webgpu-developer-features \
  --enable-unsafe-webgpu \
  --disable-background-timer-throttling \
  --disable-backgrounding-occluded-windows \
  --disable-renderer-backgrounding \
  --trace-startup='gpu,disabled-by-default-gpu.dawn,disabled-by-default-gpu.graphite.dawn,disabled-by-default-webgpu,disabled-by-default-skia.gpu,devtools.timeline,disabled-by-default-devtools.timeline,blink,blink.user_timing,renderer.scheduler,disabled-by-default-renderer.scheduler,v8,v8.execute,toplevel,benchmark' \
  --trace-startup-duration=30 \
  --trace-startup-file=/tmp/nffa_webgpu_trace.json \
  --trace-startup-format=json \
  http://localhost:8799/dist/splat3d.html
```

Keep the capture short: 2 seconds idle, 5-10 seconds optimize, 2 seconds idle.
Long traces are noisy and hard to inspect.

### Manual trace alternative

1. Start the built page in one tab.
2. Open `chrome://tracing` or `about:tracing`.
3. Record with GPU, Dawn, WebGPU, DevTools timeline, renderer scheduler, V8,
   and benchmark categories enabled.
4. Run a short optimizer window.
5. Stop and save the trace JSON.
6. Load it in `chrome://tracing` or Perfetto.

Chromium's trace docs recommend short captures, a narrow activity window, and
idle padding before/after the action. That guidance matters here because the
browser has renderer, GPU-process, compositor, V8, network, and model-loading
activity mixed into the same file.

### What Chrome tracing can prove

- The page is actually on the intended WebGPU/Metal path.
- Dawn/WebGPU command traffic exists during the measured optimizer window.
- Queue submits, command encoder finish events, command buffer submission
  cadence, and visible gaps.
- Browser-side CPU work: renderer main thread, V8, event loop, fetch/parse,
  shader/pipeline creation, and page scheduling.
- GPU-process backpressure and compositor/Metal interaction.
- Whether measured wall time is polluted by model loading, shader compilation,
  pipeline creation, readbacks, maps, or background throttling.

### What Chrome tracing cannot prove

- Shader memory bandwidth.
- Occupancy.
- Register pressure.
- Cache hit/miss behavior.
- ALU limiter versus memory limiter.
- Threadgroup-memory stalls.
- Per-WGSL-line cost.
- Correct visual quality or convergence.

Use Chrome tracing for browser/Dawn scheduling questions. Do not use it as the
primary kernel profiler.

## 3. Dawn Toggles And Dawn Capture

Dawn exposes Chrome command-line toggles:

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" \
  --user-data-dir=/tmp/nffa_dawn_profile \
  --use-angle=metal \
  --ignore-gpu-blocklist \
  --enable-webgpu-developer-features \
  --enable-unsafe-webgpu \
  --enable-dawn-features=dump_shaders,disable_symbol_renaming,use_user_defined_labels_in_backend \
  --enable-logging=stderr \
  --v=1 \
  http://localhost:8799/dist/splat3d.html
```

Useful toggles for this project:

- `dump_shaders`: log input WGSL and translated backend shaders such as MSL.
- `disable_symbol_renaming`: make dumped shader text easier to read.
- `use_user_defined_labels_in_backend`: forward labels to native debugging
  tools, useful if Metal capture/trace can see them.

What this can prove:

- Whether a WGSL emitter produced the shader shape we expected.
- Whether Tint/Dawn translation is creating surprising MSL.
- Whether labels survive far enough to help native tooling.
- Whether a failure is validation/codegen-related rather than runtime timing.

What this cannot prove:

- Any bottleneck by itself.
- Real elapsed GPU time.
- Memory/ALU/occupancy cause.
- Browser scheduling gaps, unless combined with Chrome tracing.

Dawn also documents a native GPU API tracing path:

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MTL_CAPTURE_ENABLED=1 \
  DAWN_TRACE_FILE_BASE=/tmp/nffa_dawn_metal_capture \
  "$CHROME" \
  --user-data-dir=/tmp/nffa_dawn_capture_profile \
  --use-angle=metal \
  --ignore-gpu-blocklist \
  --enable-webgpu-developer-features \
  --enable-unsafe-webgpu \
  --enable-dawn-features=use_user_defined_labels_in_backend \
  http://localhost:8799/dist/splat3d.html
```

Expected result if the Chrome/Dawn/Metal path honors the environment variables:
a `.gputrace` file loadable in Xcode's Metal Debugger. This is not currently a
complete local workflow because Xcode/Metal Debugger is not available through
the selected developer tools. Dawn's own note also says native API tracing is
backend-dependent; Metal is implemented, but DirectX/OpenGL backends are not.

## 4. Chrome Backend Sanity

For any browser benchmark, first validate the backend:

```text
chrome://gpu
```

Record:

- WebGPU enabled.
- ANGLE/backend is Metal.
- Adapter is real Apple GPU, not SwiftShader or fallback.
- No WebGPU/Dawn/Metal device-loss or blocklist warnings.

Repo smoke paths:

```bash
node tools/smoke.mjs http://localhost:1234/splat3d.html 8000 /tmp
node tools/splat/page_smoke.mjs http://localhost:1234/splat.html 8000 /tmp
```

These prove the page boots/renders and expose console errors. They are not
profilers. Headless Chrome may fail to get a real WebGPU adapter; that only
proves headless adapter availability, not app performance.

## 5. Xcode Metal System Trace

### Current state

Blocked on this machine until full Xcode is installed and selected:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun --find xctrace
xcrun xctrace list templates
```

After Xcode is available, check for Metal-related templates:

```bash
xcrun xctrace list templates
```

If the template exists, capture Chrome across all processes because WebGPU work
spans browser, renderer, GPU process, compositor, and helpers:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --all-processes \
  --time-limit 20s \
  --output /tmp/nffa_chrome_metal_system.trace
open /tmp/nffa_chrome_metal_system.trace
```

During the 20-second window: leave the page idle for 2 seconds, run optimize for
5-10 seconds, then idle for 2 seconds.

For a cleaner deterministic target, trace a Bun/WebGPU benchmark directly:

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
open /tmp/nffa_bun_metal_system.trace
```

### What Metal System Trace can prove

- CPU/GPU timeline shape.
- Whether command buffers are back-to-back or separated by idle gaps.
- Whether Chrome/Bun is CPU encode-bound before the GPU gets work.
- Queue depth, command buffer duration, and over-serialization.
- Whether readbacks/maps or resource allocation cause visible stalls.
- Performance state/thermal noise across the run if the template exposes it.
- Memory usage timeline and large resource behavior.

### What Metal System Trace may not prove by itself

- Exact WGSL source line responsible for cost.
- Clear CLIP layer names, unless labels propagate through Dawn and Chrome.
- Memory-bound versus ALU-bound cause without counter tracks enabled.
- Fine per-dispatch attribution inside a long WebGPU command buffer if labels
  are missing.

Use it to answer "is the system or queue schedule the bottleneck?" Then pair it
with WebGPU timestamps for pass identity.

## 6. GPU Counters And Metal Debugger

### Current state

Not available through the selected tools today:

```bash
xcrun --find gpudebug
xcrun --find gpucapture
xcrun --find metalperftrace
```

After full Xcode is installed, check again:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun --find gpudebug
xcrun --find gpucapture
xcrun --find metalperftrace
```

Apple's current Metal tools page describes:

- Metal Debugger in Xcode for compute/render/ML pipeline inspection.
- Metal System Trace in Instruments for CPU/GPU/memory timelines.
- Hardware counters in Xcode performance timelines.
- Shader execution heat maps and shader cost views.
- `gpucapture`, `gpudebug`, and `metalperftrace` command-line tools for scripts
  and agent workflows.

### `metalperftrace` when available

Apple's 2026 tooling examples show look-back collection and textual summaries:

```bash
metalperftrace collect /tmp --last 5h
metalperftrace collect /tmp --start 2026-04-01T09:41:00 --end 2026-04-01T12:41:00
metalperftrace overview /tmp/MetalPerfTrace_20260401_094100_to_144100.atrc
metalperftrace overview /tmp/MetalPerfTrace_20260401_094100_to_144100.atrc --json > /tmp/nffa_metalperftrace.json
```

Use this for long-running browser/app telemetry, frame-time/GPU-time summaries,
resource usage, shader compilation totals, and scripted regression checks. It
is not a replacement for per-dispatch WebGPU timestamps or for shader-counter
capture when the question is "why is this WGSL kernel slow?"

### What counters can prove if Xcode exposes them for this workload

Memory-bound pointwise or raster:

- High buffer/device memory read or write limiter.
- High memory bandwidth utilization.
- High last-level-cache or L1 eviction/stall indicators.
- Low/moderate ALU limiter while memory limiter is high.
- f16/storage/layout changes reduce both bytes and timestamps.

ALU-bound pointwise:

- High ALU/fp32/fp16 limiter or utilization.
- Memory bandwidth below saturation.
- Runtime scales with arithmetic count.
- f16 storage alone does not help unless arithmetic type also changes.

Occupancy/register/threadgroup-memory limited:

- Low occupancy while ALU and memory are not saturated.
- Occupancy manager target below full, or launch stalls from on-chip resource
  pressure.
- Threadgroup memory or register/L1 residency/eviction indicators point to
  resource exhaustion.
- Smaller tiles or lower live value count improves timestamp and occupancy.

Launch/dispatch-size limited:

- Low compute shader launch activity or poor occupancy due to too little work.
- Short dispatches separated by CPU or queue gaps.
- Fusion or batching improves timestamps mostly by removing launch overhead.

Barrier/synchronization limited:

- Barrier/threadgroup synchronization stalls.
- Queue gaps at inter-dispatch boundaries.
- Removing barriers or changing tile shape improves timestamps without changing
  bandwidth much.

Browser/system limited:

- Chrome trace and Metal System Trace show CPU encode, pipeline creation,
  command submit, map/readback, compositor, or backgrounding gaps larger than
  shader execution.
- WebGPU timestamp sums are much lower than wall time and the missing time is
  visible outside GPU dispatch execution.

### What counters still may not prove

- A portable WebGPU rule. Counter names and meanings vary by Xcode version and
  Apple GPU family.
- Source-level causality if Chrome/Dawn strips labels or Xcode cannot map
  translated MSL back to WGSL.
- A production-browser result if the capture uses debug flags that materially
  change compilation, validation, or timing.

## 7. Practical Bottleneck Decision Table

| Question | First tool | Confirm with | Can prove | Cannot prove |
| --- | --- | --- | --- | --- |
| Which optimizer segment dominates? | `TIMESTAMP=1 ... step_bench.ts` | `step_matrix.ts` | Raster/CLIP/Adam/display GPU ms | Hardware cause |
| Which CLIP kernel family dominates? | `TIMESTAMP=1 ... dispatch_profile.ts` | Integrated step bench | Per-dispatch/family GPU ms | Memory vs ALU |
| Is the browser/Dawn schedule adding gaps? | `tools/webgpu_trace.mjs` | Metal System Trace if available | Queue/submit/pipeline/browser gaps | Shader hardware cause |
| Is Chrome using Metal WebGPU? | `chrome://gpu`, trace helper status | Smoke screenshot/console | Backend/path sanity | Performance bottleneck |
| Is a WGSL rewrite producing expected MSL? | Dawn `dump_shaders` | Xcode capture if available | Translation/codegen shape | Runtime bottleneck |
| Is the workload memory-bound? | Xcode counters | Timestamp before/after byte-reducing fork | Bandwidth/cache limiter evidence | Available today without Xcode |
| Is the workload occupancy/register limited? | Xcode counters/Metal Debugger | Timestamp before/after tile/live-value changes | Occupancy/resource evidence | Available today without Xcode |
| Is the workload CPU/browser-bound? | Chrome trace | Metal System Trace | Renderer/GPU-process/submit gaps | WGSL shader line cost |

## 8. Recommended Profiling Order

1. Run timestamp baselines in Bun:

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts > /tmp/nffa_clip_b3_dispatch_ts.csv
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_LAYOUT=grid9_close2 GRID_DIRECT_RASTER=1 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

2. Capture one Chrome trace on the exact browser path:

```bash
npx parcel build --no-scope-hoist --no-cache --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
GRID9=1 DIRECT_GRID=1 SETUP_TIMEOUT_MS=180000 node tools/webgpu_trace.mjs http://localhost:8799/dist/splat3d.html 12000 /tmp/nffa_trace
```

3. Only after those say shader work is still the bottleneck, install/select full
   Xcode and run Metal System Trace/counters:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun xctrace list templates
xcrun --find gpudebug
xcrun --find metalperftrace
```

4. For any shader fork, require:

- correctness/parity gate;
- isolated `TIMESTAMP=1` dispatch improvement;
- integrated `TIMESTAMP=1 ... step_bench.ts` improvement;
- wall-clock `step_matrix.ts` improvement;
- Chrome trace check only if browser scheduling could explain a mismatch.

## 9. Source Notes

Local repo files inspected:

- `tools/webgpu_trace.mjs`
- `docs/SPLAT3D_PERF_NOTES.md`
- `experiments/clip_forks/v12_chrome_dawn_trace/README.md`
- `experiments/clip_forks/v05_gpu_counter_trace/README.md`
- `agent_notes/optimization_session/agent_gpu_profiler_plan.md`
- `agent_notes/optimization_session/agent_real_gpu_tools_2026_07_08.md`
- `agent_notes/optimization_session/integrated_step_timestamp_profile.md`
- `agent_notes/optimization_session/clip_timestamp_dispatch_profile.md`

Primary/current references checked:

- Chrome WebGPU developer features:
  https://developer.chrome.com/docs/web-platform/webgpu/developer-features
- Dawn debugging/toggles/native capture:
  https://dawn.googlesource.com/dawn/+/HEAD/docs/dawn/debugging.md
- Chromium trace recording guidance:
  https://chromium.googlesource.com/playground/chromium-org-site/+/refs/heads/main/developers/how-tos/trace-event-profiling-tool/recording-tracing-runs/index.md
- Chromium built-in trace categories:
  https://chromium.googlesource.com/chromium/src/+/main/base/trace_event/builtin_categories.h
- Apple Metal developer tools:
  https://developer.apple.com/metal/tools/
- Apple GPU counters overview:
  https://developer.apple.com/videos/play/wwdc2020/10603/
- Apple `metalperftrace` workflow:
  https://developer.apple.com/videos/play/wwdc2026/388/
- Apple shader/occupancy profiling tools:
  https://developer.apple.com/videos/play/tech-talks/111374/

## Bottom Line

Right now, the defensible workflow is:

1. WebGPU timestamps for pass and dispatch timing.
2. Chrome trace for browser/Dawn scheduling.
3. Dawn toggles for shader dumps/labels/capture setup.
4. Full Xcode later for Metal System Trace and counters.

Until full Xcode/counters are available, we can rank bottlenecks by elapsed GPU
time, but we cannot honestly claim memory-bound, occupancy-bound,
register-bound, or ALU-bound root cause.
