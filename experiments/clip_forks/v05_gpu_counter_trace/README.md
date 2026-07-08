# v05 GPU Counter Trace

Date: 2026-07-08

Scope: profiling fork note only. This fork does not change CLIP, raster, or
optimizer source code. Its job is to prove why the hot shaders are slow before
we spend more time on f16 storage or pointwise tiling rewrites.

## Hypothesis

Current timestamps already identify the hot regions:

- integrated 3D step: CLIP batch dominates the optimizer step;
- isolated CLIP dispatch profile: pointwise forward/backward is the largest
  CLIP family, with `pw_bwd`, `pw+gelu`, and `pw` around 55% of isolated B=3
  train timestamp time;
- raster backward is real but secondary for the current default sample;
- attention backward is visible but too small to be the first 2-4x lever.

What timestamps do not prove is the cause. The pointwise kernels may be limited
by global memory bandwidth, threadgroup-memory/barrier overhead, occupancy,
register pressure, ALU throughput, dispatch overhead, or browser/Dawn queue
gaps. This fork exists to separate those cases.

## Commands Available Now

Run GPU benchmarks sequentially. Do not run several WebGPU benches in parallel;
that pollutes the queue and thermals.

### 1. Integrated optimizer timestamps

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=10 WARMUP=5 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Evidence to record:

- `rasterFwd`, `rasterBwd`, `clipBatch`, `adam`, and `display` milliseconds;
- whether 9-view mode changes the bottleneck ordering;
- normal step average, because that remains the promotion gate.

### 2. Isolated CLIP dispatch timestamps

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/dispatch_profile.ts
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/dispatch_profile.ts > /tmp/nffa_clip_b3_dispatch_ts.csv
```

Evidence to record:

- grouped timing for `pw_bwd`, `pw+gelu`, `pw`, `spatial_bwd`, and `conv`;
- top individual dispatches by time;
- variance across runs;
- whether the same pointwise layers dominate after any v02/v04 experiment.

### 3. Chrome trace for browser/Dawn scheduling

Build and serve:

```bash
npx parcel build --no-scope-hoist --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
```

Launch Chrome with startup tracing:

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

Then open `/tmp/nffa_webgpu_trace.json` in `chrome://tracing` or
`https://ui.perfetto.dev`.

Evidence to record:

- renderer-thread and GPU-process gaps between command submissions;
- pipeline creation, shader compilation, or cache misses during measured runs;
- queue idle periods between CLIP dispatch groups;
- map/readback stalls;
- whether browser scheduling, not shader execution, explains wall time.

### 4. Chrome backend sanity check

Open:

```text
chrome://gpu
```

Record:

- WebGPU enabled;
- ANGLE/backend is Metal;
- real Apple GPU path, not SwiftShader;
- any WebGPU/Dawn/Metal warnings.

## Blocked Without Full Xcode

The current command-line developer tools are not enough for true Metal GPU
counters. Full Xcode must be installed and selected before this fork can answer
memory-bound versus ALU-bound with hardware evidence.

Setup check:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun xctrace list templates | rg 'Metal|GPU'
xcrun --find gpudebug
xcrun --find metalperftrace
```

If those fail, the blocked items are:

- Metal System Trace via Instruments / `xctrace`;
- Xcode GPU capture;
- shader counter tables;
- occupancy, SIMD utilization, bandwidth, cache, stall, and threadgroup-memory
  metrics;
- `metalperftrace` summaries, if the installed Xcode version provides it.

WebGPU timestamps and Chrome traces can still be collected without full Xcode,
but they only prove duration and scheduling, not hardware bottleneck cause.

## Xcode / Metal System Trace Workflow

Record the full browser session across Chrome's multi-process WebGPU stack:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --all-processes \
  --time-limit 20s \
  --output /tmp/nffa_chrome_metal_system.trace
```

During the 20-second window:

1. leave the splat page idle for about 2 seconds;
2. run the optimizer for 5-10 seconds;
3. leave it idle again for about 2 seconds.

For a simpler target, capture Bun/WebGPU directly:

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

Use the Bun capture first if Chrome capture is noisy. The process is smaller,
the benchmark is deterministic, and the timestamp output can be matched against
the Metal timeline.

## Counters To Inspect

Exact counter names vary by Xcode and Apple GPU generation. Capture the names
shown by the installed tools and map them into these buckets.

### Timeline and queue behavior

- GPU busy / idle percentage;
- command buffer duration;
- gaps between command buffers;
- CPU encode time;
- queue wait / submission delay;
- pipeline creation or shader compilation events;
- buffer map/readback stalls.

### Memory and cache behavior

- device/global memory read bytes and write bytes;
- bandwidth utilization versus peak;
- texture/storage-buffer cache hit rate, if exposed;
- load/store unit utilization;
- threadgroup-memory traffic;
- threadgroup-memory bank conflict or stall metrics, if exposed;
- memory dependency stalls.

### ALU / SIMD behavior

- arithmetic or ALU utilization;
- SIMD occupancy / active lanes;
- instruction issue rate;
- fp32/fp16 pipeline utilization, if separated;
- cycles stalled on execution dependencies.

### Occupancy and register pressure

- active threadgroups per compute unit;
- active SIMD groups / waves;
- thread execution width and utilization;
- registers/thread or register pressure warning;
- threadgroup memory per dispatch;
- occupancy limited by registers versus threadgroup memory.

### Barrier and synchronization

- barrier stall time;
- threadgroup synchronization overhead;
- inter-dispatch queue gaps;
- memory fence or dependency stalls.

## What Would Prove Each Bottleneck

### Pointwise is memory-bound

Evidence:

- `pw_bwd`, `pw+gelu`, and `pw` have high memory bandwidth utilization while
  ALU utilization is moderate or low;
- large storage-buffer read traffic from activation and weight buffers;
- cache hit rate is poor or load/store unit utilization is near saturation;
- f16 weights in v02 reduce bytes and timestamp time without changing much ALU
  structure;
- larger pointwise tiles in v04 do not help unless they reduce memory traffic.

Optimization response:

- prioritize v02 f16 weights first;
- then consider selective f16 activation/gradient slots;
- evaluate layouts that improve weight/activation reuse;
- avoid adding more arithmetic to save small dispatch overhead.

### Pointwise is ALU-bound

Evidence:

- high arithmetic/SIMD utilization;
- memory bandwidth clearly below saturation;
- dispatch duration scales with MAC count;
- f16 weights alone in v02 reduces payload but barely changes timestamps;
- v04 tiling changes help only if they improve SIMD utilization or instruction
  scheduling.

Optimization response:

- v02 weights-only is mostly a size win, not a speed win;
- v04 should focus on higher arithmetic throughput: tile shape, vectorization,
  subgroups, fewer scalar conversions, better unrolling, and lower overhead per
  output element;
- reducing CLIP call count, grid CLIP, or proxy guidance becomes the larger
  wall-time lever.

### Pointwise is occupancy/register limited

Evidence:

- memory and ALU utilization are both below peak;
- low active threadgroups/SIMD groups;
- register pressure or threadgroup memory limits occupancy;
- barrier stalls are visible around the staged `xS`/`wS` loops;
- v04 shared-memory variants slow down or flatten because they reduce occupancy.

Optimization response:

- test smaller workgroup-memory tiles;
- reduce per-thread accumulator count;
- split oversized kernels if register pressure dominates;
- avoid larger shared-W variants unless counters show weight-load reuse wins
  more than occupancy loss.

### Pointwise is dispatch/queue overhead dominated

Evidence:

- individual shader counters look healthy, but Chrome/Metal traces show gaps
  between many short dispatches;
- CPU encode/submit time is large relative to GPU work;
- pipeline creation or command buffer churn appears inside warm runs;
- fused producer/consumer gates improve wall time more than their raw shader
  compute share suggests.

Optimization response:

- prioritize v03 residual-bwd plus pointwise-bwd fusion and other legal local
  fusions;
- reduce command buffer boundaries and readbacks;
- batch more views only when it does not increase shader time enough to erase
  submit savings.

### Spatial backward is the real hidden problem

Evidence:

- `spatial_bwd` has worse memory stalls or occupancy than pointwise;
- top individual `spatial_bwd` dispatches show pathological cache behavior;
- pointwise counters are healthy but `spatial_bwd` dominates integrated CLIP
  after v02/v04 changes.

Optimization response:

- add shape-specific spatial-bwd forks after the stem specialization;
- avoid spending the whole session on pointwise if counters show spatial
  backward is now the worse kernel family.

## How This Feeds v02 and v04

### v02 f16 weights

Use this fork to decide whether f16 weights are a speed optimization or only a
payload/memory optimization.

Before v02:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=10 WARMUP=5 bun tools/splat3d/step_bench.ts
```

After v02:

```bash
PRECISION=f16 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/dispatch_profile.ts
CLIP_PRECISION=f16 TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=10 WARMUP=5 bun tools/splat3d/step_bench.ts
```

Promote v02 for speed only if:

- pointwise-family timestamp time falls materially;
- integrated `clipBatch` and normal step average improve;
- correctness gates still pass;
- counters support a bandwidth or cache-traffic explanation.

If counters show ALU-bound pointwise and v02 barely moves timestamps, keep f16
weights as an asset-size/browser-memory option rather than calling it a 2-4x
speed path.

### v04 pointwise tile rewrite

Use this fork to choose the rewrite direction.

If memory-bound:

- improve global-load coalescing and reuse;
- reduce repeated weight reads;
- consider f16 storage;
- measure bytes per output element.

If ALU-bound:

- tune arithmetic throughput;
- test subgroup/tile shapes;
- reduce scalar indexing overhead;
- improve instruction scheduling and unrolling.

If occupancy-bound:

- reduce threadgroup memory and registers per invocation;
- test smaller tiles before larger shared-W tiles;
- reject variants that lower occupancy even if they reduce load count.

Every v04 variant should record:

```bash
TIMESTAMP=1 MODE=train BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=10 WARMUP=5 bun tools/splat3d/step_bench.ts
```

And, once Xcode is available, one Metal counter capture against the top
pointwise dispatch family before and after the rewrite.

## Artifacts To Save

Save profiling outputs here or in `agent_notes/optimization_session/` with a
link back to this fork:

- `/tmp/nffa_clip_b3_dispatch_ts.csv` copied to a dated CSV;
- integrated `step_bench.ts` output for B=3 and B=9;
- `/tmp/nffa_webgpu_trace.json` or a compressed copy;
- `/tmp/nffa_bun_metal_system.trace` summary or screenshots;
- Metal counter table screenshots for the hottest `pw_bwd`, `pw+gelu`, `pw`,
  and `spatial_bwd` dispatches;
- a short conclusion: memory-bound, ALU-bound, occupancy-bound, queue-bound, or
  still inconclusive.

## Key Recommendation

Do not start another pointwise shader rewrite blind. The next serious speed
session should collect:

1. current WebGPU timestamp baselines;
2. one Chrome trace to rule out browser/Dawn queue gaps;
3. one full-Xcode Metal System Trace or GPU counter capture.

Then choose:

- v02 if counters show memory bandwidth or weight traffic is the limiter;
- v04 if counters show ALU throughput, occupancy, register pressure, or
  threadgroup-memory behavior is the limiter;
- algorithmic CLIP-call reduction if counters show the kernels are already
  healthy and the bottleneck is simply too much same-resolution CLIP work.
