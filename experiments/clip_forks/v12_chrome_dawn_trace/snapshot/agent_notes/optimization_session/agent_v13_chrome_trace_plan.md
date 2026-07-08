# Agent v13 Chrome/Dawn Trace Capture Plan

Date: 2026-07-08

Scope: inspect the repo's current WebGPU profiling tools/docs and define a
concrete Chrome/Dawn trace workflow for Neural Force Field Art. This note is
documentation only. Do not edit app source as part of this trace-planning pass.

## Local Context Checked

- Repo state at inspection: clean worktree, HEAD `5e7da8a`.
- `RTK.md` is referenced by `AGENTS.md` but is not present in this checkout.
- Installed Chrome probed through CDP:
  `Google Chrome 150.0.7871.46`.
- `Tracing.getCategories` on this Chrome advertises the WebGPU-relevant
  categories:
  - `disabled-by-default-gpu.dawn`
  - `disabled-by-default-gpu.graphite.dawn`
  - `disabled-by-default-webgpu`
  - `gpu`
  - `gpu.angle`
  - `gpu.capture`
  - `disabled-by-default-skia.gpu`
  - `disabled-by-default-viz.gpu_composite_time`
  - plus renderer categories such as `devtools.timeline`, `blink`,
    `renderer.scheduler`, `disabled-by-default-renderer.scheduler`, `v8`,
    and `benchmark`.
- `dawn` and `disabled-by-default-gpu.service` are used in older notes, but
  they are not advertised by this Chrome build. Treat them as legacy/optional,
  not required categories.

## Existing Repo Profilers To Pair With Chrome Trace

Chrome trace is not the numeric source of truth for shader duration in this
repo. Use it beside the existing WebGPU timestamp tools:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_clip_b3_dispatch_ts.csv
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
```

Relevant local files:

- `tools/clip/dispatch_profile.ts`: per-dispatch CLIP timings, optionally true
  WebGPU timestamp-query timing.
- `tools/splat3d/step_bench.ts`: integrated optimizer phase timing, optionally
  timestamped on real passes.
- `tools/splat3d/step_matrix.ts`: sequential A/B promotion gate; avoids
  parallel GPU contention.
- `src/render/webgpu/gputime.ts`: app-side timestamp-query ring buffer for the
  particle-art path.
- `tools/qa_browser.mjs`: real-GPU browser smoke driver using Metal.
- `tools/smoke.mjs`: CI-style smoke driver that forces SwiftShader fallback.
  Do not use this for performance traces.
- `tools/splat/page_smoke.mjs`: real headless-Metal smoke for `splat.html`.

## Trace Target Choices

For the original particle-art page:

```bash
yarn start
# open or trace:
# http://localhost:1234/index.html
```

For the 3D CLIP/splat optimizer, prefer a production build served from the repo
root so the large local model artifacts are same-origin and cacheable:

```bash
npx parcel build --no-scope-hoist --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
# open or trace:
# http://localhost:8799/dist/splat3d.html
```

If `models/mobileclip_s0/plan_train.json` or `weights_train.bin` are missing,
follow the `tools/clip/README.md` model compilation flow first. Do not capture
trace windows dominated by first-time model generation or downloads unless the
question is cold-start behavior.

## Category Set

Use this default category set for Chrome/Dawn WebGPU traces on the current
machine:

```text
gpu,
gpu.angle,
disabled-by-default-gpu.dawn,
disabled-by-default-gpu.graphite.dawn,
disabled-by-default-webgpu,
disabled-by-default-skia.gpu,
disabled-by-default-viz.gpu_composite_time,
devtools.timeline,
disabled-by-default-devtools.timeline,
blink,
blink.user_timing,
renderer.scheduler,
disabled-by-default-renderer.scheduler,
v8,
v8.execute,
benchmark
```

Optional heavy categories:

- `disabled-by-default-devtools.timeline.stack` if JS stack attribution is more
  important than trace size.
- `disabled-by-default-devtools.screenshot` for visual timeline screenshots.
- `gpu.capture` only for short experiments; it may increase trace volume.

Legacy categories to include only if a future Chrome advertises them:

- `dawn`
- `disabled-by-default-gpu.service`

Probe supported categories before a serious session:

```bash
node --input-type=module - <<'EOF'
import puppeteer from 'puppeteer';
const browser = await puppeteer.launch({
  headless: 'new',
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--no-sandbox', '--use-angle=metal', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'],
});
const page = await browser.newPage();
const client = await page.createCDPSession();
const { categories } = await client.send('Tracing.getCategories');
for (const c of categories.filter((c) => /gpu|dawn|webgpu|devtools\.timeline|blink|renderer\.scheduler|v8|benchmark/i.test(c)).sort()) {
  console.log(c);
}
await browser.close();
EOF
```

## Workflow A: Startup Trace For Cold Browser/WebGPU Work

Use this when the question is adapter creation, device setup, tfjs WebGPU
backend initialization, shader/pipeline JIT, or first model/page boot.

```bash
OUT=/tmp/nffa_webgpu_startup_trace.json
PROFILE=/tmp/nffa_trace_profile
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
URL="http://localhost:8799/dist/splat3d.html"

rm -rf "$PROFILE"
"$CHROME" \
  --user-data-dir="$PROFILE" \
  --use-angle=metal \
  --ignore-gpu-blocklist \
  --enable-webgpu-developer-features \
  --disable-background-timer-throttling \
  --disable-backgrounding-occluded-windows \
  --disable-renderer-backgrounding \
  --trace-startup='gpu,gpu.angle,disabled-by-default-gpu.dawn,disabled-by-default-gpu.graphite.dawn,disabled-by-default-webgpu,disabled-by-default-skia.gpu,disabled-by-default-viz.gpu_composite_time,devtools.timeline,disabled-by-default-devtools.timeline,blink,blink.user_timing,renderer.scheduler,disabled-by-default-renderer.scheduler,v8,v8.execute,benchmark' \
  --trace-startup-duration=30 \
  --trace-startup-file="$OUT" \
  --trace-startup-format=json \
  "$URL"
```

Capture shape:

1. Start the static server or dev server first.
2. Launch Chrome with the command above.
3. Let the page boot, then run one short, known interaction window.
4. Close Chrome after the trace duration or after the output file stops
   growing.
5. Open the trace in `chrome://tracing` or `https://ui.perfetto.dev`.

Use `chrome://gpu` in the same profile to confirm WebGPU is enabled and the
path is not SwiftShader/fallback when the trace is meant to represent real
hardware.

## Workflow B: Narrow Warm Trace Through Puppeteer/CDP

Use this when the question is steady-state browser scheduling or queue gaps
after warmup. This is the best unattended capture shape. It avoids the
SwiftShader override in `tools/smoke.mjs`.

```bash
URL="http://localhost:8799/dist/splat3d.html" OUT="/tmp/nffa_webgpu_warm_trace.json" node --input-type=module - <<'EOF'
import fs from 'node:fs';
import puppeteer from 'puppeteer';

const url = process.env.URL ?? 'http://localhost:1234/index.html';
const out = process.env.OUT ?? '/tmp/nffa_webgpu_warm_trace.json';
const captureMs = Number(process.env.CAPTURE_MS ?? 12000);
fs.rmSync(out, { force: true });

const browser = await puppeteer.launch({
  headless: false,
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  userDataDir: '/tmp/nffa_cdp_trace_profile',
  args: [
    '--no-sandbox',
    '--use-angle=metal',
    '--ignore-gpu-blocklist',
    '--enable-webgpu-developer-features',
    '--disable-background-timer-throttling',
    '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding',
  ],
});

const page = await browser.newPage();
page.on('console', (m) => console.log(`[console:${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

const probe = await page.evaluate(async () => {
  const adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
  return {
    webgpu: !!navigator.gpu,
    adapter: adapter ? `${adapter.info?.vendor ?? '?'} ${adapter.info?.architecture ?? '?'}` : null,
    fallback: adapter?.info?.isFallbackAdapter ?? null,
  };
});
console.log('PROBE', JSON.stringify(probe));

// Warm page/pipelines before tracing. Adjust for the target page.
await new Promise((r) => setTimeout(r, 8000));

await page.tracing.start({
  path: out,
  categories: [
    'gpu',
    'gpu.angle',
    'disabled-by-default-gpu.dawn',
    'disabled-by-default-gpu.graphite.dawn',
    'disabled-by-default-webgpu',
    'disabled-by-default-skia.gpu',
    'disabled-by-default-viz.gpu_composite_time',
    'devtools.timeline',
    'disabled-by-default-devtools.timeline',
    'blink',
    'blink.user_timing',
    'renderer.scheduler',
    'disabled-by-default-renderer.scheduler',
    'v8',
    'v8.execute',
    'benchmark',
  ],
});

// Optional page-specific trigger. Keep this explicit per investigation.
await new Promise((r) => setTimeout(r, captureMs));

await page.tracing.stop();
await browser.close();

const stat = fs.statSync(out);
console.log(`TRACE ${out} ${stat.size} bytes`);
EOF
```

Headless mode is acceptable for collecting browser event shape, but do not trust
it for real hardware WebGPU timing unless the `PROBE` and `chrome://gpu` checks
prove the adapter is not fallback. Prior notes observed headless Chrome can be
useful for traces while still being unreliable as a hardware timing path on this
Mac. Use Bun/WebGPU timestamp queries for elapsed GPU milliseconds.

## Workflow C: Manual Trace Window

Use manual tracing when a human needs to interact with a control before or
during capture.

1. Start the target URL in a clean Chrome profile with the same launch flags as
   Workflow A, but without `--trace-startup`.
2. Open `chrome://tracing` in a second tab.
3. Record with the category set above.
4. Switch to the app tab.
5. Run 2 seconds idle, 5-10 seconds of the optimizer/animation window, then
   2 seconds idle.
6. Stop, save JSON, and inspect in `chrome://tracing` or Perfetto.

Keep only one target tab open. Other tabs and extensions contaminate the
browser/GPU-process tracks.

## Expected Trace Signals

Look for these tracks/events:

- Renderer main thread: long JS tasks, rAF cadence, event handlers, tfjs setup,
  fetch/decode, and layout/compositor work.
- GPU process: WebGPU/Dawn events under the GPU process, command buffer/queue
  activity, device or pipeline events, and suspicious idle gaps.
- Dawn/WebGPU categories: API/device/queue/pipeline/command events when emitted
  by Chrome. Event names are Chrome/Dawn internals, not guaranteed WGSL labels.
- V8/timeline: pipeline-building JS, tfjs initialization, model loading, and
  any hot main-thread work that overlaps the measured GPU window.
- Scheduler categories: page throttling, backgrounding, task starvation, and
  cross-thread delays.
- Skia/viz GPU categories: compositor work that can explain display-frame
  stalls, especially on the particle renderer path.

Use the app HUD and console messages as anchors. For the original particle
page, `src/main.ts` logs whether timestamp-query is active and whether the HUD
shows GPU or CPU-encode times. For the CLIP/splat pages, correlate trace windows
with `step_bench` and `dispatch_profile` rather than trying to infer per-kernel
times from Chrome trace alone.

## Questions Chrome/Dawn Trace Can Answer

- Does startup time include adapter/device setup, shader compilation, pipeline
  creation, model fetch, or tfjs WebGPU backend initialization?
- Are steady-state frames interrupted by long renderer-main-thread tasks?
- Is the GPU process showing gaps between command submission windows?
- Is the app accidentally timing cold Metal/Dawn JIT rather than warmed
  pipelines?
- Is there evidence of readback/map/onSubmittedWorkDone synchronization stalling
  the browser thread?
- Is the page being throttled or backgrounded during the measurement window?
- Are trace-time browser events consistent with the existing Bun timestamp
  conclusion that CLIP dominates the 3D optimizer?
- Is a page using the expected real adapter, or did it fall back to SwiftShader?

## Questions It Cannot Answer

- Whether `pw_bwd`, `spatial_bwd`, or another WGSL shader is memory-bandwidth
  bound, occupancy-bound, cache-limited, or register-spilling.
- Per-shader hardware counters, cache misses, SIMD occupancy, or ALU utilization.
- Reliable per-dispatch GPU elapsed milliseconds for the full optimizer. Use
  timestamp queries in `tools/clip/dispatch_profile.ts` and
  `tools/splat3d/step_bench.ts`.
- WGSL source-line stall attribution. Chrome/Dawn may expose translated or
  browser-internal labels, not the original shader source structure.
- Cross-run performance claims by itself. Trace overhead and background GPU
  contention can perturb results.

For those missing answers, the prior tool notes still point to full Xcode/Metal
System Trace, Xcode GPU capture/counters, or `metalperftrace` after full Xcode
is installed and selected.

## Trace Hygiene

- Warm pipelines before a steady-state trace; Metal/Dawn lazy JIT makes cold
  runs misleading.
- Keep capture windows short. The disabled-by-default categories are verbose and
  can wrap buffers or create large files.
- Do not run Bun benches, browser traces, and other GPU workloads in parallel.
- Use a clean Chrome profile under `/tmp`.
- Disable background timer throttling with launch flags when measuring an
  animation/optimizer loop.
- Avoid `tools/smoke.mjs` for performance. It intentionally patches
  `requestAdapter({ forceFallbackAdapter: true })` for headless verification.
- Save raw traces to `/tmp` unless a trace is small and worth archiving.
  If archiving, store a short note or summary in `agent_notes/optimization_session/`,
  not a huge JSON trace.

## Simple Helper Script Worth Adding Later

A useful future helper would be `tools/chrome_dawn_trace.mjs`, but this session
does not create it.

It should:

- accept `URL`, `OUT`, `CAPTURE_MS`, `WARMUP_MS`, `HEADFUL`, and optional
  `ACTION` env vars;
- launch Chrome with the real-GPU flags used by `tools/qa_browser.mjs`, never
  the SwiftShader override from `tools/smoke.mjs`;
- call `Tracing.getCategories`, filter the requested category set to supported
  categories, and print unsupported category names;
- probe `navigator.gpu.requestAdapter().info` and fail loudly if
  `isFallbackAdapter` is true for a hardware trace;
- capture console/page errors beside the trace path;
- optionally click page-specific controls, such as the `splat3d` optimize
  control, via a tiny action registry;
- print `TRACE <path> <bytes>` plus the adapter probe and last console lines.

That helper would make Chrome trace collection repeatable without changing app
source. It should still be paired with the Bun timestamp tools for quantitative
GPU timing.

## References

Local:

- `agent_notes/optimization_session/agent_gpu_profiler_plan.md`
- `agent_notes/optimization_session/agent_real_gpu_tools_2026_07_08.md`
- `docs/SPLAT3D_PERF_NOTES.md`
- `tools/clip/README.md`
- `tools/clip/dispatch_profile.ts`
- `tools/splat3d/step_bench.ts`
- `tools/qa_browser.mjs`
- `tools/smoke.mjs`
- `tools/splat/page_smoke.mjs`
- `src/render/webgpu/gputime.ts`

External:

- Chromium trace event profiling overview:
  https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/
- Chrome DevTools Protocol Tracing domain:
  https://chromedevtools.github.io/devtools-protocol/tot/Tracing/
- Dawn overview:
  https://dawn.googlesource.com/dawn
