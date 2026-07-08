# Real GPU Profiling Paths On This Mac

Date: 2026-07-08

Scope: local profiling/tooling audit for
`/Users/nicholasbardy/git/Neural-Force-Field-Art`. This note is intentionally
tooling-only. I did not install anything and did not edit source code.

## Executive Read

We can profile meaningful GPU time today, but not true shader hardware counters
today.

Available now:

- Repo Bun/WebGPU benches through `bun-webgpu`.
- WebGPU timestamp queries on the Apple M4/Metal adapter.
- Chrome/Puppeteer trace capture through DevTools Protocol.
- Chrome startup trace files.
- Browser smoke/probe scripts.

Unavailable now:

- Full Xcode.
- Instruments / `xctrace`.
- Metal command-line compiler tools.
- `gpudebug`.
- `metalperftrace`.
- Per-shader hardware counters for memory bandwidth, occupancy, cache pressure,
  register pressure, or stall reasons.

Recommended before the next shader rewrite:

1. Run WebGPU timestamp baselines in Bun for CLIP dispatches and integrated
   3D optimizer steps.
2. Run one Chrome trace to check browser/Dawn scheduling, pipeline creation, and
   queue gaps.
3. Only if full Xcode is available, run Metal System Trace and try a GPU capture
   against a small Bun/WebGPU benchmark before touching a shader.
4. Treat any shader rewrite as a fork under `experiments/clip_forks/vNN_*`,
   with a copied snapshot, a correctness gate, isolated timestamp gate, and
   integrated `step_matrix` gate.

## Local Machine And Runtime

Commands run:

```bash
uname -a
sw_vers
system_profiler SPDisplaysDataType SPHardwareDataType
```

Observed:

```text
Darwin Nicholass-MacBook-Air.local 24.5.0 Darwin Kernel Version 24.5.0
ProductVersion: 15.5
BuildVersion: 24F74

Model Name: MacBook Air
Model Identifier: Mac16,13
Chip: Apple M4
Total Number of Cores: 10 (4 performance and 6 efficiency)
Memory: 24 GB

Graphics:
Apple M4
Total Number of Cores: 10
Metal Support: Metal 3
```

Node/Bun tooling:

```bash
which bun node npm npx yarn python3
bun --version
node --version
npm --version
npx --version
yarn --version || true
python3 --version
```

Observed:

```text
/Users/nicholasbardy/.bun/bin/bun
/Users/nicholasbardy/.nvm/versions/node/v24.3.0/bin/node
/Users/nicholasbardy/.nvm/versions/node/v24.3.0/bin/npm
/Users/nicholasbardy/.nvm/versions/node/v24.3.0/bin/npx
yarn not found
/opt/homebrew/bin/python3

bun 1.3.10
node v24.3.0
npm 11.4.2
npx 11.4.2
Python 3.14.0
```

Package availability:

```bash
node -e "for (const p of ['puppeteer','puppeteer-core','chrome-launcher','playwright']) { try { console.log(p + ': ' + require.resolve(p)); } catch { console.log(p + ': unavailable'); } }"
```

Observed:

```text
puppeteer: /Users/nicholasbardy/git/Neural-Force-Field-Art/node_modules/puppeteer/lib/puppeteer/puppeteer.js
puppeteer-core: /Users/nicholasbardy/git/Neural-Force-Field-Art/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js
chrome-launcher: unavailable
playwright: unavailable
```

## Bun/WebGPU Path

Plain Bun does not expose WebGPU here:

```bash
bun -e 'console.log("navigator.gpu", !!globalThis.navigator?.gpu)'
```

Observed:

```text
navigator.gpu false
```

The repo's GPU benches use `bun-webgpu`:

```ts
import { setupGlobals } from "bun-webgpu";
setupGlobals();
```

Probe command:

```bash
bun -e 'import { setupGlobals } from "bun-webgpu"; setupGlobals(); const gpu = globalThis.navigator?.gpu; console.log("navigator.gpu", !!gpu); const adapter = await gpu.requestAdapter(); console.log("adapter", !!adapter); if (adapter) { console.log("info", JSON.stringify(adapter.info ?? null)); console.log("features", Array.from(adapter.features).sort().join(",")); console.log("limits", JSON.stringify({maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize})); const features = []; if (adapter.features.has("timestamp-query")) features.push("timestamp-query"); if (adapter.features.has("shader-f16")) features.push("shader-f16"); const device = await adapter.requestDevice({ requiredFeatures: features }); console.log("deviceFeatures", Array.from(device.features).sort().join(",")); }'
```

Observed:

```text
navigator.gpu true
adapter true
info {"vendor":"apple","architecture":"metal-3","device":"apple-m4","description":"Metal driver on macOS Version 15.5 (Build 24F74)","backendType":5,"adapterType":2,"vendorID":4203,"deviceID":0,"subgroupMinSize":4,"subgroupMaxSize":64,"isFallbackAdapter":false}
features ... shader-f16 ... subgroups ... timestamp-query ...
limits {"maxComputeWorkgroupStorageSize":32768,"maxComputeInvocationsPerWorkgroup":1024,"maxStorageBufferBindingSize":12884901888,"maxBufferSize":12884901888}
deviceFeatures core-features-and-limits,shader-f16,timestamp-query
```

Conclusion: Bun plus `bun-webgpu` is the best immediate local path for
hardware-backed GPU timing. It exposes `timestamp-query` and `shader-f16` on the
real Apple M4 Metal adapter.

What it can tell us:

- Per-dispatch or per-pass GPU elapsed time in milliseconds.
- Whether a shader family is hot enough to justify work.
- Whether a fork improves isolated CLIP timing.
- Whether a fork survives the integrated optimizer schedule.

What it cannot tell us:

- Memory bandwidth.
- Cache hit/miss behavior.
- Occupancy.
- Register pressure.
- Threadgroup memory bank conflicts.
- Whether a kernel is ALU-bound or memory-bound.

## Repo Timestamp Commands That Work Now

CLIP per-dispatch profile:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=1 RUNS=1 WARMUP=0 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts > /tmp/nffa_clip_b3_dispatch_ts.csv
```

Probe result from the first command:

```text
adapter: apple metal-3
dispatch profile: mode=train, plan=plan_train.json, batch=1, precision=f32, weights=weights_train.bin, dispatches=281, runs=1, warmup=0, timing=gpu-timestamp

Groups:
  12.386 ms   34.5%  pw
   6.554 ms   18.2%  pw_bwd
   5.177 ms   14.4%  spatial_bwd
   4.391 ms   12.2%  conv
   2.687 ms    7.5%  attn_core_bwd
```

Integrated 3D optimizer profile:

```bash
TIMESTAMP=1 CLIP_BATCH=1 VIEWS=1 RUNS=1 WARMUP=0 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Probe result from the first command:

```text
adapter: apple metal-3
splat3d step bench: G=4096, views=1/9, clipBatch=1, clipLayout=per_view, clipPrecision=f32, viewSampler=epoch, weights=weights_train.bin, cap=2048, runs=1, warmup=0, stemSpatialBwd=1, spatialBwdVariant=generic, fusePointwiseGeluForward=1, fuseGeluBwdIntoPw=0, fuseResidualBwdIntoPw=0, singlePassBatchRasterForward=0, viewLaneBatchRasterForward=0, viewLaneBatchRasterBackward=0, timing=gpu-timestamp, compile+allocate=411 ms
normal step avg: 68.32 ms
profile: total=38.80 ms rasterFwd=1.18 rasterReplay=0.00 rasterBwd=6.55 clipFwd=12.78 clipBwd=17.37 clipBatch=0.00 adam=0.07 display=0.79 timing=gpu-timestamp
```

Use `RUNS >= 5` and `WARMUP >= 3` for decisions. The short probe above only
proves the timing path is available.

## Chrome And Browser Profiling

Chrome is installed:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version
```

Observed:

```text
Google Chrome 150.0.7871.46
```

### Headless WebGPU Reality

Repo smoke probe:

```bash
node tools/smoke.mjs http://localhost:1234/splat3d.html 1500 /tmp
```

Observed:

```text
URL        http://localhost:1234/splat3d.html
SCREENSHOT /tmp/smoke_1783512051820.png
PROBE      {"webgpu":true,"adapter":"null","adapterErr":null,"hud":null,"warning":false}
--- CONSOLE (3) ---
[warn] No available adapters.
[error] [splat3d_page] no WebGPU adapter available.
[warn] No available adapters.
```

Puppeteer headless probe with Chrome path and WebGPU flags:

```bash
node --input-type=module - <<'EOF'
import puppeteer from 'puppeteer';
const browser = await puppeteer.launch({
  headless: 'new',
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features', '--disable-gpu-sandbox'],
});
const page = await browser.newPage();
await page.goto('about:blank');
const result = await page.evaluate(async () => {
  const gpu = !!navigator.gpu;
  const adapter = gpu ? await navigator.gpu.requestAdapter().catch((err) => ({ error: String(err) })) : null;
  return { gpu, adapter: !!(adapter && !adapter.error), info: adapter?.info ?? null, error: adapter?.error ?? null };
});
console.log(JSON.stringify(result));
await browser.close();
EOF
```

Observed:

```text
{"gpu":false,"adapter":false,"info":null,"error":null}
```

Conclusion: headless Chrome is useful for traces and page checks, but not a
reliable hardware WebGPU timing path on this machine. Use Bun/WebGPU timestamps
for kernel timing. Use headful Chrome manually if the question is browser-only
behavior.

### Puppeteer/CDP Trace Command That Works Now

This produced `/tmp/nffa_puppeteer_trace_probe.json` at `638540` bytes:

```bash
node --input-type=module - <<'EOF'
import fs from 'node:fs';
import puppeteer from 'puppeteer';

const out = '/tmp/nffa_puppeteer_trace_probe.json';
fs.rmSync(out, { force: true });

const browser = await puppeteer.launch({
  headless: 'new',
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: [
    '--enable-unsafe-webgpu',
    '--enable-webgpu-developer-features',
    '--disable-gpu-sandbox',
  ],
});

const page = await browser.newPage();
await page.tracing.start({
  path: out,
  categories: [
    'gpu',
    'dawn',
    'disabled-by-default-gpu.dawn',
    'disabled-by-default-gpu.service',
    'devtools.timeline',
    'blink',
    'renderer.scheduler',
    'v8',
    'benchmark',
  ],
});
await page.goto('http://localhost:1234/splat3d.html', {
  waitUntil: 'domcontentloaded',
  timeout: 10000,
}).catch((err) => console.log('goto', err.message));
await new Promise((r) => setTimeout(r, 1500));
await page.tracing.stop();
await browser.close();

const stat = fs.statSync(out);
console.log(`trace ${out} ${stat.size} bytes`);
EOF
```

What Chrome/CDP trace can tell us:

- Main-thread and renderer scheduling.
- Browser GPU-process activity.
- Dawn/WebGPU trace events if emitted under these categories.
- Pipeline creation or compilation spikes.
- Command submission gaps and CPU-side stalls.
- Whether the page is blocked by JS, layout, fetch, or compile behavior.

What Chrome/CDP trace cannot tell us:

- Per-shader memory bandwidth.
- Occupancy.
- Register pressure.
- Cache behavior.
- A reliable hardware WebGPU adapter in headless mode on this Mac.

### Chrome Startup Trace Command

This command created `/tmp/nffa_chrome_trace_probe.json` at about `933K`, but
the test run stayed alive longer than expected and I interrupted it. Prefer the
Puppeteer/CDP trace above for unattended captures. Use startup tracing for
manual/headful profiling windows.

```bash
mkdir -p /tmp/nffa_trace_profile
open -na "Google Chrome" --args \
  --user-data-dir=/tmp/nffa_trace_profile \
  --enable-unsafe-webgpu \
  --enable-webgpu-developer-features \
  --ignore-gpu-blocklist \
  --trace-startup='gpu,dawn,disabled-by-default-gpu.dawn,disabled-by-default-gpu.service,devtools.timeline,blink,renderer.scheduler,v8,benchmark' \
  --trace-startup-duration=30 \
  --trace-startup-file=/tmp/nffa_webgpu_trace.json \
  --trace-startup-format=json \
  http://localhost:1234/splat3d.html
```

Then open the trace:

```bash
open -a "Google Chrome" /tmp/nffa_webgpu_trace.json
```

or load it in:

```text
chrome://tracing
https://ui.perfetto.dev
```

Browser sanity checks:

```text
chrome://gpu
chrome://tracing
chrome://version
```

Use `chrome://gpu` to confirm WebGPU status, ANGLE/backend status, driver
status, and blocklist warnings in a real headful browser session.

## Xcode And Metal Tooling

Commands run:

```bash
xcode-select -p
xcodebuild -version
ls -ld /Applications/Xcode.app /Applications/Xcode-beta.app /Library/Developer/CommandLineTools /Library/Developer/CommandLineTools/Applications/Instruments.app 2>&1 || true
for tool in xctrace Instruments metal metallib metal-dsymutil metal-ar gpudebug metalperftrace; do printf '%s: ' "$tool"; xcrun --find "$tool" 2>&1 || true; done
xcrun xctrace list templates
xcrun metalperftrace --help
```

Observed:

```text
/Library/Developer/CommandLineTools
xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance

ls: /Applications/Xcode-beta.app: No such file or directory
ls: /Applications/Xcode.app: No such file or directory
ls: /Library/Developer/CommandLineTools/Applications/Instruments.app: No such file or directory
drwxr-xr-x  5 root  wheel  160 Jul  4  2025 /Library/Developer/CommandLineTools

xctrace: xcrun: error: unable to find utility "xctrace", not a developer tool or in PATH
Instruments: xcrun: error: unable to find utility "Instruments", not a developer tool or in PATH
metal: xcrun: error: unable to find utility "metal", not a developer tool or in PATH
metallib: xcrun: error: unable to find utility "metallib", not a developer tool or in PATH
metal-dsymutil: xcrun: error: unable to find utility "metal-dsymutil", not a developer tool or in PATH
metal-ar: xcrun: error: unable to find utility "metal-ar", not a developer tool or in PATH
gpudebug: xcrun: error: unable to find utility "gpudebug", not a developer tool or in PATH
metalperftrace: xcrun: error: unable to find utility "metalperftrace", not a developer tool or in PATH
```

Conclusion: this Mac has Metal runtime support, but not the full Apple GPU
profiling toolchain selected or installed.

## Xcode/Metal Commands If Full Xcode Exists Later

First select full Xcode:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcodebuild -version
xcrun xctrace list templates | rg -i 'metal|gpu'
xcrun --find gpudebug
xcrun --find metalperftrace
xcrun --find metal
```

### Metal System Trace For Bun

Prefer Bun for shader work because it avoids Chrome's multi-process browser
layer and directly runs the same WGSL through Dawn/Metal.

Check syntax on the installed Xcode first:

```bash
xcrun xctrace record --help
```

Candidate command:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 20s \
  --output /tmp/nffa_bun_clip_metal_system.trace \
  --launch -- /Users/nicholasbardy/.bun/bin/bun tools/clip/dispatch_profile.ts
```

With the environment we actually care about:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 30s \
  --output /tmp/nffa_bun_clip_b3_metal_system.trace \
  --launch -- /usr/bin/env \
    TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 \
    /Users/nicholasbardy/.bun/bin/bun tools/clip/dispatch_profile.ts
```

Open the trace:

```bash
open /tmp/nffa_bun_clip_b3_metal_system.trace
```

### Metal System Trace For Chrome

Start the dev server:

```bash
npx parcel src/index.html src/splat.html src/splat3d.html --port 1234 --no-cache
```

Launch a clean Chrome profile:

```bash
open -na "Google Chrome" --args \
  --user-data-dir=/tmp/nffa_metal_trace_chrome_profile \
  --enable-unsafe-webgpu \
  --enable-webgpu-developer-features \
  --ignore-gpu-blocklist \
  http://localhost:1234/splat3d.html
```

Find Chrome processes:

```bash
pgrep -fl "Google Chrome"
```

Then attach with Metal System Trace, if the installed `xctrace record --help`
shows `--attach` support:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 30s \
  --attach "$(pgrep -n 'Google Chrome')" \
  --output /tmp/nffa_chrome_metal_system.trace
```

Chrome is multi-process. If the trace misses GPU work, attach to the Chrome GPU
process shown by:

```bash
ps aux | rg 'Google Chrome Helper.*--type=gpu-process'
```

### Xcode GPU Capture

Use this only after timestamp baselines identify a specific dispatch family.

Likely target order:

1. A small Bun/WebGPU repro such as `tools/clip/dispatch_profile.ts`.
2. An integrated Bun bench such as `tools/splat3d/step_bench.ts`.
3. Headful Chrome only if the bug is browser-specific.

Why Bun first:

- Fewer processes.
- No page rendering/layout noise.
- Same generated WGSL kernels.
- Easier to capture a short repeatable benchmark.

What GPU capture can tell us if it works:

- Per-dispatch command details.
- Hardware counters where available.
- Memory read/write behavior.
- Threadgroup occupancy.
- Register pressure or spill hints, depending on Xcode/GPU support.

What may fail:

- Chrome/Dawn capture can be brittle because WebGPU objects are owned under
  browser/GPU helper processes.
- Counter names and availability vary by Xcode and Apple GPU generation.
- WGSL source line mapping may be weak because Dawn translates to Metal.

### `metalperftrace`

Check availability:

```bash
xcrun --find metalperftrace
xcrun metalperftrace --help
```

If present, use it as a command-line trace/summarization path around the same
Bun benchmarks. Do not assume exact arguments until `--help` is visible on the
installed Xcode version.

## Tool Capability Matrix

| Tool path | Available now | Tells us | Does not tell us |
| --- | --- | --- | --- |
| Bun/WebGPU timestamps | Yes | Per-dispatch/pass GPU elapsed ms | Bandwidth, occupancy, cache, register pressure |
| `tools/clip/dispatch_profile.ts` | Yes | CLIP hot shader families | Integrated optimizer interaction unless paired with `step_bench` |
| `tools/splat3d/step_bench.ts` | Yes | Real optimizer phase split | Per-dispatch CLIP internals |
| `tools/splat3d/step_matrix.ts` | Yes | Sequential A/B promotion gate | Hardware stall reasons |
| Puppeteer/CDP trace | Yes | Browser/Dawn scheduling, compile spikes, queue gaps | Reliable headless hardware WebGPU timing, shader counters |
| Chrome startup trace | Yes, manual preferred | Browser/GPU process timeline | Shader counters |
| `chrome://gpu` | Yes, headful | Backend/blocklist/feature sanity | Timings |
| Instruments Metal System Trace | No | CPU/GPU timeline, queues, command buffers, utilization | WGSL source-level math by itself |
| Xcode GPU capture/counters | No | Best chance at bandwidth/occupancy/register evidence | May be brittle through Chrome/Dawn |
| `gpudebug` | No | Command-line GPU trace investigation | Unavailable until full Xcode |
| `metalperftrace` | No | CLI Metal trace/summaries if installed | Unavailable until full Xcode |

## Recommended Profiling Protocol Before Next Shader Rewrite

1. Preserve the current state.

```bash
git status --short
git rev-parse --short HEAD
```

If source files are dirty, do not rewrite shaders in place. Copy a fork:

```bash
mkdir -p experiments/clip_forks/vNN_name/snapshot
git archive HEAD src/clip src/splat3d tools/clip tools/splat3d | tar -x -C experiments/clip_forks/vNN_name/snapshot
```

2. Run a same-resolution CLIP timestamp baseline.

Use the current MobileCLIP input resolution. Do not lower resolution for shader
claims unless the experiment is explicitly a grid/contact-sheet scheduling
experiment.

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=7 WARMUP=3 bun tools/clip/dispatch_profile.ts > /tmp/nffa_clip_b3_before.csv
```

3. Run integrated optimizer baselines.

```bash
TRIALS=3 CONFIGS=base=3:3,full=9:3 RUNS=7 WARMUP=3 bun tools/splat3d/step_matrix.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=7 WARMUP=3 bun tools/splat3d/step_bench.ts
```

4. Record one Chrome trace.

Use the Puppeteer/CDP command above. Inspect it for:

- pipeline creation during measured windows;
- long gaps between command submissions;
- renderer or GPU-process stalls;
- JS/main-thread work contaminating the measurement.

5. If full Xcode is available, record one Metal System Trace before editing.

Use the Bun command first. Look for:

- GPU queue idle gaps;
- command buffer fragmentation;
- whether the run is GPU-saturated or CPU-submit-bound;
- memory pressure or performance-state changes.

6. Pick only one target family for the fork.

A rewrite target should satisfy both:

- greater than roughly `15%` of same-resolution CLIP timestamp time; and
- visible in integrated `step_bench` or `step_matrix`.

Current likely targets remain:

- `pw` / `pw_bwd`;
- `spatial_bwd`;
- `conv`;
- selected forward/backward fusions.

7. Gate the fork.

Correctness:

```bash
bun tools/clip/fused_test.ts
bun tools/clip/bwd_test.ts
```

Isolated timing:

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=7 WARMUP=3 bun tools/clip/dispatch_profile.ts > /tmp/nffa_clip_b3_after.csv
```

Integrated timing:

```bash
TRIALS=5 CONFIGS=base=3:3,variant=3:3 RUNS=7 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

8. Promote only if the integrated result moves.

Do not promote a shader fork based only on an isolated microbench. The previous
forks have shown that isolated dispatch wins can disappear once CLIP, raster,
queue scheduling, and optimizer phases are composed.

## Bottom Line

For work we can do immediately, use Bun/WebGPU timestamp queries as the numeric
source of truth and Chrome traces as the browser/queue sanity check. For the
question "is this shader memory-bound, occupancy-bound, spilling, or cache
limited?", this Mac needs full Xcode/Metal profiling tools selected first. Until
then, any 2x-4x CLIP claim should be treated as a scheduling/modeling result,
not as a proven single-shader hardware result.
