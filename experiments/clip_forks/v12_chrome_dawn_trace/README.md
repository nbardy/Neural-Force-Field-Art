# v12 Chrome Dawn Trace

Date: 2026-07-08

## Purpose

Add a reproducible Chrome DevTools trace path for the browser 3D optimizer.
This is not a replacement for WebGPU timestamp-query benches. It answers
browser/Dawn scheduling questions: command traffic, queue submits, pipeline
events, GPU process backpressure, and whether the page is actually running the
intended WebGPU optimization path.

## Snapshot

The `snapshot/` directory contains:

- `tools/webgpu_trace.mjs`
- `agent_notes/optimization_session/agent_v13_chrome_trace_plan.md`
- `docs/SPLAT3D_PERF_NOTES.md`
- `experiments/clip_forks/README.md`

Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v12_chrome_dawn_trace
```

## Working Command

Build with relative URLs, serve the repo root, then trace the built page so
`/dist/` assets and `/models/` weights are same-origin:

```bash
npx parcel build --no-scope-hoist --no-cache --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
GRID9=1 DIRECT_GRID=1 SETUP_TIMEOUT_MS=180000 node tools/webgpu_trace.mjs http://localhost:8799/dist/splat3d.html 2000 /tmp/nffa_trace_probe4
```

The earlier attempt against Parcel dev server failed because Parcel returned
`index.html` for `/models/mobileclip_s0/plan_train.json`, producing:

```text
boot failed: Unexpected token '<', "<!DOCTYPE "... is not valid JSON
```

That is a serving issue, not a WebGPU issue.

## Captured Result

Successful trace:

```text
TRACE /tmp/nffa_trace_probe4/webgpu_trace_2026-07-08T12-39-16-079Z.json
SCREENSHOT /tmp/nffa_trace_probe4/webgpu_trace_2026-07-08T12-39-16-079Z.png
BEFORE {"gpu":true,"ready":true,"running":true,"step":2,"phase":"run","clipLayout":"grid9_close2","gridDirectRaster":true,"viewsPerStep":9,"clipBatchSize":3,"error":null}
AFTER {"gpu":true,"ready":true,"running":true,"step":35,"phase":"run","clipLayout":"grid9_close2","gridDirectRaster":true,"viewsPerStep":9,"clipBatchSize":3,"error":null}
events: 40331
```

Top relevant event buckets:

```text
disabled-by-default-gpu.dawn :: DawnCommands                         2223
gpu :: CALayerTreeCoordinator::ApplyBackpressure::Metal              1655
disabled-by-default-gpu.graphite.dawn :: Queue::Tick                 1476
disabled-by-default-gpu.dawn :: DeviceBase::APITick::IsDeviceIdle    1149
disabled-by-default-gpu.graphite.dawn :: DeviceMTL::SubmitPendingCommandBuffer 632
disabled-by-default-gpu.graphite.dawn :: CommandEncoder::Finish       327
disabled-by-default-gpu.graphite.dawn :: Queue::Submit                327
```

## What It Proves

- Chrome/Metal WebGPU can run the built 3D page under Puppeteer.
- The page reached the intended `grid9_close2 + direct80 + batch x3` path.
- The trace contains Dawn/WebGPU/Metal scheduling events.
- The trace helper can produce a JSON trace and screenshot for Perfetto or
  `chrome://tracing`.

## What It Does Not Prove

- It does not provide per-shader memory bandwidth, cache, occupancy, register
  pressure, or stall counters.
- It does not replace `TIMESTAMP=1` CLIP dispatch timing.
- It is not a quality/convergence gate.

Use it to answer browser scheduling and queue questions. Use Bun/WebGPU
timestamp benches for shader timing, and use fixed-wall-clock visual/teacher
score gates for optimizer quality.
