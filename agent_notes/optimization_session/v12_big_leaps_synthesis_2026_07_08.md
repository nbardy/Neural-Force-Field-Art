# v12 Big-Leap Synthesis

Date: 2026-07-08

This note integrates the five sidecar agent notes plus the successful
Chrome/Dawn trace capture.

## What Is Settled

- fp16 was not a big jump. The landed experiment was weight-f16 storage, not
  activation-f16 or full mixed precision. It halved weight bytes but did not
  pass the strict input-gradient cosine gate and did not produce a promotable
  integrated speedup.
- `pointwise` means `1x1` dense channel mixing at each spatial location:
  `Y[co,p] = b[co] + sum_ci X[ci,p] * W[ci,co]`.
- `pw_bwd` computes the frozen-weight input gradient:
  `dX[ci,p] = sum_co dY[co,p] * W[ci,co]`.
- CLIP input tensors stay `[3,256,256]`. The grid lane also feeds a `256x256`
  CLIP tensor, but each contact-sheet cell is lower per-camera resolution.
- Whole-CLIP fusion is not realistic as one giant shader because training needs
  saved activations, shape changes, reductions, and dispatch-boundary
  synchronization. Local fusions are useful; v11 showed a small real win.
- The browser now has a real Chrome/Dawn trace helper, but not Metal hardware
  counters. WebGPU timestamps remain the shader-duration source of truth.

## What The Fresh Chrome Trace Proved

Command:

```bash
npx parcel build --no-scope-hoist --no-cache --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
GRID9=1 DIRECT_GRID=1 SETUP_TIMEOUT_MS=180000 node tools/webgpu_trace.mjs http://localhost:8799/dist/splat3d.html 2000 /tmp/nffa_trace_probe4
```

Result:

```text
TRACE /tmp/nffa_trace_probe4/webgpu_trace_2026-07-08T12-39-16-079Z.json
SCREENSHOT /tmp/nffa_trace_probe4/webgpu_trace_2026-07-08T12-39-16-079Z.png
BEFORE step=2 grid9_close2 direct80 batch x3
AFTER  step=35 grid9_close2 direct80 batch x3
events=40331
```

The trace includes `DawnCommands`, `Queue::Submit`,
`DeviceMTL::SubmitPendingCommandBuffer`, and Metal backpressure events. This is
good enough for browser scheduling and queue-gap investigation. It is not
enough for bandwidth/occupancy diagnosis.

## Best 2-4x Candidates

1. **Cached full-CLIP gradient cadence.**
   Run full MobileCLIP at 256px every `K` steps, cache `dL/dimage`, and use
   raster+Adam-only intermediate steps. Test `K=2,4` with full-teacher fixed
   wall-clock scoring.

2. **Lower full-resolution view cadence plus refresh.**
   Try `1/9` or `2/9` full-resolution views per step with epoch coverage and
   periodic all-view refresh. The gate is 9-view teacher score and screenshots
   after equal wall time.

3. **Grid/contact-sheet as schedule proxy.**
   Use `grid9_close2 + direct80 + grid prompt` most steps, periodically refresh
   full per-view 256px supervision. This is not equivalent to nine full-res
   CLIP losses, so judge by visual quality and teacher score.

4. **Pointwise tile-family fork.**
   Add shape-gated rectangular pointwise/pw_bwd tile emitters rather than
   another shared-W micro tweak. Targets: high-P `64x64/32x32` layers and
   low-P high-channel `8x8/16x16` layers. Gate with `bwd_test`, timestamp
   dispatch profile, and integrated `step_matrix`.

5. **Selective activation-f16 storage.**
   Different from weight-f16. Requires per-slot dtype metadata and f32 math for
   sensitive reductions, input, output, and `dL/dpixels`. Higher risk, but more
   plausible memory-traffic upside than weight-f16.

## UI Strategy

- Keep default page as `per-view CLIP`, `3/9 views`, `epoch views`,
  `batch CLIP x3`.
- Keep grid mode visible as `3x3 grid + 2`.
- Prefer `grid raster 80` when grid mode is selected, but do not make grid mode
  the global default until quality is tested.
- Keep f16, shared-W, raster lane scheduling, caps, and timestamp controls
  env/tool-only.
- Consider silently defaulting `depthwise4` and the paired v11 backward local
  fusions only after one more interactive visual pass.

## Next Forks Worth Doing

- `v13_cached_clip_gradient`: cached `dL/dimage` cadence experiment.
- `v14_pointwise_tile_family`: shape-gated rectangular pointwise/pw_bwd tiles.
- `v15_grid_cell_direct_state`: remove grid direct-raster copy/scatter/replay
  tax by making contact-sheet cells first-class raster lanes.
- `v16_ui_grid80_default`: make grid mode choose direct80 by default and add
  screenshot presets if needed.

Each should have a snapshot folder, correctness gates, integrated same-session
benchmarks, and a fixed-wall-clock quality readout where the objective is not
mathematically identical.
