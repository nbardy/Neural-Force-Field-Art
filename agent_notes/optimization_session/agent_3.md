# Agent 3 Optimization Session Notes - Batch-Major CLIP and 3D Splat Wiring

Date: 2026-07-08

Scope: I studied the current CLIP implementation in `src/clip`, `tools/clip`,
`docs/CLIP_BATCHING_NOTES.md`, `models/mobileclip_s0/plan_train.json`, and the
current 3D optimizer call sites in `src/splat3d`. I did not modify app code.

## Executive Summary

The production path should not use `ReplicatedBatchVisionTrainer`. The existing
notes already show that it reduces CPU encode/submit time but loses on GPU wall
time (`docs/CLIP_BATCHING_NOTES.md:35`). The useful path is the true
batch-major trainer:

- `BatchMajorVisionTrainer` exists and verifies exact full `dL/dpixels` parity
  for tested B=2, B=3, and B=9 (`docs/CLIP_BATCHING_NOTES.md:125`,
  `docs/CLIP_BATCHING_NOTES.md:148`).
- Current warmed numbers are already optimizer-relevant: B=3 forward+backward
  drops from 102.55 ms/batch to 49.48 ms/batch, and B=9 drops from 389.66
  ms/batch to 134.63 ms/batch (`docs/CLIP_BATCHING_NOTES.md:155`).
- The main integration trap is not CLIP. It is the 3D rasterizer state. Current
  `Raster3DEngine` has one `derived`, one `tileCounts`, one `binnedIds`, one
  `tileStop`, one `image`, and one `gradImage` buffer (`src/splat3d/raster.ts:67`).
  If we render N views into CLIP lanes before one batched CLIP pass, the raster
  forward state for views 0..N-2 is overwritten before their backward pass.
  The first safe integration must either rerun raster forward before each
  raster backward, or add per-lane raster forward state.

Most likely optimization priority:

1. Wire `BatchMajorVisionTrainer` into `Splat3DOptimizer` behind a toggle or
   chunked batch cap, with correctness preserved by rerunning raster forward
   before each backward. This validates real app speed immediately.
2. Add a per-dispatch profiler before changing kernels. Static plan math says
   pointwise forward and `pw_bwd` dominate, but GPU wall time must select exact
   steps.
3. Integrate selective shared-W pointwise only for shapes/batches that win in a
   full CLIP profile. The existing microbench proves the idea but also proves it
   is shape-specific, not a blanket replacement.
4. If raster duplicate-forward eats the CLIP win, add per-lane raster forward
   state or offset-bound raster image/gradient bindings.

## Current Implementation Map

Runtime:

- `src/clip/vision.ts:38` - `VisionEncoder`, single-image forward runtime.
- `src/clip/vision.ts:198` - `VisionTrainer`, single-image forward plus loss
  plus backward runtime.
- `src/clip/vision.ts:295` - `VisionTrainer.encode()` emits forward plus
  optional backward into one compute pass.
- `src/clip/vision.ts:308` and `src/clip/vision.ts:320` - split
  `encodeForward()` / `encodeBackward()` used by the current 3D profiler.
- `src/clip/vision_batch.ts:74` - replicated batch trainer. This is an
  experiment, not a production path.
- `src/clip/vision_batch.ts:269` - `BatchMajorVisionEncoder`, true batched
  forward.
- `src/clip/vision_batch.ts:398` - `BatchMajorVisionTrainer`, true batched
  forward plus backward.
- `src/clip/vision_batch.ts:493` - lane slot offset calculation for batch-major
  activation and grad buffers.
- `src/clip/vision_batch.ts:506` - lane text offset calculation.

Codegen:

- `src/clip/vision_wgsl.ts:177` - `pointwiseTiledMain()`, shared by forward
  pointwise conv and backward `pw_bwd`.
- `src/clip/vision_wgsl.ts:231` - forward pointwise conv emitter.
- `src/clip/vision_wgsl.ts:274` - spatial conv emitter for depthwise and
  general conv.
- `src/clip/vision_wgsl.ts:383` - SE emitter.
- `src/clip/vision_wgsl.ts:442` - attention core emitter.
- `src/clip/vision_wgsl.ts:540` - head emitter.
- `src/clip/vision_wgsl.ts:581` - train-mode standalone GELU emitter.
- `src/clip/vision_bwd_wgsl.ts:140` - `pw_bwd`, reusing the pointwise tiled
  matmul over transposed weights.
- `src/clip/vision_bwd_wgsl.ts:224` - spatial backward, correctness-first
  gather form.
- `src/clip/vision_bwd_wgsl.ts:289` - SE backward.
- `src/clip/vision_bwd_wgsl.ts:377` - head backward.
- `src/clip/vision_bwd_wgsl.ts:415` - cosine loss backward.
- `src/clip/vision_bwd_wgsl.ts:475` - attention core backward.
- `src/clip/vision_batch_wgsl.ts:77` - generic source rewrite that adds
  `workgroup_id.z` lane indexing and offsets every slot/text storage access.
- `src/clip/vision_batch_wgsl.ts:110` - `batchForwardDispatches()`.
- `src/clip/vision_batch_wgsl.ts:126` - `batchTrainDispatches()`.
- `src/clip/vision_batch_pointwise.ts:39` - z-batch pointwise microbench
  baseline.
- `src/clip/vision_batch_pointwise.ts:105` - shared-W pointwise microkernel.

Tools and notes:

- `tools/clip/batch_bench.ts:1` - replicated activation batch scheduling bench.
- `tools/clip/batch_major_forward_bench.ts:1` - true batch-major forward bench.
- `tools/clip/batch_major_train_bench.ts:1` - true batch-major train bench.
- `tools/clip/pointwise_batch_bench.ts:1` - shared-W pointwise microbench.
- `tools/clip/fused_test.ts:85` - per-step forward verification.
- `tools/clip/bwd_test.ts:566` - end-to-end backward directional derivative.
- `tools/clip/compile_plan.py:39` - train-plan differences.
- `tools/clip/compile_plan.py:553` - backward list generation.

3D optimizer:

- `src/splat3d/optimize.ts:57` - current optimizer owns a single
  `VisionTrainer`.
- `src/splat3d/optimize.ts:60` - current per-view text embeddings live in
  separate GPU buffers.
- `src/splat3d/optimize.ts:131` - hot `step()` path.
- `src/splat3d/optimize.ts:136` - current per-view loop starts.
- `src/splat3d/optimize.ts:137` - per-view text copy into single CLIP text
  buffer.
- `src/splat3d/optimize.ts:138` - raster forward for one view.
- `src/splat3d/optimize.ts:139` - copy raster image into single CLIP input.
- `src/splat3d/optimize.ts:140` - single-image CLIP forward plus backward.
- `src/splat3d/optimize.ts:141` - copy CLIP image gradient into raster
  `gradImage`.
- `src/splat3d/optimize.ts:142` - raster backward for the same view.
- `src/splat3d/optimize.ts:150` - split-submit profiler.
- `src/splat3d/raster.ts:121` - single raster `image` buffer.
- `src/splat3d/raster.ts:122` - single raster `gradImage` buffer.
- `src/splat3d/raster.ts:215` - `recordForward()`.
- `src/splat3d/raster.ts:234` - `recordBackwardAdd()`.
- `src/splat3d/raster_wgsl.ts:280` - raster image is already planar RGB
  `[R plane][G plane][B plane]`, matching CLIP NCHW input.
- `src/splat3d/raster_wgsl.ts:296` - raster backward reads planar
  `gradImage`, also matching CLIP `dL/dpixels`.

## Plan Facts From `plan_train.json`

Header facts:

- Model: `mobileclip_s0_vision` (`models/mobileclip_s0/plan_train.json:2`).
- Input: slot 0, shape `[3,256,256]` (`models/mobileclip_s0/plan_train.json:3`).
- Output embedding: slot 129, dim 512 (`models/mobileclip_s0/plan_train.json:9`).
- Weights: 21,515,712 floats, about 86.1 MB
  (`models/mobileclip_s0/plan_train.json:273`).
- Steps start at line 274 (`models/mobileclip_s0/plan_train.json:274`).
- Backward starts at line 2920 (`models/mobileclip_s0/plan_train.json:2920`).
- `nActSlots=130`, `inputGradSlot=130`, `embedSlot=129`,
  `embedGradSlot=259`, `textDim=512`
  (`models/mobileclip_s0/plan_train.json:4846`).

Train-plan memory:

- Slots: 260 total, activation plus mirrored grad slots.
- Total slot floats across all slots: 46,269,440 floats.
- Slot storage per batch lane: about 185.1 MB decimal, 176.5 MiB.
- B=3 slot storage: about 555 MB plus 86 MB weights.
- B=9 slot storage: about 1.67 GB plus 86 MB weights.
- This is why production should not casually allocate B=9 at boot on every
  machine. Prefer a B=3 batch-cap, chunk larger view sets, and only allocate B=9
  on explicit request after profiling.

Step mix:

- Forward steps: 129.
- Backward entries: 152.
- Forward by kind: 93 conv, 30 GELU, 3 SE, 2 attention core, 1 head.
- Conv variants: 48 pointwise, 40 depthwise, 5 general.
- Backward by kind: 48 `pw_bwd`, 45 `spatial_bwd`, 30 `gelu_bwd`,
  22 `residual_bwd`, 3 `se_bwd`, 2 `attn_core_bwd`, 1 `head_bwd`,
  1 `loss_bwd`.

Approximate FLOP ranking from the plan:

- Forward pointwise conv: about 4.429 GFLOP of about 4.935 GFLOP total.
- Backward `pw_bwd`: about 4.429 GFLOP of about 5.058 GFLOP total.
- Backward `spatial_bwd`: about 0.574 GFLOP, with the first stem
  `spatial_bwd` alone about 0.226 GFLOP
  (`models/mobileclip_s0/plan_train.json:4828`).
- Attention backward is only about 0.042 GFLOP. It may still have private-array
  pressure, but it should not be the first rewrite target without measurements.

Important representative steps:

- Early 64x64 expansion: step 8, `64->192 @64x64`
  (`models/mobileclip_s0/plan_train.json:428`).
- Early 64x64 contraction with residual: step 10, `192->64 @64x64`
  (`models/mobileclip_s0/plan_train.json:463`).
- Mid 16x16 expansion family: step 97, `256->768 @16x16`
  (`models/mobileclip_s0/plan_train.json:2297`).
- Mid 16x16 contraction with residual: step 99, `768->256 @16x16`
  (`models/mobileclip_s0/plan_train.json:2331`).
- Final input-gradient-producing stem backward: `spatial_bwd` from `dY=131`
  to `dX=130`, `cin=3`, `cout=64`, `k=3`, `stride=2`, `h=w=256`
  (`models/mobileclip_s0/plan_train.json:4828`).

## Bottleneck Hypotheses

H1: Pointwise forward and `pw_bwd` dominate real GPU time.

The static plan makes this the default hypothesis. Both directions each have
about 4.429 GFLOP, and both use the same tiled matmul body
(`src/clip/vision_wgsl.ts:177`, `src/clip/vision_bwd_wgsl.ts:140`). Batch-major
currently dispatches the same pointwise workgroup grid once per batch lane via
`workgroups.z=batch` (`src/clip/vision_batch_wgsl.ts:110`). That means every
lane reloads the same W tile into workgroup memory. The shared-W pointwise
microkernel exists exactly to attack this (`src/clip/vision_batch_pointwise.ts:105`).

H2: Shared-W is not automatically faster.

Existing data shows B=2 expansion layers can win modestly, while B=3 often
loses and contraction/residual layers are mixed
(`docs/CLIP_BATCHING_NOTES.md:203`). This is probably occupancy pressure:
B=3 uses workgroup size `(8,8,3)=192` invocations and 16 KB workgroup memory
(12 KB X plus 4 KB W), leaving less room for concurrent workgroups
(`docs/CLIP_BATCHING_NOTES.md:219`). The production rewrite must be selective,
driven by per-step profile and shape allowlists.

H3: The first `spatial_bwd` is a real non-pointwise hotspot.

The stem backward is a gather over input pixels: each input pixel loops over
up to 64 output channels and 3x3 taps (`src/clip/vision_bwd_wgsl.ts:224`).
Static estimate: about 226.5 MFLOP for just the final `dL/dpixels` conv
(`models/mobileclip_s0/plan_train.json:4828`). This is not the largest class,
but it is one exact kernel near the end of every train pass and should be in
the top profile table.

H4: Train-plan standalone GELU is probably dispatch and bandwidth overhead.

Train mode splits GELU so backward can read pre-activation
(`tools/clip/compile_plan.py:39`, `tools/clip/compile_plan.py:248`). That
creates 30 forward GELU dispatches and 30 backward GELU dispatches. FLOPs are
not huge, but the forward side reads pre and writes post in a separate pass
(`src/clip/vision_wgsl.ts:581`). A fused "write pre and post" conv/SE path could
remove 30 forward dispatches and one read/write of the large activation tensors
while still preserving pre-activation for `gelu_bwd`.

H5: Batch-major CLIP can be correct and still lose in the 3D optimizer if raster
state is handled naively.

Current 3D `step()` has this safe order for each view:

1. Raster forward for view V.
2. Copy image to CLIP.
3. CLIP forward/backward.
4. Copy CLIP image grad to raster.
5. Raster backward for view V.

The raster backward depends on `derived`, `tileCounts`, `binnedIds`, and
`tileStop` from the immediately preceding raster forward. If a new integration
renders all selected views first, only the last view's state remains in those
single buffers (`src/splat3d/raster.ts:67`). Therefore, a safe batch CLIP path
must rerun raster forward before each raster backward, or store per-lane raster
forward state.

H6: Current benchmark wall times include readback fences, not GPU timestamps.

The CLIP benches use small readbacks as the synchronization fence
(`tools/clip/batch_major_train_bench.ts:180`, `tools/clip/fused_test.ts:153`,
`tools/clip/bwd_test.ts:641`). That is good enough for coarse wall trends, but
per-dispatch ranking needs timestamp queries or carefully isolated dispatch
microbenches.

H7: There is a small type hygiene issue in the untracked batch WGSL wrapper.

`src/clip/vision_batch_wgsl.ts:83` declares `new Map<string, SlotBinding>()`,
but the local interface is named `BatchBinding` (`src/clip/vision_batch_wgsl.ts:21`).
If a future TypeScript check includes this file, it should be fixed to
`BatchBinding`. This is not a runtime optimization, but it belongs early in the
patch stack to keep tooling clean.

## Kernel-Level Rewrite Candidates

### Candidate 1: Selective Shared-W Pointwise Forward

Current state:

- Generic batch-major path wraps the normal pointwise shader and sets
  `workgroups.z=batch` (`src/clip/vision_batch_wgsl.ts:110`).
- Shared-W experiment puts batch in `local_invocation_id.z` and stages W once
  per `(pixel tile, cout tile)` workgroup (`src/clip/vision_batch_pointwise.ts:105`).
- The microbench currently assumes compact buffers with slots 0, 1, and 2
  (`src/clip/vision_batch_pointwise.ts:18`). It is not yet a drop-in full-plan
  emitter.

Rewrite needed:

- Add a full-plan shared-W pointwise emitter that accepts the real `ConvStep`,
  real slot ids, and real per-slot lane strides from `plan.slots`.
- For each slot binding, use `slotBase = lane * (plan.slots[slot] / 4)` for
  `array<vec4f>` slots, not `cin * P4` or `cout * P4` unless the benchmark is
  using compact buffers.
- Residual base must use `plan.slots[residual] / 4`, not `dstStride`. The
  compact microbench gets away with `resBase=lane*dstStride`
  (`src/clip/vision_batch_pointwise.ts:64`) because its residual buffer is
  compact and same-shaped as dst.
- Add an allowlist keyed by `(batch, cin, cout, outH, outW, residual, direction)`,
  not by step index only, so repeated block shapes share the same decision.

Initial policy:

- Do not enable for B=3 by default. Existing B=3 shared-W results lose on
  `64->192 @64x64` and `256->768 @16x16` (`docs/CLIP_BATCHING_NOTES.md:205`).
- Enable only after a full-chain profile shows a specific shape is hot and
  isolated shared-W wins on that adapter.
- For B=2, retest expansion families first: `64->192 @64x64`, `128->384
  @32x32`, `256->768 @16x16`, and `512->1536 @8x8`.

### Candidate 2: Shared-W `pw_bwd`

Current state:

- `pw_bwd` uses the same pointwise tiled matmul over transposed weights
  (`src/clip/vision_bwd_wgsl.ts:140`).
- `tools/clip/compile_plan.py` packs `wOffT` for every train-mode pointwise
  conv (`tools/clip/compile_plan.py:226`, `tools/clip/compile_plan.py:591`).
- The existing shared-W microkernel is forward-style only. It handles bias,
  optional GELU, and optional residual/layer-scale. `pw_bwd` needs no bias or
  GELU, but it needs optional accumulate into destination grad slots.

Rewrite needed:

- Add `pointwiseSharedWBatchBwdDispatch()` beside the forward shared-W emitter.
- Inputs: `PwBwdStep`, `TrainPlan`, `batch`.
- Use `dY` as src, `dX` as dst, `wOffT`, `cin`, `cout`, `outH`, `outW`,
  `accumulate`.
- Store expression:
  - no accumulate: `dst[dstBase + ...] = acc`
  - accumulate: `dst[dstBase + ...] = dst[dstBase + ...] + acc`
- Benchmark separately from forward, because contraction-like backward shapes
  may behave differently than the forward microbench.

Why this may matter:

- `pw_bwd` is about as large as forward pointwise in the plan: about 4.429 GFLOP.
- It appears 48 times and includes the late large 512/1536 stage.
- A selective B=2 win here could matter even if forward shared-W is mixed.

### Candidate 3: Stem `spatial_bwd` Specialization

Current state:

- `spatial_bwd` is one thread per input pixel and loops over output channels,
  ky, and kx (`src/clip/vision_bwd_wgsl.ts:224`).
- It supports all spatial backward steps generically: depthwise and general,
  stride 1 and 2.

Rewrite target:

- Special-case the final input-gradient stem:
  - `cin=3`, `cout=64`, `k=3`, `stride=2`, `pad=1`, `h=w=256`,
    `outH=outW=128` (`models/mobileclip_s0/plan_train.json:4828`).
- Consider a tile over input pixels with vectorized output-channel reduction:
  stage a small W tile for several output channels, load a small dy tile, and
  compute multiple adjacent input pixels per thread or per subgroup.
- Alternative: formulate as a transposed convolution over output pixels and
  scatter into input pixels. This would need atomics or a deterministic
  partitioning to avoid write races, so the gather form is safer unless a tiling
  avoids overlap.

Benchmark gate:

- Only implement if timestamp or isolated profile puts this kernel near the top
  after pointwise.
- Validate against the existing `bwd_test.ts` spatial unit tests
  (`tools/clip/bwd_test.ts:304`) and full directional derivative
  (`tools/clip/bwd_test.ts:566`).

### Candidate 4: Fused Train GELU Forward

Current state:

- Train plan emits `conv act:"none"` into a pre-activation slot, then a separate
  `gelu` step into the post-activation slot (`tools/clip/compile_plan.py:488`).
- Backward reads post grad and pre activation (`src/clip/vision_bwd_wgsl.ts:193`).

Rewrite:

- Keep the pre slot for backward, but allow the producing conv/SE dispatch to
  write both:
  - binding A: pre activation
  - binding B: post activation = `gelu(pre)`
- Remove the separate forward GELU dispatch for fused cases.
- Keep `gelu_bwd` unchanged.
- Compiler options:
  - Minimal: keep `GeluStep` in the plan for debug metadata but have codegen
    fold it into the previous dispatch when safe.
  - Cleaner: add fields to `ConvStep`/`SeStep`, e.g. `postActDst`, `postActRef`,
    and do not emit a forward `GeluStep`.

Risk:

- Per-step ORT verification currently maps one plan step to one dispatch count
  (`src/clip/vision.ts:153`, `tools/clip/fused_test.ts:88`). Fusing GELU changes
  that mapping. Do it after batch integration and profiling, not before.

### Candidate 5: Direct Raster-to-CLIP Lane Binding

Current state:

- Raster forward writes planar RGB into `raster.image`
  (`src/splat3d/raster_wgsl.ts:280`).
- CLIP input is planar RGB NCHW `[3,256,256]`; no mean/std preprocessing is
  needed (`tools/clip/README.md:13`).
- Current 3D step copies `raster.image` into `trainer.inputBuffer`
  (`src/splat3d/optimize.ts:139`) and later copies `trainer.inputGradBuffer`
  into `raster.gradImage` (`src/splat3d/optimize.ts:141`).

Rewrite:

- Add `Raster3DEngine.recordForwardToImageBuffer(enc, view, imageBuffer, offset)`
  that uses the existing forward shader but creates/caches a bind group whose
  image binding points at `imageBuffer` with a storage binding offset.
- Add `Raster3DEngine.recordBackwardAddFromGradBuffer(enc, view, gradBuffer,
  offset)` similarly for the backward shader's `gradImage` binding.
- Storage offsets are feasible: one image lane is `3*256*256*4 = 786432` bytes,
  divisible by 256. CLIP lane offsets from `BatchMajorVisionTrainer` are also
  multiples of 256 for the input and input-grad slots.

Value:

- Removes two `copyBufferToBuffer()` calls per view.
- Lets raster forward write directly into the batch CLIP input lane.
- Lets raster backward read directly from the batch CLIP input-gradient lane.
- Does not by itself solve the single-buffer raster state issue.

### Candidate 6: Per-Lane Raster Forward State

Current state:

- Single state buffers in `Raster3DEngine`: `derived`, `tileCounts`,
  `binnedIds`, `tileStop`, `image`, `gradImage` (`src/splat3d/raster.ts:67`).

Rewrite:

- Introduce a small `Raster3DFrameState`:
  - `derived`: `G * DERIVED_STRIDE_3D` floats.
  - `tileCounts`: `numTiles` u32.
  - `binnedIds`: `numTiles * cap` u32.
  - `tileStop`: `numTiles` u32.
  - optional `image`, if not writing directly into CLIP input lane.
- Allocate `batchCap` states, not `numCameras` states.
- `recordForwardState(enc, view, laneState, optionalImageTarget)`.
- `recordBackwardAddState(enc, view, laneState, gradSource)`.

Value:

- Enables the ideal order:
  1. clear raw grad once.
  2. render all selected views into lane states and CLIP input lanes.
  3. run one batched CLIP forward/backward.
  4. run raster backward for each lane using the matching saved forward state.
  5. Adam once.
- Avoids duplicate raster forward in the minimal integration.

Memory estimate for default `G=4096`, `cap=2048`, `numTiles=256`:

- `derived`: 4096*11*4 = about 176 KB per lane.
- `binnedIds`: 256*2048*4 = 2 MB per lane.
- `image`: 786 KB per lane if needed.
- Total per lane is small compared with CLIP train slots. B=3 raster state is
  not the memory concern; CLIP slots are.

## Per-Dispatch Benchmarking Plan

### Phase 0: Preserve existing coarse gates

Run before any kernel rewrite:

```bash
PLAN=plan_train.json FAST=1 BENCH_RUNS=30 bun tools/clip/fused_test.ts
BENCH_RUNS=30 bun tools/clip/bwd_test.ts
BATCH=3 RUNS=5 WARMUP=5 bun tools/clip/batch_major_train_bench.ts
BATCH=9 RUNS=2 WARMUP=2 bun tools/clip/batch_major_train_bench.ts
```

Purpose:

- Keep forward train-plan parity against ORT.
- Keep end-to-end input-gradient directional derivative.
- Keep batch-major parity and wall trend.

### Phase 1: Static dispatch inventory

Add a small no-GPU report, or extend the Node snippet used in this session:

```bash
node tools/clip/plan_profile.mjs models/mobileclip_s0/plan_train.json
```

Output columns:

- index
- direction: fwd or bwd
- kind and variant
- shape
- workgroups from generated `DispatchSpec`
- approximate FLOPs
- approximate bytes touched
- slot ids
- slot byte strides under B=1, B=3, B=9
- current kernel family: generic z-batch, shared-W candidate, spatial, SE,
  attention, GELU, head, loss

This should be checked into `tools/clip` because it will guide shape allowlists.

### Phase 2: Isolated kernel microbench by dispatch spec

Add:

```bash
BATCH=3 DIR=fwd STEP_INDEX=57 RUNS=100 WARMUP=20 bun tools/clip/dispatch_microbench.ts
BATCH=3 DIR=bwd BWD_INDEX=84 RUNS=100 WARMUP=20 bun tools/clip/dispatch_microbench.ts
```

Implementation:

- Generate specs with `batchForwardDispatches()` or `batchTrainDispatches()`
  (`src/clip/vision_batch_wgsl.ts:110`, `src/clip/vision_batch_wgsl.ts:126`).
- Allocate only the buffers bound by the target spec, sized according to
  `plan.slots[slot] * batch`, plus weights and text if needed.
- Fill inputs with deterministic nonzero data.
- Dispatch only the target spec many times.
- Use a readback fence after warmup and after timing, matching existing benches.
- For `accumulate:true` kernels, initialize dst and accept that repeated runs
  mutate values. This is timing-only, not correctness. For correctness, run one
  dispatch and compare against the generic z-batch equivalent where available.

This isolates kernel occupancy and memory behavior without full graph noise.

### Phase 3: Full-chain timestamp profile

Add:

```bash
BATCH=3 RUNS=20 WARMUP=5 MODE=train bun tools/clip/dispatch_timestamps.ts
```

Implementation if `timestamp-query` is available:

- Request device with `requiredFeatures:["timestamp-query"]` when supported.
- Create a query set with `2 * dispatchCount * RUNS` timestamps.
- For profiling only, encode each dispatch as its own compute pass with
  `timestampWrites.beginningOfPassWriteIndex` and
  `timestampWrites.endOfPassWriteIndex`.
- Keep dispatch order identical to `BatchMajorVisionTrainer.encode()`.
- Resolve query set to a buffer, map, and compute per-dispatch mean/min/p50.

Fallback if timestamps are unavailable:

- Split-submit per dispatch is too intrusive for absolute timing, but still can
  rank broad classes if each dispatch is repeated many times in isolation.
- Use Phase 2 for kernel ranking and current full benches for whole-pass effect.

Output:

- Sorted table by GPU ms:
  - dispatch index
  - original step index or backward index
  - label
  - shape
  - workgroups
  - mean ms
  - percent of total CLIP train pass
- A grouped table by kernel family:
  - pointwise fwd
  - `pw_bwd`
  - spatial fwd
  - spatial bwd
  - GELU fwd/bwd
  - attention fwd/bwd
  - SE fwd/bwd
  - head/loss

### Phase 4: A/B full-chain candidate kernels

Add an environment switch to `batchTrainDispatches()` or a new wrapper:

```bash
CLIP_BATCH_PW=generic BATCH=3 RUNS=5 WARMUP=5 bun tools/clip/batch_major_train_bench.ts
CLIP_BATCH_PW=shared-w BATCH=3 RUNS=5 WARMUP=5 bun tools/clip/batch_major_train_bench.ts
CLIP_BATCH_PW=allowlist:pw_fwd_256x768,pw_bwd_768x256 BATCH=3 RUNS=5 WARMUP=5 bun tools/clip/batch_major_train_bench.ts
```

Rules:

- Every candidate full-chain path must run exact lane gradient parity first
  (`tools/clip/batch_major_train_bench.ts:155`).
- Only compare warmed runs.
- Record CPU encode+submit too, but judge kernel rewrites on GPU-inclusive wall.
- Do B=2 and B=3 before B=9. B=9 memory pressure can hide kernel effects.

### Phase 5: 3D optimizer wall profile

Current UI profile already splits raster and CLIP through
`Splat3DOptimizer.profileStep()` (`src/splat3d/optimize.ts:150`) and displays it
in `src/splat3d_page.ts:85`. Extend it with two modes:

- `clipBatch=0`: current single-image path.
- `clipBatch=3`: batch-major path, chunking selected views in groups of 3.

Add timings:

- `rasterInput`: raster forward used to produce CLIP inputs.
- `clipBatch`: one batched CLIP train pass per chunk.
- `rasterReplay`: raster forward rerun before backward, if using the minimal
  safe integration.
- `rasterBwd`: raster backward plus chain.
- `copies`: image/text/grad copies if not using offset-bound raster bindings.
- `adam`.

This profile decides whether per-lane raster state is required.

## Wiring Batch-Major CLIP Into the 3D Optimizer

### Minimal safe path

Use a B=3 `BatchMajorVisionTrainer` and chunk selected views. This avoids B=9
allocation by default while still accelerating the common `3/9 views` setting
from `src/splat3d.html:93`.

Pseudo-order for one chunk:

```ts
// 1. Clear accumulated raw parameter gradient once per optimizer step.
raster.recordClearRawGrad(enc);

// 2. Render selected views into CLIP input lanes.
for (let lane = 0; lane < chunk.length; lane++) {
  const view = chunk[lane];
  enc.copyBufferToBuffer(textBuffers[view], 0, batchTrainer.textBuffer,
    batchTrainer.textOffsetBytes(lane), plan.textDim * 4);
  raster.recordForward(enc, view);
  enc.copyBufferToBuffer(raster.image, 0, batchTrainer.inputBuffer,
    batchTrainer.slotOffsetBytes(lane, plan.inputSlot), IMG_BYTES);
}

// 3. One batched CLIP pass.
batchTrainer.encode(enc, { backward: true });

// 4. For correctness with current single-buffer raster state, rerun raster
// forward for each view immediately before its backward.
for (let lane = 0; lane < chunk.length; lane++) {
  const view = chunk[lane];
  raster.recordForward(enc, view);
  enc.copyBufferToBuffer(batchTrainer.inputGradBuffer,
    batchTrainer.inputGradOffsetBytes(lane), raster.gradImage, 0, IMG_BYTES);
  raster.recordBackwardAdd(enc, view);
}

// 5. After all chunks, update params and refresh display.
step_ += 1;
raster.recordAdam(enc, step_, lrs, hyper);
raster.recordForward(enc, displayView);
queue.submit([enc.finish()]);
```

Why the raster replay is required:

- `recordBackwardAdd()` reads `tileCounts`, `binnedIds`, `tileStop`, and
  `derived` from the last `recordForward()` (`src/splat3d/raster.ts:234`).
- Rendering views 0, 1, 2 before CLIP leaves only view 2's state in those
  buffers.
- Rerunning forward before each backward restores the matching state.

Expected tradeoff:

- CLIP work falls from N single-image train passes to ceil(N/B) batched train
  passes.
- Raster forward work doubles for optimized views in the minimal path.
- This is still likely a win if CLIP dominates. The existing profile UI must
  verify on the target GPU.

### Better path after minimal validation

Add offset-bound raster image and grad bindings:

- Render directly into `batchTrainer.inputBuffer` at lane offset.
- Read CLIP lane grad directly from `batchTrainer.inputGradBuffer` at lane
  offset during raster backward.
- This removes image and gradient copies but still needs raster replay unless
  per-lane raster state exists.

Then add per-lane raster forward state:

- Render each view once into lane state and CLIP input lane.
- Run batch CLIP.
- Backward each view against its saved lane state.
- This is the real final shape for multi-view training.

### API changes to keep small

In `src/splat3d/optimize.ts`:

- Change `readonly trainer: VisionTrainer` to either a small interface or
  separate `singleTrainer`/`batchTrainer`.
- Add `clipBatchSize?: number` to `Splat3DOptimizerConfig`.
- In `create()`, build `BatchMajorVisionTrainer.create(device, trainPlan,
  weights, clipBatchSize)` when enabled.
- Keep the current single-image path as fallback.
- Chunk `sampleViews(viewsPerStep)` by `clipBatchSize`.
- Do not allocate B=9 by default. Use B=3 chunks for 5/9 views unless explicit
  profiling proves B=9 is safe and faster.

In `src/splat3d_page.ts`:

- Add a query param or UI mode, e.g. `?clipBatch=0|3|9`.
- Default should be B=3 batch-major only after the parity tool passes.
- For `viewBatchSelect` values larger than `clipBatchSize`, status should say
  `9 views, clip chunks 3`.

In `src/clip/vision_batch.ts`:

- Add `encodeForward()` and `encodeBackward()` to `BatchMajorVisionTrainer` if
  the profiler needs split CLIP timings like `VisionTrainer`.
- Add convenience `inputOffsetBytes(lane)` to avoid callers passing
  `plan.inputSlot`.

In `src/clip/vision.ts`:

- Add `destroy()` to `VisionTrainer`. Current `Splat3DOptimizer.destroy()` only
  destroys raster and text buffers (`src/splat3d/optimize.ts:223`), and
  `VisionTrainer` currently has no `destroy()` method. Batch trainers already
  have destroy methods (`src/clip/vision_batch.ts:554`). This matters more once
  CLIP slots can be hundreds of MB.

## Proposed Patch Sequence

Patch 1 - cleanup and profiling visibility:

- Fix the `SlotBinding` -> `BatchBinding` type typo in
  `src/clip/vision_batch_wgsl.ts:83`.
- Add `destroy()` to `VisionTrainer`.
- Add `tools/clip/plan_profile.mjs` static inventory.
- No behavior change.
- Verify:
  ```bash
  PLAN=plan_train.json FAST=1 BENCH_RUNS=5 bun tools/clip/fused_test.ts
  GATE=1 bun tools/clip/bwd_test.ts
  ```

Patch 2 - per-dispatch benchmark tools:

- Add `tools/clip/dispatch_microbench.ts`.
- Add `tools/clip/dispatch_timestamps.ts` if `timestamp-query` is available
  under bun-webgpu on the target machine.
- Verify that top dispatches align with static expectations before rewriting.

Patch 3 - minimal batch-major 3D optimizer path:

- Import `BatchMajorVisionTrainer` in `src/splat3d/optimize.ts`.
- Add `clipBatchSize` config.
- Implement chunked `step()` path with raster replay before backward.
- Keep current single-image path selectable.
- Add a one-step parity tool, e.g. `tools/splat3d/batch_clip_parity.ts`:
  - create two optimizers with identical seed and params;
  - force the same view list, e.g. `[0,1,2]`;
  - run one old single path and one new batch path;
  - compare params and/or `gradRaw` after the step.
- Verify browser still builds with `--no-scope-hoist`.

Patch 4 - 3D profile mode:

- Extend `Splat3DStepTimings` with batch-specific fields or reinterpret
  `clipFwd + clipBwd` as `clipBatch` for batch mode.
- Update `src/splat3d_page.ts:85` rendering so the UI does not imply split
  CLIP fwd/bwd when the batch path emits one combined pass.
- Profile `3/9`, `5/9`, and `9/9` with B=3 chunks.

Patch 5 - raster direct binding:

- Add raster methods that bind `image` and `gradImage` to external buffers with
  storage offsets.
- Replace image and grad copies in the batch path.
- Verify one-step parity again.

Patch 6 - per-lane raster state:

- Add `Raster3DFrameState` allocation for `batchCap`.
- Add stateful forward/backward record methods.
- Remove duplicate raster replay from the batch path.
- Verify parity and profile again. This is the first "final architecture"
  version.

Patch 7 - selective shared-W pointwise:

- Convert `vision_batch_pointwise.ts` from compact benchmark emitters into
  reusable full-plan emitters.
- Add forward shared-W allowlist support in `batchForwardDispatches()`.
- Add backward shared-W allowlist support in `batchTrainDispatches()`.
- Start with all allowlists off by default.
- Enable only measured winners for B=2/B=3.

Patch 8 - train GELU fusion:

- Add a compiler/codegen representation for "write pre and post activation in
  the producer dispatch".
- Update per-step verification mapping.
- Only do this if per-dispatch profile shows GELU dispatch/memory cost is worth
  the test churn.

## Correctness Gates

CLIP-only gates:

- `PLAN=plan_train.json bun tools/clip/fused_test.ts`
- `bun tools/clip/bwd_test.ts`
- `BATCH=3 RUNS=2 WARMUP=3 bun tools/clip/batch_major_train_bench.ts`
- `BATCH=9 RUNS=2 WARMUP=2 bun tools/clip/batch_major_train_bench.ts`

3D gates to add:

- One-step old-vs-new parity for fixed params and fixed selected views.
- Batch path with raster replay should match old path closely because each
  per-view raster backward is still run in the same order.
- Batch path with per-lane raster state should match the replay path.
- A page smoke should verify:
  - WebGPU boot.
  - text model load.
  - optimize starts.
  - cosine rises from initial value.
  - all 9 canvases remain nonblank.

Performance gates:

- Batch-major B=3 must beat current single-image 3-view step after including
  any raster replay.
- If raster replay makes total step time flat or worse, do not ship it as
  default. Move directly to per-lane raster state.
- Shared-W pointwise must improve full CLIP train wall time, not just one
  isolated microbench.

## Specific Risks

- Memory: B=9 train slots are about 1.67 GB before raster and browser overhead.
  Avoid B=9 default allocation.
- Raster state correctness: do not batch CLIP across multiple rendered views and
  then call `recordBackwardAdd()` without restoring the matching view state.
- Profile distortion: split-submit CPU wall profiles are useful but not enough
  for kernel allowlists. Use GPU timestamps or isolated dispatch microbenches.
- Type checking: fix `SlotBinding` typo before adding more batch WGSL code.
- Resource cleanup: add `VisionTrainer.destroy()` before repeated optimizer
  resets with batch trainers.
- Dynamic storage offsets: binding offsets must satisfy the device's storage
  alignment. The 256x256 RGB image lane is 786432 bytes, so it is 256-byte
  aligned; still assert this before relying on external buffer bindings.

## Recommended First Implementation Choice

Start with B=3 chunked `BatchMajorVisionTrainer` in `Splat3DOptimizer`, with
raster replay before backward. It is the smallest correct app integration and
will answer the key question: does CLIP batching still win when embedded in the
real 3D step?

Do not start by integrating shared-W pointwise. The shared-W microbench has
mixed results, while the generic batch-major train path already has exact
parity and large CLIP-only wins. The right order is app integration first,
measurement second, selective kernel rewrites third.
