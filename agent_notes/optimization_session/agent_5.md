# Worker 5 CLIP Optimization Notes From Dynaworld Fast-Mac-Gsplat

Date: 2026-07-08

Scope:

- Studied `/Users/nicholasbardy/git/gsplats_browser/dynaworld`, with emphasis on `third_party/fast-mac-gsplat` Metal variants.
- Translated only relevant GPU-programming lessons to the CLIP forward/backward path in `/Users/nicholasbardy/git/Neural-Force-Field-Art/src/clip`.
- Checked current 3D raster and CLIP interaction in `/Users/nicholasbardy/git/Neural-Force-Field-Art/src/splat3d`.
- Did not modify app code. This note is the only written artifact for worker 5.

## Executive Read

The current app has already done the obvious high-level thing: fused MobileCLIP is GPU-resident, forward/backward is encoded as fixed dispatch lists, and 3D splat training samples N of K camera views. The remaining large win is not a generic raster rewrite. The measured bottleneck is still CLIP: the 3/9 sampled optimizer profile spends about 59.03 ms of a 75.57 ms split-profile step in CLIP forward/backward, and 9/9 spends about 180.58 ms of 228.20 ms in CLIP. See `docs/SPLAT3D_PERF_NOTES.md:22-35`.

The most actionable idea is to integrate `BatchMajorVisionTrainer` into `Splat3DOptimizer`, but the integration is not as simple as "render all views, run CLIP batch, then run all raster backwards." The current raster backward depends on forward-state buffers (`derived`, `tileCounts`, `binnedIds`, `tileStop`) that are overwritten by each raster forward. The current loop keeps correctness by running raster forward, CLIP, and raster backward immediately for each view (`src/splat3d/optimize.ts:131-143`; `src/splat3d/raster.ts:215-248`). A batch CLIP integration must either:

1. Re-render each view before raster backward after batched CLIP, or
2. Store per-lane raster forward state, or
3. Make raster itself batch-major over views.

The conservative production path is option 1. It adds an extra raster forward per sampled view but can still be a net win because raster forward is much smaller than CLIP. It also avoids a large raster-state memory and API change.

The dynaworld lesson that maps best to this project is not a specific Metal kernel. It is the scheduling discipline: keep dense GPU work in fixed, shape-specialized dispatch lists; preserve exactly the forward state that backward needs; tune tile/threadgroup constants empirically; and refuse to promote clever reductions without parity and timing gates. The star UVT rows show that a theoretically more controlled reduction path can be slower than direct atomics (`dynaworld/BASELINES.md:95-108`; `star_uvt_v0/research_project/benchmarks/README.md:215-258`).

## Current Local Surface

### CLIP Runtime

The CLIP runtime is already highly specialized:

- `src/clip/vision.ts:1-15`: custom fused MobileCLIP runtime, no tfjs coupling.
- `src/clip/vision.ts:71-82`: one packed weights buffer and one storage buffer per activation slot.
- `src/clip/vision.ts:132-143`: full forward encoded into one compute pass.
- `src/clip/vision.ts:189-197`: `VisionTrainer` owns activation and gradient slots, packed weights, text embedding, and produces `dL/dpixels` with frozen weights.
- `src/clip/vision.ts:217-220`: trainer dispatch list is forward plus backward.
- `src/clip/vision.ts:295-305`: forward, loss head, and backward encoded into one compute pass.
- `src/clip/vision.ts:308-329`: split forward/backward encoders exist for profiling and staged execution.

The generated forward WGSL has the right broad shape:

- `src/clip/vision_wgsl.ts:1-26`: ONNX plan is compiled offline; shapes and offsets are baked into shader source; no structural uniforms.
- `src/clip/vision_wgsl.ts:155-162`: pointwise conv is the FLOP bulk.
- `src/clip/vision_wgsl.ts:165-218`: pointwise kernel stages X and W tiles in workgroup memory.
- `src/clip/vision_wgsl.ts:274-374`: spatial conv stages weights once per output channel workgroup and unrolls interior taps.
- `src/clip/vision_wgsl.ts:435-532`: attention stages K/V in workgroup memory and avoids materializing the score matrix.
- `src/clip/vision_wgsl.ts:581-603`: train-mode GELU is a standalone dispatch because pre-activation must be saved for backward.

The backward WGSL is correctness-first but not naive:

- `src/clip/vision_bwd_wgsl.ts:1-20`: pure backward codegen with frozen weights, no dW, no global grad zero-fill, and saved activations.
- `src/clip/vision_bwd_wgsl.ts:140-162`: `pw_bwd` reuses the forward pointwise tiled matmul over transposed weights.
- `src/clip/vision_bwd_wgsl.ts:224-280`: `spatial_bwd` is gather-form conv backward, one thread per input pixel. This is the most obvious kernel-level follow-up because it does not stage weights like forward spatial conv does.
- `src/clip/vision_bwd_wgsl.ts:289-369`: SE backward recomputes gate internals in one workgroup.
- `src/clip/vision_bwd_wgsl.ts:415-464`: loss backward computes `-cos(embed,text)` with one workgroup reduction.
- `src/clip/vision_bwd_wgsl.ts:475-571`: attention backward recomputes softmax rows and avoids `nTok x nTok` storage.

Batch-major CLIP already exists:

- `src/clip/vision_batch_wgsl.ts:1-10`: batch-major fork offsets slot/text buffers by `workgroup_id.z`.
- `src/clip/vision_batch_wgsl.ts:77-108`: slot and text accesses are rewritten with a per-lane base.
- `src/clip/vision_batch_wgsl.ts:110-146`: forward and train dispatches set `workgroups.z = batch`.
- `src/clip/vision_batch.ts:393-459`: `BatchMajorVisionTrainer` allocates activation slots and text as `[batch][slotFloats]`.
- `src/clip/vision_batch.ts:493-508`: helpers expose lane offsets for activation, input grad, and text.
- `src/clip/vision_batch.ts:536-545`: batch train path encodes the same fixed dispatch list once.

Batch-major measured well in isolation:

- `docs/CLIP_BATCHING_NOTES.md:125-171`: batch-major forward+backward exists, exact `dL/dpixels` parity passed for B=2, B=3, and B=9. Representative results: B=3 separate 102.55 ms/batch vs batch-major 49.48 ms/batch; B=9 separate 389.66 ms/batch vs batch-major 134.63 ms/batch.
- `docs/CLIP_BATCHING_NOTES.md:173-224`: shared-W pointwise microkernel verifies exactly but is not a blanket win. B=2 expansion layers modestly improved; B=3 often lost due occupancy/workgroup memory pressure.

### Current Raster/CLIP Interaction

The 3D optimizer currently performs per-view raster/CLIP/backward in one command buffer:

- `src/splat3d/optimize.ts:131-143`: for each sampled view, copy text, raster forward, copy raster image into CLIP input, run CLIP forward+backward, copy CLIP input gradient into raster grad image, run raster backward.
- `src/splat3d/optimize.ts:150-198`: split-submit profiler times raster forward, CLIP forward, CLIP backward, raster backward, Adam, and display with `onSubmittedWorkDone`.
- `src/splat3d/optimize.ts:241-263`: N-of-K sampled views use shuffled epochs, so 3/9 covers all cameras over three optimizer steps.

The raster state is mutable and view-specific:

- `src/splat3d/raster.ts:112-123`: raster buffers include `derived`, `tileCounts`, `binnedIds`, `tileStop`, `image`, and `gradImage`.
- `src/splat3d/raster.ts:142-153`: forward and backward bind the same mutable tile/state buffers.
- `src/splat3d/raster.ts:215-232`: `recordForward` prepares derived data, clears bins, emits splat IDs, renders image, and writes `tileStop`.
- `src/splat3d/raster.ts:234-248`: `recordBackwardAdd` consumes `gradImage`, `tileCounts`, `binnedIds`, `tileStop`, and `derived`, then chains to raw parameter grads.

This means view batching must preserve or rebuild raster forward state. Batch-major CLIP alone is not a sufficient scheduling change.

## Dynaworld Lessons That Matter

### Lesson 1: Compile-Time Shape Specialization Is Valuable, But Variant Promotion Must Be Empirical

The star UVT Metal path uses compile-time tile and thread constants:

- `star_uvt_v0/csrc/metal/star_uvt_kernels.metal:4-36`: default tile, thread, fixed-point, and feature-cache constants.
- `star_uvt_v0/csrc/metal/star_uvt_metal.mm:39-59`: env-configurable tile sizes and capacities with validation.
- `star_uvt_v0/csrc/metal/star_uvt_metal.mm:72-83`: host injects the chosen constants as a shader preamble.

This maps well to CLIP because CLIP already bakes shapes and offsets in `vision_wgsl.ts:1-26`. The next CLIP variants should stay generated and shape-gated, not runtime-branchy. Examples: f16 weights only on adapters with `shader-f16`; shared-W pointwise only for shapes that benchmark as wins; alternate `spatial_bwd` only for layers where timestamps justify it.

The caution is important. Dynaworld baseline rows show "more structured" was not always faster:

- `dynaworld/BASELINES.md:104-108`: feature grad cache was a small win over direct fixedbin, but the reduced-gradient variant was slower and explicitly marked "do not promote."
- `star_uvt_v0/research_project/benchmarks/README.md:215-258`: tile-pair deterministic sample/reduce had parity and pruning, but direct atomic remained the local speed target.

CLIP should follow the same rule: every shader variant gets parity plus per-shape timing before it replaces the default path.

### Lesson 2: Preserve the Forward State Backward Actually Needs

Dynaworld v7.2 fixed a scaling wall by replacing all-splat per-pixel scans with sparse tile bins:

- `variants/v7_tiled_capture/ENGINEERING_NOTES.md:5-10`: v7.1 rebuilt front-K state with `H*W` threads where every pixel scanned all Gaussians.
- `variants/v7_tiled_capture/ENGINEERING_NOTES.md:14-23`: v7.2 bins contributors by tile and both capture and overflow replay scan only local tile lists.
- `variants/v7_tiled_capture/ENGINEERING_NOTES.md:31-41`: tile bins change the candidate set, not the pixel-local visibility criterion, preserving exactness.
- `variants/v7_hybrid_v5style/README.md:64-80`: complexity moved from `O(B * H * W * G)` to tile-reference plus per-pixel local-bin complexity.

For Neural-Force-Field-Art this is directly relevant to the CLIP batching schedule. Raster backward needs the exact forward bin/order/stop state. If we batch CLIP by rendering all selected views first, later raster forwards overwrite the state needed by earlier views. A correct integration must either re-render the view before backward, save per-view state, or make raster state batch-major.

### Lesson 3: Local Memory Staging And Reductions Are Useful, But Only Where The Data Reuse Exists

Star UVT uses threadgroup sorting, local candidate arrays, and reductions:

- `star_uvt_v0/csrc/metal/star_uvt_kernels.metal:872-903`: threadgroup bitonic sort by depth.
- `star_uvt_v0/csrc/metal/star_uvt_kernels.metal:923-933`: reusable threadgroup sum reduction.
- `star_uvt_v0/csrc/metal/star_uvt_kernels.metal:935-1018`: feature-gradient reductions use SIMD sums, partials in threadgroup memory, and fewer atomics.
- `variants/v9_features_gradcache_zero_bg/csrc/metal/gsplat_v9_features_gradcache_zero_bg_kernels.metal:153-181`: per-pixel `grad_features[pix,:]` cache avoids reloading feature gradients.
- `variants/v9_features_gradcache_zero_bg/csrc/metal/gsplat_v9_features_gradcache_zero_bg_kernels.metal:184-248`: cached feature-grad reductions use SIMD sums and threadgroup partials.

For CLIP, this maps to:

- Existing pointwise forward and `pw_bwd`, which already stage X/W tiles.
- `spatial_bwd`, which currently gathers weights from global inside nested loops and is the cleanest staging target.
- Attention kernels, which already avoid materializing score matrices and stage K/V.

It does not map to copying tile-pair sample emission/reduction into CLIP. CLIP has dense tensors and mostly one-writer gradient slots; the raster-style sparse primitive aggregation problem is absent.

### Lesson 4: Fusing Loss With Backward Can Remove Dense Materialization, But CLIP's Loss Is Global

Dynaworld v12c fuses raster/color/loss backward to avoid materializing a dense feature-gradient image:

- `variants/v12c_fused_raster_color_loss_backward/README.md:1-7`: computes colorize, RGB composition, and mean-MSE pixel gradients inside raster backward without dense `grad_features[B,H,W,F]`.
- `variants/v12c_fused_raster_color_loss_backward/README.md:23-44`: this is explicitly a prototype for a constrained loss and feature dimension.

For this app, the analogous dense artifact is the CLIP image gradient copied into `raster.gradImage` (`src/splat3d/optimize.ts:139-142`). But CLIP's loss is not local per pixel; the embedding is global. We cannot fuse raster backward directly with CLIP loss without running the full CLIP backward to produce `dL/dpixels`. The direct transferable idea is buffer aliasing: make CLIP write its final input gradient into the same GPUBuffer that raster backward consumes, or allow raster backward to read from the batch lane offset. This removes copies, not the CLIP backward itself.

## Directly Applicable CLIP Ideas

### 1. Integrate Batch-Major CLIP, But Use A State-Safe Raster Schedule

Status: highest-value production candidate.

Why:

- Batch-major trainer already exists and verifies exact lane gradients (`docs/CLIP_BATCHING_NOTES.md:148-153`).
- Isolated timing suggests B=3 and B=9 are meaningful wins (`docs/CLIP_BATCHING_NOTES.md:155-168`).
- Current integrated profiles show CLIP dominates sampled wall time (`docs/SPLAT3D_PERF_NOTES.md:22-35`).

Conservative schedule:

1. Sample views as today.
2. Clear raw splat grads once.
3. For each lane/view:
   - Raster forward that view.
   - Copy `raster.image` into `BatchMajorVisionTrainer.inputBuffer` at `slotOffsetBytes(lane, inputSlot)`.
   - Copy that view's text embedding into `BatchMajorVisionTrainer.textBuffer` at `textOffsetBytes(lane)`.
4. Run `BatchMajorVisionTrainer.encode(..., { backward: true })` once.
5. For each lane/view:
   - Re-run raster forward for that view to rebuild `derived`, `tileCounts`, `binnedIds`, and `tileStop`.
   - Copy `inputGradBuffer` lane at `inputGradOffsetBytes(lane)` into `raster.gradImage`.
   - Run `raster.recordBackwardAdd(view)`.
6. Adam step.
7. Display render.

Why the re-render is intentional:

- The current raster forward writes the mutable state that backward consumes (`src/splat3d/raster.ts:215-248`).
- Re-rendering adds raster forward work, but the current 3/9 raster forward is only 2.32 ms in the split profile while CLIP is 59.03 ms (`docs/SPLAT3D_PERF_NOTES.md:24-28`).
- This mirrors the dynaworld state lesson: avoid invalidating the forward state that backward depends on.

More ambitious follow-ups:

- Save per-view raster state buffers for `derived`, `tileCounts`, `binnedIds`, and `tileStop` so no re-render is needed.
- Make raster itself view-batch-major, with per-lane tile bins and images.
- Extend raster backward to read CLIP gradients from a lane offset in a batch gradient buffer, eliminating per-lane gradient copy.

Correctness gates:

- Compare each batched lane `dL/dpixels` against existing single-view `VisionTrainer`.
- Compare final `raster.gradRaw` after one fixed sampled step against the existing per-view loop.
- Compare one Adam-updated parameter buffer after one step.

Benchmark gates:

- `viewsPerStep = 3` and `viewsPerStep = 9`.
- Current loop vs batch-major with re-render vs batch-major with saved raster state if built.
- Report normal single-submit step avg and split-stage totals, not just isolated CLIP.

### 2. Add Per-Dispatch CLIP Timing Before Rewriting Kernels

Status: necessary instrumentation before shader surgery.

The current profiler can tell "CLIP forward" versus "CLIP backward", but not which dispatch labels dominate (`src/splat3d/optimize.ts:150-198`). Local notes already list the next measurements: per-dispatch timing by label/shape for B=1, B=3, and B=9; pointwise forward/backward; stem `spatial_bwd`; GELU share; attention only if above threshold (`docs/SPLAT3D_PERF_NOTES.md:51-67`).

Implementation direction:

- Expose dispatch labels and workgroups from `VisionTrainer` and `BatchMajorVisionTrainer` behind a debug/profile API.
- Use WebGPU timestamp queries where available. If not available in the browser target, build a Bun/WebGPU harness that groups dispatches by label class and fences with readback or `onSubmittedWorkDone`.
- Warm pipelines before timing. Dynaworld explicitly separates timing evidence from quality metrics and has dedicated train-step timing probes (`star_uvt_v0/research_project/benchmarks/README.md:182-190`). Local CLIP docs also warn to warm up before measurements (`tools/clip/README.md`, see backward/results and gotchas sections).

Profile groups:

- `pw` forward by `(cin,cout,H,W)` from `src/clip/vision_wgsl.ts:231-268`.
- `pw_bwd` by `(cin,cout,H,W)` from `src/clip/vision_bwd_wgsl.ts:140-162`.
- `spatial_bwd` by kernel, stride, and groups from `src/clip/vision_bwd_wgsl.ts:224-280`.
- `gelu` and `gelu_bwd` aggregate from `src/clip/vision_wgsl.ts:581-603` and `src/clip/vision_bwd_wgsl.ts:193-216`.
- `attn_core` and `attn_core_bwd` only if visible from `src/clip/vision_wgsl.ts:442-532` and `src/clip/vision_bwd_wgsl.ts:475-571`.

Do not start with a rewrite. Get the shape histogram and timings first.

### 3. Stage Weights In `spatial_bwd`

Status: best kernel-level candidate after batch integration.

Current:

- Forward spatial conv stages a small weight footprint per output channel and unrolls the interior fast path (`src/clip/vision_wgsl.ts:274-374`).
- Backward spatial conv is a gather kernel: one workgroup per input channel, one thread per input pixel, loops output channels and taps, reading weights with `W(...)` inside the nested loops (`src/clip/vision_bwd_wgsl.ts:224-280`).

Idea:

- For each `spatial_bwd` layer, stage a block of output-channel weights for a fixed input channel/group into workgroup memory.
- Let one workgroup cover a tile of input pixels and a small output-channel block. Accumulate partial sums either within one workgroup or in multiple passes depending on `cpgOut`.
- Specialize depthwise separately (`groups == cin`, `cpgOut == 1`) because it has no output-channel fan-in and may not benefit from the same staging.

Expected benefit:

- General stem-like layers with `cin=3` and larger `cout` may benefit most.
- Depthwise layers may be bandwidth/latency limited differently; do not assume win.

Risk:

- More workgroup memory can reduce occupancy. Dynaworld shared reductions and fixed tile constants show this tradeoff repeatedly (`star_uvt_v0/csrc/metal/star_uvt_kernels.metal:923-1018`; `star_uvt_v0/csrc/metal/star_uvt_metal.mm:39-59`).
- A staged version should be layer-gated by timing, not globally substituted.

Benchmark:

- Per-layer `spatial_bwd` timestamp before/after.
- Full `VisionTrainer` forward+backward wall, because a faster isolated kernel can still lose through occupancy or scheduling side effects.
- Pixel-gradient parity against current backward.

### 4. Use Shared-W Batch Pointwise Selectively

Status: valid but shape-gated.

Existing experiment:

- `src/clip/vision_batch_pointwise.ts:1-8`: shared-W pointwise puts batch lanes in `local_invocation_id.z` and stages W once.
- `src/clip/vision_batch_pointwise.ts:104-171`: implementation supports B=1..3 and dispatches one workgroup across batch lanes.
- `docs/CLIP_BATCHING_NOTES.md:173-224`: exact parity passed; B=2 expansion layers had modest wins, B=3 often lost, contraction/residual layers were mixed.

Direct plan:

- Use per-dispatch profile to find the top pointwise shapes in batch-major train.
- Replace only shapes where shared-W wins in the full CLIP context.
- Consider B=2 as a special case. For a 3-view optimizer step, a `2 + 1` CLIP schedule could beat one B=3 shared-W schedule if B=3 occupancy pressure dominates.

Why shape-gated:

- B=3 uses 12 KB X plus 4 KB W, hitting the 16 KB workgroup-memory budget described in `docs/CLIP_BATCHING_NOTES.md:180-185`.
- The dynaworld reduced-gradient negative row is the same category of lesson: an apparently smarter reduction can be slower (`dynaworld/BASELINES.md:104-108`).

### 5. Pre-Normalize Text Embeddings For The Cosine Loss

Status: small, clean, low-risk CLIP backward optimization.

Current `loss_bwd` computes three reductions: `sum(e^2)`, `sum(t^2)`, and `sum(e*t)`, then computes `sqrt` for both norms (`src/clip/vision_bwd_wgsl.ts:415-464`). Text embeddings are prompt-specific but constant across many steps. The app already stores one text buffer per camera (`src/splat3d/optimize.ts:109-128`).

Idea:

- Normalize each text embedding on upload and store `t_norm = t / |t|`.
- Change loss backward to reduce only `sum(e^2)` and `sum(e*t_norm)`.
- Formula:
  - `cos = dot(e, t_norm) / |e|`
  - `d(-cos)/de = -(t_norm / |e| - cos * e / |e|^2)`

Expected benefit:

- Small absolute gain, because loss is one dispatch. But it removes one reduction array, one norm, and text-norm work from every CLIP backward lane.
- It also makes batch-major text lanes lighter.

Correctness:

- Compare `dL/dembed` and `dL/dpixels` with current implementation for random text vectors.
- Need exact handling for zero-norm text, though real CLIP embeddings should not be zero.

### 6. Alias Or Externalize Raster Image/Gradient Buffers

Status: direct interaction optimization, especially after batch CLIP.

Current copies:

- Raster image to CLIP input: `src/splat3d/optimize.ts:138-140`.
- CLIP input grad to raster grad image: `src/splat3d/optimize.ts:140-142`.
- Profile path includes the same copies around forward/backward (`src/splat3d/optimize.ts:172-187`).

Single-view direct approach:

- Allow `VisionTrainer` to accept external buffers for `inputSlot` and `inputGradSlot`.
- Bind `raster.image` as CLIP input and `raster.gradImage` as CLIP input grad, subject to usage flags and pass ordering.
- `raster.image` is already storage plus copy source, and `raster.gradImage` is storage plus copy destination (`src/splat3d/raster.ts:121-122`), but constructor/API changes would be needed to bind them as CLIP slots.

Batch-major approach:

- First version can keep copies into and out of batch lanes because it is simpler.
- Next version should allow raster backward to read a lane offset in `BatchMajorVisionTrainer.inputGradBuffer`, or make `raster.gradImage` a view into a lane-like external buffer if WebGPU binding offsets and alignment make this practical.

This does not eliminate CLIP work, but it removes two full-image GPU copies per view. At 256x256x3xf32, each image/grad copy is 786,432 bytes, so a 3-view step copies about 4.5 MB just across this boundary. That is not the main bottleneck today, but it becomes more visible after CLIP batching.

### 7. F16 Weights First, F16 Activations Later

Status: ambitious, must be feature-gated.

Current weights and activations are f32. MobileCLIP weights are frozen, so weight bandwidth is an obvious target. The safe sequence is:

1. Pack weights as f16 where `shader-f16` is available.
2. Accumulate in f32 for pointwise and spatial kernels.
3. Keep activations and gradients f32 initially.
4. Only then test f16 activation slots for selected forward-only or backward sections.

Why weights first:

- Weight storage is read-only and shared across lanes.
- A bad f16 activation choice can perturb `dL/dpixels` and optimizer behavior more directly.

Risks:

- WebGPU `shader-f16` is optional and not guaranteed on all browser/adapter combinations.
- Mixed f16/f32 codegen doubles variant count.
- `spatial_bwd`, attention, and cosine gradients may be sensitive enough that full f16 activation is not acceptable.

Gates:

- Adapter feature gate.
- Forward embedding cosine vs f32.
- Full `dL/dpixels` cosine and relative max error.
- 100-step optimization trajectory comparison at fixed seed and view schedule.

### 8. Fuse Train-Mode GELU/Pointwise Epilogues Only If Timestamps Justify It

Status: secondary.

Train-mode splits GELU so pre-activation is saved for backward (`src/clip/vision_wgsl.ts:77-80`; `src/clip/vision_wgsl.ts:581-603`). Backward has separate `gelu_bwd` dispatches (`src/clip/vision_bwd_wgsl.ts:193-216`). Fusing can reduce dispatches and slot traffic, but the fused forward still needs to preserve pre-activation for backward.

Possible variants:

- Forward pointwise writes both pre-activation and post-GELU in one dispatch when the plan needs both.
- Backward fuses `gelu_bwd` with an immediately following `pw_bwd` if gradient slot lifetimes allow it.
- Residual add/layer-scale remains in the pointwise epilogue where it already is for forward (`src/clip/vision_wgsl.ts:244-261`).

Risk:

- Slot lifetime and accumulation ordering are easy to break.
- If CPU dispatch overhead is already tiny, fusing only helps if memory traffic or GPU scheduling barriers show up in timestamps.

This should follow per-dispatch timing, not precede it.

### 9. Tighten Train-Plan Slot Memory And Liveness

Status: useful if batch-major memory pressure blocks bigger batches or mobile adapters.

Current:

- `VisionTrainer` allocates activation and grad slots for the full train plan (`src/clip/vision.ts:189-197`, `src/clip/vision.ts:242-248`).
- `BatchMajorVisionTrainer` multiplies each slot by batch (`src/clip/vision_batch.ts:453-459`).
- Local notes already identify memory tightening as a next iteration (`docs/CLIP_BATCHING_NOTES.md:225-239`).

Ideas:

- Use plan liveness to alias grad slots whose lifetimes do not overlap.
- Split forward-only and train-plan batches so preview embedding calls do not carry backward slots.
- Add a memory report per plan: weights, activation slots, grad slots, text, and batch multiplier.

This does not directly improve GPU time unless memory pressure hurts occupancy or causes allocation failures, but it makes B=9 or f16/f32 variants easier to test.

### 10. Keep N-of-K Scheduling, But Benchmark It With Batch CLIP

Status: already useful, still compatible.

N-of-K view sampling is a real lever:

- 9/9 normal step avg: 205.26 ms.
- 3/9 normal step avg: 69.94 ms.
- 3/9 covers all 9 views over three optimizer steps via shuffled epochs.

See `docs/SPLAT3D_PERF_NOTES.md:22-35` and `src/splat3d/optimize.ts:241-263`.

Batch CLIP changes the tradeoff. If B=9 becomes cheap enough, full-view steps may be reasonable for periodic stabilization, evaluation, or screenshot generation. But default interactive training should likely stay 3/9 until quality measurements prove otherwise.

## Raster-Only Ideas Not To Port Literally To CLIP

These are useful for the rasterizer or for thinking about scheduling, but they should not be copied into CLIP kernels directly.

### Sparse Tile Bins And UVT Tile Capacity

Raster:

- `star_uvt_v0/csrc/metal/star_uvt_kernels.metal:1784-1823`: bins tubes into UVT tiles using tile counts, fixed capacity, and overflow flags.
- `star_uvt_v0/csrc/metal/star_uvt_kernels.metal:1874-1952`: render loads per-tile candidate IDs/depths, sorts locally, and composites per sample.
- `variants/v7_hybrid_v5style/README.md:64-80`: complexity win comes from replacing all-splats scans with local tile bins.

CLIP:

- MobileCLIP is dense over channels and pixels. There is no sparse primitive candidate list analogous to splats/tubes.
- Trying to invent sparse pixel skipping for CLIP would change the model/loss unless it is a separate approximation project.

Transferable lesson only: preserve local forward state and avoid global scans when the mathematical dependency is local. CLIP's dependency is mostly dense.

### Tile-Pair Sample Emission Plus Reduction

Raster:

- `star_uvt_v0/csrc/metal/star_uvt_kernels.metal:5846-6182`: tile-pair backward emits compact per tile-slot sample gradient rows and marks rows with gradients.
- `star_uvt_v0/csrc/metal/star_uvt_kernels.metal:6184-6407`: parallel tile-pair backward maps tile pixels to threads and reduces per-row gradients in a workgroup.
- `star_uvt_v0/csrc/metal/star_uvt_metal.mm:3285-3409`: host path clears/bins/emits sample rows, then reduces via bounds scan or parallel bounds scan.

CLIP:

- Backward does not aggregate sparse primitive gradients. It propagates dense tensor gradients through a fixed graph, with one writer or controlled accumulation into slots.
- The closest CLIP analogue is not sample emission; it is per-dispatch reduction in `loss_bwd`, SE, head, and attention. Those are already small dense reductions.

Transferable lesson only: compact zero rows and avoid dense materialization when the downstream consumer can operate on sparse rows. Current CLIP has no such sparse downstream consumer.

### Fixed-Point Atomics And Deterministic Keyed Reductions

Raster:

- Star UVT has direct atomic, fixed-point, split fixed-point, tile-pair, and keyed/segmented reduction paths (`star_uvt_v0/csrc/metal/star_uvt_metal.mm:86-142`).
- Deterministic paths improve repeatability but often cost speed (`star_uvt_v0/research_project/benchmarks/README.md:197-224` and `:252-258`).

CLIP:

- CLIP backward for frozen weights does not need global atomics for dW.
- Accumulation order is mostly fixed by dispatch and slot writes.

Transferable lesson only: if future CLIP variants introduce atomics, expose deterministic and fast modes separately and benchmark both. Do not add fixed-point machinery to current CLIP backward.

### Temporal Tile T And Tube/Frame Scheduling

Raster:

- Star UVT is 3D over x/y/time with `STAR_TILE_T` and tube bounds (`star_uvt_v0/csrc/metal/star_uvt_kernels.metal:4-15`, `:835-869`).
- Fixed-tube scale probes show raster-specific sparse/sublinear behavior as target pixels grow (`star_uvt_v0/research_project/benchmarks/README.md:236-258`).

CLIP:

- Current CLIP input is a single 256x256 image. There is no temporal tile dimension.
- Batch lanes are independent image views, not neighboring time slices with shared tube state.

Transferable lesson only: z-dimension dispatch is useful for batch lanes, which `BatchMajorVisionTrainer` already uses.

### Target-Grid And Sparse-Grid Feature Objectives

Raster/dynaworld:

- Several STAR UVT rows reduce target/loss work by moving to target-grid or sparse-grid VJP objectives (`dynaworld/BASELINES.md:109-125`).

CLIP:

- CLIP embedding is the objective. Replacing it with a target-grid feature objective would be a product/model change, not a CLIP optimization.
- Patch or sparse CLIP approximations could be a research path, but they change the loss surface and need quality gates, not just speed gates.

## Benchmark Methodology

### General Rules

Use the dynaworld discipline: separate speed evidence from quality evidence. A microkernel timing row is not a training-quality row. A split-submit browser profile is not exact GPU timestamp attribution. A cold first run is not a warmed kernel timing.

Report:

- Date, machine, browser/runtime, GPU adapter, adapter features, and whether `shader-f16` and timestamp queries are available.
- Plan file (`plan_train.json` or other), model weight hash if available, batch size, view count, splat count, image size, and camera count.
- Warmup count, run count, median, p25, p75, min, max.
- Whether timing is normal single-submit wall, split-submit wall, timestamp-query GPU time, or readback-fenced harness time.

### Isolated CLIP Baseline

Use existing CLIP harnesses before app integration:

```bash
BATCH=3 RUNS=5 WARMUP=5 bun tools/clip/batch_major_train_bench.ts
BATCH=9 RUNS=5 WARMUP=5 bun tools/clip/batch_major_train_bench.ts
BATCH=2 STEP_INDEX=57 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCH=3 STEP_INDEX=57 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
```

For any codegen or plan change, rerun the existing correctness suite described in `tools/clip/README.md`:

```bash
uv run --with onnx --with numpy python tools/clip/compile_plan.py --train
PLAN=plan_train.json bun tools/clip/fused_test.ts
PLAN=plan_train.json bun tools/clip/bwd_test.ts
```

Use isolated harnesses for:

- `VisionTrainer` B=1 forward+backward.
- `BatchMajorVisionTrainer` B=2/B=3/B=5/B=9.
- Shared-W pointwise per candidate step.
- Any f16 or staged `spatial_bwd` variant.

Correctness gates:

- Forward embedding cosine: target 1.0 for exact variants; explicit tolerance for f16.
- Full `dL/dpixels` lane parity: cosine near 1.0 and relative max error tracked.
- For f16: also compare sign agreement and top-percentile gradient error, not only mean error.

### Integrated 3D Optimizer Benchmark

Use both normal step wall and split profile:

- Normal `step()` average over at least 50 warmed steps.
- `profileStep()` split profile every fixed interval for stage attribution (`src/splat3d/optimize.ts:150-198`).
- `viewsPerStep = 3` and `viewsPerStep = 9`.
- Fixed seed, fixed prompt embeddings, same view schedule.

Variants to compare:

1. Current per-view CLIP loop.
2. Batch-major CLIP with re-render before raster backward.
3. Batch-major CLIP with saved per-view raster state, if implemented.
4. Batch-major CLIP plus buffer aliasing or lane-offset raster grad read, if implemented.
5. Selective shared-W pointwise variants, if timestamps justify them.
6. F16 weights, if adapter supports it.

Integrated correctness gates:

- For one fixed optimizer step, compare current loop vs candidate:
  - Per-lane `dL/dpixels`.
  - Final `raster.gradRaw`.
  - Adam-updated `params`.
- For trajectory:
  - Same seed and view schedule for 100 steps.
  - Compare loss/embedding movement and rendered previews.
  - Report if outputs diverge due acceptable floating-order differences or f16 approximation.

### Per-Dispatch CLIP Timestamp Method

Preferred:

- Use `GPUQuerySet` timestamp queries around groups of dispatches if available.
- Group by label and shape, not only by source file.
- Measure B=1 and batch-major B=3/B=9 separately.

Fallback:

- Build a Bun/WebGPU profiling harness that runs repeated grouped dispatch slices with a readback fence.
- Avoid per-dispatch `onSubmittedWorkDone` inside the browser app unless it is explicitly a debug mode; it changes scheduling too much.

Labels to report:

- Top 10 forward dispatch labels by GPU time.
- Top 10 backward dispatch labels by GPU time.
- Aggregate by kind: pointwise, `pw_bwd`, spatial, `spatial_bwd`, GELU, SE, attention, head/loss.
- Dispatch count and total workgroups for each group.

### Quality/Promotion Rule

Promote an optimization only if it passes both:

- Speed: integrated optimizer wall improves on the target setting by a meaningful amount, not just an isolated microkernel.
- Correctness/quality: exact variants match current gradients and params within tolerance; approximate variants have an explicit quality row.

Use dynaworld's negative rows as a model. If a variant is slower or only moves work around, keep it as a note or debug harness, not a default path.

## Suggested Experiment Queue

1. Add a CLIP dispatch profiler.
   - Output CSV/JSON with label, workgroups, elapsed time, batch, and plan.
   - No source behavior change.

2. Integrate batch-major CLIP with conservative re-render-before-backward schedule.
   - This is likely the first meaningful app-level win.
   - Keep current `VisionTrainer` path behind a toggle until parity and timing are in hand.

3. Add single-view buffer aliasing API.
   - Let `VisionTrainer` bind external `inputSlot` and `inputGradSlot`.
   - Then extend the idea to batch lanes or raster lane-offset gradient reads.

4. Stage `spatial_bwd` weights for the top timed layers.
   - Start with the stem/general layers if timestamps show they matter.
   - Leave depthwise alone until measured.

5. Shape-gated shared-W pointwise in batch-major train.
   - Replace only shapes that win in full CLIP, not only microbench.
   - Consider B=2 special scheduling if B=3 remains occupancy-limited.

6. Pre-normalize text embeddings.
   - Small but simple loss-head cleanup.
   - Easy to parity-test.

7. F16 weights with f32 accumulation.
   - Feature-gated, exact fallback.
   - Promote only with integrated speed and quality rows.

8. Train-plan liveness/memory report.
   - Useful for B=9 and mobile adapters.
   - Not urgent unless memory or allocation pressure appears.

## Reference Index

Local Neural-Force-Field-Art:

- `src/clip/vision.ts:1-15`: fused runtime overview.
- `src/clip/vision.ts:132-143`: one compute pass for forward.
- `src/clip/vision.ts:189-197`: trainer owns activation/grad slots and frozen weights.
- `src/clip/vision.ts:295-329`: full and split train encoders.
- `src/clip/vision_wgsl.ts:155-218`: tiled pointwise matmul.
- `src/clip/vision_wgsl.ts:274-374`: staged/unrolled spatial conv forward.
- `src/clip/vision_wgsl.ts:435-532`: attention core.
- `src/clip/vision_wgsl.ts:581-603`: standalone train GELU.
- `src/clip/vision_bwd_wgsl.ts:140-162`: pointwise backward.
- `src/clip/vision_bwd_wgsl.ts:224-280`: spatial backward.
- `src/clip/vision_bwd_wgsl.ts:415-464`: cosine loss backward.
- `src/clip/vision_bwd_wgsl.ts:475-571`: attention backward.
- `src/clip/vision_batch.ts:393-559`: batch-major trainer.
- `src/clip/vision_batch_wgsl.ts:77-146`: batch offset rewrite and z-dispatch.
- `src/clip/vision_batch_pointwise.ts:104-171`: shared-W pointwise microkernel.
- `src/splat3d/optimize.ts:131-143`: current per-view raster/CLIP/backward schedule.
- `src/splat3d/optimize.ts:150-198`: split profile.
- `src/splat3d/optimize.ts:241-263`: shuffled N-of-K view sampling.
- `src/splat3d/raster.ts:112-153`: raster buffers and forward/backward bind groups.
- `src/splat3d/raster.ts:215-248`: raster forward/backward state use.
- `docs/CLIP_BATCHING_NOTES.md:125-171`: batch-major train results.
- `docs/CLIP_BATCHING_NOTES.md:173-224`: shared-W pointwise results.
- `docs/SPLAT3D_PERF_NOTES.md:22-35`: current integrated timing surface.
- `docs/SPLAT3D_PERF_NOTES.md:51-67`: next CLIP measurements and experiments.

Dynaworld / fast-mac-gsplat:

- `dynaworld/BASELINES.md:95-108`: direct atomic, deterministic tile-pair, and feature reduction rows.
- `dynaworld/BASELINES.md:109-125`: target-grid and sparse-grid feature objective rows.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal:4-36`: tile/thread constants.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal:872-903`: threadgroup depth sort.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal:923-1018`: threadgroup reductions and cached feature gradient atomics.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal:1784-1952`: UVT tile binning and rendering.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal:5846-6407`: tile-pair sample emission and parallel reduction.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm:39-83`: env tile config and shader preamble.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm:86-142`: kernel registry and reduction variants.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm:3285-3409`: tile-pair reduced backward host path.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/README.md:182-190`: train-step timing split methodology.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/README.md:215-258`: tile-pair diagnostics, zero pruning, direct atomic speed target.
- `third_party/fast-mac-gsplat/variants/v7_tiled_capture/ENGINEERING_NOTES.md:5-23`: avoiding per-pixel all-splat scans via local tile bins.
- `third_party/fast-mac-gsplat/variants/v7_tiled_capture/ENGINEERING_NOTES.md:31-41`: exactness of tile bins as candidate lists.
- `third_party/fast-mac-gsplat/variants/v7_hybrid_v5style/README.md:64-80`: sparse tile-bin complexity.
- `third_party/fast-mac-gsplat/variants/v9_features_gradcache_zero_bg/csrc/metal/gsplat_v9_features_gradcache_zero_bg_kernels.metal:153-248`: feature gradient cache and reduction code.
- `third_party/fast-mac-gsplat/variants/v9_features_gradcache_zero_bg/ENGINEERING_NOTES.md:1-28`: opt-in gradcache/zero-bg candidate discipline.
- `third_party/fast-mac-gsplat/variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin/ENGINEERING_NOTES.md:1-30`: fixed-capacity binning and host metadata split.
- `third_party/fast-mac-gsplat/variants/v12c_fused_raster_color_loss_backward/README.md:1-44`: fused raster/color/loss backward prototype and constraints.
