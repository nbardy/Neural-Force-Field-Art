# Agent 4 Note: CLIP Speedups Beyond N-of-K

Date: 2026-07-08

Scope: browser and CLIP runtime paths in `tools/clip`, `src/clip`, `src/splat3d`,
text and vision model loading, f16 possibilities, quantization and pruning
feasibility, ORT/WebGPU comparison scripts, and batch-major benches. This is a
planning and experiment note only. I did not modify app code.

## Executive Read

The biggest proven speed lever is still reducing how many CLIP calls the 3D
optimizer makes per step. That is already visible in the current 3D profile:
3 of 9 views per step is about 2.9x faster than 9 of 9. But after that, CLIP
still dominates the sampled 3D step. The next speedups need to attack CLIP
itself: batch-major integration, mixed f16 storage, per-dispatch timing, and
prompt/text cache strategy.

My recommended sequence:

1. Land batch-major CLIP training into `Splat3DOptimizer` for the selected
   views. This is the highest-confidence speedup because the isolated
   `BatchMajorVisionTrainer` bench already verifies gradient parity and shows
   roughly 2x faster B=3 and 2.9x faster B=9 forward+backward.
2. Add a real CLIP dispatch profiler before rewriting kernels. Pointwise matmul
   dominates static MACs, but the exact forward/backward wall-time distribution
   should decide which kernels deserve f16, shared-W batch tiling, or fusion.
3. Add f16 as a feature-gated model/runtime variant. Start with f16 weights and
   f32 accumulation, then use f16 activations/grad slots for memory pressure.
   Activation f16 is as much a batching enabler as a raw speed feature.
4. Fix prompt/text embedding waste: cache by exact expanded prompt, encode
   duplicate prompts once, persist embeddings in IndexedDB, and test batched
   text encoding for the 9 camera prompts.
5. Keep quantization and pruning as research branches only. MobileCLIP-S0 is
   already compact, the current bottleneck is f32 pointwise/storage traffic, and
   unstructured sparsity is a bad match for these WGSL kernels.

Near-term success target: keep the current 3 of 9 stochastic-view behavior, but
drop a normal 3D step from about 70 ms to the 35-50 ms range on the Apple
Metal path, with no cosine-regression or visual-regression in the browser page.

## Current Runtime Map

The current CLIP vision path is not an ONNX runtime in the hot loop. It is a
hand-compiled MobileCLIP-S0 vision encoder/trainer:

- `tools/clip/compile_plan.py` is the only ONNX-aware compiler. It emits
  `plan.json`/`weights.bin` for inference and `plan_train.json`/
  `weights_train.bin` for forward+backward. It packs f32 weights with 16-byte
  alignment and emits train-mode backward metadata.
- `src/clip/vision_wgsl.ts` generates specialized WGSL for forward steps:
  pointwise conv, spatial conv, SE, attention core, head, and train-mode GELU.
- `src/clip/vision_bwd_wgsl.ts` generates backward WGSL for loss, head, GELU,
  pointwise data-grad, spatial data-grad, SE, residual, and attention core.
- `src/clip/vision.ts` owns buffers and pipelines. `VisionTrainer` runs forward,
  loss, and backward in one compute pass list over one `GPUDevice`.
- `src/splat3d/optimize.ts` currently loops selected views one at a time:
  render one camera, copy raster image into the single-image CLIP trainer,
  run CLIP forward+backward, copy image gradient back to raster, accumulate
  raster backward, then Adam once.
- `src/splat3d_page.ts` is the browser integration. It preloads vision weights,
  preloads the text model, encodes one prompt per camera prompt, uploads one
  text embedding per camera, runs selected views per step, and samples split
  wall-time profile every 30 steps.

Static plan facts from the generated local model files:

| Item | Inference plan | Train plan |
| --- | ---: | ---: |
| Forward steps | 99 | 129 |
| Backward entries | 0 | 152 |
| Pointwise conv steps | 48 | 48 |
| Depthwise conv steps | 40 | 40 |
| General spatial conv steps | 5 | 5 |
| Attention core steps | 2 | 2 |
| Weight payload | 45.4 MB f32 | 86.1 MB f32 |
| Slot allocation | 10.9 MB | 185.1 MB act+grad |
| Forward MACs | 2.385 G | 2.385 G |
| Pointwise MACs | 2.215 G | 2.215 G |

Pointwise is about 93% of forward MACs. The top repeated pointwise blocks are
roughly 0.050 G MAC each: `64->192 @64x64`, `192->64 @64x64`,
`128->384 @32x32`, `384->128 @32x32`, and similar ConvFFN expansions and
contractions. This explains why f16 and batch-major pointwise work are
interesting, but it does not prove they are the only wall-time bottleneck.
Backward spatial gathers and activation traffic can still matter in train mode.

Current measured surface from existing notes and benches:

- Fused single-image forward: about 6.2-6.6 ms steady-state.
- ORT WebGPU forward baseline: about 11.3-12 ms mean, 9.5 ms min.
- Full forward+backward: previously measured in the 20-40 ms band depending on
  shared-GPU noise and plan.
- 3D sampled profile, default 4096 splats, 9 cameras:
  - 9/9 views: normal step 205.26 ms, split profile 228.20 ms, CLIP about
    180.58 ms.
  - 5/9 views: normal step 122.02 ms, split profile 134.83 ms, CLIP about
    97.56 ms.
  - 3/9 views: normal step 69.94 ms, split profile 75.57 ms, CLIP about
    59.03 ms.
- Batch-major train bench:
  - B=3: separate forward+backward 102.55 ms/batch, batch-major 49.48 ms/batch.
  - B=9: separate forward+backward 389.66 ms/batch, batch-major 134.63 ms/batch.
- Shared-W pointwise microbench verifies exactly, but wins are shape-specific.
  Some B=2 expansion layers improve modestly; B=3 often loses from occupancy
  and workgroup-memory pressure.

The current dirty tree contains active batch work. Before treating any batch
bench as production-ready, re-run the TypeScript/Bun benches from a clean merge.
Do not optimize on top of a half-merged batch file.

## Batch-Major Integration: Highest-Confidence Production Win

The current 3D optimizer already samples N selected camera views per optimizer
step. It still runs CLIP once per selected view. The batch-major trainer should
change only the CLIP section of a step, not the camera sampling policy.

Desired step layout for `Splat3DOptimizer.step(displayView, viewsPerStep)`:

1. Sample the selected camera indices exactly as today.
2. Record `raster.recordClearRawGrad(enc)`.
3. For each selected view and lane:
   - `raster.recordForward(enc, view)`.
   - Copy `raster.image` into `BatchMajorVisionTrainer.inputBuffer` at the
     lane offset.
4. Run one batched CLIP forward+loss+backward:
   - `batchTrainer.encode(enc, { backward: true })`.
   - Text embeddings are already laid out as `[batch][textDim]`.
5. For each selected view and lane:
   - Copy the lane's `inputGradBuffer` slice into `raster.gradImage`.
   - `raster.recordBackwardAdd(enc, view)`.
6. Record Adam once.
7. Render the display view.
8. Submit once.

This preserves the current stochastic N-of-K training semantics. It batches the
expensive CLIP calls while leaving raster forward/backward per-view. That is
fine because the current split profile says CLIP is still roughly 78% of the
sampled 3/9 step.

### Integration Details

The isolated batch-major code already has the essential runtime shape:

- `BatchMajorVisionTrainer` in `src/clip/vision_batch.ts` allocates each
  logical slot as `[batch][slotFloats]`, plus text as `[batch][textDim]`.
- `batchTrainDispatches()` in `src/clip/vision_batch_wgsl.ts` rewrites the
  generated WGSL slot/text bindings to add `batchLane * stride` offsets and
  dispatches `workgroups.z = batch`.
- `tools/clip/batch_major_train_bench.ts` verifies full `dL/dpixels` parity
  against the original `VisionTrainer` for every lane.

The app integration should not reuse the replicated batcher. The replicated
batcher reduces CPU encode/submit but does not improve GPU wall time. It is a
diagnostic path, not a production speed path.

Suggested production API:

```ts
class Splat3DOptimizer {
  private trainer: VisionTrainer | BatchMajorVisionTrainer;
  private clipBatch: 1 | 2 | 3 | 5 | 9;

  setViewPrompts(embeds: Float32Array[]): void;
  step(displayView = 0, viewsPerStep = this.cameras.length): void;
}
```

Implementation should allocate a batch trainer for the maximum configured
`viewsPerStep`, not always 9. The train plan is memory-heavy:

- Single train slots sum to about 185 MB.
- B=3 batch-major slots are about 555 MB before weights and raster buffers.
- B=9 batch-major slots are about 1.67 GB before weights and raster buffers.

B=3 is the likely browser default. B=9 should be an opt-in/high-memory path or
an evaluation mode. If B=9 fails allocation or causes device loss on commodity
adapters, that is not a batch-major failure. It means the app needs a memory
gate and B=3 default.

### Batch-Major Success Metrics

Required correctness gates:

- `BATCH=3 RUNS=2 WARMUP=3 bun tools/clip/batch_major_train_bench.ts`
  passes gradient parity for every lane.
- `BATCH=9 RUNS=2 WARMUP=2 bun tools/clip/batch_major_train_bench.ts`
  passes on the development Metal adapter, even if browser defaults remain B=3.
- A new 3D optimizer test verifies that selected lanes map back to the correct
  camera gradients. Use distinct lane prompts and distinct synthetic image
  variants so lane swaps are obvious.
- The existing 2D `tools/splat/page_smoke.mjs` remains green. Add a 3D page
  smoke for the batch path.

Performance go/no-go:

- B=3 3D normal step should drop from about 70 ms to <=50 ms. A strong result
  is <=42 ms.
- B=3 sampled CLIP time should drop from about 59 ms to <=32 ms.
- B=9 all-view step should drop from about 205 ms to <=130 ms. A strong result
  is <=100 ms, but memory risk is higher.
- CPU encode+submit should not become the new bottleneck. If it does, inspect
  whether the batch path split CLIP into too many passes or introduced
  per-step bind group churn.

### Batch-Major Risks

The current batch-major WGSL transform is string-based. It is a pragmatic proof
and it has verified exact parity, but production codegen would be more robust if
batch offset support is native in the emitters:

- Every forward/backward emitter should accept optional `batchStride` metadata.
- Binding declarations should not be parsed back out of WGSL strings.
- The plan should carry enough slot layout metadata to write direct batched
  emitters without regex mutation.

This does not need to block the first integration if the benches are green, but
it should be addressed before adding f16 variants. Combining string rewriting,
f16 layout rewrites, and backward codegen will become fragile.

## F16 Plan

F16 is two separate opportunities:

1. Reduce bandwidth and load size for weights.
2. Reduce activation/gradient memory and storage traffic so batch-major B=3
   and maybe B=9 are practical in browsers.

The important browser rule: `shader-f16` is optional and must be requested when
creating the device. You cannot create a plain device and later decide to run
f16 WGSL. The page needs to choose the CLIP precision before `requestDevice()`.

### F16 Variants To Test In Order

#### 1. F16 Weights, F32 Activations, F32 Accumulation

This is the safest first experiment.

Implementation sketch:

- Extend `compile_plan.py` to emit `weights_f16.bin` and
  `weights_train_f16.bin` as IEEE fp16 bits. Keep the same float offsets or add
  a new `weightsType`/`weightsScalars` field so the runtime knows byte sizes.
- Add a f16 weight declaration in `vision_wgsl.ts`:

```wgsl
enable f16;
@group(0) @binding(0) var<storage, read> weights : array<vec4<f16>>;
fn W(i: u32) -> f32 { return f32(weights[i >> 2u][i & 3u]); }
fn W4(i: u32) -> vec4f { return vec4f(weights[i]); }
```

- Keep all accumulators as `f32`.
- Keep all activation slots as f32.
- Do not change the text path.

Why this is attractive:

- Train weights fall from about 86 MB to about 43 MB.
- In pointwise kernels, W tiles are repeatedly staged into workgroup memory.
  Halving global weight bandwidth can help, especially in batch-major where
  each lane currently reloads W in the z-batch path.
- Correctness risk is lower than f16 activations because every layer still
  stores activations and gradients in f32.

Expected risk:

- Per-step rel-Linf against ORT f32 will fail at the old 2e-3 threshold in some
  layers. The correct reference becomes "f16-weight WGSL versus f32-weight WGSL"
  or "rounded-weight CPU/WGSL reference", not ORT's original f32 weights.
- Some tiny layer-scale or bias values may underflow less often than expected,
  but this is unlikely with fp16 normal range. Still measure.

Go/no-go:

- Fixed fixture embedding cosine versus f32 fused output >=0.9995.
- Text loss value difference <=1e-3 absolute on a prompt set.
- Input-gradient cosine versus f32 >=0.995 for B=1 and B=3.
- 3D 3/9 optimization cosine after 100 steps is at least 90% of f32 improvement
  from the same seed, with no NaNs.
- Speed improves >=10% for CLIP forward+backward or payload/load time drops
  enough to justify shipping as a memory feature.

#### 2. F16 Activations And Grad Slots, F32 Accumulation

This is the bigger browser batching lever. Train slots are about 185 MB per
lane today. F16 slots could cut that almost in half. B=3 would move from about
555 MB of CLIP slots toward about 278 MB.

Implementation should be staged:

- Keep `inputBuffer`, `outputBuffer`, `inputGradBuffer`, and `textBuffer` f32 at
  first. This preserves raster<->CLIP copy compatibility and loss-head
  stability.
- Convert only large interior activations and grad slots to f16. The slot list
  needs per-slot dtype metadata.
- Use f32 accumulators in pointwise, spatial, SE, attention, and head.
- Convert f16 loads to f32 immediately and f32 outputs back to f16 at the store.
- Keep sensitive reductions f32: head GAP/projection, loss norm/dot, attention
  softmax, and any gradient accumulation slot with multiple writers.

Do not attempt a global string replacement from `array<f32>` to `array<f16>`.
The train plan has mixed roles, and some buffers must stay f32 for raster
interop and final metrics.

Go/no-go:

- Memory estimate for B=3 CLIP train path falls by >=35%.
- B=3 batch-major browser path allocates reliably on the target Chrome/Metal
  adapter.
- B=3 CLIP forward+backward improves >=15% versus f32 batch-major or allows
  higher batch without device loss.
- Gradient cosine versus f32 >=0.99. Lower than weight-only is acceptable only
  if actual optimization quality is unchanged.
- Directional derivative gate still passes with adjusted tolerances and eps.

#### 3. F16 Accumulation

This is the riskiest and should be last. It may improve ALU throughput on some
adapters, but CLIP guidance can be sensitive to gradient direction. F16
accumulation is most plausible in layers with small reduction depth or where
per-dispatch timing proves ALU-bound rather than memory-bound.

Possible limited tests:

- Spatial depthwise forward/backward with f16 accum, because reduction depth is
  small.
- Pointwise expansion layers with f32 partial sums every 32 input channels but
  f16 storage for staged tiles.
- Never use f16 for loss norm/dot, text norm, softmax denominator, or head
  projection until everything else is proven.

Go/no-go:

- Only ship where per-layer timing shows a clear win and gradient quality
  remains within f16-activation thresholds.
- If f16 accumulation gives less than 10% additional improvement over f16
  storage, do not take the extra numerical risk.

### Browser F16 Feature Gate

The page should centralize device/runtime selection before `requestDevice()`:

```ts
async function chooseGpuRuntime() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter available");

  const hasF16 = adapter.features.has("shader-f16");
  const hasTimestamps = adapter.features.has("timestamp-query");
  const precision = hasF16 && userOrAutoAllowsF16 ? "f16" : "f32";
  const features = [];
  if (precision === "f16") features.push("shader-f16");
  if (hasTimestamps && debugProfilingEnabled) features.push("timestamp-query");

  const device = await adapter.requestDevice({ requiredFeatures: features });
  return { adapter, device, precision, hasTimestamps };
}
```

Additional gates:

- Startup microprobe for f16: compile a tiny f16 shader, write a few known
  values, read back once, and disable f16 if validation/runtime fails.
- Model-file gate: load `weights_train_f16.bin` only after choosing f16. Do not
  download f32 weights first and then decide.
- Memory gate: estimate `weights + slots * batch + raster + canvases` before
  allocation. Use B=3 default for f16/f32. Use B=9 only in explicit high-memory
  mode.
- Device-loss fallback: if f16 or B=3 allocation causes device loss, reload with
  f32/B=1 or f32/B=3 depending on where it failed. Do not silently continue
  with corrupted optimizer state.

## Prompt And Text Embedding Strategy

The text encoder is not in the per-frame hot loop, but it is a visible browser
latency source and it is currently doing avoidable work.

Current browser behavior:

- `src/splat_page.ts` and `src/splat3d_page.ts` load transformers.js from the
  jsDelivr `+esm` endpoint through native dynamic import.
- The model id is `Nbardy/nff-clip-splat-weights`.
- The text model uses `dtype: "fp16"`, `device: "wasm"`, and
  `graphOptimizationLevel: "basic"` to avoid the ORT fp16 layernorm fusion bug.
- The text embedding is 512-d f32 output and is not L2-normalized.
- 3D camera mode expands one base prompt into per-camera prompts via
  `buildViewPrompt`. Same-text mode expands one base prompt via
  `buildBasePrompt`.
- `encodePrompt()` is serial and has no app-level embedding cache. Browser
  Cache Storage caches model assets, but the 512-float prompt outputs are not
  cached by prompt key.

### Immediate Prompt Cache Fixes

Add an in-memory cache:

```ts
const textEmbedCache = new Map<string, Promise<Float32Array>>();

function textCacheKey(prompt: string): string {
  return JSON.stringify({
    model: "Nbardy/nff-clip-splat-weights",
    dtype: "fp16",
    maxLength: 77,
    prompt,
  });
}
```

Use `Promise<Float32Array>` as the value so concurrent clicks or camera prompts
dedupe in-flight work.

For same-text 3D mode:

- Build the base prompt once.
- Encode once.
- Fill `embeds = cameras.map(() => sameEmbedding)`.
- `setViewPrompts()` can write the same `Float32Array` into each per-view
  GPU text buffer.

For camera-text 3D mode:

- Build all expanded prompts.
- Deduplicate exact strings before encoding. Some camera sets may share text
  after mode changes.
- Reuse cached embeddings on Reset. Reset changes splat params, not text.
- Reuse cached embeddings when only `viewsPerStep` changes.

Add persistent IndexedDB cache:

- Store `{ key, prompt, createdAt, vec: Float32Array }`.
- A 512-float embedding is only 2048 bytes. Hundreds of cached prompts are
  negligible compared with model assets.
- Include a cache version and model revision in the key. If the HF model is
  updated, old embeddings should not be reused.
- Keep an in-memory LRU for the active session and IndexedDB for warm starts.

Go/no-go:

- Same-text 3D mode encodes exactly one prompt, not nine.
- Warm-cache camera mode returns all nine embeddings in <100 ms after model
  load.
- Re-clicking Optimize with the same prompt starts optimization in <50 ms after
  model load.
- Reset does not clear text embeddings.
- Prompt cache misses and hits are observable in a debug console or status
  object, but do not clutter the normal UI.

### Batched Text Encoding Experiment

Transformers.js may support batched tokenizer/model inputs for CLIP text. Test
this because 9 serial text model invocations are likely slower than one batched
invocation if the ONNX text model accepts dynamic batch.

Experiment:

- Add a local browser-only test page or a Node script using transformers.js.
- Compare:
  - serial `encodePrompt(prompt)` for nine camera prompts,
  - deduped serial,
  - `tokenizer(prompts, { padding: "max_length", max_length: 77, truncation: true })`
    followed by one `textModel(enc)`, if supported.
- Verify each batched output against serial output:
  - cosine >=0.999999,
  - max absolute diff <=1e-5 if the runtime is deterministic.

Success metric:

- Batched nine-prompt encode is >=2x faster than serial warm-cache encode, or
  it is not worth the code complexity.

### Text Norm Precompute

The CLIP loss normalizes text embeddings. Text embeddings are static per prompt.
The loss head can store `textInvNorm` or `{ vec, invNorm }` in the text buffer
instead of recomputing text norm every backward.

This is low priority because the loss head is tiny compared with the vision
encoder, but it is nearly free after touching text buffers for batch-major.

Go/no-go:

- Only do this if per-dispatch timing shows `loss_bwd` is nonzero enough to
  measure, or if it simplifies batch text layout.

## Per-Layer And Per-Dispatch Timing

Do not guess the next kernel rewrite. Add timing first.

The repo already has `src/render/webgpu/gputime.ts` for pass-level timestamp
queries in the particle-art path, and the 3D page has split-submit wall timing.
CLIP needs an equivalent diagnostic harness, not necessarily a hot-path feature.

### Proposed Tool: `tools/clip/dispatch_profile.ts`

Modes:

- `MODE=forward PLAN=plan.json BATCH=1`
- `MODE=train PLAN=plan_train.json BATCH=1`
- `MODE=train BATCH=3`
- `MODE=train BATCH=9`
- `PRECISION=f32|f16` once f16 exists.

Outputs:

- JSON and CSV with one row per dispatch:
  - index,
  - label,
  - plan step,
  - kind,
  - shape,
  - workgroups,
  - median GPU ms,
  - min GPU ms,
  - mean GPU ms,
  - CPU encode ms if measured,
  - batch size,
  - precision,
  - adapter info.
- Aggregates by kind and shape:
  - all pointwise forward,
  - all `pw_bwd`,
  - spatial forward,
  - `spatial_bwd`,
  - GELU,
  - SE,
  - attention core forward/backward,
  - head/loss.

Implementation choices:

- Use `timestamp-query` when available. Request the feature at device creation.
- For per-dispatch timing, begin/end one compute pass per dispatch in profiling
  mode. This may change overhead relative to the one-pass hot path, but it
  still gives good attribution and should not be used as the final wall-time
  number.
- Use a large query set or chunk into groups if the adapter has query-count
  limits.
- Warm all pipelines before timing. Metal JIT makes cold timing useless.
- Record full hot-path wall time separately with the existing one-pass encode.

Fallback if timestamp-query is unavailable:

- Split submits by step or by grouped ranges and use `queue.onSubmittedWorkDone()`.
- This is noisier and slower, but good enough for coarse grouping.

Go/no-go:

- The profiler identifies the top 20 dispatches accounting for >=80% of CLIP
  wall time.
- Repeated profiles have coefficient of variation <=15% for top groups after
  warmup.
- The profiler explains whether B=3 batch-major gains come from fewer dispatch
  submissions, better occupancy, or specific kernels.
- Shared-W pointwise integration is attempted only for shapes where the full
  profile says pointwise is a top bottleneck and the microbench showed a win.

### Specific Timing Questions To Answer

1. Is train-mode wall time dominated by forward pointwise, `pw_bwd`, or
   `spatial_bwd`?
2. Does B=3 batch-major improve all pointwise steps uniformly, or only
   low-occupancy small shapes?
3. Is the first stem `spatial_bwd` large enough to justify staged weights or
   tiled input reuse?
4. Are GELU split steps measurable enough to fuse back into adjacent kernels in
   train mode?
5. Are attention backward kernels above 5-10% of train time?
6. Does f16 weight-only improve weight-heavy pointwise steps but not spatial
   steps?
7. Does f16 activation storage reduce memory traffic enough to move B=3 closer
   to B=1 per-image time?

## ORT/WebGPU Comparison Path

`tools/clip/ort_web_bench.mjs` is currently a useful forward-only ORT WebGPU
baseline. It serves the repo, launches headless Chrome with WebGPU flags,
imports `onnxruntime-web`, forces the WebGPU EP, warms the session, times
`session.run`, and verifies image embedding cosine against the ORT CPU golden.

Extend it as a comparison harness, not as an app runtime candidate:

- Add `BATCH=1|3|9` and feed `[B,3,256,256]` image batches if the ONNX export
  accepts dynamic batch. If the model is fixed batch-1, document that.
- Compare ORT WebGPU forward B=1/B=3/B=9 against fused batch-major forward.
- Record session create time separately because graph optimization/JIT is not a
  per-step cost but matters for cold-start UX.
- Add `graphOptimizationLevel` variations only if useful. The vision model uses
  `"all"` today. The text fp16 model needs `"basic"` due the layernorm fusion
  bug.
- Add optional ORT WebGPU text model benchmark, but keep app text on WASM unless
  profiling proves GPU text is worth competing with render/CLIP work.

Success metrics:

- Fused WGSL B=1 forward should remain faster than ORT WebGPU B=1 and much
  cheaper on CPU command overhead.
- Fused batch-major B=3/B=9 should beat ORT WebGPU batch forward or at least be
  close while retaining the unique advantage: fused hand-written backward.
- If ORT WebGPU batch forward beats fused forward at large B, use that as a
  kernel-design signal, not as a replacement for the train path.

## Quantization Feasibility

Quantization is lower priority than f16. It may reduce payload size, but it is
not obviously faster in this hand-written WGSL runtime.

### Weight-Only Int8

Potential design:

- Per-output-channel symmetric int8 for pointwise weights.
- Store scales as f32 or f16.
- Decode int8 to f32 in the tiled pointwise load path.
- Keep activations f32/f16 and accumulators f32.

Why it may lose:

- The current kernels use f32 `fma` over staged tiles. If int8 decode adds
  scalar unpacking and scale multiplies without a native dot-product path, saved
  memory bandwidth can be eaten by ALU and code complexity.
- WebGPU feature availability for integer dot-product style acceleration must
  be feature-detected. Do not design around an optional feature without a
  fallback.
- Accuracy risk is higher for CLIP gradients than for classification logits.
  The optimizer cares about gradient direction every step, not just top-1
  embedding similarity.

Minimum experiment:

- Quantize only pointwise weights.
- Generate a f32 dequantized reference from the quantized weights to separate
  quantization error from kernel error.
- Compare:
  - embedding cosine,
  - text-image loss on a prompt set,
  - input-gradient cosine,
  - 100-step 2D and 3D optimization quality.

Go/no-go:

- Payload reduction >=4x for quantized segments.
- CLIP forward+backward speed improves >=20% versus f16 weight-only.
- Gradient cosine versus f32 >=0.98.
- Optimization quality remains within 90% of f32 cosine gain.

If it does not beat f16 weight-only by a real margin, drop it.

### Text Model Quantization

The Node text tool mentions `text_model_quantized.onnx` as a fallback that was
not needed. In the browser, text currently uses fp16 weights on WASM and runs
once per prompt. Optimizing text model compute is not a hot-loop priority.

Only revisit quantized text if:

- cold text load is the dominant UX complaint after prompt embedding cache,
- fp16 WASM text is unsupported on a target browser,
- or batched text prompts are still too slow.

## Pruning Feasibility

Pruning is least promising for the current goal.

Unstructured pruning:

- Bad fit for these dense WGSL kernels.
- Sparse indexing would add irregular memory access and branch overhead.
- MobileCLIP-S0 is already compact; random sparsity will not map cleanly to the
  existing tiled pointwise kernels.

Structured channel pruning:

- Could reduce pointwise MACs, but it changes tensor shapes throughout the
  model.
- Requires re-exporting/recompiling plans and likely fine-tuning or calibration.
- Without a MobileCLIP training/calibration pipeline, gradient quality could
  degrade silently.

Step/module ablation:

- Skipping SE, attention, or late blocks could be measured as a quality/speed
  experiment, but this is model surgery, not a runtime optimization.
- If attempted, use it only as a research branch with explicit quality metrics.

Go/no-go:

- Do not spend production time on pruning until batch-major and f16 are
  exhausted.
- Any pruning branch must show >=25% CLIP train-time reduction and browser
  optimization quality within 90% of baseline over several prompts/seeds.

## Browser Runtime Gating

The browser page should expose one CLIP runtime decision point. Today both
2D and 3D pages request a default device and then build f32 CLIP. F16,
timestamp profiling, and batch-major memory policy need to be decided before
device creation and before model fetch.

Suggested runtime policy:

| Gate | Decision | Default |
| --- | --- | --- |
| `navigator.gpu` | no WebGPU, show current hard error | required |
| `adapter.features.has("shader-f16")` | allow f16 model variant | yes if probe passes |
| `adapter.features.has("timestamp-query")` | allow debug CLIP profiler | off unless debug |
| estimated CLIP memory | choose B=1/B=3/B=9 | B=3 max |
| URL/debug override | force f32/f16/batch for benches | allowed |
| device loss during create | retry lower tier | f32 B=1 or f32 B=3 |

Memory estimate should be conservative:

```ts
const clipSlotBytes = sum(plan.slots) * bytesPerSlotElement * batch;
const clipWeightBytes = weightsBytes;
const imageBytes = 3 * 256 * 256 * 4;
const rasterBytes = estimateRasterBytes(G, cap, cameras);
const safety = 1.25;
const total = safety * (clipSlotBytes + clipWeightBytes + rasterBytes + imageBytes * 4);
```

Browsers do not provide a reliable "available GPU memory" number, so this is a
tier heuristic, not a guarantee. Still, it is better than attempting B=9 on
every adapter.

Runtime labels should be available to tests:

```ts
window.__splat3d.runtime = {
  clipPrecision: "f32" | "f16",
  clipBatch: 1 | 3 | 9,
  shaderF16: boolean,
  timestampQuery: boolean,
  adapter: adapter.info,
};
```

The visible UI should stay quiet. Put details in `window.__splat3d` and console
logs, not in normal explanatory text.

## Concrete Experiment Backlog

### Experiment 1: Reconfirm Current Batch Benches

Purpose: establish a clean baseline after concurrent edits settle.

Commands:

```bash
PLAN=plan_train.json bun tools/clip/fused_test.ts
bun tools/clip/bwd_test.ts
BATCH=3 RUNS=2 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
BATCH=9 RUNS=2 WARMUP=2 bun tools/clip/batch_major_train_bench.ts
BATCH=3 MODE=backward RUNS=3 WARMUP=2 bun tools/clip/batch_bench.ts
```

Metrics:

- All parity gates pass.
- Batch-major B=3 is at least 1.7x faster than separate B=3.
- Replicated batcher remains slower on GPU wall time, confirming it should not
  be integrated.

### Experiment 2: Integrate Batch-Major Into 3D Optimizer Behind A Toggle

Purpose: prove isolated B=3 speedup survives full optimizer scheduling.

Files likely touched by the implementing worker:

- `src/splat3d/optimize.ts`
- `src/splat3d_page.ts`
- maybe `src/clip/vision_batch.ts` if runtime API cleanup is needed

Toggle:

- URL or debug config: `?clipBatch=1|3|9`.
- Default remains current single path until green.

Metrics:

- Same seed/prompt/camera config, compare B=1 current path and B=3 batch path.
- 3/9 normal step <=50 ms.
- Sampled CLIP <=32 ms.
- Cosine after 150 steps not worse than current path by more than 10% of
  improvement.
- No device loss in Chrome.

### Experiment 3: CLIP Dispatch Profiler

Purpose: determine exact kernel targets for f16/shared-W/fusion.

Command shape:

```bash
MODE=train BATCH=1 RUNS=30 WARMUP=10 bun tools/clip/dispatch_profile.ts
MODE=train BATCH=3 RUNS=30 WARMUP=10 bun tools/clip/dispatch_profile.ts
MODE=forward PLAN=plan.json BATCH=1 RUNS=50 WARMUP=10 bun tools/clip/dispatch_profile.ts
```

Metrics:

- Top dispatch groups and shape groups exported as CSV/JSON.
- Pointwise vs spatial_bwd vs attention_bwd percentages.
- Per-layer f16 candidate list.
- Shared-W candidate list limited to shapes with measured bottleneck share.

### Experiment 4: F16 Weight-Only Forward

Purpose: measure the easiest f16 speed/payload win.

Command shape:

```bash
uv run --with onnx --with numpy python tools/clip/compile_plan.py --f16-weights
PRECISION=f16 bun tools/clip/fused_test.ts
PRECISION=f16 PLAN=plan_train.json bun tools/clip/bwd_test.ts
PRECISION=f16 BATCH=3 RUNS=3 WARMUP=5 bun tools/clip/batch_major_train_bench.ts
```

Metrics:

- Embedding cosine >=0.9995 versus f32 fused.
- Gradient cosine >=0.995 versus f32 fused.
- B=3 CLIP train wall improves >=10%.
- Weight payload halves.

### Experiment 5: F16 Slot Storage Prototype

Purpose: reduce train-plan memory and storage traffic.

Prototype scope:

- Keep CLIP input, output embedding, input gradient, text, loss reductions f32.
- Convert large interior activation and grad slots to f16.
- Start with forward-only, then train.

Metrics:

- B=3 slot memory drops >=35%.
- B=3 browser allocation succeeds where f32 is close to the edge.
- B=3 CLIP train wall improves >=15% over f32 batch-major.
- Gradient cosine >=0.99.

### Experiment 6: Prompt Embedding Cache

Purpose: remove text-encoding stalls and duplicate work.

Steps:

- In-memory promise cache by exact expanded prompt.
- Same-text mode encodes once for all cameras.
- IndexedDB persistent cache.
- Optional batched text encode test.

Metrics:

- Same-text 3D Optimize encodes one prompt.
- Warm-cache same prompt starts optimization in <50 ms.
- Camera mode warm-cache returns all camera embeddings in <100 ms.
- Batched text encode ships only if >=2x faster than serial warm encode.

### Experiment 7: ORT Batch Forward Comparison

Purpose: keep a standard-runtime reference for forward batching.

Command shape:

```bash
BATCH=1 node tools/clip/ort_web_bench.mjs 50
BATCH=3 node tools/clip/ort_web_bench.mjs 30
BATCH=9 node tools/clip/ort_web_bench.mjs 20
```

Metrics:

- ORT output cosine versus golden/fused.
- ORT B=3/B=9 scaling curve.
- Fused batch-major forward remains faster or provides clear train-path value.

### Experiment 8: Quantized Pointwise Branch

Purpose: decide whether int8 deserves more attention after f16.

Scope:

- Pointwise weights only.
- Per-output-channel scale.
- F32 accumulation.
- Compare against f16 weight-only, not just f32.

Metrics:

- >=20% faster than f16 weight-only CLIP train path.
- Gradient cosine >=0.98.
- 100-step optimization quality within 90% of f32.

Expected outcome: probably reject unless a native integer dot path is available
and feature-detected.

## Recommended Acceptance Matrix

| Area | Must pass | Strong result |
| --- | --- | --- |
| Batch-major B=3 | gradient parity, 3D step <=50 ms | 3D step <=42 ms |
| Batch-major B=9 | parity on dev adapter | all-view step <=100 ms |
| F16 weights | grad cosine >=0.995, >=10% speed or payload win | >=20% CLIP speed |
| F16 slots | grad cosine >=0.99, >=35% memory drop | B=3 stable on browser and >=15% speed |
| Prompt cache | same prompt <50 ms warm start | camera-mode all embeds <100 ms |
| Dispatch profiler | top 80% attributable | stable CSV across runs |
| Quantization | beats f16 by >=20% | otherwise reject |
| Pruning | >=25% CLIP speed and quality retained | likely reject |

## Practical Implementation Notes

- Keep f32 as the universal fallback. This project is WebGPU-only by design, but
  it does not need to be f16-only or batch-only.
- Request optional features once, before any device users are created. The
  canvas, rasterizer, CLIP trainer, profiler, and text path all share the
  device assumptions.
- Do not put ORT WebGPU text inference on the shared render device unless
  profiling proves it is worth it. Text runs outside the hot loop and current
  WASM fp16 avoids GPU contention.
- Do not optimize text norm or 2 KB text buffer copies before CLIP timing says
  they matter.
- Do not integrate shared-W pointwise globally. The microbench says it is not a
  blanket win. Use it selectively after full CLIP profiling.
- Do not start with int8. F16 is simpler, feature-gated, and directly reduces
  the browser memory problem that blocks batch-major.
- Do not judge f16 with the old ORT f32 per-step tolerance alone. Add f32-fused
  versus f16-fused comparisons and actual optimization quality gates.
- Add model variant metadata to `plan_train.json` or a sidecar manifest before
  shipping multiple weight files. The page should know exactly which weight
  dtype it loaded.

## Bottom Line

The most credible speed path is:

1. Keep N-of-K view sampling as the algorithmic default.
2. Batch the selected views through `BatchMajorVisionTrainer`.
3. Add f16 weights to reduce payload and pointwise bandwidth.
4. Add f16 interior slots to make B=3 memory comfortable in the browser.
5. Use dispatch timing to decide whether specific pointwise shared-W, spatial
   backward staging, or GELU/residual fusion is worth landing.
6. Cache prompt embeddings so text work disappears from repeat interactions.

If those land cleanly, 3D optimization should feel materially more interactive
without changing the art/optimization semantics. Quantization and pruning are
possible research branches, but they should not distract from batch-major plus
f16, which directly matches the current measured bottleneck and the existing
runtime architecture.
