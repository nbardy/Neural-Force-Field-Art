# Agent 1 F16 Reality Check

Date: 2026-07-08

Scope: current repo state for CLIP/splat optimization. This is a documentation-only inspection; no shader or runtime code was changed.

## 1. Current Implemented Support

CLIP vision f16 has not landed.

The hot-loop MobileCLIP vision path is still f32:

- `tools/clip/compile_plan.py` emits `weights.bin` or `weights_train.bin` as one packed f32 blob. The packer explicitly casts every segment to `np.float32`, and the output names are fixed to the f32 files.
- `models/mobileclip_s0/` currently contains `weights.bin` and `weights_train.bin`; there is no `weights_f16.bin` or `weights_train_f16.bin`.
- `src/clip/vision_wgsl.ts` declares weights as `array<vec4f>` and exposes `W()`/`W4()` as f32/vec4f. Activations are documented and generated as NCHW planar f32.
- `src/clip/vision_bwd_wgsl.ts` imports the same `weightsDecl()` and declares all grad/activation bindings as `array<f32>` or `array<vec4f>`.
- `src/clip/vision.ts`, `src/clip/vision_batch.ts`, and current CLIP tools accept/load `Float32Array` weights and allocate all slots as `floats * 4`.
- The browser 3D page still fetches `weights_train.bin`, then does `weights = new Float32Array(wbuf)`.
- CLIP bench tools request plain WebGPU devices, except timestamp-query where explicitly added. They do not request `shader-f16` for CLIP.

There are two f16 implementations in the repo, but they are not CLIP vision:

- The force-field advect path has a real f16 fast path in `src/render/webgpu/advect.ts` and `src/render/webgpu/advect_wgsl.ts`. It is gated on `device.features.has("shader-f16")`, staged weights, and the unrolled raw encoder. Comments record a measured improvement of roughly `9.3 -> 6.5 ms/step @ 1M` on Apple Metal. `src/main.ts` wraps `GPUAdapter.requestDevice()` to request `shader-f16` for that tfjs-owned device.
- The text encoder uses fp16 ONNX weights once per prompt: `tools/clip/text_onnx.mjs` loads `onnx/text_model_fp16.onnx`, and `src/splat3d_page.ts` uses `dtype: "fp16"` on WASM. It returns a `Float32Array` embedding and is not in the per-step vision hot loop.

Bottom line: the user memory of a big f16 jump is real for the older advect/force-field shader, not for CLIP vision. CLIP f16 is currently notes/plans only.

## 2. Relevant Docs And Plans

Existing notes already outline the right f16 direction:

- `agent_notes/optimization_session/agent_4.md` has the most complete plan. It separates f16 weights, f16 activations/grad slots, and f16 accumulation. It also calls out the browser rule that `shader-f16` must be requested at device creation.
- `agent_notes/optimization_session/agent_1.md` recommends f16 only as a gated variant after profiling, with f32 accumulation and f32 fallbacks.
- `agent_notes/optimization_session/agent_2.md` and `agent_5.md` agree on the safe order: f16 weights first, f16 activations later, strict gradient quality gates.
- `agent_notes/optimization_session/rollout_review.md` ranks f16 weights with f32 accumulation as a later consensus item, behind batch-major CLIP and dispatch profiling.
- `docs/SPLAT3D_ABLATION_QUEUE.md` has an open "F16 Weights With F32 Math" entry with promotion gates: feature-gated fallback, embedding cosine >= 0.9995, gradient cosine >= 0.995, and B=3 train speed improvement >= 10% unless payload reduction is the explicit reason.
- `docs/CLIP_BATCHING_NOTES.md` keeps f16 weights with f32 math in the "Next Five Iterations" list.
- `docs/SPLAT3D_PERF_NOTES.md` says f16 weights/activations with f32 reductions are one of the promising CLIP experiments, but should follow per-dispatch timestamp profiling.

The important correction to the docs: none of these plans has been implemented for CLIP vision yet.

## 3. Files Needed For CLIP F16

### F16 Weights Only, F32 Activations, F32 Accumulation

This is the first real experiment.

Files that need changes:

- `tools/clip/compile_plan.py`
  - Add `--f16-weights` or a separate precision option.
  - Emit `weights_f16.bin` and `weights_train_f16.bin` as IEEE fp16 scalar data.
  - Keep existing scalar offsets if the f16 blob preserves the same scalar indexing and pads to 4 half scalars for `vec4<f16>` loads.
  - Add plan metadata such as `weightsDtype: "f16"` or rely on runtime precision selection plus filename.

- `src/clip/vision_wgsl.ts`
  - Add a precision-aware weight declaration, for example `weightsDecl(binding, "f32" | "f16")`.
  - For f16:
    - `enable f16;`
    - `@group(0) @binding(n) var<storage, read> weights : array<vec4<f16>>;`
    - `fn W(i: u32) -> f32 { return f32(weights[i >> 2u][i & 3u]); }`
    - `fn W4(i: u32) -> vec4f { return vec4f(weights[i]); }`
  - Keep pointwise, spatial, SE, attention, head, GELU, residual, and loss math in f32.

- `src/clip/vision_bwd_wgsl.ts`
  - Thread the same weight precision option through backward dispatch generation.
  - Backward should still consume f32 activations/grads and accumulate f32.

- `src/clip/vision.ts`
  - Accept `Float32Array` or `Uint16Array`/`ArrayBufferView` weights.
  - Keep the logical scalar-count check against `plan.weightsFloats`, but use `byteLength` for the GPU buffer size.
  - Build pipelines with the selected precision.
  - Store the chosen precision in `VisionEncoder` / `VisionTrainer` enough to keep diagnostics honest.

- `src/clip/vision_batch.ts`, `src/clip/vision_batch_wgsl.ts`, `src/clip/vision_batch_pointwise.ts`
  - Add the same precision option to batch-major and shared-W variants.
  - Ensure shared-W pointwise uses the precision-aware `weightsDecl()` rather than duplicating f32 assumptions.

- CLIP tools:
  - `tools/clip/fused_test.ts`
  - `tools/clip/bwd_test.ts`
  - `tools/clip/batch_major_train_bench.ts`
  - `tools/clip/batch_major_forward_bench.ts`
  - `tools/clip/batch_bench.ts`
  - `tools/clip/dispatch_profile.ts`
  - Add `PRECISION=f16`, request `shader-f16`, load `weights*_f16.bin`, and compare f16 against the f32 fused oracle rather than strict ORT f32 per-step tolerances.

- Browser/runtime:
  - `src/splat3d_page.ts` and the 2D splat page path should choose CLIP precision before `requestDevice()`, request `shader-f16` only when needed, and fetch `weights_train_f16.bin` only after choosing f16.
  - `src/splat3d/optimize.ts` and `src/splat/optimize.ts` types will need to accept the widened weight view type when passing weights into `VisionTrainer.create()`.
  - Hosted model assets must include the f16 blobs.

### F16 Activations / Grad Slots

This is a second-phase experiment, not a global replacement.

Files that need changes:

- `tools/clip/compile_plan.py`
  - Add per-slot dtype metadata, probably `slotDtypes`, and maybe `slotBytes`.
  - Keep input image, output embedding, input gradient, text embedding, loss reductions, attention softmax reductions, and any multiply-written accumulation slots as f32 at first.
  - Mark only large interior saved activations and large interior grad slots as f16.

- `src/clip/vision.ts` and `src/clip/vision_batch.ts`
  - Allocate slots by byte size, not `floats * 4`.
  - Keep `writeInput`, `writeText`, `outputBuffer`, and `inputGradBuffer` f32-compatible in the first version.
  - Add read/diagnostic helpers that know slot dtype.

- `src/clip/vision_wgsl.ts`, `src/clip/vision_bwd_wgsl.ts`, `src/clip/vision_batch_wgsl.ts`, `src/clip/vision_batch_pointwise.ts`
  - Generate binding declarations per slot dtype.
  - Load f16 slots as f32 immediately and store f32 results back as f16 at the boundary.
  - Keep accumulators and reductions f32.
  - Handle `accumulate:true` carefully; first version should leave accumulation destinations f32 to avoid read-modify-write precision loss.

- Splat interop:
  - If input/output/inputGrad remain f32, `src/splat3d/optimize.ts`, `src/splat/optimize.ts`, and raster kernels can stay mostly unchanged.
  - If inputGrad becomes f16 later, raster backward currently cannot consume it directly; that would be a separate raster/CLIP contract change.

Do not do this as string replacement from `array<f32>` to `array<f16>`. The train graph has mixed buffer roles and f32-sensitive reductions.

## 4. Expected Speed Risk / Reward

F16 weights-only reward:

- Payload: strong. Local train weights are about 82 MB today; f16 should cut the vision train weight blob to roughly 41 MB. Inference weights should drop from about 43 MB to about 21.5 MB.
- GPU speed: plausible but not guaranteed. The hottest dispatch families include pointwise forward/backward, and those repeatedly load W tiles. Halving global weight traffic can help if the kernels are weight-bandwidth limited.
- Realistic expectation: maybe 5-25% CLIP train improvement if weight bandwidth is material. It is unlikely to give a full 2-4x optimizer-step win alone because activations, gradients, raster backward, Adam, dispatch ordering, and f32 accumulation remain.
- Risk: low-to-medium. Weight rounding can move embeddings and gradients. Old ORT f32 per-step relLinf thresholds will no longer be the right gate; compare against f32 fused CLIP and actual optimization behavior.

F16 activations/grad slots reward:

- Memory: strong. Agent 4 estimated train slots at roughly 185 MB per lane. Selective f16 slots could cut a B=3 CLIP train allocation by hundreds of MB and make B=3/B=9 browser modes more comfortable.
- GPU speed: potentially larger than weights-only if the timestamp profile proves slot traffic is the limiting factor. This can reduce storage traffic in both forward and backward.
- Realistic expectation: maybe 10-30% CLIP train improvement, plus a batching/memory stability win. Higher is possible only if current kernels are dominated by memory bandwidth, which we have not proven with Metal counters yet.
- Risk: medium-to-high. Bad f16 slot choices can perturb `dL/dpixels`, especially around attention, loss, and accumulation edges.

F16 accumulation:

- Highest numerical risk, lowest confidence. Only test after weights and slot storage.
- Never start by changing loss norm/dot, attention softmax denominator, or gradient accumulation math to f16.

2-4x CLIP speed:

- Same-resolution, same-model, same-per-step CLIP probably will not get 2-4x from f16 weights alone.
- A 2-4x wall-time improvement likely needs a combination: f16 storage, bigger pointwise strategy changes, fewer CLIP calls per optimizer step, or a proxy/distilled inner-loop model. Since the user wants same resolution, the most honest near-term path is f16 plus pointwise/memory-layout work, measured with timestamp queries and Metal/Chrome GPU tracing.

## 5. Suggested First Experiment And Gates

First experiment: CLIP vision f16 weights only, f32 activations, f32 gradients, f32 accumulators, feature-gated.

Implementation discipline:

- Fork/gate it as a variant, not by mutating the default f32 shader path.
- Suggested flag names:
  - tools: `PRECISION=f16`
  - app/runtime: `clipPrecision: "f32" | "f16"`
- Add `shader-f16` only when precision is f16 and `adapter.features.has("shader-f16")`.
- Leave f32 as the default and as the oracle.

Exact gates:

1. Asset gate
   - `uv run --with onnx --with numpy python tools/clip/compile_plan.py --train --f16-weights`
   - Verify `weights_train_f16.bin` exists, scalar count matches `plan.weightsFloats`, and byte size is about half of `weights_train.bin`.

2. Forward correctness gate
   - `PRECISION=f16 PLAN=plan_train.json bun tools/clip/fused_test.ts`
   - Compare final embedding against f32 fused output, not strict ORT f32 per-step refs.
   - Require embedding cosine >= 0.9995, finite outputs, no validation errors.

3. Backward correctness gate
   - `PRECISION=f16 bun tools/clip/bwd_test.ts`
   - Add or reuse a direct f32-vs-f16 input-gradient comparison.
   - Require gradient cosine >= 0.995 for B=1 and B=3 fixture/prompts, no NaNs, and directional derivative still sane under adjusted tolerances.

4. Performance gate
   - `PRECISION=f16 TIMESTAMP=1 MODE=train BATCH=3 RUNS=3 WARMUP=3 bun tools/clip/dispatch_profile.ts`
   - `PRECISION=f16 BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/batch_major_train_bench.ts`
   - Require CLIP train median improves >= 10%, or explicitly keep it only as a payload/memory option.

5. Integrated optimizer gate
   - Add a `CLIP_PRECISION=f16` path to the 3D step bench before promotion.
   - Run the same seed and prompt under f32 and f16 for at least a short 100-step smoke.
   - Require no NaNs, similar loss direction, and at least 90% of the f32 cosine/loss improvement over the same window.

6. Browser gate
   - `npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html`
   - Browser smoke must show graceful fallback to f32 when `shader-f16` is absent.

Promotion decision:

- Promote only if CLIP train speed improves >= 10% in the integrated path, or if the payload/memory win is valuable enough to expose as an opt-in.
- If speed improves < 5% and no browser memory issue is solved, leave the variant gated and record it as a rejected speed optimization.
