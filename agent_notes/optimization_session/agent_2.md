# Agent 2 CLIP Optimization Notes

Scope: I only wrote this note. I did not modify app code. The working tree already
had unrelated CLIP/splat edits when I arrived; this memo treats those files as the
current baseline and does not try to revert or normalize them.

## Executive Thesis

The strongest transferable lesson from `FUSED_js` and `FUSED_JS_2` is not "make
one giant shader at all costs." It is: lock math first, measure at the same
boundary the app actually pays, then spend complexity only where the memory and
dispatch evidence says it will win.

For this repo's CLIP path, the next production win is not another heroic
single-image kernel rewrite. The current MobileCLIP vision path is already
fused enough to beat ORT-WebGPU materially. The high-leverage route is:

1. Use the existing true batch-major forward+backward path in the 3D optimizer
   so 3-of-9 and 9-of-9 view training pay one batched CLIP dispatch list instead
   of repeated single-image passes.
2. Add per-dispatch GPU timing so we stop guessing whether pointwise, spatial
   backward, GELU, attention, or command overhead dominates on the target device.
3. Selectively apply shape-specific kernels only where full-CLIP profiling proves
   they win, especially shared-weight pointwise for B=2/B=3 and staged/tiled
   `spatial_bwd`.
4. Only then attempt deeper train-plan memory reductions, mixed precision, or
   block fusion.

This order follows FUSED_JS_2's "semantic lock before optimization" rule and
FUSED_js's repeated lesson that false fusion can lose by reducing occupancy or
exceeding WebGPU's practical workgroup-memory limits.

## Source Map Studied

FUSED_js:

- `/Users/nicholasbardy/git/FUSED_js/docs/fusednets/ARCHITECTURE.md:66` defines
  separate `forward_only` and `ppo_step` modes, with forward designed around one
  dispatch and intermediates staying in registers/workgroup memory.
- `/Users/nicholasbardy/git/FUSED_js/docs/fusednets/ARCHITECTURE.md:136`
  describes one-workgroup-per-board forward mapping, shared-memory targets, and
  the exact batch-gradient problem when WGSL lacks storage-buffer f32 atomics.
- `/Users/nicholasbardy/git/FUSED_js/docs/fusednets/BACKWARD_MATH.md:325`
  gives the no-f32-atomics reduction pattern: owner workgroup, local partials,
  tree reduction, then update.
- `/Users/nicholasbardy/git/FUSED_js/docs/fusednets/NUMERICS.md:16` covers
  online softmax stability and the reduction-error budget.
- `/Users/nicholasbardy/git/FUSED_js/docs/fusednets/OPTIMIZER_FUSION.md:3`
  explains why writing a full gradient buffer, then reading it back in Adam, is
  wasted VRAM traffic.
- `/Users/nicholasbardy/git/FUSED_js/docs/fusednets/BUN_WEBGPU_WRAPPER.md:104`
  lists wrapper-side speed rules: allocate once, reuse bind groups, avoid sync,
  read back tiny status slices, batch with `dispatchWorkgroups(B)`.
- `/Users/nicholasbardy/git/FUSED_js/docs/fusednets/ACTIVE_WGPU_NOTES.md:65`
  documents the active recomputation tradeoff: conv2 is recomputed because the
  full activation alone exceeds a 16 KB workgroup-memory budget.
- `/Users/nicholasbardy/git/FUSED_js/shader_principles.md:7` is the clearest
  "VRAM bandwidth is the bottleneck" playbook, including shared staging,
  recomputation, optimizer fusion, f16 hazards, reductions, false fusion, and
  internal-loop batching.
- `/Users/nicholasbardy/git/FUSED_js/inference/kernels.wgsl:41` and
  `/Users/nicholasbardy/git/FUSED_js/inference/kernels.wgsl:120` show a compact
  forward shader with workgroup-staged activations and an online masked softmax
  reduction.
- `/Users/nicholasbardy/git/FUSED_js/training/kernels.wgsl:72` shows the older
  exact trainer's shared-memory inventory; `/Users/nicholasbardy/git/FUSED_js/training/kernels.wgsl:560`
  shows backward+Adam with one workgroup owning the exact batch update.
- `/Users/nicholasbardy/git/FUSED_js/kernels/plague_ppo.wgsl` is the active
  throughput kernel. It uses f16 embeddings, compact workgroup scratch, conv2
  recomputation, and in-place Adam state. The grep hits around lines 120-145,
  262-284, 375-508, and 738-1059 are the important areas.
- `/Users/nicholasbardy/git/FUSED_js/gpu_harness.ts:78` through
  `/Users/nicholasbardy/git/FUSED_js/gpu_harness.ts:223` show the minimal
  Bun/WebGPU harness: one device, compile one module, create pipelines and bind
  groups once, then write params and dispatch.
- `/Users/nicholasbardy/git/FUSED_js/tools/run_verify.ts:163` and
  `/Users/nicholasbardy/git/FUSED_js/tools/run_bench.ts:46` show the artifact
  style: machine-readable verification and benchmark JSON, explicit blocked
  states, warmups, and device notes.

FUSED_JS_2:

- `/Users/nicholasbardy/git/FUSED_JS_2/README-compile_network.md:3` says the
  "compiler" is really a staged prompt/codegen/verification workflow.
- `/Users/nicholasbardy/git/FUSED_JS_2/FUSEDNETS_SPEC.example.md:18` states the
  intended flavor: bespoke model-specific WebGPU/WGSL, analytical math, no
  autograd, no graph engine, minimal runtime glue.
- `/Users/nicholasbardy/git/FUSED_JS_2/v4_reference/README.md:3` defines the
  proof-carrying loop: normalized IR, contract/proofs/reference, then fused
  candidates and thresholds.
- `/Users/nicholasbardy/git/FUSED_JS_2/v4_reference/PHILOSOPHY.md:20` is the
  main process guidance. The most relevant line is
  `/Users/nicholasbardy/git/FUSED_JS_2/v4_reference/PHILOSOPHY.md:40`: optimize
  only after layer-level checks pass.
- `/Users/nicholasbardy/git/FUSED_JS_2/v4_reference/prompts/02_math_and_proofs.md:12`
  requires forward/backward math, numerics, and proof notes before code.
- `/Users/nicholasbardy/git/FUSED_JS_2/v4_reference/prompts/03_reference_oracle.md:38`
  requires per-layer activation and gradient comparisons, plus honest blocked
  status when WebGPU is missing.
- `/Users/nicholasbardy/git/FUSED_JS_2/v4_reference/prompts/06_optimize.md:18`
  gives the optimization search order: schedule/workgroup tuning, shared-memory
  reductions/reuse, vectorization, constant specialization, eliminate VRAM
  round-trips, then optional fp16/subgroup fast paths.
- `/Users/nicholasbardy/git/FUSED_JS_2/v4_reference/scripts/fusednets_v4/check_thresholds.py:42`
  enforces that blocked verification/bench results are not treated as wins.

Neural-Force-Field-Art CLIP and splat context:

- `tools/clip/README.md:1` through `tools/clip/README.md:58` summarize the
  MobileCLIP-S0 forward path: 99 dispatches, one submit, GPU-resident weights,
  6.2-6.6 ms/forward, and about 1.8x wall-clock vs ORT-WebGPU.
- `tools/clip/README.md:110` through `tools/clip/README.md:154` summarize the
  backward path: frozen weights, dL/dpixels only, recomputation in SE/attention,
  forward+backward around 20-40 ms, with `spatial_bwd` explicitly called a
  correctness-first perf follow-up.
- `src/clip/vision.ts:38` and `src/clip/vision.ts:198` define the runtime
  classes: `VisionEncoder` and `VisionTrainer`.
- `src/clip/vision_wgsl.ts:154` through `src/clip/vision_wgsl.ts:268` are the
  tiled pointwise forward matmul.
- `src/clip/vision_wgsl.ts:270` through `src/clip/vision_wgsl.ts:374` are the
  spatial conv path with staged weights and unrolled interior row loads.
- `src/clip/vision_wgsl.ts:376` through `src/clip/vision_wgsl.ts:433` fuse SE
  into one workgroup.
- `src/clip/vision_wgsl.ts:435` through `src/clip/vision_wgsl.ts:532` are the
  attention core: K/V staged in workgroup memory, softmax row private, output
  written channel-planar for the surrounding pointwise convs.
- `src/clip/vision_bwd_wgsl.ts:140` through `src/clip/vision_bwd_wgsl.ts:162`
  reuse the tiled pointwise body for `pw_bwd` with transposed weights.
- `src/clip/vision_bwd_wgsl.ts:224` through `src/clip/vision_bwd_wgsl.ts:280`
  implement gather-form `spatial_bwd`.
- `src/clip/vision_bwd_wgsl.ts:289` through `src/clip/vision_bwd_wgsl.ts:369`
  recompute SE internals and deliberately alias scratch lifetime to stay below
  the workgroup memory limit.
- `src/clip/vision_bwd_wgsl.ts:415` through `src/clip/vision_bwd_wgsl.ts:464`
  implement the `-cos` loss head as one workgroup reduction.
- `src/clip/vision_bwd_wgsl.ts:475` through `src/clip/vision_bwd_wgsl.ts:571`
  implement attention backward by recomputing softmax and storing only row max,
  denom, and row dot in workgroup memory.
- `src/clip/vision_batch.ts:70` through `src/clip/vision_batch.ts:261` are the
  replicated-activation batch experiment.
- `src/clip/vision_batch.ts:393` through `src/clip/vision_batch.ts:559` are the
  true batch-major trainer.
- `src/clip/vision_batch_wgsl.ts:1` through `src/clip/vision_batch_wgsl.ts:147`
  implement the current batch-z transform by rewriting generated WGSL slot/text
  accesses.
- `src/clip/vision_batch_pointwise.ts:104` through
  `src/clip/vision_batch_pointwise.ts:170` implement the shared-W pointwise
  microkernel.
- `docs/CLIP_BATCHING_NOTES.md:125` through `docs/CLIP_BATCHING_NOTES.md:171`
  record that batch-major forward+backward verifies exactly and is about 2x
  faster at B=3 and 2.9x faster at B=9 in the warmed run.
- `docs/SPLAT3D_PERF_NOTES.md:20` through `docs/SPLAT3D_PERF_NOTES.md:35` show
  CLIP is still 78-79 percent of sampled optimizer wall time after N-of-K view
  sampling.
- `tools/clip/fused_test.ts:85` through `tools/clip/fused_test.ts:165` are the
  per-step forward oracle and warmed bench.
- `tools/clip/bwd_test.ts:546` through `tools/clip/bwd_test.ts:660` are the
  backward unit tests, directional derivative gate, and forward/backward bench.
- `tools/clip/batch_major_train_bench.ts:115` through
  `tools/clip/batch_major_train_bench.ts:194` are the current batch-major train
  parity and timing harness.
- `tools/splat/page_smoke.mjs:1` through `tools/splat/page_smoke.mjs:180` is the
  real browser/WebGPU acceptance gate.

## What FUSED_js Teaches That Applies Directly

### 1. Evidence before speed

FUSED_JS_2 treats the reference as the "executable theorem" and refuses to treat
blocked harnesses as success. This repo already has most of that discipline:

- ORT per-step refs for the forward path.
- Float64 per-kernel backward references.
- End-to-end directional derivatives.
- Browser smoke that checks cosine improvement and nonblank canvas.

The gap is optimization evidence. For current CLIP work, a "win" should require:

- `PLAN=plan_train.json bun tools/clip/fused_test.ts` still passes.
- `bun tools/clip/bwd_test.ts` still passes.
- For batch changes, `BATCH=3` and `BATCH=9` batch-major parity still pass.
- Browser page smoke still passes for at least the default prompt.
- A benchmark artifact records warmup count, run count, device/adapter, mode,
  batch/view count, and whether it is wall-time or timestamp-query time.

### 2. Shared memory is a budget, not a reflex

FUSED_js repeatedly hit the 16 KB-ish default workgroup-memory reality. Its active
kernel recomputes conv2 instead of caching a large tensor because the tensor would
consume 18,432 bytes by itself. CLIP has the same pressure:

- Forward pointwise already uses 4 KB `xS` plus 4 KB `wS`.
- Shared-W pointwise at B=3 uses 12 KB X plus 4 KB W, exactly at the 16 KB budget.
- `se_bwd` had to share `gap` and `dpre2` storage because separate arrays would
  validate over the adapter limit.
- Attention K/V staging must keep `nTok * hd` below the workgroup limit.

Consequence: many "cache more" ideas should be framed as explicit budget tables
before code. If an optimization adds a workgroup array, list the byte count and
the target adapter limit in the PR note.

### 3. Recompute is often better than storing

FUSED_js uses recomputation as a first-class memory strategy. CLIP already does:

- SE backward recomputes gap/mid/scale from the saved SE input.
- Attention backward recomputes softmax probabilities from saved qkv instead of
  storing an nTok-by-nTok probability matrix.
- GELU backward saves only pre-activation because the derivative needs it exactly.

The next question is not "can we save more?" It is "which saved train activations
are expensive enough that recomputing them would reduce batch memory or bandwidth
without multiplying pointwise matmul cost?" That points to a production train
plan that differs from the debug train plan.

### 4. False fusion can lose

`shader_principles.md` warns that fusing loops with mismatched parallelism can
idle most lanes. This exactly matches the CLIP shared-W pointwise results:

- B=2 expansion layers won modestly.
- B=3 often lost despite saving W bandwidth.
- Contraction/residual layers were mixed.

Do not make shared-W pointwise a blanket replacement. It should be a profiled,
shape allowlist.

### 5. Optimizer fusion is less relevant to frozen CLIP, but the reduction
discipline matters

CLIP vision weights are frozen here, so there is no CLIP dW or Adam to fuse. But
FUSED_js's no-f32-atomics rule matters for future trainable CLIP-side adapters,
prompt weights, per-view weights, or learned loss scalars. If any trainable
parameter is added to the CLIP side, use owner-workgroup reduction and immediate
update, not storage-buffer scatter-add.

## Current CLIP Baseline: Strengths And Weak Points

### Strengths

- Runtime is already WebGPU-native and device-agnostic. `VisionEncoder` and
  `VisionTrainer` take a `GPUDevice`; they do not couple to tfjs.
- Command encoding is already one compute pass per forward/backward run. Within
  a run, dispatches are ordered in one pass.
- The forward codegen specializes every shape and offset as literals. This keeps
  WGSL simple and avoids uniform-driven runtime branching.
- Pointwise matmul is tiled with vec4 pixel quads and staged W/X tiles.
- Spatial forward has a fast interior path and unrolled register row loads.
- Backward uses exact analytic formulas and has strong per-kernel and
  directional tests.
- Batch-major train exists and verifies full dL/dpixels parity.

### Weak Points

- The current production optimizer likely still calls single-image
  `VisionTrainer` repeatedly for multiple views. Existing batch-major train work
  is not yet integrated into the 3D optimizer path.
- There is not yet per-dispatch GPU timing. Current split profiling tells us
  "CLIP forward/backward dominates," but not which CLIP steps dominate.
- `spatial_bwd` is explicitly correctness-first and gathers per input pixel.
  This is a prime target if timestamps show stem/depthwise backward is material.
- Train mode disables slot reuse and mirrors grad slots, which is correct for
  debugging but expensive for batch-major B=9 and browser memory pressure.
- `vision_batch_wgsl.ts` relies on source-string rewrites of generated WGSL. It
  also appears to have a local type-name typo at `src/clip/vision_batch_wgsl.ts:83`
  (`SlotBinding` vs `BatchBinding`). I did not change it, but any production
  batching work should first make this path compile-clean and less regex-fragile.
- Mixed precision has not been attempted for CLIP. The 82 MB train weights and
  large activation buffers make it tempting, but CLIP cosine gradients are
  sensitive enough that this needs a feature-gated evidence path.

## Ranked Optimization Ideas

### 1. Integrate `BatchMajorVisionTrainer` into the 3D optimizer path

Why this is first:

- It is already implemented and verified in isolation.
- It maps directly to the current bottleneck: multiple CLIP calls per optimizer
  step for multiple camera views.
- It does not require changing CLIP math.
- It aligns with `docs/SPLAT3D_PERF_NOTES.md:30`: N-of-K sampling is already a
  major wall-clock lever, and CLIP remains the dominant share.

Concrete design:

- Add a batch mode to the splat optimizer that renders selected views into
  `BatchMajorVisionTrainer.inputBuffer` lanes, runs one batched CLIP
  forward+loss+backward, then routes each lane's `inputGradBuffer` slice to the
  matching raster backward view.
- Keep the current single-image path as fallback.
- Default the browser optimizer to 3-of-9 views with shuffled epochs, because
  that already cuts wall time and B=3 batch-major train has a measured win.
- Use B=9 for explicit high-quality/evaluation modes rather than every step.

Risks:

- Raster integration is the hard part: each view's dL/dpixels must feed the
  matching camera/raster buffers without aliasing the wrong lane.
- Batch-major CLIP increases activation memory by B. Browser adapter memory may
  be a limiter on integrated B=9 even if Bun benchmarks pass.
- If text embeddings differ per view, the text buffer layout already supports
  `[batch][textDim]`, but the prompt/camera assignment must be explicit.

Benchmark:

- Compare current single-view loop vs batch-major integrated loop for B=3 and
  B=9 on the same 4096-splat, 256px, 9-camera scene.
- Metrics: normal step average, sampled split profile, CLIP fwd/bwd split,
  raster fwd/bwd split, cosine trajectory after 150 steps, screenshot variance.
- Gates: no regression in `tools/splat/page_smoke.mjs`; cosine still rises by
  the configured margin.

### 2. Add per-dispatch CLIP timing with labels and shape grouping

Why:

- FUSED_JS_2 says optimize only after metrics identify the bottleneck.
- Current CLIP work has enough kernels that aggregate fwd/bwd times are not
  actionable.
- Shared-W pointwise already showed shape-specific results; we need full-CLIP
  shape attribution before integrating variants.

Concrete design:

- Add a profiling wrapper that can encode one CLIP run with timestamp-query
  writes around each dispatch if the adapter supports timestamp queries.
- If timestamp queries are unavailable in browser WebGPU, use a fallback "one
  labeled dispatch N times" microbench harness in Bun, like
  `tools/clip/pointwise_batch_bench.ts`, but generated for every dispatch label.
- Emit a JSON table keyed by `spec.label`, `step index`, `kind`, `shape`,
  `batch`, `mode` (`forward`, `backward`, `train`), and time.
- Group labels into pointwise expansion, pointwise contraction/residual,
  spatial forward, spatial backward, SE, attention, GELU, head, loss.

Risks:

- Timestamp-query support is not uniformly exposed in browsers. Do not make it a
  required correctness gate.
- Per-dispatch timestamp instrumentation can perturb scheduling. Use it for
  relative attribution, not absolute end-to-end claims.
- Microbench timings can lie if they omit upstream cache state. Confirm any
  candidate change in the full CLIP bench before landing.

Benchmark:

- Run B=1, B=3, B=9.
- Run `plan.json` and `plan_train.json`.
- Include warmed single-image `VisionTrainer`, batch-major train, and integrated
  splat optimizer once available.

### 3. Selectively integrate shared-W pointwise kernels

Current evidence:

- `src/clip/vision_batch_pointwise.ts` proves the concept.
- `docs/CLIP_BATCHING_NOTES.md:203` through `docs/CLIP_BATCHING_NOTES.md:223`
  show exact parity but mixed timings.
- B=2 expansion layers improved modestly.
- B=3 sometimes lost, likely due to 192-thread workgroups and full 16 KB
  workgroup memory use.

Concrete design:

- Do not replace `pointwise()` globally.
- Add a codegen decision table keyed by `(cin, cout, P, hasResidual, batch)`.
- Start with allowlisted expansion shapes that win in full-profile timing:
  likely `64->192 @64x64` and `256->768 @16x16` for B=2, possibly B=3 only if
  timestamp and full CLIP agree.
- Keep z-batch as default for contraction/residual shapes until measured wins.
- Require exact parity against z-batch for every selected shape.

Risks:

- Workgroup size `(8,8,B)` can reduce occupancy or exceed adapter limits.
- More shader variants increase compile time and codegen complexity.
- The source-string batch transform is fragile; integrate shared-W through the
  actual pointwise emitter, not by patching emitted strings after the fact.

Benchmark:

- Full CLIP B=3/B=9 forward+backward before and after the allowlist.
- Per-shape microbench only informs the allowlist; it does not prove end-to-end
  improvement.

### 4. Stage and tile `spatial_bwd`

Why:

- `tools/clip/README.md:153` calls gather `spatial_bwd` an obvious perf follow-up.
- `src/clip/vision_bwd_wgsl.ts:224` currently assigns one thread per input pixel
  and loops over output channels/taps, reading dY and W from global memory.
- FUSED_js forward kernels stage board/patch data and weights to avoid scattered
  global reads inside nested loops.

Concrete design options:

Option A: Weight staging only.

- Mirror forward spatial conv's `wk` staging.
- One workgroup owns `(input channel, tile of input pixels)` or `(output channel,
  tile)` depending on cpg/cpgOut.
- Stage the small per-output-channel `WK` footprint when cpg/cpgOut is small.
- Keep boundary handling simple at first.

Option B: dY tile staging.

- For stride-1/k3/k7 depthwise, a tile of input pixels maps to a slightly larger
  tile of output pixels. Stage that dY neighborhood into workgroup memory.
- This is more complex but attacks the true repeated global-read pattern.

Option C: Specialized stem backward.

- The stem's `cin=3` path produces final dL/dpixels and runs over 256x256.
- It may dominate because it touches the largest spatial area.
- Write a stem-specific backward with a 2D tile, staged dY, and staged weights.

Risks:

- Border and stride-2 parity rules are easy to get subtly wrong.
- Dynamic indexing in private arrays caused Metal spills in forward spatial conv;
  preserve unrolled/literal-index style where possible.
- Workgroup memory can disappear quickly for k7 and grouped convs.

Benchmark:

- Unit-test against the existing `spatial_bwd` output on small random shapes.
- Then run full `bwd_test.ts`.
- Timestamp stem, k3 depthwise, k7 depthwise, and grouped `spatial_bwd` separately.

### 5. Create separate "debug train" and "production train" plans

Current state:

- Train mode saves every activation and mirrors grad slots. This makes backward
  straightforward and per-step verification strong.
- That is not necessarily the best production memory/bandwidth layout.

FUSED-derived idea:

- Keep the current plan as the debug/reference plan.
- Add a production plan that recomputes selected cheap intermediates and reuses
  slots where possible, while preserving the same final dL/dpixels.
- This is the CLIP analogue of FUSED_js's "store less, recompute conv2" decision.

Candidates:

- Continue saving GELU pre-activations where needed. Do not recompute these
  unless the exact forward preactivation is easily reproduced and verified.
- Recompute SE internals as now; no need to save extra SE scratch.
- Recompute attention probabilities as now; no nTok-by-nTok storage.
- Consider recomputing qkv-adjacent intermediates only if profiling shows their
  slot traffic matters more than the pointwise recompute cost.
- Investigate grad-slot aliasing where backward writer order proves lifetimes do
  not overlap. The current accumulate flag already gives the dataflow needed for
  a liveness pass.

Risks:

- Losing per-step ORT refs for production fused steps makes debugging harder.
- Recompute can backfire if it repeats pointwise matmuls.
- Slot aliasing bugs are often second-run bugs, just like the pinned input-slot
  bug described in `tools/clip/README.md:62` through `tools/clip/README.md:69`.

Implementation guardrail:

- The production plan should be generated by `compile_plan.py`, not hand-edited.
- It should keep a machine-readable mapping back to debug-plan step names so
  failures can be bisected.

### 6. Fuse tiny elementwise kernels only after timing proves dispatch/scratch cost

Candidates:

- `gelu` forward and `gelu_bwd`.
- `residual_bwd`.
- Some layer-scale/residual epilogues around pointwise convs.
- Loss head plus final head backward, only if they show up.

Why not first:

- Current runtime already encodes all dispatches in one compute pass, so CPU
  dispatch overhead is lower than a normal graph runtime.
- Fusing GELU into conv in train mode removes the saved pre-activation boundary
  that made forward train refs easy.
- Elementwise kernels may be small compared with pointwise matmul and spatial
  backward.

Safer path:

- Preserve debug split-GELU plan.
- Add production fused variants behind a plan flag.
- Verify production dL/dpixels against debug dL/dpixels for deterministic inputs.

### 7. Mixed precision: f16 storage, f32 reductions, feature-gated

Why it is tempting:

- CLIP train weights are about 82 MB because pointwise transposed copies are
  packed too.
- Activation/grad buffers scale with batch.
- FUSED_js saw fp16 storage as useful for bandwidth-dominated embedding tables,
  while keeping accumulations/moments in f32.

Potential CLIP precision modes:

Mode A: f16 weights only, f32 activations.

- Pack selected weights as f16, load/cast to f32 in WGSL.
- Keep pointwise accumulators f32.
- Start with spatial/depthwise and maybe pointwise weights only after parity
  testing.

Mode B: f16 activations for selected saved slots, f32 gradients/accumulators.

- More memory relief, higher risk for dL/dpixels quality.
- Only consider after exact f32 production path is profiled.

Mode C: f16 text embedding or loss-head inputs.

- Low bandwidth impact, not worth first.

Risks:

- CLIP cosine loss can have tiny directional derivatives. The existing
  `bwd_test.ts` epsilon notes show fp32 noise already matters.
- f16 activations could perturb gradient enough to harm optimization even if
  final embedding cosine remains high.
- Browser feature availability varies. `shader-f16` must be optional, not a hard
  requirement for the app.

Gates:

- Forward final embedding cosine vs f32 >= 0.9999, not just 0.999.
- dL/dpixels cosine vs f32 >= 0.999 and relLinf within an agreed tolerance.
- Prompt optimization smoke must show equal or better cosine trajectory across
  several seeds/prompts, not only one cat prompt.

### 8. Attention backward profiling and possible tiling

Current attention backward:

- One workgroup per head.
- Phase 1 recomputes each query row and stores only max/denom/rowdot.
- Phase 2 re-reads q/dO globally for every key/value token.
- This avoids storing an nTok-by-nTok probability matrix.

Potential optimization:

- If timestamps show attention backward is material, stage chunks of Q and dO in
  workgroup memory during phase 2 to reduce repeated global reads.
- Use smaller token tiles if `nTok * hd` exceeds practical shared memory.
- Consider a two-dispatch attention backward only if one-dispatch recompute is
  clearly bandwidth-bound and the intermediate traffic is lower than re-reading.

Risks:

- Private arrays over `nTok` and `hd` can spill. Any change must inspect shader
  performance, not just source neatness.
- Attention appears only in two blocks; pointwise/spatial likely dominate first.

### 9. Browser/WebGPU harness hardening

Borrow from FUSED_js:

- Always print adapter vendor/architecture/feature flags.
- Distinguish blocked vs fail. No adapter or missing timestamp support is
  blocked for that specific benchmark mode, not a correctness failure.
- Read back tiny slices only for synchronization.
- Keep warmup counts high enough to pay Metal pipeline JIT.
- Preallocate max-batch buffers when sweeping B.

Concrete additions:

- `tools/clip/profile_dispatches.ts`: Bun/WebGPU dispatch timing over plan
  labels and batch.
- `tools/clip/browser_clip_bench.mjs`: Chrome/Puppeteer page that runs the same
  CLIP bench in the browser device context, because Bun/Dawn and Chrome/Dawn can
  differ.
- `tools/clip/bench_matrix.mjs`: run B=1/3/9, forward/train, with JSON output.

### 10. Future trainable CLIP-side adapters

Not a current requirement, but if a lightweight trainable adapter or learned
prompt/view weighting is added:

- Use FUSED_js's owner-workgroup reductions.
- Do not use storage-buffer f32 atomics.
- Fuse optimizer update immediately after workgroup reduction.
- If sparse rows exist, explicitly define sparse Adam vs dense Adam semantics.
- Keep this separate from frozen MobileCLIP weights.

## Detailed Benchmark Design

### Correctness gates

Run before and after any kernel change:

```bash
PLAN=plan_train.json bun tools/clip/fused_test.ts
bun tools/clip/bwd_test.ts
BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
BATCH=9 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
```

For browser/integration changes:

```bash
npx parcel build --no-scope-hoist --public-url ./ src/splat.html
node tools/splat/page_smoke.mjs
```

### Timing matrix

Minimum matrix:

| Mode | Batch/views | Harness | Required metric |
| --- | ---: | --- | --- |
| CLIP forward | 1 | `fused_test.ts` | ms/forward, CPU encode+submit |
| CLIP fwd+bwd | 1 | `bwd_test.ts` | ms/train, fwd+bwd/fwd ratio |
| Batch CLIP fwd+bwd | 3 | `batch_major_train_bench.ts` | ms/batch, ms/image |
| Batch CLIP fwd+bwd | 9 | `batch_major_train_bench.ts` | ms/batch, ms/image |
| Integrated splat step | 3/9 | page or Bun optimizer harness | normal step avg, split profile |
| Integrated splat step | 9/9 | page or Bun optimizer harness | normal step avg, split profile |

Full matrix:

- Adapter: Apple Metal, Chrome WebGPU, and any available non-Apple discrete GPU.
- Batch: 1, 2, 3, 5, 9.
- Views: 1/9, 3/9, 5/9, 9/9.
- Prompt set: cat, dog, diagram/control, and one abstract/art-style prompt.
- Seeds: at least 3 splat seeds for optimization quality checks.

### Metrics to record

- `adapter`, `browser/runtime`, `shader-f16`, timestamp-query support.
- `plan`, `weights`, `batch`, `views_per_step`, `splat_count`, `resolution`.
- `warmup_runs`, `timed_runs`, and whether there was an explicit queue sync.
- Wall ms and timestamp ms if available.
- CPU encode+submit ms separately from GPU-inclusive wall time.
- Per-dispatch label timing if profiling mode.
- Peak buffer bytes for activation/grad/weights/text.
- Correctness metrics: relLinf, cosine, finite count, max abs grad.
- Optimization quality: initial/final cosine after fixed steps, screenshot
  variance, and possibly median cosine delta across seeds.

### Anti-bench traps

- Do not compare a Bun result to a browser result without saying so.
- Do not include first-use pipeline compilation in steady-state runtime.
- Do not claim a GPU win from a CPU fallback baseline.
- Do not read back full buffers just to synchronize unless the app also pays it.
- Do not accept microbench wins until full CLIP and integrated optimizer wins.

## Implementation Order

### Phase 0: Baseline capture

1. Record current dirty worktree status in the optimization log.
2. Run or collect existing outputs for:
   - single forward,
   - single fwd+bwd,
   - B=3/B=9 batch-major train,
   - current 3D optimizer split profile.
3. Make sure `vision_batch_wgsl.ts` is compile-clean before building on it. The
   apparent `SlotBinding` type-name issue should be fixed by whoever owns app
   code edits; this memo only notes it.

### Phase 1: Production integration of existing batch-major train

1. Add a batched CLIP path to the 3D optimizer behind a flag.
2. Render selected views into batch lanes.
3. Run `BatchMajorVisionTrainer`.
4. Route lane gradients back to the matching raster backward view.
5. Make 3-of-9 the default if quality and smoke gates remain good.

Expected payoff: largest near-term wall-time reduction because it attacks
repeated CLIP calls directly with already-verified code.

### Phase 2: Profiling harness

1. Add per-dispatch profile JSON.
2. Group timings by label/shape.
3. Profile B=1, B=3, B=9 before writing new kernels.
4. Pick the top two bottleneck families only.

Expected payoff: prevents spending a week optimizing a 3 percent kernel.

### Phase 3: Selective pointwise variants

1. Turn the existing shared-W microkernel into an emitter option.
2. Allowlist only measured winning shapes.
3. Re-run all batch parity and full benchmarks.

Expected payoff: modest but real if pointwise dominates. Risk is manageable with
an allowlist.

### Phase 4: `spatial_bwd` staging

1. Start with stem-specific backward if profiling confirms it matters.
2. Add staged weights, then staged dY tiles if still needed.
3. Keep exact gather output parity against the old kernel.

Expected payoff: potentially meaningful for backward, especially on 256x256
input-resolution stem gradients.

### Phase 5: Production train-plan memory work

1. Add liveness/alias analysis for grad slots.
2. Add recompute choices where byte savings outweigh compute.
3. Keep debug train plan untouched.
4. Verify production dL/dpixels against debug dL/dpixels.

Expected payoff: enables larger B and reduces browser memory pressure. It may
also reduce bandwidth if slot writes were a bottleneck.

### Phase 6: Mixed precision

1. Add f16 weights only, feature-gated.
2. Keep f32 accumulators and f32 loss math.
3. Compare against f32 CLIP gradients and prompt optimization quality.
4. Only then consider selected activation compression.

Expected payoff: bandwidth/memory reduction. Highest numerical risk.

### Phase 7: Optional advanced paths

- Subgroup reductions or matrix-style tiling if available and measured.
- Attention backward tiling if it appears in top timings.
- Tiny-kernel fusion if dispatch/timestamp data justifies it.
- Future adapter optimizer fusion using FUSED_js owner reductions.

## Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Batch lane mismatch | Wrong view gets wrong image gradient | Lane-indexed tests with synthetic per-lane text/input; compare against single trainer |
| Workgroup memory over limit | Pipeline validation fails on browser/Metal | Byte budget in docs, validation error scopes, feature-gated variants |
| Occupancy loss from shared-W | B=3 already mixed/lossy | Shape allowlist, full CLIP benchmark required |
| Regex WGSL rewriting fragility | Batch transform can corrupt generated code | Move batch offsets into emitters or at least make the transform typed and tested |
| Per-step refs lost by fusion | Harder to debug incorrect production kernels | Keep debug plan; compare production final outputs/gradients to debug path |
| f16 gradient drift | CLIP loss derivatives are small/noisy | f32 accumulations, strict dL/dpixels cosine, multi-prompt smoke |
| Browser/Bun divergence | A Bun win may not reproduce in Chrome | Add browser bench harness and keep page smoke as integration gate |
| Hidden sync/readback | Bench claims can include artificial stalls | Tiny sync readback only, separate CPU encode and GPU-inclusive wall |
| Existing dirty tree | Multiple workers editing | Touch only owned files during note phase; app-code owner should merge deliberately |

## Concrete File-Level Recommendations

`src/clip/vision_batch_wgsl.ts`

- Replace the source-string offset injection with emitter-level batch awareness
  if this becomes production-critical.
- At minimum, fix the apparent `SlotBinding` type typo and add a test that
  generates all train dispatches for B=1/3/9.
- Add assertions that `workgroups[2] == 1` and all slot/text buffer strides are
  4-byte/vec4 aligned, which the file already partly does.

`src/clip/vision_batch.ts`

- `BatchMajorVisionTrainer` is the main production target. Add integrated
  optimizer tests that read each lane's `inputGradOffsetBytes(lane)` and feed it
  through the raster chain.
- Keep `ReplicatedBatchVisionTrainer` as a diagnostic harness, not a production
  path.

`src/clip/vision_batch_pointwise.ts`

- Treat as a prototype for a selective pointwise emitter.
- Add full-shape metadata and reject B=3 for shapes where memory/occupancy
  loses.
- Consider variants with smaller tiles for B=3/B=4 if profiling shows W bandwidth
  is still worth chasing.

`src/clip/vision_bwd_wgsl.ts`

- Profile before changing.
- First likely target: `spatialBwd`.
- Second likely target: `attnCoreBwd` only if timestamp data says it matters.
- Keep `lossBwd` simple unless timestamp data says otherwise; it is only 512 dim.

`tools/clip/*bench*.ts`

- Add JSON output modes similar to FUSED_js `run_bench.ts`.
- Include adapter info, warmup/run counts, plan file, batch, and exact command.
- Split CPU encode+submit from GPU-inclusive wall time consistently.

`tools/splat/page_smoke.mjs`

- Keep as acceptance, not as microbenchmark.
- Add optional perf JSON extraction from `window.__splat` once integrated batch
  CLIP lands.

`docs/CLIP_BATCHING_NOTES.md` and `docs/SPLAT3D_PERF_NOTES.md`

- Continue using these as the optimization attempt log.
- Add a "reverted" section for every failed kernel idea, matching
  FUSED_JS_2's trace-retention philosophy.

## Final Priority List

1. Integrate existing batch-major train into the splat optimizer for 3-of-9 and
   9-of-9 views.
2. Add per-dispatch CLIP profile JSON.
3. Selectively enable shared-W pointwise for only measured winning shapes.
4. Optimize `spatial_bwd`, starting with stem-specific staging if it is hot.
5. Build a production train plan with slot aliasing/recompute after debug parity
   remains stable.
6. Try feature-gated f16 weights with f32 math.
7. Only pursue attention tiling, subgroup features, or tiny-kernel fusion if the
   profile points there.

The near-term goal should be an integrated B=3 optimizer step that beats the
current 3/9 single-CLIP loop while preserving the page smoke's cosine rise. After
that, the profile will say whether the next wall-clock win is pointwise bandwidth,
spatial backward, memory footprint, or precision.
