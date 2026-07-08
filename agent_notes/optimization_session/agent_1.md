# Agent 1 notes - AlphaGOJS lessons for CLIP speedups

Date: 2026-07-08

Scope: this is an analysis note only. I did not modify app code. My only write target is this file.

## Executive summary

AlphaGOJS is not useful here because its model is similar to CLIP. It is useful because it shows a disciplined way to make WebGPU kernels fast: isolate the true whole-loop bottleneck, avoid attractive changes that do not move wall time, use variant kernels, validate deterministically at tiny batch, and only then promote. The highest-value AlphaGOJS transfer to `src/clip` is not "use f16 everywhere"; it is "turn repeated independent work into a single correctly occupied dispatch structure, then use shared memory only where it removes proven repeated traffic or private-memory pressure."

For this repo, the most concrete conclusion is:

1. The fastest near-term CLIP win is already mostly built: integrate `BatchMajorVisionTrainer` into the 3D optimizer. Current 3D profiling says CLIP is 78-79% of sampled step time, and existing batch-major train benches show about 2x at B=3 and about 2.9x at B=9 versus repeated single-image forward+backward.
2. The next CLIP-kernel win should be selective, not blanket: profile by dispatch label/shape, then replace only winning pointwise shapes with shared-W batch kernels or tuned tiles. The existing shared-W microbench proves correctness but shows mixed performance.
3. The biggest ambitious kernel rewrites are likely: train-mode pointwise/GELU fusion, tiled/staged `spatial_bwd`, and attention-backward register-pressure reduction. These are direct analogues to AlphaGOJS's cell-parallel and vec4 wins: change work structure first, then precision.
4. f16 is worth an experimental branch for CLIP because CLIP pointwise matmuls are bandwidth-heavy and weights are frozen. But AlphaGOJS's notes argue f16 should follow measurement and exact-math rewrites, not lead.

## AlphaGOJS patterns that matter

### Whole-loop measurement beats local intuition

AlphaGOJS's benchmark harness measures the same browser code path under Bun/WebGPU (`/Users/nicholasbardy/git/AlphaGOJS/BENCHMARKS.md:3`). It reports full `runStep()` cost, not isolated shader microseconds (`/Users/nicholasbardy/git/AlphaGOJS/BENCHMARKS.md:23`). That matters because some intuitive GPU changes did not move whole-loop throughput: B=256 to B=512 was flat in games/sec, fused readbacks were flat, and in-rollout polling was not the bottleneck (`/Users/nicholasbardy/git/AlphaGOJS/BENCHMARKS.md:115`).

CLIP has the same risk. The current CLIP forward is already about 6.2-6.6 ms after tiling and attention restructuring (`tools/clip/README.md:41`). Forward+backward is about 20-40 ms, around 2.7x forward (`tools/clip/README.md:149`). In 3D optimization, CLIP forward+backward dominates sampled wall time: at 3/9 views, CLIP is 24.07 + 34.96 ms out of 75.57 ms sampled total (`docs/SPLAT3D_PERF_NOTES.md:24`). So the speed target is not "make a toy pointwise kernel faster"; it is "reduce CLIP calls and the dominant dispatch groups in the actual optimizer."

### Occupancy and private memory beat raw FLOP counting

AlphaGOJS found that the PPO backward was slow because one serial patch loop used only 9 of 64 threads, carried large private arrays, and hit hundreds of barriers (`/Users/nicholasbardy/git/AlphaGOJS/docs/perf/BACKWARD_ANALYSIS.md:11`). The fix was not a math shortcut; it rewrote the loop so all 576 cells ran across all 64 threads, wrote per-cell deltas into shared memory, then gathered gradients without scatter races (`/Users/nicholasbardy/git/AlphaGOJS/docs/perf/CELLPARALLEL_GUIDE.md:7`).

The live AlphaGOJS kernel shows the promoted structure. The gated cell-parallel branch writes `sh_patch_delta2_all` in phase 1 (`/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:627`), gathers `DW2` over all patches (`/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:674`), and gathers `sh_bar_a1` by output index (`/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:690`). The serial fallback remains for D=16 (`/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:715`).

For CLIP, the analogous places to inspect are not the already-tiled pointwise forward path. They are the kernels where a single workgroup is responsible for a lot of work:

- `headStep` is one workgroup for GAP plus a 1024->512 projection (`src/clip/vision_wgsl.ts:535`).
- `headBwd` is one workgroup for a 512-to-1024 gradient projection plus broadcast (`src/clip/vision_bwd_wgsl.ts:371`).
- `seStep` and `seBwd` do multi-phase channel MLPs inside one workgroup (`src/clip/vision_wgsl.ts:376`, `src/clip/vision_bwd_wgsl.ts:282`).
- `attnCoreBwd` uses one workgroup per head and per-thread private arrays for query vectors, probabilities, dP, dV, and dK (`src/clip/vision_bwd_wgsl.ts:466`).

Those deserve profiling before more tile tuning, because they have the AlphaGOJS smell: lots of serial loop work in one workgroup, possible private-memory pressure, and barriers.

### Exact vec4 rewrites were promoted before f16

AlphaGOJS promoted exact `vec4<f32>` forward rewrites into baseline after validating they stacked with cell-parallel backward (`/Users/nicholasbardy/git/AlphaGOJS/docs/perf/ITERATION_LOG.md:35`). The live kernel uses `vec4<f32>` in `conv2_at_patch` (`/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:231`), `conv1` (`/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:275`), and the fuse path (`/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:317`).

CLIP already applies that lesson in the pointwise path. `pointwiseTiledMain` stages 256 vec4 activations and 256 vec4 weights, then accumulates four output channels for one pixel-quad per thread (`src/clip/vision_wgsl.ts:165`, `src/clip/vision_wgsl.ts:177`). The depthwise/general spatial forward path also produces four horizontal pixels per thread and fully unrolls interior taps to avoid Metal scratch spills (`src/clip/vision_wgsl.ts:288`).

So the next exact-math vectorization opportunities are narrower:

- `spatial_bwd` is scalar gather form today (`src/clip/vision_bwd_wgsl.ts:224`). It should get the same "4 horizontal pixels per thread + staged weights + interior fast path" treatment as forward `spatialConv`.
- `attnCoreBwd` stores per-head data as scalar f32 arrays (`src/clip/vision_bwd_wgsl.ts:493`). Rewriting hd=32 loops as `vec4f` chunks can reduce instruction count and private-array indexing.
- `headStep` and `headBwd` are scalar dot loops (`src/clip/vision_wgsl.ts:555`, `src/clip/vision_bwd_wgsl.ts:388`). A small vec4 dot kernel or tiled GEMV could be exact and low-risk.

### f16 is a feature, not a religion

AlphaGOJS enables f16 (`/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:7`) and stores some transition/embed data as f16 (`/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:95`, `/Users/nicholasbardy/git/AlphaGOJS/src/fused_ppo.wgsl:205`). But its own analysis ranks fp16 behind subgroups, parallelization, and fusion because f16 was expected to save only marginal time on the then-current bottleneck (`/Users/nicholasbardy/git/AlphaGOJS/docs/perf/ALGO_APPROACHES.md:337`).

CLIP is different enough that f16 deserves a real experiment, but not a blind rewrite. CLIP has frozen weights, no CLIP dW, and a static validation oracle. That makes f16 weights/activations less risky than in AlphaGOJS PPO. However, `tools/clip/README.md` reports the current path is already heavily optimized and verified to tight tolerances (`tools/clip/README.md:43`), and the batch notes show GPU wall time is still not universally sublinear even when dispatches are batched (`docs/CLIP_BATCHING_NOTES.md:155`). f16 should come after dispatch/occupancy profiling and be gated by `adapter.features.has("shader-f16")`.

## Current CLIP implementation map

### What is already good

The CLIP compiler has the right architecture. ONNX is only understood in `tools/clip/compile_plan.py`, which canonicalizes the graph and throws on unexpected structure (`tools/clip/compile_plan.py:1`). Weight packing is static, including pointwise `[Cin][Cout]` transposed storage for vec4 loads (`tools/clip/compile_plan.py:13`). The input slot is pinned to keep repeated `run()` calls correct (`tools/clip/compile_plan.py:146`). Train mode keeps every activation because backward needs saved values (`tools/clip/compile_plan.py:39`, `tools/clip/compile_plan.py:114`).

The forward codegen is also strong:

- Specialized dispatches bake shapes and offsets as literals (`src/clip/vision_wgsl.ts:1`).
- Pointwise matmul uses 32-Ci by 32-Cout shared-memory tiles (`src/clip/vision_wgsl.ts:154`).
- Spatial conv stages each channel's weights in workgroup memory (`src/clip/vision_wgsl.ts:310`).
- Attention core stages K then V through shared memory and keeps one score row in registers (`src/clip/vision_wgsl.ts:436`).
- Runtime encodes all forward dispatches into one compute pass (`src/clip/vision.ts:128`).

Backward is correctness-first but already well structured:

- `pw_bwd` reuses the forward tiled pointwise body over transposed weights (`src/clip/vision_bwd_wgsl.ts:136`).
- Grad slots mirror activation slots with first-writer overwrite/later-writer accumulate, avoiding a global zero fill (`tools/clip/compile_plan.py:553`).
- `VisionTrainer` encodes forward, loss, and backward into one compute pass (`src/clip/vision.ts:295`).

Batch work is more advanced than the README's earliest state:

- `batchTrainDispatches()` now applies the z-batch transform to forward and backward specs (`src/clip/vision_batch_wgsl.ts:126`).
- `BatchMajorVisionTrainer` owns `[batch][slotFloats]` slot buffers and batched text buffers (`src/clip/vision_batch.ts:393`).
- Existing notes say gradient parity is exact for B=2, B=3, and B=9, and the warmed train bench shows B=3 about 2x faster and B=9 about 2.9x faster (`docs/CLIP_BATCHING_NOTES.md:148`, `docs/CLIP_BATCHING_NOTES.md:155`).

### Where the weak spots likely are

The current 3D optimizer still uses a single-image `VisionTrainer` inside a loop over views:

- Copy per-view text to `trainer.textBuffer` (`src/splat3d/optimize.ts:137`).
- Render one raster view (`src/splat3d/optimize.ts:138`).
- Copy raster image into the single CLIP input (`src/splat3d/optimize.ts:139`).
- Run full CLIP forward+backward (`src/splat3d/optimize.ts:140`).
- Copy the single CLIP image gradient back to raster (`src/splat3d/optimize.ts:141`).
- Run per-view raster backward (`src/splat3d/optimize.ts:142`).

That is the biggest gap between existing CLIP speed work and production behavior.

The current true batch-major fork still mostly duplicates weight traffic per lane. It adds a base offset to every slot binding and dispatches `workgroups.z = batch` (`src/clip/vision_batch_wgsl.ts:77`, `src/clip/vision_batch_wgsl.ts:110`). That reduces launch list multiplicity and improves occupancy, but each z-lane pointwise workgroup still stages the same W tile. The separate microkernel proves an alternate structure: put batch lanes in `local_invocation_id.z`, stage W once, and stage per-lane X tiles (`src/clip/vision_batch_pointwise.ts:104`). The bench says this is exact but shape-specific (`docs/CLIP_BATCHING_NOTES.md:203`).

## Proposed implementation order

### Phase 0: Add dispatch-level measurement before more kernel edits

Priority: immediate.

Rationale: AlphaGOJS only got useful after it measured the actual bottleneck and kept iteration logs. CLIP has whole-forward and whole-train benches, but the next decisions need per-dispatch or grouped timing by label/shape.

Concrete work:

- Add a profiling-only tool, probably `tools/clip/dispatch_profile.ts`, that compiles the train plan and times dispatch groups by label. Do not wire into the app first.
- Use the existing `DispatchSpec.label` fields from `src/clip/vision_wgsl.ts` and `src/clip/vision_bwd_wgsl.ts`.
- Support group modes:
  - all pointwise forward by shape, e.g. `pw 64->192 @64x64`, `pw 256->768 @16x16`.
  - all `pw_bwd` by shape.
  - all `spatial_bwd`, especially the stem `3<-64 @256x256`.
  - all `gelu` and `gelu_bwd`.
  - `head`, `head_bwd`, `se`, `se_bwd`, `attn_core`, `attn_core_bwd`.
- Start with split-pass wall timing if `timestamp-query` is not available. AlphaGOJS's profiling notes explicitly call out timestamp-query uncertainty and recommend ablation when needed (`/Users/nicholasbardy/git/AlphaGOJS/docs/perf/PROFILING_TOOLS.md:157`).

Useful existing commands to keep green:

```bash
PLAN=plan_train.json bun tools/clip/fused_test.ts
bun tools/clip/bwd_test.ts
BATCH=3 RUNS=2 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
BATCH=9 RUNS=2 WARMUP=2 bun tools/clip/batch_major_train_bench.ts
```

Decision gate: no kernel rewrite should be promoted unless it improves full CLIP forward+backward or the 3D optimizer sampled step, not just one microbench in isolation.

### Phase 1: Integrate batch-major CLIP training into `Splat3DOptimizer`

Priority: highest production ROI.

Rationale: This is the one large speedup already verified. The current 3D optimizer serializes CLIP over selected views (`src/splat3d/optimize.ts:131`). Existing train batch-major parity and timing are strong: B=3 about 102.55 ms repeated versus 49.48 ms batch-major; B=9 about 389.66 ms repeated versus 134.63 ms batch-major (`docs/CLIP_BATCHING_NOTES.md:155`). Since CLIP is 78-79% of current sampled wall time (`docs/SPLAT3D_PERF_NOTES.md:30`), this should move real app performance.

Concrete work:

- Replace `readonly trainer: VisionTrainer` with a batch-capable trainer for 3D, likely `BatchMajorVisionTrainer` from `src/clip/vision_batch.ts:398`.
- Keep `viewsPerStep` as the batch size cap. For fixed 9 cameras, create either:
  - one B=9 trainer and use only the first `k` lanes for 3/9 or 5/9, or
  - separate B=3/B=5/B=9 trainers if memory and compile cost are acceptable.
- In `Splat3DOptimizer.step()`:
  - Clear raw gradients once.
  - For each selected view, render raster forward into `raster.image`, then copy into `batchTrainer.inputBuffer` at `slotOffsetBytes(lane, inputSlot)` (`src/clip/vision_batch.ts:493`).
  - Copy each selected view's text embedding into batched `textBuffer` at `textOffsetBytes(lane)` (`src/clip/vision_batch.ts:506`).
  - Call `batchTrainer.encode(enc, { backward: true })` once (`src/clip/vision_batch.ts:536`).
  - For each selected lane, copy `inputGradBuffer` from `inputGradOffsetBytes(lane)` back to `raster.gradImage`, then call `raster.recordBackwardAdd(enc, view)`.
  - Run Adam once after all view gradients are accumulated.
- Keep `VisionTrainer` path behind a toggle until the page and benchmark prove wall time.

Expected speed:

- For the 3/9 default, not a full 2x optimizer-step improvement because raster backward and Adam remain. But CLIP at 3/9 is about 59 ms of 75.57 ms sampled total (`docs/SPLAT3D_PERF_NOTES.md:24`). Cutting CLIP roughly in half could take the 3/9 profile from about 70-75 ms to about 40-45 ms before copy/raster overhead.
- For 9/9 all-view optimization, the B=9 train bench implies the largest win.

Validation:

- Reuse `tools/clip/batch_major_train_bench.ts` parity first.
- Add a 3D optimizer parity check: single-image path versus batch path on the same three selected views should produce close raw splat gradients before Adam. Exact equality may not hold if ordering changes accumulation; cosine/high relative tolerance is fine.
- Use `docs/SPLAT3D_PERF_NOTES.md` format to update measured normal step and split profile.

### Phase 2: Selective shared-W pointwise integration

Priority: high after Phase 1 and profiling.

Rationale: Pointwise matmul is the static CLIP workhorse (`docs/SPLAT3D_PERF_NOTES.md:44`). Batch-major reduces dispatch multiplicity but still stages the same W tile for each z lane. The microkernel in `src/clip/vision_batch_pointwise.ts` stages W once across B lanes (`src/clip/vision_batch_pointwise.ts:121`) and uses `@workgroup_size(8, 8, batch)` (`src/clip/vision_batch_pointwise.ts:123`). That is exactly the AlphaGOJS "shared scratch only when it removes repeated traffic" principle.

But the current results are mixed:

- B=2 expansion layers improved modestly (`docs/CLIP_BATCHING_NOTES.md:205`).
- B=3 often lost, probably because 192 invocations and full 16 KB workgroup memory reduced occupancy (`docs/CLIP_BATCHING_NOTES.md:214`).

Concrete work:

- Do not replace all pointwise kernels.
- Extend profiling to identify top pointwise shapes by aggregate time in full B=3 and B=9 train runs.
- Create a production variant of `pointwiseSharedWBatchDispatch` that binds real batch-major slot buffers, not compact test buffers. The compact microbench assumes source/dest slots 0/1/2 (`src/clip/vision_batch_pointwise.ts:18`); production needs the same `BufferRef` mapping as `pointwise(s)` in `src/clip/vision_wgsl.ts:231`.
- Add a codegen selector in `batchTrainDispatches()`:
  - If `step.kind == "conv" && step.variant == "pointwise"`.
  - If batch is 2 or 3.
  - If profile allowlist says the shape wins.
  - Emit shared-W production kernel.
  - Else use existing z-batch transform.
- Maintain exact parity against z-batch for each replaced step.

Possible tile variants to try:

- Current shared-W B=3 uses 12 KB X + 4 KB W (`docs/CLIP_BATCHING_NOTES.md:180`). Try B=3 with smaller X tile:
  - `@workgroup_size(8,4,3)` with 16 couts per group instead of 32.
  - This lowers workgroup invocations and shared memory, possibly restoring occupancy while still sharing W.
- Try B=2 shared-W plus one normal z-batch lane for a 3-view step if B=3 shared-W loses.
- Try a shape-specific B=3 only for high-Cin expansion layers, not contraction/residual layers.

Decision gate: integrate only if a full `BatchMajorVisionTrainer` B=3 or B=9 run improves, not just isolated `pointwise_batch_bench.ts`.

### Phase 3: Fuse train-mode pointwise + GELU forward and GELU backward into neighboring pointwise backward

Priority: high if profiling shows GELU dispatches and activation traffic matter.

Rationale: Train mode deliberately splits GELU to save pre-activation for backward (`tools/clip/compile_plan.py:39`, `tools/clip/compile_plan.py:248`). That is correct, but it creates extra dispatches and memory traffic. AlphaGOJS's fusion lesson is to keep correctness but remove dispatch/memory surfaces when patterns are static.

Forward opportunity:

- Today a train conv with GELU becomes `conv act:"none"` then a standalone `gelu` step (`tools/clip/compile_plan.py:248`).
- The standalone `geluStep` reads pre and writes post (`src/clip/vision_wgsl.ts:581`).
- Instead, for pointwise convs that feed only a GELU, emit a pointwise kernel that writes both:
  - pre-activation to the saved pre slot.
  - post-GELU to the next activation slot.
- This keeps backward correctness because `gelu_bwd` still reads pre.
- It removes one dispatch and one full read of pre per GELU.

Backward opportunity:

- Today `pw_bwd` produces a grad tensor, then `gelu_bwd` multiplies by `geluGrad(pre)` and writes the next grad (`src/clip/vision_bwd_wgsl.ts:193`).
- For the common pattern `fc2_bwd -> gelu_bwd -> fc1_bwd`, fold the GELU derivative into the source load of the downstream `pw_bwd`.
- Add a new backward spec kind, e.g. `pw_bwd_after_gelu`, with bindings:
  - `dy` from the post-GELU grad slot.
  - `pre` saved pre-activation slot.
  - `dst` grad before the pointwise.
  - weights.
- In `pointwiseTiledMain`, stage `src` as `dy * geluGrad(pre)` instead of raw `dy`.
- Remove the separate `gelu_bwd` entry for that pattern.

Files:

- Pattern emission belongs in `tools/clip/compile_plan.py`, around split-GELU and backward emission (`tools/clip/compile_plan.py:580`).
- Forward codegen changes live in `src/clip/vision_wgsl.ts`, likely as a pointwise option next to `post` (`src/clip/vision_wgsl.ts:244`).
- Backward codegen changes live in `src/clip/vision_bwd_wgsl.ts`, reusing `pointwiseTiledMain` (`src/clip/vision_bwd_wgsl.ts:140`).

Expected speed:

- Potentially meaningful because the train plan has 129 forward steps and 152 backward entries (`tools/clip/README.md:117`), and GELU appears throughout ConvFFN blocks.
- It will not reduce pointwise MACs, but it reduces dispatch count and slot traffic. This is likely a single-digit to low-teens percent full CLIP win if GELU shows up in profiling.

Risk:

- Plan complexity. Must keep per-step verification useful. For forward, if a fused conv writes both pre and post, the verifier needs to read the post slot for the GELU step while still optionally verifying the pre slot.
- Backward order and `accumulate` flags must remain correct.

### Phase 4: Tile and vectorize `spatial_bwd`

Priority: medium-high if profiling confirms.

Rationale: The forward spatial conv has been tuned heavily: one workgroup per output channel, staged weights, 4 horizontal pixels per thread, and unrolled interior fast path (`src/clip/vision_wgsl.ts:270`). The backward spatial gather is simpler: one thread per input pixel, scalar loops over output channels and kernel taps, no shared staged weights (`src/clip/vision_bwd_wgsl.ts:224`). The README already calls gather `spatial_bwd` an obvious perf follow-up (`tools/clip/README.md:149`).

Concrete rewrite:

- For stride 1, emit a horizontal vec4 input-pixel tile similar to forward:
  - One thread computes four adjacent input pixels for the same input channel.
  - For interior pixels, unroll valid reversed taps with no per-tap bounds checks.
  - For borders, keep checked fallback.
- Stage per-group weights in workgroup memory:
  - For depthwise, `cpgOut=1`, k=3 or k=7.
  - For grouped expansion convs, `cpgOut` may be 2; stage the small `cpgOut * k*k` footprint.
- For stride 2, split parity cases at codegen. Avoid dynamic `continue` per tap where possible.
- Keep gather form. Do not introduce scatter or f32 atomics. AlphaGOJS explicitly avoided f32 atomic complexity and used gather phases where possible.

Likely important shapes:

- The stem `spatial_bwd` is the final `dL/dpixels`, so it touches 3 x 256 x 256 inputs and gathers from 64 output channels. Profile it first.
- Depthwise k=7 layers can be memory-heavy, and forward's comment says dynamic-index private arrays cost about 2x on Metal (`src/clip/vision_wgsl.ts:288`).

Validation:

- Existing `bwd_test.ts` has per-kernel `spatial_bwd` units (`tools/clip/bwd_test.ts:8`).
- Then full directional derivative.
- Then full CLIP B=1/B=3/B=9 timing.

Expected speed:

- Unknown until profiling. If `spatial_bwd` is a visible part of CLIP backward, this could be a clean exact-math win.
- If pointwise dominates overwhelmingly, defer.

### Phase 5: Reduce `attn_core_bwd` private memory and scalar loops

Priority: medium; only if profiling shows attention backward above 5-10%.

Rationale: AlphaGOJS got badly hurt by private arrays and register spilling (`/Users/nicholasbardy/git/AlphaGOJS/docs/perf/BACKWARD_ANALYSIS.md:347`). `attnCoreBwd` has the same warning signs. Each thread holds:

- `qi[hd]` and `dOi[hd]` (`src/clip/vision_bwd_wgsl.ts:493`).
- `p[nTok]` and `dP[nTok]` (`src/clip/vision_bwd_wgsl.ts:501`).
- `kj[hd]`, `vj[hd]`, `dV[hd]`, `dK[hd]` (`src/clip/vision_bwd_wgsl.ts:533`).

For hd=32 and nTok values up to 256, that can become a lot of private storage. Even at nTok=64, the per-thread footprint is large enough to risk spills on Metal.

Concrete options:

1. Vec4 chunking:
   - Store Q/K/V/dO as `array<vec4f, hd/4>` instead of scalar `array<f32, hd>`.
   - Reuse the forward `attnCore` style (`src/clip/vision_wgsl.ts:461`).
   - This is exact enough modulo reassociation and should reduce loop overhead.

2. Remove `dP[nTok]` private storage:
   - Phase 1 currently computes p and dP, then loops again for dQ (`src/clip/vision_bwd_wgsl.ts:513`).
   - Compute row max and denom first.
   - Compute `rdot` in a second pass without storing all `dP`.
   - Compute dQ in a third pass recomputing `dpj`.
   - This trades extra FLOPs for much lower private memory. AlphaGOJS's lesson: recomputation can be better if it avoids spills, but measure.

3. Store row probabilities in shared or global only if memory profile says recompute is worse:
   - For nTok=64, p matrix per head is 16 KB f32, too large for shared if combined with other scratch.
   - f16 p storage may fit but changes numerics.

4. Split `attn_core_bwd` into two dispatches:
   - Query phase writes dQ and row scalars.
   - Key/value phase writes dK/dV.
   - This may improve occupancy and lower private pressure, but adds dispatches and global traffic.

Decision gate: do not do this before per-dispatch profiling. Current docs already say attention backward should be attacked only if timestamps show it above about 5-10% (`docs/SPLAT3D_PERF_NOTES.md:51`).

### Phase 6: Multi-workgroup head and SE GEMV kernels

Priority: medium-low unless profiling flags them.

Rationale: `headStep` and `headBwd` each do large-ish matrix-vector work in one workgroup. One workgroup cannot occupy the GPU by itself. AlphaGOJS's serial-patch problem was not the arithmetic; it was under-occupied structure. The same could apply to head and SE.

Current head:

- GAP writes `gap[cin]` in one workgroup (`src/clip/vision_wgsl.ts:546`).
- 512 output channels each scan 1024 gap values (`src/clip/vision_wgsl.ts:555`).
- Workgroups: `[1, 1, 1]` (`src/clip/vision_wgsl.ts:566`).

Current head backward:

- 1024 input channels each scan 512 embedding grad values (`src/clip/vision_bwd_wgsl.ts:388`).
- Workgroups: `[1, 1, 1]` (`src/clip/vision_bwd_wgsl.ts:404`).

Concrete rewrite:

- Split head into two dispatches:
  - `head_gap`: one or more workgroups reduce spatial P to a 1024-float global gap buffer.
  - `head_proj`: many workgroups over output channel tiles, reading global gap and W.
- For backward:
  - `head_dgap`: many workgroups over input-channel tiles.
  - `head_broadcast`: simple elementwise broadcast to spatial gradient.
- Use a temporary slot or scratch slot emitted by `compile_plan.py`.

Tradeoff:

- Adds dispatches and global intermediate traffic.
- Gains occupancy. Only worth it if current head/head_bwd are visible in profiling.

SE has similar structure but smaller matrices. It already has careful workgroup memory reuse to fit 16 KB (`tools/clip/README.md:103`, `src/clip/vision_bwd_wgsl.ts:298`). I would not touch SE before head unless profiling points there.

### Phase 7: f16 storage and mixed precision

Priority: medium, after exact structural work.

Rationale: CLIP's frozen weights and static validation make f16 more plausible than in AlphaGOJS PPO. Pointwise kernels are likely bandwidth-sensitive, and train batch-major multiplies activation traffic by B. But f16 has integration risk, model-quality risk, and browser feature gating.

Concrete variants:

1. f16 weights only:
   - Pack a second weights blob as f16 in `tools/clip/compile_plan.py`.
   - Use `enable f16;`.
   - Bind weights as `array<vec4<f16>>`, convert to f32 for accumulation.
   - Keep activations f32.
   - This halves weight bandwidth and should preserve output reasonably.

2. f16 activations for saved train slots:
   - More aggressive. Would halve batch-major slot memory and bandwidth.
   - Compute pointwise accumulations in f32, store selected activations as f16.
   - Keep input/output image gradients f32.
   - Need per-step tolerances and end-to-end cosine quality gates.

3. f16 workgroup tiles:
   - Keep storage f32, convert xS/wS to f16 in shared memory, accumulate f32 or f16.
   - This may not save global bandwidth, only shared/register pressure; likely less useful.

Feature gating:

- Current CLIP device requests do not require `shader-f16` (`tools/clip/batch_major_train_bench.ts:104`, `src/clip/vision.ts` runtime takes an existing device). A production f16 path needs device creation to request the feature, or it must be optional when the surrounding app creates the device.
- The app already has f16 experience elsewhere, but CLIP's page/device creation path must explicitly pass required features if f16 shaders are compiled.

Validation:

- Use `PLAN=plan_train.json bun tools/clip/fused_test.ts`, but with looser per-step tolerances.
- Use `bun tools/clip/bwd_test.ts` directional derivative.
- Use semantic page gates, not just numerical parity. f16 can pass rough numerics but degrade optimization.

Expected speed:

- f16 weights could be a real win for pointwise-heavy CLIP, especially B=3/B=9. But it should be tested after full-profile data identifies memory bandwidth as the limiter.

### Phase 8: Subgroups for reductions

Priority: low-medium.

Rationale: AlphaGOJS initially ranked subgroups high for barrier-heavy reductions (`/Users/nicholasbardy/git/AlphaGOJS/docs/perf/ALGO_APPROACHES.md:213`). CLIP has reductions, but fewer of them are obviously dominant. `loss_bwd` does a 256-thread tree reduction over three sums (`src/clip/vision_bwd_wgsl.ts:415`). Pointwise matmul barriers are per tile and probably necessary for shared memory.

Concrete use:

- Add optional subgroup variants for:
  - `loss_bwd` reduction.
  - any future profiling kernels that show reduction cost.
  - maybe `head` GAP reductions if rewritten.
- Use runtime feature detection on the adapter and compile subgroup variants only when available.

Expected speed:

- Small unless profiling finds reduction-heavy kernels. Do not prioritize ahead of batch-major integration or spatial/pointwise work.

### Phase 9: Memory and copy cleanup around CLIP integration

Priority: opportunistic after Phase 1.

Rationale: AlphaGOJS found fusing readbacks did not move throughput, but kept it for correctness and fewer syncs (`/Users/nicholasbardy/git/AlphaGOJS/BENCHMARKS.md:122`). Similar here: raster/CLIP copies are not the primary bottleneck, but multi-view copies are easy to eliminate if buffer ownership is cleaned up.

Concrete ideas:

- Alias raster image buffers with CLIP input slots where layouts match. The 2D optimizer already documents the copies as byte-for-byte 768 KB NCHW blits (`src/splat/optimize.ts:11`). The 3D optimizer does the same copy per view (`src/splat3d/optimize.ts:139`).
- Alias CLIP input-grad slots with raster grad image buffers if lifetimes allow.
- In batch-major 3D, render each selected view directly into the correct lane offset instead of rendering to `raster.image` then copying.
- This probably requires Raster3DEngine forward output offsets or per-lane image buffers.

Expected speed:

- Small compared with CLIP matmul, but more valuable at B=9 because copies multiply by views.
- Also reduces command list clutter and memory bandwidth.

## What not to do

Do not start with atomics. CLIP's backward is dL/dpixels only, with no dW and no optimizer. The AlphaGOJS atomic-gradient idea was explicitly low priority and numerically risky because WGSL has no native f32 atomics (`/Users/nicholasbardy/git/AlphaGOJS/docs/perf/ALGO_APPROACHES.md:144`). CLIP can stay gather-based.

Do not replace every pointwise kernel with shared-W just because it is elegant. The microbench already shows B=3 can lose (`docs/CLIP_BATCHING_NOTES.md:203`). Select by measured shape.

Do not discard the current train plan's saved-activation design without a memory reason. AlphaGOJS debated store-versus-recompute, but CLIP's backward is frozen-weight and already stores train activations by design (`tools/clip/compile_plan.py:122`). Recompute can help attention private memory, but wholesale checkpointing would risk extra CLIP compute for little gain.

Do not make f16 the default without feature-gated exact fallbacks. The current f32 path is a known-good oracle with tight tests.

## Recommended concrete roadmap

1. Add a CLIP dispatch profiler.
   - Output: table of ms by `DispatchSpec.label`, grouped by shape and kind, for B=1, B=3, B=9 forward and forward+backward.
   - Based on existing bench patterns in `tools/clip/fused_test.ts:147` and `tools/clip/batch_major_train_bench.ts:115`.

2. Wire `BatchMajorVisionTrainer` into `Splat3DOptimizer`.
   - Main target file: `src/splat3d/optimize.ts`.
   - Use batch offsets from `src/clip/vision_batch.ts:493`.
   - Re-measure `docs/SPLAT3D_PERF_NOTES.md` table.

3. Profile full batch-major CLIP after integration.
   - If pointwise is still dominant, proceed to step 4.
   - If raster becomes dominant, pause CLIP kernel work and fix raster buffer aliasing/workgroup staging.

4. Selectively integrate shared-W pointwise kernels.
   - Main target files: `src/clip/vision_batch_pointwise.ts`, `src/clip/vision_batch_wgsl.ts`.
   - Use allowlisted shapes that win in full CLIP, not only isolated microbench.

5. Fuse train-mode pointwise/GELU forward and GELU-backward into adjacent pointwise backward.
   - Main target files: `tools/clip/compile_plan.py`, `src/clip/vision_wgsl.ts`, `src/clip/vision_bwd_wgsl.ts`.
   - Keep per-step forward refs meaningful.

6. Rewrite `spatial_bwd` with staged weights and vec4 horizontal pixels.
   - Main target file: `src/clip/vision_bwd_wgsl.ts`.
   - Validate with `bwd_test.ts`, then full directional derivative.

7. If attention backward profiles high, reduce private arrays and vectorize `attn_core_bwd`.
   - Main target file: `src/clip/vision_bwd_wgsl.ts`.
   - First try exact vec4 chunking and dP storage removal before adding dispatches.

8. Try f16 weights as an optional variant.
   - Main target files: `tools/clip/compile_plan.py`, `src/clip/vision_wgsl.ts`, `src/clip/vision_bwd_wgsl.ts`, device creation sites.
   - Gate by feature and semantic optimization quality.

9. Add subgroup reductions only where the profiler shows reduction time.
   - Main target files: `src/clip/vision_bwd_wgsl.ts` and any future profiler/timing harness.

10. Clean up raster/CLIP image and gradient copies.
    - Main target files: `src/splat3d/optimize.ts`, raster engine APIs.
    - Treat as a secondary bandwidth cleanup after CLIP batch integration.

## Expected payoff stack

This is intentionally ambitious, but the stack should be staged so every layer is measurable:

- Batch-major integration into 3D optimizer: likely the biggest immediate wall-time win, already measured in isolation.
- Selective shared-W pointwise: maybe 5-15% of CLIP if the top shapes match winning microbench patterns; could be zero or negative if blanket-applied.
- GELU/pointwise fusion: likely single-digit to low-teens percent of train CLIP if dispatch and slot traffic are visible.
- Tiled `spatial_bwd`: unknown, but likely the cleanest exact backward kernel rewrite after pointwise.
- Attention private-memory reduction: potentially meaningful only if attention backward is a measured chunk.
- f16 weights/activations: potentially meaningful for bandwidth and memory, but should be last because it changes numerical behavior and device requirements.

The AlphaGOJS meta-lesson is the important one: keep a baseline, fork variants, verify tiny deterministic cases first, measure full warmed wall time second, and record the result even when a plausible GPU idea loses.
