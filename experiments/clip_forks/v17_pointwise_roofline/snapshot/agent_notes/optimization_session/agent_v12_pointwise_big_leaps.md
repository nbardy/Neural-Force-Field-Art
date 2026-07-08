# Agent v12 - Pointwise And pw_bwd Big-Leap Notes

Date: 2026-07-08

Scope: documentation-only inspection of the CLIP pointwise and `pw_bwd` WGSL
emitters, profiling notes, and benchmark harnesses. No runtime source files
were changed.

Files inspected:

- `src/clip/vision_wgsl.ts`
- `src/clip/vision_bwd_wgsl.ts`
- `src/clip/vision_batch_wgsl.ts`
- `src/clip/vision_batch_pointwise.ts`
- `src/clip/vision_batch.ts`
- `tools/clip/dispatch_profile.ts`
- `tools/clip/pointwise_batch_bench.ts`
- `tools/clip/pointwise_batch_matrix.ts`
- `tools/clip/batch_major_train_bench.ts`
- `tools/clip/bwd_test.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `docs/CLIP_BATCHING_NOTES.md`
- `docs/SPLAT3D_PERF_NOTES.md`
- `experiments/clip_forks/v03_residual_bwd_pw_fusion/README.md`
- `experiments/clip_forks/v04_pointwise_tile_rewrite/README.md`
- `experiments/clip_forks/v10_shared_w_pointwise_forward/README.md`
- `experiments/clip_forks/v11_backward_local_fusions/README.md`
- `agent_notes/optimization_session/agent_pointwise_bottleneck.md`
- `agent_notes/optimization_session/agent_pointwise_math_layout_fusion_2026_07_08.md`
- `agent_notes/optimization_session/clip_timestamp_dispatch_profile.md`
- `agent_notes/optimization_session/clip_2x_4x_trace_status.md`
- `agent_notes/optimization_session/current_strategy_reflection.md`

## Short Version

In this CLIP image tower, "pointwise" means a `1x1`, `groups=1` convolution.
Mathematically it is a dense matrix multiply over channels at every
spatial/token position:

```text
P = H * W
Y[co, p] = bias[co] + sum_ci X[ci, p] * W[ci, co]
```

`pw_bwd` is the frozen-weight input-gradient version:

```text
dX[ci, p] = sum_co dY[co, p] * W[ci, co]
```

The current kernel is already a tiled WGSL matmul. It is not naive. It is still
hot because the train plan has 48 forward pointwise dispatches and 48 backward
`pw_bwd` dispatches. The checked-in `plan_train.json` has:

```text
forward steps:             129
backward steps:            152
forward pointwise steps:    48
backward pw_bwd steps:      48
pointwise->GELU pairs:      24
forward pointwise MACs:     2.214592512B per image
backward pw_bwd MACs:       2.214592512B per image
```

Recent timestamp notes consistently put pointwise forward/backward at roughly
half of isolated B=3 CLIP train GPU time. One recorded promoted-settings run:

```text
pw_bwd   42.01 ms  26.2%
pw+gelu  23.92 ms  14.9%
pw       23.20 ms  14.5%
```

Another fresher note had the same ordering with lower absolute times:

```text
pw_bwd   25.82 ms  23.8%
pw       14.09 ms  13.0%
pw+gelu  13.96 ms  12.9%
```

So pointwise is a real target, but prior notes also show that small shared-W
forward tweaks are not enough. The plausible large wins require changing tile
shape, reduction scheduling, or storage/layout assumptions, not just moving
batch lanes into `local_invocation_id.z`.

## Current Math And Layout

The source contract in `src/clip/vision_wgsl.ts` says activations are
channel-planar NCHW f32:

```text
activation scalar X[c, y, x] = slot[c * H * W + y * W + x]
```

Pointwise kernels bind activation slots as `array<vec4f>`, so four adjacent
spatial positions are packed into one vector:

```text
P4 = P / 4
src[ci * P4 + p4] = vec4f(
  X[ci, 4*p4 + 0],
  X[ci, 4*p4 + 1],
  X[ci, 4*p4 + 2],
  X[ci, 4*p4 + 3]
)
dst[co * P4 + p4] = vec4f(Y[co, 4*p4 + 0..3])
```

Forward pointwise weights are packed as transposed `[Cin][Cout]`, with four
adjacent output channels contiguous:

```text
scalar W index = wOff + ci * Cout + co
W4((wOff + ci * Cout + co4) / 4) -> W[ci, co4..co4+3]
```

`pw_bwd` uses `wOffT`, a second compiler-emitted orientation, so it can reuse
the same tiled body as forward:

```text
PwBwdStep.cin  = reduction channels = forward Cout = dY channels
PwBwdStep.cout = output channels    = forward Cin  = dX channels
```

Batch-major CLIP allocates every slot as:

```text
[batch][slotFloats]
```

`src/clip/vision_batch_wgsl.ts` rewrites slot accesses by adding:

```text
batchBase = workgroup_id.z * (slotFloats / 4)   // for vec4f slots
```

The default batch path therefore still runs one pointwise workgroup per image
lane:

```text
workgroups = [P4 / 8, Cout / 32, batch]
```

The shared-W experiment in `src/clip/vision_batch_pointwise.ts` moved batch
lanes into `local_invocation_id.z`:

```text
workgroup_size = (8, 8, batch)
workgroups     = [P4 / 8, Cout / 32, 1]
```

That stages one W tile for all lanes, but B=3 uses exactly the local memory
budget seen in the notes:

```text
xS: 256 * 3 vec4f = 12 KB
wS: 256 vec4f     =  4 KB
total             = 16 KB
```

The production shared-W forward fork verified but did not promote. The v10 note
records roughly noise-level CLIP-only gain and no integrated 3D improvement.
That is important: "share W across B lanes" is a tested small idea, not the next
big leap.

## Current Pointwise Tile

The core emitter is `pointwiseTiledMain()` in `src/clip/vision_wgsl.ts`.
Forward `pointwise()` and backward `pwBwd()` both use it.

Current geometry:

```text
workgroup_size = 8 x 8 = 64 threads
workgroups     = [P4 / 8, Cout / 32, 1] before batch lifting
tile owns      = 8 pixel-quads x 32 output channels
               = 32 scalar pixels x 32 output channels
thread owns    = 1 pixel-quad x 4 output channels
               = 16 scalar outputs
```

Shared memory:

```text
xS: 256 vec4f = 4 KB    // 32 ci x 8 pixel-quads
wS: 256 vec4f = 4 KB    // 32 ci x 8 cout-quads
total         = 8 KB
```

Per thread:

```text
p4 = wid.x * 8 + lid.x
co = (wid.y * 8 + lid.y) * 4
acc0..acc3 are vec4f, one vec4 per output channel co+0..co+3
```

Reduction loop:

```text
for ci0 in 0..Cin step 32:
  stage xS[32 ci][8 p4]
  stage wS[32 ci][8 cout-quads]
  barrier
  for ci in 0..31:
    xv = xS[ci, lid.x]      // four pixels
    wv = wS[ci, lid.y]      // four output channels
    acc0 += wv.x * xv
    acc1 += wv.y * xv
    acc2 += wv.z * xv
    acc3 += wv.w * xv
  barrier
```

The tile assumptions are asserted by `assertPointwiseTiles()`:

```text
P % 32 == 0
Cin % 32 == 0
Cout % 32 == 0
wOff % 4 == 0
```

## Why It Is A Bottleneck

The repeated MobileCLIP ConvFFN pattern is:

```text
pointwise expansion -> GELU -> pointwise contraction -> residual/layer-scale
```

The repeated pointwise shapes dominate static MACs:

```text
10 x 256->768 @16x16    503.3M forward MACs
10 x 768->256 @16x16    503.3M
 6 x 128->384 @32x32    302.0M
 6 x 384->128 @32x32    302.0M
 4 x 512->1536 @8x8     201.3M
 2 x 64->192 @64x64     100.7M
 2 x 192->64 @64x64     100.7M
```

`pw_bwd` repeats essentially the same amount of matrix work in reverse.

The current tile has good local reuse but still pays these costs:

- For high-P early layers, the same W tile is reloaded across many spatial
  x-tiles. Example: `64x64` has `P4=1024`, so `P4/8=128` x-tiles.
- For high-Cout expansion or backward-output-heavy layers, the same X tile is
  reloaded across many output-channel y-tiles.
- Each 32-channel reduction chunk has two workgroup barriers.
- Late low-P layers such as `8x8` may have too few workgroups but very long
  per-workgroup reduction loops.
- Batch-major z dispatch reduces dispatch-list multiplicity, but by default it
  does not reduce per-lane W or X traffic.
- Train mode must keep saved pre-activation slots for backward. The promoted
  `pw+gelu` forward fusion removes standalone GELU dispatches but still writes
  both pre and post slots.
- `pw_bwd` can be fused locally with GELU or residual copies, but the v11 note
  shows these are real small wins, not a standalone 2x.

## Big-Leap Rewrite 1 - Shape-Gated Rectangular Tiles

Goal: replace the one-size `8 p4 x 32 cout` tile with several explicitly gated
WGSL tile families for hot shapes. This is the cleanest direct pointwise rewrite.

Do not mutate `pointwiseTiledMain()` globally. Add new emitters and select them
by shape/step allowlist.

Candidate tile families:

```text
dual_pixel:
  tile: 16 p4 x 32 cout
  workgroup memory: xS 8 KB + wS 4 KB = 12 KB
  intent: reuse W over twice as many pixels
  likely target: 64x64 and 32x32 high-P layers

dual_cout:
  tile: 8 p4 x 64 cout
  workgroup memory: xS 4 KB + wS 8 KB = 12 KB
  intent: reuse X over twice as many output channels
  risk: 8 vec4 accumulators per thread can spill on Metal
  likely target: 16x16 and 8x8 channel-heavy layers

rect_8x16:
  workgroup_size: 8 x 16 = 128 threads
  tile: 8 p4 x 64 cout
  workgroup memory: xS 4 KB + wS 8 KB = 12 KB
  intent: get 64-cout reuse with less per-thread accumulator growth
  risk: 128-thread groups may reduce occupancy or scheduling quality
```

Why this could produce a large win:

- `dual_pixel` attacks repeated W loads in early high-resolution layers.
- `dual_cout` attacks repeated X loads in expansion/contraction and `pw_bwd`
  layers with many output-channel tiles.
- Current pointwise is about half of isolated CLIP train time, so even a 20-30%
  pointwise-family reduction can move integrated wall time.

Why it could lose:

- Register spills from more accumulators.
- Lower occupancy from higher workgroup memory or 128-thread groups.
- The best tile differs by shape; a blanket replacement will likely regress.
- Shared-W history shows isolated wins can disappear in full-chain timing.

Concrete source targets:

- `src/clip/vision_wgsl.ts`
  - Add `pointwiseTiledMainDualPixel()`, `pointwiseTiledMainDualCout()`, and/or
    `pointwiseTiledMainRect8x16()`.
  - Add selector logic near `pointwise()`.
  - Ensure `pointwiseFusedGelu()` can use the same selected tile because hot
    train forward rows are often `pw+gelu`.
- `src/clip/vision_bwd_wgsl.ts`
  - Add matching variant selection inside `pwBwd()`, `pwBwdAfterGelu()`, and
    `pwBwdWithResidualCopy()`.
- `src/clip/vision_batch_wgsl.ts`
  - Pass variant options through `BatchDispatchOptions`; preserve z-batch
    lifting.
- `tools/clip/dispatch_profile.ts`
  - Add env flags such as `PW_TILE_VARIANT`, `PW_TILE_STEPS`,
    `PW_BWD_TILE_STEPS`.
- New or extended bench:
  - `tools/clip/pointwise_tile_bench.ts` for deterministic per-shape parity
    against the current emitter.

Initial target shapes:

```text
dual_pixel forward:
  64->192 @64x64
  192->64 @64x64
  128->384 @32x32
  384->128 @32x32

dual_cout or rect_8x16 forward/backward:
  256->768 @16x16
  768->256 @16x16
  512->1536 @8x8
  1536->512 @8x8
```

Benchmark commands:

```bash
# Same-session baseline ranking.
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_pw_base.csv

# Expected variant commands after adding gates.
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 PW_TILE_VARIANT=dual_pixel PW_TILE_STEPS=8,10,13,15,22,24,27,29 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_pw_dual_pixel.csv

CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 PW_TILE_VARIANT=dual_cout PW_TILE_STEPS=57,59,62,64 PW_BWD_TILE_STEPS=6,8,15,17,30,32 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_pw_dual_cout.csv

# Correctness and full-chain gates.
PW_TILE_VARIANT=dual_cout PW_TILE_STEPS=57,59,62,64 PW_BWD_TILE_STEPS=6,8,15,17,30,32 bun tools/clip/bwd_test.ts

PW_TILE_VARIANT=dual_cout PW_TILE_STEPS=57,59,62,64 PW_BWD_TILE_STEPS=6,8,15,17,30,32 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/batch_major_train_bench.ts

TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 SPATIAL_BWD_VARIANT=depthwise4 PW_TILE_VARIANT=dual_cout PW_TILE_STEPS=57,59,62,64 PW_BWD_TILE_STEPS=6,8,15,17,30,32 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Promotion threshold I would use:

```text
selected labels:        >= 1.20x median speedup
all pointwise groups:   >= 10% timestamp-sum reduction
CLIP train bench:       >= 8% batch-major median reduction
integrated step normal: >= 5% same-session median reduction
```

## Big-Leap Rewrite 2 - Split-K pw_bwd For Low-P, High-Channel Layers

Goal: attack the long serial reduction loop in late `8x8` and some `16x16`
pointwise backward shapes by parallelizing the K/reduction dimension across
multiple workgroups, then reducing partial sums in a second dispatch.

Current `pw_bwd` assigns one workgroup to one `(p4 tile, output-channel tile)`
and loops through all reduction channels:

```text
for ci0 in 0..pw.cin step 32:
  stage dY and W_T
  compute
```

For late shapes, `P4/8` is tiny. Example: `8x8` has `P=64`, `P4=16`, so only
two spatial x-tiles. A shape like `1536->512 @8x8` has relatively few
workgroups, but each workgroup runs 48 reduction chunks and 96 barriers. That
can under-occupy the GPU and serialize a lot of work inside each group.

Split-K design:

```text
dispatch A:
  grid = [P4 / 8, Cout / 32, batch * kSplits]
  each kSplit computes a partial sum over a contiguous reduction slice
  writes partial[kSplit][batch][cout][p4] as vec4f

dispatch B:
  grid = [P4 / 8, Cout / 32, batch]
  reads kSplits partial vec4s
  sums them
  applies normal store or accumulate semantics
```

No f32 atomics are needed. This is important. It adds one partial buffer and one
reduce dispatch, but it may win when the baseline has too few long-running
workgroups.

Why this could produce a large win:

- It increases workgroup count for low-P late layers.
- It cuts per-workgroup barrier count and long inner-loop lifetime.
- It can be applied only to `pw_bwd`, the hottest group in multiple timestamp
  notes.
- It is most plausible where partial memory is small: `8x8` and maybe `16x16`.

Why it could lose:

- Partial sums add global write/read traffic.
- FMA order changes, so parity must allow normal fp32 drift but pass gradient
  directional gates.
- It complicates the dispatcher because one logical backward step becomes two
  dispatches plus scratch.
- High-P layers already have enough workgroups and likely should not use it.

Concrete source targets:

- `src/clip/vision_wgsl.ts`
  - Add a reusable split-K partial body or parameterize a new pointwise tile
    body with `kStart`, `kCount`, and "write partial instead of final store".
- `src/clip/vision_bwd_wgsl.ts`
  - Add `pwBwdSplitK()` and a `pw_bwd_reduce` emitter.
  - Add selection in `planBwdDispatches()`.
  - Keep normal `pwBwdAfterGelu()` and `pwBwdWithResidualCopy()` as fallback
    unless split-K versions explicitly support those fused cases.
- `src/clip/vision.ts` and `src/clip/vision_batch.ts`
  - Add scratch-buffer allocation and binding. The current `BufferRef` only has
    `weights`, `slot`, and `text`, so this probably needs a new
    `{ kind: "scratch"; id: string }` binding type.
- `src/clip/vision_batch_wgsl.ts`
  - Make sure batch offsets are not string-rewritten into scratch arrays unless
    the scratch layout intentionally includes batch.
- `tools/clip/bwd_test.ts`
  - Add targeted tests for split-K `pw_bwd` with `accumulate=false` and
    `accumulate=true`.

Initial targets:

```text
pw_bwd 1536->512 @8x8
pw_bwd 512->1536 @8x8
pw_bwd 768->256 @16x16
pw_bwd 256->768 @16x16
```

First split counts:

```text
8x8:
  kSplits = 4 or 6
  k block = 256 or 384 channels

16x16:
  kSplits = 2 or 3 only if timestamps show under-occupancy
```

Benchmark commands:

```bash
# Identify exact hot backward labels before picking split shapes.
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_bwd_base.csv

# Expected variant commands after adding gates.
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 PW_BWD_SPLITK_STEPS=late8 PW_BWD_SPLITK=4 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_bwd_splitk4.csv

PW_BWD_SPLITK_STEPS=late8 PW_BWD_SPLITK=4 bun tools/clip/bwd_test.ts

PW_BWD_SPLITK_STEPS=late8 PW_BWD_SPLITK=4 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/batch_major_train_bench.ts

TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 SPATIAL_BWD_VARIANT=depthwise4 PW_BWD_SPLITK_STEPS=late8 PW_BWD_SPLITK=4 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Promotion threshold:

```text
targeted pw_bwd labels: >= 1.30x after including reduce dispatch
all pw_bwd group:       >= 12% timestamp-sum reduction
full CLIP train:        >= 8% median reduction
inputGrad:              bwd_test gate passes, directional derivative passes
```

This is higher risk than rectangular tiles because it touches dispatch
semantics and scratch allocation, but it directly targets the hottest
backward-specific failure mode: too few long-running late-layer workgroups.

## Big-Leap Rewrite 3 - Selective f16 Hidden Slots For Pointwise Sources

Goal: halve selected intermediate activation traffic without moving the whole
CLIP model to f16. Prior all-weight f16 failed the strict gradient/speed gate.
This is a narrower storage-layout shader rewrite: keep weights and accumulators
f32 by default, but store the GELU output hidden tensor of expansion pointwise
layers as f16 for the following contraction pointwise source.

The common forward pair is:

```text
pw1 expansion -> GELU -> pw2 contraction
```

Train mode currently needs the pre-GELU value for backward, so promoted
`pw+gelu` writes two f32 slots:

```text
pre slot:      f32, needed by geluGrad(pre)
geluDst slot:  f32, used as source by pw2 contraction
```

For frozen-weight CLIP backward, the GELU output slot is not needed to compute
`dL/dpixels`; `gelu_bwd` needs the pre slot and upstream gradient, not the
forward GELU output. That makes the GELU output a plausible f16 storage
candidate while preserving f32 pre-activation for gradient math:

```text
pw1+gelu:
  write pre as f32
  write gelu(pre) as vec4<f16>

pw2:
  read source as vec4<f16>
  convert to vec4f in the pointwise load path
  accumulate f32
```

Why this could produce a large win:

- It targets the exact hidden tensors that feed pointwise contraction layers.
- It halves source activation bandwidth for those contraction pointwise passes.
- It avoids perturbing loss, text embeddings, attention reductions, gradient
  accumulation slots, or saved pre-GELU values.
- It is more targeted than the rejected all-weight f16 fork.

Why it could lose:

- Forward CLIP embedding may be sensitive to f16 hidden activations.
- Source conversion can add ALU and type-conversion overhead.
- Slot allocation, batch offsets, and string rewriting currently assume f32
  slot sizes and `array<f32>` or `array<vec4f>`.
- It will not help layers where W traffic or compute dominates over activation
  source traffic.

Concrete source targets:

- `tools/clip/compile_plan.py`
  - Mark selected GELU output slots as f16-capable, probably only exact
    pointwise-expansion GELU outputs feeding one pointwise contraction.
- `src/clip/vision_wgsl.ts`
  - Extend `DispatchSpec` or slot metadata to know storage element type.
  - Add f16 source binding/load support to pointwise emitters.
  - Change `pointwiseFusedGelu()` for selected steps to write f32 pre plus f16
    GELU output.
- `src/clip/vision_bwd_wgsl.ts`
  - Keep `pw_bwd` source/destination gradients f32 initially.
  - Ensure fused `pw_bwd+gelu` still reads f32 pre and f32 dY.
- `src/clip/vision_batch_wgsl.ts`
  - Extend `batchBindings()` and `addBatchOffsets()` beyond `array<f32>` and
    `array<vec4f>` so f16 vector slots get the correct byte/element stride.
- `src/clip/vision.ts` and `src/clip/vision_batch.ts`
  - Allocate selected slots at 2 bytes per scalar, not 4.
  - Preserve public input/output/inputGrad buffers as f32.
- `tools/clip/batch_major_train_bench.ts` and `tools/clip/dispatch_profile.ts`
  - Add gates such as `POINTWISE_HIDDEN_F16=1`.

Initial target pairs:

```text
64->192 @64x64 + GELU, feeding 192->64 @64x64
128->384 @32x32 + GELU, feeding 384->128 @32x32
256->768 @16x16 + GELU, feeding 768->256 @16x16
512->1536 @8x8 + GELU, feeding 1536->512 @8x8
```

Benchmark commands:

```bash
# Baseline correctness and timing.
STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 BATCH=3 RUNS=3 WARMUP=2 bun tools/clip/batch_major_train_bench.ts

CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_f32_hidden_base.csv

# Expected variant commands after adding gates.
POINTWISE_HIDDEN_F16=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 BATCH=3 RUNS=3 WARMUP=2 bun tools/clip/batch_major_train_bench.ts

POINTWISE_HIDDEN_F16=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 bun tools/clip/bwd_test.ts

CSV=1 TIMESTAMP=1 POINTWISE_HIDDEN_F16=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_f16_hidden_variant.csv

TIMESTAMP=1 POINTWISE_HIDDEN_F16=1 CLIP_BATCH=3 VIEWS=3 SPATIAL_BWD_VARIANT=depthwise4 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Quality gates:

```text
embedding cosine versus f32:      >= 0.9999
inputGrad cosine versus f32:      >= 0.995
inputGrad relative Linf:          no worse than current f16-weight rejected fork
bwd_test directional derivative:  pass
integrated step:                  same-session win, not just dispatch-profile win
```

This is not "turn CLIP f16 on". It is a targeted pointwise source-layout fork.
It should be abandoned quickly if gradient cosine drops like the all-weight f16
experiment.

## Bench Order For Any Rewrite

Use the same order for all three ideas:

```bash
# 1. Baseline dispatch ranking.
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/nffa_base.csv

# 2. Correctness gate.
bun tools/clip/bwd_test.ts

# 3. Batch-major CLIP wall-time and gradient parity.
STEM_SPATIAL_BWD=1 SPATIAL_BWD_VARIANT=depthwise4 FUSE_PW_GELU=1 BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/batch_major_train_bench.ts

# 4. Integrated optimizer gate.
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 SPATIAL_BWD_VARIANT=depthwise4 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts

# 5. Repeated integrated matrix once a variant survives.
TRIALS=5 CONFIGS=base=3:3:dw4,candidate=3:3:dw4 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

For pointwise work, stale timing is especially misleading. Compare baseline and
variant in the same session, with the same adapter, same warmup, and same
thermal state.

## Practical Recommendation

The next serious pointwise fork should start with rectangular tile variants,
because that keeps the current buffer model and correctness surface mostly
intact. If that fails to move `pw_bwd`, try split-K specifically on late
low-resolution backward layers. Selective f16 hidden slots are the highest
layout upside, but they touch plan metadata and slot allocation, so they should
come after one pure-WGSL tile fork proves there is real headroom.

Do not promote another shared-W forward allowlist without a full-chain win. That
experiment already passed parity and still did not improve integrated 3D timing.
