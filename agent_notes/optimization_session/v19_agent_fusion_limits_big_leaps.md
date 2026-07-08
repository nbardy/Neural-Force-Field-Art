# v19 Agent Note: CLIP Fusion Limits And Big-Leap Shader Forks

Date: 2026-07-08

Scope: read-only source inspection plus one static `plan_train.json` count.
No runtime source code was edited for this note.

## Short Answer

Pointwise+GELU helps because it is a local producer/consumer fusion:

```text
pointwise computes pre = X @ W + b
train backward still needs pre
GELU consumes pre immediately
```

The fused shader keeps the train contract by writing both slots:

```text
pre slot       = pre
post-GELU slot = gelu(pre)
```

That removes a standalone GELU dispatch and the extra read of the pre tensor
without changing the saved activation set.

That logic does not extend to "fuse all of CLIP into one giant shader." CLIP
layers need global ordering between whole tensors. WebGPU only has
workgroup-local barriers inside one compute dispatch; dispatch boundaries are
the global synchronization. Pointwise, spatial conv, SE, attention, head, loss,
and backward kernels also use different workgroup geometries and reduction
patterns. Train mode must preserve saved activations for `dL/dimage`, and
backward has first-writer/accumulate gradient semantics. A giant shader would
either be incorrect, recompute large tensors many times, or turn into a huge
branchy kernel with bad occupancy, high register pressure, and poor testability.

The right strategy is not one CLIP shader. It is a small number of exact,
shape-gated local fusions and a few ambitious pointwise/backward shader forks.

## Files Inspected

- `src/clip/vision_wgsl.ts`
  - `weightsDecl()`
  - `pointwiseTiledMain()`
  - `pointwiseRect8x16Main()`
  - `pointwise()`
  - `pointwiseRect8x16()`
  - `pointwiseFusedGelu()`
  - `pointwiseRect8x16FusedGelu()`
  - `spatialConv()`
  - `seStep()`
  - `attnCore()`
  - `headStep()`
  - `geluStep()`
  - `stepDispatches()`
  - `planDispatches()`
- `src/clip/vision_bwd_wgsl.ts`
  - `pwBwd()`
  - `pwBwdAfterGelu()`
  - `pwBwdWithResidualCopy()`
  - `geluBwd()`
  - `spatialBwd()`
  - `spatialBwdDepthwise4()`
  - `spatialBwdStem4()`
  - `seBwd()`
  - `attnCoreBwd()`
  - `planBwdDispatches()`
- `src/clip/vision_batch_wgsl.ts`
  - `addBatchOffsets()`
  - `forwardDispatches()`
  - `batchForwardDispatches()`
  - `batchTrainDispatches()`
- `src/clip/vision_batch_pointwise.ts`
  - `pointwiseSharedWBatchForwardDispatch()`
  - the isolated z-batch/shared-W pointwise emitters
- `src/clip/vision.ts`
  - `VisionEncoder`
  - `VisionTrainer`
  - `BufferRef` resolution
- `src/clip/vision_batch.ts`
  - `BatchMajorVisionTrainer`
  - batch-major buffer layout and dispatch encoding
- `src/splat3d/optimize.ts`
  - CLIP dispatch option wiring into `VisionTrainer` and
    `BatchMajorVisionTrainer`
- Tools:
  - `tools/clip/dispatch_profile.ts`
  - `tools/clip/pointwise_report.ts`
  - `tools/clip/batch_major_train_bench.ts`
  - `tools/clip/batch_major_train_matrix.ts`
  - `tools/clip/fused_test.ts`
  - `tools/splat3d/step_bench.ts`
  - `tools/splat3d/step_matrix.ts`
- History:
  - `experiments/clip_forks/v10_shared_w_pointwise_forward/README.md`
  - `experiments/clip_forks/v11_backward_local_fusions/README.md`
  - `experiments/clip_forks/v18_pointwise_rect8x16/README.md`
  - v18 side notes in `agent_notes/optimization_session/`

## Current Plan Facts

From `models/mobileclip_s0/plan_train.json`:

```text
forward steps:       129
backward entries:    152
slots:               260
nActSlots:           130

forward conv:general    5
forward conv:depthwise 40
forward conv:pointwise 48
forward se              3
forward attn_core       2
forward head            1
forward gelu           30

backward pw_bwd         48
backward spatial_bwd    45
backward gelu_bwd       30
backward residual_bwd   22
backward se_bwd          3
backward attn_core_bwd   2
```

Forward producer-to-GELU pairs:

```text
total exact pairs: 30
pointwise -> gelu: 24
general conv -> gelu: 2
depthwise conv -> gelu: 1
se -> gelu: 3
```

Backward exact adjacent pairs:

```text
gelu_bwd -> pw_bwd:       24
residual_bwd -> pw_bwd:   22
gelu_bwd -> se_bwd:        3
gelu_bwd -> spatial_bwd:   3
```

The pointwise family remains the largest exact shader target. v17 recorded B=3
train pointwise totals of 48 forward pointwise dispatches, 48 backward
`pw_bwd` dispatches, 24 pointwise+GELU candidates, 26.575 GFLOP, about
3445 MiB lower-bound staged traffic, and 51,648 pointwise workgroups.

## Why Pointwise+GELU Is Legal

Current implementation:

- `src/clip/vision_wgsl.ts`
  - `pointwiseFusedGelu()`
  - `pointwiseRect8x16FusedGelu()`
- `src/clip/vision_batch_wgsl.ts`
  - `forwardDispatches()` detects adjacent `conv:pointwise -> gelu` pairs when
    `fusePointwiseGeluForward` is enabled.

The key guardrails are:

- producer is a pointwise conv;
- train conv has `act === "none"` because the train plan split GELU;
- no residual/layer-scale epilogue in the fused-GELU producer;
- `gelu.src === conv.dst`;
- `gelu.n === cout * H * W`;
- fused shader writes the conv pre-activation slot and the GELU output slot.

So the fusion removes a dispatch and a read/write pass, but it does not discard
the pre-activation. That is why backward still works.

## Why A Giant CLIP Shader Is The Wrong Target

### 1. No grid-wide barrier inside one dispatch

`VisionTrainer.encode()` and `BatchMajorVisionTrainer.encode()` put all CLIP
dispatches into one compute pass, but the dispatches are still ordered. Storage
buffer writes from one dispatch are visible to later dispatches. Inside one
dispatch, `workgroupBarrier()` only synchronizes one workgroup.

Pointwise layer N cannot safely be consumed by pointwise/spatial/SE/attention
layer N+1 inside the same dispatch unless all producing workgroups have
finished. WebGPU does not expose that barrier.

### 2. The kernels have incompatible shapes

Examples:

- pointwise: 8x8 tiled channel matmul, `array<vec4f>` pixel quads;
- rect8x16: wider output-channel tile with 12 KiB workgroup memory;
- spatial conv: one output channel by spatial tile, tap loops and border paths;
- SE: one workgroup reduction over a full channel gate;
- attention core: one workgroup per head with token-private score rows;
- head/loss: reductions;
- backward: reverse-order grad slots with overwrite/add semantics.

A giant shader would lose the current per-shape specialization or become an
enormous generated program with many branches and high register pressure.

### 3. Train mode intentionally saves activations

`compile_plan.py --train` keeps forward activations because backward needs:

- GELU pre-activations;
- SE saved inputs;
- attention QKV;
- embedding for loss;
- normal activation outputs for gradient routing.

Any fusion that stops writing those slots is not a local fusion. It is
activation checkpointing or recomputation, which is a compiler/runtime design
change.

### 4. Backward accumulation is part of correctness

The backward plan encodes first writer overwrites and later writers add via
`accumulate`. Fusing broad backward regions risks write races unless the pair is
adjacent and has a single clear producer/consumer relation.

### 5. Verification would get worse

The current architecture can bisect per step with `fused_test.ts`,
`bwd_test.ts`, and `dispatch_profile.ts`. A monolithic shader would hide the
exact broken layer and make small f32 changes harder to isolate.

## v11 History

v11 is the strongest evidence that local backward fusion is real but modest.

Implemented gates now exist in `src/clip/vision_bwd_wgsl.ts`:

- `fuseGeluBwdIntoPw`
  - `canFuseGeluBwdIntoPw()`
  - `pwBwdAfterGelu()`
- `fuseResidualBwdIntoPw`
  - `canFuseResidualBwdIntoPw()`
  - `pwBwdWithResidualCopy()`
- planner integration:
  - `planBwdDispatches()`

v11 recorded:

```text
depthwise4 baseline dispatches:                 257
depthwise4 + gelu/residual backward fusions:     211

isolated timestamp sum:
  depthwise4:                    86.704 ms
  depthwise4 + both fusions:      54.591 ms

integrated grid80+depthwise4:
  normal median: 56.20 -> 54.63 ms
  CLIP median:   36.41 -> 34.91 ms
```

Read: the dispatch removal is real. The integrated win is small, not a 2x
answer. It is still one of the largest legal local fusion opportunities because
it covers 46 adjacent backward entries.

Recommendation: retest v11 together with current defaults and consider
promoting it only if same-session `step_matrix` still shows a clean win. The
likely matrix token is:

```bash
TRIALS=7 CONFIGS=base=3:3,v11=3:3:gelubwd:resbwd RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

For the grid stack:

```bash
TRIALS=7 CONFIGS=grid80dw4=9:3:grid9:directgrid:dw4,grid80dw4v11=9:3:grid9:directgrid:dw4:gelubwd:resbwd RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

## v18 History

v18 tried a gated exact forward pointwise tile:

```text
baseline tile:   8 pixel-quads x 32 cout
rect8x16 tile:   8 pixel-quads x 64 cout
workgroup size:  8 x 16 = 128 threads
xS:              256 vec4f
wS:              512 vec4f
workgroup mem:   12 KiB
```

Implemented in current code:

- `PointwiseTileVariant = "default" | "rect8x16"`
- `DispatchOptions.pointwiseTileVariant`
- `DispatchOptions.pointwiseTileSteps`
- `pointwiseRect8x16Main()`
- `pointwiseRect8x16()`
- `pointwiseRect8x16FusedGelu()`
- tool envs:
  - `PW_TILE_VARIANT=rect8x16`
  - `PW_TILE_STEPS=...`

Correctness passed:

```text
fused_test train-forward: embedding cosine 1.000000
batch-major train gradient gate: lane gradients cosine 1.000000, relLinf 0
```

Timing did not justify promotion:

```text
CLIP-only train matrix, B=3:
  base:   41.63 ms/batch
  rect16: 41.43 ms/batch

isolated timestamp, one run:
  base total:   79.17 ms
  rect16 total: 100.73 ms
  pw:           10.29 -> 16.38 ms
  pw+gelu:       9.70 -> 16.97 ms

integrated broad allowlist:
  base normal:   55.99 ms
  rect16 normal: 55.92 ms
  base CLIP:     42.55 ms
  rect16 CLIP:   44.53 ms
```

Important tooling caveat: current `src/splat3d/optimize.ts` accepts
`pointwiseTileVariant` and `pointwiseTileSteps` in the optimizer config and
passes them to the single-image `VisionTrainer`, but the B>1
`BatchMajorVisionTrainer.create()` call does not currently pass those two
fields. The CLIP-only `dispatch_profile.ts` and `batch_major_train_matrix.ts`
do exercise v18 for B=3. The integrated `step_matrix` B=3 `pwrect` runs should
be treated as weak/noisy until that optimizer plumbing is fixed.

Decision: keep `rect8x16` gated. Do not promote it. The useful part of v18 is
the tile plumbing and matrix-token plumbing, not the measured tile.

## Biggest Legal/Local Fusion Opportunities Remaining

### 1. Retest or promote the v11 backward local fusions

Status: already implemented, not default.

Why it matters:

- 24 `gelu_bwd -> pw_bwd` pairs.
- 22 `residual_bwd -> pw_bwd` pairs.
- This removes 46 dispatches when both gates apply.
- It preserves saved activations and gradient semantics.

Files/functions:

- `src/clip/vision_bwd_wgsl.ts`
  - `pwBwdAfterGelu()`
  - `pwBwdWithResidualCopy()`
  - `planBwdDispatches()`
- `tools/clip/dispatch_profile.ts`
  - `FUSE_GELU_BWD_PW`
  - `FUSE_RESIDUAL_BWD_PW`
- `tools/clip/batch_major_train_matrix.ts`
  - `gelubwd`
  - `resbwd`
- `tools/splat3d/step_matrix.ts`
  - `gelubwd`
  - `resbwd`
- `src/splat3d/optimize.ts`
  - `fuseGeluBwdIntoPw`
  - `fuseResidualBwdIntoPw`

This is the first thing to remeasure, not because it is revolutionary, but
because it is exact, already present, and has prior positive integrated data.

### 2. Fuse remaining non-pointwise forward producer -> GELU pairs

Status: not implemented.

Exact pairs:

```text
0   conv:general   3->64 @128x128       n=1048576
2   conv:depthwise 64->64 @64x64        n=262144
16  conv:general   64->128 @32x32       n=131072
51  se             c256 mid64 @16x16    n=65536
106 se             c512 mid128 @8x8     n=32768
126 se             c1024 mid64 @8x8     n=65536
```

Why it is legal:

- same local structure as pointwise+GELU;
- producer writes pre slot;
- fused producer also writes `gelu(pre)` to the next slot;
- backward still reads the pre slot.

Expected size:

- only 6 dispatches;
- but the stem/downsample tensors are large, especially `n=1048576`.

Files/functions to change:

- `src/clip/vision_wgsl.ts`
  - add `spatialConvFusedGelu(s: ConvStep, gelu: GeluStep, opts)`
  - add `seStepFusedGelu(s: SeStep, gelu: GeluStep, opts)`
  - add a lookahead planner, or expose helpers for `vision_batch_wgsl.ts`
- `src/clip/vision_batch_wgsl.ts`
  - extend `BatchDispatchOptions`, for example
    `fuseNonPointwiseGeluForward?: boolean`
  - extend `forwardDispatches()` to detect:
    - `conv:general -> gelu`
    - `conv:depthwise -> gelu`
    - `se -> gelu`
- `tools/clip/dispatch_profile.ts`
  - parse `FUSE_NONPW_GELU=1`
  - add group labels such as `conv+gelu` and `se+gelu`
- `tools/clip/batch_major_train_bench.ts`
  - parse and print the new flag
- `tools/clip/batch_major_train_matrix.ts`
  - add token `nonpwgelu`
- `src/splat3d/optimize.ts`
  - add config field and pass it to `BatchMajorVisionTrainer`
- `tools/splat3d/step_bench.ts`
  - parse env
- `tools/splat3d/step_matrix.ts`
  - add token `nonpwgelu`

### 3. Fuse remaining non-pointwise GELU backward pairs

Status: not implemented.

Exact pairs:

```text
gelu_bwd -> se_bwd:
  backward 2   se c1024 mid64 @8x8
  backward 26  se c512 mid128 @8x8
  backward 91  se c256 mid64 @16x16

gelu_bwd -> spatial_bwd:
  backward 132 spatial 64<-128 k7s2 g64 @64x64
  backward 148 spatial 64<-64  k3s2 g64 @128x128
  backward 150 spatial 3<-64   k3s2 g1  @256x256
```

Why it is legal:

Baseline:

```text
gelu_bwd:      dPre = dPost * geluGrad(pre)
spatial/se bwd reads dPre
```

Fused:

```text
spatial/se bwd reads dPost and pre, applies geluGrad(pre) at each dy use
```

Expected size:

- 6 dispatches;
- removes large gradient-slot materialization for the stem/downsample GELUs;
- probably smaller than v11 but cleaner than broad graph fusion.

Files/functions to change:

- `src/clip/vision_bwd_wgsl.ts`
  - export or duplicate `GELU_GRAD`
  - add `canFuseGeluBwdIntoSpatial(gelu, spatial)`
  - add `canFuseGeluBwdIntoSe(gelu, se)`
  - add `spatialBwdAfterGelu()`
  - add `spatialBwdDepthwise4AfterGelu()`
  - add `spatialBwdStem4AfterGelu()`
  - add `seBwdAfterGelu()`
  - extend `planBwdDispatches()`
- `tools/clip/bwd_test.ts`
  - add focused unit coverage if the existing full gate does not isolate these
    fused paths clearly
- Same env/tool plumbing as the forward non-pointwise GELU flag, probably:
  - `FUSE_NONPW_GELU_BWD=1`
  - matrix token `nonpwgelubwd`

### 4. Shared-W batch `pw_bwd`

Status: not implemented.

This is not graph fusion, but it is a major local shader fusion across batch
lanes. Batch-major B=3 currently runs one workgroup per image lane for
`pw_bwd`, so every lane stages the same W tile. v10 tried shared-W forward
pointwise and did not promote; backward remains untried and `pw_bwd` is the
largest individual group.

Potential workgroup layout:

```text
@workgroup_size(8, 8, B)
xS: 256 * B vec4f
wS: 256 vec4f
B=3 memory: 12 KiB + 4 KiB = 16 KiB
```

Why it is plausible:

- weights are frozen and shared across image lanes;
- each lane has independent `dY`, `pre`, and `dX`;
- no cross-lane mathematical dependency;
- can target the largest `pw_bwd` shapes first.

Risks:

- B=3 is at the 16 KiB workgroup-memory envelope;
- v10 forward shared-W showed B=3 can lose from occupancy pressure;
- fused `gelu_bwd` and `residual_bwd` variants complicate the first version.

Files/functions to change:

- `src/clip/vision_batch_pointwise.ts`
  - add `pointwiseSharedWBatchBackwardDispatch(plan, pw, batch, opts)`
  - later add variants for `gelu_bwd -> pw_bwd` and
    `residual_bwd -> pw_bwd`
- `src/clip/vision_bwd_wgsl.ts`
  - export required backward types and maybe `GELU_GRAD`
- `src/clip/vision_batch_wgsl.ts`
  - add `sharedWBackwardSteps?: ReadonlySet<number>` to
    `BatchDispatchOptions`
  - replace selected `pw_bwd` entries before generic `batchSpec()`
  - avoid colliding with `fuseGeluBwdIntoPw` and `fuseResidualBwdIntoPw` in the
    first fork
- `tools/clip/dispatch_profile.ts`
  - parse `SHARED_W_BWD_STEPS`
  - group label `pw_bwd-shared-w`
- `tools/clip/batch_major_train_bench.ts`
  - parse and print `SHARED_W_BWD_STEPS`
- `tools/clip/batch_major_train_matrix.ts`
  - add token like `swbwd8-13-17-22`
- `src/splat3d/optimize.ts`
  - add `sharedWBackwardSteps` config and pass it to batch CLIP
- `tools/splat3d/step_bench.ts`
  - parse env
- `tools/splat3d/step_matrix.ts`
  - add matrix token

Suggested first targets:

```text
pw_bwd indexes 8,13,17,22: 1536->512 @8x8
then 6,15:                  512->1536 @8x8
then selected 16x16 families only if B=3 still wins
```

### 5. Split-K `pw_bwd` for low-spatial high-channel shapes

Status: designed in v18 notes, not implemented.

This is also not graph fusion; it is a scheduling/fusion-style rewrite of one
hot matmul. It targets occupancy-starved `8x8` `pw_bwd` layers where there are
few workgroups and a long K loop.

First target:

```text
SPLITK_PW_BWD_STEPS=8,13,17,22
SPLITK_PW_BWD_PARTS=4
```

These are the four `1536->512 @8x8` backward entries.

Design:

```text
dispatch 1: partial sums
  workgroups [P4/8, cout/32, batch * parts]
  write scratch[batch, part, cout, P4]

dispatch 2: reduce partial sums
  workgroups [P4/8, cout/32, batch]
  apply existing accumulate semantics only here
```

Files/functions to change:

- `src/clip/vision_wgsl.ts`
  - extend `BufferRef` with a scratch buffer kind, or introduce a parallel
    `DispatchResourceRef` type
- `src/clip/vision_bwd_wgsl.ts`
  - add `BwdDispatchOptions.splitKPwBwd`
  - add `pwBwdSplitKPartial()`
  - add `pwBwdSplitKReduce()`
  - add `pwBwdSplitKDispatches()`
  - extend `planBwdDispatches()`
- `src/clip/vision.ts`
  - allocate and resolve scratch buffers in `VisionTrainer`
- `src/clip/vision_batch.ts`
  - allocate and resolve scratch buffers in `BatchMajorVisionTrainer`
- `src/clip/vision_batch_wgsl.ts`
  - make split-K batch-aware so `workgroup_id.z` is not used twice
- `tools/clip/dispatch_profile.ts`
  - allocate scratch and group labels:
    - `pw_bwd_splitk_partial`
    - `pw_bwd_splitk_reduce`
- `tools/clip/bwd_test.ts`
  - add a two-dispatch `pw_bwd` unit path
- `tools/clip/batch_major_train_bench.ts`
  - parse `SPLITK_PW_BWD_STEPS` and `SPLITK_PW_BWD_PARTS`
- `tools/clip/batch_major_train_matrix.ts`
  - token like `splitk8-13-17-22p4`
- `src/splat3d/optimize.ts`, `tools/splat3d/step_bench.ts`,
  `tools/splat3d/step_matrix.ts`
  - plumb the same config

This is a v20-sized fork because scratch buffers affect runtime resource
resolution, not just WGSL strings.

### 6. True dual-output pointwise tile, not v18 rect8x16

Status: not implemented.

v18 widened the workgroup to `8 x 16 = 128` threads. The next exact tile should
test whether keeping 64 threads and giving each thread two output-channel quads
is better:

```text
workgroup_size: 8 x 8 = 64
tile:           8 pixel-quads x 64 cout
xS:             256 vec4f
wS:             512 vec4f
accumulators:   8 vec4f per thread
```

This saves the same X tile reload v18 was chasing, but avoids doubling thread
count. It may still lose from register pressure, so keep it gated.

Files/functions to change:

- `src/clip/vision_wgsl.ts`
  - add a separate `pointwiseDualCoutMain()`
  - add `pointwiseDualCout()`
  - add `pointwiseFusedGeluDualCout()`
  - extend `PointwiseTileVariant`, perhaps `"dual_cout"`
- `src/clip/vision_batch_wgsl.ts`
  - choose this variant before the generic pointwise+GELU branch
- Tool/env plumbing:
  - `PW_TILE_VARIANT=dual_cout`
  - existing `PW_TILE_STEPS`

Start with a narrow allowlist:

```text
57,59,62,64
```

Then broaden only if timestamps and full CLIP train matrix win.

### 7. Selective pointwise f16 matrix reads

Status: designed in v18 notes, not implemented.

This is not exact fusion, so it needs a stricter `dL/dimage` gate. Do not retry
global weight f16; v02 already showed all-weight f16 can pass embedding cosine
but fail input-gradient cosine.

Safer design:

- keep input, activations, gradients, text, and accumulators f32;
- keep bias/layer-scale f32;
- bind a second f16 weights buffer only for selected pointwise matrix reads;
- leave non-pointwise and sensitive weights f32.

Files/functions to change:

- `src/clip/vision_wgsl.ts`
  - add separate f32/f16 weight declarations for pointwise variants
  - add pointwise-only selected f16 emitters
- `src/clip/vision_bwd_wgsl.ts`
  - add selected f16 `pw_bwd` emitters
- `src/clip/vision.ts` and `src/clip/vision_batch.ts`
  - accept and bind an optional f16 weights buffer
- `tools/clip/pack_f16_weights.ts`
  - either reuse full f16 blob or emit a selected sidecar
- `tools/clip/f16_compare.ts`
  - compare f32 teacher vs selected pointwise f16
- Tool/env:
  - `PW_F16_POLICY=fwd_ffn16`
  - `PW_F16_POLICY=fwd_bwd_ffn16`

Minimum gate:

```text
embedding cosine >= 0.9995
dL/dimage cosine >= 0.995
no NaN/Inf
norm ratio close to 1.0
```

## Recommended v19/v20 Priority

### v19.0: Fix gate plumbing and retest existing wins

Do this before inventing new kernels.

1. In `src/splat3d/optimize.ts`, pass `pointwiseTileVariant` and
   `pointwiseTileSteps` into `BatchMajorVisionTrainer.create()`. Current B=3
   integrated `pwrect` paths do not appear to receive those options.
2. Rerun v11:

```bash
TRIALS=7 CONFIGS=base=3:3,v11=3:3:gelubwd:resbwd RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

3. Rerun v18 `rect8x16` only after the B=3 integrated plumbing is confirmed,
   but expect it to stay gated because CLIP-only timing was flat/worse.

### v19.1: Finish exact local GELU fusions

Implement:

- `FUSE_NONPW_GELU=1`
- `FUSE_NONPW_GELU_BWD=1`

This is the cleanest "why not one more fusion" answer. It covers the 6
remaining forward GELUs and 6 matching non-pointwise backward GELU consumers.
The likely integrated win is small, but it is exact and narrows dispatch count.

### v19.2: Shared-W batch `pw_bwd`

Implement selected B=3 shared-W backward pointwise. This is the highest-upside
local shader fusion that does not require scratch buffers or precision changes.

First run it without v11 fusions. If it wins, add variants compatible with:

- `gelu_bwd -> pw_bwd`
- `residual_bwd -> pw_bwd`

### v20.0: Split-K `pw_bwd`

Implement scratch-buffer support and target `pw_bwd` indexes `8,13,17,22`.
This is more ambitious than shared-W because it touches runtime resource
allocation and dispatch expansion. It is still plausible because it targets a
specific occupancy problem.

### v20.1: True dual-cout pointwise tile

Try `PW_TILE_VARIANT=dual_cout` with a narrow forward allowlist. This is the
better follow-up to v18 than another broad `rect8x16` sweep.

### v20.2: Selective pointwise f16

Only after exact fusions/tile experiments. Treat it as approximate and require
full input-gradient gates.

## Experiments I Would Not Prioritize

### QKV + attention core + projection mega-fusion

The forward attention neighborhood is:

```text
qkv pointwise 512->1536 @8x8
attn_core h16 n64
proj pointwise 512->512 @8x8
```

Fusing qkv into attention would make each head workgroup compute all Q/K/V
dots for that head. That is a huge amount of matmul work in only 16 workgroups,
which is likely worse than the current 96-workgroup qkv pointwise dispatch.
Fusing attention into projection needs all heads before projection and therefore
needs a global barrier. Do not start here.

### Spatial conv + following pointwise

Pointwise needs all channels at a spatial position. Spatial conv workgroups
produce per-channel/tap outputs. Without materializing the spatial output, the
pointwise kernel would have to recompute spatial conv for every output-channel
tile. That is not a local fusion.

### Full ConvFFN block in one dispatch

The block:

```text
pw1 expansion -> GELU -> pw2 contraction -> residual
```

looks tempting, but `pw2` needs the full hidden channel tensor. A single
dispatch cannot have all `pw1` workgroups finish before `pw2` consumers run.
The plausible research version is checkpoint/recompute or a multi-dispatch
block with scratch. That belongs after shared-W and split-K `pw_bwd`, not
before.

## Verification Commands For New Work

Exact forward changes:

```bash
PLAN=plan_train.json BENCH_RUNS=5 bun tools/clip/fused_test.ts
```

Backward changes:

```bash
bun tools/clip/bwd_test.ts
```

B=3 CLIP train:

```bash
TRIALS=5 BATCH=3 RUNS=5 WARMUP=5 CONFIGS='base=stem,gelu;variant=stem,gelu:<token>' bun tools/clip/batch_major_train_matrix.ts
```

Dispatch attribution:

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=2 bun tools/clip/dispatch_profile.ts
```

Integrated optimizer:

```bash
TRIALS=7 CONFIGS=base=3:3,variant=3:3:<token> RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

Promotion bar:

- correctness first;
- input-gradient cosine for any approximate/precision fork;
- same-session alternating matrices;
- real integrated step win, not only isolated timestamp movement;
- no hidden CLIP resolution, view schedule, prompt, or raster changes inside a
  shader experiment.

## Bottom Line

Pointwise+GELU helps because it is the right fusion granularity: adjacent,
same-geometry enough to use register values, and still writes train-mode saved
activations. A whole-CLIP shader violates the dependency and synchronization
shape of the model.

The largest exact local fusion already available is v11 backward
`gelu_bwd/residual_bwd -> pw_bwd`. The largest unimplemented local opportunities
are non-pointwise GELU forward/backward cleanup and shared-W batch `pw_bwd`.
The most ambitious plausible v20 shader fork is split-K `pw_bwd` with scratch
buffers. v18 `rect8x16` should stay gated; the next pointwise tile should be a
different design, not a broader rollout of the measured loser.
