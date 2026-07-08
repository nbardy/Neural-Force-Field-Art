# Agent Note: Pointwise CLIP Math, Layout, And Fusion

Date: 2026-07-08

Scope: documentation-only study of the current CLIP pointwise kernels. Source
code was not changed.

Files inspected:

- `src/clip/vision_wgsl.ts`
- `src/clip/vision_bwd_wgsl.ts`
- `src/clip/vision_batch_wgsl.ts`
- `src/clip/vision_batch_pointwise.ts`
- `tools/clip/dispatch_profile.ts`
- `tools/clip/pointwise_batch_bench.ts`
- `tools/clip/batch_major_train_bench.ts`
- `docs/CLIP_BATCHING_NOTES.md`
- `docs/clip_backward_spec.md`
- `experiments/clip_forks/v02_f16_weights/results/2026-07-08.md`
- `experiments/clip_forks/v03_residual_bwd_pw_fusion/README.md`
- `experiments/clip_forks/v04_pointwise_tile_rewrite/README.md`
- `agent_notes/optimization_session/clip_timestamp_dispatch_profile.md`
- `agent_notes/optimization_session/clip_2x_4x_trace_status.md`

## Short Answer

In this model, "pointwise" means a `1x1`, `groups=1` convolution. It is not a
spatial convolution. It is the same operation as a dense matrix multiply at
every spatial/token position:

```text
Y[p, co] = bias[co] + sum_ci X[p, ci] * W[ci, co]
```

The current MobileCLIP train plan has:

```text
forward steps:             129
backward steps:            152
forward pointwise steps:    48
backward pw_bwd steps:      48
pointwise+GELU pairs:       24
forward conv MACs:          2.376B
forward pointwise MACs:     2.215B
pointwise share of conv MACs: 93.2%
```

A fresh local timestamp run on `apple metal-3`:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

reported:

| Group | Time | Share |
| --- | ---: | ---: |
| `pw_bwd` | `21.234 ms` | `25.2%` |
| `spatial_bwd` | `18.940 ms` | `22.5%` |
| `conv` | `11.600 ms` | `13.8%` |
| `pw+gelu` | `11.534 ms` | `13.7%` |
| `pw` | `11.469 ms` | `13.6%` |

So the pointwise family is `21.234 + 11.534 + 11.469 = 44.237 ms`, or about
`52.5%` of this isolated CLIP timestamp sum. That is why it keeps coming up as
the likely CLIP bottleneck.

## Fp16 Reality Check

The big fp16 speedup we remember was real, but it was not this CLIP vision
path. It was the older force-field advect shader path.

For CLIP vision, the implemented all-weight fp16 fork is gated and is not
promoted. It halves payload size:

```text
weights_train.bin -> weights_train_f16.bin: 82.1 MB -> 41.0 MB
```

but the result note says:

```text
embedding cos: 0.99999559
inputGrad cos: 0.97493807
planned inputGrad gate: >= 0.995
```

and the stable isolated timestamp comparison was slower:

```text
f32 total isolated median sum: 67.109 ms
f16 weights total isolated median sum: 71.107 ms
```

So fp16 is not currently a solved CLIP speed win. The plausible future fp16
path is selective, probably pointwise-heavy weights and maybe selected interior
activation storage, with f32 accumulation and strict gradient gates.

## Tensor Shapes And Math

The source layout contract is explicit in `vision_wgsl.ts`:

```text
activations: NCHW planar f32
x[c][y][x] at c * H * W + y * W + x
pointwise W: transposed [Cin][Cout]
depthwise W: [C][k*k]
general W: [Cout][cpg][k][k]
```

For pointwise:

```text
P  = H * W
P4 = P / 4
src[ci][p4] = vec4f(X[ci, p4*4 + 0..3])
dst[co][p4] = vec4f(Y[co, p4*4 + 0..3])
```

Forward weights are stored so four adjacent output channels are contiguous:

```text
scalar W index = wOff + ci * Cout + co
W4((wOff + ci * Cout + co4) / 4) = W[ci, co4..co4+3]
```

Forward computes:

```text
Y[co, p] = bias[co] + sum_ci X[ci, p] * W[ci, co]
```

If the forward pointwise conv has residual/layer-scale, the epilogue is:

```text
Y[co, p] = residual[co, p] + layerScale[co] * act(pre[co, p])
```

Backward `pw_bwd` computes the input gradient only. CLIP weights are frozen, so
there is no `dW` kernel:

```text
dX[ci, p] = sum_co dY[co, p] * W[ci, co]
```

The compiler emits a second packed orientation `wOffT`, so the same tiled
pointwise body can read weights in `[reduction][output]` order for backward.
In `PwBwdStep` naming:

```text
pw.cin  = reduction channels = forward cout = dY channels
pw.cout = output channels    = forward cin  = dX channels
```

## Current Pointwise Kernel Layout

The main pointwise kernel is `pointwiseTiledMain()`.

```text
workgroup_size = 8 x 8 = 64 threads
workgroups     = [P4 / 8, Cout / 32, batch]
tile owns      = 8 pixel-quads x 32 output channels
               = 32 scalar pixels x 32 output channels
thread owns    = 1 pixel-quad x 4 output channels
               = 16 scalar outputs
```

The thread coordinates:

```text
p4 = wid.x * 8 + lid.x
co = (wid.y * 8 + lid.y) * 4
```

Each thread accumulates four `vec4f`s:

```text
acc0 = output channel co + 0, four adjacent pixels
acc1 = output channel co + 1, four adjacent pixels
acc2 = output channel co + 2, four adjacent pixels
acc3 = output channel co + 3, four adjacent pixels
```

Shared memory:

```text
xS: 256 vec4f = 4096 bytes  // 32 ci x 8 pixel-quads
wS: 256 vec4f = 4096 bytes  // 32 ci x 8 cout-quads
total: 8192 bytes per workgroup
```

The reduction loops over `Cin` in 32-channel chunks:

```text
for ci0 in 0..Cin step 32:
  stage xS and wS
  workgroupBarrier()
  for ci in 0..31:
    xv = xS[ci, lid.x]  // 4 pixels
    wv = wS[ci, lid.y]  // 4 output channels
    acc0 += wv.x * xv
    acc1 += wv.y * xv
    acc2 += wv.z * xv
    acc3 += wv.w * xv
  workgroupBarrier()
```

Batch-major CLIP stores every activation slot as:

```text
[batch][slotFloats]
```

`vision_batch_wgsl.ts` rewrites each slot/text binding access with:

```text
base = workgroup_id.z * slotStride
buffer[base + originalIndex]
```

That means the normal batch-major pointwise path has one workgroup per batch
lane in `workgroups.z`. The weights buffer is shared globally, but each batch
lane stages the same W tile independently.

## Why It Is Expensive

Pointwise dominates because it is the dense channel-mixing part of MobileCLIP.
The repeated ConvFFN blocks are mostly:

```text
pointwise expansion -> GELU -> pointwise contraction -> residual
```

The largest repeated forward pointwise shapes in `plan_train.json` are:

| Shape | Count | Forward MACs |
| --- | ---: | ---: |
| `256->768 @16x16 +gelu` | 10 | `503.3M` |
| `768->256 @16x16 residual` | 10 | `503.3M` |
| `128->384 @32x32 +gelu` | 6 | `302.0M` |
| `384->128 @32x32 residual` | 6 | `302.0M` |
| `64->192 @64x64 +gelu` | 2 | `100.7M` |
| `192->64 @64x64 residual` | 2 | `100.7M` |
| `512->1536 @8x8` / related late shapes | multiple | `~100.7M` each group |

Train mode pays the forward pointwise chain and then a similarly sized backward
pointwise chain. With `BATCH=3`, each selected camera view is also a separate
batch lane.

The current kernel is already a real tiled matmul, not a naive loop. The cost
is still high because:

- there are 48 forward pointwise steps and 48 backward pointwise steps;
- each pointwise step has high arithmetic intensity over dense channel pairs;
- every batch lane reloads/stages W in the default z-batch path;
- each tile has two workgroup barriers per 32-channel chunk;
- forward train mode often writes saved pre-activation slots for backward;
- residual/layer-scale tails add residual reads and epilogue math;
- backward may add into existing grad slots for `accumulate:true` edges;
- the source/destination slots are large f32 storage buffers, so slot traffic
  matters even when W is well tiled.

## What Pointwise + GELU Fusion Does

Train mode needs GELU pre-activations for backward, so the original split shape
is:

```text
pointwise:
  pre = X @ W + b
  write pre slot

gelu:
  post = gelu(pre)
  read pre slot
  write post slot
```

`pointwiseFusedGelu()` keeps the backward contract but performs GELU while the
pointwise result is still in registers:

```text
pointwise+gelu:
  pre = X @ W + b
  write pre slot
  write gelu(pre) to post slot
```

This removes the standalone forward GELU dispatch and the separate read of the
pre slot for that forward GELU. It still writes both slots because backward
needs the pre-activation to compute:

```text
dPre = dPost * geluGrad(pre)
```

The current gate covers exact adjacent pointwise-conv -> GELU pairs where:

```text
step.kind == "conv"
step.variant == "pointwise"
next.kind == "gelu"
next.src == step.dst
```

Existing results in `docs/CLIP_BATCHING_NOTES.md`:

```text
dispatch count: 281 -> 257
B=3 CLIP train median: 73.33 ms -> 68.06 ms
integrated 3D CLIP median: 80.68 ms -> 76.38 ms
```

So this fusion is real, but it is not a whole-CLIP fusion. It is a local
epilogue fusion.

There is also a gated backward fusion:

```text
gelu_bwd -> pw_bwd
```

It replaces:

```text
gelu_bwd:
  dPre = dPost * geluGrad(pre)

pw_bwd:
  dX = W^T @ dPre
```

with a `pw_bwd+gelu` load expression:

```text
loadSrc = dPost * geluGrad(pre)
```

That passed parity and helped a CLIP-only bench, but did not produce a clear
integrated 3D step win, so it remains gated.

## Why Not Fuse It All Into One Huge Shader?

The tempting target is a whole ConvFFN residual block:

```text
pw1 expansion -> GELU -> pw2 contraction -> layer_scale + residual
```

The problem is that `pw2` needs the full hidden vector produced by `pw1 + GELU`
for each spatial position. In the current tiling, a workgroup computes only a
tile of pixels and output channels. WebGPU workgroup barriers only synchronize
threads inside one workgroup. They cannot synchronize every workgroup that
produces hidden channels before every workgroup that consumes those hidden
channels.

So a one-dispatch block fusion has three bad options:

1. Store the hidden tensor anyway.
   - Then the main memory traffic is still there, and the fusion is mostly just
     a dispatch-count change.

2. Recompute hidden chunks inside each `pw2` tile.
   - This repeats `pw1` for each output-channel tile of `pw2`, which can make
     total math much worse than the original.

3. Make one enormous workgroup own enough pixels/channels to keep the hidden
   tensor local.
   - That breaks workgroup memory limits, register pressure, occupancy, or all
     three for the real MobileCLIP shapes.

Bigger fusion is also hard because the CLIP graph is heterogeneous:

- spatial convs use different kernels and weight layouts;
- SE blocks do reductions and tiny FCs in one workgroup;
- attention uses one workgroup per head with token-private arrays;
- backward needs saved activations, grad-slot accumulation semantics, and exact
  first-writer/add behavior;
- batch-major rewriting expects slot buffers with stable per-lane offsets;
- giant generated shaders would compile slowly and be hard to gate by shape.

The right fusion strategy is therefore local and shape-specific: fuse producer
and consumer when the data dependency is inside the same tile or can be
expressed as an epilogue/load hook. The existing pointwise+GELU fusion is a
good example. Whole-block fusion is a research fork, not a default-path edit.

## Ambitious But Plausible Shader Fork Ideas

### 1. Shape-Gated Pointwise Tile Variants

Implement new pointwise emitters behind allowlists instead of changing
`pointwiseTiledMain()` globally.

Candidate variants from the existing v04 fork plan:

```text
dual_pixel:
  current tile:  8 pixel-quads x 32 cout
  variant tile: 16 pixel-quads x 32 cout
  goal: reuse W over twice as many pixels

dual_cout:
  current tile:  8 pixel-quads x 32 cout
  variant tile: 8 pixel-quads x 64 cout
  goal: reuse X over twice as many output channels

rect_8x16:
  workgroup_size: 8 x 16 = 128 threads
  tile: 8 pixel-quads x 64 cout
  goal: reduce dual-cout register pressure by adding threads
```

The fork should target only timestamp-hot repeated shapes, for example:

```text
early/high-P: 64->192 @64x64, 192->64 @64x64
mid:          128<->384 @32x32, 256<->768 @16x16
late:         512<->1536 @8x8, 1536->512 @8x8
```

Why it could win:

- current tiles reload W across spatial tiles and X across output-channel
  tiles;
- a bigger rectangular tile may reduce one side of that reload traffic;
- pointwise is large enough that a 15-25% pointwise-family win matters.

Why it could lose:

- more accumulators can cause register spills/private memory;
- larger workgroups can reduce occupancy;
- more workgroup memory can reduce concurrent groups;
- shape wins may vanish in full-chain CLIP timing.

Promotion gates should require timestamp wins and integrated 3D wins in the
same session, not just compact microbench wins.

### 2. Selective Pointwise Precision And Layout Fork

The all-weight fp16 fork is not good enough, but selective pointwise precision
is still plausible.

Fork idea:

- keep default f32 weights and code path;
- add a separate packed fp16 pointwise-weight buffer for selected pointwise
  layers only;
- use f32 accumulators and f32 activations;
- optionally keep late/head/attention/SE weights f32;
- gate by pointwise family and exact shape.

Why this is different from the rejected all-weight fp16:

- it avoids perturbing every model subsystem at once;
- it targets the family where W bandwidth plausibly matters;
- it gives us f32 fallbacks per shape;
- it can be measured against pointwise timestamp rows before touching the
  broader graph.

A second phase could test selected f16 interior activation slots, but not input,
text, embedding, loss reductions, attention softmax reductions, or multiply
written gradient accumulation slots.

### 3. Checkpointed ConvFFN Block Fusion Prototype

This is the high-risk "can we fuse more than epilogues?" experiment.

Fork a single repeated ConvFFN shape, for example:

```text
256->768 @16x16 -> GELU -> 768->256 @16x16 residual
```

Try a checkpointed two-dispatch block:

1. Dispatch A computes `pw1 + GELU` and stores either:
   - normal hidden, or
   - a compressed/packed hidden tile suitable for `pw2`.

2. Dispatch B computes `pw2 + residual` from that representation.

Then mirror the idea in backward:

```text
pw2_bwd -> gelu_bwd -> pw1_bwd
```

The goal is not one giant shader. The goal is to reduce materialization and
improve locality at the block boundary while preserving real dispatch
synchronization between phases.

Possible variants:

- store hidden in f16 but compute both pointwise matmuls in f32;
- store hidden in a layout optimized for the following contraction;
- recompute `pw1` only for carefully chosen tiny late shapes where extra math
  is cheaper than memory traffic.

This needs strict gates because it can easily increase math or hurt gradient
quality. But it is the most direct path toward bigger-than-epilogue fusion
without pretending WebGPU has cross-workgroup barriers.

## Practical Recommendation

Do not mangle the current pointwise emitter in place. The current kernel is a
known-correct baseline and already has decent tiling.

The next useful pointwise work should be:

1. Snapshot a fork under `experiments/clip_forks/vNN_*`.
2. Add one gated variant emitter.
3. Target a small allowlist of timestamp-hot shapes.
4. Measure:
   - per-dispatch timestamp rows;
   - grouped pointwise timestamp sum;
   - `batch_major_train_bench.ts` gradient parity and wall time;
   - integrated `tools/splat3d/step_matrix.ts`.
5. Promote only if the same-session full-chain and integrated timings improve.

The plausible 2x wall-clock path is still stacked:

```text
N-of-K / grid scheduling
+ pointwise-family tile/layout wins
+ spatial backward variants
+ selective precision or proxy cadence
```

A single pointwise shader change probably will not make CLIP 4x faster by
itself, but pointwise is large enough that real wins here compound with the
view-scheduling wins.
