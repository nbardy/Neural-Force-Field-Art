# Agent 2 - CLIP Pointwise Bottleneck Analysis

Date: 2026-07-08

Scope: explain the CLIP pointwise bottleneck from current code. This note is
read-only analysis. It does not edit runtime shader code.

Inspected:

- `src/clip/vision_wgsl.ts`
- `src/clip/vision_bwd_wgsl.ts`
- `src/clip/vision_batch_wgsl.ts`
- `src/clip/vision_batch_pointwise.ts`
- `tools/clip/dispatch_profile.ts`
- `tools/clip/README.md`
- `docs/CLIP_BATCHING_NOTES.md`
- `docs/SPLAT3D_PERF_NOTES.md`
- `agent_notes/optimization_session/clip_timestamp_dispatch_profile.md`

## Short Answer

"Pointwise" here means MobileCLIP 1x1 convolution. At each spatial position, it
is a dense matrix multiply over channels:

```text
y[co, p] = bias[co] + sum_ci x[ci, p] * W[ci, co]
```

The current shader tiles this as `[P, Cin] x [Cin, Cout]`, where `P = H * W`.
It is already a real tiled WGSL matmul, not a naive scalar loop, but it is still
the biggest CLIP family because the train graph has 48 forward pointwise
dispatches and 48 backward pointwise dispatches. In `plan_train.json`, forward
pointwise is about `4.43 GFLOP` per image, and pointwise backward is another
`4.43 GFLOP` per image. Batch 3 makes that roughly `26.6 GFLOP` of pointwise
math per CLIP train call before counting memory traffic and dispatch overhead.

The timestamp profile on promoted B=3 settings reported:

| Group | Isolated GPU Time | Share |
| --- | ---: | ---: |
| `pw_bwd` | `42.01 ms` | `26.2%` |
| `pw+gelu` | `23.92 ms` | `14.9%` |
| `pw` | `23.20 ms` | `14.5%` |

So the pointwise family was about `55.6%` of isolated CLIP GPU time on that
profile. `spatial_bwd` and `conv` are also large, but pointwise is the largest
combined family.

I do not see a promoted fp16 pointwise path in the inspected source. Current
pointwise code uses `array<vec4f>`, f32 activation slots, f32 weights, and f32
accumulators.

## 1. What Pointwise Means Mathematically

Forward pointwise is a 1x1 convolution. Since the kernel is 1x1 and groups are
1, every output channel at a spatial position is just a dot product across input
channels.

For a shape `Cin -> Cout @ HxW`:

```text
P = H * W
pre[co, p] = bias[co] + sum_ci src[ci, p] * W[ci, co]
```

Then the epilogue may apply:

- no activation;
- GELU;
- residual plus layer scale:

```text
dst[co, p] = residual[co, p] + layerScale[co] * activation(pre[co, p])
```

Train-mode often splits GELU into its own activation slot so backward can read
the saved pre-activation. The promoted `pointwiseFusedGelu` forward fusion is
still exact for train mode because it writes both:

- the pointwise pre-activation slot;
- the GELU output slot.

Backward pointwise computes the gradient with frozen weights. There is no `dW`
path:

```text
dX[ci, p] = sum_co W[ci, co] * dY[co, p]
```

In the generated backward step, names are flipped:

- `PwBwdStep.cin` is the reduction dimension, equal to forward `Cout`;
- `PwBwdStep.cout` is the output dimension, equal to forward `Cin`;
- `wOffT` points to a transposed weight copy laid out for the same tiled kernel.

When layer scale exists, it is folded into the transposed backward weights by the
offline compiler, so `pw_bwd` still only reads one weight matrix.

## 2. Exact Memory Layout

### Single-image activation layout

The plan contract in `vision_wgsl.ts` says activations are NCHW planar f32:

```text
x[c][y][x] at c * H * W + y * W + x
```

The pointwise kernels bind those slots as `array<vec4f>`, so spatial positions
are packed as pixel quads:

```text
P4 = P / 4
src[ci * P4 + p4] = vec4f(
  src[ci, 4*p4 + 0],
  src[ci, 4*p4 + 1],
  src[ci, 4*p4 + 2],
  src[ci, 4*p4 + 3]
)
```

Destination and gradient slots use the same channel-major, pixel-quad layout:

```text
dst[co * P4 + p4]
grad[channel * P4 + p4]
```

The tile guard requires:

```text
P % 32 == 0
Cin % 32 == 0
Cout % 32 == 0
wOff % 4 == 0
```

### Weight layout

Weights are bound as:

```wgsl
@group(0) @binding(0) var<storage, read> weights : array<vec4f>;
```

Forward pointwise weights are stored transposed as `[Cin][Cout]`, with four
consecutive output channels packed in one `vec4f`:

```text
W scalar index = wOff + ci * Cout + co
W4((wOff + ci * Cout + co4) / 4) gives co4..co4+3
```

Bias and layer scale are also read from the shared weights blob:

```text
bias[co]       = W(bOff + co)
layerScale[co] = W(layerScaleOff + co)
```

Backward pointwise uses a separate transposed offset `wOffT` emitted by the
compiler. The shader calls the same tiled body, so the backward weight buffer is
also shaped as `[reduction][output]` for contiguous output-channel quads.

### Batch-major layout

The batch-major CLIP fork allocates every activation slot as:

```text
[batch][slotFloats]
```

For `array<vec4f>` slot bindings, the generated base offset is:

```text
batchBase = batchLane * (slotFloats / 4)
```

The normal batch path uses `workgroup_id.z` as the lane and runs one copy of the
same pointwise workgroup per image lane:

```text
workgroups = [P4 / 8, Cout / 32, batch]
```

The shared-W experiment instead puts batch lanes inside `local_invocation_id.z`
and uses:

```text
workgroups = [P4 / 8, Cout / 32, 1]
workgroup_size = (8, 8, batch)
```

This stages one W tile for all lanes while staging a private X tile per lane.

## 3. Current Workgroup Tiling

The core pointwise tile is in `pointwiseTiledMain()`.

Current baseline:

```text
workgroup_size = 8 x 8 = 64 threads
workgroups     = [P4 / 8, Cout / 32, 1]
```

Each workgroup owns:

```text
8 pixel-quads x 32 output channels
= 32 pixels x 32 output channels
```

Each thread computes:

```text
4 output channels x 4 pixels = 16 scalar outputs
```

The thread state is:

```text
p4 = wid.x * 8 + lid.x
co = (wid.y * 8 + lid.y) * 4
acc0..acc3 : vec4f
```

The reduction loops over `Cin` in chunks of 32:

```text
for ci0 in 0..Cin step 32:
  stage xS[32 ci][8 pixel-quads]
  stage wS[32 ci][8 output-channel quads]
  barrier
  for ci in 0..31:
    xv = xS[ci, lid.x]      // 4 pixels
    wv = wS[ci, lid.y]      // 4 output channels
    acc0 += wv.x * xv
    acc1 += wv.y * xv
    acc2 += wv.z * xv
    acc3 += wv.w * xv
  barrier
```

Workgroup memory:

```text
xS: 256 vec4f = 4096 bytes
wS: 256 vec4f = 4096 bytes
total baseline = 8192 bytes
```

Shared-W batch memory:

```text
xS: 256 * B vec4f
wS: 256 vec4f
B=2 total = 12288 bytes
B=3 total = 16384 bytes
```

B=3 exactly hits a 16 KiB workgroup-memory footprint for these two arrays before
counting implementation-specific overhead.

## 4. Why `pw` and `pw_bwd` Dominate

### There are many pointwise layers

`plan_train.json` contains:

```text
forward pointwise conv steps: 48
backward pw_bwd steps:       48
```

By shape, forward pointwise steps are:

```text
64->64@64x64:       1
64->192@64x64:      2
192->64@64x64:      2
128->128@32x32:     1
128->384@32x32:     6
384->128@32x32:     6
256->256@16x16:     1
256->768@16x16:     10
768->256@16x16:     10
512->512@8x8:       3
512->1536@8x8:      4
1536->512@8x8:      2
```

Backward has the same total count with the reduction/output dimensions flipped.

Many of these ConvFFN expansion/contraction pairs are intentionally balanced:
each `64->192@64x64`, `128->384@32x32`, `256->768@16x16`, or
`512->1536@8x8` pointwise step is about `0.1007 GFLOP` per image.

### The train path pays forward and backward

The optimizer does not only need a CLIP score. It needs `dLoss/dImage`, so it
runs:

```text
forward CLIP image encoder
loss backward
backward image encoder
```

That doubles pointwise matmul pressure. With B=3, the normal batch-major path
also repeats the pointwise work for three rendered views in one CLIP train call.

### The kernel is tiled, but still not hardware GEMM

This is a good hand-written tiled shader, but it is still plain WGSL f32 math.
It does not use hardware matrix/tensor instructions. Every thread executes a
loop of scalar/vector FMAs over the channel dimension, and every pointwise shape
is a separate specialized dispatch.

The pointwise family therefore costs:

- real arithmetic;
- repeated dispatches;
- repeated global activation reads and output writes;
- repeated staged weight reads for each spatial/output tile;
- extra reads/writes when residual accumulation or train-mode GELU slots are
  present.

### Backward has extra traffic edges

`pw_bwd` is the same size matmul as forward. Some steps also accumulate into an
existing gradient slot:

```text
dst = dst + acc
```

That adds a read of `dst` before the write. GELU backward can add another
activation read and gradient write unless fused into the following `pw_bwd`.
The gated `FUSE_GELU_BWD_PW=1` path removes some of that traffic, but it did not
clear the integrated 3D promotion gate.

## 5. Why Prior Shared-W Attempts Failed Full-Chain

The shared-W idea was valid: when B=2 or B=3 views share the same CLIP weights,
one workgroup can stage the W tile once and give each batch lane its own X tile.
The compact tests verified exact output parity.

The problem is that it attacks only one part of the cost.

What it saves:

- repeated global W tile loads across batch lanes for selected forward
  pointwise steps.

What it does not save:

- arithmetic;
- X activation reads;
- destination writes;
- residual reads;
- GELU/pre-activation slot writes;
- backward pointwise time;
- spatial backward time;
- conv-family time;
- attention or SE time.

It also makes the workgroup heavier:

- B=3 raises the workgroup to `8 x 8 x 3 = 192` invocations;
- B=3 uses `16 KiB` workgroup memory for `xS + wS`;
- larger workgroups and memory footprint can reduce occupancy or scheduling
  flexibility on Apple Metal;
- all lanes participate in the same barriers;
- contraction/residual shapes do not necessarily benefit from W reuse enough to
  offset the occupancy pressure.

Recorded evidence:

- Compact shared-W microbench was shape-specific. Some expansion layers won,
  many B=3 rows were flat/loss, and contraction/residual layers were mixed.
- The production selective forward path with allowlists did not survive
  full-chain timing:

```text
B=3 base:       75.13 ms
B=3 early:      76.36 ms  (steps 8,10)
B=3 candidates: 76.34 ms  (steps 8,10,111,115)
B=2 base:       41.52 ms
B=2 b2wins:     41.66 ms  (steps 8,57)
```

The lesson is not "shared-W was dumb." It is "W reuse across batch lanes is too
small and too occupancy-sensitive as a global default." Any next pointwise
rewrite needs a full-chain B=3 gate, not only compact microbench parity.

## 6. Two Bigger Rewrite Candidates

These are intentionally framed as forked variants. Do not mangle the default
pointwise emitter first. Copy or add variant emitters, gate them, measure them,
then promote only if both correctness and integrated timing pass.

### Candidate A - Wider pointwise tiles that reuse W or X inside one image

The shared-W attempt reused W across batch lanes. A different attack is to reuse
W or X across more output work inside one image, without putting batch lanes in
the local z dimension.

Two variants worth trying:

1. `dual-pixel` tile:

```text
current:    8 pixel-quads x 32 cout
candidate: 16 pixel-quads x 32 cout
```

Each thread computes two pixel-quads for the same output channels. This stages
one W tile and two X tiles:

```text
xS: 512 vec4f = 8192 bytes
wS: 256 vec4f = 4096 bytes
total = 12288 bytes
```

Potential win: halves W tile reloads over the spatial dimension and reduces
workgroup count in `x`, which should matter for high-P layers like `64x64` and
`32x32`.

Risk: twice the accumulators per thread, more register pressure, lower
occupancy, and less benefit on tiny `8x8` layers.

2. `dual-cout` tile:

```text
current:    8 pixel-quads x 32 cout
candidate: 8 pixel-quads x 64 cout
```

Each thread computes eight output channels for one pixel-quad. This stages one
X tile and two W tiles:

```text
xS: 256 vec4f = 4096 bytes
wS: 512 vec4f = 8192 bytes
total = 12288 bytes
```

Potential win: halves X tile reloads over the output-channel dimension and
reduces workgroup count in `y`, likely useful on expansion/contraction layers.

Risk: many accumulators per thread, more registers, possible spill, and weaker
occupancy.

Implementation trail:

- create a new variant emitter instead of changing `pointwiseTiledMain`;
- suggested names: `pointwiseTiledDualPixelMain` and
  `pointwiseTiledDualCoutMain`;
- gate by explicit allowlist, for example `PW_TILE_VARIANT=dual_p,dual_co` plus
  `PW_TILE_STEPS=...`;
- start with top pointwise shapes from the profile, not all 48 steps.

Correctness gates:

- generated WGSL validation for every selected step;
- compare variant against baseline for deterministic random buffers with
  `relLinf <= 1e-5` or an exact/near-exact tolerance appropriate for f32 FMA
  order changes;
- run `bun tools/clip/bwd_test.ts`;
- run B=3 `batch_major_train_bench.ts` with gradient cosine checks.

Performance gates:

- isolated timestamp: selected pointwise dispatches must improve at least
  `1.20x` median on same-session trials;
- pointwise group sum in `TIMESTAMP=1 ... dispatch_profile.ts` must improve by
  at least `10%`;
- B=3 CLIP train median must improve by at least `8%`;
- integrated `CLIP_BATCH=3 VIEWS=3` step median must improve by at least `5%`
  with no regression in raster or optimizer bookkeeping.

Kill criteria:

- any variant that only wins one compact shape but loses full B=3 train timing
  stays gated;
- any variant that wins only under split-submit wall timing but not timestamp
  timing is not promoted.

### Candidate B - Mixed f16 pointwise family with f32 accumulation

The local adapter has reported `shader-f16` support in prior probing, but the
inspected pointwise source is f32. A pointwise-specific f16 fork is the cleanest
way to test whether the bottleneck is memory bandwidth, register pressure, or
f32 arithmetic throughput.

First version should not convert the whole CLIP graph. Start with pointwise
weights, because they are large and immutable:

```text
weights_f16: array<vec4h>
src/dst:     existing f32 activation slots
acc:         vec4f
```

Inside the tile:

```wgsl
let wh = W4h(...);              // vec4h
let wv = vec4f(wh);             // f32 before fma
acc = fma(vec4f(wv.x), xv, acc)
```

This tests weight-bandwidth pressure without changing activation-slot format or
the rest of the graph. If it wins, a second fork can test f16 activation slots
for pointwise-heavy blocks, still accumulating in f32.

Potential win:

- half pointwise weight bandwidth;
- smaller W staging footprint if the shader uses f16 workgroup tiles;
- possibly better cache residency across repeated pointwise shapes.

Risks:

- converting f16 to f32 in the inner loop may erase gains;
- f16 workgroup arrays may not map to the expected Metal codegen;
- numerical differences can change CLIP gradients and splat convergence even if
  single-kernel error looks small;
- a weights-only f16 path will not reduce activation traffic.

Implementation trail:

- add a separate packed f16 pointwise weights buffer or sidecar offsets;
- do not replace the existing weights blob until the gate wins;
- request `shader-f16` only when the path is enabled;
- keep f32 baseline as the fallback;
- gate by `PW_F16=1`, with optional `PW_F16_STEPS=...` allowlist.

Correctness gates:

- offline packer verifies every f16 pointwise offset maps back to the f32
  source weight index;
- single dispatch relative error versus f32 baseline recorded per selected
  shape;
- end-to-end image gradient cosine against f32 baseline, target `>= 0.999`;
- no NaNs in B=3 train for several prompts/views;
- short splat optimization smoke: loss decreases and images remain stable.

Performance gates:

- selected pointwise timestamp improves at least `1.25x`;
- full pointwise family timestamp improves at least `15%`;
- B=3 CLIP train median improves at least `10%`;
- integrated `3/9 batch x3` optimizer step improves at least `7%`.

Kill criteria:

- if f16 weights only improve isolated pointwise but not B=3 train, keep it as
  a documented ablation and do not convert activation slots;
- if image-gradient cosine falls below threshold or splat optimization gets
  visibly worse at the same resolution, reject even if it is faster.

## Practical Next Step

Do not start with "fuse everything into one huge shader." The current code has
real saved-activation requirements for train mode, and giant FFN fusion can
increase register pressure or recompute hidden activations. The next better
step is to fork one or two pointwise emitters with narrow gates:

1. `dual-pixel` or `dual-cout` f32 tile for the hottest B=3 pointwise shapes.
2. f16 pointwise weights with f32 accumulation.

Use copied/gated variant code and one commit per attempt. The default
`pointwiseTiledMain` should not change until a variant passes:

```text
local parity -> TIMESTAMP dispatch profile -> B=3 CLIP train -> integrated 3D step matrix
```

That keeps rollback easy and makes the performance trail readable.
