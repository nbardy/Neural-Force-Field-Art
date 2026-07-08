# v19 Agent Note: CLIP Pointwise Math, Layout, And Fusion

Date: 2026-07-08

Scope: documentation-only inspection. I did not edit runtime source code. The
worktree already had modified CLIP and benchmark files, so this note describes
the current files as read and the v17 artifacts as recorded.

Files inspected:

- `src/clip/vision_wgsl.ts`
- `src/clip/vision_batch_wgsl.ts`
- `src/clip/vision_bwd_wgsl.ts` for the `pw_bwd` counterpart referenced by v17
- `tools/clip/pointwise_report.ts`
- `experiments/clip_forks/v17_pointwise_roofline/README.md`
- `experiments/clip_forks/v17_pointwise_roofline/results/2026-07-08/pointwise_report_b3.md`
- `experiments/clip_forks/v17_pointwise_roofline/results/2026-07-08/dispatch_profile_b3_group_sums.txt`
- `experiments/clip_forks/v17_pointwise_roofline/results/2026-07-08/dispatch_profile_b3_timestamp.csv`
- `agent_notes/optimization_session/pointwise_roofline_v17_2026_07_08.md`
- Relevant docs sections in `docs/CLIP_BATCHING_NOTES.md`, `docs/SPLAT3D_PERF_NOTES.md`, and `docs/clip_backward_spec.md`

I also reran the CPU-only report without `OUT`:

```bash
BATCH=3 TOP=14 bun tools/clip/pointwise_report.ts
```

The current plan still matches the v17 static counts.

## Short Answer

In this CLIP image tower, "pointwise" means a MobileCLIP `1x1`, `groups=1`
convolution. It is not an elementwise op and it is not depthwise spatial
filtering. It is dense channel mixing independently at every pixel/token
position:

```text
P = H * W
Z[co, p] = bias[co] + sum_{ci=0..Cin-1} X[ci, p] * W[ci, co]
```

The WGSL treats this as a tiled matmul:

```text
[P, Cin] x [Cin, Cout] -> [P, Cout]
```

but stores activations channel-planar, so the actual buffer indexing is:

```text
src[ci * P4 + p4] = vec4f(X[ci, 4*p4 + 0],
                          X[ci, 4*p4 + 1],
                          X[ci, 4*p4 + 2],
                          X[ci, 4*p4 + 3])
dst[co * P4 + p4] = vec4f(Y[co, 4*p4 + 0..3])
P4 = P / 4
```

It is a bottleneck because the train graph has 48 forward pointwise dispatches
and 48 backward `pw_bwd` dispatches. At `BATCH=3`, v17 reports 26.575 GFLOP of
pointwise train math, 3445.13 MiB of lower-bound staged traffic, and 51,648
pointwise workgroups. The same v17 timestamp profile attributes 67.31 ms, or
51.6% of the B=3 CLIP train timestamp sum, to the pointwise family
(`pw_bwd + pw + pw+gelu`).

Pointwise+GELU fusion helps because it is the exact local producer/consumer
case: the pointwise kernel already has `Z` in registers, and train mode needs to
save both the pre-GELU slot and the GELU output slot. `pointwiseFusedGelu()`
writes both in one dispatch, avoiding a separate GELU dispatch and avoiding a
full read of the pre-GELU tensor.

## Function Map

Forward pointwise code lives in `src/clip/vision_wgsl.ts`:

- `pointwiseTiledMain()` is the default shared tile body.
- `pointwise()` emits normal `pw Cin->Cout @HxW` dispatches.
- `pointwiseFusedGelu()` emits train-mode `pw+gelu` dispatches.
- `pointwiseRect8x16Main()` is the newer rectangular tile body.
- `pointwiseRect8x16()` emits `pw rect8x16 ...`.
- `pointwiseRect8x16FusedGelu()` emits `pw+gelu rect8x16 ...`.
- `assertPointwiseTiles()` enforces `P % 32 == 0`, `Cin % 32 == 0`,
  `Cout % 32 == 0`, and 16-byte weight alignment.
- `assertPointwiseRect8x16()` adds `Cout % 64 == 0`.
- `stepDispatches()` routes `ConvStep.variant === "pointwise"` through
  `convDispatches()` to the pointwise emitter.

Batch wrapping lives in `src/clip/vision_batch_wgsl.ts`:

- `batchBindings()` finds slot/text storage-buffer bindings and their element
  type (`array<f32>` or `array<vec4f>`).
- `addWorkgroupIdIfNeeded()` injects `@builtin(workgroup_id)` if a shader did
  not already declare it.
- `addBatchOffsets()` rewrites slot/text buffer accesses to add a per-lane base.
- `forwardDispatches()` optionally replaces adjacent pointwise+GELU pairs with
  `pointwiseFusedGelu()` and then wraps the spec for batch.
- `batchSpec()` changes `workgroups.z` from `1` to `batch`.
- `batchTrainDispatches()` concatenates batch-aware forward specs and backward
  specs.

Backward pointwise code lives in `src/clip/vision_bwd_wgsl.ts`:

- `pwBwd()` computes `dX = W^T dY` using `pointwiseTiledMain()`.
- `pwBwdAfterGelu()` fuses adjacent `gelu_bwd + pw_bwd` by loading
  `dY * geluGrad(pre)` into the pointwise tile.
- `pwBwdWithResidualCopy()` fuses a residual-gradient copy into selected
  `pw_bwd` cases.
- `planBwdDispatches()` applies those backward fusions when their options are
  enabled.

Static reporting lives in `tools/clip/pointwise_report.ts`:

- `forwardRows()` counts forward `ConvStep.variant === "pointwise"` rows.
- `backwardRows()` counts `kind === "pw_bwd"` rows.
- `workgroupsFor(p, cout)` models default tiles as `(p / 32) * (cout / 32)`.
- `stagedBytesFor(p, cin, cout, writes)` models default lower-bound traffic as
  staged activation reads plus staged weight reads plus output writes.
- `render()` produces the markdown tables used by the v17 artifacts.

## Forward Math

For a pointwise step `s: ConvStep`:

```text
P  = s.outH * s.outW
P4 = P / 4
Cin  = s.cin
Cout = s.cout
```

The unfused pre-activation is:

```text
Z[co, p] = W(s.bOff + co) +
           sum_{ci=0..Cin-1} X[ci, p] * Wpoint[ci, co]
```

with packed weight indexing:

```text
Wpoint[ci, co] = W(s.wOff + ci * Cout + co)
W4((s.wOff + ci * Cout + co4) / 4) = Wpoint[ci, co4..co4+3]
```

The normal epilogue in `pointwise()` is:

```text
A[co, p] = Z[co, p]                         if s.act == "none"
A[co, p] = gelu(Z[co, p])                   if s.act == "gelu"

Y[co, p] = A[co, p]                         if no residual
Y[co, p] = residual[co, p] +
           layerScale[co] * A[co, p]        if residual/layer scale exists
```

The exact forward GELU helper is:

```text
gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

`GELU` implements this as `gelu1()` and `gelu4()` using the same
Abramowitz-Stegun erf approximation used by backward.

Train-mode pointwise+GELU fusion is different from inference `act: "gelu"`.
The train graph needs the pre-activation for backward, so `pointwiseFusedGelu()`
does this:

```text
dst[co, p]     = Z[co, p]        // saved pre-GELU slot
geluDst[co, p] = gelu(Z[co, p])  // value consumed by later layers
```

The code implements that by passing to `pointwiseTiledMain()`:

```text
store(j)      = accj
extraStore(j) = geluDst[(co + j) * P4 + p4] = gelu4(accj)
```

`pointwiseFusedGelu()` only accepts exact split train-mode pairs:

```text
s.variant == "pointwise"
s.act == "none"
s.residual == null
s.layerScaleOff == null
gelu.src == s.dst
gelu.n == s.cout * P
```

## Backward Math

CLIP weights are frozen in this training path. There is no `dW` pointwise
kernel. Backward only propagates image-gradient signal through the fixed image
encoder.

For a forward pointwise:

```text
Z[co, p] = bias[co] + sum_ci X[ci, p] * W[ci, co]
```

the input-gradient math is:

```text
dX[ci, p] = sum_{co=0..Cout-1} dZ[co, p] * W[ci, co]
```

In `PwBwdStep`, names are deliberately in tile-kernel order, not original
forward-layer order:

```text
pw.cin  = reduction channels = forward Cout = channels of dY/dZ
pw.cout = output channels    = forward Cin  = channels of dX
```

`compile_plan.py --train` packs a second weight orientation for pointwise
backward. The offset is `wOffT`, and it is laid out as:

```text
Wbwd[reduction, output] = Wforward[output, reduction]
scalar index = wOffT + red * pw.cout + out
```

That lets `pwBwd()` call the exact same `pointwiseTiledMain()` body:

```text
src = dY or dZ, laid out as [pw.cin][P4]
dst = dX, laid out as [pw.cout][P4]
weights = Wbwd, laid out as [pw.cin][pw.cout]
```

Backward GELU math is:

```text
gelu'(x) = Phi(x) + x * phi(x)
Phi(x)  = 0.5 * (1 + erf(x / sqrt(2)))
phi(x)  = exp(-0.5 * x * x) / sqrt(2*pi)
```

`GELU_GRAD` implements `geluGrad4(x)`:

```text
cdf = 0.5 * (1 + erf4(x / sqrt(2)))
pdf = 0.3989422804014327 * exp(-0.5 * x * x)
geluGrad4(x) = cdf + x * pdf
```

`pwBwdAfterGelu()` fuses this into the pointwise load path:

```text
effectiveSrc[index] = src[index] * geluGrad4(pre[index])
```

and then computes the same tiled matmul. That removes the intermediate
`gelu_bwd` gradient slot write/read for exact adjacent pairs.

## Storage Buffers And Layout

The `DispatchSpec` storage buffer contract is:

```text
binding i in WGSL == spec.buffers[i]
```

`BufferRef` can be:

```text
{ kind: "weights" }
{ kind: "slot", slot: number }
{ kind: "text" }
```

### Activation Slots

The plan contract is NCHW planar f32:

```text
activation scalar index = c * H * W + y * W + x
```

Pointwise kernels bind those same bytes as `array<vec4f>`:

```wgsl
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;
```

The implicit shape is:

```text
src[ci * P4 + p4] = four adjacent spatial positions for one channel
dst[co * P4 + p4] = four adjacent spatial positions for one output channel
```

The tile guard requires `P % 32 == 0`, so `P4` is divisible by 8 and every
workgroup owns an exact 8-pixel-quad tile.

### Weights

Weights, bias, and layer scale live in one packed weights buffer. In f32 mode:

```wgsl
@group(0) @binding(0) var<storage, read> weights : array<vec4f>;
fn W(i : u32) -> f32 { return weights[i >> 2u][i & 3u]; }
fn W4(i : u32) -> vec4f { return weights[i]; }
```

In f16 weight-storage mode:

```wgsl
enable f16;
@group(0) @binding(0) var<storage, read> weights : array<vec4<f16>>;
fn W(i : u32) -> f32 { return f32(weights[i >> 2u][i & 3u]); }
fn W4(i : u32) -> vec4f { return vec4f(weights[i]); }
```

The important point: f16 here is storage only. Arithmetic and activations remain
f32 in these pointwise kernels.

### Forward Pointwise Bindings

Normal `pointwise()` dispatch:

```text
binding 0: weights, read
binding 1: src slot, read, array<vec4f>
binding 2: dst slot, read_write, array<vec4f>
binding 3: residual slot, read, array<vec4f>       // only when residual exists
```

`pointwiseFusedGelu()` dispatch:

```text
binding 0: weights, read
binding 1: src slot, read
binding 2: dst pre-activation slot, read_write
binding 3: geluDst slot, read_write
```

### Backward Pointwise Bindings

`pwBwd()` dispatch:

```text
binding 0: weights, read
binding 1: grad dY slot, read, array<vec4f>
binding 2: grad dX slot, read_write, array<vec4f>
```

`pwBwdAfterGelu()` dispatch:

```text
binding 0: weights, read
binding 1: grad dY before GELU derivative, read
binding 2: saved pre-GELU activation, read
binding 3: grad dX slot, read_write
```

`pwBwdWithResidualCopy()` dispatch:

```text
binding 0: weights, read
binding 1: grad dY slot, read
binding 2: grad dX slot, read_write
binding 3: residual grad slot, read_write
```

### Batch-Major Offsets

`vision_batch_wgsl.ts` allocates activation slots as:

```text
[batch][slotFloats]
```

It does not duplicate weights. Weights remain shared and un-offset.

For every slot/text binding, `addBatchOffsets()` injects:

```text
let batchLane = workgroup_id.z;
let batchBase_src = batchLane * stride;
```

where:

```text
stride = slotFloats      for array<f32>
stride = slotFloats / 4  for array<vec4f>
```

Then it rewrites every `src[` or `dst[` access to:

```text
src[batchBase_src + originalIndex]
dst[batchBase_dst + originalIndex]
```

The default batch-major pointwise grid is therefore:

```text
workgroups = [P4 / 8, Cout / 32, batch]
```

Each batch lane stages the same weights independently. The imported
`pointwiseSharedWBatchForwardDispatch()` is the separate shared-W experiment:
it can stage one W tile across multiple batch lanes for selected forward steps,
but the normal z-batch path does not do that.

## Default Workgroup Tiling

`pointwiseTiledMain()` emits:

```wgsl
@compute @workgroup_size(8, 8)
```

Thread and tile coordinates:

```text
p4     = wid.x * 8 + lid.x
co     = (wid.y * 8 + lid.y) * 4
p4base = wid.x * 8
cobase = wid.y * 32
```

Each workgroup owns:

```text
8 pixel-quads x 32 output channels
= 32 scalar spatial positions x 32 output channels
```

Each thread owns:

```text
1 pixel-quad x 4 output channels
= 4 scalar pixels x 4 output channels
= 16 scalar outputs
```

Private accumulators:

```text
acc0 = output channel co + 0 for four pixels
acc1 = output channel co + 1 for four pixels
acc2 = output channel co + 2 for four pixels
acc3 = output channel co + 3 for four pixels
```

Workgroup memory:

```text
xS: 256 vec4f = 4096 bytes  // 32 reduction channels x 8 pixel-quads
wS: 256 vec4f = 4096 bytes  // 32 reduction channels x 8 cout-quads
total: 8192 bytes
```

The reduction loop is:

```text
for ci0 = 0; ci0 < Cin; ci0 += 32:
  for t = local_invocation_index; t < 256; t += 64:
    ci   = t >> 3
    lane = t & 7

    srcIndex = (ci0 + ci) * P4 + p4base + lane
    xS[t] = src[srcIndex]

    wS[t] = W4((wOff + (ci0 + ci) * Cout + cobase + lane * 4) / 4)

  workgroupBarrier()

  for ci = 0; ci < 32; ci += 1:
    xv = xS[ci * 8 + lid.x]   // four pixels
    wv = wS[ci * 8 + lid.y]   // four output channels

    acc0 = fma(vec4f(wv.x), xv, acc0)
    acc1 = fma(vec4f(wv.y), xv, acc1)
    acc2 = fma(vec4f(wv.z), xv, acc2)
    acc3 = fma(vec4f(wv.w), xv, acc3)

  workgroupBarrier()
```

Stores:

```text
dst[co * P4 + p4]        = store(0)
dst[(co + 1) * P4 + p4]  = store(1)
dst[(co + 2) * P4 + p4]  = store(2)
dst[(co + 3) * P4 + p4]  = store(3)
```

Grid:

```text
single image: [P4 / 8, Cout / 32, 1]
batch-major: [P4 / 8, Cout / 32, batch]
```

For example, `256->768 @16x16` has:

```text
P = 256
P4 = 64
workgroups per image = (64 / 8) * (768 / 32) = 8 * 24 = 192
workgroups at B=3 = 576
MACs per image = 256 * 256 * 768 = 50,331,648
FLOPs per image = 100,663,296
```

The v17 top-shape table has 10 of these forward expansion layers, so at B=3
they account for:

```text
10 * 3 * 100,663,296 FLOPs = 3.020 GFLOP
```

## Rect8x16 Tiling In Current Source

The current `vision_wgsl.ts` also has `PointwiseTileVariant = "rect8x16"`.
When `pointwiseTileVariant` is set and the optional step filter allows the
step, `useRect8x16Pointwise()` routes to `pointwiseRect8x16()` or
`pointwiseRect8x16FusedGelu()`.

`pointwiseRect8x16Main()` emits:

```wgsl
@compute @workgroup_size(8, 16)
```

Thread and tile coordinates:

```text
p4     = wid.x * 8 + lid.x
co     = (wid.y * 16 + lid.y) * 4
p4base = wid.x * 8
cobase = wid.y * 64
```

Each workgroup owns:

```text
8 pixel-quads x 64 output channels
= 32 scalar spatial positions x 64 output channels
```

Workgroup memory:

```text
xS: 256 vec4f = 4096 bytes
wS: 512 vec4f = 8192 bytes
total: 12288 bytes
```

Grid:

```text
[P4 / 8, Cout / 64, 1]
```

This tile reuses the same staged X tile across twice as many output channels.
Its compulsory X traffic per output is lower than the default tile, while W
traffic per MAC is similar. The tradeoff is a larger 128-thread workgroup and
larger workgroup-memory footprint. The v17 static report models the default
tile via `workgroupsFor(p, cout) = (p / 32) * (cout / 32)`; it does not model
rect8x16 separately.

## Static Traffic Model

`tools/clip/pointwise_report.ts` uses:

```text
macs  = P * Cin * Cout
flops = 2 * macs
```

Default workgroups:

```text
workgroups = (P / 32) * (Cout / 32)
```

Lower-bound staged bytes:

```text
xReadFloats = macs / 32
wReadFloats = macs / 32
writeFloats = P * Cout * writes
bytes = 4 * (xReadFloats + wReadFloats + writeFloats)
```

Why `/32`? In the default tile, each staged activation scalar is reused across
32 output channels, and each staged weight scalar is reused across 32 pixels.

`writes` is:

```text
1 for ordinary pointwise output
2 for forward pointwise+GELU fusion because train mode writes:
  - saved pre-GELU activation
  - GELU output
```

The traffic number is explicitly a lower-bound model. It does not include:

- cache misses or cache effects;
- residual reads;
- layer-scale reads;
- `dst` reads for backward `accumulate:true`;
- command/dispatch overhead;
- non-pointwise kernels;
- rect8x16's different X/W traffic balance.

## v17 Evidence

The v17 CPU report for `plan_train.json`, `3x256x256`, `BATCH=3` says:

```text
forward pointwise dispatches:       48
backward pw_bwd dispatches:         48
forward pointwise+GELU candidates:  24
pointwise FLOPs:                    8.858 GFLOP per image
pointwise FLOPs at B=3:             26.575 GFLOP
approx staged traffic at B=3:       3445.13 MiB
pointwise workgroups at B=3:        51648
```

Top static shapes by B=3 FLOPs:

| Phase | Shape | Count | FLOPs B3 | Traffic B3 | Flags |
| --- | --- | ---: | ---: | ---: | --- |
| forward | `256->768 @16x16` | 10 | 3.020G | 405.00MiB | `10 gelu-fuse` |
| forward | `768->256 @16x16` | 10 | 3.020G | 367.50MiB | `10 residual` |
| backward | `256->768 @16x16` | 10 | 3.020G | 382.50MiB | `-` |
| backward | `768->256 @16x16` | 10 | 3.020G | 367.50MiB | `-` |
| forward | `128->384 @32x32` | 6 | 1.812G | 270.00MiB | `6 gelu-fuse` |
| backward | `128->384 @32x32` | 6 | 1.812G | 243.00MiB | `-` |
| forward | `512->1536 @8x8` | 4 | 1.208G | 150.75MiB | `2 gelu-fuse` |
| backward | `1536->512 @8x8` | 4 | 1.208G | 145.50MiB | `2 accumulate` |

The v17 timestamp group sums are:

| Group | Timestamp Sum | Share |
| --- | ---: | ---: |
| `pw_bwd` | 34.3412 ms | 26.3% |
| `spatial_bwd` | 26.8697 ms | 20.6% |
| `pw` | 18.0225 ms | 13.8% |
| `conv` | 16.3835 ms | 12.6% |
| `pw+gelu` | 14.9426 ms | 11.5% |
| `gelu_bwd` | 9.5684 ms | 7.3% |
| total | 130.4165 ms | 100.0% |

Pointwise family:

```text
pw_bwd + pw + pw+gelu
= 34.3412 + 18.0225 + 14.9426
= 67.3063 ms
= 51.6% of total timestamp sum
```

That is the concrete reason v17 calls pointwise the largest CLIP shader family.

## Why Pointwise Is A Bottleneck

1. It is the dense channel-mixing part of MobileCLIP.

   The ConvFFN blocks repeatedly run:

   ```text
   pointwise expansion -> GELU -> pointwise contraction -> residual
   ```

   Those expansion/contraction layers dominate forward conv MACs.

2. Train mode pays forward and backward.

   The optimizer needs `dL/dimage`, not just an image embedding. The CLIP pass is:

   ```text
   forward image encoder
   loss backward
   backward image encoder
   ```

   That creates a backward pointwise chain roughly as large as the forward one.

3. Batch-major CLIP repeats the work per view lane.

   The normal batch path puts the lane in `workgroup_id.z`, so B=3 runs three
   copies of the same pointwise tile grid. Weights are shared as a buffer, but
   each lane still stages its W tile in workgroup memory.

4. It is tiled, but it is still WGSL f32 FMA work.

   The tile is not a hardware matrix instruction. Every workgroup loops over
   channel chunks, performs vector FMAs, and uses two barriers per 32-channel
   chunk.

5. Dispatch granularity is not one giant GEMM.

   The network has many medium-sized pointwise matmuls, each emitted as its own
   specialized dispatch. Some late low-spatial high-channel layers have fewer
   workgroups and long reduction loops, which can make scheduling tails visible.

6. Train storage is large and f32.

   Activations, saved pre-GELU tensors, gradients, and most intermediates are
   f32 storage buffers. Even when arithmetic is high, slot traffic remains a real
   cost. Backward `accumulate:true` also reads `dst` before writing it.

7. Residual and GELU edges add traffic unless fused.

   Residual/layer-scale epilogues add reads and scalar weight loads. Standalone
   GELU forward reads the pre tensor and writes the GELU output. Standalone GELU
   backward reads `dy` and `pre`, writes an intermediate gradient, and is then
   consumed by `pw_bwd`.

## Why Pointwise+GELU Fusion Helps

The promoted forward fusion handles exactly this pattern:

```text
pointwise conv with act none -> standalone train-mode GELU
```

The unfused train path does:

```text
pointwise dispatch:
  compute Z
  write pre slot

gelu dispatch:
  read pre slot
  compute gelu(pre)
  write gelu output slot
```

The fused path does:

```text
pointwise dispatch:
  compute Z
  write pre slot
  write gelu(Z) output slot
```

It saves:

- one dispatch per fused pair;
- one full read of the pre-GELU tensor;
- the separate elementwise kernel's scheduling overhead;
- some command-buffer/profile noise from many tiny activation dispatches.

It does not save:

- the pointwise matmul arithmetic;
- the pre-GELU write, because backward still needs it;
- the GELU output write, because later forward layers still consume it;
- backward pointwise work.

The docs record 24 exact pointwise+GELU forward pairs. The promoted B=3 numbers:

```text
dispatch count:           281 -> 257
B=3 CLIP train median:    73.33 ms -> 68.06 ms
integrated 3D CLIP median: 80.68 ms -> 76.38 ms
```

`vision_batch_wgsl.ts` enables this through `fusePointwiseGeluForward`. In
`forwardDispatches()`, it checks the current `conv pointwise` step and the next
`gelu` step, emits `pointwiseFusedGelu()`, then increments the step index so the
standalone GELU dispatch is skipped.

The same idea exists as a gated backward ablation:

```text
standalone gelu_bwd -> pw_bwd
```

`pwBwdAfterGelu()` loads:

```text
src[index] * geluGrad4(pre[index])
```

directly into the pointwise tile. The isolated B=3 CLIP median improved:

```text
default forward GELU fusion: 68.22 ms
+ GELU backward fusion:      61.70 ms
```

but the integrated 3D matrix did not clear the promotion bar:

```text
default:                 normal 78.30 ms, profile 93.50 ms, CLIP 67.51 ms
FUSE_GELU_BWD_PW=1:      normal 76.83 ms, profile 94.71 ms, CLIP 67.02 ms
```

So forward pointwise+GELU is promoted; backward GELU fusion remains a gated
ablation.

## Why This Does Not Mean "Fuse All CLIP"

Pointwise+GELU is a local fusion where the producer value is already in
registers and the consumer is immediate. The full CLIP graph is not like that.

The graph has whole-tensor dependencies between layers. WebGPU barriers are
workgroup-local, not grid-global. Attention, SE, spatial convolution,
pointwise matmul, GELU, head, and loss kernels use different workgroup
geometries and memory layouts. Train mode also needs saved activations for
backward. A monolithic shader would likely increase register/private-memory
pressure and lose occupancy.

The correct boundary is measured local fusion: fuse adjacent operations when it
removes real dispatches or memory traffic without changing the dependency
surface.

## Practical Takeaways

- "Pointwise" in this repo means `ConvStep.variant === "pointwise"`: 1x1 dense
  channel mixing.
- The core operation is `Z[co,p] = bias[co] + sum_ci X[ci,p] * W[ci,co]`.
- The core emitter is `pointwiseTiledMain()`.
- The default tile is `8 x 8` threads, `32 pixels x 32 output channels`, with
  `8192` bytes of workgroup memory.
- Current source also has `rect8x16`: `8 x 16` threads,
  `32 pixels x 64 output channels`, with `12288` bytes of workgroup memory.
- Activations and gradients are channel-planar f32, viewed as `array<vec4f>` in
  pointwise kernels.
- Forward weights are packed as `[Cin][Cout]`; backward uses `wOffT` packed as
  `[forward Cout][forward Cin]` so the same tile body computes `dX`.
- Batch-major wrapping offsets slot/text bindings by `workgroup_id.z`; weights
  are not offset.
- v17 measured the pointwise family at 51.6% of B=3 CLIP train timestamp sum.
- Forward pointwise+GELU fusion helps because it removes 24 standalone GELU
  dispatches and one full pre-GELU tensor read while preserving the saved
  pre-activation required by backward.
