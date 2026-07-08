# v17 Pointwise Roofline

Date: 2026-07-08

## Goal

Make the pointwise bottleneck concrete. The question was not just "is
pointwise slow?", but what "pointwise" means, what math/layout the shader uses,
why `pw+gelu` fusion helps, why the whole graph cannot be fused into one giant
shader, and whether the earlier f16 work was a full precision breakthrough or a
weights-only experiment.

This fork adds:

```bash
tools/clip/pointwise_report.ts
```

It is CPU-only and reads `models/mobileclip_s0/plan_train.json`, so it can be
rerun without GPU contention.

## Snapshot

The `snapshot/` directory contains the CLIP WGSL emitters, batch pointwise
experiment code, current dispatch profiler, perf notes, and prior pointwise
agent notes.

Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v17_pointwise_roofline
```

## What Pointwise Means

In this MobileCLIP image tower, pointwise means a `1x1`, `groups=1`
convolution. It is a channel matrix multiply at every pixel/token position:

```text
Y[p, co] = bias[co] + sum_ci X[p, ci] * W[ci, co]
```

The current WGSL layout:

- activations are channel-planar NCHW, packed as `vec4f` pixel quads;
- source tensor is read as `src[ci][p4] = X[p4*4 + 0..3, ci]`;
- destination tensor is written as `dst[co][p4]`;
- pointwise weights are stored transposed as `[Cin][Cout]`, so `W4()` loads four
  adjacent output channels;
- backward `pw_bwd` uses a compiler-emitted `wOffT`, so the same tiled body can
  compute `dX = W^T dY`;
- CLIP weights are frozen, so there is no `dW` path.

Current tile:

```text
workgroup_size = 8 x 8 = 64 threads
tile           = 8 pixel-quads x 32 cout = 32 pixels x 32 output channels
workgroup mem  = xS 256 vec4f + wS 256 vec4f = 8192 bytes
thread output  = 4 output channels x 4 pixels
```

## Static Report

Command:

```bash
BATCH=3 TOP=14 OUT=experiments/clip_forks/v17_pointwise_roofline/results/2026-07-08/pointwise_report_b3.md bun tools/clip/pointwise_report.ts
```

Headline:

| Metric | Per Image | Batch 3 |
| --- | ---: | ---: |
| forward pointwise dispatches | 48 | 48 |
| backward `pw_bwd` dispatches | 48 | 48 |
| forward pointwise+GELU candidates | 24 | 24 |
| pointwise FLOPs | 8.858 GFLOP | 26.575 GFLOP |
| approx staged global traffic | 1148.38 MiB | 3445.13 MiB |
| pointwise workgroups | 17216 | 51648 |

The staged traffic estimate is a lower-bound model for the tiled kernels:
staged activation reads + staged weight reads + output writes. It does not
include cache behavior, residual/scale reads, command overhead, or other CLIP
kernels.

Top static shapes by B=3 FLOPs:

| Phase | Shape | Count | FLOPs B3 | Traffic B3 |
| --- | --- | ---: | ---: | ---: |
| forward | `256->768 @16x16` | 10 | 3.020G | 405.00MiB |
| forward | `768->256 @16x16` | 10 | 3.020G | 367.50MiB |
| backward | `256->768 @16x16` | 10 | 3.020G | 382.50MiB |
| backward | `768->256 @16x16` | 10 | 3.020G | 367.50MiB |
| forward | `128->384 @32x32` | 6 | 1.812G | 270.00MiB |
| backward | `128->384 @32x32` | 6 | 1.812G | 243.00MiB |

Full output:

- `results/2026-07-08/pointwise_report_b3.md`

## Timestamp Cross-Check

Command:

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts > experiments/clip_forks/v17_pointwise_roofline/results/2026-07-08/dispatch_profile_b3_timestamp.csv
```

Parsed group sums:

| Group | Timestamp Sum | Share |
| --- | ---: | ---: |
| `pw_bwd` | 34.34 ms | 26.3% |
| `spatial_bwd` | 26.87 ms | 20.6% |
| `pw` | 18.02 ms | 13.8% |
| `conv` | 16.38 ms | 12.6% |
| `pw+gelu` | 14.94 ms | 11.5% |
| `gelu_bwd` | 9.57 ms | 7.3% |
| total | 130.42 ms | 100.0% |

Read: the pointwise family (`pw_bwd + pw + pw+gelu`) is `67.31 ms`, or `51.6%`
of this B=3 CLIP train timestamp profile. `spatial_bwd` is the next largest
family, so it remains a valid second target.

## f16 Reconciliation

The earlier f16 fork was useful but not a full "CLIP is solved" precision
rewrite. The current `weightsDecl()` supports f16 weight storage while math
stays f32:

```wgsl
@group(...) var<storage, read> weights : array<vec4<f16>>;
fn W4(i : u32) -> vec4f { return vec4f(weights[i]); }
```

That halves weight payload/bandwidth, but activations, saved train tensors,
gradients, GELU/softmax reductions, and splat/raster gradients remain f32. The
recorded v02 result passed embedding cosine but failed the input-gradient gate
(`0.9749` cosine), and integrated timing was slower in that run. So we should
not treat f16 as already promoted. A narrower pointwise-only f16 or f16 hidden
activation fork is still plausible, but it needs the same `dL/dimage` gate.

## Why Not Fuse All Of CLIP

`pw+gelu` helps because it removes a standalone elementwise dispatch and an
extra read/write while still saving the pre-GELU activation needed for backward.
That is a local producer/consumer fusion.

The full CLIP graph cannot become one giant shader for this training path:

- every layer depends on the previous layer's complete tensor;
- attention, SE, spatial conv, pointwise matmul, GELU, and head/loss reductions
  have different workgroup shapes and memory layouts;
- WebGPU workgroup memory and barriers are local to one workgroup, not the
  whole dispatch grid;
- train mode must preserve saved activations for backward and ultimately
  `dL/dimage`;
- a giant monolithic shader would likely spill registers/private memory and
  lose occupancy on Metal.

The right fusion boundary is local and measured: fuse adjacent operations only
when it removes real traffic or dispatches without changing the dependency
surface.

## Next Coding Targets

1. **Rectangular pointwise tile fork**

   Implement `PW_TILE_VARIANT=rect_8x16` or a dual-cout variant for selected
   high-channel shapes. Target the repeated `256<->768 @16x16` and
   `512<->1536 @8x8` families first.

2. **Split-K `pw_bwd` fork**

   Late `pw_bwd` has low spatial size and high channel count. Split the
   reduction dimension, write partials, then reduce. Gate by timestamp including
   the reduce dispatch.

3. **Pointwise-specific precision fork**

   Do not rerun blanket f16. Try selected pointwise weights or selected GELU
   hidden outputs as f16, with f32 accumulation and a full input-gradient cosine
   gate.

4. **Proxy ladder after exact-math gates**

   A faster distilled/proxy CLIP is plausible, but it changes the objective.
   Build it as a teacher/proxy schedule only after exact CLIP kernels stop
   moving.

## Decision

Pointwise is still the largest CLIP shader family by measured B=3 timestamp, and
the static plan explains why: 96 pointwise train dispatches, 26.6 GFLOP at
batch 3, and multiple GiB of lower-bound staged traffic. The next real
implementation fork should be a rectangular/split-K pointwise variant, not more
generic prose and not a blanket precision change.
