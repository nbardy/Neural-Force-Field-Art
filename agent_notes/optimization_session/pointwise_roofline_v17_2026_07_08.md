# v17 Pointwise Roofline Read

Date: 2026-07-08

## Answer

Pointwise is the MobileCLIP `1x1`, `groups=1` convolution family. It is dense
channel mixing at every spatial/token position:

```text
Y[p, co] = bias[co] + sum_ci X[p, ci] * W[ci, co]
```

In the current WGSL, activations are channel-planar and packed as `vec4f` pixel
quads. Pointwise weights are stored transposed as `[Cin][Cout]`, so each `W4()`
load provides four adjacent output channels. The shader tile is:

```text
8 pixel-quads x 32 output channels = 32 pixels x 32 channels
64 threads
xS 256 vec4f + wS 256 vec4f = 8192 bytes workgroup memory
```

Backward `pw_bwd` is the same tiled matmul body over a transposed weight offset:
`dX = W^T dY`. CLIP weights are frozen; there is no `dW`.

## Fresh Evidence

New tool:

```bash
BATCH=3 TOP=14 OUT=experiments/clip_forks/v17_pointwise_roofline/results/2026-07-08/pointwise_report_b3.md bun tools/clip/pointwise_report.ts
```

Static B=3 train report:

- 48 forward pointwise dispatches.
- 48 backward `pw_bwd` dispatches.
- 24 forward pointwise+GELU fusion candidates.
- 26.575 GFLOP of pointwise train math at B=3.
- 3445.13 MiB lower-bound staged global traffic at B=3.
- 51,648 pointwise workgroups at B=3.

Timestamp cross-check:

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Group sums:

| Group | Timestamp Sum | Share |
| --- | ---: | ---: |
| `pw_bwd` | 34.34 ms | 26.3% |
| `pw` | 18.02 ms | 13.8% |
| `pw+gelu` | 14.94 ms | 11.5% |
| pointwise family | 67.31 ms | 51.6% |
| `spatial_bwd` | 26.87 ms | 20.6% |

Pointwise is therefore still the largest CLIP shader family in the current B=3
training path.

## f16 Reconciliation

The prior f16 result was f16 **weight storage**, not full f16 CLIP. The generated
WGSL converts f16 weights to f32 before arithmetic. Activations, saved train
tensors, gradients, and loss reductions remain f32. That is why it can halve
weight payload but still fail the input-gradient gate and not automatically give
a big integrated speedup.

## Why Not Fuse Everything

`pw+gelu` fusion helps because it is a local producer/consumer fusion. It writes
the pre-GELU slot needed for backward and the GELU output slot in one pointwise
dispatch.

The full CLIP graph cannot be one giant shader because dependencies cross whole
tensors, WebGPU barriers are only workgroup-local, attention/spatial/SE/pointwise
use different workgroup geometry, and train mode needs saved activations for
backward to `dL/dimage`. A monolithic shader would likely spill registers and
private memory on Metal.

## Next Bigger Leaps

1. Implement a rectangular pointwise tile fork for selected high-channel shapes.
2. Implement split-K `pw_bwd` for low-spatial/high-channel backward layers.
3. Try pointwise-specific f16 weights or f16 hidden GELU outputs with f32
   accumulation and input-gradient gates.
4. Only then build a CLIP proxy ladder; proxy/distillation changes the teacher,
   so it must be gated by embedding and `dL/dimage` cosine.
