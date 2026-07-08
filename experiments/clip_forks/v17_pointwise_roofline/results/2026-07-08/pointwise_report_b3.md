# MobileCLIP Pointwise Static Report

Plan: `plan_train.json` (mobileclip_s0_vision)
Input resolution: `3x256x256`
Batch multiplier shown: `3`

## What Pointwise Means Here

Pointwise is a `1x1`, `groups=1` convolution. For each pixel/token position, it is a dense channel matrix multiply:

```text
Y[p, co] = bias[co] + sum_ci X[p, ci] * W[ci, co]
```

The current WGSL stores activations as channel-planar `vec4f` pixel quads and stores pointwise weights transposed as `[Cin][Cout]` so one `W4()` load gives four adjacent output channels.

Current tile geometry:

- workgroup size: `8 x 8 = 64` threads
- tile: `8` pixel-quads by `32` output channels = `32` pixels by `32` channels
- workgroup memory: `xS=256 vec4f` + `wS=256 vec4f` = `8192` bytes
- each thread accumulates four `vec4f` outputs: `4` output channels by `4` pixels

## Static Totals

| Metric | Per Image | Batch 3 |
| --- | --- | --- |
| forward pointwise dispatches | 48 | 48 |
| backward `pw_bwd` dispatches | 48 | 48 |
| forward pointwise+GELU candidates | 24 | 24 |
| pointwise FLOPs | 8.858 GFLOP | 26.575 GFLOP |
| approx staged global traffic | 1148.38 MiB | 3445.13 MiB |
| pointwise workgroups | 17216 | 51648 |

The traffic number is a lower-bound model for the current tiled kernels: staged activation reads + staged weight reads + output writes. It does not include cache misses, extra residual/scale reads, command overhead, or non-pointwise kernels.

## Top Shapes By FLOPs

| Phase | Shape | Count | Indexes | FLOPs B3 | Traffic B3 | Intensity | Flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward | 256->768 @16x16 | 10 | 57,62,67,72,77,82,87,92,97,102 | 3.020G | 405.00MiB | 7.1 FLOP/B | 10 gelu-fuse |
| forward | 768->256 @16x16 | 10 | 59,64,69,74,79,84,89,94,99,104 | 3.020G | 367.50MiB | 7.8 FLOP/B | 10 residual |
| backward | 256->768 @16x16 | 10 | 30,36,42,48,54,60,66,72,78,84 | 3.020G | 382.50MiB | 7.5 FLOP/B | - |
| backward | 768->256 @16x16 | 10 | 32,38,44,50,56,62,68,74,80,86 | 3.020G | 367.50MiB | 7.8 FLOP/B | - |
| forward | 128->384 @32x32 | 6 | 22,27,32,37,42,47 | 1.812G | 270.00MiB | 6.4 FLOP/B | 6 gelu-fuse |
| forward | 384->128 @32x32 | 6 | 24,29,34,39,44,49 | 1.812G | 225.00MiB | 7.7 FLOP/B | 6 residual |
| backward | 128->384 @32x32 | 6 | 95,101,107,113,119,125 | 1.812G | 243.00MiB | 7.1 FLOP/B | - |
| backward | 384->128 @32x32 | 6 | 97,103,109,115,121,127 | 1.812G | 225.00MiB | 7.7 FLOP/B | - |
| forward | 512->1536 @8x8 | 4 | 111,115,118,122 | 1.208G | 150.75MiB | 7.6 FLOP/B | 2 gelu-fuse |
| backward | 1536->512 @8x8 | 4 | 8,13,17,22 | 1.208G | 145.50MiB | 7.9 FLOP/B | 2 accumulate |
| forward | 64->192 @64x64 | 2 | 8,13 | 0.604G | 108.00MiB | 5.3 FLOP/B | 2 gelu-fuse |
| forward | 192->64 @64x64 | 2 | 10,15 | 0.604G | 78.00MiB | 7.4 FLOP/B | 2 residual |
| forward | 1536->512 @8x8 | 2 | 117,124 | 0.604G | 72.75MiB | 7.9 FLOP/B | 2 residual |
| backward | 512->1536 @8x8 | 2 | 6,15 | 0.604G | 74.25MiB | 7.8 FLOP/B | - |

## Top Individual Dispatches By FLOPs

| Phase | Index | Label | FLOPs B3 | Traffic B3 | Workgroups B | Flags |
| --- | --- | --- | --- | --- | --- | --- |
| forward | 8 | pw 64->192 @64x64 | 0.302G | 54.00MiB | 2304 | gelu-fuse |
| forward | 10 | pw 192->64 @64x64 | 0.302G | 39.00MiB | 768 | residual |
| forward | 13 | pw 64->192 @64x64 | 0.302G | 54.00MiB | 2304 | gelu-fuse |
| forward | 15 | pw 192->64 @64x64 | 0.302G | 39.00MiB | 768 | residual |
| forward | 22 | pw 128->384 @32x32 | 0.302G | 45.00MiB | 1152 | gelu-fuse |
| forward | 24 | pw 384->128 @32x32 | 0.302G | 37.50MiB | 384 | residual |
| forward | 27 | pw 128->384 @32x32 | 0.302G | 45.00MiB | 1152 | gelu-fuse |
| forward | 29 | pw 384->128 @32x32 | 0.302G | 37.50MiB | 384 | residual |
| forward | 32 | pw 128->384 @32x32 | 0.302G | 45.00MiB | 1152 | gelu-fuse |
| forward | 34 | pw 384->128 @32x32 | 0.302G | 37.50MiB | 384 | residual |
| forward | 37 | pw 128->384 @32x32 | 0.302G | 45.00MiB | 1152 | gelu-fuse |
| forward | 39 | pw 384->128 @32x32 | 0.302G | 37.50MiB | 384 | residual |
| forward | 42 | pw 128->384 @32x32 | 0.302G | 45.00MiB | 1152 | gelu-fuse |
| forward | 44 | pw 384->128 @32x32 | 0.302G | 37.50MiB | 384 | residual |

## Why This Can Bottleneck

Pointwise is many small-to-medium matmuls rather than one huge GEMM. The arithmetic is large, but each layer is a separate specialized dispatch and the train path runs both forward and backward. Batch-major CLIP reduces outer scheduling overhead, but the normal z-batch path still repeats the same pointwise work per rendered view.

Fusing pointwise+GELU helps because it removes a separate elementwise read/write and dispatch while still preserving the pre-activation needed by backward. It cannot make the whole CLIP graph one shader because later layers need barriers between dependent tensors, attention/SE/spatial kernels have different dataflow, and train mode must keep saved activations for `dL/dimage`.

## Next Exact-Math Forks

1. Rectangular pointwise tiles for selected hot shapes, especially late high-channel shapes.
2. Split-K `pw_bwd` for low-spatial, high-channel backward layers.
3. Pointwise-specific f16 storage for selected weights or hidden GELU outputs, with full input-gradient gates.
4. Proxy/teacher schedules only after the exact CLIP report and timestamp profile agree that kernel work is still the limiting factor.

