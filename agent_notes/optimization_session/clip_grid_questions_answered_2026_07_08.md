# CLIP / Grid Questions Answered

Date: 2026-07-08

## Did FP16 Already Give A Big CLIP Jump?

No, not for the CLIP train loop.

What happened:

- `7548a4f` implemented gated f16 CLIP weights.
- `weights_train.bin -> weights_train_f16.bin` halves storage, about
  `82.1 MB -> 41.0 MB`.
- Embedding similarity is excellent: `cos=0.99999559`.
- Input-gradient similarity fails the strict gate:
  `inputGrad cos=0.97493807`, target was `>=0.995`.
- Timings are mixed/small: some fresh short runs show small speedups, but the
  recorded stable v02 dispatch profile was f16 slower (`67.109 ms -> 71.107 ms`).

The remembered "big jump" was likely one of these:

- batch-major CLIP, especially B=9 forward or forward+backward batching;
- view scheduling / N-of-K;
- the older non-CLIP force-field advect f16 shader path.

Decision: do not promote CLIP f16 weights as default unless the gradient gate is
fixed or replaced by a deliberate optimizer-level quality gate.

## What Does "Pointwise" Mean?

In this CLIP image tower, pointwise means a `1x1`, `groups=1` convolution. It is
the dense channel-mixing operation at every spatial position:

```text
Y[co, p] = bias[co] + sum_ci X[ci, p] * W[ci, co]
```

For backward, weights are frozen, so we only need the pixel/input gradient:

```text
dX[ci, p] = sum_co dY[co, p] * W[ci, co]
```

The code stores activations as channel-planar `vec4f` pixel quads:

```text
src[ci * P4 + p4] = vec4f(X[ci, p4*4 + 0..3])
dst[co * P4 + p4] = vec4f(Y[co, p4*4 + 0..3])
P4 = H * W / 4
```

Forward pointwise weights are packed as `[Cin][Cout]`, so four adjacent output
channels can be loaded with one `W4(...)`. Backward uses a transposed packed
copy so the same tiled body can compute `dX`.

The current tiled kernel:

- `workgroup_size(8, 8)` = 64 threads.
- One workgroup owns `8` pixel-quads x `32` output channels.
- One thread accumulates four adjacent pixels for four output channels.
- It stages `xS` and `wS`, each `256 vec4f` = 4 KB, so 8 KB workgroup memory.
- It loops over channels in 32-channel chunks.

Why it is a bottleneck:

- MobileCLIP has many ConvFFN blocks that are mostly
  `pointwise expansion -> GELU -> pointwise contraction -> residual`.
- The current train plan has 48 forward pointwise steps and 48 backward
  pointwise steps.
- A fresh timestamp run had `pw_bwd + pw+gelu + pw` at roughly `52.5%` of the
  isolated CLIP timestamp sum.

## Why Not Fuse All Of CLIP Into One Big Shader?

We can and should fuse local pairs, but "one giant CLIP shader" is the wrong
shape for WebGPU training.

Reasons:

- Backward needs saved intermediate activations from many layers.
- Different layers have different tensor shapes, channel counts, reductions,
  attention shapes, SE blocks, residuals, and spatial convs.
- A giant shader would explode register pressure and private storage.
- Workgroup barriers only synchronize inside one workgroup, not across the
  whole image/model.
- Many layer boundaries are real global memory boundaries because the next
  layer reads a complete tensor produced by many workgroups.
- WebGPU has no CUDA-style persistent cooperative grid that would make a full
  model fusion practical.

Good fusion targets are local:

- pointwise forward + GELU: already implemented as `pw+gelu`.
- GELU backward + pointwise backward: gated.
- residual copy + pointwise backward: gated.
- future pointwise epilogue fusions where the producer/consumer are one-to-one
  and share the same shape.

## Same CLIP Resolution And Grid Contact Sheet

We should keep the CLIP input at `256x256`. `grid9_close2` already does that:

```text
lane 0: 256x256 CLIP input containing a 3x3 grid
cells:  80x80 each, with 8px gutters
lanes 1-2: two full 256x256 close-ups
```

This lowers per-view effective CLIP resolution while preserving the model's
trained input size.

The v08 fork makes the grid text explicit:

```text
a 3x3 image grid showing the same subject, {prompt}, from nine different camera
angles, centered on a black background: ...
```

This is a quality/signal ablation, not a kernel speed change.

The bigger speed leap would be direct cell rasterization: render each grid view
straight into its `80x80` cell instead of rendering a full `256x256` scratch
image and downsampling it. That keeps CLIP at `256x256` while reducing grid
raster work.

## What Is "Alternate-Step CLIP"?

Skipping CLIP on a step by itself does not optimize the prompt. Without CLIP or
another image objective, there is no meaningful prompt gradient.

Useful variants:

- N-of-K views: optimize 1, 2, or 3 camera prompts per step, cover all views
  over time.
- Grid/contact-sheet proxy: one CLIP lane supervises all views at lower
  effective per-view resolution, with close-up lanes for detail.
- Periodic full refresh: cheap schedule most steps, full all-view CLIP every
  `M` steps.
- Non-CLIP regularizer-only steps only after adding real regularizers worth
  optimizing, such as opacity, bounds, background, or smoothness.

## What Would A Proxy / Distiller Mean?

Not "lower precision" by itself. A proxy would be a separate cheaper model or
loss that approximates useful CLIP gradients for most inner-loop steps.

Possible ladder:

1. Use real CLIP every step, current baseline.
2. N-of-K and grid schedules, still real CLIP.
3. Real CLIP every `M` steps plus regularizer steps in between.
4. Train a smaller image tower or projection to match MobileCLIP embeddings or
   image gradients on rendered splat images.
5. Use the proxy most steps, real MobileCLIP as teacher/checkpoint.

This is high-upside but high-risk: a proxy can easily optimize texture or color
statistics that look good to the proxy and bad to real CLIP.

## GPU Tooling Status

What we have used:

- WebGPU timestamp queries in `tools/clip/dispatch_profile.ts`.
- Integrated step timestamp/split profiling in `tools/splat3d/step_bench.ts`.
- Browser smoke with `tools/smoke.mjs`.

What we still lack:

- real memory bandwidth counters;
- occupancy / register pressure;
- cache miss and stall reasons;
- shader source line attribution.

The local machine currently has Command Line Tools selected, not full Xcode
GPU tooling, so Metal counter tools were not available through `xcrun` in the
last probe. Chrome tracing is still useful for browser/Dawn queue gaps and
pipeline/CPU scheduling, but not for shader memory counters.
