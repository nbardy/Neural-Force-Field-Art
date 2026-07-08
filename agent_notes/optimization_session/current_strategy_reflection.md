# Current Strategy Reflection

Date: 2026-07-08

## F16 Status

F16 has not landed in the runtime. The repo contains f16 plans and agent notes,
and the local adapter supports `shader-f16`, but the current CLIP path still
uses f32 packed weights and f32 activation/gradient slots.

What exists:

- text ONNX can use fp16 weights in WASM-side setup notes;
- WebGPU feature probing confirmed `shader-f16` exists locally;
- f16 weight/activation plans are documented in prior agent notes.

What does not exist:

- no `PRECISION=f16` runtime path;
- no `weights_train_f16.bin` load path;
- no f16 CLIP WGSL codegen;
- no f16 correctness gate for `dL/dpixels`;
- no f16 integrated 3D timing result.

So any memory/throughput jump from f16 is still available to test. It is not an
already-spent win.

## What Pointwise Means

In this CLIP port, "pointwise" means a 1x1 convolution with `groups=1`. For each
pixel/token position `p`, it is just a matrix multiply:

```text
Y[p, cout] = sum_cin X[p, cin] * W[cin, cout] + bias[cout]
```

The code stores image-like activations channel-planar. Pointwise kernels treat
the spatial/token dimension as `P = H * W` positions and operate on vec4-packed
pixel quads:

```text
src: [Cin][P/4] as vec4f
dst: [Cout][P/4] as vec4f
W:   [Cin][Cout] packed so four adjacent Cout values are a vec4
```

The current tiled kernel gives one workgroup a `32 pixel x 32 cout` output tile.
It stages:

- `xS`: a tile of source activations;
- `wS`: a tile of weights;
- accumulates over `Cin` in chunks of 32.

Backward pointwise, `pw_bwd`, is the same math shape over transposed weights:

```text
dX[p, cin] = sum_cout dY[p, cout] * W[cin, cout]
```

It dominates because CLIP has many 1x1 channel expansion/projection layers, and
training needs both forward activations and backward image gradients. The latest
timestamp evidence puts integrated default `3/9`, `batch CLIP x3` at roughly:

```text
clipBatch ~41 ms
rasterBwd ~10 ms
rasterFwd ~1 ms
```

Isolated CLIP timestamps on promoted B=3 settings ranked `pw_bwd`, `pw`,
`pw+gelu`, `spatial_bwd`, and `conv` as the main groups. Attention backward was
visible but not first-order.

## Why Shared-W Was Not Enough

The shared-W experiment tried to put multiple batch lanes inside one workgroup
so those lanes could reuse one staged `W` tile. Some microbench shapes won.
Full-chain B=2/B=3 did not improve, likely because:

- workgroup memory hit the 16 KB budget quickly;
- lane-z workgroups reduced occupancy or scheduling quality;
- not all pointwise shapes benefited;
- full CLIP time includes many non-pointwise kernels and dispatch dependencies;
- the production allowlist added complexity without enough end-to-end gain.

This does not kill pointwise work. It kills that exact small shared-W strategy.

## 3x3 Grid CLIP Idea

The proposed prompt/image shape:

```text
"a 3x3 grid of a cat from 9 different camera angles on a black background"
```

would render all 9 views into one CLIP input image. That can reduce CLIP calls,
but it changes the signal:

- each view becomes about `85x85` pixels inside a 256 image;
- CLIP is trained at the same input resolution, but each object view is lower
  effective resolution;
- composition/grid semantics may dominate unless prompt wording is careful;
- per-view camera prompts become weaker unless labels or prompt text are very
  explicit.

A reasonable ablation is not "replace 9-view CLIP." It is:

1. optimize one global 3x3-grid CLIP image every step;
2. additionally sample 1-2 close-up individual views every step or every other
   step;
3. compare wall time and multi-view quality.

This is a signal/schedule experiment, not a pure shader speedup. It could be a
real 2-4x wall-time lever because it reduces CLIP calls, but quality is the
risk.

## Alternate-Step CLIP

"Alternate-step CLIP" means not every optimizer step uses the full CLIP loss.
Examples:

- step A: optimize a cheap proxy or subset of views;
- step B: optimize full CLIP on sampled views;
- between CLIP steps: use raster regularizers, opacity/radius priors, view
  consistency losses, or stale CLIP gradients.

If there is no alternate objective, then skipping CLIP means there is no useful
prompt-gradient update. So this is only valid if paired with a real proxy or
regularizer.

## Why Not Fuse The Whole CLIP

Pointwise + GELU forward helped because it removes a standalone GELU dispatch
and avoids a memory round trip while still preserving the pre-GELU activation
needed for backward.

Full CLIP fusion is hard because training needs many intermediate activations:

- backward needs saved pre-GELU tensors;
- residual branches need both main and skip gradients;
- attention backward recomputes softmax from saved qkv;
- SE backward recomputes gate internals from saved inputs;
- WebGPU workgroup memory is capped around 16 KB here;
- one giant pass cannot globally synchronize across arbitrary workgroups except
  at dispatch boundaries.

The realistic larger fusion units are local:

- pointwise + activation + residual where saved tensors can still be written;
- backward chains like `gelu_bwd + pw_bwd` when the intermediate has one
  consumer;
- pairs of pointwise projections only if the intermediate does not need to be
  materialized for another branch;
- block-level fusion only for small shapes where workgroup memory and sync
  constraints fit.

## Distill / Proxy Meaning

This does not mean "use a worse text prompt." It means using a cheaper learned
or hand-built guidance signal for most inner-loop steps and full CLIP as the
teacher/checkpoint.

Options:

- train a smaller WGPU image encoder to mimic MobileCLIP gradients on generated
  splat renders;
- use a lower-layer / lower-channel proxy from the current CLIP;
- use a tiny learned projection on cached features;
- use grid CLIP or lower effective view resolution as a schedule proxy.

This is a larger research lane. F16 and pointwise rewrite are more direct first
attempts.

## Real GPU Profiling

Used so far:

- WebGPU timestamp queries;
- dispatch-level CLIP profiler;
- integrated step timestamp profiler;
- microbenches;
- step matrix;
- raster occupancy telemetry.

Not yet used:

- Xcode Instruments / Metal System Trace;
- Chrome tracing for WebGPU queue/command behavior;
- detailed GPU counters for memory bandwidth, occupancy, cache misses, register
  pressure, or threadgroup-memory pressure.

We should run a real profiler session before another deep shader rewrite,
especially before f16 activations or a new pointwise kernel.

## Forking Protocol Going Forward

For shader experiments, use forked variants and leave a trail:

1. create an explicit gate/env flag;
2. add a separate emitter/helper or clearly named variant, not an in-place
   default mutation;
3. for larger ideas, create a copied experiment folder such as
   `experiments/clip_forks/v01_f16_weights/` or
   `experiments/clip_forks/v02_pointwise_tiling/` with notes/patch scaffolding;
4. run correctness/parity first;
5. run CLIP-only and integrated 3D timing;
6. commit the attempt separately;
7. promote in a later commit only if the full-chain gate wins.

The current integrated timestamp profiler was safe to touch shared runtime
entry points because it is instrumentation-only and opt-in. Optimization
variants should be forked/gated more aggressively.
