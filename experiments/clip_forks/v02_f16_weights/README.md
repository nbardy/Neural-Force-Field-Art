# v02: `PRECISION=f16` Weights-Only CLIP Fork

## Goal

Test whether MobileCLIP WebGPU inference/training is meaningfully memory-bandwidth-bound by storing CLIP weights as fp16 while keeping numerically sensitive activations, reductions, logits, and optimizer-side tensors in fp32.

This is a weights-only precision fork. It should not change the CLIP graph, input resolution, prompt schedule, rasterizer, Adam state, or 3D splat parameters.

## Current Status

Implemented and measured as an opt-in experiment. Do not promote the all-weight
f16 path as the default.

Result note:

- `results/2026-07-08.md`

Headline:

- asset payload halves cleanly;
- embedding cosine passes;
- input-gradient cosine fails the planned gate;
- integrated 3D timestamp was slower in the recorded run.

## Asset Changes

Add second generated weight artifacts instead of replacing the current fp32
files. These live in `models/mobileclip_s0/` and remain gitignored:

- Keep: `models/mobileclip_s0/weights.bin`
- Keep: `models/mobileclip_s0/weights_train.bin`
- Add: `models/mobileclip_s0/weights_f16.bin`
- Add: `models/mobileclip_s0/weights_train_f16.bin`

Generation path:

```bash
bun tools/clip/pack_f16_weights.ts
```

Output from the first run:

```text
weights.bin -> weights_f16.bin: 11353536 scalars, 43.3 MB -> 21.7 MB (0.500x)
weights_train.bin -> weights_train_f16.bin: 21515712 scalars, 82.1 MB -> 41.0 MB (0.500x)
```

The packer converts existing fp32 weights to IEEE fp16 at asset-build time and
writes `Uint16Array` raw half bits. It does not convert every page load in JS.

## Runtime Gate

Introduce one runtime switch:

```bash
PRECISION=f16
```

Default stays fp32. The app and benchmarks should only use fp16 weights when all of these are true:

- `PRECISION=f16`
- `navigator.gpu` exists
- the selected adapter exposes `shader-f16`
- `requestDevice({ requiredFeatures: ["shader-f16"] })` succeeds
- generated `weights_train_f16.bin` loads successfully

If any condition fails, fail closed in benchmarks and show a clear warning in UI/dev logs. Do not silently compare fp16 benchmark results against fp32.

Current exposed tool switches:

```bash
PRECISION=f16 bun tools/clip/dispatch_profile.ts
PRECISION=f16 bun tools/clip/batch_major_train_bench.ts
CLIP_PRECISION=f16 bun tools/splat3d/step_bench.ts
```

## `requestDevice` Handling

Current WebGPU setup should be changed from unconditional device creation to feature-gated creation:

```ts
const adapter = await navigator.gpu.requestAdapter();
const wantsF16 = process.env.PRECISION === "f16";
const canF16 = !!adapter?.features.has("shader-f16");

const device = await adapter.requestDevice({
  requiredFeatures: wantsF16 && canF16 ? ["shader-f16"] : [],
});
```

Risk gate: if `wantsF16 && !canF16`, throw in benchmark scripts. In the app, surface a WebGPU capability warning and fall back to fp32 only if the UI explicitly labels the fallback.

## WGSL Declarations

All f16 shaders must enable the extension:

```wgsl
enable f16;
```

Use precision-specific declarations instead of changing call sites ad hoc:

```wgsl
// fp32 path
@group(0) @binding(N) var<storage, read> weights: array<vec4f>;

// f16 weights-only path
@group(0) @binding(N) var<storage, read> weights: array<vec4<f16>>;
```

Accumulate in fp32 unless a measured fork proves f16 accumulation is safe:

```wgsl
let w16: vec4<f16> = weights[wIndex];
let w: vec4f = vec4f(w16);
acc = fma(x, w, acc);
```

Apply this to generated WGSL for:

- pointwise forward kernels
- pointwise backward kernels
- spatial convolution kernels
- spatial backward kernels
- attention projection weights, if they are stored in the shared generated weight buffer

Keep these fp32:

- activations saved for backward
- gradients
- logits and CLIP similarity reductions
- Adam state
- loss accumulation
- softmax/sum reductions

## JS Buffer Loading

Add a precision-aware weight loader:

```ts
const weightUrl = precision === "f16"
  ? `${MODEL_BASE}weights_train_f16.bin`
  : `${MODEL_BASE}weights_train.bin`;

const bytes = await fetch(weightUrl).then((r) => r.arrayBuffer());
```

For f16, upload bytes directly to a storage buffer. Do not round-trip through `Float32Array`.

Expected buffer usage remains:

```ts
GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
```

Alignment check: every f16 tensor offset used as `array<vec4<f16>>` must be 8-byte aligned. If the existing fp32 packed layout assumes 16-byte `vec4f` slots, either:

- keep 16-byte alignment with padding for simpler indexing, or
- emit separate f16 offsets and assert alignment in the generated metadata.

First fork recommendation: keep the same logical vec4 indexing and emit f16-packed offsets with explicit alignment asserts.

## Correctness Tests

Run tests in three layers.

1. Asset sanity:

```bash
bun tools/clip/pack_f16_weights.ts
```

Expected: f16 file is close to half the fp32 size, allowing padding/metadata differences.

2. Numeric parity:

```bash
bun tools/clip/f16_compare.ts
```

Suggested gates:

- image embedding cosine similarity: `>= 0.9995`
- input-gradient cosine similarity: `>= 0.995`
- no NaN/Inf in forward activations
- no NaN/Inf in backward gradients

Recorded all-weight f16 result:

```text
embedding: cos=0.99999559
inputGrad : cos=0.97493807
```

So this fork currently fails the input-gradient gate.

3. Integrated smoke:

```bash
CLIP_PRECISION=f16 CLIP_BATCH=3 VIEWS=3 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
```

Compare against identical fp32 settings.

## Timestamp Bench Commands

Use GPU timestamps, then repeat without timestamps for wall-clock confirmation.

CLIP-only:

```bash
TIMESTAMP=1 PRECISION=f32 CLIP_BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 PRECISION=f16 CLIP_BATCH=3 RUNS=10 WARMUP=3 bun tools/clip/dispatch_profile.ts
```

Integrated 3D:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=10 WARMUP=3 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_PRECISION=f16 CLIP_BATCH=3 VIEWS=3 RUNS=10 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Wall-clock:

```bash
CLIP_BATCH=3 VIEWS=3 RUNS=20 WARMUP=5 bun tools/splat3d/step_bench.ts
CLIP_PRECISION=f16 CLIP_BATCH=3 VIEWS=3 RUNS=20 WARMUP=5 bun tools/splat3d/step_bench.ts
```

Record per-family deltas for:

- `pw_bwd`
- `pw+gelu`
- `pw`
- `spatial_bwd`
- `conv`
- total CLIP batch
- total 3D step

## Promotion Gates

Promote only if all are true:

- fp16 path is opt-in and fp32 remains default
- no unsupported-device crash in normal fp32 mode
- forward embedding parity passes
- backward gradients remain finite over at least 100 optimization steps
- CLIP-only timestamp speedup is at least 15%
- integrated 3D step speedup is at least 8%
- generated f16 asset size reduction is confirmed

Do not promote if:

- quality visibly collapses during prompt-to-splat optimization
- all gains come from timestamp noise but wall-clock does not improve
- f16 support makes device creation brittle
- kernels become harder to maintain without measurable speedup

## Why This May Produce 2-4x

This can help if the hot CLIP kernels are mostly memory-bandwidth-bound. The current profile says the largest bucket is pointwise-heavy work, and pointwise kernels repeatedly stream weights. Halving weight bandwidth can reduce pressure on memory and cache, especially for batch-major CLIP where the same weights are reused across views.

Best-case path:

- f16 weights halve weight-buffer traffic
- pointwise kernels are bandwidth-bound
- cache residency improves
- fewer bytes moved also helps backward kernels
- integrated CLIP batch time drops enough that the whole 3D step moves materially

In that best case, this fork might combine with pointwise tiling/fusion and N-of-K view scheduling toward an overall 2-4x workflow speedup.

The first measured all-weight f16 version did not show that best case. A future
f16 fork should be selective: separate f16/f32 weight buffers and only convert
families that preserve gradient quality and win timestamps.

## Why This May Not Produce 2-4x

Weights-only f16 is unlikely to create a standalone 4x speedup because:

- activations, gradients, reductions, and Adam state stay fp32
- many kernels may be compute-bound or launch-bound instead of bandwidth-bound
- `vec4h -> vec4f` conversion has a cost
- fp16 storage may reduce bandwidth but not reduce dispatch count
- WebGPU/Metal compiler behavior may already cache fp32 weights effectively
- total step time includes raster forward/backward and optimizer work

Expected standalone result is more likely 10-40% CLIP speedup if memory traffic is the bottleneck, with lower integrated step gains. Treat anything above 2x from this fork alone as suspicious until verified by timestamps, wall-clock, and shader counter traces.

## Follow-Up If It Wins

If weights-only f16 clears the gates, next forks should test:

- f16 activations for selected pointwise intermediates
- f16 pointwise accumulation on non-sensitive layers
- f16/f32 mixed precision per layer based on parity
- pointwise tile rewrite using the reduced weight bandwidth
- residual-backward plus pointwise-backward fusion on top of f16 weights
