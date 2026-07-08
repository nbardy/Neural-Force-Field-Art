# v19 Agent Note - FP16 Truth

Date: 2026-07-08

Scope: documentation-only pass. I inspected the current CLIP f16-related code,
the v02 f16 fork docs/results, and the current f16 tools. I did not edit source
code.

## Short Answer

No, CLIP fp16 did not actually give a big jump.

What landed is a gated all-weight f16 storage experiment, not full fp16 CLIP.
It halves the CLIP weight blobs and the final image embedding stays extremely
close to f32, but the optimizer-relevant input gradient fails the planned
correctness gate. Speed evidence is mixed and small. The recorded v02 isolated
and integrated timestamp runs were slower with f16 weights, and the current
short reruns still do not show a promotable jump.

The correct status is:

- weights-only f16: implemented, opt-in, useful payload reduction;
- full fp16 / activation-f16 CLIP: not implemented;
- correctness: embedding passes, `dL/dimage` fails;
- speed: not a big win, not enough to override the failed gradient gate;
- default: should remain f32.

## What The Current Code Actually Does

The current CLIP precision hook is `WeightPrecision = "f32" | "f16"` in
`src/clip/vision_wgsl.ts`. The key helper is `weightsDecl()`:

```ts
// src/clip/vision_wgsl.ts:128
/** Weight storage precision is a runtime/compiler option; math stays f32. */
export const weightsDecl = (binding: number, precision: WeightPrecision = "f32") =>
  precision === "f16"
    ? `enable f16;\n` +
      `@group(0) @binding(${binding}) var<storage, read> weights : array<vec4<f16>>;\n` +
      `fn W(i : u32) -> f32 { return f32(weights[i >> 2u][i & 3u]); }\n` +
      `fn W4(i : u32) -> vec4f { return vec4f(weights[i]); }`
    : ...
```

That is weights-only f16 storage. `W()` and `W4()` immediately widen to f32.
The pointwise workgroup tiles are still `array<vec4f>`, and the pointwise
accumulators are still `vec4f`. Spatial, SE, head, and backward emitters also
use `weightsDecl()` but still use f32 source/grad buffers and f32 math around
the weight loads.

Runtime buffers confirm this is not full fp16:

- `src/clip/vision.ts:45` accepts `Float32Array | Uint16Array` only for weights.
- `src/clip/vision.ts:89` sizes the weight buffer by `weights.byteLength`.
- `src/clip/vision.ts:94` still allocates forward slots as `floats * 4`.
- `src/clip/vision.ts:258` allocates the text buffer as `plan.textDim * 4`.
- `src/clip/vision.ts:262` still allocates train activation/grad slots as
  `floats * 4`.

So f16 does not apply to:

- input image slot;
- hidden activation slots;
- saved train activations;
- gradient slots;
- `inputGradBuffer`;
- text embedding;
- embedding/loss reductions;
- pointwise/spatial accumulators;
- raster, Adam, or optimizer state.

The f16 assets are sidecar files, not replacements. Current local sizes:

```text
weights.bin             43M
weights_f16.bin         22M
weights_train.bin       82M
weights_train_f16.bin   41M
```

That payload win is real.

## Runtime And Tool Gates

The tools fail closed when f16 is requested without `shader-f16`.

Current f16 surfaces:

- `tools/clip/pack_f16_weights.ts`
  - converts existing f32 weights to raw IEEE half bits;
  - writes `weights_f16.bin` and `weights_train_f16.bin`;
  - preserves logical scalar count.
- `tools/clip/f16_compare.ts`
  - requests `shader-f16`;
  - compares f32 `VisionTrainer` vs f16-weight `VisionTrainer`;
  - reads back final image embedding and `inputGradBuffer`;
  - strict gate fails unless embedding cosine >= `0.9995` and input-gradient
    cosine >= `0.995`.
- `tools/clip/dispatch_profile.ts`
  - `PRECISION=f16`;
  - requests `shader-f16`;
  - loads `weights_train_f16.bin` / `weights_f16.bin`.
- `tools/clip/batch_major_train_bench.ts`
  - `PRECISION=f16`;
  - requests `shader-f16`;
  - tests same-precision single vs batch-major parity and timing.
- `tools/splat3d/step_bench.ts`
  - `CLIP_PRECISION=f16` or `PRECISION=f16`;
  - requests `shader-f16`;
  - passes `clipWeightPrecision` into the 3D optimizer.
- `tools/splat3d/grid_quality.ts` and `tools/splat3d/cadence_quality.ts`
  - also accept `CLIP_PRECISION=f16`, request `shader-f16`, load f16 weights,
    and record precision in their result metadata.

The 3D optimizer accepts `clipWeightPrecision?: WeightPrecision` and passes it
into both single and batch CLIP trainer creation.

## Correctness Evidence

The v02 result recorded:

```text
embedding: cos=0.99999559 relLinf=1.096e-3 maxDiff=6.807e-4
inputGrad : cos=0.97493807 relLinf=3.933e-1 maxDiff=3.462e-4
```

Planned gate:

```text
embedding cosine >= 0.9995
input-gradient cosine >= 0.995
```

I reran the current comparator:

```bash
STRICT=0 bun tools/clip/f16_compare.ts
```

Current output:

```text
adapter: apple metal-3
embedding: cos=0.99999559 relLinf=1.096e-3 maxDiff=6.807e-4 norm(f16/f32)=1.327e+0/1.326e+0
inputGrad : cos=0.97493807 relLinf=3.933e-1 maxDiff=3.462e-4 norm(f16/f32)=1.601e-2/1.567e-2
```

This reproduces the v02 result exactly. Forward embedding passes. The actual
training signal, `dL/dimage`, fails.

That failure is the main reason not to promote f16 weights. A speed win would
still need an optimizer-level quality gate before promotion; here the direct
gradient gate is already red.

## Speed Evidence

### Recorded v02 Results

v02 CLIP dispatch timestamps:

| Precision | Total isolated median sum |
| --- | ---: |
| f32 | `67.109 ms` |
| f16 weights | `71.107 ms` |

That run had f16 weights slower by about `6.0%`.

v02 batch-major wall bench:

| Precision | Batch-major wall |
| --- | ---: |
| f32 | `41.06 ms/batch` |
| f16 weights | `38.47 ms/batch` |

This is a small isolated wall-time win, but the bench checks same-precision
single-vs-batch parity. It does not prove f16 matches the f32 gradient target.

v02 integrated 3D timestamp:

| Precision | Profile total | CLIP batch |
| --- | ---: | ---: |
| f32 | `54.98 ms` | `42.27 ms` |
| f16 weights | `59.90 ms` | `47.38 ms` |

Integrated f16 was slower in that recorded run.

### Current Short Reruns

Current isolated CLIP timestamp sample:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
PRECISION=f16 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

| Precision | Total isolated median sum |
| --- | ---: |
| f32 | `97.714 ms` |
| f16 weights | `107.872 ms` |

That short current run had f16 weights slower by about `10.4%`.

Current integrated 3D short sample:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=2 WARMUP=1 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_PRECISION=f16 CLIP_BATCH=3 VIEWS=3 RUNS=2 WARMUP=1 bun tools/splat3d/step_bench.ts
```

| Precision | Normal step avg | Profile total | CLIP batch |
| --- | ---: | ---: | ---: |
| f32 | `62.77 ms` | `53.35 ms` | `41.68 ms` |
| f16 weights | `63.65 ms` | `51.51 ms` | `38.86 ms` |

This is mixed: timestamped CLIP batch was lower for f16 in the profile, but
normal step average was slightly slower. It is a short, noisy sample and still
nowhere near a big jump. More importantly, it remains blocked by the failed
input-gradient gate.

## What Was Probably Mistaken For A Big FP16 Jump

The repo has several other wins that are easy to conflate with CLIP fp16:

- Older non-CLIP force-field advect f16 path. Comments in
  `src/render/webgpu/advect.ts` / `advect_wgsl.ts` record a real f16 win around
  `9.3 -> 6.5 ms/step @ 1M` on Apple Metal. That is not MobileCLIP training.
- Batch-major CLIP. This gave real sublinear multi-view improvements by changing
  scheduling and batch layout, not by changing precision.
- N-of-K view scheduling and grid/contact-sheet layouts. These reduce or reshape
  CLIP calls per optimizer step; they are not same-pass fp16 speedups.
- Local backward fusions and spatial backward specializations. These are narrow
  shader/scheduling changes, not precision changes.

The accurate statement is not "fp16 worked." It is "all-weight f16 was tried;
payload improved; gradient quality failed; speed did not justify promotion."

## Weights-Only vs Full FP16

Weights-only f16, current/v02:

- f16 storage buffer for all CLIP weights;
- f32 loads after `W()` / `W4()`;
- f32 activation and grad slots;
- f32 workgroup tiles;
- f32 accumulators and reductions;
- f32 loss, text, embedding, and input gradient;
- opt-in, feature-gated.

Full fp16 / broader mixed precision, not implemented:

- f16 hidden activation slots;
- f16 grad slots;
- f16 input gradient;
- f16 pointwise accumulation;
- f16 attention/loss reductions;
- per-slot dtype metadata;
- mixed f16/f32 binding layout for train-mode saved activations.

Full fp16 is a much larger and riskier change. The failed weights-only gradient
gate is a warning that full precision changes must be highly selective.

## Most Plausible Narrower FP16 Experiment

Do not rerun global all-weight f16. The next plausible f16 experiment is
pointwise-specific f16 matrix reads with f32 everything else.

Phase A should be narrow:

- keep the original f32 `weights_train.bin`;
- bind a second f16 weight buffer, either `weights_train_f16.bin` or a smaller
  pointwise sidecar;
- only selected pointwise matrix reads use f16;
- biases, layer-scale, SE, spatial, attention-sensitive, and head weights stay
  f32;
- selected `pw_bwd` transposed matrix reads may use f16 only after the forward
  pointwise subset passes;
- all global slots, gradients, inputs, outputs, and accumulators stay f32.

Best first candidates:

- repeated FFN pointwise families at `16x16`: `256->768` and `768->256`;
- then matching `pw_bwd` transposed matrices only if forward-only f16 passes;
- optionally add `128<->384 @32x32` after the first family.

Avoid at first:

- stem-adjacent layers close to pixels;
- final head projection;
- attention qkv/proj weights;
- SE gate weights;
- spatial/depthwise conv weights.

Required gates:

- same-input `f32` vs candidate embedding cosine >= `0.9995`;
- same-input `dL/dimage` cosine >= `0.995`;
- `0.95 <= norm(candidate dL/dimage) / norm(f32 dL/dimage) <= 1.05`;
- no NaN/Inf;
- timestamped selected pointwise family win >= `8-10%`;
- total isolated CLIP win >= `4-5%`;
- integrated step win survives repeated same-session trials;
- quality scored by a full f32 teacher, not by the candidate grading itself.

If Phase A passes correctness but not speed, a second narrow test can store only
the staged pointwise weight tile as `vec4<f16>` while keeping f32 accumulation.
If that also passes and still does not move speed, then activation staging or
selected hidden f16 slots can be considered, but those are broader experiments.

## Decision

Do not promote `CLIP_PRECISION=f16` as default.

Keep the current f16 path as an experiment/payload option only. The next
precision fork, if any, should be selective pointwise f16 with f32 gradients and
f32 accumulators, not global all-weight f16 and not full fp16.
