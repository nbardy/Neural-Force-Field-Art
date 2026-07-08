# v18 Agent Note - Pointwise-Specific f16 Experiment

Date: 2026-07-08

Scope: read-only design note. No runtime source code was edited in this pass.

Inspected current-state files:

- `src/clip/vision_wgsl.ts`
- `src/clip/vision_bwd_wgsl.ts`
- `src/clip/vision_batch_wgsl.ts`
- `src/clip/vision_batch_pointwise.ts`
- `src/clip/vision.ts`
- `tools/clip/f16_compare.ts`
- `tools/clip/pack_f16_weights.ts`
- `tools/clip/pointwise_report.ts`
- `experiments/clip_forks/v02_f16_weights/results/2026-07-08.md`
- `agent_notes/optimization_session/pointwise_roofline_v17_2026_07_08.md`

## Short Answer

The next f16 experiment should not be another global `PRECISION=f16` run. v02
already tested all CLIP weights stored as f16 while keeping math and activation
storage f32. That path halved weight payload but failed the `dL/dimage` cosine
gate and was not an integrated speed win.

The narrower v18 experiment should only target selected pointwise `1x1`
matmul weights, preserve the same CLIP input resolution (`3x256x256`), keep
global activation slots and `dL/dimage` f32, and compare against f32 with the
same input/text pair. The first gate remains the real objective gradient:

```text
embedding cosine >= 0.9995
dL/dimage cosine >= 0.995
```

If that fails, do not promote it even if isolated timestamps look better.

## What v02 Actually Did

v02 implemented the existing global `WeightPrecision = "f32" | "f16"` hook.
In current `src/clip/vision_wgsl.ts`, that hook lives in `weightsDecl()`:

```wgsl
// f16 weights-only path
@group(0) @binding(N) var<storage, read> weights : array<vec4<f16>>;
fn W(i : u32) -> f32 { return f32(weights[i >> 2u][i & 3u]); }
fn W4(i : u32) -> vec4f { return vec4f(weights[i]); }
```

That means:

- the storage buffer is f16;
- `W()` and `W4()` return f32 values;
- pointwise accumulators are f32;
- workgroup tiles are still `array<vec4f>`;
- activation slots are still f32 GPU buffers;
- gradient slots are still f32 GPU buffers;
- the final `inputGradBuffer` is still f32.

v02's recorded result:

```text
embedding: cos=0.99999559
inputGrad : cos=0.97493807
gate wanted input-gradient cosine >= 0.995
```

So v02 was not "full fp16 CLIP" and it was not a promoted CLIP speedup. It was
global f16 weight storage with f32 math. The remembered large f16 win came from
older/non-CLIP shader paths, not from the MobileCLIP train graph currently
driving splats.

## Why Pointwise Is the Only Sensible f16 Target

Current static report at full CLIP input resolution:

```bash
BATCH=3 TOP=20 bun tools/clip/pointwise_report.ts
```

Key totals:

```text
Input resolution: 3x256x256
forward pointwise dispatches: 48
backward pw_bwd dispatches: 48
pointwise FLOPs at B=3: 26.575 GFLOP
approx staged pointwise traffic at B=3: 3445.13 MiB
```

The current pointwise math is:

```text
Y[p, co] = bias[co] + sum_ci X[p, ci] * W[ci, co]
dX[p, ci] = sum_co dY[p, co] * W[ci, co]
```

Current memory layout:

```text
activation slot:
  NCHW channel-planar f32
  pointwise shader views it as array<vec4f>
  src[ci * P4 + p4] holds 4 adjacent spatial positions

forward pointwise matrix:
  stored transposed as [Cin][Cout]
  W4((wOff + ci * Cout + co4) / 4) returns 4 adjacent output channels

backward pointwise matrix:
  uses wOffT, a separately packed transposed copy
  same tiled kernel body, computing dX = W^T dY
```

Current tile:

```text
workgroup_size = 8 x 8 = 64 threads
tile = 8 pixel-quads x 32 output channels
     = 32 spatial positions x 32 channels
xS = 256 vec4f = 4096 bytes
wS = 256 vec4f = 4096 bytes
```

Because pointwise is the largest measured CLIP shader family, f16 should be
tested there first. Converting SE, attention, spatial conv, head projection,
or every weight at once makes the gradient failure hard to localize.

## Preserve Same CLIP Resolution

This experiment must keep MobileCLIP's actual input shape:

```text
plan_train.json inputShape = 3 x 256 x 256
```

No downsampled CLIP, no lower render size hidden in the precision gate, and no
grid/contact-sheet prompt shortcut in the correctness test. The f16 candidate
must consume exactly the same image tensor as the f32 baseline and produce a
same-shape f32 `dL/dimage`.

Grid prompting and N-of-K view schedules are separate optimization lanes. They
change how often or how many CLIP calls happen. This v18 precision experiment
should answer a narrower question: can selected pointwise math use f16 storage
or staging without corrupting the full-resolution CLIP gradient?

## Proposed v18 Experiment Ladder

### Phase A - Selective Pointwise Matrix f16, f32 Everything Else

This is the safest first implementation.

Keep:

- f32 input image slot;
- f32 all activation slots;
- f32 all gradient slots;
- f32 text embedding;
- f32 final image embedding;
- f32 `dL/dimage`;
- f32 pointwise accumulators;
- f32 bias and layer-scale reads;
- f32 non-pointwise weights.

Change only:

- selected pointwise matrix reads use f16 storage;
- selected `pw_bwd` transposed matrix reads use f16 storage;
- shader converts f16 matrix tiles to f32 before `fma`.

Implementation shape for a future source fork:

```text
weightsF32: original weights_train.bin, unchanged
weightsF16: weights_train_f16.bin or a smaller selected-pointwise sidecar

selected pointwise forward:
  bias/layerScale -> weightsF32 via W()
  matrix W[ci, co] -> weightsF16 via W4h()
  acc -> f32
  dst -> f32 activation slot

selected pw_bwd:
  transposed matrix W^T -> weightsF16 via W4h()
  dY -> f32 grad slot
  acc -> f32
  dX -> f32 grad slot
```

Using both f32 and f16 buffers is important. v02 used one global precision
choice, so every weight family moved together. v18 should leave non-selected
families on the proven f32 buffer and bind an f16 matrix source only where the
policy says to use it.

### Phase B - Workgroup Weight Tile as f16, f32 Accumulation

If Phase A compiles but is not faster, test whether storing the staged W tile
as `vec4<f16>` helps Metal/Dawn register/shared-memory pressure:

```text
wS: array<vec4<f16>, 256>
xS: array<vec4f, 256>
acc: vec4f
inner loop: vec4f(wS[...]) before fma
```

This still keeps global activations f32 and only rounds selected weights.
It may reduce workgroup memory footprint for `wS` from 4096 bytes to 2048
bytes, but conversion cost could erase the gain. Gate by timestamp and
`dL/dimage`, not intuition.

### Phase C - Optional Activation Staging f16, Not Global f16 Slots

If counters show shared-memory pressure and Phase B is not enough, test f16
staging for selected pointwise source tiles:

```text
xS: array<vec4<f16>, 256>
wS: array<vec4<f16>, 256>
acc: vec4f
global src/dst slots: still array<vec4f>
```

This is riskier because it rounds hidden activations inside the matmul, but it
does not change saved activation storage or `dL/dimage` storage. Treat it as
a second experiment, not part of the first patch.

### Phase D - Selected Hidden Activation Slots f16

Do not start here. This is the first point where global tensor storage changes.

Candidate tensors, if Phase A/B pass:

- pre-GELU slots for repeated FFN expansion layers;
- post-GELU hidden slots that are consumed only by the immediately following
  pointwise contraction;
- maybe contraction outputs before a residual add only if f32 residual path is
  preserved.

Avoid:

- CLIP input slot;
- CLIP input gradient slot;
- final embedding slot;
- embedding gradient slot;
- text embedding;
- head projection output;
- attention q/k/v or attention core inputs until separately gated;
- SE gate intermediates until separately gated.

Global f16 hidden slots require slot-precision metadata, mixed `array<vec4f>` /
`array<vec4<f16>>` bindings, and careful handling of train-mode saved
pre-activations. That is a larger fork and should only follow a passed
weight/staging result.

## Safest Weights To Try First

Use shape/index policies, not "all pointwise".

From the current full-resolution pointwise report, the best first families are
repeated ConvFFN expansion/contraction matrices:

| Family | Why Try | Risk |
| --- | --- | --- |
| forward `256->768 @16x16`, indexes `57,62,67,72,77,82,87,92,97,102` | largest repeated forward family; usually followed by GELU; high traffic | moderate |
| forward `768->256 @16x16`, indexes `59,64,69,74,79,84,89,94,99,104` | paired contraction; residual output but accumulator can stay f32 | moderate |
| backward `256->768 @16x16`, indexes `30,36,42,48,54,60,66,72,78,84` | transposed counterpart of contraction/expansion work; directly affects gradient | higher, must gate |
| backward `768->256 @16x16`, indexes `32,38,44,50,56,62,68,74,80,86` | repeated hot backward family | higher, must gate |
| forward/backward `128<->384 @32x32` | next largest family; enough spatial positions for stable timing | moderate |

Do not include in the first policy:

- first stem-adjacent low-channel layers: they are close to raw pixels and any
  error can propagate through the whole network;
- final `head` projection: small enough to leave f32 and important for final
  embedding direction;
- attention qkv/proj pointwise layers: they feed softmax, so small perturbations
  can amplify through attention probabilities;
- SE MLP weights: small timing share and nonlinear sigmoid gate sensitivity;
- spatial/depthwise conv weights: separate kernel family and already has its
  own optimization lane.

Recommended policy ladder:

```text
PW_F16_POLICY=none
PW_F16_POLICY=fwd_ffn16
PW_F16_POLICY=fwd_bwd_ffn16
PW_F16_POLICY=fwd_bwd_ffn16_32
PW_F16_POLICY=all_ffn_pointwise
```

Where:

- `fwd_ffn16` = forward `256->768` and `768->256 @16x16` only.
- `fwd_bwd_ffn16` = above plus matching backward `pw_bwd` families.
- `fwd_bwd_ffn16_32` = add `128<->384 @32x32`.
- `all_ffn_pointwise` = add repeated FFN pointwise families across all stages,
  still excluding attention, SE, head, and stem-adjacent layers.

## Required Gates

### 1. Compile/Capability Gate

The adapter must expose `shader-f16`, and device creation must request it only
when the f16 policy is enabled.

```text
PW_F16_POLICY=none should not request shader-f16.
PW_F16_POLICY!=none should fail closed if shader-f16 is unavailable.
```

### 2. Same-Input Correctness Gate

Extend the existing `tools/clip/f16_compare.ts` idea to compare:

```text
baseline: f32 weights, f32 pointwise
candidate: f32 base weights + selected f16 pointwise matrix reads
```

Use the same fixture:

```text
models/mobileclip_s0/fixtures/input_1x3x256x256.f32.bin
```

Read back:

```text
VisionTrainer.outputBuffer
VisionTrainer.inputGradBuffer
```

Report:

```text
embedding cosine
embedding relLinf
dL/dimage cosine
dL/dimage relLinf
dL/dimage norm ratio
worst-channel/perceptual-position error summary if easy
```

Minimum pass:

```text
embedding cosine >= 0.9995
dL/dimage cosine >= 0.995
0.95 <= norm(candidate dL/dimage) / norm(f32 dL/dimage) <= 1.05
```

Run more than one text vector. The current v02 tool uses a synthetic text
embedding, which is fine for a smoke gate, but a promotion gate should include
real text embeddings for at least:

```text
a photo of a cat on a black background
a shiny red sports car on a black background
a colorful abstract sculpture on a black background
```

### 3. Dispatch Timing Gate

Use warmed timestamp queries, not one-off wall time:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
PW_F16_POLICY=fwd_bwd_ffn16 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
```

Promotion threshold should be meaningful because the code complexity is real:

```text
selected pointwise family timestamp >= 8-10% faster
total isolated CLIP timestamp >= 4-5% faster
no non-selected groups regress materially
```

### 4. Integrated 3D Gate

If correctness and dispatch timing pass:

```bash
TRIALS=5 CONFIGS=base=3:3,pwf16=3:3:pwf16 RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

Then a small quality gate:

```bash
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat on a black background" CONFIGS=base3=3:3,pwf16=3:3:pwf16 OUT_DIR=/tmp/nffa_pwf16_quality bun tools/splat3d/grid_quality.ts
```

The quality gate should still score with the full f32 teacher or at least read
the candidate's final renders through f32 CLIP evaluation. Otherwise a broken
precision path can grade itself.

## Expected Outcomes

Best case:

- pointwise weight bandwidth or shared-memory pressure is real;
- selected f16 matrix reads pass `dL/dimage`;
- CLIP batch timestamp drops a few percent;
- integrated splat step sees a smaller but measurable win.

Likely case:

- f16 matrix conversion reduces weight traffic but not activation traffic;
- pointwise remains dominated by f32 source/dest traffic and f32 accumulation;
- speedup is small, but we learn which layers are gradient-sensitive.

Failure case:

- `dL/dimage` cosine falls below `0.995`, especially when backward `pw_bwd`
  reads f16 transposed weights;
- f16 conversion overhead makes timestamps flat or slower;
- the path should stay an archived fork like v02.

## Implementation Notes For Future Fork

The least invasive future design is not to mutate the global `weightsDecl()`
semantics. Instead add pointwise-specific helpers:

```text
weightsF32Decl(binding)
weightsF16Decl(binding)
Wf32(i) / W4f32(i)
Wf16(i) / W4f16(i)
```

Then selected pointwise shaders bind both buffers:

```text
binding 0: f32 weights_train.bin
binding 1: f16 weights_train_f16.bin or selected sidecar
binding 2+: slots
```

The normal non-selected dispatches keep the existing single f32 binding. That
keeps v18 narrow and makes diffs easy to review.

For a smaller asset, generate a sidecar:

```text
pointwise_selected_f16.bin
pointwise_selected_map.json
```

But for the first code fork, reusing `weights_train_f16.bin` is simpler because
logical scalar offsets match the f32 plan. The shader can read f16 at the same
`wOff` / `wOffT` indices, while bias and layer-scale stay on f32.

Avoid changing optimizer/raster code in the precision fork. The f16 experiment
should be entirely inside CLIP shader generation, CLIP weight loading, and
CLIP benchmark flags. This keeps the input resolution, prompts, splat params,
Adam, and rasterizer constant.

## Decision

Proceed only with Phase A in a source fork:

```text
selected FFN pointwise matrix weights as f16
f32 bias/layerScale
f32 global activations
f32 gradients
f32 accumulators
same 3x256x256 CLIP input
dL/dimage gate required
```

Do not test all-weight f16 again unless a profiling result specifically shows
that non-pointwise weight traffic is the bottleneck. The prior all-weight f16
result already answered that path poorly enough for the current goal.
