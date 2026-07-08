# v20 Agent Note: Remaining Local CLIP Fusions

Date: 2026-07-08

Scope: read-only inspection of `models/mobileclip_s0/plan_train.json`,
`tools/clip/compile_plan.py`, and the CLIP WGSL emitters. No runtime source code
was edited for this note.

## Current Train-Plan Shape

`plan_train.json` has:

```text
forward steps:       129
backward entries:    152
slots:               260
nActSlots:           130

forward conv:general     5
forward conv:depthwise  40
forward conv:pointwise  48
forward se               3
forward attn_core        2
forward head             1
forward gelu            30

backward loss_bwd        1
backward head_bwd        1
backward gelu_bwd       30
backward se_bwd          3
backward spatial_bwd    45
backward residual_bwd   22
backward pw_bwd         48
backward attn_core_bwd   2
```

The known local fusions are:

- Forward `conv:pointwise -> gelu`: 24 pairs. The existing
  `pointwiseFusedGelu()` path writes both the saved pre-activation slot and the
  post-GELU slot, so backward still has its `pre` tensor.
- v11 backward fusions:
  - `gelu_bwd -> pw_bwd`: 24 pairs.
  - `residual_bwd -> pw_bwd`: 22 pairs.

Dispatch arithmetic:

```text
unfused train dispatches:                  281 = 129 forward + 152 backward
minus pointwise forward GELU fusion:        24
minus v11 backward local fusions:           46
known-fused baseline if all those are on:  211

remaining additive local GELU fusions:      12 = 6 forward + 6 backward
possible floor from local GELU cleanup:    199 dispatches
```

That `199` number assumes choosing one backward orientation for each remaining
GELU backward dispatch. The alternative backward orientation described below is
not additive with `gelu_bwd -> se/spatial_bwd`; it removes the same six
standalone `gelu_bwd` dispatches.

## Additive Remaining Candidates

### 1. Forward `conv:general/depthwise -> gelu`

Legal: yes.

Dispatches removed: 3.

Exact pairs:

```text
step 0   conv:general   3->64   k3s2 g1  @128x128  n=1048576
step 2   conv:depthwise 64->64  k3s2 g64 @64x64    n=262144
step 16  conv:general   64->128 k7s2 g64 @32x32    n=131072
```

Why legal:

- The train compiler already splits these producer activations to `act:"none"`.
- The following `gelu` reads exactly the producer `dst`.
- Backward only needs the saved pre-activation slot, so a fused producer can keep
  writing `dst = pre` and additionally write `gelu.dst = gelu(pre)`.
- No residual/layer-scale epilogue is present on these spatial conv producers.

Code functions to change:

- `src/clip/vision_wgsl.ts`
  - add a dual-output variant of `spatialConv()`, for example
    `spatialConvFusedGelu(s: ConvStep, gelu: GeluStep, opts)`;
  - keep the existing `spatialConv()` fallback unchanged;
  - reuse `GELU` and preserve both pre and post slots.
- `src/clip/vision_batch_wgsl.ts`
  - extend `BatchDispatchOptions` with a gated flag such as
    `fuseNonPointwiseGeluForward`;
  - extend `forwardDispatches()` lookahead for `conv:general -> gelu` and
    `conv:depthwise -> gelu`.
- If single-image `VisionTrainer` should get the same behavior, `planDispatches()`
  or a new forward dispatch planner also needs an option-aware lookahead. Today
  the pointwise forward fusion is wired through the batch-major planner.
- Tooling/plumbing if benchmarked: `tools/clip/dispatch_profile.ts`,
  `tools/clip/batch_major_train_bench.ts`,
  `tools/clip/batch_major_train_matrix.ts`, `src/splat3d/optimize.ts`,
  `tools/splat3d/step_bench.ts`, and `tools/splat3d/step_matrix.ts`.

Risk: medium.

The algebra is simple, but the emitter is not: `spatialConv()` has vectorized
horizontal tiles, an unrolled interior path, and a border path. The fused variant
must add a second output binding and write four GELU values wherever the current
shader writes four pre values. The stem tensor is large, so this can be a real
memory/dispatch cleanup, but it adds code duplication around a hot, already tuned
shader.

### 2. Forward `se -> gelu`

Legal: yes.

Dispatches removed: 3.

Exact pairs:

```text
step 51   se c256  @16x16  n=65536
step 106  se c512  @8x8    n=32768
step 126  se c1024 @8x8    n=65536
```

Why legal:

- `seStep()` already has an activation epilogue concept through `s.act`.
- In train mode these SE steps are emitted as `act:"none"` followed by a split
  `gelu`.
- A fused train variant can write the SE pre slot and the post-GELU slot in the
  final `for (var i = li; i < c * P; ...)` loop.
- Backward `se_bwd` reads the saved SE input, not the SE output; `gelu_bwd` reads
  the pre slot. Both are preserved.

Code functions to change:

- `src/clip/vision_wgsl.ts`
  - add `seStepFusedGelu(s: SeStep, gelu: GeluStep, opts)`;
  - keep `seStep()` unchanged as the fallback;
  - optionally factor the final SE store expression so the non-fused and fused
    variants share the gate recompute body.
- `src/clip/vision_batch_wgsl.ts`
  - extend `forwardDispatches()` to detect `se -> gelu` under the same non-PW
    GELU forward flag.
- Same benchmark/tool plumbing as candidate 1.

Risk: low to medium.

This is the cleanest remaining forward fusion. SE is one workgroup and already
computes the scalar that needs GELU in the final store. Main risks are code
duplication and making sure the fused path still writes both slots for train
verification and backward.

### 3. Backward `gelu_bwd -> se_bwd`

Legal: yes.

Dispatches removed: 3.

Exact pairs:

```text
backward 2   gelu_bwd -> se_bwd  c1024 mid64  @8x8
backward 26  gelu_bwd -> se_bwd  c512  mid128 @8x8
backward 91  gelu_bwd -> se_bwd  c256  mid64  @16x16
```

Why legal:

Baseline:

```text
gelu_bwd writes dPre = dPost * geluGrad(pre)
se_bwd reads dPre as its dY
```

Fused:

```text
se_bwd binds dPost and pre
every logical dY use becomes dPost[i] * geluGrad(pre[i])
```

Required guards:

```text
gelu.kind == "gelu_bwd"
se.kind == "se_bwd"
gelu.accumulate == false
gelu.dX == se.dY
gelu.n == se.c * se.h * se.w
```

Code functions to change:

- `src/clip/vision_bwd_wgsl.ts`
  - add a scalar `geluGrad1()` helper or expose a reusable scalar/vector GELU
    derivative fragment;
  - add `canFuseGeluBwdIntoSe(gelu, se)`;
  - add `seBwdAfterGelu(gelu, se, opts)` or parameterize `seBwd()` with a
    generated `dyAt(i)` expression;
  - extend `planBwdDispatches()` after the existing v11 checks.
- `tools/clip/bwd_test.ts`
  - add an isolated test path if full `VisionTrainer` parity does not clearly
    cover the fused SE branch.
- Tooling/plumbing: same flag/matrix/profile locations as v11. A separate gate
  such as `FUSE_NONPW_GELU_BWD=1` is cleaner than overloading
  `FUSE_GELU_BWD_PW`.

Risk: medium.

`se_bwd` uses `dy` in both the gate-gradient reduction and final direct path.
A naive fused version recomputes `geluGrad(pre)` twice per element because the
full `c * P` tensor cannot live in workgroup memory. The tensors are small enough
that this may still win, but it needs timestamp data. Correctness risk is
moderate because one missed `dy` use silently changes the SE chain rule.

### 4. Backward `gelu_bwd -> spatial_bwd`

Legal: yes.

Dispatches removed: 3.

Exact pairs:

```text
backward 132  gelu_bwd -> spatial_bwd  64<-128 k7s2 g64 @64x64
backward 148  gelu_bwd -> spatial_bwd  64<-64  k3s2 g64 @128x128
backward 150  gelu_bwd -> spatial_bwd  3<-64   k3s2 g1  @256x256
```

Why legal:

Same algebra as SE:

```text
spatial_bwd currently samples dPre[co, oy, ox]
fused spatial_bwd samples dPost[co, oy, ox] * geluGrad(pre[co, oy, ox])
```

Required guards:

```text
gelu.kind == "gelu_bwd"
spatial.kind == "spatial_bwd"
gelu.accumulate == false
gelu.dX == spatial.dY
gelu.n == spatial.cout * spatial.outH * spatial.outW
```

Code functions to change:

- `src/clip/vision_bwd_wgsl.ts`
  - add `canFuseGeluBwdIntoSpatial(gelu, spatial)`;
  - add fused variants for every spatial emitter that can be selected today:
    - `spatialBwdAfterGelu()` for generic gather;
    - `spatialBwdDepthwise4AfterGelu()` for `spatialBwdVariant:"depthwise4"`;
    - `spatialBwdStem4AfterGelu()` for `stemSpatialBwd`;
  - add scalar/vector GELU derivative helpers;
  - extend `planBwdDispatches()` without breaking the v11 pair scan.
- `tools/clip/bwd_test.ts`
  - add focused generic, depthwise4, and stem fused-GELU cases.
- Same benchmark/tool plumbing as candidate 3.

Risk: high.

This is algebraically local, but it is not obviously a performance win. The
standalone `gelu_bwd` computes one expensive derivative per output gradient
element. A naive fused `spatial_bwd` computes that derivative at every sampled
`dy` tap, so the derivative can be repeated across kernel taps and input pixels.
That is especially dangerous for the `k7s2 g64` case and for the stem
specialization. Treat this as a gated experiment, not an obvious promotion.

## Alternative Backward Orientation For The Same Six GELUs

There is another legal way to remove the remaining six `gelu_bwd` dispatches:
fuse the GELU derivative into the backward producer immediately before
`gelu_bwd`, making that producer write `gelu.dX` directly.

Exact remaining triplets:

```text
head_bwd    -> gelu_bwd -> se_bwd       1 pair
pw_bwd      -> gelu_bwd -> se_bwd       2 pairs
pw_bwd      -> gelu_bwd -> spatial_bwd  2 pairs
spatial_bwd -> gelu_bwd -> spatial_bwd  1 pair
```

Total dispatches removed: 6, not additive with candidates 3 and 4.

Why it may be better:

- The derivative is applied once when the previous backward shader stores the
  gradient for the GELU input.
- The following `se_bwd` or `spatial_bwd` can stay unchanged and read a normal
  materialized pre-GELU gradient.
- This avoids the repeated derivative hazard in fused `spatial_bwd`.

Code functions to change:

- `src/clip/vision_bwd_wgsl.ts`
  - add producer-store variants such as:
    - `pwBwdStoreGeluGrad(pw, gelu, opts)`;
    - `spatialBwdStoreGeluGrad(spatial, gelu, opts)`;
    - `headBwdStoreGeluGrad(head, gelu, opts)`;
  - extend `planBwdDispatches()` to detect `producer_bwd -> gelu_bwd`;
  - define precedence with v11, because the 24 common ConvFFN GELUs can be fused
    either producer-side (`pw_bwd -> gelu_bwd`) or consumer-side
    (`gelu_bwd -> pw_bwd`), but not both.

Risk: medium to high.

This is likely a better orientation for the six non-PW consumers, but it is a
larger emitter surface. It also reopens the v11 design question for the 24
pointwise GELUs: producer-side `pw_bwd -> gelu_bwd` computes the derivative once
per produced element, while current v11 computes it while the next `pw_bwd`
loads source tiles. That could be faster or slower depending on tile reuse and
memory traffic. Do not mix both scans without a clear precedence rule and tests.

## Tempting But Not First-Pass Local Fusions

### Consumer-side forward `gelu -> next_op`

Legal only if the fused consumer still materializes the GELU output slot, or if
the train plan changes to checkpoint/recompute activations.

Dispatches removed: overlaps the same 30 forward `gelu` dispatches already
covered by producer-side pointwise+GELU and the six non-PW producer fusions.

Risk: high relative to value. A pointwise/depthwise consumer would recompute
`gelu(pre)` on source loads, often more than once per element. Producer-side
fusion computes GELU once and writes both required slots, so it dominates for the
current train plan.

### Conv-to-conv, SE-to-conv, and attention-neighbor fusions

Legal as a broad compiler rewrite, not as a local dispatch deletion.

Dispatches removed under current contract: 0 safe local removals.

Why not local:

- The adjacent plan has many `conv -> conv` and `spatial_bwd -> residual_bwd`
  pairs, but the next layer needs a complete tensor before it can start. WebGPU
  has no grid-wide barrier inside one compute dispatch.
- The kernels use incompatible workgroup geometries: pointwise tiled matmul,
  spatial tap gather/scatter, one-workgroup SE reductions, and attention core.
- Train mode intentionally preserves activation slots for backward and
  per-step verification. Dropping those slots is activation checkpointing, not a
  small WGSL emitter fusion.

## Recommended Order

1. Implement forward `se -> gelu` first. It removes 3 dispatches with the lowest
   code and performance risk.
2. Implement forward `general/depthwise conv -> gelu` next. It removes 3 more
   dispatches and touches larger tensors, but duplicates more tuned spatial code.
3. For the six remaining backward GELUs, prototype the producer-side
   `producer_bwd -> gelu_bwd` orientation before `gelu_bwd -> spatial_bwd`.
   It removes the same 6 dispatches while avoiding repeated GELU derivative work
   inside spatial gathers.
4. Keep `gelu_bwd -> se_bwd` as a reasonable small experiment. Keep
   `gelu_bwd -> spatial_bwd` as high-risk until timestamps prove the derivative
   recomputation does not erase the dispatch win.
