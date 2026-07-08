# Agent 4: CLIP Fusion Bigger Leaps

Date: 2026-07-08

Scope: fusion strategy for the fused MobileCLIP-S0 WebGPU path. I inspected
`tools/clip/compile_plan.py`, `src/clip/vision_wgsl.ts`,
`src/clip/vision_bwd_wgsl.ts`, `src/clip/vision_batch_wgsl.ts`, the current
pointwise+GELU forward fusion, and the gated GELU-backward-into-pointwise
fusion. This note is analysis only; I did not edit runtime code.

The short answer to "why not fuse it all" is that the useful fusion boundary is
not the whole CLIP graph. It is the set of adjacent producer/consumer steps
where:

- the consumer has the same tensor geometry or can use the producer value while
  it is still in registers,
- train-mode saved activations remain available for backward,
- backward accumulation order remains deterministic, and
- the fused kernel does not require a graph-wide barrier inside one dispatch.

The current code already follows that pattern for `pointwise + GELU`. Bigger
fusions should be forked/gated variants, not silent edits to the default shader
generator.

## Current Plan Facts

Local `models/mobileclip_s0/plan_train.json` summary:

| Item | Count / Share |
| --- | ---: |
| Forward train steps | 129 |
| Backward entries | 152 |
| Activation slots before grad mirroring | 130 |
| Pointwise conv steps | 48 |
| Depthwise conv steps | 40 |
| General spatial conv steps | 5 |
| Train GELU steps | 30 |
| Pointwise->GELU exact pairs | 24 |
| Pointwise residual tails | 22 |
| Residual-bwd->pw-bwd exact pairs | 22 |
| Forward MAC share from pointwise convs | 92.8% |

Remaining exact forward GELU producer pairs after the pointwise fusion are
small but real:

| Producer | Exact Pairs |
| --- | ---: |
| `conv:general -> gelu` | 2 |
| `conv:depthwise -> gelu` | 1 |
| `se -> gelu` | 3 |

Timestamp profiling on the promoted B=3 train path still ranks the hot groups
as `pw_bwd`, `spatial_bwd`, `conv`, `pw+gelu`, and `pw`. Attention backward is
too small to be the first fusion target.

## What "Pointwise" Means Here

In this codebase, pointwise means a 1x1 convolution, which is a per-spatial-cell
matrix multiply:

```text
Y[p, co] = bias[co] + sum_ci X[p, ci] * W[ci, co]
```

The WGSL layout is channel-planar NCHW:

```text
activation slot: X[c][p] where p = y * W + x
vec4 view:       X[c][p4] packs 4 adjacent spatial pixels
weights:         transposed [Cin][Cout], vec4 over 4 consecutive Cout values
```

The main pointwise kernel in `vision_wgsl.ts` uses an 8x8 workgroup. Each
workgroup owns a 32-pixel-by-32-output-channel tile:

- `xS`: 256 `vec4f` values for 32 input channels x 8 pixel-quads.
- `wS`: 256 `vec4f` values for 32 input channels x 8 output-channel quads.
- The reduction loops over `ci` in chunks of 32.
- Each invocation accumulates four output channels for one pixel-quad:
  `acc0..acc3`.

Backward pointwise (`pw_bwd`) reuses the same tiled body with a second
transposed weight copy packed by `compile_plan.py --train`. Algebraically it
computes:

```text
dX[p, ci] = sum_co dY[p, co] * W[ci, co]
```

The bottleneck is therefore not mysterious. The model is mostly repeated
channel mixing at fixed image grids. Pointwise layers are 92.8% of forward
MACs, and train mode also pays the reverse pointwise chain. Memory traffic is
substantial because every train slot is preserved for backward, and B=3 repeats
activation/gradient storage per lane while weights are shared.

## Why Pointwise+GELU Helps

Train mode normally splits GELU out of the producer conv so backward can read
the saved pre-activation:

```text
pointwise: pre = X @ W + b       // write pre slot
gelu:      post = gelu(pre)      // read pre slot, write post slot
```

`pointwiseFusedGelu()` keeps that backward contract but performs the GELU
while the pointwise result is still in registers:

```text
pointwise+gelu:
  pre  = X @ W + b
  post = gelu(pre)
  write pre slot
  write post slot
```

It helps because it removes 24 standalone forward GELU dispatches and removes
the separate storage read of the pre-activation by the GELU pass. The fused
pointwise dispatch still writes both slots, because backward still needs the
pre slot. That is the right kind of fusion: it removes launch and memory traffic
without pretending train mode can discard activations.

Measured result from existing notes:

| Variant | B=3 CLIP Train Median |
| --- | ---: |
| Stem spatial-bwd only | 73.33 ms |
| + pointwise GELU forward fusion | 68.06 ms |

Integrated 3D also improved enough to promote it by default.

The gated `gelu_bwd + pw_bwd` fusion is the reverse version:

```text
baseline:
  gelu_bwd: dPre = dPost * geluGrad(pre)
  pw_bwd:   dX   = W^T @ dPre

fused:
  pw_bwd loadSrc = dPost * geluGrad(pre)
```

That won in CLIP-only timing but was flat in the integrated 3D step, so it is
correctly kept as an ablation gate rather than promoted.

## Why One Giant Fused CLIP Pass Is Hard Or Wrong

One giant fused forward/backward CLIP shader is not a realistic first target in
WebGPU for this model.

1. There is no grid-wide barrier inside a compute dispatch.

   Workgroup barriers only synchronize threads inside one workgroup. CLIP
   layers need all workgroups from the previous layer to finish before the next
   layer reads those outputs. Dispatch boundaries provide that global ordering.
   A single monolithic dispatch would either be incorrect or would have to
   recompute large tensors locally.

2. The step geometries are incompatible.

   Pointwise uses 32x32 channel/pixel tiles. Spatial conv uses per-channel
   spatial kernels. SE uses whole-channel reductions in one workgroup.
   Attention uses one workgroup per head and token-private arrays. The head and
   loss are reductions. Combining these into one generated shader would create
   a huge branchy kernel with poor occupancy, high register pressure, more
   workgroup-memory pressure, and long compile times.

3. Train mode intentionally disables activation reuse.

   `compile_plan.py --train` allocates a unique slot for every activation. That
   is not waste by accident; backward reads saved GELU pre-activations, SE
   inputs, attention QKV, the embedding, and normal layer outputs. A giant
   forward pass cannot simply avoid these writes unless we switch to activation
   checkpointing and recomputation.

4. Backward has real reverse-order accumulation semantics.

   The backward plan mirrors grad slots. The first writer overwrites and later
   writers add. Residuals and multi-consumer tensors depend on strict reverse
   order. Folding large backward regions together risks races or duplicate
   writes unless the exact producer/consumer pair is proven.

5. The current generator is useful because every dispatch is specialized.

   Shapes, strides, offsets, and workgroup sizes are baked into each shader.
   A giant shader would give up that specialization or generate a monster that
   is harder to verify and tune. The bigger win is to add a few more specialized
   fused emitters, not collapse the compiler into one pass.

## Realistic Larger Fusion Units

### 1. Non-Pointwise Producer + GELU Forward

Current pointwise GELU fusion covers 24 of 30 train GELUs. The remaining exact
pairs are 2 general convs, 1 depthwise conv, and 3 SE gates. These are the same
safe fusion shape as pointwise+GELU:

```text
producer computes pre
write pre slot
write gelu(pre) to post slot
skip standalone gelu dispatch
```

This is low-risk and probably a small win. It removes only 6 dispatches, but
the stem/downsample GELUs include large tensors (`n=1,048,576`, `262,144`,
`131,072`), so it may still show up in timestamp profiles.

### 2. Residual Gradient Copy Folded Into Pw-Bwd

For residual pointwise tails, backward currently emits:

```text
residual_bwd: grad[res] = dY
pw_bwd:       grad[src] = W^T @ dY
```

There are 22 exact adjacent `residual_bwd -> pw_bwd` pairs where both steps read
the same `dY`, and the residual writes are non-accumulating in the current plan.
This is a good dispatch/traffic fusion candidate.

The implementation detail is subtle: `pw_bwd` has a workgroup y dimension over
output-channel blocks, so the same `dY` tile is loaded for multiple y blocks.
A fused residual copy must only write `grad[res]` from one y block, e.g.
`if (wid.y == 0u)`, otherwise it creates duplicate writes. That is manageable
and testable.

Expected benefit: remove 22 elementwise residual-backward dispatches and one
full gradient-slot copy per residual tail. This probably beats non-pointwise
GELU fusion as the next "cheap but real" fusion target.

### 3. Spatial-Bwd Specialization Beyond Stem

This is not a classic producer/consumer fusion, but it is a bigger leap in the
same spirit: specialize the remaining hot spatial backward shapes instead of
running one generic gather kernel. Timestamp profiles rank `spatial_bwd` near
the top. The stem specialization already proved this category can help.

Candidate fork: generate shape-specific unrolled or vectorized kernels for the
most frequent depthwise/grouped backward shapes after the stem, gated per shape.
This should be driven by timestamp rows, not by static counts alone.

### 4. Checkpointed ConvFFN Block Fusion

The tempting "fuse it all" unit is a ConvFFN residual block:

```text
pw1 expansion -> GELU -> pw2 contraction -> layer_scale + residual
```

There are many exact blocks:

- `64 -> 192 -> 64 @64x64`
- `128 -> 384 -> 128 @32x32`
- `256 -> 768 -> 256 @16x16`
- `512 -> 1536 -> 512 @8x8`

But fully fusing this block in train mode is not a simple epilogue fusion.
`pw2` needs all hidden channels after GELU. A workgroup computing a `pw2` output
tile can either read a materialized hidden tensor or recompute hidden chunks.
If it recomputes hidden chunks for each output-channel tile, it repeats `pw1`
work. For `256 -> 768 -> 256`, that can repeat the expansion for many output
tiles unless the kernel is redesigned around a larger multi-output tile.

This should be treated as a checkpointed fusion research branch:

- do not materialize the hidden post slot,
- maybe still save hidden pre slot for GELU backward, or recompute it too,
- recompute hidden chunks inside `pw2` forward/backward,
- compare extra ALU versus saved memory traffic and dispatches.

This can be a bigger leap, but it changes the compute/memory tradeoff
substantially and needs a separate fork.

### 5. Backward Chain Fusion For ConvFFN

The backward analog is:

```text
pw2_bwd -> gelu_bwd -> pw1_bwd
```

The existing `gelu_bwd + pw1_bwd` fusion is the safe inner piece. Fusing
`pw2_bwd` into that chain is harder because `pw2_bwd` produces the whole hidden
gradient consumed by GELU backward. Without a global barrier, one dispatch
cannot have all workgroups finish `pw2_bwd` before other workgroups consume the
hidden gradient. It again requires either materialization or recomputation.

So the practical next backward fusion is residual-copy-into-pw-bwd, not the
whole ConvFFN backward chain.

## Activation Saving And Backward Constraints

The strongest blocker to huge forward fusion is not code style. It is the train
contract.

Saved activations required by backward:

- GELU backward reads the pre-activation slot.
- SE backward recomputes gate internals from the saved SE input.
- Attention backward recomputes softmax from saved QKV.
- Head/loss backward reads the saved embedding.
- Pointwise and spatial backward read gradient slots whose first/accumulate
  semantics are generated by `compile_plan.py`.

The train plan mirrors activation slots into grad slots:

```text
grad(slot s) = nActSlots + s
```

This means a fused forward kernel may skip a consumer dispatch only if it still
preserves every slot backward expects, or if the variant explicitly changes the
plan to checkpoint/recompute that value. The latter is a real compiler change,
not a local WGSL tweak.

Forward-only inference can fuse more aggressively because it can reuse slots
and discard pre-activations. Our optimizer needs image gradients, so train-mode
constraints dominate.

## Two Concrete Forked Variant Designs

These should be copied/forked as explicit variant code and gated. Do not mangle
the default emitters in place.

### Variant A: `FUSE_NONPW_GELU=1`

Goal: finish the safe forward GELU producer epilogues not covered by current
`FUSE_PW_GELU`.

Fork shape:

```text
src/clip/fusion_variants/v1_nonpw_gelu/vision_wgsl.ts
```

Runtime gate:

```text
FUSE_NONPW_GELU=1
```

Implementation sketch:

- Add `spatialConvFusedGelu(s: ConvStep, gelu: GeluStep)`.
- Add `seFusedGelu(s: SeStep, gelu: GeluStep)`.
- In batch forward dispatch planning, detect exact producer->GELU pairs where
  `next.src === step.dst`.
- Like `pointwiseFusedGelu`, write both the pre slot and the post-GELU slot.
- Keep the old standalone `gelu` path as the default fallback.

Expected affected pairs:

- 2 general conv GELUs.
- 1 depthwise conv GELU.
- 3 SE GELUs.

Verification:

```bash
FUSE_NONPW_GELU=1 FUSE_PW_GELU=1 STEM_SPATIAL_BWD=1 BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
FUSE_NONPW_GELU=1 bun tools/clip/bwd_test.ts
TIMESTAMP=1 FUSE_NONPW_GELU=1 FUSE_PW_GELU=1 STEM_SPATIAL_BWD=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Promotion bar:

- Exact gradient parity passes.
- Timestamp sum and integrated 3D `CLIP` median both improve.
- If the win is only dispatch-count cleanup and does not survive integrated
  timing, keep it gated.

Why this is a good fork:

- It is exact and small.
- It extends a proven fusion pattern.
- Rollback is easy because only the dispatch planner chooses the variant.

### Variant B: `FUSE_RESIDUAL_BWD_PW=1`

Goal: fold the 22 residual-gradient copies into the adjacent pointwise backward
dispatches.

Fork shape:

```text
src/clip/fusion_variants/v2_residual_bwd_pw/vision_bwd_wgsl.ts
```

Runtime gate:

```text
FUSE_RESIDUAL_BWD_PW=1
```

Implementation sketch:

- Add `canFuseResidualBwdIntoPw(residual: ResidualBwdStep, pw: BwdStep)`.
- Requirements:
  - `pw.kind === "pw_bwd"`.
  - `residual.dY === pw.dY`.
  - `residual.n === pw.cin * pw.outH * pw.outW`.
  - current plan has `residual.accumulate === false`; require that at first.
- Add `pwBwdWithResidualCopy(residual, pw)`.
- Bind an extra residual grad destination slot.
- Inside the pointwise tile, copy `dY` to residual grad only from one output
  channel-block lane, probably `if (wid.y == 0u)`, to avoid duplicate writes.
- Preserve normal `pw_bwd` math exactly.

Expected affected pairs:

- 22 residual tails across `64/128/256/512` channel stages.

Verification:

```bash
FUSE_RESIDUAL_BWD_PW=1 FUSE_PW_GELU=1 STEM_SPATIAL_BWD=1 BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
FUSE_RESIDUAL_BWD_PW=1 bun tools/clip/bwd_test.ts
TIMESTAMP=1 FUSE_RESIDUAL_BWD_PW=1 FUSE_PW_GELU=1 STEM_SPATIAL_BWD=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Promotion bar:

- Exact gradient parity passes for B=1 and B=3.
- `residual_bwd` group mostly disappears from timestamp rows.
- `pw_bwd` does not rise enough to erase the benefit.
- Integrated 3D normal/profile/CLIP medians improve over noise.

Why this is a good fork:

- It targets 22 dispatches instead of 6.
- It removes actual gradient-slot copy traffic.
- It does not change CLIP resolution, text prompts, or optimizer semantics.

## Forking Discipline

For these fusion attempts, keep the audit trail explicit:

1. Copy the emitter being changed into a named `fusion_variants/vN_*` path or
   add a clearly named variant emitter beside the default.
2. Add a single env/config gate.
3. Run parity first, timestamp profile second, integrated 3D third.
4. Commit the gated attempt even if it loses, with the benchmark table in
   `docs/SPLAT3D_ABLATION_QUEUE.md` or a matching agent note.
5. Promote in a later commit only after integrated timing wins.

This preserves rollback and keeps default shader behavior understandable. The
current pointwise+GELU fusion was worth promoting because it followed that
shape: exact pair, clear gate/negative control, parity, and integrated timing.

## Recommendation

Do not pursue one giant CLIP shader. It is likely to be slower, harder to
compile, and harder to verify, while still needing most saved activations.

Pursue these in order:

1. `FUSE_RESIDUAL_BWD_PW=1` as the next serious fusion ablation.
2. `FUSE_NONPW_GELU=1` as a low-risk cleanup ablation.
3. Shape-specific spatial backward variants for the next timestamp-hot shapes.
4. Only then try a checkpointed ConvFFN residual-block fork, and treat it as a
   larger compiler experiment rather than a local shader fusion.

