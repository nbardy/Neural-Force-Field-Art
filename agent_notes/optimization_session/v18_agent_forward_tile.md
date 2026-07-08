# v18 Agent - Forward Pointwise Rectangular / Dual-Cout Tile Design

Date: 2026-07-08

Scope: read-only design note. I inspected the current WGSL emitters and did not
edit source code. This note describes the smallest exact-math forward pointwise
tile fork to implement next.

## Files And Functions Inspected

- `src/clip/vision_wgsl.ts`
  - `weightsDecl()`
  - `PW_TILE_DECLS`
  - `pointwiseTiledMain()`
  - `assertPointwiseTiles()`
  - private `pointwise()`
  - exported `pointwiseFusedGelu()`
- `src/clip/vision_batch_wgsl.ts`
  - `BatchDispatchOptions`
  - `forwardDispatches()`
  - `batchSpec()`
  - `batchForwardDispatches()`
  - `batchTrainDispatches()`
- `src/clip/vision_batch_pointwise.ts`
  - `pointwiseZBatchDispatch()`
  - `pointwiseSharedWBatchDispatch()`
  - `pointwiseSharedWBatchForwardDispatch()`
  - `postExpr()`
  - `actualPointwiseBuffers()`
- `src/clip/vision_bwd_wgsl.ts`
  - `pwBwd()`
  - `pwBwdAfterGelu()`
  - `pwBwdWithResidualCopy()`
- `tools/clip/dispatch_profile.ts`
- `tools/clip/pointwise_batch_bench.ts`
- `tools/clip/pointwise_batch_matrix.ts`
- `tools/clip/batch_major_train_bench.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `models/mobileclip_s0/plan_train.json`
- `experiments/clip_forks/v17_pointwise_roofline/README.md`
- `agent_notes/optimization_session/agent_pointwise_bottleneck.md`

## Current Baseline

The active forward pointwise implementation is generated in
`src/clip/vision_wgsl.ts`.

`pointwiseTiledMain()` emits the shared tiled matmul body:

```text
workgroup_size = 8 x 8 = 64 threads
tile           = 8 pixel-quads x 32 output channels
pixel coverage = 32 scalar pixels x 32 output channels
thread output  = 4 scalar pixels x 4 output channels
workgroups     = [P4 / 8, Cout / 32, 1]
```

The math is:

```text
P = H * W
P4 = P / 4
dst[co, p4] = bias[co] + sum_ci src[ci, p4] * W[ci, co]
```

With `array<vec4f>` activation bindings:

```text
src[ci * P4 + p4] = vec4f(src[ci, 4*p4+0..3])
dst[co * P4 + p4] = vec4f(dst[co, 4*p4+0..3])
```

Forward pointwise weights are transposed by the offline compiler as
`[Cin][Cout]`, with adjacent output channels packed into `vec4f`:

```text
W4((wOff + ci * Cout + co4) / 4) -> W[ci, co4..co4+3]
```

The current tile stages, per `ci0 += 32` chunk:

```text
xS: 32 ci x 8 p4         = 256 vec4f = 4096 bytes
wS: 32 ci x 8 cout-quads = 256 vec4f = 4096 bytes
total                    = 8192 bytes
```

`pointwiseFusedGelu()` reuses `pointwiseTiledMain()` and writes both the saved
pre-GELU slot and the GELU output slot. `src/clip/vision_batch_wgsl.ts` selects
that path in `forwardDispatches()` when `opts.fusePointwiseGeluForward` is set
and the next plan step is a split `gelu`.

The previous shared-W batch forward path in `src/clip/vision_batch_pointwise.ts`
is a different experiment. It puts batch lanes into `local_invocation_id.z` to
reuse W across images. It is not the same as the proposed rectangular/dual-cout
single-lane tile.

## Proposed Smallest Exact Forward Variant

Implement one forward-only variant first: `dual_cout`.

```text
baseline tile:  8 pixel-quads x 32 cout
dual_cout tile: 8 pixel-quads x 64 cout
```

This is the smallest useful rectangular tile because it keeps the same pixel
dimension and workgroup shape idea while doubling the output-channel span. It
tries to reuse the same staged X tile for two neighboring 32-channel output
tiles.

Per `ci0 += 32` chunk:

```text
xS:  32 ci x 8 p4          = 256 vec4f = 4096 bytes
wS0: 32 ci x 8 cout-quads  = 256 vec4f = 4096 bytes
wS1: 32 ci x 8 cout-quads  = 256 vec4f = 4096 bytes
total                      = 12288 bytes
```

The baseline runs two separate workgroups for the same `(p4 tile, two cout
tiles)`, so it stages the same X tile twice. `dual_cout` stages X once and two W
tiles, computes both cout halves, and stores 64 output channels.

The arithmetic is identical modulo f32 FMA order being the same as baseline
inside each channel. It should preserve exact graph semantics:

```text
for each ci:
  acc0..acc3   += W[ci, co0+0..3] * X[ci]
  acc4..acc7   += W[ci, co0+4..7] * X[ci]
  acc8..acc11  += W[ci, co1+0..3] * X[ci]
  acc12..acc15 += W[ci, co1+4..7] * X[ci]
```

In the current code's indexing language:

```text
co0 = (wid.y * 8u + lid.y) * 4u
co1 = co0 + 32u
cobase0 = wid.y * 64u
cobase1 = cobase0 + 32u
workgroups = [P4 / 8, Cout / 64, 1]
```

Each thread would own one pixel-quad and two output-channel quads for each of
the two `lid.y`-selected rows:

```text
lid.y = 0..7
co0 = base + lid.y*4
co1 = base + 32 + lid.y*4
```

That means 8 `vec4f` accumulators per thread if we compute two 4-channel groups:

```text
acc0..acc3 for co0+0..3
acc4..acc7 for co1+0..3
```

The store epilogue must be parameterized by channel expression, not only by
`j`. The current `pointwiseTiledMain()` `store(j)` callback assumes `co + j`.
For the variant, use a new emitter rather than forcing the baseline callback to
grow:

```text
pointwiseTiledDualCoutMain({
  cin,
  cout,
  P4,
  wOff,
  init: (coExpr) => ...,
  store: (coExpr, accExpr) => ...,
  extraStore?: (coExpr, accExpr) => ...
})
```

Do not change `pointwiseTiledMain()` until the variant wins.

## Exact Source Insertion Points

### `src/clip/vision_wgsl.ts`

Add new declarations beside the baseline pointwise tile:

- `PW_DUAL_COUT_TILE_DECLS`
- `pointwiseTiledDualCoutMain()`
- `pointwiseDualCout()`
- optionally `pointwiseFusedGeluDualCout()`

Recommended shape:

```ts
export function pointwiseDualCout(
  s: ConvStep,
  opts: DispatchOptions = {}
): DispatchSpec
```

and:

```ts
export function pointwiseFusedGeluDualCout(
  s: ConvStep,
  gelu: GeluStep,
  opts: DispatchOptions = {}
): DispatchSpec
```

`pointwiseDualCout()` should mirror private `pointwise()` exactly:

- same buffer order:
  - binding 0: weights
  - binding 1: source slot
  - binding 2: destination slot
  - optional binding 3: residual slot
- same `GELU` import in the generated shader
- same residual/layer-scale rule
- same `weightsDecl(0, opts.weightPrecision)`
- same label prefix strategy, e.g. `pw-dual-cout 256->768 @16x16`

`pointwiseFusedGeluDualCout()` should mirror `pointwiseFusedGelu()` exactly:

- binding 0: weights
- binding 1: source slot
- binding 2: saved pre-GELU slot
- binding 3: GELU output slot
- same exact `gelu4(acc)` expression
- label prefix e.g. `pw+gelu-dual-cout 256->768 @16x16`

Keep `assertPointwiseTiles()` unchanged and add one stricter guard:

```text
Cout % 64 == 0
```

The existing `P % 32`, `Cin % 32`, `Cout % 32`, and `wOff % 4` guarantees still
apply.

### `src/clip/vision_batch_wgsl.ts`

Extend `BatchDispatchOptions` with a forward-only step allowlist:

```ts
pointwiseDualCoutForwardSteps?: ReadonlySet<number>;
```

Then in `forwardDispatches()`, check this before the generic fused-GELU branch:

```text
if selected step is conv:pointwise and the next step is a split GELU candidate:
  emit pointwiseFusedGeluDualCout(...)
  index += 1
else if selected step is conv:pointwise:
  emit pointwiseDualCout(...)
```

Ordering matters. If `FUSE_PW_GELU=1` is active, selected `fc1` expansion steps
should use the dual-cout fused-GELU variant, not fall back to baseline
`pointwiseFusedGelu()`. If the selected step is a residual `fc2` contraction, use
the non-GELU dual-cout variant and preserve the residual epilogue.

Do not wire this through the shared-W path. `sharedWForwardSteps` and
`pointwiseDualCoutForwardSteps` should be treated as mutually exclusive for a
given step. If both allowlists contain a step, throw or prefer the new dual-cout
only in benchmark tools with an explicit warning. Silent precedence will make
results hard to read.

### `tools/clip/dispatch_profile.ts`

Add env parsing:

```text
PW_DUAL_COUT_FWD_STEPS=57,59,62,64
```

Pass it into `batchForwardDispatches()` / `batchTrainDispatches()` as
`pointwiseDualCoutForwardSteps`.

Add group labeling:

```text
pw-dual-cout -> pw-dual-cout
pw+gelu-dual-cout -> pw+gelu-dual-cout
```

This lets timestamp CSVs show whether the variant itself wins and whether total
pointwise family time improves.

### `tools/clip/batch_major_train_bench.ts`

Add the same env gate and print it in the config line:

```text
PW_DUAL_COUT_FWD_STEPS=...
```

This is the first full CLIP train timing gate.

### `tools/splat3d/step_bench.ts` and `tools/splat3d/step_matrix.ts`

Add the env passthrough for integrated optimizer timing:

```text
PW_DUAL_COUT_FWD_STEPS=...
```

For `step_matrix.ts`, add a compact config token such as:

```text
dc57-59-62-64
```

which expands to:

```text
PW_DUAL_COUT_FWD_STEPS=57,59,62,64
```

This keeps the `CONFIGS=` strings readable alongside existing `sw...`, `dw4`,
`gelubwd`, `resbwd`, `grid9`, and `cache...` tokens.

## Proposed Env Gates

Primary gate:

```text
PW_DUAL_COUT_FWD_STEPS=57,59,62,64
```

Optional mode gate if we expect more variants later:

```text
PW_FWD_TILE=dual_cout
PW_FWD_TILE_STEPS=57,59,62,64
```

I would use the explicit first gate now because the codebase already has
`SHARED_W_FWD_STEPS`, and the new path is only one variant. If we add
`dual_pixel`, `split_k`, or f16 pointwise later, migrate to the generic
`PW_FWD_TILE` naming.

Do not enable the variant by default. The existing default should stay:

```text
FUSE_PW_GELU=1
PW_DUAL_COUT_FWD_STEPS=
```

## Shape Allowlist

Start with forward-only exact math. The goal is to see whether reducing X tile
reloads helps without touching backward.

### First microbench allowlist

Use repeated high-channel shapes where `Cout % 64 == 0`:

```text
57,59,62,64
```

These are:

```text
57 256->768 @16x16 fc1, GELU candidate
59 768->256 @16x16 fc2, residual
62 256->768 @16x16 fc1, GELU candidate
64 768->256 @16x16 fc2, residual
```

Rationale:

- same static GFLOPs per step as many other ConvFFN layers;
- enough channels for a 64-cout tile;
- moderate `P4 = 64`, so baseline has enough workgroups for scheduling but not
  the huge `64x64` activation traffic of early layers;
- includes both fused-GELU expansion and residual contraction epilogues.

### Second allowlist if first wins

Expand to all repeated `256<->768 @16x16` ConvFFN steps:

```text
57,59,62,64,67,69,72,74,77,79,82,84,87,89,92,94,97,99,102,104
```

### Third allowlist if timestamp still wins

Add late attention/FFN high-channel `8x8` shapes:

```text
111,115,117,118,122,124
```

These have `P = 64`, `P4 = 16`, and large channel counts. They are a good test
for register pressure and occupancy. They might win less because there are fewer
spatial tiles, but they have high `Cin/Cout`.

### Avoid initially

Do not start with early `64x64` / `32x32` shapes:

```text
8,10,13,15,22,24,27,29,32,34,37,39,42,44,47,49
```

They are important, but the large spatial dimension may make `dual_pixel` a
better fit than `dual_cout`. Also, if `dual_cout` loses on mid/late shapes, it
probably should not be broadened to early shapes without a separate reason.

## Correctness Gates

### 1. WGSL validation

Run at least:

```bash
PW_DUAL_COUT_FWD_STEPS=57,59,62,64 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=0 bun tools/clip/dispatch_profile.ts
```

This validates all generated shaders through WebGPU pipeline creation.

### 2. Single-dispatch parity

Add or extend a compact pointwise bench to compare baseline pointwise vs
dual-cout on deterministic random buffers. The existing
`tools/clip/pointwise_batch_bench.ts` is close, but it currently compares
z-batch vs shared-W. For this variant, add a forward tile bench that compares:

```text
baseline pointwise or baseline pw+gelu
vs
dual-cout pointwise or dual-cout pw+gelu
```

Use step indices:

```text
57,59,62,64,111,115,117
```

Target:

```text
relLinf <= 1e-6 for f32 baseline-order-compatible cases
relLinf <= 1e-5 acceptable if compiler reorders FMA enough to move last bits
```

Because the reduction loop order can remain identical, large differences would
be a bug, not an expected approximation.

### 3. Existing CLIP correctness tests

Run:

```bash
PW_DUAL_COUT_FWD_STEPS=57,59,62,64 FUSE_PW_GELU=1 bun tools/clip/fused_test.ts
```

If `fused_test.ts` does not pick up the batch train path, add the new variant to
the smallest existing bench that does compare generated dispatches against
reference output. Do not rely on timing-only tests as correctness evidence.

### 4. Gradient objective gate

Even though this is forward-only exact math, the train graph uses saved forward
activations for backward. Run a full image-gradient comparison against baseline:

```bash
PW_DUAL_COUT_FWD_STEPS=57,59,62,64 FUSE_PW_GELU=1 BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
```

The benchmark should report or be extended to report:

```text
embedding cosine vs baseline >= 0.999999
input-gradient cosine vs baseline >= 0.99999
no NaN / inf
```

If the tool currently only times, add a separate deterministic gradient parity
script before promoting.

## Performance Gates

Run same-session A/B because the user already called out GPU contention.

### Isolated dispatch timestamp

Baseline:

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts > /tmp/pw_base.csv
```

Variant:

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 PW_DUAL_COUT_FWD_STEPS=57,59,62,64 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts > /tmp/pw_dual_cout.csv
```

Required before broadening:

```text
selected dual-cout dispatches >= 1.15x faster median than selected baseline dispatches
total forward pointwise family improves >= 5%
total CLIP train timestamp improves >= 2%
```

The full train timestamp gate is intentionally modest because this variant only
touches four forward dispatches at first.

### Full CLIP train bench

Baseline:

```bash
STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
```

Variant:

```bash
STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 PW_DUAL_COUT_FWD_STEPS=57,59,62,64 BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
```

Required before integrated testing:

```text
B=3 CLIP train median improves >= 2%
no worse p95 than baseline across same-session trials
```

### Integrated splat3d step bench

Baseline:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Variant:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 PW_DUAL_COUT_FWD_STEPS=57,59,62,64 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Required before promotion:

```text
overall step median improves >= 1.5%
CLIP bucket improves
raster buckets do not regress materially
```

If the result is only a tiny same-session win, keep the path as an ablation
instead of a default.

## Expected Risks

### Register pressure

`dual_cout` doubles per-thread accumulators from four `vec4f` values to eight.
That may spill on Apple Metal or reduce occupancy enough to erase the saved X
loads.

### Workgroup memory pressure

Baseline pointwise uses 8 KiB workgroup memory. `dual_cout` uses 12 KiB. That is
less risky than shared-W B=3, which reaches 16 KiB, but it still may reduce
parallel residency.

### Weights traffic doubles per workgroup

The variant loads two W tiles per workgroup. It saves X tile reloads across the
two output-channel halves. It is only a win if the baseline was meaningfully
re-reading X or if fewer workgroups improves scheduling enough. On W-bandwidth
bound shapes, it may be flat or worse.

### Fused-GELU store pressure

For `fc1` split-GELU steps, the exact train-mode fusion writes both:

```text
pre-activation slot
GELU output slot
```

`dual_cout` doubles the per-workgroup store payload. That is expected because it
covers twice as many channels, but it can still interact poorly with cache/write
combining.

### Residual epilogue reads

For `fc2` residual contractions, the variant reads residual data for 64 output
channels per workgroup. This should be equivalent total traffic to two baseline
workgroups, but the larger per-workgroup footprint may change scheduling.

### Interaction with `sharedWForwardSteps`

Both shared-W and dual-cout target forward pointwise. They should not stack in
the first implementation. A combined batch-lane plus dual-cout tile would have:

```text
xS: 256 * B vec4f
wS: 512 vec4f
```

At `B=3`, that is:

```text
xS = 12288 bytes
wS = 8192 bytes
total = 20480 bytes
```

That is too much for the current safe envelope. Do not combine them initially.

### Timing noise

The user is correct that GPU contention can distort these small deltas. The
variant must be evaluated with repeated same-session A/B, rotated config order,
and timestamp CSVs. Single-run wins should not be promoted.

## Why Not `dual_pixel` First

`dual_pixel` is also plausible:

```text
baseline tile:   8 p4 x 32 cout
dual_pixel tile: 16 p4 x 32 cout
```

It stages:

```text
xS0+xS1 = 512 vec4f = 8192 bytes
wS      = 256 vec4f = 4096 bytes
total   = 12288 bytes
```

That saves W reloads across spatial tiles and may fit early large-P layers
better. But the smallest forward implementation should start with `dual_cout`
because:

- the current code's weight layout already packs contiguous output-channel quads;
- `Cout % 64` holds for the repeated hot shapes;
- the store layout remains channel-major with the same `P4`;
- no bounds branch is needed for selected shapes;
- it directly attacks X tile reuse without changing pixel-tile addressing.

If `dual_cout` loses but parity infrastructure is good, `dual_pixel` is the next
variant to try for early `64x64` and `32x32` shapes.

## Minimal Implementation Checklist

1. Add `pointwiseTiledDualCoutMain()` and forward wrappers in
   `src/clip/vision_wgsl.ts`.
2. Export the wrappers.
3. Add `pointwiseDualCoutForwardSteps?: ReadonlySet<number>` to
   `BatchDispatchOptions`.
4. Select dual-cout wrappers in `forwardDispatches()` before baseline
   `pointwiseFusedGelu()`.
5. Add `PW_DUAL_COUT_FWD_STEPS` parsing to:
   - `tools/clip/dispatch_profile.ts`
   - `tools/clip/batch_major_train_bench.ts`
   - `tools/splat3d/step_bench.ts`
   - `tools/splat3d/step_matrix.ts`
6. Add a compact single-dispatch parity bench for baseline vs dual-cout.
7. Run gates in this order:
   - WGSL validation
   - single-dispatch parity
   - embedding and input-gradient parity
   - timestamp dispatch profile
   - B=3 CLIP train bench
   - integrated `CLIP_BATCH=3 VIEWS=3` step bench

## Decision

The smallest exact-math forward tile fork is a gated `dual_cout` emitter for
selected forward pointwise steps. It should be added beside the existing
`pointwiseTiledMain()`, not by mutating the baseline body. Start with
`PW_DUAL_COUT_FWD_STEPS=57,59,62,64`, prove parity, then broaden only if
same-session timestamp and full-chain benches survive GPU noise.
