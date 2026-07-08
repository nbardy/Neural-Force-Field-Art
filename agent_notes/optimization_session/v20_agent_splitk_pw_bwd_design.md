# v20 Agent Note: Split-K `pw_bwd` Fork With Scratch Partials

Date: 2026-07-08

Scope: design only. I inspected `src/clip/vision_bwd_wgsl.ts`,
`src/clip/vision.ts`, `src/clip/vision_batch.ts`, the batch codegen/runtime,
and the CLIP/splat benchmark tools. No source code was edited.

## Current State

`pw_bwd` is emitted by `src/clip/vision_bwd_wgsl.ts`.

- `BwdDispatchOptions` currently has spatial, GELU-backward fusion, and
  residual-backward fusion flags.
- `pwBwd()` emits one dispatch per `PwBwdStep`.
- It binds:
  - binding 0: weights
  - binding 1: `src`, incoming gradient `dY`, layout `[Cin][P4]`
  - binding 2: `dst`, outgoing gradient `dX`, layout `[Cout][P4]`
- Its label is `pw_bwd ${cin}->${cout} @${outH}x${outW}` plus ` +=` when
  `accumulate` is true.
- Its workgroups are `[P4 / 8, cout / 32, 1]`.
- Accumulation semantics are applied in the final store:

```wgsl
dst[(co + j) * P4 + p4] + accj   // accumulate=true
accj                             // accumulate=false
```

The tile body is shared with forward pointwise in `src/clip/vision_wgsl.ts`.

- Workgroup size: `8 x 8 = 64` threads.
- Tile: `8` pixel-quads by `32` output channels.
- Reduction chunk: `32` channels per loop.
- Workgroup memory: `xS: array<vec4f, 256>` plus
  `wS: array<vec4f, 256>`, or `8192` bytes.
- Each thread computes four `vec4f` outputs: four adjacent output channels for
  one pixel-quad.

Batch-major train code in `src/clip/vision_batch_wgsl.ts` currently wraps normal
specs by injecting a batch lane into `workgroup_id.z` and adding per-binding
base offsets. `batchTrainDispatches()` maps every backward spec through
`batchSpec()`. A split-K partial shader also needs `workgroup_id.z`, so split-K
must bypass this generic wrapper.

Runtime buffer resolution in `src/clip/vision.ts` and `src/clip/vision_batch.ts`
currently supports only `weights`, `text`, and `slot`. A scratch partial buffer
requires extending the dispatch buffer model and all runners that compile these
specs.

## Static Target Data

`BATCH=3 TOP=24 bun tools/clip/pointwise_report.ts` reports:

```text
forward pointwise dispatches:       48
backward pw_bwd dispatches:         48
forward pointwise+GELU candidates:  24
pointwise FLOPs:                    8.858 GFLOP per image
pointwise FLOPs at B=3:             26.575 GFLOP
approx staged traffic at B=3:       3445.13 MiB
pointwise workgroups at B=3:        51648
```

The first split-K target should be the late low-spatial contraction backward
shape:

| Backward indexes | Shape | Count | Accumulate | Baseline WG/image | Baseline WG B3 | FLOPs B3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `8,13,17,22` | `1536->512 @8x8` | 4 | 2 | 32 | 384 total across four steps | 1.208G |

Each individual `1536->512 @8x8` dispatch has:

```text
P = 64
P4 = 16
cout = 512
baseline WG/image = (P4 / 8) * (cout / 32) = 2 * 16 = 32
baseline WG at B=3 = 96
baseline K chunks = 1536 / 32 = 48
```

With `parts=4`, each selected dispatch becomes:

```text
partial WG at B=3 = 2 * 16 * (3 * 4) = 384
reduce WG at B=3  = 2 * 16 * 3       = 96
K chunks/partial  = (1536 / 4) / 32  = 12
```

Secondary target only after the first gate wins:

| Backward indexes | Shape | Count | Accumulate | Baseline WG/image | Scratch B3 with p4 |
| --- | --- | ---: | ---: | ---: | ---: |
| `6,15` | `512->1536 @8x8` | 2 | 0 | 96 | 4.50 MiB |

Do not start with the `16x16` families. They have more spatial workgroups and
will pay more partial traffic before proving occupancy-starvation.

## Proposed API

Add a gated option; do not change the default path:

```ts
export interface SplitKPwBwdOptions {
  parts: 2 | 3 | 4 | 6 | 8;
  steps?: ReadonlySet<number>; // plan.backward indexes
}

export interface BwdDispatchOptions extends DispatchOptions {
  stemSpatialBwd?: boolean;
  spatialBwdVariant?: "generic" | "depthwise4";
  fuseGeluBwdIntoPw?: boolean;
  fuseResidualBwdIntoPw?: boolean;
  splitKPwBwd?: SplitKPwBwdOptions;
}
```

The `steps` set must reference `plan.backward` indexes, not final dispatch
indexes. Final dispatch indexes shift when fusions are enabled.

First benchmark configuration:

```text
SPLITK_PW_BWD_STEPS=8,13,17,22
SPLITK_PW_BWD_PARTS=4
```

Keep split-K mutually exclusive with the existing fused backward pointwise
variants in the first fork. If a selected `pw_bwd` would be consumed by
`fuseGeluBwdIntoPw` or `fuseResidualBwdIntoPw`, fail loudly or skip split-K with
a log in benchmark tools. Do not silently produce a different path.

## Scratch Buffer Model

Extend `BufferRef` with scratch:

```ts
type BufferRef =
  | { kind: "weights" }
  | { kind: "slot"; slot: number }
  | { kind: "text" }
  | { kind: "scratch"; id: string; floats: number };
```

Use one reusable scratch id:

```text
clip-pw-bwd-splitk-partials
```

Runtime allocation:

- Collect every scratch ref in the compiled spec list.
- For each `id`, allocate `max(ref.floats) * 4` bytes, rounded up to 256 bytes.
- Usage: `STORAGE | COPY_DST | COPY_SRC`.
- Label: `clip-scratch-${id}` or `clip-batch-major-scratch-${id}`.
- Destroy scratch buffers in `destroy()`.

`floats` is the exact total scalar capacity needed by that generated spec,
including batch when the spec is batch-aware.

Scratch layout is `array<vec4f>`:

```text
partial[batch][part][cout][P4]

vec4 index =
  (((batchLane * parts + part) * cout + channel) * P4 + p4)
```

Scratch floats:

```text
batch * parts * cout * P
```

Scratch examples:

| Shape | Parts | Batch | Floats | Bytes |
| --- | ---: | ---: | ---: | ---: |
| `1536->512 @8x8` | 4 | 1 | 131,072 | 512 KiB |
| `1536->512 @8x8` | 4 | 3 | 393,216 | 1.50 MiB |
| `512->1536 @8x8` | 4 | 1 | 393,216 | 1.50 MiB |
| `512->1536 @8x8` | 4 | 3 | 1,179,648 | 4.50 MiB |

For `ReplicatedBatchVisionTrainer`, scratch must be per lane. The step-major
schedule runs all lane partial dispatches before all lane reduce dispatches, so
a single shared single-lane scratch buffer would be overwritten before lane 0
reduces. Either allocate scratch as `scratchBuffers[lane][id]`, or explicitly
disallow split-K in the replicated trainer.

## Batch Offsets

The batch-major train buffers are laid out as `[batch][slotFloats]`, so split-K
must bake slot strides into the shader:

```text
srcStride4 = plan.slots[dY] / 4
dstStride4 = plan.slots[dX] / 4
scratchBatchStride4 = parts * cout * P4
scratchPartStride4 = cout * P4
```

All selected shapes must have vec4-aligned slot sizes. If
`plan.slots[dY] % 4 !== 0` or `plan.slots[dX] % 4 !== 0`, reject the fork.

Partial dispatch uses `workgroup_id.z` for both batch and K part:

```wgsl
let z = wid.z;
let batchLane = z / PARTS;
let part = z - batchLane * PARTS;
let srcBase = batchLane * SRC_STRIDE4;
let partialBase = (batchLane * PARTS + part) * COUT * P4;
```

Reduce dispatch uses `workgroup_id.z` for batch only:

```wgsl
let batchLane = wid.z;
let dstBase = batchLane * DST_STRIDE4;
let partialBatchBase = batchLane * PARTS * COUT * P4;
```

Do not run generic `batchSpec()` on split-K specs. Add a batch-aware backward
expansion in `vision_batch_wgsl.ts`:

```text
batchTrainDispatches()
  fwd = current forward logic
  bwd = loop plan.backward:
    if selected split-K pw_bwd:
      emit batch-aware partial + reduce specs
    else:
      emit bwdStepDispatch(step) through batchSpec()
```

For the single-image `VisionTrainer`, emit the same shaders with `batch=1`.

## Partial Dispatch

Label:

```text
pw_bwd_splitk.partial#${bwdIndex} p${parts} ${cin}->${cout} @${H}x${W} B${batch}
```

Buffers:

```text
0 weights
1 src dY slot
2 scratch partials
```

Workgroups:

```text
[P4 / 8, cout / 32, batch * parts]
```

WGSL tile shape: keep the current `8 x 8` tile and `32`-channel reduction
chunks. Do not introduce a new tile shape until the split-K scheduling idea is
measured.

Skeleton:

```wgsl
@group(0) @binding(1) var<storage, read> src : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> partial : array<vec4f>;

var<workgroup> xS : array<vec4f, 256>;
var<workgroup> wS : array<vec4f, 256>;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let z = wid.z;
  let batchLane = z / PARTS;
  let part = z - batchLane * PARTS;
  let p4 = wid.x * 8u + lid.x;
  let co = (wid.y * 8u + lid.y) * 4u;
  let p4base = wid.x * 8u;
  let cobase = wid.y * 32u;
  let srcBase = batchLane * SRC_STRIDE4;
  let partialBase = (batchLane * PARTS + part) * COUT * P4;
  let ciStart = part * K_PER_PART;
  let ciEnd = ciStart + K_PER_PART;

  var acc0 = vec4f(0.0);
  var acc1 = vec4f(0.0);
  var acc2 = vec4f(0.0);
  var acc3 = vec4f(0.0);

  for (var ci0 = ciStart; ci0 < ciEnd; ci0 = ci0 + 32u) {
    for (var t = li; t < 256u; t = t + 64u) {
      let ci = t >> 3u;
      let lane = t & 7u;
      let srcIndex = srcBase + (ci0 + ci) * P4 + p4base + lane;
      xS[t] = src[srcIndex];
      wS[t] = W4((WOFFT + (ci0 + ci) * COUT + cobase + lane * 4u) / 4u);
    }
    workgroupBarrier();
    for (var ci = 0u; ci < 32u; ci = ci + 1u) {
      let xv = xS[ci * 8u + lid.x];
      let wv = wS[ci * 8u + lid.y];
      acc0 = fma(vec4f(wv.x), xv, acc0);
      acc1 = fma(vec4f(wv.y), xv, acc1);
      acc2 = fma(vec4f(wv.z), xv, acc2);
      acc3 = fma(vec4f(wv.w), xv, acc3);
    }
    workgroupBarrier();
  }

  partial[partialBase + co * P4 + p4] = acc0;
  partial[partialBase + (co + 1u) * P4 + p4] = acc1;
  partial[partialBase + (co + 2u) * P4 + p4] = acc2;
  partial[partialBase + (co + 3u) * P4 + p4] = acc3;
}
```

The partial dispatch always overwrites its scratch slice. It must never read or
add `dst`; `accumulate` belongs only in the reduce dispatch.

## Reduce Dispatch

Label:

```text
pw_bwd_splitk.reduce#${bwdIndex} p${parts} ${cin}->${cout} @${H}x${W} B${batch}${accumulate ? " +=" : ""}
```

Buffers:

```text
0 scratch partials
1 dst dX slot
```

Workgroups:

```text
[P4 / 8, cout / 32, batch]
```

Skeleton:

```wgsl
@group(0) @binding(0) var<storage, read> partial : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u) {
  let batchLane = wid.z;
  let p4 = wid.x * 8u + lid.x;
  let co = (wid.y * 8u + lid.y) * 4u;
  let dstBase = batchLane * DST_STRIDE4;
  let partialBatchBase = batchLane * PARTS * COUT * P4;

  var acc0 = vec4f(0.0);
  var acc1 = vec4f(0.0);
  var acc2 = vec4f(0.0);
  var acc3 = vec4f(0.0);

  for (var part = 0u; part < PARTS; part = part + 1u) {
    let base = partialBatchBase + part * COUT * P4;
    acc0 = acc0 + partial[base + co * P4 + p4];
    acc1 = acc1 + partial[base + (co + 1u) * P4 + p4];
    acc2 = acc2 + partial[base + (co + 2u) * P4 + p4];
    acc3 = acc3 + partial[base + (co + 3u) * P4 + p4];
  }

  dst[dstBase + co * P4 + p4] = STORE0;
  dst[dstBase + (co + 1u) * P4 + p4] = STORE1;
  dst[dstBase + (co + 2u) * P4 + p4] = STORE2;
  dst[dstBase + (co + 3u) * P4 + p4] = STORE3;
}
```

Where `STOREj` is:

```text
accj
dst[dstBase + (co + j) * P4 + p4] + accj
```

depending on `step.accumulate`.

Partial and reduce can be encoded in the same compute pass. WebGPU dispatch
ordering provides storage-buffer visibility between dispatches, and this code
already relies on that for the full CLIP chain. They cannot be one dispatch
because WebGPU has no cross-workgroup barrier.

## Shape Allowlist

Hard guards before emitting split-K:

```text
step.kind === "pw_bwd"
bwdIndex in opts.splitKPwBwd.steps
P === 64
((cin === 1536 && cout === 512) || laterEnable512To1536)
parts === 4 for the first benchmark gate
cin % parts === 0
(cin / parts) % 32 === 0
cin % 32 === 0
cout % 32 === 0
P % 32 === 0
wOffT % 4 === 0
plan.slots[dY] % 4 === 0
plan.slots[dX] % 4 === 0
not consumed by a fused `pw_bwd+gelu` or `pw_bwd+residual` path
```

First allowlist:

| Backward index | Name suffix | Shape | Accumulate |
| ---: | --- | --- | --- |
| 8 | `network.7.1/convffn/fc1/Conv:bwd` | `1536->512 @8x8` | no |
| 13 | `network.7.1/norm/BatchNormalization:qkv:bwd` | `1536->512 @8x8` | yes |
| 17 | `network.7.0/convffn/fc1/Conv:bwd` | `1536->512 @8x8` | no |
| 22 | `network.7.0/norm/BatchNormalization:qkv:bwd` | `1536->512 @8x8` | yes |

Second allowlist only if the first wins:

```text
6,15  // 512->1536 @8x8
```

No broad `P <= 256` rule in the promoted path. Use exact shapes until measured
evidence says otherwise.

## Source Insertion Plan

Implementation should be staged, but the fork boundaries are:

1. `src/clip/vision_wgsl.ts`
   - Extend `BufferRef` with scratch.
   - Keep `pointwiseTiledMain()` unchanged for the first fork. Copy the tile
     body into a split-K-specific emitter if factoring would disturb the stable
     pointwise path.

2. `src/clip/vision_bwd_wgsl.ts`
   - Add `splitKPwBwd` options.
   - Add a shape guard helper keyed by `plan.backward` index.
   - Add single-image split-K emitters with `batch=1`.
   - Add exported batch-aware split-K emitters taking `plan`, `bwdIndex`,
     `step`, `parts`, and `batch`.
   - Leave `bwdStepDispatch()` unchanged for ordinary per-kernel tests.
   - In `planBwdDispatches()`, selected unfused `pw_bwd` expands to
     `[partial, reduce]`.

3. `src/clip/vision_batch_wgsl.ts`
   - Add backward loop logic instead of unconditional
     `planBwdDispatches(plan, opts).map(batchSpec)`.
   - Route selected split-K steps to batch-aware emitters.
   - Route all other specs through `batchSpec()` as today.

4. `src/clip/vision.ts`
   - Collect and allocate scratch buffers in `VisionTrainer`.
   - Resolve `scratch` refs in bind groups.
   - Destroy scratch buffers.
   - `VisionEncoder` should reject scratch, because forward-only specs should
     not need it.

5. `src/clip/vision_batch.ts`
   - Add scratch support to `BatchMajorVisionTrainer`.
   - If keeping `ReplicatedBatchVisionTrainer` compatible, allocate scratch per
     lane. Otherwise reject `splitKPwBwd` there with a clear error.

6. Benchmark/test tools
   - Add scratch allocation to every tool that builds bind groups directly:
     `tools/clip/dispatch_profile.ts` and `tools/clip/bwd_test.ts` at minimum.
   - Add env parsing/logging to the train and splat benchmark tools.

## Benchmark Tool Changes

`tools/clip/dispatch_profile.ts`

- Parse `SPLITK_PW_BWD_STEPS` and `SPLITK_PW_BWD_PARTS`.
- Pass options to `planBwdDispatches()` and `batchTrainDispatches()`.
- Allocate scratch from specs before binding.
- Add group labels:
  - `pw_bwd_splitk_partial`
  - `pw_bwd_splitk_reduce`
  - Keep normal `pw_bwd` group for unsplit pointwise backward.
- When comparing, report selected split-K net as:

```text
splitK selected net = sum(partial rows for selected steps) + sum(reduce rows for selected steps)
baseline selected net = sum(original pw_bwd rows for the same labels)
```

`tools/clip/bwd_test.ts`

- Add a `runDispatches()` helper for multi-dispatch kernels sharing buffers.
- Add isolated split-K `pw_bwd` tests for:
  - `parts=2`, `accumulate=false`
  - `parts=4`, `accumulate=false`
  - `parts=4`, `accumulate=true`
- Use shapes satisfying the normal tile rules, e.g. `P=64`, `cin` reduction
  divisible by `parts * 32`, and `cout % 32 === 0`.
- Compare against the existing float64 analytic reference. Expect fp32
  reduction-order differences, not bitwise equality.

`tools/clip/batch_major_train_bench.ts`

- Parse and log split-K envs.
- Pass split-K options into both single and batch-major trainers for same-opts
  lane parity.
- Add a baseline-vs-splitK input-gradient compare mode or a separate helper.
  Same-opts lane parity proves batch offsets; it does not prove split-K matches
  the current baseline.

`tools/clip/batch_major_train_matrix.ts`

- Add config token:

```text
splitk8-13-17-22p4
```

- Plumb to `SPLITK_PW_BWD_STEPS=8,13,17,22` and
  `SPLITK_PW_BWD_PARTS=4`.

`tools/splat3d/step_bench.ts` and `tools/splat3d/step_matrix.ts`

- Parse the same env/config token.
- Add `splitKPwBwd` to `Splat3DOptimizerConfig`.
- Pass it into both `VisionTrainer.create()` and
  `BatchMajorVisionTrainer.create()`.
- Log the selected steps and parts in the bench header.

`tools/clip/pointwise_report.ts`

- Optional but useful: when a split-K env is present, print a static split-K
  section with partial/reduce workgroups and scratch bytes for selected rows.

## Promotion Gates

Correctness gates:

1. Pipeline validation
   - Every generated partial and reduce shader compiles for `BATCH=1` and
     `BATCH=3`.
   - Invalid shapes fail loudly before shader creation.

2. Isolated kernel reference
   - `tools/clip/bwd_test.ts` split-K unit tests pass for
     `accumulate=false` and `accumulate=true`.
   - Gate: `relLinf <= 5e-4` vs float64 analytic for split-K-specific tests.

3. Full-model gradient equivalence
   - Compare baseline `dL/dimage` to split-K `dL/dimage` on deterministic image
     and text.
   - Gate: cosine `>= 0.99999`, `relLinf <= 3e-3`, all finite.

4. Directional derivative
   - Existing gate 2 in `tools/clip/bwd_test.ts` still passes:
     `8/8` directions under relative error `2e-2`.

5. Batch-major offset parity
   - `tools/clip/batch_major_train_bench.ts` passes all lanes with same opts.
   - Existing gate is `rel < 2e-3` and `cos > 0.999`; keep it.

6. Integrated optimizer smoke
   - `CLIP_BATCH=3 VIEWS=3` produces finite CLIP input gradients, finite raster
     gradients, and finite Adam-updated splat params for at least one profiled
     step.

Performance gates:

1. Dispatch profile, selected rows
   - With `TIMESTAMP=1 MODE=train BATCH=3`, the net selected
     `1536->512 @8x8` split-K rows, including reduce, must be at least `1.20x`
     faster than the same four baseline rows.

2. Dispatch profile, group level
   - Total `pw_bwd`-equivalent time
     (`pw_bwd + pw_bwd_splitk_partial + pw_bwd_splitk_reduce`) must improve by
     at least `6%`.
   - Total isolated CLIP train timestamp sum must improve by at least `3%`.

3. Full batch-major CLIP bench
   - `tools/clip/batch_major_train_matrix.ts`, `TRIALS>=5`, `RUNS>=5`,
     `WARMUP>=3`.
   - Median `batch-major` time must improve by at least `5%`.

4. Integrated splat step matrix
   - `tools/splat3d/step_matrix.ts`, `TRIALS>=5`, `RUNS>=8`, `WARMUP>=6`.
   - Median `normal` step time must improve by at least `3%`, and median
     `clip` profile time must improve by at least `5%`.
   - No promotion if profile total regresses by more than `2%` even when the
     isolated CLIP bench wins.

5. Memory ceiling
   - First promoted shape set must keep split-K scratch at or below `1.50 MiB`
     for `BATCH=3`.
   - The secondary `512->1536 @8x8` set raises scratch to `4.50 MiB`; treat that
     as a separate gate, not part of the initial promotion.

Suggested commands after implementation:

```bash
GATE=1 SPLITK_PW_BWD_STEPS=8,13,17,22 SPLITK_PW_BWD_PARTS=4 \
  bun tools/clip/bwd_test.ts

TIMESTAMP=1 CSV=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 \
  STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 \
  SPLITK_PW_BWD_STEPS=8,13,17,22 SPLITK_PW_BWD_PARTS=4 \
  bun tools/clip/dispatch_profile.ts > /tmp/clip_splitk_b3.csv

BATCH=3 RUNS=5 WARMUP=3 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 \
  SPLITK_PW_BWD_STEPS=8,13,17,22 SPLITK_PW_BWD_PARTS=4 \
  bun tools/clip/batch_major_train_bench.ts

TRIALS=5 RUNS=5 WARMUP=3 \
  CONFIGS='base=;splitk=splitk8-13-17-22p4' \
  bun tools/clip/batch_major_train_matrix.ts

TRIALS=5 RUNS=8 WARMUP=6 \
  CONFIGS='base=3:3,splitk=3:3:splitk8-13-17-22p4' \
  bun tools/splat3d/step_matrix.ts
```

## Why This Fork Is Worth Testing

The late `1536->512 @8x8` backward pointwise layers have high K and only
`32` workgroups per image. At `BATCH=3`, one dispatch has `96` workgroups and
`48` 32-channel K chunks per workgroup. Split-K by four turns that into
`384` partial workgroups with `12` K chunks, plus a small reduce pass.

The cost is an extra dispatch and partial global traffic. For the first shape
set, that traffic is only `1.50 MiB` written and `1.50 MiB` read per selected
B3 dispatch. The first fork is therefore a narrowly scoped occupancy experiment,
not a broad rewrite of pointwise backward.

## Bottom Line

Implement split-K as a gated, batch-aware two-dispatch fork for only
`plan.backward` indexes `8,13,17,22` with `parts=4`. Add scratch buffers as a
first-class dispatch binding, store all K partials into one reusable scratch
buffer, reduce them in a second dispatch that alone applies `accumulate`, and
promote only if both full-gradient correctness and integrated splat timing clear
the gates above.
