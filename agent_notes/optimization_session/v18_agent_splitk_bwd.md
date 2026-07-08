# v18 Agent Note: Split-K `pw_bwd` Fork Design

Date: 2026-07-08

Scope: read-only inspection and design only. No source code was edited for this
note.

## Current Implementation

`pw_bwd` is generated in `src/clip/vision_bwd_wgsl.ts`.

- `PwBwdStep` is the train-plan backward step type. It stores `cin`, `cout`,
  `outH`, `outW`, `wOffT`, `dY`, `dX`, and `accumulate`.
- `pwBwd()` is at `src/clip/vision_bwd_wgsl.ts:148`.
- `pwBwd()` binds:
  - binding 0: weights
  - binding 1: `src`, the incoming gradient `dY`, layout `[Cin][P4]`
  - binding 2: `dst`, the outgoing gradient `dX`, layout `[Cout][P4]`
- It returns one dispatch with workgroups `[P4 / 8, cout / 32, 1]`.
- In batch-major mode, `batchSpec()` in `src/clip/vision_batch_wgsl.ts:158`
  injects a batch lane into `workgroup_id.z`, so the effective optimizer path is
  `[P4 / 8, cout / 32, batch]`.

The actual math is:

```text
dX[co_out, p] = sum_ci dY[ci, p] * W_T[ci, co_out]
```

Here `cin` is the reduction width. It equals the forward layer's `cout`. `cout`
is the output gradient channel count. It equals the forward layer's `cin`.
Weights are frozen, so there is no `dW`.

The tile body is shared with forward pointwise:

- `pointwiseTiledMain()` is at `src/clip/vision_wgsl.ts:188`.
- Workgroup size is `8 x 8 = 64` threads.
- One workgroup owns `8` pixel-quads by `32` output channels.
- Each thread computes four `vec4f` accumulators: four output channels by four
  pixels.
- Workgroup memory is:
  - `xS: array<vec4f, 256>`
  - `wS: array<vec4f, 256>`
  - total `8192` bytes.
- The K loop runs `ci0 += 32`.

The current store handles accumulation directly:

```text
dst[...] = acc
dst[...] = dst[...] + acc   // when s.accumulate is true
```

That is important: any split-K fork must preserve first-writer overwrite vs
later-writer add semantics.

## Why Split-K Targets Low-Spatial / High-Channel `pw_bwd`

For high spatial layers, there are already many workgroups because `P4 / 8` is
large. For low spatial layers, each dispatch has few workgroups and a long K
loop. That can underfill the GPU even though each workgroup is expensive.

Current train-plan candidates from `models/mobileclip_s0/plan_train.json`:

```text
P=64, 8x8:
index 6   512->1536 @8x8   wg/img=96   FLOPs B3=0.302G
index 8   1536->512 @8x8   wg/img=32   FLOPs B3=0.302G
index 13  1536->512 @8x8   wg/img=32   FLOPs B3=0.302G   accumulate=true
index 15  512->1536 @8x8   wg/img=96   FLOPs B3=0.302G
index 17  1536->512 @8x8   wg/img=32   FLOPs B3=0.302G
index 22  1536->512 @8x8   wg/img=32   FLOPs B3=0.302G   accumulate=true

P=256, 16x16:
10x 256->768 @16x16        wg/img=192  FLOPs B3=3.020G total
10x 768->256 @16x16        wg/img=64   FLOPs B3=3.020G total
```

The first split-K fork should target only the `P=64` shapes. The `1536->512
@8x8` case is the cleanest first target: only `32` workgroups per image, but
`48` K chunks per workgroup. With batch 3 that is `96` workgroups total, which
is plausibly too little parallelism on Apple/Metal.

The `512->1536 @8x8` case already has `96` workgroups per image and `288` at
batch 3, so it is less obviously occupancy-starved. It may still win if the
long K loop is the bottleneck, but it is not the first proof point.

## Proposed v18 Design

Do not change the current `pwBwd()` emitter. Add a gated alternate emitter and
planner path:

- `BwdDispatchOptions.splitKPwBwd?: { steps?: ReadonlySet<number>; parts: 2 | 3 | 4 | 6 | 8 }`
- Start with an env-plumbed allowlist, e.g. `SPLITK_PW_BWD_STEPS=8,13,17,22`
  and `SPLITK_PW_BWD_PARTS=4`.
- Only allow steps where:
  - `step.kind === "pw_bwd"`
  - `P <= 64` for the first fork
  - `step.cin % parts === 0`
  - `(step.cin / parts) % 32 === 0`
  - `step.cout % 32 === 0`
  - `P % 32 === 0`
- Leave `pw_bwd+gelu` and `pw_bwd+residual` fused paths out of the first fork.
  Split-K should be tested on plain `pw_bwd` first.

### Do Not Use Workgroup `z` for K Parts in the First Version

`vision_batch_wgsl.ts` currently uses `workgroup_id.z` as the batch lane after
codegen. If a split-K shader also uses `z` for K parts, the generic batch
wrapper will collide with it.

The conservative first version should return two or more ordinary dispatch
specs before batch wrapping:

1. A partial-sum dispatch with workgroups `[P4 / 8, cout / 32, parts]`.
2. A reduce dispatch with workgroups `[P4 / 8, cout / 32, 1]`.

Then provide a custom batch wrapper for these specs or make the split-K emitter
batch-aware from the start. The least invasive design is to make the split-K
emitter batch-aware and call it from `batchTrainDispatches()` before generic
`batchSpec()` is applied. That avoids reusing `z` twice.

For single-image tools/tests, emit the same shader with `batch=1`.

## Partial Buffer Design

The current `DispatchSpec.buffers` can only reference `weights`, `text`, or a
logical `slot`. A partial buffer is neither a model slot nor text. So a real
implementation needs one of these runtime changes.

Recommended: extend `BufferRef` with scratch buffers:

```ts
type BufferRef =
  | { kind: "weights" }
  | { kind: "text" }
  | { kind: "slot"; slot: number }
  | { kind: "scratch"; id: string; floats: number };
```

Runtime changes:

- `VisionTrainer` / `BatchMajorVisionTrainer` allocate one GPUBuffer per
  scratch id.
- Scratch size is `floats * batch * 4`.
- `resolve(ref)` maps `scratch` to that scratch buffer.
- Scratch buffers need `STORAGE | COPY_SRC | COPY_DST` for tests/profiling.
- Destroy scratch buffers in `destroy()`.

For one split-K `pw_bwd` step:

```text
partial layout, vec4f-indexed:

partial[
  batch,
  part,
  cout,
  P4
]

vec4f index =
  batchBase +
  part * (cout * P4) +
  channel * P4 +
  p4

batchBase = batchLane * parts * cout * P4
```

Scratch floats per step:

```text
parts * cout * P * batch
```

For `1536->512 @8x8`, `parts=4`, `batch=3`:

```text
4 * 512 * 64 * 3 floats = 393,216 floats = 1.5 MiB
```

That is acceptable for one shared scratch buffer reused across split-K steps.
Do not allocate one scratch per step if all split-K dispatches are sequential.
Instead compute a maximum required scratch size across selected steps and bind
the same scratch buffer id to every split-K pair.

Recommended scratch id:

```text
"pw_bwd_splitk_partial"
```

Maximum selected scratch size:

```text
max(parts * step.cout * P * batch)
```

## Partial Dispatch

`pwBwdSplitKPartial(step, parts, partIndex?)` should not take `partIndex` as a
uniform because the codebase bakes structure into WGSL. For code size, though,
baking one shader per part would add many pipelines. Better design:

- Use `workgroup_id.z` as `part`.
- In the batch-aware version, use `workgroup_id.z` for `batch * parts`:

```wgsl
let z = wid.z;
let batchLane = z / PARTS;
let part = z - batchLane * PARTS;
```

Effective workgroups:

```text
[P4 / 8, cout / 32, batch * parts]
```

K range:

```wgsl
let k0 = part * K_PER_PART;
let k1 = k0 + K_PER_PART;
for (var ci0 = k0; ci0 < k1; ci0 = ci0 + 32u) { ... }
```

The math inside the tile is the same as `pointwiseTiledMain()`, but the final
store goes to `partial` instead of `dst`:

```wgsl
partial[partialBase + co * P4 + p4] = acc0;
partial[partialBase + (co + 1u) * P4 + p4] = acc1;
partial[partialBase + (co + 2u) * P4 + p4] = acc2;
partial[partialBase + (co + 3u) * P4 + p4] = acc3;
```

Where:

```wgsl
let partBase = (batchLane * PARTS + part) * COUT * P4;
```

For `pw_bwd+gelu` in a later fork, the partial dispatch can reuse the current
`loadSrc` hook:

```wgsl
src[srcIndex] * geluGrad4(pre[srcIndex])
```

Do not include this in v18 until plain `pw_bwd` wins.

## Reduce Dispatch

The reduce pass reads `parts` partial sums and writes final `dst`. It must
preserve the existing `accumulate` behavior.

Workgroups:

```text
[P4 / 8, cout / 32, batch]
```

Each thread owns the same output element family as the current tile: four
output channels by one pixel-quad. It performs a short loop over parts:

```wgsl
var acc0 = vec4f(0.0);
var acc1 = vec4f(0.0);
var acc2 = vec4f(0.0);
var acc3 = vec4f(0.0);
for (var part = 0u; part < PARTS; part = part + 1u) {
  let base = (batchLane * PARTS + part) * COUT * P4;
  acc0 = acc0 + partial[base + co * P4 + p4];
  acc1 = acc1 + partial[base + (co + 1u) * P4 + p4];
  acc2 = acc2 + partial[base + (co + 2u) * P4 + p4];
  acc3 = acc3 + partial[base + (co + 3u) * P4 + p4];
}
```

Then:

```wgsl
dst[...] = accN
```

or:

```wgsl
dst[...] = dst[...] + accN
```

when `step.accumulate` is true.

This keeps the non-atomic property. No two workgroups write the same final
`dst` element in the reduce pass, and partial writers have disjoint `part`
slices.

## Dispatch Ordering

For each selected `pw_bwd`, replace one dispatch with two dispatches:

```text
splitk_partial
splitk_reduce
```

The partial and reduce dispatches can live in the same compute pass. WebGPU
guarantees dispatch order within a pass for storage buffer visibility. This
codebase already relies on that between every CLIP step.

Do not put partial and reduce into the same shader. Cross-workgroup
synchronization is impossible inside one WebGPU compute dispatch.

## Planner / Codegen Insertion Points

Likely source edits for a future implementation:

1. `src/clip/vision_wgsl.ts`
   - Extend `BufferRef` with scratch support.
   - Possibly factor `pointwiseTiledMain()` to accept `ciStart`, `ciEnd`, and
     alternate output variable names. If this makes the normal path messy, copy
     the 80-line tile body into a split-K-specific emitter for v18.

2. `src/clip/vision_bwd_wgsl.ts`
   - Extend `BwdDispatchOptions`.
   - Add `pwBwdSplitKPartial()`.
   - Add `pwBwdSplitKReduce()`.
   - Add `pwBwdSplitKDispatches()` returning two `DispatchSpec`s.
   - Modify `planBwdDispatches()` so selected `pw_bwd` entries expand to two
     specs.
   - Keep `bwdStepDispatch()` unchanged for old tests unless the test harness
     explicitly asks for split-K.

3. `src/clip/vision_batch_wgsl.ts`
   - Add a batch-aware split-K expansion before generic `batchSpec()`.
   - Avoid generic `batchSpec()` on any split-K partial that already consumes
     `workgroup_id.z`.
   - Alternatively add a `batchAware?: true` flag to `DispatchSpec` so wrappers
     know not to inject z again.

4. `src/clip/vision.ts` and `src/clip/vision_batch.ts`
   - Allocate and resolve scratch buffers.
   - For batch-major trainer, scratch size must include batch.
   - Destroy scratch buffers.

5. `tools/clip/dispatch_profile.ts`
   - Allocate scratch buffers in the profiler's `resolve()`.
   - Group labels should include `pw_bwd_splitk_partial` and
     `pw_bwd_splitk_reduce`.

6. `tools/clip/bwd_test.ts`
   - Add an isolated split-K `pw_bwd` test using the existing `testPw()` small
     random reference.
   - The current test creates a single `DispatchSpec` via `bwdStepDispatch()` at
     `tools/clip/bwd_test.ts:267`; split-K testing will need a helper that runs
     two specs against shared buffers.

7. `tools/clip/batch_major_train_bench.ts`, `tools/splat3d/step_bench.ts`,
   `tools/splat3d/step_matrix.ts`, `src/splat3d/optimize.ts`
   - Plumb env/config flags for selected split-K steps and parts.

## Correctness Gates

Do not promote from an isolated timing win. Required gates:

1. Pipeline validation
   - Every generated split-K partial/reduce shader compiles.
   - `SPLITK_PW_BWD_STEPS` with invalid shapes must fail loudly.

2. Small-shape kernel reference
   - Add `pw_bwd splitK` to `tools/clip/bwd_test.ts`.
   - Compare against the existing float64 JS analytic reference.
   - Test both `accumulate=false` and `accumulate=true`.
   - Include a shape where `parts=2` and `parts=4`.

3. Real model directional derivative
   - Run the existing full CLIP backward directional derivative gate if present.
   - If not present as a one-command tool, use `tools/clip/batch_major_train_bench.ts`
     relative input-gradient checks against baseline.

4. Input-gradient cosine
   - Compare full `dL/dimage` from baseline vs split-K for deterministic input
     and text.
   - Gate: cosine `> 0.99999`, max/rel error close to normal f32 reordering
     tolerance.
   - Expect tiny fp32 summation differences because split-K changes reduction
     order. Do not require bitwise identity.

5. End-to-end optimizer smoke
   - `CLIP_BATCH=3 VIEWS=3` one-step splat optimizer must produce finite image
     gradients and finite splat gradients.

6. Performance gates
   - Isolated timestamp: selected `pw_bwd` shape must improve by at least
     `1.20x` net including reduce pass.
   - Group timestamp: total `pw_bwd` group must improve by at least `8%`.
   - Integrated step matrix: median step time must improve by at least `5%` in
     `TRIALS>=5`, serialized GPU runs.
   - If isolated wins but integrated loses, keep as an experiment only.

Recommended commands once implemented:

```bash
SPLITK_PW_BWD_STEPS=8,13,17,22 SPLITK_PW_BWD_PARTS=4 bun tools/clip/bwd_test.ts

CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 \
  SPLITK_PW_BWD_STEPS=8,13,17,22 SPLITK_PW_BWD_PARTS=4 \
  MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts

SPLITK_PW_BWD_STEPS=8,13,17,22 SPLITK_PW_BWD_PARTS=4 \
  FUSE_PW_GELU=1 STEM_SPATIAL_BWD=1 BATCH=3 RUNS=5 WARMUP=3 \
  bun tools/clip/batch_major_train_bench.ts

TRIALS=5 CONFIGS=base=3:3,splitk=3:3:splitk8-13-17-22p4 \
  RUNS=8 WARMUP=6 bun tools/splat3d/step_matrix.ts
```

## Why It May Win on Metal/WebGPU

1. More resident workgroups for the `8x8` high-channel backward layers.
   `1536->512 @8x8` has only `32` workgroups per image. At batch 3 that is
   `96` workgroups. Split-K by 4 makes the partial pass `384` workgroups.

2. Shorter per-workgroup loops. The current `1536` reduction has `48` chunks of
   32 channels. Split-K by 4 gives `12` chunks per partial workgroup. That can
   reduce long-running workgroup tail effects and improve scheduling.

3. Lower register/live-range pressure in the partial shader. The accumulators
   are the same, but the loop lifetime is shorter and the compiler has less to
   chew through. On Metal, that can sometimes improve occupancy.

4. It keeps exact f32 CLIP math, except for reduction order. No approximation,
   no CLIP resolution change, no teacher/proxy drift.

## Why It May Lose

1. Extra global traffic. Split-K writes all partial sums and then reads them
   back. For `1536->512 @8x8`, `parts=4`, `batch=3`, that is roughly:

```text
partial write: 4 * 512 * 64 * 3 floats = 1.5 MiB
partial read:  same = 1.5 MiB
total extra:   about 3.0 MiB per dispatch
```

The baseline lower-bound traffic for one such B3 dispatch is about `36.38 MiB`,
so the extra traffic is not insane, but it is real.

2. Extra dispatch overhead. One `pw_bwd` becomes two dispatches. For tiny shapes
   or already-occupied runs, this can wipe out the gain.

3. Cache behavior may get worse. The baseline keeps accumulation inside one
   workgroup and writes final output once. Split-K materializes intermediate
   data in device memory.

4. The batch-major wrapper currently assumes it owns `workgroup_id.z`. A naive
   implementation can accidentally double-use z and silently compute wrong
   offsets. This is the main implementation risk.

5. Accumulation semantics are easy to break. Only the reduce pass should apply
   `dst + acc`. Partial passes should always overwrite their own scratch slice.

6. It may not help the larger `16x16` shapes. Those have more workgroups
   already, especially `256->768 @16x16` with `192` workgroups per image.
   Splitting those may add traffic without increasing useful occupancy.

## Recommended First Experiment

Implement the smallest useful fork:

```text
SPLITK_PW_BWD_STEPS=8,13,17,22
SPLITK_PW_BWD_PARTS=4
```

These are the four `1536->512 @8x8` backward steps. They are the most
occupancy-starved high-channel `pw_bwd` layers:

```text
8   /model/network.7/network.7.1/convffn/fc1/Conv:bwd
13  /model/network.7/network.7.1/norm/BatchNormalization:qkv:bwd  accumulate=true
17  /model/network.7/network.7.0/convffn/fc1/Conv:bwd
22  /model/network.7/network.7.0/norm/BatchNormalization:qkv:bwd  accumulate=true
```

If that wins, test:

```text
SPLITK_PW_BWD_STEPS=6,8,13,15,17,22
```

That adds the two `512->1536 @8x8` steps. If it still wins, test selected
`768->256 @16x16` steps, but do not assume the 16x16 family benefits.

## Bottom Line

Split-K `pw_bwd` is worth a fork because the late `8x8` MobileCLIP backward
layers have high K, low spatial parallelism, and nontrivial timestamp share.
The most likely winning path is not a broad rewrite; it is a tightly gated
batch-aware split-K partial/reduce pair for the four `1536->512 @8x8` backward
steps, with a reusable scratch buffer and strict input-gradient gates.
