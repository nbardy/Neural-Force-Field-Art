# v03: FUSE_RESIDUAL_BWD_PW=1

Experiment note for a gated CLIP backward fusion:

```text
residual_bwd(dY -> dRes) + pw_bwd(dY -> dSrc)
```

The goal is to fold the residual gradient copy into the following pointwise
backward matmul when the two backward entries are adjacent and read the same
gradient tensor. This should remove the standalone `residual_bwd` dispatches
without changing the train plan or CLIP math.

This is an implementation fork note only. Do not promote by default until it
passes parity, timestamp, and integrated 3D timing gates.

## Current Evidence

Current `models/mobileclip_s0/plan_train.json` facts, counted from the checked
in plan:

```text
backward entries: 152
adjacent residual_bwd -> pw_bwd pairs: 22
legal same-dY pairs: 22
residual_bwd accumulate=true in those pairs: 0
pw_bwd accumulate=true in those pairs: 0
```

Current timestamp baseline, measured with:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Single-run B=3 isolated dispatch profile:

```text
Total isolated median sum: 155.714 ms

Groups:
  pw_bwd          37.290 ms   23.9%
  spatial_bwd     35.062 ms   22.5%
  pw+gelu         23.003 ms   14.8%
  pw              22.610 ms   14.5%
  conv            21.561 ms   13.8%
  residual_bwd     1.704 ms    1.1%
```

The direct dispatch-removal ceiling is roughly the current `residual_bwd`
group, about `1.7 ms` in this B=3 isolated timestamp run. A larger win is only
possible if folding the copy also improves scheduling or memory traffic around
the adjacent `pw_bwd` passes. Treat this as a clean dispatch/traffic experiment,
not a likely 2x CLIP lever by itself.

## Plan Ordering Assumptions

`tools/clip/compile_plan.py --train` emits backward entries in strict reverse
forward order. For a pointwise conv with a residual tail, it currently emits:

```python
if k == "conv" and s["variant"] == "pointwise":
    if s["residual"] is not None:
        emit_back({
            "kind": "residual_bwd",
            "dY": gslot(s["dst"]),
            "dX": gslot(s["residual"]),
            "n": s["cout"] * s["outH"] * s["outW"],
        })

    emit_back({
        "kind": "pw_bwd",
        "dY": gslot(s["dst"]),
        "dX": gslot(s["src"]),
        "wOffT": s["wOffT"],
        "cin": s["cout"],
        "cout": s["cin"],
        "outH": s["outH"],
        "outW": s["outW"],
    })
```

So the fused planner may assume:

1. The residual copy appears immediately before the pointwise backward matmul.
2. Both entries read the same `dY = grad[conv dst]`.
3. `residual_bwd.n == pw_bwd.cin * pw_bwd.outH * pw_bwd.outW`.
4. `pw_bwd.cin` means forward `cout`; `pw_bwd.cout` means forward `cin`.
5. The fused step must preserve first-writer versus accumulate semantics for
   both destination grad slots.
6. No train-plan/codegen change is required for the first experiment. The
   fusion belongs in `planBwdDispatches()`, exactly like the existing
   `gelu_bwd -> pw_bwd` gate.

Do not reorder non-adjacent entries. If a future plan inserts any backward
entry between `residual_bwd` and `pw_bwd`, skip the fusion.

## Legality Check

Add this as a new planner predicate in `src/clip/vision_bwd_wgsl.ts`:

```ts
function canFuseResidualBwdIntoPw(
  res: ResidualBwdStep,
  pw: BwdStep | undefined
): pw is PwBwdStep {
  if (!pw || pw.kind !== "pw_bwd") return false;
  if (res.dY !== pw.dY) return false;
  if (res.n !== pw.cin * pw.outH * pw.outW) return false;

  // Required to avoid read/write aliasing in one compute pass.
  if (res.dX === res.dY) return false;
  if (pw.dX === pw.dY) return false;
  if (res.dX === pw.dX) return false;

  // First fork can require the current exact-plan shape.
  if (res.accumulate || pw.accumulate) return false;

  return true;
}
```

The current plan satisfies this for all 22 adjacent pairs. Requiring
`!res.accumulate && !pw.accumulate` is intentionally conservative for v03.
After parity is proven, `pw.accumulate` can reuse the existing pointwise store
logic, and `res.accumulate` can be supported with a single-writer guarded
`resDst[srcIndex] + srcVal` copy.

## WGSL Math

Baseline math:

```text
residual_bwd:
  dRes[c, p] = dY[c, p]

pw_bwd:
  dSrc[ci, p] = sum_co dY[co, p] * W[ci, co]
```

In current backward step naming:

```text
pw.cin  = number of dY channels = forward cout
pw.cout = number of dSrc channels = forward cin
P       = pw.outH * pw.outW
P4      = P / 4
```

The pointwise kernel layout is channel-planar:

```text
src/dY:  array<vec4f> [pw.cin][P4]
dst/dSrc: array<vec4f> [pw.cout][P4]
weights: transposed training weights at pw.wOffT
```

The fused kernel should compute exactly:

```text
for c in 0 .. pw.cin:
  for p4 in 0 .. P4:
    dRes[c, p4] = dY[c, p4]

for ci in 0 .. pw.cout:
  for p4 in 0 .. P4:
    dSrc[ci, p4] = sum_co dY[co, p4] * W_T[co, ci]
```

Use the existing tiled pointwise matmul body and keep the fused behavior in a
variant emitter. The implementation can use the existing `extraStore` hook as
long as the residual write is guarded by the output-channel index:

```wgsl
if (co + j < PW_CIN) {
  resDst[(co + j) * P4 + p4] = src[(co + j) * P4 + p4];
}
```

That works because the pointwise output-channel tiles cover `pw.cout` in
disjoint 32-channel slices. For the legal pairs here `pw.cout >= pw.cin`, so
the subset `co + j < pw.cin` writes every residual channel exactly once and
does not duplicate writes across y tiles.

The relevant shape is:

```wgsl
// bindings listed below
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u,
        @builtin(local_invocation_index) li : u32) {
  let p4 = wid.x * 8u + lid.x;
  let co = (wid.y * 8u + lid.y) * 4u;
  let p4base = wid.x * 8u;
  let cobase = wid.y * 32u;

  var acc0 = vec4f(0.0);
  var acc1 = vec4f(0.0);
  var acc2 = vec4f(0.0);
  var acc3 = vec4f(0.0);

  for (var ci0 = 0u; ci0 < PW_CIN; ci0 = ci0 + 32u) {
    for (var t = li; t < 256u; t = t + 64u) {
      let ci = t >> 3u;
      let lane = t & 7u;
      let srcIndex = (ci0 + ci) * P4 + p4base + lane;
      let srcVal = src[srcIndex];

      xS[t] = srcVal;

      wS[t] = W4((W_OFF_T + (ci0 + ci) * PW_COUT + cobase + lane * 4u) / 4u);
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

  dst[co * P4 + p4] = acc0;
  if (co < PW_CIN) { resDst[co * P4 + p4] = src[co * P4 + p4]; }
  dst[(co + 1u) * P4 + p4] = acc1;
  if (co + 1u < PW_CIN) { resDst[(co + 1u) * P4 + p4] = src[(co + 1u) * P4 + p4]; }
  dst[(co + 2u) * P4 + p4] = acc2;
  if (co + 2u < PW_CIN) { resDst[(co + 2u) * P4 + p4] = src[(co + 2u) * P4 + p4]; }
  dst[(co + 3u) * P4 + p4] = acc3;
  if (co + 3u < PW_CIN) { resDst[(co + 3u) * P4 + p4] = src[(co + 3u) * P4 + p4]; }
}
```

The real emitter should bake all constants like the existing `pwBwd()` emitter:

```text
PW_CIN  -> pw.cin
PW_COUT -> pw.cout
P4      -> pw.outH * pw.outW / 4
W_OFF_T -> pw.wOffT
```

Workgroups remain identical to `pw_bwd`:

```ts
workgroups: [P4 / 8, pw.cout / 32, 1]
```

For batch-major CLIP, `batchSpec()` should lift this to:

```ts
workgroups: [P4 / 8, pw.cout / 32, batch]
```

Because the shader contains normal slot bindings, the existing batch offset
rewriter should add lane offsets for `src`, `dst`, and `resDst`.

## Buffer Bindings

Use the existing `pw_bwd` binding order plus one residual destination:

```wgsl
@group(0) @binding(0) var<storage, read> weights : array<vec4f>;
@group(0) @binding(1) var<storage, read> src : array<vec4f>;          // dY [pw.cin][P4]
@group(0) @binding(2) var<storage, read_write> dst : array<vec4f>;    // dSrc [pw.cout][P4]
@group(0) @binding(3) var<storage, read_write> resDst : array<vec4f>; // dRes [pw.cin][P4]
```

Dispatch spec buffers:

```ts
buffers: [
  { kind: "weights" },
  gradSlot(res.dY),
  gradSlot(pw.dX),
  gradSlot(res.dX),
]
```

Label recommendation:

```text
pw_bwd+residual <pw.cin>-><pw.cout> @<H>x<W>
```

Also update profiler grouping so `pw_bwd+residual` gets its own group or is
included under `pw_bwd`. For decision-making, a separate group is better:

```ts
if (label.startsWith("pw_bwd+residual")) return "pw_bwd+residual";
```

## Alias And Accumulate Hazards

### Source/destination aliasing

This fused pass reads `src = grad[conv dst]` and writes both
`dst = grad[conv src]` and `resDst = grad[residual producer]`.

Reject the fusion if any of these grad slots alias:

```text
res.dY == res.dX
pw.dY == pw.dX
res.dX == pw.dX
```

Also require `res.dY == pw.dY`, otherwise the fused kernel would copy one tensor
and multiply another.

### Duplicate residual writes

The pointwise dispatch has multiple `wid.y` output-channel tiles. A naive write
inside the staging loop races because every y tile stages the same `srcIndex`.
Guard residual writes in the epilogue with the output channel instead:

```wgsl
if (co + j < PW_CIN) {
  resDst[(co + j) * P4 + p4] = src[(co + j) * P4 + p4];
}
```

That leaves exactly one writer per `(source channel, pixel quad, batch lane)`.

### Accumulation

Current legal pairs are all non-accumulating. The implemented gate still handles
the `accumulate` flags explicitly:

```wgsl
if (co + j < PW_CIN) {
  resDst[ix] = res.accumulate ? resDst[ix] + src[ix] : src[ix];
}
```

and reuses the existing `pw_bwd` store expression:

```ts
pw.accumulate ? `dst[...] + acc${j}` : `acc${j}`
```

### Batch-major offset rewriting

`src`, `dst`, and `resDst` must be declared as top-level storage arrays with
normal `var<storage, ...>` syntax so `src/clip/vision_batch_wgsl.ts` can detect
and offset them. Avoid helper functions that hide slot array access behind
names the regex cannot find.

## Implementation Steps

1. Add a new option:

   ```ts
   export interface BwdDispatchOptions {
     stemSpatialBwd?: boolean;
     fuseGeluBwdIntoPw?: boolean;
     fuseResidualBwdIntoPw?: boolean;
   }
   ```

2. Add `canFuseResidualBwdIntoPw(res, next)` with the legality rules above.

3. Add `pwBwdWithResidual(res, pw)` beside `pwBwdAfterGelu()`.

   Prefer a dedicated emitter for v03 instead of modifying
   `pointwiseTiledMain()` globally. The emitter can copy the current
   `pointwiseTiledMain()` body and add the guarded residual copy in the staging
   loop. If the experiment wins and is promoted, consider refactoring the
   shared helper to accept a staging hook.

4. Update `planBwdDispatches()`:

   ```ts
   if (opts.fuseResidualBwdIntoPw) {
     const step = plan.backward[i];
     const next = plan.backward[i + 1];
     if (step.kind === "residual_bwd" && canFuseResidualBwdIntoPw(step, next)) {
       out.push(pwBwdWithResidual(step, next));
       i += 1;
       continue;
     }
   }
   ```

   Keep existing `gelu_bwd -> pw_bwd` fusion behavior. If both gates are enabled,
   apply the residual fusion when the current step is `residual_bwd` and GELU
   fusion when the current step is `gelu_bwd`. The two patterns should not
   overlap on the same pair in the current plan.

5. Thread the env flag through:

   ```text
   FUSE_RESIDUAL_BWD_PW=1
   ```

   Files to update in the actual fork:

   ```text
   src/clip/vision_bwd_wgsl.ts
   src/clip/vision_batch_wgsl.ts only if type plumbing requires it
   tools/clip/dispatch_profile.ts
   tools/clip/batch_major_train_bench.ts
   tools/clip/batch_major_train_matrix.ts
   tools/splat3d/step_bench.ts
   tools/splat3d/step_matrix.ts
   src/splat3d/optimize.ts only if UI/runtime config should expose the gate
   ```

6. Add a profiler group label for `pw_bwd+residual`.

7. Do not regenerate `plan_train.json` for this experiment.

## Test Commands

Baseline correctness and build:

```bash
bun tools/clip/bwd_test.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

After implementing the gate, add or update tests so this command exercises the
fused path:

```bash
FUSE_RESIDUAL_BWD_PW=1 bun tools/clip/bwd_test.ts
```

Recommended test additions:

1. A unit test for `pwBwdWithResidual()` on a small legal shape:

   ```text
   src/dY random
   weights random
   baseline residual output = src/dY
   baseline pw output = JS pw_bwd reference
   fused binding 2 must equal baseline pw output
   fused binding 3 must equal src/dY
   ```

2. A plan-level parity test:

   ```text
   run full backward with fuseResidualBwdIntoPw=false
   run full backward with fuseResidualBwdIntoPw=true
   compare dPixels and a sample of intermediate grad slots
   ```

3. Existing end-to-end directional derivative should remain within the current
   tolerance.

Dispatch profile before/after:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 FUSE_RESIDUAL_BWD_PW=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Batch-major train wall time:

```bash
STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=20 WARMUP=5 bun tools/clip/batch_major_train_bench.ts
STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 FUSE_RESIDUAL_BWD_PW=1 MODE=train BATCH=3 RUNS=20 WARMUP=5 bun tools/clip/batch_major_train_bench.ts
```

Integrated 3D step:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=3 WARMUP=1 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 FUSE_RESIDUAL_BWD_PW=1 RUNS=3 WARMUP=1 bun tools/splat3d/step_bench.ts
```

Matrix token to add:

```text
resbwd
```

Example after adding the token:

```bash
CONFIGS=base=3:3,resbwd=3:3:resbwd RUNS=5 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

## Metrics To Record

Record these in a follow-up note under this directory:

```text
dispatch count before
dispatch count after
isolated total timestamp before/after
residual_bwd group before/after
pw_bwd group before/after
pw_bwd+residual group after
B=3 batch_major_train median before/after
integrated step_bench total, clipBatch, rasterFwd, rasterBwd, adam before/after
```

Expected dispatch-count change:

```text
before: backward includes 22 residual_bwd dispatches
after:  those 22 dispatches disappear
net dispatch count: -22
```

Expected timestamp shape:

```text
residual_bwd group should go to 0 or near 0
pw_bwd+residual should be close to pw_bwd plus a small copy overhead
total isolated timestamp should improve by at least 0.8 ms B=3 to be worth more testing
```

## Rollback Criteria

Keep the gate off and revert the experiment branch if any of these happen:

```text
pipeline validation error on Apple Metal or Chrome WebGPU
any bwd_test parity failure
directional derivative regression outside existing tolerance
batch-major path fails because batch offset rewriting misses resDst
TIMESTAMP=1 total isolated B=3 changes by less than noise after repeated runs
integrated step_bench is flat or slower across repeated RUNS>=3
shader compile time or generated code size becomes problematic
```

## Promotion Criteria

Promote only if all are true:

```text
correctness: bwd_test passes with and without FUSE_RESIDUAL_BWD_PW=1
profiling: residual_bwd dispatches are removed from dispatch_profile output
profiling: repeated B=3 timestamp total improves by >= 0.8 ms or >= 0.5%
integrated: repeated 3D step_bench improves clipBatch or total step by >= 0.5%
risk: no new source/grad alias special cases are accepted silently
maintainability: fused emitter remains isolated and the default path remains readable
```

If it only removes dispatches in the isolated profiler but integrated 3D remains
flat, leave it as an off-by-default ablation gate. It is still useful evidence
that residual copies are not a major speed lever.

## Measured First Pass

Commands run after implementation:

```bash
FUSE_RESIDUAL_BWD_PW=1 BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 FUSE_RESIDUAL_BWD_PW=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
TRIALS=2 CONFIGS='base=stem,gelu;resbwd=stem,gelu,resbwd' BATCH=3 RUNS=2 WARMUP=1 bun tools/clip/batch_major_train_matrix.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 FUSE_RESIDUAL_BWD_PW=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
TRIALS=2 CONFIGS='base=3:3,resbwd=3:3:resbwd' RUNS=2 WARMUP=1 bun tools/splat3d/step_matrix.ts
```

Results:

```text
batch-major parity: PASS lanes 0, 1, 2
dispatches: 257 -> 235
isolated timestamp sum: 170.131 ms -> 164.954 ms
residual_bwd group: 1.966 ms -> removed
pw_bwd+residual group after: 22.151 ms
B=3 batch-major median: 39.75 ms -> 39.30 ms
integrated timestamp clipBatch: 39.32 ms -> 40.76 ms
integrated timestamp total: 51.05 ms -> 52.17 ms
integrated short matrix normal median: 54.54 ms -> 52.76 ms
integrated short matrix profile median: 55.97 ms -> 56.67 ms
integrated short matrix clip median: 40.09 ms -> 40.81 ms
```

Conclusion: keep `FUSE_RESIDUAL_BWD_PW=1` as an off-by-default ablation. It
removes the expected 22 dispatches and helps isolated CLIP slightly, but the
integrated profile is mixed and does not justify promotion yet.

## Key Recommendation

Implement this as a gated planner fusion, not a train-plan rewrite. The safe
kernel is a specialized `pw_bwd+residual` emitter that copies `dY` to `dRes`
from the pointwise epilogue for output channels `co + j < pw.cin`, while all y
tiles keep computing the normal pointwise backward matmul. This removes 22
dispatches with low correctness risk, but the expected speedup is small because
current `residual_bwd` is only about 1 percent of isolated B=3 CLIP timestamp
time.
