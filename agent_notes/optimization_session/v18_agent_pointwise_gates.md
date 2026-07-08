# v18 Pointwise Variant Promotion Gate

Date: 2026-07-08

Scope: inspect the existing CLIP/pointwise benchmark and correctness tools and
define a promotion gate for the next exact pointwise variants. No implementation
edits were made for this note.

## Tools Inspected

- `tools/clip/pointwise_report.ts`
- `tools/clip/dispatch_profile.ts`
- `tools/clip/pointwise_batch_bench.ts`
- `tools/clip/pointwise_batch_matrix.ts`
- `tools/clip/batch_major_train_bench.ts`
- `tools/clip/batch_major_train_matrix.ts`
- `tools/clip/bwd_test.ts`
- `tools/clip/fused_test.ts`
- `tools/clip/f16_compare.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `experiments/clip_forks/v17_pointwise_roofline/README.md`
- `experiments/clip_forks/v02_f16_weights/README.md`
- `experiments/clip_forks/v02_f16_weights/results/2026-07-08.md`
- `docs/SPLAT3D_PERF_NOTES.md`
- `docs/CLIP_BATCHING_NOTES.md`

## Baseline Facts The Gate Should Preserve

Current CLIP resolution is `3x256x256`. Do not promote a v18 pointwise variant
by lowering CLIP resolution. If a separate low-resolution schedule is tested, it
belongs to a different gate.

Current B=3 static pointwise workload from `pointwise_report.ts`:

- forward pointwise dispatches: `48`
- backward `pw_bwd` dispatches: `48`
- pointwise+GELU forward fusion candidates: `24`
- pointwise math: `26.575 GFLOP`
- approximate staged global traffic lower bound: `3445.13 MiB`
- pointwise workgroups: `51648`

Current tile geometry:

- activations are channel-planar `vec4f` pixel quads
- pointwise weights are transposed `[Cin][Cout]`
- workgroup size is `8 x 8 = 64` threads
- tile is `8` pixel-quads by `32` output channels, so `32` pixels by `32` channels
- workgroup memory is `8192` bytes (`xS=256 vec4f` plus `wS=256 vec4f`)

The v02 f16 result is a cautionary example. It passed embedding cosine but
failed input-gradient cosine:

- embedding cosine: `0.99999559`
- input-gradient cosine: `0.97493807`
- planned input-gradient gate was `>= 0.995`

Therefore embedding parity alone is not a valid promotion gate for anything
that changes pointwise math, storage precision, or accumulation order enough to
affect `dL/dimage`.

## Candidate v18 Variants

The most plausible v18 pointwise candidates are:

1. Rectangular pointwise forward tile for selected hot shapes.
2. Rectangular or split-K `pw_bwd` for selected hot backward shapes.
3. Pointwise-specific precision fork, such as selective f16 storage, only if it
   passes the input-gradient gate.
4. Selective shared-W forward is already implemented and measured. Prior data
   says it should not be promoted unless a new shape-specific allowlist clears
   the full-chain gate.

Do not promote a blanket rewrite. The current evidence says pointwise is hot,
but previous shared-W and GELU-backward attempts show that isolated CLIP wins
can disappear inside the real 3D optimizer.

## Required Hook Before Promotion

Before a v18 implementation can be promoted, the variant must be available
through the same tools used by the existing gates:

- `tools/clip/dispatch_profile.ts`
- `tools/clip/batch_major_train_bench.ts`
- `tools/clip/batch_major_train_matrix.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`

Current matrix runners already support flags such as `stem`, `gelu`,
`gelubwd`, `resbwd`, `dw4`, and `sw...`. A new rectangular or split-K pointwise
variant should get an explicit matrix token before measuring, for example
`pwrect`, `pwbwdrect`, or `pwbwdsplitk`. One-off env flags in a single bench are
not enough for promotion because they make alternating same-session trials hard.

## Noise Controls

Use these controls for all promotion measurements:

1. Run one GPU workload at a time. Do not run browser screenshots, Chrome
   tracing, subagents with GPU tests, or parallel benchmark shells during the
   gate.
2. Compare base and variant in the same session. Do not compare v18 numbers to
   old docs unless the old command is rerun in the same gate.
3. Use process-based alternating matrices where available. The matrix tools
   reverse order on odd trials, which helps reduce thermal and background drift.
4. Record the adapter line printed by each tool.
5. Use `WARMUP >= 5` for wall-time gates unless the tool is too expensive.
6. Use `TRIALS >= 5` for CLIP-only matrix gates and `TRIALS >= 7` for the final
   integrated 3D gate.
7. Prefer medians. Also inspect min/max ranges and paired trial direction.
8. Use GPU timestamps for kernel attribution, but use wall-clock normal step
   medians for default-promotion decisions. Timestamp profiles are diagnostic;
   the app path is what matters.
9. Keep `BATCH=3`, `VIEWS=3`, `CLIP_BATCH=3`, and CLIP input resolution
   unchanged for the main promotion gate.
10. If the machine is thermally unstable or visibly under GPU contention, rerun
    rather than relaxing thresholds.

## Gate 0: Static Target Selection

Run before coding or before final gate review:

```bash
BATCH=3 TOP=14 OUT=/tmp/v18_pointwise_report_b3.md bun tools/clip/pointwise_report.ts
```

Use the report to choose explicit target shapes and step indexes. Current hot
families include:

- `256->768 @16x16`
- `768->256 @16x16`
- `128->384 @32x32`
- `384->128 @32x32`
- early `64->192 @64x64` and `192->64 @64x64` individual dispatches

Promotion should identify exactly which shapes changed and why. A variant that
touches cold pointwise shapes without improving hot ones should not move
forward.

## Gate 1: Correctness For Exact-Math Variants

For a forward pointwise variant, run ORT forward parity on the train plan:

```bash
PLAN=plan_train.json BENCH_RUNS=5 bun tools/clip/fused_test.ts
```

Required:

- all per-step checks pass at the existing `REL_TOL = 2e-3`
- final embedding cosine vs ORT is `>= 0.999`
- no NaN/Inf output

For a backward pointwise variant, run the full backward suite:

```bash
bun tools/clip/bwd_test.ts
```

Required:

- gate 1 per-kernel references all pass
- gate 2 directional derivative passes at least `7/8` directions under
  relative error `2e-2`
- no finite-gradient sanity failure

For the optimizer-relevant batch path, run lane parity:

```bash
BATCH=1 RUNS=1 WARMUP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 bun tools/clip/batch_major_train_bench.ts
BATCH=2 RUNS=1 WARMUP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 bun tools/clip/batch_major_train_bench.ts
BATCH=3 RUNS=1 WARMUP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 bun tools/clip/batch_major_train_bench.ts
```

Add the v18 flag to each command once implemented, for example
`PW_TILE_VARIANT=rect` or the equivalent matrix token.

Required:

- every lane reports `PASS grad parity`
- existing exact parity threshold holds: `cos > 0.999` and `relLinf < 2e-3`
- finite summaries remain all finite

If the variant only applies for B=3, B=1/B=2 may be negative controls rather
than active paths, but they still must not regress or fail.

## Gate 2: Correctness For Approximate Or Precision Variants

If v18 changes precision or approximation behavior, exact parity will not be
bitwise. Use f32 as teacher and require both embedding and input-gradient
quality.

For f16-style variants, start with:

```bash
bun tools/clip/pack_f16_weights.ts
STRICT=0 bun tools/clip/f16_compare.ts
```

Minimum required to keep testing:

- embedding cosine `>= 0.9995`
- input-gradient cosine `>= 0.995`
- no NaN/Inf in gradients

For default promotion of an approximate variant, use a stricter practical bar:

- input-gradient cosine `>= 0.999`, or
- input-gradient cosine `>= 0.995` plus a same-budget 3D quality gate showing
  no loss across multiple prompts/seeds

The v02 all-weight f16 path failed here, so do not reuse it as default without
new evidence.

## Gate 3: Isolated Kernel Attribution

Use GPU timestamps to prove the target family actually got faster:

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=2 bun tools/clip/dispatch_profile.ts > /tmp/v18_base_dispatch_b3.csv
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=2 <V18_FLAG>=1 bun tools/clip/dispatch_profile.ts > /tmp/v18_variant_dispatch_b3.csv
```

Also run the readable form for group summaries:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=2 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=2 <V18_FLAG>=1 bun tools/clip/dispatch_profile.ts
```

Numbers that justify continuing:

- targeted pointwise group median improves by at least `15%`
- total isolated CLIP timestamp sum improves by at least `8%`
- no non-target family regresses enough to erase more than half the target win
- dispatch count does not increase unless the timestamp win already includes
  the extra dispatches

Numbers that do not justify promotion:

- target group win below `5%`
- total isolated CLIP win below `3%`
- any correctness gate fails
- timestamp win exists only in a single run and disappears with `RUNS=3`

## Gate 4: Microbench For Shape-Specific Pointwise Ideas

For shared-W style forward variants already supported by existing tools:

```bash
BATCHES=2,3 STEPS=8,10,22,24,57,59,111,117 TRIALS=3 RUNS=30 WARMUP=10 bun tools/clip/pointwise_batch_matrix.ts
```

Required to add a shape to an allowlist:

- parity `relLinf <= 1e-6`
- median variant/base ratio `<= 0.90` for that shape and batch
- ratio is not a mixed result where one trial is a large loss

Do not promote a shape on a `0.97x` ratio. That is inside normal noise and has
already failed to translate to full-chain speed in prior shared-W work.

For new rectangular or split-K kernels, add an equivalent matrix path before
promotion. The matrix must compare the new kernel against the current z-batch
pointwise or current `pw_bwd` for the same shape, batch, and input data.

## Gate 5: Full CLIP Train Matrix

Run full CLIP train with alternating process trials:

```bash
TRIALS=5 BATCH=3 RUNS=5 WARMUP=5 CONFIGS='base=;v18=<V18_MATRIX_TOKEN>' bun tools/clip/batch_major_train_matrix.ts
```

For existing shared-W style allowlists, the token can use the current syntax:

```bash
TRIALS=5 BATCH=3 RUNS=5 WARMUP=5 CONFIGS='base=;sw=8,10,57,59' bun tools/clip/batch_major_train_matrix.ts
```

Required to continue to integrated testing:

- v18 median `batch-major` time is at least `8%` faster than base
- v18 is faster than base in at least `4/5` paired trials
- `ms/image` improves consistently with `ms/batch`
- all child parity checks pass

Promotion should be denied or kept opt-in if:

- median full CLIP train win is below `5%`
- parity passes but the min/max range overlaps so heavily that trial direction
  is ambiguous
- separate single-lane timing improves but batch-major does not

## Gate 6: Integrated 3D Optimizer Matrix

This is the default-promotion gate. Use the real optimizer path:

```bash
TRIALS=7 CONFIGS=base=3:3,v18=3:3:<V18_STEP_MATRIX_TOKEN> RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

If the variant is a default CLIP code path controlled by env rather than a
matrix token, add a matrix token first. Do not manually alternate shell commands
for final promotion unless the matrix tool cannot express the variant yet, in
which case the variant is not ready for default promotion.

Required for default promotion:

- normal step median improves by at least `8%` (`v18/base <= 0.92`)
- CLIP median improves by at least `10%`
- v18 normal step is faster in at least `5/7` paired trials
- raster, Adam, and display buckets do not regress materially
- profile total moves in the same direction as normal step median
- no change to CLIP resolution, number of rendered views, prompt schedule, or
  splat count unless the config label explicitly says so

Opt-in only:

- normal step median improves `3%` to `8%`
- CLIP median improves but total step is mostly flat
- win depends on a narrow shape allowlist that needs more prompt/quality tests

Reject or keep as research:

- normal step median improves less than `3%`
- CLIP median improves but raster/other work absorbs the gain
- any trial has a large unexplained loss
- integrated result disagrees with full CLIP matrix and cannot be explained

## Gate 7: Integrated Timestamp Smoke

After the matrix win, run timestamp attribution to make sure the speedup lands
where expected:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=3 WARMUP=2 <V18_FLAG>=1 bun tools/splat3d/step_bench.ts
```

Required:

- `clipBatch` or `clipFwd + clipBwd` decreases in line with the CLIP matrix
- total profile decreases
- raster buckets do not change unless the variant intentionally touched raster

This timestamp smoke is not enough by itself. It supports the step matrix; it
does not replace it.

## Gate 8: Build And Diff Hygiene

Run:

```bash
git diff --check
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Required:

- no whitespace errors
- production build passes
- fork notes include commands and raw results
- v18 snapshot exists before source edits if this becomes an experiment fork

## Promotion Decision Table

Promote as default only if all are true:

- exact correctness passes, or approximate precision passes the stricter
  input-gradient/quality gate
- isolated timestamp target group improves at least `15%`
- full CLIP train median improves at least `8%`
- integrated normal step median improves at least `8%`
- integrated CLIP bucket improves at least `10%`
- no CLIP resolution reduction
- no prompt/view schedule change hidden inside the pointwise variant
- build passes

Keep as opt-in if:

- correctness passes
- full CLIP improves, but integrated step improves only `3%` to `8%`
- the result is useful for screenshots or future combinations but not a clear
  default

Reject or leave as research if:

- any correctness gate fails
- input-gradient cosine fails for precision variants
- full CLIP win is below `5%`
- integrated normal step win is below `3%`
- the win only appears in a single noisy run

## Recommended v18 Measurement Order

1. `pointwise_report.ts` to choose shapes.
2. Microbench selected shapes if the variant is shape-specific.
3. `dispatch_profile.ts` with timestamps.
4. `fused_test.ts` and/or `bwd_test.ts`.
5. `batch_major_train_bench.ts` for B=1/B=2/B=3 parity.
6. `batch_major_train_matrix.ts` for full CLIP speed.
7. `step_matrix.ts` for integrated app speed.
8. `step_bench.ts TIMESTAMP=1` for final attribution.
9. `parcel build`.

This order avoids the main failure mode from earlier work: spending effort on a
kernel that is interesting in isolation but does not move the real optimizer.
