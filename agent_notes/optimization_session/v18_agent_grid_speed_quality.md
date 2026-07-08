# v18 Grid80 + Pointwise Speed/Quality Separation

Date: 2026-07-08

## Task

Inspect the current 3x3 grid CLIP path and explain how to combine pointwise
CLIP speed work with the same-resolution `grid80` strategy without conflating
quality and speed. This note intentionally does not edit source code.

## Current Grid Path

`grid9_close2` is a schedule, not a CLIP-resolution change.

The CLIP model still receives `3x256x256` inputs. In grid mode the batch lanes
are:

```text
lane 0: one 256x256 contact sheet
        nine 80x80 rendered camera cells with 8px gutters

lane 1: one full-resolution 256x256 close-up view
lane 2: one full-resolution 256x256 close-up view
```

The live implementation is in:

- `src/splat3d/grid_clip.ts`
- `src/splat3d/optimize.ts`
- `src/splat3d/cameras.ts`

The lane contract is visible in `recordGrid9Close2Inputs()`:

```text
1. copy the grid prompt into CLIP batch lane 0
2. copy the two selected per-view prompts into lanes 1 and 2
3. clear lane 0's 256x256 input
4. render/copy each of the nine cells into the 3x3 contact sheet
5. render the two close-up views into lanes 1 and 2
6. run one B=3 CLIP train pass
7. scatter lane 0's contact-sheet gradient back into the nine grid views
8. apply lane 1/2 gradients to the two close-up views
```

`GRID_DIRECT_RASTER=1` / the `directgrid` token changes the raster side only:
the grid cells are rendered directly at `80x80` by a separate `Raster3DEngine`
whose cameras are prepared for `CELL=80`. Without direct grid raster, each cell
is rendered at `256x256` and then downsampled into the contact sheet. Direct
grid is therefore a real raster speed optimization, not a different CLIP input
resolution.

The prompt side is also separate. `buildGrid9Prompt()` currently describes:

```text
a 3x3 image grid showing the same subject, <prompt>, from nine different
camera angles, centered on a black background: top-down view, front-facing
view, right side view, rear view, left side view, elevated front-left view
looking down, elevated front-right view looking down, low rear-right view
looking up, and low rear-left view looking up
```

With `gridPromptMode="same"`, lane 0 instead gets the same base prompt as a
single image. That is a quality ablation, not a speed ablation.

## Current Pointwise Speed Path

Pointwise CLIP work changes how expensive one CLIP train pass is. It does not
change what supervision signal the grid schedule asks for.

In this repo, pointwise means MobileCLIP `1x1`, `groups=1` convolution:

```text
Y[p, co] = bias[co] + sum_ci X[p, ci] * W[ci, co]
```

The current pointwise WGSL layout is:

```text
activations: NCHW channel-planar, packed as vec4f pixel-quads
weights:     transposed [Cin][Cout], loaded as vec4 adjacent output channels
tile:        8 pixel-quads x 32 output channels
             = 32 pixels x 32 output channels
threads:     workgroup_size(8, 8)
shared mem:  xS 256 vec4f + wS 256 vec4f = 8192 bytes
```

Backward `pw_bwd` reuses the same tiled matmul idea with a compiler-emitted
transposed weight offset to compute `dX = W^T dY`. CLIP weights are frozen, so
there is no `dW`.

The latest v17 pointwise report showed, at B=3:

| Metric | Value |
| --- | ---: |
| forward pointwise dispatches | 48 |
| backward `pw_bwd` dispatches | 48 |
| pointwise train math | 26.575 GFLOP |
| lower-bound staged traffic | 3445.13 MiB |
| timestamp pointwise-family share | 51.6% |

So pointwise is the largest CLIP shader family, but that statement is about
the cost of the CLIP teacher implementation. It says nothing by itself about
whether `grid80` is a good supervision schedule.

## Keep These Axes Separate

There are three independent axes that have been easy to mix together:

| Axis | What Changes | Primary Gate |
| --- | --- | --- |
| CLIP kernel speed | WGSL kernels for the same MobileCLIP math | embedding and input-gradient parity, then step timing |
| Raster schedule speed | how many views are rendered and at what resolution | raster/step timing and gradient parity |
| Supervision schedule quality | which prompts/images CLIP sees per step | fixed-budget convergence against full per-view teacher scoring |

`grid80` touches the second and third axes:

- speed: fewer full-resolution CLIP lanes than full `9/9`, plus direct `80x80`
  grid raster;
- quality: one low-detail all-view contact-sheet prompt plus two full-res
  close-up prompts.

Pointwise kernels touch only the first axis:

- if exact, they should preserve the teacher exactly within normal numerical
  tolerance;
- if approximate, they become a fourth axis and must be gated separately before
  being mixed into grid quality tests.

The clean combination is:

```text
same grid schedule + same prompts + same seeds + same wall-clock gate
only change: CLIP kernel variant
```

Then, separately:

```text
same CLIP kernels
compare: base3 vs full9 vs grid80 vs grid80 plus schedule variants
```

Do not use a faster pointwise fork to claim `grid80` quality improved. It may
only make `grid80` take more optimizer steps in a fixed wall-clock budget. That
is valuable, but the interpretation is "faster exact CLIP makes this schedule
converge farther in the same time", not "the grid prompt got better".

## Recommended Gate Stack

### Stage 1 - CLIP Kernel Correctness

For any pointwise speed fork, first prove it is still the same teacher.

Required checks:

```bash
bun tools/clip/bwd_test.ts
MODE=train BATCH=1 RUNS=3 WARMUP=1 TIMESTAMP=1 bun tools/clip/dispatch_profile.ts
MODE=train BATCH=3 RUNS=3 WARMUP=1 TIMESTAMP=1 bun tools/clip/dispatch_profile.ts
```

If the fork changes precision or math approximation, add a full teacher parity
gate before any 3D training gate:

```text
embedding cosine versus baseline MobileCLIP
dL/dimage cosine versus baseline MobileCLIP
gradient norm ratio
worst-case per-image/per-prompt outliers
```

For exact f32/fused pointwise work, `bwd_test.ts` plus dispatch/timestamp
profiling is the minimum. For f16, low-rank, int4, skipped layers, or proxy
CLIP, the input-gradient gate is mandatory because splats optimize through
`dL/dimage`.

Promotion bar for pointwise speed work:

- exact correctness passes;
- B=3 CLIP timestamp improves in the intended family, not just one microkernel;
- integrated `step_matrix.ts` improves the same config against its own baseline;
- no change to `grid_quality.ts` teacher scoring unless the fork is explicitly
  an approximate/proxy teacher experiment.

### Stage 2 - Integrated Speed Matrix

After CLIP correctness, test the kernel under both the ordinary and grid
schedules. Use `step_matrix.ts`, because it runs configs sequentially and
already alternates order by trial.

Suggested matrix:

```bash
TRIALS=7 RUNS=7 WARMUP=5 \
CONFIGS=base3=3:3,base3fast=3:3:<kernel-token>,grid80=9:3:grid9:directgrid,grid80fast=9:3:grid9:directgrid:<kernel-token> \
bun tools/splat3d/step_matrix.ts
```

Replace `<kernel-token>` with whatever the fork actually exposes, for example:

```text
dw4
gelubwd
resbwd
sw10-15-24-34-49
future pointwise tile token
future split-K token
```

The important comparison pairs are:

```text
base3fast   / base3
grid80fast  / grid80
```

Only after those two pairs are understood should we compare:

```text
grid80fast / base3
```

That last number is the user-facing experience number, but it combines schedule
and kernel improvements.

### Stage 3 - Grid Quality Gate After Pointwise Changes

Use `tools/splat3d/grid_quality.ts` for quality. It is already the right shape
because it trains each schedule and then evaluates all nine camera views with
full `256x256` per-view MobileCLIP image embeddings against the real per-view
text prompts. The contact-sheet lane's own score is not the primary metric.

Baseline gate:

```bash
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat" \
CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid \
OUT_DIR=/tmp/nffa_grid_quality_cat_baseline \
bun tools/splat3d/grid_quality.ts
```

After an exact pointwise speed change:

```bash
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat" \
CONFIGS=base3=3:3:<kernel-token>,grid80=9:3:grid9:directgrid:<kernel-token> \
OUT_DIR=/tmp/nffa_grid_quality_cat_fastclip \
bun tools/splat3d/grid_quality.ts
```

If `grid_quality.ts` does not yet parse the new kernel token, add that parser
only in the implementation fork, not in this note. Keep the conceptual gate the
same.

For exact CLIP kernels, interpret the fixed-budget quality result this way:

```text
quality delta = schedule effect + extra optimizer steps from faster exact CLIP
```

To isolate schedule quality from speed, also run a fixed-step gate:

```bash
TRIALS=3 RUN_STEPS=80 PROMPT="a photo of a cat" \
CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid \
OUT_DIR=/tmp/nffa_grid_quality_cat_fixed_steps \
bun tools/splat3d/grid_quality.ts
```

Then repeat the same fixed-step gate with the pointwise variant. For exact
kernels the fixed-step quality should be statistically indistinguishable from
the baseline. If it is not, the "speed fork" changed the teacher or gradient
path and must be treated as an approximation.

### Stage 4 - More Prompts Before Default

The current recorded repeat gate is one prompt:

```text
prompt: a photo of a cat
base3:  median meanCos 0.25410, meanDelta 0.12579, 18.02 steps/s
grid80: median meanCos 0.23802, meanDelta 0.11083, 15.55 steps/s
```

Read: `grid80` captured `88.1%` of `base3`'s mean-cosine improvement and
`93.7%` of final mean cosine in the same 5s budget. That is promising but not
enough for default.

Before promoting `grid80` as a default UI path, rerun at least:

```text
a photo of a cat
a red sports car
a human face portrait
a small house
a shiny teapot
```

Keep black-background text consistent for both configs unless that is the
ablation being tested.

## How To Avoid GPU Contention Misleading Us

The user is right to be cautious: the repo has many measurements taken during
active agent work, and local GPU contention can easily turn a real 5-10% effect
into noise or reverse a small result.

Rules for speed gates:

1. Run GPU benchmarks sequentially, not in parallel.
2. Do not launch browser screenshots, Chrome traces, Bun WebGPU benches, and
   quality gates at the same time.
3. Use `step_matrix.ts` for integrated speed because it spawns one child process
   per config and alternates order across trials.
4. Prefer `TRIALS=7 RUNS=7 WARMUP=5` for promotion decisions; use shorter runs
   only for smoke tests.
5. Report medians and min/max, not single runs.
6. Keep adapter info in the note/result when available.
7. If a result is below about 5%, assume it is provisional until repeated in a
   clean session.
8. Re-run the winner and baseline adjacent in the same command, not from two
   different terminals.

Rules for quality gates:

1. Use fixed seeds and rotate config order by trial.
2. Do not compare quality runs collected while other GPU jobs are running.
3. Use both fixed-wall-clock and fixed-step tests:
   - wall-clock shows user-visible convergence speed;
   - fixed-step shows whether the schedule/teacher changed quality per step.
4. Save the JSON and contact sheets from `grid_quality.ts`.
5. Do not compare screenshots by eye unless the numeric full-view teacher gate
   also agrees.

For Chrome/Dawn/Xcode profiling:

- Chrome/Dawn traces are good for validating browser behavior and command/pass
  structure, but they should not be mixed into Bun timestamp runs.
- Xcode/Metal System Trace should be captured during one isolated browser run
  with a fixed mode selected, then the app should be closed before Bun timing.
- Treat external profiler overhead as diagnostic, not as the promotion number.
  Promotion numbers should still come from clean `step_matrix.ts` and
  `grid_quality.ts` runs.

## Combining The Work Without Conflation

The next clean fork structure should be:

```text
v18_pointwise_variant
  purpose: one exact CLIP kernel speed change
  gates: bwd_test, dispatch_profile, step_matrix base3/grid80 pairs

v19_grid_quality_after_fastclip
  purpose: same grid schedule, faster exact CLIP
  gates: grid_quality fixed-wall-clock and fixed-step, multiple prompts
```

If the pointwise change is approximate, split it instead:

```text
v18_clip_proxy_or_precision
  purpose: approximate/proxy teacher
  gates: embedding cosine, input-gradient cosine, norm ratio

v19_proxy_grid_schedule
  purpose: use proxy/faster teacher inside grid schedule
  gates: fixed-step and wall-clock grid_quality against full teacher
```

That distinction matters because exact pointwise speedups are infrastructure
wins, while proxy/precision/skip work changes the optimizer's objective.

## Practical Read

The best current product path is still:

```text
grid9_close2 + directgrid + two close-ups
```

The best current CLIP-kernel path is still:

```text
target pointwise/pw_bwd with exact variants first
```

These combine well because `grid80` still runs one B=3 CLIP train pass every
step. A faster B=3 CLIP train pass directly improves `grid80` wall-clock
convergence. But grid quality should continue to be judged by full-resolution
per-view teacher scores, not by the contact-sheet prompt score and not by raw
steps per second alone.

## Concrete Next Measurements

Run these in a clean session with no other GPU jobs:

```bash
TRIALS=7 RUNS=7 WARMUP=5 \
CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid,grid80dw4=9:3:grid9:directgrid:dw4,grid80dw4both=9:3:grid9:directgrid:dw4:gelubwd:resbwd \
bun tools/splat3d/step_matrix.ts
```

Then, for quality:

```bash
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat" \
CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid \
OUT_DIR=/tmp/nffa_grid_quality_cat_clean \
bun tools/splat3d/grid_quality.ts
```

And isolate per-step quality:

```bash
TRIALS=3 RUN_STEPS=80 PROMPT="a photo of a cat" \
CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid \
OUT_DIR=/tmp/nffa_grid_quality_cat_fixed_steps_clean \
bun tools/splat3d/grid_quality.ts
```

Only after those baselines are clean should a new pointwise fork be layered on
top. Otherwise we will not know whether an apparent improvement came from
better kernels, extra steps, different prompt supervision, or just a quieter
GPU.
