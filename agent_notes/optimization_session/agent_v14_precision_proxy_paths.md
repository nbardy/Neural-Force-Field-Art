# Agent v14 Precision And Proxy Paths

Date: 2026-07-08

Scope: documentation-only inspection of the current f16 notes, MobileCLIP
trainer/runtime, grid CLIP path, and benchmark docs. No source files were
changed.

## Inspected

- `agent_notes/optimization_session/agent_fp16_reconciliation_2026_07_08.md`
- `agent_notes/optimization_session/agent_f16_reality_check.md`
- `agent_notes/optimization_session/agent_clip_proxy_distillation_2026_07_08.md`
- `agent_notes/optimization_session/clip_2x_4x_trace_status.md`
- `experiments/clip_forks/v02_f16_weights/README.md`
- `experiments/clip_forks/v02_f16_weights/results/2026-07-08.md`
- `src/clip/vision.ts`
- `src/clip/vision_wgsl.ts`
- `src/clip/vision_bwd_wgsl.ts`
- `src/clip/vision_batch.ts`
- `src/splat3d/optimize.ts`
- `src/splat3d/grid_clip.ts`
- `tools/clip/README.md`
- `tools/clip/f16_compare.ts`
- `tools/clip/pack_f16_weights.ts`
- `tools/clip/batch_major_train_bench.ts`
- `tools/splat3d/step_bench.ts`
- `docs/clip_backward_spec.md`
- `docs/CLIP_BATCHING_NOTES.md`
- `docs/SPLAT3D_PERF_NOTES.md`
- `docs/SPLAT3D_ABLATION_QUEUE.md`

`AGENTS.md` references `@RTK.md`, but `RTK.md` is not present in this repo
root.

## Short Answer

The previous CLIP fp16 experiment was a weights-only storage experiment. It did
not convert saved activations, gradients, loss math, reductions, workgroup
tiles, raster tensors, or Adam state to f16. That is why it halved model
payload but did not create a large wall-clock jump.

The current evidence says:

- all-weight f16 assets pack cleanly: `weights_train.bin` `82.1 MB` to
  `weights_train_f16.bin` `41.0 MB`;
- f16 shaders compile and run when `shader-f16` is requested;
- final embedding parity is excellent: cosine `0.99999559`;
- input-gradient parity fails the planned gate: cosine `0.97493807` versus the
  `0.995` target;
- timing is mixed and small: recorded isolated timestamp was slower
  (`67.109 ms` f32 to `71.107 ms` f16), while some later short benches show
  small wins;
- no recorded f16 result is a promotable 2x-4x CLIP or optimizer-step speedup.

The likely 2x-4x wall-clock path is not "turn on fp16". It is a stacked path:
reduce how often full MobileCLIP backward runs per useful optimizer progress,
keep full 256x256 CLIP as teacher/referee, and add targeted precision/kernel
work only where correctness and timestamps show it survives in the integrated
step.

## Three Different Things

### 1. Weight-f16

This exists today as an opt-in CLIP path.

Runtime surface:

- `WeightPrecision = "f32" | "f16"` in `src/clip/vision_wgsl.ts`
- `weightsDecl()` emits `array<vec4<f16>>`, `enable f16`, then casts `W()` and
  `W4()` back to f32 for math
- `WeightArray = Float32Array | Uint16Array` in `src/clip/vision.ts`
- `CLIP_PRECISION=f16` in `tools/splat3d/step_bench.ts`
- `PRECISION=f16` in CLIP benches
- `clipWeightPrecision` is threaded through `Splat3DOptimizer`

What it changes:

- weight storage bytes and weight load type.

What it deliberately keeps f32:

- input image;
- all activation and gradient slot buffers;
- text embedding;
- final embedding;
- input gradient `dL/dpixels`;
- softmax, loss, dot/norm, and reductions;
- pointwise/spatial accumulators;
- raster gradients and Adam state.

Why it did not jump:

- the train loop still moves large f32 slot traffic;
- `plan_train.json` has `[3,256,256]` input, 129 forward steps, 152 backward
  entries, 260 slots, and about `176.5 MB` of slot storage per B=1 lane before
  weights;
- f16-to-f32 conversion is not free;
- weight bandwidth is only part of the hot path;
- dispatch/materialization overhead and raster work remain in integrated wall
  time;
- f16 weight rounding perturbed `dL/dpixels` enough to fail the gate, even
  though the final embedding stayed close.

Use going forward:

- do not promote all-weight f16;
- try selective pointwise-heavy f16 weights only if timestamps show those
  families improve and f32-vs-selective-f16 input-gradient cosine clears
  `0.995`;
- leave stem/spatial/head/attention/SE weights f32 until each family proves
  both gradient safety and speed.

### 2. Activation-f16

This is not implemented. It is a different experiment with higher possible
memory-traffic upside and higher correctness risk.

It would require:

- per-slot dtype metadata in the train plan, not string replacement from
  `array<f32>` to `array<f16>`;
- slot allocation by bytes, not `floats * 4`;
- generated WGSL bindings per slot dtype;
- f16 load/store only at selected slot boundaries, with f32 math inside;
- diagnostic/readback helpers that know slot dtype;
- explicit handling of `accumulate:true` grad destinations.

First safe lane:

- keep these f32: input image, text embedding, loss/logit reductions, attention
  softmax reductions, final embedding, input gradient, and all multi-writer
  accumulation destinations;
- test f16 only on large interior saved activations and single-writer interior
  grad scratch after a profile identifies slot traffic as material.

Expected payoff:

- more plausible than weight-f16 for memory pressure and B=3 stability;
- still more likely to be a 10%-30% CLIP train win than a standalone 2x-4x
  wall-clock win;
- useful only if it preserves teacher gradients, not just forward embeddings.

### 3. Proxy Or Distillation

This is not precision. It changes the training signal or its cadence while full
MobileCLIP remains the 256x256 teacher.

Current optimizer contract:

- CLIP weights are frozen;
- CLIP backward produces only `dL/dpixels`;
- raster backward maps `dL/dpixels` to splat parameter gradients;
- Adam updates splat parameters, not CLIP.

So a proxy must replace or approximate `dL/dpixels` on some steps. The realistic
forms are:

- cached full-CLIP image gradients for skipped steps;
- cheap surrogate regularizers between full CLIP steps;
- grid/contact-sheet supervision as a schedule proxy;
- a learned 256x256 gradient proxy trained from full CLIP teacher labels;
- a smaller CLIP-like student image tower, if we want a real model project.

Resolution note:

- current per-view CLIP and any full-view student/proxy can preserve a
  `[3,256,256]` CLIP input and per-camera 256x256 supervision;
- `grid9_close2` also feeds CLIP a `[3,256,256]` tensor, but each of the nine
  grid cells is about `80x80`; it preserves CLIP input resolution, not
  full-resolution per-camera semantic supervision.

## Concrete Next Experiments For A 2x-4x Wall-Clock Path

### Experiment A: Baseline Referee Matrix

Purpose: make every later claim comparable.

Run current default and all-view references:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
GRID_DIRECT_RASTER=1 CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Record:

- normal step average;
- timestamp profile total;
- CLIP batch time;
- raster forward/backward split;
- same-seed screenshots from all 9 views after fixed step count and fixed wall
  time;
- full per-view teacher CLIP score, not only proxy loss.

Gate:

- future changes must beat the default `3/9`, `CLIP_BATCH=3`, 256x256 teacher
  path on same-wall-clock teacher score or clearly improve memory/headroom.

### Experiment B: Cached Full-CLIP Gradient Steps

Purpose: first real 2x-class proxy attempt without training a new model.

Design:

- run full MobileCLIP at 256x256 every K steps for selected full-resolution
  views;
- cache the resulting `dL/dimage` per view;
- for the K-1 intermediate steps, render the same 256x256 view and run raster
  backward with cached image gradient, optionally scaled by a decay;
- combine with weak cheap regularizers only after the pure cached-gradient
  version is understood.

Matrix:

```text
K = 2, 4
gradient decay = 1.0, 0.5, 0.25
views = 3/9 epoch
regularizers = none first, then weak centering+opacity
```

Why this can hit 2x-4x:

- if skipped steps are mostly raster+Adam, CLIP cadence drops by about K;
- K=2 is the likely safe first 1.5x-2x wall-clock candidate;
- K=4 is the aggressive 2x-3x candidate if stale gradients do not derail
  convergence.

Correctness gates:

- K=1 path must match current full-CLIP behavior within normal benchmark noise;
- cached gradient buffer must be finite and same shape as current
  `dL/dpixels`;
- no NaNs in raster raw grad or params over at least 100 steps;
- fixed-wall-clock full teacher score improves versus default, not just step
  time;
- 9-view screenshots do not show view collapse or high-frequency artifacts.

### Experiment C: Lower CLIP Cadence With Full-Resolution View Sampling

Purpose: reduce CLIP calls while preserving selected-view 256x256 supervision.

Variants:

```text
1/9 full-resolution epoch view every step, periodic 9-view teacher refresh
2/9 full-resolution epoch views every step, if batch x2 is worth testing
3/9 default full-resolution epoch views as control
1/9 or 2/9 plus K=2 cached-gradient intermediate steps
```

Why this can hit 2x:

- view sampling is the proven wall-clock lever because it removes CLIP work;
- the default moved from all-view to `3/9` for this reason;
- a lower cadence may still converge if epoch coverage and teacher refreshes
  prevent camera overfit.

Gate:

- compare fixed wall-clock, not fixed step count;
- require full 9-view teacher score and screenshots at checkpoints;
- require no single-view overfitting or missing-side collapse;
- keep the selected CLIP inputs at `[3,256,256]`.

### Experiment D: Grid/Contact-Sheet As A Schedule Proxy

Purpose: use one 256x256 CLIP lane to touch all nine cameras, while being
explicit that the cells are lower per-camera resolution.

Starting point:

```bash
GRID_DIRECT_RASTER=1 CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Variants:

- direct-grid on/off;
- current grid prompt versus same-text prompt;
- grid every step plus full per-view teacher refresh every N steps;
- grid plus cached-gradient full-view correction.

Why this can contribute:

- existing docs show direct-grid can make all-view grid supervision near the
  ordinary 3-view baseline in step cost;
- it provides all-camera signal every step;
- it is a schedule/proxy lever, not a faster CLIP kernel.

Gate:

- do not call it equivalent to nine full-resolution CLIP losses;
- judge by full per-view teacher score and visual convergence;
- require all nine camera screenshots after fixed wall-clock;
- reject if it optimizes for contact-sheet artifacts instead of the object.

### Experiment E: Selective Activation-F16 Storage

Purpose: test the precision lane with bigger memory-traffic upside than
weight-f16 while protecting the CLIP loss contract.

Design:

- add per-slot dtype metadata in a fork;
- leave input, text, output embedding, loss/head/attention reductions,
  `dL/dpixels`, and accumulate destinations f32;
- convert only a small allowlist of large interior single-consumer slots to f16;
- keep f32 accumulators in kernels.

Candidate selection:

- use `TIMESTAMP=1 MODE=train BATCH=3` dispatch profiles first;
- prioritize slots around hot pointwise and conv families only if their
  producer/consumer pair is simple;
- avoid loss, head, attention softmax, residual accumulation, and final input
  gradient in the first pass.

Gate:

```bash
bun tools/clip/bwd_test.ts
STRICT=0 bun tools/clip/f16_compare.ts
TIMESTAMP=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Required:

- embedding cosine versus f32 >= `0.9995`;
- input-gradient cosine versus f32 >= `0.995` for B=1 and B=3 fixtures;
- directional derivative gate still passes;
- integrated 3D CLIP time improves at least `10%`;
- 100-step same-seed optimizer smoke has finite params and at least `90%` of
  f32 teacher-score improvement.

### Experiment F: Selective Weight-F16, Not All-Weight F16

Purpose: salvage payload/cache benefits without repeating the failed global
rounding decision.

Design:

- split f32 and f16 weight buffers or emit per-family precision metadata;
- start with pointwise-heavy weights only;
- leave stem, spatial, SE, attention, head, and loss-sensitive pieces f32 until
  proven otherwise;
- keep f32 activations, gradients, and accumulators.

Gate:

- same f32-vs-selective-f16 embedding and input-gradient gates as above;
- per-family timestamp must show the converted family improves;
- integrated 3D step must improve at least `8%` to promote as speed work;
- otherwise keep only as payload/memory option.

Expected result:

- likely 0%-15% integrated improvement if it works;
- not a standalone 2x-4x path, but useful if stacked with cached/proxy cadence.

### Experiment G: Learned 256x256 Gradient Proxy

Purpose: longer-horizon 2x-4x candidate if cached gradients are too stale.

First build the teacher dataset, not the model:

- random splat states;
- partially optimized splat states;
- all nine camera views rendered at 256x256;
- target prompt plus distractor prompts;
- text embedding;
- teacher embedding, teacher score, and teacher `dL/dpixels`.

Then train a small 256x256 image/text-conditioned proxy:

- output either a scalar score with backprop or direct `dL/dimage`;
- train on splat-render distribution, not only natural images;
- alternate proxy steps with full MobileCLIP correction.

Gate:

- held-out gradient cosine versus teacher, initially exploratory at `>= 0.8`
  and tightened before promotion;
- prompt-ranking agreement against distractors;
- fixed-wall-clock full teacher score beats default;
- no collapse on rare prompts, view phrases, or black-background artifacts;
- full CLIP teacher remains the referee.

## Benchmark And Correctness Rules

Use exact parity tools for kernel/layout changes:

```bash
bun tools/clip/bwd_test.ts
BATCH=3 RUNS=3 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Use timestamp and same-session matrices for performance:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
TRIALS=3 CONFIGS=base=3:3,candidate=3:3 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

For any precision fork:

- require `shader-f16` only for f16 paths;
- fail closed in benches if unsupported;
- keep f32 as oracle and default;
- check embedding cosine, input-gradient cosine, finite outputs, and integrated
  wall-clock;
- do not promote on CLIP-only microbench wins unless integrated step wins too.

For any proxy fork:

- fixed-step speed is not enough;
- promotion requires better fixed-wall-clock full-teacher score or equivalent
  visual quality;
- always include all 9 camera screenshots;
- include distractor prompt ranking where practical;
- state whether it preserves full per-camera 256x256 supervision or only the
  CLIP tensor shape.

## Recommendation

Prioritize the 2x-4x search in this order:

1. cached full-CLIP gradient steps at K=2 and K=4;
2. lower full-resolution view cadence with periodic all-view teacher refresh;
3. grid/direct contact-sheet schedule plus full per-view referee checks;
4. selective activation-f16 only after a slot-aware fork exists;
5. selective pointwise weight-f16 as a small stacked win, not the main path;
6. learned 256x256 gradient proxy after teacher-data capture exists.

The hard line: keep full MobileCLIP at `[3,256,256]` as the correctness
referee. A path can be faster, approximate, or proxy-guided, but it should not
be promoted unless the full-resolution teacher says the optimizer improved per
wall-clock.
