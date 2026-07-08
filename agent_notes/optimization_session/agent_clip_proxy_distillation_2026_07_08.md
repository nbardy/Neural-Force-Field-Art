# Agent Note: Proxy / Distilled CLIP Options For Browser WGPU

Date: 2026-07-08

Scope: realistic options for "distillor/proxy CLIP" in this repo. This is
documentation-only analysis. No source code was changed.

Inspected:

- `models/mobileclip_s0/config.json`
- `models/mobileclip_s0/preprocessor_config.json`
- `models/mobileclip_s0/plan.json`
- `models/mobileclip_s0/plan_train.json`
- `tools/clip/README.md`
- `tools/clip/f16_compare.ts`
- `tools/clip/dispatch_profile.ts`
- `tools/splat3d/step_bench.ts`
- `src/clip/vision.ts`
- `src/splat3d/optimize.ts`
- `src/splat3d/grid_clip.ts`
- `docs/CLIP_BATCHING_NOTES.md`
- `docs/SPLAT3D_PERF_NOTES.md`
- `docs/SPLAT3D_ABLATION_QUEUE.md`
- `experiments/clip_forks/v02_f16_weights/results/2026-07-08.md`
- `agent_notes/optimization_session/agent_pointwise_bottleneck.md`
- `agent_notes/optimization_session/agent_grid_clip_strategy.md`
- `agent_notes/optimization_session/agent_gpu_profiler_plan.md`
- `agent_notes/optimization_session/agent_fusion_bigger_leaps.md`
- `agent_notes/optimization_session/clip_2x_4x_trace_status.md`

## Executive Answer

A "proxy/distiller" in this project should not mean training CLIP weights in the
browser. The current CLIP image tower is frozen; it produces `dL/dpixels`, and
the optimizer updates Gaussian splat parameters through the rasterizer. If we
skip full CLIP on an optimizer step, we still need another differentiable signal
for the splats: a cached CLIP image-gradient, a cheaper learned proxy, or a
non-semantic surrogate/regularizer.

The most realistic 2x to 4x wall-clock path is not one replacement shader. It is
a schedule/proxy ladder:

1. Run full MobileCLIP less often per unit of useful progress.
2. Use `grid9_close2` / contact-sheet CLIP as a low-resolution all-view proxy.
3. Add cached-gradient alternate steps with decay and frequent full-CLIP
   correction.
4. Only then try a learned student/proxy image tower or gradient predictor.
5. Keep full MobileCLIP at 256x256 as the teacher/referee for promotion.

Same-resolution matters. We can preserve the MobileCLIP tensor shape
`[3,256,256]` in every option. But a 3x3 contact sheet does not preserve
per-camera 256x256 detail; each cell is currently `80x80` with black gutters.
That is still a valid proxy schedule, just not equivalent to nine independent
full-resolution camera losses.

## Current CLIP Facts That Matter

MobileCLIP-S0 in this repo is already a small model, but the train path is still
large for a browser hot loop:

- Input is `[3,256,256]`.
- Preprocessing is RGB rescale only, no mean/std normalization.
- Text embeddings are 512-d and computed only when prompts change.
- Vision forward plan has 99 dispatches and about 43.3 MB f32 weights.
- Vision train plan has 129 forward steps, 152 backward entries, 260 slots, and
  about 82.1 MB f32 weights.
- Train-plan slot storage is about 176.5 MB for B=1 and 529.5 MB for B=3,
  before the 82.1 MB train weights.
- The current batch-major B=3 path therefore has about 611.6 MB of CLIP
  weight + slot buffers before raster buffers.
- Pointwise 1x1 convs are about 88.5% of forward MACs in the parsed plan.

The current optimizer contract:

- CLIP weights are frozen.
- CLIP backward computes only `dL/dpixels`.
- Raster backward maps `dL/dpixels` into splat parameter gradients.
- Adam updates splat parameters, not CLIP parameters.

So "alternate step CLIP" means: full CLIP on some steps, and on skipped steps
optimize splats with a cheaper replacement signal. It does not mean optimizing
CLIP itself.

## Did Fp16 Already Give A Big Jump?

For CLIP vision, no. The repo now has a gated f16-weights experiment, but it was
not a promotable big speed win.

From `v02_f16_weights`:

- `weights_train.bin` went from 82.1 MB to 41.0 MB.
- Embedding cosine versus f32 was excellent: `0.99999559`.
- Input-gradient cosine versus f32 was only `0.97493807`, below the planned
  `0.995` gate.
- Isolated timestamp sum moved from `67.109 ms` f32 to `71.107 ms` f16 in the
  recorded run.
- Integrated 3D timestamp moved from `54.98 ms` total / `42.27 ms` CLIP to
  `59.90 ms` total / `47.38 ms` CLIP in the recorded run.

The "big f16 jump" memory likely applies to the older force-field advect path,
where f16 shader work had a recorded win. It does not apply to current CLIP
vision training. For CLIP, f16 remains a targeted/mixed-precision ablation, not
a solved 2x lever.

## What "Proxy / Distiller" Could Mean Here

### 1. New Faster CLIP-Like Student Image Tower

Meaning:

- Train a smaller image encoder offline to mimic current MobileCLIP-S0 image
  embeddings, prompt rankings, and ideally input gradients.
- Reuse the existing 512-d text embeddings.
- At runtime, feed the same `[3,256,256]` rendered image into the student and
  compute the same cosine loss against text.
- Hand-port the student to WGPU/WGSL or compile it into the same style of
  static dispatch plan.

What it preserves:

- Same 256x256 input tensor.
- Same text tower and prompt embeddings.
- A true semantic image loss every step, if the student is good enough.

Why it could be fast:

- A purpose-built student can be much smaller than MobileCLIP-S0 train mode.
- It can avoid the current 82 MB train-weight blob and 176 MB per-lane saved
  train slots.
- It can be architected around WGPU-friendly kernels, e.g. fewer pointwise
  blocks, smaller channel widths, and simpler backward.

Risks:

- Prompt fidelity can drop hard, especially for rare nouns, styles, negations,
  counts, and camera-angle phrases.
- A student trained only on generic images may fail on adversarial splat renders.
- A student trained only on final embeddings may match scores but produce bad
  gradients.
- A student trained only on gradients may guide optimization but lose text
  ranking fidelity.

Reality:

- This is the highest-upside model path, but it is a real ML project. It needs
  data generation, offline training, export, WGPU implementation, and strict
  gates. It is not a quick shader tweak.

### 2. Gradient Proxy / Distilled Image-Gradient Network

Meaning:

- Train a small network `g(image, text_embed) -> dL/dimage`, or
  `g(image, text_embed) -> scalar score` with backprop through the proxy.
- Teacher labels come from full MobileCLIP-S0 `dL/dpixels`.
- Runtime uses proxy gradients for most steps and full CLIP periodically as
  correction.

What it preserves:

- Can preserve `[3,256,256]` input.
- Can be trained directly on splat renders, which are the actual distribution.

Why it could be fast:

- It can be much cheaper than full CLIP forward+backward.
- It can be designed as a shallow U-Net/CNN with direct image-gradient output,
  avoiding a full contrastive embedding tower.

Risks:

- The network may learn generic saliency rather than prompt-specific semantics.
- The proxy can become stale as splat images leave its training distribution.
- If it predicts gradients directly, it may not correspond to any scalar loss,
  which can cause non-conservative update fields and weird optimizer behavior.
- If it is text-conditioned only by a 512-d embedding, it must learn how that
  embedding maps to visual features. That may need much more data than expected.

Reality:

- More plausible than a full student tower if the goal is "accelerate this art
  optimizer", not "ship a general CLIP replacement".
- Needs teacher-data tooling first.

### 3. Lower Precision

Meaning:

- Keep the same MobileCLIP-S0 architecture and weights semantically, but store
  some weights, activations, or gradients in f16.

What it preserves:

- Same model.
- Same 256x256 input.
- Same prompt semantics in theory.

What the repo already learned:

- All-weight f16 is not safe/promotable yet for CLIP: gradient cosine failed
  and integrated timing was slower in the recorded run.

Better next precision variants:

- Selective f16 only for pointwise-heavy weights.
- Keep head/loss/text/attention-sensitive pieces f32.
- Selective f16 saved activation slots, with f32 accumulators and f32 gradient
  destinations for multi-writer slots.
- Never start by changing loss dot/norm, attention reductions, or gradient
  accumulation to f16.

Reality:

- This is not a "proxy" in the semantic sense.
- It may give payload and memory wins.
- Same-model 2x to 4x from precision alone is unlikely without proving memory
  bandwidth/occupancy limits with real counters.

### 4. Lower Resolution Or Grid/Contact-Sheet Proxy

Meaning:

- Feed MobileCLIP a single 256x256 image that contains multiple camera views.
- Existing `grid9_close2` does this as lane 0: a `3x3` grid of nine downsampled
  views, plus lanes 1 and 2: two full-resolution closeups.

What it preserves:

- CLIP input shape remains `[3,256,256]`.
- All nine camera angles can influence each optimizer step.

What it does not preserve:

- Each grid cell is about `80x80`, not `256x256`.
- One CLIP embedding supervises the entire contact sheet, not each view
  independently.
- The model may optimize for "collage/contact sheet" instead of the object.

Reality:

- This is a strong schedule proxy and already fits the browser WGPU pipeline.
- It should be judged on wall-clock visual convergence, not just per-step loss.
- It is not a same-resolution substitute for nine full camera losses.

### 5. Cached Targets / Cached Gradients

Meaning:

- Run full CLIP every K steps.
- Cache the resulting `dL/dimage` for each active view.
- On intermediate steps, rerender the current splats and apply a cheap image
  objective using the cached CLIP gradient, or directly feed the cached
  `dL/dimage` into raster backward as a local linearization.

What skipped steps optimize:

- They optimize splat parameters under a stale first-order approximation of the
  full CLIP loss.
- They can also optimize non-semantic priors: alpha coverage, centering, opacity
  sparsity, background contrast, scale bounds, multi-view consistency.

What it preserves:

- Full CLIP correction still uses `[3,256,256]`.
- Skipped steps can still render at 256x256.

Risks:

- Cached gradients go stale quickly as the image changes.
- The optimizer may exploit the old gradient and move away from the prompt.
- This can amplify high-frequency artifacts unless damped.

Reality:

- This is the cheapest true proxy to try.
- It is especially attractive for "full CLIP every 2 or 4 steps, cached
  gradient in between" because it requires no new model.

### 6. Random Projections

Meaning:

- Reduce a high-dimensional teacher signal to a smaller random feature space.
- Could project final embeddings, intermediate features, or gradients.

Important limitation:

- If full MobileCLIP must run before the projection, it does not save the hot
  path.

Useful versions:

- Train a proxy/student to match random projections of teacher features instead
  of full features.
- Use random projections as a cheap distillation loss offline.
- Use projected gradients only as a diagnostic/gate, not as runtime speedup.

Reality:

- Random projection alone is not a browser speed solution.
- It is an offline distillation tool or a validation compression trick.

### 7. Non-CLIP Surrogate Losses

Meaning:

- Add cheap differentiable image/geometry losses that make the CLIP problem
  easier:
  - black background agreement;
  - alpha/coverage target;
  - opacity sparsity;
  - centered object / radius bounds;
  - depth/normal smoothness if available;
  - multi-view color/opacity consistency;
  - silhouette/edge contrast;
  - anti-fog/transmittance pressure.

What it preserves:

- Same 256x256 render can be used.
- Full CLIP remains the semantic referee.

What it does not do:

- It cannot replace prompt semantics.
- It cannot tell "cat" from "dog" without CLIP or another semantic model.

Reality:

- Very practical as alternate-step work.
- Best used to prevent CLIP hacks and improve convergence stability, not to
  claim semantic replacement.

## How To Preserve Same 256 Resolution

There are two meanings:

1. Same CLIP input resolution: MobileCLIP or the proxy receives `[3,256,256]`.
2. Same per-camera supervision resolution: each camera gets a full 256x256
   semantic loss.

Options that preserve both:

- Current per-view CLIP.
- N-of-K view sampling, when a selected view is evaluated.
- Batch-major CLIP x3 for selected full-resolution views.
- A student/proxy image tower that accepts one full-resolution camera image.
- Cached-gradient alternate steps if the cached gradient came from that same
  full-resolution view.

Options that preserve only the tensor shape:

- `grid9_close2` contact sheet.
- Any 3x3 grid prompt such as "a 3 by 3 contact sheet of the same cat from nine
  different camera angles".

For the user's stated preference, the safest default interpretation is: keep
full-resolution 256x256 full CLIP as the teacher/referee, and let proxies reduce
how often we pay that cost. Do not call grid mode equivalent to nine full-res
CLIP losses.

## Prompt Fidelity Risks

The main risk of a proxy is not just lower cosine accuracy. It is optimization
behavior. A proxy can have high embedding cosine on static images and still give
bad `dL/dpixels` for splat images.

Specific risks:

- Camera prompts: a student may ignore "right side", "top-down", "from behind",
  or "looking down" phrases.
- Background prompts: it may overfit to black background and underfit object
  identity.
- Contact-sheet prompts: it may learn "grid/collage" instead of enforcing
  multi-view consistency.
- Rare concepts: current user prompts may include styles or subjects not in the
  distillation set.
- Gradient hacking: splats can exploit proxy artifacts faster than natural
  images reveal them.
- Prompt ranking collapse: a proxy may preserve the target prompt score but not
  preserve separation from nearby prompts.

Promotion gates should therefore include:

- Final embedding cosine versus teacher.
- Input-gradient cosine versus teacher.
- Prompt-ranking agreement over distractor prompts.
- Same-wall-clock visual convergence.
- Full CLIP teacher score after proxy-optimized steps.
- Multi-view screenshots, not only loss curves.

## Concrete Ablation Ladder

### Rung 0: Baseline And Referee

Goal: make every later result comparable.

Use:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
CLIP_LAYOUT=grid9_close2 CLIP_BATCH=3 VIEWS=9 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Record:

- normal step median;
- CLIP/raster split;
- teacher CLIP score after fixed wall-clock;
- screenshots from all 9 views.

Promotion rule for any proxy: it must improve same-wall-clock teacher score or
visual quality, not only make one optimizer step faster.

### Rung 1: Schedule-Only Proxy

Variants:

- `3/9` epoch views, batch x3.
- `2/9` random views, batch x2 if supported later.
- `1/9` random views plus periodic full-grid screenshots.
- `9/9 grid9_close2`, batch x3.
- Alternating: `grid9_close2` every step, full per-view CLIP every N steps.

Why first:

- No new model.
- Keeps full CLIP teacher path intact.
- Largest near-term wall-clock lever.

Gate:

- Same wall-clock progress beats default `3/9` x3 on several prompts.
- No obvious multi-view collapse.

### Rung 2: Cached CLIP Image-Gradient Steps

Prototype:

- Every K steps, run full CLIP for selected full-resolution views.
- Cache `dL/dimage` per view.
- For K-1 steps, render the view and run raster backward using cached gradient
  scaled by a decay factor.
- Combine with cheap regularizers.

Suggested matrix:

```text
K = 2, 4
gradient decay = 1.0, 0.5
regularizers = none, weak centering+opacity
```

Gate:

- Full teacher CLIP score improves per wall-clock.
- No runaway artifacts after skipped steps.
- Similar or better 9-view screenshots after fixed wall-clock.

This is probably the fastest first proxy experiment.

### Rung 3: Surrogate Regularizer Alternate Steps

Prototype:

- Full CLIP every K steps.
- Intermediate steps use only cheap priors:
  - centered object;
  - alpha/opacity pressure;
  - black background / low background energy;
  - soft radius bounds;
  - optional smoothness or anti-fog.

Purpose:

- Not semantic replacement.
- Make full CLIP steps more effective by keeping the representation in a sane
  part of image space.

Gate:

- Same teacher CLIP cadence, better visual convergence and fewer fog/background
  hacks.

### Rung 4: Selective Mixed Precision

Prototype:

- Do not repeat all-weight f16 as the main path.
- Try selective f16 only for pointwise-heavy weights or large interior slots.
- Keep sensitive reductions, head/loss, text, input, and input gradients f32.

Gate:

- Gradient cosine versus f32 teacher at least `0.995`.
- Integrated CLIP time improves at least 10%.
- No prompt regression in fixed-step optimization.

Expected payoff:

- More likely 5% to 25% than 2x to 4x by itself.
- Useful for memory and batch stability even if not a huge speed win.

### Rung 5: Offline Teacher Dataset For Splat Renders

Before training any student, collect data:

- random splat states;
- partially optimized splat states;
- black-background and no-background prompt variants;
- all nine camera views;
- prompt set with target + distractors;
- teacher embedding, teacher score, and teacher `dL/dpixels`.

Why:

- A proxy trained on natural images may not understand splat artifacts.
- The actual optimization distribution is strange, sparse, and adversarial.

Gate:

- Dataset loader can reproduce teacher scores/gradients for held-out rendered
  images.
- Prompt-ranking and gradient labels are stored deterministically.

### Rung 6: Tiny Gradient Proxy

Prototype:

- Small CNN/U-Net-like model at 256x256.
- Inputs: image plus text embedding conditioning.
- Output: predicted `dL/dimage`, or scalar score with backprop.
- Train against teacher gradients on the splat-render dataset.

Gate:

- Held-out gradient cosine against teacher is good enough to move splats, e.g.
  start with `>= 0.8` as exploratory and tighten if promising.
- Proxy-guided optimization improves full teacher score per wall-clock when
  alternated with teacher correction.
- Does not collapse on distractor prompts.

This is a real proxy CLIP for the art optimizer. It does not need to be a
general CLIP replacement.

### Rung 7: Tiny CLIP-Like Student Tower

Prototype:

- Student image encoder outputs a 512-d embedding.
- Reuse current text embeddings.
- Train with teacher embedding cosine, prompt-ranking loss, and optional
  gradient matching.
- Export to a static WGPU plan.

Gate:

- Embedding cosine versus teacher on held-out splat renders.
- Prompt-ranking agreement over target/distractors.
- `dL/dpixels` cosine versus teacher.
- Same-wall-clock optimization wins after periodic teacher correction.

This is the cleanest long-term model replacement, but it is the largest effort.

### Rung 8: Hybrid Teacher-Student Schedule

If Rung 6 or 7 works:

- Proxy every step.
- Full MobileCLIP every N steps as correction.
- Teacher refresh resets or blends proxy gradients.
- Teacher score remains the displayed loss/referee.

Suggested schedules:

```text
proxy:teacher = 1:1
proxy:teacher = 3:1
proxy:teacher = 7:1
```

Gate:

- Same wall-clock teacher score beats full-CLIP baseline.
- Visual quality does not degrade across prompts.
- Proxy does not diverge without teacher correction for short windows.

## Recommended Next Move

Do not start with a brand-new CLIP student. Start with cached-gradient alternate
steps and the existing `grid9_close2` schedule, because they use the current
teacher and require the least new model risk.

In parallel, prepare teacher-data capture for a later tiny gradient proxy. If
that dataset shows teacher gradients are stable enough across nearby splat
states, a proxy can be trained specifically for this optimizer. If teacher
gradients are too chaotic, a student image tower is less likely to behave well
without frequent full-CLIP correction.

The north-star speed claim should be conservative:

- shader/fusion/precision work: stacked 5% to 25% improvements;
- schedule and grid proxies: plausible 2x wall-clock on useful progress;
- learned proxy with teacher correction: possible 2x to 4x, but only after
  evidence that prompt fidelity survives on splat-render data.
