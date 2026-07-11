# 3D Splat Convergence Plan

Date: 2026-07-08

This note is about improving the optimizer's result quality, not raw loop speed.
The question is: for the same wall-clock budget or the same number of CLIP
passes, does the system form a centered, coherent, multi-view object faster?

## Main Thesis

The paper trail says the next quality wins are probably not more CLIP throughput.
They are better optimization incentives:

1. prevent background and transparency cheating;
2. keep the object centered and bounded;
3. expose CLIP to randomized but controlled views/backgrounds;
4. avoid high-frequency splat artifacts that CLIP likes too much;
5. move toward surface-biased splats after a coarse object exists.

This is the part Dream Fields, PureCLIPNeRF, Mip-Splatting, 2DGS, Gaussian
Surfels, DreamFusion, DreamView, and CLIPDraw are useful for.

## Convergence Ladder

Ranked by expected impact per implementation effort for this browser/WebGPU
system:

| Rank | Lane | Expected Effect | Implementation Cost | First Gate |
| ---: | --- | --- | --- | --- |
| 1 | Dark random-background curriculum + opacity/transmittance loss | Less fog, fewer full-frame hacks, clearer object | Low/medium | 5s quality gate and screenshots |
| 2 | Object bounds, centering, and zoom-out framing | Stops off-center and frame-filling solutions | Low | center-of-mass and full-view teacher score |
| 3 | Shared prompt plus coarse view suffixes | More semantic consistency than nine bespoke prompts | Low | compare same/camera/coarse prompts |
| 4 | Anti-alias/frequency control | Reduces tiny high-frequency CLIP exploits | Medium | scale histogram and quality gate |
| 5 | Anisotropic Gaussian parameters | More expressive shapes without jumping to surfels | Medium/high | equal-splat quality and stability |
| 6 | Surface-flattening schedule | Better real geometry and cross-view consistency | High | after object exists |
| 7 | Normal/depth consistency | Less multi-view inconsistency | High | after surface-like splats exist |
| 8 | Textured surfels/billboards | Better appearance with fewer primitives | High | late only; risky for CLIP cheating |

## Paper-Derived Lanes

### 1. Background And Opacity

Relevant papers:

- Dream Fields: random backgrounds plus transmittance regularization helped
  coherent object formation under CLIP guidance.
- DreamFusion: accumulated-alpha/opacity regularization is a recurring defense
  against filling empty space.
- PureCLIPNeRF: augmentation and anti-adversarial pressure matter under pure
  CLIP guidance.

Browser version:

- Keep black as the final display target.
- During optimization, use a dark-background curriculum:
  - early: black, charcoal, navy, very dark noise;
  - middle: occasional brighter random backgrounds;
  - final: black-only refinement.
- Add a loss on rendered alpha/transmittance so transparent fog and background
  painting are penalized.

Candidate toggles:

- `background: black | dark-random | random | curriculum`
- `alpha regularizer: off | weak | medium | strong`
- `final black tune: off | on`

What to measure:

- 5s and 15s screenshot grids;
- full-view teacher CLIP cosine;
- mean alpha coverage;
- number of pixels with near-zero alpha versus saturated alpha;
- whether the subject is visible on black at the end.

Risk:

- Too much opacity pressure can make the object vanish.
- Too much random background can fight the desired black-background art style.

### 2. Bounds, Centering, And Framing

Relevant papers:

- Dream Fields used bounded density and pose/view sampling to avoid degenerate
  geometry.
- CLIPDraw-style primitive systems benefit from constrained search spaces:
  constraints can make outputs simpler and more legible.

Browser version:

- Add a soft radial bound around the origin.
- Penalize center-of-mass drift.
- Start slightly zoomed out and ask for `centered on a black background`.
- Track whether splats are leaving the useful camera volume.

Candidate toggles:

- `bounds: off | radial weak | radial medium`
- `center loss: off | weak | medium`
- `framing: normal | zoomed-out`

What to measure:

- center-of-mass distance from origin;
- percent of opacity outside the object radius;
- min/mean CLIP cosine across the nine full views;
- screenshots for "painted the whole frame" failures.

Risk:

- Too tight a bound can prevent large or wide concepts.

### 3. Prompt And View Curriculum

Relevant papers:

- Dream Fields supports broad/random pose exposure.
- DreamFusion found view-dependent prompts useful, but used coarse azimuth
  guidance, not nine unrelated prompts.
- DreamView frames this as balancing an overall prompt with view-specific
  guidance.

Browser version:

- Default to one shared base prompt.
- Add only coarse direction words when useful:
  - `front view`
  - `side view`
  - `back view`
  - `top-down view`
- Avoid internal labels like `front-left-high`.
- Test the grid prompt variant:

```text
a grid of 9 different camera angles of the same object, the object is centered, and the object is {prompt}
```

Candidate toggles:

- `prompt mode: same | camera-natural | coarse-direction | grid-literal-v2`
- `view schedule: shuffled 3/9 | adaptive 3/9 | grid80 + 2 full | periodic full9`

What to measure:

- per-view teacher cosine;
- min view score, not only mean score;
- whether one view collapses while others improve;
- visual coherence of the 3x3 grid.

Risk:

- Over-specific per-view text may optimize nine inconsistent images instead of
  one shared 3D object.

### 4. Anti-Alias And Frequency Control

Relevant papers:

- Mip-Splatting: unconstrained Gaussian scale and sampling mismatch cause
  aliasing and unstable artifacts.
- Surface Splatting: EWA-style footprints are the classic way to make splats
  sample correctly.
- Pure CLIP optimization work warns that CLIP can reward adversarial texture.

Browser version:

- Add scale floors tied to render resolution and camera distance.
- Add a mip/EWA-inspired footprint option before CLIP sees the image.
- Penalize too many tiny, high-opacity splats.

Candidate toggles:

- `scale floor: off | screen-space | world-space`
- `filter: current | mip-ish | ewa-ish`
- `tiny opacity penalty: off | on`

What to measure:

- scale histogram over time;
- CLIP cosine versus screenshot legibility;
- whether the image becomes noisy before it becomes object-like;
- grid consistency at different camera distances.

Risk:

- Too much smoothing can slow detail formation and make all prompts look like
  soft blobs.

### 5. Representation Schedule

Relevant papers:

- 3DGS: explicit volumetric Gaussians are a forgiving optimization substrate.
- 2DGS and Gaussian Surfels: flattened, oriented splats improve surface
  coherence and normals once there is geometry to preserve.
- SuGaR and Gaussian Opacity Fields: surface alignment and opacity structure
  are useful even if mesh extraction is not the goal.
- Gaussian Billboards / GStex: textured primitives help appearance, but should
  come after geometry is stable.

Browser version:

- Phase 1: current 3D Gaussians for coarse object formation.
- Phase 2: anisotropic Gaussians with scale/rotation parameters.
- Phase 3: flatten high-opacity stable splats toward surfels.
- Phase 4: add normal/depth consistency.
- Phase 5: only then consider tiny per-splat textures.

Candidate toggles:

- `representation: isotropic | anisotropic | flatten-schedule`
- `surface loss: off | depth-distortion | normal-consistency | normal-depth`
- `texture: off | tiny-billboard`

What to measure:

- same prompt, same splat count, same wall-clock budget;
- cross-view silhouette consistency;
- depth/normal map stability if exposed;
- whether CLIP score improves by texture cheating before geometry exists.

Risk:

- Pure surfels too early may make flat billboards under CLIP-only supervision.
- Textures too early increase appearance capacity without enforcing 3D shape.

## First Ablation Pack

Do these before another CLIP kernel push if the goal is better images:

1. `black` vs `dark-random curriculum`.
2. alpha/transmittance regularizer off/weak/medium.
3. radial bound and center loss off/weak.
4. `same` prompt vs `coarse-direction` prompt vs `grid-literal-v2`.
5. zoomed-out framing with `centered on a black background`.

Keep each test screenshot-friendly. Use a fixed prompt set:

- `a photo of a cat`
- `a small red chair`
- `a skull`
- `a toy car`
- `a potted plant`

Run both short and medium budgets:

- `5s`: early convergence and screenshot story;
- `15s`: whether the regularizer keeps helping or over-constrains.

## Metrics To Add

The current CLIP score is not enough. Add these cheap readouts:

- mean and min full-view teacher cosine;
- alpha coverage percentage;
- mean rendered alpha;
- center-of-mass radius;
- opacity outside target radius;
- splat scale min/median/max;
- optional image entropy or high-frequency energy proxy.

These make convergence ablations less subjective while keeping screenshots as
the final arbiter.

## Blog Framing

Speed is "how fast the loop turns." Convergence is "whether the loop is being
rewarded for the right thing."

The clean blog story:

1. CLIP alone happily cheats.
2. Multi-view splats make the cheating visible.
3. Background and opacity regularization reduce fog/background hacks.
4. Bounds and centering make the subject form in the shared 3D volume.
5. Surface-biased splats are the next geometry chapter after coarse objects
   appear.

## Recommendation

The highest-priority convergence change is:

```text
dark-background curriculum + weak alpha/transmittance loss + weak center/radius loss
```

That is the most direct translation of the paper evidence into this app. It is
also screenshot-friendly and easier to ablate than anisotropic/surfel geometry.

## Implementation Checkpoint

Landed first browser-toggleable convergence pack:

- `backgroundMode`: `black`, `dark_random`, `curriculum`.
- `alphaReg`: `off`, `weak`, `medium`.
- `boundsReg`: `off`, `weak`, `medium`.
- `framingMode`: `normal`, `zoom_out`.
- `promptMode`: `camera`, `coarse`, `same`.
- `gridPromptMode`: added `literal_v2` / `object grid` wording:

## July 10 Implementation Update

The second convergence pack is now implemented behind browser controls:

- `transmit off/weak/paper`: the actual Dream Fields one-sided global mean
  transmittance loss, with `tau=0.40 -> 0.88` over 500 steps. This replaces
  the old symmetric per-pixel coverage target that caused the tiny-ball/fog
  collapse.
- `ray compact off/weak/med`: mip-NeRF 360-style pairwise depth distortion for
  point splats, evaluated in O(N) per ray using prefix/suffix moments inside
  the existing reverse compositing traversal.
- `ray entropy off/weak/med`: an InfoNeRF-inspired normalized alpha entropy
  alternative. It is off in the Dream preset because it adds logarithms and
  needs a quality ablation against depth distortion.
- `coarse-to-fine`: a Mip-Splatting-inspired screen-space covariance floor,
  annealed from `4.0 px^2` to `0.0625 px^2` over 500 steps. The off path keeps
  the old 0.25 px hard footprint exactly.
- `staged rates`: appearance learning ramps up during the first 250 steps;
  geometry learning then decays to 20% over 1000 steps without resetting Adam.
- `adapt splats`: every 200 profiled steps, dead low-opacity splats are
  deterministically relocated by splitting high-gradient parents. Splat count
  stays fixed and only changed Adam moment slices are reset.
- Procedural background modes: blurred noise, checkerboard, and low-frequency
  Fourier textures. Training uses the texture; display renders remain black.
- Centering now uses an opacity-weighted centroid while retaining one broadcast
  translation gradient, so transparent outliers are ignored without contracting
  pairwise splat offsets.

The anisotropic and feature32 paths began as standalone forks in
`src/splat3d_aniso/` and `src/splat3d_feature/`. As of July 11, anisotropic is
the live `full 3D` default: tiled conic RGB forward/backward/Adam, batch-major
CLIP x3 with exact parity, biased 3-of-9 views, opacity-weighted bounds,
geometric anti-tiny regularization, fixed-budget relocation, and phase
profiling. Anisotropic contact-sheet/grid IO, continuous dynamic camera poses,
and sampling-rate-aware mip filtering remain open. Feature32 still has a
correctness-reference 32-channel compositing raster, residual colorizer, and
complete backward chain, but is not a live mode; it needs a tiled anisotropic
3D-camera raster, batch views, and production optimizer state.

Correctness gates:

```bash
bun tools/splat3d/transmittance_distortion_test.ts
RAY_REG=1 bun tools/splat3d/raster_batch_forward_test.ts
bun tools/splat3d/grid_retain_parity.ts
bun tools/splat3d/regularizer_test.ts
bun tools/splat3d/background_textures_test.ts
bun tools/splat3d/adaptive_test.ts
bun tools/splat3d/aniso_projection_test.ts
bun tools/splat3d/aniso_raster_test.ts
bun tools/splat3d/aniso_optimizer_test.ts
bun tools/splat3d/aniso_batch_test.ts
bun tools/splat3d/aniso_regularizer_test.ts
bun tools/splat3d/aniso_adaptive_test.ts
bun tools/splat3d/feature_colorizer_test.ts
bun tools/splat3d/feature_raster_test.ts
bun tools/splat3d/feature_pipeline_test.ts
```

On the local Apple Metal adapter, the same 9-view direct80 grid step measured:

| Configuration | Total GPU | Raster Fwd+Bwd | Note |
| --- | ---: | ---: | ---: |
| base | `47.64 ms` | `15.20 ms` | paired baseline |
| ray compact weak | `46.40 ms` | `15.07 ms` | within run noise |
| coarse-to-fine only | `49.87 ms` | `17.18 ms` | earlier isolated sample |
| background curriculum + ray + mip + staged rates | `48.43 ms` | `16.98 ms` | `+1.7%` total |

These are sequential timestamp runs. The paired normal-step average was
`47.95 -> 50.03 ms` (`+4.3%`). Ray-only mode no longer runs the global
transmittance reduction; that replay is compiled only when `transmit` is on.
The transmittance prior is deliberately
off in the Dream preset: it is faithful to Dream Fields, but its purpose is to
reserve background area, not to cure low foreground opacity.

```text
a grid of 9 different camera angles of the same object, the object is centered, and the object is {prompt}
```

Implementation notes:

- Static black render mode keeps the old raster shader bind layout.
- Random/curriculum background compiles a dynamic-background raster variant.
- Alpha and bounds regularizers run as a separate GPU pass after CLIP/raster
  gradients and before Adam.
- When regularizer toggles are off, no regularizer pass is recorded.
- Browser convergence-toggle changes stop the current run, clear prompt embeds,
  and rebuild the optimizer so stale flags cannot keep running.

First tiny-G validation command:

```bash
G=256 TRIALS=1 RUNS=1 WARMUP=0 CONFIGS='base=1:1,conv=1:1:bgdark:alphaweak:boundsweak' bun tools/splat3d/step_matrix.ts
```

Observed on the first smoke run:

| Config | Normal | Profile | Raster | Regularizer |
| --- | ---: | ---: | ---: | ---: |
| `base` | `64.91 ms` | `32.61 ms` | `2.07 ms` | `0.00 ms` |
| `conv` | `65.95 ms` | `29.67 ms` | `1.88 ms` | `0.38 ms` |

This was a shader/binding smoke, not a quality or final speed conclusion. The
numbers are noisy at `G=256`, but the important result is that the dynamic
background and regularizer paths execute, and the regularizer overhead is visible
as its own measured timing row instead of being hidden in the hot raster/CLIP
paths.

Follow-up branch/fusion gate at `G=512` compared the current all-off path to the
pre-toggle parent commit (`9423aee`) and found no default-path regression:

- pre-toggle `base=3:3`: `46.49 ms` normal-step median;
- current all-off `base=3:3`: `46.17 ms` normal-step median;
- current dark background only: `46.88 ms` normal-step median;
- current dark background + weak alpha/bounds: `46.48 ms` normal-step median,
  with `0.30 ms` median regularizer pass.

Decision: keep these as browser toggles. They are not implemented as hot
per-pixel shader branches: black render keeps the static bind layout, dynamic
background uses a separate raster variant, and regularization is a skipped pass
when disabled.

## Quality Default Checkpoint

The first browser default was still the fast ablation baseline:

- camera-specific prompt wording;
- black render;
- alpha/bounds regularizers off;
- normal framing;
- 3 epoch-sampled views.

That is useful for speed comparisons, but it is a bad default for visual
quality and still looks far behind Dream Fields. The page now opens in a
`dream-ish` preset:

- coarse view prompts;
- `centered on a black background` prompt suffix;
- background curriculum;
- alpha off and weak bounds regularization;
- zoomed-out framing;
- 5 random views per step;
- a geometry-biased LR profile: higher position/radius LR and lower color LR.

The old setup is preserved as `fast base`, and any manual control edit switches
the readout to `manual`.

This is still not Dream Fields parity. The next real quality gap is not another
UI preset; it is differentiable image augmentation and a rendered alpha /
transmittance coverage loss. The current alpha regularizer is per-splat opacity
sparsity, which is weaker than a rendered occupancy/transmittance objective.

## Dream Fields And Splat Regularizer Toggle Pack

The Dream Fields ideas that directly fit this splat optimizer are:

1. broad/random camera sampling;
2. background augmentation instead of a fixed black/white background during the
   whole optimization;
3. rendered transmittance/coverage pressure;
4. scene bounds around the object;
5. CLIP image-space augmentation, still not implemented here.

Existing toggles covered the first, second, and fourth items. An earlier
checkpoint attempted the missing rendered-transmittance piece with a
`coverageReg` toggle:

- `coverage off`: no coverage uniform, old backward shader layout;
- `coverage weak`: coverage target `0.18`, weight `8`;
- `coverage med`: coverage target `0.24`, weight `24`.

That retired implementation applied a rendered-alpha target inside backward:

```text
coverage = 1 - final_transmittance
loss ~= mean((coverage - target)^2)
```

It was an approximation, not a faithful Dream Fields transmittance prior. A
controlled 4096-splat run showed that
`coverage weak` contracts RMS cloud spread from `0.9031` to `0.3352` in 40
steps and raises the hottest tile from the non-coverage control's `830` splats
to `2665`. The symmetric per-pixel target rewards overlapping splats in a
small central region. That implementation has now been replaced by the clipped
global mean-transmittance objective; the UI label is `transmit`, not coverage.

The replacement fits splats because raster backward already reconstructs final
transmittance `T` per pixel before walking splats in reverse. `transmit` now
uses a clipped global mean-transmittance reduction; when all ray priors are off,
their uniforms, reductions, and branches are absent from the fast baseline.

Splat-specific regularization now has its own `splatReg` toggle:

- `splat reg off`: no scale/anti-alias regularizer;
- `anti-tiny`: penalizes tiny high-opacity splats;
- `scale band`: anti-tiny plus a weak world-radius band.

This is the simplest Mip-Splatting / surface-splatting-inspired guard we can add
without changing the representation. It is not full EWA/mip filtering, but it
directly attacks the common CLIP failure where many tiny opaque dots become
adversarial texture instead of object structure.

`dream-ish` now defaults to:

- `grid9_close2`;
- `grid raster 80`;
- `object grid` prompt;
- weak grid gradient scale;
- random full-resolution closeups;
- black renderer background;
- alpha and transmittance off, corrected weak bounds, and anti-tiny splat regularization;
- ray compactness, entropy, staged rates, adaptive relocation, and mip smoothing
  off until quality gates establish a win;
- zoomed-out framing.

This makes `grid + 2 random` the default candidate. It is probably the best
semantic signal per CLIP pass because all 9 views are visible to CLIP every
step and two full-resolution views carry detail. It is still not proven best
for geometry: the grid lane compresses each view to 80px, so full-view random
batches can still beat it on some prompts. Treat `grid + 2 random` as the first
quality default, not as a settled result.

Prompt wording remains an ablation. `coarse` means broad suffixes like
`front view`, `side view`, and `back view`; `camera` means the longer natural
camera-angle descriptions. The evidence only says nine unrelated prompts can
create semantic inconsistency. It does not prove `coarse` wins in this app.
Use `same`, `coarse`, `camera`, and `object grid` as screenshot gates.

A real-prompt 40-step cat gate justified keeping the new priors out of the
default: ray compactness was mean-neutral but reduced the minimum view
(`+0.00016/-0.01648` mean/min cosine), mip smoothing was
`-0.00828/-0.03455`, and staged rates were `-0.01713/-0.01363`. These are
short-budget, one-prompt results rather than final rejections, but they are
enough to require explicit opt-in.

## Dual Grid Plus Zoom Objective

The next convergence branch adds a heavier CLIP layout called `dual_grid4`.
It is exposed in the browser as `2 grids + 4` and uses six CLIP lanes:

- lane 0: a 3x3 grid of the canonical nine fixed cameras;
- lane 1: a 3x3 grid of nine varied orbit cameras;
- lanes 2-3: two full-resolution varied camera views;
- lanes 4-5: two full-resolution zoomed-in camera views.

The fixed grid prompt still uses the selected grid prompt mode. The random grid
uses:

```text
a 3x3 grid of nine varied camera views of the same object, the object is centered, and the object is {prompt}
```

The zoom lanes use:

```text
a zoomed-in close-up view of {prompt}
```

with the selected black-background wording appended by the same prompt helper.

This is the closest browser-side version of the current Dream-Fields-ish idea
without changing the splat primitive: many views, explicit same-object grid
supervision, extra full-resolution views, and close-up pressure. It is not true
image-space CLIP augmentation. A crop/resize augmentation pass would render once
and feed CLIP transformed crops of that render, then scatter image gradients
back through the crop transform.

Early browser screenshots showed a sharp black square in the middle of every
camera view. Pixel inspection put its edges on the raster's `16x16` tile
boundaries. A concentrated 4096-splat reproduction then proved the cause:
`cap=2048` overflowed 16 central tiles and dropped 21,387 of 65,659 splat-tile
pairs. `cap=4096` dropped zero. This was raster tile-bin overflow, not the CLIP
grid being pasted into the object.

The default tile cap now scales to the scene splat count up to `4096`, which
guarantees no tile overflow for the default 4096-splat scene. The forward
kernel reuses its ID workgroup array for the final `tileStop` reduction so a
4096-entry tile fits the WebGPU-minimum 16KB workgroup-storage limit. The page
also samples tile counts in the timing overlay and reports any overflow.

Grid strength remains a useful convergence ablation, independent of the square:

- `grid off`: grid lanes still run through CLIP but do not backprop to splats;
- `grid weak`: fixed grid `0.25x`, random grid `0.15x`;
- `grid med`: fixed grid `0.5x`, random grid `0.35x`;
- `grid full`: old full-strength grid gradients.

The page also exposes `grid raster 512`. The original fast path rendered each
contact-sheet view directly at `80x80`, then pasted it into the `256x256` CLIP
image. The high-res path renders each contact-sheet view at `512x512` and
bilinearly downsamples a packed, no-gutter 3x3 contact sheet into the same
`256x256` CLIP input, with gradients scattered back through the resize. This is
closer to "render high, concatenate, resize for CLIP" and should reduce
scale/aliasing mismatch, but CLIP still ultimately sees each grid view as a
small panel inside a `256x256` image.

Representation branch status: surfels remain deferred. Anisotropic splats are
available as an experimental live toggle with conic rasterization, 14G Adam,
and CLIP forward/backward. The first integration intentionally uses single-CLIP
per-view training; batch/grid IO and the isotropic convergence controls remain
disabled in that mode.

## Tiny-Ball Collapse Diagnosis

Two independent convergence bugs were identified on 2026-07-10:

1. The old `centerWeight` gradient was `2 * weight * position` for every
   splat. That is per-splat L2 shrinkage, not object centering. It now reduces
   the cloud centroid on GPU and applies one identical translation gradient to
   every splat. `tools/splat3d/regularizer_test.ts` verifies zero pairwise
   gradient delta.
2. The rendered coverage target independently causes central collapse. With
   all convergence regularizers off, spread remains `0.9031 -> 0.8733` over
   40 steps. With `coverage weak` alone it becomes `0.9031 -> 0.3352`.

The corrected Dream-ish preset keeps coverage off. A 40-step diagnostic run
with dark background curriculum, the then-enabled weak alpha term, corrected
weak bounds, anti-tiny splats, and coverage off retained spread at `0.8601`
with a near-zero centroid. The opacity probe below subsequently showed why the
alpha term also had to leave the default.

## Opacity Fade Diagnosis

The old `alpha weak` toggle is also not the Dream Fields transmittance loss. It
adds the gradient of `weight * sum(per_splat_opacity)` every step and never
stops. With Adam's opacity learning rate of `0.03`, even a small regularizer
weight produces a persistent downward update.

`tools/splat3d/opacity_decay_probe.ts` isolates this term from CLIP:

| Step | Mean per-splat opacity |
| ---: | ---: |
| 0 | `0.574443` |
| 100 | `0.097784` |
| 500 | `0.008264` |
| 800 | `0.003901` |

Dream-ish now defaults to `alpha off`. The timing overlay reports mean opacity,
p10/p90 opacity, mean world-space splat radius, and RMS cloud spread.

The faithful Dream Fields replacement is a clipped global mean-transmittance
loss, not a per-splat opacity sum and not the current per-pixel coverage target:

```text
L_T = -min(tau, mean(final_transmittance))
tau: 0.40 -> 0.88 over the first 500 iterations
```

This is now implemented with a per-view GPU reduction of final transmittance.
It stops contributing gradients after the target is reached and is exposed as
`transmit off/weak/paper`. It remains off in the Dream preset until screenshot
tests establish that reserving 88% mean transmittance fits the desired framing.

## Feature Splat Fork

Feature splatting is a reasonable representation ablation, but it should be a
forked renderer mode rather than an in-place expansion of the RGB hot path.
DynaWorld's relevant lessons are:

- raster arbitrary `F`-channel features, then colorize at the loss boundary;
- keep the first three channels as an RGB skip connection;
- initialize the decoder residual to zero so feature mode starts at RGB parity;
- cache a pixel's feature gradient in registers and reduce per-splat feature
  gradients in SIMD/threadgroup memory before global atomics;
- use power-of-two/vector-aligned dimensions in the hot path.

Use `F=32`, not `F=22`. Thirty-two channels are eight `vec4` groups and match
the dimensions exercised by DynaWorld's feature raster variants. For 4096
splats, the feature parameters are only about `0.5 MB`, but a 256x256 feature
image is `8 MB` per lane and feature-gradient work grows from 3 to 32 channels.

### Version 1

1. Add a rebuild-time `RGB | feature32` representation toggle.
2. Fork `raster_wgsl.ts` into a feature-specific shader module; keep RGB code
   unchanged as the reference and rollback path.
3. Initialize feature channels 0-2 from current RGB logits and channels 3-31
   near zero.
4. Alpha-composite 32D features plus a separate alpha output.
5. Decode with an identity-safe residual colorizer:

```text
rgb = sigmoid(feature[0:3] + 0.1 * (W[3x32] * feature + bias))
```

Initialize `W` and `bias` to zero. Start with this linear 1x1 decoder before a
hidden MLP. It is cheap, exactly preserves the RGB baseline at initialization,
and gives the extra channels a route to affect color.
6. Backpropagate CLIP RGB gradients through the colorizer into the 32D rendered
   feature image, then through the feature raster backward.

Do not add view direction to version 1. View-conditioned color can let CLIP
paint a different object from each camera and weaken the shared-3D constraint.

### Feature Loss

Keep decoded-RGB CLIP as the primary objective. A direct loss on arbitrary
rendered features is unsafe without a well-defined teacher target: it lets the
optimizer bypass CLIP's natural-image input distribution and can improve the
embedding while making RGB worse.

The first auxiliary losses should only stabilize the representation:

- small L2 penalty on non-RGB feature channels;
- feature variance floor to prevent all extra channels collapsing;
- optional consistency between augmented renders of the same view.

A later experimental lane can learn a `32 -> CLIP stem channels` adapter and
inject after CLIP's stem, but it must retain the RGB CLIP loss and stay at a
small auxiliary weight.

### Gates

- exact RGB parity at feature-mode initialization;
- finite-difference forward/backward gate for feature and decoder gradients;
- equal-CLIP-pass cat prompt screenshots at 500/1000/2000 steps;
- total-step slowdown target below `1.5x`;
- kill the lane if it raises CLIP score without visibly improving decoded RGB.
