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
