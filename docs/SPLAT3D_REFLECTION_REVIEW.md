# 3D Splat Reflection And Review

Date: 2026-07-08

## Measurement Rule

Do not mix these two claims:

1. Product-level speedup versus the original full 9-view optimizer.
2. Same-CLIP-budget speedup versus the old `3/9` optimizer.

Both are useful, but they answer different questions. The honest kernel/scheduler
anchor is the same number of CLIP images per step.

## Same-CLIP-Budget Ladder

Baseline here is the old `3/9` path: three full MobileCLIP image passes per
optimizer step.

| Stage | Step Time | Time vs Old `3/9` | Speedup | Read |
| --- | ---: | ---: | ---: | --- |
| Old `3/9` per-view | `69.94 ms` | `100.0%` | `1.00x` | Same-resolution baseline |
| Current `3/9` per-view | `52.51 ms` | `75.1%` | `1.33x` | Best same-budget speed claim |
| Current `grid80 + 2 full` | `57.05 ms` | `81.6%` | `1.23x` | Same three CLIP images, but one is a 9-view grid |
| Current `grid80 + 2 full` vs current `3/9` | `57.05 ms / 52.51 ms` | `108.6%` | `0.92x` | Grid costs about 8.6% more wall time |

Takeaway: the corrected same-CLIP-count speedup is about `1.3x`, not `3x`.
The grid path is slightly slower than current `3/9`, but it applies all-view
pressure while staying inside the same three-image CLIP budget.

## Product-Level Ladder

This comparison is still useful for the user-facing story, but it is not a pure
kernel optimization claim.

| Stage | Step Time | Time vs Full `9/9` | Speedup |
| --- | ---: | ---: | ---: |
| Original full `9/9` per-view | `205.26 ms` | `100.0%` | `1.00x` |
| Old `5/9` per-view | `122.02 ms` | `59.4%` | `1.68x` |
| Old `3/9` per-view | `69.94 ms` | `34.1%` | `2.94x` |
| Current `3/9` per-view | `52.51 ms` | `25.6%` | `3.91x` |
| Current `grid80 + 2 full` | `57.05 ms` | `27.8%` | `3.60x` |

Takeaway: the `3x` win came from reducing independent full-view CLIP passes, not
from a single shader rewrite. The later kernel/layout work moved the already
sampled regime from roughly `70 ms` to the low/mid `50 ms` range.

## Grid State

The implemented `grid9_close2` layout keeps the CLIP batch at three images:

- lane 0: one normal `3x256x256` MobileCLIP input containing a 3x3 contact sheet
  of all nine camera views;
- lanes 1 and 2: two full-resolution per-view renders;
- with `grid raster 80`, each grid cell is rasterized directly at `80x80` and
  copied into the 256px contact sheet.

Prompt modes:

- `contact_sheet`: `a 3x3 image grid showing the same subject, {prompt}, from nine different camera angles, centered on a black background: ...`
- `literal`: `a 3x3 grid showing {prompt} from 9 different camera angles, centered on a black background. The 9 panels show the same subject in reading order: ...`
- `same`: the base prompt only.

The user's proposed wording is still worth testing as another literal variant:

```text
a grid of 9 different camera angles of the same object, the object is centered, and the object is {prompt}
```

## Quality State

The first real-prompt cat quality gate found a tradeoff, not a failure:

| Variant | Steps / 5s | Steps / Sec | Mean Cos | Mean Delta | Min Cos |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base3` | 91 | 18.02 | 0.25410 | 0.12579 | 0.20980 |
| `grid80` | 78 | 15.55 | 0.23802 | 0.11083 | 0.19165 |

Read: `grid80` reached `93.7%` of the final mean cosine and `88.1%` of the
mean-cosine improvement versus `base3` in the same 5s budget. That is behind,
but close enough to keep as a serious lane.

## Convergence State

The speed ladder is not the whole story. The paper trail points to a separate
convergence track: better incentives per step, not just more steps per second.

See `docs/SPLAT3D_CONVERGENCE_PLAN.md` for the ranked convergence lanes. The
short version:

1. Dream Fields-style dark random backgrounds plus opacity/transmittance control
   are the most direct fix for fog, transparent holes, and background cheating.
2. Bounds, centering, and slightly zoomed-out framing should stop CLIP from
   painting the whole frame instead of forming an object at the origin.
3. Prompt policy should probably be one shared base prompt plus coarse direction
   words, not nine fully bespoke prompts.
4. Mip-Splatting-style scale/frequency control should reduce tiny high-frequency
   splat artifacts that CLIP may over-reward.
5. Surface-biased splats are a staged geometry upgrade: coarse 3DGS first,
   anisotropic/flattened surfels later, normal/depth consistency after an object
   exists.

## Low-Precision State

Low precision is not promoted.

What exists:

- opt-in f16 weight storage for MobileCLIP vision weights;
- fp32 math, activations, reductions, gradients, and `dL/dimage`;
- weight payload roughly halves;
- strict f16 comparison checks embedding and input-gradient cosine.

Why it is not default:

- embedding cosine passed;
- input-gradient cosine failed the planned strict gate;
- integrated timing was mixed and in at least one recorded run was slower.

Conclusion: f16 is an artifact-size experiment, not the current convergence-speed
answer. Future mixed precision should be narrower and gradient-gated.

## What Moved The Needle

1. View scheduling: `9/9 -> 3/9` was the largest speed change.
2. Grid CLIP layout: one contact-sheet CLIP lane plus two full lanes gives
   all-view pressure at nearly the current `3/9` cost.
3. Direct grid raster: rasterizing grid cells at `80x80` avoided the worst
   contact-sheet raster overhead.
4. Backward stack promotion: `depthwise4` plus GELU/residual backward fusions
   shaved CLIP time in the grid envelope.
5. Measurement tooling: step matrices, dispatch profiles, quality gates, and
   fork snapshots made the work auditable instead of relying on one-off runs.

## Where We Churned

- Blanket f16 did not preserve the input-gradient gate and did not produce a
  trustworthy speed win.
- Rectangular `8x16` pointwise tiling was correct but basically flat or worse in
  integrated timing.
- Cached CLIP-gradient cadence produced large wall-clock speedups, but quality
  regressed too much in the first teacher gate.
- Larger CLIP batches helped in some contexts, but replay and raster bookkeeping
  prevented a clean linear win in the integrated optimizer.

## Forward-Looking Directions

### 1. Measurement And UI Instrumentation

- Keep comparing same CLIP image count unless explicitly telling the product
  story.
- Add or keep a top-right overlay with `CLIP fwd`, `CLIP bwd`, raster fwd,
  raster bwd, optimizer step, milliseconds, and percent of step time.
- Run matrices without obvious GPU contention and record median/min/max.

### 2. View Scheduling

- Keep `3/9` as the fast baseline.
- Keep `grid80 + 2 full` as the all-view baseline.
- Test adaptive `N of K`: sample harder views more often, periodically refresh
  all nine full views, and compare against fixed shuffled epochs.
- Try the user's literal grid prompt wording as a quality gate, not a speed gate.

### 3. Prompt And Loss Semantics

- Keep black renderer background as the default visual baseline.
- Test prompt toggles: `on a black background`, `centered on a black background`,
  and no background text.
- Prefer natural camera phrases over internal labels.
- Test whether `top-down view` beats `top view`, and whether "same object" in
  the grid prompt improves cross-view coherence.

### 4. CLIP Kernel Speed

- The pointwise family remains the largest exact-math target, especially
  `pw_bwd`.
- Next serious forks should be shared-W or split-K pointwise backward, not more
  broad fp16.
- Local fusions are useful, but probably incremental rather than another 3x.
- Approximate/low-rank/layer-skip CLIP is a separate research lane because it
  must be judged by convergence quality, not only milliseconds.

### 5. Raster And Splat Representation

- Continue checking FasterGS/dynaworld-style ideas for culling, pruning, and
  backward sharing, especially if raster backward grows with more views.
- Batch-view rasterization may be useful, but the current bottleneck is often
  CLIP. It should be tested only with integrated timings.
- Geometry upgrades on the table: anisotropic 3D Gaussians, opacity pruning,
  surface flattening, and later surfel/2DGS-style constraints.

### 6. Convergence And Regularization

- Use `docs/SPLAT3D_CONVERGENCE_PLAN.md` as the main queue for paper-derived
  convergence work.
- Add object-centering and zoom-out priors before exotic kernels if screenshots
  still show fog/fill-frame hacks.
- Test opacity/transmittance pressure to reduce background leakage.
- Later: depth/normal/surface consistency once the splat cloud forms a coherent
  object.

### 7. Blog Trail

- Capture screenshots at each rung: random 3D, old `3/9`, current `3/9`,
  `grid80`, literal grid, black-background toggle on/off, and perf overlay.
- The clean story is: visible 3D grid first, then speed ladder correction, then
  quality tradeoffs, then the research lanes still open.

## Recommended Next Checkpoint

Run one focused prompt-quality gate before more kernel work:

```bash
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat" CONFIGS='base3=3:3,grid80=9:3:grid9:directgrid,grid80literal=9:3:grid9:directgrid:literal' OUT_DIR=/tmp/nffa_grid_literal_cat bun tools/splat3d/grid_quality.ts
```

Then add the user's exact grid wording as a fourth prompt mode if `literal` is
not clearly better than `contact_sheet`.
