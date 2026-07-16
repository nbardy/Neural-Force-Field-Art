# Blog Progress Notes

## Working Checkpoint

- 2D prompt-to-splats page works with WebGPU CLIP guidance and a `NUDGE` button.
- The 2D page now has an experimental `feature painter` mode: tiled 32D splat
  features, local-coordinate appearance coefficients, and a trainable residual
  RGB decoder. It remains separate from the RGB default while its full
  appearance-to-geometry Jacobian is validated.
- 3D fork renders one shared 3D splat cloud from nine fixed cameras in a 3x3 grid.
- 3D optimizer accumulates gradients from all nine camera views, then applies one shared Adam update.
- Prompt modes:
  - `camera text`: appends a natural camera-angle phrase per view.
  - `same text`: uses the same base text for all views.
- Black-background prompt text is a toggle:
  - first setting to try: append `on a black background`.
  - toggle off: no black-background wording, while the renderer still uses a black background.
- Current camera prompt style is natural language, e.g. `a camera angle from the right side`, not internal labels like `front-left-high`.

## Screenshot Plan

Capture these as the system gets better:

1. 2D page before optimization: random initial splats.
2. 2D page after early CLIP optimization: first recognizable structure.
3. 2D `NUDGE`: before/after showing a partial rerandomization without full reset.
4. 3D page at boot: 3x3 camera grid with random shared splats.
5. 3D page after a short run: same object beginning to appear across multiple views.
6. 3D page after a longer run: best current multi-view result.
7. Prompt toggle comparison: `black bg` on vs `no bg text`.
8. Prompt mode comparison: `camera text` vs `same text`.

## Blog Story Beats

- Start with the 2D CLIP-over-splats idea: explicit pixels, no diffusion, no CPU readback.
- Show the `NUDGE` button as an artist-facing control for escaping local visual basins.
- Explain the 3D leap: one shared splat cloud, nine camera views, one joint optimization step.
- Show why the grid matters: the viewer can see whether the object is actually 3D-consistent.
- Call out the first prompt/background hypothesis: black renderer background plus optional `on a black background` CLIP text.
- Leave ablations for later; this checkpoint is about making the system visible and testable.

## Reflection Checkpoint

See `docs/SPLAT3D_REFLECTION_REVIEW.md` for the corrected speed ladder and
forward-looking direction map. The blog should separate the product-level
`9/9 -> 3/9` speedup story from the same-CLIP-budget optimization story:

- product-level: original full `9/9` independent views to current sampled/grid
  paths is roughly `3.6x-3.9x` faster depending on layout;
- same CLIP count: old `3/9` to current `3/9` is about `1.33x` faster;
- `grid80 + 2 full` uses the same three CLIP images per step, costs about `8.6%`
  more than current `3/9`, and keeps pressure on all nine views.

See `docs/SPLAT3D_CONVERGENCE_PLAN.md` for the separate paper-driven convergence
story: background/opacity regularization, centering, prompt/view curriculum,
anti-aliasing, and staged surface-biased splats.

## Next Plan Notes

- Do not start broad ablations yet.
- First manual tests should focus on screenshots and qualitative behavior:
  - Does `on a black background` help object formation?
  - Does `camera text` make views more distinct or more stable?
  - Does `same text` produce a more coherent shared object?
  - Does the 3x3 grid show collapse/fog/hole failures clearly?

## Dream Fields-Inspired Ablation Plan

Dream Fields is a good mental model, but the browser version should stay splat-first:
explicit splat rasterization should be much faster than NeRF-style ray marching in
WebGPU. The likely bottleneck is CLIP per view, not the rasterizer. Dream Fields
also did not use CLIP alone: the transferable pieces are geometric/background
priors around CLIP.

Run these as screenshot-friendly toggles, one at a time:

1. **Prompt/background text**
   - `black bg`: append `on a black background`.
   - `no bg text`: no background wording.
   - Later wording to try: `centered on a black background`.

2. **Renderer background**
   - fixed black;
   - dark random background;
   - full random background.
   Keep black as the first visual baseline, but Dream Fields suggests random
   backgrounds are important once opacity regularization exists.

3. **Opacity/transmittance pressure**
   - no regularizer;
   - weak alpha/coverage target;
   - stronger transmittance-style sparsity.
   This is the highest-priority Dream Fields prior to port after screenshots.

4. **Scene bounds / object centering**
   - no bounds beyond current init;
   - soft radial bound around origin;
   - zoomed-out centered object framing.
   This should reduce frame-filling fog and off-center CLIP hacks.

5. **View sampling**
   - all 9 views per step;
   - rotating 3-view subset;
   - 1 random view per step plus periodic 9-view display refresh.
   This is the main speed ablation before CLIP batching.

6. **Prompt directionality**
   - same base prompt for all views;
   - coarse direction suffixes only (`front`, `side`, `back`, `top`);
   - all 9 natural camera phrases.
   Avoid internal labels like `front-left-high`; use phrases like
   `a camera angle from the right side`.

7. **Representation schedule**
   - current isotropic 3D blobs;
   - anisotropic 3D Gaussian scale;
   - later surfel/2DGS-style flattening once the object exists.
   Do not start here; this is a later geometry upgrade if volumetric blobs stay
   incoherent.

## July 10: Square Fixed, Then The Ball Collapse

- The sharp center square was tile-bin overflow, not a CLIP grid paste. A
  concentrated scene dropped 32.6% of splat/tile pairs at cap 2048; cap 4096
  drops none.
- Fixing overflow exposed the true cost of a collapsed scene: nearly all 4096
  splats were being sorted and differentiated in the same central tiles.
- The first centering prior was wrong. It pulled every splat toward zero and
  has been replaced by a centroid translation gradient that preserves spread.
- The stronger remaining collapse came from `coverage weak`: its symmetric
  per-pixel alpha target made overlapping splats centrally an easy solution.
  Dream-ish now defaults to coverage off, while the toggle remains for screenshots.
- Keeping each 80px grid cell's raster scratch removes nine redundant forward
  replays per optimizer step for about 5.2 MB of extra GPU memory. Replay and
  retained modes produce bit-identical optimized parameters.
- The next screenshot exposed a second regularizer mismatch: `alpha weak`
  drives mean per-splat opacity from `0.574` to `0.0039` by step 800 when
  isolated. Dream-ish now defaults to alpha off and displays opacity telemetry.
- Feature splatting remains a separate fork proposal: 32D features, RGB
  residual skip, zero-initialized 1x1 colorizer, and no direct CLIP feature
  injection in the first version.

## July 10: Paper Priors Become Real Toggles

- The old `coverage` control was removed conceptually: it was a symmetric
  per-pixel alpha target, not Dream Fields, and it caused collapse.
- `transmit paper` now implements Dream Fields' clipped global mean
  transmittance with the published `0.40 -> 0.88` 500-step schedule.
- `ray compact` pulls separated contributions on one camera ray toward a common
  depth; `ray entropy` is available as a slower alternative.
- `coarse-to-fine` starts with a 2 px screen-space covariance floor and anneals
  to the old quarter-pixel footprint.
- GPU procedural backgrounds now include blurred noise, checkerboard, and
  Fourier textures while the displayed result stays black.
- `adapt splats` recycles low-opacity splats into high-gradient regions without
  changing the 4096-splat budget.
- The Dream preset keeps the renderer black and leaves all new ray/mip/staging/
  relocation priors off. A 40-step real-prompt cat gate found no default-worthy
  win yet; they remain explicit screenshot ablations.
- Same-layout paired timestamp sample: `47.64 ms -> 48.43 ms` with background
  curriculum + ray compactness + mip curriculum + staged rates (`+1.7%` GPU;
  `+4.3%` in the normal-step wall sample).
- The anisotropic fork now has a complete tiled conic raster/backward/Adam path.
  It is available in the representation menu through conservative single-CLIP
  per-view training.
  Feature32 now has a complete correctness-reference raster -> colorizer ->
  backward chain. Neither representation has production batch/grid IO yet,
  which should be
  stated plainly rather than presenting isolated fork tests as live results.

## July 10: Why The "Anisotropic" Splats Still Looked Round

- The first anisotropic live initializer copied one scalar radius into all
  three axes and used identity quaternions. It was an anisotropic parameter
  layout initialized as exact spheres, so the screenshots correctly looked
  circular and rotation initially had no useful gradient.
- The fix preserves each splat's geometric-mean radius while applying shuffled
  zero-mean log-axis offsets and a uniform random 3D orientation. The resulting
  initial mean axis ratio is about `2.5:1` in the Metal integration gate. The
  actual projected ratio averages `1.74-1.76:1` across all nine cameras, with a
  p90 around `2.3:1`.
- The profiler now prints both 3D axis ratio and projected screen ratio. A 3D
  ellipsoid can still look circular from a camera aligned with its long axis;
  the two numbers make that distinction visible.
- Random three-view batches now favor the common front-on, side-on, and directly
  overhead cameras about `2.5x` over rear/oblique cameras, without replacement.
  Epoch mode remains uniformly covering.
- Canonical prompts are now natural prefixes such as `a front-on view of a cat`
  rather than internal camera labels appended to the subject.

## July 11: Full 3D Becomes The Default

- The old `dream-ish` preset was conservative in the wrong places: it booted
  isotropic, used coarse text that bypassed the new canonical captions, and
  requested all nine views so weighted random sampling did nothing.
- The selected `full 3D` preset now boots the live anisotropic conic renderer
  with random orientations, natural camera-prefix prompts, centered black
  background text/rendering, zoomed-out framing, a biased random 3-of-9 view
  subset, and batch-3 CLIP.
- The anisotropic optimizer now supports batch-major CLIP with shared raster
  replay. Parameter parity against three single CLIP passes is byte exact.
  A stable 4,096-splat ten-step sample measured `81.63 ms -> 61.28 ms`
  (`1.33x`); shorter samples ranged more widely under GPU contention, so the
  phase profile is the more useful number.
- The 4,096-splat phase profile measured `41.08 ms` CLIP, `2.16 ms` initial
  raster, `4.15 ms` replay, `15.87 ms` raster backward, `0.15 ms`
  regularization, `0.30 ms` Adam, and `0.87 ms` display.
- Opacity-weighted centroid bounds and anti-tiny regularization now operate on
  anisotropic splats using geometric-mean radius. The gradient is split equally
  over all three log-scale axes, so the regularizer does not erase learned
  anisotropy.
- Fixed-budget adaptation now works for anisotropic splats: dead destinations
  clone a high-gradient parent's color, quaternion, and axis ratios; both
  children shrink uniformly and preserve coverage mass. A 240-step quality run
  had no genuinely dead destinations, so the default correctly performed zero
  relocations rather than changing healthy splats.
- The first 240-step adaptation run caught a crash when an elongated trained
  splat could not fit every axis inside a scalar radius band. Adaptation now
  clamps geometric-mean scale with a shared log shift and never rejects a splat
  merely for having a wide learned axis ratio.
- On the 1,024-splat, 240-step real-cat gate, full 3D beat the same anisotropic
  setup without geometry safeguards by `+0.00754` mean CLIP cosine and
  `+0.04074` worst-view cosine while reducing RMS cloud spread by `11.3%` at
  essentially identical throughput (`18.94` vs `19.01` steps/s).
- At 4,096 splats and 60 steps, full 3D improved mean cosine by `+0.00840` and
  reduced spread by `5.4%`. A separate production-size occupancy gate reached
  `1,231/4,096` splats in the busiest tile with zero overflow.
- Transmittance, ray entropy/compactness, mip smoothing, and random training
  backgrounds remain explicit experiments. They are not enabled simply to make
  the preset look busy; previous controlled runs showed fading or no quality
  win. The work remains available without weakening the evidence-backed default.
