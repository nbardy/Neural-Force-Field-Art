# 3D Splat Open Research Handoff

Date: 2026-07-11

This is the authoritative thread-close index for unfinished 3D splat work.
Older plans and agent notes remain valuable experiment records, but their
descriptions of the active default may be historical.

## Current Full 3D Checkpoint

The browser default is now:

- 4096 anisotropic RGB splats at 256px;
- random initial orientations and about 2.5:1 initial 3D axis ratio;
- natural camera-prefix prompts and centered black-background text;
- black display/training background;
- zoomed-out framing;
- biased random 3-of-9 canonical camera sampling;
- batch-major CLIP x3 with exact parameter parity against three single passes;
- weak opacity-weighted centroid/bounds control;
- geometric-mean anti-tiny regularization;
- fixed-budget adaptive relocation, enabled but a no-op until splats are dead;
- phase profiling for raster, CLIP, replay, backward, regularizer, Adam, and display.

The 1024-splat, 240-step cat gate improved mean CLIP cosine by `0.00754`,
worst-view cosine by `0.04074`, and reduced RMS cloud spread by `11.3%` versus
the same anisotropic optimizer without geometry safeguards. The preserved
contact sheets and JSON are under `docs/assets/splat3d_full3d_2026-07-11/`.

## Status Vocabulary

- `PROMOTED`: active in the default.
- `IMPLEMENTED`: available as a control or tested fork, but not default.
- `PARKED`: correct implementation measured flat/worse or not worth current complexity.
- `OPEN EXACT`: should preserve the present objective/math.
- `OPEN APPROX`: changes the guidance signal or representation and needs teacher-quality gates.
- `DEFERRED`: valuable only after a prerequisite milestone.

## Priority Queue

### P0: Quality And Camera Coverage

1. `OPEN EXACT` Continuous random-orbit cameras.
   - Current "biased random" samples without replacement from nine fixed cameras.
   - The dual-grid "random" cameras are also a deterministic nine-pose pool.
   - Move camera values from baked single-view WGSL constants into GPU camera
     buffers or a bounded precompiled pool.
   - Sample seeded azimuth, elevation, radius, FOV, and framing distributions.
   - Preserve minimum front, left/right side, and directly-overhead exposure,
     while reserving probability for arbitrary orbit poses.
   - Bucket arbitrary poses into natural prompts: front-on, side-on, rear,
     directly overhead, elevated looking down, and low looking up.
   - Gate against the fixed-nine weighted sampler at equal CLIP lanes and wall time.

2. `OPEN APPROX` Adaptive hard-view sampling.
   - Maintain an EMA of per-camera teacher loss or cosine.
   - Sample weak views with capped importance weights.
   - Keep a minimum canonical-view probability and run a full-nine refresh every M steps.
   - Report all-nine worst-view and mean teacher score, not only sampled views.

3. `OPEN APPROX` Grid/close-up schedule matrix.
   - At exactly three CLIP image lanes, compare:
     - biased per-view 3-of-9;
     - canonical 3x3 grid plus two full views;
     - random-orbit 3x3 grid plus two full views.
   - Separately test the six-lane layout: canonical grid, random grid, two full
     views, and two zoomed/cropped close-ups.
   - Compare 80px direct cells, 256px cells, and 512px render-then-resize.
   - Do not describe a six-lane quality gain as a schedule win without an
     equal-CLIP-lane comparison.

4. `OPEN APPROX` Differentiable image-space CLIP augmentation.
   - Reuse one render and apply differentiable crop/resize, mild affine or
     perspective jitter, cutout, and restrained color jitter.
   - Scatter CLIP gradients through each transform back to the original render.
   - Treat zoom crops as augmentations, not higher raster resolution by itself.
   - Gate at equal CLIP image-lane count against no augmentation.

5. `OPEN EXACT` Universal quality protocol.
   - Every promotion reports equal processed CLIP lanes and equal wall-clock.
   - Use at least three prompts and three seeds, serialized GPU runs, all-nine
     full-resolution teacher scores, worst-view score, cloud telemetry, and tile overflow.
   - Preserve contact sheets at initialization, 60, 240, 500, 1000, and 2000 steps.
   - Record seed, config, commit, adapter, timing mode, and prompt text with each image.

### P1: Anisotropic Geometry And Raster

1. `OPEN EXACT` Remove anisotropic raster replay for batch-3.
   - Add per-lane conic/tile state or a view-lane anisotropic raster dispatch.
   - Keep one camera-specific projection and depth order per lane.
   - Require image/parameter parity and integrated phase timing.

2. `OPEN EXACT` Dynamic camera-buffer projection.
   - Stop compiling one prep/chain pipeline per camera.
   - Use camera storage/uniform data in both forward and projection backward.
   - This is the prerequisite for continuous random poses without long rebuilds.

3. `OPEN EXACT` Anisotropic projection ablation.
   - Compare `legacy-affine` with `perspective-jacobian` on quality, gradient
     stability, projected axis ratio, occupancy, and runtime.
   - Track optimized-scene axis-ratio and tile-count histograms by step and camera.

4. `OPEN EXACT` Contact-sheet IO without scratch copy/scatter.
   - Retained 80px cell state already removed raster replay.
   - The remaining exact grid lane is viewport-aware forward/backward directly
     into the CLIP contact sheet, eliminating image copy and gradient scatter.
   - Promotion bar: gradient parity and at least 8% lower grid raster backward overhead.

5. `PARKED` Existing isotropic view-lane batch raster.
   - Forward/backward variants passed parity but did not improve the default
     integrated step enough to promote.
   - Revisit only if raster again dominates after CLIP changes.

6. `OPEN EXACT` FasterGS/DynaWorld follow-ups.
   - Evaluate nearest-K tails, visibility-aware culling, sparse backward,
     tile-local gradient reductions, feature-gradient caching, and lower atomic traffic.
   - Re-run occupancy over trained anisotropic checkpoints before changing the 4096 cap.
   - The old cap-1024 and cap-2048 notes are historical; current production-size
     gates have zero overflow with dynamic cap 4096.

7. `OPEN APPROX` World-tube/gauge representation research.
   - STAR UVT/world tubes do not let unrelated static cameras share one exact
     tile list or depth order.
   - Remaining ambitious representations are camera-bundle splats, projective
     atlas plus residuals, and Plucker/ray-fiber primitives.
   - Treat these as new render/backward contracts, not scheduler optimizations.

### P1: Surface And Appearance Representations

1. `DEFERRED` Surface-phase flattening toward surfels/2DGS.
   - Define a trigger from step, opacity, spread, silhouette stability, and
     multi-view score stability.
   - Gradually flatten the smallest anisotropic axis on stable opaque splats.
   - Then add depth-distortion, normal consistency, normal-depth consistency,
     and optional view-facing orientation losses.
   - Pure surfels from initialization remain deferred because CLIP-only
     supervision can produce brittle billboards before an object exists.

2. `IMPLEMENTED` Feature32 correctness fork; `OPEN` production integration.
   - Existing reference path composites 32 channels, applies an identity-safe
     residual colorizer, and backpropagates through both.
   - Build a tiled anisotropic 3D-camera Feature32 raster with batch-major views.
   - Connect colorizer Adam state and benchmark 32-channel memory/backward cost.
   - Keep decoded-RGB CLIP as the primary loss.
   - Direct arbitrary feature-to-CLIP loss is unsafe without a teacher target.
   - Any CLIP-stem adapter must retain RGB quality parity and a small auxiliary weight.

3. `DEFERRED` Textured surfels, Gaussian billboards, GStex-style local textures.
   - Add only after geometry stabilizes; the extra appearance capacity creates
     another route for CLIP cheating.

4. `IMPLEMENTED` AbsGS/Pixel-GS fixed-budget parent ranking.
   - Every 200-step adaptation sample, raster backward records absolute
     per-pixel screen-centre gradients before pixel reduction plus visible-pixel
     counts. This is used only to rank relocation parents; Adam keeps the exact
     signed gradient.
   - The per-pixel accumulation already makes the AbsGS score pixel-aware, so
     coverage is used as a capped confidence term rather than multiplied twice.
   - The stats shader runs only on the sampled adaptation step, preserving
     normal raster-backward speed. The implementation is exact for the sampled
     views but remains a CLIP-guided approximation of reconstruction papers.
   - Primary references: [AbsGS](https://arxiv.org/abs/2404.10484) and
     [Pixel-GS](https://arxiv.org/abs/2403.15530).
   - Initial 200-step smoke gates at 64 and 1024 splats recorded `201,528` and
     `2,914,414` visible-pixel contributions respectively, with zero tile
     overflow. Neither moved splats because no opacity-qualified destination
     existed; do not mistake signal collection for a demonstrated quality win.

5. `OPEN APPROX` Adaptive densification beyond fixed-budget relocation.
   - Compare current relocation against gradient and screen-radius split/prune,
     [MCMC-style stochastic relocation](https://arxiv.org/abs/2404.09591), and
     full count-changing densification.
   - Track count, memory, opacity pruning, tile occupancy, overflow, worst-view
     cosine, and equal-CLIP-pass convergence.

### P1: Convergence And Priors

1. `IMPLEMENTED, OFF DEFAULT` Dream Fields transmittance and background curriculum.
   - Current implementation has the one-sided global mean-transmittance target,
     blurred-noise/checker/Fourier backgrounds, and black display refinement.
   - Earlier tests showed fading or no default-quality win.
   - Re-test weak weights on the now-coherent anisotropic default before rejection.

2. `IMPLEMENTED, OFF DEFAULT` Ray compactness and entropy.
   - Compare O(N) depth distortion and normalized alpha entropy only after the
     anisotropic default is stable across multiple prompts/seeds.

3. `IMPLEMENTED, OFF DEFAULT` Staged rates and covariance curriculum.
   - Current mip mode is an additive screen-space covariance floor, not full Mip-Splatting.
   - Keep staged rates and current coarse-to-fine mode as explicit ablations.

4. `OPEN EXACT/APPROX` Sampling-rate-aware AA.
   - Derive a world-space scale floor from focal length, distance, and training
     camera sampling density.
   - Compare current footprint, covariance curriculum, EWA-style footprint,
     and world-space mip filtering at multiple camera distances.

5. `OPEN APPROX` FreGS-inspired spectral lane.
   - Text guidance has no target image spectrum, so exact FreGS parity is impossible.
   - Test a coarse-to-fine low-pass CLIP curriculum and an excessive
     high-frequency energy penalty.
   - Keep this distinct from Fourier background generation.

6. `DEFERRED` External depth/normal priors.
   - Monocular depth/normal teachers or diffusion/SDS priors materially increase
     model/runtime scope and are not browser-first.
   - First exhaust self-derived depth, normals, silhouette, and surface consistency.

### P1: Exact CLIP Kernel Speed

1. `OPEN EXACT` Split-K pointwise backward.
   - Implement `v21_splitk_pw_bwd` for selected large backward indexes with four
     K partitions, reusable scratch, batch-aware z indexing, and reduction.
   - Gate dL/dimage parity, isolated timestamps, and integrated full-3D step time.

2. `OPEN EXACT` Dual-output-channel pointwise tile.
   - Amortize source staging without increasing workgroup dimensions.
   - Restrict first tests to repeated 256<->768 and 512<->1536 shapes.

3. `PARKED` Rectangular 8x16 pointwise tile and broad shared-W variants.
   - Correct implementations measured neutral or worse in integrated timing.
   - Keep their forks as negative evidence; do not silently retry the same layout.

4. `OPEN EXACT` Remaining local fusions.
   - Shape-gated SE/spatial/pointwise forward and backward fusions are documented.
   - Preserve required backward activations; whole-CLIP fusion is not feasible
     under WebGPU synchronization and workgroup-memory limits.

5. `CONDITIONAL` Attention backward.
   - Fresh profiling must exceed 5% before an implementation fork.
   - Candidates: vec4 private arrays, dP recomputation, chunked Q/dO staging,
     and two-dispatch designs that avoid large gather loops.

6. `OPEN EXACT` Raster/CLIP IO aliasing for anisotropic batches.
   - Remove dense image and dL/dimage copies where storage offsets/alignment allow.
   - Keep ownership and buffer-lifetime contracts explicit.

### P2: Approximate CLIP, Precision, And Distillation

1. `PARKED` Blanket f16 weights.
   - Payload halved, but strict dL/dimage quality and integrated timing did not win.

2. `OPEN APPROX` Sensitivity-ranked mixed precision.
   - Generate an offline per-layer report for skip, low-rank, f16/int8/int4,
     and outlier-branch perturbations.
   - Measure embedding error, dL/dimage cosine/directional error, gradient norm,
     and prompt ranking.
   - Keep loss, normalization, attention reductions, and sensitive outliers in f32.

3. `OPEN APPROX` Nunchaku-inspired int4/outlier path.
   - WebGPU lacks a native int4 tensor-core path; packed int4 can lose to unpack cost.
   - Start only after sensitivity ranking identifies large tolerant pointwise layers.
   - Consider low-rank/outlier side branches before a blanket quantized model.

4. `OPEN APPROX` Layer skipping and low-rank CLIP.
   - Test residual-block skipping, truncated image towers, and low-rank pointwise
     approximations with periodic full-CLIP correction.

5. `OPEN APPROX` Proxy/teacher schedule.
   - Implement `tools/clip/proxy_quality_gate.ts` for embedding cosine,
     dL/dimage cosine, gradient norm ratio, prompt-ranking agreement, runtime,
     and fixed-wall-clock teacher convergence.
   - Lanes: cached gradients, grid/low-resolution proxy, truncated CLIP, then a
     learned splat-render gradient student.

6. `OPEN` Hessian/Jacobian ideas.
   - Hessian information is useful for offline sensitivity/ranking, not as a
     magic browser int4 optimizer.
   - Jacobian or random-projection distillation must be validated on prompt
     gradients, not only static embedding similarity.

### P2: Profiling And Engineering Discipline

1. `OPEN` Real hardware-counter capture.
   - Select/install full Xcode and capture Metal bandwidth, occupancy, cache,
     register spill, and threadgroup-memory counters for pw_bwd, pw,
     spatial_bwd, attention backward, and anisotropic raster backward.
   - Keep Chrome/Dawn traces for queue depth, command-buffer duration, and over-serialization.

2. `PROMOTED` Measurement overlay and timestamp harnesses.
   - Keep normal single-submit wall, split-submit wall, and GPU timestamp claims separate.
   - Serialize variants to avoid GPU contention and report median/min/max.

3. `PROMOTED` Fork-and-gate protocol.
   - Copy risky shader variants into `experiments/clip_forks/vNN_*` with snapshots.
   - Correctness first, isolated timing second, integrated timing and quality last.
   - Record rejected variants; do not mangle the only hot shader in place.

## Closed Or Settled Questions

- 2D `NUDGE` is implemented and documented.
- Gaussian splatting is the browser-first 3D representation; a NeRF/Dream
  Fields ray marcher would be substantially slower for this runtime.
- Dream Fields' learned semantic prior was CLIP rather than diffusion; its
  other important contributions were camera/background sampling and geometric
  regularization.
- Black remains the intended display background. Background augmentation stays
  an optional training-only experiment.
- The sharp center square was tile overflow, not a pasted CLIP grid. Dynamic cap
  4096 and occupancy telemetry fixed and exposed it.
- The tiny-ball collapse came from incorrect centering/coverage incentives;
  opacity fading came from the old alpha regularizer. Both failures are recorded.
- Feature dimension is 32, not 22, because it maps cleanly to eight vec4 groups.
- Direct feature loss into CLIP is not promoted without a teacher-defined target.

## Source Map

- Current implementation story: `docs/BLOG_PROGRESS_NOTES.md`
- Convergence math and literature: `docs/SPLAT3D_CONVERGENCE_PLAN.md`
- Performance measurements: `docs/SPLAT3D_PERF_NOTES.md`
- Corrected speed ladder and churn review: `docs/SPLAT3D_REFLECTION_REVIEW.md`
- Historical ablation chronology: `docs/SPLAT3D_ABLATION_QUEUE.md`
- Deep research report: `docs/deep_research_clip_splat_report.md`
- CLIP fork trail: `experiments/clip_forks/README.md`
- Detailed worker/research notes: `agent_notes/optimization_session/`

High-value detailed notes:

- `agent_clip_proxy_distillation_2026_07_08.md`
- `nunchaku_clip_approx_2026_07_08.md`
- `v20_agent_splitk_pw_bwd_design.md`
- `v20_agent_remaining_local_fusions.md`
- `v19_agent_real_gpu_profiling.md`
- `multiview_raster_worldtube_review.md`
- `static_multiview_worldtube_followup.md`
- `aniso_live_raster_worker.md`
- `splat3d_feature_splat_fork_2026_07_10.md`
- `convergence_priors_2026_07_10.md`
