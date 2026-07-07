# Spec — WGSL 2D Gaussian-splat rasterizer (fwd + bwd) + fused Adam

Goal: the drawing half of the prompt→splats page — ~200K trainable 2D
Gaussians rasterized at 256×256, differentiable, optimized by Adam against a
gradient image that will come from the CLIP backward (docs/clip_backward_spec.md,
being built in parallel — do NOT depend on it; your gradient input is just a
dense buffer).

READ FIRST:
- `docs/splat_raster_brief.md` — recon of the Metal reference implementation
  (~/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat). Follow its
  kernel structure (v11 fixedbin binning, v6 backward shape) and its listed
  Metal→WGSL replacements. You may read the referenced Metal sources directly
  for exact formulas.
- `tools/clip/README.md` + `src/clip/vision_wgsl.ts` + `tools/kernel_test.ts`
  — this repo's codegen/test house style (pure codegen files with zero
  imports, baked literals, loud asserts, bun-webgpu Metal test harness,
  validation-scope pipeline creation).

## Data model (OURS — differs from the reference; this is decided)
SoA f32 raw parameter buffers, G splats (default 200_000):
- `mean[G,2]` px; `logScale[G,2]`; `theta[G]`; `colorRaw[G,3]`; `opacityRaw[G]`
Reparameterizations computed IN-KERNEL (and chained in backward — Adam updates
the RAW params): `scale = clamp(exp(logScale), 0.3, 64.0)` px;
`color = sigmoid(colorRaw)`; `opacity = sigmoid(opacityRaw)`;
conic = inverse of R(θ)·diag(scale²)·R(θ)ᵀ (derive the 3-term packed a,b,c and
its full Jacobian w.r.t. logScale/theta — the reference backprops only to the
conic; we go all the way to raw params).
- NO depth: painter's order = splat INDEX (ascending). The per-tile ascending
  integer-ID sort from the reference is exactly the right order recovery.
- Alpha/visibility math, thresholds, tile size 16, per-tile cap 2048, fixedbin
  constant-stride bins, transmittance cutoff 1e-4: as in the brief.
- Forward output: **NCHW planar f32 [3][256][256] in [0,1]** (this later binds
  directly as the CLIP encoder's input slot — that's why planar). Uniform
  background color (default 0.5 gray), composited as `accum + T·bg`.
- Backward input: dense `dL/dpixels` planar [3][H][W] f32. Outputs: grads for
  all raw param buffers (zeroed by a clear pass each step).
- Gradient accumulation: WGSL has no f32 atomicAdd. Start with the simplest
  CORRECT scheme (fixed-point atomicAdd<i32> with a documented scale, or u32
  CAS loop) gated by the gradcheck; then, if profiling shows contention, adopt
  the v6 two-level workgroup reduction (256 pixels → 1 accumulation per splat
  per tile). Correct-first, fast-second; keep the fast path behind the same
  gradcheck gate.

## Kernels (src/splat/raster_wgsl.ts — pure codegen; src/splat/raster.ts —
runtime class owning buffers + encode; src/splat/adam_wgsl.ts — optimizer)
1. `count/emit` fixedbin binning (thread/splat, exact ellipse-vs-rect test)
2. `tile_forward` — 1 workgroup(256)/tile: shared ID bitonic sort (256-thread
   stride variant), chunked shared staging of splat params, front-to-back
   composite, early-out, writes sorted ids back + tile_stop_counts + image.
3. `tile_backward` — replay prefix for T_final, reverse recurrence per the
   brief's math, chain through conic→(logScale,theta) and sigmoid/exp reparams,
   accumulate raw-param grads. UNIFORM BARRIERS (per-pixel end_i gates
   contributions only — brief's critical rule).
4. `adam` — thread/param over all raw buffers: m/v (one f32 buffer each, same
   layout as params), bias-corrected update, per-group LR from a small uniform
   (groups: mean, logScale, theta, color, opacity; defaults lr=1e-2 for mean,
   5e-3 others — tune in the demo), optional global grad-scale uniform (for
   undoing the fixed-point scale for free). Plus a `clear_grads` pass.

## Acceptance gates (tools/splat/raster_test.ts + tools/splat/fit_demo.ts, bun)
1. **Gradcheck**: small scene (64 splats, 32×32, cap 256): central finite
   differences THROUGH THE GPU FORWARD (deterministic) vs GPU backward, on
   ~200 randomly-probed raw params across all 5 buffers + a fixed random
   dL/dpixels. Per-param-type ε (e.g. 1e-2 px for means, 1e-3 for raws).
   Gate: rel err < 2e-2 on ≥95% of probes (fp32 FD noise is real; report the
   distribution). Probes crossing the alpha/visibility threshold may be
   excluded when detected (report count).
2. **L2 image-fit convergence** (end-to-end, no CLIP): 4096 splats @128²
   fitting a deterministic synthetic target (e.g. smooth 3-color gradient +
   a filled circle, generated in JS): dL/dpixels = 2(render−target)/N fed to
   the backward, 500 Adam steps. Gate: final L2 < 10% of initial. Write
   before/after .ppm files (P6) so a human can eyeball them; print paths.
3. **Perf report** (no gate): 200K splats @256²: forward ms, forward+backward
   ms, at a spread-out random init (report; hope: fwd+bwd well under 15 ms).
   Follow the warmup + shared-GPU rules from tools/clip/README.md gotchas.

## Non-goals
Densification/pruning (reference has none either), depth/3D, SH, display/UI
(page comes later), CLIP wiring, fp16. Do not touch src/clip/, src/main.ts,
src/render/, tools/clip/.
