# Pixel-space discriminator brainstorm

## Goal

Answer: can the particle-art engine grow a cheap pixel-space discriminator
(conv2d → embedding LUT + pooling), and do we need a differentiable
rasterizer? How far are we?

## Verified facts (repo state @ `a92b357`)

Two apps share infrastructure:

1. **Particle engine** (`main.ts`): fused advect + fused field trainer +
   relational adversary (particle/force space, not pixels) + forward-only
   compute-splat renderer (`render/webgpu/splat.ts`).
2. **Prompt→splats** (`splat_page.ts`): differentiable 2D Gaussian raster
   (`splat/raster_wgsl.ts`) + CLIP / Feature8 / direct pixel-buffer critics.
   Gradcheck-verified; gradients flow pixels → splat params.

Particle adversary (`docs/ADVERSARY_STATUS.md`): observes relational tuples of
positions + predicts `F(x)` (or post-velocity). Never sees an image.

Open thread: fused soft-angle / relative-scale discriminator grads still
disagree with tfjs oracle
(`agent_notes/2026-07-30_040000_KST_fused_discriminator_gradient_debug.md`).

Design doc already scopes pixel feedback as future art direction
(`DESIGN_SPACE_PARTICLE_ART.md` §4–5: feed splat buffer back; flow-match to
reference image) — not implemented for particles.

## Proposal under discussion

Cheap pixel critic for particle frames:

```text
render (low-res density/HDR) → depthwise/cheap conv → embedding LUT
  → pool → score / multi-head residual
```

Questions raised:
- Need differentiable rasterizer?
- Distance from current codebase?

## Assessment (hypotheses + engineering gaps)

**Need a differentiable rasterizer?** Yes, if the field (generator) should
receive pixel-space gradients. A frozen/non-diff render can still train a
discriminator on detached frames, but cannot push the field. Soft
density/alpha-composited accumulation (simpler than full Gaussian splat
backward) is enough for particles.

**Reusable:** verified diff raster + CLIP pixel-grad seam in the splat app;
atomic scatter pattern; field BPTT chain already exists for pos→weights.

**Missing for particle pixel-disc:**
1. Differentiable point/density render w.r.t. particle pos (vis splat is
   forward-only).
2. Trainable image critic (not frozen CLIP); no CNN op set in scalar AD IR —
   hand-written WGSL like CLIP/splat backward is the template.
3. Frame-batch training loop; O(N) render cost vs current train-few/advect-many.
4. Grad chain: dL/dpixels → dL/dpos → (existing BPTT) → field weights.

**Cheap LUT critic:** compatible with repo instincts (hash-grid embeddings,
Feature8 residual decoder). Softmax/bilinear-pooled patch embeddings avoid
hard argmax. Prefer low-res density (e.g. 64²–128²) over full retina canvas.

## Next actions (if pursued)

1. Finish fused adversary soft-angle/scale grad parity (open bug) before
   stacking a second critic.
2. Prototype detached low-res density buffer → tiny pooled embedding critic
   (discriminator-only) to see if pixel features beat relational surprise.
3. If yes: add soft alpha-accumulate backward w.r.t. particle pos (subset of
   existing splat tile_backward), wire into field trainer via existing BPTT.
4. Keep as selectable gallery objective, not silent replacement of relational
   adversary.

## Unresolved

- Art-tuning knobs for current relational pieces vs new pixel critic priority.
- Whether CLIP-as-frozen-critic (already built) is enough vs trainable disc.
