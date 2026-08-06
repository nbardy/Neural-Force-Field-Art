# Pixel-space discriminator — plan & JVP decision

## JVP vs reverse-mode

**Do not use JVP for this.** JVP (forward-mode) in this repo is for exact
chaos/div probes: two JVPs give ∂F/∂x and ∂F/∂y at a point
(`ad/rollout.ts` `probes: "jvp"`). That is cheap when the *number of
directions* is tiny.

A pixel critic ends in a **scalar** residual. Full gradients w.r.t. field
weights via JVP would need one forward sweep per weight (or Hutchinson
noise). Reverse-mode is one backward: critic → soft density → virtual
positions → F → weights. Same discipline as the relational adversary's
`fieldGrad` and the splat raster `tile_backward`.

## Design (shipped)

```text
Disc:
  D = soft_bilinear_splat(pos)          # low-res G×G, detached from W
  ŷ = Critic(D)                         # 3×3 → soft codebook → GAP → MLP→ℝ²
  y = mean_i F(pos_i)                   # batch mean force
  minimize soft residual(ŷ, y)

Gen (disagree):
  pos' = pos + dt · F(pos; W)           # virtual one-step (makes D depend on W)
  D' = soft_bilinear_splat(pos')
  ŷ' = Critic_frozen(D')
  maximize residual(ŷ', stopgrad(y))
  → VJP through density (diff raster) and F → extGrads → FusedTrainer pass B
```

Critic is cheap by construction: depthwise-ish 3×3 (E channels), K-way soft
embedding LUT, global average pool, tiny MLP. No MobileCLIP. No full-res
canvas. Default G=64, E=8, K=32, B≤512.

## Why differentiable rasterizer

Yes for the generator path: without ∂D/∂pos', pixel loss cannot reach W
through the image. Soft bilinear splat + VJP is enough (subset of the
existing Gaussian splat backward). Disc-only training can use detached D.

## Files

- `src/core/gan/pixel_disc.ts` — CPU oracle
- `src/render/webgpu/pixel_disc_wgsl.ts` — WGSL codegen
- `src/render/webgpu/pixel_disc_train.ts` — host trainer + extGradsBuf
- `tools/pixel_disc_test.ts` — FD + GPU/CPU gates
- Gallery piece + `main.ts` wiring

## Status

Implemented and gated (`bun tools/pixel_disc_test.ts` ALL PASS).

Shipped:
- CPU oracle with FD-checked splat VJP + critic VJP
- Fused WGSL trainer (soft density + codebook critic + fieldGrad → extGrads)
- Gallery piece `Adversary · Pixel Density`
- Docs: `docs/PIXEL_DISC.md`

JVP deliberately unused — reverse-mode only.
