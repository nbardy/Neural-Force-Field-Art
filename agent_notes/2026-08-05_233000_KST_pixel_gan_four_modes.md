# Pixel GANs — four drawing modes

**Date:** 2026-08-05 ~23:30 KST  
**Goal:** Branch the pixel density critic into four 2D GAN games with new gallery names; fully implement CPU + fused WebGPU.

## Modes (docs/PIXEL_DISC.md)

| Kind | Gallery | Game |
|---|---|---|
| `vec-field` | `Pixel · VecField` | Per-cell \(\hat V\) vs \(F\) at cell centers |
| `next-frame` | `Pixel · NextFrame` | Predict \(D_1\) from \(D_0\) (drawing→drawing) |
| `real-fake` | `Pixel · RealFake` | BCE live density vs same-B random spray |
| `inpaint` | `Pixel · Inpaint` | Masked block completion (~25% area) |

Shared trunk: soft splat → conv3×3 → soft codebook. Reverse-mode gen through `D(pos')` → `extGrads`. No JVP.

## Files

- `src/core/gan/pixel_disc.ts` — multi-kind CPU oracle
- `src/render/webgpu/pixel_disc_wgsl.ts` — fused WGSL codegen (kind switch)
- `src/render/webgpu/pixel_disc_train.ts` — kind-aware pass order + `maskSeed`
- `src/main.ts` — four gallery pieces; `pixelDisc.kind`
- `tools/pixel_disc_test.ts` — splat FD + disc descent + GPU smoke ×4
- `docs/PIXEL_DISC.md`, `docs/DESIGN_SPACE_PARTICLE_ART.md`

## Verified

```text
bun tools/pixel_disc_test.ts  → ALL PASS
```

CPU: disc loss drops and gen `dF` finite for all four.  
GPU: shader validates + finite nonzero `extGrads` for all four (Metal).

## Caveats

- Agree+Disagree still uses 2 extGrad slots — cannot combine with pixel disc in the same frame.
- G=16 default (was 32 tile-mean path); critic is single-threaded `@workgroup_size(1)`.
- Old piece `Adversary · Pixel Density` removed.

## Next

- Live visual A/B of the four gallery pieces.
- Optional: third extGrad slot so pixel disc can stack with Agree+Disagree.
