# Named pieces and gallery taxonomy

## Goal

Preserve the supplied dock capture as a named artwork, then assess whether the
gallery's labels expose artistic dimensions or implementation experiments.

## Verified findings

- Gallery indices are persisted in dock links, so the existing piece at index
  17 must not be renamed or replaced in place.
- `Adversary · WTA K=8`, `Pair WTA K=4`, `Tri WTA K=6`, and `Quad WTA K=6`
  differ in observer encoding and K; they are related experiments, but their
  names mix the artistic family (`Adversary`) with implementation settings.
- `Pixel · VecField`, `NextFrame`, `RealFake`, and `Inpaint` are distinct pixel
  critic objectives and should remain separate modes under one Pixel family.
- `Spiral`, `Vortex`, `Galaxy`, and `Spiral Cover` share a shape-reward family,
  but Spiral/Galaxy include direct spiral attraction while Cover is a distinct
  coverage objective.

## Change made

Added `Sand of Times` as a new append-only gallery entry. It captures the
supplied pair / rotation-scale-adjusted / WTA10 / wider-predictor recipe with
RAW/inferno colour, blend 0.8, and the supplied learning rates.

## Recommendation

Keep named pieces as immutable curated recipes. Treat Adversary, Pixel, and
Shape Reward as top-level families, with settings in the right-side dock. Use
the dock for exploration; only promote a configuration to a named piece when
the visual result has a name and a reproducible recipe.

For the shape family, a single `Shape reward` selector can expose `None`,
`Spiral`, `Galaxy`, `Galaxy field`, and `Galaxy cover`, while retaining a
compact advanced panel for objective weights. Do not collapse the adversary
observer/K settings into shape reward: those change the game, not merely the
appearance.

## Validation

- `bun tools/gallery_config_test.ts` passes after the change.
- Production build remains required before deployment.
