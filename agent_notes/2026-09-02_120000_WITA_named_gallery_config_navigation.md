# Named gallery versus configuration navigation

## Goal

Keep the bottom navigation for curated named artworks while moving experiment
presets (shape losses, Pixel critic variants, and adversary K/observer variants)
into the right-side configuration controls.

## Change

- Added optional `ArtPieceConfig.named` metadata.
- Marked exploratory shape, adversary-control, Pixel-critic, and generic field
  variants as `named: false`.
- Added a reusable `SelectRow` control and an `objective` section. Its
  Adversary/Shape segmented control acts as the top-level radio choice, and a
  second selector lists only the recipes in that family. Selecting one still
  uses the existing piece rebuild path, so every existing recipe remains
  available and its settings appear in the shared dock.
- The bottom radio strip now filters to entries explicitly marked `named: true`.
- Existing piece indices, `?piece=`, and `?dock=` resolution remain unchanged.

## Scope note

This reorganizes navigation and ownership but does not yet extract every inline
gallery object into factory functions. The next cleanup can do that separately
without changing the compatibility contract.

## Validation

- `npm run build -- --no-cache` passed.
- `bun tools/gallery_config_test.ts` passed.
- `bun tools/pixel_disc_test.ts` passed, including the GPU smoke checks.
- `git diff --check` passed.
