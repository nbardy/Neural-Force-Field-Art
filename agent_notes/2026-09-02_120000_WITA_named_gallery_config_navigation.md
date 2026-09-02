# Named gallery versus configuration navigation

## Goal

Keep the bottom navigation for curated named artworks while moving experiment
presets (shape losses, Pixel critic variants, and adversary K/observer variants)
into the right-side configuration controls.

## Change

- Added optional `ArtPieceConfig.named` metadata.
- Marked exploratory shape, adversary-control, Pixel-critic, and generic field
  variants as `named: false`.
- Added a reusable `SelectRow` control and a `config` section listing those
  presets. Selecting one still uses the existing piece rebuild path, so every
  existing recipe remains available and its settings appear in the shared dock.
- The bottom radio strip now filters to entries whose `named` flag is not false.
- Existing piece indices, `?piece=`, and `?dock=` resolution remain unchanged.

## Scope note

This reorganizes navigation and ownership but does not yet extract every inline
gallery object into factory functions. The next cleanup can do that separately
without changing the compatibility contract.

## Validation

- `npm run build -- --no-cache` passed.
- `bun tools/gallery_config_test.ts` passed.
- `git diff --check` passed.
