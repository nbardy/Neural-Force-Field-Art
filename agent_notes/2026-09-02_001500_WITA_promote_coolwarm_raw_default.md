# Promote the supplied Coolwarm RAW WTA10 recipe

## Goal

Make the supplied piece-17 dock recipe the fresh-install default while keeping
the existing gallery entries and persisted-link indices stable.

## Verified facts

- The existing promoted WTA10 entry already matched the supplied particle
  count, sample count, velocity, drive, G LR, reset rate, decay/alpha-fade,
  ghost/curl settings, and dual HashGrid architecture.
- The supplied recipe differs in discriminator LR (`0.00002660725059798809`),
  WTA relaxation epsilon (`0.1`), and initial colour mode (`surprise-raw`,
  `coolwarm`).
- The app's diagnostic colour path is orthogonal to curl ink, so the default
  remains on the normal alpha-fade renderer while selecting RAW/coolwarm.

## Changes

- Added an explicit optional `ArtPieceConfig.colorMode` initial mode.
- Added the supplied RAW/coolwarm mode to the new default entry.
- Updated the default entry's D LR and WTA epsilon.
- Updated the fresh-install React state to seed from the piece's colour mode.
- Expanded `tools/gallery_config_test.ts` to assert the new recipe.

## Validation

- `bun tools/gallery_config_test.ts` — pass.
- `npm run build -- --no-cache` — pass.

## Next action

Deploy with `tools/deploy.sh` and verify both remote commit heads.
