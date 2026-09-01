# Gallery recipe promotion — 2026-09-01 23:30 WITA

## Goal

Append the supplied Point/WTA-K10/soft-angle/dual-HashGrid/curl dock recipe to
`src/main.ts` and make it the fresh-install default without changing existing
gallery indices, dock-link compatibility, UI, or renderer shaders.

## Verified before editing

- `GALLERY` already has an explicit append-only boundary near the end of
  `src/main.ts`.
- The current default is `Adversary · Pair · HashGrid · Curl`, with the older
  Pair/K4/relative-scale recipe.
- `startLoop` previously hardcoded initial train batch `256`, discriminator LR
  `3e-3`, and wrap border; these are config-promotable in `main.ts`.
- The UI is intentionally out of scope and passes its own runtime border
  override, so config defaults must not alter that UI contract.

## Completed edits

- Added optional `sampleRate`, `discriminatorLearningRate`, and `border`
  metadata to `ArtPieceConfig`.
- Consumed those defaults in `resolveLiveGameControls` and `startLoop`.
- Appended one new gallery object and pointed `DEFAULT_PIECE_NAME` at it.
- Did not add optimizer-group metadata: no trainer wiring is allowed by scope,
  and an unconsumed field would be misleading.
- Added `tools/gallery_config_test.ts` as a focused pure config gate.

## Verified limitation

`src/index.tsx` passes a wrap border override during startup. The new piece's
declared reset border is therefore honored by direct `startLoop` callers but
cannot override the UI's existing runtime contract without editing the UI.
