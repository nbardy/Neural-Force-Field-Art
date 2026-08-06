# Gallery cleanup — recipes vs dock axes

**Date:** 2026-08-06  
**Goal:** Stop exploding the gallery with arch×look duplicates; keep rows as
loss/game recipes and move arch + look to dock controls.

## What changed

### Gallery (17 recipes; was ~27)

| Keep | Notes |
|---|---|
| Spiral | was Ghost (+ Trails folded into look) |
| Vortex | was Ghost |
| Galaxy | was Ghost/Clean/Fourier |
| Spiral Cover | was Clean/Ghost/Fourier/SIREN/HashGrid/Fourier+SIREN |
| Neural Field · Max Chaos | was + SIREN/Fourier/HashGrid Chaos variants |
| Neural Field · Species | classes load-bearing; no arch dock |
| Adversary / Pixel pieces | unchanged (`createField` recipes) |

### Dock axes

- **look** (`lookEditable`): Ghost / Clean / Trails → splat `decay` presets (live)
- **arch** (`archEditable` + `archDock`): aesthetic single-head list, or dual
  list for Max Chaos; `applyArchDockPreset` preserves α / semantic / classes

### Code

- `ArtPieceConfig.lookEditable`, `archDock`
- `ARCH_DOCK_DUAL`, `archDockPresets`, `applyArchDockPreset`
- `INK_LOOK_DECAY` / `inkLookFromRenderer`
- Field resolve: `fieldArch` (incl. dock override) wins over `createField`

## Verified

- Gallery list loads; dual preset merge keeps α=0.7 when swapping Fourier
- Piece count 17

## Follow-ups

- Live-watch Spiral Cover look/arch toggles
- Optional: adversary pieces with shared game could later get dual dock too
