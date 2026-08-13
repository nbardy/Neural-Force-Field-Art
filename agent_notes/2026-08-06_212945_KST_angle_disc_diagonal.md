# Angle discriminator → diagonal / learns nothing

## Report

User: the angle-based discriminator used to make amazing swirls; now it
learns nothing or goes all diagonal. Unsure how much earlier work was
committed.

## Verified cause (dock, not WGSL)

`runtimeForPieceSwitch` (dock localStorage work,
`agent_notes/2026-08-06_145416_KST_dock_localstorage.md`) **kept**
encoding / loss / K across gallery clicks and only updated `adversaryKind`.

Pair WTA’s swirls need:

- encoding `pair-rotation-scale-adjusted`
- loss `soft-angle` (via `adversaryLossOf`)

Quad (and other raw pieces) leave the dock on **RAW VECTOR**. Clicking Pair
then ran the Euclidean amplitude game on pair context → diagonal / flat
aesthetics, while the HUD still said you were on Pair.

Soft-angle WGSL path was not removed by the NaN harden (`isFiniteF` is on
raw resid / surprise / Adam; soft resid path unchanged).

## Fix

`runtimeForPieceSwitch` now re-adopts the piece’s baked `encoding`,
`target`, and `loss`. Live dials stay outside runtime. WTA K/ε kept only
when switching WTA→WTA with prior `k ≥ 2`.

## If still diagonal after the fix

1. Click **ANGLE** in the adversary dock (or re-click Pair in the gallery).
2. Or clear `localStorage` key `nffa.dock.v1` and reload — an old blob can
   still restore RAW while sitting on the Pair piece index until you switch.

## Commit context

NaN harden was committed earlier (`89269ba`). Dock persistence landed in
the working tree / later commits and is what changed Pair’s effective loss.
