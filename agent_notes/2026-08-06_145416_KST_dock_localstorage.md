# Dock config persistence (localStorage + gallery switch)

## Goal

Keep dock settings across page refresh, without poisoning gallery recipes
when switching art pieces.

## Changes (`src/index.tsx`)

- Persist dock state under `localStorage` key `nffa.dock.v2` (v1 retired —
  it carried raw-vector / weight=0 onto Pair and killed soft-angle swirls).
- Restore on load when the blob validates; invalid / missing → piece 0
  defaults.
- Explicit URL dock knobs (`?adv`, `?gLR`, `?dLR`, `?drive`, `?color`, …)
  still win over a saved dock so shareable links stay honest.
- Gallery click uses `runtimeForPieceSwitch`: **re-adopts** the piece's
  encoding / target / loss / adversary kind / K / ε. Border may persist;
  archPreset resets. (Earlier “keep WTA K across pieces” stranded Quad on
  Pair’s K=4.)
- Live dials: preserved on same-piece compile rebuilds and first paint from
  storage; **re-synced from the new piece** on gallery switch (so Pair does
  not inherit Spiral's particle count / advWeight=0 / velocity colour).

## Unresolved / follow-ups

- No “reset to piece defaults” button yet (piece switch is the reset).
- Piece index is not URL-addressable; refresh restores last piece from
  storage only.
- Soft-angle still goes laminar/diagonal even with correct knobs — see
  `agent_notes/2026-08-06_214641_KST_angle_disc_diagonal_regression.md`.
