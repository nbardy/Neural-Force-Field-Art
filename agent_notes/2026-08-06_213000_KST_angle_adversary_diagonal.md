# Angle adversary “no swirls / all diagonal” — root cause

**Date:** 2026-08-06  
**Symptom:** Pair soft-angle used to make great swirls; now looks inert or
shears into diagonals.

## Verified (not a math regression)

`bun tools/train_wta_test.ts` — force + soft-angle disc/gen grads vs tfjs:
**cos = 1.0000000**. Fused soft-angle is numerically fine.

## Actual cause

Dock persistence (`nffa.dock.v1`) + `runtimeForPieceSwitch` used to **carry
observer/loss/K across gallery clicks**, and live dials (including
`advWeight`) across piece switches.

Recipe:

1. Sit on Spiral / Galaxy → `loss=raw-vector`, `advWeight≈0`, velocity colour.
2. Click **Adversary · Pair WTA K=4**.
3. Pair runs with **raw-vector** (amplitude game → diagonal shear) and often
   **weight 0** (learns nothing), not soft-angle + weight 0.015.

Historical “diagonal stripes” notes for density discs are a different failure
mode; this was config contamination.

## Fix

- `runtimeForPieceSwitch` re-adopts piece encoding / target / loss / kind.
- Live dials re-sync from the new piece on gallery switch (same-piece compile
  rebuilds still preserve dials; first paint from storage still applies
  saved dials).
- Storage key → `nffa.dock.v2` (clears contaminated v1 blobs).
- Pair piece declares explicit `loss: { tag: "soft-angle", tau: … }`.

## How to recover in browser

Hard refresh after pull. If still weird: DevTools → Application → Local
Storage → delete `nffa.dock.v*`, reload, select **Adversary · Pair WTA K=4**.
Dock should show loss **soft-angle**, observer scale-adjusted, weight ~0.015.
