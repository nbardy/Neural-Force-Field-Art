# Soft-angle swirls regression vs dock persistence

## Symptom

Angle-based discriminator used to make Pair swirls; now "learns nothing" or
goes all diagonal.

## Verified cause (high confidence)

Not a broken soft-angle Jacobian. Pair's gallery recipe is still explicit
`soft-angle τ=0.05` on `pair-rotation-scale-adjusted` (`main.ts`). Comments
there already say soft-angle made the swirls; **raw-vector** on that observer
collapses into amplitude / shear cheats (diagonal clip art).

The dock persistence work (same day) initially **kept loss across gallery
switches**, so visiting Spiral/Tri/Quad stranded Pair on `raw-vector` and/or
`advWeight=0`. That matches both symptoms:

- all diagonal ≈ raw-vector amplitude + drive saturation
- learns nothing ≈ generator reward weight 0

NaN hardening (89269ba) is a weak suspect for healthy soft-angle grads;
parity tests still gate soft-angle vs tfjs.

## Fix in `src/index.tsx` (this session)

- Gallery switch re-adopts piece encoding / target / loss / kind (live dials
  still re-sync from the new piece).
- Reload adopts the piece's baked adversary recipe (not a poisoned saved
  loss); live dials still restore.
- `advWeight === 0` on an adversary piece is replaced with the gallery weight.
- Delete leftover `nffa.dock.v1` on load (`nffa.dock.v2` is current).

## Recovery for a live browser

1. Open **Adversary · Pair WTA K=4**
2. Dock objective row must read **ANGLE OPPONENT GAME** (loss = ANGLE, not RAW)
3. Reward / game weight should be ~0.015 (not 0)
4. If still wrong: DevTools → `localStorage.removeItem('nffa.dock.v2')` → reload
5. Optional deep link:
   `?advLoss=soft-angle&advTau=0.05` on Pair

## If diagonals persist with confirmed soft-angle

Then try lower drive (`?drive=0.35`) — direction-only loss does not punish
|F|≫τ, so both-axis clip can still paint diagonals.
