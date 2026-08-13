# Angle discriminator → diagonal / “learns nothing”

**Status:** diagnosed; UI poison fixed in WIP dock; art dynamics still open  
**Trigger:** user remembered soft-angle adversary making amazing swirls; now
noise / laminar diagonals.

## What was committed earlier

On `main` (ahead of / matching origin as of this note):

| commit | what |
|---|---|
| `a92b357` | relational adversary + surprise renderer |
| `89269ba` | Quad/Tri NaN harden (SELU clamp + finite gates) |
| `d5d6333` | pixel GANs + field arch + cover/center |

**Not committed:** dock `localStorage` persistence + D/G sparklines in
`src/index.tsx` / `src/ui.css` (see
`agent_notes/2026-08-06_145416_KST_dock_localstorage.md`).

## Verified: soft-angle math is not broken

`bun tools/train_wta_test.ts` — force/post-velocity × soft-angle /
angle-relative-scale / angle-scale-hold all report **DISC/GEN cos = 1.0** vs
tfjs. The 2026-07-30 fused discriminator grad debug note is stale; parity
shipped with the relational adversary.

NaN harden only adds `isFiniteF` selects + SELU ±80 clamp; it does not change
the soft-angle Jacobian when values are finite.

## UI poison that made Pair look dead (fixed in WIP)

Early dock persistence / gallery-switch policy could:

1. Carry **RAW** loss from Quad/Tri onto Pair (amplitude shortcut → diagonal).
2. Carry **`advWeight = 0`** from Spiral/aesthetic pieces onto Pair (no G
   reward → “learns nothing”).
3. Carry **K=4** from Pair onto Quad (HUD showed `wta k=4` on a K=6 piece).

WIP `index.tsx` now:

- re-adopts piece baked **encoding / loss / target / K / ε** on gallery switch;
- on storage restore, replaces `advWeight === 0` with the piece recipe weight
  when the piece is an adversary;
- retires `nffa.dock.v1` (key is `nffa.dock.v2`).

Live gate after fix: Spiral → Pair → Quad → Pair keeps **ANGLE + reward 0.015**
and Quad shows **RAW**.

## Deeper art issue (still open)

Even with correct Pair soft-angle (`w=0.015`, fused), ~35–100s soaks still
settle toward **laminar / diagonal streaks**, not swirls:

- Pair ANGLE + surprise renderer (trails intentionally 0): noise / faint diagonal.
- Pair A+S HOLD + trails 0.92 + VEL: sharp diagonal fans.
- Point WTA K=8 + ANGLE + trails: more edge turbulence, still diagonal-dominated.

**Hypothesis (not fully proved):** on the rotation-quotiented pair observer,
a near-constant world-frame force is hard for D to predict from `u = r` alone
(pair-frame direction depends on hidden orientation). Soft-angle then pays G
for laminar fields → diagonals are an easy high-residual attractor. Point
observer avoids that free lunch but still drifts diagonal without an
isotropy/curl pressure (standalone pieces use `ZERO_FIELD_LOSS`).

Surprise pieces keep `SPLAT_DECAY_BY_RENDERER.surprise = 0` on purpose (stale
trails would lie about residual); swirls are harder to *see* there than on
`alpha-fade` WTA K=8.

## Defaults that surprise users

| piece | default loss |
|---|---|
| Pair (`pair-rotation-scale-adjusted`) | **soft-angle** |
| Single / Tri / Quad / WTA K=8 | **raw-vector** |

Dock **ANGLE** is a live override; Tri/Quad are not angle games until you flip
them.

## Next actions (pick with user)

1. Retune Pair / point ANGLE with a small fused `W_ISO` (or weak curl) so
   laminar diagonals are expensive.
2. Ship a named gallery recipe “Angle swirl” = point or pair soft-angle +
   ghost trails + light isotropy (and maybe Chaos Weave–style mix).
3. Commit the dock persistence WIP once K/weight/loss re-adopt policy is
   signed off.
4. Optional longer soak A/B: RAW vs ANGLE vs A+S HOLD with identical ink, to
   quantify diagonal bias (velocity histogram axis alignment).
