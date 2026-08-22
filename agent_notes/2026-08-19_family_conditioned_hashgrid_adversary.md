# Family-conditioned hashgrid adversary — "RGB Families" piece

Status: LANDED (implementation complete, gates green). Not yet deployed.
Started 2026-08-19. Author: agent session (Opus 5).

## Goal (user's words, restated)

A new gallery piece:

- Hashgrid field, deeper than the default piece's `[32,32]`.
- The generator is conditioned on **coordinate AND family**, families = R/G/B (C = 3).
- Adversarial (fused WTA relaxed predictor), **the discriminator does NOT see the
  family** — it only sees the coordinate. So the conditional `P(F | x)` it must
  model is a C-mode mixture, and the generator is paid for making the families
  disagree at the same coordinate.
- HUD logs the payoff **per family, independently** — the open question is
  whether the three families balance, collapse onto each other, or trade roles
  over time.

## Design decisions (and why)

### D1 — family enters through the GRID, not through one-hot MLP channels

Existing `classes` support (the `Neural Field · Species` piece) appends a
one-hot to **head 1's layer-0 input** and is **raw-encoding only**
(`layoutField` throws on `encoding != raw && classes > 0`; the tfjs field and
the fused adversary throw too).

Instead of loosening that, the family is expressed as an **encoding** property:
the hashgrid feature table gets `planes = C` stacked planes and the lookup
offsets the cell index by `cls * gridSize²`. Reasons:

1. It is literally what was asked — "feed in not just coord, but coordinate and
   family" — the grid is indexed by (family, x, y).
2. It is **local**: the generator can make two families conflict *in one region*
   without dragging the whole field, which is the mechanism the idea depends on
   ("optimize the two families that conflict the most").
3. Blast radius is small: the cell index arithmetic is already linear
   (`(iy*gs+ix)*F`), so forward and the gather-side backward each take one extra
   term. Layer widths, head shapes, scratch strides and every existing generated
   shader are untouched.
4. It does not loosen a single existing invariant. `onehot` + hashgrid still
   throws, and `tools/train_wta_hashgrid_test.ts` §1 (which asserts exactly
   that) stays true.

Canonical type added in `advect_wgsl.ts` — computed ONCE in `layoutField`, and
every consumer dispatches on it:

```ts
export type FamilyRoute =
  | { tag: "none" }
  | { tag: "onehot"; count: number }      // raw encoding, head-1 layer-0 channels
  | { tag: "grid-plane"; count: number }  // hashgrid, per-family feature plane
```

`FieldLayout.classes` keeps its meaning ("how many families exist") because the
`cls = pcg(i ^ CLASS_SALT) % C` derivation in advect / trainer / renderer /
adversary is shared and storage-free; `family` says **where the label enters**.

### D2 — POINT observer (m = 1)

The discriminator context is the coordinate itself (`tupleDims("point")` =
m 1, du 2, dy 2). That is the exact realization of "D predicts F from the
coordinate and does not know the family".

It also makes the per-family instrument **exact**: with m = 1 a tuple has one
member, so it has one unambiguous family. For m > 1 a tuple can mix families and
there is no honest per-family attribution, so `perFamily` reports the typed
state `unmeasured` there rather than inventing a bucketing rule.

### D3 — K = 2 predictor heads against C = 3 families

With K >= C the relaxed-WTA predictor can park one head on each family and the
family game buys the generator nothing beyond an ordinary single-family game.
The interesting regime is **K < C**: D can only cover K of the C modes, so one
family is always the mispredicted one and the generator is paid to keep it that
way. That is precisely the "do they balance / collapse / shift" question — the
per-family chart shows whether the odd-one-out role rotates.

K stays live (`?advK`, dock slider 1..12), so sweeping K is a one-click
experiment.

### D4 — soft-angle loss + anti-collapse pressure

Same as every other adversarial piece: direction-only payoff (raw-vector on this
observer collapses into amplitude cheats), and `GALLERY_ANTI_COLLAPSE` because
the soft-angle north pole pays 3.15x more for a DEAD field
(`agent_notes/2026-08-17_120215_KST_collapse_fix.md`).

## Per-family telemetry

`advFwd` already reduces scalar stats over the batch. Add `2C` reduction slots
(sum of payoff per family, active-tuple count per family) after the existing
finalized prefix (`7 + k`, plus 4 pressure moments when the pressure is
compiled). `ADV_STATS_BASE = 32` is the partials base, and the finalized prefix
must stay under it — with k <= 12 and C <= 3 the worst case is 7+12+4+6 = 29 < 32,
which `advStatsLayout` asserts (it already throws on overlap).

`AdvStats.perFamily` is a sum type mirroring `DirectionOrder`:
`{tag:"unmeasured"} | {tag:"measured"; mean: number[]; count: number[]}`.

## Files to touch

1. `src/render/webgpu/advect_wgsl.ts` — `Encoding.hashgrid.planes`, `FamilyRoute`,
   `layoutField` κ, `encodingParamFloats`, looped head grid lookup + `cls`.
2. `src/render/webgpu/train_wgsl.ts` — `encodeSite(cls)`, grid pass-B block cell
   offset.
3. `src/render/webgpu/adversary_wgsl.ts` — accept `grid-plane` fields, per-member
   `cls` in scratch, `encodeAt(cls)`, grid backward cell offset, per-family stats.
4. `src/render/webgpu/adversary_train.ts` — `perFamily` readback.
5. `src/render/webgpu/{splat,points}.ts` — exact-RGB palette keyed on CLASS_SALT.
6. `src/core/field/arch.ts` — `familyHashgrid` preset.
7. `src/main.ts` — the piece, telemetry plumbing.
8. `src/index.tsx` — per-family HUD row (3 coloured sparklines + numbers).
9. `tools/family_grid_test.ts` — gates (below).

## Gates (falsifiable, not mirrors)

- **FD vs analytic grid gradient** on a `grid-plane` layout: perturb one grid
  float, remeasure the payoff, compare to `extGrads` — this is the only check
  that catches a wrong plane offset in the backward (a wrong offset still
  *runs*, it just trains the wrong family's plane).
- **Plane isolation**: with only family c present in the batch, every grid float
  outside plane c has exactly zero gradient. A missing `cls*gs²` term in the
  backward fails this loudly; an aggregate cosine would not.
- **Byte-identity**: `planes = 1` regenerates the pre-change shader text
  verbatim, so no existing piece changes numerically.
- **Existing** `tools/train_wta_hashgrid_test.ts` §1 must still pass unchanged
  (hashgrid + `onehot` classes still throws).

## Results (measured 2026-08-19, Apple Metal via bun-webgpu)

### Gates — `bun tools/family_grid_test.ts`, ALL PASS

- §1 κ: `familyRoute` accepts hashgrid+classes+matching planes → `grid-plane`,
  raw+classes → `onehot`; throws by name on hashgrid-without-planes,
  planes-without-classes, and fourier+classes. Head 1's input width is NOT
  widened on the planed route (the grid carries the label).
- §2 byte-identity: `planes: 1` regenerates the advect kernel, both train
  passes and both adversary passes CHARACTER FOR CHARACTER vs an encoding with
  no `planes` key. The shipped hashgrid pieces did not move.
- §3 FD vs analytic on the grid slice: analytic/FD is ONE constant
  (**-64.0155**, worst deviation **0.66%**) across grid floats and MLP floats.
  That constant is genSeed's normalization at B=64, which the test deliberately
  never re-derives — sharing it is the verdict.
- §4 plane isolation: **0** grid cells receive gradient outside the
  family-qualified support, and all three planes receive gradient (35/41/34
  cells). Dropping the `cls·gs²` term puts everything in plane 0 and fails both
  halves.
- §5 end to end (the shipped arch through adversary extGrads → the field
  trainer's Adam, 60 fused steps): every weight finite, **every plane trained**
  (max|Δ| 7.4e-2 / 7.2e-2 / 7.0e-2), the per-family instrument live, and
  `Σ familyMean·count / B ≡ surprise` to 1e-4 — the chart and the headline
  number are one quantity, bucketed.

- §6 the ADVECT hot path — added after the first pass, which had a real hole:
  the trainers compile their own pass A/B, so **nothing had ever compiled the
  kernel that moves the particles** with a planed grid. A WGSL error there takes
  the page down at load and no other gate would have seen it. Now compiled via
  a real pipeline (bun-webgpu has no `getCompilationInfo`, so pipeline creation
  IS the check), plus the behavioural claim: **one coordinate, three families,
  three different forces** (min pairwise |ΔF| 5.7e-2) and the field's
  `trainableWeights` pair 1:1 with the packed segments, plane count included.

Existing suites re-run and green with these changes:
`train_wta_hashgrid_test`, `train_wta_pressure_test`, `train_wta_test`,
`kernel_test`, `ad_wta_test`. `yarn build` (parcel, --no-scope-hoist) succeeds.

**Flake note.** `tools/train_wta_test.ts` failed twice (the
`force + angle-relative-scale` block, cos 0.0) when run back-to-back with other
GPU suites in one shell loop. Attribution run: 5/5 clean on this branch when run
ALONE, 3/3 clean at baseline. Consistent with the suite header's "GPU suites are
SEQUENTIAL — run nothing else on the GPU", not with a regression here. Run them
one at a time.

### Cost

At B=512, K=2, hidden [64,64,64] (`adv.step` only, Apple Metal):

| grid | totalFloats | ms/step |
|---|---|---|
| 32²×8, **3 planes** (SHIPPED) | 34,312 | **3.39** |
| 32²×4, 1 plane | 13,320 | 3.02 |
| 24²×6, 3 planes | 19,848 | 3.17 |

**The family planes are nearly free — the MLP depth is the cost.** Tripling the
grid adds ~0.4 ms while the same 3-layer 64-wide MLP on a single-plane grid
already costs 3.02. If this piece needs to be cheaper, shrink `hiddenUnits`
before shrinking the grid.

Note also that `layout.totalFloats * 4` (137 KB) exceeds
`maxComputeWorkgroupStorageSize`, so the advect kernel automatically takes the
unstaged weight path (`advect.ts` decides this from the device limit — nothing
to configure).

## Open / unresolved

- **Not yet run in a browser.** The headless Chromium on this box reports
  `adapter: null` (no SwiftShader-WebGPU), so `tools/smoke.mjs` can only confirm
  the app shows its WebGPU notice — the known caveat in AGENTS.md. Everything
  above was verified on real Metal through `bun-webgpu` against the same code
  paths, but nobody has WATCHED the piece yet. What §6 does NOT cover, because
  it needs tfjs's webgpu device (bun has none) and a real canvas:
    - `AdvectKernel.fromField` itself (§6 builds the same layout and compiles
      the same WGSL, but the constructor's device lookup and weight upload are
      only exercised in the browser);
    - the RENDER path — the `rgb-families` palette branch is patched into both
      splat variants and points.ts but has never executed, so "are the three
      families actually red/green/blue on screen" is unverified;
    - the HUD row appearing and moving;
    - frame budget at 90k particles with a ~3.4 ms/step trainer.

- **A latent landmine for the NEXT family piece.** `applyArchDockPreset`
  (arch.ts) deliberately preserves `classes` across a preset swap. On a piece
  with `archEditable: true`, swapping a family-planed hashgrid to a raw preset
  would carry `classes: 3` onto the raw encoding → the `onehot` route → the
  fused adversary refuses it and the game silently turns OFF (console warn
  only). The shipped piece is NOT `archEditable`, so this is unreachable today;
  anyone making a family piece editable must make the preset swap carry the
  family route, not just the count.
- **K=2 is a guess, not a measurement.** The reasoning (K < C is the only regime
  where family conditioning changes the game) is sound but untested against the
  actual dynamics. Sweep `?advK=1..4` with the per-family row open; that sweep
  IS the experiment.
- tfjs reference: `HelmholtzField.forces` still throws for `classes > 0`, so
  `?train=tfjs` is unavailable on this piece and the gate is FD against the
  fused kernel rather than tfjs parity. Extending the tfjs field with a family
  plane would buy a full parity sweep if one is ever wanted.
- The per-family instrument is point-observer only by construction
  (`familyInstrument`). A pair/tri family piece would render and train fine but
  report `unmeasured`; giving it a chart needs a defensible rule for a tuple
  whose members disagree about their family.
