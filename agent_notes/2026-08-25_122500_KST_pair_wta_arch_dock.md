# Pair WTA K=4: which model, and why the arch was not selectable

2026-08-25 12:25 KST. Question asked: *"for adversary pair WTA K=4 what model is
that? why can't we select the model architecture there? should we be able to?
can we swap that in?"*

## 1. There are TWO models in that piece — the question is ambiguous by nature

| role | what it is | where |
|---|---|---|
| generator (the field being trained) | `ARCH.dualStd` @ α 0.55 — raw (x,y) → SELU [32,32] → **2 heads**, blended `(1−α)·A + α·B`, tanh out | `src/core/field/arch.ts:118` |
| discriminator (the K=4 predictor) | **4 heads**, each `u(du=1) → 32 selu → 16 selu → dy(=2) linear`, Adam, relaxed-WTA ε 0.05 | `adversary_train.ts:384-390`, tfjs mirror `adversary.ts:1429-1431` |

The predictor widths (32/16) are `hiddenUnits`/`featureDim`, plumbed as
**optional constructor opts that no caller ever passes** — every piece and every
tool runs the 32/16 default. The fused codegen is general over `LayerDims[]`
(arbitrary depth/width; only `sin` is refused, `adversary_wgsl.ts:601-608`), so
that one is an unexposed knob, not a hard-coded net.

## 2. Why the generator arch was not selectable — MEASURED, not a design refusal

Two independent gates, both incidental:

- The dock's whole `model` section is gated on `piece.fieldArch`
  (`index.tsx:1898`); the piece declared `createField: () =>
  createFieldFromArch({...ARCH.dualStd, alpha: 0.55})` instead, so the section
  did not render **at all** — not even the read-only arch summary line.
- The preset selector additionally needs `archEditable`
  (`index.tsx:1904`), which the piece did not set.

`createField` is documented as the escape hatch "for load-bearing game recipes
that bake semantics the dock must not overwrite" (`main.ts:4100`). Here it was
baking **nothing** — it called `createFieldFromArch` on a plain literal. What is
load-bearing on this piece is the OBSERVER (`pair-rotation-scale-adjusted`) and
the soft-angle objective, not the position encoding.

## 3. Should it be selectable — yes, and the supported set is exactly the dock list

`ARCH_DOCK_DUAL` = dualStd / dualFourier / dualSiren / dualHashgrid. The fused
relational adversary accepts precisely the two-head non-one-hot arches
(`validateAdversaryFusion`, matrix in `docs/PLAN_PIXEL_GENERATOR_ARCH.md` §1).
The dock list and the capability set coincide, so no dock choice can reach a
refusal. `applyArchDockPreset` preserves α / semantic / classes, so α stays 0.55
across a swap — the same discipline `DEFAULT_PIECE_NAME`'s comment states for
the hand-curated hashgrid variant of this game.

**VERIFIED (not doc-trust):** codegen probe built `adversaryPassAShader` +
`adversaryPassBShader` for this piece's exact game (pair-rotation-scale-adjusted,
soft-angle τ 0.05, K=4, ε 0.05, GALLERY_ANTI_COLLAPSE, periodic) against all four
dual presets — all four compile:

```
ok dualStd      passA 37320 B, passB 11814 B
ok dualFourier  passA 40207 B, passB 11824 B
ok dualSiren    passA 37677 B, passB 11850 B
ok dualHashgrid passA 40794 B, passB 12824 B
```

`dualHashgrid` on this game is also proven in production: "Adversary · Pair ·
HashGrid · Curl" is the same observer/K on a hashgrid field.

## 4. Change made

`src/main.ts`, the Pair WTA K=4 piece: `createField` → `fieldArch: {...ARCH.dualStd,
alpha: 0.55}` + `archEditable: true` + `archDock: "dual"`. Default behaviour is
byte-identical (startLoop feeds the same object to the same constructor).

`tools/adversary_wire_test.ts:612` asserted every adversary piece has
`createField || createModel` — **stale**, `fieldArch` trains the same field and
startLoop resolves it FIRST. Widened. (`:967` in the same file already counted
`fieldArch`, so the two assertions disagreed with each other.)

Gates run: tsc clean on `main.ts`/`index.tsx`; `bun tools/adversary_wire_test.ts`
ALL PASS. **NOT run:** a live browser check that the dock section renders for
this piece (the piece is not `DEFAULT_PIECE_NAME` and there is no `?piece=`
param, so `tools/smoke.mjs` cannot land on it without a `?dock=` blob). The React
path is identical to the six pieces that already set `archEditable`.

## 5. Open, not done

- Predictor width/depth (`hiddenUnits`/`featureDim`) is a real unexposed knob —
  fused codegen supports it, nothing passes it, no dock control exists.
- **Concurrency:** while this was written another session was editing
  `src/main.ts` in this same worktree (pixel pieces → `fieldArch`,
  `PixelCriticSpec.guesses`). `git diff` here contains BOTH sets of work. Diff
  before assuming authorship.
