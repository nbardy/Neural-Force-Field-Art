# Pixel pieces: why the config surface was minimal, and what is now shared

Date: 2026-08-25. Branch `multi-guess-modularization`.

Goal (user): the Pixel GAN pieces expose almost no configuration. Make the
generator ARCHITECTURE configurable by reusing the existing arch machinery, make
the discriminator/generator learning rates configurable, and in general share
the configurable surface between the two game families — accepting that the
FUSED kernels cannot be shared.

## Verified findings — the "minimal config" was four gates, not a fusion limit

1. **The dock's whole game section is gated on a RELATIONAL predicate.**
   `src/index.tsx` — `const adversary = runtime.adversaryKind !== "off"`, then
   `{adversary && <ControlSection title="adversary">…}`. A Pixel piece declares
   `pixelDisc` and no `adversary`, so it rendered zero game controls.

2. **Both learning rates already reached the pixel critic.** `main.ts` `tick`
   passes `lr: discriminatorLearningRate` to BOTH `AdversaryTrainer.encodeStep`
   and `PixelDiscTrainer.encodeStep`, and `generatorLearningRate` to the field
   trainer for every piece. `resolveLiveGameControls` runs unconditionally, so
   `?gLR=` / `?dLR=` steered a Pixel piece before this change — only the sliders
   were hidden. This was a UI gate, not missing plumbing.

3. **The model/arch dock is gated on `piece.fieldArch`, and every GAME piece
   used `createField:` instead** (pixel and relational alike). The four pixel
   pieces' `createField` was literally
   `() => createFieldFromArch({ ...ARCH.dualFourier, alpha: 0.55 })` — not a
   load-bearing recipe. `startLoop` already resolves `overrides.fieldArch` >
   `cfg.fieldArch` > `createField`, so the swap path existed and was unreachable.

4. **The multi-guess head shipped unreachable.** `eb211d1` added K guesses with
   relaxed WTA to the pixel critics — CPU oracle, WGSL, per-guess win counters,
   equivalence tests. But `PixelCriticSpec` had no `guesses` field and the
   `new PixelDiscTrainer` site did not forward one, so no piece could ask for
   K > 1. `AdversaryKind` IS `GuessKind` (`src/core/gan/adversary.ts:122`) —
   the two games already share the WTA spec (`src/core/gan/wta.ts`, one spec for
   four backends, `19bf275`); only the two CONFIG surfaces had diverged.

## What was changed

- `PixelCriticSpec.guesses?: GuessKind` (`src/main.ts`), forwarded UNRESOLVED to
  `PixelDiscTrainer` so `resolvePixelDims`/`validatePixelDims` stay the only κ.
- The four Pixel pieces: `createField` → `fieldArch: {...ARCH.dualFourier,
  alpha: 0.55}` + `archEditable: true` + `archDock: "dual"`. Identical field at
  startup; the model dock now works and can swap among ARCH_DOCK_DUAL.
- Dock: G lr / D lr / D-G ratio moved OUT of the adversary section into one
  `training` section gated on `hasGame = adversary || pixelDisc !== undefined`.
  One copy, both games.
- Live pixel reward: `pixelGenWeight` (was `cfg.pixelDisc.weight` read at the
  hot site), `LoopHandle.get/setPixelCriticWeight`, `?pixW=`,
  `PIXEL_CRITIC_WEIGHT_RANGE = {min: 0, max: 0.5}`, and a `pixel critic` dock
  section (summary + reward + a note when reward is 0).
- `tools/adversary_wire_test.ts` §8d: every arch the pixel dock can now select
  must classify OK **and** emit a shader. 4 pieces x 4 presets = 16 combos,
  0 refused.

## Deliberately NOT shared

- **One reward slider for both games.** `advRt.weight` multiplies a relaxed-WTA
  payoff in predictor-output units (dock range 0..20); `pixelGenWeight`
  multiplies a soft-density residual (shipped 0.03-0.04). Same name, different
  units — a shared range would silently mistune one of them.
- **The critic capacity knobs.** Relational has `encoding`(tuple)/`target`/
  `loss`; pixel has `kind`/`G`/`E`/`K`/`hidden`/`dt`. These describe what the
  critic OBSERVES and are genuinely different.
- **The fused WGSL emitters.** Both bake compile-time literals.

## Not done — the next seam (proposal, not implemented)

`ArtPieceConfig` still carries two independent optional fields (`adversary`,
`pixelDisc`) and the code asks "which one?" at several sites. The canonical
form is one sum — `GameSpec = off | relational | pixel` — with one κ and one
capability δ replacing the pair `classifyPixelDiscFusion` (returns data) /
`validateAdversaryFusion` (throws). `docs/PLAN_PIXEL_GENERATOR_ARCH.md` already
observes that `classifyPixelDiscFusion` "is the closest thing in the repo to a
capability table — there is no other. Support is otherwise ad-hoc across ~30
call sites gating on `spec.kind`, `encoding.kind`, `classes`, or `family.tag`."
That is the refactor with real leverage; it touches 13 gallery pieces, the
persisted-dock schema and the URL parser, so it is a separate change.

Also pending: `guesses` is a piece-declaration knob only — no dock control yet
(the relational `guesses K` / `relax ε` rows go through
`StartLoopOptions.overrides.k/relaxEps`, which the pixel trainer does not read).

## Verification

- `bun tools/adversary_wire_test.ts` — ALL PASS (incl. new §8d).
- `bun tools/pixel_disc_test.ts` — ALL PASS (CPU + GPU smoke, 4 kinds).
- `bun tools/pixel_disc_equiv_test.ts` — ALL PASS (CPU/GPU oracle, incl. the
  family-planed hashgrid arms).
- `tsc --noEmit` on `src/index.tsx`: 15 errors, all pre-existing WebGPU/TypedArray
  lib skew in `render/webgpu/*` + `core/gan/adversary.ts`; **zero** in `main.ts`
  or `index.tsx`.
- `npx parcel src/index.html` builds clean; app mounted in a real Chrome with
  `navigator.gpu` present. On "Pixel · VecField" the dock renders
  `model-arch-controls` + `model-arch-presets` ("fourier · [32x32] · 2 heads"),
  `game-training-controls` (G lr 1.5e-3 / D lr 3.0e-3 / D-G 2.00x),
  `pixel-critic-controls` ("vec-field G16 E8 K16 h32 · 1 guess", reward 0.040),
  and NO `adversary-controls`. Selecting the "Dual HashGrid" preset restarted
  the piece with `hashgrid · [32x32] · 2 heads` and `window.__nffHealth.pixel`
  present — i.e. the critic was CONSTRUCTED on the swapped arch (a refused arch
  throws at startup).

**Not verified:** live rendering/training dynamics. The embedded browser pane
reports `window.innerWidth === 0`, so the swapchain texture is size 0, the render
loop halted at frame 6 and `pixel.{dLoss,gLoss,extGradNorm}` stayed 0. Headless
`tools/smoke.mjs` on this box reports `adapter: null` (no SwiftShader-WebGPU) and
the app correctly shows the WebGPU warning — the caveat AGENTS.md already
records. Someone should open a Pixel piece in a real browser and confirm
`extGradNorm > 0` on each of the four dock archs before this is called done.


## Concurrency

A second session worked in this same worktree on the same day and made the
matching change for the RELATIONAL side — "Adversary · Pair WTA K=4"
`createField` → `fieldArch` + `archEditable` + `archDock: "dual"`, plus widening
a stale `createField || createModel` assertion in
`tools/adversary_wire_test.ts`. See
`agent_notes/2026-08-25_122500_KST_pair_wta_arch_dock.md`. Both sets of edits are
in `git diff` together; neither clobbered the other, and the suite passes on the
merged tree. Diff before assuming authorship.

Their §5 leaves one item this note agrees with: the **predictor** widths
(`hiddenUnits`/`featureDim`, `adversary_train.ts`) are optional constructor opts
that no caller ever passes, and the fused codegen is general over `LayerDims[]`.
That is the relational counterpart of the pixel critic's `G`/`E`/`K`/`hidden` —
a real unexposed capacity knob on both games.

Their change also leaves the relational side without §8d's equivalent: the four
now-selectable dual presets on "Adversary · Pair WTA K=4" were verified by a
one-off codegen probe, not by a gate in the suite.
