# Adversary dock: encoding for the other 6 pieces, and the predictor's width

2026-08-26 ~01:55 KST. Follow-on to
`2026-08-25_122500_KST_pair_wta_arch_dock.md`, which made ONE piece
("Adversary · Pair WTA K=4") arch-selectable. This note covers finishing that
sweep, the gate that makes the dock's safety non-accidental, and exposing the
predictor — the one model in this project that had never been varied.

## Goal

Answer, in the artwork itself, "what model is this piece?" — for BOTH models a
relational game trains.

## What changed

### 1. Six more pieces off `createField` (`src/main.ts`)

| Piece | arch |
|---|---|
| Adversary · Single (control) | dualStd α 0.7 |
| Adversary · WTA K=8 | dualStd α 0.62 |
| Adversary · Tri WTA K=6 | dualStd α 0.55 |
| Adversary · Quad WTA K=6 | dualStd α 0.55 |
| Adversary · Chaos Weave | dualFourier α 0.45 |
| Adversary · Pair · HashGrid · Curl | dualHashgrid α 0.55 |

Each was `createField: () => createFieldFromArch(<plain literal>)` — the hatch
for load-bearing arches used where nothing was baked. The last one is
`DEFAULT_PIECE_NAME`, so the piece everyone lands on had no model section at
all.

The rationale now lives in ONE place, `DUAL_ARCH_DOCK` in `main.ts`, which the
13 dual-dock pieces point at (7 relational adversary, 4 pixel critic, 2 plain
neural field). Previously it was a 9-line comment on one piece.

**Two pieces deliberately keep `createField`**, and this is a decision, not an
omission:

- **Agree + Disagree RGB** — tunes `fourierOctaves: 3` against the preset
  default 4, and `applyArchDockPreset`'s preserve list is `alpha` / `semantic`
  / `classes` only. A dock swap would silently reset the octaves.
- **RGB Families · HashGrid** — `familyHashgrid` → the `grid-plane`
  `FamilyRoute`; the dual dock has no preset carrying `classes`/`planes`.

I considered adding `fourierOctaves` to the preserve list and rejected it: α is
an OUTPUT blend and `semantic`/`classes` are GAME facts, meaningful under any
encoding, while octaves is part of the fourier encoding itself. Carrying it
across would make "Dual Fourier" mean different things depending on which piece
you came from. The omission is now documented AT the preserve list
(`src/core/field/arch.ts`) instead of being a trap.

### 2. §8e — the dock/capability coincidence made load-bearing

`ARCH_DOCK_DUAL` happens to equal the set `validateAdversaryFusion` accepts.
Nothing enforced that. `tools/adversary_wire_test.ts` §8e now proves it by
CODEGEN — emitting BOTH fused passes for every (piece × arch preset × predictor
width) the dock can produce.

    ok   every (arch x predictor width) the adversary dock can select emits
         both fused passes (7 pieces = 140 combos, 0 refused)      [~12 s]

Emitting the shaders, not just building the layout, is the point: the refusals
that bite (activation with no backward checkpoint, family route with no dEnc
counterpart) live in the emitters, past `layoutAdversary`.

**Falsified, not assumed.** Injecting `ARCH.siren` (single-head) into
`ARCH_DOCK_DUAL` makes §8e fail with
`fused adversary needs a two-head neural field (got vector)`. Verified, then
reverted.

### 3. A real hole found in §8d (the pixel gate) by that same injection

§8d **stayed green** under the injected bad preset. It hardcoded a two-head
`layoutField` regardless of `arch.heads`, so a single-head preset added to the
dual dock produced a helmholtz layout in the test and a `vector` layout in
production — which `classifyPixelDiscFusion` refuses with the same message.
The gate could not have caught the thing it exists to catch.

Both §8d and §8e now honour `arch.heads`. Under injection both fail (4 and 7
refusals); on the real tree both pass. This is the most valuable single finding
of the session — an existing gate that was structurally unfalsifiable.

### 4. The predictor is now a declared, selectable model

`AdversaryTrainer` and the tfjs `Adversary` have both accepted
`hiddenUnits`/`featureDim` since the port and **no caller ever passed either**.
Every predictor in this project has always been `du → 32 selu → 16 selu → dy`.

> Consequence worth carrying: every adversary reading in `agent_notes/` — win
> EMAs, payoff curves, R₁/R₂, the pole-exploit refutation, the NaN probes — is
> a property of a **32/16** adversary, not of "the adversary".

Added (`src/main.ts`):

- `PredictorArch { hiddenUnits, featureDim }`, `PREDICTOR_ARCH_DEFAULT` (32/16),
  `PREDICTOR_ARCH_DOCK` (16/8, 32/16, 64/32, 128/64).
- `AdversarySpec.predictor?` + `predictorArchOf(spec)` — resolved ONCE. Both
  trainers previously carried their own `?? 32` / `?? 16`, i.e. two places for
  the fused path and its own tfjs oracle to drift on a number neither declared.
- `?advHidden=` / `?advFeature=`, integers in [1, 256], validated with a throw
  rather than a silent clamp.
- `startLoop` override + a dock `Segmented` in the **model** section, so the
  dock now names both nets. The `[adversary] FUSED …` log line reports
  `predictor=H/F`.

Default path is byte-identical: no URL knob and no dock choice resolves to
32/16, the value every piece already ran.

The only predictor restriction in the fused codegen is activation — SELU only,
because `emitBwdStore` needs pre-activation checkpoints that `sin` does not
keep. **That refusal walks the PREDICTOR, not the field**, which is why
`dualSiren` (SIREN *generator*) is accepted. Easy misreading of
`validateAdversaryFusion`; noted at the type.

## Verified

- `npx tsc --noEmit … src/main.ts src/index.tsx` — clean (repo has no
  `tsconfig.json`; pre-existing errors elsewhere are untouched).
- `bun tools/adversary_wire_test.ts` — ALL PASS, §8d 16 combos / §8e 140 combos,
  0 refused.
- Both new gates proven to FAIL on an injected bad preset.
- 9 of the 10 pure-CPU suites pass.

## NOT verified / open

- **`bun tools/adversary_strict_test.ts` FAILS — and it is PRE-EXISTING.**
  Bisected to 7a8d12d, 7f4dba0 and eb211d1: failing at all three, i.e. before
  any of this work. Symptoms: `adjusted active target value is exactly
  invariant under positive signal scaling`, `adjusted active target is an exact
  unit direction`, `adjusted public predictions are direction-normalized
  (0.301868, 0.396196)`. Someone should own this; it is the strict invariant
  suite for the exact observer the Pair pieces use.
- **No live browser check of the dock.** Deferred deliberately, see below.
- The 21 GPU suites were not run.
- The two `createField` holdouts still show no model section at all — the
  `piece.fieldArch` gate in `index.tsx:1898` hides the read-only summary too.
  Fixing that needs a read-only arch description for `createField` pieces.

## Concurrency hazard — read this before trusting the tree

Another session was writing this worktree throughout (`tools/health_audit.mjs`,
`tools/health_report.ts`, `tools/health_sweep.ts`, and the r2-gate note), with
writes as recent as 01:53:36 while this work was in progress.

**I ran `git stash` + `git checkout <old-sha> -- src tools` + `git stash pop`
at ~01:51 to bisect the strict-test failure.** That was a mistake on a shared
worktree: for a few seconds `src/` and `tools/` held old-commit content, and a
concurrent write landing in that window would have been either lost or
restored on top of stale content. The pop applied cleanly and their files look
intact, but this is not something the git history can confirm — flagging it
explicitly rather than assuming.

Commits here are therefore scoped to `src/main.ts`, `src/index.tsx`,
`src/core/field/arch.ts`, `tools/adversary_wire_test.ts` and this note. The
other session's files are left uncommitted for them to own.
