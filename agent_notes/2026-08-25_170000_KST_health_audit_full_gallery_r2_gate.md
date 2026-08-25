# 2026-08-25 — full-gallery health audit re-run, and the R2 gate that was missing

Continues `agent_notes/2026-08-18_health_metrics_audit.md`. Goal: re-run the
headless gallery audit against the CURRENT tree (branch
`multi-guess-modularization`, with another session's uncommitted
`src/main.ts` / `src/index.tsx` / `tools/adversary_wire_test.ts` edits in it)
and compare to the 2026-08-18 baseline.

## How it was run (reproducible)

```
./node_modules/.bin/parcel build --no-scope-hoist --no-cache --public-url ./ src/index.html
cd dist && python3 -m http.server 8811 &
node tools/health_audit.mjs all http://localhost:8811/index.html 90 2
```

Artifacts: `output/health-audit/2026-08-25T16-08-34-194Z/` (gitignored).
`yarn` is not on PATH on this box — use `./node_modules/.bin/parcel` directly.

## Result: 3 unhealthy as run, 5 after the R2 gate below

| piece | 2026-08-18 | 2026-08-25 | grid R1 | grid R2 | sat | ac | fps |
|---|---|---|---|---|---|---|---|
| Single (control) | healthy sat .081 | **FROZEN** | 0.057 | 0.130 | **0.473** | 1.265 | 59.9 |
| WTA K=8 | healthy | healthy | 0.171 | 0.171 | 0 | 0.685 | 60.0 |
| Pair WTA K=4 | healthy | healthy | 0.050 | 0.198 | 0 | 0.062 | 59.9 |
| Tri WTA K=6 | FROZEN .673 | **NEMATIC** (was PASS) | 0.337 | **0.613** | 0.213 | 0.972 | 59.9 |
| Quad WTA K=6 | FROZEN .618 | **FROZEN** | 0.046 | 0.188 | **0.756** | 1.350 | 59.9 |
| Agree+Disagree RGB | healthy | healthy | 0.142 | 0.075 | 0.244 | 1.007 | 59.9 |
| Chaos Weave | healthy | healthy | 0.096 | 0.117 | 0.075 | 0.905 | 60.0 |
| Pair·HashGrid·Curl | healthy ac 6.7e-4 | healthy | 0.105 | 0.092 | 0 | **0.419** | 41.2 |
| Pixel · VecField | FROZEN .512 | **LAMINAR** | **0.782** | 0.556 | 0.194 | 0.616 | 41.7 |
| Max Structure | healthy | **NEMATIC** (was PASS) | 0.034 | **0.927** | 0 | 0.647 | 50.0 |

## Finding 1 — the R2 escape route was documented but never gated (FIXED)

`CLAUDE.md` has said the whole time: *"Never read `r1` without `r2`. R₁ alone is
escapable: a ±F₀ counter-streaming field scores R₁ ≈ 0 and looks exactly as
laminar."* `classify()` gated `r1` and never `r2`, so a counter-streaming sheet
— half the domain flowing +x, half −x, mean direction ZERO — passed as healthy.

Measured this run: **Max Structure R1 0.034 / R2 0.927**. It is one axis end to
end and scored PASS. Tri WTA K=6 at R1 0.337 / R2 0.613 is on the same road.
Healthy pieces in the same run sit at R2 0.075–0.198, so the two regimes are
cleanly separated; the gate is `HEALTH_R2_MAX`, default **0.5**, same margin as
`r1Laminar`.

Added verdict `nematic-collapse`, ranked BELOW both laminar arms (polar order
implies nematic order, so when both fire R1 is the more specific reading).
Three self-test cases, including a regression guard on the exact
R1-low/R2-high stream that used to read healthy.

Re-scoring the recorded samples through the new gate: **5 unhealthy, not 3.**

## Finding 2 — VecField's collapse was previously UNMEASURABLE

The 2026-08-18 summary records `vecfield.r1 = NaN` and a `frozen-saturated`
verdict. That NaN was not a property of the piece: the grid R1 had not yet been
wired into the verdict, so a piece with no adversary had no direction reading at
all. It now reads **grid R1 0.782** — the most collapsed field in the gallery.

This overturns the working assumption from the 2026-08-18 session that "the
pixel GAN doesn't seem to have that problem so much". It had it worst; there was
no instrument pointed at it. Anything concluded about pixel-critic direction
health before the grid-R1 wiring is unsupported, not merely stale.

## Finding 3 — the frozen-saturation set MOVED

`single` went healthy (sat .081) → FROZEN (sat .473); `tri6` went FROZEN (.673)
→ sat .213; `quad6` got worse (.618 → .756). One run each, so run-to-run
variance is NOT excluded — this needs a repeat before anyone calls `single` a
regression from the branch. What is not variance: saturation is the dominant
failure mode of this gallery, appearing on 2–3 of 10 pieces in both runs.

## Finding 4 — hashgrid AC recovered 600x, and three pieces lost fps

`Pair·HashGrid·Curl` ac 6.7e-4 → **0.419**. The old value was one order above
the dead-field gate (1e-4) — it was nearly a dead field and passed. Good news,
cause not attributed here (candidates: the family-planed hashgrid work in
38471d6, or the pressure defaults).

fps fell to 41.2 (hashgrid), 41.7 (vecfield), 50.0 (struct) from 60.0 across the
board in the baseline. Above the `fpsFloor` 30 so nothing failed, but three
pieces regressing together is worth attributing before it reaches the floor.

## Also changed in `tools/health_audit.mjs`

- **Coverage 10 → 16 pieces.** `all` audited 10 of the 16 gallery pieces and
  called itself `all`; the RGB-families adversary and three of the four pixel
  critics had no headless coverage. Added `families`, `nextframe`, `realfake`,
  `inpaint`, `chaos`, `species`, plus a `pixel` group alias.
- **Entry-point guard.** `import(".../health_audit.mjs")` — the obvious way to
  reuse `classify`/`aggregate` to re-score a finished run — used to launch a
  real GPU audit as an import side effect. Hit for real while writing this note.

## Open / not done

1. **`rmsP` is still not in the health snapshot.** This is the one recommended
   follow-up from `2026-08-18_pole_exploit_refuted_predictor_scale_blindness.md`
   and it is the tripwire that says WHICH SIDE failed: D drifting past ~5τ while
   rmsY < τ fires BEFORE the collapse completes. `src/health.ts`,
   `adversary_train.ts` and `adversary_wgsl.ts` are clean in the tree now (only
   `main.ts`/`index.tsx`/`adversary_wire_test.ts` are another session's dirty
   work), but the CPU oracle in `core/gan/adversary.ts` must move with it or
   `adversary_wire_test.ts` — which IS dirty — will fail.
2. Repeat run to separate `single`'s flip from variance.
3. Attribute the three-piece fps drop.
4. Nobody has decided what to DO about Max Structure reading nematic. It
   optimizes W_STRUCT directly; a single-axis field may be what that loss wants.
   The gate is a health reading, not a verdict on the art — but the user's
   original complaint was "picks a direction and gets stuck", and a nematic
   field looks single-axis, so this is on the complaint's path.
