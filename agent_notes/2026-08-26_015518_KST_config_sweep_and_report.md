# Config sweeps: `health_sweep.ts` (driver) + `health_report.ts` (reducer)

Goal: answer "can the audit generate reports of long rollouts across many
configs / objectives / architectures?" — then close the gap.

## What the audit could and could not do (verified, not assumed)

Could: long rollouts (`durationSec` arbitrary) with the **full ~1 Hz sample
stream** written to `<key>.json` — `stripSeries` trims the aggregate, never
`samples`. Could also carry ONE config, because `base` is a full URL and
`defaultsForPiece` deliberately re-resolves through `location.search`
(src/index.tsx) so URL knobs survive the gallery click.

Could not: sweep. One `base` per process, no matrix. No cross-run comparison —
each invocation wrote a fresh `output/health-audit/<iso-ts>/` and nothing read
across them. `aggregate` collapses to medians plus one `acTrend`.

Architecture was not URL-addressable at all: `archPreset` was a dock control and
piece-switch hard-reset it to `null`.

## What changed

**`?arch=<preset>`** (src/index.tsx) — a first-class URL knob, GLOBAL like
`?advM`/`?advK` (it re-resolves in `runtimeForPieceSwitch` via `nextDefaults`
rather than being nulled), throwing `URIError` on an unknown value. A DOCK-set
preset keeps the old don't-follow-me behaviour; the two are different things.

**Measured `ArchHealth` in the snapshot** (src/health.ts + main.ts) —
`{kind, weightFloats, macsPerParticle, encoding, classes}` read off
`advect.layout`, i.e. what actually COMPILED. This exists because `?arch=` is
honoured only on `archEditable` pieces: a sweep that labelled cells by the
requested URL would run one network N times and report "architecture makes no
difference", after burning the GPU hours to do it.

**`tools/health_sweep.ts`** — cartesian product of axes → one `runPiece` per
cell. Reuses `runPiece`/`aggregate`/`classify` from the audit verbatim (which
required exporting them; the audit's own `INVOKED_DIRECTLY` guard, added
independently by a concurrent session on 2026-08-25, is what makes the import
safe). Resumable: run dir keyed by `spec.name`, a cell whose result *parses* is
skipped, a truncated file is re-run. `collisions()` reports any arch axis whose
values produced one fingerprint, per piece.

**`tools/health_report.ts`** — reads a run dir, emits one self-contained
theme-aware HTML page: collision banner first, ranked table, per-metric overlay
charts (R1 fixed 0–1 with the laminar gate, R2, AC with the dead gate, payoff),
then per-cell R1-vs-AC cards. Nonfinite samples BREAK the polyline rather than
bridging it.

## Measured

`bun tools/health_sweep.ts --self-test` and `bun tools/health_report.ts
--self-test` — both PASS (spec rejection, product determinism/stability under
axis reordering, per-piece collision detection, sentinel decoding, trace
breaking, scale-invariant trend, HTML escaping).

Live 4-cell validation, `pair4 x {dualStd,dualFourier} x advPolar{0,0.05}`,
40 s each, real Metal:

| arch | advPolar | AC | AC trend/s | R1 | R2 | fingerprint |
|---|---|---|---|---|---|---|
| dualFourier | 0.05 | **0.530** | +8.1e-4 | 0.229 | 0.119 | fourier/w3464 |
| dualFourier | 0    | 0.131 | +0.038 | 0.227 | 0.058 | fourier/w3464 |
| dualStd     | 0    | 0.001 | **−0.125** | 0.055 | 0.003 | raw/w2440 |
| dualStd     | 0.05 | 0.001 | **−0.096** | 0.023 | 8.9e-4 | raw/w2440 |

Distinct fingerprints ⇒ `?arch=` verified end-to-end, no collision.

## Findings and unresolved

1. **dualStd is dying on this piece**: AC ~1e-3 and falling ~10%/s over the
   rollout, roughly one order above the `acDead` gate (1e-4). Its LOW R1 (0.02–0.06)
   reads healthy and is not — a near-empty field has isotropic directions
   trivially. This is the exact pairing the report ranks by AC *with* R1
   alongside. 40 s is short; needs a real soak before it is a claim.
2. **The anti-collapse pressure raised AC 4x on dualFourier** (0.131 → 0.530)
   and did nothing for dualStd. Suggestive, one seed, not a result.
3. **AC would read better on a log axis.** Two cells at ~1e-3 against two at
   ~0.5 render as a flat floor. Honest, but not legible.
4. **A stale Parcel bundle produced three bogus `page-error` verdicts** on the
   first validation run (`predictorArchOf is not defined`, while the symbol is
   exported at main.ts:795 and imported fine). `--no-cache` fixed it. Worth
   knowing: a sweep is only as trustworthy as the bundle it drove, and the
   audit's page-error arm correctly refused to score those cells.

## Concurrency hazard hit during this work

Another session was editing the same tree throughout, and its commit `7a8d12d`
swept in this session's uncommitted `?arch=` changes to `src/index.tsx` and
`src/health.ts`. Nothing was lost, but the attribution is wrong and the two
sessions were one `git checkout` away from destroying each other's work.
`git status` before starting, not after.
