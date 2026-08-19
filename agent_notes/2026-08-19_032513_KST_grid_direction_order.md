# Direction convergence on EVERY piece: grid R₁/R₂ in the field probe

Goal: answer "did we add metrics for direction convergence (collapse to one
direction), and can the chart be on all pieces?" — then make the answer yes.

## What already existed (verified, not assumed)

`R₁` (polar) and `R₂` (nematic) direction order have existed since the
anti-collapse work: `src/core/losses/isotropy.ts::directionOrderParameters`
(tfjs) and the fused twin whose moments `adversary_wgsl.ts` pass-A already
reduces (`AdvStats.directionOrder`). The HUD charted R₁.

The gap was coverage, and it was structural, not cosmetic:

* the batch R₁/R₂ are only reduced when **anti-collapse pressure is compiled**
  (`?advPolar`/`?advNematic` > 0). Otherwise `directionOrder` is
  `{tag:"unmeasured"}` and `AdvHealth.r1` is `null`.
* the chart lived inside `telemetry.tag === "on"`, so **non-adversary pieces had
  no direction instrument at all** — and rendered a permanent `—` when an
  adversary ran without pressure.
* `tools/health_audit.mjs`'s `laminar-collapse` verdict read `agg.r1` from the
  adv block, so a pixel-critic piece was *structurally incapable* of reaching
  that verdict however unidirectional its field went.

## What changed

**`src/render/webgpu/field_probe.ts`** — `fieldMetricsFrom` now also returns
`r1`/`r2`, accumulated in the loop that already walks the centre-point block.
New `PROBE_TAU = 0.05`, the canonical soft-angle τ, so the grid statistic and
the adversary's batch statistic are literally the same statistic on two sample
populations. The probe runs on **every** piece (constructed next to `advect`,
unconditionally), so the metric is now universal. Cost: 4 more scalar
accumulators in an existing 1024-iteration host loop, ~1 Hz. No GPU change, no
new dispatch, no new buffer — the probe still cannot touch weights.

**`src/index.tsx`** — new always-visible `field-direction-chart` row ("R1
direction") in the diagnostics section, fed from the same metrics object as
`acHistory`, so the AC and R₁ traces are aligned point-for-point (the collapse
signature is R₁ climbing while AC falls). R₂ goes on the `field-structure-detail`
line. The adversary row is renamed **"R1 batch"** and now renders only when
actually measured, instead of a dead `—`.

**`tools/health_audit.mjs`** — `field.r1`/`field.r2` are REQUIRED in the
finiteness sweep (the grid probe always runs; a missing one is a schema
regression), `adv.r1` stays NULLABLE. New `agg.fieldR1`/`fieldR2` kept as
separate fields rather than `??`-coalesced with the batch pair — two
populations, and a verdict that silently substituted one would print a number
the reader cannot locate. `laminar-collapse` now fires on the grid pair first
and carries `where: "grid" | "batch"`.

**`tools/field_probe_test.ts`** — §1 closed-form: constant field ⇒
`r1 = ‖c‖/√(‖c‖²+τ²)` exactly (a hard-normalizing or wrong-τ implementation
passes "R₁ ≈ 1" and fails this); ± counter-streaming sheets ⇒ `r1 ≈ 0` but
`r2 > 0.9`; vortex ⇒ both ≈ 0; near-zero field ⇒ `r1 → 0` (the softener refuses
to invent a direction). §2 adds cross-implementation parity of the GPU probe
against `directionOrderParameters` itself — the HUD's whole claim is that grid
R₁ and batch R₁ are one statistic, and only that comparison supports it.

## Measured

`bun tools/field_probe_test.ts` — ALL PASS. Parity `r1`/`r2` at 1e-4 on
32²/64²/saturated cases (e.g. `probe 0.985612 vs collapse_probe 0.985612`).
`node tools/health_audit.mjs --self-test` — ALL PASS, including the new case
`grid R1 0.97 on a piece with NO adversary → laminar-collapse (was: healthy)`.

Live, headless Metal against `parcel src/index.html`, ~25–60 s per piece
(scratch probes, not committed):

| piece | grid R₁ | grid R₂ | batch R₁ | DC/AC | sat |
|---|---|---|---|---|---|
| Adversary · Pair · HashGrid · Curl (pressure ON) | 0.155 → 0.003 | 0.07 → 0.00 | 0.083 → 0.003 | 0.25 | 0.00 |
| same piece, `?advPolar=0&advNematic=0` | **0.52** | 0.27 | — (absent) | **71.9** | 0.00 |
| Pixel · VecField | **0.87** | 0.71 | n/a (no adv block) | 2.13 | **0.46** |
| Neural Field · Max Structure | 0.10 | **0.81** | n/a | 0.05 | 0.00 |

The pressure-off row is the instrument working: R₁ 0.52 with DC/AC 72 is the
documented collapse, and with the pressure on the same piece sits at 0.003.

## Unresolved / next

1. **Pixel · VecField reads laminar** (grid R₁ 0.87, sat 0.46). Both the
   `frozen-saturated` and the new `laminar-collapse` gate would fire on it;
   `frozen-saturated` outranks and wins. This piece previously had NO direction
   metric, so this is a first sighting, from a short headless run — needs a real
   soak (`node tools/health_audit.mjs vecfield …`) before anyone concludes the
   shipped piece is broken rather than the headless run being unrepresentative.
2. **Max Structure reads R₂ 0.81 with R₁ 0.10** — the ± counter-streaming
   nematic escape, which R₁ alone calls healthy. Deliberately NOT gated: this is
   the one piece that optimizes structure directly and the look may be
   intentional. Decide with a soak before adding an `r2Nematic` gate.
3. Sustained-AC decay observed on the hashgrid piece in headless (0.16 → 5e-4
   over ~1 min). Pre-existing, unrelated to this change, but it means headless
   numbers above are early-run values, not steady state.
