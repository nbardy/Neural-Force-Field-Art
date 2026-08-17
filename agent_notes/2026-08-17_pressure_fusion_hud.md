# Fusing the anti-collapse pressure + the R₁ / √2 HUD

Date: 2026-08-17 (KST)
Predecessors: `2026-08-17_120215_KST_collapse_fix.md` (the diagnosis + tfjs
prototype), `2026-08-17_120215_KST_hashgrid_adversary_fusion.md` (the codegen
patterns and gate conventions this follows).
Scope: implementation + verification. Working tree only — **nothing committed.**

Legend: **[V]** verified by running it on this machine (Apple M4 / Metal).
**[H]** hypothesis.

---

## 0. Result in one picture

Live, headless Chrome on the real Metal adapter, DEFAULT gallery piece
(`Adversary · Pair · HashGrid · Curl`), 70 000 particles, 60 fps, same seed,
same 45 s — the only difference is the pressure weight. [V]

| | pressure inert (`?advPolar=1e-9&advNematic=1e-9`) | pressure ON (shipped default 0.05/0.05) |
|---|---|---|
| R₁ over the run | 0.88 → **0.999** | **0.001 – 0.11** |
| artwork | one hue of diagonal laminar streaks across the whole canvas | a full field of **vortices** and swirling filaments |
| payoff | 0.53–0.70, then a √2 excursion at ~42 s as it locks laminar | √2 for ~25 s while D relearns, then **0.56–0.71** |
| speed | 13.7 px/f (that is the DC constant, i.e. the collapse) | 0.6–1.2 px/f |
| fps | 60.0 | 60.0 |

Screenshots were captured and read during the run; the qualitative flip
(streaks → vortices) is the same one the prototype note produced offline with
its own PPM renderer, now reproduced in the real artwork.

**The collapse the whole investigation was about is fixed on the shipped
default piece, on the fused path, at 60 fps.**

---

## 1. What was built

### `src/render/webgpu/adversary_wgsl.ts` (+250 −11)

- **`FusedGamePressure`** — the codegen twin of `main.ts`'s `GamePressure` and
  of `directionOrderLoss`. `none` | `anti-collapse{polar,nematic,tau}`, with a
  `checkedPressure` κ (τ > 0, weights finite ≥ 0, exhaustive `never` arm). It is
  part of the emitted-shader type, exactly like `FusedAdversaryLoss`; there is
  **no runtime branch in the hot loop**.
- **`advStatsLayout(k, pressure)`** — one source of truth for the stats
  geometry, shared by the codegen and the host. `none` → the historical
  `7+k` finalized / `7+k` partial stride; `anti-collapse` → four direction
  moments appended at `[7+k, 11+k)`.
- **Pass A forward.** Per member, capture raw `F` (nonfinite → 0, mirroring the
  existing `sig` guard) and accumulate
  `statMom += (uₓ, u_y, uₓ²−u_y², 2uₓu_y)` with `u = F/√(‖F‖²+τ²)`. **One vec4
  workgroup reduction** (`red4`) rather than four scalar ones.
- **Pass A backward.** `dM₁ = 2·wP·M₁/N`, `dM₂ = 2·wN·M₂/N`,
  `dL/du = (dM₁ₓ + 2dM₂c·uₓ + 2dM₂s·u_y, dM₁ᵧ − 2dM₂c·u_y + 2dM₂s·uₓ)`,
  `dL/dF = (dL/du − u(u·dL/du))/s` — added into **`dSig`**, i.e. the same
  cotangent the generator reward already fills, AFTER the physics Jacobian so
  the term is on raw `F` and not on a post-velocity signal.
- **Pass B: not one line of pressure code.** That is the load-bearing design
  claim (see §3).

### `src/render/webgpu/adversary_train.ts` (+99 −13)

- `pressure` constructor option (default `none`), stats buffer / staging /
  readback lengths all derived from `advStatsLayout`.
- **`DirectionOrder`** = `unmeasured | measured{r1, r2}`, derived on the host
  from the four moments (`R₁ = ‖M₁‖`, `R₂ = ‖M₂‖`). `unmeasured` is a real
  state, not a missing number — a pressure-free shader never reduces the
  moments, and reporting `0` would paint "perfectly isotropic" onto a field
  that may be fully laminar.
- **`AdvStats.discLoss` → `AdvStats.payoffUngated`** (rename + docstring; see
  §4 for why the slot was NOT recycled).

### `src/main.ts` (+194 −20)

- `fusedAdvOk` drops the `pressureTag === "none"` clause. `?train=tfjs` keeps
  the `directionOrderLoss` path selectable — it is now the parity oracle.
- `pressure: gamePressureOf(advSpec)` into `commonAdversaryOpts`, so both
  Agree+Disagree lanes get it (each prices its OWN field head; disjoint
  parameter blocks, additive, not double-counted — noted in the option's doc,
  with the hashgrid+agree-disagree caveat that nothing ships).
- **`GALLERY_ANTI_COLLAPSE` = polar 0.05 / nematic 0.05 / τ 0.05, baked into
  all 8 adversarial gallery pieces**, default piece included. Pixel-GAN pieces
  untouched.
- `PayoffReference` (`none | north-pole{chord}`) + `payoffReferenceOf(loss)`.
  Emitted only for `soft-angle`: the scale-augmented angle objectives carry an
  ADDITIVE local-scale term in the payoff, so √2 is not their reference value
  and drawing it would be a lie.
- `AdversaryTelemetry` gains `directionOrder` and `payoffReference`; BOTH
  trainers fill them. The tfjs path uses the new
  `directionOrderParameters(rawSignal, τ)`.
- `describePressure` in the FUSED trainer log line.

### `src/core/losses/isotropy.ts`

`directionOrderParameters(force, tau) → {r1, r2}` beside `directionOrderLoss`,
sharing one `softUnit` helper so the HUD number and the term the generator pays
are provably the same statistic. Exported from `core/losses/index.ts`.

### `src/index.tsx` + `src/ui.css`

- `Sparkline` gains two NAMED variants rather than loose optional props:
  `SparkScale = auto | fixed{lo,hi}` and `SparkRule = none | at{value,label}`.
  An `auto` domain is extended to contain a rule so "how far from that line"
  stays legible.
- The **G-residual chart is gone**, replaced by **`R1 laminar`** on a fixed 0–1
  scale with a dashed rule at the 0.10 healthy ceiling. The old two charts were
  provably one curve drawn twice (`predLoss` and `surprise` are the same
  zero-sum scalar), so the HUD contained no instrument that could have shown
  this collapse.
- The remaining chart is relabelled **`payoff`** and carries the dashed √2 rule
  under soft-angle only. `.sparkline-rule` in ui.css reuses the existing green
  token at lower opacity — a landmark, not data.

---

## 2. Test gates — all [V]

| gate | result |
|---|---|
| **NEW `tools/train_wta_pressure_test.ts`** | **ALL FUSED PRESSURE CHECKS PASS — 71 checks** |
| byte-identity sweep, 1080 blocks | **BYTE-IDENTICAL** pre/post |
| `train_wta_test` | ALL FUSED WTA CHECKS PASS |
| `train_wta_hashgrid_test` | ALL HASHGRID ADVERSARY CHECKS PASS |
| `train_types_test` | ALL PASS (5 fixtures, worst cos 1.0000000) |
| `kernel_test` | ALL PASS |
| `train_test` | ALL PASS (fused train step 0.750 ms @ B=256) |
| `pixel_disc_test` | ALL PASS |
| `adversary_wire_test` | ALL PASS |
| `field_loss_routing_test` | ALL PASS |
| stability, particle-coupled | `TAG=pair-rotation-scale-adjusted ENC=hashgrid K=4 N=60000 WEIGHT=0.015 POLAR=0.05 NEMATIC=0.05 STEPS=4000` → **QUAD COUPLED PROBE FINITE** |
| effect, tfjs oracle | `PRESET=pair STEPS=1000 POLAR=0.05 NEMATIC=0.05 collapse_probe` → R₁ 0.173 → **0.022**, never above 0.064 |
| build | `parcel build --no-scope-hoist` clean, 4.7 s |
| live soak | `soak_adversary.mjs hashgrid` → **ALL 20 GATES PASS**, median 60 fps |
| live pressure probe | **LIVE PRESSURE PROBE PASS** (6 gates) |

### Parity numbers (new suite)

Oracle: a real `HelmholtzField` + `tf.variableGrads` of
`directionOrderLoss(field.forces(pos), τ, wP, wN)` over all B·m member outputs.
Eight configurations; **every non-degenerate one at cos = 1.0000000**:

| case | pressure-only extGrads | superposition | tfjs ∇(reward+pressure) |
|---|---|---|---|
| raw · point · soft-angle | cos 1.0000000, rel 2.9e-7 | cos 1.0000000, rel 2.6e-7 | cos 1.0000000 |
| raw · pair-rot-scale-adj (the shipped game) | cos 1.0000000, rel 2.4e-7 | cos 1.0000000, rel 3.0e-7 | cos 1.0000000, rel 2.5e-7 |
| **hashgrid 16²×4 · pair-rot-scale-adj (default piece)** | cos 1.0000000 | cos 1.0000000 | cos 1.0000000 |
| hashgrid 8²×4 · point | cos 1.0000000 | cos 1.0000000 | cos 1.0000000 |
| raw · point · **post-velocity** target | cos 1.0000000, rel 3.3e-7 | cos 1.0000000, rel 1.8e-7 | cos 1.0000000, rel 6.7e-7 |
| raw · pair · **F ≡ 0 exactly** | grad EXACT 0 and finite on both sides; oracle loss exactly 0 | rel 0.00e+0 | cos 1.0000000 |
| raw · pair · **F ≡ const ≠ 0** | cos 1.0000000, rel 4.4e-5 | cos 1.0000000, rel 2.0e-6 | cos 1.0000000, rel 1.5e-6 |
| hashgrid 8²×4 · pair · **F ≡ 0 exactly** | grad EXACT 0 and finite | rel 0.00e+0 | cos 1.0000000 |

Checks worth calling out, because each one can actually fail:

- **The post-velocity case is a trap detector.** There `sig` is a velocity, so a
  pressure that read `sig` instead of raw `F` would disagree with the oracle by
  exactly the physics Jacobian. cos 1.0000000 says it reads `F`.
- **R₁/R₂ vs `directionOrderParameters`** agree to < 2e-5 on every case, so the
  HUD number and the penalized quantity are the same statistic, on-device and
  on the CPU.
- **The degenerate cases are pinned in CLOSED FORM, not "close to".** F ≡ 0 ⇒
  R₁ = R₂ = 0 exactly and the gradient is exactly 0 **and finite on both
  sides** — a 0/0 in `unit_τ` would produce NaN, which the `isFiniteF` gate
  would then silently replace with the same 0, so the tfjs side is asserted
  finite too. F ≡ const ⇒ R₁ = ‖F‖/√(‖F‖²+τ²) = 0.998104 and R₂ = R₁² =
  0.996212 — **not** R₁ = 1; the τ shortfall is the softener doing its job, and
  getting that wrong was the one real bug this suite caught in its first run.
- **Superposition** — `extGrads(reward+pressure) ≡ extGrads(reward) +
  extGrads(pressure)` at cos 1.0000000 from two SEPARATELY COMPILED shaders,
  and the reward component recovered from the pressured shader is bit-close to
  the plain shader's. A wrong seam would show as a scale error even when each
  part is individually right.

### Byte-identity sweep

Same technique as the fusion agent: `git archive HEAD` into a temp tree, run one
generator against both trees, `cmp`. **1080 blocks** — adversary pass A + pass B
across {raw, fourier, hashgrid} × {helmholtz, agree-disagree} × 7 tuple tags ×
4 losses × 2 targets × 3 field lanes, plus the field trainer's pass A/B at two
rollouts × extGrad on/off, plus all 4 pixel-GAN kinds — **byte-identical**.

One wrinkle worth recording: the first sweep failed on a single line, because I
had corrected a COMMENT inside the generated pass-B WGSL ("genSeed == 0 ⇒ exact
zeros", which stops being true once a pressure is declared). Rather than leave a
false comment or break the property, that comment is now itself pressure-variant
— a pressure-free call emits the historical text verbatim. Stated out loud in
`adversaryPassBShader`'s docstring so the next reader is not surprised.

### Overhead

Pressure OFF costs **exactly zero** — the shader is byte-identical, so there is
nothing to measure. That is why the moments are compile-gated rather than
always-on (the brief's fallback), and the cost of that choice is that
`?advPolar=0&advNematic=0` also turns the R₁ instrument off; the honest way to
watch a baseline collapse is `?advPolar=1e-9&advNematic=1e-9`, which measures
the moments with an inert gradient. This is documented in
`tools/pressure_live_probe.mjs`.

Pressure ON, full `encodeStep` (5 passes incl. disc Adam + fieldGrad), 300 steps
after 30 warmup, Apple M4 Metal [V]:

```
raw dualStd                    B=256   off 0.7418 ms  on 0.7476 ms   Δ +5.8 µs (0.8%)
raw dualStd                    B=512   off 0.9505 ms  on 0.9554 ms   Δ +4.9 µs (0.5%)
hashgrid 32²×4 (default piece) B=256   off 0.8235 ms  on 0.8241 ms   Δ +0.6 µs (0.1%)
hashgrid 32²×4 (default piece) B=512   off 1.0470 ms  on 1.0518 ms   Δ +4.8 µs (0.5%)
```

Effectively free (a second run put two of the four deltas NEGATIVE, i.e. the
term is inside run-to-run noise). One vec4 reduction, one reciprocal and one
projection per member.

---

## 3. Does the pressure really reach every encoding for free? — YES [V]

The brief asked this to be verified rather than assumed. It holds, and the
mechanism is worth stating because it is the reason the diff is small:

pass A's generator backward ends at `dSig[t] = dL/dF` for each member, and
`fieldBackward(t)` then calls `bwdCall(h, dSig[t], t)` — the ONE site that is
already dispatched on `enc.kind` (raw: 2 args, fourier: 3, hashgrid: 4 with the
`dEncBase` block). The pressure adds its `dL/dF` into `dSig` **before** that
call, so it inherits the entire dEnc/bwd machinery, and `fieldGrad` in pass B —
including the gather-side hashgrid grid block — assembles it with no
pressure-specific code at all.

Verified, not just argued: the hashgrid cases in the parity table hit
cos = 1.0000000 including a 16²×4 grid on the production game, and
`adversaryPassBShader` is identical modulo comments with and without pressure
(asserted in §1 of the new suite).

---

## 4. Deviation from the brief, with the reason

The brief said to **replace stats[0]'s semantics** with R₁, on the ground that
`stats[0] ≡ stats[1]`. I renamed it instead of recycling it, for two reasons:

1. **One slot could never have carried this.** The backward needs all FOUR
   moments as batch constants (`dL/du` depends on M₁ₓ, M₁ᵧ, M₂c and M₂s
   separately), not one scalar. R₁ is derived on the host from them.
2. **stats[0] is not fully redundant.** It is the payoff BEFORE the `isFiniteF`
   gate; stats[1] is after. They are bit-identical *whenever the payoff is
   finite*, and their disagreement is the only nonfinite-payoff canary the fused
   adversary has — `tools/quad_nan_probe.ts` uses exactly that. Deleting it
   would have removed a numerical tripwire from the default piece.

So: four moments appended at `[7+k, 11+k)` under pressure only (`ADV_STATS_BASE`
= 32 still clears the largest legal prefix, 11+16 = 27), and
`AdvStats.discLoss` → **`AdvStats.payoffUngated`** with a docstring that states
the redundancy and the canary role. The "do not overload silently" requirement
is met by naming, without losing an instrument. Call sites updated in
`quad_nan_probe`, `adversary_stability_probe`, `train_wta_test` (mechanical).

---

## 5. Finding: the √2 line is AMBIGUOUS, and the brief's reading was half of it

The predecessor note (and my first draft of the tooltip) says a payoff parked
near √2 means the encoded target has gone to zero. Measuring it live showed
that is only one of two ways to get there, and the other one is the GOOD case:

- `payoff ≈ √2` **and** `R₁ → 1` — the target went to zero; the field is flat
  and G is collecting the north-pole bonus. This is the collapse. Seen live in
  the inert-pressure baseline at ~42 s, as R₁ climbed 0.6 → 0.94.
- `payoff ≈ √2` **and** `R₁ → 0` — the field is so direction-isotropic that D's
  best response IS the pole. Seen live with the pressure ON: a **~25 s
  transient** (payoff 1.31–1.39 while R₁ sat at 0.004) that decays to 0.56–0.71
  as D relearns. That is the game working.

Both readings are now in `PayoffReference`'s docstring and in the HUD tooltip,
which says READ IT WITH R1. This is also the strongest justification for the
chart layout: the two numbers are only diagnostic together, which is exactly
what the old duplicated-chart HUD could not express.

**Practical consequence:** a 25 s probe of the default piece looks alarming and
a 45 s probe does not. Anyone re-running `pressure_live_probe.mjs` should give
it ≥ 45 s.

## 6. Finding: the fused term bites harder than the tfjs prototype at the same λ

Same functional, different sampling measure — and nobody had reason to notice
until both existed:

- the tfjs prototype evaluates the pressure on the field trainer's **fresh
  uniform** batch (a domain average);
- the fused term evaluates it on the adversary's **live-particle** tuple
  members (a particle-weighted average — where the art actually is).

Measured live at λ = 0.05 the fused version drives R₁ to **0.001–0.11**, an
order of magnitude below the 0.057 the tfjs prototype reached on the pair
preset. A λ sweep on the live default piece [V]:

| λ (polar = nematic) | R₁ | payoff | note |
|---|---|---|---|
| 1e-9 (inert, measurement only) | 0.88 → **0.999** | 0.53–0.70 then a √2 excursion | the collapse |
| 0.01 | **0.013–0.090** | 0.54–0.68 throughout | no √2 phase at all |
| 0.05 (shipped) | **0.001–0.11** | √2 for ~25 s, then 0.56–0.71 | very isotropic |

Both 0.01 and 0.05 meet the note's stated target (R₁ ≤ ~0.1) and both produce
vortices. I shipped **0.05 as instructed** — it is the value the predecessor
note recommends and it holds R₁ an order of magnitude lower. If the artwork
reads as too slow (speed 0.6–1.2 px/f versus the collapsed baseline's 13.7 —
though that 13.7 is 100 % DC, i.e. the dead constant field), **λ = 0.01 is the
first thing to try**, and it removes the √2 warm-up phase as a side effect.

---

## 7. Unresolved / next actions

1. **λ is still untuned in the proper sense.** §6 is three points on one piece.
   The two weights have never been varied independently, and only the default
   piece was swept live; the other seven adversarial pieces inherit 0.05/0.05
   on the strength of the offline prototype.
2. **Magnitude is still unpriced**, exactly as the predecessor note warned. The
   direction-order term has no opinion about ‖F‖, and `u = F/√(‖F‖²+τ²)` means
   shrinking ‖F‖ below τ is itself a way to zero the loss — an escape hatch
   nothing currently closes. It did not fire in any measurement here (offline
   `yRms` ROSE 0.125 → 0.41 over 4000 pressured steps), but the honest pairing
   remains direction-order **plus** the two-sided rms‖y‖ anchor. Deliberately
   NOT stacked with a swirl/Okubo-Weiss term — the Late Lesson.
3. **RMS‖y‖ is not on the HUD.** It is in `AdvStats.batchRms` and it is the
   third number needed to disambiguate §5 without waiting out the transient.
   One telemetry field and one HUD line.
4. **`?advPolar=0&advNematic=0` also disables the R₁ instrument** (§2,
   Overhead). That is a deliberate consequence of the byte-identity requirement.
   If R₁ is ever wanted on a pressure-free piece, the fix is a separate
   `measureDirectionOrder` compile flag — at which point the byte-identity claim
   has to be re-scoped to "shaders that request neither".
5. **`src/main.ts.orig`** — a 160 KB merge-conflict leftover is TRACKED in the
   repo (committed at c342cb7). Nothing imports it; it pollutes every search
   over `src/`. Filed as a separate task chip.
6. **One transient bun-webgpu flake** during a pressure-suite run: a shader
   module failed to compile with `Error while parsing WGSL: null character
   found` at 1:1 on a config whose generated text is provably fine (it compiled
   on every other run, and the byte-identity sweep parses the same string).
   Did not reproduce in 4 subsequent runs. Looks like a bun-webgpu string
   marshalling issue, not a codegen one — but if it recurs, that is the lead.

---

## 8. Files changed (nothing committed)

```
M src/core/losses/index.ts             (+12 −4)    directionOrderParameters export
M src/core/losses/isotropy.ts          (+71 −20)   directionOrderParameters + shared softUnit
M src/index.tsx                        (+164 −40)  SparkScale/SparkRule, R1 chart, √2 rule
M src/main.ts                          (+194 −20)  gate, GALLERY_ANTI_COLLAPSE, PayoffReference, telemetry
M src/render/webgpu/adversary_train.ts (+99 −13)   pressure opt, DirectionOrder, payoffUngated
M src/render/webgpu/adversary_wgsl.ts  (+250 −11)  FusedGamePressure, moments, backward, stats layout
M src/ui.css                           (+9)        .sparkline-rule
M tools/adversary_stability_probe.ts   (+2 −2)     payoffUngated rename
M tools/quad_nan_probe.ts              (+30 −4)    POLAR/NEMATIC/PTAU knobs, R1/R2 in the trace
M tools/train_wta_test.ts              (+9 −9)     payoffUngated rename
? tools/train_wta_pressure_test.ts     (new, ~560) the parity suite
? tools/pressure_live_probe.mjs        (new, ~130) live R1 / √2 / fused-pressure probe
```

## Reproduce, in order

```bash
bun tools/train_wta_pressure_test.ts          # the new parity suite
bun tools/train_wta_test.ts                   # raw/fourier adversary regression
bun tools/train_wta_hashgrid_test.ts          # hashgrid regression
bun tools/train_types_test.ts
bun tools/kernel_test.ts && bun tools/train_test.ts
bun tools/pixel_disc_test.ts && bun tools/adversary_wire_test.ts
bun tools/field_loss_routing_test.ts

TAG=pair-rotation-scale-adjusted ENC=hashgrid K=4 N=60000 WEIGHT=0.015 \
  POLAR=0.05 NEMATIC=0.05 STEPS=4000 REPORT=400 bun tools/quad_nan_probe.ts
PRESET=pair STEPS=1000 POLAR=0.05 NEMATIC=0.05 bun tools/collapse_probe.ts

npm run build
cd dist && touch favicon.ico && python3 -m http.server 8811 &
node tools/soak_adversary.mjs hashgrid http://localhost:8811/index.html 60 10
node tools/pressure_live_probe.mjs http://localhost:8811/index.html 45
SHOT=/tmp/base.png node tools/pressure_live_probe.mjs \
  "http://localhost:8811/index.html?advPolar=1e-9&advNematic=1e-9" 45
```

GPU suites are SEQUENTIAL — run nothing else on the GPU.
```
