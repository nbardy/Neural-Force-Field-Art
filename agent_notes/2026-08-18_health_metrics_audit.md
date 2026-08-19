# Health metrics, headless audit, and the Max Structure piece

Date: 2026-08-18 (KST)
Predecessors: `2026-08-17_120215_KST_collapse_fix.md` (the diagnosis and the
DC/AC/R1/R2/OW/sat definitions), `2026-08-17_pressure_fusion_hud.md` (the fused
batch-moment machinery and the √2 ambiguity), `2026-08-17_soak_flake_attribution.md`
(why gates must read exact floats, never HUD text).
Scope: implementation + verification, in an isolated worktree. **Nothing committed.**

Legend: **[V]** verified by running it on this machine (Apple M4 / Metal).
**[H]** hypothesis.

---

## 0. Result in one table

`node tools/health_audit.mjs …`, real Metal adapter, production `dist` build.
All rows [V]. The BUILD column matters because Max Structure's weights were
tuned mid-session against exactly these measurements (§4); rows are the runs
that motivated or confirmed each decision.

| piece | build | run | verdict | ac | trend | dc/ac | sat | OW | R1 | payoff | fps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Adversary · Pair · HashGrid · Curl** (default) | — | 60 s | **healthy** | 0.2506 | rising | 0.216 | 0.000 | +0.28 | 0.055 | 0.572 | 60.0 |
| **Adversary · Pair · HashGrid · Curl** | — | 60 s (repeat) | **healthy** | 0.2481 | rising | 0.146 | 0.000 | +0.27 | 0.043 | 0.577 | 60.0 |
| Neural Field · Max Structure | W_DIV 0.05, lr 3e-3 | 150 s | healthy | 0.8735 | rising | 0.082 | 0.000 | −0.04 | — | — | 60.0 |
| Neural Field · Max Structure | W_DIV 0.05, lr 3e-3 | 60 s | **frozen-saturated** | 1.0609 | rising | 0.079 | **0.353** | +0.02 | — | — | 60.0 |
| Neural Field · Max Structure | W_DIV 0.3, lr 3e-3 | 75 s ×2 | healthy, healthy | 0.82 / 0.93 | rising | 0.030 / 0.087 | 0.000 / 0.139 | +0.04 / 0.00 | — | — | 60.0 |
| **Neural Field · Max Structure** (SHIPPED) | **W_DIV 0.6, lr 2e-3** | **120 s ×2** | **healthy, healthy** | 0.6994 / 0.6231 | rising / flat | 0.036 / 0.209 | **0.000 / 0.000** | −0.14 / 0.00 | — | — | 60.0 |
| **Pixel · VecField** | — | 60 s | healthy | 0.6830 | flat | 0.664 | 0.121 | +0.41 | — | — | 60.0 |
| **Pixel · VecField** | — | 150 s | **frozen-saturated** | 0.9111 | flat | 0.843 | **0.426** | +0.75 | — | — | 60.0 |

Four things worth reading off that:

1. **The default piece is healthy on the numbers, and the numbers agree with
   yesterday's qualitative finding.** R₁ 0.043–0.055 is inside the 0.01–0.11
   band the pressure note measured for the fixed default piece; dc/ac 0.15–0.22
   is essentially the 0.25 the offline prototype reached. Nothing in this work
   optimizes any of those — the pressure did that, and this is the first time
   the running artwork has said so in exact floats.
2. **W_STRUCT works, and by a lot.** At the shipped weights Max Structure holds
   ac 0.62–0.70 against the default piece's 0.25 — 2.6× the structure with
   dc/ac at 0.04 — at the same 60 fps.
3. **The audit changed the artwork.** Max Structure's first weights passed one
   run and failed the next (satFrac 0.35), which is what sent me looking for the
   Adam amplitude drift in §5 and produced the shipped W_DIV/lr. A single green
   run would have shipped a piece that freezes half the time.
4. **A REAL FINDING on the pixel critic** (§6): Pixel · VecField drives the
   field into tanh saturation as a STEADY STATE, and the audit caught it the
   first time it ran.

---

## 1. What was built

### `src/health.ts` (new, ~145 lines) — the wire format

`window.__nffHealth`, republished at ~1 Hz. The whole schema is documented in
the file; the two contracts that matter:

- **Absent ≠ zero.** `r1`/`r2` are `number | null`, and `null` means the
  anti-collapse pressure is not compiled so the moments are never reduced. A 0
  there would read as "perfectly isotropic", which is the single most flattering
  lie this instrument could tell about a collapsed field.
- **`AdvHealth` is a sum over `trainer`.** The fused arm carries `payoffUngated`
  and `batchRms`; the tfjs oracle arm has neither (it has no `isFiniteF` gate and
  no batch-RMS slot) and reports a DIFFERENT shape rather than padding the gaps
  with zeros.

```jsonc
{
  "piece": "Adversary · Pair · HashGrid · Curl",
  "frame": 3612, "t": 60.2, "fps": 60.01, "learnMs": 1.4,
  "backend": "webgpu", "trainer": "fused",
  "adv": { "trainer": "fused",
           "payoff": 0.5724, "payoffUngated": 0.5724, "surprise": 0.5724,
           "r1": 0.0550, "r2": 0.0612, "batchRms": 3.11, "heads": [61,74,59,62] },
  "field": { "ac": 0.2506, "dc": 0.0540, "rmsF": 0.2563,
             "satFrac": 0.0, "okuboWeiss": 0.2804, "gridN": 32 },
  "pixel": null
}
```

### `src/render/webgpu/field_probe.ts` (new, ~245 lines) + `fieldProbeShader`

- `fieldProbeShader(layout)` in `advect_wgsl.ts` — a diagnostics-only compute
  module that evaluates `forceAt` at an arbitrary NORMALIZED point list. It
  reuses the existing `emitHeadLooped` / `emitEncode` / `emitForce` emitters, so
  it supports raw, fourier and hashgrid encodings with no new field code.
- `FieldProbe` binds the **AdvectKernel's** packed weights buffer — the one
  buffer that is current on BOTH trainer paths (weights are born there when
  fused; `advect.encodeStep` syncs tfjs weights into it every frame otherwise).
  One instrument, one clean path, no per-trainer branch.
- **The reduction is on the HOST, on purpose.** 32² sites × a 5-point stencil is
  a 40 KB readback; doing the reduction in JS lets `fieldMetricsFrom` be the
  LITERAL transcription of `collapse_probe.ts::diagnostics`, which is the only
  reason the live number and the offline number are comparable at all
  (§2, gate 2). It is also pure, so most of it is testable without a GPU.
- AC is the **two-pass centred sum**, not `sqrt(mean‖F‖² − ‖mean F‖²)`. On a
  nearly-pure-DC field — exactly the collapse this instrument exists to catch —
  the one-pass form subtracts two nearly equal f32 sums and can land below zero.
  Cancellation there reads as "even more collapsed", or NaN.

### `src/render/webgpu/train_wgsl.ts` (+102 −13) — `FieldLossSpec.W_STRUCT`

Fused. It follows the `W_ISO` batch-moment + broadcast-backward pattern exactly:

- **fwd** already accumulates `(ΣFs.x², ΣFs.y², ΣFs.xy, Σloss)` for isotropy;
  the structure term reuses the first two (that IS `mean‖Fs‖²`) and adds ONE
  second reduction `red2 = (ΣFs.x, ΣFs.y)` into a second partials block at
  `[wgCount, 2·wgCount)`.
- **finalize** computes `L = (dc² + ε)/(rmsF² + ε)` and broadcasts three
  gradient scalars into `lossOut[7..9]`, plus the unweighted ratio at `[10]` for
  telemetry.
- **bwd** adds `(dS.x + 2·dS.z·Fs.x)/N·K` into the SAME `dFs` isotropy uses.

~55 lines of genuinely new WGSL, well inside the brief's 300-line stop-loss, so
it is fused and the tfjs `computeLoss` path exists only as the parity oracle.

### `src/core/losses/structure.ts` (new) — `constantModeFraction`

The tfjs oracle for the same functional, plus `acDcSplit` for telemetry. Shares
`STRUCTURE_EPS = 1e-8` with the WGSL `STRUCT_EPS`, and §1 of the new suite
asserts the two constants are literally equal.

### `src/main.ts` (+244 −16)

- `advHealthBlock()` / `pixelHealthBlock()` / `publishHealth()`, called once per
  ~1 Hz from `tick` AFTER the HUD block (so the snapshot and the HUD can never
  disagree about a frame). Both readbacks are fire-and-forget into local state —
  awaiting them would put a pipeline sync inside `tick`.
- `FieldProbe` constructed next to `AdvectKernel`; destroyed in the cleanup, and
  `window.__nffHealth` is **deleted** there, so a piece switch cannot let an
  auditor gate on the previous piece's numbers.
- `LoopHandle.getFieldHealth()`.
- `MAX_STRUCTURE_FIELD_LOSS` + the **appended** gallery piece (§4).
- `helmholtzChaosLoss` learned `W_STRUCT` (the tfjs oracle path).

### `src/index.tsx` (+83 −7) + `src/ui.css` (unchanged)

- **AC structure** row on the diagnostics panel: a sparkline of AC on the
  existing `Sparkline` (auto scale — AC has no absolute ceiling the way R₁ does)
  plus an exact readout, and a `chart-legend` line carrying `DC/AC · sat · OW`.
  Styling reuses `.diagnostic-row` / `.diagnostic-name` / `.chart-legend`
  verbatim, so no new CSS was needed.
- The diagnostics section's gate widened from `adversary` to
  `adversary || fieldHealth.tag === "measured"`, because AC is a FIELD metric and
  Max Structure — the one piece that optimizes it — has no adversary. The
  "game initialising…" row is now gated on `adversary` so a field piece does not
  claim to have a stuck game.
- The AC series folds on metrics-object IDENTITY, not on the 200 ms poll: the
  probe lands at ~1 Hz, so without that the series would be 80% repeated points
  and the EMA would lag by seconds.

### `tools/health_audit.mjs` (new, ~470 lines)

```bash
node tools/health_audit.mjs [pieceKeys|all|adversary] [baseURL] [sec] [sampleSec]
node tools/health_audit.mjs all http://localhost:8821/index.html 90 2
node tools/health_audit.mjs --self-test        # the pure half, no GPU
```

Exit code = number of unhealthy pieces. Artifacts (per-piece time series +
`summary.json`) under `output/health-audit/<iso>/`.

---

## 2. What each metric detects, with its threshold

Thresholds are MEASURED numbers from the predecessor notes, not guesses; each is
overridable by env var so a future tuner can move one without editing code.

| datum | gate | env | what it detects | provenance |
|---|---|---|---|---|
| `adv.r1` | > **0.5** sustained | `HEALTH_R1_MAX` | **laminar-collapse** — one global flow direction | collapse note: 0.88 → 0.999 collapsed, 0.01–0.11 healthy; 0.5 is clear of both |
| `adv.payoff` within **0.05** of √2 **AND** `r1` > 0.5 | | `HEALTH_POLE_BAND` | **pole-exploit** — encoded target went to zero, G is collecting the north-pole bonus | pressure note §5. **Both halves are required**: √2 with LOW R₁ is the healthy transient while D relearns, and gating on the payoff alone would fail the default piece for ~25 s of correct behaviour |
| `field.satFrac` | > **0.3** | `HEALTH_SAT_MAX` | **frozen-saturated** — both tanh components pinned past ±0.9; the field cannot move where the gradient wants it to | collapse note: sat 0 → 0.46 on the collapsed point observer |
| `field.ac` | < **1e-4** | `HEALTH_AC_DEAD` | **dead-field** — no spatially varying mode left; one constant push | collapse note: AC = 0.0007 caught the collapse in the act at step 800 |
| `adv.payoff` / `batchRms` / `pixel.extGradNorm` | > 1e3 / 1e4 / 1e6 | `HEALTH_PAYOFF_MAX` etc. | **blown-up** — diverging while still finite | steady-state payoff is 0.48–0.7, so 1e3 is three orders clear |
| any float nonfinite, **or** `payoffUngated ≠ payoff` | | | **nonfinite** — NaN/Inf, or the fused `isFiniteF` gate silently zeroed a nonfinite payoff | pressure note §4: that disagreement is the fused adversary's ONLY nonfinite canary |
| `fps` | < **30** | `HEALTH_FPS_FLOOR` | **perf-regression** | every shipped piece measures 60 on an M4 |
| `field.ac` slope | rel. slope ±0.002/s | `HEALTH_TREND_BAND` | rising / flat / falling — CONVERGENCE, which no single AC number can express | — |

**The verdict order is load-bearing.** These failures nest: a NaN field is also a
dead field and also a laminar one. `classify` is a κ that returns the FIRST
explanation accounting for everything after it, so a run producing NaNs is never
reported as "laminar-collapse" (which would send the reader to the pressure
weights when the bug is in a shader). `describe` is a thin exhaustive dispatcher
with one handler per tag and a `never` arm.

### Deviations from the brief's 8 verdicts — four added, each because a real run
### produced a state none of the eight described

| added | why |
|---|---|
| `no-signal` | The snapshot never produced a usable sample. Folding it into `nonfinite` would be a lie (there were no floats to be nonfinite) and throwing would lose the per-piece artifact. |
| `stalled` | **Found the hard way.** A 180 s Max Structure run froze at t = 36 s and the audit scored it **healthy** on 14 identical samples — a stopped loop republishes its last good snapshot forever and every metric in it stays finite and nominal. Caught by the snapshot's `frame` counter not advancing across the trailing samples. `soak_adversary.mjs` gates the same failure by watching the HUD element mutate; this is the same idea without touching the DOM. |
| `page-error` | A thrown page error invalidates every metric taken after it. It used to be reported as `nonfinite`, which produced the line "NONFINITE — 0 sample(s) carry NaN/Inf" — both untrue and unactionable. |
| `browser-stalled` | See §7. ATTRIBUTION BEFORE BLAME: a run in which the browser itself never produced a frame is not a statement about the artwork, and must not be printed as one. |

All four count as unhealthy (exit code).

### The bug this file was written to prevent, which it had anyway

`JSON.stringify(NaN)` is `null`. So is `JSON.stringify(Infinity)`.

The first 150 s VecField audit returned `ac: null` on a third of its samples and
scored the piece **healthy**, because the finiteness sweep skips non-numbers and
a NaN had been laundered into a legitimate-looking `null` in transit. A gate that
cannot see a NaN is not a gate — and this is the exact failure mode the brief's
"exact floats, never parsed HUD strings" rule is about, arriving through a door I
had not thought to close.

`null` is ALSO a real value in this schema (unmeasured R₁, unprobed field, a
pixel extGrad readback that has not landed), so "treat null as NaN" is not the
fix — the two must stay distinguishable. Nonfinite floats now travel as SENTINEL
STRINGS (`__nff_NaN__`, `__nff_Infinity__`, `__nff_-Infinity__`) and are restored
on arrival; genuine nulls travel as themselves. The same replacer is used for the
written artifacts, because a saved run whose NaNs have become `null` is a record
of a run that did not happen. `numbersOf` additionally distinguishes REQUIRED
slots (a non-number there is a schema/transport regression → reported as NaN)
from NULLABLE ones (absent is not nonfinite, and it is not zero). Five self-test
checks cover the round trip.

---

## 3. Test gates — all [V]

| gate | result |
|---|---|
| **NEW `tools/train_struct_test.ts`** | **ALL FUSED W_STRUCT CHECKS PASS** (49 checks) |
| **NEW `tools/field_probe_test.ts`** | **ALL FIELD PROBE CHECKS PASS** (34 checks) |
| **NEW `node tools/health_audit.mjs --self-test`** | **HEALTH AUDIT SELF-TEST PASS** (23 checks) |
| byte-identity sweep, **1098 blocks** | **BYTE-IDENTICAL** pre/post |
| `train_wta_pressure_test` | ALL FUSED PRESSURE CHECKS PASS |
| `train_wta_test` | ALL FUSED WTA CHECKS PASS |
| `train_wta_hashgrid_test` | ALL HASHGRID ADVERSARY CHECKS PASS |
| `train_types_test` | ALL TRAIN-TYPE CHECKS PASS (5 fixtures) |
| `adversary_wire_test` | ALL ADVERSARY WIRING CHECKS PASS |
| `pixel_disc_test` | ALL PASS |
| `url_guard_test` | ALL PASS |
| `field_loss_routing_test` | ALL PASS |
| `train_test` (readLoss ABI changed) | ALL PASS, fused step 0.670 ms @ B=256 |
| `npm run build` | clean, 3.60 s (final source) |
| live audit, real Metal | §0 table — 11 piece-runs, all before the §7 stall |

### W_STRUCT parity numbers

Oracle: `tf.variableGrads` of `constantModeFraction(field.forces(x)·forceMag)`
through a real `HelmholtzField`, mirrored bit-for-bit from the packed buffer.

| case | loss rel | grads |
|---|---|---|
| raw · struct only | 0.00e+0 | **cos 1.0000000**, relMax 4.6e-7 |
| raw · struct + iso (shared reduction) | 6.6e-8 | **cos 1.0000000**, relMax 2.5e-7 |
| fourier×3 · struct only | 1.0e-6 | **cos 1.0000000**, relMax 7.0e-7 |
| fourier×3 · struct + iso | 4.3e-7 | **cos 1.0000000**, relMax 7.4e-7 |
| hashgrid 16²×4 · struct only | 0.00e+0 | **cos 1.0000000**, relMax 5.1e-7 |
| hashgrid 16²×4 · struct + iso | 0.00e+0 | **cos 1.0000000**, relMax 1.2e-6 |
| **F ≡ const ≠ 0** (dc>0, ac=0) | L = **1 exactly** (ε cancels) | gradient **0 on both sides**, finite |
| **F ≡ 0 exactly** | L = ε/ε = **1**, the WORST score | gradient **exactly 0 on both sides**, finite |
| amplitude invariance | L(F) = L(4F) to 1e-7 | — |

Checks worth calling out, because each one can actually fail:

- **Isotropy is the superposition partner on purpose.** It is the OTHER consumer
  of pass A's `acc` vec4 reduction, so a wrong seam between the two shows up
  there and nowhere else. (My first draft superposed with `W_DIV` and the test
  failed — correctly: the fused divergence probes at `pos_K` on RAW `F`, not at
  the input site on `Fs`, so my oracle was simply the wrong function. Recorded
  because the next person will reach for W_DIV too.)
- **The degenerate cases are a DIFFERENT verdict type, not a special case.** A
  constant field is a stationary point of L (`∂L/∂F_i = (2F̄ − 2L·F_i)/N(ms+ε)`
  and `F_i ≡ F̄`, `L = 1`), so the true gradient is analytically zero and a
  cosine there compares f32 noise to f32 noise. `GradVerdict` is
  `parity{minCos,maxRel} | both-zero{tol}` and the degenerate cases assert an
  ABSOLUTE bound on both sides. Both sides are also asserted FINITE: a 0/0 inside
  the ratio would be NaN, and the downstream `isFiniteF` gates would silently
  turn that NaN into the same 0 the test wanted to see.
- **F ≡ 0 scoring 1 is the whole ε argument.** With ε in the denominator only, a
  dead field scores L = 0/ε = 0 — a PERFECT score — and the term would drive the
  field to zero to collect it. With ε in both it scores 1, the worst value. This
  is the same trap `directionOrderLoss` closes with its τ.

### Field-probe parity (gate 2)

The live `FieldProbe` on the packed weights vs `HelmholtzField.forces` reduced by
`collapse_probe.ts`'s own tf ops (written out again in the test rather than
sharing a helper — a shared reducer would make the comparison tautological), on
the same weights, two entirely separate evaluators:

| grid | ac | dc | rmsF | satFrac | OW |
|---|---|---|---|---|---|
| 32² structured | 0.117729 / 0.117729 | 0.577811 / 0.577811 | 0.589683 / 0.589683 | exact | −0.657282 / −0.657283 |
| 64² structured | 0.117769 / 0.117769 | 0.577799 / 0.577799 | 0.589679 / 0.589679 | exact | −0.657141 / −0.657140 |
| 32² **saturated (dead)** | 2.1e-6 / 2.1e-6 | 1.004987 / 1.004987 | 1.004987 / 1.004987 | exact | −0.0676 / −0.0693 |

Both regimes are tested on purpose: a parity test that only ever saw a
structured field could not tell whether the instrument reports structure or
merely reports something, and the saturated case doubles as a check that the
audit's `acDead` gate fires on a field that really is dead.

Plus closed-form fields that pin the definitions with no reference to any other
implementation: constant (`ac = 0`, `dc = ‖c‖`), pure shear (**OW = 0** — the
knife edge, which catches a swapped curl/strain sign that a one-sided test would
not), rotation (**OW = −1**), extension (**OW = +1**), saturated diagonal
(`satFrac = 1`), half-saturated (`0.5`), and `rmsF² = ac² + dc²`.

### Gate 7 — the no-optimization constraint, enforced

Three independent statements, in increasing strength:

1. **Source-level.** `git diff --stat HEAD` shows `src/render/webgpu/adversary_wgsl.ts`
   and `src/render/webgpu/adversary_train.ts` are **not in the diff at all**. The
   adversary's `extGrads` codegen and readback are byte-identical because their
   source is byte-identical. [V]
2. **Shader interface** (`field_probe_test.ts` §4a). The probe module binds
   weights as `var<storage, read>`, declares exactly bindings 0..3, and contains
   no `extGrad`/`grads`/`adam` identifier anywhere. It is not physically able to
   write a gradient buffer. [V]
3. **Behavioural** (§4b). Five probe samples leave all 440 packed weight floats
   **BIT-identical** — `Object.is`, not a tolerance, because an optimizer step of
   any size at all is a failure here. [V]

And the byte-identity sweep (1098 blocks: advect × {raw,fourier,hashgrid} ×
{helmholtz,agree-disagree,vector} × {staged,unstaged}, plus train pass A/B across
8 loss specs × K∈{1,2,4} × 2 borders × 3 extGrad counts) is BYTE-IDENTICAL
pre/post, so no piece that does not request `W_STRUCT` compiles a different
shader.

---

## 4. "Neural Field · Max Structure" — design

**Appended at the end of GALLERY**, after the default piece. Nothing reordered.

```ts
MAX_STRUCTURE_FIELD_LOSS = { W_STRUCT: 1, W_DIV: 0.6, everything else 0, HH: 1e-2 }
particleCount 200 000 · friction 0.985 · forceMagnitude 4.5 · maxVelocity 24
resetRate 0.006 · learningRate 0.002 · alpha-fade · stroke "curl" · bg [3,2,14]
fieldArch ARCH.dualStd α=0.7 · archEditable · lookEditable
```

**FUSED**, not tfjs: the WGSL turned out to be ~55 lines of genuinely new code
(one extra workgroup reduction, one partials block, three broadcast scalars, two
lines in bwd), well inside the brief's 300-line stop-loss. The tfjs
`computeLoss` path exists and is correct, but only as the parity oracle.

**Why normalized, not raw variance.** Stated in the piece comment and in the
`W_STRUCT` docstring, because it is the one thing a future editor will get wrong:
`ac` is homogeneous of degree 1 in `F`, so maximizing it is satisfied by GROWING
the field, not by structuring it. The optimizer walks into tanh saturation
(`satFrac 0.46` on the measured collapse baseline) and stops, having acquired
zero new spatial features. What the piece maximizes is
`ac²/(ac²+dc²) = 1 − L_struct`, which is invariant under `F → cF`, so the only
way to move it is to trade constant push for spatial variation. `train_struct_test`
§4 gates that invariance directly (L unchanged when `forceMagnitude` is 4×'d).

**Weights.**
- `W_STRUCT: 1` is the natural scale — `L_struct` is exactly [0,1] by Jensen, so
  1 means "one loss unit between a pure-DC field and a pure-AC one" and there is
  nothing to tune against.
- `W_DIV: 0.6` and `learningRate: 0.002` were **TUNED against the audit**, and
  the reason they had to be is the finding in §5b: this objective has a small but
  systematic Adam amplitude drift, and left alone the field random-walks its
  amplitude into tanh saturation. The sweep is in §0 and in the
  `MAX_STRUCTURE_FIELD_LOSS` docstring; the short version is

      W_DIV 0.05, lr 0.003 → satFrac 0.00 on one run, 0.35 on another
      W_DIV 0.3,  lr 0.003 → 0.00 and 0.14
      W_DIV 0.6,  lr 0.002 → 0.00 and 0.00

  Divergence is the right counterweight precisely BECAUSE `div_i = (∇·F)²` is
  unnormalized and grows with the field's derivatives — it is the one term in
  `FieldLossSpec` that has an opinion about amplitude at all. It must not go much
  higher, or the piece becomes a divergence-free piece that happens to import
  W_STRUCT. It also earns its place aesthetically: the structure term has no
  opinion about the local CHARACTER of the variation, and a compressible field
  satisfies it by building sinks that eat the cloud.
- Chaos / isotropy deliberately NOT stacked on. The Late Lesson from the collapse
  investigation is that piling terms on produces a field optimized for the sum
  and legible as none of them.

**Init is unseeded** (tfjs glorot, no seed), so Max Structure's trajectory is
genuinely run-to-run. One green run does not clear a weight here — that is how
the first configuration nearly shipped.

**Ink:** curl stroke + alpha-fade + `alphaBlend 0.05`, i.e. the default piece's
ink, because curl strokes draw each particle's curved per-frame trajectory and
that is what makes structure read as filaments rather than a dot cloud.
`forceMagnitude 4.5` (vs Max Chaos's 3.5) because the structure objective does
not itself ask for speed.

---

## 5. Finding: the ε placement is the entire loss

Recorded because it is not obvious and the wrong version type-checks, runs, and
looks like it is working for a few hundred steps:

```
L = (dc² + ε) / (rmsF² + ε)      correct — dead field scores 1 (worst)
L = dc² / (rmsF² + ε)            WRONG   — dead field scores 0 (best)
```

Both are scale-invariant, both are in [0,1] on a live field, and they agree to
seven digits on anything with real amplitude. They differ only in the limit — and
the limit is where an optimizer goes. The second version has a free global
minimum at `F ≡ 0` that the first does not, i.e. it is a field-killer wearing a
structure prior's clothes. Gated in `train_struct_test` §3.

## 5b. Finding: a scale-INVARIANT loss is not an amplitude-NEUTRAL gradient

The whole argument for the normalized ratio (§4) is that it cannot be satisfied
by growing the field. That is true of its VALUE and false of its GRADIENT:

    ∂L/∂(mean‖Fs‖²) = −L/(ms + ε)   < 0

so the descent direction contains a component proportional to `+F_i`, i.e. GROW
every force. Its size is proportional to `L`, which falls to ~0.001 once the DC
mode is gone — negligible as a magnitude, and completely non-negligible under
Adam, which renormalizes a small-but-systematic direction into a full-size step.
The end of that walk is tanh saturation, which is `satFrac`, which is the
`frozen-saturated` verdict — the audit caught its own new piece doing it.

It is worth being precise about what this is NOT: it is not the amplitude cheat
§4 rejects. There is no reward for a bigger field here; there is a slow drift
with nothing opposing it. A field that grows is not scoring better, it is
wandering along a direction the objective does not price. The two fixes available
were (a) an explicit amplitude anchor, which is a new fused term, and (b) a
counterweight that already prices derivatives — `W_DIV`. (b) shipped, measured
(§4). [H] The cleaner long-term fix is a stop-gradient on the denominator
(`dc²/detach(ms)`), which removes the drift direction entirely while keeping the
value identical; it was not done here because it would make the gradient a
semi-gradient and break the exact `tf.variableGrads` parity that §3 gates.

---

## 6. Finding: Pixel · VecField saturates its field

**The audit found this on its first real run, and nothing else in the repo
measures it.** [V]

The 60 s run flagged it (`satFrac` median 0.386) but was ambiguous — satFrac was
still FALLING at the end of the window, so it could have been a long transient.
The **150 s re-run settles it: it is a steady state.** [V]

```
t=5s   ac 0.835  dc 0.792  sat 0.309
t=30s  ac 0.954  dc 0.708  sat 0.411
t=75s  ac 0.935  dc 0.739  sat 0.443     ← plateau
t=150s ac 0.862  dc 0.796  sat 0.424
```

Median over the judged window: **satFrac 0.426, flat**, `dc/ac` 0.84, and
`OW +0.75` — strain-dominated, not vortical. A third to a half of the domain
sits pinned in the corners of the tanh box, where the gradient is ~0 and the
generator cannot move however hard the critic pushes.

Reading, stated as hypothesis [H]: the pixel critic's density gradient is a
LOW-FREQUENCY signal (it prices where the cloud IS, not which way the field
points), so the cheapest generator response is a large push — and nothing in a
pixel piece prices amplitude, since they carry `ZERO_FIELD_LOSS`. Same family as
§5b: an unopposed direction, not a reward. `W_STRUCT` (or any amplitude anchor)
is exactly the pressure that would price it — but adding a loss to a pixel piece
is a DIFFERENT decision from measuring it, and the no-optimization constraint
says this pass measures. Filed as an unresolved item (§8), not fixed here.

Run-to-run variance is large on this piece too (the 60 s runs saw satFrac 0.12
and 0.39 on the same build), so the 150 s plateau is the number to trust.

---

## 7. Environment: headless Chrome on this box stopped animating

**Late in the session, after roughly fifteen headless WebGPU browser sessions,
`requestAnimationFrame` stopped firing entirely — for a counter installed by the
probe itself, before the artwork loaded.** [V]

```
0s  {"raf":0,"health":false,"frame":null,"hudLen":0,"vis":"visible"}
…   (unchanged for 30 s; the app's own init logs DO appear, including
     "starting: Adversary · Pair · HashGrid · Curl (webgpu)")
```

Attribution, because it matters:

- It is not the new piece: **Neural Field · Max Chaos** (untouched, shipped, same
  200 k particles) fails identically.
- It is not the new code: the **default piece**, unmodified and measured healthy
  twice on the same `dist` forty minutes earlier, fails identically.
- It is not the artwork at all: the failing counter is `window.__nffRaf`,
  installed by the probe at `goto` time and owned by nothing in `src/`.
- Progressive, not sudden: a 180 s run froze mid-stream at t = 36 s (with
  `Page.captureScreenshot` timing out — a hung compositor), and subsequent runs
  produced no frames at all.
- Tried and did NOT help: `--disable-background-timer-throttling`,
  `--disable-backgrounding-occluded-windows`, `--disable-renderer-backgrounding`,
  `--disable-features=CalculateNativeWinOcclusion`, and `caffeinate -d -u`
  (display sleep would be a plausible cause — rAF is vsync-driven — but the
  machine reports `SleepDisabled 1` and waking it changed nothing). No orphaned
  `Chrome for Testing` processes were left behind.

Consequences, stated plainly:

1. **Every live number in §0 was measured before the stall and stands.** Nothing
   was re-measured afterwards, and the table names which build each row came
   from.
2. The **final comment-only edits** (the `MAX_STRUCTURE_FIELD_LOSS` docstring and
   two piece comments) were made after the last live run. `npm run build` is
   clean on the final source and the byte-identity sweep still passes, but the
   shipped `dist` has not been re-audited live since those edits. They change no
   value.
3. The auditor now DETECTS this rather than blaming the piece: the rAF counter is
   part of `tools/health_audit.mjs`, and a no-signal/stalled run with a dead
   counter reports `browser-stalled` with the tick count.

**A machine restart (or at least a fresh boot of the browser stack) is the first
thing to try before re-running the live gate.**

---

## 8. Unresolved / next actions

1. **Pixel · VecField's saturation (§6) is measured, confirmed as a steady
   state, and not diagnosed.** The next step is a `?pixelWeight=` sweep at 150 s
   per point, then either a small `W_STRUCT` on the pixel pieces or an explicit
   amplitude anchor. That is a design change to the artwork, deliberately out of
   scope here. The other three pixel pieces (NextFrame, RealFake, Inpaint) have
   NOT been audited at all — `node tools/health_audit.mjs all` covers only
   VecField today, and adding the other three to `PIECES` is a one-line change.
7. **The stop-gradient variant of W_STRUCT (§5b) is untried.** It would remove
   the amplitude drift at the source rather than counterweighting it, at the cost
   of turning the exact-parity gate into a semi-gradient gate.
8. **`Max Structure` at 150 s+ was measured only at the OLD weights.** The
   shipped W_DIV 0.6 / lr 0.002 config has two clean 120 s runs; the §7 browser
   stall stopped the longer confirmation. First thing to run after a restart.
2. **`okuboWeiss` has no gate.** It is reported and charted but nothing verdicts
   on it, because "vortex-dominated" is an aesthetic preference, not a health
   fact — the collapse note's `POLAR+NEMATIC+SWIRL` row reached OW −0.94 while
   the artwork got worse. Left as data.
3. **The audit's warmup is 3 samples.** At the default 2 s cadence that is 6 s,
   which is NOT enough to sit out the ~25 s √2 transient the pressure note
   measured on the default piece. It does not matter today because the
   `pole-exploit` verdict requires high R₁ as well, and the transient has R₁ ≈
   0.004 — but anyone who loosens that AND-condition must raise the warmup.
4. **`FieldProbe` samples the agree-disagree role mixture per grid site** (same
   hash the advect kernel uses per particle). That is the honest ensemble field
   the cloud sees, but it means the `agree` piece's AC includes role variance as
   well as spatial variance. Not yet separated; noted in the shader comment.
5. **One transient `train_wta_test` failure** ("6 FUSED WTA CHECK(S) FAILED")
   on the first of three consecutive runs; the next two passed with zero
   failures, and the run was concurrent with other GPU work. Same family as the
   bun-webgpu flake the pressure note recorded (§7.6 there). If it recurs on an
   idle GPU it is a real lead; it did not reproduce here.
6. **`W_STRUCT` is not exposed as a URL knob.** Every other field weight is
   piece-declared too, so this is consistent — but a `?wStruct=` would make the
   sweep in (1) a one-liner.

---

## 9. Files changed (nothing committed)

```
M src/core/losses/index.ts           (+8)     structure exports
M src/index.tsx                      (+83 −7) AC chart, field-health poll, section gate
M src/main.ts                        (+280)   health publisher, FieldProbe, W_STRUCT oracle,
                                              MAX_STRUCTURE_FIELD_LOSS, the appended piece
M src/render/webgpu/advect_wgsl.ts   (+89)    fieldProbeShader
M src/render/webgpu/train.ts         (+55 −5) W_STRUCT buffers, StructFraction, structWeight
M src/render/webgpu/train_wgsl.ts    (+102 −13) W_STRUCT codegen (fwd/finalize/bwd)
? src/core/losses/structure.ts       (new ~66)  constantModeFraction, acDcSplit
? src/health.ts                      (new ~145) the __nffHealth schema
? src/render/webgpu/field_probe.ts   (new ~245) FieldProbe + fieldMetricsFrom
? tools/health_audit.mjs             (new ~560) the headless auditor
? tools/train_struct_test.ts         (new ~330) W_STRUCT parity suite
? tools/field_probe_test.ts          (new ~300) probe parity + the no-optimization gate

UNTOUCHED (this is gate 7's strongest form):
  src/render/webgpu/adversary_wgsl.ts
  src/render/webgpu/adversary_train.ts
  src/render/webgpu/pixel_disc_wgsl.ts
  src/render/webgpu/pixel_disc_train.ts
  src/ui.css
```

## Reproduce, in order

```bash
bun tools/train_struct_test.ts          # W_STRUCT parity vs tfjs autograd
bun tools/field_probe_test.ts           # probe parity vs collapse_probe + gate 7
node tools/health_audit.mjs --self-test # verdict κ, no GPU

bun tools/train_wta_pressure_test.ts
bun tools/train_wta_test.ts
bun tools/train_wta_hashgrid_test.ts
bun tools/train_types_test.ts
bun tools/adversary_wire_test.ts
bun tools/pixel_disc_test.ts
bun tools/url_guard_test.ts
bun tools/field_loss_routing_test.ts

npm run build
cd dist && touch favicon.ico && python3 -m http.server 8821 &
node tools/health_audit.mjs hashgrid,struct,vecfield http://localhost:8821/index.html 60 2
node tools/health_audit.mjs all http://localhost:8821/index.html 90 2   # the full audit
```

GPU suites are SEQUENTIAL — run nothing else on the GPU.
