# Adversary release completion

## Goal

Finish every item from the fresh-eye audit without touching unrelated splat
work, then verify the current source in real WebGPU browser soaks.

## Verified starting state

- Production build passes.
- 13/14 offline suites pass.
- `tools/adversary_wire_test.ts` is stale and fails four assertions because it
  still feeds a position transition where the strict API now requires raw
  `F(x)`, expects the legacy `pair` spelling, rejects the newly-added labelled
  quad, and expects the retired `data-limited` telemetry overclaim.
- Strict raw-force math, relaxed-WTA shared payoff, post-discriminator generator
  recomputation, adjusted target tangent gradient, boundary Jacobians, and
  A/B lane-isolation kernels have focused passing CPU/Metal gates.
- Current-source browser QA contradicts the short gates:
  - `Adversary · Agree + Disagree RGB` immediately throws
    `adversary: fused adversary needs the helmholtz field`.
  - R+S Adjusted reaches `loss NaN` in roughly four seconds.
  - R and R+S Raw eventually stop with a non-finite `genSeed` (observed within
    roughly 5–21 seconds).
- The live loop owns one fused adversary trainer and never reads
  `AdversarySpec.game`; Agree+Disagree therefore has neither A/B trainers nor
  two external gradient buffers.
- Fused surprise uses random-with-replacement indices, non-atomic duplicate
  scatter writes, stale untouched values, a fixed-prefix statistic, and a
  hard-coded 100% coverage claim.
- `quad-labelled` is implemented only in the tfjs reference. `?advM=4` can
  select it even though `TupleTag`/fused runtime do not support it.
- The top-right dock and bottom gallery exist, but the header renders `CONFIG`
  twice and several material training controls remain hidden.
- Documentation still contains one-step-transition, tfjs-only, old
  Helmholtz/order-vs-chaos, and overconfident head-support claims.

## Mathematical concern requiring a real fix

The raw payoff is unbounded in `F`: a generator may increase residual by scaling
the raw field. The adjusted angular payoff has zero instantaneous radial
gradient in signal space, but magnitude is a null direction, not a restoring
constraint. A shared nonlinear network and coordinate-wise Adam can still move
field magnitude, and the live browser proves that the current game becomes
non-finite. Componentwise velocity clipping protects particles but not raw field
weights or adversary targets.

A releasable game therefore needs an explicit compact/finite generator action
space or restoring constraint, plus non-finite containment. Candidate policies
must be tested rather than asserted:

1. fixed or softly-targeted RMS raw-field magnitude;
2. bounded/normalized physical field output;
3. explicit clip-fraction or force-energy penalty;
4. gradient/update clipping and fail-fast finite telemetry.

The adjusted observer still needs its tangent payoff; a magnitude constraint is
orthogonal and must not be misrepresented as the quotient itself.

## Work stages

1. Wire Agree+Disagree as A lane 0/disagree and B lane 1/agree, with independent
   predictors, two ext-gradient buffers, zero direct C loss, branch telemetry,
   and browser/runtime gates.
2. Add a bounded field-magnitude policy and non-finite guards. Gate both
   short analytical properties and a long live co-training soak.
3. Replace surprise scatter with deterministic unique rotating coverage and
   truthful coverage/statistics.
4. Finish labelled quad in fused WGSL or force an explicit safe tfjs fallback;
   expose the result honestly in the UI.
5. Repair `adversary_wire_test`, tri/quad invalid-win accounting, complete the
   top-right controls, and remove the duplicate header.
6. Rewrite stale docs/comments.
7. Run all CPU/Metal/build suites and Playwright current-source desktop/mobile
   interaction plus per-adversary soaks.

## Completion rule

Do not call the work complete merely because the static suites pass. Every
shipped adversarial gallery piece must initialize, remain finite for the soak
window, keep rendering and telemetry alive, and produce no page/console error.

## Final diagnostic audit (2026-07-29)

Verified open issues discovered after the first fused release pass:

- The UI/docs promise a per-unit-signal surprise diagnostic, but production
  WGSL currently exposes only the raw shared WTA payoff.
- The tfjs fallback `SurpriseChannel` is semantically wrong for the strict
  game: it observes a previous/next particle transition while discriminator
  training observes raw `F(x)`. It is also intrinsically wrap-specific. The
  honest fallback is velocity rendering with a loud warning; tfjs remains an
  oracle/reference trainer, not a second cloud-diagnostic implementation.
- Fused head health is still classified with `{tag:"unprobed"}`. A real
  predictor-head spread reduction must feed the conservative health verdict.
- Agree+Disagree currently averages A/B win evidence and uses A's scale for one
  combined verdict, which can hide a failure isolated to one branch. Branches
  must be classified independently and combined conservatively.
- `ADV_STATS_BASE=16` overlaps the finalized-stats region for otherwise legal
  high head counts. The repaired layout reserves 32 floats and adds explicit
  mean/closest head-spread fields before the workgroup partials.

Implementation decision:

1. Keep raw-payoff and per-unit-signal diagnostics as two planes in one fused
   GPU buffer. The diagnostic selection affects rendering/stat sampling only;
   it must leave discriminator loss, generator payoff, gradients, Adam state,
   and field weights bit-identical.
2. Define per-unit display value as `payoff / ||y||` only for an active strict
   target with `||y|| > 1e-3`; otherwise write exact zero via a branch so WGSL
   never evaluates `0/0`.
3. Measure head spread over all predictor outputs at a context, including
   target-inactive contexts because spread is a property of `g_j(u)`. Adjusted
   R+S compares normalized predicted directions, matching the tfjs oracle.
4. Remove the transition-based live fallback and reject raw/per-unit cloud
   diagnostics whenever the fused adversary is unavailable.

## Agree+Disagree live wiring (completed 2026-07-29)

Verified implementation, not a proposal:

- `validateAdversaryFusion` now accepts both supported two-head field
  semantics (`helmholtz` and `agree-disagree`) while retaining the class,
  hash-grid and activation safety gates.
- The live loop constructs two independent `AdversaryTrainer` instances for
  the Agree+Disagree piece:
  - A: `fieldLane: 0`, `generatorRole: "disagree"`, seed `20260727`;
  - B: `fieldLane: 1`, `generatorRole: "agree"`, seed `20260728`.
- A and B share only the field's read-only packed weights. Each owns its own
  predictor weights, Adam state, scratch, statistics, surprise and external
  gradient buffer.
- Each frame records A's full alternating `D -> G` step, then B's full
  alternating `D -> G` step, before one `FusedTrainer` step. That field trainer
  binds both external-gradient buffers and sums them before exactly one field
  Adam update.
- C remains `C=(1-alpha)A+alpha B` in advection/rendering only. The packed
  layout contains exactly two field heads; C has no weights, optimizer or
  direct loss.
- Particle-buffer replacement and cleanup now update/destroy both trainers.
  The named generator role owns the sign; live weights are clamped non-negative
  rather than smuggling "agree" through a negative generator seed.
- Telemetry retains aggregate compatibility while reporting branch-specific
  A/B surprise and predictor loss in the FPS HUD; C is labelled display-only.
- `?train=tfjs` is rejected for this piece with an explicit warning because
  the tfjs fallback does not implement two lane-isolated games.

Focused gate added: `tools/agree_disagree_live_test.ts`. It instantiates the
exact Fourier `agree-disagree` layout, proves independent A/B predictors,
exact lane isolation, exact A+B external-gradient summation, both discriminator
updates, absence of a C head, and one finite summed field update.

Measured commands:

- `npm run build` — PASS, Parcel production build in 5.28 s.
- `bun tools/agree_disagree_live_test.ts` — ALL PASS (7 assertions).
- `bun tools/train_wta_test.ts` — ALL FUSED WTA CHECKS PASS, including
  alternating-order and lane/role isolation gates; pair K=4 0.992 ms/step,
  tri K=6 1.355 ms/step on this run.

This closes the construction/runtime-topology blocker. It does **not** by
itself close the separately tracked long-soak numerical-stability, surprise
coverage, labelled-quad, stale wire-test, or documentation work.

## UI, wiring-test and documentation repair (completed 2026-07-29)

Verified changes:

- `src/index.tsx` / `src/ui.css`
  - removed the redundant literal `CONFIG` heading while retaining the active
    piece name as the dock header;
  - kept every `LoopHandle`-backed live control in the compact top-right dock;
  - labelled border and observer/K/epsilon as restart/compiled choices;
  - renamed ambiguous `samples`/`random` labels to `train B`/`respawn`;
  - exposed the signed adversary-weight range supported by the runtime;
  - pair observer modes show R / R+S RAW / R+S ADJ, while point, tri and
    labelled-quad observers render honest read-only descriptions rather than
    an unselected pair radiogroup;
  - Agree+Disagree keeps exact RGB A/B/derived-C coloring instead of offering a
    generic color toggle that would hide the experiment.
- `tools/adversary_wire_test.ts`
  - discriminator and generator gates now consume strict raw `F(x)`, not a
    post-physics transition;
  - the reward-sign gate uses one fixed tuple set and symmetric directional
    derivative steps, avoiding random resampling/Adam momentum as confounders;
  - URL gates expect explicit `pair-rotation`,
    `pair-rotation-scale-adjusted`, and supported `quad-labelled`;
  - separated heads with skew are asserted as `separated-unresolved`, never
    guessed to mean `K > modes`.
- `docs/ADVERSARY_STATUS.md` was rewritten as the operational truth: strict
  target, observer semantics, Agree+Disagree contract, neutral two-head field,
  conservative health states, fused/AD distinction, honest surprise coverage,
  UI live/restart boundary, and release gates.
- `docs/PLAN_RELATIONAL_ADVERSARY.md` was rewritten without dropping the
  earlier idea ledger. It retains the degeneracy proof, WTA/geometric-median
  math, rotation/scale observers, labelled-quad rationale, noisy-TV concern,
  Agree+Disagree equations, chaos/Helmholtz distinction, AD-IR role,
  compression progress, replay novelty, BALD, temporal targets, finite-time
  Lyapunov, JVP probes, spectrum/harmonic ideas, and strict verdicts.
- `HANDOFF.md` now marks the old Phase-1/Phase-2 prose as historical, qualifies
  the 1M/60-FPS number as one benchmark rather than a universal guarantee, and
  points to the current adversary docs/test matrix.
- Stale order/chaos architecture comments were corrected in
  `src/core/field/helmholtz.ts`, `src/render/webgpu/advect.ts`, and the AD
  comments. The class name remains for compatibility; alpha is documented as a
  neutral A/B mix.

Measured commands from this phase:

- `bun tools/adversary_wire_test.ts` — PASS,
  `ALL ADVERSARY WIRING CHECKS PASS`.
- `npm run build` — PASS, Parcel production build in 17.02 s.
- `git diff --check --` over every owned source/doc/test file — PASS, no
  whitespace errors.

Still intentionally open for the root completion pass: broad CPU/Metal
regressions, real-browser desktop/mobile interaction, long NaN/magnitude soaks,
and final revision of the pending surprise-coverage/quad status after their
parallel implementation lands.

## Numerical-stability root cause and repair (completed 2026-07-29)

### Questions investigated

- Why did adjusted R+S become `NaN` first even though its payoff is radially
  scale-invariant?
- Why did the raw control later become non-finite even while its external
  adversary gradient was finite?
- Was `maxVelocity` hiding an optimization exploit or causing either NaN?
- Does the gallery reward weight actually control the generator step under
  Adam?

### Verified pre-fix failures

`tools/adversary_stability_probe.ts` traces stats, predictor gradients/weights,
external generator gradients, field gradients/weights and target geometry on a
shared real Metal buffer.

- Adjusted R+S at field/G LR `0.008` first became non-finite at step 830 in the
  external field gradient. Its normalized target has Jacobian
  `(I - yyᵀ)/||q||`; the former `1e-8` active floor allowed a condition number
  approaching `1e8` as `q = F(x₂)-F(x₁)` approached zero.
- Raw R+S remained finite through its external gradient at step 1708, then the
  field trainer produced non-finite internal gradients/weights. The standalone
  field loss was numerically zero only by multiplication: disabled structural
  terms were still evaluated, so a saturated locally-constant field could
  create an undefined intermediate and IEEE-754 `0 * undefined` became NaN.
- Standalone G LR was `0.006–0.008` while predictor/D LR was `0.003`. Moreover,
  rescaling a standalone loss by a positive reward weight is approximately
  cancelled by Adam's first/second-moment ratio. The reward weight is therefore
  not an honest generator-step-strength control; field Adam LR is.
- `maxVelocity` clips particle integration, but the strict adversary observes
  raw `F(x)`, not velocity. It did not cause either numerical failure and cannot
  contain field-weight or target-gradient NaNs.

### Shipped fixes

- `train_wgsl.ts` now codegen-elides every disabled structural term. An all-zero
  field objective gets a no-op pass A; pass B does not scan undefined structural
  scratch and applies only explicit external gradients. Partial objectives
  independently omit chaos, isotropy, divergence and spiral arithmetic.
- The adjusted direction active floor is `1e-3` in tfjs, fused WGSL and the AD
  oracle. Hence the target Jacobian norm is bounded by `1e3`; samples below the
  floor are explicitly inactive with exact zero payoff/gradient.
- All five standalone adversary pieces now use field/G Adam LR `0.001`; D
  remains `0.003`. Chaos Weave retains `0.006` because it has an active composed
  structural objective. Comments and public config docs state the distinction.
- `tools/adversary_strict_test.ts` gates below-floor inactivity and
  just-above-floor finite tangent gradients in a common coordinate frame.
- `tools/field_loss_routing_test.ts` gates source-level term elision,
  external-only pass-B routing, and 2,000 saturated ZERO-loss Adam steps with
  exact zero loss/gradient/weight movement.

### Measured after-fix results

Both production-policy co-training gates completed 3,000 Metal steps with no
non-finite value:

- adjusted R+S, G LR `0.001`, D LR `0.003`: final field-weight
  `max=0.6527`, `rms=0.1913`; field-gradient `max=8.14e-4`; finite.
- raw R+S negative control, same LRs: final field-weight `max=0.9112`,
  `rms=0.2595`; field-gradient `max=2.21e-4`; finite.

The adjusted run still sampled `||q|| < 1e-3`, but those tuples were safely
inactive; its final p1 was `3.57e-3` and median `3.37e-1`. The raw control
reached exact-zero low-tail `q` values without destabilizing the structurally
empty field pass.

Verification:

- `bun tools/adversary_strict_test.ts` — PASS.
- `bun tools/ad_wta_test.ts` — PASS; emitted Metal WGSL remains oracle-identical.
- `bun tools/train_wta_test.ts` — PASS; all tfjs/IR/fused parity and lane gates.
- `bun tools/field_loss_routing_test.ts` — PASS, including all term-elision
  checks and the 2,000-step saturated ZERO soak.
- adjusted and raw 3,000-step `tools/adversary_stability_probe.ts` — both
  `STABILITY PROBE FINITE`.
- `npm run build` — PASS in 11.31 s.
- focused `git diff --check` — PASS.

### Remaining interpretation, not a hidden blocker

The adjusted quotient removes the instantaneous radial incentive; it does not
impose a preferred force magnitude. If long-run aesthetics still drift toward
the velocity cap, that is now a design choice to address with a separately
named force-RMS regularizer or optimizer constraint—not by claiming the angular
quotient supplies one. The raw R+S mode remains intentionally exposed as the
scale-cheating negative control. A generic non-finite/gradient-clipping last
defense may still be useful operational hardening, but is no longer required to
make these measured configurations finite.

## Inactive tuple win accounting (completed 2026-07-29)

The tfjs reference previously masked invalid triangle/labelled-quad rows to
exact zero payoff and gradient, then nevertheless ran `argMin([0,…,0])` and
credited the sentinel tie to head 0. This disagreed with fused semantics, whose
win counter is guarded by `targetActive`, and could manufacture a false
head-collapse diagnosis.

`src/core/gan/adversary.ts` now has one `tupleActivity` handler shared by payoff
masking and win telemetry. `trainStep` packs `(argmin, active)` into one
readback, counts only active rows, and documents that batch win totals are the
active tuple count (`<= B`). Directionless adjusted targets follow the same
rule. Invalid rows continue to participate in neither predictor nor generator
gradients.

Focused §8 gates in `tools/adversary_strict_test.ts` cover both triangle and
labelled quad:

- mixed active/inactive batch: active payoff/gradient remains nonzero, inactive
  payoff/gradient is exact zero, and exactly one—not two—winner is recorded;
- all-inactive batch: zero payoff/gradient, zero wins for every head, cumulative
  total zero, and no false collapse.

`bun tools/adversary_strict_test.ts` passes all sections. The broad
`tools/adversary_test.ts` progressed through sections 1–9 without failure but
was intentionally stopped during the slow head-spread section after more than
six minutes of shared-host contention; the root final suite will rerun it once
the parallel load clears.

## Race-free surprise coverage (completed 2026-07-29)

The fused live-particle sampler now uses a deterministic rotating window:

```text
index(s,t) = (cursor + s*m + t) mod N,
effectiveB = min(requestedB, floor(N/m)).
```

Thus every index written by one dispatch is unique and the non-atomic
per-particle surprise scatter is race-free. Coverage, cursor, generation and
the latest fresh window are tracked exactly. Resizing/rebinding clears the
buffer and resets the epoch; `N < m` is an explicit zero-data step that neither
advances Adam nor fabricates statistics. The surprise renderer samples only
the latest fresh window (including a two-copy wrapped window), deduplicates
generations and never folds untouched stale zeros into its percentiles.

Agree+Disagree keeps A and B on the same coverage schedule and reports the
minimum exact branch coverage. Generic surprise coloring is rejected for that
piece because one scalar channel cannot truthfully represent both A and B while
the UI promises exact RGB roles.

Focused verification:

- `bun tools/surprise_test.ts` — PASS, including wrapped fresh-window,
  generation-deduplication and resize-reset tests on real WebGPU.
- `bun tools/train_wta_test.ts` — PASS, including uniqueness, effective-batch,
  uploaded-source isolation, resize clear, `N < m`, and A/B lockstep gates.
- `bun tools/agree_disagree_live_test.ts` — PASS.
- `npm run build` — PASS.

## Fused labelled-quad observer (completed 2026-07-29)

The production adversary kernel now supports the original four-position
experiment as an explicitly **labelled** quotient:

```text
m = 4, du = 6, dy = 8
member 0 = anchor
member 1 = co-rotating-frame direction
u = labelled coordinates of members 1..3 in that frame
y = four mean-centered field vectors in that frame
```

Minimum-image torus geometry is used before frame construction. An anchor of
length `<= 1e-5` is inactive and produces exact zero target, payoff, wins,
predictor gradient and generator gradient. The generated handler is exhaustive;
a future tuple tag cannot accidentally fall through to triangle math.

This mode removes global translation and rotation only. Member labels remain
semantic, and the mode is deliberately neither permutation-invariant nor
scale-invariant. Those are limitations, not hidden claims.

Measured Metal gates:

- AD/IR context max delta `1.38e-7`, target `1.37e-7`, surprise `3.84e-7`;
  zero winner mismatches over 64 fixtures.
- Predictor and generator gradient cosine `1.0000000`.
- tfjs context max delta `2.38e-7`, target `1.79e-7`, field-gradient cosine
  `1.0000000`, scale-relative error `6.50e-7`.
- Torus, label-swap, degeneracy, coverage-lockstep, and Agree A/B lane gates
  all pass.
- Quad K=4, B=512 measured `3.262 ms/step`, versus the historical 19 ms tfjs
  target.

## Packed surprise-plane selection (completed 2026-07-29)

The surprise renderer and percentile feed now select an N-float plane from a
packed GPU buffer without allocating or copying a second buffer:

```text
render scalar(i) = surprise[surpriseOffset + i]
stats source      = surpriseOffset + rotating fresh-window index
```

The former uniform padding word is now the `u32` offset, so the render uniform
remains 32 bytes. `encodeRender(..., ts?, offsetFloats = 0)` and
`encodeSample(..., window, offsetFloats = 0)` retain the original offset-zero
behavior. Both halves of a wrapped stats copy add the plane offset. Changing
planes resets the robust-normalization history before generation
deduplication, preventing one branch's scale from leaking into another;
explicit `reset()` still clears the complete history as well.

Focused verification:

- `bun tools/surprise_test.ts` — PASS on real Metal. Offset 0 and offset N
  render distinct packed planes pixel-exactly (`Δ0/255` for both); a nonzero
  offset wrapped stats window excludes stale poison; duplicate generations,
  explicit reset, and automatic plane-history reset all pass.
- `npm run build` — PASS.

## Fused adversary diagnostics and two-plane producer (completed 2026-07-29)

The fused adversary now produces the data that the packed-plane renderer and
head-health UI require without changing the adversarial game.

### Finalized statistics

`ADV_STATS_BASE` moved from 16 to 32. The finalized prefix and every workgroup
partial now use this exact layout:

```text
[discLoss, rawPayoff, batchRms,
 meanPairwiseHeadSpread, meanClosestPairSpread,
 winCount_0 ... winCount_(K-1)]

partial stride = 5 + K
partial base   = 32
```

This fixes a real layout bug: the old base 16 overlapped finalized win slots
for legal `K=12..16`. The largest legal finalized prefix is `5+16=21`, leaving
the reserved interval `[21,32)` disjoint from partials. A real Metal K=16,
two-workgroup gate verifies all 120 head pairs, all 129 wins, and the untouched
reserved gap.

Head geometry is measured at every sampled context, including an inactive
target. For `K>1`, each context contributes the mean over all unordered
prediction pairs and its closest pair; the final stats average those two
values over contexts. The adjusted rotation+scale observer compares softly
normalized predictor directions, matching the tfjs diagnostic. `K=1` is
reported as the explicit `{tag:"single-head"}` case rather than a fabricated
zero spread. A controlled three-head fixture (two identical heads, one at
`(1,-1)`) measures closest spread exactly `0` while mean spread is
`2*sqrt(2)/3 = 0.942809`, pinning the distinction.

### Packed raw and per-unit planes

One allocation contains two planes with a host-visible stride in UAdv slot 13:

```text
plane 0: raw shared relaxed-WTA payoff
plane 1: rawPayoff / max(||y||, 1e-3)
```

Plane 1 is written only when the tuple target is active and `||y|| > 1e-3`;
otherwise it is exact zero. This explicit branch fixes the misleading
epsilon/epsilon frozen-tuple value. Both planes are always computed together;
the selection is visualization-only and cannot affect scratch, scalar stats,
predictor/field gradients, weights, or Adam state.

The host exports:

- `SurpriseMetric = "raw-payoff" | "per-unit-signal"`;
- `surprisePlane(metric) -> {buffer, offsetFloats}`;
- `readSurprise(n, metric)` for blocking verification;
- `surpriseBuf` and default `readSurprise(n)` as raw-payoff compatibility
  aliases.

Resize/rebind clears both planes and retains one exact shared coverage epoch.
A linear controlled fixture scales the field target by 5: raw payoff changes
by exactly `5.000000x`, while the per-unit plane changes by `0.00e+0`. A zero
adjusted target writes exact zero to both planes while still reporting
nonzero predictor head geometry, proving target validity does not mask the
head probe.

### Verification

- `bun tools/train_wta_test.ts` — **ALL FUSED WTA CHECKS PASS**. Every
  point/pair/tri/labelled-quad, periodic/Euclidean and raw/Fourier oracle gate
  retains discriminator and generator gradient cosine `1.0000000`.
  Packed-plane max deltas were `0` for raw/compat and at most `1.68e-5` for
  per-unit CPU parity across the exhaustive configurations. Head mean/closest
  relative errors were below `2e-5`, including normalized adjusted heads.
- Real Metal B=512 measurements in this run: pair K=4 `0.844 ms`, tri K=6
  `1.006 ms`, labelled quad K=4 `0.785 ms`; all remain far below the historical
  19 ms tfjs target.
- `bun tools/agree_disagree_live_test.ts` — PASS.
- `bun tools/border_modes_test.ts` — PASS.
- `npm run build` — PASS.

Intentional limits remain explicit: head spread separates predictor pileup
from geometric separation but does not prove separated heads are calibrated
on conditional support; labelled quad remains label-sensitive and
scale-sensitive; the per-unit plane is a rendering diagnostic, never a second
training objective.

## Final current-source release verification (completed 2026-07-29)

The complete AD, fused-trainer, adversary, border, drive, Agree+Disagree,
renderer, splat, feature-painter, optimizer, integration, and production-build
matrix passed on Apple Metal. Both adjusted and raw rotation+scale observers
also completed 3,000-step stability probes with finite weights and gradients.

Real-browser testing found one issue that the kernel suites could not see:
`physicsForward` consumed a free `forceMagnitude` identifier because its live
final argument had been omitted from the declaration. Parcel does not
type-check this signature, and the fused renderer continued drawing, masking
the tfjs reference-training exception. The parameter was restored with
`cfg.forceMagnitude` as its default. `tools/adversary_wire_test.ts` now calls
the production function with 1x and 3x magnitudes and proves the physical force
scales exactly 3x. Unsampled fused loss now displays `warming`, not the
misleading text `NaN`.

The index now has a real mobile viewport declaration and the title `Neural
Force-Field Art`; without the viewport declaration, phone browsers would lay
the supposedly responsive dock out against a desktop-width viewport.

The Playwright release script at `output/playwright/release_qa.js` passed:

- all seven adversarial pieces selected and trained on the fused WebGPU path;
- all three pair observers, all three borders, K rebuild, max velocity,
  dimensionless drive, RAW and PER UNIT diagnostics;
- independent Agree/Disagree roles and A/B health, with C display-only;
- desktop and 390x844 layout with no horizontal overflow or dock/gallery
  overlap;
- honest tfjs fallback to velocity diagnostics;
- zero page or console errors.

The 30-second-per-piece soak under
`output/playwright/adversary-soak/2026-07-28T17-38-54-170Z/` passed every gate.
All modes held stable tensor counts, finite advancing telemetry and no
confirmed pileup. Diagnostic pieces reached 100% coverage with non-flat raw and
per-unit spans. Median FPS was 60.0 throughout; trailing mean-speed/clip ranged
from 0.328 to 0.442, directly falsifying the earlier “always pins max velocity”
failure under the new drive policy.

## Final-audit reference-state removal

The read-only final audit caught one unfinished earlier concern: supported
fused pieces still constructed a complete unused tfjs `Adversary`, including
K predictor heads and their Adam slots. That explained the K-dependent idle
tensor baselines: 23 for K=1, 47 for K=4, 63 for K=6 and 79 for K=8.

`AdversaryRuntime` is now a discriminated union over `off`, `tfjs`, and
`fused`. The fused variant contains only canonical game metadata and cannot
hold a tfjs predictor. Startup decides fused eligibility from the spec/layout
before constructing the runtime; only the explicit fallback builds the
reference game. The tfjs field optimizer is also delayed until the code knows
no `FusedTrainer` exists.

Verified:

- `tools/adversary_wire_test.ts` proves ordinary and Agree+Disagree fused
  runtime construction adds exactly zero tfjs tensors; the fallback adds 32
  predictor tensors and returns to baseline on disposal.
- Current-browser traversal of all seven fused pieces reports the same 13
  blueprint tensors, independent of K, with zero page/console errors.
- `?train=tfjs` Pair remains finite and honestly labelled, with 119 tensors.
- `tools/agree_disagree_live_test.ts` and `npm run build` pass after the
  refactor.
- A post-refactor smoke kept every piece finite with stable 13-tensor counts.
  The intentionally short WTA-8 run needed a separate 12-second coverage pass;
  it reached 100% at
  `output/playwright/adversary-soak/2026-07-28T18-02-23-027Z/`.

## Final terminology and artifact audit

A final repository-wide stale-claim search found two remaining pre-refactor
descriptions outside the main release docs. `docs/DESIGN_SPACE_PARTICLE_ART.md`
still called alpha an order/chaos axis and called the class-routed second
vector head a chaos head; `tools/kernel_test.ts` repeated that head identity in
a fixture comment. Both now describe neutral A/B direct-vector heads and state
that semantics come only from an explicit loss or game.

`output/playwright/release_qa_result.json` now distinguishes the full release
QA measurements from the later current-source tensor traversal. This avoids
implying that fields gathered in two verification phases came from one browser
sample. `git diff --check`, JSON parsing, QA-script syntax checking and live
HTTP availability were clean at the end of the audit.

The independent final audit then found one functional URL/UI mismatch:
piece-independent `?adv=wta` still inherited the obsolete pre-normalization
reward weight `0.35`, while the live slider represents `[0, 0.05]`. The shared
canonical range is now exported as `ADVERSARY_WEIGHT_RANGE`; URL ingestion
clamps to it, the UI consumes the same constants, and the neutral forced-on
default is `0.012`. The wiring suite pins both the default and high/low URL
clamps. Pair/Tri gallery comments now say the strict target is raw `F(x)`, not
particle displacement, and the AD rollout describes neutral A/B vector heads
rather than intrinsic order/chaos lanes.
