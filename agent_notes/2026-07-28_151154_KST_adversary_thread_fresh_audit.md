# Adversary thread: fresh audit and question ledger

Date: 2026-07-28 15:11:54 KST  
Scope: reconstruct the entire adversary discussion, verify it against the
current dirty worktree, identify mathematical and implementation concerns, and
leave a resumable record. This note distinguishes observed code, measured
results, theoretical claims, and proposals. It must not treat prior agent
summaries as proof.

## Working rules for this audit

- Preserve the user's existing splat and adversary changes.
- Do not infer completion from a workflow notification; inspect files and run
  gates.
- Cite source locations for every material conclusion.
- Mark claims as `VERIFIED`, `PARTIAL`, `UNPROVEN`, `INCORRECT`, or `PROPOSED`.
- Separate a mathematical equilibrium claim from a finite-time optimizer
  observation.
- Do not describe a test oracle as a production compiler.
- Do not call a path "fused" merely because its shader exists; verify routing,
  dispatch, buffers, and UI selection.

## Complete question and discovery ledger from the thread

This is the compact chronological record of every substantive question,
discovery, correction, concern, and requested feature raised in the resumed
conversation.

### Repository and runtime

1. Which repository is newer: `force-field-ml-art` or
   `Neural-Force-Field-Art`?
2. Can the current repository's server be run so the art can be viewed?
3. Which gallery URLs exist (`index.html`, `splat.html`, `splat3d.html`) and do
   they render rather than merely return HTTP 200?
4. Are the new art modes locally viewable, publicly deployed, or both?
5. Are all new modes selectable in the gallery and overridable by URL/UI
   controls?

### Original "surprise" objective

6. What did the existing `chaosLoss` measure?
7. Is one-step spatial sensitivity a genuine prediction-surprise objective?
8. Could high-frequency spatial noise maximize the chaos proxy without
   producing interesting motion?
9. Did a predictor ensemble already exist, and was it actually called?
10. Was ensemble disagreement intended to distinguish epistemic uncertainty
    from aleatoric noise?
11. Does absolute-position input make the target a deterministic realizable
    function and therefore make ensemble disagreement vanish?
12. Is raw prediction distance a noisy-TV objective whose optimum is random
    motion?
13. Would BALD / learned aleatoric uncertainty add value in this deterministic
    field?
14. Would compression progress be a better controller than surprise level?
15. Could a replay-buffer discriminator reward motion unlike the piece's own
    history?
16. Would k-step, no-reset rollout surprise produce trajectory choreography
    instead of one-frame jitter?
17. How should surprise compose with divergence, isotropy, spiral, and
    Helmholtz chaos constraints?

### Position tuples and WTA

18. Should the adversary consume position pairs or small tuples rather than the
    whole particle array?
19. Should tuples be sampled as `B × m` (initially phrased as four points per
    tuple and B tuples per batch)?
20. Does merely grouping four independently predictable point targets change
    anything, or must the encoding discard pose information so
    `P(Y|U)` becomes one-to-many?
21. How should pair, triple, and four-point tuples be canonicalized under
    translations, rotations, reflections, permutations, and torus seams?
22. Is a pair's only SE(2) invariant its separation, making pair context just
    one-dimensional?
23. Does `m=3` provide a richer invariant through triangle side lengths and
    angles?
24. Is `m=4` actually impossible/unsafe to canonicalize, or was it prematurely
    rejected?
25. What are the exact permutation and PCA sign/reflection ambiguities?
26. What is the mathematical condition for residual disagreement to remain
    positive at equilibrium?
27. What should the single adversary predict and what makes it a useful
    control?
28. What changes when the adversary emits K guesses and the loss uses the
    nearest guess?
29. Is min-over-K conditional K-means, K-medians, or another quantization
    objective under the chosen norm?
30. Does a one-head norm loss converge to the conditional mean?
31. Correction raised in-thread: with the unsquared norm it converges to a
    geometric-median set; for two equal point masses every point on the
    connecting segment is optimal.
32. Does relaxed WTA prevent dead heads, and what exactly does it guarantee?
33. Is skewed head usage optimizer collapse, or simply `K > number of modes`?
34. Can pairwise head-output spread distinguish benign surplus heads from
    optimizer pileup?
35. Should K be reduced for pair encodings and increased only when the
    conditional supports it?

### Scale, velocity, and long-run dynamics

36. Can raw displacement residual be increased just by making particles move
    faster?
37. Does the max-velocity clip become a cheat code / attractor for the
    generator?
38. Does normalized reward using a stop-gradient EMA of RMS displacement
    actually close that scale exploit?
39. Is normalization global enough, or should surprise also be normalized per
    tuple / per unit motion?
40. Should the color renderer offer "velocity-adjusted surprise" so the
    colormap is not merely a speed proxy?
41. Does the art begin interesting and degrade after velocities pin at the
    clip?
42. Was the claimed reward-on/off speed A/B rigorous enough to rule out the
    ratchet?
43. Should `maxVelocity` be a live slider and URL parameter?
44. Are max velocity, physics rollout, adversary transition, trainer backward,
    and HUD all updated consistently when that slider moves?
45. Does the adversary see one step even when the field trainer uses a
    multi-step rollout?
46. Is the true co-training equilibrium measured, or only one-step gradient
    sign and short browser soaks?
47. Does the single control's reward "die," and is that compatible with a GAN
    struggle?
48. More precise question: in a two-player game with comparable learning rates
    and capacity, why should either player's scalar loss stay "even" rather
    than track the changing Bayes risk?
49. Does the point/single control become exactly predictable because the
    deterministic field is a function of the same absolute position input?
50. Does online simultaneous training leave a transient nonzero residual even
    if the population optimum is zero?

### Helmholtz and other fields

51. Are all adversarial gallery pieces Helmholtz fields?
52. What exactly is a Helmholtz field in this codebase?
53. How does a learned scalar potential/curl-potential decomposition compare
    with the ordinary MLP neural vector fields?
54. What does the order/chaos alpha control blend?
55. Is alpha an architecture control or a loss weight?
56. Which pieces use only adversarial reward plus weak isotropy/divergence
    anchors?
57. Which piece mixes in the full `helmholtzChaosLoss`?
58. What is "Chaos Weave," mathematically and in gallery configuration?
59. Are the standalone anchor terms strong enough to prevent the noisy-TV
    optimum?
60. Is there a true adversary-only piece without anchors?

### Borders and resets

61. What currently happens when particles cross borders: wrap, bounce, reset,
    or a mixture?
62. Is the domain a flat torus under wrap?
63. Is the random-reset slider independent of border behavior?
64. Do respawn teleports get excluded from adversary targets and surprise
    statistics?
65. Does minimum-image displacement apply only under periodic wrap?
66. Can border mode be toggled among wrap, bounce, and reset?
67. Are border Jacobians correct in field rollout backward and adversary
    backward?
68. If bounce or border reset is partially implemented in WGSL, is it wired
    into gallery state and the React UI?

### AD IR, fused kernels, and performance

69. Is `src/render/webgpu/ad/` a production AD compiler or an independent
    oracle?
70. Does production use hand-written WGSL, literal handwritten shader text, or
    TypeScript code generation with handwritten analytic derivatives?
71. Are layer and activation derivatives analytically coded, with
    post-activation formulas and SIREN preactivation checkpoints?
72. Why does the chain rule exist in a math document, the production WGSL
    generator, and the AD oracle?
73. Would making the AD IR production-load-bearing require more code?
74. Would it improve speed, or primarily reduce duplicated derivative logic?
75. Was emitted WGSL ever compiled and dispatched on a real GPU?
76. Was AD-emitted WGSL ever benchmarked against the hand-fused trainer?
77. Was the claimed 25–40× number actually fused-vs-tfjs rather than
    AD-generated-vs-handwritten WGSL?
78. What was the `f32lit` scientific-notation bug and does the new guard cover
    the full emitted-shader path?
79. Is min-over-K expressible in the scalar AD IR?
80. Is K folded within one thread while B is the dispatch/thread dimension?
81. Are reductions, predictor weights, Adam state, tuple sampling, generator
    field gradients, and surprise output all fused?
82. Are adversary and field weights provably separated?
83. Does the fused kernel match the AD oracle and tfjs reference, including
    generator gradients and winners?
84. Which field types force fallback to tfjs?
85. Is fused head-spread telemetry implemented or explicitly unresolved?

### Verification, process, and completion

86. Which workflow phases actually completed, which died from 529/session
    limits, and which left usable partial edits?
87. Was work resumed from cached results or redundantly re-derived?
88. Did any workflow claim a document path even though no document existed?
89. Did agents write source before being killed at report time?
90. Are build/test/performance claims reproducible from repository tools?
91. Were all Metal suites rerun after app wiring?
92. Are long tests in a handoff checklist?
93. Are there stale comments, generated directories, lockfiles, QA scripts, or
    screenshots that need cleanup?
94. Is the worktree a clean, reviewable change, or does it mix unrelated splat
    work with adversary work?
95. Was `PLAN_RELATIONAL_ADVERSARY.md` eventually written and does it describe
    shipped facts separately from proposals?
96. Are all original goals now complete: mathematical plan, single and K-guess
    adversaries, tuple modes, fast fused shaders, UI swappability, surprise
    visualization, and rigorous tests?
97. What remained in flight when the latest control-knobs agent was killed?
98. Did that killed agent partially implement border modes, live max velocity,
    or normalized-surprise rendering without completing routing/tests?

## Facts established before the fresh audit

- `docs/PLAN_RELATIONAL_ADVERSARY.md` now exists, despite earlier workflow
  failures. Its claims still require source/test verification.
- The worktree is heavily dirty and mixes adversary changes with pre-existing
  splat changes.
- The latest killed controls agent left border-related edits in
  `advect_wgsl.ts`, `advect.ts`, `train_wgsl.ts`, and likely other files.
  Whether they are coherent and reachable from the UI is not yet established.
- `CLAUDE.md` did not exist at audit start.
- `AGENTS.md` described the runtime but did not require timestamped agent notes.

## Fresh-audit findings

Direct verification performed so far:

- `npm run build` — PASS.
- `bun tools/ad_wta_test.ts` — PASS, including emitted WGSL on Metal.
- `bun tools/train_wta_test.ts` — PASS for the currently shipped wrap/default
  configuration; pair-k4 0.830 ms and tri-k6 0.852 ms in this run.
- `bun tools/adversary_wire_test.ts` — PASS.
- `bun tools/surprise_test.ts` — PASS.
- `bun tools/train_types_test.ts` — PASS on five fixtures.
- `bun tools/kernel_test.ts` — PASS; current run measured fused f32 advect
  4.747 ms/step at 1M particles.
- `bun tools/adversary_test.ts` — PASS. Important caveat: its generator sign
  check increased surprise from `7.2848e-3` to `9.0366e-1` (about 124×) by
  optimizing free next positions. That satisfies its one-sided `>1.05×`
  assertion but is also direct evidence that the check does not distinguish
  structured surprise from a magnitude blow-up.

### Mathematical findings

1. **INCORRECT — the live point/single target is not a deterministic function
   of position.** The fused adversary gathers both position and persistent
   incoming velocity, then constructs
   `y = clamp((v + F(x)) * friction) / resolution`
   (`src/render/webgpu/adversary_wgsl.ts:547-588`), while the point context is
   position only (`adversary_wgsl.ts:317-324`). Therefore hidden velocity makes
   `P(Y|X=x)` non-degenerate in general. The exact-realizability ladder in
   `tools/adversary_test.ts:521-559` explicitly uses the easier zero-velocity
   target `next = x + F(x)`. It does not prove that the shipped fused control's
   residual must die.
2. **UNPROVEN — predictor realizability.** Even without velocity, the adversary
   head and the two-head field use different architectures/capacities. “The
   field is a function of x” does not imply the smaller predictor class can fit
   it exactly.
3. **VERIFIED — the WTA implementation is a conditional K-median-style
   quantizer, not K-means.** Residuals use the softened Euclidean norm, not its
   square (`src/core/gan/adversary.ts:855-866`). The single-head population
   optimizer approaches a geometric median. **Precision correction:** because
   the implemented residual is `sqrt(||e||² + 1e-12)`, the two-point objective
   is not exactly flat along the segment; it has a unique midpoint minimizer.
   The softening is tiny, so observed near-flatness is reasonable, but the
   repeated “exactly flat” and “exactly half separation” statements are not
   literally true for this code.
4. **VERIFIED — relaxed WTA gives every head gradient, but does not guarantee
   balanced wins or distinct modes.** Winner/loser weights are held constant
   on the tape; the oracle and fused tests agree. On a unimodal conditional all
   heads may legitimately converge together.
5. **PARTIAL — global RMS normalization reduces scale sensitivity but does not
   prove that velocity cannot ratchet.** It is a lagged, stop-gradient EMA, so
   each instantaneous generator step still has a radial incentive; only later
   EMA updates cancel steady uniform scaling. The current §8 test compares
   objective values after separately training on 1×/10× data and accepts a wide
   0.5–2 ratio. It does not test online generator gradients, saturation
   occupancy, or adversary lag.
6. **INCORRECT / DOUBLE NORMALIZATION — the full generator term divides by
   both RMS displacement and `maxVelocity/min(width,height)`.**
   `Adversary.generatorLoss` already divides by EMA RMS‖y‖
   (`adversary.ts:946-951`), and `adversaryGeneratorTerm` divides it again by
   `stepScale` (`main.ts:710-742`). The fused `genSeed` mirrors this
   (`adversary_train.ts:370-377`). After the RMS change, the old step-scale
   rationale and “dimensionless” comments are stale; the second division
   reintroduces resolution and max-velocity dependence into reward strength.
7. **UNPROVEN — noisy-TV immunity.** Normalization removes a pure magnitude
   scaling reward, not the incentive to create conditionally high-entropy,
   visually noisy dynamics. The code itself acknowledges this and relies on
   aesthetic loss composition, which is currently misrouted (runtime finding
   2 below).
8. **PARTIAL — tuple quotienting can create a one-to-many conditional, but
   irreducibility is generator-dependent.** For a generic absolute-position
   field, discarding pose creates multiple targets per invariant context. A
   specially equivariant field could reduce that ambiguity; positive residual
   is an empirical property of the current field/distribution, not a theorem
   for every generator.
9. **INCORRECT OVERCLAIM — m=4 is not mathematically impossible.** The documented
   argument (`adversary.ts:538-554`) shows that one vertex-sort
   canonicalization is discontinuous/ambiguous. It does not rule out
   permutation-invariant set/graph encoders, an all-six-distances context
   modulo S4, equivariant targets, or assignment-invariant output losses.
   For m=4, enumerating all 24 permutations of the full pairwise-distance
   matrix and selecting the lexicographically minimal labeling is already a
   constructive small-m alternative; equal minimizers are metric
   automorphisms.
10. **BUG — tri canonicalization is not permutation-invariant on exact
    isosceles ties.** The source admits label-based tie resolution
    (`adversary.ts:484-494`) while stronger comments/docs call the construction
    safe. An independent audit reproduced a material target change when the two
    tied labels were swapped. The existing six-permutation test uses a scalene
    triangle (`tools/adversary_test.ts:883-886`), so it cannot catch this.
11. **SCOPE — “true gradient” means a truncated one-step gradient.** Holding
    tuple positions/velocities as data and differentiating through current
    `y` is correct for the one-step objective, but it omits how earlier field
    weights created the current position/velocity distribution. It is not a
    full long-horizon game gradient.
12. **MISLEADING EDGE CASE — the partially added
    `surprisePerUnitMotion()` returns 1 for a perfectly predicted frozen tuple**
    because both softened residual and denominator equal epsilon
    (`adversary.ts:888-917`). That may be numerically defined, but “100% of a
    zero step was surprising” is not a useful visualization semantics.
13. **OVERCLAIM — head spread does not prove `K > modes`.** Large pairwise
    head distance can also mean losing heads diverged or sit far off-support.
    Recorded spreads of 38–104× RMS target motion are not obvious evidence of
    healthy mode coverage. Add per-head residual and support/calibration
    metrics plus a third `off-support/diverged` state.
14. **DOMAIN ISSUE — relaxed-WTA validation permits winner inversion.** For
    `epsilon > (K-1)/K`, each loser receives more weight than the winner. The
    shipped 0.05 is safe; the public config bound is not if winner dominance is
    intended.
15. **OBJECTIVE LANGUAGE — quantization distortion is not entropy.** High
    conditional K-quantization error can correlate with broad/noisy targets,
    but it is not mathematically equivalent to maximizing conditional entropy.

### Runtime, boundary, and UI findings

1. **VERIFIED — every named adversary gallery preset currently constructs a
   `HelmholtzField`** (`main.ts:1395-1571`). URL overrides can attach an
   adversary to other pieces, but the five shipped adversary entries are all
   Helmholtz-named fields.
2. **CRITICAL BUG — standalone gallery loss composition is ignored on the
   default fused path.** Single, point-WTA, pair-WTA, and tri-WTA declare
   `adversaryAnchorLoss()` (isotropy 0.25 + divergence 0.15;
   `main.ts:1163-1189,1422,1449,1494,1535`). `FusedTrainer` is constructed
   without any loss spec (`main.ts:2403-2460`), and `train_wgsl.ts:73-81`
   hardcodes the full chaos 1 + isotropy 1 + divergence 0.5 + spiral 2e-5 loss.
   Therefore all four “standalone” adversary pieces run the full Helmholtz chaos
   objective by default. Only `?train=tfjs` honors their declared anchor. The
   prior answer “only Chaos Weave includes Helmholtz chaos” was false for the
   live fused runtime.
3. **SEMANTIC A/B BUG — `?train=tfjs` changes more than implementation.** It
   changes the field loss (finding 2) and data distribution: the tfjs
   discriminator/generator uses fresh random positions with zero velocity
   (`main.ts:2045-2062`), while the fused adversary uses live particles and
   persistent velocities (`adversary_wgsl.ts:547-588`). It is not currently a
   fair implementation-only comparison. It also changes game ordering: tfjs
   updates D and then recomputes G against updated D, whereas fused computes
   and stores both D and G deltas from pre-update adversary weights before
   applying either update. Fused is simultaneous-gradient play, contrary to
   comments describing an up-to-date opponent.
4. **INCORRECT NAMING — `HelmholtzField` is not a Helmholtz decomposition.**
   It contains two unconstrained direct-vector MLPs blended by alpha
   (`src/core/field/helmholtz.ts:1-17,192-207`). The file explicitly says it
   abandoned gradient/curl scalar potentials (`helmholtz.ts:20-37`). Worse,
   the loss constrains only their blend, so there is no separate objective that
   makes one learned head “order” and the other “chaos.” The alpha bar is a
   blend between two learned vector heads, not a guaranteed
   curl-free↔divergence-free control.
5. **VERIFIED CURRENT BOUNDARY — runtime is wrap plus independent random
   respawn.** Default advect performs torus wrap, then applies the live
   `resetRate` probability as a uniform zero-velocity respawn. These are
   separate mechanisms.
6. **PARTIAL/UNSAFE — the killed control-knobs agent left border code but no
   complete feature.** `BorderMode`, forward emitters, and a Jacobian helper
   exist in `advect_wgsl.ts:88-198`; `AdvectKernel` accepts a codegen-time
   border. But `ArtPieceConfig`, `LoopHandle`, React controls, URL parsing, and
   runtime reconstruction do not expose it. `FusedTrainer` does not receive the
   border option, its backward still assumes `dpos/dq = +1`
   (`train_wgsl.ts:880-903`), and the fused adversary remains explicitly
   wrap/minimum-image-specific. Bounce/reset must not be exposed yet.
7. **PARTIAL — live max velocity is not a user feature.**
   `AdvectKernel.setMaxVelocity()` was added (`advect.ts:401-419`), but there is
   no LoopHandle method, React slider, or URL parameter. The loop continues to
   pass static `cfg.maxVelocity` into adversary, trainer, HUD, renderer
   normalization, and teleport thresholds.
8. **PARTIAL — velocity-adjusted surprise is not reachable.** A tfjs-only
   `surprisePerUnitMotion()` method exists, but has zero callers. The fused
   shader, GPU buffer, color mode, URL parser, and UI have no normalized
   surprise variant.
9. **BUG — fused surprise rendering has unreported stale coverage and data
   races.** Each tuple thread writes `surprise[midx[t]] = sur`
   (`adversary_wgsl.ts:615`); randomly sampled indices can repeat, so concurrent
   non-atomic writes can race. Only B·m particles are touched per adversary
   step, and other values remain old/uninitialized. Stats read a fixed prefix
   (`surprise_points.ts:190-213`), yet the UI reports `covered: 1`
   unconditionally on the fused path (`main.ts:2656-2665`).
10. **STALE DEFAULT — adversary pieces still default to training every second
    frame because an old comment assumes tfjs cost** (`main.ts:1740-1753`).
    The later fused-path comment says no amortization is needed, but
    `trainEvery` gates both adversary and field updates. This changes game
    dynamics and leaves fused performance unused.
11. **BROKEN URL COMBINATION — class-aware Species + adversary has no valid
    fallback.** The fused adversary rejects classes, then the tfjs field path
    throws because it has no class input. Configuration should reject this
    before runtime.
12. **DOC SUPPORT-MATRIX DRIFT — SIREN is fused-adversary eligible.** Current
    selection rejects class-aware and hashgrid fields, not SIREN; contrary docs
    are stale.

### Fused implementation and verification findings

1. **VERIFIED — the WTA production path is genuinely fused WebGPU code, not
   tfjs.** `AdversaryTrainer.encodeStep()` records four passes:
   forward/reduction, finalize, adversary Adam, and generator field-gradient
   (`adversary_train.ts:380-456`). It owns separate adversary weights/moments
   and emits external field gradients.
2. **VERIFIED — current oracle/parity gates are strong for the math they cover.**
   WTA AD reverse mode matches finite differences; emitted AD WGSL compiles and
   runs on Metal; the hand production kernel matches the IR/tfjs fixtures at
   cosine 1.0; adversary and field buffers remain separate.
3. **IMPORTANT LIMIT — those gates verify matching implementations, not the
   correctness of the game specification.** They do not catch the hidden
   velocity premise, gallery loss misrouting, double normalization, or stale
   surprise coverage because all compared implementations/fixtures share or
   omit those semantics.
4. **NO VALID AD-vs-hand speed comparison exists.** The AD-emitted test reports
   submit→map time while emitting thousands of lines and tens of thousands of
   gradient outputs; the production hand-fused trainer performs reductions and
   updates with a different interface. The measured 19–32 ms versus ~0.8 ms
   result is fused WebGPU versus tfjs, not generated-AD WGSL versus hand-coded
   WGSL.
5. **PARTIAL BORDER WORK IS UNTESTED.** Current tests exercise default wrap.
   No forward/backward parity matrix covers wrap/bounce/border-reset across
   advect, field rollout, adversary transition, target encoding, teleport
   filtering, and renderer statistics.
6. **FALLBACKS:** fused adversary selection excludes class-aware and hashgrid
   fields; `?train=tfjs` also chooses tfjs. Standard, SIREN, and Fourier
   classless fields are eligible. This is a supported fallback, not “all field
   types fused” for the adversary.
7. **PERFORMANCE CLAIMS NEED QUALIFICATION.** Offline fused step timings
   (~0.8 ms) are reproducible and strong, but they are not whole-frame timings.
   Fresh current-source browser QA observed one Pair run around 40.8 FPS, so the
   documentation's blanket “all five at 60 FPS” claim is not established across
   live configurations/hardware/load.

### Completion matrix

| Goal | Fresh verdict |
|---|---|
| Single and K-guess WTA reference | Implemented and numerically tested |
| Pair and tri tuple encodings | Implemented; tri exact-tie bug remains |
| Original four-point tuple request | Not implemented; rejection rationale overclaims |
| Fused WTA kernels | Implemented and fast on covered configurations |
| AD IR as correctness oracle | Implemented; not a production compiler |
| Gallery/UI switching | Implemented for adversary kind/weight/color; no border/maxVel/per-unit-surprise controls |
| Surprise visualization | Implemented, but fused coverage/race semantics are incorrect |
| Standalone adversary loss composition | Incorrect on fused path |
| Clean single control | Not established on live fused data |
| Velocity exploit closed | Partially mitigated, not rigorously closed; double normalization remains |
| Border wrap/bounce/reset toggle | Partial shader edits only; unsafe/unwired |
| Max-velocity slider | Partial kernel setter only; unwired |
| Velocity-adjusted surprise toggle | Partial tfjs helper only; unwired |
| Math plan | Exists, but contains invalid/overstated claims above |
| Public URL/deployment | Not part of this audit; local build works |

### Repository/process findings

1. `dist_wire/` is an untracked ~397 MB build artifact. A pre-existing Python
   server on port 8799 serves it, so opening that port shows stale code rather
   than the current source. `tools/qa_adversary.mjs` defaults to that port and
   can therefore produce misleading failures/results.
2. The dirty tree mixes roughly 3,500 tracked-line changes in adversary/runtime
   work with older splat work, plus many untracked source/test/doc files. It is
   not reviewable as one conceptual commit.
3. `docs/ADVERSARY_STATUS.md` is stale in material places: it claims raw
   surprise renderer settings that differ from gallery code, retains the
   already-fixed old slider warning, calls the short speed A/B proof of “no
   ratchet,” and has incomplete placeholder soak/perf tables.
4. Several source comments still describe the pre-fused state, including the
   claim that adversary pieces are forced to tfjs and the point target is
   exactly `F(x)`.

## Corrected answers to the user's live questions

### Are all adversarial modes Helmholtz?

All five named adversary gallery entries instantiate the class called
`HelmholtzField`. However, the name is misleading: the current class does not
implement the mathematical Helmholtz decomposition. It is two arbitrary
vector-output neural networks blended by alpha. URL overrides can attach an
adversary to other gallery pieces, but at least one such combination
(class-aware Species + adversary) is currently broken because it misses fused
eligibility and the tfjs field path throws.

### What is the Helmholtz field compared with the other neural fields?

- Legacy MLP pieces: one sigmoid vector MLP, output recentered by `-0.5`.
- Current `HelmholtzField`: two signed-vector heads, usually SELU→tanh, blended
  as `(1-alpha)g + alpha r`; optional SIREN, Fourier, or learned-grid encoding.
- Historical/abandoned true decomposition: scalar potentials differentiated as
  gradient and rotated gradient, which required second-order autograd during
  weight training and was too slow.

The present `g` and `r` heads have no separate curl/divergence constraints.
Calling alpha “order↔chaos” is therefore a visual hypothesis, not a property.

### Is Single adversary-only or mixed with the chaos/order loss?

The intended gallery declaration is weak anchors plus adversary reward. The
actual default fused runtime uses the full chaos + isotropy + divergence +
spiral objective because the fused trainer hardcodes that loss. Therefore the
live Single piece is mixed with full chaos. This is a bug, not a design choice.

### Why is the control expected to die, and should equal learning rates balance
the game?

Equal learning rates/capacities do not imply equal scalar losses in a GAN or
minimax game. The discriminator's optimum is determined by its conditional
Bayes risk, while both distributions and gradients move over time.

The intended control argument was: if input is x, target is deterministic
`F(x)`, and the predictor can represent F, its residual tends to zero. The live
fused control violates the premise because the target also contains hidden
incoming velocity. A clean control must either:

1. feed `(x,v)` to the predictor;
2. use zero-velocity samples consistently on both fused and tfjs paths; or
3. predict force `F(x)` rather than the full state transition.

Then freeze the field and demonstrate discriminator convergence before using
the control to interpret co-training.

### What happens at borders now?

The shipped behavior is:

1. integrate;
2. component-wise velocity clamp;
3. torus wrap;
4. independent random reset with probability `resetRate`.

The reset slider already controls step 4. Bounce and border-triggered reset are
partially written but are not safe or reachable. Their field-trainer backward,
adversary geometry, teleport filtering, UI, URL parsing, and parity tests are
unfinished.

### Is max velocity a cheat, and is normalization enough?

The concern is valid. The present EMA normalizer does not remove the
instantaneous radial gradient because the denominator is stop-gradient. If
`y=c y0` and residual scales with c, the numerical normalized value may stay
similar while the gradient still pushes c upward. The objective is also divided
again by a max-velocity-derived step scale, so changing max velocity changes
adversary strength.

The recorded reward-on/off mean-speed A/B is useful but insufficient: it did
not measure clamp occupancy, distribution tails, direction/structure, or a long
transient. It also ran under the accidentally hardcoded full-chaos field loss.

The right experiment is a live max-velocity control plus time series for:

- mean and P50/P90/P99 speed;
- fraction of x/y components at the clamp;
- raw and per-unit-motion residual;
- aesthetic loss terms;
- reward on/off, with identical initial weights/seeds.

### Can max velocity, borders, and velocity-adjusted surprise be toggles?

Yes, but they are not complete now.

- `maxVelocity`: update one live physics state used by advect, field trainer,
  adversary, tfjs reference, HUD, renderer color scale, teleport threshold, and
  reward scaling. The current partial setter updates only advect.
- `border`: because it is code-generated, changing it currently requires
  rebuilding the relevant kernels. All three forward paths and both backwards
  must share the same mode. Do not expose until that parity matrix passes.
- `velocity-adjusted surprise`: implement the same named metric in tfjs and
  WGSL and make the renderer select it. Avoid the current zero-motion
  `epsilon/epsilon = 1` semantic; define a scale floor or an explicit
  zero-motion policy.

### Is fused WTA the AD compiler?

No. Production WTA is hand-authored TypeScript→WGSL code generation with
analytic backward rules. The AD IR is an independent scalar graph/oracle used
for finite-difference and emitted-WGSL correctness checks. One AD-emitted WTA
shader now runs on Metal in tests, but production does not compile the training
kernel from that IR.

### How much faster is AD IR than the hand-fused kernel?

Unknown. No apples-to-apples benchmark exists. The valid measured speedup is
hand-fused WebGPU adversary versus tfjs (roughly tens of times faster in the
recorded tests). The AD-emitted GPU test has different outputs, work, readback,
and batch size, so its timing cannot be compared to the production trainer.

## Concerns to resolve before further feature work

1. **Partial control-knob edits:** border/max-velocity code may have landed in
   only some physics paths. A half-routed physics feature is more dangerous
   than an absent one because forward/backward parity can silently diverge.
2. **Equilibrium language:** "single reward dies" is a statement about Bayes
   risk under a realizable deterministic conditional, not a general rule that
   equal learning rates make two losses balance.
3. **Velocity evidence:** a short reward-on/off mean-speed comparison does not
   by itself establish that long-run aesthetics are not dominated by the clip.
   Saturation rate, component-wise clamp occupancy, speed distribution, and
   time series matter.
4. **Noisy-TV objective:** normalized WTA still rewards conditional
   unpredictability. Scale normalization closes a magnitude exploit but does
   not make noise aesthetically meaningful.
5. **m=4 rejection:** the current document makes a strong impossibility-style
   claim. It must be checked carefully; failure of one canonical ordering does
   not prove no permutation-invariant representation exists.
6. **AD/compiler terminology:** the production relationship must remain
   precise. Correctness equivalence does not imply performance equivalence, and
   no AD-vs-hand speed claim is valid without dispatching both comparable
   shaders.
7. **Dirty worktree:** no broad cleanup, formatting, or destructive action is
   safe until adversary and splat changes are separated conceptually.

## Next actions

1. Complete independent math, runtime, and completeness audits.
2. Verify the partial control-knob changes compile and identify missing wiring.
3. Run the smallest decisive suites first, then production build.
4. Answer the user's questions with evidence and exact uncertainty.
5. Only after the audit, implement remaining fixes in a separate, explicit
   pass.
