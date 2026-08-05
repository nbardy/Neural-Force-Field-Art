# Adversary corrective implementation

Started: 2026-07-28 17:23:24 KST

## Goal

Complete the adversarial-art work without losing any prior finding, and correct
the concepts that the implementation and UI currently conflate:

1. field architecture;
2. aesthetic/structural loss;
3. adversarial observation and target;
4. generator role in the game;
5. visualization.

The exhaustive 98-question discovery ledger and fresh audit remain in
`agent_notes/2026-07-28_151154_KST_adversary_thread_fresh_audit.md`. This note
is its implementation continuation; nothing in that ledger is superseded
unless a measured result below explicitly says so.

## User-requested deliverables

- Preserve and address every earlier issue, plan, and concern.
- Keep the gallery selector as the bottom global radio/navigation surface.
- Move all selected-piece configuration into a compact, responsive top-right
  panel below the FPS/HUD panel.
- Add live max-velocity and complete boundary controls.
- Add adversary observation modes:
  - rotation-only;
  - rotation+scale (raw negative control);
  - rotation+scale (adjusted).
- Implement a mathematically genuine scale-adjusted objective which cannot gain
  reward merely by uniformly scaling the generated field/motion.
- Replace the misleading two-vector-MLP “Helmholtz” story with an explicit
  Agree+Disagree dual-generator mode:
  - generator A maximizes prediction residual;
  - generator B minimizes prediction residual;
  - C is a rendered A/B blend and has no direct game loss;
  - render A/B/C as red/green/blue channels or particle groups with an auditable
    legend.
- Make adversarial controls and runtime semantics honest: state exactly whether
  each mode is a strict generator-vs-predictor game, which loss updates which
  weights, and whether updates are simultaneous or alternating.
- Keep the fast fused WebGPU path and parity with the tfjs/reference path.
- Verify numerics, GPU kernels, build, and the visible UI.

## Verified starting facts

- The production `HelmholtzField` is two unconstrained signed-vector MLP heads:
  `F=(1-alpha)g+alpha*r`. It is not a Helmholtz decomposition, and the two heads
  are not assigned different losses.
- The default fused field trainer hardcodes chaos + isotropy + divergence +
  spiral regardless of a gallery piece’s `computeLoss`. Therefore the default
  fused adversary pieces do not run their declared anchor-only objective.
- The point-control premise is invalid in the live fused implementation because
  the predictor receives position while its target includes an unobserved
  incoming velocity through a one-step transition.
- The current frozen RMS denominator makes the displayed normalized value
  roughly scale stable, but does not remove the generator’s radial scale
  gradient. It is not a mathematical fix for the max-velocity exploit.
- The generator seed is additionally divided by
  `stepScale=maxVelocity/min(width,height)`, coupling adversary strength to
  viewport and velocity settings.
- The partial border/max-velocity work is unsafe:
  advection has some support, field training has incomplete backward parity,
  the adversary and tfjs paths remain wrap-only, and the UI exposes none of it.
- Fused surprise writes are sparse/stale, allow duplicate non-atomic writes,
  report coverage incorrectly, and pair pre-advect residuals with post-advect
  positions.
- Pair encoding has a clear E(2) quotient interpretation away from coincident
  pairs. Tri canonicalization is discontinuous/permutation-incorrect at exact
  or near isosceles ties. The previous claim that m=4 is impossible is
  unsupported.
- “K > modes” cannot be inferred from head spread alone. Diverged or
  off-support heads also have large spread; per-head residual/calibration is
  required.
- Production adversary WGSL is handwritten TypeScript-to-WGSL code generation.
  The AD IR is an independent correctness oracle, not the production compiler,
  and no apples-to-apples AD-vs-handwritten performance comparison exists.
- Existing unrelated splat work shares this dirty worktree. It must not be
  reset, cleaned, or mechanically reformatted.

## Semantic decisions being validated

These are proposals until the math and fused-architecture reviews return.

### Strict adversary target

Prefer predicting the generator’s field output `F(x)` (or relational
differences of field outputs) rather than the full next-state displacement.
This removes incoming velocity, clip, friction, boundary mode, and reset events
from the conditional target. It makes the point control genuinely realizable:
`x -> F(x)`.

### Observation variants

- **Rotation-only:** quotient global translation and rotation while retaining
  scale as observable context.
- **Rotation+scale raw:** quotient translation, rotation, and input-geometry
  scale, but retain a raw vector target. This is an intentional negative
  control which can expose the scale exploit.
- **Rotation+scale adjusted:** use the same quotient and a differentiably
  normalized target. For a positive scalar `c`, the target and generator loss
  must satisfy `T(cF)=T(F)` and `dL(cF)/dc=0` away from an explicit zero-field
  floor. A stop-gradient denominator does not satisfy this.

### Agree+Disagree

Reuse the two-head storage/layout only as an implementation substrate, not as a
“Helmholtz” claim. Let `F_A` and `F_B` be separately observable vector
generators and `F_C=(1-beta)F_A+beta F_B`.

- Predictor minimizes residual on labelled branch samples.
- A minimizes `-surprise_A` (tries to defeat predictor).
- B minimizes `+surprise_B` (tries to agree with predictor).
- C receives no direct adversarial loss; it is derived for visualization.
- A/B/C must remain separately renderable and their loss routing testable.

The exact weak regularizers, discriminator conditioning, update ordering, and
whether the predictor is shared or split are pending the rigorous math review.

## Work plan

1. Freeze equations, invariances, update order, and falsifiable tests.
2. Introduce an explicit fused loss specification; make declared tfjs and WGSL
   objectives match for every piece.
3. Replace transition targets with strict field-output targets.
4. Add raw and scale-adjusted relational variants and prove/test the radial
   gradient behavior.
5. Fix epsilon validation, tie/degeneracy handling, head calibration, surprise
   sampling/coverage, and one-step/temporal claims.
6. Complete max-velocity and boundary parity across all forward/backward paths.
7. Implement and fuse Agree+Disagree, including exact A/B/C visualization.
8. Consolidate controls into the top-right selected-piece panel.
9. Run unit/oracle/Metal/parity/build suites.
10. Use the Playwright CLI skill against current source for interaction,
    responsive layout, console, and screenshots.
11. Update `HANDOFF.md`, status/plan docs, gallery copy, URL docs, and this note
    with measured results and remaining limitations.

## Parallel reviews

- `game_math`: equations, invariance domains, scale-gradient proof, fair game.
- `fused_arch`: exact WGSL/layout/loss-routing implementation map.
- `ui_arch`: responsive top-right TUI controls and A/B/C visualization map.

## Commands inspected so far

- `git status --short`
- `sed`/`rg` across `src/main.ts`, `src/index.tsx`,
  `src/core/field/helmholtz.ts`, `src/core/gan/adversary.ts`,
  `src/render/webgpu/{advect,train,adversary}_*.ts`
- `package.json`, `AGENTS.md`, and the prior audit note

## Open concerns

- A fully strict scale quotient at zero vector magnitude requires an explicit
  policy; numerical epsilon smoothing reintroduces a small radial gradient.
- Agree+Disagree is not zero-sum as a whole: B and the predictor cooperate while
  A and the predictor oppose. The UI must not call this a conventional GAN.
- Exact permutation invariance for symmetric tri/quad configurations may
  require dropping ambiguous samples or an assignment-invariant/set model.
- Runtime border switching may require rebuilding code-generated pipelines.
  It must not be exposed until all forward/backward/reference parity tests pass.
- “Fix all issues” includes correcting or removing unsupported claims; it does
  not justify silently changing unrelated splat code.

## Rigorous math review (completed)

The independent review resolved the central definitions:

- Strict target: raw learned field output before force magnitude, velocity,
  friction, clipping, boundary, or reset. Point control is `u=x, y=F(x)`.
- Pair rotation-only:
  `u=[r]`, `y=R(d)^T(F(x1)-F(x0))`.
- Pair rotation+scale raw negative control:
  geometry scale is removed from `u`, but the vector target remains raw, so it
  deliberately retains a radial scale exploit.
- Pair rotation+scale adjusted: normalize active target and prediction and use
  angular distortion `rho=1-yhat·qhat`. For active samples,
  `d yhat/dy=(I-yhat yhat^T)/||y||`, hence
  `y^T grad_y rho=0`. Zero targets are explicitly inactive; no epsilon-smoothed
  claim of exact invariance is allowed.
- Relaxed WTA must require `epsilon < (K-1)/K`, otherwise each loser can receive
  more weight than the winner.
- A strict predictor/generator subgame must use the same relaxed-WTA payoff on
  both sides. The previous D-weighted/G-minimum pair was not zero-sum.
- Agree+Disagree is a **general-sum** game:
  `L_D=V_A+V_B`, `L_A=-V_A`, `L_B=+V_B`,
  `C=(1-beta)A+beta B`, with no direct C loss or C optimizer.
- Equal learning rates/capacities do not imply 50/50 loss or a GAN-classifier
  equilibrium. Correct telemetry is held-out distortion/tracking lag and
  branch-specific gradients.
- Square-torus arbitrary-rotation claims are local-chart claims; globally the
  square torus only has lattice-preserving rotational symmetries.
- m=4 canonical distance matrices still have automorphism/target-assignment
  ambiguity; a 24-permutation lexicographic minimum alone is not a full target
  solution.

## Implementation progress after review

- Fused field loss is now explicit per gallery piece and passed into
  `FusedTrainer`; standalone adversary pieces no longer silently acquire
  chaos+spiral.
- Maximum-chaos field pieces now use a no-spiral finite-difference sensitivity
  objective and are renamed “Neural Field”, avoiding a false Helmholtz claim.
- Standalone adversary pieces declare zero structural loss, making their game
  routing auditable; Chaos Weave remains the explicitly composed exception.
- `LoopHandle` and `startLoop` now have the runtime/config API for max velocity,
  neutral head blend, React-owned stroke controls, telemetry host, boundary,
  and adversary-encoding overrides.
- Max velocity now updates advection, fused/tfjs physics arguments, both
  velocity renderers, HUD, step scale, and legacy surprise threshold.
- The imperative stroke-control DOM island was removed.
- Boundary mode is passed into advection and field training. The analytic BPTT
  now applies the emitted wrap/bounce/reset Jacobian to both post-border
  position and velocity paths.
- Added an explicit `agree-disagree` field semantic using the existing packed
  two-head storage only as implementation substrate.
- Added stable A/B/C particle roles in advection and exact RGB-role palette
  support in both point and splat renderers.
- Added the `Adversary · Agree + Disagree RGB` gallery piece with zero direct C
  objective. Its fused game kernel is still being completed and must pass
  branch-isolation/sign tests before this item is marked finished.
- Replaced unsupported `K > modes` telemetry with conservative
  `separated-unresolved`; head spread alone cannot establish support validity.
