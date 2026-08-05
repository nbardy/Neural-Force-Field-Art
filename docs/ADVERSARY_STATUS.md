# Adversary status

Last audited: 2026-07-30.

This is the operational status of the adversarial art subsystem. Mathematical
motivation and the full idea ledger live in
`docs/PLAN_RELATIONAL_ADVERSARY.md`. A passing unit/kernel test proves the
named invariant; it does not substitute for a browser soak.

## What the game observes

Target and loss are independent controls. The default `force` target is the
learned field output itself:

```text
x ──field──> F(x)
u = observer(context positions)
y = observer(target signals F(x))
```

`F(x)` is sampled before `forceMagnitude`, velocity integration, friction,
velocity clipping, border handling, and random respawn. Consequently
`maxVelocity`, bounce/wrap/reset, and particle speed are not inputs to this
target and cannot directly buy its predictor error.

The selectable `post-velocity` target instead predicts

```text
v+ = clip((v + forceMagnitude*F(x))*friction, +/-maxVelocity)/maxVelocity
```

before border handling or reset. Its point observer sees both normalized
position and incoming normalized velocity, so hidden momentum cannot masquerade
as surprise. It intentionally depends on the live physics controls and is
point-only: the componentwise velocity clip is aligned to world axes and does
not respect the current relational rotation quotient.

The WTA predictor is regression, not a classifier GAN:

```text
d_j(u,y) = residual(g_j(u), y)
w_j      = relaxed-WTA assignment, held constant on the tape
V        = E[sum_j w_j d_j]

predictor: minimize  V
field:     minimize the configured generator game
```

For `K>1`, the winner receives `1-epsilon` and losers share `epsilon`.
Configuration validation requires
`0 <= epsilon < (K-1)/K`; otherwise a nominal loser would receive at least as
much weight as the winner.

Equal learning rates and model capacities do **not** imply a 50/50 equilibrium.
The players solve different optimization problems, see different parameter
spaces, and alternate updates. Telemetry should be read as a time series, not
as a classifier accuracy expected to settle at one half.

## Observer, target, and loss modes

The observer controls which geometry is visible:

| observer | context `u` | force target |
|---|---|---|
| `point` | absolute position | `F(x)` |
| `pair-rotation` | pair separation | relative force in the pair frame |
| `pair-rotation-scale-raw` | constant similarity context | raw relative force; deliberate scale-cheat control |
| `pair-rotation-scale-adjusted` | same scale-blind pair context | compatibility alias; the explicit loss now owns adjustment |
| `tri` | canonically ordered side lengths | centered forces in the triangle frame |
| `quad-labelled` | three labelled anchor-frame coordinates | four centered labelled forces |

The loss controls what prediction error means:

| loss | predictor score | disagreeing-field score | scale behavior |
|---|---|---|---|
| `raw-vector` | raw softened vector distance `R` | `-R` | deliberate amplitude-shortcut control |
| `soft-angle` | smooth spherical chord `A` | `-A` | asymptotically direction-only, finite at zero |
| `angle-relative-scale` | `A + ws*S` | `-A - ws*S + we*H` | relative log-scale is adversarial; fixed RMS energy anchor |
| `angle-scale-hold` | `A + ws*S` | `-A + ws*S + we*H` | direction adversarial, relative scale cooperative |

The smooth angular map is

```text
psi_tau(z) = (z.x, z.y, tau) / sqrt(||z||^2 + tau^2)
A(p,y) = sqrt(||psi_tau(p)-psi_tau(y)||^2 + eps^2) - eps
```

This changes the forward scalar loss and differentiates it exactly. It is not a
capped or swapped surrogate gradient. Its Jacobian norm is bounded by `1/tau`;
smaller `tau` is more direction-like away from zero but stiffer near zero.

Relative scale uses within-tuple log-magnitude contrasts. Uniformly multiplying
every tuple member by the same positive scale leaves `S` unchanged, and equal
all-max magnitudes produce the easy zero contrast. Thus uniform magnitude
inflation cannot buy relative-scale surprise. `H` holds absolute RMS energy near
a fixed target; it has zero derivative at the exact all-zero field, so nonzero
initialization remains load-bearing.

The historical adjusted-pair path no longer divides by
`|F_2-F_1|`, pre-normalizes the target, or injects a custom tangent Jacobian.
`pair-rotation-scale-adjusted` keeps only its scale-blind geometry name; the
explicit loss selector defines the current objective.

The square periodic domain is a torus. Arbitrary planar rotation is a local
chart statement, not a global symmetry of the square lattice at the seam.
Invariance tests distinguish non-wrapping Euclidean tuples from minimum-image
torus behavior.

## Agree + Disagree

The new A/B/C piece is a general-sum two-generator experiment:

```text
V_A = predictor payoff on field head A
V_B = predictor payoff on field head B

predictor system: minimize V_A + V_B
A:                minimize -V_A  (disagree)
B:                minimize +V_B  (agree)
C:                (1-beta) A + beta B
```

C is derived. It has no weights, optimizer, external-gradient lane, or direct
loss. Stable particle roles render A/B/C as exact red/green/blue. This is not
the same as the legacy two-head field blend: that architecture simply mixed two
direct-vector MLP outputs and did not make one lane agree and the other oppose
the predictor.

The implementation must pass all of these before being called live:

- A and B use independent predictor state.
- A's external gradient touches only field lane A.
- B's external gradient touches only field lane B and has the opposite sign.
- changing beta changes only derived C rendering/advection, not either loss.
- the browser piece starts, trains, and soaks without NaN.

## Field and loss terminology

`HelmholtzField` remains an internal compatibility class name. In current
production it contains two direct-vector MLPs and blends their outputs. It does
**not** compute `-grad(phi) + rot90(grad(psi))`, and the two ordinary lanes are
not intrinsically “order” and “chaos.”

Chaos is a loss choice:

```text
L_chaos = -log(||J_F|| proxy + epsilon)
```

The maximum-chaos gallery modes route an explicit field-loss specification into
the fused trainer. Standalone adversary pieces use zero structural field loss;
Chaos Weave is the explicitly composed exception. Exact Helmholtz potential/curl
construction was measured at roughly 800–1700 ms/frame in the old tfjs path
because training requires second derivatives, so it remains rejected for this
interactive implementation.

## Head-health telemetry

A skewed win histogram is not a diagnosis by itself.

- skew + small prediction spread: `pileup`
- skew + separated predictions: `separated-unresolved`
- missing spread or scale probe: `unresolved`
- otherwise: `ok`

Separated heads do **not** prove `K` exceeds the number of conditional modes.
The earlier `data-limited` / `K>modes` label overclaimed what head geometry can
establish and has been removed.

The production fused path reduces the exact mean and closest unordered
predictor-head separation in the same forward pass as its win assignments.
Agree+Disagree classifies A and B against their own win EMA, spread and reward
scale, then combines those verdicts conservatively; branch evidence is never
averaged before classification. `UNPROBED` is reserved for a genuinely missing
probe, not used as the fused default.

## Fused implementation

The production adversary is hand-generated WGSL. The scalar AD IR is an
independent derivative oracle; it is not the production compiler and has no
separate runtime speed number. The relevant performance comparison is fused
WGSL versus the tfjs reference.

One final B=512 Metal gate measured:

| path/config | measured step time |
|---|---:|
| fused pair K=4 | 0.844 ms |
| fused tri K=6 | 1.006 ms |
| fused labelled quad | 0.785 ms |
| old in-app tfjs adversary | 19–32 ms |

Those are machine- and load-specific kernel measurements, not a universal
60-FPS guarantee. Retina resolution, renderer, particle count, GPU load, and
the selected piece still determine frame rate.

Fused execution must remain numerically gated against both the tfjs reference
and the AD oracle. The important contracts are:

- predictor and field parameter buffers are disjoint;
- predictor update occurs before generator-gradient recomputation;
- both players use the same relaxed-WTA assignment selected by the predictor's
  combined residual;
- every configured player sign matches the objective table above;
- soft-angle uses the exact bounded S² derivative;
- relative-scale coordinates have zero uniform-radial derivative;
- post-velocity propagates through the exact unsaturated physical-update
  Jacobian and contributes zero force derivative after a component clips;
- invalid tri/quad rows contribute exact zero;
- external gradient lanes are isolated;
- the finalized-stats prefix cannot overlap workgroup partials at any supported
  K.

## Surprise rendering

Surprise is a tuple property scattered to tuple members. Honest coverage
requires unique scheduled writes or an explicit reduction: random sampling
with replacement plus non-atomic duplicate writes is a race and cannot report
`covered=1`.

The fused repair now uses a unique rotating tuple window and exact measured
coverage. Its focused real-GPU gates prove:

- no duplicate non-atomic writes in one dispatch;
- a deterministic rotating window eventually visits all particles;
- buffer resize resets coverage and stale values;
- the reported fraction is the measured fraction, not a constant;
- render normalization ignores untouched entries.

The fused buffer has two display-only planes:

```text
raw       = shared relaxed-WTA payoff
per-unit  = raw / ||target||, for active ||target|| > 1e-3; else exactly 0
```

Selecting a plane changes neither player's objective, gradients, Adam state nor
field weights. Its percentile history resets on a live metric switch so values
from the two units are never mixed. The old tfjs particle-transition
`SurpriseChannel` was removed: it observed an implicit transition random
variable rather than either named target and was wrap-specific. When the fused
adversary is unavailable, the app falls back to velocity colour with a loud
warning; tfjs remains a numerical oracle/reference trainer.

Observer geometry follows the selected physical topology explicitly. Wrap uses
periodic minimum-image coordinates. Bounce and reset use Euclidean coordinates;
they never inherit torus geometry merely because the original implementation
did.

## UI contract

The bottom strip is the global art-piece radio. The compact top-right dock is
piece-responsive.

Live controls backed by `LoopHandle`:

- particle count, shared field/adversary training batch, max velocity,
  dimensionless drive and respawn rate;
- generator and predictor/discriminator learning rates plus their displayed
  ratio;
- trail decay, stroke style, stroke length;
- neutral A/B or derived-C blend;
- non-negative adversary coefficient (the named generator role owns sign);
- velocity/raw-surprise/per-unit-surprise color mode and colormap where
  applicable.

Restart/compile-time controls:

- border mode;
- tuple arity and observer quotient;
- K and relaxed-WTA epsilon.

The dock labels restart controls explicitly and preserves every live dial when
those GPU pipelines rebuild. Pair quotient radios appear only for pair
encodings; point, pair, tri and labelled-quad are all selectable in the tuple
row. Point, tri and labelled-quad then show their exact observer contract as
read-only text rather than displaying an unselected pair radiogroup.
Agree+Disagree keeps RGB role coloring fixed so a generic colormap cannot
silently hide the A/B/C experiment.

## Verification inventory

The corrective implementation was release-verified on 2026-07-29 with:

```sh
bun tools/adversary_test.ts
bun tools/adversary_strict_test.ts
bun tools/adversary_wire_test.ts
bun tools/ad_wta_test.ts
bun tools/train_wta_test.ts
bun tools/field_loss_routing_test.ts
bun tools/border_modes_test.ts
bun tools/drive_controls_test.ts
bun tools/agree_disagree_live_test.ts
bun tools/adversary_stability_probe.ts
bun tools/surprise_test.ts
bun tools/train_test.ts
bun tools/train_types_test.ts
bun tools/kernel_test.ts
bun tools/integration_test.ts
bun tools/splat_test.ts
bun tools/splat_stroke_test.ts
bun tools/splat/raster_test.ts
bun tools/splat/feature_painter_math_test.ts
bun tools/splat/feature_painter_test.ts
bun tools/splat/optimize_test.ts
bun tools/splat/pixel_optimize_test.ts
bun tools/splat/feature_nudge_test.ts
bun tools/ad_test.ts
bun tools/ad_train_test.ts
bun tools/ad_jvp_test.ts
npm run build
```

Both adjusted and deliberately raw rotation+scale observers also completed
3,000-step finite-weight/finite-gradient probes.

Real-browser QA used the current Parcel source in a real WebGPU browser at
desktop and 390x844 mobile viewports. It exercised all seven adversarial gallery
pieces, three observer modes, three border modes, K rebuild, max-velocity and
dimensionless-drive edits, RAW/PER UNIT switching, fixed Agree+Disagree RGB,
independent A/B health, and the honest tfjs fallback. All seven fused pieces
reported approximately 60 FPS during the interaction run; the tfjs reference
path remained explicitly labelled at about 25 FPS. Screenshots and the
reproducible Playwright script are under `output/playwright/`.

The final 30-second-per-piece behavioral soak passed every gate for all seven
adversarial modes: no page/console errors, no NaN/Infinity, stable tensor
counts, no confirmed head pileup, 100% diagnostic coverage where applicable,
non-flat RAW and PER UNIT spans, and trailing mean-speed/clip ratios between
0.328 and 0.442. Chaos Weave measured a median 60 FPS with roughly 0.59 ms
rollout and 0.12 ms optimizer time. The run artifacts are under
`output/playwright/adversary-soak/2026-07-28T17-38-54-170Z/`.

One browser-only regression was found and fixed during this verification:
`physicsForward` used `forceMagnitude` without declaring the live final
parameter. Parcel strips TypeScript without type-checking, so the fused
renderer continued drawing while the tfjs training path threw. A behavioral
wire gate now calls the function with 1x and 3x magnitudes and asserts exact
3x physical force. Warm-up loss is rendered as `warming`, never the misleading
string `NaN`.

The final audit also removed K-dependent idle reference state from fused
startup. `AdversaryRuntime` now distinguishes fused and tfjs implementations;
only the latter can own a tfjs `Adversary` and its per-head Adam state. The
tfjs generator optimizer is likewise constructed only when no `FusedTrainer`
exists. A focused wire gate proves fused ordinary and Agree+Disagree runtime
construction retains zero predictor tensors while the explicit fallback still
constructs and disposes its reference game. Current live fused pieces all hold
the same 13 tfjs blueprint tensors, independent of K (previously 23–79); the
tfjs Pair fallback remains functional and explicitly labelled with 119 live
tensors.

A post-refactor smoke covered all seven pieces with zero page/console errors,
finite advancing telemetry, stable 13-tensor counts, and no pileup. The
deliberately five-second WTA-8 pass ended before one deterministic coverage
cycle at the browser's current 30 Hz vsync, so WTA-8 was rerun for 12 seconds
and reached 100%. That successful artifact is
`output/playwright/adversary-soak/2026-07-28T18-02-23-027Z/`; the earlier
30-second-per-piece run remains the long-run release gate.

A fast kernel suite cannot replace these browser/soak gates. Re-run both after
changing game composition, topology, pipeline rebuilds, or live control
wiring.
