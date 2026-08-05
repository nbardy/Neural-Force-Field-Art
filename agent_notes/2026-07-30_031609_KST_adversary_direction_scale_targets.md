# Adversary direction, scale, force, and velocity targets

## Goal

Implement and verify selectable adversarial objectives that separate vector
direction, relative magnitude structure, absolute magnitude control, raw force
prediction, and post-update velocity prediction.

Requested selectable loss modes:

1. soft angular/directional;
2. angle plus adversarial relative scale;
3. angle plus scale hold;
4. raw vector negative control;
5. raw force prediction;
6. predicted post-update velocity.

The top-right piece-responsive TUI and URL configuration must expose the modes.
The tfjs implementation remains the independent reference/oracle; supported
production modes must run in the fused WebGPU trainer and match the oracle.

## Questions that must stay explicit

- What exactly does each predictor observe as context?
- What exact target does it predict: raw field force, relative tuple force,
  current velocity, or post-update velocity?
- Which terms does the predictor minimize?
- Which terms does the field maximize, minimize, or merely hold at an energy
  target?
- Does a mode remain invariant under uniform field scaling?
- Does it introduce a direct route to velocity clipping or raw-force
  saturation?
- Is its backward the exact derivative of the displayed scalar loss or an
  explicitly named surrogate gradient?
- How are zero/near-zero vector directions handled without an unbounded
  normalization Jacobian?
- How are direction and scale gradients normalized so one cannot dominate only
  because of units?

## Initial mathematical facts

- Autograd applies the chain rule by composing local Jacobian-transpose-vector
  products. “Chain rule” and “using Jacobians” are not alternatives.
- Solving an ODE is not an alternative for the static normalization map. A
  trajectory adjoint would still consume Jacobian-vector products.
- Any nonconstant exactly scale-invariant directional scalar has gradients that
  scale as `1/c` under `q -> c q`; a bounded nonzero derivative at the origin is
  therefore impossible without relaxing exact scale invariance, suppressing
  the near-zero signal, or using a surrogate backward.
- The mathematically clean smooth embedding candidate is
  `phi_tau(q) = q / sqrt(||q||^2 + tau^2)`, whose exact Jacobian is bounded by
  `1/tau`.
- Adversarial absolute scale error can recreate the old amplitude shortcut.
  Relative/centered log scale plus a separate energy anchor is the candidate
  for unpredictable magnitude structure without rewarding uniform inflation.

## Verified codebase facts and answers

- The shipped strict adversary already predicts the raw neural field output
  `F(x)`. It does **not** predict particle displacement, physical acceleration
  after `forceMagnitude`, or velocity.
- The point observer sees absolute normalized position. Relational observers
  intentionally hide absolute position/orientation; that quotient is what
  makes the conditional target nontrivial.
- Reverse-mode autograd uses the chain rule by evaluating local
  Jacobian-transpose-vector products:

  ```text
  grad_x L = J_f(x)^T grad_f L
  ```

  It normally does not construct a full dense Jacobian. “Chain rule” and
  “Jacobian” are therefore not competing derivative methods.
- An ODE solver is not an alternative derivative for a static normalization
  or loss. A multi-step continuous-time adjoint would still use Jacobian-vector
  products internally.
- A force-induced normalized velocity increment,
  `z_next - friction*z`, is a known scalar multiple of `F(x)` under the current
  drive rule. Raw-force prediction is the cleaner way to test that information.
- Post-update velocity is genuinely different because it also includes
  incoming particle momentum. The predictor must receive normalized incoming
  velocity; otherwise it would be rewarded for state deliberately hidden from
  it even when the field is zero.
- Border bounce/reset and stochastic respawn are excluded from the velocity
  target. They are topology/RNG effects rather than the field update being
  predicted.

## Work log

### Architecture decision

The implementation uses three orthogonal axes:

- observer/tuple geometry;
- target signal: `force` or `post-velocity`;
- loss: `raw-vector`, `soft-angle`, `angle-relative-scale`, or
  `angle-scale-hold`.

Force is the already-shipped strict target `F(x)`, before physical scaling.
Post velocity is

```text
z+ = clamp((v + forceMagnitude*F(x))*friction, +/-maxVelocity)/maxVelocity
```

before border handling or stochastic reset. Its point-state context includes
`[xNorm, v/maxVelocity]`. Post-velocity is restricted to the point observer:
componentwise clipping is world-axis aligned, so presenting it through the
rotation-quotiented pair/tri/quad observers would make a false equivariance
claim.

### Exact soft-spherical objective

For every 2-vector `z`,

```text
psi_tau(z) = (z.x, z.y, tau) / sqrt(||z||^2 + tau^2)
```

and the directional loss uses spherical chord distance between predictor and
target embeddings. This changes the scalar objective and uses its exact
Jacobian. It is not a surrogate backward. The Jacobian norm is bounded by
`1/tau`, including at zero.

Explicitly,

```text
L_angle(p,y) =
  sqrt(||psi_tau(p) - psi_tau(y)||^2 + residualEps^2) - residualEps
```

This is a softened chord loss on the unit sphere in three dimensions. Since
`||a-b||² = 2-2(a·b)` for unit vectors, it has the same ordering as cosine
distance but is **not** the scalar `1-cos`; its gradient magnitude is different.
The added `tau` coordinate represents confidence near zero. At ordinary
magnitudes it approaches direction-only cosine loss; near zero it intentionally
retains a small radial/latitude sensitivity so its exact gradient stays smooth
and finite.

### Relative scale

Relative scale is local to a tuple so it stays compatible with the existing
single-thread fused pass.

- Pair: a swap-invariant soft absolute contrast between the two member
  log-magnitudes, with the softener proportional to tuple RMS. Equal/maximal
  member magnitudes give zero.
- Tri: centered log-magnitudes in canonical vertex order.
- Labelled quad: centered log-magnitudes in labelled order.
- Point: no relative-scale coordinate exists; the UI/runtime reject that
  combination rather than silently substituting a batch statistic.

Every scale coordinate is homogeneous of degree zero above a named all-zero
energy floor. Uniformly scaling all member signals leaves it unchanged, so the
field cannot buy this reward by pushing all outputs to their maximum. It
rewards unpredictable within-tuple magnitude contrast.

This is not a theorem that the learned scale distribution must maximize raw
variance. It maximizes conditional K-head quantization error of a bounded,
dimensionless contrast coordinate. Equal all-max magnitudes map to the same
easy zero contrast. If a particular target variance is later desired, add a
bounded variance-to-target penalty rather than maximizing unbounded variance.

### Energy hold and player roles

`angle-relative-scale` makes direction and relative scale adversarial and adds
a generator-only absolute-energy anchor. `angle-scale-hold` keeps direction
adversarial, makes relative-scale prediction cooperative, and applies the
energy anchor. It is intentionally a general-sum objective, not a strict
zero-sum game. `raw-vector` deliberately retains the amplitude shortcut as the
negative control and receives no anchor.

The anchor is explicit and fixed, never a continuously following EMA:

```text
E2 = mean_active_tuple mean_member ||signal||^2
R  = sqrt(E2 + energyEps^2)
L_energy = energyWeight * (R - energyTarget)^2
```

The predictor never receives the energy-anchor gradient.

The anchor is radial and has zero derivative at an exactly zero field. It
stabilizes and restores nonzero low-energy fields, but cannot resurrect an
exact all-zero initialization by itself. The production neural fields use
nonzero random initialization; tests must not overstate this as a proof that
zero is non-absorbing.

Implementation in progress.

## Required falsifiable gates

1. Exact soft-sphere value and derivative versus finite differences at zero,
   far below `tau`, at `tau`, and far above it; gradients finite and bounded.
2. No discontinuity at the old hard `1e-3` direction cutoff.
3. Relative-scale coordinates unchanged under uniform scale over multiple
   decades; uniform radial derivative approximately zero.
4. Equal/all-max tuple magnitudes produce zero/easy contrast, while
   fixed-energy redistribution changes it.
5. Pair scale mode uses a nondegenerate invariant contrast; it must never
   silently use centered member magnitudes, which are identically degenerate
   for an antisymmetric pair target.
6. Energy gradients pull nonzero fields above and below the configured target
   toward it and vanish at the target.
7. Force target is independent of velocity, friction, max velocity, borders,
   and reset.
8. Post-velocity target matches the analytical update; point context
   distinguishes equal-position/different-velocity states.
9. Unsupported relational post-clip velocity combinations throw visibly until
   their rotational equivariance is proved.
10. Reference, AD oracle, and fused WGSL agree in value and gradient for every
    supported observer/target/loss combination.
11. Agree+Disagree signs remain correct: lane A opposes the predictor, lane B
    cooperates, and display-only blend C receives no direct loss.
12. Near-zero and long-soak probes remain finite and monitor raw-force RMS,
    normalized speed, clipping occupancy, direction loss, scale loss, and
    energy.

## Supported-combination contract

The runtime must validate this matrix rather than silently changing objectives:

| target | observer | raw vector | soft angle | angle + relative scale | angle + scale hold |
|---|---|---:|---:|---:|---:|
| force | point | yes | yes | no (no relative coordinate) | no (same reason) |
| force | pair R / R+S | yes | yes | yes | yes |
| force | tri | yes | yes | yes | yes |
| force | labelled quad | yes | yes | yes | yes |
| post velocity | point + incoming velocity | yes | yes | no initially | no initially |
| post velocity | relational | no | no | no | no |

The point/scale rows are rejected because a single member has no internal
relative scale. The relational/post-velocity rows are rejected because the
current componentwise physical clip is aligned to world axes; quotienting a
clipped vector by arbitrary rotations would claim an equivariance the
integrator does not have.

## Player/objective matrix

Let `A` be the positive angle prediction error, `S` the positive relative-scale
prediction error, and `E` the absolute-energy anchor.

| loss | predictor D | disagreeing field G | agreeing field G |
|---|---|---|---|
| raw vector | minimize `R` | minimize `-R` | minimize `+R` |
| soft angle | minimize `A` | minimize `-A` | minimize `+A` |
| angle + relative scale | minimize `wa*A + ws*S` | minimize `-wa*A - ws*S + we*E` | minimize `+wa*A + ws*S + we*E` |
| angle + scale hold | minimize `wa*A + ws*S` | minimize `-wa*A + ws*S + we*E` | minimize `+wa*A + ws*S + we*E` |

Thus the first three rows are strict opposite-sign predictor games apart from
the generator-only energy anchor. `angle-scale-hold` is deliberately
general-sum: the field challenges direction while cooperating on relative
scale. The UI/HUD must say this rather than calling every configuration
strictly adversarial.

The implementation fixes `wa = 1` as the reference unit and exposes `scale w`
as the relative `ws/wa` control. This is sufficient to balance the two terms
without an unidentifiable common multiplier; the separate game/reward control
sets the overall generator coefficient.
