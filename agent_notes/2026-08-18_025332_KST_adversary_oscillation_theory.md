# Adversary oscillation: why it happens, 10 candidate cures with proof sketches

**Status: theory/proposal note — nothing implemented.** Requested 2026-08-18: the
adversary pair has no ground truth, so D pushes G between two answers and the
pair orbits instead of settling. Question raised: "memory for the discriminator
— maybe EMA is enough?"

## Verified code facts (this session)

- Adam `beta1 = 0.9` in all fused trainers: `src/render/webgpu/train.ts:65`,
  `adversary_train.ts:60`, `pixel_disc_train.ts:29`. Positive momentum is a
  proven oscillation amplifier in adversarial games (Gidel et al. 2019).
- The "5 guesses" mechanism is K=8 relaxed-WTA predictor heads
  (`src/core/gan/adversary.ts` VARIANT B `wta`, `src/main.ts:2115`, ε=0.05).
  It is a **spatial** mixture (holds several hypotheses at one instant); it has
  no **temporal** memory — the whole ensemble can still orbit.
- EMA exists only for HUD/stat smoothing (`src/index.tsx`, `src/main.ts`), not
  for weights. R1 instruments landed in commit b72aa54.

## The master picture (why oscillation is structural, not a bug)

Zero-sum game `min_θ max_ψ f`. Simultaneous gradient descent-ascent is Euler
integration of the flow `ż = -v(z)` with `v = (∂f/∂θ, -∂f/∂ψ)`. At an
equilibrium the Jacobian of v is

```
J = [  f_θθ   f_θψ ]
    [ -f_ψθ  -f_ψψ ]
```

Near GAN equilibria the diagonal (own-curvature) blocks are ≈ 0, so J is
antisymmetric → eigenvalues purely imaginary → the continuous flow is a pure
rotation. Minimal example (Dirac-GAN, Mescheder et al. 2018): `f(θ,ψ) = θ·ψ`
gives `θ' = -ψ, ψ' = θ`, and `d/dt(θ²+ψ²) = 0` — exact circles. Euler with step
η multiplies the radius by `√(1+η²) > 1` every step — an outward spiral for
*every* learning rate. Adam's normalized steps turn the divergence into a
bounded limit cycle — which is exactly the observed "oscillate between two
answers."

Every known cure does one of three things:

- **(A) Damp**: move eigenvalues into the left half-plane.
- **(B) Average**: integrate the rotation away over time.
- **(C) Convexify with memory**: lift players to mixtures over strategies
  (Sion's minimax theorem applies in distribution space; averages converge).

## The 10

### 1. G-weight EMA for the displayed field (B) — Polyak averaging / Yazıcı et al. 2019
`θ̄_t = β θ̄_{t-1} + (1-β) θ_t`; render/advect with θ̄, train on θ.
*Proof sketch:* on an orbit `z_t = R(ωηt) z_0` the uniform average
`(1/T) Σ z_t = O(1/T) → 0` = the orbit's center = the equilibrium; the EMA is
the geometric version with residual amplitude `(1-β)/|1-βe^{iωη}| ≈
(1-β)/√((1-β)² + (βωη)²)`, small whenever the oscillation period is short
relative to the EMA window.
*Verdict on "is EMA enough":* for the **art output**, yes, provably — the
averaged generator sits at the center while the raw one circles. For the
**training dynamics**, no — it is cosmetic; the orbit is untouched.

### 2. D-weight EMA as a proximal anchor (A) — the rigorous "discriminator memory"
Keep `ψ̄` = EMA of D's weights (rate μ) and add `λ‖ψ - ψ̄‖²` to D's loss, i.e.
one extra term `-λ(ψ - ψ̄)` in D's fused update. (Historical-averaging penalty,
Salimans et al. 2016.)
*Proof (Dirac-GAN, complete):* the augmented linear system
`θ' = -ψ`, `ψ' = θ - λ(ψ-ψ̄)`, `ψ̄' = μ(ψ-ψ̄)` has characteristic polynomial
`s³ + (λ+μ)s² + s + μ`. Routh–Hurwitz: coefficients positive and
`a₂·a₁ > a₀ ⇔ λ+μ > μ ⇔ λ > 0`. So **any** λ>0, μ>0 turns the perfect orbit
into asymptotic convergence, with the same equilibrium. Memory alone kills the
cycle.
*Cost:* one extra f32 per D weight + 2 FMAs in the fused Adam kernel. Best
theory-per-byte on this list.

### 3. Optimistic gradient / OGDA (A) — Popov 1980; Daskalakis et al. 2018
`z_{t+1} = z_t - η(2F_t - F_{t-1})` — reuse last step's gradient as a forward
extrapolation.
*Proof sketch:* `2F_t - F_{t-1} ≈ F_t + η Ḟ`; the correction is the tangent
derivative = centripetal on a rotation field. Rigorously, OGDA tracks the
proximal-point (implicit Euler) method to O(η²) (Mokhtari–Ozdaglar–Pattathil
2019), and implicit Euler on a rotation is unconditionally stable:
`|1/(1+iηω)| < 1`. Linear convergence on bilinear games for η ≤ 1/(2L).
*Cost:* persist one previous-gradient buffer, one extra FMA. Composes with Adam
("optimistic Adam").

### 4. Extragradient (A) — Korpelevich 1976
Half-step lookahead: `z_{t+½} = z_t - ηF(z_t)`; `z_{t+1} = z_t - ηF(z_{t+½})`.
*Proof sketch:* for monotone L-Lipschitz F,
`‖z_{t+1}-z*‖² ≤ ‖z_t-z*‖² - (1-η²L²)‖z_t-z_{t+½}‖²` — descent for η < 1/L.
The gradient a quarter-turn ahead points inward.
*Cost:* 2 full passes per step (doubles encodeStep). #3 is this at half price —
prefer #3.

### 5. Momentum surgery (A) — Gidel et al. 2019
Current code: β1=0.9 everywhere. Theorem (bilinear): simultaneous GDA diverges
for any heavy-ball β ≥ 0; **alternating** GDA converges linearly for β ∈ (-1,0)
(optimum near -½); positive momentum adds inertia along the tangent and widens
the spiral.
*Proof sketch:* spectrum of the augmented (z_t, z_{t-1}) iteration matrix;
β > 0 pushes the rotation eigenvalues outside the unit circle, β < 0 pulls them
in.
*Practical:* try `beta1 = 0` for D (and G) — a one-constant experiment; true
negative momentum needs a heavy-ball term. Also verify the fused step is
alternating (G sees post-update D or vice versa); if both gradients are
computed from the same snapshot inside one encoder, it is simultaneous — the
strictly worse discretization (spectral radius √(1+η²) vs 1).

### 6. R1 zero-centered gradient penalty (A) — Mescheder et al. 2018 (already shipped; it's the damping dial)
*Proof (2 lines):* Dirac-GAN + R1 with weight γ: `θ' = -ψ`, `ψ' = θ - γψ`.
Flow matrix `[[0,-1],[1,-γ]]`: trace −γ < 0, det 1 > 0 ⇒ eigenvalues
`(-γ ± √(γ²-4))/2`, real part −γ/2 ⇒ exponentially damped spiral, envelope
`e^{-γt/2}`, ringing frequency `√(1-γ²/4)`.
*Interpretation:* γ **is** the damping coefficient. Measure the oscillation
period on the HUD and set γ so the decay time constant spans a few periods.
Slightly underdamped (γ < 2 in Dirac units) may be aesthetically right.

### 7. Fictitious play / replay buffer for D (C) — Brown 1951; Robinson 1951
D trains against the mixture of **all past** generator outputs: reservoir-
sample particle/field snapshots into a ring buffer; D's batch = 50% fresh
fakes + 50% replayed (Shrivastava et al. 2017 form).
*Theorem (Robinson 1951):* in two-player zero-sum games, fictitious play's
empirical strategy averages converge to Nash. Best-responding to the
time-averaged opponent is cure (B) applied inside the game.
*Cost:* a small GPU ring buffer; particle snapshots already live on-GPU.

### 8. Two-timescale (TTUR) — honest caveat (A only with curvature)
Already running D LR = 3× G LR. Two-timescale stochastic approximation
(Borkar; Heusel et al. 2017): if D's inner problem is locally strongly concave
and η_D/η_G → ∞, D tracks ψ*(θ) and G descends the true dual J(θ) = max_ψ f —
a gradient flow, which cannot cycle (J is its own Lyapunov function).
*Caveat with proof:* timescale separation alone does NOT fix the bilinear
core: `[[0,-a],[b,0]]` has eigenvalues `±i√(ab)` — purely imaginary for every
a,b > 0; rescaling rows cannot rotate them off the axis. TTUR needs R1-style
curvature in D's block to bite. Observed: still oscillating at 3:1 — consistent
with the theory.

### 9. Unrolled generator objective (C) — Metz et al. 2017
G descends `f_k(θ) = f(θ, D after k inner steps)`, differentiating through the
updates. As k→∞, `f_k → J(θ) = max_ψ f`; f is **linear** in the generator
distribution p_G, so J = sup of linear functionals = convex in p_G — the
idealized dynamics minimize a convex functional; Sion's minimax theorem gives
min max = max min. No cycling in the limit.
*Cost:* backprop through Adam steps in WGSL — prohibitive in the fused path.
k=1 without backprop-through-update ≈ alternating updates (#5's free half).

### 10. Consensus optimization / SGA (A) — Mescheder et al. 2017; Balduzzi et al. 2018
Replace v by `v + γ∇(½‖v‖²) = v + γJᵀv`.
*Proof (1 line, local):* rotation mode `Jx = iωx` ⇒ modified flow eigenvalue
`-(iω + γω²)`: real part `-γω² < 0` for every nonzero rotation frequency. On
Dirac-GAN `½‖v‖² = ½(θ²+ψ²)` — the correction is literally a radial pull to the
equilibrium. SGA refines: split J = S + A (symmetric + antisymmetric); add
`γAᵀv` to damp only the Hamiltonian part.
*Cost:* one Jacobian-vector product = second backward pass, ~2× step cost. Most
invasive; most surgically targeted at exactly this disease.

## On the K=8 WTA heads

They convexify across **modes at one instant** (D/predictor holds several
hypotheses simultaneously — good against mode-hopping), but every head chases
the *current* G, so the ensemble orbits together. Orthogonal axis to #2/#7,
which convexify across **time**. Keep the heads; add temporal memory.

## Ranked shortlist for the fused-WGSL budget

1. **#2 D-EMA proximal anchor** — 1 buffer + 2 FMAs, Routh–Hurwitz-provable,
   the user's instinct made rigorous.
2. **#3 optimistic Adam** — 1 previous-grad buffer + 1 FMA.
3. **#6 R1 γ retune** — already shipped; set γ from the measured HUD period.
4. **#5 β1=0 experiment** — free; and confirm alternating update order.
5. **#7 replay rows in D's batch** — small ring buffer.
6. **#1 G-EMA for display** — stabilizes the art even while the game orbits.

Next actions: none started. If implementing, #2 + #5 first (cheapest, both
provable on Dirac-GAN), instrument the oscillation period on the HUD before/
after so the damping is measured, not guessed.
