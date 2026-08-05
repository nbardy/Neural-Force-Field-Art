# Relational adversary: mathematics and complete idea ledger

Re-audited 2026-07-29. This document preserves the reasoning from the original
thread while correcting claims invalidated by code inspection or browser
testing. Operational status is in `docs/ADVERSARY_STATUS.md`.

Status words are strict:

- **implemented**: code exists;
- **verified**: the named test was run and passed;
- **pending**: required for release but not yet verified;
- **experimental**: worthwhile hypothesis, not a completion requirement;
- **rejected**: investigated and deliberately not used.

## 1. The game is prediction, not classification

Let `theta` be field parameters and `phi_j` predictor-head parameters.

```text
F_theta(x)          raw learned 2-vector
kappa(X, F(X))      tuple observer producing context U and target Y
g_phi_j(U)          predictor head j
d_j                 residual(g_phi_j(U), Y)
```

For K relaxed-WTA guesses:

```text
j* = first argmin_j d_j

w_j = 1-epsilon                 if j=j*
      epsilon/(K-1)             otherwise

V(theta,phi) = E[sum_j w_j d_j]
```

Winner selection and `w_j` are constants on the differentiation tape. The
ordinary adversarial game is:

```text
min_phi    V(theta,phi)
min_theta -V(theta,phi)
```

Both sides therefore use the same scalar payoff with opposite signs. The
generator is a neural force field and the “discriminator” is a regressor; this
is not a real/fake classifier GAN. Similar learning rates/capacities do not
imply 50% accuracy or equal loss because the game has no classifier accuracy
and the parameter spaces are asymmetric.

The relaxed assignment is only semantically winner-dominant when

```text
0 <= epsilon < (K-1)/K.
```

At epsilon zero, losing heads receive exact zero gradient and can freeze.
Relaxed WTA keeps them learning; it cannot manufacture modes absent from the
conditional distribution.

## 2. Why the original pointwise ensemble dies

If `Y=f(U)` almost surely and `f` is realizable by the predictor class, then:

```text
inf_g E[d(g(U),Y)] = 0
```

Every independently trained predictor converges toward the same `f`, so
ensemble disagreement also tends toward zero. The earlier
`PredictorEnsemble(position -> displacement)` had no callers and its premise
was degenerate: the field is itself a deterministic function of position.

The measured historical ladder supported the diagnosis:

```text
point/single normalized residual: 0.58 -> 0.50 -> 0.20
```

Adding more guesses does not fix a point-mass conditional. A persistent game
requires intentionally hiding information from the observer so `P(Y|U)` is
one-to-many.

## 3. Strict signal boundary

The target is raw field output:

```text
Y <- F_theta(x)
```

not:

```text
velocity, displacement, forceMagnitude*F, clipped motion,
post-border position, or respawn transition.
```

This boundary removes the most direct max-velocity cheat. It also makes the
adversarial objective independent of border semantics. The physical renderer
still uses force magnitude, friction, clipping and wrap/bounce/reset.

This is a deliberate change from the early one-step-displacement prototype.
Any old proof, test, comment or diagram using `Delta x` as the production
target is historical, not current.

## 4. Observer family

### 4.1 Point: deterministic control

```text
U = x
Y = F(x)
```

No symmetry is quotiented. This is retained as a visible negative control, not
as the preferred live signal.

### 4.2 Pair, rotation only

For two positions on the unit torus, using the minimum-image difference:

```text
d      = MI(x_2 - x_1)
r      = ||d||
e      = d/r
e_perp = rot90(e)
deltaF = F(x_2)-F(x_1)

U = [r]
Y = [dot(deltaF,e), dot(deltaF,e_perp)]
```

Translation and local Euclidean rotation are removed. Scale remains in both
the context and target magnitude.

### 4.3 Pair, rotation+scale raw: negative control

Geometry scale is removed while target amplitude remains raw:

```text
U = [1]
Y = pair-frame deltaF
```

This deliberately exposes the scale exploit. Inflating `deltaF` can increase
distance-based payoff.

### 4.4 Pair, rotation+scale adjusted: active fix

For active nonzero targets:

```text
y_hat = Y/||Y||
q_hat = q/||q||
rho   = ||q_hat-y_hat||    (equivalently a monotone angular error)
```

Zero-magnitude targets are inactive and contribute exact zero; no arbitrary
direction is invented. The derivative of normalization is:

```text
d y_hat / dY = (I-y_hat*y_hat^T)/||Y||
```

therefore:

```text
Y^T grad_Y rho = 0.
```

The generator gradient is tangent to the target direction and has no uniform
radial component. This is stronger than dividing by a stop-gradient batch RMS:
a tape-constant denominator rescales the gradient but does not remove its
radial direction.

The angular game does not by itself bound field magnitude. Radial parameters
can drift under Adam or other losses while the angular objective is
indifferent. Production therefore combines bounded tanh field output with the
dimensionless physical-drive invariant, and verifies optimization separately:
the adjusted and raw controls both completed 3,000-step finite-state probes;
all seven pieces then passed the current browser soak without magnitude or
velocity-clip drift. This is empirical stability evidence, not a theorem that
the angular payoff controls radial parameters.

### 4.5 Triangle

Triangle context uses the three side lengths in canonical order; target is the
three centered forces in its canonical frame. Near-equal side lengths make the
canonical label ambiguous. Such rows must be marked inactive rather than
resolved with an index tie-break and then described as exactly
permutation-invariant.

The square torus is not globally invariant under arbitrary planar rotations.
Rigid-motion claims apply to non-wrapping Euclidean tuples; seam behavior is a
separate minimum-image property.

### 4.6 Labelled quad

There is no generic permutation-invariant four-point canonical label with the
simple distance-key construction originally proposed. Equal keys need not imply
a configuration symmetry, and side lengths alone are incomplete.

The supported four-point experiment is consequently explicit and labelled:

```text
member 0: anchor
member 1: defines rotation frame
members 2,3: keep their labels

U = three labelled relative coordinates in the anchor frame   (6-D)
Y = four mean-centered labelled forces in that frame           (8-D)
```

Reordering members changes the sample by design. Near-coincident anchor pairs
are inactive. This preserves the original “four points in B tuples” experiment
without making a false invariance claim.

## 5. What “surprise” means

With one exact Euclidean head, minimizing expected norm finds a geometric
median, not generally the mean. In the ideal unsoftened two-equal-mode case,
every point on the segment between them minimizes the objective:

```text
0.5*(||y_1-g|| + ||y_2-g||) >= 0.5*||y_1-y_2||.
```

Production uses `sqrt(||e||^2 + 10^-12)` to keep the derivative finite. That
tiny softening removes the exactly-flat segment (the symmetric fixture prefers
its midpoint), while converging to the geometric-median objective away from the
epsilon-scale neighborhood. Tests must compare the implemented softened loss,
not assert exact flatness.

With K heads, min-over-K performs conditional vector quantization. The
irreducible quantity is the conditional K-point distortion:

```text
D_K = E_U [ inf_{c_1...c_K} E[min_j d(c_j,Y) | U] ].
```

`D_K>0` is persistent predictor error under that observer. It is not a proof of
semantic novelty, entropy, creativity, or useful structure. A raw residual can
still prefer noisy-TV behavior.

## 6. Head-health inference limits

Win-count skew has two compatible explanations:

1. predictors pile up on the same output;
2. predictors remain separated but some win rarely.

Prediction spread separates those geometries, so a small spread plus skew is
evidence of pileup. A large spread plus skew only establishes
**separated-unresolved**. It does not establish that K exceeds the number of
modes, because head separation is not a support estimator.

The fused path must report `unresolved` wherever it lacks a head-spread probe.

## 7. Agree + Disagree

The requested replacement for the meaningless ordinary two-head blend is a
general-sum game:

```text
V_A = V(theta_A, phi_A)
V_B = V(theta_B, phi_B)

predictor A: min_phi_A V_A
predictor B: min_phi_B V_B
field A:    min_theta_A      -V_A
field B:    min_theta_B      +V_B

C_beta(x) = (1-beta) A(x) + beta B(x)
```

A tries to defeat its predictor. B cooperates with its predictor. The shipped
experiment uses two independent predictors, one per field lane—not one shared
discriminator. C is derived only; it has no optimizer and no direct loss.
A/B/C are rendered as exact R/G/B roles.

The kernel needs two isolated external-gradient lanes and predictor state for
both games. The discriminator step must occur before recomputing generator
gradients, otherwise both players use stale simultaneous gradients rather than
the intended alternating update.

Branch-isolation, sign, blend-independence, browser-startup and NaN-soak gates
all pass in the 2026-07-29 release verification.

## 8. Field architecture and chaos

The internal `HelmholtzField` compatibility class is two direct-vector MLPs:

```text
F = (1-alpha) A + alpha B.
```

Those outputs are not automatically `-grad(phi)` and `rot90(grad(psi))`.
Blending them is merely interpolation/pooling unless their objectives give the
lanes distinct roles. UI and docs must not call alpha an order/chaos axis.

Chaos is a loss:

```text
L_chaos = -log(||J_F||_F/sqrt(2) + epsilon)
```

implemented by finite-difference probes. Divergence and isotropy can be
composed with it:

```text
L_div  = E[(trace J_F)^2]
L_iso  = normalized anisotropy of batch force covariance
```

Maximum-chaos pieces explicitly omit the spiral objective. Chaos Weave is the
explicit chaos+adversary composition. Standalone adversary pieces have no
structural field loss, making the zero-sum game auditable.

Exact potential/curl construction:

```text
F = -grad(phi) + rot90(grad(psi)) + h
```

was implemented historically but required second-order differentiation through
training and measured roughly 800–1700 ms/frame in tfjs. It is rejected for the
interactive hot path, not rejected mathematically. A spectral Helmholtz
projection would require a grid FFT and remains unsuitable per particle.

## 9. Fused implementation and AD IR

Production uses hand-generated, specialized WGSL. The scalar AD IR provides an
independent reverse-mode oracle and has compiled test shaders, but is not the
production trainer compiler.

Consequences:

- AD IR versus hand-fused runtime speed has not been measured as two production
  paths; there is no honest speedup number for that comparison.
- the recorded speedup is fused WGSL versus tfjs;
- IR value/gradient parity improves correctness and maintainability, not
  automatically performance.

Previously measured B=512 adversary steps were 0.682 ms (pair K=4) and
0.726 ms (tri K=6), compared with 19–32 ms for the old tfjs app path. These
numbers are hardware/load/config specific, not a promise that every gallery
piece renders at 60 FPS.

## 10. Surprise rendering correctness

Per-tuple residuals must be attributed to particles without racing:

```text
tuple residual s_t -> every member of tuple t
```

Sampling with replacement and non-atomic writes permits multiple threads to
overwrite one particle nondeterministically. It also leaves untouched entries
stale. An honest implementation uses a rotating unique schedule (or a real
reduction), resets coverage on resize, and reports measured coverage.

Per-unit-signal coloring is a diagnostic:

```text
render_value = payoff / ||Y||
```

It prevents the colormap from being merely a magnitude meter. It is distinct
from the training payoff and must not silently alter the game.

## 11. Additional ideas preserved from the thread

### Compression progress

Reward learning progress rather than residual level:

```text
R_progress(t) = loss_D(t-window) - loss_D(t).
```

Unlearnable noise gives little progress, and learned regions become boring.
The value is attractive but progress is a two-timepoint controller, not a
straight differentiable pull on current field parameters. A practical hybrid
would use disagreement/WTA for gradient and progress to modulate its weight.
**Experimental.**

### Replay-buffer novelty discriminator

Train a discriminator to separate present rollouts from the piece's replayed
history. This asks “is current motion unlike what this piece has already done?”
rather than “is it predictable from this observer?” It is orthogonal to WTA but
introduces replay distribution and GAN-stability problems. **Experimental.**

### BALD / epistemic versus aleatoric

Predict mean and uncertainty per head, then estimate mixture entropy minus
expected component entropy. This matters for stochastic dynamics; the current
field is deterministic and symmetry quotienting deliberately creates the
conditional ambiguity. **Experimental, lower priority.**

### K-step temporal targets

Predict relational state after K physical steps, without reset between windows.
This rewards trajectory choreography instead of per-frame sizzle. It requires
BPTT through tuple trajectories and careful border/clamp Jacobians.
**Experimental.**

### Finite-time Lyapunov

The present chaos proxy is local. A true finite-time exponent uses products of
the actual tangent map:

```text
lambda_T = (1/T) log || product_t D Phi(x_t) ||.
```

Feasible with existing rollout machinery but aesthetically unmeasured.
**Experimental.**

### Exact JVP probes

Forward-mode JVP can remove finite-difference step-size error for divergence and
Jacobian norms. The IR supports the mathematical oracle, but production WGSL
and aesthetic A/B are still needed. **Experimental.**

### Spectrum and harmonic drift

A spectrum-slope objective and mean-force/harmonic penalty remain useful
orthogonal structural controls. Spectrum code exists as an axis-aligned proxy;
neither should be described as part of the strict adversarial game unless
explicitly composed. **Experimental.**

## 12. Verdict ledger

| idea | verdict |
|---|---|
| absolute-position ensemble variance | rejected: deterministic conditional |
| point single predictor | implemented as negative control |
| pair rotation observer | implemented |
| pair R+S raw | implemented as explicit cheat control |
| pair R+S angular adjusted | implemented and stability-soaked |
| unordered tri | implemented with ambiguous ties inactive |
| permutation-invariant generic quad | rejected |
| labelled quad | implemented and verified in reference, fused Metal and live UI |
| plain WTA epsilon=0 | retained only as experiment; known starvation |
| relaxed WTA | implemented with strict epsilon bound |
| different predictor/generator payoffs | rejected for the strict game |
| tape-constant RMS as scale proof | rejected; tuning stabilizer only |
| angular tangent payoff | active scale fix |
| `data-limited` diagnosis from spread | rejected; use separated-unresolved |
| two ordinary MLPs called Helmholtz/order/chaos | rejected terminology |
| explicit maximum-chaos loss without spiral | implemented |
| Agree+Disagree A/B with derived C | implemented; isolation and live soak verified |
| AD IR as production compiler | false; oracle only today |
| unique measured surprise coverage | implemented; 100% live coverage verified |
| compression progress | experimental |
| replay-buffer novelty GAN | experimental |
| BALD uncertainty | experimental |
| temporal K-step adversary | experimental |
| finite-time Lyapunov | experimental |
| exact JVP probes | experimental |

## 13. Release gates

All originally required gates are green as of 2026-07-29:

1. ✅ tfjs reference math and invariance suites;
2. ✅ AD-oracle and fused-gradient parity;
3. ✅ strict raw-`F(x)` wiring and parameter separation;
4. ✅ adjusted-mode zero radial gradient;
5. ✅ Agree+Disagree branch/sign/blend isolation;
6. ✅ race-free measured surprise coverage;
7. ✅ labelled-quad fused Metal support;
8. ✅ full WebGPU regression matrix and production build;
9. ✅ real-browser UI interaction at desktop and 390x844 mobile sizes;
10. ✅ 30-second-per-piece adversarial soak with no NaN, memory drift,
    pileup, or velocity-clip ratchet.

The exact commands and browser artifact paths are recorded in
`docs/ADVERSARY_STATUS.md` and the durable release note.
