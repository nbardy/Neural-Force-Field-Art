# Design space — coordinate-network particle art

The conceptual map for future pieces now that the fused-kernel platform exists.
Companion to docs/PLAN_FUSED_TRAIN.md, PLAN_SPECIES_AND_BATCHES.md,
MATH_ANALYTIC_GRADIENTS.md, and the two QA_FUSED_KERNELS rounds. This doc
captures the design axes and the reasoning worked out with the maintainer — it
is a map, not a plan; each piece recipe (§6) is a point in the space.

**Design principle (maintainer's):** creative tools expose REAL degrees of
freedom grounded in theory — never arbitrary knobs. Every axis below has a
"why" (a theory) and a visible artistic consequence. And new model/loss types
ship as SELECTABLE OPTIONS in the bottom panel to compare against each other,
not as silent "upgrades" that replace what came before.

---

## 1. The platform (what exists)

Coordinate-network force fields `F(pos[, class])` evaluated per-particle in one
fused WGSL dispatch (advect); an analytic-BPTT trainer (2 dispatches) with
K-step rollouts on live particle states; per-species conditioning via a one-hot
class (storage-free index hash); a compute-splat renderer with decay trails,
per-class channels, radial dots, retina/dpr, and a tonemap grading stage; a
headless verification loop on real Metal (bun-webgpu) with tfjs-autograd
oracles. Headline: 1M particles @ 60 FPS (non-retina), all knobs live. Every
axis below plugs into this without disturbing the verified core.

---

## 2. The MODEL axis (coordinate networks — the current frontier)

**What it is.** The field is a coordinate network (implicit neural
representation): input is literally `[x, y]` (plus the class one-hot), each
evaluation is independent, and the "very wide batch" is millions of independent
point queries. That independence is why the per-thread kernel works — and why
some architectures don't apply (below).

**Current model** (`src/core/field/helmholtz.ts`): plain MLP on raw `[x,y]`,
SELU hidden layers, tanh output; two heads (order `g`, chaos `r`) blended by α.

**Its known limitation — SPECTRAL BIAS.** Plain coordinate MLPs can only
express smooth, low-frequency functions; a raw-`[x,y]` MLP literally cannot
represent fine spatial detail. That is why the current fields look like soft
dunes. Buying detail by making the MLP bigger/deeper (the `mlpDeep` route) is
the WORST lever — `mlpDeep` costs ~33k MACs/particle vs the Helmholtz field's
~4.6k (14×) for only modest gains against spectral bias, and it doesn't fuse to
the f16 fast path (too many MACs to unroll). Prefer the input-encoding fixes.

### 2a. Fourier feature encoding — FIRST RECOMMENDATION
Replace the raw input with `γ(p) = [x, y, sin(ωₖx), cos(ωₖx), sin(ωₖy),
cos(ωₖy), …]` over geometric frequencies `ωₖ = 2ᵏ·ω₀` (keep the raw coords
too). Fixed preprocessing — NO parameters.
- **Why the sin/cos PAIR per frequency:** `sin(ωx+φ) = a·sin(ωx) + b·cos(ωx)`,
  so supplying both lets the first dense layer learn ANY phase offset as a
  linear combination — the offsets come free, already differentiable, inside
  the existing weights. (Maintainer: "cos is basically an offset" — exactly;
  cos IS the offset basis, so no learnable phase term is needed.)
- **Backward:** the encoding has no weights to train; but the two places we
  need INPUT-gradients (the probe-loss backward and the BPTT position chain)
  chain through `d sin(ωx)/dx = ω·cos(ωx)` — one extra line of backward
  codegen, fully analytic.
- **Artistic dial:** the base frequency `ω₀` is a real degree of freedom —
  low ω₀ = flowing dunes, high ω₀ = intricate filigree. Number of octaves
  trades detail vs compute.

### 2b. SIREN (sinusoidal activations)
Replace SELU with `sin(ω₀·Wx)` throughout. Known for detailed fields with
well-defined, smooth HIGHER derivatives.
- **Why that matters HERE specifically:** the chaos and divergence loss terms
  are finite-difference PROBES of the field's spatial derivatives — they
  evaluate `F` at `p`, `p+[h,0]`, `p+[0,h]` and difference them (chaos ≈ local
  Jacobian magnitude, divergence ≈ ∇·F). So the loss literally optimizes
  `∂F/∂position`. ReLU-family activations (SELU on its positive side) give
  PIECEWISE Jacobians → a kinky, noisy probe signal; sinusoidal activations
  make `∂F/∂pos` (and second derivatives) smooth, so the very quantities the
  loss measures become clean. SIREN is unusually well-matched to *this*
  objective.
- **Caveats:** needs its specific init scheme and `ω₀` tuning; ship as an
  option, tune per-piece.

### 2c. Hash-grid features (Instant-NGP)
Learned multiresolution feature tables + a tiny MLP: `feat(p) = interp over the
surrounding grid cells of learned vectors Tᵢ`, then a small MLP on `feat`.
- **"Isn't a lookup non-differentiable?"** (maintainer's question) — the lookup
  is INTERPOLATED, and you differentiate the interpolation, not the indexing.
  Two paths:
  - w.r.t. the table entries `Tᵢ` (the trained parameters): `∂feat/∂Tᵢ = wᵢ`
    (the interpolation weight). Fully differentiable; backward = scatter
    `grad·wᵢ` into the corner cells with atomics — the SAME machinery the splat
    renderer already uses, and mathematically IDENTICAL to transformer token
    embeddings (gather on forward, scatter-add on backward; the discrete
    "which cell" never needed a derivative).
  - w.r.t. position (what the probes need): `∂feat/∂p = Σ ∇wᵢ·Tᵢ` — under
    BILINEAR interpolation this is piecewise-constant (ReLU-shaped), which is
    exactly what the probe losses dislike. Fixable with smoothstep/cubic
    interpolation.
- **Tradeoff:** best detail-per-FLOP known for coordinate nets, but the field
  becomes "painted texture" rather than "equation" — an artistic character
  change, and real engineering (grid storage + atomic backward). Do last.

### 2d. What does NOT apply, and why (maintainer's correct instincts)
- **Attention:** each field eval is an independent point — there is no set or
  sequence to attend over WITHIN an evaluation. (Attention across PARTICLES is
  a different object — see §3, N-body.)
- **Convolution:** no grid to slide a window over — the field is a function,
  not an image. (Adopting hash-grids §2c is what CREATES a griddable thing.)

### 2e. Activation reference
Current: SELU (hidden), tanh / sigmoid (output). Ranked by derivative
smoothness for probe-loss quality: sin (SIREN) > tanh / Gaussian > SELU/ELU >
ReLU. Output activation stays bounded (tanh) so `forceMagnitude` reads
consistently.

---

## 3. The DYNAMICS axis

**Current physics:** `vel += F·mag`, friction, velocity clip, floored wrap,
PCG per-particle resets; K-step BPTT training (`?rollout`, `?window` ≡
trajectory-window, proven equal to the real advected trajectory to 6e-5 px).

**Extensions with theory:**
- `F(pos, t)` time-conditioned field — animated flows; feed `t` as an extra
  input (Fourier-encode it for periodicity). Turns a static field into a
  living one.
- `F(pos, vel)` — this makes the object a per-particle CONTROLLER, not a field
  (same place → different force depending on motion); it also muddies the probe
  semantics (what velocity would a finite-difference probe carry?). Reserved
  branch, not a default.
- A second field head predicting per-position friction / mass — spatially
  varying "material."
- α (order↔chaos) as an INPUT to a single conditioned net rather than a blend
  of two nets.

**N-BODY / swarm direction (maintainer's stated future).** Particles
attending to / repelling neighbors = learned interaction kernels. This turns
the art from a "field" into a "society."
- Naive is O(N²); the WebGPU-practical route is grid-binned neighborhoods
  (spatial hash — the SAME binning idea as the splat accumulation buffer).
- A permutation-invariant SUM over neighbors of an interaction MLP on
  `[Δpos, Δvel]` is the attention-analog that actually fits here (order-
  independent, differentiable, fuses).
- The fused-kernel + analytic-backward playbook still applies; this is its own
  phase (needs the spatial binning first).

---

## 4. The OBJECTIVE axis (losses as art direction)

**Current composite** (`helmholtzChaosLoss` in src/main.ts): chaos (−log local
sensitivity) + isotropy (force-covariance balance, class-blind by design) +
0.5·divergence penalty + a faint spiral structural anchor. Each term is a
theory-grounded dial.

**Space to explore:**
- Per-class objectives — species with different temperaments (one seeks order,
  one seeks chaos); the class-conditioned chaos head already supports this.
- Inter-species terms — predator/prey: class A's field repels from class B's
  DENSITY. This needs a density INPUT, i.e. feed the splat accumulation buffer
  back in as a field input (closes the loop; see §5).
- Curl / vorticity targets — drive the field toward rotational structure.
- Spectral targets — push the field's Fourier spectrum toward pink/blue noise
  for texture control (pairs naturally with Fourier features, §2a).
- Flow-matching to a reference image — the splat buffer vs a target image is a
  differentiable loss because the WHOLE chain (splat → field → weights) is
  differentiable. Paint with particles.

**Hazard to respect:** the field shapes the cloud and the cloud trains the
field — a mode-seeking feedback loop (the GAN failure mode). Dampers already
exist: the reset slider and `mixRandom` (the random-batch coverage floor).

---

## 5. The RENDERING axis

The splat accumulation buffer is an HDR canvas, not just a display: per-class
channels, a tonemap grading stage (arbitrary palettes / curves per channel),
decay trails. Future:
- Velocity-aligned ANISOTROPIC splats (streak kernels) — motion blur / ink
  strokes instead of round dots.
- Density → hue mappings in the tonemap.
- Feed the accumulation buffer BACK as a field input — the art sees itself
  (also enables the inter-species density terms in §4).

---

## 6. Piece recipes (concrete next pieces)

Each combines axes above. All ship as selectable gallery entries to compare.

- **Filigree** — Fourier features, high ω₀ + more octaves, chaos weight up.
  Theory: beat spectral bias to expose fine spatial structure. Look: intricate
  lace instead of dunes.
- **Predator / Prey** — 2 classes + an inter-species density-repulsion loss
  (needs the splat buffer as field input). Theory: coupled objectives over
  species. Look: one species carves voids the other flees.
- **Tide** — time-conditioned `F(pos, t)`, slow `t`. Theory: animated field.
  Look: a slowly breathing flow that never settles.
- **Ink** — high decay (long trails) + velocity-anisotropic streak splats +
  divergence-negative sinks. Theory: trails + directional deposition + inflow.
  Look: ink diffusing into water.
- **Murmuration** — N-body neighbor interaction kernel (grid-binned). Theory:
  learned local interaction, permutation-invariant. Look: flocking, not field.
- **SIREN vs MLP (A/B piece)** — the SAME loss/config on a SIREN field and a
  SELU-MLP field, side by side as two gallery entries. Theory: make the
  spectral-bias difference VISIBLE by comparison, per the maintainer's "options
  to compare, not pure upgrades" principle.

---

## 7. Order of work (recommendation)

1. **Fourier features** — biggest art-per-effort: a codegen input-encoding
   change that extends the existing oracle fixtures; directly attacks the
   visible smoothness limitation. Ship the encoded field as a NEW selectable
   model type alongside the plain MLP (compare, don't replace).
2. **SIREN** experiment — as another selectable model type; verify derivative
   smoothness helps the probe losses.
3. **Time-conditioning** — `F(pos, t)`.
4. **N-body** — its own phase (spatial binning first).
5. **Hash-grid** — last (engineering-heavy, changes character).

**Cross-cutting requirement (maintainer):** model type and loss type must be
SELECTABLE in the bottom panel — every new lane is a comparison option, not a
silent upgrade. This means the gallery/config needs a `modelType` /
`fieldEncoding` dimension (plain | fourier | siren | hashgrid) and the codegen
must dispatch on it, the same way it already dispatches on `classes`.
