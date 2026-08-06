# Spiral cover loss — derivation & naming

**Status:** v3 fused cover shipped (`W_COVER` in train_wgsl + COVER_FIELD_LOSS on gallery Cover pieces)
**Shipped:** `spiralCoverLoss` = curve→particles only; samples are **arc-length
uniform** (skip inner 8% arc) and scaled by `maxR²`. Gallery **Spiral Cover ·
Clean / Ghost**.  
**Bug (v1):** equal-θ samples pack near the origin; combined with radial `L↓`
(also soft at the hub) and raw pixel² `L↑`, mass collapsed into a center cross.  
**Trigger:** Galaxy pieces use one-sided particle→spiral distance; particles can
collapse to a single locus on the curve and still drive loss → 0.

## What the user asked for

Not just “how far is each particle from the spiral,” but also “how empty is
each part of the spiral” — i.e. for sample points along the curve, distance to
the nearest particle. Without the second term, clustering on one spiral spot is
optimal.

## Verified current objective

Archimedean spiral `r = b·θ`, `θ ∈ [0, Θ]`, `Θ = SPIRAL_TURNS·2π`,
`b = maxR/Θ`, `maxR = min(w,h)·0.38`.

Current Galaxy loss (`spiralPlusCenterLoss`):

```
L↓ = (1/N) Σ_i min_k (r_i − b·relu(φ_i + 2πk))²
   + ε_center · mean_i ‖p_i − center‖²
```

Facts:

- This is a **radial residual after branch unwrap**, not full Euclidean nearest-
  point-on-curve distance (cheap approximation; already in fused WGSL).
- Gradient only exists for the winning branch `k*` (same WTA as §4.4 of
  `docs/MATH_ANALYTIC_GRADIENTS.md`).
- **Degeneracy:** any configuration with all mass at one spiral point has
  `L↓ ≈ 0`. Coverage of the filament is unconstrained.

## Formal target

Let `γ(θ) = (cx + bθ cos θ, cy + bθ sin θ)` for `θ ∈ [0, Θ]`.
Discretize with `M` samples `s_m = γ(θ_m)`, `θ_m = Θ·(m+½)/M`.

We want a **bidirectional** match:

| Direction | Meaning | Fixes |
|---|---|---|
| particle → curve (`L↓`) | stay on the filament | off-spiral drift |
| curve → particles (`L↑`) | every arc segment has nearby mass | single-spot collapse |

This is the discrete **Chamfer distance** between the particle set and the
spiral polyline (plus optional soft variants).

---

## Candidate A — Spiral Chamfer (literal reading)

```
L↓_eucl = (1/N) Σ_i  min_m ‖p_i − s_m‖²
L↑      = (1/M) Σ_m  min_i ‖s_m − p_i‖²
L       = α L↓_eucl + β L↑
```

Or keep the existing cheap radial `L↓` and only add `L↑`:

```
L = α L↓_radial + β L↑ + ε_center L_center
```

### Differentiability

`min` is differentiable almost everywhere; reverse mode routes to the unique
argmin (ties: pick first, same contract as current spiral WTA). Same class of
non-smoothness already shipped.

**Softmin** (fully C∞):

```
d̃(a, S) = −τ log Σ_{x∈S} exp(−‖a−x‖² / τ)
L↑_soft = (1/M) Σ_m d̃(s_m, {p_i})
```

As `τ → 0`, softmin → hard min. Softmin gives dense gradients (every particle
feels every sample); hard min is sparser and often stabler for art.

### Complexity

| N (particles) | M (curve samples) | ops |
|---|---|---|
| 1 500 (Galaxy) | 256–512 | ~0.4–0.8M — trivial |
| 200 000 (Max Chaos) | 256 | ~50M — OK if rare; not free |

**Proposal:** use Chamfer for Galaxy-scale pieces; do not bolt it onto 200k
chaos pieces without tiling / a different cover term.

### Analytic seed for hard `L↑` (one sample `s`, winner particle `i*`)

```
∂L↑/∂p_{i*} = (β/M) · 2 (p_{i*} − s)
∂L↑/∂p_{j≠i*} = 0
```

So empty spiral loci pull the **nearest** particle toward them. That is exactly
the anti-clustering force requested.

For softmin, every particle gets

```
∂d̃/∂p_i = 2 (p_i − s) · softmax_i(−‖p_i−s‖²/τ)
```

---

## Candidate B — Soft arc occupancy (fast at large N)

Project each particle with the existing WTA unwrap → `θ_i*` (already computed
for radial spiral). Soft-assign to `B` bins on `[0, Θ]`:

```
w_{i,b} = softmax_b( −(θ_i* − θ_b)² / τ )
mass_b  = Σ_i w_{i,b}
ρ_b     = mass_b / N
L_occ   = Σ_b (ρ_b − 1/B)²          # or KL(U ‖ ρ), or −Σ log(ρ_b+ε)
```

Cost `O(NB)` with `B ~ 64…256`, independent of a second spatial min. Gradients
flow through soft weights and through `θ_i*(p_i)`.

Caveats:

- Only penalizes **along-curve** gaps; off-curve particles still need `L↓`.
- Projection discontinuity at branch boundaries (same as current spiral).
- Hard histograms are almost nowhere differentiable in bin membership — **must
  stay soft**.

Good as a fused-path cover term when `N` is huge.

---

## Candidate C — RBF density along the curve

```
dens_m = Σ_i exp(−‖s_m − p_i‖² / (2σ²))
L_rbf  = (1/M) Σ_m 1/(dens_m + ε)     # or (τ − dens_m)_+²
```

Fully smooth, `O(NM)`, always-dense gradients. Easy to over-smooth or fight
`L↓`. Prefer Chamfer/softmin unless we need C∞ everywhere.

---

## Recommendation

**Ship Candidate A for Galaxy:** keep cheap radial `L↓` (or upgrade to Euclidean
Chamfer ↓ if we want consistency), add hard or lightly-soft `L↑` over `M≈256`
curve samples.

**Name (math / API):** `spiralCoverLoss` / field weight `W_SPIRAL_COVER`

**Name (gallery):** replace “Galaxy · …” framing that implies one-sided spiral
distance with something that signals fill:

| Option | Tone |
|---|---|
| **Spiral Cover** (preferred) | exact: cover the curve |
| Filament Cover | more poetic |
| Spiral Chamfer | math-literal, colder |
| Galaxy Filament | keeps Galaxy brand, implies extended structure |

Preferred product name: **Spiral Cover**  
Preferred loss symbol: `L_cover = α L↓ + β L↑`

Suggested default weights (hypothesis — needs live tuning):

- `α = 1` (existing radial scale)
- `β ≈ 1…10` relative to mean nearest-neighbor scale on canvas
- keep tiny center weight or drop it once `L↑` anchors the core

## Proof sketch: why clustering fails under L↑

Let all `p_i = s_*` for one curve sample, and let the farthest sample be
`s_far` with ‖s_far − s_*‖ = D > 0.

Then `L↓ ≈ 0` but

```
L↑ ≥ (1/M) · D²
```

(at least the one empty sample contributes `D²`; typically Θ(1) fraction of
samples are far). Any redistribution that places mass near empty loci strictly
decreases `L↑` while keeping `L↓` small if mass stays on-curve. Hence single-
spot optima of the old loss are **not** optima of `L_cover`.

## Fused cover (v3)

- `FieldLossSpec.W_COVER` / `COVER_SAMPLES` (default 256)
- Arc-length offsets baked into WGSL at codegen; loss in `finalize`, grads in `bwd`
- Spiral Cover gallery pieces set `fieldLoss: COVER_FIELD_LOSS` → fused trainer
- tfjs `spiralCoverLoss` remains for `?train=tfjs`

## Open decisions (remaining)

1. ~~Hard min vs softmin~~ → hard min
2. ~~Bidirectional vs cover-only~~ → cover-only (v2)
3. ~~Equal-θ vs arc-length~~ → arc-length + skip inner 8% (v2)
4. Live-watch filament fill; retune `β` / `SPIRAL_COVER_ARC_SKIP` if needed
5. Optional fused WGSL later

## Next actions

- [x] Cover-only + arc-length fix after hub/cross collapse
- [ ] Re-select Spiral Cover after reload so the MLP re-inits
- [ ] Watch live vs Galaxy · Clean
- [ ] Optional: fused WGSL / unit oracle


## Status (2026-08-06)

Shipped: cover-only Chamfer ↑, arc-length samples, fused `W_COVER`, oracle PASS,
gallery Clean/Ghost/Fourier/SIREN/HashGrid/Fourier+SIREN. Softmin still deferred.
