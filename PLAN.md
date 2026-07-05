# Neural Force Field Art — GPU GAN Engine

## Execution Plan

This is an end-to-end roadmap to transform the live particle-art engine into a learned Helmholtz force field with GAN-driven objectives and zero-copy GPU rendering.

**Current state**: Live engine (main.ts + renderers.ts) runs physics simulation inside gradient tape, minimizes predefined losses (spiral, center). Single MLP predicts 2D force vectors. CPU renders via arraySync→Canvas2D.

**Target**: Learned decomposition F=(1-α)∇φ + α·curl(ψ) where φ is a potential (ScoreNetwork) and ψ is a vorticity stream (learned angular field). Alpha blends chaos↔order. GAN losses reward structured predictive information instead of predictability-minimization (which only breeds noise). GPU render via WebGPU instanced points, zero-copy from tfjs tensors.

---

## Lane 0: Repo Consolidation

**Goal**: Clean branch topology. main is ancestor of feat/helmholtz-gan-engine; prepare for fast-forward.

**Tasks**:
- [ ] Verify main is ancestor of HEAD: `git merge-base --is-ancestor main HEAD` ✓
- [ ] Verify no uncommitted changes: `git status`
- [ ] If feature branch has new commits beyond main, squash or rebase onto main's HEAD
- [ ] Once stable: `git checkout main && git merge --ff-only feat/helmholtz-gan-engine`
- [ ] Delete obsolete cursor branches if any

**Outcome**: Single clean main branch with lane work branching off.

---

## Lane 1: GPU Zero-Copy Renderer

**Blocker**: pos.arraySync() each frame = GPU→CPU stall. Kills performance on large particle counts.

**Goal**: Render 100k+ particles directly from tfjs tensors without readback.

**Strategy**:
1. **Data upload (one-time)**: Allocate WebGPU buffer, create WGSL compute shader to unpack [N,2] pos tensor → instanced point data (x,y,size,color).
2. **Bind to twgl**: Use twgl.js `setBindings()` to attach WebGPU buffer as attribute stream.
3. **Instance draw**: `gl.drawArraysInstanced()` calls one tiny quad-vert shader, positions N particles via instance index.
4. **Implement**: `src/renderers/webgpu-instanced.ts`
   - Export `GPUInstancedRenderer(cfg, particleCount, spiralPts)` implementing Renderer interface
   - Keep alpha-fade/trail-buffer/clean logic in CPU branch selection
   - In GPU-aware variant: accumulate pos tensors on GPU side, read back sparse keyframes only (every 60 frames for UI stats)

**Files**:
- Create: `src/gpu/dataToGPU.ts` — convert tfjs Tensor2D → WebGPU buffer + descriptor
- Create: `src/gpu/shaders.wgsl` — compute shader: unpack pos→instanced attrs
- Create: `src/renderers/gpu-instanced.ts` — Renderer impl using twgl + WebGPU buffers
- Modify: `src/renderers.ts` — add "gpu-instanced" variant to createRenderer switch

**Test**: Bench main.ts with 50k particles, measure frame time before/after.

---

## Lane 2: Helmholtz Field (Score + Curl)

**Goal**: Replace MLP-direct with decomposition F = (1-α)∇φ + α·curl(ψ), α ∈ [0,1].

**Semantics**:
- **φ** (potential, α→0): smooth gradient flow, particles cluster at minima. Learned via score matching.
- **ψ** (stream/vorticity, α→1): curl generates circulation, particles orbit/swirl without dissipation.
- **α** slider: order (α=0, particles settle) ↔ chaos (α=1, organized vortex).

**Architecture**:
```
ScoreNetwork(pos) → [2] score vector s (reuse existing code)
AngularField(pos) → scalar ψ (new MLP: pos→ψ)

F(pos, α) = (1-α)·∇[∇log p(pos)] + α·[∇ψ^⊥]
          = (1-α)·s + α·[−ψ_y, ψ_x]
```

**Implementation**: `src/models/HelmholtzField.ts`
```typescript
class HelmholtzField {
  scoreNet: ScoreNetwork;
  angularNet: tf.Sequential;  // pos[N,2] → scalar[N,1]
  
  predict(pos: Tensor2D, alpha: number): Tensor2D {
    // Compute ∇φ via score net
    const score = this.scoreNet.predict(pos);
    
    // Compute ψ and curl via autograd
    const psi = this.angularNet.predict(pos);  // [N,1]
    
    // curl(ψ) = [-∂ψ/∂y, ∂ψ/∂x]
    // Use tf.grad on angular net's output to get ∂ψ/∂x, ∂ψ/∂y
    const curl = computeCurl(pos, psi);  // [N,2]
    
    // Blend: F = (1-α)·score + α·curl
    return score.mul(1-alpha).add(curl.mul(alpha));
  }
}
```

**Key detail**: computeCurl must stay differentiable. Use tf.grad(net, wrt=[pos]).

**Integrate into physicsForward**: Replace `model.predict()` with `helmholtzField.predict(pos, config.alpha)`.

**Config**: Add to ArtPieceConfig:
```typescript
interface ArtPieceConfig {
  // ... existing
  helmholtzAlpha?: number;  // 0=pure order, 1=pure chaos
}
```

**Gallery examples**:
- "Spiral Vortex" (α=0.3): Score net pulls to spiral, weak curl adds swirl.
- "Pure Chaos" (α=0.9): Angular noise field dominates, organized structures emerge.

---

## Lane 3: GAN Objectives (Chaos/Isotropy/Spectral/RND)

**Key insight**: Minimizing predictability → pure noise (random walk). Instead, **maximize predictive information**: the neural network should learn patterns that are *structured but not predictable to simple models*. This is the inverse of noise minimization.

**Four Loss Objectives** (to minimize or maximize in different configurations):

### 3.1 Lyapunov / Chaos Loss
Measures exponential divergence of nearby trajectories. High chaos = particles starting close end up far apart.
```
L_lyap = -mean(||Δx(t+1) / Δx(t)||) → maximize divergence
```
- Compute via finite differences: perturb pos by ε, integrate both, measure ratio.
- **Effect**: Encourages sensitive dependence; wild but structured.

### 3.2 Isotropy Loss  
Covariance of force vectors should be ~isotropic (trace[F F^T] ∝ I). Penalize preferred directions.
```
L_iso = variance(eigenvalues([<FF^T>]))
```
- Compute force covariance matrix: Cov = (1/N) ∑_i F_i ⊗ F_i.
- Penalize anisotropy: if one eigenvalue >> others, loss is high.
- **Effect**: Prevents degenerate behaviors (all particles pushed same direction).

### 3.3 Spectral (Power Law) Loss
Radial power spectrum E(k) should follow E(k) ~ k^(-5/3) (Kolmogorov turbulence).
```
L_spec = || log E(k) + 5/3·log(k) ||²
```
- Compute via FFT of position field (grid particles on 2D grid, FFT each axis).
- Radial bins in Fourier space.
- **Effect**: Rewards scale-invariant, self-similar structure (like real fluid turbulence).

### 3.4 RND (Random Network Distillation)
Ensemble of frozen random networks predict pos → high loss = surprising to all networks.
```
L_rnd = mean_j || predictor_j(pos_{t+1}) - target_frozen_j(pos_t) ||²
```
- Pre-train fixed random predictors (no gradient).
- Train learnable predictors to match target on old data.
- Minimize (predictor(new) - frozen(new))² to find regions network hasn't seen.
- **Effect**: Exploration: find novel dynamics, avoid local optima.

### 3.5 Divergence Penalty (Measure Preservation)
Force field should have low divergence to preserve particle density (incompressible flow).
```
L_div = mean(||∇·F||)
```
- Compute via autodiff: F = [f_x, f_y], div = ∂f_x/∂x + ∂f_y/∂y.
- Penalize non-zero divergence.
- **Effect**: Swirling motion, vortex stability.

**Composite Loss**:
```
L_total = λ_lyap·L_lyap - λ_rnd·L_rnd + λ_iso·L_iso + λ_spec·L_spec + λ_div·L_div
```
(negative RND because we maximize surprise; others minimized)

**Implement**: `src/objectives/ganLosses.ts`
```typescript
export function lyapunovLoss(pos, vel, model, cfg, w, h): tf.Scalar;
export function isotropyLoss(forces: Tensor2D): tf.Scalar;
export function spectralLoss(posGrid: Tensor2D, w, h): tf.Scalar;
export function rndLoss(pos, predictors, targetNetworks): tf.Scalar;
export function divergenceLoss(pos, model): tf.Scalar;
```

**Gallery**: Create pieces that blend different objectives:
- "Chaos Maximalist" (λ_lyap=1, λ_iso=0.5): Wild, fractal-like.
- "Turbulent Flow" (λ_spec=1, λ_div=0.8): Recognizable vortices, energy cascade.
- "Exploration" (λ_rnd=1): Continuously discovers new patterns.

---

## Lane 4: App Controls & Gallery

**Goal**: UI sliders for α (order↔chaos), loss weights, and playback.

**Controls**:
1. **Mode selector**: Switch between loss objectives (or blend them).
2. **Helmholtz α slider**: 0 (order) ↔ 1 (chaos).
3. **Learning rate / iteration pace**: Slow down to observe.
4. **Reset** button: Respawn particles, zero model weights.
5. **Record/export**: Save canvas to video or image sequence.

**Gallery update** (src/main.ts GALLERY):
```typescript
{
  name: "Helmholtz Vortex",
  helmholtzAlpha: 0.4,
  computeLoss: blendedLoss([
    (w) => lyapunovLoss(...),
    (w) => divergenceLoss(...),
  ], [0.6, 0.4]),
  // ...
}
```

**UI layout** (src/index.tsx):
- Top bar: Gallery buttons (existing).
- Right sidebar (new): Helmholtz α, loss weights, FPS counter.
- Bottom: Status (frame, active piece, learning rate).

---

## Lane 5: TrashPanda Triage

**Current**: ~2028 lines, mostly unrelated (vision, NLP, transformers).

**Action**:
- **Keep**: ScoreNetwork, linalg.ts (normalize), kernels.ts (gaussian).
- **Quarantine** (docs/EXPERIMENTAL.md): embeddings/, models/ (clipTransformer, mobilevit, rotaryEmbeddingTransformer), layers/, train/ (gpt, fastTransformer), experimental/.
- **Delete**: agentSets/ (transformer experiments, not used), quickDraw/ (GPU-renderer concept, superseded by Lane 1), draw/ (webgpu stub imports missing deps), src/physics/updateParticles.ts (old physics, main.ts has its own).

**Rationale**: These were exploration artifacts. Lane 1 (instanced GPU render) replaces quickDraw concept; Lane 2 (Helmholtz) replaces NLP/vision transformer tangents.

---

## Delete / Merge / Keep Table

| Path | Status | Reason |
|------|--------|--------|
| src/main.ts | **KEEP** | Live engine, core loop. |
| src/index.tsx | **KEEP** | React entry, controls. |
| src/renderers.ts | **KEEP** | Renderer interface & impls (alpha-fade, etc). |
| src/trashPanda/models/ScoreNetwork.ts | **KEEP** | Used in Lane 2 (Helmholtz potential). |
| src/trashPanda/linalg.ts | **KEEP** | normalize() may be used in objectives. |
| src/trashPanda/utils/kernels.ts | **KEEP** | Gaussian for spectral loss windows. |
| src/trashPanda/embeddings/ | **QUARANTINE** | Vision/NLP embeddings, unrelated. |
| src/trashPanda/models/clipTransformer.ts | **QUARANTINE** | Vision transformer, not for force fields. |
| src/trashPanda/models/mobilevit.ts | **QUARANTINE** | Same. |
| src/trashPanda/models/rotaryEmbeddingTransformer.ts | **QUARANTINE** | Same. |
| src/trashPanda/layers/ | **QUARANTINE** | Attention layers for NLP/vision. |
| src/trashPanda/train/gpt.ts | **QUARANTINE** | NLP training. |
| src/trashPanda/experimental/ | **QUARANTINE** | Speculative work, no active use. |
| src/agentSets/ | **DELETE** | Transformer agent sets, orthogonal to particle art. |
| src/quickDraw/ | **QUARANTINE→DELETE** | WebGL zero-copy renderer concept, superseded by Lane 1 (WebGPU instanced). |
| src/draw/ | **QUARANTINE** | draw_webgpu.ts imports @swissgl/swissgl2 (unpublished). draw_canvas2d is used? Check. |
| src/physics/updateParticles.ts | **DELETE** | Old physics engine; main.ts has inline physics. |
| src/types/all.ts | **AUDIT** | May contain unrelated type defs; triage by usage. |
| src/utils/* | **KEEP** | color.ts, math.ts, etc. may be reused; audit on use. |

---

## Execution Order

1. **Lane 0**: Branch cleanup (1–2 commits).
2. **Lane 1**: GPU renderer (3–5 commits, parallel with design).
3. **Lane 2**: Helmholtz field (2–4 commits, depends on Lane 1 for testing).
4. **Lane 3**: GAN objectives (4–6 commits, interleaved gallery examples).
5. **Lane 4**: App controls (2–3 commits, gating: Lane 2 + 3).
6. **Lane 5**: Triage & cleanup (1–2 commits, last; document in docs/EXPERIMENTAL.md).

---

## Key Design Principles

### Predictability ≠ Interest

**Anti-pattern**: Minimize sum-squared error → network learns to predict particle motion → loss bottoms out → no learning happens. Or: minimize predictability → network outputs random noise → Brownian motion → uninteresting.

**Better**: Maximize *reducible surprise*. The network should learn patterns that:
- Are structured (not random noise) → low spectral high-k power, low variance.
- Cannot be predicted by simple models → RND loss high.
- Have interesting dynamics → Lyapunov positive, isotropic forces.

This is the sweet spot: **organized chaos**. The GAN losses (Lyapunov + RND + isotropy) steer toward it; the Helmholtz decomposition gives the network the expressive degrees of freedom.

### Helmholtz Decomposition as Inductive Bias

Every 2D vector field decomposes as F = ∇φ + ∇×ψ (curl-free + divergence-free). By separating:
- **Potential φ** captures attraction/repulsion (score matching, density-seeking).
- **Vorticity ψ** captures circulation (energy-preserving, organized rotation).

We get a richer parameter space without adding model size, and the α slider gives a principled way to interpolate between *settling* and *swirling*.

---

## Success Metrics

- Frame rate ≥ 60 FPS with 50k particles (Lane 1).
- Loss converges to a moving equilibrium (not a fixed point, not random) (Lanes 2–3).
- Gallery shows 5+ distinct visual styles (Lane 4).
- docs/EXPERIMENTAL.md catalog complete, trashPanda pruned by 80%+ (Lane 5).

---

## Open Questions / Future Work

- **Higher-order**: Can we learn the time derivative of α? (e.g., α(t) = trained network → drift between order and chaos).
- **Multi-agent**: Ensemble of Helmholtz fields competing (game-theoretic loss).
- **Video codec**: Real-time encode to H.264, stream to browser for recording.
- **Interpretability**: Visualize φ and ψ fields separately (via 2D heatmaps overlaid).
