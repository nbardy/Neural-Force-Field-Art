# Soft-angle / Pair WTA "learns nothing" / "all diagonal" regression

## Goal

Investigate user report that angle-based adversary (soft-angle /
angle-relative-scale / angle-scale-hold) used to make amazing swirls and now
"learns nothing" or goes "all diagonal". Focus: fused WGSL vs oracle, gallery
defaults, NaN hardening (89269ba), dock localStorage.

## How soft-angle is supposed to work (verified)

Exact embedding `ψ_τ(q)=(qx,qy,τ)/√(|q|²+τ²)` then softened chord
`√(|ψ(p)-ψ(y)|²+ε²)-ε`. Asymptotically direction-only for |q|≫τ; finite and
smooth at zero (Jacobian ≤ 1/τ). Generator zero-sum: minimize −A (maximize
surprise). Default τ=0.05. Oracle: `softAngleResidual` in
`src/core/gan/adversary.ts`. Fused: `softSphereEmbed`/`softSphereChord`/
hand-written `softSphereGradP`/`GradY` in `adversary_wgsl.ts`; AD-IR twin in
`ad/losses.ts` `softSphere2`/`wtaObjectiveTerm`.

## Gallery defaults today (verified)

| Piece | Encoding | Default loss (via `adversaryLossOf`) | G LR | D LR | weight |
|---|---|---|---|---|---|
| Pair WTA K=4 | `pair-rotation-scale-adjusted` | **soft-angle τ=0.05** | 0.001 | 3e-3 | 0.015 |
| Agree+Disagree | `pair-rotation-scale-adjusted` | **soft-angle τ=0.05** | 0.001 | 3e-3 | 0.012 |
| Single / WTA K=8 / Tri / Quad | point/tri/quad | **raw-vector** | 0.001 | 3e-3 | 0.01–0.012 |
| Chaos Weave | `pair-rotation` | **raw-vector** | 0.006 | 3e-3 | 0.006 |

`ADVERSARY_OBJECTIVE_DEFAULTS`: τ=0.05, scaleWeight=0.5, energyWeight=0.1,
energyTarget=0.35. Soft-angle `genSeed = weight/B` (no RMS EMA); raw-vector
divides by `max(rmsEma,1e-6)`.

`pair-rotation-scale-adjusted` is scale-blind context (`u=1`) + soft-angle loss
alias — it does **not** unit-normalize the target anymore (legacy-adjusted path
retired).

## 89269ba NaN guards vs soft-angle grads

Touched: SELU ±80 clamp (COMMON + advect); `isFiniteF` on raw resid, surprise,
sig/u/y, Adam, extGrads.

**Does not kill healthy soft-angle grads** in the intended sense:
- Soft-angle resid path was **not** given the new `isFiniteF(r)` gate (only
  raw-vector resid got it at ~1083–1085).
- Soft-angle disc/gen bwd use exact sphere grads, not `/resid`.
- ADV_SOFT_EPS2=1e-12 unchanged; τ still 0.05.

**Can still distort learning if NaNs recur:** zeroing nonfinite `sig`/`y` to 0
trains D against north-pole targets (`ψ(0)=(0,0,1)`), which is not the true
field — intermittent poison → collapse or saturation.

`tools/train_wta_test.ts` §3b still asserts soft-angle disc+gen cos≈1 vs tfjs;
Aug 5 "open disc grad bug" notes look **stale** relative to that gate.

## Dock localStorage — can lock bad settings? YES

Key `nffa.dock.v1` (`src/index.tsx`). Persists full runtime including
encoding/loss/K/LRs. `runtimeForPieceSwitch` **keeps** observer/loss across
gallery clicks — only refreshes adversary kind (+ K if stale). Cold start
without URL dock knobs restores last dock → Pair can run as **raw-vector**
(or Tri encoding) after browsing other pieces. No "reset to piece defaults"
button yet (`agent_notes/2026-08-06_145416_KST_dock_localstorage.md`).

## Ranked hypotheses

1. **Dock / piece-switch locked wrong loss or encoding** (highest for
   "regression" timing — dock landed same day). Symptom: objective row shows
   RAW VECTOR on Pair, or wrong tuple arity.
2. **Looking at Tri/Quad/Chaos Weave** which bake **raw-vector** — amplitude
   cheat + drive saturation → visually "all diagonal" (cf. Quad NaN note
   speed≈24√2 = both-axis clip).
3. **Soft-angle + drive saturation** — direction-only loss does not punish
   |F| once |F|≫τ; dual-head field saturates both components → diagonal
   trajectories dominate even with correct soft-angle.
4. **Intermittent NaN→zero target training** after 89269ba guards — less
   likely if HUD stays finite, but explains "learns nothing" if surprise≈0
   and heads collapse.
5. **89269ba eps/clamps killed angle grads** — **low**; math path largely
   untouched; parity tests still pass.

## Smallest next experiment

1. DevTools → clear `localStorage['nffa.dock.v1']` (or use private window).
2. Open Pair WTA with `?advLoss=soft-angle&advTau=0.05` (forces honest loss).
3. Confirm dock "ANGLE OPPONENT GAME" + HUD `force · soft-angle`.
4. 60s soak screenshot; then same URL with `?advLoss=raw-vector` A/B.
5. Optional: soft-angle + lower `?drive=0.35` — if diagonals vanish, suspect
   magnitude saturation under direction-only loss, not a broken chord grad.

## Unresolved

- No live soak run in this investigation pass.
- Whether user was on Pair (soft-angle) vs Tri/Quad (raw) is unknown.
