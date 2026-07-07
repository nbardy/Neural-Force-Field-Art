# Spec — CLIP vision backward: dL/dpixels in fused WGSL

Goal: given the verified forward (tools/clip/README.md — READ IT FIRST), add a
hand-written backward pass producing the loss gradient **w.r.t. the input
image only** (all weights FROZEN — no dW anywhere, which removes ~all of the
usual training-kernel complexity). This gradient feeds the Gaussian-splat
rasterizer backward so splats can optimize toward a text prompt.

Read these before writing code (patterns to copy, not re-derive):
- `tools/clip/README.md` — pipeline, gotchas (pinned input slot, warmup, Node
  buffer pooling), file map.
- `tools/clip/compile_plan.py` (κ) — the ONLY place ONNX is understood; you
  will extend it. Loud failure on unmatched structure; no silent fallbacks.
- `src/clip/vision_wgsl.ts` — codegen style: pure functions, every
  shape/offset a baked literal, one emitter per step kind, `assertStep` loud.
- `tools/clip/fused_test.ts` + `tools/kernel_test.ts` — bun-webgpu harness
  idioms (device init, validation-scope pipeline creation, readback).
- HANDOFF.md §"Headless GPU verification" — shared-GPU ±30%, serialize runs.

## Architecture (decided — do not redesign)

### 1. Train-mode plan (κ: `--train` flag → `plan_train.json`)
- Every step's output gets a UNIQUE slot (no slot reuse; ~60 MB total — fine).
  Input slot stays slot 0. Keep the inference plan.json unchanged.
- Steps with `act: "gelu"` are SPLIT in train mode: the conv/se emits
  `act:"none"` into its own slot (pre-activation, needed by gelu-backward),
  followed by a new elementwise step `{kind:"gelu", src, dst, n}`. Both have
  real ONNX ref tensors (the Conv output and the activation Mul_1 output), so
  per-step forward verification still works. Fused residual+layer-scale on
  pointwise convs stays fused.
- κ additionally emits `backward: [...]` in plan_train.json — the reverse
  step list, each entry naming: kind, the saved-activation slots it reads,
  gradient src/dst slots, weight offsets (including NEW packed transposed
  copies where needed), and an `accumulate: bool` flag.
- Gradient slots: one grad slot per forward tensor that needs a gradient
  (mirror of the activation slots; another ~60 MB). A tensor consumed by TWO
  backward edges (e.g. the token-mixer output: convffn input AND residual)
  gets `accumulate: true` on the second writer — κ computes writer order and
  sets the flag; kernels with the flag ADD into dst instead of overwriting.
  No global zero-fill: the first writer overwrites.
- Weight packing: for each pointwise conv, ALSO pack the untransposed
  `[Cout][Cin]` orientation (bwd-data needs dX = Wᵀ·dY, which in our layout
  convention is exactly the pointwise kernel reading the OTHER orientation).
  This roughly doubles weights.bin for pointwise segments (~85 MB total) —
  acceptable. Record as `wOffT` on the forward step / backward entry.

### 2. Backward step kinds (WGSL emitters in a NEW file
`src/clip/vision_bwd_wgsl.ts`, same pure-codegen style; runtime extensions in
`src/clip/vision.ts` — a `VisionTrainer` class or extension of VisionEncoder
that owns activation+grad slots and can encode forward, loss-head, backward
in one pass list)

- `loss_bwd` — L = −cos(embed, text). One workgroup: reads embed[512] (saved)
  + a text-embedding uniform buffer (bind as storage, uploaded per prompt),
  writes dL/dembed[512]. d(−cos)/de = −(t/(|e||t|) − cos·e/|e|²).
- `head_bwd` — dX[ci][p] = (1/P)·Σ_co W[ci][co]·dEmb[co]. (GAP backward is
  the 1/P broadcast; matmul backward reads W in its STORED orientation.)
- `gelu_bwd` — dX = dY ⊙ gelu'(x_pre); gelu'(x) = Φ(x) + x·φ(x) where
  Φ(x)=0.5(1+erf(x/√2)), φ(x)=exp(−x²/2)/√(2π). Reuse the erf poly from the
  forward GELU (copy the A&S constants exactly).
- `pw_bwd` — dX = pointwise-matmul(dY, W in [Cout][Cin] orientation via
  wOffT), no bias, no act. REUSE the existing tiled pointwise emitter by
  parameterizing it (export a shared core from vision_wgsl.ts rather than
  duplicating 80 lines). Epilogues:
  - forward had layer-scale γ + residual: incoming dY is d(out); the conv
    branch gets dY⊙γ[c] applied BEFORE the matmul (fold into a per-channel
    scale on load), and κ emits a separate tiny `residual_bwd` elementwise
    step routing dY (unscaled) into the residual producer's grad slot
    (respecting accumulate).
- `spatial_bwd` — gather form, thread per INPUT pixel (mirror of the forward
  spatialConv): dX[ci][iy][ix] = Σ_{co ∈ group outputs} Σ_{ky,kx} valid-tap
  dY[co][oy][ox]·W[co][ci_local][ky][kx], where oy=(iy+pad−ky)/stride only
  when divisible and in-bounds — bake stride∈{1,2} handling at codegen
  (stride-1: plain flipped-kernel correlation, unrolled; stride-2: parity
  checks baked). Depthwise is the cpg=1 special case, same emitter. For the
  grouped convs with cpgOut=2 (lkb/conv_exp) each input channel gathers from
  its 2 output channels. For the STEM (cin=3) — NOT needed: step 0's dX is
  the final answer... it IS needed: dL/dpixels is exactly step 0's spatial_bwd.
- `se_bwd` — one workgroup like forward. Recompute gap/mid/scale from the
  saved INPUT activation (cheaper than saving them), then:
  dX = dY⊙scale + (per-channel gate-path term): standard SE chain rule
  through sigmoid → fc2 → relu → fc1 → GAP (the GAP backward broadcasts
  1/P). If forward had fused gelu, that's already split out in train mode.
- `attn_core_bwd` — one workgroup per head, mirroring the forward's staging:
  recompute the softmax row p_i from saved qkv (do NOT store the 64×64
  probs), then standard MHSA backward: dV = pᵀ·dO; dP = dO·Vᵀ;
  dS_ij = p_ij(dP_ij − Σ_k p_ik dP_ik); dQ = dS·K; dK = dSᵀ·Q. Note the
  forward FOLDED the 1/√d scale into the q weights, so Q here is already
  scaled — dQ/dK need no extra scale factor (the fold is upstream in pw_bwd's
  weights). Writes d_qkv (channel-planar, same layout as forward qkv slot).
- Residual pass-throughs and layer-scale handled as above; BatchNorm needs NO
  backward step of its own (folded into qkv weights, so pw_bwd through the
  folded weights IS the BN backward — verify this in the directional test).

### 3. Verification (the acceptance gates — in `tools/clip/bwd_test.ts`, bun)
1. **Per-kernel float64 JS references** on SMALL random shapes (not the real
   model): e.g. pointwise 8→12 @4×4, depthwise k3/k7 stride 1/2 @8×8, se
   c16/mid4, attn c32/heads2/n16, gelu, head. Compare kernel dX vs JS
   analytic reference at 1e-4 rel, AND JS reference vs central finite
   differences of the JS forward at 1e-3 (catches formula errors in the
   reference itself). Follow tools/kernel_test.ts's reference style.
2. **End-to-end directional derivative** on the REAL model + fixture input:
   grad = full backward from the cosine loss (use a fixed synthetic
   text embedding, e.g. normalize(pcg-noise[512]) — deterministic);
   for K=8 random unit directions v (pcg-seeded, over all 196608 pixels):
   compare ⟨grad, v⟩ vs (L(x+εv)−L(x−εv))/(2ε) computed with TWO forward
   passes of the ALREADY-VERIFIED forward encoder (ε≈1e-2 since fp32; report
   all 8 relative errors; gate at rel < 2e-2 for at least 7/8).
3. Per-step forward verification must still pass for plan_train.json (the
   split-gelu steps have ONNX refs; extend dump_refs.py + fused_test.ts to
   accept a PLAN=plan_train.json env override).
4. Bench: full forward+backward wall time (same warmup rules; expect roughly
   2–3× forward; report the number, don't gate on it).

### Non-goals (do NOT build)
- dW for any weight. No optimizer here (Adam lives with the splat kernels).
- fp16. batch>1. Browser page wiring. Do not touch src/main.ts, the advect/
  train kernels, or anything outside src/clip/, tools/clip/, docs/.

### Workflow requirements
- Regenerate plan/refs via the README commands after κ changes; keep
  `bun tools/clip/fused_test.ts` green (inference plan) throughout.
- Every κ structural assumption asserted loudly (house rule: unmatched
  structure throws; no default-case fallbacks).
- Update tools/clip/README.md (backward section: how to run tests, measured
  fwd+bwd ms) and add gotchas you hit to its gotcha list.
