# Phase 2 — Fused TRAIN kernel (analytic backward + Adam, K-step rollouts)

Status: **v1 SHIPPED + VERIFIED** (K=1). Gradients match tfjs autograd at
cos=1.0000000 / relMax 1.8e-5 on the exact composite loss; loss parity 3e-7;
Adam parity 3e-7; fused train step 1.4–2.3 ms at batch 256 vs 8–20 ms tfjs
(`bun tools/train_test.ts`). main.ts trains field pieces fused by default
(trainer co-owns the advect weights buffer, `advect.syncFromTfjs=false`,
tfjs idle on the hot path); `?train=tfjs` is the A/B fallback.

**K-step rollout: SHIPPED + VERIFIED.** Pass A runs a K-step BPTT rollout
(sites = K physics evals + 3 final-state probes; velocity carried; isotropy
over all K steps' forces, chaos/div/spiral at pos_K). K=4 gradients match a
tfjs BPTT fixture at cos=1.000000 / relMax 1.9e-5
(`tools/fixtures/grad_ref_k4.json`, regenerate with
`K=4 OUT=tools/fixtures/grad_ref_k4.json bun tools/grad_reference.ts`).
In the app: `?rollout=K` (1–16, default 1). Remaining: particle-subsample
train source, browser QA, maybe make rollout a gallery-config knob once QA'd.

## The three questions this answers

**1. Analytic backprop vs autograd — faster? more accurate?**
tfjs autograd IS analytic (chain rule op-by-op); hand-writing the backward does
not change the math, so accuracy is identical up to fp32 rounding order. The
win is structural: the current learn stage is ~40–100 tiny dispatches (some
CPU-forwarded with readbacks — tfjs's small-tensor CPU handoff); fused it is
TWO dispatches. Expected 10–20× on the train stage, same numbers. The deeper
reason to own the backward: it unlocks things tfjs cannot express — sparse
backward over a particle subset, K-step BPTT at register speed, and weights
that never leave the GPU.

**2. Richer training signal — multi-step rollouts.**
Today the loss sees ONE physics step from zero velocity. A K-step rollout
(K≈4–16) lets gradients flow through evolving position AND velocity — the
field is optimized for how particles actually flow through it, which is what
the art renders. In tfjs this is K× dispatch cost (prohibitive); in the fused
kernel a K-step rollout of the 2.3K-MAC field is register math (~40K MACs per
sample — trivial). BPTT storage: per-step activations in a global scratch
buffer (~MBs at batch 256, K 16). K=1 must reproduce current semantics EXACTLY
(verified against a tfjs-autograd fixture).

**3. Still splitting advect and train?**
Yes — but the split is now free and the pieces share everything. Advect (all N
particles, 1 step/frame, forward-only) and train (small batch, K steps, full
backward) share the same GPUDevice, queue, and — after this phase — the same
kernel-owned weights buffer, with zero copies anywhere. A literal "one forward
serves both" only makes sense for K=1 and batch=N, which is the WORST config
(no rollout signal, dense backward). What we DO unify is the source of
training states: a knob picks random points (better field coverage — current
behavior) or a hashed subsample of the live particle buffers (trajectory-true).
The tfjs→kernel weight sync (dataToGPU / the dataSync CPU-residency fallback)
exists only because tfjs owns training today; when the fused trainer lands,
weights are born on the GPU and never leave — that whole seam disappears, and
tfjs drops out of the hot path entirely.

## Architecture: 2 dispatches per training step

**Pass A — rollout + backward (ONE workgroup owns the whole batch, ≤1024
samples, 256 threads, ≥1 sample/thread).**
1. K-step rollout from (tp, vel=0), storing per-step per-site activations to a
   global scratch buffer. Eval sites per step: 1 physics force eval at
   norm(pos_k) + (final step only) 3 loss probes at norm(pos_K){,+HH x,+HH y}.
2. Batch reductions in workgroup shared memory (isotropy needs C00, C01, C11
   of the scaled force batch; all loss means need 1/N sums).
3. Per-sample backward in registers: loss terms → dL/dF per site, BPTT chain
   dL/dpos_k, dL/dvel_k across steps (clip mask, friction, mod≡identity),
   input-jacobians of the MLP for the probe/pos chains, head blend
   δ_g=(1−α)δ_f, δ_r=α·δ_f. Writes per-(sample,site,head,layer) deltas next to
   the stored activations in scratch.

**Pass B — gradient assembly + Adam (thread = weight entry).**
dW[i][j] = Σ_samples Σ_sites a_in[i]·δ_out[j]; db[j] = Σ δ[j]; then Adam in
place (m, v buffers + step-count uniform; match tfjs defaults β1=.9, β2=.999,
eps — VERIFY empirically, tfjs may use backend epsilon 1e-7). Updates the SAME
packed weights buffer the advect kernel reads.

## Loss (v1): exact parity with `helmholtzChaosLoss` in main.ts

loss = 1.0·chaos + 1.0·isotropy + 0.5·divergence + 2e-5·spiral, HH=1e-2.
Backward pieces (a = post-activation; derivatives from post-acts only):
- selu′ = a>0 ? 1.0507009873554805 : a + 1.7580993408473768; tanh′ = 1−a²;
  sigmoid′ = a(1−a)
- chaos: sep=√(sepx+sepy+1e-12)/(HH·√2+1e-9); d(−mean log(sep+1e-6)) — per
  sample, through fx−f0 / fy−f0.
- isotropy (batch stats): S=C00+C11+eps, D=C00−C11, L=(D²+4C01²)/S²;
  dL/dC00=2D/S²−2L/S, dL/dC11=−2D/S²−2L/S, dL/dC01=8C01/S²;
  dL/dFx_i=(2·dLdC00·Fx_i + dLdC01·Fy_i)/N, dL/dFy_i=(2·dLdC11·Fy_i +
  dLdC01·Fx_i)/N. NOTE: applies to the SCALED force (×forceMagnitude).
- divergence: g=(fx.x−f0.x + fy.y−f0.y)/HH, d(mean g²) per sample /HH.
- spiral: winner-take-all k over (r−b·relu(θ+2πk))²; dr/dxy=xy/r,
  datan2: (−dy, dx)/r²; weight 2e-5.
- physics: dnewVel/dF_scaled = friction·[|pre-clip|≤maxVel]; dnewPos/dnewVel=1;
  mod′=1; probes read pn=newPos/[w,h] → their input-grads flow back into the
  physics chain (÷[w,h]).

## Verification (all headless on real Metal, bun-webgpu)

- `tools/fixtures/grad_ref.json` — tfjs-CPU autograd fixture (variableGrads on
  the exact composite, seeded weights mulberry32(42), batch mulberry32(7),
  N=256, w=800 h=600, α=0.7). Kernel gradients must match at ~1e-3 rel.
- Then a full-loop test: T fused Adam steps vs T tfjs optimizer steps from the
  same init → loss curves must track (catches Adam-detail mismatches).
- K>1: slow tfjs BPTT reference at K=2 on a tiny batch for correctness, then
  kernel-only benches.

## Files

- `src/render/webgpu/train_wgsl.ts` — PURE codegen (mirror of advect_wgsl.ts):
  layout of scratch/grad/adam buffers + pass A/B WGSL from the same
  FieldLayout. No hardcoded dims (AlphaGOJS trap rule applies).
- `src/render/webgpu/train.ts` — kernel-owned weights: init once from tfjs (or
  PRNG), owns m/v/step, exposes step(seed, alpha, lr, K); advect kernel gains
  a constructor option to read an external weights buffer instead of syncing.
- `tools/train_test.ts` — fixture comparison + loss-curve tracking + bench.
