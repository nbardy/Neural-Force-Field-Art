# Plan — training batches (shipped) + multi-species classes (next)

Companion to docs/PLAN_FUSED_TRAIN.md (trainer core) and
docs/QA_FUSED_KERNELS_2.md (the question round that produced this plan).

## 1. Batch sources — SHIPPED, verified in tools/train_test.ts

One knob, three canonical sources, one optional mix floor:

```
TrainStepOpts.source : "particles" | "random" | "uploaded"
TrainStepOpts.mixRandom : number    // 0..1, only meaningful for "particles"
```

- `"particles"` (app default): sample live advect states by hashed index —
  real pos AND vel. Trains hardest where the attractors are (the art), which
  is the point. Exploration is the RESET slider's job: resets inject fresh
  uniform vel-0 states into the cloud → into this batch. UI slider is live.
- `"random"`: fresh uniform points, vel 0 — the original source (`?batch=random`).
- `"uploaded"`: fixture batches for verification only.
- `mixRandom` (`?mix=F`): first ⌊F·n⌋ samples come from the random source even
  under `"particles"` — a coverage floor independent of the reset rate. Default 0
  per the maintainer's call; it exists because it's one uniform, not a rewrite.

### Rollout semantics decisions (the questions, answered as math)

- **Rollouts are imagined, never reset mid-chain — by construction.** The
  trainer's K-step rollout integrates `pos_{k+1} = mod(pos_k + clip((vel_k +
  F(pos_k)·mag)·fric))` with NO reset branch. This is not an approximation
  dodge: a reset is a weight-independent teleport, so ∂(post-reset state)/∂θ
  through the pre-reset chain is exactly 0 — including resets would just
  truncate BPTT into a shorter chain. Fresh shorter rollouts ≡ reset-broken
  long ones, minus bookkeeping. ("Learns history doesn't matter" is therefore
  not something we teach — history within a chain matters exactly K steps.)
- **The full chain IS differentiable and we use the full repeated-network
  chain derivative**: two adjoints (dL/dpos_k, dL/dvel_k) walk the transitions
  backward, picking up the clip mask × friction, the mod identity, and the
  MLP's input-Jacobian at every step (docs/MATH_ANALYTIC_GRADIENTS.md §BPTT).
  Verified against tfjs autograd at K=4: cos = 1.000000.
- **Recorded (true-trailing) windows** would be the only place resets need
  masking: drop particles whose window crosses a reset (~K·resetRate of the
  batch — at 1%/frame, negligible bias). Still a back-burner branch (task #9)
  because imagined-from-live-states dominates: same trajectory distribution,
  fresher by k frames, no recording machinery. Revisit only if K·batch grows
  past what ~2-4 ms buys.
  **UPDATE — leading-window equivalence PROVEN by tools/window_test.ts**: with
  frozen weights the trainer's imagined K-step rollout from a sampled
  particle's live state reproduces the advect kernel's next K real frames to
  max Δ = 6.1e-5 px at K=6 (resetRate 0; the only drift is fp32 op-order
  between the two MLP evaluators), so `?window=K` (= `rollout=K` +
  `trainEvery=K`) IS trajectory-window training on real particles, in leading
  form. Task #9 (recording machinery) is hereby CLOSED as dominated: recorded
  trailing windows would buy the same trajectories, k frames staler, plus
  reset-masking bookkeeping the imagined rollout doesn't need.
- **Persistent shadow batch** ("keep a batch, reset it every N steps"): a
  non-rendered training population that persists and re-rolls each frame is
  exactly what `"particles"` already provides using the real population —
  which has resets built in. A separate shadow adds value only if training
  wants different physics/reset constants than the art. Available branch;
  not planned.

## 2. Field conditioning — what the network actually sees

**Today: F = blend(g(pn), r(pn)), pn = pos/[w,h] ∈ [0,1]².** The network input
is the CONTINUOUS position coordinate — a plain 2-float input to the MLP, no
grid, no index lookup, no embedding table. That's what makes it a true force
field: continuous, resolution-independent, and differentiable in position
(the chaos/divergence probes and the BPTT position-chain both use ∂F/∂pos).

**Velocity is NOT an input.** F(pos) only — same force at the same place
regardless of how a particle moves through it (an autonomous field; friction
supplies the only velocity coupling). Making it F(pos, vel) is a different
artistic object (a per-particle controller, not a field) and muddies the
probe losses (chaos/divergence probe positions — what velocity would a probe
have?). Branch reserved in the type (`FieldInputs = "pos" | "pos+vel"`), not
planned.

## 3. Multi-species classes — NEXT UP

Per-particle class c ∈ {0..C-1}; the CHAOS lane is class-aware, the ORDER
lane is class-blind (the maintainer's framing: the thing measuring order
cannot see species).

```
F_c(p) = (1-α)·g(pn) + α·r(pn, onehot(c))      g : R² → R²
                                                r : R^(2+C) → R²
```

**Class assignment is a pure hash — no storage anywhere.** Class is identity,
stable across resets and frames: `c(i) = pcg(i ^ CLASS_SALT) % C` derived from
the particle index i in the advect kernel, from the sampled index in the
trainer, and from `instance_index` in the renderer. Zero new buffers; equal
class fractions for free. (Dynamic reassignment would force a real buffer —
out of scope.)

**Codegen deltas** (all driven by `classes: number` on the helmholtz spec —
0 = today's behavior, bit-identical):
- `advect_wgsl.layoutField`: validateChain's first-layer inSize becomes
  per-head data: g expects 2, r expects 2+C. Throws on mismatch with the live
  tfjs net (the D-trap rule extends to C).
- head emitters: layer-0 input vector = `[p.x, p.y, onehot(c)]`; unrolled
  emitter adds `+ select(0,1, c==k)·W(...)` terms — better: skip the multiply
  and ADD the class column directly: for layer 0,
  `s = bias[j] + p.x·W[0][j] + p.y·W[1][j] + W[2+c][j]` (one extra indexed
  row load — the one-hot never materializes).
- backward: input-gradient du stays 2-D (position dims only; the one-hot is
  constant). Layer-0 dW rows: pos rows as today; class rows get
  `dW[2+k][j] += (k==c) · δ0[j]` — pass B reads the sample's class from the
  scratch header (ONE new header float) and accumulates row 2+c only.
- trainer pass A: class derived at sourcing time (hash of particle idx, or of
  sample slot for random/mix), written to the header; probe sites inherit the
  sample's class (chaos measures WITHIN-species sensitivity automatically).
  Isotropy stays global — class-blind order pressure, as specified.
- HelmholtzField (tfjs blueprint): `hiddenUnits` unchanged; r's inputShape
  becomes [2+C]; `HelmholtzFieldConfig.classes?: number`.
- renderer: per-class base hue (golden-angle spaced), brightness by speed.
- gallery: new piece "Helmholtz · Species" with classes: 3; classless pieces
  compile the exact same WGSL as today (C=0 path must stay bit-identical —
  regression-tested).

**Verification**: extend tools/grad_reference.ts with CLASSES env (tfjs side:
r takes concat([pn, onehot]) — classes assigned by the same pcg hash ported
to JS); new fixture; train_test case "classes C=3". Advect side: kernel_test
config with mismatched-head input sizes (2 and 2+C) — also kills any lingering
assumption that heads share input width.

**Function signatures (target):**
```ts
layoutField(kind, headsDims, opts?: { classes?: number }): FieldLayout
advectShader(layout, { stageWeights, workgroupSize?, unroll? })   // unchanged API
trainPassAShader(layout, { kSteps? })                              // unchanged API
new HelmholtzField({ alpha, hiddenUnits?, classes? })
AdvectKernel.fromField(field, physics, n)      // ingests+validates classes
trainer.step(phys, { n, alpha, lr, seed, source, mixRandom, apply })
```

## 4. Knob inventory (all live now)

| knob | where | meaning |
|---|---|---|
| `?rollout=K` | URL | BPTT rollout length 1–16 (compiled into WGSL) |
| `?trainEvery=N` | URL | train every Nth frame |
| `?batch=random` | URL | revert to uniform-random batches |
| `?mix=F` | URL | random coverage floor within particle batches |
| `?train=tfjs` | URL | legacy tfjs optimizer A/B |
| `?handoff=N` | URL | tfjs CPU-handoff threshold (tfjs path only) |
| reset slider | UI | respawn fraction ≡ training exploration dial |
| particles / samples / α sliders | UI | as before |

## 5. Order of work

1. ~~particle-source batches + mixRandom + live reset slider~~ SHIPPED.
2. ~~Classes (§3)~~ SHIPPED: codegen (both kernels), HelmholtzField
   `classes` config, renderer species hues, "Helmholtz · Species" gallery
   piece (C=3). Verified headless: advect classes=3 numeric vs class-aware
   reference (kernel_test); trainer via two EXACT invariants (train_test) —
   zero-class-rows ≡ classless bit-for-bit, and Σ_c class-row grads ≡ the
   layer-0 bias grad (the one-hot partitions the batch); plus a full
   tfjs→trainer→advect seam test (integration_test). Optional
   belt-and-suspenders left: a CLASSES tfjs-autograd fixture.
3. Browser QA of everything (task #8) before gh-pages deploy.
4. Back-burner branches: trajectory-window (task #9), F(pos,vel), shadow batch.
