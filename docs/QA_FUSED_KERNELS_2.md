# Fused kernels — maintainer Q&A, round 2

Answers to the second round of questions, asked while the training-batch work
was landing. Status at time of writing: particle-sourced batches + `mixRandom`
+ the live reset slider are SHIPPED and verified (`tools/train_test.ts`);
multi-species classes are DESIGNED and next up. The decisions recorded here are
expanded into an implementation plan in docs/PLAN_SPECIES_AND_BATCHES.md;
round 1 is docs/QA_FUSED_KERNELS.md.

---

## Q1 — "Multi-step trailing batch on current positions — is that working?"

Yes — shipped this session. `TrainStepOpts.source: "particles"`
(`src/render/webgpu/train.ts`) makes pass A sample the LIVE advect state:
the sourcing block in `train_wgsl.ts` draws a hashed index
`pcg(s ^ (u.seed * 3266489917u)) % partCount` and reads BOTH `partPos[idx]`
and `partVel[idx]` — real position AND real velocity, not points from rest —
then runs the normal K-step imagined rollout from that state. (The mixing
constant is distinct from the advect kernel's reset RNG so the index stream is
uncorrelated with which particles just respawned.) The buffers are the advect
kernel's own pos/vel storage, bound zero-copy via
`trainer.setParticleBuffers(advect.posBuffer, advect.velBuffer, advect.count)`
in `main.ts`.

Verified falsifiably in `tools/train_test.ts` case 3.6: with `partCount=1`
every sample hashes to the same particle, so gradients must be bit-identical
to an uploaded batch of N copies of that point (measured Δ < 1e-6) — that
pins the sampling plumbing. Then writing a nonzero velocity into the particle
buffer must CHANGE the gradients and keep them finite — that proves v0
actually enters the rollout instead of being silently zeroed. Both pass.

This is the app default for field pieces (`main.ts` — the batch trains
hardest where the attractors are, which is where the art lives);
`?batch=random` reverts to the original uniform source.

## Q2 — "Mix random train batch as a config option — does that mean rolling out a random un-rendered batch multiple steps?"

Yes, exactly that. `TrainStepOpts.mixRandom` (`?mix=F`) reserves the first
⌊F·n⌋ batch slots for fresh uniform vel-0 points — the WGSL condition is
`batchSource == 2 && s < u.mixCount` falling into the same random-point branch
as `?batch=random` — and those samples get the SAME K-step rollout treatment
as everyone else: imagined, never rendered, discarded once the gradient is
taken. It's a coverage floor independent of the reset rate, not a separate
code path.

Default is 0 per the maintainer's call: the reset slider already injects
fresh uniform vel-0 states into the cloud (hence into the particle-sourced
batch), so exploration is its job. `mixRandom` exists because it cost one
uniform field, not a rewrite.

## Q3 — "Could we have a persistent batch that resets every N steps and trains on?"

The "shadow population" idea — a non-rendered batch that persists across
steps, re-rolls each frame, and periodically resets. The answer is that the
real particle cloud already IS that object: it persists, the advect kernel
resets a fraction of it every frame (the reset slider), and
`source:"particles"` trains on it directly. Standing up a second, hidden
population would reproduce the same distribution with extra buffers and its
own reset machinery.

A separate shadow batch only pays off if training wants DIFFERENT physics or
reset constants than the art does (e.g. train on a harsher reset rate than
you'd want to watch). That's an available branch, recorded in
PLAN_SPECIES_AND_BATCHES.md §1, not planned.

## Q4 — "If we always train on rollout chains, do we reset mid-rollout, so it learns history doesn't matter?"

No resets inside rollouts — and it's principled, not a shortcut. The trainer's
rollout integrates `pos_{k+1} = mod(pos_k + clamp((vel_k + Fs_k)·friction))`
with NO reset branch (`train_wgsl.ts` forward loop). A reset is a
weight-independent teleport: the post-reset state doesn't depend on θ through
the pre-reset chain, so the gradient through a reset is EXACTLY zero. A K-chain
broken by a reset is therefore mathematically identical to two shorter
independent chains — including resets would only truncate BPTT plus add
bookkeeping.

So nothing about "history doesn't matter" gets taught. Within a chain, history
matters for exactly K steps — the adjoints carry loss signal from `pos_K` all
the way back to `pos_0` — and K is a knob (`?rollout=K`, 1–16, compiled into
the WGSL at construction).

## Q5 — "Is the full replay chain differentiable — are we using the full repeated network's chain derivative?"

Yes. That's BPTT: the chain rule composed across the unrolled steps, applied
as vector-Jacobian products rather than a materialized Jacobian product. The
kernel carries two running adjoints, `dL/dpos_k` and `dL/dvel_k`, and pulls
them backward through each transition — picking up the clip mask × friction
(velocity chain), mod ≡ identity (position chain), and the MLP's
input-Jacobian at every step's force eval (the `bwd_head_*` input-gradient).
The same network's weights are hit by every step's δ's; pass B sums
`a⊗δ` over all K physics sites plus the 3 probe sites, which is precisely
`dW` of the whole unrolled graph.

Full derivation with the exact update order: docs/MATH_ANALYTIC_GRADIENTS.md
§5 (the BPTT recurrence) and §6 (assembly). Verified against a tfjs autograd
fixture at K=4: cos = 1.000000, relMax 1.9e-5 (`tools/train_test.ts`).

## Q6 — "Reset in the middle of a chain is not differentiable — do we drop those points?"

In the imagined rollouts we ship, the situation never arises: the rollout has
no reset branch (Q4), so no chain ever crosses one. The question becomes real
only in the future recorded-window variant (round 1 Q6b, back-burner task #9),
where a REAL particle's trajectory is recorded over k frames and the advect
kernel's resets do fire mid-window. There: reset-crossing particles get
DROPPED from that window's batch. Expected loss is ~K·resetRate of the batch —
roughly 1–4% at default rates (resetRate ≈ 0.008–0.015/frame) — negligible,
and unbiased for the surviving chains.

Dropping is also exactly equivalent to truncating, not an approximation: per
Q4 the pre-reset segment's gradient contribution through any post-reset loss
is identically zero, so the only thing discarded is a shorter chain we could
have kept as a separate sample. See PLAN_SPECIES_AND_BATCHES.md §1.

## Q7 — "How is the network conditioned on position — index lookup or position input?"

Position input. The network sees `pn = pos / [w,h] ∈ [0,1]²` fed directly as
the MLP's 2-float input — `src/core/field/helmholtz.ts` builds each head with
`inputShape: [2]` (2→32→32→2, selu/selu/tanh), and the kernel forward
computes `uk = p / res` and hands it straight to `fwd_head_*`
(`train_wgsl.ts`). No grid, no index lookup table, no positional embedding.

That continuity is load-bearing, not incidental: ∂F/∂pos exists everywhere,
which is what the chaos/divergence finite-difference probes measure and what
the BPTT position chain differentiates through (the `du/res` term in the
recurrence — MATH_ANALYTIC_GRADIENTS.md §5 step 7). A lookup-table field would
have no usable position derivative and the whole training signal changes
character.

## Q8 — "Does the model take velocity too, or is it a true force field?"

Position only: `F(pos)`. It's a true autonomous field — the same force at the
same point regardless of how a particle is moving through it. Velocity enters
the dynamics only through integration and friction
(`velPre = (vel + Fs)·friction`), never through the network input
(`helmholtz.ts`: both heads are R²→R²).

`F(pos, vel)` is a different artistic object — a per-particle controller
rather than a field — and it muddies the probe losses: chaos and divergence
probe fixed POSITIONS, and there's no principled answer to what velocity a
probe should carry. The branch is reserved as a type
(`FieldInputs = "pos" | "pos+vel"`, PLAN_SPECIES_AND_BATCHES.md §2), not
planned.

## Q9 — "Add a 'class' per particle; chaos handles each class differently; the order-measurer is class-blind?"

Designed and next up — the full plan is PLAN_SPECIES_AND_BATCHES.md §3. The
shape is exactly the asymmetry asked for:

```
F_c(p) = (1-α)·g(pn) + α·r(pn, onehot(c))
```

`g` (the order lane) never sees the class — class-blind by construction, not
by regularization. Isotropy, the order pressure, stays a GLOBAL batch
statistic over all samples' forces — the thing measuring order cannot see
species. The chaos/divergence probes inherit the class of the sample whose
final position they probe, so chaos sensitivity is measured WITHIN-species
automatically.

Class assignment is a pure index hash — `c(i) = pcg(i ^ CLASS_SALT) % C` —
so class is identity: zero new buffers, stable across resets and frames, and
the same derivation runs in the advect kernel (from particle index), the
trainer (from the sampled index), and the renderer (from `instance_index`,
driving a golden-angle per-class hue). Codegen deltas (layer-0 one-hot as a
direct class-column add, the one-new-header-float backward, per-head input
sizes) and the verification plan (CLASSES fixture, C=0 bit-identical
regression) are all in the plan doc; `classes: 0` compiles today's WGSL
exactly.

---

## What's now live — knob inventory

(From PLAN_SPECIES_AND_BATCHES.md §4.)

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
