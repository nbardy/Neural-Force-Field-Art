# Fused kernels — maintainer Q&A

Answers to the questions asked while the fused TRAIN kernel was landing. Status
at time of writing: fused advect (Phase 1) and fused train (Phase 2, K-step BPTT)
are both SHIPPED and verified; see HANDOFF.md and docs/PLAN_FUSED_TRAIN.md.
Measured numbers below are Apple Metal-3 on a shared GPU (±30% run-to-run —
read trends, not single numbers).

---

## Q1 — "Do we need analytical backprop instead of autograd? Is it faster? More accurate?"

Autograd IS analytic backprop — tfjs applies the exact chain rule op-by-op, same
math we now hand-write. So "more accurate" is a non-question: identical formulas,
identical answers up to fp32 rounding order. That's precisely why the fused
kernel verifies against tfjs autograd at cos=1.0000000 / relMax 1.8e-5
(`tools/train_test.ts`) — if hand-written backprop were different math, that
number would be unreachable.

The win is structural, not mathematical. On the webgpu backend every tfjs op is
one GPU dispatch (HANDOFF §2), so a train step is ~50 dispatches of trivial math;
the fused trainer is TWO (`src/render/webgpu/train_wgsl.ts`: pass A rollout +
backward, pass B assembly + Adam). Measured: 1.4–3.8 ms/step at batch 256 vs
8–20 ms for the tfjs learn line. And owning the backward unlocks what tfjs cannot
express at all: K-step BPTT at register speed, backward over a sparse particle
subset, and weights that live GPU-side permanently.

## Q2 — "Could we get a richer training signal — multiple steps instead of one?"

Shipped. Pass A runs a K-step BPTT rollout (`?rollout=K`, 1–16, default 1 —
`src/main.ts`, compiled into the trainer's WGSL at construction): the loss sees
how a sample point FLOWS through the field — evolving position AND accumulated
velocity — instead of one step from rest. Isotropy generalizes to the covariance
of ALL K steps' scaled forces (an N·K batch); chaos/divergence/spiral act at the
final state pos_K. K=4 gradients match a tfjs BPTT fixture at cos=1.000000 /
relMax 1.9e-5. In tfjs this signal costs K× dispatches; in the kernel a K-step
rollout of the 2.3K-MAC field is a few more register loops.

## Q3 — "Are we still splitting advect and train? Why not one forward + sparse backward on a particle subset?"

Still split, deliberately. The two stages do different jobs on different
populations: advect = 1M particles × 1 forward-only step per frame
(`advect_wgsl.ts`); train = 256 samples × K steps with a full backward
(`train_wgsl.ts`). A literally-shared forward only makes sense when K=1 and
batch=N — which is the WORST training config: no rollout signal, and a dense
backward over rows the loss never touches. That's the tfjs O(N)-backward trap
from HANDOFF §2 (`dL/dW = Xᵀ·dF` runs over all N even when dF is 256 nonzero
rows) — sharing the forward re-creates it.

And the split costs nothing: both kernels share the device, the queue, and ONE
packed weights buffer (`FusedTrainer` is constructed with
`advect.weightsBuffer`; zero copies anywhere). The unification that IS worth
having is the training-SOURCE knob — sample the training batch from the live
particle buffers instead of random points. Planned (PLAN_FUSED_TRAIN "remaining");
cheap now because the trainer can read the advect buffers directly.

## Q4 — "Why did it look like we were editing tfjs/tensor code instead of a parallel WGPU-only approach?"

No tfjs code was edited. The new stack IS parallel and WGPU-only
(`advect_wgsl.ts` / `train_wgsl.ts` are pure codegen with zero tfjs imports;
`FusedTrainer` is a pure WebGPU object). tfjs plays two roles: the BLUEPRINT —
WGSL is generated from the live model's layer dims so the kernel can't drift
from the architecture (the AlphaGOJS D=8 trap rule) — and the offline TEST
ORACLE (`tools/grad_reference.ts`). The tfjs-touching code seen mid-flight was
the Phase-1 bridge in `advect.ts step()`: per-frame weight sync while tfjs still
owned training. That seam still exists for the legacy MLP pieces but is bypassed
whenever the fused trainer is active (`advect.syncFromTfjs = false`, set in
`main.ts`).

## Q5 — "What was the .data()/.dataSync() stuff? Can't data always stay on GPU? Can train hand off to render — even render in the same kernel launch?"

The dataSync fallback existed because tfjs-webgpu CPU-forwards small-tensor ops
(`WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD`) — our weight tensors are tiny, so freshly
Adam-updated values were often literally CPU-resident inside tfjs, `dataToGPU()`
threw "not on GPU", and we uploaded via `dataSync()+writeBuffer`
(`advect.ts step()`, both paths land identical bytes). That was a tfjs
residency problem, not ours.

With the fused trainer that seam is OFF: weights are born on the GPU
(`trainer.uploadWeights(advect.packCurrentWeights())` once at init, then
`syncFromTfjs=false`) and never leave. The remaining `.data()`-ish calls are
test-harness readbacks only (`FusedTrainer.readGrads/readWeights/readLoss` —
MAP_READ staging, off the hot path; plus the HUD's loss peek every 30 frames).

Trainer→advect→render already share with zero copies: train writes the packed
weights buffer in place, advect binds it, and the renderer binds advect's
pos/vel buffers directly (`renderFromBuffers`); the vertex shader pulls straight
from the particle storage buffer. One literal kernel launch for compute+raster
isn't possible in WebGPU — rasterization needs a render pass, compute a compute
pass — but that boundary is free-ish; the part that matters is zero-copy on one
queue, which we have end to end.

## Q6 — "Doesn't the K-rollout compute steps we never use? Should we train only every k frames instead?"

The K-step rollout is an IMAGINED trajectory on the 256-point training batch —
a few million MACs, ~2–4 ms — not the visible particles. Nothing rendered is
discarded; the rollout exists purely to shape the gradient.

But the instinct is right, and there are two cheap upgrades. (a) `?trainEvery=N`
— shipped in `main.ts`: run the fused step every Nth frame, amortizing the cost;
since the rollout batch is imagined, skipping frames loses nothing but update
frequency. (b) The principled version: record the REAL particle subsample's
trajectory over a k-frame window (weights frozen during the window), then run
ONE BPTT step over the recorded states — zero extra forwards, trajectory-true
signal. Needs reset-masking (respawned particles would poison the chain).
Planned task.

## Q7 — "With analytic gradients, how do you track gradients across k steps — is there a k-step chain rule?"

Yes — that's BPTT (backpropagation through time). The chain rule composes across
time as a product of per-step Jacobians; you never materialize the product, you
apply it as vector-Jacobian products walking backwards. The kernel carries two
running adjoints, dL/dpos_k and dL/dvel_k, and pulls them through each
transition: the clamp mask × friction (velocity chain), mod ≡ identity (position
chain), plus the MLP input-Jacobian at each step's force eval. See the backward
loop in `train_wgsl.ts` (`walk transitions k = K-1 .. 0`) and the full
derivation with the exact update order in docs/MATH_ANALYTIC_GRADIENTS.md §5.
Verified at K=4 against a tfjs BPTT fixture: cos=1.000000.

## Q8 — "Why 'vs tfjs' in the K=4 test — are we adding features to tfjs?"

No. `tools/grad_reference.ts` is an OFFLINE oracle: tfjs-CPU autograd runs the
exact composite loss once at K=1/K=4, dumps a JSON gradient fixture
(`tools/fixtures/grad_ref*.json`), and the runtime never touches any of it.
Hand-written backward passes are exactly the kind of code that fails SILENTLY —
wrong gradients still "train", just worse — so the oracle is how cos=1.0000000
is a known fact rather than a hope. And yes, the asymmetry is the point: long
rollouts are viable precisely because the kernel is fast. tfjs can't do K=8 in
real time (K× its already-dispatch-bound step); the kernel barely notices.

## Q9 — "tfjs CPU-forwards tiny ops — can we fix that for a massive tfjs speedup?"

The knob exists: `tf.env().set('WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD', 0)` forces
every op to stay on the GPU (default 1000: tensors under ~1000 elements get
CPU-forwarded when their inputs are CPU-resident). `tools/handoff_bench.ts`
benches the real learn step three ways (default / handoff disabled / handoff
everything at threshold 100000).

The headless attempt was INCONCLUSIVE: the machine's GPU was under heavy
contention from other processes (the bench hadn't finished 3×75 steps after
20 minutes and was killed rather than keep competing). Measure it where it
matters instead — in the browser: `?train=tfjs&handoff=0` (main.ts wires
`?handoff=N` to WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD) and compare the HUD's
`learn` line against plain `?train=tfjs`. Rerun `bun tools/handoff_bench.ts`
only on a quiet GPU.

Expected conclusion, and why no massive win is on the table either way: the
handoff trades CPU-sync overhead for dispatch overhead — both are the same
disease, per-op launch cost on ~40–100 tiny ops (HANDOFF §2). Moving a tiny op
to the GPU doesn't remove its launch; it just relabels the fixed cost. The fix
is not tuning the threshold, it's removing the per-op structure — which is the
fused kernel, already shipped. Whatever the three numbers say, report them
honestly; they change the tfjs A/B baseline, not the architecture.

## Q10 — "Did we save the analytic-gradient math in a document?"

Yes — docs/MATH_ANALYTIC_GRADIENTS.md (written alongside this doc): forward
definitions, post-activation derivative table, every loss term's gradient with
full derivations, the BPTT recurrence, pass B + Adam, and the verification map.
Derivative comments also live inline at the relevant sites in
`src/render/webgpu/train_wgsl.ts`, and the short-form summary table is in
docs/PLAN_FUSED_TRAIN.md §Loss.

## Q11 — "If advect is fused, does train stay in tfjs? Are we sharing compute between the advect forward and the train forward?"

Train is now ALSO fused — `FusedTrainer` is the default for field pieces
(`main.ts`; `?train=tfjs` is the A/B fallback). tfjs trains only the legacy MLP
pieces, whose losses aren't in the kernel yet. The advect forward and the train
forward are NOT shared computations — different populations (1M particles vs 256
samples), different counts, different step semantics (1 forward step vs K steps
with stored activations) — but they read the SAME packed weights buffer with
zero copies. Sharing the actual forward is an anti-goal: see Q3.

## Q12 — "Why was tfjs ~40–50 kernel launches? Did we try compiling? Does tfjs have a compiler?"

Because on the webgpu backend every tfjs op — matmul, add, activation, slice,
clip, mod — is one WebGPU dispatch, and the learn step's graph is ~40–100 such
ops. tfjs has NO XLA-style fusing compiler and no `tf.function`-style graph
compile; the only fusion it has is a handful of hand-fused ops (e.g.
`fusedMatMul` with bias+activation), which don't cover this graph (finite-diff
probes, batch-stat losses, the physics chain). The browser also has no
CUDA-graph equivalent to amortize launch overhead across a recorded graph. So
the ~1-dispatch-per-op floor is structural — HANDOFF §2's dispatch-bound model —
and that floor is exactly why the fused kernels exist: advect ~40→1 (≈7–13 ms @
1M particles vs ~18–20 naive), train ~50→2 (1.4–3.8 ms vs 8–20 ms).

## Q13 — "Why is the analytic backward separate from the forward/advect? Isn't it better to store partials during the forward, since backward needs forward values?"

Storing forward partials is EXACTLY what pass A does — read
`train_wgsl.ts fwd_head_*`: the forward stores every post-activation to the
scratch buffer as it computes it, and the backward reads those stored values.
They're not even separate dispatches — forward and backward run in the SAME
pass-A dispatch, separated only by a workgroup barrier, because the
batch-statistic reduction sits between them: isotropy needs C00/C01/C11 over the
whole batch before ANY per-sample backward can start. The derivatives are even
formulated from post-activations (selu′/tanh′/sigmoid′ in terms of a —
MATH_ANALYTIC_GRADIENTS §2) precisely so pre-activations never need storing.

Pass B is separate for a different reason: `dW = Σ over (sample,site) of a⊗δ`
is a cross-sample reduction whose natural parallel shape is thread-per-WEIGHT,
not thread-per-sample — a different dispatch geometry, hence a second pass.
And the backward is not in the ADVECT kernel because advect's 1M particles
aren't being trained on — see Q3: attaching a backward there is the dense-O(N)
trap, not a saving.
