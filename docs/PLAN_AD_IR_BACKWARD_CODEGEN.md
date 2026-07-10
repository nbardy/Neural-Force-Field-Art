# Source-to-source AD: generating the backward WGSL from the forward codegen

**Status:** M0–M4 SHIPPED (2026-07-10), M5 (forward-mode JVP) landed in the IR.
Sibling to `MATH_ANALYTIC_GRADIENTS.md` (the *hand-derived* math this AD pass
must reproduce bit-for-bit) and `PLAN_FUSED_TRAIN.md` (the shipped trainer).

**M4 outcome (fused training for every field type):** `train_wgsl.ts` now
generates the backward for all four model types — standard, SIREN, Fourier,
HashGrid — and `main.ts` routes them ALL through the fused trainer (the
`?train=tfjs` fallback remains for A/B). Per-type derivative machinery, each
piece exactly what §5 predicted:
- **SIREN** — sin layers checkpoint their PRE-activation to scratch
  (`pOff` blocks); backward computes `cos(stored s)` (1 SFU op — recompute
  beat the extra `cos` store, §5.2). The checkpoint requirement was surfaced
  mechanically by the IR's sin rule (`ad/autodiff.ts`).
- **Fourier** — `encodeSite` stores γ(u) per site; the encoding jacobian
  REUSES the stored sin/cos features as their own derivatives
  (d sin(ωx)/dx = ω·cos(ωx) = ω·enc[o+2]) — zero new transcendentals, the
  derivative-from-value trick applied to the encoding (§5.1-style reuse).
- **HashGrid** — bilinear-interp jacobian (grid-value differences × (gs−1),
  clip-masked to match tfjs `clipByValue`); pass A stores per-site dL/dEnc;
  pass B gained a `grid` segment handler that gather-side scatters
  corner-weight × dEnc into each cell (tfjs's onehotᵀ matmul backward,
  expressed per-thread).
Verification (tools/train_types_test.ts, real Metal via bun-webgpu): loss +
per-variable grads vs tfjs-autograd fixtures for standard/siren/fourier/
hashgrid (K=1) and siren (K=4 BPTT) — **worst cos = 1.0000000 across all
five fixtures**, and every type's loss strictly drops over 30 fused steps.
The IR oracle (`tools/ad_train_test.ts`) independently reproduces the
siren/fourier/standard gradients in pure JS (hashgrid's data-dependent gather
is not expressible in the static scalar graph — Metal-verified only).
Fixtures: `tools/grad_reference.ts` gained `MODEL=` (fixture recipe unchanged
otherwise). The encoding/activation dispatch happens at CODEGEN time — each
generated shader still has exactly one path.

**One-line thesis:** `train_wgsl.ts` is, today, *the hand-written output of what
reverse-mode autodiff would generate* for the standard architecture. Adding
SIREN / Fourier / HashGrid by hand means re-running that human process per type.
Instead: build a small **expression IR + reverse-mode pass that emits the
backward WGSL**, so a new field type = declare its forward + one local
derivative rule, and the backward, δ-chains, BPTT recurrence and loss gradients
fall out. Verified against the existing tfjs fixtures at `cos = 1.0`.

---

## 0. The vocabulary, pinned down (kill the false dichotomies)

Several framings sound opposed but are not. Getting these right is the whole
design rationale.

**"Autograd vs analytical" is a false split.** Reverse-mode AD produces the
*exact, closed-form analytical* gradient — the same expression a human derives,
to the last bit. It is **not** finite differences. The real axis is *who applies
the chain rule*:

| Kind | What it is | Used here? |
|---|---|---|
| Numerical (finite diff) | `(f(x+ε) − f(x))/ε`, approximate | no (except the chaos/div *probes* — see §6) |
| Hand-analytical | a human writes the chain rule | ← `train_wgsl.ts` today |
| Machine-analytical (AD) | a program applies the chain rule | ← this plan |

Hand and AD produce the **same** analytical answer. That is *why* the current
kernel verifies against tfjs autograd at `cos = 1.0000000`.

**"Is this a WGSL compiler?" No.** We do not parse WGSL, build a language AST, or
do dataflow analysis on shader source. We build a tiny **expression graph in
TypeScript** whose nodes know how to *print* a line of WGSL. Think **micrograd**
(Karpathy's ~100-line autodiff) except each node *emits text* instead of
*executing*. ~15 op types cover this whole domain.

**Nearest real systems, in order of closeness:**
- **JAX** — `grad` does source-to-source AD on the `jaxpr` *ahead of time*,
  producing a *new function*; XLA then compiles it. This is exactly our shape:
  transform the forward *program* into a backward *program* (WGSL text), at build
  time, not a runtime tape.
- **Enzyme** — AD at the LLVM-IR level, at compile time. Our idea, one level
  lower. The best mental model for "AD over an IR that then lowers to a kernel."
- **`torch.compile` (Inductor)** — traces a graph → generates fused Triton/C++
  *at runtime*, tape-based autograd. Fair one-liner ("torch.compile for WebGPU")
  but tape/runtime, so a looser fit than JAX.

**Not ONNX.** ONNX is an *inference interchange format* — a serialized op graph.
ONNX Runtime's WebGPU EP runs **forward inference** via a **library of
pre-written per-op kernels** (+ some fusion patterns), *not* generated fused
*backward* kernels. ONNX Runtime *Training* (ORTModule) builds gradient graphs
but is a separate, heavier CPU/CUDA path. Philosophy is opposite: ONNX =
portable graph + general kernel library, op-at-a-time; ours = specialize-and-fuse
the whole net into one unrolled kernel via codegen. (The `onnxruntime` dep in
`package.json` is for the *separate* CLIP/splat inference work — unrelated.)

---

## 1. Why the current split exists (not a revert)

The standard Helmholtz field still trains **fully fused**: analytical backward,
GPU-resident, `cos = 1.0`, ~1 ms. Untouched.

The new types (SIREN / Fourier / HashGrid) train via **tfjs autograd** because
the fused trainer's backward is **hand-written and specialized to the standard
architecture** (selu/tanh, raw `[x,y]` input). tfjs differentiates *any*
architecture automatically, so routing the new types through it landed all three
**verified + selectable** in one session, with their *advect* still on the fused
forward kernel. The cost: their `learn` step is tfjs (~25 fps) not fused (~1 ms).

To make them fused-fast we need *their* analytical backward. This doc is the
argument for **generating** it rather than hand-writing three more.

---

## 2. Where the 766 lines actually go (it is not bloat)

The *math per derivative is one line*. `actDeriv` (train_wgsl.ts:180) is a switch
of one-liners; `tanhD(a) = 1 - a*a` (train_wgsl.ts:161) is the entire "calculus."
The line count is **codegen plumbing**, and it splits cleanly:

| Bucket | ~lines | AD-generatable? |
|---|---|---|
| Scratch memory layout | ~55 | ❌ plumbing — but *derivable* from what bwd checkpoints (§4) |
| Forward-store emit | ~30 | ❌ plumbing |
| Backward δ-chain emit (`bwd_head_*`) | ~45 | ✅ mechanical (the reverse sweep) |
| K-step BPTT recurrence (two adjoints) | ~150 | ✅ **this is literally reverse-mode over a chain** |
| Per-loss-term gradients (chaos/iso/div/spiral) | ~120 | ✅ mechanical |
| Multi-WG reduction + covariance + Adam | ~200 | ❌ plumbing (stable across field types) |

So ~60 % (the ✅ rows) is chain-rule-applied-mechanically = exactly what an AD
pass emits. The ❌ rows are GPU scaffolding that **does not change when you add a
field type**, so they stay hand-written.

```
today:  add a field type → hand-write forward + backward + scratch + loss grads
after:  add a field type → declare forward + one-line local rule.  Done.
```

---

## 3. The backward math this system actually uses

Ground truth (see `MATH_ANALYTIC_GRADIENTS.md` for the full derivation; this is
the compressed index, keyed to `train_wgsl.ts`).

### 3.1 Per-layer VJP (`bwd_head_*`, train_wgsl.ts:231)
Forward layer: `s = W·x + b`, `a = φ(s)`. Reverse:
```
output layer:   δ_out = dOut ⊙ φ'(a_out)
hidden layer l: δ_l   = φ'(a_l) ⊙ (Wᵀ_{l+1} · δ_{l+1})
input grad:     du    = Wᵀ_0 · δ_0          (a 2-vec for raw input)
weight grad:    dW    = δ · aᵀ_in           (assembled in pass B)
```

### 3.2 Activation derivatives (the "derivative-from-value" trick — §5)
```
selu'    = λ           (s>0)      = a + λα   (s≤0, from post-act)
tanh'(a) = 1 − a²
sigmoid'(a) = a(1 − a)
sin'(s)  = cos(s)      ← NOT recoverable from a = sin(s). The SIREN blocker.
```

### 3.3 Loss seeds `dL/dF` (train_wgsl.ts:552)
- **chaos** (`−log` of finite-diff separation):
  `cch = −(W_chaos/N)·(1/(sep+ε))·(1/denom)/sq`; seeds `dfx, dfy, df0`.
- **divergence** (`g = ((fx.x−f0.x)+(fy.y−f0.y))/h`, loss `g²`):
  `gd = W_div·2g/(N·h)` into `dfx.x, dfy.y, df0.{x,y}`.
- **isotropy** (`Liso = (D²+4C01²)/S²`, `D=C00−C11`, `S=C00+C11+ε`), gradient
  through the batch covariance, computed once in `finalize` (train_wgsl.ts:522):
  ```
  dLiso/dC00 =  2D/S² − 2·Liso/S
  dLiso/dC11 = −2D/S² − 2·Liso/S
  dLiso/dC01 =  8·C01/S²
  ```
  then per force: `dFs.x += W_iso(2·dC00·Fs.x + dC01·Fs.y)/NK`, symmetric for y.
- **spiral** (winner-take-all over turns, Archimedean `rsp = b·max(θ,0)`):
  `spc = (W_spiral/N)·2(r − rspBest)`; grad flows through `r` and `atan2` to
  `pos_K` (closed-form polar jacobian, §5).

### 3.4 The K-step BPTT recurrence (train_wgsl.ts:606) — *this is reverse mode*
Two adjoints, `dpos` and `dvel`, walked back over the integrator
`velPre = (v+Fs)·friction`, `v = clamp(velPre)`, `pos = pos + v`:
```
init:  dpos = spiral_grad + dpn/res ;  dvel = 0
for k = K−1 … 0:
  dvel += dpos                          # pos_{k+1} used vel_{k+1} directly
  dpre  = dvel ⊙ clampMask
  dFs   = dpre·friction   (+ isotropy contribution)
  dF    = dFs·forceMag
  du    = head0.bwd(dF·(1−α)) + head1.bwd(dF·α)
  dpos += du/res                        # dL/dpos_k
  dvel  = dpre·friction                 # dL/dvel_k
```
**Key insight for the AD pass:** a two-adjoint recurrence like this is not a
special trick — it is *precisely* what reverse mode over an unrolled chain
produces. Replicate the forward graph K times with `(pos,vel)` threaded between
copies, run the generic reverse pass, and this recurrence is the output. That is
where the ~150 hand-written lines vanish.

---

## 4. How the IR + reverse pass works (3 steps)

**Step 1 — build a graph, not strings.** The codegen runs the same shape, but
`a = sin(s)` produces a `Node`, not text. One declaration per op is all you write
to add an activation/encoding:
```ts
const Sin = {
  forward:  (x) => node("sin", [x], (r) => `sin(${r(x)})`),
  backward: (n, g) => [`${g(n.in[0])} += ${g(n)} * cos(${save(n.in[0])});`],
  //                                              ↑ references cos(x) → forces x to be checkpointed
};
```

**Step 2 — emit forward.** Walk nodes in order, print one line each
(`let n7 = sin(n6);`), and checkpoint exactly the nodes a backward rule
referenced via `save(...)`. This **replaces the hand-written scratch layout** —
the layout becomes *derived* from what the backward needs, not guessed. The
SIREN storage problem (§5) surfaces *mechanically*: because `Sin.backward` asks
for `cos(x)`, the graph marks `x` (pre-activation) as must-store. Nobody has to
remember it.

**Step 3 — emit backward.** Give each node a `grad` accumulator. Seed the output
node's grad with `dL/d(output)`. Walk nodes in **reverse**; each node prints its
local rule pushing gradient into its inputs. Reaching an input node, its `grad`
*is* `dL/d(input)`. ~40 lines of driver + one rule per op. This is exactly the
δ-chain loop (train_wgsl.ts:249) — generated instead of typed.

**What does NOT come free (stays hand-written, but is field-type-stable):**
bindings, the multi-workgroup reduction, covariance `finalize`, Adam, and
HashGrid's atomic scatter. The IR emits the math *body*; the harness wraps it.

---

## 5. Simplifying the derivatives — the Inigo-Quilez lane

iq's signature move (analytical SDF normals instead of 6-eval finite-difference
normals): derive the closed form once, reuse quantities you already computed,
exploit structure. Direct analogues here:

1. **Derivative-from-value (primary, already used).** `φ'` in terms of the
   *post-activation* `a`, not `s`: `tanh'=1−a²`, `sigmoid'=a(1−a)`,
   `selu'=a+λα`. Zero extra transcendentals in the backward — it reads stored
   `a`. This is *the* iq reuse trick, and it is why scratch stores post-acts.

2. **SIREN breaks it — and the fix is a measurable ALU/memory tradeoff.**
   `sin'=cos(s)` is not recoverable from `sin(s)` (sign ambiguity;
   `cos=±√(1−sin²)` needs a sign bit anyway). Options:
   - store `cos(s)`: +1 float/neuron scratch, 0 transcendental in bwd.
   - store `s`, recompute `cos(s)`: 1 SFU op in bwd (cheap on GPU), less
     bandwidth. **Measure** — on Apple GPUs `cos` is one special-function-unit
     op, so recompute may beat the extra store. The AD pass makes either a
     one-line policy choice, not a rewrite.

3. **Sparse / winner-take-all gradients.** Spiral's loss is `min_k`; gradient
   flows *only* to the argmin branch — skip the K losers. Already exploited
   (`bestTheta`). AD respects this automatically if `min` is a node with a
   select-mask backward.

4. **Closed-form polar jacobian.** No AD-through-`atan2` needed:
   `∂φ/∂(x,y) = (−y, x)/r²`, `∂r/∂(x,y) = (x,y)/r`. Register these as the
   `atan2`/`length` local rules once.

5. **Constant-coefficient BPTT collapse.** `friction`, `forceMag` are constants,
   so the *unmasked* adjoint recurrence has constant multipliers → a geometric
   series with a closed form. Clamp/wrap masks make it piecewise, but the common
   (unclamped) path collapses. A CSE/algebraic pass can spot this.

6. **CSE + algebraic simplify pass over the generated graph.** This is the
   answer to "can we simplify better than naive autograd?": **yes.** Pure
   *symbolic* diff suffers **expression swell** (the derivative of a deep
   composition explodes because it does not share subexpressions — the historical
   reason AD beat CAS for ML). Naive AD, conversely, can emit redundant terms.
   The state-of-the-art (JAX/XLA, Enzyme, tinygrad) is the **hybrid**:
   - AD for global structure (avoids swell, generates δ-chains + BPTT),
   - symbolic-quality *local* rules per op (`1−a²`, not mechanical `sech²`),
   - a **CSE + constant-fold + strength-reduce pass** over the emitted graph.
   That combination beats *both* naive-AD and naive-symbolic. It is worth a small
   peephole pass so the generated WGSL is as tight as the hand-written version.

### 5.1 The deepest iq-style win: forward-mode kills the finite-diff probes
The chaos and divergence losses currently estimate the Jacobian `∂F/∂p` by
**finite differences** — `fx = F(pn + h·x̂)`, `fy = F(pn + h·ŷ)` — costing 2 extra
MLP evals *and* introducing `h`-tuning + finite-diff error. The exact analytical
Jacobian is just a **forward-mode (JVP) pass** through the same MLP
(`∂F/∂p = Wᵀ_L diag(φ') … Wᵀ_0`). An IR that supports forward-mode too replaces
the probes with the exact derivative: fewer evals, no `h`, no truncation error.
This is the direct analogue of iq swapping finite-difference SDF normals for the
analytical gradient — and it is a *quality* win, not just speed. (Defer past v1,
but the IR should be built so forward-mode is a second traversal of the same
graph, not a rewrite.)

---

## 6. Plan (de-risked, incremental — the tfjs fixtures are the oracle)

The safety net: `cos = 1.0000000` against tfjs autograd + the 47 green checks.
Every step below is "regenerate, re-run, require cos still 1.0 and all tests
green" → each refactor is *proven equivalent* to the hand-written version.

- **M0 — core IR.** `src/render/webgpu/ad/`: `Node` types (const, var, add, mul,
  matvec, `selu/tanh/sigmoid/sin`, dot, length, atan2, select/min), a forward
  emitter, a reverse-mode pass, a WGSL string emitter, and a small CSE pass.
- **M1 — one head, prove parity.** Generate `bwd_head_0` for the *standard* arch
  from the IR; numerically diff against the existing hand-written `bwd_head_0`
  (and against tfjs). Require `cos = 1.0`. This validates the whole approach on
  known-good ground before touching anything shipped.
- **M2 — losses.** Express chaos/iso/div/spiral in the IR; regenerate their
  seeds; parity vs the hand-written `dL/dF`.
- **M3 — BPTT.** Build the K-step unrolled graph; confirm the two-adjoint
  recurrence emerges and matches train_wgsl.ts:606; parity vs fixtures.
- **M4 — new field types for free.** Add SIREN (store `cos`/pre-act via the
  checkpoint mechanism), Fourier (`γ` encoding node with `ω·cos(ωx)` local rule),
  HashGrid (bilinear-interp node + atomic-scatter backward). Each verified
  against its existing tfjs training run.
- **M5 (stretch) — forward-mode JVP** to replace the finite-diff chaos/div
  probes (§5.1). **DONE (IR layer, 2026-07-10):** `ad/autodiff.ts` gained
  `jvp(g, roots, seeds)` — seeds keyed by NODE ID so derived nodes (the probe
  position `pn`) can be seeded directly, which is exactly the spatial-jacobian
  use case. `ad/losses.ts` gained `chaosTermExact`/`divergenceTermExact` (the
  h→0 limits of the FD terms) and `RolloutCfg.probes: "fd" | "jvp"` selects
  them in `buildSample` (default stays "fd" — the SHIPPED loss semantics;
  flipping changes what the art trains toward, so it is opt-in).
  Verified (tools/ad_jvp_test.ts): per-op JVP ≈ central FD; JVP columns ≡
  reverse-mode rows to 2e-16 on standard AND SIREN heads; the FD probe terms
  and their full training gradients converge to the exact-probe ones as h→0
  (grad cos 0.99997 → 0.999999978 across h=1e-2→1e-4) — reverse-over-forward
  composes with zero extra machinery because jvp rules BUILD IR.
  Not yet done: emitting the JVP probes in the WGSL trainer (would change the
  shipped loss; do it behind a trainer option + fresh fixtures if the exact
  probes prove artistically better).

**Non-goals for v1:** replacing the hand-written harness (bindings, reduction,
Adam, atomic scatter); a general WGSL compiler; Lean. **Lean's** only role would
be *verifying* the emitted backward equals the true derivative (`mathlib` can
prove `d/dx sin = cos`) — but we already have a cheaper oracle in the tfjs
fixtures, so Lean is "nice someday," not on the path.
