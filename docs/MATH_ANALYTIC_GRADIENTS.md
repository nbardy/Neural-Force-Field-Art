# The analytic gradients of the fused TRAIN kernel — complete derivation

This is the full math behind `src/render/webgpu/train_wgsl.ts` (pass A backward +
pass B assembly), written so the WGSL can be re-verified line by line. The code is
the ground truth — it is verified against tfjs autograd at **cos=1.0000000 /
relMax 1.8e-5 (K=1)** and **cos=1.000000 / relMax 1.9e-5 (K=4 BPTT)** by
`tools/train_test.ts` against fixtures from `tools/grad_reference.ts`. The summary
table in `docs/PLAN_FUSED_TRAIN.md` §Loss is the short form of this document.

Notation: scalars lowercase, 2-vectors bold-ish (`pos`, `F`), `⊙` elementwise,
`⊗` outer product, `[P]` the 0/1 indicator of predicate P. All arithmetic is f32.

---

## 1. Forward definitions (K-step rollout)

Constants (all from `main.ts`, copied into `train_wgsl.ts` `LOSS` and the
`TrainPhysics` uniform — `res=(w,h)`, `forceMag`, `friction`, `maxVel`, `α`):

```
W_CHAOS = 1.0   W_ISO = 1.0   W_DIV = 0.5   W_SPIRAL = 2e-5
HH = 1e-2       SPIRAL_TURNS = 3
```

Per sample `i` (batch size N, rollout length K; sample index dropped below):

```
pos_0 = tp            (pixel coords; PCG-generated in-kernel or uploaded)
vel_0 = v0 (0 for random/uploaded batches; the sampled particle's real velocity for source:"particles" — v0 is θ-independent either way, so the backward recurrence is unchanged)

for k = 0 .. K-1:
  u_k        = pos_k / res                                   (normalized input)
  F_k        = (1-α)·h_g(u_k) + α·h_r(u_k)                   (two-head blend)
  Fs_k       = forceMag · F_k                                (scaled force)
  velPre_k+1 = (vel_k + Fs_k) · friction
  vel_k+1    = clamp(velPre_k+1, -maxVel, +maxVel)           (per component)
  pos_k+1    = mod(pos_k + vel_k+1, res)                     (floored mod, per axis)
```

Each head `h` is a dense stack (helmholtz: `2→32→32→2`, selu, selu, tanh — dims
are data, read off the live model; nothing is hardcoded). Layer `l` with kernel
`W_l` (`[in][out]` row-major, tfjs layout), bias `b_l`, activation `σ_l`:

```
a_-1 = u        (the site input)
z_l  = W_lᵀ a_{l-1} + b_l
a_l  = σ_l(z_l)
```

Pass A's forward (`fwd_head_*` in train_wgsl.ts) **stores every post-activation
`a_l` to the scratch buffer** as it computes it. `pos_0..pos_K`, `velPre_1..K`,
and `Fs_0..K-1` are also stored (the scratch header) — everything the backward
needs, and nothing else (see §2 on why pre-activations are never stored).

**Probe sites** (chaos + divergence), evaluated once at the final state:

```
pn = pos_K / res
f0 = F(pn)      fx = F(pn + (HH, 0))      fy = F(pn + (0, HH))
```

where `F(·)` is the same α-blend of the two heads. Sites are indexed
`0..K-1` = physics evals, `K, K+1, K+2` = probes — `(K+3)` sites total, each with
stored activations for both heads.

### The composite loss (exact parity with `main.ts helmholtzChaosLoss`)

```
loss = W_CHAOS·chaos + W_ISO·iso + W_DIV·div + W_SPIRAL·spiral
```

**chaos** — mean over samples of `−log(sep + 1e-6)` with

```
dxv = fx − f0        dyv = fy − f0
q   = |dxv|² + |dyv|² + 1e-12
sep = √q / (HH·1.4142 + 1e-9)
```

Note the denominator constant is the **literal `1.4142`** (not `Math.SQRT2`) —
main.ts uses that literal, so the kernel does too. Parity is with the code, not
with √2.

**divergence** — mean over samples of `g²` with

```
g = ((fx.x − f0.x) + (fy.y − f0.y)) / HH
```

**isotropy** — a batch statistic over the **scaled** forces of **all K steps**
(M = N·K vectors; at K=1 this is exactly `isotropyLoss(force)` in
`src/core/losses/isotropy.ts`):

```
C00 = (1/M) Σ Fs.x²     C11 = (1/M) Σ Fs.y²     C01 = (1/M) Σ Fs.x·Fs.y
S = C00 + C11 + 1e-6    D = C00 − C11
iso = (D² + 4·C01²) / S²
```

**spiral** — mean over samples of a winner-take-all distance to an Archimedean
spiral, evaluated at `pos_K`:

```
cx = w/2   cy = h/2   maxR = min(w,h)·0.38   b = maxR / (SPIRAL_TURNS·2π)
dx = pos_K.x − cx     dy = pos_K.y − cy
r  = √(dx² + dy² + 1e-4)                       (ε INSIDE the sqrt)
φ  = atan2(dy, dx)
d_k = (r − b·relu(φ + 2πk))²   for k = 0 .. SPIRAL_TURNS+1
spiral_i = min_k d_k                            (earlier k wins ties — see §4.4)
```

Pass A accumulates `W_CHAOS·chaos_i + W_DIV·div_i + W_SPIRAL·spiral_i` plus the
three `Fs` moments per thread, tree-reduces across the workgroup, and thread 0
writes `lossOut[0] = (Σ per-sample)/N + W_ISO·iso` and the shared
`(dL/dC00, dL/dC11, dL/dC01)` (§4.2) to workgroup memory for the backward.

---

## 2. Activation derivatives from POST-activations

The backward never needs `z_l`, only `a_l`, because each supported activation's
derivative is expressible in its own output:

| σ | σ′(z) in terms of a = σ(z) | WGSL |
|---|---|---|
| selu | `a > 0 ? 1.0507009873554805 : a + 1.7580993408473768` | `seluD` |
| tanh | `1 − a²` | `tanhD` |
| sigmoid | `a·(1 − a)` | `sigmoidD` |
| linear | `1` | — |

The selu line deserves the derivation. tfjs selu is

```
selu(x) = scale·x                 x > 0      scale = 1.0507009873554805
        = scale·λ·(eˣ − 1)        x ≤ 0      λ     = 1.6732632423543772
```

and the codegen fuses `scale·λ = 1.7580993408473768` (the constant in both
`selu` and `seluD`). For `x ≤ 0`:

```
σ′(x) = scale·λ·eˣ = scale·λ·(eˣ − 1) + scale·λ = a + 1.7580993408473768
```

Branch selection from the post-activation is exact: selu is strictly increasing
with `selu(0)=0`, so `a > 0 ⇔ x > 0`. At the measure-zero point `x = 0` both the
kernel and tfjs take the `x ≤ 0` branch and produce `scale·λ` — identical.

This is **why storing post-activations suffices**: the forward stores `a_l` only,
and every σ′ the backward needs is recovered from those stored values. Nothing
about the pre-activations survives the forward, on purpose.

---

## 3. The MLP backward per head (δ-chain, `bwd_head_*`)

Each head receives `dOut = ∂L/∂(head output)` with the **blend factor already
applied** by the caller: since `F = (1−α)·h_g(u) + α·h_r(u)`,

```
∂L/∂h_g-output = (1−α)·∂L/∂F        ∂L/∂h_r-output = α·∂L/∂F
```

(the six `bwd_head_*(df·(1−α)/·α, …)` calls at the probe sites and the two per
physics step). Inside a head with layers `0..nL-1`:

```
output layer:  δ_{nL-1} = dOut ⊙ σ′_{nL-1}(a_{nL-1})
hidden l:      δ_l[i]   = σ′_l(a_l[i]) · Σ_j W_{l+1}[i][j] · δ_{l+1}[j]
```

(`W_{l+1}[i][j]` at packed offset `weightOffset + i·outSize + j` — row = input
index, tfjs dense layout). Every `δ_l` is written to scratch next to `a_l`.
The parameter gradients are NOT accumulated here — pass B does that (§6) from
the stored `(a, δ)` pairs:

```
∂L/∂W_l[i][j] = a_{l-1}[i] · δ_l[j]          (a_{-1} = u, the site input)
∂L/∂b_l[j]   = δ_l[j]
```

Finally the **input gradient** (needed by the probe/position chains):

```
du[c] = Σ_j W_0[c][j] · δ_0[j]        c = 0,1     (inSize = 2, validated)
```

which is the `du.x / du.y` loop at the end of `bwd_head_*`.

---

## 4. Per-loss-term gradients (the `dL/dF` seeds)

All means over samples carry `1/N`; the isotropy mean carries `1/(N·K)`.

### 4.1 chaos

`chaos_i = −log(sep + 1e-6)`, `sep = √q / c` with `q = |dxv|² + |dyv|² + 1e-12`,
`c = HH·1.4142 + 1e-9`. Chain rule:

```
∂chaos_i/∂sep = −1/(sep + 1e-6)
∂sep/∂q       = 1/(2c·√q)
∂q/∂dxv       = 2·dxv          ∂q/∂dyv = 2·dyv
```

The 2s cancel, giving the kernel's single coefficient

```
cch = −(W_CHAOS/N) · 1/(sep+1e-6) · (1/c) / √q
dL/dfx += cch·dxv       dL/dfy += cch·dyv       dL/df0 += −cch·(dxv + dyv)
```

(`df0` collects both because `dxv` and `dyv` each subtract `f0`). The `1e-12`
inside `q` and the `1e-9` inside `c` are treated as constants — same as autograd.

### 4.2 isotropy

`L = (D² + 4C01²)/S²` with `S = C00 + C11 + ε`, `D = C00 − C11`, ε = 1e-6
constant. Quotient rule, using `∂S/∂C00 = ∂S/∂C11 = 1`, `∂D/∂C00 = 1`,
`∂D/∂C11 = −1`:

```
∂L/∂C00 = [2D·S² − (D²+4C01²)·2S] / S⁴ =  2D/S² − 2L/S
∂L/∂C11 = [−2D·S² − (D²+4C01²)·2S] / S⁴ = −2D/S² − 2L/S
∂L/∂C01 = 8·C01 / S²
```

— exactly the `gStats` vec3 computed once by thread 0. Then through the moment
definitions `C00 = (1/M)ΣFs.x²` etc. (M = N·K):

```
dL/dFs.x[m] = W_ISO · (2·∂L/∂C00·Fs.x[m] + ∂L/∂C01·Fs.y[m]) / M
dL/dFs.y[m] = W_ISO · (2·∂L/∂C11·Fs.y[m] + ∂L/∂C01·Fs.x[m]) / M
```

Note this seed applies to the **scaled** force `Fs_k` at every step k — it is
added to `dFs` inside the BPTT loop (§5), then picks up the `forceMag` factor on
the way into the MLP. At K=1, M = N and this is verbatim PLAN_FUSED_TRAIN §Loss.

### 4.3 divergence

`div_i = g²`, `g = (fx.x − f0.x + fy.y − f0.y)/HH`:

```
gd = W_DIV · 2g / (N·HH)
dL/dfx.x += gd     dL/dfy.y += gd     dL/df0.x −= gd     dL/df0.y −= gd
```

### 4.4 spiral

Winner-take-all: only the argmin branch `k*` carries gradient. The forward folds
`best = min(best, d_k)` in ascending k with tfjs `tf.minimum`, whose gradient
sends ties to the FIRST (accumulated) operand — i.e. **earlier k wins ties**.
The kernel's strict `if (d < best)` replicates exactly that.

At the winner, `spiral_i = (r − rsp)²` with `rsp = b·relu(θ*)`, `θ* = φ + 2πk*`:

```
∂spiral_i/∂r   =  2(r − rsp)
∂spiral_i/∂θ*  = −2(r − rsp)·b·[θ* > 0]          (relu mask; tfjs step(0)=0)
```

Position chains, with `dx = pos_K.x − cx`, `dy = pos_K.y − cy`:

- `r = √(dx² + dy² + 1e-4)` — the ε lives **inside** the sqrt, so
  `∂r/∂(dx,dy) = (dx, dy)/r` with the epsiloned `r` in the denominator.
- `φ = atan2(dy, dx)` — the tfjs atan2 Jacobian is
  `∂φ/∂(dx,dy) = (−dy, dx)/r²` where `r² = dx² + dy²` **WITHOUT the ε** (tfjs
  computes it from the raw squared sum; the kernel keeps a separate un-epsiloned
  `r2` for exactly this reason).

Combined seed on the final position:

```
dL/dpos_K += (W_SPIRAL/N) · 2(r − rsp) · [ (dx,dy)/r − b·[θ*>0]·(−dy,dx)/r² ]
```

— the `spc`/`reluMask` lines in the shader.

---

## 5. The K-step BPTT recurrence

The kernel carries **two running adjoints** per sample:

```
dpos = ∂L/∂pos_{k+1}        dvel = ∂L/∂vel_{k+1}  (excluding the pos_{k+1} path)
```

**Initial conditions** (state after the loop, k+1 = K):

```
dpos = spiral seed (§4.4) + dpn/res         dvel = 0
```

where `dpn = Σ over the 3 probe sites of bwd_head input-gradients` (blend factors
applied per §3) — the probes read `pn = pos_K/res`, so their input gradients
divide by `res` on the way back to pixel coordinates. `dvel_K = 0` because no
loss term reads velocity.

**Per transition** `k = K−1 .. 0`, inverting

```
velPre_{k+1} = (vel_k + Fs_k)·friction
vel_{k+1}    = clamp(velPre_{k+1}, ±maxVel)
pos_{k+1}    = mod(pos_k + vel_{k+1}, res)
```

the update order in the shader loop is, exactly:

```
1.  dvel += dpos                        ∂pos_{k+1}/∂vel_{k+1} = 1  (completes dvel_{k+1})
2.  mask  = [|velPre_{k+1}| ≤ maxVel]   clamp′, per component, inclusive (= tfjs clipByValue)
    dpre  = dvel ⊙ mask                 ∂L/∂velPre_{k+1}
3.  dFs   = dpre·friction               ∂velPre/∂Fs_k = friction
4.  dFs  += isotropy seed at Fs_k       (§4.2, the ONLY loss term touching interior steps)
5.  dF    = dFs·forceMag
6.  du    = bwd_head_0(dF·(1−α)) + bwd_head_1(dF·α)      (writes this site's δ's)
7.  dpos  = dpos + du/res               ∂pos_{k+1}/∂pos_k = 1  (mod ≡ identity)
                                        + force chain: u_k = pos_k/res
8.  dvel  = dpre·friction               ∂velPre_{k+1}/∂vel_k = friction
```

**Why mod contributes identity:** `mod(x, m) = x − floor(x/m)·m` and tf.mod's
gradient w.r.t. the dividend is 1 (the floor term is piecewise constant; tfjs
registers exactly this). So `pos_{k+1}` back-propagates to both `pos_k` and
`vel_{k+1}` with factor 1, and the wrap is gradient-transparent. `res` is a
constant, so the divisor path doesn't exist.

Step 8 is correct as the FULL `∂L/∂vel_k` at that point because `vel_k` feeds
only `velPre_{k+1}`; its other consumer — `pos_k = pos_{k−1} + vel_k` — belongs
to transition k−1 and is exactly what step 1 adds on the next iteration. This is
the two-adjoint invariant.

**K=1 collapse.** With K=1: `dvel` starts 0, step 1 gives `dvel = dpos`, so
`dFs = dpos ⊙ mask · friction + iso-seed` — i.e. the PLAN doc's single-step
physics line `dnewVel/dF_scaled = friction·[|pre-clip| ≤ maxVel]`,
`dnewPos/dnewVel = 1`, `mod′ = 1`, with the probe input-grads flowing back
through `÷res`. The M = N·K isotropy normalization degenerates to 1/N. K=1 is
verified to reproduce tfjs single-step autograd at cos=1.0000000 and the
original fixture loss `1.0965325832366943` bit-close.

The product of per-step Jacobians never materializes: each transition applies
its Jacobian to the running adjoints as vector-Jacobian products — friction and
the clamp mask are diagonal, mod is identity, and the field's input-Jacobian
enters only through `bwd_head`'s `du` (§3).

---

## 6. Pass B — gradient assembly + Adam

Pass A leaves scratch holding, per (sample, site, head): all `a_l` and all
`δ_l`. Pass B is thread-per-packed-weight-float (`TRAIN_WG_B = 64`); thread `t`
decodes which segment it falls in (codegen emits one range check per segment —
segments are compile-time constants from the FieldLayout):

```
kernel entry:  i = local / outSize,  j = local % outSize
               g = Σ_samples Σ_sites  a_in[i] · δ_l[j]
bias entry:    g = Σ_samples Σ_sites  δ_l[j]
```

with `a_in` = the stored site input `u` for layer 0, else the stored `a_{l-1}`.
Sites run over all K physics evals AND the 3 probes — the same weight is hit by
every site's δ, which is precisely `dW = Σ a⊗δ` over the whole unrolled graph.
`grads[t]` is always written (verification reads it even with `apply=0`).

**Adam (tfjs conventions, verified to 3.3e-7):**

```
m ← β1·m + (1−β1)·g               β1 = 0.9
v ← β2·v + (1−β2)·g²              β2 = 0.999
m̂ = m/(1−β1ᵗ)    v̂ = v/(1−β2ᵗ)    bias correction via βᵗ, t ≥ 1 on the FIRST applied step
w ← w − lr·m̂/(√v̂ + ε)             ε = 1e-7
```

`t` comes from `FusedTrainer.adamStep` (incremented before the step, clamped
`max(1, ·)` — a t=0 step would divide by zero). ε = 1e-7 is not the Adam-paper
1e-8: tfjs's AdamOptimizer defaults `epsilon = null` and falls back to the
backend epsilon, which is **1e-7 for float32** — `ADAM_DEFAULTS` in
`src/render/webgpu/train.ts` pins that. The update writes the SAME packed
weights buffer the advect kernel binds, which is the whole point.

---

## 7. Verification

All headless on a real Metal adapter via bun-webgpu; the GPU is shared, so bench
numbers carry ±30% but parity numbers are stable.

- `tools/grad_reference.ts` — the OFFLINE oracle: tfjs-CPU `tf.variableGrads` on
  the exact composite (seeds mulberry32(42)/mulberry32(7), N=256, 800×600,
  α=0.7), self-checked for bit-identical determinism across two fresh runs, with
  a K=1 regression tripwire on the original fixture loss. Writes
  `tools/fixtures/grad_ref.json` (K=1) and `grad_ref_k4.json`
  (`K=4 OUT=tools/fixtures/grad_ref_k4.json bun tools/grad_reference.ts`).
- `tools/train_test.ts` — loads the fixtures and asserts, per variable:
  - loss parity: rel < 1e-3 (measured 3.3e-7)
  - gradient parity K=1: cos > 0.99999 and relMax < 2e-3 (measured
    cos=1.0000000, relMax 1.8e-5 across all 12 variables)
  - gradient parity K=4 BPTT: cos > 0.9999 and relMax < 5e-3 (measured
    cos=1.000000, relMax 1.9e-5) — looser bound because fp32 error compounds
    through the unrolled chain, but cosine stays essentially exact
  - Adam formula parity: one applied step vs JS Adam on the kernel's own grads,
    maxΔw < 1e-5 (measured 3.3e-7)
  - training sanity: loss strictly drops over 40 self-generated fused steps
    (measured 1.156 → −0.354)
- Gradient-chain gotchas the fixture pins down (from `grad_reference.ts`):
  floored mod is grad-identity, clipByValue is grad-transparent for this piece's
  physics range even at K=4 (worst-case |vel| ≈ 13.65 < maxVel=26), no −0.5
  shift on the field path.
