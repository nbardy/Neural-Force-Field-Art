# Pixel GANs — four 2D drawing games

These are **not** the relational adversary (tuples → predict F). They operate on
a low-res soft density image of the particle cloud. None of them need a dataset
of real artworks except where noted; negatives are synthetic.

Shared trunk (all kinds):

```text
D = soft_bilinear_splat(particles)     # G×G, mass G²/B → mean dens ≈ 1
feat = ReLU(conv3×3(D))                # E channels
soft = softmax(feat · codebook)        # K soft codes per cell
```

Reverse-mode only (no JVP): scalar losses → VJP through critic and soft splat
into a virtual one-step `pos' = pos + dt·F` → field weights via `extGrads`.

---

## 1. `Pixel · VecField` — predict a 2D direction field

**Idea.** Discriminator outputs a 2-vector at every grid cell — a spatial
force/direction field — from the density drawing alone.

```text
V̂(x,y) = linear(soft(x,y)) ∈ ℝ²          # shared head, per cell
Y(x,y) = F(cell_center(x,y))              # true field at that pixel coord
R = mean_{cells with D>ε} soft‖V̂ − Y‖
```

Disc minimizes R; field maximizes R through `D(pos')` (and optionally Y).

**Why it fits.** Matches “predict a special field” and “what direction a
particle there would actually go.” A global diagonal is easy *if* density
gives it away — residual collapses; spatially varying F that disagrees with
what density suggests keeps the game alive.

---

## 2. `Pixel · NextFrame` — image → next image

**Idea.** Drawing-only temporal prediction: given the current density, predict
the next density.

```text
D̂₁ = head(soft(D₀))                      # per-cell scalar
D₁  = soft_splat(pos + dt·F)              # actual / virtual next drawing
R = mean soft|D̂₁ − D₁|
```

Disc minimizes R; field maximizes R (make the next frame surprising).

**Why it fits.** Closest to “predict the next image given the current image.”
No force labels in the critic I/O — pure drawing→drawing.

---

## 3. `Pixel · RealFake` — classifier with synthetic negatives

**Idea.** Classic real/fake, but “real” = density of the live cloud and
“fake” = density of the **same count of fully random** positions (uniform in
the unit square, same splat mass). No external image dataset.

```text
logit = MLP(GAP(soft(D)))
L_disc = BCE(logit_real, 1) + BCE(logit_fake, 0)
L_gen  = −logit(D(pos'))                  # look more “real” than random
```

**Why it fits.** Answers “we don’t have fake/real pairs” by manufacturing
fakes every step. The critic learns “structured cloud vs spray of noise.”

---

## 4. `Pixel · Inpaint` — masked drawing completion

**Idea.** Drawing-native: zero out a random block of the density, predict the
missing patch from context (same head as NextFrame).

```text
mask = random axis-aligned block (~25% of cells)
D_in = D · (1 − mask)
D̂   = head(soft(D_in))
R = mean_{mask=1} soft|D̂ − D|
```

Disc minimizes R; field maximizes R through `D(pos')` (harder-to-complete
drawings).

**Why it fits.** Forces the critic to use spatial context; the field is
pressured toward structures that aren’t locally interpolable — filaments,
gaps, asymmetry — not a single fillable diagonal stripe.

---

## Gallery names

| Kind | Piece name |
|---|---|
| `vec-field` | `Pixel · VecField` |
| `next-frame` | `Pixel · NextFrame` |
| `real-fake` | `Pixel · RealFake` |
| `inpaint` | `Pixel · Inpaint` |

## The critic must be the ONLY gradient

Every Pixel piece ships `fieldLoss: ZERO_FIELD_LOSS`. This is load-bearing, not
housekeeping.

The field trainer's pass B builds one gradient and hands it to one Adam:

```wgsl
// src/render/webgpu/train_wgsl.ts
g = <structural field loss>;
g = g + extGrad0[t];      // ← the critic's entire influence
grads[t] = g;
```

So a piece that declares both a structural loss *and* a game is running two
optimizers against one weight buffer, and the scales are nowhere near each
other. Measured at the shipped dims (`tools/pixel_disc_authority_probe.ts`):

| piece fieldLoss | ‖extGrad‖ | ‖grads‖ (total) | critic authority |
|---|---|---|---|
| `ZERO_FIELD_LOSS` | 3e-4 … 9e-3 | same | **100%** |
| `W_CHAOS .2 / W_ISO .6 / W_DIV .1` | 5e-4 … 8e-3 | ~8.5 | **0.006% – 0.09%** |

The failure is silent by construction. Nothing errors, no pass is skipped, no
gradient is non-finite, `tools/pixel_disc_test.ts` stays green — the artwork is
just 99.99% W_ISO, which looks exactly like "the pixel adversary does nothing."

Two corollaries worth remembering:

- **It is not a device bug.** All four pieces were first reported dead on a
  phone. `sampleAndSplat` normalizes by `uni.width`/`uni.height`, so authority
  is identical at 390×844 and 1280×800 — the probe asserts both. A pixel piece
  that behaves differently across viewports has a *new* bug.
- **`pixelDisc.weight` is only meaningful once the critic is uncontested.**
  Adam rescales by `sqrt(v)`, so a small-but-sole `extGrad` still takes
  full-size steps; the same `extGrad` summed under a structural loss is noise
  no matter how the weight is set. Shape a piece with `weight`, not by
  reintroducing `W_ISO`.

## The critic kernels are workgroup-parallel (they were not, and it showed)

`criticDisc` and `criticGen` used to be `@compute @workgroup_size(1)`: **one**
GPU thread walked all G² cells, holding `cFeat[E·G²]` and `cSoft[K·G²]` (plus
`dSoft`, `gD`, `gW` in the gen pass) as function-scope private arrays. Cost grew
as G²·(E+K) in both work *and* per-thread private memory, with no parallelism.

Advect + render alone are ~16 ms at 80k particles, so the G=16 gallery config
put the whole piece at ~37 ms/frame — 27 FPS on a fast desktop GPU, and on a
phone slow enough that the canvas visibly held the *previous* artwork. That is
what "the pixel pieces freeze on mobile" was.

They are now parallel on two axes:

- **Activations are cell-parallel** — grid-strided, so G² may exceed the
  workgroup. The per-cell workspace (`cFeat`, `cSoft`, `dSoft`, `gf`, per-cell
  head grads) lives in the tail of `scratch`, sized by `pixelScratchBytes(field,
  batchCap, dims)`. It is deliberately NOT workgroup memory — `cSoft` alone is
  16 KB at G=16, which is the entire mobile `maxComputeWorkgroupStorageSize`.
- **Gradients are weight-parallel** — one invocation owns one weight index and
  sums over all cells. No atomics and no partials buffer are needed, because
  every backbone weight gradient is already a sum over cells for a fixed weight;
  inverting the loop nest gives each output a single owner. It also preserves
  the old serial summation order, which is why most of `gW` is bit-exact.

Measured on an M-series desktop GPU (`tools/pixel_disc_cost_probe.ts`, 80k
particles, b=256, dualFourier field), median of 15:

| config | vec-field | next-frame | real-fake | inpaint |
|---|---|---|---|---|
| G=16 E=8 K=16 h=32 — **before** | 19.3 ms | 10.7 ms | 22.3 ms | 18.0 ms |
| G=16 E=8 K=16 h=32 — **after (shipped)** | **1.5 ms** | **1.2 ms** | **1.4 ms** | **1.3 ms** |

G is no longer a frame-time dial; the critic sits below per-dispatch overhead,
and advect+render dominate completely.

### Two traps this rewrite has to keep avoiding

- **Workgroup storage is capped at 16 KB.** `pixelCritWorkgroupBytes()` reports
  what the kernels declare (peak 1408 B, real-fake at K=16/h=32) and
  `pixelDiscShader` **throws** above 16384, so the mobile limit cannot be
  reintroduced silently. Never put anything that scales with G² there.
- **Barriers must be `workgroupBarrier(); storageBarrier();`** — the workspace
  is a *storage* buffer, so omitting the storage half fails silently: execution
  still syncs and a lane merely reads a stale neighbour. Every cell loop is
  grid-strided rather than early-returning, precisely so no lane skips a
  barrier in divergent control flow.

### Numerical equivalence is the bar, not speed

`§3` of `pixel_disc_test.ts` only checks norms, and norms hide a lot. When
changing these kernels, verify against a float64 reference, not against the
previous GPU kernel — "differs from before" is not "wrong", and the rewrite is
in fact *closer* to the float64 `gW` than the serial original was for
`vec-field` and `next-frame`.

## Verify

```bash
bun tools/pixel_disc_test.ts
```

Kernel-vs-oracle correctness (CPU oracle, gradients, GPU smoke). Note it runs
`G=8 E=4 K=8` on a small raw field — smaller than the gallery's
`G=16 E=8 K=16 hidden=32` on `ARCH.dualFourier`.

```bash
bun tools/pixel_disc_authority_probe.ts
```

Regression guard for the above: runs the **shipped** dims and asserts each
kind's critic still owns ≥50% of the applied gradient, across both viewports.

```bash
bun tools/pixel_disc_cost_probe.ts
```

Frame-budget guard: times one `encodeStep` per kind at the shipped dims and
fails if any exceeds 12 ms. Run it after touching G/E/K or the critic kernels.
Quit anything else using the GPU first — a browser tab rendering the same piece
moved these numbers 2–3× run to run, which is why it reports a median of 15.
