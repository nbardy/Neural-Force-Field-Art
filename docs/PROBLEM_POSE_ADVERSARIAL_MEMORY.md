# Problem pose — adversarial particle art without ground truth

**Audience:** chief scientist / research lead  
**Repo:** Neural Force Field Art  
**Status:** fused WebGPU paths shipped; aesthetics and training dynamics still open  
**Refs:** `docs/PIXEL_DISC.md`, `docs/PLAN_RELATIONAL_ADVERSARY.md`, `docs/ADVERSARY_STATUS.md`

---

## 1. Setting

We train tiny neural **force fields** that advect a particle cloud for generative
visual art (WebGPU-only). There is **no dataset of “good” drawings** and no
external real/fake image corpus. “Quality” is whatever online objective we put
on the field.

Two adversarial families exist side by side:

| Family | Observer | Target | Gallery |
|---|---|---|---|
| **Relational adversary** | particle tuples / encodings → predict **F** (or relational vectors) | surprise / WTA residual | WTA, Agree+Disagree, etc. |
| **Pixel GANs** | soft **density image** of the cloud | four 2D games (below) | `Pixel · *` |

Production paths are **fused WGSL** (critic + reverse-mode through density or
tuples → `extGrads` into the field). CPU code is oracle/tests only.

---

## 2. What we designed (Pixel GANs)

Shared trunk: soft bilinear splat → 3×3 conv → soft codebook on a `G×G` density.

| Kind | Game | Disc objective | Field (gen) objective |
|---|---|---|---|
| **VecField** | density → per-cell 2-vector | match `F` at cell centers | maximize residual via `D(pos')` |
| **NextFrame** | density → next density | predict `splat(pos+dt·F)` | maximize next-frame surprise |
| **RealFake** | classic classifier | BCE: live density vs **fully random** same-B spray | maximize “real” logit of `D(pos')` |
| **Inpaint** | masked completion | reconstruct random ~25% hole from context | maximize inpaint residual |

**Intentional design bets:**

- Prefer **prediction / surprise** games over pure real/fake when possible
  (NextFrame, VecField, Inpaint, relational WTA) — no need for a real art dataset.
- Use **K predictor heads** (relational WTA) for *in-frame* multi-modality
  (“several plausible F’s for one context”), not as temporal memory.
- Generator pressure is **reverse-mode only** through a differentiable soft
  density (or relational path), not JVP.

**Also retained:** the original relational vector adversary; pixel modes do not
replace it.

---

## 3. Issues we’re seeing / expect

### 3.1 Oscillation / thrashing (primary concern)

Without a fixed real distribution, disc and field **chase each other**:

> disc learns “cloud / force looks like LEFT” → field flips RIGHT → disc
> refits → field flips LEFT → …

Symptom class: large-scale coherent flips (all-left ↔ all-right), slow
periodicity, or residual that collapses then reappears as a different global
mode — rather than refining spatial structure.

**Hypothesis:** this is classic online GAN dynamics with a *moving* “real”
(the live cloud / live F), amplified by a tiny critic and a strongly coupled
particle–field feedback loop.

### 3.2 K heads ≠ memory of past generators

WTA’s K guesses help **conditional** ambiguity in one step (don’t average two
force modes into a useless median). They do **not** remember what the generator
did 10 training steps ago. So K alone is unlikely to stop left↔right thrashing.

We likely need **temporal memory**: e.g. a ring buffer of past densities /
`(u,F)` tuples, or weight-checkpoint snapshots of past generators, so the disc
trains against a *history* of gens — not only the instantaneous one.

### 3.3 RealFake negatives are too easy / too pure

Fully random spray is a soft negative (“structured vs noise”). Once the critic
sees that, pressure may not shape *interesting* structure — only “not spray.”
Under discussion: **%-corrupt** fakes (mix live particles with noise),
multi-fake batches, loss reweighting, hinge vs BCE. Class imbalance (many fakes,
one real) invites an “always fake” disc cheat unless weighted or balanced.

### 3.4 Earlier pixel-density collapse (historical)

A prior single-mode density disc (global/tile mean force) tended toward
**diagonal stripes**: easy for the critic once residual → 0, adversarial
pressure dies. VecField + spatial targets and isotropy/chaos field losses were
meant to harden that; **live aesthetic validation of the four new pieces is
still thin** (unit/GPU smoke tests pass; visual A/B is incomplete).

### 3.5 Inpaint is not a classic GAN

Inpaint is a **completion / surprise** game (mask → predict hole; field makes
holes hard to fill). Easy to misread as real/fake. Worth keeping the taxonomy
clear when diagnosing failures.

### 3.6 Engineering constraints (secondary)

- Agree+Disagree already uses 2 `extGrad` slots; pixel disc cannot stack in the
  same frame without a third slot.
- Pixel critic is currently single-threaded `@workgroup_size(1)` (fine at G≈16).

---

## 4. Open questions for the chief scientist

1. **Is replay the right stabilizer?** Density/tuple ring buffer vs EMA field
   weights vs discrete weight checkpoints — what theory / prior art should we
   copy (experience replay GANs, historical averaging, unrolled GAN, …)?

2. **Which game is the research wedge?** NextFrame (pure drawing→drawing) vs
   relational WTA (vector space, already battle-tested) vs hardened RealFake
   (closest to textbook GAN, most thrash-prone).

3. **What should “success” mean without GT?** Persistent residual? Spatial
   spectrum? Human look? Coverage / filament metrics? Avoid optimizing a proxy
   that silently prefers noisy TV or diagonal fill.

4. **Should prediction games and real/fake share one memory substrate?**
   One replay bank feeding all pixel kinds + relational adversary, or
   kind-specific buffers?

5. **Do we accept oscillation as aesthetic** (breathing left/right as art) and
   only damp *pathological* collapse — or do we require asymptotic stability of
   the disc–field pair?

---

## 5. Proposed next experiments (engineering-ready)

Ranked cheap → expensive:

1. **Ring-buffer replay on RealFake + NextFrame** — store last N soft densities
   (and/or particle batches); disc sees current + historical fakes/reals;
   gen still only from live `pos'`.
2. **%-corrupt fakes** instead of pure spray; optional hinge loss.
3. **EMA shadow field** for disc targets / extra fakes.
4. **Weight-checkpoint bank** (last 10 generators) if replay of outputs is
   insufficient.
5. Visual soak + HUD of disc/gen loss spectra to confirm thrash vs collapse.

---

## 6. One-sentence summary

We built fused pixel and relational adversaries for particle art **without a
real dataset**; the central scientific risk is **online disc–gen thrashing**,
and **K multi-heads do not substitute for memory of past generators** — we need
a clear call on replay / checkpoint memory and which game is the wedge.
