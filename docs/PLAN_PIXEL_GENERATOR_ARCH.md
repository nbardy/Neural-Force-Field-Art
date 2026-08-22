# Pixel critic × generator architecture: support matrix and porting plan

Audited 2026-08-19; §2, §2a and §3 IMPLEMENTED and VERIFIED 2026-08-22 — the
support matrix in §1 and the gate quoted below are the post-change state.
Originally audited against `src/core/field/arch.ts`,
`src/render/webgpu/pixel_disc_wgsl.ts`, `src/render/webgpu/adversary_wgsl.ts`,
`src/render/webgpu/train_wgsl.ts`.

Status words are strict (same ledger as `docs/PLAN_RELATIONAL_ADVERSARY.md`):
**implemented** / **verified** / **pending** / **experimental** / **rejected**.

The generator is the neural force field `F(x)`. The pixel critics
(`docs/PIXEL_DISC.md`) consume it and push reward back through `extGrads`. This
document records which generator architectures each critic family accepts,
which refusals are fundamental, and what porting the rest costs.

---

## 1. The support matrix

`ARCH` (`src/core/field/arch.ts:51-168`) has 13 presets. Before this work the
pixel critic accepted **3**; it now accepts **5**, the same as the relational
adversary. The `pixel` column below is the post-change state.

| arch | encoding | act | heads | classes | pixel | relational |
|---|---|---|---|---|---|---|
| `mlp256` | raw | selu | 1 | 0 | no (heads) | no (heads) |
| `mlpDeep` | raw | selu | 1 | 0 | no (heads) | no (heads) |
| `mlpShallow` | raw | selu | 1 | 0 | no (heads) | no (heads) |
| `fourier` | fourier | selu | 1 | 0 | no (heads) | no (heads) |
| `fourierWide` | fourier | selu | 1 | 0 | no (heads) | no (heads) |
| `fourierSiren` | fourier | sin | 1 | 0 | no (heads) | no (heads) |
| `siren` | raw | sin | 1 | 0 | no (heads) | no (heads) |
| `hashgrid` | hashgrid | selu | 1 | 0 | no (heads) | no (heads) |
| `dualStd` | raw | selu | 2 | 0 | **yes** | **yes** |
| `dualFourier` | fourier | selu | 2 | 0 | **yes** | **yes** |
| `dualSiren` | raw | sin | 2 | 0 | **yes** | **yes** |
| `dualHashgrid` | hashgrid | selu | 2 | 0 | **yes** | **yes** |
| `familyHashgrid` | hashgrid | selu | 2 | 3 | **yes** (except vec-field) | **yes** |

The gate is now the adversary's family switch verbatim rather than a second
ladder — `classes > 0` was never the right question, and asking it refused
`familyHashgrid` for the same reason it refused the genuinely-unsupported
one-hot route:

```ts
export function classifyPixelDiscFusion(field: FieldLayout): PixelDiscFusion {
  if (field.spec.kind !== "helmholtz" && field.spec.kind !== "agree-disagree") {
    return { tag: "unsupported",
             reason: `needs a two-head neural field (got ${field.spec.kind})` };
  }
  switch (field.family.tag) {
    case "none":
    case "grid-plane":
      break;                       // family-planed hashgrid rides the dEnc path
    case "onehot":
      return { tag: "unsupported", reason: "one-hot class channels widen …" };
  }
  return { tag: "ok" };
}
```

One refusal was ADDED that this plan did not anticipate: **vec-field on a
family-planed field**. Its target is `F` at cell centres, and a cell centre has
no family — a C-plane field has C different `F`s there. Picking `cls = 0` would
fit the critic to family 0's field and look perfectly healthy, so
`pixelDiscShader` throws. `familyHashgrid` works with next-frame, real-fake and
inpaint. This is a `(kind, field)` question, so it lives at the same boundary as
the `real-fake + guesses > 1` refusal rather than in the field-only host gate.

`classifyPixelDiscFusion` returning **data** rather than throwing is deliberate
and worth preserving: the constructor needs a loud refusal, the host needs the
same answer plus a reason without throwing so it can say why a declared critic
is off (`:92-101`). It is the closest thing in the repo to a capability table —
there is no other. Support is otherwise ad-hoc across ~30 call sites gating on
`spec.kind`, `encoding.kind`, `classes`, or `family.tag`.

A note on SIREN, because the two gates read differently: the adversary's `sin`
refusal (`adversary_wgsl.ts:596-605`) walks `advL.heads` — the **predictor**
net — not the field. A SIREN *generator* is fine for both critics, carried by
`train_wgsl`'s pre-activation checkpoints (`train_wgsl.ts:265-277`), which is
why `dualSiren` passes. A SIREN *adversary head* is not.

---

## 2. Refusal 1: hashgrid encoding — was incidental (**verified** 2026-08-22)

The module header already says so (`adversary_wgsl.ts:57-65`): "the PIXEL
critic still refuses hashgrid; this port covers the RELATIONAL adversary only."

The root cause is that hashgrid is the only encoding with trainable parameters
(`encodingParamFloats > 0`, `advect_wgsl.ts:239-243`), and the pixel path has
neither the scratch to accumulate their gradient nor the reduction to scatter
it. Three concrete defects, each with a working reference in the adversary:

**(a) No `dEnc` block in the scratch layout.**

```ts
export function pixelParticleScratchFloats(field: FieldLayout): number {
  const sl = trainScratchLayout(field, 1);
  return 8 + sl.encStore + sl.siteBlk;      // pixel_disc_wgsl.ts:139-142
}
```

`sl.dEncStore` is absent, and the hand-rolled offsets confirm it — `oEnc = 8`,
`oField = 8 + sl.encStore` (`:849-854`), with no region between. Hashgrid needs
a per-site `dL/dEnc` block that `emitBwdStore`'s hashgrid arm writes
(`train_wgsl.ts:625-629`). Reference: `advScratchLayout` reserves
`dEncOff = encOff + m * fieldSl.dEncStore` (`adversary_wgsl.ts:547-548`).

**(b) The backward call has the wrong arity.**

```ts
const bwdCall = (h: number, dExpr: string) =>
  enc.kind === "raw"
    ? `bwd_head_${h}(${dExpr}, ${fieldBase(h)})`
    : `bwd_head_${h}(${dExpr}, ${fieldBase(h)}, ${encBase()})`;
```

(`pixel_disc_wgsl.ts:877-880`) — two arms, raw and fourier. Hashgrid's emitted
`bwd_head_h` takes `(dOut, base, uIn, dEncBase[, cls])`, 4-5 args
(`train_wgsl.ts:549-554`), so this produces a WGSL arity error rather than a
wrong number. Reference: `adversary_wgsl.ts:1185-1191` has the third arm.

**(c) The grid segment's gradient is dropped.**

```ts
for (const seg of field.segments) {
  if (seg.role === "grid") continue;         // pixel_disc_wgsl.ts:911-912
```

The learned feature table is `segments[0]`, `role: "grid"`, `floatOffset: 0`
(`advect_wgsl.ts:454-463`). Skipping it means `extGrads[t] === 0` for every
grid float — the encoding's trainable parameters would receive **zero**
generator reward while everything still runs and looks healthy. Reference:
`adversary_wgsl.ts:2160-2211` does the gather-side bilinear scatter, one writer
per grid float, no atomics (rationale at `:2162-2168`), transliterated from
`train_wgsl.ts:1352-1398`.

Of the three, (c) is the one that fails silently. (a) and (b) are compile or
layout errors.

### 2a. Latent bug this port hit, as predicted (**verified**)

```ts
// pixel_disc_wgsl.ts:903-905
i === 0 ? "seed" : "accumulate"
```

The `dEnc` seed-vs-accumulate choice is keyed on head index. The adversary keys
it on the **lane** (`adversary_wgsl.ts:1192-1196`), because a `fieldLane: 1`
game would otherwise `+=` into a block nobody seeded. Unreachable today — the
pixel critic is always `"blend"` (`main.ts` never passes `fieldLane`, though
`pixel_disc_train.ts:106,156` plumbs it) and hashgrid is refused. It becomes
reachable the moment hashgrid lands. Fix it in the same change.

### 2b. Scratch is not the constraint

Worth stating because it is the intuitive wrong answer. Hashgrid is the
**cheapest** encoding for scratch — `encStore = gridFeatures = 4`, against
fourier's `2 + 4*octaves = 18` (`train_wgsl.ts:296-297`). `familyHashgrid`'s 3
planes cost nothing extra; planes live in `weights`, not scratch.

```text
bytes = 4 * ( batchCap*(8 + encStore + siteBlk) + 8 + critSites*(encStore + siteBlk) )
critSites = G^2 for vec-field, else 1        (pixel_disc_train.ts:127-132)
```

`siteBlk` — the depth and width of the field heads — dominates, multiplied by
`(batchCap + nCell)`. Worked: `dualStd` [32,32] x 2 heads gives `siteBlk = 264`;
at `batchCap 512` plus 256 crit sites, ~835 KB. The `critSites = nCell`
allocation exists because `fillForceGrid` is `@workgroup_size(256)`, one cell
per invocation (`pixel_disc_wgsl.ts:462-477`), so a shared workspace would race
— documented at `:153-158`. None of that is implicated in the hashgrid refusal;
`encodeSite` is already called on the critic path at `:908`.

---

## 3. Refusal 2: class-aware fields — partly fundamental (**verified**)

The pixel critic refuses all `classes > 0`. The adversary is more precise: it
allows `family.tag === "grid-plane"` and refuses only `"onehot"`, with the
reason in code (`adversary_wgsl.ts:582-595`):

> One-hot channels widen head 1's layer-0 input, and the adversary's field
> backward has no counterpart for those rows. The family-planed hashgrid needs
> none: the label only moves the grid's cell index, so it rides the dEnc
> machinery the reward already uses.

That distinction transfers directly. Once §2 lands, `familyHashgrid` costs
almost nothing additional — the class label only selects a grid plane, and the
`cls` argument is already threaded through `emitEncode` when
`encodingPlanes > 1` (`train_wgsl.ts:435-437`). One-hot remains genuinely
unsupported on both paths and should keep a typed refusal rather than silently
widening.

So the pixel gate should become the adversary's gate: refuse `onehot`, accept
`grid-plane`, and drop the blanket `classes > 0` check.

---

## 4. Refusal 3: single-head fields — the larger gap (**experimental**)

Eight of thirteen archs are refused by **both** critic families for the same
reason: `spec.kind` must be `helmholtz` or `agree-disagree`
(`pixel_disc_wgsl.ts:107`, `adversary_wgsl.ts:576`). `heads: 1` produces
`FieldSpec.kind: "vector"` (`advect.ts:135-147`), and no critic emits against
it.

This is the biggest single expansion available — it would roughly triple the
usable arch surface for both families at once — but it is also the one that
touches the most emitter code, because the blend chain rule (`dSig*(1-alpha)`
into head 0, `dSig*alpha` into head 1) collapses to a single call and every
`fieldBlocks` loop is written around two heads.

Treated as experimental rather than pending: it is not required for the pixel
family to reach parity with the relational adversary, and it should be scoped
as its own plan against `train_wgsl.ts:780-782` and `:1329-1334`, which already
accept `vector` on the field's own trainer. The critic emitters are the only
holdouts.

---

## 5. Divergence worth closing while in here (**implemented** in 19bf275)

The pixel `fieldGrad` writes `extGrads[t] = g` raw (`pixel_disc_wgsl.ts:1226`).
The adversary writes `extGrads[t] = select(0.0, g, isFiniteF(g))`
(`adversary_wgsl.ts:2315`). Same seam, same consumer
(`train_wgsl.ts:1520-1521` sums both into the field's Adam), one guard.

A non-finite pixel `extGrad` therefore reaches the field optimizer directly.
Given the pixel critic is the sole gradient by construction
(`ZERO_FIELD_LOSS`, `docs/PIXEL_DISC.md`), a single NaN poisons the whole
field with nothing upstream to dilute it. Port the guard.

Related constraint, not a bug: `extGradCount` is capped at 2
(`train_wgsl.ts:1337-1340`, `train.ts:218-222`), which is why a piece declaring
both Agree+Disagree and a pixel critic throws at `main.ts:960-968`.

---

## 6. Sequencing

1. **Port the NaN guard** (§5). One line, no dependencies.
2. **Hashgrid support** (§2): `dEncStore` in the scratch layout, the third
   `bwdCall` arm, the grid-segment scatter in `fieldGrad`, and the lane-keyed
   `dEnc` mode from §2a. Unlocks `dualHashgrid`. Gate: the CPU↔GPU equivalence
   test from `docs/PLAN_MULTI_GUESS_MODULARIZATION.md` §4 extended to a
   hashgrid field, plus `tools/pixel_disc_cost_probe.ts`.
3. **Class gate parity** (§3): swap the blanket `classes > 0` refusal for the
   adversary's `onehot`-only refusal. Unlocks `familyHashgrid`. Small once (2)
   lands.
4. **Single-head fields** (§4) — separate plan, both families at once.

Steps 1-3 take the pixel critic from 3 of 13 architectures to 5 of 13, which is
exact parity with the relational adversary. Step 4 is where the remaining 8
live, and it is not pixel-specific.

---

## 7. Rejected

- **Deleting the `classifyPixelDiscFusion` guard to "see what happens".**
  Defect (c) in §2 is silent: the field trains, gradients are finite, every
  existing test stays green, and the encoding simply never learns. The guard is
  the only thing currently reporting the gap.
- **A generic per-arch capability table covering every call site.** There are
  ~30 gates across constructor, layout, trainer, adversary, and pixel paths,
  and most encode genuinely different questions (can this be unrolled? does
  this need a pre-act checkpoint? does this widen layer 0?). Collapsing them
  into one table would replace precise local refusals with a coarse one.
  `classifyPixelDiscFusion`'s pattern — data-not-throw, at the family
  boundary — is the right granularity, and the relational adversary should
  adopt it rather than the reverse.
