# Multi-guess modularization: one WTA spec, four backends

Audited 2026-08-19 against `src/core/gan/adversary.ts`, `src/core/gan/pixel_disc.ts`,
`src/render/webgpu/ad/losses.ts`, `src/render/webgpu/adversary_wgsl.ts`,
`src/render/webgpu/pixel_disc_wgsl.ts`.

Status words are strict (same ledger as `docs/PLAN_RELATIONAL_ADVERSARY.md`):

- **implemented**: code exists;
- **verified**: the named test was run and passed;
- **pending**: required for release but not yet verified;
- **experimental**: worthwhile hypothesis, not a completion requirement;
- **rejected**: investigated and deliberately not used.

Goal: give the four pixel GAN kinds the same K-guesses / relaxed-WTA treatment
the relational adversary already has, and stop three backends from
independently deriving the same two scalars — without moving the fused critic
kernels off their frame budget.

---

## 1. The finding that shapes this plan

The relaxed-WTA weighting is **implemented three times, and that is correct.**

| backend | site | representation |
|---|---|---|
| tfjs reference | `weightsWta`, `src/core/gan/adversary.ts:1382` | `Tensor2D [B,k] -> [B,k]`, `oneHot(argMin)` |
| scalar AD IR | `relaxedWtaWeights`, `src/render/webgpu/ad/losses.ts:195` | `Node[]`, winner as a product of `gt` nodes |
| fused WGSL | `adversary_wgsl.ts:1144-1145`, `:1866-1877` | runtime loop, compile-time `k`, `select(loserW, winW, j == win)` |

These are three genuinely different representations (tensors, IR nodes, shader
strings). They are already cross-verified — `tools/ad_wta_test.ts` §2 gates
tfjs ≡ IR (loss rel < 1e-5, grad cos > 0.99999, per-tuple winner agreement) and
`tools/train_wta_test.ts` §2 gates kernel ≡ IR on real Metal. **verified**

Merging the *reductions* is therefore not the work, and attempting it is what
would cost kernel time. What is actually duplicated is the **spec**: the
config type, the bound on epsilon, the two weight scalars, and the tie rule.
That is small, pure, and belongs in exactly one module.

```text
loser  = epsilon / (K-1)
winner = 1 - epsilon
w_j    = loser + (j == first_argmin ? winner - loser : 0)
payoff = sum_j w_j d_j
```

with `d w / d resid == 0` (selection is off-tape) and ties routing to the
lowest head index.

---

## 2. Layer 0 — `src/core/gan/wta.ts`, a spec module (**pending**)

New file, ~50 lines, zero dependencies. Canonical types + thin dispatcher, one
handler per variant.

```ts
export type GuessKind =
  | { readonly tag: "single" }
  | { readonly tag: "wta"; readonly k: number; readonly relaxEps: number };

export interface WtaScalars { readonly winner: number; readonly loser: number }

export function wtaScalars(kind: GuessKind): WtaScalars;
export function headCount(kind: GuessKind): number;
export function validateGuessKind(kind: GuessKind): void;   // typed error
```

`AdversaryKind` (`adversary.ts:107`) is renamed to `GuessKind` and moved here.
Once the pixel critics consume it, "Adversary" is the wrong noun. Re-export
from `adversary.ts` for one release if callers need the old spelling.

All four backends import `wtaScalars` instead of recomputing `relaxEps/(k-1)`.

Three things this buys, in order of importance:

1. **The `k === 1` trap stops being validation-dependent.** `weightsWta`
   divides by `k - 1` with no guard (`adversary.ts:1384`); it is safe today
   only because `validate` rejects `k < 2` first (`adversary.ts:1483`).
   Extracted standalone it yields `loserShare = Infinity`. Putting the
   arithmetic behind the `GuessKind` dispatch makes the guard structural.
2. The bound `0 <= epsilon < (K-1)/K` lives in one place instead of two
   (`adversary.ts:1489`, `ad/losses.ts:257-278`).
3. The first-argmin tie rule becomes a written invariant rather than a
   convention three files independently honor.

Kernel cost: zero. The WGSL still emits its own unrolled reduction; only the
two interpolated float literals change provenance.

### 2a. Bundle: consolidate the f32 literal emitters (**pending**)

Three implementations of "WGSL needs a decimal point":

| fn | site | notes |
|---|---|---|
| `f32lit` | `src/render/webgpu/ad/emit_wgsl.ts:28` | carries the exponent-formatting fix its own docstring documents as having silently corrupted constants |
| `flit` | `src/render/webgpu/adversary_wgsl.ts:332` | `Number.isInteger(v) ? v.toFixed(1) : String(v)` |
| `fl` | `src/render/webgpu/pixel_disc_wgsl.ts:72` | `if (!/[.eE]/.test(s)) s += ".0"` |

`flit` is precisely what formats `loserW`/`winW` today, and it lacks the fix.
Since Layer 0 makes those scalars shared, point all three at `f32lit` in the
same change. `flit` already exists because `epsilon = 0` emitted as abstract-int
and failed to compile (caught by `tools/train_wta_test.ts` §2) — that class of
bug is exactly what `f32lit` was hardened against.

---

## 3. Layer 1 — pixel disc multi-guess head (**pending**)

### 3a. Guesses are a head stride, not a fifth kind

```ts
headFloats(d) = headCount(d.guesses) * headFloatsPerGuess(d)
```

The four `step*` handlers (`pixel_disc.ts:460, 576, 656, 786`) stay four
handlers. Each gains an inner fold over guesses — a loop inside a fixed kind,
i.e. algorithmic branching, not structural. `pixelWeightLayout`
(`pixel_disc_wgsl.ts:199`) gains a `headStride`; the contiguous
`convW | convB | code | head` packing is otherwise unchanged.

**The trunk stays shared across guesses.** Conv and codebook are computed once
per cell into `cFeat`/`cSoft` and every guess reads the same features. This is
what makes the change affordable, and it matches the relational adversary's
shared-`u` head arrangement (`ad/losses.ts:219-220`: "all k heads share `dims`,
never the leaves").

### 3b. Name the field `guesses`, never `K`

`PixelDiscDims.K` is the **soft-codebook size** (`pixel_disc.ts:26`, consumed
as a per-cell softmax over K logits at `:286-303`). A second `K` in the same
struct is the single most likely way this port goes wrong, because every index
in `pixel_disc_wgsl.ts` is a baked literal interpolated from those dims.

### 3c. Cost budget

Per cell at the shipped `G=8 E=4 K=8` (`src/main.ts:2403` et al.), the trunk is
roughly `9E + K*E = 36 + 32 = 68` MACs and the vec-field head is `2K = 16`. The
head is ~19% of per-cell work, and guesses multiply only that fraction.

| guesses | per-cell | ratio | vec-field est. | vs `BUDGET_MS = 12` |
|---|---|---|---|---|
| 1 (today) | 84 | 1.00x | 5.8 ms **measured** | passes |
| 2 | 100 | 1.19x | ~7 ms | passes |
| 4 | 132 | 1.57x | ~9 ms | passes, little room |
| 8 | 196 | 2.33x | ~13.5 ms | **fails** |

Estimates from MAC counts, not measurements — `tools/pixel_disc_cost_probe.ts`
(`BUDGET_MS = 12`, `:84`) is the arbiter and must run before a default is
chosen. The shape is the load-bearing part: 2-4 guesses fit, 8 does not. The
relational adversary's shipped `guesses K 8` is **not** transferable here.

### 3d. Store the winner index, not the residuals

The `@compute @workgroup_size(1)` critics already hold `cFeat[E*nCell]`,
`cSoft[K*nCell]`, and `dSoft[K*nCell]` as function-scope private arrays
(`pixel_disc_wgsl.ts:1114-1115`, `:495`). A naive `resid[guesses][nCell]` adds
another `guesses * nCell`.

Do not do that. Fold the min online in the forward and retain only:

```wgsl
var winIdx : array<u32, ${nCell}>;   // 256 entries at G=8
```

O(1) per cell instead of O(guesses). The backward then gates each
`gW[...] +=` and `dSoft[...] +=` on `j == winIdx[c]`, which reproduces the
stop-gradient-through-selection the other three backends already guarantee.

This is the difference between fitting and not fitting the private-memory
budget, which the cost probe reports per invocation alongside milliseconds.

### 3e. Sites that must change together

Head shape is declared in three places and mirrored by hand in two more:

| what | site |
|---|---|
| head float count | `headFloats`, `pixel_disc.ts:57` |
| GPU offsets | `pixelWeightLayout`, `pixel_disc_wgsl.ts:199` |
| init | `initPixelDiscWeights`, `pixel_disc.ts:119` |
| CPU head fwd/bwd | `pixel_disc.ts:488-514, 599-611, 700-721, 811-823` + gen mirrors `:538-562, 639-646, 847-860` |
| WGSL head fwd/bwd | `pixel_disc_wgsl.ts:501-529, 558-572, 642-657` + gen mirrors `:689-712, 732-741, 786-801` |

Also: the vec-field weight-normalization sweep hardcodes the head as exactly
the trailing `2K+2` floats (`pixel_disc.ts:517`; WGSL `:532-534`). It must
become `headCount * (2K+2)`.

---

## 4. The test gap that gates everything (**pending**, do first)

`tools/pixel_disc_test.ts` §2 (`:114-187`) asserts only that `discLoss` fell
below `0.95x` its start over 80-150 CPU steps, and §3 (`:190-311`) is a GPU
smoke test that checks `readExtGrads()` is finite and non-zero after **one**
`encodeStep`.

**Nothing compares CPU `discGradPacked` against GPU `critMeta[0..nW)`, and
nothing compares `discLoss`/`genLoss` values across the two implementations.**
The correspondence is maintained purely by hand-mirrored code plus structural
tag re-assertions (`pixel_disc_wgsl.ts:490, 548, 580, 622`).

Section 3e adds a min-over-K reduction to ten hand-mirrored sites. Writing the
CPU↔GPU numerical equivalence test **before** that change, not after, is the
difference between a caught mismatch and a silent one. This step is worth
doing even if guesses never ship.

Note the two implementations use different softeners — `PIXEL_DISC_SOFT_EPS =
1e-4` (`pixel_disc.ts:15`) vs the adversary's `SOFT_EPS = 1e-6`
(`adversary.ts:500`) vs `ADV_SOFT_EPS2 = 1e-12` (`adversary_wgsl.ts:103`). The
pixel pair is internally consistent; the equivalence test should assert that
rather than assume it.

Also note the real-fake fake cloud is generated differently on each side (CPU:
one sequential `mulberry32(12345)` stream, `pixel_disc.ts:672-677`; GPU:
per-particle independent seeds, `pixel_disc_wgsl.ts:1084-1086`). Bit-equality
was never the goal there, so the equivalence test must inject a shared fake
cloud via the existing `fakePos` opt (`pixel_disc.ts:441`) rather than compare
the default paths.

---

## 5. Rejected

- **Merging the three WTA reductions into one implementation.** They target
  three representations with no common substrate. The shared surface is the
  spec (§2), and it is already the whole overlap.
- **Routing the pixel critic through the scalar AD IR.** The IR is scalars-only
  and fully unrolled by design (`ad/ir.ts:11-13`); `nCell * guesses` unrolled
  would mint an enormous DAG. The IR is sized for per-sample math, not
  per-grid.

## 6. Experimental

- **IR-emitted per-cell body.** Emit only one cell's head + residual + min-fold
  through the IR and wrap it in a hand-written WGSL cell loop, using the
  documented `input`-name seam (`ad/emit_wgsl.ts:8-10`: "that is the seam where
  this generated math body plugs into the hand-written harness"). The backward
  would fall out of `grad()` instead of being hand-mirrored across the ten
  sites in §3e — which is the actual risk area. Distinct from the rejected
  item in §5: the DAG is per-cell, not per-grid. Should not ride along with
  this plan.

---

## 7. Sequencing

Each step is independently shippable and independently verifiable.

1. **Layer 0 + literal consolidation** (§2, §2a). No behavior change.
   Gate: `tools/ad_wta_test.ts`, `tools/train_wta_test.ts` still pass.
2. **CPU↔GPU equivalence test for the pixel critic** (§4). Gate: it passes on
   today's single-guess code.
3. **Multi-guess head, CPU oracle only** (§3a-3b), default
   `guesses: { tag: "single" }`. Gate: step 2's test still passes and output is
   byte-identical to today.
4. **WGSL mirror with `winIdx` gating** (§3d-3e). Gate: step 2's test at
   `guesses.k > 1`, then `tools/pixel_disc_cost_probe.ts` picks the ceiling.
5. **UI toggle**, reusing the existing "guesses K" control shape
   (`src/index.tsx:2133`).

Steps 1 and 2 are worth landing regardless of whether steps 3-5 follow.
