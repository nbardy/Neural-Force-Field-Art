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

### 3c. Cost budget (**verified** 2026-08-22 — supersedes the original estimate)

The original version of this section modelled the critic as
`@compute @workgroup_size(1)` and predicted 5.8 ms at one guess, with 8
guesses failing the 12 ms gate at ~13.5 ms. **That model was already obsolete
when written**: commit `fe7834d` ("Parallelize the pixel critic") had made
`criticDisc`/`criticGen` `@workgroup_size(${PIXEL_DISC_WG})` = 256, one cell
per lane, with the workspace moved out of private memory into `scratch`.

Measured on this machine (median of 21 steps, 80k particles, b=256, two
agreeing runs, `tools/pixel_disc_cost_probe.ts`, `BUDGET_MS = 12`):

| config | kind | g=1 | g=2 | g=4 | g=8 |
|---|---|---|---|---|---|
| gallery `G=16 E=8 K=16 h=32` | vec-field | 1.8 | 1.8 | 1.8 | 1.8-1.9 |
| | next-frame | 1.2-1.3 | 1.2-1.3 | 1.3 | 1.3-1.4 |
| | real-fake | 1.4-1.5 | n/a | n/a | n/a |
| | inpaint | 1.2-1.3 | 1.3 | 1.3-1.4 | 1.3-1.4 |
| stress `G=32 E=8 K=32 h=32` | vec-field | 2.7 | 2.7 | 2.7 | 3.2 |
| | inpaint | 2.3 | 2.4 | 2.4 | 2.7 |

**Cost is no longer the binding constraint.** The step is dominated by fixed
per-dispatch overhead; the head fold is one lane-iteration at `G^2 = 256`.
Guesses of 2, 4 and 8 all fit with roughly 6x headroom. The probe is noisy on
this machine — one stress-config cell reported k=4 *below* k=1, which is noise,
not speedup. All figures are medians.

Consequence for §5 of this plan: whichever default the UI eventually selects
should be argued from the §3f collapse telemetry, not from milliseconds.

### 3d. Store the winner index, not the residuals (**verified**, premise corrected)

The conclusion stands; the reason given originally does not. This section
argued from private-memory pressure in a `@workgroup_size(1)` kernel. Since
`fe7834d` the critics are cell-parallel and the workspace lives in `scratch`,
so the winner never outlives its own cell iteration and the math needs no
`array<u32, nCell>` at all.

What was implemented: the min is folded **online** during the forward, and a
per-cell winner slot is written purely so `tools/pixel_disc_equiv_test.ts` can
compare the CPU and GPU *selection* rather than only the summed gradient. That
distinction matters — a winner mismatch can partially cancel inside the sum and
pass a gradient-only comparison. Measured: 0 disagreeing cells across every
kind at k=2 and k=4.

The backward gates each `gW[...] +=` and `dSoft[...] +=` on the recorded
winner, reproducing the stop-gradient-through-selection the other three
backends guarantee.

Win counters likewise need no atomics — but because each lane keeps a private
`array<f32, guesses>` reduced through the existing `wgSum`, **not** because the
kernel is single-threaded.

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

### 3f. Collapse must be detectable, not silent (**pending**)

This was missing from the first draft of this plan and is the most important
omission, because it is the exact failure relaxed-WTA exists to prevent.

`adversary.ts:76-84` states the hazard: a guess that never wins receives zero
gradient, never moves, and keeps never winning. `K` silently degrades to 1
while the loss looks fine and the model has stopped being a mixture. Relaxed
epsilon makes that *less likely*; it does not make it *observable*. The
relational adversary therefore records per-head win counts every `trainStep`
and exposes `winStats()` (`:2104`), `collapsed()` (`:2117`), and
`headSpread()` (`:2139`).

Adding guesses to the pixel critics without the same telemetry ships a knob
whose failure mode is invisible.

The fused adversary already has the portable shape:

```wgsl
var<workgroup> winCnt : array<atomic<u32>, ${k}>;   // adversary_wgsl.ts:1763
  if (tid < ${k}u) { atomicStore(&winCnt[tid], 0u); }        // :1769
      atomicAdd(&winCnt[win], 1u);                           // :1947
    stats[pb + 5u + j] = f32(atomicLoad(&winCnt[j]));        // :2012
```

The pixel port is **simpler**: `criticDisc` is `@compute @workgroup_size(1)`,
so a plain `var winCnt : array<u32, guesses>` in private scope suffices — no
atomics, no workgroup reduction.

**This work item is coupled to the stats-buffer defect.** `PixelDiscStats`
reads six floats (`pixel_disc_train.ts:391-398`) but the shader only ever
writes `metaStats + 0..3`; slots 4 and 5 (`meanFx`/`meanFy`) are written by no
kernel. Whoever fixes that should size the stats region to carry `guesses` win
counters, rather than fixing it to six slots and re-widening it later.

### 3g. Inactive cells must not be counted as wins (**pending**)

`adversary.ts:737-741` names this invariant explicitly:

> The exact per-row activity predicate used by both payoff masking and win
> accounting. Keeping one handler prevents an inactive row from having zero
> payoff/gradient yet still being reported as an argMin tie won by head 0.

and `trainStep` excludes inactive rows from the histogram (`:2035-2037`).

Every pixel kind has an activity predicate already, and they differ:

| kind | predicate | site |
|---|---|---|
| `vec-field` | `D[c] >= 1e-3` (`densFloor`), gen path uses `D2[c]` | `pixel_disc.ts:487-489`, `:539` |
| `inpaint` | `mask[c] == 1`, normalized by `nMask` | `pixel_disc.ts:811-825` |
| `next-frame` | all cells, normalized by `nCell` | `pixel_disc.ts:599-613` |
| `real-fake` | no per-cell term (GAP then MLP) | `pixel_disc.ts:680-725` |

An inactive cell still has a mathematical argmin over guesses, and it routes to
guess 0 under the first-argmin tie rule. Counting those wins would make guess 0
look dominant on a mostly-empty density grid — the collapse detector from §3f
would report collapse that is not happening, or mask collapse that is.

So §3f's counter must be gated by the same predicate that already gates the
residual, per kind. `real-fake` has no per-cell residual and therefore takes
guesses at the MLP output, not per cell — worth deciding deliberately rather
than by default (see §6).

### 3h. Per-guess init symmetry is currently implicit (**pending**)

`adversary.ts:1533-1537` makes symmetry-breaking explicit and documents it as
load-bearing: identical heads produce identical residuals, every tie routes to
head 0, and the mixture is dead on arrival. Its seeds are derived per head:
`seedOf = cfg.seed * 7919 + j * 101 + layer`.

`initPixelDiscWeights` (`pixel_disc.ts:119-158`) draws from a single sequential
`mulberry32` stream, so replicating the head `guesses` times through the
existing `fillHead` calls yields distinct draws **for free** — the stream
advances.

That is a correct outcome reached implicitly, which is the fragile kind. Record
it in a comment at the init site: the per-guess loop must call `fillHead` once
per guess, and must never be "optimized" into initializing one head and
broadcasting it. Add an assertion in the §4 equivalence test that two guesses'
head slices differ.

### 3i. Adam is already per-parameter — no change needed (**verified**)

Noting the non-issue so it is not re-litigated. `adversary.ts:2015-2027`
warns that "shared Adam moments would couple the heads and undo the
quantization," which reads like a constraint on the pixel port. It is not:
`discAdam` (`pixel_disc_wgsl.ts:1369-1382`) is one thread per weight float with
its own `critMeta[metaM + t]` / `critMeta[metaV + t]` slots, so moments are
elementwise. Replicated guess heads occupy disjoint float ranges and are
already decorrelated in exactly the way the adversary's per-head optimizers
achieve.

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
4b. **Win counters + activity gating** (§3f-3g), landed with step 4, not after.
   A guesses knob without collapse telemetry is a knob whose failure mode is
   invisible. Sized into the stats-region fix, not bolted on later.
5. **UI toggle**, reusing the existing "guesses K" control shape
   (`src/index.tsx:2133`).

Steps 1 and 2 are worth landing regardless of whether steps 3-5 follow.

The stats-buffer defect (slots 4/5 unwritten, `pixel_disc_train.ts:391-398`) is
tracked separately but must be scheduled **before or with** step 4b, since 4b
needs that region resized for `guesses` counters.
