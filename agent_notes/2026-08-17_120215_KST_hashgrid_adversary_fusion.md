# Fusing the hashgrid adversary — the last unfused training path

Date: 2026-08-17 (KST)
Predecessor: `agent_notes/2026-08-17_120215_KST_fused_kernel_audit.md` (read-only audit)
Scope: implementation + verification. Working tree only — **nothing committed**.

Legend: **[V]** verified by running it on this machine. **[H]** hypothesis.

---

## 0. Goal and result

**Goal.** The DEFAULT gallery piece `Adversary · Pair · HashGrid · Curl`
(`src/main.ts:2107`, `DEFAULT_PIECE_NAME`) was the one of 18 gallery pieces
whose D, generator reward AND field optimizer all ran on tfjs autograd, because
the fused adversary refused hashgrid encodings. Make it train fully fused.

**Result [V].** Same machine (Apple M4 / Metal), headless Chrome, 70k
particles, default piece, measured by driving the live HUD on two parcel builds
that differ ONLY in the `fusedAdvOk` hashgrid exclusion:

| | before | after |
|---|---|---|
| HUD | `FPS 24.6 (40.7 ms)`<br>`learn 39.6 ms (cpu·tfjs)` | `FPS 60.0 (16.7 ms)`<br>`rollout 0.01 ms  optim 0.98 ms  loss 0.000` (gpu) |
| tfjs tensors | 123 | 15 |
| `tools/soak_adversary.mjs hashgrid` | aborts — "RAW diagnostic did not become active" (that diagnostic is fused-only) | **"every sample reports the fused adversary" PASSES**, 20/20 other gates pass |

Offline bun bench of the FULL learn step (adversary 5 passes incl. disc Adam +
fieldGrad, then the field trainer's rollout+optim with extGrad), same game
(pair-rot-scale-adjusted, soft-angle, k=4), Apple M4 Metal [V]:

```
  raw dualStd (fused, reference)     B=256  0.911 ms/learn-step  (2440 field floats)
  hashgrid 32²×4 (fused, NEW)        B=256  0.866 ms/learn-step  (6664 field floats)
  raw dualStd (fused, reference)     B=512  0.994 ms/learn-step  (2440 field floats)
  hashgrid 32²×4 (fused, NEW)        B=512  1.090 ms/learn-step  (6664 field floats)
```

The 4096-float grid segment costs essentially nothing next to the raw MLP
reference — the audit's [H] estimate of "~1.5-3 ms, 30-60× better" was
pessimistic on the absolute number and right on the direction. No follow-up
fixed-point atomic scatter is warranted; the deterministic gather form is fast
enough and is what the cos = 1.0 gates were written against.

---

## 1. The audit's gap analysis — verified against the code

Every claim in §4.1 of the audit checked out, at the cited lines (offsets in my
worktree differ by ~40-55 lines from the audit's numbers; the code is the same).

- **The refusal** was one throw in `validateAdversaryFusion`
  (`adversary_wgsl.ts`, was :419-424) plus `fusedAdvOk`'s
  `advect.layout.encoding.kind !== "hashgrid"` (`main.ts`, was :3350). Because
  `main.ts:3401-3406` gates the FusedTrainer on `(advRt.tag === "off" ||
  fusedAdvOk)`, refusing the adversary also refused the FIELD trainer. [V]
- **The hashgrid backward is gather-side, no atomics.** `train_wgsl.ts:1183-1216`
  — thread = one grid float, scans (sample, site) and claims the bilinear
  corners landing on its cell. One writer per grid float by construction. [V]
- **`adversary_wgsl.ts` already imported and emitted** `emitEncode` /
  `emitFwdStore` / `emitBwdStore` (:1370-1373 pre-edit); the forward interp and
  the per-head `dL/dEnc` + `du` jacobian were already generated correctly. [V]
- **Gap A** — `advScratchLayout` used `fieldSl.encStore` but never
  `fieldSl.dEncStore`, so there was no `dEncOff`. [V]
- **Gap B** — `bwdCall` knew only the raw (2-arg) and fourier (3-arg) shapes;
  hashgrid's is `bwd_head_h(dOut, base, uIn: vec2f, dEncBase: u32)`. [V]
- **Gap C** — `adversaryPassBShader`'s field-segment loop opened with
  `const h = seg.head`, and a grid segment carries `head: -1, layer: -1`
  (`advect_wgsl.ts:366`), so it would have indexed `heads[-1]`. [V]
- **The seeding hazard** — `emitBwdStore` chose `=` vs `+=` by head index. The
  adversary runs BOTH heads only on `fieldLane === "blend"`; with
  `fieldLane === 1` (Agree+Disagree lane B, `main.ts:3510`) only head 1 runs and
  would have `+=`'d into never-written scratch. [V]

---

## 2. What changed, file by file

### `src/render/webgpu/train_wgsl.ts` (+18 −9)

`emitBwdStore` takes a required 6th parameter `dEncMode: "seed" | "accumulate"`.
The seed/accumulate choice is a property of the SHADER BEING EMITTED, not of the
head index — that is the whole fix for the lane hazard, expressed as a type
rather than an assumption. Its own call site passes `i === 0 ? "seed" :
"accumulate"` (both heads always run there, so behaviour is unchanged).
`raw`/`fourier` ignore the parameter entirely (no dEnc block), so their emitted
text is unchanged.

### `src/render/webgpu/pixel_disc_wgsl.ts` (+1 −1)

Call-site update only (`i === 0 ? "seed" : "accumulate"`). The pixel critic
still refuses hashgrid at `validatePixelDiscFusion` — **deliberately out of
scope**, and now said out loud in the adversary's scope docstring so nobody
reads "hashgrid works now" as covering both.

### `src/render/webgpu/adversary_wgsl.ts` (+77 −14)

1. **Gap A** — `AdvScratchLayout.dEncOff` added;
   `dEncOff = encOff + m·fieldSl.encStore`,
   `fieldSiteOff = dEncOff + m·fieldSl.dEncStore`. `dEncStore` is 0 for
   raw/fourier so the offsets collapse and the stride is unchanged — the same
   discipline `trainScratchLayout` uses.
2. **Gap B** — a `dEncBase(site)` helper and a third arm in `bwdCall`, keyed on
   `enc.kind` (three kinds, three call shapes, no default arm). Passes
   `pn[site]` — the normalized member position, already a live local at every
   backward site.
3. **Seeding** — `dEncSeedHead = fieldLane === "blend" ? 0 : fieldLane`, fed to
   `emitBwdStore`. Chose this over the audit's suggested explicit zero-fill: it
   costs zero WGSL, and it makes the invariant impossible to get wrong from the
   emitter's side rather than patching over it.
4. **Gap C** — a `seg.role === "grid"` block at the TOP of `fieldGrad`'s segment
   loop, transliterated from `train_wgsl.ts:1193-1215` with `ub.n → ub.b`,
   `sl.sites → m`, and the adversary's own `siteInOff`/`dEncOff`/stride. It sits
   **above** the `fieldLane !== "blend" && h !== fieldLane` skip on purpose: the
   grid table is shared by both heads and lane isolation already happened
   upstream (only the active lane's `bwd_head` wrote dEnc). A grid segment on a
   non-hashgrid encoding throws a typed error rather than silently continuing.
5. The `validateAdversaryFusion` hashgrid throw is deleted; the file's SCOPE
   docstring now describes what IS supported and names the remaining gaps
   (classes, pixel critic).

**Numerical policy held.** The new block introduces no `sqrt` and no new
transcendental — it is integer compares and one multiply-add into the existing
`var g`, so it inherits `extGrads[t] = select(0.0, g, isFiniteF(g))` for free
(no separate write path was added). The SELU ±80 clamp, `ADV_SOFT_EPS2` inside
every soft-sphere radicand, the residual/Adam `isFiniteF` gates and the hashgrid
`du` clip mask are untouched. Everything is inline in the fused shader; the hot
path is still one `encodeStep` encoder → one `queue.submit`.

### `src/main.ts` (+15 −14)

`fusedAdvOk` drops the hashgrid exclusion (classes stay excluded). Two comment
blocks updated: the `fusedAdvOk` rationale, and the piece-17 comment that
documented the refusal as permanent. **`main.ts:3232`'s `(cpu·tfjs)` label was
NOT touched** — fixed separately in the main tree, left alone here to avoid a
merge conflict. (It is still the label the "before" build printed above, which
is why the before/after table quotes it verbatim.)

### `tools/train_wta_hashgrid_test.ts` (NEW, ~470 lines)

New file rather than a section in `train_wta_test.ts` because that suite's §2
oracle is the AD IR, and **the IR cannot express a hashgrid** — its gather
indices are data-dependent, and `ad/rollout.ts:52-60` types `encoding` as
`raw | fourier` with that reason in a comment. The only available oracle is a
LIVE tfjs graph, i.e. the §3/§3b pattern. Details in §3 below.

### `tools/quad_nan_probe.ts` (+38 −7)

Added `ENC=raw|hashgrid` (plus `GRID`/`GRID_F`, default 32/4) and extended the
`TAG` union with the three `pair-rotation*` observers so the default piece's
exact game is reachable. **Grid-table init is the trap the audit flagged**: the
existing loop only fills `role === "kernel"` segments, so a grid segment would
have stayed all zeros — a constant field whose backward is trivially finite,
i.e. a FALSE pass. It now seeds U(−0.1, 0.1), matching `HelmholtzField`'s own
init, with a comment saying why.

### `tools/soak_adversary.mjs` (+5)

Added the `hashgrid` piece key pointing at the default piece. Its
*"every sample reports the fused adversary"* gate is now the end-to-end proof of
this work — it cannot pass on a build where `fusedAdvOk` still excludes
hashgrid.

---

## 3. Test results — all gates [V]

### Gate 1 — new hashgrid adversary parity vs LIVE tfjs

`bun tools/train_wta_hashgrid_test.ts` → **ALL HASHGRID ADVERSARY CHECKS PASS**.

Oracle: a real `HelmholtzField({ modelType: "hashgrid" })` (the same
`gridInterp` oneHot·matMul the app's `?train=tfjs` route runs) + the real
`Adversary`, differentiated with `tf.variableGrads`. Field weights are assigned
from the packed buffer in `layout.segments` order, and the test ASSERTS that
`trainableWeights.length === segments.length` and that each shape matches rather
than assuming the ordering.

Six configurations, every one at **cos = 1.0000000**:

| case | u/y | D grad | extGrads (all) | GRID slice | MLP slice |
|---|---|---|---|---|---|
| default piece (pair-rot-scale-adj, soft-angle, 32²×4) | Δu 0.00e+0, Δy 5.4e-7 | rel 3.8e-7 | 4356 floats, rel 2.4e-6 | [0,4096), rel 3.7e-6, 1656 nz | 260 floats, rel 2.4e-6 |
| dense 4²×3 grid (pair, raw-vector) | 8.9e-8 / 1.6e-7 | 1.8e-7 | rel 1.3e-6 | 48/48 nz, rel 5.9e-7 | rel 1.5e-6 |
| coincident corners + clamp border (8²×4) | 1.2e-7 / 2.2e-7 | 2.5e-7 | rel 8.3e-7 | 168 nz, rel 6.6e-7 | rel 2.0e-6 |
| post-velocity target (point, soft-angle, 16²×4) | 1.2e-7 / 1.2e-7 | 2.8e-7 | rel 5.6e-7 | 608 nz, rel 1.1e-6 | rel 5.6e-7 |
| **fieldLane 0** (8²×4) | 1.2e-7 / 2.9e-7 | 1.6e-7 | rel 6.5e-7 | 256 nz, rel 4.8e-7 | rel 6.5e-7 |
| **fieldLane 1** (8²×4) | 8.9e-8 / 3.4e-7 | 1.3e-7 | rel 6.6e-7 | 256 nz, rel 6.3e-7 | rel 6.6e-7 |

Why the slices are asserted separately: the grid is 4096 floats against ~260 MLP
floats on the production config, so a completely wrong grid block would leave
the aggregate cosine misleadingly high. There is also a **support** check — the
set of grid floats with nonzero gradient must match tfjs EXACTLY (0 mismatches
in all six cases). A wrong cell index lights up different cells and fails that
even when the magnitudes happen to correlate.

The coincident-corner case is deliberate and bit-robust: members are placed at
u ≥ 1 and u ≤ 0 on both axes, so `clamp(u,0,1)` is exactly 1.0/0.0, `fx = 0` and
`ix1 == ix` — the SAME cell matches two corner tests and both must add. Plus two
members at an identical position (one cell, two sites) and two members in the
same cell at different points. No reliance on floating-point luck at interior
cell boundaries.

Lane isolation additionally asserts the inactive field head's extGrad floats are
**exact** zero (max |g| == 0, not a tolerance).

Other sections of the new file:
- §1 gates: hashgrid ACCEPTED; hashgrid+classes still throws; sin predictor
  heads still throw.
- §2 codegen invariants: `dEncOff === fieldSiteOff` for raw and fourier (block
  empty, stride unchanged — raw 263, fourier 299 floats) and the generated
  pass A contains no dEnc scratch block; hashgrid's block is m·encDim and its
  pass A does store dL/dEnc.

### Gate 1b — raw/fourier codegen is byte-identical (stronger than the test)

Generated **1056 shader blocks** from the pre-change tree (`git archive HEAD`)
and the post-change tree — train pass A, train pass B (extGrad), all 4 pixel-GAN
kinds, and adversary pass A + pass B across {raw, fourier} × {helmholtz,
agree-disagree} × 7 tuple tags × 3 field lanes × 2 targets × 4 losses — and
diffed. **Byte-identical.** [V] So `kernel_test`'s f32 codegen guard and every
existing parity gate remain meaningful for the untouched encodings.

### Gate 2 — existing suites

| suite | result |
|---|---|
| `bun tools/train_types_test.ts` | ALL PASS (5 fixtures incl. `grad_ref_hashgrid.json`; worst cos 1.0000000, worst relMax 1.33e-5) |
| `bun tools/train_wta_test.ts` | ALL FUSED WTA CHECKS PASS (bench: pair k=4 1.771 ms, tri k=6 1.612, quad k=4 1.315 @ B=512) |
| `bun tools/kernel_test.ts` | ALL PASS |
| `bun tools/train_test.ts` | ALL PASS (fused train step @ batch 256: 0.715 ms) |
| `bun tools/pixel_disc_test.ts` | ALL PASS |
| `bun tools/adversary_wire_test.ts` | ALL PASS |
| `bun tools/field_loss_routing_test.ts` | ALL PASS |

### Gate 3 — NaN/stability, particle-coupled, default-piece config

```
TAG=pair-rotation-scale-adjusted ENC=hashgrid K=4 N=60000 WEIGHT=0.015 \
  STEPS=4000 REPORT=250 bun tools/quad_nan_probe.ts
→ QUAD COUPLED PROBE FINITE
```
4000 steps of advect + particle-sourced D→G→field with a live 32²×4 grid table.
No nonfinite stage at any of the six checkpoints (stats, advGrad, advWeight,
extGrad, fieldGrad, fieldWeight). Field-weight max drifted 0.70 → 0.89, extGrad
max stayed ~1e-3 throughout, win counts stayed spread (two transient
winner-collapse frames at step 1000 and 3500 that recovered — normal WTA
behaviour, also seen on raw pieces). [V]

### Gate 4 — build

`parcel build --no-scope-hoist --no-cache src/index.html src/splat.html
src/splat3d.html` → Built in 5.19s, no errors. [V]

### Gate 5 — live browser

This box's headless Chrome **does** get a real Metal WebGPU adapter with
`--enable-unsafe-webgpu --use-angle=metal` (the flags `soak_adversary.mjs`
already uses). Note `tools/smoke.mjs` CANNOT be used here: it forces
`forceFallbackAdapter: true` and this Chromium has no SwiftShader-WebGPU, so it
reports `adapter: null` and the page correctly shows the WebGPU warning. Use the
soak runner, not smoke, for anything that needs to render on this machine.

`node tools/soak_adversary.mjs hashgrid <url> 60 10` — three of four runs fully
green (all 20 gates). One 40 s run intermittently failed two warm-up-timed
diagnostic-span gates (`RAW and PER UNIT each expose a finite non-flat
percentile span`, `p98 > p2`); those are measured by `exerciseColorDiagnostics`
at `SOAK_WARMUP_MS` (4 s), when the surprise distribution can still be
degenerate. `pair4` (untouched raw piece) passed at the same duration, but this
looks like harness warm-up sensitivity rather than something hashgrid-specific —
**unresolved, see §6.** The gate that matters here, *"every sample reports the
fused adversary"*, passed on every run.

The only console error in the first run was `/favicon.ico` 404 from my ad-hoc
`python3 -m http.server` on `dist/`; adding an empty favicon made it green. Not
an app issue.

---

## 4. The additive-vs-double-count verdict (fieldLoss ≠ 0 + hashgrid + adversary)

**Verdict: correctly ADDITIVE. There is no double count.** [V]

Reasoning. `trainPassBShader` computes, per field weight float `t`:
`g = (internal structural-loss gradient, grid block included) + extGrad0[t]`.
Those are gradients of **two different objectives over the same parameters**:

- the field's own structural loss (chaos/iso/div/spiral/cover/center) over the
  FIELD trainer's batch, assembled from the FIELD trainer's scratch at
  `trainScratchLayout.dEncOff`;
- the generator reward `L_gen` over the ADVERSARY's tuple batch, assembled from
  the ADVERSARY's scratch at `advScratchLayout.dEncOff` and written to
  `extGrads`.

Two disjoint scratch buffers, two disjoint dEnc blocks, two different batches,
two different loss functions. Summing them is exactly ∇(L_field + L_gen), which
is what an additional objective term is supposed to do. A double count would be
the SAME term applied twice.

Empirically pinned, not just argued: §4 of the new test builds BOTH
`FusedTrainer`s **without** an explicit `loss`, so both compile the default
`train_wgsl.LOSS` (W_CHAOS 1, W_ISO 1, W_DIV 0.5, W_SPIRAL 2e-5) and
`hasStructuralLoss` is TRUE — i.e. this is precisely the hashgrid + nonzero
fieldLoss + adversary combination. The test first asserts the field's OWN
structural grid gradient is live (max |g| 2.39e-1, so the check is not vacuous),
then asserts `grads(with extGrad) − grads(without) ≡ extGrads` over all 520
floats at **cos 1.000000, scale-rel 3.84e-8**, and separately over the grid
slice (256 nonzero, cos 1.000000). The field's own grid gradient is present in
both terms and cancels exactly; the residue is exactly the adversary's
contribution. If the field's grid block were re-applying the adversary's term,
that difference could not equal extGrads.

Note the DEFAULT piece itself uses `ZERO_FIELD_LOSS`, so on it
`hasStructuralLoss` is false, the field's internal `g` is literal `0.0` and
100% of the grid update arrives through the extGrad seam. The additive result
above matters for any future hashgrid piece that also sets a structural loss
(e.g. a hashgrid Chaos Weave), which nothing ships today.

---

## 5. Invariants deliberately preserved

- **`classes === 0` stays required.** Class channels are raw-only by
  construction (`layoutField` rejects non-raw + classes), and `train_wgsl`'s
  class-aware layer-0 block has no encoded-input counterpart. Orthogonal gap,
  untouched, and now asserted in §1 of the new test.
- **Disjoint packed segments stay disjoint.** The grid table lives at FIELD
  offset 0, inside the `FusedTrainer`'s buffer. The adversary binds the field
  weights READ-ONLY at pass A binding 1 and only ever writes `extGrads`. Test §4
  asserts 5 adversary steps leave the field buffer — grid table included —
  BIT-IDENTICAL.
- **No new host round-trip, no new dispatch.** The grid block is extra work
  inside the existing `fieldGrad` entry point; `adversary_train.ts` already
  dispatched `ceil(field.totalFloats / ADV_WG_B)` workgroups and needed no
  change. `advScratchBytes` derives from the stride, so the +m·encDim floats are
  picked up automatically.

---

## 6. Unresolved / follow-ups

1. **[V] Soak warm-up flake.** One of four `soak_adversary.mjs hashgrid` runs
   failed the two surprise-percentile-span gates, which are sampled at
   `SOAK_WARMUP_MS = 4000`. Reproduced once out of four; `pair4` passed at the
   same duration but was only run once, so I cannot claim it is
   hashgrid-specific. Cheapest next step: run both keys ×5 at
   `SOAK_WARMUP_MS=8000` and see whether the flake survives. Not a blocker —
   the fused-path gate passed every time.
2. **The PIXEL critic still refuses hashgrid** (`validatePixelDiscFusion`,
   `main.ts:3499`). Out of scope by decision; now stated in the adversary's
   scope docstring so the two are not conflated. Fusing it would need the same
   three pieces in `pixel_disc_wgsl.ts`'s own scratch layout and fieldGrad.
3. **`main.ts:3232`'s `(cpu·tfjs)` label** was left alone per instruction (fixed
   separately in the main tree). With this change the default piece no longer
   takes that branch at all, but `?train=tfjs` still does.
4. **The two pixel-gate bugs the audit filed** (silent extGrad drop when a piece
   sets both Agree+Disagree and `pixelDisc`; silent critic disable under
   `?train=tfjs`) are untouched and still live. Neither is triggered by a
   shipped piece.
5. **Batch-size divergence under `?train=tfjs`** (audit §6) is unchanged; fusing
   the default piece makes it moot for that piece only.
6. **[H] Bigger grids.** Everything here was measured at gridSize ≤ 32. The
   gather formulation is O(gs²·F · B·m) in thread-iterations; at gs = 128 that
   is 16× the threads and the fixed-point atomic scatter
   (`pixel_disc_wgsl.ts:1111-1114` has the precedent) may become the better
   shape. Do that only after re-locking parity — fixed point breaks bit-exact
   agreement with the tfjs oracle.

---

## 7. Files changed (nothing committed)

```
M src/main.ts                          (+15 −14)  gate + two comment blocks
M src/render/webgpu/adversary_wgsl.ts  (+77 −14)  layout, bwdCall, seeding, grid block, scope doc
M src/render/webgpu/pixel_disc_wgsl.ts (+1  −1)   emitBwdStore call site
M src/render/webgpu/train_wgsl.ts      (+18 −9)   emitBwdStore dEncMode parameter
M tools/quad_nan_probe.ts              (+38 −7)   ENC= knob + grid-table init
M tools/soak_adversary.mjs             (+5)       hashgrid piece key
? tools/train_wta_hashgrid_test.ts     (new)      live-tfjs parity suite
```

Reproduce, in order:

```bash
bun tools/train_wta_hashgrid_test.ts     # new parity suite
bun tools/train_types_test.ts            # field trainer, all 4 encodings
bun tools/train_wta_test.ts              # raw/fourier adversary regression
TAG=pair-rotation-scale-adjusted ENC=hashgrid K=4 N=60000 WEIGHT=0.015 \
  STEPS=4000 bun tools/quad_nan_probe.ts
yarn build
node tools/soak_adversary.mjs hashgrid http://localhost:1234/index.html 60 10
```
GPU suites are SEQUENTIAL — run nothing else on the GPU.
