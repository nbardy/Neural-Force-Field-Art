# Fused-kernel audit — which pieces actually run fused, and what it takes to close the last gap

Date: 2026-08-17 12:02 KST
Scope: READ-ONLY audit. No source files were modified by this pass.
Trigger: the belief that "all kernels are fused", contradicted by the DEFAULT
gallery piece (`Adversary · Pair · HashGrid · Curl`) showing
`learn 92.7 ms (cpu·tfjs)` in the HUD.

Legend: **[V]** = verified by reading the code at the cited line.
**[H]** = hypothesis / estimate, explicitly not measured here.

---

## 0. Executive summary

1. **17 of 18 gallery pieces train fully fused at their baked defaults. Exactly
   one does not — and it is the default piece.** [V]
2. The refusal is a single `if` in `validateAdversaryFusion`
   (`src/render/webgpu/adversary_wgsl.ts:419-424`) plus its mirror in the
   startup gate (`src/main.ts:3390`). Because the field trainer's construction
   is gated on `(advRt.tag === "off" || fusedAdvOk)`
   (`src/main.ts:3445`), refusing the *adversary* also refuses the *field*
   trainer. One unsupported encoding knocks out **D, G, and the field
   optimizer** together. [V]
3. **The `(cpu·tfjs)` HUD label is a hardcoded string, not a backend check**
   (`src/main.ts:3232`) — and it is **wrong**. tfjs is genuinely on the `webgpu`
   backend, every kernel on that path has a webgpu implementation, and the HUD
   header two lines above prints `backend webgpu`. The cost is blocking syncs,
   forced queue flushes, and an `O(N·gs²)` one-hot hashgrid encoding — not CPU
   execution. See §3.
4. The hashgrid backward the adversary is missing **already exists, is already
   exported, and is already imported by `adversary_wgsl.ts`**. It is a
   *gather-side* loop with **no atomics** — the original hypothesis ("scatter
   with atomics") is **wrong**, and the truth makes the port easier. See §4.
5. Estimated port: ~5 edits, ~60 lines of codegen, zero new numerical
   algorithms. The only genuinely new WGSL is one grid block in `fieldGrad`,
   transliterated from `train_wgsl.ts:1225-1247`.

---

## 1. Deliverable 1 — every gallery piece → its learn path at baked defaults

`GALLERY` is defined at `src/main.ts:1609-2200`; 18 entries. `DEFAULT_PIECE_NAME
= "Adversary · Pair · HashGrid · Curl"` (`src/main.ts:2207`), resolved by name
to `DEFAULT_PIECE_INDEX` (`:2209-2219`).

### The gates, verbatim

```
main.ts:3344  wantTfjsTrainer   = ?train=tfjs   (forced false for classes>0 :3360, agree-disagree :3367)
main.ts:3384  fusedAdvOk        = advSpec.tag === "on"
              && !adversaryDisabled            (adversaryDisabled = adv on && fieldClasses > 0, :3383)
              && !!field
              && !wantTfjsTrainer
              && advect.layout.classes === 0
              && advect.layout.encoding.kind !== "hashgrid"     <-- THE GATE
main.ts:3441  FusedTrainer built iff field && cfg.fieldLoss !== undefined
              && !wantTfjsTrainer
              && (advRt.tag === "off" || fusedAdvOk)             <-- COUPLING
main.ts:3480  AdversaryTrainer built iff fusedAdvOk && advRt.implementation === "fused"
main.ts:3534  PixelDiscTrainer built iff cfg.pixelDisc && field && !wantTfjsTrainer
              && classes === 0 && encoding.kind !== "hashgrid"
              && spec.kind ∈ {helmholtz, agree-disagree}
```

Arch presets resolve encodings at `src/core/field/arch.ts:51-150`:
`mlpShallow`/`mlp256`/`dualStd`/`dualSiren` → `raw`; `dualFourier` → `fourier`;
`dualHashgrid` → `hashgrid` (`gridSize:32, gridFeatures:4`). [V]

### The table

| # | Piece (`name`) | line | arch → encoding | fieldLoss | adversary | pixelDisc | **Learn path at baked defaults** | Deciding gate |
|---|---|---|---|---|---|---|---|---|
| 0 | Spiral | 1612 | mlpShallow → raw (1 head ⇒ `vector`) | SPIRAL+center | off | – | **Fused field trainer** | 3441 (adv off) |
| 1 | Vortex | 1630 | mlpShallow → raw | CENTER | off | – | **Fused field trainer** | 3441 |
| 2 | Galaxy | 1650 | mlp256 → raw | SPIRAL | off | – | **Fused field trainer** | 3441 |
| 3 | Spiral Cover | 1669 | mlp256 → raw | COVER (Chamfer↑) | off | – | **Fused field trainer** | 3441; `W_COVER` compiled in `train_wgsl.ts:106,617` |
| 4 | Neural Field · Max Chaos | 1692 | dualStd → raw | MAX_CHAOS | off | – | **Fused field trainer** | 3441 |
| 5 | Neural Field · Species | 1715 | dualStd → raw, `classes:3` | MAX_CHAOS | off | – | **Fused field trainer** (fused-only; `?train=tfjs` refused at 3360) | 3441 |
| 6 | Adversary · Single (control) | 1748 | dualStd → raw | ZERO | on, point, k=1 | – | **Fused adversary + fused field** | 3384 ✓ → 3480, 3441 |
| 7 | Adversary · WTA K=8 | 1782 | dualStd → raw | ZERO | on, point, k=8 | – | **Fused adversary + fused field** | 3384 ✓ |
| 8 | Adversary · Pair WTA K=4 | 1824 | dualStd → raw | ZERO | on, pair-rot-scale-adj, soft-angle, k=4 | – | **Fused adversary + fused field** | 3384 ✓ |
| 9 | Adversary · Tri WTA K=6 | 1874 | dualStd → raw | ZERO | on, tri, k=6 | – | **Fused adversary + fused field** | 3384 ✓ |
| 10 | Adversary · Quad WTA K=6 | 1902 | dualStd → raw | ZERO | on, quad-labelled, k=6 | – | **Fused adversary + fused field** | 3384 ✓ |
| 11 | Pixel · VecField | 1947 | dualFourier → fourier | ZERO | off | vec-field, w=.04 | **Fused pixel-D + fused field** | 3534 ✓ |
| 12 | Pixel · NextFrame | 1974 | dualFourier → fourier | ZERO | off | next-frame, w=.04 | **Fused pixel-D + fused field** | 3534 ✓ |
| 13 | Pixel · RealFake | 2001 | dualFourier → fourier | ZERO | off | real-fake, w=.03 | **Fused pixel-D + fused field** | 3534 ✓ |
| 14 | Pixel · Inpaint | 2028 | dualFourier → fourier | ZERO | off | inpaint, w=.04 | **Fused pixel-D + fused field** | 3534 ✓ |
| 15 | Adversary · Agree + Disagree RGB | 2059 | dualFourier → fourier, `semantic:"agree-disagree"` | ZERO | on, game=agree-disagree, pair, k=4 | – | **Two fused AdversaryTrainers (A lane0/disagree, B lane1/agree) + one fused field step** | 3384 ✓; A/B at 3497/3508 |
| 16 | Adversary · Chaos Weave | 2105 | dualFourier → fourier | MAX_CHAOS | on, pair-rotation, k=6 | – | **Fused adversary + fused field** (structural loss AND extGrad both summed at `train_wgsl.ts:1343`) | 3384 ✓ |
| **17** | **Adversary · Pair · HashGrid · Curl** ← **DEFAULT** | **2167** | **dualHashgrid → hashgrid** | ZERO | on, pair-rot-scale-adj, soft-angle, k=4 | – | **tfjs autograd for D, G, AND the field** — no `FusedTrainer`, no `AdversaryTrainer` | **3390** (`encoding.kind !== "hashgrid"` fails) → `fusedAdvOk=false` → **3445** also fails |

Notes on the table:

- **Every one of the 18 pieces sets `fieldLoss`.** [V] So the documented
  "aesthetic pieces that omit `fieldLoss` keep the tfjs `computeLoss` path"
  (`main.ts:3438-3440`) describes an **empty set** today. `computeLoss` is
  reachable only via piece 17 or `?train=tfjs`.
- Pieces 0–4 set `archEditable: true`, so a user *can* select HashGrid from the
  dock and drop those pieces onto the tfjs path too. That is a runtime
  override, not a baked default. [V] (`ARCH_DOCK_PRESETS`,
  `arch.ts:155-171`, includes `dualHashgrid` at `:171`.)
- Piece 17's own source comment (`main.ts:2151-2157`) already documents the
  cost honestly and names the fix: *"Fusing hashgrid backward for the adversary
  is the one change that would buy it back."* [V]
- **Pixel GAN pieces are fully fused on BOTH sides.** D train
  (`clearDens → sampleAndSplat → densToFloat → criticDisc → discAdam`,
  `pixel_disc_train.ts:317-338`) and the generator reward
  (`clearDensGen → virtualSplat → criticGen → densityVjpAndFieldBwd →
  fieldGrad`, `:340-346`) are all WGSL recorded into the frame encoder. No tfjs
  anywhere on that path. `fieldGrad` writes `extGrads[t]`
  (`pixel_disc_wgsl.ts:1168-1172`), consumed by `train_wgsl.ts:1343`. [V]
  The critic's input is **not** the framebuffer — it re-splats its own G×G soft
  density from `advect.posBuffer` (`pixel_disc_wgsl.ts:977-984`, atomicAdd at
  fixed point `:1111-1114`). Zero `texture` references in the file. [V]

### Two latent bugs found in the pixel gate (out of scope to fix, worth filing)

- **Silent extGrad drop.** `main.ts:3568-3571`:
  ```ts
  if (advTrainerB) extGradBuffers.push(advTrainerB.extGradsBuf);
  else if (pixelDiscTrainer) extGradBuffers.push(pixelDiscTrainer.extGradsBuf);
  ```
  The `else` binds to `advTrainerB`. An Agree+Disagree piece that also sets
  `pixelDisc` constructs the trainer, logs `[pixel-disc] FUSED`, dispatches all
  10 passes every frame — and never binds its `extGradsBuf`, so the critic has
  literally zero effect. The guard at `:3572-3577` meant to catch this is
  **unreachable**: `extGradBuffers.length` can never exceed 2 given the branch
  structure. [V] Not triggered by any shipped piece (no piece sets both).
- **Silent disable under `?train=tfjs`.** `main.ts:3537` requires
  `!wantTfjsTrainer`, so `?train=tfjs` on a Pixel piece drops the critic with
  no warning and no log — the piece becomes a pure `ZERO_FIELD_LOSS` field
  (i.e. nothing drives it). The two neighbouring overrides at `:3360-3374`
  both `console.warn`; this one is silent. [V]
- Dead readback: `PixelDiscTrainer.lastStats` (`pixel_disc_train.ts:380`) is
  assigned every training frame from a `mapAsync` and **never read anywhere in
  `src/`**. Cheap and non-blocking, but it is pure waste. [V]

---

## 2. Deliverable 2 — every other compute path on the hot loop

Frame structure: **one `GPUCommandEncoder` per frame, one `queue.submit`**
(`main.ts:2784`, `:3193`). The tfjs learn path cannot join it —
`optimizer.minimize` does its own internal submits (`main.ts:2782-2783`). [V]

| Path | Where | Fused? | Per-frame GPU→CPU readback? |
|---|---|---|---|
| Advect (all pieces) | `advect.encodeStep`, `main.ts:3113` → `advect.ts:342-404` | **Fused WGSL**, 1 dispatch over all particles | No — *except* on the tfjs path, see below |
| Weight sync tfjs→GPU | `advect.ts:365-392` | n/a | **Yes, per frame, per variable**, when `syncFromTfjs === true` |
| Field learn (fused) | `trainer.encodeStep`, `main.ts:2865-2884` | **Fused**, 2 dispatches (rollout A, optim B) | No |
| Fused loss readback | `trainer.readLoss()`, `main.ts:2885-2890` | – | `mapAsync`, **every 30 frames**, non-blocking (`train.ts:461-474`) |
| Adversary (fused) | `advTrainer.encodeStep`, `main.ts:2830-2850` | **Fused**, **5 compute passes** per branch: `pipeFwd` (pre-D) → `finalize` → `advOpt` → `pipeFwd` (post-D, gen) → `fieldGrad` (`adversary_train.ts:655-683`) | No |
| Adversary stats | `advTrainer.encodeStatsRead`, `main.ts:3004-3005` | – | `copyBufferToBuffer` + `mapAsync`, per training frame, **non-blocking**, self-throttled by a `pending` guard (`adversary_train.ts:754-756`) |
| Pixel-disc | `pixelDiscTrainer.encodeStep`, `main.ts:2852-2864` | **Fused**, 10 passes | `recordStats` 32-byte `mapAsync` per training frame, **non-blocking** — and dead (see §1) |
| Surprise diagnostic render | `main.ts:3125-3149` | **Fused** — `surprisePlane` is a GPU buffer offset, `sr.encodeRender` records into `enc` | `GpuSurpriseStats.encodeSample` → `mapAsync`, every N frames, non-blocking (`surprise_points.ts:296-298`) |
| Mean-speed probe | `GpuSpeedStats`, `main.ts:1488-1535` | – | 8 KB prefix copy every 8 frames, `mapAsync`, non-blocking (`:1520`) |
| Splat / curl-stroke render | `splat.encodeRender`, `main.ts:3167-3175` | **Fused**, recorded into `enc` | No |
| Points render | `renderer.encodeRender`, `main.ts:3177-3185` | **Fused**, zero-copy from `posBuffer`/`velBuffer` | No |
| GPU timestamp profiler | `timer.maybeResolve(enc, …)`, `main.ts:3192` | – | resolves ~every 15 frames, async |
| **tfjs learn (piece 17 only)** | `main.ts:2891-2989` | **NOT fused** | **Yes — 3 blocking `dataSync()` per step.** See §3 |
| `helmholtzChaosLoss(ZERO_FIELD_LOSS)` | `main.ts:949-962` | n/a | **Short-circuits to `() => tf.scalar(0)`** at `:960-962` — so on piece 17 the aesthetic term is free. The cost is the tape, not this. [V] |
| `spiralCoverLoss` / `spiralPlusCenterLoss` | `main.ts` | Their fused equivalents (`W_COVER`, `W_SPIRAL`, `W_CENTER`) are compiled into `train_wgsl` (`:97-126`, `:613-617`) and used by pieces 0–3. The tfjs versions are **unreached at baked defaults**. | n/a |
| Agree+Disagree game | `main.ts:3505-3521` | **Fused** — two independent `AdversaryTrainer`s sharing only the READ-ONLY field weights buffer, summed into one field Adam step | No (stats only, async) |

**Dead / revert-path code confirmed off the hot loop:** `src/renderers.ts`
(Canvas2D), `src/render/gpuPoints.ts` (WebGL2, `arraySync` stall documented at
`:9-13`), `src/draw/draw_canvas2d.ts`, `src/draw/draw_cpu.ts`,
`src/physics/updateParticles.ts:121`, `src/trashPanda/**`. All contain blocking
`arraySync`/`dataSync` but none is imported by the production loop. [V]

---

## 3. Deliverable 3 — is `(cpu·tfjs)` accurate?

### The label

`src/main.ts:3226-3242`, three branches:

```ts
if (gt && trainer)  → "rollout … optim … (gpu)"
else if (gt)        → `learn ${emaTrain} ms  (cpu·tfjs)`        // line 3232  ← HARDCODED
else                → `learn ${emaTrain} ms  ${trainer ? "(fused)" : `(tfjs·${tf.getBackend()})`}`
```

**[V] Line 3232 is a string literal. It performs no backend check.** It is
selected purely by "the GPU timestamp profiler exists AND there is no fused
trainer". The *third* branch (`:3240`) does the honest thing and prints
`` `(tfjs·${tf.getBackend()})` `` — its own comment at `:3237-3239` says
*"Honest label: tfjs's actual backend, not an assumed 'cpu'."* The middle
branch was never given the same treatment.

Worse: the HUD header two lines above prints `backend ${tf.getBackend()}`
(`:3210`). So on the default piece the HUD simultaneously displays
`backend webgpu` and `learn 92.7 ms (cpu·tfjs)`. **The label is misleading.**

### But is it *wrong*?

Not entirely. Three things are true at once:

1. **tfjs is on the webgpu backend.** `main.ts:3271-3279` does
   `await tf.setBackend("webgpu"); await tf.ready();` and *bails to the WebGPU
   warning* if `tf.getBackend() !== "webgpu"`. `?backend=` overrides are
   rejected loudly at `:3261-3268`. There is no path on which this page runs
   tfjs-cpu. [V]
2. **The CPU-handoff story is weaker than the label implies.**
   `WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD` (default 1000, registered at
   `node_modules/@tensorflow/tfjs-backend-webgpu/dist/flags_webgpu.js:48`, used by
   `shouldExecuteOnCPU` at `backend_webgpu.js:812`) **does not force readbacks
   and does not move the learn math to the CPU.** Its predicate requires
   `tensorMap.get(...).resource == null` — i.e. the data is *already* CPU-side —
   and only 11 kernels consult it (`Cast, Concat_impl, GatherV2, Neg, Slice,
   StridedSlice, Tile, Transpose, TopK, GatherNd, BroadcastArgs`). [V] Its real
   effect here is the one `advect.ts:353-363` documents: small weight tensors
   stay CPU-resident, which is why `advect.ts:389`'s `dataSync` is free and
   `dataToGPU` throws so often. **Every op on this learn path has a WebGPU
   kernel** — `OneHot, Mod, ArgMin, Selu, GatherV2, BatchMatMul, ClipByValue,
   Concat, Slice, Tile, Multiply, RealDiv, Select, Cumsum` all present. [V]
   The genuinely host-side work is narrower: `tf.randomUniform` is a plain JS
   loop (`tfjs-core/dist/ops/random_uniform.js:43-51`) run **twice per step**
   (`main.ts:2902`, `:2942`), and `Adversary.sampleIndices`
   (`adversary.ts:612-633`) does a CPU `Int32Array` + O(b·m·log m) sort per step.
3. **The dominant cost is stalls and dispatch, not "CPU".** See below.

**Verdict: `(cpu·tfjs)` is inaccurate as a mechanism claim.** The backend is
webgpu, every kernel on the path is a webgpu kernel, and the cost is
(a) blocking syncs, (b) forced queue flushes, (c) an O(N·gs²) one-hot encoding,
(d) five separate optimizer graphs. The precise label would be
**"tfjs·webgpu, dispatch- and sync-bound"**. Recommend changing `:3232` to reuse
the `:3240` expression, i.e. `` `(tfjs·${tf.getBackend()})` ``, so both branches
tell the same truth. (The rationale comment at `main.ts:3212-3219` says the
label was only ever meant to mean "CPU **wall time** for a span with no clean
GPU boundary" — the string just doesn't say that.)

### Named sync points on the tfjs learn step (piece 17, per step)

`advEvery` defaults to 1 (`main.ts:2566`), `trainEvery` defaults to 1
(`:2582-2585`), so all of this runs **every frame**.

| # | Site | Kind | Frequency |
|---|---|---|---|
| 1 | `src/core/gan/adversary.ts:2046` — `winMeta.dataSync()` (winner+active histogram) | **blocking readback** | per D step |
| 2 | `src/core/gan/adversary.ts:2079` — `packed.dataSync()` (k costs + batch RMS‖y‖, already batched from k separate reads) | **blocking readback** | per D step |
| 3 | `src/main.ts:1420` — `meanTen.dataSync()` (HUD surprise) | **blocking readback** | per D step |
| 4 | `src/core/gan/adversary.ts:2165` — `packed.dataSync()` inside `headSpread` | **blocking readback** | wall-clock gated to ~1 Hz (`main.ts:1430`) |
| 5 | `src/render/webgpu/advect.ts:389` — `this.vars[i].dataSync()` in the weight sync | readback, but **free when the tensor is CPU-resident** (which is the common case here) | **per variable, per frame** — 13 variables for dualHashgrid |
| 6 | `src/render/webgpu/advect.ts:369` — `dataToGPU()` | **forced tfjs queue flush.** `backend_webgpu.js:442-462` `readToGPU()` does `ensureCommandEncoderReady → endComputePassEncoder → copyBufferToBuffer → submitQueue()` | **per variable, per frame — up to 13 forced submits/frame**, plus 13 try/catch trips (`advect.ts:373-375` uses the `"not on GPU"` throw as control flow) |

Two things make items 1-4 worse than an ordinary readback:

- **tfjs's `readSync` on a GPU-resident tensor is an OffscreenCanvas hack.**
  `node_modules/@tensorflow/tfjs-backend-webgpu/dist/backend_webgpu.js:263` blits
  through a 256×256 `bgra8unorm` texture, after emitting a one-time
  `console.warn("The performance of synchronously reading data from GPU to CPU
  is poor on the webgpu backend, please use asynchronous APIs instead.")`. [V]
- **The weight sync runs on EVERY frame, not just training frames.**
  `advect.encodeStep` is called unconditionally at `main.ts:3113`, outside the
  `frame % trainEvery` guard. [V]

The codebase already measured the stall cost. `main.ts:2553-2560`: *"The tfjs
reference adversary does K small GPU→CPU readbacks per step (one Adam minimize
per head), which on the webgpu backend is a pipeline stall per head — measured
in browser QA at ~8 ms per head per frame."* [V]

### Tensor churn — why `tensors 207`

Per learn step on piece 17:

- **k = 4 separate `optimizers[j].minimize()` calls**, each building its own
  graph and its own Adam update (`adversary.ts:2053-2069`). [V]
- **1 generator `optimizer.minimize()`** over `varList = field.trainableWeights`
  (`main.ts:2939-2988`). [V]
- **The hashgrid encoding is the expensive part.** `HelmholtzField.gridInterp`
  (`src/core/field/helmholtz.ts:377-409`) cannot use `tf.gather` — gather's
  backward rejects int32 indices inside the tape — so it does
  **four `tf.oneHot(cell, gs*gs).matMul(grid)` calls per field evaluation**
  (`:395-398`). At `gs=32` that is a `[N, 1024]` one-hot **materialized in
  memory** and a `[N,1024] × [1024,4]` matmul, four times.
  With `sampleRate = 256` (`main.ts:2508`) each one-hot is 262,144 floats ≈ 1 MB.
  [V]
- The field is evaluated **twice per step** (once detached for the D batch at
  `main.ts:2913`, once on-tape for the G batch at `:2955`), so **8 one-hot
  materializations forward**, plus the backward through four of them
  (`[1024,256] × [256,4]` each). [V for the call structure, [H] for the
  arithmetic being the dominant term.]
- `adversaryTrainStep` deliberately does **not** use `tf.tidy` (`main.ts:1393-1396`)
  and hand-disposes every intermediate; the HUD's `tensors` counter
  (`main.ts:3244`) is the tripwire for that. 207 is the steady-state graph, not
  a leak. [H — not observed live, but the disposal code is complete.]

- **No particle readback.** [V] The tfjs learn step never touches
  `advect.posBuffer`/`velBuffer`. It synthesizes fresh uniform-random batches
  (`main.ts:2902`, `:2942`) with `tf.zeros` velocities (`:2912`, `:2954`).
  Worth flagging as a *semantic* difference, not just a perf one: the fused path
  defaults to `trainSource = "particles"` (`main.ts:3461-3468`), so **piece 17
  is not merely a slower version of the same objective** — it trains on a
  different state distribution than a fused piece would.
- Two tfjs gradient gaps force extra work on this path, both documented in
  place: `helmholtz.ts:390-397` (`tf.gather` backward rejects int32 indices on
  the tape ⇒ one-hot matmul) and `adversary.ts:640-650`
  (`tf.customGrad` wrapper because "tfjs's Greater op has no registered
  gradient"). [V]

**Conclusion for §3, cost ranked** [ordering H, mechanisms V]:

1. `helmholtz.ts:390-397` — 4× `[B, 1024]` one-hot matmuls per field forward,
   ×2 forwards per step, four of them **retained on the autograd tape** through
   backward. At B=256 that is ~4 MB/forward; at the dock's "train B 800" it is
   ~13 MB/forward. Roughly 256× larger than the data it selects.
2. `adversary.ts:2046` + `:2079` + `main.ts:1420` — 3 blocking `dataSync`
   pipeline stalls per step, each on the OffscreenCanvas `readSync` path.
3. `advect.ts:365-392` — up to 13 `dataToGPU` calls per **frame**, each forcing
   a `submitQueue()` inside tfjs.
4. `adversary.ts:2054-2069` — k=4 separate `minimize` calls (4 forwards, 4
   backwards, 4 Adam graphs) rather than one batched head.
5. ~10³ tiny dispatch-bound tensor ops per step, on a backend where every op is
   its own compute pass.

So: the 92.7 ms is **not** "tfjs fell back to the CPU backend". Fusing removes
all five at once.

---

## 4. Deliverable 4 — THE MAIN EVENT: fusing the hashgrid adversary

### 4.1 Why hashgrid is refused today (verified, and the hypothesis was half wrong)

The refusal is one throw:

```ts
// src/render/webgpu/adversary_wgsl.ts:419-424
if (field.encoding.kind === "hashgrid") {
  throw new Error(
    "adversary: hashgrid fields are not supported by the fused adversary yet " +
      "(fieldGrad lacks the grid-scatter blocks)"
  );
}
```

and the matching scope note at `adversary_wgsl.ts:58-59`: *"field must be
helmholtz, classes == 0, encoding raw | fourier. hashgrid needs the dEnc
scatter blocks ported into fieldGrad — not done."* [V]

**The stated hypothesis was: "the generator-reward backward needs d(force)/d(field-weights)
through the encoding — hashgrid interp/scatter with atomics." Verified against
the code, this is CORRECT about *what* is missing and WRONG about *how* it is
implemented.**

- ✅ Correct: the missing piece is exactly `dL_gen/dW` for the grid table, i.e.
  the `role: "grid"` segment of `field.segments` (`advect_wgsl.ts:225`,
  `:360-368` — the grid is reserved at **offset 0** of the packed field weights
  buffer, length `gridSize² · features`).
- ❌ Wrong: **there are no atomics.** `train_wgsl.ts:1215-1248` implements the
  grid gradient **gather-side**: *thread = one grid float*, which scans every
  `(sample, site)` and claims the bilinear corners that land on its cell. Its
  own comment says so: *"the scatter that tfjs's oneHotᵀ·(w·dOut) matmul
  backward performs, expressed gather-side"*. Deterministic, race-free, and
  bit-reproducible. [V]
- ❌ Also wrong in the framing: `adversary_wgsl.ts` does **not** "predate"
  train_wgsl's encoding support. It already `import`s `emitEncode`,
  `emitFwdStore`, `emitBwdStore`, `trainScratchLayout` from `train_wgsl`
  (`adversary_wgsl.ts:71-80`) and emits them into pass A
  (`:1370-1373`). The **forward** hashgrid interp and the **per-head
  `dL/dEnc` + `du` jacobian** would already be generated correctly today. Three
  small things stop it working:

**Gap A — the adversary scratch has no `dEnc` block.**
`advScratchLayout` (`adversary_wgsl.ts:343-390`) reuses `trainScratchLayout` as
`fieldSl` and allocates `encOff … fieldSiteOff = encOff + m * fieldSl.encStore`
(`:381`). It uses `encStore` but **never `dEncStore`**. In `train_wgsl` those
are two separate blocks (`:232-236`: `encStore = encDim` for non-raw,
`dEncStore = encDim` for hashgrid only, `dEncOff = encOff + sites·encStore`).
So the adversary layout is missing `m · encDim` floats and a `dEncOff`. [V]

**Gap B — `bwdCall` emits the wrong signature.**
`emitBwdStore`'s hashgrid signature is
`bwd_head_h(dOut, base, uIn: vec2f, dEncBase: u32)` (`train_wgsl.ts:474`),
four arguments. The adversary's `bwdCall` helper
(`adversary_wgsl.ts:985-988`) only knows two shapes — raw (2 args) and fourier
(3 args, passing `encBase`). A hashgrid field would emit a call with the
fourier arity and fail to compile. [V]

**Gap C — `fieldGrad` cannot even iterate the segment list.**
`adversaryPassBShader`'s field-block loop (`adversary_wgsl.ts:1749-1793`) opens
with the comment *"grid segments are rejected by validateAdversaryFusion"* and
immediately does `const h = seg.head; … heads[h].layers[l]`. A grid segment has
`head: -1, layer: -1` (`advect_wgsl.ts:366`), so it would index `heads[-1]`.
And even guarded, there is no block that computes the grid gradient. **This is
the one piece of genuinely new WGSL.** [V]

### 4.2 The WGSL that already exists and can be lifted

| Piece | Location | Reusable as-is? |
|---|---|---|
| Forward bilinear interp `encodeSite(uIn, encBase)` | `train_wgsl.ts:363-383` (`emitEncode`) | **Yes, unchanged.** Reads `weights[b00+fi]…` and pass A binds the field weights buffer as `weights` at binding 1 (`adversary_wgsl.ts:1320`, read-only). Already emitted at `:1370`. |
| Per-head `dL/dEnc` accumulation + store | `train_wgsl.ts:541-546` (`emitBwdStore` tail) — head 0 seeds (`=`), head 1 accumulates (`+=`) | **Yes**, once Gap A gives it a `dEncBase` to write to. |
| `du` through the bilinear-interp jacobian (grid-value differences × (gs−1), clip-masked) | `train_wgsl.ts:547-567` | **Yes, unchanged.** |
| **Grid-table gradient block** (gather-side, thread = grid float) | `train_wgsl.ts:1225-1247` | **Transliterate.** Offsets/loop bounds differ: `sl.siteInOff`→adversary's, `sl.dEncOff`→new, `sl.sites`→`m`, `ub.n`→`ub.b`, `STRIDE`→`sl.stride`. Math identical. |
| `maxWField` sizing | `adversary_wgsl.ts:940-943` already does `Math.max(2, fsl.encDim, …)` | **Yes, no change.** |
| `extGrads` buffer length | `adversary_train.ts:335` — `mkStorage(field.totalFloats * 4)`; `totalFloats` already includes the grid segment | **Yes, no change.** |
| Field pass B extGrad sum | `train_wgsl.ts:1343` — `g = g + extGrad0[t]`, applied to **every** weight float including grid floats | **Yes, no change.** |

**Important consequence [V]:** `train_wgsl.ts:1214` guards the *entire* field
gradient-assembly loop — grid block included — behind `if (hasStructuralLoss)`.
Piece 17 uses `ZERO_FIELD_LOSS`, so `hasStructuralLoss` is false, internal `g`
starts at literal `0.0` (`:1340,1342`) and `g` is **identically** `extGrad0[t]`.
Meaning: on the default piece, the field trainer's own grid backward never
runs, and once the adversary produces grid extGrads, **100% of the grid update
comes through the extGrad seam.** No summation subtlety, no double-count.

### 4.3 The plan

**Files to touch (3 source files, 1 optional):**

1. `src/render/webgpu/adversary_wgsl.ts` — the whole port.
2. `src/main.ts` — delete `&& advect.layout.encoding.kind !== "hashgrid"` from
   `fusedAdvOk` (`:3390`), and update the piece-17 comment at `:2151-2157`
   which currently documents the refusal as permanent.
3. `tools/train_wta_test.ts` — new hashgrid section (see §4.5).
4. *(optional)* `src/render/webgpu/pixel_disc_wgsl.ts:92-104` +
   `main.ts:3539` carry the identical hashgrid refusal for the pixel critic.
   Out of scope; note it so nobody assumes "hashgrid works now" means both.

**Change list, in order:**

**(1) `advScratchLayout` — add the dEnc block.** `adversary_wgsl.ts:374-389`:
```
  const encOff       = siteInOff + 2 * m;
+ const dEncOff      = encOff + m * fieldSl.encStore;
- const fieldSiteOff = encOff + m * fieldSl.encStore;
+ const fieldSiteOff = dEncOff + m * fieldSl.dEncStore;
```
plus `dEncOff` on the `AdvScratchLayout` interface (`:310-341`). For non-hashgrid
encodings `dEncStore === 0` so the offsets collapse and **every existing
generated shader stays byte-identical** — the same discipline `train_wgsl` uses
at `:229`. This matters: `tools/kernel_test.ts`'s f32 codegen guard must remain
meaningful for raw/fourier.

Buffer growth: `advScratchBytes` (`:392-401`) derives from `stride`, so
`adversary_train.ts:322-331` picks it up automatically. For the pair/hashgrid
layout the added block is `m·F = 2·4 = 8` floats/tuple against a stride of
~840 → **+1%**, ~27 KB at `batchCap: 1024`. Non-issue. [H — stride arithmetic
estimated from the layout, not printed.]

**(2) `bwdCall` — emit the hashgrid arity.** `adversary_wgsl.ts:985-988`:
```ts
const dEncBase = (site: string) =>
  `sBase + ${sl.dEncOff}u + (${site}) * ${fsl.encDim}u`;
const bwdCall = (h: number, dExpr: string, site: string) =>
  enc.kind === "raw"
    ? `bwd_head_${h}(${dExpr}, ${fieldBase(site, h)})`
    : enc.kind === "fourier"
    ? `bwd_head_${h}(${dExpr}, ${fieldBase(site, h)}, ${encBase(site)})`
    : `bwd_head_${h}(${dExpr}, ${fieldBase(site, h)}, pn[${site}], ${dEncBase(site)})`;
```
`pn[t]` is the normalized member position, already a live local at the backward
site (`adversary_wgsl.ts:1429,1435`, backward loop at `:1538-1540`). [V]

**(3) The `fieldLane` seeding hazard — READ THIS BEFORE SHIPPING AGREE+DISAGREE.**
`emitBwdStore`'s hashgrid store is `head 0 → "=", head 1 → "+="`
(`train_wgsl.ts:543-544`). In `train_wgsl` both heads always run, so head 0
always seeds. In the adversary, `fieldBackward` (`adversary_wgsl.ts:993-997`)
runs **both** heads only when `fieldLane === "blend"`. With `fieldLane === 1`
(Agree+Disagree lane B, `main.ts:3510`) **only head 1 runs, and it does `+=`
into a block nobody seeded.** Two options:
   - (a) Emit an explicit zero-fill of the dEnc block before the backward loop
     when `fieldLane !== "blend"` (3 lines, always correct, negligible cost).
   - (b) Make the seed/accumulate choice lane-aware in the emitter.
   Prefer **(a)** — it keeps `emitBwdStore` shared and unmodified.
   Note this hazard does **not** bite the blend lane, and pass A runs twice per
   step (pre-D `pipeFwd` then post-D `pipeFwd`, `adversary_train.ts:655-676`),
   with head 0 re-seeding on each — so the two forwards do not contaminate each
   other in blend mode. [V]

**(4) `fieldGrad` — the new grid block.** `adversary_wgsl.ts:1749-1793`, at the
top of the `for (const seg of field.segments)` loop, before `const h = seg.head`:
```ts
if (seg.role === "grid") {
  const { gridSize: gs, features: F } = enc as Extract<Encoding, {kind:"hashgrid"}>;
  fieldBlocks.push(/* transliteration of train_wgsl.ts:1226-1247 with:
     ub.n      -> ub.b
     ${STRIDE} -> ${sl.stride}
     sl.siteInOff -> ${sl.siteInOff}
     sl.dEncOff   -> ${sl.dEncOff}
     sl.encDim    -> ${fsl.encDim}
     site < ${sl.sites}u -> site < ${m}u
  */);
  continue;
}
```
The `fieldLane !== "blend" && h !== fieldLane` skip at `:1755` must **not**
apply to the grid segment — the grid is shared by both heads, and lane
isolation is already achieved upstream (only the active lane's `bwd_head`
contributed to `dEnc`). Put the `role === "grid"` branch **before** that check.

**(5) Delete the throws.** `adversary_wgsl.ts:419-424` and `main.ts:3390`.
Update the scope docstring at `adversary_wgsl.ts:58-59`.

### 4.4 Constraints and invariants that must survive

- **`fieldClasses === 0` stays.** `validateAdversaryFusion:414-418` and
  `main.ts:3389` (`advect.layout.classes === 0`). Class channels are raw-only by
  construction (`layoutField`), and `train_wgsl`'s class-aware layer-0 block
  (`:1267-1288`) has no encoded-input counterpart. **Do not touch this** — it is
  an orthogonal gap. `dualHashgrid` sets no `classes`, so piece 17 is unaffected.
- **Disjoint packed segments stay disjoint.** `FusedTrainer` owns
  `advect.weightsBuffer` (`main.ts:3585`) and Adam-updates it in place;
  `AdversaryTrainer` owns `advWBuf`/`adamM`/`adamV` and binds the field weights
  **read-only** at pass A binding 1 (`adversary_wgsl.ts:1320`). The hashgrid
  table lives at **field** offset 0 (`advect_wgsl.ts:360-368`) — i.e. inside
  the FusedTrainer's segment. The adversary never writes it; it writes
  `extGrads[0 … gridFloats)` and the field's Adam applies it. **The disjointness
  invariant is preserved unchanged.** [V]
- **Numerical stability policy — all of it is already in the shared emitters
  and must not be weakened:**
  - ε inside every `sqrt` radicand: `ADV_SOFT_EPS2 = 1e-12`
    (`adversary_wgsl.ts:90`), used in the soft-sphere gradients at `:1355-1371`.
    The new grid block introduces **no new sqrt** — it is `wsum * dEnc`.
  - SELU preact clamp ±80: in `COMMON` (`train_wgsl.ts:277-280`), shared by both
    shaders. Untouched.
  - `isFiniteF` gates: residuals/κ at `adversary_wgsl.ts:1442-1445,1454`; Adam
    at `:1828,1841`; **extGrads at `:1855`** —
    `extGrads[t] = select(0.0, g, isFiniteF(g))`. Because the new grid block
    accumulates into the same `var g`, it inherits that gate for free. **Do not
    add a separate write path.**
  - The field's own Adam gate `train_wgsl.ts:1347,1361` likewise covers the
    grid floats.
  - The hashgrid `du` clip mask (`train_wgsl.ts:564-567`) reproduces tfjs
    `clipByValue`'s zero-gradient-outside-[0,1]. Keep it — probe sites can
    exceed 1.

### 4.5 Test gates

**Existing precedent (all verified by reading the suites):**

| Suite | Oracle | Assertion |
|---|---|---|
| `tools/train_types_test.ts:138-164` | offline tfjs fixture from `tools/grad_reference.ts:325` (`tf.variableGrads`) | per-variable **`cos > 0.99999 && relMax < 2e-3`**; loss rel-err `< 1e-3` (`:123-134`); 30-step monotone-decrease liveness (`:166-182`) |
| `train_types_test.ts:55-88` | — | **hashgrid fixture already exists**: `tools/fixtures/grad_ref_hashgrid.json` (378 KB), regenerate with `MODEL=hashgrid OUT=… bun tools/grad_reference.ts`. Grid table is **variable index 0**, sliced off before head-dim derivation (`:86-88`) |
| `tools/train_wta_test.ts` §2 `:583`, `:801-804` | **AD-IR** (`src/render/webgpu/ad/losses.ts::wtaTerm`) | disc `cos > 0.99999 && rel < 1e-3`; extGrads `cos > 0.99999 && rel < 1e-3` |
| §3 `:1607`, `:1769-1771` | live tfjs `Adversary` + two tfjs field heads | generator reward `cos > 0.99999 && rel < 1e-3` |
| §3b `:1782`, `:2100-2104` / `:2154-2158` | live tfjs `tf.variableGrads` | disc `cos > 0.99999 && rel < 2e-3`; `generatorLoss` `cos > 0.99999 && rel < 3e-3` |
| §5 `:2252-2330` | — | **the extGrad seam**: (a) 5 adversary steps leave field weights bit-identical `:2277`; (b) 5 field steps leave adversary weights bit-identical `:2299` + a liveness gate `:2301`; (c) `grads(with) − grads(without) ≡ extGrads`, `cos > 0.9999 && rel < 1e-3` `:2305-2323` |

**Critical constraint for the new fixture [V]:** **the AD-IR cannot be the
oracle for hashgrid.** `src/render/webgpu/ad/rollout.ts:52-60` types `encoding`
as `raw | fourier` only, with the comment *"HashGrid is NOT expressible in this
static scalar graph (its gather indices are data-dependent) — its fused
backward is verified directly on Metal vs the tfjs fixture instead."* So §2's
`oracleCheck` pattern is unavailable; the new section must follow **§3/§3b**
(live tfjs) or `train_types_test.ts` (offline fixture).

Also verified: **`train_wta_test.ts` contains zero occurrences of `hashgrid`.**
Every `oracleCheck` uses `enc: {kind:"raw"}` except one fourier case at `:833`;
§3/§3b hardcode raw via `layoutField("helmholtz", [fieldDims(2), fieldDims(2)])`.

**What the new hashgrid-adversary fixture must cover:**

1. **Negative gate first** — assert `new AdversaryTrainer(hashgridLayout, …)`
   *no longer* throws, and that a `classes > 0` hashgrid layout *still* does
   (add to §1's throw-gates at `:562-579`).
2. **D grads.** §3b pattern, hashgrid field: the discriminator's own weights are
   structurally independent of the field encoding, so this should pass
   unchanged — but assert it, because a scratch-stride bug would silently
   corrupt `advOff` reads. Threshold `cos > 0.99999 && rel < 2e-3`.
3. **G reward grads through the hashgrid.** §3 pattern against a live tfjs
   `HelmholtzField` with `modelType: "hashgrid"` (`helmholtz.ts:119,230`), which
   routes through `gridInterp`'s `oneHot.matMul` backward. Compare **the whole
   `extGrads` vector including the grid segment**. Threshold
   `cos > 0.99999 && rel < 1e-3`.
4. **Grid segment specifically, not just the aggregate cosine.** The grid table
   is `gs²·F = 4096` floats vs ~2564 MLP floats, so a *completely wrong* grid
   block would still leave the aggregate cosine misleadingly high if the MLP
   part dominates the norm — and vice versa. **Assert the grid slice
   separately.** Mirror `grad_reference.ts:421-423`, which already excludes the
   grid table from the "≥95% nonzero entries" tripwire because it is
   legitimately sparse (only touched cells get gradient).
5. **Sparsity/coincidence corners.** Force a batch where (a) two tuple members
   land in the *same* cell, and (b) a member sits exactly on the clamp border so
   `ix1 == ix` and **two corners match the same cell** — `train_wgsl.ts:1221`
   documents that both must add, matching tfjs summing coincident scatters.
   These are the only places the gather-side formulation can diverge from tfjs.
   *(Note: there is no race to test. The gather formulation has one writer per
   grid float by construction.)*
6. **extGrads → pass B seam.** §5(c) pattern verbatim, on a hashgrid layout:
   `grads(with extGrad) − grads(without) ≡ extGrads` over **all** `totalFloats`
   including offsets `[0, 4096)`. `cos > 0.9999 && rel < 1e-3`.
7. **Lane isolation** for `fieldLane: 0|1` (the Gap-3 hazard): assert the dEnc
   block is correctly seeded, e.g. that a lane-1 run's grid extGrad is
   *not* NaN and matches a tfjs single-head reference.

**Runner:** no test runner is configured (`AGENTS.md:86`). Commands are
`bun tools/train_wta_test.ts` (`:6`) and `bun tools/train_types_test.ts`
(`:11-12`); GPU suites are **sequential** — run nothing else on the GPU
(`train_wta_test.ts:4`).

**Stability probes — all three currently hardcode raw and would throw. [V]**

- `tools/quad_nan_probe.ts` — `dims()` uses `inSize: 2` (`:84-90`) and
  `layoutField("helmholtz", [dims(), dims()])` (`:92`) passes **no encoding**,
  defaulting to raw (`advect_wgsl.ts:338`). `TAG` selects the *tuple* encoding
  (`quad-labelled|tri|pair|point`, `:45-49`), **not** the field encoding. Env:
  `STEPS`(4000) `REPORT`(30) `B`(256) `K`(6) `N`(20000) `FIELD_LR`(.001)
  `DISC_LR`(.003) `WEIGHT`(.012) `DRIVE`(.65) `TAG` `FINE_AFTER`(-1). It is a
  **finiteness** probe, not a parity test: six stages checked in order
  (`stats, advGrad, advWeight, extGrad, fieldGrad, fieldWeight`, `:292-299`),
  prints `FIRST NONFINITE step=… stage=…`, exits 1.
  **To soak hashgrid it needs (a) an `ENC=hashgrid` knob, and (b) grid-table
  weight init** — its init loop only fills `role === "kernel"` segments
  (`:96-103`), so a grid segment would be left at all zeros, which is a
  degenerate field and a false "finite" pass.
- `tools/adversary_stability_probe.ts` — same raw hardcoding, same `TAG`
  confusion, no advect (which is why `quad_nan_probe` exists; `AGENTS.md:93-95`).
- `tools/soak_adversary.mjs` — positional args
  `node tools/soak_adversary.mjs <pieceKey|all> [baseURL] [durationSec] [sampleSec]`
  (`:4-5,614,629-631`); **no `TAG=`/`N=`/`FINE_AFTER=`**. Piece keys are a
  frozen map `{single, wta8, pair4, tri6, quad6, agree, weave}` (`:42-50`) — all
  raw pieces. Its gates include *"every sample reports the fused adversary"*
  (`:435`) and *"no NaN/Infinity in telemetry"* (`:436`). **Adding a `hashgrid`
  key pointing at piece 17 would make that suite the end-to-end proof that the
  default piece is fused** — the gate at `:435` is precisely the assertion we
  want, and it fails today.

### 4.6 Scope estimate

| Work | Kind | Size |
|---|---|---|
| `advScratchLayout` dEnc block + interface field | mechanical | ~6 lines |
| `bwdCall`/`dEncBase` helpers | mechanical | ~8 lines |
| `fieldLane` dEnc zero-fill guard | new, small | ~5 lines WGSL + a codegen conditional |
| `fieldGrad` grid block | **new WGSL — transliterated**, math unchanged from `train_wgsl.ts:1226-1247` | ~25 lines |
| `role === "grid"` early-branch before the lane skip | mechanical | ~3 lines |
| Delete 2 throws + update 3 comment blocks | mechanical | ~10 lines |
| New `train_wta_test.ts` §8 hashgrid section | test | ~200-300 lines, following §3b + §5 |
| `ENC=` knob + grid init in the two bun probes | test infra | ~30 lines |
| `soak_adversary.mjs` piece key | test infra | 1 line |

**Genuinely new algorithm: none.** Every formula already exists and is already
gated at cos = 1.0 against tfjs on the field-trainer side. The port is a layout
change plus a transliteration.

**Expected result [H, not measured]:** `fieldGrad` grows from
`ceil(2564/64) = 41` to `ceil(6660/64) = 105` workgroups; the 4096 grid threads
each loop `b · m` iterations (≈512 at `b=256, m=2`) of ~4 integer compares and
one multiply-add — ≈2.1 M lightweight iterations. Against the measured 0.7-0.8
ms/step for the fused raw adversary at B=512, expect **~1.5-3 ms/step**, i.e.
a **30-60×** improvement over 92.7 ms, and the piece should return to 60 fps.
If it proves hot, the follow-up is a fixed-point atomic scatter (`b·m·4·F` =
8192 `atomicAdd`s instead of 2.1 M gather iterations) — **precedent already in
this repo** at `pixel_disc_wgsl.ts:1111-1114`. But do that *after* parity is
locked: fixed-point breaks bit-exactness with the tfjs oracle, and the gather
form is what the existing `cos > 0.99999` gate was written against.

---

## 5. Recommended order of work

1. **One-line honesty fix first** (independent of everything else): change
   `main.ts:3232`'s `(cpu·tfjs)` to `` `(tfjs·${tf.getBackend()})` `` so it
   matches `:3240` and stops contradicting the `backend webgpu` line at `:3210`.
2. Add the negative gate to `train_wta_test.ts` §1 asserting the current
   hashgrid throw — so the suite records the *before* state.
3. Layout changes (Gap A) + `bwdCall` (Gap B) + lane guard (Gap 3). Verify
   raw/fourier shaders are still **byte-identical** (`tools/kernel_test.ts`).
4. `fieldGrad` grid block (Gap C). Delete the throws.
5. New hashgrid parity section, thresholds per §4.5.
6. `ENC=hashgrid` in `quad_nan_probe.ts` (with grid init!) → run the coupled
   NaN probe.
7. Add the `hashgrid` piece key to `soak_adversary.mjs`, soak piece 17, confirm
   the *"every sample reports the fused adversary"* gate now passes.
8. Re-measure the HUD line and update the piece-17 comment at `main.ts:2151-2157`.

---

## 6. Open questions / unresolved

- **[H]** The 92.7 ms attribution is reasoned from the call structure, not
  profiled. If someone wants the receipt, `?handoff=0` (`main.ts:3298-3302`)
  forces every tfjs op onto the GPU — A/B the HUD `learn` line with and without
  it to separate "CPU-forwarded small ops" from "pipeline stalls + oneHot".
- The pixel critic carries the **same** hashgrid refusal
  (`pixel_disc_wgsl.ts:92-104`, `main.ts:3539`). Fusing the relational adversary
  for hashgrid does **not** fix the pixel path. Decide whether that is a
  follow-up or explicitly out of scope, and say so in the code comment — right
  now a reader could reasonably assume one implies the other.
- The two pixel-gate bugs in §1 (silent extGrad drop under Agree+Disagree,
  silent disable under `?train=tfjs`) are unrelated to this work but are live
  footguns. Neither is triggered by a shipped piece today.
- **[V] Batch-size divergence on the tfjs path.** `createAdversary(advSpec,
  sampleRate, …)` (`main.ts:3418-3423`) bakes the adversary's tuple count at
  construction, but `setSampleRate` (`main.ts:3725-3737`) only mutates the
  `sampleRate` feeding `tf.randomUniform`. Dragging the "train B" control grows
  the field batch N while the adversary's B stays pinned at its startup value —
  they silently diverge. Fusing piece 17 makes this moot for the default piece
  (the fused path routes B through `adversaryBatchSize(sampleRate, 1024)` at
  `main.ts:2822` every step), but the bug remains for `?train=tfjs`.
- **[V] The `packCurrentWeights` init path only fills `role === "kernel"`
  segments in the bun probes** (`quad_nan_probe.ts:96-103`). Not a production
  bug — flagged again here because it is the trap waiting for whoever adds
  `ENC=hashgrid` to those probes.
- `docs/PIXEL_DISC.md` (currently modified in the working tree) never uses the
  words "fused", "WGSL", or "GPU" in prose, even though `pixelDisc` is
  **fused-only** — unlike the relational adversary, which has both paths. Given
  that fused-vs-tfjs is the load-bearing distinction across this codebase, that
  is a real documentation gap. Its measured-authority numbers also disagree with
  `main.ts:1935-1937` in the last digit (`1e-2`/`0.12%` vs `8e-3`/`0.09%`) —
  same incident, two transcriptions.
