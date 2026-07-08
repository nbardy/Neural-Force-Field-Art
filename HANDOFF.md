# Handoff — Neural Force Field Art (WebGPU) + the fused-kernel plan

> **NEW (2026-07): prompt→splats app — COMPLETE & verified end-to-end.**
> A SECOND app (separate from the particle art): type a text prompt → ~12K 2D
> Gaussian splats live-optimize on a canvas to match it, entirely GPU-resident.
> The loop, one submit per step: rasterize splats → **fused WGSL MobileCLIP-S0
> forward** (99 dispatches, ≈6.2 ms/forward on Metal-3, cosine 1.000000 vs the
> ONNX oracle) → −cos(image, text) loss → **hand-written analytical CLIP
> backward** (weights frozen → dL/dpixels only) → **splat rasterizer backward**
> → **fused Adam**. Text embedding via transformers.js (once per prompt).
> Verified: `bun tools/splat/optimize_test.ts` (cos to "cat" rises 0.17→0.34 in
> 20 steps, ranks cat>dog>diagram) and `node tools/splat/page_smoke.mjs` (real
> GPU: readable cat, cosine 0.15→0.46). Page: `src/splat_page.ts` +
> `src/splat.html` (2nd parcel entry). **Read `tools/clip/README.md` first**
> (pipeline, perf ladder, gotchas) then `docs/splat_raster_spec.md` +
> `docs/clip_backward_spec.md`. **Deploy target: static GitHub Pages** (see the
> pipeline memory note / `tools/clip/README.md`). Known next quality lever:
> CLIP-augmentation (random crops) to reduce the rainbow "confetti".
>
> Non-obvious config trap: the LEGIBLE defaults are ~12K LARGE opaque splats
> (`LEGIBLE_*` in `src/splat/optimize.ts`); the old 200K tiny translucent
> splats average to gray NOISE by construction — never use that for output.

Live: **https://nbardy.github.io/Neural-Force-Field-Art/** · `main` @ deploy is pushed to the `gh-pages` branch (see AGENTS.md for build/deploy).

> **SESSION WRAP (2026-07-07): the whole engine is now fused + fast + shipped.**
> Both Phase 1 (advect) AND Phase 2 (train) are DONE, plus a new render path.
> Headline: **1,000,000 particles @ 60 FPS in-browser, including on retina**
> (dpr 2). Full state below; the design map for what's next is
> `docs/DESIGN_SPACE_PARTICLE_ART.md`.
>
> - **Fused advect** (`render/webgpu/advect{,_wgsl}.ts`): 1 dispatch; f16 fast
>   path auto-picked when `shader-f16` present (~3.7 ms @ 1M; f32 byte-identical).
> - **Fused trainer** (`render/webgpu/train{,_wgsl}.ts`): analytic BPTT + Adam
>   in 2 dispatches; grads ≡ tfjs autograd (cos 1.0000000). K-step rollouts
>   (`?rollout=K`), particle-sourced batches (real pos+vel; default), `?mix=F`,
>   `?window=K` (= trajectory-window, proven ≡ real advected path to 6e-5 px),
>   `?trainEvery=N`. Multi-species classes (chaos head `r(pos, onehot(c))`,
>   storage-free class hash; "Helmholtz · Species" piece).
> - **Compute-splat renderer** (`render/webgpu/splat.ts`): atomic accumulation
>   + fused-decay tonemap; radial cone dots (round, not squares), retina/dpr,
>   ghost trails via decay, per-class hues. Default renderer at all counts;
>   `?render=quads` reverts. Knobs: trails slider, `?dot=`, `?decay=`, `?exposure=`.
>   Perf: native cone radius capped at 1.6 + decay fused into tonemap took
>   1M-retina render 32 ms → 8.7 ms.
> - **HUD**: real per-pass GPU times via timestamp-query (rollout/optim/advect/
>   render, in ms). The old sub-0.1ms "render" line was CPU *encode* time, not
>   GPU time — now every shown time is either real GPU `(gpu)` or explicitly
>   `(cpu·tfjs)` / `(cpu-encode)`.
> - **Verify** (headless, real Metal via bun-webgpu): `bun tools/{kernel,train,
>   integration,window,splat}_test.ts` + `tools/grad_reference.ts` fixtures.
>   Browser QA: `node tools/qa_browser.mjs` (new-headless Chrome, --use-angle=metal).
> - **Open threads**: task #16 subgroup pre-summation (last WebGPU splat-perf
>   lever, parked — 8.7ms already clears 60); task #17 selectable Fourier/SIREN
>   model types (per `DESIGN_SPACE_PARTICLE_ART.md`; ship as COMPARISON options).
> - Deploy needs `--public-url ./`; parcel cache can serve stale bundles
>   (`--no-cache`). The tfjs device is shimmed to request shader-f16 +
>   timestamp-query (features are creation-time-only; see main.ts top).
>
> The older Phase-1/Phase-2 narrative below is retained for the reasoning/traps.

This doc is the pick-up point. It covers (1) where the engine is now — **Phase 1
(the fused ADVECT WGSL kernel) is SHIPPED and verified** — (2) the perf mental
model, (3) the remaining big step — the **fused TRAIN kernel** — and what
**AlphaGOJS got right** that we copied, plus the traps it hit that we avoided.

---

## 1. Where it is now

A tiny neural "force field" is trained with TF.js autograd to move particles;
advection is now a **single fused WGSL compute dispatch**, and rendering binds the
kernel's own GPUBuffers directly. Zero-copy, end to end.

**The loop (per frame), three stages:**
- **learn** — one Adam step (tfjs autograd) on a small random batch (`sampleRate`,
  default 256). It samples **random points**, not particles — training never
  touches the particle cloud, so cost is independent of particle count.
- **advect** — **ONE fused `@compute` dispatch over ALL `particleCount` particles**:
  the generated MLP eval + integrate + friction + clip + wrap + fused PCG random
  reset, all as register math. Replaces the old ~40-dispatch tfjs advect stage.
  "Advect" = transport by the flow.
- **render** — the renderer binds the kernel's **pos/vel `GPUBuffer`s directly**
  (`renderer.renderFromBuffers(posBuffer, velBuffer, count, …)`) and draws
  instanced WGSL round dots. **True zero-copy — no readback, no tfjs tensor in the
  render path.**

**Where the state lives now.** Particle state is in **kernel-owned GPUBuffers**
(`AdvectKernel.posBuffer` / `velBuffer`) — NOT tfjs tensors. tfjs never touches
particles. Each frame the trained weights flow **tfjs → kernel** as ~10KB of
GPU→GPU copies: `variable.dataToGPU()` → `copyBufferToBuffer` into the packed
weights buffer, one segment per variable, zero readback. There is a second,
equally-valid upload path for CPU-resident weights: **tfjs-webgpu forwards
small-tensor ops to the CPU backend** (`WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD`), and
our weight tensors are tiny, so even freshly Adam-updated values often live
CPU-side — then `dataToGPU()` throws "not on GPU" and we `dataSync()` +
`writeBuffer` instead. **Both paths land identical bytes** (proven by the
integration test). This is COMMON, not an edge case.

**Deliberate design: "train few, advect many."** Training is decoupled from
advection so training cost doesn't scale with particle count. The advect HUD line
is **gone** — advect is now just GPU encoding on the CPU side, so the HUD shows
only `learn` and `render` ms. The field-based **Helmholtz · Chaos** piece now
defaults to **200k particles** with a **log-scale slider up to 1M**
(`src/index.tsx`: linear had no low-end resolution over 200 → 1M). It's
**first-order** (two direct-vector heads `F = (1-α)·g + α·r`, NO `tf.grad` inside
`forces()`); its chaos+divergence loss shares a single batched `[3N,2]` forward.

**Key files**
- `src/main.ts` — loop, physics (`physicsForward`), gallery, losses, WebGPU init,
  the `learn → advect.step() → renderFromBuffers` tick.
- `src/render/webgpu/advect_wgsl.ts` — **PURE WGSL codegen** for the fused kernel
  (zero imports, testable headless).
- `src/render/webgpu/advect.ts` — **AdvectKernel**, the tfjs-coupled half: owns the
  pos/vel buffers, the per-frame weight sync, the one dispatch.
- `src/core/field/helmholtz.ts` — the order↔chaos field.
- `src/core/losses/*` — isotropy / chaos / divergence / spectrum.
- `src/render/webgpu/{microgpu.ts,points.ts}` — WebGPU microlib + renderer
  (`renderFromBuffers`).
- `tools/kernel_test.ts`, `tools/integration_test.ts` — **headless GPU suites**
  (see next section). `tools/smoke.mjs` — Puppeteer WebGPU smoke (screenshot + console).
- Dead code kept as a revert path: `src/renderers.ts` (Canvas2D), `src/render/gpuPoints.ts` (WebGL2).

**WebGPU gotchas already handled in `main.ts` (keep them — do NOT re-break):**
1. `import "@tensorflow/tfjs-backend-webgpu"` — the union package only registers cpu+webgl.
2. `GPUAdapter.prototype.requestAdapterInfo` shim — tfjs 4.10 calls the API Chrome removed (`adapter.info`).
3. All tensor/model/**kernel** creation deferred until after `await tf.ready()` (webgpu init is async).
4. Build with `parcel build --no-scope-hoist` (scope-hoist crashes tfjs at runtime).
5. **Deploys need `--public-url ./`** — the site lives at the
   /Neural-Force-Field-Art/ subpath; parcel's default absolute `/index-*.js`
   404s on github.io. Deploy build:
   `./node_modules/.bin/parcel build --no-scope-hoist --no-cache --public-url ./ --dist-dir dist_deploy`
6. **Parcel cache can serve STALE bundles silently** — a gallery/src change once
   built "successfully" without appearing in dist. When a change doesn't show up:
   `./node_modules/.bin/parcel build --no-scope-hoist --no-cache`.
7. **Splat renderer is atomic-bound; taps ≈ cost.** At retina (dpr>1) the
   native cone radius is capped at 1.6 (NATIVE_RADIUS_MAX) — bigger blows up
   the tap count (radius×dpr → box×dpr²). Decay is FUSED into the tonemap pass
   (fragment reads acc → displays → writes acc×decay), so there is NO separate
   decay compute pass. Together these took 1M@retina from ~25 FPS to ~60
   (render 32ms→8.7ms). Measured via scratchpad pass-isolation benches — splat
   dominates, decay/tonemap are cheap.
8. **Real-GPU browser QA is automatable**: `node tools/qa_browser.mjs` drives
   new-headless Chrome with `--use-angle=metal` (real Apple adapter, unlike
   smoke.mjs's SwiftShader), clicks through pieces, probes HUD, screenshots.

---

## 2. The perf mental model (read this before optimizing)

**On the WebGPU backend, cost is DISPATCH-bound, not element-bound.** Every tfjs op
(matmul, add, activation, slice, clip, mod, …) is a separate GPU dispatch with fixed
overhead. A step whose actual math is trivial still fires ~40–50 dispatches. So:
- Tensor size (256 vs 1M) barely changes cost — dispatch count dominates.
- The wins are all **op-count reductions**: batch the 3 finite-diff evals into one
  `[3N,2]` forward; clip/wrap the whole tensor in one op each.
- The floor for anything in tfjs is "~1 dispatch per op." You cannot get below it.

**The fused advect kernel is the proof.** Collapsing the entire advect stage
(~40 tfjs dispatches) into **ONE** `@compute` dispatch — the whole MLP + physics as
register math — was worth **~40× → 1 on the advect stage**, and it lifted the
particle ceiling from "starts to hurt past ~100k" to a comfortable **1M**. Dispatch
collapse *is* the speedup; element count is nearly free once you're in one kernel.

**The unsolved tension for TRAINING (this is why Phase 2 wants a fused kernel too):**
- We'd love ONE forward over all N particles (the advect forward) to *also* be the
  training forward, with the backward run only over a small sub-sample.
- tfjs can't express it: backward of a taped `[N,…]` forward is **dense O(N)** even
  if the loss touches 256 rows (`dL/dW = Xᵀ·dF`, `dF` is `[N,k]` mostly-zeros — the
  matmul still runs over all N). So sharing the taped forward buys nothing on the
  backward. The efficient tfjs shape is two forwards (forward-only advect + small
  taped train). We now do the forward-only advect **as a kernel**; the train forward
  is still tfjs (Phase 2 fuses it).

---

## Headless GPU verification (the big unlock)

**The old "kernels are un-verifiable in this sandbox" constraint is DEAD.**
`bun-webgpu` (a devDependency) provides a **real Dawn/Metal adapter under Bun** — no
browser, no display — so kernels are tested against an actual GPU right here. Every
kernel change must pass BOTH suites before shipping:

- **`bun tools/kernel_test.ts`** — pure codegen numerics vs an independent **float64
  JS reference** of the exact same semantics (`physicsForward`: normalize → MLP →
  forceMag → friction → clip → floored wrap). 5 configs exercise the codegen paths:
  unrolled/looped × staged/unstaged, asymmetric widths `[16,48]` (guards the
  hardcoded-dim trap), legacy sigmoid MLP (the −0.5 recenter), deep net
  `[64,128,128,64]` (too big to stage → storage-read path). Plus `resetRate=1`
  boundary (every particle respawns in-bounds, zero velocity) and a **1M-particle
  bench**.
- **`bun tools/integration_test.ts`** — the **REAL tfjs webgpu backend under Bun**,
  exercising the tfjs→kernel weight-sync seam that `kernel_test` can't. It shims the
  environment (`window`, a never-matching `GPUBuffer`/`GPUTexture` stand-in so
  `values instanceof GPUBuffer` is correctly `false`, the `requestAdapterInfo`
  shim) and wraps `requestDevice` to flatten a **literal tfjs 4.10 bug**: tfjs
  pushes `['bgra8unorm-storage']` — an *array* — into `requiredFeatures` (browsers
  stringify-coerce it; bun-webgpu's strict validator throws), and it also emits a
  `requiredLimits` map bun-webgpu can't pack (dropped — Dawn's device defaults are
  ample). It then verifies the seam end to end: initial weights, post-Adam-step
  weights, live `alpha` change, and resize grow/shrink — each against a float64
  reference built from the weights **tfjs itself reports** (so packing + variable
  ordering + activation introspection are proven against the LIVE model).

**The GPU is shared with the user's interactive session.** Bench numbers vary
**~±30%** run to run; **serialize** runs (don't fan them out in parallel) and read
the trend, not a single number.

---

## 3. Fused WGSL compute kernels — Phase 1 DONE, Phase 2 next

A hand-written `@compute` shader collapses the whole MLP + physics (+ loss +
gradient) into **one dispatch**; the tiny ops become register math. The 40→1
dispatch collapse *is* the speedup. It also unlocks the thing tfjs can't: **shared
forward + sparse backward** (forward all N once, backprop only the training
sub-sample — you own the backward, so you skip the untrained rows).

### What AlphaGOJS got right (copied) and the traps it hit (avoided)
Reference: `~/git/AlphaGOJS` (its `GPUTrainer` + WGSL kernels) and the memory notes
`alphagojs-*` — a WebGPU AlphaGo/PPO trainer that got big validated speedups
(~14–20×) by doing exactly the dispatch collapse above.
- **Fuse the whole step into one kernel** (forward + backward + optimizer). That's
  the order of magnitude. **Hand-roll the matmul for the tiny net** — WebGPU has
  **no matmul intrinsics** (only subgroups / `vec4` / int8-dot; same ceiling in
  Rust+WASM). Our field is `2→h→h→2`, a handful of MACs/particle, so hand-rolled
  WGSL is trivial and hits none of that wall. **Stay tiny.**
- **Bench headless** — AlphaGOJS ran its real trainer under bun-webgpu; **we now do
  too** (previous section), which is what made Phase 1 shippable.
- **AVOID the hardcoded-dim trap.** AlphaGOJS hardcoded its backward to `D=8`, so
  only `D=8` ever trained (silent, wrong for every other config). Our codegen
  **generates WGSL from live model dims + loud asserts** — a mismatch throws at
  construction. Test more than one size (we test `[16,48]` asymmetric).
- **AVOID "fast-but-plateaus."** AlphaGOJS trained ~14× faster but the task
  plateaued (same skill, sooner). Speed ≠ better art — watch the *output*, not FPS.
- **The backward dominates** (~96% of a step in AlphaGOJS). The advect (forward-only)
  kernel is cheap; the *training* kernel's backward is the cost. Budget Phase 2 there.

### Phase 1 — fused ADVECT kernel (forward-only). **DONE ✅ (shipped + verified).**
One `@compute` pass, thread = particle: read pos/vel from kernel-owned storage
buffers → evaluate the tiny MLP from a packed weights buffer → integrate + friction
+ clip + wrap + fused PCG random reset → write back. No backward, no autograd.
Weights sync tfjs → kernel with no readback (see §1). Renderer binds the buffers
directly. Scales to **1M+ particles**; the advect HUD line is gone.

**Measured (Apple Metal-3, 1M particles, helmholtz `2→32→32→2 ×2`):**
| codegen path | ms/step |
| --- | --- |
| naive loop codegen, wg=64, storage reads | ~18–20 |
| **vec4-tiled fully-unrolled + workgroup staging, wg=256 (shipped)** | **≈7–13** |

The shipped kernel is **vec4-tiled, fully-unrolled** codegen: every activation is a
named register, weights are **staged in workgroup memory as `vec4`** (loads
accumulate 4 outputs at once), `WORKGROUP_SIZE = 256`. It's **weight-load-throughput
bound**, which is why staging + vec4 loads are the win. **Big nets auto-fall back:**
above `UNROLL_MAC_LIMIT = 8192` MACs (e.g. `mlpDeep`, ~33k MACs) the codegen emits
**vec4-tiled loops + storage reads** instead, to keep shader compile time sane.
(The ≈7–13 spread is the shared-GPU ±30%; clean it's ≈7–9.)

**Key files:**
- `src/render/webgpu/advect_wgsl.ts` — **PURE codegen, zero imports**, testable
  headless. Validates + packs the layout (`κ`: the ONLY place shapes are checked),
  then GENERATES the shader from the validated layout so the two cannot disagree.
  Packed segments are **padded to 16B (4-float) boundaries** for the vec4 loads;
  weight layout matches tfjs dense (`[in][out]` row-major, then `[out]` bias, per
  layer, heads back-to-back) so a variable copies in verbatim. **AlphaGOJS D=8 trap
  avoided:** dims arrive as data off the live model + loud asserts.
- `src/render/webgpu/advect.ts` — **AdvectKernel**, the tfjs-coupled class: owns the
  pos/vel/uniform/weights buffers, does the per-frame weight sync (both residency
  paths), runs the one dispatch, and handles live resize (grow appends fresh random
  particles, shrink slices the tail — same semantics the old tfjs path had).

### Phase 2 — fused TRAIN kernel (forward + hand-written backward + Adam). **NEXT.**
**Direction decided: a 2-dispatch trainer.**

- **Dispatch A — one workgroup owns the whole batch** (batch ≤ 1024, so it fits one
  workgroup): a K-step rollout **forward with stored activations**; **batch
  reductions via workgroup barriers** (isotropy needs batch stats, so threads must
  share); then **per-sample backward** — BPTT through integrate/clip/wrap — writing
  each sample's `(activation, delta)` pairs to a **global scratch buffer**.
- **Dispatch B — thread per weight entry:** reduce `dW = Σ over samples×steps of
  a·δ` from the scratch buffer, then the **Adam update in place** (`m`/`v` buffers +
  a step counter).

Once this ships, **weights live GPU-side permanently**: the per-frame tfjs weight
sync in §1 **disappears**, and **tfjs leaves the hot path** entirely (kept only for
prototyping new objectives). This is where **shared-forward / sparse-backward + the
~20×** lands. Parameterize every dim (AlphaGOJS D=8 trap) and lean on the headless
suites.

- **Gradient verification:** a fixture from **tfjs autograd** — `tf.variableGrads`
  on the *exact* `helmholtzChaosLoss` composite — compared to the kernel's gradients
  at **1e-3 relative tolerance**. (Same shape as the existing forward checks: real
  tfjs as the oracle, kernel under test.)
- **Training source becomes a knob:** **random points** (better field coverage — the
  Helmholtz default) vs **particle sub-sample** (tracks the real cloud — better for
  shape pieces). The kernel can read particle states **directly from the advect
  buffers**, so the sub-sample flavor is now cheap to wire.

### Open decisions for whoever picks this up
- Phase-1 hybrid (tfjs trains, kernel advects) is the **current shipped state**; the
  question is only *when* to flip Phase 2 on and drop tfjs from the hot path.
- Batch cap for Dispatch A: 1024 assumes one workgroup owns the batch — revisit if a
  piece wants a bigger training batch (would need a multi-workgroup reduction).
- Watch the *output*, not just the step time (AlphaGOJS "fast-but-plateaus").
- Every kernel change: run **both** headless suites (`kernel_test` + `integration_test`)
  and `tools/smoke.mjs` for a real browser screenshot before pushing.
