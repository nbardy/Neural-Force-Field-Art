# Handoff — Neural Force Field Art (WebGPU) + the fused-kernel plan

Live: **https://nbardy.github.io/Neural-Force-Field-Art/** · `main` @ deploy is pushed to the `gh-pages` branch (see AGENTS.md for build/deploy).

This doc is the pick-up point. It covers (1) where the engine is now, (2) the perf
mental model, (3) the next big step — **fused WGSL compute kernels** — and what
**AlphaGOJS got right** that we're copying, plus the traps it hit that we must avoid.

---

## 1. Where it is now

A tiny neural "force field" is trained with TF.js autograd to move particles; the
whole thing runs on the GPU via **WebGPU**, zero-copy.

**The loop (per frame), three stages shown on the HUD:**
- **learn** — one Adam step on a small batch (`sampleRate`, default 256) of points.
- **advect** — a forward-only pass of the field over ALL `particleCount` particles,
  moving them (integrate → clip → wrap). "Advect" = transport by the flow.
- **render** — `src/render/webgpu/points.ts` reads particle positions STRAIGHT from
  the tfjs tensor via `tensor.dataToGPU()` (a `GPUBuffer` on the webgpu backend) and
  draws instanced WGSL round dots. **True zero-copy — no readback.**

**Deliberate design: "train few, render many."** Training is decoupled from
advection so training cost doesn't scale with particle count. The field-based
"Helmholtz" piece is **first-order** (two direct-vector heads `F = (1-α)·g + α·r`,
NO `tf.grad` inside `forces()`); its chaos+divergence loss shares a single batched
`[3N,2]` forward.

**Key files**
- `src/main.ts` — loop, physics (`physicsForward`), gallery, losses, WebGPU init.
- `src/core/field/helmholtz.ts` — the order↔chaos field.
- `src/core/losses/*` — isotropy / chaos / divergence / spectrum.
- `src/render/webgpu/{microgpu.ts,points.ts}` — WebGPU microlib + renderer.
- `tools/smoke.mjs` — **Puppeteer WebGPU smoke harness** (screenshot + console).
- Dead code kept as a revert path: `src/renderers.ts` (Canvas2D), `src/render/gpuPoints.ts` (WebGL2).

**WebGPU gotchas already handled in `main.ts` (keep them):**
1. `import "@tensorflow/tfjs-backend-webgpu"` — the union package only registers cpu+webgl.
2. `GPUAdapter.prototype.requestAdapterInfo` shim — tfjs 4.10 calls the API Chrome removed (`adapter.info`).
3. All tensor/model creation deferred until after `await tf.ready()` (webgpu init is async).
4. Build with `parcel build --no-scope-hoist` (scope-hoist crashes tfjs at runtime).

---

## 2. The perf mental model (read this before optimizing)

**On the WebGPU backend, cost is DISPATCH-bound, not element-bound.** Every tfjs op
(matmul, add, activation, slice, clip, mod, …) is a separate GPU dispatch with fixed
overhead. A training step whose actual math is trivial still fires ~40–50 dispatches.
So:
- Tensor size (256 vs 100k) barely changes cost — dispatch count dominates.
- The wins so far were all **op-count reductions**: batch the 3 finite-diff evals
  into one `[3N,2]` forward; clip/wrap the whole `[N,2]` tensor in one op each.
- The floor for anything in tfjs is "~1 dispatch per op." You cannot get below it.

**The unsolved tension (this is why we want a fused kernel):**
- We'd love ONE forward over all N particles (the advect forward) to *also* be the
  training forward, with the backward run only over a small sub-sample.
- tfjs can't express it: backward of a taped `[N,…]` forward is **dense O(N)** even
  if the loss touches 256 rows (`dL/dW = Xᵀ·dF`, `dF` is `[N,k]` mostly-zeros — the
  matmul still runs over all N). So sharing the taped forward buys nothing on the
  backward. The efficient tfjs shape is therefore two forwards (forward-only advect +
  small taped train), which is what we run.

---

## 3. Next big step — FUSED WGSL COMPUTE KERNELS

A hand-written `@compute` shader collapses the whole MLP + physics (+ loss + gradient)
into **one dispatch**; the tiny ops become register math. The 40→1 dispatch collapse
*is* the speedup. It also unlocks the thing tfjs can't: **shared forward + sparse
backward** (forward all N once, backprop only the training sub-sample — you own the
backward, so you skip the untrained rows).

### What AlphaGOJS got right (copy this)
Reference: `~/git/AlphaGOJS` (its `GPUTrainer` + WGSL kernels) and the memory notes
`alphagojs-*`. It's a WebGPU AlphaGo/PPO trainer that ran fused WGSL kernels and got
big, validated speedups (~14–20×) by doing exactly the dispatch collapse above.

- **Fuse the whole training step into one kernel** (forward + backward + optimizer),
  not per-op tfjs graphs. That's where the order-of-magnitude comes from.
- **Hand-roll the matmul for the tiny net.** WebGPU has **no matmul intrinsics** —
  only subgroups / `vec4` / int8-dot (AlphaGOJS's "WebGPU ceiling"; same ceiling in
  Rust+WASM). For a big net that's a wall; **our field is 2→h→2**, a handful of MACs
  per particle, so hand-rolled WGSL matmul is trivial and hits none of it. Stay tiny.
- **Bench headless.** AlphaGOJS ran its real GPU trainer headless (bun-webgpu). We
  have `tools/smoke.mjs` (Puppeteer). Verify every kernel change against a real
  adapter — the CI/sandbox here has NO WebGPU adapter, so kernels are un-verifiable
  in this repo's sandbox; they must be tested on a real GPU box.

### Traps AlphaGOJS hit — AVOID
- **Hardcoded dimensions in the hand-written backward.** AlphaGOJS had a backward
  hardcoded to `D=8`, so only D=8 ever trained (silent, wrong for other configs).
  Parameterize dims via uniforms; assert them; test more than one size.
- **The backward dominates (~96% of a step in AlphaGOJS).** Budget accordingly — the
  advect (forward-only) kernel is cheap; the *training* kernel's backward is the cost.
- **Fast-but-plateaus.** AlphaGOJS trained ~14× faster but the task plateaued (same
  skill, reached sooner). Speed ≠ better art — keep an eye on whether the *output*
  actually improves, not just the FPS.

### The plan for THIS repo (staged)
**Phase 1 — fused ADVECT kernel (forward-only). Do first — low risk, big win.**
- One `@compute` pass: thread = particle → read pos/vel from storage buffers,
  evaluate the tiny MLP from a weights buffer, integrate + clip + wrap, write back.
  No backward, no autograd. Scales to **1M+ particles**; deletes the `advect` HUD line.
- Weights flow tfjs → kernel with **no readback**: `field.trainableWeights` →
  `dataToGPU()` → `copyBufferToBuffer` into the kernel's weight buffer once/frame
  (weights are ~KB). tfjs keeps doing `learn`; the kernel does advect + feeds `render`.
- Reuse `src/render/webgpu/microgpu.ts`. This is the original "200k+ points" goal.

**Phase 2 — fused TRAIN kernel (forward + hand-written backward + Adam).**
- The tiny MLP backward in WGSL (matmul-transposes + activation derivatives for the
  fixed architecture) + an Adam step in a buffer. This is where **shared-forward /
  sparse-backward + the 20×** lands. Parameterize every dim (AlphaGOJS D=8 trap).
- Once this exists, the field lives entirely on the GPU; tfjs can be dropped from the
  hot path (or kept only for prototyping new objectives).

### Open decisions for whoever picks this up
- Phase-1 hybrid (tfjs trains, kernel advects) vs. going straight to full-GPU.
- Where the weights live (tfjs-owned + copied each frame, or kernel-owned buffers).
- Whether to keep the "train on random samples" vs. "train on a particle sub-sample"
  (coupled flavor) — see the git log around the decoupling discussion; for shape
  pieces the particle sub-sample tracks the real cloud, for Helmholtz random gives
  better field coverage.
- Verification: every kernel change needs a real-WebGPU box + `tools/smoke.mjs`
  (screenshot + console). Nothing WebGPU is verifiable in this repo's headless CI.
