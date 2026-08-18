# 2026-08-17 — train throughput: why big batches were slow, and split-K pass B

Follow-on to `2026-08-17_train_batch_cap_raise.md`. That note raised the batch
cap; this one answers "why is the raised cap slow, and why doesn't training
scale like rendering does" — and fixes the dominant cause.

## Measured baseline (this box, Dawn via bun-webgpu)

Helmholtz field, 2440 weight floats, 2304 MACs/particle, K=2. Step split by
dispatching pass A and pass B separately against the trainer's own pipelines:

| batch | full | pass A (fwd+bwd) | pass B (grad reduce) | B share | per sample |
|---|---|---|---|---|---|
| 256 | 0.90 | 0.88 | 0.14 | 13% | 3.52 µs |
| 1024 | 1.51 | 0.83 | 0.61 | 43% | 1.47 µs |
| 4096 | 3.85 | 1.14 | 2.72 | 70% | 0.94 µs |
| 16384 | 15.49 | 4.68 | 10.77 | 70% | 0.95 µs |

Advect, same field: 16384 → 29.6 ns/particle, 262144 → 5.7, 1M → **3.9**.

So ~3.9 ns per rendered particle vs ~945 ns per trained sample ≈ **240×**. An
independent analysis pass decomposed it as **258× = 15.3× (more arithmetic per
element) × 16.9× (lower arithmetic efficiency)**; advect achieves 598 GMAC/s,
pass A 80.5, pass B 17.8 — under 1% of f32 peak.

## Root causes (verified, not guessed)

1. **Pass B had FIXED parallelism.** One thread per weight —
   `ceil(totalFloats/64)` workgroups, ~2.4k threads — **independent of batch**,
   each looping over every sample (`for s = 0..n { for site }`). The batch was
   serial *inside* the thread, so wall time was O(n·sites) no matter how wide
   the GPU. Pass A meanwhile scales its dispatch with n. This is why the ratio
   got worse with batch by construction.
2. **AoS scratch, 5380-byte stride.** Adjacent threads touch addresses
   `sampleStride*4` apart — fully uncoalesced. Advect reads `array<vec2f>` by
   `gid`: 32 lanes, 256 contiguous bytes. Isolated measurement of the identical
   store set: AoS 1.138 ms vs SoA 0.805 ms (1.41×).
3. **Scalar vs vec4.** The trainer emits a scalar MLP loop with dynamically
   indexed `array<f32,32>` ping-pong locals; advect emits unrolled `fma` on
   vec4 with weights staged in workgroup memory. Per field eval: trainer 20.74
   ns, trainer minus stores 13.37, advect 5.34.
4. **60% of pass A's field work is finite-difference probes.** Forward evals
   per sample are K+3 = 5: two physics sites plus **three probes** (`p0`,
   `p0+(hh,0)`, `p0+(0,hh)`) that exist only to difference chaos and div. They
   also lengthen pass B's reduction (`sites = K+3`).

**The headline answer:** rendering is a pure *map* (1M independent threads,
coalesced, no scratch). Training is *map → reduce*, and the reduce was written
with fixed parallelism and a serial batch loop. Handing a GPU 2.4k threads is
why more batch bought nothing.

## What was changed — split-K pass B only

`trainPassBShader` now emits two entry points sharing one explicit bind-group
layout (same idiom pass A already uses for fwd/finalize/bwd):

- `reduce` — one thread per (weight, chunk); each sums its weight over ONE
  slice of the batch into `gradPartials[chunk*totalFloats + t]`. The four
  emitted sample loops changed from `s < ub.n` to `s < sHi` starting at `sLo`;
  the reduction bodies are otherwise byte-identical.
- `applyStep` — sums the chunk partials, adds the game gradients, applies the
  finite guard and Adam.

Host side: `chunks = clamp(ceil(n / 64), 1, gradChunkCap)`, dispatched as
`dispatchWorkgroups(wgB, chunks)`. `gradChunkCap` falls out of a fixed 16 MiB
byte budget for the partials buffer (`totalFloats × chunks × 4`) rather than a
fixed chunk count, so a fat hashgrid layout gets fewer chunks instead of a
huge buffer.

**`extGrad` moved to stage 2.** It is a per-WEIGHT quantity, so adding it in
the reduce would multiply it by the chunk count — invisible at small n, 256×
wrong at n=16384. This is the one bug class split-K introduces that the tfjs
oracle cannot catch (`tools/train_test.ts` has no extGrad), so it has its own
guard.

## Result

| batch | before | after | speedup |
|---|---|---|---|
| 256 | 0.90 | ~0.97 | ~noise (extra pass; batch was already tiny) |
| 1024 | 1.51 | 1.02 | 1.48× |
| 4096 | 3.85 | 2.24 | 1.72× |
| 8192 | 7.83 | 4.82 | 1.62× |
| 16384 | 15.49 | 10.35 | 1.50× |

Pass B itself went ~10.8 → ~5.6 ms at 16384 (≈1.9×), matching the 2.08×
predicted by the standalone prototype. Run-to-run variance is ~±8%, and the
first run of any process is warm-up — discard it.

## Verification

- `tools/train_test.ts` — **gradient parity held with NO fixture change**:
  worst cos=1.0000000, relMax 1.80e-5 (was 1.82e-5), Adam maxΔw 3.28e-7. The
  concern that split-K would force a tolerance loosening did not materialize;
  the oracle was already relative, not bit-exact. Its N=256 fixture means the
  oracle runs at **4 chunks**, so chunk-boundary coverage is validated there.
- `tools/train_batch_cap_test.ts` §5 (NEW) — zero field loss ⇒ internal g is
  literally 0, so `grads` must equal `extGrad` EXACTLY. Asserted at n=64
  (1 chunk) and n=16384 (chunk cap); both max|Δ| = 0.
- `train_types_test` (5 fixtures), `window_test`, `integration_test`,
  `train_wta_test`, `train_wta_hashgrid_test`, `field_loss_routing_test`,
  `adversary_wire_test`, `pixel_disc_test`, `drive_controls_test` — all pass.
- `parcel build --no-scope-hoist --no-cache` — clean.

Also fixed in passing: `destroy()` omitted `partialsBuf` and `partDummy` (both
already owned). Added alongside `gradPartialsBuf`.

## Deliberately NOT done

Each of these is real and measured, and each is a much larger or
semantics-changing change than split-K. Left for an explicit decision:

- **vec4 + workgroup-staged weights in the trainer's evaluator** — 2.50×
  headroom on pass A's evaluator (~1.5-2 ms). Requires rewriting
  `emitFwdStore`/`emitBwdStore`; the layer-0 `du` tail reads a column and is
  not vec4-friendly, and the same head codegen feeds `adversary_wgsl.ts`.
  Biggest remaining win; now that pass B is fixed, pass A dominates (4.7 of
  10.35 ms).
- **SoA scratch** — 1.41× on the store pattern, but `sampleStride` stops being
  a codegen constant and every emitter changes, `adversary_wgsl.ts` included.
  Smaller than the above; should not go first.
- **Fusing dW into pass A's backward and deleting pass B + the 84 MiB
  scratch** — would also make the batch cap memory-unbounded, but WGSL has no
  f32 workgroup atomics and the workgroup budget scales with `totalFloats`, so
  hashgrid layouts would not fit. High risk.
- **Replacing the finite-difference probes with forward-mode JVP**
  (`src/render/webgpu/ad/` already exists) — up to ~40% of both passes, but it
  **changes the loss**. Shipped pieces would look different. Not an
  optimization; a design decision.
- **Cover loss O(COVER_SAMPLES·n²)** — still capped at 1024 by
  `maxBatchForLoss`. Lifting it needs spatial binning for the nearest-neighbour
  scan.
