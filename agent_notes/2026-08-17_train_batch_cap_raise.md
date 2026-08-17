# 2026-08-17 — train batch: raise the cap, stop crashing above it

Parallel thread to the pixel-adversary debugging work. Touches the field
trainer's batch plumbing only; no adversary/pixel-disc semantics changed.

## Goal

1. The train batch ("train B") should be able to go larger — raise the limit.
2. Setting it ABOVE the limit crashed instead of being handled. Make over-cap
   clamp cleanly with a clear warning, then raise the cap.

## The crash — verified mechanism, not a hypothesis

- The slider is `RangeRow label="train B"` in `src/index.tsx`, **`max={4096}`**.
- Its `onChange` → `LoopHandle.setSampleRate`, which was
  `sampleRate = Math.max(1, Math.round(n))` — **clamped the low end only**.
- `FusedTrainer` was constructed in `src/main.ts` with a hard-coded
  **`batchCap: 1024`**.
- Every frame, `trainer.encodeStep(..., { n: sampleRate })` →
  `FusedTrainer.record()` → `if (o.n > this.batchCap) throw`.
- That throw lands inside `tick()`, and `tick` re-arms itself with
  `requestAnimationFrame(tick)` **as its last statement**. So the throw skips
  the re-arm: the animation stops **permanently**, not for one frame.

Net: the top ~3/4 of the batch slider (1025…4096) froze the app.

Three separate numbers disagreed about the same limit: slider max 4096, dock
restore clamp `clamp(samples, 16, 1024)` (`src/index.tsx`), trainer cap 1024,
and `MAX_BATCH` 4096 which nothing ever reached.

Adjacent paths already clamped and were never the crash: the fused adversary
via `adversaryBatchSize(sampleRate, 1024)`, the pixel disc via
`Math.min(sampleRate, 512)`. They still clamp — a train B above 1024 trains the
field on the full batch while the games see their own smaller slices.

## What the real limit is

`MAX_BATCH` was never a physical constraint. Nothing in the WGSL is compiled
against it: fwd/bwd dispatch `ceil(N/TRAIN_WG)` workgroups and `finalize`
reduces partials with a grid-stride loop (`for i = tid; i < wgCount; i += WG`),
so the shaders are batch-agnostic. **Memory** binds: scratch is
`sampleStride * batch * 4` bytes and must fit
`device.limits.maxStorageBufferBindingSize` (128 MiB floor).

Measured on this box (Dawn via bun-webgpu, 128 MiB limit):

| layout | K | resolved cap | scratch at cap |
|---|---|---|---|
| helmholtz (2440 weight floats) | 2 | 16384 (MAX_BATCH) | 84.1 MiB |
| same layout | 16 | **6511** (device-limited) | ≤128 MiB |

So a single constant cannot be right for every piece — the cap is a function of
(device, layout, K).

## What the bigger batches cost

Measured on this box (helmholtz, K=2, 30 steps after warmup, `trainer.step`
only — no advect, no render):

| batch | ms/step |
|---|---|
| 256 (default) | 0.99 |
| 1024 (old cap) | 1.37 |
| 4096 | 3.93 |
| 8192 | 7.83 |
| 16384 | 15.32 |

Roughly linear above ~1k. The new ceiling is reachable and correct, but ~4096
is the practical interactive setting; 16384 spends ~15 ms of the frame on
training alone. Same posture as the particle slider, which also lets you pick a
setting that is slow on purpose. No cap was chosen to hide this — the trainer
reports its real limit and the user picks the point on the curve.

## The compute wall (found in review — memory is NOT the only bound)

An independent review pass flagged, and a direct measurement confirmed, that
the **cover objective is quadratic in the batch**: `bwd` runs a full-batch
nearest-neighbour scan per cover sample (`for m < COVER_SAMPLES { for j < n }`)
— O(COVER_SAMPLES·n) per thread, O(COVER_SAMPLES·n²) aggregate — and
`finalize` adds a single-threaded O(COVER_SAMPLES·n) pass for the diagnostic
`Lcover`. Its scratch stays small (K=1), so `maxBatchForScratch` resolved
Spiral Cover to the **full 16384**.

Measured (Spiral Cover: helmholtz, K=1, COVER_SAMPLES=256), ms/step:

| batch | 256 | 512 | 1024 | 2048 | 4096 | 16384 |
|---|---|---|---|---|---|---|
| ms | 9.4 | 18.4 | 36.8 | 79 | 223 | **1811** |

1.8 s/step is driver-TDR territory, and `device.lost` is handled nowhere in
`src/` — i.e. raising the general cap would have re-created the same
frozen-app symptom this change exists to remove, reachable by dragging the
now-wider slider on one shipped piece.

Fix: the cap resolution now takes the min of memory AND
`maxBatchForLoss(loss)`, which returns `COVER_MAX_BATCH = 1024` (the
previously shipped ceiling, already ~37 ms/step) for `W_COVER ≠ 0` and
`MAX_BATCH` otherwise. Cover pieces therefore keep exactly today's behavior;
every linear-objective piece gets the raised cap. Because the slider reads
`.batchCap`, Spiral Cover's control bounds itself at 1024 automatically, with
no extra plumbing.

## Change

- `train_wgsl.ts`: `MAX_BATCH` 4096 → **16384** (architectural ceiling), plus
  `maxBatchForScratch(field, kSteps, maxBytes)` — the honest per-device bound —
  and `maxBatchForLoss(loss)` / `COVER_MAX_BATCH` for the compute bound.
- `train.ts`: `batchCap` is now **public readonly** and resolved in the
  constructor as `min(requested, MAX_BATCH, memoryCap, lossCap)`. The memory
  side takes the min of `maxStorageBufferBindingSize` and `maxBufferSize` —
  both gate that buffer and neither is guaranteed smaller. Over-asking **warns
  and clamps** (naming which bound bit) instead of surfacing as an opaque
  `createBuffer` OOM at startup. `record()`'s guard stays as a core invariant with a comment
  explaining why it must remain unreachable from any live control.
- `main.ts`: exported `TRAIN_BATCH_MIN/MAX` and
  `resolveTrainBatchSize(requested, cap) → {tag:"ok"|"clamped", …}` (κ; the
  clamp carries provenance rather than silently defaulting). `setSampleRate`
  uses it and warns once per distinct over-cap value (a drag would otherwise
  warn per pointer event). Added `LoopHandle.getMaxSampleRate()`. Trainer now
  built with `batchCap: MAX_BATCH`. Second κ site after construction:
  `sampleRate` is re-canonicalized against the NEW trainer's cap, because a
  gallery switch carries a live sampleRate into a different layout/K.
- `index.tsx`: slider `max` comes from `getMaxSampleRate()`; dock-restore clamp
  uses `TRAIN_BATCH_MIN/MAX` instead of a stale hard-coded 1024; the restore
  path mirrors back the accepted value so the slider can't display a number the
  trainer refused.

Effective user-visible limit: **1024 → 16384** where memory and the objective
allow; a correct device-derived number (e.g. 6511 at K=16) where memory binds;
and **unchanged at 1024** for cover-loss pieces, where compute binds.

## Verification (all run, all pass)

- `bun tools/train_batch_cap_test.ts` — NEW, real GPU. Proves the resolved cap
  equals MAX_BATCH ∧ device limit; that a real step **at** the cap runs, keeps
  every weight finite, and actually moves the weights; that over-cap clamps and
  the clamped n runs; that a K=16 constructor clamps (6511) instead of throwing;
  that a `W_COVER` objective caps at 1024 **even though memory would allow
  16384** while a linear objective on the same layout does not; plus
  deterministic `maxBatchForScratch` arithmetic (floor, ≥1, ≤MAX_BATCH).
- `bun tools/drive_controls_test.ts` — extended with the `resolveTrainBatchSize`
  regression block. Passes.
- `bun tools/window_test.ts` — unchanged trajectory-window equivalence still
  passes (the constructor reorder — kSteps now assigned before batchCap — did
  not regress it).
- `parcel build --no-scope-hoist --no-cache` — clean.

## Open / not done

- **Not deployed.** Prepared for review only.
- The adversary (1024) and pixel-disc (512) caps were left alone: they clamp
  correctly and raising them changes GAN dynamics and memory, which belongs
  with the pixel-adversary thread, not here.
- No headless page smoke run; `tools/smoke.mjs` needs a WebGPU adapter and the
  slider interaction isn't scriptable from it. The GPU test covers the
  mechanism the UI drives.
