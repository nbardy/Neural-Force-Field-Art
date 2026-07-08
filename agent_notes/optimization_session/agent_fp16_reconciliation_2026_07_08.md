# FP16 Reconciliation

Date: 2026-07-08

Scope: reconcile whether CLIP fp16/f16 gave a big jump, using current repo
evidence only. This note is documentation-only; no source code was changed.

## Short Answer

We did implement a gated CLIP f16 weights-only path in commit `7548a4f`:

```text
7548a4f Gate f16 CLIP weight experiment
```

It did not produce a promotable big jump. It halves CLIP weight storage and
sometimes gives a small timing win, but the f32-vs-f16 input-gradient cosine
fails the planned correctness gate. The "big jump" memory is probably from
batch-major CLIP, the older force-field advect f16 path, or view-schedule
changes, not from promoted CLIP f16.

## Evidence Inspected

Command:

```bash
git log --oneline --decorate -12
```

Relevant commits:

```text
4d26652 Gate depthwise spatial backward CLIP fork
3ffce77 Expose N-of-K view sampling controls
7548a4f Gate f16 CLIP weight experiment
05bd8e1 Record CLIP tracing status
f64fbcd Expose grid9 CLIP layout in 3D UI
5536ad5 Gate grid9 closeup CLIP layout
b7eeeb7 Gate residual backward pointwise fusion
287ef70 Record CLIP optimization fork plans
```

Command:

```bash
ls -lh models/mobileclip_s0 | sed -n '1,80p'
```

Relevant current assets:

```text
weights.bin             43M
weights_f16.bin         22M
weights_train.bin       82M
weights_train_f16.bin   41M
```

So the generated f16 assets exist locally and are roughly half-size.

## v02 F16 Result

Primary files:

- `experiments/clip_forks/v02_f16_weights/README.md`
- `experiments/clip_forks/v02_f16_weights/results/2026-07-08.md`
- `tools/clip/pack_f16_weights.ts`
- `tools/clip/f16_compare.ts`
- `tools/clip/dispatch_profile.ts`
- `tools/clip/batch_major_train_bench.ts`
- `tools/splat3d/step_bench.ts`

The v02 result says this was a weights-only precision fork:

- f16 weights
- f32 activations
- f32 gradients
- f32 workgroup tiles
- f32 reductions and accumulators

Recorded asset gate:

```bash
bun tools/clip/pack_f16_weights.ts
```

Recorded output:

```text
weights.bin -> weights_f16.bin: 11353536 scalars, 43.3 MB -> 21.7 MB (0.500x)
weights_train.bin -> weights_train_f16.bin: 21515712 scalars, 82.1 MB -> 41.0 MB (0.500x)
```

Fresh correctness rerun:

```bash
STRICT=0 bun tools/clip/f16_compare.ts
```

Output:

```text
adapter: apple metal-3
embedding: cos=0.99999559 relLinf=1.096e-3 maxDiff=6.807e-4 norm(f16/f32)=1.327e+0/1.326e+0
inputGrad : cos=0.97493807 relLinf=3.933e-1 maxDiff=3.462e-4 norm(f16/f32)=1.601e-2/1.567e-2
```

The planned gate was:

```text
embedding cosine >= 0.9995
input-gradient cosine >= 0.995
```

Embedding passes. Input gradient fails.

## Timing Evidence

The committed v02 timing note recorded f16 as slower in the stable isolated
timestamp run:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
PRECISION=f16 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=5 WARMUP=3 bun tools/clip/dispatch_profile.ts
```

Recorded result:

| Precision | Total isolated median sum |
| --- | ---: |
| f32 | `67.109 ms` |
| f16 weights | `71.107 ms` |

Fresh shorter rerun during this reconciliation:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
PRECISION=f16 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Output summary:

| Precision | Total isolated median sum |
| --- | ---: |
| f32 | `102.629 ms` |
| f16 weights | `101.384 ms` |

That is only about a `1.2%` f16 win in this short current run, not a big jump.

Fresh batch-major B=3 train rerun:

```bash
BATCH=3 RUNS=3 WARMUP=2 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 bun tools/clip/batch_major_train_bench.ts
PRECISION=f16 BATCH=3 RUNS=3 WARMUP=2 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 bun tools/clip/batch_major_train_bench.ts
```

Output summary:

| Precision | Separate | Batch-major |
| --- | ---: | ---: |
| f32 | `58.46 ms/batch` | `41.57 ms/batch` |
| f16 weights | `48.65 ms/batch` | `38.93 ms/batch` |

This shows a small B=3 batch-major f16 speed win in the current tree. However,
the bench's lane parity is same-precision parity; it does not prove f16 matches
the f32 gradient target. The `f16_compare.ts` gate above is still the relevant
cross-precision correctness check, and it fails on `dL/dpixels`.

Fresh integrated 3D short rerun:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=3 WARMUP=1 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_PRECISION=f16 CLIP_BATCH=3 VIEWS=3 RUNS=3 WARMUP=1 bun tools/splat3d/step_bench.ts
```

Output summary:

| Precision | Normal step avg | Profile total | CLIP batch |
| --- | ---: | ---: | ---: |
| f32 | `54.23 ms` | `58.52 ms` | `43.58 ms` |
| f16 weights | `51.61 ms` | `50.27 ms` | `39.19 ms` |

This current short run is encouraging for speed, but it is still not a 2x-4x
jump and it remains blocked by the input-gradient quality gate.

The committed v02 integrated timing went the other way:

| Precision | Profile total | CLIP batch |
| --- | ---: | ---: |
| f32 | `54.98 ms` | `42.27 ms` |
| f16 weights | `59.90 ms` | `47.38 ms` |

So f16 timing is noisy/small, while the gradient mismatch is stable.

## What Was Probably Confused With FP16

1. Batch-major CLIP, not f16:

`docs/CLIP_BATCHING_NOTES.md` records the real optimizer-relevant batch-major
forward+backward win:

| Batch | Separate forward+backward | Batch-major forward+backward |
| ---: | ---: | ---: |
| 3 | `102.55 ms/batch` | `49.48 ms/batch` |
| 9 | `389.66 ms/batch` | `134.63 ms/batch` |

That is where the "B=9 is about 2.9x faster" line came from. It is batching and
kernel scheduling, not fp16.

2. Batch-major forward-only, not full training:

The same notes record train-plan forward-only B=9 as:

| Plan | Batch | Separate forwards | Batch-major forward |
| --- | ---: | ---: | ---: |
| `plan_train.json` | 9 | `244.79 ms/batch` | `62.30 ms/batch` |

That is close to a 4x forward-only result, but the splat optimizer needs
forward plus backward.

3. Older force-field advect f16 path, not CLIP:

`agent_notes/optimization_session/agent_f16_reality_check.md` says the older
force-field advect shader had a real f16 fast path, with a recorded improvement
around `9.3 -> 6.5 ms/step @ 1M` on Apple Metal. That was not the MobileCLIP
vision train loop.

4. View scheduling and grid/contact-sheet CLIP, not f16:

`v01_grid9_close2` and `v06_view_sampling` reduce or reshape how many CLIP
views are optimized per step. These can move wall time substantially, but they
change scheduling/signal use rather than making one same-resolution CLIP pass
2x-4x faster.

5. Spatial backward specialization, not f16:

`v07_spatial_bwd_depthwise4` improved integrated step median from `53.12 ms` to
`49.96 ms` in its recorded matrix. Useful, but a narrow shader fork, not fp16.

## Decision

The repo evidence does not support "we already did fp16 and got a big jump" for
CLIP. The accurate statement is:

- CLIP f16 weights were implemented and gated.
- Weight files halve cleanly.
- Final embedding similarity is excellent.
- Input-gradient similarity fails the planned gate.
- Timing is mixed: sometimes small win, sometimes small loss.
- No current evidence shows a promotable 2x-4x CLIP speedup from all-weight f16.

Do not promote `CLIP_PRECISION=f16` as default until the gradient gate is fixed
or intentionally replaced with an optimizer-level quality gate.

## Plausible Selective Precision Lanes

1. Pointwise-only f16 weights with f32 accumulation.
   The hot CLIP groups include `pw_bwd`, `pw`, and `pw+gelu`. A two-buffer or
   per-family weight layout could test f16 only where weight bandwidth matters,
   while leaving stem, attention, SE, and head weights in f32.

2. Selective interior activation/gradient slot f16.
   Keep image input, text embedding, logits/loss reductions, final embedding,
   input gradient, and accumulation destinations in f32. Try f16 only for large
   interior saved activations/grad slots after adding per-slot dtype metadata.

3. Depthwise/spatial f16 read paths only after profiling.
   The current all-weight f16 path made some `conv` timings worse in short
   traces. Do not blanket-convert spatial kernels again without per-kernel
   timing or GPU-counter evidence.

4. Mixed precision inside new pointwise tiling forks.
   If a pointwise rewrite stages W or X in workgroup memory, f16 storage plus
   f32 accumulation may become more useful than the current simple f16 load and
   cast path.

5. Keep text fp16 separate from vision CLIP.
   The text encoder already uses fp16 ONNX/WASM outside the per-step vision hot
   loop. It is not the main optimization target for 3D splat steps.

Next f16 action, if any: make a selective pointwise-weight f16 fork with an
explicit f32-vs-selective-f16 `dL/dpixels` cosine gate before running long
optimizer screenshots.
