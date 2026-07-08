# CLIP Batching Notes

Goal: test whether the fused MobileCLIP vision path can become faster for
multi-view batches before wiring anything into the 3D splat optimizer.

## Iteration 1 - Replicated Activations, Shared Weights

Implemented:

- `src/clip/vision_batch.ts`
- `tools/clip/batch_bench.ts`

This is intentionally an isolation harness. It keeps one shared weights buffer
and one compiled pipeline set, but gives each image lane its own activation,
gradient, and text buffers. It compares three schedules:

- `separate`: one full CLIP submit per image lane.
- `lane-major`: one submit; run full CLIP for lane 0, then full CLIP for lane 1.
- `step-major`: one submit; run CLIP step 0 for every lane, then step 1 for
  every lane.

Commands run on Apple `metal-3` via `bun-webgpu`:

```bash
BATCH=2 MODE=forward RUNS=1 WARMUP=1 bun tools/clip/batch_bench.ts
BATCH=2 MODE=backward RUNS=1 WARMUP=1 bun tools/clip/batch_bench.ts
BATCH=2 MODE=backward RUNS=4 WARMUP=3 bun tools/clip/batch_bench.ts
BATCH=3 MODE=backward RUNS=3 WARMUP=2 bun tools/clip/batch_bench.ts
BATCH=9 MODE=backward RUNS=1 WARMUP=1 bun tools/clip/batch_bench.ts
BATCH=9 MODE=forward RUNS=2 WARMUP=2 bun tools/clip/batch_bench.ts
```

Representative results:

| batch | mode | separate | lane-major | step-major |
| --- | --- | ---: | ---: | ---: |
| 2 | backward | 27.89 ms/batch | 41.67 ms/batch | 34.91 ms/batch |
| 3 | backward | 43.45 ms/batch | 49.83 ms/batch | 51.70 ms/batch |
| 9 | backward | 124.20 ms/batch | 154.17 ms/batch | 164.19 ms/batch |
| 9 | forward | 52.08 ms/batch | 107.60 ms/batch | 71.72 ms/batch |

The batch schedules reduce CPU encode/submit time, especially `step-major`, but
they do not reduce GPU wall time. This means the current replicated batcher is
not the production speed win for 3D splats.

Current conclusion:

- Default the optimizer to N-of-K random views first; that cuts CLIP work
  directly.
- Do not wire the replicated batcher into the app unless profiling shows CPU
  command encoding is the bottleneck on a target machine.
- True sublinear batching requires a real batch dimension in the WGSL kernels,
  not just more per-lane dispatches using shared weights.

## How Sublinear Could Actually Happen

The realistic path is a second codegen lane:

1. Add batch-major activation slots sized `batch * slotFloats`.
2. Add `workgroups.z = batch` to every simple kernel so dispatch count becomes
   O(CLIP steps), not O(batch * CLIP steps).
3. Offset all activation/text reads by `batchIndex * slotSize`.
4. For pointwise forward/backward, test a B=2 or B=3 shared-W tile:
   stage W once, process multiple image lanes for the same `(pixel tile, cout
   tile)` before moving on. Current shared memory is about 4 KB for X and 4 KB
   for W. B=2 fits comfortably; B=3 is near the 16 KB adapter limit; B=4 likely
   needs smaller tiles.
5. Keep spatial conv and attention simpler at first: batch-z parallelism should
   improve occupancy and launch amortization, even if memory traffic remains
   roughly linear.

## Iteration 2 - Batch-Major Forward

Implemented:

- `src/clip/vision_batch_wgsl.ts`
- `BatchMajorVisionEncoder` in `src/clip/vision_batch.ts`
- `tools/clip/batch_major_forward_bench.ts`

This is the first true batch-dimension fork. It reuses the existing proven
forward WGSL emitters, rewrites slot-buffer indexing to add a lane base offset,
allocates activation slots as `[batch][slotFloats]`, and dispatches each CLIP
step once with `workgroups.z = batch`.

Commands run:

```bash
BATCH=2 RUNS=1 WARMUP=1 bun tools/clip/batch_major_forward_bench.ts
BATCH=2 RUNS=5 WARMUP=3 bun tools/clip/batch_major_forward_bench.ts
BATCH=3 RUNS=5 WARMUP=5 bun tools/clip/batch_major_forward_bench.ts
BATCH=9 RUNS=3 WARMUP=5 bun tools/clip/batch_major_forward_bench.ts
PLAN=plan_train.json BATCH=3 RUNS=5 WARMUP=5 bun tools/clip/batch_major_forward_bench.ts
PLAN=plan_train.json BATCH=9 RUNS=3 WARMUP=5 bun tools/clip/batch_major_forward_bench.ts
```

Verification:

- `plan.json`: parity passed exactly for B=2, B=3, and B=9.
- `plan_train.json`: parity passed exactly for B=3 and B=9.
- Parity means lane-by-lane cosine `1.000000` and `relLinf=0.00e+0` versus the
  original single-image fused encoder.

Representative results from the cleaned benchmark layout:

| plan | batch | separate forwards | batch-major forward |
| --- | ---: | ---: | ---: |
| `plan.json` | 3 | 22.50 ms/batch | 15.21 ms/batch |
| `plan.json` | 9 | 205.23 ms/batch | 106.71 ms/batch |
| `plan_train.json` | 3 | 25.12 ms/batch | 34.59 ms/batch |
| `plan_train.json` | 9 | 244.79 ms/batch | 62.30 ms/batch |

Takeaways:

- The true batch dimension works. We now have a real fork of the fused CLIP
  forward path that runs all image lanes inside one dispatch list.
- It can be sublinear for larger B. B=9 train-plan forward was about 4x faster
  than repeated single forwards in this run.
- B=3 is not reliably better on the train plan. The likely default optimizer
  setting (`3 of 9` random views) still needs either plain repeated CLIP or a
  more specialized shared-W pointwise kernel.
- This does not solve optimizer batching yet, because the splat optimizer needs
  forward+backward. Backward needs the same batch-major treatment before this
  can replace `VisionTrainer`.

## Iteration 3 - Batch-Major Forward+Backward

Implemented:

- `batchTrainDispatches()` in `src/clip/vision_batch_wgsl.ts`
- `BatchMajorVisionTrainer` in `src/clip/vision_batch.ts`
- `tools/clip/batch_major_train_bench.ts`

This applies the same z-batch transform to the full train dispatch list:
forward, loss head, and backward. The loss head's text embedding is also
batched as `[batch][textDim]`, so each lane can optimize against a different
view prompt.

Commands run:

```bash
BATCH=2 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
BATCH=3 RUNS=2 WARMUP=3 bun tools/clip/batch_major_train_bench.ts
BATCH=9 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
BATCH=9 RUNS=2 WARMUP=2 bun tools/clip/batch_major_train_bench.ts
```

Verification:

- Gradient parity passed exactly for B=2, B=3, and B=9.
- Parity compares full `dL/dpixels` for every lane against the original
  single-image `VisionTrainer`.
- Every checked lane had cosine `1.000000` and `relLinf=0.00e+0`.

Representative results:

| batch | separate forward+backward | batch-major forward+backward |
| ---: | ---: | ---: |
| 3 | 102.55 ms/batch | 49.48 ms/batch |
| 9 | 389.66 ms/batch | 134.63 ms/batch |

Takeaways:

- The actual optimizer-relevant batched CLIP path now exists and verifies.
- B=3 is already about 2x faster in the warmed run, which matches the likely
  `3 of 9` random-view optimizer setting.
- B=9 is about 2.9x faster in the warmed run, so all-view optimization becomes
  less absurd when needed for evaluation or screenshots.
- This still is not the final integrated optimizer path. The 3D rasterizer must
  render selected views into batched input slots, and then scatter batched image
  gradients back into per-view splat gradients.

## Iteration 4 - Shared-W Pointwise Microkernel

Implemented:

- `src/clip/vision_batch_pointwise.ts`
- `tools/clip/pointwise_batch_bench.ts`

This tests the specific "can we share memory bandwidth for the weights?"
question. The existing batch-major CLIP path runs one pointwise workgroup per
batch lane, so every lane loads the same W tile. The new microkernel moves the
batch lane into `local_invocation_id.z`: one workgroup stages W once, while
each lane stages its own X tile. B=2 and B=3 fit under the 16 KB workgroup
memory limit; B=3 uses exactly 12 KB X + 4 KB W.

Commands run:

```bash
BATCH=2 STEP_INDEX=57 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCH=3 STEP_INDEX=57 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCH=2 STEP_INDEX=8 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCH=3 STEP_INDEX=8 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCH=2 STEP_INDEX=59 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCH=3 STEP_INDEX=59 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
```

Verification:

- Shared-W output matched z-batch output exactly for all tested shapes:
  `relLinf=0.00e+0`.

Representative results:

| step | shape | B | z-batch | shared-W |
| ---: | --- | ---: | ---: | ---: |
| 57 | `256->768 @16x16`, fc1 | 2 | 0.563 ms | 0.500 ms |
| 57 | `256->768 @16x16`, fc1 | 3 | 1.026 ms | 1.253 ms |
| 8 | `64->192 @64x64`, fc1 | 2 | 0.541 ms | 0.510 ms |
| 8 | `64->192 @64x64`, fc1 | 3 | 0.565 ms | 0.626 ms |
| 59 | `768->256 @16x16`, fc2 residual | 2 | 0.219 ms | 0.352 ms |
| 59 | `768->256 @16x16`, fc2 residual | 3 | 0.726 ms | 0.719 ms |

Takeaways:

- The shared-W idea is valid and verifies exactly.
- It is not a blanket win. B=2 expansion layers improved modestly, B=3 often
  lost, and contraction/residual layers were mixed.
- The likely reason is occupancy pressure: B=3 raises workgroup invocations to
  192 and uses the full 16 KB workgroup-memory budget. Saved W bandwidth can be
  offset by lower occupancy or scheduling pressure.
- Integration should be selective. A full CLIP profile should identify which
  pointwise steps are bottlenecks before replacing only the shapes that win.

## Iteration 4b - Shared-W Pointwise Matrix Runner

Implemented:

- `tools/clip/pointwise_batch_matrix.ts`

This runs the shared-W microbench sequentially over multiple pointwise steps and
batch sizes, then reports medians and win/flat/loss verdicts. It is deliberately
process-based so benchmark rows do not run concurrently and contaminate GPU
timing.

Commands run:

```bash
BATCHES=3 STEPS=8,10,57,59,111,113,115,117 TRIALS=3 RUNS=20 WARMUP=5 bun tools/clip/pointwise_batch_matrix.ts
BATCHES=2 STEPS=8,10,57,59,111,113,115,117 TRIALS=3 RUNS=20 WARMUP=5 bun tools/clip/pointwise_batch_matrix.ts
```

B=3 median results:

| step | shape | verdict | ratio range |
| ---: | --- | --- | --- |
| 8 | `64->192 @64x64` | win, `0.800x` | `0.763-0.998` |
| 10 | `192->64 @64x64` residual | win, `0.824x` | `0.735-0.868` |
| 57 | `256->768 @16x16` | win, `0.947x` | `0.718-1.023` |
| 59 | `768->256 @16x16` residual | flat, `1.010x` | `0.823-1.067` |
| 111 | `512->1536 @8x8` | win, `0.774x` | `0.707-1.005` |
| 113 | `512->512 @8x8` residual | loss, `1.108x` | `0.725-1.453` |
| 115 | `512->1536 @8x8` | win, `0.883x` | `0.792-0.944` |
| 117 | `1536->512 @8x8` residual | loss, `1.100x` | `1.100-1.205` |

B=2 median results:

| step | shape | verdict | ratio range |
| ---: | --- | --- | --- |
| 8 | `64->192 @64x64` | win, `0.838x` | `0.723-0.866` |
| 10 | `192->64 @64x64` residual | flat, `0.993x` | `0.741-1.136` |
| 57 | `256->768 @16x16` | win, `0.884x` | `0.802-1.067` |
| 59 | `768->256 @16x16` residual | loss, `1.225x` | `0.915-1.315` |
| 111 | `512->1536 @8x8` | flat, `0.977x` | `0.605-1.276` |
| 113 | `512->512 @8x8` residual | loss, `1.442x` | `1.130-1.886` |
| 115 | `512->1536 @8x8` | loss, `1.272x` | `0.944-1.274` |
| 117 | `1536->512 @8x8` residual | loss, `1.102x` | `1.076-1.198` |

Takeaways:

- The exact shared-W kernel still verifies, but timing is shape-specific and
  noisy enough that single-row microbench results are not a promotion gate.
- Do not integrate shared-W globally.
- Candidate full-plan allowlists are B=3 steps `8`, `10`, `115`, and maybe
  `111`; B=2 steps `8` and `57` are the cleanest if we test a `2 + 1` schedule.
- Promotion still requires a full `BatchMajorVisionTrainer` wall-time win, not
  only compact-kernel microbench wins.

## Next Five Iterations

1. **Production speed win:** add `views/step` N-of-K to the 3D page, default 3
   of 9, and avoid refreshing the 9-view grid every optimizer step.
2. **3D optimizer integration:** render N selected views into batched CLIP input
   lanes, run `BatchMajorVisionTrainer`, then apply each lane's image gradient
   to the matching camera view.
3. **Selective pointwise integration:** use the shared-W pointwise kernel only
   for shape/batch combinations that win in a full-CLIP profile.
4. **Train-plan memory tightening:** allocate only slots touched by the active
   forward/backward batch path, or split forward-only and train-plan batches so
   benchmark memory pressure is easier to reason about.
5. **Integrate only if proven:** wire the fastest measured batch path behind a
   toggle; keep N-of-K random views as the default if CLIP batching is not a
   clear GPU win.
