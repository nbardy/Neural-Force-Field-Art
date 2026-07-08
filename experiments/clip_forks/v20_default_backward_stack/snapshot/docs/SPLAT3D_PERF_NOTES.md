# 3D Splat Performance Notes

Last updated: 2026-07-08

## Current Timing Surface

The 3D page now samples a split-submit wall-time profile every 30 optimizer
steps. The overlay reports:

- raster forward and backward
- CLIP forward and backward
- Adam step
- clear/display overhead
- sampled optimizer-step total

This is intentionally labeled as sampled wall time. It is good enough to decide
whether CLIP or raster is the current bottleneck, but it is not exact GPU
timestamp attribution for the normal single-submit `step()` path. For CLIP
kernel ranking, `tools/clip/dispatch_profile.ts` now has an opt-in
`TIMESTAMP=1` path that uses WebGPU timestamp queries when the adapter supports
them.

## Measurements

Default 4096-splat scene, 256px renders, 9 cameras, Bun/WebGPU on 2026-07-08:

| Views / Step | Normal Step Avg | Split Profile Total | Raster Fwd | Raster Bwd | CLIP Fwd | CLIP Bwd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 / 9 | 205.26 ms | 228.20 ms | 8.16 ms | 33.38 ms | 77.77 ms | 102.81 ms |
| 5 / 9 | 122.02 ms | 134.83 ms | 5.38 ms | 19.85 ms | 39.77 ms | 57.79 ms |
| 3 / 9 | 69.94 ms | 75.57 ms | 2.32 ms | 11.20 ms | 24.07 ms | 34.96 ms |

Takeaway: random N-of-K view sampling is a real wall-clock lever. The 3/9 path
is roughly 2.9x faster than 9/9 while preserving stochastic coverage of all
camera prompts over multiple steps. The implementation uses shuffled epochs, so
3/9 still randomizes order but covers all 9 views every 3 optimizer steps. CLIP
remains around 78-79% of sampled wall time, so deeper speedups should attack
CLIP calls/batching/precision before large raster rewrites.

Tried and reverted: exact circle-vs-tile support pruning in emit/binning. It was
mathematically safe, but on this workload it made raster forward/backward flat
to slightly slower, likely because the extra per-tile math outweighed the small
bin-count reduction at 256px / 4096 splats.

## Four-Agent Pass: Main Read

### CLIP

The most likely CLIP bottleneck is still pointwise matmul. Static work is
dominated by forward pointwise and `pw_bwd`, so another clean 4x from kernel
tuning alone is unlikely unless we change precision, reduce CLIP calls, or make
a larger matmul strategy shift.

Next measurements:

- Per-dispatch timing grouped by label/shape for B=1, B=3, and B=9.
- Pointwise forward/backward timing for the repeated 256->768 and 768->256
  blocks.
- Stem `spatial_bwd`, especially the first 256x256 3<-64 kernel.
- Aggregate GELU forward/backward share.
- Attention backward only if timestamps show it above roughly 5-10%.

Most promising CLIP experiments:

- Use batch-major CLIP for multi-view training.
- Try f16 weights/activations with f32 reductions behind feature gating.
- Stage weights in `spatial_bwd`.
- Fuse pointwise/GELU/residual blocks if dispatch and scratch traffic show up.
- Use per-dispatch timestamp-query profiling before choosing the next CLIP
  rewrite.

### Raster

The current rasterizer already has two key FasterGS-style ideas: per-tile sorted
IDs are saved in forward, and backward reuses the saved order plus tile stop
counts. The best next raster experiments are smaller and directly applicable:

- Alias raster image/grad buffers with CLIP input/grad buffers to remove two
  full-image copies per view.
- Stage derived splat parameters in workgroup chunks for forward/backward.
- Add tile overflow telemetry before tuning caps or thresholds.
- Benchmark workgroup-local gradient reductions before replacing global fixed
  point atomics.

Do not port the dynamic viewer bucket-sort path into training. It is useful for
preview, but the training backward path needs the saved per-tile order and stop
state.

## Next Decision Rule

Use the overlay first:

- If CLIP dominates, land batch-major training in the 3D optimizer before deep
  shader rewrites.
- If raster dominates, start with buffer aliasing, overflow telemetry, and
  workgroup staging. Re-test circle-tile pruning only at higher resolution or
  larger splat counts.
- If Adam/display/clear are visible but small, leave them alone until CLIP and
  raster are under control.

## Batch-Major CLIP Integration Attempt

The first app-level batch integration is now behind an opt-in 3D page control:

- `single CLIP`
- `batch CLIP x3`
- `batch CLIP x9`

The implementation is conservative. It renders selected views into CLIP batch
lanes, runs one batch-major CLIP train pass, then re-renders each view before
applying that lane's image gradient. This preserves current raster correctness
because the rasterizer still owns only one view's tile bins and sorted order at a
time.

Headless integrated benchmark:

```bash
CLIP_BATCH=1 VIEWS=3 RUNS=8 WARMUP=4 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=3 RUNS=8 WARMUP=4 bun tools/splat3d/step_bench.ts
CLIP_BATCH=1 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
CLIP_BATCH=9 VIEWS=9 RUNS=3 WARMUP=2 bun tools/splat3d/step_bench.ts
```

Representative measurements on Apple `metal-3`:

| Views / Step | CLIP Batch | Normal Step Avg | Split Profile Total | Notes |
| --- | ---: | ---: | ---: | --- |
| 3 / 9 | 1 | 104.32 ms | 129.47 ms | sequential run |
| 3 / 9 | 3 | 93.41 ms | 121.43 ms | modest, noisy win |
| 9 / 9 | 1 | 381.18 ms | 412.02 ms | same-session matrix |
| 9 / 9 | 3 | 292.10 ms | 314.28 ms | clear all-view win |
| 9 / 9 | 9 | 299.93 ms | 330.83 ms | not better than x3 here |

Takeaway: keep batch CLIP as an ablation/screenshot toggle, but do not make it
the default yet. The replayed raster forward eats enough of the CLIP batching win
that the next promotion step should be per-lane raster state or direct
raster/CLIP buffer binding.

## Dispatch Profile Snapshot

`tools/clip/dispatch_profile.ts` now provides warmed isolated dispatch timings:

```bash
MODE=train BATCH=1 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
CSV=1 MODE=train BATCH=3 bun tools/clip/dispatch_profile.ts > /tmp/clip_b3.csv
```

Without `TIMESTAMP=1`, this remains a warmed split-submit wall-time ranking.
With `TIMESTAMP=1`, it resolves one begin/end timestamp pair around each
isolated compute dispatch. First warmed split-submit results:

| Batch | Dominant Groups |
| ---: | --- |
| 1 | `pw` 19.9%, `pw_bwd` 18.7%, `spatial_bwd` 17.1%, `conv` 14.3% |
| 3 | `pw` 20.8%, `spatial_bwd` 19.6%, `pw_bwd` 19.5%, `conv` 14.7% |

The profiler changes the priority slightly: `spatial_bwd` is at least as
important as pointwise in B=3, while attention backward is too small for the
first wave.

Timestamp-query smoke on the current promoted B=3 CLIP settings
(`STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1`, `RUNS=1`, `WARMUP=1`) reported:

| Group | Timestamp Isolated Sum |
| --- | ---: |
| `pw_bwd` | `42.01 ms` / `26.2%` |
| `spatial_bwd` | `30.08 ms` / `18.8%` |
| `conv` | `25.10 ms` / `15.7%` |
| `pw+gelu` | `23.92 ms` / `14.9%` |
| `pw` | `23.20 ms` / `14.5%` |
| `attn_core_bwd` | `4.33 ms` / `2.7%` |

Total isolated timestamp sum was `160.30 ms`. The main conclusion is unchanged:
optimize pointwise backward/forward, spatial backward, and conv-family kernels
before spending serious time on attention backward.

## Pointwise Static Roofline

`tools/clip/pointwise_report.ts` is a CPU-only report that reads
`plan_train.json` and explains the pointwise workload without requiring a GPU
run:

```bash
BATCH=3 TOP=14 OUT=/tmp/pointwise_report.md bun tools/clip/pointwise_report.ts
```

For the current `3x256x256` MobileCLIP train plan at batch 3:

| Metric | Batch 3 |
| --- | ---: |
| forward pointwise dispatches | 48 |
| backward `pw_bwd` dispatches | 48 |
| forward pointwise+GELU candidates | 24 |
| pointwise FLOPs | 26.575 GFLOP |
| lower-bound staged global traffic | 3445.13 MiB |
| pointwise workgroups | 51648 |

Same-session timestamp cross-check:

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

| Group | Timestamp Sum | Share |
| --- | ---: | ---: |
| `pw_bwd` | 34.34 ms | 26.3% |
| `pw` | 18.02 ms | 13.8% |
| `pw+gelu` | 14.94 ms | 11.5% |
| pointwise family | 67.31 ms | 51.6% |
| `spatial_bwd` | 26.87 ms | 20.6% |

This reconciles the earlier f16 question: the existing f16 path is weights-only
storage with f32 math/activations. It halves weight payload but does not remove
the f32 activation traffic or the need for accurate `dL/dimage`, and the v02
gate did not promote it. The next exact-math CLIP fork should target selected
rectangular pointwise tiles or split-K `pw_bwd`, not blanket f16.

## Rect8x16 Pointwise Tile Gate

`PW_TILE_VARIANT=rect8x16` adds a gated exact-math forward pointwise tile. The
default pointwise tile covers `8 pixel-quads x 8 cout-quads` per workgroup. The
rect variant covers `8 pixel-quads x 16 cout-quads`, uses a `8x16` workgroup,
and stages `12 KB` of workgroup memory.

Correctness passed for the broad repeated `256<->768 @16x16` allowlist:

```bash
PW_TILE_VARIANT=rect8x16 PW_TILE_STEPS=57,59,62,64,67,69,72,74,77,79,82,84,87,89,92,94,97,99,102,104 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
```

The train-plan forward gate also passed with final embedding cosine
`1.000000`, and the B=3 gradient lanes all reported cosine `1.000000`.

Timing did not promote:

| Test | Base | Rect8x16 | Read |
| --- | ---: | ---: | --- |
| CLIP B=3 train matrix | `41.63 ms` | `41.43 ms` | flat |
| 3D step, broad allowlist normal | `55.99 ms` | `55.92 ms` | flat |
| 3D step, broad allowlist CLIP profile | `42.55 ms` | `44.53 ms` | worse |
| 3D step, first-four allowlist normal | `54.78 ms` | `55.41 ms` | worse |
| timestamp isolated total | `79.17 ms` | `100.73 ms` | worse |

Decision: keep the tile gate and plumbing, but do not enable it by default. The
next pointwise leap should be more structural than simply widening the
workgroup: true dual-cout accumulation, split-K `pw_bwd`, or narrow precision
forks with a full `dL/dimage` gate.

## Grid Literal Prompt Gate

`grid9_close2` does not change MobileCLIP's input resolution. Lane 0 remains a
normal `3x256x256` CLIP image; it contains a 3x3 contact sheet of nine camera
views. Lanes 1 and 2 remain full-resolution per-view renders. With
`grid raster 80`, the grid cells are rasterized directly at `80x80` before
being copied into the 256px contact sheet.

`Grid9PromptMode` now has a third option:

```text
contact_sheet: a 3x3 image grid showing the same subject...
literal:       a 3x3 grid showing {prompt} from 9 different camera angles...
same:          {prompt}
```

This is an opt-in prompt-quality gate, not a speed fork. Browser screenshots can
select `3x3 grid + 2`, `literal grid`, and `grid raster 80`. The quality harness
uses:

```bash
CONFIGS='base3=3:3,grid80=9:3:grid9:directgrid,grid80literal=9:3:grid9:directgrid:literal' bun tools/splat3d/grid_quality.ts
```

## Integrated Timestamp Step Profile

`tools/splat3d/step_bench.ts` now also accepts `TIMESTAMP=1` for integrated step
profiling. This requests `timestamp-query` and attaches begin/end timestamp
writes to the actual raster, CLIP, and Adam compute passes.

Important gotcha: a marker-pass approach was tested and rejected. Empty or tiny
timestamped passes did not bracket intervening GPU work reliably on this
Dawn/Metal runtime. Timestamp writes must live on the compute pass being timed.

Commands:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 VIEW_LANE_RASTER_BWD=1 CLIP_BATCH=3 VIEWS=3 RUNS=1 WARMUP=1 bun tools/splat3d/step_bench.ts
```

Default `3/9`, `batch CLIP x3` smoke:

| Timing Mode | Normal Step Avg | Profile Total | Raster Fwd | Raster Bwd | CLIP Batch | Adam | Display |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU timestamp | `79.80 ms` | `52.82 ms` | `1.31 ms` | `10.03 ms` | `41.03 ms` | `0.00 ms` | `0.46 ms` |
| split-submit wall | `64.72 ms` | `60.13 ms` | `2.07 ms` | `11.41 ms` | `43.91 ms` | `0.34 ms` | `0.93 ms` |

View-lane backward timestamp smoke moved raster backward from `10.03 ms` to
`9.24 ms` while CLIP batch stayed around `43 ms`, reinforcing the earlier
non-promotion decision for raster scheduler work.

Takeaway: integrated GPU attribution agrees with the CLIP dispatch profiler.
The next large win is still CLIP, not Adam/display and not shallow raster
scheduling.

## CLIP Depthwise Spatial Backward Fork

`SPATIAL_BWD_VARIANT=depthwise4` adds a gated depthwise-only CLIP backward
shader that computes four adjacent horizontal input pixels per thread. It passed
the backward correctness suite, including the end-to-end directional derivative
gate.

Isolated timestamp profiling was mixed: the `spatial_bwd` bucket improved
`15.47 ms -> 13.63 ms`, but total isolated median sum moved
`69.99 ms -> 75.76 ms` in that run. Integrated 3D step timing was more useful:

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| baseline | `53.12 ms` | `54.37 ms` | `40.00 ms` | `11.59 ms` |
| `depthwise4` | `49.96 ms` | `51.80 ms` | `38.24 ms` | `11.26 ms` |

Decision: preserve the gate and keep testing it, but do not call it a 2x-class
breakthrough. It is a real-looking ~6% integrated step win and evidence that
kernel shape still matters.

## Grid Contact Sheet Prompt

`grid9_close2` already keeps the CLIP input at `256x256`, but packs the nine
camera views into `80x80` cells inside that input and uses two extra
full-resolution close-up lanes. The page now makes the grid lane text explicit:

```text
a 3x3 image grid showing the same subject, {prompt}, from nine different camera
angles, centered on a black background: ...
```

The UI can toggle this against `same grid text`. This does not change kernel
speed; it is a quality/signal ablation for whether the contact-sheet image gets
a better CLIP target than a normal single-image prompt.

## Direct Grid Raster

`GRID_DIRECT_RASTER=1` changes `grid9_close2` so the nine contact-sheet views
render directly at `80x80` instead of rendering `256x256` scratch images and
downsampling. CLIP still receives one `256x256` lane containing nine `80x80`
cells plus two full-resolution close-up lanes.

Measured result:

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `grid9_close2` | `87.93 ms` | `89.85 ms` | `41.07 ms` | `46.01 ms` |
| `grid9_close2 + directgrid` | `59.03 ms` | `60.48 ms` | `38.72 ms` | `18.76 ms` |

Stacked with `SPATIAL_BWD_VARIANT=depthwise4`, all-view grid supervision reached
`56.08 ms` normal step median, close to the ordinary `3/9` per-view baseline
while touching all nine cameras every step. This is a real schedule+raster leap,
but it still needs visual quality testing before becoming default.

## Grid Real-Prompt Quality Gate

`tools/splat3d/grid_quality.ts` evaluates grid/contact-sheet schedules against
the real MobileCLIP text prompts used by the browser page. It trains each config
from deterministic initial splats, then scores all 9 views with full-resolution
per-view image embeddings. The grid lane's own contact-sheet prompt score is
not the primary metric.

First cat prompt repeat gate:

```bash
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat" CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid OUT_DIR=/tmp/nffa_grid_quality_cat_trials bun tools/splat3d/grid_quality.ts
```

| Variant | Trials | Steps / 5s | Steps / Sec | Mean Cos | Mean Delta | Min Cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base3` | 3 | 91 | 18.02 | 0.25410 | 0.12579 | 0.20980 |
| `grid80` | 3 | 78 | 15.55 | 0.23802 | 0.11083 | 0.19165 |

Read: `grid80` captured `88.1%` of `base3`'s median mean-cosine improvement and
`93.7%` of its final median mean cosine in the same 5s budget. That is a real
quality gap, but it is not a blow-up. Keep it gated while testing more prompts,
close-up lane policies, and periodic full per-view refresh.

## Shared-W Pointwise Forward

`SHARED_W_FWD_STEPS=...` wires the existing shared-weight batch pointwise
forward experiment into the 3D bench path. The idea is to avoid staging the same
pointwise weight tile once per batch lane.

The CLIP-only bench showed a tiny residual-step allowlist win:

| Variant | Batch Median |
| --- | ---: |
| baseline | `40.86 ms` |
| `SHARED_W_FWD_STEPS=10,15,24,34,49` | `40.65 ms` |

Integrated 3D timing did not promote it:

| Variant | Normal Median | CLIP Median |
| --- | ---: | ---: |
| `3/9 baseline` | `52.88 ms` | `41.30 ms` |
| `3/9 shared-W residuals` | `53.27 ms` | `40.99 ms` |
| `grid80 + dw4` | `55.47 ms` | `36.20 ms` |
| `grid80 + dw4 + shared-W` | `56.41 ms` | `36.02 ms` |

Decision: keep it as a diagnostic env gate, but do not enable it by default.
The next pointwise attempt should target the base tile shape or backward
pointwise, not shared-W forward.

## Backward Local Fusion Refresh

The older single `FUSE_GELU_BWD_PW=1` gate did not clear the integrated
promotion bar on the then-current stack. Re-testing both legal backward fusions
together on the newer grid contact-sheet stack changed the read:

```bash
TRIALS=7 CONFIGS=grid80dw4=9:3:grid9:directgrid:dw4,grid80dw4both=9:3:grid9:directgrid:dw4:gelubwd:resbwd RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `grid80 + depthwise4` | `56.20 ms` | `58.33 ms` | `36.41 ms` | `19.51 ms` |
| `grid80 + depthwise4 + GELU/residual bwd fusions` | `54.63 ms` | `56.81 ms` | `34.91 ms` | `19.29 ms` |

Correctness:

```bash
SPATIAL_BWD_VARIANT=depthwise4 FUSE_GELU_BWD_PW=1 FUSE_RESIDUAL_BWD_PW=1 bun tools/clip/bwd_test.ts
```

`bwd_test` passed all per-kernel references and the end-to-end directional
derivative gate. The isolated timestamp profile also showed dispatch count
falling from `257` to `211`, but the integrated gain is the number to trust.

Decision: this is a real small CLIP win on the current best stack. Keep it
gated until the interactive default path is chosen; do not mistake it for the
2-4x structural speedup.

## Chrome Dawn Trace Helper

`tools/webgpu_trace.mjs` captures a Chrome DevTools trace around the running
browser optimizer. It complements, but does not replace, Bun/WebGPU timestamp
profiling. It can show Dawn/WebGPU command traffic, queue submits, Metal
backpressure, browser scheduling, and pipeline events. It does not expose
per-shader memory bandwidth, occupancy, cache behavior, register pressure, or
stall counters.

The correct local trace path uses a built page served from the repo root so
`/dist/` assets and `/models/` weights are same-origin:

```bash
npx parcel build --no-scope-hoist --no-cache --public-url ./ src/splat3d.html
node tools/splat/serve.mjs 8799
GRID9=1 DIRECT_GRID=1 SETUP_TIMEOUT_MS=180000 node tools/webgpu_trace.mjs http://localhost:8799/dist/splat3d.html 2000 /tmp/nffa_trace_probe4
```

Tracing the Parcel dev server directly failed because Parcel returned
`index.html` for `/models/mobileclip_s0/plan_train.json`, which made app boot
fail with `Unexpected token '<'`.

Successful probe:

| Field | Value |
| --- | --- |
| Trace | `/tmp/nffa_trace_probe4/webgpu_trace_2026-07-08T12-39-16-079Z.json` |
| Screenshot | `/tmp/nffa_trace_probe4/webgpu_trace_2026-07-08T12-39-16-079Z.png` |
| Mode | `grid9_close2`, `gridDirectRaster=true`, `views=9`, `clipBatch=3` |
| Steps | `2 -> 35` |
| Events | `40,331` |

Top relevant buckets included `DawnCommands`,
`DeviceMTL::SubmitPendingCommandBuffer`, `CommandEncoder::Finish`, and
`Queue::Submit`. Use Perfetto or `chrome://tracing` to inspect the JSON.

## Cached CLIP Gradient Cadence

`CLIP_REFRESH_INTERVAL=N` is an env-gated benchmark path for the default
per-view batch optimizer. It does not update CLIP. CLIP remains a frozen
teacher; refresh steps compute `dL/dimage`, and cached steps reuse the previous
full-resolution image gradient while still running raster forward, raster
backward, Adam, and display.

Scope of the first fork:

- works for `per_view` with one complete `CLIP_BATCH` chunk, e.g.
  `VIEWS=3 CLIP_BATCH=3`;
- keeps selected CLIP inputs at full `256x256`;
- does not apply to `grid9_close2`, single-CLIP paths, or multi-chunk `9/9`;
- epoch view sampling advances on refresh steps only.

Clean exact-cycle timing:

```bash
TRIALS=5 CONFIGS=base=3:3,cache2=3:3:cache2,cache4=3:3:cache4 RUNS=8 WARMUP=6 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `base` | `53.03 ms` | `56.52 ms` | `41.00 ms` | `13.66 ms` |
| `cache2` | `32.60 ms` | `55.58 ms` | `40.96 ms` | `12.38 ms` |
| `cache4` | `22.40 ms` | `14.09 ms` | `0.00 ms` | `12.48 ms` |

Normal-step speedup over base:

- `cache2`: `1.63x`
- `cache4`: `2.37x`

The sampled profile may land on either a refresh step or a cached step, so use
normal-step averages over exact refresh cycles for cadence comparisons.

Decision: keep this gated. It is the first concrete 2x-class wall-clock lever
that preserves full-resolution selected CLIP views, but it is a proxy schedule,
not an equivalent loss. Promotion requires a fixed-wall-clock quality gate
using full-teacher scores and all nine camera screenshots.

## Cached CLIP Quality Gate

`tools/splat3d/cadence_quality.ts` runs the missing fixed-budget gate for
`CLIP_REFRESH_INTERVAL`. It starts each config from the same deterministic splat
initialization, optimizes for the same wall-clock budget, then evaluates all 9
views with the full frozen CLIP image tower and writes a 3x3 PNG contact sheet.

```bash
BUDGET_MS=5000 CONFIGS=base=1,cache2=2,cache4=4 bun tools/splat3d/cadence_quality.ts
```

Result on Apple `metal-3`, `G=4096`, `views=3`, `clipBatch=3`, f32 weights:

| Variant | Refresh Interval | Steps / 5s | Steps / Sec | Mean Full-Teacher Cos | Min Cos | Max Cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 1 | 88 | 17.59 | 0.09336 | 0.04962 | 0.15112 |
| `cache2` | 2 | 147 | 29.33 | 0.04363 | 0.01030 | 0.07999 |
| `cache4` | 4 | 221 | 44.11 | 0.02805 | -0.01557 | 0.07909 |

Artifacts are saved under
`experiments/clip_forks/v14_cadence_quality_gate/results/2026-07-08/`.

Decision: do not promote naive cached gradients. The wall-clock step-rate win
is real, but the first full-teacher quality gate regressed. `cache4` ran `2.51x`
as many optimizer steps as base but reached only about `30%` of base's mean
teacher cosine. This path needs a smarter schedule, likely lower cached-step
learning rates, different cached-step objectives, or periodic full 9-view
teacher refreshes.

## Cached-Step LR Scale

`CLIP_CACHED_LR_SCALE=N` scales splat Adam learning rates only on cached-gradient
steps. Refresh steps keep the normal learning rates. This tests whether stale
`dL/dimage` was being applied too aggressively.

```bash
BUDGET_MS=5000 CONFIGS=base=1,cache2=2,cache2lr50=2:0.5,cache4=4,cache4lr25=4:0.25 OUT_DIR=/tmp/nffa_cadence_lr_quality bun tools/splat3d/cadence_quality.ts
```

| Variant | Refresh Interval | Cached LR Scale | Steps / 5s | Steps / Sec | Mean Full-Teacher Cos |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 1 | 1.00 | 87 | 17.26 | 0.09413 |
| `cache2` | 2 | 1.00 | 149 | 29.49 | 0.04559 |
| `cache2lr50` | 2 | 0.50 | 142 | 28.37 | 0.05842 |
| `cache4` | 4 | 1.00 | 213 | 42.16 | 0.02450 |
| `cache4lr25` | 4 | 0.25 | 217 | 43.32 | 0.04000 |

Step-speed matrix:

```bash
TRIALS=3 CONFIGS=base=3:3,cache2=3:3:cache2,cache2lr50=3:3:cache2:lr0.5,cache4=3:3:cache4,cache4lr25=3:3:cache4:lr0.25 RUNS=8 WARMUP=6 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `base` | 53.64 ms | 57.31 ms | 41.94 ms | 13.31 ms |
| `cache2` | 32.90 ms | 56.65 ms | 41.81 ms | 12.76 ms |
| `cache2lr50` | 32.69 ms | 57.28 ms | 41.64 ms | 12.74 ms |
| `cache4` | 22.61 ms | 15.01 ms | 0.00 ms | 12.51 ms |
| `cache4lr25` | 22.75 ms | 14.53 ms | 0.00 ms | 12.68 ms |

Decision: keep gated, do not promote. Cached-step LR scaling improves quality
without hurting speed (`cache2` +28%, `cache4` +63% mean teacher cosine versus
their naive versions), but both still trail base badly. Stale gradient step size
is part of the issue; repeated camera batches and stale objective semantics are
still unresolved.

## Prompt Encoding Cache

The 3D page now caches text embeddings by exact expanded prompt. In `same text`
mode it encodes one prompt and reuses the embedding for all 9 views. In camera
text mode, repeated runs of the same prompt reuse the cached camera-specific
embeddings.

This does not change optimizer step speed, but it removes duplicate text tower
work from prompt setup and makes same-text ablations less annoying to run.

## Raster/CLIP Buffer Aliasing

The rasterizer can now bind external image and image-gradient buffers through
`Raster3DIOState`. The 3D optimizer uses this to render directly into CLIP input
slots and read gradients directly from CLIP input-gradient slots.

This removes two full-image copies per optimized view. It is not a large
standalone speedup in current measurements because CLIP dominates, but it is the
right setup for per-lane raster state and batched raster/CLIP scheduling.

## Per-Lane Raster State

Batch CLIP lanes now get private raster scratch state: `derived`, `tileCounts`,
`binnedIds`, and `tileStop`. The optimizer can render a chunk of selected views
into CLIP batch input lanes, run one batch CLIP train pass, then apply each
lane's raster backward using the saved tile state. The replay forward pass is
gone for complete batch chunks.

Sequential benchmark on Apple `metal-3` after this change:

| Views / Step | CLIP Batch | Normal Step Avg | Split Profile Total | Raster Replay |
| --- | ---: | ---: | ---: | ---: |
| 3 / 9 | 1 | 80.87 ms | 101.99 ms | 0.00 ms |
| 3 / 9 | 3 | 58.40 ms | 62.39 ms | 0.00 ms |
| 9 / 9 | 1 | 195.65 ms | 215.00 ms | 0.00 ms |
| 9 / 9 | 3 | 167.23 ms | 176.97 ms | 0.00 ms |
| 9 / 9 | 9 | 235.48 ms | 295.60 ms | 0.00 ms |

Takeaway: `batch CLIP x3` is now the useful multi-view speed mode. `batch CLIP
x9` is still not worth promoting on this hardware.

The 3D page now defaults to `3 / 9 views` plus `batch CLIP x3`. Single CLIP and
batch x9 remain available as explicit ablation toggles.

## Step Matrix Runner

`tools/splat3d/step_matrix.ts` runs `step_bench.ts` sequentially across a config
matrix and reports medians plus min/max ranges. Use it for future performance
promotion decisions when the GPU is noisy:

```bash
TRIALS=2 CONFIGS=3:1,3:3 RUNS=4 WARMUP=2 bun tools/splat3d/step_matrix.ts
```

First control run under a slow/noisy GPU state still showed the same-session
direction: `3/9 batch x3` median normal step `143.66 ms` versus `194.68 ms` for
`3/9 single CLIP`.

## Multi-View Raster Batching Read

STAR UVT/world tubes do not directly turn our 9 fixed CLIP cameras into one
cheap raster. Their speed comes from a primitive whose support is native over
ordered time in one camera gauge, or over a projective rational moving-camera
gauge. Our 9 views have different projection maps, tile support, visibility,
and depth order, so one shared cross-view tile sort would be wrong or too loose.

The near-term raster path is still valuable:

- move camera constants out of baked WGSL into a camera storage/uniform buffer;
- make `derived`, `tileCounts`, `binnedIds`, and `tileStop` lane-strided;
- dispatch prep/emit/forward/backward with `workgroup_id.z = view lane`;
- keep one tile list and one depth order per camera lane.

This should cut CPU/pass/pipeline overhead and bind churn, but it will not remove
the per-view projection, binning, sort, compositing, or backward transmittance
walk. A true STAR-style sublinear multi-camera primitive would be a separate
representation project: projective rational camera bundles or atlas-residual
splats with a new backward chain.

Follow-up note: `agent_notes/optimization_session/static_multiview_worldtube_followup.md`
re-checked the STAR UVT/PRT world-tube math against the current browser raster.
The exact next ablation does not require a new splat object: move camera data
into a buffer and dispatch raster kernels over `workgroup_id.z = view lane`.
The only backward wrinkle is that `accGrad` is currently shared and cleared per
view, so a true batched backward needs lane-strided `accGrad` first, or a larger
fixed-point raw-gradient atomic rewrite later.

## Stem Spatial Backward Specialization

The B=3 dispatch profile made the MobileCLIP stem backward the largest single
CLIP kernel: `spatial_bwd k3s2 3<-64 g1 @256x256`. A stem-only
`spatial_bwd_stem4` variant now computes four horizontal input pixels per
thread and is enabled by default for the 3D batch optimizer.

Same-session integrated matrix:

```bash
STEM_SPATIAL_BWD=0 TRIALS=2 CONFIGS=3:3 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
TRIALS=2 CONFIGS=3:3 RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Step Median | Profile Median | CLIP Median |
| --- | ---: | ---: | ---: |
| stem off | `95.58 ms` | `116.12 ms` | `86.56 ms` |
| stem on | `72.72 ms` | `90.13 ms` | `65.30 ms` |

Use `STEM_SPATIAL_BWD=0` as the negative-control path when future CLIP changes
need to compare against the pre-specialization baseline.

## Pointwise GELU Forward Fusion

The B=3 post-stem dispatch profile still showed standalone train-mode GELU
forward/backward as a visible traffic cost. The promoted forward-only fusion
handles the 24 exact pointwise-conv + GELU pairs by writing both the saved
pre-activation slot and the GELU output slot from one pointwise dispatch.
Spatial and SE GELU pairs remain split.

Verification:

```bash
FUSE_PW_GELU=1 STEM_SPATIAL_BWD=1 BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/batch_major_train_bench.ts
bun tools/clip/bwd_test.ts
npx parcel build --no-scope-hoist --no-cache src/index.html src/splat.html src/splat3d.html
```

Same-session B=3 CLIP matrix:

| Variant | B=3 CLIP Train Median |
| --- | ---: |
| stem only | `73.33 ms` |
| stem + pointwise GELU fusion | `68.06 ms` |

Integrated 3D step matrix:

| Variant | Normal Step Median | Profile Median | CLIP Median |
| --- | ---: | ---: | ---: |
| `FUSE_PW_GELU=0` | `88.99 ms` | `111.50 ms` | `80.68 ms` |
| default fused | `87.28 ms` | `104.88 ms` | `76.38 ms` |

The 3D batch optimizer enables this by default. Use `FUSE_PW_GELU=0` in
`tools/splat3d/step_bench.ts` / `step_matrix.ts` as the negative control.

## GELU Backward Fusion Gate

A follow-on ablation fused exact adjacent `gelu_bwd` + `pw_bwd` pairs by loading
`dY * geluGrad(pre)` directly into the pointwise backward tile. This reduced
B=3 CLIP train median in isolation:

| Variant | B=3 CLIP Train Median |
| --- | ---: |
| default forward GELU fusion | `68.22 ms` |
| + GELU backward fusion | `61.70 ms` |

The integrated alternating 3D matrix did not clear the promotion bar:

| Variant | Normal Step Median | Profile Median | CLIP Median |
| --- | ---: | ---: | ---: |
| default | `78.30 ms` | `93.50 ms` | `67.51 ms` |
| `FUSE_GELU_BWD_PW=1` | `76.83 ms` | `94.71 ms` | `67.02 ms` |

Decision: keep `FUSE_GELU_BWD_PW=1` as a gated ablation, but do not enable it
by default in the 3D optimizer.

## Single-Pass Batch Raster Forward Gate

A shallow raster scheduler ablation tried recording all selected batch-view
forwards inside one compute pass:

```bash
SINGLE_PASS_RASTER_FWD=1 CLIP_BATCH=3 VIEWS=3 bun tools/splat3d/step_bench.ts
TRIALS=3 CONFIGS=base=3:3,rasterpass=3:3:rasterpass RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

The path is exact and remains available behind `SINGLE_PASS_RASTER_FWD=1`, but
it is not enabled by default. Follow-up reruns after testing default promotion
favored the existing separate-forward encoding for both `3/9` and `9/9` batch
x3. This suggests pass-boundary churn is not the useful raster target.

Next raster work should move directly to the camera-buffer / view-lane dispatch
design with lane-strided raster state and, eventually, lane-strided `accGrad`.

## View-Lane Raster Forward Gate

The exact camera-buffer / view-lane forward path has now been tested behind
`VIEW_LANE_RASTER_FWD=1`. It moves camera constants into a compact storage
buffer, dispatches `prep`, `emit`, and `forward` over `workgroup_id.z = lane`,
and stores the resulting tile state in lane-strided scratch. Backward still uses
the existing per-lane `recordBackwardAdd()` path through aligned buffer slices.

Correctness passed exactly:

```bash
bun tools/splat3d/raster_batch_forward_test.ts
```

```text
image diff: max=0.000e+0 mean=0.000e+0
grad diff:  max=0.000e+0 mean=0.000e+0
```

It did not clear the promotion bar:

| Views / Batch | Variant | Normal Median | Profile Median | Raster Median |
| --- | --- | ---: | ---: | ---: |
| `3/3` | default separate forward | `52.92 ms` | `55.90 ms` | `12.81 ms` |
| `3/3` | `VIEW_LANE_RASTER_FWD=1` | `53.03 ms` | `57.23 ms` | `13.77 ms` |
| `9/3` | default separate forward | `154.26 ms` | `161.91 ms` | `36.85 ms` |
| `9/3` | `VIEW_LANE_RASTER_FWD=1` | `152.82 ms` | `163.51 ms` | `38.03 ms` |

Decision: keep the gated path and parity tool, but do not enable it by default.
The next raster attempt should target batched backward/lane-strided `accGrad` or
overflow/workgroup-staging telemetry instead of forward scheduling alone.

## View-Lane Raster Backward Gate

The lane-strided raster state now also has a gated batched backward path behind
`VIEW_LANE_RASTER_BWD=1`. It adds lane-strided `accGrad`, runs tile backward
with `workgroup_id.z = lane`, then applies the existing camera-specific
`chainAdd` dispatch sequentially per lane.

Correctness passed exactly:

```bash
bun tools/splat3d/raster_batch_forward_test.ts
```

```text
image diff: max=0.000e+0 mean=0.000e+0
grad diff:  max=0.000e+0 mean=0.000e+0
batch backward diff: max=0.000e+0 mean=0.000e+0
```

It did not clear the default-promotion bar:

| Views / Batch | Variant | Normal Median | Profile Median | Raster Median |
| --- | --- | ---: | ---: | ---: |
| `3/3` | default | `52.69 ms` | `56.81 ms` | `13.45 ms` |
| `3/3` | `VIEW_LANE_RASTER_BWD=1` | `53.37 ms` | `57.37 ms` | `12.63 ms` |
| `3/3` | `VIEW_LANE_RASTER_FWD=1 VIEW_LANE_RASTER_BWD=1` | `53.44 ms` | `57.60 ms` | `12.73 ms` |
| `9/3` | default | `157.01 ms` | `162.59 ms` | `37.66 ms` |
| `9/3` | `VIEW_LANE_RASTER_BWD=1` | `157.41 ms` | `163.03 ms` | `35.00 ms` |
| `9/3` | `VIEW_LANE_RASTER_FWD=1 VIEW_LANE_RASTER_BWD=1` | `152.42 ms` | `163.78 ms` | `36.14 ms` |

Decision: keep the gate and parity coverage, but do not enable it by default.
The sampled raster split improved in places, but the default 3-view optimizer
step got slower. Further raster work should add overflow/occupancy telemetry or
real timestamp attribution before more scheduling rewrites.

## Raster Occupancy Telemetry

`tools/splat3d/raster_telemetry.ts` now reads `tileCounts` and `tileStop` after
real raster forwards:

```bash
bun tools/splat3d/raster_telemetry.ts
G=12000 CAP=2048 bun tools/splat3d/raster_telemetry.ts
```

Default `G=4096`, `CAP=2048` has no overflow:

| Metric | Value |
| --- | ---: |
| Aggregate overflow pairs | `0 / 576871` |
| Max tile count | `911` |
| Max tile stop | `365` |
| Worst dropped pair pct | `0.0%` |

Stress `G=12000`, `CAP=2048` starts to overflow:

| Metric | Value |
| --- | ---: |
| Aggregate overflow pairs | `21440 / 1717219` (`1.2%`) |
| Max tile count | `2678` |
| Max tile stop | `702` |
| Worst dropped pair pct | `3.6%` |

Takeaway: overflow handling is not a default 4096-splat bottleneck. The current
`2048` cap is conservative for the default scene, so the next concrete raster
ablation should test `cap=1024` in the integrated step matrix, with telemetry as
the safety gate.

## Cap 1024 Gate

`CAP=1024` was tested as a smaller tile capacity after telemetry showed default
max tile count at `911`. The smaller cap is overflow-free for the measured
initial default scene:

```bash
CAP=1024 bun tools/splat3d/raster_telemetry.ts
```

Integrated timing was mixed:

| Views / Batch | Cap | Normal Median | Profile Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `3/3` | `2048` | `53.00 ms` | `58.08 ms` | `13.56 ms` |
| `3/3` | `1024` | `53.44 ms` | `56.78 ms` | `13.83 ms` |
| `9/3` | `2048` | `156.02 ms` | `166.22 ms` | `38.69 ms` |
| `9/3` | `1024` | `156.21 ms` | `165.15 ms` | `38.41 ms` |

Decision: keep `CAP` / `capNNN` benchmark controls, but do not change the
default cap. The result is too flat, and lower cap has less safety margin during
optimization if radii or positions concentrate.
