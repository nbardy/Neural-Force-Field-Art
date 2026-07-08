# v04 Pointwise Tile Rewrite

Goal: create a clean fork for pointwise WGSL matmul variants that can plausibly
move total CLIP train time by a real amount. This is not a source change yet.
The next implementation should add gated variant emitters and keep the current
`pointwiseTiledMain()` as the fallback until a variant wins full CLIP and 3D
step gates.

## Why This Is the Right Target

Current timestamp ranking for promoted B=3 train settings:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Recorded group result:

| Group | Time | Share |
| --- | ---: | ---: |
| `pw_bwd` | `42.01 ms` | `26.2%` |
| `pw+gelu` | `23.92 ms` | `14.9%` |
| `pw` | `23.20 ms` | `14.5%` |

The pointwise family is therefore about `55.6%` of isolated B=3 CLIP timestamp
time. A true 2x pointwise-family win would be roughly a `28%` CLIP win before
secondary effects. A 2x whole-CLIP win from this fork alone is unlikely; a
2x-ish perceived optimizer speedup needs this plus view scheduling, f16, or a
proxy cadence. Still, pointwise is the largest shader area where a real kernel
rewrite can matter.

## Current Math

Pointwise means 1x1 convolution with no groups. It is just a matrix multiply
over every spatial/token position:

```text
Y[p, co] = bias[co] + sum_ci X[p, ci] * W[ci, co]
```

The code stores activations channel-planar, packed as four adjacent pixels:

```text
src[ci][p4] : vec4f = X[p4*4 + 0..3, ci]
dst[co][p4] : vec4f = Y[p4*4 + 0..3, co]
P = H * W
P4 = P / 4
```

Forward weights are compiled transposed as `[Cin][Cout]` so four adjacent output
channels are contiguous:

```text
W scalar index = wOff + ci * Cout + co
W4((wOff + ci * Cout + co4) / 4) = W[ci, co4..co4+3]
```

Backward `pw_bwd` computes `dX = W^T * dY`, but the compiler emits `wOffT` so
the same tiled body still reads `[reduction][output]` with contiguous output
channel quads. There is no `dW`; CLIP weights are frozen.

## Current Layout and Tile

Current kernel:

```text
workgroup_size = 8 x 8 = 64 threads
workgroups     = [P4 / 8, Cout / 32, 1]
tile owns      = 8 pixel-quads x 32 cout = 32 pixels x 32 cout
thread owns    = 1 pixel-quad x 4 cout = 16 scalar outputs
```

Shared memory:

```text
xS: 256 vec4f = 4096 bytes  // 32 ci x 8 pixel-quads
wS: 256 vec4f = 4096 bytes  // 32 ci x 8 cout-quads
total = 8192 bytes
```

Per thread:

```text
acc0..acc3 : vec4f
for ci0 in 0..Cin step 32:
  stage xS and wS
  barrier
  for ci in 0..31:
    xv = xS[ci, lid.x]
    wv = wS[ci, lid.y]
    acc0..acc3 += wv.xyzw * xv
  barrier
```

The current tile is already decent. The fork should avoid blanket replacement
and instead test only shapes that appear heavily in full CLIP.

Initial forward target shapes from `plan_train.json`:

```text
steps 8,10,13,15:       64<->192 @64x64
steps 22,24,...,49:     128<->384 @32x32
steps 57,59,...:        256<->768 @16x16
```

Initial backward target shapes from `plan_train.json`:

```text
top pw_bwd shapes:      512<->1536 @8x8
repeated pw_bwd shapes: 256<->768 @16x16
```

## Variant A: Dual Pixel Tile

Shape:

```text
current: 8 pixel-quads x 32 cout
variant: 16 pixel-quads x 32 cout
```

Intent: reuse one staged W tile over twice as many pixels. This should help
large-P layers, especially `64x64` and `32x32`.

Memory:

```text
xS: 512 vec4f = 8192 bytes
wS: 256 vec4f = 4096 bytes
total = 12288 bytes
```

Register cost:

```text
two pixel-quads per thread
acc0a..acc3a + acc0b..acc3b = 8 vec4f accumulators
```

Expected tradeoff:

- Better W reuse and fewer x-dimension workgroups.
- Higher register pressure and possibly lower occupancy.
- Likely best for early high-resolution pointwise layers.
- Risky on `8x8` and late layers where workgroup count is already small.

Gate name for future source work:

```text
PW_TILE_VARIANT=dual_pixel
PW_TILE_STEPS=8,10,13,15,22,24,27,29
```

## Variant B: Dual Cout Tile

Shape:

```text
current: 8 pixel-quads x 32 cout
variant: 8 pixel-quads x 64 cout
```

Intent: reuse one staged X tile over twice as many output channels. This should
help expansion/contraction layers where the same input pixels are reread across
many output-channel tiles.

Memory:

```text
xS: 256 vec4f = 4096 bytes
wS: 512 vec4f = 8192 bytes
total = 12288 bytes
```

Register cost:

```text
eight output channels per thread, likely 8 vec4f accumulators
```

Expected tradeoff:

- Better X reuse and fewer y-dimension workgroups.
- More W staging and more accumulators.
- Could win on `256->768`, `768->256`, `512->1536`, and `1536->512`.
- Can lose badly if Metal spills accumulators to private memory.

Gate name for future source work:

```text
PW_TILE_VARIANT=dual_cout
PW_TILE_STEPS=57,59,62,64
PW_BWD_TILE_STEPS=6,8,15,17,30,32
```

## Variant C: Rectangular 8x16 Threads

Shape:

```text
current: workgroup_size 8 x 8,  tile 8 p4 x 32 cout
variant: workgroup_size 8 x 16, tile 8 p4 x 64 cout
```

Intent: keep one pixel-quad per thread while using more threads to cover more
cout lanes, reducing per-thread accumulator growth versus Variant B.

Memory:

```text
xS: 256 vec4f = 4096 bytes
wS: 512 vec4f = 8192 bytes
total = 12288 bytes
threads = 128
```

Expected tradeoff:

- Less register pressure than dual-cout.
- More threads can reduce occupancy if the adapter's threadgroup limits or
  scheduling prefer 64-thread groups.
- More shared-memory staging traffic per workgroup than the current kernel.
- Worth testing if dual-cout wins isolated dispatches but fails full CLIP due
  to spills or occupancy.

Gate name for future source work:

```text
PW_TILE_VARIANT=rect_8x16
PW_TILE_STEPS=57,59,62,64
```

## Measurement Plan

Always run same-session baseline and variant. Do not compare against stale
numbers from another thermal state.

### 1. Pick Hot Shapes

```bash
CSV=1 TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts > /tmp/clip_b3_baseline.csv
```

Sort by labels starting with `pw `, `pw+gelu`, and `pw_bwd`. The first variant
should target only a handful of repeated shapes.

### 2. Existing Pointwise Microbench Sanity

This does not benchmark the new tile variants yet, but it is useful context for
batch/shared-W behavior and shape sensitivity:

```bash
BATCH=3 STEP_INDEX=8 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCH=3 STEP_INDEX=57 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
BATCH=3 STEP_INDEX=59 RUNS=40 WARMUP=10 bun tools/clip/pointwise_batch_bench.ts
```

The v04 implementation should add a dedicated tile microbench, expected command:

```bash
PW_TILE_VARIANT=dual_pixel STEP_INDEX=8 RUNS=80 WARMUP=20 bun tools/clip/pointwise_tile_bench.ts
PW_TILE_VARIANT=dual_cout STEP_INDEX=57 RUNS=80 WARMUP=20 bun tools/clip/pointwise_tile_bench.ts
PW_TILE_VARIANT=rect_8x16 STEP_INDEX=57 RUNS=80 WARMUP=20 bun tools/clip/pointwise_tile_bench.ts
```

The microbench must compare variant output against the current `pointwise()`
emitter on deterministic random buffers and fail on relative Linf drift above
the chosen f32 tolerance.

### 3. Full CLIP Timestamp Gate

Baseline:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Variant:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 PW_TILE_VARIANT=dual_pixel PW_TILE_STEPS=8,10,13,15,22,24,27,29 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 PW_TILE_VARIANT=dual_cout PW_TILE_STEPS=57,59,62,64 PW_BWD_TILE_STEPS=6,8,15,17,30,32 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 PW_TILE_VARIANT=rect_8x16 PW_TILE_STEPS=57,59,62,64 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Promotion threshold:

```text
selected pointwise labels: >= 1.20x median speedup
all pointwise groups:      >= 10% timestamp-sum reduction
all CLIP train groups:     >= 8% timestamp-sum reduction
```

### 4. Integrated 3D Step Gate

Baseline:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Variant:

```bash
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 PW_TILE_VARIANT=dual_pixel PW_TILE_STEPS=8,10,13,15,22,24,27,29 bun tools/splat3d/step_bench.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 PW_TILE_VARIANT=dual_cout PW_TILE_STEPS=57,59,62,64 PW_BWD_TILE_STEPS=6,8,15,17,30,32 bun tools/splat3d/step_bench.ts
```

Promotion threshold:

```text
clipBatch:       >= 8% reduction
normal step avg: >= 5% reduction
no raster/adam regression attributable to changed scheduling
```

Use the matrix runner for repeated trials:

```bash
TRIALS=3 CONFIGS=base=3:3,dual_pixel=3:3:pwdualpx,dual_cout=3:3:pwdualco RUNS=5 WARMUP=3 bun tools/splat3d/step_matrix.ts
```

If `step_matrix.ts` does not yet understand these names, add that mapping in
the implementation commit, not in this note.

## Counter Profiling Plan

Timestamp queries answer which dispatch is slow. They do not answer whether the
variant is limited by memory bandwidth, registers, occupancy, or barriers.

### WebGPU Timestamp Evidence

Use timestamps first because they already work in repo tools:

```bash
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 CLIP_BATCH=3 VIEWS=3 RUNS=5 WARMUP=3 bun tools/splat3d/step_bench.ts
```

Record:

- adapter line;
- selected pointwise labels and workgroup counts;
- grouped `pw`, `pw+gelu`, `pw_bwd`, total CLIP time;
- integrated `clipBatch` and `normal step avg`.

### Chrome Trace

For browser-side Dawn scheduling evidence:

```bash
yarn start
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --enable-unsafe-webgpu \
  --user-data-dir=/tmp/nffa-chrome-pw-profile \
  --trace-startup=gpu,dawn,disabled-by-default-gpu.dawn \
  --trace-startup-file=/tmp/nffa_pointwise_trace.json \
  http://localhost:1234/splat3d.html
```

Use this to check dispatch ordering, submit gaps, and whether JavaScript or
pipeline creation noise is hiding the shader change.

### Xcode / Metal System Trace

Full Xcode is required; Command Line Tools alone were not enough in the earlier
profiler note.

Check availability:

```bash
xcode-select -p
xcrun xctrace list templates | rg -i "metal|gpu"
```

Record a CLI benchmark:

```bash
xcrun xctrace record \
  --template "Metal System Trace" \
  --time-limit 30s \
  --output /tmp/nffa_pointwise_baseline.trace \
  --launch -- /usr/bin/env TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Record a variant:

```bash
xcrun xctrace record \
  --template "Metal System Trace" \
  --time-limit 30s \
  --output /tmp/nffa_pointwise_dual_cout.trace \
  --launch -- /usr/bin/env TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 PW_TILE_VARIANT=dual_cout PW_TILE_STEPS=57,59,62,64 MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

Counters to look for if Instruments exposes them for Dawn-generated Metal:

- GPU duration by compute encoder and shader;
- threadgroup occupancy / SIMD occupancy;
- registers per thread or spill/private-memory indicators;
- device memory read/write bandwidth;
- threadgroup memory bandwidth;
- cache hit rate if available;
- stall reasons, especially memory dependency and threadgroup barrier stalls.

Expected signatures:

- Dual-pixel should reduce W/device reads for high-P shapes. If duration does
  not improve and register pressure rises, reject it.
- Dual-cout should reduce X/device reads but may spill. If private memory or
  register pressure jumps, reject or try `rect_8x16`.
- Rectangular 8x16 should show lower spill risk than dual-cout but may have
  worse occupancy from 128-thread groups.

## Failure Modes

- Isolated tile wins but full CLIP loses because changed workgroup geometry
  hurts scheduling or cache behavior across all dispatches.
- Variant wins split-submit wall time but not timestamp time.
- Register pressure causes Metal private-memory spills.
- Larger threadgroups reduce occupancy more than the memory reuse helps.
- The selected steps are not the actual hot labels after B=3 batch-major codegen.
- FMA order changes create small numeric drift that compounds into worse image
  gradients.
- Fused-GELU and batch-major paths accidentally bypass the variant gate.
- `pw_bwd` uses different shape names/indices than forward; allowlists must
  distinguish forward plan step indices from backward step indices.
- Extra source variants increase pipeline compile time enough to hurt app load
  even if steady-state timing improves.

## Key Recommendation

Start with `dual_pixel` on the early `64x64` and `32x32` forward shapes, and
`dual_cout` on the repeated `16x16` and `8x8` forward/backward shapes. Keep both
behind explicit allowlists. Promote neither unless timestamped full CLIP and
integrated `CLIP_BATCH=3 VIEWS=3` improve in the same session.
