# v15 Agent Synthesis

Date: 2026-07-08

## What The Five Agents Clarified

### FP16

CLIP fp16 exists as weights-only f16. It halves weight payload, but it did not
give a promotable big jump:

- embedding cosine stayed excellent (`0.99999559`);
- input-gradient cosine failed the planned gate (`0.97493807`, target
  `>=0.995`);
- stable isolated timing was f16 slower (`67.109 ms -> 71.107 ms`);
- integrated 3D recorded run was slower (`54.98 ms -> 59.90 ms`).

The remembered large jump was batch-major scheduling or the older non-CLIP
particle f16 advect path, not CLIP fp16.

Next fp16 work should be selective pointwise-family f16 weights with f32
accumulation, not another all-weight f16 pass.

### Pointwise

Pointwise means a `1x1`, `groups=1` convolution:

```text
Y[co, p] = bias[co] + sum_ci X[ci, p] * W[ci, co]
dX[ci, p] = sum_co dY[co, p] * W[ci, co]
```

The train graph has 48 forward pointwise steps, 48 backward pointwise steps,
and 24 pointwise->GELU pairs. The pointwise family was over half of the
recorded isolated B=3 CLIP timestamp profile.

Activations are planar f32 slots viewed as `array<vec4f>`:

```text
src[ci * P4 + p4] = four adjacent pixels for channel ci
dst[co * P4 + p4] = four adjacent pixels for channel co
```

Current tiling uses an `8x8` workgroup, stages `xS` and `wS` as 256 `vec4f`
each, and computes an `8 pixel-quads x 32 output-channel` tile. The next
pointwise experiment should be a shape-gated rectangular tile variant, not
another shared-W forward allowlist.

### GPU Profiling

Available now:

- Bun/WebGPU on Apple M4 Metal;
- WebGPU timestamp queries;
- `tools/clip/dispatch_profile.ts`;
- `tools/splat3d/step_bench.ts`;
- `tools/splat3d/step_matrix.ts`;
- Chrome/Dawn trace via `tools/webgpu_trace.mjs` on real Metal.

Unavailable now:

- full Xcode Instruments / Metal System Trace;
- `metalperftrace`;
- Metal shader counters for bandwidth, occupancy, register pressure, cache
  misses, and stall reasons.

So the honest profiler status is: we can rank dispatches and browser queue
behavior, but we still do not have hardware-counter proof for memory-vs-ALU
causes.

### Grid / Contact Sheet

`grid9_close2` is real and keeps CLIP input at `256x256`: lane 0 is a 3x3 grid
of nine `80x80` views with gutters, lanes 1-2 are full `256x256` close-ups.
`GRID_DIRECT_RASTER=1` avoids full-size scratch rendering for grid cells.

The missing gate is real-prompt, fixed-wall-clock comparison judged by full
per-view `256x256` teacher scores and 3x3 screenshots, not by the grid lane's
own loss.

### Cached Cadence

v14 rejected naive cached `dL/dimage`: fast but low teacher score. v15 tested a
smaller cached-step Adam LR:

| Variant | Mean Full-Teacher Cos | Step Rate |
| --- | ---: | ---: |
| `base` | 0.09413 | 17.26/s |
| `cache2` | 0.04559 | 29.49/s |
| `cache2lr50` | 0.05842 | 28.37/s |
| `cache4` | 0.02450 | 42.16/s |
| `cache4lr25` | 0.04000 | 43.32/s |

LR scaling helps without hurting speed, but it is not enough. The repeated
camera batch and stale objective semantics remain the bigger issues.

## Next Best Big Leaps

1. Real-prompt grid quality harness: fixed wall-clock, base3/full9/grid80,
   evaluated by full per-view teacher score and screenshots.
2. Rolling per-view gradient cache: cache gradients by camera view instead of
   replaying the same batch on cached steps.
3. Shape-gated rectangular pointwise tile fork for the hottest pointwise and
   `pw_bwd` shapes.
4. Selective pointwise f16 weights with f32 accumulation and strict input-grad
   gate.
5. Install/select full Xcode if hardware-counter profiling is needed before a
   deeper shader rewrite.
