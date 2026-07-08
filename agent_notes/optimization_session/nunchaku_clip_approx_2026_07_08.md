# CLIP Approximation / Nunchaku-Style Read

Date: 2026-07-08

## Question

Can we get another 2-4x by borrowing ideas from Nunchaku/SVDQuant-style low-bit
diffusion inference, Hessian/sensitivity approximations, layer skipping, or
learned approximations of less important CLIP layers?

## Current Read

The useful idea is not "make all of CLIP int4" as a first browser move. WGSL has
no native int4 tensor-core path, so packed int4 can easily trade memory bandwidth
for unpacking, scalarization, and worse shader occupancy. Direct int4 is worth a
fork only if a microbench shows packed matmul wins on Apple Metal through Dawn.

The more applicable Nunchaku/SVDQuant pattern is:

- keep the sensitive/outlier part in a small higher-precision side branch;
- compress or approximate the bulk path;
- fuse the side branch with the main path so activation traffic does not erase
  the win;
- judge by task-level teacher alignment, not just kernel speed.

For this repo, the target is a frozen MobileCLIP image tower used as an
optimization teacher. That means approximation is allowed only if it preserves
both:

- image embedding cosine versus the full teacher;
- input-gradient direction, because the splats are optimized through
  `dL/dimage`.

## Most Promising Forks

1. **CLIP proxy ladder**

   Build a harness that runs the same image batch through full CLIP and cheaper
   proxy variants, then reports embedding cosine and `dL/dimage` cosine. Test
   truncating late blocks, skipping selected residual blocks, and lower
   resolution internal stems. This is the fastest way to find whether layer
   skipping is real or wishful.

2. **Periodic teacher / cheap inner loop**

   Use a cheap CLIP proxy for several inner steps, then refresh with full CLIP.
   This is the safer version of cached gradients: the proxy still reacts to the
   current image, while the full teacher corrects drift.

3. **Low-rank pointwise approximation**

   Pointwise forward/backward remain major CLIP buckets. Try SVD/low-rank
   approximations for selected `1x1`/linear weights and gate by embedding and
   gradient cosine. This is likely more WebGPU-friendly than packed int4 because
   it can stay in ordinary f32/f16 matmul kernels.

4. **Selective quantization, not global quantization**

   The prior f16 weight path already showed that precision changes need real
   quality gates. A better fork would quantize only pointwise weights or only
   layers with low gradient sensitivity, leaving stem/spatial and projection
   layers high precision.

5. **Distilled browser proxy**

   Train a small WebGPU-native image encoder on rendered splat/contact-sheet
   images to match full MobileCLIP embeddings and input gradients. This is more
   work, but it is the most plausible route to a real 2-4x inner-loop speedup
   without relying on hardware int4.

## Gate Before Promotion

Do not promote any approximation based on speed alone. Required checks:

```bash
MODE=train BATCH=3 RUNS=3 WARMUP=1 bun tools/clip/dispatch_profile.ts
CLIP_PROXY=... bun tools/clip/proxy_quality_gate.ts
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat" CONFIGS=base3=3:3,proxy=3:3:proxy bun tools/splat3d/grid_quality.ts
```

`proxy_quality_gate.ts` does not exist yet. It should compare:

- embedding cosine versus full MobileCLIP;
- input-gradient cosine versus full MobileCLIP;
- gradient norm ratio;
- integrated 3D convergence in a fixed wall-clock budget.

## Decision

There may be room for a 2-4x CLIP-side speedup, but it probably comes from a
teacher/proxy schedule or low-rank/skip approximation with strict gradient
gates, not from a blind int4 rewrite. The next concrete code fork should be the
proxy ladder harness before touching main CLIP shaders.
