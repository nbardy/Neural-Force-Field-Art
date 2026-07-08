# Recon brief — dynaworld fast-mac-gsplat (Metal) → WGSL port notes

Source: `~/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat` (read-only recon, 2026-07).
A Torch-first, Metal-backed **2D projected-Gaussian** rasterizer with a memory-lean
recompute backward. Key files, in reading order:

1. `csrc/metal/gsplat_fast_kernels.metal` — whole rasterizer in 396 lines:
   `count_tiles`, `emit_binned_ids`, `bitonic_sort_ids`, `tile_sort_render_forward`,
   `tile_sort_render_backward`. Port this first.
2. `csrc/metal/gsplat_metal.mm` — host dispatch order: count → cumsum → emit →
   forward → backward; buffer sizing + overflow guard.
3. `torch_gsplat_bridge_fast/rasterize.py` — global depth-argsort trick (gid order ==
   depth order, so per-tile sort is just integer-ID ascending), autograd wiring, meta packing.
4. `csrc/shared/common.h` — param/meta layout.
5. `docs/hardware_rasterizer_fast_backward_handoff.md` — the design bible: backward
   math contract, atomics strategy, simdgroup plan, memory budgets.
6. `variants/v6/csrc/metal/gsplat_v6_kernels.metal` — production kernel: GSP_CHUNK=64
   shared staging + simd_sum two-level reductions before atomics (256 pixels → 1 atomic).
   Reduction shape at lines ~582-603.
7. `variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin/...` — fixed-capacity bins,
   NO prefix sum, NO CPU readback (the WebGPU-friendly binning).
8. `variants/v12c_fused_raster_color_loss_backward/...` — fuses pixel-loss grad into
   raster backward (not applicable to CLIP loss; we keep dense grad_rgb input).
9. `src/train/train_optim.py` — plain fused Adam; NO densification/pruning exists in repo.

## Data model (theirs)
SoA f32: `means2d [G,2]`, `conics [G,3]` (inverse covariance a,b,c), `colors [G,3]`,
`opacities [G]`, `depths [G]` (sort key only, zero grad). Grads same-shape SoA,
zero-init then accumulate. They backprop into the CONIC directly; scale/rot Jacobian
upstream. `power = -0.5(a dx² + 2b dx dy + c dy²)`, `raw_α = op·exp(power)`,
`α = min(0.99, raw_α)`, visible iff `power ≤ 0 && α ≥ 1/255`.
Config: tile 16×16, max_tile_pairs 4096 (v6: 2048), transmittance cutoff 1e-4, eps 1e-8.

## Forward
1. `count_tiles` (thread/splat): support radius `tau = -2·log(alpha_threshold/opacity)`,
   snugbox AABB, then EXACT ellipse-vs-rect test per candidate tile → atomic count.
2. Prefix sum (host) — replaced by v11 fixedbin: `tile_offsets[t] = t·cap`, no readback.
3. `emit_binned_ids` (thread/splat): same test, atomic cursor → `binned_ids[slot]=gid`.
4. `tile_sort_render_forward` (1 workgroup/tile, thread/pixel): stage tile IDs in shared,
   bitonic-sort ascending (recovers depth order thanks to global argsort), write back
   sorted (backward skips re-sorting), then front-to-back composite:
   `w = T·α; accum += w·color; T *= 1-α; if (T < 1e-4) break;` v6 streams splats in
   64-chunk shared staging + whole-tile early-out via simd_sum(alive).
   Saves `tile_stop_counts[tile]` = max visible prefix length.

## Backward ("well-scoped")
Saves ONLY compact bins (counts/offsets/sorted ids/stop counts) — no dense G×H×W state.
Per pixel, two phases: (A) replay composite over the visible prefix (bounded by
tile_stop_counts) to recover `T_final` and stop index `end_i`; (B) walk BACK-TO-FRONT
with `T_prev = T_cur/max(1-α,eps)` reconstructing per-splat grads:
```
g_alpha  = T_prev·(dot(grad_out,color) - gT)
g_color += grad_out·(T_prev·α)
gate     = raw_α < max_alpha
g_power  = g_alpha·gate·raw_α
g_conic += g_power·(-0.5dx², -dx·dy, -0.5dy²)
g_mean  += -g_power·(-(a·dx+b·dy), -(b·dx+c·dy))       [sign per their kernel]
g_op    += g_alpha·gate·(raw_α/opacity)
gT       = α·dot(grad_out,color) + (1-α)·gT ;  T_cur = T_prev
```
`gT` = running "gradient of everything behind this splat" carry.
CRITICAL uniformity rule: barrier count identical across all lanes — per-pixel `end_i`
gates contributions only, never changes tile-wide control flow.
Accumulation: v6 reduces the 9 partials simd→threadgroup so ONE atomic add per splat
per tile (vs per pixel). Deferred segmented-reduction variant exists for contended scenes.

## Perf (Apple Silicon, from their docs)
64k splats fwd 9.6 ms @4K (dense torch ref: 95 s); fwd+bwd 27.5 ms @4K. Wins: tile-first
compute; recompute-backward (compact bins); forward writes sorted order back; 64-chunk
shared staging; two-level reduction before atomics; no CPU sync (v11).

## Metal → WGSL replacements (MUST address)
1. **No f32 atomics in WGSL.** Options: (a) v6 workgroup reduction + splat-major
   segmented reduction (atomic-free common path); (b) fixed-point `atomicAdd<i32>`
   with scale (range/overflow risk); (c) u32 CAS-loop bitcast (slow under contention).
2. **No guaranteed subgroups.** Replace simd_sum with workgroup-shared tree reductions
   (v6's partial[8] arrays show the fallback shape); feature-detect subgroups later.
3. **16 KB workgroup storage / 256 invocations.** Use v6 sizing: cap 2048 ids (8 KB)
   + chunk staging ~2.3 KB + 256 threads (one per pixel of a 16×16 tile). Rewrite the
   bitonic stride loops for 256 threads.
4. **No CPU readback/prefix-sum per frame:** v11 fixedbin constant-stride bins.
5. Bitonic sort ports fine (rewrite for 256 threads) or global radix sort of
   (tile,depth) keys — for OUR use case there is no depth at all: painter's order by
   splat index, so per-tile ascending-ID sort alone is sufficient and deterministic.
6. No texture tricks needed — plain storage buffers throughout.
