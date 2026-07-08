# CLIP Timestamp Dispatch Profile

Date: 2026-07-08

## Question

Can we measure true GPU time for individual CLIP WGSL dispatches instead of
only warmed split-submit wall time?

## Change

`tools/clip/dispatch_profile.ts` now accepts `TIMESTAMP=1`.

When the WebGPU adapter exposes `timestamp-query`, the tool requests that device
feature and wraps each isolated dispatch in a begin/end timestamp pair. It then
resolves the query into a readback buffer and reports the same top-dispatch and
group tables as before. If timestamp queries are unavailable, the profiler keeps
the old split-submit wall-time fallback.

## Commands

```bash
TIMESTAMP=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=1 bun tools/clip/dispatch_profile.ts
```

The local adapter reported `apple metal-3` with `timestamp-query` support.

## Result

Current promoted B=3 CLIP settings:

```text
TIMESTAMP=1 STEM_SPATIAL_BWD=1 FUSE_PW_GELU=1 MODE=train BATCH=3 RUNS=1 WARMUP=1
```

Top groups by isolated GPU timestamp sum:

| Group | Time | Share |
| --- | ---: | ---: |
| `pw_bwd` | `42.01 ms` | `26.2%` |
| `spatial_bwd` | `30.08 ms` | `18.8%` |
| `conv` | `25.10 ms` | `15.7%` |
| `pw+gelu` | `23.92 ms` | `14.9%` |
| `pw` | `23.20 ms` | `14.5%` |
| `gelu_bwd` | `5.51 ms` | `3.4%` |
| `attn_core_bwd` | `4.33 ms` | `2.7%` |

Total isolated timestamp sum was `160.30 ms`.

## Interpretation

This does not mean the full optimizer step spends exactly `160 ms` in CLIP; the
tool profiles each dispatch in isolation. It is still the right ranking tool for
kernel work.

The timestamp path confirms the existing direction:

- pointwise backward/forward, spatial backward, and conv-family kernels remain
  the first CLIP optimization targets;
- attention backward is visible but too small to be the first 4x lever;
- future CLIP shader rewrites should be chosen with `TIMESTAMP=1` plus an
  integrated `step_matrix.ts` timing gate.
