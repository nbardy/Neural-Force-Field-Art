# Sequential Convergence Quality Probe

Date: 2026-07-10

## Scope

Added `tools/splat3d/convergence_quality.ts` as an end-to-end quality gate for
the live convergence priors. No production files were changed by this worker.

## Comparison

The probe compares two optimizers sequentially on one WebGPU device:

- `base`: black training background, no convergence priors.
- `convergence`: weak ray distortion (`0.02`), screen-space mip annealing
  (`4.0 -> 0.0625` over 500 steps), and staged geometry/appearance learning
  rates. A procedural background can be enabled with `BACKGROUND_MODE`.

Both variants receive a copy of the same `randomSplats3D(G, SEED)` output and
the same nine real MobileCLIP text embeddings. The base optimizer is trained,
evaluated, destroyed, and drained before the convergence optimizer is created.
There are no parallel GPU runs.

## Quality Signal

After training, all nine fixed views are independently rendered and encoded by
MobileCLIP. The report includes:

- mean, minimum, maximum, and per-view image/text cosine;
- mean and p10-p90 sigmoid opacity;
- mean and p10-p90 world-space radius;
- RMS position spread around the unweighted cloud centroid;
- convergence-minus-base deltas;
- a 3x3 contact sheet for each variant and a JSON record.

The default is a small fixed-step gate (`G=1024`, `STEPS=24`, three sampled
views per step). Useful invocations:

```bash
bun tools/splat3d/convergence_quality.ts
STEPS=100 PROMPT="a carved jade owl" bun tools/splat3d/convergence_quality.ts
STEPS=100 BACKGROUND_MODE=blurred_noise bun tools/splat3d/convergence_quality.ts
WALL_MS=5000 bun tools/splat3d/convergence_quality.ts
```

Outputs default to `/tmp/nffa_convergence_quality` and can be redirected with
`OUT_DIR`.

## Smoke Result

The real-model Metal smoke passed:

```bash
STEPS=1 G=128 CAP=512 VIEWS=1 \
  OUT_DIR=/tmp/nffa_convergence_quality_smoke \
  bun tools/splat3d/convergence_quality.ts
```

It completed both sequential runs and all nine teacher evaluations. Base scored
`0.13918` mean / `0.11759` minimum cosine; convergence scored `0.13833` mean /
`0.11598` minimum cosine. These one-step, 128-splat values are only an execution
gate. Both contact sheets and `convergence_quality.json` were written.

## Interpretation

This is a sequential A/B gate, not a claim that one short prompt run establishes
general quality. Fixed-step mode isolates convergence per optimizer update;
fixed-wall mode includes the priors' runtime overhead. The minimum-view cosine
is especially useful because mean cosine can hide one collapsed camera view.
