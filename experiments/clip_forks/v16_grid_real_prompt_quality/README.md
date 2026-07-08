# v16 Grid Real-Prompt Quality

Date: 2026-07-08

## Hypothesis

`grid9_close2 + directgrid` is nearly as fast as the default `3/9` per-view
schedule while supervising all nine cameras through a contact-sheet lane plus
two full-resolution close-up lanes. The missing question was quality: does it
improve the full per-view MobileCLIP teacher score for real prompts?

This fork adds:

```bash
tools/splat3d/grid_quality.ts
```

It uses the real MobileCLIP text ONNX model to encode the same camera prompts
used by the browser page, trains each schedule from the same deterministic
initial splats, then evaluates every result with full-resolution per-view
`256x256` image embeddings. The grid lane's own prompt score is not the primary
metric.

## Snapshot

The `snapshot/` directory contains:

- `src/splat3d/optimize.ts`
- `src/splat3d/cameras.ts`
- `src/splat3d/grid_clip.ts`
- `tools/splat3d/cadence_quality.ts`
- `tools/splat3d/grid9_close2_test.ts`
- `tools/splat3d/step_bench.ts`
- `tools/splat3d/step_matrix.ts`
- `docs/SPLAT3D_PERF_NOTES.md`
- `experiments/clip_forks/README.md`

Diff helper:

```bash
node experiments/clip_forks/diff_fork.mjs v16_grid_real_prompt_quality
```

## Quality Measurement: Single Run

Command:

```bash
BUDGET_MS=5000 PROMPT="a photo of a cat" CONFIGS=base3=3:3,full9=9:3,grid80=9:3:grid9:directgrid OUT_DIR=/tmp/nffa_grid_quality_cat bun tools/splat3d/grid_quality.ts
```

Adapter: Apple `metal-3`, `apple-m4`, non-fallback.

Config: `G=4096`, `cap=2048`, `seed=1`, `viewSampler=epoch`, f32 weights,
black-background prompt suffix enabled.

Initial mean full-view teacher cosine: `0.12831`.

This was the first smoke-quality read, not the final decision number. It is kept
because it includes the `full9` reference and screenshot sheets, but the repeat
trial below is the cleaner answer for `base3` vs `grid80`.

| Variant | Training Path | Steps / 5s | Steps / Sec | Mean Cos | Mean Delta | Min Cos |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `base3` | per-view `3/9`, batch 3 | 96 | 19.11 | 0.25896 | 0.13066 | 0.22520 |
| `full9` | per-view `9/9`, batch 3 | 34 | 6.64 | 0.26863 | 0.14032 | 0.25341 |
| `grid80` | grid9+2, direct `80x80` grid raster | 86 | 17.09 | 0.23530 | 0.10699 | 0.21014 |

Artifacts:

- `results/2026-07-08/grid_quality.json`
- `results/2026-07-08/initial_views.png`
- `results/2026-07-08/base3_views.png`
- `results/2026-07-08/full9_views.png`
- `results/2026-07-08/grid80_views.png`

## Repeat Quality Gate

Command:

```bash
TRIALS=3 BUDGET_MS=5000 PROMPT="a photo of a cat" CONFIGS=base3=3:3,grid80=9:3:grid9:directgrid OUT_DIR=/tmp/nffa_grid_quality_cat_trials bun tools/splat3d/grid_quality.ts
```

The harness rotates config order across trials and uses seeds `1..3`. This is
still only one prompt, but it is less sensitive to same-session GPU contention
than a single run.

Artifacts:

- `results/2026-07-08/grid_quality_trials.json`
- `results/2026-07-08/trials/initial_t0_views.png`
- `results/2026-07-08/trials/base3_t0_views.png`
- `results/2026-07-08/trials/grid80_t0_views.png`

Median summary:

| Variant | Trials | Steps / 5s | Steps / Sec | Mean Cos | Mean Delta | Min Cos | Min Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base3` | 3 | 91 | 18.02 | 0.25410 | 0.12579 | 0.20980 | 0.09576 |
| `grid80` | 3 | 78 | 15.55 | 0.23802 | 0.11083 | 0.19165 | 0.08511 |

Read:

- `grid80` reached `88.1%` of `base3`'s median mean-cosine improvement.
- `grid80` reached `88.9%` of `base3`'s median min-cosine improvement.
- `grid80` reached `93.7%` of `base3`'s final median mean cosine.
- `grid80` ran at `86.3%` of `base3`'s step rate in this repeated quality run.

## Speed Measurement

Command:

```bash
TRIALS=3 CONFIGS=base3=3:3,full9=9:3,grid256=9:3:grid9,grid80=9:3:grid9:directgrid RUNS=7 WARMUP=5 bun tools/splat3d/step_matrix.ts
```

| Variant | Normal Median | Profile Median | CLIP Median | Raster Median |
| --- | ---: | ---: | ---: | ---: |
| `base3` | 46.55 ms | 48.60 ms | 34.91 ms | 11.32 ms |
| `full9` | 137.52 ms | 144.15 ms | 106.29 ms | 33.62 ms |
| `grid256` | 78.46 ms | 80.73 ms | 35.23 ms | 42.31 ms |
| `grid80` | 52.91 ms | 56.17 ms | 35.37 ms | 17.71 ms |

## Decision

Keep `grid80` gated. It is a strong speed candidate, and the repeat cat prompt
gate says the quality tradeoff is moderate rather than catastrophic. It still
does not beat the default `base3` schedule on full per-view teacher score.

What we learned:

- `grid80` is `2.60x` faster than full per-view `9/9` by median step time.
- In the step matrix, `grid80` is only about `14%` slower than base `3/9` while
  touching all nine cameras each step.
- In the repeated 5s quality gate, `grid80` reached `88.1%` of `base3`'s median
  mean-cosine improvement and `93.7%` of its final median mean cosine.

Next grid work should test multiple prompts/seeds and close-up policy before
promoting. The promising follow-up is not more raster speed; it is better grid
signal: prompt wording, randomized close-up lanes, and possibly one periodic
full per-view refresh.
