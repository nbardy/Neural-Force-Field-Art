# Full 3D Quality Checkpoint

Date: 2026-07-11

These artifacts preserve the quality evidence that originally lived in `/tmp`.

## 1024 Splats, 240 Steps

- `aniso_base_1024_step240.png`: anisotropic optimizer without geometry safeguards.
- `full3d_1024_step240.png`: full-3D default with weak bounds and anti-tiny regularization.
- `results_1024_step240.json`: prompts, per-view cosine, cloud telemetry, and timing.

Full 3D versus base:

- mean CLIP cosine: `+0.00754`;
- worst-view cosine: `+0.04074`;
- RMS cloud spread: `-11.3%`;
- throughput: `18.94` versus `19.01` steps/s.

## 4096 Splats, 60 Steps

- `full3d_4096_step60.png`: production-size early checkpoint.
- `results_4096_step60.json`: corresponding metrics and prompts.

The production-size full default improved mean cosine by `0.00840` and reduced
spread by `5.4%` versus its matched base in that run. A separate occupancy gate
measured `1231/4096` maximum tile count with zero overflow.

The original UI screenshots from temporary macOS capture paths were no longer
present at thread close. Their observations remain recorded in
`docs/BLOG_PROGRESS_NOTES.md`: center-square overflow, tiny-ball collapse,
opacity fading, and the later coherent cat contact sheets.
