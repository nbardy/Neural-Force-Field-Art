# Blog Progress Notes

## Working Checkpoint

- 2D prompt-to-splats page works with WebGPU CLIP guidance and a `NUDGE` button.
- 3D fork renders one shared 3D splat cloud from nine fixed cameras in a 3x3 grid.
- 3D optimizer accumulates gradients from all nine camera views, then applies one shared Adam update.
- Prompt modes:
  - `camera text`: appends a natural camera-angle phrase per view.
  - `same text`: uses the same base text for all views.
- Black-background prompt text is a toggle:
  - first setting to try: append `on a black background`.
  - toggle off: no black-background wording, while the renderer still uses a black background.
- Current camera prompt style is natural language, e.g. `a camera angle from the right side`, not internal labels like `front-left-high`.

## Screenshot Plan

Capture these as the system gets better:

1. 2D page before optimization: random initial splats.
2. 2D page after early CLIP optimization: first recognizable structure.
3. 2D `NUDGE`: before/after showing a partial rerandomization without full reset.
4. 3D page at boot: 3x3 camera grid with random shared splats.
5. 3D page after a short run: same object beginning to appear across multiple views.
6. 3D page after a longer run: best current multi-view result.
7. Prompt toggle comparison: `black bg` on vs `no bg text`.
8. Prompt mode comparison: `camera text` vs `same text`.

## Blog Story Beats

- Start with the 2D CLIP-over-splats idea: explicit pixels, no diffusion, no CPU readback.
- Show the `NUDGE` button as an artist-facing control for escaping local visual basins.
- Explain the 3D leap: one shared splat cloud, nine camera views, one joint optimization step.
- Show why the grid matters: the viewer can see whether the object is actually 3D-consistent.
- Call out the first prompt/background hypothesis: black renderer background plus optional `on a black background` CLIP text.
- Leave ablations for later; this checkpoint is about making the system visible and testable.

## Next Plan Notes

- Do not start broad ablations yet.
- First manual tests should focus on screenshots and qualitative behavior:
  - Does `on a black background` help object formation?
  - Does `camera text` make views more distinct or more stable?
  - Does `same text` produce a more coherent shared object?
  - Does the 3x3 grid show collapse/fog/hole failures clearly?
- Likely next engineering knobs after screenshots:
  - view subset per step (`1`, `3`, `9`) for speed;
  - zoom-out/object scale control;
  - geometry-first learning-rate schedule;
  - alpha/coverage regularizer;
  - better 2DGS/surfel-style parameterization if volumetric blobs stay incoherent.
