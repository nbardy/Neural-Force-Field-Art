# Optimization Subagent Rollout Review

Date: 2026-07-08

## What We Asked For

Five agents investigated aggressive CLIP and raster speedups from different
angles: AlphaGOJS-style GPU lessons, FUSED_js/FUSED_JS_2, the current CLIP batch
fork, browser/runtime options, and dynaworld/FasterGS-style renderer lessons.

## What Worked

- The agents converged on the same first production bet: integrate the existing
  `BatchMajorVisionTrainer` into the 3D optimizer before rewriting kernels.
- The notes separated "algorithmic less work" from "kernel faster work".
  N-of-K view sampling is already a real wall-time lever, while batch-major CLIP
  is the highest-confidence way to make multi-view CLIP less linear.
- Several notes correctly warned that plausible GPU rewrites can lose. The
  shared-W pointwise microbench already shows this: some B=2 expansion cases win,
  B=3 often loses or is flat, and contraction layers are mixed.
- The raster notes aligned with our measured profile. Raster should improve, but
  CLIP is still the dominant sampled wall-time share, so raster rewrites should
  be gated by measurement.

## What Did Not Work

- The five-agent rollout had too much overlap. Agents 1, 2, 3, 4, and 5 all
  independently promoted batch-major CLIP and dispatch profiling. That agreement
  is useful, but it means the prompt under-specified non-overlapping lanes.
- Some ideas were too broad to implement directly: "f16", "fusion", and
  "attention bwd" each need a smaller acceptance gate before code changes.
- The notes mix app-level speed, CLIP-only speed, and kernel-local speed. Future
  prompts should force each agent to label whether a proposal affects optimizer
  wall time, CLIP batch wall time, isolated dispatch time, memory, or quality.
- The rollout did not initially require a reject/kill criterion for every idea.
  That made the backlog larger than necessary.

## Consensus Ranking

1. Integrate batch-major CLIP into the 3D optimizer behind a toggle.
2. Add CLIP dispatch-level profiling grouped by label and shape.
3. Add raster/CLIP buffer aliasing after the batch path works.
4. Promote shared-W pointwise only by shape and batch size after profiling.
5. Stage or specialize `spatial_bwd` if timestamps put it near the top.
6. Try f16 weights with f32 accumulation behind a feature gate.
7. Consider GELU/pointwise fusion only if dispatch and slot traffic show up.
8. Treat attention backward rewrites as conditional, not assumed.

## Iteration Outcome

The rollout was useful once the recommendations were forced through the
commit-and-measure loop:

- Batch-major CLIP, raster/CLIP buffer aliasing, per-lane raster forward state,
  and default `batch CLIP x3` all landed.
- Dispatch profiling landed and directly selected the stem `spatial_bwd` kernel
  as the first non-pointwise shader target.
- Shared-W pointwise forward was implemented behind a gate but not promoted,
  because isolated microbench wins did not survive full-chain B=2/B=3 train
  timing.
- Generic spatial-backward weight staging was rejected and recorded after
  correctness passed but integrated timing failed to prove a win.
- Stem spatial backward specialization was promoted for 3D batch CLIP.
- Pointwise + GELU forward fusion was promoted for 3D batch CLIP after it
  reduced B=3 train median and integrated CLIP split time.
- STAR UVT/world-tube review clarified that exact multi-view raster batching is
  a scheduler problem first; sublinear camera-bundle rendering needs a different
  primitive.

The highest-value process lesson remains: every attractive GPU idea needs both
a local correctness gate and a full-chain optimizer timing gate. The shared-W
attempt was the clearest example of a plausible microkernel that should remain
gated instead of becoming default.

## Process Rule Going Forward

Each performance attempt should be one commit when it lands, and failed attempts
should be recorded in the ablation queue with the command, metric, and reason for
rejecting or reverting. If an attempt changes runtime behavior, measure it in the
full optimizer path before calling it a win.

For future subagent rollouts, assign narrower roles:

- one agent owns app-level integration and correctness gates;
- one owns CLIP dispatch profiling and kernel ranking;
- one owns raster state/buffer scheduling;
- one owns precision and browser memory constraints;
- one owns negative controls and rejection criteria.

That should produce less repeated advice and more directly actionable patches.
