# Grouped generator learning rates

## Goal

Implement only optional shared-hashgrid/head learning rates in the fused WebGPU generator trainer. Do not edit `src/main.ts` or `src/index.tsx`.

## Verified facts

- `FusedTrainer` previously uploaded one 32-byte pass-B uniform with one Adam learning rate.
- Packed `FieldLayout.segments` already identifies `grid`, `head`, and layer ranges.
- The worktree already contained an incomplete grouped-LR implementation in `train.ts` and `train_wgsl.ts`; this pass completed validation and normalization without touching app wiring.

## Changes

- `TrainStepOpts.learningRates` is optional and supports `uniform` (legacy `lr`) or `shared-heads` (`shared`, `head0`, `head1`).
- Added pure `resolveGeneratorLearningRates` validation/expansion; rates must be finite and non-negative.
- Expanded pass-B uniforms to 48 bytes and uploads four rates plus the existing Adam/batch fields.
- Generated WGSL selects `sharedLR` for grid segments, `head0LR`/`head1LR` for head segments, and falls back to `lr` for unclassified/padding ranges.

## Validation

- Pure normalization and invalid-rate rejection check: passed.
- Generated WGSL selector check: passed.
- `bun run build`: passed.
- `bun tools/train_test.ts`: passed all loss/gradient/Adam/K-step/particle/class/optimization checks.

## Unresolved

- Existing unrelated worktree changes remain; they were not reverted or folded into this task.
