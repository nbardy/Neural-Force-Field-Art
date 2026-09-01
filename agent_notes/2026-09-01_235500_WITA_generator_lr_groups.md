# Generator grouped learning rates

## Goal

Add optional grouped learning rates to the fused WebGPU generator trainer,
without editing `src/main.ts` or `src/index.tsx`.

## Verified facts

- `FusedTrainer` previously uploaded a 32-byte pass-B uniform block with one
  `lr` field.
- `FieldLayout.segments` already identifies packed `grid` segments and head
  numbers 0/1.
- WGSL pass B applies Adam once per packed weight in `applyStep`.

## Changes

- Added `GeneratorLearningRates` and optional `TrainStepOpts.learningRates`.
- Expanded the pass-B uniform block to 48 bytes and uploads shared/head rates;
  legacy `lr` and the `uniform` variant fill all group rates equally.
- Generated a packed segment-range selector in `train_wgsl.ts`.
- Added pure codegen coverage in `tools/train_lr_groups_test.ts` and indexed it
  in `AGENTS.md`.

## Checks

- `bun tools/train_lr_groups_test.ts` — PASS; all generated segment routes
  select the intended rate.
- `bun tools/train_test.ts` — PASS; existing loss, gradient, Adam, BPTT,
  particle-source, classes, and convergence checks passed.
- `npm run build` — PASS; Parcel produced all three application bundles.
- `yarn build` was unavailable because `yarn` is not installed; npm was used
  with the same package script.
