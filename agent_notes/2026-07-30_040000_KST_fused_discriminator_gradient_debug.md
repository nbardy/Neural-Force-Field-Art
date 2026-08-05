# Fused discriminator gradient debug

## Goal

Independently identify why the production fused adversary discriminator
gradients disagree with the TF.js reference for:

- soft angle,
- angle + relative scale,
- angle + scale hold, and
- post-update velocity,

while generator `extGrad` and encoded `u` / target `y` already match.

## Constraints

- Inspect before editing.
- Focus on `src/render/webgpu/adversary_wgsl.ts`,
  `src/render/webgpu/adversary_train.ts`,
  `src/core/gan/adversary.ts`, and `tools/train_wta_test.ts`.
- Check sign, tensor/packed layout, batch means, WTA tape constants, and target
  construction.

## Status

Investigation started. No source edits yet.

