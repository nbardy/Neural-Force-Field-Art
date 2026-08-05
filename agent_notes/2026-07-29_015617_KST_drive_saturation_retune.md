# Adversary physical-drive saturation audit and retune

## Goal

Explain and fix the observed failure where adversary pieces look best for the
first 2–3 seconds and then particle velocity pins the componentwise
`maxVelocity` clip. Add honest live controls for the physical drive and both
sides' learning rates without changing legacy/non-adversary physics.

## Verified diagnosis

The physical integrator is componentwise:

```text
v[t+1] = clamp(friction * (v[t] + forceMagnitude * F[t]), ±maxVelocity)
```

For a constant raw field component, the unclipped steady state is:

```text
v* = friction * forceMagnitude * F / (1 - friction)
```

Define:

```text
drive = friction * forceMagnitude / ((1-friction) * maxVelocity)
```

Then `|F| >= 1/drive` is sufficient to pin a component. Before this retune the
then-six adversary presets had (the labelled-quad piece was added afterward):

| Piece | old drive | raw |F| clip threshold | fresh-init components above threshold |
|---|---:|---:|---:|
| Single | 13.50 | 0.074 | 69.7% |
| WTA K=8 | 15.26 | 0.066 | 81.7% |
| Pair | 11.66 | 0.086 | 74.0% |
| Tri | 11.66 | 0.086 | 72.5% |
| Agree+Disagree | 14.40 | 0.069 | 84.4% |
| Chaos Weave | 13.33 | 0.075 | 84.6% |

Measurement: five fresh field initializations per piece, 20,000 uniform points
per initialization, tfjs CPU forward. This demonstrates that saturation was
already strongly favored by the physical presets; it did not require the
strict adversarial reward to observe velocity or clipping.

A second dynamic CPU probe used 768 particles for 180 frames through the
actual initialized spatial field. Representative old presets reached mean
speed 18.9–21.5 px/frame with 17–20% of components exactly clipped by frame
180. The requested conservative `.97` friction retune removed all measured
clipping.

## Implemented invariant

For opted-in adversary pieces:

```text
forceMagnitude = drive * maxVelocity * (1-friction) / friction
```

If the tanh field satisfies `|F[t]| <= 1` and
`|v[0]| <= drive*maxVelocity`, induction gives:

```text
|v[t+1]|
  <= friction * (drive*maxVelocity + forceMagnitude)
   = drive*maxVelocity
```

Thus defaults with `drive < 1` cannot touch the component clip, regardless of
how adversarial training changes a bounded field.

Shipped adversary defaults, all at `friction = 0.97`:

| Piece | drive | forceMagnitude |
|---|---:|---:|
| Single | 0.55 | 0.374226804 |
| WTA K=8 | 0.65 | 0.522680412 |
| Pair | 0.65 | 0.482474227 |
| Tri | 0.65 | 0.482474227 |
| Quad labelled | 0.65 | 0.482474227 |
| Agree+Disagree | 0.60 | 0.408247423 |
| Chaos Weave | 0.75 | 0.603092784 |

Legacy/non-adversary pieces do not declare `drive`; their literal historical
`forceMagnitude` is unchanged and `?drive` is ignored.

## Live control contract

- Top-right adversary config now has `drive` in `[0,1]`.
- `?drive=` is canonicalized once with the gallery config.
- Changing `maxVelocity` preserves drive by recomputing `forceMagnitude`.
- `AdvectKernel.setForceMagnitude()` writes the live physics uniform.
- Fused field/adversary trainer physics and tfjs `physicsForward` receive the
  same live value every frame.
- Generator LR (`?gLR`, default piece `learningRate`) and discriminator LR
  (`?dLR`, default `0.003`) have separate logarithmic controls and a D/G ratio.
- The top-right `train B` value now drives the fused adversary batch too
  (`min(sampleRate, 1024)`); it is no longer silently hard-coded to 512.
- Fused Adam keeps its moments and reads live LR uniforms.
- Tfjs has no supported Adam LR setter, so changing a tfjs LR rebuilds only the
  relevant optimizer. Field/predictor weights remain bit-identical; Adam
  moments deliberately restart.
- Agree+Disagree retains named nonnegative roles. LR changes do not encode role
  sign and the display-only C blend still owns no loss.

## Files changed

- `src/main.ts`
- `src/index.tsx`
- `src/render/webgpu/advect.ts`
- `src/core/gan/adversary.ts`
- `tools/drive_controls_test.ts` (new)

No splat production/test files were touched.

## Verification

- `bun tools/drive_controls_test.ts` — all checks pass.
  - exact seven defaults
  - algebraic recurrence bound
  - 10,000-step force-sequence gates
  - URL + gallery re-resolution
  - maxVelocity/drive parity
  - AdvectKernel live uniform setter
  - tfjs discriminator LR update preserves predictor weights bit-for-bit
- `bun tools/adversary_wire_test.ts` — all checks pass.
- `npm run build` — succeeds after the full wiring.
- `git diff --check` on owned files — clean.

The long `tools/adversary_test.ts` was stopped during §6 at the coordinator's
request because it is redundant with the root agent's final broad suite; every
completed section through §5 passed.

## Caveats and completed acceptance gates

- Reducing drive or maxVelocity live does not teleport/rescale an already-fast
  velocity buffer. The recurrence brings it into the new invariant
  geometrically (factor `friction` per frame); fresh starts begin at zero and
  satisfy the bound immediately.
- The proof is componentwise, matching the shader clip. HUD mean speed is
  Euclidean and may exceed one component's `maxVelocity` without clipping
  (`|v| <= sqrt(2)*maxVelocity`).
- The final browser soak asserted, for each adversary piece over more than
  1,800 frames:
  - no NaN/Infinity/page error;
  - no component-clip telemetry if exposed;
  - mean Euclidean speed remains materially below
    `sqrt(2)*drive*maxVelocity`;
  - changing drive updates motion without changing surprise target units;
  - maxVelocity changes keep the displayed drive fixed;
  - G LR, D LR, and D/G ratio update independently.

All seven modes passed. Trailing mean-speed/clip ranged from 0.328 to 0.442,
with no NaN/Infinity, page/console error, tensor drift or confirmed predictor
pileup. The exact run is
`output/playwright/adversary-soak/2026-07-28T17-38-54-170Z/`.
