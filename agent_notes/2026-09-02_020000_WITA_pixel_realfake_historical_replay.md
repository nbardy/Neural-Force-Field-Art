# Historical Pixel RealFake replay

## Goal

Try a Pixel RealFake critic that can distinguish the current population from
prior generated populations, rather than training only on the latest cloud
versus uniform random positions.

## Design decision

Use detached `G×G` density snapshots, not particle positions. The critic
already consumes rasterized densities, and storing positions would add another
splat path without preserving the exact example seen by the critic. Keep the
history on the GPU to avoid readback/upload stalls.

The replay is intentionally bounded: an actually unbounded GPU population
would eventually exhaust browser/device memory. The current named trial keeps
256 snapshots, captures every 4 critic steps, and uses historical negatives on
75% of steps. It is a rolling replay population, not a claim of theoretical
unbounded fictitious play.

## Implementation

- `PixelCriticSpec.historicalReplay` declares capacity, capture cadence, and
  replay probability.
- `PixelDiscTrainer` owns a GPU ring buffer of detached density snapshots.
- The current real density is captured before the discriminator pass.
- An older snapshot is copied into the fake density region on replay hits;
  otherwise the existing uniform fake path runs unchanged.
- Generator training remains current and differentiable: historical replay is
  discriminator-only and never becomes a gradient source.
- Added `Pixel · RealFake · Historical` as a named gallery trial.

## Risks and follow-up

- Historical generated examples are labelled negative relative to the current
  population; this can reward perpetual novelty and should be A/B tested.
- A finite ring is recent-history replay, not unbounded history. Larger
  capacities should be chosen from measured device memory, not blindly raised.
- Add replay fill/hit/age telemetry before treating the trial as a final
  gallery piece.

## Validation

- `npm run build -- --no-cache` passed.
- `bun tools/gallery_config_test.ts` passed.
- `bun tools/pixel_disc_test.ts` passed, including GPU smoke.
