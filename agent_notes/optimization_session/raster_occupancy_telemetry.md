# Raster Occupancy Telemetry

Date: 2026-07-08

## Question

After several exact raster scheduler gates failed to improve the integrated
default step, what is the rasterizer actually spending capacity on? Are we
overflowing tile bins, carrying an oversized cap, or walking long compositing
chains?

## Tool

Added:

```bash
tools/splat3d/raster_telemetry.ts
```

It renders selected views and reads:

- `tileCounts`: emitted tile/splat pairs, including pairs beyond cap;
- `tileStop`: max compositing stop index reached by any pixel in the tile.

The tool reports active-tile percentage, overflow tiles/pairs, count
percentiles, stop percentiles, and mean `stop/count`.

`tileStop` needed `COPY_SRC` usage so the tool could read it.

## Default Result

Command:

```bash
bun tools/splat3d/raster_telemetry.ts
```

Summary:

```text
G=4096 cap=2048
overflowPairs=0/576871 (0.0%)
maxCount=911
maxStop=365
```

View-level notes:

- all default views had `0` overflow tiles;
- count p99 stayed below `911`;
- stop p99 stayed below `365`;
- most views have high active tile coverage, so the whole screen is participating
  even though cap is not stressed.

## Stress Result

Command:

```bash
G=12000 CAP=2048 bun tools/splat3d/raster_telemetry.ts
```

Summary:

```text
overflowPairs=21440/1717219 (1.2%)
maxCount=2678
maxStop=702
```

Overflow starts mainly in the oblique high/low views. The front/side/top views
still stayed below cap in this random initialization.

## Decision

Default overflow handling is not the current bottleneck. The current `2048` cap
is conservative for the 4096-splat default scene. A smaller cap, especially
`1024`, is the next clean raster ablation because it may reduce workgroup memory
pressure without changing default outputs for the measured initial state.

Before promoting any smaller cap, run:

- default telemetry to prove no initial overflow;
- integrated `3/9 batch x3` matrix;
- at least one stress telemetry run to document the safety tradeoff.
