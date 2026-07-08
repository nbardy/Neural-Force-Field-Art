/**
 * 3D raster occupancy telemetry.
 *
 * Reads tileCounts and tileStop after real raster forwards so the next raster
 * optimization is driven by overflow/occupancy data instead of scheduler
 * guesses.
 *
 *   bun tools/splat3d/raster_telemetry.ts
 *   G=12000 CAP=2048 VIEWS=0,1,2 bun tools/splat3d/raster_telemetry.ts
 *   JSON=1 bun tools/splat3d/raster_telemetry.ts
 */
import { setupGlobals } from "bun-webgpu";
import { DEFAULT_3D_CAMERAS, prepareCamera } from "../../src/splat3d/cameras";
import { LEGIBLE_3D_G, LEGIBLE_3D_INIT, randomSplats3D } from "../../src/splat3d/optimize";
import { Raster3DEngine } from "../../src/splat3d/raster";

setupGlobals();

const U = { MAP_READ: 1, COPY_DST: 8 };
const SIDE = 256;
const G = Number(process.env.G ?? LEGIBLE_3D_G);
const CAP = Number(process.env.CAP ?? 2048);
const SEED = Number(process.env.SEED ?? 1);
const JSON_OUT = process.env.JSON === "1";
const VIEWS = parseViews(process.env.VIEWS, DEFAULT_3D_CAMERAS.length);

interface ViewTelemetry {
  view: number;
  name: string;
  tiles: number;
  nonEmptyTiles: number;
  activeTilePct: number;
  overflowTiles: number;
  overflowTilePct: number;
  overflowPairs: number;
  totalPairs: number;
  keptPairs: number;
  droppedPairPct: number;
  countMax: number;
  countMean: number;
  countP50: number;
  countP90: number;
  countP95: number;
  countP99: number;
  stopMax: number;
  stopMean: number;
  stopP50: number;
  stopP90: number;
  stopP95: number;
  stopP99: number;
  stopToCountMean: number;
}

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: GPUDevice = await adapter.requestDevice();
const info = adapter.info ?? {};
if (!JSON_OUT) console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

const cameras = DEFAULT_3D_CAMERAS.map((camera) => prepareCamera(camera, SIDE));
const raster = await Raster3DEngine.create(device, {
  H: SIDE,
  W: SIDE,
  G,
  cap: CAP,
  cameras,
  bg: [0, 0, 0],
});
raster.setParams(randomSplats3D(G, SEED, LEGIBLE_3D_INIT));

const rows: ViewTelemetry[] = [];
for (const view of VIEWS) {
  raster.runForward(view);
  await device.queue.onSubmittedWorkDone();
  const counts = await readU32(raster.tileCounts, raster.dims.numTiles);
  const stops = await readU32(raster.tileStop, raster.dims.numTiles);
  rows.push(summarize(view, raster.cameras[view]?.name ?? `view-${view}`, counts, stops, CAP));
}
raster.destroy();

if (JSON_OUT) {
  console.log(JSON.stringify({ G, cap: CAP, seed: SEED, side: SIDE, views: rows }, null, 2));
} else {
  console.log(`splat3d raster telemetry: G=${G} cap=${CAP} seed=${SEED} side=${SIDE} views=${VIEWS.join(",")}`);
  console.log(
    "view              active overflow dropped pairs  count p50/p90/p99/max      stop p50/p90/p99/max       stop/count"
  );
  for (const r of rows) {
    const countSummary = `${fmt0(r.countP50)}/${fmt0(r.countP90)}/${fmt0(r.countP99)}/${fmt0(r.countMax)}`.padEnd(24);
    const stopSummary = `${fmt0(r.stopP50)}/${fmt0(r.stopP90)}/${fmt0(r.stopP99)}/${fmt0(r.stopMax)}`.padEnd(24);
    console.log(
      `${r.name.padEnd(17).slice(0, 17)} ` +
        `${pct(r.activeTilePct).padStart(6)} ` +
        `${String(r.overflowTiles).padStart(4)} ${pct(r.overflowTilePct).padStart(7)} ` +
        `${pct(r.droppedPairPct).padStart(7)} ` +
        `${String(r.totalPairs).padStart(7)} ` +
        `${countSummary} ${stopSummary} ${r.stopToCountMean.toFixed(3).padStart(9)}`
    );
  }
  const totalOverflow = rows.reduce((sum, r) => sum + r.overflowPairs, 0);
  const totalPairs = rows.reduce((sum, r) => sum + r.totalPairs, 0);
  const maxCount = Math.max(...rows.map((r) => r.countMax));
  const maxStop = Math.max(...rows.map((r) => r.stopMax));
  console.log(
    `aggregate: overflowPairs=${totalOverflow}/${totalPairs} (${pct(totalPairs ? totalOverflow / totalPairs : 0)}), ` +
      `maxCount=${maxCount}, maxStop=${maxStop}`
  );
}

async function readU32(buffer: GPUBuffer, words: number): Promise<Uint32Array> {
  const staging = device.createBuffer({ size: words * 4, usage: U.MAP_READ | U.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, staging, 0, words * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

function summarize(view: number, name: string, counts: Uint32Array, stops: Uint32Array, cap: number): ViewTelemetry {
  const countVals = Array.from(counts);
  const stopVals = Array.from(stops);
  let nonEmptyTiles = 0;
  let overflowTiles = 0;
  let overflowPairs = 0;
  let totalPairs = 0;
  let keptPairs = 0;
  let countSum = 0;
  let stopSum = 0;
  let stopToCountSum = 0;
  for (let i = 0; i < counts.length; i++) {
    const count = counts[i];
    const stop = stops[i];
    countSum += count;
    stopSum += stop;
    totalPairs += count;
    keptPairs += Math.min(count, cap);
    if (count > 0) {
      nonEmptyTiles += 1;
      stopToCountSum += stop / count;
    }
    if (count > cap) {
      overflowTiles += 1;
      overflowPairs += count - cap;
    }
  }
  return {
    view,
    name,
    tiles: counts.length,
    nonEmptyTiles,
    activeTilePct: nonEmptyTiles / Math.max(1, counts.length),
    overflowTiles,
    overflowTilePct: overflowTiles / Math.max(1, counts.length),
    overflowPairs,
    totalPairs,
    keptPairs,
    droppedPairPct: totalPairs ? overflowPairs / totalPairs : 0,
    countMax: Math.max(...countVals),
    countMean: countSum / Math.max(1, counts.length),
    countP50: percentile(countVals, 0.5),
    countP90: percentile(countVals, 0.9),
    countP95: percentile(countVals, 0.95),
    countP99: percentile(countVals, 0.99),
    stopMax: Math.max(...stopVals),
    stopMean: stopSum / Math.max(1, counts.length),
    stopP50: percentile(stopVals, 0.5),
    stopP90: percentile(stopVals, 0.9),
    stopP95: percentile(stopVals, 0.95),
    stopP99: percentile(stopVals, 0.99),
    stopToCountMean: nonEmptyTiles ? stopToCountSum / nonEmptyTiles : 0,
  };
}

function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx];
}

function parseViews(src: string | undefined, total: number): number[] {
  if (!src || src.trim() === "" || src.trim().toLowerCase() === "all") {
    return Array.from({ length: total }, (_unused, i) => i);
  }
  const views = src
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isInteger(n) && n >= 0 && n < total);
  if (!views.length) throw new Error(`raster_telemetry: VIEWS='${src}' produced no valid views`);
  return views;
}

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

function fmt0(x: number): string {
  return Math.round(x).toString();
}
