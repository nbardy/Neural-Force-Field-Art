/**
 * Isolates the old per-splat opacity sparsity term from CLIP.
 *
 *   bun tools/splat3d/opacity_decay_probe.ts
 */
import { setupGlobals } from "bun-webgpu";
import { DEFAULT_3D_CAMERAS, prepareCamera } from "../../src/splat3d/cameras";
import { Raster3DEngine, type Raster3DRegularizerOptions } from "../../src/splat3d/raster";
import { randomSplats3D } from "../../src/splat3d/optimize";

setupGlobals();

const G = 256;
const SIDE = 32;
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("opacity_decay_probe: no WebGPU adapter");
const device: GPUDevice = await adapter.requestDevice();
const raster = await Raster3DEngine.create(device, {
  H: SIDE,
  W: SIDE,
  G,
  cap: 256,
  cameras: [prepareCamera(DEFAULT_3D_CAMERAS[0], SIDE)],
  bg: [0, 0, 0],
});
raster.setParams(randomSplats3D(G, 1));
raster.zeroAdamState();

const regularizer: Raster3DRegularizerOptions = {
  centerWeight: 0,
  radiusWeight: 0,
  targetRadius: 1.15,
  opacitySparsity: 0.01,
  smallRadiusWeight: 0,
  smallRadius: 0.024,
  radiusBandWeight: 0,
  minRadius: 0.016,
  maxRadius: 0.16,
};

console.log(`step 0:   mean opacity ${(await meanOpacity()).toFixed(6)}`);
let completed = 0;
for (const target of [100, 500, 800]) {
  for (let step = completed + 1; step <= target; step++) {
    const enc = device.createCommandEncoder();
    raster.recordClearRawGrad(enc);
    raster.recordRegularizerAdd(enc, regularizer);
    raster.recordAdam(enc, step, { position: 0, logRadius: 0, color: 0, opacity: 0.03 });
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  completed = target;
  console.log(`step ${String(target).padEnd(3)}: mean opacity ${(await meanOpacity()).toFixed(6)}`);
}
raster.destroy();

async function meanOpacity(): Promise<number> {
  const params = await raster.readParams();
  let sum = 0;
  for (let g = 0; g < G; g++) {
    const raw = params[7 * G + g];
    sum += raw >= 0 ? 1 / (1 + Math.exp(-raw)) : Math.exp(raw) / (1 + Math.exp(raw));
  }
  return sum / G;
}
