/**
 * Gate 2 — L2 image-fit convergence demo (end-to-end, no CLIP). 4096 splats @
 * 128x128 fit a deterministic synthetic target (smooth 3-colour gradient + a
 * filled circle) by Adam. dL/dpixels = 2(render - target)/N is fed to the GPU
 * backward each step; 500 steps. Gate: final L2 < 10% of initial L2. Writes
 * before/after P6 PPMs (+ PNGs for the Read tool) and prints the paths.
 *
 *   bun tools/splat/fit_demo.ts
 *
 * Deterministic pcg-seeded init (tools/splat/scene.ts) — reproducible.
 */
import { setupGlobals } from "bun-webgpu";
import { RasterEngine } from "../../src/splat/raster";
import { DEFAULT_HYPER } from "../../src/splat/adam_wgsl";
import { makeRng, packParams, writePPM, writePNG, l2 } from "./scene";

setupGlobals();
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter");
  process.exit(1);
}
const device: any = await adapter.requestDevice();
console.log(`adapter: ${(adapter.info ?? {}).vendor ?? "?"} ${(adapter.info ?? {}).architecture ?? "?"}\n`);

const G = 4096;
const H = 128;
const W = 128;
const HW = H * W;
const STEPS = 500;
const OUTDIR = process.env.OUTDIR ?? "/tmp";

// --------------------------------------------------------------------------
// deterministic synthetic target: smooth 3-colour gradient + a filled circle
// --------------------------------------------------------------------------
const target = new Float32Array(3 * HW);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const i = y * W + x;
    let r = x / (W - 1);
    let g = y / (H - 1);
    let b = 0.35 + 0.3 * Math.sin((x + y) * 0.05);
    const dx = x - 78;
    const dy = y - 50;
    if (dx * dx + dy * dy < 26 * 26) {
      r = 0.98;
      g = 0.85;
      b = 0.12;
    } // yellow disc
    target[0 * HW + i] = r;
    target[1 * HW + i] = g;
    target[2 * HW + i] = b;
  }
}

// --------------------------------------------------------------------------
// init 4096 splats: spread means, ~4px, mid opacity, gray-ish colours
// --------------------------------------------------------------------------
const rng = makeRng(424242);
const mean = new Float32Array(2 * G);
const logScale = new Float32Array(2 * G);
const theta = new Float32Array(G);
const colorRaw = new Float32Array(3 * G);
const opacityRaw = new Float32Array(G);
for (let i = 0; i < G; i++) {
  mean[2 * i] = rng.next() * W;
  mean[2 * i + 1] = rng.next() * H;
  logScale[2 * i] = Math.log(4) + 0.2 * rng.normal();
  logScale[2 * i + 1] = Math.log(4) + 0.2 * rng.normal();
  theta[i] = rng.range(0, Math.PI);
  colorRaw[3 * i] = 0.3 * rng.normal();
  colorRaw[3 * i + 1] = 0.3 * rng.normal();
  colorRaw[3 * i + 2] = 0.3 * rng.normal();
  opacityRaw[i] = -0.5 + 0.3 * rng.normal(); // sigmoid ~0.38: semi-transparent
}

// gradScale: dL/dpixels is tiny (~2/N ~ 4e-5), so use a large fixed-point scale
// to keep resolution (no overflow — grads stay small through the whole fit).
const eng = await RasterEngine.create(device, { G, H, W, cap: 1024, bg: [0.5, 0.5, 0.5], gradScale: 1 << 20 });
eng.setParams(packParams(G, { mean, logScale, theta, colorRaw, opacityRaw }));
eng.zeroAdamState();

const N = 3 * HW;
const grad = new Float32Array(N);
const lrs = { mean: 8e-3, logScale: 6e-3, theta: 6e-3, color: 2e-2, opacity: 2e-2 };

let initialL2 = 0;
let initialRender: Float32Array | null = null;
let render = new Float32Array(N);

for (let step = 1; step <= STEPS; step++) {
  eng.runForward();
  render = await eng.readImage();
  if (step === 1) {
    initialRender = render.slice(0);
    initialL2 = l2(render, target);
  }
  // dL/dpixels = 2 (render - target) / N
  for (let i = 0; i < N; i++) grad[i] = (2 * (render[i] - target[i])) / N;
  eng.setGradImage(grad);
  eng.runBackward();
  eng.runAdam(step, lrs, DEFAULT_HYPER);

  if (step % 50 === 0 || step === 1) {
    const cur = l2(render, target);
    console.log(`  step ${String(step).padStart(3)}  L2 = ${cur.toExponential(4)}  (${((cur / initialL2) * 100).toFixed(1)}% of initial)`);
  }
}

// final render
eng.runForward();
const finalRender = await eng.readImage();
const finalL2 = l2(finalRender, target);

const beforePPM = `${OUTDIR}/splat_fit_before.ppm`;
const afterPPM = `${OUTDIR}/splat_fit_after.ppm`;
const targetPPM = `${OUTDIR}/splat_fit_target.ppm`;
writePPM(beforePPM, initialRender!, W, H);
writePPM(afterPPM, finalRender, W, H);
writePPM(targetPPM, target, W, H);
writePNG(`${OUTDIR}/splat_fit_before.png`, initialRender!, W, H);
writePNG(`${OUTDIR}/splat_fit_after.png`, finalRender, W, H);
writePNG(`${OUTDIR}/splat_fit_target.png`, target, W, H);

const ratio = finalL2 / initialL2;
const ok = ratio < 0.1;
console.log("");
console.log(`initial L2 = ${initialL2.toExponential(4)}`);
console.log(`final   L2 = ${finalL2.toExponential(4)}  (${(ratio * 100).toFixed(2)}% of initial)`);
console.log(`PPM: ${beforePPM}\n     ${afterPPM}\n     ${targetPPM}`);
console.log(`${ok ? "PASS" : "FAIL"}  fit demo: final L2 < 10% of initial (${(ratio * 100).toFixed(2)}%)`);
process.exit(ok ? 0 : 1);
