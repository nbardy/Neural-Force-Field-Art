/** Verify that Feature8 NUDGE replaces sparse candidates and clears only their latent payload. */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { FeaturePainterOptimizer } from "../../src/splat/feature_optimize";
import { FEATURE_STRIDE } from "../../src/splat/feature_painter_wgsl";
import { nudgeSplatMask } from "../../src/splat/optimize";
import type { TrainPlan } from "../../src/clip/vision";

setupGlobals();

const G = Number(process.argv[2] ?? 512);
const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const plan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8")) as TrainPlan;
const weightBytes = readFileSync(join(MODEL_DIR, "weights_train.bin"));
const weights = new Float32Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength / 4).slice();
const text = new Float32Array((JSON.parse(readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")) as Record<string, number[]>)["a photo of a cat"]);
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("feature nudge test: no WebGPU adapter");
const device = await adapter.requestDevice({ requiredFeatures: ["subgroups"] });

const opt = await FeaturePainterOptimizer.create(device, plan, weights, { G, seed: 1 });
opt.setPrompt(text);
for (let step = 0; step < 8; step++) opt.step();
await device.queue.onSubmittedWorkDone();
const before = await opt.raster.readFeatureParams();
const seed = 37;
const amount = 0.1;
const selection = nudgeSplatMask(G, seed, amount);
await opt.nudge({ seed, amount });
const after = await opt.raster.readFeatureParams();

let selected = 0;
for (let g = 0; g < G; g++) {
  const start = g * FEATURE_STRIDE;
  if (selection[g] !== 0) {
    selected += 1;
    for (let i = start; i < start + FEATURE_STRIDE; i++) {
      if (after[i] !== 0) throw new Error(`selected splat ${g} retained latent value`);
    }
  } else {
    for (let i = start; i < start + FEATURE_STRIDE; i++) {
      if (after[i] !== before[i]) throw new Error(`unselected splat ${g} feature payload changed`);
    }
  }
}
console.log(`feature nudge: replaced ${selected}/${G} splats; selected latent payloads reset`);
opt.destroy();
