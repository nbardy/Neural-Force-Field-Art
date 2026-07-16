/**
 * Real-Metal end-to-end gate for the fused compact Feature8 painter.
 *
 * bun tools/splat/feature_painter_test.ts [G=2048] [steps=20]
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { FeaturePainterOptimizer, cosine } from "../../src/splat/feature_optimize";
import type { TrainPlan } from "../../src/clip/vision";
import { writePNG } from "./scene";

setupGlobals();
const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = new Float32Array(readFileSync(join(MODEL_DIR, "weights_train.bin")).buffer);
const text = new Float32Array((JSON.parse(readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")) as Record<string, number[]>)["a photo of a cat"]);
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();
const G = Number(process.argv[2] ?? 2048);
const STEPS = Number(process.argv[3] ?? 20);
const opt = await FeaturePainterOptimizer.create(device, plan, weights, { G });
opt.setPrompt(text);
const before = cosine(await opt.currentEmbedding(), text);
writePNG("/tmp/feature8_before.png", await opt.renderImage(), opt.side, opt.side);
const featureBefore = await opt.raster.readFeatureParams();
for (let step = 0; step < 3; step++) opt.step(); // warm Metal pipeline compilation
await device.queue.onSubmittedWorkDone();
const started = performance.now();
for (let step = 0; step < STEPS; step++) opt.step();
await device.queue.onSubmittedWorkDone();
const msPerStep = (performance.now() - started) / STEPS;
const after = cosine(await opt.currentEmbedding(), text);
writePNG("/tmp/feature8_after.png", await opt.renderImage(), opt.side, opt.side);
const featureAfter = await opt.raster.readFeatureParams();
const decoder = await opt.raster.readDecoderParams();
let featureDelta2 = 0;
for (let i = 0; i < featureBefore.length; i++) featureDelta2 += (featureAfter[i] - featureBefore[i]) ** 2;
let decoderNorm2 = 0;
for (const value of decoder) decoderNorm2 += value * value;
console.log(`feature8: cos ${before.toFixed(5)} -> ${after.toFixed(5)} (delta ${(after - before).toFixed(5)})`);
console.log(`feature8: ${msPerStep.toFixed(1)} ms/step at G=${G}; feature delta L2=${Math.sqrt(featureDelta2).toFixed(4)} decoder L2=${Math.sqrt(decoderNorm2).toFixed(4)}`);
console.log("feature8 PNGs: /tmp/feature8_before.png -> /tmp/feature8_after.png");
if (after - before <= 0.01) throw new Error(`feature8 convergence gate failed: delta ${(after - before).toFixed(5)}`);
opt.destroy();
