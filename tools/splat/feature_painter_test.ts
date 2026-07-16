/** Small real-Metal smoke for the tiled Feature Painter. */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { FeaturePainterOptimizer, cosine } from "../../src/splat/feature_optimize";
import type { TrainPlan } from "../../src/clip/vision";

setupGlobals();
const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const plan: TrainPlan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8"));
const weights = new Float32Array(readFileSync(join(MODEL_DIR, "weights_train.bin")).buffer);
const text = new Float32Array((JSON.parse(readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")) as Record<string, number[]>)["a photo of a cat"]);
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();
const opt = await FeaturePainterOptimizer.create(device, plan, weights, { G: Number(process.argv[2] ?? 256) });
opt.setPrompt(text);
const before = cosine(await opt.currentEmbedding(), text);
for (let step = 0; step < Number(process.argv[3] ?? 5); step++) opt.step();
await device.queue.onSubmittedWorkDone();
const after = cosine(await opt.currentEmbedding(), text);
console.log(`feature painter smoke: ${before.toFixed(5)} -> ${after.toFixed(5)}`);
opt.destroy();
