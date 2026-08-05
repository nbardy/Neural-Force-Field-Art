/** End-to-end quality/smoke gate for the direct trainable image baseline. */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { PixelBufferOptimizer, cosine } from "../../src/splat/pixel_optimize";
import type { TrainPlan } from "../../src/clip/vision";
import { writePNG } from "./scene";

setupGlobals();

const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const plan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8")) as TrainPlan;
const weightBytes = readFileSync(join(MODEL_DIR, "weights_train.bin"));
const weights = new Float32Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength / 4).slice();
const text = new Float32Array((JSON.parse(readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")) as Record<string, number[]>)["a photo of a cat"]);
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("pixel optimizer test: no WebGPU adapter");
const device = await adapter.requestDevice();
const STEPS = Number(process.argv[2] ?? 40);

const warm = await PixelBufferOptimizer.create(device, plan, weights, 1);
warm.setPrompt(text);
for (let step = 0; step < 3; step++) warm.step();
await device.queue.onSubmittedWorkDone();
warm.destroy();

const opt = await PixelBufferOptimizer.create(device, plan, weights, 1);
opt.setPrompt(text);
const before = cosine(await opt.currentEmbedding(), text);
for (let step = 0; step < STEPS; step++) opt.step();
await device.queue.onSubmittedWorkDone();
const after = cosine(await opt.currentEmbedding(), text);
const image = await opt.renderImage();
writePNG("/tmp/pixel_buffer_after.png", image, opt.side, opt.side);
console.log(`pixel buffer: cos ${before.toFixed(5)} -> ${after.toFixed(5)} (delta ${(after - before).toFixed(5)})`);
console.log("pixel buffer PNG: /tmp/pixel_buffer_after.png");
if (after - before <= 0.01) throw new Error(`pixel buffer convergence gate failed: delta ${(after - before).toFixed(5)}`);
opt.destroy();
