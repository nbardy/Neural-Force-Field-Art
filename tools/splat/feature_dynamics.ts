/**
 * Compare Feature8 optimizer dynamics, not only its final cosine.
 *
 * MODE=default STEPS=1000 OUT_DIR=/tmp/feature-default bun tools/splat/feature_dynamics.ts
 * MODE=default EXPLORE=1 STEPS=1000 OUT_DIR=/tmp/feature-explore bun tools/splat/feature_dynamics.ts
 *
 * `mean-rms` reports physical center migration in raster pixels. A run whose
 * cosine rises while `mean-rms` stays tiny is using an appearance shortcut.
 */
import { mkdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { setupGlobals } from "bun-webgpu";
import { DEFAULT_HYPER } from "../../src/splat/adam_wgsl";
import {
  FEATURE_PAINTER_INIT,
  FEATURE_PAINTER_LRS,
  FeaturePainterOptimizer,
  cosine,
} from "../../src/splat/feature_optimize";
import { LEGIBLE_INIT, LEGIBLE_LRS } from "../../src/splat/optimize";
import type { TrainPlan } from "../../src/clip/vision";
import { writePNG } from "./scene";

setupGlobals();

const G = Number(process.env.G ?? 2048);
const STEPS = Number(process.env.STEPS ?? 1000);
const SEED = Number(process.env.SEED ?? 1);
const MODE = process.env.MODE ?? "default";
const EXPLORE = process.env.EXPLORE === "1";
const OUT_DIR = process.env.OUT_DIR;
const OUT = process.env.OUT;
const CHECKPOINTS = new Set(
  (process.env.CHECKPOINTS ?? "20,120,300,1000")
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((step) => Number.isInteger(step) && step > 0 && step <= STEPS),
);
CHECKPOINTS.add(STEPS);
const MODEL_DIR = fileURLToPath(new URL("../../models/mobileclip_s0", import.meta.url));
const plan = JSON.parse(readFileSync(join(MODEL_DIR, "plan_train.json"), "utf8")) as TrainPlan;
const weightBytes = readFileSync(join(MODEL_DIR, "weights_train.bin"));
const weights = new Float32Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength / 4).slice();
const text = new Float32Array((JSON.parse(readFileSync(join(MODEL_DIR, "fixtures", "text_embeds_test.json"), "utf8")) as Record<string, number[]>)["a photo of a cat"]);
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("feature dynamics: no WebGPU adapter");
const device = await adapter.requestDevice({
  requiredFeatures: adapter.features.has("subgroups" as GPUFeatureName) ? ["subgroups" as GPUFeatureName] : [],
});

const config = MODE === "legacy"
  ? { G, seed: SEED, init: LEGIBLE_INIT, lrs: LEGIBLE_LRS, hyper: DEFAULT_HYPER, featureLR: 0.025, decoderLR: 0.03 }
  : MODE === "airy"
    ? { G, seed: SEED, init: FEATURE_PAINTER_INIT, lrs: LEGIBLE_LRS, hyper: DEFAULT_HYPER, featureLR: 0.025, decoderLR: 0.03 }
  : MODE === "default"
    ? { G, seed: SEED, init: FEATURE_PAINTER_INIT, lrs: FEATURE_PAINTER_LRS, hyper: DEFAULT_HYPER }
    : (() => { throw new Error(`unknown MODE=${MODE}`); })();

const opt = await FeaturePainterOptimizer.create(device, plan, weights, config);
opt.setPrompt(text);
const initial = await opt.raster.readParams();
const before = cosine(await opt.currentEmbedding(), text);
if (OUT_DIR) mkdirSync(OUT_DIR, { recursive: true });

async function checkpoint(step: number): Promise<void> {
  await device.queue.onSubmittedWorkDone();
  const embedding = await opt.currentEmbedding();
  const current = await opt.raster.readParams();
  let meanDistance2 = 0;
  let movedOver8 = 0;
  for (let g = 0; g < G; g++) {
    const dx = current[g * 2] - initial[g * 2];
    const dy = current[g * 2 + 1] - initial[g * 2 + 1];
    const distance2 = dx * dx + dy * dy;
    meanDistance2 += distance2;
    if (distance2 > 64) movedOver8 += 1;
  }
  const score = cosine(embedding, text);
  const label = `${MODE}${EXPLORE ? "+explore" : ""}`;
  console.log(
    `feature dynamics (${label}) step ${String(step).padStart(4)}: ` +
    `cos ${score.toFixed(5)} delta ${(score - before).toFixed(5)}; ` +
    `mean-rms ${Math.sqrt(meanDistance2 / G).toFixed(2)} px; ` +
    `>8px ${(100 * movedOver8 / G).toFixed(1)}%`,
  );
  if (OUT_DIR) {
    const image = await opt.renderImage();
    const path = join(OUT_DIR, `step-${String(step).padStart(4, "0")}.png`);
    writePNG(path, image, opt.side, opt.side);
    console.log(`feature dynamics PNG: ${path}`);
  }
}

console.log(`feature dynamics (${MODE}${EXPLORE ? "+explore" : ""}, seed ${SEED}): initial cosine ${before.toFixed(5)}; horizon ${STEPS} steps`);
for (let step = 0; step < STEPS; step++) {
  opt.step();
  if (EXPLORE && step + 1 === 96) await opt.nudge({ seed: SEED + 0x10001, amount: 0.12 });
  if (EXPLORE && step + 1 === 260) await opt.nudge({ seed: SEED + 0x20001, amount: 0.06 });
  if (CHECKPOINTS.has(step + 1)) await checkpoint(step + 1);
}
if (OUT) {
  writePNG(OUT, await opt.renderImage(), opt.side, opt.side);
  console.log(`feature dynamics PNG: ${OUT}`);
}
opt.destroy();
