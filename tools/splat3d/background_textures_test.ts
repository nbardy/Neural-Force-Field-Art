/**
 * WebGPU smoke/statistics gate for procedural 3D splat backgrounds.
 *
 *   bun tools/splat3d/background_textures_test.ts
 */
import { setupGlobals } from "bun-webgpu";
import {
  BackgroundTextureGenerator,
  type BackgroundTextureMode,
} from "../../src/splat3d/background_textures";

setupGlobals();

const U = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8 };
const H = 37;
const W = 53;
const HW = H * W;
const FLOATS = 3 * HW;
const MODES: BackgroundTextureMode[] = ["black", "dark_solid", "blurred_noise", "checkerboard", "fourier"];

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
const device: GPUDevice = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}`);

interface Stats {
  min: number;
  max: number;
  mean: [number, number, number];
  variance: number;
  neighborDelta: number;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`GATE FAIL: ${message}`);
}

async function readFloats(buffer: GPUBuffer): Promise<Float32Array> {
  const staging = device.createBuffer({ size: FLOATS * 4, usage: U.MAP_READ | U.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, FLOATS * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(1);
  const output = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return output;
}

async function generate(generator: BackgroundTextureGenerator, step: number, strength: number): Promise<Float32Array> {
  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder();
  generator.recordGenerate(encoder, step, strength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const error = await device.popErrorScope();
  assert(!error, `${generator.mode} dispatch validation: ${error?.message}`);
  return readFloats(generator.buffer);
}

function stats(values: Float32Array): Stats {
  let min = Infinity;
  let max = -Infinity;
  const mean: [number, number, number] = [0, 0, 0];
  for (let channel = 0; channel < 3; channel++) {
    for (let pixel = 0; pixel < HW; pixel++) {
      const value = values[channel * HW + pixel];
      assert(Number.isFinite(value), `non-finite value at channel ${channel}, pixel ${pixel}`);
      min = Math.min(min, value);
      max = Math.max(max, value);
      mean[channel] += value / HW;
    }
  }
  let variance = 0;
  let neighborDelta = 0;
  let neighborCount = 0;
  for (let channel = 0; channel < 3; channel++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const pixel = y * W + x;
        const value = values[channel * HW + pixel];
        const delta = value - mean[channel];
        variance += delta * delta / FLOATS;
        if (x + 1 < W) {
          neighborDelta += Math.abs(value - values[channel * HW + pixel + 1]);
          neighborCount++;
        }
        if (y + 1 < H) {
          neighborDelta += Math.abs(value - values[channel * HW + pixel + W]);
          neighborCount++;
        }
      }
    }
  }
  return { min, max, mean, variance, neighborDelta: neighborDelta / neighborCount };
}

function maxAbsDiff(a: Float32Array, b: Float32Array, scaleB = 1): number {
  assert(a.length === b.length, "diff input lengths differ");
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i] * scaleB));
  return max;
}

function maxPlaneDelta(values: Float32Array): number {
  let max = 0;
  for (let channel = 0; channel < 3; channel++) {
    const first = values[channel * HW];
    for (let pixel = 1; pixel < HW; pixel++) {
      max = Math.max(max, Math.abs(values[channel * HW + pixel] - first));
    }
  }
  return max;
}

function uniquePixels(values: Float32Array): number {
  const unique = new Set<string>();
  for (let pixel = 0; pixel < HW; pixel++) {
    unique.add(
      `${values[pixel].toFixed(6)},${values[HW + pixel].toFixed(6)},${values[2 * HW + pixel].toFixed(6)}`
    );
  }
  return unique.size;
}

const generators: BackgroundTextureGenerator[] = [];
const fullByMode = new Map<BackgroundTextureMode, Float32Array>();

try {
  for (const mode of MODES) {
    const generator = await BackgroundTextureGenerator.create(device, { H, W, mode, seed: 0x1234abcd });
    generators.push(generator);
    const full = await generate(generator, 17, 1);
    const repeat = await generate(generator, 17, 1);
    const nextStep = await generate(generator, 18, 1);
    const s = stats(full);
    fullByMode.set(mode, full);

    assert(maxAbsDiff(full, repeat) === 0, `${mode} is not deterministic for the same step/seed`);
    assert(s.min >= 0 && s.max <= 1, `${mode} escaped RGB [0,1]: [${s.min}, ${s.max}]`);
    if (mode === "black") {
      assert(s.max === 0, "black mode wrote non-zero RGB");
    } else {
      assert(s.max > 0.01, `${mode} is unexpectedly black`);
      assert(maxAbsDiff(full, nextStep) > 1e-5, `${mode} did not change with step`);
    }

    console.log(
      `${mode.padEnd(13)} min=${s.min.toFixed(5)} max=${s.max.toFixed(5)} ` +
        `mean=${s.mean.map((value) => value.toFixed(5)).join("/")} ` +
        `var=${s.variance.toExponential(3)} neighbor=${s.neighborDelta.toExponential(3)}`
    );
  }

  const dark = fullByMode.get("dark_solid")!;
  assert(maxPlaneDelta(dark) === 0, "dark_solid is not spatially constant in planar RGB layout");
  assert(
    dark[0] !== dark[HW] || dark[HW] !== dark[2 * HW],
    "dark_solid did not preserve distinct planar RGB channels"
  );

  const blurred = fullByMode.get("blurred_noise")!;
  assert(stats(blurred).variance > 1e-5, "blurred_noise has no spatial variation");
  assert(uniquePixels(blurred) > 100, "blurred_noise does not produce a continuous texture");

  const checker = fullByMode.get("checkerboard")!;
  assert(uniquePixels(checker) === 2, `checkerboard produced ${uniquePixels(checker)} colors instead of two`);
  assert(stats(checker).variance > 1e-4, "checkerboard contrast is too low");

  const fourierGenerator = generators[MODES.indexOf("fourier")];
  const fourier = fullByMode.get("fourier")!;
  assert(stats(fourier).variance > 1e-4, "fourier texture has no spatial variation");
  const half = await generate(fourierGenerator, 17, 0.5);
  const zero = await generate(fourierGenerator, 17, -1);
  assert(maxAbsDiff(half, fourier, 0.5) < 1e-7, "strength is not a linear black-to-texture fade");
  assert(stats(zero).max === 0, "strength <= 0 did not clamp to black");

  const snapshot = device.createBuffer({ size: FLOATS * 4, usage: U.COPY_SRC | U.COPY_DST });
  const dual = device.createCommandEncoder();
  fourierGenerator.recordGenerate(dual, 17, 1, 0);
  dual.copyBufferToBuffer(fourierGenerator.buffer, 0, snapshot, 0, FLOATS * 4);
  fourierGenerator.recordGenerate(dual, 17, 0, 1);
  device.queue.submit([dual.finish()]);
  await device.queue.onSubmittedWorkDone();
  const trainingSnapshot = await readFloats(snapshot);
  const displayBlack = await readFloats(fourierGenerator.buffer);
  snapshot.destroy();
  assert(maxAbsDiff(trainingSnapshot, fourier) === 0, "training/display uniforms aliased in one encoder");
  assert(stats(displayBlack).max === 0, "display slot did not regenerate black after training texture");

  const otherSeed = await BackgroundTextureGenerator.create(device, {
    H,
    W,
    mode: "blurred_noise",
    seed: 0x1234abce,
  });
  generators.push(otherSeed);
  const otherSeedValues = await generate(otherSeed, 17, 1);
  assert(maxAbsDiff(blurred, otherSeedValues) > 1e-4, "blurred_noise did not change with seed");

  console.log("GATE PASS: procedural backgrounds are valid, planar, deterministic, and mode-distinct.");
} finally {
  for (const generator of generators) generator.destroy();
  device.destroy();
}
