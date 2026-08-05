/**
 * Live Agree+Disagree fused-topology gate.
 *
 * This is deliberately smaller than train_wta_test.ts. It exercises the exact
 * production shape that previously crashed at construction:
 *
 *   agree-disagree two-head Fourier field
 *     ├─ independent predictor A, fieldLane=0, generatorRole=disagree
 *     ├─ independent predictor B, fieldLane=1, generatorRole=agree
 *     └─ one FusedTrainer summing both ext-gradient buffers
 *
 * C is not a third field head: it is a render-time blend of A/B.
 */
import { setupGlobals } from "bun-webgpu";
import {
  layoutField,
  type FieldLayout,
  type LayerDims,
} from "../src/render/webgpu/advect_wgsl";
import { AdversaryTrainer } from "../src/render/webgpu/adversary_train";
import { FusedTrainer } from "../src/render/webgpu/train";
import type { FieldLossSpec } from "../src/render/webgpu/train_wgsl";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1,
  MAP_WRITE: 2,
  COPY_SRC: 4,
  COPY_DST: 8,
  UNIFORM: 64,
  STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("agree_disagree_live_test: no WebGPU adapter");
const device: GPUDevice = await adapter.requestDevice();

let failures = 0;
const ok = (condition: boolean, message: string) => {
  if (!condition) failures++;
  console.log(`${condition ? "PASS" : "FAIL"}  ${message}`);
};

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const mkStorage = (bytes: number) =>
  device.createBuffer({
    size: Math.max(16, bytes),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });

async function readBuffer(buffer: GPUBuffer, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: floats * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, floats * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}

const maxDelta = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result = Math.max(result, Math.abs(a[i] - b[i]));
  }
  return result;
};

const maxAbsInHead = (layout: FieldLayout, values: ArrayLike<number>, head: number) => {
  let result = 0;
  for (const segment of layout.segments) {
    if (segment.head !== head) continue;
    for (let i = 0; i < segment.floatLength; i++) {
      result = Math.max(result, Math.abs(values[segment.floatOffset + i]));
    }
  }
  return result;
};

const encoding = { kind: "fourier", octaves: 3 } as const;
const encDim = 2 + 4 * encoding.octaves;
const dims = (): LayerDims[] => [
  { inSize: encDim, outSize: 8, activation: "selu" },
  { inSize: 8, outSize: 8, activation: "selu" },
  { inSize: 8, outSize: 2, activation: "tanh" },
];
const layout = layoutField("agree-disagree", [dims(), dims()], { encoding });
ok(
  layout.spec.kind === "agree-disagree" &&
    (layout.spec.heads as readonly unknown[]).length === 2,
  "production field has exactly A/B heads; C has no weights or optimizer"
);

const fieldWeights = mkStorage(layout.totalFloats * 4);
const random = mulberry32(2901);
const packed = Float32Array.from(
  { length: layout.totalFloats },
  () => Math.fround((random() * 2 - 1) * 0.35)
);
device.queue.writeBuffer(fieldWeights, 0, packed);

const common = {
  // The largest shipped tuple exercises the A/B lane isolation and summed
  // field-gradient path without relying on the older pair-only kernel.
  tag: "quad-labelled" as const,
  k: 4,
  relaxEps: 0.05,
  hiddenUnits: 8,
  featureDim: 8,
  batchCap: 128,
  fieldWeightsBuffer: fieldWeights,
  particleCount: 384,
};
const branchA = new AdversaryTrainer(device, layout, {
  observerGeometry: "periodic",
  ...common,
  fieldLane: 0,
  generatorRole: "disagree",
  seed: 2902,
});
const branchB = new AdversaryTrainer(device, layout, {
  observerGeometry: "periodic",
  ...common,
  fieldLane: 1,
  generatorRole: "agree",
  seed: 2903,
});

const zeroLoss: FieldLossSpec = {
  W_CHAOS: 0,
  W_ISO: 0,
  W_DIV: 0,
  W_SPIRAL: 0,
  HH: 1e-2,
  SPIRAL_TURNS: 3,
};
const fieldTrainer = new FusedTrainer(device, layout, {
  batchCap: 128,
  weightsBuffer: fieldWeights,
  extGradBuffers: [branchA.extGradsBuf, branchB.extGradsBuf],
  loss: zeroLoss,
});

const physics = {
  width: 800,
  height: 600,
  forceMagnitude: 3.2,
  friction: 0.99,
  maxVelocity: 22,
};
const batch = 96;
const tupleRandom = mulberry32(2904);
const tuples = Float32Array.from({ length: batch * 4 * 4 }, (_, i) => {
  const component = i % 4;
  if (component === 0) return tupleRandom() * physics.width;
  if (component === 1) return tupleRandom() * physics.height;
  return (tupleRandom() * 2 - 1) * 4;
});
branchA.uploadTuples(tuples);
branchB.uploadTuples(tuples);

const trainBatch = Float32Array.from({ length: batch * 2 }, (_, i) =>
  i % 2 === 0 ? tupleRandom() * physics.width : tupleRandom() * physics.height
);
fieldTrainer.uploadBatch(trainBatch);

const aBefore = await branchA.readAdvWeights();
const bBefore = await branchB.readAdvWeights();
ok(maxDelta(aBefore, bBefore) > 0, "A/B predictors start independently");

// Exact production ordering in one encoder: A(D→G), B(D→G), then one field
// pass. `apply:false` lets us inspect the field gradient before Adam mutates W.
const encoder = device.createCommandEncoder();
for (const branch of [branchA, branchB]) {
  branch.encodeStep(encoder, physics, {
    b: batch,
    alpha: 0.5,
    lr: 3e-3,
    seed: 77,
    source: "uploaded",
    genSeed: branch.genSeed(0.012, 1, batch),
    applyDisc: true,
  });
}
fieldTrainer.encodeStep(encoder, physics, {
  n: batch,
  alpha: 0.5,
  lr: 0.006,
  source: "uploaded",
  apply: false,
});
device.queue.submit([encoder.finish()]);

const [gradientA, gradientB, combined, aAfter, bAfter] = await Promise.all([
  branchA.readExtGrads(),
  branchB.readExtGrads(),
  fieldTrainer.readGrads(),
  branchA.readAdvWeights(),
  branchB.readAdvWeights(),
]);
ok(
  maxAbsInHead(layout, gradientA, 0) > 0 &&
    maxAbsInHead(layout, gradientA, 1) === 0,
  "A/disagree writes exactly field lane 0"
);
ok(
  maxAbsInHead(layout, gradientB, 1) > 0 &&
    maxAbsInHead(layout, gradientB, 0) === 0,
  "B/agree writes exactly field lane 1"
);
let sumError = 0;
let sumScale = 0;
for (let i = 0; i < combined.length; i++) {
  const expected = gradientA[i] + gradientB[i];
  sumError = Math.max(sumError, Math.abs(combined[i] - expected));
  sumScale = Math.max(sumScale, Math.abs(expected));
}
ok(
  sumError / (sumScale + 1e-30) < 2e-6,
  `one field pass sums A+B ext gradients (rel ${(sumError / (sumScale + 1e-30)).toExponential(2)})`
);
ok(
  maxDelta(aBefore, aAfter) > 0 && maxDelta(bBefore, bAfter) > 0,
  "both independent predictors receive their discriminator update"
);

const fieldBefore = await readBuffer(fieldWeights, layout.totalFloats);
fieldTrainer.step(physics, {
  n: batch,
  alpha: 0.5,
  lr: 0.006,
  source: "uploaded",
  apply: true,
});
const fieldAfter = await readBuffer(fieldWeights, layout.totalFloats);
ok(
  maxDelta(fieldBefore, fieldAfter) > 0 &&
    Array.from(fieldAfter).every(Number.isFinite),
  "one summed field Adam update moves both A/B lanes and remains finite"
);

branchA.destroy();
branchB.destroy();
fieldTrainer.destroy();
fieldWeights.destroy();

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
