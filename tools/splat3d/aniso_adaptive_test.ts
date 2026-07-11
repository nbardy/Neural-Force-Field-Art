/**
 * Deterministic CPU-only gate for fixed-budget anisotropic splat adaptation.
 *
 *   bun tools/splat3d/aniso_adaptive_test.ts
 */
import assert from "node:assert/strict";
import {
  ANISO_PARAM_STRIDE_3D,
  planFixedBudgetAnisotropicSplatAdaptation,
} from "../../src/splat3d_aniso";

const G = 8;
const params = makeParams(G);
const gradients = new Float32Array(params.length);
writeTuple(gradients, 4 * 3, [8, 0, 0]);
writeTuple(gradients, 5 * 3, [0, 12, 0]);
writeTuple(gradients, 6 * 3, [0, 0, 3]);
const originalParams = params.slice();
const originalGradients = gradients.slice();
const options = {
  maxRelocations: 2,
  deadOpacityThreshold: 0.01,
  minParentOpacity: 0.2,
  minRadius: 0.005,
  maxRadius: 0.8,
  minOpacity: 1e-4,
  maxOpacity: 0.99,
  seed: 0x12345678,
};

const first = planFixedBudgetAnisotropicSplatAdaptation(params, gradients, options);
const second = planFixedBudgetAnisotropicSplatAdaptation(params, gradients, options);

assert.equal(first.params.length, ANISO_PARAM_STRIDE_3D * G, "fixed budget must retain 14G parameters");
assert.equal(first.diagnostics.splatCount, G);
assert.equal(first.diagnostics.relocationCount, 2);
assert.deepEqual(Array.from(first.changedIndices), [0, 1, 4, 5]);
assert.deepEqual(params, originalParams, "planner must not mutate parameters");
assert.deepEqual(gradients, originalGradients, "planner must not mutate gradients");
assert.deepEqual(first.params, second.params, "same input and seed must reproduce parameters");
assert.deepEqual(first.relocations, second.relocations, "relocation diagnostics must reproduce");
assert.deepEqual(first.diagnostics, second.diagnostics, "aggregate diagnostics must reproduce");
assert.equal(first.relocations[0].parentIndex, 5, "largest position gradient must rank first");
assert.equal(first.relocations[0].destinationIndex, 0, "lowest-opacity destination must rank first");
assert.ok(first.diagnostics.coverageMassRelativeError < 2e-6, "coverage mass must be conserved");

const absScreenNeed = new Float32Array(G);
absScreenNeed[4] = 40;
absScreenNeed[6] = 120;
const pixelConfidence = Float32Array.from({ length: G }, () => 1);
pixelConfidence[6] = 0.5;
const densitySelected = planFixedBudgetAnisotropicSplatAdaptation(params, gradients, {
  ...options,
  selectionNeed: absScreenNeed,
  coverage: pixelConfidence,
});
assert.equal(
  densitySelected.relocations[0].parentIndex,
  6,
  "AbsGS selection need must override the signed accumulated position gradient"
);

for (const value of first.params) assert.ok(Number.isFinite(value), "all output parameters must be finite");
for (const relocation of first.relocations) {
  const parent = relocation.parentIndex;
  const child = relocation.destinationIndex;
  const parentScaleAfter = readTuple(first.params, 3 * G + parent * 3, 3);
  const childScaleAfter = readTuple(first.params, 3 * G + child * 3, 3);
  const parentQuaternionAfter = readTuple(first.params, 6 * G + parent * 4, 4);
  const childQuaternionAfter = readTuple(first.params, 6 * G + child * 4, 4);
  const parentColorAfter = readTuple(first.params, 10 * G + parent * 3, 3);
  const childColorAfter = readTuple(first.params, 10 * G + child * 3, 3);

  assert.deepEqual(childScaleAfter, parentScaleAfter, "child must clone the shrunken parent axes");
  assert.deepEqual(parentQuaternionAfter, relocation.parentQuaternion, "parent orientation must not change");
  assert.deepEqual(childQuaternionAfter, parentQuaternionAfter, "child must clone orientation");
  assert.deepEqual(childColorAfter, parentColorAfter, "child must clone color");

  const shrinkDeltas = parentScaleAfter.map(
    (value, axis) => value - relocation.parentLogScaleBefore[axis]
  );
  assert.ok(
    Math.max(...shrinkDeltas) - Math.min(...shrinkDeltas) < 2e-7,
    "all axes must shrink by one uniform log-scale delta"
  );
  for (let a = 0; a < 3; a++) {
    for (let b = a + 1; b < 3; b++) {
      const ratioBefore = relocation.parentLogScaleBefore[a] - relocation.parentLogScaleBefore[b];
      const ratioAfter = parentScaleAfter[a] - parentScaleAfter[b];
      assert.ok(Math.abs(ratioAfter - ratioBefore) < 2e-7, "axis ratios must be preserved");
    }
  }
  assert.ok(
    Math.abs(relocation.coverageMassAfter - relocation.coverageMassBefore) /
      relocation.coverageMassBefore <
      2e-6,
    "each split pair must conserve coverage mass"
  );
}

console.log(
  `PASS anisotropic adaptive splats: relocations=${first.diagnostics.relocationCount} ` +
    `changed=[${Array.from(first.changedIndices).join(",")}] ` +
    `coverageError=${first.diagnostics.coverageMassRelativeError.toExponential(3)}`
);

function makeParams(splatCount: number): Float32Array {
  const output = new Float32Array(ANISO_PARAM_STRIDE_3D * splatCount);
  for (let index = 0; index < splatCount; index++) {
    writeTuple(output, index * 3, [index * 0.2 - 0.5, index * -0.1 + 0.25, index * 0.05]);
    writeTuple(output, 3 * splatCount + index * 3, [
      Math.log(0.07 + index * 0.002),
      Math.log(0.11 + index * 0.003),
      Math.log(0.16 + index * 0.004),
    ]);
    const angle = 0.13 * (index + 1);
    writeTuple(output, 6 * splatCount + index * 4, [
      Math.sin(angle) * 0.3,
      Math.sin(angle) * 0.2,
      Math.sin(angle) * 0.1,
      Math.cos(angle),
    ]);
    writeTuple(output, 10 * splatCount + index * 3, [index * 0.1, -index * 0.2, index * 0.3]);
    setOpacity(output, splatCount, index, 0.6);
  }
  setOpacity(output, splatCount, 0, 0.001);
  setOpacity(output, splatCount, 1, 0.002);
  writeTuple(output, 0, [20, 20, 20]);
  writeTuple(output, 3, [-20, -20, -20]);
  return output;
}

function setOpacity(values: Float32Array, splatCount: number, index: number, opacity: number): void {
  values[13 * splatCount + index] = Math.log(opacity / (1 - opacity));
}

function writeTuple(values: Float32Array, offset: number, tuple: readonly number[]): void {
  for (let component = 0; component < tuple.length; component++) {
    values[offset + component] = tuple[component];
  }
}

function readTuple(values: Float32Array, offset: number, length: number): number[] {
  return Array.from({ length }, (_unused, component) => values[offset + component]);
}
