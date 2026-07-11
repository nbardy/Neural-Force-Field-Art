/**
 * Deterministic CPU-only gate for the fixed-budget splat adaptation planner.
 *
 *   bun tools/splat3d/adaptive_test.ts
 */
import assert from "node:assert/strict";
import {
  planFixedBudgetSplatAdaptation,
  type SplatAdaptationPlan,
} from "../../src/splat3d/adaptive";

const G = 8;
const MIN_RADIUS = 0.01;
const MAX_RADIUS = 0.45;
const MIN_OPACITY = 1e-4;
const MAX_OPACITY = 0.99;

const params = makeParams(G);
const gradients = new Float32Array(params.length);
setPosition(gradients, 4, [8, 0, 0]);
setPosition(gradients, 5, [0, 12, 0]);
setPosition(gradients, 6, [0, 0, 3]);
const originalParams = params.slice();
const originalGradients = gradients.slice();
const options = {
  maxRelocations: 2,
  deadOpacityThreshold: 0.01,
  minParentOpacity: 0.2,
  minRadius: MIN_RADIUS,
  maxRadius: MAX_RADIUS,
  minOpacity: MIN_OPACITY,
  maxOpacity: MAX_OPACITY,
  seed: 0x12345678,
};

const first = planFixedBudgetSplatAdaptation(params, gradients, options);
const second = planFixedBudgetSplatAdaptation(params, gradients, options);

assert.equal(first.params.length, params.length, "fixed budget must retain exactly 8G parameters");
assert.equal(first.diagnostics.splatCount, G);
assert.equal(first.diagnostics.relocationCount, 2);
assert.deepEqual(Array.from(first.changedIndices), [0, 1, 4, 5]);
assert.deepEqual(params, originalParams, "planner must not mutate params");
assert.deepEqual(gradients, originalGradients, "planner must not mutate gradients");
assert.deepEqual(first.params, second.params, "same input and seed must reproduce parameters");
assert.deepEqual(first.changedIndices, second.changedIndices, "changed indices must reproduce");
assert.deepEqual(first.relocations, second.relocations, "relocation diagnostics must reproduce");
assert.deepEqual(first.diagnostics, second.diagnostics, "aggregate diagnostics must reproduce");
assertFiniteAndBounded(first);
assert.ok(first.diagnostics.coverageMassRelativeError < 2e-6, "split should preserve coverage mass");

for (const relocation of first.relocations) {
  const oldDestination = readPosition(params, relocation.destinationIndex);
  const oldDistance = distance(oldDestination, relocation.parentPositionBefore);
  const newDistance = distance(relocation.childPosition, relocation.parentPositionBefore);
  assert.ok(newDistance < oldDistance, "dead destination must move toward its high-gradient parent");
  assert.ok(relocation.radiusAfter < relocation.parentRadiusBefore, "split radius must shrink");
  assert.ok(relocation.opacityAfter < relocation.parentOpacityBefore, "split opacity must shrink");
}

assert.equal(first.relocations[0].parentIndex, 5, "largest absolute gradient must rank first");
assert.equal(first.relocations[0].destinationIndex, 0, "lowest-opacity destination must rank first");

testCoverageWeighting();
testBoundsOnlyPlan();

console.log(
  `PASS adaptive splats: relocations=${first.diagnostics.relocationCount} ` +
    `changed=[${Array.from(first.changedIndices).join(",")}] ` +
    `coverageError=${first.diagnostics.coverageMassRelativeError.toExponential(3)}`
);

function testCoverageWeighting(): void {
  const localG = 4;
  const localParams = makeParams(localG);
  setOpacity(localParams, localG, 0, 0.001);
  setOpacity(localParams, localG, 1, 0.6);
  setPosition(localParams, 0, [20, 20, 20]);
  const localGradients = new Float32Array(localParams.length);
  setPosition(localGradients, 1, [10, 0, 0]);
  setPosition(localGradients, 2, [6, 0, 0]);

  const unweighted = planFixedBudgetSplatAdaptation(localParams, localGradients, {
    maxRelocations: 1,
    deadOpacityThreshold: 0.01,
    minParentOpacity: 0.2,
  });
  const weighted = planFixedBudgetSplatAdaptation(localParams, localGradients, {
    maxRelocations: 1,
    deadOpacityThreshold: 0.01,
    minParentOpacity: 0.2,
    coverage: Float32Array.of(0, 0.1, 1, 0),
  });

  assert.equal(unweighted.relocations[0].parentIndex, 1, "raw gradient should select parent 1");
  assert.equal(weighted.relocations[0].parentIndex, 2, "coverage must reweight parent need");
}

function testBoundsOnlyPlan(): void {
  const localG = 3;
  const localParams = makeParams(localG);
  localParams[3 * localG + 0] = Math.log(1e-6);
  localParams[3 * localG + 1] = Math.log(4);
  localParams[7 * localG + 0] = -80;
  localParams[7 * localG + 1] = 80;
  const plan = planFixedBudgetSplatAdaptation(localParams, new Float32Array(localParams.length), {
    maxRelocations: 0,
    minRadius: MIN_RADIUS,
    maxRadius: MAX_RADIUS,
    minOpacity: MIN_OPACITY,
    maxOpacity: MAX_OPACITY,
  });

  assert.equal(plan.diagnostics.relocationCount, 0);
  assert.equal(plan.diagnostics.radiusClampCount, 2);
  assert.equal(plan.diagnostics.opacityClampCount, 2);
  assert.deepEqual(Array.from(plan.changedIndices), [0, 1]);
  assertFiniteAndBounded(plan);
}

function makeParams(splatCount: number): Float32Array {
  const out = new Float32Array(8 * splatCount);
  for (let index = 0; index < splatCount; index++) {
    setPosition(out, index, [index * 0.2 - 0.5, index * -0.1 + 0.25, index * 0.05]);
    out[3 * splatCount + index] = Math.log(0.1);
    out[4 * splatCount + index * 3 + 0] = index * 0.1;
    out[4 * splatCount + index * 3 + 1] = -index * 0.2;
    out[4 * splatCount + index * 3 + 2] = index * 0.3;
    setOpacity(out, splatCount, index, 0.6);
  }
  setOpacity(out, splatCount, 0, 0.001);
  setOpacity(out, splatCount, 1, 0.002);
  setPosition(out, 0, [20, 20, 20]);
  setPosition(out, 1, [-20, -20, -20]);
  return out;
}

function assertFiniteAndBounded(plan: SplatAdaptationPlan): void {
  const splatCount = plan.diagnostics.splatCount;
  for (const value of plan.params) assert.ok(Number.isFinite(value), "all output params must be finite");
  for (let index = 0; index < splatCount; index++) {
    const radius = Math.exp(plan.params[3 * splatCount + index]);
    const opacity = sigmoid(plan.params[7 * splatCount + index]);
    assert.ok(radius >= MIN_RADIUS && radius <= MAX_RADIUS, `radius ${index} out of bounds: ${radius}`);
    assert.ok(opacity >= MIN_OPACITY && opacity <= MAX_OPACITY, `opacity ${index} out of bounds: ${opacity}`);
  }
}

function setPosition(values: Float32Array, index: number, position: [number, number, number]): void {
  values[index * 3 + 0] = position[0];
  values[index * 3 + 1] = position[1];
  values[index * 3 + 2] = position[2];
}

function readPosition(values: Float32Array, index: number): [number, number, number] {
  return [values[index * 3 + 0], values[index * 3 + 1], values[index * 3 + 2]];
}

function setOpacity(values: Float32Array, splatCount: number, index: number, opacity: number): void {
  values[7 * splatCount + index] = Math.log(opacity / (1 - opacity));
}

function sigmoid(raw: number): number {
  return 1 / (1 + Math.exp(-raw));
}

function distance(a: [number, number, number], b: [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
