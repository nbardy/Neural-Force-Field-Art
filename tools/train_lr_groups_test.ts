/** Pure codegen checks for fused generator learning-rate group routing. */
import { layoutField, type LayerDims } from "../src/render/webgpu/advect_wgsl";
import {
  resolveGeneratorLearningRates,
  type GeneratorLearningRates,
} from "../src/render/webgpu/train";
import { trainPassBShader } from "../src/render/webgpu/train_wgsl";

let failures = 0;
function ok(condition: boolean, label: string): void {
  if (!condition) failures++;
  console.log(`${condition ? "PASS" : "FAIL"}  ${label}`);
}

const uniform = resolveGeneratorLearningRates(0.01, { tag: "uniform" });
ok(
  uniform.shared === 0.01 && uniform.head0 === 0.01 && uniform.head1 === 0.01,
  "uniform mode expands the legacy lr to every group"
);
const grouped: GeneratorLearningRates = {
  tag: "shared-heads",
  shared: 0.001,
  head0: 0.002,
  head1: 0.003,
};
const resolved = resolveGeneratorLearningRates(0.01, grouped);
ok(
  resolved.uniform === 0.01 &&
    resolved.shared === 0.001 &&
    resolved.head0 === 0.002 &&
    resolved.head1 === 0.003,
  "shared-heads mode preserves all four rates"
);
let rejected = false;
try {
  resolveGeneratorLearningRates(0.01, {
    tag: "shared-heads",
    shared: -1,
    head0: 0.002,
    head1: 0.003,
  });
} catch (_) {
  rejected = true;
}
ok(rejected, "invalid grouped rates are rejected before upload");

const dims: LayerDims[] = [
  { inSize: 4, outSize: 4, activation: "selu" },
  { inSize: 4, outSize: 2, activation: "tanh" },
];
const layout = layoutField("helmholtz", [dims, dims], {
  encoding: { kind: "hashgrid", gridSize: 8, features: 4 },
});
const shader = trainPassBShader(layout);

ok(shader.includes("sharedLR : f32"), "WGSL exposes shared hashgrid rate");
ok(shader.includes("head0LR : f32"), "WGSL exposes head 0 rate");
ok(shader.includes("head1LR : f32"), "WGSL exposes head 1 rate");
ok(shader.includes("fn learningRateForWeight(t : u32) -> f32"), "WGSL emits packed-rate selector");
ok(shader.includes("return ub.sharedLR"), "grid segments select shared rate");
ok(shader.includes("return ub.head0LR"), "head 0 segments select head 0 rate");
ok(shader.includes("return ub.head1LR"), "head 1 segments select head 1 rate");
ok(shader.includes("learningRateForWeight(t) * mhat"), "Adam applies the selected rate");

for (const seg of layout.segments) {
  const rate = seg.role === "grid"
    ? "sharedLR"
    : seg.head === 0
      ? "head0LR"
      : "head1LR";
  ok(
    shader.includes(
      `t >= ${seg.floatOffset}u && t < ${seg.floatOffset + seg.floatLength}u) { return ub.${rate}`
    ),
    `${rate} routes segment ${seg.floatOffset}:${seg.floatLength}`
  );
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
