/**
 * Fused field-loss routing gate.
 *
 * Prevents the regression where every two-head piece silently inherited the
 * historical chaos+isotropy+divergence+spiral shader regardless of its gallery
 * declaration.
 */
import { setupGlobals } from "bun-webgpu";
import { readFileSync } from "node:fs";
import { layoutField, type LayerDims } from "../src/render/webgpu/advect_wgsl";
import { FusedTrainer } from "../src/render/webgpu/train";
import {
  trainPassAShader,
  trainPassBShader,
  type FieldLossSpec,
} from "../src/render/webgpu/train_wgsl";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
  UNIFORM: 64, STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("field_loss_routing_test: no WebGPU adapter");
const device: GPUDevice = await adapter.requestDevice();

const fix = JSON.parse(
  readFileSync(new URL("./fixtures/grad_ref.json", import.meta.url), "utf8")
) as any;
const dims = (vars: any[]): LayerDims[] =>
  [0, 2, 4].map((i, l) => ({
    inSize: vars[i].shape[0],
    outSize: vars[i].shape[1],
    activation: l === 2 ? "tanh" : "selu",
  }));
const layout = layoutField("helmholtz", [
  dims(fix.variables.slice(0, 6)),
  dims(fix.variables.slice(6, 12)),
]);
const packed = new Float32Array(layout.totalFloats);
fix.variables.forEach((v: any, i: number) =>
  packed.set(v.values, layout.segments[i].floatOffset)
);
const phys = {
  width: fix.meta.W,
  height: fix.meta.H,
  forceMagnitude: fix.meta.forceMagnitude,
  friction: fix.meta.friction,
  maxVelocity: fix.meta.maxVelocity,
};
const zero: FieldLossSpec = {
  W_CHAOS: 0, W_ISO: 0, W_DIV: 0, W_SPIRAL: 0, W_COVER: 0, W_CENTER: 0, HH: 1e-2, SPIRAL_TURNS: 3,
};
const maxChaos: FieldLossSpec = {
  W_CHAOS: 1, W_ISO: 1, W_DIV: 0.5, W_SPIRAL: 0, W_COVER: 0, W_CENTER: 0, HH: 1e-2, SPIRAL_TURNS: 3,
};

function only(term: "chaos" | "iso" | "div" | "spiral" | "cover" | "center"): FieldLossSpec {
  return {
    ...zero,
    W_CHAOS: term === "chaos" ? 1 : 0,
    W_ISO: term === "iso" ? 1 : 0,
    W_DIV: term === "div" ? 1 : 0,
    W_SPIRAL: term === "spiral" ? 1 : 0,
    W_COVER: term === "cover" ? 1 : 0,
    W_CENTER: term === "center" ? 0.001 : 0,
    COVER_SAMPLES: 16,
  };
}

async function run(loss: FieldLossSpec) {
  const t = new FusedTrainer(device, layout, { batchCap: 1024, loss });
  t.uploadWeights(packed);
  t.uploadBatch(Float32Array.from(fix.batch));
  t.step(phys, {
    n: fix.meta.N, alpha: fix.meta.alpha, lr: 0, source: "uploaded", apply: false,
  });
  const value = (await t.readLoss()).loss;
  const grad = await t.readGrads();
  t.destroy();
  return { value, grad };
}

/** Regression for the actual live failure: a locally saturated field made an
 * unused chaos intermediate undefined, then `0 * undefined` poisoned the
 * external-gradient-only Adam path. ZERO must be a generated no-op, so even
 * deliberately extreme finite weights remain bit-identical over a long run. */
async function runSaturatedZeroSoak(steps = 2000) {
  const t = new FusedTrainer(device, layout, { batchCap: 1024, loss: zero });
  const saturated = Float32Array.from(packed, (v, i) =>
    layout.segments.some(
      (s) => i >= s.floatOffset && i < s.floatOffset + s.floatLength
    )
      ? (i & 1 ? -24 : 24)
      : v
  );
  t.uploadWeights(saturated);
  t.uploadBatch(Float32Array.from(fix.batch));
  for (let i = 0; i < steps; i++) {
    t.step(phys, {
      n: fix.meta.N,
      alpha: fix.meta.alpha,
      lr: 0.008,
      seed: i,
      source: "uploaded",
      apply: true,
    });
  }
  const after = await t.readWeights();
  const grad = await t.readGrads();
  const loss = (await t.readLoss()).loss;
  let maxWeightDelta = 0;
  for (let i = 0; i < after.length; i++) {
    maxWeightDelta = Math.max(maxWeightDelta, Math.abs(after[i] - saturated[i]));
  }
  t.destroy();
  return {
    loss,
    maxWeightDelta,
    maxGrad: Math.max(...Array.from(grad, Math.abs)),
    finite: after.every(Number.isFinite) && grad.every(Number.isFinite),
  };
}

const z = await run(zero);
const c = await run(maxChaos);
const zs = await runSaturatedZeroSoak();
const zeroA = trainPassAShader(layout, { loss: zero });
const zeroB = trainPassBShader(layout, { loss: zero, extGradCount: 1 });
const termMarkers = {
  chaos: "let chaos_i",
  iso: "let Liso",
  div: "let div_i",
  spiral: "var bestTheta",
  cover: "spiralCoverOff",
  center: "(np.x - cx)",
} as const;
const zMax = Math.max(...Array.from(z.grad, Math.abs));
const cMax = Math.max(...Array.from(c.grad, Math.abs));
let failures = 0;
function ok(cond: boolean, msg: string) {
  if (!cond) failures++;
  console.log(`${cond ? "PASS" : "FAIL"}  ${msg}`);
}
ok(Object.is(z.value, 0) || Math.abs(z.value) < 1e-12,
  `zero spec produces zero loss (${z.value})`);
ok(zMax === 0, `zero spec produces exact zero field gradient (${zMax})`);
ok(Number.isFinite(c.value) && cMax > 1e-8,
  `max-chaos/no-spiral spec is active (loss ${c.value.toFixed(6)}, max|g| ${cMax.toExponential(2)})`);
ok(
  zs.finite && zs.loss === 0 && zs.maxGrad === 0 && zs.maxWeightDelta === 0,
  `ZERO codegen survives 2000 saturated-field steps: finite=${zs.finite}, ` +
    `loss=${zs.loss}, max|g|=${zs.maxGrad}, max|Δw|=${zs.maxWeightDelta}`
);
ok(
  Object.values(termMarkers).every((marker) => !zeroA.includes(marker)),
  "ZERO pass A contains no disabled structural-term arithmetic"
);
ok(
  !zeroB.includes("for (var s = 0u") && zeroB.includes("extGrad0[t]"),
  "external-only pass B skips undefined structural scratch and keeps extGrad"
);
for (const enabled of Object.keys(termMarkers) as (keyof typeof termMarkers)[]) {
  const source = trainPassAShader(layout, { loss: only(enabled) });
  ok(
    Object.entries(termMarkers).every(([term, marker]) =>
      term === enabled ? source.includes(marker) : !source.includes(marker)
    ),
    `${enabled}-only pass A emits that term and elides every disabled term`
  );
}
console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
