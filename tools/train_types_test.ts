/**
 * Fused-trainer verification for ALL FIELD TYPES vs tfjs autograd — the M4
 * acceptance gate (docs/PLAN_AD_IR_BACKWARD_CODEGEN.md §6): standard, SIREN,
 * Fourier and HashGrid must all reproduce the tfjs training gradient on REAL
 * METAL at cos ≈ 1.0 before their fused path ships.
 *
 *   bun tools/grad_reference.ts                                  # standard
 *   MODEL=siren    OUT=tools/fixtures/grad_ref_siren.json    bun tools/grad_reference.ts
 *   MODEL=fourier  OUT=tools/fixtures/grad_ref_fourier.json  bun tools/grad_reference.ts
 *   MODEL=hashgrid OUT=tools/fixtures/grad_ref_hashgrid.json bun tools/grad_reference.ts
 *   bun tools/train_types_test.ts          # every fixture present
 *   FIX=fixtures/grad_ref_siren.json bun tools/train_types_test.ts   # one
 *
 * Checks per fixture: (1) pass-A loss parity, (2) per-variable gradient cosine
 * + relative max error, (3) 30 self-generated training steps decrease the loss
 * (the field actually TRAINS fused, not just matches one gradient).
 *
 * tools/train_test.ts remains the deeper standard-arch regression (Adam
 * formula parity, bench); this file is the per-type parity sweep.
 */
import { setupGlobals } from "bun-webgpu";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  layoutField,
  type Encoding,
  type LayerDims,
} from "../src/render/webgpu/advect_wgsl";
import { FusedTrainer } from "../src/render/webgpu/train";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
  UNIFORM: 64, STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter");
  process.exit(1);
}
const device: any = await adapter.requestDevice();
let failures = 0;

interface FixVar { name: string; shape: number[]; values: number[] }
interface Fixture {
  meta: any;
  variables: FixVar[];
  batch: number[];
  loss: number;
  grads: FixVar[];
}

const DEFAULT_FIXTURES = [
  "fixtures/grad_ref.json",
  "fixtures/grad_ref_siren.json",
  "fixtures/grad_ref_fourier.json",
  "fixtures/grad_ref_hashgrid.json",
  "fixtures/grad_ref_siren_k4.json",
];
const wanted = process.env.FIX ? [process.env.FIX] : DEFAULT_FIXTURES;
const fixtures = wanted.filter((f) =>
  existsSync(fileURLToPath(new URL(f, import.meta.url)))
);
if (fixtures.length === 0) {
  console.error("no fixtures found — run tools/grad_reference.ts first");
  process.exit(1);
}

async function verify(fixPath: string): Promise<void> {
  const fix = JSON.parse(
    readFileSync(new URL(fixPath, import.meta.url), "utf8")
  ) as Fixture;
  const m = fix.meta;
  const model: string = m.model ?? "standard";
  console.log(`\n=== ${fixPath} — model=${model} K=${m.K} N=${m.N} ===`);

  const encoding: Encoding =
    model === "fourier"
      ? { kind: "fourier", octaves: m.fourierOctaves ?? 4 }
      : model === "hashgrid"
      ? { kind: "hashgrid", gridSize: m.gridSize ?? 32, features: m.gridFeatures ?? 4 }
      : { kind: "raw" };

  // hashgrid lists the grid variable FIRST (matches the layout's grid segment)
  const gridVars = model === "hashgrid" ? 1 : 0;
  const headVars = fix.variables.slice(gridVars);
  const hidden = model === "siren" ? ("sin" as const) : ("selu" as const);
  const dimsOfHead = (vars: FixVar[]): LayerDims[] =>
    [0, 2, 4].map((i, idx) => ({
      inSize: vars[i].shape[0],
      outSize: vars[i].shape[1],
      activation: idx === 2 ? ("tanh" as const) : hidden,
    }));
  const layout = layoutField(
    "helmholtz",
    [dimsOfHead(headVars.slice(0, 6)), dimsOfHead(headVars.slice(6, 12))],
    { classes: m.classes ?? 0, encoding }
  );

  const packed = new Float32Array(layout.totalFloats);
  fix.variables.forEach((v, i) => {
    packed.set(v.values, layout.segments[i].floatOffset);
  });

  const PHYS = {
    width: m.W,
    height: m.H,
    forceMagnitude: m.forceMagnitude,
    friction: m.friction,
    maxVelocity: m.maxVelocity,
  };

  const trainer = new FusedTrainer(device, layout, {
    batchCap: 1024,
    kSteps: m.K,
  });
  trainer.uploadWeights(packed);
  trainer.uploadBatch(Float32Array.from(fix.batch));

  // --- 1+2: loss + gradient parity -----------------------------------------
  trainer.step(PHYS, { n: m.N, alpha: m.alpha, lr: 0, source: "uploaded", apply: false });
  const { loss } = await trainer.readLoss();
  const grads = await trainer.readGrads();

  {
    const rel = Math.abs(loss - fix.loss) / Math.abs(fix.loss);
    const ok = rel < 1e-3;
    if (!ok) failures++;
    console.log(
      `${ok ? "PASS" : "FAIL"}  loss parity: kernel=${loss.toFixed(6)} tfjs=${fix.loss.toFixed(6)} (rel=${rel.toExponential(2)})`
    );
  }

  let worstCos = 1;
  let worstRel = 0;
  fix.grads.forEach((gv, i) => {
    const seg = layout.segments[i];
    const got = grads.subarray(seg.floatOffset, seg.floatOffset + seg.floatLength);
    const ref = gv.values;
    let dot = 0, ng = 0, nr = 0, maxAbs = 0, maxRefAbs = 0;
    for (let k = 0; k < ref.length; k++) {
      dot += got[k] * ref[k];
      ng += got[k] * got[k];
      nr += ref[k] * ref[k];
      maxAbs = Math.max(maxAbs, Math.abs(got[k] - ref[k]));
      maxRefAbs = Math.max(maxRefAbs, Math.abs(ref[k]));
    }
    const cos = dot / (Math.sqrt(ng * nr) + 1e-30);
    const rel = maxAbs / (maxRefAbs + 1e-30);
    worstCos = Math.min(worstCos, cos);
    worstRel = Math.max(worstRel, rel);
    const ok = cos > 0.99999 && rel < 2e-3;
    if (!ok) {
      failures++;
      console.log(
        `FAIL  grad[${i}] ${gv.name}: cos=${cos.toFixed(7)} relMax=${rel.toExponential(2)}`
      );
    }
  });
  console.log(
    `${worstCos > 0.99999 && worstRel < 2e-3 ? "PASS" : "FAIL"}  grads vs tfjs autograd (${fix.grads.length} vars): worst cos=${worstCos.toFixed(7)}, worst relMax=${worstRel.toExponential(2)}`
  );

  // --- 3: it trains — 30 fused steps on self-generated batches drop the loss
  {
    trainer.uploadWeights(packed);
    trainer.resetAdam();
    trainer.step(PHYS, { n: 256, alpha: m.alpha, lr: 0, seed: 1, apply: false });
    const start = (await trainer.readLoss()).loss;
    for (let s = 0; s < 30; s++) {
      trainer.step(PHYS, { n: 256, alpha: m.alpha, lr: 0.01, seed: s + 2, apply: true });
    }
    trainer.step(PHYS, { n: 256, alpha: m.alpha, lr: 0, seed: 1, apply: false });
    const end = (await trainer.readLoss()).loss;
    const ok = Number.isFinite(end) && end < start;
    if (!ok) failures++;
    console.log(
      `${ok ? "PASS" : "FAIL"}  training curve: loss ${start.toFixed(4)} → ${end.toFixed(4)} over 30 fused steps`
    );
  }

  trainer.destroy();
}

for (const f of fixtures) {
  await verify(f);
}

console.log(
  failures === 0
    ? `\nALL TRAIN-TYPE CHECKS PASS (${fixtures.length} fixtures)`
    : `\n${failures} CHECK(S) FAILED`
);
process.exit(failures === 0 ? 0 : 1);
