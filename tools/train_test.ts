/**
 * Fused-trainer verification: WGSL analytic gradients vs tfjs autograd.
 *
 *   bun tools/grad_reference.ts    # (re)generate the fixture (tfjs CPU)
 *   bun tools/train_test.ts        # this file — real Metal via bun-webgpu
 *
 * Checks:
 *   1. loss:   kernel pass-A loss == fixture loss (tfjs autograd value)
 *   2. grads:  per-variable cosine similarity + relative max error vs fixture
 *   3. adam:   one applied step == JS Adam on the fixture grads (formula + ε)
 *   4. curve:  40 self-generated training steps strictly decrease... loosely —
 *              loss must drop vs start (sanity that training actually trains)
 *   5. bench:  ms/step at batch 256 (the tfjs learn line is ~8-20ms)
 */
import { setupGlobals } from "bun-webgpu";
import { readFileSync } from "node:fs";
import { layoutField, type LayerDims } from "../src/render/webgpu/advect_wgsl";
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

// ---- fixture ----
interface FixVar { name: string; shape: number[]; values: number[] }
const fix = JSON.parse(
  readFileSync(new URL("./fixtures/grad_ref.json", import.meta.url), "utf8")
) as {
  meta: any;
  variables: FixVar[];
  batch: number[];
  loss: number;
  grads: FixVar[];
};

// variables are [kernel,bias]×3 layers ×2 heads in trainableWeights order
const dimsOfHead = (vars: FixVar[]): LayerDims[] =>
  [0, 2, 4].map((i, idx) => ({
    inSize: vars[i].shape[0],
    outSize: vars[i].shape[1],
    activation: idx === 2 ? ("tanh" as const) : ("selu" as const),
  }));
const gVars = fix.variables.slice(0, 6);
const rVars = fix.variables.slice(6, 12);
const layout = layoutField("helmholtz", [dimsOfHead(gVars), dimsOfHead(rVars)]);

const packed = new Float32Array(layout.totalFloats);
fix.variables.forEach((v, i) => {
  packed.set(v.values, layout.segments[i].floatOffset);
});

const PHYS = {
  width: fix.meta.W,
  height: fix.meta.H,
  forceMagnitude: fix.meta.forceMagnitude,
  friction: fix.meta.friction,
  maxVelocity: fix.meta.maxVelocity,
};
const N = fix.meta.N;
const ALPHA = fix.meta.alpha;

const trainer = new FusedTrainer(device, layout, { batchCap: 1024 });
trainer.uploadWeights(packed);
trainer.uploadBatch(Float32Array.from(fix.batch));

// ---- 1+2: gradient parity ----
trainer.step(PHYS, { n: N, alpha: ALPHA, lr: 0, source: "uploaded", apply: false });
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
    console.log(`FAIL  grad[${i}] ${gv.name}: cos=${cos.toFixed(7)} relMax=${rel.toExponential(2)}`);
  }
});
console.log(
  `${worstCos > 0.99999 && worstRel < 2e-3 ? "PASS" : "FAIL"}  grads vs tfjs autograd (12 vars): worst cos=${worstCos.toFixed(7)}, worst relMax=${worstRel.toExponential(2)}`
);

// ---- 3: Adam formula parity (one applied step vs JS Adam on fixture grads) ----
{
  trainer.resetAdam();
  trainer.uploadWeights(packed);
  const lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-7;
  trainer.step(PHYS, { n: N, alpha: ALPHA, lr, source: "uploaded", apply: true });
  const after = await trainer.readWeights();

  // expected: Adam(t=1) applied to the KERNEL's own grads (isolates the
  // update formula from fp32 gradient noise)
  let maxDiff = 0;
  for (let t = 0; t < layout.totalFloats; t++) {
    const g = grads[t];
    const m = (1 - b1) * g;
    const v = (1 - b2) * g * g;
    const mhat = m / (1 - b1);
    const vhat = v / (1 - b2);
    const exp = packed[t] - (lr * mhat) / (Math.sqrt(vhat) + eps);
    maxDiff = Math.max(maxDiff, Math.abs(after[t] - exp));
  }
  const ok = maxDiff < 1e-5;
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  adam step parity (maxΔw=${maxDiff.toExponential(2)})`);
}

// ---- 3.5: K-step rollout (BPTT) gradients vs tfjs autograd, K=4 ----
{
  let fix4: typeof fix | null = null;
  try {
    fix4 = JSON.parse(
      readFileSync(new URL("./fixtures/grad_ref_k4.json", import.meta.url), "utf8")
    );
  } catch (_) {
    failures++;
    console.log("FAIL  K=4 fixture missing — run: K=4 OUT=tools/fixtures/grad_ref_k4.json bun tools/grad_reference.ts");
  }
  if (fix4) {
    const K = fix4.meta.K ?? 4;
    const packed4 = new Float32Array(layout.totalFloats);
    fix4.variables.forEach((v, i) => {
      packed4.set(v.values, layout.segments[i].floatOffset);
    });
    const t4 = new FusedTrainer(device, layout, { batchCap: 1024, kSteps: K });
    t4.uploadWeights(packed4);
    t4.uploadBatch(Float32Array.from(fix4.batch));
    t4.step(PHYS, { n: fix4.meta.N, alpha: fix4.meta.alpha, lr: 0, source: "uploaded", apply: false });
    const l4 = (await t4.readLoss()).loss;
    const g4 = await t4.readGrads();
    const relL = Math.abs(l4 - fix4.loss) / Math.abs(fix4.loss);
    let wCos = 1, wRel = 0;
    fix4.grads.forEach((gv, i) => {
      const seg = layout.segments[i];
      const got = g4.subarray(seg.floatOffset, seg.floatOffset + seg.floatLength);
      let dot = 0, ng = 0, nr = 0, maxAbs = 0, maxRefAbs = 0;
      for (let k = 0; k < gv.values.length; k++) {
        dot += got[k] * gv.values[k];
        ng += got[k] * got[k];
        nr += gv.values[k] * gv.values[k];
        maxAbs = Math.max(maxAbs, Math.abs(got[k] - gv.values[k]));
        maxRefAbs = Math.max(maxRefAbs, Math.abs(gv.values[k]));
      }
      wCos = Math.min(wCos, dot / (Math.sqrt(ng * nr) + 1e-30));
      wRel = Math.max(wRel, maxAbs / (maxRefAbs + 1e-30));
    });
    // K-step BPTT compounds fp32 error through the unrolled chain — tolerance
    // is looser than K=1 but cosine must stay essentially exact.
    const ok = relL < 1e-3 && wCos > 0.9999 && wRel < 5e-3;
    if (!ok) failures++;
    console.log(
      `${ok ? "PASS" : "FAIL"}  K=${K} BPTT: loss rel=${relL.toExponential(2)}, worst cos=${wCos.toFixed(6)}, worst relMax=${wRel.toExponential(2)}`
    );
    t4.destroy();
  }
}

// ---- 3.6: particle-state batch source (source:"particles") ----
// Falsifiable plumbing check: with partCount=1 and vel=0, EVERY sample reads
// the same particle — gradients must equal an uploaded batch of N copies of
// that point. Then a nonzero velocity must change the result (and stay
// finite), proving v0 actually enters the rollout.
{
  const tP = new FusedTrainer(device, layout, { batchCap: 1024 });
  const pt = new Float32Array([123.4, 456.7]);
  const mkPart = (vals: Float32Array) => {
    const GBU = (globalThis as any).GPUBufferUsage;
    const b = device.createBuffer({ size: vals.byteLength, usage: GBU.STORAGE | GBU.COPY_DST });
    device.queue.writeBuffer(b, 0, vals);
    return b;
  };
  const posB = mkPart(pt);
  const velB = mkPart(new Float32Array([0, 0]));
  tP.uploadWeights(packed);
  tP.setParticleBuffers(posB, velB, 1);
  tP.step(PHYS, { n: N, alpha: ALPHA, lr: 0, source: "particles", apply: false, seed: 3 });
  const gPart = await tP.readGrads();
  const lPart = (await tP.readLoss()).loss;

  const rep = new Float32Array(2 * N);
  for (let i = 0; i < N; i++) { rep[2 * i] = pt[0]; rep[2 * i + 1] = pt[1]; }
  tP.uploadBatch(rep);
  tP.step(PHYS, { n: N, alpha: ALPHA, lr: 0, source: "uploaded", apply: false });
  const gUp = await tP.readGrads();
  const lUp = (await tP.readLoss()).loss;
  let maxD = 0;
  for (let i = 0; i < gPart.length; i++) maxD = Math.max(maxD, Math.abs(gPart[i] - gUp[i]));
  const okA = maxD < 1e-6 && Math.abs(lPart - lUp) < 1e-6;

  device.queue.writeBuffer(velB, 0, new Float32Array([30, -20]));
  tP.step(PHYS, { n: N, alpha: ALPHA, lr: 0, source: "particles", apply: false, seed: 3 });
  const gVel = await tP.readGrads();
  const lVel = (await tP.readLoss()).loss;
  let diff = 0, finite = Number.isFinite(lVel);
  for (let i = 0; i < gVel.length; i++) {
    if (!Number.isFinite(gVel[i])) finite = false;
    diff = Math.max(diff, Math.abs(gVel[i] - gPart[i]));
  }
  const okB = finite && diff > 1e-6;
  if (!(okA && okB)) failures++;
  console.log(
    `${okA && okB ? "PASS" : "FAIL"}  particle source: sampling≡uploaded (Δ=${maxD.toExponential(2)}), v0 flows (Δgrad=${diff.toExponential(2)}, loss ${lPart.toFixed(4)}→${lVel.toFixed(4)})`
  );
  tP.destroy(); posB.destroy(); velB.destroy();
}

// ---- 3.7: multi-species classes (C=3) — two exact invariants, no fixture ----
// (a) With the class rows of r's layer-0 kernel ZEROED, the one-hot channels
//     contribute nothing → loss and every shared-parameter gradient must match
//     the CLASSLESS trainer on the same base weights.
// (b) The one-hot partitions the batch: Σ_c dW[2+c][j] must equal r's layer-0
//     BIAS gradient exactly (same summands, regrouped).
{
  const C = 3;
  const layoutC = layoutField(
    "helmholtz",
    [dimsOfHead(gVars), dimsOfHead(rVars).map((d, i) => (i === 0 ? { ...d, inSize: 2 + C } : d))],
    { classes: C }
  );
  // shared base weights: copy the fixture packing; r layer-0 kernel rows 0,1
  // copied row-by-row (row-major [in][out]), class rows 2..4 left at zero
  const packedC = new Float32Array(layoutC.totalFloats);
  layoutC.segments.forEach((segC, i) => {
    const seg0 = layout.segments[i];
    const src = packed.subarray(seg0.floatOffset, seg0.floatOffset + seg0.floatLength);
    if (segC.floatLength === seg0.floatLength) {
      packedC.set(src, segC.floatOffset);
    } else {
      packedC.set(src, segC.floatOffset); // rows 0,1 = first 2·outSize floats
    }
  });

  const tC = new FusedTrainer(device, layoutC, { batchCap: 1024 });
  tC.uploadWeights(packedC);
  tC.uploadBatch(Float32Array.from(fix.batch));
  tC.step(PHYS, { n: N, alpha: ALPHA, lr: 0, source: "uploaded", apply: false });
  const lC = (await tC.readLoss()).loss;
  const gC = await tC.readGrads();

  // (a) loss + shared-grad equivalence vs the classless run (case 1+2's grads)
  let maxShared = 0;
  layoutC.segments.forEach((segC, i) => {
    const seg0 = layout.segments[i];
    const g0 = grads.subarray(seg0.floatOffset, seg0.floatOffset + seg0.floatLength);
    const gc = gC.subarray(segC.floatOffset, segC.floatOffset + segC.floatLength);
    const sharedLen = Math.min(seg0.floatLength, segC.floatLength); // rows 0,1 for r L0
    for (let k = 0; k < sharedLen; k++) {
      maxShared = Math.max(maxShared, Math.abs(gc[k] - g0[k]));
    }
  });
  const okA = Math.abs(lC - loss) < 1e-6 && maxShared < 1e-6;

  // (b) Σ_c classRowGrad[c][j] == biasGrad0[j] for r layer 0
  const rL0 = (layoutC.spec.heads as any)[1].layers[0];
  const out = rL0.outSize;
  const kOff = rL0.weightOffset;
  const bOff = rL0.biasOffset;
  let maxPart = 0;
  for (let j = 0; j < out; j++) {
    let sum = 0;
    for (let c = 0; c < C; c++) sum += gC[kOff + (2 + c) * out + j];
    maxPart = Math.max(maxPart, Math.abs(sum - gC[bOff + j]));
  }
  const okB = maxPart < 1e-4;
  if (!(okA && okB)) failures++;
  console.log(
    `${okA && okB ? "PASS" : "FAIL"}  classes C=3: zero-row ≡ classless (Δloss=${Math.abs(lC - loss).toExponential(2)}, Δshared=${maxShared.toExponential(2)}), Σclass-rows≡bias (Δ=${maxPart.toExponential(2)})`
  );
  tC.destroy();
}

// ---- 3.8: multi-species classes (C=3) vs tfjs autograd — FIXTURE parity ----
// The invariant case above proves the class channels are wired correctly but
// only exercises ZEROED class rows. This case is the full analytic-gradient
// parity: a class-aware tfjs autograd fixture (tools/fixtures/grad_ref_c3.json)
// where r's layer-0 class rows carry REAL per-class gradients. Each sample's
// class comes from its SLOT index via pcg(s ^ SALT) % 3 — the same derivation
// the kernel uses for the uploaded source — so the kernel and fixture must
// agree on both loss and all 12 per-variable gradients (r's [2+C,32] layer-0
// kernel included, class rows and all).
{
  let fix3: typeof fix | null = null;
  try {
    fix3 = JSON.parse(
      readFileSync(new URL("./fixtures/grad_ref_c3.json", import.meta.url), "utf8")
    );
  } catch (_) {
    failures++;
    console.log("FAIL  C=3 fixture missing — run: CLASSES=3 OUT=tools/fixtures/grad_ref_c3.json bun tools/grad_reference.ts");
  }
  if (fix3) {
    const C = fix3.meta.classes ?? 3;
    // Derive dims from the fixture shapes exactly like the top of the file:
    // r's layer-0 inSize (=2+C) comes straight from its kernel shape [2+C,32].
    const gVars3 = fix3.variables.slice(0, 6);
    const rVars3 = fix3.variables.slice(6, 12);
    const layout3 = layoutField(
      "helmholtz",
      [dimsOfHead(gVars3), dimsOfHead(rVars3)],
      { classes: C }
    );
    const packed3 = new Float32Array(layout3.totalFloats);
    fix3.variables.forEach((v, i) => {
      packed3.set(v.values, layout3.segments[i].floatOffset);
    });
    const t3 = new FusedTrainer(device, layout3, { batchCap: 1024 }); // kSteps 1
    t3.uploadWeights(packed3);
    t3.uploadBatch(Float32Array.from(fix3.batch));
    t3.step(PHYS, { n: fix3.meta.N, alpha: fix3.meta.alpha, lr: 0, source: "uploaded", apply: false });
    const l3 = (await t3.readLoss()).loss;
    const g3 = await t3.readGrads();
    const relL = Math.abs(l3 - fix3.loss) / Math.abs(fix3.loss);
    let wCos = 1, wRel = 0;
    fix3.grads.forEach((gv, i) => {
      const seg = layout3.segments[i];
      const got = g3.subarray(seg.floatOffset, seg.floatOffset + seg.floatLength);
      let dot = 0, ng = 0, nr = 0, maxAbs = 0, maxRefAbs = 0;
      for (let k = 0; k < gv.values.length; k++) {
        dot += got[k] * gv.values[k];
        ng += got[k] * got[k];
        nr += gv.values[k] * gv.values[k];
        maxAbs = Math.max(maxAbs, Math.abs(got[k] - gv.values[k]));
        maxRefAbs = Math.max(maxRefAbs, Math.abs(gv.values[k]));
      }
      wCos = Math.min(wCos, dot / (Math.sqrt(ng * nr) + 1e-30));
      wRel = Math.max(wRel, maxAbs / (maxRefAbs + 1e-30));
    });
    const ok = relL < 1e-3 && wCos > 0.9999 && wRel < 5e-3;
    if (!ok) failures++;
    console.log(
      `${ok ? "PASS" : "FAIL"}  C=${C} classes fixture: loss rel=${relL.toExponential(2)}, worst cos=${wCos.toFixed(6)}, worst relMax=${wRel.toExponential(2)}`
    );
    t3.destroy();
  }
}

// ---- 4: training actually trains (self-generated batches, 40 steps) ----
{
  trainer.resetAdam();
  trainer.uploadWeights(packed);
  trainer.step(PHYS, { n: 256, alpha: ALPHA, lr: 0, seed: 1, apply: false });
  const start = (await trainer.readLoss()).loss;
  for (let s = 0; s < 40; s++) {
    trainer.step(PHYS, { n: 256, alpha: ALPHA, lr: 0.01, seed: 100 + s, apply: true });
  }
  trainer.step(PHYS, { n: 256, alpha: ALPHA, lr: 0, seed: 1, apply: false });
  const end = (await trainer.readLoss()).loss;
  const ok = Number.isFinite(end) && end < start;
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  loss decreases over 40 fused steps: ${start.toFixed(4)} → ${end.toFixed(4)}`);
}

// ---- 5: bench ----
{
  const WARM = 20, TIMED = 200;
  for (let s = 0; s < WARM; s++) {
    trainer.step(PHYS, { n: 256, alpha: ALPHA, lr: 0.01, seed: s, apply: true });
  }
  await trainer.readLoss();
  const t0 = performance.now();
  for (let s = 0; s < TIMED; s++) {
    trainer.step(PHYS, { n: 256, alpha: ALPHA, lr: 0.01, seed: WARM + s, apply: true });
  }
  await trainer.readLoss();
  const ms = (performance.now() - t0) / TIMED;
  console.log(`\nBENCH  fused train step @ batch 256: ${ms.toFixed(3)} ms  (tfjs learn line: ~8-20 ms)`);
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
