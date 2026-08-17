/**
 * TRAIN BATCH CAP test — the raised cap actually runs, and over-cap requests
 * clamp instead of killing the frame loop.
 *
 *   bun tools/train_batch_cap_test.ts
 *
 * Regression context (2026-08-17): the "train B" slider went to 4096 while
 * FusedTrainer was constructed with a hard-coded `batchCap: 1024`. Any n above
 * 1024 hit the guard in FusedTrainer.record(), which THROWS — and it throws
 * inside the rAF `tick`, whose `requestAnimationFrame(tick)` re-arm is its last
 * statement. So the top three quarters of the slider permanently froze the
 * animation. The fix: FusedTrainer resolves its own cap from MAX_BATCH ∧ the
 * device's storage-buffer limit for this layout, publishes it as `.batchCap`,
 * and the UI/handle canonicalize against it (main.ts resolveTrainBatchSize).
 *
 * Checks:
 *   1. the resolved cap is exactly MAX_BATCH ∧ device-scratch-limit, and beats
 *      the old hard-coded 1024
 *   2. a REAL training step at that cap runs and moves the weights to finite
 *      values (the cap is usable, not just a bigger number)
 *   3. an over-cap batchCap request clamps in the constructor instead of
 *      throwing an opaque WebGPU OOM
 *   4. maxBatchForScratch's arithmetic under a tight byte budget (deterministic
 *      — does not depend on this box's limits)
 */
import { setupGlobals } from "bun-webgpu";
setupGlobals();

// Adapter shims for tfjs 4.10 under bun-webgpu — copied verbatim from
// tools/window_test.ts (proven).
{
  const gpu = (navigator as any).gpu;
  const origReq = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (o?: any) => {
    const a = await origReq(o);
    if (!a) return a;
    if (!a.requestAdapterInfo) {
      a.requestAdapterInfo = async () => a.info ?? {};
    }
    const origDev = a.requestDevice.bind(a);
    a.requestDevice = async (desc?: any) => {
      const d = desc
        ? { ...desc, requiredFeatures: [...(desc.requiredFeatures ?? [])].flat(Infinity) }
        : desc;
      if (d) delete d.requiredLimits;
      for (;;) {
        try {
          return await origDev(d);
        } catch (e: any) {
          const m = /Invalid feature required: (\S+)/.exec(String(e?.message ?? e));
          if (!m || !d?.requiredFeatures?.includes(m[1])) throw e;
          d.requiredFeatures = d.requiredFeatures.filter((f: string) => f !== m[1]);
        }
      }
    };
    return a;
  };
}

const tf = await import("@tensorflow/tfjs");
(globalThis as any).window ??= globalThis;
(globalThis as any).GPUBuffer ??= class GPUBufferShim {};
(globalThis as any).GPUTexture ??= class GPUTextureShim {};
await import("@tensorflow/tfjs-backend-webgpu");
const { HelmholtzField } = await import("../src/core/field/helmholtz");
const { AdvectKernel } = await import("../src/render/webgpu/advect");
const { FusedTrainer } = await import("../src/render/webgpu/train");
const { MAX_BATCH, COVER_MAX_BATCH, maxBatchForScratch, maxBatchForLoss, scratchBytes } =
  await import("../src/render/webgpu/train_wgsl");
const { resolveTrainBatchSize, COVER_FIELD_LOSS } = await import("../src/main");

await tf.setBackend("webgpu");
await tf.ready();
if (tf.getBackend() !== "webgpu") {
  console.error(`FATAL: backend is ${tf.getBackend()}, not webgpu`);
  process.exit(1);
}

const device: any = (tf.backend() as any).device;
let failures = 0;
function ok(condition: boolean, message: string): void {
  console.log(`${condition ? "  ok  " : " FAIL "} ${message}`);
  if (!condition) failures++;
}

async function readF32(buf: any, floats: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: floats * 4, usage: 1 | 8 });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

const PHYS = {
  width: 800, height: 600,
  forceMagnitude: 3.5, friction: 0.99, maxVelocity: 26,
  resetRate: 0,
};
const ALPHA = 0.6;
const K = 2;
const OLD_HARD_CODED_CAP = 1024;
const storageLimit: number = device.limits.maxStorageBufferBindingSize;

console.log(
  `device maxStorageBufferBindingSize = ${(storageLimit / 1048576).toFixed(0)} MiB\n`
);

const field = new HelmholtzField({ alpha: ALPHA });
const kernel = AdvectKernel.fromField(field as any, PHYS, 4096);
const trainer = new FusedTrainer(device, kernel.layout, {
  weightsBuffer: kernel.weightsBuffer,
  batchCap: MAX_BATCH,
  kSteps: K,
});
trainer.uploadWeights(kernel.packCurrentWeights());
kernel.syncFromTfjs = false;

console.log("§1 RESOLVED CAP");
const expected = maxBatchForScratch(kernel.layout, K, storageLimit);
ok(
  trainer.batchCap === expected,
  `batchCap ${trainer.batchCap} == MAX_BATCH ∧ device scratch limit (${expected})`
);
ok(
  trainer.batchCap > OLD_HARD_CODED_CAP,
  `batchCap ${trainer.batchCap} exceeds the old hard-coded ${OLD_HARD_CODED_CAP}`
);
ok(
  scratchBytes(kernel.layout, trainer.batchCap, K) <= storageLimit,
  `scratch at the cap (${(scratchBytes(kernel.layout, trainer.batchCap, K) / 1048576).toFixed(1)} MiB) fits the device limit`
);

console.log("\n§2 A REAL STEP AT THE CAP");
// The point of the whole change: the largest batch the UI can now request must
// TRAIN, not merely be accepted. Old code threw here for anything > 1024.
const before = await readF32(trainer.weightsBuf, kernel.layout.totalFloats);
let threw: string | null = null;
try {
  trainer.step(
    {
      width: PHYS.width, height: PHYS.height,
      forceMagnitude: PHYS.forceMagnitude,
      friction: PHYS.friction, maxVelocity: PHYS.maxVelocity,
    },
    { n: trainer.batchCap, alpha: ALPHA, lr: 1e-3, apply: true, source: "random", seed: 7 }
  );
} catch (e: any) {
  threw = String(e?.message ?? e);
}
ok(threw === null, `a step at n=${trainer.batchCap} does not throw${threw ? ` (${threw})` : ""}`);
const after = await readF32(trainer.weightsBuf, kernel.layout.totalFloats);
ok(after.every(Number.isFinite), "all weights finite after the max-batch step");
ok(
  after.some((w, i) => w !== before[i]),
  "the max-batch step actually moved the weights (Adam applied)"
);

console.log("\n§3 OVER-CAP IS CLAMPED, NOT FATAL");
// The UI-side κ: over-cap resolves to a runnable n carrying its provenance.
const over = resolveTrainBatchSize(trainer.batchCap + 5000, trainer.batchCap);
ok(over.tag === "clamped" && over.n === trainer.batchCap, "over-cap request clamps to the cap");
let overThrew: string | null = null;
try {
  trainer.step(
    {
      width: PHYS.width, height: PHYS.height,
      forceMagnitude: PHYS.forceMagnitude,
      friction: PHYS.friction, maxVelocity: PHYS.maxVelocity,
    },
    { n: over.n, alpha: ALPHA, lr: 1e-3, apply: true, source: "random", seed: 8 }
  );
} catch (e: any) {
  overThrew = String(e?.message ?? e);
}
ok(overThrew === null, "the clamped n runs — the frame loop survives an over-cap slider");

// A constructor asking for more than the device can hold must clamp, not throw
// an opaque createBuffer OOM. Force it with the fattest legal rollout.
const fat = new FusedTrainer(device, kernel.layout, {
  batchCap: MAX_BATCH,
  kSteps: 16,
});
ok(
  fat.batchCap === maxBatchForScratch(kernel.layout, 16, storageLimit),
  `K=16 trainer resolves to its own cap ${fat.batchCap} without throwing`
);

console.log("\n§4 THE COVER OBJECTIVE IS COMPUTE-BOUND, NOT MEMORY-BOUND");
// Raising the general cap must NOT raise this one. The cover backward scans
// the whole batch per cover sample (O(COVER_SAMPLES·n²) aggregate) while its
// scratch stays small, so the memory bound resolves to the full 16384 while a
// real step there costs ~1.8 SECONDS (measured) — driver-TDR territory, and
// nothing in src/ handles device-lost. Caught by review, 2026-08-17.
{
  const coverTrainer = new FusedTrainer(device, kernel.layout, {
    batchCap: MAX_BATCH,
    kSteps: 1,
    loss: COVER_FIELD_LOSS,
  });
  const memoryOnly = maxBatchForScratch(kernel.layout, 1, storageLimit);
  ok(
    memoryOnly > COVER_MAX_BATCH,
    `memory alone would allow ${memoryOnly} for this layout — it cannot see the compute wall`
  );
  ok(
    coverTrainer.batchCap === COVER_MAX_BATCH,
    `a W_COVER objective caps at ${COVER_MAX_BATCH}, not ${memoryOnly}`
  );
  const linearTrainer = new FusedTrainer(device, kernel.layout, {
    batchCap: MAX_BATCH,
    kSteps: 1,
  });
  ok(
    linearTrainer.batchCap === memoryOnly,
    "a linear objective on the same layout still gets the full memory-bound cap"
  );
  ok(
    maxBatchForLoss(undefined) === MAX_BATCH &&
      maxBatchForLoss({ ...COVER_FIELD_LOSS, W_COVER: 0 }) === MAX_BATCH,
    "only a NONZERO W_COVER triggers the compute cap"
  );
}

console.log("\n§5 CAP ARITHMETIC (deterministic)");
// Independent of this box: a byte budget that admits exactly 3 samples.
const perSample = scratchBytes(kernel.layout, 1, K);
ok(
  maxBatchForScratch(kernel.layout, K, perSample * 3) === 3,
  "maxBatchForScratch returns the largest batch that fits the budget"
);
ok(
  maxBatchForScratch(kernel.layout, K, perSample * 3.5) === 3,
  "a partial sample never counts — floor, not round"
);
ok(
  maxBatchForScratch(kernel.layout, K, 1) === 1,
  "a hopeless budget still yields a runnable batch of 1, not 0"
);
ok(
  maxBatchForScratch(kernel.layout, K, Number.MAX_SAFE_INTEGER) === MAX_BATCH,
  "an unlimited budget stops at MAX_BATCH"
);

console.log(
  failures === 0
    ? "\nALL TRAIN BATCH CAP CHECKS PASS"
    : `\n${failures} CHECK(S) FAILED`
);
process.exit(failures === 0 ? 0 : 1);
