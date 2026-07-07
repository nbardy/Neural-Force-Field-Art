// Does forcing tfjs-webgpu to keep tiny ops on the GPU (no CPU handoff) speed
// up the REAL learn step? Replicates main.ts helmholtzChaosLoss training.
import { setupGlobals } from "bun-webgpu";
setupGlobals();
{
  const gpu = (navigator as any).gpu;
  const orig = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (o?: any) => {
    const a = await orig(o);
    if (!a) return a;
    if (!a.requestAdapterInfo) a.requestAdapterInfo = async () => a.info ?? {};
    const od = a.requestDevice.bind(a);
    a.requestDevice = async (d?: any) => {
      const dd = d ? { ...d, requiredFeatures: [...(d.requiredFeatures ?? [])].flat(Infinity) } : d;
      if (dd) delete dd.requiredLimits;
      return od(dd);
    };
    return a;
  };
}
const tf = await import("@tensorflow/tfjs");
(globalThis as any).window ??= globalThis;
(globalThis as any).GPUBuffer ??= class {};
(globalThis as any).GPUTexture ??= class {};
await import("@tensorflow/tfjs-backend-webgpu");
const { HelmholtzField } = await import("../src/core/field/helmholtz");
const { isotropyLoss } = await import("../src/core/losses");

await tf.setBackend("webgpu");
await tf.ready();
console.log("backend:", tf.getBackend(), "| default CPU handoff threshold:",
  tf.env().getNumber("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD"));

const W = 800, H = 600, N = 256, HH = 1e-2;
const wh = tf.tensor2d([[W, H]]);

function makeLoss(field: any) {
  return () => tf.tidy(() => {
    const tp = tf.randomUniform([N, 2], 0, 1).mul(wh) as any;
    const posNorm = tp.div(wh) as any;
    const force = field.forces(posNorm).mul(3.5) as any;
    const vel = force.mul(0.99).clipByValue(-26, 26) as any;
    const newPos = tp.add(vel).mod(wh) as any;
    const pn = newPos.div(wh) as any;
    const all = tf.concat([pn, pn.add(tf.tensor2d([[HH, 0]])), pn.add(tf.tensor2d([[0, HH]]))], 0) as any;
    const allF = field.forces(all);
    const f0 = allF.slice([0, 0], [N, -1]);
    const fx = allF.slice([N, 0], [N, -1]);
    const fy = allF.slice([2 * N, 0], [N, -1]);
    const sepx = fx.sub(f0).square().sum(1);
    const sepy = fy.sub(f0).square().sum(1);
    const sep = sepx.add(sepy).add(1e-12).sqrt().div(HH * 1.4142 + 1e-9);
    const chaos = sep.add(1e-6).log().mean().neg();
    const dFxdx = fx.slice([0, 0], [-1, 1]).sub(f0.slice([0, 0], [-1, 1])).div(HH);
    const dFydy = fy.slice([0, 1], [-1, 1]).sub(f0.slice([0, 1], [-1, 1])).div(HH);
    const div = dFxdx.add(dFydy).square().mean();
    const iso = isotropyLoss(force);
    return chaos.add(iso).add(div.mul(0.5)).asScalar();
  });
}

async function bench(label: string): Promise<void> {
  const field = new HelmholtzField({ alpha: 0.7 });
  const opt = tf.train.adam(0.01);
  const lossFn = makeLoss(field);
  const step = () => opt.minimize(lossFn as any, false, field.trainableWeights);
  for (let i = 0; i < 15; i++) step();
  await (field.trainableWeights[0] as any).data(); // fence
  const t0 = performance.now();
  for (let i = 0; i < 60; i++) step();
  await (field.trainableWeights[0] as any).data(); // fence
  console.log(`${label}: ${((performance.now() - t0) / 60).toFixed(2)} ms/step`);
  opt.dispose(); field.dispose();
}

await bench("tfjs learn, DEFAULT handoff  ");
tf.env().set("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD", 0);
await bench("tfjs learn, handoff DISABLED ");
tf.env().set("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD", 100000);
await bench("tfjs learn, handoff EVERYTHING");
process.exit(0);
