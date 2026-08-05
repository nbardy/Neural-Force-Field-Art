/** Real-GPU forward/Jacobian gate for wrap, bounce, and edge-respawn modes. */
import { setupGlobals } from "bun-webgpu";
import * as tf from "@tensorflow/tfjs";
import {
  borderJacobianExpr,
  emitBorder,
  type BorderMode,
} from "../src/render/webgpu/advect_wgsl";
import {
  Adversary,
  defaultAdversaryConfig,
} from "../src/core/gan/adversary";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
  UNIFORM: 64, STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("border_modes_test: no WebGPU adapter");
const device: GPUDevice = await adapter.requestDevice();

const common = `
fn pcg(v : u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let t = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (t >> 22u) ^ t;
}
fn rand01(x : u32) -> f32 { return f32(x) * 2.3283064365386963e-10; }
@group(0) @binding(0) var<storage, read_write> io : array<f32>;
`;

async function run(mode: BorderMode): Promise<Float32Array> {
  const code = `${common}
@compute @workgroup_size(1)
fn main() {
  let res = vec2f(100.0, 80.0);
  var p = vec2f(-2.0, 85.0);
  var v = vec2f(-3.0, 7.0);
  let q = p;
  let jac = ${borderJacobianExpr(mode, "q", "res")};
  ${emitBorder(mode, "p", "v", "res", "123u")}
  io[0]=p.x; io[1]=p.y; io[2]=v.x; io[3]=v.y; io[4]=jac.x; io[5]=jac.y;
}`;
  const module = device.createShaderModule({ code });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const out = device.createBuffer({
    size: 24, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const read = device.createBuffer({
    size: 24, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const bind = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1); pass.end();
  enc.copyBufferToBuffer(out, 0, read, 0, 24);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(read.getMappedRange().slice(0));
  read.unmap(); read.destroy(); out.destroy();
  return result;
}

let failures = 0;
const ok = (c: boolean, m: string) => {
  if (!c) failures++;
  console.log(`${c ? "PASS" : "FAIL"}  ${m}`);
};
const wrap = await run({ tag: "wrap" });
ok(wrap[0] === 98 && wrap[1] === 5 && wrap[4] === 1 && wrap[5] === 1,
  `wrap p=(${wrap[0]},${wrap[1]}) J=(${wrap[4]},${wrap[5]})`);
const bounce = await run({ tag: "bounce" });
ok(bounce[0] === 2 && bounce[1] === 75 && bounce[2] === 3 && bounce[3] === -7 &&
   bounce[4] === -1 && bounce[5] === -1,
  `bounce p=(${bounce[0]},${bounce[1]}) v=(${bounce[2]},${bounce[3]}) J=(${bounce[4]},${bounce[5]})`);
const reset = await run({ tag: "reset" });
ok(reset[0] >= 0 && reset[0] < 100 && reset[1] >= 0 && reset[1] < 80 &&
   reset[2] === 0 && reset[3] === 0 && reset[4] === 0 && reset[5] === 0,
  `edge respawn in bounds, v=0, J=0 (p=${reset[0].toFixed(2)},${reset[1].toFixed(2)})`);

/* Relational-observer geometry must follow the boundary topology. The same
 * seam pair is near on wrap's torus and far for bounce/reset's rectangle. */
await tf.setBackend("cpu");
await tf.ready();
const pos = tf.tensor2d([[0.99, 0.5], [0.01, 0.5]]);
const signal = tf.tensor2d([[0, 0], [1, 0]]);
const idx = new Int32Array([0, 1]);
const encodePair = (observerGeometry: "periodic" | "euclidean") => {
  const adv = new Adversary(
    defaultAdversaryConfig(
      { tag: "single" },
      { tag: "pair-rotation" },
      observerGeometry
    )
  );
  const sample = adv.encodeSignal(pos, signal, idx);
  const u = sample.u.dataSync()[0];
  const y = Array.from(sample.y.dataSync());
  sample.u.dispose();
  sample.y.dispose();
  adv.dispose();
  return { u, y };
};
const torusObserver = encodePair("periodic");
const rectangleObserver = encodePair("euclidean");
ok(
  Math.abs(torusObserver.u - 0.02) < 2e-6 &&
    Math.abs(rectangleObserver.u - 0.98) < 2e-6,
  `observer seam distance: periodic=${torusObserver.u.toFixed(6)}, ` +
    `euclidean=${rectangleObserver.u.toFixed(6)}`
);
ok(
  torusObserver.y[0] > 0.999 &&
    rectangleObserver.y[0] < -0.999 &&
    Math.abs(torusObserver.y[1]) < 1e-6 &&
    Math.abs(rectangleObserver.y[1]) < 1e-6,
  "observer frame follows the selected shortest-periodic vs raw-Euclidean delta"
);
let implicitGeometryThrew = false;
try {
  const bad = defaultAdversaryConfig(
    { tag: "single" },
    { tag: "pair-rotation" },
    undefined as never
  );
  new Adversary(bad);
} catch {
  implicitGeometryThrew = true;
}
ok(
  implicitGeometryThrew,
  "missing observer geometry throws instead of silently assuming a torus"
);
pos.dispose();
signal.dispose();

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
