/**
 * Headless verification for the compute-splat renderer
 * (src/render/webgpu/splat.ts) on a REAL WebGPU adapter (Dawn/Metal) via
 * bun-webgpu — no browser, no tfjs backend (explicit device + canvas:null).
 *
 *   bun tools/splat_test.ts
 *
 * What it checks (end-to-end: render → copyTextureToBuffer → pixel asserts):
 *   1. three known particles land as brighter pixels with the exact
 *      speed-palette colours; a far-away pixel equals the tonemapped
 *      background (±1/255)
 *   2. a half-pixel particle spreads energy across its 2x2 bilinear footprint
 *   3. decay=0.9: the previous location keeps a dimmer-but-visible trail
 *   4. classes=3: indices hashing to the three classes give distinct hues
 *      matching the pcg cosine palette
 *   5. bench: 1M particles, 200 frames of the 3-pass render @1920x1080
 *
 * Expected pixel values come from a float64 mirror of the pipeline
 * (fixed-point floor at 4096 → background + energy → c/(1+c) → gamma 1/2.2),
 * so the asserts are numeric, not just "something changed".
 */
import { setupGlobals } from "bun-webgpu";
import { SplatRenderer, SPLAT_FIXED_POINT } from "../src/render/webgpu/splat";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
  UNIFORM: 64, STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };
(globalThis as any).GPUTextureUsage ??= {
  COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8,
  RENDER_ATTACHMENT: 16,
};

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: any = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}\n`);

const USAGE = (globalThis as any).GPUBufferUsage;
let failures = 0;
const check = (ok: boolean, msg: string): void => {
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`);
};

// --------------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------------
// deterministic RNG (bench inputs must reproduce)
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// JS port of the WGSL pcg hash (copied from tools/kernel_test.ts — class
// derivation must match bit-for-bit)
function pcgJS(v: number): number {
  const st = (Math.imul(v, 747796405) + 2891336453) >>> 0;
  const t = Math.imul(((st >>> (((st >>> 28) + 4) & 31)) ^ st) >>> 0, 277803737) >>> 0;
  return ((t >>> 22) ^ t) >>> 0;
}
const CLASS_SALT = 2166136261;

const FIXED = SPLAT_FIXED_POINT;
const f32 = Math.fround;
type RGB = [number, number, number];

function mkBuf(data: Float32Array): any {
  const b = device.createBuffer({
    size: data.byteLength,
    usage: USAGE.STORAGE | USAGE.COPY_DST,
  });
  device.queue.writeBuffer(b, 0, data);
  return b;
}

interface Img { data: Uint8Array; bpr: number }
async function readTex(tex: any, w: number, h: number): Promise<Img> {
  const bpr = Math.ceil((w * 4) / 256) * 256; // 256-byte-aligned bytesPerRow
  const staging = device.createBuffer({
    size: bpr * h,
    usage: USAGE.MAP_READ | USAGE.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: tex },
    { buffer: staging, bytesPerRow: bpr, rowsPerImage: h },
    { width: w, height: h, depthOrArrayLayers: 1 }
  );
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1 /* GPUMapMode.READ */);
  const data = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return { data, bpr };
}

const px = (img: Img, x: number, y: number): RGB => [
  img.data[y * img.bpr + x * 4],
  img.data[y * img.bpr + x * 4 + 1],
  img.data[y * img.bpr + x * 4 + 2],
];

// float64 mirror of the render pipeline ------------------------------------
const BG: RGB = [10, 20, 30];

// splat quantisation: energy -> fixed-point counts (u32 truncation)
const countsOf = (col: number[], wgt: number): number[] =>
  col.map((c) => Math.floor(f32(c) * wgt * FIXED));
// decay quantisation: acc = u32(f32(acc) * decay)
const decayed = (counts: number[], d: number): number[] =>
  counts.map((c) => Math.floor(f32(c * f32(d))));
// tonemap: bg + energy*exposure -> c/(1+c) -> gamma -> unorm8
const outColor = (counts: number[], exposure = 1): RGB =>
  counts.map((cnt, i) => {
    let c = BG[i] / 255 + (cnt / FIXED) * exposure;
    c = c / (1 + c);
    return Math.round(Math.pow(c, 1 / 2.2) * 255);
  }) as RGB;

// points.ts colour formulas (t = clamped speed fraction)
const speedColor = (vx: number, vy: number, maxSpeed = 4): number[] => {
  const t = Math.min(1, Math.max(0, Math.hypot(vx, vy) / maxSpeed));
  const a = [0.25, 0.55, 1.0];
  const b = [1.0, 0.55, 0.2];
  return a.map((av, i) => av + (b[i] - av) * t);
};
const classColor = (iid: number, classes: number, t = 0): number[] => {
  const cls = pcgJS((iid ^ CLASS_SALT) >>> 0) % classes;
  const hue = cls * 2.399963;
  return [0, 2.0944, 4.1888].map(
    (o) => (0.55 + 0.45 * Math.cos(hue + o)) * (0.55 + 0.45 * t)
  );
};

const near = (a: RGB, b: RGB, tol: number): boolean =>
  a.every((v, i) => Math.abs(v - b[i]) <= tol);
const lum = (c: RGB): number => c[0] + c[1] + c[2];
const maxDiff = (a: RGB, b: RGB): number =>
  Math.max(...a.map((v, i) => Math.abs(v - b[i])));
const fmt = (c: RGB): string => `(${c.join(",")})`;

// --------------------------------------------------------------------------
// 1+2: known particles, background, bilinear footprint  (256x128, decay 0)
// --------------------------------------------------------------------------
const W = 256, H = 128;

device.pushErrorScope("validation");
const r = new SplatRenderer(null, {
  background: BG,
  maxSpeed: 4,
  classes: 0,
  decay: 0,
  exposure: 1,
  device,
});

// 3 integer-position particles (full weight in one texel) + 1 half-pixel
const pos = new Float32Array([32, 32, 128, 64, 200, 100, 100.5, 50.5]);
const vel = new Float32Array([0, 0, 4, 0, 2, 0, 0, 0]);
const posB = mkBuf(pos);
const velB = mkBuf(vel);
r.render(posB, velB, 4, W, H);
{
  const err = await device.popErrorScope();
  if (err) {
    failures++;
    console.log(`FAIL  validation error during construct/render: ${err.message}`);
  }
}
const img1 = await readTex(r.offscreen!.texture!, W, H);

const bgOut = outColor([0, 0, 0]);
{
  const far = px(img1, 16, 100); // nowhere near any particle
  check(
    near(far, bgOut, 1),
    `far pixel == tonemapped background ±1: got ${fmt(far)} want ${fmt(bgOut)}`
  );
}
([
  [32, 32, 0, 0, "v=0 (blue)"],
  [128, 64, 4, 0, "v=maxSpeed (orange)"],
  [200, 100, 2, 0, "v=half (mix)"],
] as [number, number, number, number, string][]).forEach(([x, y, vx, vy, label]) => {
  const got = px(img1, x, y);
  const want = outColor(countsOf(speedColor(vx, vy), 1));
  check(
    near(got, want, 2) && lum(got) > lum(bgOut) + 10,
    `particle @(${x},${y}) ${label}: got ${fmt(got)} want ${fmt(want)}, brighter than bg`
  );
});
{
  // bilinear proof: (100.5, 50.5) puts weight 0.25 on each 2x2 neighbour
  const want = outColor(countsOf(speedColor(0, 0), 0.25));
  const quad: [number, number][] = [[100, 50], [101, 50], [100, 51], [101, 51]];
  const got = quad.map(([x, y]) => px(img1, x, y));
  const ok = got.every((g) => near(g, want, 2) && lum(g) > lum(bgOut) + 10);
  check(
    ok,
    `half-pixel splat spreads 2x2 @0.25 weight: got ${got.map(fmt).join(" ")} want ${fmt(want)}`
  );
}

// --------------------------------------------------------------------------
// 3: decay=0.9 ghost trail — render, move particle, render again
// --------------------------------------------------------------------------
{
  device.queue.writeBuffer(posB, 0, new Float32Array([60, 60]));
  device.queue.writeBuffer(velB, 0, new Float32Array([0, 0]));
  r.decay = 0; // frame 1 hard-clears the previous test's energy
  r.render(posB, velB, 1, W, H);

  r.decay = 0.9;
  device.queue.writeBuffer(posB, 0, new Float32Array([160, 60]));
  r.render(posB, velB, 1, W, H);

  const img = await readTex(r.offscreen!.texture!, W, H);
  const trail = px(img, 60, 60);
  const fresh = px(img, 160, 60);
  const freshCounts = countsOf(speedColor(0, 0), 1);
  const wantTrail = outColor(decayed(freshCounts, 0.9));
  const wantFresh = outColor(freshCounts);
  check(
    near(trail, wantTrail, 2) && lum(trail) > lum(bgOut) + 10 && lum(trail) < lum(fresh),
    `decay 0.9 trail: old spot ${fmt(trail)} ≈ ${fmt(wantTrail)}, > bg, < fresh ${fmt(fresh)}`
  );
  check(
    near(fresh, wantFresh, 2),
    `decay 0.9 fresh splat: ${fmt(fresh)} ≈ ${fmt(wantFresh)}`
  );
}

// --------------------------------------------------------------------------
// 4: classes=3 — three indices hashing to the three classes, distinct hues
// --------------------------------------------------------------------------
{
  const idxByClass = [-1, -1, -1];
  for (let i = 0; idxByClass.includes(-1) && i < 10000; i++) {
    const c = pcgJS((i ^ CLASS_SALT) >>> 0) % 3;
    if (idxByClass[c] === -1) idxByClass[c] = i;
  }
  const n = Math.max(...idxByClass) + 1;
  // park unused slots offscreen (negative coords fail the bounds guard)
  const posC = new Float32Array(2 * n).fill(-100);
  const velC = new Float32Array(2 * n);
  const spots: [number, number][] = [[40, 40], [120, 64], [200, 90]];
  idxByClass.forEach((iid, c) => {
    posC[2 * iid] = spots[c][0];
    posC[2 * iid + 1] = spots[c][1];
  });
  const posBC = mkBuf(posC);
  const velBC = mkBuf(velC);
  r.classes = 3;
  r.decay = 0;
  r.render(posBC, velBC, n, W, H); // new buffers → exercises bind-group swap
  const img = await readTex(r.offscreen!.texture!, W, H);

  const got = spots.map(([x, y]) => px(img, x, y));
  got.forEach((g, c) => {
    const want = outColor(countsOf(classColor(idxByClass[c], 3), 1));
    check(
      near(g, want, 3),
      `classes=3: iid ${idxByClass[c]} (class ${c}) palette colour ${fmt(g)} ≈ ${fmt(want)}`
    );
  });
  check(
    maxDiff(got[0], got[1]) > 10 &&
      maxDiff(got[1], got[2]) > 10 &&
      maxDiff(got[0], got[2]) > 10,
    `classes=3: three hues pairwise distinct ${got.map(fmt).join(" ")}`
  );
  posBC.destroy();
  velBC.destroy();
}

// --------------------------------------------------------------------------
// 5: bench — 1M particles, 200 frames of decay+splat+tonemap @1920x1080
// --------------------------------------------------------------------------
{
  const BW = 1920, BH = 1080, N = 1_000_000;
  const rnd = mulberry32(7);
  const posN = new Float32Array(2 * N);
  const velN = new Float32Array(2 * N);
  for (let i = 0; i < N; i++) {
    posN[2 * i] = rnd() * BW;
    posN[2 * i + 1] = rnd() * BH;
    velN[2 * i] = (rnd() - 0.5) * 8;
    velN[2 * i + 1] = (rnd() - 0.5) * 8;
  }
  const posBN = mkBuf(posN);
  const velBN = mkBuf(velN);
  r.decay = 0.9;    // realistic trails config
  r.classes = 3;    // full colour path (pcg + palette)
  r.exposure = 1;

  // fence: tiny texture readback forces the queue to drain
  const fence = async (): Promise<void> => {
    const staging = device.createBuffer({ size: 256, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: r.offscreen!.texture! },
      { buffer: staging, bytesPerRow: 256 },
      { width: 1, height: 1, depthOrArrayLayers: 1 }
    );
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    staging.unmap();
    staging.destroy();
  };

  const WARM = 20, TIMED = 200;
  for (let s = 0; s < WARM; s++) r.render(posBN, velBN, N, BW, BH);
  await fence(); // settle (also proves the 256x128 → 1920x1080 realloc path)
  const t0 = performance.now();
  for (let s = 0; s < TIMED; s++) r.render(posBN, velBN, N, BW, BH);
  await fence();
  const ms = (performance.now() - t0) / TIMED;
  console.log(
    `\nBENCH  splat render @ ${N.toLocaleString()} particles, ${BW}x${BH}: ` +
      `${ms.toFixed(3)} ms/frame (${(1000 / ms).toFixed(0)} fps) — target ≤8 ms; ` +
      `GPU is shared, so contention inflates this`
  );
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
