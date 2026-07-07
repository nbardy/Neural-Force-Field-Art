/**
 * Headless verification for the compute-splat renderer
 * (src/render/webgpu/splat.ts) on a REAL WebGPU adapter (Dawn/Metal) via
 * bun-webgpu — no browser, no tfjs backend (explicit device + canvas:null).
 *
 *   bun tools/splat_test.ts
 *
 * What it checks (end-to-end: render → copyTextureToBuffer → pixel asserts,
 * plus raw accumulation-buffer readbacks for the kernel-shape tests):
 *   1. three known particles land as brighter pixels with the exact
 *      speed-palette colours; a far-away pixel equals the tonemapped
 *      background (±1/255)
 *   2. a half-pixel particle spreads energy across the 4 texels nearest to
 *      its radial-cone centre (equal weights ≈ 0.25, by symmetry)
 *   3. decay=0.9: the previous location keeps a dimmer-but-visible trail
 *   4. classes=3: indices hashing to the three classes give distinct hues
 *      matching the pcg cosine palette
 *   5. energy conservation: the in-shader weight normalization deposits
 *      total energy 4096 per unit colour channel at radius 1.0 AND 2.5 (±1%)
 *   6. dpr=2: a CSS-space particle deposits centred at native (css*2) texels
 *      in a 2x-sized accumulator
 *   7. roundness: cone weights depend only on distance — (±2,0)/(0,±2) taps
 *      equal, (±2,±2) corners exactly zero (a disc, not the old 2x2 square)
 *   8. bench: 1M particles, 200 frames of the 3-pass render @1920x1080
 *      (radius 1.25, dpr 1)
 *
 * Expected pixel values come from a float64/f32 mirror of the pipeline
 * (radial cone kernel with two-pass normalization → fixed-point floor at
 * 4096 → background + energy → c/(1+c) → gamma 1/2.2), so the asserts are
 * numeric, not just "something changed".
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

// raw accumulation-buffer readback (native W×H×3 u32 fixed-point counts) —
// for the kernel-shape tests (conservation / dpr / roundness) that need the
// deposited energy itself, not its 8-bit tonemapped image
async function readAcc(rr: SplatRenderer, w: number, h: number): Promise<Uint32Array> {
  const size = w * h * 3 * 4;
  const staging = device.createBuffer({
    size,
    usage: USAGE.MAP_READ | USAGE.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(rr.accumBuffer!, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1 /* GPUMapMode.READ */);
  const data = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

// float64 mirror of the render pipeline ------------------------------------
const BG: RGB = [10, 20, 30];

// f32 mirror of SPLAT_WGSL's RADIAL CONE kernel. Texel "centres" sit at
// INTEGER coords (the old bilinear's convention — an integer-position
// particle is centred on that texel index); weight = max(0, 1 - d/r) over
// the integer box covering the circle, then normalized by the f32 running
// sum in the shader's loop order (y outer, x inner) so every particle
// deposits total energy 1.0. Mirrors the shader op-for-op.
const DOT_RADIUS = 1.25; // SplatRenderer default (CSS px; native = r*dpr)
interface Tap { x: number; y: number; w: number }
function radialTaps(pxx: number, pyy: number, r: number): Tap[] {
  const rf = f32(r);
  const x0 = Math.ceil(f32(pxx - rf));
  const x1 = Math.floor(f32(pxx + rf));
  const y0 = Math.ceil(f32(pyy - rf));
  const y1 = Math.floor(f32(pyy + rf));
  const cone = (x: number, y: number): number => {
    const d = f32(Math.hypot(f32(x - pxx), f32(y - pyy)));
    return Math.max(0, f32(1 - f32(d / rf)));
  };
  let wsum = 0;
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) wsum = f32(wsum + cone(x, y));
  const taps: Tap[] = [];
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++)
      taps.push({ x, y, w: f32(cone(x, y) / wsum) });
  return taps;
}
// normalized cone weight landing on texel (x,y) — 0 outside the cone
const tapWeight = (pxx: number, pyy: number, r: number, x: number, y: number): number =>
  radialTaps(pxx, pyy, r).find((t) => t.x === x && t.y === y)?.w ?? 0;

// splat quantisation: per-tap deposit u32((col * wgt) * 4096), all f32
const countsOf = (col: number[], wgt: number): number[] =>
  col.map((c) => Math.floor(f32(f32(f32(c) * f32(wgt)) * FIXED)));
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
// centre-tap weight of an INTEGER-position particle at the default radius:
// wsum = 1 + 4*(1 - 1/1.25) = 1.8 -> centre gets 1/1.8 of the 4096 energy
// (the radial kernel spreads the rest onto the 4 edge neighbours at d=1)
const W0 = tapWeight(0, 0, DOT_RADIUS, 0, 0);
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
  const want = outColor(countsOf(speedColor(vx, vy), W0));
  check(
    near(got, want, 2) && lum(got) > lum(bgOut) + 10,
    `particle @(${x},${y}) ${label}: got ${fmt(got)} want ${fmt(want)}, brighter than bg`
  );
});
{
  // subpixel-phase proof: (100.5, 50.5) sits equidistant (d = sqrt(0.5)) from
  // its 4 nearest texel centres — the normalized cone gives each exactly 1/4
  // of the particle's energy (the box's other taps are outside the cone).
  const quad: [number, number][] = [[100, 50], [101, 50], [100, 51], [101, 51]];
  const got = quad.map(([x, y]) => px(img1, x, y));
  const wants = quad.map(([x, y]) =>
    outColor(countsOf(speedColor(0, 0), tapWeight(100.5, 50.5, DOT_RADIUS, x, y)))
  );
  const ok = got.every((g, i) => near(g, wants[i], 2) && lum(g) > lum(bgOut) + 10);
  check(
    ok,
    `half-pixel splat spreads 4-way @~0.25 weight: got ${got.map(fmt).join(" ")} want ${wants.map(fmt).join(" ")}`
  );
}

// --------------------------------------------------------------------------
// 3: decay=0.9 ghost trail — render, move particle, render again
// --------------------------------------------------------------------------
{
  device.queue.writeBuffer(posB, 0, new Float32Array([60, 60]));
  device.queue.writeBuffer(velB, 0, new Float32Array([0, 0]));
  // Decay is now FUSED at END of frame, so frame 1 must deposit A AND leave it
  // (faded once) for frame 2 to show as a trail. Pre-clear the prior test's
  // energy, then run frame 1 at decay 0.9 (its own end-of-frame decay fades A
  // to 0.9 before frame 2 renders).
  r.clearAccum();
  r.decay = 0.9;
  r.render(posB, velB, 1, W, H);

  device.queue.writeBuffer(posB, 0, new Float32Array([160, 60]));
  r.render(posB, velB, 1, W, H);

  const img = await readTex(r.offscreen!.texture!, W, H);
  const trail = px(img, 60, 60);
  const fresh = px(img, 160, 60);
  const freshCounts = countsOf(speedColor(0, 0), W0);
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
    const want = outColor(countsOf(classColor(idxByClass[c], 3), W0));
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
// 5: energy conservation — the two-pass in-shader normalization must deposit
//    a TOTAL of 4096 counts per unit colour channel no matter the radius or
//    subpixel phase. Blue of the v=0 colour is exactly 1.0, so Σ(blue) over
//    the whole accumulator ≈ 4096; each tap floors independently, so the sum
//    can only fall short by < tapCount counts (≪ 1%).
// --------------------------------------------------------------------------
{
  r.classes = 0;
  // decay is FUSED into tonemap now; decay=1 leaves the raw deposit readable,
  // and clearAccum gives each radius a known-clean buffer (was: decay=0
  // hard-clear, which under the fused decay would zero the readback). Native
  // radius is capped at 1.6 (NATIVE_RADIUS_MAX), so test at 1.0 and 1.6 — the
  // wsum normalization conserves energy at ANY radius.
  r.decay = 1;
  const posE = mkBuf(new Float32Array([40.3, 20.7])); // deliberately subpixel
  const velE = mkBuf(new Float32Array([0, 0]));
  for (const radius of [1.0, 1.6]) {
    r.clearAccum();
    r.radius = radius;
    r.render(posE, velE, 1, W, H);
    const acc = await readAcc(r, W, H);
    let sum = 0;
    for (let i = 2; i < acc.length; i += 3) sum += acc[i]; // blue channel
    check(
      Math.abs(sum - FIXED) <= FIXED * 0.01,
      `energy conservation r=${radius}: Σ blue counts ${sum} ≈ ${FIXED} ±1%`
    );
  }
  r.radius = DOT_RADIUS;
  posE.destroy();
  velE.destroy();
}

// --------------------------------------------------------------------------
// 6: dpr=2 — CSS coords scale into a 2x native accumulator. Particle at CSS
//    (10.25, 7.5) = native (20.5, 15): x is mirror-symmetric about the
//    20/21 column boundary and y is centred on row 15, so the energy
//    centroid is exactly (20.5, 15) in native texels.
// --------------------------------------------------------------------------
{
  const W2 = 64, H2 = 32; // CSS size; native accumulator = 128x64
  const NW = 128, NH = 64;
  const r2 = new SplatRenderer(null, {
    background: BG,
    maxSpeed: 4,
    decay: 1, // fused decay: decay=1 leaves the single frame's raw deposit
    exposure: 1,
    dpr: 2,
    device,
  });
  const posD = mkBuf(new Float32Array([10.25, 7.5]));
  const velD = mkBuf(new Float32Array([0, 0]));
  r2.render(posD, velD, 1, W2, H2);
  const acc = await readAcc(r2, NW, NH);
  const blue = (x: number, y: number): number => acc[(y * NW + x) * 3 + 2];
  let sw = 0, sx = 0, sy = 0;
  for (let y = 0; y < NH; y++)
    for (let x = 0; x < NW; x++) {
      const c = blue(x, y);
      sw += c;
      sx += c * x;
      sy += c * y;
    }
  const cx = sx / sw, cy = sy / sw;
  check(
    Math.abs(cx - 20.5) < 0.02 && Math.abs(cy - 15) < 0.02,
    `dpr=2 deposit centroid (${cx.toFixed(3)}, ${cy.toFixed(3)}) == native (20.5, 15)`
  );
  check(
    blue(20, 15) === blue(21, 15) && blue(20, 15) > 0,
    `dpr=2 columns 20/21 mirror-equal about x=20.5: ${blue(20, 15)} == ${blue(21, 15)}`
  );
  // the offscreen tonemap target is native-sized too (readable end to end)
  const imgD = await readTex(r2.offscreen!.texture!, NW, NH);
  check(
    lum(px(imgD, 20, 15)) > lum(bgOut) + 10,
    `dpr=2 offscreen texture is native 128x64 and lit at (20,15): ${fmt(px(imgD, 20, 15))}`
  );
  r2.destroy();
  posD.destroy();
  velD.destroy();
}

// --------------------------------------------------------------------------
// 7: roundness — cone weights depend only on DISTANCE. At the native radius
//    cap (1.6), particle on a texel centre: the 4 d=1 edge taps (±1,0)/(0,±1)
//    are bit-equal, the 4 d=√2≈1.414 corner taps (±1,±1) are bit-equal but
//    STRICTLY DIMMER (radial falloff — a square/box-uniform kernel would make
//    corner==edge), and the d=2 axis taps are EXACTLY zero (2 > 1.6, bounded
//    disc). Together: a radial cone, not a square.
// --------------------------------------------------------------------------
{
  r.radius = 1.6; // == NATIVE_RADIUS_MAX at dpr 1
  r.decay = 1; // fused decay; decay=1 keeps the raw deposit for readback
  r.classes = 0;
  r.clearAccum();
  const posR = mkBuf(new Float32Array([64, 64]));
  const velR = mkBuf(new Float32Array([0, 0]));
  r.render(posR, velR, 1, W, H);
  const acc = await readAcc(r, W, H);
  const blue = (x: number, y: number): number => acc[(y * W + x) * 3 + 2];
  const edges = [blue(63, 64), blue(65, 64), blue(64, 63), blue(64, 65)]; // d=1
  const corners = [blue(63, 63), blue(65, 63), blue(63, 65), blue(65, 65)]; // d=√2
  const far = [blue(62, 64), blue(66, 64)]; // d=2 > 1.6
  check(
    edges.every((e) => e === edges[0]) && edges[0] > 0,
    `roundness: d=1 edges (±1,0)/(0,±1) equal & nonzero: [${edges.join(",")}]`
  );
  check(
    corners.every((c) => c === corners[0]) && corners[0] > 0 && corners[0] < edges[0],
    `roundness: d=√2 corners equal, nonzero, DIMMER than edges (radial falloff): [${corners.join(",")}] < ${edges[0]}`
  );
  check(
    far.every((c) => c === 0),
    `roundness: d=2 axis taps outside the cone == 0: [${far.join(",")}]`
  );
  r.radius = DOT_RADIUS;
  posR.destroy();
  velR.destroy();
}

// --------------------------------------------------------------------------
// 8: bench — 1M particles, 200 frames of splat+tonemap(fused decay) @1920x1080
//    (radius 1.25, dpr 1 — the shipped defaults)
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
  r.radius = 1.25;  // shipped default (dpr is 1 on this renderer)

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
    `\nBENCH  splat render @ ${N.toLocaleString()} particles, ${BW}x${BH}, ` +
      `radius 1.25, dpr 1: ${ms.toFixed(3)} ms/frame (${(1000 / ms).toFixed(0)} fps) ` +
      `— target ≤10 ms (radial kernel: ~9 taps vs the old 4); ` +
      `GPU is shared, so contention inflates this`
  );
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
