/**
 * Headless verification of the splat renderer's STROKE styles
 * (src/render/webgpu/splat.ts: style = "dot" | "vel" | "curl") on a REAL
 * WebGPU adapter (Dawn/Metal) via bun-webgpu — no browser, no tfjs backend.
 *
 *   bun tools/splat_stroke_test.ts
 *
 * What it checks (raw accumulation-buffer readbacks against a float64/f32
 * mirror of the stroke sampling math — numeric, not "something changed"):
 *   1. REGRESSION: style="dot" (the default) renders a 300-particle fixture
 *      to the EXACT accumulator bytes captured from the pre-stroke renderer
 *      (FNV-1a hash + total-count sum, bit-exact: u32 atomicAdds are order-
 *      independent). Protects the shipped look.
 *   2. ENERGY: the same deterministic particle set deposits equal total
 *      energy under dot / vel / curl (±1%) AND matches the analytic
 *      Σ 4096·(r+g+b) expectation — the per-sample taper weights are
 *      normalized so per-particle energy is style-invariant (main.ts's
 *      count-adaptive auto-exposure contract).
 *   3. GEOMETRY (vel): one particle with known pos/v/T stamps along the
 *      straight backward line p(t) = pos - t·T·v — every predicted sample
 *      texel is lit, energy is confined to the line, the head outweighs the
 *      tapered tail, and the energy centroid matches the mirror.
 *   4. GEOMETRY (curl): with a = v - vPrev the samples follow
 *      p(t) = pos - τv + ½τ²a — the centroid displaces off the straight
 *      line by the mirror-predicted amount (toward +a: the backward-Taylor
 *      trail of an accelerating particle), and the straight-line tail
 *      position is dark.
 *   5. RESET GUARD: v=0 with a large stale prevVel (the fused random-reset
 *      case) — the |a| ≤ 1.5|v| clamp kills the spurious arc: all energy
 *      lands AT pos, same total & centroid as a dot.
 *   6. WRAP: a stroke crossing x=0 floored-mods each sample individually —
 *      the tail reappears at the right screen edge, nothing in between.
 *   7. BENCH (informational): worst-case stroke load — 200k particles at
 *      |v| up to ~26 px/frame (S saturates at 24 samples) @1920x1080.
 */
import { setupGlobals } from "bun-webgpu";
import { SplatRenderer, SPLAT_FIXED_POINT, SplatStyle } from "../src/render/webgpu/splat";

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
// deterministic RNG — failures must reproduce
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

const FIXED = SPLAT_FIXED_POINT;
const f32 = Math.fround;

// COPY_SRC: the stroke path snapshots vel -> prevVel with a buffer copy (the
// live advect buffers carry COPY_SRC for the same reason).
function mkBuf(data: Float32Array): any {
  const b = device.createBuffer({
    size: data.byteLength,
    usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
  });
  device.queue.writeBuffer(b, 0, data);
  return b;
}

async function readAcc(rr: SplatRenderer, w: number, h: number): Promise<Uint32Array> {
  const size = w * h * 3 * 4;
  const staging = device.createBuffer({ size, usage: USAGE.MAP_READ | USAGE.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(rr.accumBuffer!, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(1 /* GPUMapMode.READ */);
  const data = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

// points.ts / splat.ts speed-palette mirror (t = clamped speed fraction)
const speedColor = (vx: number, vy: number, maxSpeed = 4): number[] => {
  const t = Math.min(1, Math.max(0, Math.hypot(vx, vy) / maxSpeed));
  const a = [0.25, 0.55, 1.0];
  const b = [1.0, 0.55, 0.2];
  return a.map((av, i) => av + (b[i] - av) * t);
};

// --------------------------------------------------------------------------
// mirror of STROKE_WGSL's sampling math (dpr=1). Returns the predicted stamp
// positions (torus-wrapped), taper weights (Σw = 1) and tapered radii —
// the geometric contract the shader must honour. float64 is fine here: the
// test inputs are chosen so S and the sample positions are exactly
// representable in f32 as well (asserted tolerances cover per-tap rounding).
// --------------------------------------------------------------------------
interface Stamp { x: number; y: number; w: number; ri: number }
function strokeSamples(
  pos: [number, number], v: [number, number], vp: [number, number],
  o: { r: number; T: number; curl: boolean; W: number; H: number; n?: number }
): Stamp[] {
  let ax = o.curl ? v[0] - vp[0] : 0;
  let ay = o.curl ? v[1] - vp[1] : 0;
  const vLen = Math.hypot(v[0], v[1]);
  const aLen = Math.hypot(ax, ay);
  // glitch guard mirror: |a| <= 1.5|v| (zeroes a when v = 0)
  const aScale = Math.min(1, (1.5 * vLen) / Math.max(aLen, 1e-6));
  ax *= aScale;
  ay *= aScale;
  const pixLen = vLen * o.T + 0.5 * (aLen * aScale) * o.T * o.T;
  // count-aware sample cap mirror (record(): tap budget / (n * taps-per-
  // stamp)). The geometry fixtures use a handful of particles, so maxS = 24
  // and the cap never engages there; it exists so the mirror stays faithful
  // if a fixture ever uses a big n.
  const maxS = Math.max(2, Math.min(24, Math.floor(24e6 / (Math.max(1, o.n ?? 1) * 10))));
  const S = Math.max(2, Math.min(maxS, Math.ceil(f32(pixLen / (o.r * 1.5)))));
  // continuity clamp mirror: capped strokes shorten instead of beading.
  const usedFrac = Math.min(1, (S * o.r * 1.5) / Math.max(pixLen, 1e-6));
  const wNorm = 1 / (0.65 * S);
  const out: Stamp[] = [];
  for (let i = 0; i < S; i++) {
    const ti = i / (S - 1);
    const tau = ti * o.T * usedFrac;
    let x = pos[0] - tau * v[0] + 0.5 * tau * tau * ax;
    let y = pos[1] - tau * v[1] + 0.5 * tau * tau * ay;
    x -= Math.floor(x / o.W) * o.W; // floored mod, matches the WGSL wrap
    y -= Math.floor(y / o.H) * o.H;
    out.push({ x, y, w: (1 - 0.7 * ti) * wNorm, ri: Math.max(o.r * (1 + (0.55 - 1) * ti), 0.75) });
  }
  return out;
}

// channel sum / centroid over the accumulator
const chanSum = (acc: Uint32Array, ch: number): number => {
  let s = 0;
  for (let i = ch; i < acc.length; i += 3) s += acc[i];
  return s;
};
function centroid(acc: Uint32Array, W: number, ch: number): [number, number] {
  let sw = 0, sx = 0, sy = 0;
  for (let i = ch; i < acc.length; i += 3) {
    const c = acc[i];
    if (c === 0) continue;
    const t = (i - ch) / 3;
    sw += c;
    sx += c * (t % W);
    sy += c * Math.floor(t / W);
  }
  return [sx / sw, sy / sw];
}
// sum of a channel inside an inclusive box
function boxSum(acc: Uint32Array, W: number, ch: number,
  x0: number, x1: number, y0: number, y1: number): number {
  let s = 0;
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) s += acc[(y * W + x) * 3 + ch];
  return s;
}

// --------------------------------------------------------------------------
// shared renderer (256x128, dpr 1) + render helpers
// --------------------------------------------------------------------------
const W = 256, H = 128;

device.pushErrorScope("validation");
const r = new SplatRenderer(null, {
  background: [10, 20, 30],
  maxSpeed: 4,
  classes: 0,
  decay: 1, // fused end-of-frame decay: 1 leaves the raw deposit readable
  exposure: 1,
  radius: 1.25,
  device,
});
check(r.style === "dot", `style defaults to "dot" (got "${r.style}")`);

/**
 * Render one measured frame of `style` from a CLEAN accumulator and return
 * the raw counts. `primeVel` (optional) is written to the vel buffer for one
 * throwaway stroke render first — the renderer's own end-of-pass
 * vel->prevVel copy then makes it the v_old the measured frame sees.
 */
async function renderStyle(
  style: SplatStyle,
  pos: Float32Array,
  vel: Float32Array,
  primeVel?: Float32Array
): Promise<Uint32Array> {
  const n = pos.length / 2;
  const posB = mkBuf(pos);
  const velB = mkBuf(primeVel ?? vel);
  r.style = style;
  if (primeVel) {
    r.render(posB, velB, n, W, H); // populates prevVel = primeVel
    device.queue.writeBuffer(velB, 0, vel);
  }
  r.clearAccum();
  r.render(posB, velB, n, W, H);
  const acc = await readAcc(r, W, H);
  posB.destroy();
  velB.destroy();
  return acc;
}

// --------------------------------------------------------------------------
// 1: REGRESSION — dot output must be BIT-EXACT vs the pre-stroke renderer.
// Baseline captured from the pre-change splat.ts (scratchpad capture,
// 2026-07-10, apple metal-3): 300 mulberry32(1234) particles, classes=3,
// radius 1.25, decay 1 -> accumulator SUM / FNV-1a below. u32 atomicAdds are
// order-independent, so the bytes are deterministic across runs; a changed
// hash means the shipped dot look changed.
// ADAPTER-SPECIFIC BASELINE (accepted, review finding): the counts flow
// through WGSL cos()/sqrt()/division, which the spec only bounds within ULPs
// — this hash is a contract for THIS project's verify box (apple metal-3,
// where all kernel gating runs). On a different adapter a 1-count texel
// wobble can fail this check without any real regression: recapture the
// baseline there rather than deleting the guard.
// --------------------------------------------------------------------------
{
  const BASE_SUM = 1813704;
  const BASE_FNV = 0x78c0d9df;
  const N = 300;
  const rnd = mulberry32(1234);
  const pos = new Float32Array(2 * N);
  const vel = new Float32Array(2 * N);
  for (let i = 0; i < N; i++) {
    pos[2 * i] = rnd() * W;
    pos[2 * i + 1] = rnd() * H;
    vel[2 * i] = (rnd() - 0.5) * 8;
    vel[2 * i + 1] = (rnd() - 0.5) * 8;
  }
  const rd = new SplatRenderer(null, {
    background: [10, 20, 30], maxSpeed: 4, classes: 3, decay: 1,
    exposure: 1, radius: 1.25, device,
  }); // no `style` option: exercises the default-construction dot path
  const posB = mkBuf(pos), velB = mkBuf(vel);
  rd.render(posB, velB, N, W, H);
  {
    const err = await device.popErrorScope();
    if (err) {
      failures++;
      console.log(`FAIL  validation error during construct/render: ${err.message}`);
    }
  }
  const acc = await readAcc(rd, W, H);
  let sum = 0;
  for (let i = 0; i < acc.length; i++) sum += acc[i];
  const bytes = new Uint8Array(acc.buffer);
  let fnv = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    fnv ^= bytes[i];
    fnv = Math.imul(fnv, 0x01000193) >>> 0;
  }
  check(
    sum === BASE_SUM && fnv === BASE_FNV,
    `dot regression: fixture accumulator bit-exact vs pre-stroke renderer ` +
      `(sum ${sum}==${BASE_SUM}, fnv 0x${fnv.toString(16)}==0x${BASE_FNV.toString(16)})`
  );
  rd.destroy();
  posB.destroy();
  velB.destroy();
}

// --------------------------------------------------------------------------
// 2: ENERGY — same deterministic particle set, total accumulated energy is
// style-invariant (±1%) and matches Σ 4096·(r+g+b) analytically. Particles
// keep a 30px margin so no stroke clips a screen edge (edge taps drop, wrap
// is tested separately) — max reach = |v|·T + ½|a|T² + r ≈ 5.7·3 + ½·0.9·9
// + 1.25 ≈ 22px < 30.
// --------------------------------------------------------------------------
{
  const N = 200, T = 3;
  r.strokeLen = T;
  const rnd = mulberry32(99);
  const pos = new Float32Array(2 * N);
  const vel = new Float32Array(2 * N);
  const vprev = new Float32Array(2 * N);
  for (let i = 0; i < N; i++) {
    pos[2 * i] = 30 + rnd() * (W - 60);
    pos[2 * i + 1] = 30 + rnd() * (H - 60);
    vel[2 * i] = (rnd() - 0.5) * 8;
    vel[2 * i + 1] = (rnd() - 0.5) * 8;
    // prevVel near vel: modest real accelerations (well under the clamp)
    vprev[2 * i] = vel[2 * i] + (rnd() - 0.5) * 0.9;
    vprev[2 * i + 1] = vel[2 * i + 1] + (rnd() - 0.5) * 0.9;
  }
  let expected = 0;
  for (let i = 0; i < N; i++) {
    const c = speedColor(vel[2 * i], vel[2 * i + 1]);
    expected += FIXED * (c[0] + c[1] + c[2]);
  }
  const total = (acc: Uint32Array) => chanSum(acc, 0) + chanSum(acc, 1) + chanSum(acc, 2);
  const dotSum = total(await renderStyle("dot", pos, vel));
  const velSum = total(await renderStyle("vel", pos, vel));
  const curlSum = total(await renderStyle("curl", pos, vel, vprev));
  const near1pc = (a: number, b: number) => Math.abs(a - b) <= 0.01 * Math.max(a, b);
  check(
    near1pc(dotSum, velSum) && near1pc(dotSum, curlSum) && near1pc(velSum, curlSum),
    `energy style-invariant ±1%: dot ${dotSum}  vel ${velSum}  curl ${curlSum}`
  );
  check(
    near1pc(dotSum, expected) && near1pc(curlSum, expected),
    `energy matches analytic Σ4096·(r+g+b) = ${expected.toFixed(0)} ±1%`
  );
}

// --------------------------------------------------------------------------
// 3: GEOMETRY (vel) — pos (150,64), v (6,0), T=3 (maxSpeed 4 -> t=1 -> col
// (1.0,0.55,0.2): red channel is exactly 1.0). Mirror: pixLen = 18,
// S = ceil(18/1.875) = 10, samples at x = 150-2i (integers!), y = 64.
// --------------------------------------------------------------------------
{
  const pos: [number, number] = [150, 64];
  const v: [number, number] = [6, 0];
  r.strokeLen = 3;
  const stamps = strokeSamples(pos, v, [0, 0], { r: 1.25, T: 3, curl: false, W, H });
  check(stamps.length === 10, `vel mirror: S == 10 (got ${stamps.length})`);

  const acc = await renderStyle("vel", new Float32Array(pos), new Float32Array(v));
  const red = (x: number, y: number) => acc[(y * W + x) * 3 + 0];

  check(
    Math.abs(chanSum(acc, 0) - FIXED) <= FIXED * 0.01,
    `vel: Σ red counts ${chanSum(acc, 0)} ≈ ${FIXED} ±1% (stroke conserves energy)`
  );
  check(
    stamps.every((s) => red(Math.round(s.x), Math.round(s.y)) > 0),
    `vel: all ${stamps.length} predicted sample texels x=150..132 are lit`
  );
  // confinement: stamps reach at most ri (≤1.25) from the segment
  const inside = boxSum(acc, W, 0, 129, 153, 61, 67);
  check(
    inside === chanSum(acc, 0),
    `vel: ALL energy inside the stroke box [129,153]x[61,67] (${inside} == ${chanSum(acc, 0)})`
  );
  // taper: head (t=0, w max, r max) outweighs tail (t=1, w 0.3x, r 0.55x)
  const head = boxSum(acc, W, 0, 148, 152, 62, 66);
  const tail = boxSum(acc, W, 0, 130, 134, 62, 66);
  check(
    head > 2 * tail && tail > 0,
    `vel: taper — head window ${head} > 2x tail window ${tail} > 0`
  );
  // centroid matches the mirror (integer sample positions -> symmetric
  // stamps -> exact expectation; per-tap ±0.5-count rounding => 0.2px tol)
  const [cx, cy] = centroid(acc, W, 0);
  const ex = stamps.reduce((s, t) => s + t.w * t.x, 0);
  check(
    Math.abs(cx - ex) < 0.2 && Math.abs(cy - 64) < 0.1,
    `vel: centroid (${cx.toFixed(3)}, ${cy.toFixed(3)}) == mirror (${ex.toFixed(3)}, 64)`
  );

  // dot control: the same particle under style=dot stamps ONLY at pos
  const accD = await renderStyle("dot", new Float32Array(pos), new Float32Array(v));
  const [dx, dy] = centroid(accD, W, 0);
  check(
    boxSum(accD, W, 0, 148, 152, 62, 66) === chanSum(accD, 0) &&
      Math.abs(dx - 150) < 0.05 && Math.abs(dy - 64) < 0.05,
    `dot control: all energy at pos (centroid ${dx.toFixed(3)}, ${dy.toFixed(3)})`
  );
}

// --------------------------------------------------------------------------
// 4: GEOMETRY (curl) — same particle, vPrev (6,-2) => a = (0,2). Samples
// follow p(τ) = pos - τv + ½τ²a: the trail bends to y = 64 + τ² (toward +a —
// the task formula's backward Taylor: an accelerating particle's PAST
// positions sit on the +a side of the straight back-extrapolation). Mirror:
// pixLen = 18 + ½·2·9 = 27, S = ceil(27/1.875) = 15.
// --------------------------------------------------------------------------
{
  const pos: [number, number] = [150, 64];
  const v: [number, number] = [6, 0];
  const vp: [number, number] = [6, -2];
  const stamps = strokeSamples(pos, v, vp, { r: 1.25, T: 3, curl: true, W, H });
  check(stamps.length === 15, `curl mirror: S == 15 (got ${stamps.length})`);

  const acc = await renderStyle(
    "curl", new Float32Array(pos), new Float32Array(v), new Float32Array(vp)
  );
  const red = (x: number, y: number) => acc[(y * W + x) * 3 + 0];
  check(
    Math.abs(chanSum(acc, 0) - FIXED) <= FIXED * 0.01,
    `curl: Σ red counts ${chanSum(acc, 0)} ≈ ${FIXED} ±1%`
  );
  check(
    stamps.every((s) => red(Math.round(s.x), Math.round(s.y)) > 0),
    `curl: all ${stamps.length} predicted arc texels are lit (tail bends to (132,73))`
  );
  // the arc's tail (132, 73) is lit; the STRAIGHT line's tail (132, 64) is
  // dark — 9px off the curve, far beyond any stamp radius
  check(
    boxSum(acc, W, 0, 131, 133, 63, 65) === 0,
    `curl: straight-line tail (132,64)±1 is dark (trajectory bent away)`
  );
  // centroid displacement off the straight stroke == mirror Σw·½τ²·a
  const accV = await renderStyle("vel", new Float32Array(pos), new Float32Array(v));
  const [, cyCurl] = centroid(acc, W, 0);
  const [, cyVel] = centroid(accV, W, 0);
  const velStamps = strokeSamples(pos, v, [0, 0], { r: 1.25, T: 3, curl: false, W, H });
  const eyCurl = stamps.reduce((s, t) => s + t.w * t.y, 0);
  const eyVel = velStamps.reduce((s, t) => s + t.w * t.y, 0);
  const bendGot = cyCurl - cyVel;
  const bendWant = eyCurl - eyVel; // > 0: toward +a
  check(
    bendGot > 0.5 && Math.abs(bendGot - bendWant) < 0.5,
    `curl: centroid bends +${bendGot.toFixed(3)}px toward +a ` +
      `(mirror predicts +${bendWant.toFixed(3)}px)`
  );
}

// --------------------------------------------------------------------------
// 5: RESET GUARD — v = 0 with a big stale prevVel (8,0): exactly the fused
// random-reset state (advect writes v=0, prevVel is one frame stale).
// Unguarded, a = v - vPrev = (-8,0) would smear ½τ²·8 = 36px of arc across
// [64,100]; the |a| ≤ 1.5|v| clamp zeroes a, collapsing the stroke onto pos.
// --------------------------------------------------------------------------
{
  const pos: [number, number] = [100, 64];
  const acc = await renderStyle(
    "curl",
    new Float32Array(pos),
    new Float32Array([0, 0]),
    new Float32Array([8, 0])
  );
  // v=0 -> blue palette end: blue channel is exactly 1.0
  const accD = await renderStyle("dot", new Float32Array(pos), new Float32Array([0, 0]));
  const sumC = chanSum(acc, 2);
  const sumD = chanSum(accD, 2);
  check(
    boxSum(acc, W, 2, 60, 96, 58, 70) === 0,
    `reset guard: the would-be arc region x∈[64,96] holds ZERO energy`
  );
  check(
    boxSum(acc, W, 2, 97, 103, 61, 67) === sumC,
    `reset guard: all curl energy within 3px of pos (clamp killed the arc)`
  );
  const [cx, cy] = centroid(acc, W, 2);
  const [dx, dy] = centroid(accD, W, 2);
  check(
    Math.abs(sumC - sumD) <= 0.01 * sumD &&
      Math.abs(cx - dx) < 0.05 && Math.abs(cy - dy) < 0.05,
    `reset guard: curl ≡ dot at pos — energy ${sumC}≈${sumD}, ` +
      `centroid (${cx.toFixed(3)},${cy.toFixed(3)})≈(${dx.toFixed(3)},${dy.toFixed(3)})`
  );
}

// --------------------------------------------------------------------------
// 6: WRAP — pos (2,64), v (6,0), T=3: samples at x = 2-2i wrap through x=0
// to the right edge (2, 0, 254, 252, …, 240). Each SAMPLE is floored-mod
// independently, so the stroke shows on both sides with nothing in between.
// --------------------------------------------------------------------------
{
  const pos: [number, number] = [2, 64];
  const v: [number, number] = [6, 0];
  const stamps = strokeSamples(pos, v, [0, 0], { r: 1.25, T: 3, curl: false, W, H });
  const acc = await renderStyle("vel", new Float32Array(pos), new Float32Array(v));
  const left = boxSum(acc, W, 0, 0, 4, 61, 67);
  const right = boxSum(acc, W, 0, 238, 255, 61, 67);
  const middle = boxSum(acc, W, 0, 10, 230, 0, H - 1);
  check(
    stamps.some((s) => s.x > 200) && stamps.some((s) => s.x < 5),
    `wrap mirror: samples land on BOTH sides (xs ${stamps.map((s) => s.x.toFixed(0)).join(",")})`
  );
  check(
    left > 0 && right > 0 && middle === 0,
    `wrap: head lit at left edge (${left}), tail wrapped to right edge (${right}), ` +
      `middle of screen dark (${middle})`
  );
  check(
    stamps.every((s) => acc[(Math.round(s.y) * W + Math.round(s.x)) * 3] > 0),
    `wrap: every wrapped sample texel is lit`
  );
}

// --------------------------------------------------------------------------
// 7: BENCH (informational) — worst-case stroke load: 200k particles with
// |v| up to ~26 px/frame @1920x1080, curl style, trails on. The count-aware
// tap-budget cap gives maxS = floor(24e6/(200k*10)) = 12 here (the review's
// perf fix); strokes shorten via the continuity clamp instead of beading.
// Target: ≤16.7ms (60fps).
// --------------------------------------------------------------------------
{
  const BW = 1920, BH = 1080, N = 200_000;
  const rnd = mulberry32(7);
  const posN = new Float32Array(2 * N);
  const velN = new Float32Array(2 * N);
  const vpN = new Float32Array(2 * N);
  for (let i = 0; i < N; i++) {
    posN[2 * i] = rnd() * BW;
    posN[2 * i + 1] = rnd() * BH;
    velN[2 * i] = (rnd() - 0.5) * 52; // components ±26
    velN[2 * i + 1] = (rnd() - 0.5) * 52;
    vpN[2 * i] = velN[2 * i] + (rnd() - 0.5) * 4;
    vpN[2 * i + 1] = velN[2 * i + 1] + (rnd() - 0.5) * 4;
  }
  const posB = mkBuf(posN);
  const velB = mkBuf(vpN);
  r.style = "curl";
  r.strokeLen = 3;
  r.decay = 0.9;
  r.classes = 3;
  r.radius = 1.25;
  r.render(posB, velB, N, BW, BH); // prime prevVel + realloc accumulator
  device.queue.writeBuffer(velB, 0, velN);

  const fence = async (): Promise<void> => {
    const staging = device.createBuffer({ size: 256, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(r.accumBuffer!, 0, staging, 0, 256);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    staging.unmap();
    staging.destroy();
  };

  const WARM = 20, TIMED = 100;
  for (let s = 0; s < WARM; s++) r.render(posB, velB, N, BW, BH);
  await fence();
  const t0 = performance.now();
  for (let s = 0; s < TIMED; s++) r.render(posB, velB, N, BW, BH);
  await fence();
  const ms = (performance.now() - t0) / TIMED;
  console.log(
    `\nBENCH  curl stroke @ ${N.toLocaleString()} particles, ${BW}x${BH}, ` +
      `|v|≈26, budget cap maxS=${Math.max(2, Math.min(24, Math.floor(24e6 / (N * 10))))}: ` +
      `${ms.toFixed(3)} ms/frame (${(1000 / ms).toFixed(0)} fps) ` +
      `— target ≤16.7 ms for 60fps; GPU is shared, so contention inflates this`
  );
  posB.destroy();
  velB.destroy();
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
