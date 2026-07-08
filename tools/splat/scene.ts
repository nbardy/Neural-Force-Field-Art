/**
 * Shared helpers for the splat raster tests (tools/splat/*): deterministic
 * pcg-seeded RNG (no Math.random — failures must reproduce), SoA param packing,
 * and image writers (P6 PPM for the spec's before/after deliverable, plus a
 * minimal stored-zlib PNG so the forward can be eyeballed with the Read tool).
 */
import { writeFileSync } from "fs";
import { PARAM_STRIDE } from "../../src/splat/raster_wgsl";

// --------------------------------------------------------------------------
// deterministic RNG: pcg32-ish (seeded). Gaussian via Box-Muller.
// --------------------------------------------------------------------------
export function makeRng(seed: number): { next: () => number; normal: () => number; range: (a: number, b: number) => number } {
  let state = (seed >>> 0) || 1;
  const next = (): number => {
    // pcg-xsh-rr style step (same family as the WGSL pcg used elsewhere)
    state = (Math.imul(state, 747796405) + 2891336453) >>> 0;
    let t = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
    t = ((t >>> 22) ^ t) >>> 0;
    return t / 4294967296;
  };
  const normal = (): number => {
    let u = 0;
    let v = 0;
    while (u === 0) u = next();
    while (v === 0) v = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const range = (a: number, b: number): number => a + (b - a) * next();
  return { next, normal, range };
}

// --------------------------------------------------------------------------
// SoA param packing: [mean 2G][logScale 2G][theta G][colorRaw 3G][opacityRaw G]
// --------------------------------------------------------------------------
export interface SplatArrays {
  mean: Float32Array; // 2G
  logScale: Float32Array; // 2G
  theta: Float32Array; // G
  colorRaw: Float32Array; // 3G
  opacityRaw: Float32Array; // G
}

export function packParams(G: number, s: SplatArrays): Float32Array {
  const p = new Float32Array(G * PARAM_STRIDE);
  p.set(s.mean, 0);
  p.set(s.logScale, 2 * G);
  p.set(s.theta, 4 * G);
  p.set(s.colorRaw, 5 * G);
  p.set(s.opacityRaw, 8 * G);
  return p;
}

/** Index of a raw param inside the packed SoA buffer (mirrors seg() in codegen). */
export function paramIndex(G: number, group: keyof SplatArrays, splat: number, comp = 0): number {
  switch (group) {
    case "mean":
      return 0 + splat * 2 + comp;
    case "logScale":
      return 2 * G + splat * 2 + comp;
    case "theta":
      return 4 * G + splat;
    case "colorRaw":
      return 5 * G + splat * 3 + comp;
    case "opacityRaw":
      return 8 * G + splat;
  }
}

// --------------------------------------------------------------------------
// image writers.  Input: planar NCHW f32 [3][H][W] in ~[0,1] (raster output).
// --------------------------------------------------------------------------
function toRGBBytes(planar: Float32Array, W: number, H: number): Uint8Array {
  const HW = H * W;
  const out = new Uint8Array(HW * 3);
  for (let i = 0; i < HW; i++) {
    for (let c = 0; c < 3; c++) {
      const v = planar[c * HW + i];
      out[i * 3 + c] = Math.max(0, Math.min(255, Math.round(v * 255)));
    }
  }
  return out;
}

/** P6 (binary) PPM — the spec's before/after eyeball deliverable. */
export function writePPM(path: string, planar: Float32Array, W: number, H: number): void {
  const rgb = toRGBBytes(planar, W, H);
  const header = Buffer.from(`P6\n${W} ${H}\n255\n`, "ascii");
  writeFileSync(path, Buffer.concat([header, Buffer.from(rgb)]));
}

// ---- minimal PNG (stored/uncompressed zlib) so the Read tool can render it --
function crc32(buf: Uint8Array): number {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return (~c) >>> 0;
}
function adler32(buf: Uint8Array): number {
  let a = 1;
  let b = 0;
  for (let i = 0; i < buf.length; i++) {
    a = (a + buf[i]) % 65521;
    b = (b + a) % 65521;
  }
  return ((b << 16) | a) >>> 0;
}
function chunk(type: string, data: Uint8Array): Uint8Array {
  const t = Buffer.from(type, "ascii");
  const body = Buffer.concat([t, Buffer.from(data)]); // type + data (CRC covers both)
  const out = new Uint8Array(4 + body.length + 4); // len(4) + body + crc(4)
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  out.set(body, 4);
  dv.setUint32(4 + body.length, crc32(body));
  return out;
}
function zlibStored(raw: Uint8Array): Uint8Array {
  const parts: Uint8Array[] = [Uint8Array.from([0x78, 0x01])];
  let off = 0;
  while (off < raw.length) {
    const len = Math.min(65535, raw.length - off);
    const last = off + len >= raw.length ? 1 : 0;
    const hdr = new Uint8Array(5);
    hdr[0] = last;
    hdr[1] = len & 0xff;
    hdr[2] = (len >> 8) & 0xff;
    hdr[3] = ~len & 0xff;
    hdr[4] = (~len >> 8) & 0xff;
    parts.push(hdr, raw.subarray(off, off + len));
    off += len;
  }
  const ad = new Uint8Array(4);
  new DataView(ad.buffer).setUint32(0, adler32(raw));
  parts.push(ad);
  return Buffer.concat(parts);
}

export function writePNG(path: string, planar: Float32Array, W: number, H: number): void {
  const rgb = toRGBBytes(planar, W, H);
  const raw = new Uint8Array(H * (1 + W * 3)); // filter byte 0 per scanline
  for (let y = 0; y < H; y++) {
    raw[y * (1 + W * 3)] = 0;
    raw.set(rgb.subarray(y * W * 3, (y + 1) * W * 3), y * (1 + W * 3) + 1);
  }
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, W);
  dv.setUint32(4, H);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type RGB
  const sig = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]);
  writeFileSync(
    path,
    Buffer.concat([sig, chunk("IHDR", ihdr), chunk("IDAT", zlibStored(raw)), chunk("IEND", new Uint8Array(0))])
  );
}

/** L2 (sum of squared error) between two planar images. */
export function l2(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

// --------------------------------------------------------------------------
// float64 CPU reference of the whole forward + backward + reparam chain
// (an independent mirror of raster_wgsl.ts using the SAME hard gates). Because
// it composites every splat in ascending index order — the exact order the GPU
// recovers after its per-tile ascending sort, and the binning never drops a
// pixel where a splat is visible — its result matches the GPU to f32 precision.
// This is the authoritative gradient check (no finite-difference gate noise).
// --------------------------------------------------------------------------
const THR = {
  SCALE_MIN: 0.3,
  SCALE_MAX: 64.0,
  ALPHA: 1.0 / 255.0,
  MAX_ALPHA: 0.99,
  TRANS: 1e-4,
  EPS: 1e-8,
};
const clampd = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
const sig = (x: number) => 1 / (1 + Math.exp(-x));

/** Returns the raw-param gradient (SoA, length G*9) for the scene + dL/dpixels. */
export function cpuGradRef(G: number, H: number, W: number, s: SplatArrays, go: Float32Array, bg: [number, number, number]): Float64Array {
  const mx: number[] = [], my: number[] = [], a: number[] = [], b: number[] = [], c: number[] = [];
  const cr: number[] = [], cg: number[] = [], cb: number[] = [], op: number[] = [];
  for (let g = 0; g < G; g++) {
    const sx = clampd(Math.exp(s.logScale[2 * g]), THR.SCALE_MIN, THR.SCALE_MAX);
    const sy = clampd(Math.exp(s.logScale[2 * g + 1]), THR.SCALE_MIN, THR.SCALE_MAX);
    const ix = 1 / (sx * sx), iy = 1 / (sy * sy);
    const cs = Math.cos(s.theta[g]), sn = Math.sin(s.theta[g]);
    mx[g] = s.mean[2 * g]; my[g] = s.mean[2 * g + 1];
    a[g] = cs * cs * ix + sn * sn * iy;
    b[g] = cs * sn * (ix - iy);
    c[g] = sn * sn * ix + cs * cs * iy;
    cr[g] = sig(s.colorRaw[3 * g]); cg[g] = sig(s.colorRaw[3 * g + 1]); cb[g] = sig(s.colorRaw[3 * g + 2]);
    op[g] = sig(s.opacityRaw[g]);
  }
  const gMx = new Float64Array(G), gMy = new Float64Array(G);
  const gA = new Float64Array(G), gB = new Float64Array(G), gC = new Float64Array(G);
  const gCr = new Float64Array(G), gCg = new Float64Array(G), gCb = new Float64Array(G), gOp = new Float64Array(G);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pxc = x + 0.5, pyc = y + 0.5;
      const gor = go[y * W + x], gog = go[H * W + y * W + x], gob = go[2 * H * W + y * W + x];
      const vis: number[] = [];
      let T = 1.0;
      for (let g = 0; g < G; g++) {
        const dx = pxc - mx[g], dy = pyc - my[g];
        const power = -0.5 * (a[g] * dx * dx + 2 * b[g] * dx * dy + c[g] * dy * dy);
        if (power > 0) continue;
        const alpha = Math.min(THR.MAX_ALPHA, op[g] * Math.exp(power));
        if (alpha < THR.ALPHA) continue;
        vis.push(g);
        T *= 1 - alpha;
        if (T < THR.TRANS) break;
      }
      let Tcur = T;
      let gT = gor * bg[0] + gog * bg[1] + gob * bg[2];
      for (let k = vis.length - 1; k >= 0; k--) {
        const g = vis[k];
        const dx = pxc - mx[g], dy = pyc - my[g];
        const power = -0.5 * (a[g] * dx * dx + 2 * b[g] * dx * dy + c[g] * dy * dy);
        const raw = op[g] * Math.exp(power);
        const alpha = Math.min(THR.MAX_ALPHA, raw);
        const Tprev = Tcur / Math.max(1 - alpha, THR.EPS);
        const dotgc = gor * cr[g] + gog * cg[g] + gob * cb[g];
        const gAlpha = Tprev * (dotgc - gT);
        gCr[g] += gor * Tprev * alpha; gCg[g] += gog * Tprev * alpha; gCb[g] += gob * Tprev * alpha;
        const gRaw = (raw < THR.MAX_ALPHA ? 1 : 0) * gAlpha;
        const gPower = gRaw * raw;
        gA[g] += gPower * -0.5 * dx * dx;
        gB[g] += gPower * -1.0 * dx * dy;
        gC[g] += gPower * -0.5 * dy * dy;
        gMx[g] += gPower * (a[g] * dx + b[g] * dy);   // -(-(a dx + b dy)) = +(a dx + b dy)
        gMy[g] += gPower * (b[g] * dx + c[g] * dy);
        gOp[g] += gRaw * (raw / Math.max(op[g], THR.EPS));
        gT = alpha * dotgc + (1 - alpha) * gT;
        Tcur = Tprev;
      }
    }
  }
  const out = new Float64Array(G * 9);
  for (let g = 0; g < G; g++) {
    const ex = Math.exp(s.logScale[2 * g]), ey = Math.exp(s.logScale[2 * g + 1]);
    const sx = clampd(ex, THR.SCALE_MIN, THR.SCALE_MAX), sy = clampd(ey, THR.SCALE_MIN, THR.SCALE_MAX);
    const gateX = ex > THR.SCALE_MIN && ex < THR.SCALE_MAX ? 1 : 0;
    const gateY = ey > THR.SCALE_MIN && ey < THR.SCALE_MAX ? 1 : 0;
    const ix = 1 / (sx * sx), iy = 1 / (sy * sy);
    const cs = Math.cos(s.theta[g]), sn = Math.sin(s.theta[g]);
    const gix = gA[g] * cs * cs + gB[g] * cs * sn + gC[g] * sn * sn;
    const giy = gA[g] * sn * sn - gB[g] * cs * sn + gC[g] * cs * cs;
    out[g * 2] = gMx[g]; out[g * 2 + 1] = gMy[g];
    out[2 * G + g * 2] = gix * (-2 * ix) * gateX;
    out[2 * G + g * 2 + 1] = giy * (-2 * iy) * gateY;
    out[4 * G + g] = (ix - iy) * ((cs * cs - sn * sn) * gB[g] + 2 * cs * sn * (gC[g] - gA[g]));
    out[5 * G + g * 3] = gCr[g] * cr[g] * (1 - cr[g]);
    out[5 * G + g * 3 + 1] = gCg[g] * cg[g] * (1 - cg[g]);
    out[5 * G + g * 3 + 2] = gCb[g] * cb[g] * (1 - cb[g]);
    out[8 * G + g] = gOp[g] * op[g] * (1 - op[g]);
  }
  return out;
}
