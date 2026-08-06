/**
 * pixel_disc_wgsl — PURE WGSL codegen for four pixel-space GAN kinds.
 *
 * Reverse-mode only (no JVP). Soft bilinear splat + conv→codebook critic;
 * generator reward chains through virtual positions into field weights via
 * extGrads (same seam as the relational adversary).
 *
 * Kinds (CPU oracle src/core/gan/pixel_disc.ts, spec docs/PIXEL_DISC.md):
 *   vec-field   per-cell 2-vector vs F(cell center)
 *   next-frame  D0 → predict D1
 *   real-fake   GAP+MLP logit, real vs random-position fake
 *   inpaint     masked completion on random block
 *
 * densPack layout (4 × nCell f32, nCell = G²):
 *   [0, n)     dens   — current working density
 *   [n, 2n)    dDens  — generator ∂L/∂D (VJP output)
 *   [2n, 3n)   auxA   — D0 / forceX / fake dens (kind-dependent)
 *   [3n, 4n)   auxB   — forceY / inpaint mask (0|1 f32)
 *
 * Trainer must allocate densPack with densPackFloats(G) floats (4×G²).
 * See PIXEL_DISC_KIND_PASSES for per-kind pass order.
 */

import { type FieldLayout, type HeadSpec } from "./advect_wgsl";
import {
  emitEncode,
  emitFwdStore,
  emitBwdStore,
  trainScratchLayout,
  COMMON,
} from "./train_wgsl";
import {
  PIXEL_DISC_SOFT_EPS,
  PIXEL_DISC_DEFAULTS,
  pixelDiscWeightCount,
  type PixelDiscDims,
  type PixelGanKind,
} from "../../core/gan/pixel_disc";

export const PIXEL_DISC_WG = 256;
export const PIXEL_DISC_MAX_BATCH = 512;
/** Fixed-point scale for atomic density accumulation (energy 1 ≡ SCALE). */
export const PIXEL_DISC_DENS_SCALE = 1e6;

const DENS_FLOOR = 1e-3;

export interface PixelDiscShaderOpts {
  dims: PixelDiscDims;
  batchCap: number;
  /** `blend` | 0 | 1 — mirrors adversary fieldLane. */
  fieldLane?: "blend" | 0 | 1;
}

/** Trainer pass orchestration notes (host owns encodeStep). */
export const PIXEL_DISC_KIND_PASSES: Record<PixelGanKind, string> = {
  "vec-field":
    "clearDens → sampleAndSplat → densToFloat → criticDisc → discAdam;" +
    " gen: clearDensGen → virtualSplat → densToFloat → criticGen → vjp → fieldGrad",
  "next-frame":
    "clearDens → sampleAndSplat → densToFloat → copyDensToAux → clearDensGen →" +
    " virtualSplat → densToFloat → criticDisc → discAdam;" +
    " gen: clearDensGen → virtualSplat → densToFloat → criticGen → vjp → fieldGrad",
  "real-fake":
    "clearDens → sampleAndSplat → densToFloat → clearAtomics → fakeSplat →" +
    " densToFloatFake → criticDisc → discAdam;" +
    " gen: clearDensGen → virtualSplat → criticGen → vjp → fieldGrad",
  inpaint:
    "clearDens → sampleAndSplat → densToFloat → criticDisc → discAdam;" +
    " gen: clearDensGen → virtualSplat → densToFloat → criticGen → vjp → fieldGrad",
};

function fl(x: number): string {
  if (!Number.isFinite(x)) throw new Error(`pixel_disc: non-finite ${x}`);
  let s = x.toString();
  if (!/[.eE]/.test(s)) s += ".0";
  return s;
}

export function resolvePixelDims(
  partial?: Partial<PixelDiscDims> & { kind?: PixelGanKind }
): PixelDiscDims {
  return {
    G: partial?.G ?? PIXEL_DISC_DEFAULTS.G,
    E: partial?.E ?? PIXEL_DISC_DEFAULTS.E,
    K: partial?.K ?? PIXEL_DISC_DEFAULTS.K,
    hidden: partial?.hidden ?? PIXEL_DISC_DEFAULTS.hidden,
    dt: partial?.dt ?? PIXEL_DISC_DEFAULTS.dt,
    kind: partial?.kind ?? "vec-field",
  };
}

export function validatePixelDiscFusion(field: FieldLayout): void {
  if (field.spec.kind !== "helmholtz" && field.spec.kind !== "agree-disagree") {
    throw new Error(
      `pixel_disc: needs two-head neural field (got ${field.spec.kind})`
    );
  }
  if (field.classes > 0) {
    throw new Error("pixel_disc: class-aware fields not supported yet");
  }
  if (field.encoding.kind === "hashgrid") {
    throw new Error("pixel_disc: hashgrid not supported yet");
  }
}

/** f32 slots in densPack (4 slices × G²). Trainer: densPack = mk(densPackFloats(G)*4). */
export function densPackFloats(G: number): number {
  return 4 * G * G;
}

/** Bytes for densPack f32 buffer (atomics densI32 allocated separately). */
export function pixelDensBytes(G: number): number {
  return densPackFloats(G) * 4;
}

export function pixelParticleScratchFloats(field: FieldLayout): number {
  const sl = trainScratchLayout(field, 1);
  return 8 + sl.encStore + sl.siteBlk;
}

export function pixelScratchBytes(
  field: FieldLayout,
  batchCap: number
): number {
  const sl = trainScratchLayout(field, 1);
  const partStride = pixelParticleScratchFloats(field);
  // particles + one-site critic field eval workspace (vec-field force grid)
  return (batchCap * partStride + sl.encStore + sl.siteBlk) * 4;
}

export type PixelWeightLayout =
  | {
      kind: "vec-field";
      convW: number;
      convB: number;
      code: number;
      headW: number;
      headB: number;
      total: number;
    }
  | {
      kind: "next-frame" | "inpaint";
      convW: number;
      convB: number;
      code: number;
      headW: number;
      headB: number;
      total: number;
    }
  | {
      kind: "real-fake";
      convW: number;
      convB: number;
      code: number;
      mlp0W: number;
      mlp0B: number;
      mlp1W: number;
      mlp1B: number;
      total: number;
    };

/** Packed critic weight offsets — matches packPixelDiscWeights / headFloats(kind). */
export function pixelWeightLayout(d: PixelDiscDims): PixelWeightLayout {
  let o = 0;
  const convW = o;
  o += d.E * 9;
  const convB = o;
  o += d.E;
  const code = o;
  o += d.K * d.E;
  switch (d.kind) {
    case "vec-field": {
      const headW = o;
      o += 2 * d.K;
      const headB = o;
      o += 2;
      return { kind: "vec-field", convW, convB, code, headW, headB, total: o };
    }
    case "next-frame":
    case "inpaint": {
      const headW = o;
      o += d.K;
      const headB = o;
      o += 1;
      return { kind: d.kind, convW, convB, code, headW, headB, total: o };
    }
    case "real-fake": {
      const mlp0W = o;
      o += d.hidden * d.K;
      const mlp0B = o;
      o += d.hidden;
      const mlp1W = o;
      o += d.hidden;
      const mlp1B = o;
      o += 1;
      return { kind: "real-fake", convW, convB, code, mlp0W, mlp0B, mlp1W, mlp1B, total: o };
    }
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

/** mulberry32 — matches CPU oracle (one sample, state = seed). */
function emitHashRng(): string {
  return /* wgsl */ `
fn mulberry32(seed : u32) -> f32 {
  var a = seed + 0x6d2b79f5u;
  var t = (a ^ (a >> 15u)) * (1u | a);
  t = t + (t ^ (t >> 7u)) * (61u | t);
  t = t ^ (t >> 14u);
  return f32(t) / 4294967296.0;
}
`;
}

/** Read density slice: off=0 dens, off=2 auxA. */
function emitBackboneForward(d: PixelDiscDims, densOffExpr: string): string {
  const { G, E, K } = d;
  const nCell = G * G;
  const wl = pixelWeightLayout(d);
  const convB = wl.convB;
  const convW = wl.convW;
  const code = wl.code;
  return /* wgsl */ `
  for (var e = 0u; e < ${E}u; e = e + 1u) {
    let b = critW[${convB}u + e];
    for (var y = 0u; y < ${G}u; y = y + 1u) {
      for (var x = 0u; x < ${G}u; x = x + 1u) {
        var s = b;
        for (var dy = 0u; dy < 3u; dy = dy + 1u) {
          let yy = u32(clamp(i32(y) + i32(dy) - 1, 0, ${G - 1}));
          for (var dx = 0u; dx < 3u; dx = dx + 1u) {
            let xx = u32(clamp(i32(x) + i32(dx) - 1, 0, ${G - 1}));
            let di = ${densOffExpr} + yy * ${G}u + xx;
            s = s + critW[${convW}u + e * 9u + dy * 3u + dx] * densPack[di];
          }
        }
        cFeat[e * ${nCell}u + y * ${G}u + x] = max(s, 0.0);
      }
    }
  }
  for (var cell = 0u; cell < ${nCell}u; cell = cell + 1u) {
    var maxL = -1e30;
    var logits : array<f32, ${K}>;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      var s = 0.0;
      for (var e = 0u; e < ${E}u; e = e + 1u) {
        s = s + cFeat[e * ${nCell}u + cell] * critW[${code}u + k * ${E}u + e];
      }
      logits[k] = s;
      maxL = max(maxL, s);
    }
    var sum = 0.0;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      let v = exp(logits[k] - maxL);
      cSoft[k * ${nCell}u + cell] = v;
      sum = sum + v;
    }
    let inv = 1.0 / sum;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      cSoft[k * ${nCell}u + cell] = cSoft[k * ${nCell}u + cell] * inv;
    }
  }
`;
}

/** VJP through backbone; dSoft[K*nCell] in, gW/gD out; densOff like forward. */
function emitBackboneBackward(
  d: PixelDiscDims,
  densOffExpr: string,
  nW: number
): string {
  const { G, E, K } = d;
  const nCell = G * G;
  const wl = pixelWeightLayout(d);
  const convB = wl.convB;
  const convW = wl.convW;
  const code = wl.code;
  return /* wgsl */ `
  var dFeat : array<f32, ${E * nCell}>;
  for (var i = 0u; i < ${E * nCell}u; i = i + 1u) { dFeat[i] = 0.0; }
  for (var cell = 0u; cell < ${nCell}u; cell = cell + 1u) {
    var dot = 0.0;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      dot = dot + cSoft[k * ${nCell}u + cell] * dSoft[k * ${nCell}u + cell];
    }
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      let dLogit = cSoft[k * ${nCell}u + cell] * (dSoft[k * ${nCell}u + cell] - dot);
      for (var e = 0u; e < ${E}u; e = e + 1u) {
        let f = cFeat[e * ${nCell}u + cell];
        gW[${code}u + k * ${E}u + e] = gW[${code}u + k * ${E}u + e] + dLogit * f;
        dFeat[e * ${nCell}u + cell] =
          dFeat[e * ${nCell}u + cell] + dLogit * critW[${code}u + k * ${E}u + e];
      }
    }
  }
  for (var e = 0u; e < ${E}u; e = e + 1u) {
    for (var y = 0u; y < ${G}u; y = y + 1u) {
      for (var x = 0u; x < ${G}u; x = x + 1u) {
        let idx = e * ${nCell}u + y * ${G}u + x;
        if (cFeat[idx] <= 0.0) { continue; }
        let gf = dFeat[idx];
        gW[${convB}u + e] = gW[${convB}u + e] + gf;
        for (var dy = 0u; dy < 3u; dy = dy + 1u) {
          let yy = u32(clamp(i32(y) + i32(dy) - 1, 0, ${G - 1}));
          for (var dx = 0u; dx < 3u; dx = dx + 1u) {
            let xx = u32(clamp(i32(x) + i32(dx) - 1, 0, ${G - 1}));
            let wi = ${convW}u + e * 9u + dy * 3u + dx;
            let di = ${densOffExpr} + yy * ${G}u + xx;
            gW[wi] = gW[wi] + gf * densPack[di];
            gD[yy * ${G}u + xx] = gD[yy * ${G}u + xx] + gf * critW[wi];
          }
        }
      }
    }
  }
`;
}

function emitGapMlpForward(d: PixelDiscDims): string {
  const { K, hidden } = d;
  const wl = pixelWeightLayout(d);
  if (wl.kind !== "real-fake") throw new Error("GAP MLP only for real-fake");
  return /* wgsl */ `
  for (var k = 0u; k < ${K}u; k = k + 1u) { cZ[k] = 0.0; }
  for (var cell = 0u; cell < N_CELL; cell = cell + 1u) {
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      cZ[k] = cZ[k] + cSoft[k * N_CELL + cell];
    }
  }
  for (var k = 0u; k < ${K}u; k = k + 1u) { cZ[k] = cZ[k] / f32(N_CELL); }
  for (var i = 0u; i < ${hidden}u; i = i + 1u) {
    var s = critW[${wl.mlp0B}u + i];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      s = s + critW[${wl.mlp0W}u + i * ${K}u + k] * cZ[k];
    }
    cH[i] = tanh(s);
  }
  logit = critW[${wl.mlp1B}u];
  for (var i = 0u; i < ${hidden}u; i = i + 1u) {
    logit = logit + critW[${wl.mlp1W}u + i] * cH[i];
  }
`;
}

function emitGapMlpBackward(d: PixelDiscDims, dLogitExpr: string): string {
  const { G, K, hidden } = d;
  const nCell = G * G;
  const wl = pixelWeightLayout(d);
  if (wl.kind !== "real-fake") throw new Error("GAP MLP only for real-fake");
  return /* wgsl */ `
  let dLogit = ${dLogitExpr};
  gW[${wl.mlp1B}u] = gW[${wl.mlp1B}u] + dLogit;
  var dh : array<f32, ${hidden}>;
  for (var i = 0u; i < ${hidden}u; i = i + 1u) {
    gW[${wl.mlp1W}u + i] = gW[${wl.mlp1W}u + i] + dLogit * cH[i];
    dh[i] = dLogit * critW[${wl.mlp1W}u + i];
  }
  var dz : array<f32, ${K}>;
  for (var k = 0u; k < ${K}u; k = k + 1u) { dz[k] = 0.0; }
  for (var i = 0u; i < ${hidden}u; i = i + 1u) {
    let pre = dh[i] * (1.0 - cH[i] * cH[i]);
    gW[${wl.mlp0B}u + i] = gW[${wl.mlp0B}u + i] + pre;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      gW[${wl.mlp0W}u + i * ${K}u + k] =
        gW[${wl.mlp0W}u + i * ${K}u + k] + pre * cZ[k];
      dz[k] = dz[k] + pre * critW[${wl.mlp0W}u + i * ${K}u + k];
    }
  }
  for (var k = 0u; k < ${K}u; k = k + 1u) {
    let gz = dz[k] / f32(N_CELL);
    for (var cell = 0u; cell < ${nCell}u; cell = cell + 1u) {
      dSoft[k * ${nCell}u + cell] = dSoft[k * ${nCell}u + cell] + gz;
    }
  }
`;
}

function emitBceGradLogit(): string {
  return /* wgsl */ `
fn bceGradLogit(logit : f32, tgt : f32) -> vec2f {
  let x = clamp(logit, -20.0, 20.0);
  let p = 1.0 / (1.0 + exp(-x));
  let loss = -(tgt * log(p + 1e-8) + (1.0 - tgt) * log(1.0 - p + 1e-8));
  return vec2f(loss, p - tgt);
}
`;
}

function emitInpaintMask(d: PixelDiscDims): string {
  const G = d.G;
  return /* wgsl */ `
fn buildInpaintMask() {
  let bw = max(2u, u32(floor(f32(${G}u) * 0.5)));
  let bh = max(2u, u32(floor(f32(${G}u) * 0.5)));
  let r0 = mulberry32(uni.maskSeed);
  let r1 = mulberry32(uni.maskSeed + 17u);
  let x0 = u32(floor(r0 * f32(${G}u - bw + 1u)));
  let y0 = u32(floor(r1 * f32(${G}u - bh + 1u)));
  for (var i = 0u; i < N_CELL; i = i + 1u) { set_auxB(i, 0.0); }
  for (var y = y0; y < y0 + bh; y = y + 1u) {
    for (var x = x0; x < x0 + bw; x = x + 1u) {
      set_auxB(y * ${G}u + x, 1.0);
    }
  }
}
`;
}

function emitVecFieldForceGrid(critFwdExpr: string, encInit: string): string {
  return /* wgsl */ `
fn evalForceGrid() {
  for (var y = 0u; y < G; y = y + 1u) {
    for (var x = 0u; x < G; x = x + 1u) {
      let cell = y * G + x;
      let uk = vec2f((f32(x) + 0.5) / f32(G), (f32(y) + 0.5) / f32(G));
      ${encInit}
      let F = ${critFwdExpr};
      set_auxA(cell, F.x);
      set_auxB(cell, F.y);
    }
  }
}
`;
}

function emitCriticDiscBody(d: PixelDiscDims, metaStats: number): string {
  const { G, K } = d;
  const nCell = G * G;
  const nW = pixelDiscWeightCount(d);
  const wl = pixelWeightLayout(d);
  const dens0 = "0u";
  const densAux = "2u * N_CELL";

  switch (d.kind) {
    case "vec-field": {
      if (wl.kind !== "vec-field") throw new Error("layout mismatch");
      return /* wgsl */ `
  evalForceGrid();
  ${emitBackboneForward(d, dens0)}
  var discR = 0.0;
  var nActive = 0.0;
  var dSoft : array<f32, ${K * nCell}>;
  for (var i = 0u; i < ${K * nCell}u; i = i + 1u) { dSoft[i] = 0.0; }
  var gW : array<f32, ${nW}>;
  for (var i = 0u; i < ${nW}u; i = i + 1u) { gW[i] = 0.0; }
  var gD : array<f32, ${nCell}>;
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { gD[i] = 0.0; }
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    if (dens_at(c) < ${fl(DENS_FLOOR)}) { continue; }
    nActive = nActive + 1.0;
    var vx = critW[${wl.headB}u];
    var vy = critW[${wl.headB}u + 1u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      vx = vx + critW[${wl.headW}u + k] * cSoft[k * ${nCell}u + c];
      vy = vy + critW[${wl.headW}u + ${K}u + k] * cSoft[k * ${nCell}u + c];
    }
    let tx = auxA(c);
    let ty = auxB(c);
    let dx = vx - tx;
    let dy = vy - ty;
    let r = sqrt(dx * dx + dy * dy + SOFT_EPS2);
    discR = discR + r;
    let s = 1.0 / r;
    let dvx = dx * s;
    let dvy = dy * s;
    gW[${wl.headB}u] = gW[${wl.headB}u] + dvx;
    gW[${wl.headB}u + 1u] = gW[${wl.headB}u + 1u] + dvy;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      gW[${wl.headW}u + k] = gW[${wl.headW}u + k] + dvx * cSoft[k * ${nCell}u + c];
      gW[${wl.headW}u + ${K}u + k] =
        gW[${wl.headW}u + ${K}u + k] + dvy * cSoft[k * ${nCell}u + c];
      dSoft[k * ${nCell}u + c] =
        dSoft[k * ${nCell}u + c] +
        dvx * critW[${wl.headW}u + k] + dvy * critW[${wl.headW}u + ${K}u + k];
    }
  }
  let activeN = max(nActive, 1.0);
  discR = discR / activeN;
  for (var i = ${wl.headW}u; i < ${wl.headW + 2 * K + 2}u; i = i + 1u) {
    gW[i] = gW[i] / activeN;
  }
  for (var k = 0u; k < ${K}u; k = k + 1u) {
    for (var c = 0u; c < ${nCell}u; c = c + 1u) {
      dSoft[k * ${nCell}u + c] = dSoft[k * ${nCell}u + c] / activeN;
    }
  }
  critMeta[${metaStats}u] = discR;
  critMeta[${metaStats}u + 2u] = auxA(0u);
  critMeta[${metaStats}u + 3u] = auxB(0u);
  ${emitBackboneBackward(d, dens0, nW)}
  for (var i = 0u; i < ${nW}u; i = i + 1u) { critMeta[${0}u + i] = gW[i]; }
`;
    }
    case "next-frame": {
      if (wl.kind !== "next-frame") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitBackboneForward(d, densAux)}
  var discR = 0.0;
  var dSoft : array<f32, ${K * nCell}>;
  for (var i = 0u; i < ${K * nCell}u; i = i + 1u) { dSoft[i] = 0.0; }
  var gW : array<f32, ${nW}>;
  for (var i = 0u; i < ${nW}u; i = i + 1u) { gW[i] = 0.0; }
  var gD : array<f32, ${nCell}>;
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { gD[i] = 0.0; }
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    var pred = critW[${wl.headB}u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      pred = pred + critW[${wl.headW}u + k] * cSoft[k * ${nCell}u + c];
    }
    let diff = pred - dens_at(c);
    let r = sqrt(diff * diff + SOFT_EPS2);
    discR = discR + r;
    let dPred = diff / r / f32(N_CELL);
    gW[${wl.headB}u] = gW[${wl.headB}u] + dPred;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      gW[${wl.headW}u + k] = gW[${wl.headW}u + k] + dPred * cSoft[k * ${nCell}u + c];
      dSoft[k * ${nCell}u + c] = dSoft[k * ${nCell}u + c] + dPred * critW[${wl.headW}u + k];
    }
  }
  discR = discR / f32(N_CELL);
  critMeta[${metaStats}u] = discR;
  ${emitBackboneBackward(d, densAux, nW)}
  for (var i = 0u; i < ${nW}u; i = i + 1u) { critMeta[${0}u + i] = gW[i]; }
`;
    }
    case "real-fake": {
      if (wl.kind !== "real-fake") throw new Error("layout mismatch");
      return /* wgsl */ `
  var cZ : array<f32, ${d.K}>;
  var cH : array<f32, ${d.hidden}>;
  var discLoss = 0.0;
  var gW : array<f32, ${nW}>;
  for (var i = 0u; i < ${nW}u; i = i + 1u) { gW[i] = 0.0; }
  var dSoft : array<f32, ${K * nCell}>;
  var gD : array<f32, ${nCell}>;
  var logit = 0.0;

  {
    for (var i = 0u; i < ${K * nCell}u; i = i + 1u) { dSoft[i] = 0.0; }
    for (var i = 0u; i < ${nCell}u; i = i + 1u) { gD[i] = 0.0; }
    ${emitBackboneForward(d, dens0)}
    ${emitGapMlpForward(d)}
    {
      let bce = bceGradLogit(logit, 1.0);
      discLoss = discLoss + bce.x;
      ${emitGapMlpBackward(d, "bce.y")}
    }
    ${emitBackboneBackward(d, dens0, nW)}
  }

  {
    for (var i = 0u; i < ${K * nCell}u; i = i + 1u) { dSoft[i] = 0.0; }
    for (var i = 0u; i < ${nCell}u; i = i + 1u) { gD[i] = 0.0; }
    ${emitBackboneForward(d, densAux)}
    ${emitGapMlpForward(d)}
    {
      let bce = bceGradLogit(logit, 0.0);
      discLoss = discLoss + bce.x;
      ${emitGapMlpBackward(d, "bce.y")}
    }
    ${emitBackboneBackward(d, densAux, nW)}
  }

  critMeta[${metaStats}u] = discLoss;
  for (var i = 0u; i < ${nW}u; i = i + 1u) { critMeta[${0}u + i] = gW[i]; }
`;
    }
    case "inpaint": {
      if (wl.kind !== "inpaint") throw new Error("layout mismatch");
      return /* wgsl */ `
  buildInpaintMask();
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    let m = auxB(c);
    densPack[${densAux} + c] = dens_at(c) * (1.0 - m);
  }
  ${emitBackboneForward(d, densAux)}
  var nMask = 0.0;
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    if (auxB(c) > 0.5) { nMask = nMask + 1.0; }
  }
  nMask = max(nMask, 1.0);
  var discR = 0.0;
  var dSoft : array<f32, ${K * nCell}>;
  for (var i = 0u; i < ${K * nCell}u; i = i + 1u) { dSoft[i] = 0.0; }
  var gW : array<f32, ${nW}>;
  for (var i = 0u; i < ${nW}u; i = i + 1u) { gW[i] = 0.0; }
  var gD : array<f32, ${nCell}>;
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { gD[i] = 0.0; }
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    if (auxB(c) <= 0.5) { continue; }
    var pred = critW[${wl.headB}u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      pred = pred + critW[${wl.headW}u + k] * cSoft[k * ${nCell}u + c];
    }
    let diff = pred - dens_at(c);
    let r = sqrt(diff * diff + SOFT_EPS2);
    discR = discR + r;
    let dPred = (diff / r) / nMask;
    gW[${wl.headB}u] = gW[${wl.headB}u] + dPred;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      gW[${wl.headW}u + k] = gW[${wl.headW}u + k] + dPred * cSoft[k * ${nCell}u + c];
      dSoft[k * ${nCell}u + c] = dSoft[k * ${nCell}u + c] + dPred * critW[${wl.headW}u + k];
    }
  }
  discR = discR / nMask;
  critMeta[${metaStats}u] = discR;
  ${emitBackboneBackward(d, densAux, nW)}
  for (var i = 0u; i < ${nW}u; i = i + 1u) { critMeta[${0}u + i] = gW[i]; }
`;
    }
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

function emitCriticGenBody(d: PixelDiscDims, metaStats: number): string {
  const { G, K } = d;
  const nCell = G * G;
  const wl = pixelWeightLayout(d);
  const dens0 = "0u";
  const densAux = "2u * N_CELL";

  switch (d.kind) {
    case "vec-field": {
      if (wl.kind !== "vec-field") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitBackboneForward(d, dens0)}
  var genR = 0.0;
  var nActive = 0.0;
  var dSoft : array<f32, ${K * nCell}>;
  for (var i = 0u; i < ${K * nCell}u; i = i + 1u) { dSoft[i] = 0.0; }
  var gD : array<f32, ${nCell}>;
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { gD[i] = 0.0; }
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    if (dens_at(c) < ${fl(DENS_FLOOR)}) { continue; }
    nActive = nActive + 1.0;
    var vx = critW[${wl.headB}u];
    var vy = critW[${wl.headB}u + 1u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      vx = vx + critW[${wl.headW}u + k] * cSoft[k * ${nCell}u + c];
      vy = vy + critW[${wl.headW}u + ${K}u + k] * cSoft[k * ${nCell}u + c];
    }
    let tx = auxA(c);
    let ty = auxB(c);
    let dx = vx - tx;
    let dy = vy - ty;
    let r = sqrt(dx * dx + dy * dy + SOFT_EPS2);
    genR = genR + r;
    let s = uni.genSign / r;
    let dvx = dx * s;
    let dvy = dy * s;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      dSoft[k * ${nCell}u + c] =
        dSoft[k * ${nCell}u + c] +
        dvx * critW[${wl.headW}u + k] + dvy * critW[${wl.headW}u + ${K}u + k];
    }
  }
  let activeN = max(nActive, 1.0);
  genR = genR / activeN;
  critMeta[${metaStats}u + 1u] = uni.genSign * genR;
  for (var k = 0u; k < ${K}u; k = k + 1u) {
    for (var c = 0u; c < ${nCell}u; c = c + 1u) {
      dSoft[k * ${nCell}u + c] = dSoft[k * ${nCell}u + c] / activeN;
    }
  }
  var gW : array<f32, ${pixelDiscWeightCount(d)}>;
  for (var i = 0u; i < ${pixelDiscWeightCount(d)}u; i = i + 1u) { gW[i] = 0.0; }
  ${emitBackboneBackward(d, dens0, pixelDiscWeightCount(d))}
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { set_d_dens(i, gD[i]); }
`;
    }
    case "next-frame": {
      if (wl.kind !== "next-frame") throw new Error("layout mismatch");
      return /* wgsl */ `
  ${emitBackboneForward(d, densAux)}
  var genR = 0.0;
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    var pred = critW[${wl.headB}u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      pred = pred + critW[${wl.headW}u + k] * cSoft[k * ${nCell}u + c];
    }
    let diff = pred - dens_at(c);
    let r = sqrt(diff * diff + SOFT_EPS2);
    genR = genR + r;
    add_d_dens(c, uni.genSign * (-diff / r) / f32(N_CELL));
  }
  genR = genR / f32(N_CELL);
  critMeta[${metaStats}u + 1u] = uni.genSign * genR;
`;
    }
    case "real-fake": {
      if (wl.kind !== "real-fake") throw new Error("layout mismatch");
      const nW = pixelDiscWeightCount(d);
      return /* wgsl */ `
  ${emitBackboneForward(d, dens0)}
  var cZ : array<f32, ${d.K}>;
  var cH : array<f32, ${d.hidden}>;
  var logit = 0.0;
  ${emitGapMlpForward(d)}
  critMeta[${metaStats}u + 1u] = uni.genSign * (-logit);
  var dSoft : array<f32, ${K * nCell}>;
  for (var i = 0u; i < ${K * nCell}u; i = i + 1u) { dSoft[i] = 0.0; }
  var gW : array<f32, ${nW}>;
  for (var i = 0u; i < ${nW}u; i = i + 1u) { gW[i] = 0.0; }
  var gD : array<f32, ${nCell}>;
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { gD[i] = 0.0; }
  ${emitGapMlpBackward(d, "uni.genSign * -1.0")}
  ${emitBackboneBackward(d, dens0, nW)}
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { set_d_dens(i, gD[i]); }
`;
    }
    case "inpaint": {
      if (wl.kind !== "inpaint") throw new Error("layout mismatch");
      const nW = pixelDiscWeightCount(d);
      return /* wgsl */ `
  var nMask = 0.0;
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    if (auxB(c) > 0.5) { nMask = nMask + 1.0; }
  }
  nMask = max(nMask, 1.0);
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    let m = auxB(c);
    densPack[${densAux} + c] = dens_at(c) * (1.0 - m);
  }
  ${emitBackboneForward(d, densAux)}
  var genR = 0.0;
  var dSoft : array<f32, ${K * nCell}>;
  for (var i = 0u; i < ${K * nCell}u; i = i + 1u) { dSoft[i] = 0.0; }
  var dD2 : array<f32, ${nCell}>;
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { dD2[i] = 0.0; }
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    if (auxB(c) <= 0.5) { continue; }
    var pred = critW[${wl.headB}u];
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      pred = pred + critW[${wl.headW}u + k] * cSoft[k * ${nCell}u + c];
    }
    let diff = pred - dens_at(c);
    let r = sqrt(diff * diff + SOFT_EPS2);
    genR = genR + r;
    let scale = uni.genSign / r / nMask;
    dD2[c] = dD2[c] - diff * scale;
    let dPred = diff * scale;
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      dSoft[k * ${nCell}u + c] = dSoft[k * ${nCell}u + c] + dPred * critW[${wl.headW}u + k];
    }
  }
  genR = genR / nMask;
  critMeta[${metaStats}u + 1u] = uni.genSign * genR;
  var gW : array<f32, ${nW}>;
  for (var i = 0u; i < ${nW}u; i = i + 1u) { gW[i] = 0.0; }
  var gD : array<f32, ${nCell}>;
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { gD[i] = 0.0; }
  ${emitBackboneBackward(d, densAux, nW)}
  for (var c = 0u; c < ${nCell}u; c = c + 1u) {
    if (auxB(c) <= 0.5) { dD2[c] = dD2[c] + gD[c]; }
  }
  for (var i = 0u; i < ${nCell}u; i = i + 1u) { set_d_dens(i, dD2[i]); }
`;
    }
    default: {
      const _e: never = d.kind;
      throw new Error(`pixel_disc: bad kind ${_e}`);
    }
  }
}

/**
 * Main fused module: particle sample + field + density + critic + disc Adam +
 * gen density path + fieldGrad into extGrads.
 *
 * Bindings (8 storage + uniform):
 *  0 uni, 1 fieldW, 2 critW, 3 scratch, 4 densI32, 5 densPack, 6 critMeta, 7 extGrads, 8 partPos
 */
export function pixelDiscShader(
  field: FieldLayout,
  opts: PixelDiscShaderOpts
): string {
  validatePixelDiscFusion(field);
  const d = opts.dims;
  const batchCap = Math.min(opts.batchCap, PIXEL_DISC_MAX_BATCH);
  const fieldLane = opts.fieldLane ?? "blend";
  const sl = trainScratchLayout(field, 1);
  const heads = field.spec.heads as HeadSpec[];
  const enc = field.encoding;
  const partStride = pixelParticleScratchFloats(field);
  const nCell = d.G * d.G;
  const nW = pixelDiscWeightCount(d);
  const maxW = Math.max(
    2,
    sl.encDim,
    ...heads.flatMap((h) => h.layers.map((L) => Math.max(L.outSize, L.inSize)))
  );

  const oPos = 0;
  const oF = 2;
  const oPos2 = 4;
  const oDF = 6;
  const oEnc = 8;
  const oField = 8 + sl.encStore;

  const critWorkspace = batchCap * partStride;
  const critEncBase = `${critWorkspace + 8}u`;
  const critFieldBase = (h: number) =>
    `${critWorkspace + 8 + sl.encStore}u + ${h === 0 ? 0 : sl.headBlk[0]}u`;

  const fieldBase = (h: number) =>
    `pBase + ${oField}u + ${h === 0 ? 0 : sl.headBlk[0]}u`;
  const encBase = () => `pBase + ${oEnc}u`;
  const fwdCall = (h: number) =>
    enc.kind === "raw"
      ? `fwd_head_${h}(uk, ${fieldBase(h)}, 0u)`
      : `fwd_head_${h}(${encBase()}, ${fieldBase(h)})`;
  const critFwdCall = (h: number) =>
    enc.kind === "raw"
      ? `fwd_head_${h}(uk, ${critFieldBase(h)}, 0u)`
      : `fwd_head_${h}(${critEncBase}, ${critFieldBase(h)})`;
  const bwdCall = (h: number, dExpr: string) =>
    enc.kind === "raw"
      ? `bwd_head_${h}(${dExpr}, ${fieldBase(h)})`
      : `bwd_head_${h}(${dExpr}, ${fieldBase(h)}, ${encBase()})`;
  const fieldForward =
    fieldLane === "blend"
      ? `(1.0 - uni.alpha) * ${fwdCall(0)} + uni.alpha * ${fwdCall(1)}`
      : fwdCall(fieldLane);
  const critFieldForward =
    fieldLane === "blend"
      ? `(1.0 - uni.alpha) * ${critFwdCall(0)} + uni.alpha * ${critFwdCall(1)}`
      : critFwdCall(fieldLane);
  const fieldBackward =
    fieldLane === "blend"
      ? `let _du0 = ${bwdCall(0, "dSig * (1.0 - uni.alpha)")};
    let _du1 = ${bwdCall(1, "dSig * uni.alpha")};`
      : `let _duLane = ${bwdCall(fieldLane, "dSig")};`;

  const metaGrads = 0;
  const metaM = nW;
  const metaV = 2 * nW;
  const metaStats = 3 * nW;

  const fwdStores = heads
    .map((h, i) => emitFwdStore(i, h, sl, maxW, enc))
    .join("\n");
  const bwdStores = heads
    .map((h, i) => emitBwdStore(i, h, sl, maxW, enc))
    .join("\n");
  const encodeFn = enc.kind === "raw" ? "" : emitEncode(enc);
  const critEncInit =
    enc.kind === "raw" ? "" : `encodeSite(uk, ${critEncBase});`;

  const fieldBlocks: string[] = [];
  for (const seg of field.segments) {
    if (seg.role === "grid") continue;
    const h = seg.head;
    if (fieldLane !== "blend" && h !== fieldLane) continue;
    const l = seg.layer;
    const L = heads[h].layers[l];
    const start = seg.floatOffset;
    const end = seg.floatOffset + seg.floatLength;
    const headOff = h === 0 ? 0 : sl.headBlk[0];
    const fb = `pBase + ${oField}u + ${headOff}u`;
    const aIn =
      l === 0
        ? enc.kind === "raw"
          ? `scratch[pBase + ${oPos}u + i]`
          : `scratch[pBase + ${oEnc}u + i]`
        : `scratch[${fb} + ${sl.aOff[h][l - 1]}u + i]`;
    if (seg.role === "kernel") {
      fieldBlocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let local = t - ${start}u;
    let i = local / ${L.outSize}u;
    let j = local % ${L.outSize}u;
    for (var s = 0u; s < uni.b; s = s + 1u) {
      let pBase = s * ${partStride}u;
      g = g + ${aIn} * scratch[${fb} + ${sl.dOff[h][l]}u + j];
    }
  }`);
    } else {
      fieldBlocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let j = t - ${start}u;
    for (var s = 0u; s < uni.b; s = s + 1u) {
      let pBase = s * ${partStride}u;
      g = g + scratch[${fb} + ${sl.dOff[h][l]}u + j];
    }
  }`);
    }
  }

  const hashFns = emitHashRng();
  const bceFn = d.kind === "real-fake" ? emitBceGradLogit() : "";
  const inpaintFn = d.kind === "inpaint" ? emitInpaintMask(d) : "";
  const forceGridFn =
    d.kind === "vec-field"
      ? emitVecFieldForceGrid(critFieldForward, critEncInit)
      : "";

  const criticDiscBody = emitCriticDiscBody(d, metaStats);
  const criticGenBody = emitCriticGenBody(d, metaStats);

  return /* wgsl */ `
struct Uni {
  width : f32,
  height : f32,
  alpha : f32,
  dt : f32,
  b : u32,
  partCount : u32,
  cursor : u32,
  applyDisc : u32,
  lr : f32,
  beta1 : f32,
  beta2 : f32,
  eps : f32,
  adamT : u32,
  genSign : f32,
  maskSeed : u32,
  pad1 : u32,
};
@group(0) @binding(0) var<uniform> uni : Uni;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> critW : array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch : array<f32>;
@group(0) @binding(4) var<storage, read_write> densI32 : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> densPack : array<f32>;
@group(0) @binding(6) var<storage, read_write> critMeta : array<f32>;
@group(0) @binding(7) var<storage, read_write> extGrads : array<f32>;
@group(0) @binding(8) var<storage, read> partPos : array<f32>;

const N_CELL : u32 = ${nCell}u;
const G : u32 = ${d.G}u;
const DENS_SCALE : f32 = ${fl(PIXEL_DISC_DENS_SCALE)};
const SOFT_EPS2 : f32 = ${fl(PIXEL_DISC_SOFT_EPS * PIXEL_DISC_SOFT_EPS)};

${COMMON}
${encodeFn}
${fwdStores}
${bwdStores}
${hashFns}
${bceFn}

fn dens_at(i : u32) -> f32 { return densPack[i]; }
fn set_dens(i : u32, v : f32) { densPack[i] = v; }
fn d_dens(i : u32) -> f32 { return densPack[N_CELL + i]; }
fn set_d_dens(i : u32, v : f32) { densPack[N_CELL + i] = v; }
fn add_d_dens(i : u32, v : f32) { densPack[N_CELL + i] = densPack[N_CELL + i] + v; }
fn auxA(i : u32) -> f32 { return densPack[2u * N_CELL + i]; }
fn set_auxA(i : u32, v : f32) { densPack[2u * N_CELL + i] = v; }
fn auxB(i : u32) -> f32 { return densPack[3u * N_CELL + i]; }
fn set_auxB(i : u32, v : f32) { densPack[3u * N_CELL + i] = v; }

${inpaintFn}
${forceGridFn}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn clearDens(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  atomicStore(&densI32[i], 0);
  densPack[i] = 0.0;
  densPack[N_CELL + i] = 0.0;
}

/** Zero densI32 only — keeps densPack dens (real) for real-fake fake splat. */
@compute @workgroup_size(${PIXEL_DISC_WG})
fn clearAtomics(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  atomicStore(&densI32[i], 0);
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn sampleAndSplat(@builtin(global_invocation_id) gid : vec3u) {
  let s = gid.x;
  if (s >= uni.b) { return; }
  let pBase = s * ${partStride}u;
  let idx = (uni.cursor + s) % max(uni.partCount, 1u);
  let px = partPos[idx * 2u];
  let py = partPos[idx * 2u + 1u];
  let uk = vec2f(px / uni.width, py / uni.height);
  scratch[pBase + ${oPos}u] = uk.x;
  scratch[pBase + ${oPos}u + 1u] = uk.y;
  ${enc.kind === "raw" ? "" : `encodeSite(uk, ${encBase()});`}
  let F = ${fieldForward};
  scratch[pBase + ${oF}u] = F.x;
  scratch[pBase + ${oF}u + 1u] = F.y;

  let scale = f32(G);
  var fx = uk.x * scale - 0.5;
  var fy = uk.y * scale - 0.5;
  fx = clamp(fx, 0.0, f32(G) - 1.0000001);
  fy = clamp(fy, 0.0, f32(G) - 1.0000001);
  let ix = u32(floor(fx));
  let iy = u32(floor(fy));
  let tx = fx - f32(ix);
  let ty = fy - f32(iy);
  let ix1 = min(ix + 1u, G - 1u);
  let iy1 = min(iy + 1u, G - 1u);
  let mass = f32(G * G) / f32(uni.b);
  atomicAdd(&densI32[iy * G + ix], i32((1.0 - tx) * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy * G + ix1], i32(tx * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix], i32((1.0 - tx) * ty * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix1], i32(tx * ty * mass * DENS_SCALE));
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn densToFloat(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  densPack[i] = f32(atomicLoad(&densI32[i])) / DENS_SCALE;
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn copyDensToAux(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  set_auxA(i, dens_at(i));
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn fakeSplat(@builtin(global_invocation_id) gid : vec3u) {
  let s = gid.x;
  if (s >= uni.b) { return; }
  let seed = uni.maskSeed + s * 2654435761u;
  let rx = mulberry32(seed);
  let ry = mulberry32(seed + 1013904223u);
  let scale = f32(G);
  var fx = rx * scale - 0.5;
  var fy = ry * scale - 0.5;
  fx = clamp(fx, 0.0, f32(G) - 1.0000001);
  fy = clamp(fy, 0.0, f32(G) - 1.0000001);
  let ix = u32(floor(fx));
  let iy = u32(floor(fy));
  let tx = fx - f32(ix);
  let ty = fy - f32(iy);
  let ix1 = min(ix + 1u, G - 1u);
  let iy1 = min(iy + 1u, G - 1u);
  let mass = f32(G * G) / f32(uni.b);
  atomicAdd(&densI32[iy * G + ix], i32((1.0 - tx) * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy * G + ix1], i32(tx * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix], i32((1.0 - tx) * ty * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix1], i32(tx * ty * mass * DENS_SCALE));
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn densToFloatFake(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  set_auxA(i, f32(atomicLoad(&densI32[i])) / DENS_SCALE);
}

@compute @workgroup_size(1)
fn criticDisc() {
  var cFeat : array<f32, ${d.E * nCell}>;
  var cSoft : array<f32, ${d.K * nCell}>;
  ${criticDiscBody}
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn discAdam(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x;
  if (t >= ${nW}u) { return; }
  let g = critMeta[${metaGrads}u + t];
  if (uni.applyDisc == 0u) { return; }
  let mm = uni.beta1 * critMeta[${metaM}u + t] + (1.0 - uni.beta1) * g;
  let vv = uni.beta2 * critMeta[${metaV}u + t] + (1.0 - uni.beta2) * g * g;
  critMeta[${metaM}u + t] = mm;
  critMeta[${metaV}u + t] = vv;
  let tf_ = f32(uni.adamT);
  let mhat = mm / (1.0 - pow(uni.beta1, tf_));
  let vhat = vv / (1.0 - pow(uni.beta2, tf_));
  critW[t] = critW[t] - uni.lr * mhat / (sqrt(vhat) + uni.eps);
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn clearDensGen(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= N_CELL) { return; }
  atomicStore(&densI32[i], 0);
  densPack[i] = 0.0;
  densPack[N_CELL + i] = 0.0;
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn virtualSplat(@builtin(global_invocation_id) gid : vec3u) {
  let s = gid.x;
  if (s >= uni.b) { return; }
  let pBase = s * ${partStride}u;
  let uk = vec2f(scratch[pBase + ${oPos}u], scratch[pBase + ${oPos}u + 1u]);
  let F = vec2f(scratch[pBase + ${oF}u], scratch[pBase + ${oF}u + 1u]);
  let up = uk + uni.dt * F;
  scratch[pBase + ${oPos2}u] = up.x;
  scratch[pBase + ${oPos2}u + 1u] = up.y;
  let scale = f32(G);
  var fx = up.x * scale - 0.5;
  var fy = up.y * scale - 0.5;
  fx = clamp(fx, 0.0, f32(G) - 1.0000001);
  fy = clamp(fy, 0.0, f32(G) - 1.0000001);
  let ix = u32(floor(fx));
  let iy = u32(floor(fy));
  let tx = fx - f32(ix);
  let ty = fy - f32(iy);
  let ix1 = min(ix + 1u, G - 1u);
  let iy1 = min(iy + 1u, G - 1u);
  let mass = f32(G * G) / f32(uni.b);
  atomicAdd(&densI32[iy * G + ix], i32((1.0 - tx) * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy * G + ix1], i32(tx * (1.0 - ty) * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix], i32((1.0 - tx) * ty * mass * DENS_SCALE));
  atomicAdd(&densI32[iy1 * G + ix1], i32(tx * ty * mass * DENS_SCALE));
}

@compute @workgroup_size(1)
fn criticGen() {
  for (var i = 0u; i < N_CELL; i = i + 1u) {
    densPack[i] = f32(atomicLoad(&densI32[i])) / DENS_SCALE;
    densPack[N_CELL + i] = 0.0;
  }
  var cFeat : array<f32, ${d.E * nCell}>;
  var cSoft : array<f32, ${d.K * nCell}>;
  ${criticGenBody}
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn densityVjpAndFieldBwd(@builtin(global_invocation_id) gid : vec3u) {
  let s = gid.x;
  if (s >= uni.b) { return; }
  let pBase = s * ${partStride}u;
  let up = vec2f(scratch[pBase + ${oPos2}u], scratch[pBase + ${oPos2}u + 1u]);
  let scale = f32(G);
  var fx = up.x * scale - 0.5;
  var fy = up.y * scale - 0.5;
  let insideX = fx > 0.0 && fx < f32(G) - 1.0;
  let insideY = fy > 0.0 && fy < f32(G) - 1.0;
  fx = clamp(fx, 0.0, f32(G) - 1.0000001);
  fy = clamp(fy, 0.0, f32(G) - 1.0000001);
  let ix = u32(floor(fx));
  let iy = u32(floor(fy));
  let tx = fx - f32(ix);
  let ty = fy - f32(iy);
  let ix1 = min(ix + 1u, G - 1u);
  let iy1 = min(iy + 1u, G - 1u);
  let mass = f32(G * G) / f32(uni.b);
  let g00 = densPack[N_CELL + iy * G + ix];
  let g10 = densPack[N_CELL + iy * G + ix1];
  let g01 = densPack[N_CELL + iy1 * G + ix];
  let g11 = densPack[N_CELL + iy1 * G + ix1];
  var dfx = mass * (-(1.0 - ty) * g00 + (1.0 - ty) * g10 - ty * g01 + ty * g11);
  var dfy = mass * (-(1.0 - tx) * g00 - tx * g10 + (1.0 - tx) * g01 + tx * g11);
  if (!insideX) { dfx = 0.0; }
  if (!insideY) { dfy = 0.0; }
  let dPosX = dfx * scale;
  let dPosY = dfy * scale;
  let dSig = vec2f(uni.dt * dPosX, uni.dt * dPosY);
  scratch[pBase + ${oDF}u] = dSig.x;
  scratch[pBase + ${oDF}u + 1u] = dSig.y;
  ${fieldBackward}
}

@compute @workgroup_size(${PIXEL_DISC_WG})
fn fieldGrad(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x;
  if (t >= ${field.totalFloats}u) { return; }
  var g = 0.0;
${fieldBlocks.join("\n")}
  extGrads[t] = g;
}
`;
}
