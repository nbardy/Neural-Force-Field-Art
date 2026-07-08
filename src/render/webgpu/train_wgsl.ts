/**
 * train_wgsl — PURE WGSL codegen for the fused TRAIN step (Phase 2).
 *
 * Two dispatches replace the entire tfjs learn stage (~40-100 dispatches,
 * some CPU-forwarded):
 *
 *   PASS A (one workgroup owns the whole batch, N ≤ 1024):
 *     per sample: a K-STEP ROLLOUT from (tp, vel=0) — force eval at
 *     norm(pos_k) → integrate → clip → wrap, carrying velocity — then 3 loss
 *     probes at norm(pos_K){,+HH x,+HH y}; all activations stored to a global
 *     scratch buffer; workgroup reduction for the isotropy covariance (over
 *     ALL K steps' scaled forces) + total loss; then per-sample ANALYTIC
 *     BPTT backward: loss → probe deltas → walk dL/dpos_k, dL/dvel_k chains
 *     back through the K transitions, per-layer δ's written next to the
 *     activations.
 *   PASS B (thread = packed weight entry):
 *     dW = Σ over (sample, site) of a_in ⊗ δ_out — sites are the K physics
 *     evals + 3 probes — then an in-place Adam update on the SAME packed
 *     weights buffer the advect kernel reads.
 *
 * K=1 reproduces main.ts helmholtzChaosLoss semantics EXACTLY (verified vs a
 * tfjs-autograd fixture at cos=1.0000000 — tools/train_test.ts). K>1 is the
 * "richer training signal": gradients see how particles FLOW through the
 * evolving field (position + accumulated velocity), which tfjs could only do
 * at K× dispatch cost. Isotropy generalizes to the concatenation of all K
 * steps' forces; chaos/divergence/spiral act on the final state.
 *
 * ZERO imports beyond ./advect_wgsl types. No hardcoded dims — everything is
 * generated from the validated FieldLayout (AlphaGOJS D=8 trap rule).
 *
 * Gradient semantics matched to tfjs: selu′/tanh′ from post-activations;
 * clipByValue′ = [lo ≤ pre-clip ≤ hi]; mod′ = 1; atan2 grads (−dy, dx)/r²;
 * minimum chain = earlier-k wins ties; means carry 1/N (isotropy 1/(N·K)).
 *
 * Scratch layout (floats, per sample block), K = rollout steps:
 *   header (6K+2): pos_0..pos_K, velPre_1..velPre_K, Fs_0..Fs_{K-1}
 *   site inputs ((K+3)·2): u per site — sites 0..K-1 are the physics evals,
 *     sites K..K+2 the probes at pos_K
 *   per site, per head: [a_l for every layer] then [δ_l for every layer]
 */

import { CLASS_SALT, type FieldLayout, type HeadSpec } from "./advect_wgsl";

export const TRAIN_WG = 256;
/** Pass A is now MULTI-workgroup (ceil(N/TRAIN_WG) workgroups for fwd/bwd),
 *  so the batch parallelizes across the GPU — batches up to this cap stay
 *  cheap (measured ~flat 256→1024). */
export const MAX_BATCH = 4096;
export const MAX_ROLLOUT = 16;

/** Loss constants — copied from main.ts helmholtzChaosLoss. */
export const LOSS = {
  W_CHAOS: 1.0,
  W_ISO: 1.0,
  W_DIV: 0.5,
  W_SPIRAL: 0.00002,
  HH: 1e-2,
  SPIRAL_TURNS: 3,
} as const;

// ---------------------------------------------------------------------------
// Scratch layout
// ---------------------------------------------------------------------------

export interface TrainScratchLayout {
  kSteps: number;
  /** physics sites (K) + probe sites (3) */
  sites: number;
  /** header floats: 6K+2 */
  headerSize: number;
  /** header offsets */
  posOff: number;     // pos_0..pos_K        (2(K+1) floats)
  velPreOff: number;  // velPre_1..velPre_K  (2K floats)
  fsOff: number;      // Fs_0..Fs_{K-1}      (2K floats)
  clsOff: number;     // sample class id, stored as f32 (1 float)
  /** floats per (site, head) block for head h */
  headBlk: number[];
  /** Σ headBlk */
  siteBlk: number;
  /** offset of site-input array within a sample block */
  siteInOff: number;
  /** offset of the (site,head) blocks within a sample block */
  headSiteOff: number;
  /** floats per sample block */
  sampleStride: number;
  /** per head: activation offsets per layer within its block */
  aOff: number[][];
  /** per head: delta offsets per layer within its block */
  dOff: number[][];
}

export function trainScratchLayout(
  field: FieldLayout,
  kSteps = 1
): TrainScratchLayout {
  if (!Number.isInteger(kSteps) || kSteps < 1 || kSteps > MAX_ROLLOUT) {
    throw new Error(`train: kSteps ${kSteps} outside [1, ${MAX_ROLLOUT}]`);
  }
  const heads = field.spec.heads as HeadSpec[];
  const aOff: number[][] = [];
  const dOff: number[][] = [];
  const headBlk: number[] = [];
  for (const h of heads) {
    const outs = h.layers.map((L) => L.outSize);
    const total = outs.reduce((a, b) => a + b, 0);
    let o = 0;
    const ao: number[] = [];
    for (const s of outs) {
      ao.push(o);
      o += s;
    }
    let d = total;
    const dg: number[] = [];
    for (const s of outs) {
      dg.push(d);
      d += s;
    }
    aOff.push(ao);
    dOff.push(dg);
    headBlk.push(2 * total);
  }
  const sites = kSteps + 3;
  const siteBlk = headBlk.reduce((a, b) => a + b, 0);
  const posOff = 0;
  const velPreOff = 2 * (kSteps + 1);
  const fsOff = velPreOff + 2 * kSteps;
  const clsOff = fsOff + 2 * kSteps;
  const headerSize = clsOff + 1; // = 6K+3
  const siteInOff = headerSize;
  const headSiteOff = siteInOff + sites * 2;
  const sampleStride = headSiteOff + sites * siteBlk;
  return {
    kSteps, sites, headerSize, posOff, velPreOff, fsOff, clsOff,
    headBlk, siteBlk, siteInOff, headSiteOff, sampleStride, aOff, dOff,
  };
}

/** Bytes for the scratch buffer at a given batch capacity. */
export function scratchBytes(
  field: FieldLayout,
  batchCap: number,
  kSteps = 1
): number {
  return trainScratchLayout(field, kSteps).sampleStride * batchCap * 4;
}

// ---------------------------------------------------------------------------
// Shared WGSL fragments
// ---------------------------------------------------------------------------

const COMMON = /* wgsl */ `
fn selu(x : f32) -> f32 {
  return select(1.7580993408473768 * (exp(x) - 1.0), 1.0507009873554805 * x, x > 0.0);
}
// derivative from the POST-activation value a = selu(x):
//   x>0  ⇔ a>0 : scale
//   x<=0        : scale·alpha·e^x = a + scale·alpha
fn seluD(a : f32) -> f32 {
  return select(a + 1.7580993408473768, 1.0507009873554805, a > 0.0);
}
fn tanhD(a : f32) -> f32 { return 1.0 - a * a; }
fn sigmoid_(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn sigmoidD(a : f32) -> f32 { return a * (1.0 - a); }
fn pcg(v : u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let t = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (t >> 22u) ^ t;
}
fn rand01(x : u32) -> f32 { return f32(x) * 2.3283064365386963e-10; }
`;

function actApply(act: string, expr: string): string {
  switch (act) {
    case "selu": return `selu(${expr})`;
    case "tanh": return `tanh(${expr})`;
    case "sigmoid": return `sigmoid_(${expr})`;
    default: return expr;
  }
}
function actDeriv(act: string, postAct: string): string {
  switch (act) {
    case "selu": return `seluD(${postAct})`;
    case "tanh": return `tanhD(${postAct})`;
    case "sigmoid": return `sigmoidD(${postAct})`;
    default: return `1.0`;
  }
}

/**
 * Forward evaluator for head `h` that STORES every post-activation to
 * scratch at `base` (the (sample,site,head) block start) and returns the
 * 2-vector output.
 */
function emitFwdStore(h: number, head: HeadSpec, sl: TrainScratchLayout, maxW: number): string {
  // inputs beyond [x,y] are one-hot class channels (validated by layoutField)
  const classChannels = head.layers[0].inSize - 2;
  const lines: string[] = [];
  lines.push(`fn fwd_head_${h}(p : vec2f, base : u32, cls : u32) -> vec2f {`);
  if (classChannels === 0) lines.push(`  let _cls = cls; // classless head`);
  lines.push(`  var cur : array<f32, ${maxW}>;`);
  lines.push(`  var nxt : array<f32, ${maxW}>;`);
  lines.push(`  cur[0] = p.x; cur[1] = p.y;`);
  if (classChannels > 0) {
    lines.push(`  for (var k = 0u; k < ${classChannels}u; k = k + 1u) {`);
    lines.push(`    cur[2u + k] = select(0.0, 1.0, k == cls);`);
    lines.push(`  }`);
  }
  head.layers.forEach((L, l) => {
    lines.push(`  // layer ${l}: ${L.inSize} -> ${L.outSize} (${L.activation})`);
    lines.push(`  for (var j = 0u; j < ${L.outSize}u; j = j + 1u) {`);
    lines.push(`    var s = weights[${L.biasOffset}u + j];`);
    lines.push(`    for (var i = 0u; i < ${L.inSize}u; i = i + 1u) {`);
    lines.push(`      s = s + cur[i] * weights[${L.weightOffset}u + i * ${L.outSize}u + j];`);
    lines.push(`    }`);
    lines.push(`    let a = ${actApply(L.activation, "s")};`);
    lines.push(`    scratch[base + ${sl.aOff[h][l]}u + j] = a;`);
    lines.push(`    nxt[j] = a;`);
    lines.push(`  }`);
    lines.push(`  for (var j = 0u; j < ${L.outSize}u; j = j + 1u) { cur[j] = nxt[j]; }`);
  });
  lines.push(`  return vec2f(cur[0], cur[1]);`);
  lines.push(`}`);
  return lines.join("\n");
}

/**
 * Backward for head `h`: takes dL/d(head output) (blend factor already
 * applied), reads stored activations, writes per-layer δ's to scratch, and
 * returns dL/d(input) — the probe/position chains need it.
 */
function emitBwdStore(h: number, head: HeadSpec, sl: TrainScratchLayout, maxW: number): string {
  const nL = head.layers.length;
  const lines: string[] = [];
  lines.push(`fn bwd_head_${h}(dOut : vec2f, base : u32) -> vec2f {`);
  lines.push(`  var dcur : array<f32, ${maxW}>;`);
  lines.push(`  var dprev : array<f32, ${maxW}>;`);
  {
    const L = head.layers[nL - 1];
    lines.push(`  // output layer ${nL - 1} (${L.activation})`);
    lines.push(`  {`);
    lines.push(`    let a0 = scratch[base + ${sl.aOff[h][nL - 1]}u];`);
    lines.push(`    let a1 = scratch[base + ${sl.aOff[h][nL - 1]}u + 1u];`);
    lines.push(`    dcur[0] = dOut.x * ${actDeriv(L.activation, "a0")};`);
    lines.push(`    dcur[1] = dOut.y * ${actDeriv(L.activation, "a1")};`);
    lines.push(`    scratch[base + ${sl.dOff[h][nL - 1]}u] = dcur[0];`);
    lines.push(`    scratch[base + ${sl.dOff[h][nL - 1]}u + 1u] = dcur[1];`);
    lines.push(`  }`);
  }
  for (let l = nL - 2; l >= 0; l--) {
    const Lnext = head.layers[l + 1];
    const L = head.layers[l];
    lines.push(`  // layer ${l} (${L.activation}): δ via W of layer ${l + 1}`);
    lines.push(`  for (var i = 0u; i < ${L.outSize}u; i = i + 1u) {`);
    lines.push(`    var s = 0.0;`);
    lines.push(`    for (var j = 0u; j < ${Lnext.outSize}u; j = j + 1u) {`);
    lines.push(`      s = s + weights[${Lnext.weightOffset}u + i * ${Lnext.outSize}u + j] * dcur[j];`);
    lines.push(`    }`);
    lines.push(`    let a = scratch[base + ${sl.aOff[h][l]}u + i];`);
    lines.push(`    dprev[i] = s * ${actDeriv(L.activation, "a")};`);
    lines.push(`    scratch[base + ${sl.dOff[h][l]}u + i] = dprev[i];`);
    lines.push(`  }`);
    lines.push(`  for (var i = 0u; i < ${L.outSize}u; i = i + 1u) { dcur[i] = dprev[i]; }`);
  }
  {
    const L0 = head.layers[0];
    lines.push(`  var du = vec2f(0.0, 0.0);`);
    lines.push(`  for (var j = 0u; j < ${L0.outSize}u; j = j + 1u) {`);
    lines.push(`    du.x = du.x + weights[${L0.weightOffset}u + j] * dcur[j];`);
    lines.push(`    du.y = du.y + weights[${L0.weightOffset + L0.outSize}u + j] * dcur[j];`);
    lines.push(`  }`);
    lines.push(`  return du;`);
  }
  lines.push(`}`);
  return lines.join("\n");
}

export interface TrainShaderOpts {
  /** rollout length K (compile-time; scratch layout must match) */
  kSteps?: number;
}

/**
 * Pass A bindings:
 *   0 uniform UA { res, forceMag, friction, maxVel, alpha, hh, _pad, n, seed, genRandom, kSteps }
 *     (kSteps in the uniform is informational — K is compiled in)
 *   1 storage read        weights  (packed FieldLayout — shared with advect)
 *   2 storage read_write  batch    array<vec2f>  (tp; written when genRandom=1)
 *   3 storage read_write  scratch  (per-sample activation/delta blocks)
 *   4 storage read_write  lossOut  array<f32,8>: [0]=total loss [1..3]=C00,C11,C01
 */
export function trainPassAShader(
  field: FieldLayout,
  opts: TrainShaderOpts = {}
): string {
  if (field.spec.kind !== "helmholtz") {
    throw new Error("train: v1 trains only the helmholtz field (2 heads)");
  }
  const K = opts.kSteps ?? 1;
  const sl = trainScratchLayout(field, K);
  const heads = field.spec.heads as HeadSpec[];
  const maxW = Math.max(
    2,
    ...heads.flatMap((h) => h.layers.map((L) => Math.max(L.outSize, L.inSize)))
  );
  const WG = TRAIN_WG;
  const STRIDE = sl.sampleStride;
  const hsBase = (site: string, h: number) =>
    `sBase + ${sl.headSiteOff}u + (${site}) * ${sl.siteBlk}u + ${h === 0 ? 0 : sl.headBlk[0]}u`;
  const aoutOff = (h: number) => sl.aOff[h][heads[h].layers.length - 1];
  const CLS_SALT = CLASS_SALT;
  const CLASSES_OR_1 = Math.max(1, field.classes);
  // probe outputs recomputed from stored post-activations at probe site `sx`
  const blendAt = (sx: string) =>
    `(1.0 - u.alpha) * vec2f(scratch[${hsBase(sx, 0)} + ${aoutOff(0)}u], scratch[${hsBase(sx, 0)} + ${aoutOff(0) + 1}u])` +
    ` + u.alpha * vec2f(scratch[${hsBase(sx, 1)} + ${aoutOff(1)}u], scratch[${hsBase(sx, 1)} + ${aoutOff(1) + 1}u])`;

  const TWO_PI = 6.283185307179586;

  return /* wgsl */ `
struct UA {
  res : vec2f,
  forceMag : f32,
  friction : f32,
  maxVel : f32,
  alpha : f32,
  hh : f32,
  partCount : u32,
  n : u32,
  seed : u32,
  batchSource : u32,   // 0=uploaded batch, 1=random points, 2=live particles
  kSteps : u32,
  mixCount : u32,      // first mixCount samples use the random source even
                       // when batchSource==2 (coverage floor, optional)
  wgCount : u32,       // number of fwd/bwd workgroups = ceil(n / TRAIN_WG)
  pad2 : u32,
  pad3 : u32,
};
@group(0) @binding(0) var<uniform> u : UA;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> batch : array<vec2f>;
@group(0) @binding(3) var<storage, read_write> scratch : array<f32>;
// lossOut: [0]=loss, [1..3]=C00/C11/C01, [4..6]=dLiso/dC00,dC11,dC01 (written by
// finalize, read by bwd — the batch-statistic gradient the whole batch shares).
@group(0) @binding(4) var<storage, read_write> lossOut : array<f32>;
// live particle state (the advect kernel's buffers) for batchSource==2
@group(0) @binding(5) var<storage, read> partPos : array<vec2f>;
@group(0) @binding(6) var<storage, read> partVel : array<vec2f>;
// per-workgroup partial reduction: vec4(Σ Fs.x², Σ Fs.y², Σ Fs.xy, Σ loss).
// MULTI-WORKGROUP: fwd runs ceil(n/WG) workgroups (full GPU, not one core),
// each writes its slice's partial here; finalize combines them. This is the
// vectorized batch reduction — the old single-workgroup pass A serialized the
// whole batch on ONE core.
@group(0) @binding(7) var<storage, read_write> partials : array<vec4f>;

${COMMON}

${heads.map((h, i) => emitFwdStore(i, h, sl, maxW)).join("\n\n")}

${heads.map((h, i) => emitBwdStore(i, h, sl, maxW)).join("\n\n")}

// per-workgroup reduction scratch: vec4(Σ Fs.x², Σ Fs.y², Σ Fs.xy, Σ loss)
// — the Fs sums run over ALL K steps' forces (isotropy batch = N·K).
var<workgroup> red : array<vec4f, ${WG}>;

fn mod2(p : vec2f, m : vec2f) -> vec2f { return p - floor(p / m) * m; }

// FWD: one sample per thread, ceil(n/WG) workgroups. Forward K-step rollout,
// store activations to scratch, reduce this workgroup's slice into partials[].
@compute @workgroup_size(${WG})
fn fwd(@builtin(global_invocation_id) gid : vec3u,
       @builtin(local_invocation_index) tid : u32,
       @builtin(workgroup_id) wgid : vec3u) {
  let n = u.n;
  let res = u.res;

  // spiral-loss constants (main.ts spiralLoss)
  let cx = res.x * 0.5;
  let cy = res.y * 0.5;
  let maxR = min(res.x, res.y) * 0.38;
  let b = maxR / (${LOSS.SPIRAL_TURNS}.0 * ${TWO_PI});

  // ---------------- FORWARD: K-step rollout ----------------
  let s = gid.x;
  var acc = vec4f(0.0);
  if (s < n) {
    var tp : vec2f;
    var v0 = vec2f(0.0, 0.0);
    // class: for particle samples, the particle's identity hash; for random/
    // uploaded samples, a slot hash (seed-independent so fixtures can
    // reproduce it). Ignored entirely when the layout is classless.
    var cls = pcg(s ^ ${CLS_SALT}u) % ${CLASSES_OR_1}u;
    if (u.batchSource == 1u || (u.batchSource == 2u && s < u.mixCount)) {
      // fresh uniform points (vel 0) — the original training source
      let r0 = pcg(s ^ (u.seed * 2654435769u));
      let r1 = pcg(r0);
      tp = vec2f(rand01(r0) * res.x, rand01(r1) * res.y);
      batch[s] = tp;
    } else if (u.batchSource == 2u) {
      // live particle states: real positions AND velocities. Coverage comes
      // from the reset slider (resets inject uniform vel-0 particles into the
      // cloud, hence into this batch). Distinct mixing constant so the index
      // stream is uncorrelated with the advect kernel's reset RNG.
      let idx = pcg(s ^ (u.seed * 3266489917u)) % max(u.partCount, 1u);
      tp = partPos[idx];
      v0 = partVel[idx];
      cls = pcg(idx ^ ${CLS_SALT}u) % ${CLASSES_OR_1}u;
      batch[s] = tp;
    } else {
      // uploaded batch (verification fixtures)
      tp = batch[s];
    }
    let sBase = s * ${STRIDE}u;

    var p = tp;
    var v = v0;
    scratch[sBase + ${sl.posOff}u] = p.x;
    scratch[sBase + ${sl.posOff}u + 1u] = p.y;
    scratch[sBase + ${sl.clsOff}u] = f32(cls);
    for (var k = 0u; k < ${K}u; k = k + 1u) {
      let uk = p / res;
      scratch[sBase + ${sl.siteInOff}u + k * 2u] = uk.x;
      scratch[sBase + ${sl.siteInOff}u + k * 2u + 1u] = uk.y;
      let F = (1.0 - u.alpha) * fwd_head_0(uk, ${hsBase("k", 0)}, cls)
            + u.alpha * fwd_head_1(uk, ${hsBase("k", 1)}, cls);
      let Fs = F * u.forceMag;
      scratch[sBase + ${sl.fsOff}u + k * 2u] = Fs.x;
      scratch[sBase + ${sl.fsOff}u + k * 2u + 1u] = Fs.y;
      let velPre = (v + Fs) * u.friction;
      scratch[sBase + ${sl.velPreOff}u + k * 2u] = velPre.x;
      scratch[sBase + ${sl.velPreOff}u + k * 2u + 1u] = velPre.y;
      v = clamp(velPre, vec2f(-u.maxVel), vec2f(u.maxVel));
      p = mod2(p + v, res);
      scratch[sBase + ${sl.posOff}u + (k + 1u) * 2u] = p.x;
      scratch[sBase + ${sl.posOff}u + (k + 1u) * 2u + 1u] = p.y;
      acc = acc + vec4f(Fs.x * Fs.x, Fs.y * Fs.y, Fs.x * Fs.y, 0.0);
    }
    let np = p;

    // probe sites at pn = pos_K/res (chaos + divergence)
    let pn = np / res;
    let p0 = pn;
    let px = pn + vec2f(u.hh, 0.0);
    let py = pn + vec2f(0.0, u.hh);
    scratch[sBase + ${sl.siteInOff}u + ${K * 2}u] = p0.x;
    scratch[sBase + ${sl.siteInOff}u + ${K * 2 + 1}u] = p0.y;
    scratch[sBase + ${sl.siteInOff}u + ${K * 2 + 2}u] = px.x;
    scratch[sBase + ${sl.siteInOff}u + ${K * 2 + 3}u] = px.y;
    scratch[sBase + ${sl.siteInOff}u + ${K * 2 + 4}u] = py.x;
    scratch[sBase + ${sl.siteInOff}u + ${K * 2 + 5}u] = py.y;
    let f0 = (1.0 - u.alpha) * fwd_head_0(p0, ${hsBase(`${K}u`, 0)}, cls) + u.alpha * fwd_head_1(p0, ${hsBase(`${K}u`, 1)}, cls);
    let fx = (1.0 - u.alpha) * fwd_head_0(px, ${hsBase(`${K + 1}u`, 0)}, cls) + u.alpha * fwd_head_1(px, ${hsBase(`${K + 1}u`, 1)}, cls);
    let fy = (1.0 - u.alpha) * fwd_head_0(py, ${hsBase(`${K + 2}u`, 0)}, cls) + u.alpha * fwd_head_1(py, ${hsBase(`${K + 2}u`, 1)}, cls);

    // per-sample loss terms (means applied via 1/n)
    let dxv = fx - f0;
    let dyv = fy - f0;
    let sepx = dot(dxv, dxv);
    let sepy = dot(dyv, dyv);
    let sq = sqrt(sepx + sepy + 1e-12);
    let sep = sq / (u.hh * 1.4142 + 1e-9);
    let chaos_i = -log(sep + 1e-6);

    let g = ((fx.x - f0.x) + (fy.y - f0.y)) / u.hh;
    let div_i = g * g;

    // spiral (winner-take-all over k, earlier k wins ties like tfjs minimum)
    let dx = np.x - cx;
    let dy = np.y - cy;
    let r = sqrt(dx * dx + dy * dy + 1e-4);
    let phi = atan2(dy, dx);
    var best = 1e8;
    for (var k = 0u; k <= ${LOSS.SPIRAL_TURNS + 1}u; k = k + 1u) {
      let theta = phi + ${TWO_PI} * f32(k);
      let rsp = b * max(theta, 0.0);
      let d = (r - rsp) * (r - rsp);
      if (d < best) { best = d; }
    }

    acc = acc + vec4f(0.0, 0.0, 0.0,
                      ${LOSS.W_CHAOS} * chaos_i + ${LOSS.W_DIV} * div_i + ${LOSS.W_SPIRAL} * best);
  }
  red[tid] = acc;
  workgroupBarrier();
  var stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) { partials[wgid.x] = red[0]; }
}

// FINALIZE: combine the per-workgroup partials → full-batch covariance → the
// shared isotropy gradient dLiso/dC (into lossOut[4..6]) + total loss. One
// workgroup; wgCount = ceil(n/WG) is small.
@compute @workgroup_size(${WG})
fn finalize(@builtin(local_invocation_index) tid : u32) {
  let nf = f32(u.n);
  let nkf = nf * ${K}.0;
  var acc = vec4f(0.0);
  for (var i = tid; i < u.wgCount; i = i + ${WG}u) { acc = acc + partials[i]; }
  red[tid] = acc;
  workgroupBarrier();
  var stride = ${WG / 2}u;
  loop {
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
    if (stride == 0u) { break; }
  }
  if (tid == 0u) {
    let C00 = red[0].x / nkf;
    let C11 = red[0].y / nkf;
    let C01 = red[0].z / nkf;
    let eps = 1e-6;
    let S = C00 + C11 + eps;
    let D = C00 - C11;
    let Liso = (D * D + 4.0 * C01 * C01) / (S * S);
    lossOut[0] = red[0].w / nf + ${LOSS.W_ISO} * Liso;
    lossOut[1] = C00; lossOut[2] = C11; lossOut[3] = C01;
    lossOut[4] = 2.0 * D / (S * S) - 2.0 * Liso / S;   // dLiso/dC00
    lossOut[5] = -2.0 * D / (S * S) - 2.0 * Liso / S;  // dLiso/dC11
    lossOut[6] = 8.0 * C01 / (S * S);                  // dLiso/dC01
  }
}

// BWD (BPTT): one sample per thread, ceil(n/WG) workgroups. Reads the shared
// batch-statistic gradient dC from lossOut, walks the K transitions back.
@compute @workgroup_size(${WG})
fn bwd(@builtin(global_invocation_id) gid : vec3u) {
  let n = u.n;
  let nf = f32(n);
  let nkf = nf * ${K}.0;
  let res = u.res;
  let cx = res.x * 0.5;
  let cy = res.y * 0.5;
  let maxR = min(res.x, res.y) * 0.38;
  let b = maxR / (${LOSS.SPIRAL_TURNS}.0 * ${TWO_PI});
  let s = gid.x;
  if (s >= n) { return; }
  let dC = vec3f(lossOut[4], lossOut[5], lossOut[6]);
  {
    let sBase = s * ${STRIDE}u;
    let np = vec2f(scratch[sBase + ${sl.posOff + 2 * K}u], scratch[sBase + ${sl.posOff + 2 * K + 1}u]);

    // recompute probe outputs from stored post-activations
    let f0 = ${blendAt(`${K}u`)};
    let fx = ${blendAt(`${K + 1}u`)};
    let fy = ${blendAt(`${K + 2}u`)};

    // chaos deltas
    let dxv = fx - f0;
    let dyv = fy - f0;
    let sepx = dot(dxv, dxv);
    let sepy = dot(dyv, dyv);
    let sq = sqrt(sepx + sepy + 1e-12);
    let denomc = u.hh * 1.4142 + 1e-9;
    let sep = sq / denomc;
    let cch = -(${LOSS.W_CHAOS} / nf) * (1.0 / (sep + 1e-6)) * (1.0 / denomc) / sq;
    var dfx = cch * dxv;
    var dfy = cch * dyv;
    var df0 = -cch * (dxv + dyv);

    // divergence deltas
    let g = ((fx.x - f0.x) + (fy.y - f0.y)) / u.hh;
    let gd = ${LOSS.W_DIV} * 2.0 * g / (nf * u.hh);
    dfx.x = dfx.x + gd;
    dfy.y = dfy.y + gd;
    df0.x = df0.x - gd;
    df0.y = df0.y - gd;

    // spiral delta on pos_K
    let dx = np.x - cx;
    let dy = np.y - cy;
    let r2e = dx * dx + dy * dy + 1e-4;
    let r = sqrt(r2e);
    let r2 = dx * dx + dy * dy; // atan2 jacobian uses the UN-epsiloned r²
    let phi = atan2(dy, dx);
    var best = 1e8;
    var bestTheta = 0.0;
    for (var k = 0u; k <= ${LOSS.SPIRAL_TURNS + 1}u; k = k + 1u) {
      let theta = phi + ${TWO_PI} * f32(k);
      let rsp = b * max(theta, 0.0);
      let d = (r - rsp) * (r - rsp);
      if (d < best) { best = d; bestTheta = theta; }
    }
    let rspBest = b * max(bestTheta, 0.0);
    let spc = (${LOSS.W_SPIRAL} / nf) * 2.0 * (r - rspBest);
    let reluMask = select(0.0, 1.0, bestTheta > 0.0);

    // probe backward (input grads flow to pn); blend factors on the way in
    var dpn = vec2f(0.0);
    dpn = dpn + bwd_head_0(df0 * (1.0 - u.alpha), ${hsBase(`${K}u`, 0)});
    dpn = dpn + bwd_head_1(df0 * u.alpha,         ${hsBase(`${K}u`, 1)});
    dpn = dpn + bwd_head_0(dfx * (1.0 - u.alpha), ${hsBase(`${K + 1}u`, 0)});
    dpn = dpn + bwd_head_1(dfx * u.alpha,         ${hsBase(`${K + 1}u`, 1)});
    dpn = dpn + bwd_head_0(dfy * (1.0 - u.alpha), ${hsBase(`${K + 2}u`, 0)});
    dpn = dpn + bwd_head_1(dfy * u.alpha,         ${hsBase(`${K + 2}u`, 1)});

    // dL/dpos_K = spiral + probes/res ; dL/dvel_K = 0 (loss ignores velocity)
    var dpos = spc * (vec2f(dx, dy) / r - b * reluMask * vec2f(-dy, dx) / r2)
             + dpn / res;
    var dvel = vec2f(0.0);

    // walk transitions k = K-1 .. 0:
    //   velPre_{k+1} = (vel_k + Fs_k)·friction ; vel_{k+1} = clamp(velPre)
    //   pos_{k+1} = pos_k + vel_{k+1}  (mod ≡ identity)
    for (var kk = 0u; kk < ${K}u; kk = kk + 1u) {
      let k = ${K - 1}u - kk;
      let velPre = vec2f(scratch[sBase + ${sl.velPreOff}u + k * 2u],
                         scratch[sBase + ${sl.velPreOff}u + k * 2u + 1u]);
      let Fs = vec2f(scratch[sBase + ${sl.fsOff}u + k * 2u],
                     scratch[sBase + ${sl.fsOff}u + k * 2u + 1u]);
      let mask = vec2f(
        select(0.0, 1.0, abs(velPre.x) <= u.maxVel),
        select(0.0, 1.0, abs(velPre.y) <= u.maxVel)
      );
      dvel = dvel + dpos;                    // pos_{k+1} uses vel_{k+1} directly
      let dpre = dvel * mask;
      var dFs = dpre * u.friction;
      dFs.x = dFs.x + ${LOSS.W_ISO} * (2.0 * dC.x * Fs.x + dC.z * Fs.y) / nkf;
      dFs.y = dFs.y + ${LOSS.W_ISO} * (2.0 * dC.y * Fs.y + dC.z * Fs.x) / nkf;
      let dF = dFs * u.forceMag;
      let du0 = bwd_head_0(dF * (1.0 - u.alpha), ${hsBase("k", 0)});
      let du1 = bwd_head_1(dF * u.alpha,         ${hsBase("k", 1)});
      dpos = dpos + (du0 + du1) / res;       // dL/dpos_k
      dvel = dpre * u.friction;              // dL/dvel_k
    }
  }
}
`;
}

// ---------------------------------------------------------------------------
// PASS B — gradient assembly + Adam
// ---------------------------------------------------------------------------

export const TRAIN_WG_B = 64;

/**
 * Pass B bindings:
 *   0 uniform UB { lr, beta1, beta2, eps, t, apply, n, pad }
 *   1 storage read_write weights
 *   2 storage read       scratch
 *   3 storage read_write grads   (always written — verification reads this)
 *   4 storage read_write adamM
 *   5 storage read_write adamV
 * Thread = packed weight float. dW[i][j] = Σ_{sample,site} a_in[i]·δ[j],
 * sites = K physics evals + 3 probes.
 */
export function trainPassBShader(
  field: FieldLayout,
  opts: TrainShaderOpts = {}
): string {
  if (field.spec.kind !== "helmholtz") {
    throw new Error("train: v1 trains only the helmholtz field (2 heads)");
  }
  const K = opts.kSteps ?? 1;
  const sl = trainScratchLayout(field, K);
  const heads = field.spec.heads as HeadSpec[];
  const STRIDE = sl.sampleStride;

  const blocks: string[] = [];
  for (const seg of field.segments) {
    const h = seg.head;
    const l = seg.layer;
    const L = heads[h].layers[l];
    const classChannels = l === 0 ? heads[h].layers[0].inSize - 2 : 0;
    const start = seg.floatOffset;
    const end = seg.floatOffset + seg.floatLength;
    const headOff = h === 0 ? 0 : sl.headBlk[0];
    const hsBase = `sBase + ${sl.headSiteOff}u + site * ${sl.siteBlk}u + ${headOff}u`;
    const aIn =
      l === 0
        ? `scratch[sBase + ${sl.siteInOff}u + site * 2u + i]`
        : `scratch[${hsBase} + ${sl.aOff[h][l - 1]}u + i]`;
    if (seg.role === "kernel" && classChannels > 0) {
      // layer 0 of a class-aware head: rows 0,1 read the site position, rows
      // 2+k are the one-hot — only the sample's class row accumulates δ.
      blocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let local = t - ${start}u;
    let i = local / ${L.outSize}u;
    let j = local % ${L.outSize}u;
    for (var s = 0u; s < ub.n; s = s + 1u) {
      let sBase = s * ${STRIDE}u;
      let cls = u32(scratch[sBase + ${sl.clsOff}u]);
      for (var site = 0u; site < ${sl.sites}u; site = site + 1u) {
        var aIn = 0.0;
        if (i < 2u) {
          aIn = scratch[sBase + ${sl.siteInOff}u + site * 2u + i];
        } else if ((i - 2u) == cls) {
          aIn = 1.0;
        }
        g = g + aIn * scratch[${hsBase} + ${sl.dOff[h][l]}u + j];
      }
    }
  }`);
    } else if (seg.role === "kernel") {
      blocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let local = t - ${start}u;
    let i = local / ${L.outSize}u;
    let j = local % ${L.outSize}u;
    for (var s = 0u; s < ub.n; s = s + 1u) {
      let sBase = s * ${STRIDE}u;
      for (var site = 0u; site < ${sl.sites}u; site = site + 1u) {
        g = g + ${aIn} * scratch[${hsBase} + ${sl.dOff[h][l]}u + j];
      }
    }
  }`);
    } else {
      blocks.push(`
  if (t >= ${start}u && t < ${end}u) {
    let j = t - ${start}u;
    for (var s = 0u; s < ub.n; s = s + 1u) {
      let sBase = s * ${STRIDE}u;
      for (var site = 0u; site < ${sl.sites}u; site = site + 1u) {
        g = g + scratch[${hsBase} + ${sl.dOff[h][l]}u + j];
      }
    }
  }`);
    }
  }

  return /* wgsl */ `
struct UB {
  lr : f32,
  beta1 : f32,
  beta2 : f32,
  eps : f32,
  t : u32,
  apply : u32,
  n : u32,
  pad : u32,
};
@group(0) @binding(0) var<uniform> ub : UB;
@group(0) @binding(1) var<storage, read_write> weights : array<f32>;
@group(0) @binding(2) var<storage, read> scratch : array<f32>;
@group(0) @binding(3) var<storage, read_write> grads : array<f32>;
@group(0) @binding(4) var<storage, read_write> adamM : array<f32>;
@group(0) @binding(5) var<storage, read_write> adamV : array<f32>;

@compute @workgroup_size(${TRAIN_WG_B})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let t = gid.x;
  if (t >= ${field.totalFloats}u) { return; }
  var g = 0.0;
${blocks.join("\n")}
  grads[t] = g;

  if (ub.apply == 1u) {
    // Adam, tfjs conventions: correction via β^t, t ≥ 1 on the first step.
    let m = ub.beta1 * adamM[t] + (1.0 - ub.beta1) * g;
    let v = ub.beta2 * adamV[t] + (1.0 - ub.beta2) * g * g;
    adamM[t] = m;
    adamV[t] = v;
    let tf_ = f32(ub.t);
    let mhat = m / (1.0 - pow(ub.beta1, tf_));
    let vhat = v / (1.0 - pow(ub.beta2, tf_));
    weights[t] = weights[t] - ub.lr * mhat / (sqrt(vhat) + ub.eps);
  }
}
`;
}
