/**
 * Gate 1 (gradcheck) + gate 3 (perf) for the WGSL 2D Gaussian-splat rasterizer
 * (src/splat/raster_wgsl.ts, src/splat/raster.ts), on a REAL WebGPU adapter
 * (Dawn/Metal) via bun-webgpu — no browser, explicit device + canvas-less.
 *
 *   bun tools/splat/raster_test.ts
 *
 * Checks:
 *   0. reparam Jacobian (conic + sigmoid/exp) analytic vs float64 JS finite
 *      differences on random single splats — catches sign errors BEFORE the GPU
 *      gradcheck (docs derivation-care note). Pure JS, no GPU.
 *   1. GRADCHECK: 64 splats @32x32, cap 256. Central finite differences THROUGH
 *      the GPU forward vs the GPU backward+chain, ~200 random raw-param probes
 *      across all 5 buffers with a fixed random dL/dpixels. Probes crossing an
 *      alpha/visibility kink (large second difference) are detected & excluded.
 *      Gate: rel err < 2e-2 on >=95% of the remaining probes.
 *   3. PERF: 200K splats @256^2, spread random init. forward ms and fwd+bwd ms
 *      (warmup + shared-GPU rules from tools/clip/README.md). No gate — report.
 *
 * Deterministic pcg-seeded RNG throughout (tools/splat/scene.ts).
 */
import { setupGlobals } from "bun-webgpu";
import { RasterEngine } from "../../src/splat/raster";
import { SCALE_MIN, SCALE_MAX } from "../../src/splat/raster_wgsl";
import { makeRng, packParams, cpuGradRef, SplatArrays } from "./scene";

setupGlobals();
const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
  process.exit(1);
}
const device: any = await adapter.requestDevice();
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? "?"}\n`);

let failures = 0;
const check = (ok: boolean, msg: string): void => {
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`);
};

// ==========================================================================
// Check 0 — reparam Jacobian vs float64 JS finite differences (no GPU)
// ==========================================================================
// float64 mirror of the prep reparam (conic from logScale/theta) and the chain
// Jacobian. These are the exact formulas baked in raster_wgsl.ts.
const clampd = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
function conicOf(lsx: number, lsy: number, th: number): [number, number, number] {
  const sx = clampd(Math.exp(lsx), SCALE_MIN, SCALE_MAX);
  const sy = clampd(Math.exp(lsy), SCALE_MIN, SCALE_MAX);
  const ix = 1 / (sx * sx);
  const iy = 1 / (sy * sy);
  const c = Math.cos(th);
  const s = Math.sin(th);
  return [c * c * ix + s * s * iy, c * s * (ix - iy), s * s * ix + c * c * iy];
}
// analytic chain: given upstream grads on (a,b,c), return grads on (lsx,lsy,th)
function chainConic(
  lsx: number,
  lsy: number,
  th: number,
  gA: number,
  gB: number,
  gC: number
): [number, number, number] {
  const ex = Math.exp(lsx);
  const ey = Math.exp(lsy);
  const sx = clampd(ex, SCALE_MIN, SCALE_MAX);
  const sy = clampd(ey, SCALE_MIN, SCALE_MAX);
  const gateX = ex > SCALE_MIN && ex < SCALE_MAX ? 1 : 0;
  const gateY = ey > SCALE_MIN && ey < SCALE_MAX ? 1 : 0;
  const ix = 1 / (sx * sx);
  const iy = 1 / (sy * sy);
  const c = Math.cos(th);
  const s = Math.sin(th);
  const gix = gA * c * c + gB * c * s + gC * s * s;
  const giy = gA * s * s - gB * c * s + gC * c * c;
  const glsx = gix * (-2 * ix) * gateX;
  const glsy = giy * (-2 * iy) * gateY;
  const D = ix - iy;
  const gth = D * ((c * c - s * s) * gB + 2 * c * s * (gC - gA));
  return [glsx, glsy, gth];
}
{
  const rng = makeRng(0xc0ffee);
  let worst = 0;
  const N = 200;
  for (let t = 0; t < N; t++) {
    // keep scales comfortably inside the clamp so exp() branch is smooth
    const lsx = rng.range(Math.log(0.6), Math.log(40));
    const lsy = rng.range(Math.log(0.6), Math.log(40));
    const th = rng.range(0, Math.PI);
    // arbitrary upstream conic grads
    const gA = rng.normal();
    const gB = rng.normal();
    const gC = rng.normal();
    const [glsx, glsy, gth] = chainConic(lsx, lsy, th, gA, gB, gC);
    // L(params) = gA*a + gB*b + gC*c ; FD each raw param
    const L = (x: number, y: number, z: number): number => {
      const [a, b, c] = conicOf(x, y, z);
      return gA * a + gB * b + gC * c;
    };
    const e = 1e-5;
    const fdx = (L(lsx + e, lsy, th) - L(lsx - e, lsy, th)) / (2 * e);
    const fdy = (L(lsx, lsy + e, th) - L(lsx, lsy - e, th)) / (2 * e);
    const fdt = (L(lsx, lsy, th + e) - L(lsx, lsy, th - e)) / (2 * e);
    const rel = (an: number, fd: number) => Math.abs(an - fd) / (Math.abs(an) + Math.abs(fd) + 1e-9);
    worst = Math.max(worst, rel(glsx, fdx), rel(glsy, fdy), rel(gth, fdt));
  }
  check(worst < 1e-4, `reparam Jacobian vs float64 FD: worst rel err ${worst.toExponential(2)} (< 1e-4)`);
}

// sigmoid reparam is trivial; the GPU gradcheck below covers color/opacity end
// to end. (A separate JS unit here would just mirror d/dx sigmoid = s(1-s).)

// ==========================================================================
// Check 1 — GRADCHECK (finite diff through GPU forward vs GPU backward+chain)
// ==========================================================================
// gradScale 2^20 (step ~1e-6): grads here are moderate (|<~1|), so no i32
// overflow, and tiny grads (~1e-4) keep ~1% fixed-point resolution — enough to
// compare against the float64 reference in 1a. (The fit demo/perf scenes use a
// smaller scale; the scale is per-instance config, see report.)
const GC = { G: 64, H: 32, W: 32, cap: 256, gradScale: 1 << 20 };
const PARAM_STRIDE = 9;

function benignScene(rng: ReturnType<typeof makeRng>, G: number, H: number, W: number): SplatArrays {
  const mean = new Float32Array(2 * G);
  const logScale = new Float32Array(2 * G);
  const theta = new Float32Array(G);
  const colorRaw = new Float32Array(3 * G);
  const opacityRaw = new Float32Array(G);
  for (let i = 0; i < G; i++) {
    // spread means away from the border so no splat is clipped at an edge
    mean[2 * i] = rng.range(4, W - 4);
    mean[2 * i + 1] = rng.range(4, H - 4);
    // scales ~2-6 px, well inside the [0.3,64] clamp (gate == 1, smooth exp)
    logScale[2 * i] = Math.log(3) + 0.3 * rng.normal();
    logScale[2 * i + 1] = Math.log(3) + 0.3 * rng.normal();
    theta[i] = rng.range(0, Math.PI);
    colorRaw[3 * i] = 0.8 * rng.normal();
    colorRaw[3 * i + 1] = 0.8 * rng.normal();
    colorRaw[3 * i + 2] = 0.8 * rng.normal();
    // opacity sigmoid(~1) ~= 0.73: comfortably above 1/255, below the 0.99 clamp
    opacityRaw[i] = 1.0 + 0.4 * rng.normal();
  }
  return { mean, logScale, theta, colorRaw, opacityRaw };
}

function groupOf(idx: number, G: number): { name: string; eps: number } {
  if (idx < 2 * G) return { name: "mean", eps: 1e-2 };
  if (idx < 4 * G) return { name: "logScale", eps: 2e-3 };
  if (idx < 5 * G) return { name: "theta", eps: 2e-3 };
  if (idx < 8 * G) return { name: "color", eps: 2e-3 };
  return { name: "opacity", eps: 2e-3 };
}

{
  const rng = makeRng(20260707);
  const eng = await RasterEngine.create(device, GC);
  const scene = benignScene(rng, GC.G, GC.H, GC.W);
  const base = packParams(GC.G, scene);
  // fixed random dL/dpixels
  const go = new Float32Array(3 * GC.H * GC.W);
  for (let i = 0; i < go.length; i++) go[i] = 0.5 * rng.normal();

  // analytic grads (one forward+backward+chain at base)
  eng.setParams(base);
  eng.runForward();
  eng.setGradImage(go);
  eng.runBackward();
  const analytic = await eng.readGradRaw();

  // -------------------------------------------------------------------------
  // Check 1a (authoritative) — GPU backward vs float64 CPU forward+backward+
  // chain reference. Same algorithm, same HARD gates, so there is no finite-
  // difference gate-crossing noise: every raw param is compared directly. This
  // is a stronger, exclusion-free realization of the spec's "GPU backward is
  // correct" intent (see 1b for the spec's literal FD-through-forward form).
  // -------------------------------------------------------------------------
  {
    const ref = cpuGradRef(GC.G, GC.H, GC.W, scene, go, [0.5, 0.5, 0.5]);
    const ATOL_A = 2e-4; // f32/fixed-point noise floor for near-zero grads
    let maxRelBig = 0; // worst rel among grads not covered by the atol floor
    let n = 0;
    let pass = 0;
    const gmax: Record<string, number> = {};
    for (let i = 0; i < GC.G * PARAM_STRIDE; i++) {
      const a = ref[i];
      const b = analytic[i];
      if (Math.abs(a) + Math.abs(b) < 1e-6) continue; // exact-zero grads
      const rel = Math.abs(a - b) / (Math.abs(a) + Math.abs(b) + 1e-9);
      const ok = rel < 2e-2 || Math.abs(a - b) < ATOL_A;
      const g = groupOf(i, GC.G).name;
      gmax[g] = Math.max(gmax[g] ?? 0, rel);
      if (Math.abs(a - b) >= ATOL_A) maxRelBig = Math.max(maxRelBig, rel);
      n++;
      if (ok) pass++;
    }
    console.log(
      `  1a per-group max rel: ` +
        Object.entries(gmax)
          .map(([g, v]) => `${g}=${v.toExponential(2)}`)
          .join("  ")
    );
    check(
      pass / n >= 0.99 && maxRelBig < 3e-2,
      `gradcheck 1a (GPU vs float64 CPU ref): ${pass}/${n} params ok (rel<2e-2 or |Δ|<${ATOL_A}), worst non-tiny rel ${maxRelBig.toExponential(2)}`
    );
  }

  // -------------------------------------------------------------------------
  // Check 1b (spec form) — central finite differences THROUGH the GPU forward.
  // -------------------------------------------------------------------------
  // L(params) = <go, image>
  const dot = (img: Float32Array): number => {
    let s = 0;
    for (let i = 0; i < img.length; i++) s += go[i] * img[i];
    return s;
  };
  // probe ~200 distinct random raw-param indices across all 5 groups
  const total = GC.G * PARAM_STRIDE;
  const probeSet = new Set<number>();
  while (probeSet.size < 200) probeSet.add((rng.next() * total) | 0);
  const probes = [...probeSet];

  const work = new Float32Array(base); // scratch we mutate one index at a time
  const Lat = async (idx: number, delta: number): Promise<number> => {
    work[idx] = base[idx] + delta;
    eng.setParams(work);
    eng.runForward();
    const v = dot(await eng.readImage());
    work[idx] = base[idx];
    return v;
  };

  // Richardson finite differences: a smooth function's central difference at
  // eps and eps/2 agree to O(eps^2); an alpha/visibility GATE crossing anywhere
  // in +-eps makes them disagree by O(jump/eps). So |fd(eps)-fd(eps/2)| is an
  // analytic-INDEPENDENT detector for "this probe crosses the hard visibility
  // threshold" (the spec's excludable case). Kept probes use the Richardson
  // extrapolation (4*fd2 - fd1)/3, accurate to O(eps^4).
  // atol floor: a relative tolerance alone is meaningless for near-zero grads
  // (fixed-point + f32 noise ~1e-4). Standard gradcheck practice (PyTorch
  // gradcheck uses atol+rtol): a probe passes if EITHER rel<2e-2 OR |an-fd|<atol.
  const ATOL = 1.5e-4;
  const results: { group: string; ok: boolean; rel: number; excluded: boolean }[] = [];
  for (const idx of probes) {
    const { name, eps } = groupOf(idx, GC.G);
    const Lp1 = await Lat(idx, eps);
    const Lm1 = await Lat(idx, -eps);
    const Lp2 = await Lat(idx, eps / 2);
    const Lm2 = await Lat(idx, -eps / 2);
    const fd1 = (Lp1 - Lm1) / (2 * eps);
    const fd2 = (Lp2 - Lm2) / eps;
    const inst = Math.abs(fd1 - fd2) / (Math.abs(fd1) + Math.abs(fd2) + 1e-7);
    const unstable = inst > 0.02; // Richardson kink detector: crosses a hard gate
    const fd = (4 * fd2 - fd1) / 3;
    const an = analytic[idx];
    const rel = Math.abs(an - fd) / (Math.abs(an) + Math.abs(fd) + 1e-9);
    const ok = rel < 2e-2 || Math.abs(an - fd) < ATOL;
    results.push({ group: name, ok, rel, excluded: unstable });
  }

  const kept = results.filter((r) => !r.excluded);
  const excluded = results.length - kept.length;
  const passed = kept.filter((r) => r.ok).length;
  const passRate = passed / kept.length;
  const rels = kept.map((r) => r.rel).sort((a, b) => a - b);
  const pct = (p: number) => rels[Math.min(rels.length - 1, Math.floor(p * rels.length))];
  const byGroup: Record<string, { n: number; pass: number }> = {};
  for (const r of kept) {
    byGroup[r.group] ??= { n: 0, pass: 0 };
    byGroup[r.group].n++;
    if (r.ok) byGroup[r.group].pass++;
  }
  console.log(`  1b probes=${results.length}  excluded(gate-cross)=${excluded}  kept=${kept.length}`);
  console.log(
    `  1b rel err distribution: median=${pct(0.5).toExponential(2)} p90=${pct(0.9).toExponential(2)} ` +
      `p95=${pct(0.95).toExponential(2)} max=${rels[rels.length - 1].toExponential(2)}`
  );
  console.log(
    `  1b per-group pass: ` +
      Object.entries(byGroup)
        .map(([g, v]) => `${g} ${v.pass}/${v.n}`)
        .join("  ")
  );
  check(
    passRate >= 0.95,
    `gradcheck 1b (FD through GPU forward): ${passed}/${kept.length} (${(passRate * 100).toFixed(1)}%) probes ok (gate >=95%)`
  );
  eng.destroy();
}

// ==========================================================================
// Check 3 — PERF (no gate): 200K splats @256^2, spread random init
// ==========================================================================
{
  const G = 200_000;
  const H = 256;
  const W = 256;
  const eng = await RasterEngine.create(device, { G, H, W, cap: 2048, gradScale: 65536 });
  const rng = makeRng(777);
  const mean = new Float32Array(2 * G);
  const logScale = new Float32Array(2 * G);
  const theta = new Float32Array(G);
  const colorRaw = new Float32Array(3 * G);
  const opacityRaw = new Float32Array(G);
  for (let i = 0; i < G; i++) {
    mean[2 * i] = rng.next() * W;
    mean[2 * i + 1] = rng.next() * H;
    logScale[2 * i] = Math.log(1.5) + 0.3 * rng.normal(); // small splats -> spread out
    logScale[2 * i + 1] = Math.log(1.5) + 0.3 * rng.normal();
    theta[i] = rng.range(0, Math.PI);
    colorRaw[3 * i] = rng.normal();
    colorRaw[3 * i + 1] = rng.normal();
    colorRaw[3 * i + 2] = rng.normal();
    opacityRaw[i] = 0.5 + 0.5 * rng.normal();
  }
  eng.setParams(packParams(G, { mean, logScale, theta, colorRaw, opacityRaw }));
  const go = new Float32Array(3 * H * W);
  for (let i = 0; i < go.length; i++) go[i] = 0.01 * rng.normal();
  eng.setGradImage(go);

  const fence = async () => {
    await eng.readImage(); // small-ish readback drains the queue
  };

  // warmup (Metal JITs each pipeline lazily; shared GPU has +-30% noise)
  for (let i = 0; i < 8; i++) {
    eng.runForward();
    eng.runBackward();
  }
  await fence();

  const TIMED = 30;
  // forward-only
  let t0 = performance.now();
  for (let i = 0; i < TIMED; i++) eng.runForward();
  await fence();
  const fwdMs = (performance.now() - t0) / TIMED;
  // forward + backward
  t0 = performance.now();
  for (let i = 0; i < TIMED; i++) {
    eng.runForward();
    eng.runBackward();
  }
  await fence();
  const fbMs = (performance.now() - t0) / TIMED;

  console.log(
    `\nBENCH  ${G.toLocaleString()} splats @${W}x${H}: forward ${fwdMs.toFixed(2)} ms  |  ` +
      `forward+backward ${fbMs.toFixed(2)} ms  (shared GPU, +-30% noise; hope: fwd+bwd < 15 ms)`
  );
  eng.destroy();
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PASS");
process.exit(failures ? 1 : 0);
