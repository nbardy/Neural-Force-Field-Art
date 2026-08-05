/**
 * Verification for the "surprise" render mode — the instrument that makes the
 * adversary observable (src/draw/colormap.ts, src/draw/robust_norm.ts,
 * src/renderers.ts, src/render/webgpu/surprise_wgsl.ts).
 *
 *   bun tools/surprise_test.ts          # colormap + normaliser + real-GPU shader
 *   SKIP_GPU=1 bun tools/surprise_test.ts
 *
 * What it checks, and what each check would catch:
 *   1. colormap anchors vs the PUBLISHED matplotlib values — a mistyped
 *      polynomial coefficient yields a map that still looks pretty but is no
 *      longer perceptually uniform, which is invisible by eye.
 *   2. the emitted-WGSL LUT reproduces the exact CPU ramp to <2/255 — the CPU
 *      and GPU renderers must not drift; without this the two paths silently
 *      show different pictures of the same run.
 *   3. every float literal in the generated shader is WGSL-typed f32 — the same
 *      class of latent bug as the f32lit exponent incident (emit_wgsl.ts:16-27):
 *      GPU-only, and no gradient/colour test above it would notice.
 *   4. RobustSpan TRACKS a ×100 drift (a hardcoded range would saturate) and
 *      RESISTS 1e6 outliers (min/max would).
 *   5. ANTI-FAKE-SIGNAL: a collapsed adversary must NOT be rescaled into a full
 *      rainbow. This is the one that keeps the mode honest — variant A is
 *      expected to collapse and the render must show that, not hide it.
 *   6. non-finite surprise (blown-up adversary loss) is dropped and counted,
 *      never sorted into the percentiles.
 *   7. end-to-end on a real Metal adapter: 4 particles with known scalars are
 *      rendered offscreen and the read-back pixels match the CPU colormap.
 *   8. packed surprise planes select independently in both renderer and stats;
 *      the percentile feed reads only the latest fresh rotating window,
 *      handles wrap, deduplicates generations, and resets history on resize or
 *      plane change.
 */
import {
  COLORMAPS,
  rampLUT,
  sampleLUT,
  LUT_STOPS,
  type ColormapName,
  type RGB,
} from "../src/draw/colormap";
import { RobustSpan, SPAN_FLOOR, type Span } from "../src/draw/robust_norm";
import { surprisePointsShader } from "../src/render/webgpu/surprise_wgsl";

let failures = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
  if (!cond) failures++;
};

const NAMES: ColormapName[] = ["inferno", "viridis", "coolwarm"];
const dist = (a: RGB, b: RGB) =>
  Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2]));

// deterministic PRNG — a failure must reproduce
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
/** Box–Muller from a uniform source. */
function gauss(rnd: () => number): number {
  const u = Math.max(1e-12, rnd());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
}

console.log("--- 1. colormap anchors vs published matplotlib/ParaView values ---");
{
  // External references, not restatements of the implementation: matplotlib's
  // viridis/inferno endpoints and midpoint, and Moreland's cool-warm centre.
  const anchors: [ColormapName, number, RGB, number][] = [
    ["viridis", 0.0, [68, 1, 84], 5],
    ["viridis", 0.5, [33, 145, 140], 6],
    ["viridis", 1.0, [253, 231, 37], 6],
    ["inferno", 0.0, [0, 0, 4], 5],
    ["inferno", 0.5, [188, 55, 84], 6],
    ["inferno", 1.0, [252, 255, 164], 6],
    ["coolwarm", 0.0, [59, 76, 192], 1],
    ["coolwarm", 0.5, [221, 221, 221], 1],
    ["coolwarm", 1.0, [180, 4, 38], 1],
  ];
  for (const [name, t, want, tol] of anchors) {
    const got = COLORMAPS[name].ramp(t);
    ok(
      dist(got, want) <= tol,
      `${name}(${t}) = [${got}] ≈ [${want}] (Δ${dist(got, want)} ≤ ${tol})`
    );
  }
}

console.log("\n--- 2. ramps stay in gamut and are monotone in luminance ---");
{
  for (const name of NAMES) {
    const cm = COLORMAPS[name];
    let inRange = true;
    for (let i = 0; i <= 1000; i++) {
      const c = cm.ramp(i / 1000);
      for (const ch of c) {
        if (!(Number.isInteger(ch) && ch >= 0 && ch <= 255)) inRange = false;
      }
    }
    // The polynomial fits overshoot near the endpoints; the clamp inside `byte`
    // is what keeps them legal. Deleting it produces >255 and wrapped colours.
    ok(inRange, `${name}: all 1001 samples are integers in [0,255]`);
  }
  // Sequential maps must be monotone in luminance or they invent structure that
  // is not in the data — the entire reason not to use hsv/jet here.
  const lum = (c: RGB) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  for (const name of ["viridis", "inferno"] as ColormapName[]) {
    const cm = COLORMAPS[name];
    let worst = 0;
    for (let i = 1; i <= 256; i++) {
      const d = lum(cm.ramp(i / 256)) - lum(cm.ramp((i - 1) / 256));
      if (d < worst) worst = d;
    }
    ok(worst > -1.0, `${name}: luminance monotone increasing (worst step ${worst.toFixed(3)})`);
  }
}

console.log("\n--- 3. emitted LUT reproduces the exact CPU ramp (CPU↔GPU agreement) ---");
{
  for (const name of NAMES) {
    const cm = COLORMAPS[name];
    const lut = rampLUT(cm, LUT_STOPS);
    let worst = 0;
    let worstAt = 0;
    let sum = 0;
    const S = 2000;
    for (let i = 0; i <= S; i++) {
      const t = i / S;
      const d = dist(sampleLUT(lut, t), cm.ramp(t));
      sum += d;
      if (d > worst) {
        worst = d;
        worstAt = t;
      }
    }
    const mean = sum / (S + 1);
    // MEAN is the real gate — it is below 8-bit quantisation, so the two paths
    // are visually identical. MAX is capped at 4 because inferno's fit dips
    // negative in blue near t=0 and the exact ramp clamps there; that kink is
    // #000004-vs-#000407 and is NOT reduced by more stops (measured at 65/129).
    ok(mean < 1.0, `${name}: ${LUT_STOPS}-stop LUT mean error ${mean.toFixed(3)}/255 vs the exact ramp`);
    ok(
      worst <= 4,
      `${name}: worst-case LUT error ${worst}/255 (at t=${worstAt.toFixed(3)})`
    );
  }
}

console.log("\n--- 4. generated WGSL is well-formed f32 ---");
{
  for (const name of NAMES) {
    const src = surprisePointsShader(name);
    ok(!/NaN|Infinity|undefined/.test(src), `${name}: shader has no NaN/Infinity/undefined`);
    ok(
      src.includes(COLORMAPS[name].wgslPosition),
      `${name}: shader embeds the colormap's own position() expression`
    );
    // Every LUT component must be a float literal, not an i32 — the f32lit trap.
    const lits = src.match(/vec3f\(([^)]*)\)/g) ?? [];
    let allTyped = true;
    for (const l of lits) {
      for (const part of l.slice(6, -1).split(",")) {
        const s = part.trim();
        if (!/[.eE]/.test(s)) allTyped = false;
      }
    }
    ok(allTyped && lits.length >= LUT_STOPS, `${name}: all ${lits.length} vec3f components are f32-typed`);

    // and they must round-trip to the TS LUT — a truncated literal would shift
    // the whole ramp on the GPU only.
    const lut = rampLUT(COLORMAPS[name], LUT_STOPS);
    const nums: number[] = [];
    for (const l of lits.slice(0, LUT_STOPS)) {
      for (const part of l.slice(6, -1).split(",")) nums.push(Number(part.trim()));
    }
    let worst = 0;
    for (let i = 0; i < lut.length; i++) worst = Math.max(worst, Math.abs(nums[i] - lut[i]));
    ok(worst < 1e-6, `${name}: emitted literals round-trip to the TS LUT (worst ${worst.toExponential(2)})`);
  }
}

console.log("\n--- 5. RobustSpan tracks drift a fixed range would miss ---");
{
  // Surprise scale ramps ×100 over 400 frames (untrained → fitted adversary).
  const rnd = mulberry32(0xc0ffee);
  const rs = new RobustSpan();
  const first: number[] = [];
  const last: number[] = [];
  for (let f = 0; f < 400; f++) {
    const sigma = 0.01 * Math.pow(100, f / 399);
    const v = new Float32Array(2048);
    for (let i = 0; i < v.length; i++) v[i] = Math.abs(gauss(rnd)) * sigma;
    rs.update(v, v.length);
    if (f === 20) first.push(rs.span.hi);
    if (f === 399) last.push(rs.span.hi);
  }
  const growth = last[0] / first[0];
  ok(growth > 40, `span.hi grew ${growth.toFixed(1)}× with a 100× scale ramp (tracks drift)`);
  // and the final estimate is近 the true P98 of |N(0,1)| = 2.326σ at σ=1.
  const relErr = Math.abs(rs.span.hi - 2.326) / 2.326;
  ok(relErr < 0.15, `final span.hi = ${rs.span.hi.toFixed(3)} ≈ P98(|N(0,1)|) = 2.326 (rel ${relErr.toFixed(3)})`);
}

console.log("\n--- 6. percentiles resist outliers that min/max cannot ---");
{
  const rnd = mulberry32(7);
  const v = new Float32Array(1024);
  for (let i = 0; i < v.length; i++) v[i] = rnd();
  for (let i = 0; i < 5; i++) v[i * 37] = 1e6; // 0.5% singular particles
  const rs = new RobustSpan();
  rs.update(v, v.length);
  let max = 0;
  for (const x of v) max = Math.max(max, x);
  ok(max >= 1e6, `raw max is ${max.toExponential(1)} (what a min/max normaliser would use)`);
  ok(rs.span.hi < 2, `P98 = ${rs.span.hi.toFixed(3)} — bulk preserved, outliers ignored`);
}

console.log("\n--- 7. ANTI-FAKE-SIGNAL: a collapsed adversary is not rescaled ---");
{
  // Variant A is EXPECTED to collapse: residuals go to ~0 and the only thing
  // left is float noise. A spread-normalising renderer would paint that noise
  // across the whole ramp and fake a working adversary.
  const rnd = mulberry32(99);
  const rs = new RobustSpan();
  for (let f = 0; f < 60; f++) {
    const v = new Float32Array(1024);
    for (let i = 0; i < v.length; i++) v[i] = 1e-9 * rnd();
    rs.update(v, v.length);
  }
  ok(rs.collapsed, `collapsed flag set (span ${(rs.span.hi - rs.span.lo).toExponential(2)} < floor ${SPAN_FLOOR})`);

  const cm = COLORMAPS.inferno;
  let tmin = 1;
  let tmax = 0;
  for (let i = 0; i < 1024; i++) {
    const t = cm.position(rs.span, 1e-9 * rnd());
    tmin = Math.min(tmin, t);
    tmax = Math.max(tmax, t);
  }
  ok(tmax - tmin < 0.02, `ramp positions span only ${(tmax - tmin).toExponential(2)} — reads as collapsed, not as a rainbow`);

  // Sanity counterweight: a genuinely spread channel DOES use the ramp, so the
  // floor is not just clamping everything to one colour.
  const live = new RobustSpan();
  const lv = new Float32Array(1024);
  for (let i = 0; i < lv.length; i++) lv[i] = rnd();
  live.update(lv, lv.length);
  let lmin = 1;
  let lmax = 0;
  for (let i = 0; i < 1024; i++) {
    const t = cm.position(live.span, rnd());
    lmin = Math.min(lmin, t);
    lmax = Math.max(lmax, t);
  }
  ok(lmax - lmin > 0.9, `a live channel still spans the ramp (${(lmax - lmin).toFixed(3)})`);
}

console.log("\n--- 8. non-finite surprise is dropped and counted ---");
{
  const rnd = mulberry32(1234);
  const rs = new RobustSpan();
  const clean = new Float32Array(1024);
  for (let i = 0; i < clean.length; i++) clean[i] = rnd();
  rs.update(clean, clean.length);
  const before: Span = { ...rs.span };

  const dirty = new Float32Array(1024);
  for (let i = 0; i < dirty.length; i++) dirty[i] = i % 2 === 0 ? NaN : Infinity;
  rs.update(dirty, dirty.length);
  ok(rs.rejected === 1024, `all ${rs.rejected} non-finite values rejected`);
  ok(
    Number.isFinite(rs.span.lo) && rs.span.lo === before.lo && rs.span.hi === before.hi,
    "span unchanged by an all-NaN frame (no new information, nothing invented)"
  );

  // A partially poisoned frame must still use its finite half rather than
  // sorting NaNs (Float64Array.sort would push them to the end and corrupt P98).
  const half = new Float32Array(1024);
  for (let i = 0; i < half.length; i++) half[i] = i < 512 ? 0.5 : NaN;
  const rs2 = new RobustSpan({ lambda: 1 });
  rs2.update(half, half.length);
  ok(
    Math.abs(rs2.span.hi - 0.5) < 1e-6,
    `partially-NaN frame yields P98 = ${rs2.span.hi} from the finite half`
  );
}

console.log("\n--- 9. SurpriseRenderer end-to-end on a stub 2D context ---");
{
  // Integration over the real renderer (no mocks of the colour path): the only
  // stub is the canvas boundary itself, and we assert on the COLOURS it was
  // asked to fill, not on which methods were called.
  const { createRenderer } = await import("../src/renderers");
  type Call = { style: string; r: number };
  const calls: Call[] = [];
  let style = "";
  const ctx = {
    set fillStyle(v: string) {
      style = v;
    },
    get fillStyle() {
      return style;
    },
    font: "",
    textAlign: "left",
    globalAlpha: 1,
    fillRect: () => {},
    fillText: () => {},
    beginPath: () => {},
    stroke: () => {},
    moveTo: () => {},
    lineTo: () => {},
    strokeStyle: "",
    lineWidth: 1,
    arc: (_x: number, _y: number, r: number) => calls.push({ style, r }),
    fill: () => {},
  } as unknown as CanvasRenderingContext2D;

  const cfg = {
    name: "surprise-test",
    backgroundColor: [0, 0, 0] as [number, number, number],
  } as any;

  const N = 512;
  const positions = Array.from({ length: N }, (_, i) => [i % 64, Math.floor(i / 64)]);
  const rnd = mulberry32(2024);

  // (a) a genuinely multimodal surprise channel — the variant-B signature.
  const live = new Float32Array(N);
  for (let i = 0; i < N; i++) live[i] = i < N / 2 ? 0.05 * rnd() : 1 + rnd();
  const rLive = createRenderer("surprise", cfg, N, undefined, {
    colormap: "inferno",
    source: { read: () => live },
  });
  calls.length = 0;
  for (let f = 0; f < 30; f++) rLive.render(ctx, 64, 8, positions, positions, f);
  const liveStyles = new Set(calls.map((c) => c.style));
  ok(liveStyles.size > 20, `bimodal channel drew ${liveStyles.size} distinct colours`);

  // (b) the collapsed channel — same renderer, same code path, must NOT bloom.
  const dead = new Float32Array(N);
  for (let i = 0; i < N; i++) dead[i] = 1e-9 * rnd();
  const rDead = createRenderer("surprise", cfg, N, undefined, {
    colormap: "inferno",
    source: { read: () => dead },
  });
  calls.length = 0;
  for (let f = 0; f < 30; f++) rDead.render(ctx, 64, 8, positions, positions, f);
  const deadStyles = new Set(calls.map((c) => c.style));
  ok(
    deadStyles.size <= 4,
    `collapsed channel drew only ${deadStyles.size} distinct colours (no fake rainbow)`
  );

  // (c) BOUNDARY DEFENCE: a blown-up adversary (NaN residual) must be VISIBLE.
  //     Canvas2D silently ignores `fillStyle = undefined`, so an out-of-range
  //     ramp index would leave those particles wearing the previous particle's
  //     colour and a diverged run would render as a healthy one.
  const poisoned = new Float32Array(N);
  for (let i = 0; i < N; i++) poisoned[i] = i % 7 === 0 ? NaN : rnd();
  const rNaN = createRenderer("surprise", cfg, N, undefined, {
    colormap: "viridis",
    source: { read: () => poisoned },
  });
  calls.length = 0;
  rNaN.render(ctx, 64, 8, positions, positions, 1);
  const styles = calls.map((c) => c.style);
  ok(
    styles.every((s) => typeof s === "string" && s.length > 0),
    "no particle was assigned an undefined fillStyle"
  );
  ok(
    styles.some((s) => s === "rgb(255,0,255)"),
    "NaN surprise paints the reserved magenta, not a ramp colour"
  );

  // (c2) a channel shorter than the particle count is a plumbing bug, not a
  //      colour to invent.
  let short = false;
  try {
    createRenderer("surprise", cfg, N, undefined, {
      colormap: "viridis",
      source: { read: () => new Float32Array(4) },
    }).render(ctx, 64, 8, positions, positions, 1);
  } catch (_) {
    short = true;
  }
  ok(short, "a SurpriseSource shorter than the particle count throws");

  // (d) the typed error at the ingestion boundary — no silent fallback to
  //     another renderer when a piece declares "surprise" with no channel.
  let threw = false;
  try {
    createRenderer("surprise", cfg, N);
  } catch (_) {
    threw = true;
  }
  ok(threw, 'createRenderer("surprise") without a source throws instead of falling back');

  // (e) the three pre-existing renderers still construct and draw.
  for (const t of ["alpha-fade", "trail-buffer", "clean"] as const) {
    calls.length = 0;
    const r = createRenderer(t, cfg, N);
    r.render(ctx, 64, 8, positions, positions, 1);
    ok(calls.length === 2 * N, `${t} still draws ${calls.length / N} arcs per particle`);
    r.destroy();
  }
}

// ---------------------------------------------------------------------------
// 10. real GPU: compile the generated shader on the actual adapter and verify
//     the rendered pixels equal the CPU colormap. This is the only check that
//     can catch a WGSL construct that typechecks in my head but not in Tint.
// ---------------------------------------------------------------------------
if (process.env.SKIP_GPU === "1") {
  console.log("\n--- 10. real-GPU shader check SKIPPED (SKIP_GPU=1) ---");
} else {
  console.log("\n--- 10. real-GPU: generated shader compiles and colours correctly ---");
  const { setupGlobals } = await import("bun-webgpu");
  setupGlobals();
  (globalThis as any).GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128,
  };
  (globalThis as any).GPUTextureUsage ??= {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16,
  };
  (globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };
  const USAGE = (globalThis as any).GPUBufferUsage;
  const TEX = (globalThis as any).GPUTextureUsage;

  const adapter = await (navigator as any).gpu.requestAdapter();
  if (!adapter) {
    console.error("FATAL: no WebGPU adapter (bun-webgpu found no GPU)");
    process.exit(1);
  }
  const device: any = await adapter.requestDevice();
  const { renderPipeline, uniformBuffer, bindGroup, BLEND_ADD } = await import(
    "../src/render/webgpu/microgpu"
  );
  const { GpuSurpriseStats } = await import("../src/render/webgpu/surprise_points");

  console.log("\n--- 10a. real-GPU fresh-window stats: wrap, dedupe, reset ---");
  {
    const N = 10;
    // Latest window is indices [8,9,0,1,2]. Everything else is stale poison
    // that would dominate the old fixed-prefix estimator.
    const values = new Float32Array([
      // plane 0
      10, 11, 12, 1e9, 1e9, 1e9, 1e9, 1e9, 8, 9,
      // plane 1 at offset N
      110, 111, 112, 1e9, 1e9, 1e9, 1e9, 1e9, 108, 109,
    ]);
    const surBuf = device.createBuffer({
      size: values.byteLength,
      usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
    });
    device.queue.writeBuffer(surBuf, 0, values);
    const stats = new GpuSurpriseStats(device, { sample: 5, every: 1 });
    const enc = device.createCommandEncoder();
    const recorded = stats.encodeSample(
      enc,
      surBuf,
      N,
      0,
      { start: 8, count: 5, generation: 1 }
    );
    device.queue.submit([enc.finish()]);
    stats.afterSubmit(recorded);
    for (let i = 0; i < 100 && stats.norm.updates === 0; i++) {
      await new Promise((resolve) => setTimeout(resolve, 2));
    }
    ok(
      recorded &&
        stats.norm.updates === 1 &&
        stats.norm.raw.lo >= 8 &&
        stats.norm.raw.hi < 20,
      `wrapped fresh window excludes stale poison (p2=${stats.norm.raw.lo.toFixed(2)}, ` +
        `p98=${stats.norm.raw.hi.toFixed(2)})`
    );
    const duplicateEncoder = device.createCommandEncoder();
    ok(
      !stats.encodeSample(
        duplicateEncoder,
        surBuf,
        N,
        1,
        { start: 8, count: 5, generation: 1 }
      ),
      "duplicate coverage generation is not folded twice"
    );

    stats.reset();
    ok(
      stats.norm.updates === 0 &&
        stats.norm.raw.lo === 0 &&
        stats.norm.raw.mid === 0 &&
        stats.norm.raw.hi === 0,
      "stats reset drops the previous particle-buffer epoch"
    );
    const enc2 = device.createCommandEncoder();
    const rerecorded = stats.encodeSample(
      enc2,
      surBuf,
      N,
      2,
      { start: 8, count: 5, generation: 1 },
      N
    );
    device.queue.submit([enc2.finish()]);
    stats.afterSubmit(rerecorded);
    for (let i = 0; i < 100 && stats.norm.updates === 0; i++) {
      await new Promise((resolve) => setTimeout(resolve, 2));
    }
    ok(
      rerecorded &&
        stats.norm.updates === 1 &&
        stats.norm.raw.lo >= 108 &&
        stats.norm.raw.hi < 120,
      "nonzero plane offset applies to both halves of a wrapped fresh window"
    );

    // Same generation, different plane: the plane-history gate resets the EMA
    // before generation deduplication, so offset 0 is accepted and cannot
    // inherit plane 1's ~100-unit normalization.
    const enc3 = device.createCommandEncoder();
    const switched = stats.encodeSample(
      enc3,
      surBuf,
      N,
      3,
      { start: 8, count: 5, generation: 1 },
      0
    );
    device.queue.submit([enc3.finish()]);
    stats.afterSubmit(switched);
    for (let i = 0; i < 100 && stats.norm.updates === 0; i++) {
      await new Promise((resolve) => setTimeout(resolve, 2));
    }
    ok(
      switched &&
        stats.norm.updates === 1 &&
        stats.norm.raw.lo >= 8 &&
        stats.norm.raw.hi < 20,
      "changing packed planes resets normalization history before folding"
    );
    stats.destroy();
    surBuf.destroy();
  }

  const W = 64;
  const H = 64;
  const span: Span = { lo: 0, mid: 0.5, hi: 1 };
  const vals = [0, 0.25, 0.6, 1.0];
  const xs = [8, 24, 40, 56];

  for (const name of NAMES) {
    const pipeline = renderPipeline(device, {
      code: surprisePointsShader(name),
      format: "rgba8unorm",
      blend: BLEND_ADD,
      topology: "triangle-strip",
    });

    const pos = new Float32Array(xs.length * 2);
    for (let i = 0; i < xs.length; i++) {
      pos[i * 2] = xs[i];
      pos[i * 2 + 1] = 32;
    }
    const posBuf = device.createBuffer({ size: pos.byteLength, usage: USAGE.STORAGE | USAGE.COPY_DST });
    device.queue.writeBuffer(posBuf, 0, pos);
    const sur = new Float32Array(vals);
    const surBuf = device.createBuffer({ size: sur.byteLength, usage: USAGE.STORAGE | USAGE.COPY_DST });
    device.queue.writeBuffer(surBuf, 0, sur);

    const uni = uniformBuffer(device, 32);
    const uf = new Float32Array(8);
    uf[0] = W; uf[1] = H; uf[2] = 10; uf[3] = span.lo; uf[4] = span.mid; uf[5] = span.hi; uf[6] = 1;
    device.queue.writeBuffer(uni, 0, uf);

    const group = bindGroup(device, pipeline, [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: { buffer: posBuf } },
      { binding: 2, resource: { buffer: surBuf } },
    ]);

    const tex = device.createTexture({
      size: { width: W, height: H },
      format: "rgba8unorm",
      usage: TEX.RENDER_ATTACHMENT | TEX.COPY_SRC,
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: tex.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(4, xs.length);
    pass.end();

    const bpr = Math.ceil((W * 4) / 256) * 256;
    const staging = device.createBuffer({ size: bpr * H, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    enc.copyTextureToBuffer(
      { texture: tex },
      { buffer: staging, bytesPerRow: bpr, rowsPerImage: H },
      { width: W, height: H, depthOrArrayLayers: 1 }
    );
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const img = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();

    const cm = COLORMAPS[name];
    const lut = rampLUT(cm, LUT_STOPS);
    let worst = 0;
    for (let i = 0; i < xs.length; i++) {
      const o = 32 * bpr + xs[i] * 4;
      const got: RGB = [img[o], img[o + 1], img[o + 2]];
      const want = sampleLUT(lut, cm.position(span, vals[i]));
      worst = Math.max(worst, dist(got, want));
    }
    ok(worst <= 3, `${name}: GPU pixels match the CPU colormap (worst Δ${worst}/255)`);

    staging.destroy();
    tex.destroy();
    posBuf.destroy();
    surBuf.destroy();
    uni.destroy();
  }

  console.log("\n--- 10b. real-GPU packed surprise plane selection ---");
  {
    const name: ColormapName = "inferno";
    const pipeline = renderPipeline(device, {
      code: surprisePointsShader(name),
      format: "rgba8unorm",
      blend: BLEND_ADD,
      topology: "triangle-strip",
    });
    const pos = new Float32Array(xs.length * 2);
    for (let i = 0; i < xs.length; i++) {
      pos[2 * i] = xs[i];
      pos[2 * i + 1] = 32;
    }
    const plane0 = vals;
    const plane1 = [...vals].reverse();
    const packed = new Float32Array([...plane0, ...plane1]);
    const posBuf = device.createBuffer({
      size: pos.byteLength,
      usage: USAGE.STORAGE | USAGE.COPY_DST,
    });
    const surBuf = device.createBuffer({
      size: packed.byteLength,
      usage: USAGE.STORAGE | USAGE.COPY_DST,
    });
    device.queue.writeBuffer(posBuf, 0, pos);
    device.queue.writeBuffer(surBuf, 0, packed);

    const makeUniform = (offsetFloats: number) => {
      const uni = uniformBuffer(device, 32);
      const data = new ArrayBuffer(32);
      const f = new Float32Array(data);
      const u = new Uint32Array(data);
      f[0] = W; f[1] = H; f[2] = 10;
      f[3] = span.lo; f[4] = span.mid; f[5] = span.hi; f[6] = 1;
      u[7] = offsetFloats;
      device.queue.writeBuffer(uni, 0, data);
      return uni;
    };
    const uni0 = makeUniform(0);
    const uni1 = makeUniform(xs.length);
    const group0 = bindGroup(device, pipeline, [
      { binding: 0, resource: { buffer: uni0 } },
      { binding: 1, resource: { buffer: posBuf } },
      { binding: 2, resource: { buffer: surBuf } },
    ]);
    const group1 = bindGroup(device, pipeline, [
      { binding: 0, resource: { buffer: uni1 } },
      { binding: 1, resource: { buffer: posBuf } },
      { binding: 2, resource: { buffer: surBuf } },
    ]);
    const makeTexture = () => device.createTexture({
      size: { width: W, height: H },
      format: "rgba8unorm",
      usage: TEX.RENDER_ATTACHMENT | TEX.COPY_SRC,
    });
    const tex0 = makeTexture();
    const tex1 = makeTexture();
    const bpr = Math.ceil((W * 4) / 256) * 256;
    const stage0 = device.createBuffer({
      size: bpr * H,
      usage: USAGE.MAP_READ | USAGE.COPY_DST,
    });
    const stage1 = device.createBuffer({
      size: bpr * H,
      usage: USAGE.MAP_READ | USAGE.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    for (const [tex, group] of [[tex0, group0], [tex1, group1]] as const) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: tex.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(4, xs.length);
      pass.end();
    }
    encoder.copyTextureToBuffer(
      { texture: tex0 },
      { buffer: stage0, bytesPerRow: bpr, rowsPerImage: H },
      { width: W, height: H, depthOrArrayLayers: 1 }
    );
    encoder.copyTextureToBuffer(
      { texture: tex1 },
      { buffer: stage1, bytesPerRow: bpr, rowsPerImage: H },
      { width: W, height: H, depthOrArrayLayers: 1 }
    );
    device.queue.submit([encoder.finish()]);
    await Promise.all([stage0.mapAsync(1), stage1.mapAsync(1)]);
    const image0 = new Uint8Array(stage0.getMappedRange().slice(0));
    const image1 = new Uint8Array(stage1.getMappedRange().slice(0));
    const cm = COLORMAPS[name];
    const lut = rampLUT(cm, LUT_STOPS);
    const worstFor = (image: Uint8Array, values: number[]) => {
      let worst = 0;
      for (let i = 0; i < xs.length; i++) {
        const o = 32 * bpr + xs[i] * 4;
        const got: RGB = [image[o], image[o + 1], image[o + 2]];
        const want = sampleLUT(lut, cm.position(span, values[i]));
        worst = Math.max(worst, dist(got, want));
      }
      return worst;
    };
    const worst0 = worstFor(image0, plane0);
    const worst1 = worstFor(image1, plane1);
    ok(
      worst0 <= 3 && worst1 <= 3,
      `offset 0 and offset N select distinct packed planes (worst Δ${worst0}/Δ${worst1})`
    );
    stage0.unmap();
    stage1.unmap();
    for (const b of [stage0, stage1, uni0, uni1, posBuf, surBuf]) b.destroy();
    tex0.destroy();
    tex1.destroy();
  }
}

console.log(
  failures === 0 ? "\nALL SURPRISE CHECKS PASS" : `\n${failures} SURPRISE CHECK(S) FAILED`
);
process.exit(failures === 0 ? 0 : 1);
