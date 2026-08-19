/**
 * FAMILY-PLANED HASHGRID — the gates for the RGB Families piece.
 * Real Metal via bun-webgpu. GPU suites are SEQUENTIAL — run nothing else
 * on the GPU.
 *
 *   bun tools/family_grid_test.ts
 *
 * The family label reaches the field through the GRID: the table is indexed by
 * (family, y, x) and every cell index carries `cls · gridSize²`. That term is
 * easy to get wrong in a way NOTHING ELSE NOTICES — a missing offset still
 * compiles, still runs, still trains, and quietly collapses the three families
 * onto one shared plane. Aggregate parity numbers hide it (the field is still
 * "a" valid field), so these gates go at the offset directly.
 *
 * Sections (each falsifiable):
 *  §1 κ: `familyRoute` accepts exactly the supported combinations and throws,
 *     by name, on every other one. Includes the invariants the PREVIOUS design
 *     depended on — raw+classes is still the one-hot route, and the fused
 *     adversary still refuses that route.
 *  §2 byte-identity: `planes: 1` (and an encoding with no `planes` key at all)
 *     regenerate the pre-family WGSL CHARACTER FOR CHARACTER, for the advect
 *     kernel, both train passes and both adversary passes. This is what says
 *     the shipped hashgrid pieces did not move.
 *  §3 FD vs the analytic grid gradient (GPU). Perturb one grid float, remeasure
 *     the batch payoff, and compare against `extGrads`. Calibrated by RATIO
 *     against the already-gated MLP slice, so the test never has to re-derive
 *     genSeed's normalization: every weight must share ONE constant
 *     analytic/FD ratio. A wrong plane offset breaks that immediately — the
 *     perturbed float is read by the wrong family's members, or by none.
 *  §4 plane isolation: the analytic gradient of a grid float in plane p must be
 *     EXACTLY zero when no sampled member carries family p. Run three times,
 *     once per family, over a batch restricted to one family.
 */
import { setupGlobals } from "bun-webgpu";
import {
  layoutField,
  familyRoute,
  advectShader,
  CLASS_SALT,
  type Encoding,
  type LayerDims,
} from "../src/render/webgpu/advect_wgsl";
import {
  adversaryPassAShader,
  adversaryPassBShader,
  familyInstrument,
} from "../src/render/webgpu/adversary_wgsl";
import {
  trainPassAShader,
  trainPassBShader,
} from "../src/render/webgpu/train_wgsl";
import { AdversaryTrainer } from "../src/render/webgpu/adversary_train";
import { FusedTrainer } from "../src/render/webgpu/train";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) {
  console.error("FATAL: no WebGPU adapter");
  process.exit(1);
}
const device: any = await adapter.requestDevice();
const mkStorage = (bytes: number) =>
  device.createBuffer({
    size: Math.max(16, bytes),
    usage: 128 /*STORAGE*/ | 8 /*COPY_DST*/ | 4 /*COPY_SRC*/,
  });

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** The family derivation, on the host. MUST match the WGSL pcg + CLASS_SALT. */
function pcg(v: number): number {
  const s = Math.imul(v >>> 0, 747796405) + 2891336453;
  const t = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737);
  return ((t >>> 22) ^ t) >>> 0;
}
const familyOf = (index: number): number => pcg(index ^ CLASS_SALT) % C;

let failures = 0;
function check(name: string, ok: boolean, detail = ""): void {
  if (!ok) failures++;
  console.log(`${ok ? "  ok  " : "FAIL  "}${name}${detail ? ` — ${detail}` : ""}`);
}
function throws(name: string, fn: () => unknown, needle: string): void {
  try {
    fn();
    check(name, false, "did not throw");
  } catch (e) {
    const msg = String((e as Error).message ?? e);
    check(name, msg.includes(needle), msg.includes(needle) ? "" : `message was "${msg}"`);
  }
}

const HID = 8;
const F = 4;
const GS = 8;
const C = 3;
const fieldDims = (inDim: number): LayerDims[] => [
  { inSize: inDim, outSize: HID, activation: "selu" },
  { inSize: HID, outSize: 2, activation: "tanh" },
];
const planedEnc: Encoding = { kind: "hashgrid", gridSize: GS, features: F, planes: C };
const plainEnc: Encoding = { kind: "hashgrid", gridSize: GS, features: F };

// ===========================================================================
console.log("\n§1 κ — familyRoute");
// ===========================================================================
{
  check(
    "classless hashgrid → none",
    familyRoute(0, plainEnc).tag === "none"
  );
  const planed = familyRoute(C, planedEnc);
  check(
    "hashgrid + classes + matching planes → grid-plane",
    planed.tag === "grid-plane" && planed.count === C,
    JSON.stringify(planed)
  );
  const oneHot = familyRoute(C, { kind: "raw" });
  check(
    "raw + classes → onehot (the Species piece is untouched)",
    oneHot.tag === "onehot" && oneHot.count === C,
    JSON.stringify(oneHot)
  );
  throws(
    "hashgrid + classes without planes throws",
    () => familyRoute(C, plainEnc),
    "one feature plane per family"
  );
  throws(
    "planes without classes throws",
    () => familyRoute(0, planedEnc),
    "nothing would ever"
  );
  throws(
    "fourier + classes still throws",
    () => familyRoute(C, { kind: "fourier", octaves: 4 }),
    "fourier + classes"
  );

  const layout = layoutField("helmholtz", [fieldDims(F), fieldDims(F)], {
    classes: C,
    encoding: planedEnc,
  });
  const grid = layout.segments.find((s) => s.role === "grid")!;
  check(
    "grid segment is C planes long",
    grid.floatLength === C * GS * GS * F,
    `${grid.floatLength} vs ${C * GS * GS * F}`
  );
  check(
    "head 1 input is NOT widened by the family (the grid carries it)",
    layout.spec.heads[1].layers[0].inSize === F,
    `inSize ${layout.spec.heads[1].layers[0].inSize}`
  );

  // The instrument is a property of the OBSERVER, not only of the field.
  check(
    "instrument on m=1 → per-family",
    familyInstrument(planed, 1).tag === "per-family"
  );
  check(
    "instrument on m=2 → off (a pair can mix families)",
    familyInstrument(planed, 2).tag === "off"
  );
  check(
    "instrument on a classless field → off",
    familyInstrument({ tag: "none" }, 1).tag === "off"
  );
}

// ===========================================================================
console.log("\n§2 byte-identity — planes:1 regenerates the pre-family shaders");
// ===========================================================================
{
  const withKey: Encoding = { kind: "hashgrid", gridSize: GS, features: F, planes: 1 };
  const a = layoutField("helmholtz", [fieldDims(F), fieldDims(F)], { encoding: plainEnc });
  const b = layoutField("helmholtz", [fieldDims(F), fieldDims(F)], { encoding: withKey });
  check("layout strides agree", a.totalFloats === b.totalFloats);
  check(
    "advect WGSL identical",
    advectShader(a, { stageWeights: true }) === advectShader(b, { stageWeights: true })
  );
  check(
    "train pass A WGSL identical",
    trainPassAShader(a, {}) === trainPassAShader(b, {})
  );
  check(
    "train pass B WGSL identical",
    trainPassBShader(a, { extGrad: true }) === trainPassBShader(b, { extGrad: true })
  );
  const advOpts = {
    tag: "point" as const,
    relaxEps: 0.05,
    observerGeometry: "euclidean" as const,
    target: { tag: "force" as const },
    loss: { tag: "raw-vector" as const },
  };
  const mkAdv = (enc: Encoding) => {
    const layout = layoutField("helmholtz", [fieldDims(F), fieldDims(F)], { encoding: enc });
    const t = new AdversaryTrainer(device, layout, {
      ...advOpts, k: 1, hiddenUnits: HID, featureDim: HID, batchCap: 8,
      fieldWeightsBuffer: mkStorage(layout.totalFloats * 4), seed: 1,
    });
    return {
      a: adversaryPassAShader(layout, t.advL, advOpts),
      b: adversaryPassBShader(layout, t.advL, advOpts),
    };
  };
  const sa = mkAdv(plainEnc);
  const sb = mkAdv(withKey);
  check("adversary pass A WGSL identical", sa.a === sb.a);
  check("adversary pass B WGSL identical", sa.b === sb.b);
  // And the planed shader must actually DIFFER — a no-op plane term would pass
  // every identity check above and gate nothing.
  const planedLayout = layoutField("helmholtz", [fieldDims(F), fieldDims(F)], {
    classes: C, encoding: planedEnc,
  });
  check(
    "planed advect WGSL differs",
    advectShader(planedLayout, { stageWeights: true }) !==
      advectShader(a, { stageWeights: true })
  );
}

// ===========================================================================
// GPU harness — one planed field, one point-observer game.
// ===========================================================================
const PHYS = {
  width: 800, height: 600, forceMagnitude: 3.0, friction: 0.97, maxVelocity: 26,
};
const ALPHA = 0.6;
const B = 64;
const GRID_FLOATS = C * GS * GS * F;

const planedLayout = layoutField("helmholtz", [fieldDims(F), fieldDims(F)], {
  classes: C,
  encoding: planedEnc,
});

function makeFieldWeights(seed: number): Float32Array {
  const rnd = mulberry32(seed);
  const w = new Float32Array(planedLayout.totalFloats);
  for (const seg of planedLayout.segments) {
    // grid ~ U(-0.1, 0.1) (HelmholtzField's init); kernels glorot-ish; biases 0
    const scale = seg.role === "grid" ? 0.1 : seg.role === "kernel" ? 0.4 : 0;
    for (let i = 0; i < seg.floatLength; i++) {
      w[seg.floatOffset + i] = scale === 0 ? 0 : (rnd() * 2 - 1) * scale;
    }
  }
  return w;
}
/** Interior positions only — the clamp borders are already gated by the
 *  classless hashgrid suite, and they would blur the per-plane support sets. */
function makePoints(b: number, seed: number): Float32Array {
  const rnd = mulberry32(seed);
  const t = new Float32Array(b * 4);
  for (let i = 0; i < b; i++) {
    t[i * 4] = Math.fround((0.08 + 0.84 * rnd()) * PHYS.width);
    t[i * 4 + 1] = Math.fround((0.08 + 0.84 * rnd()) * PHYS.height);
  }
  return t;
}
/** The four bilinear corner CELLS (plane-qualified) member `i` reads. */
function cornersOf(tuples: Float32Array, i: number, family: number): number[] {
  const ux = tuples[i * 4] / PHYS.width;
  const uy = tuples[i * 4 + 1] / PHYS.height;
  const gxf = Math.min(Math.max(ux, 0), 1) * (GS - 1);
  const gyf = Math.min(Math.max(uy, 0), 1) * (GS - 1);
  const ix = Math.floor(gxf);
  const iy = Math.floor(gyf);
  const ix1 = Math.min(ix + 1, GS - 1);
  const iy1 = Math.min(iy + 1, GS - 1);
  const plane = family * GS * GS;
  return [
    plane + iy * GS + ix, plane + iy * GS + ix1,
    plane + iy1 * GS + ix, plane + iy1 * GS + ix1,
  ];
}

const ADV_OPTS = {
  tag: "point" as const,
  target: { tag: "force" as const },
  // raw-vector + k=1 keeps the payoff SMOOTH in the weights: no winner index to
  // switch under an FD perturbation, no chord kink. The grid backward is
  // independent of both choices, so this loses no coverage.
  loss: { tag: "raw-vector" as const },
  k: 1,
  relaxEps: 0,
  observerGeometry: "euclidean" as const,
};

const fieldWBuf = mkStorage(planedLayout.totalFloats * 4);
const trainer = new AdversaryTrainer(device, planedLayout, {
  ...ADV_OPTS,
  hiddenUnits: HID,
  featureDim: HID,
  batchCap: 256,
  fieldWeightsBuffer: fieldWBuf,
  seed: 4242,
});
const packed = makeFieldWeights(7001);
const tuples = makePoints(B, 7002);
trainer.uploadTuples(tuples);

async function payoffWith(w: Float32Array): Promise<number> {
  device.queue.writeBuffer(fieldWBuf, 0, w);
  trainer.step(PHYS, {
    b: B, alpha: ALPHA, lr: 0, source: "uploaded", genSeed: 1, applyDisc: false,
  });
  return (await trainer.readStats()).surprise;
}

// ===========================================================================
console.log("\n§3 FD vs the analytic grid gradient");
// ===========================================================================
{
  device.queue.writeBuffer(fieldWBuf, 0, packed);
  trainer.step(PHYS, {
    b: B, alpha: ALPHA, lr: 0, source: "uploaded", genSeed: 1, applyDisc: false,
  });
  const ext = await trainer.readExtGrads();

  // Sample the largest-|g| floats of each slice: FD on a float the batch barely
  // touches is all rounding noise, and would make this gate pass on anything.
  const pick = (lo: number, hi: number, n: number): number[] =>
    Array.from({ length: hi - lo }, (_, i) => lo + i)
      .filter((t) => Math.abs(ext[t]) > 0)
      .sort((a, b) => Math.abs(ext[b]) - Math.abs(ext[a]))
      .slice(0, n);
  const gridIdx = pick(0, GRID_FLOATS, 6);
  const mlpIdx = pick(GRID_FLOATS, planedLayout.totalFloats, 4);
  check("grid floats carry gradient at all", gridIdx.length === 6, `${gridIdx.length}`);
  check("mlp floats carry gradient at all", mlpIdx.length === 4, `${mlpIdx.length}`);

  const H = 2e-3;
  const ratios: { t: number; slice: string; ratio: number; fd: number }[] = [];
  for (const [slice, idx] of [["mlp", mlpIdx], ["grid", gridIdx]] as const) {
    for (const t of idx) {
      const up = packed.slice();
      up[t] += H;
      const dn = packed.slice();
      dn[t] -= H;
      const fd = ((await payoffWith(up)) - (await payoffWith(dn))) / (2 * H);
      ratios.push({ t, slice, ratio: ext[t] / fd, fd });
    }
  }
  const rs = ratios.map((r) => r.ratio);
  const mean = rs.reduce((a, b) => a + b, 0) / rs.length;
  const worst = Math.max(...rs.map((r) => Math.abs(r / mean - 1)));
  for (const r of ratios) {
    console.log(
      `        ${r.slice} t=${r.t} analytic/FD = ${r.ratio.toFixed(5)} ` +
        `(FD ${r.fd.toExponential(2)})`
    );
  }
  // ONE constant relates every weight's analytic gradient to its FD — that
  // constant is genSeed's normalization, which this test deliberately never
  // re-derives. Grid floats sharing it with MLP floats is the whole verdict:
  // a wrong plane offset reads a cell no sampled family touches, so its FD
  // goes to ~0 while its analytic value does not (or vice versa), and the
  // ratio explodes.
  check(
    "analytic/FD is ONE constant across the grid and MLP slices",
    worst < 0.02,
    `worst deviation ${(worst * 100).toFixed(2)}% (mean ratio ${mean.toFixed(4)})`
  );
}

// ===========================================================================
console.log("\n§4 plane isolation");
// ===========================================================================
{
  device.queue.writeBuffer(fieldWBuf, 0, packed);
  trainer.step(PHYS, {
    b: B, alpha: ALPHA, lr: 0, source: "uploaded", genSeed: 1, applyDisc: false,
  });
  const ext = await trainer.readExtGrads();

  // Host-side support: which plane-qualified cells each family's members read.
  const support = new Set<number>();
  const perPlane = new Array<number>(C).fill(0);
  for (let i = 0; i < B; i++) {
    const fam = familyOf(i);
    perPlane[fam]++;
    for (const cell of cornersOf(tuples, i, fam)) support.add(cell);
  }
  check(
    "the batch actually sampled all three families",
    perPlane.every((n) => n > 0),
    perPlane.join("/")
  );

  let outside = 0;
  const nonzeroPerPlane = new Array<number>(C).fill(0);
  for (let cell = 0; cell < C * GS * GS; cell++) {
    let any = false;
    for (let f = 0; f < F; f++) if (ext[cell * F + f] !== 0) any = true;
    if (!any) continue;
    nonzeroPerPlane[Math.floor(cell / (GS * GS))]++;
    if (!support.has(cell)) outside++;
  }
  // Drop the plane term and EVERY family's gradient lands in plane 0: planes 1
  // and 2 go silent and plane 0 collects cells no family-0 member ever read.
  check(
    "no gradient outside the family-qualified support",
    outside === 0,
    `${outside} cells`
  );
  check(
    "every plane received gradient",
    nonzeroPerPlane.every((n) => n > 0),
    nonzeroPerPlane.join("/")
  );
}

// ===========================================================================
console.log("\n§5 end to end — the shipped piece's arch through the real seam");
// ===========================================================================
{
  // The SHIPPED geometry (ARCH.familyHashgrid, K = 2 soft-angle) driven through
  // the same seam main.ts wires: adversary extGrads → the field trainer's Adam.
  // A dropped plane offset survives every unit-level check that looks at ONE
  // pass; what it cannot survive is this — only plane 0 would ever move.
  const HID2 = 32;
  const GS2 = 16;
  const F2 = 6;
  const dims = (inDim: number): LayerDims[] => [
    { inSize: inDim, outSize: HID2, activation: "selu" },
    { inSize: HID2, outSize: HID2, activation: "selu" },
    { inSize: HID2, outSize: 2, activation: "tanh" },
  ];
  const enc: Encoding = { kind: "hashgrid", gridSize: GS2, features: F2, planes: C };
  const layout = layoutField("helmholtz", [dims(F2), dims(F2)], {
    classes: C,
    encoding: enc,
  });
  const planeFloats = GS2 * GS2 * F2;
  const wbuf = mkStorage(layout.totalFloats * 4);
  const rnd = mulberry32(31337);
  const w0 = new Float32Array(layout.totalFloats);
  for (const seg of layout.segments) {
    const scale = seg.role === "grid" ? 0.1 : seg.role === "kernel" ? 0.25 : 0;
    for (let i = 0; i < seg.floatLength; i++) {
      w0[seg.floatOffset + i] = scale === 0 ? 0 : (rnd() * 2 - 1) * scale;
    }
  }
  device.queue.writeBuffer(wbuf, 0, w0);

  const adv = new AdversaryTrainer(device, layout, {
    tag: "point",
    target: { tag: "force" },
    loss: { tag: "soft-angle", tau: 0.15 },
    k: 2,
    relaxEps: 0.05,
    observerGeometry: "periodic",
    pressure: { tag: "anti-collapse", polar: 0.5, nematic: 0.5, tau: 0.15 },
    hiddenUnits: HID2,
    featureDim: HID2,
    batchCap: 256,
    fieldWeightsBuffer: wbuf,
    seed: 606,
  });
  const trainer = new FusedTrainer(device, layout, {
    batchCap: 256,
    weightsBuffer: wbuf,
    extGradBuffer: adv.extGradsBuf,
  });
  const b = 192;
  const pts = makePoints(b, 909);
  adv.uploadTuples(pts);
  const batch = new Float32Array(2 * b);
  for (let i = 0; i < b; i++) {
    batch[2 * i] = pts[i * 4];
    batch[2 * i + 1] = pts[i * 4 + 1];
  }
  trainer.uploadBatch(batch);

  for (let s = 0; s < 60; s++) {
    adv.step(PHYS, {
      b, alpha: ALPHA, lr: 3e-3, source: "uploaded", genSeed: 0.01, applyDisc: true,
    });
    trainer.step(PHYS, {
      n: b, alpha: ALPHA, lr: 1e-3, source: "uploaded", apply: true,
    });
  }

  const w1 = await (async () => {
    const staging = device.createBuffer({ size: layout.totalFloats * 4, usage: 1 | 8 });
    const e = device.createCommandEncoder();
    e.copyBufferToBuffer(wbuf, 0, staging, 0, layout.totalFloats * 4);
    device.queue.submit([e.finish()]);
    await staging.mapAsync(1);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  })();

  check(
    "60 fused steps leave every weight finite",
    w1.every(Number.isFinite),
    `${w1.filter((x) => !Number.isFinite(x)).length} nonfinite`
  );
  const moved = Array.from({ length: C }, (_, p) => {
    let d = 0;
    for (let i = 0; i < planeFloats; i++) {
      d = Math.max(d, Math.abs(w1[p * planeFloats + i] - w0[p * planeFloats + i]));
    }
    return d;
  });
  check(
    "EVERY family plane is trained by the game",
    moved.every((d) => d > 1e-6),
    `max|Δ| per plane ${moved.map((d) => d.toExponential(1)).join(" ")}`
  );

  const stats = await adv.readStats();
  const fam = stats.perFamily;
  check(
    "the per-family instrument is live on the point observer",
    fam.tag === "measured",
    fam.tag
  );
  if (fam.tag === "measured") {
    check(
      "all C families reported, all sampled",
      fam.count.length === C && fam.count.every((n) => n > 0),
      fam.count.join("/")
    );
    check(
      "family tuple counts sum to the batch",
      Math.abs(fam.count.reduce((a, x) => a + x, 0) - b) < 0.5,
      `${fam.count.reduce((a, x) => a + x, 0)} vs ${b}`
    );
    // The per-family means must reconstruct the global mean EXACTLY (it is the
    // same payoff, bucketed) — this is what says the chart and the headline
    // number describe one quantity and not two.
    const recomposed =
      fam.mean.reduce((a, v, c) => a + v * fam.count[c], 0) / b;
    check(
      "Σ familyMean·count / B ≡ the reported surprise",
      Math.abs(recomposed - stats.surprise) < 1e-4 * Math.max(1, Math.abs(stats.surprise)),
      `${recomposed.toExponential(4)} vs ${stats.surprise.toExponential(4)}`
    );
    console.log(
      `        per-family payoff ${fam.mean.map((v) => v.toFixed(3)).join(" ")} ` +
        `(counts ${fam.count.join("/")}), global ${stats.surprise.toFixed(3)}`
    );
  }
}

console.log(
  failures === 0 ? "\nALL FAMILY GATES PASS" : `\n${failures} FAMILY GATE FAILURE(S)`
);
process.exit(failures === 0 ? 0 : 1);
