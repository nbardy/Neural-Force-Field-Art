/**
 * HEALTH AUDIT — headless, on the REAL adapter, gated on EXACT floats.
 *
 *   node tools/health_audit.mjs [pieceKeys|all|adversary] [baseURL] [sec] [sampleSec]
 *   node tools/health_audit.mjs hashgrid,struct http://localhost:8811/index.html 60 2
 *   node tools/health_audit.mjs --self-test
 *
 * Loads each gallery piece in a fresh page, samples `window.__nffHealth` every
 * `sampleSec`, writes a per-piece time series, and prints one verdict per piece.
 * EXIT CODE = the number of unhealthy pieces.
 *
 * TWO RULES THIS FILE EXISTS TO ENFORCE
 *
 * 1. **Never parse the HUD.** Every previous headless gate here regexed numbers
 *    out of `[data-testid="fps-hud"]`, and the 2026-08-17 soak flake
 *    (`agent_notes/2026-08-17_soak_flake_attribution.md`) was the bill for it: a
 *    `toExponential(2)` rounding a value across a threshold is indistinguishable
 *    from the artwork changing. Everything below reads `window.__nffHealth`,
 *    which carries raw f64 — see `src/health.ts`.
 *
 * 2. **`tools/smoke.mjs` is unusable on an Apple box.** It forces a SOFTWARE
 *    fallback adapter that does not exist there, so the page correctly shows the
 *    "needs WebGPU" notice and nothing renders. The flags below are the ones
 *    `soak_adversary.mjs` and `pressure_live_probe.mjs` already use to get a
 *    real Metal adapter.
 *
 * VERDICTS are a typed sum with one handler each; see `classify` (κ, ordered
 * because the failures NEST — a NaN field is also a dead field, and reporting
 * the second would send the reader to the wrong place) and `describe`.
 */
import puppeteer from "puppeteer";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DEFAULT_BASE = "http://localhost:8811/index.html";

/**
 * Short key → gallery piece NAME. Name is the identity convention this repo
 * commits to (see `resolveSharedPiece` in src/share.ts); indices move.
 */
export const PIECES = Object.freeze({
  single: "Adversary · Single (control)",
  wta8: "Adversary · WTA K=8",
  pair4: "Adversary · Pair WTA K=4",
  tri6: "Adversary · Tri WTA K=6",
  quad6: "Adversary · Quad WTA K=6",
  agree: "Adversary · Agree + Disagree RGB",
  weave: "Adversary · Chaos Weave",
  hashgrid: "Adversary · Pair · HashGrid · Curl",
  families: "Adversary · RGB Families · HashGrid",
  vecfield: "Pixel · VecField",
  nextframe: "Pixel · NextFrame",
  realfake: "Pixel · RealFake",
  inpaint: "Pixel · Inpaint",
  chaos: "Neural Field · Max Chaos",
  species: "Neural Field · Species",
  struct: "Neural Field · Max Structure",
});

/**
 * `all` = EVERY piece in the gallery. It used to be "every adversary piece plus
 * ONE pixel critic", which is how the 2026-08-18 baseline audited 10 of 16 and
 * left the RGB-families adversary and three of the four pixel critics with no
 * headless coverage at all. A partial `all` is worse than an honest subset: it
 * reads as "the gallery is green".
 *
 * Selection is by NAME (see the gallery click below), so a key costs nothing
 * but wall-clock — keep this map in sync with GALLERY in src/main.ts.
 */
const ALL_KEYS = Object.keys(PIECES);
const ADVERSARY_KEYS = ALL_KEYS.filter((k) => PIECES[k].startsWith("Adversary"));
const PIXEL_KEYS = ALL_KEYS.filter((k) => PIECES[k].startsWith("Pixel"));

/**
 * Gate thresholds. Every one is a MEASURED number from this repo's notes, not a
 * guess; the source is named so a future tuner knows what would invalidate it.
 */
export const GATES = Object.freeze({
  /** collapse note §5: R₁ 0.88 → 0.999 is the laminar route; 0.01–0.10 healthy.
   *  0.5 is the midpoint — well clear of both measured regimes. */
  r1Laminar: numberEnv("HEALTH_R1_MAX", 0.5),
  /** R₂ (nematic order, same τ) is the ESCAPE ROUTE from the R₁ gate above and
   *  has to be gated separately: a ±F₀ counter-streaming field — half the
   *  domain flowing +x, half −x — has mean direction ZERO, so it scores R₁ ≈ 0
   *  while looking exactly as single-axis as a laminar field does. Measured
   *  2026-08-25: "Neural Field · Max Structure" passed the R₁ gate at 0.034
   *  with R₂ 0.927, and Tri WTA K=6 at R₁ 0.337 / R₂ 0.613. Healthy pieces in
   *  that same run sat at R₂ 0.075–0.198, so 0.5 separates the two regimes with
   *  the same margin r1Laminar has. */
  r2Nematic: numberEnv("HEALTH_R2_MAX", 0.5),
  /** collapse note: sat 0 → 0.46 on the collapsed point observer. */
  satFrozen: numberEnv("HEALTH_SAT_MAX", 0.3),
  /** collapse note: AC = 0.0007 caught the collapse in the act at step 800. */
  acDead: numberEnv("HEALTH_AC_DEAD", 1e-4),
  /** pressure note §5: the payoff parks at √2 under soft-angle. AMBIGUOUS on
   *  its own — √2 with R₁→1 is the pole exploit, √2 with R₁→0 is a healthy
   *  transient while D relearns. Both halves are required below. */
  poleChord: Math.SQRT2,
  poleBand: numberEnv("HEALTH_POLE_BAND", 0.05),
  /** Divergence caught before it becomes NaN. The shared payoff sits at
   *  0.48–0.7 in steady state, so 1e3 is three orders clear of normal. */
  payoffMax: numberEnv("HEALTH_PAYOFF_MAX", 1e3),
  batchRmsMax: numberEnv("HEALTH_RMS_MAX", 1e4),
  extGradMax: numberEnv("HEALTH_EXTGRAD_MAX", 1e6),
  /** Every shipped piece measures 60 fps on an M4; 30 is "something regressed". */
  fpsFloor: numberEnv("HEALTH_FPS_FLOOR", 30),
  /** Relative AC slope per second, above which a trend is called rising. */
  trendBand: numberEnv("HEALTH_TREND_BAND", 0.002),
  /** Samples to drop before judging — the field probe and the predictor both
   *  need a warmup, and the pressure note measured a ~25 s √2 transient on the
   *  DEFAULT piece that is the game working, not failing. */
  warmupSamples: numberEnv("HEALTH_WARMUP_SAMPLES", 3),
});

function numberEnv(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite, got ${JSON.stringify(raw)}`);
  }
  return value;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ── pure statistics (self-testable without a GPU) ───────────────────────── */

export function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Ordinary-least-squares slope of `ys` against `ts`, in units of y per second.
 * Used for the AC convergence trend — a single AC number cannot distinguish
 * "converged low" from "on its way down".
 */
export function slope(ts, ys) {
  const pairs = ts.map((t, i) => [t, ys[i]]).filter(([t, y]) =>
    Number.isFinite(t) && Number.isFinite(y)
  );
  if (pairs.length < 3) return NaN;
  const n = pairs.length;
  const mt = pairs.reduce((a, [t]) => a + t, 0) / n;
  const my = pairs.reduce((a, [, y]) => a + y, 0) / n;
  let num = 0;
  let den = 0;
  for (const [t, y] of pairs) {
    num += (t - mt) * (y - my);
    den += (t - mt) * (t - mt);
  }
  return den === 0 ? NaN : num / den;
}

/** Trend as a NAMED state, so "flat" is a measurement and not a missing slope. */
export function trendOf(ts, ys, band) {
  const s = slope(ts, ys);
  const level = median(ys);
  if (!Number.isFinite(s) || !Number.isFinite(level) || level === 0) {
    return { tag: "unknown" };
  }
  const rel = s / Math.abs(level);
  if (rel > band) return { tag: "rising", perSec: s, rel };
  if (rel < -band) return { tag: "falling", perSec: s, rel };
  return { tag: "flat", perSec: s, rel };
}

/**
 * Every exact float a snapshot carries, flattened for the finiteness sweep.
 *
 * Two kinds of slot, and the difference decides whether a `null` is a finding:
 *
 *  - REQUIRED (`req`): the block exists, so this number must exist. Anything
 *    that is not a number is a transport or schema regression and is reported
 *    as NaN — see the NAN_SENTINEL note. Skipping it is how a laundered NaN
 *    scored a piece healthy.
 *  - NULLABLE (`opt`): `null` is a documented state (R₁ with no pressure
 *    compiled, a pixel extGrad readback that has not landed). Absent is not
 *    nonfinite, and it is not zero either.
 */
export function numbersOf(snap) {
  const out = [];
  const req = (x) => out.push(typeof x === "number" ? x : NaN);
  const opt = (x) => {
    if (x !== null && x !== undefined) req(x);
  };
  req(snap.fps);
  req(snap.learnMs);
  if (snap.adv) {
    req(snap.adv.payoff);
    req(snap.adv.surprise);
    if (snap.adv.trainer === "fused") {
      req(snap.adv.payoffUngated);
      req(snap.adv.batchRms);
    }
    opt(snap.adv.r1);
    opt(snap.adv.r2);
    for (const h of snap.adv.heads ?? []) req(h);
  }
  if (snap.field) {
    // r1/r2 are REQUIRED here and NULLABLE on `adv`: the grid probe runs on
    // every piece, so a missing grid R₁ is a schema regression, whereas a
    // missing batch R₁ just means no pressure was compiled.
    for (const k of ["ac", "dc", "rmsF", "satFrac", "okuboWeiss", "r1", "r2"]) {
      req(snap.field[k]);
    }
  }
  if (snap.pixel) {
    req(snap.pixel.dLoss);
    req(snap.pixel.gLoss);
    opt(snap.pixel.extGradNorm);
  }
  return out;
}

/**
 * Aggregate a piece's sample stream into the exact floats the verdicts read.
 * The warmup prefix is DROPPED, not down-weighted: the first seconds contain a
 * pre-probe `field: null` and a predictor that has not seen a batch.
 */
export function aggregate(samples, gates = GATES) {
  const judged = samples.slice(gates.warmupSamples);
  const useful = judged.length ? judged : samples;
  const at = (f) => useful.map(f);
  return {
    count: useful.length,
    totalCount: samples.length,
    lastFrame: useful.length ? useful[useful.length - 1].frame : NaN,
    t: at((s) => s.t),
    fps: median(at((s) => s.fps)),
    ac: median(at((s) => s.field?.ac ?? NaN)),
    dc: median(at((s) => s.field?.dc ?? NaN)),
    rmsF: median(at((s) => s.field?.rmsF ?? NaN)),
    satFrac: median(at((s) => s.field?.satFrac ?? NaN)),
    okuboWeiss: median(at((s) => s.field?.okuboWeiss ?? NaN)),
    payoff: median(at((s) => s.adv?.payoff ?? NaN)),
    batchRms: median(at((s) => s.adv?.batchRms ?? NaN)),
    r1: median(at((s) => (s.adv?.r1 ?? null) === null ? NaN : s.adv.r1)),
    r2: median(at((s) => (s.adv?.r2 ?? null) === null ? NaN : s.adv.r2)),
    /**
     * GRID direction order, from the field probe — the same statistic as
     * `r1`/`r2` above (same τ) over a uniform 32² grid instead of the training
     * batch. Kept as separate fields rather than coalesced with `??`: they are
     * two measurements of two populations, and a verdict that silently
     * substituted one for the other would print a number the reader cannot
     * locate. This pair is the reason the laminar verdict now applies to
     * pieces with NO adversary at all (pixel critics), where `r1` is NaN.
     */
    fieldR1: median(at((s) => s.field?.r1 ?? NaN)),
    fieldR2: median(at((s) => s.field?.r2 ?? NaN)),
    extGradNorm: median(at((s) => s.pixel?.extGradNorm ?? NaN)),
    acSeries: at((s) => s.field?.ac ?? NaN),
    acTrend: trendOf(
      at((s) => s.t),
      at((s) => s.field?.ac ?? NaN),
      gates.trendBand
    ),
    /**
     * Trailing samples that share the LAST sample's frame number.
     *
     * A stopped render loop keeps publishing its last good snapshot forever,
     * and every metric in it stays perfectly finite and perfectly healthy —
     * so without this, a frozen page is the easiest way to pass this audit.
     * (Observed for real: a 180 s Max Structure run froze at t = 36 s and
     * scored `healthy` on 14 identical samples.) `soak_adversary.mjs` gates the
     * same failure by watching the HUD element mutate; the snapshot carries a
     * frame counter, which is the same idea without touching the DOM.
     */
    stalledTail: (() => {
      if (useful.length < 2) return 0;
      const lastFrame = useful[useful.length - 1].frame;
      let n = 0;
      for (let i = useful.length - 1; i >= 0 && useful[i].frame === lastFrame; i--) n++;
      return n;
    })(),
    nonfinite: useful.filter((s) => numbersOf(s).some((x) => !Number.isFinite(x))),
    /** The fused adversary's ONLY nonfinite canary: payoffUngated is the same
     *  scalar BEFORE the isFiniteF gate, so a disagreement means the gate
     *  silently replaced some tuple's NaN payoff with 0. */
    gateCanary: useful.filter(
      (s) =>
        s.adv?.trainer === "fused" &&
        Number.isFinite(s.adv.payoff) &&
        Number.isFinite(s.adv.payoffUngated) &&
        Math.abs(s.adv.payoffUngated - s.adv.payoff) > 1e-6
    ),
  };
}

/**
 * κ — one canonical verdict from the aggregate.
 *
 * ORDERED, and the order is the point: these failures nest. A NaN field is also
 * a dead field and also a laminar one; reporting "laminar-collapse" for a run
 * that is actually producing NaNs sends the reader to the pressure weights when
 * the bug is in the shader. Each arm below is the FIRST explanation that
 * accounts for everything after it.
 */
export function classify(agg, gates = GATES) {
  if (agg.count === 0) {
    return { tag: "no-signal" };
  }
  if (agg.nonfinite.length > 0 || agg.gateCanary.length > 0) {
    return {
      tag: "nonfinite",
      samples: agg.nonfinite.length,
      canary: agg.gateCanary.length,
    };
  }
  // Ranked here because a stopped loop makes every verdict below it a
  // statement about a moment that has already passed — but BELOW `nonfinite`,
  // because a NaN is the more specific explanation when both are present.
  if (agg.stalledTail >= 2) {
    return { tag: "stalled", samples: agg.stalledTail, frame: agg.lastFrame };
  }
  if (
    Math.abs(agg.payoff) > gates.payoffMax ||
    agg.batchRms > gates.batchRmsMax ||
    agg.extGradNorm > gates.extGradMax
  ) {
    return {
      tag: "blown-up",
      payoff: agg.payoff,
      batchRms: agg.batchRms,
      extGradNorm: agg.extGradNorm,
    };
  }
  if (Number.isFinite(agg.ac) && agg.ac < gates.acDead) {
    return { tag: "dead-field", ac: agg.ac, dc: agg.dc };
  }
  if (Number.isFinite(agg.satFrac) && agg.satFrac > gates.satFrozen) {
    return { tag: "frozen-saturated", satFrac: agg.satFrac };
  }
  // The √2 reading is AMBIGUOUS by itself (pressure note §5): the payoff parks
  // there both when the encoded target has gone to zero (R₁ → 1, the exploit)
  // and when the field is so isotropic D cannot predict it (R₁ → 0, healthy and
  // transient). Only the pair is diagnostic, so both halves are required.
  const atPole =
    Number.isFinite(agg.payoff) &&
    Math.abs(agg.payoff - gates.poleChord) < gates.poleBand;
  if (atPole && Number.isFinite(agg.r1) && agg.r1 > gates.r1Laminar) {
    return { tag: "pole-exploit", payoff: agg.payoff, r1: agg.r1 };
  }
  // Read on the GRID, so this fires on every piece — a pixel-critic piece has
  // no batch R₁ at all and used to be structurally incapable of reaching this
  // verdict no matter how laminar its field went.
  if (Number.isFinite(agg.fieldR1) && agg.fieldR1 > gates.r1Laminar) {
    return {
      tag: "laminar-collapse",
      where: "grid",
      r1: agg.fieldR1,
      r2: agg.fieldR2,
      ac: agg.ac,
    };
  }
  if (Number.isFinite(agg.r1) && agg.r1 > gates.r1Laminar) {
    return { tag: "laminar-collapse", where: "batch", r1: agg.r1, r2: agg.r2, ac: agg.ac };
  }
  // Ranked BELOW both laminar arms deliberately: nematic order is implied by
  // polar order (a field with one direction also has one axis), so when both
  // fire the R₁ reading is the more specific one and sends the reader to the
  // right place. This arm is what catches the case R₁ CANNOT see — see
  // GATES.r2Nematic. Un-gated until 2026-08-25 even though CLAUDE.md has named
  // the exploit the whole time, which is how Max Structure scored "healthy"
  // on a field that is one axis end to end.
  if (Number.isFinite(agg.fieldR2) && agg.fieldR2 > gates.r2Nematic) {
    return {
      tag: "nematic-collapse",
      r2: agg.fieldR2,
      r1: agg.fieldR1,
      ac: agg.ac,
    };
  }
  if (Number.isFinite(agg.fps) && agg.fps < gates.fpsFloor) {
    return { tag: "perf-regression", fps: agg.fps };
  }
  return { tag: "healthy", ac: agg.ac, r1: agg.fieldR1, fps: agg.fps };
}

/** Thin dispatcher: verdict → one line. One handler per tag, exhaustive. */
export function describe(v) {
  switch (v.tag) {
    case "healthy":
      return `healthy — ac ${fmt(v.ac)}, grid R1 ${fmt(v.r1)}, ${fmt(v.fps)} fps`;
    case "laminar-collapse":
      return (
        `LAMINAR COLLAPSE — ${v.where} R1 ${fmt(v.r1)} (> ${GATES.r1Laminar}) ` +
        `sustained; the field has one global direction. R2 ${fmt(v.r2)}, ` +
        `ac ${fmt(v.ac)}`
      );
    case "nematic-collapse":
      return (
        `NEMATIC COLLAPSE — grid R2 ${fmt(v.r2)} (> ${GATES.r2Nematic}) with R1 ` +
        `${fmt(v.r1)} LOW: the field has one global AXIS, not one direction. ` +
        `Counter-streaming sheets read as isotropic to R1 and look laminar. ` +
        `ac ${fmt(v.ac)}`
      );
    case "pole-exploit":
      return (
        `POLE EXPLOIT — payoff ${fmt(v.payoff)} sits on √2 AND R1 ${fmt(v.r1)} ` +
        `is high: the encoded target went to zero and G is collecting the ` +
        `north-pole bonus (√2 with LOW R1 would be the healthy transient)`
      );
    case "frozen-saturated":
      return (
        `FROZEN GENERATOR — satFrac ${fmt(v.satFrac)} of the domain has BOTH ` +
        `tanh components past ±0.9; the field cannot move`
      );
    case "dead-field":
      return `DEAD FIELD — ac ${fmt(v.ac)} (< ${GATES.acDead}); only the constant mode dc ${fmt(v.dc)} is left`;
    case "blown-up":
      return (
        `BLOWN UP — payoff ${fmt(v.payoff)}, batchRms ${fmt(v.batchRms)}, ` +
        `extGrad ${fmt(v.extGradNorm)}: diverging, still finite`
      );
    case "nonfinite":
      return (
        `NONFINITE — ${v.samples} sample(s) carry NaN/Inf` +
        (v.canary
          ? `, and ${v.canary} sample(s) tripped the payoffUngated≠payoff gate canary`
          : "")
      );
    case "perf-regression":
      return `PERF REGRESSION — ${fmt(v.fps)} fps (floor ${GATES.fpsFloor})`;
    case "stalled":
      return (
        `STALLED — the render loop stopped advancing: ${v.samples} consecutive ` +
        `samples all report frame ${v.frame}. Every metric below is a snapshot ` +
        `of a moment that has already passed`
      );
    case "page-error":
      return (
        `PAGE ERROR — ${v.detail}. A thrown page error invalidates every health ` +
        `metric taken after it, so it outranks a healthy reading rather than ` +
        `being a footnote`
      );
    case "no-signal":
      return (
        `NO SIGNAL — window.__nffHealth never produced a usable sample, but the ` +
        `browser WAS animating. Either the loop did not start (check the WebGPU ` +
        `adapter) or the snapshot publisher regressed`
      );
    case "browser-stalled":
      return (
        `BROWSER STALLED — requestAnimationFrame fired ${v.rafTicks} time(s) for a ` +
        `counter installed before the artwork loaded. NOTHING was measured; this ` +
        `is not a statement about the piece. Restart the browser/box and re-run`
      );
    default: {
      const never = v;
      throw new Error(`unhandled verdict ${JSON.stringify(never)}`);
    }
  }
}

export const isHealthy = (v) => v.tag === "healthy";

const fmt = (x) =>
  !Number.isFinite(x)
    ? "—"
    : Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0)
      ? x.toExponential(2)
      : x.toFixed(4);

/* ── NaN-preserving transport ────────────────────────────────────────────── */

/**
 * `JSON.stringify(NaN)` is `null`. So is `JSON.stringify(Infinity)`.
 *
 * That is not a footnote here — it is the exact failure this whole file exists
 * to prevent, and it BIT: the first 150 s audit of Pixel · VecField returned
 * `ac: null` for a third of its samples and scored the piece **healthy**,
 * because the nonfinite sweep skips non-numbers and a NaN had been laundered
 * into a legitimate-looking `null` in transit. A gate that cannot see a NaN is
 * not a gate.
 *
 * `null` is ALSO a real value in this schema (an unmeasured R₁, an unprobed
 * field, a pixel extGrad readback that has not landed), so the fix cannot be
 * "treat null as NaN" — the two must stay distinguishable. Nonfinite floats are
 * therefore encoded as SENTINEL STRINGS on the way out and restored on the way
 * in; genuine nulls travel as themselves.
 */
export const NAN_SENTINEL = "__nff_NaN__";
export const POS_INF_SENTINEL = "__nff_Infinity__";
export const NEG_INF_SENTINEL = "__nff_-Infinity__";

/** Runs INSIDE the page. Kept as a function value so it is testable here too. */
export function encodeSnapshot() {
  const h = window.__nffHealth;
  if (!h) return null;
  return JSON.stringify(h, (_k, v) => {
    if (typeof v !== "number" || Number.isFinite(v)) return v;
    if (Number.isNaN(v)) return "__nff_NaN__";
    return v > 0 ? "__nff_Infinity__" : "__nff_-Infinity__";
  });
}
const SNAPSHOT_ENCODER_SOURCE = encodeSnapshot;

/** `JSON.stringify` replacer that survives NaN/±Infinity, for written files. */
export function nanReplacer(_k, v) {
  if (typeof v !== "number" || Number.isFinite(v)) return v;
  if (Number.isNaN(v)) return NAN_SENTINEL;
  return v > 0 ? POS_INF_SENTINEL : NEG_INF_SENTINEL;
}

export function decodeSnapshot(encoded) {
  if (encoded === null || encoded === undefined) return null;
  return JSON.parse(encoded, (_k, v) => {
    if (v === NAN_SENTINEL) return NaN;
    if (v === POS_INF_SENTINEL) return Infinity;
    if (v === NEG_INF_SENTINEL) return -Infinity;
    return v;
  });
}

/* ── browser driving ─────────────────────────────────────────────────────── */

/**
 * REAL adapter flags. `--use-angle=metal` is the one that matters on this box;
 * `tools/smoke.mjs`'s software-fallback flags produce `adapter: null` here.
 */
const CHROME_ARGS = [
  "--no-sandbox",
  "--enable-unsafe-webgpu",
  "--enable-webgpu-developer-features",
  "--ignore-gpu-blocklist",
  "--use-angle=metal",
];

async function runPiece(browser, key, pieceName, opts) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  const pageErrors = [];
  const consoleErrors = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error" && !/favicon|404 \(File not found\)/.test(m.text())) {
      consoleErrors.push(m.text());
    }
  });

  const samples = [];
  let thrown = null;
  let rafTicks = -1;
  try {
    await page.goto(opts.base, { waitUntil: "domcontentloaded", timeout: 30_000 });
    // An INDEPENDENT rAF counter, installed before the artwork's loop and owned
    // by nothing in src/. When a run produces no signal, the only question worth
    // answering first is "did the artwork fail, or did the browser stop
    // producing frames at all?" — and on this box the second happens: after a
    // long series of headless WebGPU sessions, Chrome stops driving rAF
    // entirely, so a piece that measured 60 fps twenty minutes earlier reports
    // nothing. Without this counter that is indistinguishable from a regression,
    // which is exactly the attribution failure the soak-flake note is about.
    await page.evaluate(() => {
      window.__nffRaf = 0;
      const bump = () => {
        window.__nffRaf++;
        requestAnimationFrame(bump);
      };
      requestAnimationFrame(bump);
    });
    await page.waitForSelector('[data-testid="art-piece-gallery"]', { timeout: 30_000 });
    // Same selection mechanism soak_adversary.mjs uses: click the gallery
    // radio whose label contains the piece NAME. Deliberately not an index —
    // GALLERY is append-only but indices still move between builds.
    await page.evaluate((name) => {
      const button = [...document.querySelectorAll("button")].find((b) =>
        (b.textContent || "").includes(name)
      );
      if (!button) throw new Error(`no gallery button containing '${name}'`);
      button.click();
    }, pieceName);
    // Wait for the SNAPSHOT to name this piece — not the HUD text. That closes
    // the race where the old piece's loop is still publishing.
    await page.waitForFunction(
      (name) => window.__nffHealth?.piece === name,
      { timeout: 45_000 },
      pieceName
    );

    const total = Math.max(1, Math.round(opts.durationSec / opts.sampleSec));
    for (let i = 0; i < total; i++) {
      await sleep(opts.sampleSec * 1000);
      const snap = decodeSnapshot(
        await page.evaluate(SNAPSHOT_ENCODER_SOURCE)
      );
      if (snap && snap.piece === pieceName) samples.push(snap);
      const last = samples[samples.length - 1];
      console.log(
        `[${key}] ${String(i + 1).padStart(3)}/${total} ` +
          (last
            ? `t=${last.t.toFixed(0)}s fps=${fmt(last.fps)} ac=${fmt(last.field?.ac)} ` +
              `dc=${fmt(last.field?.dc)} sat=${fmt(last.field?.satFrac)} ` +
              `R1g=${fmt(last.field?.r1)} ` +
              `R1b=${last.adv?.r1 === null || last.adv?.r1 === undefined ? "—" : fmt(last.adv.r1)} ` +
              `payoff=${fmt(last.adv?.payoff)}`
            : "no snapshot")
      );
    }
    if (opts.shots) {
      await page.screenshot({ path: path.join(opts.runDir, `${key}.png`) });
    }
  } catch (e) {
    thrown = e instanceof Error ? e.message : String(e);
  }
  try {
    rafTicks = await page.evaluate(() => window.__nffRaf ?? -1);
  } catch {
    rafTicks = -1;
  }
  await page.close().catch(() => {});

  const agg = aggregate(samples);
  let verdict = classify(agg);
  // Attribution BEFORE blame: a `no-signal`/`stalled` run in which the browser
  // itself never produced a frame says nothing about the artwork, and must not
  // be reported as if it did.
  if ((verdict.tag === "no-signal" || verdict.tag === "stalled") && rafTicks <= 1) {
    verdict = { tag: "browser-stalled", rafTicks };
  }
  // A page error is not a health metric, but it invalidates every health metric
  // taken after it — so it OVERRIDES a healthy verdict rather than being
  // appended as a footnote nobody reads. It gets its OWN tag: reporting it as
  // `nonfinite` once produced the line "NONFINITE — 0 sample(s) carry NaN/Inf",
  // which is both untrue and unactionable.
  if (isHealthy(verdict) && (pageErrors.length || consoleErrors.length || thrown)) {
    verdict = {
      tag: "page-error",
      detail:
        thrown ??
        pageErrors[0] ??
        consoleErrors[0] ??
        "unknown",
    };
  }
  // Same sentinel replacer as the transport: a written artifact whose NaNs have
  // become `null` is a record of a run that did not happen.
  fs.writeFileSync(
    path.join(opts.runDir, `${key}.json`),
    JSON.stringify(
      { piece: pieceName, key, verdict, aggregate: stripSeries(agg), samples, pageErrors, consoleErrors, thrown },
      nanReplacer,
      2
    )
  );
  return { key, pieceName, verdict, agg, pageErrors, consoleErrors, thrown };
}

const stripSeries = (agg) => {
  const { acSeries, t, nonfinite, gateCanary, ...rest } = agg;
  return {
    ...rest,
    nonfiniteSamples: nonfinite.length,
    gateCanarySamples: gateCanary.length,
  };
};

/* ── self-test: the pure half, no GPU ────────────────────────────────────── */

function selfTest() {
  let failures = 0;
  const ok = (cond, msg) => {
    console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
    if (!cond) failures++;
  };
  const snap = (over = {}) => ({
    piece: "x", frame: 1, t: 1, fps: 60, learnMs: 1, backend: "webgpu",
    trainer: "fused",
    adv: { trainer: "fused", payoff: 0.6, payoffUngated: 0.6, surprise: 0.6, r1: 0.05, r2: 0.05, batchRms: 3, heads: [1, 1] },
    field: { ac: 0.2, dc: 0.05, rmsF: 0.21, satFrac: 0.02, okuboWeiss: -0.3, r1: 0.04, r2: 0.06, gridN: 32 },
    pixel: null,
    ...over,
  });
  // Distinct `frame` per sample: a stream that repeats a frame is a STALL, and
  // the fixtures below are meant to exercise everything except that.
  const run = (over, n = 8) =>
    classify(
      aggregate(Array.from({ length: n }, (_, i) => ({ ...snap(over), t: i, frame: 100 * i })))
    );

  ok(run({}).tag === "healthy", "a nominal snapshot stream is healthy");
  // REGRESSION GUARD, 2026-08-25. This exact stream — R₁ low, R₂ high — scored
  // "healthy" for the whole life of this file, and it is the shape a
  // counter-streaming ±F₀ sheet actually produces. It is the one collapse the
  // R₁ gate is structurally blind to, so if this case ever reads healthy again
  // the gallery can go single-axis with every verdict green.
  ok(
    run({ field: { ...snap().field, r1: 0.03, r2: 0.93 } }).tag === "nematic-collapse",
    "R1 LOW with R2 HIGH is a counter-streaming sheet, not an isotropic field"
  );
  ok(
    run({ field: { ...snap().field, r1: 0.97, r2: 0.95 }, adv: null }).tag ===
      "laminar-collapse",
    "when BOTH orders are high, R1 wins: polar order is the more specific reading"
  );
  ok(
    run({ field: { ...snap().field, r1: 0.03, r2: 0.49 } }).tag === "healthy",
    "R2 just under the gate is still healthy — the arm is a threshold, not a slope"
  );
  ok(
    run({ adv: { ...snap().adv, r1: 0.98 } }).tag === "laminar-collapse",
    "batch R1 0.98 → laminar-collapse"
  );
  {
    // The coverage this pair was added for: a piece with NO adversary block.
    // Before the grid R₁ existed, `agg.r1` was NaN here and the stream scored
    // healthy however unidirectional the field had gone.
    const noAdv = { adv: null, field: { ...snap().field, r1: 0.97, r2: 0.95 } };
    ok(
      run(noAdv).tag === "laminar-collapse",
      "grid R1 0.97 on a piece with NO adversary → laminar-collapse (was: healthy)"
    );
    ok(
      run({ adv: null }).tag === "healthy",
      "…and the same adversary-free stream with a swirling field is still healthy"
    );
  }
  ok(
    run({ adv: { ...snap().adv, r1: 0.98, payoff: Math.SQRT2, surprise: Math.SQRT2, payoffUngated: Math.SQRT2 } }).tag ===
      "pole-exploit",
    "√2 payoff AND high R1 → pole-exploit (the collapse half of the ambiguity)"
  );
  ok(
    run({ adv: { ...snap().adv, r1: 0.004, payoff: Math.SQRT2, surprise: Math.SQRT2, payoffUngated: Math.SQRT2 } }).tag ===
      "healthy",
    "√2 payoff with LOW R1 → healthy (the transient half — pressure note §5)"
  );
  ok(
    run({ field: { ...snap().field, satFrac: 0.46 } }).tag === "frozen-saturated",
    "satFrac 0.46 (the measured collapsed point observer) → frozen-saturated"
  );
  ok(
    run({ field: { ...snap().field, ac: 7e-4 * 0.1 } }).tag === "dead-field",
    "ac 7e-5 → dead-field"
  );
  ok(
    run({
      adv: { ...snap().adv, payoff: 1e6, surprise: 1e6, payoffUngated: 1e6 },
    }).tag === "blown-up",
    "payoff 1e6 (with the gate agreeing) → blown-up: finite but diverging"
  );
  ok(run({ fps: 12 }).tag === "perf-regression", "12 fps → perf-regression");
  ok(
    run({ field: { ...snap().field, ac: NaN } }).tag === "nonfinite",
    "a NaN metric → nonfinite, and it OUTRANKS the dead-field reading it implies"
  );
  ok(
    run({ adv: { ...snap().adv, payoffUngated: 0.9 } }).tag === "nonfinite",
    "payoffUngated ≠ payoff → nonfinite via the gate canary, with no NaN present"
  );
  ok(
    run({ adv: { ...snap().adv, r1: null, r2: null } }).tag === "healthy",
    "r1: null (pressure not compiled) is UNMEASURED, not a 0 that reads isotropic"
  );
  ok(classify(aggregate([])).tag === "no-signal", "no samples → no-signal");
  {
    const rising = Array.from({ length: 10 }, (_, i) => ({
      ...snap({ field: { ...snap().field, ac: 0.1 + 0.02 * i } }),
      t: i * 2,
      frame: 100 * i,
    }));
    ok(aggregate(rising).acTrend.tag === "rising", "a climbing AC series reads as rising");
    const flat = Array.from({ length: 10 }, (_, i) => ({ ...snap(), t: i * 2, frame: 100 * i }));
    ok(aggregate(flat).acTrend.tag === "flat", "a constant AC series reads as flat");
  }
  // STALL — the failure a frozen 180 s Max Structure run actually produced, and
  // that every metric-based gate above reports as perfectly healthy.
  {
    const frozen = Array.from({ length: 10 }, (_, i) => ({
      ...snap(),
      t: i < 4 ? i * 2 : 6,
      frame: i < 4 ? 100 * i : 300,
    }));
    ok(classify(aggregate(frozen)).tag === "stalled", "a frozen frame counter → stalled");
    ok(
      aggregate(frozen).nonfinite.length === 0 && median(frozen.map((s) => s.fps)) === 60,
      "…and every NUMBER in that frozen stream is finite and nominal — which is " +
        "exactly why the frame counter, not the metrics, has to catch it"
    );
    const hitching = Array.from({ length: 10 }, (_, i) => ({
      ...snap(),
      t: i * 2,
      frame: i === 5 ? 400 : 100 * i,
    }));
    ok(
      classify(aggregate(hitching)).tag === "healthy",
      "a single non-monotone frame in the middle is NOT a stall (only the tail counts)"
    );
  }
  // TRANSPORT — the regression that scored a NaN-producing piece "healthy".
  {
    const withNaN = snap({ field: { ...snap().field, ac: NaN, dc: -Infinity } });
    globalThis.window = { __nffHealth: withNaN };
    const round = decodeSnapshot(encodeSnapshot());
    delete globalThis.window;
    ok(
      Number.isNaN(round.field.ac) && round.field.dc === -Infinity,
      "NaN/−Infinity survive the page→node transport (plain JSON turns both into null)"
    );
    ok(
      round.adv.r1 === 0.05 && round.pixel === null,
      "genuine nulls and finite numbers are unchanged by the sentinel encoding"
    );
    ok(
      classify(aggregate(Array.from({ length: 8 }, (_, i) => ({ ...round, t: i })))).tag ===
        "nonfinite",
      "a transported NaN reaches the verdict as nonfinite"
    );
    ok(
      numbersOf({
        ...snap(),
        field: { ac: null, dc: 1, rmsF: 1, satFrac: 0, okuboWeiss: 0, r1: 0, r2: 0 },
      }).some(Number.isNaN),
      "a REQUIRED field metric arriving as null reads as NaN, not as 'skip it'"
    );
    ok(
      !numbersOf({ ...snap(), adv: { ...snap().adv, r1: null, r2: null } }).some(Number.isNaN),
      "a NULLABLE slot arriving as null is absent, not nonfinite"
    );
  }
  // Every verdict tag must have a handler — an unhandled one throws by design.
  for (const v of [
    { tag: "healthy" }, { tag: "laminar-collapse" }, { tag: "nematic-collapse" },
    { tag: "pole-exploit" },
    { tag: "frozen-saturated" }, { tag: "dead-field" }, { tag: "blown-up" },
    { tag: "nonfinite" }, { tag: "perf-regression" }, { tag: "no-signal" },
    { tag: "stalled" }, { tag: "page-error" }, { tag: "browser-stalled" },
  ]) {
    try {
      describe(v);
    } catch {
      ok(false, `describe() has no handler for '${v.tag}'`);
    }
  }
  ok(true, "describe() is exhaustive over every verdict tag");
  console.log(failures === 0 ? "\nHEALTH AUDIT SELF-TEST PASS" : `\n${failures} SELF-TEST FAILURE(S)`);
  return failures;
}

/* ── main ────────────────────────────────────────────────────────────────── */

/**
 * Run the audit ONLY when this file is the process entry point.
 *
 * Without this guard, `import("./tools/health_audit.mjs")` — the obvious way to
 * reuse `classify`/`aggregate` to re-score an already-recorded run offline —
 * launches a real 16-piece GPU audit as an import side effect. Hit for real on
 * 2026-08-25 while re-scoring a finished run against a new gate.
 */
const INVOKED_DIRECTLY =
  !!process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (INVOKED_DIRECTLY && process.argv[2] === "--self-test") {
  process.exit(selfTest() === 0 ? 0 : 1);
}

if (INVOKED_DIRECTLY) {
  const arg = process.argv[2] ?? "all";
  const base = process.argv[3] ?? DEFAULT_BASE;
  const durationSec = Number(process.argv[4] ?? 90);
  const sampleSec = Number(process.argv[5] ?? 2);
  const keys =
    arg === "all"
      ? ALL_KEYS
      : arg === "adversary"
        ? ADVERSARY_KEYS
        : arg === "pixel"
          ? PIXEL_KEYS
          : arg.split(",").map((k) => k.trim()).filter(Boolean);
  for (const k of keys) {
    if (!PIECES[k]) {
      console.error(`unknown piece key '${k}' — known: ${ALL_KEYS.join(", ")}`);
      process.exit(2);
    }
  }

  const runDir = path.join(
    ROOT,
    "output",
    "health-audit",
    new Date().toISOString().replace(/[:.]/g, "-")
  );
  fs.mkdirSync(runDir, { recursive: true });

  const browser = await puppeteer.launch({ headless: "new", args: CHROME_ARGS });
  const results = [];
  try {
    // SERIALIZED: every piece contends for the same GPU, and a parallel run would
    // measure scheduler contention rather than the artwork.
    for (const key of keys) {
      console.log(`\n=== ${key}: ${PIECES[key]} (${durationSec}s) ===`);
      results.push(
        await runPiece(browser, key, PIECES[key], {
          base, durationSec, sampleSec, runDir, shots: !!process.env.HEALTH_SHOTS,
        })
      );
    }
  } finally {
    await browser.close();
  }

  console.log("\n╔══ VERDICTS ═══════════════════════════════════════════════════");
  const pad = Math.max(...results.map((r) => r.pieceName.length));
  for (const r of results) {
    const a = r.agg;
    console.log(
      `║ ${r.pieceName.padEnd(pad)}  ${isHealthy(r.verdict) ? "PASS" : "FAIL"}  ${describe(r.verdict)}`
    );
    console.log(
      `║ ${" ".repeat(pad)}        ac ${fmt(a.ac)} (${a.acTrend.tag}) · dc ${fmt(a.dc)} · ` +
        `dc/ac ${fmt(a.dc / a.ac)} · sat ${fmt(a.satFrac)} · OW ${fmt(a.okuboWeiss)} · ` +
        `R1 grid ${fmt(a.fieldR1)}/R2 ${fmt(a.fieldR2)} · R1 batch ${fmt(a.r1)} · ` +
        `payoff ${fmt(a.payoff)} · ${fmt(a.fps)} fps · n=${a.count}`
    );
  }
  console.log("╚═══════════════════════════════════════════════════════════════");
  console.log(`artifacts: ${runDir}`);

  fs.writeFileSync(
    path.join(runDir, "summary.json"),
    JSON.stringify(
      {
        base, durationSec, sampleSec, gates: GATES,
        pieces: results.map((r) => ({
          key: r.key, piece: r.pieceName, verdict: r.verdict,
          healthy: isHealthy(r.verdict), aggregate: stripSeries(r.agg),
        })),
      },
      nanReplacer,
      2
    )
  );

  const unhealthy = results.filter((r) => !isHealthy(r.verdict)).length;
  console.log(`${unhealthy} unhealthy piece(s)`);
  process.exit(unhealthy);

}
