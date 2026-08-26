/**
 * HEALTH SWEEP — one health audit per cell of a config MATRIX.
 *
 *   bun tools/health_sweep.ts <spec.json> [baseURL]
 *   bun tools/health_sweep.ts --example > sweep.json
 *   bun tools/health_sweep.ts --self-test
 *
 * `tools/health_audit.mjs` measures ONE config: it takes a single base URL and
 * a list of gallery pieces. That is the right unit for "is the shipped gallery
 * healthy" and the wrong unit for "which objective/architecture actually
 * produces structure", which needs the same instrument run across a product of
 * axes and the results compared. This file is the driver for the second
 * question; `tools/health_report.ts` is the reducer that reads what it writes.
 *
 * IT REUSES THE AUDIT, IT DOES NOT REIMPLEMENT IT. `runPiece`, `aggregate` and
 * `classify` are imported from `health_audit.mjs` verbatim, so a cell's verdict
 * here and a plain audit's verdict on the same URL are the same computation.
 * (That import is only safe because the audit guards its CLI behind
 * INVOKED_DIRECTLY — see the comment there.) An editor may show red squiggles
 * on that import: `health_audit.mjs` is plain JS with no declarations and this
 * repo has no tsconfig, so the TS service cannot see into it. bun resolves it
 * fine and the sweep is verified end-to-end — do not "fix" it by duplicating
 * the audit's logic here, which is the one thing this file must not do.
 *
 * THREE THINGS THIS FILE EXISTS TO GET RIGHT
 *
 * 1. **RESUMABLE.** A serious matrix is hours: 4 objectives x 3 archs x 5
 *    pieces x 300 s is over five hours, and the audit is deliberately
 *    serialized (parallel runs measure GPU scheduler contention, not the
 *    artwork). A crash at cell 47 must not cost cells 1..46, so a cell whose
 *    result file already parses is SKIPPED. Re-running the same spec into the
 *    same run directory resumes it.
 *
 * 2. **CELLS THAT DID NOT ACTUALLY DIFFER ARE FOUND AND SAID SO.** `?arch=` is
 *    honoured only on `archEditable` pieces; adversary knobs are inert on a
 *    piece with no adversary. A sweep that trusted its own URLs would run the
 *    identical network N times and report "architecture makes no difference".
 *    Every cell records the MEASURED `ArchHealth` fingerprint from the snapshot
 *    (see src/health.ts), and `collisions()` reports any axis whose values
 *    produced one fingerprint. That check is the reason to trust the report.
 *
 * 3. **UNKNOWN AXIS VALUES FAIL AT PARSE TIME, NOT AT HOUR THREE.** `arch` and
 *    `piece` values are validated against `ARCH` and the audit's `PIECES` map
 *    before the browser opens.
 */
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import puppeteer from "puppeteer";
import {
  PIECES,
  CHROME_ARGS,
  runPiece,
  isHealthy,
  describe,
} from "./health_audit.mjs";
import { ARCH } from "../src/core/field/arch";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DEFAULT_BASE = "http://localhost:1234/index.html";

/** The axis that selects a gallery piece; every other axis is a URL parameter. */
const PIECE_AXIS = "piece";

/* ── the spec ─────────────────────────────────────────────────────────────── */

export interface SweepSpec {
  readonly name: string;
  readonly durationSec: number;
  readonly sampleSec: number;
  /**
   * Axis name -> the values it takes. `piece` selects the gallery piece by the
   * audit's short key; EVERY other axis becomes a URL query parameter verbatim,
   * so the axis names are the app's own knobs (`arch`, `advLoss`, `advM`,
   * `advK`, `advPolar`, `advNematic`, `advWeight`, `gLR`, `dLR`, …). Keeping
   * them verbatim rather than behind a translation table means a knob added to
   * the app is sweepable the day it lands, with no change here.
   */
  readonly axes: Readonly<Record<string, readonly (string | number)[]>>;
}

export interface Cell {
  readonly id: string;
  readonly pieceKey: string;
  readonly pieceName: string;
  readonly params: Readonly<Record<string, string>>;
  readonly url: string;
}

/** κ — messy JSON to a canonical spec, or a typed throw. No silent defaults. */
export function parseSpec(raw: unknown): SweepSpec {
  if (!raw || typeof raw !== "object") throw new Error("spec must be an object");
  const o = raw as Record<string, unknown>;
  const name = typeof o.name === "string" && o.name ? o.name : null;
  if (!name) throw new Error("spec.name must be a non-empty string");
  const num = (k: string, fallback: number): number => {
    if (o[k] === undefined) return fallback;
    const v = Number(o[k]);
    if (!Number.isFinite(v) || v <= 0) throw new Error(`spec.${k} must be > 0, got ${o[k]}`);
    return v;
  };
  const durationSec = num("durationSec", 120);
  const sampleSec = num("sampleSec", 2);
  if (!o.axes || typeof o.axes !== "object") throw new Error("spec.axes must be an object");
  const axesIn = o.axes as Record<string, unknown>;
  const axes: Record<string, readonly (string | number)[]> = {};
  for (const [key, values] of Object.entries(axesIn)) {
    if (!Array.isArray(values) || values.length === 0) {
      throw new Error(`spec.axes.${key} must be a non-empty array`);
    }
    for (const v of values) {
      if (typeof v !== "string" && typeof v !== "number") {
        throw new Error(`spec.axes.${key} values must be string|number, got ${typeof v}`);
      }
    }
    axes[key] = values as (string | number)[];
  }
  if (!axes[PIECE_AXIS]) throw new Error(`spec.axes.${PIECE_AXIS} is required`);
  // Validate the two closed vocabularies BEFORE the browser opens. A mistyped
  // arch value would otherwise be a URIError on every page load an hour in —
  // or worse, on a non-archEditable piece, silently nothing.
  for (const p of axes[PIECE_AXIS]) {
    if (!(String(p) in PIECES)) {
      throw new Error(
        `unknown piece '${p}' — known: ${Object.keys(PIECES).join(", ")}`
      );
    }
  }
  for (const a of axes.arch ?? []) {
    if (!(String(a) in ARCH)) {
      throw new Error(`unknown arch '${a}' — known: ${Object.keys(ARCH).join(", ")}`);
    }
  }
  return { name, durationSec, sampleSec, axes };
}

/* ── the matrix ───────────────────────────────────────────────────────────── */

/**
 * Cartesian product of the axes, in a STABLE order (axis order as written,
 * values in order). Stability is what makes a resumed run line up with the
 * interrupted one and what makes two runs of the same spec comparable
 * cell-for-cell.
 */
export function cells(spec: SweepSpec, base: string): Cell[] {
  const names = Object.keys(spec.axes);
  let combos: Record<string, string>[] = [{}];
  for (const axis of names) {
    const next: Record<string, string>[] = [];
    for (const combo of combos) {
      for (const value of spec.axes[axis]) {
        next.push({ ...combo, [axis]: String(value) });
      }
    }
    combos = next;
  }
  return combos.map((combo) => {
    const { [PIECE_AXIS]: pieceKey, ...params } = combo;
    const q = new URLSearchParams();
    // Sorted so the URL — and therefore the cell id — does not depend on the
    // key order the JSON happened to use.
    for (const k of Object.keys(params).sort()) q.set(k, params[k]);
    const sep = base.includes("?") ? "&" : "?";
    const query = q.toString();
    return {
      id: cellId(pieceKey, params),
      pieceKey,
      pieceName: (PIECES as Record<string, string>)[pieceKey],
      params,
      url: query ? `${base}${sep}${query}` : base,
    };
  });
}

/**
 * Filesystem-safe, human-readable cell id. Readable on purpose: these become
 * directory names an operator greps through at 2am, and an opaque hash there
 * means opening manifest.json to answer "which one is this".
 */
export function cellId(pieceKey: string, params: Record<string, string>): string {
  const parts = [pieceKey];
  for (const k of Object.keys(params).sort()) {
    parts.push(`${k}-${params[k]}`);
  }
  return parts.join("__").replace(/[^A-Za-z0-9._-]/g, "_");
}

/* ── the collision check (reason 2 in the header) ─────────────────────────── */

export interface Collision {
  readonly axis: string;
  readonly pieceKey: string;
  readonly fingerprint: string;
  readonly values: readonly string[];
}

/**
 * Axes that were supposed to vary the network and did not.
 *
 * Compares, WITHIN one piece and holding every other axis fixed, the measured
 * arch fingerprints of the cells that differ only along `axis`. If they are all
 * identical, the app ignored that axis on that piece and every downstream
 * comparison along it is between two runs of the same network.
 *
 * Only `arch` is checked against the fingerprint: it is the axis that changes
 * the compiled network. A `advPolar` difference SHOULD leave the fingerprint
 * alone — same network, different loss — so flagging it would be noise.
 */
export function collisions(
  results: readonly { cell: Cell; fingerprint: string | null }[]
): Collision[] {
  const out: Collision[] = [];
  const axis = "arch";
  const groups = new Map<string, { cell: Cell; fingerprint: string | null }[]>();
  for (const r of results) {
    if (!(axis in r.cell.params)) continue;
    const rest = Object.keys(r.cell.params)
      .filter((k) => k !== axis)
      .sort()
      .map((k) => `${k}=${r.cell.params[k]}`)
      .join("&");
    const key = `${r.cell.pieceKey}|${rest}`;
    groups.set(key, [...(groups.get(key) ?? []), r]);
  }
  for (const group of groups.values()) {
    if (group.length < 2) continue;
    const prints = new Set(group.map((g) => g.fingerprint ?? "unmeasured"));
    if (prints.size === 1) {
      out.push({
        axis,
        pieceKey: group[0].cell.pieceKey,
        fingerprint: [...prints][0],
        values: group.map((g) => g.cell.params[axis]),
      });
    }
  }
  return out;
}

/** The measured fingerprint of a finished cell, from its written result file. */
export function fingerprintOf(result: unknown): string | null {
  const r = result as { samples?: { arch?: Record<string, unknown> | null }[] };
  for (const s of r?.samples ?? []) {
    const a = s?.arch;
    if (a) return `${a.kind}/${a.encoding}/w${a.weightFloats}/m${a.macsPerParticle}/c${a.classes}`;
  }
  return null;
}

/* ── the example spec ─────────────────────────────────────────────────────── */

const EXAMPLE: SweepSpec = {
  name: "objective-x-arch",
  durationSec: 180,
  sampleSec: 2,
  axes: {
    piece: ["pair4", "hashgrid"],
    arch: ["dualStd", "dualFourier"],
    advPolar: [0, 0.05],
  },
};

/* ── self-test: pure, no GPU, no server ───────────────────────────────────── */

export function selfTest(): number {
  let failures = 0;
  const ok = (cond: boolean, msg: string) => {
    console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`);
    if (!cond) failures++;
  };
  const throws = (fn: () => unknown, msg: string) => {
    try {
      fn();
      ok(false, `${msg} (did not throw)`);
    } catch {
      ok(true, msg);
    }
  };

  console.log("\n--- 1. spec parsing rejects what would waste GPU hours ---");
  throws(() => parseSpec({ name: "x", axes: {} }), "no piece axis → throws");
  throws(
    () => parseSpec({ name: "x", axes: { piece: ["nope"] } }),
    "unknown piece key → throws BEFORE the browser opens"
  );
  throws(
    () => parseSpec({ name: "x", axes: { piece: ["pair4"], arch: ["mlpTypo"] } }),
    "unknown arch value → throws (a typo must not silently sweep the default arch)"
  );
  throws(() => parseSpec({ name: "", axes: { piece: ["pair4"] } }), "empty name → throws");
  throws(
    () => parseSpec({ name: "x", durationSec: 0, axes: { piece: ["pair4"] } }),
    "durationSec 0 → throws"
  );
  throws(
    () => parseSpec({ name: "x", axes: { piece: ["pair4"], arch: [] } }),
    "empty axis → throws (it would silently drop the axis from the product)"
  );
  ok(
    parseSpec({ name: "x", axes: { piece: ["pair4"] } }).durationSec === 120,
    "durationSec defaults to 120 when omitted"
  );

  console.log("\n--- 2. the matrix is complete and stable ---");
  {
    const spec = parseSpec(EXAMPLE);
    const c = cells(spec, "http://h/i.html");
    ok(c.length === 2 * 2 * 2, `2x2x2 spec → ${c.length} cells (expected 8)`);
    ok(new Set(c.map((x) => x.id)).size === c.length, "every cell id is unique");
    // Re-deriving must give byte-identical ids, or a resumed run re-runs
    // everything into new directories and the old cells become orphans.
    const again = cells(parseSpec(EXAMPLE), "http://h/i.html");
    ok(
      c.every((x, i) => x.id === again[i].id && x.url === again[i].url),
      "the product is deterministic across two independent derivations"
    );
    // Key ORDER in the JSON must not change the URL — otherwise two specs that
    // mean the same thing produce two incomparable run directories.
    const shuffled = cells(
      parseSpec({ ...EXAMPLE, axes: { advPolar: [0, 0.05], arch: ["dualStd", "dualFourier"], piece: ["pair4", "hashgrid"] } }),
      "http://h/i.html"
    );
    ok(
      new Set(shuffled.map((x) => x.id)).size === 8 &&
        shuffled.every((x) => c.some((y) => y.id === x.id)),
      "axis order in the spec does not change the cell ids"
    );
    ok(
      c.every((x) => x.url.includes("arch=") && x.url.includes("advPolar=")),
      "every non-piece axis lands in the URL"
    );
    ok(
      !c.some((x) => x.url.includes("piece=")),
      "the piece axis does NOT leak into the URL (it is a gallery click)"
    );
    const withQuery = cells(spec, "http://h/i.html?train=tfjs");
    ok(
      withQuery.every((x) => x.url.includes("train=tfjs&")),
      "a base URL that already has a query is extended with & not ?"
    );
  }

  console.log("\n--- 3. collisions: an axis that did not actually vary ---");
  {
    const spec = parseSpec(EXAMPLE);
    const c = cells(spec, "http://h/i.html");
    // Every arch produced the same compiled network ⇒ the app ignored ?arch=.
    const ignored = c.map((cell) => ({ cell, fingerprint: "helmholtz/raw/w100/m10/c0" }));
    const found = collisions(ignored);
    ok(
      found.length === 4,
      `an ignored arch axis is reported for all 4 (piece x advPolar) groups — got ${found.length}`
    );
    // Distinct fingerprints per arch ⇒ nothing to report.
    const honoured = c.map((cell) => ({
      cell,
      fingerprint: `helmholtz/raw/w${cell.params.arch === "dualStd" ? 100 : 900}/m10/c0`,
    }));
    ok(collisions(honoured).length === 0, "a genuinely varying arch axis reports no collision");
    // The check must be per-piece: arch honoured on one piece and ignored on
    // the other is the REAL case (archEditable is per-piece), and reporting it
    // as "no collision" would hide exactly half a sweep.
    const mixed = c.map((cell) => ({
      cell,
      fingerprint:
        cell.pieceKey === "pair4"
          ? `helmholtz/raw/w${cell.params.arch === "dualStd" ? 100 : 900}/m10/c0`
          : "helmholtz/raw/w555/m10/c0",
    }));
    const m = collisions(mixed);
    ok(
      m.length === 2 && m.every((x) => x.pieceKey === "hashgrid"),
      `arch honoured on one piece and ignored on another is caught per-piece — got ${m.length} on ${[...new Set(m.map((x) => x.pieceKey))].join(",")}`
    );
  }

  console.log("\n--- 4. fingerprint extraction ---");
  {
    ok(fingerprintOf({ samples: [] }) === null, "no samples → null (not a fake fingerprint)");
    ok(
      fingerprintOf({ samples: [{ arch: null }, { arch: { kind: "helmholtz", encoding: "raw", weightFloats: 8, macsPerParticle: 4, classes: 0 } }] }) ===
        "helmholtz/raw/w8/m4/c0",
      "a leading pre-advect null sample is skipped, the first real arch wins"
    );
  }

  console.log(
    failures === 0 ? "\nSWEEP SELF-TEST PASS" : `\n${failures} SWEEP SELF-TEST FAILURE(S)`
  );
  return failures;
}

/* ── CLI ──────────────────────────────────────────────────────────────────── */

const INVOKED_DIRECTLY =
  !!process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (INVOKED_DIRECTLY) {
  const arg = process.argv[2];
  if (arg === "--self-test") {
    process.exit(selfTest() === 0 ? 0 : 1);
  }
  if (arg === "--example") {
    console.log(JSON.stringify(EXAMPLE, null, 2));
    process.exit(0);
  }
  if (!arg) {
    console.error(
      "usage: bun tools/health_sweep.ts <spec.json> [baseURL]\n" +
        "       bun tools/health_sweep.ts --example > sweep.json\n" +
        "       bun tools/health_sweep.ts --self-test"
    );
    process.exit(2);
  }

  const spec = parseSpec(JSON.parse(fs.readFileSync(arg, "utf8")));
  const base = process.argv[3] ?? DEFAULT_BASE;
  const matrix = cells(spec, base);

  // The run directory is keyed by the SPEC NAME, not a timestamp, precisely so
  // that re-invoking resumes instead of starting a parallel universe. Rename
  // the spec (or the directory) to force a fresh run.
  const runDir = path.join(ROOT, "output", "health-sweep", spec.name);
  const cellsDir = path.join(runDir, "cells");
  fs.mkdirSync(cellsDir, { recursive: true });

  const totalSec = matrix.length * spec.durationSec;
  console.log(
    `sweep '${spec.name}': ${matrix.length} cells x ${spec.durationSec}s ` +
      `= ~${(totalSec / 3600).toFixed(1)}h serialized (one GPU)\n` +
      `axes: ${Object.entries(spec.axes).map(([k, v]) => `${k}[${v.length}]`).join(" x ")}\n` +
      `out:  ${runDir}\n`
  );

  const browser = await puppeteer.launch({ headless: true, args: CHROME_ARGS });
  const results: { cell: Cell; fingerprint: string | null; verdict: unknown }[] = [];
  try {
    for (const [i, cell] of matrix.entries()) {
      const cellDir = path.join(cellsDir, cell.id);
      const resultFile = path.join(cellDir, `${cell.pieceKey}.json`);
      const tag = `[${i + 1}/${matrix.length}] ${cell.id}`;

      // RESUME: a cell whose result parses is done. Deliberately parses rather
      // than stat()s — a run killed mid-write leaves a truncated file, and
      // treating that as complete would silently drop a cell from the report.
      if (fs.existsSync(resultFile)) {
        try {
          const prior = JSON.parse(fs.readFileSync(resultFile, "utf8"));
          results.push({
            cell,
            fingerprint: fingerprintOf(prior),
            verdict: prior.verdict,
          });
          console.log(`${tag} — SKIP (already recorded)`);
          continue;
        } catch {
          console.log(`${tag} — re-running (prior result file did not parse)`);
        }
      }

      fs.mkdirSync(cellDir, { recursive: true });
      console.log(`${tag} — ${cell.pieceName} ${cell.url}`);
      const r = await runPiece(browser, cell.pieceKey, cell.pieceName, {
        base: cell.url,
        durationSec: spec.durationSec,
        sampleSec: spec.sampleSec,
        runDir: cellDir,
        shots: !!process.env.HEALTH_SHOTS,
      });
      const written = JSON.parse(fs.readFileSync(resultFile, "utf8"));
      const fingerprint = fingerprintOf(written);
      results.push({ cell, fingerprint, verdict: r.verdict });
      console.log(
        `${tag} — ${isHealthy(r.verdict) ? "PASS" : "FAIL"} ${describe(r.verdict)}` +
          `${fingerprint ? ` · arch ${fingerprint}` : " · arch UNMEASURED"}`
      );

      // Written after every cell, not at the end: an interrupted sweep must
      // still produce a manifest the reporter can read.
      fs.writeFileSync(
        path.join(runDir, "manifest.json"),
        JSON.stringify(
          {
            spec,
            base,
            complete: results.length === matrix.length,
            cells: matrix.map((c) => {
              const done = results.find((x) => x.cell.id === c.id);
              return {
                ...c,
                dir: path.relative(runDir, path.join(cellsDir, c.id)),
                done: !!done,
                fingerprint: done?.fingerprint ?? null,
              };
            }),
          },
          null,
          2
        )
      );
    }
  } finally {
    await browser.close();
  }

  const bad = collisions(results);
  if (bad.length) {
    console.log("\n╔══ AXIS COLLISIONS ════════════════════════════════════════════");
    for (const c of bad) {
      console.log(
        `║ '${c.axis}' did NOT vary on piece '${c.pieceKey}': ` +
          `${c.values.join(", ")} all compiled ${c.fingerprint}`
      );
    }
    console.log("║ Those comparisons are between identical networks — the piece");
    console.log("║ is probably not archEditable. Do not read them as a result.");
    console.log("╚═══════════════════════════════════════════════════════════════");
  }

  const failed = results.filter((r) => !isHealthy(r.verdict as { tag: string })).length;
  console.log(`\nsweep '${spec.name}' complete — ${results.length} cells, ${failed} unhealthy`);
  console.log(`artifacts: ${runDir}`);
  console.log(`report:    bun tools/health_report.ts ${path.relative(ROOT, runDir)}`);
  process.exit(0);
}

export { EXAMPLE, PIECE_AXIS, pathToFileURL };
