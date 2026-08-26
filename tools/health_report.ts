/**
 * HEALTH REPORT — a finished sweep directory, reduced to one self-contained page.
 *
 *   bun tools/health_report.ts <sweepRunDir> [out.html]
 *   bun tools/health_report.ts output/health-sweep/objective-x-arch
 *   bun tools/health_report.ts --self-test
 *
 * SEPARATE FROM THE DRIVER ON PURPOSE. A matrix is hours of serialized GPU;
 * the reduction is milliseconds. Keeping them apart means the report can be
 * re-cut — different ranking metric, a new gate, a bug found in the summary —
 * without touching the measurement. It reads only what `health_sweep.ts` wrote
 * and never opens a browser.
 *
 * WHAT IT REFUSES TO DO
 *
 *  - **Plot an unmeasured value as 0.** The audit writes NaN through a sentinel
 *    (`__nff_NaN__`) precisely so a nonfinite survives JSON; decoding it back to
 *    0 here would redraw a broken run as a healthy flat line. Nonfinite samples
 *    BREAK the trace instead, leaving a visible gap.
 *  - **Compare cells that did not actually differ.** The collision banner is
 *    rendered FIRST, above every chart, when an axis produced one compiled
 *    network across all its values. A report whose headline finding is
 *    "architecture makes no difference" must say so when the reason is that the
 *    architecture never changed.
 *  - **Rank on a single scalar without showing the trace.** A median AC hides
 *    whether the run was climbing or dying; every cell gets its trace.
 */
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import {
  NAN_SENTINEL,
  POS_INF_SENTINEL,
  NEG_INF_SENTINEL,
  GATES,
} from "./health_audit.mjs";
import { collisions, type Cell } from "./health_sweep";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");

/* ── reading what the sweep wrote ─────────────────────────────────────────── */

/** Sentinel-aware number read. Anything that is not a real number is NaN. */
export function num(v: unknown): number {
  if (v === NAN_SENTINEL) return NaN;
  if (v === POS_INF_SENTINEL) return Infinity;
  if (v === NEG_INF_SENTINEL) return -Infinity;
  return typeof v === "number" ? v : NaN;
}

export interface Series {
  readonly t: number[];
  readonly ac: number[];
  readonly r1: number[];
  readonly r2: number[];
  readonly payoff: number[];
  readonly fps: number[];
}

/** Pull the time series out of a cell's raw sample stream. */
export function seriesOf(samples: readonly Record<string, any>[]): Series {
  const s: Series = { t: [], ac: [], r1: [], r2: [], payoff: [], fps: [] };
  for (const x of samples) {
    s.t.push(num(x?.t));
    s.ac.push(num(x?.field?.ac));
    s.r1.push(num(x?.field?.r1));
    s.r2.push(num(x?.field?.r2));
    // adv is legitimately absent on non-adversary pieces: NaN, never 0.
    s.payoff.push(x?.adv ? num(x.adv.payoff) : NaN);
    s.fps.push(num(x?.fps));
  }
  return s;
}

export const median = (xs: readonly number[]): number => {
  const f = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (!f.length) return NaN;
  const m = f.length >> 1;
  return f.length % 2 ? f[m] : (f[m - 1] + f[m]) / 2;
};

/**
 * Least-squares slope of `ys` against `ts`, normalized by |mean y| so it is a
 * RELATIVE trend per second and comparable across metrics of different scale.
 * NaN when there is nothing to fit — a flat 0 would read as "stable".
 */
export function trend(ts: readonly number[], ys: readonly number[]): number {
  const pts = ts.map((t, i) => [t, ys[i]] as const).filter(([t, y]) => Number.isFinite(t) && Number.isFinite(y));
  if (pts.length < 3) return NaN;
  const n = pts.length;
  const mt = pts.reduce((a, [t]) => a + t, 0) / n;
  const my = pts.reduce((a, [, y]) => a + y, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (const [t, y] of pts) {
    sxy += (t - mt) * (y - my);
    sxx += (t - mt) * (t - mt);
  }
  if (sxx === 0) return NaN;
  return sxy / sxx / Math.max(Math.abs(my), 1e-12);
}

export interface CellReport {
  readonly cell: Cell & { dir: string; fingerprint: string | null };
  readonly verdict: { tag: string } & Record<string, unknown>;
  readonly series: Series;
  readonly ac: number;
  readonly r1: number;
  readonly r2: number;
  readonly fps: number;
  readonly acTrend: number;
  readonly r1Trend: number;
}

export function loadRun(runDir: string): {
  manifest: any;
  reports: CellReport[];
} {
  const manifest = JSON.parse(fs.readFileSync(path.join(runDir, "manifest.json"), "utf8"));
  const reports: CellReport[] = [];
  for (const c of manifest.cells) {
    const file = path.join(runDir, c.dir, `${c.pieceKey}.json`);
    if (!fs.existsSync(file)) continue;
    let raw: any;
    try {
      raw = JSON.parse(fs.readFileSync(file, "utf8"));
    } catch {
      continue;
    }
    const series = seriesOf(raw.samples ?? []);
    reports.push({
      cell: c,
      verdict: raw.verdict ?? { tag: "no-signal" },
      series,
      ac: median(series.ac),
      r1: median(series.r1),
      r2: median(series.r2),
      fps: median(series.fps),
      acTrend: trend(series.t, series.ac),
      r1Trend: trend(series.t, series.r1),
    });
  }
  return { manifest, reports };
}

/* ── rendering ────────────────────────────────────────────────────────────── */

export const esc = (s: unknown): string =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!
  );

const fmt = (x: number, d = 3): string =>
  !Number.isFinite(x) ? "—" : Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : x.toFixed(d);

/** Distinct hues, stable per index so a cell keeps its colour across charts. */
const hue = (i: number, n: number): string => `hsl(${Math.round((i * 360) / Math.max(n, 1))} 70% 55%)`;

/**
 * One multi-series SVG. NONFINITE SAMPLES BREAK THE PATH — each contiguous run
 * of finite points is its own polyline, so a gap is visibly a gap rather than a
 * line drawn straight through a value that was never measured.
 */
export function chart(
  series: { label: string; values: number[]; t: number[]; color: string }[],
  opts: { width?: number; height?: number; lo?: number; hi?: number; rule?: number } = {}
): string {
  const W = opts.width ?? 660;
  const H = opts.height ?? 150;
  const pad = { l: 46, r: 8, t: 8, b: 18 };
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  const ts = series.flatMap((s) => s.t).filter(Number.isFinite);
  if (!all.length || !ts.length) {
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="no finite samples"><text x="${W / 2}" y="${H / 2}" text-anchor="middle" class="empty">no finite samples</text></svg>`;
  }
  const lo = opts.lo ?? Math.min(...all);
  const hi = opts.hi ?? Math.max(...all);
  const t0 = Math.min(...ts);
  const t1 = Math.max(...ts);
  const span = Math.max(hi - lo, 1e-30);
  const x = (t: number) => pad.l + ((t - t0) / Math.max(t1 - t0, 1e-9)) * (W - pad.l - pad.r);
  const y = (v: number) => H - pad.b - ((v - lo) / span) * (H - pad.t - pad.b);

  const paths = series
    .map((s) => {
      const runs: string[] = [];
      let cur: string[] = [];
      for (let i = 0; i < s.values.length; i++) {
        if (Number.isFinite(s.values[i]) && Number.isFinite(s.t[i])) {
          cur.push(`${x(s.t[i]).toFixed(1)},${y(s.values[i]).toFixed(1)}`);
        } else if (cur.length) {
          runs.push(cur.join(" "));
          cur = [];
        }
      }
      if (cur.length) runs.push(cur.join(" "));
      return runs
        .filter((r) => r.includes(" "))
        .map((r) => `<polyline points="${r}" fill="none" stroke="${s.color}" stroke-width="1.6"/>`)
        .join("");
    })
    .join("");

  const ruleLine =
    opts.rule !== undefined && opts.rule >= lo && opts.rule <= hi
      ? `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(opts.rule).toFixed(1)}" y2="${y(opts.rule).toFixed(1)}" class="rule"/>`
      : "";

  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(series.map((s) => s.label).join(", "))}">
  <line x1="${pad.l}" x2="${pad.l}" y1="${pad.t}" y2="${H - pad.b}" class="axis"/>
  <line x1="${pad.l}" x2="${W - pad.r}" y1="${H - pad.b}" y2="${H - pad.b}" class="axis"/>
  <text x="${pad.l - 5}" y="${pad.t + 8}" text-anchor="end" class="tick">${fmt(hi, 2)}</text>
  <text x="${pad.l - 5}" y="${H - pad.b}" text-anchor="end" class="tick">${fmt(lo, 2)}</text>
  <text x="${W - pad.r}" y="${H - 4}" text-anchor="end" class="tick">${Math.round(t1)}s</text>
  ${ruleLine}${paths}
</svg>`;
}

export function render(manifest: any, reports: CellReport[]): string {
  const spec = manifest.spec;
  const axisNames = Object.keys(spec.axes);
  const done = manifest.cells.filter((c: any) => c.done).length;
  const bad = collisions(
    manifest.cells.map((c: any) => ({ cell: c, fingerprint: c.fingerprint ?? null }))
  );

  // Ranked by structure (AC) descending — the thing the whole experiment is
  // trying to produce — with the collapse metrics alongside, because a high AC
  // with R1 pinned at 1 is a laminar field with a lot of energy, not a win.
  const ranked = [...reports].sort((a, b) => (Number.isFinite(b.ac) ? b.ac : -Infinity) - (Number.isFinite(a.ac) ? a.ac : -Infinity));
  const colored = new Map(reports.map((r, i) => [r.cell.id, hue(i, reports.length)]));

  const banner = bad.length
    ? `<section class="alert">
  <h2>⚠ ${bad.length} axis collision${bad.length > 1 ? "s" : ""} — these comparisons are not real</h2>
  <p>Every listed value compiled the <strong>same network</strong>. The piece is almost certainly not <code>archEditable</code>, so the app ignored the parameter. Any conclusion drawn along this axis on this piece is a comparison of a run against itself.</p>
  <ul>${bad
    .map(
      (c) =>
        `<li><code>${esc(c.axis)}</code> on <code>${esc(c.pieceKey)}</code>: ${esc(c.values.join(", "))} → all <code>${esc(c.fingerprint)}</code></li>`
    )
    .join("")}</ul>
</section>`
    : `<section class="ok-banner">✓ every swept axis produced a distinct compiled network — the comparisons below are between different things.</section>`;

  const overlay = (label: string, key: keyof Series, o: { lo?: number; hi?: number; rule?: number; note: string }) => `
<section class="panel">
  <h3>${esc(label)}</h3>
  <p class="note">${o.note}</p>
  ${chart(
    reports.map((r) => ({
      label: r.cell.id,
      values: r.series[key] as number[],
      t: r.series.t,
      color: colored.get(r.cell.id)!,
    })),
    { lo: o.lo, hi: o.hi, rule: o.rule }
  )}
</section>`;

  return `<title>${esc(spec.name)} · sweep report</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b68;--line:#dededa;--card:#fff;--accent:#2f6f4f;--warn:#8a3324;--warnbg:#fdf0ec}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--fg:#eeeeec;--dim:#9a9a96;--line:#2e2e2c;--card:#1c1c1a;--accent:#6fbf95;--warn:#e8836b;--warnbg:#2a1b17}}
:root[data-theme="dark"]{--bg:#141413;--fg:#eeeeec;--dim:#9a9a96;--line:#2e2e2c;--card:#1c1c1a;--accent:#6fbf95;--warn:#e8836b;--warnbg:#2a1b17}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:32px 24px 64px}
main{max-width:1080px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 8px}h3{font-size:14px;margin:0 0 4px;letter-spacing:.02em;text-transform:uppercase;color:var(--dim)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}
.sub{color:var(--dim);margin:0 0 24px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}
.note{color:var(--dim);margin:0 0 10px;font-size:13px}
.alert{background:var(--warnbg);border:1px solid var(--warn);border-radius:10px;padding:16px;margin:0 0 20px}
.alert h2{color:var(--warn)}.alert ul{margin:8px 0 0;padding-left:20px}
.ok-banner{color:var(--accent);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:0 0 20px;background:var(--card)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num{text-align:right;font-family:ui-monospace,Menlo,monospace}
.pass{color:var(--accent);font-weight:600}.fail{color:var(--warn);font-weight:600}
.swatch{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:7px;vertical-align:middle}
.chart{width:100%;height:auto;display:block}
.axis{stroke:var(--line);stroke-width:1}.rule{stroke:var(--dim);stroke-width:1;stroke-dasharray:3 3;opacity:.7}
.tick,.empty{fill:var(--dim);font:10px ui-monospace,Menlo,monospace}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h4{margin:0 0 2px;font-size:12px;font-family:ui-monospace,Menlo,monospace;overflow-wrap:anywhere}
.card .meta{color:var(--dim);font-size:11px;margin:0 0 8px}
</style>
<main>
<h1>${esc(spec.name)}</h1>
<p class="sub">
  ${done}/${manifest.cells.length} cells${manifest.complete ? "" : " · <strong>INCOMPLETE</strong>"} ·
  ${esc(spec.durationSec)}s each · axes ${axisNames.map((a) => `<code>${esc(a)}</code>[${spec.axes[a].length}]`).join(" × ")} ·
  base <code>${esc(manifest.base)}</code>
</p>

${banner}

<section class="panel">
  <h3>Cells, ranked by structure (AC)</h3>
  <p class="note">AC is the spatially varying part of the field — what the experiment is trying to grow. Read it WITH R1: high AC at R1→1 is a strong laminar field, not a win. R1 and R2 are direction order, 0 = isotropic, 1 = fully aligned; R2 catches the ± counter-streaming escape that R1 alone scores as healthy. Trend columns are relative slope per second over the whole rollout.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>cell</th>${axisNames.filter((a) => a !== "piece").map((a) => `<th>${esc(a)}</th>`).join("")}<th>verdict</th><th class="num">AC</th><th class="num">AC trend/s</th><th class="num">R1</th><th class="num">R1 trend/s</th><th class="num">R2</th><th class="num">fps</th><th>arch fingerprint</th></tr></thead>
    <tbody>
    ${ranked
      .map(
        (r) => `<tr>
      <td><span class="swatch" style="background:${colored.get(r.cell.id)}"></span><code>${esc(r.cell.pieceKey)}</code></td>
      ${axisNames.filter((a) => a !== "piece").map((a) => `<td><code>${esc(r.cell.params[a] ?? "—")}</code></td>`).join("")}
      <td class="${r.verdict.tag === "healthy" ? "pass" : "fail"}">${esc(r.verdict.tag)}</td>
      <td class="num">${fmt(r.ac)}</td>
      <td class="num">${fmt(r.acTrend, 4)}</td>
      <td class="num">${fmt(r.r1)}</td>
      <td class="num">${fmt(r.r1Trend, 4)}</td>
      <td class="num">${fmt(r.r2)}</td>
      <td class="num">${fmt(r.fps, 0)}</td>
      <td><code>${esc(r.cell.fingerprint ?? "unmeasured")}</code></td>
    </tr>`
      )
      .join("")}
    </tbody>
  </table></div>
</section>

${overlay("Direction order R1 — collapse to one heading", "r1", {
  lo: 0,
  hi: 1,
  rule: GATES.r1Laminar,
  note: `Fixed 0–1. Low is healthy. The dashed rule is the audit's laminar gate (${GATES.r1Laminar}); a trace crossing it and staying is a directional collapse.`,
})}
${overlay("Nematic order R2 — the ± counter-streaming escape", "r2", {
  lo: 0,
  hi: 1,
  note: "Fixed 0–1. A cell with low R1 and high R2 has split into counter-streaming sheets: R1 alone calls it healthy and it looks just as laminar on screen.",
})}
${overlay("Structure AC — the spatially varying mode", "ac", {
  rule: GATES.acDead,
  note: `Auto scale. The dashed rule is the dead-field gate (${GATES.acDead}); below it only the constant mode is left and the cloud is being pushed by one global vector.`,
})}
${overlay("Shared payoff", "payoff", {
  note: "Auto scale. Gaps are cells with no adversary — absent, never zero.",
})}

<section class="panel">
  <h3>Per cell</h3>
  <p class="note">R1 (solid, 0–1 scale) against AC (rescaled to the same box). The shape that matters: R1 climbing while AC falls is the collapse in progress.</p>
  <div class="cards">
  ${ranked
    .map((r) => {
      const acMax = Math.max(...r.series.ac.filter(Number.isFinite), 1e-12);
      return `<div class="card">
      <h4><span class="swatch" style="background:${colored.get(r.cell.id)}"></span>${esc(r.cell.id)}</h4>
      <p class="meta">${esc(r.verdict.tag)} · AC ${fmt(r.ac)} · R1 ${fmt(r.r1)} · R2 ${fmt(r.r2)}</p>
      ${chart(
        [
          { label: "R1", values: r.series.r1, t: r.series.t, color: colored.get(r.cell.id)! },
          { label: "AC", values: r.series.ac.map((v) => v / acMax), t: r.series.t, color: "var(--dim)" },
        ],
        { width: 320, height: 96, lo: 0, hi: 1, rule: GATES.r1Laminar }
      )}
    </div>`;
    })
    .join("")}
  </div>
</section>
</main>`;
}

/* ── self-test ────────────────────────────────────────────────────────────── */

export function selfTest(): number {
  let failures = 0;
  const ok = (c: boolean, m: string) => {
    console.log(`${c ? "  ok  " : " FAIL "} ${m}`);
    if (!c) failures++;
  };

  console.log("\n--- 1. an unmeasured value is never plotted as a number ---");
  {
    const s = seriesOf([
      { t: 0, fps: 60, field: { ac: 0.2, r1: 0.1, r2: 0.2 }, adv: { payoff: 0.6 } },
      { t: 1, fps: 60, field: { ac: NAN_SENTINEL, r1: 0.1, r2: 0.2 }, adv: null },
      { t: 2, fps: 60, field: null, adv: null },
    ]);
    ok(Number.isNaN(s.ac[1]), "a NaN sentinel decodes to NaN, not to the string and not to 0");
    ok(Number.isNaN(s.payoff[1]), "adv:null → payoff NaN (absent, not a zero-payoff game)");
    ok(Number.isNaN(s.ac[2]), "field:null → NaN");
    ok(s.ac[0] === 0.2, "a real number survives unchanged");
  }

  console.log("\n--- 2. a nonfinite sample BREAKS the trace instead of bridging it ---");
  {
    // The bug this guards: joining across a gap draws a straight line through
    // values that were never measured, which is exactly how a broken run looks
    // like a smoothly converging one.
    const svg = chart([
      { label: "x", values: [0, NaN, 1], t: [0, 1, 2], color: "#000" },
    ]);
    ok(
      (svg.match(/<polyline/g) ?? []).length === 0,
      "two isolated points either side of a gap produce NO bridging polyline"
    );
    const joined = chart([
      { label: "x", values: [0, 0.5, 1], t: [0, 1, 2], color: "#000" },
    ]);
    ok(
      (joined.match(/<polyline/g) ?? []).length === 1,
      "…while three finite points produce exactly one polyline"
    );
    const twoRuns = chart([
      { label: "x", values: [0, 0.2, NaN, 0.8, 1], t: [0, 1, 2, 3, 4], color: "#000" },
    ]);
    ok(
      (twoRuns.match(/<polyline/g) ?? []).length === 2,
      "a gap in the middle of a long series yields TWO separate polylines"
    );
    ok(
      chart([{ label: "x", values: [NaN, NaN], t: [0, 1], color: "#000" }]).includes("no finite samples"),
      "an all-NaN series renders an explicit 'no finite samples', not an empty box"
    );
  }

  console.log("\n--- 3. trend is signed, relative, and refuses to guess ---");
  {
    const t = [0, 1, 2, 3, 4];
    ok(trend(t, [1, 2, 3, 4, 5]) > 0, "a rising series trends positive");
    ok(trend(t, [5, 4, 3, 2, 1]) < 0, "a falling series trends negative");
    ok(Math.abs(trend(t, [3, 3, 3, 3, 3])) < 1e-12, "a flat series trends ~0");
    ok(Number.isNaN(trend([0, 1], [1, 2])), "fewer than 3 finite points → NaN, not 0");
    ok(Number.isNaN(trend(t, [NaN, NaN, NaN, NaN, NaN])), "all-NaN → NaN");
    // Relative normalization: same shape at 1000x scale must trend the same.
    const a = trend(t, [1, 2, 3, 4, 5]);
    const b = trend(t, [1000, 2000, 3000, 4000, 5000]);
    ok(Math.abs(a - b) < 1e-9, "the trend is scale-invariant (relative slope), so metrics compare");
  }

  console.log("\n--- 4. config values cannot inject markup into the report ---");
  {
    // Axis values come from a JSON file a human wrote; the report is a page a
    // human opens. Unescaped, `advLoss=<img onerror=...>` becomes live markup.
    ok(esc(`<script>x</script>`) === "&lt;script&gt;x&lt;/script&gt;", "angle brackets escape");
    ok(esc(`a"b'c&d`) === "a&quot;b&#39;c&amp;d", "quotes and ampersands escape");
  }

  console.log("\n--- 5. median ignores nonfinite rather than poisoning ---");
  {
    ok(median([1, NaN, 3]) === 2, "a NaN in the middle does not poison the median");
    ok(Number.isNaN(median([NaN, NaN])), "all-NaN → NaN");
    ok(median([4, 1, 3, 2]) === 2.5, "even count averages the two middles");
  }

  console.log(failures === 0 ? "\nREPORT SELF-TEST PASS" : `\n${failures} REPORT SELF-TEST FAILURE(S)`);
  return failures;
}

/* ── CLI ──────────────────────────────────────────────────────────────────── */

const INVOKED_DIRECTLY =
  !!process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (INVOKED_DIRECTLY) {
  const arg = process.argv[2];
  if (arg === "--self-test") process.exit(selfTest() === 0 ? 0 : 1);
  if (!arg) {
    console.error("usage: bun tools/health_report.ts <sweepRunDir> [out.html]");
    process.exit(2);
  }
  const runDir = path.resolve(ROOT, arg);
  const { manifest, reports } = loadRun(runDir);
  if (!reports.length) {
    console.error(`no completed cells found in ${runDir}`);
    process.exit(2);
  }
  const out = process.argv[3] ? path.resolve(ROOT, process.argv[3]) : path.join(runDir, "report.html");
  fs.writeFileSync(out, render(manifest, reports));
  const bad = collisions(manifest.cells.map((c: any) => ({ cell: c, fingerprint: c.fingerprint ?? null })));
  console.log(
    `report: ${out}\n${reports.length} cells` +
      (bad.length ? ` · ⚠ ${bad.length} AXIS COLLISION(S) — see the banner at the top` : "")
  );
  process.exit(0);
}
