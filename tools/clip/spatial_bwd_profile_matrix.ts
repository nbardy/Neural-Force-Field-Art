/**
 * Sequential spatial_bwd dispatch profile matrix.
 *
 * Wraps tools/clip/dispatch_profile.ts in CSV mode, filters spatial_bwd rows,
 * and aggregates by exact generated label. This is a ranking gate for spatial
 * backward variants; it does not use timestamp queries, so treat medians as
 * directional and compare configs in the same run.
 *
 *   TRIALS=2 BATCHES=1,3 RUNS=3 WARMUP=1 bun tools/clip/spatial_bwd_profile_matrix.ts
 */
import { spawnSync } from "node:child_process";

interface Row {
  trial: number;
  batch: number;
  index: number;
  label: string;
  workgroups: number;
  ms: number;
}

const TRIALS = envInt("TRIALS", 2);
const RUNS = envInt("RUNS", 3);
const WARMUP = envInt("WARMUP", 1);
const BATCHES = parseInts(process.env.BATCHES ?? "1,3", "BATCHES");
const TOP = envInt("TOP", 16);
const JSON_OUT = process.env.JSON === "1";

function envInt(name: string, fallback: number): number {
  const n = Number(process.env[name]);
  return Number.isFinite(n) ? n | 0 : fallback;
}

function parseInts(src: string, name: string): number[] {
  const vals = src
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((n) => Number.isFinite(n))
    .map((n) => n | 0);
  if (!vals.length) throw new Error(`spatial_bwd_profile_matrix: ${name} produced no values`);
  return vals;
}

function splitCsv(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        quoted = !quoted;
      }
    } else if (ch === "," && !quoted) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function runProfile(batch: number, trial: number): Row[] {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    CSV: "1",
    MODE: "train",
    BATCH: String(batch),
    RUNS: String(RUNS),
    WARMUP: String(WARMUP),
  };
  delete env.BATCHES;
  delete env.TRIALS;
  delete env.TOP;
  delete env.JSON;

  const child = spawnSync("bun", ["tools/clip/dispatch_profile.ts"], {
    cwd: process.cwd(),
    env,
    encoding: "utf8",
    maxBuffer: 1024 * 1024 * 16,
  });
  const output = `${child.stdout ?? ""}${child.stderr ?? ""}`;
  if (child.status !== 0) {
    throw new Error(`spatial_bwd_profile_matrix: dispatch_profile failed for B=${batch}\n${output}`);
  }

  const rows: Row[] = [];
  for (const line of output.trim().split(/\r?\n/).slice(1)) {
    if (!line.trim()) continue;
    const cols = splitCsv(line);
    if (cols[2] !== "spatial_bwd") continue;
    rows.push({
      trial,
      batch,
      index: Number(cols[0]),
      label: cols[3],
      workgroups: Number(cols[7]),
      ms: Number(cols[8]),
    });
  }
  return rows;
}

function median(xs: number[]): number {
  const ys = xs.slice().sort((a, b) => a - b);
  if (!ys.length) return 0;
  const mid = Math.floor(ys.length / 2);
  return ys.length % 2 ? ys[mid] : (ys[mid - 1] + ys[mid]) * 0.5;
}

function min(xs: number[]): number {
  return xs.reduce((a, b) => Math.min(a, b), Number.POSITIVE_INFINITY);
}

function max(xs: number[]): number {
  return xs.reduce((a, b) => Math.max(a, b), Number.NEGATIVE_INFINITY);
}

function fmt(n: number): string {
  return n.toFixed(3).padStart(8);
}

const rows: Row[] = [];
if (!JSON_OUT) {
  console.log(
    `spatial_bwd profile matrix: batches=${BATCHES.join(",")} trials=${TRIALS} ` +
      `runs=${RUNS} warmup=${WARMUP}`
  );
}

for (let trial = 0; trial < TRIALS; trial++) {
  const order = trial % 2 === 0 ? BATCHES : BATCHES.slice().reverse();
  for (const batch of order) {
    const got = runProfile(batch, trial);
    rows.push(...got);
    if (!JSON_OUT) {
      const total = got.reduce((sum, r) => sum + r.ms, 0);
      console.log(`trial ${trial} B=${batch}: ${got.length} spatial_bwd dispatches, sum=${total.toFixed(3)} ms`);
    }
  }
}

if (JSON_OUT) {
  console.log(JSON.stringify({ trials: TRIALS, runs: RUNS, warmup: WARMUP, rows }, null, 2));
} else {
  console.log("\nSummary:");
  for (const batch of BATCHES) {
    const batchRows = rows.filter((r) => r.batch === batch);
    const byLabel = new Map<string, Row[]>();
    for (const row of batchRows) {
      const group = byLabel.get(row.label) ?? [];
      group.push(row);
      byLabel.set(row.label, group);
    }
    const summaries = [...byLabel.entries()]
      .map(([label, rs]) => {
        const ms = rs.map((r) => r.ms);
        const med = median(ms);
        return {
          label,
          count: rs.length / TRIALS,
          workgroups: median(rs.map((r) => r.workgroups)),
          med,
          min: min(ms),
          max: max(ms),
        };
      })
      .sort((a, b) => b.med - a.med)
      .slice(0, TOP);

    const totalByTrial = Array.from({ length: TRIALS }, (_unused, trial) =>
      batchRows.filter((r) => r.trial === trial).reduce((sum, r) => sum + r.ms, 0)
    );
    console.log(`\nB=${batch} total spatial_bwd median sum: ${median(totalByTrial).toFixed(3)} ms`);
    console.log("median ms   [min,max]   count  workgroups  label");
    for (const s of summaries) {
      console.log(
        `${fmt(s.med)} [${s.min.toFixed(3)},${s.max.toFixed(3)}] ` +
          `${String(s.count).padStart(5)} ${String(s.workgroups).padStart(10)}  ${s.label}`
      );
    }
  }
}
