/**
 * Sequential shared-W pointwise batch benchmark matrix.
 *
 * Runs tools/clip/pointwise_batch_bench.ts one shape/batch at a time and
 * summarizes whether shared-W beats the normal z-batch pointwise kernel.
 *
 *   BATCHES=3 STEPS=8,10,57,59,111,117 RUNS=20 WARMUP=5 bun tools/clip/pointwise_batch_matrix.ts
 *   BATCHES=2,3 TRIALS=2 RUNS=20 WARMUP=5 bun tools/clip/pointwise_batch_matrix.ts
 */
import { spawnSync } from "node:child_process";

interface Config {
  step: number;
  batch: number;
}

interface Result extends Config {
  trial: number;
  shape: string;
  residual: boolean;
  zMs: number;
  sharedMs: number;
  ratio: number;
  parityRel: number;
}

const STEPS = parseInts(process.env.STEPS ?? "8,10,57,59,111,117", "STEPS");
const BATCHES = parseInts(process.env.BATCHES ?? "2,3", "BATCHES");
const TRIALS = envInt("TRIALS", 1);
const RUNS = envInt("RUNS", 20);
const WARMUP = envInt("WARMUP", 5);
const JSON_OUT = process.env.JSON === "1";
const WIN_RATIO = envNumber("WIN_RATIO", 0.97);
const LOSS_RATIO = envNumber("LOSS_RATIO", 1.03);

function envInt(name: string, fallback: number): number {
  const n = Number(process.env[name]);
  return Number.isFinite(n) ? n | 0 : fallback;
}

function envNumber(name: string, fallback: number): number {
  const n = Number(process.env[name]);
  return Number.isFinite(n) ? n : fallback;
}

function parseInts(src: string, name: string): number[] {
  const vals = src
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((n) => Number.isFinite(n))
    .map((n) => n | 0);
  if (!vals.length) throw new Error(`pointwise_batch_matrix: ${name} produced no values`);
  return vals;
}

function configsForTrial(trial: number): Config[] {
  const configs: Config[] = [];
  for (const step of STEPS) {
    for (const batch of BATCHES) configs.push({ step, batch });
  }
  return trial % 2 === 0 ? configs : configs.reverse();
}

function parseBench(output: string, config: Config, trial: number): Result {
  const header = output.match(
    /pointwise batch: B=(\d+), step=(\d+), ([^@]+@\d+x\d+), residual=(true|false), runs=/
  );
  const z = output.match(/z-batch\s*:\s*([\d.]+) ms/);
  const shared = output.match(/shared-W:\s*([\d.]+) ms/);
  const parity = output.match(/final parity: relLinf=([\deE.+-]+)/);
  if (!header || !z || !shared || !parity) {
    throw new Error(`pointwise_batch_matrix: could not parse bench output\n${output}`);
  }
  const zMs = Number(z[1]);
  const sharedMs = Number(shared[1]);
  return {
    ...config,
    trial,
    shape: header[3].trim(),
    residual: header[4] === "true",
    zMs,
    sharedMs,
    ratio: sharedMs / zMs,
    parityRel: Number(parity[1]),
  };
}

function runOne(config: Config, trial: number): Result {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    STEP_INDEX: String(config.step),
    BATCH: String(config.batch),
    RUNS: String(RUNS),
    WARMUP: String(WARMUP),
  };
  delete env.STEPS;
  delete env.BATCHES;
  delete env.TRIALS;
  delete env.JSON;
  delete env.WIN_RATIO;
  delete env.LOSS_RATIO;

  const child = spawnSync("bun", ["tools/clip/pointwise_batch_bench.ts"], {
    cwd: process.cwd(),
    env,
    encoding: "utf8",
    maxBuffer: 1024 * 1024 * 16,
  });
  const output = `${child.stdout ?? ""}${child.stderr ?? ""}`;
  if (child.status !== 0) {
    throw new Error(`pointwise_batch_matrix: child failed for step=${config.step} B=${config.batch}\n${output}`);
  }
  return parseBench(output, config, trial);
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

function verdict(ratio: number): "win" | "flat" | "loss" {
  if (ratio <= WIN_RATIO) return "win";
  if (ratio >= LOSS_RATIO) return "loss";
  return "flat";
}

function fmt(n: number): string {
  return n.toFixed(3).padStart(8);
}

const results: Result[] = [];
if (!JSON_OUT) {
  console.log(
    `pointwise shared-W matrix: steps=${STEPS.join(",")} batches=${BATCHES.join(",")} ` +
      `trials=${TRIALS} runs=${RUNS} warmup=${WARMUP}`
  );
}

for (let trial = 0; trial < TRIALS; trial++) {
  for (const config of configsForTrial(trial)) {
    const result = runOne(config, trial);
    results.push(result);
    if (!JSON_OUT) {
      console.log(
        `trial ${trial} step=${String(config.step).padStart(3)} B=${config.batch} ` +
          `${result.shape} z=${result.zMs.toFixed(3)} shared=${result.sharedMs.toFixed(3)} ` +
          `ratio=${result.ratio.toFixed(3)} ${verdict(result.ratio)} parity=${result.parityRel.toExponential(1)}`
      );
    }
  }
}

if (JSON_OUT) {
  console.log(JSON.stringify({ trials: TRIALS, runs: RUNS, warmup: WARMUP, results }, null, 2));
} else {
  console.log("\nSummary:");
  console.log("step batch shape                    residual        z   shared    ratio  verdict");
  for (const step of STEPS) {
    for (const batch of BATCHES) {
      const rows = results.filter((r) => r.step === step && r.batch === batch);
      if (!rows.length) continue;
      const ratio = median(rows.map((r) => r.ratio));
      const z = rows.map((r) => r.zMs);
      const shared = rows.map((r) => r.sharedMs);
      const shape = rows[0].shape.padEnd(24).slice(0, 24);
      const residual = rows[0].residual ? "yes" : "no";
      console.log(
        `${String(step).padStart(4)} ${String(batch).padStart(5)} ${shape} ` +
          `${residual.padStart(8)} ${fmt(median(z))} ${fmt(median(shared))} ` +
          `${ratio.toFixed(3).padStart(8)} ${verdict(ratio)}` +
          (rows.length > 1 ? ` [${min(rows.map((r) => r.ratio)).toFixed(3)},${max(rows.map((r) => r.ratio)).toFixed(3)}]` : "")
      );
    }
  }
}
