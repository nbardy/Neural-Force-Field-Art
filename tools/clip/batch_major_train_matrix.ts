/**
 * Sequential full batch-major CLIP train matrix.
 *
 * Runs tools/clip/batch_major_train_bench.ts one config at a time and parses
 * the final full-chain batch-major wall time. This is intentionally process
 * based to preserve the same parity checks as the normal bench and avoid
 * parallel GPU contention.
 *
 *   TRIALS=2 CONFIGS='base=;early=8,10;stem=stem;gelu=gelu' bun tools/clip/batch_major_train_matrix.ts
 */
import { spawnSync } from "node:child_process";

interface Config {
  name: string;
  steps: string;
  stemSpatialBwd: boolean;
  fusePointwiseGeluForward: boolean;
}

interface Result extends Config {
  trial: number;
  separateMs: number;
  batchMs: number;
  msPerImage: number;
}

const TRIALS = envInt("TRIALS", 2);
const BATCH = envInt("BATCH", 3);
const RUNS = envInt("RUNS", 3);
const WARMUP = envInt("WARMUP", 3);
const CONFIGS = parseConfigs(process.env.CONFIGS ?? "base=;early=8,10;candidates=8,10,111,115");
const JSON_OUT = process.env.JSON === "1";

function envInt(name: string, fallback: number): number {
  const n = Number(process.env[name]);
  return Number.isFinite(n) ? n | 0 : fallback;
}

function parseConfigs(src: string): Config[] {
  const configs = src
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [nameRaw, raw = ""] = part.split("=");
      const name = nameRaw.trim();
      if (!name) throw new Error(`batch_major_train_matrix: bad config '${part}'`);
      const tokens = raw.split(",").map((token) => token.trim()).filter(Boolean);
      const stemSpatialBwd = tokens.includes("stem");
      const fusePointwiseGeluForward = tokens.includes("gelu");
      const steps = tokens.filter((token) => token !== "stem" && token !== "gelu").join(",");
      return { name, steps, stemSpatialBwd, fusePointwiseGeluForward };
    });
  if (!configs.length) throw new Error("batch_major_train_matrix: CONFIGS produced no configs");
  return configs;
}

function parseMs(pattern: RegExp, output: string, label: string): number {
  const m = output.match(pattern);
  if (!m) throw new Error(`batch_major_train_matrix: could not parse ${label}\n${output}`);
  return Number(m[1]);
}

function runOne(config: Config, trial: number): Result {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    BATCH: String(BATCH),
    RUNS: String(RUNS),
    WARMUP: String(WARMUP),
    SHARED_W_FWD_STEPS: config.steps,
    STEM_SPATIAL_BWD: config.stemSpatialBwd ? "1" : "0",
    FUSE_PW_GELU: config.fusePointwiseGeluForward ? "1" : "0",
  };
  delete env.CONFIGS;
  delete env.TRIALS;
  delete env.JSON;

  const child = spawnSync("bun", ["tools/clip/batch_major_train_bench.ts"], {
    cwd: process.cwd(),
    env,
    encoding: "utf8",
    maxBuffer: 1024 * 1024 * 16,
  });
  const output = `${child.stdout ?? ""}${child.stderr ?? ""}`;
  if (child.status !== 0) {
    throw new Error(`batch_major_train_matrix: child failed for ${config.name}\n${output}`);
  }
  const separateMs = parseMs(/separate\s+:\s+([\d.]+) ms\/batch/, output, "separate ms");
  const batchMs = parseMs(/batch-major:\s+([\d.]+) ms\/batch/, output, "batch-major ms");
  const msPerImage = parseMs(/batch-major:\s+[\d.]+ ms\/batch · ([\d.]+) ms\/image/, output, "batch-major ms/image");
  return { ...config, trial, separateMs, batchMs, msPerImage };
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
  return n.toFixed(2).padStart(8);
}

const results: Result[] = [];
if (!JSON_OUT) {
  console.log(
    `batch-major train matrix: configs=${CONFIGS.map((c) => c.name).join(",")} ` +
      `trials=${TRIALS} batch=${BATCH} runs=${RUNS} warmup=${WARMUP}`
  );
}

for (let trial = 0; trial < TRIALS; trial++) {
  const order = trial % 2 === 0 ? CONFIGS : CONFIGS.slice().reverse();
  for (const config of order) {
    const result = runOne(config, trial);
    results.push(result);
    if (!JSON_OUT) {
      console.log(
        `trial ${trial} ${config.name.padEnd(10)} steps=${config.steps || "-"} ` +
          `${config.stemSpatialBwd ? "stem=1 " : ""}` +
          `${config.fusePointwiseGeluForward ? "gelu=1 " : ""}` +
          `batch=${result.batchMs.toFixed(2)} ms (${result.msPerImage.toFixed(2)} ms/img) ` +
          `separate=${result.separateMs.toFixed(2)}`
      );
    }
  }
}

if (JSON_OUT) {
  console.log(JSON.stringify({ trials: TRIALS, batch: BATCH, runs: RUNS, warmup: WARMUP, results }, null, 2));
} else {
  console.log("\nSummary:");
  console.log("config       steps                 stem gelu   batch med [min,max]   img med");
  for (const config of CONFIGS) {
    const rows = results.filter((r) => r.name === config.name);
    const batchMs = rows.map((r) => r.batchMs);
    const imgMs = rows.map((r) => r.msPerImage);
    console.log(
      `${config.name.padEnd(11)} ${(config.steps || "-").padEnd(21).slice(0, 21)} ` +
        `${(config.stemSpatialBwd ? "yes" : "no").padStart(4)} ` +
        `${(config.fusePointwiseGeluForward ? "yes" : "no").padStart(4)} ` +
        `${fmt(median(batchMs))} [${min(batchMs).toFixed(2)},${max(batchMs).toFixed(2)}] ${fmt(median(imgMs))}`
    );
  }
}
