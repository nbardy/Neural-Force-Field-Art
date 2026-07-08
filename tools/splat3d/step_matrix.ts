/**
 * Sequential 3D splat optimizer benchmark matrix.
 *
 * Runs tools/splat3d/step_bench.ts one config at a time and summarizes medians.
 * This is intentionally process-based because it exercises the same path we use
 * during manual ablations while preventing accidental parallel GPU contention.
 *
 *   TRIALS=2 CONFIGS=3:1,3:3 RUNS=4 WARMUP=2 bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,rasterpass=3:3:rasterpass bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,viewlane=3:3:viewlane bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,viewbwd=3:3:viewbwd bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,resbwd=3:3:resbwd bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,cap1024=3:3:cap1024 bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=3:1,3:3,9:1,9:3,9:9 bun tools/splat3d/step_matrix.ts
 */
import { spawnSync } from "node:child_process";

interface Config {
  label: string;
  views: number;
  clipBatch: number;
  fuseGeluBwdIntoPw: boolean;
  fuseResidualBwdIntoPw: boolean;
  singlePassRasterForward: boolean | null;
  viewLaneRasterForward: boolean | null;
  viewLaneRasterBackward: boolean | null;
  cap: number | null;
}

interface TrialResult extends Config {
  trial: number;
  normal: number;
  profileTotal: number;
  rasterFwd: number;
  rasterReplay: number;
  rasterBwd: number;
  clipFwd: number;
  clipBwd: number;
  clipBatchMs: number;
  adam: number;
  display: number;
}

const TRIALS = envInt("TRIALS", 2);
const RUNS = envInt("RUNS", 5);
const WARMUP = envInt("WARMUP", 3);
const SEED = envInt("SEED", 1);
const G = process.env.G;
const CONFIGS = parseConfigs(process.env.CONFIGS ?? "3:1,3:3,9:1,9:3,9:9");
const JSON_OUT = process.env.JSON === "1";

function envInt(name: string, fallback: number): number {
  const n = Number(process.env[name]);
  return Number.isFinite(n) ? n | 0 : fallback;
}

function parseConfigs(src: string): Config[] {
  const configs = src
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [labelRaw, bodyRaw] = part.includes("=") ? part.split("=") : ["", part];
      const [viewsRaw, batchRaw, ...tokens] = bodyRaw.split(/[:x]/);
      const views = Number(viewsRaw);
      const clipBatch = Number(batchRaw);
      if (!Number.isFinite(views) || !Number.isFinite(clipBatch)) {
        throw new Error(`step_matrix: bad config '${part}', expected views:clipBatch`);
      }
      const fuseGeluBwdIntoPw = tokens.includes("gelubwd");
      const fuseResidualBwdIntoPw = tokens.includes("resbwd");
      const singlePassRasterForward = tokens.includes("rasterpass")
        ? true
        : tokens.includes("norasterpass")
          ? false
          : null;
      const viewLaneRasterForward = tokens.includes("viewlane")
        ? true
        : tokens.includes("noviewlane")
          ? false
          : null;
      const viewLaneRasterBackward = tokens.includes("viewbwd")
        ? true
        : tokens.includes("noviewbwd")
          ? false
          : null;
      const capToken = tokens.find((token) => /^cap\d+$/.test(token));
      const cap = capToken ? Number(capToken.slice(3)) : null;
      const suffix = tokens.length ? `:${tokens.join(":")}` : "";
      const label = labelRaw.trim() || `${views | 0}:${clipBatch | 0}${suffix}`;
      return {
        label,
        views: views | 0,
        clipBatch: clipBatch | 0,
        fuseGeluBwdIntoPw,
        fuseResidualBwdIntoPw,
        singlePassRasterForward,
        viewLaneRasterForward,
        viewLaneRasterBackward,
        cap,
      };
    });
  if (!configs.length) throw new Error("step_matrix: CONFIGS produced no configs");
  return configs;
}

function parseNumber(pattern: RegExp, output: string, name: string): number {
  const m = output.match(pattern);
  if (!m) throw new Error(`step_matrix: could not parse ${name}\n${output}`);
  return Number(m[1]);
}

function parseProfile(output: string): Omit<TrialResult, "trial" | "views" | "clipBatch" | "normal"> {
  const m = output.match(
    /profile: total=([\d.]+) ms rasterFwd=([\d.]+) rasterReplay=([\d.]+) rasterBwd=([\d.]+) clipFwd=([\d.]+) clipBwd=([\d.]+) clipBatch=([\d.]+) adam=([\d.]+) display=([\d.]+)/
  );
  if (!m) throw new Error(`step_matrix: could not parse profile\n${output}`);
  return {
    profileTotal: Number(m[1]),
    rasterFwd: Number(m[2]),
    rasterReplay: Number(m[3]),
    rasterBwd: Number(m[4]),
    clipFwd: Number(m[5]),
    clipBwd: Number(m[6]),
    clipBatchMs: Number(m[7]),
    adam: Number(m[8]),
    display: Number(m[9]),
  };
}

function runTrial(config: Config, trial: number): TrialResult {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    VIEWS: String(config.views),
    CLIP_BATCH: String(config.clipBatch),
    RUNS: String(RUNS),
    WARMUP: String(WARMUP),
    SEED: String(SEED),
    FUSE_GELU_BWD_PW: config.fuseGeluBwdIntoPw ? "1" : "0",
    FUSE_RESIDUAL_BWD_PW: config.fuseResidualBwdIntoPw ? "1" : "0",
  };
  if (config.singlePassRasterForward !== null) {
    env.SINGLE_PASS_RASTER_FWD = config.singlePassRasterForward ? "1" : "0";
  }
  if (config.viewLaneRasterForward !== null) {
    env.VIEW_LANE_RASTER_FWD = config.viewLaneRasterForward ? "1" : "0";
  }
  if (config.viewLaneRasterBackward !== null) {
    env.VIEW_LANE_RASTER_BWD = config.viewLaneRasterBackward ? "1" : "0";
  }
  if (config.cap !== null) {
    env.CAP = String(config.cap);
  }
  delete env.CONFIGS;
  delete env.TRIALS;
  delete env.JSON;
  if (G) env.G = G;

  const child = spawnSync("bun", ["tools/splat3d/step_bench.ts"], {
    cwd: process.cwd(),
    env,
    encoding: "utf8",
    maxBuffer: 1024 * 1024 * 16,
  });
  const output = `${child.stdout ?? ""}${child.stderr ?? ""}`;
  if (child.status !== 0) {
    throw new Error(`step_matrix: child failed for ${config.views}:${config.clipBatch}\n${output}`);
  }
  const normal = parseNumber(/normal step avg: ([\d.]+) ms/, output, "normal step avg");
  return {
    ...config,
    trial,
    normal,
    ...parseProfile(output),
  };
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

const results: TrialResult[] = [];
if (!JSON_OUT) {
  console.log(
    `splat3d step matrix: configs=${CONFIGS.map((c) => `${c.label}=${c.views}:${c.clipBatch}${c.fuseGeluBwdIntoPw ? ":gelubwd" : ""}${c.fuseResidualBwdIntoPw ? ":resbwd" : ""}${c.singlePassRasterForward === true ? ":rasterpass" : ""}${c.singlePassRasterForward === false ? ":norasterpass" : ""}${c.viewLaneRasterForward === true ? ":viewlane" : ""}${c.viewLaneRasterForward === false ? ":noviewlane" : ""}${c.viewLaneRasterBackward === true ? ":viewbwd" : ""}${c.viewLaneRasterBackward === false ? ":noviewbwd" : ""}${c.cap !== null ? `:cap${c.cap}` : ""}`).join(",")} ` +
      `trials=${TRIALS} runs=${RUNS} warmup=${WARMUP} seed=${SEED}${G ? ` G=${G}` : ""}`
  );
}

for (let trial = 0; trial < TRIALS; trial++) {
  const order = trial % 2 === 0 ? CONFIGS : CONFIGS.slice().reverse();
  for (const config of order) {
    const result = runTrial(config, trial);
    results.push(result);
    if (!JSON_OUT) {
      const flags = [
        config.fuseGeluBwdIntoPw ? "gelubwd=1" : "",
        config.fuseResidualBwdIntoPw ? "resbwd=1" : "",
        config.singlePassRasterForward === true ? "rasterpass=1" : "",
        config.singlePassRasterForward === false ? "rasterpass=0" : "",
        config.viewLaneRasterForward === true ? "viewlane=1" : "",
        config.viewLaneRasterForward === false ? "viewlane=0" : "",
        config.viewLaneRasterBackward === true ? "viewbwd=1" : "",
        config.viewLaneRasterBackward === false ? "viewbwd=0" : "",
        config.cap !== null ? `cap=${config.cap}` : "",
      ].filter(Boolean);
      console.log(
        `trial ${trial} ${config.label} ${config.views}/${config.clipBatch}` +
          `${flags.length ? ` ${flags.join(" ")}` : ""}: ` +
          `normal=${result.normal.toFixed(2)} profile=${result.profileTotal.toFixed(2)} ` +
          `clip=${(result.clipBatchMs || result.clipFwd + result.clipBwd).toFixed(2)} ` +
          `raster=${(result.rasterFwd + result.rasterReplay + result.rasterBwd).toFixed(2)}`
      );
    }
  }
}

if (JSON_OUT) {
  console.log(JSON.stringify({ trials: TRIALS, runs: RUNS, warmup: WARMUP, seed: SEED, results }, null, 2));
} else {
  console.log("\nSummary:");
  console.log("config           views batch   cap gbwd rbwd rpass vlane  vbwd  normal med [min,max]     profile med     clip med   raster med");
  for (const config of CONFIGS) {
    const rows = results.filter((r) => r.label === config.label);
    const normal = rows.map((r) => r.normal);
    const profile = rows.map((r) => r.profileTotal);
    const clip = rows.map((r) => r.clipBatchMs || r.clipFwd + r.clipBwd);
    const raster = rows.map((r) => r.rasterFwd + r.rasterReplay + r.rasterBwd);
    console.log(
      `${config.label.padEnd(16).slice(0, 16)} ${String(config.views).padStart(5)} ${String(config.clipBatch).padStart(5)} ` +
        `${String(config.cap ?? "def").padStart(5)} ` +
        `${(config.fuseGeluBwdIntoPw ? "yes" : "no").padStart(4)} ` +
        `${(config.fuseResidualBwdIntoPw ? "yes" : "no").padStart(4)} ` +
        `${(config.singlePassRasterForward === null ? "def" : config.singlePassRasterForward ? "yes" : "no").padStart(5)} ` +
        `${(config.viewLaneRasterForward === null ? "def" : config.viewLaneRasterForward ? "yes" : "no").padStart(5)} ` +
        `${(config.viewLaneRasterBackward === null ? "def" : config.viewLaneRasterBackward ? "yes" : "no").padStart(5)} ` +
        `${fmt(median(normal))} [${min(normal).toFixed(2)},${max(normal).toFixed(2)}] ` +
        `${fmt(median(profile))} ${fmt(median(clip))} ${fmt(median(raster))}`
    );
  }
}
