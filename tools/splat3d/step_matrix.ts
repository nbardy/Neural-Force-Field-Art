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
 *   TRIALS=3 CONFIGS=base9=9:3,grid=9:3:grid9 bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=grid=9:3:grid9,grid80=9:3:grid9:directgrid bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,swres=3:3:sw10-15-24-34-49 bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,cap1024=3:3:cap1024 bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=k1=1:1,k2=2:2,k2rand=2:2:random,base=3:3 bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,dw4=3:3:dw4 bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,cache2=3:3:cache2,cache4=3:3:cache4 bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,cache4=3:3:cache4,cache4lr25=3:3:cache4:lr0.25 bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=base=3:3,bg=3:3:bgdark,conv=3:3:bgdark:alphaweak:boundsweak bun tools/splat3d/step_matrix.ts
 *   TRIALS=3 CONFIGS=3:1,3:3,9:1,9:3,9:9 bun tools/splat3d/step_matrix.ts
 */
import { spawnSync } from "node:child_process";

interface Config {
  label: string;
  views: number;
  clipBatch: number;
  clipLayout: "per_view" | "grid9_close2";
  viewSampler: "epoch" | "random";
  spatialBwdVariant: "default" | "generic" | "depthwise4";
  fuseGeluBwdIntoPw: boolean | null;
  fuseResidualBwdIntoPw: boolean | null;
  singlePassRasterForward: boolean | null;
  viewLaneRasterForward: boolean | null;
  viewLaneRasterBackward: boolean | null;
  gridDirectRaster: boolean;
  sharedWForwardSteps: string;
  pointwiseTileVariant: "default" | "rect8x16";
  pointwiseTileSteps: string;
  clipRefreshInterval: number;
  cachedLrScale: number;
  backgroundMode: "black" | "dark_random" | "curriculum";
  alphaReg: "off" | "weak" | "medium";
  boundsReg: "off" | "weak" | "medium";
  coverageReg: "off" | "weak" | "medium";
  splatReg: "off" | "tiny" | "band";
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
  regularizer: number;
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
      const clipLayout = tokens.includes("grid9") || tokens.includes("grid9_close2") ? "grid9_close2" : "per_view";
      const viewSampler = tokens.includes("random") ? "random" : "epoch";
      const spatialBwdVariant =
        tokens.includes("dw4") || tokens.includes("depthwise4")
          ? "depthwise4"
          : tokens.includes("generic") || tokens.includes("gen")
            ? "generic"
            : "default";
      const noFusions = tokens.includes("nofusions");
      const fuseGeluBwdIntoPw = tokens.includes("gelubwd")
        ? true
        : noFusions || tokens.includes("nogelubwd")
          ? false
          : null;
      const fuseResidualBwdIntoPw = tokens.includes("resbwd")
        ? true
        : noFusions || tokens.includes("noresbwd")
          ? false
          : null;
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
      const gridDirectRaster = tokens.includes("directgrid") || tokens.includes("grid80");
      const sharedWToken = tokens.find((token) => /^sw\d+(?:-\d+)*$/.test(token));
      const sharedWForwardSteps = sharedWToken ? sharedWToken.slice(2).split("-").join(",") : "";
      const pointwiseTileVariant = tokens.includes("pwrect") || tokens.includes("rect8x16") ? "rect8x16" : "default";
      const pwStepsToken = tokens.find((token) => /^pwsteps\d+(?:-\d+)*$/.test(token));
      const pointwiseTileSteps = pwStepsToken ? pwStepsToken.slice("pwsteps".length).split("-").join(",") : "";
      const cacheToken = tokens.find((token) => /^cache\d+$/.test(token) || /^refresh\d+$/.test(token));
      const clipRefreshInterval = cacheToken ? Math.max(1, Number(cacheToken.replace(/^\D+/, "")) | 0) : 1;
      const cachedLrToken = tokens.find((token) => /^lr\d*(?:\.\d+)?$/.test(token) || /^cachedlr\d*(?:\.\d+)?$/.test(token));
      const cachedLrScale = cachedLrToken
        ? Math.max(0, Number(cachedLrToken.replace(/^(cached)?lr/, "")))
        : 1;
      const backgroundMode = tokens.includes("bgdark")
        ? "dark_random"
        : tokens.includes("bgcurr") || tokens.includes("bgcurriculum")
          ? "curriculum"
          : "black";
      const alphaReg = tokens.includes("alphamed") || tokens.includes("alphamedium")
        ? "medium"
        : tokens.includes("alphaweak")
          ? "weak"
          : "off";
      const boundsReg = tokens.includes("boundsmed") || tokens.includes("boundsmedium")
        ? "medium"
        : tokens.includes("boundsweak")
          ? "weak"
          : "off";
      const coverageReg = tokens.includes("coveragemed") || tokens.includes("coveragemedium")
        ? "medium"
        : tokens.includes("coverageweak")
          ? "weak"
          : "off";
      const splatReg = tokens.includes("splatband") || tokens.includes("scaleband")
        ? "band"
        : tokens.includes("splattiny") || tokens.includes("antitiny")
          ? "tiny"
          : "off";
      const capToken = tokens.find((token) => /^cap\d+$/.test(token));
      const cap = capToken ? Number(capToken.slice(3)) : null;
      const suffix = tokens.length ? `:${tokens.join(":")}` : "";
      const label = labelRaw.trim() || `${views | 0}:${clipBatch | 0}${suffix}`;
      return {
        label,
        views: views | 0,
        clipBatch: clipBatch | 0,
        clipLayout,
        viewSampler,
        spatialBwdVariant,
        fuseGeluBwdIntoPw,
        fuseResidualBwdIntoPw,
        singlePassRasterForward,
        viewLaneRasterForward,
        viewLaneRasterBackward,
        gridDirectRaster,
        sharedWForwardSteps,
        pointwiseTileVariant,
        pointwiseTileSteps,
        clipRefreshInterval,
        cachedLrScale: Number.isFinite(cachedLrScale) ? cachedLrScale : 1,
        backgroundMode,
        alphaReg,
        boundsReg,
        coverageReg,
        splatReg,
        cap,
      };
    });
  if (!configs.length) throw new Error("step_matrix: CONFIGS produced no configs");
  return configs;
}

function configSpec(config: Config): string {
  const tokens = [
    config.clipLayout === "grid9_close2" ? "grid9" : "",
    config.viewSampler === "random" ? "random" : "",
    config.spatialBwdVariant === "depthwise4" ? "dw4" : "",
    config.spatialBwdVariant === "generic" ? "generic" : "",
    config.gridDirectRaster ? "directgrid" : "",
    config.sharedWForwardSteps ? `sw${config.sharedWForwardSteps.split(",").join("-")}` : "",
    config.pointwiseTileVariant === "rect8x16" ? "pwrect" : "",
    config.pointwiseTileSteps ? `pwsteps${config.pointwiseTileSteps.split(",").join("-")}` : "",
    config.clipRefreshInterval > 1 ? `cache${config.clipRefreshInterval}` : "",
    config.cachedLrScale !== 1 ? `lr${config.cachedLrScale}` : "",
    config.fuseGeluBwdIntoPw === true ? "gelubwd" : "",
    config.fuseGeluBwdIntoPw === false ? "nogelubwd" : "",
    config.fuseResidualBwdIntoPw === true ? "resbwd" : "",
    config.fuseResidualBwdIntoPw === false ? "noresbwd" : "",
    config.singlePassRasterForward === true ? "rasterpass" : "",
    config.singlePassRasterForward === false ? "norasterpass" : "",
    config.viewLaneRasterForward === true ? "viewlane" : "",
    config.viewLaneRasterForward === false ? "noviewlane" : "",
    config.viewLaneRasterBackward === true ? "viewbwd" : "",
    config.viewLaneRasterBackward === false ? "noviewbwd" : "",
    config.backgroundMode === "dark_random" ? "bgdark" : "",
    config.backgroundMode === "curriculum" ? "bgcurr" : "",
    config.alphaReg === "weak" ? "alphaweak" : "",
    config.alphaReg === "medium" ? "alphamed" : "",
    config.boundsReg === "weak" ? "boundsweak" : "",
    config.boundsReg === "medium" ? "boundsmed" : "",
    config.coverageReg === "weak" ? "coverageweak" : "",
    config.coverageReg === "medium" ? "coveragemed" : "",
    config.splatReg === "tiny" ? "splattiny" : "",
    config.splatReg === "band" ? "splatband" : "",
    config.cap !== null ? `cap${config.cap}` : "",
  ].filter(Boolean);
  return `${config.label}=${config.views}:${config.clipBatch}${tokens.length ? `:${tokens.join(":")}` : ""}`;
}

function parseNumber(pattern: RegExp, output: string, name: string): number {
  const m = output.match(pattern);
  if (!m) throw new Error(`step_matrix: could not parse ${name}\n${output}`);
  return Number(m[1]);
}

function parseProfile(output: string): Omit<TrialResult, "trial" | "views" | "clipBatch" | "normal"> {
  const m = output.match(
    /profile: total=([\d.]+) ms rasterFwd=([\d.]+) rasterReplay=([\d.]+) rasterBwd=([\d.]+) clipFwd=([\d.]+) clipBwd=([\d.]+) clipBatch=([\d.]+)(?: regularizer=([\d.]+))? adam=([\d.]+) display=([\d.]+)/
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
    regularizer: Number(m[8] ?? 0),
    adam: Number(m[9]),
    display: Number(m[10]),
  };
}

function runTrial(config: Config, trial: number): TrialResult {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    VIEWS: String(config.views),
    CLIP_BATCH: String(config.clipBatch),
    CLIP_LAYOUT: config.clipLayout,
    VIEW_SAMPLER: config.viewSampler,
    SPATIAL_BWD_VARIANT: config.spatialBwdVariant === "default" ? "" : config.spatialBwdVariant,
    GRID_DIRECT_RASTER: config.gridDirectRaster ? "1" : "0",
    SHARED_W_FWD_STEPS: config.sharedWForwardSteps,
    PW_TILE_VARIANT: config.pointwiseTileVariant === "rect8x16" ? "rect8x16" : "",
    PW_TILE_STEPS: config.pointwiseTileSteps,
    CLIP_REFRESH_INTERVAL: String(config.clipRefreshInterval),
    CLIP_CACHED_LR_SCALE: String(config.cachedLrScale),
    BACKGROUND_MODE: config.backgroundMode,
    ALPHA_REG: config.alphaReg,
    BOUNDS_REG: config.boundsReg,
    COVERAGE_REG: config.coverageReg,
    SPLAT_REG: config.splatReg,
    RUNS: String(RUNS),
    WARMUP: String(WARMUP),
    SEED: String(SEED),
    FUSE_GELU_BWD_PW: config.fuseGeluBwdIntoPw === null ? "" : config.fuseGeluBwdIntoPw ? "1" : "0",
    FUSE_RESIDUAL_BWD_PW:
      config.fuseResidualBwdIntoPw === null ? "" : config.fuseResidualBwdIntoPw ? "1" : "0",
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
    `splat3d step matrix: configs=${CONFIGS.map(configSpec).join(",")} ` +
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
        config.fuseGeluBwdIntoPw === true ? "gelubwd=1" : "",
        config.fuseGeluBwdIntoPw === false ? "gelubwd=0" : "",
        config.clipLayout === "grid9_close2" ? "grid9=1" : "",
        config.viewSampler === "random" ? "random=1" : "",
        config.spatialBwdVariant === "depthwise4" ? "dw4=1" : "",
        config.spatialBwdVariant === "generic" ? "generic=1" : "",
        config.gridDirectRaster ? "directgrid=1" : "",
        config.sharedWForwardSteps ? `sw=${config.sharedWForwardSteps}` : "",
        config.pointwiseTileVariant === "rect8x16" ? "pwrect=1" : "",
        config.pointwiseTileSteps ? `pwsteps=${config.pointwiseTileSteps}` : "",
        config.clipRefreshInterval > 1 ? `cache=${config.clipRefreshInterval}` : "",
        config.cachedLrScale !== 1 ? `cachedlr=${config.cachedLrScale}` : "",
        config.fuseResidualBwdIntoPw === true ? "resbwd=1" : "",
        config.fuseResidualBwdIntoPw === false ? "resbwd=0" : "",
        config.singlePassRasterForward === true ? "rasterpass=1" : "",
        config.singlePassRasterForward === false ? "rasterpass=0" : "",
        config.viewLaneRasterForward === true ? "viewlane=1" : "",
        config.viewLaneRasterForward === false ? "viewlane=0" : "",
        config.viewLaneRasterBackward === true ? "viewbwd=1" : "",
        config.viewLaneRasterBackward === false ? "viewbwd=0" : "",
        config.backgroundMode !== "black" ? `bg=${config.backgroundMode}` : "",
        config.alphaReg !== "off" ? `alpha=${config.alphaReg}` : "",
        config.boundsReg !== "off" ? `bounds=${config.boundsReg}` : "",
        config.coverageReg !== "off" ? `coverage=${config.coverageReg}` : "",
        config.splatReg !== "off" ? `splat=${config.splatReg}` : "",
        config.cap !== null ? `cap=${config.cap}` : "",
      ].filter(Boolean);
      console.log(
        `trial ${trial} ${config.label} ${config.views}/${config.clipBatch}` +
          `${flags.length ? ` ${flags.join(" ")}` : ""}: ` +
          `normal=${result.normal.toFixed(2)} profile=${result.profileTotal.toFixed(2)} ` +
          `clip=${(result.clipBatchMs || result.clipFwd + result.clipBwd).toFixed(2)} ` +
          `raster=${(result.rasterFwd + result.rasterReplay + result.rasterBwd).toFixed(2)} ` +
          `reg=${result.regularizer.toFixed(2)}`
      );
    }
  }
}

if (JSON_OUT) {
  console.log(JSON.stringify({ trials: TRIALS, runs: RUNS, warmup: WARMUP, seed: SEED, results }, null, 2));
} else {
  console.log("\nSummary:");
  console.log("config           views batch  layout sampler spbwd gridd      sw  pwrect  pwsteps cache     lr   cap gbwd rbwd rpass vlane  vbwd        bg  alpha bounds   cover   splat  normal med [min,max]     profile med     clip med   raster med      reg med");
  for (const config of CONFIGS) {
    const rows = results.filter((r) => r.label === config.label);
    const normal = rows.map((r) => r.normal);
    const profile = rows.map((r) => r.profileTotal);
    const clip = rows.map((r) => r.clipBatchMs || r.clipFwd + r.clipBwd);
    const raster = rows.map((r) => r.rasterFwd + r.rasterReplay + r.rasterBwd);
    const regularizer = rows.map((r) => r.regularizer);
    console.log(
      `${config.label.padEnd(16).slice(0, 16)} ${String(config.views).padStart(5)} ${String(config.clipBatch).padStart(5)} ` +
        `${(config.clipLayout === "grid9_close2" ? "grid9" : "view").padStart(7)} ` +
        `${config.viewSampler.padStart(7)} ` +
        `${(config.spatialBwdVariant === "depthwise4" ? "dw4" : config.spatialBwdVariant === "generic" ? "gen" : "def").padStart(5)} ` +
        `${(config.gridDirectRaster ? "yes" : "no").padStart(5)} ` +
        `${(config.sharedWForwardSteps ? "yes" : "no").padStart(7)} ` +
        `${(config.pointwiseTileVariant === "rect8x16" ? "yes" : "no").padStart(7)} ` +
        `${(config.pointwiseTileSteps || "-").padStart(8).slice(0, 8)} ` +
        `${String(config.clipRefreshInterval).padStart(5)} ` +
        `${config.cachedLrScale.toFixed(2).padStart(6)} ` +
        `${String(config.cap ?? "def").padStart(5)} ` +
        `${(config.fuseGeluBwdIntoPw === null ? "def" : config.fuseGeluBwdIntoPw ? "yes" : "no").padStart(4)} ` +
        `${(config.fuseResidualBwdIntoPw === null ? "def" : config.fuseResidualBwdIntoPw ? "yes" : "no").padStart(4)} ` +
        `${(config.singlePassRasterForward === null ? "def" : config.singlePassRasterForward ? "yes" : "no").padStart(5)} ` +
        `${(config.viewLaneRasterForward === null ? "def" : config.viewLaneRasterForward ? "yes" : "no").padStart(5)} ` +
        `${(config.viewLaneRasterBackward === null ? "def" : config.viewLaneRasterBackward ? "yes" : "no").padStart(5)} ` +
        `${(config.backgroundMode === "dark_random" ? "dark" : config.backgroundMode === "curriculum" ? "curr" : "black").padStart(9)} ` +
        `${config.alphaReg.padStart(6)} ` +
        `${config.boundsReg.padStart(6)} ` +
        `${config.coverageReg.padStart(7)} ` +
        `${config.splatReg.padStart(7)} ` +
        `${fmt(median(normal))} [${min(normal).toFixed(2)},${max(normal).toFixed(2)}] ` +
        `${fmt(median(profile))} ${fmt(median(clip))} ${fmt(median(raster))} ${fmt(median(regularizer))}`
    );
  }
}
