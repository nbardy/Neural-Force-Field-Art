/// <reference types="@webgpu/types" />
import {
  buildBasePrompt,
  buildCoarseViewPrompt,
  buildGrid9Prompt,
  buildRandomGrid9Prompt,
  buildViewPrompt,
  buildZoomPrompt,
  camerasForFraming,
  dualGridCamerasForFraming,
  type BackgroundPromptMode,
  type CameraFramingMode,
  type Grid9PromptMode,
  type PreparedCamera3D,
  type ViewPromptMode,
} from "./splat3d/cameras";
import {
  Splat3DOptimizer,
  cosine,
  type Splat3DBackgroundMode,
  type Splat3DClipLayout,
  type Splat3DConvergenceConfig,
  type Splat3DStepTimings,
  type Splat3DViewSampler,
} from "./splat3d/optimize";
import type { Raster3DTileTelemetry } from "./splat3d/raster";
import { PARAM_STRIDE_3D } from "./splat3d/raster_wgsl";
import {
  ANISO_PARAM_STRIDE_3D,
  projectAnisotropicGaussian,
  Splat3DAnisotropicOptimizer,
} from "./splat3d_aniso";
import { loadClipTrainAssets } from "./splat/model_assets";
import type { TrainPlan } from "./clip/vision";

const SIDE = 256;
const HW = SIDE * SIDE;
type QualityPreset = "full3d" | "fast" | "manual";
type RepresentationMode = "isotropic" | "anisotropic";
type GridStrength = "off" | "weak" | "medium" | "full";

interface CloudTelemetry {
  opacityMean: number;
  opacityP10: number;
  opacityP90: number;
  radiusMean: number;
  spreadRms: number;
  axisRatioMean: number;
  axisRatioP90: number;
  screenAxisRatioMean: number;
  screenAxisRatioP90: number;
}

interface Status {
  gpu: boolean;
  ready: boolean;
  running: boolean;
  step: number;
  view: number;
  cos: number | null;
  initialCos: number | null;
  error: string | null;
  phase: string;
  qualityPreset: QualityPreset;
  representation: RepresentationMode;
  promptMode: ViewPromptMode;
  gridPromptMode: Grid9PromptMode;
  bgPromptMode: BackgroundPromptMode;
  backgroundMode: Splat3DBackgroundMode;
  alphaReg: "off" | "weak" | "medium";
  boundsReg: "off" | "weak" | "medium";
  coverageReg: "off" | "weak" | "medium";
  rayReg: "off" | "weak" | "medium";
  entropyReg: "off" | "weak" | "medium";
  stageMode: "joint" | "staged";
  adaptiveSplats: boolean;
  mipSmoothing: boolean;
  splatReg: "off" | "tiny" | "band";
  framingMode: CameraFramingMode;
  profiling: boolean;
  viewsPerStep: number;
  viewSampler: Splat3DViewSampler;
  clipBatchSize: number;
  clipLayout: Splat3DClipLayout;
  gridDirectRaster: boolean;
  gridRasterSide: number;
  gridStrength: GridStrength;
}

const status: Status = {
  gpu: !!navigator.gpu,
  ready: false,
  running: false,
  step: 0,
  view: 0,
  cos: null,
  initialCos: null,
  error: null,
  phase: "boot",
  qualityPreset: "full3d",
  representation: "anisotropic",
  promptMode: "camera",
  gridPromptMode: "contact_sheet",
  bgPromptMode: "centered",
  backgroundMode: "black",
  alphaReg: "off",
  boundsReg: "weak",
  coverageReg: "off",
  rayReg: "off",
  entropyReg: "off",
  stageMode: "joint",
  adaptiveSplats: true,
  mipSmoothing: false,
  splatReg: "tiny",
  framingMode: "zoom_out",
  profiling: false,
  viewsPerStep: 3,
  viewSampler: "random",
  clipBatchSize: 3,
  clipLayout: "per_view",
  gridDirectRaster: true,
  gridRasterSide: 80,
  gridStrength: "weak",
};
(window as any).__splat3d = status;

const gridEl = document.getElementById("grid") as HTMLDivElement;
const promptInput = document.getElementById("prompt") as HTMLInputElement;
const qualityPresetSelect = document.getElementById("qualityPreset") as HTMLSelectElement;
const representationSelect = document.getElementById("representation") as HTMLSelectElement;
const viewSelect = document.getElementById("view") as HTMLSelectElement;
const promptModeSelect = document.getElementById("promptMode") as HTMLSelectElement;
const bgTextModeSelect = document.getElementById("bgTextMode") as HTMLSelectElement;
const backgroundModeSelect = document.getElementById("backgroundMode") as HTMLSelectElement;
const alphaRegSelect = document.getElementById("alphaReg") as HTMLSelectElement;
const boundsRegSelect = document.getElementById("boundsReg") as HTMLSelectElement;
const coverageRegSelect = document.getElementById("coverageReg") as HTMLSelectElement;
const rayRegSelect = document.getElementById("rayReg") as HTMLSelectElement;
const entropyRegSelect = document.getElementById("entropyReg") as HTMLSelectElement;
const stageModeSelect = document.getElementById("stageMode") as HTMLSelectElement;
const adaptiveSplatsSelect = document.getElementById("adaptiveSplats") as HTMLSelectElement;
const mipSmoothingSelect = document.getElementById("mipSmoothing") as HTMLSelectElement;
const splatRegSelect = document.getElementById("splatReg") as HTMLSelectElement;
const framingModeSelect = document.getElementById("framingMode") as HTMLSelectElement;
const viewBatchSelect = document.getElementById("viewBatch") as HTMLSelectElement;
const viewSamplerSelect = document.getElementById("viewSampler") as HTMLSelectElement;
const clipModeSelect = document.getElementById("clipMode") as HTMLSelectElement;
const clipLayoutSelect = document.getElementById("clipLayout") as HTMLSelectElement;
const gridPromptModeSelect = document.getElementById("gridPromptMode") as HTMLSelectElement;
const gridRasterModeSelect = document.getElementById("gridRasterMode") as HTMLSelectElement;
const gridStrengthSelect = document.getElementById("gridStrength") as HTMLSelectElement;
const optimizeBtn = document.getElementById("optimize") as HTMLButtonElement;
const resetBtn = document.getElementById("reset") as HTMLButtonElement;
const readoutEl = document.getElementById("readout") as HTMLDivElement;
const noticeEl = document.getElementById("notice") as HTMLDivElement;
const timingsEl = document.getElementById("timings") as HTMLDivElement;

function setNotice(msg: string): void {
  noticeEl.textContent = msg;
}

function fail(msg: string): void {
  status.error = msg;
  status.phase = "error";
  setNotice(msg);
  readoutEl.textContent = "—";
  console.error("[splat3d_page]", msg);
}

function renderReadout(): void {
  status.step = opt ? opt.stepCount : 0;
  const camera = opt?.cameras[displayView]?.name ?? "view";
  const parts: string[] = [`step ${status.step}`, camera];
  parts.push(status.qualityPreset === "full3d" ? "full 3D" : status.qualityPreset === "fast" ? "fast base" : "manual");
  parts.push(status.representation === "anisotropic" ? "anisotropic" : "isotropic");
  if (opt) parts.push(`${status.viewsPerStep}/${opt.cameras.length} views`);
  if (status.viewSampler === "random") parts.push("random");
  parts.push(status.clipBatchSize > 1 ? `clip x${status.clipBatchSize}` : "clip x1");
  if (status.clipLayout === "grid9_close2") parts.push("grid+2");
  if (status.clipLayout === "dual_grid4") parts.push("2 grids+4");
  if (status.clipLayout === "grid9_close2" || status.clipLayout === "dual_grid4") {
    const gridText =
      status.gridPromptMode === "same"
        ? "grid=same text"
        : status.gridPromptMode === "literal_v2"
          ? "object grid text"
        : status.gridPromptMode === "literal"
          ? "literal grid text"
          : "grid text";
    parts.push(gridText);
    parts.push(`${status.gridRasterSide}px grid raster`);
    parts.push(`grid ${status.gridStrength}`);
  }
  parts.push(status.promptMode === "camera" ? "camera text" : status.promptMode === "coarse" ? "coarse text" : "same text");
  if (status.bgPromptMode === "black") parts.push("black bg");
  if (status.bgPromptMode === "centered") parts.push("centered bg");
  if (status.backgroundMode !== "black") parts.push(`${status.backgroundMode.replaceAll("_", " ")} bg`);
  if (status.alphaReg !== "off") parts.push(`alpha ${status.alphaReg}`);
  if (status.boundsReg !== "off") parts.push(`bounds ${status.boundsReg}`);
  if (status.coverageReg !== "off") parts.push(`transmit ${status.coverageReg}`);
  if (status.rayReg !== "off") parts.push(`ray compact ${status.rayReg}`);
  if (status.entropyReg !== "off") parts.push(`ray entropy ${status.entropyReg}`);
  if (status.stageMode === "staged") parts.push("staged rates");
  if (status.adaptiveSplats) parts.push("adaptive splats");
  if (status.mipSmoothing) parts.push("coarse-to-fine");
  if (status.splatReg !== "off") parts.push(status.splatReg === "band" ? "scale band" : "anti-tiny");
  if (status.framingMode === "zoom_out") parts.push("zoom out");
  if (status.cos !== null) {
    const init = status.initialCos ?? status.cos;
    const d = status.cos - init;
    parts.push(`cos ${status.cos.toFixed(4)}`);
    parts.push(`init ${init.toFixed(4)}`);
    parts.push(`Δ ${d >= 0 ? "+" : ""}${d.toFixed(4)}`);
  }
  if (status.phase && status.phase !== "run") parts.push(`(${status.phase})`);
  readoutEl.textContent = parts.join("  ·  ");
}

function renderTimings(): void {
  if (!latestTimings) {
    timingsEl.textContent = "sampled wall profile waiting...";
    return;
  }
  const t = latestTimings;
  const total = Math.max(t.total, 0.001);
  const line = (name: string, ms: number): string => {
    const pct = (100 * ms) / total;
    return `${name.padEnd(11)} ${ms.toFixed(1).padStart(6)} ms ${pct.toFixed(0).padStart(3)}%`;
  };
  const lines = [
    `${t.timing === "gpu-timestamp" ? "sampled GPU step" : "sampled wall step"} ${status.step}`,
    `${t.views}/${t.totalViews} views · ${status.viewSampler} · ${t.clipMode === "batch" ? `batch CLIP x${t.clipBatchSize}` : "single CLIP"} · ${t.timing}`,
    line("opt total", t.total),
    line("raster", t.rasterFwd + t.rasterReplay + t.rasterBwd),
    line("  fwd", t.rasterFwd),
  ];
  if (t.rasterReplay > 0) lines.push(line("  replay", t.rasterReplay));
  lines.push(line("  bwd", t.rasterBwd));
  if (t.clipMode === "batch") {
    lines.push(line("clip batch", t.clipBatch));
  } else {
    lines.push(line("clip", t.clipFwd + t.clipBwd), line("  fwd", t.clipFwd), line("  bwd", t.clipBwd));
  }
  if (t.regularizer > 0) lines.push(line("reg", t.regularizer));
  lines.push(line("adam", t.adam), line("display", t.display), line("clear", t.clear), `sample every ${PROFILE_PERIOD} steps`);
  if (latestTileTelemetry) {
    const tile = latestTileTelemetry;
    lines.push(`tile count   ${tile.maxCount}/${tile.cap} max`);
    lines.push(
      tile.overflowTiles > 0
        ? `OVERFLOW     ${tile.overflowTiles} tiles · ${tile.overflowPairs} pairs`
        : `tile overflow 0`
    );
  }
  if (latestCloudTelemetry) {
    const cloud = latestCloudTelemetry;
    lines.push(
      `opacity      ${cloud.opacityMean.toFixed(3)} mean · ${cloud.opacityP10.toFixed(3)}-${cloud.opacityP90.toFixed(3)} p10-p90`
    );
    lines.push(`splat radius ${cloud.radiusMean.toFixed(4)} · spread ${cloud.spreadRms.toFixed(3)}`);
    if (status.representation === "anisotropic") {
      lines.push(`axis ratio   ${cloud.axisRatioMean.toFixed(2)} mean · ${cloud.axisRatioP90.toFixed(2)} p90`);
      lines.push(
        `screen ratio ${cloud.screenAxisRatioMean.toFixed(2)} mean · ${cloud.screenAxisRatioP90.toFixed(2)} p90`
      );
    }
  }
  const adaptation = opt?.adaptationDiagnostics;
  if (adaptation) {
    lines.push(`adapt splats ${adaptation.relocationCount} moved · ${adaptation.eligibleDestinations} dead`);
    if (adaptation.densityStatsSampled) {
      lines.push(
        `density sample ${adaptation.densityVisiblePixels ?? 0} px · ` +
          `abs grad ${(adaptation.densityMaxScreenGradient ?? 0).toExponential(2)}`
      );
    }
  }
  timingsEl.textContent = lines.join("\n");
}

let device!: GPUDevice;
let plan!: TrainPlan;
let weights!: Float32Array;
type ActiveOptimizer = Splat3DOptimizer | Splat3DAnisotropicOptimizer;
let opt!: ActiveOptimizer;
let seed = 1;
let displayView = 0;
let gridDirty = false;
let latestTimings: Splat3DStepTimings | null = null;
let latestTileTelemetry: Raster3DTileTelemetry | null = null;
let latestCloudTelemetry: CloudTelemetry | null = null;
let profileBusy = false;
const PROFILE_PERIOD = 30;

let blitPipe!: GPURenderPipeline;
let blitBind: GPUBindGroup | null = null;
let viewCtxs: GPUCanvasContext[] = [];
let viewTiles: HTMLDivElement[] = [];
let canvasFormat!: GPUTextureFormat;

function selectedClipBatchSize(): number {
  const layout = selectedClipLayout();
  if (layout === "grid9_close2") return 3;
  if (layout === "dual_grid4") return 6;
  const n = Number(clipModeSelect.value);
  return Number.isFinite(n) && n > 1 ? Math.min(9, n | 0) : 1;
}

function selectedQualityPreset(): QualityPreset {
  if (qualityPresetSelect.value === "fast") return "fast";
  if (qualityPresetSelect.value === "manual") return "manual";
  return "full3d";
}

function selectedIsotropicLrs() {
  if (selectedQualityPreset() !== "full3d") return undefined;
  return {
    position: 0.035,
    logRadius: 0.018,
    color: 0.035,
    opacity: 0.025,
  };
}

function selectedAnisotropicLrs() {
  if (selectedQualityPreset() !== "full3d") return undefined;
  return {
    position: 0.03,
    logScale: 0.018,
    quaternion: 0.01,
    color: 0.04,
    opacity: 0.025,
  };
}

function selectedClipLayout(): Splat3DClipLayout {
  if (selectedRepresentation() === "anisotropic") return "per_view";
  if (clipLayoutSelect.value === "dual_grid4") return "dual_grid4";
  return clipLayoutSelect.value === "grid9_close2" ? "grid9_close2" : "per_view";
}

function selectedRepresentation(): RepresentationMode {
  return representationSelect.value === "anisotropic" ? "anisotropic" : "isotropic";
}

function selectedViewsPerStep(): number {
  const layout = selectedClipLayout();
  if (layout === "grid9_close2") return 9;
  if (layout === "dual_grid4") return 22;
  const n = Number(viewBatchSelect.value);
  const maxViews = opt?.cameras.length ?? 9;
  return Number.isFinite(n) ? Math.max(1, Math.min(maxViews, n | 0)) : 3;
}

function selectedViewSampler(): Splat3DViewSampler {
  return viewSamplerSelect.value === "random" ? "random" : "epoch";
}

function selectedGridPromptMode(): Grid9PromptMode {
  if (gridPromptModeSelect.value === "same") return "same";
  if (gridPromptModeSelect.value === "literal_v2") return "literal_v2";
  if (gridPromptModeSelect.value === "literal") return "literal";
  return "contact_sheet";
}

function selectedGridDirectRaster(): boolean {
  return gridRasterModeSelect.value !== "scratch256";
}

function selectedGridRasterSide(): number {
  if (gridRasterModeSelect.value === "hi512") return 512;
  if (gridRasterModeSelect.value === "direct80") return 80;
  return 256;
}

function gridRasterModeValue(side: number): string {
  if (side === 512) return "hi512";
  if (side === 80) return "direct80";
  return "scratch256";
}

function selectedGridStrength(): GridStrength {
  if (gridStrengthSelect.value === "off") return "off";
  if (gridStrengthSelect.value === "medium") return "medium";
  if (gridStrengthSelect.value === "full") return "full";
  return "weak";
}

function selectedGridGradientScales(): { grid: number; randomGrid: number } {
  const strength = selectedGridStrength();
  if (strength === "off") return { grid: 0, randomGrid: 0 };
  if (strength === "medium") return { grid: 0.5, randomGrid: 0.35 };
  if (strength === "full") return { grid: 1, randomGrid: 1 };
  return { grid: 0.25, randomGrid: 0.15 };
}

function selectedPromptMode(): ViewPromptMode {
  if (promptModeSelect.value === "same") return "same";
  if (promptModeSelect.value === "coarse") return "coarse";
  return "camera";
}

function selectedBgPromptMode(): BackgroundPromptMode {
  if (bgTextModeSelect.value === "none") return "none";
  if (bgTextModeSelect.value === "centered") return "centered";
  return "black";
}

function selectedBackgroundMode(): Splat3DBackgroundMode {
  if (backgroundModeSelect.value === "dark_random") return "dark_random";
  if (backgroundModeSelect.value === "curriculum") return "curriculum";
  if (backgroundModeSelect.value === "blurred_noise") return "blurred_noise";
  if (backgroundModeSelect.value === "checkerboard") return "checkerboard";
  if (backgroundModeSelect.value === "fourier") return "fourier";
  return "black";
}

function selectedAlphaReg(): Status["alphaReg"] {
  return alphaRegSelect.value === "medium" ? "medium" : alphaRegSelect.value === "weak" ? "weak" : "off";
}

function selectedBoundsReg(): Status["boundsReg"] {
  return boundsRegSelect.value === "medium" ? "medium" : boundsRegSelect.value === "weak" ? "weak" : "off";
}

function selectedCoverageReg(): Status["coverageReg"] {
  return coverageRegSelect.value === "medium" ? "medium" : coverageRegSelect.value === "weak" ? "weak" : "off";
}

function selectedRayReg(): Status["rayReg"] {
  return rayRegSelect.value === "medium" ? "medium" : rayRegSelect.value === "weak" ? "weak" : "off";
}

function selectedEntropyReg(): Status["entropyReg"] {
  return entropyRegSelect.value === "medium" ? "medium" : entropyRegSelect.value === "weak" ? "weak" : "off";
}

function selectedSplatReg(): Status["splatReg"] {
  return splatRegSelect.value === "band" ? "band" : splatRegSelect.value === "tiny" ? "tiny" : "off";
}

function selectedFramingMode(): CameraFramingMode {
  return framingModeSelect.value === "zoom_out" ? "zoom_out" : "normal";
}

function selectedConvergenceConfig(): Splat3DConvergenceConfig {
  const alphaReg = selectedAlphaReg();
  const boundsReg = selectedBoundsReg();
  const coverageReg = selectedCoverageReg();
  const rayReg = selectedRayReg();
  const entropyReg = selectedEntropyReg();
  const splatReg = selectedSplatReg();
  return {
    backgroundMode: selectedBackgroundMode(),
    opacitySparsity: alphaReg === "medium" ? 0.03 : alphaReg === "weak" ? 0.01 : 0,
    centerWeight: boundsReg === "medium" ? 0.006 : boundsReg === "weak" ? 0.002 : 0,
    radiusWeight: boundsReg === "medium" ? 0.012 : boundsReg === "weak" ? 0.004 : 0,
    targetRadius: 1.15,
    coverageWeight: coverageReg === "medium" ? 0.2 : coverageReg === "weak" ? 0.05 : 0,
    coverageTarget: 0.12,
    transmittanceStart: 0.4,
    transmittanceEnd: 0.88,
    transmittanceAnnealSteps: 500,
    rayDistortionWeight: rayReg === "medium" ? 0.1 : rayReg === "weak" ? 0.02 : 0,
    rayEntropyWeight: entropyReg === "medium" ? 0.05 : entropyReg === "weak" ? 0.01 : 0,
    rayEntropyMask: 0.05,
    smallRadiusWeight: splatReg === "band" ? 0.035 : splatReg === "tiny" ? 0.02 : 0,
    smallRadius: 0.024,
    radiusBandWeight: splatReg === "band" ? 0.012 : 0,
    minRadius: 0.016,
    maxRadius: 0.16,
    stagedOptimization: stageModeSelect.value === "staged",
    geometryWarmupSteps: 250,
    geometryDecaySteps: 1000,
    geometryFinalScale: 0.2,
    appearanceWarmupScale: 0.35,
    adaptiveRelocation: adaptiveSplatsSelect.value === "on",
    adaptationInterval: 200,
    adaptationFraction: 0.01,
    mipSmoothing: mipSmoothingSelect.value === "on",
    mipVarianceStart: 4,
    mipVarianceEnd: 0.0625,
    mipAnnealSteps: 500,
  };
}

let applyingPreset = false;

function applyQualityPresetToControls(preset: QualityPreset): void {
  if (preset === "manual") return;
  applyingPreset = true;
  try {
    if (preset === "full3d") {
      representationSelect.value = "anisotropic";
      promptModeSelect.value = "camera";
      bgTextModeSelect.value = "centered";
      backgroundModeSelect.value = "black";
      alphaRegSelect.value = "off";
      boundsRegSelect.value = "weak";
      coverageRegSelect.value = "off";
      rayRegSelect.value = "off";
      entropyRegSelect.value = "off";
      stageModeSelect.value = "joint";
      adaptiveSplatsSelect.value = "on";
      mipSmoothingSelect.value = "off";
      splatRegSelect.value = "tiny";
      framingModeSelect.value = "zoom_out";
      viewBatchSelect.value = "3";
      viewSamplerSelect.value = "random";
      clipLayoutSelect.value = "per_view";
      clipModeSelect.value = "3";
      gridPromptModeSelect.value = "literal_v2";
      gridRasterModeSelect.value = "direct80";
      gridStrengthSelect.value = "weak";
      return;
    }
    representationSelect.value = "isotropic";
    promptModeSelect.value = "camera";
    bgTextModeSelect.value = "black";
    backgroundModeSelect.value = "black";
    alphaRegSelect.value = "off";
    boundsRegSelect.value = "off";
    coverageRegSelect.value = "off";
    rayRegSelect.value = "off";
    entropyRegSelect.value = "off";
    stageModeSelect.value = "joint";
    adaptiveSplatsSelect.value = "off";
    mipSmoothingSelect.value = "off";
    splatRegSelect.value = "off";
    framingModeSelect.value = "normal";
    viewBatchSelect.value = "3";
    viewSamplerSelect.value = "epoch";
    clipLayoutSelect.value = "per_view";
    clipModeSelect.value = "3";
    gridStrengthSelect.value = "full";
  } finally {
    applyingPreset = false;
  }
}

function markManualPreset(): void {
  if (applyingPreset) return;
  if (qualityPresetSelect.value !== "manual") {
    qualityPresetSelect.value = "manual";
    status.qualityPreset = "manual";
  }
}

function syncPromptStatus(): void {
  status.promptMode = selectedPromptMode();
  status.gridPromptMode = selectedGridPromptMode();
  status.gridDirectRaster = selectedGridDirectRaster();
  status.gridRasterSide = selectedGridRasterSide();
  status.gridStrength = selectedGridStrength();
  status.bgPromptMode = selectedBgPromptMode();
}

function syncConvergenceStatus(): void {
  status.qualityPreset = selectedQualityPreset();
  status.representation = selectedRepresentation();
  status.backgroundMode = selectedBackgroundMode();
  status.alphaReg = selectedAlphaReg();
  status.boundsReg = selectedBoundsReg();
  status.coverageReg = selectedCoverageReg();
  status.rayReg = selectedRayReg();
  status.entropyReg = selectedEntropyReg();
  status.stageMode = stageModeSelect.value === "staged" ? "staged" : "joint";
  status.adaptiveSplats = adaptiveSplatsSelect.value === "on";
  status.mipSmoothing = mipSmoothingSelect.value === "on";
  status.splatReg = selectedSplatReg();
  status.framingMode = selectedFramingMode();
}

function syncClipLayoutControls(): void {
  if (selectedRepresentation() === "anisotropic") {
    clipLayoutSelect.value = "per_view";
    backgroundModeSelect.value = "black";
    coverageRegSelect.value = "off";
    rayRegSelect.value = "off";
    entropyRegSelect.value = "off";
    mipSmoothingSelect.value = "off";
    return;
  }
  const layout = selectedClipLayout();
  if (layout === "grid9_close2") {
    clipModeSelect.value = "3";
    viewBatchSelect.value = "9";
  } else if (layout === "dual_grid4") {
    clipModeSelect.value = "6";
    viewBatchSelect.value = "9";
  }
}

function setControlsDisabled(disabled: boolean): void {
  const grid = selectedClipLayout() === "grid9_close2" || selectedClipLayout() === "dual_grid4";
  const anisotropic = selectedRepresentation() === "anisotropic";
  optimizeBtn.disabled = disabled;
  resetBtn.disabled = disabled;
  qualityPresetSelect.disabled = disabled;
  representationSelect.disabled = disabled;
  viewSelect.disabled = disabled;
  promptModeSelect.disabled = disabled;
  bgTextModeSelect.disabled = disabled;
  backgroundModeSelect.disabled = disabled || anisotropic;
  alphaRegSelect.disabled = disabled;
  boundsRegSelect.disabled = disabled;
  coverageRegSelect.disabled = disabled || anisotropic;
  rayRegSelect.disabled = disabled || anisotropic;
  entropyRegSelect.disabled = disabled || anisotropic;
  stageModeSelect.disabled = disabled;
  adaptiveSplatsSelect.disabled = disabled;
  mipSmoothingSelect.disabled = disabled || anisotropic;
  splatRegSelect.disabled = disabled;
  framingModeSelect.disabled = disabled;
  clipLayoutSelect.disabled = disabled || anisotropic;
  gridPromptModeSelect.disabled = disabled || !grid;
  gridRasterModeSelect.disabled = disabled || !grid;
  gridStrengthSelect.disabled = disabled || !grid;
  viewBatchSelect.disabled = disabled || grid;
  viewSamplerSelect.disabled = disabled;
  clipModeSelect.disabled = disabled || grid;
}

async function rebuildOptimizer(nextSeed: number, phase: string): Promise<void> {
  status.phase = phase;
  status.clipLayout = selectedClipLayout();
  status.gridPromptMode = selectedGridPromptMode();
  status.gridDirectRaster = selectedGridDirectRaster();
  status.gridRasterSide = selectedGridRasterSide();
  status.gridStrength = selectedGridStrength();
  status.viewsPerStep = selectedViewsPerStep();
  status.viewSampler = selectedViewSampler();
  status.clipBatchSize = selectedClipBatchSize();
  syncPromptStatus();
  syncConvergenceStatus();
  renderReadout();
  const old = opt;
  opt = await createActiveOptimizer(nextSeed);
  status.clipLayout = opt.clipLayout;
  status.clipBatchSize = opt.clipBatchSize;
  status.viewSampler = opt.viewSampler;
  old?.destroy();
  populateViews();
  rebuildBlitBind();
  gridDirty = true;
  status.step = 0;
  latestTimings = null;
  latestTileTelemetry = null;
  renderTimings();
  renderReadout();
}

async function createActiveOptimizer(nextSeed: number): Promise<ActiveOptimizer> {
  if (selectedRepresentation() === "anisotropic") {
    return Splat3DAnisotropicOptimizer.create(device, plan, weights, {
      seed: nextSeed,
      clipBatchSize: status.clipBatchSize,
      viewSampler: selectedViewSampler(),
      lrs: selectedAnisotropicLrs(),
      convergence: selectedConvergenceConfig(),
      cameras: camerasForFraming(status.framingMode),
    });
  }
  const gridScales = selectedGridGradientScales();
  return Splat3DOptimizer.create(device, plan, weights, {
    seed: nextSeed,
    clipBatchSize: status.clipBatchSize,
    clipLayout: status.clipLayout,
    viewSampler: status.viewSampler,
    gridDirectRaster: status.gridDirectRaster,
    gridRasterSide: status.gridRasterSide,
    gridGradientScale: gridScales.grid,
    randomGridGradientScale: gridScales.randomGrid,
    lrs: selectedIsotropicLrs(),
    convergence: selectedConvergenceConfig(),
    cameras: status.clipLayout === "dual_grid4" ? dualGridCamerasForFraming(status.framingMode) : camerasForFraming(status.framingMode),
  });
}

const BLIT_WGSL = /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  return vec4<f32>(p[vi], 0.0, 1.0);
}

@group(0) @binding(0) var<storage, read> img : array<f32>;

@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
  let x : u32 = u32(pos.x);
  let y : u32 = u32(pos.y);
  let HW : u32 = ${HW}u;
  let i : u32 = y * ${SIDE}u + x;
  return vec4<f32>(img[i], img[HW + i], img[2u * HW + i], 1.0);
}
`;

async function buildBlitPipeline(): Promise<void> {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: BLIT_WGSL });
  blitPipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format: canvasFormat }] },
    primitive: { topology: "triangle-list" },
  });
  const err = await device.popErrorScope();
  if (err) throw new Error(`blit pipeline invalid: ${err.message}`);
}

function rebuildBlitBind(): void {
  blitBind = device.createBindGroup({
    layout: blitPipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: opt.raster.image } }],
  });
}

function recordBlit(enc: GPUCommandEncoder, target: GPUCanvasContext): void {
  if (!blitBind) return;
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: target.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
  });
  pass.setPipeline(blitPipe);
  pass.setBindGroup(0, blitBind);
  pass.draw(3);
  pass.end();
}

function renderGrid(): void {
  if (!blitBind || !viewCtxs.length) return;
  opt.prepareDisplayFrame();
  const enc = device.createCommandEncoder();
  for (let view = 0; view < viewCtxs.length; view++) {
    opt.raster.recordForward(enc, view);
    recordBlit(enc, viewCtxs[view]);
  }
  device.queue.submit([enc.finish()]);
}

const TF_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm";
const nativeImport = new Function("u", "return import(u)") as (u: string) => Promise<any>;
let tokenizer: any = null;
let textModel: any = null;
const promptEmbedCache = new Map<string, Promise<Float32Array>>();

async function loadTextModel(onProgress?: (msg: string) => void): Promise<void> {
  if (textModel) return;
  const tf: any = await nativeImport(TF_URL);
  tf.env.allowRemoteModels = true;
  const id = "Nbardy/nff-clip-splat-weights"; // self-hosted alongside the vision weights
  const progress_callback = (p: any) => {
    if (p.status === "progress" && p.total) {
      const pct = Math.round(p.progress ?? (p.loaded / p.total) * 100);
      const fill = Math.round((pct / 100) * 16);
      const bar = "█".repeat(fill) + "░".repeat(16 - fill);
      onProgress?.(`loading text encoder  [${bar}] ${pct}%  ·  ${(p.loaded / 1e6).toFixed(1)}/${(p.total / 1e6).toFixed(0)} MB`);
    }
  };
  tokenizer = await tf.AutoTokenizer.from_pretrained(id, { progress_callback });
  textModel = await tf.CLIPTextModelWithProjection.from_pretrained(id, {
    dtype: "fp16", // 84 MB, lossless vs fp32
    device: "wasm", // keep text off the shared render GPU
    session_options: { graphOptimizationLevel: "basic" }, // dodges the LayerNormFusion bug
    progress_callback,
  });
}

async function encodePrompt(text: string): Promise<Float32Array> {
  await loadTextModel();
  const enc = await tokenizer(text, {
    padding: "max_length",
    max_length: 77,
    truncation: true,
  });
  const out = await textModel(enc);
  const d = out.text_embeds.data as ArrayLike<number>;
  const vec = new Float32Array(512);
  for (let i = 0; i < 512; i++) vec[i] = d[i];
  return vec;
}

function encodePromptCached(text: string): Promise<Float32Array> {
  const key = text.trim();
  let cached = promptEmbedCache.get(key);
  if (!cached) {
    cached = encodePrompt(key).catch((e) => {
      promptEmbedCache.delete(key);
      throw e;
    });
    promptEmbedCache.set(key, cached);
  }
  return cached;
}

let viewEmbeds: Float32Array[] | null = null;
let stepsSinceReadout = 0;
let cosBusy = false;

async function runProfiledStep(): Promise<void> {
  if (!viewEmbeds || profileBusy) return;
  const profiledOpt = opt;
  const profiledView = displayView;
  const profiledViewsPerStep = status.viewsPerStep;
  profileBusy = true;
  status.profiling = true;
  status.phase = "profile";
  renderReadout();
  try {
    const timings = await profiledOpt.profileStep(profiledView, profiledViewsPerStep);
    if (profiledOpt !== opt || !status.running) return;
    latestTimings = timings;
    latestTileTelemetry = await profiledOpt.raster.readTileTelemetry();
    latestCloudTelemetry = summarizeCloud(
      await profiledOpt.raster.readParams(),
      profiledOpt.cameras[profiledView]
    );
    if (profiledOpt !== opt || !status.running) return;
    status.step = profiledOpt.stepCount;
    stepsSinceReadout += 1;
    gridDirty = true;
    renderTimings();
    if (stepsSinceReadout >= 3) {
      stepsSinceReadout = 0;
      void updateCos();
    }
  } catch (e: any) {
    fail(`profile step failed: ${e?.message ?? e}`);
  } finally {
    status.profiling = false;
    if (status.phase === "profile") status.phase = status.running ? "run" : "idle";
    profileBusy = false;
    renderReadout();
  }
}

function summarizeCloud(params: Float32Array, camera?: PreparedCamera3D): CloudTelemetry {
  const anisotropic = status.representation === "anisotropic";
  const stride = anisotropic ? ANISO_PARAM_STRIDE_3D : PARAM_STRIDE_3D;
  const G = Math.floor(params.length / stride);
  const center = [0, 0, 0];
  const opacities = new Float32Array(G);
  const axisRatios = new Float32Array(G);
  const screenAxisRatios = new Float32Array(G);
  let radiusSum = 0;
  let opacitySum = 0;
  for (let g = 0; g < G; g++) {
    center[0] += params[g * 3 + 0];
    center[1] += params[g * 3 + 1];
    center[2] += params[g * 3 + 2];
    const logRadius = anisotropic
      ? (params[3 * G + g * 3 + 0] + params[3 * G + g * 3 + 1] + params[3 * G + g * 3 + 2]) / 3
      : params[3 * G + g];
    radiusSum += Math.exp(logRadius);
    if (anisotropic) {
      const base = 3 * G + g * 3;
      const minLogScale = Math.min(params[base], params[base + 1], params[base + 2]);
      const maxLogScale = Math.max(params[base], params[base + 1], params[base + 2]);
      axisRatios[g] = Math.exp(maxLogScale - minLogScale);
      if (camera) {
        const quaternionBase = 6 * G + g * 4;
        const projected = projectAnisotropicGaussian(
          {
            position: [params[g * 3], params[g * 3 + 1], params[g * 3 + 2]],
            logScale: [params[base], params[base + 1], params[base + 2]],
            quaternion: [
              params[quaternionBase],
              params[quaternionBase + 1],
              params[quaternionBase + 2],
              params[quaternionBase + 3],
            ],
          },
          {
            eye: camera.eye,
            right: camera.right,
            up: camera.cameraUp,
            forward: camera.forward,
            focalPx: camera.focalPx,
            centerPx: [SIDE / 2, SIDE / 2],
            near: 0.2,
          }
        );
        const [c00, c01, c11] = projected.covariance;
        const discriminant = Math.sqrt(Math.max(0, (c00 - c11) ** 2 + 4 * c01 * c01));
        const lambdaMax = Math.max(1e-12, 0.5 * (c00 + c11 + discriminant));
        const lambdaMin = Math.max(1e-12, 0.5 * (c00 + c11 - discriminant));
        screenAxisRatios[g] = Math.sqrt(lambdaMax / lambdaMin);
      } else {
        screenAxisRatios[g] = 1;
      }
    } else {
      axisRatios[g] = 1;
      screenAxisRatios[g] = 1;
    }
    const raw = params[(anisotropic ? 13 : 7) * G + g];
    const opacity = raw >= 0 ? 1 / (1 + Math.exp(-raw)) : Math.exp(raw) / (1 + Math.exp(raw));
    opacities[g] = opacity;
    opacitySum += opacity;
  }
  center[0] /= G;
  center[1] /= G;
  center[2] /= G;
  let spread2 = 0;
  for (let g = 0; g < G; g++) {
    const dx = params[g * 3 + 0] - center[0];
    const dy = params[g * 3 + 1] - center[1];
    const dz = params[g * 3 + 2] - center[2];
    spread2 += dx * dx + dy * dy + dz * dz;
  }
  opacities.sort();
  axisRatios.sort();
  screenAxisRatios.sort();
  let axisRatioSum = 0;
  let screenAxisRatioSum = 0;
  for (const ratio of axisRatios) axisRatioSum += ratio;
  for (const ratio of screenAxisRatios) screenAxisRatioSum += ratio;
  return {
    opacityMean: opacitySum / G,
    opacityP10: opacities[Math.floor((G - 1) * 0.1)],
    opacityP90: opacities[Math.floor((G - 1) * 0.9)],
    radiusMean: radiusSum / G,
    spreadRms: Math.sqrt(spread2 / G),
    axisRatioMean: axisRatioSum / G,
    axisRatioP90: axisRatios[Math.floor((G - 1) * 0.9)],
    screenAxisRatioMean: screenAxisRatioSum / G,
    screenAxisRatioP90: screenAxisRatios[Math.floor((G - 1) * 0.9)],
  };
}

async function updateCos(): Promise<void> {
  if (!viewEmbeds || cosBusy) return;
  cosBusy = true;
  try {
    const emb = await opt.currentEmbedding(displayView);
    const c = cosine(emb, viewEmbeds[displayView]);
    status.cos = c;
    if (status.initialCos === null) status.initialCos = c;
    renderReadout();
  } finally {
    cosBusy = false;
  }
}

function frame(): void {
  if (status.running && viewEmbeds && !profileBusy) {
    const shouldProfile = opt.stepCount > 0 && opt.stepCount % PROFILE_PERIOD === 0;
    if (shouldProfile) {
      void runProfiledStep();
    } else {
      opt.step(displayView, status.viewsPerStep);
      gridDirty = true;
      stepsSinceReadout += 1;
      status.step = opt.stepCount;
      if (stepsSinceReadout >= 3) {
        stepsSinceReadout = 0;
        void updateCos();
      }
    }
  }
  if (gridDirty) {
    renderGrid();
    gridDirty = false;
  }
  requestAnimationFrame(frame);
}

async function onOptimize(): Promise<void> {
  if (!status.ready) return;
  if (profileBusy) return;
  syncClipLayoutControls();
  const text = promptInput.value.trim() || "a photo of a cat";
  setControlsDisabled(true);
  status.running = false;
  status.phase = "encoding";
  status.cos = null;
  status.initialCos = null;
  latestTimings = null;
  latestTileTelemetry = null;
  status.clipLayout = selectedClipLayout();
  status.gridPromptMode = selectedGridPromptMode();
  status.gridDirectRaster = selectedGridDirectRaster();
  status.gridRasterSide = selectedGridRasterSide();
  status.gridStrength = selectedGridStrength();
  status.viewsPerStep = selectedViewsPerStep();
  status.viewSampler = selectedViewSampler();
  status.clipBatchSize = selectedClipBatchSize();
  status.promptMode = selectedPromptMode();
  status.bgPromptMode = selectedBgPromptMode();
  syncConvergenceStatus();
  renderTimings();
  renderReadout();
  try {
    const embeds: Float32Array[] = [];
    if (status.promptMode === "same") {
      setNotice("encoding prompt 1/1...");
      const embed = await encodePromptCached(buildBasePrompt(text, status.bgPromptMode));
      for (let i = 0; i < opt.cameras.length; i++) embeds.push(embed);
    } else {
      for (let i = 0; i < opt.cameras.length; i++) {
        setNotice(`encoding prompt ${i + 1}/${opt.cameras.length}...`);
        const promptText =
          status.promptMode === "coarse"
            ? buildCoarseViewPrompt(text, opt.cameras[i], status.bgPromptMode)
            : buildViewPrompt(text, opt.cameras[i], status.bgPromptMode);
        embeds.push(await encodePromptCached(promptText));
      }
	    }
	    viewEmbeds = embeds;
	    opt.setViewPrompts(embeds);
    if (status.clipLayout === "grid9_close2" || status.clipLayout === "dual_grid4") {
      setNotice("encoding grid prompt...");
      opt.setGridPrompt(await encodePromptCached(buildGrid9Prompt(text, status.bgPromptMode, status.gridPromptMode)));
    }
    if (status.clipLayout === "dual_grid4") {
      setNotice("encoding random grid prompt...");
      opt.setRandomGridPrompt(await encodePromptCached(buildRandomGrid9Prompt(text, status.bgPromptMode)));
      setNotice("encoding zoom prompt...");
      opt.setZoomPrompt(await encodePromptCached(buildZoomPrompt(text, status.bgPromptMode)));
    }
	    const e0 = await opt.currentEmbedding(displayView);
    status.initialCos = cosine(e0, embeds[displayView]);
    status.cos = status.initialCos;
    stepsSinceReadout = 0;
    setNotice("");
    status.phase = "run";
    status.running = true;
    gridDirty = true;
    renderReadout();
	  } catch (e: any) {
	    fail(`text encode failed: ${e?.message ?? e}`);
	  } finally {
	    setControlsDisabled(false);
	  }
}

async function onReset(): Promise<void> {
  if (!status.ready) return;
  if (profileBusy) {
    setNotice("wait for profiling sample to finish before reset");
    return;
  }
  status.running = false;
  viewEmbeds = null;
  status.cos = null;
  status.initialCos = null;
  latestTimings = null;
  latestTileTelemetry = null;
  status.phase = "reset";
  seed += 1;
  await rebuildOptimizer(seed, "reset");
  status.phase = "idle";
  setNotice("");
  renderReadout();
}

async function onViewChange(): Promise<void> {
  setDisplayView(Math.max(0, viewSelect.selectedIndex));
  if (!status.ready) return;
  status.cos = null;
  status.initialCos = null;
  if (viewEmbeds) void updateCos();
  renderReadout();
}

function setDisplayView(view: number): void {
  displayView = Math.max(0, Math.min(viewCtxs.length ? viewCtxs.length - 1 : 0, view | 0));
  status.view = displayView;
  viewSelect.selectedIndex = displayView;
  for (let i = 0; i < viewTiles.length; i++) {
    viewTiles[i].classList.toggle("active", i === displayView);
  }
}

function onPromptModeChange(): void {
  markManualPreset();
  syncPromptStatus();
  latestTimings = null;
  latestTileTelemetry = null;
  if (viewEmbeds) {
    status.running = false;
    viewEmbeds = null;
    status.cos = null;
    status.initialCos = null;
    status.phase = "idle";
    setNotice("");
  }
  renderTimings();
  renderReadout();
}

async function onConvergenceSettingsChange(): Promise<void> {
  if (!status.ready) return;
  if (profileBusy) {
    setNotice("wait for profiling sample to finish before changing convergence settings");
    backgroundModeSelect.value = status.backgroundMode;
    alphaRegSelect.value = status.alphaReg;
    boundsRegSelect.value = status.boundsReg;
    coverageRegSelect.value = status.coverageReg;
    rayRegSelect.value = status.rayReg;
    entropyRegSelect.value = status.entropyReg;
    stageModeSelect.value = status.stageMode;
    adaptiveSplatsSelect.value = status.adaptiveSplats ? "on" : "off";
    mipSmoothingSelect.value = status.mipSmoothing ? "on" : "off";
    representationSelect.value = status.representation;
    splatRegSelect.value = status.splatReg;
    framingModeSelect.value = status.framingMode;
    return;
  }
  markManualPreset();
  syncConvergenceStatus();
  status.running = false;
  viewEmbeds = null;
  status.cos = null;
  status.initialCos = null;
  latestTimings = null;
  latestTileTelemetry = null;
  setControlsDisabled(true);
  try {
    await rebuildOptimizer(seed, "convergence");
    setNotice("");
    status.phase = "idle";
  } catch (e: any) {
    fail(`convergence settings change failed: ${e?.message ?? e}`);
  } finally {
    setControlsDisabled(false);
    renderReadout();
  }
}

async function onRepresentationChange(): Promise<void> {
  syncClipLayoutControls();
  await onConvergenceSettingsChange();
}

async function onQualityPresetChange(): Promise<void> {
  const previous = status.qualityPreset;
  const preset = selectedQualityPreset();
  status.qualityPreset = preset;
  if (!status.ready) {
    applyQualityPresetToControls(preset);
    syncPromptStatus();
    syncConvergenceStatus();
    status.viewsPerStep = selectedViewsPerStep();
    status.viewSampler = selectedViewSampler();
    status.clipBatchSize = selectedClipBatchSize();
    renderReadout();
    return;
  }
  if (profileBusy) {
    setNotice("wait for profiling sample to finish before changing quality preset");
    qualityPresetSelect.value = previous;
    status.qualityPreset = previous;
    return;
  }
  applyQualityPresetToControls(preset);
  syncClipLayoutControls();
  status.running = false;
  viewEmbeds = null;
  status.cos = null;
  status.initialCos = null;
  latestTimings = null;
  latestTileTelemetry = null;
  setControlsDisabled(true);
  try {
    await rebuildOptimizer(seed, "preset");
    setNotice("");
    status.phase = "idle";
  } catch (e: any) {
    fail(`quality preset change failed: ${e?.message ?? e}`);
  } finally {
    setControlsDisabled(false);
    renderReadout();
  }
}

function onViewBatchChange(): void {
  markManualPreset();
  syncClipLayoutControls();
  status.viewsPerStep = selectedViewsPerStep();
  latestTimings = null;
  latestTileTelemetry = null;
  renderTimings();
  renderReadout();
}

async function onClipSettingsChange(): Promise<void> {
  if (!status.ready) return;
  syncClipLayoutControls();
  if (profileBusy) {
    setNotice("wait for profiling sample to finish before changing CLIP settings");
    clipModeSelect.value = String(status.clipBatchSize);
    clipLayoutSelect.value = status.clipLayout;
    viewSamplerSelect.value = status.viewSampler;
    gridRasterModeSelect.value = gridRasterModeValue(status.gridRasterSide);
    gridStrengthSelect.value = status.gridStrength;
    syncClipLayoutControls();
    return;
  }
  markManualPreset();
  status.running = false;
  viewEmbeds = null;
  status.cos = null;
  status.initialCos = null;
  latestTimings = null;
  latestTileTelemetry = null;
  setControlsDisabled(true);
  try {
    await rebuildOptimizer(seed, "optimizer");
    setNotice("");
    status.phase = "idle";
  } catch (e: any) {
    fail(`clip settings change failed: ${e?.message ?? e}`);
  } finally {
    setControlsDisabled(false);
    renderReadout();
  }
}

function populateViews(): void {
  viewSelect.textContent = "";
  gridEl.textContent = "";
  viewCtxs = [];
  viewTiles = [];
  const visibleViews = Math.min(9, opt.cameras.length);
  for (let i = 0; i < visibleViews; i++) {
    const camera = opt.cameras[i];
    const option = document.createElement("option");
    option.value = camera.name;
    option.textContent = camera.name;
    viewSelect.appendChild(option);

    const tile = document.createElement("div");
    tile.className = "tile";
    const canvas = document.createElement("canvas");
    canvas.className = "view";
    canvas.width = SIDE;
    canvas.height = SIDE;
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = camera.name;
    tile.append(canvas, label);
    tile.addEventListener("click", () => {
      setDisplayView(i);
      status.cos = null;
      status.initialCos = null;
      if (viewEmbeds) void updateCos();
      renderReadout();
    });
    const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
    ctx.configure({ device, format: canvasFormat, alphaMode: "opaque" });
    gridEl.appendChild(tile);
    viewCtxs.push(ctx);
    viewTiles.push(tile);
  }
  setDisplayView(displayView);
}

async function boot(): Promise<void> {
  if (!navigator.gpu) {
    fail("this page needs WebGPU (no navigator.gpu) — use Chrome/Edge with WebGPU enabled.");
    setControlsDisabled(true);
    return;
  }
  status.phase = "adapter";
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return fail("no WebGPU adapter available.");
  device = await adapter.requestDevice();
  device.addEventListener?.("uncapturederror", (ev: any) => {
    console.error("[webgpu]", ev.error?.message ?? ev.error);
  });

  canvasFormat = navigator.gpu.getPreferredCanvasFormat();

  status.phase = "weights";
  try {
    const assets = await loadClipTrainAssets((msg) => {
      readoutEl.textContent = msg;
    });
    plan = assets.plan;
    weights = assets.weights;
  } catch (e: any) {
    return fail(e?.message ?? String(e));
  }

  status.phase = "optimizer";
  readoutEl.textContent = "building 3D optimizer…";
  await buildBlitPipeline();
  applyQualityPresetToControls(selectedQualityPreset());
  syncClipLayoutControls();
  syncPromptStatus();
  syncConvergenceStatus();
  opt = await createActiveOptimizer(seed);
  status.clipLayout = opt.clipLayout;
  status.viewsPerStep = selectedViewsPerStep();
  status.viewSampler = opt.viewSampler;
  status.clipBatchSize = opt.clipBatchSize;
  status.gridPromptMode = selectedGridPromptMode();
  status.gridDirectRaster = selectedGridDirectRaster();
  status.gridRasterSide = selectedGridRasterSide();
  status.gridStrength = selectedGridStrength();
  populateViews();
  rebuildBlitBind();
  gridDirty = true;

  // Preload the text encoder at boot (with its own progress bar) so the first
  // Optimize is instant instead of stalling on an 84 MB download (× the 9 views).
  status.phase = "textmodel";
  await loadTextModel((msg) => { readoutEl.textContent = msg; });

  status.ready = true;
  status.phase = "idle";
  setControlsDisabled(false);
  setNotice("");
  renderReadout();
  requestAnimationFrame(frame);
}

optimizeBtn.addEventListener("click", () => void onOptimize());
resetBtn.addEventListener("click", () => void onReset());
viewSelect.addEventListener("change", () => void onViewChange());
qualityPresetSelect.addEventListener("change", () => void onQualityPresetChange());
representationSelect.addEventListener("change", () => void onRepresentationChange());
promptModeSelect.addEventListener("change", onPromptModeChange);
bgTextModeSelect.addEventListener("change", onPromptModeChange);
backgroundModeSelect.addEventListener("change", () => void onConvergenceSettingsChange());
alphaRegSelect.addEventListener("change", () => void onConvergenceSettingsChange());
boundsRegSelect.addEventListener("change", () => void onConvergenceSettingsChange());
coverageRegSelect.addEventListener("change", () => void onConvergenceSettingsChange());
rayRegSelect.addEventListener("change", () => void onConvergenceSettingsChange());
entropyRegSelect.addEventListener("change", () => void onConvergenceSettingsChange());
stageModeSelect.addEventListener("change", () => void onConvergenceSettingsChange());
adaptiveSplatsSelect.addEventListener("change", () => void onConvergenceSettingsChange());
mipSmoothingSelect.addEventListener("change", () => void onConvergenceSettingsChange());
splatRegSelect.addEventListener("change", () => void onConvergenceSettingsChange());
framingModeSelect.addEventListener("change", () => void onConvergenceSettingsChange());
viewBatchSelect.addEventListener("change", onViewBatchChange);
viewSamplerSelect.addEventListener("change", () => void onClipSettingsChange());
clipModeSelect.addEventListener("change", () => void onClipSettingsChange());
clipLayoutSelect.addEventListener("change", () => void onClipSettingsChange());
gridPromptModeSelect.addEventListener("change", onPromptModeChange);
gridRasterModeSelect.addEventListener("change", () => void onClipSettingsChange());
gridStrengthSelect.addEventListener("change", () => void onClipSettingsChange());
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") void onOptimize();
});

boot().catch((e) => fail(`boot failed: ${e?.message ?? e}`));
