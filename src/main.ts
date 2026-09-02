/**
 * Neural Force Field Art — Gallery Engine
 *
 * Core algorithm:
 *   1. Neural network predicts force vectors from particle positions
 *   2. Forces are applied as acceleration to particles (velocity + position update)
 *   3. Loss is computed on the resulting positions (spiral distance, center distance)
 *   4. Gradients flow back through the entire physics chain to update model weights
 *   5. Random reset keeps particles exploring instead of collapsing
 *
 * The model DISCOVERS how to move particles creatively while minimising a
 * simple constraint — it is NOT told the answer directly.
 */
import * as tf from "@tensorflow/tfjs";
// Registers the 'webgpu' backend with tfjs. The @tensorflow/tfjs union package
// ships only cpu + webgl backends; without this import tf.setBackend('webgpu')
// throws "Backend name 'webgpu' not found in registry".
import "@tensorflow/tfjs-backend-webgpu";
// Only the TYPE from renderers.ts: this page renders WebGPU-only, so the
// Canvas2D factory (createRenderer) is never called here — it still serves
// the legacy pages and tools/surprise_test.ts.
import { RendererType } from "./renderers";
import { HelmholtzField } from "./core/field/helmholtz";
import type { ForceField } from "./core/field/helmholtz";
import {
  ARCH,
  ARCH_DOCK_DUAL,
  ARCH_DOCK_PRESETS,
  applyArchDockPreset,
  archDockPresets,
  createFieldFromArch,
  describeFieldArch,
  isArchPresetKey,
  type ArchDockKind,
  type ArchPresetKey,
  type FieldArch,
} from "./core/field/arch";
export {
  ARCH,
  ARCH_DOCK_DUAL,
  ARCH_DOCK_PRESETS,
  applyArchDockPreset,
  archDockPresets,
  createFieldFromArch,
  describeFieldArch,
  isArchPresetKey,
  type ArchDockKind,
  type ArchPresetKey,
  type FieldArch,
} from "./core/field/arch";
export type { ForceField } from "./core/field/helmholtz";
export { HelmholtzField } from "./core/field/helmholtz";
// isotropyLoss only: the chaos + divergence probes used by this file's losses
// are inlined (they share one 3×-batched field forward — see
// helmholtzChaosLoss); the standalone chaosLoss/divergencePenalty exports are
// consumed by the legacy pages and tests, not by the gallery loop.
import {
  constantModeFraction,
  directionOrderLoss,
  directionOrderParameters,
  isotropyLoss,
} from "./core/losses";
// OPTIONAL zero-copy GPU renderer (perf lane). Imported so it compiles and is
// ready, but only used when a preset sets `gpu: true` (none do by default — it
// needs browser QA). See src/render/gpuPoints.ts.
import { GpuPointRenderer } from "./render/gpuPoints";
import { GpuPointRendererWebGPU } from "./render/webgpu/points";
import { SplatRenderer, SplatStyle } from "./render/webgpu/splat";
import { AdvectKernel } from "./render/webgpu/advect";
import type { BorderMode, FieldLayout } from "./render/webgpu/advect_wgsl";
import { totalMacs } from "./render/webgpu/advect_wgsl";
import { FusedTrainer, type GeneratorLearningRates } from "./render/webgpu/train";
import { MAX_BATCH, type FieldLossSpec } from "./render/webgpu/train_wgsl";
import { FieldProbe, type FieldHealth } from "./render/webgpu/field_probe";
import {
  HEALTH_GRID_N,
  HEALTH_PERIOD_MS,
  l2Norm,
  type AdvHealth,
  type HealthSnapshot,
  type HealthWindow,
  type PixelHealth,
} from "./health";
// FUSED ADVERSARY (adversary_train.ts + adversary_wgsl.ts): the WGSL port of
// the tfjs adversary below — discriminator train + generator reward + the
// packed raw/per-unit surprise buffer, ~0.7-0.8 ms/step at B=512 vs tfjs's
// 19-32 ms.
// Oracle-gated by tools/train_wta_test.ts (cos = 1.0000000 vs the AD-IR
// oracle AND tf.variableGrads) and tools/train_wta_hashgrid_test.ts /
// tools/train_wta_pressure_test.ts. Classless fields AND family-PLANED
// hashgrid fields (tools/family_grid_test.ts); one-hot class channels still
// fall back to the tfjs path below, loudly.
import {
  AdversaryTrainer,
  type SurpriseMetric,
  type DirectionOrder,
  type FamilyPayoff,
} from "./render/webgpu/adversary_train";
import { PixelDiscTrainer } from "./render/webgpu/pixel_disc_train";
import { classifyPixelDiscFusion } from "./render/webgpu/pixel_disc_wgsl";
import { GpuTimer } from "./render/webgpu/gputime";
// ADVERSARY (src/core/gan/adversary.ts): a relaxed winner-take-all
// multiple-choice predictor whose irreducible residual is the generator's
// reward. Pure tfjs — the numeric REFERENCE for the fused port above, and
// still the live path for ?train=tfjs / unsupported field types.
import {
  Adversary,
  headCount,
  encodingDims,
  defaultAdversaryConfig,
  disposeTupleSample,
  type AdversaryKind,
  type GuessKind,
  type AdversaryTarget,
  type AdversaryLoss,
  type TupleEncoding,
  type HeadSpread,
  type ObserverGeometry,
  type RewardScale,
  DEFAULT_ADVERSARY_TARGET,
  DEFAULT_ADVERSARY_LOSS,
  objectiveDims,
} from "./core/gan/adversary";
import { GpuSurpriseRendererWebGPU, GpuSurpriseStats } from "./render/webgpu/surprise_points";
import type { ColormapName } from "./draw/colormap";

// tfjs-backend-webgpu 4.10 calls adapter.requestAdapterInfo(), which current
// Chrome removed in favour of the synchronous `adapter.info` property — without
// this shim the webgpu backend fails to init ("requestAdapterInfo is not a
// function"). Safe no-op where WebGPU is absent (GPUAdapter undefined).
{
  const GA = (globalThis as any).GPUAdapter;
  if (GA && !GA.prototype.requestAdapterInfo) {
    GA.prototype.requestAdapterInfo = function () {
      return Promise.resolve((this as any).info ?? {});
    };
  }
  // tfjs also OWNS GPUDevice creation and requests only the features it wants
  // for itself — but device features are fixed at creation time, so anything
  // OUR kernels need must be appended to tfjs's requestDevice call. Wrap it to
  // add "shader-f16" (advect kernel's f16 fast path, see render/webgpu/
  // advect.ts) and "timestamp-query" (upcoming GPU profiling), each only when
  // the adapter reports it, deduped, preserving every other descriptor field.
  if (GA && !(GA.prototype as any).__nffaRequestDeviceWrapped) {
    (GA.prototype as any).__nffaRequestDeviceWrapped = true;
    const origRequestDevice = GA.prototype.requestDevice;
    GA.prototype.requestDevice = function (
      desc?: GPUDeviceDescriptor
    ): Promise<GPUDevice> {
      const extras = ["shader-f16", "timestamp-query"].filter((f) =>
        (this as GPUAdapter).features.has(f as GPUFeatureName)
      ) as GPUFeatureName[];
      const requiredFeatures = [
        ...new Set([...(desc?.requiredFeatures ?? []), ...extras]),
      ];
      return origRequestDevice.call(this, { ...desc, requiredFeatures });
    };
  }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
/**
 * Extra context handed to {@link ArtPieceConfig.computeLoss} on each step.
 * `force` is the per-step force tensor `[N,2]` (already scaled by
 * `forceMagnitude`) that produced the current positions — reused so
 * force-based losses (isotropy) do not recompute the field. `field`, when
 * present, is the live {@link ForceField} so field-sampling losses (chaos,
 * divergence) can probe it at arbitrary positions. Both are `undefined` for
 * the legacy MLP pieces, which ignore this argument entirely.
 */
export interface LossContext {
  force?: tf.Tensor2D;
  field?: ForceField | null;
}

/**
 * Whether a piece plays the adversarial prediction game, and how.
 *
 * A SUM TYPE rather than an optional bag of knobs: "no adversary" is a real
 * variant with its own handler everywhere (loss composition, trainer selection,
 * HUD), not a null that every call site has to re-check.
 */
export type AdversarySpec =
  | { readonly tag: "off" }
  | {
      readonly tag: "on";
      readonly kind: AdversaryKind;
      readonly encoding: TupleEncoding;
      /** Predicted physical quantity. Omitted legacy specs resolve to raw F(x). */
      readonly target?: AdversaryTarget;
      /**
       * Prediction metric and generator game. Omitted legacy specs retain the
       * historical raw-vector game, except the legacy adjusted R+S observer,
       * which resolves to the exact smooth soft-angle game.
       */
      readonly loss?: AdversaryLoss;
      /** A opposes D while B cooperates with D; omitted = ordinary disagree. */
      readonly game?: "disagree" | "agree-disagree";
      /** Multiplier on the generator reward before it joins the aesthetic loss. */
      readonly weight: number;
      /**
       * Anti-collapse pressure on the generator. Omitted resolves to
       * {@link DEFAULT_GAME_PRESSURE} — none, i.e. the shipped behaviour.
       */
      readonly pressure?: GamePressure;
      /**
       * The PREDICTOR's own net. Omitted resolves to
       * {@link PREDICTOR_ARCH_DEFAULT} — 32/16, which is what every shipped
       * piece and every number in agent_notes/ was measured at.
       */
      readonly predictor?: PredictorArch;
    };

/**
 * THE OTHER MODEL IN A RELATIONAL GAME. `FieldArch` describes the generator;
 * this describes each of the K discriminator heads, whose shape is
 * `du → hiddenUnits selu → featureDim selu → dy linear` (du and dy come from
 * the OBSERVER, not from here — see `fusedObjectiveDims`).
 *
 * `AdversaryTrainer` and the tfjs `Adversary` have both accepted these two
 * numbers since the port, and until now NO caller passed either, so the
 * predictor was the one model in this project that had never been varied:
 * every reading in agent_notes/ is at 32/16. That is worth remembering when
 * reading a result as a property of "the adversary" — it is a property of a
 * 32/16 adversary.
 *
 * The fused codegen is general over depth and width; the ONLY predictor
 * restriction is activation, because `emitBwdStore` needs the pre-activation
 * checkpoints SELU keeps and `sin` does not (`validateAdversaryFusion`). Note
 * that refusal walks the PREDICTOR, so a SIREN *generator* (`dualSiren`) is
 * fine — an easy misreading of the validator.
 */
export interface PredictorArch {
  readonly hiddenUnits: number;
  readonly featureDim: number;
}

/** What every shipped piece ran before the knob existed. Do not "improve". */
export const PREDICTOR_ARCH_DEFAULT: PredictorArch = {
  hiddenUnits: 32,
  featureDim: 16,
};

/** Widths the dock offers. Labels are what the dock shows. */
export const PREDICTOR_ARCH_DOCK: readonly {
  readonly key: string;
  readonly label: string;
  readonly arch: PredictorArch;
}[] = [
  { key: "tiny", label: "16/8", arch: { hiddenUnits: 16, featureDim: 8 } },
  { key: "std", label: "32/16", arch: PREDICTOR_ARCH_DEFAULT },
  { key: "wide", label: "64/32", arch: { hiddenUnits: 64, featureDim: 32 } },
  { key: "wider", label: "128/64", arch: { hiddenUnits: 128, featureDim: 64 } },
] as const;

/**
 * GENERATOR-SIDE PRESSURE that keeps the game from paying for a DEAD field.
 *
 * Measured mechanism (agent_notes/2026-08-17_120215_KST_collapse_fix.md,
 * reproduce with `PRESET=pair bun tools/collapse_probe.ts`): the soft-angle
 * payoff embeds a 2-vector as ψτ(z)=(z,τ)/√(‖z‖²+τ²), so the ZERO vector is
 * the sphere's north pole and sits chord √2 ≈ 1.414 away from EVERY
 * equatorial prediction — while a genuinely direction-varied target only
 * earns the K-way angular quantization error, ≈ 0.44 at K=4/ε=0.05. The
 * generator therefore has a 3.15× incentive to drive the ENCODED TARGET to
 * zero, and on the pair observer y = F(x₂) − F(x₁), so "target zero" IS
 * "spatially constant field" — laminar streaks. It is not a gradient bug and
 * no amount of retuning removes it; the payoff's own maximum is the dead field.
 *
 * This is the counter-pressure, and it is a property of the FIELD (raw F over
 * the batch), not of the predictor — hence a named term composed at the same
 * one site as the reward, never folded into the adversary's own objective.
 */
export type GamePressure =
  /** Shipped behaviour: the game is the only thing steering the field. */
  | { readonly tag: "none" }
  /**
   * Penalize the batch's DIRECTION order parameters (see
   * {@link directionOrderLoss}). `polar` alone is escapable by a ±F₀ field,
   * so `nematic` is part of the same variant rather than a separate mode.
   */
  | {
      readonly tag: "anti-collapse";
      readonly polar: number;
      readonly nematic: number;
      /** Direction softener; normally the objective's own soft-angle τ. */
      readonly tau: number;
    };

export const DEFAULT_GAME_PRESSURE: GamePressure = { tag: "none" };

/** How the particle cloud is coloured. */
export type ColorMode =
  /** The shipped look: hue from speed / species (splat + quad renderers). */
  | { readonly tag: "velocity" }
  /** Raw shared relaxed-WTA payoff, in predictor-output units. */
  | { readonly tag: "surprise-raw"; readonly colormap: ColormapName }
  /** Raw payoff divided by the active target norm, for a scale-neutral display. */
  | { readonly tag: "surprise-per-unit"; readonly colormap: ColormapName };

export type SurpriseColorMode = Exclude<ColorMode, { readonly tag: "velocity" }>;

export function isSurpriseColorMode(m: ColorMode): m is SurpriseColorMode {
  return m.tag === "surprise-raw" || m.tag === "surprise-per-unit";
}

export function surpriseMetricOf(m: ColorMode): SurpriseMetric | null {
  switch (m.tag) {
    case "velocity":
      return null;
    case "surprise-raw":
      return "raw-payoff";
    case "surprise-per-unit":
      return "per-unit-signal";
    default:
      return assertNeverPiece(m, "surpriseMetricOf");
  }
}

/**
 * A piece's pixel-space density critic. Named (rather than inline in
 * {@link ArtPieceConfig}) so the routing decision can carry the very spec it
 * approved — see {@link PixelCriticPlan}.
 */
export interface PixelCriticSpec {
  readonly weight: number;
  /** One of the four Pixel GAN games — see docs/PIXEL_DISC.md. */
  readonly kind?: "vec-field" | "next-frame" | "real-fake" | "inpaint";
  readonly G?: number;
  readonly E?: number;
  readonly K?: number;
  readonly hidden?: number;
  readonly dt?: number;
  /**
   * How many guesses the critic head gets, and how they are weighted.
   *
   * SAME TYPE as the relational adversary's `kind` — `AdversaryKind` is
   * `GuessKind` (src/core/gan/adversary.ts), and src/core/gan/wta.ts is the one
   * spec four backends already share. The two games differ in what they OBSERVE
   * (tuples vs a density image); they do not differ in what a relaxed
   * winner-take-all head is, so this knob is deliberately not a second spelling
   * of it.
   *
   * Absent ≡ {tag:"single"}, canonicalized in `guessesOf` (pixel_disc.ts) — the
   * ONE place. `real-fake` accepts only `single` and says so loudly
   * (`validatePixelDims`): it scores a LABEL through BCE, so it has no per-cell
   * winner to take all.
   */
  readonly guesses?: GuessKind;
  /** Optional bounded replay of prior real density snapshots for RealFake. */
  readonly historicalReplay?: {
    /** Number of G×G density snapshots retained in the rolling population. */
    readonly capacity: number;
    /** Capture one live real snapshot every N critic steps. */
    readonly captureEvery?: number;
    /** Probability of comparing against history instead of uniform fake noise. */
    readonly probability?: number;
    /** Normalization horizon, in critic steps, for age conditioning. */
    readonly horizon?: number;
  };
}

export interface ArtPieceConfig {
  name: string;
  /** Curated artwork shown in the bottom named-piece strip. */
  named?: boolean;
  particleCount: number;
  friction: number;
  forceMagnitude: number;
  /**
   * Optional dimensionless physical-drive bound.
   *
   * For tanh-bounded raw field components |F| <= 1, choosing
   *
   *   forceMagnitude = drive * maxVelocity * (1-friction) / friction
   *
   * makes |v| <= drive*maxVelocity an invariant of
   * v' = friction*(v + forceMagnitude*F). Only adversary pieces opt into this;
   * legacy/non-adversary pieces retain their literal forceMagnitude unchanged.
   */
  drive?: number;
  maxVelocity: number;
  resetRate: number;
  /** Initial train-B sample count for this piece; absent defaults to 256. */
  sampleRate?: number;
  drawRate: number;
  /**
   * Field/generator Adam learning rate. On a standalone adversary piece the
   * external game gradient is the only field gradient, so Adam largely removes
   * a scalar reward-weight rescale; this LR is the effective generator-step
   * control. The predictor/discriminator LR defaults separately to 3e-3; both
   * are live controls because their ratio changes the game, not just its speed.
   */
  learningRate: number;
  /** Initial predictor/discriminator Adam LR; absent defaults to 3e-3. */
  discriminatorLearningRate?: number;
  /** Initial border mode; an explicit start-loop override wins. */
  border?: BorderMode;
  backgroundColor: [number, number, number];
  alphaBlend: number;
  renderer: RendererType;
  /** Optional per-packed-segment generator rates for fused dual-head fields. */
  generatorLearningRates?: GeneratorLearningRates;
  /** Explicit particle palette; Agree+Disagree uses stable exact RGB roles. */
  palette?: "speed" | "species" | "rgb-roles" | "rgb-families" | "optimizer-groups-v1";
  /** Initial diagnostic colour mode for a fresh gallery load. */
  colorMode?: ColorMode;
  mode?: "standard" | "agree-disagree";
  /**
   * Legacy path: a sigmoid MLP whose `[0,1]` output is re-centered by
   * `(raw - 0.5)`. Mutually exclusive with {@link createField}.
   */
  createModel?: () => tf.Sequential;
  /**
   * Field path: a {@link ForceField} (e.g. {@link HelmholtzField}) whose raw
   * signed output is used directly (NO `-0.5` shift). Its `trainableWeights`
   * become the optimizer varList. Takes precedence over {@link createModel}.
   * Prefer {@link fieldArch} when the architecture is declarative.
   */
  createField?: () => ForceField;
  /**
   * Declarative force-field architecture (encoding / widths / heads).
   * Used when {@link createField} is omitted: `createFieldFromArch(fieldArch)`.
   * Orthogonal to loss and renderer. Dock presets can override via startLoop.
   */
  fieldArch?: FieldArch;
  /**
   * When true, the model dock may swap among {@link archDockPresets}.
   * Default false for game pieces whose arch is load-bearing.
   */
  archEditable?: boolean;
  /**
   * Which dock preset list when {@link archEditable}. Default `"aesthetic"`
   * (single-head). Chaos comparison pieces use `"dual"`.
   */
  archDock?: ArchDockKind;
  /**
   * When true, the ink dock exposes Ghost / Clean / Trails look presets
   * (decay only on the WebGPU splat path). Default false for instrument
   * renderers like surprise.
   */
  lookEditable?: boolean;
  /**
   * Structural/aesthetic objective compiled into the fused field trainer.
   * This is independent of architecture: arch does not imply loss.
   * Omitted ⇒ skip the fused trainer so custom aesthetic `computeLoss`
   * (e.g. Spiral Cover) is not replaced by the default chaos objective.
   */
  fieldLoss?: FieldLossSpec;
  /**
   * OPTIONAL: route rendering through the zero-copy {@link GpuPointRenderer}
   * instead of the Canvas2D renderers. Off by default (needs browser QA); no
   * shipped preset sets it.
   */
  gpu?: boolean;
  /**
   * Adversary game for this piece. Absent ≡ `{ tag: "off" }` — canonicalized
   * exactly once, in {@link resolveAdversary}; no handler downstream sees
   * `undefined`.
   *
   * The adversary term is NOT part of `computeLoss`. The loop composes
   * `computeLoss(...) + weight · generatorReward` at ONE site (see `tick`), so
   * a piece's aesthetic loss stays a pure function of positions and any
   * aesthetic loss can be paired with any adversary without rewriting it.
   */
  adversary?: AdversarySpec;
  /**
   * Pixel-space density discriminator (cheap conv→codebook→GAP→MLP on a
   * low-res soft splat). Optional and independent of {@link adversary}: the
   * critic observes an image of the cloud, not relational tuples. Generator
   * reward flows through a differentiable soft density (virtual one-step) into
   * extGrads — reverse-mode, not JVP. See src/core/gan/pixel_disc.ts.
   */
  pixelDisc?: PixelCriticSpec;
  /**
   * Colormap for `renderer: "surprise"` pieces. Absent ≡ "inferno"
   * (canonicalized in {@link resolveColorMode}); ignored by every other
   * renderer, which colours by velocity.
   */
  colormap?: ColormapName;
  /**
   * Splat draw style this piece wants on load. Absent ≡ "dot" (the shipped
   * look), canonicalized exactly once in {@link resolveStrokeStyle}, which
   * fixes the order `?stroke=` > this > "dot". The dock's live control owns
   * the value after startup — this is an initial condition, not a lock.
   *
   * ONLY meaningful on splat-rendered pieces. A `renderer: "surprise"` piece
   * hands the whole render pass to the surprise renderer (see the render step
   * in `tick`), which has no stroke concept, so a stroke declared there would
   * be a dead flag.
   */
  stroke?: SplatStyle;
  /** Stroke length in FRAMES of travel; same resolution order via
   *  {@link resolveStrokeLength}. Absent ≡ 3. Ignored while stroke is "dot". */
  strokeLen?: number;
  computeLoss: (
    pos: tf.Tensor2D,
    w: number,
    h: number,
    ctx?: LossContext
  ) => tf.Scalar;
}

// ---------------------------------------------------------------------------
// κ — URL / piece-default ingestion for the adversary and the colour mode.
//
// This is the ONLY place a query string is turned into an adversary decision.
// Everything past it consumes canonical values, so no handler re-parses,
// re-clamps or re-defaults.
// ---------------------------------------------------------------------------

/** Canonical reward range shared by URL ingestion and the live UI. */
export const ADVERSARY_WEIGHT_RANGE = { min: 0, max: 0.05 } as const;

/** Piece-independent starting point when `?adv=` turns an adversary ON.
 * Weight uses the current dimensionless reward units and must be representable
 * by the live control.
 */
const ADV_DEFAULTS = { k: 8, eps: 0.05, weight: 0.012, m: 1 } as const;

function intParam(q: URLSearchParams, key: string, fallback: number): number {
  const raw = q.get(key);
  if (raw === null) return fallback;
  const v = parseInt(raw, 10);
  return Number.isFinite(v) ? v : fallback;
}

function floatParam(q: URLSearchParams, key: string, fallback: number): number {
  const raw = q.get(key);
  if (raw === null) return fallback;
  const v = parseFloat(raw);
  // Number.isFinite, NOT `|| fallback`: an explicit `?advWeight=0` is a
  // meaningful request (run the adversary, feed the HUD, but do not steer the
  // field) and must not be silently rewritten to the default.
  return Number.isFinite(v) ? v : fallback;
}

/** Physical force scale for a dimensionless steady-state component bound.
 *
 * If |F_t| <= 1 and |v_0| <= drive*maxVelocity, then the unclipped recurrence
 * |v_{t+1}| <= friction|v_t| + friction*forceMagnitude preserves
 * |v_t| <= drive*maxVelocity for every t. Therefore drive <= 1 never reaches
 * the component clip through the field force.
 */
export function forceMagnitudeForDrive(
  drive: number,
  maxVelocity: number,
  friction: number
): number {
  if (!(Number.isFinite(drive) && drive >= 0)) {
    throw new Error(`drive must be finite and >= 0, got ${drive}`);
  }
  if (!(Number.isFinite(maxVelocity) && maxVelocity > 0)) {
    throw new Error(`maxVelocity must be finite and > 0, got ${maxVelocity}`);
  }
  if (!(Number.isFinite(friction) && friction > 0 && friction < 1)) {
    throw new Error(`friction must be finite and in (0,1), got ${friction}`);
  }
  return (drive * maxVelocity * (1 - friction)) / friction;
}

/** Inverse of {@link forceMagnitudeForDrive}; used only to report legacy
 * configurations through the same LoopHandle API. */
export function driveForForceMagnitude(
  forceMagnitude: number,
  maxVelocity: number,
  friction: number
): number {
  if (!(Number.isFinite(forceMagnitude) && forceMagnitude >= 0)) {
    throw new Error(`forceMagnitude must be finite and >= 0, got ${forceMagnitude}`);
  }
  if (!(Number.isFinite(maxVelocity) && maxVelocity > 0)) {
    throw new Error(`maxVelocity must be finite and > 0, got ${maxVelocity}`);
  }
  if (!(Number.isFinite(friction) && friction > 0 && friction < 1)) {
    throw new Error(`friction must be finite and in (0,1), got ${friction}`);
  }
  return (friction * forceMagnitude) / ((1 - friction) * maxVelocity);
}

/**
 * The two Adam step sizes, and the reason they are ONE range table for BOTH
 * games: `discriminatorLearningRate` is already the `lr` uniform of the fused
 * relational trainer AND of the fused pixel critic (see the two `encodeStep`
 * calls in `tick`), and `generatorLearningRate` is the field trainer's for
 * every piece. `?gLR=` / `?dLR=` therefore already steer a Pixel piece; only
 * the dock's sliders were gated on the RELATIONAL game being on.
 */
export const GAME_LEARNING_RATE_RANGE = {
  generator: { min: 1e-6, max: 1e-1 },
  discriminator: { min: 1e-6, max: 1e-1 },
} as const;

/**
 * Reward range for the PIXEL critic — its own table, not `ADV_WEIGHT`'s 0..20.
 * The shipped pieces sit at 0.03-0.04 and the critic is the ONLY gradient on
 * them (`fieldLoss: ZERO_FIELD_LOSS`), so the useful span is small and near
 * zero; 0 is included because "critic trains, generator ignores it" is a real
 * and useful diagnostic state.
 */
export const PIXEL_CRITIC_WEIGHT_RANGE = { min: 0, max: 0.5 } as const;

/** Smallest useful training batch — matches the "train B" slider's minimum. */
export const TRAIN_BATCH_MIN = 16;

/**
 * Fallback ceiling for the train-B control when no fused trainer exists yet
 * (startup, or the tfjs-legacy path, which allocates its batch per step and has
 * no compiled cap). The fused path publishes a tighter, device-derived cap via
 * `LoopHandle.getMaxSampleRate()` — prefer that whenever it is available.
 */
export const TRAIN_BATCH_MAX = MAX_BATCH;

/** Requested train batch vs. the cap the trainer can actually run. */
export type TrainBatchSize =
  | { readonly tag: "ok"; readonly n: number }
  | { readonly tag: "clamped"; readonly n: number; readonly requested: number; readonly cap: number };

/**
 * Canonicalize a requested train batch size (κ — the ONE place a live control's
 * number becomes a legal `TrainStepOpts.n`).
 *
 * Over-cap used to reach `FusedTrainer.record()`, which throws — inside the rAF
 * tick, whose self-rearm is its last statement, so the whole animation stopped
 * dead. The slider went to 4096 while the trainer was built with batchCap 1024,
 * so the top 3/4 of the control froze the app. Now over-cap CLAMPS and reports
 * itself as data, and the caller warns.
 */
export function resolveTrainBatchSize(requested: number, cap: number): TrainBatchSize {
  if (!Number.isFinite(requested) || !Number.isFinite(cap)) {
    throw new Error(`train batch needs finite requested/cap, got ${requested}, ${cap}`);
  }
  const hi = Math.max(1, Math.floor(cap));
  const n = Math.max(1, Math.round(requested));
  return n > hi ? { tag: "clamped", n: hi, requested: n, cap: hi } : { tag: "ok", n };
}

/** Live train-B → fused adversary batch. The UI controls sampleRate; the
 * compiled predictor buffers cap one step, while particle coverage may reduce
 * it further inside AdversaryTrainer. */
export function adversaryBatchSize(sampleRate: number, batchCap: number = 1024): number {
  if (!Number.isFinite(sampleRate) || !Number.isFinite(batchCap)) {
    throw new Error(`adversary batch needs finite sampleRate/cap, got ${sampleRate}, ${batchCap}`);
  }
  return Math.max(1, Math.min(Math.round(sampleRate), Math.max(1, Math.floor(batchCap))));
}

/** Canonical URL + gallery ingestion for live physical/game controls. Exported
 * so a gallery switch can be tested without constructing WebGPU or React. */
export function resolveLiveGameControls(
  cfg: ArtPieceConfig,
  q: URLSearchParams,
  maxVelocity: number = cfg.maxVelocity
): {
  readonly driveEnabled: boolean;
  readonly drive: number;
  readonly forceMagnitude: number;
  readonly generatorLearningRate: number;
  readonly discriminatorLearningRate: number;
  /** `?pixW=` — pixel critic reward. 0 on a piece that declares no critic. */
  readonly pixelCriticWeight: number;
} {
  const driveEnabled = cfg.drive !== undefined;
  const drive = driveEnabled
    ? Math.max(0, Math.min(1, floatParam(q, "drive", cfg.drive!)))
    : driveForForceMagnitude(cfg.forceMagnitude, maxVelocity, cfg.friction);
  return {
    driveEnabled,
    drive,
    forceMagnitude: driveEnabled
      ? forceMagnitudeForDrive(drive, maxVelocity, cfg.friction)
      : cfg.forceMagnitude,
    generatorLearningRate: Math.max(
      GAME_LEARNING_RATE_RANGE.generator.min,
      Math.min(
        GAME_LEARNING_RATE_RANGE.generator.max,
        floatParam(q, "gLR", cfg.learningRate)
      )
    ),
    discriminatorLearningRate: Math.max(
      GAME_LEARNING_RATE_RANGE.discriminator.min,
      Math.min(
        GAME_LEARNING_RATE_RANGE.discriminator.max,
        floatParam(q, "dLR", cfg.discriminatorLearningRate ?? 3e-3)
      )
    ),
    // A piece with no declared critic has no reward to scale: 0 is the MEANING
    // here, not a default, so `?pixW=` on such a piece stays 0 rather than
    // arming a knob whose trainer does not exist.
    pixelCriticWeight:
      cfg.pixelDisc === undefined
        ? 0
        : Math.max(
            PIXEL_CRITIC_WEIGHT_RANGE.min,
            Math.min(
              PIXEL_CRITIC_WEIGHT_RANGE.max,
              floatParam(q, "pixW", cfg.pixelDisc.weight)
            )
          ),
  };
}

/** δ: `?advM=` → encoding. Unknown values are a hard error, not a shrug. */
function encodingFromM(m: number): TupleEncoding {
  if (m === 1) return { tag: "point" };
  if (m === 2) return { tag: "pair-rotation" };
  if (m === 3) return { tag: "tri" };
  if (m === 4) return { tag: "quad-labelled" };
  throw new Error(`?advM must be 1 (point), 2 (pair), 3 (tri) or 4 (labelled quad), got ${m}`);
}

/** Named, shared defaults for the objective controls and URL parser. */
export const ADVERSARY_OBJECTIVE_DEFAULTS = {
  tau: 0.05,
  scaleWeight: 0.5,
  energyWeight: 0.1,
  energyTarget: 0.35,
} as const;

/**
 * SHIPPED anti-collapse setting for adversarial GALLERY pieces.
 *
 * 0.05/0.05 is the pair measured in tools/collapse_probe.ts: R₁ 0.98 → 0.057
 * (pair preset) and 0.95 → 0.10 (point preset) with the payoff staying in the
 * healthy 0.48–0.63 band and structure (AC) growing monotonically. Polar alone
 * was measured to be escapable — the generator answered it with ±F₀
 * counter-streaming sheets (R₂ = 0.95), which reads on screen as the SAME
 * laminar streaks — so both weights ship together.
 *
 * τ is the canonical soft-angle τ, which every soft-angle gallery piece also
 * uses, and is the same value `?advPolar=` alone resolves to for a raw-vector
 * objective. `?advPolar=0&advNematic=0` turns the whole term off explicitly
 * (parseGamePressure then returns the NAMED `none`), mirroring ?advWeight=0.
 *
 * NOT stacked with an Okubo-Weiss/swirl term, deliberately: a 2000-step run
 * measured that it crushes ‖F‖ 0.035 → 0.005 and parks the payoff at √2 for a
 * thousand steps — ANY term that shrinks ‖F‖ feeds the north-pole exploit this
 * pressure exists to price.
 */
export const GALLERY_ANTI_COLLAPSE: GamePressure = {
  tag: "anti-collapse",
  polar: 0.05,
  nematic: 0.05,
  tau: ADVERSARY_OBJECTIVE_DEFAULTS.tau,
};

/**
 * THE ENCODING IS A DOCK KNOB, NOT A GAME INVARIANT — every piece that points
 * here declares `fieldArch` + `archEditable` instead of `createField`, and the
 * dock offers ARCH_DOCK_DUAL (Dual MLP / Fourier / SIREN / HashGrid).
 *
 * On a relational-adversary piece what is load-bearing is the OBSERVER (point
 * / pair / tri / quad), the objective (soft-angle vs raw-vector), K and ε. The
 * position ENCODING is orthogonal to all four: "Adversary · Pair WTA K=4" and
 * "Adversary · Pair · HashGrid · Curl" are the SAME game at the SAME α on two
 * different encodings, and shipped that way on purpose. The Pixel critics and
 * the two plain Max Chaos / Max Structure fields point here for the same
 * reason — a dual-head field, nothing about the game riding on the encoding.
 *
 * `createField` is the hatch for pieces whose arch genuinely bakes semantics,
 * and exactly two still need it: "Agree + Disagree RGB" (`fourierOctaves: 3`,
 * which `applyArchDockPreset` deliberately does NOT carry across a swap — see
 * the preserve list there) and "RGB Families · HashGrid" (`familyHashgrid` →
 * the `grid-plane` FamilyRoute, whose `classes`/`planes` the dual dock has no
 * presets for). Everything else was calling `createFieldFromArch` on a plain
 * literal — the hatch reached for because it was there, at the cost of the
 * dock hiding the model section entirely: index.tsx gates the WHOLE section on
 * `piece.fieldArch`, so those pieces showed no arch info at all, not even the
 * read-only summary.
 *
 * SAFE BY CONSTRUCTION, not by convention: ARCH_DOCK_DUAL is exactly the set
 * `validateAdversaryFusion` accepts — two-head field, no one-hot family — so
 * no dock choice can reach a refusal. `adversary_wire_test.ts` proves that by
 * CODEGEN over every (piece × preset) pair rather than trusting this comment:
 * §8d for the pixel critics, §8e for the relational adversary. Add a fifth
 * preset to the dual dock and those gates are what fail.
 */
export const DUAL_ARCH_DOCK = "dual" as const;

/** One-line pressure description for the trainer log. */
function describePressure(p: GamePressure): string {
  switch (p.tag) {
    case "none":
      return "none";
    case "anti-collapse":
      return `anti-collapse polar=${p.polar} nematic=${p.nematic} tau=${p.tau}`;
    default:
      return assertNeverPiece(p, "describePressure");
  }
}

/**
 * Compatibility mapping for adversary specs written before target/loss became
 * orthogonal axes. The legacy "adjusted" R+S tag meant angular prediction, so
 * preserve that visible behavior with the new exact smooth S² loss. Every
 * other old spec predicted raw force with raw vector distance.
 */
export function adversaryTargetOf(
  spec: Extract<AdversarySpec, { readonly tag: "on" }>
): AdversaryTarget {
  return spec.target ?? DEFAULT_ADVERSARY_TARGET;
}

export function gamePressureOf(
  spec: Extract<AdversarySpec, { readonly tag: "on" }>
): GamePressure {
  return spec.pressure ?? DEFAULT_GAME_PRESSURE;
}

/**
 * Canonical predictor dims. Both trainers take `hiddenUnits`/`featureDim` as
 * OPTIONAL with their own `?? 32` / `?? 16` fallbacks, which is two more places
 * for the fused path and the tfjs oracle to drift apart on a number neither
 * declares. Resolve once, here, and pass a concrete pair to both.
 */
export function predictorArchOf(
  spec: Extract<AdversarySpec, { readonly tag: "on" }>
): PredictorArch {
  return spec.predictor ?? PREDICTOR_ARCH_DEFAULT;
}

export function adversaryLossOf(
  spec: Extract<AdversarySpec, { readonly tag: "on" }>
): AdversaryLoss {
  if (spec.loss) return spec.loss;
  return spec.encoding.tag === "pair-rotation-scale-adjusted"
    ? { tag: "soft-angle", tau: ADVERSARY_OBJECTIVE_DEFAULTS.tau }
    : DEFAULT_ADVERSARY_LOSS;
}

function parseAdversaryTarget(
  q: URLSearchParams,
  fallback: AdversaryTarget
): AdversaryTarget {
  const tag = q.get("advTarget");
  if (tag === null) return fallback;
  if (tag === "force" || tag === "post-velocity") return { tag };
  throw new Error(`?advTarget must be force|post-velocity, got "${tag}"`);
}

function parseAdversaryLoss(q: URLSearchParams, fallback: AdversaryLoss): AdversaryLoss {
  const tag = q.get("advLoss") ?? fallback.tag;
  const tau = Math.max(1e-4, floatParam(q, "advTau", "tau" in fallback
    ? fallback.tau
    : ADVERSARY_OBJECTIVE_DEFAULTS.tau));
  const scaleWeight = Math.max(
    0,
    floatParam(
      q,
      "advScaleWeight",
      "scaleWeight" in fallback
        ? fallback.scaleWeight
        : ADVERSARY_OBJECTIVE_DEFAULTS.scaleWeight
    )
  );
  const energyWeight = Math.max(
    0,
    floatParam(
      q,
      "advEnergyWeight",
      "energyWeight" in fallback
        ? fallback.energyWeight
        : ADVERSARY_OBJECTIVE_DEFAULTS.energyWeight
    )
  );
  const energyTarget = Math.max(
    1e-4,
    floatParam(
      q,
      "advEnergyTarget",
      "energyTarget" in fallback
        ? fallback.energyTarget
        : ADVERSARY_OBJECTIVE_DEFAULTS.energyTarget
    )
  );
  switch (tag) {
    case "raw-vector":
      return { tag };
    case "soft-angle":
      return { tag, tau };
    case "angle-relative-scale":
    case "angle-scale-hold":
      return { tag, tau, scaleWeight, energyWeight, energyTarget };
    default:
      throw new Error(
        `?advLoss must be raw-vector|soft-angle|angle-relative-scale|angle-scale-hold, ` +
          `got "${tag}"`
      );
  }
}

/**
 * Merge a piece's declared adversary with the URL overrides.
 *
 * `?adv=off|single|wta` chooses the variant outright. When `?adv` is absent the
 * piece's own spec stands, but the numeric knobs (`?advK`, `?advM`, `?advEps`,
 * `?advWeight`) still override it — that is what makes a shipped piece
 * explorable without editing the gallery.
 */
export function resolveAdversary(
  piece: AdversarySpec | undefined,
  q: URLSearchParams
): AdversarySpec {
  const base: AdversarySpec = piece ?? { tag: "off" };
  const mode = q.get("adv");
  if (mode === "off") return { tag: "off" };
  if (mode === null && base.tag === "off") return base;

  // Knob defaults come from the piece when it already declares an adversary, so
  // `?adv=wta` on a pair piece keeps the pair encoding unless `?advM` says
  // otherwise.
  const from = base.tag === "on" ? base : null;
  const k = intParam(q, "advK", from?.kind.tag === "wta" ? from.kind.k : ADV_DEFAULTS.k);
  const eps = floatParam(
    q,
    "advEps",
    from?.kind.tag === "wta" ? from.kind.relaxEps : ADV_DEFAULTS.eps
  );
  // encodingDims, not a tag ternary: the old `tag === "pair" ? 2 : 1` silently
  // mapped every FUTURE encoding to m=1 — it did exactly that to tri when the
  // tri encoding landed. Deriving m from the same dispatcher the adversary uses
  // means this line cannot drift from the encoding set.
  const m = intParam(q, "advM", from ? encodingDims(from.encoding).m : ADV_DEFAULTS.m);
  const weight = Math.max(
    ADVERSARY_WEIGHT_RANGE.min,
    Math.min(
      ADVERSARY_WEIGHT_RANGE.max,
      floatParam(q, "advWeight", from ? from.weight : ADV_DEFAULTS.weight)
    )
  );
  // Preserve the piece's full observer semantics. Tuple arity alone cannot
  // distinguish pair rotation, raw rotation+scale, and adjusted
  // rotation+scale. Only an explicit legacy `?advM=` request should collapse
  // that choice to an arity default.
  const encoding = q.has("advM") ? encodingFromM(m) : from?.encoding ?? encodingFromM(m);
  const legacyActive =
    from ??
    ({
      tag: "on",
      kind: { tag: "wta", k, relaxEps: eps },
      encoding,
      weight,
    } as const);
  const target = parseAdversaryTarget(q, adversaryTargetOf(legacyActive));
  const loss = parseAdversaryLoss(q, adversaryLossOf(legacyActive));
  // One validation dispatcher is shared with both reference and fused shape
  // construction. Invalid combinations throw here instead of silently
  // changing observer, target, or loss semantics.
  objectiveDims(encoding, target, loss);

  const game = from?.game;
  const gamePart = game ? { game } : {};
  // `?advHidden` / `?advFeature` — the PREDICTOR's width, the one model in
  // this project that had no knob. Resolved against the piece's own predictor
  // so a URL that names neither is byte-identical to the piece.
  const predictorBase = from ? predictorArchOf(from) : PREDICTOR_ARCH_DEFAULT;
  const predictor: PredictorArch = {
    hiddenUnits: intParam(q, "advHidden", predictorBase.hiddenUnits),
    featureDim: intParam(q, "advFeature", predictorBase.featureDim),
  };
  if (
    !Number.isInteger(predictor.hiddenUnits) ||
    predictor.hiddenUnits < 1 ||
    predictor.hiddenUnits > 256 ||
    !Number.isInteger(predictor.featureDim) ||
    predictor.featureDim < 1 ||
    predictor.featureDim > 256
  ) {
    throw new Error(
      `?advHidden/?advFeature must be integers in [1, 256], got ` +
        `${predictor.hiddenUnits}/${predictor.featureDim}`
    );
  }
  const pressure = parseGamePressure(q, from ? gamePressureOf(from) : DEFAULT_GAME_PRESSURE, loss);
  if (mode === "single") {
    return {
      tag: "on",
      kind: { tag: "single" },
      encoding,
      target,
      loss,
      weight,
      pressure,
      predictor,
      ...gamePart,
    };
  }
  if (mode === "wta") {
    return {
      tag: "on",
      kind: { tag: "wta", k, relaxEps: eps },
      encoding,
      target,
      loss,
      weight,
      pressure,
      predictor,
      ...gamePart,
    };
  }
  if (mode !== null) {
    throw new Error(`?adv must be off|single|wta, got "${mode}"`);
  }
  // No `?adv=`: keep the piece's variant, apply the knob overrides to it.
  //
  // `from` is the piece's own spec, already narrowed to the active variant. It
  // cannot be null here — `mode === null && base.tag === "off"` returned above,
  // and every other `mode` either returned or threw — but that invariant spans
  // statements, so state it as a typed error rather than reading `base.kind`
  // (which TS correctly rejects: an off spec has no `kind`) or defaulting to
  // wta. A silent wta default here would invent an adversary for a piece that
  // never declared one, which is precisely the class of bug this file's gates
  // exist to prevent.
  if (from === null) {
    throw new Error(
      "resolveAdversary: reached the piece-variant path with no piece spec — " +
        "the `?adv` ladder above must return for every mode when the piece " +
        "declares no adversary"
    );
  }
  const kind: AdversaryKind =
    from.kind.tag === "single" ? { tag: "single" } : { tag: "wta", k, relaxEps: eps };
  return {
    tag: "on",
    kind,
    encoding,
    target,
    loss,
    weight,
    pressure,
    predictor,
    ...gamePart,
  };
}

/**
 * `?advPolar=λ` / `?advNematic=λ` — the anti-collapse pressure knobs.
 *
 * Both zero is the NAMED `none` variant, not a weight-0 `anti-collapse`: the
 * fused routing decision below reads the tag, and "declared but inert" would
 * push a piece onto the slow tfjs trainer for nothing. `?advPolarTau` defaults
 * to the objective's own soft-angle τ so the pressure and the payoff agree on
 * what "a direction" means; raw-vector objectives have no τ and fall back to
 * the canonical one.
 */
function parseGamePressure(
  q: URLSearchParams,
  fallback: GamePressure,
  loss: AdversaryLoss
): GamePressure {
  const from = fallback.tag === "anti-collapse" ? fallback : null;
  const lossTau =
    loss.tag === "raw-vector" ? ADVERSARY_OBJECTIVE_DEFAULTS.tau : loss.tau;
  const polar = Math.max(0, floatParam(q, "advPolar", from ? from.polar : 0));
  const nematic = Math.max(0, floatParam(q, "advNematic", from ? from.nematic : 0));
  if (polar === 0 && nematic === 0) return { tag: "none" };
  const tau = floatParam(q, "advPolarTau", from ? from.tau : lossTau);
  if (!(Number.isFinite(tau) && tau > 0)) {
    throw new Error(`?advPolarTau must be finite and > 0, got ${tau}`);
  }
  return { tag: "anti-collapse", polar, nematic, tau };
}

/**
 * `?color=surprise|surprise-per-unit|velocity` overrides the piece; `?cmap=`
 * picks the ramp. `?color=surprise` remains the raw-payoff compatibility alias;
 * `?surNorm=1` selects the per-unit diagnostic for old bookmarked experiments.
 * A piece that declares `renderer: "surprise"` gets the surprise mode by
 * default. Falling back to velocity when the piece has no adversary is NOT a
 * silent default — an empty surprise channel would paint a flat field and read
 * as "the adversary is collapsed", which is a lie about a piece that never had
 * one; see the SPAN_FLOOR note in src/draw/robust_norm.ts.
 */
export function resolveColorMode(
  cfg: ArtPieceConfig,
  adv: AdversarySpec,
  q: URLSearchParams
): ColorMode {
  const cmapParam = q.get("cmap");
  const colormap: ColormapName =
    cmapParam === "viridis" || cmapParam === "coolwarm" || cmapParam === "inferno"
      ? cmapParam
      : cfg.colormap ?? "inferno";
  const want = q.get("color");
  if (want === "velocity") return { tag: "velocity" };
  const perUnit =
    want === "surprise-per-unit" ||
    want === "surprise-unit" ||
    q.get("surNorm") === "1";
  const raw =
    want === "surprise" ||
    want === "surprise-raw" ||
    (want === null &&
      (cfg.colorMode?.tag === "surprise-raw" || cfg.renderer === "surprise") &&
      !perUnit);
  const on = raw || perUnit;
  if (!on) return { tag: "velocity" };
  if (adv.tag === "off") {
    console.warn(
      "[adversary] surprise colour ignored: this piece has no adversary, so " +
        "there is no per-particle residual to colour by."
    );
    return { tag: "velocity" };
  }
  if (cfg.mode === "agree-disagree") {
    console.warn(
      "[adversary] surprise colour ignored for Agree+Disagree: its required " +
        "RGB output is A=disagree, B=agree, C=loss-free blend."
    );
    return { tag: "velocity" };
  }
  return { tag: perUnit ? "surprise-per-unit" : "surprise-raw", colormap };
}

/**
 * What the pixel critic actually IS this run — a named state per outcome.
 *
 * Every one of these used to be an unnamed skip in the trainer-gate `if`, and
 * a skipped critic is invisible: the gallery entry still says "Pixel · …", the
 * HUD still says the piece is training, and the four shipped Pixel pieces carry
 * `fieldLoss: ZERO_FIELD_LOSS` — so "critic dropped" means NOTHING drives the
 * field, not "the piece trains a bit less". `dropped` carries its reason so the
 * loop can say which gate did it.
 */
export type PixelCriticPlan =
  | { readonly tag: "absent" }
  /** Carries the approved spec so the construction site re-checks nothing. */
  | { readonly tag: "fused"; readonly spec: PixelCriticSpec }
  | { readonly tag: "dropped"; readonly reason: string };

/**
 * κ for the pixel critic: the ONE place that decides whether a declared critic
 * runs, why it doesn't, and which combinations are not configurations at all.
 *
 * The Agree+Disagree case is a THROW, not a `dropped`, because it is a piece
 * authoring error rather than a runtime override: the game and the critic both
 * want an external-gradient slot in the fused field trainer's pass B, which
 * carries at most two (`train_wgsl.ts` `extGradCount ∈ [0,2]`, one read-only
 * binding each) and the game's two lanes take both. The old code expressed that
 * as `if (advTrainerB) … else if (pixelDiscTrainer)`, so such a piece built the
 * critic, logged `[pixel-disc] FUSED`, and dispatched all ten of its passes
 * every frame while contributing 0% of the field update — the same failure the
 * anti-collapse pressure gate below is written to prevent.
 */
export function resolvePixelCritic(gates: {
  /** The piece's `pixelDisc` declaration; absent ≡ this piece has no critic. */
  readonly declared: PixelCriticSpec | undefined;
  readonly hasField: boolean;
  /** The piece declares `fieldLoss`, i.e. the fused field trainer can run. */
  readonly fieldLossDeclared: boolean;
  readonly wantTfjsTrainer: boolean;
  /** An adversary is on but routed to the tfjs autograd trainer. */
  readonly adversaryOnTfjs: boolean;
  readonly agreeDisagreeGame: boolean;
  readonly layout: FieldLayout;
}): PixelCriticPlan {
  const spec = gates.declared;
  if (spec === undefined) return { tag: "absent" };
  if (gates.agreeDisagreeGame) {
    throw new Error(
      "[pixel-disc] this piece declares BOTH the Agree+Disagree game and a " +
        "pixel critic. The fused field trainer's pass B sums at most TWO " +
        "external gradients and the game's two lanes take both, so the critic " +
        "would train, log and dispatch every frame while contributing 0% of " +
        "the field update. Drop one of the two on this piece, or extend " +
        "trainPassBShader to a third extGrad binding first."
    );
  }
  if (!gates.hasField) {
    return {
      tag: "dropped",
      reason:
        "this piece has no neural field (legacy MLP path); the critic's " +
        "generator reward has nowhere to land",
    };
  }
  if (gates.wantTfjsTrainer) {
    return {
      tag: "dropped",
      reason:
        "?train=tfjs selected the tfjs autograd trainer and there is no tfjs " +
        "implementation of the density critic — on a ZERO_FIELD_LOSS Pixel " +
        "piece that leaves NOTHING training the field, and optimizer.minimize " +
        "then throws \"Cannot find a connection between any variable and the " +
        "result of the loss function\" (measured on this build; the throw is " +
        "the constant loss, not a tape bug)",
    };
  }
  if (!gates.fieldLossDeclared) {
    return {
      tag: "dropped",
      reason:
        "the piece declares no fieldLoss, so the fused field trainer (the " +
        "only consumer of the critic's extGrads) never runs",
    };
  }
  if (gates.adversaryOnTfjs) {
    return {
      tag: "dropped",
      reason:
        "this piece's adversary game routed to the tfjs trainer, which takes " +
        "the whole field step with it",
    };
  }
  const fusion = classifyPixelDiscFusion(gates.layout);
  if (fusion.tag === "unsupported") {
    return { tag: "dropped", reason: fusion.reason };
  }
  return { tag: "fused", spec };
}

// ---------------------------------------------------------------------------
// Backend — webgpu ONLY, selected in startLoop's async init (near the bottom).
//
// The old cpu-first `initBackend` (with its `?backend=` fallback ladder) was
// DELETED 2026-07-27 rather than kept: it had zero callers since the zero-copy
// renderer landed, and its premise no longer holds — the renderer binds tfjs's
// own GPUBuffers on tfjs's own GPUDevice, so a cpu/webgl tfjs backend has
// nothing the render path can draw from. A `?backend=cpu` override therefore
// cannot be "supported" here even in principle; the init IIFE rejects the
// param LOUDLY instead of silently ignoring it. (The dispatch-bound cost of
// tiny-op training on webgpu that initBackend's comment worried about is real
// — that dial is now `?handoff=N`, the tfjs small-tensor CPU forwarding
// threshold, handled in the same IIFE.)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Loss functions  (all differentiable through the physics chain)
// ---------------------------------------------------------------------------
const SPIRAL_TURNS = 3;
const SPIRAL_MAX_THETA = SPIRAL_TURNS * 2 * Math.PI;

function spiralLoss(pos: tf.Tensor2D, w: number, h: number): tf.Scalar {
  return tf.tidy(() => {
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(w, h) * 0.38;
    const b = maxR / SPIRAL_MAX_THETA;

    const dx = pos.slice([0, 0], [-1, 1]).sub(cx);
    const dy = pos.slice([0, 1], [-1, 1]).sub(cy);
    const r = dx.square().add(dy.square()).add(1e-4).sqrt();
    const phi = tf.atan2(dy, dx);

    let best = tf.fill(r.shape, 1e8) as tf.Tensor;
    for (let k = 0; k <= SPIRAL_TURNS + 1; k++) {
      const theta = phi.add(2 * Math.PI * k);
      const rSpiral = theta.relu().mul(b);
      best = tf.minimum(best, r.sub(rSpiral).square());
    }
    return best.mean().asScalar();
  });
}

function centerLoss(pos: tf.Tensor2D, w: number, h: number): tf.Scalar {
  return tf.tidy(() => {
    const center = tf.tensor2d([[w / 2, h / 2]]);
    return pos.sub(center).square().sum(1).mean().asScalar();
  });
}

function spiralPlusCenterLoss(
  centerWeight: number
): (pos: tf.Tensor2D, w: number, h: number) => tf.Scalar {
  return (pos, w, h) =>
    tf.tidy(() => {
      const sL = spiralLoss(pos, w, h);
      const cL = centerLoss(pos, w, h);
      return sL.add(cL.mul(centerWeight)).asScalar();
    });
}

/** Default curve samples for the cover half of {@link spiralCoverLoss}. */
const SPIRAL_COVER_SAMPLES = 256;
/** Skip the innermost fraction of arc length so cover does not overweight the hub. */
const SPIRAL_COVER_ARC_SKIP = 0.08;

/**
 * Arc length of Archimedean `r = b·θ` from 0 to `theta`:
 *   L(θ) = (b/2)·(θ√(1+θ²) + asinh(θ))
 */
function archimedeanArcLength(b: number, theta: number): number {
  const s = Math.sqrt(1 + theta * theta);
  return 0.5 * b * (theta * s + Math.asinh(theta));
}

/**
 * Invert arc length for `r = b·θ` on `[thetaMin, thetaMax]` (binary search).
 */
function thetaFromArcLength(
  b: number,
  targetArc: number,
  thetaMin: number,
  thetaMax: number
): number {
  const base = archimedeanArcLength(b, thetaMin);
  let lo = thetaMin;
  let hi = thetaMax;
  for (let i = 0; i < 48; i++) {
    const mid = 0.5 * (lo + hi);
    if (archimedeanArcLength(b, mid) - base < targetArc) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * Uniform-along-filament samples of the Archimedean spiral (same `b, Θ, maxR`
 * as {@link spiralLoss}). Equal-θ sampling packs near the origin and makes
 * cover collapse into a hub/cross; arc-length sampling keeps arms equally
 * costly when empty.
 */
function archimedeanSpiralSamples(
  w: number,
  h: number,
  samples: number
): Float32Array {
  const cx = w / 2;
  const cy = h / 2;
  const maxR = Math.min(w, h) * 0.38;
  const b = maxR / SPIRAL_MAX_THETA;
  const thetaMax = SPIRAL_MAX_THETA;
  const totalArc = archimedeanArcLength(b, thetaMax);
  const skipArc = Math.min(
    Math.max(SPIRAL_COVER_ARC_SKIP, 0),
    0.45
  ) * totalArc;
  const usable = Math.max(totalArc - skipArc, 1e-6);
  const thetaMin = thetaFromArcLength(b, skipArc, 0, thetaMax);
  const out = new Float32Array(samples * 2);
  for (let i = 0; i < samples; i++) {
    const target = usable * ((i + 0.5) / samples);
    const theta = thetaFromArcLength(b, target, thetaMin, thetaMax);
    const r = b * theta;
    out[i * 2] = cx + r * Math.cos(theta);
    out[i * 2 + 1] = cy + r * Math.sin(theta);
  }
  return out;
}

/**
 * Curve → particles: mean over spiral samples of squared distance to the
 * nearest particle, scaled by `maxR²` so weights are O(1). Empty filament
 * segments stay costly even when all mass sits on-curve at one locus.
 */
function spiralCoverUpLoss(
  pos: tf.Tensor2D,
  w: number,
  h: number,
  samples: number
): tf.Scalar {
  return tf.tidy(() => {
    const maxR = Math.min(w, h) * 0.38;
    const scale = Math.max(maxR * maxR, 1);
    const s = tf.tensor2d(archimedeanSpiralSamples(w, h, samples), [
      samples,
      2,
    ]);
    // [N,1,2] − [1,M,2] → [N,M,2] → min over particles → mean over samples
    const d2 = pos.expandDims(1).sub(s.expandDims(0)).square().sum(2);
    return d2.min(0).mean().div(scale).asScalar();
  });
}

/**
 * Spiral Cover — curve→particles only (not bidirectional).
 *
 * Each sample on the spiral wants a nearby particle. Attractors lie on the
 * filament, so mass both covers the arms and stays near the curve. The old
 * particle→spiral residual is omitted: it is minimized at the hub and fights
 * arm fill when combined with a center-heavy cover term.
 *
 *   L = β · mean_m ‖s_m − p_nearest‖² / maxR²
 */
function spiralCoverLoss(opts?: {
  /** Weight on curve→particles cover term (default 1). */
  beta?: number;
  samples?: number;
}): ArtPieceConfig["computeLoss"] {
  const beta = opts?.beta ?? 1;
  const samples = opts?.samples ?? SPIRAL_COVER_SAMPLES;
  return (pos, w, h) =>
    tf.tidy(() => spiralCoverUpLoss(pos, w, h, samples).mul(beta).asScalar());
}

/**
 * Composite loss for the Helmholtz field piece: mostly chaos + isotropy, held
 * together by a faint spiral so the mixing has a ghost of structure.
 *
 *   loss = W_CHAOS·chaos + W_ISO·isotropy + W_DIV·divergence + W_SPIRAL·spiral
 *
 * - chaos      maximises local sensitivity (returns −log separation).
 * - isotropy   keeps force energy directionally balanced (reads the SAME
 *              per-step force tensor via {@link LossContext.force}).
 * - divergence lightly pins the flow toward area-preserving.
 * - spiral     a tiny structural anchor (pixel-scale MSE → small weight).
 *
 * Requires `ctx.field` and `ctx.force` (the field piece always supplies them).
 * Weights are artistic knobs — tune freely.
 */
function helmholtzChaosLoss(
  spec: FieldLossSpec = {
    W_CHAOS: 1,
    W_ISO: 1,
    W_DIV: 0.5,
    W_SPIRAL: 0.00002,
    HH: 1e-2,
    SPIRAL_TURNS: 3,
  }
): ArtPieceConfig["computeLoss"] {
  const { W_CHAOS, W_ISO, W_DIV, W_SPIRAL, HH } = spec;
  const W_STRUCT = spec.W_STRUCT ?? 0;
  if (
    W_CHAOS === 0 &&
    W_ISO === 0 &&
    W_DIV === 0 &&
    W_SPIRAL === 0 &&
    W_STRUCT === 0
  ) {
    return () => tf.scalar(0);
  }
  return (pos, w, h, ctx) =>
    tf.tidy(() => {
      const field = ctx!.field!;
      const force = ctx!.force!;
      const posNorm = pos.div(tf.tensor2d([[w, h]])) as tf.Tensor2D;

      // PERF: evaluate the field just 3× — a shared centre f0 plus one +x and one
      // +y neighbour — and reuse them for BOTH the chaos (Lyapunov) and the
      // divergence terms. (Was 7 evals: chaos 2 + divergence 4 + physics 1, each
      // = 2 MLP heads. On the dispatch-bound webgpu backend those tiny ops are
      // the dominant cost of this piece; sharing the centre roughly halves it.)
      // SINGLE (sharded) forward: batch the 3 sample sets — centre, +x, +y —
      // into one [3N,2] tensor so the field runs ONCE (1 set of GPU dispatches
      // instead of 3), then slice. Same math, ~1/3 the dispatch overhead, still
      // one backward pass over the whole graph (first-order autograd).
      const N = posNorm.shape[0];
      const allPos = tf.concat(
        [
          posNorm,
          posNorm.add(tf.tensor2d([[HH, 0]])),
          posNorm.add(tf.tensor2d([[0, HH]])),
        ],
        0
      ) as tf.Tensor2D;
      const allF = field.forces(allPos);
      const f0 = allF.slice([0, 0], [N, -1]);
      const fx = allF.slice([N, 0], [N, -1]);
      const fy = allF.slice([2 * N, 0], [N, -1]);

      // chaos: local sensitivity — how much F changes for a small +x/+y nudge.
      const sepx = fx.sub(f0).square().sum(1);
      const sepy = fy.sub(f0).square().sum(1);
      const sep = sepx.add(sepy).add(1e-12).sqrt().div(HH * 1.4142 + 1e-9);
      const chaos = sep.add(1e-6).log().mean().neg();

      // forward-difference divergence sharing the centre: ∂Fx/∂x + ∂Fy/∂y.
      const dFxdx = fx.slice([0, 0], [-1, 1]).sub(f0.slice([0, 0], [-1, 1])).div(HH);
      const dFydy = fy.slice([0, 1], [-1, 1]).sub(f0.slice([0, 1], [-1, 1])).div(HH);
      const div = dFxdx.add(dFydy).square().mean();

      const iso = isotropyLoss(force);
      const spiral = spiralLoss(pos, w, h);
      // Same measure the fused W_STRUCT uses: the PHYSICAL force (post
      // forceMagnitude), exactly as isotropyLoss above — Fs in the shader.
      const struct = constantModeFraction(force);

      return chaos
        .mul(W_CHAOS)
        .add(iso.mul(W_ISO))
        .add(div.mul(W_DIV))
        .add(struct.mul(W_STRUCT))
        .add(spiral.mul(W_SPIRAL))
        .asScalar();
    });
}

// ---------------------------------------------------------------------------
// Adversary runtime — the live game, plus the two places it touches the loop.
//
// THE SEPARATION THAT MATTERS. The adversary and field use disjoint optimizers
// and disjoint parameter storage:
//
//   fused           AdversaryTrainer owns predictor weights/Adam and emits an
//                   external field-gradient buffer; FusedTrainer alone owns
//                   field Adam and consumes that buffer.
//   tfjs reference  Adversary#trainStep owns predictor variables, while
//                   optimizer.minimize receives only field.trainableWeights.
//
// Neither direction can co-optimize the other. The freeze is structural on
// both implementations, not accidental tape pruning.
// ---------------------------------------------------------------------------

type AdversaryRuntimeOff = { readonly tag: "off" };

type ActiveAdversaryRuntimeBase = {
  readonly tag: "on";
  /** Canonical spec data needed by both implementations. Keeping it here
   * avoids constructing a tfjs predictor merely to read cfg.kind/k. */
  readonly kind: AdversaryKind;
  readonly k: number;
  weight: number;
  /** Anti-collapse pressure composed alongside the reward. See {@link GamePressure}. */
  readonly pressure: GamePressure;
  /**
   * EMA of the PER-BATCH win counts, length k.
   *
   * NOT `Adversary#winStats()`. That counter is cumulative since
   * construction and never decays, so it reports the ALL-TIME share: a head
   * that was starved for the first thousand frames and has since recovered
   * still reads as dead, and — worse — a head that has JUST died stays
   * looking healthy for as long as it took to earn its history. For a live
   * collapse indicator the statistic has to be recent, so this folds each
   * step's `TrainReport.winCounts` with λ = 0.02 (≈ 50-step window).
   */
  readonly winEma: number[];
  /** Latest head-spread probe (~1 Hz, see {@link SPREAD_PROBE_MS}) — the
   * collapse DISAMBIGUATOR that {@link classifyHeads} folds against the win
   * histogram. Starts `unprobed`, which is a named state, not a null. */
  spread: SpreadProbe;
  /** `performance.now()` of the last spread probe (0 = never). */
  lastProbeMs: number;
};

export type TfjsAdversaryRuntime = ActiveAdversaryRuntimeBase & {
  readonly implementation: "tfjs";
  /** Exists only on the explicit tfjs oracle/fallback path. */
  readonly adv: Adversary;
};

export type FusedAdversaryRuntime = ActiveAdversaryRuntimeBase & {
  readonly implementation: "fused";
};

/** The live adversary, or the absence of one. Same sum shape as the spec. */
export type AdversaryRuntime =
  | AdversaryRuntimeOff
  | TfjsAdversaryRuntime
  | FusedAdversaryRuntime;

/**
 * The spread probe's lifecycle. Distinct from {@link HeadSpread} because "no
 * probe has run yet" (the first beat of a run) is a different fact from any
 * probe result, and conflating them would let the HUD claim a verdict it has
 * not measured.
 */
export type SpreadProbe = { readonly tag: "unprobed" } | HeadSpread;

/** Wall-clock cadence of the head-spread probe. One forward per head plus ONE
 *  dataSync — cheap, but a pipeline sync nonetheless, so ~1 Hz, never per frame. */
const SPREAD_PROBE_MS = 1000;

/**
 * The pileup threshold: `meanPair` below this fraction of the reward scale
 * (the adversary's EMA of RMS‖y‖ — the typical step length, i.e. the natural
 * yardstick for distances between head predictions) reads as OPTIMIZER PILEUP;
 * above it, skewed wins with separated heads remain UNRESOLVED: without
 * per-head support calibration, large spread could be valid modes or a head
 * that diverged off support.
 *
 * Anchored to measurement, not taste (tools/adversary_test.ts §10, k=4
 * fixture): benign parking measured meanPair ≈ 1.06·RMS‖y‖, engineered pileup
 * ≈ 0.08·RMS‖y‖ — 12× apart, so 0.2 sits between the two states rather than
 * on either. tools/adversary_wire_test.ts §7 trips if this constant drifts
 * out of that measured gap.
 */
const PILEUP_SPREAD_FRACTION = 0.2;

/**
 * The conservative collapse verdict. Win histogram + spread can identify
 * pileup, but cannot prove that separated low-win heads are valid modes.
 * apart — tools/adversary_test.ts §10 measures both states firing the same
 * win-share tripwire with identical-looking histograms:
 *
 *   `separated-unresolved` — wins skew and heads remain far apart. This needs
 *                    assigned-distortion/support calibration before it may be
 *                    called benign surplus.
 *   `pileup`       — wins skew AND the heads have converged onto the same
 *                    function: effective K ≈ 1. The real pathology the red
 *                    light exists for.
 *   `unresolved`   — wins skew but no spread probe (or no reward scale) exists
 *                    yet; only reachable in the first beats of a run.
 */
export type HeadHealth =
  | { readonly tag: "ok" }
  | { readonly tag: "separated-unresolved"; readonly meanPair: number; readonly scale: number }
  | { readonly tag: "pileup"; readonly meanPair: number; readonly scale: number }
  | { readonly tag: "unresolved" };

/**
 * δ: (win skew, spread probe, reward scale) → verdict. Pure and exported so
 * the threshold is gateable offline (tools/adversary_wire_test.ts §7).
 */
export function classifyHeads(
  winsSkewed: boolean,
  spread: SpreadProbe,
  scale: RewardScale
): HeadHealth {
  if (!winsSkewed) return { tag: "ok" };
  switch (spread.tag) {
    case "unprobed":
      return { tag: "unresolved" };
    case "single-head":
      // k = 1: there is no mixture whose effective K could degrade, and the
      // win floor (0.05/k) cannot trip on a lone head that wins everything.
      // Classified deliberately rather than defaulted.
      return { tag: "ok" };
    case "spread": {
      if (scale.tag === "unseeded") return { tag: "unresolved" };
      return spread.meanPair < PILEUP_SPREAD_FRACTION * scale.rms
        ? { tag: "pileup", meanPair: spread.meanPair, scale: scale.rms }
        : { tag: "separated-unresolved", meanPair: spread.meanPair, scale: scale.rms };
    }
    default:
      return assertNeverPiece(spread, "classifyHeads");
  }
}

/**
 * Combine independent game-lane verdicts without averaging away a failure.
 * A measured pileup is strongest, then an uninstrumented branch, then
 * separated-but-support-unresolved skew, then healthy.
 */
export function combineHeadHealth(a: HeadHealth, b: HeadHealth): HeadHealth {
  const rank = (h: HeadHealth): number => {
    switch (h.tag) {
      case "ok":
        return 0;
      case "separated-unresolved":
        return 1;
      case "unresolved":
        return 2;
      case "pileup":
        return 3;
      default:
        return assertNeverPiece(h, "combineHeadHealth.rank");
    }
  };
  return rank(a) >= rank(b) ? a : b;
}

/** δ: HeadHealth → HUD suffix for the `heads` line. One string per verdict —
 *  red only for the real pathology; the benign state is labelled as such. */
function headHealthHud(h: HeadHealth): string {
  switch (h.tag) {
    case "ok":
      return "";
    case "unresolved":
      return "  skew (unprobed)";
    case "separated-unresolved":
      return `  skew + separated (support unprobed, ${(h.meanPair / h.scale).toFixed(2)}·rms)`;
    case "pileup":
      return `  HEAD COLLAPSE (spread ${(h.meanPair / h.scale).toFixed(2)}·rms)`;
    default:
      return assertNeverPiece(h, "headHealthHud");
  }
}

/**
 * REFERENCE VALUE for the payoff chart — free, no new plumbing.
 *
 * Under a soft-angle objective the residual is a chord on S²: ψτ(0) is the
 * north pole and sits exactly √2 ≈ 1.4142 from every equatorial embedding. The
 * payoff parks there whenever ONE side of the comparison is polar and the other
 * equatorial, which happens for two opposite reasons — so the line is only a
 * diagnosis when read together with R₁ (which is why the two charts are
 * adjacent):
 *
 *   payoff ≈ √2 AND R₁ → 1  the encoded TARGET went to zero: the field is flat
 *                           and G is collecting the north-pole bonus. This is
 *                           the collapse (measured 1.362 at the prototype's
 *                           step-800 event).
 *   payoff ≈ √2 AND R₁ → 0  the field is so direction-isotropic that D's best
 *                           response IS the pole. Measured live on the default
 *                           piece: a ~25 s transient right after the pressure
 *                           takes hold, decaying to 0.56–0.71 as D relearns.
 *                           That is the game working, not failing.
 *
 * `none` for raw-vector (unbounded residual, no such value) and for the
 * scale-augmented angle objectives, whose payoff carries an ADDITIVE local
 * scale term — √2 is not their reference value and drawing it would be a lie.
 */
export type PayoffReference =
  | { readonly tag: "none" }
  | { readonly tag: "north-pole"; readonly chord: number };

/** δ: objective → its collapse payoff value. */
export function payoffReferenceOf(loss: AdversaryLoss): PayoffReference {
  switch (loss.tag) {
    case "soft-angle":
      return { tag: "north-pole", chord: Math.SQRT2 };
    case "raw-vector":
    case "angle-relative-scale":
    case "angle-scale-hold":
      return { tag: "none" };
    default:
      return assertNeverPiece(loss, "payoffReferenceOf");
  }
}

/** What the HUD and the React panel read. */
export type AdversaryTelemetry =
  | { readonly tag: "off" }
  | {
      readonly tag: "on";
      readonly variant: string;
      readonly k: number;
      /** Mean relaxed-WTA payoff on the most recent training batch. */
      readonly surprise: number;
      /** Predictor (discriminator) loss on the most recent batch. NOTE this is
       *  the same zero-sum scalar as `surprise` — see AdvStats.payoffUngated. */
      readonly predLoss: number;
      /**
       * Direction order of the field over the most recent training batch, from
       * the anti-collapse pressure's own moments. `unmeasured` when no pressure
       * is declared: neither trainer computes the moments then, and a 0 would
       * read as "perfectly isotropic" on a piece that may be fully laminar.
       */
      readonly directionOrder: DirectionOrder;
      /**
       * Payoff split by particle FAMILY. `unmeasured` on every piece except a
       * family-conditioned one played on a point observer — the tfjs trainer
       * has no family input at all, and a relational observer's tuples can mix
       * families. See AdvStats.perFamily.
       */
      readonly perFamily: FamilyPayoff;
      /**
       * The payoff value that means "the encoded target has gone to zero", for
       * objectives whose residual is bounded by a fixed chord. See
       * {@link PayoffReference}.
       */
      readonly payoffReference: PayoffReference;
      /** Cumulative share of wins per head; length k. Uniform = 1/k each. */
      readonly winFractions: readonly number[];
      /** Conservative head-health verdict. Win skew plus small measured spread
       *  can identify pileup; separated skew never claims a mode count. */
      readonly health: HeadHealth;
      readonly weight: number;
      /**
       * Present for the Agree+Disagree game. The aggregate fields above keep
       * the existing UI ABI stable; these branch values prevent the FPS HUD
       * from pretending the general-sum game is one scalar adversary.
       */
      readonly branches?: {
        readonly disagree: {
          readonly surprise: number;
          readonly predLoss: number;
          readonly winFractions: readonly number[];
          readonly health: HeadHealth;
        };
        readonly agree: {
          readonly surprise: number;
          readonly predLoss: number;
          readonly winFractions: readonly number[];
          readonly health: HeadHealth;
        };
      };
    };

/**
 * δ: spec → runtime. `batchTuples` is the loop's live sample rate so the
 * adversary and the field see batches of the same size.
 */
export function createAdversary(
  spec: AdversarySpec,
  batchTuples: number,
  observerGeometry: ObserverGeometry,
  learningRate?: number
): AdversaryRuntimeOff | TfjsAdversaryRuntime;
export function createAdversary(
  spec: AdversarySpec,
  batchTuples: number,
  observerGeometry: ObserverGeometry,
  learningRate: number,
  implementation: "fused"
): AdversaryRuntimeOff | FusedAdversaryRuntime;
export function createAdversary(
  spec: AdversarySpec,
  batchTuples: number,
  observerGeometry: ObserverGeometry,
  learningRate: number = 3e-3,
  implementation: "tfjs" | "fused" = "tfjs"
): AdversaryRuntime {
  switch (spec.tag) {
    case "off":
      return { tag: "off" };
    case "on": {
      const base: ActiveAdversaryRuntimeBase = {
        tag: "on",
        kind: spec.kind,
        k: headCount(spec.kind),
        weight: spec.weight,
        pressure: gamePressureOf(spec),
        winEma: new Array(headCount(spec.kind)).fill(0),
        spread: { tag: "unprobed" },
        lastProbeMs: 0,
      };
      if (implementation === "fused") {
        // Deliberately no tfjs Adversary here. The production WGSL trainer owns
        // its predictor weights and Adam state; constructing the reference
        // models used to retain 23–79 idle tensors depending on K.
        return { ...base, implementation: "fused" };
      }
      return {
        ...base,
        implementation: "tfjs",
        adv: new Adversary({
          ...defaultAdversaryConfig(spec.kind, spec.encoding, observerGeometry),
          target: adversaryTargetOf(spec),
          loss: adversaryLossOf(spec),
          // Same resolved pair the fused trainer gets — the oracle must be the
          // same net, or a fused/tfjs disagreement reads as a kernel bug.
          hiddenUnits: predictorArchOf(spec).hiddenUnits,
          featureDim: predictorArchOf(spec).featureDim,
          batchTuples,
          // 3e-3 default (vs the module default 1e-3): the discriminator must track a
          // field that is itself moving every frame. A predictor that lags is
          // not a harder opponent, it is a stale one, and the generator then
          // farms reward for beating last second's guess.
          learningRate,
          seed: 20260727,
        }),
      };
    }
    default:
      return assertNeverPiece(spec, "createAdversary");
  }
}

/** Fold rate for the per-head win EMA — ≈ a 50-step window. */
const WIN_EMA_LAMBDA = 0.02;

function assertNeverPiece(x: never, where: string): never {
  throw new Error(`${where}: unhandled variant ${JSON.stringify(x)}`);
}

/**
 * Generator game term: `weight · L_G` on the explicitly selected target.
 *
 * The force target uses rawSignal and bypasses forceMagnitude/physics. The
 * post-velocity target instead receives normalized incoming velocity and the
 * pre-border post-update velocity explicitly. Soft-angle uses the exact smooth
 * S² loss; relative-scale modes use homogeneous local contrast plus a fixed
 * energy anchor. Scale-hold is intentionally general-sum rather than a strict
 * opposite-sign scalar game.
 */
export function adversaryGeneratorTerm(
  rt: AdversaryRuntime,
  pos: tf.Tensor2D,
  velocity: tf.Tensor2D,
  postUpdateVelocity: tf.Tensor2D,
  rawSignal: tf.Tensor2D,
  w: number,
  h: number,
  maxVelocity: number
): tf.Scalar {
  switch (rt.tag) {
    case "off":
      return tf.scalar(0);
    case "on": {
      if (rt.implementation !== "tfjs") {
        throw new Error(
          "adversaryGeneratorTerm: fused runtime cannot enter the tfjs oracle path"
        );
      }
      return tf.tidy(() => {
        const wh2 = tf.tensor2d([[w, h]]);
        const pn = pos.div(wh2) as tf.Tensor2D;
        const s =
          rt.adv.cfg.target.tag === "force"
            ? rt.adv.sampleTarget({ tag: "force", pos: pn, force: rawSignal })
            : rt.adv.sampleTarget({
                tag: "post-velocity",
                pos: pn,
                velocity: velocity.div(maxVelocity) as tf.Tensor2D,
                nextVelocity: postUpdateVelocity.div(maxVelocity) as tf.Tensor2D,
              });
        const term = rt.adv.generatorLoss(s).mul(rt.weight).asScalar();
        disposeTupleSample(s);
        // ONE composition site for the game's own counter-pressure. It reads
        // the RAW field output, never the encoded target: on the pair observer
        // the collapse is invisible in y (y → 0 IS the collapse) and only the
        // world-frame directions show it.
        return term.add(gamePressureLoss(rt.pressure, rawSignal)).asScalar();
      });
    }
    default:
      return assertNeverPiece(rt, "adversaryGeneratorTerm");
  }
}

/** δ: GamePressure → handler. Trivial body; the work is in the loss. */
export function gamePressureLoss(
  pressure: GamePressure,
  rawSignal: tf.Tensor2D
): tf.Scalar {
  switch (pressure.tag) {
    case "none":
      return tf.scalar(0);
    case "anti-collapse":
      return directionOrderLoss(
        rawSignal,
        pressure.tau,
        pressure.polar,
        pressure.nematic
      );
    default:
      return assertNeverPiece(pressure, "gamePressureLoss");
  }
}

/**
 * Legacy transition scale retained for URL/test compatibility. The strict
 * force-target game does not use this quantity.
 */
export function stepScaleOf(
  cfg: ArtPieceConfig,
  w: number,
  h: number,
  maxVelocity: number = cfg.maxVelocity,
  border: BorderMode = { tag: "wrap" },
  forceMagnitude: number = cfg.forceMagnitude
): number {
  return Math.max(1e-6, maxVelocity / Math.min(w, h));
}

/**
 * Discriminator step on one detached batch of raw `F(x)`, plus telemetry.
 * Called with no generator tape open: the predictor owns the only variables.
 */
export function adversaryTrainStep(
  rt: AdversaryRuntime,
  pos: tf.Tensor2D,
  velocity: tf.Tensor2D,
  postUpdateVelocity: tf.Tensor2D,
  rawSignal: tf.Tensor2D,
  w: number,
  h: number,
  maxVelocity: number
): AdversaryTelemetry {
  switch (rt.tag) {
    case "off":
      return { tag: "off" };
    case "on": {
      if (rt.implementation !== "tfjs") {
        throw new Error(
          "adversaryTrainStep: fused runtime cannot enter the tfjs oracle path"
        );
      }
      // NOT wrapped in tf.tidy: tidy's return type is `TensorContainer`, which
      // cannot carry a telemetry record. Every intermediate is disposed by hand
      // below instead — verified by the `tensors` counter in the HUD, which must
      // stay flat while an adversary piece runs.
      const wh2 = tf.tensor2d([[w, h]]);
      const pn = pos.div(wh2) as tf.Tensor2D;
      let vNorm: tf.Tensor2D | null = null;
      let nextVNorm: tf.Tensor2D | null = null;
      const s = (() => {
        if (rt.adv.cfg.target.tag === "force") {
          return rt.adv.sampleTarget({ tag: "force", pos: pn, force: rawSignal });
        }
        vNorm = velocity.div(maxVelocity) as tf.Tensor2D;
        nextVNorm = postUpdateVelocity.div(maxVelocity) as tf.Tensor2D;
        return rt.adv.sampleTarget({
          tag: "post-velocity",
          pos: pn,
          velocity: vNorm,
          nextVelocity: nextVNorm,
        });
      })();
      const report = rt.adv.trainStep(s);
      // The HUD reports the same relaxed-WTA payoff used by both sides of the
      // configured D objective. The pure nearest-head residual remains a diagnostic in the
      // core, but is not silently substituted for either player's objective.
      const surTen = rt.adv.surprise(s);
      const meanTen = surTen.mean();
      const surprise = meanTen.dataSync()[0];
      surTen.dispose();
      meanTen.dispose();
      // R₁/R₂ over the SAME raw field batch the pressure is priced on, using
      // the same estimator the fused trainer reduces on-device. Measured only
      // when a pressure is declared — see AdversaryTelemetry.directionOrder.
      const directionOrder: DirectionOrder =
        rt.pressure.tag === "anti-collapse"
          ? {
              tag: "measured",
              ...directionOrderParameters(rawSignal, rt.pressure.tau),
            }
          : { tag: "unmeasured" };
      // ~1 Hz HEAD-SPREAD PROBE — wall-clock gated, NOT per step: it costs one
      // forward per head plus a dataSync (a pipeline sync on webgpu). This
      // batch's `u` serves as the probe contexts — the spread is a property of
      // the HEADS, so any recent batch of real contexts is a valid probe. Must
      // run before the sample is disposed. The first step probes immediately so
      // `unresolved` lasts one discriminator step, not one second.
      const nowMs = performance.now();
      if (rt.spread.tag === "unprobed" || nowMs - rt.lastProbeMs >= SPREAD_PROBE_MS) {
        rt.spread = rt.adv.headSpread(s.u);
        rt.lastProbeMs = nowMs;
      }
      disposeTupleSample(s);
      wh2.dispose();
      pn.dispose();
      vNorm?.dispose();
      nextVNorm?.dispose();

      // Recent-window win share — see the `winEma` doc on AdversaryRuntime for
      // why the adversary's own cumulative counter is the wrong statistic here.
      // First step seeds directly; an EMA crawling up from zero would report a
      // total collapse for its first 50 steps on every run.
      const seeded = rt.winEma.some((x) => x > 0);
      for (let j = 0; j < rt.winEma.length; j++) {
        rt.winEma[j] = seeded
          ? rt.winEma[j] + WIN_EMA_LAMBDA * (report.winCounts[j] - rt.winEma[j])
          : report.winCounts[j];
      }
      const total = rt.winEma.reduce((a, x) => a + x, 0) || 1;
      const winFractions = rt.winEma.map((x) => x / total);
      const kind = rt.adv.cfg.kind;
      // Win-skew floor at a twentieth of uniform. A live field is genuinely
      // non-stationary, so head shares wobble; the tripwire should fire on a
      // head that has STOPPED winning, not on one that is merely unpopular.
      // The skew alone is not the verdict. classifyHeads can identify pileup
      // from small spread, but separated skew cannot estimate support size.
      const winsSkewed = winFractions.some((f) => f < 0.05 / rt.adv.k);
      return {
        tag: "on",
        variant:
          `${kind.tag === "wta" ? `wta k=${kind.k} ε=${kind.relaxEps}` : "single"}` +
          ` · ${rt.adv.cfg.target.tag} · ${rt.adv.cfg.loss.tag}`,
        k: rt.adv.k,
        surprise,
        predLoss: report.loss,
        directionOrder,
        // The tfjs field REFUSES a family label (HelmholtzField.forces throws
        // for classes > 0), so this arm can never have a per-family split.
        perFamily: { tag: "unmeasured" },
        payoffReference: payoffReferenceOf(rt.adv.cfg.loss),
        winFractions,
        health: classifyHeads(winsSkewed, rt.spread, rt.adv.rewardScaleState()),
        weight: rt.weight,
      };
    }
    default:
      return assertNeverPiece(rt, "adversaryTrainStep");
  }
}

/**
 * MEAN-SPEED READBACK — an operational check on the dimensionless physical
 * drive bound. The adjusted observer removes the reward's radial signal, while
 * the drive policy is what actually keeps bounded-field particle velocities
 * away from the component clip. Particle state lives in GPU buffers the HUD
 * cannot otherwise see. Exactly the
 * GpuSurpriseStats subsample pattern, applied to the vel buffer: a fixed
 * 1024-particle prefix (8 KB) copied every `every` frames, mapped async,
 * mean |v| computed on the CPU. The prefix is unbiased for the same reason
 * the surprise prefix is — particle index carries no state structure.
 */
class GpuSpeedStats {
  /** mean |v| in px/frame over the sampled prefix (NaN until first sample) */
  mean = NaN;
  private readonly staging: GPUBuffer;
  private readonly sample: number;
  private readonly every: number;
  private pending = false;
  private pendingBytes = 0;

  constructor(device: GPUDevice, opts: { sample?: number; every?: number } = {}) {
    this.sample = opts.sample ?? 1024;
    this.every = Math.max(1, opts.every ?? 8);
    this.staging = device.createBuffer({
      size: this.sample * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  encodeSample(enc: GPUCommandEncoder, velBuf: GPUBuffer, n: number, frame: number): boolean {
    if (this.pending || frame % this.every !== 0) return false;
    const bytes = Math.min(this.sample, n) * 8;
    if (bytes === 0) return false;
    enc.copyBufferToBuffer(velBuf, 0, this.staging, 0, bytes);
    this.pendingBytes = bytes;
    return true;
  }

  afterSubmit(recorded: boolean): void {
    if (!recorded || this.pending) return;
    this.pending = true;
    const bytes = this.pendingBytes;
    this.staging
      .mapAsync(GPUMapMode.READ, 0, bytes)
      .then(() => {
        const v = new Float32Array(this.staging.getMappedRange(0, bytes).slice(0));
        this.staging.unmap();
        this.pending = false;
        let acc = 0;
        const n = v.length >> 1;
        for (let i = 0; i < n; i++) {
          acc += Math.hypot(v[2 * i], v[2 * i + 1]);
        }
        this.mean = acc / Math.max(1, n);
      })
      .catch(() => {
        this.pending = false;
      });
  }

  destroy(): void {
    try {
      this.staging.destroy();
    } catch (_) {}
  }
}

// ---------------------------------------------------------------------------
// Gallery
// ---------------------------------------------------------------------------

/** Maximum local sensitivity without imposing any spiral destination. */
export const MAX_CHAOS_FIELD_LOSS: FieldLossSpec = {
  W_CHAOS: 1,
  W_ISO: 1,
  W_DIV: 0.5,
  W_SPIRAL: 0,
  W_COVER: 0,
  W_CENTER: 0,
  HH: 1e-2,
  SPIRAL_TURNS: 3,
};

/**
 * NORMALIZED STRUCTURE ↑ plus a small divergence anchor — the "Neural Field ·
 * Max Structure" objective.
 *
 * `W_STRUCT: 1` is the natural scale: L_struct is exactly [0,1] (Jensen), so a
 * weight of 1 means "one loss unit between a pure-DC field and a pure-AC one"
 * and there is nothing to tune against.
 *
 * `W_DIV: 0.6` is NOT decoration, and it is the one number here that was tuned
 * rather than derived. TUNED AGAINST WHAT, MEASURED:
 *
 * The structure ratio is scale-invariant in VALUE but its gradient is not
 * amplitude-neutral. `∂L/∂(mean‖Fs‖²) = −L/(ms+ε)` is negative, so the descent
 * direction contains a component that GROWS every force — small (it is
 * proportional to L, and L falls to ~0.001 once the DC mode is gone) but
 * systematic, and Adam renormalizes small-but-systematic into full-size steps.
 * Left alone the field random-walks its amplitude up until tanh clips.
 *
 * Live measurement, `tools/health_audit.mjs struct`, satFrac after 60–120 s
 * (satFrac = fraction of the domain with BOTH tanh components past ±0.9):
 *
 *     W_DIV 0.05, lr 0.003  →  0.00 on one run, **0.35** on another (frozen)
 *     W_DIV 0.3,  lr 0.003  →  0.00 and 0.14
 *     W_DIV 0.6,  lr 0.002  →  **0.00 and 0.00**, ac 0.62–0.70, dc/ac 0.04–0.21
 *
 * Init is unseeded (tfjs glorot), so this is genuinely run-to-run: a single
 * healthy run does NOT clear a weight here. `div_i = (∇·F)²` is UNNORMALIZED
 * and grows with the field's derivatives, which is exactly why it is the
 * available counterweight to an amplitude drift — and why it must not go much
 * higher, or the piece becomes a divergence-free piece that happens to import
 * W_STRUCT.
 */
export const MAX_STRUCTURE_FIELD_LOSS: FieldLossSpec = {
  W_CHAOS: 0,
  W_ISO: 0,
  W_DIV: 0.6,
  W_STRUCT: 1,
  W_SPIRAL: 0,
  W_COVER: 0,
  W_CENTER: 0,
  HH: 1e-2,
  SPIRAL_TURNS: 3,
};

/** No direct aesthetic loss; weights move only through external game gradients. */
export const ZERO_FIELD_LOSS: FieldLossSpec = {
  W_CHAOS: 0,
  W_ISO: 0,
  W_DIV: 0,
  W_SPIRAL: 0,
  W_COVER: 0,
  W_CENTER: 0,
  HH: 1e-2,
  SPIRAL_TURNS: 3,
};

/** Fused radial spiral + tiny center (matches spiralPlusCenterLoss(2e-5)). */
export const SPIRAL_FIELD_LOSS: FieldLossSpec = {
  W_CHAOS: 0,
  W_ISO: 0,
  W_DIV: 0,
  W_SPIRAL: 1,
  W_COVER: 0,
  W_CENTER: 0.00002,
  HH: 1e-2,
  SPIRAL_TURNS: 3,
};

/** Fused Spiral Cover objective (curve→particles Chamfer ↑). */
export const COVER_FIELD_LOSS: FieldLossSpec = {
  W_CHAOS: 0,
  W_ISO: 0,
  W_DIV: 0,
  W_SPIRAL: 0,
  W_COVER: 1,
  W_CENTER: 0,
  COVER_SAMPLES: 256,
  HH: 1e-2,
  SPIRAL_TURNS: 3,
};

/** Fused center attractor (Vortex). */
export const CENTER_FIELD_LOSS: FieldLossSpec = {
  W_CHAOS: 0,
  W_ISO: 0,
  W_DIV: 0,
  W_SPIRAL: 0,
  W_COVER: 0,
  W_CENTER: 0.001,
  HH: 1e-2,
  SPIRAL_TURNS: 3,
};

export const GALLERY: ArtPieceConfig[] = [
  {
    // Spiral attractor — look (Ghost/Clean/Trails) and arch are dock axes.
    name: "Spiral",
    named: false,
    particleCount: 1000,
    friction: 0.985,
    forceMagnitude: 3.0,
    maxVelocity: 22,
    resetRate: 0.012,
    drawRate: 2,
    learningRate: 0.01,
    backgroundColor: [12, 0, 34],
    alphaBlend: 0.06,
    renderer: "alpha-fade",
    fieldArch: ARCH.mlpShallow,
    archEditable: true,
    lookEditable: true,
    fieldLoss: { ...SPIRAL_FIELD_LOSS, W_CENTER: 0.00005 },
    computeLoss: spiralPlusCenterLoss(0.00005),
  },
  {
    name: "Vortex",
    named: false,
    particleCount: 1200,
    friction: 0.985,
    forceMagnitude: 3.0,
    maxVelocity: 20,
    resetRate: 0.015,
    drawRate: 1,
    learningRate: 0.01,
    backgroundColor: [12, 0, 34],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    fieldArch: ARCH.mlpShallow,
    archEditable: true,
    lookEditable: true,
    fieldLoss: CENTER_FIELD_LOSS,
    computeLoss: (p, w, h) =>
      tf.tidy(() => centerLoss(p, w, h).mul(0.001).asScalar()),
  },
  {
    // Particle→spiral radial residual (+ tiny center). Arch/look via dock.
    name: "Galaxy",
    named: false,
    particleCount: 1500,
    friction: 0.975,
    forceMagnitude: 4.0,
    maxVelocity: 30,
    resetRate: 0.01,
    drawRate: 3,
    learningRate: 0.005,
    backgroundColor: [2, 0, 12],
    alphaBlend: 0.03,
    renderer: "alpha-fade",
    fieldArch: ARCH.mlp256,
    archEditable: true,
    lookEditable: true,
    fieldLoss: SPIRAL_FIELD_LOSS,
    computeLoss: spiralPlusCenterLoss(0.00002),
  },
  {
    // Cover-only Chamfer ↑. Arch (Fourier/SIREN/…) and look via dock.
    name: "Spiral Cover",
    named: false,
    particleCount: 1500,
    friction: 0.975,
    forceMagnitude: 4.0,
    maxVelocity: 30,
    resetRate: 0.01,
    drawRate: 3,
    learningRate: 0.005,
    backgroundColor: [2, 0, 12],
    alphaBlend: 0.03,
    renderer: "alpha-fade",
    fieldArch: ARCH.mlp256,
    archEditable: true,
    lookEditable: true,
    fieldLoss: COVER_FIELD_LOSS,
    computeLoss: spiralCoverLoss({ beta: 1 }),
  },
  {
    // Two direct-vector MLP heads share one explicit maximum-sensitivity loss.
    // Encoding variants (SIREN / Fourier / HashGrid) are dock presets — same
    // loss. The alpha slider is a neutral output mix, not order/chaos.
    // particleCount is 200k (slider to 1M): advection is a single fused WGSL
    // dispatch (see render/webgpu/advect.ts), so count no longer gates FPS.
    name: "Neural Field · Max Chaos",
    named: false,
    particleCount: 200000,
    friction: 0.99,
    forceMagnitude: 3.5,
    maxVelocity: 26,
    resetRate: 0.01,
    drawRate: 2,
    learningRate: 0.01,
    backgroundColor: [6, 2, 20],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    fieldArch: { ...ARCH.dualStd, alpha: 0.7 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    lookEditable: true,
    fieldLoss: MAX_CHAOS_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(MAX_CHAOS_FIELD_LOSS),
  },
  {
    // MULTI-SPECIES: 3 particle classes. Head B takes r(pos, onehot(class))
    // while head A and the isotropy pressure are class-blind. Class is a
    // stable hash of particle index (storage-free); renderer colours by
    // species. FUSED-ONLY (tfjs has no class input — ?train=tfjs ignored).
    name: "Neural Field · Species",
    named: false,
    particleCount: 200000,
    friction: 0.99,
    forceMagnitude: 3.5,
    maxVelocity: 26,
    resetRate: 0.01,
    drawRate: 2,
    learningRate: 0.01,
    backgroundColor: [4, 2, 16],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    fieldArch: { ...ARCH.dualStd, alpha: 0.7, classes: 3 },
    lookEditable: true,
    fieldLoss: MAX_CHAOS_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(MAX_CHAOS_FIELD_LOSS),
  },

  // ══ ADVERSARY PIECES ═══════════════════════════════════════════════════
  // Standalone games default to generator/field Adam LR=1e-3 and predictor
  // LR=3e-3. Both are live (and URL-addressable as ?gLR / ?dLR) because their
  // ratio is a real game-capacity dial, not a cosmetic multiplier.
  // Do not describe `adversary.weight` as a generator learning-rate knob:
  // with no structural field gradient, Adam is approximately invariant to
  // multiplying the whole external gradient by a positive scalar.
  // The standalone field is rewarded for being hard to predict, not for any
  // particular shape. Supported standard/Fourier classless configurations run
  // the fused WTA kernel; the tfjs path remains an explicit reference/fallback.
  {
    // ONE-HEAD CONTROL. Input x and strict target F(x) form a deterministic
    // conditional for a frozen field. A sufficiently expressive, sufficiently
    // trained predictor can drive its Bayes risk toward approximation error.
    // The live field co-trains and has finite capacity, so no claim is made that
    // the displayed residual must monotonically vanish.
    name: "Adversary · Single (control)",
    named: false,
    particleCount: 90000,
    friction: 0.97,
    drive: 0.55,
    forceMagnitude: forceMagnitudeForDrive(0.55, 22, 0.97),
    maxVelocity: 22,
    // Low on purpose so the artwork has long coherent trajectories. The strict
    // adversary observes raw F(x), not the respawn transition.
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.001,
    backgroundColor: [3, 2, 10],
    alphaBlend: 0.05,
    renderer: "surprise",
    colormap: "inferno",
    // Encoding is a dock knob here, not a game invariant — the observer,
    // objective, K and ε are what this piece is. See DUAL_ARCH_DOCK.
    fieldArch: { ...ARCH.dualStd, alpha: 0.7 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    adversary: {
      tag: "on",
      kind: { tag: "single" },
      encoding: { tag: "point" },
      // REWARD UNITS CHANGED 2026-07-27: generatorLoss is now normalized by an
      // EMA of RMS‖y‖ — dimensionless residual per unit field signal.
      weight: 0.01,
      // Anti-collapse: the soft-angle north pole pays the generator 3.15x
      // more for a DEAD field than for a varied one, so every adversarial
      // piece prices direction order. See GALLERY_ANTI_COLLAPSE.
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // Eight guesses instead of one, relaxed (ε = 0.05) so a starved head still
    // receives gradient even when it rarely wins. With a frozen field the point
    // conditional is deterministic, so extra heads add capacity rather than
    // creating multimodality. The co-trained finite-capacity game may retain a
    // nonzero residual; K alone is not evidence of novelty.
    // Coloured by velocity with long ghost trails — the motion is the subject.
    name: "Adversary · WTA K=8",
    named: false,
    particleCount: 120000,
    friction: 0.97,
    drive: 0.65,
    forceMagnitude: forceMagnitudeForDrive(0.65, 26, 0.97),
    maxVelocity: 26,
    resetRate: 0.005,
    drawRate: 2,
    learningRate: 0.001,
    backgroundColor: [8, 2, 22],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    // Encoding is a dock knob here, not a game invariant — the observer,
    // objective, K and ε are what this piece is. See DUAL_ARCH_DOCK.
    fieldArch: { ...ARCH.dualStd, alpha: 0.62 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 8, relaxEps: 0.05 },
      encoding: { tag: "point" },
      weight: 0.012, // reward units — see the note on the Single piece
      // Anti-collapse: the soft-angle north pole pays the generator 3.15x
      // more for a DEAD field than for a varied one, so every adversarial
      // piece prices direction order. See GALLERY_ANTI_COLLAPSE.
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // THE INTERESTING ONE. m = 2 with an SE(2)-canonicalized pair encoding: the
    // context is the separation r alone (the complete invariant of two labelled
    // points under translation + rotation) and the target is the relative raw
    // field signal F(x), resolved in the pair's own frame. Absolute position and
    // orientation are thrown away. Measured fixtures retain substantially more
    // residual than the point control, but that does not prove an irreducible
    // population floor for every trained field.
    //
    // K = 4, DOWN FROM 8 (2026-07-27). At K=8 this piece ran with a permanently
    // skewed win histogram (▁▁▁▁▁█▃▁): u = r is 1-D, so P(y|r) offers fewer
    // skewed win histogram. Histogram skew alone cannot estimate support or
    // distinguish separated rare winners from predictor pileup. K=4 keeps the
    // measured quantization benefit with less surplus capacity; the HUD only
    // makes a pileup claim when it has an actual spread measurement.
    //
    // Coolwarm and diverging, pivoting on the RUNNING MEDIAN, so the picture
    // answers "which pairs are harder than typical right now" rather than
    // "which are large" — the interesting question once the residual is
    // permanently nonzero.
    name: "Adversary · Pair WTA K=4",
    named: false,
    particleCount: 70000,
    friction: 0.97,
    drive: 0.65,
    forceMagnitude: forceMagnitudeForDrive(0.65, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.001,
    backgroundColor: [2, 3, 9],
    alphaBlend: 0.05,
    renderer: "surprise",
    colormap: "coolwarm",
    // Encoding is a dock knob here, not a game invariant — the observer
    // (pair-rotation-scale-adjusted) and the soft-angle objective are what
    // this piece is. See DUAL_ARCH_DOCK.
    fieldArch: { ...ARCH.dualStd, alpha: 0.55 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 4, relaxEps: 0.05 },
      encoding: { tag: "pair-rotation-scale-adjusted" },
      // Explicit — do not rely only on the legacy encoding→soft-angle alias.
      // Soft-angle (direction-only) is what made the pair swirls; raw-vector
      // on this observer collapses into amplitude / shear cheats.
      loss: {
        tag: "soft-angle",
        tau: ADVERSARY_OBJECTIVE_DEFAULTS.tau,
      },
      weight: 0.015, // reward units — see the note on the Single piece
      // Anti-collapse: the soft-angle north pole pays the generator 3.15x
      // more for a DEAD field than for a varied one, so every adversarial
      // piece prices direction order. See GALLERY_ANTI_COLLAPSE.
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // THREE-BODY CONGRUENCE GAME. m = 3: the context is the triangle's three
    // side lengths in canonical (descending) order — the COMPLETE congruence
    // invariant of an unordered triple under the full E(2) including
    // reflection — and the target is all three vertices' centered raw field
    // signals, resolved in the triangle's own frame (6-D). Even MORE context than pair
    // (3 numbers vs 1) yet the residual plateaus HIGHER (§6 ladder: tri/single
    // ~0.96 vs pair's ~0.91): richer shape information does not restore
    // predictability, because the pose ambiguity survives canonicalization and
    // the 6-D target spreads the conditional wider. That inversion — more
    // knowledge, less predictable — is the didactic point of the piece.
    //
    // K = 6 quantizes the tri conditional to ~0.64 (§6: tri/wta4 reached 0.64;
    // six heads give the 6-D target room without re-running the K=8 surplus
    // that the pair piece just retired). Viridis on the surprise channel so the
    // three adversary instruments read as a set: inferno = control, coolwarm =
    // pair (diverging, median-pivoted), viridis = tri. Slightly deeper
    // blue-black background than the pair piece keeps the viridis low end
    // legible. 60k particles = 20k triangles per surprise cycle — the tuple
    // count, not the particle count, is what the encode step pays for.
    name: "Adversary · Tri WTA K=6",
    named: false,
    particleCount: 60000,
    friction: 0.97,
    drive: 0.65,
    forceMagnitude: forceMagnitudeForDrive(0.65, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.001,
    backgroundColor: [1, 3, 12],
    alphaBlend: 0.05,
    renderer: "surprise",
    colormap: "viridis",
    // Encoding is a dock knob here, not a game invariant — the observer,
    // objective, K and ε are what this piece is. See DUAL_ARCH_DOCK.
    fieldArch: { ...ARCH.dualStd, alpha: 0.55 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 6, relaxEps: 0.05 },
      encoding: { tag: "tri" },
      weight: 0.012, // reward units — see the note on the Single piece
      // Anti-collapse: the soft-angle north pole pays the generator 3.15x
      // more for a DEAD field than for a varied one, so every adversarial
      // piece prices direction order. See GALLERY_ANTI_COLLAPSE.
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // FOUR LABELLED POSITIONS PER TUPLE. Member 0 anchors translation and
    // member 1 defines the co-rotating frame; members 2 and 3 keep their labels.
    // The observer removes global translation+rotation only. It deliberately
    // does not claim permutation or scale invariance.
    name: "Adversary · Quad WTA K=6",
    named: false,
    particleCount: 60000,
    friction: 0.97,
    drive: 0.65,
    forceMagnitude: forceMagnitudeForDrive(0.65, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.001,
    backgroundColor: [3, 1, 12],
    alphaBlend: 0.05,
    renderer: "surprise",
    colormap: "inferno",
    // Encoding is a dock knob here, not a game invariant — the observer,
    // objective, K and ε are what this piece is. See DUAL_ARCH_DOCK.
    fieldArch: { ...ARCH.dualStd, alpha: 0.55 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 6, relaxEps: 0.05 },
      encoding: { tag: "quad-labelled" },
      weight: 0.012,
      // Anti-collapse: the soft-angle north pole pays the generator 3.15x
      // more for a DEAD field than for a varied one, so every adversarial
      // piece prices direction order. See GALLERY_ANTI_COLLAPSE.
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  // Four Pixel GAN games on soft density drawings (docs/PIXEL_DISC.md).
  // Shared trunk: splat → conv3×3 → codebook. Reverse-mode gen through D(pos').
  //
  // ZERO_FIELD_LOSS IS LOAD-BEARING, exactly as on the relational-adversary
  // pieces above. Pass B computes `g = fieldLossGrad + extGrad` and hands the
  // SUM to Adam (train_wgsl.ts `g = g + extGrad0[t]`), so a game piece that
  // also carries a structural loss is really running two optimizers against
  // one weight buffer — and the loser is invisible, not merely weaker.
  //
  // These four originally shipped W_CHAOS .2 / W_ISO .6 / W_DIV .1. Measured
  // at the gallery dims (tools/pixel_disc_authority_probe.ts): ‖extGrad‖ was
  // 5e-4…1e-2 against a total ‖g‖ of ~8.5, i.e. the critic owned 0.006%–0.12%
  // of every field update. Every pass ran, every gradient was finite, and the
  // artwork was pure W_ISO — which reads as "the pixel adversary does nothing"
  // on any device. It is NOT a mobile bug: the probe measures the same ratio
  // at 390×844 and 1280×800, since the splat normalizes by width/height.
  //
  // Do not reintroduce a structural loss here to "shape" a pixel piece. The
  // knob for that is pixelDisc.weight, and it is only meaningful once the
  // critic is the sole gradient — Adam rescales by sqrt(v), so a small-but-
  // uncontested extGrad still produces full-size steps.
  {
    name: "Pixel · VecField",
    named: false,
    particleCount: 80000,
    friction: 0.97,
    drive: 0.65,
    forceMagnitude: forceMagnitudeForDrive(0.65, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.0015,
    backgroundColor: [4, 6, 14],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    // DECLARATIVE, not `createField`, so the model dock can swap the generator
    // architecture — the pixel critic accepts exactly the four ARCH_DOCK_DUAL
    // presets (docs/PLAN_PIXEL_GENERATOR_ARCH.md §1: dualStd / dualFourier /
    // dualSiren / dualHashgrid), the same five-arch capability the relational
    // adversary has. `createField` here was never a load-bearing recipe — it
    // called `createFieldFromArch` with this very object — but it made the arch
    // section invisible, because the dock gates on `piece.fieldArch`.
    //
    // `applyArchDockPreset` preserves α / semantic / classes, and no dual preset
    // carries `classes`, so the dock cannot reach the one refusal that lives
    // beyond the field gate (vec-field on a family-PLANED field, which has no
    // single F at a cell centre — `pixelDiscShader` throws).
    fieldArch: { ...ARCH.dualFourier, alpha: 0.55 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    pixelDisc: {
      kind: "vec-field",
      weight: 0.04,
      // G was briefly dropped to 8 because criticDisc/criticGen were
      // workgroup_size(1) — one GPU thread walked all G² cells, and G=16 cost
      // 17-37 ms/step against a ~16 ms frame budget, which is what made these
      // pieces look frozen on a phone. Those kernels are now workgroup-parallel
      // (cell-parallel activations, weight-parallel gradients), so G=16 measures
      // 1.2-1.5 ms/step — indistinguishable from G=8 — and the full-resolution
      // density grid is back. tools/pixel_disc_cost_probe.ts gates this.
      G: 16,
      E: 8,
      K: 16,
      hidden: 32,
      dt: 0.15,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    name: "Pixel · NextFrame",
    named: false,
    particleCount: 80000,
    friction: 0.97,
    drive: 0.65,
    forceMagnitude: forceMagnitudeForDrive(0.65, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.0015,
    backgroundColor: [4, 6, 14],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    // DECLARATIVE, not `createField`, so the model dock can swap the generator
    // architecture — the pixel critic accepts exactly the four ARCH_DOCK_DUAL
    // presets (docs/PLAN_PIXEL_GENERATOR_ARCH.md §1: dualStd / dualFourier /
    // dualSiren / dualHashgrid), the same five-arch capability the relational
    // adversary has. `createField` here was never a load-bearing recipe — it
    // called `createFieldFromArch` with this very object — but it made the arch
    // section invisible, because the dock gates on `piece.fieldArch`.
    //
    // `applyArchDockPreset` preserves α / semantic / classes, and no dual preset
    // carries `classes`, so the dock cannot reach the one refusal that lives
    // beyond the field gate (vec-field on a family-PLANED field, which has no
    // single F at a cell centre — `pixelDiscShader` throws).
    fieldArch: { ...ARCH.dualFourier, alpha: 0.55 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    pixelDisc: {
      kind: "next-frame",
      weight: 0.04,
      // G was briefly dropped to 8 because criticDisc/criticGen were
      // workgroup_size(1) — one GPU thread walked all G² cells, and G=16 cost
      // 17-37 ms/step against a ~16 ms frame budget, which is what made these
      // pieces look frozen on a phone. Those kernels are now workgroup-parallel
      // (cell-parallel activations, weight-parallel gradients), so G=16 measures
      // 1.2-1.5 ms/step — indistinguishable from G=8 — and the full-resolution
      // density grid is back. tools/pixel_disc_cost_probe.ts gates this.
      G: 16,
      E: 8,
      K: 16,
      hidden: 32,
      dt: 0.15,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    name: "Pixel · RealFake",
    named: false,
    particleCount: 80000,
    friction: 0.97,
    drive: 0.65,
    forceMagnitude: forceMagnitudeForDrive(0.65, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.0015,
    backgroundColor: [4, 6, 14],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    // DECLARATIVE, not `createField`, so the model dock can swap the generator
    // architecture — the pixel critic accepts exactly the four ARCH_DOCK_DUAL
    // presets (docs/PLAN_PIXEL_GENERATOR_ARCH.md §1: dualStd / dualFourier /
    // dualSiren / dualHashgrid), the same five-arch capability the relational
    // adversary has. `createField` here was never a load-bearing recipe — it
    // called `createFieldFromArch` with this very object — but it made the arch
    // section invisible, because the dock gates on `piece.fieldArch`.
    //
    // `applyArchDockPreset` preserves α / semantic / classes, and no dual preset
    // carries `classes`, so the dock cannot reach the one refusal that lives
    // beyond the field gate (vec-field on a family-PLANED field, which has no
    // single F at a cell centre — `pixelDiscShader` throws).
    fieldArch: { ...ARCH.dualFourier, alpha: 0.55 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    pixelDisc: {
      kind: "real-fake",
      weight: 0.03,
      // G was briefly dropped to 8 because criticDisc/criticGen were
      // workgroup_size(1) — one GPU thread walked all G² cells, and G=16 cost
      // 17-37 ms/step against a ~16 ms frame budget, which is what made these
      // pieces look frozen on a phone. Those kernels are now workgroup-parallel
      // (cell-parallel activations, weight-parallel gradients), so G=16 measures
      // 1.2-1.5 ms/step — indistinguishable from G=8 — and the full-resolution
      // density grid is back. tools/pixel_disc_cost_probe.ts gates this.
      G: 16,
      E: 8,
      K: 16,
      hidden: 32,
      dt: 0.15,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // Historical RealFake: the current cloud remains the positive example,
    // while the negative example is sampled from a rolling population of old
    // density snapshots. This makes the critic learn temporal improvement
    // rather than only the latest-vs-uniform distinction.
    name: "Pixel · RealFake · Historical",
    named: false,
    particleCount: 80000,
    friction: 0.97,
    drive: 0.65,
    forceMagnitude: forceMagnitudeForDrive(0.65, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.0015,
    backgroundColor: [4, 6, 14],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    fieldArch: { ...ARCH.dualFourier, alpha: 0.55 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    pixelDisc: {
      kind: "real-fake",
      weight: 0.03,
      G: 16,
      E: 8,
      K: 16,
      hidden: 32,
      dt: 0.15,
      historicalReplay: {
        capacity: 256,
        captureEvery: 4,
        probability: 0.75,
        horizon: 1024,
      },
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    name: "Pixel · Inpaint",
    named: false,
    particleCount: 80000,
    friction: 0.97,
    drive: 0.65,
    forceMagnitude: forceMagnitudeForDrive(0.65, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.0015,
    backgroundColor: [4, 6, 14],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    // DECLARATIVE, not `createField`, so the model dock can swap the generator
    // architecture — the pixel critic accepts exactly the four ARCH_DOCK_DUAL
    // presets (docs/PLAN_PIXEL_GENERATOR_ARCH.md §1: dualStd / dualFourier /
    // dualSiren / dualHashgrid), the same five-arch capability the relational
    // adversary has. `createField` here was never a load-bearing recipe — it
    // called `createFieldFromArch` with this very object — but it made the arch
    // section invisible, because the dock gates on `piece.fieldArch`.
    //
    // `applyArchDockPreset` preserves α / semantic / classes, and no dual preset
    // carries `classes`, so the dock cannot reach the one refusal that lives
    // beyond the field gate (vec-field on a family-PLANED field, which has no
    // single F at a cell centre — `pixelDiscShader` throws).
    fieldArch: { ...ARCH.dualFourier, alpha: 0.55 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    pixelDisc: {
      kind: "inpaint",
      weight: 0.04,
      // G was briefly dropped to 8 because criticDisc/criticGen were
      // workgroup_size(1) — one GPU thread walked all G² cells, and G=16 cost
      // 17-37 ms/step against a ~16 ms frame budget, which is what made these
      // pieces look frozen on a phone. Those kernels are now workgroup-parallel
      // (cell-parallel activations, weight-parallel gradients), so G=16 measures
      // 1.2-1.5 ms/step — indistinguishable from G=8 — and the full-resolution
      // density grid is back. tools/pixel_disc_cost_probe.ts gates this.
      G: 16,
      E: 8,
      K: 16,
      hidden: 32,
      dt: 0.15,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // GENERAL-SUM PREDICTOR GAME. Head A opposes the predictor, head B
    // cooperates with it, and C=(1-beta)A+beta B is derived only for display.
    // There are no C weights, C optimizer, or direct C loss. Stable particle
    // roles render A/B/C as exact red/green/blue.
    name: "Adversary · Agree + Disagree RGB",
    mode: "agree-disagree",
    palette: "rgb-roles",
    particleCount: 90000,
    friction: 0.97,
    drive: 0.6,
    forceMagnitude: forceMagnitudeForDrive(0.6, 22, 0.97),
    maxVelocity: 22,
    resetRate: 0.004,
    drawRate: 2,
    learningRate: 0.001,
    backgroundColor: [1, 1, 5],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    createField: () =>
      createFieldFromArch({
        ...ARCH.dualFourier,
        alpha: 0.5,
        fourierOctaves: 3,
        semantic: "agree-disagree",
      }),
    fieldLoss: ZERO_FIELD_LOSS,
    adversary: {
      tag: "on",
      game: "agree-disagree",
      kind: { tag: "wta", k: 4, relaxEps: 0.05 },
      encoding: { tag: "pair-rotation-scale-adjusted" },
      // Same observer as Pair — keep the swirl recipe (soft-angle), not the
      // encoding-only alias that used to look like RAW if the dock carried it.
      loss: {
        tag: "soft-angle",
        tau: ADVERSARY_OBJECTIVE_DEFAULTS.tau,
      },
      weight: 0.012,
      // Anti-collapse: the soft-angle north pole pays the generator 3.15x
      // more for a DEAD field than for a varied one, so every adversarial
      // piece prices direction order. See GALLERY_ANTI_COLLAPSE.
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // COMPOSITION, not a solo. The explicit maximum-sensitivity objective
    // (chaos + isotropy + divergence; spiral weight is exactly zero) is paired
    // with the relational adversary. Chaos maximises LOCAL sensitivity — how
    // fast two neighbours separate — while the adversary maximises RESIDUAL
    // UNPREDICTABILITY under an SE(2)-blind observer. They are not the same
    // objective and they do not want the same field: chaos is happy with a
    // smooth field that stretches, the adversary is not. The tension is the
    // point, and the weight is the dial between them.
    name: "Adversary · Chaos Weave",
    particleCount: 90000,
    friction: 0.97,
    drive: 0.75,
    forceMagnitude: forceMagnitudeForDrive(0.75, 26, 0.97),
    maxVelocity: 26,
    resetRate: 0.004,
    drawRate: 2,
    learningRate: 0.006,
    backgroundColor: [14, 3, 12],
    alphaBlend: 0.05,
    // alpha-fade (decay 0.94), not trail-buffer (0.97). At 0.97 and ~1e5
    // particles the Fourier field mixes the cloud to uniform coverage within
    // seconds and the long trails accumulate it into a flat brown wash —
    // checked by screenshot. The shorter trail keeps the filaments legible.
    // The neutral A/B output mix is tuned to 0.45 because that aesthetic blend
    // preserved more legible structure in the screenshot comparison. It is not
    // an order/chaos axis and the two direct-vector heads have no separate roles.
    renderer: "alpha-fade",
    // Encoding is a dock knob here, not a game invariant — the observer,
    // objective, K and ε are what this piece is. See DUAL_ARCH_DOCK.
    fieldArch: { ...ARCH.dualFourier, alpha: 0.45 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    fieldLoss: MAX_CHAOS_FIELD_LOSS,
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 6, relaxEps: 0.05 },
      encoding: { tag: "pair-rotation" },
      weight: 0.006, // reward units — see the note on the Single piece
      // Anti-collapse: the soft-angle north pole pays the generator 3.15x
      // more for a DEAD field than for a varied one, so every adversarial
      // piece prices direction order. See GALLERY_ANTI_COLLAPSE.
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    computeLoss: helmholtzChaosLoss(MAX_CHAOS_FIELD_LOSS),
  },
  // ══ APPEND ONLY — NEVER REORDER OR INSERT ABOVE THIS LINE ══════════════
  // A piece's GALLERY INDEX is persisted in two places outside this file:
  // shareable deep links carry it, and the dock's localStorage blob
  // (DOCK_STORAGE_KEY, src/index.tsx) stores `runtime.piece` as an integer.
  // Reordering or inserting silently re-points saved links and restored
  // sessions at a DIFFERENT artwork — a rename is loud (see
  // DEFAULT_PIECE_INDEX below), a renumber is not. New pieces go at the end.
  {
    // ORIGINAL DEFAULT PIECE (see DEFAULT_PIECE_NAME). The Pair WTA K=4 game — the
    // SE(2)-canonicalized pair observer with an explicit ANGULAR payoff, which
    // is what makes the swirls — moved onto the HASHGRID dual field.
    // The grid's local features give the generator per-cell freedom instead of
    // one global MLP surface, so the hard-to-predict structure lands as fine
    // filaments; α stays at the Pair piece's 0.55 so the position ENCODING is
    // the only deliberate difference between the two.
    //
    // THIS PIECE TRAINS FULLY FUSED as of the hashgrid adversary port
    // (D, generator reward AND field in one encodeStep). It used to be the one
    // gallery piece on the tfjs autograd path — the fused adversary refused
    // hashgrid fields, which cost ~40 ms/step here against ~0.8 ms for the
    // fused raw-MLP Pair piece. The port added a per-member dL/dEnc scratch
    // block plus the gather-side grid block in adversary_wgsl's fieldGrad; see
    // agent_notes/2026-08-17_120215_KST_hashgrid_adversary_fusion.md.
    //
    // renderer "alpha-fade", NOT "surprise" — deliberate, and load-bearing.
    // A surprise piece resolves to a surprise colour mode (resolveColorMode)
    // and the surprise renderer then takes the WHOLE render pass (see the
    // render step in `tick`); the splat, and therefore `stroke`, never runs.
    // Curl strokes are the point of this piece, so it colours by velocity with
    // ghost trails. (The RAW/PER-UNIT surprise diagnostic is fused-only; it now
    // WOULD be available here, but the stroke still wins the render pass.)
    //
    // ── PROMOTED GREAT WORK ────────────────────────────────────────────────
    // `drive`, `learningRate` and the angle+relative-scale `loss` below are a
    // CAPTURED LIVE TUNING, not chosen numbers: see GREAT_WORKS.md, entry
    // "Adversary · Pair · HashGrid · Curl — ink swirls" (commit b72aa54),
    // which holds the whole dock recipe plus its `?dock=` restore link. Keep
    // the two in step. Two of the recorded dials have NO piece field to be
    // promoted INTO, so the link stays the only way to get them back:
    //   · discriminator lr 7.2e-4 — `?dLR=` only; startLoop defaults to 3e-3.
    //   · border RESET — the dock hardcodes `{tag:"wrap"}` in defaultsForPiece
    //     (src/index.tsx) and always passes it down as an override, so this
    //     piece still OPENS ON WRAP however this entry is written.
    // (Train B 256 needs no field: it is already startLoop's sampleRate.)
    name: "Adversary · Pair · HashGrid · Curl",
    particleCount: 70000,
    friction: 0.97,
    drive: 0.9,
    // DERIVED from drive — the two move TOGETHER. Editing `drive` alone
    // silently keeps the old force scale (cfg.forceMagnitude is what a
    // no-dock, no-`?drive` start uses before resolveLiveGameControls agrees).
    forceMagnitude: forceMagnitudeForDrive(0.9, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.003,
    drawRate: 2,
    learningRate: 0.0048,
    backgroundColor: [2, 3, 9],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    // Curl: each particle draws its curved per-frame trajectory (2nd-order,
    // curlAmp=1) rather than a dot, so at 24 px/frame the cloud reads as ink.
    stroke: "curl",
    // Encoding is a dock knob here, not a game invariant — the observer,
    // objective, K and ε are what this piece is. See DUAL_ARCH_DOCK.
    fieldArch: { ...ARCH.dualHashgrid, alpha: 0.55 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 4, relaxEps: 0.05 },
      encoding: { tag: "pair-rotation-scale-adjusted" },
      // Explicit — do not rely only on the legacy encoding→soft-angle alias.
      // The ANGULAR term (direction-only) is what made the pair swirls;
      // raw-vector on this observer collapses into amplitude / shear cheats.
      // Relative-scale adds a local, homogeneous log-magnitude descriptor on
      // top, and the energy anchor holds ABSOLUTE rms so the generator cannot
      // buy that extra variance by blowing the field up — together they are
      // what stretch the swirls into long laminar filaments.
      //
      // LITERALS, deliberately, not ADVERSARY_OBJECTIVE_DEFAULTS.* (which
      // coincide with them today): these four are RECORDED values from the
      // GREAT_WORKS.md capture noted above, so they must not silently drift
      // when a shared default moves.
      loss: {
        tag: "angle-relative-scale",
        tau: 0.05,
        scaleWeight: 0.5,
        energyWeight: 0.1,
        energyTarget: 0.35,
      },
      weight: 0.015, // reward units — see the note on the Single piece
      // Anti-collapse: the soft-angle north pole pays the generator 3.15x
      // more for a DEAD field than for a varied one, so every adversarial
      // piece prices direction order. See GALLERY_ANTI_COLLAPSE.
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // THE ONE PIECE THAT OPTIMIZES STRUCTURE DIRECTLY.
    //
    // Every other piece in this gallery treats AC/DC as an OBSERVATION (see
    // src/health.ts): on the adversarial pieces nothing may feed the structure
    // metrics into a loss, because the whole claim under test is that a healthy
    // game raises them by itself. This piece is the control experiment for that
    // claim — it asks the field for the structure directly and shows what the
    // objective alone buys.
    //
    // WHY NORMALIZED, NOT RAW VARIANCE. The obvious objective is "maximize
    // AC = rms‖F − mean F‖". It does not work, and the reason is not subtle:
    // AC is homogeneous of degree 1 in F, so scaling every force by c scales AC
    // by c. Maximizing it is therefore satisfied by GROWING THE FIELD, not by
    // structuring it — the optimizer walks straight into tanh saturation
    // (satFrac 0.46 on the measured collapse baseline) and stops, having
    // acquired zero new spatial features. What this piece maximizes is the
    // SCALE-INVARIANT fraction
    //
    //     ac² / (ac² + dc²) = 1 − L_struct,   L_struct = (dc²+ε)/(rmsF²+ε)
    //
    // which is unchanged by F → cF, so the only way to move it is to trade
    // global constant push for spatial variation. See FieldLossSpec.W_STRUCT.
    //
    // W_DIV pairs with it for two reasons, one aesthetic and one numerical.
    // Aesthetic: the structure term has no opinion about the local CHARACTER of
    // the variation, and a compressible field satisfies it perfectly well by
    // building sinks that eat the cloud. Numerical: it is the counterweight to
    // this objective's Adam amplitude drift, and its weight was MEASURED against
    // exactly that — see MAX_STRUCTURE_FIELD_LOSS. Chaos/isotropy are
    // deliberately NOT stacked on: the Late Lesson from the collapse
    // investigation is that piling terms on produces a field optimized for the
    // sum and legible as none of them.
    name: "Neural Field · Max Structure",
    particleCount: 200000,
    friction: 0.985,
    forceMagnitude: 4.5,
    maxVelocity: 24,
    resetRate: 0.006,
    drawRate: 2,
    // 0.002, not Max Chaos's 0.01: see MAX_STRUCTURE_FIELD_LOSS on the Adam
    // amplitude drift this objective has once the DC mode is gone.
    learningRate: 0.002,
    backgroundColor: [3, 2, 14],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    // Same ink as the default piece: curl strokes draw each particle's curved
    // per-frame trajectory, so the structure reads as filaments rather than a
    // dot cloud — which is the entire point of a piece about structure.
    stroke: "curl",
    fieldArch: { ...ARCH.dualStd, alpha: 0.7 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    lookEditable: true,
    fieldLoss: MAX_STRUCTURE_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(MAX_STRUCTURE_FIELD_LOSS),
  },
  {
    // RGB FAMILIES — the first piece where the generator is conditioned on
    // something the DISCRIMINATOR CANNOT SEE.
    //
    // Every other adversarial piece plays a game over one field: the predictor
    // observes a context u and the field answers with one signal, so a
    // sufficiently strong predictor can in principle drive the residual toward
    // its approximation error. Here each particle carries a FAMILY label
    // c ∈ {R, G, B} (derived, never stored: pcg(i ^ CLASS_SALT) % 3) and the
    // generator is F(x, c) — a hashgrid with one feature plane per family, so
    // the three fields can differ CELL BY CELL. The predictor's context is the
    // POINT observer: the coordinate alone. It never receives c.
    //
    // WHY THAT IS A DIFFERENT GAME. P(F | x) is now a 3-mode mixture, and the
    // generator is paid for the modes being far apart — for two families
    // wanting opposite things at the same place. The relaxed-WTA predictor
    // answers with K hypotheses, so the whole game is set by K vs the family
    // count:
    //
    //   K >= 3  the predictor can park one head per family and the conditioning
    //           buys the generator nothing — an ordinary single-field game.
    //   K == 2  (SHIPPED) the predictor structurally cannot cover three modes.
    //           One family is always the mispredicted one, and the generator is
    //           paid to keep it that way — which family that is, and whether the
    //           role rotates, is the question the HUD's per-family row answers.
    //   K == 1  the predictor must answer with the mean, so all three families
    //           spread symmetrically. The stable, least interesting corner.
    //
    // K is live (?advK / the dock slider), so sweeping 1→4 with the per-family
    // chart open is the experiment this piece exists for.
    //
    // The per-family instrument REQUIRES the point observer: with m = 1 a tuple
    // has exactly one family, so attributing its payoff is exact. On a pair/tri
    // observer a tuple can mix families and AdvStats.perFamily correctly reports
    // `unmeasured` rather than inventing a bucketing rule — see
    // familyInstrument (adversary_wgsl.ts).
    //
    // Colour IS the instrument: palette "rgb-families" paints family 0/1/2 as
    // exact R/G/B from the same hash the kernels use, so "the green family
    // stopped fighting" is a thing you can see as well as read.
    //
    // Design + gates: agent_notes/2026-08-19_family_conditioned_hashgrid_adversary.md
    name: "Adversary · RGB Families · HashGrid",
    particleCount: 90000,
    friction: 0.97,
    drive: 0.6,
    forceMagnitude: forceMagnitudeForDrive(0.6, 24, 0.97),
    maxVelocity: 24,
    resetRate: 0.004,
    drawRate: 2,
    learningRate: 0.001,
    backgroundColor: [3, 3, 8],
    alphaBlend: 0.05,
    renderer: "alpha-fade",
    // Curl strokes, same as the default piece: three interleaved families read
    // as three inks only if each particle draws its trajectory rather than a dot.
    stroke: "curl",
    palette: "rgb-families",
    createField: () => createFieldFromArch(ARCH.familyHashgrid),
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 2, relaxEps: 0.05 },
      encoding: { tag: "point" },
      // Direction-only, like every other shipped game: raw-vector on a point
      // observer collapses into amplitude cheats.
      loss: {
        tag: "soft-angle",
        tau: ADVERSARY_OBJECTIVE_DEFAULTS.tau,
      },
      weight: 0.012, // reward units — see the note on the Single piece
      // Anti-collapse: the soft-angle north pole pays the generator 3.15x more
      // for a DEAD field than for a varied one. It matters MORE here — three
      // families all going quiet is a cheap way to be unpredictable.
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // Captured dock recipe promoted as a new work. Keep this entry appended:
    // gallery indices are part of persisted dock state and shared URLs.
    // The supplied recipe is point-observed, WTA-K10, soft-angle, and uses a
    // dual HashGrid field with curl ink.
    name: "Adversary · Point · HashGrid · Curl · WTA10 · Coolwarm Raw",
    particleCount: 190000,
    friction: 0.97,
    drive: 0.9,
    forceMagnitude: forceMagnitudeForDrive(0.9, 65.75, 0.97),
    maxVelocity: 65.75,
    resetRate: 0.014,
    drawRate: 2,
    learningRate: 0.0048,
    discriminatorLearningRate: 0.00002660725059798809,
    sampleRate: 9584,
    border: { tag: "reset" },
    colorMode: { tag: "surprise-raw", colormap: "coolwarm" },
    backgroundColor: [2, 3, 9],
    alphaBlend: 0.17,
    renderer: "alpha-fade",
    stroke: "curl",
    strokeLen: 3,
    fieldArch: { ...ARCH.dualHashgrid, alpha: 0.17 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 10, relaxEps: 0.1 },
      encoding: { tag: "point" },
      loss: { tag: "soft-angle", tau: 0.05 },
      weight: 0.015,
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  {
    // NAMED PIECE: Sand of Times. This is a preserved dock capture, not a
    // rename of the older Pair/HashGrid work at index 17. Gallery entries are
    // append-only because shared dock links persist their numeric piece index.
    name: "Sand of Times",
    particleCount: 190000,
    friction: 0.97,
    drive: 0.9,
    forceMagnitude: forceMagnitudeForDrive(0.9, 65.75, 0.97),
    maxVelocity: 65.75,
    resetRate: 0.014,
    drawRate: 2,
    learningRate: 0.0048,
    discriminatorLearningRate: 0.000005308844442309883,
    sampleRate: 9584,
    border: { tag: "reset" },
    colorMode: { tag: "surprise-raw", colormap: "inferno" },
    backgroundColor: [2, 3, 9],
    alphaBlend: 0.8,
    renderer: "alpha-fade",
    stroke: "curl",
    strokeLen: 3,
    fieldArch: { ...ARCH.dualHashgrid, alpha: 0.8 },
    archEditable: true,
    archDock: DUAL_ARCH_DOCK,
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 10, relaxEps: 0.1 },
      encoding: { tag: "pair-rotation-scale-adjusted" },
      loss: {
        tag: "angle-relative-scale",
        tau: 0.05,
        scaleWeight: 0.5,
        energyWeight: 0.1,
        energyTarget: 0.35,
      },
      predictor: { hiddenUnits: 128, featureDim: 64 },
      weight: 0.015,
      pressure: GALLERY_ANTI_COLLAPSE,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
];

/**
 * The piece the app loads into. Resolved BY NAME, never by a hardcoded index,
 * so that appending pieces cannot shift the default and renaming this one
 * fails LOUDLY at module load instead of silently falling back to GALLERY[0].
 */
export const DEFAULT_PIECE_NAME =
  "Adversary · Point · HashGrid · Curl · WTA10 · Coolwarm Raw";

export const DEFAULT_PIECE_INDEX: number = (() => {
  const index = GALLERY.findIndex((piece) => piece.name === DEFAULT_PIECE_NAME);
  if (index < 0) {
    throw new Error(
      `DEFAULT_PIECE_NAME "${DEFAULT_PIECE_NAME}" is not in GALLERY — ` +
        `rename the constant with the piece, or the app would silently open ` +
        `on a different artwork.`
    );
  }
  return index;
})();

// ---------------------------------------------------------------------------
// Physics step (inside optimizer.minimize — gradients flow through)
// ---------------------------------------------------------------------------
export function physicsForward(
  pos: tf.Tensor2D,
  vel: tf.Tensor2D,
  model: tf.Sequential | null,
  field: ForceField | null,
  cfg: ArtPieceConfig,
  w: number,
  h: number,
  maxVelocity: number = cfg.maxVelocity,
  border: BorderMode = { tag: "wrap" },
  forceMagnitude: number = cfg.forceMagnitude
): {
  newPos: tf.Tensor2D;
  newVel: tf.Tensor2D;
  /** Velocity after force/friction/clip and before border bounce/reset. */
  postUpdateVelocity: tf.Tensor2D;
  /** Raw learned output F(x), before forceMagnitude and every integrator term. */
  rawSignal: tf.Tensor2D;
  /** Physical force after the piece's forceMagnitude, used by aesthetic losses. */
  force: tf.Tensor2D;
} {
  const posNorm = pos.div(tf.tensor2d([[w, h]])) as tf.Tensor2D;
  // Field path: raw signed output used directly (NO -0.5 shift).
  // MLP path (legacy): sigmoid output re-centered by (raw - 0.5).
  const rawSignal = field
    ? field.forces(posNorm)
    : ((model!.predict(posNorm) as tf.Tensor2D).sub(0.5) as tf.Tensor2D);
  const forces = rawSignal.mul(forceMagnitude) as tf.Tensor2D;

  // PERF: clip/wrap the WHOLE [N,2] tensor in one op each instead of
  // slice-x/clip, slice-y/clip, concat (5 ops -> 1) and likewise for the wrap.
  // maxVelocity is symmetric so a single clipByValue covers both axes; mod
  // broadcasts a [1,2] to wrap x by w and y by h at once. ~8 fewer GPU
  // dispatches per call, and this runs twice a frame (learn + advect).
  const clippedVel = vel
    .add(forces)
    .mul(cfg.friction)
    .clipByValue(-maxVelocity, maxVelocity) as tf.Tensor2D;

  const q = pos.add(clippedVel) as tf.Tensor2D;
  let newPos: tf.Tensor2D;
  let newVel: tf.Tensor2D;
  switch (border.tag) {
    case "wrap":
      newPos = q.mod(tf.tensor2d([[w, h]])) as tf.Tensor2D;
      newVel = clippedVel;
      break;
    case "bounce": {
      const res = tf.tensor2d([[w, h]]);
      const lo = q.less(0);
      const hi = q.greaterEqual(res);
      const reflectedLo = tf.where(lo, q.neg(), q) as tf.Tensor2D;
      newPos = tf.where(hi, res.mul(2).sub(reflectedLo), reflectedLo) as tf.Tensor2D;
      newVel = tf.where(lo.logicalOr(hi), clippedVel.neg(), clippedVel) as tf.Tensor2D;
      break;
    }
    case "reset": {
      const res = tf.tensor2d([[w, h]]);
      const outside = q
        .less(0)
        .logicalOr(q.greaterEqual(res))
        .any(1)
        .reshape([-1, 1]);
      const respawn = tf.randomUniform(q.shape).mul(res) as tf.Tensor2D;
      newPos = tf.where(outside, respawn, q) as tf.Tensor2D;
      newVel = tf.where(outside, tf.zerosLike(clippedVel), clippedVel) as tf.Tensor2D;
      break;
    }
    default:
      throw new Error(`physicsForward: unknown border ${(border as { tag: string }).tag}`);
  }

  return {
    newPos,
    newVel,
    postUpdateVelocity: clippedVel,
    rawSignal,
    force: forces,
  };
}

// Random reset now lives INSIDE the fused advect kernel (PCG hash per
// particle+frame) — see src/render/webgpu/advect_wgsl.ts. The old tfjs
// randomReset (~10 dispatches/frame) is gone with the rest of the tfjs
// advect stage.

// Renderer is now in src/renderers.ts — three implementations:
//   "alpha-fade"   — dual-buffer ghost trails (fast, hardware-composited)
//   "trail-buffer" — ring buffer clean trails (precise, no ghosts)
//   "clean"        — no trails (fastest, debug/iteration)

// (spiralPixelPoints — the Canvas2D spiral-overlay helper — was deleted
// 2026-07-27: zero callers since the overlay died with the Canvas2D path.)

// ---------------------------------------------------------------------------
// Main simulation loop
// ---------------------------------------------------------------------------
/** Full-screen "needs WebGPU" notice. There is NO Canvas2D/WebGL fallback — by
 *  design (we're WebGPU-only). Shown when the browser has no WebGPU. */
function showWebGPUWarning(): void {
  document.documentElement.style.margin = "0";
  document.body.style.margin = "0";
  const o = document.createElement("div");
  o.style.cssText =
    "position:fixed;inset:0;z-index:10000;display:flex;align-items:center;" +
    "justify-content:center;background:#05010f;color:#cbd5ff;text-align:center;" +
    "font:16px/1.6 ui-monospace,monospace;padding:24px";
  o.innerHTML =
    '<div style="max-width:560px">' +
    '<div style="font-size:44px;margin-bottom:12px">⚡</div>' +
    '<div style="font-size:20px;margin-bottom:10px;color:#fff">This needs WebGPU</div>' +
    '<div style="margin-bottom:16px;color:#94a0c8">Neural Force Field Art runs ' +
    "entirely on the GPU (zero-copy tfjs → WebGPU). Your browser doesn't have " +
    "WebGPU enabled.</div>" +
    '<div><a href="https://caniuse.com/webgpu" target="_blank" ' +
    'style="color:#8ab4ff">Go get WebGPU working →</a> ' +
    '<span style="color:#5b6890">(Chrome / Edge / Safari 18+ / Firefox, latest)</span></div>' +
    "</div>";
  document.body.appendChild(o);
}

/**
 * Per-piece trail defaults for the splat renderer, keyed by the piece's
 * declared renderer style (the splat's decay is now the one real trail
 * mechanism behind all three looks): ghost pieces get soft trails,
 * trail-buffer pieces long streaks, clean pieces none. `?decay=F` overrides.
 */
const SPLAT_DECAY_BY_RENDERER: Record<RendererType, number> = {
  "alpha-fade": 0.94,
  "trail-buffer": 0.97,
  clean: 0,
  // ZERO, deliberately. The surprise mode is an INSTRUMENT: a faded trail shows
  // a stale residual, so trails would misreport the adversary's current state.
  // (The surprise renderer clears its attachment every frame anyway; this entry
  // is what keeps the splat path honest if a run switches back to it live.)
  surprise: 0,
};

/** Ink look presets — dock axis orthogonal to loss and field arch. */
export type InkLook = "ghost" | "clean" | "trails";

export const INK_LOOK_DECAY: Record<InkLook, number> = {
  ghost: SPLAT_DECAY_BY_RENDERER["alpha-fade"],
  clean: SPLAT_DECAY_BY_RENDERER.clean,
  trails: SPLAT_DECAY_BY_RENDERER["trail-buffer"],
};

export function inkLookFromRenderer(renderer: RendererType): InkLook {
  switch (renderer) {
    case "clean":
      return "clean";
    case "trail-buffer":
      return "trails";
    default:
      return "ghost";
  }
}

export function decayForRenderer(renderer: RendererType): number {
  return SPLAT_DECAY_BY_RENDERER[renderer];
}

/** Live stroke-length bounds, shared by URL ingestion and the dock slider. */
export const STROKE_LENGTH_RANGE = { min: 0.5, max: 16 } as const;

/**
 * κ for the splat stroke style. ONE resolution order, in priority order:
 *
 *   1. `?stroke=` — an explicit URL is the user's stated intent and wins.
 *   2. the piece's declared {@link ArtPieceConfig.stroke} — the artwork's recipe.
 *   3. "dot" — the shipped look, for the pieces that declare nothing.
 *
 * Exported because BOTH the loop and the React dock's first paint need the
 * same answer; two copies of this ladder would be two chances to disagree.
 * An unrecognised `?stroke=` is a hard error, exactly like `?adv=` / `?advM=`:
 * a typo must not silently paint dots and look like the feature is broken.
 */
export function resolveStrokeStyle(
  cfg: ArtPieceConfig,
  q: URLSearchParams
): SplatStyle {
  const raw = q.get("stroke");
  if (raw === null) return cfg.stroke ?? "dot";
  if (raw === "dot" || raw === "vel" || raw === "curl") return raw;
  throw new Error(`?stroke must be dot, vel or curl, got ${raw}`);
}

/**
 * Stroke length in FRAMES of travel, same ladder: `?strokeLen=` > the piece's
 * {@link ArtPieceConfig.strokeLen} > 3. `floatParam` (Number.isFinite, not
 * `|| 3`) so an explicit `?strokeLen=0` clamps to the documented 0.5 floor
 * instead of silently becoming the default.
 */
export function resolveStrokeLength(
  cfg: ArtPieceConfig,
  q: URLSearchParams
): number {
  return Math.max(
    STROKE_LENGTH_RANGE.min,
    Math.min(STROKE_LENGTH_RANGE.max, floatParam(q, "strokeLen", cfg.strokeLen ?? 3))
  );
}

export interface LoopHandle {
  field: HelmholtzField | null;
  getParticleCount(): number;
  setParticleCount(n: number): void;
  getSampleRate(): number;
  /** Clamps to [1, getMaxSampleRate()] — over-cap warns, never throws. */
  setSampleRate(n: number): void;
  /** Largest batch the live trainer can run (fused: device+layout derived).
   *  The UI bounds its "train B" control with this. */
  getMaxSampleRate(): number;
  /** Live respawn fraction — with particle-sourced training this is also the
   *  exploration dial (resets feed fresh uniform states into the batch). */
  getResetRate(): number;
  setResetRate(r: number): void;
  /** Splat trail persistence (0 = hard clear … 0.99 = long streaks) — wired
   *  straight to SplatRenderer.decay, the "trails" slider's backing knob. */
  getDecay(): number;
  setDecay(d: number): void;
  /** Symmetric component velocity clip, in pixels/frame. */
  getMaxVelocity(): number;
  setMaxVelocity(v: number): void;
  /** Dimensionless physical drive. On opted-in adversary pieces, drive<=1 is
   *  a proof-level bound preventing a tanh-bounded field component from ever
   *  reaching maxVelocity. */
  getDrive(): number;
  setDrive(v: number): void;
  /** Generator/field Adam step size. Live on both fused and tfjs paths. */
  getGeneratorLearningRate(): number;
  setGeneratorLearningRate(v: number): void;
  /** Predictor/discriminator Adam step size. Live on both fused and tfjs paths. */
  getDiscriminatorLearningRate(): number;
  setDiscriminatorLearningRate(v: number): void;
  /** Neutral two-head output blend. It is not an order/chaos coordinate. */
  getBlend(): number;
  setBlend(v: number): void;
  /** Splat stroke controls, owned by React rather than an imperative DOM island. */
  getStrokeStyle(): SplatStyle;
  setStrokeStyle(v: SplatStyle): void;
  getStrokeLength(): number;
  setStrokeLength(v: number): void;
  /** Dimensionless multiplier on the adversarial reward — see
   *  {@link adversaryGeneratorTerm} for why it is dimensionless. No-op when the
   *  active piece has no adversary. */
  getAdversaryWeight(): number;
  setAdversaryWeight(x: number): void;
  /** Dimensionless multiplier on the PIXEL critic's generator reward. A distinct
   *  scalar from {@link LoopHandle.getAdversaryWeight} because it multiplies a
   *  soft-density residual, not a WTA payoff in predictor-output units. No-op
   *  when the active piece has no pixel critic. */
  getPixelCriticWeight(): number;
  setPixelCriticWeight(x: number): void;
  /** Latest discriminator telemetry: surprise, predictor loss, per-head win
   *  shares and the collapse tripwire. `{tag:"off"}` for non-adversary pieces. */
  getAdversaryTelemetry(): AdversaryTelemetry;
  /** How the cloud is coloured. Raw/per-unit diagnostics are fused-only. */
  getColorMode(): ColorMode;
  setColorMode(m: ColorMode): void;
  /** Normalisation window of the selected fused surprise plane plus exact cloud
   *  coverage. `null` for velocity/RGB or an oracle-only tfjs path. */
  getSurpriseSpan(): { lo: number; mid: number; hi: number; covered: number; collapsed: boolean } | null;
  /** Latest ~1 Hz field-grid measurement (AC/DC/saturation/Okubo–Weiss).
   *  `{tag:"unprobed"}` until the first readback lands — a real state, so the
   *  chart shows a gap rather than a run of zeros that reads as a dead field. */
  getFieldHealth(): FieldHealth;
}

export interface StartLoopOptions {
  /** React-owned dock slot for the imperative high-frequency HUD. */
  telemetryHost?: HTMLElement;
  /** Compile-time/layout choices. Changing these restarts the active piece. */
  overrides?: {
    border?: BorderMode;
    adversaryEncoding?: TupleEncoding;
    adversaryTarget?: AdversaryTarget;
    adversaryLoss?: AdversaryLoss;
    k?: number;
    relaxEps?: number;
    /** Swap declarative field architecture (aesthetic / archEditable pieces). */
    fieldArch?: FieldArch;
    /** Swap the PREDICTOR's width. Compiled, like fieldArch — restarts. */
    predictor?: PredictorArch;
  };
}

export function startLoop(
  canvas: HTMLCanvasElement,
  configIndex: number,
  onReady?: (handle: LoopHandle) => void,
  options: StartLoopOptions = {}
): () => void {
  let running = true;
  const cfg = GALLERY[configIndex];
  let particleCount = cfg.particleCount; // rendered/advected particles (live)
  let sampleRate = cfg.sampleRate ?? 256; // points the field trains on per frame (live)
  let warnedSampleRate = 0; // last over-cap request warned about (drag de-dupe)
  let resetRate = cfg.resetRate; // respawn fraction (live — see setResetRate)
  let maxVelocity = cfg.maxVelocity;
  const query = new URLSearchParams(location.search);
  const initialGameControls = resolveLiveGameControls(cfg, query, maxVelocity);
  const driveEnabled = initialGameControls.driveEnabled;
  let drive = initialGameControls.drive;
  let forceMagnitude = initialGameControls.forceMagnitude;
  let generatorLearningRate = initialGameControls.generatorLearningRate;
  let discriminatorLearningRate = initialGameControls.discriminatorLearningRate;
  const groupedGeneratorRates = cfg.generatorLearningRates;
  // κ runs ONCE, here: piece defaults merged with `?adv/?advK/?advM/?advEps/
  // ?advWeight`, the physical `?drive`, game `?gLR/?dLR`, and `?color/?cmap`.
  // Everything below consumes canonical values.
  const resolvedAdv = resolveAdversary(cfg.adversary, query);
  const advSpec: AdversarySpec =
    resolvedAdv.tag === "on" && options.overrides
      ? {
          ...resolvedAdv,
          encoding: options.overrides.adversaryEncoding ?? resolvedAdv.encoding,
          target: options.overrides.adversaryTarget ?? adversaryTargetOf(resolvedAdv),
          loss: options.overrides.adversaryLoss ?? adversaryLossOf(resolvedAdv),
          predictor: options.overrides.predictor ?? predictorArchOf(resolvedAdv),
          kind:
            resolvedAdv.kind.tag === "wta"
              ? {
                  tag: "wta",
                  k: options.overrides.k ?? resolvedAdv.kind.k,
                  relaxEps: options.overrides.relaxEps ?? resolvedAdv.kind.relaxEps,
                }
              : resolvedAdv.kind,
        }
      : resolvedAdv;
  if (advSpec.tag === "on") {
    objectiveDims(
      advSpec.encoding,
      adversaryTargetOf(advSpec),
      adversaryLossOf(advSpec)
    );
  }
  const border = options.overrides?.border ?? cfg.border ?? ({ tag: "wrap" } as const);
  const observerGeometry: ObserverGeometry =
    border.tag === "wrap" ? "periodic" : "euclidean";
  let colorMode = resolveColorMode(cfg, advSpec, query);
  // `?advEvery=N` (default 1): frames between tfjs-reference discriminator
  // updates. Matching the fused path is the correctness default; larger values
  // are an explicit weaker-opponent experiment, not a performance default. The tfjs
  // reference adversary does K small GPU→CPU readbacks per step (one Adam
  // minimize per head), which on the webgpu backend is a pipeline stall per head
  // — measured in browser QA at ~8 ms per head per frame, i.e. the whole frame
  // budget at K = 8. MEASURED on this machine (Metal, the pair piece back when
  // it ran K=8 — now "Pair WTA K=4" — headless
  // Chrome): advEvery=1 → 156 ms learn, 4 → 99 ms, 8 → 114 ms (run-to-run noise
  // is ±15%, so 4 and 8 are within each other).
  //
  // IT IS NOT A FREE KNOB. A discriminator that updates a quarter as often is a
  // WEAKER opponent, so the equilibrium surprise rises for reasons that have
  // nothing to do with the field getting more interesting. Raise it to see the
  // motion, drop it to 1 when reading the numbers.
  const advEvery = Math.max(1, intParam(query, "advEvery", 1));
  // `?window=K` (1..16): trajectory-window training — sets BOTH rollout=K and
  // trainEvery=K. tools/window_test.ts proves the K-step imagined rollout from
  // live particle states IS the next K real frames (maxΔ ≈ 6e-5 px at K=6), so
  // this is true window training with zero recording machinery. 0 = not set.
  const windowK = Math.max(
    0,
    Math.min(
      16,
      parseInt(new URLSearchParams(location.search).get("window") ?? "0", 10) || 0
    )
  );
  // `?trainEvery=N`: run the train step every Nth frame. Default 1 for every
  // piece now that the adversary is fused; the old adversary default of 2 was a
  // stale workaround for the removed 19–32 ms tfjs hot path and also changed
  // the strength of the opponent.
  const trainEvery =
    windowK > 0
      ? windowK
      : Math.max(1, intParam(query, "trainEvery", 1));

  // WebGPU-only — no Canvas2D/WebGL fallback (by design). Warn + bail if absent.
  if (!GpuPointRendererWebGPU.isSupported()) {
    showWebGPUWarning();
    return () => {};
  }

  // RETINA: the canvas BACKING store is native resolution (devicePixelRatio,
  // capped at 2 — 3x phone DPRs quadruple the accumulator for little gain)
  // while the physics WORLD stays w×h CSS pixels: kernels, trainer and losses
  // are untouched; only the backing store and the splat accumulator scale.
  // Math.ceil keeps canvas.width EXACTLY equal to the splat renderer's native
  // accumulator width (its tonemap indexes the accumulator by fragment coord —
  // a row-stride mismatch would shear the image). The quad renderer needs no
  // change: its vertex math is pos/resolution-relative (incl. pointSize, which
  // is in world units), and its fwidth AA sharpens automatically at native res.
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = window.innerWidth;
  const h = window.innerHeight;
  canvas.width = Math.ceil(w * dpr);
  canvas.height = Math.ceil(h * dpr);
  // Full-screen, no scroll (canvas is inline by default -> descender gap).
  // Explicit CSS width/height pin the element to CSS pixels so the native
  // backing maps 1:1 onto device pixels.
  canvas.style.cssText =
    `display:block;position:fixed;inset:0;width:${w}px;height:${h}px`;
  document.documentElement.style.margin = "0";
  document.body.style.cssText = "margin:0;overflow:hidden;background:#000";

  // ALL tensor/model creation is DEFERRED to the async init below: tfjs throws
  // if you build tensors/models before the highest-priority backend (webgpu)
  // has finished initializing (needs await tf.ready()). Assigned there.
  let field: ForceField | null = null;
  let model: tf.Sequential | null = null;
  let varList: tf.Variable[] | undefined = undefined;
  let optimizer: tf.Optimizer | null = null;
  // Particle state lives in the fused kernel's GPUBuffers, NOT tfjs tensors —
  // training samples random points, so tfjs never touches the particles.
  let advect: AdvectKernel | null = null;
  // Fused trainer (field pieces): analytic backward + Adam in 2 WGSL
  // dispatches, updating the advect kernel's weights buffer IN PLACE — the
  // whole hot path is then GPU-only, tfjs idle. Gradients verified against
  // tfjs autograd (cos=1.0000000) by tools/train_test.ts. `?train=tfjs`
  // falls back to the tfjs optimizer path for A/B comparison.
  let trainer: FusedTrainer | null = null;
  // Fused adversary (adversary pieces on fused-capable fields): discriminator
  // train + generator reward recorded into the SAME frame encoder, extGrads
  // feeding the field trainer's pass B. Null on the tfjs path.
  let advTrainer: AdversaryTrainer | null = null;
  let pixelDiscTrainer: PixelDiscTrainer | null = null;
  /**
   * LIVE generator-reward multiplier for the pixel critic — the counterpart of
   * `advRt.weight` for the relational game, and the reason the dock's reward
   * row means something on a Pixel piece.
   *
   * It is a SEPARATE scalar from the adversary's on purpose. They multiply
   * different quantities: `advRt.weight` scales a relaxed-WTA payoff in
   * predictor-output units (dock range 0..20), while this scales a soft-density
   * residual (shipped values 0.03-0.04). One slider driving both under one range
   * would be a shared NAME over two different units — the kind of sharing that
   * looks tidy and silently mistunes a game.
   */
  let pixelGenWeight = initialGameControls.pixelCriticWeight;
  // Agree+Disagree owns TWO independent predictors. `advTrainer` is lane A
  // (field head 0, disagree) and this is lane B (field head 1, agree). Both
  // read the same field-weight buffer; neither owns or mutates it.
  let advTrainerB: AdversaryTrainer | null = null;
  // Percentile feed for the fused packed surprise planes + the mean-speed
  // physical-drive instrument.
  let advSurStats: GpuSurpriseStats | null = null;
  let speedStats: GpuSpeedStats | null = null;
  // last stats object folded into the win EMA (identity-compared: the trainer
  // mints a fresh object per readback, so this cannot double-fold a batch)
  let advSeenStats: object | null = null;
  let advSeenStatsB: object | null = null;
  let advWinEmaB: number[] = [];
  let trainSource: "particles" | "random" = "particles";
  let mixRandom = 0;
  let hudLoss = NaN;
  // ---- health instrumentation (OBSERVATION ONLY — see src/health.ts) -------
  // The field probe reads the SAME packed weights the advect kernel draws from,
  // so it works identically on the fused and tfjs trainer paths. It runs on its
  // own encoder, on a ~1 Hz timer, and writes to buffers nothing else reads:
  // the hot path stays one encodeStep encoder → one queue.submit.
  let fieldProbe: FieldProbe | null = null;
  let fieldHealth: FieldHealth = { tag: "unprobed" };
  /** L2 of the pixel critic's external field gradient; NaN until the first
   *  1 Hz readback lands. Read-only — this never re-enters the buffer. */
  let pixelExtGradNorm = NaN;
  let pixelExtGradPending = false;
  let lastHealthMs = 0;
  const loopStartMs = performance.now();
  // The shared GPUDevice (tfjs's) and the optional GPU profiler. Assigned in
  // the async init once the webgpu backend is confirmed; timer is null when the
  // adapter lacks "timestamp-query" (→ HUD falls back to CPU-encode lines).
  let device: GPUDevice | null = null;
  let timer: GpuTimer | null = null;
  let wh: tf.Tensor2D | null = null;
  let renderer: GpuPointRendererWebGPU | null = null;
  // Compute-splat renderer (accumulation buffer + tonemap): ~5 ms at 1M vs
  // ~25 ms for instanced quads (4M verts + additive overdraw). The DEFAULT
  // path at every count — its radial cone kernel gives round dots at low N
  // too (the old 2x2 bilinear looked square, which is why quads used to win
  // below 20k), and its decay gives real ghost trails. `?render=quads`
  // restores the quad path; `?decay=F` overrides decay; `?dot=F` sets the
  // splat radius in CSS px.
  let splat: SplatRenderer | null = null;
  // The live adversary game and surprise renderers (one per colormap, built
  // lazily and CACHED — rebuilding would call context.unconfigure(), which the
  // splat/quad renderers share).
  let advRt: AdversaryRuntime = { tag: "off" };
  const surpriseRenderers = new Map<ColormapName, GpuSurpriseRendererWebGPU>();
  let advTele: AdversaryTelemetry = { tag: "off" };
  const renderOverride = new URLSearchParams(location.search).get("render");
  const SPLAT_MIN_N = 0;
  const exposureScale =
    parseFloat(new URLSearchParams(location.search).get("exposure") ?? "1") || 1;
  // `?stroke=dot|vel|curl` — splat draw style. "vel"/"curl" draw per-frame
  // geometric strokes along each particle's backward trajectory, so fast
  // particles (maxVelocity ~26 px/frame vs a ~1.6px dot) read as continuous
  // filaments instead of disconnected dots. The ladder (URL > piece > "dot",
  // and `?strokeLen=` > piece > 3) lives in resolveStrokeStyle /
  // resolveStrokeLength so React's first paint resolves it identically.
  let strokeStyle: SplatStyle = resolveStrokeStyle(cfg, query);
  let strokeLen = resolveStrokeLength(cfg, query);
  let frame = 0;

  // --- Telemetry HUD: FPS + per-stage timing so the bottleneck is visible ----
  const tele = document.createElement("div");
  tele.dataset.testid = "fps-hud";
  tele.setAttribute("aria-label", "Performance telemetry");
  tele.style.cssText =
    (options.telemetryHost ? "" : "position:fixed;top:8px;right:8px;z-index:9999;") +
    "font:11px/1.45 ui-monospace,monospace;color:#8f8;background:rgba(0,0,0,.72);" +
    "padding:6px 9px;border:1px solid rgba(130,170,255,.2);border-radius:5px;" +
    "white-space:pre;pointer-events:none;letter-spacing:.02em";
  (options.telemetryHost ?? document.body).appendChild(tele);
  const ema = (prev: number, x: number, a = 0.12) =>
    prev === 0 ? x : prev * (1 - a) + x * a;
  let emaFrame = 0,
    emaTrain = 0,
    emaRender = 0,
    lastT = performance.now();

  /**
   * ADVERSARY block of the health snapshot, from the EXACT floats each trainer
   * produces — not from the HUD, and not from the aggregated telemetry record.
   *
   * The fused arm reads `AdversaryTrainer.lastStats` directly, so `payoff` and
   * `payoffUngated` are the two independent readback slots rather than one
   * number printed twice; their disagreement is the nonfinite canary the audit
   * gates on. The tfjs arm has no gate and no batch-RMS slot, so it reports a
   * DIFFERENT shape rather than padding the missing fields with zeros.
   */
  function advHealthBlock(): AdvHealth | null {
    // Dispatch on the RUNTIME, not on "did lastStats happen to be there". A
    // fused piece whose first readback has not landed must report `null` (no
    // sample yet), never a tfjs-shaped record — the audit reads `trainer` to
    // decide whether the payoffUngated canary is even applicable.
    if (advRt.tag !== "on") return null;
    const t = advTele;
    if (t.tag === "off") return null;
    const order = t.directionOrder;
    // `unmeasured` → null, never 0: a zero R₁ reads as "perfectly isotropic",
    // which is the exact opposite of what an unmeasured collapsed field is.
    const r1 = order.tag === "measured" ? order.r1 : null;
    const r2 = order.tag === "measured" ? order.r2 : null;
    if (advRt.implementation === "fused") {
      // Lane A only on the two-lane Agree+Disagree game — the same choice
      // `AdversaryTelemetry.directionOrder` documents: each lane prices its OWN
      // field head, so a blended payoff/R₁ is not a statistic anything computes
      // and inventing one would be a lie. Lane B is visible in the HUD.
      const stats = advTrainer?.lastStats;
      if (!stats) return null;
      return {
        trainer: "fused",
        payoff: stats.surprise,
        payoffUngated: stats.payoffUngated,
        surprise: stats.surprise,
        r1,
        r2,
        batchRms: stats.batchRms,
        heads: [...stats.winCounts],
      };
    }
    return {
      trainer: "tfjs",
      payoff: t.surprise,
      surprise: t.surprise,
      r1,
      r2,
      heads: [...t.winFractions],
    };
  }

  /** PIXEL CRITIC block. `extGradNorm` is the 1 Hz readback of the buffer the
   *  critic hands the field trainer — an observation of the gradient, never a
   *  write to it. */
  function pixelHealthBlock(): PixelHealth | null {
    const s = pixelDiscTrainer?.lastStats;
    if (!s) return null;
    return {
      dLoss: s.discLoss,
      gLoss: s.genLoss,
      extGradNorm: Number.isFinite(pixelExtGradNorm) ? pixelExtGradNorm : null,
    };
  }

  /**
   * Publish `window.__nffHealth` and kick the two async 1 Hz readbacks.
   *
   * Both readbacks are FIRE-AND-FORGET into local state: awaiting them here
   * would put a pipeline sync inside `tick`, which is the one thing the fused
   * path exists to avoid. A late sample simply lands in the next snapshot.
   */
  function publishHealth(nowMs: number): void {
    if (fieldProbe && advect) {
      const alpha = field ? (field as HelmholtzField).alpha : 0;
      void fieldProbe
        .sample(alpha)
        .then((m) => {
          if (m) fieldHealth = { tag: "measured", metrics: m };
        })
        .catch((e: unknown) => {
          console.warn(`[health] field probe failed — ${String(e)}`);
        });
    }
    if (pixelDiscTrainer && !pixelExtGradPending) {
      pixelExtGradPending = true;
      void pixelDiscTrainer
        .readExtGrads()
        .then((g) => {
          pixelExtGradNorm = l2Norm(g);
        })
        .catch(() => {})
        .finally(() => {
          pixelExtGradPending = false;
        });
    }
    const snapshot: HealthSnapshot = {
      piece: cfg.name,
      frame,
      t: (nowMs - loopStartMs) / 1000,
      fps: emaFrame > 0 ? 1000 / emaFrame : 0,
      learnMs: emaTrain,
      backend: tf.getBackend(),
      trainer: trainer ? "fused" : "tfjs",
      adv: advHealthBlock(),
      field: fieldHealth.tag === "measured" ? fieldHealth.metrics : null,
      pixel: pixelHealthBlock(),
      // Read off the LAYOUT, not off cfg/options: this must describe the
      // network that is running, so that a sweep cell which silently ignored
      // its `?arch=` is visibly identical to its neighbour instead of being
      // labelled as a distinct architecture. See ArchHealth in src/health.ts.
      arch: advect
        ? {
            kind: advect.layout.spec.kind,
            weightFloats: advect.layout.totalFloats,
            macsPerParticle: totalMacs(advect.layout),
            encoding: advect.layout.encoding.kind,
            classes: advect.layout.classes,
          }
        : null,
    };
    (window as unknown as HealthWindow).__nffHealth = snapshot;
  }

  /**
   * Adversary block of the telemetry HUD. Prints the two numbers that can lie
   * about each other (surprise vs predictor loss — they differ once ε > 0), the
   * per-head win shares as a bar so a dead head is visible at a glance, and the
   * surprise channel's RAW percentile window so the colour normalisation can be
   * audited instead of trusted (src/draw/robust_norm.ts, SPAN_FLOOR note).
   */
  function adversaryHudLines(): string {
    // Snapshot into a const: `advTele` is a mutable outer binding, so TypeScript
    // would drop the narrowing inside the .map callback below.
    const t = advTele;
    if (t.tag === "off") return "";
    // Bar height is the win share RELATIVE TO UNIFORM (1/k), so a healthy
    // mixture reads as a flat row at ▄ whatever k is, and a starved head reads
    // as ▁ next to it. An absolute scale would make every k=8 run look collapsed.
    const bars = t.winFractions
      .map((f) => "▁▂▃▄▅▆▇█"[Math.max(0, Math.min(7, Math.floor(f * t.k * 3)))])
      .join("");
    const displayedPayoff =
      advSpec.tag === "on" &&
      adversaryLossOf(advSpec).tag === "angle-scale-hold"
        ? "D-joint"
        : "surprise";
    let out =
      // 3 decimals, not 2: the shipped composition weights are 0.006–0.015
      // (post /100 rescale) and toFixed(2) rendered EVERY one of them as
      // "0.01" — the HUD could not tell Chaos Weave (0.006) from Pair K=4
      // (0.015). Caught reading the headless HUD capture, 2026-07-28.
      `adv     ${t.variant}  w ${t.weight.toFixed(3)}${advTrainer ? "  (fused)" : ""}\n` +
      `${displayedPayoff.padEnd(8)} ${t.surprise.toExponential(2)}  ` +
      `pred ${t.predLoss.toFixed(4)}\n` +
      // "HEAD COLLAPSE" is printed only for measured predictor pileup.
      // Separated skew stays support-unresolved.
      `heads   ${bars}${headHealthHud(t.health)}\n`;
    if (t.branches) {
      out +=
        `A disagree  sur ${t.branches.disagree.surprise.toExponential(2)} ` +
        `pred ${t.branches.disagree.predLoss.toExponential(2)}` +
        `${headHealthHud(t.branches.disagree.health)}\n` +
        `B agree     sur ${t.branches.agree.surprise.toExponential(2)} ` +
        `pred ${t.branches.agree.predLoss.toExponential(2)}` +
        `${headHealthHud(t.branches.agree.health)}\n` +
        `C blend     display only (zero direct loss)\n`;
    }
    // Physical-drive instrument: print mean Euclidean speed next to the
    // component clip. The drive invariant, not reward normalization, is the
    // anti-saturation mechanism.
    if (speedStats && Number.isFinite(speedStats.mean)) {
      out += `speed   ${speedStats.mean.toFixed(2)} px/f  (clip ${maxVelocity.toFixed(1)})\n`;
    }
    // Fused surprise span. The metric name is load-bearing: raw shared payoff
    // and payoff per unit target signal answer different questions.
    if (isSurpriseColorMode(colorMode) && advTrainer && advSurStats) {
      const s = advSurStats.norm.raw;
      const coverageA = advTrainer.surpriseCoverage().covered;
      const coverageB = advTrainerB?.surpriseCoverage().covered ?? coverageA;
      const metric = colorMode.tag === "surprise-raw" ? "raw" : "per-unit";
      out +=
        `span(${metric}) p2 ${s.lo.toExponential(2)} p50 ${s.mid.toExponential(2)} ` +
        `p98 ${s.hi.toExponential(2)}  covered ` +
        `${(100 * Math.min(coverageA, coverageB)).toFixed(0)}%\n`;
    }
    return out;
  }

  // Learning is DECOUPLED from motion: we train the field on a small random
  // batch each frame (real-time, cheap) while the FUSED WGSL KERNEL advects
  // ALL particles in ONE compute dispatch (MLP forward + integrate + clip +
  // wrap + random reset — was ~40 tfjs dispatches). Weights flow tfjs→kernel
  // as ~10KB of GPU→GPU copies per frame; particle state never touches tfjs,
  // so particle count scales to 1M+ without touching the train cost.
  async function tick() {
    // Bail WITHOUT rescheduling when the loop is torn down (or a dependency
    // vanished mid-teardown) — rearming here would resurrect a dead piece.
    if (!running || !(optimizer || trainer) || !advect || !wh || !device) {
      return;
    }
    frame++;

    // ONE command encoder for the whole frame. On the FUSED path it records
    // trainer pass A+B, the advect pass, and the render, then submits ONCE —
    // versus the old three-submits-per-frame (train, advect, render). The tfjs
    // legacy path can't share the encoder for its learn (optimizer.minimize
    // does its own internal submits), but advect+render still share this one.
    const enc = device.createCommandEncoder();
    // Per-pass timestamp descriptors (undefined when the profiler is absent):
    // 0/1 rollout(A), 2/3 optim(B), 4/5 advect, 6/7 render.
    const tsA = timer?.writes(0, 1);
    const tsB = timer?.writes(2, 3);
    const tsAdvect = timer?.writes(4, 5);
    const tsRender = timer?.writes(6, 7);

    // (1) LEARN — one gradient step on a SMALL batch.
    //     Fused path (field pieces): 2 WGSL dispatches (analytic backward +
    //     Adam) recorded into `enc`, writing the shared weights buffer in place
    //     — no tfjs, no readback; the advect pass below then reads the freshly
    //     trained weights (pass ordering in one encoder inserts the barrier).
    //     tfjs path (legacy MLP pieces / ?train=tfjs): optimizer.minimize.
    const trainStart = performance.now();
    // async-readback record flags for this frame (fused adversary path)
    let advStatsRec = false;
    let advStatsRecB = false;
    let pixelDiscStatsRec = false;
    let advSurRec = false;
    let speedRec = false;
    // `?trainEvery=N`: amortize training over N frames — applies to BOTH the
    // fused path (imagined-rollout batch) and the tfjs path (SIREN/Fourier,
    // whose autograd + encoding learn stage is heavy — this is how a Fourier
    // piece reaches 60fps: train every 2-3 frames, advect every frame).
    if (frame % trainEvery !== 0) {
      // skip training this frame (both paths)
    } else if (trainer) {
      // FUSED ADVERSARY first (when present): discriminator train + generator
      // reward, recorded BEFORE the field passes so (a) the discriminator sees
      // the pre-update field — the correct minimax ordering, same as the tfjs
      // path below — and (b) extGrads are fresh when the field's pass B adds
      // them. ~0.8 ms at B=512 (tools/train_wta_test.ts §6), so no advEvery
      // amortization is needed on this path.
      if (advTrainer && advRt.tag === "on") {
        // Snapshot the NARROWED runtime into a const: `advRt` is a mutable
        // outer binding, so TS drops the `tag === "on"` narrowing inside the
        // closure below (same pattern as `rtSnap` in the telemetry block). It
        // is the same object, so the dock's live weight slider — which mutates
        // `advRt.weight` in place — is still read fresh every frame.
        const advOn = advRt;
        // The top-right "train B" control owns BOTH sides' live batch size.
        // Adversary buffers are compiled to 1024 tuples; particle coverage can
        // reduce effective B further inside AdversaryTrainer.
        const b = adversaryBatchSize(sampleRate, 1024);
        const advPhysics = {
          width: w,
          height: h,
          forceMagnitude,
          friction: cfg.friction,
          maxVelocity,
        };
        const encodeAdversaryBranch = (branch: AdversaryTrainer) =>
          branch.encodeStep(
            enc,
            advPhysics,
            {
              b,
              alpha: (field as HelmholtzField).alpha,
              lr: discriminatorLearningRate,
              // A and B see the same tuple indices. Their predictor weights,
              // Adam state, field lane and named generator role are independent.
              seed: frame,
              source: "particles",
              genSeed: branch.genSeed(advOn.weight, 1, b),
              applyDisc: true,
            }
          );
        // Both complete their alternating D→G sequence while field weights
        // remain read-only. The ONE field step below then sums the two fresh
        // external gradients and applies exactly one field Adam update.
        encodeAdversaryBranch(advTrainer);
        if (advTrainerB) encodeAdversaryBranch(advTrainerB);
      }
      // `pixelDiscTrainer !== null` IS "resolvePixelCritic approved a critic and
      // it was constructed" — the old `&& cfg.pixelDisc` re-asked the declaration
      // question at the hot site, which is the same double-gate resolvePixelCritic
      // exists to remove.
      if (pixelDiscTrainer) {
        const b = Math.min(sampleRate, 512);
        pixelDiscTrainer.encodeStep(enc, {
          b,
          alpha: (field as HelmholtzField).alpha,
          lr: discriminatorLearningRate,
          genWeight: pixelGenWeight,
          applyDisc: true,
          width: w,
          height: h,
        });
        pixelDiscStatsRec = pixelDiscTrainer.recordStats(enc);
      }
      trainer.encodeStep(
        enc,
        {
          width: w,
          height: h,
          forceMagnitude,
          friction: cfg.friction,
          maxVelocity,
        },
        {
          n: sampleRate,
          alpha: (field as HelmholtzField).alpha,
          lr: generatorLearningRate,
          learningRates:
            groupedGeneratorRates?.tag === "shared-heads"
              ? {
                  tag: "shared-heads",
                  shared:
                    groupedGeneratorRates.shared *
                    (generatorLearningRate / Math.max(cfg.learningRate, 1e-12)),
                  head0:
                    groupedGeneratorRates.head0 *
                    (generatorLearningRate / Math.max(cfg.learningRate, 1e-12)),
                  head1:
                    groupedGeneratorRates.head1 *
                    (generatorLearningRate / Math.max(cfg.learningRate, 1e-12)),
                }
              : undefined,
          seed: frame,
          source: trainSource,
          mixRandom,
        },
        tsA,
        tsB
      );
      if (frame % 30 === 0) {
        trainer
          .readLoss()
          .then((l) => (hudLoss = l.loss))
          .catch(() => {});
      }
    } else {
      // (1a) DISCRIMINATOR FIRST, on its OWN detached raw-force batch. No tape
      //      is open here, so nothing can reach the field; and the predictor
      //      sees the field as it is BEFORE this frame's generator step, which
      //      is the correct alternating order (the generator is rewarded
      //      against the just-updated opponent, not a one-step-stale one).
      //      It is a SEPARATE forward pass, not a reuse of the generator's:
      //      reusing would mean sharing a graph between the two optimizers,
      //      which is exactly the coupling this split exists to prevent.
      if (advRt.tag === "on" && frame % advEvery === 0) {
        tf.tidy(() => {
          const dp = tf.randomUniform([sampleRate, 2], 0, 1).mul(wh!) as tf.Tensor2D;
          const dv =
            advRt.tag === "on" &&
            advRt.implementation === "tfjs" &&
            advRt.adv.cfg.target.tag === "post-velocity"
              ? (tf.randomUniform(
                  [sampleRate, 2],
                  -maxVelocity,
                  maxVelocity
                ) as tf.Tensor2D)
              : (tf.zeros([sampleRate, 2]) as tf.Tensor2D);
          const dr = physicsForward(
            dp,
            dv,
            model,
            field,
            cfg,
            w,
            h,
            maxVelocity,
            border,
            forceMagnitude
          );
          advTele = adversaryTrainStep(
            advRt,
            dp,
            dv,
            dr.postUpdateVelocity,
            dr.rawSignal,
            w,
            h,
            maxVelocity
          );
        });
      }
      // (1b) GENERATOR. varList is the FIELD's weights only, so the predictor
      //      heads inside the adversarial term are frozen structurally.
      optimizer!.minimize(
        () =>
          tf.tidy(() => {
            const tp = tf.randomUniform([sampleRate, 2], 0, 1).mul(
              wh!
            ) as tf.Tensor2D;
            const tv =
              advRt.tag === "on" &&
              advRt.implementation === "tfjs" &&
              advRt.adv.cfg.target.tag === "post-velocity"
                ? (tf.randomUniform(
                    [sampleRate, 2],
                    -maxVelocity,
                    maxVelocity
                  ) as tf.Tensor2D)
                : (tf.zeros([sampleRate, 2]) as tf.Tensor2D);
            const r = physicsForward(
              tp,
              tv,
              model,
              field,
              cfg,
              w,
              h,
              maxVelocity,
              border,
              forceMagnitude
            );
            // ONE composition site. The piece's loss stays a pure function of
            // positions; the adversarial reward is added here so any aesthetic
            // term can be paired with any adversary without editing either.
            const aesthetic = cfg.computeLoss(r.newPos, w, h, {
              force: r.force,
              field,
            });
            const adversarial = adversaryGeneratorTerm(
              advRt,
              tp,
              tv,
              r.postUpdateVelocity,
              r.rawSignal,
              w,
              h,
              maxVelocity
            );
            return aesthetic.add(adversarial).asScalar();
          }),
        false,
        varList
      );
    }
    const trainMs = performance.now() - trainStart;

    // Fused adversary telemetry: record the tiny stats copy (non-blocking; a
    // pending map skips a frame), fold freshly-landed stats into the win EMA
    // exactly once (object identity), and publish the HUD record.
    // Snapshot into a const: `advRt` is a mutable outer binding, so TS drops
    // its narrowing inside the .some/.map callbacks below (same pattern as the
    // HUD's `const t = advTele`).
    const rtSnap = advRt;
    if (
      advTrainer &&
      rtSnap.tag === "on" &&
      rtSnap.implementation === "fused"
    ) {
      advStatsRec = advTrainer.encodeStatsRead(enc);
      if (advTrainerB) advStatsRecB = advTrainerB.encodeStatsRead(enc);
      speedStats ??= new GpuSpeedStats(device);
      speedRec = speedStats.encodeSample(enc, advect.velBuffer, advect.count, frame);
      const stA = advTrainer.lastStats;
      const stB = advTrainerB?.lastStats ?? null;
      const freshA = !!stA && (stA as object) !== advSeenStats;
      const freshB = !!stB && (stB as object) !== advSeenStatsB;
      const foldWins = (emaWins: number[], counts: readonly number[]) => {
        const seeded = emaWins.some((x) => x > 0);
        for (let j = 0; j < emaWins.length; j++) {
          emaWins[j] = seeded
            ? emaWins[j] + WIN_EMA_LAMBDA * (counts[j] - emaWins[j])
            : counts[j];
        }
      };
      if (freshA) {
        advSeenStats = stA as object;
        foldWins(rtSnap.winEma, stA!.winCounts);
      }
      if (freshB) {
        advSeenStatsB = stB as object;
        foldWins(advWinEmaB, stB!.winCounts);
      }
      // Publish only after every configured branch has produced a real batch.
      // Until then the previous/initial telemetry remains visible; no branch
      // is silently substituted for the missing one.
      if ((freshA || freshB) && stA && (!advTrainerB || stB)) {
        const fractions = (wins: readonly number[]) => {
          const total = wins.reduce((a, x) => a + x, 0) || 1;
          return wins.map((x) => x / total);
        };
        const aFractions = fractions(rtSnap.winEma);
        const bFractions = advTrainerB ? fractions(advWinEmaB) : null;
        // The averaged bars remain a compact aggregate display, but never feed
        // the verdict: averaging can hide a dead A lane behind a healthy B lane.
        const winFractions = bFractions
          ? aFractions.map((x, j) => 0.5 * (x + bFractions[j]))
          : aFractions;
        const kind = rtSnap.kind;
        const twoLane = !!(advTrainerB && stB);
        const branchHealth = (
          branch: AdversaryTrainer,
          stats: NonNullable<typeof stA>,
          branchFractions: readonly number[]
        ): HeadHealth => {
          const winsSkewed = branchFractions.some((f) => f < 0.05 / rtSnap.k);
          const scale = branch.rewardScaleState();
          return classifyHeads(
            winsSkewed,
            stats.headSpread,
            scale.tag === "seeded"
              ? { tag: "seeded", rms: scale.rms }
              : { tag: "unseeded" }
          );
        };
        const healthA = branchHealth(advTrainer, stA, aFractions);
        const healthB =
          twoLane && bFractions
            ? branchHealth(advTrainerB!, stB!, bFractions)
            : null;
        const objectiveSuffix =
          advSpec.tag === "on"
            ? ` · ${adversaryTargetOf(advSpec).tag} · ${adversaryLossOf(advSpec).tag}`
            : "";
        advTele = {
          tag: "on",
          variant: twoLane
            ? kind.tag === "wta"
              ? `agree+disagree k=${kind.k} ε=${kind.relaxEps}${objectiveSuffix}`
              : `agree+disagree single${objectiveSuffix}`
            : kind.tag === "wta"
            ? `wta k=${kind.k} ε=${kind.relaxEps}${objectiveSuffix}`
            : `single${objectiveSuffix}`,
          k: rtSnap.k,
          // Aggregate summaries preserve the existing UI shape. The branch
          // records below are the authoritative values for this general-sum
          // game; predictor objective is V_A + V_B.
          surprise: twoLane ? 0.5 * (stA.surprise + stB!.surprise) : stA.surprise,
          predLoss: twoLane ? stA.payoffUngated + stB!.payoffUngated : stA.payoffUngated,
          // Lane A's, NOT an average: on the two-lane game each lane measures
          // the direction order of its OWN field head, so there is no single
          // blended R₁ in the stats and inventing one would be a lie.
          directionOrder: stA.directionOrder,
          // Lane A's, for the same reason directionOrder is: each lane prices
          // its own field head. Single-lane games (every family piece today)
          // have only lane A anyway.
          perFamily: stA.perFamily,
          payoffReference:
            advSpec.tag === "on"
              ? payoffReferenceOf(adversaryLossOf(advSpec))
              : { tag: "none" },
          winFractions,
          // A/B games are diagnosed independently. A measured failure in one
          // branch cannot be averaged away by the other.
          health: healthB ? combineHeadHealth(healthA, healthB) : healthA,
          weight: rtSnap.weight,
          branches: twoLane
            ? {
                disagree: {
                  surprise: stA.surprise,
                  predLoss: stA.payoffUngated,
                  winFractions: aFractions,
                  health: healthA,
                },
                agree: {
                  surprise: stB!.surprise,
                  predLoss: stB!.payoffUngated,
                  winFractions: bFractions!,
                  health: healthB!,
                },
              }
            : undefined,
        };
      }
    }

    // (2) ADVECT — ONE fused dispatch over ALL particles, recorded into `enc`.
    //     Returns the tfjs weight-sync clones (empty on the fused path); they
    //     must be disposed AFTER submit so their source buffers survive the
    //     in-flight copies.
    const advectRefs = advect.encodeStep(
      enc,
      frame,
      field ? (field as HelmholtzField).alpha : 0,
      tsAdvect
    );

    // (3) RENDER — dots drawn straight from the kernel's particle buffers,
    //     recorded into `enc`.
    let renderMs = 0;
    // Diagnostic colour is orthogonal to ink geometry. The selected surprise
    // plane is sampled for the HUD and handed to the normal point/splat
    // renderer; DOT/VEL/CURL therefore remains active for RAW and PER UNIT.
    if (isSurpriseColorMode(colorMode) && advTrainer && advSurStats) {
      const metric = surpriseMetricOf(colorMode)!;
      const plane = advTrainer.surprisePlane(metric);
      // Use the normalizer's EMA-smoothed P2/P98 span for rendering. The raw
      // sample remains a HUD/debug readout; using it here makes the palette
      // flicker and can resurrect numerical noise as a full rainbow.
      const span = advSurStats.norm.span;
      const scale = 1 / Math.max(span.hi - span.lo, 1e-6);
      const diagnostic = {
        buffer: plane.buffer,
        mode: metric === "raw-payoff" ? ("raw" as const) : ("per-unit" as const),
        offsetFloats: plane.offsetFloats,
        bias: -span.lo,
        scale,
        colormap: colorMode.colormap,
      };
      splat?.setColorDiagnostic(diagnostic);
      renderer?.setColorDiagnostic(diagnostic);
      advSurRec = advSurStats.encodeSample(
        enc,
        plane.buffer,
        advect.count,
        frame,
        advTrainer.surpriseCoverage().window,
        plane.offsetFloats
      );
    } else {
      splat?.setColorDiagnostic(null);
      renderer?.setColorDiagnostic(null);
    }
    if (renderer) {
      const r0 = performance.now();
      const useSplat =
        splat !== null &&
        renderOverride !== "quads" &&
        (renderOverride === "splat" || advect.count >= SPLAT_MIN_N);
      if (useSplat) {
        // AUTO-EXPOSURE: accumulated energy scales with particle density and
        // the trail steady-state 1/(1-decay); normalize so the MEAN displayed
        // energy stays constant across counts (attractor hot-spots still
        // bloom through the tonemap shoulder — that's the aesthetic).
        // The mean is over NATIVE texels — w*h CSS px times dpr² — since each
        // particle's 4096 energy spreads over the dpr-scaled accumulator.
        // `?exposure=F` scales the target.
        splat!.exposure =
          (0.35 * w * h * dpr * dpr * (1 - Math.min(splat!.decay, 0.995))) /
          Math.max(1, advect.count) * exposureScale;
        splat!.encodeRender(
          enc,
          advect.posBuffer,
          advect.velBuffer,
          advect.count,
          w,
          h,
          tsRender
        );
      } else {
        renderer.encodeRender(
          enc,
          advect.posBuffer,
          advect.velBuffer,
          advect.count,
          w,
          h,
          tsRender
        );
      }
      renderMs = performance.now() - r0;
    }

    // Resolve the timestamp query set into a staging buffer within THIS frame's
    // encoder (~every 15 frames), then SINGLE submit for the whole frame.
    if (timer) timer.maybeResolve(enc, frame);
    device.queue.submit([enc.finish()]);
    for (const t of advectRefs) t.dispose();
    if (timer) timer.afterSubmit();
    if (advTrainer) advTrainer.afterSubmit(advStatsRec);
    if (advTrainerB) advTrainerB.afterSubmit(advStatsRecB);
    if (pixelDiscTrainer) pixelDiscTrainer.afterSubmit(pixelDiscStatsRec);
    if (advSurStats) advSurStats.afterSubmit(advSurRec);
    if (speedStats) speedStats.afterSubmit(speedRec);

    const now = performance.now();
    emaFrame = ema(emaFrame, now - lastT);
    lastT = now;
    emaTrain = ema(emaTrain, trainMs);
    emaRender = ema(emaRender, renderMs);
    if (frame % 6 === 0) {
      const head =
        `${cfg.name}\n` +
        `backend ${tf.getBackend()}  render=${particleCount} train=${sampleRate}\n` +
        `FPS     ${(1000 / emaFrame).toFixed(1)}  (${emaFrame.toFixed(1)} ms)\n`;
      // HONEST TIMINGS. advect + render are OUR GPU passes on EVERY piece, so
      // when the profiler is live we always show their real GPU time — never
      // the CPU-encode time (which is ~0.1ms: the cost of RECORDING commands,
      // not the async GPU work). The fused path also has real GPU rollout/optim
      // passes; the legacy tfjs path's "learn" has no clean GPU span (tfjs owns
      // its submits), so it's shown as CPU wall time, explicitly labelled.
      // Without timestamp-query at all, everything is labelled (cpu-encode) so
      // nothing masquerades as a real render time.
      const gt = timer?.timings;
      // readLoss() is asynchronous and the first readback has not completed
      // during pipeline warm-up.  Do not render that absence as "NaN": it
      // resembles a numerical failure even though no loss has been sampled.
      const lossText = Number.isFinite(hudLoss) ? hudLoss.toFixed(3) : "warming";
      let body: string;
      if (gt && trainer) {
        body =
          `rollout ${gt.rollout.toFixed(2)} ms  optim ${gt.optim.toFixed(2)} ms  loss ${lossText}\n` +
          `advect  ${gt.advect.toFixed(2)} ms  render ${gt.render.toFixed(2)} ms  (gpu)\n`;
      } else if (gt) {
        body =
          // Honest label (same policy as the no-timer branch below): the tfjs
          // fallback trains on tf.getBackend() — webgpu here, dispatch-bound —
          // not on the CPU. "(cpu·tfjs)" misled a 2026-08-17 investigation.
          `learn   ${emaTrain.toFixed(1)} ms  (tfjs·${tf.getBackend()})\n` +
          `advect  ${gt.advect.toFixed(2)} ms  render ${gt.render.toFixed(2)} ms  (gpu)\n`;
      } else {
        body =
          `learn   ${emaTrain.toFixed(1)} ms${
            // Honest label: tfjs's actual backend, not an assumed "cpu". The
            // zero-copy renderer forces webgpu (see the IIFE near the bottom),
            // where tiny-op training is dispatch-bound — worth seeing plainly.
            trainer ? `  (fused)  loss ${lossText}` : `  (tfjs·${tf.getBackend()})`
          }\n` + `render  ${emaRender.toFixed(1)} ms  (cpu-encode)\n`;
      }
      tele.textContent =
        head + body + adversaryHudLines() + `tensors ${tf.memory().numTensors}`;
    }

    // ~1 Hz health snapshot. AFTER the HUD block on purpose: the snapshot is
    // built from the same EMAs the HUD just refreshed, so the two can never
    // disagree about a frame.
    if (now - lastHealthMs >= HEALTH_PERIOD_MS) {
      lastHealthMs = now;
      publishHealth(now);
    }

    requestAnimationFrame(tick);
  }

  (async () => {
    // WebGPU backend so tfjs tensors live in GPUBuffers we can render from with
    // zero copy (same GPUDevice). No fallback — warn and bail if it won't init.
    //
    // `?backend=` belonged to the deleted cpu-first initBackend and is REJECTED
    // LOUDLY rather than honored or silently dropped: the renderer draws
    // straight from tfjs-owned GPUBuffers, so a cpu/webgl tfjs backend has
    // nothing it could render from — "supporting" the override would just be a
    // slower way to show the WebGPU warning. The dispatch-cost dial that
    // initBackend's cpu-first ordering used to be is `?handoff=N` below.
    {
      const backendOverride = new URLSearchParams(location.search).get("backend");
      if (backendOverride !== null && backendOverride !== "webgpu") {
        console.warn(
          `[webgpu] ?backend=${backendOverride} ignored — this page is ` +
            `webgpu-only by construction (zero-copy tfjs→render buffers). ` +
            `Use ?handoff=N to move small tfjs ops to the CPU instead.`
        );
      }
    }
    try {
      await tf.setBackend("webgpu");
      await tf.ready();
    } catch (e) {
      console.error("[webgpu] backend init failed", e);
    }
    if (!running) return;
    if (tf.getBackend() !== "webgpu") {
      showWebGPUWarning();
      return;
    }

    // The shared GPUDevice tfjs created — everything (advect/train/render/timer)
    // records onto it. The optional GPU profiler needs the "timestamp-query"
    // feature (main.ts's requestDevice shim appends it when present); when it's
    // absent GpuTimer.create returns null and the HUD keeps its CPU-encode lines.
    device = (tf.backend() as unknown as { device: GPUDevice }).device;
    timer = GpuTimer.create(device);
    console.log(
      timer
        ? "[gputime] timestamp-query active — HUD shows per-pass GPU ms"
        : "[gputime] no timestamp-query feature — HUD uses CPU-encode ms"
    );

    // `?handoff=N`: override tfjs's small-tensor CPU forwarding threshold
    // (WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD; 0 = force every op onto the GPU).
    // Only affects the tfjs learn path (legacy pieces / ?train=tfjs) — pair
    // with the HUD's learn line to A/B it on real hardware.
    const handoff = new URLSearchParams(location.search).get("handoff");
    if (handoff !== null) {
      tf.env().set("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD", parseInt(handoff, 10) || 0);
      console.log(`[tfjs] CPU handoff threshold -> ${parseInt(handoff, 10) || 0}`);
    }

    // Architecture resolution: declarative fieldArch (piece default or dock
    // override) wins; createField is for load-bearing game recipes that bake
    // semantics the dock must not overwrite. Legacy createModel is last resort.
    const archOverride = options.overrides?.fieldArch;
    const resolvedArch = archOverride
      ? archOverride
      : cfg.fieldArch
      ? cfg.fieldArch
      : null;
    field = resolvedArch
      ? createFieldFromArch(resolvedArch)
      : cfg.createField
      ? cfg.createField()
      : null;
    model = !field && cfg.createModel ? cfg.createModel() : null;
    varList = field ? field.trainableWeights : undefined;
    wh = tf.tensor2d([[w, h]]);
    tf.keep(wh);

    // Fused advect kernel: WGSL is GENERATED from the live model's layer dims
    // (see advect_wgsl.ts) — works for both the field and legacy MLP pieces.
    // Owns pos/vel as raw GPUBuffers; construction throws loudly on any
    // unsupported architecture instead of silently falling back.
    const physics = {
      width: w,
      height: h,
      forceMagnitude,
      friction: cfg.friction,
      maxVelocity,
      resetRate: cfg.resetRate,
      border,
    };
    advect = field
      ? AdvectKernel.fromField(field as HelmholtzField, physics, particleCount)
      : AdvectKernel.fromModel(model!, physics, particleCount);

    // Health probe on the advect kernel's OWN weights buffer — the one buffer
    // that is current on both trainer paths (born there when fused, synced
    // every frame by advect.encodeStep when tfjs). Diagnostics only: separate
    // encoder, separate buffers, ~1 Hz. See src/health.ts.
    fieldProbe = new FieldProbe(
      device!,
      advect.layout,
      advect.weightsBuffer,
      HEALTH_GRID_N
    );

    // Field pieces train FUSED by default: the trainer co-owns the advect
    // kernel's weights buffer and Adam-updates it in place — weights never
    // leave the GPU. tfjs remains only as the (idle) blueprint. Legacy MLP
    // pieces keep the tfjs optimizer (their losses aren't in the kernel yet).
    let wantTfjsTrainer =
      new URLSearchParams(location.search).get("train") === "tfjs";
    const agreeDisagreeGame =
      advSpec.tag === "on" && advSpec.game === "agree-disagree";
    const fieldClasses = field ? (field as HelmholtzField).classes ?? 0 : 0;
    const renderClasses =
      field instanceof HelmholtzField && field.semantic === "agree-disagree"
        ? 3
        : fieldClasses;
    const renderPalette =
      cfg.palette ??
      (field instanceof HelmholtzField && field.semantic === "agree-disagree"
        ? "rgb-roles"
        : renderClasses > 0
        ? "species"
        : "speed");
    if (wantTfjsTrainer && fieldClasses > 0) {
      console.warn(
        "[train] ?train=tfjs ignored: class-aware fields are fused-only " +
          "(tfjs has no class input)"
      );
      wantTfjsTrainer = false;
    }
    if (wantTfjsTrainer && agreeDisagreeGame) {
      console.warn(
        "[train] ?train=tfjs ignored for Agree+Disagree: the correct game " +
          "requires two lane-isolated predictors and two external-gradient " +
          "buffers, which are implemented only by the fused trainer"
      );
      wantTfjsTrainer = false;
    }
    // The THIRD `?train=tfjs` consequence — it also disables a piece's pixel
    // critic, since the density GAN exists only in WGSL — is named by
    // resolvePixelCritic below and warned at the pixel gate, so the reason
    // ladder stays in one place. It is NOT ignored like the two above: the
    // tfjs autograd path remains selectable for A/B comparison, it just runs
    // a Pixel piece with nothing driving its field.
    // ADVERSARY PIECES NOW TRAIN FUSED when the field supports it (raw or
    // fourier encoding, classless): AdversaryTrainer records the K-head
    // relaxed-WTA discriminator + the generator reward (extGrads → the field
    // trainer's pass B) into the same frame encoder. Verified vs the AD-IR
    // oracle AND tf.variableGrads at cos = 1.0000000 (tools/train_wta_test.ts);
    // measured 0.7-0.8 ms/step at B=512 vs the tfjs path's 19-32 ms.
    // HASHGRID IS NOW FUSED TOO (the default piece): the adversary scratch
    // carries a per-member dL/dEnc block and fieldGrad carries the gather-side
    // grid block, gated vs live tfjs autograd in tools/train_wta_hashgrid_test.ts.
    // Class-aware fields and ?train=tfjs keep the tfjs autograd path, and the
    // loop says which one it picked out loud.
    // FAMILY-CONDITIONED FIELDS ARE NOT "class-aware fields" for this gate.
    // The blocker was never the family label — it was the ONE-HOT ROUTE, whose
    // extra layer-0 rows the adversary's field backward has no counterpart for.
    // A family-PLANED hashgrid moves the label into the grid's cell index, so
    // it rides the dEnc machinery the reward already uses and the fused game is
    // exact. `familyRoute` (advect_wgsl) is the single place that distinction
    // is made; this reads it rather than re-deriving it from `classes`.
    const familyRouteTag = advect.layout.family.tag;
    const adversaryDisabled =
      advSpec.tag === "on" && familyRouteTag === "onehot";
    // THE ANTI-COLLAPSE PRESSURE IS NOW FUSED (2026-08-17). It compiles into
    // adversary pass A as a compile-time variant and rides the generator's own
    // dSig seam, so it reaches every encoding through the machinery the reward
    // already uses. The tfjs `directionOrderLoss` stays as the parity oracle,
    // reachable with ?train=tfjs. The clause that used to force a declared
    // pressure onto the tfjs trainer is therefore gone — what must never
    // happen is a fused kernel silently DROPPING a declared term, and the
    // codegen now carries it (gated in tools/train_wta_pressure_test.ts).
    const fusedAdvOk =
      advSpec.tag === "on" &&
      !adversaryDisabled &&
      !!field &&
      !wantTfjsTrainer &&
      familyRouteTag !== "onehot";
    const agreeDisagreeField = advect.layout.spec.kind === "agree-disagree";
    if (
      (agreeDisagreeGame && !agreeDisagreeField) ||
      (agreeDisagreeField && advSpec.tag === "on" && !agreeDisagreeGame)
    ) {
      throw new Error(
        "[adversary] Agree+Disagree game/field mismatch: the game requires " +
          "the two-head agree-disagree field semantic, and that field semantic " +
          "must be trained by the two-lane game"
      );
    }
    if (adversaryDisabled) {
      console.warn(
        "[adversary] disabled for ONE-HOT class-conditioned fields: the extra " +
          "layer-0 rows have no counterpart in the fused field backward. A " +
          "family-PLANED hashgrid (FamilyRoute 'grid-plane') is supported and " +
          "is what the RGB Families piece uses."
      );
      advRt = { tag: "off" };
    } else if (fusedAdvOk) {
      advRt = createAdversary(
        advSpec,
        sampleRate,
        observerGeometry,
        discriminatorLearningRate,
        "fused"
      );
    } else {
      advRt = createAdversary(
        advSpec,
        sampleRate,
        observerGeometry,
        discriminatorLearningRate
      );
    }
    if (advRt.tag === "on" && advRt.implementation === "tfjs") {
      console.log(
        `[adversary] ${advSpec.tag === "on" ? advSpec.kind.tag : "?"} ` +
          `encoding=${advSpec.tag === "on" ? advSpec.encoding.tag : "?"} ` +
          `k=${advRt.k} weight=${advRt.weight} pressure=${advRt.pressure.tag} ` +
          `— tfjs autograd trainer (${
            wantTfjsTrainer
              ? "?train=tfjs"
              : "field type unsupported by the fused adversary"
          })`
      );
    }
    // PIXEL CRITIC ROUTING — resolved HERE, with the rest of the trainer gate,
    // and never re-decided at the construction site below. Every branch that
    // can silence a declared critic is a named state with a reason, because a
    // silently-dropped critic on a ZERO_FIELD_LOSS Pixel piece leaves the field
    // with no gradient at all while the gallery and HUD still advertise a GAN.
    // The Agree+Disagree combination throws inside the resolver — see its doc.
    const pixelCritic = resolvePixelCritic({
      declared: cfg.pixelDisc,
      hasField: !!field,
      fieldLossDeclared: cfg.fieldLoss !== undefined,
      wantTfjsTrainer,
      adversaryOnTfjs: advRt.tag === "on" && !fusedAdvOk,
      agreeDisagreeGame,
      layout: advect.layout,
    });
    if (pixelCritic.tag === "dropped") {
      console.warn(
        `[pixel-disc] this piece's declared pixel critic is OFF: ${pixelCritic.reason}`
      );
    }
    // ALL field types (standard/siren/fourier/hashgrid) now train FUSED: the
    // trainer codegen handles sin backward (pre-act checkpoint), the fourier
    // encoding jacobian, and the hashgrid interp/scatter — each verified vs a
    // tfjs-autograd fixture on Metal at cos=1.0 (tools/train_types_test.ts).
    // `?train=tfjs` keeps the autograd path selectable for A/B comparison.
    // Field pieces with an explicit fieldLoss train FUSED by default
    // (chaos / cover / … compiled into train_wgsl). Aesthetic pieces that
    // omit fieldLoss keep the tfjs computeLoss path.
    if (
      field &&
      cfg.fieldLoss !== undefined &&
      !wantTfjsTrainer &&
      (advRt.tag === "off" || fusedAdvOk)
    ) {
      // `?rollout=K` (1..16, default 1): K-step BPTT rollout — the loss sees
      // how particles FLOW through the field (evolving pos+vel), not just one
      // step. K is compiled into the trainer's WGSL. K=1 ≡ the tfjs loss.
      // `?window=K` overrides this (and trainEvery) — see windowK above.
      const rollout =
        windowK > 0
          ? windowK
          : Math.max(
              1,
              Math.min(
                16,
                parseInt(new URLSearchParams(location.search).get("rollout") ?? "1", 10) || 1
              )
            );
      // Training states come from the LIVE PARTICLE CLOUD by default: real
      // positions AND velocities, denser where the attractors are (that's
      // where the art lives). Coverage/exploration is the reset slider's job —
      // resets continuously inject fresh uniform vel-0 states into the cloud,
      // hence into the batch. `?batch=random` restores the old uniform source.
      if (new URLSearchParams(location.search).get("batch") === "random") {
        trainSource = "random";
      }
      // `?mix=F` (0..1): coverage floor — fraction of the particle-sourced
      // batch replaced by fresh uniform random points each step.
      mixRandom = Math.max(
        0,
        Math.min(
          1,
          parseFloat(new URLSearchParams(location.search).get("mix") ?? "0") || 0
        )
      );
      // Fused adversary FIRST: the field trainer's pass B needs its extGrads
      // buffer at construction (a codegen flag, not a runtime branch).
      if (
        fusedAdvOk &&
        advRt.tag === "on" &&
        advRt.implementation === "fused" &&
        advSpec.tag === "on"
      ) {
        const commonAdversaryOpts = {
          tag: advSpec.encoding.tag,
          target: adversaryTargetOf(advSpec),
          loss: adversaryLossOf(advSpec),
          // The declared counter-pressure is compiled into pass A. Structural
          // identity with FusedGamePressure is deliberate and mirrors how
          // target/loss cross this boundary.
          pressure: gamePressureOf(advSpec),
          k: headCount(advSpec.kind),
          relaxEps: advSpec.kind.tag === "wta" ? advSpec.kind.relaxEps : 0,
          // Resolved, never left to the trainer's own `?? 32` / `?? 16`: the
          // tfjs oracle has a second copy of those fallbacks.
          hiddenUnits: predictorArchOf(advSpec).hiddenUnits,
          featureDim: predictorArchOf(advSpec).featureDim,
          batchCap: 1024,
          fieldWeightsBuffer: advect.weightsBuffer,
          particleCount: advect.count,
          observerGeometry,
        } as const;
        advTrainer = new AdversaryTrainer(device!, advect.layout, {
          ...commonAdversaryOpts,
          // Ordinary pieces retain the historical blended-field game. In
          // Agree+Disagree this is A: direct field head 0, predictor opponent.
          fieldLane: agreeDisagreeGame ? 0 : "blend",
          generatorRole: "disagree",
          seed: 20260727,
        });
        if (agreeDisagreeGame) {
          // B owns a separate predictor, optimizer, scratch and statistics.
          // The only shared object is the READ-ONLY field weights buffer.
          advTrainerB = new AdversaryTrainer(device!, advect.layout, {
            ...commonAdversaryOpts,
            fieldLane: 1,
            generatorRole: "agree",
            seed: 20260728,
          });
          advWinEmaB = new Array(headCount(advSpec.kind)).fill(0);
        }
        advTrainer.setParticleBuffers(advect.posBuffer, advect.velBuffer, advect.count);
        advTrainerB?.setParticleBuffers(
          advect.posBuffer,
          advect.velBuffer,
          advect.count
        );
        advSurStats = new GpuSurpriseStats(device!);
        console.log(
          agreeDisagreeGame
            ? `[adversary] FUSED Agree+Disagree encoding=${advSpec.encoding.tag} ` +
                `k=${headCount(advSpec.kind)} weight=${advRt.weight} — ` +
                `A=lane0/disagree, B=lane1/agree, C=display-only blend; ` +
                `two independent predictors, one summed field update`
            : `[adversary] FUSED ${advSpec.kind.tag} encoding=${advSpec.encoding.tag} ` +
                `k=${headCount(advSpec.kind)} weight=${advRt.weight} ` +
                `predictor=${predictorArchOf(advSpec).hiddenUnits}/` +
                `${predictorArchOf(advSpec).featureDim} ` +
                `pressure=${describePressure(gamePressureOf(advSpec))} — disc train + ` +
                `generator reward in-frame, tfjs idle`
        );
      }
      // One decision, made once (resolvePixelCritic, above) — this site only
      // executes it. The gate that used to live here re-derived the same field
      // conditions in a second, silently-skipping form.
      if (pixelCritic.tag === "fused") {
        const pixelSpec = pixelCritic.spec;
        pixelDiscTrainer = new PixelDiscTrainer(device!, advect.layout, {
          fieldWeightsBuffer: advect.weightsBuffer,
          dims: {
            kind: pixelSpec.kind ?? "vec-field",
            G: pixelSpec.G,
            E: pixelSpec.E,
            K: pixelSpec.K,
            hidden: pixelSpec.hidden,
            dt: pixelSpec.dt,
            // Forwarded UNRESOLVED: `resolvePixelDims` owns the default and
            // `validatePixelDims` owns the (kind, guesses) refusal. Dropping
            // this line is how the fused multi-guess head shipped unreachable —
            // every backend supported K guesses and no piece could ask for them.
            guesses: pixelSpec.guesses,
          },
          batchCap: 512,
          seed: 20260805,
          historicalReplay: pixelSpec.historicalReplay,
        });
        pixelDiscTrainer.setParticleBuffers(
          advect.posBuffer,
          advect.velBuffer,
          advect.count
        );
        console.log(
          `[pixel-disc] FUSED kind=${pixelDiscTrainer.kind} ` +
            `G=${pixelDiscTrainer.dims.G} E=${pixelDiscTrainer.dims.E} ` +
            `K=${pixelDiscTrainer.dims.K} weight=${pixelSpec.weight} — ` +
            `soft density critic + reverse-mode gen path (no JVP)`
        );
      }
      // Tripwire: the plan above and this block's enclosing `if` are two
      // expressions of the same routing decision, and drift between them is
      // exactly the failure mode being fixed — a critic that the HUD says is on
      // and that was never built. Fail at startup instead of at frame 1.
      if (pixelCritic.tag === "fused" && !pixelDiscTrainer) {
        throw new Error(
          "[pixel-disc] resolved FUSED but no critic was constructed — the " +
            "trainer gate and resolvePixelCritic have drifted"
        );
      }
      // EVERY claimant, listed once. The old form was
      // `if (advTrainerB) … else if (pixelDiscTrainer) …`, whose `else` bound to
      // advTrainerB and silently dropped a fully-constructed pixel critic, and
      // whose count guard below could therefore never fire. Pass B sums at most
      // two (train_wgsl `extGradCount ∈ [0,2]`); resolvePixelCritic refuses the
      // one three-claimant combination up front, and this stays as the
      // structural backstop for the next game that wants a slot.
      const extGradClaims: { name: string; buffer: GPUBuffer }[] = [];
      if (advTrainer) {
        extGradClaims.push({ name: "adversary", buffer: advTrainer.extGradsBuf });
      }
      if (advTrainerB) {
        extGradClaims.push({
          name: "adversary lane B (agree)",
          buffer: advTrainerB.extGradsBuf,
        });
      }
      if (pixelDiscTrainer) {
        extGradClaims.push({
          name: "pixel critic",
          buffer: pixelDiscTrainer.extGradsBuf,
        });
      }
      if (extGradClaims.length > 2) {
        throw new Error(
          `[train] at most 2 extGrad buffers, got ${extGradClaims.length} ` +
            `(${extGradClaims.map((c) => c.name).join(", ")}); a third claimant ` +
            `needs a third binding in trainPassBShader, not a dropped gradient`
        );
      }
      const extGradBuffers = extGradClaims.map((c) => c.buffer);
      trainer = new FusedTrainer(device!, advect.layout, {
        // Ask for the architectural ceiling and let FusedTrainer resolve it
        // against this device+layout+K. The old hard-coded 1024 was invisible
        // to the UI, so the "train B" slider (max 4096) could hand the trainer
        // an n it rejects — which throws inside the rAF tick and stops the
        // animation for good.
        batchCap: MAX_BATCH,
        weightsBuffer: advect.weightsBuffer,
        kSteps: rollout,
        extGradBuffers,
        loss: cfg.fieldLoss,
        border,
      });
      trainer.uploadWeights(advect.packCurrentWeights());
      trainer.setParticleBuffers(advect.posBuffer, advect.velBuffer, advect.count);
      advect.syncFromTfjs = false;
      // Second κ site: a live sampleRate carried across a gallery switch may
      // exceed the NEW trainer's cap (different layout/K ⇒ different scratch
      // cost). Re-canonicalize here so the invariant in record() holds.
      sampleRate = resolveTrainBatchSize(sampleRate, trainer.batchCap).n;
      console.log(
        `[train] fused trainer active (2 dispatches/step, tfjs idle, ` +
          `batch=${trainSource}, rollout=${rollout}, ` +
          `batchCap=${trainer.batchCap})`
      );
    }
    // Construct the tfjs generator optimizer only for an actual tfjs training
    // path. Fused startup now retains neither this unused reference optimizer
    // nor the former K-dependent tfjs predictor/Adam graph.
    if (!trainer) optimizer = tf.train.adam(generatorLearningRate);

    if (isSurpriseColorMode(colorMode) && !advTrainer) {
      console.warn(
        "[adversary] tfjs is an oracle-only training path; raw/per-unit cloud " +
          "diagnostics require the fused adversary. Falling back to velocity colour."
      );
      colorMode = { tag: "velocity" };
    }

    try {
      renderer = new GpuPointRendererWebGPU(canvas, {
        pointSize: (cfg as { pointSize?: number }).pointSize ?? 2.5,
        background: cfg.backgroundColor,
        maxSpeed: maxVelocity,
        classes: renderClasses,
        palette: renderPalette,
      });
      // splat shares the canvas context with the quad renderer (same device/
      // format); its passes only run on frames where it's picked.
      const decayParam = new URLSearchParams(location.search).get("decay");
      const dotParam = new URLSearchParams(location.search).get("dot");
      splat = new SplatRenderer(canvas, {
        background: cfg.backgroundColor,
        maxSpeed: maxVelocity,
        classes: renderClasses,
        palette: renderPalette,
        dpr,
        // `?dot=F` — radial splat radius in CSS px (default 1.25)
        radius: dotParam !== null ? parseFloat(dotParam) || 1.25 : 1.25,
        // Trails come from the piece's declared renderer style (the splat's
        // decay is the real mechanism behind all of them); `?decay=F` overrides.
        decay:
          decayParam !== null
            ? Math.max(0, Math.min(0.995, parseFloat(decayParam) || 0))
            : SPLAT_DECAY_BY_RENDERER[cfg.renderer],
        // `?stroke=` / `?strokeLen=` — geometric stroke trails (parsed above)
        style: strokeStyle,
        strokeLen,
      });
    } catch (e) {
      console.error("[webgpu] renderer init failed", e);
      showWebGPUWarning();
      return;
    }

    // --- Fused surprise renderer -------------------------------------------
    // POINT SIZE AND GAIN. The surprise renderer draws additively and does NOT
    // have the splat path's auto-exposure (deliberately: an exposure that tracks
    // density would rescale the very quantity the colour is reporting). At the
    // adversary pieces' 40-120k particles a 2 px unit-gain dot renders as a
    // near-black field — measured in browser QA, the whole frame read as
    // collapsed when it was not. These are display constants, chosen at those
    // counts; they scale brightness only, never the value → ramp mapping, so
    // the picture stays auditable against the printed p2/p98.
    const SURPRISE_POINT = 2.5;
    // 1.5, not 3.0. At 3.0 the additive blend saturates and the whole field
    // renders WHITE — legible, and completely uninformative, because the colour
    // carrying the residual is clipped away. Verified by screenshot at both
    // values. Brightness must stop below the point where the ramp stops being
    // readable, or the mode is decoration rather than an instrument.
    const SURPRISE_GAIN = 1.5;

    /** Lazily build (and cache) the surprise renderer for a colormap. Cached
     *  rather than rebuilt because construction reconfigures the canvas context,
     *  and destroying one would `unconfigure()` the context the splat and quad
     *  renderers are still using. Three ramps ⇒ at most three pipelines. */
    const surpriseRendererFor = (name: ColormapName): GpuSurpriseRendererWebGPU => {
      const have = surpriseRenderers.get(name);
      if (have) return have;
      const made = new GpuSurpriseRendererWebGPU(canvas, {
        colormap: name,
        background: cfg.backgroundColor,
        pointSize: SURPRISE_POINT,
        gain: SURPRISE_GAIN,
      });
      surpriseRenderers.set(name, made);
      return made;
    };
    if (isSurpriseColorMode(colorMode)) surpriseRendererFor(colorMode.colormap);

    // Live controls for the UI: particle count (resizes kernel buffers,
    // preserving state — grow appends, shrink slices), sample rate, trails.
    // Fired AFTER renderer/splat construction so getDecay() reads the live
    // splat default (the trails slider initializes from it).
    if (onReady) {
      onReady({
        field: field ? (field as HelmholtzField) : null,
        getParticleCount: () => particleCount,
        setParticleCount: (n: number) => {
          if (!advect) return;
          advect.setParticleCount(n);
          particleCount = advect.count;
          // resize replaces the pos/vel buffers — refresh the trainers' views
          if (trainer) {
            trainer.setParticleBuffers(advect.posBuffer, advect.velBuffer, advect.count);
          }
          if (advTrainer) {
            // may also grow the surprise buffer; the renderer re-binds by
            // buffer identity on the next encodeRender, so no extra plumbing
            advTrainer.setParticleBuffers(advect.posBuffer, advect.velBuffer, advect.count);
          }
          if (advTrainerB) {
            advTrainerB.setParticleBuffers(
              advect.posBuffer,
              advect.velBuffer,
              advect.count
            );
          }
          advSurStats?.reset();
        },
        getResetRate: () => resetRate,
        setResetRate: (r: number) => {
          resetRate = Math.max(0, Math.min(0.2, r));
          if (advect) advect.setResetRate(resetRate);
        },
        getSampleRate: () => sampleRate,
        getMaxSampleRate: () => trainer?.batchCap ?? TRAIN_BATCH_MAX,
        setSampleRate: (n: number) => {
          const r = resolveTrainBatchSize(n, trainer?.batchCap ?? TRAIN_BATCH_MAX);
          if (r.tag === "clamped" && r.requested !== warnedSampleRate) {
            // Once per distinct over-cap value — a slider drag would otherwise
            // emit one warning per pointer event.
            warnedSampleRate = r.requested;
            console.warn(
              `[train] batch ${r.requested} exceeds this trainer's cap ${r.cap} ` +
                `(device/layout limited); training at ${r.n}`
            );
          }
          sampleRate = r.n;
        },
        getDecay: () => splat!.decay,
        setDecay: (d: number) => {
          if (splat) splat.decay = Math.max(0, Math.min(0.99, d));
        },
        getMaxVelocity: () => maxVelocity,
        setMaxVelocity: (v: number) => {
          maxVelocity = Math.max(0.25, Math.min(200, v));
          advect?.setMaxVelocity(maxVelocity);
          // Drive is a FRACTION of the live clip. Preserve that physical
          // meaning when maxVelocity changes instead of silently increasing or
          // weakening the field relative to the clip.
          if (driveEnabled) {
            forceMagnitude = forceMagnitudeForDrive(drive, maxVelocity, cfg.friction);
            advect?.setForceMagnitude(forceMagnitude);
          }
          renderer?.setMaxSpeed(maxVelocity);
          if (splat) splat.maxSpeed = maxVelocity;
        },
        getDrive: () => drive,
        setDrive: (v: number) => {
          // Non-adversary presets did not opt into drive semantics; their
          // literal historical forceMagnitude remains untouched.
          if (!driveEnabled) return;
          drive = Math.max(0, Math.min(1, v));
          forceMagnitude = forceMagnitudeForDrive(drive, maxVelocity, cfg.friction);
          advect?.setForceMagnitude(forceMagnitude);
        },
        getGeneratorLearningRate: () => generatorLearningRate,
        setGeneratorLearningRate: (v: number) => {
          const next = Math.max(
            GAME_LEARNING_RATE_RANGE.generator.min,
            Math.min(GAME_LEARNING_RATE_RANGE.generator.max, v)
          );
          if (next === generatorLearningRate) return;
          generatorLearningRate = next;
          // Fused Adam reads the live uniform on every encodeStep and keeps its
          // moments. The tfjs optimizer has no public LR setter; rebuilding it
          // is the safe supported path and intentionally resets only Adam
          // moments, never field weights.
          if (!trainer && optimizer) {
            optimizer.dispose();
            optimizer = tf.train.adam(generatorLearningRate);
          }
        },
        getDiscriminatorLearningRate: () => discriminatorLearningRate,
        setDiscriminatorLearningRate: (v: number) => {
          const next = Math.max(
            GAME_LEARNING_RATE_RANGE.discriminator.min,
            Math.min(GAME_LEARNING_RATE_RANGE.discriminator.max, v)
          );
          if (next === discriminatorLearningRate) return;
          discriminatorLearningRate = next;
          // Fused predictors read this variable through their step uniform.
          // The tfjs fallback rebuilds only predictor optimizer state.
          if (advRt.tag === "on" && advRt.implementation === "tfjs") {
            advRt.adv.setLearningRate(discriminatorLearningRate);
          }
        },
        getBlend: () => (field instanceof HelmholtzField ? field.alpha : 0),
        setBlend: (v: number) => {
          if (!(field instanceof HelmholtzField)) return;
          // Single-head arches keep α=0 (no second head to blend).
          field.alpha =
            field.headCount === 1 ? 0 : Math.max(0, Math.min(1, v));
        },
        getStrokeStyle: () => strokeStyle,
        setStrokeStyle: (v: SplatStyle) => {
          strokeStyle = v;
          if (splat) splat.style = v;
        },
        getStrokeLength: () => strokeLen,
        setStrokeLength: (v: number) => {
          strokeLen = Math.max(
            STROKE_LENGTH_RANGE.min,
            Math.min(STROKE_LENGTH_RANGE.max, v)
          );
          if (splat) splat.strokeLen = strokeLen;
        },
        getAdversaryWeight: () => (advRt.tag === "on" ? advRt.weight : 0),
        setAdversaryWeight: (x: number) => {
          // Generator role owns the sign. A negative UI weight is never used
          // as a hidden "agree" switch (Agree+Disagree has a named B role).
          if (advRt.tag === "on") advRt.weight = Math.max(0, Math.min(20, x));
        },
        getPixelCriticWeight: () => pixelGenWeight,
        setPixelCriticWeight: (x: number) => {
          // Magnitude only — `encodeStep` owns the sign (`L_gen = -|w|·R`), so a
          // negative here would be a second, hidden place that decides whether
          // the generator agrees or disagrees with the critic.
          if (pixelDiscTrainer) {
            pixelGenWeight = Math.max(
              PIXEL_CRITIC_WEIGHT_RANGE.min,
              Math.min(PIXEL_CRITIC_WEIGHT_RANGE.max, x)
            );
          }
        },
        getAdversaryTelemetry: () => advTele,
        getColorMode: () => colorMode,
        setColorMode: (m: ColorMode) => {
          if (isSurpriseColorMode(m) && cfg.mode === "agree-disagree") {
            console.warn(
              "[adversary] surprise colour is disabled for Agree+Disagree; " +
                "the RGB channels are its A/B/C output contract."
            );
            return;
          }
          if (isSurpriseColorMode(m) && !advTrainer) {
            console.warn(
              "[adversary] tfjs is an oracle-only training path; raw/per-unit cloud " +
                "diagnostics require the fused adversary."
            );
            return;
          }
          const previousMetric = surpriseMetricOf(colorMode);
          const nextMetric = surpriseMetricOf(m);
          if (isSurpriseColorMode(m)) surpriseRendererFor(m.colormap);
          if (previousMetric !== nextMetric) advSurStats?.reset();
          colorMode = m;
        },
        getSurpriseSpan: () =>
          advTrainer && advSurStats && isSurpriseColorMode(colorMode)
            ? {
                ...advSurStats.norm.raw,
                covered: Math.min(
                  advTrainer.surpriseCoverage().covered,
                  advTrainerB?.surpriseCoverage().covered ?? 1
                ),
                collapsed: advSurStats.norm.collapsed,
              }
            : null,
        getFieldHealth: () => fieldHealth,
      });
    }
    console.log(`starting: ${cfg.name} (webgpu)`);
    // `tick` is async and reschedules itself on its LAST line, so a throw
    // anywhere inside it stops the loop for good — and as a bare `tick()` the
    // rejection was silent, which cost a 2026-08-17 investigation a lot of time
    // chasing a "frozen" canvas. Surface it instead.
    tick().catch((e) => {
      console.error("[loop] tick() rejected — the render loop has stopped", e);
    });
  })();

  return () => {
    running = false;
    if (renderer) renderer.destroy();
    if (splat) splat.destroy?.();
    if (timer) timer.destroy(); // querySet + resolve/staging GPUBuffers
    tele.remove();
    if (optimizer) optimizer.dispose(); // frees Adam accumulators (leaked on tab switch)
    if (wh) wh.dispose();
    if (trainer) trainer.destroy(); // batch/scratch/grads/adam GPUBuffers
    if (advTrainer) advTrainer.destroy(); // adv weights/adam/scratch/stats/surprise/extGrads
    if (advTrainerB) advTrainerB.destroy(); // independent B predictor + extGrad
    if (pixelDiscTrainer) pixelDiscTrainer.destroy();
    if (advSurStats) advSurStats.destroy();
    if (speedStats) speedStats.destroy();
    if (fieldProbe) fieldProbe.destroy(); // diagnostics points/out/staging
    // The snapshot describes a loop that no longer exists; leaving it would let
    // a headless auditor gate a piece switch on the PREVIOUS piece's numbers.
    delete (window as unknown as HealthWindow).__nffHealth;
    if (advect) advect.destroy(); // pos/vel/weights GPUBuffers
    if (model) model.dispose();
    if (field) field.dispose();
    // Only the explicit tfjs oracle/fallback owns tensor predictors. Fused
    // gallery runs never construct them; their WGSL trainer was destroyed above.
    if (advRt.tag === "on" && advRt.implementation === "tfjs") {
      advRt.adv.dispose();
    }
    for (const r of surpriseRenderers.values()) r.destroy();
    surpriseRenderers.clear();
  };
}
