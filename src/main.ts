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
import { isotropyLoss } from "./core/losses";
// OPTIONAL zero-copy GPU renderer (perf lane). Imported so it compiles and is
// ready, but only used when a preset sets `gpu: true` (none do by default — it
// needs browser QA). See src/render/gpuPoints.ts.
import { GpuPointRenderer } from "./render/gpuPoints";
import { GpuPointRendererWebGPU } from "./render/webgpu/points";
import { SplatRenderer, SplatStyle } from "./render/webgpu/splat";
import { AdvectKernel } from "./render/webgpu/advect";
import type { BorderMode } from "./render/webgpu/advect_wgsl";
import { FusedTrainer } from "./render/webgpu/train";
import type { FieldLossSpec } from "./render/webgpu/train_wgsl";
// FUSED ADVERSARY (adversary_train.ts + adversary_wgsl.ts): the WGSL port of
// the tfjs adversary below — discriminator train + generator reward + the
// packed raw/per-unit surprise buffer, ~0.7-0.8 ms/step at B=512 vs tfjs's
// 19-32 ms.
// Oracle-gated by tools/train_wta_test.ts (cos = 1.0000000 vs the AD-IR
// oracle AND tf.variableGrads). Raw/fourier classless fields only; anything
// else falls back to the tfjs path below, loudly.
import {
  AdversaryTrainer,
  type SurpriseMetric,
} from "./render/webgpu/adversary_train";
import { PixelDiscTrainer } from "./render/webgpu/pixel_disc_train";
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
    };

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

export interface ArtPieceConfig {
  name: string;
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
  drawRate: number;
  /**
   * Field/generator Adam learning rate. On a standalone adversary piece the
   * external game gradient is the only field gradient, so Adam largely removes
   * a scalar reward-weight rescale; this LR is the effective generator-step
   * control. The predictor/discriminator LR defaults separately to 3e-3; both
   * are live controls because their ratio changes the game, not just its speed.
   */
  learningRate: number;
  backgroundColor: [number, number, number];
  alphaBlend: number;
  renderer: RendererType;
  /** Explicit particle palette; Agree+Disagree uses stable exact RGB roles. */
  palette?: "speed" | "species" | "rgb-roles";
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
  pixelDisc?: {
    readonly weight: number;
    /** One of the four Pixel GAN games — see docs/PIXEL_DISC.md. */
    readonly kind?:
      | "vec-field"
      | "next-frame"
      | "real-fake"
      | "inpaint";
    readonly G?: number;
    readonly E?: number;
    readonly K?: number;
    readonly hidden?: number;
    readonly dt?: number;
  };
  /**
   * Colormap for `renderer: "surprise"` pieces. Absent ≡ "inferno"
   * (canonicalized in {@link resolveColorMode}); ignored by every other
   * renderer, which colours by velocity.
   */
  colormap?: ColormapName;
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

export const GAME_LEARNING_RATE_RANGE = {
  generator: { min: 1e-5, max: 1e-2 },
  discriminator: { min: 1e-4, max: 3e-2 },
} as const;

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
        floatParam(q, "dLR", 3e-3)
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
  if (mode === "single") {
    return {
      tag: "on",
      kind: { tag: "single" },
      encoding,
      target,
      loss,
      weight,
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
      ...gamePart,
    };
  }
  if (mode !== null) {
    throw new Error(`?adv must be off|single|wta, got "${mode}"`);
  }
  // No `?adv=`: keep the piece's variant, apply the knob overrides to it.
  const kind: AdversaryKind =
    base.kind.tag === "single" ? { tag: "single" } : { tag: "wta", k, relaxEps: eps };
  return { tag: "on", kind, encoding, target, loss, weight, ...gamePart };
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
    (want === null && cfg.renderer === "surprise" && !perUnit);
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
  if (W_CHAOS === 0 && W_ISO === 0 && W_DIV === 0 && W_SPIRAL === 0) {
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

      return chaos
        .mul(W_CHAOS)
        .add(iso.mul(W_ISO))
        .add(div.mul(W_DIV))
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

/** What the HUD and the React panel read. */
export type AdversaryTelemetry =
  | { readonly tag: "off" }
  | {
      readonly tag: "on";
      readonly variant: string;
      readonly k: number;
      /** Mean relaxed-WTA payoff on the most recent training batch. */
      readonly surprise: number;
      /** Predictor (discriminator) loss on the most recent batch. */
      readonly predLoss: number;
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
        return term;
      });
    }
    default:
      return assertNeverPiece(rt, "adversaryGeneratorTerm");
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
    archDock: "dual",
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
    createField: () => createFieldFromArch({ ...ARCH.dualStd, alpha: 0.7 }),
    adversary: {
      tag: "on",
      kind: { tag: "single" },
      encoding: { tag: "point" },
      // REWARD UNITS CHANGED 2026-07-27: generatorLoss is now normalized by an
      // EMA of RMS‖y‖ — dimensionless residual per unit field signal.
      weight: 0.01,
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
    createField: () => createFieldFromArch({ ...ARCH.dualStd, alpha: 0.62 }),
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 8, relaxEps: 0.05 },
      encoding: { tag: "point" },
      weight: 0.012, // reward units — see the note on the Single piece
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
    createField: () => createFieldFromArch({ ...ARCH.dualStd, alpha: 0.55 }),
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
    createField: () => createFieldFromArch({ ...ARCH.dualStd, alpha: 0.55 }),
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 6, relaxEps: 0.05 },
      encoding: { tag: "tri" },
      weight: 0.012, // reward units — see the note on the Single piece
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
    createField: () => createFieldFromArch({ ...ARCH.dualStd, alpha: 0.55 }),
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 6, relaxEps: 0.05 },
      encoding: { tag: "quad-labelled" },
      weight: 0.012,
    },
    fieldLoss: ZERO_FIELD_LOSS,
    computeLoss: helmholtzChaosLoss(ZERO_FIELD_LOSS),
  },
  // Four Pixel GAN games on soft density drawings (docs/PIXEL_DISC.md).
  // Shared trunk: splat → conv3×3 → codebook. Reverse-mode gen through D(pos').
  {
    name: "Pixel · VecField",
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
    createField: () =>
      createFieldFromArch({ ...ARCH.dualFourier, alpha: 0.55 }),
    pixelDisc: {
      kind: "vec-field",
      weight: 0.04,
      G: 16,
      E: 8,
      K: 16,
      hidden: 32,
      dt: 0.15,
    },
    fieldLoss: {
      W_CHAOS: 0.2,
      W_ISO: 0.6,
      W_DIV: 0.1,
      W_SPIRAL: 0,
      HH: 1e-2,
      SPIRAL_TURNS: 3,
    },
    computeLoss: helmholtzChaosLoss({
      W_CHAOS: 0.2,
      W_ISO: 0.6,
      W_DIV: 0.1,
      W_SPIRAL: 0,
      HH: 1e-2,
      SPIRAL_TURNS: 3,
    }),
  },
  {
    name: "Pixel · NextFrame",
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
    createField: () =>
      createFieldFromArch({ ...ARCH.dualFourier, alpha: 0.55 }),
    pixelDisc: {
      kind: "next-frame",
      weight: 0.04,
      G: 16,
      E: 8,
      K: 16,
      hidden: 32,
      dt: 0.15,
    },
    fieldLoss: {
      W_CHAOS: 0.2,
      W_ISO: 0.6,
      W_DIV: 0.1,
      W_SPIRAL: 0,
      HH: 1e-2,
      SPIRAL_TURNS: 3,
    },
    computeLoss: helmholtzChaosLoss({
      W_CHAOS: 0.2,
      W_ISO: 0.6,
      W_DIV: 0.1,
      W_SPIRAL: 0,
      HH: 1e-2,
      SPIRAL_TURNS: 3,
    }),
  },
  {
    name: "Pixel · RealFake",
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
    createField: () =>
      createFieldFromArch({ ...ARCH.dualFourier, alpha: 0.55 }),
    pixelDisc: {
      kind: "real-fake",
      weight: 0.03,
      G: 16,
      E: 8,
      K: 16,
      hidden: 32,
      dt: 0.15,
    },
    fieldLoss: {
      W_CHAOS: 0.2,
      W_ISO: 0.6,
      W_DIV: 0.1,
      W_SPIRAL: 0,
      HH: 1e-2,
      SPIRAL_TURNS: 3,
    },
    computeLoss: helmholtzChaosLoss({
      W_CHAOS: 0.2,
      W_ISO: 0.6,
      W_DIV: 0.1,
      W_SPIRAL: 0,
      HH: 1e-2,
      SPIRAL_TURNS: 3,
    }),
  },
  {
    name: "Pixel · Inpaint",
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
    createField: () =>
      createFieldFromArch({ ...ARCH.dualFourier, alpha: 0.55 }),
    pixelDisc: {
      kind: "inpaint",
      weight: 0.04,
      G: 16,
      E: 8,
      K: 16,
      hidden: 32,
      dt: 0.15,
    },
    fieldLoss: {
      W_CHAOS: 0.2,
      W_ISO: 0.6,
      W_DIV: 0.1,
      W_SPIRAL: 0,
      HH: 1e-2,
      SPIRAL_TURNS: 3,
    },
    computeLoss: helmholtzChaosLoss({
      W_CHAOS: 0.2,
      W_ISO: 0.6,
      W_DIV: 0.1,
      W_SPIRAL: 0,
      HH: 1e-2,
      SPIRAL_TURNS: 3,
    }),
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
    createField: () =>
      createFieldFromArch({ ...ARCH.dualFourier, alpha: 0.45 }),
    fieldLoss: MAX_CHAOS_FIELD_LOSS,
    adversary: {
      tag: "on",
      kind: { tag: "wta", k: 6, relaxEps: 0.05 },
      encoding: { tag: "pair-rotation" },
      weight: 0.006, // reward units — see the note on the Single piece
    },
    computeLoss: helmholtzChaosLoss(MAX_CHAOS_FIELD_LOSS),
  },
];

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

export interface LoopHandle {
  field: HelmholtzField | null;
  getParticleCount(): number;
  setParticleCount(n: number): void;
  getSampleRate(): number;
  setSampleRate(n: number): void;
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
  /** Latest discriminator telemetry: surprise, predictor loss, per-head win
   *  shares and the collapse tripwire. `{tag:"off"}` for non-adversary pieces. */
  getAdversaryTelemetry(): AdversaryTelemetry;
  /** How the cloud is coloured. Raw/per-unit diagnostics are fused-only. */
  getColorMode(): ColorMode;
  setColorMode(m: ColorMode): void;
  /** Normalisation window of the selected fused surprise plane plus exact cloud
   *  coverage. `null` for velocity/RGB or an oracle-only tfjs path. */
  getSurpriseSpan(): { lo: number; mid: number; hi: number; covered: number; collapsed: boolean } | null;
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
  let sampleRate = 256; // points the field trains on per frame (live)
  let resetRate = cfg.resetRate; // respawn fraction (live — see setResetRate)
  let maxVelocity = cfg.maxVelocity;
  const query = new URLSearchParams(location.search);
  const initialGameControls = resolveLiveGameControls(cfg, query, maxVelocity);
  const driveEnabled = initialGameControls.driveEnabled;
  let drive = initialGameControls.drive;
  let forceMagnitude = initialGameControls.forceMagnitude;
  let generatorLearningRate = initialGameControls.generatorLearningRate;
  let discriminatorLearningRate = initialGameControls.discriminatorLearningRate;
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
  const border = options.overrides?.border ?? ({ tag: "wrap" } as const);
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
  // `?stroke=dot|vel|curl` — splat draw style (default dot, the shipped look).
  // "vel"/"curl" draw per-frame geometric strokes along each particle's
  // backward trajectory, so fast particles (maxVelocity ~26 px/frame vs a
  // ~1.6px dot) read as continuous filaments instead of disconnected dots.
  const strokeParam = new URLSearchParams(location.search).get("stroke");
  let strokeStyle: SplatStyle =
    strokeParam === "vel" || strokeParam === "curl" ? strokeParam : "dot";
  // `?strokeLen=F` — stroke length in FRAMES of travel (default 3, [0.5, 16]).
  // Number.isFinite (not `|| 3`) so an explicit `?strokeLen=0` clamps to the
  // documented 0.5 floor instead of silently becoming the default.
  const strokeLenParam = new URLSearchParams(location.search).get("strokeLen");
  const strokeLenParsed = strokeLenParam !== null ? parseFloat(strokeLenParam) : NaN;
  let strokeLen = Number.isFinite(strokeLenParsed)
    ? Math.max(0.5, Math.min(16, strokeLenParsed))
    : 3;
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
    if (!running || !(optimizer || trainer) || !advect || !wh || !device) return;
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
              genSeed: branch.genSeed(advRt.weight, 1, b),
              applyDisc: true,
            }
          );
        // Both complete their alternating D→G sequence while field weights
        // remain read-only. The ONE field step below then sums the two fresh
        // external gradients and applies exactly one field Adam update.
        encodeAdversaryBranch(advTrainer);
        if (advTrainerB) encodeAdversaryBranch(advTrainerB);
      }
      if (pixelDiscTrainer && cfg.pixelDisc) {
        const b = Math.min(sampleRate, 512);
        pixelDiscTrainer.encodeStep(enc, {
          b,
          alpha: (field as HelmholtzField).alpha,
          lr: discriminatorLearningRate,
          genWeight: cfg.pixelDisc.weight,
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
          predLoss: twoLane ? stA.discLoss + stB!.discLoss : stA.discLoss,
          winFractions,
          // A/B games are diagnosed independently. A measured failure in one
          // branch cannot be averaged away by the other.
          health: healthB ? combineHeadHealth(healthA, healthB) : healthA,
          weight: rtSnap.weight,
          branches: twoLane
            ? {
                disagree: {
                  surprise: stA.surprise,
                  predLoss: stA.discLoss,
                  winFractions: aFractions,
                  health: healthA,
                },
                agree: {
                  surprise: stB!.surprise,
                  predLoss: stB!.discLoss,
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
    // A surprise diagnostic takes the whole render pass: colour comes from one
    // plane of the fused per-particle diagnostic buffer, not from velocity.
    if (isSurpriseColorMode(colorMode) && advTrainer && advSurStats) {
      const metric = surpriseMetricOf(colorMode)!;
      const plane = advTrainer.surprisePlane(metric);
      const r0 = performance.now();
      const sr = surpriseRenderers.get(colorMode.colormap)!;
      sr.span = advSurStats.norm.span;
      sr.encodeRender(
        enc,
        advect.posBuffer,
        plane.buffer,
        advect.count,
        w,
        h,
        tsRender,
        plane.offsetFloats
      );
      advSurRec = advSurStats.encodeSample(
        enc,
        plane.buffer,
        advect.count,
        frame,
        advTrainer.surpriseCoverage().window,
        plane.offsetFloats
      );
      renderMs = performance.now() - r0;
    } else if (renderer) {
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
          `learn   ${emaTrain.toFixed(1)} ms  (cpu·tfjs)\n` +
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
    // ADVERSARY PIECES NOW TRAIN FUSED when the field supports it (raw or
    // fourier encoding, classless): AdversaryTrainer records the K-head
    // relaxed-WTA discriminator + the generator reward (extGrads → the field
    // trainer's pass B) into the same frame encoder. Verified vs the AD-IR
    // oracle AND tf.variableGrads at cos = 1.0000000 (tools/train_wta_test.ts);
    // measured 0.7-0.8 ms/step at B=512 vs the tfjs path's 19-32 ms.
    // hashgrid/class-aware fields and ?train=tfjs keep the tfjs autograd path,
    // and the loop says which one it picked out loud.
    const adversaryDisabled = advSpec.tag === "on" && fieldClasses > 0;
    const fusedAdvOk =
      advSpec.tag === "on" &&
      !adversaryDisabled &&
      !!field &&
      !wantTfjsTrainer &&
      advect.layout.classes === 0 &&
      advect.layout.encoding.kind !== "hashgrid";
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
        "[adversary] disabled for class-conditioned fields: class identity is " +
          "not yet part of the predictor context, so enabling it would create " +
          "fake hidden-state surprise."
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
          `k=${advRt.k} weight=${advRt.weight} — tfjs autograd trainer ` +
          `(${wantTfjsTrainer ? "?train=tfjs" : "field type unsupported by the fused adversary"})`
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
          k: headCount(advSpec.kind),
          relaxEps: advSpec.kind.tag === "wta" ? advSpec.kind.relaxEps : 0,
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
                `k=${headCount(advSpec.kind)} weight=${advRt.weight} — disc train + ` +
                `generator reward in-frame, tfjs idle`
        );
      }
      if (
        cfg.pixelDisc &&
        !!field &&
        !wantTfjsTrainer &&
        advect.layout.classes === 0 &&
        advect.layout.encoding.kind !== "hashgrid" &&
        (advect.layout.spec.kind === "helmholtz" ||
          advect.layout.spec.kind === "agree-disagree")
      ) {
        pixelDiscTrainer = new PixelDiscTrainer(device!, advect.layout, {
          fieldWeightsBuffer: advect.weightsBuffer,
          dims: {
            kind: cfg.pixelDisc.kind ?? "vec-field",
            G: cfg.pixelDisc.G,
            E: cfg.pixelDisc.E,
            K: cfg.pixelDisc.K,
            hidden: cfg.pixelDisc.hidden,
            dt: cfg.pixelDisc.dt,
          },
          batchCap: 512,
          seed: 20260805,
        });
        pixelDiscTrainer.setParticleBuffers(
          advect.posBuffer,
          advect.velBuffer,
          advect.count
        );
        console.log(
          `[pixel-disc] FUSED kind=${pixelDiscTrainer.kind} ` +
            `G=${pixelDiscTrainer.dims.G} E=${pixelDiscTrainer.dims.E} ` +
            `K=${pixelDiscTrainer.dims.K} weight=${cfg.pixelDisc.weight} — ` +
            `soft density critic + reverse-mode gen path (no JVP)`
        );
      }
      const extGradBuffers: GPUBuffer[] = [];
      if (advTrainer) extGradBuffers.push(advTrainer.extGradsBuf);
      if (advTrainerB) extGradBuffers.push(advTrainerB.extGradsBuf);
      else if (pixelDiscTrainer) extGradBuffers.push(pixelDiscTrainer.extGradsBuf);
      if (extGradBuffers.length > 2) {
        throw new Error(
          `[train] at most 2 extGrad buffers (got ${extGradBuffers.length}); ` +
            `Agree+Disagree + pixel-disc cannot share the same frame yet`
        );
      }
      trainer = new FusedTrainer(device!, advect.layout, {
        weightsBuffer: advect.weightsBuffer,
        batchCap: 1024,
        kSteps: rollout,
        extGradBuffers,
        loss: cfg.fieldLoss,
        border,
      });
      trainer.uploadWeights(advect.packCurrentWeights());
      trainer.setParticleBuffers(advect.posBuffer, advect.velBuffer, advect.count);
      advect.syncFromTfjs = false;
      console.log(
        `[train] fused trainer active (2 dispatches/step, tfjs idle, ` +
          `batch=${trainSource}, rollout=${rollout})`
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
        setSampleRate: (n: number) => {
          sampleRate = Math.max(1, Math.round(n));
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
          strokeLen = Math.max(0.5, Math.min(16, v));
          if (splat) splat.strokeLen = strokeLen;
        },
        getAdversaryWeight: () => (advRt.tag === "on" ? advRt.weight : 0),
        setAdversaryWeight: (x: number) => {
          // Generator role owns the sign. A negative UI weight is never used
          // as a hidden "agree" switch (Agree+Disagree has a named B role).
          if (advRt.tag === "on") advRt.weight = Math.max(0, Math.min(20, x));
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
      });
    }
    console.log(`starting: ${cfg.name} (webgpu)`);
    tick();
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
