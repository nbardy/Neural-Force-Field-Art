// MUST STAY THE FIRST IMPORT. It repairs an undecodable query string, and it
// only works if it runs before "./main" pulls in tfjs-core, whose module-level
// Environment constructor parses location.search and throws URIError on a bad
// "%" — blanking the page before any app code exists. See src/url_guard.ts.
import "./url_guard";
import React, {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import { createRoot } from "react-dom/client";
import {
  ADVERSARY_WEIGHT_RANGE,
  ADVERSARY_OBJECTIVE_DEFAULTS,
  ARCH,
  applyArchDockPreset,
  archDockPresets,
  DEFAULT_PIECE_INDEX,
  GALLERY,
  GAME_LEARNING_RATE_RANGE,
  INK_LOOK_DECAY,
  PIXEL_CRITIC_WEIGHT_RANGE,
  TRAIN_BATCH_MAX,
  TRAIN_BATCH_MIN,
  adversaryLossOf,
  adversaryTargetOf,
  describeFieldArch,
  inkLookFromRenderer,
  isArchPresetKey,
  resolveAdversary,
  resolveStrokeLength,
  resolveStrokeStyle,
  startLoop,
  type AdversaryTelemetry,
  type ArchPresetKey,
  type ColorMode,
  type HeadHealth,
  type InkLook,
  type LoopHandle,
  type PixelCriticSpec,
  predictorArchOf,
  PREDICTOR_ARCH_DEFAULT,
  PREDICTOR_ARCH_DOCK,
} from "./main";
import type { FieldHealth } from "./render/webgpu/field_probe";
import type { FamilyPayoff } from "./render/webgpu/adversary_train";
import { objectiveDims } from "./core/gan/adversary";
import type {
  AdversaryLoss,
  AdversaryTarget,
  TupleEncoding,
} from "./core/gan/adversary";
import {
  decodeDockParam,
  encodeDockParam,
  resolveSharedPiece,
  type SharedPiece,
} from "./share";
import type { ColormapName } from "./draw/colormap";
import type { BorderMode } from "./render/webgpu/advect_wgsl";
import type { SplatStyle } from "./render/webgpu/splat";
import "./ui.css";

const container = document.getElementById("app");
if (!container) throw new Error("Container not found");

const PMIN = 200;
const PMAX = 1_000_000;
const SURPRISE_HISTORY = 360; // ~72s at 200ms poll
const HISTORY_SMOOTH = 0.08; // EMA α for overlay (higher = snappier)
/**
 * Ceiling drawn on the R₁ chart. It is where the anti-collapse pressure is
 * measured to hold the direction order (the prototype note's target was
 * "R₁ ≤ ~0.1"; the live default piece sits at 0.01–0.10 with it on and 0.99
 * with it off). It is a LANDMARK, not a threshold anything reacts to — the
 * chart's fixed 0–1 scale would otherwise pin a healthy trace to the floor
 * where it is indistinguishable from an empty chart.
 */
const R1_HEALTHY_CEILING = 0.1;
/**
 * AC history length. The field probe lands at ~1 Hz (not the 200 ms poll rate),
 * so 180 points is a ~3 minute window — long enough to read a convergence
 * TREND, which is the thing a single AC number cannot tell you.
 */
const STRUCTURE_HISTORY = 180;
const CMAPS: ColormapName[] = ["inferno", "viridis", "coolwarm"];
const G_LR_MIN = GAME_LEARNING_RATE_RANGE.generator.min;
const G_LR_MAX = GAME_LEARNING_RATE_RANGE.generator.max;
const D_LR_MIN = GAME_LEARNING_RATE_RANGE.discriminator.min;
const D_LR_MAX = GAME_LEARNING_RATE_RANGE.discriminator.max;
/**
 * The critic's capacity, as the dock's one-line summary — the counterpart of
 * `describeFieldArch` for the OTHER net in the game. `guesses` is the same
 * `GuessKind` the relational adversary calls `kind` (src/core/gan/wta.ts is the
 * one spec), so it reads the same way in both sections.
 */
function describePixelCritic(spec: PixelCriticSpec): string {
  const g = spec.guesses ?? { tag: "single" as const };
  const guesses =
    g.tag === "single" ? "1 guess" : `${g.k} guesses ε${g.relaxEps}`;
  return `G${spec.G ?? 16} E${spec.E ?? 8} K${spec.K ?? 16} h${spec.hidden ?? 32} · ${guesses}`;
}

const ADV_WEIGHT_MIN = ADVERSARY_WEIGHT_RANGE.min;
const ADV_WEIGHT_MAX = ADVERSARY_WEIGHT_RANGE.max;

type RelationalView = "rotation" | "rotation-scale-raw" | "rotation-scale-adjusted";
type TupleView = "point" | "pair" | "tri" | "quad-labelled";

interface RuntimeConfig {
  piece: number;
  border: BorderMode;
  encoding: TupleEncoding;
  target: AdversaryTarget;
  loss: AdversaryLoss;
  adversaryKind: "off" | "single" | "wta";
  k: number;
  relaxEps: number;
  /** Dock override for archEditable pieces; null = piece default fieldArch. */
  archPreset: ArchPresetKey | null;
  /**
   * Dock override for the PREDICTOR's width; null = the piece's own (which is
   * PREDICTOR_ARCH_DEFAULT for every shipped piece). A KEY, not a
   * `PredictorArch`, for the same reason archPreset is: the persisted blob then
   * validates against a closed set instead of two free integers, and a stored
   * width cannot outlive a change to what the dock offers.
   */
  predictorPreset: string | null;
}

/** Live dials + compile-time dock knobs persisted across refresh. */
interface PersistedDock {
  runtime: RuntimeConfig;
  particles: number;
  samples: number;
  maxVelocity: number;
  drive: number;
  generatorLearningRate: number;
  discriminatorLearningRate: number;
  resetRate: number;
  decay: number;
  look: InkLook;
  blend: number;
  strokeStyle: SplatStyle;
  strokeLength: number;
  advWeight: number;
  colorMode: ColorMode;
}

/**
 * The v2 blob as it travels in a LINK: the same object localStorage holds,
 * plus the piece's name.
 *
 * The name is redundant with `runtime.piece` on the build that wrote it and is
 * the tie-breaker on every other build — see {@link resolveSharedPiece}. It is
 * additive on purpose: a shared blob is still a valid stored blob, and a
 * stored blob is still (name-less) shareable input.
 */
interface SharedDock extends PersistedDock {
  readonly pieceName: string;
}

const DOCK_STORAGE_KEY = "nffa.dock.v2";
/** Query parameter carrying a whole dock, base64url-encoded. */
const DOCK_SHARE_PARAM = "dock";
/** How long "COPIED ✓" stays on the button. */
const COPY_FLASH_MS = 1500;

/**
 * Where "mobile" starts for the collapsible HUD — the SAME 640px breakpoint the
 * layout rules in src/ui.css already use. React owns the collapsed STATE, CSS
 * owns the collapsed LOOK; a second, different breakpoint would let the two
 * disagree about which regime the viewport is in.
 */
const HUD_COLLAPSE_QUERY = "(max-width: 640px)";

/**
 * Initial collapse decision. Called from a useState initializer, i.e. during
 * the first render and BEFORE first paint — reading it in an effect instead
 * would paint the dock expanded over the artwork and then snap it shut, which
 * on a phone is the whole screen flashing.
 */
function hudStartsCollapsed(): boolean {
  return window.matchMedia(HUD_COLLAPSE_QUERY).matches;
}

type AdversaryLossTag = AdversaryLoss["tag"];

function lossWithTag(tag: AdversaryLossTag, previous: AdversaryLoss): AdversaryLoss {
  const tau =
    "tau" in previous ? previous.tau : ADVERSARY_OBJECTIVE_DEFAULTS.tau;
  const scaleWeight =
    "scaleWeight" in previous
      ? previous.scaleWeight
      : ADVERSARY_OBJECTIVE_DEFAULTS.scaleWeight;
  const energyWeight =
    "energyWeight" in previous
      ? previous.energyWeight
      : ADVERSARY_OBJECTIVE_DEFAULTS.energyWeight;
  const energyTarget =
    "energyTarget" in previous
      ? previous.energyTarget
      : ADVERSARY_OBJECTIVE_DEFAULTS.energyTarget;
  switch (tag) {
    case "raw-vector":
      return { tag };
    case "soft-angle":
      return { tag, tau };
    case "angle-relative-scale":
    case "angle-scale-hold":
      return { tag, tau, scaleWeight, energyWeight, energyTarget };
  }
}

function lossHasAngle(loss: AdversaryLoss): loss is Exclude<
  AdversaryLoss,
  { readonly tag: "raw-vector" }
> {
  return loss.tag !== "raw-vector";
}

function lossHasScale(loss: AdversaryLoss): loss is Extract<
  AdversaryLoss,
  { readonly tag: "angle-relative-scale" | "angle-scale-hold" }
> {
  return (
    loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold"
  );
}

const particleToSlider = (n: number): number =>
  Math.log(Math.min(Math.max(n, PMIN), PMAX) / PMIN) / Math.log(PMAX / PMIN);

const sliderToParticle = (t: number): number => {
  const raw = PMIN * Math.pow(PMAX / PMIN, t);
  const magnitude = Math.pow(10, Math.max(0, Math.floor(Math.log10(raw)) - 1));
  return Math.round(raw / magnitude) * magnitude;
};

const learningRateToSlider = (value: number, min: number, max: number): number =>
  Math.log(Math.min(max, Math.max(min, value)) / min) / Math.log(max / min);

const sliderToLearningRate = (value: number, min: number, max: number): number =>
  min * Math.pow(max / min, value);

/**
 * `?arch=<preset>` — the ARCHITECTURE axis, as a first-class URL knob.
 *
 * It is deliberately GLOBAL and survives a gallery switch, exactly like
 * `?advM`/`?advK`/`?advPolar` (see the comment in `defaultsForPiece`): a config
 * sweep varies architecture ACROSS pieces, and a knob that silently reverted on
 * every piece change could not express that. A DOCK-set preset keeps the old
 * behaviour of not following you to the next piece — that one is an aesthetic
 * choice about the piece you are looking at, not a declared experiment.
 *
 * Unknown values THROW rather than falling back to the piece default. A sweep
 * that mistypes an axis value must fail at the first cell, not spend an hour
 * measuring the default architecture N times and report it as N architectures.
 * (The measured `ArchHealth` fingerprint in the snapshot is the second line of
 * defence, for the pieces that are not `archEditable` at all.)
 */
function resolveArchPreset(q: URLSearchParams): ArchPresetKey | null {
  const raw = q.get("arch");
  if (raw === null) return null;
  if (!isArchPresetKey(raw)) {
    throw new URIError(
      `?arch must be one of ${Object.keys(ARCH).join(", ")} — got '${raw}'`
    );
  }
  return raw;
}

function defaultsForPiece(piece: number): RuntimeConfig {
  // URL adversary knobs are intentionally GLOBAL: selecting another gallery
  // piece re-resolves that piece through the same query. This matches
  // startLoop's canonical URL policy and prevents React from masking
  // ?advM/?advK/?advEps by passing the piece defaults back as overrides.
  const adv = resolveAdversary(
    GALLERY[piece].adversary,
    new URLSearchParams(window.location.search)
  );
  return {
    piece,
    border: GALLERY[piece].border ?? { tag: "wrap" },
    encoding: adv.tag === "on" ? adv.encoding : ({ tag: "pair-rotation" } as TupleEncoding),
    target: adv.tag === "on" ? adversaryTargetOf(adv) : { tag: "force" },
    loss: adv.tag === "on" ? adversaryLossOf(adv) : { tag: "raw-vector" },
    adversaryKind: adv.tag === "on" ? adv.kind.tag : "off",
    k: adv.tag === "on" && adv.kind.tag === "wta" ? adv.kind.k : 1,
    relaxEps: adv.tag === "on" && adv.kind.tag === "wta" ? adv.kind.relaxEps : 0,
    archPreset: resolveArchPreset(new URLSearchParams(window.location.search)),
    // Not URL-resolved: `?advHidden`/`?advFeature` are free integers that
    // resolveAdversary already applied to `adv` above, and there is no key for
    // an arbitrary pair. null means "whatever the spec says", which is exactly
    // right — the dock shows the resolved width either way.
    predictorPreset: null,
  };
}

/** Switch gallery piece without wiping dock dials. */
function runtimeForPieceSwitch(
  previous: RuntimeConfig,
  piece: number
): RuntimeConfig {
  if (previous.piece === piece) return previous;
  const nextDefaults = defaultsForPiece(piece);
  // Re-adopt the piece's baked observer + loss + target + K/ε. Those are the
  // didactic identity of each gallery entry (Pair = soft-angle, Quad =
  // raw-vector K=6, …). Keeping loss across switches used to strand Pair on
  // RAW after a Quad visit — Euclidean amplitude games go diagonal and look
  // like "the angle disc learned nothing." Keeping K across WTA pieces was
  // the same class of bug (Pair K=4 silently stuck on Quad). Live dials
  // (particles, LRs, drive, trails, …) stay in React state outside this
  // object and are re-synced from the new piece on gallery switch.
  return {
    ...previous,
    piece,
    encoding: nextDefaults.encoding,
    target: nextDefaults.target,
    loss: nextDefaults.loss,
    adversaryKind: nextDefaults.adversaryKind,
    k: nextDefaults.k,
    relaxEps: nextDefaults.relaxEps,
    // Don't carry an aesthetic DOCK arch preset onto a locked game piece — but
    // a URL-declared `?arch=` is a global knob and re-resolves here, the same
    // way ?advM/?advK do through nextDefaults. See resolveArchPreset.
    archPreset: nextDefaults.archPreset,
    predictorPreset: nextDefaults.predictorPreset,
  };
}

function isBorderMode(value: unknown): value is BorderMode {
  return (
    !!value &&
    typeof value === "object" &&
    "tag" in value &&
    (value.tag === "wrap" || value.tag === "bounce" || value.tag === "reset")
  );
}

function isTupleEncoding(value: unknown): value is TupleEncoding {
  if (!value || typeof value !== "object" || !("tag" in value)) return false;
  const tag = (value as { tag: string }).tag;
  return (
    tag === "point" ||
    tag === "pair" ||
    tag === "pair-rotation" ||
    tag === "pair-rotation-scale-raw" ||
    tag === "pair-rotation-scale-adjusted" ||
    tag === "tri" ||
    tag === "quad-labelled"
  );
}

function isAdversaryTarget(value: unknown): value is AdversaryTarget {
  return (
    !!value &&
    typeof value === "object" &&
    "tag" in value &&
    (value.tag === "force" || value.tag === "post-velocity")
  );
}

function isAdversaryLoss(value: unknown): value is AdversaryLoss {
  if (!value || typeof value !== "object" || !("tag" in value)) return false;
  const loss = value as AdversaryLoss;
  switch (loss.tag) {
    case "raw-vector":
      return true;
    case "soft-angle":
      return Number.isFinite(loss.tau) && loss.tau > 0;
    case "angle-relative-scale":
    case "angle-scale-hold":
      return (
        Number.isFinite(loss.tau) &&
        loss.tau > 0 &&
        Number.isFinite(loss.scaleWeight) &&
        loss.scaleWeight >= 0 &&
        Number.isFinite(loss.energyWeight) &&
        loss.energyWeight >= 0 &&
        Number.isFinite(loss.energyTarget) &&
        loss.energyTarget > 0
      );
    default:
      return false;
  }
}

function isColorMode(value: unknown): value is ColorMode {
  if (!value || typeof value !== "object" || !("tag" in value)) return false;
  const mode = value as ColorMode;
  if (mode.tag === "velocity") return true;
  if (mode.tag === "surprise-raw" || mode.tag === "surprise-per-unit") {
    return (CMAPS as readonly string[]).includes(mode.colormap);
  }
  return false;
}

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n));
}

/**
 * Who owns the adversary RECIPE — observer / target / loss / K / ε and the
 * reward — inside a restored blob.
 *
 * - "piece": the gallery piece's baked recipe wins. This is the localStorage
 *   policy. A saved blob is a side effect of clicking around, and an early
 *   dock build carried raw-vector / weight 0 from another piece onto Pair and
 *   SAVED that poison (Euclidean amplitude games go diagonal; the soft-angle
 *   swirls vanish). Nobody chose it, so it does not get to win.
 * - "blob": the blob's own recipe wins. This is the share-link policy. A
 *   `?dock=` link is an explicit act carrying an explicitly-tuned recipe —
 *   exactly like `?advLoss=` / `?advTau=` / `?advK=`, which already outrank
 *   storage AND the piece defaults (see urlHasDockOverrides). Under "piece"
 *   the settings this feature exists to share are unrepresentable: tuple
 *   POINT / K 8 / ε 0.22 on a piece baked as PAIR / K 4 / ε 0.05 would come
 *   back silently as PAIR / 4 / 0.05, i.e. a link that lies about itself.
 */
type RecipePolicy = "piece" | "blob";

/** Exactly the fields {@link RecipePolicy} arbitrates. */
interface RestoredRecipe {
  readonly encoding: TupleEncoding;
  readonly target: AdversaryTarget;
  readonly loss: AdversaryLoss;
  readonly k: number;
  readonly relaxEps: number;
  readonly advWeight: number;
}

/** Thin dispatcher: one handler per policy, both fed already-validated values. */
function restoredRecipe(
  policy: RecipePolicy,
  piece: RuntimeConfig,
  pieceWeight: number,
  blob: RestoredRecipe
): RestoredRecipe {
  switch (policy) {
    case "blob":
      return blob;
    case "piece":
      return {
        encoding: piece.encoding,
        target: piece.target,
        loss: piece.loss,
        k: piece.k,
        relaxEps: piece.relaxEps,
        // A STORED 0 reward with the game on is v1 poison, not a parked dial,
        // so the piece's own reward comes back. (pieceWeight is itself 0 on a
        // piece with no adversary, which is why this needs no kind test.) A
        // LINK's 0 is a real choice — "run the game, feed the HUD, do not
        // steer the field" — and "blob" keeps it, the same reading main.ts's
        // floatParam already gives an explicit `?advWeight=0`.
        advWeight: blob.advWeight === 0 ? pieceWeight : blob.advWeight,
      };
  }
}

function parsePersistedDock(
  raw: unknown,
  policy: RecipePolicy
): PersistedDock | null {
  if (!raw || typeof raw !== "object") return null;
  const data = raw as Partial<PersistedDock>;
  const runtime = data.runtime as Partial<RuntimeConfig> | undefined;
  if (!runtime) return null;
  const piece = Number(runtime.piece);
  if (!Number.isInteger(piece) || piece < 0 || piece >= GALLERY.length) return null;
  if (!isBorderMode(runtime.border)) return null;
  if (!isTupleEncoding(runtime.encoding)) return null;
  if (!isAdversaryTarget(runtime.target)) return null;
  if (!isAdversaryLoss(runtime.loss)) return null;
  if (
    runtime.adversaryKind !== "off" &&
    runtime.adversaryKind !== "single" &&
    runtime.adversaryKind !== "wta"
  ) {
    return null;
  }
  const k = Number(runtime.k);
  const relaxEps = Number(runtime.relaxEps);
  if (!Number.isFinite(k) || !Number.isFinite(relaxEps)) return null;
  const archPreset =
    runtime.archPreset === null || runtime.archPreset === undefined
      ? null
      : isArchPresetKey(runtime.archPreset)
        ? runtime.archPreset
        : null;
  // Unknown key -> null (the piece's own width), never a throw: a blob written
  // by a build whose dock offered a different set must degrade to the default,
  // the same way an unknown archPreset does.
  const predictorPreset =
    typeof runtime.predictorPreset === "string" &&
    PREDICTOR_ARCH_DOCK.some((o) => o.key === runtime.predictorPreset)
      ? runtime.predictorPreset
      : null;

  const particles = Number(data.particles);
  const samples = Number(data.samples);
  const maxVelocity = Number(data.maxVelocity);
  const drive = Number(data.drive);
  const generatorLearningRate = Number(data.generatorLearningRate);
  const discriminatorLearningRate = Number(data.discriminatorLearningRate);
  const resetRate = Number(data.resetRate);
  const decay = Number(data.decay);
  const blend = Number(data.blend);
  const strokeLength = Number(data.strokeLength);
  const advWeight = Number(data.advWeight);
  if (
    ![
      particles,
      samples,
      maxVelocity,
      drive,
      generatorLearningRate,
      discriminatorLearningRate,
      resetRate,
      decay,
      blend,
      strokeLength,
      advWeight,
    ].every(Number.isFinite)
  ) {
    return null;
  }
  if (data.look !== "ghost" && data.look !== "clean" && data.look !== "trails") {
    return null;
  }
  if (data.strokeStyle !== "dot" && data.strokeStyle !== "vel" && data.strokeStyle !== "curl") {
    return null;
  }
  if (!isColorMode(data.colorMode)) return null;

  // Which half of the runtime the blob is trusted with is a POLICY decision,
  // made once, by the caller — see RecipePolicy.
  const pieceRecipe = defaultsForPiece(piece);
  const galleryAdv = GALLERY[piece].adversary;
  const pieceWeight = galleryAdv?.tag === "on" ? galleryAdv.weight : 0;
  // `k = 1` IS the variant "single"; createAdversary THROWS on a wta with
  // k < 2 (src/core/gan/adversary.ts:1483), which on a hand-edited link would
  // take the page down instead of the link. The floor therefore follows the
  // piece's kind — the only place that knows whether k is even read.
  const kFloor = pieceRecipe.adversaryKind === "wta" ? 2 : 1;
  const recipe = restoredRecipe(policy, pieceRecipe, pieceWeight, {
    encoding: runtime.encoding,
    target: runtime.target,
    loss: runtime.loss,
    k: clamp(Math.round(k), kFloor, 12),
    relaxEps: clamp(relaxEps, 0, 0.45),
    advWeight: clamp(advWeight, ADV_WEIGHT_MIN, ADV_WEIGHT_MAX),
  });
  // The one cross-field invariant the per-field guards above cannot see, and
  // the one startLoop itself THROWS on (main.ts calls objectiveDims before
  // building the loop): a post-velocity target is point-only. A hand-edited
  // link pairing it with a relational observer must be rejected as bad input
  // here, not crash the page there.
  try {
    objectiveDims(recipe.encoding, recipe.target, recipe.loss);
  } catch {
    return null;
  }
  return {
    runtime: {
      piece,
      border: runtime.border,
      encoding: recipe.encoding,
      target: recipe.target,
      loss: recipe.loss,
      // The adversary's EXISTENCE is piece identity, never a dial: startLoop
      // resolves it from GALLERY + `?adv=` and no dock override can switch a
      // game on. Adopting a blob's kind onto a piece that has none would draw
      // a full adversary panel over a loop that reports no telemetry.
      adversaryKind: pieceRecipe.adversaryKind,
      k: recipe.k,
      relaxEps: recipe.relaxEps,
      archPreset,
      predictorPreset,
    },
    particles: clamp(Math.round(particles), PMIN, PMAX),
    // Restored dock values are bounded by the architectural ceiling only; the
    // live trainer's device-derived cap re-clamps in setSampleRate. (Was a
    // hard 1024 — a stale narrower bound than the slider itself allowed.)
    samples: clamp(Math.round(samples), TRAIN_BATCH_MIN, TRAIN_BATCH_MAX),
    maxVelocity: clamp(maxVelocity, 1, 200),
    drive: clamp(drive, 0, 1),
    generatorLearningRate: clamp(generatorLearningRate, G_LR_MIN, G_LR_MAX),
    discriminatorLearningRate: clamp(
      discriminatorLearningRate,
      D_LR_MIN,
      D_LR_MAX
    ),
    resetRate: clamp(resetRate, 0, 1),
    decay: clamp(decay, 0, 0.99),
    look: data.look,
    blend: clamp(blend, 0, 1),
    strokeStyle: data.strokeStyle,
    strokeLength: clamp(strokeLength, 0.5, 16),
    advWeight: recipe.advWeight,
    colorMode: data.colorMode,
  };
}

/** The single-knob deep links. Explicit and per-field, so they win outright. */
const DOCK_OVERRIDE_PARAMS = [
  "adv",
  "advK",
  "advM",
  "advEps",
  "advWeight",
  "advTarget",
  "advLoss",
  "advTau",
  "advScaleWeight",
  "advEnergyWeight",
  "advEnergyTarget",
  "advPolar",
  "advNematic",
  "advPolarTau",
  "gLR",
  "dLR",
  "pixW",
  "drive",
  "color",
  "cmap",
  "decay",
  "stroke",
  "strokeLen",
] as const;

/** Explicit shareable URL knobs win over a saved dock (deep links stay honest). */
function urlHasDockOverrides(q: URLSearchParams): boolean {
  return DOCK_OVERRIDE_PARAMS.some((key) => q.has(key));
}

function loadStoredDock(): PersistedDock | null {
  try {
    const raw = window.localStorage.getItem(DOCK_STORAGE_KEY);
    if (!raw) return null;
    return parsePersistedDock(JSON.parse(raw), "piece");
  } catch {
    return null;
  }
}

/* ─── export / share ──────────────────────────────────────────────────────
   The dock's live state, out as a link and back in again. ONE serializer:
   the App builds a single PersistedDock per render and hands the SAME object
   to savePersistedDock and to these, so a copied link and a saved session can
   never describe different settings.                                        */

function sharedDock(dock: PersistedDock): SharedDock {
  return { ...dock, pieceName: GALLERY[dock.runtime.piece].name };
}

/** Every dial in the dock, as a link back to this page. */
function dockToShareUrl(dock: PersistedDock): string {
  const base = `${window.location.origin}${window.location.pathname}`;
  return `${base}?${DOCK_SHARE_PARAM}=${encodeDockParam(sharedDock(dock))}`;
}

/** The same blob, readable — for bug reports and durable pasting. */
function dockToShareJson(dock: PersistedDock): string {
  return JSON.stringify(sharedDock(dock), null, 2);
}

/** The query string as parameters, or the reason it is not readable at all. */
type Query =
  | { readonly tag: "ok"; readonly params: URLSearchParams }
  | { readonly tag: "malformed"; readonly reason: string };

/**
 * CORRECTION (measured 2026-08-17, Node 24 and Chrome/Metal): the premise this
 * function was written on is FALSE. `new URLSearchParams("?%%%")` does NOT
 * throw — WHATWG urlencoded parsing is lenient and hands back the key `"%%%"`
 * verbatim. So the `malformed` branch below is unreachable in practice; it is
 * kept as a cheap invariant rather than removed, because dock ingestion runs
 * inside a render, where any unhandled throw unmounts the tree and leaves a
 * BLANK PAGE — the worst possible answer to a mistyped link.
 *
 * The blanking really was real, but it was never here. tfjs-core hand-rolls its
 * own parser (`populateURLFlags` → `decodeURIComponent`) and runs it at MODULE
 * SCOPE, so `?dock=%%%` threw URIError while `import * as tf` was still being
 * evaluated — before this function, this component, or any log line existed.
 * That hole is now closed at the only layer that can close it: `src/url_guard.ts`,
 * the first import of this file, which drops undecodable parameters (loudly)
 * before tfjs is reached.
 */
function readQuery(search: string): Query {
  try {
    return { tag: "ok", params: new URLSearchParams(search) };
  } catch (error) {
    return {
      tag: "malformed",
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

/** A `?dock=` parameter, canonicalized. */
type DockParam =
  | { readonly tag: "absent" }
  | { readonly tag: "invalid"; readonly reason: string }
  | {
      readonly tag: "ok";
      readonly dock: PersistedDock;
      readonly piece: SharedPiece;
    };

/**
 * κ — the ONE place a `?dock=` parameter becomes a dock. Transport, then piece
 * identity, then the same v2 validator storage uses (under "blob" policy).
 * Reports; never repairs.
 */
function parseDockParam(search: string): DockParam {
  const query = readQuery(search);
  if (query.tag === "malformed") {
    return { tag: "invalid", reason: `query string is undecodable (${query.reason})` };
  }
  const raw = query.params.get(DOCK_SHARE_PARAM);
  if (raw === null) return { tag: "absent" };
  const decoded = decodeDockParam(raw);
  if (decoded.tag === "invalid") return decoded;
  if (!decoded.json || typeof decoded.json !== "object") {
    return { tag: "invalid", reason: "decoded to a non-object" };
  }
  const blob = decoded.json as Partial<SharedDock>;
  const piece = resolveSharedPiece(
    GALLERY.map((entry) => entry.name),
    (blob.runtime as Partial<RuntimeConfig> | undefined)?.piece,
    blob.pieceName
  );
  // Name beats stale index — so the index the validator bounds-checks below is
  // the resolved one, not the one that travelled.
  const dock = parsePersistedDock(
    { ...blob, runtime: { ...(blob.runtime ?? {}), piece: piece.piece } },
    "blob"
  );
  if (!dock) return { tag: "invalid", reason: "failed v2 dock validation" };
  return { tag: "ok", dock, piece };
}

/** One handler per identity outcome; two of the three are worth saying aloud. */
function warnSharedPiece(piece: SharedPiece): void {
  switch (piece.tag) {
    case "index":
      return;
    case "renamed":
      console.warn(
        `[dock] shared link names piece "${piece.name}" but carries index ` +
          `${piece.staleIndex}; resolving BY NAME to index ${piece.piece}. ` +
          `GALLERY is append-only, so this link predates a reorder.`
      );
      return;
    case "unknown-name":
      console.warn(
        `[dock] shared link names piece "${piece.name}", which is not in this ` +
          `build's GALLERY (renamed or removed); falling back to its index ` +
          `${piece.piece}, which may be a DIFFERENT artwork.`
      );
      return;
  }
}

/**
 * Where the dock's opening state comes from. Precedence, highest first:
 *
 *   explicit knob params (?gLR, ?advLoss, …) > ?dock= > localStorage > piece defaults
 *
 * The single-knob params win COARSELY: their presence suppresses both restore
 * sources and hands the whole decision to main.ts's κ, which is the only thing
 * that can apply them to the running loop. Half-restoring a dock underneath
 * them would put the sliders and the artwork in different states — the exact
 * dishonesty the "deep links stay honest" rule exists to prevent.
 */
function initialDock(): PersistedDock | null {
  try {
    // Retire the v1 blob that could store raw-vector on Pair.
    window.localStorage.removeItem("nffa.dock.v1");
  } catch {
    // Private mode / disabled storage — nothing to retire.
  }
  const search = window.location.search;
  const query = readQuery(search);
  if (query.tag === "malformed") {
    // Nothing in this URL can be trusted — including whether a knob param is
    // present — so neither restore source may run. Piece defaults it is.
    console.warn(
      `[dock] query string is undecodable (${query.reason}); ` +
        `ignoring the whole URL and opening on piece defaults.`
    );
    return null;
  }
  const shared = parseDockParam(search);
  if (urlHasDockOverrides(query.params)) {
    if (shared.tag !== "absent") {
      const knobs = DOCK_OVERRIDE_PARAMS.filter((key) => query.params.has(key));
      console.warn(
        `[dock] ignoring ?${DOCK_SHARE_PARAM}= — explicit knob params ` +
          `(${knobs.join(", ")}) outrank a shared dock. Drop them to use the link.`
      );
    }
    return null;
  }
  switch (shared.tag) {
    case "ok":
      warnSharedPiece(shared.piece);
      console.info(
        `[dock] adopted ?${DOCK_SHARE_PARAM}= share link · piece ` +
          `"${GALLERY[shared.dock.runtime.piece].name}"`
      );
      return shared.dock;
    case "invalid":
      console.warn(
        `[dock] ignoring ?${DOCK_SHARE_PARAM}= — ${shared.reason}. ` +
          `Falling back to localStorage, then piece defaults.`
      );
      return loadStoredDock();
    case "absent":
      return loadStoredDock();
  }
}

/**
 * κ for the clipboard API. Some contexts (plain http:, old webviews, locked-
 * down embeds) do not expose it at all, and that has to reach the button as a
 * rejected promise — not as a TypeError thrown out of a click handler.
 */
function writeClipboard(text: string): Promise<void> {
  const clipboard = navigator.clipboard;
  return clipboard
    ? clipboard.writeText(text)
    : Promise.reject(
        new Error("navigator.clipboard is unavailable (needs a secure context)")
      );
}

type CopyTarget = "link" | "json";

/** What the share row is showing right now. */
type CopyState =
  | { readonly tag: "idle" }
  | { readonly tag: "copied"; readonly target: CopyTarget }
  | { readonly tag: "failed"; readonly target: CopyTarget };

/** Label + flash for ONE button, so the two can never disagree. */
function copyView(
  state: CopyState,
  target: CopyTarget,
  idle: string
): { label: string; flash: string } {
  switch (state.tag) {
    case "idle":
      return { label: idle, flash: "none" };
    case "copied":
      return state.target === target
        ? { label: "COPIED ✓", flash: "copied" }
        : { label: idle, flash: "none" };
    case "failed":
      return state.target === target
        ? { label: "FAILED ✗", flash: "failed" }
        : { label: idle, flash: "none" };
  }
}

function savePersistedDock(dock: PersistedDock): void {
  try {
    window.localStorage.setItem(DOCK_STORAGE_KEY, JSON.stringify(dock));
  } catch {
    // Quota / private mode — dock still works in-session.
  }
}

function encodingForView(view: RelationalView): TupleEncoding {
  switch (view) {
    case "rotation":
      return { tag: "pair-rotation" };
    case "rotation-scale-raw":
      return { tag: "pair-rotation-scale-raw" };
    case "rotation-scale-adjusted":
      return { tag: "pair-rotation-scale-adjusted" };
  }
}

function viewForEncoding(encoding: TupleEncoding): RelationalView | null {
  switch (encoding.tag) {
    case "pair":
    case "pair-rotation":
      return "rotation";
    case "pair-rotation-scale-raw":
      return "rotation-scale-raw";
    case "pair-rotation-scale-adjusted":
      return "rotation-scale-adjusted";
    default:
      return null;
  }
}

function tupleViewForEncoding(encoding: TupleEncoding): TupleView {
  switch (encoding.tag) {
    case "point":
      return "point";
    case "pair":
    case "pair-rotation":
    case "pair-rotation-scale-raw":
    case "pair-rotation-scale-adjusted":
      return "pair";
    case "tri":
      return "tri";
    case "quad-labelled":
      return "quad-labelled";
  }
}

function encodingForTuple(
  tuple: TupleView,
  current: TupleEncoding
): TupleEncoding {
  switch (tuple) {
    case "point":
      return { tag: "point" };
    case "pair":
      // Preserve the selected pair quotient. Entering pair mode from another
      // arity chooses the scale-adjusted observer (flagship), not raw
      // pair-rotation (the easier amplitude-baseline observer).
      return viewForEncoding(current)
        ? current
        : { tag: "pair-rotation-scale-adjusted" };
    case "tri":
      return { tag: "tri" };
    case "quad-labelled":
      return { tag: "quad-labelled" };
  }
}

function observerLabel(
  encoding: TupleEncoding,
  target: AdversaryTarget
): string {
  switch (encoding.tag) {
    case "point":
      return target.tag === "post-velocity"
        ? "POINT · absolute x + incoming v → pre-border v+"
        : "POINT · absolute x → F(x)";
    case "pair":
    case "pair-rotation":
      return "PAIR · rotation quotient";
    case "pair-rotation-scale-raw":
      return "PAIR · similarity-blind context, raw preset";
    case "pair-rotation-scale-adjusted":
      return "PAIR · similarity-blind context, angle preset";
    case "tri":
      return "TRI · unordered E(2); ties inactive";
    case "quad-labelled":
      return "QUAD-L · labelled rotation quotient; raw scale";
  }
}

function ControlSection({
  title,
  children,
  testid,
}: {
  title: string;
  children: ReactNode;
  testid?: string;
}): ReactElement {
  return (
    <section className="tui-section" aria-label={title} data-testid={testid}>
      <div className="tui-section-title">{title}</div>
      <div className="tui-section-body">{children}</div>
    </section>
  );
}

function ControlSubsection({
  title,
  children,
  testid,
}: {
  title: string;
  children: ReactNode;
  testid?: string;
}): ReactElement {
  return (
    <section className="tui-subsection" aria-label={title} data-testid={testid}>
      <div className="tui-subsection-title">{title}</div>
      <div className="tui-subsection-body">{children}</div>
    </section>
  );
}

function RangeRow({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
  testid,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
  testid: string;
}): ReactElement {
  return (
    <label className="tui-row" data-testid={testid}>
      <span className="tui-label">{label}</span>
      <input
        className="block-slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        aria-label={label}
      />
      <output className="tui-value">{display}</output>
    </label>
  );
}

function Segmented<T extends string>({
  label,
  value,
  choices,
  onChange,
  testid,
}: {
  label: string;
  value: T | null;
  choices: readonly { value: T; label: string; title?: string }[];
  onChange: (value: T) => void;
  testid: string;
}): ReactElement {
  return (
    <div className="tui-segment-row" data-testid={testid}>
      <span className="tui-label">{label}</span>
      <div className="tui-segments" role="radiogroup" aria-label={label}>
        {choices.map((choice) => (
          <button
            key={choice.value}
            type="button"
            className="tui-chip"
            role="radio"
            aria-checked={value === choice.value}
            data-active={value === choice.value ? "true" : "false"}
            data-tooltip={choice.title}
            title={choice.title}
            onClick={() => onChange(choice.value)}
          >
            {choice.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function emaSeries(data: readonly number[], alpha: number): number[] {
  const out: number[] = [];
  let ema = Number.NaN;
  for (const sample of data) {
    if (!Number.isFinite(sample)) {
      out.push(ema);
      continue;
    }
    ema = Number.isFinite(ema) ? ema + alpha * (sample - ema) : sample;
    out.push(ema);
  }
  return out;
}

/**
 * y-axis policy. `auto` is the historical per-series autoscale. `fixed` is for
 * quantities with an absolute meaning — an order parameter lives in [0,1] and
 * autoscaling it would magnify 0.05 → 0.07 noise into a full-height alarm.
 */
type SparkScale =
  | { readonly tag: "auto" }
  | { readonly tag: "fixed"; readonly lo: number; readonly hi: number };

/**
 * Optional horizontal reference rule. `at` extends an `auto` domain to contain
 * the value, so "how far are we from that line" is legible even when the
 * series never approaches it.
 */
type SparkRule =
  | { readonly tag: "none" }
  | { readonly tag: "at"; readonly value: number; readonly label: string };

/**
 * Family colours — the SAME R/G/B the `rgb-families` palette paints the cloud
 * with (splat.ts / points.ts). The chart is only readable if "the red curve"
 * and "the red particles" are the same family, so these two lists are a pair:
 * change one and change the other.
 */
const FAMILY_COLORS = ["#ff5555", "#3ddc6b", "#5f7bff"] as const;

/**
 * Multi-series sparkline on ONE SHARED domain.
 *
 * The shared domain is the whole point: the question this row answers is
 * "are the families balanced, or is one of them carrying the payoff", and
 * per-series auto-scaling would normalize exactly that difference away — three
 * curves at 0.1, 0.6 and 1.3 would each fill the box and look identical.
 */
function FamilySparkline({
  series,
  label,
}: {
  series: readonly (readonly number[])[];
  label: string;
}): ReactElement {
  const width = 160;
  const height = 28;
  const pool = series.flat().filter(Number.isFinite);
  const drawable = series.filter((s) => s.length >= 2);
  if (!pool.length || !drawable.length) {
    return (
      <svg
        className="sparkline"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${label} awaiting samples`}
      />
    );
  }
  const lo = Math.min(...pool);
  const hi = Math.max(...pool);
  const flat = hi - lo <= Math.max(Math.abs(hi), 1e-12) * 1e-3;
  const span = Math.max(hi - lo, 1e-30);
  const toY = (sample: number) =>
    flat ? height / 2 : height - 2 - ((sample - lo) / span) * (height - 4);
  return (
    <svg
      className="sparkline"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${label} from ${lo.toExponential(1)} to ${hi.toExponential(1)}`}
    >
      {series.map((s, c) =>
        s.length < 2 ? null : (
          <polyline
            key={c}
            className="sparkline-raw"
            stroke={FAMILY_COLORS[c % FAMILY_COLORS.length]}
            strokeOpacity={0.95}
            points={s
              .map((sample, index) => {
                const x = (index / (s.length - 1)) * (width - 2) + 1;
                return `${x.toFixed(1)},${toY(sample).toFixed(1)}`;
              })
              .join(" ")}
          />
        )
      )}
    </svg>
  );
}

/**
 * The per-family payoff row. Takes the sum type whole so the `unmeasured` case
 * is a HANDLER (render nothing) rather than a truthiness check at the call site
 * — the row must be absent on every piece that has no families, not present
 * and empty.
 */
function FamilyPayoffRow({
  payoff,
  series,
}: {
  payoff: FamilyPayoff;
  series: readonly (readonly number[])[];
}): ReactElement | null {
  if (payoff.tag === "unmeasured") return null;
  return (
    <div className="diagnostic-row" data-testid="family-payoff-chart">
      <span
        className="diagnostic-name"
        title={
          "Shared payoff split by particle FAMILY. The generator sees" +
          " (coordinate, family); the predictor sees the coordinate only, so it" +
          " cannot tell the families apart and must spend its K hypotheses on a" +
          " multi-mode conditional. Curves share ONE scale: the family ABOVE the" +
          " others is the one the predictor is currently failing on. Watch for" +
          " the roles rotating (a live game), converging (families collapsed" +
          " onto one field), or one curve pinned high forever (a permanent" +
          " scapegoat). Colours match the particles."
        }
      >
        family payoff
      </span>
      <FamilySparkline series={series} label="payoff per family" />
      <strong>
        {payoff.mean
          .map((v, c) => (payoff.count[c] > 0 ? v.toFixed(2) : "—"))
          .join(" ")}
      </strong>
    </div>
  );
}

function Sparkline({
  data,
  smoothed,
  label = "history",
  scale = { tag: "auto" },
  rule = { tag: "none" },
}: {
  data: readonly number[];
  /** Optional second series (e.g. EMA) drawn brighter on top. */
  smoothed?: readonly number[];
  label?: string;
  scale?: SparkScale;
  rule?: SparkRule;
}): ReactElement {
  const width = 160;
  const height = 28;
  if (data.length < 2) {
    return (
      <svg
        className="sparkline"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${label} awaiting samples`}
      />
    );
  }

  const pool = [...data, ...(smoothed ?? [])].filter(Number.isFinite);
  if (!pool.length) {
    return (
      <svg
        className="sparkline"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${label} unavailable`}
      />
    );
  }

  const auto = scale.tag === "auto";
  const domain =
    scale.tag === "fixed"
      ? [scale.lo, scale.hi]
      : rule.tag === "at"
        ? [Math.min(...pool, rule.value), Math.max(...pool, rule.value)]
        : [Math.min(...pool), Math.max(...pool)];
  const [lo, hi] = domain;
  const flat = auto && hi - lo <= Math.max(Math.abs(hi), 1e-12) * 1e-3;
  const span = Math.max(hi - lo, 1e-30);
  const toY = (sample: number) =>
    flat
      ? height / 2
      : height -
        2 -
        ((Number.isFinite(sample) ? sample - lo : 0) / span) * (height - 4);
  const toPoints = (series: readonly number[]) =>
    series
      .map((sample, index) => {
        const x = (index / (series.length - 1)) * (width - 2) + 1;
        return `${x.toFixed(1)},${toY(sample).toFixed(1)}`;
      })
      .join(" ");

  return (
    <svg
      className="sparkline"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${label} from ${lo.toExponential(1)} to ${hi.toExponential(1)}`}
    >
      {rule.tag === "at" && (
        <line
          className="sparkline-rule"
          x1={1}
          x2={width - 1}
          y1={toY(rule.value).toFixed(1)}
          y2={toY(rule.value).toFixed(1)}
        >
          <title>{rule.label}</title>
        </line>
      )}
      <polyline className="sparkline-raw" points={toPoints(data)} />
      {smoothed && smoothed.length >= 2 && (
        <polyline className="sparkline-smooth" points={toPoints(smoothed)} />
      )}
    </svg>
  );
}

function HeadBars({ fractions }: { fractions: readonly number[] }): ReactElement {
  const k = Math.max(1, fractions.length);
  const floor = 0.05 / k;
  return (
    <div className="head-bars" aria-label="Predictor-head win fractions">
      {fractions.map((fraction, index) => (
        <span
          key={index}
          className="head-bar"
          data-starved={fraction < floor ? "true" : "false"}
          title={`head ${index}: ${(fraction * 100).toFixed(1)}%`}
          style={{ height: `${Math.max(2, Math.min(18, fraction * k * 9))}px` }}
        />
      ))}
    </div>
  );
}

function healthText(health: HeadHealth): string {
  switch (health.tag) {
    case "pileup":
      return "PILEUP";
    case "separated-unresolved":
      return "SKEW · SUPPORT?";
    case "unresolved":
      return "UNPROBED";
    case "ok":
      return "OK";
  }
}

function App(): ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const telemetryHostRef = useRef<HTMLDivElement>(null);
  const handleRef = useRef<LoopHandle | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const lastStartedPieceRef = useRef<number | null>(null);
  // useState's LAZY initializer, NOT `useRef(initialDock())`: a bare useRef
  // argument is re-evaluated on EVERY render even though only the first result
  // is kept. That was invisible while ingestion was a silent localStorage read;
  // with URL ingestion it re-decoded the `?dock=` param and re-logged every
  // [dock] diagnostic on each render — five times a second, forever, once the
  // telemetry poll starts. The value never changes, so the setter is dropped.
  const [restoredDock] = useState<PersistedDock | null>(initialDock);
  const seededFromRestore = restoredDock !== null;

  // No saved dock ⇒ open on the default piece. Resolved by NAME in main.ts
  // (DEFAULT_PIECE_INDEX), never a literal here: a hardcoded 0 is what made
  // "the default" a property of GALLERY's ORDER instead of a named choice.
  const [runtime, setRuntime] = useState<RuntimeConfig>(
    () => restoredDock?.runtime ?? defaultsForPiece(DEFAULT_PIECE_INDEX)
  );

  const [particles, setParticles] = useState(
    () => restoredDock?.particles ?? 1_000
  );
  const [samples, setSamples] = useState(
    () => restoredDock?.samples ?? GALLERY[DEFAULT_PIECE_INDEX].sampleRate ?? 256
  );
  // The "train B" ceiling is a property of the LIVE trainer (device storage
  // limits × field layout × rollout K), not a constant. Seeded optimistically
  // and replaced from the handle once the loop is up; the handle clamps too, so
  // a stale value here is cosmetic, never fatal.
  const [maxSamples, setMaxSamples] = useState(TRAIN_BATCH_MAX);
  const [maxVelocity, setMaxVelocity] = useState(
    () => restoredDock?.maxVelocity ?? GALLERY[DEFAULT_PIECE_INDEX].maxVelocity
  );
  const [drive, setDrive] = useState(() => restoredDock?.drive ?? 0.65);
  const [generatorLearningRate, setGeneratorLearningRate] = useState(
    () => restoredDock?.generatorLearningRate ?? GALLERY[DEFAULT_PIECE_INDEX].learningRate
  );
  const [discriminatorLearningRate, setDiscriminatorLearningRate] = useState(
    () =>
      restoredDock?.discriminatorLearningRate ??
      GALLERY[DEFAULT_PIECE_INDEX].discriminatorLearningRate ??
      3e-3
  );
  /**
   * Pixel-critic reward. NOT restored from the persisted dock and not in
   * `PersistedDock`: the share/save schema is versioned and `advWeight` already
   * occupies the "game weight" slot with a different unit and range. The loop
   * owns the value (`?pixW=` > piece default) and the mount handler below reads
   * it back, which is the same direction every other non-persisted control uses
   * on a piece switch.
   */
  const [pixelWeight, setPixelWeight] = useState(0);
  const [resetRate, setResetRate] = useState(
    () => restoredDock?.resetRate ?? GALLERY[DEFAULT_PIECE_INDEX].resetRate
  );
  const [decay, setDecay] = useState(() => restoredDock?.decay ?? 0);
  const [look, setLook] = useState<InkLook>(
    () => restoredDock?.look ?? "ghost"
  );
  const [blend, setBlend] = useState(() => restoredDock?.blend ?? 0.5);
  // Same ladder the loop uses (main.ts resolveStrokeStyle): a saved dock is the
  // user's own last choice and outranks both, then `?stroke=` > the piece's
  // declared stroke > "dot". Calling the shared resolver — rather than
  // repeating `?? "dot"` — is what keeps the dock's first paint from showing
  // DOT while the canvas is already drawing the piece's curl.
  const [strokeStyle, setStrokeStyle] = useState<SplatStyle>(
    () =>
      restoredDock?.strokeStyle ??
      resolveStrokeStyle(GALLERY[runtime.piece], new URLSearchParams(window.location.search))
  );
  const [strokeLength, setStrokeLength] = useState(
    () =>
      restoredDock?.strokeLength ??
      resolveStrokeLength(GALLERY[runtime.piece], new URLSearchParams(window.location.search))
  );
  const [advWeight, setAdvWeight] = useState(
    () => restoredDock?.advWeight ?? 0
  );
  // The artwork is the point of the app, so on a narrow viewport BOTH panels
  // start collapsed and leave only their toggle chips over the canvas. Desktop
  // starts expanded, exactly as before. Independent state: reading the FPS
  // while the controls are shut is a normal thing to want.
  const [telemetryOpen, setTelemetryOpen] = useState(() => !hudStartsCollapsed());
  const [dockOpen, setDockOpen] = useState(() => !hudStartsCollapsed());
  const [telemetry, setTelemetry] = useState<AdversaryTelemetry>({ tag: "off" });
  const [colorMode, setColorMode] = useState<ColorMode>(
    () =>
      restoredDock?.colorMode ??
      GALLERY[DEFAULT_PIECE_INDEX].colorMode ?? { tag: "velocity" }
  );
  const [surpriseSpan, setSurpriseSpan] = useState<{
    lo: number;
    mid: number;
    hi: number;
    covered: number;
    collapsed: boolean;
  } | null>(null);
  const [payoffHistory, setPayoffHistory] = useState<number[]>([]);
  /**
   * R₁, the direction ORDER parameter — replaces the old "G residual" series.
   * That chart plotted `surprise` while "D loss" plotted `predLoss`, and those
   * are provably the SAME zero-sum scalar (the fused shader's stats[0] and
   * stats[1] are bit-identical whenever the payoff is finite), so the HUD drew
   * one curve twice and contained no instrument that could show the directional
   * collapse this row now measures.
   */
  const [orderHistory, setOrderHistory] = useState<number[]>([]);
  /**
   * PER-FAMILY payoff, one series per particle family (RGB Families piece).
   * `[]` means the running piece has no family split — the row is not drawn at
   * all rather than drawn flat, because a flat zero would read as "all three
   * families are perfectly predicted", the exact opposite of "not measured".
   *
   * A family that contributed NO tuples this batch is skipped rather than
   * pushed as 0, for the same reason (see AdvStats.perFamily.count).
   */
  const [familyHistory, setFamilyHistory] = useState<number[][]>([]);
  /**
   * AC — rms‖F − mean F‖ over a 32² grid, i.e. how much of the field is
   * spatially VARYING rather than one global push. DISPLAY ONLY on every piece
   * except Max Structure: on adversarial pieces nothing optimizes it, so a
   * rising trace is evidence the GAME is producing structure.
   */
  const [acHistory, setAcHistory] = useState<number[]>([]);
  /**
   * R₁ — DIRECTION CONVERGENCE on the same 32² grid: ‖mean of the field's soft
   * unit directions‖. This is the "has it collapsed to one direction?" series,
   * and unlike `orderHistory` it exists on EVERY piece, because the field probe
   * runs on every piece while the batch R₁ needs anti-collapse pressure
   * compiled. Sampled from the same metrics object as `acHistory`, so the two
   * traces are aligned point-for-point and can be read together: the collapse
   * signature is R₁ climbing while AC falls.
   */
  const [dirHistory, setDirHistory] = useState<number[]>([]);
  const [fieldHealth, setFieldHealth] = useState<FieldHealth>({ tag: "unprobed" });
  /** Identity of the last metrics object folded into `acHistory`. The probe
   *  mints a fresh object per sample (~1 Hz) while this poll runs at 200 ms, so
   *  without it the series would be 80% repeated points and the EMA would lag. */
  const lastFieldMetricsRef = useRef<object | null>(null);
  const [copyState, setCopyState] = useState<CopyState>({ tag: "idle" });
  const copyTimerRef = useRef<number | null>(null);

  // The flash timer outlives a fast unmount otherwise, and setState on a dead
  // component is a warning in the console of an app whose console IS its HUD.
  useEffect(
    () => () => {
      if (copyTimerRef.current !== null) window.clearTimeout(copyTimerRef.current);
    },
    []
  );

  const piece = GALLERY[runtime.piece];
  const adversary = runtime.adversaryKind !== "off";
  /**
   * The piece's declared density critic, or `undefined`. This is a PIECE
   * property, not a dock one: unlike the relational game there is no `?adv=off`
   * equivalent, and `resolvePixelCritic` (main.ts) can still drop it at startup
   * for a reason the console prints — so this gates the section, and a dropped
   * critic shows a section whose reward slider is a no-op rather than silently
   * pretending the piece has no critic at all.
   */
  const pixelCritic = piece.pixelDisc;
  /** Either game. The step-size sliders belong to whichever one is running. */
  const hasGame = adversary || pixelCritic !== undefined;
  const dockPresets = archDockPresets(piece.archDock ?? "aesthetic");
  const activeArch = (() => {
    if (!piece.fieldArch) return null;
    if (
      piece.archEditable &&
      runtime.archPreset &&
      isArchPresetKey(runtime.archPreset)
    ) {
      return applyArchDockPreset(piece.fieldArch, ARCH[runtime.archPreset]);
    }
    return piece.fieldArch;
  })();
  /**
   * The predictor the running loop is actually using. `null` means "no dock
   * override" — startLoop then keeps the spec's own, which is what
   * `piecePredictor` displays. Reported either way: the dock's job here is to
   * stop the predictor being the one model you cannot see.
   */
  const pieceAdv = piece.adversary?.tag === "on" ? piece.adversary : null;
  const piecePredictor = pieceAdv
    ? predictorArchOf(pieceAdv)
    : PREDICTOR_ARCH_DEFAULT;
  const activePredictor =
    PREDICTOR_ARCH_DOCK.find((o) => o.key === runtime.predictorPreset)?.arch ??
    null;
  const shownPredictor = activePredictor ?? piecePredictor;
  const showHeadBlend =
    (activeArch?.heads ?? piece.fieldArch?.heads ?? (piece.createField ? 2 : 1)) ===
    2;
  const isAgreeDisagree = piece.mode === "agree-disagree";
  const isWta = runtime.adversaryKind === "wta";
  const hasStructuralFieldLoss =
    !!piece.fieldLoss &&
    (piece.fieldLoss.W_CHAOS !== 0 ||
      piece.fieldLoss.W_ISO !== 0 ||
      piece.fieldLoss.W_DIV !== 0 ||
      piece.fieldLoss.W_SPIRAL !== 0 ||
      (piece.fieldLoss.W_COVER ?? 0) !== 0 ||
      (piece.fieldLoss.W_CENTER ?? 0) !== 0);
  const relationalView = viewForEncoding(runtime.encoding);

  // Re-apply the regime default when the breakpoint is CROSSED (rotation, a
  // resized window). MediaQueryList fires only when `matches` actually flips,
  // so resizing inside one regime — or a mobile URL bar sliding away — leaves
  // a manual toggle alone; only a real phone↔desktop transition overrides it.
  useEffect(() => {
    const query = window.matchMedia(HUD_COLLAPSE_QUERY);
    const onChange = (event: MediaQueryListEvent): void => {
      setTelemetryOpen(!event.matches);
      setDockOpen(!event.matches);
    };
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  // ONE canonical dock value per render. The save effect below and the share
  // buttons serialize THIS object, so a stored session and a copied link can
  // never drift apart — there is only one serializer to keep in sync.
  const dock: PersistedDock = {
    runtime,
    particles,
    samples,
    maxVelocity,
    drive,
    generatorLearningRate,
    discriminatorLearningRate,
    resetRate,
    decay,
    look,
    blend,
    strokeStyle,
    strokeLength,
    advWeight,
    colorMode,
  };

  useEffect(() => {
    savePersistedDock(dock);
    // Deps are the FIELDS, not `dock`: the object is rebuilt every render, so
    // depending on its identity would rewrite localStorage on every telemetry
    // poll (5 Hz). These are exactly the fields it is built from.
  }, [
    runtime,
    particles,
    samples,
    maxVelocity,
    drive,
    generatorLearningRate,
    discriminatorLearningRate,
    resetRate,
    decay,
    look,
    blend,
    strokeStyle,
    strokeLength,
    advWeight,
    colorMode,
  ]);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Preserve live dials only across same-piece compile rebuilds (border /
    // arch / adversary recipe change) and the first paint from localStorage.
    // Gallery switches adopt the new piece's baked particles / LRs / advWeight /
    // colorMode — otherwise Pair inherits weight=0 and raw-vector from Spiral.
    // STROKE follows that same ownership rule (the else-branch below re-reads
    // handle.getStrokeStyle()): a gallery click means "show me that piece", so
    // its declared stroke wins on switch exactly like its renderer-derived ink
    // look does two lines down. The dock control stays live afterwards, and a
    // same-piece rebuild pushes the user's current stroke back in.
    const previousPiece = lastStartedPieceRef.current;
    const isFirstStart = previousPiece === null;
    const samePieceRebuild = previousPiece === runtime.piece;
    lastStartedPieceRef.current = runtime.piece;
    const preserveLiveControls =
      (isFirstStart && seededFromRestore) || (!isFirstStart && samePieceRebuild);
    cleanupRef.current?.();
    handleRef.current = null;
    let current = true;
    cleanupRef.current = startLoop(
      canvas,
      runtime.piece,
      (handle) => {
        if (!current) return;
        handleRef.current = handle;
        // Bound the batch slider by what this trainer can actually run before
        // any restore below pushes a value through it.
        setMaxSamples(handle.getMaxSampleRate());
        if (preserveLiveControls) {
          handle.setParticleCount(particles);
          handle.setSampleRate(samples);
          // setSampleRate clamps; mirror the accepted value so the slider does
          // not sit at a number the trainer refused.
          setSamples(handle.getSampleRate());
          handle.setMaxVelocity(maxVelocity);
          handle.setDrive(drive);
          handle.setGeneratorLearningRate(generatorLearningRate);
          handle.setDiscriminatorLearningRate(discriminatorLearningRate);
          handle.setResetRate(resetRate);
          handle.setDecay(decay);
          handle.setBlend(blend);
          handle.setStrokeStyle(strokeStyle);
          handle.setStrokeLength(strokeLength);
          handle.setAdversaryWeight(advWeight);
          handle.setPixelCriticWeight(pixelWeight);
          handle.setColorMode(colorMode);
        } else {
          setParticles(handle.getParticleCount());
          setSamples(handle.getSampleRate());
          setMaxVelocity(handle.getMaxVelocity());
          setDrive(handle.getDrive());
          setGeneratorLearningRate(handle.getGeneratorLearningRate());
          setDiscriminatorLearningRate(handle.getDiscriminatorLearningRate());
          setResetRate(handle.getResetRate());
          setDecay(handle.getDecay());
          setLook(inkLookFromRenderer(piece.renderer));
          setBlend(handle.getBlend());
          setStrokeStyle(handle.getStrokeStyle());
          setStrokeLength(handle.getStrokeLength());
          setAdvWeight(handle.getAdversaryWeight());
          setPixelWeight(handle.getPixelCriticWeight());
          setColorMode(handle.getColorMode());
        }
      },
      {
        telemetryHost: telemetryHostRef.current ?? undefined,
        overrides: {
          border: runtime.border,
          adversaryEncoding: runtime.encoding,
          adversaryTarget: runtime.target,
          adversaryLoss: runtime.loss,
          k: runtime.k,
          relaxEps: runtime.relaxEps,
          fieldArch: activeArch ?? undefined,
          predictor: activePredictor ?? undefined,
        },
      }
    );

    return () => {
      current = false;
      cleanupRef.current?.();
      cleanupRef.current = null;
      handleRef.current = null;
    };
  }, [runtime]);

  useEffect(() => {
    setPayoffHistory([]);
    setOrderHistory([]);
    setAcHistory([]);
    setDirHistory([]);
    setFieldHealth({ tag: "unprobed" });
    lastFieldMetricsRef.current = null;
    setTelemetry({ tag: "off" });
    setSurpriseSpan(null);
    const poll = window.setInterval(() => {
      const handle = handleRef.current;
      if (!handle) return;
      const next = handle.getAdversaryTelemetry();
      setTelemetry(next);
      setSurpriseSpan(handle.getSurpriseSpan());
      setColorMode(handle.getColorMode());
      const health = handle.getFieldHealth();
      setFieldHealth(health);
      if (
        health.tag === "measured" &&
        (health.metrics as object) !== lastFieldMetricsRef.current
      ) {
        lastFieldMetricsRef.current = health.metrics as object;
        const ac = health.metrics.ac;
        const dir = health.metrics.r1;
        setAcHistory((previous) => previous.concat(ac).slice(-STRUCTURE_HISTORY));
        setDirHistory((previous) => previous.concat(dir).slice(-STRUCTURE_HISTORY));
      }
      if (next.tag === "on") {
        // ONE shared payoff: D minimizes it, disagreeing G maximizes it.
        setPayoffHistory((previous) =>
          previous.concat(next.surprise).slice(-SURPRISE_HISTORY)
        );
        // Only real samples enter the order series — an `unmeasured` pressure
        // must leave a gap, not a run of zeros that reads as "isotropic".
        if (next.directionOrder.tag === "measured") {
          const r1 = next.directionOrder.r1;
          setOrderHistory((previous) =>
            previous.concat(r1).slice(-SURPRISE_HISTORY)
          );
        }
        const fam = next.perFamily;
        if (fam.tag === "measured") {
          setFamilyHistory((previous) => {
            const series =
              previous.length === fam.mean.length
                ? previous
                : fam.mean.map(() => []);
            return series.map((s, c) =>
              // An unsampled family holds its series still. Charting a 0 for it
              // would draw a cliff that no measurement produced.
              fam.count[c] > 0
                ? s.concat(fam.mean[c]).slice(-SURPRISE_HISTORY)
                : s
            );
          });
        }
      }
    }, 200);
    return () => window.clearInterval(poll);
  }, [runtime]);

  const updateColor = (next: ColorMode): void => {
    handleRef.current?.setColorMode(next);
    setColorMode(next);
  };

  // Clipboard writes need a user gesture, which a click is — but they still
  // reject (permission policy, insecure context, an embed that blocks it), and
  // a share button that silently does nothing is worse than one that says so.
  const copy = (target: CopyTarget, text: string): void => {
    if (copyTimerRef.current !== null) window.clearTimeout(copyTimerRef.current);
    const flash = (state: CopyState): void => {
      setCopyState(state);
      copyTimerRef.current = window.setTimeout(
        () => setCopyState({ tag: "idle" }),
        COPY_FLASH_MS
      );
    };
    writeClipboard(text).then(
      () => flash({ tag: "copied", target }),
      (error: unknown) => {
        console.warn(`[dock] clipboard write failed — ${String(error)}`);
        flash({ tag: "failed", target });
      }
    );
  };
  const linkView = copyView(copyState, "link", "COPY LINK");
  const jsonView = copyView(copyState, "json", "JSON");

  return (
    <main className="art-shell">
      <canvas ref={canvasRef} id="myCanvas" aria-label="Neural force-field artwork" />

      <aside className="hud-stack" aria-label="Performance and piece controls">
        {/* Both panels collapse to their toggle chip and nothing else. The
            button lives OUTSIDE the collapsible region on purpose: it is the
            only affordance left once the region is hidden. */}
        <div
          className="telemetry-panel"
          data-testid="telemetry-panel"
          data-collapsed={telemetryOpen ? "false" : "true"}
        >
          <button
            type="button"
            className="hud-toggle"
            aria-expanded={telemetryOpen}
            aria-controls="telemetry-host"
            data-testid="telemetry-toggle"
            title={telemetryOpen ? "Hide telemetry" : "Show telemetry"}
            onClick={() => setTelemetryOpen((open) => !open)}
          >
            <span className="hud-toggle-title">fps · telemetry</span>
            <span className="hud-caret" aria-hidden="true">
              {telemetryOpen ? "▾" : "▸"}
            </span>
          </button>
          <div
            ref={telemetryHostRef}
            id="telemetry-host"
            className="telemetry-host"
            data-testid="telemetry-host"
            aria-live="polite"
          />
        </div>

        <div
          className="config-dock"
          id="piece-config-dock"
          data-testid="piece-config-dock"
          data-collapsed={dockOpen ? "false" : "true"}
        >
          <header className="dock-header" aria-label={`Controls for ${piece.name}`}>
            {/* aria-controls points at the dock itself: the collapsed region is
                every child EXCEPT this header (see .config-dock[data-collapsed]
                in ui.css), so the dock is the smallest element that names it.
                The title carries the full piece name because .dock-piece
                ellipsizes it — that tooltip used to live on the span. */}
            <button
              type="button"
              className="hud-toggle"
              aria-expanded={dockOpen}
              aria-controls="piece-config-dock"
              data-testid="piece-config-toggle"
              title={`${piece.name} — ${dockOpen ? "hide" : "show"} controls`}
              onClick={() => setDockOpen((open) => !open)}
            >
              <span className="dock-piece">{piece.name}</span>
              <span className="hud-caret" aria-hidden="true">
                {dockOpen ? "▾" : "▸"}
              </span>
            </button>
          </header>

          {/* Inside the collapsible region, not the header: collapsing exists
              to uncover the artwork, and a share row pinned above it would be
              the one chrome that never goes away on a phone. */}
          <ControlSection title="share" testid="share-controls">
            <div className="tui-segment-row" data-testid="share-row">
              <span className="tui-label">export</span>
              <div className="tui-segments">
                <button
                  type="button"
                  className="tui-chip"
                  data-testid="copy-link-button"
                  data-flash={linkView.flash}
                  title="Copy a link that reproduces every dial in this dock"
                  onClick={() => copy("link", dockToShareUrl(dock))}
                >
                  {linkView.label}
                </button>
                <button
                  type="button"
                  className="tui-chip share-secondary"
                  data-testid="copy-json-button"
                  data-flash={jsonView.flash}
                  title="Copy the raw settings blob — durable sharing, bug reports"
                  onClick={() => copy("json", dockToShareJson(dock))}
                >
                  {jsonView.label}
                </button>
              </div>
            </div>
            <p className="tui-note">
              ?dock= carries every dial · piece resolved by name
            </p>
          </ControlSection>

          <ControlSection title="simulation" testid="simulation-controls">
            <RangeRow
              label="particles"
              value={particleToSlider(particles)}
              min={0}
              max={1}
              step={0.002}
              display={particles.toLocaleString()}
              testid="particles-control"
              onChange={(value) => {
                const count = sliderToParticle(value);
                setParticles(count);
                handleRef.current?.setParticleCount(count);
              }}
            />
            <RangeRow
              label="train B"
              value={samples}
              min={TRAIN_BATCH_MIN}
              max={maxSamples}
              step={16}
              display={`${samples}`}
              testid="samples-control"
              onChange={(value) => {
                setSamples(value);
                handleRef.current?.setSampleRate(value);
              }}
            />
            <RangeRow
              label="max vel"
              value={maxVelocity}
              min={0.25}
              max={80}
              step={0.25}
              display={maxVelocity.toFixed(1)}
              testid="max-velocity-control"
              onChange={(value) => {
                setMaxVelocity(value);
                handleRef.current?.setMaxVelocity(value);
              }}
            />
            {adversary && piece.drive !== undefined && (
              <RangeRow
                label="drive"
                value={drive}
                min={0}
                max={1}
                step={0.01}
                display={`${drive.toFixed(2)}× clip`}
                testid="drive-control"
                onChange={(value) => {
                  setDrive(value);
                  handleRef.current?.setDrive(value);
                }}
              />
            )}
            <RangeRow
              label="respawn"
              value={resetRate}
              min={0}
              max={0.05}
              step={0.001}
              display={`${(resetRate * 100).toFixed(1)}%`}
              testid="random-reset-control"
              onChange={(value) => {
                setResetRate(value);
                handleRef.current?.setResetRate(value);
              }}
            />
            <Segmented
              label="border"
              value={runtime.border.tag}
              testid="border-control"
              choices={[
                { value: "wrap", label: "WRAP", title: "Periodic torus" },
                { value: "bounce", label: "BOUNCE", title: "Reflect at the box edge" },
                { value: "reset", label: "RESET", title: "Respawn when leaving the box" },
              ]}
              onChange={(tag) =>
                setRuntime((previous) => ({ ...previous, border: { tag } }))
              }
            />
            <p className="tui-note restart-note">border is compiled · changing it restarts</p>
          </ControlSection>

          <ControlSection title="display" testid="display-controls">
            <ControlSubsection title="ink" testid="ink-controls">
            {piece.lookEditable && (
              <Segmented
                label="look"
                value={look}
                testid="ink-look-control"
                choices={[
                  { value: "ghost" as const, label: "GHOST", title: "Soft alpha trails" },
                  { value: "clean" as const, label: "CLEAN", title: "No trails" },
                  {
                    value: "trails" as const,
                    label: "TRAILS",
                    title: "Long streak persistence",
                  },
                ]}
                onChange={(value) => {
                  const d = INK_LOOK_DECAY[value];
                  setLook(value);
                  setDecay(d);
                  handleRef.current?.setDecay(d);
                }}
              />
            )}
            <RangeRow
              label="trails"
              value={decay}
              min={0}
              max={0.99}
              step={0.005}
              display={decay.toFixed(2)}
              testid="trails-control"
              onChange={(value) => {
                setDecay(value);
                handleRef.current?.setDecay(value);
              }}
            />
            <Segmented
              label="stroke"
              value={strokeStyle}
              testid="stroke-style-control"
              choices={[
                { value: "dot", label: "DOT" },
                { value: "vel", label: "VEL" },
                { value: "curl", label: "CURL" },
              ]}
              onChange={(value) => {
                setStrokeStyle(value);
                handleRef.current?.setStrokeStyle(value);
              }}
            />
            {strokeStyle !== "dot" && (
              <RangeRow
                label="length"
                value={strokeLength}
                min={0.5}
                max={16}
                step={0.5}
                display={strokeLength.toFixed(1)}
                testid="stroke-length-control"
                onChange={(value) => {
                  setStrokeLength(value);
                  handleRef.current?.setStrokeLength(value);
                }}
              />
            )}
            </ControlSubsection>

            {adversary && (
              <ControlSubsection title="color" testid="color-controls">
                {isAgreeDisagree ? (
                  <div className="tui-static-row" data-testid="color-mode-control">
                    <span>color</span>
                    <strong>RGB · A / B / derived C</strong>
                  </div>
                ) : (
                  <>
                    <Segmented
                      label="color"
                      value={colorMode.tag}
                      testid="color-mode-control"
                      choices={[
                        {
                          value: "velocity",
                          label: "VEL",
                          title:
                            "Velocity color maps each particle's current post-force speed; it changes with the rendered motion, not the ink shape.",
                        },
                        {
                          value: "surprise-raw",
                          label: "RAW",
                          title:
                            "Raw payoff color shows the absolute adversarial payoff. It can look particular because magnitude and local scale dominate; stroke geometry still comes from the ink controls.",
                        },
                        {
                          value: "surprise-per-unit",
                          label: "PER UNIT",
                          title:
                            "Per-unit payoff normalizes payoff by the local force/velocity scale, emphasizing relative surprise. It can look particular while VEL/CURL still affect ink geometry.",
                        },
                      ]}
                      onChange={(tag) =>
                        updateColor(
                          tag === "velocity"
                            ? { tag: "velocity" }
                            : {
                                tag,
                                colormap:
                                  colorMode.tag !== "velocity" ? colorMode.colormap : "inferno",
                              }
                        )
                      }
                    />
                    <p className="tui-note color-mode-note">
                      VEL colors motion; RAW / PER UNIT color payoff. Ink geometry is separate.
                    </p>
                  </>
                )}
                {!isAgreeDisagree && colorMode.tag !== "velocity" && (
                  <Segmented
                    label="map"
                    value={colorMode.colormap}
                    testid="colormap-control"
                    choices={CMAPS.map((name) => ({ value: name, label: name.toUpperCase() }))}
                    onChange={(colormap) => updateColor({ ...colorMode, colormap })}
                  />
                )}
                {piece.palette === "optimizer-groups-v1" && (
                  <p className="tui-note color-mode-note" data-testid="special-color-legend">
                    SPECIAL · cyan shared · pink head A · gold head B · brightness = speed
                  </p>
                )}
              </ControlSubsection>
            )}
          </ControlSection>

          {piece.fieldArch && (
            <ControlSection title="model" testid="model-arch-controls">
              <div className="tui-note" data-testid="model-arch-summary">
                {describeFieldArch(activeArch ?? piece.fieldArch)}
                {!piece.fieldLoss ? " · train tfjs" : ""}
              </div>
              {piece.archEditable && (
                <Segmented
                  label="arch"
                  value={(runtime.archPreset ?? "default") as string}
                  choices={[
                    { value: "default", label: "default" },
                    ...dockPresets.map((preset) => ({
                      value: preset.key,
                      label: preset.label,
                    })),
                  ]}
                  onChange={(value) =>
                    setRuntime((r) => ({
                      ...r,
                      archPreset:
                        value === "default"
                          ? null
                          : (value as ArchPresetKey),
                    }))
                  }
                  testid="model-arch-presets"
                />
              )}
              {piece.archEditable && (
                <p className="tui-note restart-note">arch is compiled · changing it restarts</p>
              )}
              {/* THE SECOND MODEL. A relational piece trains two nets and the
                  dock named only one, so "what model is this piece?" had no
                  answer for the predictor — which is also the net whose width
                  had never been varied (every reading in agent_notes/ is
                  32/16). Shown for every adversary piece, editable on all of
                  them: the fused codegen is general over these dims and only
                  refuses `sin`, which no predictor uses. */}
              {adversary && (
                <>
                  <div className="tui-note" data-testid="model-predictor-summary">
                    {`predictor · ${runtime.k} × [${shownPredictor.hiddenUnits}, ` +
                      `${shownPredictor.featureDim}] · selu`}
                  </div>
                  <Segmented
                    label="pred"
                    value={runtime.predictorPreset ?? "default"}
                    choices={[
                      { value: "default", label: "default" },
                      ...PREDICTOR_ARCH_DOCK.map((o) => ({
                        value: o.key,
                        label: o.label,
                      })),
                    ]}
                    onChange={(value) =>
                      setRuntime((r) => ({
                        ...r,
                        predictorPreset: value === "default" ? null : value,
                      }))
                    }
                    testid="model-predictor-presets"
                  />
                  <p className="tui-note restart-note">
                    predictor is compiled · changing it restarts
                  </p>
                </>
              )}
            </ControlSection>
          )}

          {showHeadBlend && (
            <ControlSection
              title={isAgreeDisagree ? "A/B/C roles" : "two-head field"}
              testid="field-controls"
            >
              <RangeRow
                label={isAgreeDisagree ? "blend C" : "blend A/B"}
                value={blend}
                min={0}
                max={1}
                step={0.01}
                display={blend.toFixed(2)}
                testid="head-blend-control"
                onChange={(value) => {
                  setBlend(value);
                  handleRef.current?.setBlend(value);
                }}
              />
              {isAgreeDisagree ? (
                <div className="rgb-role-legend" data-testid="rgb-role-legend">
                  <span className="role-a">R · A disagree</span>
                  <span className="role-b">G · B agree</span>
                  <span className="role-c">B · C blend (no loss)</span>
                </div>
              ) : (
                <p className="tui-note">neutral output mix — not order ↔ chaos</p>
              )}
            </ControlSection>
          )}

          {/* THE TWO STEP SIZES ARE SHARED, so they live in ONE section rather
              than inside the relational game's. `discriminatorLearningRate` is
              the `lr` uniform of BOTH fused critics and `generatorLearningRate`
              is the field trainer's for every piece — the values already
              reached a Pixel piece through `?gLR=`/`?dLR=`; only these sliders
              were behind `adversary &&`, which is a RELATIONAL predicate
              (`runtime.adversaryKind !== "off"`) standing in for "has a game". */}
          {hasGame && (
            <ControlSection title="training" testid="game-training-controls">
              <RangeRow
                label="G lr"
                value={learningRateToSlider(
                  generatorLearningRate,
                  G_LR_MIN,
                  G_LR_MAX
                )}
                min={0}
                max={1}
                step={0.005}
                display={generatorLearningRate.toExponential(1)}
                testid="generator-learning-rate-control"
                onChange={(value) => {
                  const next = sliderToLearningRate(value, G_LR_MIN, G_LR_MAX);
                  setGeneratorLearningRate(next);
                  handleRef.current?.setGeneratorLearningRate(next);
                }}
              />
              <RangeRow
                label="D lr"
                value={learningRateToSlider(
                  discriminatorLearningRate,
                  D_LR_MIN,
                  D_LR_MAX
                )}
                min={0}
                max={1}
                step={0.005}
                display={discriminatorLearningRate.toExponential(1)}
                testid="discriminator-learning-rate-control"
                onChange={(value) => {
                  const next = sliderToLearningRate(value, D_LR_MIN, D_LR_MAX);
                  setDiscriminatorLearningRate(next);
                  handleRef.current?.setDiscriminatorLearningRate(next);
                }}
              />
              <div className="tui-static-row" data-testid="learning-rate-ratio">
                <span>D / G</span>
                <strong>
                  {(discriminatorLearningRate / generatorLearningRate).toFixed(2)}×
                </strong>
              </div>
              {piece.generatorLearningRates?.tag === "shared-heads" && (
                <div className="tui-note" data-testid="generator-lr-groups">
                  G groups · shared {piece.generatorLearningRates.shared.toExponential(1)} ·
                  A {piece.generatorLearningRates.head0.toExponential(1)} ·
                  B {piece.generatorLearningRates.head1.toExponential(1)}
                </div>
              )}
            </ControlSection>
          )}

          {pixelCritic && (
            <ControlSection title="pixel critic" testid="pixel-critic-controls">
              <div className="tui-static-row" data-testid="pixel-critic-summary">
                <span>{pixelCritic.kind ?? "vec-field"}</span>
                <strong>{describePixelCritic(pixelCritic)}</strong>
              </div>
              {/* Its own range, NOT the adversary's 0..20: this multiplies a
                  soft-density residual, not a WTA payoff in predictor-output
                  units (main.ts PIXEL_CRITIC_WEIGHT_RANGE). The shipped pieces
                  carry fieldLoss: ZERO_FIELD_LOSS, so 0 here means the field
                  gets NO gradient — a real diagnostic state, and the reason the
                  row shows the value rather than only a slider position. */}
              <RangeRow
                label="reward"
                value={pixelWeight}
                min={PIXEL_CRITIC_WEIGHT_RANGE.min}
                max={PIXEL_CRITIC_WEIGHT_RANGE.max}
                step={0.005}
                display={pixelWeight.toFixed(3)}
                testid="pixel-critic-reward-control"
                onChange={(value) => {
                  setPixelWeight(value);
                  handleRef.current?.setPixelCriticWeight(value);
                }}
              />
              {pixelWeight === 0 && (
                <p className="tui-note" data-testid="pixel-critic-zero-note">
                  reward 0 · critic trains, field does not (ZERO_FIELD_LOSS piece)
                </p>
              )}
            </ControlSection>
          )}

          {adversary && (
            <ControlSection title="adversary" testid="adversary-controls">
              <div className="tui-static-row" data-testid="objective-contract">
                <span>objective</span>
                <strong>
                  {runtime.loss.tag === "angle-scale-hold"
                    ? "ANGLE OPPOSE · SCALE AGREE"
                    : runtime.loss.tag === "angle-relative-scale"
                      ? "ANGLE + SCALE OPPOSE · ENERGY HOLD"
                    : isAgreeDisagree
                      ? "A OPPOSE · B COOPERATE"
                      : hasStructuralFieldLoss
                        ? "GAME + MAX CHAOS"
                        : runtime.loss.tag === "raw-vector"
                          ? "RAW VECTOR · BASELINE"
                          : "ANGLE OPPONENT GAME"}
                </strong>
              </div>
              <Segmented
                label="target"
                value={runtime.target.tag}
                testid="adversary-target-control"
                choices={[
                  {
                    value: "force",
                    label: "FORCE",
                    title: "Predict raw neural field output F(x)",
                  },
                  {
                    value: "post-velocity",
                    label: "POST-V",
                    title:
                      "Predict normalized velocity after force, friction and clip; context includes incoming velocity",
                  },
                ]}
                onChange={(tag) =>
                  setRuntime((previous) => {
                    const target: AdversaryTarget = { tag };
                    if (tag === "post-velocity") {
                      return {
                        ...previous,
                        target,
                        encoding: { tag: "point" },
                        loss: lossHasScale(previous.loss)
                          ? lossWithTag("soft-angle", previous.loss)
                          : previous.loss,
                      };
                    }
                    return { ...previous, target };
                  })
                }
              />
              <Segmented
                label="loss"
                value={runtime.loss.tag}
                testid="adversary-loss-control"
                choices={[
                  {
                    value: "raw-vector",
                    label: "RAW",
                    title:
                      "Euclidean ‖ŷ−y‖ — easy baseline (amplitude shortcut OK). Compare to ANGLE.",
                  },
                  {
                    value: "soft-angle",
                    label: "ANGLE",
                    title: "Exact smooth S² chord loss with bounded Jacobian",
                  },
                  {
                    value: "angle-relative-scale",
                    label: "A+S ADV",
                    title: "Adversarial direction and relative magnitude contrast",
                  },
                  {
                    value: "angle-scale-hold",
                    label: "A+S HOLD",
                    title:
                      "Direction adversarial; relative scale cooperative; absolute energy held",
                  },
                ]}
                onChange={(tag) =>
                  setRuntime((previous) => {
                    const nextLoss = lossWithTag(tag, previous.loss);
                    if (lossHasScale(nextLoss)) {
                      return {
                        ...previous,
                        target: { tag: "force" },
                        encoding:
                          previous.encoding.tag === "point"
                            ? { tag: "pair-rotation-scale-adjusted" }
                            : previous.encoding,
                        loss: nextLoss,
                      };
                    }
                    return { ...previous, loss: nextLoss };
                  })
                }
              />
              {lossHasAngle(runtime.loss) && (
                <RangeRow
                  label="soft τ"
                  value={runtime.loss.tau}
                  min={0.005}
                  max={0.25}
                  step={0.005}
                  display={runtime.loss.tau.toFixed(3)}
                  testid="adversary-angle-tau-control"
                  onChange={(tau) =>
                    setRuntime((previous) => ({
                      ...previous,
                      loss:
                        previous.loss.tag === "raw-vector"
                          ? previous.loss
                          : { ...previous.loss, tau },
                    }))
                  }
                />
              )}
              {lossHasScale(runtime.loss) && (
                <>
                  <RangeRow
                    label="scale w"
                    value={runtime.loss.scaleWeight}
                    min={0}
                    max={2}
                    step={0.05}
                    display={runtime.loss.scaleWeight.toFixed(2)}
                    testid="adversary-scale-weight-control"
                    onChange={(scaleWeight) =>
                      setRuntime((previous) => ({
                        ...previous,
                        loss: lossHasScale(previous.loss)
                          ? { ...previous.loss, scaleWeight }
                          : previous.loss,
                      }))
                    }
                  />
                  <RangeRow
                    label="energy w"
                    value={runtime.loss.energyWeight}
                    min={0}
                    max={1}
                    step={0.01}
                    display={runtime.loss.energyWeight.toFixed(2)}
                    testid="adversary-energy-weight-control"
                    onChange={(energyWeight) =>
                      setRuntime((previous) => ({
                        ...previous,
                        loss: lossHasScale(previous.loss)
                          ? { ...previous.loss, energyWeight }
                          : previous.loss,
                      }))
                    }
                  />
                  <RangeRow
                    label="energy"
                    value={runtime.loss.energyTarget}
                    min={0.02}
                    max={1}
                    step={0.01}
                    display={runtime.loss.energyTarget.toFixed(2)}
                    testid="adversary-energy-target-control"
                    onChange={(energyTarget) =>
                      setRuntime((previous) => ({
                        ...previous,
                        loss: lossHasScale(previous.loss)
                          ? { ...previous.loss, energyTarget }
                          : previous.loss,
                      }))
                    }
                  />
                </>
              )}
              {runtime.target.tag === "post-velocity" && (
                <p className="tui-note" data-testid="post-velocity-contract">
                  context = x + incoming v · target is pre-border normalized v+
                </p>
              )}
              {lossHasScale(runtime.loss) && (
                <p className="tui-note" data-testid="relative-scale-contract">
                  scale = within-tuple contrast · energy fixes absolute RMS
                </p>
              )}
              {runtime.loss.tag === "angle-scale-hold" && (
                <p className="tui-note" data-testid="scale-hold-diagnostic-contract">
                  displayed D-joint is not G reward · G raises angle, lowers scale error
                </p>
              )}
              <Segmented
                label="tuple"
                value={tupleViewForEncoding(runtime.encoding)}
                testid="adversary-tuple-control"
                choices={[
                  { value: "point", label: "1 · POINT" },
                  { value: "pair", label: "2 · PAIR" },
                  { value: "tri", label: "3 · TRI" },
                  {
                    value: "quad-labelled",
                    label: "4L · QUAD",
                    title: "Four labelled points; translation+rotation quotient only",
                  },
                ]}
                onChange={(tuple) =>
                  setRuntime((previous) => {
                    const encoding = encodingForTuple(tuple, previous.encoding);
                    return {
                      ...previous,
                      encoding,
                      target:
                        tuple === "point"
                          ? previous.target
                          : { tag: "force" },
                      loss:
                        tuple === "point" && lossHasScale(previous.loss)
                          ? lossWithTag("soft-angle", previous.loss)
                          : previous.loss,
                    };
                  })
                }
              />
              {relationalView ? (
                <Segmented
                  label="observer"
                  value={relationalView}
                  testid="adversary-view-control"
                  choices={[
                    {
                      value: "rotation",
                      label: "R",
                      title: "Quotient translation and rotation; keep scale observable",
                    },
                    {
                      value: "rotation-scale-raw",
                      label: "R+S RAW",
                      title:
                        "Scale-blind context; selecting it also chooses the raw-vector cheat control",
                    },
                    {
                      value: "rotation-scale-adjusted",
                      label: "R+S ADJ",
                      title:
                        "Same scale-blind context; selecting it chooses exact smooth soft-angle",
                    },
                  ]}
                  onChange={(view) =>
                    setRuntime((previous) => ({
                      ...previous,
                      encoding: encodingForView(view),
                      loss:
                        view === "rotation-scale-raw"
                          ? lossWithTag("raw-vector", previous.loss)
                          : view === "rotation-scale-adjusted"
                            ? lossWithTag("soft-angle", previous.loss)
                            : previous.loss,
                    }))
                  }
                />
              ) : (
                <div className="tui-static-row" data-testid="adversary-view-control">
                  <span>observer</span>
                  <strong>{observerLabel(runtime.encoding, runtime.target)}</strong>
                </div>
              )}
              <RangeRow
                label={runtime.loss.tag === "angle-scale-hold" ? "game w" : "reward"}
                value={advWeight}
                min={ADV_WEIGHT_MIN}
                max={ADV_WEIGHT_MAX}
                step={0.0005}
                display={advWeight.toFixed(3)}
                testid="adversary-reward-control"
                onChange={(value) => {
                  setAdvWeight(value);
                  handleRef.current?.setAdversaryWeight(value);
                }}
              />
              {isWta ? (
                <>
                  <RangeRow
                    label="guesses K"
                    value={runtime.k}
                    min={2}
                    max={12}
                    step={1}
                    display={`${runtime.k}`}
                    testid="adversary-k-control"
                    onChange={(value) =>
                      setRuntime((previous) => ({ ...previous, k: Math.round(value) }))
                    }
                  />
                  <RangeRow
                    label="relax ε"
                    value={runtime.relaxEps}
                    min={0}
                    max={0.45}
                    step={0.01}
                    display={runtime.relaxEps.toFixed(2)}
                    testid="adversary-epsilon-control"
                    onChange={(value) =>
                      setRuntime((previous) => ({ ...previous, relaxEps: value }))
                    }
                  />
                </>
              ) : (
                <div className="tui-static-row">
                  <span>predictor</span>
                  <strong>single-head control</strong>
                </div>
              )}
              <p className="tui-note restart-note">
                target, loss, tuple, observer, K and ε rebuild GPU pipelines
              </p>
            </ControlSection>
          )}

          {(adversary || fieldHealth.tag === "measured") && (
            <ControlSection title="diagnostics" testid="adversary-diagnostics">
              <div className="diagnostic-row" data-testid="field-structure-chart">
                <span
                  className="diagnostic-name"
                  title={
                    "AC = rms‖F − mean F‖ on a 32² grid — the spatially VARYING" +
                    " part of the field. DC = ‖mean F‖ is the constant push that" +
                    " survives when a field collapses. HIGH AC IS STRUCTURE;" +
                    " AC below ~1e-4 is a dead field and DC/AC above ~3 is the" +
                    " laminar end state (measured 3.7 at full collapse, 0.25 with" +
                    " the anti-collapse pressure working). On every piece but" +
                    " Max Structure this is OBSERVED, never optimized — a rising" +
                    " trace here is evidence the GAME made the structure."
                  }
                >
                  AC structure
                </span>
                <Sparkline
                  label="field AC"
                  data={acHistory}
                  smoothed={emaSeries(acHistory, HISTORY_SMOOTH)}
                />
                <strong>
                  {fieldHealth.tag === "measured"
                    ? fieldHealth.metrics.ac.toExponential(2)
                    : "—"}
                </strong>
              </div>
              <div className="diagnostic-row" data-testid="field-direction-chart">
                <span
                  className="diagnostic-name"
                  title={
                    "DIRECTION CONVERGENCE. R₁ = ‖mean of the field's soft unit" +
                    " directions‖ over the same 32² grid as AC. 0 = the" +
                    " directions are isotropic (a field of vortices), 1 = every" +
                    " sample points the same way (laminar streaks — the collapse" +
                    " this measures). LOW IS HEALTHY: 0.99 with the anti-collapse" +
                    " pressure off on the default piece, 0.01–0.10 with it on." +
                    " Fixed 0–1 scale; the dashed rule is the 0.10 healthy" +
                    " ceiling, so a trace hugging the floor is the good case." +
                    " READ IT WITH R₂ in the line below — a field that splits" +
                    " into ± counter-streaming sheets scores R₁ ≈ 0 and looks" +
                    " just as laminar. This one is measured on EVERY piece; the" +
                    " adversary's `R1 batch` row is the same statistic over the" +
                    " training batch and only exists under ?advPolar."
                  }
                >
                  R1 direction
                </span>
                <Sparkline
                  label="field direction order R1"
                  data={dirHistory}
                  smoothed={emaSeries(dirHistory, HISTORY_SMOOTH)}
                  scale={{ tag: "fixed", lo: 0, hi: 1 }}
                  rule={{ tag: "at", value: R1_HEALTHY_CEILING, label: "healthy ceiling 0.10" }}
                />
                <strong>
                  {fieldHealth.tag === "measured"
                    ? fieldHealth.metrics.r1.toFixed(3)
                    : "—"}
                </strong>
              </div>
              <div className="chart-legend" data-testid="field-structure-detail">
                {fieldHealth.tag === "measured"
                  ? `DC/AC ${(
                      fieldHealth.metrics.dc / Math.max(fieldHealth.metrics.ac, 1e-12)
                    ).toFixed(2)} · sat ${fieldHealth.metrics.satFrac.toFixed(
                      2
                    )} · OW ${fieldHealth.metrics.okuboWeiss.toFixed(
                      2
                    )} · R2 ${fieldHealth.metrics.r2.toFixed(2)} · R1 0=swirl 1=laminar`
                  : "field probe warming"}
              </div>
              {telemetry.tag === "on" ? (
                <>
                  <div className="diagnostic-row" data-testid="disc-loss-chart">
                    <span
                      className="diagnostic-name"
                      title={
                        "Shared relaxed-WTA payoff — D minimizes it, disagreeing G maximizes it." +
                        (telemetry.payoffReference.tag === "north-pole"
                          ? ` The dashed rule is √2 ≈ ${telemetry.payoffReference.chord.toFixed(4)}:` +
                            " under soft-angle that is the chord from the embedding's north pole" +
                            " ψτ(0) to every equatorial one. READ IT WITH R1 — the payoff sits" +
                            " there both when the target has gone to zero (a flat field: R1 → 1)" +
                            " and when the field is so isotropic that D cannot predict it at all" +
                            " (R1 → 0, which is the good case and is transient while D relearns)." +
                            " Steady state: 0.48–0.7."
                          : "")
                      }
                    >
                      payoff
                    </span>
                    <Sparkline
                      label="shared payoff"
                      data={payoffHistory}
                      smoothed={emaSeries(payoffHistory, HISTORY_SMOOTH)}
                      rule={
                        telemetry.payoffReference.tag === "north-pole"
                          ? {
                              tag: "at",
                              value: telemetry.payoffReference.chord,
                              label: "√2 — north-pole collapse",
                            }
                          : { tag: "none" }
                      }
                    />
                    <strong>{telemetry.surprise.toExponential(2)}</strong>
                  </div>
                  {/* The BATCH twin of the always-on `field-direction-chart`
                      above: same statistic, same τ, sampled at the particles
                      the adversary actually trained on instead of on a uniform
                      grid. It is drawn only when the pressure is compiled —
                      before the grid R₁ existed this row rendered a permanent
                      "—" on every piece without ?advPolar, which is a dead row,
                      not a measurement. */}
                  {telemetry.directionOrder.tag === "measured" && (
                  <div className="diagnostic-row" data-testid="direction-order-chart">
                    <span
                      className="diagnostic-name"
                      title={
                        "R₁ = ‖mean of the field's soft-unit directions‖ over the training" +
                        " BATCH — the particles the adversary trained on this step, where" +
                        " the grid row above samples the domain uniformly. They diverge when" +
                        " the cloud has bunched into part of the domain, and that gap is" +
                        " itself the reading. 0 = isotropic (swirls), 1 = one global" +
                        " direction (laminar streaks). LOW IS HEALTHY — measured live on the" +
                        " default piece: 0.99 with the pressure off (diagonal streaks)," +
                        " 0.01–0.10 with it on (a field of vortices)."
                      }
                    >
                      R1 batch
                    </span>
                    <Sparkline
                      label="direction order R1"
                      data={orderHistory}
                      smoothed={emaSeries(orderHistory, HISTORY_SMOOTH)}
                      scale={{ tag: "fixed", lo: 0, hi: 1 }}
                      rule={{ tag: "at", value: R1_HEALTHY_CEILING, label: "healthy ceiling 0.10" }}
                    />
                    <strong>{telemetry.directionOrder.r1.toFixed(3)}</strong>
                  </div>
                  )}
                  <FamilyPayoffRow
                    payoff={telemetry.perFamily}
                    series={familyHistory}
                  />
                  <div className="chart-legend" aria-hidden="true">
                    dim=raw · bright=EMA · ~{Math.round((SURPRISE_HISTORY * 0.2) / 60)}m window
                    {telemetry.directionOrder.tag === "measured"
                      ? " · R1 batch vs R1 grid above"
                      : " · batch R1 needs ?advPolar"}
                  </div>
                  <div className="diagnostic-row">
                    <span>heads</span>
                    <HeadBars fractions={telemetry.winFractions} />
                    <strong
                      className={`health health-${telemetry.health.tag}`}
                      data-testid="head-health"
                    >
                      {healthText(telemetry.health)}
                    </strong>
                  </div>
                  {telemetry.branches && (
                    <>
                      <div className="diagnostic-row" data-testid="disagree-head-health">
                        <span>A disagree</span>
                        <HeadBars fractions={telemetry.branches.disagree.winFractions} />
                        <strong
                          className={`health health-${telemetry.branches.disagree.health.tag}`}
                        >
                          {healthText(telemetry.branches.disagree.health)}
                        </strong>
                      </div>
                      <div className="diagnostic-row" data-testid="agree-head-health">
                        <span>B agree</span>
                        <HeadBars fractions={telemetry.branches.agree.winFractions} />
                        <strong
                          className={`health health-${telemetry.branches.agree.health.tag}`}
                        >
                          {healthText(telemetry.branches.agree.health)}
                        </strong>
                      </div>
                    </>
                  )}
                </>
              ) : (
                // "initialising…" is a claim about a GAME. A piece with no
                // adversary has nothing to initialise, and printing it there
                // would read as a stuck adversary rather than an absent one.
                adversary && (
                  <div className="tui-static-row">
                    <span>game</span>
                    <strong>initialising…</strong>
                  </div>
                )
              )}
              {colorMode.tag !== "velocity" && surpriseSpan && (
                <div className="span-readout" data-testid="surprise-span">
                  {colorMode.tag === "surprise-raw" ? "RAW" : "PER UNIT"} · p2{" "}
                  {surpriseSpan.lo.toExponential(1)} · p98{" "}
                  {surpriseSpan.hi.toExponential(1)} ·{" "}
                  {Math.round(surpriseSpan.covered * 100)}%
                  {surpriseSpan.collapsed ? " · FLAT" : ""}
                </div>
              )}
            </ControlSection>
          )}
        </div>
      </aside>

      <nav
        className="gallery-radio"
        role="radiogroup"
        aria-label="Art piece"
        data-testid="art-piece-gallery"
      >
        {GALLERY.map((galleryPiece, index) => (
          <button
            key={galleryPiece.name}
            type="button"
            role="radio"
            aria-checked={index === runtime.piece}
            data-active={index === runtime.piece ? "true" : "false"}
            onClick={() =>
              setRuntime((previous) => runtimeForPieceSwitch(previous, index))
            }
          >
            {galleryPiece.name}
          </button>
        ))}
      </nav>
    </main>
  );
}

createRoot(container).render(<App />);
