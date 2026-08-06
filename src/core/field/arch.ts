/**
 * Field architecture — orthogonal to loss, renderer, and adversary game.
 *
 * Gallery pieces pick a {@link FieldArch} (or a named {@link ARCH} preset).
 * Encoding, depth/width, activation, and head count are independent knobs;
 * "HelmholtzField" is only the dual-vector ForceField runtime that implements
 * them (legacy name — not a Helmholtz decomposition).
 */

import { HelmholtzField, type HelmholtzFieldConfig } from "./helmholtz";

/** How position is turned into the MLP input. */
export type FieldEncodingKind = "raw" | "fourier" | "hashgrid";

/** Hidden-layer family. `sin` ⇒ SIREN; pairs with raw or Fourier encoding. */
export type FieldHiddenAct = "selu" | "sin";

/**
 * Declarative force-field architecture.
 *
 * - `encoding` + `activation` map onto Helmholtz `modelType`
 * - `heads: 1` → single vector net (α forced to 0; fused `FieldSpec.kind: "vector"`)
 * - `heads: 2` → blend `(1-α)·A + α·B` (Agree+Disagree / dual games)
 */
export interface FieldArch {
  /** Position encoding. Default `"raw"`. */
  encoding?: FieldEncodingKind;
  /** Hidden activation. Default `"selu"`; `"sin"` selects SIREN. */
  activation?: FieldHiddenAct;
  /** Hidden widths. Default `[32, 32]`. */
  hiddenUnits?: number[];
  /** 1 = single force MLP; 2 = dual-head blend. Default 2. */
  heads?: 1 | 2;
  /** Blend when `heads: 2`. Ignored for single-head (forced 0). Default 0.55. */
  alpha?: number;
  fourierOctaves?: number;
  gridSize?: number;
  gridFeatures?: number;
  sirenOmega0?: number;
  classes?: number;
  semantic?: "blend" | "agree-disagree";
}

export interface ArchPreset {
  readonly key: string;
  readonly label: string;
  readonly arch: FieldArch;
}

/** Named presets for gallery defaults and the model dock. */
export const ARCH = {
  /** Real depth — replaces the old `mlpWide` (256→2) one-liner. */
  mlp256: {
    encoding: "raw",
    activation: "selu",
    hiddenUnits: [256, 256],
    heads: 1,
    alpha: 0,
  },
  mlpDeep: {
    encoding: "raw",
    activation: "selu",
    hiddenUnits: [64, 128, 128, 64],
    heads: 1,
    alpha: 0,
  },
  mlpShallow: {
    encoding: "raw",
    activation: "selu",
    hiddenUnits: [32, 64],
    heads: 1,
    alpha: 0,
  },
  fourier: {
    encoding: "fourier",
    activation: "selu",
    hiddenUnits: [64, 64],
    heads: 1,
    alpha: 0,
    fourierOctaves: 4,
  },
  fourierWide: {
    encoding: "fourier",
    activation: "selu",
    hiddenUnits: [128, 128],
    heads: 1,
    alpha: 0,
    fourierOctaves: 4,
  },
  /** Fourier features + SIREN hidden (encoding × activation). */
  fourierSiren: {
    encoding: "fourier",
    activation: "sin",
    hiddenUnits: [64, 64],
    heads: 1,
    alpha: 0,
    fourierOctaves: 4,
    sirenOmega0: 6,
  },
  siren: {
    encoding: "raw",
    activation: "sin",
    hiddenUnits: [64, 64],
    heads: 1,
    alpha: 0,
    sirenOmega0: 6,
  },
  hashgrid: {
    encoding: "hashgrid",
    activation: "selu",
    hiddenUnits: [64, 64],
    heads: 1,
    alpha: 0,
    gridSize: 32,
    gridFeatures: 4,
  },
  /** Dual-head chaos / adversary default. */
  dualStd: {
    encoding: "raw",
    activation: "selu",
    hiddenUnits: [32, 32],
    heads: 2,
    alpha: 0.7,
  },
  dualFourier: {
    encoding: "fourier",
    activation: "selu",
    hiddenUnits: [32, 32],
    heads: 2,
    alpha: 0.55,
    fourierOctaves: 4,
  },
  dualSiren: {
    encoding: "raw",
    activation: "sin",
    hiddenUnits: [32, 32],
    heads: 2,
    alpha: 0.7,
    sirenOmega0: 6,
  },
  dualHashgrid: {
    encoding: "hashgrid",
    activation: "selu",
    hiddenUnits: [32, 32],
    heads: 2,
    alpha: 0.7,
    gridSize: 32,
    gridFeatures: 4,
  },
} as const satisfies Record<string, FieldArch>;

export type ArchPresetKey = keyof typeof ARCH;

/** Presets offered in the dock for aesthetic (single-head) pieces. */
export const ARCH_DOCK_PRESETS: readonly ArchPreset[] = [
  { key: "mlpShallow", label: "MLP shallow", arch: ARCH.mlpShallow },
  { key: "mlp256", label: "MLP 256×2", arch: ARCH.mlp256 },
  { key: "mlpDeep", label: "MLP deep", arch: ARCH.mlpDeep },
  { key: "fourier", label: "Fourier", arch: ARCH.fourier },
  { key: "fourierWide", label: "Fourier wide", arch: ARCH.fourierWide },
  { key: "fourierSiren", label: "Fourier+SIREN", arch: ARCH.fourierSiren },
  { key: "siren", label: "SIREN", arch: ARCH.siren },
  { key: "hashgrid", label: "HashGrid", arch: ARCH.hashgrid },
] as const;

/** Presets for dual-head chaos pieces (encoding A/B, same max-chaos loss). */
export const ARCH_DOCK_DUAL: readonly ArchPreset[] = [
  { key: "dualStd", label: "Dual MLP", arch: ARCH.dualStd },
  { key: "dualFourier", label: "Dual Fourier", arch: ARCH.dualFourier },
  { key: "dualSiren", label: "Dual SIREN", arch: ARCH.dualSiren },
  { key: "dualHashgrid", label: "Dual HashGrid", arch: ARCH.dualHashgrid },
] as const;

export type ArchDockKind = "aesthetic" | "dual";

export function archDockPresets(kind: ArchDockKind = "aesthetic"): readonly ArchPreset[] {
  return kind === "dual" ? ARCH_DOCK_DUAL : ARCH_DOCK_PRESETS;
}

/**
 * Apply a dock preset onto a piece's base arch, preserving game-load-bearing
 * knobs (α, semantic, classes) while swapping encoding / widths / activation.
 */
export function applyArchDockPreset(base: FieldArch, preset: FieldArch): FieldArch {
  return {
    ...preset,
    alpha: base.alpha ?? preset.alpha,
    semantic: base.semantic ?? preset.semantic,
    classes: base.classes ?? preset.classes,
  };
}

export function isArchPresetKey(k: string): k is ArchPresetKey {
  return Object.prototype.hasOwnProperty.call(ARCH, k);
}

/** Map declarative arch → Helmholtz modelType (encoding wins for input layout). */
export function modelTypeForArch(
  arch: FieldArch
): NonNullable<HelmholtzFieldConfig["modelType"]> {
  if (arch.encoding === "fourier") return "fourier";
  if (arch.encoding === "hashgrid") return "hashgrid";
  if (arch.activation === "sin") return "siren";
  return "standard";
}

export function mergeFieldArch(
  base: FieldArch,
  override?: Partial<FieldArch> | null
): FieldArch {
  if (!override) return { ...base };
  return { ...base, ...override };
}

/** Human-readable one-liner for the HUD / dock. */
export function describeFieldArch(arch: FieldArch): string {
  const model = modelTypeForArch(arch);
  const act = arch.activation ?? (model === "siren" ? "sin" : "selu");
  const enc = arch.encoding ?? (model === "fourier" || model === "hashgrid" ? model : "raw");
  const widths = (arch.hiddenUnits ?? [32, 32]).join("×");
  const heads = arch.heads ?? 2;
  const encAct =
    enc === "raw" && act === "selu"
      ? "standard"
      : enc === "raw" && act === "sin"
      ? "siren"
      : act === "sin"
      ? `${enc}+siren`
      : enc;
  return `${encAct} · [${widths}] · ${heads} head${heads === 1 ? "" : "s"}`;
}

/**
 * Build a {@link HelmholtzField} from a declarative arch.
 * Single-head arches allocate only head A (fused `vector` layout).
 * Fourier + SIREN is supported: Fourier γ(p) into sin-hidden layers.
 */
export function createFieldFromArch(arch: FieldArch): HelmholtzField {
  const heads = arch.heads ?? 2;
  const modelType = modelTypeForArch(arch);
  const hiddenAct: "selu" | "sin" =
    arch.activation ?? (modelType === "siren" ? "sin" : "selu");
  if (arch.encoding === "hashgrid" && hiddenAct === "sin") {
    throw new Error(
      "FieldArch: hashgrid + sin (SIREN) is not supported yet — pick one"
    );
  }
  return new HelmholtzField({
    alpha: heads === 1 ? 0 : arch.alpha ?? 0.55,
    heads,
    hiddenUnits: arch.hiddenUnits ?? [32, 32],
    modelType,
    hiddenAct,
    fourierOctaves: arch.fourierOctaves ?? 4,
    gridSize: arch.gridSize ?? 32,
    gridFeatures: arch.gridFeatures ?? 4,
    sirenOmega0: arch.sirenOmega0 ?? 6,
    classes: arch.classes ?? 0,
    semantic: arch.semantic ?? "blend",
  });
}
