import type {
  SplatAdaptationDiagnostics,
  SplatAdaptationOptions,
} from "../splat3d/adaptive";
import { MAX_ALPHA, RADIUS_MAX, RADIUS_MIN } from "../splat3d/raster_wgsl";
import { ANISO_PARAM_STRIDE_3D } from "./layout";

export interface AnisotropicSplatAdaptationOptions extends SplatAdaptationOptions {}

export interface AnisotropicSplatRelocation {
  parentIndex: number;
  destinationIndex: number;
  parentNeed: number;
  parentGradientMagnitude: number;
  parentCoverageWeight: number;
  parentPositionBefore: [number, number, number];
  parentPositionAfter: [number, number, number];
  childPosition: [number, number, number];
  parentLogScaleBefore: [number, number, number];
  destinationLogScaleBefore: [number, number, number];
  logScaleAfter: [number, number, number];
  parentQuaternion: [number, number, number, number];
  parentOpacityBefore: number;
  destinationOpacityBefore: number;
  opacityAfter: number;
  coverageMassBefore: number;
  coverageMassAfter: number;
}

export interface AnisotropicSplatAdaptationPlan {
  params: Float32Array;
  /** Sorted unique splat indices whose parameter slices changed. */
  changedIndices: Uint32Array;
  relocations: AnisotropicSplatRelocation[];
  diagnostics: SplatAdaptationDiagnostics;
}

interface Candidate {
  index: number;
  opacity: number;
  gradientMagnitude: number;
  coverageWeight: number;
  need: number;
}

interface NormalizedOptions {
  maxRelocations: number;
  coverage?: ArrayLike<number>;
  deadOpacityThreshold: number;
  minParentOpacity: number;
  minParentNeed: number;
  minScale: number;
  maxScale: number;
  minOpacity: number;
  maxOpacity: number;
  splitScale: number;
  splitOffsetScale: number;
  seed: number;
}

const DEFAULT_DEAD_OPACITY = 0.05;
const DEFAULT_MIN_PARENT_OPACITY = 0.1;
const DEFAULT_MIN_OPACITY = 1e-4;
const DEFAULT_SPLIT_SCALE = 2 ** (-1 / 3);
const DEFAULT_SPLIT_OFFSET_SCALE = 0.5;
const MASS_EPSILON = 1e-12;

/**
 * Plans a CPU-side, fixed-count adaptation of anisotropic 3D splats.
 *
 * Dead, low-opacity destinations are replaced with children of the highest
 * coverage-weighted position-gradient parents. A split clones color and
 * orientation, and applies one log-scale delta to every axis so anisotropy is
 * exactly preserved. The input arrays are never mutated.
 */
export function planFixedBudgetAnisotropicSplatAdaptation(
  params: Float32Array,
  rawGradients: Float32Array,
  options: AnisotropicSplatAdaptationOptions = {}
): AnisotropicSplatAdaptationPlan {
  assertParameterArrays(params, rawGradients);
  const splatCount = params.length / ANISO_PARAM_STRIDE_3D;
  const cfg = normalizeOptions(options, splatCount);
  if (cfg.coverage !== undefined && cfg.coverage.length !== splatCount) {
    throw new Error(
      `splat3d aniso adaptive: coverage length ${cfg.coverage.length} != splat count ${splatCount}`
    );
  }

  const nextParams = params.slice();
  const changed = new Set<number>();
  const scaleOffset = 3 * splatCount;
  const quaternionOffset = 6 * splatCount;
  const colorOffset = 10 * splatCount;
  const opacityOffset = 13 * splatCount;
  let scaleClampCount = 0;
  let opacityClampCount = 0;

  for (let index = 0; index < splatCount; index++) {
    const logScales = readTuple3(nextParams, scaleOffset + index * 3);
    const scaleCorrection = uniformScaleCorrection(logScales, cfg.minScale, cfg.maxScale);
    if (scaleCorrection !== 0) {
      writeTuple(
        nextParams,
        scaleOffset + index * 3,
        logScales.map((value) => Math.fround(value + scaleCorrection))
      );
      changed.add(index);
      scaleClampCount++;
    }

    const opacityIndex = opacityOffset + index;
    const opacity = sigmoid(nextParams[opacityIndex]);
    if (opacity < cfg.minOpacity || opacity > cfg.maxOpacity) {
      nextParams[opacityIndex] = boundedRawOpacity(opacity, cfg.minOpacity, cfg.maxOpacity);
      changed.add(index);
      opacityClampCount++;
    }
  }

  const coverageMassBefore = totalCoverageMass(nextParams, splatCount);
  const candidates: Candidate[] = [];
  for (let index = 0; index < splatCount; index++) {
    const gx = finiteOrZero(rawGradients[index * 3 + 0]);
    const gy = finiteOrZero(rawGradients[index * 3 + 1]);
    const gz = finiteOrZero(rawGradients[index * 3 + 2]);
    const gradientMagnitude = Math.hypot(Math.abs(gx), Math.abs(gy), Math.abs(gz));
    const coverageWeight = cfg.coverage === undefined ? 1 : nonNegativeFiniteOrZero(cfg.coverage[index]);
    const need = finiteProduct(gradientMagnitude, coverageWeight);
    candidates.push({
      index,
      opacity: sigmoid(nextParams[opacityOffset + index]),
      gradientMagnitude,
      coverageWeight,
      need,
    });
  }

  const destinations = candidates
    .filter((candidate) => candidate.opacity <= cfg.deadOpacityThreshold)
    .sort(compareDestinations);
  const destinationIndices = new Set(destinations.map((candidate) => candidate.index));
  const parents = candidates
    .filter(
      (candidate) =>
        !destinationIndices.has(candidate.index) &&
        candidate.opacity >= cfg.minParentOpacity &&
        candidate.need > cfg.minParentNeed
    )
    .sort(compareParents);
  const relocationCount = Math.min(cfg.maxRelocations, destinations.length, parents.length);
  const relocations: AnisotropicSplatRelocation[] = [];

  for (let relocation = 0; relocation < relocationCount; relocation++) {
    const destination = destinations[relocation];
    const parent = parents[relocation];
    const parentIndex = parent.index;
    const destinationIndex = destination.index;
    const parentPositionBefore = readTuple3(nextParams, parentIndex * 3);
    const parentLogScaleBefore = readTuple3(nextParams, scaleOffset + parentIndex * 3);
    const destinationLogScaleBefore = readTuple3(nextParams, scaleOffset + destinationIndex * 3);
    const parentQuaternion = readTuple4(nextParams, quaternionOffset + parentIndex * 4);
    const parentOpacityBefore = sigmoid(nextParams[opacityOffset + parentIndex]);
    const destinationOpacityBefore = sigmoid(nextParams[opacityOffset + destinationIndex]);
    const parentArea = equivalentArea(parentLogScaleBefore);
    const destinationArea = equivalentArea(destinationLogScaleBefore);
    const coveragePairBefore =
      coverageMass(parentOpacityBefore, parentArea) +
      coverageMass(destinationOpacityBefore, destinationArea);

    const requestedLogScale = parentLogScaleBefore.map(
      (value) => value + Math.log(cfg.splitScale)
    ) as [number, number, number];
    const uniformCorrection = uniformScaleCorrection(requestedLogScale, cfg.minScale, cfg.maxScale);
    const logScaleAfter = requestedLogScale.map(
      (value) => Math.fround(value + uniformCorrection)
    ) as [number, number, number];
    const areaAfter = equivalentArea(logScaleAfter);
    const opticalDensityAfter = coveragePairBefore / (2 * areaAfter);
    const opacityAfter = clamp(1 - Math.exp(-opticalDensityAfter), cfg.minOpacity, cfg.maxOpacity);
    const rawOpacityAfter = boundedRawOpacity(opacityAfter, cfg.minOpacity, cfg.maxOpacity);
    const equivalentScaleAfter = Math.sqrt(areaAfter);
    const direction = deterministicUnitVector(parentIndex, destinationIndex, relocation, cfg.seed);
    const offset = equivalentScaleAfter * cfg.splitOffsetScale;
    const parentPositionAfter: [number, number, number] = [
      parentPositionBefore[0] - direction[0] * offset,
      parentPositionBefore[1] - direction[1] * offset,
      parentPositionBefore[2] - direction[2] * offset,
    ];
    const childPosition: [number, number, number] = [
      parentPositionBefore[0] + direction[0] * offset,
      parentPositionBefore[1] + direction[1] * offset,
      parentPositionBefore[2] + direction[2] * offset,
    ];

    writeTuple(nextParams, parentIndex * 3, parentPositionAfter);
    writeTuple(nextParams, destinationIndex * 3, childPosition);
    writeTuple(nextParams, scaleOffset + parentIndex * 3, logScaleAfter);
    writeTuple(nextParams, scaleOffset + destinationIndex * 3, logScaleAfter);
    writeTuple(nextParams, quaternionOffset + destinationIndex * 4, parentQuaternion);
    for (let channel = 0; channel < 3; channel++) {
      nextParams[colorOffset + destinationIndex * 3 + channel] =
        nextParams[colorOffset + parentIndex * 3 + channel];
    }
    nextParams[opacityOffset + parentIndex] = rawOpacityAfter;
    nextParams[opacityOffset + destinationIndex] = rawOpacityAfter;
    changed.add(parentIndex);
    changed.add(destinationIndex);

    const boundedOpacityAfter = sigmoid(rawOpacityAfter);
    const coveragePairAfter = 2 * coverageMass(boundedOpacityAfter, equivalentArea(logScaleAfter));
    relocations.push({
      parentIndex,
      destinationIndex,
      parentNeed: parent.need,
      parentGradientMagnitude: parent.gradientMagnitude,
      parentCoverageWeight: parent.coverageWeight,
      parentPositionBefore,
      parentPositionAfter: readTuple3(nextParams, parentIndex * 3),
      childPosition: readTuple3(nextParams, destinationIndex * 3),
      parentLogScaleBefore,
      destinationLogScaleBefore,
      logScaleAfter: readTuple3(nextParams, scaleOffset + parentIndex * 3),
      parentQuaternion,
      parentOpacityBefore,
      destinationOpacityBefore,
      opacityAfter: boundedOpacityAfter,
      coverageMassBefore: coveragePairBefore,
      coverageMassAfter: coveragePairAfter,
    });
  }

  assertFiniteOutput(nextParams);
  const coverageMassAfter = totalCoverageMass(nextParams, splatCount);
  return {
    params: nextParams,
    changedIndices: Uint32Array.from(Array.from(changed).sort((a, b) => a - b)),
    relocations,
    diagnostics: {
      splatCount,
      requestedRelocations: cfg.maxRelocations,
      eligibleDestinations: destinations.length,
      eligibleParents: parents.length,
      relocationCount,
      radiusClampCount: scaleClampCount,
      opacityClampCount,
      maxNeed: parents.length > 0 ? parents[0].need : 0,
      minSelectedNeed: relocationCount > 0 ? parents[relocationCount - 1].need : 0,
      coverageMassBefore,
      coverageMassAfter,
      coverageMassRelativeError:
        Math.abs(coverageMassAfter - coverageMassBefore) /
        Math.max(coverageMassBefore, MASS_EPSILON),
    },
  };
}

function assertParameterArrays(params: Float32Array, rawGradients: Float32Array): void {
  if (params.length === 0 || params.length % ANISO_PARAM_STRIDE_3D !== 0) {
    throw new Error(
      `splat3d aniso adaptive: params length ${params.length} must be a positive multiple of ${ANISO_PARAM_STRIDE_3D}`
    );
  }
  if (rawGradients.length !== params.length) {
    throw new Error(
      `splat3d aniso adaptive: raw gradient length ${rawGradients.length} != params length ${params.length}`
    );
  }
  for (let index = 0; index < params.length; index++) {
    if (!Number.isFinite(params[index])) {
      throw new Error(`splat3d aniso adaptive: non-finite parameter at offset ${index}`);
    }
  }
}

function normalizeOptions(
  options: AnisotropicSplatAdaptationOptions,
  splatCount: number
): NormalizedOptions {
  const maxRelocations = options.maxRelocations ?? Math.max(1, Math.floor(splatCount * 0.01));
  const deadOpacityThreshold = options.deadOpacityThreshold ?? DEFAULT_DEAD_OPACITY;
  const minParentOpacity = options.minParentOpacity ?? DEFAULT_MIN_PARENT_OPACITY;
  const minParentNeed = options.minParentNeed ?? 0;
  const minScale = options.minRadius ?? RADIUS_MIN;
  const maxScale = options.maxRadius ?? RADIUS_MAX;
  const minOpacity = options.minOpacity ?? DEFAULT_MIN_OPACITY;
  const maxOpacity = options.maxOpacity ?? MAX_ALPHA;
  const splitScale = options.splitRadiusScale ?? DEFAULT_SPLIT_SCALE;
  const splitOffsetScale = options.splitOffsetScale ?? DEFAULT_SPLIT_OFFSET_SCALE;

  assertFiniteRange(maxRelocations, 0, Number.MAX_SAFE_INTEGER, "maxRelocations");
  assertFiniteRange(deadOpacityThreshold, 0, 1, "deadOpacityThreshold");
  assertFiniteRange(minParentOpacity, 0, 1, "minParentOpacity");
  assertFiniteRange(minParentNeed, 0, Number.MAX_VALUE, "minParentNeed");
  assertFiniteRange(minScale, Number.MIN_VALUE, Number.MAX_VALUE, "minRadius");
  assertFiniteRange(maxScale, Number.MIN_VALUE, Number.MAX_VALUE, "maxRadius");
  if (maxScale <= minScale) throw new Error("splat3d aniso adaptive: maxRadius must exceed minRadius");
  if (!(minOpacity > 0 && minOpacity < 1)) {
    throw new Error("splat3d aniso adaptive: minOpacity must be in (0, 1)");
  }
  if (!(maxOpacity > minOpacity && maxOpacity < 1)) {
    throw new Error("splat3d aniso adaptive: maxOpacity must be in (minOpacity, 1)");
  }
  if (!(Number.isFinite(splitScale) && splitScale > 0 && splitScale <= 1)) {
    throw new Error("splat3d aniso adaptive: splitRadiusScale must be in (0, 1]");
  }
  assertFiniteRange(splitOffsetScale, 0, 100, "splitOffsetScale");

  return {
    maxRelocations: Math.min(splatCount, Math.floor(maxRelocations)),
    coverage: options.coverage,
    deadOpacityThreshold,
    minParentOpacity,
    minParentNeed,
    minScale,
    maxScale,
    minOpacity,
    maxOpacity,
    splitScale,
    splitOffsetScale,
    seed: (options.seed ?? 1) >>> 0,
  };
}

function equivalentArea(logScale: readonly number[]): number {
  return Math.exp((2 / 3) * (logScale[0] + logScale[1] + logScale[2]));
}

function coverageMass(opacity: number, equivalentAreaValue: number): number {
  return -Math.log1p(-opacity) * equivalentAreaValue;
}

function totalCoverageMass(params: Float32Array, splatCount: number): number {
  const scaleOffset = 3 * splatCount;
  const opacityOffset = 13 * splatCount;
  let total = 0;
  for (let index = 0; index < splatCount; index++) {
    total += coverageMass(
      sigmoid(params[opacityOffset + index]),
      equivalentArea(readTuple3(params, scaleOffset + index * 3))
    );
  }
  return total;
}

/** Clamps geometric-mean scale with one shared shift, preserving axis ratios. */
function uniformScaleCorrection(
  logScales: readonly number[],
  minScale: number,
  maxScale: number
): number {
  const minLog = Math.log(minScale);
  const maxLog = Math.log(maxScale);
  const meanLog = (logScales[0] + logScales[1] + logScales[2]) / 3;
  return Math.max(minLog, Math.min(maxLog, meanLog)) - meanLog;
}

function compareDestinations(a: Candidate, b: Candidate): number {
  return a.opacity - b.opacity || a.need - b.need || a.index - b.index;
}

function compareParents(a: Candidate, b: Candidate): number {
  return (
    b.need - a.need ||
    b.gradientMagnitude - a.gradientMagnitude ||
    b.opacity - a.opacity ||
    a.index - b.index
  );
}

function finiteOrZero(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function nonNegativeFiniteOrZero(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : 0;
}

function finiteProduct(a: number, b: number): number {
  if (a === 0 || b === 0) return 0;
  const product = a * b;
  return Number.isFinite(product) ? product : Number.MAX_VALUE;
}

function sigmoid(raw: number): number {
  if (raw >= 0) return 1 / (1 + Math.exp(-raw));
  const expRaw = Math.exp(raw);
  return expRaw / (1 + expRaw);
}

function logit(opacity: number): number {
  return Math.log(opacity) - Math.log1p(-opacity);
}

function boundedRawOpacity(opacity: number, minOpacity: number, maxOpacity: number): number {
  const margin = Math.min((maxOpacity - minOpacity) * 0.25, 1e-7);
  return Math.fround(logit(clamp(opacity, minOpacity + margin, maxOpacity - margin)));
}

function deterministicUnitVector(
  parentIndex: number,
  destinationIndex: number,
  relocation: number,
  seed: number
): [number, number, number] {
  const key =
    seed ^
    Math.imul(parentIndex + 1, 0x9e3779b1) ^
    Math.imul(destinationIndex + 1, 0x85ebca77) ^
    Math.imul(relocation + 1, 0xc2b2ae3d);
  const u = (hash32(key) + 0.5) / 0x100000000;
  const v = (hash32(key ^ 0x27d4eb2f) + 0.5) / 0x100000000;
  const z = 2 * u - 1;
  const radial = Math.sqrt(Math.max(0, 1 - z * z));
  const phi = 2 * Math.PI * v;
  return [radial * Math.cos(phi), radial * Math.sin(phi), z];
}

function hash32(value: number): number {
  let x = value >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}

function readTuple3(values: Float32Array, offset: number): [number, number, number] {
  return [values[offset], values[offset + 1], values[offset + 2]];
}

function readTuple4(values: Float32Array, offset: number): [number, number, number, number] {
  return [values[offset], values[offset + 1], values[offset + 2], values[offset + 3]];
}

function writeTuple(values: Float32Array, offset: number, tuple: readonly number[]): void {
  for (let component = 0; component < tuple.length; component++) {
    values[offset + component] = tuple[component];
  }
}

function assertFiniteRange(value: number, min: number, max: number, name: string): void {
  if (!(Number.isFinite(value) && value >= min && value <= max)) {
    throw new Error(`splat3d aniso adaptive: ${name} must be finite and in [${min}, ${max}]`);
  }
}

function assertFiniteOutput(params: Float32Array): void {
  for (let index = 0; index < params.length; index++) {
    if (!Number.isFinite(params[index])) {
      throw new Error(
        `splat3d aniso adaptive: adaptation produced non-finite parameter at offset ${index}`
      );
    }
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
