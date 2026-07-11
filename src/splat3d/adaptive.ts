import { MAX_ALPHA, PARAM_STRIDE_3D, RADIUS_MAX, RADIUS_MIN } from "./raster_wgsl";

export interface SplatAdaptationOptions {
  /** Maximum number of dead destinations to replace. The splat count never changes. */
  maxRelocations?: number;
  /**
   * Optional non-negative parent-ranking signal. It is separate from Adam's
   * signed gradient so density-control statistics cannot alter normal updates.
   */
  selectionNeed?: ArrayLike<number>;
  /** Per-splat non-negative multiplier for the absolute position-gradient magnitude. */
  coverage?: ArrayLike<number>;
  /** Splats at or below this sigmoid opacity are eligible destinations. */
  deadOpacityThreshold?: number;
  /** Splats below this sigmoid opacity cannot be parents. */
  minParentOpacity?: number;
  /** Parents must have a strictly greater coverage-weighted need score. */
  minParentNeed?: number;
  minRadius?: number;
  maxRadius?: number;
  minOpacity?: number;
  maxOpacity?: number;
  /** Radius multiplier used for both halves of a split. Must be in (0, 1]. */
  splitRadiusScale?: number;
  /** Offset from the old parent center, measured in child radii. */
  splitOffsetScale?: number;
  /** Changes only the deterministic split directions. */
  seed?: number;
}

export interface SplatRelocation {
  parentIndex: number;
  destinationIndex: number;
  parentNeed: number;
  parentGradientMagnitude: number;
  parentCoverageWeight: number;
  parentPositionBefore: [number, number, number];
  parentPositionAfter: [number, number, number];
  childPosition: [number, number, number];
  parentRadiusBefore: number;
  destinationRadiusBefore: number;
  radiusAfter: number;
  parentOpacityBefore: number;
  destinationOpacityBefore: number;
  opacityAfter: number;
  coverageMassBefore: number;
  coverageMassAfter: number;
}

export interface SplatAdaptationDiagnostics {
  splatCount: number;
  requestedRelocations: number;
  eligibleDestinations: number;
  eligibleParents: number;
  relocationCount: number;
  radiusClampCount: number;
  opacityClampCount: number;
  maxNeed: number;
  minSelectedNeed: number;
  coverageMassBefore: number;
  coverageMassAfter: number;
  coverageMassRelativeError: number;
  /** Present when the parent ranking used sampled AbsGS/Pixel-GS statistics. */
  densityStatsSampled?: boolean;
  densityVisiblePixels?: number;
  densityMaxScreenGradient?: number;
}

export interface SplatAdaptationPlan {
  params: Float32Array;
  /** Sorted unique splat indices whose parameter slices changed. */
  changedIndices: Uint32Array;
  relocations: SplatRelocation[];
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
  selectionNeed?: ArrayLike<number>;
  coverage?: ArrayLike<number>;
  deadOpacityThreshold: number;
  minParentOpacity: number;
  minParentNeed: number;
  minRadius: number;
  maxRadius: number;
  minOpacity: number;
  maxOpacity: number;
  splitRadiusScale: number;
  splitOffsetScale: number;
  seed: number;
}

const DEFAULT_DEAD_OPACITY = 0.05;
const DEFAULT_MIN_PARENT_OPACITY = 0.1;
const DEFAULT_MIN_OPACITY = 1e-4;
const DEFAULT_SPLIT_RADIUS_SCALE = 2 ** (-1 / 3);
const DEFAULT_SPLIT_OFFSET_SCALE = 0.5;
const MASS_EPSILON = 1e-12;

/**
 * Plans a CPU-side, fixed-count adaptation of isotropic 3D splats.
 *
 * Layout is the current 8G SoA layout:
 * `[position xyz * G, logRadius * G, colorRaw rgb * G, opacityRaw * G]`.
 * The input arrays are never mutated.
 */
export function planFixedBudgetSplatAdaptation(
  params: Float32Array,
  rawGradients: Float32Array,
  options: SplatAdaptationOptions = {}
): SplatAdaptationPlan {
  assertParameterArrays(params, rawGradients);
  const splatCount = params.length / PARAM_STRIDE_3D;
  const cfg = normalizeOptions(options, splatCount);
  if (cfg.coverage !== undefined && cfg.coverage.length !== splatCount) {
    throw new Error(
      `splat3d adaptive: coverage length ${cfg.coverage.length} != splat count ${splatCount}`
    );
  }
  if (cfg.selectionNeed !== undefined && cfg.selectionNeed.length !== splatCount) {
    throw new Error(
      `splat3d adaptive: selectionNeed length ${cfg.selectionNeed.length} != splat count ${splatCount}`
    );
  }

  const nextParams = params.slice();
  const changed = new Set<number>();
  const radiusOffset = 3 * splatCount;
  const colorOffset = 4 * splatCount;
  const opacityOffset = 7 * splatCount;
  let radiusClampCount = 0;
  let opacityClampCount = 0;

  for (let index = 0; index < splatCount; index++) {
    const radiusIndex = radiusOffset + index;
    const radius = Math.exp(nextParams[radiusIndex]);
    if (radius < cfg.minRadius || radius > cfg.maxRadius) {
      nextParams[radiusIndex] = boundedLogRadius(radius, cfg.minRadius, cfg.maxRadius);
      changed.add(index);
      radiusClampCount++;
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
    const gradientMagnitude =
      cfg.selectionNeed === undefined
        ? Math.hypot(Math.abs(gx), Math.abs(gy), Math.abs(gz))
        : nonNegativeFiniteOrZero(cfg.selectionNeed[index]);
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
  const relocations: SplatRelocation[] = [];

  for (let relocation = 0; relocation < relocationCount; relocation++) {
    const destination = destinations[relocation];
    const parent = parents[relocation];
    const parentIndex = parent.index;
    const destinationIndex = destination.index;
    const parentPositionBefore: [number, number, number] = [
      nextParams[parentIndex * 3 + 0],
      nextParams[parentIndex * 3 + 1],
      nextParams[parentIndex * 3 + 2],
    ];
    const parentRadiusBefore = Math.exp(nextParams[radiusOffset + parentIndex]);
    const destinationRadiusBefore = Math.exp(nextParams[radiusOffset + destinationIndex]);
    const parentOpacityBefore = sigmoid(nextParams[opacityOffset + parentIndex]);
    const destinationOpacityBefore = sigmoid(nextParams[opacityOffset + destinationIndex]);
    const coveragePairBefore =
      coverageMass(parentOpacityBefore, parentRadiusBefore) +
      coverageMass(destinationOpacityBefore, destinationRadiusBefore);

    const radiusAfter = clamp(
      parentRadiusBefore * cfg.splitRadiusScale,
      cfg.minRadius,
      cfg.maxRadius
    );
    const opticalDensityAfter = coveragePairBefore / (2 * radiusAfter * radiusAfter);
    const opacityAfter = clamp(
      1 - Math.exp(-opticalDensityAfter),
      cfg.minOpacity,
      cfg.maxOpacity
    );
    const logRadiusAfter = boundedLogRadius(radiusAfter, cfg.minRadius, cfg.maxRadius);
    const rawOpacityAfter = boundedRawOpacity(opacityAfter, cfg.minOpacity, cfg.maxOpacity);
    const boundedRadiusAfter = Math.exp(logRadiusAfter);
    const direction = deterministicUnitVector(parentIndex, destinationIndex, relocation, cfg.seed);
    const offset = boundedRadiusAfter * cfg.splitOffsetScale;
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

    writePosition(nextParams, parentIndex, parentPositionAfter);
    writePosition(nextParams, destinationIndex, childPosition);
    nextParams[radiusOffset + parentIndex] = logRadiusAfter;
    nextParams[radiusOffset + destinationIndex] = logRadiusAfter;
    nextParams[opacityOffset + parentIndex] = rawOpacityAfter;
    nextParams[opacityOffset + destinationIndex] = rawOpacityAfter;
    for (let channel = 0; channel < 3; channel++) {
      nextParams[colorOffset + destinationIndex * 3 + channel] =
        nextParams[colorOffset + parentIndex * 3 + channel];
    }
    changed.add(parentIndex);
    changed.add(destinationIndex);

    relocations.push({
      parentIndex,
      destinationIndex,
      parentNeed: parent.need,
      parentGradientMagnitude: parent.gradientMagnitude,
      parentCoverageWeight: parent.coverageWeight,
      parentPositionBefore,
      parentPositionAfter: readPosition(nextParams, parentIndex),
      childPosition: readPosition(nextParams, destinationIndex),
      parentRadiusBefore,
      destinationRadiusBefore,
      radiusAfter: Math.exp(nextParams[radiusOffset + parentIndex]),
      parentOpacityBefore,
      destinationOpacityBefore,
      opacityAfter: sigmoid(nextParams[opacityOffset + parentIndex]),
      coverageMassBefore: coveragePairBefore,
      coverageMassAfter:
        2 *
        coverageMass(
          sigmoid(nextParams[opacityOffset + parentIndex]),
          Math.exp(nextParams[radiusOffset + parentIndex])
        ),
    });
  }

  assertFiniteOutput(nextParams);
  const coverageMassAfter = totalCoverageMass(nextParams, splatCount);
  const maxNeed = parents.length > 0 ? parents[0].need : 0;
  const minSelectedNeed =
    relocationCount > 0 ? parents[relocationCount - 1].need : 0;

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
      radiusClampCount,
      opacityClampCount,
      maxNeed,
      minSelectedNeed,
      coverageMassBefore,
      coverageMassAfter,
      coverageMassRelativeError:
        Math.abs(coverageMassAfter - coverageMassBefore) /
        Math.max(coverageMassBefore, MASS_EPSILON),
    },
  };
}

function assertParameterArrays(params: Float32Array, rawGradients: Float32Array): void {
  if (params.length === 0 || params.length % PARAM_STRIDE_3D !== 0) {
    throw new Error(
      `splat3d adaptive: params length ${params.length} must be a positive multiple of ${PARAM_STRIDE_3D}`
    );
  }
  if (rawGradients.length !== params.length) {
    throw new Error(
      `splat3d adaptive: raw gradient length ${rawGradients.length} != params length ${params.length}`
    );
  }
  for (let index = 0; index < params.length; index++) {
    if (!Number.isFinite(params[index])) {
      throw new Error(`splat3d adaptive: non-finite parameter at offset ${index}`);
    }
  }
}

function normalizeOptions(options: SplatAdaptationOptions, splatCount: number): NormalizedOptions {
  const maxRelocations = options.maxRelocations ?? Math.max(1, Math.floor(splatCount * 0.01));
  const deadOpacityThreshold = options.deadOpacityThreshold ?? DEFAULT_DEAD_OPACITY;
  const minParentOpacity = options.minParentOpacity ?? DEFAULT_MIN_PARENT_OPACITY;
  const minParentNeed = options.minParentNeed ?? 0;
  const minRadius = options.minRadius ?? RADIUS_MIN;
  const maxRadius = options.maxRadius ?? RADIUS_MAX;
  const minOpacity = options.minOpacity ?? DEFAULT_MIN_OPACITY;
  const maxOpacity = options.maxOpacity ?? MAX_ALPHA;
  const splitRadiusScale = options.splitRadiusScale ?? DEFAULT_SPLIT_RADIUS_SCALE;
  const splitOffsetScale = options.splitOffsetScale ?? DEFAULT_SPLIT_OFFSET_SCALE;

  assertFiniteRange(maxRelocations, 0, Number.MAX_SAFE_INTEGER, "maxRelocations");
  assertFiniteRange(deadOpacityThreshold, 0, 1, "deadOpacityThreshold");
  assertFiniteRange(minParentOpacity, 0, 1, "minParentOpacity");
  assertFiniteRange(minParentNeed, 0, Number.MAX_VALUE, "minParentNeed");
  assertFiniteRange(minRadius, Number.MIN_VALUE, Number.MAX_VALUE, "minRadius");
  assertFiniteRange(maxRadius, Number.MIN_VALUE, Number.MAX_VALUE, "maxRadius");
  if (maxRadius <= minRadius) {
    throw new Error("splat3d adaptive: maxRadius must be greater than minRadius");
  }
  if (!(minOpacity > 0 && minOpacity < 1)) {
    throw new Error("splat3d adaptive: minOpacity must be in (0, 1)");
  }
  if (!(maxOpacity > minOpacity && maxOpacity < 1)) {
    throw new Error("splat3d adaptive: maxOpacity must be in (minOpacity, 1)");
  }
  if (!(Number.isFinite(splitRadiusScale) && splitRadiusScale > 0 && splitRadiusScale <= 1)) {
    throw new Error("splat3d adaptive: splitRadiusScale must be in (0, 1]");
  }
  assertFiniteRange(splitOffsetScale, 0, 100, "splitOffsetScale");

  return {
    maxRelocations: Math.min(splatCount, Math.floor(maxRelocations)),
    selectionNeed: options.selectionNeed,
    coverage: options.coverage,
    deadOpacityThreshold,
    minParentOpacity,
    minParentNeed,
    minRadius,
    maxRadius,
    minOpacity,
    maxOpacity,
    splitRadiusScale,
    splitOffsetScale,
    seed: (options.seed ?? 1) >>> 0,
  };
}

function assertFiniteRange(value: number, min: number, max: number, name: string): void {
  if (!(Number.isFinite(value) && value >= min && value <= max)) {
    throw new Error(`splat3d adaptive: ${name} must be finite and in [${min}, ${max}]`);
  }
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

function boundedLogRadius(radius: number, minRadius: number, maxRadius: number): number {
  const margin = Math.min((maxRadius - minRadius) * 0.25, Math.max(maxRadius, 1) * 1e-7);
  return Math.fround(Math.log(clamp(radius, minRadius + margin, maxRadius - margin)));
}

function boundedRawOpacity(opacity: number, minOpacity: number, maxOpacity: number): number {
  const margin = Math.min((maxOpacity - minOpacity) * 0.25, 1e-7);
  return Math.fround(logit(clamp(opacity, minOpacity + margin, maxOpacity - margin)));
}

function coverageMass(opacity: number, radius: number): number {
  return -Math.log1p(-opacity) * radius * radius;
}

function totalCoverageMass(params: Float32Array, splatCount: number): number {
  const radiusOffset = 3 * splatCount;
  const opacityOffset = 7 * splatCount;
  let total = 0;
  for (let index = 0; index < splatCount; index++) {
    total += coverageMass(
      sigmoid(params[opacityOffset + index]),
      Math.exp(params[radiusOffset + index])
    );
  }
  return total;
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

function writePosition(params: Float32Array, index: number, position: [number, number, number]): void {
  params[index * 3 + 0] = position[0];
  params[index * 3 + 1] = position[1];
  params[index * 3 + 2] = position[2];
}

function readPosition(params: Float32Array, index: number): [number, number, number] {
  return [params[index * 3 + 0], params[index * 3 + 1], params[index * 3 + 2]];
}

function assertFiniteOutput(params: Float32Array): void {
  for (let index = 0; index < params.length; index++) {
    if (!Number.isFinite(params[index])) {
      throw new Error(`splat3d adaptive: adaptation produced non-finite parameter at offset ${index}`);
    }
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
