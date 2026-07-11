export const ANISO_PARAM_STRIDE_3D = 14;

export const ANISO_PARAM_COMPONENTS_3D = {
  position: 3,
  logScale: 3,
  quaternion: 4,
  color: 3,
  opacity: 1,
} as const;

export interface AnisotropicSplat3D {
  position: [number, number, number];
  logScale: [number, number, number];
  /** Quaternion order is [x, y, z, w]. It is normalized during projection. */
  quaternion: [number, number, number, number];
  color: [number, number, number];
  opacity: number;
}

export interface AnisotropicParamSegment3D {
  name: keyof typeof ANISO_PARAM_COMPONENTS_3D;
  offset: number;
  length: number;
  components: number;
}

/**
 * SoA segments, compatible with the current 3D optimizer's packing style:
 * position3 | logScale3 | quaternion4 | color3 | opacity1.
 */
export function anisotropicParamSegments3D(splatCount: number): AnisotropicParamSegment3D[] {
  if (!Number.isInteger(splatCount) || splatCount < 0) {
    throw new Error(`splat3d_aniso: invalid splat count ${splatCount}`);
  }
  return [
    { name: "position", offset: 0, length: 3 * splatCount, components: 3 },
    { name: "logScale", offset: 3 * splatCount, length: 3 * splatCount, components: 3 },
    { name: "quaternion", offset: 6 * splatCount, length: 4 * splatCount, components: 4 },
    { name: "color", offset: 10 * splatCount, length: 3 * splatCount, components: 3 },
    { name: "opacity", offset: 13 * splatCount, length: splatCount, components: 1 },
  ];
}

export function packAnisotropicSplats3D(splats: readonly AnisotropicSplat3D[]): Float32Array {
  const count = splats.length;
  const segments = anisotropicParamSegments3D(count);
  const offsets = Object.fromEntries(segments.map((segment) => [segment.name, segment.offset])) as Record<
    keyof typeof ANISO_PARAM_COMPONENTS_3D,
    number
  >;
  const packed = new Float32Array(ANISO_PARAM_STRIDE_3D * count);

  for (let g = 0; g < count; g++) {
    const splat = splats[g];
    packed.set(splat.position, offsets.position + 3 * g);
    packed.set(splat.logScale, offsets.logScale + 3 * g);
    packed.set(splat.quaternion, offsets.quaternion + 4 * g);
    packed.set(splat.color, offsets.color + 3 * g);
    packed[offsets.opacity + g] = splat.opacity;
  }
  return packed;
}

export function unpackAnisotropicSplat3D(
  packed: ArrayLike<number>,
  splatCount: number,
  index: number
): AnisotropicSplat3D {
  if (!Number.isInteger(index) || index < 0 || index >= splatCount) {
    throw new Error(`splat3d_aniso: splat index ${index} outside [0, ${splatCount})`);
  }
  if (packed.length < ANISO_PARAM_STRIDE_3D * splatCount) {
    throw new Error(
      `splat3d_aniso: packed length ${packed.length} is smaller than ${ANISO_PARAM_STRIDE_3D * splatCount}`
    );
  }
  const segments = anisotropicParamSegments3D(splatCount);
  const offsets = Object.fromEntries(segments.map((segment) => [segment.name, segment.offset])) as Record<
    keyof typeof ANISO_PARAM_COMPONENTS_3D,
    number
  >;
  const tuple = <N extends number>(offset: number, length: N): number[] =>
    Array.from({ length }, (_, component) => packed[offset + component]);

  return {
    position: tuple(offsets.position + 3 * index, 3) as [number, number, number],
    logScale: tuple(offsets.logScale + 3 * index, 3) as [number, number, number],
    quaternion: tuple(offsets.quaternion + 4 * index, 4) as [number, number, number, number],
    color: tuple(offsets.color + 3 * index, 3) as [number, number, number],
    opacity: packed[offsets.opacity + index],
  };
}
