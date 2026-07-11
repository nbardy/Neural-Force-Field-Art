export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];
export type Mat3Columns = [Vec3, Vec3, Vec3];

/** Symmetric 2x2 matrix stored as [m00, m01, m11]. */
export type Symmetric2 = [number, number, number];

export type CovarianceProjectionMode = "legacy-affine" | "perspective-jacobian";

export interface AnisotropicProjectionCamera {
  eye: Vec3;
  right: Vec3;
  up: Vec3;
  forward: Vec3;
  focalPx: number;
  centerPx: Vec2;
  near: number;
}

export interface AnisotropicProjectionParams {
  position: Vec3;
  logScale: Vec3;
  /** Raw [x, y, z, w] quaternion. Projection normalizes it. */
  quaternion: Vec4;
}

export interface AnisotropicProjectionSettings {
  mode: CovarianceProjectionMode;
  minScale: number;
  maxScale: number;
  /** Isotropic variance added to the projected covariance, in pixel^2. */
  screenVariancePx2: number;
  quaternionEpsilon: number;
  determinantEpsilon: number;
}

export interface AnisotropicProjection {
  meanPx: Vec2;
  covariance: Symmetric2;
  /** Inverse covariance [a, b, c], used as a*x^2 + 2*b*x*y + c*y^2. */
  conic: Symmetric2;
  cameraPosition: Vec3;
  scales: Vec3;
  normalizedQuaternion: Vec4;
  determinant: number;
}

export interface AnisotropicProjectionUpstream {
  meanPx: Vec2;
  conic: Symmetric2;
}

export interface AnisotropicProjectionGradient {
  position: Vec3;
  logScale: Vec3;
  quaternion: Vec4;
}

export const DEFAULT_ANISOTROPIC_PROJECTION_SETTINGS: Readonly<AnisotropicProjectionSettings> = {
  mode: "legacy-affine",
  minScale: 0.01,
  maxScale: 0.45,
  screenVariancePx2: 0,
  quaternionEpsilon: 1e-12,
  determinantEpsilon: 1e-12,
};

interface ForwardState extends AnisotropicProjection {
  rawScales: Vec3;
  scaleGates: Vec3;
  rotation: Mat3Columns;
  rotationCamera: Mat3Columns;
  transformCamera: Mat3Columns;
  covarianceJacobian: [Vec3, Vec3];
  meanJacobian: [Vec3, Vec3];
  rawDeterminant: number;
  safeZ: number;
  zGate: number;
  quaternionNorm: number;
  quaternionNormActive: boolean;
}

const dot3 = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const scale3 = (a: Vec3, scale: number): Vec3 => [a[0] * scale, a[1] * scale, a[2] * scale];

const addScaled3 = (a: Vec3, b: Vec3, scale: number): Vec3 => [
  a[0] + b[0] * scale,
  a[1] + b[1] * scale,
  a[2] + b[2] * scale,
];

const clamp = (value: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, value));

function resolveSettings(settings?: Partial<AnisotropicProjectionSettings>): AnisotropicProjectionSettings {
  const resolved = { ...DEFAULT_ANISOTROPIC_PROJECTION_SETTINGS, ...settings };
  if (!(resolved.minScale > 0) || !(resolved.maxScale > resolved.minScale)) {
    throw new Error("splat3d_aniso: scales require 0 < minScale < maxScale");
  }
  if (!(resolved.quaternionEpsilon > 0) || !(resolved.determinantEpsilon > 0)) {
    throw new Error("splat3d_aniso: epsilon values must be positive");
  }
  if (!(resolved.screenVariancePx2 >= 0)) {
    throw new Error("splat3d_aniso: screenVariancePx2 must be non-negative");
  }
  return resolved;
}

export function normalizeQuaternionXYZW(raw: Vec4, epsilon = 1e-12): Vec4 {
  const norm = Math.max(Math.hypot(raw[0], raw[1], raw[2], raw[3]), epsilon);
  return [raw[0] / norm, raw[1] / norm, raw[2] / norm, raw[3] / norm];
}

/** Rotation columns for a normalized [x, y, z, w] quaternion. */
export function quaternionRotationColumnsXYZW(q: Vec4): Mat3Columns {
  const [x, y, z, w] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
    [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
    [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
  ];
}

function worldToCamera(vector: Vec3, camera: AnisotropicProjectionCamera): Vec3 {
  return [dot3(vector, camera.right), dot3(vector, camera.up), dot3(vector, camera.forward)];
}

function cameraToWorld(vector: Vec3, camera: AnisotropicProjectionCamera): Vec3 {
  let result = scale3(camera.right, vector[0]);
  result = addScaled3(result, camera.up, vector[1]);
  return addScaled3(result, camera.forward, vector[2]);
}

function projectionJacobians(
  cameraPosition: Vec3,
  focalPx: number,
  safeZ: number,
  mode: CovarianceProjectionMode
): { mean: [Vec3, Vec3]; covariance: [Vec3, Vec3] } {
  const invZ = 1 / safeZ;
  const focalInvZ = focalPx * invZ;
  const mean: [Vec3, Vec3] = [
    [focalInvZ, 0, -focalPx * cameraPosition[0] * invZ * invZ],
    [0, -focalInvZ, focalPx * cameraPosition[1] * invZ * invZ],
  ];
  const covariance: [Vec3, Vec3] =
    mode === "perspective-jacobian" ? mean : [[focalInvZ, 0, 0], [0, -focalInvZ, 0]];
  return { mean, covariance };
}

function forwardState(
  params: AnisotropicProjectionParams,
  camera: AnisotropicProjectionCamera,
  partialSettings?: Partial<AnisotropicProjectionSettings>
): ForwardState {
  const settings = resolveSettings(partialSettings);
  const relative: Vec3 = [
    params.position[0] - camera.eye[0],
    params.position[1] - camera.eye[1],
    params.position[2] - camera.eye[2],
  ];
  const cameraPosition = worldToCamera(relative, camera);
  const safeZ = Math.max(cameraPosition[2], camera.near);
  const zGate = cameraPosition[2] > camera.near ? 1 : 0;
  const jacobians = projectionJacobians(cameraPosition, camera.focalPx, safeZ, settings.mode);
  const meanPx: Vec2 = [
    camera.centerPx[0] + camera.focalPx * cameraPosition[0] / safeZ,
    camera.centerPx[1] - camera.focalPx * cameraPosition[1] / safeZ,
  ];

  const rawScales: Vec3 = [Math.exp(params.logScale[0]), Math.exp(params.logScale[1]), Math.exp(params.logScale[2])];
  const scales: Vec3 = rawScales.map((scale) => clamp(scale, settings.minScale, settings.maxScale)) as Vec3;
  const scaleGates: Vec3 = rawScales.map((scale) =>
    scale > settings.minScale && scale < settings.maxScale ? 1 : 0
  ) as Vec3;

  const rawQuaternionNorm = Math.hypot(...params.quaternion);
  const quaternionNorm = Math.max(rawQuaternionNorm, settings.quaternionEpsilon);
  const normalizedQuaternion: Vec4 = params.quaternion.map((value) => value / quaternionNorm) as Vec4;
  const rotation = quaternionRotationColumnsXYZW(normalizedQuaternion);
  const rotationCamera = rotation.map((axis) => worldToCamera(axis, camera)) as Mat3Columns;
  const transformCamera = rotationCamera.map((axis, index) => scale3(axis, scales[index])) as Mat3Columns;
  const projectedAxes = transformCamera.map((axis) => [
    dot3(jacobians.covariance[0], axis),
    dot3(jacobians.covariance[1], axis),
  ] as Vec2);

  let covariance00 = settings.screenVariancePx2;
  let covariance01 = 0;
  let covariance11 = settings.screenVariancePx2;
  for (const axis of projectedAxes) {
    covariance00 += axis[0] * axis[0];
    covariance01 += axis[0] * axis[1];
    covariance11 += axis[1] * axis[1];
  }
  const covariance: Symmetric2 = [covariance00, covariance01, covariance11];
  const rawDeterminant = covariance00 * covariance11 - covariance01 * covariance01;
  const determinant = Math.max(rawDeterminant, settings.determinantEpsilon);
  const conic: Symmetric2 = [covariance11 / determinant, -covariance01 / determinant, covariance00 / determinant];

  return {
    meanPx,
    covariance,
    conic,
    cameraPosition,
    scales,
    normalizedQuaternion,
    determinant,
    rawScales,
    scaleGates,
    rotation,
    rotationCamera,
    transformCamera,
    covarianceJacobian: jacobians.covariance,
    meanJacobian: jacobians.mean,
    rawDeterminant,
    safeZ,
    zGate,
    quaternionNorm,
    quaternionNormActive: rawQuaternionNorm > settings.quaternionEpsilon,
  };
}

export function projectAnisotropicGaussian(
  params: AnisotropicProjectionParams,
  camera: AnisotropicProjectionCamera,
  settings?: Partial<AnisotropicProjectionSettings>
): AnisotropicProjection {
  const state = forwardState(params, camera, settings);
  return {
    meanPx: state.meanPx,
    covariance: state.covariance,
    conic: state.conic,
    cameraPosition: state.cameraPosition,
    scales: state.scales,
    normalizedQuaternion: state.normalizedQuaternion,
    determinant: state.determinant,
  };
}

function conicToCovarianceGradient(conic: Symmetric2, upstream: Symmetric2): Symmetric2 {
  const [a, b, c] = conic;
  const [ga, gb, gc] = upstream;
  const h00 = ga;
  const h01 = 0.5 * gb;
  const h11 = gc;
  const t00 = a * h00 + b * h01;
  const t01 = a * h01 + b * h11;
  const t10 = b * h00 + c * h01;
  const t11 = b * h01 + c * h11;
  return [
    -(t00 * a + t01 * b),
    -2 * (t00 * b + t01 * c),
    -(t10 * b + t11 * c),
  ];
}

function normalizedQuaternionGradient(raw: Vec4, normalized: Vec4, gradient: Vec4, norm: number, active: boolean): Vec4 {
  if (!active) return gradient.map((value) => value / norm) as Vec4;
  const radial =
    normalized[0] * gradient[0] +
    normalized[1] * gradient[1] +
    normalized[2] * gradient[2] +
    normalized[3] * gradient[3];
  return normalized.map((value, index) => (gradient[index] - value * radial) / norm) as Vec4;
}

function rotationQuaternionGradient(rotationGradient: Mat3Columns, q: Vec4): Vec4 {
  const [x, y, z, w] = q;
  const g00 = rotationGradient[0][0];
  const g10 = rotationGradient[0][1];
  const g20 = rotationGradient[0][2];
  const g01 = rotationGradient[1][0];
  const g11 = rotationGradient[1][1];
  const g21 = rotationGradient[1][2];
  const g02 = rotationGradient[2][0];
  const g12 = rotationGradient[2][1];
  const g22 = rotationGradient[2][2];
  return [
    2 * y * (g01 + g10) + 2 * z * (g02 + g20) - 4 * x * (g11 + g22) + 2 * w * (g21 - g12),
    -4 * y * (g00 + g22) + 2 * x * (g01 + g10) + 2 * z * (g12 + g21) + 2 * w * (g02 - g20),
    -4 * z * (g00 + g11) + 2 * x * (g02 + g20) + 2 * y * (g12 + g21) + 2 * w * (g10 - g01),
    2 * z * (g10 - g01) + 2 * y * (g02 - g20) + 2 * x * (g21 - g12),
  ];
}

export function backwardAnisotropicProjection(
  params: AnisotropicProjectionParams,
  camera: AnisotropicProjectionCamera,
  upstream: AnisotropicProjectionUpstream,
  partialSettings?: Partial<AnisotropicProjectionSettings>
): AnisotropicProjectionGradient {
  const settings = resolveSettings(partialSettings);
  const state = forwardState(params, camera, settings);
  if (!(state.rawDeterminant > settings.determinantEpsilon)) {
    throw new Error("splat3d_aniso: covariance determinant floor is active; projection gradient is undefined there");
  }

  const covarianceGradient = conicToCovarianceGradient(state.conic, upstream.conic);
  const [gCov00, gCov01, gCov11] = covarianceGradient;
  const projectedAxes = state.transformCamera.map((axis) => [
    dot3(state.covarianceJacobian[0], axis),
    dot3(state.covarianceJacobian[1], axis),
  ] as Vec2);
  const projectedAxesGradient = projectedAxes.map((axis) => [
    2 * gCov00 * axis[0] + gCov01 * axis[1],
    gCov01 * axis[0] + 2 * gCov11 * axis[1],
  ] as Vec2);

  const transformCameraGradient = projectedAxesGradient.map((gradient) => [
    state.covarianceJacobian[0][0] * gradient[0] + state.covarianceJacobian[1][0] * gradient[1],
    state.covarianceJacobian[0][1] * gradient[0] + state.covarianceJacobian[1][1] * gradient[1],
    state.covarianceJacobian[0][2] * gradient[0] + state.covarianceJacobian[1][2] * gradient[1],
  ] as Vec3) as Mat3Columns;

  const jacobianGradient: [Vec3, Vec3] = [[0, 0, 0], [0, 0, 0]];
  for (let axis = 0; axis < 3; axis++) {
    for (let component = 0; component < 3; component++) {
      jacobianGradient[0][component] += projectedAxesGradient[axis][0] * state.transformCamera[axis][component];
      jacobianGradient[1][component] += projectedAxesGradient[axis][1] * state.transformCamera[axis][component];
    }
  }

  const rotationGradient = state.rotation.map((_, axis) =>
    scale3(cameraToWorld(transformCameraGradient[axis], camera), state.scales[axis])
  ) as Mat3Columns;
  const logScale: Vec3 = [0, 1, 2].map((axis) =>
    dot3(transformCameraGradient[axis], state.rotationCamera[axis]) * state.rawScales[axis] * state.scaleGates[axis]
  ) as Vec3;

  const quaternionNormalizedGradient = rotationQuaternionGradient(rotationGradient, state.normalizedQuaternion);
  const quaternion = normalizedQuaternionGradient(
    params.quaternion,
    state.normalizedQuaternion,
    quaternionNormalizedGradient,
    state.quaternionNorm,
    state.quaternionNormActive
  );

  const cameraPositionGradient: Vec3 = [
    state.meanJacobian[0][0] * upstream.meanPx[0] + state.meanJacobian[1][0] * upstream.meanPx[1],
    state.meanJacobian[0][1] * upstream.meanPx[0] + state.meanJacobian[1][1] * upstream.meanPx[1],
    state.meanJacobian[0][2] * upstream.meanPx[0] + state.meanJacobian[1][2] * upstream.meanPx[1],
  ];
  const invZ = 1 / state.safeZ;
  const invZ2 = invZ * invZ;
  if (settings.mode === "perspective-jacobian") {
    cameraPositionGradient[0] += jacobianGradient[0][2] * (-camera.focalPx * invZ2);
    cameraPositionGradient[1] += jacobianGradient[1][2] * (camera.focalPx * invZ2);
    cameraPositionGradient[2] +=
      jacobianGradient[0][0] * (-camera.focalPx * invZ2) +
      jacobianGradient[0][2] * (2 * camera.focalPx * state.cameraPosition[0] * invZ2 * invZ) +
      jacobianGradient[1][1] * (camera.focalPx * invZ2) +
      jacobianGradient[1][2] * (-2 * camera.focalPx * state.cameraPosition[1] * invZ2 * invZ);
  } else {
    cameraPositionGradient[2] +=
      jacobianGradient[0][0] * (-camera.focalPx * invZ2) +
      jacobianGradient[1][1] * (camera.focalPx * invZ2);
  }
  cameraPositionGradient[2] *= state.zGate;

  return {
    position: cameraToWorld(cameraPositionGradient, camera),
    logScale,
    quaternion,
  };
}
