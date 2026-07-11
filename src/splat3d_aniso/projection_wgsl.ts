import type { CovarianceProjectionMode } from "./projection";

export interface AnisotropicProjectionWGSLOptions {
  mode?: CovarianceProjectionMode;
}

/**
 * Reusable WGSL math for anisotropic 3D Gaussian projection and its backward
 * chain. The caller owns storage declarations and compute entry points.
 */
export function anisotropicProjectionWGSL(options: AnisotropicProjectionWGSLOptions = {}): string {
  const mode = options.mode ?? "legacy-affine";
  const covarianceJacobian =
    mode === "perspective-jacobian"
      ? /* wgsl */ `
  let inv_z = 1.0 / safe_z;
  let inv_z2 = inv_z * inv_z;
  return mat3x2f(
    vec2f(focal_px * inv_z, 0.0),
    vec2f(0.0, -focal_px * inv_z),
    vec2f(-focal_px * camera_position.x * inv_z2, focal_px * camera_position.y * inv_z2)
  );`
      : /* wgsl */ `
  let focal_inv_z = focal_px / safe_z;
  return mat3x2f(
    vec2f(focal_inv_z, 0.0),
    vec2f(0.0, -focal_inv_z),
    vec2f(0.0)
  );`;
  const jacobianPositionBackward =
    mode === "perspective-jacobian"
      ? /* wgsl */ `
  camera_gradient.x = camera_gradient.x + jacobian_gradient[2].x * (-camera.focal_px * inv_z2);
  camera_gradient.y = camera_gradient.y + jacobian_gradient[2].y * camera.focal_px * inv_z2;
  camera_gradient.z = camera_gradient.z
    + jacobian_gradient[0].x * (-camera.focal_px * inv_z2)
    + jacobian_gradient[2].x * (2.0 * camera.focal_px * projected.camera_position.x * inv_z3)
    + jacobian_gradient[1].y * (camera.focal_px * inv_z2)
    + jacobian_gradient[2].y * (-2.0 * camera.focal_px * projected.camera_position.y * inv_z3);`
      : /* wgsl */ `
  camera_gradient.z = camera_gradient.z
    + jacobian_gradient[0].x * (-camera.focal_px * inv_z2)
    + jacobian_gradient[1].y * (camera.focal_px * inv_z2);`;

  return /* wgsl */ `
struct AnisoCamera {
  eye       : vec3f,
  right     : vec3f,
  up        : vec3f,
  forward   : vec3f,
  focal_px  : f32,
  center_px : vec2f,
  near      : f32,
};

struct AnisoProjectionSettings {
  min_scale            : f32,
  max_scale            : f32,
  screen_variance_px2  : f32,
  quaternion_epsilon   : f32,
  determinant_epsilon  : f32,
};

struct AnisoProjection {
  mean_px               : vec2f,
  covariance            : vec3f,
  conic                 : vec3f,
  camera_position       : vec3f,
  scales                : vec3f,
  normalized_quaternion : vec4f,
  determinant           : f32,
};

struct AnisoProjectionUpstream {
  mean_px : vec2f,
  conic   : vec3f,
};

struct AnisoProjectionGradient {
  position   : vec3f,
  log_scale  : vec3f,
  quaternion : vec4f,
};

fn anisoWorldToCamera(camera : AnisoCamera, vector : vec3f) -> vec3f {
  return vec3f(dot(vector, camera.right), dot(vector, camera.up), dot(vector, camera.forward));
}

fn anisoCameraToWorld(camera : AnisoCamera, vector : vec3f) -> vec3f {
  return camera.right * vector.x + camera.up * vector.y + camera.forward * vector.z;
}

fn anisoNormalizeQuaternion(raw : vec4f, epsilon : f32) -> vec4f {
  return raw * inverseSqrt(max(dot(raw, raw), epsilon * epsilon));
}

// Quaternion order is [x, y, z, w]; the matrix constructor arguments are columns.
fn anisoQuaternionRotation(q : vec4f) -> mat3x3f {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
    vec3f(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
    vec3f(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))
  );
}

// mat3x2f is a 2-row, 3-column matrix. Its columns correspond to camera x/y/z.
fn anisoMeanJacobian(camera_position : vec3f, focal_px : f32, safe_z : f32) -> mat3x2f {
  let inv_z = 1.0 / safe_z;
  let inv_z2 = inv_z * inv_z;
  return mat3x2f(
    vec2f(focal_px * inv_z, 0.0),
    vec2f(0.0, -focal_px * inv_z),
    vec2f(-focal_px * camera_position.x * inv_z2, focal_px * camera_position.y * inv_z2)
  );
}

fn anisoCovarianceJacobian(camera_position : vec3f, focal_px : f32, safe_z : f32) -> mat3x2f {${covarianceJacobian}
}

fn anisoProject(
  position : vec3f,
  log_scale : vec3f,
  quaternion : vec4f,
  camera : AnisoCamera,
  settings : AnisoProjectionSettings
) -> AnisoProjection {
  let camera_position = anisoWorldToCamera(camera, position - camera.eye);
  let safe_z = max(camera_position.z, camera.near);
  let mean_px = camera.center_px + vec2f(
    camera.focal_px * camera_position.x / safe_z,
    -camera.focal_px * camera_position.y / safe_z
  );
  let scales = clamp(exp(log_scale), vec3f(settings.min_scale), vec3f(settings.max_scale));
  let normalized_quaternion = anisoNormalizeQuaternion(quaternion, settings.quaternion_epsilon);
  let rotation = anisoQuaternionRotation(normalized_quaternion);
  let transform = mat3x3f(
    anisoWorldToCamera(camera, rotation[0]) * scales.x,
    anisoWorldToCamera(camera, rotation[1]) * scales.y,
    anisoWorldToCamera(camera, rotation[2]) * scales.z
  );
  let jacobian = anisoCovarianceJacobian(camera_position, camera.focal_px, safe_z);
  let axis0 = jacobian * transform[0];
  let axis1 = jacobian * transform[1];
  let axis2 = jacobian * transform[2];
  let covariance = vec3f(
    dot(vec3f(axis0.x, axis1.x, axis2.x), vec3f(axis0.x, axis1.x, axis2.x)) + settings.screen_variance_px2,
    axis0.x * axis0.y + axis1.x * axis1.y + axis2.x * axis2.y,
    dot(vec3f(axis0.y, axis1.y, axis2.y), vec3f(axis0.y, axis1.y, axis2.y)) + settings.screen_variance_px2
  );
  let determinant = max(
    covariance.x * covariance.z - covariance.y * covariance.y,
    settings.determinant_epsilon
  );
  let conic = vec3f(covariance.z, -covariance.y, covariance.x) / determinant;
  return AnisoProjection(
    mean_px,
    covariance,
    conic,
    camera_position,
    scales,
    normalized_quaternion,
    determinant
  );
}

fn anisoConicToCovarianceGradient(conic : vec3f, upstream : vec3f) -> vec3f {
  // The stored off-diagonal appears once in upstream, but twice in a matrix trace.
  let h00 = upstream.x;
  let h01 = 0.5 * upstream.y;
  let h11 = upstream.z;
  let t00 = conic.x * h00 + conic.y * h01;
  let t01 = conic.x * h01 + conic.y * h11;
  let t10 = conic.y * h00 + conic.z * h01;
  let t11 = conic.y * h01 + conic.z * h11;
  return vec3f(
    -(t00 * conic.x + t01 * conic.y),
    -2.0 * (t00 * conic.y + t01 * conic.z),
    -(t10 * conic.y + t11 * conic.z)
  );
}

fn anisoRotationQuaternionGradient(rotation_gradient : mat3x3f, q : vec4f) -> vec4f {
  let g00 = rotation_gradient[0].x; let g10 = rotation_gradient[0].y; let g20 = rotation_gradient[0].z;
  let g01 = rotation_gradient[1].x; let g11 = rotation_gradient[1].y; let g21 = rotation_gradient[1].z;
  let g02 = rotation_gradient[2].x; let g12 = rotation_gradient[2].y; let g22 = rotation_gradient[2].z;
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return vec4f(
    2.0 * y * (g01 + g10) + 2.0 * z * (g02 + g20) - 4.0 * x * (g11 + g22) + 2.0 * w * (g21 - g12),
    -4.0 * y * (g00 + g22) + 2.0 * x * (g01 + g10) + 2.0 * z * (g12 + g21) + 2.0 * w * (g02 - g20),
    -4.0 * z * (g00 + g11) + 2.0 * x * (g02 + g20) + 2.0 * y * (g12 + g21) + 2.0 * w * (g10 - g01),
    2.0 * z * (g10 - g01) + 2.0 * y * (g02 - g20) + 2.0 * x * (g21 - g12)
  );
}

fn anisoNormalizedQuaternionGradient(
  raw : vec4f,
  normalized : vec4f,
  gradient : vec4f,
  epsilon : f32
) -> vec4f {
  let norm2 = dot(raw, raw);
  let norm = sqrt(max(norm2, epsilon * epsilon));
  if (norm2 <= epsilon * epsilon) {
    return gradient / norm;
  }
  return (gradient - normalized * dot(normalized, gradient)) / norm;
}

fn anisoProjectBackward(
  position : vec3f,
  log_scale : vec3f,
  quaternion : vec4f,
  camera : AnisoCamera,
  settings : AnisoProjectionSettings,
  upstream : AnisoProjectionUpstream
) -> AnisoProjectionGradient {
  let projected = anisoProject(position, log_scale, quaternion, camera, settings);
  let safe_z = max(projected.camera_position.z, camera.near);
  let jacobian = anisoCovarianceJacobian(projected.camera_position, camera.focal_px, safe_z);
  let mean_jacobian = anisoMeanJacobian(projected.camera_position, camera.focal_px, safe_z);
  let rotation = anisoQuaternionRotation(projected.normalized_quaternion);
  let rotation_camera = mat3x3f(
    anisoWorldToCamera(camera, rotation[0]),
    anisoWorldToCamera(camera, rotation[1]),
    anisoWorldToCamera(camera, rotation[2])
  );
  let transform = mat3x3f(
    rotation_camera[0] * projected.scales.x,
    rotation_camera[1] * projected.scales.y,
    rotation_camera[2] * projected.scales.z
  );
  let axis0 = jacobian * transform[0];
  let axis1 = jacobian * transform[1];
  let axis2 = jacobian * transform[2];
  let covariance_gradient = anisoConicToCovarianceGradient(projected.conic, upstream.conic);
  let axis_gradient0 = vec2f(
    2.0 * covariance_gradient.x * axis0.x + covariance_gradient.y * axis0.y,
    covariance_gradient.y * axis0.x + 2.0 * covariance_gradient.z * axis0.y
  );
  let axis_gradient1 = vec2f(
    2.0 * covariance_gradient.x * axis1.x + covariance_gradient.y * axis1.y,
    covariance_gradient.y * axis1.x + 2.0 * covariance_gradient.z * axis1.y
  );
  let axis_gradient2 = vec2f(
    2.0 * covariance_gradient.x * axis2.x + covariance_gradient.y * axis2.y,
    covariance_gradient.y * axis2.x + 2.0 * covariance_gradient.z * axis2.y
  );
  let transform_gradient0 = transpose(jacobian) * axis_gradient0;
  let transform_gradient1 = transpose(jacobian) * axis_gradient1;
  let transform_gradient2 = transpose(jacobian) * axis_gradient2;
  let jacobian_gradient = mat3x2f(
    axis_gradient0 * transform[0].x + axis_gradient1 * transform[1].x + axis_gradient2 * transform[2].x,
    axis_gradient0 * transform[0].y + axis_gradient1 * transform[1].y + axis_gradient2 * transform[2].y,
    axis_gradient0 * transform[0].z + axis_gradient1 * transform[1].z + axis_gradient2 * transform[2].z
  );

  let rotation_gradient = mat3x3f(
    anisoCameraToWorld(camera, transform_gradient0) * projected.scales.x,
    anisoCameraToWorld(camera, transform_gradient1) * projected.scales.y,
    anisoCameraToWorld(camera, transform_gradient2) * projected.scales.z
  );
  let raw_scale = exp(log_scale);
  let scale_gate = vec3f(
    select(0.0, 1.0, raw_scale.x > settings.min_scale && raw_scale.x < settings.max_scale),
    select(0.0, 1.0, raw_scale.y > settings.min_scale && raw_scale.y < settings.max_scale),
    select(0.0, 1.0, raw_scale.z > settings.min_scale && raw_scale.z < settings.max_scale)
  );
  let log_scale_gradient = vec3f(
    dot(transform_gradient0, rotation_camera[0]),
    dot(transform_gradient1, rotation_camera[1]),
    dot(transform_gradient2, rotation_camera[2])
  ) * raw_scale * scale_gate;
  let normalized_quaternion_gradient = anisoRotationQuaternionGradient(
    rotation_gradient,
    projected.normalized_quaternion
  );
  let quaternion_gradient = anisoNormalizedQuaternionGradient(
    quaternion,
    projected.normalized_quaternion,
    normalized_quaternion_gradient,
    settings.quaternion_epsilon
  );

  var camera_gradient = transpose(mean_jacobian) * upstream.mean_px;
  let inv_z = 1.0 / safe_z;
  let inv_z2 = inv_z * inv_z;
  let inv_z3 = inv_z2 * inv_z;${jacobianPositionBackward}
  camera_gradient.z = camera_gradient.z * select(0.0, 1.0, projected.camera_position.z > camera.near);

  return AnisoProjectionGradient(
    anisoCameraToWorld(camera, camera_gradient),
    log_scale_gradient,
    quaternion_gradient
  );
}
`;
}
