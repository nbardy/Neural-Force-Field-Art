/**
 * Correctness-first differentiable raster for 32-channel isotropic splats.
 *
 * Splat features are splat-major vec4 groups. Raster images are channel-planar
 * f32 so their storage is directly compatible with Feature32Colorizer.
 */

export const FEATURE_RASTER_CHANNELS = 32;
export const FEATURE_RASTER_GROUPS = FEATURE_RASTER_CHANNELS / 4;
export const FEATURE_RASTER_WORKGROUP_SIZE = 64;
export const FEATURE_RASTER_DEFAULT_MIN_RADIUS = 0.25;
export const FEATURE_RASTER_DEFAULT_MAX_OPACITY = 0.99;

export interface FeatureRasterShaderConfig {
  width: number;
  height: number;
  splats: number;
  minRadius?: number;
  maxOpacity?: number;
}

interface ResolvedFeatureRasterShaderConfig {
  width: number;
  height: number;
  pixels: number;
  splats: number;
  minRadius: number;
  maxOpacity: number;
}

function f32(value: number): string {
  const text = value.toString();
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

function resolveConfig(input: FeatureRasterShaderConfig): ResolvedFeatureRasterShaderConfig {
  if (!Number.isInteger(input.width) || input.width <= 0 || !Number.isInteger(input.height) || input.height <= 0) {
    throw new Error(`feature raster: invalid image shape ${input.width}x${input.height}`);
  }
  if (!Number.isInteger(input.splats) || input.splats <= 0) {
    throw new Error(`feature raster: splats must be a positive integer, got ${input.splats}`);
  }
  const minRadius = input.minRadius ?? FEATURE_RASTER_DEFAULT_MIN_RADIUS;
  const maxOpacity = input.maxOpacity ?? FEATURE_RASTER_DEFAULT_MAX_OPACITY;
  if (!Number.isFinite(minRadius) || minRadius <= 0) {
    throw new Error(`feature raster: minRadius must be finite and positive, got ${minRadius}`);
  }
  if (!Number.isFinite(maxOpacity) || maxOpacity <= 0 || maxOpacity >= 1) {
    throw new Error(`feature raster: maxOpacity must be in (0, 1), got ${maxOpacity}`);
  }
  return {
    width: input.width,
    height: input.height,
    pixels: input.width * input.height,
    splats: input.splats,
    minRadius,
    maxOpacity,
  };
}

function common(config: ResolvedFeatureRasterShaderConfig): string {
  return /* wgsl */ `
const WIDTH : u32 = ${config.width}u;
const PIXELS : u32 = ${config.pixels}u;
const SPLATS : u32 = ${config.splats}u;
const FEATURE_GROUPS : u32 = ${FEATURE_RASTER_GROUPS}u;
const MIN_RADIUS : f32 = ${f32(config.minRadius)};
const MAX_OPACITY : f32 = ${f32(config.maxOpacity)};

fn imageIndex(channel : u32, pixel : u32) -> u32 {
  return channel * PIXELS + pixel;
}

fn splatFeatureIndex(splat : u32, group : u32) -> u32 {
  return splat * FEATURE_GROUPS + group;
}

fn sigmoid(x : f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn alphaAt(geometry : vec4f, pixelCenter : vec2f) -> f32 {
  let radius = max(exp(geometry.z), MIN_RADIUS);
  let delta = pixelCenter - geometry.xy;
  let gaussian = exp(-0.5 * dot(delta, delta) / (radius * radius));
  return MAX_OPACITY * sigmoid(geometry.w) * gaussian;
}
`;
}

export function featureRasterForwardShader(input: FeatureRasterShaderConfig): string {
  const config = resolveConfig(input);
  return /* wgsl */ `
${common(config)}
@group(0) @binding(0) var<storage, read> geometry : array<vec4f>;
@group(0) @binding(1) var<storage, read> splatFeatures : array<vec4f>;
@group(0) @binding(2) var<storage, read> sortedIds : array<u32>;
@group(0) @binding(3) var<storage, read> background : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> imageFeatures : array<f32>;

@compute @workgroup_size(${FEATURE_RASTER_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let pixel = gid.x;
  if (pixel >= PIXELS) { return; }
  let x = pixel % WIDTH;
  let y = pixel / WIDTH;
  let pixelCenter = vec2f(f32(x) + 0.5, f32(y) + 0.5);

  var accumulated : array<vec4f, ${FEATURE_RASTER_GROUPS}>;
  for (var group = 0u; group < FEATURE_GROUPS; group++) {
    accumulated[group] = vec4f(0.0);
  }
  var transmittance = 1.0;
  for (var order = 0u; order < SPLATS; order++) {
    let splat = sortedIds[order];
    let alpha = alphaAt(geometry[splat], pixelCenter);
    let weight = transmittance * alpha;
    for (var group = 0u; group < FEATURE_GROUPS; group++) {
      accumulated[group] += weight * splatFeatures[splatFeatureIndex(splat, group)];
    }
    transmittance *= 1.0 - alpha;
  }

  for (var group = 0u; group < FEATURE_GROUPS; group++) {
    let value = accumulated[group] + transmittance * background[group];
    let channel = group * 4u;
    imageFeatures[imageIndex(channel, pixel)] = value.x;
    imageFeatures[imageIndex(channel + 1u, pixel)] = value.y;
    imageFeatures[imageIndex(channel + 2u, pixel)] = value.z;
    imageFeatures[imageIndex(channel + 3u, pixel)] = value.w;
  }
}
`;
}

export function featureRasterBackwardShader(input: FeatureRasterShaderConfig): string {
  const config = resolveConfig(input);
  return /* wgsl */ `
${common(config)}
@group(0) @binding(0) var<storage, read> geometry : array<vec4f>;
@group(0) @binding(1) var<storage, read> splatFeatures : array<vec4f>;
@group(0) @binding(2) var<storage, read> sortedIds : array<u32>;
@group(0) @binding(3) var<storage, read> background : array<vec4f>;
@group(0) @binding(4) var<storage, read> imageFeatureGrad : array<f32>;
@group(0) @binding(5) var<storage, read_write> geometryGrad : array<vec4f>;
@group(0) @binding(6) var<storage, read_write> splatFeatureGrad : array<vec4f>;

// One invocation owns every gradient write. This is intentionally serial: it
// is the exact reference path used to validate a future tiled/atomic raster.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x != 0u) { return; }
  for (var splat = 0u; splat < SPLATS; splat++) {
    geometryGrad[splat] = vec4f(0.0);
    for (var group = 0u; group < FEATURE_GROUPS; group++) {
      splatFeatureGrad[splatFeatureIndex(splat, group)] = vec4f(0.0);
    }
  }

  for (var pixel = 0u; pixel < PIXELS; pixel++) {
    let x = pixel % WIDTH;
    let y = pixel / WIDTH;
    let pixelCenter = vec2f(f32(x) + 0.5, f32(y) + 0.5);
    var upstream : array<vec4f, ${FEATURE_RASTER_GROUPS}>;
    for (var group = 0u; group < FEATURE_GROUPS; group++) {
      let channel = group * 4u;
      upstream[group] = vec4f(
        imageFeatureGrad[imageIndex(channel, pixel)],
        imageFeatureGrad[imageIndex(channel + 1u, pixel)],
        imageFeatureGrad[imageIndex(channel + 2u, pixel)],
        imageFeatureGrad[imageIndex(channel + 3u, pixel)]
      );
    }

    var transmittanceAfter = 1.0;
    for (var order = 0u; order < SPLATS; order++) {
      let splat = sortedIds[order];
      transmittanceAfter *= 1.0 - alphaAt(geometry[splat], pixelCenter);
    }
    var transmittanceGradient = 0.0;
    for (var group = 0u; group < FEATURE_GROUPS; group++) {
      transmittanceGradient += dot(upstream[group], background[group]);
    }

    var reverseOrder = SPLATS;
    loop {
      if (reverseOrder == 0u) { break; }
      reverseOrder -= 1u;
      let splat = sortedIds[reverseOrder];
      let packedGeometry = geometry[splat];
      let radiusUnclamped = exp(packedGeometry.z);
      let radius = max(radiusUnclamped, MIN_RADIUS);
      let delta = pixelCenter - packedGeometry.xy;
      let inverseRadius2 = 1.0 / (radius * radius);
      let gaussian = exp(-0.5 * dot(delta, delta) * inverseRadius2);
      let opacityUnit = sigmoid(packedGeometry.w);
      let opacity = MAX_OPACITY * opacityUnit;
      let alpha = opacity * gaussian;
      let oneMinusAlpha = 1.0 - alpha;
      let transmittanceBefore = transmittanceAfter / oneMinusAlpha;

      var featureDot = 0.0;
      for (var group = 0u; group < FEATURE_GROUPS; group++) {
        let index = splatFeatureIndex(splat, group);
        let feature = splatFeatures[index];
        featureDot += dot(upstream[group], feature);
        splatFeatureGrad[index] += transmittanceBefore * alpha * upstream[group];
      }
      let alphaGradient = transmittanceBefore * (featureDot - transmittanceGradient);
      let geometryGradient = geometryGrad[splat];
      let meanGradient = alphaGradient * alpha * delta * inverseRadius2;
      let radiusActive = select(0.0, 1.0, radiusUnclamped > MIN_RADIUS);
      let logRadiusGradient = alphaGradient * alpha * dot(delta, delta) * inverseRadius2 * radiusActive;
      let opacityGradient = alphaGradient * gaussian * MAX_OPACITY * opacityUnit * (1.0 - opacityUnit);
      geometryGrad[splat] = geometryGradient + vec4f(
        meanGradient,
        logRadiusGradient,
        opacityGradient
      );

      transmittanceGradient = alpha * featureDot + oneMinusAlpha * transmittanceGradient;
      transmittanceAfter = transmittanceBefore;
    }
  }
}
`;
}
