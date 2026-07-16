/**
 * Pure WGSL emitters for the feature32 -> RGB residual colorizer.
 *
 * Images are batch-major, channel-planar f32. Each channel plane is viewed as
 * vec4 pixel groups, so pixels must be a multiple of four. Weights are packed
 * row-major as [rgb channel][8 feature groups] of vec4<f32>.
 */

export const FEATURE32_CHANNELS = 32;
export const FEATURE32_RGB_CHANNELS = 3;
export const FEATURE32_GROUPS = FEATURE32_CHANNELS / 4;
export const FEATURE32_WEIGHT_FLOATS = FEATURE32_CHANNELS * FEATURE32_RGB_CHANNELS;
export const FEATURE32_BIAS_FLOATS = 4;
export const FEATURE32_WORKGROUP_SIZE = 64;
export const FEATURE32_DEFAULT_RESIDUAL_SCALE = 0.1;

export interface Feature32ColorizerShaderConfig {
  pixels: number;
  batch: number;
  residualScale?: number;
}

interface ResolvedShaderConfig {
  pixels: number;
  pixelGroups: number;
  batch: number;
  totalPixelGroups: number;
  residualScale: number;
}

function resolveConfig(config: Feature32ColorizerShaderConfig): ResolvedShaderConfig {
  if (!Number.isInteger(config.pixels) || config.pixels <= 0 || config.pixels % 4 !== 0) {
    throw new Error(`feature32 colorizer: pixels must be a positive multiple of 4, got ${config.pixels}`);
  }
  if (!Number.isInteger(config.batch) || config.batch <= 0) {
    throw new Error(`feature32 colorizer: batch must be a positive integer, got ${config.batch}`);
  }
  const residualScale = config.residualScale ?? FEATURE32_DEFAULT_RESIDUAL_SCALE;
  if (!Number.isFinite(residualScale) || residualScale < 0) {
    throw new Error(`feature32 colorizer: residualScale must be finite and non-negative, got ${residualScale}`);
  }
  return {
    pixels: config.pixels,
    pixelGroups: config.pixels / 4,
    batch: config.batch,
    totalPixelGroups: (config.pixels / 4) * config.batch,
    residualScale,
  };
}

function f32(value: number): string {
  const text = value.toString();
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

function common(config: ResolvedShaderConfig): string {
  return /* wgsl */ `
const PIXEL_GROUPS : u32 = ${config.pixelGroups}u;
const TOTAL_PIXEL_GROUPS : u32 = ${config.totalPixelGroups}u;
const FEATURE_GROUPS : u32 = ${FEATURE32_GROUPS}u;
const RESIDUAL_SCALE : f32 = ${f32(config.residualScale)};

fn featureIndex(image : u32, channel : u32, pixelGroup : u32) -> u32 {
  return (image * ${FEATURE32_CHANNELS}u + channel) * PIXEL_GROUPS + pixelGroup;
}

fn rgbIndex(image : u32, channel : u32, pixelGroup : u32) -> u32 {
  return (image * ${FEATURE32_RGB_CHANNELS}u + channel) * PIXEL_GROUPS + pixelGroup;
}

fn sigmoid4(x : vec4f) -> vec4f {
  return vec4f(1.0) / (vec4f(1.0) + exp(-x));
}
`;
}

function forwardFeatureGroups(): string {
  return Array.from({ length: FEATURE32_GROUPS }, (_unused, group) => {
    const channel = group * 4;
    return /* wgsl */ `
  {
    let x0 = features[featureIndex(image, ${channel}u, pixelGroup)];
    let x1 = features[featureIndex(image, ${channel + 1}u, pixelGroup)];
    let x2 = features[featureIndex(image, ${channel + 2}u, pixelGroup)];
    let x3 = features[featureIndex(image, ${channel + 3}u, pixelGroup)];
    let wr = weights[${group}u];
    let wg = weights[${FEATURE32_GROUPS + group}u];
    let wb = weights[${2 * FEATURE32_GROUPS + group}u];
    residualR = fma(x0, vec4f(wr.x), residualR);
    residualR = fma(x1, vec4f(wr.y), residualR);
    residualR = fma(x2, vec4f(wr.z), residualR);
    residualR = fma(x3, vec4f(wr.w), residualR);
    residualG = fma(x0, vec4f(wg.x), residualG);
    residualG = fma(x1, vec4f(wg.y), residualG);
    residualG = fma(x2, vec4f(wg.z), residualG);
    residualG = fma(x3, vec4f(wg.w), residualG);
    residualB = fma(x0, vec4f(wb.x), residualB);
    residualB = fma(x1, vec4f(wb.y), residualB);
    residualB = fma(x2, vec4f(wb.z), residualB);
    residualB = fma(x3, vec4f(wb.w), residualB);
  }`;
  }).join("\n");
}

export function feature32ColorizerForwardShader(input: Feature32ColorizerShaderConfig): string {
  const config = resolveConfig(input);
  return /* wgsl */ `
${common(config)}
@group(0) @binding(0) var<storage, read> features : array<vec4f>;
@group(0) @binding(1) var<storage, read> weights : array<vec4f>;
@group(0) @binding(2) var<storage, read> bias : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> rgb : array<vec4f>;

@compute @workgroup_size(${FEATURE32_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let linearGroup = gid.x;
  if (linearGroup >= TOTAL_PIXEL_GROUPS) { return; }
  let image = linearGroup / PIXEL_GROUPS;
  let pixelGroup = linearGroup % PIXEL_GROUPS;

  var residualR = vec4f(bias[0].x);
  var residualG = vec4f(bias[0].y);
  var residualB = vec4f(bias[0].z);
${forwardFeatureGroups()}

  let logitsR = features[featureIndex(image, 0u, pixelGroup)] + RESIDUAL_SCALE * residualR;
  let logitsG = features[featureIndex(image, 1u, pixelGroup)] + RESIDUAL_SCALE * residualG;
  let logitsB = features[featureIndex(image, 2u, pixelGroup)] + RESIDUAL_SCALE * residualB;
  rgb[rgbIndex(image, 0u, pixelGroup)] = sigmoid4(logitsR);
  rgb[rgbIndex(image, 1u, pixelGroup)] = sigmoid4(logitsG);
  rgb[rgbIndex(image, 2u, pixelGroup)] = sigmoid4(logitsB);
}
`;
}

export function feature32ColorizerFeatureGradShader(input: Feature32ColorizerShaderConfig): string {
  const config = resolveConfig(input);
  return /* wgsl */ `
${common(config)}
@group(0) @binding(0) var<storage, read> rgbGrad : array<vec4f>;
@group(0) @binding(1) var<storage, read> rgb : array<vec4f>;
@group(0) @binding(2) var<storage, read> weights : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> featureGrad : array<vec4f>;

@compute @workgroup_size(${FEATURE32_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let linearGroup = gid.x;
  let featureGroup = gid.y;
  if (linearGroup >= TOTAL_PIXEL_GROUPS || featureGroup >= FEATURE_GROUPS) { return; }
  let image = linearGroup / PIXEL_GROUPS;
  let pixelGroup = linearGroup % PIXEL_GROUPS;

  let rgbR = rgb[rgbIndex(image, 0u, pixelGroup)];
  let rgbG = rgb[rgbIndex(image, 1u, pixelGroup)];
  let rgbB = rgb[rgbIndex(image, 2u, pixelGroup)];
  let dzR = rgbGrad[rgbIndex(image, 0u, pixelGroup)] * rgbR * (vec4f(1.0) - rgbR);
  let dzG = rgbGrad[rgbIndex(image, 1u, pixelGroup)] * rgbG * (vec4f(1.0) - rgbG);
  let dzB = rgbGrad[rgbIndex(image, 2u, pixelGroup)] * rgbB * (vec4f(1.0) - rgbB);

  let wr = weights[featureGroup];
  let wg = weights[FEATURE_GROUPS + featureGroup];
  let wb = weights[2u * FEATURE_GROUPS + featureGroup];
  var dx0 = RESIDUAL_SCALE * (dzR * wr.x + dzG * wg.x + dzB * wb.x);
  var dx1 = RESIDUAL_SCALE * (dzR * wr.y + dzG * wg.y + dzB * wb.y);
  var dx2 = RESIDUAL_SCALE * (dzR * wr.z + dzG * wg.z + dzB * wb.z);
  var dx3 = RESIDUAL_SCALE * (dzR * wr.w + dzG * wg.w + dzB * wb.w);

  if (featureGroup == 0u) {
    dx0 += dzR;
    dx1 += dzG;
    dx2 += dzB;
  }

  let channel = featureGroup * 4u;
  featureGrad[featureIndex(image, channel, pixelGroup)] = dx0;
  featureGrad[featureIndex(image, channel + 1u, pixelGroup)] = dx1;
  featureGrad[featureIndex(image, channel + 2u, pixelGroup)] = dx2;
  featureGrad[featureIndex(image, channel + 3u, pixelGroup)] = dx3;
}
`;
}

export function feature32ColorizerParameterGradShader(input: Feature32ColorizerShaderConfig): string {
  const config = resolveConfig(input);
  return /* wgsl */ `
${common(config)}
@group(0) @binding(0) var<storage, read> features : array<vec4f>;
@group(0) @binding(1) var<storage, read> rgbGrad : array<vec4f>;
@group(0) @binding(2) var<storage, read> rgb : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> weightGrad : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> biasGrad : array<vec4f>;

var<workgroup> reduceR : array<vec4f, ${FEATURE32_WORKGROUP_SIZE}>;
var<workgroup> reduceG : array<vec4f, ${FEATURE32_WORKGROUP_SIZE}>;
var<workgroup> reduceB : array<vec4f, ${FEATURE32_WORKGROUP_SIZE}>;
var<workgroup> reduceBias : array<vec4f, ${FEATURE32_WORKGROUP_SIZE}>;

@compute @workgroup_size(${FEATURE32_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId : vec3u,
  @builtin(local_invocation_id) localId : vec3u
) {
  let featureGroup = workgroupId.x;
  if (featureGroup >= FEATURE_GROUPS) { return; }
  let lane = localId.x;
  let channel = featureGroup * 4u;
  var gradR = vec4f(0.0);
  var gradG = vec4f(0.0);
  var gradB = vec4f(0.0);
  var gradBias = vec4f(0.0);
  var linearGroup = lane;

  loop {
    if (linearGroup >= TOTAL_PIXEL_GROUPS) { break; }
    let image = linearGroup / PIXEL_GROUPS;
    let pixelGroup = linearGroup % PIXEL_GROUPS;
    let rgbR = rgb[rgbIndex(image, 0u, pixelGroup)];
    let rgbG = rgb[rgbIndex(image, 1u, pixelGroup)];
    let rgbB = rgb[rgbIndex(image, 2u, pixelGroup)];
    let dzR = rgbGrad[rgbIndex(image, 0u, pixelGroup)] * rgbR * (vec4f(1.0) - rgbR);
    let dzG = rgbGrad[rgbIndex(image, 1u, pixelGroup)] * rgbG * (vec4f(1.0) - rgbG);
    let dzB = rgbGrad[rgbIndex(image, 2u, pixelGroup)] * rgbB * (vec4f(1.0) - rgbB);
    let x0 = features[featureIndex(image, channel, pixelGroup)];
    let x1 = features[featureIndex(image, channel + 1u, pixelGroup)];
    let x2 = features[featureIndex(image, channel + 2u, pixelGroup)];
    let x3 = features[featureIndex(image, channel + 3u, pixelGroup)];
    gradR += vec4f(dot(dzR, x0), dot(dzR, x1), dot(dzR, x2), dot(dzR, x3));
    gradG += vec4f(dot(dzG, x0), dot(dzG, x1), dot(dzG, x2), dot(dzG, x3));
    gradB += vec4f(dot(dzB, x0), dot(dzB, x1), dot(dzB, x2), dot(dzB, x3));
    if (featureGroup == 0u) {
      gradBias += vec4f(
        dot(dzR, vec4f(1.0)),
        dot(dzG, vec4f(1.0)),
        dot(dzB, vec4f(1.0)),
        0.0
      );
    }
    linearGroup += ${FEATURE32_WORKGROUP_SIZE}u;
  }

  reduceR[lane] = gradR;
  reduceG[lane] = gradG;
  reduceB[lane] = gradB;
  reduceBias[lane] = gradBias;
  workgroupBarrier();

  var stride = ${FEATURE32_WORKGROUP_SIZE / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      reduceR[lane] += reduceR[lane + stride];
      reduceG[lane] += reduceG[lane + stride];
      reduceB[lane] += reduceB[lane + stride];
      reduceBias[lane] += reduceBias[lane + stride];
    }
    workgroupBarrier();
    stride /= 2u;
  }

  if (lane == 0u) {
    weightGrad[featureGroup] = RESIDUAL_SCALE * reduceR[0];
    weightGrad[FEATURE_GROUPS + featureGroup] = RESIDUAL_SCALE * reduceG[0];
    weightGrad[2u * FEATURE_GROUPS + featureGroup] = RESIDUAL_SCALE * reduceB[0];
    if (featureGroup == 0u) {
      biasGrad[0] = RESIDUAL_SCALE * reduceBias[0];
    }
  }
}

`;
}

/** Small generic SGD update for the residual decoder parameters. Keeping this
 * separate from the splat Adam pass lets feature experiments own their compact
 * decoder without changing the proven splat optimizer contract. */
export function feature32ColorizerSgdShader(): string {
  return /* wgsl */ `
struct UpdateU { count : u32, _pad0 : u32, lr : f32, _pad1 : f32 };
@group(0) @binding(0) var<uniform> update : UpdateU;
@group(0) @binding(1) var<storage, read_write> parameter : array<vec4f>;
@group(0) @binding(2) var<storage, read> gradient : array<vec4f>;
@compute @workgroup_size(${FEATURE32_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= update.count) { return; }
  parameter[gid.x] = parameter[gid.x] - update.lr * gradient[gid.x];
}`;
}
