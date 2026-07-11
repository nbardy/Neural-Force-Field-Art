/// <reference types="@webgpu/types" />

const U = { COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const WORKGROUP_X = 8;
const WORKGROUP_Y = 8;
const UNIFORM_BYTES = 16;
const UINT32_MAX = 0xffff_ffff;

export type BackgroundTextureMode =
  | "black"
  | "dark_solid"
  | "blurred_noise"
  | "checkerboard"
  | "fourier";

export interface BackgroundTextureConfig {
  H: number;
  W: number;
  mode: BackgroundTextureMode;
  seed?: number;
  label?: string;
}

const MODE_IDS: Record<BackgroundTextureMode, number> = {
  black: 0,
  dark_solid: 1,
  blurred_noise: 2,
  checkerboard: 3,
  fourier: 4,
};

function assertUint32(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0 || value > UINT32_MAX) {
    throw new Error(`background_textures: ${name} must be an unsigned 32-bit integer`);
  }
  return value >>> 0;
}

function assertDimension(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`background_textures: ${name} must be a positive integer`);
  }
  return value;
}

function backgroundShader(H: number, W: number, mode: BackgroundTextureMode): string {
  const hw = H * W;
  const minSide = Math.min(H, W);
  const blurredCell = Math.max(4, minSide * 0.16);
  const checkerCell = Math.max(2, Math.floor(minSide / 8));
  return /* wgsl */ `
struct GenerateUniforms {
  step     : u32,
  seed     : u32,
  strength : f32,
  _pad     : u32,
};

@group(0) @binding(0) var<uniform> uniforms : GenerateUniforms;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

const WIDTH : u32 = ${W}u;
const HEIGHT : u32 = ${H}u;
const PIXELS : u32 = ${hw}u;
const MODE : u32 = ${MODE_IDS[mode]}u;
const TAU : f32 = 6.283185307179586;
const BLURRED_CELL : f32 = ${blurredCell};
const CHECKER_CELL : u32 = ${checkerCell}u;

fn hash32(value : u32) -> u32 {
  var x = value;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn keyedHash(a : u32, b : u32, c : u32) -> u32 {
  var h = hash32(uniforms.seed ^ 0xa511e9b3u);
  h = hash32(h ^ hash32(uniforms.step + 0x63d83595u));
  h = hash32(h ^ hash32(a + 0x9e3779b9u));
  h = hash32(h ^ hash32(b + 0x85ebca6bu));
  return hash32(h ^ hash32(c + 0xc2b2ae35u));
}

fn random01(a : u32, b : u32, c : u32) -> f32 {
  return f32(keyedHash(a, b, c) >> 8u) * 0.000000059604644775390625;
}

fn valueNoise(pixel : vec2f, channel : u32, octave : u32) -> f32 {
  let cellSize = max(2.0, BLURRED_CELL / exp2(f32(octave)));
  let offset = vec2f(
    random01(channel, octave, 101u),
    random01(channel, octave, 211u)
  ) * cellSize;
  let p = (pixel + offset) / cellSize;
  let base = vec2u(floor(p));
  let f = fract(p);
  let curve = f * f * (vec2f(3.0) - 2.0 * f);
  let salt = channel * 17u + octave * 131u;
  let n00 = random01(base.x, base.y, salt);
  let n10 = random01(base.x + 1u, base.y, salt);
  let n01 = random01(base.x, base.y + 1u, salt);
  let n11 = random01(base.x + 1u, base.y + 1u, salt);
  return mix(mix(n00, n10, curve.x), mix(n01, n11, curve.x), curve.y);
}

fn blurredNoise(pixel : vec2f, channel : u32) -> f32 {
  var total = 0.0;
  var weight = 0.0;
  var amplitude = 1.0;
  for (var octave = 0u; octave < 3u; octave = octave + 1u) {
    total = total + amplitude * valueNoise(pixel, channel, octave);
    weight = weight + amplitude;
    amplitude = amplitude * 0.5;
  }
  return total / weight;
}

fn fourierValue(pixel : vec2f, channel : u32) -> f32 {
  let p = pixel / ${minSide}.0;
  var total = 0.0;
  var weight = 0.0;
  for (var wave = 0u; wave < 6u; wave = wave + 1u) {
    let angle = TAU * random01(channel, wave, 307u);
    let direction = vec2f(cos(angle), sin(angle));
    let frequency = 0.75 + 5.25 * random01(channel, wave, 401u);
    let phase = TAU * random01(channel, wave, 503u);
    let amplitude = 1.0 / (1.0 + 0.35 * f32(wave));
    total = total + amplitude * sin(TAU * frequency * dot(p, direction) + phase);
    weight = weight + amplitude;
  }
  return clamp(0.5 + 0.5 * total / weight, 0.0, 1.0);
}

fn darkSolid() -> vec3f {
  return vec3f(
    0.015 + 0.075 * random01(1u, 0u, 601u),
    0.012 + 0.070 * random01(1u, 1u, 601u),
    0.018 + 0.080 * random01(1u, 2u, 601u)
  );
}

fn blurredTexture(pixel : vec2f) -> vec3f {
  let baseNoise = blurredNoise(pixel, 0u);
  let detail = vec3f(
    blurredNoise(pixel, 1u),
    blurredNoise(pixel, 2u),
    blurredNoise(pixel, 3u)
  );
  let value = vec3f(baseNoise) * 0.72 + detail * 0.28;
  return vec3f(0.012, 0.010, 0.014) + value * vec3f(0.17, 0.16, 0.19);
}

fn checkerTexture(coord : vec2u) -> vec3f {
  let offset = vec2u(
    keyedHash(7u, 0u, 701u) % CHECKER_CELL,
    keyedHash(7u, 1u, 701u) % CHECKER_CELL
  );
  let checker = ((coord.x + offset.x) / CHECKER_CELL + (coord.y + offset.y) / CHECKER_CELL) & 1u;
  let low = vec3f(
    0.012 + 0.025 * random01(0u, 0u, 709u),
    0.014 + 0.022 * random01(0u, 1u, 709u),
    0.018 + 0.028 * random01(0u, 2u, 709u)
  );
  let high = vec3f(
    0.105 + 0.080 * random01(1u, 0u, 719u),
    0.095 + 0.075 * random01(1u, 1u, 719u),
    0.115 + 0.085 * random01(1u, 2u, 719u)
  );
  return select(low, high, checker == 1u);
}

fn fourierTexture(pixel : vec2f) -> vec3f {
  let baseWave = fourierValue(pixel, 0u);
  let value = vec3f(
    0.68 * baseWave + 0.32 * fourierValue(pixel, 1u),
    0.72 * baseWave + 0.28 * fourierValue(pixel, 2u),
    0.64 * baseWave + 0.36 * fourierValue(pixel, 3u)
  );
  return vec3f(0.012, 0.010, 0.014) + value * vec3f(0.25, 0.23, 0.28);
}

@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y})
fn main(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= WIDTH || gid.y >= HEIGHT) { return; }

  let index = gid.y * WIDTH + gid.x;
  let pixel = vec2f(gid.xy) + vec2f(0.5);
  var rgb = vec3f(0.0);
  if (MODE == 1u) {
    rgb = darkSolid();
  } else if (MODE == 2u) {
    rgb = blurredTexture(pixel);
  } else if (MODE == 3u) {
    rgb = checkerTexture(gid.xy);
  } else if (MODE == 4u) {
    rgb = fourierTexture(pixel);
  }
  rgb = clamp(rgb * uniforms.strength, vec3f(0.0), vec3f(1.0));

  output[index] = rgb.r;
  output[PIXELS + index] = rgb.g;
  output[2u * PIXELS + index] = rgb.b;
}
`;
}

export class BackgroundTextureGenerator {
  readonly H: number;
  readonly W: number;
  readonly mode: BackgroundTextureMode;
  readonly seed: number;
  readonly buffer: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly uniformBuffers: GPUBuffer[];
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroups: GPUBindGroup[];
  private readonly dispatchX: number;
  private readonly dispatchY: number;
  private destroyed = false;

  private constructor(
    device: GPUDevice,
    cfg: Required<Pick<BackgroundTextureConfig, "H" | "W" | "mode" | "seed">> & { label: string },
    buffer: GPUBuffer,
    uniformBuffers: GPUBuffer[],
    pipeline: GPUComputePipeline,
    bindGroups: GPUBindGroup[]
  ) {
    this.device = device;
    this.H = cfg.H;
    this.W = cfg.W;
    this.mode = cfg.mode;
    this.seed = cfg.seed;
    this.buffer = buffer;
    this.uniformBuffers = uniformBuffers;
    this.pipeline = pipeline;
    this.bindGroups = bindGroups;
    this.dispatchX = Math.ceil(cfg.W / WORKGROUP_X);
    this.dispatchY = Math.ceil(cfg.H / WORKGROUP_Y);
  }

  static async create(device: GPUDevice, cfg: BackgroundTextureConfig): Promise<BackgroundTextureGenerator> {
    const H = assertDimension(cfg.H, "H");
    const W = assertDimension(cfg.W, "W");
    if (!Object.prototype.hasOwnProperty.call(MODE_IDS, cfg.mode)) {
      throw new Error(`background_textures: unsupported mode ${String(cfg.mode)}`);
    }
    const seed = assertUint32(cfg.seed ?? 0, "seed");
    const pixels = H * W;
    if (!Number.isSafeInteger(pixels) || pixels > Math.floor(UINT32_MAX / 3)) {
      throw new Error("background_textures: H*W is too large for planar u32 indexing");
    }
    const byteSize = pixels * 3 * 4;
    const maxStorageSize = Number(device.limits.maxStorageBufferBindingSize);
    const maxBufferSize = Number(device.limits.maxBufferSize);
    if (byteSize > maxStorageSize || byteSize > maxBufferSize) {
      throw new Error(`background_textures: ${byteSize} byte output exceeds device storage limits`);
    }
    const dispatchX = Math.ceil(W / WORKGROUP_X);
    const dispatchY = Math.ceil(H / WORKGROUP_Y);
    const maxDispatch = Number(device.limits.maxComputeWorkgroupsPerDimension);
    if (dispatchX > maxDispatch || dispatchY > maxDispatch) {
      throw new Error(`background_textures: ${dispatchX}x${dispatchY} dispatch exceeds device limits`);
    }

    const label = cfg.label ?? `splat3d-background-${cfg.mode}`;
    const buffer = device.createBuffer({
      label: `${label}-rgb`,
      size: byteSize,
      usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
    });
    const uniformBuffers = Array.from({ length: 2 }, (_unused, slot) =>
      device.createBuffer({
        label: `${label}-uniforms-${slot}`,
        size: UNIFORM_BYTES,
        usage: U.UNIFORM | U.COPY_DST,
      })
    );

    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label: `${label}-shader`, code: backgroundShader(H, W, cfg.mode) });
    const pipeline = device.createComputePipeline({
      label: `${label}-pipeline`,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const error = await device.popErrorScope();
    if (error) {
      buffer.destroy();
      for (const buffer of uniformBuffers) buffer.destroy();
      throw new Error(`background_textures: pipeline validation failed: ${error.message}`);
    }

    const bindGroups = uniformBuffers.map((uniformBuffer, slot) =>
      device.createBindGroup({
        label: `${label}-bind-group-${slot}`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer } },
        ],
      })
    );
    return new BackgroundTextureGenerator(
      device,
      { H, W, mode: cfg.mode, seed, label },
      buffer,
      uniformBuffers,
      pipeline,
      bindGroups
    );
  }

  recordGenerate(commandEncoder: GPUCommandEncoder, step: number, strength: number, slot = 0): void {
    if (this.destroyed) throw new Error("background_textures: generator has been destroyed");
    const normalizedStep = assertUint32(step, "step");
    if (!Number.isFinite(strength)) {
      throw new Error("background_textures: strength must be finite");
    }
    const uniforms = new ArrayBuffer(UNIFORM_BYTES);
    const u32 = new Uint32Array(uniforms);
    const f32 = new Float32Array(uniforms);
    u32[0] = normalizedStep;
    u32[1] = this.seed;
    f32[2] = Math.max(0, Math.min(1, strength));
    if (!Number.isInteger(slot) || slot < 0 || slot >= this.uniformBuffers.length) {
      throw new Error(`background_textures: uniform slot ${slot} is outside [0, ${this.uniformBuffers.length})`);
    }
    this.device.queue.writeBuffer(this.uniformBuffers[slot], 0, uniforms);

    const pass = commandEncoder.beginComputePass({ label: `splat3d-background-${this.mode}-generate` });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroups[slot]);
    pass.dispatchWorkgroups(this.dispatchX, this.dispatchY);
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.buffer.destroy();
    for (const buffer of this.uniformBuffers) buffer.destroy();
  }
}
