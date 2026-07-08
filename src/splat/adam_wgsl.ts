/**
 * adam_wgsl — PURE WGSL codegen for the fused Adam optimizer over the raster's
 * raw parameter buffers (src/splat/raster_wgsl.ts data model).
 *
 * One generic kernel, thread/param over a contiguous SoA segment [offset,
 * offset+count) of the shared params/grad/m/v buffers. It is dispatched once
 * per parameter GROUP (mean, logScale, theta, color, opacity) with that group's
 * learning rate — so each dispatch has ONE learning rate and there is no
 * per-thread group lookup (the group distinction is data in the uniform, not
 * structural branching in the shader). Bias-corrected update; the caller passes
 * bc1 = 1-beta1^t and bc2 = 1-beta2^t so t need not be an integer in the shader.
 *
 * The i32 fixed-point gradient scale is already undone by the raster `chain`
 * kernel, so `grad` here is a true f32 raw-parameter gradient — Adam applies no
 * further scaling. (Adam's m/sqrt(v) is scale-invariant anyway; unscaling in
 * `chain` keeps this kernel a plain textbook Adam.)
 *
 * Zero imports; the uniform layout is the ONLY contract with the runtime.
 */

/** Adam uniform: 8 x 4 bytes = 32 bytes (std140-safe: all scalars). */
export const ADAM_UNIFORM_BYTES = 32;

export function adamShader(): string {
  return /* wgsl */ `
struct AdamU {
  offset : u32,
  count  : u32,
  lr     : f32,
  beta1  : f32,
  beta2  : f32,
  eps    : f32,
  bc1    : f32,   // 1 - beta1^t
  bc2    : f32,   // 1 - beta2^t
};
@group(0) @binding(0) var<uniform>              u      : AdamU;
@group(0) @binding(1) var<storage, read_write>  params : array<f32>;
@group(0) @binding(2) var<storage, read>        grad   : array<f32>;
@group(0) @binding(3) var<storage, read_write>  mBuf   : array<f32>;
@group(0) @binding(4) var<storage, read_write>  vBuf   : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= u.count) { return; }
  let idx = u.offset + i;
  let g = grad[idx];
  let m = u.beta1 * mBuf[idx] + (1.0 - u.beta1) * g;
  let v = u.beta2 * vBuf[idx] + (1.0 - u.beta2) * g * g;
  mBuf[idx] = m;
  vBuf[idx] = v;
  let mhat = m / u.bc1;
  let vhat = v / u.bc2;
  params[idx] = params[idx] - u.lr * mhat / (sqrt(vhat) + u.eps);
}
`;
}

/** Per-group learning rates (spec defaults: mean 1e-2, others 5e-3). */
export interface AdamLRs {
  mean: number;
  logScale: number;
  theta: number;
  color: number;
  opacity: number;
}

export const DEFAULT_LRS: AdamLRs = {
  mean: 1e-2,
  logScale: 5e-3,
  theta: 5e-3,
  color: 5e-3,
  opacity: 5e-3,
};

export interface AdamHyper {
  beta1: number;
  beta2: number;
  eps: number;
}

export const DEFAULT_HYPER: AdamHyper = { beta1: 0.9, beta2: 0.999, eps: 1e-8 };
