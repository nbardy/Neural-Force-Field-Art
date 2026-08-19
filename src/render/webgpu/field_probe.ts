/**
 * FIELD HEALTH PROBE — the AC/DC/saturation/Okubo–Weiss/direction-order
 * instrument, live.
 *
 * `tools/collapse_probe.ts` measures these offline on a tfjs field and is the
 * reason the collapse was found at all; nothing measured them on the RUNNING
 * artwork, which is why the default piece shipped laminar for months. This is
 * that same measurement, on the real field, at ~1 Hz.
 *
 * OBSERVATION ONLY. Nothing here writes a gradient buffer, touches `extGrads`,
 * or is recorded into the frame encoder. `sample()` builds its OWN
 * `GPUCommandEncoder` and submits it, so the fused hot path stays exactly one
 * `encodeStep` encoder → one `queue.submit`. On adversarial pieces that is the
 * whole point: a health metric that fed a loss would stop being a measurement
 * of whether the GAME produces structure.
 */

import {
  fieldProbeShader,
  WORKGROUP_SIZE,
  type FieldLayout,
} from "./advect_wgsl";

/**
 * Field metrics over a fixed grid. Definitions are transcribed from
 * `tools/collapse_probe.ts::diagnostics` so the live number and the offline
 * probe's number are the same statistic — `tools/field_probe_test.ts` gates
 * that against a closed-form field.
 */
export interface FieldMetrics {
  /** ‖mean_grid F‖ — the spatially CONSTANT (DC) mode: one global push. */
  dc: number;
  /** rms‖F − mean F‖ — the spatially VARYING (AC) mode, i.e. the structure. */
  ac: number;
  /** sqrt(mean ‖F‖²). Exactly sqrt(ac² + dc²) — the identity a test checks. */
  rmsF: number;
  /** Fraction of grid points with BOTH tanh components past ±0.9. */
  satFrac: number;
  /** Normalized Okubo–Weiss (⟨|S|²⟩−⟨ω²⟩)/(⟨|S|²⟩+⟨ω²⟩); < 0 ⇔ vortex-dominated. */
  okuboWeiss: number;
  /**
   * POLAR direction order over the grid: R₁ = ‖mean u‖ with the soft unit
   * u = F/√(‖F‖²+τ²), τ = {@link PROBE_TAU}. 0 = isotropic directions
   * (vortices), 1 = every sample points the same way (laminar streaks).
   * LOW IS HEALTHY — this is the direction-convergence number.
   */
  r1: number;
  /**
   * NEMATIC direction order: R₂ = ‖mean (uₓ²−u_y², 2uₓu_y)‖. R₁ alone is
   * escapable — a field that splits into ±F₀ counter-streaming sheets reads
   * R₁ ≈ 0 and looks exactly as laminar on screen (measured R₂ = 0.95 with the
   * polar term alone, `directionOrderLoss` docs). Read the pair, never R₁ only.
   */
  r2: number;
  /** Grid resolution per axis (the sample count is gridN²). */
  gridN: number;
}

/** Latest field measurement. `unprobed` is a state, not a zeroed metric set. */
export type FieldHealth =
  | { readonly tag: "unprobed" }
  | { readonly tag: "measured"; readonly metrics: FieldMetrics };

/**
 * Central-difference step in NORMALIZED coordinates, matching collapse_probe's
 * default `FD_H = 1/256`. Central (not forward) differences on purpose:
 * `forces()` is a plain forward pass, and taking the Jacobian through autograd
 * would be the second-order path the whole architecture exists to avoid.
 */
export const PROBE_FD_H = 1 / 256;

/**
 * Direction softener for R₁/R₂: u = F/√(‖F‖²+τ²), exact at F = 0 so a
 * zero-force cell contributes a zero vector rather than an arbitrary unit one.
 *
 * The value is the canonical soft-angle τ — the same 0.05 that
 * `ADVERSARY_OBJECTIVE_DEFAULTS.tau` and `GALLERY_ANTI_COLLAPSE.tau` use — so
 * the grid R₁ here and the batch R₁ the adversary reports are the SAME
 * statistic on two different sample populations, and the HUD can show them
 * side by side. It is duplicated rather than imported because this module is
 * deliberately tfjs-free and main.ts is not; `tools/field_probe_test.ts` §4
 * gates the two against each other on identical data, so a drift fails there.
 */
export const PROBE_TAU = 0.05;

/** Stencil offsets per grid point: centre, ±x, ±y — 5 evaluations each. */
const STENCIL: readonly (readonly [number, number])[] = [
  [0, 0],
  [PROBE_FD_H, 0],
  [-PROBE_FD_H, 0],
  [0, PROBE_FD_H],
  [0, -PROBE_FD_H],
];

/**
 * Build the probe point list: `gridN²` cell centres, then the four shifted
 * copies, contiguous per stencil offset so the reduction can slice them.
 */
export function probePoints(gridN: number): Float32Array {
  const n = gridN * gridN;
  const out = new Float32Array(n * STENCIL.length * 2);
  STENCIL.forEach(([dx, dy], s) => {
    const base = s * n * 2;
    for (let j = 0; j < gridN; j++) {
      for (let i = 0; i < gridN; i++) {
        const k = base + 2 * (j * gridN + i);
        out[k] = (i + 0.5) / gridN + dx;
        out[k + 1] = (j + 0.5) / gridN + dy;
      }
    }
  });
  return out;
}

/**
 * Reduce a probe readback into {@link FieldMetrics}. Pure — no GPU, no tfjs —
 * so it can be tested against a closed-form field on the CPU.
 *
 * `forces` is `probePoints(gridN)`'s output evaluated pointwise: 5 blocks of
 * `gridN²` vec2, in STENCIL order.
 */
export function fieldMetricsFrom(forces: Float32Array, gridN: number): FieldMetrics {
  const n = gridN * gridN;
  if (forces.length !== n * STENCIL.length * 2) {
    throw new Error(
      `field probe: expected ${n * STENCIL.length * 2} floats for gridN=${gridN}, ` +
        `got ${forces.length}`
    );
  }
  const at = (block: number, i: number): [number, number] => {
    const k = (block * n + i) * 2;
    return [forces[k], forces[k + 1]];
  };
  let sumX = 0;
  let sumY = 0;
  let sumSq = 0;
  let sat = 0;
  // Soft-unit direction moments — the same four the fused adversary's pass-A
  // reduction accumulates (adversary_wgsl.ts) and the same four
  // `directionOrderParameters` stacks. R₁/R₂ are read off them below.
  let sumUx = 0;
  let sumUy = 0;
  let sumC2 = 0;
  let sumS2 = 0;
  let msCurl = 0;
  let msStrain = 0;
  const inv2h = 1 / (2 * PROBE_FD_H);
  for (let i = 0; i < n; i++) {
    const [fx, fy] = at(0, i);
    sumX += fx;
    sumY += fy;
    sumSq += fx * fx + fy * fy;
    // Head outputs are tanh, so |F| ≤ 1 per component. "Saturated" means BOTH
    // components are pinned — the state that reads as an axis-clipped diagonal.
    if (Math.min(Math.abs(fx), Math.abs(fy)) > 0.9) sat++;
    const inv = 1 / Math.sqrt(fx * fx + fy * fy + PROBE_TAU * PROBE_TAU);
    const ux = fx * inv;
    const uy = fy * inv;
    sumUx += ux;
    sumUy += uy;
    // The double-angle vector is left UNNORMALIZED on purpose (it carries |u|²)
    // — same contract as directionOrderLoss: a near-zero force has no reliable
    // direction and should not vote at full weight.
    sumC2 += ux * ux - uy * uy;
    sumS2 += 2 * ux * uy;
    const [xpx, xpy] = at(1, i);
    const [xmx, xmy] = at(2, i);
    const [ypx, ypy] = at(3, i);
    const [ymx, ymy] = at(4, i);
    const dFxdx = (xpx - xmx) * inv2h;
    const dFydx = (xpy - xmy) * inv2h;
    const dFxdy = (ypx - ymx) * inv2h;
    const dFydy = (ypy - ymy) * inv2h;
    const curl = dFydx - dFxdy;
    const s1 = dFxdx - dFydy;
    const s2 = dFydx + dFxdy;
    msCurl += curl * curl;
    msStrain += s1 * s1 + s2 * s2;
  }
  const mx = sumX / n;
  const my = sumY / n;
  msCurl /= n;
  msStrain /= n;
  // AC is the TWO-PASS centred sum, not sqrt(mean‖F‖² − ‖mean F‖²). The two are
  // algebraically identical and numerically are not: on a nearly-pure-DC field
  // — exactly the collapse this instrument exists to catch — the one-pass form
  // subtracts two nearly equal sums and can land below zero. Cancellation there
  // would read as "even more collapsed", or NaN. 1024 points is not a place to
  // trade accuracy for a pass.
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const [fx, fy] = at(0, i);
    const dx = fx - mx;
    const dy = fy - my;
    acc += dx * dx + dy * dy;
  }
  return {
    dc: Math.hypot(mx, my),
    ac: Math.sqrt(acc / n),
    rmsF: Math.sqrt(sumSq / n),
    satFrac: sat / n,
    okuboWeiss: (msStrain - msCurl) / (msStrain + msCurl + 1e-12),
    r1: Math.hypot(sumUx / n, sumUy / n),
    r2: Math.hypot(sumC2 / n, sumS2 / n),
    gridN,
  };
}

export class FieldProbe {
  readonly gridN: number;
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly bind: GPUBindGroup;
  private readonly pointsBuf: GPUBuffer;
  private readonly outBuf: GPUBuffer;
  private readonly staging: GPUBuffer;
  private readonly uni: GPUBuffer;
  private readonly uniData = new ArrayBuffer(16);
  private readonly count: number;
  /** One in-flight readback at a time; a pending map skips the tick. */
  private pending = false;

  constructor(
    device: GPUDevice,
    layout: FieldLayout,
    weightsBuffer: GPUBuffer,
    gridN = 32
  ) {
    if (!Number.isInteger(gridN) || gridN < 2 || gridN > 256) {
      throw new Error(`field probe: gridN ${gridN} outside [2, 256]`);
    }
    this.device = device;
    this.gridN = gridN;
    const points = probePoints(gridN);
    this.count = points.length / 2;
    const module = device.createShaderModule({ code: fieldProbeShader(layout) });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "probe" },
    });
    this.pointsBuf = device.createBuffer({
      size: points.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.pointsBuf, 0, points);
    this.outBuf = device.createBuffer({
      size: this.count * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.staging = device.createBuffer({
      size: this.count * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.uni = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: weightsBuffer } },
        { binding: 2, resource: { buffer: this.pointsBuf } },
        { binding: 3, resource: { buffer: this.outBuf } },
      ],
    });
  }

  /**
   * One diagnostics dispatch + readback on its OWN encoder. Returns `null`
   * when a previous sample is still mapping — deliberately a skip, not a queue:
   * this is a 1 Hz instrument and a backlog would only report stale fields.
   */
  async sample(alpha: number): Promise<FieldMetrics | null> {
    if (this.pending) return null;
    this.pending = true;
    try {
      new Float32Array(this.uniData, 0, 1)[0] = alpha;
      new Uint32Array(this.uniData, 4, 1)[0] = this.count;
      this.device.queue.writeBuffer(this.uni, 0, this.uniData);
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bind);
      pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE));
      pass.end();
      enc.copyBufferToBuffer(this.outBuf, 0, this.staging, 0, this.count * 8);
      this.device.queue.submit([enc.finish()]);
      await this.staging.mapAsync(GPUMapMode.READ);
      const raw = new Float32Array(this.staging.getMappedRange().slice(0));
      this.staging.unmap();
      return fieldMetricsFrom(raw, this.gridN);
    } finally {
      this.pending = false;
    }
  }

  destroy(): void {
    for (const b of [this.pointsBuf, this.outBuf, this.staging, this.uni]) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }
}
