/**
 * FusedTrainer — the fused TRAIN step as a PURE WebGPU object. No tfjs.
 *
 * Owns (or shares) the packed weights buffer plus Adam state, batch, scratch
 * and gradient buffers, and runs the two generated dispatches from
 * ./train_wgsl.ts per training step:
 *   pass A: rollout forward + batch reductions + analytic backward → scratch
 *   pass B: dW reduction + in-place Adam on the packed weights
 *
 * When constructed with the AdvectKernel's weights buffer, training updates
 * are immediately visible to the advect pass with no copies — weights are
 * born on the GPU and never leave. tfjs's only remaining roles are (a) the
 * blueprint the FieldLayout/initial weights come from and (b) the autograd
 * oracle in tools/train_test.ts.
 */

import type { FieldLayout } from "./advect_wgsl";
import type { PassTimestampWrites } from "./gputime";
import {
  trainPassAShader,
  trainPassBShader,
  scratchBytes,
  TRAIN_WG,
  TRAIN_WG_B,
  MAX_BATCH,
} from "./train_wgsl";

export interface TrainPhysics {
  width: number;
  height: number;
  forceMagnitude: number;
  friction: number;
  maxVelocity: number;
}

export interface TrainStepOpts {
  /** batch size (≤ MAX_BATCH) */
  n: number;
  alpha: number;
  /** Adam learning rate (used when apply) */
  lr: number;
  /** RNG stream for in-kernel batch generation/sampling (frame counter is perfect) */
  seed?: number;
  /**
   * Where training states come from:
   *   "random"    — fresh uniform points, vel 0 (default; original behavior)
   *   "particles" — live particle states from setParticleBuffers (real pos AND
   *                 vel; coverage comes from the reset slider)
   *   "uploaded"  — the uploadBatch data, vel 0 (verification fixtures)
   */
  source?: "random" | "particles" | "uploaded";
  /**
   * With source:"particles": fraction of the batch drawn from fresh uniform
   * random points instead (a coverage floor independent of the reset rate).
   * 0 (default) = pure particle states, 1 = all random.
   */
  mixRandom?: number;
  /** false = compute gradients only (verification); true = Adam-update weights */
  apply?: boolean;
}

const ADAM_DEFAULTS = { beta1: 0.9, beta2: 0.999, eps: 1e-7 } as const;

export class FusedTrainer {
  readonly layout: FieldLayout;
  readonly weightsBuf: GPUBuffer;
  /** true when weightsBuf is owned by someone else (e.g. AdvectKernel) */
  private readonly weightsShared: boolean;

  private readonly device: GPUDevice;
  // Pass A is now THREE dispatches sharing one explicit bind-group layout:
  // fwd (multi-workgroup forward + per-workgroup partial), finalize (combine
  // partials → dC), bwd (multi-workgroup backward). This vectorizes the batch
  // across the GPU instead of a single workgroup on one core.
  private readonly pipeFwd: GPUComputePipeline;
  private readonly pipeFinalize: GPUComputePipeline;
  private readonly pipeBwd: GPUComputePipeline;
  private readonly bglA: GPUBindGroupLayout;
  private readonly partialsBuf: GPUBuffer;
  private readonly pipeB: GPUComputePipeline;
  private readonly batchBuf: GPUBuffer;
  private readonly scratchBuf: GPUBuffer;
  private readonly lossBuf: GPUBuffer;
  private readonly gradsBuf: GPUBuffer;
  private readonly adamM: GPUBuffer;
  private readonly adamV: GPUBuffer;
  private readonly uniA: GPUBuffer;
  private readonly uniB: GPUBuffer;
  private readonly uniAData = new ArrayBuffer(64);
  private readonly uniBData = new ArrayBuffer(32);
  private bindA: GPUBindGroup;
  private readonly bindB: GPUBindGroup;
  private readonly batchCap: number;
  private partPos: GPUBuffer | null = null;
  private partVel: GPUBuffer | null = null;
  private partCount = 0;
  /** stands in for bindings 5/6 until setParticleBuffers — must NOT alias
   *  batchBuf (read_write at binding 2 + read at 5/6 in the same pass is a
   *  WebGPU usage-validation error) */
  private readonly partDummy: GPUBuffer;

  /** Adam step counter (t ≥ 1 on the first applied update, tfjs convention). */
  private adamStep = 0;

  /** rollout length K — compiled into the shaders at construction */
  readonly kSteps: number;

  constructor(
    device: GPUDevice,
    layout: FieldLayout,
    opts: { batchCap?: number; weightsBuffer?: GPUBuffer; kSteps?: number } = {}
  ) {
    this.device = device;
    this.layout = layout;
    this.batchCap = Math.min(opts.batchCap ?? MAX_BATCH, MAX_BATCH);
    this.kSteps = opts.kSteps ?? 1;

    // Pass A: one module, three entry points (fwd/finalize/bwd), one EXPLICIT
    // bind-group layout so all three share a single bind group (their `auto`
    // layouts would differ — each uses a different subset of bindings).
    const COMPUTE = 4; // GPUShaderStage.COMPUTE (literal — not shimmed in bun)
    const ro = { type: "read-only-storage" as const };
    const rw = { type: "storage" as const };
    this.bglA = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: "uniform" as const } },
        { binding: 1, visibility: COMPUTE, buffer: ro }, // weights
        { binding: 2, visibility: COMPUTE, buffer: rw }, // batch
        { binding: 3, visibility: COMPUTE, buffer: rw }, // scratch
        { binding: 4, visibility: COMPUTE, buffer: rw }, // lossOut/dC
        { binding: 5, visibility: COMPUTE, buffer: ro }, // partPos
        { binding: 6, visibility: COMPUTE, buffer: ro }, // partVel
        { binding: 7, visibility: COMPUTE, buffer: rw }, // partials
      ],
    });
    const plA = device.createPipelineLayout({ bindGroupLayouts: [this.bglA] });
    const moduleA = device.createShaderModule({
      code: trainPassAShader(layout, { kSteps: this.kSteps }),
    });
    const mkA = (entryPoint: string) =>
      device.createComputePipeline({ layout: plA, compute: { module: moduleA, entryPoint } });
    this.pipeFwd = mkA("fwd");
    this.pipeFinalize = mkA("finalize");
    this.pipeBwd = mkA("bwd");
    this.pipeB = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: trainPassBShader(layout, { kSteps: this.kSteps }),
        }),
        entryPoint: "main",
      },
    });

    const mkStorage = (bytes: number) =>
      device.createBuffer({
        size: bytes,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });

    this.weightsShared = !!opts.weightsBuffer;
    this.weightsBuf = opts.weightsBuffer ?? mkStorage(layout.totalFloats * 4);
    this.batchBuf = mkStorage(this.batchCap * 8);
    this.scratchBuf = mkStorage(scratchBytes(layout, this.batchCap, this.kSteps));
    this.lossBuf = mkStorage(8 * 4);
    this.gradsBuf = mkStorage(layout.totalFloats * 4);
    this.adamM = mkStorage(layout.totalFloats * 4); // zero-init by spec
    this.adamV = mkStorage(layout.totalFloats * 4);
    // one vec4 partial per fwd workgroup; ceil(batchCap / TRAIN_WG) of them.
    this.partialsBuf = mkStorage(Math.ceil(this.batchCap / TRAIN_WG) * 16);
    this.uniA = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uniB = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.partDummy = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.STORAGE,
    });

    this.bindA = this.makeBindA();
    this.bindB = device.createBindGroup({
      layout: this.pipeB.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniB } },
        { binding: 1, resource: { buffer: this.weightsBuf } },
        { binding: 2, resource: { buffer: this.scratchBuf } },
        { binding: 3, resource: { buffer: this.gradsBuf } },
        { binding: 4, resource: { buffer: this.adamM } },
        { binding: 5, resource: { buffer: this.adamV } },
      ],
    });
  }

  private makeBindA(): GPUBindGroup {
    // ONE bind group shared by fwd/finalize/bwd via the explicit bglA layout.
    // bindings 5/6 must always be bound; a dedicated dummy stands in until real
    // particle buffers are provided (must NOT alias batchBuf — usage conflict).
    return this.device.createBindGroup({
      layout: this.bglA,
      entries: [
        { binding: 0, resource: { buffer: this.uniA } },
        { binding: 1, resource: { buffer: this.weightsBuf } },
        { binding: 2, resource: { buffer: this.batchBuf } },
        { binding: 3, resource: { buffer: this.scratchBuf } },
        { binding: 4, resource: { buffer: this.lossBuf } },
        { binding: 5, resource: { buffer: this.partPos ?? this.partDummy } },
        { binding: 6, resource: { buffer: this.partVel ?? this.partDummy } },
        { binding: 7, resource: { buffer: this.partialsBuf } },
      ],
    });
  }

  /**
   * Point the trainer at live particle state (the AdvectKernel's buffers) for
   * source:"particles" steps. Re-call after AdvectKernel.setParticleCount —
   * resize replaces the buffers, which would leave this bind group stale.
   */
  setParticleBuffers(pos: GPUBuffer, vel: GPUBuffer, count: number): void {
    this.partPos = pos;
    this.partVel = vel;
    this.partCount = count;
    this.bindA = this.makeBindA();
  }

  uploadWeights(w: Float32Array): void {
    if (w.length !== this.layout.totalFloats) {
      throw new Error(
        `train: uploadWeights got ${w.length} floats, layout needs ${this.layout.totalFloats}`
      );
    }
    this.device.queue.writeBuffer(this.weightsBuf, 0, w as unknown as BufferSource);
  }

  /** Interleaved xy positions, pixel coords, for genRandom=false steps. */
  uploadBatch(b: Float32Array): void {
    if (b.length / 2 > this.batchCap) {
      throw new Error(`train: batch ${b.length / 2} > cap ${this.batchCap}`);
    }
    this.device.queue.writeBuffer(this.batchBuf, 0, b as unknown as BufferSource);
  }

  /** One fused training step: 4 dispatches (fwd/finalize/bwd/adam), 1 submit.
   *  (Self-submitting — tests
   *  and the tfjs-legacy path use this; the fused hot path uses encodeStep.) */
  step(phys: TrainPhysics, o: TrainStepOpts): void {
    const enc = this.device.createCommandEncoder();
    this.record(enc, phys, o);
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Same two passes as {@link step}, recorded into a CALLER-owned encoder and
   * NOT submitted — so a whole frame (train + advect + render) collapses to one
   * queue.submit. Uniform writeBuffers stay on the queue (ordered before the
   * caller's submit). `tsA`/`tsB` optionally timestamp passes A and B.
   */
  encodeStep(
    encoder: GPUCommandEncoder,
    phys: TrainPhysics,
    o: TrainStepOpts,
    tsA?: PassTimestampWrites,
    tsB?: PassTimestampWrites
  ): void {
    this.record(encoder, phys, o, tsA, tsB);
  }

  private record(
    encoder: GPUCommandEncoder,
    phys: TrainPhysics,
    o: TrainStepOpts,
    tsA?: PassTimestampWrites,
    tsB?: PassTimestampWrites
  ): void {
    if (o.n > this.batchCap) {
      throw new Error(`train: n=${o.n} > batchCap ${this.batchCap}`);
    }
    const apply = o.apply ?? true;
    if (apply) this.adamStep++;

    const fA = new Float32Array(this.uniAData);
    const uA = new Uint32Array(this.uniAData);
    fA[0] = phys.width;
    fA[1] = phys.height;
    fA[2] = phys.forceMagnitude;
    fA[3] = phys.friction;
    fA[4] = phys.maxVelocity;
    fA[5] = o.alpha;
    fA[6] = 0.01; // HH — loss constant, fixed in codegen's LOSS too
    const source = o.source ?? "random";
    if (source === "particles" && !this.partPos) {
      throw new Error("train: source 'particles' needs setParticleBuffers first");
    }
    uA[7] = this.partCount;
    uA[8] = o.n;
    uA[9] = (o.seed ?? 0) >>> 0;
    uA[10] = source === "uploaded" ? 0 : source === "random" ? 1 : 2;
    uA[11] = this.kSteps; // informational — K is compiled into the WGSL
    uA[12] = Math.round(Math.max(0, Math.min(1, o.mixRandom ?? 0)) * o.n);
    const wgA = Math.ceil(o.n / TRAIN_WG); // fwd/bwd workgroups this step
    uA[13] = wgA;
    this.device.queue.writeBuffer(this.uniA, 0, this.uniAData);

    const fB = new Float32Array(this.uniBData);
    const uB = new Uint32Array(this.uniBData);
    fB[0] = o.lr;
    fB[1] = ADAM_DEFAULTS.beta1;
    fB[2] = ADAM_DEFAULTS.beta2;
    fB[3] = ADAM_DEFAULTS.eps;
    uB[4] = Math.max(1, this.adamStep);
    uB[5] = apply ? 1 : 0;
    uB[6] = o.n;
    this.device.queue.writeBuffer(this.uniB, 0, this.uniBData);

    // PASS A = three dispatches sharing this.bindA: fwd (wgA workgroups —
    // full GPU) → finalize (1 workgroup, combines partials) → bwd (wgA). The
    // dispatch boundaries insert the barriers fwd→finalize→bwd need (bwd reads
    // the batch-wide dC that finalize computes from all fwd partials).
    // @webgpu/types 0.1.30 predates object-form timestampWrites; cast (gputime).
    // Span the whole pass A: begin on fwd, end on bwd (like splat's decay→tone).
    const fwdTs = tsA
      ? { querySet: tsA.querySet, beginningOfPassWriteIndex: tsA.beginningOfPassWriteIndex }
      : undefined;
    const bwdTs = tsA
      ? { querySet: tsA.querySet, endOfPassWriteIndex: tsA.endOfPassWriteIndex }
      : undefined;
    const pFwd = encoder.beginComputePass(
      (fwdTs ? { timestampWrites: fwdTs } : undefined) as GPUComputePassDescriptor
    );
    pFwd.setPipeline(this.pipeFwd);
    pFwd.setBindGroup(0, this.bindA);
    pFwd.dispatchWorkgroups(wgA);
    pFwd.end();
    const pFin = encoder.beginComputePass();
    pFin.setPipeline(this.pipeFinalize);
    pFin.setBindGroup(0, this.bindA);
    pFin.dispatchWorkgroups(1);
    pFin.end();
    const pBwd = encoder.beginComputePass(
      (bwdTs ? { timestampWrites: bwdTs } : undefined) as GPUComputePassDescriptor
    );
    pBwd.setPipeline(this.pipeBwd);
    pBwd.setBindGroup(0, this.bindA);
    pBwd.dispatchWorkgroups(wgA);
    pBwd.end();
    const pb = encoder.beginComputePass(
      (tsB ? { timestampWrites: tsB } : undefined) as GPUComputePassDescriptor
    );
    pb.setPipeline(this.pipeB);
    pb.setBindGroup(0, this.bindB);
    pb.dispatchWorkgroups(Math.ceil(this.layout.totalFloats / TRAIN_WG_B));
    pb.end();
  }

  resetAdam(): void {
    this.adamStep = 0;
    const zeros = new Float32Array(this.layout.totalFloats);
    this.device.queue.writeBuffer(this.adamM, 0, zeros as unknown as BufferSource);
    this.device.queue.writeBuffer(this.adamV, 0, zeros as unknown as BufferSource);
  }

  // ---- test/debug readbacks (MAP_READ staging; not used on the hot path) ----
  private async read(buf: GPUBuffer, floats: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({
      size: floats * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, floats * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }
  readGrads(): Promise<Float32Array> {
    return this.read(this.gradsBuf, this.layout.totalFloats);
  }
  readWeights(): Promise<Float32Array> {
    return this.read(this.weightsBuf, this.layout.totalFloats);
  }
  async readLoss(): Promise<{ loss: number; C00: number; C11: number; C01: number }> {
    const l = await this.read(this.lossBuf, 8);
    return { loss: l[0], C00: l[1], C11: l[2], C01: l[3] };
  }

  destroy(): void {
    const own: GPUBuffer[] = [
      this.batchBuf, this.scratchBuf, this.lossBuf, this.gradsBuf,
      this.adamM, this.adamV, this.uniA, this.uniB,
    ];
    if (!this.weightsShared) own.push(this.weightsBuf);
    for (const b of own) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }
}
