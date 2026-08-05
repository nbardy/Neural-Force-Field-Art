/**
 * AdversaryTrainer — the fused relaxed-WTA adversary as a PURE WebGPU object.
 * No tfjs. Host counterpart of ./adversary_wgsl.ts; the numeric oracle is
 * src/core/gan/adversary.ts (via the IR layer, tools/train_wta_test.ts).
 *
 * Owns the ADVERSARY weights buffer + Adam moments + tuple scratch + stats +
 * two packed N-capacity per-particle surprise planes (raw payoff and
 * payoff-per-unit-signal), and the extGrads buffer through which the generator
 * reward joins the field's pass B (FusedTrainer opts.extGradBuffer).
 *
 * SEPARATION IS STRUCTURAL: the field weights bind READ-ONLY in every
 * adversary dispatch, and the adversary weights are in a buffer the field
 * trainer never binds. tools/train_wta_test.ts asserts both directions
 * bit-identical.
 *
 * AGREE+DISAGREE. Two instances may share the same read-only field-weight
 * buffer while owning independent predictor/Adam state. Configure lanes 0/1
 * and roles disagree/agree; feed their two extGrads buffers to one
 * FusedTrainer so the field update sums both games exactly once.
 *
 * THE REWARD NORMALIZER LIVES HERE, ON THE HOST, as a tape constant by
 * design (mirrors adversary.ts RewardScale): `finalize` writes the batch
 * RMS‖y‖ into stats[2]; the async stats readback folds it into a λ=0.02 EMA;
 * `genSeed()` bakes max(rms, 1e-6) into the raw-vector game's scalar. Bounded
 * soft-spherical modes bypass the EMA. Scale-aware modes additionally report
 * raw member-signal energy and use an exact positive energy-anchor derivative.
 * A few frames of readback latency is irrelevant at a ~50-step EMA horizon.
 */

import type { FieldLayout, AdversaryLayout, LayerDims } from "./advect_wgsl";
import { layoutAdversary } from "./advect_wgsl";
import type { PassTimestampWrites } from "./gputime";
import type { HeadSpread } from "../../core/gan/adversary";
import {
  adversaryPassAShader,
  adversaryPassBShader,
  advScratchBytes,
  validateAdversaryFusion,
  adjustedTuple,
  fusedObjectiveDims,
  ADV_WG,
  ADV_WG_B,
  ADV_MAX_BATCH,
  ADV_STATS_BASE,
  type TupleTag,
  type FieldLane,
  type ObserverGeometry,
  type FusedAdversaryTarget,
  type FusedAdversaryLoss,
} from "./adversary_wgsl";

const ADAM_DEFAULTS = { beta1: 0.9, beta2: 0.999, eps: 1e-7 } as const;
/** mirror adversary.ts RMS_EMA_LAMBDA / RMS_FLOOR */
const RMS_EMA_LAMBDA = 0.02;
const RMS_FLOOR = 1e-6;

export type RewardScaleState = { tag: "unseeded" } | { tag: "seeded"; rms: number };
/** Named direction-game role. Scale-hold is general-sum: both roles cooperate
 * on scale matching and the positive energy anchor. */
export type GeneratorRole = "disagree" | "agree";
/** Visualization-only projection of the shared payoff. Neither choice changes
 * discriminator/generator losses, optimizer state, scratch, or scalar stats. */
export type SurpriseMetric = "raw-payoff" | "per-unit-signal";
export interface SurprisePlane {
  /** Both metrics share one packed allocation. */
  buffer: GPUBuffer;
  /** Plane origin in f32 elements (multiply by 4 for a byte binding offset). */
  offsetFloats: number;
}

export interface AdvStats {
  /** mean relaxed-weighted residual (the discriminator objective) */
  discLoss: number;
  /** Mean raw shared relaxed-WTA payoff. Retains the historical property name
   * for UI/test compatibility; it is not the per-unit visualization plane. */
  surprise: number;
  /** batch RMS‖y‖ */
  batchRms: number;
  /** Predictor geometry over every sampled context, including target-inactive
   * contexts. K=1 is explicitly not a zero-distance mixture. */
  headSpread: HeadSpread;
  /** [k] ACTIVE tuples each head won this batch. Explicitly invalid tuples
   * (currently near-tie triangles) contribute no win, so the sum may be < b. */
  winCounts: number[];
  /** RMS raw member-signal energy over active tuples. Appended after legacy
   * stats/wins so the existing HUD ABI remains stable. */
  energyRms: number;
  /** Exact active tuple count used by the batch energy-anchor derivative. */
  energyActive: number;
}

export interface AdvStepOpts {
  /** tuples this step (≤ batchCap) */
  b: number;
  alpha: number;
  /** discriminator Adam learning rate */
  lr: number;
  /** Retained for source compatibility. Live sampling now follows an exact
   * rotating coverage cursor, so this value is deliberately ignored. */
  seed?: number;
  source?: "particles" | "uploaded";
  /**
   * Non-negative generator coefficient magnitude. Raw-vector uses
   * weight/(max(rmsEma,1e-6)·B); bounded soft-angle and scale-aware modes use
   * weight/B.
   * use {@link AdversaryTrainer.genSeed}. The trainer applies the sign from its
   * named `generatorRole`; callers must not encode roles as magic signs.
   * 0 disables the generator path (fieldGrad then writes exact zeros).
   */
  genSeed: number;
  /** false = compute adversary grads only (verification); default true */
  applyDisc?: boolean;
}

export interface TrainPhysicsLike {
  width: number;
  height: number;
  forceMagnitude: number;
  friction: number;
  maxVelocity: number;
}

export interface SurpriseWindow {
  /** First particle index refreshed by the latest live-particle step. */
  start: number;
  /** Number of distinct particle indices refreshed by that step. */
  count: number;
  /** Monotone within one particle-buffer binding; reset to 0 on resize. */
  generation: number;
}

export interface SurpriseCoverage {
  /** Exact fraction of the current cloud refreshed since the last resize. */
  covered: number;
  coveredCount: number;
  total: number;
  /** First particle index that the next live step will refresh. */
  cursor: number;
  /** Latest fresh window. A zero count means no live sample exists yet. */
  window: SurpriseWindow;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Seeded glorot-normal packing for the K heads (Box-Muller over mulberry32,
 * distinct stream per head — symmetry breaking is load-bearing: identical
 * heads tie every argmin toward head 0 and the mixture is dead on arrival).
 */
export function packAdversaryInit(advL: AdversaryLayout, seed: number): Float32Array {
  const out = new Float32Array(advL.totalFloats);
  advL.heads.forEach((h, j) => {
    const rnd = mulberry32(seed * 7919 + j * 101 + 1);
    const gauss = () => {
      const u1 = Math.max(rnd(), 1e-12);
      const u2 = rnd();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
    h.layers.forEach((L) => {
      const std = Math.sqrt(2 / (L.inSize + L.outSize));
      for (let i = 0; i < L.inSize * L.outSize; i++) {
        out[L.weightOffset + i] = gauss() * std;
      }
      // biases zero (tfjs dense default)
    });
  });
  return out;
}

export class AdversaryTrainer {
  readonly field: FieldLayout;
  readonly advL: AdversaryLayout;
  readonly tag: TupleTag;
  readonly target: FusedAdversaryTarget;
  readonly loss: FusedAdversaryLoss;
  /** Explicit relational geometry, chosen from the live particle boundary. */
  readonly observerGeometry: ObserverGeometry;
  readonly fieldLane: FieldLane;
  readonly generatorRole: GeneratorRole;
  readonly k: number;
  readonly m: number;
  readonly batchCap: number;

  /** dL_gen/dW_field — hand to FusedTrainer as opts.extGradBuffer. */
  readonly extGradsBuf: GPUBuffer;
  /** Raw-payoff compatibility alias. New rendering code should use
   * {@link surprisePlane} so the packed-plane offset is explicit. */
  get surpriseBuf(): GPUBuffer {
    return this.surBuf;
  }

  private readonly device: GPUDevice;
  private readonly fieldWeightsBuf: GPUBuffer;
  private readonly advWBuf: GPUBuffer;
  private readonly adamM: GPUBuffer;
  private readonly adamV: GPUBuffer;
  private readonly advGradsBuf: GPUBuffer;
  private readonly scratchBuf: GPUBuffer;
  private readonly statsBuf: GPUBuffer;
  private readonly statsFloats: number;
  private readonly tupleBuf: GPUBuffer;
  private surBuf: GPUBuffer;
  private surFloats: number;
  private readonly uniA: GPUBuffer;
  private readonly uniB: GPUBuffer;
  private readonly uniAData = new ArrayBuffer(64);
  private readonly uniBData = new ArrayBuffer(32);
  private readonly pipeFwd: GPUComputePipeline;
  private readonly pipeFinalize: GPUComputePipeline;
  private readonly pipeOpt: GPUComputePipeline;
  private readonly pipeFieldGrad: GPUComputePipeline;
  private readonly bglA: GPUBindGroupLayout;
  private readonly bglB: GPUBindGroupLayout;
  private bindA: GPUBindGroup;
  private readonly bindB: GPUBindGroup;
  private partPos: GPUBuffer | null = null;
  private partVel: GPUBuffer | null = null;
  private partCount = 0;
  private readonly partDummy: GPUBuffer;
  private adamStep = 0;
  private coverageCursor = 0;
  private coveredCount = 0;
  private coverageGeneration = 0;
  private lastCoverageWindow: SurpriseWindow = { start: 0, count: 0, generation: 0 };

  private rms: RewardScaleState = { tag: "unseeded" };
  /** latest finalized stats (a few frames stale on the polling path) */
  lastStats: AdvStats | null = null;
  private statsStaging: GPUBuffer;
  private statsPending = false;

  constructor(
    device: GPUDevice,
    field: FieldLayout,
    opts: {
      tag: TupleTag;
      /** Defaults to the historical raw-force target. */
      target?: FusedAdversaryTarget;
      /** Defaults to raw-vector, except the legacy adjusted tag now selects
       * exact soft-angle(tau=.05), matching defaultAdversaryConfig. */
      loss?: FusedAdversaryLoss;
      /**
       * Required: wrap -> periodic; bounce/reset -> euclidean. Relational
       * encodings must never silently retain torus geometry after a border
       * mode switch.
       */
      observerGeometry: ObserverGeometry;
      /** `blend` is the historical path; 0/1 isolate one generator head. */
      fieldLane?: FieldLane;
      /** disagree minimizes -payoff; agree minimizes +payoff. */
      generatorRole?: GeneratorRole;
      k: number;
      relaxEps: number;
      hiddenUnits?: number;
      featureDim?: number;
      batchCap?: number;
      fieldWeightsBuffer: GPUBuffer;
      /** Sizes each packed surprise plane; grow via setParticleBuffers. */
      particleCount?: number;
      seed?: number;
    }
  ) {
    this.device = device;
    this.field = field;
    this.tag = opts.tag;
    this.target = opts.target ?? { tag: "force" };
    this.loss =
      opts.loss ??
      (adjustedTuple(opts.tag)
        ? { tag: "soft-angle", tau: 0.05 }
        : { tag: "raw-vector" });
    if (
      opts.observerGeometry !== "periodic" &&
      opts.observerGeometry !== "euclidean"
    ) {
      throw new Error(
        `adversary: observerGeometry must be explicitly periodic or euclidean, got ` +
          `${String(opts.observerGeometry)}`
      );
    }
    this.observerGeometry = opts.observerGeometry;
    this.fieldLane = opts.fieldLane ?? "blend";
    this.generatorRole = opts.generatorRole ?? "disagree";
    if (this.generatorRole !== "disagree" && this.generatorRole !== "agree") {
      throw new Error(`adversary: invalid generatorRole ${String(this.generatorRole)}`);
    }
    const objective = fusedObjectiveDims(this.tag, this.target, this.loss);
    const { m, du, outDy: dy } = objective;
    this.m = m;
    this.k = opts.k;
    this.batchCap = Math.min(opts.batchCap ?? 1024, ADV_MAX_BATCH);
    const hidden = opts.hiddenUnits ?? 32;
    const feature = opts.featureDim ?? 16;
    const dims: LayerDims[] = [
      { inSize: du, outSize: hidden, activation: "selu" },
      { inSize: hidden, outSize: feature, activation: "selu" },
      { inSize: feature, outSize: dy, activation: "linear" },
    ];
    this.advL = layoutAdversary(opts.k, dims, { du, dy });
    validateAdversaryFusion(field, this.advL);
    this.fieldWeightsBuf = opts.fieldWeightsBuffer;

    const mkStorage = (bytes: number) =>
      device.createBuffer({
        size: bytes,
        usage:
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

    this.advWBuf = mkStorage(this.advL.totalFloats * 4);
    this.adamM = mkStorage(this.advL.totalFloats * 4);
    this.adamV = mkStorage(this.advL.totalFloats * 4);
    this.advGradsBuf = mkStorage(this.advL.totalFloats * 4);
    this.scratchBuf = mkStorage(
      advScratchBytes(
        field,
        this.advL,
        this.tag,
        this.batchCap,
        this.target,
        this.loss
      )
    );
    const wgCap = Math.ceil(this.batchCap / ADV_WG);
    this.statsFloats = ADV_STATS_BASE + wgCap * (7 + this.k);
    this.statsBuf = mkStorage(this.statsFloats * 4);
    this.extGradsBuf = mkStorage(field.totalFloats * 4);
    this.tupleBuf = mkStorage(this.batchCap * m * 4 * 4);
    this.surFloats = Math.max(opts.particleCount ?? 0, this.batchCap * m, 1);
    this.surBuf = mkStorage(this.surFloats * 2 * 4);
    this.uniA = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uniB = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.partDummy = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    this.statsStaging = device.createBuffer({
      size: (7 + this.k) * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const COMPUTE = 4; // GPUShaderStage.COMPUTE (literal — not shimmed in bun)
    const ro = { type: "read-only-storage" as const };
    const rw = { type: "storage" as const };
    this.bglA = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: "uniform" as const } },
        { binding: 1, visibility: COMPUTE, buffer: ro }, // field weights
        { binding: 2, visibility: COMPUTE, buffer: ro }, // advW
        { binding: 3, visibility: COMPUTE, buffer: rw }, // scratch
        { binding: 4, visibility: COMPUTE, buffer: rw }, // stats
        { binding: 5, visibility: COMPUTE, buffer: ro }, // partPos
        { binding: 6, visibility: COMPUTE, buffer: ro }, // partVel
        { binding: 7, visibility: COMPUTE, buffer: rw }, // surprise
        { binding: 8, visibility: COMPUTE, buffer: ro }, // tuples
      ],
    });
    this.bglB = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: "uniform" as const } },
        { binding: 1, visibility: COMPUTE, buffer: rw }, // advW
        { binding: 2, visibility: COMPUTE, buffer: ro }, // scratch
        { binding: 3, visibility: COMPUTE, buffer: rw }, // advGrads
        { binding: 4, visibility: COMPUTE, buffer: rw }, // adamM
        { binding: 5, visibility: COMPUTE, buffer: rw }, // adamV
        { binding: 6, visibility: COMPUTE, buffer: rw }, // extGrads
      ],
    });
    const shaderOpts = {
      tag: this.tag,
      relaxEps: opts.relaxEps,
      fieldLane: this.fieldLane,
      observerGeometry: this.observerGeometry,
      target: this.target,
      loss: this.loss,
    };
    const plA = device.createPipelineLayout({ bindGroupLayouts: [this.bglA] });
    const moduleA = device.createShaderModule({
      code: adversaryPassAShader(field, this.advL, shaderOpts),
    });
    this.pipeFwd = device.createComputePipeline({
      layout: plA,
      compute: { module: moduleA, entryPoint: "advFwd" },
    });
    this.pipeFinalize = device.createComputePipeline({
      layout: plA,
      compute: { module: moduleA, entryPoint: "advFinalize" },
    });
    const plB = device.createPipelineLayout({ bindGroupLayouts: [this.bglB] });
    const moduleB = device.createShaderModule({
      code: adversaryPassBShader(field, this.advL, shaderOpts),
    });
    this.pipeOpt = device.createComputePipeline({
      layout: plB,
      compute: { module: moduleB, entryPoint: "advOpt" },
    });
    this.pipeFieldGrad = device.createComputePipeline({
      layout: plB,
      compute: { module: moduleB, entryPoint: "fieldGrad" },
    });

    this.bindA = this.makeBindA();
    this.bindB = device.createBindGroup({
      layout: this.bglB,
      entries: [
        { binding: 0, resource: { buffer: this.uniB } },
        { binding: 1, resource: { buffer: this.advWBuf } },
        { binding: 2, resource: { buffer: this.scratchBuf } },
        { binding: 3, resource: { buffer: this.advGradsBuf } },
        { binding: 4, resource: { buffer: this.adamM } },
        { binding: 5, resource: { buffer: this.adamV } },
        { binding: 6, resource: { buffer: this.extGradsBuf } },
      ],
    });

    this.uploadAdvWeights(packAdversaryInit(this.advL, opts.seed ?? 1234));
  }

  private makeBindA(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.bglA,
      entries: [
        { binding: 0, resource: { buffer: this.uniA } },
        { binding: 1, resource: { buffer: this.fieldWeightsBuf } },
        { binding: 2, resource: { buffer: this.advWBuf } },
        { binding: 3, resource: { buffer: this.scratchBuf } },
        { binding: 4, resource: { buffer: this.statsBuf } },
        { binding: 5, resource: { buffer: this.partPos ?? this.partDummy } },
        { binding: 6, resource: { buffer: this.partVel ?? this.partDummy } },
        { binding: 7, resource: { buffer: this.surBuf } },
        { binding: 8, resource: { buffer: this.tupleBuf } },
      ],
    });
  }

  /** Point at live particle state; re-call after AdvectKernel.setParticleCount.
   *  Any identity/count change starts a new exact coverage epoch and clears
   *  every retained value in both surprise planes, including stale prefixes
   *  after shrink.
   *  Grows the packed surprise buffer when the cloud grows (renderer re-binds by
   *  buffer identity, so consumers must re-read {@link surpriseBuf}). */
  setParticleBuffers(pos: GPUBuffer, vel: GPUBuffer, count: number): void {
    if (!Number.isInteger(count) || count < 0) {
      throw new Error(`adversary: particle count must be a non-negative integer, got ${count}`);
    }
    const changed =
      this.partPos !== pos || this.partVel !== vel || this.partCount !== count;
    if (!changed) return;
    this.partPos = pos;
    this.partVel = vel;
    this.partCount = count;
    if (count > this.surFloats) {
      try {
        this.surBuf.destroy();
      } catch (_) {}
      this.surFloats = count;
      this.surBuf = this.device.createBuffer({
        size: count * 2 * 4,
        usage:
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }
    this.coverageCursor = 0;
    this.coveredCount = 0;
    this.coverageGeneration = 0;
    this.lastCoverageWindow = { start: 0, count: 0, generation: 0 };
    // WebGPU buffers are zero-initialized, but the explicit clear is
    // load-bearing for shrink/rebind where the allocation is retained.
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(this.surBuf);
    this.device.queue.submit([encoder.finish()]);
    this.bindA = this.makeBindA();
  }

  /** Exact live-particle surprise coverage since the last buffer resize. */
  surpriseCoverage(): SurpriseCoverage {
    return {
      covered: this.partCount > 0 ? this.coveredCount / this.partCount : 0,
      coveredCount: this.coveredCount,
      total: this.partCount,
      cursor: this.coverageCursor,
      window: { ...this.lastCoverageWindow },
    };
  }

  uploadAdvWeights(w: Float32Array): void {
    if (w.length !== this.advL.totalFloats) {
      throw new Error(
        `adversary: uploadAdvWeights got ${w.length} floats, layout needs ${this.advL.totalFloats}`
      );
    }
    this.device.queue.writeBuffer(this.advWBuf, 0, w as unknown as BufferSource);
  }

  /** Uploaded tuples: per tuple, m members × [posX, posY, velX, velY] (pixels). */
  uploadTuples(t: Float32Array): void {
    const per = this.m * 4;
    if (t.length % per !== 0 || t.length / per > this.batchCap) {
      throw new Error(
        `adversary: uploadTuples got ${t.length} floats (per-tuple ${per}, cap ${this.batchCap})`
      );
    }
    this.device.queue.writeBuffer(this.tupleBuf, 0, t as unknown as BufferSource);
  }

  /** Reward-normalizer telemetry (named state, mirrors adversary.ts). */
  rewardScaleState(): RewardScaleState {
    return this.rms;
  }

  /**
   * The uniform coefficient on the generator objective.
   *
   * `stepScale` is retained in the ABI while callers migrate, but deliberately
   * ignored: the target is now raw F(x), not a velocity/displacement, so screen
   * resolution and maxVelocity are not units of the force-target game. Raw
   * modes retain the host EMA scale. Soft-spherical modes are bounded and need
   * no EMA. Scale-hold reverses only direction while cooperating on local
   * relative scale and a positive batch-energy anchor.
   */
  genSeed(weight: number, _stepScale: number, b: number): number {
    if (!(weight >= 0) || !(b > 0)) {
      throw new Error(`adversary: genSeed needs weight >= 0 and b > 0, got ${weight}, ${b}`);
    }
    if (weight === 0) return 0;
    if (this.loss.tag !== "raw-vector") return weight / b;
    const scale = this.rms.tag === "seeded" ? Math.max(this.rms.rms, RMS_FLOOR) : 1;
    return weight / (scale * b);
  }

  /** One fused alternating-game step, recorded into a caller-owned encoder:
   *
   *  advFwd(pre-D) → advFinalize → advOpt → advFwd(post-D) → fieldGrad.
   *
   * The second forward is load-bearing: it recomputes the generator cotangent
   * with the discriminator weights AFTER their update, matching the tfjs
   * `trainStep(); generatorLoss()` ordering. Its partial-stat writes are not
   * finalized, so the reported discriminator loss remains the pre-update loss.
   */
  encodeStep(
    encoder: GPUCommandEncoder,
    phys: TrainPhysicsLike,
    o: AdvStepOpts,
    ts?: PassTimestampWrites
  ): void {
    if (!Number.isInteger(o.b) || o.b <= 0) {
      throw new Error(`adversary: b must be a positive integer, got ${o.b}`);
    }
    if (o.b > this.batchCap) {
      throw new Error(`adversary: b=${o.b} > batchCap ${this.batchCap}`);
    }
    const source = o.source ?? "particles";
    if (source !== "particles" && source !== "uploaded") {
      throw new Error(`adversary: unsupported source ${String(source)}`);
    }
    if (source === "particles" && !this.partPos) {
      throw new Error("adversary: source 'particles' needs setParticleBuffers first");
    }
    if (!(o.genSeed >= 0)) {
      throw new Error(
        `adversary: AdvStepOpts.genSeed is a non-negative magnitude; role ` +
          `'${this.generatorRole}' supplies the sign, got ${o.genSeed}`
      );
    }
    const effectiveB =
      source === "particles"
        ? Math.min(o.b, Math.floor(this.partCount / this.m))
        : o.b;
    if (effectiveB === 0) {
      // No complete tuple exists. Clear every externally observable product so
      // neither the field optimizer nor telemetry can consume stale data.
      // Predictor weights/Adam/RMS deliberately do not advance.
      encoder.clearBuffer(this.extGradsBuf);
      encoder.clearBuffer(this.advGradsBuf);
      encoder.clearBuffer(this.statsBuf);
      this.lastStats = {
        discLoss: 0,
        surprise: 0,
        batchRms: 0,
        headSpread:
          this.k === 1
            ? { tag: "single-head" }
            : {
                tag: "spread",
                meanPair: 0,
                minPair: 0,
                pairs: (this.k * (this.k - 1)) / 2,
              },
        winCounts: new Array(this.k).fill(0),
        energyRms: 0,
        energyActive: 0,
      };
      this.lastCoverageWindow = {
        start: this.coverageCursor,
        count: 0,
        generation: this.coverageGeneration,
      };
      return;
    }
    const applyDisc = o.applyDisc ?? true;
    if (applyDisc) this.adamStep++;

    const fA = new Float32Array(this.uniAData);
    const uA = new Uint32Array(this.uniAData);
    fA[0] = phys.width;
    fA[1] = phys.height;
    fA[2] = phys.forceMagnitude;
    fA[3] = phys.friction;
    fA[4] = phys.maxVelocity;
    fA[5] = o.alpha;
    // genSeed() is normalized by requested B. When a small live cloud cannot
    // supply B unique tuples, restore mean-over-effective-batch semantics.
    fA[6] =
      o.genSeed *
      (o.b / effectiveB) *
      (this.generatorRole === "disagree" ? 1 : -1);
    fA[7] = 1 / effectiveB;
    uA[8] = effectiveB;
    uA[9] = this.partCount;
    uA[10] = source === "particles" ? this.coverageCursor : 0;
    uA[11] = source === "uploaded" ? 0 : 2;
    const wg = Math.ceil(effectiveB / ADV_WG);
    uA[12] = wg;
    uA[13] = this.surFloats;
    this.device.queue.writeBuffer(this.uniA, 0, this.uniAData);

    const fB = new Float32Array(this.uniBData);
    const uB = new Uint32Array(this.uniBData);
    fB[0] = o.lr;
    fB[1] = ADAM_DEFAULTS.beta1;
    fB[2] = ADAM_DEFAULTS.beta2;
    fB[3] = ADAM_DEFAULTS.eps;
    uB[4] = Math.max(1, this.adamStep);
    uB[5] = applyDisc ? 1 : 0;
    uB[6] = effectiveB;
    this.device.queue.writeBuffer(this.uniB, 0, this.uniBData);

    const tsBegin = ts
      ? { querySet: ts.querySet, beginningOfPassWriteIndex: ts.beginningOfPassWriteIndex }
      : undefined;
    const tsEnd = ts
      ? { querySet: ts.querySet, endOfPassWriteIndex: ts.endOfPassWriteIndex }
      : undefined;
    const p1 = encoder.beginComputePass(
      (tsBegin ? { timestampWrites: tsBegin } : undefined) as GPUComputePassDescriptor
    );
    p1.setPipeline(this.pipeFwd);
    p1.setBindGroup(0, this.bindA);
    p1.dispatchWorkgroups(wg);
    p1.end();
    const p2 = encoder.beginComputePass();
    p2.setPipeline(this.pipeFinalize);
    p2.setBindGroup(0, this.bindA);
    p2.dispatchWorkgroups(1);
    p2.end();
    const p3 = encoder.beginComputePass();
    p3.setPipeline(this.pipeOpt);
    p3.setBindGroup(0, this.bindB);
    p3.dispatchWorkgroups(Math.ceil(this.advL.totalFloats / ADV_WG_B));
    p3.end();
    const pGen = encoder.beginComputePass();
    pGen.setPipeline(this.pipeFwd);
    pGen.setBindGroup(0, this.bindA);
    pGen.dispatchWorkgroups(wg);
    pGen.end();
    const p4 = encoder.beginComputePass(
      (tsEnd ? { timestampWrites: tsEnd } : undefined) as GPUComputePassDescriptor
    );
    p4.setPipeline(this.pipeFieldGrad);
    p4.setBindGroup(0, this.bindB);
    p4.dispatchWorkgroups(Math.ceil(this.field.totalFloats / ADV_WG_B));
    p4.end();

    // Both forwards use the same offset; advance exactly once per logical
    // alternating-game step, after all commands have been recorded.
    if (source === "particles") {
      const touched = effectiveB * this.m;
      const start = this.coverageCursor;
      this.coverageGeneration++;
      this.lastCoverageWindow = {
        start,
        count: touched,
        generation: this.coverageGeneration,
      };
      this.coverageCursor = (start + touched) % this.partCount;
      this.coveredCount = Math.min(this.partCount, this.coveredCount + touched);
    }
  }

  /** Self-submitting step (tests / offline). */
  step(phys: TrainPhysicsLike, o: AdvStepOpts): void {
    const enc = this.device.createCommandEncoder();
    this.encodeStep(enc, phys, o);
    this.device.queue.submit([enc.finish()]);
  }

  private parseStats(f: Float32Array): AdvStats {
    return {
      discLoss: f[0],
      surprise: f[1],
      batchRms: f[2],
      headSpread:
        this.k === 1
          ? { tag: "single-head" }
          : {
              tag: "spread",
              meanPair: f[3],
              minPair: f[4],
              pairs: (this.k * (this.k - 1)) / 2,
            },
      winCounts: Array.from(f.subarray(5, 5 + this.k)),
      energyRms: f[5 + this.k],
      energyActive: f[6 + this.k],
    };
  }

  private foldRms(batchRms: number): void {
    if (!Number.isFinite(batchRms) || batchRms <= 0) return;
    this.rms =
      this.rms.tag === "unseeded"
        ? { tag: "seeded", rms: batchRms }
        : {
            tag: "seeded",
            rms: this.rms.rms + RMS_EMA_LAMBDA * (batchRms - this.rms.rms),
          };
  }

  /**
   * Non-blocking stats path for the hot loop: record the tiny copy into this
   * frame's encoder, then {@link afterSubmit} maps it. Skipped while a map is
   * pending (a slow GPU degrades to fewer updates, never to a stall).
   */
  encodeStatsRead(encoder: GPUCommandEncoder): boolean {
    if (this.statsPending) return false;
    encoder.copyBufferToBuffer(this.statsBuf, 0, this.statsStaging, 0, (7 + this.k) * 4);
    return true;
  }

  afterSubmit(recorded: boolean): void {
    if (!recorded || this.statsPending) return;
    this.statsPending = true;
    this.statsStaging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const f = new Float32Array(this.statsStaging.getMappedRange().slice(0));
        this.statsStaging.unmap();
        this.statsPending = false;
        this.lastStats = this.parseStats(f);
        this.foldRms(this.lastStats.batchRms);
      })
      .catch(() => {
        this.statsPending = false;
      });
  }

  // ---- blocking readbacks (tests; MAP_READ staging, not the hot path) -----
  private async read(
    buf: GPUBuffer,
    floats: number,
    offsetFloats: number = 0
  ): Promise<Float32Array> {
    const staging = this.device.createBuffer({
      size: floats * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offsetFloats * 4, staging, 0, floats * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }
  async readStats(): Promise<AdvStats> {
    const f = await this.read(this.statsBuf, 7 + this.k);
    const s = this.parseStats(f);
    this.lastStats = s;
    this.foldRms(s.batchRms);
    return s;
  }
  readAdvGrads(): Promise<Float32Array> {
    return this.read(this.advGradsBuf, this.advL.totalFloats);
  }
  readExtGrads(): Promise<Float32Array> {
    return this.read(this.extGradsBuf, this.field.totalFloats);
  }
  readAdvWeights(): Promise<Float32Array> {
    return this.read(this.advWBuf, this.advL.totalFloats);
  }
  readScratch(floats: number): Promise<Float32Array> {
    return this.read(this.scratchBuf, floats);
  }
  /** Packed surprise-plane descriptor for zero-copy render binding. */
  surprisePlane(metric: SurpriseMetric): SurprisePlane {
    switch (metric) {
      case "raw-payoff":
        return { buffer: this.surBuf, offsetFloats: 0 };
      case "per-unit-signal":
        return { buffer: this.surBuf, offsetFloats: this.surFloats };
      default:
        throw new Error(`adversary: unsupported surprise metric ${String(metric)}`);
    }
  }

  /** Blocking diagnostic readback. Defaults to the historical raw payoff. */
  readSurprise(
    n: number,
    metric: SurpriseMetric = "raw-payoff"
  ): Promise<Float32Array> {
    const plane = this.surprisePlane(metric);
    return this.read(
      plane.buffer,
      Math.min(n, this.surFloats),
      plane.offsetFloats
    );
  }

  resetAdam(): void {
    this.adamStep = 0;
    const zeros = new Float32Array(this.advL.totalFloats);
    this.device.queue.writeBuffer(this.adamM, 0, zeros as unknown as BufferSource);
    this.device.queue.writeBuffer(this.adamV, 0, zeros as unknown as BufferSource);
  }

  destroy(): void {
    const own: GPUBuffer[] = [
      this.advWBuf, this.adamM, this.adamV, this.advGradsBuf, this.scratchBuf,
      this.statsBuf, this.extGradsBuf, this.tupleBuf, this.surBuf,
      this.uniA, this.uniB, this.partDummy, this.statsStaging,
    ];
    for (const b of own) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }
}
