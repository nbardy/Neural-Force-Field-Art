/**
 * Neural force field — two direct-vector heads (FIRST-ORDER).
 * ===================================================================
 *
 * `HelmholtzField` is retained as a compatibility class name. The production
 * architecture is simply two small MLPs that output 2D vectors DIRECTLY:
 *
 *     g(posNorm) : R^2 -> R^2   vector head A
 *     r(posNorm) : R^2 -> R^2   vector head B
 *
 *     forces = (1 - alpha) * g(posNorm)  +  alpha * r(posNorm)
 *
 * The live `alpha ∈ [0,1]` knob is a neutral output interpolation. It is NOT an
 * intrinsic order↔chaos coordinate: that distinction can only come from an
 * explicitly routed loss/game (for example Agree+Disagree).
 *
 * ─────────────────────────────────────────────────────────────────────
 * WHY DIRECT VECTORS INSTEAD OF grad(φ) / curl(ψ)  (the whole point)
 * ─────────────────────────────────────────────────────────────────────
 * The previous design built the force as F = ∇φ + curl(ψ), computing each
 * lane with `tf.grad` of a scalar net w.r.t. the INPUT position. That gave
 * an EXACT divergence-free chaos lane by construction — but at a fatal cost:
 * because `forces()` itself called `tf.grad`, training (optimizer.minimize
 * differentiates the loss w.r.t. the WEIGHTS) had to differentiate THROUGH
 * that inner gradient. That is SECOND-ORDER autograd, measured at
 * ~800–1700 ms/frame — roughly 10x too slow, and it also forced tanh-only
 * hidden layers (SELU/ELU have no registered 2nd-order gradient in tfjs).
 *
 * TRADEOFF (deliberate): we drop the *exact* divergence-free guarantee.
 * Pieces that request divergence/chaos do so through an explicit field-loss
 * specification; the architecture itself assigns no such semantic. In return,
 * `forces()` is a plain FORWARD pass — no `tf.grad` inside — so training is a
 * single FIRST-order backward, identical in cost to the fast MLP pieces
 * (~10x faster). Approximate-and-cheap beats exact-and-unusable here.
 *
 * ─────────────────────────────────────────────────────────────────────
 * DIFFERENTIABILITY
 * ─────────────────────────────────────────────────────────────────────
 * `forces()` only runs `net.predict` (a forward pass) wrapped in `tf.tidy`.
 * Nothing detaches the tape: when the integrator runs the physics step
 * inside `optimizer.minimize`, a single first-order gradient flows from the
 * loss, through the two heads' outputs, into `g`'s and `r`'s weights.
 *
 * ─────────────────────────────────────────────────────────────────────
 * INTEGRATOR NOTE
 * ─────────────────────────────────────────────────────────────────────
 * Each head ends in a `tanh` layer, so the raw force is bounded ~O(1) (and
 * can be negative). Like the sigmoid MLPs in main.ts this is a RAW signed
 * vector, but do NOT apply the `(raw - 0.5)` shift — that only re-centers a
 * [0,1] sigmoid. Feed the vector straight in and let `forceMagnitude` scale:
 *
 *     const forces = field.forces(posNorm).mul(cfg.forceMagnitude);
 *
 * `trainableWeights` exposes the underlying `tf.Variable`s so the integrator
 * may pass them as the `varList` to `optimizer.minimize`.
 */

import * as tf from "@tensorflow/tfjs";

/**
 * A differentiable 2D force field over normalized positions.
 *
 * `forces(posNorm)` maps `[N,2]` normalized positions (each coord in
 * roughly `[0,1]`) to `[N,2]` raw force vectors. The result stays
 * differentiable w.r.t. `trainableWeights` so it can be optimized in place.
 */
export interface ForceField {
  /** Map normalized positions `[N,2]` -> raw force vectors `[N,2]`. */
  forces(posNorm: tf.Tensor2D): tf.Tensor2D;
  /** The learnable variables of the field (for optimizer varLists). */
  readonly trainableWeights: tf.Variable[];
  /** Release all GPU/CPU tensors owned by the field. */
  dispose(): void;
}

/** Config for {@link HelmholtzField}. */
export interface HelmholtzFieldConfig {
  /**
   * Neutral head mix in `[0,1]`: `0` = pure A/`g`, `1` = pure B/`r`.
   * Mutable at runtime; the loss/game, not this number, supplies semantics.
   */
  alpha: number;
  /**
   * Semantic use of the two vector heads. "blend" preserves the historical
   * interpolated field. "agree-disagree" exposes A, B and their derived blend
   * as separate particle roles; it is not a Helmholtz decomposition.
   */
  semantic?: "blend" | "agree-disagree";
  /** Hidden layer widths for BOTH vector heads. Default `[32, 32]`. */
  hiddenUnits?: number[];
  /**
   * Multi-species class count C (default 0 = classless). When C > 0 the
   * second head `r` takes `[pos, onehot(class)]` (2+C inputs) while the first
   * head `g` stays class-blind. Class-aware fields are FUSED-KERNEL-ONLY:
   * {@link forces} throws, because the tfjs path has no class to feed —
   * training and advection both run in the WGSL kernels (which derive class
   * from the particle index hash).
   */
  classes?: number;
  /**
   * Field architecture — a SELECTABLE model type for comparison (see
   * docs/DESIGN_SPACE_PARTICLE_ART.md):
   *   "standard" (default) — SELU hidden + tanh out, raw [x,y] input.
   *   "siren"    — sin hidden (SIREN); ω0 folded into first-layer weights so
   *                the WGSL activation is plain sin. Smooth higher derivatives.
   *   "fourier"  — SELU hidden but the input is Fourier-encoded γ(p) =
   *                [x, y, sin/cos(ωk·x/y)] over `fourierOctaves` octaves —
   *                beats spectral bias, exposes fine structure.
   * ALL types train FUSED (train_wgsl.ts generates each type's backward —
   * sin pre-act checkpoints, encoding jacobians, grid scatter — verified vs
   * tfjs fixtures on Metal at cos=1.0, tools/train_types_test.ts) and advect
   * via the fused forward kernel. `?train=tfjs` selects this class's autograd
   * path instead, for A/B comparison.
   *
   * Prefer declarative {@link import("./arch").FieldArch} +
   * {@link import("./arch").createFieldFromArch} at gallery sites — this
   * config is the runtime shape those helpers produce.
   */
  modelType?: "standard" | "siren" | "fourier" | "hashgrid";
  /** SIREN first-layer frequency (folded into weights). Default 6. */
  sirenOmega0?: number;
  /** Fourier octaves (encDim = 2 + 4·octaves). Default 4. */
  fourierOctaves?: number;
  /** Hashgrid resolution (gridSize×gridSize learned feature cells). Default 32. */
  gridSize?: number;
  /** Hashgrid features per cell (encDim). Default 4. */
  gridFeatures?: number;
  /**
   * Number of vector heads. `1` = single MLP (`vector` fused layout). `2` =
   * blend. Default 2.
   */
  heads?: 1 | 2;
  /**
   * Hidden activation family. Default: `"sin"` when modelType is `"siren"`,
   * else `"selu"`. Set `"sin"` with `modelType: "fourier"` for Fourier+SIREN.
   */
  hiddenAct?: "selu" | "sin";
}

/**
 * Build a vector head R^2 -> R^2: SELU hidden stack -> tanh dense(2).
 *
 * SELU hidden layers are fine now that training is first-order only — the
 * previous design banned SELU because its gradient uses the `Greater` op,
 * which has no registered SECOND-order gradient in tfjs and killed the old
 * `tf.grad`-based path. The final `tanh` bounds the output to ~O(1) per
 * component so the raw force magnitude matches the old grad/curl scale and
 * existing `forceMagnitude` configs still read correctly.
 */
function makeVectorNet(hiddenUnits: number[], inputDim = 2): tf.Sequential {
  const net = tf.sequential();
  hiddenUnits.forEach((units, i) => {
    const cfg: any = { units, activation: "selu" };
    if (i === 0) cfg.inputShape = [inputDim];
    net.add(tf.layers.dense(cfg));
  });
  net.add(tf.layers.dense({ units: 2, activation: "tanh" }));
  return net;
}

/**
 * SIREN head: LINEAR dense layers (activations applied manually in forces() as
 * sin/…/tanh, since tfjs has no "sin" activation string). The ω0 frequency is
 * folded into the first layer's weights at build time so the WGSL advect
 * activation stays plain `sin`. Uses the SIREN init (Sitzmann et al.): first
 * layer U(-ω0/in, ω0/in), hidden U(-√(6/in), √(6/in)).
 */
function makeSirenNet(
  hiddenUnits: number[],
  inputDim: number,
  omega0: number
): tf.Sequential {
  const net = tf.sequential();
  const widths = [inputDim, ...hiddenUnits];
  hiddenUnits.forEach((units, i) => {
    const fin = widths[i];
    const lim = i === 0 ? omega0 / fin : Math.sqrt(6 / fin);
    const cfg: any = {
      units,
      activation: "linear",
      kernelInitializer: tf.initializers.randomUniform({ minval: -lim, maxval: lim }),
    };
    if (i === 0) cfg.inputShape = [inputDim];
    net.add(tf.layers.dense(cfg));
  });
  net.add(tf.layers.dense({ units: 2, activation: "linear" }));
  return net;
}

/** Fourier feature encoding γ(p) = [x, y, sin(ωk x), cos(ωk x), sin(ωk y),
 *  cos(ωk y)] for k=0..octaves-1, ωk = 2^k · 2π. Fixed (no weights); the same
 *  transform is generated in the advect kernel. encDim = 2 + 4·octaves. */
export function fourierEncode(pn: tf.Tensor2D, octaves: number): tf.Tensor2D {
  return tf.tidy(() => {
    const parts: tf.Tensor2D[] = [pn];
    for (let k = 0; k < octaves; k++) {
      const w = Math.pow(2, k) * 2 * Math.PI;
      const wp = pn.mul(w) as tf.Tensor2D;
      parts.push(tf.sin(wp) as tf.Tensor2D, tf.cos(wp) as tf.Tensor2D);
    }
    return tf.concat(parts, 1) as tf.Tensor2D;
  });
}
export const fourierDim = (octaves: number) => 2 + 4 * octaves;

/**
 * Direct-vector neural force field.
 *
 * Two small MLP heads output vectors directly; `forces()` blends them by
 * `alpha`. Neither head is intrinsically a gradient/curl or order/chaos lane.
 * See the file header for why the exact second-order construction was removed.
 */
export class HelmholtzField implements ForceField {
  /** Neutral A/B output mix in `[0,1]`; mutate freely at runtime. */
  alpha: number;
  /** Multi-species class count (0 = classless). Immutable. */
  readonly classes: number;
  readonly semantic: "blend" | "agree-disagree";
  /** 1 = single-head (α forced 0 in forces); 2 = blend. */
  readonly headCount: 1 | 2;

  /** First direct-vector head A. */
  private readonly g: tf.Sequential;
  /** Second direct-vector head B — null when {@link headCount} is 1. */
  private readonly r: tf.Sequential | null;

  private readonly weights: tf.Variable[];

  /** Selectable architecture — see {@link HelmholtzFieldConfig.modelType}. */
  readonly modelType: "standard" | "siren" | "fourier" | "hashgrid";
  /** Effective hidden activation (may be sin on a fourier encoding). */
  private readonly hiddenAct: "selu" | "sin";
  readonly sirenOmega0: number;
  readonly fourierOctaves: number;
  readonly gridSize: number;
  readonly gridFeatures: number;
  /** Stacked hashgrid feature planes (= max(classes, 1)); 1 when classless. */
  readonly gridPlanes: number = 1;
  /** hashgrid learned feature table [planes·gridSize², features]; null otherwise. */
  readonly grid: tf.Variable | null = null;

  constructor({
    alpha,
    semantic = "blend",
    hiddenUnits = [32, 32],
    classes = 0,
    modelType = "standard",
    sirenOmega0 = 6,
    fourierOctaves = 4,
    gridSize = 32,
    gridFeatures = 4,
    heads = 2,
    hiddenAct,
  }: HelmholtzFieldConfig) {
    this.headCount = heads === 1 ? 1 : 2;
    this.alpha = this.headCount === 1 ? 0 : alpha;
    this.semantic = semantic;
    this.classes = classes;
    this.modelType = modelType;
    this.hiddenAct =
      hiddenAct ?? (modelType === "siren" ? "sin" : "selu");
    this.sirenOmega0 = sirenOmega0;
    this.fourierOctaves = fourierOctaves;
    this.gridSize = gridSize;
    this.gridFeatures = gridFeatures;
    if (modelType === "hashgrid" && this.hiddenAct === "sin") {
      throw new Error(`HelmholtzField: hashgrid + sin not supported`);
    }
    // HASHGRID + classes IS supported, but the family reaches the field through
    // the GRID (one stacked feature plane per family, cell index offset by
    // cls·gridSize²), not through one-hot channels on head B. Both routes are
    // named and validated once, in advect_wgsl's `familyRoute`; this object
    // only has to allocate the matching variables.
    if (modelType === "fourier" && classes > 0) {
      throw new Error(`HelmholtzField: ${modelType} + classes not supported yet`);
    }
    if (this.headCount === 1 && classes > 0) {
      throw new Error("HelmholtzField: single-head + classes not supported");
    }
    if (this.headCount === 1 && semantic === "agree-disagree") {
      throw new Error(
        "HelmholtzField: agree-disagree requires two heads"
      );
    }
    const encIn =
      modelType === "fourier"
        ? fourierDim(fourierOctaves)
        : modelType === "hashgrid"
        ? gridFeatures
        : 2;
    if (modelType === "hashgrid") {
      // planes = max(classes, 1): one feature table per family, stacked so the
      // packed buffer stays a single contiguous grid segment.
      this.gridPlanes = Math.max(classes, 1);
      this.grid = tf.variable(
        tf.randomUniform(
          [this.gridPlanes * gridSize * gridSize, gridFeatures],
          -0.1,
          0.1
        )
      );
    }
    const makeNet = (inDim: number) =>
      this.hiddenAct === "sin"
        ? makeSirenNet(hiddenUnits, inDim, sirenOmega0)
        : makeVectorNet(hiddenUnits, inDim);
    this.g = makeNet(encIn);
    // One-hot class channels are the RAW-encoding route only. On a hashgrid the
    // family already moved the grid lookup, so head B's input width is
    // unchanged — widening it here would allocate weights the fused kernel has
    // no rows for and the packed-segment check would fail loudly at upload.
    const onehotChannels = modelType === "hashgrid" ? 0 : classes;
    this.r =
      this.headCount === 1 ? null : makeNet(encIn + onehotChannels);

    const collect = (net: tf.Sequential): tf.Variable[] =>
      net.trainableWeights.map((w) => (w as any).val as tf.Variable);
    this.weights = [
      ...(this.grid ? [this.grid] : []),
      ...collect(this.g),
      ...(this.r ? collect(this.r) : []),
    ];
  }

  /** Hidden-layer activation the advect kernel should generate for this type. */
  get hiddenActivation(): "selu" | "sin" {
    return this.hiddenAct;
  }

  get trainableWeights(): tf.Variable[] {
    return this.weights;
  }

  /**
   * The vector heads — length 1 or 2. Fused codegen reads this for layout.
   */
  get heads(): tf.Sequential[] {
    return this.r ? [this.g, this.r] : [this.g];
  }

  /**
   * Raw force vectors `[N,2]` at the given normalized positions `[N,2]`.
   *
   *   g       = direct-vector head A
   *   r       = direct-vector head B
   *   forces  = (1 - alpha) * g + alpha * r
   *
   * A plain FORWARD pass (no `tf.grad`): differentiable w.r.t.
   * {@link trainableWeights} in a SINGLE first-order backward. Wrapped in
   * `tf.tidy` to free intermediates; the tape retains what backprop needs.
   */
  forces(posNorm: tf.Tensor2D): tf.Tensor2D {
    if (this.classes > 0) {
      throw new Error(
        "HelmholtzField.forces: class-aware fields are fused-kernel-only " +
          "(the tfjs path has no class input) — do not use ?train=tfjs with " +
          "a classes>0 piece."
      );
    }
    return tf.tidy(() => {
      // Encode the input per model type; SIREN applies sin manually (its tf
      // layers are linear). tfjs autograd differentiates all of these.
      const enc =
        this.modelType === "fourier"
          ? fourierEncode(posNorm, this.fourierOctaves)
          : this.modelType === "hashgrid"
          ? this.gridInterp(posNorm)
          : posNorm;
      const gVec = this.evalHead(this.g, enc);
      if (this.headCount === 1 || !this.r) {
        return gVec as tf.Tensor2D;
      }
      const rVec = this.evalHead(this.r, enc);
      const a = this.alpha;
      return gVec.mul(1 - a).add(rVec.mul(a)) as tf.Tensor2D;
    });
  }

  /** Separately evaluate the two independent vector generators. */
  headForces(posNorm: tf.Tensor2D): [tf.Tensor2D, tf.Tensor2D] {
    if (this.classes > 0) {
      throw new Error("HelmholtzField.headForces: class-aware fields are fused-only");
    }
    const enc =
      this.modelType === "fourier"
        ? fourierEncode(posNorm, this.fourierOctaves)
        : this.modelType === "hashgrid"
        ? this.gridInterp(posNorm)
        : posNorm;
    return [this.evalHead(this.g, enc), this.evalHead(this.r!, enc)];
  }

  /** Bilinear interpolation of the learned feature grid — the SAME row-major
   *  indexing + weights the advect WGSL emitter uses, so trained grid values
   *  line up. Differentiable (tf.gather backward scatters into cells). */
  private gridInterp(posNorm: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      const gs = this.gridSize;
      const grid = this.grid!; // [gs*gs, F]
      const gc = posNorm.clipByValue(0, 1).mul(gs - 1); // [N,2]
      const i0 = gc.floor();
      const f = gc.sub(i0);
      const ix = i0.slice([0, 0], [-1, 1]) as tf.Tensor2D; // [N,1]
      const iy = i0.slice([0, 1], [-1, 1]) as tf.Tensor2D;
      const fx = f.slice([0, 0], [-1, 1]) as tf.Tensor2D;
      const fy = f.slice([0, 1], [-1, 1]) as tf.Tensor2D;
      const ix1 = ix.add(1).minimum(gs - 1);
      const iy1 = iy.add(1).minimum(gs - 1);
      // one-hot × matmul instead of tf.gather: gather's BACKWARD chokes on
      // int32 indices inside the tape ("gradient of input indices must be
      // float32"); onehot(cell) @ grid is differentiable w.r.t. the grid (the
      // one-hot is a constant selector) and gives the identical [N,F] rows.
      const gsq = gs * gs;
      const gather = (jx: tf.Tensor2D, jy: tf.Tensor2D) => {
        const cell = jy.mul(gs).add(jx).reshape([-1]).toInt();
        return tf.oneHot(cell, gsq).matMul(grid) as tf.Tensor2D; // [N,F]
      };
      const w00 = fx.mul(-1).add(1).mul(fy.mul(-1).add(1)); // (1-fx)(1-fy)
      const w10 = fx.mul(fy.mul(-1).add(1));
      const w01 = fx.mul(-1).add(1).mul(fy);
      const w11 = fx.mul(fy);
      return gather(ix as tf.Tensor2D, iy as tf.Tensor2D)
        .mul(w00)
        .add(gather(ix1 as tf.Tensor2D, iy as tf.Tensor2D).mul(w10))
        .add(gather(ix as tf.Tensor2D, iy1 as tf.Tensor2D).mul(w01))
        .add(gather(ix1 as tf.Tensor2D, iy1 as tf.Tensor2D).mul(w11)) as tf.Tensor2D;
    });
  }

  /** Forward one head: selu/tanh via predict(); SIREN applies sin + tanh. */
  private evalHead(net: tf.Sequential, input: tf.Tensor2D): tf.Tensor2D {
    if (this.hiddenAct !== "sin") {
      return net.predict(input) as tf.Tensor2D;
    }
    let h: tf.Tensor = input;
    const layers = net.layers;
    layers.forEach((layer, i) => {
      h = layer.apply(h) as tf.Tensor; // LINEAR dense
      h = i < layers.length - 1 ? tf.sin(h) : tf.tanh(h);
    });
    return h as tf.Tensor2D;
  }

  dispose(): void {
    this.g.dispose();
    this.r?.dispose();
    this.grid?.dispose();
  }
}
