/**
 * advect_wgsl — PURE WGSL codegen for the fused ADVECT compute kernel.
 *
 * One `@compute` dispatch replaces the entire tfjs advect stage (~40 GPU
 * dispatches): thread = particle → evaluate the tiny MLP field from a packed
 * weights buffer → integrate + friction + clip + wrap → fused random reset →
 * write back. Forward-only, no autograd. This is HANDOFF.md §3 Phase 1.
 *
 * ZERO imports on purpose: everything here is pure data → string, so the
 * kernel is testable headless under Bun + bun-webgpu (tools/kernel_test.ts)
 * on a real Metal adapter — no browser needed. The tfjs-coupled half lives
 * in ./advect.ts.
 *
 * ANTI-TRAP (AlphaGOJS "hardcoded D=8" bug): no dimension is baked into
 * handwritten WGSL. Layer dims arrive as data (LayerDims, read off the live
 * tfjs model), are validated loudly, and the shader is GENERATED to match.
 * A mismatch throws at construction — it cannot silently advect wrong.
 *
 * PERF (measured on Apple M-series Metal, 1M particles, helmholtz 2→32→32→2×2):
 *   naive loop codegen, wg=64, storage reads . ~18-20 ms/step
 *   loop codegen, wg=128, workgroup staging .. ~14 ms/step
 *   fully-unrolled scalar registers ......... ~12 ms/step
 *   UNROLLED VEC4 TILES + staging + wg=256 .. ~7-9 ms/step   ← shipped (f32)
 *   + f16 stage/MAC/accum (precision:"f16") . ~4-7 ms/step   ← shipped when
 *                                              the device has "shader-f16"
 * The kernel is weight-load-throughput bound, so the wins are (a) staging the
 * ~10KB of weights in workgroup memory, (b) vec4 weight loads accumulating 4
 * outputs at once, (c) full unrolling so activations live in registers, and
 * (d) f16 halving the staged-weight traffic (see emitHeadUnrolledF16 for the
 * f16/f32 precision split; physics is always f32).
 *
 * Weight layout matches tfjs dense so packed buffers are filled by verbatim
 * GPU→GPU copies of each variable: kernel `[in][out]` row-major (row = input
 * index), then bias `[out]`, per layer, heads back-to-back — each segment's
 * offset padded to a 4-float (16-byte) boundary for the vec4 loads.
 */

// ---------------------------------------------------------------------------
// Canonical types
// ---------------------------------------------------------------------------

// "sin" is the SIREN activation. The SIREN ω0 frequency is folded into the
// trained weights (init scales them), so the WGSL activation is plain sin.
export type Activation = "linear" | "selu" | "tanh" | "sigmoid" | "sin";

/** Shape+activation of one dense layer, as read off the live model. */
export interface LayerDims {
  inSize: number;
  outSize: number;
  activation: Activation;
}

/** LayerDims + its float offsets into the packed weights buffer. */
export interface LayerSpec extends LayerDims {
  /** float offset of the `[inSize][outSize]` kernel matrix (16B-aligned) */
  weightOffset: number;
  /** float offset of the `[outSize]` bias vector (16B-aligned) */
  biasOffset: number;
}

export interface HeadSpec {
  layers: LayerSpec[];
}

/**
 * The field variants the kernel can evaluate (sum type — force blending is
 * type-directed, one generated handler per kind):
 *   helmholtz : F = (1-alpha)·heads[0](p) + alpha·heads[1](p)   (tanh heads g,r)
 *   mlp       : F = heads[0](p) - 0.5                           (legacy sigmoid MLP)
 */
export type FieldSpec =
  | { kind: "helmholtz"; heads: [HeadSpec, HeadSpec] }
  | { kind: "mlp"; heads: [HeadSpec] };

/**
 * Input encoding for the field heads (a SELECTABLE model axis):
 *   raw     — the head reads [x, y] directly (standard / SIREN).
 *   fourier — the head reads γ(p) = [x, y, sin(ωk x), sin(ωk y), cos(ωk x),
 *             cos(ωk y)] for k=0..octaves-1, ωk = 2^k·2π. encDim = 2 + 4·octaves.
 *             MUST match src/core/field/helmholtz.ts fourierEncode (feature
 *             order = the tfjs concat order the trained weights expect).
 */
export type Encoding =
  | { kind: "raw" }
  | { kind: "fourier"; octaves: number }
  | { kind: "hashgrid"; gridSize: number; features: number };

/** encoded input dimension (before the class one-hot). */
export function encodingDim(e: Encoding): number {
  if (e.kind === "fourier") return 2 + 4 * e.octaves;
  if (e.kind === "hashgrid") return e.features;
  return 2;
}

/** Trainable-parameter floats the encoding adds AHEAD of the heads (a learned
 *  feature grid for hashgrid; 0 otherwise). Packed at offset 0 of the weights
 *  buffer so the fill can read it via W(). */
export function encodingParamFloats(e: Encoding): number {
  return e.kind === "hashgrid" ? e.gridSize * e.gridSize * e.features : 0;
}

/** One packed tensor's location — pairs 1:1 with the model's variable list. */
export interface PackedSegment {
  floatOffset: number;
  floatLength: number;
  role: "kernel" | "bias" | "grid";
  head: number;
  layer: number;
}

export interface FieldLayout {
  spec: FieldSpec;
  /**
   * Multi-species class count C. 0 = classless (bit-identical to the original
   * kernels). When C > 0 the CHAOS head (heads[1]) takes 2+C inputs — pos plus
   * a one-hot class — while the order head stays class-blind at 2. Class ids
   * are NEVER stored: c(i) = pcg(i ^ CLASS_SALT) % C, derived identically in
   * the advect kernel, the trainer, and the renderer.
   */
  classes: number;
  /** input encoding (raw / fourier) — the head's layer-0 input width. */
  encoding: Encoding;
  /** total packed weights buffer length in floats (multiple of 4) */
  totalFloats: number;
  /**
   * Segments in tfjs variable order (kernel0, bias0, kernel1, bias1, … per
   * head, heads concatenated) — zip against `trainableWeights` to copy.
   */
  segments: PackedSegment[];
}

export const WORKGROUP_SIZE = 256;
/** dispatchWorkgroups is capped at 65535 per dimension (1D dispatch). */
export const MAX_PARTICLES = 65535 * WORKGROUP_SIZE;
/**
 * Nets at or under this many multiply-accumulates compile fully unrolled
 * (every activation a named register — fastest, ~0.3-1s one-time compile).
 * Bigger nets (mlpDeep is ~33k MACs) fall back to vec4-tiled loops to keep
 * shader compile time sane.
 */
export const UNROLL_MAC_LIMIT = 8192;

// ---------------------------------------------------------------------------
// κ — validation + packing (the ONLY place shapes are checked; the shader is
// generated from the validated layout so it cannot disagree with it)
// ---------------------------------------------------------------------------

const ACTIVATIONS: ReadonlySet<string> = new Set([
  "linear",
  "selu",
  "tanh",
  "sigmoid",
  "sin",
]);

function validateChain(dims: LayerDims[], label: string, wantIn: number): void {
  if (dims.length === 0) {
    throw new Error(`advect: head '${label}' has no layers`);
  }
  if (dims[0].inSize !== wantIn) {
    throw new Error(
      `advect: head '${label}' input must be ${wantIn} (got ${dims[0].inSize})`
    );
  }
  if (dims[dims.length - 1].outSize !== 2) {
    throw new Error(
      `advect: head '${label}' output must be 2 (got ${
        dims[dims.length - 1].outSize
      })`
    );
  }
  dims.forEach((d, i) => {
    if (!ACTIVATIONS.has(d.activation)) {
      throw new Error(
        `advect: head '${label}' layer ${i} activation '${d.activation}' ` +
          `not supported (need one of ${[...ACTIVATIONS].join(", ")})`
      );
    }
    if (!Number.isInteger(d.inSize) || !Number.isInteger(d.outSize) ||
        d.inSize < 1 || d.outSize < 1) {
      throw new Error(
        `advect: head '${label}' layer ${i} has bad dims ${d.inSize}→${d.outSize}`
      );
    }
    if (i > 0 && d.inSize !== dims[i - 1].outSize) {
      throw new Error(
        `advect: head '${label}' layer ${i} inSize ${d.inSize} != ` +
          `previous outSize ${dims[i - 1].outSize}`
      );
    }
  });
}

/**
 * Validate head chains and assign packed-buffer offsets (each segment padded
 * to a 4-float boundary so kernel rows with outSize%4==0 are vec4-loadable).
 * `headsDims` must have exactly 2 heads for "helmholtz" (g then r) and
 * exactly 1 for "mlp".
 */
export function layoutField(
  kind: FieldSpec["kind"],
  headsDims: LayerDims[][],
  opts: { classes?: number; encoding?: Encoding } = {}
): FieldLayout {
  const classes = opts.classes ?? 0;
  const encoding: Encoding = opts.encoding ?? { kind: "raw" };
  const encDim = encodingDim(encoding);
  if (!Number.isInteger(classes) || classes < 0 || classes > 16) {
    throw new Error(`advect: classes ${classes} outside [0, 16]`);
  }
  if (classes > 0 && kind !== "helmholtz") {
    throw new Error(`advect: classes need the helmholtz kind (got '${kind}')`);
  }
  if (encoding.kind !== "raw" && classes > 0) {
    throw new Error(`advect: ${encoding.kind} + classes not supported yet`);
  }
  const wantHeads = kind === "helmholtz" ? 2 : 1;
  if (headsDims.length !== wantHeads) {
    throw new Error(
      `advect: kind '${kind}' needs ${wantHeads} head(s), got ${headsDims.length}`
    );
  }

  const align4 = (x: number) => (x + 3) & ~3;
  let off = 0;
  const segments: PackedSegment[] = [];
  // hashgrid: reserve the learned feature grid at the FRONT of the buffer
  // (offset 0) so the encoding-fill reads it via W(); pairs 1:1 with the grid
  // tf.Variable, which the field lists FIRST in trainableWeights.
  const gridFloats = encodingParamFloats(encoding);
  if (gridFloats > 0) {
    segments.push({
      floatOffset: 0, floatLength: gridFloats, role: "grid", head: -1, layer: -1,
    });
    off = align4(gridFloats);
  }
  const heads: HeadSpec[] = headsDims.map((dims, h) => {
    // head 1 (chaos lane) carries the one-hot class channels; head 0 (order)
    // and the legacy mlp stay class-blind. Both take the ENCODED input width
    // (encDim = 2 for raw, 2+4·octaves for fourier).
    validateChain(dims, `${kind}[${h}]`, h === 1 ? encDim + classes : encDim);
    const layers: LayerSpec[] = dims.map((d, l) => {
      const weightOffset = align4(off);
      off = weightOffset + d.inSize * d.outSize;
      const biasOffset = align4(off);
      off = biasOffset + d.outSize;
      segments.push(
        { floatOffset: weightOffset, floatLength: d.inSize * d.outSize,
          role: "kernel", head: h, layer: l },
        { floatOffset: biasOffset, floatLength: d.outSize,
          role: "bias", head: h, layer: l }
      );
      return { ...d, weightOffset, biasOffset };
    });
    return { layers };
  });

  const spec: FieldSpec =
    kind === "helmholtz"
      ? { kind, heads: heads as [HeadSpec, HeadSpec] }
      : { kind, heads: heads as [HeadSpec] };
  return { spec, classes, encoding, totalFloats: align4(off), segments };
}

/** Salt for the storage-free class hash — must match trainer + renderer. */
export const CLASS_SALT = 2166136261;

/** Total multiply-accumulates for one field evaluation (drives unroll choice). */
export function totalMacs(layout: FieldLayout): number {
  return layout.spec.heads.reduce(
    (n, h) => n + h.layers.reduce((m, L) => m + L.inSize * L.outSize, 0),
    0
  );
}

// ---------------------------------------------------------------------------
// Codegen
// ---------------------------------------------------------------------------

/** thin dispatchers: activation → WGSL expression around `s` (scalar / vec4) */
function actExpr(a: Activation, s: string): string {
  switch (a) {
    case "linear":  return s;
    case "selu":    return `selu(${s})`;
    case "tanh":    return `tanh(${s})`;
    case "sigmoid": return `sigmoid_(${s})`;
    case "sin":     return `sin(${s})`;
  }
}
function actExpr4(a: Activation, s: string): string {
  switch (a) {
    case "linear":  return s;
    case "selu":    return `selu4(${s})`;
    case "tanh":    return `tanh(${s})`;
    case "sigmoid": return `sigmoid4(${s})`;
    case "sin":     return `sin(${s})`;
  }
}

/**
 * UNROLLED head evaluator: every activation is a named register; layers with
 * outSize%4==0 accumulate vec4 tiles (one shared-mem load feeds 4 MACs),
 * others fall back to scalar terms. First layer (inSize==2, validated) reads
 * `p` directly.
 */
function emitHeadUnrolled(h: number, head: HeadSpec): string {
  // layer-0 inputs beyond [x, y] are ONE-HOT class channels (validated by
  // layoutField): only row 2+cls contributes, so instead of materializing the
  // one-hot we ADD that single weight row directly.
  const classChannels = head.layers[0].inSize - 2;
  const lines: string[] = [`fn eval_head_${h}(p : vec2f, cls : u32) -> vec2f {`];
  if (classChannels === 0) lines.push(`  let _cls = cls; // classless head`);
  let prev: string[] = [];
  head.layers.forEach((L, l) => {
    const cur: string[] = [];
    lines.push(`  // layer ${l}: ${L.inSize} -> ${L.outSize} (${L.activation})`);
    if (L.outSize % 4 === 0) {
      const T = L.outSize / 4; // vec4 tiles per row; offsets are 16B-aligned
      for (let t = 0; t < T; t++) {
        const acc = `q${h}_${l}_${t}`;
        lines.push(`  var ${acc} = W4(${L.biasOffset / 4 + t}u);`);
        const row = (i: number) => (L.weightOffset + i * L.outSize) / 4 + t;
        if (l === 0) {
          lines.push(`  ${acc} = fma(vec4f(p.x), W4(${row(0)}u), ${acc});`);
          lines.push(`  ${acc} = fma(vec4f(p.y), W4(${row(1)}u), ${acc});`);
          if (classChannels > 0) {
            // dynamic row 2+cls; OUT%4==0 + 16B-aligned offset ⇒ exact /4
            lines.push(
              `  ${acc} = ${acc} + W4((${L.weightOffset}u + (2u + cls) * ${L.outSize}u) / 4u + ${t}u);`
            );
          }
        } else {
          for (let i = 0; i < L.inSize; i++) {
            lines.push(`  ${acc} = fma(vec4f(${prev[i]}), W4(${row(i)}u), ${acc});`);
          }
        }
        lines.push(`  let a${h}_${l}_${t} = ${actExpr4(L.activation, acc)};`);
        for (const c of ["x", "y", "z", "w"]) cur.push(`a${h}_${l}_${t}.${c}`);
      }
    } else {
      for (let j = 0; j < L.outSize; j++) {
        const terms = [`W(${L.biasOffset + j}u)`];
        if (l === 0) {
          terms.push(`p.x * W(${L.weightOffset + j}u)`);
          terms.push(`p.y * W(${L.weightOffset + L.outSize + j}u)`);
          if (classChannels > 0) {
            terms.push(`W(${L.weightOffset}u + (2u + cls) * ${L.outSize}u + ${j}u)`);
          }
        } else {
          for (let i = 0; i < L.inSize; i++) {
            terms.push(`${prev[i]} * W(${L.weightOffset + i * L.outSize + j}u)`);
          }
        }
        lines.push(`  let s${h}_${l}_${j} = ${actExpr(L.activation, terms.join(" + "))};`);
        cur.push(`s${h}_${l}_${j}`);
      }
    }
    prev = cur;
  });
  lines.push(`  return vec2f(${prev[0]}, ${prev[1]});`);
  lines.push(`}`);
  return lines.join("\n");
}

/**
 * f16 UNROLLED head evaluator — same tiling as emitHeadUnrolled, but the
 * staged weights, MACs, and accumulators are f16 (halves the workgroup-memory
 * traffic the kernel is bound on, 2× ALU rate on Apple GPUs; measured
 * ~9.1-9.6 → ~6.0-7.5 ms/step @ 1M particles on M-series Metal).
 *
 * Precision split (deliberate, matches the measured A/B in the f16 bench):
 *   - MACs + accumulation ... f16 (the win; error ~2^-11 relative per term)
 *   - activations ........... f32 (exp/tanh lose REAL precision in f16 —
 *                              compute via a vec4f cast, truncate back to f16)
 *   - positions / physics ... f32 (untouched — only the field eval narrows)
 * Layer-0 class row add (`acc + W4(row 2+cls)`) is a plain f16 add.
 * Scalar (outSize%4!=0) layers widen each f16 weight to f32, compute in f32,
 * and hand the next layer an f16 activation — the final outSize=2 layer always
 * takes this path, so head outputs are f16-rounded, well inside the kernel
 * test's f16 tolerance.
 */
function emitHeadUnrolledF16(h: number, head: HeadSpec): string {
  const classChannels = head.layers[0].inSize - 2;
  const lines: string[] = [`fn eval_head_${h}(p : vec2f, cls : u32) -> vec2f {`];
  if (classChannels === 0) lines.push(`  let _cls = cls; // classless head`);
  // position narrowed ONCE per head; p in [0,1] so f16 costs ~5e-4 absolute
  lines.push(`  let hx = f16(p.x); let hy = f16(p.y);`);
  let prev: string[] = []; // f16-typed expressions
  head.layers.forEach((L, l) => {
    const cur: string[] = [];
    lines.push(`  // layer ${l}: ${L.inSize} -> ${L.outSize} (${L.activation})`);
    if (L.outSize % 4 === 0) {
      const T = L.outSize / 4; // vec4 tiles per row; offsets are 16B-aligned
      for (let t = 0; t < T; t++) {
        const acc = `q${h}_${l}_${t}`;
        lines.push(`  var ${acc} = W4(${L.biasOffset / 4 + t}u);`);
        const row = (i: number) => (L.weightOffset + i * L.outSize) / 4 + t;
        if (l === 0) {
          lines.push(`  ${acc} = fma(vec4<f16>(hx), W4(${row(0)}u), ${acc});`);
          lines.push(`  ${acc} = fma(vec4<f16>(hy), W4(${row(1)}u), ${acc});`);
          if (classChannels > 0) {
            // dynamic row 2+cls; OUT%4==0 + 16B-aligned offset ⇒ exact /4
            lines.push(
              `  ${acc} = ${acc} + W4((${L.weightOffset}u + (2u + cls) * ${L.outSize}u) / 4u + ${t}u);`
            );
          }
        } else {
          for (let i = 0; i < L.inSize; i++) {
            lines.push(`  ${acc} = fma(vec4<f16>(${prev[i]}), W4(${row(i)}u), ${acc});`);
          }
        }
        // activation in f32, truncated back to f16 for the next layer's MACs
        lines.push(
          `  let a${h}_${l}_${t} = vec4<f16>(${actExpr4(L.activation, `vec4f(${acc})`)});`
        );
        for (const c of ["x", "y", "z", "w"]) cur.push(`a${h}_${l}_${t}.${c}`);
      }
    } else {
      for (let j = 0; j < L.outSize; j++) {
        const terms = [`f32(W(${L.biasOffset + j}u))`];
        if (l === 0) {
          terms.push(`p.x * f32(W(${L.weightOffset + j}u))`);
          terms.push(`p.y * f32(W(${L.weightOffset + L.outSize + j}u))`);
          if (classChannels > 0) {
            terms.push(`f32(W(${L.weightOffset}u + (2u + cls) * ${L.outSize}u + ${j}u))`);
          }
        } else {
          for (let i = 0; i < L.inSize; i++) {
            terms.push(`f32(${prev[i]}) * f32(W(${L.weightOffset + i * L.outSize + j}u))`);
          }
        }
        lines.push(`  let s${h}_${l}_${j} = ${actExpr(L.activation, terms.join(" + "))};`);
        cur.push(`f16(s${h}_${l}_${j})`);
      }
    }
    prev = cur;
  });
  lines.push(`  return vec2f(f32(${prev[0]}), f32(${prev[1]}));`);
  lines.push(`}`);
  return lines.join("\n");
}

/**
 * LOOPED head evaluator (big nets): ping-pong activation arrays, vec4-tiled
 * inner loops where outSize%4==0. Const loop bounds; ~4× fewer weight loads
 * than a scalar loop.
 */
function emitHeadLooped(
  h: number,
  head: HeadSpec,
  maxW: number,
  encoding: Encoding = { kind: "raw" }
): string {
  const encDim = encodingDim(encoding);
  const classChannels = head.layers[0].inSize - encDim;
  const bufs = [`h0_${h}`, `h1_${h}`];
  const lines: string[] = [`fn eval_head_${h}(p : vec2f, cls : u32) -> vec2f {`];
  if (classChannels === 0) lines.push(`  let _cls = cls; // classless head`);
  lines.push(`  var ${bufs[0]} : array<f32, ${maxW}>;`);
  lines.push(`  var ${bufs[1]} : array<f32, ${maxW}>;`);
  // encoded input γ(p): raw = [x,y]; fourier prepends the raw coords then, per
  // octave, [sin(ωk·x), sin(ωk·y), cos(ωk·x), cos(ωk·y)] — the SAME feature
  // order as helmholtz.ts fourierEncode (what the trained weights expect).
  lines.push(`  ${bufs[0]}[0] = p.x;`);
  lines.push(`  ${bufs[0]}[1] = p.y;`);
  if (encoding.kind === "fourier") {
    for (let k = 0; k < encoding.octaves; k++) {
      const w = (Math.pow(2, k) * 2 * Math.PI).toFixed(8);
      const o = 2 + 4 * k;
      lines.push(`  ${bufs[0]}[${o}] = sin(${w} * p.x);`);
      lines.push(`  ${bufs[0]}[${o + 1}] = sin(${w} * p.y);`);
      lines.push(`  ${bufs[0]}[${o + 2}] = cos(${w} * p.x);`);
      lines.push(`  ${bufs[0]}[${o + 3}] = cos(${w} * p.y);`);
    }
  } else if (encoding.kind === "hashgrid") {
    // Bilinear interp of the learned feature grid (row-major, offset 0 in W —
    // same indexing as helmholtz.ts). encDim = features; fills h0[0..F-1],
    // OVERWRITING the raw [x,y] above (the grid IS the encoded input).
    const { gridSize: gs, features: F } = encoding;
    lines.push(`  {`);
    lines.push(`    let gxf = clamp(p.x, 0.0, 1.0) * ${(gs - 1).toFixed(1)};`);
    lines.push(`    let gyf = clamp(p.y, 0.0, 1.0) * ${(gs - 1).toFixed(1)};`);
    lines.push(`    let ix = u32(floor(gxf)); let iy = u32(floor(gyf));`);
    lines.push(`    let fx = gxf - floor(gxf); let fy = gyf - floor(gyf);`);
    lines.push(`    let ix1 = min(ix + 1u, ${gs - 1}u); let iy1 = min(iy + 1u, ${gs - 1}u);`);
    lines.push(`    let b00 = (iy * ${gs}u + ix) * ${F}u;`);
    lines.push(`    let b10 = (iy * ${gs}u + ix1) * ${F}u;`);
    lines.push(`    let b01 = (iy1 * ${gs}u + ix) * ${F}u;`);
    lines.push(`    let b11 = (iy1 * ${gs}u + ix1) * ${F}u;`);
    lines.push(`    let w00 = (1.0 - fx) * (1.0 - fy); let w10 = fx * (1.0 - fy);`);
    lines.push(`    let w01 = (1.0 - fx) * fy; let w11 = fx * fy;`);
    lines.push(`    for (var fi = 0u; fi < ${F}u; fi = fi + 1u) {`);
    lines.push(`      ${bufs[0]}[fi] = w00 * W(b00 + fi) + w10 * W(b10 + fi) + w01 * W(b01 + fi) + w11 * W(b11 + fi);`);
    lines.push(`    }`);
    lines.push(`  }`);
  }
  if (classChannels > 0) {
    lines.push(`  for (var k = 0u; k < ${classChannels}u; k = k + 1u) {`);
    lines.push(`    ${bufs[0]}[${encDim}u + k] = select(0.0, 1.0, k == cls);`);
    lines.push(`  }`);
  }
  head.layers.forEach((L, l) => {
    const src = bufs[l % 2];
    const dst = bufs[(l + 1) % 2];
    lines.push(`  // layer ${l}: ${L.inSize} -> ${L.outSize} (${L.activation})`);
    if (L.outSize % 4 === 0) {
      const T = L.outSize / 4;
      lines.push(`  for (var t = 0u; t < ${T}u; t = t + 1u) {`);
      lines.push(`    var q = W4(${L.biasOffset / 4}u + t);`);
      lines.push(`    for (var i = 0u; i < ${L.inSize}u; i = i + 1u) {`);
      lines.push(`      q = fma(vec4f(${src}[i]), W4(${L.weightOffset / 4}u + i * ${T}u + t), q);`);
      lines.push(`    }`);
      lines.push(`    let a = ${actExpr4(L.activation, "q")};`);
      lines.push(`    ${dst}[4u * t] = a.x;`);
      lines.push(`    ${dst}[4u * t + 1u] = a.y;`);
      lines.push(`    ${dst}[4u * t + 2u] = a.z;`);
      lines.push(`    ${dst}[4u * t + 3u] = a.w;`);
      lines.push(`  }`);
    } else {
      lines.push(`  for (var j = 0u; j < ${L.outSize}u; j = j + 1u) {`);
      lines.push(`    var s = W(${L.biasOffset}u + j);`);
      lines.push(`    for (var i = 0u; i < ${L.inSize}u; i = i + 1u) {`);
      lines.push(`      s = s + ${src}[i] * W(${L.weightOffset}u + i * ${L.outSize}u + j);`);
      lines.push(`    }`);
      lines.push(`    ${dst}[j] = ${actExpr(L.activation, "s")};`);
      lines.push(`  }`);
    }
  });
  const last = bufs[head.layers.length % 2];
  lines.push(`  return vec2f(${last}[0], ${last}[1]);`);
  lines.push(`}`);
  return lines.join("\n");
}

/** thin dispatcher: field kind → generated forceAt handler (one path each) */
function emitForce(spec: FieldSpec): string {
  switch (spec.kind) {
    case "helmholtz":
      return (
        `fn forceAt(pn : vec2f, cls : u32) -> vec2f {\n` +
        `  return (1.0 - u.alpha) * eval_head_0(pn, cls) + u.alpha * eval_head_1(pn, cls);\n` +
        `}`
      );
    case "mlp":
      // Legacy sigmoid MLP pieces: [0,1] output re-centered by -0.5.
      return (
        `fn forceAt(pn : vec2f, cls : u32) -> vec2f {\n` +
        `  return eval_head_0(pn, cls) - vec2f(0.5, 0.5);\n` +
        `}`
      );
  }
}

export interface AdvectShaderOpts {
  /**
   * Stage the packed weights into workgroup shared memory before evaluating
   * (measured ~2× vs storage reads — every thread reads every weight).
   * Caller must ensure totalFloats*4 <= device.limits.
   * maxComputeWorkgroupStorageSize; when it doesn't fit, pass false to read
   * straight from the storage buffer.
   */
  stageWeights: boolean;
  /** Threads per workgroup; default WORKGROUP_SIZE. Dispatch must match. */
  workgroupSize?: number;
  /** Force unrolled/looped head codegen; default auto by UNROLL_MAC_LIMIT. */
  unroll?: boolean;
  /**
   * Field-eval precision. "f32" (default) is BYTE-IDENTICAL to the original
   * codegen (regression guard in tools/kernel_test.ts). "f16" stages weights
   * as vec4<f16> and runs the MLP MACs/accumulators in f16 (activations f32,
   * physics f32) — the measured fast path (~9.3 → ~6.5 ms/step @ 1M, Apple
   * Metal). Requires the DEVICE to have the "shader-f16" feature, and only
   * the unrolled+staged codegen has an f16 variant — requesting "f16" with a
   * looped net or stageWeights:false throws (no silent fallback).
   */
  precision?: "f32" | "f16";
}

/**
 * Generate the fused advect WGSL module. Bindings:
 *   @binding(0) uniform Uni        (see Uni struct / ./advect.ts uniData)
 *   @binding(1) storage  weights   array<vec4f>, packed per FieldLayout
 *   @binding(2) storage  pos       array<vec2f>, read_write  [count]
 *   @binding(3) storage  vel       array<vec2f>, read_write  [count]
 *
 * Physics matches main.ts physicsForward + randomReset EXACTLY:
 *   pn  = p / resolution
 *   f   = forceAt(pn) * forceMag
 *   v'  = clamp((v + f) * friction, -maxVel, +maxVel)
 *   p'  = floored-mod(p + v', resolution)      // tf.mod is floored, WGSL % is not
 *   reset (after integration): rand < resetRate → p' = rand*res, v' = 0
 */
export function advectShader(
  layout: FieldLayout,
  opts: AdvectShaderOpts
): string {
  const { spec, totalFloats } = layout;
  const CLASS_SALT_LITERAL = CLASS_SALT;
  const WG = opts.workgroupSize ?? WORKGROUP_SIZE;
  // Fourier encoding is only wired into the LOOPED emitter (it fills the
  // encoded input array); force looped when encoding != raw.
  const forceLooped = layout.encoding.kind !== "raw";
  const unroll = forceLooped
    ? false
    : opts.unroll ?? totalMacs(layout) <= UNROLL_MAC_LIMIT;
  const precision = opts.precision ?? "f32";
  const total4 = totalFloats / 4; // layoutField pads to a multiple of 4
  const maxW = Math.max(
    2,
    ...spec.heads.flatMap((h) => h.layers.map((L) => Math.max(L.outSize, L.inSize)))
  );

  // f16 is a fast path for the SHIPPED configuration (small net → unrolled +
  // staged); there is deliberately no f16 looped/unstaged emitter. Requesting
  // an impossible combination throws — no silent f32 downgrade (Rule T4).
  if (precision === "f16" && !unroll) {
    throw new Error(
      `advect: precision 'f16' needs the unrolled emitter, but this net ` +
        `(${totalMacs(layout)} MACs > ${UNROLL_MAC_LIMIT}) uses the looped ` +
        `fallback — pass precision 'f32' for big nets`
    );
  }
  if (precision === "f16" && !opts.stageWeights) {
    throw new Error(
      `advect: precision 'f16' requires stageWeights (the f16 win IS the ` +
        `halved workgroup-memory traffic) — pass precision 'f32' when weights ` +
        `don't fit workgroup memory`
    );
  }

  const weightAccess =
    precision === "f16"
      ? // staged only (enforced above): weights live as vec4<f16>
        `var<workgroup> wsW : array<vec4<f16>, ${total4}>;\n` +
        `fn W4(i : u32) -> vec4<f16> { return wsW[i]; }\n` +
        `fn W(i : u32) -> f16 { return wsW[i >> 2u][i & 3u]; }`
      : opts.stageWeights
      ? `var<workgroup> wsW : array<vec4f, ${total4}>;\n` +
        `fn W4(i : u32) -> vec4f { return wsW[i]; }\n` +
        `fn W(i : u32) -> f32 { return wsW[i >> 2u][i & 3u]; }`
      : `fn W4(i : u32) -> vec4f { return weights[i]; }\n` +
        `fn W(i : u32) -> f32 { return weights[i >> 2u][i & 3u]; }`;

  // Staging must happen in UNIFORM control flow (workgroupBarrier), i.e.
  // before the `gid >= count` early-out — dead threads still help copy.
  // f16: the storage buffer stays f32 (packed by tfjs copies); each thread
  // narrows on the fly while staging.
  const stagePrelude = opts.stageWeights
    ? `  for (var i = li; i < ${total4}u; i = i + ${WG}u) { wsW[i] = ${
        precision === "f16" ? "vec4<f16>(weights[i])" : "weights[i]"
      }; }\n` +
      `  workgroupBarrier();`
    : ``;

  const emitUnrolled = precision === "f16" ? emitHeadUnrolledF16 : emitHeadUnrolled;
  const heads = spec.heads
    .map((h, i) =>
      unroll ? emitUnrolled(i, h) : emitHeadLooped(i, h, maxW, layout.encoding)
    )
    .join("\n\n");

  // `enable` directives must precede all declarations; "" for f32 keeps the
  // f32 output byte-identical to the pre-f16 codegen.
  const enableDirective = precision === "f16" ? `enable f16;\n` : ``;

  return /* wgsl */ `
${enableDirective}struct Uni {
  resolution : vec2f,
  forceMag   : f32,
  friction   : f32,
  maxVel     : f32,
  alpha      : f32,
  resetRate  : f32,
  seed       : u32,
  count      : u32,
};

@group(0) @binding(0) var<uniform> u : Uni;
@group(0) @binding(1) var<storage, read> weights : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> pos : array<vec2f>;
@group(0) @binding(3) var<storage, read_write> vel : array<vec2f>;

${weightAccess}

// tfjs 'selu' constants (1.7580993… = scale*alpha, fused)
fn selu(x : f32) -> f32 {
  return select(1.7580993408473768 * (exp(x) - 1.0), 1.0507009873554805 * x, x > 0.0);
}
fn selu4(x : vec4f) -> vec4f {
  return select(1.7580993408473768 * (exp(x) - vec4f(1.0)), 1.0507009873554805 * x, x > vec4f(0.0));
}
fn sigmoid_(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn sigmoid4(x : vec4f) -> vec4f { return vec4f(1.0) / (vec4f(1.0) + exp(-x)); }

// PCG hash (Olano) — stateless per-(particle,frame) randomness for the reset.
fn pcg(v : u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let t = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (t >> 22u) ^ t;
}
fn rand01(x : u32) -> f32 { return f32(x) * 2.3283064365386963e-10; }

${heads}

${emitForce(spec)}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gidv : vec3u,
        @builtin(local_invocation_index) li : u32) {
${stagePrelude}
  let gid = gidv.x;
  if (gid >= u.count) { return; }

  let res = u.resolution;
  var p = pos[gid];
  var v = vel[gid];

  // class is IDENTITY — a pure hash of the particle index, stable across
  // frames and resets; same derivation in trainer + renderer (CLASS_SALT).
  let cls = ${layout.classes > 0 ? `pcg(gid ^ ${CLASS_SALT_LITERAL}u) % ${layout.classes}u` : `0u`};
  let f = forceAt(p / res, cls) * u.forceMag;
  v = clamp((v + f) * u.friction, vec2f(-u.maxVel), vec2f(u.maxVel));
  p = p + v;
  p = p - floor(p / res) * res; // floored mod == tf.mod (always in [0,res))

  // Fused random reset (matches main.ts randomReset: after integration,
  // resetRate fraction respawns uniformly with zero velocity).
  let r = pcg(gid ^ (u.seed * 2654435769u));
  if (rand01(r) < u.resetRate) {
    let rx = pcg(r);
    let ry = pcg(rx);
    p = vec2f(rand01(rx) * res.x, rand01(ry) * res.y);
    v = vec2f(0.0, 0.0);
  }

  pos[gid] = p;
  vel[gid] = v;
}
`;
}
